//! Vision-assisted take review for the local filmmaking harness (epic 22708, sc-22714).
//!
//! Three documents, all JSON (JSONC comments tolerated on read), and all deliberately **separate
//! from the production plan**:
//!
//! * a [`ReviewPlan`] — the questions to put to a take, per shot. It is authored beside the plan
//!   rather than inside it so the intended state (`shots[].startState` / `endState`, owned by
//!   [`crate::film_plan`]) and the *interrogation* of a take stay different documents with
//!   different versions;
//! * an [`ObservedState`] — what a reviewer actually read off one take: the frames it sampled,
//!   every per-question observation with its confidence and its explicit `unobserved` value, and
//!   the mismatch flags those produced. One per reviewed take, written to its own file. It
//!   references the intended state **by pointer** ([`IntendedRef`]) and never copies it, so the
//!   two cannot drift and an observation can never be mistaken for an intent;
//! * an [`EvalSet`] / [`EvalResults`] pair — a fixed labeled set of correct and deliberately
//!   broken takes, and the detections / misses / false alarms a reviewer scored against it.
//!
//! **Review is assistive, not quality assurance** ([`ASSISTIVE_NOTICE`]). Every document written
//! here carries that sentence, and every CLI surface prints it. Two rules in this module exist to
//! keep that honest:
//!
//! 1. an observation that was not made is [`Verdict::Unobserved`] and carries **no value** — in
//!    particular a parcel handoff that was never seen is never recorded as completed
//!    ([`Observation::observed`] is `None` whenever [`Observation::unobserved`] is set); and
//! 2. nothing in this module is an input to generation. Observed state is not conditioning, is not
//!    intended state, and only a human's selection (the `film_plan` decision log) changes what a
//!    later shot is built on.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::film_plan::{is_safe_plan_id, PlanDiagnostic, ProductionPlan};
use crate::jsonc::strip_jsonc_comments;

/// Schema version of [`ReviewPlan`] documents this module reads and writes.
pub const REVIEW_PLAN_SCHEMA_VERSION: u32 = 1;
/// Schema version of [`ObservedState`] documents this module writes.
pub const OBSERVED_STATE_SCHEMA_VERSION: u32 = 1;
/// Schema version of [`EvalSet`] / [`EvalResults`] documents.
pub const REVIEW_EVAL_SCHEMA_VERSION: u32 = 1;

/// The sentence every review surface carries, verbatim. A reviewer reads pixels through a local
/// vision model and is wrong in both directions; it flags things for a person to look at and
/// decides nothing.
pub const ASSISTIVE_NOTICE: &str = "Review is ASSISTIVE, not quality assurance: a local vision \
                                    model both misses real faults and flags correct takes. \
                                    Nothing here approves, rejects or conditions anything — only a \
                                    human decision recorded through the controller does.";

/// The continuity dimensions a review question may address. Fixed set: a question outside it is a
/// document error, so the evaluation always reports per known topic.
/// * `character_identity` — is the intended character in the shot at all;
/// * `costume` — is that character wearing what the plan says they wear;
/// * `location` — is this the intended place;
/// * `parcel_identity` — is the intended parcel/prop present and as described;
/// * `parcel_custody` — who has the parcel: the ownership/handoff half of parcel continuity;
/// * `action_completion` — did the action the shot exists to perform actually finish;
/// * `cut_continuity` — does this take sit against the adjacent selected take without a jump.
pub const REVIEW_TOPICS: &[&str] = &[
    "character_identity",
    "costume",
    "location",
    "parcel_identity",
    "parcel_custody",
    "action_completion",
    "cut_continuity",
];

/// Mismatch severities, in descending order of how much a person should care.
pub const MISMATCH_SEVERITIES: &[&str] = &["mismatch", "unobserved", "uncertain"];

/// Below this confidence a contradiction is reported as `uncertain` rather than `mismatch`.
pub const DEFAULT_UNCERTAIN_BELOW: f64 = 0.5;

/// Phrases in an answer that mean the reviewer could not see the thing. Checked before any
/// expect/contradict token, so "I cannot tell whether the parcel is red" is `unobserved`, never a
/// colour observation.
const UNOBSERVED_MARKERS: &[&str] = &[
    "cannot tell",
    "can't tell",
    "cannot determine",
    "can not tell",
    "unable to tell",
    "unable to determine",
    "not visible",
    "isn't visible",
    "is not visible",
    "not shown",
    "not clear",
    "unclear",
    "too blurry",
    "obscured",
    "out of frame",
    "off screen",
    "off-screen",
    "no way to tell",
    "impossible to tell",
    "i don't know",
    "i do not know",
    "unknown",
];

/// Hedges that keep an answer usable but drop its confidence. A hedged contradiction becomes an
/// `uncertain` flag rather than a `mismatch`, which is the whole point: the reviewer's doubt has
/// to survive into the record instead of being rounded to a fact.
const HEDGE_MARKERS: &[&str] = &[
    "appears",
    "appear to",
    "seems",
    "seem to",
    "possibly",
    "probably",
    "might",
    "may be",
    "maybe",
    "likely",
    "hard to",
    "difficult to",
    "i think",
    "looks like it could",
    "somewhat",
    "partially",
];

/// Confidence a decisive, unhedged answer carries.
const CONFIDENT: f64 = 0.9;
/// Confidence a hedged answer carries.
const HEDGED: f64 = 0.45;

// ---------------------------------------------------------------------------------------------
// Review plan document
// ---------------------------------------------------------------------------------------------

/// Which of a take's sampled frames a question is graded on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FrameScope {
    /// The earliest sampled frame — what the take starts on.
    First,
    /// The latest sampled frame — what the take ends on. The scope an action-completion question
    /// wants: the action has to have finished by the end, not at some point in the middle.
    Last,
    /// Every sampled frame must agree. A single contradiction is a mismatch.
    #[default]
    All,
    /// Any one sampled frame agreeing is enough (the thing only has to be visible once).
    Any,
}

impl FrameScope {
    /// The frames, out of `sampled` (ordered by timestamp), this scope actually asks about.
    pub fn select<'a, T>(&self, sampled: &'a [T]) -> Vec<&'a T> {
        match self {
            Self::First => sampled.first().into_iter().collect(),
            Self::Last => sampled.last().into_iter().collect(),
            Self::All | Self::Any => sampled.iter().collect(),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Last => "last",
            Self::All => "all",
            Self::Any => "any",
        }
    }
}

/// One question put to the vision backend about one take.
///
/// `expect` / `contradict` are the graded half: a question with neither cannot produce a verdict
/// and is refused by [`validate_review_plan`]. They are matched case-insensitively as substrings
/// of the backend's answer, `contradict` first — an answer that says both "no courier" and
/// "courier" is a contradiction, because the negative phrasing is the specific one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReviewQuestion {
    /// Stable id, unique within the shot. It is the key the evaluation scores on, so it must not
    /// change meaning between versions of the document.
    pub id: String,
    /// One of [`REVIEW_TOPICS`].
    pub topic: String,
    /// The intended-state claim this question checks, in the plan's own words. Shown verbatim on
    /// the mismatch flag as the `intended` side. It is a restatement for the human reading the
    /// flag — the authoritative intended state stays in the run record, referenced by pointer.
    pub intended: String,
    /// The question put to the vision model, verbatim. Kept in the document rather than generated
    /// so a review is reproducible from the document alone.
    pub ask: String,
    /// Answer substrings that mean the take matches the intent.
    #[serde(default)]
    pub expect: Vec<String>,
    /// Answer substrings that mean the take contradicts the intent.
    #[serde(default)]
    pub contradict: Vec<String>,
    /// Which sampled frames to grade on.
    #[serde(default)]
    pub frames: FrameScope,
    /// When set, an `unobserved` outcome is itself an actionable flag rather than silence. Set it
    /// on anything that has to be SEEN to be believed — a handoff, a parcel changing hands, an
    /// action the next shot depends on having completed.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub must_observe: bool,
    /// Ask this question of the adjacent selected take's last frame as well as this take's first
    /// frame — the across-the-cut question. Requires the shot to declare a `dependsOn` edge.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub across_cut: bool,
}

/// Every question for one shot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ShotReviewSpec {
    pub questions: Vec<ReviewQuestion>,
}

/// Where in a take to sample frames, as fractions of the take's own duration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReviewSampling {
    /// Strictly increasing fractions in `[0, 1)`. The frame extracted at each is the evidence the
    /// observations cite.
    pub positions: Vec<f64>,
}

/// Finite bounds every review declares BEFORE it dispatches anything (epic requirement E5). A
/// review that reaches one stops and says so in [`ObservedState::stop`]; it never retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReviewLimits {
    /// Wall-clock for one shot's whole review, frame extraction included.
    pub max_seconds: u64,
    /// Frames extracted per take, whatever `sampling.positions` asks for.
    pub max_frames_per_shot: u32,
    /// Questions put to the backend per take.
    pub max_questions_per_shot: u32,
    /// Wall-clock for one backend answer.
    pub max_answer_seconds: u64,
}

/// The review document: what to ask about each shot, how to sample it, and what bounds the run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReviewPlan {
    pub schema_version: u32,
    pub id: String,
    pub version: u32,
    #[serde(default)]
    pub description: String,
    pub sampling: ReviewSampling,
    pub limits: ReviewLimits,
    /// Shot id -> its questions. A shot the document does not name is not reviewable; the harness
    /// says so rather than inventing questions for it.
    pub shots: BTreeMap<String, ShotReviewSpec>,
    /// Contradictions below this confidence are reported `uncertain` instead of `mismatch`.
    #[serde(default = "default_uncertain_below")]
    pub uncertain_below: f64,
}

fn default_uncertain_below() -> f64 {
    DEFAULT_UNCERTAIN_BELOW
}

/// Read and parse a review plan (JSONC tolerated).
pub fn read_review_plan_file(path: &std::path::Path) -> Result<ReviewPlan, PlanDiagnostic> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        PlanDiagnostic::plan(
            "reviewPlan",
            format!("cannot read {}: {error}", path.display()),
        )
    })?;
    parse_review_plan(&text)
        .map_err(|error| PlanDiagnostic::plan("reviewPlan", format!("{}: {error}", path.display())))
}

/// Parse a review plan from JSON/JSONC text.
pub fn parse_review_plan(text: &str) -> Result<ReviewPlan, String> {
    serde_json::from_str(&strip_jsonc_comments(text)).map_err(|error| error.to_string())
}

/// Findings on a review plan, on its own and against the production plan it reviews. Every finding
/// names the shot and the field, so a refused document is actionable without reading code.
pub fn validate_review_plan(review: &ReviewPlan, plan: &ProductionPlan) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if review.schema_version != REVIEW_PLAN_SCHEMA_VERSION {
        findings.push(PlanDiagnostic::plan(
            "schemaVersion",
            format!(
                "review plan schema version {} is not supported (this build reads \
                 {REVIEW_PLAN_SCHEMA_VERSION})",
                review.schema_version
            ),
        ));
    }
    if !is_safe_plan_id(&review.id) {
        findings.push(PlanDiagnostic::plan(
            "id",
            format!("{:?} is not a safe id ([A-Za-z0-9_-]{{1,64}})", review.id),
        ));
    }
    if review.version == 0 {
        findings.push(PlanDiagnostic::plan("version", "version must be >= 1"));
    }
    if !(0.0..1.0).contains(&review.uncertain_below) {
        findings.push(PlanDiagnostic::plan(
            "uncertainBelow",
            "uncertainBelow must be in [0, 1)",
        ));
    }
    if review.sampling.positions.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "sampling.positions",
            "at least one sample position is required",
        ));
    }
    let mut previous = -1.0_f64;
    for (index, position) in review.sampling.positions.iter().enumerate() {
        if !position.is_finite() || !(0.0..1.0).contains(position) {
            findings.push(PlanDiagnostic::plan(
                format!("sampling.positions[{index}]"),
                format!("{position} is not a fraction in [0, 1)"),
            ));
        } else if *position <= previous {
            findings.push(PlanDiagnostic::plan(
                format!("sampling.positions[{index}]"),
                "sample positions must strictly increase",
            ));
        }
        previous = *position;
    }
    for (field, value) in [
        ("limits.maxSeconds", review.limits.max_seconds),
        (
            "limits.maxFramesPerShot",
            u64::from(review.limits.max_frames_per_shot),
        ),
        (
            "limits.maxQuestionsPerShot",
            u64::from(review.limits.max_questions_per_shot),
        ),
        ("limits.maxAnswerSeconds", review.limits.max_answer_seconds),
    ] {
        if value == 0 {
            findings.push(PlanDiagnostic::plan(field, format!("{field} must be >= 1")));
        }
    }
    if review.shots.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "shots",
            "a review plan with no shots reviews nothing",
        ));
    }
    let topics: BTreeSet<&str> = REVIEW_TOPICS.iter().copied().collect();
    for (shot_id, spec) in &review.shots {
        let Some(shot) = plan.shots.iter().find(|shot| &shot.id == shot_id) else {
            findings.push(PlanDiagnostic::plan(
                format!("shots.{shot_id}"),
                format!("{shot_id:?} is not a shot in plan {:?}", plan.id),
            ));
            continue;
        };
        if spec.questions.is_empty() {
            findings.push(PlanDiagnostic::shot(
                shot_id,
                "questions",
                "a reviewed shot needs at least one question",
            ));
        }
        if spec.questions.len() > review.limits.max_questions_per_shot as usize {
            findings.push(PlanDiagnostic::shot(
                shot_id,
                "questions",
                format!(
                    "{} questions exceeds limits.maxQuestionsPerShot ({})",
                    spec.questions.len(),
                    review.limits.max_questions_per_shot
                ),
            ));
        }
        let mut seen = BTreeSet::new();
        for question in &spec.questions {
            if !is_safe_plan_id(&question.id) {
                findings.push(PlanDiagnostic::shot(
                    shot_id,
                    "questions[].id",
                    format!("{:?} is not a safe id", question.id),
                ));
            } else if !seen.insert(question.id.as_str()) {
                findings.push(PlanDiagnostic::shot(
                    shot_id,
                    "questions[].id",
                    format!("duplicate question id {:?}", question.id),
                ));
            }
            if !topics.contains(question.topic.as_str()) {
                findings.push(PlanDiagnostic::shot(
                    shot_id,
                    format!("questions.{}.topic", question.id),
                    format!(
                        "{:?} is not a review topic (expected one of {})",
                        question.topic,
                        REVIEW_TOPICS.join(", ")
                    ),
                ));
            }
            if question.ask.trim().is_empty() {
                findings.push(PlanDiagnostic::shot(
                    shot_id,
                    format!("questions.{}.ask", question.id),
                    "ask must not be empty",
                ));
            }
            if question.intended.trim().is_empty() {
                findings.push(PlanDiagnostic::shot(
                    shot_id,
                    format!("questions.{}.intended", question.id),
                    "intended must restate what the plan intends, for the flag to be actionable",
                ));
            }
            if question.expect.is_empty() && question.contradict.is_empty() {
                findings.push(PlanDiagnostic::shot(
                    shot_id,
                    format!("questions.{}.expect", question.id),
                    "a question needs expect and/or contradict tokens to be gradable",
                ));
            }
            for (field, tokens) in [
                ("expect", &question.expect),
                ("contradict", &question.contradict),
            ] {
                if tokens.iter().any(|token| token.trim().is_empty()) {
                    findings.push(PlanDiagnostic::shot(
                        shot_id,
                        format!("questions.{}.{field}", question.id),
                        "tokens must not be empty",
                    ));
                }
            }
            if question.across_cut && shot.depends_on.is_empty() {
                findings.push(PlanDiagnostic::shot(
                    shot_id,
                    format!("questions.{}.acrossCut", question.id),
                    "acrossCut needs the shot to declare a dependsOn edge to compare against",
                ));
            }
        }
    }
    findings
}

// ---------------------------------------------------------------------------------------------
// Observed state
// ---------------------------------------------------------------------------------------------

/// Where the intended state this take was checked against actually lives.
///
/// A **pointer, not a copy**: the plan's own hash plus the path into the run record. Copying the
/// intended state in here would create a second version of it that could drift from the plan, and
/// an observed-state document is exactly the wrong place for an authoritative intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntendedRef {
    pub run_id: String,
    pub shot_id: String,
    pub plan_id: String,
    pub plan_version: u32,
    /// The plan document hash the run recorded, so a reader can prove which intent was in force.
    pub plan_sha256: String,
    /// JSON pointer into the run record, e.g. `/shots/1/intended`.
    pub record_pointer: String,
    /// The run record this pointer is into, relative to the review document.
    pub record_path: String,
}

/// A source document referenced by hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewSourceRef {
    pub id: String,
    pub version: u32,
    pub path: String,
    pub sha256: String,
}

/// Which vision backend answered, and whether the answers came from a model at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewBackendRecord {
    /// `image_vqa` for the API seam, `scripted` for a fake. A `scripted` record is evidence of a
    /// test or an evaluation rehearsal, never of a real review — hence the explicit field.
    pub kind: String,
    /// Catalog id of the model the backend drove.
    pub model: String,
    /// The route the answers came through.
    pub route: String,
    /// What the worker reported: `false` (or absent) means no weights ran.
    pub real_model_inference: bool,
}

/// One frame of evidence, with the timestamp it was taken at and the asset it became.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameEvidence {
    /// Stable within one observed-state document, e.g. `SH020-a1-f2`.
    pub id: String,
    pub shot_id: String,
    pub attempt: u32,
    /// Seconds into the take the frame was sampled at.
    pub timestamp_seconds: f64,
    /// Project asset id of the extracted frame — the thing a person can open.
    pub asset_id: String,
    /// Project-relative path of the frame file.
    pub path: String,
    /// `frame_extract`, `imported`, or `labeled_set`.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    pub captured_at: String,
}

/// What a reviewer concluded about one question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The take agrees with the intent.
    Match,
    /// The take contradicts the intent.
    Mismatch,
    /// The reviewer could not see the thing. **Never** a statement about the take.
    Unobserved,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Mismatch => "mismatch",
            Self::Unobserved => "unobserved",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "match" => Some(Self::Match),
            "mismatch" => Some(Self::Mismatch),
            "unobserved" => Some(Self::Unobserved),
            _ => None,
        }
    }
}

/// One backend answer about one frame, kept verbatim so a person can audit the grading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameAnswer {
    pub frame_id: String,
    /// The backend's answer, verbatim.
    pub answer: String,
    pub verdict: Verdict,
    pub confidence: f64,
    pub hedged: bool,
    pub elapsed_seconds: f64,
}

/// One question's observation across the frames it was graded on.
///
/// Invariant, asserted in tests: `unobserved == true` implies `observed.is_none()`. An action the
/// reviewer did not see is not a completed action, and this type cannot express that it is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Observation {
    pub question_id: String,
    pub topic: String,
    /// The question as put to the backend.
    pub question: String,
    /// The intended claim, restated from the review document for the reader of this file.
    pub intended: String,
    pub frames: String,
    pub verdict: Verdict,
    /// What was read off the evidence. `None` whenever `unobserved` — there is no such thing as an
    /// unobserved value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<String>,
    pub unobserved: bool,
    /// Confidence in `observed`, in `[0, 1]`. `0.0` when unobserved.
    pub confidence: f64,
    pub evidence_frame_ids: Vec<String>,
    pub answers: Vec<FrameAnswer>,
}

impl Observation {
    /// The invariant this type exists to hold. Used by the writer and asserted by tests.
    pub fn is_well_formed(&self) -> bool {
        if self.unobserved {
            return self.verdict == Verdict::Unobserved
                && self.observed.is_none()
                && self.confidence == 0.0;
        }
        self.verdict != Verdict::Unobserved && self.observed.is_some()
    }
}

/// An actionable flag: which shot, which question, what was intended, what was observed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MismatchFlag {
    pub shot_id: String,
    pub question_id: String,
    pub topic: String,
    /// One of [`MISMATCH_SEVERITIES`].
    pub severity: String,
    pub intended: String,
    /// The observed value, or the literal `"unobserved"` when nothing was seen.
    pub observed: String,
    pub confidence: f64,
    pub evidence_frame_ids: Vec<String>,
    pub detail: String,
}

/// The adjacent selected take a `cut_continuity` question was compared against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdjacentTake {
    pub shot_id: String,
    pub attempt: u32,
    pub asset_id: String,
    pub dependency: String,
    /// The frame of the adjacent take the comparison used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_id: Option<String>,
}

/// Everything one review of one take observed. Written to its own file beside the run record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservedState {
    pub schema_version: u32,
    /// `<runId>:<shotId>:a<attempt>:r<n>` — unique per review, so re-reviewing a take appends
    /// rather than overwrites and the earlier evidence survives.
    pub review_id: String,
    pub reviewed_at: String,
    pub shot_id: String,
    /// The attempt whose take was reviewed. A review is always about ONE take.
    pub attempt: u32,
    pub take_asset_id: String,
    pub intended: IntendedRef,
    pub review_plan: ReviewSourceRef,
    pub backend: ReviewBackendRecord,
    pub limits: ReviewLimits,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjacent: Option<AdjacentTake>,
    pub frames: Vec<FrameEvidence>,
    pub observations: Vec<Observation>,
    pub mismatches: Vec<MismatchFlag>,
    /// Set when a declared limit ended the review early. The partial evidence is kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    pub elapsed_seconds: f64,
    /// [`ASSISTIVE_NOTICE`], verbatim.
    pub notice: String,
}

impl ObservedState {
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Flags a person has to look at: everything except the low-confidence `uncertain` tail.
    pub fn actionable(&self) -> Vec<&MismatchFlag> {
        self.mismatches
            .iter()
            .filter(|flag| flag.severity != "uncertain")
            .collect()
    }

    pub fn unobserved_count(&self) -> usize {
        self.observations
            .iter()
            .filter(|observation| observation.unobserved)
            .count()
    }
}

// ---------------------------------------------------------------------------------------------
// Grading
// ---------------------------------------------------------------------------------------------

fn contains_any(haystack: &str, needles: &[&str]) -> Option<String> {
    needles
        .iter()
        .find(|needle| haystack.contains(**needle))
        .map(|needle| (*needle).to_owned())
}

fn contains_any_owned(haystack: &str, needles: &[String]) -> Option<String> {
    needles
        .iter()
        .map(|needle| needle.trim().to_ascii_lowercase())
        .filter(|needle| !needle.is_empty())
        .find(|needle| haystack.contains(needle.as_str()))
}

/// Grade one backend answer against one question.
///
/// Order is load-bearing:
///
/// 1. an empty answer, or one carrying an "I cannot see it" marker, is [`Verdict::Unobserved`] —
///    checked FIRST so a hedge about a colour never becomes a colour;
/// 2. then `contradict`, because the negative phrasing ("no courier") is the specific one and an
///    answer usually restates the subject of the question either way;
/// 3. then `expect`;
/// 4. an answer that matches nothing is `Unobserved`, not a match. Silence is not agreement.
pub fn grade_answer(
    question: &ReviewQuestion,
    answer: &str,
) -> (Verdict, Option<String>, f64, bool) {
    let lowered = answer.trim().to_ascii_lowercase();
    if lowered.is_empty() {
        return (Verdict::Unobserved, None, 0.0, false);
    }
    if contains_any(&lowered, UNOBSERVED_MARKERS).is_some() {
        return (Verdict::Unobserved, None, 0.0, true);
    }
    let hedged = contains_any(&lowered, HEDGE_MARKERS).is_some();
    let confidence = if hedged { HEDGED } else { CONFIDENT };
    if let Some(token) = contains_any_owned(&lowered, &question.contradict) {
        return (Verdict::Mismatch, Some(token), confidence, hedged);
    }
    if let Some(token) = contains_any_owned(&lowered, &question.expect) {
        return (Verdict::Match, Some(token), confidence, hedged);
    }
    (Verdict::Unobserved, None, 0.0, hedged)
}

/// Combine the per-frame answers for one question into its observation.
///
/// A contradiction on any graded frame wins outright (the fault was seen); otherwise the scope
/// decides whether the matches are enough. Anything else is `Unobserved` and carries no value.
pub fn aggregate_observation(question: &ReviewQuestion, answers: Vec<FrameAnswer>) -> Observation {
    let graded = answers.len();
    let mismatch = answers.iter().find(|a| a.verdict == Verdict::Mismatch);
    let matches: Vec<&FrameAnswer> = answers
        .iter()
        .filter(|a| a.verdict == Verdict::Match)
        .collect();

    let (verdict, observed, confidence, evidence): (Verdict, Option<String>, f64, Vec<String>) =
        if let Some(worst) = mismatch {
            // Cite every frame that saw the fault, worst (most confident) first in confidence.
            let ids = answers
                .iter()
                .filter(|a| a.verdict == Verdict::Mismatch)
                .map(|a| a.frame_id.clone())
                .collect();
            let best = answers
                .iter()
                .filter(|a| a.verdict == Verdict::Mismatch)
                .map(|a| a.confidence)
                .fold(worst.confidence, f64::max);
            (
                Verdict::Mismatch,
                worst.answer.trim().to_owned().into(),
                best,
                ids,
            )
        } else {
            let enough = match question.frames {
                FrameScope::All => graded > 0 && matches.len() == graded,
                FrameScope::Any | FrameScope::First | FrameScope::Last => !matches.is_empty(),
            };
            if enough {
                let confidence = matches
                    .iter()
                    .map(|a| a.confidence)
                    .fold(f64::INFINITY, f64::min);
                (
                    Verdict::Match,
                    matches
                        .first()
                        .map(|answer| answer.answer.trim().to_owned()),
                    if confidence.is_finite() {
                        confidence
                    } else {
                        0.0
                    },
                    matches.iter().map(|a| a.frame_id.clone()).collect(),
                )
            } else {
                (
                    Verdict::Unobserved,
                    None,
                    0.0,
                    answers.iter().map(|a| a.frame_id.clone()).collect(),
                )
            }
        };

    Observation {
        question_id: question.id.clone(),
        topic: question.topic.clone(),
        question: question.ask.clone(),
        intended: question.intended.clone(),
        frames: question.frames.as_str().to_owned(),
        verdict,
        unobserved: verdict == Verdict::Unobserved,
        observed: if verdict == Verdict::Unobserved {
            None
        } else {
            observed
        },
        confidence: if verdict == Verdict::Unobserved {
            0.0
        } else {
            confidence
        },
        evidence_frame_ids: evidence,
        answers,
    }
}

/// The flag one observation raises, if any.
///
/// * a confident contradiction is a `mismatch`;
/// * a hedged contradiction is `uncertain` — the doubt survives instead of being rounded up;
/// * an `unobserved` answer to a `mustObserve` question is an `unobserved` flag, which is how a
///   parcel handoff nobody saw stays visible without ever being called completed;
/// * an `unobserved` answer to any other question raises nothing: the reviewer simply has nothing
///   to say, and saying nothing loudly is noise.
pub fn flag_for(
    shot_id: &str,
    question: &ReviewQuestion,
    observation: &Observation,
    uncertain_below: f64,
) -> Option<MismatchFlag> {
    let base = |severity: &str, observed: String, detail: String| MismatchFlag {
        shot_id: shot_id.to_owned(),
        question_id: question.id.clone(),
        topic: question.topic.clone(),
        severity: severity.to_owned(),
        intended: question.intended.clone(),
        observed,
        confidence: observation.confidence,
        evidence_frame_ids: observation.evidence_frame_ids.clone(),
        detail,
    };
    match observation.verdict {
        Verdict::Match => None,
        Verdict::Mismatch => {
            let observed = observation.observed.clone().unwrap_or_default();
            let severity = if observation.confidence < uncertain_below {
                "uncertain"
            } else {
                "mismatch"
            };
            let detail = format!(
                "{shot_id} {}: intended {:?}, the reviewer read {:?} off {} (confidence {:.2}). \
                 Look at the frames before acting.",
                question.topic,
                question.intended,
                observed,
                frame_list(&observation.evidence_frame_ids),
                observation.confidence
            );
            Some(base(severity, observed, detail))
        }
        Verdict::Unobserved if question.must_observe => Some(base(
            "unobserved",
            "unobserved".to_owned(),
            format!(
                "{shot_id} {}: intended {:?}, and the reviewer could NOT see it in {}. This is \
                 recorded as unobserved, NOT as completed — look at the take yourself.",
                question.topic,
                question.intended,
                frame_list(&observation.evidence_frame_ids)
            ),
        )),
        Verdict::Unobserved => None,
    }
}

fn frame_list(ids: &[String]) -> String {
    if ids.is_empty() {
        "no frames".to_owned()
    } else {
        ids.join(", ")
    }
}

// ---------------------------------------------------------------------------------------------
// Labeled evaluation
// ---------------------------------------------------------------------------------------------

/// One frame of a labeled case, already on disk (the evaluation does not render anything).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvalFrame {
    /// Path relative to the set's `mediaRoot` (or to the set document's own directory).
    pub file: String,
    pub timestamp_seconds: f64,
}

/// One labeled take: its frames, and the verdict a correct reviewer should reach per question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvalCase {
    pub id: String,
    /// The shot in the review plan whose questions apply.
    pub shot_id: String,
    /// `correct` or a short name for the deliberate fault (`wrong_parcel_colour`, ...). Reported
    /// so a reader can see which kinds of fault the reviewer is blind to.
    pub label: String,
    #[serde(default)]
    pub description: String,
    pub frames: Vec<EvalFrame>,
    /// question id -> expected verdict (`match` / `mismatch` / `unobserved`). A question the case
    /// does not name is not scored for that case.
    pub expected: BTreeMap<String, String>,
}

/// A fixed labeled set: correct takes and deliberately broken ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvalSet {
    pub schema_version: u32,
    pub id: String,
    pub version: u32,
    #[serde(default)]
    pub description: String,
    /// Directory the frames are under, relative to the set document (or absolute). Absent means
    /// the document's own directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_root: Option<String>,
    /// The review plan whose questions are scored, relative to the set document.
    pub review_plan: String,
    pub cases: Vec<EvalCase>,
}

/// Parse a labeled set from JSON/JSONC text.
pub fn parse_eval_set(text: &str) -> Result<EvalSet, String> {
    serde_json::from_str(&strip_jsonc_comments(text)).map_err(|error| error.to_string())
}

/// Findings on a labeled set against the review plan it scores.
pub fn validate_eval_set(set: &EvalSet, review: &ReviewPlan) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if set.schema_version != REVIEW_EVAL_SCHEMA_VERSION {
        findings.push(PlanDiagnostic::plan(
            "schemaVersion",
            format!(
                "labeled set schema version {} is not supported (this build reads \
                 {REVIEW_EVAL_SCHEMA_VERSION})",
                set.schema_version
            ),
        ));
    }
    if !is_safe_plan_id(&set.id) {
        findings.push(PlanDiagnostic::plan(
            "id",
            format!("{:?} is not a safe id", set.id),
        ));
    }
    if set.cases.is_empty() {
        findings.push(PlanDiagnostic::plan("cases", "a labeled set needs cases"));
    }
    if !set.cases.iter().any(|case| case.label == "correct") {
        findings.push(PlanDiagnostic::plan(
            "cases",
            "a set with no correct takes cannot report false alarms; include at least one",
        ));
    }
    if !set.cases.iter().any(|case| case.label != "correct") {
        findings.push(PlanDiagnostic::plan(
            "cases",
            "a set with no broken takes cannot report detections or misses; include at least one",
        ));
    }
    let mut ids = BTreeSet::new();
    for case in &set.cases {
        if !is_safe_plan_id(&case.id) {
            findings.push(PlanDiagnostic::plan(
                "cases[].id",
                format!("{:?} is not a safe id", case.id),
            ));
        } else if !ids.insert(case.id.as_str()) {
            findings.push(PlanDiagnostic::plan(
                "cases[].id",
                format!("duplicate case id {:?}", case.id),
            ));
        }
        let Some(spec) = review.shots.get(&case.shot_id) else {
            findings.push(PlanDiagnostic::plan(
                format!("cases.{}.shotId", case.id),
                format!(
                    "{:?} is not a shot the review plan {:?} asks about",
                    case.shot_id, review.id
                ),
            ));
            continue;
        };
        if case.frames.is_empty() {
            findings.push(PlanDiagnostic::plan(
                format!("cases.{}.frames", case.id),
                "a case needs at least one frame",
            ));
        }
        if case.expected.is_empty() {
            findings.push(PlanDiagnostic::plan(
                format!("cases.{}.expected", case.id),
                "an unlabelled case scores nothing",
            ));
        }
        for (question_id, verdict) in &case.expected {
            if !spec
                .questions
                .iter()
                .any(|question| &question.id == question_id)
            {
                findings.push(PlanDiagnostic::plan(
                    format!("cases.{}.expected.{question_id}", case.id),
                    format!(
                        "{question_id:?} is not a question of shot {:?} in review plan {:?}",
                        case.shot_id, review.id
                    ),
                ));
            }
            if Verdict::parse(verdict).is_none() {
                findings.push(PlanDiagnostic::plan(
                    format!("cases.{}.expected.{question_id}", case.id),
                    format!("{verdict:?} is not match / mismatch / unobserved"),
                ));
            }
        }
    }
    findings
}

/// Counts for one question across the whole labeled set.
///
/// The definitions, stated once so the numbers mean something:
///
/// | expected | reported | counted as |
/// | --- | --- | --- |
/// | `mismatch` | `mismatch` | **detection** |
/// | `mismatch` | `match` | **miss** |
/// | `mismatch` | `unobserved` | **miss** (a fault nobody saw is a fault nobody caught) |
/// | `match` | `mismatch` | **false alarm** |
/// | `match` | `match` | correct |
/// | `match` | `unobserved` | **abstention** (not a false alarm: nothing was claimed) |
/// | `unobserved` | `unobserved` | correct |
/// | `unobserved` | `match` / `mismatch` | **overclaim** — the reviewer claimed to see something
///   the label says is not visible. Tracked separately because it is the failure mode that would
///   turn an uncertain observation into a fact. |
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScoreCounts {
    pub scored: u32,
    pub correct: u32,
    pub detections: u32,
    pub misses: u32,
    pub false_alarms: u32,
    pub abstentions: u32,
    pub overclaims: u32,
}

impl ScoreCounts {
    fn add(&mut self, expected: Verdict, reported: Verdict) {
        self.scored += 1;
        match (expected, reported) {
            (Verdict::Mismatch, Verdict::Mismatch) => {
                self.detections += 1;
                self.correct += 1;
            }
            (Verdict::Mismatch, _) => self.misses += 1,
            (Verdict::Match, Verdict::Mismatch) => self.false_alarms += 1,
            (Verdict::Match, Verdict::Match) => self.correct += 1,
            (Verdict::Match, Verdict::Unobserved) => self.abstentions += 1,
            (Verdict::Unobserved, Verdict::Unobserved) => self.correct += 1,
            (Verdict::Unobserved, _) => self.overclaims += 1,
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.scored += other.scored;
        self.correct += other.correct;
        self.detections += other.detections;
        self.misses += other.misses;
        self.false_alarms += other.false_alarms;
        self.abstentions += other.abstentions;
        self.overclaims += other.overclaims;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionScore {
    pub question_id: String,
    pub topic: String,
    #[serde(flatten)]
    pub counts: ScoreCounts,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaseOutcome {
    pub case_id: String,
    pub shot_id: String,
    pub label: String,
    #[serde(flatten)]
    pub counts: ScoreCounts,
    /// question id -> `expected/reported`, for reading a single case's story.
    pub verdicts: BTreeMap<String, String>,
    /// Where the observed-state document for this case was written.
    pub observed_state_path: String,
}

/// What one run of the labeled set found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalResults {
    pub schema_version: u32,
    pub ran_at: String,
    pub set_id: String,
    pub set_version: u32,
    pub review_plan: ReviewSourceRef,
    pub backend: ReviewBackendRecord,
    pub totals: ScoreCounts,
    pub per_question: Vec<QuestionScore>,
    pub per_topic: BTreeMap<String, ScoreCounts>,
    pub per_case: Vec<CaseOutcome>,
    pub elapsed_seconds: f64,
    /// [`ASSISTIVE_NOTICE`], verbatim.
    pub notice: String,
}

/// Score one case's observations against its labels.
pub fn score_case(case: &EvalCase, observed: &ObservedState) -> CaseOutcome {
    let mut counts = ScoreCounts::default();
    let mut verdicts = BTreeMap::new();
    for (question_id, expected_text) in &case.expected {
        let Some(expected) = Verdict::parse(expected_text) else {
            continue;
        };
        // A question the reviewer never answered (a limit cut the review short) is `unobserved`:
        // the reviewer has nothing to say, which is exactly what the absence means.
        let reported = observed
            .observations
            .iter()
            .find(|observation| &observation.question_id == question_id)
            .map(|observation| observation.verdict)
            .unwrap_or(Verdict::Unobserved);
        counts.add(expected, reported);
        verdicts.insert(
            question_id.clone(),
            format!("{}/{}", expected.as_str(), reported.as_str()),
        );
    }
    CaseOutcome {
        case_id: case.id.clone(),
        shot_id: case.shot_id.clone(),
        label: case.label.clone(),
        counts,
        verdicts,
        observed_state_path: String::new(),
    }
}

/// Roll per-case outcomes up into per-question, per-topic and total counts.
pub fn tally(
    review: &ReviewPlan,
    set: &EvalSet,
    outcomes: &[(&EvalCase, &ObservedState, CaseOutcome)],
) -> (
    ScoreCounts,
    Vec<QuestionScore>,
    BTreeMap<String, ScoreCounts>,
) {
    let mut totals = ScoreCounts::default();
    let mut per_question: BTreeMap<String, (String, ScoreCounts)> = BTreeMap::new();
    let mut per_topic: BTreeMap<String, ScoreCounts> = BTreeMap::new();
    // Seed every question the set could score, so a question that was never reached still appears
    // with zeroes instead of silently vanishing from the report.
    for case in &set.cases {
        let Some(spec) = review.shots.get(&case.shot_id) else {
            continue;
        };
        for question in &spec.questions {
            per_question
                .entry(question.id.clone())
                .or_insert_with(|| (question.topic.clone(), ScoreCounts::default()));
            per_topic.entry(question.topic.clone()).or_default();
        }
    }
    for (case, observed, outcome) in outcomes {
        let _ = outcome;
        let Some(spec) = review.shots.get(&case.shot_id) else {
            continue;
        };
        for (question_id, expected_text) in &case.expected {
            let Some(expected) = Verdict::parse(expected_text) else {
                continue;
            };
            let Some(question) = spec.questions.iter().find(|q| &q.id == question_id) else {
                continue;
            };
            let reported = observed
                .observations
                .iter()
                .find(|observation| &observation.question_id == question_id)
                .map(|observation| observation.verdict)
                .unwrap_or(Verdict::Unobserved);
            totals.add(expected, reported);
            per_question
                .entry(question_id.clone())
                .or_insert_with(|| (question.topic.clone(), ScoreCounts::default()))
                .1
                .add(expected, reported);
            per_topic
                .entry(question.topic.clone())
                .or_default()
                .add(expected, reported);
        }
    }
    let scores = per_question
        .into_iter()
        .map(|(question_id, (topic, counts))| QuestionScore {
            question_id,
            topic,
            counts,
        })
        .collect();
    (totals, scores, per_topic)
}

/// The human-readable report `film-harness review-eval` prints and writes beside the results.
///
/// It always ends with [`ASSISTIVE_NOTICE`]: a table of detections is exactly the artefact someone
/// would otherwise read as a quality bar.
pub fn format_eval_report(results: &EvalResults) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "labeled review evaluation — set {} v{} ({} cases, {:.1}s) via {} [{}]\n",
        results.set_id,
        results.set_version,
        results.per_case.len(),
        results.elapsed_seconds,
        results.backend.model,
        results.backend.kind
    ));
    if !results.backend.real_model_inference {
        out.push_str(
            "  NOTE: no model weights ran for this evaluation; the answers were scripted.\n",
        );
    }
    let t = &results.totals;
    out.push_str(&format!(
        "  totals: {} scored, {} correct, {} detections, {} misses, {} false alarms, {} \
         abstentions, {} overclaims\n",
        t.scored, t.correct, t.detections, t.misses, t.false_alarms, t.abstentions, t.overclaims
    ));
    out.push_str("  per question:\n");
    for score in &results.per_question {
        let c = &score.counts;
        out.push_str(&format!(
            "    {:<24} {:<20} scored={} det={} miss={} fa={} abst={} over={}\n",
            score.question_id,
            score.topic,
            c.scored,
            c.detections,
            c.misses,
            c.false_alarms,
            c.abstentions,
            c.overclaims
        ));
    }
    out.push_str("  per case:\n");
    for case in &results.per_case {
        let c = &case.counts;
        out.push_str(&format!(
            "    {:<24} {:<22} det={} miss={} fa={} over={}\n",
            case.case_id, case.label, c.detections, c.misses, c.false_alarms, c.overclaims
        ));
    }
    out.push_str(&format!("\n{ASSISTIVE_NOTICE}\n"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn question(id: &str) -> ReviewQuestion {
        ReviewQuestion {
            id: id.to_owned(),
            topic: "parcel_identity".to_owned(),
            intended: "A bright red parcel is on the bench.".to_owned(),
            ask: "What colour is the parcel on the bench?".to_owned(),
            expect: vec!["red".to_owned()],
            contradict: vec!["blue".to_owned(), "no parcel".to_owned()],
            frames: FrameScope::All,
            must_observe: false,
            across_cut: false,
        }
    }

    fn answer(frame: &str, text: &str, question: &ReviewQuestion) -> FrameAnswer {
        let (verdict, _, confidence, hedged) = grade_answer(question, text);
        FrameAnswer {
            frame_id: frame.to_owned(),
            answer: text.to_owned(),
            verdict,
            confidence,
            hedged,
            elapsed_seconds: 0.1,
        }
    }

    #[test]
    fn an_unseeable_answer_is_unobserved_and_never_a_value() {
        let q = question("parcel_colour");
        for text in [
            "I cannot tell — the parcel is out of frame.",
            "It is not visible in this frame.",
            "",
            "unknown",
        ] {
            let (verdict, observed, confidence, _) = grade_answer(&q, text);
            assert_eq!(verdict, Verdict::Unobserved, "{text:?}");
            assert!(observed.is_none(), "{text:?} produced a value");
            assert_eq!(confidence, 0.0);
        }
    }

    #[test]
    fn a_hedge_keeps_the_answer_but_halves_its_confidence() {
        let q = question("parcel_colour");
        let (verdict, observed, confidence, hedged) =
            grade_answer(&q, "It appears to be blue, though the light is warm.");
        assert_eq!(verdict, Verdict::Mismatch);
        assert_eq!(observed.as_deref(), Some("blue"));
        assert!(hedged);
        assert!(confidence < DEFAULT_UNCERTAIN_BELOW, "{confidence}");
    }

    #[test]
    fn an_answer_matching_nothing_is_unobserved_not_a_match() {
        let q = question("parcel_colour");
        let (verdict, observed, _, _) = grade_answer(&q, "There is a wooden bench and sawdust.");
        assert_eq!(verdict, Verdict::Unobserved);
        assert!(observed.is_none());
    }

    #[test]
    fn a_contradiction_on_any_frame_beats_matches_on_the_others() {
        let q = question("parcel_colour");
        let observation = aggregate_observation(
            &q,
            vec![
                answer("f1", "The parcel is red.", &q),
                answer("f2", "The parcel is blue.", &q),
            ],
        );
        assert_eq!(observation.verdict, Verdict::Mismatch);
        assert_eq!(observation.evidence_frame_ids, vec!["f2".to_owned()]);
        assert!(observation.is_well_formed());
    }

    #[test]
    fn scope_all_refuses_to_call_a_partial_sighting_a_match() {
        let q = question("parcel_colour");
        let observation = aggregate_observation(
            &q,
            vec![
                answer("f1", "The parcel is red.", &q),
                answer("f2", "I cannot tell.", &q),
            ],
        );
        assert_eq!(observation.verdict, Verdict::Unobserved);
        assert!(observation.observed.is_none());
        assert!(observation.is_well_formed());

        let mut any = q.clone();
        any.frames = FrameScope::Any;
        let observation = aggregate_observation(
            &any,
            vec![
                answer("f1", "The parcel is red.", &any),
                answer("f2", "I cannot tell.", &any),
            ],
        );
        assert_eq!(observation.verdict, Verdict::Match);
    }

    #[test]
    fn an_unobserved_must_observe_question_flags_as_unobserved_never_completed() {
        let mut q = question("handoff");
        q.topic = "action_completion".to_owned();
        q.intended = "The courier lets go of the parcel on the bench.".to_owned();
        q.must_observe = true;
        q.frames = FrameScope::Last;
        q.expect = vec!["lets go".to_owned(), "hands are empty".to_owned()];
        q.contradict = vec!["still holding".to_owned()];
        let observation =
            aggregate_observation(&q, vec![answer("f3", "I cannot tell from this frame.", &q)]);
        assert_eq!(observation.verdict, Verdict::Unobserved);
        let flag = flag_for("SH030", &q, &observation, DEFAULT_UNCERTAIN_BELOW)
            .expect("a mustObserve question raises a flag when nothing was seen");
        assert_eq!(flag.severity, "unobserved");
        assert_eq!(flag.observed, "unobserved");
        assert!(flag.detail.contains("NOT as completed"), "{}", flag.detail);
        // The serialized form must not carry a value either.
        let json = serde_json::to_value(&observation).expect("observation serializes");
        assert!(json.get("observed").is_none(), "{json}");
        assert_eq!(json["unobserved"], json!(true));
    }

    #[test]
    fn an_unobserved_ordinary_question_raises_nothing() {
        let q = question("parcel_colour");
        let observation = aggregate_observation(&q, vec![answer("f1", "I cannot tell.", &q)]);
        assert!(flag_for("SH030", &q, &observation, DEFAULT_UNCERTAIN_BELOW).is_none());
    }

    #[test]
    fn a_hedged_contradiction_is_uncertain_and_a_confident_one_is_a_mismatch() {
        let q = question("parcel_colour");
        let hedged = aggregate_observation(&q, vec![answer("f1", "It seems blue.", &q)]);
        let flag = flag_for("SH030", &q, &hedged, DEFAULT_UNCERTAIN_BELOW).expect("flag");
        assert_eq!(flag.severity, "uncertain");

        let sure = aggregate_observation(&q, vec![answer("f1", "The parcel is blue.", &q)]);
        let flag = flag_for("SH030", &q, &sure, DEFAULT_UNCERTAIN_BELOW).expect("flag");
        assert_eq!(flag.severity, "mismatch");
        assert!(flag.detail.contains("Look at the frames"));
    }

    #[test]
    fn score_counts_separate_misses_false_alarms_and_overclaims() {
        let mut counts = ScoreCounts::default();
        counts.add(Verdict::Mismatch, Verdict::Mismatch); // detection
        counts.add(Verdict::Mismatch, Verdict::Match); // miss
        counts.add(Verdict::Mismatch, Verdict::Unobserved); // miss
        counts.add(Verdict::Match, Verdict::Mismatch); // false alarm
        counts.add(Verdict::Match, Verdict::Match); // correct
        counts.add(Verdict::Match, Verdict::Unobserved); // abstention
        counts.add(Verdict::Unobserved, Verdict::Unobserved); // correct
        counts.add(Verdict::Unobserved, Verdict::Match); // overclaim
        assert_eq!(counts.scored, 8);
        assert_eq!(counts.detections, 1);
        assert_eq!(counts.misses, 2);
        assert_eq!(counts.false_alarms, 1);
        assert_eq!(counts.abstentions, 1);
        assert_eq!(counts.overclaims, 1);
        assert_eq!(counts.correct, 3);
    }

    fn plan_with_shots(ids: &[&str]) -> ProductionPlan {
        let shots: Vec<Value> = ids
            .iter()
            .enumerate()
            .map(|(index, id)| {
                let mut shot = json!({
                    "id": id, "beat": "b", "framing": "f", "prompt": "p",
                    "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                    "conditioning": { "mode": "text_to_video" }
                });
                if index > 0 {
                    shot["dependsOn"] =
                        json!([{ "shotId": ids[index - 1], "kind": "continuity", "note": "" }]);
                }
                shot
            })
            .collect();
        serde_json::from_value(json!({
            "schemaVersion": 2, "id": "p", "version": 1, "title": "t",
            "model": { "id": "minimax_h3", "tier": "q4", "resolution": "576x320" },
            "limits": { "maxRunSeconds": 10, "maxShotSeconds": 10, "maxAttemptsPerShot": 1, "maxMemoryGb": 1 },
            "shots": shots
        }))
        .expect("plan parses")
    }

    fn review_doc(shot: &str, question: Value) -> ReviewPlan {
        serde_json::from_value(json!({
            "schemaVersion": 1, "id": "r", "version": 1,
            "sampling": { "positions": [0.1, 0.5, 0.9] },
            "limits": { "maxSeconds": 600, "maxFramesPerShot": 3, "maxQuestionsPerShot": 8, "maxAnswerSeconds": 120 },
            "shots": { shot: { "questions": [question] } }
        }))
        .expect("review plan parses")
    }

    #[test]
    fn a_review_plan_is_refused_for_an_unknown_shot_topic_or_ungradable_question() {
        let plan = plan_with_shots(&["SH010", "SH020"]);
        let review = review_doc(
            "SH999",
            json!({ "id": "q", "topic": "location", "intended": "i", "ask": "a", "expect": ["x"] }),
        );
        let findings = validate_review_plan(&review, &plan);
        assert!(
            findings.iter().any(|f| f.message.contains("not a shot")),
            "{findings:#?}"
        );

        let review = review_doc(
            "SH010",
            json!({ "id": "q", "topic": "vibes", "intended": "i", "ask": "a", "expect": ["x"] }),
        );
        assert!(validate_review_plan(&review, &plan)
            .iter()
            .any(|f| f.message.contains("not a review topic")));

        let review = review_doc(
            "SH010",
            json!({ "id": "q", "topic": "location", "intended": "i", "ask": "a" }),
        );
        assert!(validate_review_plan(&review, &plan)
            .iter()
            .any(|f| f.message.contains("gradable")));

        let review = review_doc(
            "SH010",
            json!({ "id": "q", "topic": "cut_continuity", "intended": "i", "ask": "a",
                    "expect": ["x"], "acrossCut": true }),
        );
        assert!(
            validate_review_plan(&review, &plan)
                .iter()
                .any(|f| f.message.contains("dependsOn")),
            "SH010 declares no edge, so acrossCut has nothing to compare against"
        );
    }

    #[test]
    fn a_valid_review_plan_produces_no_findings() {
        let plan = plan_with_shots(&["SH010", "SH020"]);
        let review = review_doc(
            "SH020",
            json!({ "id": "cut", "topic": "cut_continuity", "intended": "same room",
                    "ask": "a", "expect": ["same"], "contradict": ["different"],
                    "frames": "first", "acrossCut": true }),
        );
        assert!(
            validate_review_plan(&review, &plan).is_empty(),
            "{:#?}",
            validate_review_plan(&review, &plan)
        );
    }

    #[test]
    fn a_labeled_set_needs_both_correct_and_broken_cases() {
        let review = review_doc(
            "SH010",
            json!({ "id": "q", "topic": "location", "intended": "i", "ask": "a", "expect": ["x"] }),
        );
        let set: EvalSet = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "s", "version": 1, "reviewPlan": "review.jsonc",
            "cases": [{
                "id": "only_broken", "shotId": "SH010", "label": "wrong_location",
                "frames": [{ "file": "a.png", "timestampSeconds": 0.0 }],
                "expected": { "q": "mismatch" }
            }]
        }))
        .expect("set parses");
        let findings = validate_eval_set(&set, &review);
        assert!(
            findings.iter().any(|f| f.message.contains("false alarms")),
            "{findings:#?}"
        );
    }

    #[test]
    fn the_eval_report_always_says_review_is_not_quality_assurance() {
        let results = EvalResults {
            schema_version: REVIEW_EVAL_SCHEMA_VERSION,
            ran_at: "2026-09-13T00:00:00Z".to_owned(),
            set_id: "s".to_owned(),
            set_version: 1,
            review_plan: ReviewSourceRef {
                id: "r".to_owned(),
                version: 1,
                path: "review.jsonc".to_owned(),
                sha256: "0".to_owned(),
            },
            backend: ReviewBackendRecord {
                kind: "scripted".to_owned(),
                model: "none".to_owned(),
                route: "none".to_owned(),
                real_model_inference: false,
            },
            totals: ScoreCounts::default(),
            per_question: Vec::new(),
            per_topic: BTreeMap::new(),
            per_case: Vec::new(),
            elapsed_seconds: 0.0,
            notice: ASSISTIVE_NOTICE.to_owned(),
        };
        let text = format_eval_report(&results);
        assert!(text.contains("ASSISTIVE, not quality assurance"), "{text}");
        assert!(text.contains("no model weights ran"), "{text}");
    }
}
