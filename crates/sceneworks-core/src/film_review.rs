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
    // The conditioning plate of an image-conditioned shot is a flat field, and this is what the
    // model says about it (sc-22714 real-weights smoke): "The image is too dark to determine the
    // color of the jacket." Without it that answer matched nothing and read as silence.
    "too dark",
    "cannot make out",
    "could not make out",
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

/// Tokens one backend answer is truncated to when the review document declares no `maxNewTokens`.
/// The questions are closed with declared vocabularies, so this is generous; it bounds one backend
/// call, it does not shape the answer.
pub const DEFAULT_MAX_NEW_TOKENS: u32 = 192;

/// Memory the review declares it needs, in GB, when the document names none. The understanding
/// model the review drives (SenseNova-U1-8B) declares `minMemoryGb: 16`, so a host with less than
/// this cannot answer a question at all and the review says so before it creates anything.
pub const DEFAULT_MAX_MEMORY_GB: f64 = 16.0;

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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
    /// Tokens one backend answer is truncated to. Declared here rather than hard-coded in the
    /// caller so a review's whole cost is readable off its own document (epic requirement E5).
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,
    /// Memory the review declares it needs, in GB, checked against the API HOST's reported memory
    /// before anything is dispatched — the same shape as a production plan's `limits.maxMemoryGb`.
    #[serde(default = "default_max_memory_gb")]
    pub max_memory_gb: f64,
}

fn default_max_new_tokens() -> u32 {
    DEFAULT_MAX_NEW_TOKENS
}

fn default_max_memory_gb() -> f64 {
    DEFAULT_MAX_MEMORY_GB
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
        (
            "limits.maxNewTokens",
            u64::from(review.limits.max_new_tokens),
        ),
    ] {
        if value == 0 {
            findings.push(PlanDiagnostic::plan(field, format!("{field} must be >= 1")));
        }
    }
    if !review.limits.max_memory_gb.is_finite() || review.limits.max_memory_gb <= 0.0 {
        findings.push(PlanDiagnostic::plan(
            "limits.maxMemoryGb",
            "the memory ceiling must be a finite number > 0",
        ));
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
    /// The declared token the grader actually matched, and whether the clause around it affirmed
    /// or negated it. Without these a mis-grade is undebuggable from the record: the sc-22714
    /// real-weights smoke produced three of them and none could be diagnosed from the document
    /// alone (the negated "workshop" that scored a match, the "red" jacket that matched nothing,
    /// the "person's hands" that matched nothing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched: Option<String>,
    /// `affirmed`, `negated` or `none` — the polarity of the clause the token was found in.
    pub polarity: String,
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
    /// Why nothing was read, when the question was never put to the backend at all — an
    /// `acrossCut` question with no adjacent selected take, above all. A question that reached the
    /// backend carries its answers instead, and a skipped one used to be omitted from the document
    /// entirely, which reads exactly like a question nobody declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Observation {
    /// The invariant this type exists to hold. Used by the writer and asserted by tests.
    pub fn is_well_formed(&self) -> bool {
        self.well_formed_error().is_none()
    }

    /// The invariant violation this observation carries, named — `None` when it is well formed.
    ///
    /// The pair (`unobserved`, `observed`) is independently settable, so the "an unobserved
    /// observation carries no value" rule is a rule about VALUES rather than about types. It is
    /// therefore checked wherever an `Observation` is built or read back
    /// (`film_harness::review::answer_questions` and `read_observed_state`), unconditionally — a
    /// `debug_assert!` would hold it in the test binary and nowhere a person's document is read.
    pub fn well_formed_error(&self) -> Option<String> {
        let question = &self.question_id;
        if self.unobserved {
            if self.verdict != Verdict::Unobserved {
                return Some(format!(
                    "observation {question:?} is marked unobserved but carries verdict {}",
                    self.verdict.as_str()
                ));
            }
            if let Some(observed) = &self.observed {
                return Some(format!(
                    "observation {question:?} is unobserved but carries the value {observed:?}; \
                     there is no such thing as an unobserved value"
                ));
            }
            if self.confidence != 0.0 {
                return Some(format!(
                    "observation {question:?} is unobserved but claims confidence {}",
                    self.confidence
                ));
            }
            return None;
        }
        if self.verdict == Verdict::Unobserved {
            return Some(format!(
                "observation {question:?} reads unobserved but is not marked unobserved"
            ));
        }
        if self.observed.is_none() {
            return Some(format!(
                "observation {question:?} claims verdict {} but names no observed value",
                self.verdict.as_str()
            ));
        }
        None
    }
}

/// The observation for a question the reviewer could not put at all, naming why in `note`.
///
/// Skipping such a question silently — which is what an `acrossCut` question with no adjacent
/// selected take used to do — leaves the document indistinguishable from one where the question was
/// never declared, and a reader cannot tell "nobody asked" from "nobody answered".
pub fn unasked_observation(question: &ReviewQuestion, reason: &str) -> Observation {
    Observation {
        question_id: question.id.clone(),
        topic: question.topic.clone(),
        question: question.ask.clone(),
        intended: question.intended.clone(),
        frames: question.frames.as_str().to_owned(),
        verdict: Verdict::Unobserved,
        observed: None,
        unobserved: true,
        confidence: 0.0,
        evidence_frame_ids: Vec::new(),
        answers: Vec::new(),
        note: Some(reason.to_owned()),
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

/// Words that negate the clause they appear in.
const NEGATIONS: &[&str] = &[
    "no", "not", "nor", "never", "without", "cannot", "none", "nothing", "nobody", "neither",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "can't",
];

/// How many words may sit between two words of a declared token and still match it. Enough for the
/// modifiers a VLM inserts ("no SMALL parcel", "not a CLUTTERED WOODWORKING workshop") without
/// letting a token drift across a whole clause.
const TOKEN_GAP: usize = 3;

/// Words that open a closed question's answer as the ANSWER ITSELF rather than as a negation of
/// everything after them. "No, this is a kitchen." says the room is a kitchen; reading its leading
/// "No," as a clause negation loses the most common negative answer shape these questions get.
const ANSWER_PARTICLES: &[&str] = &["yes", "no", "yeah", "yep", "nope", "nah"];

/// Punctuation that terminates a leading answer particle. A particle is only an answer when it is
/// punctuated off from the sentence — "no parcel is visible" is an ordinary negated clause.
const PARTICLE_TERMINATORS: &[char] = &[',', ':', '-', '\u{2013}', '\u{2014}'];

/// Sentence separators an answer is split on.
const CLAUSE_SEPARATORS: &[char] = &['.', ';', '!', '?', '\n'];

/// Words a clause must have before it can be read as a restatement of the question.
const MIN_ECHO_WORDS: usize = 4;

/// One sentence of an answer, as lowercased words.
struct Clause {
    words: Vec<String>,
    /// The word index the negation scan starts at: `1` when the clause opens with a yes/no answer
    /// particle terminated by a comma, dash or colon, so that particle is graded as the answer
    /// token it is instead of negating every token after it.
    negation_from: usize,
    /// The punctuation that ended this clause, when it had one. `?` marks an interrogative, which
    /// is how a restatement of the question is told from an answer.
    terminator: Option<char>,
}

/// Split an answer into clauses and each clause into lowercased words.
///
/// Sentence boundaries matter twice: a token must lie inside ONE clause (so "no ... parcel" cannot
/// span two sentences), and both the negation scan and the hedge scan are clause-local (so
/// "Yes, there is a person. The room appears to be a workshop." is not read as a hedged sighting
/// of the person).
fn clauses_of(answer: &str) -> Vec<Clause> {
    let mut clauses = Vec::new();
    let mut start = 0;
    for (index, character) in answer.char_indices() {
        if CLAUSE_SEPARATORS.contains(&character) {
            push_clause(&answer[start..index], Some(character), &mut clauses);
            start = index + character.len_utf8();
        }
    }
    push_clause(&answer[start..], None, &mut clauses);
    clauses
}

fn push_clause(raw: &str, terminator: Option<char>, clauses: &mut Vec<Clause>) {
    let words: Vec<String> = raw
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|word| !word.is_empty())
        .map(|word| word.to_ascii_lowercase())
        .collect();
    if words.is_empty() {
        return;
    }
    let negation_from = usize::from(opens_with_answer_particle(raw));
    clauses.push(Clause {
        words,
        negation_from,
        terminator,
    });
}

/// Whether a clause opens with a yes/no answer particle punctuated off from the rest.
fn opens_with_answer_particle(raw: &str) -> bool {
    let trimmed = raw.trim_start().to_ascii_lowercase();
    ANSWER_PARTICLES.iter().any(|particle| {
        trimmed
            .strip_prefix(particle)
            .is_some_and(|rest| rest.trim_start().starts_with(PARTICLE_TERMINATORS))
    })
}

/// Whether `needle` occurs in `haystack` in order, gaps allowed.
fn is_subsequence(needle: &[String], haystack: &[String]) -> bool {
    let mut cursor = 0;
    for word in needle {
        match haystack[cursor..].iter().position(|other| other == word) {
            Some(offset) => cursor += offset + 1,
            None => return false,
        }
    }
    true
}

/// Drop the leading clauses that merely restate the question.
///
/// These models often lead with the question before answering it — "Is the door in this frame open
/// or closed? It is closed." — and the restatement carries every declared answer word the question
/// listed. Graded, it decides the verdict on the question's own wording: the echo above scored a
/// confident `Match` on `open` for a shot expecting an open door and a confident `Mismatch` on the
/// shot expecting a closed one, neither of which the model said.
fn strip_question_echo(clauses: Vec<Clause>, ask: &str) -> Vec<Clause> {
    let asked = token_words(ask);
    if asked.is_empty() {
        return clauses;
    }
    let echoes = clauses
        .iter()
        .take_while(|clause| is_question_echo(clause, &asked))
        .count();
    clauses.into_iter().skip(echoes).collect()
}

/// Whether one clause is a restatement of the question: an interrogative whose words all come from
/// the question in order, or a run of the question's own opening words.
fn is_question_echo(clause: &Clause, asked: &[String]) -> bool {
    if clause.words.len() < MIN_ECHO_WORDS {
        return false;
    }
    match clause.terminator {
        Some('?') => is_subsequence(&clause.words, asked),
        _ => asked.starts_with(&clause.words[..]),
    }
}

/// The word index at which `token`'s words appear in order inside `words`, allowing at most
/// [`TOKEN_GAP`] intervening words between consecutive token words. `None` when it does not occur.
///
/// Word-indexed rather than substring: `"red"` must not match inside `"covered"`, and
/// `"no parcel"` must match `"no small parcel or box"`.
fn token_position(words: &[String], token: &[String]) -> Option<usize> {
    if token.is_empty() || token.len() > words.len() {
        return None;
    }
    'start: for start in 0..=words.len() - token.len() {
        if words[start] != token[0] {
            continue;
        }
        let mut cursor = start + 1;
        for part in &token[1..] {
            let limit = (cursor + TOKEN_GAP + 1).min(words.len());
            match (cursor..limit).find(|index| &words[*index] == part) {
                Some(index) => cursor = index + 1,
                None => continue 'start,
            }
        }
        return Some(start);
    }
    None
}

fn token_words(token: &str) -> Vec<String> {
    token
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|word| !word.is_empty())
        .map(|word| word.to_ascii_lowercase())
        .collect()
}

/// Whether a token carries its own negation (`"no parcel"`, `"not a workshop"`, `"nobody"`).
///
/// Such a token is taken at FACE VALUE: applying clause negation to it would double-negate the
/// very phrasing it was written to catch ("No, there is no small parcel" would cancel itself out,
/// which is exactly how the real smoke turned a correctly-detected missing parcel into silence).
fn token_is_negating(token: &[String]) -> bool {
    token.iter().any(|word| NEGATIONS.contains(&word.as_str()))
}

/// Whether the words before `position` in the same clause negate what follows.
///
/// The scan starts at [`Clause::negation_from`], which skips a leading yes/no answer particle: in
/// "No, this is a kitchen." the "No," is the answer, not a negation of "kitchen". An inner negation
/// ("No, the room is not a workshop") still negates, because "not" sits inside the scanned range.
fn clause_negates(clause: &Clause, position: usize) -> bool {
    let from = clause.negation_from.min(position);
    clause.words[from..position]
        .iter()
        .any(|word| NEGATIONS.contains(&word.as_str()))
}

/// Where a declared token was found, and what the clause around it did to it.
struct TokenHit {
    token: String,
    /// Global word index, so the EARLIEST decisive hit in an answer can win.
    position: usize,
    negated: bool,
    /// Whether the clause it sits in hedges.
    hedged: bool,
}

fn find_hits(clauses: &[Clause], tokens: &[String]) -> Vec<TokenHit> {
    let mut hits = Vec::new();
    let mut offset = 0;
    for clause in clauses {
        for token in tokens {
            let parts = token_words(token);
            if parts.is_empty() {
                continue;
            }
            let Some(position) = token_position(&clause.words, &parts) else {
                continue;
            };
            hits.push(TokenHit {
                token: token.trim().to_ascii_lowercase(),
                position: offset + position,
                negated: !token_is_negating(&parts) && clause_negates(clause, position),
                hedged: clause_hedges(clause),
            });
        }
        offset += clause.words.len();
    }
    hits
}

/// Whether a clause hedges. Clause-local on purpose: "Yes, there is a person standing in the
/// doorway. The room appears to be a workshop." hedges the ROOM, not the person.
fn clause_hedges(clause: &Clause) -> bool {
    let text = clause.words.join(" ");
    HEDGE_MARKERS.iter().any(|marker| {
        let parts = token_words(marker);
        token_position(&clause.words, &parts).is_some() || text.contains(marker)
    })
}

fn answer_is_unobservable(clauses: &[Clause]) -> bool {
    clauses.iter().any(|clause| {
        UNOBSERVED_MARKERS
            .iter()
            .any(|marker| token_position(&clause.words, &token_words(marker)).is_some())
    })
}

/// What the grader concluded about one answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Grade {
    pub verdict: Verdict,
    /// The declared token that decided it, verbatim from the review plan.
    pub matched: Option<String>,
    /// `affirmed`, `negated` or `none`.
    pub polarity: String,
    pub confidence: f64,
    pub hedged: bool,
}

impl Grade {
    fn unobserved(hedged: bool) -> Self {
        Self {
            verdict: Verdict::Unobserved,
            matched: None,
            polarity: "none".to_owned(),
            confidence: 0.0,
            hedged,
        }
    }
}

/// Grade one backend answer against one question.
///
/// Matching is **word-indexed and polarity-aware**, both of which the sc-22714 real-weights smoke
/// proved necessary:
///
/// 1. an empty answer, or one carrying an "I cannot see it" marker, is [`Verdict::Unobserved`] —
///    checked FIRST, so a hedge about a colour never becomes a colour;
/// 2. every `expect` and `contradict` token is located by WORD, inside a single clause, tolerating
///    up to [`TOKEN_GAP`] intervening modifiers. Substring matching read `"red"` out of
///    `"covered"` and failed to find `"no parcel"` in `"no small parcel or box"`;
/// 3. each hit takes the POLARITY of its clause. An `expect` token inside a negated clause is a
///    **contradiction**, not a match — without this, "No, the room is not a cluttered woodworking
///    workshop" scored a match on the bare word "workshop". A token that carries its own negation
///    is taken at face value instead, so "no parcel" is not cancelled by the leading "No,". A
///    leading yes/no ANSWER PARTICLE ("No, this is a kitchen.") is the answer rather than a
///    negation scope, so the most common negative shape these closed questions get still grades;
/// 4. a `contradict` token inside a negated clause ("it is not red") decides nothing: it rules one
///    value out without establishing another, and inventing agreement from it is exactly the kind
///    of overclaim this module exists to prevent;
/// 5. a leading restatement of the question is stripped before any of this. These models lead with
///    the question ("Is the door in this frame open or closed? It is closed."), and the
///    restatement carries every declared answer word the question listed;
/// 6. an answer that AFFIRMS both an `expect` and a `contradict` value ("the parcel is red-brown")
///    is self-contradictory. It is `Unobserved`, with `matched` naming the conflict — letting word
///    order pick the winner reported a confident 0.9 verdict the answer never supported;
/// 7. otherwise the EARLIEST decisive hit wins — these answers lead with their verdict and then
///    elaborate;
/// 8. an answer that matches nothing is `Unobserved`, not a match. Silence is not agreement.
pub fn grade_answer(question: &ReviewQuestion, answer: &str) -> Grade {
    let clauses = strip_question_echo(clauses_of(answer.trim()), &question.ask);
    if clauses.is_empty() {
        return Grade::unobserved(false);
    }
    if answer_is_unobservable(&clauses) {
        return Grade::unobserved(true);
    }

    let mut hits: Vec<(Verdict, TokenHit)> = Vec::new();
    for hit in find_hits(&clauses, &question.contradict) {
        // A ruled-out value ("not red") establishes nothing; only an affirmed one contradicts.
        if !hit.negated {
            hits.push((Verdict::Mismatch, hit));
        }
    }
    for hit in find_hits(&clauses, &question.expect) {
        let verdict = if hit.negated {
            Verdict::Mismatch
        } else {
            Verdict::Match
        };
        hits.push((verdict, hit));
    }
    hits.sort_by_key(|(_, hit)| hit.position);

    let agreeing = hits
        .iter()
        .find(|(verdict, _)| *verdict == Verdict::Match)
        .map(|(_, hit)| hit);
    let contradicting = hits
        .iter()
        .find(|(verdict, _)| *verdict == Verdict::Mismatch)
        .map(|(_, hit)| hit);
    if let (Some(agreeing), Some(contradicting)) = (agreeing, contradicting) {
        // The answer both agrees and contradicts. Whichever came first is not evidence of anything,
        // so the reviewer says what it saw and claims nothing.
        return Grade {
            verdict: Verdict::Unobserved,
            matched: Some(format!(
                "conflict: {:?} and {:?} in one answer",
                agreeing.token, contradicting.token
            )),
            polarity: "none".to_owned(),
            confidence: 0.0,
            hedged: agreeing.hedged || contradicting.hedged,
        };
    }

    let Some((verdict, hit)) = hits.into_iter().next() else {
        return Grade::unobserved(clauses.iter().any(clause_hedges));
    };
    Grade {
        verdict,
        matched: Some(hit.token),
        polarity: if hit.negated { "negated" } else { "affirmed" }.to_owned(),
        confidence: if hit.hedged { HEDGED } else { CONFIDENT },
        hedged: hit.hedged,
    }
}

/// Combine an ACROSS-THE-CUT question into its observation by COMPARING the two sides.
///
/// This is what makes `cut_continuity` a continuity test rather than another single-frame test.
/// The same short closed question is put to this take's frames and to the adjacent selected take's
/// frame, and the two answers are compared:
///
/// * either side unobserved -> `Unobserved` (there is nothing to compare, and a comparison nobody
///   could make must never read as agreement);
/// * the two sides landed on DIFFERENT declared values -> `Mismatch`, with both values named;
/// * the same value on both sides -> `Match`.
///
/// The sc-22714 smoke is why this exists: SH020 renders a visibly different workshop from SH010,
/// and the old question — "is this a woodworking workshop?", asked of each frame independently —
/// answered yes on both sides and reported a clean cut. Asking what is ON THE WALL and comparing
/// the two answers is a question a single frame cannot fake.
pub fn aggregate_cut_observation(
    question: &ReviewQuestion,
    own: Vec<FrameAnswer>,
    adjacent: FrameAnswer,
) -> Observation {
    let mut evidence: Vec<String> = own.iter().map(|a| a.frame_id.clone()).collect();
    evidence.push(adjacent.frame_id.clone());

    let decisive_own: Vec<&FrameAnswer> = own
        .iter()
        .filter(|a| a.verdict != Verdict::Unobserved)
        .collect();
    let (verdict, observed, confidence) =
        if adjacent.verdict == Verdict::Unobserved || decisive_own.is_empty() {
            (Verdict::Unobserved, None, 0.0)
        } else {
            let theirs = adjacent.matched.clone().unwrap_or_default();
            let disagreeing = decisive_own.iter().find(|a| {
                a.matched.as_deref().unwrap_or_default() != theirs || a.verdict != adjacent.verdict
            });
            match disagreeing {
                Some(ours) => (
                    Verdict::Mismatch,
                    Some(format!(
                        "this take reads {:?}, the take it cuts from reads {:?}",
                        ours.matched.clone().unwrap_or_default(),
                        theirs
                    )),
                    ours.confidence.min(adjacent.confidence),
                ),
                None => (
                    Verdict::Match,
                    Some(format!("both takes read {theirs:?}")),
                    decisive_own
                        .iter()
                        .map(|a| a.confidence)
                        .fold(adjacent.confidence, f64::min),
                ),
            }
        };

    let mut answers = own;
    answers.push(adjacent);
    Observation {
        question_id: question.id.clone(),
        topic: question.topic.clone(),
        question: question.ask.clone(),
        intended: question.intended.clone(),
        frames: format!("{} + adjacent take", question.frames.as_str()),
        verdict,
        unobserved: verdict == Verdict::Unobserved,
        observed: if verdict == Verdict::Unobserved {
            None
        } else {
            observed
        },
        confidence,
        evidence_frame_ids: evidence,
        answers,
        note: None,
    }
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
        note: None,
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
    /// Frames of the take this one cuts FROM. An `acrossCut` question compares this take's answer
    /// against the last of these, which is the only way a cut-continuity question can be scored at
    /// all — a labeled case is one take, and comparing it against its own last frame would measure
    /// something nobody asked about. A case without them simply does not score its cut question.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adjacent_frames: Vec<EvalFrame>,
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

/// Why a labeled case's frame path may not be used, or `None` when it is a plain relative name
/// under the set's media root.
///
/// Checked here rather than at the read, so a bad document is refused before anything opens a file
/// — and so `review-fixtures`, which WRITES one file per named frame, is refused by the same rule.
pub fn unsafe_media_path(file: &str) -> Option<&'static str> {
    let trimmed = file.trim();
    if trimmed.is_empty() {
        return Some("is empty");
    }
    if trimmed.starts_with('/') || trimmed.starts_with('~') || trimmed.starts_with('\\') {
        return Some("must be relative to the set's mediaRoot, not an absolute or ~ path");
    }
    // Windows drive letters (`C:\frames\x.png`) are absolute too, and `Path::is_absolute` says so
    // only on Windows.
    if trimmed
        .chars()
        .nth(1)
        .is_some_and(|character| character == ':')
    {
        return Some("must be relative to the set's mediaRoot, not a drive-qualified path");
    }
    if trimmed
        .split(['/', '\\'])
        .any(|component| component == ".." || component == "~")
    {
        return Some("must not climb out of the set's mediaRoot with a `..` component");
    }
    None
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
        // A frame file is a name UNDER the set's media root, never a way out of it: the evaluation
        // both reads these paths and (through `review-fixtures`) WRITES them, so a document that
        // could name `../../etc/x` or an absolute path would make a labels file a write primitive.
        for (field, frames) in [
            ("frames", &case.frames),
            ("adjacentFrames", &case.adjacent_frames),
        ] {
            for (index, frame) in frames.iter().enumerate() {
                if let Some(reason) = unsafe_media_path(&frame.file) {
                    findings.push(PlanDiagnostic::plan(
                        format!("cases.{}.{field}[{index}].file", case.id),
                        format!("{:?} {reason}", frame.file),
                    ));
                }
            }
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
        let grade = grade_answer(question, text);
        FrameAnswer {
            frame_id: frame.to_owned(),
            answer: text.to_owned(),
            verdict: grade.verdict,
            matched: grade.matched,
            polarity: grade.polarity,
            confidence: grade.confidence,
            hedged: grade.hedged,
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
            let grade = grade_answer(&q, text);
            assert_eq!(grade.verdict, Verdict::Unobserved, "{text:?}");
            assert!(grade.matched.is_none(), "{text:?} produced a value");
            assert_eq!(grade.polarity, "none");
            assert_eq!(grade.confidence, 0.0);
        }
    }

    #[test]
    fn a_hedge_keeps_the_answer_but_halves_its_confidence() {
        let q = question("parcel_colour");
        let grade = grade_answer(&q, "It appears to be blue, though the light is warm.");
        assert_eq!(grade.verdict, Verdict::Mismatch);
        assert_eq!(grade.matched.as_deref(), Some("blue"));
        assert_eq!(grade.polarity, "affirmed");
        assert!(grade.hedged);
        assert!(grade.confidence < DEFAULT_UNCERTAIN_BELOW, "{grade:?}");
    }

    #[test]
    fn an_answer_matching_nothing_is_unobserved_not_a_match() {
        let q = question("parcel_colour");
        let grade = grade_answer(&q, "There is a wooden bench and sawdust.");
        assert_eq!(grade.verdict, Verdict::Unobserved);
        assert!(grade.matched.is_none());
    }

    // ------------------------------------------------------------------------------------------
    // The three mis-grades the sc-22714 real-weights smoke produced, pinned to the EXACT answer
    // strings SenseNova-U1-8B returned (evidence:
    // ~/SceneWorks/film-harness-evidence/sc-22714/smoke-20260914T001144Z/review-eval-real/).
    // Every one of them was a grader fault, not a model fault.
    // ------------------------------------------------------------------------------------------

    /// FACE (a): the reported "overclaim" was the GRADER's. On the flat conditioning plate the
    /// model answered correctly and decisively that the room is NOT a workshop; substring matching
    /// found the bare word "workshop" inside the negated clause and scored a Match.
    #[test]
    fn a_negated_expect_word_is_a_contradiction_not_a_match() {
        let mut q = question("sh020_location");
        q.topic = "location".to_owned();
        q.intended = "A cluttered woodworking workshop.".to_owned();
        q.expect = vec!["workshop".to_owned()];
        q.contradict = vec!["kitchen".to_owned(), "studio".to_owned()];
        let grade = grade_answer(
            &q,
            "No, the room in this frame is not a cluttered woodworking workshop with a long \
             wooden workbench. The room appears to be a plain, empty space with a uniform color, \
             possibly a studio or a minimalist room.",
        );
        assert_eq!(
            grade.verdict,
            Verdict::Mismatch,
            "a correct, decisive denial must not score a match: {grade:?}"
        );
        assert_eq!(grade.matched.as_deref(), Some("workshop"));
        assert_eq!(grade.polarity, "negated");

        // ... and the affirmative form of the same sentence still matches.
        let grade = grade_answer(
            &q,
            "Yes, the room in this frame is a cluttered woodworking workshop with a long wooden \
             workbench. The room you actually see is a workshop.",
        );
        assert_eq!(grade.verdict, Verdict::Match);
        assert_eq!(grade.polarity, "affirmed");
        assert_eq!(
            grade.confidence, CONFIDENT,
            "the hedge in the LATER clause must not weaken this one"
        );
    }

    /// FACE (b): a real model error was laundered into silence. The model said the jacket is red
    /// (it is navy); the token was the phrase "red jacket", and the answer put the colour AFTER
    /// the noun, so nothing matched and the wrong answer read as "could not tell".
    #[test]
    fn a_colour_named_after_the_noun_still_matches_a_single_word_token() {
        let mut q = question("sh010_courier_jacket");
        q.topic = "costume".to_owned();
        q.intended = "The courier wears a blue jacket.".to_owned();
        q.expect = vec!["blue".to_owned(), "navy".to_owned()];
        q.contradict = vec!["red".to_owned(), "green".to_owned(), "grey".to_owned()];
        let grade = grade_answer(&q, "The jacket of the person in the doorway is red.");
        assert_eq!(
            grade.verdict,
            Verdict::Mismatch,
            "a wrong colour must be reported as a fault, not as an abstention: {grade:?}"
        );
        assert_eq!(grade.matched.as_deref(), Some("red"));
        assert_eq!(grade.polarity, "affirmed");

        // Word-indexed, so a colour word inside another word is not a hit.
        let grade = grade_answer(&q, "The jacket is covered in sawdust.");
        assert_eq!(
            grade.verdict,
            Verdict::Unobserved,
            "\"red\" inside \"covered\" is not a colour: {grade:?}"
        );
    }

    /// FACE (c): a correct concise answer scored unobserved and raised a spurious `mustObserve`
    /// flag on a CORRECT take, because the declared tokens were long phrases the model did not use.
    #[test]
    fn a_concise_correct_answer_matches_a_single_word_token() {
        let mut q = question("sh010_parcel_custody");
        q.topic = "parcel_custody".to_owned();
        q.intended = "The parcel is in the courier's hands.".to_owned();
        q.must_observe = true;
        q.frames = FrameScope::Last;
        q.expect = vec!["hands".to_owned()];
        q.contradict = vec![
            "surface".to_owned(),
            "bench".to_owned(),
            "nobody".to_owned(),
        ];
        let grade = grade_answer(&q, "person's hands");
        assert_eq!(grade.verdict, Verdict::Match, "{grade:?}");
        assert_eq!(grade.matched.as_deref(), Some("hands"));
        let observation = aggregate_observation(&q, vec![answer("f3", "person's hands", &q)]);
        assert!(
            flag_for("SH010", &q, &observation, DEFAULT_UNCERTAIN_BELOW).is_none(),
            "a correct take must not be flagged because the grader could not read the answer"
        );
    }

    /// FACE (c), second form: the missing parcel WAS correctly reported by the model and the
    /// grader lost it, because "no parcel" was matched as a substring and the model said
    /// "no small parcel or box".
    #[test]
    fn a_negating_token_tolerates_intervening_modifiers_and_is_taken_at_face_value() {
        let mut q = question("sh020_parcel");
        q.expect = vec!["red".to_owned()];
        q.contradict = vec![
            "no parcel".to_owned(),
            "no box".to_owned(),
            "none".to_owned(),
        ];
        let grade = grade_answer(
            &q,
            "No, there is no small parcel or box in this frame. The image appears to be a plain, \
             dark brown surface with no visible objects.",
        );
        assert_eq!(grade.verdict, Verdict::Mismatch, "{grade:?}");
        assert_eq!(grade.matched.as_deref(), Some("no parcel"));
        assert_eq!(
            grade.polarity, "affirmed",
            "a token that carries its own negation must not be cancelled by the leading \"No,\""
        );
    }

    /// The shipped `location` question, whose vocabulary the leading-particle tests use.
    fn location_question() -> ReviewQuestion {
        let mut q = question("sh010_location");
        q.topic = "location".to_owned();
        q.intended = "A cluttered woodworking workshop with a long wooden workbench.".to_owned();
        q.ask = "What kind of room is in this frame? Answer with exactly one word from this list: \
                 workshop, kitchen, office, bedroom, outdoors, studio, or unclear."
            .to_owned();
        q.expect = vec!["workshop".to_owned()];
        q.contradict = ["kitchen", "office", "bedroom", "outdoors", "studio"]
            .iter()
            .map(|token| (*token).to_owned())
            .collect();
        q
    }

    /// A leading "No," / "Yes," is the ANSWER to a closed question, not a negation scope for every
    /// token after it. Read as a negation, the most common negative shape these questions get —
    /// "No, this is a kitchen." — dropped its affirmed `contradict` hit under rule 4 and abstained,
    /// which depressed detections on exactly the axis the evaluation reports.
    #[test]
    fn a_leading_answer_particle_is_the_answer_not_a_negation_scope() {
        let grade = grade_answer(&location_question(), "No, this is a kitchen.");
        assert_eq!(
            grade.verdict,
            Verdict::Mismatch,
            "a decisive wrong room must be flagged, not abstained on: {grade:?}"
        );
        assert_eq!(grade.matched.as_deref(), Some("kitchen"));
        assert_eq!(grade.polarity, "affirmed");
        assert_eq!(grade.confidence, CONFIDENT);

        // The same shape on the custody and colour questions, which is where it cost detections.
        let mut custody = question("sh040_parcel_custody");
        custody.topic = "parcel_custody".to_owned();
        custody.ask = "Where is the parcel in this frame? Answer with exactly one word: hands, \
                       surface, nobody, or unclear."
            .to_owned();
        custody.expect = vec!["surface".to_owned(), "bench".to_owned()];
        custody.contradict = vec!["hands".to_owned(), "holding".to_owned()];
        let grade = grade_answer(&custody, "No - the parcel is in the courier's hands.");
        assert_eq!(grade.verdict, Verdict::Mismatch, "{grade:?}");
        assert_eq!(grade.matched.as_deref(), Some("hands"));

        let grade = grade_answer(&question("parcel_colour"), "Yes: the parcel is blue.");
        assert_eq!(grade.verdict, Verdict::Mismatch, "{grade:?}");
        assert_eq!(grade.matched.as_deref(), Some("blue"));

        // A particle that is NOT punctuated off is an ordinary negation and still negates.
        let grade = grade_answer(&location_question(), "No kitchen is visible here.");
        assert_eq!(
            grade.verdict,
            Verdict::Unobserved,
            "\"no kitchen\" rules a value out without establishing another: {grade:?}"
        );
    }

    /// An answer that affirms a value from BOTH lists is self-contradictory. Letting word order
    /// decide reported a confident 0.9 verdict on half the sentence.
    #[test]
    fn an_answer_that_affirms_both_lists_claims_nothing() {
        let q = question("parcel_colour"); // expect red, contradict blue / no parcel
        let mut brown = q.clone();
        brown.contradict.push("brown".to_owned());
        for text in ["The parcel is red-brown.", "Red? No - blue."] {
            let grade = grade_answer(if text.contains("brown") { &brown } else { &q }, text);
            assert_eq!(
                grade.verdict,
                Verdict::Unobserved,
                "{text:?} agrees and contradicts at once: {grade:?}"
            );
            assert_eq!(grade.confidence, 0.0, "{text:?}");
            let matched = grade
                .matched
                .as_deref()
                .unwrap_or_else(|| panic!("{text:?} must name the conflict"));
            assert!(matched.starts_with("conflict:"), "{text:?}: {matched}");
            assert!(matched.contains("red"), "{text:?}: {matched}");
        }

        // An answer that only agrees still agrees, at full confidence.
        let grade = grade_answer(&q, "The parcel is red.");
        assert_eq!(grade.verdict, Verdict::Match);
        assert_eq!(grade.confidence, CONFIDENT);
    }

    /// These models lead with the question and then answer it. Graded, the restatement decides the
    /// verdict on the QUESTION's own wording: the echo below scored `Match(open)` on the shot that
    /// expects an open door and a confident `Mismatch(open)` on the shot that expects a closed one,
    /// neither of which the model said.
    #[test]
    fn a_leading_restatement_of_the_question_is_stripped_before_grading() {
        let echoed = "Is the door in this frame open or closed? It is closed.";
        let mut opens = question("sh010_action");
        opens.topic = "action_completion".to_owned();
        opens.ask =
            "Is the door in this frame open or closed? Answer with exactly one word: open, \
                     closed, or unclear."
                .to_owned();
        opens.expect = vec!["open".to_owned()];
        opens.contradict = vec!["closed".to_owned(), "shut".to_owned()];
        let grade = grade_answer(&opens, echoed);
        assert_eq!(
            grade.verdict,
            Verdict::Mismatch,
            "the model said closed, and the shot wants it open: {grade:?}"
        );
        assert_eq!(grade.matched.as_deref(), Some("closed"));

        // The same echo on the shot that WANTS the door closed is agreement, not a false alarm.
        let mut closes = opens.clone();
        closes.id = "sh040_action".to_owned();
        closes.expect = vec!["closed".to_owned(), "shut".to_owned()];
        closes.contradict = vec!["open".to_owned()];
        let grade = grade_answer(&closes, echoed);
        assert_eq!(
            grade.verdict,
            Verdict::Match,
            "a confident false alarm on a correct take: {grade:?}"
        );
        assert_eq!(grade.matched.as_deref(), Some("closed"));

        // An answer that is nothing BUT the question claims nothing at all.
        let grade = grade_answer(&opens, "Is the door in this frame open or closed?");
        assert_eq!(grade.verdict, Verdict::Unobserved, "{grade:?}");
        assert!(grade.matched.is_none());
    }

    /// A question the reviewer could not put at all is an observation NAMING why, not an absence.
    #[test]
    fn an_unasked_question_is_recorded_unobserved_with_its_reason() {
        let mut q = question("sh020_cut");
        q.topic = "cut_continuity".to_owned();
        q.across_cut = true;
        let observation = unasked_observation(&q, "no adjacent selected take to compare against");
        assert_eq!(observation.verdict, Verdict::Unobserved);
        assert!(observation.unobserved);
        assert!(observation.observed.is_none());
        assert_eq!(observation.confidence, 0.0);
        assert_eq!(
            observation.note.as_deref(),
            Some("no adjacent selected take to compare against")
        );
        assert!(observation.is_well_formed(), "{observation:?}");
        let json = serde_json::to_value(&observation).expect("serializes");
        assert!(json.get("observed").is_none(), "{json}");
        assert_eq!(
            json["note"],
            json!("no adjacent selected take to compare against")
        );
    }

    /// The "unobserved carries no value" rule is a rule about VALUES, so the type can express its
    /// own violation. Every violation must be NAMED, because that name is what the reader of a
    /// refused document gets.
    #[test]
    fn a_malformed_observation_names_what_is_wrong_with_it() {
        let q = question("parcel_colour");
        let good = aggregate_observation(&q, vec![answer("f1", "The parcel is red.", &q)]);
        assert_eq!(good.well_formed_error(), None, "{good:?}");

        let mut valued = good.clone();
        valued.verdict = Verdict::Unobserved;
        valued.unobserved = true;
        let error = valued
            .well_formed_error()
            .expect("an unobserved value is malformed");
        assert!(
            error.contains("no such thing as an unobserved value"),
            "{error}"
        );

        let mut confident = good.clone();
        confident.verdict = Verdict::Unobserved;
        confident.unobserved = true;
        confident.observed = None;
        let error = confident
            .well_formed_error()
            .expect("an unobserved observation cannot be confident");
        assert!(error.contains("confidence"), "{error}");

        let mut silent = good.clone();
        silent.observed = None;
        let error = silent
            .well_formed_error()
            .expect("a verdict with no value is malformed");
        assert!(error.contains("names no observed value"), "{error}");

        let mut mislabeled = good;
        mislabeled.verdict = Verdict::Unobserved;
        let error = mislabeled
            .well_formed_error()
            .expect("an unobserved verdict must be marked unobserved");
        assert!(error.contains("not marked unobserved"), "{error}");
    }

    /// A contradiction token inside a negated clause rules one value out without establishing
    /// another, so it decides nothing — inventing agreement from it would be the overclaim this
    /// module exists to prevent.
    #[test]
    fn a_ruled_out_value_establishes_nothing() {
        let mut q = question("jacket");
        q.expect = vec!["blue".to_owned()];
        q.contradict = vec!["red".to_owned()];
        let grade = grade_answer(&q, "The jacket is not red.");
        assert_eq!(grade.verdict, Verdict::Unobserved, "{grade:?}");
        assert!(grade.matched.is_none());
    }

    #[test]
    fn an_across_cut_question_compares_the_two_sides_rather_than_aggregating_them() {
        let mut q = question("sh020_cut");
        q.topic = "cut_continuity".to_owned();
        q.intended = "The same room, with the same pegboard behind the bench.".to_owned();
        q.frames = FrameScope::Last;
        q.across_cut = true;
        q.expect = vec!["yes".to_owned()];
        q.contradict = vec!["no".to_owned()];

        // Both sides agree -> the cut holds.
        let observation = aggregate_cut_observation(
            &q,
            vec![answer("SH020-f3", "Yes", &q)],
            answer("SH010-cut", "Yes", &q),
        );
        assert_eq!(observation.verdict, Verdict::Match);

        // The two sides disagree -> a jump, which is the case the old per-frame question missed:
        // both frames were "a woodworking workshop", but they were DIFFERENT workshops.
        let observation = aggregate_cut_observation(
            &q,
            vec![answer("SH020-f3", "No", &q)],
            answer("SH010-cut", "Yes", &q),
        );
        assert_eq!(observation.verdict, Verdict::Mismatch);
        let observed = observation.observed.as_deref().expect("a named difference");
        assert!(observed.contains("this take reads"), "{observed}");
        assert!(observed.contains("cuts from"), "{observed}");

        // Either side unreadable -> unobserved. A comparison nobody could make is never agreement.
        let observation = aggregate_cut_observation(
            &q,
            vec![answer("SH020-f3", "Yes", &q)],
            answer("SH010-cut", "The image is too dark to tell.", &q),
        );
        assert_eq!(observation.verdict, Verdict::Unobserved);
        assert!(observation.observed.is_none());
        assert!(observation.is_well_formed());
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

    /// A labeled set's frame paths are names UNDER its media root. `review-eval` reads them and
    /// `review-fixtures` WRITES them, so a document that could climb out is a write primitive.
    #[test]
    fn a_labeled_set_that_climbs_out_of_its_media_root_is_refused() {
        let review = review_doc(
            "SH010",
            json!({ "id": "q", "topic": "location", "intended": "i", "ask": "a", "expect": ["x"] }),
        );
        let set: EvalSet = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "s", "version": 1, "reviewPlan": "review.jsonc",
            "cases": [
                {
                    "id": "escapes", "shotId": "SH010", "label": "wrong_location",
                    "frames": [{ "file": "../../../etc/passwd", "timestampSeconds": 0.0 }],
                    "adjacentFrames": [{ "file": "/etc/hosts", "timestampSeconds": 0.0 }],
                    "expected": { "q": "mismatch" }
                },
                {
                    "id": "tilde", "shotId": "SH010", "label": "correct",
                    "frames": [{ "file": "~/secrets/a.png", "timestampSeconds": 0.0 }],
                    "expected": { "q": "match" }
                }
            ]
        }))
        .expect("set parses");
        let findings = validate_eval_set(&set, &review);
        let messages: Vec<String> = findings.iter().map(|f| f.to_string()).collect();
        let joined = messages.join("\n");
        assert!(joined.contains("cases.escapes.frames[0].file"), "{joined}");
        assert!(joined.contains("`..` component"), "{joined}");
        assert!(
            joined.contains("cases.escapes.adjacentFrames[0].file"),
            "an adjacent frame is read the same way: {joined}"
        );
        assert!(joined.contains("cases.tilde.frames[0].file"), "{joined}");

        // ...and a plain relative name in a subdirectory is fine.
        assert_eq!(unsafe_media_path("frames/sh010_1.png"), None);
        assert_eq!(unsafe_media_path("sh010_1.png"), None);
    }

    /// Every bound the review runs under is declared in the document, `maxNewTokens` and
    /// `maxMemoryGb` included — a bound that lives in a `const` is not a declared bound.
    #[test]
    fn the_review_limits_declare_the_token_and_memory_ceilings() {
        let plan = plan_with_shots(&["SH010"]);
        let review = review_doc(
            "SH010",
            json!({ "id": "q", "topic": "location", "intended": "i", "ask": "a", "expect": ["x"] }),
        );
        // A document that names neither keeps the shipped defaults rather than failing to parse.
        assert_eq!(review.limits.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(review.limits.max_memory_gb, DEFAULT_MAX_MEMORY_GB);
        assert!(validate_review_plan(&review, &plan).is_empty());

        let declared: ReviewPlan = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "r", "version": 1,
            "sampling": { "positions": [0.5] },
            "limits": { "maxSeconds": 60, "maxFramesPerShot": 1, "maxQuestionsPerShot": 1,
                        "maxAnswerSeconds": 10, "maxNewTokens": 64, "maxMemoryGb": 12.5 },
            "shots": { "SH010": { "questions": [
                { "id": "q", "topic": "location", "intended": "i", "ask": "a", "expect": ["x"] }
            ] } }
        }))
        .expect("review plan parses");
        assert_eq!(declared.limits.max_new_tokens, 64);
        assert_eq!(declared.limits.max_memory_gb, 12.5);
        assert!(validate_review_plan(&declared, &plan).is_empty());

        let mut zero = declared.clone();
        zero.limits.max_new_tokens = 0;
        assert!(
            validate_review_plan(&zero, &plan)
                .iter()
                .any(|f| f.message.contains("maxNewTokens")),
            "an answer truncated to zero tokens answers nothing"
        );
        let mut negative = declared;
        negative.limits.max_memory_gb = 0.0;
        assert!(
            validate_review_plan(&negative, &plan)
                .iter()
                .any(|f| f.message.contains("memory ceiling")),
            "a review that declares no memory ceiling has not declared its bounds"
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
