//! Hand-authored production plans, reference packs, and run records for the local filmmaking
//! harness (epic 22708, sc-22710).
//!
//! Three documents, all JSON (JSONC comments tolerated on read):
//!
//! * a [`ProductionPlan`] — stable shot ids, narrative beats, framing, target timing, intended
//!   start/end state, dialogue/sound intent and the conditioning each shot wants, expressed as
//!   **reference roles** rather than asset ids so the plan stays addressable before anything is
//!   imported;
//! * a [`ReferencePack`] — the approved references, each with an explicit role, that the plan's
//!   roles resolve against. Kept as a separate versioned document so approved references remain
//!   addressable independently of any generated take;
//! * a [`RunRecord`] — what one execution of a plan actually did: the reference assets it imported,
//!   every dispatched attempt (shot -> job -> asset), the timeline it assembled and the export it
//!   produced, plus the model/backend/hardware that were observed.
//!
//! Validation is split so each layer can run before the next one costs anything:
//! [`validate_plan_structure`] and [`validate_reference_pack`] need only the documents,
//! [`validate_plan_against_pack`] needs both, [`validate_reference_pack_files`] needs the pack's
//! directory, and [`validate_plan_against_model`] needs the chosen model's manifest entry. Every
//! finding is a [`PlanDiagnostic`] naming the shot and the field, so a rejected plan is actionable
//! without reading code. Nothing here dispatches; the harness (rust-api `film_harness`) refuses
//! to create a single job while any diagnostic is outstanding.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::jsonc::strip_jsonc_comments;
use crate::video_request::{
    default_fps, default_resolution, duration_limit_error, fps_limit_error, reference_caps,
};

/// Schema version of [`ProductionPlan`] documents this module reads and writes. Version 2 (sc-22711)
/// adds `shots[].dependsOn`; a version 1 plan reads unchanged and simply declares no edges.
pub const PLAN_SCHEMA_VERSION: u32 = 2;
/// Plan schema versions this build accepts.
pub const SUPPORTED_PLAN_SCHEMA_VERSIONS: &[u32] = &[1, 2];
/// Schema version of [`ReferencePack`] documents this module reads and writes.
pub const REFERENCE_PACK_SCHEMA_VERSION: u32 = 1;
/// Schema version of [`RunRecord`] documents this module writes. Version 2 (sc-22711) adds the
/// durable resume state: `state`, `stop`, per-attempt idempotency keys and rejections, per-shot
/// take selection and review flags, and the human decision log.
///
/// sc-22714 adds `shots[].reviews` and `shots[].humanDecision` **within version 2**, deliberately:
/// both are optional, defaulted and skipped when empty, so a record written before them still
/// reads and a record written with them still resumes on a build without them. Bumping the version
/// would refuse in-flight runs for a purely additive index. The observed state they point at is a
/// separate document with its own version
/// ([`crate::film_review::OBSERVED_STATE_SCHEMA_VERSION`]).
pub const RUN_RECORD_SCHEMA_VERSION: u32 = 2;

/// Conditioning modes a shot may declare. Each mode fixes which reference slots it takes, which
/// is what keeps a plan honest about model-specific constraints: a first/last-frame model pins
/// literal frames and a reference model conditions on unpositioned media, and no shipped video
/// model accepts both on one request.
pub const SHOT_CONDITIONING_MODES: &[&str] = &[
    "text_to_video",
    "image_to_video",
    "first_last_frame",
    "reference_to_video",
];

/// Reference kinds a pack entry may declare.
pub const REFERENCE_KINDS: &[&str] = &["character", "prop", "location", "style", "plate"];

/// Dependency kinds one shot may declare on another (sc-22711).
///
/// * `conditioning` — this shot's conditioning is derived from the other shot's **selected take**
///   (last-frame chaining, a plate cut from a take). Re-selecting the other shot's take changes
///   what this shot was conditioned on.
/// * `continuity` — this shot's `startState` is the other shot's `endState`: the take is still
///   valid input-wise, but the story state it continues from may no longer match.
///
/// Either way a changed selection downstream is a **review** signal, never an automatic
/// regeneration: the harness flags the dependent and stops there.
pub const SHOT_DEPENDENCY_KINDS: &[&str] = &["conditioning", "continuity"];

/// Image extensions a reference file may carry; the import route accepts any `image/*` but the
/// pack is checked in, so the list stays explicit.
const REFERENCE_IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

/// Longest prompt the video route accepts (`MAX_PROMPT_CHARS` in rust-api).
const MAX_PROMPT_CHARS: usize = 4000;

/// Tolerance when matching a target duration against a model's declared menu (seconds).
const DURATION_MENU_TOLERANCE: f64 = 0.001;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProductionPlan {
    pub schema_version: u32,
    pub id: String,
    pub version: u32,
    pub title: String,
    #[serde(default)]
    pub synopsis: String,
    pub model: PlanModel,
    pub limits: PlanLimits,
    pub shots: Vec<Shot>,
}

/// The one local model/backend the plan renders through. `tier` is the quantization tier to
/// request (`q4` / `q8` / `bf16`; the worker's tier order decides what actually loads and the run
/// record captures which one did). `fps` and `resolution` are plan-wide defaults resolved from the
/// model's manifest `defaults` when omitted; a shot may override `resolution`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlanModel {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
}

/// Finite limits declared BEFORE dispatch. Exceeding any of them stops new dispatch and leaves the
/// reason in the run record; none of them may be zero or non-finite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlanLimits {
    /// Wall-clock budget for the whole run, including the export.
    pub max_run_seconds: u64,
    /// Wall-clock budget for one attempt of one shot (and for the export job).
    pub max_shot_seconds: u64,
    /// Attempts per shot, counting the first. `1` means no retry.
    pub max_attempts_per_shot: u32,
    /// Memory the run is allowed to use, in GB. Checked against the model's declared minimum
    /// before dispatch and against the observed peak after every attempt.
    pub max_memory_gb: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Shot {
    /// Stable id (`[A-Za-z0-9_-]{1,64}`), unique within the plan. Later stories key resume,
    /// take replacement and review on it.
    pub id: String,
    pub beat: String,
    pub framing: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    pub target_duration_seconds: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    pub start_state: String,
    pub end_state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialogue: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sound: Option<String>,
    pub conditioning: ShotConditioning,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Roles the shot depicts (traceability only; they are not sent to the model). Every entry
    /// must exist in the pack.
    #[serde(default)]
    pub continuity_roles: Vec<String>,
    /// Shots this one depends on (sc-22711). Declaring the edge is what lets the harness flag this
    /// shot for review when the shot it depends on gets a different selected take; nothing here is
    /// sent to the model and nothing is ever regenerated automatically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<ShotDependency>,
}

/// One declared edge from a shot to a shot it depends on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ShotDependency {
    /// Id of the shot depended on. Must be another shot in the same plan.
    pub shot_id: String,
    /// One of [`SHOT_DEPENDENCY_KINDS`].
    pub kind: String,
    /// Why the edge exists, shown verbatim on the review flag it raises.
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ShotConditioning {
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame_role: Option<String>,
    #[serde(default)]
    pub reference_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferencePack {
    pub schema_version: u32,
    pub id: String,
    pub version: u32,
    #[serde(default)]
    pub description: String,
    pub references: Vec<ReferenceEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceEntry {
    /// Role name (`[A-Za-z0-9_-]{1,64}`), unique within the pack.
    pub role: String,
    pub kind: String,
    /// Image path relative to the pack document's directory.
    pub file: String,
    #[serde(default)]
    pub description: String,
    /// Only approved references may be used as conditioning.
    #[serde(default = "default_true")]
    pub approved: bool,
}

fn default_true() -> bool {
    true
}

/// One actionable finding. `shot_id` is `None` for plan-level and pack-level findings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanDiagnostic {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shot_id: Option<String>,
    pub field: String,
    pub message: String,
}

impl PlanDiagnostic {
    pub fn plan(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            shot_id: None,
            field: field.into(),
            message: message.into(),
        }
    }

    pub fn shot(
        shot_id: impl Into<String>,
        field: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            shot_id: Some(shot_id.into()),
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for PlanDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.shot_id {
            Some(shot) => write!(f, "[{shot}] {}: {}", self.field, self.message),
            None => write!(f, "[plan] {}: {}", self.field, self.message),
        }
    }
}

/// Read and parse a plan document (JSONC tolerated). A parse failure is returned as a diagnostic
/// so the harness can print it beside structural findings.
pub fn read_plan_file(path: &Path) -> Result<ProductionPlan, PlanDiagnostic> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        PlanDiagnostic::plan("plan", format!("cannot read {}: {error}", path.display()))
    })?;
    parse_plan(&text)
        .map_err(|error| PlanDiagnostic::plan("plan", format!("{}: {error}", path.display())))
}

/// Parse a plan from JSON/JSONC text.
pub fn parse_plan(text: &str) -> Result<ProductionPlan, String> {
    let stripped = strip_jsonc_comments(text);
    serde_json::from_str(&stripped).map_err(|error| error.to_string())
}

/// Read and parse a reference pack document (JSONC tolerated).
pub fn read_reference_pack_file(path: &Path) -> Result<ReferencePack, PlanDiagnostic> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        PlanDiagnostic::plan(
            "referencePack",
            format!("cannot read {}: {error}", path.display()),
        )
    })?;
    parse_reference_pack(&text).map_err(|error| {
        PlanDiagnostic::plan("referencePack", format!("{}: {error}", path.display()))
    })
}

/// Parse a reference pack from JSON/JSONC text.
pub fn parse_reference_pack(text: &str) -> Result<ReferencePack, String> {
    let stripped = strip_jsonc_comments(text);
    serde_json::from_str(&stripped).map_err(|error| error.to_string())
}

/// `[A-Za-z0-9_-]{1,64}` — the charset every id and role in these documents must satisfy, so an
/// id can be embedded in file names, tags and job payloads without escaping.
pub fn is_safe_plan_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// Parse `"WxH"` into `(width, height)`.
pub fn parse_resolution(value: &str) -> Option<(u32, u32)> {
    let (width, height) = value.trim().split_once('x')?;
    let width = width.trim().parse::<u32>().ok()?;
    let height = height.trim().parse::<u32>().ok()?;
    (width > 0 && height > 0).then_some((width, height))
}

/// Structural findings on the plan alone: schema version, ids, limits, per-shot fields and the
/// slot shape each conditioning mode demands.
pub fn validate_plan_structure(plan: &ProductionPlan) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if !SUPPORTED_PLAN_SCHEMA_VERSIONS.contains(&plan.schema_version) {
        findings.push(PlanDiagnostic::plan(
            "schemaVersion",
            format!(
                "unsupported plan schema version {} (this build reads {SUPPORTED_PLAN_SCHEMA_VERSIONS:?})",
                plan.schema_version
            ),
        ));
    }
    if !is_safe_plan_id(&plan.id) {
        findings.push(PlanDiagnostic::plan(
            "id",
            "plan id must be 1-64 characters of [A-Za-z0-9_-]",
        ));
    }
    if plan.version == 0 {
        findings.push(PlanDiagnostic::plan("version", "plan version must be >= 1"));
    }
    if plan.title.trim().is_empty() {
        findings.push(PlanDiagnostic::plan("title", "plan title is required"));
    }
    if plan.model.id.trim().is_empty() {
        findings.push(PlanDiagnostic::plan("model.id", "model id is required"));
    }
    if let Some(tier) = plan.model.tier.as_deref() {
        if !matches!(tier, "q4" | "q8" | "bf16") {
            findings.push(PlanDiagnostic::plan(
                "model.tier",
                format!("unknown tier {tier:?}; expected one of q4, q8, bf16"),
            ));
        }
    }
    if let Some(fps) = plan.model.fps {
        if !(1..=60).contains(&fps) {
            findings.push(PlanDiagnostic::plan(
                "model.fps",
                format!("fps {fps} is outside 1..=60"),
            ));
        }
    }
    if let Some(resolution) = plan.model.resolution.as_deref() {
        if parse_resolution(resolution).is_none() {
            findings.push(PlanDiagnostic::plan(
                "model.resolution",
                format!("{resolution:?} is not a WxH resolution"),
            ));
        }
    }
    findings.extend(validate_limits(&plan.limits));
    if plan.shots.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "shots",
            "a plan needs at least one shot",
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, shot) in plan.shots.iter().enumerate() {
        if !is_safe_plan_id(&shot.id) {
            findings.push(PlanDiagnostic::plan(
                format!("shots[{index}].id"),
                format!(
                    "shot id {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                    shot.id
                ),
            ));
            continue;
        }
        if !seen.insert(shot.id.as_str()) {
            findings.push(PlanDiagnostic::shot(
                &shot.id,
                "id",
                format!("duplicate shot id {:?}", shot.id),
            ));
        }
        findings.extend(validate_shot_structure(shot));
    }
    findings.extend(validate_dependencies(plan));
    findings
}

/// Dependency edges need the whole plan: every target must be another shot in it, an edge may not
/// be declared twice, and the graph may not contain a cycle (a cycle would make "flag everything
/// downstream of this shot" unbounded).
fn validate_dependencies(plan: &ProductionPlan) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let ids: BTreeSet<&str> = plan.shots.iter().map(|shot| shot.id.as_str()).collect();
    for shot in &plan.shots {
        let mut seen = BTreeSet::new();
        for edge in &shot.depends_on {
            if !SHOT_DEPENDENCY_KINDS.contains(&edge.kind.as_str()) {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "dependsOn.kind",
                    format!(
                        "unknown dependency kind {:?}; expected one of {}",
                        edge.kind,
                        SHOT_DEPENDENCY_KINDS.join(", ")
                    ),
                ));
            }
            if edge.shot_id == shot.id {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "dependsOn.shotId",
                    "a shot cannot depend on itself",
                ));
            } else if !ids.contains(edge.shot_id.as_str()) {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "dependsOn.shotId",
                    format!("{:?} is not a shot in this plan", edge.shot_id),
                ));
            }
            if !seen.insert(edge.shot_id.as_str()) {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "dependsOn.shotId",
                    format!("shot {:?} is depended on twice", edge.shot_id),
                ));
            }
        }
    }
    if findings.is_empty() {
        if let Some(cycle) = dependency_cycle(plan) {
            findings.push(PlanDiagnostic::plan(
                "shots.dependsOn",
                format!("dependency cycle: {}", cycle.join(" -> ")),
            ));
        }
    }
    findings
}

/// The first dependency cycle in `plan`, as the shot ids on it (closing back on the first), or
/// `None` when the graph is acyclic. Iterative depth-first search so a long chain cannot blow the
/// stack.
fn dependency_cycle(plan: &ProductionPlan) -> Option<Vec<String>> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Open,
        Done,
    }
    let edges: BTreeMap<&str, Vec<&str>> = plan
        .shots
        .iter()
        .map(|shot| {
            (
                shot.id.as_str(),
                shot.depends_on
                    .iter()
                    .map(|edge| edge.shot_id.as_str())
                    .collect(),
            )
        })
        .collect();
    let mut marks: BTreeMap<&str, Mark> = BTreeMap::new();
    for root in edges.keys().copied() {
        if marks.contains_key(root) {
            continue;
        }
        // (shot, index of the next edge to walk) — the stack IS the current path.
        let mut stack: Vec<(&str, usize)> = vec![(root, 0)];
        marks.insert(root, Mark::Open);
        while let Some((shot, index)) = stack.pop() {
            let Some(next) = edges.get(shot).and_then(|targets| targets.get(index)) else {
                marks.insert(shot, Mark::Done);
                continue;
            };
            stack.push((shot, index + 1));
            match marks.get(next) {
                Some(Mark::Done) => {}
                Some(Mark::Open) => {
                    let start = stack
                        .iter()
                        .position(|(id, _)| id == next)
                        .unwrap_or_default();
                    let mut cycle: Vec<String> = stack[start..]
                        .iter()
                        .map(|(id, _)| (*id).to_owned())
                        .collect();
                    cycle.push((*next).to_owned());
                    return Some(cycle);
                }
                None => {
                    marks.insert(next, Mark::Open);
                    stack.push((next, 0));
                }
            }
        }
    }
    None
}

/// Shots that declare a direct dependency on `shot_id`, in plan order. Direct only: a dependent's
/// own take did not change, so the change does not propagate past one hop on its own.
pub fn direct_dependents<'a>(
    plan: &'a ProductionPlan,
    shot_id: &str,
) -> Vec<(&'a Shot, &'a ShotDependency)> {
    plan.shots
        .iter()
        .filter_map(|shot| {
            shot.depends_on
                .iter()
                .find(|edge| edge.shot_id == shot_id)
                .map(|edge| (shot, edge))
        })
        .collect()
}

fn validate_limits(limits: &PlanLimits) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if limits.max_run_seconds == 0 {
        findings.push(PlanDiagnostic::plan(
            "limits.maxRunSeconds",
            "the run wall-clock budget must be > 0",
        ));
    }
    if limits.max_shot_seconds == 0 {
        findings.push(PlanDiagnostic::plan(
            "limits.maxShotSeconds",
            "the per-shot wall-clock budget must be > 0",
        ));
    }
    if limits.max_shot_seconds > limits.max_run_seconds {
        findings.push(PlanDiagnostic::plan(
            "limits.maxShotSeconds",
            format!(
                "the per-shot budget ({}s) exceeds the run budget ({}s)",
                limits.max_shot_seconds, limits.max_run_seconds
            ),
        ));
    }
    if limits.max_attempts_per_shot == 0 {
        findings.push(PlanDiagnostic::plan(
            "limits.maxAttemptsPerShot",
            "at least one attempt per shot is required",
        ));
    }
    if !limits.max_memory_gb.is_finite() || limits.max_memory_gb <= 0.0 {
        findings.push(PlanDiagnostic::plan(
            "limits.maxMemoryGb",
            "the memory budget must be a finite number > 0",
        ));
    }
    findings
}

fn validate_shot_structure(shot: &Shot) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let id = shot.id.as_str();
    for (field, value) in [
        ("beat", &shot.beat),
        ("framing", &shot.framing),
        ("startState", &shot.start_state),
        ("endState", &shot.end_state),
    ] {
        if value.trim().is_empty() {
            findings.push(PlanDiagnostic::shot(
                id,
                field,
                format!("{field} is required"),
            ));
        }
    }
    let prompt_chars = shot.prompt.chars().count();
    if shot.prompt.trim().is_empty() || prompt_chars > MAX_PROMPT_CHARS {
        findings.push(PlanDiagnostic::shot(
            id,
            "prompt",
            format!("prompt must be 1-{MAX_PROMPT_CHARS} characters (got {prompt_chars})"),
        ));
    }
    if !shot.target_duration_seconds.is_finite() || shot.target_duration_seconds <= 0.0 {
        findings.push(PlanDiagnostic::shot(
            id,
            "targetDurationSeconds",
            "target duration must be a finite number > 0",
        ));
    }
    if let Some(resolution) = shot.resolution.as_deref() {
        if parse_resolution(resolution).is_none() {
            findings.push(PlanDiagnostic::shot(
                id,
                "resolution",
                format!("{resolution:?} is not a WxH resolution"),
            ));
        }
    }
    let conditioning = &shot.conditioning;
    let mode = conditioning.mode.as_str();
    if !SHOT_CONDITIONING_MODES.contains(&mode) {
        findings.push(PlanDiagnostic::shot(
            id,
            "conditioning.mode",
            format!(
                "unknown conditioning mode {mode:?}; expected one of {}",
                SHOT_CONDITIONING_MODES.join(", ")
            ),
        ));
        return findings;
    }
    let wants_first = matches!(mode, "image_to_video" | "first_last_frame");
    let wants_last = mode == "first_last_frame";
    let wants_references = mode == "reference_to_video";
    match (wants_first, conditioning.first_frame_role.is_some()) {
        (true, false) => findings.push(PlanDiagnostic::shot(
            id,
            "conditioning.firstFrameRole",
            format!("mode {mode} requires a first-frame reference role"),
        )),
        (false, true) => findings.push(PlanDiagnostic::shot(
            id,
            "conditioning.firstFrameRole",
            format!("mode {mode} does not take a first-frame role"),
        )),
        _ => {}
    }
    match (wants_last, conditioning.last_frame_role.is_some()) {
        (true, false) => findings.push(PlanDiagnostic::shot(
            id,
            "conditioning.lastFrameRole",
            format!("mode {mode} requires a last-frame reference role"),
        )),
        (false, true) => findings.push(PlanDiagnostic::shot(
            id,
            "conditioning.lastFrameRole",
            format!("mode {mode} does not take a last-frame role"),
        )),
        _ => {}
    }
    match (wants_references, conditioning.reference_roles.is_empty()) {
        (true, true) => findings.push(PlanDiagnostic::shot(
            id,
            "conditioning.referenceRoles",
            format!("mode {mode} requires at least one reference role"),
        )),
        (false, false) => findings.push(PlanDiagnostic::shot(
            id,
            "conditioning.referenceRoles",
            format!(
                "mode {mode} does not take reference roles (keyframes and references are \
                 different conditioning tasks; use reference_to_video for references)"
            ),
        )),
        _ => {}
    }
    let mut seen = BTreeSet::new();
    for role in &conditioning.reference_roles {
        if !seen.insert(role.as_str()) {
            findings.push(PlanDiagnostic::shot(
                id,
                "conditioning.referenceRoles",
                format!("reference role {role:?} is listed twice"),
            ));
        }
    }
    findings
}

/// Structural findings on the reference pack alone.
pub fn validate_reference_pack(pack: &ReferencePack) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if pack.schema_version != REFERENCE_PACK_SCHEMA_VERSION {
        findings.push(PlanDiagnostic::plan(
            "referencePack.schemaVersion",
            format!(
                "unsupported reference pack schema version {} (this build reads \
                 {REFERENCE_PACK_SCHEMA_VERSION})",
                pack.schema_version
            ),
        ));
    }
    if !is_safe_plan_id(&pack.id) {
        findings.push(PlanDiagnostic::plan(
            "referencePack.id",
            "reference pack id must be 1-64 characters of [A-Za-z0-9_-]",
        ));
    }
    if pack.version == 0 {
        findings.push(PlanDiagnostic::plan(
            "referencePack.version",
            "reference pack version must be >= 1",
        ));
    }
    if pack.references.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "referencePack.references",
            "a reference pack needs at least one reference",
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, entry) in pack.references.iter().enumerate() {
        let field = format!("referencePack.references[{index}]");
        if !is_safe_plan_id(&entry.role) {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.role"),
                format!(
                    "role {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                    entry.role
                ),
            ));
        } else if !seen.insert(entry.role.as_str()) {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.role"),
                format!("duplicate reference role {:?}", entry.role),
            ));
        }
        if !REFERENCE_KINDS.contains(&entry.kind.as_str()) {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.kind"),
                format!(
                    "unknown reference kind {:?}; expected one of {}",
                    entry.kind,
                    REFERENCE_KINDS.join(", ")
                ),
            ));
        }
        let file = Path::new(&entry.file);
        let extension_ok = file
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                REFERENCE_IMAGE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str())
            });
        if entry.file.trim().is_empty()
            || file.is_absolute()
            || file
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.file"),
                format!(
                    "file {:?} must be a relative path inside the pack directory",
                    entry.file
                ),
            ));
        } else if !extension_ok {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.file"),
                format!(
                    "file {:?} must be an image ({})",
                    entry.file,
                    REFERENCE_IMAGE_EXTENSIONS.join(", ")
                ),
            ));
        }
    }
    findings
}

/// Findings that need both documents: every role a shot names must exist in the pack and be
/// approved.
pub fn validate_plan_against_pack(
    plan: &ProductionPlan,
    pack: &ReferencePack,
) -> Vec<PlanDiagnostic> {
    let roles: BTreeMap<&str, &ReferenceEntry> = pack
        .references
        .iter()
        .map(|entry| (entry.role.as_str(), entry))
        .collect();
    let mut findings = Vec::new();
    for shot in &plan.shots {
        let conditioning = &shot.conditioning;
        let mut slots: Vec<(&str, &str)> = Vec::new();
        if let Some(role) = conditioning.first_frame_role.as_deref() {
            slots.push(("conditioning.firstFrameRole", role));
        }
        if let Some(role) = conditioning.last_frame_role.as_deref() {
            slots.push(("conditioning.lastFrameRole", role));
        }
        for role in &conditioning.reference_roles {
            slots.push(("conditioning.referenceRoles", role.as_str()));
        }
        for role in &shot.continuity_roles {
            slots.push(("continuityRoles", role.as_str()));
        }
        for (field, role) in slots {
            match roles.get(role) {
                None => findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    field,
                    format!(
                        "reference role {role:?} is not in reference pack {:?} (roles: {})",
                        pack.id,
                        roles.keys().copied().collect::<Vec<_>>().join(", ")
                    ),
                )),
                Some(entry) if !entry.approved && field.starts_with("conditioning.") => {
                    findings.push(PlanDiagnostic::shot(
                        &shot.id,
                        field,
                        format!("reference role {role:?} is not approved for conditioning"),
                    ));
                }
                Some(_) => {}
            }
        }
    }
    findings
}

/// Findings that need the pack directory: every referenced file must exist and be non-empty.
pub fn validate_reference_pack_files(pack: &ReferencePack, pack_dir: &Path) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    for (index, entry) in pack.references.iter().enumerate() {
        let path = pack_dir.join(&entry.file);
        match std::fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {}
            Ok(_) => findings.push(PlanDiagnostic::plan(
                format!("referencePack.references[{index}].file"),
                format!(
                    "reference {:?}: {} is empty or not a file",
                    entry.role,
                    path.display()
                ),
            )),
            Err(error) => findings.push(PlanDiagnostic::plan(
                format!("referencePack.references[{index}].file"),
                format!(
                    "reference {:?}: {} is missing ({error})",
                    entry.role,
                    path.display()
                ),
            )),
        }
    }
    findings
}

/// The lane a manifest entry's memory minimum is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLane {
    Mlx,
    Candle,
}

impl ModelLane {
    /// The lane this host renders video on: MLX on macOS, candle everywhere else.
    pub fn for_current_platform() -> Self {
        if cfg!(target_os = "macos") {
            Self::Mlx
        } else {
            Self::Candle
        }
    }

    pub fn manifest_key(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Candle => "candle",
        }
    }
}

/// The model's declared memory minimum for `lane` (`<lane>.minMemoryGb`), if it declares one.
pub fn model_min_memory_gb(entry: &Map<String, Value>, lane: ModelLane) -> Option<f64> {
    entry
        .get(lane.manifest_key())
        .and_then(Value::as_object)
        .and_then(|block| block.get("minMemoryGb"))
        .and_then(Value::as_f64)
}

/// Frames per second the plan renders at: the plan's own `fps` or the model's declared default.
pub fn plan_fps(plan: &ProductionPlan, entry: &Map<String, Value>) -> Option<u32> {
    plan.model.fps.or_else(|| default_fps(entry))
}

/// Output geometry for `shot`: the shot's `resolution`, else the plan's, else the model's
/// declared default.
pub fn shot_resolution(
    plan: &ProductionPlan,
    shot: &Shot,
    entry: &Map<String, Value>,
) -> Option<(u32, u32)> {
    shot.resolution
        .as_deref()
        .or(plan.model.resolution.as_deref())
        .and_then(parse_resolution)
        .or_else(|| default_resolution(entry))
}

/// Findings that need the chosen model's manifest entry: capability per mode, duration/fps/
/// resolution against the declared menus and caps, reference counts against the declared caps,
/// negative prompts against `video.supportsNegativePrompt`, and the plan's memory budget against
/// the model's declared minimum on `lane`.
pub fn validate_plan_against_model(
    plan: &ProductionPlan,
    entry: &Map<String, Value>,
    lane: ModelLane,
) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let model_id = plan.model.id.as_str();
    if let Some(minimum) = model_min_memory_gb(entry, lane) {
        if plan.limits.max_memory_gb < minimum {
            findings.push(PlanDiagnostic::plan(
                "limits.maxMemoryGb",
                format!(
                    "budget {} GB is below {model_id}'s declared {}.minMemoryGb of {minimum} GB",
                    plan.limits.max_memory_gb,
                    lane.manifest_key()
                ),
            ));
        }
    }
    let Some(fps) = plan_fps(plan, entry) else {
        findings.push(PlanDiagnostic::plan(
            "model.fps",
            format!("{model_id} declares no default fps; set model.fps in the plan"),
        ));
        return findings;
    };
    if let Some(message) = fps_limit_error(model_id, fps, entry) {
        findings.push(PlanDiagnostic::plan("model.fps", message));
    }
    let capabilities: Vec<&str> = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let supports_negative = entry
        .get("video")
        .and_then(Value::as_object)
        .and_then(|video| video.get("supportsNegativePrompt"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let caps = reference_caps(entry);
    let limits = entry.get("limits").and_then(Value::as_object);
    let duration_menu: Vec<f64> = limits
        .and_then(|limits| limits.get("durations"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_f64)
        .collect();
    let resolution_menu: Vec<(u32, u32)> = limits
        .and_then(|limits| limits.get("resolutions"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(parse_resolution)
        .collect();
    let max_pixels = limits
        .and_then(|limits| limits.get("maxPixels"))
        .and_then(Value::as_u64);

    for shot in &plan.shots {
        let id = shot.id.as_str();
        let mode = shot.conditioning.mode.as_str();
        if !capabilities.contains(&mode) {
            findings.push(PlanDiagnostic::shot(
                id,
                "conditioning.mode",
                format!(
                    "{model_id} does not declare {mode} (declared: {})",
                    capabilities.join(", ")
                ),
            ));
        }
        let references = shot.conditioning.reference_roles.len();
        if references > caps.images {
            findings.push(PlanDiagnostic::shot(
                id,
                "conditioning.referenceRoles",
                format!(
                    "{references} reference roles exceed {model_id}'s limits.maxReferenceAssets \
                     of {}",
                    caps.images
                ),
            ));
        }
        if shot
            .negative_prompt
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
            && !supports_negative
        {
            findings.push(PlanDiagnostic::shot(
                id,
                "negativePrompt",
                format!(
                    "{model_id} has no negative prompt (video.supportsNegativePrompt is false)"
                ),
            ));
        }
        let duration = shot.target_duration_seconds;
        if let Some(message) = duration_limit_error(model_id, duration as f32, entry) {
            findings.push(PlanDiagnostic::shot(id, "targetDurationSeconds", message));
        } else if !duration_menu.is_empty()
            && !duration_menu
                .iter()
                .any(|candidate| (candidate - duration).abs() <= DURATION_MENU_TOLERANCE)
        {
            findings.push(PlanDiagnostic::shot(
                id,
                "targetDurationSeconds",
                format!(
                    "{duration}s is not on {model_id}'s duration menu {duration_menu:?}; the \
                     engine would render a different length than the plan intends"
                ),
            ));
        }
        match shot_resolution(plan, shot, entry) {
            None => findings.push(PlanDiagnostic::shot(
                id,
                "resolution",
                format!("{model_id} declares no default resolution; set resolution in the plan"),
            )),
            Some((width, height)) => {
                if !resolution_menu.is_empty() && !resolution_menu.contains(&(width, height)) {
                    let menu: Vec<String> = resolution_menu
                        .iter()
                        .map(|(w, h)| format!("{w}x{h}"))
                        .collect();
                    findings.push(PlanDiagnostic::shot(
                        id,
                        "resolution",
                        format!(
                            "{width}x{height} is not on {model_id}'s resolution menu [{}]",
                            menu.join(", ")
                        ),
                    ));
                } else if let Some(max_pixels) = max_pixels {
                    if u64::from(width) * u64::from(height) > max_pixels {
                        findings.push(PlanDiagnostic::shot(
                            id,
                            "resolution",
                            format!(
                                "{width}x{height} exceeds {model_id}'s limits.maxPixels of \
                                 {max_pixels}"
                            ),
                        ));
                    }
                }
            }
        }
    }
    findings
}

/// Every document-level check in one call: structure of both documents, cross-references, and
/// (when given) the pack directory and the model entry. Structural findings short-circuit the
/// rest, because cross-checks over a malformed document only repeat them.
pub fn validate_all(
    plan: &ProductionPlan,
    pack: &ReferencePack,
    pack_dir: Option<&Path>,
    model_entry: Option<(&Map<String, Value>, ModelLane)>,
) -> Vec<PlanDiagnostic> {
    let mut findings = validate_plan_structure(plan);
    findings.extend(validate_reference_pack(pack));
    if !findings.is_empty() {
        return findings;
    }
    findings.extend(validate_plan_against_pack(plan, pack));
    if let Some(dir) = pack_dir {
        findings.extend(validate_reference_pack_files(pack, dir));
    }
    if let Some((entry, lane)) = model_entry {
        findings.extend(validate_plan_against_model(plan, entry, lane));
    }
    findings
}

// ---------------------------------------------------------------------------------------------
// Run record
// ---------------------------------------------------------------------------------------------

/// Terminal outcome of one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    /// Every selected shot rendered and the timeline exported.
    Completed,
    /// Validation refused the plan before any job was created.
    Rejected,
    /// The run wall-clock budget ran out; no further shots were dispatched.
    StoppedRunBudget,
    /// An attempt's observed peak memory exceeded the budget; no further shots were dispatched.
    StoppedMemoryLimit,
    /// A cancel was requested; live jobs were canceled and no further shot was dispatched.
    Canceled,
    /// At least one selected shot did not render (or the export failed) within its limits.
    Failed,
}

/// Whether a record on disk is one a controller is (or was) working, or one that is done with it.
///
/// This is the field a resume reads first: a `running` record means the controller either is alive
/// or died mid-run, and either way the shots, jobs and takes it names are the ones to reconcile
/// against the API rather than re-create. It is written at EVERY state transition, so the record is
/// never further behind the API than one transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    /// A controller holds this run. `outcome` is not meaningful yet.
    Running,
    /// No controller holds this run; `outcome` and `stop` describe how it ended.
    Finished,
}

/// Why a run stopped dispatching, and whether it can be picked up again.
///
/// `resumable` is the whole point of the field: a cancel or a crash leaves work a `resume` can
/// finish under the SAME declared limits, while an exhausted budget or attempt cap does not —
/// continuing those would be exactly the unbounded retry the limits exist to prevent, so they are
/// terminal and say what the human would have to change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunStop {
    /// Stable code: `canceled`, `run_budget`, `memory_limit`, `attempts_exhausted`,
    /// `export_failed`, `interrupted`.
    pub reason: String,
    /// What happened, in one sentence, including what to change when it is not resumable.
    pub detail: String,
    /// Whether `film-harness resume` can continue this run under its declared limits.
    pub resumable: bool,
}

/// A human's rejection of a take. The take, its job and its asset all stay in the record: this is
/// the decision, not a deletion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TakeRejection {
    pub at: String,
    pub reason: String,
}

/// A dependent shot that must be looked at because something it declared a dependency on changed.
/// Raised only by an explicit human action (a take replacement); never by replay, and never
/// accompanied by an automatic re-render.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewFlag {
    pub raised_at: String,
    /// The shot whose selected take changed.
    pub source_shot_id: String,
    /// The declared [`SHOT_DEPENDENCY_KINDS`] edge that carried the change.
    pub dependency: String,
    pub reason: String,
}

/// One review of one take, indexed in the run record. The **observations themselves live in their
/// own document** ([`crate::film_review::ObservedState`]) at `record_path`: this is a pointer and a
/// count, never the observed state itself, so nothing a vision model said can be mistaken for part
/// of the run's intent (sc-22714).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TakeReviewSummary {
    /// `<runId>:<shotId>:a<attempt>:r<n>`.
    pub review_id: String,
    pub reviewed_at: String,
    /// The attempt whose take was reviewed.
    pub attempt: u32,
    /// Path of the observed-state document, relative to the run directory.
    pub record_path: String,
    /// `image_vqa` for the API seam, `scripted` for a fake. A `scripted` summary is evidence of a
    /// rehearsal, not of a review.
    pub backend: String,
    pub model: String,
    pub observations: u32,
    pub unobserved: u32,
    /// Flags a person has to look at (`mismatch` + `unobserved`, not the `uncertain` tail).
    pub actionable_flags: u32,
    pub topics_flagged: Vec<String>,
    /// Set when a declared review limit ended it early.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
}

/// Where a human got to on one shot. Only a person writes this; a review never does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanTakeDecision {
    /// `accepted` or `rejected`.
    pub state: String,
    pub at: String,
    /// The attempt the decision was about.
    pub attempt: u32,
    pub reason: String,
}

/// One human decision, in the order the decisions were made: the provenance half of the production
/// record. Replay adds nothing here — only a person's `resume`, `cancel`, `replace-take`,
/// `accept-take`, `reject-take` or `request-repair` does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProductionDecision {
    pub at: String,
    /// `resume`, `cancel`, `replace_take`, `review`, `accept_take`, `reject_take` or
    /// `request_repair`.
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shot_id: Option<String>,
    pub detail: String,
}

/// Outcome of one shot within a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShotOutcome {
    Rendered,
    Failed,
    TimedOut,
    /// The attempt in flight was canceled. Distinct from `Failed`: nothing went wrong with it, and
    /// the shot still has its remaining attempts.
    Canceled,
    /// Selected but never dispatched because a run-level limit stopped dispatch first.
    NotDispatched,
    /// In the plan but not in this run's selection.
    NotSelected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceDocument {
    pub id: String,
    pub version: u32,
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareRecord {
    pub platform: String,
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_memory_gb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
}

/// The model as the plan asked for it and as the run observed it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRecord {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_requested: Option<String>,
    pub fps: u32,
    pub lane: String,
    /// Backend label the worker reported on the first completed take (`mlx` / `cuda` / ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_observed: Option<String>,
    /// Primary weights download for the requested tier, as the manifest declares it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<Value>,
    pub hardware: HardwareRecord,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReferenceAssetRecord {
    pub role: String,
    pub kind: String,
    pub file: String,
    pub sha256: String,
    pub asset_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntendedState {
    pub mode: String,
    pub start_state: String,
    pub end_state: String,
    pub target_duration_seconds: f64,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialogue: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sound: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditioningAssets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame_asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame_asset_id: Option<String>,
    #[serde(default)]
    pub reference_asset_ids: Vec<String>,
}

/// One take as the API persisted it: the asset the job produced plus the provenance the worker
/// reported for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TakeRecord {
    pub asset_id: String,
    pub media_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoded_duration_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoded_fps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoded_frame_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_audio: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    pub model: String,
    /// The worker's `rawAdapterSettings` verbatim (tier that loaded, task that denoised, steps).
    #[serde(default)]
    pub raw_adapter_settings: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttemptRecord {
    pub attempt: u32,
    /// `<runId>:<shotId>:a<attempt>` — written into the job payload and into this record BEFORE the
    /// job is created, so a controller that died between the two finds its own job on replay
    /// instead of enqueuing a second one.
    #[serde(default)]
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    /// `dispatching` until a job id is known, then the job's own status, or `timed_out` /
    /// `canceled` when a limit or a cancel ended the wait.
    pub status: String,
    pub started_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    pub elapsed_seconds: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_gpu_memory_pct: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub take: Option<TakeRecord>,
    /// Set when a human rejected this take. The take, job and asset stay recorded; this only says a
    /// person asked for a different one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejection: Option<TakeRejection>,
    /// True for the one attempt an explicit `replace-take` authorised. It is not an automatic
    /// retry, so it is not counted against `limits.maxAttemptsPerShot`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub human_requested: bool,
}

impl AttemptRecord {
    /// Whether this attempt produced a take that is still a candidate for selection.
    pub fn has_live_take(&self) -> bool {
        self.take.is_some() && self.rejection.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShotRunRecord {
    pub shot_id: String,
    pub outcome: ShotOutcome,
    pub intended: IntendedState,
    #[serde(default)]
    pub conditioning_assets: ConditioningAssets,
    /// Every attempt this shot ever made, with its provenance. Nothing is ever removed: a rejected
    /// take stays beside the one that replaced it.
    #[serde(default)]
    pub attempts: Vec<AttemptRecord>,
    /// The [`AttemptRecord::attempt`] whose take represents this shot. At most one, and only ever
    /// set when it was `None` (first successful take) or changed by an explicit `replace-take`:
    /// resuming a run NEVER moves it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_attempt: Option<u32>,
    /// Raised when a shot this one declared a dependency on got a different selected take. The
    /// human resolves these; the harness never regenerates a flagged shot on its own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub needs_review: Vec<ReviewFlag>,
    /// Every vision-assisted review this shot has had, oldest first (sc-22714). Append-only: a
    /// re-review never replaces the evidence of an earlier one. Each entry POINTS AT an
    /// observed-state document; none of them carries observed values.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reviews: Vec<TakeReviewSummary>,
    /// The human's standing decision on this shot's take, if they have made one. Absent means
    /// nobody has looked yet — which is not the same as approved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_decision: Option<HumanTakeDecision>,
}

impl ShotRunRecord {
    /// The selected take's attempt, if this shot has one.
    pub fn selected(&self) -> Option<&AttemptRecord> {
        let attempt = self.selected_attempt?;
        self.attempts
            .iter()
            .find(|candidate| candidate.attempt == attempt)
    }

    /// Automatic attempts made so far — a `replace-take` attempt is a human decision, not a retry,
    /// so it does not spend the plan's attempt cap.
    pub fn automatic_attempts(&self) -> u32 {
        u32::try_from(
            self.attempts
                .iter()
                .filter(|attempt| !attempt.human_requested)
                .count(),
        )
        .unwrap_or(u32::MAX)
    }

    /// The most recent review of this shot, if it has had one.
    pub fn latest_review(&self) -> Option<&TakeReviewSummary> {
        self.reviews.last()
    }

    /// The most recent review OF THE SELECTED TAKE. A review of a take that has since been
    /// replaced says nothing about the one selected now, so the accept/repair paths ask for this
    /// rather than for `latest_review`.
    pub fn review_of_selected(&self) -> Option<&TakeReviewSummary> {
        let attempt = self.selected_attempt?;
        self.reviews
            .iter()
            .rev()
            .find(|review| review.attempt == attempt)
    }

    /// The next attempt number for this shot (attempt numbers never repeat within a shot).
    pub fn next_attempt_number(&self) -> u32 {
        self.attempts
            .iter()
            .map(|attempt| attempt.attempt)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineItemRecord {
    pub shot_id: String,
    pub item_id: String,
    pub asset_id: String,
    pub timeline_start: f64,
    pub timeline_end: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineRecord {
    pub timeline_id: String,
    pub name: String,
    pub aspect_ratio: String,
    pub fps: u32,
    pub items: Vec<TimelineItemRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportRecord {
    pub job_id: String,
    pub status: String,
    /// True once a selected take changed after this export ran: the MP4 no longer matches the
    /// selected takes. Flag only — re-exporting is an explicit request.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stale: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// An export the controller is about to dispatch, written BEFORE the job is created so a
/// controller that died in between adopts its own export job instead of starting a second render.
/// `supersedes` is the export job this one replaces (a re-export after a take changed), which is
/// what tells a resume apart the old finished job from the new one on the same timeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportPending {
    pub requested_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<String>,
}

/// The production record of one run: shot -> attempt -> job -> asset, the takes and which one is
/// selected, the human decisions, the timeline, the export, and the observed model/backend/hardware.
///
/// This is the **durable state of the run**, not a report written at the end: it is rewritten at
/// every state transition (before a job is created, after its id is known, after a take is
/// adopted, after the timeline and after the export), so a controller killed at any point leaves a
/// record that `resume` can reconcile against the API without re-dispatching anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRecord {
    pub schema_version: u32,
    pub run_id: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    /// `running` while a controller holds the run — the field a resume reads first.
    pub state: RunState,
    pub outcome: RunOutcome,
    /// Why dispatch stopped and whether a resume may continue. `None` on a clean completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<RunStop>,
    pub plan: SourceDocument,
    pub reference_pack: SourceDocument,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelRecord>,
    pub limits: PlanLimits,
    pub selected_shot_ids: Vec<String>,
    #[serde(default)]
    pub references: Vec<ReferenceAssetRecord>,
    #[serde(default)]
    pub shots: Vec<ShotRunRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline: Option<TimelineRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<ExportRecord>,
    /// Set while an export job has been asked for but its id is not recorded yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_pending: Option<ExportPending>,
    #[serde(default)]
    pub diagnostics: Vec<PlanDiagnostic>,
    /// Every human decision taken on this run, oldest first.
    #[serde(default)]
    pub decisions: Vec<ProductionDecision>,
    /// Wall-clock the run has spent across every controller that has held it, so the plan's
    /// `limits.maxRunSeconds` bounds the run and not merely one attempt at it.
    pub elapsed_seconds: f64,
}

impl RunRecord {
    /// Serialize with stable key order for diffs.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({}))
    }

    /// The shot record for `shot_id`, if the run has one.
    pub fn shot(&self, shot_id: &str) -> Option<&ShotRunRecord> {
        self.shots.iter().find(|shot| shot.shot_id == shot_id)
    }

    /// Mutable access to the shot record for `shot_id`.
    pub fn shot_mut(&mut self, shot_id: &str) -> Option<&mut ShotRunRecord> {
        self.shots.iter_mut().find(|shot| shot.shot_id == shot_id)
    }

    /// Whether a controller may pick this run up again: it either died holding it (`running`) or
    /// stopped for a reason that left work to do under the same declared limits.
    pub fn is_resumable(&self) -> bool {
        match self.state {
            RunState::Running => self.outcome != RunOutcome::Rejected,
            RunState::Finished => self.stop.as_ref().is_some_and(|stop| stop.resumable),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_json() -> Value {
        json!({
            "schemaVersion": 1,
            "id": "courier-workshop",
            "version": 1,
            "title": "Courier",
            "model": { "id": "minimax_h3", "tier": "q4", "resolution": "576x320" },
            "limits": { "maxRunSeconds": 3600, "maxShotSeconds": 1800, "maxAttemptsPerShot": 1, "maxMemoryGb": 96 },
            "shots": [
                {
                    "id": "SH010", "beat": "enter", "framing": "wide", "prompt": "a courier enters",
                    "targetDurationSeconds": 5.1667, "startState": "empty", "endState": "courier inside",
                    "conditioning": { "mode": "text_to_video" }
                },
                {
                    "id": "SH020", "beat": "place", "framing": "medium", "prompt": "places the parcel",
                    "targetDurationSeconds": 5.1667, "startState": "courier inside", "endState": "parcel on table",
                    "conditioning": { "mode": "image_to_video", "firstFrameRole": "workshop_plate" },
                    "continuityRoles": ["red_parcel"]
                }
            ]
        })
    }

    fn pack_json() -> Value {
        json!({
            "schemaVersion": 1,
            "id": "courier-refs",
            "version": 1,
            "references": [
                { "role": "workshop_plate", "kind": "plate", "file": "references/workshop_plate.png" },
                { "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png" },
                { "role": "unapproved_look", "kind": "style", "file": "references/look.png", "approved": false }
            ]
        })
    }

    fn model_entry() -> Map<String, Value> {
        json!({
            "id": "minimax_h3",
            "capabilities": ["text_to_video", "image_to_video", "first_last_frame"],
            "video": { "supportsGuidance": false, "supportsNegativePrompt": false },
            "defaults": { "duration": 5.1667, "fps": 24, "resolution": "1344x768" },
            "limits": {
                "durations": [5.1667, 5.875, 14.375],
                "hardMinDuration": 5.1667,
                "hardMaxDuration": 14.375,
                "fps": [24],
                "maxPixels": 1032192,
                "resolutions": ["1344x768", "576x320"],
                "maxReferenceAssets": 0
            },
            "mlx": { "minMemoryGb": 64 }
        })
        .as_object()
        .cloned()
        .expect("object")
    }

    fn plan() -> ProductionPlan {
        serde_json::from_value(plan_json()).expect("plan parses")
    }

    fn pack() -> ReferencePack {
        serde_json::from_value(pack_json()).expect("pack parses")
    }

    fn messages(findings: &[PlanDiagnostic]) -> Vec<String> {
        findings.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn well_formed_plan_pack_and_model_produce_no_findings() {
        let plan = plan();
        let pack = pack();
        assert!(validate_plan_structure(&plan).is_empty());
        assert!(validate_reference_pack(&pack).is_empty());
        assert!(validate_plan_against_pack(&plan, &pack).is_empty());
        assert!(validate_plan_against_model(&plan, &model_entry(), ModelLane::Mlx).is_empty());
    }

    #[test]
    fn jsonc_comments_are_tolerated_and_unknown_fields_are_refused() {
        let text = format!(
            "// plan\n{}",
            serde_json::to_string_pretty(&plan_json()).unwrap()
        );
        assert!(parse_plan(&text).is_ok());
        let mut bad = plan_json();
        bad["shots"][0]["framng"] = json!("wide");
        let error = parse_plan(&bad.to_string()).expect_err("unknown field refused");
        assert!(error.contains("framng"), "{error}");
    }

    #[test]
    fn structural_findings_name_the_shot_and_field() {
        let mut value = plan_json();
        value["shots"][1]["id"] = json!("SH010");
        value["shots"][0]["prompt"] = json!("");
        value["shots"][0]["conditioning"] = json!({ "mode": "image_to_video" });
        value["limits"]["maxAttemptsPerShot"] = json!(0);
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_structure(&plan));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("duplicate shot id \"SH010\"")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.starts_with("[SH010] prompt:")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.starts_with("[SH010] conditioning.firstFrameRole:")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("limits.maxAttemptsPerShot")),
            "{findings:?}"
        );
    }

    #[test]
    fn keyframes_and_references_cannot_mix_on_one_shot() {
        let mut value = plan_json();
        value["shots"][1]["conditioning"] = json!({
            "mode": "image_to_video",
            "firstFrameRole": "workshop_plate",
            "referenceRoles": ["red_parcel"]
        });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_structure(&plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].starts_with("[SH020] conditioning.referenceRoles:"));
        let mut value = plan_json();
        value["shots"][0]["conditioning"] = json!({ "mode": "reference_to_video" });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_structure(&plan));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("requires at least one reference role")),
            "{findings:?}"
        );
    }

    #[test]
    fn dangling_and_unapproved_roles_are_reported_per_shot() {
        let mut value = plan_json();
        value["shots"][1]["conditioning"]["firstFrameRole"] = json!("missing_role");
        value["shots"][0]["conditioning"] =
            json!({ "mode": "image_to_video", "firstFrameRole": "unapproved_look" });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_against_pack(&plan, &pack()));
        assert_eq!(findings.len(), 2, "{findings:?}");
        assert!(findings.iter().any(
            |m| m.contains("[SH020] conditioning.firstFrameRole") && m.contains("missing_role")
        ));
        assert!(findings
            .iter()
            .any(|m| m.contains("[SH010]") && m.contains("not approved")));
    }

    #[test]
    fn pack_findings_cover_duplicates_kinds_paths_and_missing_files() {
        let mut value = pack_json();
        value["references"][1]["role"] = json!("workshop_plate");
        value["references"][2]["kind"] = json!("mood");
        value["references"][2]["file"] = json!("../escape.png");
        let pack: ReferencePack = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_reference_pack(&pack));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("duplicate reference role")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("unknown reference kind")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("relative path")),
            "{findings:?}"
        );

        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("references")).unwrap();
        std::fs::write(dir.path().join("references/workshop_plate.png"), b"png").unwrap();
        std::fs::write(dir.path().join("references/look.png"), b"").unwrap();
        let findings = messages(&validate_reference_pack_files(&self::pack(), dir.path()));
        assert_eq!(findings.len(), 2, "{findings:?}");
        assert!(findings
            .iter()
            .any(|m| m.contains("\"red_parcel\"") && m.contains("missing")));
        assert!(findings
            .iter()
            .any(|m| m.contains("\"unapproved_look\"") && m.contains("empty")));
    }

    #[test]
    fn model_findings_cover_mode_duration_resolution_references_negative_prompt_and_memory() {
        let mut value = plan_json();
        value["limits"]["maxMemoryGb"] = json!(32);
        value["shots"][0]["conditioning"] =
            json!({ "mode": "reference_to_video", "referenceRoles": ["red_parcel"] });
        value["shots"][0]["negativePrompt"] = json!("blurry");
        value["shots"][1]["targetDurationSeconds"] = json!(6.0);
        value["shots"][1]["resolution"] = json!("640x360");
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_against_model(
            &plan,
            &model_entry(),
            ModelLane::Mlx,
        ));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("limits.maxMemoryGb") && m.contains("64")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(
                |m| m.contains("[SH010] conditioning.mode") && m.contains("reference_to_video")
            ),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] conditioning.referenceRoles")
                    && m.contains("maxReferenceAssets")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] negativePrompt")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH020] targetDurationSeconds") && m.contains("menu")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH020] resolution") && m.contains("640x360")),
            "{findings:?}"
        );

        let mut value = plan_json();
        value["shots"][1]["targetDurationSeconds"] = json!(20.0);
        value["model"]["fps"] = json!(30);
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_against_model(
            &plan,
            &model_entry(),
            ModelLane::Mlx,
        ));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH020] targetDurationSeconds")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("model.fps")),
            "{findings:?}"
        );
    }

    #[test]
    fn candle_lane_reads_the_candle_memory_minimum() {
        let mut entry = model_entry();
        entry.insert("candle".to_owned(), json!({ "minMemoryGb": 43 }));
        assert_eq!(model_min_memory_gb(&entry, ModelLane::Candle), Some(43.0));
        assert_eq!(model_min_memory_gb(&entry, ModelLane::Mlx), Some(64.0));
    }

    #[test]
    fn validate_all_stops_at_structure_before_cross_checks() {
        let mut value = plan_json();
        value["shots"][0]["id"] = json!("bad id");
        value["shots"][1]["conditioning"]["firstFrameRole"] = json!("missing");
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_all(
            &plan,
            &pack(),
            None,
            Some((&model_entry(), ModelLane::Mlx)),
        ));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("shots[0].id"));
    }

    #[test]
    fn run_record_round_trips() {
        let record = RunRecord {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            run_id: "run_1".into(),
            created_at: "2026-09-13T00:00:00Z".into(),
            finished_at: None,
            state: RunState::Finished,
            outcome: RunOutcome::Rejected,
            stop: Some(RunStop {
                reason: "interrupted".into(),
                detail: "refused".into(),
                resumable: false,
            }),
            plan: SourceDocument {
                id: "p".into(),
                version: 1,
                path: "plan.json".into(),
                sha256: "0".into(),
            },
            reference_pack: SourceDocument {
                id: "r".into(),
                version: 1,
                path: "references.json".into(),
                sha256: "0".into(),
            },
            project_id: None,
            project_path: None,
            model: None,
            limits: plan().limits,
            selected_shot_ids: vec!["SH010".into()],
            references: vec![],
            shots: vec![],
            timeline: None,
            export: None,
            export_pending: None,
            diagnostics: vec![PlanDiagnostic::shot("SH010", "prompt", "empty")],
            decisions: vec![ProductionDecision {
                at: "2026-09-13T00:00:01Z".into(),
                action: "cancel".into(),
                shot_id: None,
                detail: "operator".into(),
            }],
            elapsed_seconds: 0.0,
        };
        let json = record.to_json();
        assert_eq!(json["outcome"], "rejected");
        assert_eq!(json["state"], "finished");
        assert_eq!(json["stop"]["resumable"], false);
        let back: RunRecord = serde_json::from_value(json).unwrap();
        assert_eq!(back, record);
        assert!(!back.is_resumable());
    }

    // -----------------------------------------------------------------------------------------
    // sc-22711 — dependency edges
    // -----------------------------------------------------------------------------------------

    fn plan_with_edges(edges: Value) -> ProductionPlan {
        let mut value = plan_json();
        value["schemaVersion"] = json!(2);
        value["shots"][1]["dependsOn"] = edges;
        serde_json::from_value(value).expect("plan parses")
    }

    #[test]
    fn a_v1_plan_still_reads_and_declares_no_edges() {
        let plan = plan();
        assert_eq!(plan.schema_version, 1);
        assert!(validate_plan_structure(&plan).is_empty());
        assert!(plan.shots.iter().all(|shot| shot.depends_on.is_empty()));
    }

    #[test]
    fn dependency_edges_must_name_another_shot_in_the_plan_with_a_known_kind() {
        let plan = plan_with_edges(json!([
            { "shotId": "SH010", "kind": "continuity", "note": "carries the parcel in" }
        ]));
        assert!(
            validate_plan_structure(&plan).is_empty(),
            "{:?}",
            messages(&validate_plan_structure(&plan))
        );
        assert_eq!(direct_dependents(&plan, "SH010").len(), 1);
        assert_eq!(direct_dependents(&plan, "SH020").len(), 0);

        let plan = plan_with_edges(json!([
            { "shotId": "SH999", "kind": "continuity" },
            { "shotId": "SH020", "kind": "continuity" },
            { "shotId": "SH010", "kind": "vibes" },
            { "shotId": "SH010", "kind": "continuity" }
        ]));
        let findings = messages(&validate_plan_structure(&plan));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("dependsOn.shotId") && m.contains("not a shot in this plan")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("cannot depend on itself")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("dependsOn.kind") && m.contains("vibes")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("depended on twice")),
            "{findings:?}"
        );
    }

    #[test]
    fn a_dependency_cycle_is_refused() {
        let mut value = plan_json();
        value["schemaVersion"] = json!(2);
        value["shots"][0]["dependsOn"] = json!([{ "shotId": "SH020", "kind": "conditioning" }]);
        value["shots"][1]["dependsOn"] = json!([{ "shotId": "SH010", "kind": "continuity" }]);
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_structure(&plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("dependency cycle") && findings[0].contains("SH010"),
            "{findings:?}"
        );
    }

    #[test]
    fn a_long_acyclic_chain_is_accepted() {
        let shots: Vec<Value> = (0..200)
            .map(|index| {
                let mut shot = plan_json()["shots"][0].clone();
                shot["id"] = json!(format!("SH{index:03}"));
                if index > 0 {
                    shot["dependsOn"] =
                        json!([{ "shotId": format!("SH{:03}", index - 1), "kind": "continuity" }]);
                }
                shot
            })
            .collect();
        let mut value = plan_json();
        value["schemaVersion"] = json!(2);
        value["shots"] = Value::Array(shots);
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        assert!(
            validate_plan_structure(&plan).is_empty(),
            "{:?}",
            messages(&validate_plan_structure(&plan))
        );
    }

    #[test]
    fn take_selection_and_attempt_accounting_ignore_human_requested_attempts() {
        let attempt = |number: u32, human: bool, take: bool| AttemptRecord {
            attempt: number,
            idempotency_key: format!("run_1:SH010:a{number}"),
            job_id: Some(format!("job{number}")),
            status: "completed".into(),
            started_at: "t".into(),
            finished_at: None,
            elapsed_seconds: 1.0,
            peak_gpu_memory_pct: None,
            error: None,
            take: take.then(|| TakeRecord {
                asset_id: format!("asset{number}"),
                media_path: "p".into(),
                encoded_duration_seconds: None,
                encoded_fps: None,
                encoded_frame_count: None,
                has_audio: None,
                seed: None,
                adapter: None,
                backend: None,
                model: "m".into(),
                raw_adapter_settings: Value::Null,
            }),
            rejection: None,
            human_requested: human,
        };
        let mut shot = ShotRunRecord {
            shot_id: "SH010".into(),
            outcome: ShotOutcome::Rendered,
            intended: IntendedState {
                mode: "text_to_video".into(),
                start_state: "a".into(),
                end_state: "b".into(),
                target_duration_seconds: 5.0,
                width: 576,
                height: 320,
                fps: 24,
                dialogue: None,
                sound: None,
            },
            conditioning_assets: ConditioningAssets::default(),
            attempts: vec![attempt(1, false, false), attempt(2, false, true)],
            selected_attempt: Some(2),
            needs_review: Vec::new(),
            reviews: Vec::new(),
            human_decision: None,
        };
        assert_eq!(shot.automatic_attempts(), 2);
        assert_eq!(shot.next_attempt_number(), 3);
        assert_eq!(shot.selected().map(|a| a.attempt), Some(2));

        shot.attempts.push(attempt(3, true, true));
        assert_eq!(
            shot.automatic_attempts(),
            2,
            "a human-requested attempt does not spend the cap"
        );
        assert_eq!(shot.next_attempt_number(), 4);
        shot.attempts[1].rejection = Some(TakeRejection {
            at: "t".into(),
            reason: "wrong parcel".into(),
        });
        assert!(!shot.attempts[1].has_live_take());
        assert!(shot.attempts[2].has_live_take());
    }
}
