//! Turning a brief into a [`ProductionPlan`](crate::film_plan::ProductionPlan) with a LOCAL
//! planner (epic 22708, sc-22713).
//!
//! This module is the pure half of the planner: it owns the brief document, the capability
//! envelope the installed model actually declares, the request text handed to the local LLM, the
//! strict parse of what comes back, and the findings that decide whether a draft is a plan or a
//! repair round. It performs no I/O beyond reading the two documents it is given and knows nothing
//! about jobs, routes or transports — the driver (`rust-api` `film_planner`) supplies the model
//! replies, so every rule below is testable against a scripted reply with no API and no GPU.
//!
//! Three properties the rest of the harness relies on:
//!
//! * **The output is the SAME document the hand-authored path uses.** A generated plan is a
//!   [`ProductionPlan`] that `film-harness validate` accepts unchanged; there is no second schema
//!   and no planner-only field the driver has to interpret.
//! * **The brief's beats are a contract.** Every [`RequiredBeat`] must be covered by at least one
//!   shot. A draft that drops one is a finding, never a shorter plan — and the finding is fed back
//!   verbatim to the next repair round, so a bounded repair loop can fix it or fail loudly.
//! * **Nothing is coerced to fit.** Timing off the model's declared menu, a mode it does not
//!   declare, a reference role the pack does not contain: all findings. The planner never rounds a
//!   duration, never silently drops a reference and never invents a role.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::film_plan::{
    parse_resolution, validate_all, ModelLane, PlanDiagnostic, PlanLimits, PlanModel,
    ProductionPlan, ReferencePack, Shot, ShotConditioning, PLAN_SCHEMA_VERSION,
    SHOT_CONDITIONING_MODES,
};
use crate::jsonc::strip_jsonc_comments;
use crate::video_request::{default_fps, default_resolution, reference_caps};

/// Schema version of [`ProductionBrief`] documents this module reads.
pub const BRIEF_SCHEMA_VERSION: u32 = 1;

/// Hard ceiling on the shots one planner draft may contain, whatever the brief asks for. A finite
/// bound on the planner's output is what keeps a runaway reply from becoming a runaway run (E5);
/// the brief's own `maxShots` may only narrow it.
pub const MAX_PLANNER_SHOTS: usize = 24;

/// The user's input: what the film is, how long it should run, and the beats it must contain. The
/// approved reference pack is the planner's other input and stays a separate document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProductionBrief {
    pub schema_version: u32,
    /// Id of the plan this brief produces (`[A-Za-z0-9_-]{1,64}`).
    pub id: String,
    pub version: u32,
    pub title: String,
    pub synopsis: String,
    /// Free-text direction: look, palette, camera language, sound. Handed to the planner verbatim.
    #[serde(default)]
    pub style_notes: String,
    /// Total running time the sequence should land in, in seconds.
    pub target_total_seconds: DurationWindow,
    /// Beats the film must contain, in narrative order. Every one must map to at least one shot.
    pub required_beats: Vec<RequiredBeat>,
    /// The model the plan renders through, exactly as a hand-authored plan declares it.
    pub model: PlanModel,
    /// The limits the generated plan declares.
    pub limits: PlanLimits,
    /// Ceiling on the number of shots, never above [`MAX_PLANNER_SHOTS`].
    #[serde(default = "default_max_shots")]
    pub max_shots: usize,
}

fn default_max_shots() -> usize {
    MAX_PLANNER_SHOTS
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DurationWindow {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RequiredBeat {
    /// Stable id (`[A-Za-z0-9_-]{1,64}`) the planner tags each shot with, so coverage is checked by
    /// identity rather than by matching prose.
    pub id: String,
    pub summary: String,
}

/// Read and parse a brief document (JSONC tolerated).
pub fn read_brief_file(path: &Path) -> Result<ProductionBrief, PlanDiagnostic> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        PlanDiagnostic::plan("brief", format!("cannot read {}: {error}", path.display()))
    })?;
    parse_brief(&text)
        .map_err(|error| PlanDiagnostic::plan("brief", format!("{}: {error}", path.display())))
}

/// Parse a brief from JSON/JSONC text. Unknown fields are refused.
pub fn parse_brief(text: &str) -> Result<ProductionBrief, String> {
    serde_json::from_str(&strip_jsonc_comments(text)).map_err(|error| error.to_string())
}

/// Structural findings on the brief alone — checked before a single token is generated.
pub fn validate_brief(brief: &ProductionBrief) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if brief.schema_version != BRIEF_SCHEMA_VERSION {
        findings.push(PlanDiagnostic::plan(
            "brief.schemaVersion",
            format!(
                "unsupported brief schema version {} (this build reads {BRIEF_SCHEMA_VERSION})",
                brief.schema_version
            ),
        ));
    }
    if !crate::film_plan::is_safe_plan_id(&brief.id) {
        findings.push(PlanDiagnostic::plan(
            "brief.id",
            "brief id must be 1-64 characters of [A-Za-z0-9_-]",
        ));
    }
    if brief.version == 0 {
        findings.push(PlanDiagnostic::plan(
            "brief.version",
            "brief version must be >= 1",
        ));
    }
    if brief.title.trim().is_empty() {
        findings.push(PlanDiagnostic::plan("brief.title", "a title is required"));
    }
    if brief.synopsis.trim().is_empty() {
        findings.push(PlanDiagnostic::plan(
            "brief.synopsis",
            "a synopsis is required; it is what the planner writes the film from",
        ));
    }
    let window = brief.target_total_seconds;
    if !window.min.is_finite() || !window.max.is_finite() || window.min <= 0.0 {
        findings.push(PlanDiagnostic::plan(
            "brief.targetTotalSeconds",
            "the target running time must be finite seconds > 0",
        ));
    } else if window.max < window.min {
        findings.push(PlanDiagnostic::plan(
            "brief.targetTotalSeconds",
            format!(
                "max ({}) is below min ({}); the window is empty",
                window.max, window.min
            ),
        ));
    }
    if brief.required_beats.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "brief.requiredBeats",
            "a brief needs at least one required beat; beats are what the plan is checked against",
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, beat) in brief.required_beats.iter().enumerate() {
        if !crate::film_plan::is_safe_plan_id(&beat.id) {
            findings.push(PlanDiagnostic::plan(
                format!("brief.requiredBeats[{index}].id"),
                format!(
                    "beat id {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                    beat.id
                ),
            ));
        } else if !seen.insert(beat.id.as_str()) {
            findings.push(PlanDiagnostic::plan(
                format!("brief.requiredBeats[{index}].id"),
                format!("duplicate beat id {:?}", beat.id),
            ));
        }
        if beat.summary.trim().is_empty() {
            findings.push(PlanDiagnostic::plan(
                format!("brief.requiredBeats[{index}].summary"),
                "a beat needs a summary",
            ));
        }
    }
    if brief.max_shots == 0 || brief.max_shots > MAX_PLANNER_SHOTS {
        findings.push(PlanDiagnostic::plan(
            "brief.maxShots",
            format!("maxShots must be between 1 and {MAX_PLANNER_SHOTS}"),
        ));
    } else if brief.max_shots < brief.required_beats.len() {
        findings.push(PlanDiagnostic::plan(
            "brief.maxShots",
            format!(
                "maxShots ({}) is below the {} required beats, so a covering plan cannot exist",
                brief.max_shots,
                brief.required_beats.len()
            ),
        ));
    }
    findings
}

// ---------------------------------------------------------------------------------------------
// Capability envelope
// ---------------------------------------------------------------------------------------------

/// What the INSTALLED model declares it can do, read off its catalog entry. The planner is told
/// only this — never a capability from a model card, an upstream repo or a hosted API — so a draft
/// that is inside the envelope is dispatchable and one that is not is refused with the same
/// diagnostics `film-harness validate` would produce.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerCapabilities {
    pub model_id: String,
    /// Conditioning modes the entry declares AND this schema can express.
    pub modes: Vec<String>,
    /// The declared clip-length menu, in seconds. Empty when the model declares no menu.
    pub durations: Vec<f64>,
    /// Frames per second the plan will render at.
    pub fps: Option<u32>,
    /// The declared resolution menu, `WxH`. Empty when the model declares no menu.
    pub resolutions: Vec<String>,
    /// The resolution every shot uses unless it overrides it (plan default, else model default).
    pub default_resolution: Option<String>,
    /// `limits.maxReferenceAssets` — zero means the checkpoint has no reference conditioning.
    pub max_reference_images: usize,
    pub supports_negative_prompt: bool,
    /// `<lane>.minMemoryGb`, when declared.
    pub min_memory_gb: Option<f64>,
}

/// Read the capability envelope for `plan_model` out of its catalog entry.
pub fn capabilities_for(
    plan_model: &PlanModel,
    entry: &Map<String, Value>,
    lane: ModelLane,
) -> PlannerCapabilities {
    let limits = entry.get("limits").and_then(Value::as_object);
    let modes = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|mode| SHOT_CONDITIONING_MODES.contains(mode))
        .map(str::to_owned)
        .collect();
    let durations = limits
        .and_then(|limits| limits.get("durations"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_f64)
        .collect();
    let resolutions: Vec<String> = limits
        .and_then(|limits| limits.get("resolutions"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let default_resolution = plan_model
        .resolution
        .clone()
        .filter(|value| parse_resolution(value).is_some())
        .or_else(|| default_resolution(entry).map(|(width, height)| format!("{width}x{height}")));
    PlannerCapabilities {
        model_id: plan_model.id.clone(),
        modes,
        durations,
        fps: plan_model.fps.or_else(|| default_fps(entry)),
        resolutions,
        default_resolution,
        max_reference_images: reference_caps(entry).images,
        supports_negative_prompt: entry
            .get("video")
            .and_then(Value::as_object)
            .and_then(|video| video.get("supportsNegativePrompt"))
            .and_then(Value::as_bool)
            .unwrap_or(true),
        min_memory_gb: crate::film_plan::model_min_memory_gb(entry, lane),
    }
}

impl PlannerCapabilities {
    /// The envelope as the planner is shown it: one line per axis, values only, no prose the model
    /// could read as optional.
    pub fn as_prompt_section(&self) -> String {
        let mut lines = vec![format!("Model: {}", self.model_id)];
        lines.push(format!(
            "Allowed conditioning modes (use no others): {}",
            if self.modes.is_empty() {
                "none".to_owned()
            } else {
                self.modes.join(", ")
            }
        ));
        if self.durations.is_empty() {
            lines.push(
                "Allowed targetDurationSeconds: any positive value this model accepts.".to_owned(),
            );
        } else {
            lines.push(format!(
                "Allowed targetDurationSeconds (copy one of these EXACTLY, they are not \
                 roundable): {}",
                self.durations
                    .iter()
                    .map(|value| format!("{value}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if let Some(fps) = self.fps {
            lines.push(format!(
                "Frame rate: {fps} fps for every shot (fixed; do not write an fps field)."
            ));
        }
        match (&self.default_resolution, self.resolutions.is_empty()) {
            (Some(default), false) => lines.push(format!(
                "Resolution: every shot renders at {default}. Omit the resolution field; the only \
                 other legal values are {}.",
                self.resolutions.join(", ")
            )),
            (Some(default), true) => lines.push(format!(
                "Resolution: every shot renders at {default}. Omit the resolution field."
            )),
            (None, _) => lines.push("Resolution: omit the resolution field.".to_owned()),
        }
        lines.push(if self.max_reference_images == 0 {
            "Reference conditioning: THIS CHECKPOINT HAS NONE. referenceRoles must be empty on \
             every shot, and the reference_to_video mode is unavailable. Keyframe modes \
             (image_to_video, first_last_frame) take a first/last frame role instead; a shot never \
             mixes keyframes with references."
                .to_owned()
        } else {
            format!(
                "Reference conditioning: at most {} reference roles on a reference_to_video shot. \
                 A shot never mixes keyframe roles with reference roles — they are different \
                 conditioning tasks.",
                self.max_reference_images
            )
        });
        if !self.supports_negative_prompt {
            lines.push(
                "Negative prompts: this model has none. Never write a negativePrompt field; state \
                 everything positively in the prompt instead."
                    .to_owned(),
            );
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------------------------
// The draft the planner returns
// ---------------------------------------------------------------------------------------------

/// One shot as the planner writes it: the plan's own shot fields plus the id of the brief beat it
/// covers. Unknown fields are refused, so an invented field is a finding rather than silent data
/// loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DraftShot {
    pub id: String,
    /// The [`RequiredBeat::id`] this shot covers.
    pub beat_id: String,
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
    #[serde(default)]
    pub continuity_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannerDraft {
    pub shots: Vec<DraftShot>,
}

/// Isolate the outermost `{ … }` span of a model reply and parse it strictly. A reply with prose
/// around the object still parses; a reply with an unknown or misspelled field does not.
pub fn parse_planner_output(text: &str) -> Result<PlannerDraft, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("the planner returned nothing".to_owned());
    }
    let start = trimmed
        .find('{')
        .ok_or_else(|| "the planner returned no JSON object".to_owned())?;
    let end = trimmed
        .rfind('}')
        .ok_or_else(|| "the planner's JSON object is not closed".to_owned())?;
    if end <= start {
        return Err("the planner's JSON object is not closed".to_owned());
    }
    serde_json::from_str(&trimmed[start..=end]).map_err(|error| error.to_string())
}

/// Assemble a [`ProductionPlan`] from the brief and a parsed draft. Everything outside the shots —
/// id, version, title, model, limits — comes from the brief, so the planner cannot widen a limit or
/// swap the model.
pub fn draft_to_plan(brief: &ProductionBrief, draft: &PlannerDraft) -> ProductionPlan {
    ProductionPlan {
        schema_version: PLAN_SCHEMA_VERSION,
        id: brief.id.clone(),
        version: brief.version,
        title: brief.title.clone(),
        synopsis: brief.synopsis.clone(),
        model: brief.model.clone(),
        limits: brief.limits.clone(),
        shots: draft
            .shots
            .iter()
            .map(|shot| Shot {
                id: shot.id.clone(),
                beat_id: Some(shot.beat_id.clone()),
                beat: shot.beat.clone(),
                framing: shot.framing.clone(),
                prompt: shot.prompt.clone(),
                negative_prompt: shot.negative_prompt.clone(),
                target_duration_seconds: shot.target_duration_seconds,
                resolution: shot.resolution.clone(),
                start_state: shot.start_state.clone(),
                end_state: shot.end_state.clone(),
                dialogue: shot.dialogue.clone(),
                sound: shot.sound.clone(),
                conditioning: shot.conditioning.clone(),
                seed: shot.seed,
                continuity_roles: shot.continuity_roles.clone(),
            })
            .collect(),
    }
}

/// Beat coverage: every required beat maps to at least one shot, every shot names a beat the brief
/// declares, and the shots that cover the beats run in the brief's narrative order.
///
/// This is the check that makes "never silently drops a required narrative beat" mechanical: a
/// dropped beat is a finding with the beat's id and summary in it, which the repair round hands
/// straight back to the planner.
pub fn coverage_findings(brief: &ProductionBrief, draft: &PlannerDraft) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let declared: BTreeMap<&str, &RequiredBeat> = brief
        .required_beats
        .iter()
        .map(|beat| (beat.id.as_str(), beat))
        .collect();
    let mut covered: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, shot) in draft.shots.iter().enumerate() {
        match declared.get(shot.beat_id.as_str()) {
            Some(_) => covered.entry(&shot.beat_id).or_default().push(index),
            None => findings.push(PlanDiagnostic::shot(
                &shot.id,
                "beatId",
                format!(
                    "{:?} is not a beat in this brief (beats: {})",
                    shot.beat_id,
                    declared.keys().copied().collect::<Vec<_>>().join(", ")
                ),
            )),
        }
    }
    for beat in &brief.required_beats {
        if !covered.contains_key(beat.id.as_str()) {
            findings.push(PlanDiagnostic::plan(
                "shots",
                format!(
                    "required beat {:?} ({}) is covered by no shot; add a shot for it — a beat is \
                     never dropped to make the plan fit",
                    beat.id,
                    beat.summary.trim()
                ),
            ));
        }
    }
    // The beats the brief declares are in narrative order; the shots that cover them must be too,
    // or the film tells the story out of sequence.
    let mut last_position: Option<(usize, &str)> = None;
    for shot in &draft.shots {
        let Some(position) = brief
            .required_beats
            .iter()
            .position(|beat| beat.id == shot.beat_id)
        else {
            continue;
        };
        if let Some((previous, previous_id)) = last_position {
            if position < previous {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "beatId",
                    format!(
                        "beat {:?} comes before {previous_id:?} in the brief but its shot comes \
                         after; keep the shots in the brief's narrative order",
                        shot.beat_id
                    ),
                ));
            }
        }
        last_position = Some((position, &shot.beat_id));
    }
    findings
}

/// Beat coverage of a PLAN (rather than a draft), checked through the `beatId` each generated shot
/// carries. This is what a hand edit is re-checked against: a plan with no beat ids at all was
/// hand-authored and has no brief to answer to, but once a shot claims a beat, every beat the brief
/// declares must still be claimed by one.
pub fn plan_coverage_findings(
    brief: &ProductionBrief,
    plan: &ProductionPlan,
) -> Vec<PlanDiagnostic> {
    let claimed: BTreeSet<&str> = plan
        .shots
        .iter()
        .filter_map(|shot| shot.beat_id.as_deref())
        .collect();
    if claimed.is_empty() {
        return Vec::new();
    }
    let declared: BTreeSet<&str> = brief
        .required_beats
        .iter()
        .map(|beat| beat.id.as_str())
        .collect();
    let mut findings: Vec<PlanDiagnostic> = brief
        .required_beats
        .iter()
        .filter(|beat| !claimed.contains(beat.id.as_str()))
        .map(|beat| {
            PlanDiagnostic::plan(
                "shots",
                format!(
                    "required beat {:?} ({}) is covered by no shot in this plan; restore a shot \
                     for it, or update the brief if the beat is genuinely gone",
                    beat.id,
                    beat.summary.trim()
                ),
            )
        })
        .collect();
    for shot in &plan.shots {
        if let Some(beat_id) = shot.beat_id.as_deref() {
            if !declared.contains(beat_id) {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "beatId",
                    format!(
                        "{beat_id:?} is not a beat in this brief (beats: {})",
                        declared.iter().copied().collect::<Vec<_>>().join(", ")
                    ),
                ));
            }
        }
    }
    findings
}

/// Findings about the shape of the draft itself: how many shots it has and how long they add up to.
pub fn shape_findings(brief: &ProductionBrief, plan: &ProductionPlan) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let ceiling = brief.max_shots.min(MAX_PLANNER_SHOTS);
    if plan.shots.len() > ceiling {
        findings.push(PlanDiagnostic::plan(
            "shots",
            format!(
                "{} shots exceed the {ceiling} this brief allows; cover the beats in fewer shots \
                 rather than dropping one",
                plan.shots.len()
            ),
        ));
    }
    let total: f64 = plan
        .shots
        .iter()
        .map(|shot| shot.target_duration_seconds)
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .sum();
    let window = brief.target_total_seconds;
    let window_usable =
        window.min.is_finite() && window.max.is_finite() && window.max >= window.min && total > 0.0;
    if window_usable && (total < window.min || total > window.max) {
        findings.push(PlanDiagnostic::plan(
            "shots",
            format!(
                "the shots total {total:.4}s, outside the brief's {}-{}s window; change how many \
                 shots cover a beat or which legal clip length each one uses — the lengths \
                 themselves are not adjustable",
                window.min, window.max
            ),
        ));
    }
    findings
}

/// Every finding a generated plan is judged on: the document-level checks the hand-authored path
/// runs ([`validate_all`]), plus the planner-only ones — beat coverage, shot count and total
/// running time. The order is deliberate: coverage first, because a dropped beat is the failure
/// that must never be repaired away by shortening the film.
pub fn validate_generated_plan(
    brief: &ProductionBrief,
    draft: &PlannerDraft,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    pack_dir: Option<&Path>,
    model_entry: Option<(&Map<String, Value>, ModelLane)>,
) -> Vec<PlanDiagnostic> {
    let mut findings = coverage_findings(brief, draft);
    findings.extend(shape_findings(brief, plan));
    findings.extend(validate_all(plan, pack, pack_dir, model_entry));
    findings
}

// ---------------------------------------------------------------------------------------------
// The request the local planner is given
// ---------------------------------------------------------------------------------------------

/// The user turn for the first planning round: the brief, the approved reference roles, the
/// capability envelope, and the exact JSON contract. Deterministic for a given input, so a test can
/// assert what the planner is told rather than what it happened to answer.
pub fn build_planner_request(
    brief: &ProductionBrief,
    pack: &ReferencePack,
    caps: &PlannerCapabilities,
) -> String {
    let mut out = String::new();
    out.push_str("# Brief\n\n");
    out.push_str(&format!("Title: {}\n", brief.title.trim()));
    out.push_str(&format!("Synopsis: {}\n", brief.synopsis.trim()));
    if !brief.style_notes.trim().is_empty() {
        out.push_str(&format!("Style: {}\n", brief.style_notes.trim()));
    }
    out.push_str(&format!(
        "Total running time: between {} and {} seconds across at most {} shots.\n",
        brief.target_total_seconds.min,
        brief.target_total_seconds.max,
        brief.max_shots.min(MAX_PLANNER_SHOTS)
    ));

    out.push_str("\n# Required beats (in order)\n\n");
    out.push_str(
        "Every beat below must be covered by at least one shot, and the shots must run in this \
         order. Tag each shot with the beat id it covers. Never drop a beat to make the running \
         time fit.\n\n",
    );
    for beat in &brief.required_beats {
        out.push_str(&format!("- {}: {}\n", beat.id, beat.summary.trim()));
    }

    out.push_str("\n# Approved reference roles\n\n");
    out.push_str(
        "These are the only reference roles that exist. Name them exactly; never invent one. \
         Every shot lists in continuityRoles the roles it depicts, so the sequence stays anchored \
         to these approved references rather than to whatever the previous shot happened to end \
         on.\n\n",
    );
    for entry in &pack.references {
        if !entry.approved {
            continue;
        }
        out.push_str(&format!(
            "- {} ({}): {}\n",
            entry.role,
            entry.kind,
            entry.description.trim()
        ));
    }

    out.push_str("\n# What this model can actually do\n\n");
    out.push_str(&caps.as_prompt_section());

    out.push_str("\n\n# Output contract\n\n");
    out.push_str(PLAN_JSON_CONTRACT);
    out
}

/// The user turn for a repair round: the draft that was refused and the findings it was refused
/// for, verbatim. Nothing is summarised — the planner is corrected with the same text a human
/// running `film-harness validate` would read.
pub fn build_repair_request(
    brief: &ProductionBrief,
    previous_output: &str,
    findings: &[PlanDiagnostic],
    round: u32,
    max_rounds: u32,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# Repair round {round} of {max_rounds}\n\nYour previous answer was rejected by the \
         validator. Fix EVERY finding below and return the whole plan again as one JSON object in \
         the same format. Do not drop a shot or a beat to make a finding go away: the required \
         beats are\n\n"
    ));
    for beat in &brief.required_beats {
        out.push_str(&format!("- {}: {}\n", beat.id, beat.summary.trim()));
    }
    out.push_str("\n# Findings\n\n");
    for finding in findings {
        out.push_str(&format!("- {finding}\n"));
    }
    out.push_str("\n# Your previous answer\n\n");
    out.push_str(previous_output.trim());
    out.push_str("\n\n# Output contract\n\n");
    out.push_str(PLAN_JSON_CONTRACT);
    out
}

/// The JSON contract both the first round and every repair round end with. Kept as one constant so
/// the two rounds cannot drift.
pub const PLAN_JSON_CONTRACT: &str = "\
Answer with ONE JSON object and nothing else — no prose, no markdown fence, no commentary:

{
  \"shots\": [
    {
      \"id\": \"SH010\",
      \"beatId\": \"<a beat id from the list above>\",
      \"beat\": \"<one sentence: what this shot accomplishes in the story>\",
      \"framing\": \"<shot size, angle, camera movement>\",
      \"prompt\": \"<what the model should render, as a single paragraph>\",
      \"targetDurationSeconds\": <one of the allowed durations, copied exactly>,
      \"startState\": \"<the world at the first frame>\",
      \"endState\": \"<the world at the last frame>\",
      \"sound\": \"<the diegetic sound of this shot>\",
      \"dialogue\": \"<a spoken line, or omit the field>\",
      \"conditioning\": {
        \"mode\": \"<one of the allowed modes>\",
        \"firstFrameRole\": \"<a role, only for image_to_video and first_last_frame>\",
        \"lastFrameRole\": \"<a role, only for first_last_frame>\",
        \"referenceRoles\": [],
        \"chainFromShotId\": \"<an earlier shot id this one continues from, or omit the field>\"
      },
      \"seed\": <an integer, optional>,
      \"continuityRoles\": [\"<the approved roles this shot depicts>\"]
    }
  ]
}

Rules:
- Shot ids ascend in tens: SH010, SH020, SH030 ...
- Every field name is spelled exactly as above. An extra or misspelled field is rejected outright.
- Omit an optional field rather than writing null, \"\" or a placeholder.
- firstFrameRole/lastFrameRole appear only on the keyframe modes that take them, and referenceRoles \
only on reference_to_video. A shot never carries both kinds.
- chainFromShotId records that a shot continues an earlier one. It is never a substitute for \
continuityRoles: a chained shot still names the approved roles it depicts.
- Every shot needs at least one approved role in continuityRoles.";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn brief_json() -> Value {
        json!({
            "schemaVersion": 1,
            "id": "courier-workshop",
            "version": 1,
            "title": "Courier at the Workshop",
            "synopsis": "A courier leaves a red parcel; the recipient finds it.",
            "styleNotes": "Warm late-afternoon light, film grain.",
            "targetTotalSeconds": { "min": 30.0, "max": 60.0 },
            "requiredBeats": [
                { "id": "arrival", "summary": "The courier arrives at the workshop." },
                { "id": "delivery", "summary": "The parcel is left on the bench." },
                { "id": "discovery", "summary": "The recipient finds the parcel." }
            ],
            "model": { "id": "minimax_h3", "tier": "q4", "fps": 24, "resolution": "576x320" },
            "limits": { "maxRunSeconds": 7200, "maxShotSeconds": 2700, "maxAttemptsPerShot": 1, "maxMemoryGb": 96 },
            "maxShots": 8
        })
    }

    fn brief() -> ProductionBrief {
        serde_json::from_value(brief_json()).expect("brief parses")
    }

    fn pack() -> ReferencePack {
        serde_json::from_value(json!({
            "schemaVersion": 1,
            "id": "courier-refs",
            "version": 1,
            "references": [
                { "role": "courier", "kind": "character", "file": "references/courier.png", "description": "Blue jacket." },
                { "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png", "description": "Red box." },
                { "role": "workshop_plate", "kind": "plate", "file": "references/workshop_plate.png", "description": "Wide plate." },
                { "role": "draft_look", "kind": "style", "file": "references/draft.png", "description": "Not approved.", "approved": false }
            ]
        }))
        .expect("pack parses")
    }

    fn model_entry() -> Map<String, Value> {
        json!({
            "id": "minimax_h3",
            "type": "video",
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

    fn draft_shot(id: &str, beat_id: &str) -> Value {
        json!({
            "id": id,
            "beatId": beat_id,
            "beat": "something happens",
            "framing": "wide static",
            "prompt": "A workshop in warm light.",
            "targetDurationSeconds": 14.375,
            "startState": "empty",
            "endState": "not empty",
            "sound": "room tone",
            "conditioning": { "mode": "text_to_video" },
            "continuityRoles": ["courier", "red_parcel"]
        })
    }

    fn good_draft() -> PlannerDraft {
        serde_json::from_value(json!({
            "shots": [
                draft_shot("SH010", "arrival"),
                draft_shot("SH020", "delivery"),
                draft_shot("SH030", "discovery")
            ]
        }))
        .expect("draft parses")
    }

    fn messages(findings: &[PlanDiagnostic]) -> Vec<String> {
        findings.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn a_well_formed_draft_becomes_a_plan_the_hand_authored_validator_accepts() {
        let brief = brief();
        let draft = good_draft();
        let plan = draft_to_plan(&brief, &draft);
        assert_eq!(plan.id, "courier-workshop");
        assert_eq!(plan.model.id, "minimax_h3");
        assert_eq!(plan.limits.max_memory_gb, 96.0);
        let findings = validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&model_entry(), ModelLane::Mlx)),
        );
        assert!(findings.is_empty(), "{:?}", messages(&findings));
        // 3 x 14.375 = 43.125s, inside the 30-60s window.
        assert!(shape_findings(&brief, &plan).is_empty());
    }

    #[test]
    fn malformed_planner_output_is_refused_rather_than_coerced() {
        assert!(parse_planner_output("").is_err());
        assert!(parse_planner_output("I cannot help with that.").is_err());
        assert!(parse_planner_output("{\"shots\": [").is_err());
        // An unknown field is refused outright, not dropped.
        let error = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": {"mode": "text_to_video"}, "cameraLens": "35mm"}]}"#,
        )
        .expect_err("unknown field refused");
        assert!(error.contains("cameraLens"), "{error}");
        // A missing required field is refused.
        let error = parse_planner_output(r#"{"shots": [{"id": "SH010"}]}"#)
            .expect_err("incomplete shot refused");
        assert!(error.contains("beatId"), "{error}");
        // Prose around a well-formed object still parses — the object is isolated.
        let draft = parse_planner_output(&format!(
            "Sure! Here is the plan:\n{}\nLet me know if you want changes.",
            serde_json::to_string(&good_draft()).unwrap()
        ))
        .expect("isolated object parses");
        assert_eq!(draft.shots.len(), 3);
    }

    #[test]
    fn a_dropped_beat_is_a_finding_that_names_the_beat() {
        let brief = brief();
        let mut draft = good_draft();
        draft.shots.retain(|shot| shot.beat_id != "delivery");
        let findings = messages(&coverage_findings(&brief, &draft));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("required beat \"delivery\"")
                && findings[0].contains("The parcel is left on the bench."),
            "{findings:?}"
        );

        // An invented beat id is named too.
        let mut draft = good_draft();
        draft.shots[1].beat_id = "montage".to_owned();
        let findings = messages(&coverage_findings(&brief, &draft));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH020] beatId") && m.contains("montage")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("required beat \"delivery\"")),
            "{findings:?}"
        );

        // Beats covered out of the brief's order are a finding.
        let mut draft = good_draft();
        draft.shots.swap(0, 1);
        let findings = messages(&coverage_findings(&brief, &draft));
        assert!(
            findings.iter().any(|m| m.contains("narrative order")),
            "{findings:?}"
        );
    }

    #[test]
    fn impossible_timing_and_oversized_drafts_are_findings_not_silent_coercion() {
        let brief = brief();
        let mut draft = good_draft();
        // Three of the shortest rung is 15.5s — under the brief's 30s floor.
        for shot in &mut draft.shots {
            shot.target_duration_seconds = 5.1667;
        }
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&shape_findings(&brief, &plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("30-60s window"), "{findings:?}");
        // The durations themselves are untouched — nothing was rounded to fit.
        assert_eq!(plan.shots[0].target_duration_seconds, 5.1667);

        // A duration off the model's menu is a model finding, never a coercion.
        let mut draft = good_draft();
        draft.shots[0].target_duration_seconds = 12.0;
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&model_entry(), ModelLane::Mlx)),
        ));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] targetDurationSeconds") && m.contains("menu")),
            "{findings:?}"
        );

        let mut draft = good_draft();
        for index in 0..10 {
            let mut extra = draft.shots[0].clone();
            extra.id = format!("SH1{index}0");
            draft.shots.push(extra);
        }
        let plan = draft_to_plan(&brief, &draft);
        assert!(
            messages(&shape_findings(&brief, &plan))
                .iter()
                .any(|m| m.contains("exceed the 8 this brief allows")),
            "{:?}",
            messages(&shape_findings(&brief, &plan))
        );
    }

    #[test]
    fn unsupported_conditioning_from_the_planner_is_refused_against_the_installed_entry() {
        let brief = brief();
        let mut draft = good_draft();
        draft.shots[1].conditioning = ShotConditioning {
            mode: "reference_to_video".to_owned(),
            first_frame_role: None,
            last_frame_role: None,
            reference_roles: vec!["courier".to_owned()],
            chain_from_shot_id: None,
        };
        draft.shots[2].negative_prompt = Some("blurry".to_owned());
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&model_entry(), ModelLane::Mlx)),
        ));
        assert!(
            findings.iter().any(
                |m| m.contains("[SH020] conditioning.mode") && m.contains("reference_to_video")
            ),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH020] conditioning.referenceRoles")
                    && m.contains("maxReferenceAssets")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH030] negativePrompt")),
            "{findings:?}"
        );

        // An unapproved role cannot be conditioned on, and leaves the shot unanchored.
        let mut draft = good_draft();
        draft.shots[0].conditioning = ShotConditioning {
            mode: "image_to_video".to_owned(),
            first_frame_role: Some("draft_look".to_owned()),
            last_frame_role: None,
            reference_roles: Vec::new(),
            chain_from_shot_id: None,
        };
        draft.shots[0].continuity_roles = Vec::new();
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&model_entry(), ModelLane::Mlx)),
        ));
        assert!(
            findings.iter().any(|m| m.contains("not approved")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] continuityRoles")),
            "{findings:?}"
        );
    }

    #[test]
    fn the_capability_envelope_reports_only_what_the_entry_declares() {
        let brief = brief();
        let caps = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        assert_eq!(
            caps.modes,
            vec!["text_to_video", "image_to_video", "first_last_frame"]
        );
        assert_eq!(caps.fps, Some(24));
        assert_eq!(caps.default_resolution.as_deref(), Some("576x320"));
        assert_eq!(caps.max_reference_images, 0);
        assert!(!caps.supports_negative_prompt);
        assert_eq!(caps.min_memory_gb, Some(64.0));
        let section = caps.as_prompt_section();
        assert!(section.contains("5.1667, 5.875, 14.375"), "{section}");
        assert!(section.contains("THIS CHECKPOINT HAS NONE"), "{section}");
        assert!(
            section.contains("Never write a negativePrompt"),
            "{section}"
        );
        // reference_to_video is not declared, so it is never offered.
        assert!(!section.contains("Allowed conditioning modes (use no others): text_to_video, image_to_video, first_last_frame, reference_to_video"));

        // A model WITH references says so instead.
        let mut entry = model_entry();
        entry["capabilities"] = json!(["text_to_video", "reference_to_video"]);
        entry["limits"]["maxReferenceAssets"] = json!(3);
        entry["video"] = json!({ "supportsNegativePrompt": true });
        let caps = capabilities_for(&brief.model, &entry, ModelLane::Mlx);
        assert_eq!(caps.max_reference_images, 3);
        let section = caps.as_prompt_section();
        assert!(section.contains("at most 3 reference roles"), "{section}");
        assert!(!section.contains("negativePrompt"), "{section}");
    }

    #[test]
    fn the_planner_request_carries_the_beats_roles_and_envelope_and_never_an_unapproved_role() {
        let brief = brief();
        let pack = pack();
        let caps = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        let request = build_planner_request(&brief, &pack, &caps);
        for beat in &brief.required_beats {
            assert!(request.contains(&beat.id), "{request}");
            assert!(request.contains(beat.summary.trim()), "{request}");
        }
        assert!(request.contains("courier (character)"), "{request}");
        assert!(request.contains("workshop_plate (plate)"), "{request}");
        assert!(
            !request.contains("draft_look"),
            "an unapproved role must never be offered to the planner"
        );
        assert!(request.contains("576x320"), "{request}");
        assert!(request.contains(PLAN_JSON_CONTRACT), "{request}");
        assert!(request.contains("at most 8 shots"), "{request}");
    }

    #[test]
    fn a_repair_request_returns_every_finding_and_restates_the_beats() {
        let brief = brief();
        let findings = vec![
            PlanDiagnostic::plan("shots", "required beat \"delivery\" is covered by no shot"),
            PlanDiagnostic::shot("SH010", "targetDurationSeconds", "6s is not on the menu"),
        ];
        let request = build_repair_request(&brief, "{\"shots\": []}", &findings, 1, 2);
        assert!(request.contains("Repair round 1 of 2"), "{request}");
        for finding in &findings {
            assert!(request.contains(&finding.to_string()), "{request}");
        }
        assert!(request.contains("arrival:"), "{request}");
        assert!(request.contains("{\"shots\": []}"), "{request}");
        assert!(request.contains(PLAN_JSON_CONTRACT), "{request}");
    }

    #[test]
    fn brief_findings_cover_versions_windows_beats_and_ceilings() {
        let mut value = brief_json();
        value["schemaVersion"] = json!(9);
        value["targetTotalSeconds"] = json!({ "min": 60.0, "max": 30.0 });
        value["requiredBeats"][1]["id"] = json!("arrival");
        value["maxShots"] = json!(2);
        let broken: ProductionBrief = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_brief(&broken));
        assert!(
            findings.iter().any(|m| m.contains("schema version 9")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("the window is empty")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("duplicate beat id")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("below the 3 required beats")),
            "{findings:?}"
        );
        assert!(validate_brief(&brief()).is_empty());
        // An unknown field in a brief is refused on parse, like every other document here.
        let mut value = brief_json();
        value["tone"] = json!("wistful");
        assert!(parse_brief(&value.to_string())
            .expect_err("unknown field refused")
            .contains("tone"));
    }
}
