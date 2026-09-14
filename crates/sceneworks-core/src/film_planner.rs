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
    parse_resolution, validate_all, ModelLane, PlanDiagnostic, PlanLimits, PlanModel, PlanSound,
    ProductionPlan, ReferencePack, Shot, ShotConditioning, ShotDependency, PLAN_SCHEMA_VERSION,
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
    /// Approved pack roles this beat is ABOUT: the subject, the prop the story turns on, whoever
    /// has to be on screen for the beat to have happened. The shots covering the beat must bind
    /// every one of them, so "the beat is covered" cannot be satisfied by a shot of the right
    /// length in the right place that leaves the parcel out of the delivery (sc-22713).
    ///
    /// Empty means "the brief makes no claim", which is the pre-sc-22713 behaviour and what a brief
    /// written before this field says.
    #[serde(default)]
    pub required_roles: Vec<String>,
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
        let mut roles = BTreeSet::new();
        for role in &beat.required_roles {
            if !crate::film_plan::is_safe_plan_id(role) {
                findings.push(PlanDiagnostic::plan(
                    format!("brief.requiredBeats[{index}].requiredRoles"),
                    format!("role {role:?} must be 1-64 characters of [A-Za-z0-9_-]"),
                ));
            } else if !roles.insert(role.as_str()) {
                findings.push(PlanDiagnostic::plan(
                    format!("brief.requiredBeats[{index}].requiredRoles"),
                    format!("role {role:?} is listed twice"),
                ));
            }
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
        // Which mode to REACH FOR, not merely which are legal. Without this the planner satisfies
        // the slot shape the cheapest way it can see — every shot first_last_frame with the same
        // plate at both ends, which asks each clip to finish exactly where it began (sc-22713).
        if self.modes.iter().any(|mode| mode == "first_last_frame") {
            lines.push(
                "Choosing a mode: text_to_video is the default and the right answer for most \
                 shots. Use image_to_video when a shot should BEGIN on a specific approved plate. \
                 Use first_last_frame only when you have TWO DIFFERENT approved roles for the \
                 first and the last frame — the same role in both slots is refused, because it \
                 asks the shot to end exactly where it started."
                    .to_owned(),
            );
        } else if self.modes.iter().any(|mode| mode == "image_to_video") {
            lines.push(
                "Choosing a mode: text_to_video is the default and the right answer for most \
                 shots. Use image_to_video when a shot should BEGIN on a specific approved plate."
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

/// Isolate the outermost `{ … }` span of a model reply, normalise the shapes a local planner
/// reliably gets slightly wrong, and parse the result strictly.
///
/// Strict where it matters: an unknown or misspelled field is still refused outright, because a
/// field the schema does not know is data the plan would silently lose. Lenient where the model's
/// mistake carries no ambiguity ([`normalize_draft`]).
///
/// A failure names the FIELD PATH (`shots[3].startState`) rather than a line and column. The
/// consumer of this message is a local 8B model being asked to correct itself: it never sees the
/// text it emitted as numbered lines, so "at line 10 column 20" is unusable and cost the sc-22713
/// smoke all three of its decodes.
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
    let mut value: Value = serde_json::from_str(&trimmed[start..=end])
        .map_err(|error| format!("the JSON does not parse: {error}"))?;
    normalize_draft(&mut value);
    serde_path_to_error::deserialize(value).map_err(|error| {
        let path = error.path().to_string();
        let inner = error.inner().to_string();
        if path.is_empty() || path == "." {
            inner
        } else {
            format!("{path}: {inner}")
        }
    })
}

/// The unambiguous shape repairs applied to a draft before it is parsed.
///
/// Each one exists because the local planner made it on a real run and there is exactly one thing
/// it could have meant, so bouncing it would cost a full decode to be told what a `match` arm can
/// say for free. Nothing here changes WHAT the plan says — only how it is spelled:
///
/// * `startState`/`endState` written as a role→description object, or as a list of phrases,
///   become one string. An object's keys are sorted (see [`flatten_to_prose`] for why explicitly),
///   so the same draft always normalises to the same plan and the plan's sha256 is reproducible.
/// * an optional field written as the empty string — `dialogue: ""`, `chainFromShotId: ""`,
///   `firstFrameRole: ""` — is REMOVED, which is what the contract asks for ("omit an optional
///   field rather than writing null, \"\" or a placeholder") and what the model meant. Left in
///   place, `chainFromShotId: ""` is a chain to a shot named "" and `firstFrameRole: ""` a
///   reference role that is not in any pack — two findings for one placeholder.
/// * a keyframe slot on a mode that takes none — `firstFrameRole` on a `text_to_video` shot — is
///   REMOVED. The mode is the shot's actual statement about its conditioning; the slots are
///   subordinate to it, and one the mode cannot use conditions nothing. Every draft of the
///   sc-22713 smoke filled both slots on every `text_to_video` shot and all three rounds were
///   refused for it and nothing else — the model was copying the shape of the contract's own
///   template, which now shows the slot-free form.
///
/// Anything else stays wrong and is reported. In particular a NON-EMPTY `referenceRoles` on a
/// keyframe mode is never dropped: unlike a keyframe slot, which names one frame, that is a list of
/// subjects the shot claims to be conditioned on, and silently deleting it would change what the
/// plan asks for.
fn normalize_draft(value: &mut Value) {
    let Some(shots) = value
        .get_mut("shots")
        .and_then(|shots| shots.as_array_mut())
    else {
        return;
    };
    for shot in shots {
        let Some(shot) = shot.as_object_mut() else {
            continue;
        };
        for field in ["startState", "endState", "beat", "framing", "prompt"] {
            if let Some(entry) = shot.get_mut(field) {
                if let Some(flattened) = flatten_to_prose(entry) {
                    *entry = Value::String(flattened);
                }
            }
        }
        for field in [
            "dialogue",
            "sound",
            "negativePrompt",
            "resolution",
            "beatId",
        ] {
            drop_if_blank(shot, field);
        }
        if let Some(conditioning) = shot
            .get_mut("conditioning")
            .and_then(|value| value.as_object_mut())
        {
            for field in ["firstFrameRole", "lastFrameRole", "chainFromShotId"] {
                drop_if_blank(conditioning, field);
            }
            let mode = conditioning
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if mode != "image_to_video" && mode != "first_last_frame" {
                conditioning.remove("firstFrameRole");
            }
            if mode != "first_last_frame" {
                conditioning.remove("lastFrameRole");
            }
            // An EMPTY list on a mode that takes no references is the same empty statement the
            // schema's own default is; a non-empty one is left to be reported.
            if conditioning
                .get("referenceRoles")
                .and_then(Value::as_array)
                .is_some_and(|roles| roles.is_empty())
            {
                conditioning.remove("referenceRoles");
            }
        }
    }
}

/// A string-valued field the model filled with a blank placeholder is removed outright. `beatId` is
/// deliberately included even though it is required: "" is not a beat id, and `beatId` missing is a
/// finding that names the field, where `beatId: ""` is one that names a beat nobody declared.
fn drop_if_blank(object: &mut Map<String, Value>, field: &str) {
    let blank = match object.get(field) {
        Some(Value::String(text)) => text.trim().is_empty(),
        Some(Value::Null) => true,
        _ => false,
    };
    if blank {
        object.remove(field);
    }
}

/// One line of prose for a value the schema wants as a string but the model wrote as an object or a
/// list. `None` for a value that is already a string (or anything else, which stays a type error).
///
/// An object's keys are sorted HERE rather than taken in `Map` order. `serde_json`'s map is a
/// `BTreeMap` or an `IndexMap` depending on whether anything in the build graph turns on
/// `preserve_order` — which this workspace's graph does — so relying on its iteration order would
/// make the normalised text, and with it the plan's sha256, depend on a transitive feature flag.
/// Sorting here is the only way the same draft normalises to the same plan in every build.
fn flatten_to_prose(value: &Value) -> Option<String> {
    match value {
        Value::Object(fields) => Some({
            // Collected through a BTreeMap so the order is key-sorted by construction.
            let entries: BTreeMap<&str, &Value> = fields
                .iter()
                .map(|(key, field)| (key.as_str(), field))
                .collect();
            entries
                .into_iter()
                .filter_map(|(key, field)| {
                    let text = match field {
                        Value::String(text) => text.trim().to_owned(),
                        other => other.to_string(),
                    };
                    (!text.is_empty()).then(|| format!("{key}: {text}"))
                })
                .collect::<Vec<_>>()
                .join("; ")
        }),
        Value::Array(items) => Some(
            items
                .iter()
                .map(|item| match item {
                    Value::String(text) => text.trim().to_owned(),
                    other => other.to_string(),
                })
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("; "),
        ),
        _ => None,
    }
}

/// Assemble a [`ProductionPlan`] from the brief and a parsed draft. Everything outside the shots —
/// id, version, title, model, limits — comes from the brief, so the planner cannot widen a limit or
/// swap the model.
///
/// The planner writes sound INTENT only (each shot's `dialogue` and `sound` prose, below); it never
/// places sound. Resolving intent into a placed bed or a [`Shot::dialogue_clip`] needs a reference
/// pack to resolve roles against, and a brief does not carry one — so the generated plan takes
/// [`PlanSound::default`] (no beds, generated clip audio muted, the policy that cannot double a
/// line) and leaves every clip unset, which `validate_plan_structure` and
/// `validate_plan_against_pack` both accept. Placing sound is an edit to the generated plan.
pub fn draft_to_plan(brief: &ProductionBrief, draft: &PlannerDraft) -> ProductionPlan {
    ProductionPlan {
        schema_version: PLAN_SCHEMA_VERSION,
        id: brief.id.clone(),
        version: brief.version,
        title: brief.title.clone(),
        synopsis: brief.synopsis.clone(),
        model: brief.model.clone(),
        limits: brief.limits.clone(),
        sound: PlanSound::default(),
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
                generated_audio: None,
                dialogue_clip: None,
                conditioning: shot.conditioning.clone(),
                seed: shot.seed,
                continuity_roles: shot.continuity_roles.clone(),
                // A declared chain IS a continuity dependency in sc-22711's vocabulary — "this
                // shot's start state is that shot's end state" — so a generated plan carries the
                // edge rather than leaving a planner-written film with nothing to flag when a take
                // it continues is replaced. The planner is not asked for edges: this is the one it
                // already stated, restated where `replace-take` reads it.
                depends_on: shot
                    .conditioning
                    .chain_from_shot_id
                    .iter()
                    .map(|shot_id| ShotDependency {
                        shot_id: shot_id.clone(),
                        kind: "continuity".to_owned(),
                        note: "declared by conditioning.chainFromShotId".to_owned(),
                    })
                    .collect(),
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
    // A hand edit can also hollow a beat out without deleting it — the shot stays, the subject
    // stops being named — so the roles are re-checked on the same pass as the beats.
    findings.extend(role_coverage_findings(brief, plan));
    findings
}

/// Every approved pack role one shot binds: the roles it declares it depicts, plus whatever its
/// conditioning slots name. A role bound in either place is on screen by the plan's own account.
fn bound_roles(shot: &Shot) -> BTreeSet<&str> {
    let mut roles: BTreeSet<&str> = shot
        .continuity_roles
        .iter()
        .map(String::as_str)
        .chain(shot.conditioning.reference_roles.iter().map(String::as_str))
        .collect();
    roles.extend(shot.conditioning.first_frame_role.as_deref());
    roles.extend(shot.conditioning.last_frame_role.as_deref());
    roles
}

/// Beat ROLE coverage: for every beat that declares `requiredRoles`, the shots covering that beat
/// must between them bind each one.
///
/// Beat coverage alone only asks that a beat has a shot. It cannot tell a delivery from a shot of
/// an empty workbench, which is exactly what the first real-LLM draft produced: the parcel bound on
/// two of six shots, the courier absent from their own departure (sc-22713). The brief is where
/// "this beat is ABOUT the parcel" belongs, because only the brief knows the story.
pub fn role_coverage_findings(
    brief: &ProductionBrief,
    plan: &ProductionPlan,
) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    for beat in &brief.required_beats {
        if beat.required_roles.is_empty() {
            continue;
        }
        let covering: Vec<&Shot> = plan
            .shots
            .iter()
            .filter(|shot| shot.beat_id.as_deref() == Some(beat.id.as_str()))
            .collect();
        if covering.is_empty() {
            // "No shot at all" is `coverage_findings`' report; do not say it twice.
            continue;
        }
        let bound: BTreeSet<&str> = covering.iter().flat_map(|shot| bound_roles(shot)).collect();
        let missing: Vec<&str> = beat
            .required_roles
            .iter()
            .map(String::as_str)
            .filter(|role| !bound.contains(role))
            .collect();
        if !missing.is_empty() {
            findings.push(PlanDiagnostic::shot(
                &covering[0].id,
                "continuityRoles",
                format!(
                    "beat {:?} is about {} — no shot covering it binds {}; name the missing \
                     role(s) in continuityRoles of the shot that shows them (and write them into \
                     that shot's prompt)",
                    beat.id,
                    beat.required_roles.join(", "),
                    missing.join(", ")
                ),
            ));
        }
    }
    findings
}

/// Findings about the brief read against the reference pack it will be planned with: a beat cannot
/// require a role the pack does not approve, because no draft could ever satisfy it. Checked before
/// the first decode, so an unsatisfiable brief is a refusal rather than `1 + rounds` decodes that
/// were never going to converge.
pub fn brief_pack_findings(brief: &ProductionBrief, pack: &ReferencePack) -> Vec<PlanDiagnostic> {
    let approved: BTreeSet<&str> = pack
        .references
        .iter()
        .filter(|entry| entry.approved)
        .map(|entry| entry.role.as_str())
        .collect();
    let mut findings = Vec::new();
    for (index, beat) in brief.required_beats.iter().enumerate() {
        for role in &beat.required_roles {
            if !approved.contains(role.as_str()) {
                findings.push(PlanDiagnostic::plan(
                    format!("brief.requiredBeats[{index}].requiredRoles"),
                    format!(
                        "beat {:?} requires role {role:?}, which reference pack {:?} does not \
                         approve (approved: {})",
                        beat.id,
                        pack.id,
                        approved.iter().copied().collect::<Vec<_>>().join(", ")
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
    findings.extend(role_coverage_findings(brief, plan));
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
        out.push_str(&format!("- {}: {}", beat.id, beat.summary.trim()));
        if !beat.required_roles.is_empty() {
            // Named as a requirement, not as colour: this is checked mechanically after the reply.
            out.push_str(&format!(
                " [this beat MUST show: {}. Name them in that shot's continuityRoles AND describe \
                 them in its prompt.]",
                beat.required_roles.join(", ")
            ));
        }
        out.push('\n');
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

/// The user turn for a repair round: what to change, the beats that must survive the change, the
/// draft being corrected, and the contract. Nothing is summarised — the planner is corrected with
/// the same text a human running `film-harness validate` would read.
///
/// The ORDER is load-bearing. The first version of this prompt led with the findings and then
/// handed over the whole rejected draft; on the real 8B planner all three rounds came back byte
/// for byte identical (sc-22713), because the last thing the model read was a complete, fluent
/// answer to the question and copying it is the likeliest continuation. So the draft is framed as
/// raw material rather than as an answer, and the findings are repeated after it as the last thing
/// read before the contract.
pub fn build_repair_request(
    brief: &ProductionBrief,
    previous_output: &str,
    findings: &[PlanDiagnostic],
    round: u32,
    max_rounds: u32,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# Repair round {round} of {max_rounds}\n\nYour previous answer was REJECTED. Below are \
         the changes it needs, the beats it must still cover, and the text to correct. Return the \
         WHOLE plan as one JSON object with those changes made. Returning the same answer again \
         fails the round: every finding names a field, and every field it names must come back \
         different.\n\n"
    ));
    out.push_str("# Changes to make\n\n");
    for finding in findings {
        out.push_str(&format!("- {finding}\n"));
    }
    out.push_str("\n# Beats the corrected plan must still cover (in order)\n\n");
    out.push_str(
        "Fixing a finding never removes a beat, shortens the film or drops a shot's subject.\n\n",
    );
    for beat in &brief.required_beats {
        out.push_str(&format!("- {}: {}", beat.id, beat.summary.trim()));
        if !beat.required_roles.is_empty() {
            out.push_str(&format!(" [MUST show: {}]", beat.required_roles.join(", ")));
        }
        out.push('\n');
    }
    out.push_str(
        "\n# The text to correct (this is a DRAFT with the faults listed above, not an answer)\n\n",
    );
    out.push_str(previous_output.trim());
    out.push_str("\n\n# Before you answer, re-read these and fix each one\n\n");
    for finding in findings {
        out.push_str(&format!("- {finding}\n"));
    }
    out.push_str("\n# Output contract\n\n");
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
      \"conditioning\": { \"mode\": \"<one of the allowed modes>\" },
      \"seed\": <an integer, optional>,
      \"continuityRoles\": [\"<the approved roles this shot depicts>\"]
    }
  ]
}

Rules:
- Shot ids ascend in tens: SH010, SH020, SH030 ...
- Every field name is spelled exactly as above. An extra or misspelled field is rejected outright.
- Omit an optional field rather than writing null, \"\" or a placeholder.
- beat, framing, prompt, startState and endState are STRINGS — one piece of prose each. Never an \
object, never a list, and never a role-by-role breakdown.
- startState and endState describe WHAT THE CAMERA SEES in the first and the last frame of this \
shot: who is in frame, where they are, and where the objects that matter are. They are different \
from each other — if they are not, the shot has no action in it.
- conditioning carries NOTHING BUT \"mode\" unless the mode itself takes a slot. A text_to_video \
shot's conditioning object is exactly { \"mode\": \"text_to_video\" } — writing firstFrameRole, \
lastFrameRole, referenceRoles or chainFromShotId on it is a fault, not extra detail. The four \
slot-bearing forms, and there are no others:
    image_to_video:      { \"mode\": \"image_to_video\", \"firstFrameRole\": \"<a role>\" }
    first_last_frame:    { \"mode\": \"first_last_frame\", \"firstFrameRole\": \"<a role>\", \
\"lastFrameRole\": \"<a DIFFERENT role>\" }
    reference_to_video:  { \"mode\": \"reference_to_video\", \"referenceRoles\": [\"<a role>\"] }
    any of the above, continuing the shot just before it: add \
\"chainFromShotId\": \"<the previous shot's id>\"
- A shot never carries keyframe roles and reference roles together; they are different \
conditioning tasks.
- first_last_frame needs TWO DIFFERENT roles. The same role in both slots is refused.
- chainFromShotId names the shot IMMEDIATELY BEFORE this one, or is omitted. It is never a \
substitute for continuityRoles: a chained shot still names the approved roles it depicts.
- Every shot needs at least one approved role in continuityRoles, and a shot lists every approved \
role that is on screen in it — the character, the prop the beat turns on, the location.

One filled shot, for shape only. It is from a DIFFERENT film: copy the spelling and the level of \
detail, never the content.

{
  \"id\": \"SH020\",
  \"beatId\": \"handover\",
  \"beat\": \"The mechanic hands the key across the counter and the customer takes it.\",
  \"framing\": \"Medium two-shot, eye level, locked off\",
  \"prompt\": \"A cramped garage office in hard noon light. A mechanic in an oil-stained shirt \
slides a brass key across a steel counter; the customer's hand closes around it and lifts it away. \
Dust turns in the light from the roller door behind them. The camera does not move.\",
  \"targetDurationSeconds\": 5.1667,
  \"startState\": \"The mechanic stands behind the counter with the brass key flat under their \
palm; the customer waits opposite with both hands at their sides.\",
  \"endState\": \"The counter is empty and the customer holds the brass key at chest height; the \
mechanic's hand is withdrawn.\",
  \"sound\": \"key scraping on steel, a compressor cycling somewhere off screen\",
  \"conditioning\": { \"mode\": \"text_to_video\" },
  \"continuityRoles\": [\"mechanic\", \"customer\", \"brass_key\"]
}";

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
    fn a_parse_failure_names_the_field_path_the_planner_can_act_on() {
        // The real sc-22713 smoke failure: `startState` as an object on a shot whose fields are
        // otherwise fine. The 8B planner never sees its answer as numbered lines, so "line 10
        // column 20" told it nothing and all three decodes came back identical.
        let error = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": {"mode": "text_to_video"}},
                {"id": "SH020", "beatId": "delivery", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s",
                "endState": 12, "conditioning": {"mode": "text_to_video"}}]}"#,
        )
        .expect_err("a wrong type is refused");
        assert!(
            error.contains("shots[1].endState") && error.contains("expected a string"),
            "{error}"
        );
        assert!(!error.contains("line 1"), "{error}");
    }

    #[test]
    fn a_structurally_close_draft_is_normalised_rather_than_bounced() {
        // Verbatim shapes from the refused real-LLM draft (run2-planner-refused): the states as
        // role -> description objects, and `dialogue`/`chainFromShotId` as empty-string
        // placeholders the contract asks to be omitted.
        let draft = parse_planner_output(
            r#"{"shots": [{
                "id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f", "prompt": "p",
                "targetDurationSeconds": 5.1667,
                "startState": {"workshop_plate": "empty workshop", "courier": "not yet in frame"},
                "endState": ["the courier fills the doorway", "the parcel is against their chest"],
                "dialogue": "", "sound": "  ",
                "conditioning": {
                    "mode": "text_to_video", "firstFrameRole": "", "lastFrameRole": "",
                    "referenceRoles": [], "chainFromShotId": ""
                },
                "continuityRoles": ["courier"]
            }]}"#,
        )
        .expect("a near-miss draft parses");
        let shot = &draft.shots[0];
        // Deterministic: keys SORTED, joined into one line. Not `Map` order — `serde_json`'s map is
        // a BTreeMap or an IndexMap depending on whether anything in the build graph enables
        // `preserve_order`, so the same draft would otherwise normalise to a different plan (and a
        // different plan sha256) in a different build. The reversed spelling below is the same
        // object written the other way round and must normalise identically.
        assert_eq!(
            shot.start_state,
            "courier: not yet in frame; workshop_plate: empty workshop"
        );
        let reversed = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667,
                "startState": {"workshop_plate": "empty workshop", "courier": "not yet in frame"},
                "endState": "e", "conditioning": {"mode": "text_to_video"}}]}"#,
        )
        .expect("parses");
        let forwards = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667,
                "startState": {"courier": "not yet in frame", "workshop_plate": "empty workshop"},
                "endState": "e", "conditioning": {"mode": "text_to_video"}}]}"#,
        )
        .expect("parses");
        assert_eq!(reversed.shots[0].start_state, forwards.shots[0].start_state);
        assert_eq!(
            shot.end_state,
            "the courier fills the doorway; the parcel is against their chest"
        );
        // A blank placeholder is an omission, not a value: `chainFromShotId: ""` would otherwise be
        // a chain to a shot named "" and `firstFrameRole: ""` a role no pack can contain.
        assert_eq!(shot.dialogue, None);
        assert_eq!(shot.sound, None);
        assert_eq!(shot.conditioning.chain_from_shot_id, None);
        assert_eq!(shot.conditioning.first_frame_role, None);
        assert_eq!(shot.conditioning.last_frame_role, None);
        // Normalisation never invents: a state object with nothing in it stays empty, and the
        // ordinary structural validator reports it as the missing field it is.
        let draft = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": {}, "endState": "e",
                "conditioning": {"mode": "text_to_video"}}]}"#,
        )
        .expect("an empty object flattens");
        assert_eq!(draft.shots[0].start_state, "");

        // The fault EVERY draft of the sc-22713 smoke made on EVERY shot, through all three
        // rounds: keyframe slots filled on a text_to_video shot. The mode is what the shot says
        // about its conditioning; a slot the mode cannot use conditions nothing.
        let draft = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": {
                    "mode": "text_to_video", "firstFrameRole": "workshop_plate",
                    "lastFrameRole": "courier", "referenceRoles": []
                },
                "continuityRoles": ["courier"]}]}"#,
        )
        .expect("a text_to_video shot with filled keyframe slots parses");
        assert_eq!(draft.shots[0].conditioning.first_frame_role, None);
        assert_eq!(draft.shots[0].conditioning.last_frame_role, None);

        // A slot the mode DOES take is untouched, and so is the second slot's absence.
        let draft = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": {
                    "mode": "image_to_video", "firstFrameRole": "workshop_plate",
                    "lastFrameRole": "courier"
                },
                "continuityRoles": ["courier"]}]}"#,
        )
        .expect("an image_to_video shot parses");
        assert_eq!(
            draft.shots[0].conditioning.first_frame_role.as_deref(),
            Some("workshop_plate")
        );
        assert_eq!(draft.shots[0].conditioning.last_frame_role, None);

        // A NON-EMPTY referenceRoles list is a claim about what the shot depicts, so it survives
        // normalisation and is reported by the validator instead of being deleted.
        let draft = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": { "mode": "text_to_video", "referenceRoles": ["courier"] },
                "continuityRoles": ["courier"]}]}"#,
        )
        .expect("parses");
        assert_eq!(draft.shots[0].conditioning.reference_roles, ["courier"]);
        let plan = draft_to_plan(&brief(), &draft);
        assert!(
            messages(&crate::film_plan::validate_plan_structure(&plan))
                .iter()
                .any(|message| message.contains("does not take reference roles")),
            "the surviving list is reported"
        );
    }

    #[test]
    fn a_beat_that_names_required_roles_is_not_covered_by_a_shot_that_omits_them() {
        let mut brief = brief();
        brief.required_beats[1].required_roles =
            vec!["red_parcel".to_owned(), "courier".to_owned()];
        let draft = good_draft();
        let mut plan = draft_to_plan(&brief, &draft);
        // The shot for the beat is present and names one of the two roles.
        plan.shots[1].continuity_roles = vec!["courier".to_owned()];
        let findings = messages(&role_coverage_findings(&brief, &plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("beat \"delivery\"") && findings[0].contains("red_parcel"),
            "{findings:?}"
        );
        // Only the role that is actually missing is reported as missing; `courier` appears in the
        // message only as part of what the beat is about.
        assert!(findings[0].contains("binds red_parcel;"), "{findings:?}");

        // Bound in a conditioning slot rather than continuityRoles still counts as on screen.
        plan.shots[1].conditioning.mode = "image_to_video".to_owned();
        plan.shots[1].conditioning.first_frame_role = Some("red_parcel".to_owned());
        assert!(role_coverage_findings(&brief, &plan).is_empty());

        // A beat with no declared roles makes no claim, which is what every pre-sc-22713 brief is.
        brief.required_beats[1].required_roles.clear();
        plan.shots[1].conditioning.first_frame_role = None;
        plan.shots[1].conditioning.mode = "text_to_video".to_owned();
        assert!(role_coverage_findings(&brief, &plan).is_empty());
    }

    #[test]
    fn a_brief_cannot_require_a_role_the_pack_does_not_approve() {
        let clean = brief();
        let mut brief = brief();
        brief.required_beats[0].required_roles = vec![
            "courier".to_owned(),
            "draft_look".to_owned(),
            "wolf".to_owned(),
        ];
        let findings = messages(&brief_pack_findings(&brief, &pack()));
        assert_eq!(findings.len(), 2, "{findings:?}");
        // `draft_look` is in the pack but unapproved, so it is as unusable as one that is absent.
        assert!(
            findings.iter().any(|m| m.contains("\"draft_look\"")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("\"wolf\"")),
            "{findings:?}"
        );
        assert!(brief_pack_findings(&clean, &pack()).is_empty());
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
