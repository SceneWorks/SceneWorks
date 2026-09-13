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
// Longest prompt the generation routes accept: the route's own declaration, not a copy of it, so
// this validator cannot bless a prompt length the enqueue would refuse (sc-22710).
use crate::MAX_PROMPT_CHARS;

/// Schema version of [`ProductionPlan`] documents this module reads and writes.
pub const PLAN_SCHEMA_VERSION: u32 = 1;
/// Schema version of [`ReferencePack`] documents this module reads and writes.
pub const REFERENCE_PACK_SCHEMA_VERSION: u32 = 1;
/// Schema version of [`RunRecord`] documents this module writes.
pub const RUN_RECORD_SCHEMA_VERSION: u32 = 1;

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

/// Sound kinds a pack entry may declare (sc-22712).
///
/// Deliberately a SEPARATE list from [`REFERENCE_KINDS`] over a separate `sound` array, rather
/// than an extra reference kind: a reference is something a shot can be conditioned on, and an
/// audio file is not. Keeping them apart means `referenceRoles: ["main_theme"]` is a structural
/// error the validator can name instead of a request the model would have to refuse.
pub const SOUND_KINDS: &[&str] = &["dialogue", "ambience", "music", "sfx"];

/// Audio extensions a pack's sound entry may carry. Import normalises every one of them to
/// PCM-16 WAV (`ProjectStore::import_asset`), so this list only has to cover what a human is
/// likely to have on disk.
const SOUND_AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus"];

/// Widest gain the plan admits on a bus, a bed or a line. Matches the timeline's own per-track
/// ceiling (`project_store::validate_timeline_track`), so a plan cannot express a level the
/// timeline would then refuse to persist.
const MAX_SOUND_GAIN: f64 = 4.0;

/// Image extensions a reference file may carry; the import route accepts any `image/*` but the
/// pack is checked in, so the list stays explicit.
const REFERENCE_IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

/// Characters a reference file's basename may use. The name is interpolated into a multipart
/// `Content-Disposition` header on import, so anything outside this set — a CR/LF above all — is
/// refused here rather than sanitized downstream (sc-22710).
fn is_safe_reference_basename(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

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
    /// How the sequence's sound is assembled (sc-22712). Optional: a plan that says nothing about
    /// sound gets [`PlanSound::default`], which places no beds and mutes generated clip audio.
    #[serde(default)]
    pub sound: PlanSound,
    pub shots: Vec<Shot>,
}

/// What a generated take's OWN audio does in the export.
///
/// The default is [`GeneratedAudio::Mute`], and that choice is the doubling policy: a take whose
/// model spoke the line and a recorded dialogue clip for the same beat would otherwise both land
/// in the mix, with nothing in the documents saying which one was meant. Including it is always an
/// explicit act — at the run level, at the shot level, or both.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratedAudio {
    /// Mix the take's own audio alongside whatever else is placed.
    Include,
    /// Drop the take's own audio; only placed sound is heard.
    #[default]
    Mute,
}

impl GeneratedAudio {
    /// The spelling the timeline's `generatedAudio` field uses.
    pub fn as_timeline_str(self) -> &'static str {
        match self {
            Self::Include => "include",
            Self::Mute => "mute",
        }
    }
}

/// The sequence's sound design: one default for generated clip audio plus three independently
/// controlled buses.
///
/// Dialogue is placed per shot (each line sits against the beat it belongs to), while ambience and
/// music are placed ONCE across the whole sequence. That asymmetry is the point: a bed that is
/// re-placed per shot restarts at every cut, and "continuous sound across intentional cuts" is
/// precisely what this POC has to demonstrate.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlanSound {
    /// Run-level default for every shot that does not override it.
    #[serde(default)]
    pub generated_audio: GeneratedAudio,
    /// Bus settings for the per-shot dialogue track.
    #[serde(default)]
    pub dialogue: SoundBus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ambience: Option<SoundBed>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub music: Option<SoundBed>,
}

/// Gain and mute for one audio bus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SoundBus {
    #[serde(default = "default_gain")]
    pub gain: f64,
    #[serde(default)]
    pub muted: bool,
}

impl Default for SoundBus {
    fn default() -> Self {
        Self {
            gain: default_gain(),
            muted: false,
        }
    }
}

/// A continuous bed placed once across the sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SoundBed {
    /// Role resolving against the pack's `sound` entries.
    pub role: String,
    #[serde(default = "default_gain")]
    pub gain: f64,
    #[serde(default)]
    pub muted: bool,
    /// Where on the sequence the bed starts. Beds normally run from the head, so this defaults to
    /// zero; it exists so music can come in after the opening beat.
    #[serde(default)]
    pub start_seconds: f64,
    /// Where in the SOURCE file the bed starts.
    #[serde(default)]
    pub source_in_seconds: f64,
    #[serde(default)]
    pub fade_in_seconds: f64,
    #[serde(default)]
    pub fade_out_seconds: f64,
}

/// One dialogue line placed against a shot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DialogueClip {
    /// Role resolving against the pack's `sound` entries; must be a `dialogue` entry.
    pub role: String,
    /// Offset from the START OF THE SHOT, not from the head of the sequence — so a line stays
    /// against its beat when earlier shots are trimmed or reordered.
    #[serde(default)]
    pub offset_seconds: f64,
    #[serde(default = "default_gain")]
    pub gain: f64,
    #[serde(default)]
    pub source_in_seconds: f64,
    /// How much of the clip to play. Defaults to the whole file (measured on import).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    #[serde(default)]
    pub fade_in_seconds: f64,
    #[serde(default)]
    pub fade_out_seconds: f64,
}

fn default_gain() -> f64 {
    1.0
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
    /// Override the run-level generated-audio policy for this shot (sc-22712).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_audio: Option<GeneratedAudio>,
    /// The dialogue line placed against this shot, if any. `dialogue` above is the INTENT (what is
    /// said); this is the actual audio, resolved from the pack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialogue_clip: Option<DialogueClip>,
    pub conditioning: ShotConditioning,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Roles the shot depicts (traceability only; they are not sent to the model). Every entry
    /// must exist in the pack.
    #[serde(default)]
    pub continuity_roles: Vec<String>,
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
    /// Approved audio, kept beside the approved images so a sequence's sound is as addressable and
    /// as versioned as its pictures (sc-22712). Optional, so a pack written before sound existed
    /// still reads.
    #[serde(default)]
    pub sound: Vec<SoundEntry>,
}

/// One approved audio file the plan's sound roles resolve against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SoundEntry {
    /// Role name (`[A-Za-z0-9_-]{1,64}`), unique within the pack's sound entries.
    pub role: String,
    /// One of [`SOUND_KINDS`]. A role may only be placed on the bus its kind names.
    pub kind: String,
    /// Audio path relative to the pack document's directory.
    pub file: String,
    #[serde(default)]
    pub description: String,
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
    if plan.schema_version != PLAN_SCHEMA_VERSION {
        findings.push(PlanDiagnostic::plan(
            "schemaVersion",
            format!(
                "unsupported plan schema version {} (this build reads {PLAN_SCHEMA_VERSION})",
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
    findings.extend(validate_plan_sound(&plan.sound));
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
    findings
}

/// A gain, a fade or an offset must be a real, non-negative number inside the range the timeline
/// can persist. A `NaN` here would survive every comparison below and arrive at ffmpeg as the
/// string `NaN`, which is the sort of finding that is only cheap while it is still in the document.
fn validate_sound_number(
    findings: &mut Vec<PlanDiagnostic>,
    field: &str,
    label: &str,
    value: f64,
    max: f64,
) {
    if !value.is_finite() || value < 0.0 || value > max {
        findings.push(PlanDiagnostic::plan(
            field.to_owned(),
            format!("{label} {value} must be a finite number in 0..={max}"),
        ));
    }
}

fn validate_sound_bus(findings: &mut Vec<PlanDiagnostic>, field: &str, bus: &SoundBus) {
    validate_sound_number(
        findings,
        &format!("{field}.gain"),
        "gain",
        bus.gain,
        MAX_SOUND_GAIN,
    );
}

fn validate_sound_bed(findings: &mut Vec<PlanDiagnostic>, field: &str, bed: &SoundBed) {
    if !is_safe_plan_id(&bed.role) {
        findings.push(PlanDiagnostic::plan(
            format!("{field}.role"),
            format!(
                "sound role {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                bed.role
            ),
        ));
    }
    validate_sound_number(
        findings,
        &format!("{field}.gain"),
        "gain",
        bed.gain,
        MAX_SOUND_GAIN,
    );
    validate_sound_number(
        findings,
        &format!("{field}.startSeconds"),
        "start",
        bed.start_seconds,
        f64::MAX,
    );
    validate_sound_number(
        findings,
        &format!("{field}.sourceInSeconds"),
        "source in",
        bed.source_in_seconds,
        f64::MAX,
    );
    validate_sound_number(
        findings,
        &format!("{field}.fadeInSeconds"),
        "fade in",
        bed.fade_in_seconds,
        60.0,
    );
    validate_sound_number(
        findings,
        &format!("{field}.fadeOutSeconds"),
        "fade out",
        bed.fade_out_seconds,
        60.0,
    );
}

/// Structural findings on the plan's sound block alone (sc-22712).
fn validate_plan_sound(sound: &PlanSound) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    validate_sound_bus(&mut findings, "sound.dialogue", &sound.dialogue);
    if let Some(bed) = &sound.ambience {
        validate_sound_bed(&mut findings, "sound.ambience", bed);
    }
    if let Some(bed) = &sound.music {
        validate_sound_bed(&mut findings, "sound.music", bed);
    }
    findings
}

/// Structural findings on one shot's dialogue clip.
fn validate_dialogue_clip(shot_id: &str, clip: &DialogueClip) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if !is_safe_plan_id(&clip.role) {
        findings.push(PlanDiagnostic::shot(
            shot_id,
            "dialogueClip.role",
            format!(
                "sound role {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                clip.role
            ),
        ));
    }
    let mut plan_findings = Vec::new();
    validate_sound_number(
        &mut plan_findings,
        "dialogueClip.gain",
        "gain",
        clip.gain,
        MAX_SOUND_GAIN,
    );
    validate_sound_number(
        &mut plan_findings,
        "dialogueClip.offsetSeconds",
        "offset",
        clip.offset_seconds,
        f64::MAX,
    );
    validate_sound_number(
        &mut plan_findings,
        "dialogueClip.sourceInSeconds",
        "source in",
        clip.source_in_seconds,
        f64::MAX,
    );
    validate_sound_number(
        &mut plan_findings,
        "dialogueClip.fadeInSeconds",
        "fade in",
        clip.fade_in_seconds,
        60.0,
    );
    validate_sound_number(
        &mut plan_findings,
        "dialogueClip.fadeOutSeconds",
        "fade out",
        clip.fade_out_seconds,
        60.0,
    );
    if let Some(duration) = clip.duration_seconds {
        if !duration.is_finite() || duration <= 0.0 {
            plan_findings.push(PlanDiagnostic::plan(
                "dialogueClip.durationSeconds",
                format!("duration {duration} must be a finite number > 0"),
            ));
        }
    }
    findings.extend(
        plan_findings
            .into_iter()
            .map(|finding| PlanDiagnostic::shot(shot_id, finding.field, finding.message)),
    );
    findings
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
    if let Some(clip) = &shot.dialogue_clip {
        findings.extend(validate_dialogue_clip(id, clip));
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
        let basename = file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
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
        } else if entry.file.contains(['\r', '\n']) || !is_safe_reference_basename(basename) {
            // The basename is interpolated into the multipart `Content-Disposition` header the
            // import posts, so a CR/LF in it injects multipart headers. sc-22713 generates these
            // documents, so the charset is enforced here rather than trusted.
            findings.push(PlanDiagnostic::plan(
                format!("{field}.file"),
                format!(
                    "file {:?} must have a 1-128 character [A-Za-z0-9._-] basename (it is sent as \
                     a multipart filename)",
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
    // Sound entries live in their own namespace: a role may be an image OR a sound, never both,
    // because the two are placed through different slots and a collision would make
    // `referenceRoles: ["theme"]` resolve to something a video model cannot take.
    let mut heard = BTreeSet::new();
    for (index, entry) in pack.sound.iter().enumerate() {
        let field = format!("referencePack.sound[{index}]");
        if !is_safe_plan_id(&entry.role) {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.role"),
                format!(
                    "role {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                    entry.role
                ),
            ));
        } else if !heard.insert(entry.role.as_str()) {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.role"),
                format!("duplicate sound role {:?}", entry.role),
            ));
        } else if seen.contains(entry.role.as_str()) {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.role"),
                format!(
                    "role {:?} is already a reference role; sound and reference roles share one \
                     namespace so a role always names one kind of thing",
                    entry.role
                ),
            ));
        }
        if !SOUND_KINDS.contains(&entry.kind.as_str()) {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.kind"),
                format!(
                    "unknown sound kind {:?}; expected one of {}",
                    entry.kind,
                    SOUND_KINDS.join(", ")
                ),
            ));
        }
        let file = Path::new(&entry.file);
        let extension_ok = file
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                SOUND_AUDIO_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str())
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
                    "file {:?} must be audio ({})",
                    entry.file,
                    SOUND_AUDIO_EXTENSIONS.join(", ")
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
    findings.extend(validate_sound_against_pack(plan, pack));
    findings
}

/// Every sound role the plan places must exist in the pack AND be the kind the bus it is placed on
/// expects (sc-22712).
///
/// The kind check is not pedantry: `ambience.role` pointing at a `dialogue` entry is a plan that
/// will run, export, and sound wrong, and the cost of finding that out is a whole GPU render. A
/// role that names the wrong kind is refused here, before the first job exists.
fn validate_sound_against_pack(plan: &ProductionPlan, pack: &ReferencePack) -> Vec<PlanDiagnostic> {
    let sound: BTreeMap<&str, &SoundEntry> = pack
        .sound
        .iter()
        .map(|entry| (entry.role.as_str(), entry))
        .collect();
    let known = || {
        if sound.is_empty() {
            "none".to_owned()
        } else {
            sound.keys().copied().collect::<Vec<_>>().join(", ")
        }
    };
    let mut findings = Vec::new();
    let mut check = |shot_id: Option<&str>, field: &str, role: &str, expected: &str| {
        let finding = match sound.get(role) {
            None => Some(format!(
                "sound role {role:?} is not in reference pack {:?} (sound roles: {})",
                pack.id,
                known()
            )),
            Some(entry) if entry.kind != expected => Some(format!(
                "sound role {role:?} is a {:?} entry but it is placed on the {expected} bus",
                entry.kind
            )),
            Some(_) => None,
        };
        if let Some(message) = finding {
            findings.push(match shot_id {
                Some(id) => PlanDiagnostic::shot(id, field, message),
                None => PlanDiagnostic::plan(field, message),
            });
        }
    };
    if let Some(bed) = &plan.sound.ambience {
        check(None, "sound.ambience.role", &bed.role, "ambience");
    }
    if let Some(bed) = &plan.sound.music {
        check(None, "sound.music.role", &bed.role, "music");
    }
    for shot in &plan.shots {
        if let Some(clip) = &shot.dialogue_clip {
            check(Some(&shot.id), "dialogueClip.role", &clip.role, "dialogue");
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
    for (index, entry) in pack.sound.iter().enumerate() {
        let path = pack_dir.join(&entry.file);
        match std::fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {}
            Ok(_) => findings.push(PlanDiagnostic::plan(
                format!("referencePack.sound[{index}].file"),
                format!(
                    "sound {:?}: {} is empty or not a file",
                    entry.role,
                    path.display()
                ),
            )),
            Err(error) => findings.push(PlanDiagnostic::plan(
                format!("referencePack.sound[{index}].file"),
                format!(
                    "sound {:?}: {} is missing ({error})",
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
    /// The lane a host running `platform` (an `std::env::consts::OS` spelling, as
    /// `GET /api/v1/host-capabilities` reports it) renders video on: MLX on macOS, candle
    /// everywhere else.
    ///
    /// The API host is the one that matters, not the client: a harness run with `--api` pointed at
    /// another machine would otherwise check the wrong lane's `minMemoryGb` (mlx 64 vs candle 43
    /// for `minimax_h3`) — sc-22710.
    pub fn for_platform(platform: &str) -> Self {
        if platform == "macos" {
            Self::Mlx
        } else {
            Self::Candle
        }
    }

    /// The lane THIS process's host renders video on. Only a fallback for a host that reports no
    /// platform; prefer [`ModelLane::for_platform`] with the API's reported platform.
    pub fn for_current_platform() -> Self {
        Self::for_platform(std::env::consts::OS)
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
    /// At least one selected shot did not render (or the export failed) within its limits.
    Failed,
}

/// Outcome of one shot within a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShotOutcome {
    Rendered,
    Failed,
    TimedOut,
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
    /// The RENDER host's platform, as `GET /api/v1/host-capabilities` reports it — not the
    /// harness client's, which may be a different machine entirely (sc-22710).
    pub platform: String,
    /// The render host's CPU architecture when the harness runs on it, `"unknown"` when the API
    /// reports a different platform than this process (the host-capabilities route reports no
    /// arch, and the client's would be a fabrication).
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
    /// Whether the pack approved this reference for conditioning. Unapproved entries are still
    /// imported (so the record can point at them) but carry a distinct tag and are never resolved
    /// into a shot's conditioning slots — AC1 is about APPROVED references staying addressable, so
    /// the record has to be able to tell them apart (sc-22710).
    #[serde(default = "default_true")]
    pub approved: bool,
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
    /// The generated-audio policy this shot resolved to — its own override if it declared one,
    /// otherwise the run-level default (sc-22712). Recorded because it is what the export obeyed.
    #[serde(default)]
    pub generated_audio: GeneratedAudio,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    pub status: String,
    pub started_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    pub elapsed_seconds: f64,
    /// Observed peak memory as a percentage of the render host's memory, when whichever source
    /// supplied the peak expressed one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_gpu_memory_pct: Option<f64>,
    /// Observed peak memory in GiB — the number compared against `limits.maxMemoryGb`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_memory_gb: Option<f64>,
    /// Which signal the peak came from (`metrics.peakMemoryBytes`, `metrics.peakMemoryPct`, or
    /// `job.peakGpuMemoryPct`), so a record says what its memory claim rests on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_memory_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub take: Option<TakeRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShotRunRecord {
    pub shot_id: String,
    pub outcome: ShotOutcome,
    pub intended: IntendedState,
    #[serde(default)]
    pub conditioning_assets: ConditioningAssets,
    #[serde(default)]
    pub attempts: Vec<AttemptRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineItemRecord {
    /// The shot this item belongs to. `None` for a sequence-level bed, which belongs to the
    /// sequence rather than to any one shot — that is exactly what makes it continuous across cuts
    /// (sc-22712).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shot_id: Option<String>,
    pub item_id: String,
    pub asset_id: String,
    pub timeline_start: f64,
    pub timeline_end: f64,
    /// Source range the item takes, so a trim is legible in the record without re-reading the
    /// timeline document.
    #[serde(default)]
    pub source_in: f64,
    #[serde(default)]
    pub source_out: f64,
    /// Effective gain (track gain times item volume) and the fades applied to this item.
    #[serde(default = "default_gain")]
    pub gain: f64,
    #[serde(default)]
    pub fade_in_seconds: f64,
    #[serde(default)]
    pub fade_out_seconds: f64,
    /// The RESOLVED generated-audio policy for a picture item — what the export actually did, not
    /// what the plan asked for. `None` on a placed sound clip, which has no generated audio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_audio: Option<GeneratedAudio>,
}

/// One track of the assembled sequence, with the bus controls that were in force.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineTrackRecord {
    pub track_id: String,
    /// `video` or `audio`, as the timeline document spells it.
    pub kind: String,
    /// `picture`, `dialogue`, `ambience` or `music`.
    pub role: String,
    pub gain: f64,
    pub muted: bool,
    pub items: Vec<TimelineItemRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineRecord {
    pub timeline_id: String,
    pub name: String,
    /// The aspect ratio the timeline was CREATED at. The route admits only `16:9` / `9:16` /
    /// `1:1`, so a take of any other shape is coerced to the nearest of those and letterboxed by
    /// the export; `source_aspect_ratio` records what the takes actually are (sc-22710).
    pub aspect_ratio: String,
    /// The takes' own reduced aspect ratio (e.g. `9:5` for 576x320).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_aspect_ratio: Option<String>,
    /// Geometry of the first rendered take the timeline was sized from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_height: Option<u32>,
    pub fps: u32,
    /// Length of the assembled picture. The export is exactly this long.
    #[serde(default)]
    pub duration_seconds: f64,
    /// The picture track's items, in cut order. Kept as its own field (rather than only inside
    /// `tracks`) because it is the sequence: shot -> take -> position.
    ///
    /// Read back from the saved timeline, never from the harness's intent, so the record cannot
    /// claim items the project does not hold.
    pub items: Vec<TimelineItemRecord>,
    /// Every track, picture and sound, with its bus controls (sc-22712).
    #[serde(default)]
    pub tracks: Vec<TimelineTrackRecord>,
    /// The run-level generated-audio default every shot inherited unless it overrode it.
    #[serde(default)]
    pub generated_audio_default: GeneratedAudio,
    /// Edits applied to the assembled sequence after the initial assembly, oldest first.
    #[serde(default)]
    pub edits: Vec<TimelineEditRecord>,
}

/// One trim / reorder / take-replacement applied to the assembled sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineEditRecord {
    /// `trim`, `reorder` or `replace_take`.
    pub kind: String,
    pub applied_at: String,
    /// Human-readable description of what changed, e.g. `SH010 source range 0.000..2.500`.
    pub detail: String,
    /// Duration of the sequence after the edit.
    pub duration_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportRecord {
    pub job_id: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// The production record of one run: shot -> attempt -> job -> asset, the timeline, the export,
/// and the observed model/backend/hardware. Written at the end (and on refusal), so a partial run
/// still leaves a diagnosable document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRecord {
    pub schema_version: u32,
    pub run_id: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    pub outcome: RunOutcome,
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
    /// Sound files the run imported, with the same shape as `references` (sc-22712).
    #[serde(default)]
    pub sound: Vec<ReferenceAssetRecord>,
    #[serde(default)]
    pub shots: Vec<ShotRunRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline: Option<TimelineRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<ExportRecord>,
    #[serde(default)]
    pub diagnostics: Vec<PlanDiagnostic>,
    pub elapsed_seconds: f64,
}

impl RunRecord {
    /// Serialize with stable key order for diffs.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({}))
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
    fn a_reference_filename_that_could_inject_multipart_headers_is_refused() {
        let mut value = pack_json();
        value["references"][0]["file"] = json!("references/plate\r\nX-Injected: 1.png");
        let pack: ReferencePack = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_reference_pack(&pack));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("multipart filename") && m.contains("basename")),
            "{findings:?}"
        );
        // A quote, a semicolon or a space in the basename is the same class of problem.
        let mut value = pack_json();
        value["references"][0]["file"] = json!("references/a\"b; c.png");
        let pack: ReferencePack = serde_json::from_value(value).unwrap();
        assert!(!validate_reference_pack(&pack).is_empty());
        // The shipped shape — a subdirectory plus an ordinary name — stays clean.
        assert!(validate_reference_pack(&self::pack()).is_empty());
    }

    #[test]
    fn candle_lane_reads_the_candle_memory_minimum() {
        let mut entry = model_entry();
        entry.insert("candle".to_owned(), json!({ "minMemoryGb": 43 }));
        assert_eq!(model_min_memory_gb(&entry, ModelLane::Candle), Some(43.0));
        assert_eq!(model_min_memory_gb(&entry, ModelLane::Mlx), Some(64.0));
        // The lane follows the RENDER host's platform, not the process reading the plan.
        assert_eq!(ModelLane::for_platform("macos"), ModelLane::Mlx);
        assert_eq!(ModelLane::for_platform("linux"), ModelLane::Candle);
        assert_eq!(ModelLane::for_platform("windows"), ModelLane::Candle);
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

    // ---------------------------------------------------------------------------------------
    // Sound (sc-22712)
    // ---------------------------------------------------------------------------------------

    /// A pack and plan carrying sound, for the checks below.
    fn sound_pack_json() -> Value {
        let mut pack = pack_json();
        pack["sound"] = json!([
            { "role": "courier_line", "kind": "dialogue", "file": "sound/courier_line.wav" },
            { "role": "room_tone", "kind": "ambience", "file": "sound/room_tone.wav" },
            { "role": "theme", "kind": "music", "file": "sound/theme.mp3" }
        ]);
        pack
    }

    fn sound_plan_json() -> Value {
        let mut plan = plan_json();
        plan["sound"] = json!({
            "generatedAudio": "mute",
            "dialogue": { "gain": 1.0 },
            "ambience": { "role": "room_tone", "gain": 0.3, "fadeInSeconds": 1.0 },
            "music": { "role": "theme", "gain": 0.2 }
        });
        plan["shots"][1]["dialogueClip"] = json!({ "role": "courier_line", "offsetSeconds": 0.5 });
        plan
    }

    /// A plan that says nothing about sound is still a valid plan, and its policy is `mute`.
    ///
    /// This is the compatibility claim for every plan written for sc-22710: adding sound must not
    /// make an existing document unreadable, and the absence of a policy must resolve to the one
    /// that cannot double a line.
    #[test]
    fn a_plan_without_a_sound_block_defaults_to_muting_generated_audio() {
        let plan: ProductionPlan = serde_json::from_value(plan_json()).expect("plan parses");
        assert_eq!(plan.sound.generated_audio, GeneratedAudio::Mute);
        assert!(plan.sound.ambience.is_none() && plan.sound.music.is_none());
        assert_eq!(plan.sound.dialogue.gain, 1.0);
        assert!(!plan.sound.dialogue.muted);
        assert!(plan.shots.iter().all(|shot| shot.dialogue_clip.is_none()));
        assert!(validate_plan_structure(&plan).is_empty());

        let pack: ReferencePack = serde_json::from_value(pack_json()).expect("pack parses");
        assert!(pack.sound.is_empty());
        assert!(validate_reference_pack(&pack).is_empty());
        assert!(validate_plan_against_pack(&plan, &pack).is_empty());
    }

    #[test]
    fn a_plan_and_pack_carrying_sound_validate_clean() {
        let plan: ProductionPlan = serde_json::from_value(sound_plan_json()).expect("plan parses");
        let pack: ReferencePack = serde_json::from_value(sound_pack_json()).expect("pack parses");
        assert!(validate_plan_structure(&plan).is_empty());
        assert!(validate_reference_pack(&pack).is_empty());
        assert!(
            validate_plan_against_pack(&plan, &pack).is_empty(),
            "{:?}",
            validate_plan_against_pack(&plan, &pack)
        );
        assert_eq!(
            plan.shots[1]
                .dialogue_clip
                .as_ref()
                .expect("SH020 has a line")
                .role,
            "courier_line"
        );
    }

    /// A sound role placed on the wrong bus is refused BEFORE anything renders.
    ///
    /// This is the finding worth the most: `ambience.role` pointing at a dialogue take is a plan
    /// that runs, exports and simply sounds wrong, and discovering that costs a whole GPU render.
    #[test]
    fn a_sound_role_on_the_wrong_bus_is_refused() {
        let mut plan_value = sound_plan_json();
        plan_value["sound"]["ambience"]["role"] = json!("courier_line");
        plan_value["shots"][1]["dialogueClip"]["role"] = json!("theme");
        let plan: ProductionPlan = serde_json::from_value(plan_value).expect("plan parses");
        let pack: ReferencePack = serde_json::from_value(sound_pack_json()).expect("pack parses");
        let findings = messages(&validate_plan_against_pack(&plan, &pack));
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("sound.ambience.role")
                    && finding.contains("\"dialogue\" entry")
                    && finding.contains("ambience bus")),
            "{findings:#?}"
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("dialogueClip.role")
                    && finding.contains("\"music\" entry")),
            "{findings:#?}"
        );
    }

    #[test]
    fn an_unknown_sound_role_names_the_roles_the_pack_does_have() {
        let mut plan_value = sound_plan_json();
        plan_value["sound"]["music"]["role"] = json!("missing_cue");
        let plan: ProductionPlan = serde_json::from_value(plan_value).expect("plan parses");
        let pack: ReferencePack = serde_json::from_value(sound_pack_json()).expect("pack parses");
        let findings = messages(&validate_plan_against_pack(&plan, &pack));
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("missing_cue")
                    && finding.contains("courier_line, room_tone, theme")),
            "{findings:#?}"
        );
    }

    #[test]
    fn a_pack_refuses_a_bad_sound_entry() {
        let mut pack_value = sound_pack_json();
        pack_value["sound"] = json!([
            { "role": "courier_line", "kind": "dialogue", "file": "sound/a.wav" },
            { "role": "courier_line", "kind": "dialogue", "file": "sound/b.wav" },
            { "role": "bad_kind", "kind": "foley", "file": "sound/c.wav" },
            { "role": "not_audio", "kind": "music", "file": "sound/c.png" },
            { "role": "escapes", "kind": "music", "file": "../outside.wav" },
            { "role": "red_parcel", "kind": "sfx", "file": "sound/d.wav" }
        ]);
        let pack: ReferencePack = serde_json::from_value(pack_value).expect("pack parses");
        let findings = messages(&validate_reference_pack(&pack));
        for expected in [
            "duplicate sound role",
            "unknown sound kind",
            "must be audio",
            "must be a relative path",
            "already a reference role",
        ] {
            assert!(
                findings.iter().any(|finding| finding.contains(expected)),
                "expected a finding containing {expected:?}: {findings:#?}"
            );
        }
    }

    #[test]
    fn out_of_range_gains_fades_and_offsets_are_refused() {
        let mut plan_value = sound_plan_json();
        plan_value["sound"]["ambience"]["gain"] = json!(12.0);
        plan_value["sound"]["music"]["fadeOutSeconds"] = json!(-1.0);
        plan_value["shots"][1]["dialogueClip"]["offsetSeconds"] = json!(-0.5);
        plan_value["shots"][1]["dialogueClip"]["durationSeconds"] = json!(0.0);
        let plan: ProductionPlan = serde_json::from_value(plan_value).expect("plan parses");
        let findings = messages(&validate_plan_structure(&plan));
        for expected in [
            "sound.ambience.gain",
            "sound.music.fadeOutSeconds",
            "dialogueClip.offsetSeconds",
            "dialogueClip.durationSeconds",
        ] {
            assert!(
                findings.iter().any(|finding| finding.contains(expected)),
                "expected a finding for {expected:?}: {findings:#?}"
            );
        }
    }

    #[test]
    fn a_missing_sound_file_is_found_before_dispatch() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir_all(dir.path().join("references")).expect("references dir");
        for entry in ["workshop_plate.png", "red_parcel.png", "look.png"] {
            std::fs::write(dir.path().join("references").join(entry), b"x").expect("plate writes");
        }
        std::fs::create_dir_all(dir.path().join("sound")).expect("sound dir");
        std::fs::write(dir.path().join("sound/room_tone.wav"), b"x").expect("clip writes");
        let pack: ReferencePack = serde_json::from_value(sound_pack_json()).expect("pack parses");
        let findings = messages(&validate_reference_pack_files(&pack, dir.path()));
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("courier_line") && finding.contains("is missing")),
            "{findings:#?}"
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("theme") && finding.contains("is missing")),
            "{findings:#?}"
        );
        assert!(
            !findings.iter().any(|finding| finding.contains("room_tone")),
            "the clip that IS on disk must not be reported: {findings:#?}"
        );
    }

    /// A shot's policy wins over the run's, and both spell the same way the timeline does.
    #[test]
    fn a_shot_can_override_the_runs_generated_audio_policy() {
        let mut plan_value = sound_plan_json();
        plan_value["sound"]["generatedAudio"] = json!("mute");
        plan_value["shots"][0]["generatedAudio"] = json!("include");
        let plan: ProductionPlan = serde_json::from_value(plan_value).expect("plan parses");
        assert!(validate_plan_structure(&plan).is_empty());
        assert_eq!(plan.sound.generated_audio, GeneratedAudio::Mute);
        assert_eq!(
            plan.shots[0].generated_audio,
            Some(GeneratedAudio::Include),
            "an explicit per-shot opt-in"
        );
        assert_eq!(plan.shots[1].generated_audio, None, "SH020 inherits");
        assert_eq!(GeneratedAudio::Include.as_timeline_str(), "include");
        assert_eq!(GeneratedAudio::Mute.as_timeline_str(), "mute");
        assert_eq!(GeneratedAudio::default(), GeneratedAudio::Mute);
    }
    #[test]
    fn run_record_round_trips() {
        let record = RunRecord {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            run_id: "run_1".into(),
            created_at: "2026-09-13T00:00:00Z".into(),
            finished_at: None,
            outcome: RunOutcome::Rejected,
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
            sound: vec![],
            shots: vec![],
            timeline: None,
            export: None,
            diagnostics: vec![PlanDiagnostic::shot("SH010", "prompt", "empty")],
            elapsed_seconds: 0.0,
        };
        let json = record.to_json();
        assert_eq!(json["outcome"], "rejected");
        let back: RunRecord = serde_json::from_value(json).unwrap();
        assert_eq!(back, record);
    }
}
