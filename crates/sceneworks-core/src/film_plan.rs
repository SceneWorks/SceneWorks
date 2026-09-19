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
    REFERENCE_IMAGE_SHORT_EDGE_MAX, REFERENCE_IMAGE_SHORT_EDGE_MIN,
};
// Longest prompt the generation routes accept: the route's own declaration, not a copy of it, so
// this validator cannot bless a prompt length the enqueue would refuse (sc-22710).
use crate::MAX_PROMPT_CHARS;

/// Schema version of [`ProductionPlan`] documents this module reads and writes.
///
/// **2** (sc-22711) added `shots[].dependsOn`.
/// **3** (sc-24026) replaces the optional `shots[].sound` prose with a REQUIRED [`Shot::audio`].
/// The rename is the point rather than a tidy-up: `sound` was traceability-only prose that never
/// left the documents, while `audio` is dispatched — MiniMax-H3 generates its soundtrack from the
/// same prompt as the picture, so a shot that says nothing about sound gets whatever the model
/// invents. Requiring it is what makes "this shot is silent" a statement the plan had to make.
///
/// Versions 1 and 2 are REFUSED, not migrated: the harness is unreleased, every checked-in document
/// moved with this bump, and a v2 plan read under a default would dispatch an empty audio sentence
/// on every shot — exactly the silence this version exists to remove.
pub const PLAN_SCHEMA_VERSION: u32 = 3;
/// Plan schema versions this build accepts.
pub const SUPPORTED_PLAN_SCHEMA_VERSIONS: &[u32] = &[PLAN_SCHEMA_VERSION];
/// The label `film_compile::audio_text` writes in front of a shot's [`Shot::audio`] when it
/// composes the dispatched prompt (sc-24026).
///
/// Declared here rather than in the compiler because it is also what `validate_shot_structure`
/// refuses an authored value for starting with: the compiler owns the prefix, so a plan that
/// carries it too would dispatch it twice. One constant, so the check and the text it guards can
/// never disagree about the spelling.
pub const AUDIO_PROMPT_PREFIX: &str = "Audio:";
/// Schema version of [`ReferencePack`] documents this module reads and writes. Version 2 (sc-24024)
/// adds `references[].locator` — the phrase that picks one subject out of an image several roles
/// share ("the woman on the left").
///
/// Only the current version is read. The harness is unreleased, so a version 1 document is refused
/// by version rather than migrated on read: the alternative is a build that silently accepts a pack
/// whose author could not have known about the rule that now governs shared files.
pub const REFERENCE_PACK_SCHEMA_VERSION: u32 = 2;
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

/// The subset of [`REFERENCE_KINDS`] a `reference_to_video` shot may legitimately BIND.
///
/// Ref2VA treats every bound image as a **subject to depict**, so only the kinds that name a
/// subject belong in `conditioning.referenceRoles`. The two excluded kinds are excluded for that
/// reason, not by oversight:
///
///   * `style` — a look, not a subject. Binding it asks for a shot OF the look, which is why the
///     shipped pack's `house_style` lives in `continuityRoles` and is bound nowhere.
///   * `plate` — a literal frame. It is placed through the KEYFRAME slots (`image_to_video` /
///     `first_last_frame`), a different conditioning task.
///
/// Used in two places, and they are the same rule read from one constant so they cannot drift:
/// [`validate_plan_against_pack`] REFUSES any `conditioning.referenceRoles` entry whose pack kind
/// is not listed here, and the planner ([`crate::film_planner`]) counts a pack's bindable entries
/// to decide whether a pack can fill a reference shot at all — a pack approving only a style and a
/// plate approves nothing a `reference_to_video` shot could bind, so the mode comes off the
/// envelope rather than being offered and then refused a decode later.
pub const BINDABLE_REFERENCE_KINDS: &[&str] = &["character", "prop", "location"];

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

/// TTS models a `dialogue` entry may name in `model` (sc-23404).
///
/// Deliberately a short explicit list rather than "any `type: audio` catalog id": a `sfx` or music
/// model posted here would enqueue happily and come back as something nobody can speak, and the
/// point of naming it in the pack is that a reader knows what voice they are asking for. Every id
/// here is a speech model the audio route already serves; the per-model voice/language surface is
/// still owned by the generator's own `validate` at the gen-core floor, which is why `voice` is
/// bounded here but never allow-listed.
pub const SOUND_SYNTHESIS_MODELS: &[&str] = &[
    "kokoro_82m",
    "chatterbox_tts",
    "moss_tts_realtime",
    "moss_ttsd_v05",
];

/// The TTS model a `dialogue` entry that names none synthesizes through. Matches the audio route's
/// own default (`apps/rust-api/src/defaults.rs`), so an entry that says nothing gets what a caller
/// posting the bare route would get.
pub const DEFAULT_SOUND_SYNTHESIS_MODEL: &str = "kokoro_82m";

/// Longest line a `dialogue` entry may ask to have synthesized.
///
/// A declared finite bound, not a guess at the model's ceiling: the audio route bounds the prompt
/// at 4000 characters and each model's advertised `audio.maxDurationSecs` is the real cap the
/// worker applies. This is the harness's own — a "line" in a film plan that runs past a thousand
/// characters is a document error, and catching it here costs no GPU.
pub const MAX_DIALOGUE_TEXT_CHARS: usize = 1_000;

/// Longest voice id a `dialogue` entry may name. The route bounds nothing here (the generator owns
/// the per-model voice bank), so the pack bounds the string it would interpolate.
const MAX_SOUND_VOICE_CHARS: usize = 64;

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
    /// Sound-effect beds, each placed once at its own sequence start.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sfx: Vec<SoundBed>,
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
    /// Catalog LoRA ids this plan renders with, declared ONCE on the family exactly as `tier` is
    /// (sc-23406).
    ///
    /// A plan does not say which of a split family's partitions a LoRA attaches to, because the
    /// catalog already does: an entry's `modelIds` allowlist names the partitions it was distilled
    /// for, and [`plan_loras_for_partition`] emits each id only onto the shots whose resolved
    /// partition it is declared for. So one list covers a mixed plan — the ref2v turbo reaches the
    /// reference shots, an fl2v turbo reaches the base ones, and neither reaches the other.
    ///
    /// Empty is the base regime: a plan authored before this field dispatches exactly what it did.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    /// Per-family engine knobs the plan may set. Omitted by every plan that wants the engine's own
    /// defaults, which is what a plan authored before sc-23402 is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advanced: Option<PlanModelAdvanced>,
}

/// The plan's opt-in engine knobs (sc-23402). Each one is a REQUEST axis, not a document axis: it
/// rides `advanced` on the dispatched job exactly as the Video Studio's own knobs do, and a plan
/// that names none dispatches exactly what it did before the knob existed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlanModelAdvanced {
    /// The short edge an image REFERENCE is encoded at, in pixels — MiniMax-H3's `ref2va` knob
    /// (`advanced.referenceImageShortEdge`), admitted over
    /// [`REFERENCE_IMAGE_SHORT_EDGE_MIN`]`..=`[`REFERENCE_IMAGE_SHORT_EDGE_MAX`] inclusive and
    /// defaulting to [`REFERENCE_IMAGE_SHORT_EDGE_DEFAULT`].
    ///
    /// It sizes the reference, never the render: lowering it buys reference token count (roughly
    /// quadratic in the short edge) at the cost of reference detail. It reaches only the shots that
    /// resolve to the family's REFERENCE partition — a base-partition shot has no reference to
    /// size, so the knob is not written into its request, its job body or its attempt record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image_short_edge: Option<u32>,
    /// Model evaluations (NFE) every shot renders at, overriding whatever the rest of the plan
    /// would have resolved — the plan-level twin of the Video Studio's `advanced.steps` (sc-23406).
    ///
    /// It wins over a selected turbo recipe's own step count, exactly as it does on the worker
    /// (`minimax_h3_sampling`): a caller who knows the checkpoint may run the 8-step file at 4.
    /// Omitted, the recipe's count governs, and with no recipe the model's declared default does.
    ///
    /// Typed as a signed integer so a plan that writes `0` or `-4` is refused BY NAME here rather
    /// than failing to parse with a serde message that names no plan field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<i64>,
}

/// Finite limits declared BEFORE dispatch. Exceeding any of them stops new dispatch and leaves the
/// reason in the run record; none of them may be zero or non-finite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlanLimits {
    /// Wall-clock budget for the whole run, including the export.
    pub max_run_seconds: u64,
    /// Wall-clock budget for one JOB the run dispatches: one attempt of one shot, the export job,
    /// and — since sc-23404 — one dialogue synthesis (`POST /api/v1/audio/jobs`) for a `dialogue`
    /// sound entry carrying `text`.
    ///
    /// There is deliberately no separate `maxSpeechSeconds`: a speech job is a job, it is dispatched
    /// and polled on exactly the same seam as a render, and a second knob would be one more number
    /// a plan author has to get right for no bound this one does not already state. A pack whose
    /// lines need longer than a shot does raises this value.
    pub max_shot_seconds: u64,
    /// Attempts per shot, counting the first. `1` means no retry.
    pub max_attempts_per_shot: u32,
    /// Memory the run is allowed to use, in GB. Checked against the model's declared minimum
    /// before dispatch and against the observed peak after every attempt.
    pub max_memory_gb: f64,
    /// Memory the PLANNER's LLM decodes are allowed, in GB (sc-22715). A brief must declare it —
    /// the planner runs `1 + rounds` full local decodes plus one rewrite per shot, and a budget
    /// nobody wrote down is not a declared bound — and it is checked against the API host's
    /// reported memory before the first token. A hand-authored plan runs no planner and may omit
    /// it; a generated plan carries the brief's value through unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_max_memory_gb: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Shot {
    /// Stable id (`[A-Za-z0-9_-]{1,64}`), unique within the plan. Later stories key resume,
    /// take replacement and review on it.
    pub id: String,
    /// The brief beat this shot covers, when the plan was generated from one
    /// ([`crate::film_planner`]). Kept in the plan so beat coverage survives a hand edit: a
    /// recompile checks it against the brief by identity rather than by matching prose. A
    /// hand-authored plan carries no brief and so no beat ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beat_id: Option<String>,
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
    /// What this shot SOUNDS like, as one sentence the compiler appends to the dispatched prompt
    /// (sc-24026). Diegetic sound, ambience, music or "no music" — or an explicit statement of
    /// silence ("No audio. Silence.").
    ///
    /// REQUIRED on every shot, and never pattern-matched: MiniMax-H3 generates its soundtrack from
    /// the same prompt as the picture, so whatever the prompt does not describe the model invents.
    /// A blank value is refused rather than defaulted, because "the author had nothing to say about
    /// sound" and "this shot is silent" are different films and only the author knows which one this
    /// is. It replaces the optional `sound` prose of schema version 2, which never left the
    /// documents.
    pub audio: String,
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
    /// Roles the shot depicts. Every entry must exist in the pack.
    ///
    /// These ARE sent to the model, as text (sc-24025). For every role listed here that this shot
    /// does NOT bind to an image — a described-only role, or an image-backed one on a shot that
    /// resolves to the base checkpoint or simply binds nothing — the compiler inserts the pack's
    /// `description` of it, word for word, into the dispatched prompt. That is the identity lock:
    /// reference images are optional in this harness, and a subject nothing conditions on drifts
    /// as soon as two shots word it differently, so the compiler words it once and repeats itself.
    /// A role this shot DOES bind already carries its description in its binding sentence and is
    /// not described twice.
    ///
    /// Listing a role here is therefore a claim about the film, not a note in the margin: it says
    /// this shot shows that subject, and the prompt will say so.
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
    /// DECLARED continuity intent: "this shot continues from the end of that earlier shot".
    ///
    /// It is recorded on the compiled request and in the run record so a reviewer can see which
    /// shots were meant to chain, and it is **never** a conditioning anchor by itself: the frames
    /// and references a shot is actually conditioned on are the pack roles above, which resolve to
    /// approved, canonical reference assets. [`validate_plan_against_pack`] enforces that — a shot
    /// that names a chain but binds no canonical role is refused — so a sequence can never drift by
    /// depending solely on the previous shot's last frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_from_shot_id: Option<String>,
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

/// One approved audio clip the plan's sound roles resolve against — a file on disk, a line the run
/// synthesizes, or (when both are given) a line synthesized INTO the named file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SoundEntry {
    /// Role name (`[A-Za-z0-9_-]{1,64}`), unique within the pack's sound entries.
    pub role: String,
    /// One of [`SOUND_KINDS`]. A role may only be placed on the bus its kind names.
    pub kind: String,
    /// Audio path relative to the pack document's directory.
    ///
    /// Optional only because an entry may carry [`SoundEntry::text`] instead (sc-23404): synthesis
    /// writes the clip into the pack directory and the run records the path it wrote. An entry
    /// carrying BOTH is synthesized into the named path, which is how a pack pins the filename of
    /// a line it means to keep. An entry with NEITHER is a finding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default)]
    pub description: String,
    /// The line to speak (sc-23404). `dialogue` entries only — a synthesized bed or sound effect is
    /// a different job type with a different surface, and letting `text` sit on an `ambience` entry
    /// would quietly enqueue a TTS model against a room-tone description.
    ///
    /// Present ⇒ the run synthesizes the clip through `POST /api/v1/audio/jobs` during
    /// `ensure_sound`, under the plan's own `limits`, and imports the result exactly as it imports
    /// a pre-recorded one. Absent ⇒ [`SoundEntry::file`] is a clip a human put there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Voice id for the synthesis (e.g. Kokoro's `am_michael`). `None` ⇒ the model's own default.
    /// NOT allow-listed here: the per-model voice bank is the generator's, and an unknown id is a
    /// typed refusal at the gen-core floor rather than a guess this document could make.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// TTS model, one of [`SOUND_SYNTHESIS_MODELS`]. `None` ⇒ [`DEFAULT_SOUND_SYNTHESIS_MODEL`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl SoundEntry {
    /// Whether this entry's clip is produced by the run rather than read off disk.
    pub fn is_synthesized(&self) -> bool {
        self.text.is_some()
    }

    /// The line to speak, trimmed — the exact text the synthesis job is sent, and the text the
    /// clip's deterministic name is digested from, so both agree on one canonicalization.
    pub fn synthesis_text(&self) -> Option<&str> {
        self.text
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
    }

    /// The TTS model this entry synthesizes through.
    pub fn synthesis_model(&self) -> &str {
        self.model
            .as_deref()
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .unwrap_or(DEFAULT_SOUND_SYNTHESIS_MODEL)
    }

    /// The voice this entry asks for, if any.
    pub fn synthesis_voice(&self) -> Option<&str> {
        self.voice
            .as_deref()
            .map(str::trim)
            .filter(|voice| !voice.is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceEntry {
    /// Role name (`[A-Za-z0-9_-]{1,64}`), unique within the pack.
    pub role: String,
    pub kind: String,
    /// Image path relative to the pack document's directory.
    ///
    /// SEVERAL roles may name the SAME file (sc-24024) — one photograph holding two people is one
    /// image with two subjects in it. The file is then supplied to the engine ONCE, under one
    /// `<Picture N>` that every role sharing it is bound to, and imported as ONE project asset.
    /// "The same file" means this string, compared literally: bindings in this repo are keyed on
    /// the configured path, and resolving through the filesystem would make two packs that read
    /// identically behave differently depending on symlinks and case folding.
    ///
    /// OPTIONAL (sc-24025). A role with no file is DESCRIBED-ONLY: it exists so the compiler can
    /// say the same words about it in every shot that names it in `continuityRoles`, which is the
    /// only thing holding a subject steady in a film whose shots have no image to condition on.
    /// Reference images are optional in this harness, and a described-only role is how a pack
    /// describes a courier it has no photograph of.
    ///
    /// A described-only role therefore takes no part in anything an image does: it is never
    /// grouped with another role by file, never imported as a project asset, never counted against
    /// the reference limit, never sent as planner pixels, and may carry no [`ReferenceEntry::locator`]
    /// (a locator picks a subject out of an image, and there is no image to pick it out of). It is
    /// never BINDABLE either — [`validate_plan_against_pack`] refuses one named in any
    /// `conditioning.*` slot — because conditioning supplies a picture and this role has none.
    ///
    /// An entry with NEITHER a file nor a [`ReferenceEntry::description`] is refused by name: it
    /// says nothing and shows nothing, so no shot could depict it and nothing could be written
    /// about it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// The phrase that picks THIS role's subject out of [`ReferenceEntry::file`] — "the woman on
    /// the left", "the parcel on the bench" (sc-24024).
    ///
    /// Required on every role that shares its file with another, because a picture holding two
    /// people cannot bind two roles without saying which is which: the compiler writes "The courier
    /// is the woman on the left in `<Picture 1>`." Optional on a role with an image to itself,
    /// where "the person shown in `<Picture 1>`" is already unambiguous — but honoured when given,
    /// since a lone reference may still be a crowded photograph.
    ///
    /// **A noun phrase, with its article, that completes "The courier is …".** The compiler drops
    /// the phrase in verbatim and adds nothing of its own, so the leading "the"/"a" is the
    /// author's to supply: "the woman on the left" reads "The courier is the woman on the left in
    /// `<Picture 1>`.", while "woman on the left" reads "The courier is woman on the left in
    /// `<Picture 1>`." Nothing can check this — a locator is free prose — so it is stated here,
    /// in the refusal that asks for one, and in the pack documentation.
    ///
    /// Repeated into the dispatched prompt, so it is held to the same rules as
    /// [`ReferenceEntry::description`]: no `<` or `>`, no control characters, whitespace
    /// normalized by `film_compile::normalized_description`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// Project-library asset this reference was copied from, when it entered through the Film
    /// workspace. The copied file remains the runnable input; this id is provenance and lets the
    /// authoring UI identify the original without making a pinned run depend on mutable library
    /// state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_asset_id: Option<String>,
    /// What this role IS, in the author's own words — and, because the compiler repeats it into
    /// prompts, the words that hold it steady from shot to shot.
    ///
    /// **Write it as a COMPLETE SENTENCE about the subject, naming the subject itself.** The
    /// compiler repeats it verbatim (whitespace normalized by `film_compile::normalized_description`)
    /// as a sentence of its own and supplies nothing but a closing `.` when the text lacks terminal
    /// punctuation. It never prefixes the role name, so the description has to carry its own
    /// subject or the prompt reads as a fragment: "The courier: blue jacket, carries the parcel."
    /// and "Small bright red cardboard parcel." both work; "blue jacket" alone does not.
    ///
    /// ONE rule for both places it is repeated, so the two never drift (sc-24025):
    ///   * appended to a reference's binding sentence — "The courier is the person shown in
    ///     `<Picture 1>`. The courier: blue jacket, carries the parcel." (sc-24023);
    ///   * inserted ALONE as the text identity lock, on every shot that lists this role in
    ///     `continuityRoles` without binding its image, which is what keeps a described subject
    ///     worded identically across a film whose shots carry no reference (sc-24025).
    ///
    /// Both go through `film_compile::description_sentence`. REQUIRED on a described-only role
    /// (one with no [`ReferenceEntry::file`]): it is everything that role is. Optional on an
    /// image-backed one, which still shows the picture when it says nothing.
    #[serde(default)]
    pub description: String,
    /// Only approved references may be used as conditioning.
    #[serde(default = "default_true")]
    pub approved: bool,
    /// This plate was GENERATED as a test fixture rather than supplied by a person (sc-23403).
    ///
    /// Provenance only. The harness treats a generated reference exactly like any other — the same
    /// import, the same tags, the same `approved` gate decides conditioning — and the flag exists
    /// so a pack, and the asset imported from it, always says whether its plates came from a
    /// person or from a fixture generator. In the product a user supplies the references; the
    /// generator (`film-harness make-references`) exists for the harness's own fixtures.
    #[serde(default)]
    pub generated: bool,
    /// What produced a generated plate, for the record: model, geometry, prompt, seed and the job
    /// and asset it came out of. Present only on a `generated` entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GeneratedReference>,
}

impl ReferenceEntry {
    /// This entry's image path, or `None` when the role is DESCRIBED-ONLY (sc-24025).
    ///
    /// THE one reading of the field, so everything that treats a role as having a picture — the
    /// grouping that decides which roles share a `<Picture N>`, the import, the planner's pixel
    /// upload, the file-existence check — agrees on what "has an image" means. A declared but
    /// blank path is `None` here AND a finding from [`reference_image_file_findings`], so a pack
    /// that writes `"file": "  "` is refused rather than silently read as described-only.
    pub fn file(&self) -> Option<&str> {
        self.file
            .as_deref()
            .map(str::trim)
            .filter(|file| !file.is_empty())
    }

    /// Whether this role exists only as words: no image, and therefore nothing to condition on,
    /// import, number or count against a reference limit (sc-24025).
    pub fn is_described_only(&self) -> bool {
        self.file().is_none()
    }

    /// This entry's locator, trimmed, or `None` when it declares none or declares only whitespace.
    /// The one reading of the field, so the validator that REQUIRES one on a shared file and the
    /// compiler that writes one into a sentence agree on what "has a locator" means.
    pub fn locator(&self) -> Option<&str> {
        self.locator
            .as_deref()
            .map(str::trim)
            .filter(|locator| !locator.is_empty())
    }
}

/// Provenance of one generated reference plate (sc-23403).
///
/// Every field is what the generation route was actually told or what it actually answered; none
/// of it is read back by the harness, and changing it changes nothing about how the reference is
/// used. `negative_prompt` is absent for a model that declares no negative-prompt support (Krea 2
/// Turbo is CFG-free and declares `image.supportsNegativePrompt: false`), rather than recorded as
/// an empty string the model never saw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeneratedReference {
    /// Catalog model id the plate was rendered with.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    /// Backend the job reported (`mlx` / `candle`), when it reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Image mode the job was created with (`text_to_image`).
    pub mode: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    pub width: u32,
    pub height: u32,
    /// The job that rendered it.
    pub job_id: String,
    /// The asset the job wrote in the generating project.
    pub asset_id: String,
    /// SHA-256 of the file as it was written into the pack.
    pub sha256: String,
    pub created_at: String,
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
    parse_plan_document(&text).map_err(|diagnostic| {
        PlanDiagnostic::plan(
            diagnostic.field,
            format!("{}: {}", path.display(), diagnostic.message),
        )
    })
}

/// The message [`validate_plan_structure`] and the document pre-scan both report for a plan this
/// build does not read — written once so the refusal names the same remedy wherever it surfaces.
fn unsupported_plan_schema_message(version: u32) -> String {
    format!(
        "unsupported plan schema version {version} (this build reads \
         {SUPPORTED_PLAN_SCHEMA_VERSIONS:?}); a version 1 or 2 plan predates the required \
         shots[].audio sentence — set \"schemaVersion\": {PLAN_SCHEMA_VERSION} and give every shot \
         an \"audio\" value saying what it sounds like (or that it is silent)"
    )
}

/// Parse a plan document, reporting a version this build cannot read BEFORE the typed decode
/// (sc-24026).
///
/// The pre-scan is what makes the version refusal reachable at all. Schema version 3 renamed
/// `shots[].sound` to a required `shots[].audio`, and [`Shot`] is `deny_unknown_fields`, so a
/// version 1 or 2 document fails inside serde with `unknown field \`sound\`` at a byte offset —
/// [`validate_plan_structure`]'s remedy sentence is never reached, and the operator is handed a
/// parser position instead of the one line that fixes it. Every plan DOCUMENT comes through here:
/// the CLI's `--plan`, and the `plan.json` each run pins and reads back on resume, replace-take and
/// review.
///
/// Refusal, never migration: a document has an author who can edit it, and a version 2 plan carried
/// forward under an empty default would dispatch a silent prompt on every shot while reporting
/// clean. Project-store DRAFTS are the other case and are carried forward on read instead, because
/// a draft's `schemaVersion` is state with no author and no way to edit it — see
/// `ProjectStore::carry_film_draft_forward`.
pub fn parse_plan_document(text: &str) -> Result<ProductionPlan, PlanDiagnostic> {
    let stripped = strip_jsonc_comments(text);
    let scouted: Value = serde_json::from_str(&stripped)
        .map_err(|error| PlanDiagnostic::plan("plan", error.to_string()))?;
    if let Some(version) = scouted.get("schemaVersion").and_then(Value::as_u64) {
        let version = u32::try_from(version).unwrap_or(u32::MAX);
        if !SUPPORTED_PLAN_SCHEMA_VERSIONS.contains(&version) {
            return Err(PlanDiagnostic::plan(
                "schemaVersion",
                unsupported_plan_schema_message(version),
            ));
        }
    }
    // Decoded from the text rather than from `scouted` so a genuine structural error still carries
    // serde's line and column.
    serde_json::from_str(&stripped).map_err(|error| PlanDiagnostic::plan("plan", error.to_string()))
}

/// Parse a plan from JSON/JSONC text. [`parse_plan_document`] with the diagnostic flattened, for
/// the callers that only print one string.
pub fn parse_plan(text: &str) -> Result<ProductionPlan, String> {
    parse_plan_document(text).map_err(|diagnostic| diagnostic.message)
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
            unsupported_plan_schema_message(plan.schema_version),
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
    // sc-23402. Refused, never clamped: the value is the reference TOKEN BUDGET the author asked
    // for, so silently rendering at a different one would make the plan a false record of its own
    // run. The same range the engine admits (gen-core's
    // `validate_reference_image_short_edge`), refused here so a typo costs a document read rather
    // than a 53 GB text-encoder load.
    if let Some(edge) = plan
        .model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.reference_image_short_edge)
    {
        if !(REFERENCE_IMAGE_SHORT_EDGE_MIN..=REFERENCE_IMAGE_SHORT_EDGE_MAX).contains(&edge) {
            findings.push(PlanDiagnostic::plan(
                "model.advanced.referenceImageShortEdge",
                format!(
                    "referenceImageShortEdge must be from {REFERENCE_IMAGE_SHORT_EDGE_MIN} to \
                     {REFERENCE_IMAGE_SHORT_EDGE_MAX}, got {edge}"
                ),
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
    findings.extend(validate_plan_loras(plan));
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
        // A declared continuity chain must point BACKWARDS at a shot this plan already established,
        // so the intent is readable in plan order and cannot form a cycle.
        if let Some(target) = shot.conditioning.chain_from_shot_id.as_deref() {
            // Not merely "some earlier shot": the shot IMMEDIATELY before this one. "Continues
            // from" is a claim about adjacency in the cut, and a chain that skips over the shots
            // between reads as continuity the sequence does not have — a planner that chains every
            // shot back to the first one says nothing while looking like it says something
            // (sc-22713 real-LLM smoke).
            let previous = plan.shots[..index].last().map(|prior| prior.id.as_str());
            if previous != Some(target) {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "conditioning.chainFromShotId",
                    match previous {
                        Some(previous) => format!(
                            "{target:?} is not the shot immediately before this one ({previous:?}); \
                             a continuity chain names the shot it cuts from, or is omitted"
                        ),
                        None => format!(
                            "{target:?} cannot be chained from: this is the first shot in the plan"
                        ),
                    },
                ));
            }
        }
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
    for (index, bed) in sound.sfx.iter().enumerate() {
        validate_sound_bed(&mut findings, &format!("sound.sfx[{index}]"), bed);
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

pub(crate) fn validate_limits(limits: &PlanLimits) -> Vec<PlanDiagnostic> {
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
    if limits
        .planner_max_memory_gb
        .is_some_and(|budget| !budget.is_finite() || budget <= 0.0)
    {
        findings.push(PlanDiagnostic::plan(
            "limits.plannerMaxMemoryGb",
            "the planner memory budget must be a finite number > 0",
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
    if let Some(beat_id) = shot.beat_id.as_deref() {
        if !is_safe_plan_id(beat_id) {
            findings.push(PlanDiagnostic::shot(
                id,
                "beatId",
                format!("beat id {beat_id:?} must be 1-64 characters of [A-Za-z0-9_-]"),
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
    // A first/last-frame shot whose two keyframes are the SAME reference asks the engine to start
    // and end on one still — the clip is told to go nowhere, and whatever motion it invents has to
    // return to the frame it began on. It is the degenerate way to satisfy the slot shape without
    // planning a shot, and it is what the local planner reached for on every shot of the first
    // real-LLM draft (sc-22713).
    if wants_last {
        if let (Some(first), Some(last)) = (
            conditioning.first_frame_role.as_deref(),
            conditioning.last_frame_role.as_deref(),
        ) {
            if first == last {
                findings.push(PlanDiagnostic::shot(
                    id,
                    "conditioning.lastFrameRole",
                    format!(
                        "the first and last frame are both {first:?}, so this shot is asked to end \
                         exactly where it started; use two different reference roles, or a mode \
                         that takes one keyframe (image_to_video) or none (text_to_video)"
                    ),
                ));
            }
        }
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
    // The audio sentence (sc-24026). Required and never pattern-matched: "No audio. Silence." is a
    // complete, valid answer, and the only thing refused is having said NOTHING. The same two
    // checks a pack description gets apply, because it lands in the dispatched prompt the same way.
    if shot.audio.trim().is_empty() {
        findings.push(PlanDiagnostic::shot(
            id,
            "audio",
            "audio is required: say what this shot sounds like (diegetic sound, ambience, music or \
             \"no music\"), or state that it is silent — MiniMax-H3 scores the prompt it is given, \
             so anything left unsaid is invented",
        ));
    } else if shot
        .audio
        .trim_start()
        .get(..AUDIO_PROMPT_PREFIX.len())
        .is_some_and(|start| start.eq_ignore_ascii_case(AUDIO_PROMPT_PREFIX))
    {
        // sc-24026. The compiler writes the `Audio: ` prefix itself (`film_compile::audio_text`),
        // so an author who wrote it too would dispatch `Audio: Audio: room tone`. Refused rather
        // than stripped: a value that opens with the label is an author who misread the field, and
        // silently rewriting authored prose would make the plan a false record of itself.
        findings.push(PlanDiagnostic::shot(
            id,
            "audio",
            format!(
                "audio must not start with {AUDIO_PROMPT_PREFIX:?}: the compiler writes that \
                 prefix into the dispatched prompt itself, so keeping it here would send it twice \
                 — state only what the shot sounds like"
            ),
        ));
    }
    findings.extend(
        inserted_prose_findings("audio", "audio", &shot.audio)
            .into_iter()
            .map(|(field, message)| PlanDiagnostic::shot(id, field, message)),
    );
    findings
}

/// The checks ONE piece of authored text the compiler repeats into a dispatched prompt must pass —
/// a pack entry's `description` (sc-24023) and a shot's `audio` sentence (sc-24026).
///
/// Such text is not inert prose: the compiler repeats it into the sentence it writes into the
/// dispatched prompt, inside `insertedText` — the one field
/// `film_compile::CompiledPlan::conformance_findings` treats as the compiler's own authored, derived
/// text and therefore never reads back. So text carrying `<Picture 3>` or `<Audio 1>` forges a
/// binding to media the shot never supplies, and the compiled document still reports clean. Refused
/// here, at the document boundary, rather than stripped, because a forged marker is an authoring
/// mistake to name.
///
/// Ordinary line breaks and tabs are NOT refused — multi-line prose is a reasonable thing to write —
/// they are collapsed to single spaces by `film_compile::normalized_description` before they reach a
/// prompt. Every other control character is a byte no author typed on purpose.
///
/// Returns `(field, message)` pairs rather than diagnostics, because the same two checks are raised
/// as a PLAN-level finding on a pack entry and as a SHOT-level finding on a shot's audio, and the
/// caller is the one that knows which.
fn inserted_prose_findings(field: &str, noun: &str, text: &str) -> Vec<(String, String)> {
    let mut findings = Vec::new();
    if text.contains(['<', '>']) {
        findings.push((
            field.to_owned(),
            format!(
                "{noun} {text:?} must not contain '<' or '>': it is repeated into the dispatched \
                 prompt, where a marker like <Picture 1> would bind the model to media this shot \
                 never supplies"
            ),
        ));
    }
    if text
        .chars()
        .any(|ch| ch.is_control() && !matches!(ch, '\n' | '\r' | '\t'))
    {
        findings.push((
            field.to_owned(),
            format!(
                "{noun} {text:?} must not contain control characters: it is repeated into the \
                 dispatched prompt verbatim apart from whitespace"
            ),
        ));
    }
    findings
}

/// THE rule for every pack-authored phrase the compiler repeats into a prompt, as a plan-level
/// diagnostic. One function over every such field because they are repeated by the same code into
/// the same sentence, and a check that covered only `description` would let a `locator` say
/// `<Picture 3>` (sc-24023, sc-24024).
fn reference_prose_findings(field: &str, name: &str, text: &str) -> Vec<PlanDiagnostic> {
    inserted_prose_findings(&format!("{field}.{name}"), name, text)
        .into_iter()
        .map(|(field, message)| PlanDiagnostic::plan(field, message))
        .collect()
}

/// [`reference_prose_findings`] for one pack entry's `description`.
fn reference_description_findings(field: &str, description: &str) -> Vec<PlanDiagnostic> {
    reference_prose_findings(field, "description", description)
}

/// The same rules for [`ReferenceEntry::locator`] (sc-24024). A locator is repeated into the
/// binding sentence exactly as a description is — "The courier is **the woman on the left** in
/// `<Picture 1>`." — so a locator that forged a marker would forge one just as effectively.
fn reference_locator_findings(field: &str, locator: Option<&str>) -> Vec<PlanDiagnostic> {
    locator
        .map(|locator| reference_prose_findings(field, "locator", locator))
        .unwrap_or_default()
}

/// Structural findings on the reference pack alone.
/// The checks one reference FILE path must pass, wherever it is declared — a pack entry or a
/// [`ReferenceSpec`] entry. Kept in one place because the basename rule is security-relevant: the
/// name is interpolated into the multipart `Content-Disposition` header the import posts, so a
/// CR/LF in it injects multipart headers.
fn reference_image_file_findings(field: &str, file: &str) -> Vec<PlanDiagnostic> {
    let path = Path::new(file);
    let extension_ok = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            REFERENCE_IMAGE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str())
        });
    let basename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if file.trim().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return vec![PlanDiagnostic::plan(
            format!("{field}.file"),
            format!("file {file:?} must be a relative path inside the pack directory"),
        )];
    }
    // ONE path may be spelled ONE way (sc-24024). Everything that decides whether two roles share
    // an image compares this string LITERALLY — `film_compile::shot_reference_pictures`,
    // `shared_reference_file_findings`, `Session::ensure_references` — so `references/pair.png`
    // and `./references/pair.png` (likewise `references//pair.png`, `references/./pair.png`,
    // `references/pair.png/`, a `\` separator) would import ONE photograph of two people twice,
    // number it `<Picture 1>` and `<Picture 2>`, and require a `locator` on neither: exactly the
    // silent ambiguity this feature exists to remove, and nothing downstream catches it — the
    // import keys on role, so even an identical sha256 does not collapse them.
    //
    // Refused rather than canonicalized on read, for the reason the literal comparison exists in
    // the first place: a pack document has to mean the same thing on every machine, and a
    // normalizing read would quietly make two entries one without the author ever seeing it.
    let forward_slashed = file.replace('\\', "/");
    let canonical = std::path::Path::new(&forward_slashed)
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(name) => name.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/");
    if canonical.is_empty() {
        return vec![PlanDiagnostic::plan(
            format!("{field}.file"),
            format!("file {file:?} must be a relative path inside the pack directory"),
        )];
    }
    if canonical != file {
        return vec![PlanDiagnostic::plan(
            format!("{field}.file"),
            format!(
                "file {file:?} must be spelled in canonical form, as {canonical:?}: no leading \
                 \"./\", no \".\" component, no doubled '/', no trailing '/', no '\\'. Roles \
                 share one image when this string matches LITERALLY, so a second spelling of one \
                 path is imported and numbered as a second image"
            ),
        )];
    }
    if file.contains(['\r', '\n']) || !is_safe_reference_basename(basename) {
        return vec![PlanDiagnostic::plan(
            format!("{field}.file"),
            format!(
                "file {file:?} must have a 1-128 character [A-Za-z0-9._-] basename (it is sent as \
                 a multipart filename)"
            ),
        )];
    }
    if !extension_ok {
        return vec![PlanDiagnostic::plan(
            format!("{field}.file"),
            format!(
                "file {file:?} must be an image ({})",
                REFERENCE_IMAGE_EXTENSIONS.join(", ")
            ),
        )];
    }
    Vec::new()
}

/// `generated` and `generation` must agree (sc-23403): a pack that claims a plate was generated
/// has to say what generated it, and provenance without the flag is a document that has been
/// hand-edited into a state nothing wrote.
fn generated_reference_findings(field: &str, entry: &ReferenceEntry) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    match (entry.generated, entry.generation.as_ref()) {
        (true, None) => findings.push(PlanDiagnostic::plan(
            format!("{field}.generation"),
            format!(
                "reference {:?} is marked generated but declares no generation provenance",
                entry.role
            ),
        )),
        (false, Some(_)) => findings.push(PlanDiagnostic::plan(
            format!("{field}.generated"),
            format!(
                "reference {:?} carries generation provenance but is not marked generated",
                entry.role
            ),
        )),
        (true, Some(generation)) => {
            for (name, value) in [
                ("model", generation.model.as_str()),
                ("mode", generation.mode.as_str()),
                ("prompt", generation.prompt.as_str()),
                ("jobId", generation.job_id.as_str()),
                ("assetId", generation.asset_id.as_str()),
                ("sha256", generation.sha256.as_str()),
            ] {
                if value.trim().is_empty() {
                    findings.push(PlanDiagnostic::plan(
                        format!("{field}.generation.{name}"),
                        format!(
                            "reference {:?}: generation provenance needs a non-empty {name}",
                            entry.role
                        ),
                    ));
                }
            }
            if generation.width == 0 || generation.height == 0 {
                findings.push(PlanDiagnostic::plan(
                    format!("{field}.generation"),
                    format!(
                        "reference {:?}: generation provenance needs the geometry it was rendered \
                         at",
                        entry.role
                    ),
                ));
            }
        }
        (false, None) => {}
    }
    findings
}

/// The rules that apply to a FILE several roles name, and to nothing else (sc-24024).
///
/// Sharing is the point of the feature — one photograph of two people is one image with two
/// subjects — and each of these rules exists because the shared file collapses something that was
/// per-role into something per-image:
///
///   * **Every sharing role needs a `locator`.** The roles are bound to ONE `<Picture N>`, so
///     "The courier is the person shown in `<Picture 1>`. The recipient is the person shown in
///     `<Picture 1>`." tells the model nothing. The finding names every role and the file, because
///     the author is looking at a document whose entries they know by name.
///   * **Their locators must differ.** Two roles picking the same subject out of one image is the
///     same ambiguity written out longhand.
///   * **They must agree on `approved`.** One file is imported as ONE project asset, and the asset
///     carries exactly one approval tag; roles that disagree would make the tag depend on which of
///     them the import happened to reach last.
///   * **They must agree on `generated` / `generation`.** That provenance describes the FILE — the
///     job that rendered it, its sha256 — so two roles on one file cannot honestly claim two.
fn shared_reference_file_findings(pack: &ReferencePack) -> Vec<PlanDiagnostic> {
    // Only roles with an IMAGE can share one (sc-24025). A described-only role has no file, so it
    // is not grouped with anything — and emphatically not with the other described-only roles,
    // which is what a naive grouping on the raw `Option` would do by treating `None` as a path they
    // all name.
    let mut groups: BTreeMap<&str, Vec<&ReferenceEntry>> = BTreeMap::new();
    for entry in &pack.references {
        if let Some(file) = entry.file() {
            groups.entry(file).or_default().push(entry);
        }
    }
    let mut findings = Vec::new();
    // Two spellings that differ only by ASCII case are ONE file on a case-insensitive volume —
    // which APFS and NTFS are by default — while every comparison in this feature is literal. The
    // compiler would number two pictures of one photograph and require a locator on neither, so
    // the pack has to pick a spelling rather than have one picked for it.
    let mut by_fold: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for entry in &pack.references {
        let Some(file) = entry.file() else {
            continue;
        };
        let spellings = by_fold.entry(file.to_ascii_lowercase()).or_default();
        if !spellings.contains(&file) {
            spellings.push(file);
        }
    }
    for spellings in by_fold.values() {
        if spellings.len() < 2 {
            continue;
        }
        findings.push(PlanDiagnostic::plan(
            "referencePack.references.file",
            format!(
                "files {} differ only by ASCII case, so a case-insensitive volume holds ONE file \
                 while this pack is read as {}; spell one path one way, or the roles naming them \
                 will be bound to two pictures of the same image",
                spellings
                    .iter()
                    .map(|spelling| format!("{spelling:?}"))
                    .collect::<Vec<_>>()
                    .join(" and "),
                spellings.len()
            ),
        ));
    }
    for (file, entries) in groups {
        if entries.len() < 2 {
            continue;
        }
        let roles = entries
            .iter()
            .map(|entry| format!("{:?}", entry.role))
            .collect::<Vec<_>>()
            .join(", ");
        let missing: Vec<String> = entries
            .iter()
            .filter(|entry| entry.locator().is_none())
            .map(|entry| format!("{:?}", entry.role))
            .collect();
        if !missing.is_empty() {
            findings.push(PlanDiagnostic::plan(
                "referencePack.references.locator",
                format!(
                    "roles {roles} share the file {file:?}, so each of them needs a `locator` \
                     saying which subject in that image it names: a phrase that completes \"The \
                     {} is …\", article included, e.g. \"the woman on the left\" — the compiler \
                     writes the phrase in verbatim and supplies no article of its own; {} declare \
                     none",
                    entries[0].role,
                    missing.join(", ")
                ),
            ));
        } else {
            let mut locators: Vec<&str> =
                entries.iter().filter_map(|entry| entry.locator()).collect();
            let before = locators.len();
            locators.sort_unstable();
            locators.dedup();
            if locators.len() != before {
                findings.push(PlanDiagnostic::plan(
                    "referencePack.references.locator",
                    format!(
                        "roles {roles} share the file {file:?} but do not all have a DIFFERENT \
                         `locator`; two roles picking the same subject out of one image is the \
                         ambiguity a locator exists to remove"
                    ),
                ));
            }
        }
        if entries
            .iter()
            .any(|entry| entry.approved != entries[0].approved)
        {
            findings.push(PlanDiagnostic::plan(
                "referencePack.references.approved",
                format!(
                    "roles {roles} share the file {file:?} but disagree on `approved`; the file is \
                     imported as ONE asset carrying ONE approval, so it is approved for all of \
                     them or for none"
                ),
            ));
        }
        if entries.iter().any(|entry| {
            entry.generated != entries[0].generated || entry.generation != entries[0].generation
        }) {
            findings.push(PlanDiagnostic::plan(
                "referencePack.references.generation",
                format!(
                    "roles {roles} share the file {file:?} but declare different generation \
                     provenance; the provenance describes the IMAGE, so one file has one"
                ),
            ));
        }
    }
    findings
}

pub fn validate_reference_pack(pack: &ReferencePack) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if pack.schema_version != REFERENCE_PACK_SCHEMA_VERSION {
        findings.push(PlanDiagnostic::plan(
            "referencePack.schemaVersion",
            format!(
                "unsupported reference pack schema version {} (this build reads \
                 {REFERENCE_PACK_SCHEMA_VERSION}); set \"schemaVersion\": \
                 {REFERENCE_PACK_SCHEMA_VERSION} — version 2 only ADDS the optional \
                 references[].locator and makes references[].file itself optional, so a version 1 \
                 document needs no other edit unless two of its roles share one file",
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
        if entry
            .source_asset_id
            .as_deref()
            .is_some_and(|asset_id| !is_safe_plan_id(asset_id))
        {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.sourceAssetId"),
                format!(
                    "source asset id {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                    entry.source_asset_id.as_deref().unwrap_or_default()
                ),
            ));
        }
        // An image-backed role's path is held to every rule it always was. A DESCRIBED-ONLY role
        // (sc-24025) has no path to check and must instead carry the words that are all it is —
        // and may carry no locator, which points at a subject inside an image it does not have.
        match entry.file.as_deref() {
            Some(file) => findings.extend(reference_image_file_findings(&field, file)),
            None => {
                if entry.description.trim().is_empty() {
                    findings.push(PlanDiagnostic::plan(
                        format!("{field}.description"),
                        format!(
                            "reference {:?} has no `file` and no `description`: a role with no \
                             image is DESCRIBED-ONLY, so its description is the whole of it — \
                             give it one, or give the role an image",
                            entry.role
                        ),
                    ));
                }
                if entry.locator().is_some() {
                    findings.push(PlanDiagnostic::plan(
                        format!("{field}.locator"),
                        format!(
                            "reference {:?} has no `file` but declares a `locator`; a locator \
                             picks one subject out of an IMAGE (\"the woman on the left\"), and \
                             this role has none — put the words in `description` instead",
                            entry.role
                        ),
                    ));
                }
            }
        }
        findings.extend(reference_description_findings(&field, &entry.description));
        findings.extend(reference_locator_findings(&field, entry.locator.as_deref()));
        findings.extend(generated_reference_findings(&field, entry));
    }
    findings.extend(shared_reference_file_findings(pack));
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
        findings.extend(validate_sound_source(&field, entry));
        // Same rule as an image entry's: a sound description is prose a compiler-owned binding
        // sentence may repeat, so it may not forge a marker of its own (sc-24023).
        findings.extend(reference_description_findings(&field, &entry.description));
        let Some(declared) = entry.file.as_deref() else {
            continue;
        };
        let file = Path::new(declared);
        let extension_ok = file
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                SOUND_AUDIO_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str())
            });
        if declared.trim().is_empty()
            || file.is_absolute()
            || file
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.file"),
                format!("file {declared:?} must be a relative path inside the pack directory"),
            ));
        } else if !extension_ok {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.file"),
                format!(
                    "file {declared:?} must be audio ({})",
                    SOUND_AUDIO_EXTENSIONS.join(", ")
                ),
            ));
        }
    }
    findings
}

/// Where one sound entry's audio comes from: a file, a synthesized line, or — the finding this
/// exists for — neither (sc-23404).
///
/// Every finding names the entry by ROLE as well as by index, because the operator reading it is
/// looking at a pack whose entries they know by name, and "referencePack.sound[2]" alone makes them
/// count array elements to find out which line the harness refused to speak.
fn validate_sound_source(field: &str, entry: &SoundEntry) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let role = entry.role.as_str();
    let has_file = entry
        .file
        .as_deref()
        .is_some_and(|file| !file.trim().is_empty());
    let text = entry.text.as_deref();
    match text {
        None => {
            if !has_file {
                findings.push(PlanDiagnostic::plan(
                    format!("{field}.file"),
                    format!(
                        "sound {role:?} declares neither `file` nor `text`; a pack entry is either \
                         a clip on disk or a `dialogue` line to synthesize"
                    ),
                ));
            }
            // `voice` / `model` are synthesis knobs. Carried without a line to speak they say the
            // author meant to write one, so they are a finding rather than an ignored field.
            for (name, value) in [("voice", &entry.voice), ("model", &entry.model)] {
                if value.is_some() {
                    findings.push(PlanDiagnostic::plan(
                        format!("{field}.{name}"),
                        format!(
                            "sound {role:?} sets `{name}` but has no `text`; \
                             `{name}` only applies to a synthesized dialogue line"
                        ),
                    ));
                }
            }
        }
        Some(text) => {
            if entry.kind != "dialogue" {
                findings.push(PlanDiagnostic::plan(
                    format!("{field}.text"),
                    format!(
                        "sound {role:?} is kind {:?}; only a `dialogue` entry may carry `text` \
                         (synthesis speaks a line, it does not render a bed or an effect)",
                        entry.kind
                    ),
                ));
            }
            let length = text.trim().chars().count();
            if length == 0 || length > MAX_DIALOGUE_TEXT_CHARS {
                findings.push(PlanDiagnostic::plan(
                    format!("{field}.text"),
                    format!(
                        "sound {role:?}: `text` must be 1-{MAX_DIALOGUE_TEXT_CHARS} characters, \
                         got {length}"
                    ),
                ));
            }
            let model = entry.synthesis_model();
            if !SOUND_SYNTHESIS_MODELS.contains(&model) {
                findings.push(PlanDiagnostic::plan(
                    format!("{field}.model"),
                    format!(
                        "sound {role:?}: unknown speech model {model:?}; expected one of {}",
                        SOUND_SYNTHESIS_MODELS.join(", ")
                    ),
                ));
            }
            if let Some(voice) = entry.voice.as_deref() {
                let voice = voice.trim();
                if voice.is_empty() || voice.chars().count() > MAX_SOUND_VOICE_CHARS {
                    findings.push(PlanDiagnostic::plan(
                        format!("{field}.voice"),
                        format!(
                            "sound {role:?}: `voice` must be 1-{MAX_SOUND_VOICE_CHARS} characters"
                        ),
                    ));
                }
            }
        }
    }
    findings
}

/// Findings that need both documents: every role a shot names must exist in the pack and be
/// approved, and every role bound in `conditioning.referenceRoles` must be a
/// [`BINDABLE_REFERENCE_KINDS`] kind. A nonempty pack does not force every shot to use it: mixed
/// plans intentionally allow reference-backed and reference-free shots in one production.
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
            // A BOUND reference is a Ref2VA subject, and only the subject kinds belong there
            // ([`BINDABLE_REFERENCE_KINDS`]). Without this the planner's rule — which counts a
            // pack's bindable entries to decide whether a reference shot can be offered at all —
            // and the validator disagree, and a plan binding a `plate` or a `style` validates,
            // compiles and dispatches it as a subject to depict. Checked independently of
            // `approved`, so approving the plate does not make the binding legal. `continuityRoles`
            // and the keyframe slots are unaffected: a plate is exactly what a keyframe slot takes.
            if field == "conditioning.referenceRoles" {
                if let Some(entry) = roles.get(role) {
                    if !BINDABLE_REFERENCE_KINDS.contains(&entry.kind.as_str()) {
                        findings.push(PlanDiagnostic::shot(
                            &shot.id,
                            field,
                            format!(
                                "reference role {role:?} is kind {:?}; only {} may be BOUND as a \
                                 reference_to_video subject (a `style` is a look, not a subject, \
                                 and a `plate` is a literal frame placed through firstFrameRole / \
                                 lastFrameRole)",
                                entry.kind,
                                BINDABLE_REFERENCE_KINDS.join(", ")
                            ),
                        ));
                    }
                }
            }
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
                // A DESCRIBED-ONLY role is never bindable, in ANY conditioning slot (sc-24025).
                // Every one of them supplies a picture — `referenceRoles` a Ref2VA subject image,
                // the keyframe slots a literal frame — and this role has no image at all. Refused
                // here rather than left to fail at import, because the import is a whole run's
                // worth of setup later and the refusal it would raise names a missing file rather
                // than the authoring mistake. `continuityRoles` is exactly where such a role
                // BELONGS: the compiler writes its description into the prompt from there.
                Some(entry) if entry.is_described_only() && field.starts_with("conditioning.") => {
                    findings.push(PlanDiagnostic::shot(
                        &shot.id,
                        field,
                        format!(
                            "reference role {role:?} is DESCRIBED-ONLY — it declares no `file`, so \
                             there is no image to condition on. Name it in this shot's \
                             continuityRoles instead, where its description is written into the \
                             prompt word for word; or give the pack entry an image"
                        ),
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
    for (index, bed) in plan.sound.sfx.iter().enumerate() {
        check(None, &format!("sound.sfx[{index}].role"), &bed.role, "sfx");
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
        // A described-only role names no file, so there is none to find on disk (sc-24025).
        let Some(file) = entry.file() else {
            continue;
        };
        let path = pack_dir.join(file);
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
        // A synthesized line has no file on disk until the run speaks it — including one that also
        // names a `file`, which is the path synthesis WRITES rather than one a human already put
        // there. Checking for it here would refuse every speech pack before its first run
        // (sc-23404); `ensure_sound` fails loudly if the clip does not appear.
        if entry.is_synthesized() {
            continue;
        }
        let Some(declared) = entry.file.as_deref() else {
            // Structural: `validate_reference_pack` already named this entry. Nothing to stat.
            continue;
        };
        let path = pack_dir.join(declared);
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

// ------------------------------------------------------------------------------------------
// Reference SPEC (sc-23403) — the document `film-harness make-references` reads
// ------------------------------------------------------------------------------------------

/// Schema version of [`ReferenceSpec`] documents this build reads.
pub const REFERENCE_SPEC_SCHEMA_VERSION: u32 = 1;

/// Image modes a reference spec may drive. Generation only: an `edit_image` request needs a source
/// asset, which a spec that starts from nothing does not have.
pub const REFERENCE_SPEC_MODES: &[&str] = &["text_to_image"];

/// Attempts one role may cost. A retry re-renders on the GPU, so the ceiling is low on purpose.
pub const MAX_REFERENCE_SPEC_ATTEMPTS: u32 = 5;

/// Wall clock ONE job may declare, in seconds (24 hours). A spec is a bounded fixture run, not a
/// standing render: the ceiling keeps `maxJobSeconds` from being a budget in name only, and keeps
/// the generator's `Instant::now() + Duration::from_secs(..)` deadline off the overflow that a
/// declaration near `u64::MAX` would otherwise panic on.
pub const MAX_REFERENCE_SPEC_JOB_SECONDS: u64 = 86_400;

/// A recipe for GENERATING a reference pack's plates (sc-23403).
///
/// **Test fixtures only.** In the product the user supplies the reference images; this document
/// exists so the film harness can produce its own courier/workshop fixtures locally, with the
/// provenance of every plate recorded in the pack it writes. Nothing in the product reads it, and
/// a pack it produced is an ordinary [`ReferencePack`] with `generated: true` on the entries it
/// rendered.
///
/// The roles a plan needs but the spec does not render — a style plate, a keyframe plate — are
/// named in [`ReferenceSpecInherit`] and copied verbatim from an existing pack, along with its
/// sound, so the written pack validates against the same plan the source pack did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceSpec {
    pub schema_version: u32,
    /// Id of the pack this spec writes (`[A-Za-z0-9_-]{1,64}`).
    pub id: String,
    /// Version of the pack this spec writes.
    pub version: u32,
    #[serde(default)]
    pub description: String,
    pub model: ReferenceSpecModel,
    pub limits: ReferenceSpecLimits,
    /// First seed of the run; role `n` renders at `seedBase + n` so a re-run of the same spec asks
    /// for the same images. Absent means the route picks a seed and the pack records what it got.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_base: Option<i64>,
    /// Roles copied verbatim from an existing pack instead of generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inherit: Option<ReferenceSpecInherit>,
    pub references: Vec<ReferenceSpecEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceSpecModel {
    /// Catalog model id (`krea_2_turbo` / `krea_2_raw` for the shipped fixture spec).
    pub id: String,
    /// Quant tier, when the spec pins one. Checked against the catalog's install state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    /// One of [`REFERENCE_SPEC_MODES`].
    #[serde(default = "default_reference_spec_mode")]
    pub mode: String,
    /// `"WxH"`, or absent to render at the model's own declared default resolution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    /// Negative prompt applied to every role that does not declare its own. Refused for a model
    /// whose catalog entry declares `image.supportsNegativePrompt: false`, rather than silently
    /// dropped — Krea 2 Turbo is CFG-free and would ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
}

fn default_reference_spec_mode() -> String {
    REFERENCE_SPEC_MODES[0].to_owned()
}

/// What bounds a `make-references` run. Finite by declaration, like a plan's [`PlanLimits`]: the
/// generator is driving a real GPU and nothing else stops it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceSpecLimits {
    /// Wall-clock budget for ONE image job.
    pub max_job_seconds: u64,
    /// Attempts per role, counting the first. `1` means no retry.
    pub max_attempts_per_role: u32,
    /// Memory the generation is allowed, in GB. Checked against the host's reported memory before
    /// dispatch and against each job's observed peak after it.
    pub max_memory_gb: f64,
}

/// Roles (and sound) copied from an existing pack rather than generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceSpecInherit {
    /// The pack document to copy from, relative to this spec's own directory.
    pub pack: String,
    /// Reference roles to copy. Each must exist in that pack and must not also be generated here.
    #[serde(default)]
    pub references: Vec<String>,
    /// Copy the source pack's whole `sound` array (and its files) too.
    #[serde(default)]
    pub sound: bool,
}

/// One role the spec renders.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceSpecEntry {
    /// Role name (`[A-Za-z0-9_-]{1,64}`), unique within the spec.
    pub role: String,
    /// One of [`REFERENCE_KINDS`].
    pub kind: String,
    /// Where the rendered plate is written, relative to the pack directory.
    pub file: String,
    #[serde(default)]
    pub description: String,
    /// The prompt this role renders from. Required: a role with no prompt is refused by name
    /// rather than rendered from its description.
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Seed for this role, overriding `seedBase + index`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// `"WxH"` for this role, overriding the spec's model resolution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
}

impl ReferenceSpecEntry {
    /// The negative prompt this role sends, falling back to the spec-wide one.
    pub fn negative_prompt_with<'a>(&'a self, spec: &'a ReferenceSpec) -> Option<&'a str> {
        self.negative_prompt
            .as_deref()
            .or(spec.model.negative_prompt.as_deref())
            .map(str::trim)
            .filter(|text| !text.is_empty())
    }

    /// The geometry this role renders at, falling back to the spec-wide one.
    pub fn resolution_with<'a>(&'a self, spec: &'a ReferenceSpec) -> Option<&'a str> {
        self.resolution
            .as_deref()
            .or(spec.model.resolution.as_deref())
    }
}

/// Read and parse a reference spec document (JSONC tolerated).
pub fn read_reference_spec_file(path: &Path) -> Result<ReferenceSpec, PlanDiagnostic> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        PlanDiagnostic::plan(
            "referenceSpec",
            format!("cannot read {}: {error}", path.display()),
        )
    })?;
    parse_reference_spec(&text).map_err(|error| {
        PlanDiagnostic::plan("referenceSpec", format!("{}: {error}", path.display()))
    })
}

/// Parse a reference spec from JSON/JSONC text.
pub fn parse_reference_spec(text: &str) -> Result<ReferenceSpec, String> {
    let stripped = strip_jsonc_comments(text);
    serde_json::from_str(&stripped).map_err(|error| error.to_string())
}

/// Structural findings on a reference spec alone. Every one of them is answerable before a single
/// job is created, which is the point: a spec with a role that declares no prompt must never cost
/// four renders before it says so.
pub fn validate_reference_spec(spec: &ReferenceSpec) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if spec.schema_version != REFERENCE_SPEC_SCHEMA_VERSION {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.schemaVersion",
            format!(
                "unsupported reference spec schema version {} (this build reads \
                 {REFERENCE_SPEC_SCHEMA_VERSION})",
                spec.schema_version
            ),
        ));
    }
    if !is_safe_plan_id(&spec.id) {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.id",
            "reference spec id must be 1-64 characters of [A-Za-z0-9_-]",
        ));
    }
    if spec.version == 0 {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.version",
            "reference spec version must be >= 1",
        ));
    }
    if !is_safe_plan_id(&spec.model.id) {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.model.id",
            format!(
                "model id {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                spec.model.id
            ),
        ));
    }
    if !REFERENCE_SPEC_MODES.contains(&spec.model.mode.as_str()) {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.model.mode",
            format!(
                "unsupported image mode {:?}; a reference spec generates from text ({})",
                spec.model.mode,
                REFERENCE_SPEC_MODES.join(", ")
            ),
        ));
    }
    if let Some(tier) = spec.model.tier.as_deref() {
        if !is_safe_plan_id(tier) {
            findings.push(PlanDiagnostic::plan(
                "referenceSpec.model.tier",
                format!("tier {tier:?} must be 1-64 characters of [A-Za-z0-9_-]"),
            ));
        }
    }
    findings.extend(spec_resolution_finding(
        "referenceSpec.model.resolution",
        spec.model.resolution.as_deref(),
    ));
    findings.extend(spec_prompt_length_finding(
        "referenceSpec.model.negativePrompt",
        spec.model.negative_prompt.as_deref(),
    ));
    if !(1..=MAX_REFERENCE_SPEC_JOB_SECONDS).contains(&spec.limits.max_job_seconds) {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.limits.maxJobSeconds",
            format!(
                "a reference spec must declare a per-job budget between 1 and \
                 {MAX_REFERENCE_SPEC_JOB_SECONDS} seconds (got {})",
                spec.limits.max_job_seconds
            ),
        ));
    }
    if !(1..=MAX_REFERENCE_SPEC_ATTEMPTS).contains(&spec.limits.max_attempts_per_role) {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.limits.maxAttemptsPerRole",
            format!(
                "attempts per role must be between 1 and {MAX_REFERENCE_SPEC_ATTEMPTS} (got {})",
                spec.limits.max_attempts_per_role
            ),
        ));
    }
    if !(spec.limits.max_memory_gb.is_finite() && spec.limits.max_memory_gb > 0.0) {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.limits.maxMemoryGb",
            "a reference spec must declare a positive memory budget",
        ));
    }
    if spec.references.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "referenceSpec.references",
            "a reference spec needs at least one role to generate",
        ));
    }
    let mut seen = BTreeSet::new();
    let mut files = BTreeSet::new();
    for (index, entry) in spec.references.iter().enumerate() {
        let field = format!("referenceSpec.references[{index}]");
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
        let file_findings = reference_image_file_findings(&field, &entry.file);
        if file_findings.is_empty() && !files.insert(entry.file.as_str()) {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.file"),
                format!(
                    "file {:?} is declared by more than one role; each role writes its own plate",
                    entry.file
                ),
            ));
        }
        findings.extend(file_findings);
        if entry.prompt.trim().is_empty() {
            findings.push(PlanDiagnostic::plan(
                format!("{field}.prompt"),
                format!(
                    "role {:?} declares no prompt; every generated role needs the text it renders \
                     from",
                    entry.role
                ),
            ));
        }
        findings.extend(spec_prompt_length_finding(
            &format!("{field}.prompt"),
            Some(entry.prompt.as_str()),
        ));
        findings.extend(spec_prompt_length_finding(
            &format!("{field}.negativePrompt"),
            entry.negative_prompt.as_deref(),
        ));
        findings.extend(spec_resolution_finding(
            &format!("{field}.resolution"),
            entry.resolution.as_deref(),
        ));
    }
    if let Some(inherit) = &spec.inherit {
        let path = Path::new(&inherit.pack);
        if inherit.pack.trim().is_empty()
            || path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            findings.push(PlanDiagnostic::plan(
                "referenceSpec.inherit.pack",
                format!(
                    "pack {:?} must be a relative path beside the spec",
                    inherit.pack
                ),
            ));
        }
        let mut inherited = BTreeSet::new();
        for (index, role) in inherit.references.iter().enumerate() {
            let field = format!("referenceSpec.inherit.references[{index}]");
            if !is_safe_plan_id(role) {
                findings.push(PlanDiagnostic::plan(
                    field,
                    format!("role {role:?} must be 1-64 characters of [A-Za-z0-9_-]"),
                ));
            } else if !inherited.insert(role.as_str()) {
                findings.push(PlanDiagnostic::plan(
                    field,
                    format!("duplicate inherited role {role:?}"),
                ));
            } else if seen.contains(role.as_str()) {
                findings.push(PlanDiagnostic::plan(
                    field,
                    format!("role {role:?} is generated by this spec and cannot also be inherited"),
                ));
            }
        }
    }
    findings
}

fn spec_resolution_finding(field: &str, resolution: Option<&str>) -> Option<PlanDiagnostic> {
    let value = resolution?;
    parse_resolution(value)
        .is_none()
        .then(|| PlanDiagnostic::plan(field, format!("resolution {value:?} must be \"WxH\"")))
}

fn spec_prompt_length_finding(field: &str, text: Option<&str>) -> Option<PlanDiagnostic> {
    let value = text?;
    (value.chars().count() > MAX_PROMPT_CHARS).then(|| {
        PlanDiagnostic::plan(
            field,
            format!(
                "prompt is {} characters; the generation route accepts at most {MAX_PROMPT_CHARS}",
                value.chars().count()
            ),
        )
    })
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

// ---------------------------------------------------------------------------------------------
// Model partitions (sc-23402)
// ---------------------------------------------------------------------------------------------

/// Families whose REFERENCE conditioning ships as a second catalog entry, as
/// `(base model id, reference partition model id)`.
///
/// MiniMax-H3 is two 18.78 GB DiT checkpoints under one family: `minimax_h3` serves
/// `text_to_video | image_to_video | first_last_frame` and declares `limits.maxReferenceAssets: 0`,
/// while `minimax_h3_ref` serves `reference_to_video` ONLY and declares 9 images / 3 clips / 3
/// audio (`config/manifests/builtin.models.jsonc`; the routing arm that refuses the other pairings
/// is `jobs_store::routing::mlx`, "routing a t2v request at the reference one loads the wrong
/// checkpoint").
///
/// A plan therefore declares the FAMILY once — `model.id: "minimax_h3"` — and each shot resolves to
/// the partition its own conditioning needs. The alternative, a per-shot model override, would let
/// a plan mix unrelated families inside one sequence; this table cannot, because the only id it can
/// ever produce is the declared model's own reference partition.
const REFERENCE_PARTITIONS: &[(&str, &str)] = &[("minimax_h3", "minimax_h3_ref")];

/// The reference partition of `model_id`, when its family has one.
pub fn reference_partition_for(model_id: &str) -> Option<&'static str> {
    REFERENCE_PARTITIONS
        .iter()
        .find(|(base, _)| *base == model_id)
        .map(|(_, reference)| *reference)
}

/// Whether `model_id` IS a family's reference partition — the half of the split that conditions on
/// references (sc-23402).
///
/// Read off the same table [`reference_partition_for`] reads, so "this request carries references"
/// cannot be decided by one rule in the compiler and another in the recorder. It is what gates the
/// reference-only knobs (`referenceImageShortEdge`): a base-partition request has no reference to
/// size, so the knob must not appear on it at all.
pub fn is_reference_partition_id(model_id: &str) -> bool {
    REFERENCE_PARTITIONS
        .iter()
        .any(|(_, reference)| *reference == model_id)
}

/// Which catalog entry one shot renders through, and why.
///
/// The reason is not decoration: it is written onto the compiled request, the dispatched payload's
/// provenance and the attempt record, so a run says which of a family's checkpoints produced each
/// take without anyone re-deriving it from the plan (epic 23401 E1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShotPartition {
    /// The catalog model id this shot dispatches as.
    pub model_id: String,
    /// One sentence saying why that partition and not the other.
    pub reason: String,
}

/// The partition a shot of `plan_model_id` binding `reference_roles` roles renders through.
///
/// References are OPTIONAL input: a shot that binds none is never refused for it, it simply stays
/// on the plan's declared model. Only a shot that actually binds reference roles moves, and only
/// when the declared model's family HAS a reference partition — otherwise it stays put and
/// [`validate_plan_against_model`] refuses it against the declared model's own
/// `limits.maxReferenceAssets`, which is the same refusal a single-entry family has always given.
pub fn resolve_shot_partition(plan_model_id: &str, reference_roles: usize) -> ShotPartition {
    match (reference_roles, reference_partition_for(plan_model_id)) {
        (0, _) | (_, None) => ShotPartition {
            model_id: plan_model_id.to_owned(),
            reason: if reference_roles == 0 {
                format!("no reference roles; renders on the plan's model {plan_model_id}")
            } else {
                format!(
                    "{reference_roles} reference role(s); {plan_model_id} has no separate \
                     reference partition, so the shot renders on it directly"
                )
            },
        },
        (_, Some(reference)) => ShotPartition {
            model_id: reference.to_owned(),
            reason: format!(
                "{reference_roles} reference role(s); {plan_model_id} declares no reference \
                 conditioning, so the shot renders on its family's reference partition {reference}"
            ),
        },
    }
}

/// The catalog entries a plan's shots may resolve to: the plan's declared model, plus the family's
/// reference partition when the catalog serves one.
///
/// It is a view over borrowed entries rather than an owned map so the one resolution rule
/// ([`Self::resolve`]) is shared by the validator, the compiler and the harness driver. A shot is
/// checked against the limits of the entry it will ACTUALLY dispatch as — the whole point of the
/// split, since the two partitions disagree on `capabilities` and `limits.maxReferenceAssets`.
#[derive(Debug, Clone, Copy)]
pub struct ModelEntries<'a> {
    base_id: &'a str,
    base: &'a Map<String, Value>,
    reference: Option<(&'a str, &'a Map<String, Value>)>,
}

impl<'a> ModelEntries<'a> {
    /// A plan whose shots all render through one entry: no reference partition is available, so a
    /// shot that binds references is judged against `entry`'s own declared caps.
    pub fn single(model_id: &'a str, entry: &'a Map<String, Value>) -> Self {
        Self {
            base_id: model_id,
            base: entry,
            reference: None,
        }
    }

    /// The plan's entry plus the family's reference partition as the catalog serves it. `reference`
    /// is `None` when the catalog has no such entry — which becomes a finding on the first shot
    /// that needs it, never a silent dispatch at the base checkpoint.
    pub fn with_reference_partition(
        model_id: &'a str,
        entry: &'a Map<String, Value>,
        reference: Option<(&'a str, &'a Map<String, Value>)>,
    ) -> Self {
        Self {
            base_id: model_id,
            base: entry,
            reference,
        }
    }

    pub fn base_entry(&self) -> &'a Map<String, Value> {
        self.base
    }

    /// The partition a shot binding `reference_roles` roles resolves to, with its catalog entry.
    /// The entry is `None` exactly when the resolved partition is not one this view holds.
    fn resolve(&self, reference_roles: usize) -> (ShotPartition, Option<&'a Map<String, Value>>) {
        let partition = resolve_shot_partition(self.base_id, reference_roles);
        if partition.model_id == self.base_id {
            return (partition, Some(self.base));
        }
        let entry = self
            .reference
            .and_then(|(id, entry)| (id == partition.model_id).then_some(entry));
        (partition, entry)
    }

    /// The partition `shot` resolves to, with its catalog entry.
    pub fn resolve_shot(&self, shot: &Shot) -> (ShotPartition, Option<&'a Map<String, Value>>) {
        self.resolve(shot.conditioning.reference_roles.len())
    }

    /// The catalog entry for an already-resolved partition, paired back with it.
    pub fn resolve_shot_partition_entry(
        &self,
        partition: &ShotPartition,
    ) -> (&'a str, Option<&'a Map<String, Value>>) {
        if partition.model_id == self.base_id {
            return (self.base_id, Some(self.base));
        }
        match self.reference {
            Some((id, entry)) if id == partition.model_id => (id, Some(entry)),
            _ => (self.base_id, None),
        }
    }

    /// Every distinct partition `plan`'s shots resolve to, in plan order.
    pub fn partitions_used(&self, plan: &ProductionPlan) -> Vec<ShotPartition> {
        let mut used: Vec<ShotPartition> = Vec::new();
        for shot in &plan.shots {
            let (partition, _) = self.resolve_shot(shot);
            if !used.iter().any(|seen| seen.model_id == partition.model_id) {
                used.push(partition);
            }
        }
        if used.is_empty() {
            used.push(resolve_shot_partition(self.base_id, 0));
        }
        used
    }
}

// ---------------------------------------------------------------------------------------------
// Plan LoRAs (sc-23406)
// ---------------------------------------------------------------------------------------------

/// One catalog LoRA, as the plan validator and the compiler need it.
///
/// Read from the EMBEDDED builtin manifest rather than from the API's live catalog, for the same
/// reason [`crate::minimax_h3_turbo`] resolves its recipe there: the identity of a published
/// adapter — which family it belongs to, which partitions it was distilled for, what weight it
/// folds at — is a property of the shipped catalog, not of the host. Install state is the one fact
/// that IS per-host, and it is gated where it belongs: the video route refuses an uninstalled
/// adapter at enqueue, and the planner's envelope offers only installed ones.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanLoraEntry {
    pub id: String,
    /// Display name — what a refusal names, so an operator reads the adapter rather than the key.
    pub name: String,
    /// The catalog family token (`minimax-h3`).
    pub family: String,
    /// The `modelIds` allowlist. EMPTY means family-wide: the adapter attaches to any model of its
    /// family. Non-empty means it was distilled for exactly those partitions (sc-19563).
    pub model_ids: Vec<String>,
    /// The runtime `lora_scale` multiplier the catalog declares (`defaultWeight`), which is what
    /// the Studio sends and what the route would fill in for a request that omits it.
    pub default_weight: f64,
}

impl PlanLoraEntry {
    /// Whether this adapter may attach to catalog model `model_id` on its declared allowlist alone.
    /// Family compatibility is a separate question ([`lora_family_matches_model`]).
    pub fn allows_model(&self, model_id: &str) -> bool {
        self.model_ids.is_empty() || self.model_ids.iter().any(|id| id == model_id)
    }
}

static BUILTIN_PLAN_LORAS: std::sync::OnceLock<Vec<PlanLoraEntry>> = std::sync::OnceLock::new();

/// Every LoRA in the embedded builtin catalog, in catalog order.
pub fn builtin_plan_loras() -> &'static [PlanLoraEntry] {
    BUILTIN_PLAN_LORAS.get_or_init(|| parse_plan_loras(embedded_manifest("builtin.loras.jsonc")))
}

/// The embedded builtin catalog entry for `id`, or `None` when the id names nothing shipped.
pub fn builtin_plan_lora(id: &str) -> Option<&'static PlanLoraEntry> {
    builtin_plan_loras().iter().find(|lora| lora.id == id)
}

fn embedded_manifest(name: &str) -> &'static str {
    crate::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(manifest, _)| *manifest == name)
        .map(|(_, contents)| *contents)
        .unwrap_or("")
}

/// Parse a `builtin.loras.jsonc` body. Split out so tests can drive synthetic catalogs; a malformed
/// manifest yields an EMPTY list rather than a panic, exactly as [`crate::minimax_h3_turbo`] does.
fn parse_plan_loras(contents: &str) -> Vec<PlanLoraEntry> {
    let stripped = strip_jsonc_comments(contents);
    let Ok(manifest) = serde_json::from_str::<Value>(&stripped) else {
        return Vec::new();
    };
    let Some(loras) = manifest.get("loras").and_then(Value::as_array) else {
        return Vec::new();
    };
    loras
        .iter()
        .filter_map(|lora| {
            let id = lora.get("id")?.as_str()?.to_owned();
            let family = lora.get("family")?.as_str()?.to_owned();
            let model_ids = lora
                .get("modelIds")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|id| !id.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let name = lora
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&id)
                .to_owned();
            Some(PlanLoraEntry {
                id,
                name,
                family,
                model_ids,
                default_weight: lora
                    .get("defaultWeight")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0),
            })
        })
        .collect()
}

/// The LoRA families catalog model `model_id` declares it can load
/// (`loraCompatibility.families`), read from the embedded model catalog.
///
/// Empty for a model the embedded catalog does not hold — a user-installed or external entry —
/// which makes [`lora_family_matches_model`] permissive there rather than refusing a pairing this
/// side cannot judge. The video route's own compatibility gate still judges it at enqueue.
///
/// Parsed ONCE ([`builtin_plan_loras`] does the same for the LoRA catalog): the embedded model
/// manifest is a compile-time constant, and re-parsing it per call put a whole-catalog JSON parse
/// inside [`plan_loras_for_partition`], which every shot's compile and every envelope offer runs.
pub fn model_lora_families(model_id: &str) -> Vec<String> {
    static MODEL_LORA_FAMILIES: std::sync::OnceLock<BTreeMap<String, Vec<String>>> =
        std::sync::OnceLock::new();
    MODEL_LORA_FAMILIES
        .get_or_init(|| {
            let stripped = strip_jsonc_comments(embedded_manifest("builtin.models.jsonc"));
            let Ok(manifest) = serde_json::from_str::<Value>(&stripped) else {
                return BTreeMap::new();
            };
            manifest
                .get("models")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|model| {
                    let id = model.get("id").and_then(Value::as_str)?.to_owned();
                    let families = model
                        .get("loraCompatibility")?
                        .get("families")?
                        .as_array()?
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect();
                    Some((id, families))
                })
                .collect()
        })
        .get(model_id)
        .cloned()
        .unwrap_or_default()
}

/// Whether `lora`'s family is one catalog model `model_id` declares it can load. Permissive for a
/// model the embedded catalog does not hold; see [`model_lora_families`].
pub fn lora_family_matches_model(lora: &PlanLoraEntry, model_id: &str) -> bool {
    let families = model_lora_families(model_id);
    families.is_empty() || families.contains(&lora.family)
}

/// The partitions a plan declaring `plan_model_id` may ever dispatch as: the declared model, plus
/// its family's reference partition when the catalog serves one.
///
/// Read off the same table [`resolve_shot_partition`] reads, so the set the validator judges a
/// LoRA list against is exactly the set the compiler can produce.
pub fn plan_partition_ids(plan_model_id: &str) -> Vec<String> {
    let mut ids = vec![plan_model_id.to_owned()];
    if let Some(reference) = reference_partition_for(plan_model_id) {
        ids.push(reference.to_owned());
    }
    ids
}

/// The plan's declared LoRAs that actually attach to `partition_model_id`, in the plan's own order.
///
/// This is the per-shot resolution sc-23406 asks for, and it is ONE function so the compiled
/// request, the dispatched payload and the attempt record cannot each answer it differently. Both
/// halves of the question are asked: the adapter's declared `modelIds` allowlist (a ref2v turbo
/// never reaches a base-partition shot, and an fl2v turbo never reaches a reference one) and its
/// family against the partition's own declared `loraCompatibility.families`.
///
/// An id that names nothing in the embedded catalog is skipped rather than guessed at;
/// [`validate_plan_structure`] has already refused it by name, so reaching here means the caller
/// chose to compile a plan it was told not to.
pub fn plan_loras_for_partition(
    plan_lora_ids: &[String],
    partition_model_id: &str,
) -> Vec<&'static PlanLoraEntry> {
    plan_lora_ids
        .iter()
        .filter_map(|id| builtin_plan_lora(id))
        .filter(|lora| lora.allows_model(partition_model_id))
        .filter(|lora| lora_family_matches_model(lora, partition_model_id))
        .collect()
}

/// The `loras` entries a job body carries for `partition_model_id` — the SHAPE the Video Studio
/// sends, not a second spelling of it.
///
/// `generationStudio.jsx` posts `selectedLoras.map((lora) => ({ id, weight }))`, the route's
/// `hydrate_lora_spec` keys on `id` and `preset_lora_weight` fills an omitted weight from the
/// catalog's `defaultWeight`. Sending the id AND the declared weight is therefore byte-identical to
/// what the studio sends for the same selection, and identical to what the route would have filled
/// in for an id alone — which is the property that makes a harness render and a studio render the
/// same render.
pub fn plan_lora_payload_entries(plan_lora_ids: &[String], partition_model_id: &str) -> Vec<Value> {
    plan_loras_for_partition(plan_lora_ids, partition_model_id)
        .into_iter()
        .map(|lora| json!({ "id": lora.id, "weight": lora.default_weight }))
        .collect()
}

/// Findings on the plan's `model.loras` and `model.advanced.steps` (sc-23406).
///
/// Four refusals, each by NAME, because every one of them is a document the author can fix in the
/// plan and none of them is worth a model load to discover:
///
/// 1. an id no shipped catalog entry carries — a typo, refused with the id in the message;
/// 2. the same id twice — a selection the route would silently collapse;
/// 3. an adapter whose family the plan's model cannot load at all;
/// 4. two step-distill recipes that would BOTH apply to one partition. A render has one schedule,
///    so the harness refuses here rather than letting `resolve_turbo_recipe` refuse it on the
///    worker after the weights are resident.
fn validate_plan_loras(plan: &ProductionPlan) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let families = model_lora_families(&plan.model.id);
    for id in &plan.model.loras {
        if !seen.insert(id.as_str()) {
            findings.push(PlanDiagnostic::plan(
                "model.loras",
                format!("{id:?} is listed twice; declare each LoRA once"),
            ));
            continue;
        }
        let Some(lora) = builtin_plan_lora(id) else {
            findings.push(PlanDiagnostic::plan(
                "model.loras",
                format!(
                    "{id:?} is not a LoRA in this build's catalog; the installed ids are listed by \
                     `GET /api/v1/loras`"
                ),
            ));
            continue;
        };
        if !families.is_empty() && !families.contains(&lora.family) {
            findings.push(PlanDiagnostic::plan(
                "model.loras",
                format!(
                    "{:?} is a {} LoRA, which {} cannot load (it declares {})",
                    lora.id,
                    lora.family,
                    plan.model.id,
                    families.join(", ")
                ),
            ));
        }
    }
    // One recipe per partition. Checked per partition rather than across the list, because a plan
    // that names the fl2v turbo AND the ref2v turbo is CORRECT — they reach different checkpoints —
    // while two fl2v turbos would both reach the base one.
    //
    // The judgement is [`crate::minimax_h3_turbo::resolve_turbo_recipe`]'s, not a second copy of
    // it: it is the resolver the WORKER runs, on the same payload entries this plan will dispatch,
    // so a list this accepts cannot be one the worker then refuses after the weights are resident.
    // It also already draws the distinctions a hand-rolled count gets wrong — the same id twice,
    // and two distinct adapters that declare the SAME schedule, are neither of them conflicts.
    for partition in plan_partition_ids(&plan.model.id) {
        let entries = plan_lora_payload_entries(&plan.model.loras, &partition);
        if let Err(error) = crate::minimax_h3_turbo::resolve_turbo_recipe(&partition, &entries) {
            findings.push(PlanDiagnostic::plan("model.loras", error));
        }
    }
    if let Some(steps) = plan
        .model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.steps)
    {
        // Both ends, because the COMPILER's `u32::try_from(..).ok()` silently drops anything
        // outside the range: a `steps` above `u32::MAX` validated clean and then rendered at the
        // recipe's own count, with no field named and nothing in the record to say the override
        // was discarded. The validator owns the whole admitted range, so what validates compiles.
        if steps < 1 || u32::try_from(steps).is_err() {
            findings.push(PlanDiagnostic::plan(
                "model.advanced.steps",
                format!(
                    "steps must be a model-evaluation count between 1 and {}, got {steps}",
                    u32::MAX
                ),
            ));
        }
    }
    findings
}

/// The finding for a shot whose resolved partition is not in the catalog this run judged against.
fn missing_partition_finding(shot_id: &str, partition: &ShotPartition) -> PlanDiagnostic {
    PlanDiagnostic::shot(
        shot_id,
        "conditioning.referenceRoles",
        format!(
            "{} is not in this API's model catalog, so this shot cannot be dispatched ({}); \
             install it in the Model Manager, or drop the shot's reference roles",
            partition.model_id, partition.reason
        ),
    )
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
///
/// Every per-shot rule is checked against the entry of the partition that shot RESOLVES to
/// (sc-23402), not against the plan's declared model: on a split family the two entries declare
/// different `capabilities` and different `limits.maxReferenceAssets`, so judging a
/// `reference_to_video` shot against the base entry would refuse a shot the route would have
/// accepted, and judging a `text_to_video` shot against the reference entry would refuse one the
/// base checkpoint renders every day.
///
/// Takes the PACK because one of those rules — `limits.maxReferenceAssets` — counts the images a
/// shot supplies, and only the pack says which of a shot's roles share one (sc-24024).
pub fn validate_plan_against_model(
    plan: &ProductionPlan,
    pack: &ReferencePack,
    entries: &ModelEntries<'_>,
    lane: ModelLane,
) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let model_id = plan.model.id.as_str();
    let entry = entries.base_entry();
    // `limits.maxMemoryGb` bounds ONE JOB's observed peak (`AttemptRecord::peak_memory_gb`), and
    // shots dispatch one job at a time — a run never has two partitions resident at once. So the
    // budget has to clear the LARGEST declared minimum among the partitions this plan uses, not
    // their sum: every partition must fit on its own, and the largest is the binding one. Checking
    // the declared model's alone would let a plan whose reference shots need more sail past
    // preflight and blow the budget mid-run.
    let binding = entries
        .partitions_used(plan)
        .into_iter()
        .filter_map(|partition| {
            let (_, entry) = entries.resolve_shot_partition_entry(&partition);
            let minimum = model_min_memory_gb(entry?, lane)?;
            Some((partition.model_id, minimum))
        })
        .max_by(|(_, left), (_, right)| left.total_cmp(right));
    if let Some((partition_id, minimum)) = binding {
        if plan.limits.max_memory_gb < minimum {
            findings.push(PlanDiagnostic::plan(
                "limits.maxMemoryGb",
                format!(
                    "budget {} GB is below {partition_id}'s declared {}.minMemoryGb of {minimum} \
                     GB",
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

    for shot in &plan.shots {
        let id = shot.id.as_str();
        let (partition, partition_entry) = entries.resolve_shot(shot);
        let Some(entry) = partition_entry else {
            findings.push(missing_partition_finding(id, &partition));
            continue;
        };
        let model_id = partition.model_id.as_str();
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
        // A partition whose fps menu disagrees with the plan's would render a different cadence
        // than the sequence is cut at; the plan-level check above only saw the base entry.
        if partition.model_id != plan.model.id {
            if let Some(message) = fps_limit_error(model_id, fps, entry) {
                findings.push(PlanDiagnostic::shot(id, "model.fps", message));
            }
        }
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
        // PICTURES, not roles (sc-24024). `maxReferenceAssets` bounds the images the request
        // SUPPLIES, and roles that share a file are supplied once — so a ten-role shot over nine
        // files is inside a cap of nine, and counting roles would refuse a request the route
        // accepts. Counted by the one function that decides the supply order, so the number checked
        // here is the length of the list `resolve_conditioning` will build.
        let pictures =
            crate::film_compile::shot_reference_pictures(&shot.conditioning.reference_roles, pack)
                .len();
        if pictures > caps.images {
            let roles = shot.conditioning.reference_roles.len();
            findings.push(PlanDiagnostic::shot(
                id,
                "conditioning.referenceRoles",
                format!(
                    "{roles} reference roles supply {pictures} distinct images, exceeding \
                     {model_id}'s limits.maxReferenceAssets of {}",
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
            // The nearest legal values on either side are named as the correction, because a
            // planner told only that a value is off the menu re-derives the "right" one — the
            // real local planner wrote 6.8333 between 6.5833 and 7.2917 (sc-23406) — while a value
            // it is shown, it copies.
            let below = duration_menu
                .iter()
                .copied()
                .filter(|candidate| *candidate < duration)
                .fold(None, |best: Option<f64>, candidate| {
                    Some(best.map_or(candidate, |best| best.max(candidate)))
                });
            let above = duration_menu
                .iter()
                .copied()
                .filter(|candidate| *candidate > duration)
                .fold(None, |best: Option<f64>, candidate| {
                    Some(best.map_or(candidate, |best| best.min(candidate)))
                });
            let nearest: Vec<String> = [below, above]
                .into_iter()
                .flatten()
                .map(|value| format!("{value}"))
                .collect();
            findings.push(PlanDiagnostic::shot(
                id,
                "targetDurationSeconds",
                format!(
                    "{duration}s is not on {model_id}'s duration menu {duration_menu:?}; the \
                     engine would render a different length than the plan intends — write {} \
                     instead",
                    nearest.join(" or ")
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
    model_entry: Option<(&ModelEntries<'_>, ModelLane)>,
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
    if let Some((entries, lane)) = model_entry {
        findings.extend(validate_plan_against_model(plan, pack, entries, lane));
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
    /// Primary weights download for the requested tier, as the manifest declares it. The DECLARED
    /// model's own row — see `partition_weights` for what a mixed run actually loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<Value>,
    /// The primary weights download for the requested tier of EVERY partition this run dispatches
    /// on, keyed by catalog model id (sc-23402 review). On a split family the reference partition's
    /// `transformer_ref` rows are a second 18.78 GB download that `weights` above never named, so a
    /// mixed run's record could not say which files produced its reference takes. A run that uses
    /// one partition carries one entry, and it is the same row as `weights`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub partition_weights: BTreeMap<String, Value>,
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

/// One dialogue line this run SYNTHESIZED (sc-23404) — the provenance of a clip nobody recorded,
/// kept beside the imported-clip record rather than folded into it.
///
/// Two asset ids are in play and they are not the same thing: [`Self::asset_id`] is the `type:
/// audio` asset the synthesis job produced in the project's library, and the clip the DIALOGUE BUS
/// plays is the pack-directory copy imported afterwards, which appears in [`RunRecord::sound`] like
/// any pre-recorded clip. Keeping both is what lets a reader go from the line in the film back to
/// the job that spoke it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SynthesizedSoundRecord {
    pub role: String,
    /// The exact text the job was sent (trimmed), so a reader can check the line against the plan's
    /// `dialogue` intent without opening the job table.
    pub text: String,
    /// `sha256(text)` — the deterministic half of the clip's filename.
    pub text_sha256: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Which try this is, counting the first. A synthesis that failed is retried by the NEXT
    /// `resume` under a new attempt and therefore a new key — re-polling the failed job under the
    /// old one would make every resume re-read the same failure and never speak the line.
    ///
    /// At most one new attempt per line per invocation, so a retry loop is the operator's to run,
    /// not the harness's to spin; the run's `limits.maxRunSeconds` bounds it either way. The plan's
    /// `maxAttemptsPerShot` deliberately does NOT apply: it caps GPU renders, and capping a
    /// seconds-long TTS call with it would make "resumable" untrue on the fixture's cap of 1.
    #[serde(default = "default_attempt")]
    pub attempt: u32,
    /// Stamped into the dispatched body's `advanced.filmHarness` block, exactly as a render's is:
    /// a controller that died between the POST and this write finds its OWN job instead of speaking
    /// the line twice. Covers model + voice + text, so changing the voice is a different key rather
    /// than an adoption of the wrong clip.
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    /// `dispatching` while the job is being created, `running` while it is polled, then the job's
    /// own terminal status (`completed` / `failed` / `canceled` / `timed_out`).
    pub status: String,
    /// The `type: audio` asset the synthesis job wrote into the project library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    /// Pack-relative path the WAV was written to — the entry's `file` from here on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub started_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
}

fn default_attempt() -> u32 {
    1
}

/// Statuses a synthesis record will not leave on its own. A record in one of these has a job that
/// reached its end; anything else is a job a later pass keeps polling.
pub const TERMINAL_SYNTHESIS_STATUSES: &[&str] = &[
    "completed",
    "failed",
    "rejected",
    "canceled",
    "canceled_by_operator",
    "timed_out",
];

impl SynthesizedSoundRecord {
    /// Whether this line is spoken, written into the pack, and ready to import.
    pub fn is_usable(&self) -> bool {
        self.status == "completed" && self.asset_id.is_some() && self.file.is_some()
    }

    /// Whether this record's job is over, however it ended.
    pub fn is_terminal(&self) -> bool {
        TERMINAL_SYNTHESIS_STATUSES.contains(&self.status.as_str())
    }

    /// Whether this record was made for exactly the line the pack now asks for. A pack edited
    /// between passes re-casts the line rather than adopting the clip that says the old thing.
    pub fn matches(&self, model: &str, voice: Option<&str>, text: &str) -> bool {
        self.model == model && self.voice.as_deref() == voice && self.text == text
    }
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
    /// What the shot was declared to sound like: [`Shot::audio`], the same sentence the compiler
    /// dispatched (sc-24026). Since schema version 3 it is always present, because `audio` is
    /// required on every shot.
    ///
    /// The FIELD keeps the name `sound` for run-record compatibility: it is serialized into run
    /// records already on disk, written when it held the optional version-2 `shots[].sound` prose,
    /// and renaming it would make every one of those records fail to read back.
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
    /// `<runId>:<shotId>:a<attempt>` — written into the job payload and into this record BEFORE the
    /// job is created, so a controller that died between the two finds its own job on replay
    /// instead of enqueuing a second one.
    #[serde(default)]
    pub idempotency_key: String,
    /// The catalog model id this attempt was DISPATCHED as — the partition the shot resolved to,
    /// which on a split family is not the plan's declared model (sc-23402). It is the same string
    /// the compiled request carries and the same string the job payload's `model` holds, so the
    /// three cannot disagree about which checkpoint produced the take.
    #[serde(default)]
    pub resolved_model_id: String,
    /// Why that partition and not the other, in one sentence ([`ShotPartition::reason`]).
    #[serde(default)]
    pub partition_reason: String,
    /// The EFFECTIVE reference-image short edge this attempt was dispatched at, in pixels — the
    /// plan's requested value, or the engine's own default when it named none (sc-23402).
    ///
    /// Present only for an attempt on the family's REFERENCE partition: a base-partition attempt
    /// encodes no reference, so recording a number for it would claim a knob that never applied.
    /// Resolved through [`crate::video_request::effective_reference_image_short_edge`] — the local
    /// twin of gen-core's resolver the engine itself uses — so the recorded value cannot drift from
    /// the rendered one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image_short_edge: Option<u32>,
    /// The catalog LoRA ids this attempt was DISPATCHED with, in the order they ride the payload
    /// (sc-23406). Empty means none applied to this attempt's partition — which is a real,
    /// recorded state, not an absence: a mixed plan that declares only the ref2v turbo renders its
    /// base-partition shots at the full step count, and the empty list beside `effectiveSteps` is
    /// what says so.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<String>,
    /// The model-evaluation count this attempt actually rendered at: `model.advanced.steps` when
    /// the plan set one, else the selected turbo recipe's own count, else the model's declared
    /// `defaults.steps` (sc-23406).
    ///
    /// Recorded rather than derived on read, because the three sources resolve differently per
    /// shot on a mixed plan and a reader comparing a turbo run against a 50-step one needs the
    /// number that ran, not the rule that produced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_steps: Option<u32>,
    /// The video flow-matching sigma shift the selected turbo recipe imposed, when one applied.
    /// Absent in the base regime, where the engine's own `VIDEO_SIGMA_SHIFT` governs — the same
    /// three-state distinction `referenceImageShortEdge` keeps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turbo_scheduler_shift: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    /// `dispatching` until a job id is known, then the job's own status, or `timed_out` /
    /// `canceled` when a limit or a cancel ended the wait.
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
    #[serde(default)]
    pub revision: u64,
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
    /// `trim`, `reorder` or `swap_take`.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline_revision: Option<u64>,
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
    /// Audio layers the export DROPPED from the mix (sc-22715): a placed clip whose asset was
    /// missing from the project, whose media file was gone, or which carried no decodable audio
    /// stream. The worker reports them in the job result (`droppedAudioLayers`) rather than only in
    /// its log, so a sequence that exported with one bus missing says so in the run record.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_audio_layers: Vec<Value>,
}

/// An export the controller is about to dispatch, written BEFORE the job is created so a
/// controller that died in between adopts its own export job instead of starting a second render.
/// `supersedes` is the export job this one replaces (a re-export after a take changed), which is
/// what tells a resume apart the old finished job from the new one on the same timeline.
///
/// `supersedes` names only the MOST RECENT one; the full exclusion set a lookup needs is
/// [`RunRecord::superseded_export_job_ids`], because a second re-export has to exclude the first
/// export as well as the second.
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
    /// Human operation still owed after controller loss; its attempt carries the durable job ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_take_operation: Option<TakeOperation>,
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
    /// The compiled request document this run dispatched from, when one was supplied. Its `sha256`
    /// is what makes a run reproducible: the compiled prompts are the exact text the model saw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiled: Option<SourceDocument>,
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
    /// Sound files the run imported, with the same shape as `references` (sc-22712). A synthesized
    /// line appears here too, once its WAV is in the pack directory — from the dialogue bus's point
    /// of view a spoken line and a recorded one are the same thing.
    #[serde(default)]
    pub sound: Vec<ReferenceAssetRecord>,
    /// Dialogue lines this run SPOKE, with the model, voice, text, job and asset behind each
    /// (sc-23404). Empty on a run whose pack carries only pre-recorded clips.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synthesized_sound: Vec<SynthesizedSoundRecord>,
    #[serde(default)]
    pub shots: Vec<ShotRunRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline: Option<TimelineRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<ExportRecord>,
    /// Set while an export job has been asked for but its id is not recorded yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_pending: Option<ExportPending>,
    /// EVERY export job this record has ever held and then superseded, oldest first.
    ///
    /// The export route carries no field of the harness's own, so an export job is recognised by
    /// the timeline it renders — and the run's timeline is the same one for every export it ever
    /// dispatches. Excluding only the most recently superseded job would let the SECOND re-export
    /// adopt the FIRST export: a finished job whose asset is the timeline from before both
    /// replacements, recorded as current. The whole history is the exclusion set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub superseded_export_job_ids: Vec<String>,
    #[serde(default)]
    pub diagnostics: Vec<PlanDiagnostic>,
    /// Every human decision taken on this run, oldest first.
    #[serde(default)]
    pub decisions: Vec<ProductionDecision>,
    /// Wall-clock the run has spent across every controller that has held it, so the plan's
    /// `limits.maxRunSeconds` bounds the run and not merely one attempt at it.
    ///
    /// AUTOMATIC work only. A `replace-take` / `request-repair` is a human decision that
    /// authorises one bounded attempt outside the plan's automatic budgets, so its wall-clock is
    /// booked in [`RunRecord::human_requested_elapsed_seconds`] instead — otherwise one replacement
    /// could exhaust `maxRunSeconds` and make the run's own `resume` refuse with "raise
    /// limits.maxRunSeconds" (sc-22715).
    pub elapsed_seconds: f64,
    /// Wall-clock spent on human-requested attempts and their exports, kept apart from
    /// `elapsedSeconds` so the terminal evaluation can still report the run's TOTAL cost
    /// (`elapsedSeconds + humanRequestedElapsedSeconds`).
    #[serde(default)]
    pub human_requested_elapsed_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TakeOperation {
    pub kind: String,
    pub shot_id: String,
    pub attempt: u32,
    pub idempotency_key: String,
    pub prior_outcome: RunOutcome,
    pub prior_stop: Option<RunStop>,
    pub reason: String,
    pub export: bool,
    pub max_shot_seconds: u64,
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
            "schemaVersion": PLAN_SCHEMA_VERSION,
            "id": "courier-workshop",
            "version": 1,
            "title": "Courier",
            "model": { "id": "minimax_h3", "tier": "q4", "resolution": "576x320" },
            "limits": { "maxRunSeconds": 3600, "maxShotSeconds": 1800, "maxAttemptsPerShot": 1, "maxMemoryGb": 96 },
            "shots": [
                {
                    "id": "SH010", "beat": "enter", "framing": "wide", "prompt": "a courier enters",
                    "targetDurationSeconds": 5.1667, "startState": "empty", "endState": "courier inside", "audio": "Room tone, no music.",
                    "conditioning": { "mode": "text_to_video" },
                    "continuityRoles": ["red_parcel"]
                },
                {
                    "id": "SH020", "beat": "place", "framing": "medium", "prompt": "places the parcel",
                    "targetDurationSeconds": 5.1667, "startState": "courier inside", "endState": "parcel on table", "audio": "Room tone, no music.",
                    "conditioning": { "mode": "image_to_video", "firstFrameRole": "workshop_plate" },
                    "continuityRoles": ["red_parcel"]
                }
            ]
        })
    }

    fn pack_json() -> Value {
        json!({
            "schemaVersion": REFERENCE_PACK_SCHEMA_VERSION,
            "id": "courier-refs",
            "version": 1,
            "references": [
                { "role": "workshop_plate", "kind": "plate", "file": "references/workshop_plate.png" },
                { "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png" },
                { "role": "unapproved_look", "kind": "style", "file": "references/look.png", "approved": false }
            ]
        })
    }

    /// The single-entry view of a catalog entry, for the tests whose plans use one partition.
    fn single_entries(entry: &Map<String, Value>) -> ModelEntries<'_> {
        ModelEntries::single("minimax_h3", entry)
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

    /// A plan whose LoRA list names nothing the shipped catalog carries is refused, and the
    /// refusal NAMES the id — the fix is a one-word edit, so the message has to say which word.
    #[test]
    fn film_loras_refuse_an_id_the_catalog_does_not_carry() {
        let mut document = plan_json();
        document["model"]["loras"] = json!(["minimax_h3_turbo_4step_768"]);
        let plan: ProductionPlan = serde_json::from_value(document).expect("plan parses");
        let findings = validate_plan_structure(&plan);
        assert_eq!(findings.len(), 1, "{:?}", messages(&findings));
        assert_eq!(findings[0].field, "model.loras");
        assert!(
            findings[0].message.contains("minimax_h3_turbo_4step_768")
                && findings[0].message.contains("not a LoRA"),
            "{}",
            findings[0].message
        );
    }

    /// A catalog LoRA of a family this model cannot load is refused by name, and the refusal says
    /// which family it is and which the model declares. This is the arm a plan hits by copying an
    /// id out of another film's plan.
    #[test]
    fn film_loras_refuse_a_family_the_model_cannot_load() {
        let mut document = plan_json();
        document["model"]["loras"] = json!(["scail2_lightning"]);
        let plan: ProductionPlan = serde_json::from_value(document).expect("plan parses");
        let findings = validate_plan_structure(&plan);
        assert_eq!(findings.len(), 1, "{:?}", messages(&findings));
        assert_eq!(findings[0].field, "model.loras");
        assert!(
            findings[0].message.contains("scail2") && findings[0].message.contains("minimax-h3"),
            "the refusal names the LoRA's family and the model's: {}",
            findings[0].message
        );
    }

    /// TWO step-distill accelerators that would both apply to ONE partition is refused by name.
    ///
    /// `minimax_h3_turbo_8step` and `minimax_h3_turbo_4step_v01` both declare
    /// `modelIds: ["minimax_h3"]`, so both reach the base checkpoint and they ask for different
    /// schedules (8 NFE vs 4). A render has one schedule. The pairing that is NOT a conflict —
    /// one fl2v adapter plus the ref2v one, which reach different checkpoints — is asserted in the
    /// same test, because a rule that refused it would make a mixed turbo plan inexpressible.
    #[test]
    fn film_loras_refuse_two_recipes_for_one_partition_but_allow_one_per_partition() {
        let mut document = plan_json();
        document["model"]["loras"] =
            json!(["minimax_h3_turbo_8step", "minimax_h3_turbo_4step_v01"]);
        let plan: ProductionPlan = serde_json::from_value(document).expect("plan parses");
        let findings = validate_plan_structure(&plan);
        assert_eq!(findings.len(), 1, "{:?}", messages(&findings));
        assert_eq!(findings[0].field, "model.loras");
        assert!(
            findings[0].message.starts_with("minimax_h3:")
                && findings[0].message.contains("MiniMax-H3 Turbo (8-step)")
                && findings[0]
                    .message
                    .contains("MiniMax-H3 Turbo (4-step, v0.1)"),
            "the refusal names the partition and BOTH adapters: {}",
            findings[0].message
        );

        // One per partition — the shape `plan.v2.turbo.jsonc` ships — is accepted.
        let mut document = plan_json();
        document["model"]["loras"] =
            json!(["minimax_h3_ref2v_turbo_4step", "minimax_h3_turbo_4step_v01"]);
        let plan: ProductionPlan = serde_json::from_value(document).expect("plan parses");
        assert!(
            validate_plan_structure(&plan).is_empty(),
            "{:?}",
            messages(&validate_plan_structure(&plan))
        );

        // The same id twice is a plain duplicate, refused on its own terms.
        let mut document = plan_json();
        document["model"]["loras"] =
            json!(["minimax_h3_turbo_4step_v01", "minimax_h3_turbo_4step_v01"]);
        let plan: ProductionPlan = serde_json::from_value(document).expect("plan parses");
        let findings = validate_plan_structure(&plan);
        assert_eq!(findings.len(), 1, "{:?}", messages(&findings));
        assert!(
            findings[0].message.contains("listed twice"),
            "{}",
            findings[0].message
        );
    }

    /// `model.advanced.steps` is a model-evaluation count in `1..=u32::MAX`. Zero, a negative and a
    /// value above the compiler's `u32` are all refused by name rather than by a serde error that
    /// names no plan field — which is why the field is typed signed and wide.
    ///
    /// The upper bound is not decoration: `compile_shot` narrows the field with
    /// `u32::try_from(..).ok()`, so a value above `u32::MAX` that validated clean would be dropped
    /// silently and the film would render at the recipe's count with nothing saying so.
    #[test]
    fn film_advanced_steps_must_be_positive() {
        for steps in [0, -4, i64::from(u32::MAX) + 1] {
            let mut document = plan_json();
            document["model"]["advanced"] = json!({ "steps": steps });
            let plan: ProductionPlan = serde_json::from_value(document).expect("plan parses");
            let findings = validate_plan_structure(&plan);
            assert_eq!(
                findings.len(),
                1,
                "steps {steps}: {:?}",
                messages(&findings)
            );
            assert_eq!(findings[0].field, "model.advanced.steps");
            assert!(
                findings[0].message.contains(&steps.to_string()),
                "{}",
                findings[0].message
            );
        }
        // Both ends of the admitted range are INCLUSIVE, so the refusals above are about the range
        // rather than about a number near it.
        for steps in [1, 4, i64::from(u32::MAX)] {
            let mut document = plan_json();
            document["model"]["advanced"] = json!({ "steps": steps });
            let plan: ProductionPlan = serde_json::from_value(document).expect("plan parses");
            assert!(
                validate_plan_structure(&plan).is_empty(),
                "steps {steps}: {:?}",
                messages(&validate_plan_structure(&plan))
            );
        }
    }

    /// A plan that declares NO LoRAs is unchanged: no field is serialized, nothing resolves, and
    /// the payload helper produces nothing to send.
    #[test]
    fn a_plan_without_loras_is_unchanged() {
        let plan: ProductionPlan = serde_json::from_value(plan_json()).expect("plan parses");
        assert!(plan.model.loras.is_empty());
        assert!(validate_plan_structure(&plan).is_empty());
        let round_trip = serde_json::to_value(&plan).expect("serializes");
        assert!(
            round_trip["model"].get("loras").is_none(),
            "an empty list writes no field: {}",
            round_trip["model"]
        );
        assert!(plan_loras_for_partition(&plan.model.loras, "minimax_h3").is_empty());
        assert!(plan_lora_payload_entries(&plan.model.loras, "minimax_h3_ref").is_empty());
    }

    /// 🔴 The `modelIds` allowlist routes each accelerator to its OWN partition, and the payload
    /// entry is the shape the Video Studio sends.
    ///
    /// Both directions are asserted because both are silent failures: the ref2v adapter on the base
    /// checkpoint and an fl2v adapter on the reference one BOTH fold cleanly and render at the
    /// wrong quality (sc-19563). A resolution that emitted the whole list on every partition would
    /// pass any test that only checked "the turbo reached the shot".
    #[test]
    fn the_model_ids_allowlist_routes_each_turbo_to_its_own_partition() {
        let declared = vec![
            "minimax_h3_ref2v_turbo_4step".to_owned(),
            "minimax_h3_turbo_4step_v01".to_owned(),
        ];
        let ids = |partition: &str| -> Vec<String> {
            plan_loras_for_partition(&declared, partition)
                .into_iter()
                .map(|lora| lora.id.clone())
                .collect()
        };
        assert_eq!(ids("minimax_h3_ref"), vec!["minimax_h3_ref2v_turbo_4step"]);
        assert_eq!(ids("minimax_h3"), vec!["minimax_h3_turbo_4step_v01"]);

        // The payload shape: `{ id, weight }` with the catalog's declared `defaultWeight`, which is
        // what `generationStudio.jsx` posts and what `preset_lora_weight` would have filled in.
        assert_eq!(
            plan_lora_payload_entries(&declared, "minimax_h3_ref"),
            vec![json!({ "id": "minimax_h3_ref2v_turbo_4step", "weight": 1.0 })]
        );

        // A partition with no compatible entry gets NOTHING rather than a fallback.
        assert!(plan_loras_for_partition(
            &["minimax_h3_ref2v_turbo_4step".to_owned()],
            "minimax_h3"
        )
        .is_empty());
    }

    /// sc-23402. `model.advanced.referenceImageShortEdge` is admitted over 1024..=2048 INCLUSIVE and
    /// an out-of-range value is REFUSED naming the field and the range — never clamped, since the
    /// value is the reference token budget the author asked for.
    #[test]
    fn plan_refuses_a_reference_image_short_edge_outside_1024_through_2048() {
        let with_edge = |edge: Value| -> ProductionPlan {
            let mut document = plan_json();
            document["model"]["advanced"] = json!({ "referenceImageShortEdge": edge });
            serde_json::from_value(document).expect("plan parses")
        };
        for admitted in [1024, 1536, 2048] {
            assert!(
                validate_plan_structure(&with_edge(json!(admitted))).is_empty(),
                "{admitted} is inside the admitted range: {:?}",
                messages(&validate_plan_structure(&with_edge(json!(admitted))))
            );
        }
        for refused in [0, 1, 1023, 2049, 4096] {
            let findings = validate_plan_structure(&with_edge(json!(refused)));
            assert_eq!(findings.len(), 1, "{:?}", messages(&findings));
            assert_eq!(
                findings[0].field, "model.advanced.referenceImageShortEdge",
                "the refusal names the field"
            );
            assert!(
                findings[0].message.contains("1024")
                    && findings[0].message.contains("2048")
                    && findings[0].message.contains(&refused.to_string()),
                "the refusal names the range and the value, got {:?}",
                findings[0].message
            );
        }
    }

    /// A plan that names no knob is byte-for-byte the plan it was before sc-23402: the block parses
    /// to `None` and serializes with no `advanced` key at all, so an existing plan's sha256 — which
    /// `staleness_findings` compares — does not move.
    #[test]
    fn a_plan_without_the_advanced_block_is_unchanged() {
        let plan = plan();
        assert_eq!(plan.model.advanced, None);
        assert!(validate_plan_structure(&plan).is_empty());
        let round_tripped = serde_json::to_value(&plan).expect("serializes");
        assert!(
            round_tripped["model"].get("advanced").is_none(),
            "an absent block must not serialize a key: {}",
            round_tripped["model"]
        );
        assert!(
            !is_reference_partition_id("minimax_h3"),
            "the base partition is not a reference partition"
        );
        assert!(is_reference_partition_id("minimax_h3_ref"));
    }

    #[test]
    fn well_formed_plan_pack_and_model_produce_no_findings() {
        let plan = plan();
        let pack = pack();
        assert!(validate_plan_structure(&plan).is_empty());
        assert!(validate_reference_pack(&pack).is_empty());
        assert!(validate_plan_against_pack(&plan, &pack).is_empty());
        assert!(validate_plan_against_model(
            &plan,
            &pack,
            &single_entries(&model_entry()),
            ModelLane::Mlx
        )
        .is_empty());
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

    /// sc-24026. Every run pins its `plan.json` and reads it back on resume, replace-take and
    /// review. A run pinned before schema version 3 holds a version 2 document with
    /// `shots[].sound`, and `Shot` is `deny_unknown_fields` — so without the pre-scan the read dies
    /// inside serde at a byte offset, `validate_plan_structure` is never reached, and the operator
    /// is handed a parser position with no remedy in it. The pre-scan is what makes the version
    /// refusal the thing they actually see.
    #[test]
    fn a_version_2_plan_document_is_refused_by_version_with_the_remedy_not_by_serde_position() {
        let mut value = plan_json();
        value["schemaVersion"] = json!(2);
        for shot in value["shots"].as_array_mut().unwrap() {
            let shot = shot.as_object_mut().unwrap();
            shot.remove("audio");
            shot.insert("sound".to_owned(), json!("Room tone. No music."));
        }
        let document = value.to_string();

        let diagnostic =
            parse_plan_document(&document).expect_err("a version 2 plan document is refused");
        assert_eq!(diagnostic.field, "schemaVersion");
        assert!(
            diagnostic
                .message
                .contains("unsupported plan schema version 2")
                && diagnostic.message.contains("shots[].audio")
                && diagnostic
                    .message
                    .contains(&format!("\"schemaVersion\": {PLAN_SCHEMA_VERSION}")),
            "the refusal names the version and the edit that fixes it: {diagnostic:?}"
        );
        assert!(
            !diagnostic.message.contains("unknown field")
                && !diagnostic.message.contains("missing field")
                && !diagnostic.message.contains("line "),
            "the operator gets the remedy, not a serde position: {diagnostic:?}"
        );

        // The pinned path and the CLI path are the same entry, so both get it.
        let flattened = parse_plan(&document).expect_err("parse_plan refuses it too");
        assert_eq!(flattened, diagnostic.message);
        let file = tempfile::NamedTempFile::new().expect("temp plan");
        std::fs::write(file.path(), &document).expect("plan writes");
        let from_file = read_plan_file(file.path()).expect_err("read_plan_file refuses it too");
        assert_eq!(from_file.field, "schemaVersion");
        assert!(
            from_file
                .message
                .contains("unsupported plan schema version 2"),
            "{from_file:?}"
        );

        // A genuine structural error at a supported version still reports as one, so the pre-scan
        // has not swallowed serde's own diagnostics.
        let mut malformed = plan_json();
        malformed["shots"][0]["framng"] = json!("wide");
        let decode =
            parse_plan_document(&malformed.to_string()).expect_err("unknown field still refused");
        assert_eq!(decode.field, "plan");
        assert!(decode.message.contains("framng"), "{decode:?}");
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
        // The dangling and unapproved conditioning slots are the only findings.
    }

    /// Ref2VA treats every BOUND reference as a subject to depict, so only
    /// [`BINDABLE_REFERENCE_KINDS`] may appear in `conditioning.referenceRoles`. The planner
    /// already counts a pack's bindable entries to decide whether a reference shot can be offered
    /// at all; before sc-23401's feature-end pass the validator did not check the kind, so a plan
    /// binding a `plate` (or a `style`) validated, compiled and dispatched it as a subject.
    #[test]
    fn a_plate_or_style_role_bound_as_a_reference_subject_is_refused_naming_shot_role_and_kind() {
        let mut value = plan_json();
        value["shots"][0]["conditioning"] = json!({
            "mode": "reference_to_video",
            "referenceRoles": ["red_parcel", "workshop_plate"]
        });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        // Structurally and geometrically the plan is fine — nothing else refuses it.
        assert!(validate_plan_structure(&plan).is_empty());
        let findings = messages(&validate_plan_against_pack(&plan, &pack()));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].starts_with("[SH010] conditioning.referenceRoles:")
                && findings[0].contains("\"workshop_plate\"")
                && findings[0].contains("\"plate\"")
                && findings[0].contains("character, prop, location"),
            "{findings:?}"
        );

        // A `style` is refused the same way, and APPROVING it does not make the binding legal:
        // the kind rule is about what a bound reference means, not about review state.
        let mut pack_value = pack_json();
        pack_value["references"][2]["approved"] = json!(true);
        let approved_style: ReferencePack =
            serde_json::from_value(pack_value).expect("pack parses");
        let mut value = plan_json();
        value["shots"][0]["conditioning"] = json!({
            "mode": "reference_to_video",
            "referenceRoles": ["unapproved_look"]
        });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_against_pack(&plan, &approved_style));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].starts_with("[SH010] conditioning.referenceRoles:")
                && findings[0].contains("\"unapproved_look\"")
                && findings[0].contains("\"style\""),
            "{findings:?}"
        );

        // The kind rule applies to the BOUND slot only: the same plate in a keyframe slot, and in
        // continuityRoles, is exactly where a plate belongs.
        let mut value = plan_json();
        value["shots"][0]["conditioning"] =
            json!({ "mode": "image_to_video", "firstFrameRole": "workshop_plate" });
        value["shots"][0]["continuityRoles"] = json!(["red_parcel", "workshop_plate"]);
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        assert!(
            validate_plan_against_pack(&plan, &pack()).is_empty(),
            "{:?}",
            messages(&validate_plan_against_pack(&plan, &pack()))
        );

        // And the shipped reference fixture binds only bindable kinds.
        let plan: ProductionPlan = serde_json::from_value(mixed_plan_json()).unwrap();
        assert!(
            validate_plan_against_pack(&plan, &pack()).is_empty(),
            "{:?}",
            messages(&validate_plan_against_pack(&plan, &pack()))
        );
    }

    #[test]
    fn a_nonempty_pack_does_not_invent_bindings_for_reference_free_shots() {
        let mut value = plan_json();
        value["shots"][0]["continuityRoles"] = json!([]);
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        assert!(validate_plan_against_pack(&plan, &pack()).is_empty());

        // A declared chain remains traceability rather than a fabricated reference-pack binding.
        let mut value = plan_json();
        value["shots"][1]["continuityRoles"] = json!([]);
        value["shots"][1]["conditioning"] =
            json!({ "mode": "text_to_video", "chainFromShotId": "SH010" });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        assert!(validate_plan_structure(&plan).is_empty());
        assert!(validate_plan_against_pack(&plan, &pack()).is_empty());

        // A chain on the FIRST shot has nothing to chain from.
        let mut value = plan_json();
        value["shots"][0]["conditioning"] =
            json!({ "mode": "text_to_video", "chainFromShotId": "SH020" });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_structure(&plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("conditioning.chainFromShotId")
                && findings[0].contains("first shot"),
            "{findings:?}"
        );

        // A chain that skips over the shot between is refused: "continues from" is a claim about
        // the cut, and a planner that chains everything back to SH010 says nothing (sc-22713).
        let mut value = plan_json();
        let third = json!({
            "id": "SH030", "beat": "leave", "framing": "wide", "prompt": "the courier leaves",
            "targetDurationSeconds": 5.1667, "startState": "parcel on table", "endState": "empty", "audio": "Room tone, no music.",
            "conditioning": { "mode": "text_to_video", "chainFromShotId": "SH010" },
            "continuityRoles": ["red_parcel"]
        });
        value["shots"].as_array_mut().unwrap().push(third);
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_structure(&plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("[SH030]")
                && findings[0].contains("not the shot immediately before")
                && findings[0].contains("\"SH020\""),
            "{findings:?}"
        );

        // Chained AND canonically anchored is accepted.
        let mut value = plan_json();
        value["shots"][1]["conditioning"] = json!({
            "mode": "image_to_video",
            "firstFrameRole": "workshop_plate",
            "chainFromShotId": "SH010"
        });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        assert!(validate_plan_structure(&plan).is_empty());
        assert!(validate_plan_against_pack(&plan, &pack()).is_empty());
    }

    #[test]
    fn a_first_last_frame_shot_cannot_start_and_end_on_the_same_reference() {
        // What the real local planner produced on every shot of its first draft: the slot shape
        // satisfied by the same plate twice, so the clip is told to end where it began (sc-22713).
        let mut value = plan_json();
        value["shots"][1]["conditioning"] = json!({
            "mode": "first_last_frame",
            "firstFrameRole": "workshop_plate",
            "lastFrameRole": "workshop_plate"
        });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_structure(&plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].starts_with("[SH020] conditioning.lastFrameRole:")
                && findings[0].contains("end exactly where it started"),
            "{findings:?}"
        );

        // Two different roles is the shape the mode is for.
        let mut value = plan_json();
        value["shots"][1]["conditioning"] = json!({
            "mode": "first_last_frame",
            "firstFrameRole": "workshop_plate",
            "lastFrameRole": "red_parcel"
        });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        assert!(validate_plan_structure(&plan).is_empty());
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
        // A shot that binds references but asks for a keyframe mode. It resolves to the reference
        // partition (sc-23402), and the capability check runs against THAT entry's declared modes —
        // which is the whole point of resolving per shot.
        value["shots"][0]["conditioning"] =
            json!({ "mode": "image_to_video", "referenceRoles": ["red_parcel"] });
        value["shots"][0]["negativePrompt"] = json!("blurry");
        value["shots"][1]["targetDurationSeconds"] = json!(6.0);
        value["shots"][1]["resolution"] = json!("640x360");
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let base = model_entry();
        let reference = reference_model_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        let findings = messages(&validate_plan_against_model(
            &plan,
            &pack(),
            &entries,
            ModelLane::Mlx,
        ));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("limits.maxMemoryGb") && m.contains("64")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] conditioning.mode")
                    && m.contains("minimax_h3_ref")
                    && m.contains("image_to_video")),
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
            &pack(),
            &single_entries(&model_entry()),
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

    /// The reference partition's catalog entry, as `minimax_h3_ref` declares itself: the SAME
    /// geometry as the base entry and 9 reference images where the base declares 0
    /// (`config/manifests/builtin.models.jsonc`, asserted equal by
    /// `both_minimax_h3_partitions_declare_one_geometry`).
    fn reference_model_entry() -> Map<String, Value> {
        json!({
            "id": "minimax_h3_ref",
            "capabilities": ["reference_to_video"],
            "video": { "supportsGuidance": false, "supportsNegativePrompt": false },
            "defaults": { "duration": 5.1667, "fps": 24, "resolution": "1344x768" },
            "limits": {
                "durations": [5.1667, 5.875, 14.375],
                "hardMinDuration": 5.1667,
                "hardMaxDuration": 14.375,
                "fps": [24],
                "maxPixels": 1032192,
                "resolutions": ["1344x768", "576x320"],
                "maxReferenceAssets": 9
            },
            "mlx": { "minMemoryGb": 64 }
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    /// A mixed plan: SH010 binds a reference role, SH020 binds none (sc-23402). The bound role is
    /// the pack's only BINDABLE entry — `workshop_plate` is a `plate`, which
    /// [`validate_plan_against_pack`] refuses in this slot.
    fn mixed_plan_json() -> Value {
        let mut value = plan_json();
        value["shots"][0]["conditioning"] = json!({
            "mode": "reference_to_video",
            "referenceRoles": ["red_parcel"]
        });
        value["shots"][1]["conditioning"] = json!({ "mode": "text_to_video" });
        value
    }

    #[test]
    fn each_shot_is_validated_against_the_partition_it_resolves_to() {
        let plan: ProductionPlan = serde_json::from_value(mixed_plan_json()).unwrap();
        let base = model_entry();
        let reference = reference_model_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        // Both shots validate: the reference shot against `minimax_h3_ref` (which declares
        // reference_to_video and 9 images), the bare one against `minimax_h3`.
        let findings = messages(&validate_plan_against_model(
            &plan,
            &pack(),
            &entries,
            ModelLane::Mlx,
        ));
        assert!(findings.is_empty(), "{findings:?}");
        let resolved: Vec<(String, String)> = plan
            .shots
            .iter()
            .map(|shot| {
                let (partition, entry) = entries.resolve_shot(shot);
                assert!(entry.is_some(), "{} resolved to no entry", shot.id);
                (shot.id.clone(), partition.model_id)
            })
            .collect();
        assert_eq!(
            resolved,
            vec![
                ("SH010".to_owned(), "minimax_h3_ref".to_owned()),
                ("SH020".to_owned(), "minimax_h3".to_owned()),
            ]
        );

        // Without the partition in the catalog the reference shot is refused BY NAME rather than
        // dispatched at the base checkpoint, and the shot that binds nothing is untouched.
        let findings = messages(&validate_plan_against_model(
            &plan,
            &pack(),
            &single_entries(&base),
            ModelLane::Mlx,
        ));
        assert!(
            findings.iter().any(|m| m.contains("[SH010]")
                && m.contains("minimax_h3_ref")
                && m.contains("not in this API's model catalog")),
            "{findings:?}"
        );
        assert!(
            !findings.iter().any(|m| m.contains("[SH020]")),
            "{findings:?}"
        );
    }

    #[test]
    fn the_resolved_partitions_limits_are_what_refuse_a_shot() {
        let base = model_entry();
        let reference = reference_model_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );

        // Ten roles exceed the reference partition's declared nine.
        let mut value = mixed_plan_json();
        value["shots"][0]["conditioning"]["referenceRoles"] =
            json!((0..10).map(|i| format!("role_{i}")).collect::<Vec<_>>());
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_against_model(
            &plan,
            &pack(),
            &entries,
            ModelLane::Mlx,
        ));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] conditioning.referenceRoles")
                    && m.contains("maxReferenceAssets")
                    && m.contains("minimax_h3_ref")
                    && m.contains('9')),
            "{findings:?}"
        );

        // A family with no reference partition keeps the old rule: the declared model's own
        // `limits.maxReferenceAssets` is what refuses the shot.
        let plan: ProductionPlan = serde_json::from_value(mixed_plan_json()).unwrap();
        let findings = messages(&validate_plan_against_model(
            &plan,
            &pack(),
            &ModelEntries::single("ltx_2_5", &base),
            ModelLane::Mlx,
        ));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] conditioning.referenceRoles")
                    && m.contains("maxReferenceAssets")
                    && m.contains("ltx_2_5")),
            "{findings:?}"
        );

        // A reference_to_video shot with NO roles never reaches the partition table: it is a
        // structural contradiction, refused naming the shot and the requirement.
        let mut value = mixed_plan_json();
        value["shots"][0]["conditioning"] = json!({ "mode": "reference_to_video" });
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_plan_structure(&plan));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] conditioning.referenceRoles")
                    && m.contains("at least one reference role")),
            "{findings:?}"
        );

        // References are OPTIONAL: a shot that binds none is never refused for it.
        let plan: ProductionPlan = serde_json::from_value(plan_json()).unwrap();
        assert!(
            validate_plan_against_model(&plan, &pack(), &entries, ModelLane::Mlx).is_empty(),
            "a plan with no reference shots must still validate on a split family"
        );
    }

    /// A pack of `roles` character entries over `files` distinct plates: the first
    /// `roles - files + 1` roles share plate 0 and carry locators, the rest get one each
    /// (sc-24024).
    fn crowded_pack(roles: usize, files: usize) -> ReferencePack {
        assert!(files <= roles && files >= 1);
        let shared = roles - files + 1;
        let entries: Vec<Value> = (0..roles)
            .map(|index| {
                let plate = index
                    .saturating_sub(shared.saturating_sub(1))
                    .min(files - 1);
                let mut entry = json!({
                    "role": format!("role_{index}"),
                    "kind": "character",
                    "file": format!("references/plate_{plate}.png"),
                });
                if index < shared && shared > 1 {
                    entry["locator"] = json!(format!("the {index}th person from the left"));
                }
                entry
            })
            .collect();
        serde_json::from_value(json!({
            "schemaVersion": REFERENCE_PACK_SCHEMA_VERSION,
            "id": "crowded-refs",
            "version": 1,
            "references": entries,
        }))
        .expect("the crowded pack parses")
    }

    /// `limits.maxReferenceAssets` bounds the IMAGES a request supplies, and roles sharing a file
    /// are supplied once (sc-24024). So ten roles over nine files fit a cap of nine, and ten roles
    /// over ten files do not — the one difference between the two runs below is which pack the
    /// same plan is counted against.
    #[test]
    fn the_reference_cap_counts_distinct_files_not_bound_roles() {
        let base = model_entry();
        let reference = reference_model_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        let mut value = mixed_plan_json();
        value["shots"][0]["conditioning"]["referenceRoles"] =
            json!((0..10).map(|i| format!("role_{i}")).collect::<Vec<_>>());
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();

        let nine_files = crowded_pack(10, 9);
        assert!(
            validate_reference_pack(&nine_files).is_empty(),
            "{:?}",
            messages(&validate_reference_pack(&nine_files))
        );
        let findings = messages(&validate_plan_against_model(
            &plan,
            &nine_files,
            &entries,
            ModelLane::Mlx,
        ));
        assert!(
            findings.is_empty(),
            "ten roles over nine files supply nine images, inside a cap of nine: {findings:?}"
        );

        let ten_files = crowded_pack(10, 10);
        let findings = messages(&validate_plan_against_model(
            &plan,
            &ten_files,
            &entries,
            ModelLane::Mlx,
        ));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] conditioning.referenceRoles")
                    && m.contains("10 distinct images")
                    && m.contains("maxReferenceAssets")
                    && m.contains('9')),
            "{findings:?}"
        );
    }

    /// The rules a FILE several roles name has to satisfy (sc-24024), each refused by naming the
    /// roles and the file the author has to go and look at.
    #[test]
    fn roles_sharing_one_file_need_distinct_locators_and_one_approval() {
        let shared = |edit: &dyn Fn(&mut Value)| -> ReferencePack {
            let mut value = json!({
                "schemaVersion": REFERENCE_PACK_SCHEMA_VERSION,
                "id": "pair-refs",
                "version": 1,
                "references": [
                    {
                        "role": "courier", "kind": "character", "file": "references/pair.png",
                        "locator": "the woman on the left"
                    },
                    {
                        "role": "recipient", "kind": "character", "file": "references/pair.png",
                        "locator": "the man on the right"
                    }
                ]
            });
            edit(&mut value);
            serde_json::from_value(value).expect("the pair pack parses")
        };

        // The control: two roles on one file, each saying which subject it is, is legal.
        assert!(
            validate_reference_pack(&shared(&|_| {})).is_empty(),
            "{:?}",
            messages(&validate_reference_pack(&shared(&|_| {})))
        );

        // A locator that is absent, or present but blank, is no locator.
        for blank in [json!(null), json!("   ")] {
            let pack = shared(&|value: &mut Value| {
                value["references"][1]["locator"] = blank.clone();
            });
            let findings = messages(&validate_reference_pack(&pack));
            assert!(
                findings.iter().any(|m| m.contains("locator")
                    && m.contains("\"courier\"")
                    && m.contains("\"recipient\"")
                    && m.contains("references/pair.png")),
                "{blank} must be refused, naming both roles and the file: {findings:?}"
            );
        }

        // Two roles picking the SAME subject out of one image says nothing either.
        let findings = messages(&validate_reference_pack(&shared(&|value: &mut Value| {
            value["references"][1]["locator"] = json!("the woman on the left");
        })));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("locator") && m.contains("DIFFERENT")),
            "{findings:?}"
        );

        // One asset carries one approval, so the roles on it cannot disagree about it.
        let findings = messages(&validate_reference_pack(&shared(&|value: &mut Value| {
            value["references"][1]["approved"] = json!(false);
        })));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("approved") && m.contains("references/pair.png")),
            "{findings:?}"
        );

        // A locator reaches the dispatched prompt verbatim, so it may no more forge a picture
        // marker than a description may.
        let findings = messages(&validate_reference_pack(&shared(&|value: &mut Value| {
            value["references"][0]["locator"] = json!("the woman in <Picture 3>");
        })));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("locator") && m.contains("'<' or '>'")),
            "{findings:?}"
        );
    }

    /// Sharing is decided by comparing `file` LITERALLY, so a path the filesystem would resolve to
    /// one image but this pack spells two ways is refused (sc-24024). Left unrefused, one
    /// photograph of two people is imported twice, numbered `<Picture 1>` and `<Picture 2>`, and
    /// — because the entries never group — neither role is asked for a locator: the exact
    /// ambiguity the feature removes, reintroduced by a stray `./`.
    #[test]
    fn a_reference_file_must_already_be_in_canonical_lexical_form() {
        let with_file = |file: &str| -> Vec<String> {
            let mut value = pack_json();
            value["references"][0]["file"] = json!(file);
            let pack: ReferencePack =
                serde_json::from_value(value).expect("the pack parses whatever the path says");
            messages(&validate_reference_pack(&pack))
        };

        // The control: the spelling every checked-in pack uses is accepted.
        assert!(
            with_file("references/workshop_plate.png").is_empty(),
            "{:?}",
            with_file("references/workshop_plate.png")
        );

        for spelling in [
            "./references/workshop_plate.png",
            "references//workshop_plate.png",
            "references/./workshop_plate.png",
            "references/workshop_plate.png/",
            "references\\workshop_plate.png",
        ] {
            let findings = with_file(spelling);
            assert!(
                findings.iter().any(|m| m.contains("canonical form")
                    && m.contains("\"references/workshop_plate.png\"")),
                "{spelling:?} must be refused, naming the canonical spelling: {findings:?}"
            );
        }
    }

    /// Two spellings that differ only by case are ONE file on APFS and two in this pack, so the
    /// pack is refused rather than read as two pictures of one photograph (sc-24024).
    #[test]
    fn two_reference_files_differing_only_by_case_are_refused() {
        let mut value = pack_json();
        value["references"][0]["file"] = json!("references/plate.png");
        value["references"][1]["file"] = json!("references/Plate.png");
        let pack: ReferencePack = serde_json::from_value(value).expect("the pack parses");
        let findings = messages(&validate_reference_pack(&pack));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("differ only by ASCII case")
                    && m.contains("\"references/plate.png\"")
                    && m.contains("\"references/Plate.png\"")),
            "{findings:?}"
        );
    }

    /// A locator is free prose, so nothing can check that it reads as a noun phrase — but the
    /// refusal that ASKS for one has to say that the article is the author's to supply, because
    /// "woman on the left" yields "The courier is woman on the left in <Picture 1>." in silence.
    #[test]
    fn the_missing_locator_refusal_says_the_article_is_the_authors_to_supply() {
        let value = json!({
            "schemaVersion": REFERENCE_PACK_SCHEMA_VERSION,
            "id": "pair-refs",
            "version": 1,
            "references": [
                { "role": "courier", "kind": "character", "file": "references/pair.png" },
                { "role": "recipient", "kind": "character", "file": "references/pair.png" }
            ]
        });
        let pack: ReferencePack = serde_json::from_value(value).expect("the pair pack parses");
        let findings = messages(&validate_reference_pack(&pack));
        assert!(
            findings.iter().any(|m| m.contains("locator")
                && m.contains("The courier is …")
                && m.contains("the woman on the left")),
            "{findings:?}"
        );
    }

    /// A version 1 pack is refused BY VERSION, and the refusal says what to do about it (E6).
    #[test]
    fn a_version_1_reference_pack_is_refused_by_version() {
        let mut value = pack_json();
        value["schemaVersion"] = json!(1);
        let pack: ReferencePack = serde_json::from_value(value).expect("pack parses");
        let findings = validate_reference_pack(&pack);
        assert!(
            findings.iter().any(|finding| {
                finding.field == "referencePack.schemaVersion"
                    && finding.message.contains("version 1")
                    && finding.message.contains("schemaVersion")
                    && finding.message.contains("locator")
            }),
            "{:?}",
            messages(&findings)
        );
    }

    #[test]
    fn the_memory_budget_must_clear_every_partition_so_the_largest_minimum_binds() {
        let mut value = mixed_plan_json();
        value["limits"]["maxMemoryGb"] = json!(70);
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        let base = model_entry();
        let mut reference = reference_model_entry();
        reference.insert("mlx".to_owned(), json!({ "minMemoryGb": 80 }));
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        let findings = messages(&validate_plan_against_model(
            &plan,
            &pack(),
            &entries,
            ModelLane::Mlx,
        ));
        assert!(
            findings.iter().any(|m| m.contains("limits.maxMemoryGb")
                && m.contains("minimax_h3_ref")
                && m.contains("80")),
            "the budget bounds ONE job's peak and each partition must fit on its own, so the \
             larger of the two minimums binds: {findings:?}"
        );
    }

    #[test]
    fn a_family_with_no_reference_partition_resolves_to_itself() {
        let partition = resolve_shot_partition("ltx_2_5", 3);
        assert_eq!(partition.model_id, "ltx_2_5");
        assert!(
            partition.reason.contains("no separate reference partition"),
            "{}",
            partition.reason
        );
        assert_eq!(reference_partition_for("ltx_2_5"), None);
        assert_eq!(
            reference_partition_for("minimax_h3"),
            Some("minimax_h3_ref")
        );
        // The reference partition named as the plan's own model stays put.
        assert_eq!(
            resolve_shot_partition("minimax_h3_ref", 2).model_id,
            "minimax_h3_ref"
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
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
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

    /// sc-24023. A description is repeated into the compiler-owned binding sentence, inside
    /// `insertedText` — the one field `film_compile::CompiledPlan::conformance_findings` reads as
    /// the compiler's own derived text and never checks against anything. So a description that
    /// spells a marker would forge a binding to media the shot never supplies and the compiled
    /// document would still report clean. Refused here, on both entry namespaces, because a later
    /// story binds a sound entry's description the same way (`<Audio N>`).
    #[test]
    fn a_description_that_forges_a_media_marker_is_refused() {
        let mut pack_value = sound_pack_json();
        pack_value["references"][1]["description"] = json!("Actually the parcel in <Picture 3>.");
        pack_value["sound"][2]["description"] = json!("The bed in <Audio 1>");
        let pack: ReferencePack = serde_json::from_value(pack_value).expect("pack parses");
        let findings = messages(&validate_reference_pack(&pack));
        for expected in ["referencePack.references[1].description", "<Picture 3>"] {
            assert!(
                findings.iter().any(|finding| finding.contains(expected)),
                "expected a finding containing {expected:?}: {findings:#?}"
            );
        }
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("referencePack.sound[2].description")),
            "a sound description is bound the same way: {findings:#?}"
        );
        // The shipped shape — ordinary prose, and a description that WRAPS — is untouched: the
        // compiler collapses its whitespace rather than the document refusing it.
        let mut clean = sound_pack_json();
        clean["references"][1]["description"] = json!("Small bright red parcel,\n  centre frame.");
        let clean: ReferencePack = serde_json::from_value(clean).expect("pack parses");
        assert!(
            !messages(&validate_reference_pack(&clean))
                .iter()
                .any(|finding| finding.contains("description")),
            "{:#?}",
            messages(&validate_reference_pack(&clean))
        );
    }

    /// The other half of the same rule: a byte no author typed on purpose never reaches the prompt.
    #[test]
    fn a_description_carrying_a_control_character_is_refused() {
        let mut pack_value = pack_json();
        pack_value["references"][1]["description"] = json!("Small red parcel\u{0007}");
        let pack: ReferencePack = serde_json::from_value(pack_value).expect("pack parses");
        let findings = messages(&validate_reference_pack(&pack));
        assert!(
            findings.iter().any(|finding| finding
                .contains("referencePack.references[1].description")
                && finding.contains("control characters")),
            "{findings:#?}"
        );
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
            compiled: None,
            project_id: None,
            project_path: None,
            model: None,
            limits: plan().limits,
            selected_shot_ids: vec!["SH010".into()],
            references: vec![],
            sound: vec![],
            synthesized_sound: vec![],
            shots: vec![],
            timeline: None,
            export: None,
            export_pending: None,
            superseded_export_job_ids: Vec::new(),
            diagnostics: vec![PlanDiagnostic::shot("SH010", "prompt", "empty")],
            decisions: vec![ProductionDecision {
                at: "2026-09-13T00:00:01Z".into(),
                action: "cancel".into(),
                shot_id: None,
                detail: "operator".into(),
            }],
            elapsed_seconds: 0.0,
            human_requested_elapsed_seconds: 0.0,
            active_take_operation: None,
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
        value["shots"][1]["dependsOn"] = edges;
        serde_json::from_value(value).expect("plan parses")
    }

    #[test]
    fn a_v1_or_v2_plan_is_refused_by_version_and_named_the_remedy() {
        // sc-24026. Both older versions are refused BY VERSION rather than read under a default:
        // neither carries `shots[].audio`, and a plan read with an empty one would dispatch six
        // shots whose soundtrack the model invents while every document reported clean.
        for stale in [1, 2] {
            let mut value = plan_json();
            value["schemaVersion"] = json!(stale);
            let plan: ProductionPlan = serde_json::from_value(value).unwrap();
            let findings = messages(&validate_plan_structure(&plan));
            assert!(
                findings.iter().any(|message| {
                    message.contains("schemaVersion")
                        && message.contains(&stale.to_string())
                        && message.contains("audio")
                }),
                "version {stale} must be refused by version, naming the remedy: {findings:?}"
            );
        }
        // The current version, otherwise identical, is accepted — so the refusal above is about the
        // version and not about anything else in the document.
        let plan = plan();
        assert_eq!(plan.schema_version, PLAN_SCHEMA_VERSION);
        assert!(validate_plan_structure(&plan).is_empty());
        assert!(plan.shots.iter().all(|shot| shot.depends_on.is_empty()));
    }

    /// sc-24026. `audio` is required on every shot: a missing key does not parse, a blank one is a
    /// finding that NAMES the shot, and a value that states silence is simply accepted — the
    /// validator never reads the prose, because "this shot is silent" is a legitimate answer and
    /// the only unanswerable one is saying nothing.
    #[test]
    fn a_shot_without_an_audio_sentence_is_refused_naming_the_shot() {
        let mut value = plan_json();
        value["shots"][1].as_object_mut().unwrap().remove("audio");
        let error = serde_json::from_value::<ProductionPlan>(value)
            .expect_err("a shot with no audio key does not parse")
            .to_string();
        assert!(error.contains("audio"), "{error}");

        for blank in ["", "   ", "\n\t"] {
            let mut value = plan_json();
            value["shots"][1]["audio"] = json!(blank);
            let plan: ProductionPlan = serde_json::from_value(value).unwrap();
            let findings = validate_plan_structure(&plan);
            assert_eq!(findings.len(), 1, "{blank:?}: {findings:?}");
            assert_eq!(findings[0].shot_id.as_deref(), Some("SH020"), "{blank:?}");
            assert_eq!(findings[0].field, "audio", "{blank:?}");
        }

        // Silence, stated. Accepted verbatim and with no finding anywhere.
        let mut value = plan_json();
        value["shots"][1]["audio"] = json!("No audio. Silence.");
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        assert!(
            validate_plan_structure(&plan).is_empty(),
            "{:?}",
            messages(&validate_plan_structure(&plan))
        );
        assert_eq!(plan.shots[1].audio, "No audio. Silence.");
    }

    /// The audio sentence reaches the dispatched prompt verbatim, so it gets the SAME two document
    /// -boundary checks a pack description does (sc-24026) — and gets them as a SHOT finding, so
    /// the operator is told which shot forged the marker.
    #[test]
    fn an_audio_sentence_may_not_forge_a_marker_or_carry_control_characters() {
        for (audio, expected) in [
            ("The bed from <Audio 1>.", "'<' or '>'"),
            ("room tone\u{7}", "control characters"),
        ] {
            let mut value = plan_json();
            value["shots"][0]["audio"] = json!(audio);
            let plan: ProductionPlan = serde_json::from_value(value).unwrap();
            let findings = validate_plan_structure(&plan);
            assert_eq!(findings.len(), 1, "{audio:?}: {findings:?}");
            assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"), "{audio:?}");
            assert_eq!(findings[0].field, "audio", "{audio:?}");
            assert!(findings[0].message.contains(expected), "{:?}", findings[0]);
        }
        // Newlines and tabs are NOT refused — `film_compile::normalized_description` collapses them
        // before the sentence reaches a prompt, exactly as it does for a pack description.
        let mut value = plan_json();
        value["shots"][0]["audio"] = json!("room tone,\n\ta door latch");
        let plan: ProductionPlan = serde_json::from_value(value).unwrap();
        assert!(validate_plan_structure(&plan).is_empty());
    }

    /// sc-24026. `film_compile::audio_text` writes `Audio: ` in front of the author's words, so an
    /// `audio` value that repeats the label would dispatch `Audio: Audio: room tone`. Refused
    /// naming the shot rather than stripped: the value is authored prose, and silently rewriting it
    /// would make the plan a false record of what was dispatched.
    #[test]
    fn an_audio_sentence_may_not_repeat_the_prefix_the_compiler_writes() {
        for audio in [
            "Audio: room tone",
            "audio: room tone",
            "AUDIO: room tone",
            "  Audio: room tone",
            "Audio:room tone",
        ] {
            let mut value = plan_json();
            value["shots"][0]["audio"] = json!(audio);
            let plan: ProductionPlan = serde_json::from_value(value).unwrap();
            let findings = validate_plan_structure(&plan);
            assert_eq!(findings.len(), 1, "{audio:?}: {findings:?}");
            assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"), "{audio:?}");
            assert_eq!(findings[0].field, "audio", "{audio:?}");
            assert!(
                findings[0]
                    .message
                    .contains("the compiler writes that prefix"),
                "the refusal says whose prefix it is: {:?}",
                findings[0]
            );
        }

        // Only the OPENING label is refused. The word elsewhere, and a sentence that merely starts
        // with "Audio" as a word, are ordinary prose.
        for audio in [
            "Room tone. Audio: mixed low.",
            "Audio equipment hums in the corner.",
        ] {
            let mut value = plan_json();
            value["shots"][0]["audio"] = json!(audio);
            let plan: ProductionPlan = serde_json::from_value(value).unwrap();
            assert!(
                validate_plan_structure(&plan).is_empty(),
                "{audio:?}: {:?}",
                messages(&validate_plan_structure(&plan))
            );
        }
    }

    /// No document this repository ships trips the prefix refusal — the check is a guard on future
    /// authoring, not a break for the plans already checked in.
    #[test]
    fn no_shipped_plan_repeats_the_audio_prefix() {
        let mut checked = 0;
        let shipped = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../config/film-harness/courier-workshop");
        for entry in std::fs::read_dir(&shipped).expect("the shipped plan directory is readable") {
            let path = entry.expect("dir entry").path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if !name.starts_with("plan") || !name.ends_with(".jsonc") {
                continue;
            }
            let plan = read_plan_file(&path).unwrap_or_else(|e| panic!("{name} reads: {e}"));
            for shot in &plan.shots {
                assert!(
                    !shot
                        .audio
                        .trim_start()
                        .to_ascii_lowercase()
                        .starts_with("audio:"),
                    "{name} shot {} repeats the compiler's prefix: {:?}",
                    shot.id,
                    shot.audio
                );
                checked += 1;
            }
        }
        assert!(checked >= 6, "the shipped plans were actually walked");
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
            resolved_model_id: "minimax_h3".into(),
            partition_reason: "no reference roles; renders on the plan's model minimax_h3".into(),
            reference_image_short_edge: None,
            loras: Vec::new(),
            effective_steps: None,
            turbo_scheduler_shift: None,
            job_id: Some(format!("job{number}")),
            status: "completed".into(),
            started_at: "t".into(),
            finished_at: None,
            elapsed_seconds: 1.0,
            peak_gpu_memory_pct: None,
            peak_memory_gb: None,
            peak_memory_source: None,
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
                generated_audio: GeneratedAudio::default(),
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

    // -----------------------------------------------------------------------------------------
    // sc-23404 — synthesized dialogue entries
    // -----------------------------------------------------------------------------------------

    /// A pack carrying exactly `sound`, so the findings under test are the sound findings.
    fn sound_pack(entries: Value) -> ReferencePack {
        serde_json::from_value(json!({
            "schemaVersion": REFERENCE_PACK_SCHEMA_VERSION,
            "id": "pack",
            "version": 1,
            "references": [
                { "role": "plate", "kind": "plate", "file": "references/plate.png" }
            ],
            "sound": entries,
        }))
        .expect("pack parses")
    }

    fn sound_findings(entries: Value) -> Vec<PlanDiagnostic> {
        validate_reference_pack(&sound_pack(entries))
    }

    fn lines(findings: &[PlanDiagnostic]) -> String {
        findings
            .iter()
            .map(|finding| format!("{}: {}", finding.field, finding.message))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn a_dialogue_entry_may_carry_text_instead_of_a_file() {
        let findings = sound_findings(json!([
            { "role": "line", "kind": "dialogue", "text": "Delivery." },
            { "role": "voiced", "kind": "dialogue", "text": "Hello.", "voice": "af_heart", "model": "chatterbox_tts" },
            // `text` AND `file`: synthesis writes INTO the named path.
            { "role": "pinned", "kind": "dialogue", "text": "Oh.", "file": "sound/pinned.wav" },
            { "role": "bed", "kind": "ambience", "file": "sound/bed.wav" },
        ]));
        assert!(findings.is_empty(), "{}", lines(&findings));
    }

    #[test]
    fn an_entry_with_neither_text_nor_file_is_refused_by_role() {
        let findings = sound_findings(json!([
            { "role": "silent_line", "kind": "dialogue", "description": "nothing to play" },
        ]));
        assert_eq!(findings.len(), 1, "{}", lines(&findings));
        assert_eq!(findings[0].field, "referencePack.sound[0].file");
        assert!(
            findings[0].message.contains("silent_line")
                && findings[0].message.contains("neither `file` nor `text`"),
            "the finding must name the entry: {}",
            findings[0].message
        );
    }

    #[test]
    fn text_on_a_non_dialogue_kind_is_refused_by_role() {
        for kind in ["ambience", "music", "sfx"] {
            let findings = sound_findings(json!([
                { "role": "bed", "kind": kind, "text": "a quiet workshop" },
            ]));
            let named: Vec<_> = findings
                .iter()
                .filter(|finding| finding.field == "referencePack.sound[0].text")
                .collect();
            assert_eq!(named.len(), 1, "{kind}: {}", lines(&findings));
            assert!(
                named[0].message.contains("bed") && named[0].message.contains("dialogue"),
                "{kind}: {}",
                named[0].message
            );
        }
    }

    #[test]
    fn synthesis_knobs_without_a_line_to_speak_are_findings() {
        let findings = sound_findings(json!([
            { "role": "recorded", "kind": "dialogue", "file": "sound/recorded.wav",
              "voice": "af_heart", "model": "kokoro_82m" },
        ]));
        let fields: Vec<&str> = findings.iter().map(|f| f.field.as_str()).collect();
        assert_eq!(
            fields,
            vec![
                "referencePack.sound[0].voice",
                "referencePack.sound[0].model"
            ],
            "{}",
            lines(&findings)
        );
    }

    #[test]
    fn an_unknown_speech_model_or_an_overlong_line_is_refused() {
        let findings = sound_findings(json!([
            { "role": "wrong_model", "kind": "dialogue", "text": "hi", "model": "minimax_h3" },
        ]));
        assert_eq!(findings.len(), 1, "{}", lines(&findings));
        assert_eq!(findings[0].field, "referencePack.sound[0].model");
        assert!(
            findings[0].message.contains("kokoro_82m"),
            "{}",
            findings[0].message
        );

        let long = "a".repeat(MAX_DIALOGUE_TEXT_CHARS + 1);
        let findings = sound_findings(json!([
            { "role": "long", "kind": "dialogue", "text": long },
        ]));
        assert_eq!(findings.len(), 1, "{}", lines(&findings));
        assert_eq!(findings[0].field, "referencePack.sound[0].text");

        let findings = sound_findings(json!([
            { "role": "blank", "kind": "dialogue", "text": "   " },
        ]));
        assert_eq!(findings.len(), 1, "{}", lines(&findings));
        assert_eq!(findings[0].field, "referencePack.sound[0].text");

        let findings = sound_findings(json!([
            { "role": "voiceless", "kind": "dialogue", "text": "hi", "voice": "   " },
        ]));
        assert_eq!(findings.len(), 1, "{}", lines(&findings));
        assert_eq!(findings[0].field, "referencePack.sound[0].voice");
    }

    #[test]
    fn every_shipped_speech_model_is_accepted_and_the_default_is_one_of_them() {
        assert!(SOUND_SYNTHESIS_MODELS.contains(&DEFAULT_SOUND_SYNTHESIS_MODEL));
        for model in SOUND_SYNTHESIS_MODELS {
            let findings = sound_findings(json!([
                { "role": "line", "kind": "dialogue", "text": "hi", "model": model },
            ]));
            assert!(findings.is_empty(), "{model}: {}", lines(&findings));
        }
    }

    #[test]
    fn a_synthesized_entry_is_not_checked_for_a_file_on_disk() {
        let dir = std::env::temp_dir().join(format!("film-plan-sound-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("references")).expect("dir");
        std::fs::write(dir.join("references/plate.png"), b"x").expect("plate");
        // Neither `sound/line.wav` (pinned by the synthesized entry) nor the derived name exists
        // yet: the clip is produced by the run, so the existence check must let it through.
        let pack = sound_pack(json!([
            { "role": "line", "kind": "dialogue", "text": "Delivery." },
            { "role": "pinned", "kind": "dialogue", "text": "Oh.", "file": "sound/line.wav" },
        ]));
        let findings = validate_reference_pack_files(&pack, &dir);
        assert!(findings.is_empty(), "{}", lines(&findings));

        // A RECORDED entry whose file is missing is still a finding — this exemption is about
        // synthesis, not about relaxing the check.
        let pack = sound_pack(json!([
            { "role": "recorded", "kind": "dialogue", "file": "sound/gone.wav" },
        ]));
        let findings = validate_reference_pack_files(&pack, &dir);
        assert_eq!(findings.len(), 1, "{}", lines(&findings));
        assert!(
            findings[0].message.contains("recorded"),
            "{}",
            findings[0].message
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_sound_entry_round_trips_its_synthesis_fields_and_omits_what_it_does_not_carry() {
        let entry: SoundEntry = serde_json::from_value(json!({
            "role": "line", "kind": "dialogue", "text": "  Delivery.  ", "voice": " am_michael "
        }))
        .expect("entry parses");
        assert!(entry.is_synthesized());
        assert_eq!(entry.synthesis_text(), Some("Delivery."));
        assert_eq!(entry.synthesis_voice(), Some("am_michael"));
        assert_eq!(entry.synthesis_model(), DEFAULT_SOUND_SYNTHESIS_MODEL);

        let recorded: SoundEntry = serde_json::from_value(json!({
            "role": "bed", "kind": "ambience", "file": "sound/bed.wav"
        }))
        .expect("entry parses");
        assert!(!recorded.is_synthesized());
        // A pack written before synthesis existed serializes back to exactly what it was: no
        // `text` / `voice` / `model` keys appear on an entry that carries none.
        let json = serde_json::to_value(&recorded).expect("serializes");
        assert_eq!(
            json,
            json!({ "role": "bed", "kind": "ambience", "file": "sound/bed.wav", "description": "" })
        );
    }

    // ---------------------------------------------------------------------------------------
    // Reference spec (sc-23403)
    // ---------------------------------------------------------------------------------------

    fn spec_json() -> Value {
        json!({
            "schemaVersion": 1,
            "id": "courier-workshop-refs",
            "version": 2,
            "model": { "id": "krea_2_turbo", "tier": "q8", "resolution": "1024x1024" },
            "limits": { "maxJobSeconds": 900, "maxAttemptsPerRole": 2, "maxMemoryGb": 96 },
            "seedBase": 4200,
            "inherit": { "pack": "references.jsonc", "references": ["house_style"], "sound": true },
            "references": [
                {
                    "role": "courier", "kind": "character", "file": "references/courier.png",
                    "description": "the courier", "prompt": "a courier in a blue jacket"
                },
                {
                    "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png",
                    "description": "the parcel", "prompt": "a small red parcel on a plain surface"
                }
            ]
        })
    }

    fn spec_from(value: Value) -> ReferenceSpec {
        serde_json::from_value(value).expect("spec parses")
    }

    #[test]
    fn a_well_formed_reference_spec_validates() {
        let spec = spec_from(spec_json());
        assert_eq!(validate_reference_spec(&spec), Vec::new());
        assert_eq!(spec.model.mode, "text_to_image");
        assert_eq!(
            spec.references[0].resolution_with(&spec),
            Some("1024x1024"),
            "a role with no resolution renders at the spec's"
        );
        assert_eq!(
            spec.references[0].negative_prompt_with(&spec),
            None,
            "no negative prompt is declared anywhere, so none is sent"
        );
    }

    #[test]
    fn a_reference_spec_role_without_a_prompt_is_refused_by_name() {
        let mut value = spec_json();
        value["references"][1]["prompt"] = json!("   ");
        let findings = validate_reference_spec(&spec_from(value));
        assert!(
            findings.iter().any(|finding| {
                finding.field == "referenceSpec.references[1].prompt"
                    && finding.message.contains("red_parcel")
            }),
            "{findings:#?}"
        );
    }

    #[test]
    fn a_reference_spec_refuses_unbounded_or_colliding_declarations() {
        let mut value = spec_json();
        value["limits"] = json!({ "maxJobSeconds": 0, "maxAttemptsPerRole": 9, "maxMemoryGb": 0 });
        value["references"][1]["role"] = json!("courier");
        value["references"][1]["file"] = json!("references/courier.png");
        value["inherit"]["references"] = json!(["courier"]);
        let findings = validate_reference_spec(&spec_from(value));
        for field in [
            "referenceSpec.limits.maxJobSeconds",
            "referenceSpec.limits.maxAttemptsPerRole",
            "referenceSpec.limits.maxMemoryGb",
            "referenceSpec.references[1].role",
            "referenceSpec.references[1].file",
            "referenceSpec.inherit.references[0]",
        ] {
            assert!(
                findings.iter().any(|finding| finding.field == field),
                "expected a finding on {field}: {findings:#?}"
            );
        }

        // maxJobSeconds has a CEILING as well as a floor: a spec is a bounded fixture run, and a
        // declaration near u64::MAX would make the generator's `Instant::now() + Duration` deadline
        // an overflow panic rather than a budget.
        for seconds in [MAX_REFERENCE_SPEC_JOB_SECONDS + 1, u64::MAX] {
            let mut value = spec_json();
            value["limits"]["maxJobSeconds"] = json!(seconds);
            let findings = validate_reference_spec(&spec_from(value));
            assert!(
                findings.iter().any(|finding| {
                    finding.field == "referenceSpec.limits.maxJobSeconds"
                        && finding
                            .message
                            .contains(&MAX_REFERENCE_SPEC_JOB_SECONDS.to_string())
                }),
                "maxJobSeconds {seconds} was admitted: {findings:#?}"
            );
        }
        // And the ceiling itself is admitted, so the range is a range and not an off-by-one.
        let mut value = spec_json();
        value["limits"]["maxJobSeconds"] = json!(MAX_REFERENCE_SPEC_JOB_SECONDS);
        let findings = validate_reference_spec(&spec_from(value));
        assert!(
            !findings
                .iter()
                .any(|finding| finding.field == "referenceSpec.limits.maxJobSeconds"),
            "the ceiling itself must be admitted: {findings:#?}"
        );
    }

    #[test]
    fn a_reference_spec_file_cannot_escape_the_pack_or_inject_a_multipart_header() {
        for file in [
            "../outside.png",
            "/tmp/outside.png",
            "references/co\r\nurier.png",
            "references/courier.txt",
        ] {
            let mut value = spec_json();
            value["references"][0]["file"] = json!(file);
            let findings = validate_reference_spec(&spec_from(value));
            assert!(
                findings
                    .iter()
                    .any(|finding| finding.field == "referenceSpec.references[0].file"),
                "{file} was admitted: {findings:#?}"
            );
        }
    }

    #[test]
    fn an_unknown_reference_spec_key_or_schema_version_is_refused() {
        let mut value = spec_json();
        value["references"][0]["steps"] = json!(8);
        assert!(
            serde_json::from_value::<ReferenceSpec>(value).is_err(),
            "an unknown key must not be silently dropped"
        );
        let mut value = spec_json();
        value["schemaVersion"] = json!(99);
        let findings = validate_reference_spec(&spec_from(value));
        assert!(
            findings
                .iter()
                .any(|finding| finding.field == "referenceSpec.schemaVersion"),
            "{findings:#?}"
        );
    }

    #[test]
    fn the_shipped_courier_reference_spec_validates() {
        let text = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/film-harness/courier-workshop/references.spec.jsonc"
        ));
        let spec = parse_reference_spec(text).expect("the shipped spec parses");
        assert_eq!(validate_reference_spec(&spec), Vec::new());
    }

    #[test]
    fn a_generated_reference_must_carry_its_provenance() {
        let mut value = pack_json();
        value["references"][0]["generated"] = json!(true);
        let pack: ReferencePack = serde_json::from_value(value).expect("pack parses");
        let findings = validate_reference_pack(&pack);
        assert!(
            findings.iter().any(|finding| {
                finding.field == "referencePack.references[0].generation"
                    && finding.message.contains("no generation provenance")
            }),
            "{findings:#?}"
        );

        let mut value = pack_json();
        value["references"][0]["generation"] = json!({
            "model": "krea_2_turbo", "mode": "text_to_image", "prompt": "a courier",
            "width": 1024, "height": 1024, "jobId": "job_1", "assetId": "asset_1",
            "sha256": "abc", "createdAt": "2026-09-14T00:00:00Z"
        });
        let pack: ReferencePack = serde_json::from_value(value).expect("pack parses");
        let findings = validate_reference_pack(&pack);
        assert!(
            findings
                .iter()
                .any(|finding| finding.field == "referencePack.references[0].generated"),
            "provenance without the flag is a document nothing wrote: {findings:#?}"
        );
    }

    #[test]
    fn a_generated_reference_with_full_provenance_validates() {
        let mut value = pack_json();
        value["references"][0]["generated"] = json!(true);
        value["references"][0]["generation"] = json!({
            "model": "krea_2_turbo", "tier": "q8", "backend": "mlx", "mode": "text_to_image",
            "prompt": "a courier", "seed": 4200, "width": 1024, "height": 1024,
            "jobId": "job_1", "assetId": "asset_1", "sha256": "abc",
            "createdAt": "2026-09-14T00:00:00Z"
        });
        let pack: ReferencePack = serde_json::from_value(value).expect("pack parses");
        assert_eq!(validate_reference_pack(&pack), Vec::new());
        assert!(pack.references[0].generated);
        assert_eq!(
            pack.references[0]
                .generation
                .as_ref()
                .map(|generation| generation.model.as_str()),
            Some("krea_2_turbo")
        );
    }
}
