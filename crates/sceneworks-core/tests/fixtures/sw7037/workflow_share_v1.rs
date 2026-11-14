//! The sanitized workflow share envelope (sc-15946, epic 15945).
//!
//! A generated image that leaves the app — pasted into Discord, attached to a bug report,
//! posted anywhere — takes its recipe with it only if the recipe is *inside the file*. The
//! local sidecar (`<media>.sceneworks.json` + `recipes/<id>.recipe.json`) dies at the first
//! copy-paste, which is the whole reason this envelope exists. This module owns the
//! contract; the PNG chunk writer (sc-15947) and the drag-to-load reader (sc-15952) build
//! on top of it. No file I/O and no PNG work happens here.
//!
//! Modeled on `apps/web/src/promptBatchIO.js` (sc-9954), which already solves this shape for
//! prompt batches: a versioned envelope under a marker key, authored content only, ids /
//! paths / timestamps stripped, and descriptive errors on parse rather than a silent default.
//!
//! # Two versions, deliberately not one
//!
//! * [`WorkflowShare::schema_version`] — the *contract* version. Bumped only when a field
//!   changes meaning or disappears. This is the only thing [`parse_workflow_share`] branches
//!   on.
//! * [`WorkflowProducer::version`] — the *build* that wrote the file, straight from
//!   `CARGO_PKG_VERSION` (the `[workspace.package] version` that `scripts/sync-version.mjs`
//!   keeps in lockstep with the web/desktop manifests). It never drives parsing. It exists so
//!   a bug report is actionable: "this came out of 0.8.1, which still had the old scheduler
//!   default".
//!
//! The producer version is the ONE field the allow-list below cannot protect, because it is a
//! field we deliberately include. So it must be the released version string and nothing else —
//! never `git describe`, never a dirty-tree suffix, never a CI build number, never anything
//! path- or host-derived. `0.8.1-dirty-Michael` would walk straight into every shared image.
//! [`PRODUCER_VERSION`] is `env!("CARGO_PKG_VERSION")` for exactly that reason, and a test
//! asserts it is strict `MAJOR.MINOR.PATCH`.
//!
//! # Allow-list, never a deny-list
//!
//! `advanced` (cloned verbatim into an asset's `rawAdapterSettings`) is untyped and grows
//! whenever someone adds a knob to one of the studio's job builders. A deny-list leaks every
//! future field by default, so [`ADVANCED_KEY_RULES`] classifies every key those builders can
//! emit and anything unclassified is dropped. `crates/sceneworks-core/tests/workflow_share.rs`
//! parses each builder named in `tests/common/workflow_registry.rs::ADVANCED_BUILDERS` and fails the build when a key is neither
//! allow-listed nor explicitly denied — a new knob can neither silently leak nor silently vanish.
//!
//! sc-15946 shipped that lint against ONE builder on the premise that it was the only one. It was
//! not: sc-15948 found a second (`buildDetailJobBody`, whose `cnScale` was being dropped) and then
//! two more (the character lane's `angleSet`, the interleave lane's `systemMessage` /
//! `imageGuidanceScale`). A lint that is bolted on per discovery closes no class, so
//! `tests/common/workflow_registry.rs::ADVANCED_BUILDERS` is a registry the lint ENUMERATES, and the same test walks
//! `apps/web/src` for anything that builds an `advanced` map and refuses to pass while one is in
//! neither the registry nor `DEFERRED_ADVANCED_BUILDERS` in that test registry.
//!
//! The line between in and out is *what to make* vs *what this machine can afford*:
//! sampler / steps / guidance / PiD describe the intended output and travel; quant tier,
//! flash-attention and the requested GPU describe this install's hardware budget and do not.
//! Ids, paths, project and machine names, and timestamps never travel. Note the asymmetry
//! with the producer block: build identity is in, *installation* identity is out.
//!
//! # Input images by shape, not by id
//!
//! A recipe that needs a source / reference / mask / control image records **that it needs
//! one, and of what kind** ([`WorkflowInput`]) — never a dangling local UUID. That is what
//! lets sc-15952 render a useful missing-inputs panel without leaking ids or bloating the
//! file with base64.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::contracts::{Asset, JsonObject};

/// Marker key the envelope is published under. Its presence is what makes a blob
/// identifiable as ours (the `promptBatchIO` marker-key pattern).
pub const WORKFLOW_SHARE_MARKER_KEY: &str = "sceneworksWorkflow";

/// Marker value for the image lane. The value names the workflow *kind* so the video half
/// (sc-15956) can share the marker key without a reader having to sniff the body.
pub const WORKFLOW_KIND_IMAGE: &str = "image";

/// Marker value for the video lane (sc-15956).
///
/// A NEW KIND rather than a `schemaVersion` bump, and that is the load-bearing decision. The two
/// gates in [`parse_workflow_share`] fail differently and only one of them is right here:
///
/// * a version bump would make an older build report *"this file was written by a newer version of
///   SceneWorks"* for a video — true, but it would say the same about an image, and the image
///   contract did not change;
/// * an unknown KIND makes an older build report exactly what is wrong — it does not understand
///   this kind of workflow — and it does so for videos ONLY, leaving every shared image reading as
///   it always did.
///
/// The kind gate deliberately runs BEFORE the version gate for that reason. sc-15954 established
/// the surrounding rule this follows: an older reader that would present a recipe it does not
/// really understand is worse than one that refuses, so the refusal is arranged to happen.
pub const WORKFLOW_KIND_VIDEO: &str = "video";

/// Normalize a full SHA-256 digest without accepting filenames, short hashes, or prefixed values.
/// Lowercase is canonical so byte-identical files serialize identically across producers.
fn normalize_sha256(value: &str) -> Option<String> {
    let value = value.trim();
    (value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| value.to_ascii_lowercase())
}

/// A malformed foreign attribution field must not make the rest of an otherwise useful recipe
/// unreadable. Deserialize through `Value` so wrong JSON types degrade to `None` as well.
fn deserialize_optional_sha256<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(value.as_str().and_then(normalize_sha256))
}

/// The workflow kinds this build understands, in the order a reader should think about them.
///
/// The parse gate is a membership test against this rather than a chain of `==`, so adding a third
/// lane is one entry and cannot half-land.
pub const WORKFLOW_KINDS: &[&str] = &[WORKFLOW_KIND_IMAGE, WORKFLOW_KIND_VIDEO];

/// The contract version. Bump ONLY on a breaking field change — an additive field does not
/// need it (an older reader drops what it does not know). [`parse_workflow_share`] branches
/// on this and on nothing else.
pub const WORKFLOW_SHARE_SCHEMA_VERSION: u32 = 1;

/// Producer name. Names the software, never the installation.
pub const PRODUCER_NAME: &str = "SceneWorks";

/// Canonical repository URL, so a file that reaches someone with no context is
/// self-identifying.
pub const PRODUCER_URL: &str = "https://github.com/SceneWorks/SceneWorks";

/// The released build that wrote the file — the workspace version, at compile time.
///
/// `sceneworks-core` inherits `version.workspace = true`, so this is the root `Cargo.toml`
/// `[workspace.package] version` verbatim. Deliberately NOT derived from git, the environment,
/// the build host or the build path; see the module docs.
pub const PRODUCER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifies the software that wrote the envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowProducer {
    pub name: String,
    pub url: String,
    /// Semver of the build that wrote the file. Never parsed against; see the module docs.
    pub version: String,
}

impl Default for WorkflowProducer {
    fn default() -> Self {
        Self {
            name: PRODUCER_NAME.to_owned(),
            url: PRODUCER_URL.to_owned(),
            version: PRODUCER_VERSION.to_owned(),
        }
    }
}

/// The kind of input image a recipe needs, recorded WITHOUT the local asset id that supplied
/// it. `count` is how many of that kind the original run used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowInput {
    /// One of [`INPUT_KIND_SOURCE`], [`INPUT_KIND_REFERENCE`], [`INPUT_KIND_MASK`],
    /// [`INPUT_KIND_CONTROL`].
    pub kind: String,
    pub count: u32,
    /// For [`INPUT_KIND_CONTROL`]: the conditioning the map feeds (`canny`, `depth`, …), when
    /// the original run named one. Never a path or an id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_mode: Option<String>,
}

/// The image an edit starts from (`sourceAssetId`).
pub const INPUT_KIND_SOURCE: &str = "source";
/// Identity / style reference image(s) (`referenceAssetId`, `referenceAssetIds`).
pub const INPUT_KIND_REFERENCE: &str = "reference";
/// Inpaint mask (`maskAssetId`).
pub const INPUT_KIND_MASK: &str = "mask";
/// A pre-made control map fed verbatim (`advanced.controlImage`).
pub const INPUT_KIND_CONTROL: &str = "control";
/// A source CLIP a video run continues, re-times or bridges from (`sourceClipAssetId`,
/// `sourceClipAssetIds`, `bridgeRightClipAssetId`) — sc-15956.
///
/// A separate kind from [`INPUT_KIND_SOURCE`] because the two are not interchangeable to a reader:
/// "this recipe needs a still to start from" and "this recipe needs a clip to continue" are
/// different missing-input panels and different asks of the person replaying it.
pub const INPUT_KIND_SOURCE_CLIP: &str = "sourceClip";
/// A reference CLIP a video run conditions on (`referenceClipAssetId`, Bernini's ads2v) —
/// sc-15956. The moving counterpart of [`INPUT_KIND_REFERENCE`].
pub const INPUT_KIND_REFERENCE_CLIP: &str = "referenceClip";
/// A reference AUDIO clip a video run conditions on (`referenceAudioAssetIds`, MiniMax-H3's
/// Ref2VA) — sc-17160. The audible counterpart of [`INPUT_KIND_REFERENCE`], and a separate kind
/// for the same reason [`INPUT_KIND_SOURCE_CLIP`] is: "this recipe needs a voice or a piece of
/// music" is a different ask of whoever replays it than "this recipe needs a picture".
pub const INPUT_KIND_REFERENCE_AUDIO: &str = "referenceAudio";

/// A LoRA the run applied, reduced to what another install can act on: the display name, the
/// weight, the Hugging Face repo id when the catalog entry resolved to one, and (only when the
/// worker proved it from the exact bytes used) the content digest galleries use for attribution.
/// No local id, no installed path, no source path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowLora {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// `owner/name` on Hugging Face, when the payload's catalog entry named one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// SHA-256 of the exact adapter file the inference route loaded. The request builder never
    /// reads this from client input; the worker overlays it after resolving and hashing the file.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_sha256"
    )]
    pub hash: Option<String>,
}

/// The upscale pass, when the run enabled one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowUpscale {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factor: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub softness: Option<f64>,
    /// Iris restoration (sc-25683): the applied output scale (any positive factor; 1 = restore only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    /// Iris restoration: `budgeted` or `original` input sizing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_sizing: Option<String>,
    /// Iris restoration: whether the wavelet colour correction ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_fix: Option<bool>,
}

/// The portable, sanitized workflow of one generated image.
///
/// Deliberately has NO `#[serde(flatten)] extra` bucket, unlike the sidecar contracts in
/// [`crate::contracts`]. Those preserve unknown keys so a newer writer's fields survive an
/// older reader; here that would defeat the whole point — an envelope arriving from outside
/// must be reduced to the fields we classify, on parse as well as on build. Unknown keys are
/// dropped in both directions.
///
/// Dropping unknown KEYS is only half of it: the strings under the keys we *do* declare are
/// still attacker-chosen in a file that came from a stranger. `reduce_workflow_share` runs the
/// same value-level guards on parse as on build — the path check on every label, `owner/name`
/// validation on `loras[].repo`, the closed [`INPUT_KINDS`] vocabulary on `inputs[].kind`, and
/// a bounded producer block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowShare {
    /// The marker key. Its value names the workflow kind ([`WORKFLOW_KIND_IMAGE`]).
    #[serde(rename = "sceneworksWorkflow")]
    pub kind: String,
    /// The contract version — the ONLY field the parser branches on.
    pub schema_version: u32,
    pub producer: WorkflowProducer,
    pub mode: String,
    /// The model catalog slug (`z_image_turbo`), never a resolved weights location.
    pub model: String,
    /// SHA-256 of the exact checkpoint bytes used for this image, when the worker can prove it.
    /// This is content identity for gallery attribution, never a local path or user-entered id.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_sha256"
    )]
    pub model_hash: Option<String>,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub negative_prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    /// **Video lane** (sc-15956): the clip length the user ASKED for, in seconds.
    ///
    /// The ask, not the measurement. `file.duration` on the sidecar is what the encoder actually
    /// produced — those differ whenever a model's temporal stride snaps the frame count, and a 6.0 s
    /// ask that rendered 5.96 s replays as 6.0 (sc-12371 established the same split for the
    /// sidecar). An envelope is a recipe: it must round-trip the ask.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// **Video lane**: the frame rate the user asked for. The ask, for the reason above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// **Video lane**: the quality preset the run was submitted at — `fast`, `balanced` or `best`,
    /// the closed vocabulary `qualityChoices` in `apps/web/src/jobTypes.js` offers (the studio
    /// LABELS them "Draft" / "Balanced" / "Final"; the value is what travels). Written as
    /// `draft` / `standard` here until the sc-15956 review measured it — neither was a value the
    /// app can emit.
    ///
    /// A named tier off a menu, not a hardware budget — the receiving install has the same menu.
    /// The field is still an unvalidated string on the parse side, because an envelope from a
    /// newer build may name a tier this one has never heard of and refusing the whole recipe over
    /// a menu label would be worse than reporting it unresolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style_preset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upscale: Option<WorkflowUpscale>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<WorkflowLora>,
    /// Input images by shape, never by id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<WorkflowInput>,
    /// The allow-listed subset of the request's `advanced` map. See [`ADVANCED_KEY_RULES`].
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub advanced: JsonObject,
    /// The collections this envelope declared but could not record whole. See [`OMITTED_FIELDS`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub omitted: Vec<String>,
}

// ---------------------------------------------------------------------------
// The omission marker
// ---------------------------------------------------------------------------

/// `loras`, dropped or thinned.
pub const OMITTED_LORAS: &str = "loras";
/// `inputs`, dropped or thinned.
pub const OMITTED_INPUTS: &str = "inputs";
/// `advanced.poses`, dropped.
pub const OMITTED_POSES: &str = "advanced.poses";
/// `advanced.phases`, dropped.
pub const OMITTED_PHASES: &str = "advanced.phases";
/// A phase's own `loras` schedule, dropped.
pub const OMITTED_PHASE_LORAS: &str = "advanced.phases[].loras";

/// Every name [`WorkflowShare::omitted`] may hold, and nothing else.
///
/// The marker exists because the drop-vs-truncate doctrine [`over_cap`] documents rested
/// on a premise that is empirically false. It argued a reader "cannot tell a plausible subset from
/// the real thing but can tell an absence" — and an absence is exactly what it cannot tell:
/// `loras` carries `skip_serializing_if = "Vec::is_empty"`, so an over-cap LoRA envelope whose LoRAs were
/// dropped over the cap and a genuinely LoRA-free one serialize BYTE-IDENTICALLY. Same for
/// `advanced.poses` and `advanced.phases`, which are simply absent keys. sc-15952 would render "no
/// LoRAs" and offer one-click "Use this recipe" for a recipe that had five.
///
/// Dropping is still right (truncating fabricates a specific membership, which is worse), so the
/// fix is to make the drop SELF-DESCRIBING rather than to stop dropping. With this field the reader
/// can say "this file declared more LoRAs than a job can have; they were not recorded" and withhold
/// the replay.
///
/// A closed vocabulary of FIELD NAMES rather than free text, deliberately: it adds no new attack
/// surface — the parse side keeps only members of this list, so the worst an envelope can put here
/// is a subset of five short constants — and it is bounded by construction rather than by another
/// cap that would have to compose with the rest.
///
/// Emitted in BOTH directions. On the build side it is the difference between a silently lost
/// 70-pose selection ([`MAX_SHARE_POSES`]) and a visible one, which is the only thing standing
/// between our own writer and the silent-loss failure this epic exists to prevent.
///
/// Scoped to collections, and to entries that had something shareable to say. A `loras` entry
/// carrying only a local `id` is not an omission — the id was never going to travel and the entry
/// declared nothing else — where an entry count over [`MAX_SHARE_LORAS`] is, because the recipe
/// named LoRAs and the envelope records none of them.
pub const OMITTED_FIELDS: &[&str] = &[
    OMITTED_LORAS,
    OMITTED_INPUTS,
    OMITTED_POSES,
    OMITTED_PHASES,
    OMITTED_PHASE_LORAS,
];

/// The marker, reduced to the closed vocabulary: unknown names dropped, duplicates collapsed,
/// sorted so the field is a SET rather than an order an envelope could carry information in.
///
/// Sanitized on parse like everything else. Sorting also makes it deterministic across the two
/// `serde_json` map-ordering configurations the workspace builds under.
fn reduce_omitted(names: Vec<String>) -> Vec<String> {
    let mut kept: Vec<String> = names
        .into_iter()
        .filter(|name| OMITTED_FIELDS.contains(&name.as_str()))
        .collect();
    kept.sort_unstable();
    kept.dedup();
    kept
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why an envelope could not be read. Every variant carries enough detail to show the user a
/// sentence, which is the point: a workflow that arrived from a stranger's PNG must never
/// surface as a raw serde message or, worse, silently default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowShareError {
    /// The text was not JSON at all.
    InvalidJson(String),
    /// The JSON parsed but is not an object (an array, a bare string, …).
    NotAnObject,
    /// No `sceneworksWorkflow` marker key — this blob was not written by us.
    MissingMarker,
    /// The marker is present but names a workflow kind this reader does not handle
    /// (the image reader handed a video envelope, sc-15956).
    UnsupportedKind { found: String, supported: String },
    /// `schemaVersion` is absent (or explicitly `null`). A key that IS present but is not a
    /// whole number is [`Self::Malformed`] instead — telling someone a field they can see in
    /// the file is missing sends them looking in the wrong place.
    MissingSchemaVersion,
    /// The file's contract version is newer than this build's. Names both versions so the
    /// user is told to update rather than shown a parse failure.
    UnsupportedSchemaVersion { file: u32, supported: u32 },
    /// A declared field is present but the wrong shape.
    Malformed { field: String, detail: String },
    /// The envelope is bigger than this reader records (see [`WORKFLOW_SHARE_MAX_BYTES`]).
    ///
    /// Measured on the SANITIZED envelope, after every per-field bound has already run — because
    /// per-field bounds are what did not compose. `bytes` is what it reduced to, not what arrived.
    TooLarge { bytes: usize, limit: usize },
}

impl fmt::Display for WorkflowShareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(detail) => {
                write!(f, "Not a valid SceneWorks workflow: invalid JSON ({detail}).")
            }
            Self::NotAnObject => write!(
                f,
                "Not a valid SceneWorks workflow: the workflow must be a JSON object."
            ),
            Self::MissingMarker => write!(
                f,
                "This file is not a SceneWorks workflow (no `{WORKFLOW_SHARE_MARKER_KEY}` marker)."
            ),
            Self::UnsupportedKind { found, supported } => write!(
                f,
                "This is a `{found}` SceneWorks workflow; this build reads: {supported}. Update {PRODUCER_NAME} to load it."
            ),
            Self::MissingSchemaVersion => write!(
                f,
                "This SceneWorks workflow is missing its `schemaVersion`, so it cannot be read safely."
            ),
            Self::UnsupportedSchemaVersion { file, supported } => write!(
                f,
                "This workflow was written with schema version {file}; this build of {PRODUCER_NAME} reads version {supported}. Update {PRODUCER_NAME} to load it."
            ),
            Self::Malformed { field, detail } => write!(
                f,
                "This SceneWorks workflow is malformed: `{field}` {detail}."
            ),
            Self::TooLarge { bytes, limit } => write!(
                f,
                "This SceneWorks workflow is {bytes} bytes of settings, over the {limit} bytes {PRODUCER_NAME} records, so no recipe was read from it."
            ),
        }
    }
}

impl std::error::Error for WorkflowShareError {}

// ---------------------------------------------------------------------------
// The `advanced` allow-list
// ---------------------------------------------------------------------------

/// Whether a key travels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedDisposition {
    /// Authored generation intent — travels.
    Allow,
    /// Install-specific, identifying, or otherwise not shareable — dropped, on purpose.
    Deny,
}

/// Which builder a rule is held accountable to, so the coverage lint knows what JS to read.
///
/// Every variant except [`Self::Server`] names exactly one entry in `tests/common/workflow_registry.rs::ADVANCED_BUILDERS`, and
/// the lint asserts that correspondence in both directions — a variant with no registry entry, or
/// a registry entry with no variant, fails. That is what makes adding a builder a decision rather
/// than an omission.
///
/// A key MORE THAN ONE builder emits is tagged with its primary builder (the largest,
/// fastest-moving surface that emits it). The "is every emitted key classified?" half of the lint
/// runs for every registered builder regardless of tags; only the "is every rule still emitted?"
/// half is per-tag, and a key needs just one builder holding it honest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedKeySource {
    /// `buildImageJobAdvanced` — the Image Studio's ~30-knob builder.
    StudioBuilder,
    /// The shared MOSS edit controls, also used by the eager Library/editor request builder.
    MossEditBuilder,
    /// `imageRecipeRequest` — replays persisted adapter settings without authoring new controls.
    ImageRecipeReplayBuilder,
    /// `buildEditJobBody` — the Image Editor's prompt-edit body (sc-15948 follow-up: it emits
    /// `advanced.guidanceScale` and feeds an embedding lane, so it is registered even though
    /// every key it emits was already classified).
    EditBuilder,
    /// `buildDetailJobBody` — the standalone `image_detail` pass's own two-knob builder
    /// (sc-15948). Before it was registered, `cnScale` was unclassified and silently dropped.
    DetailBuilder,
    /// `buildAdvanced` in `CharacterAdvancedOptions.jsx` — the character lane's shared tuning
    /// block, which merges one of the `advancedExtras` maps below into its own knobs.
    CharacterBuilder,
    /// `useAngleController`'s `advancedExtras` — the Angle Set form (sc-15948 follow-up).
    /// `angleSet` is what makes the worker emit one image per view angle, so dropping it made a
    /// shared angle-set image replay as a single image.
    CharacterAngleExtras,
    /// `usePoseController`'s `advancedExtras` — the Pose Library form.
    CharacterPoseExtras,
    /// `DocumentStudio.submit` — the interleaved-document lane, which sc-15948 turned INTO an
    /// embedding lane (`sensenova_jobs.rs` threads `workflow_source` into `ImagePlan`).
    InterleaveBuilder,
    /// `buildUpscaleJobBody` — the standalone `image_upscale` job. It emits NO `advanced` map, and
    /// is registered precisely so that stays true: it feeds an embedding lane (sc-15948 ISSUE 1),
    /// so the day it grows a knob the lint demands a classification. No rule is tagged with it.
    UpscaleBuilder,
    /// `VideoStudio.submit` — the Video Studio's own ~20-knob builder, and the video lane's
    /// equivalent of [`Self::StudioBuilder`] (sc-15956).
    VideoStudioBuilder,
    /// `useEditorGeneration.buildBasePayload` — the timeline editor's shared video payload, spread
    /// into all three [`Self::TimelineExtendBuilder`]-family actions (sc-15956).
    TimelineBaseBuilder,
    /// `EditorScreen.extendSelectedClip` — the timeline "extend this clip" action. It and its two
    /// siblings each spread `buildBasePayload`'s map and add `timelineAction` / `timelineContext`;
    /// those two keys are tagged to this one, because a key needs just one builder holding it
    /// honest and all three emit the same pair.
    TimelineExtendBuilder,
    /// `EditorScreen.replaceSelectedItem` — the timeline "replace this clip" action.
    TimelineReplaceBuilder,
    /// `EditorScreen.bridgeGap` — the timeline "bridge this gap" action.
    TimelineBridgeBuilder,
    /// `simpleJobs.buildSimpleVideoRequest` — the Simple shell's video request (sc-15956). Its
    /// image sibling delegates to `buildImageJobRequest` and so rides [`Self::StudioBuilder`];
    /// this one builds its own two-key map.
    SimpleVideoBuilder,
    /// `VideoUpscalePanel.onUpscale` — the standalone `video_upscale` job's payload (sc-15956
    /// review). It emits NO `advanced` map, and is registered for exactly the reason
    /// [`Self::UpscaleBuilder`] is: its lane embeds, so the day it grows a knob the lint demands a
    /// classification. No rule is tagged with it.
    VideoUpscaleBuilder,
    /// Stamped onto `advanced` by the API after the request arrives (recipe-preset resolution
    /// in `apps/rust-api/src/generation.rs`). Classified here for the record; the lint does
    /// not expect to find them in the JS, and no `tests/common/workflow_registry.rs::ADVANCED_BUILDERS` entry claims it.
    Server,
}

impl AdvancedKeySource {
    /// Every variant, so the lint can assert the registry covers them all. A new variant that
    /// nobody registered fails `tests/common/workflow_registry.rs::ADVANCED_BUILDERS`'s round-trip test.
    pub const ALL: &'static [Self] = &[
        Self::StudioBuilder,
        Self::MossEditBuilder,
        Self::ImageRecipeReplayBuilder,
        Self::EditBuilder,
        Self::DetailBuilder,
        Self::CharacterBuilder,
        Self::CharacterAngleExtras,
        Self::CharacterPoseExtras,
        Self::InterleaveBuilder,
        Self::UpscaleBuilder,
        Self::VideoStudioBuilder,
        Self::TimelineBaseBuilder,
        Self::TimelineExtendBuilder,
        Self::TimelineReplaceBuilder,
        Self::TimelineBridgeBuilder,
        Self::SimpleVideoBuilder,
        Self::VideoUpscaleBuilder,
        Self::Server,
    ];
}

/// How an allowed value is reduced. Anything that does not match its shape is dropped rather
/// than passed through — that is what stops an object (and any path inside it) from being
/// smuggled under an allow-listed scalar key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedShape {
    /// Public catalog recipe IDs, without arbitrary nested metadata.
    Appearance,
    /// String, number or bool. Path-shaped strings are dropped (see [`is_path_shaped`]).
    Scalar,
    /// The structured-prompt recipe (`apps/web/src/ideogramCaption.js`), reduced to its scalar
    /// fields. The free-form `caption` object is dropped — its serialized form already rides
    /// in `runtimePrompt` and in the top-level `prompt`.
    StructuredPrompt,
    /// The pose-library selection, reduced to its numeric coordinate arrays ([`POSE_FIELDS`]:
    /// `keypoints`, `hands`, `face` — all three drive the rendered skeleton). Pose-library ids,
    /// and anything else the picker attached, do not travel.
    Poses,
    /// MiniMax-H3 Veda request controls, kept as the exact five-field wire object after the shared
    /// H3 validator refuses malformed values or unknown fields.
    H3VedaSettings,
    /// Canonical MOSS request options, validated by gen-core.
    MossImageOptions,
    /// Provider receipt with local adapter paths reduced to their frozen relative filename.
    MossImageReport,
    MossImageEditOptions,
    MossImageEditReport,
    MossImageAdvanced,
    PromptBatch,
    SampleIndex,
    /// The Krea multi-phase denoise list, reduced to `{ steps, guidance, loras:[{index, weight}] }`.
    Phases,
    /// A `[start, end]` pair of finite numbers (the CFG interval, sc-25681). Anything else —
    /// another length, a string, a nested value — is dropped, like an object under a scalar key.
    NumberPair,
    /// FLUX.3 authored box document, reduced to public coordinate/identity/description fields.
    Flux3Layout,
    /// Ordered attachment-instance IDs (not paths or asset locators), at most ten unique values.
    StringList,
}

/// One classified `advanced` key.
#[derive(Debug, Clone, Copy)]
pub struct AdvancedKeyRule {
    pub key: &'static str,
    pub disposition: AdvancedDisposition,
    pub shape: AdvancedShape,
    pub source: AdvancedKeySource,
    /// Why. Read by a human deciding where a NEW knob belongs.
    pub reason: &'static str,
}

const fn allow(key: &'static str, shape: AdvancedShape, reason: &'static str) -> AdvancedKeyRule {
    AdvancedKeyRule {
        key,
        disposition: AdvancedDisposition::Allow,
        shape,
        source: AdvancedKeySource::StudioBuilder,
        reason,
    }
}

const fn deny(key: &'static str, reason: &'static str) -> AdvancedKeyRule {
    AdvancedKeyRule {
        key,
        disposition: AdvancedDisposition::Deny,
        shape: AdvancedShape::Scalar,
        source: AdvancedKeySource::StudioBuilder,
        reason,
    }
}

/// `allow`, tagged to a builder other than the studio's.
const fn allow_from(
    key: &'static str,
    shape: AdvancedShape,
    source: AdvancedKeySource,
    reason: &'static str,
) -> AdvancedKeyRule {
    AdvancedKeyRule {
        key,
        disposition: AdvancedDisposition::Allow,
        shape,
        source,
        reason,
    }
}

/// `deny`, tagged to a builder other than the studio's.
const fn deny_from(
    key: &'static str,
    source: AdvancedKeySource,
    reason: &'static str,
) -> AdvancedKeyRule {
    AdvancedKeyRule {
        key,
        disposition: AdvancedDisposition::Deny,
        shape: AdvancedShape::Scalar,
        source,
        reason,
    }
}

const fn deny_server(key: &'static str, reason: &'static str) -> AdvancedKeyRule {
    AdvancedKeyRule {
        key,
        disposition: AdvancedDisposition::Deny,
        shape: AdvancedShape::Scalar,
        source: AdvancedKeySource::Server,
        reason,
    }
}

/// Every `advanced` key, classified.
///
/// The load-bearing table. `crates/sceneworks-core/tests/workflow_share.rs` parses every builder
/// in `tests/common/workflow_registry.rs::ADVANCED_BUILDERS` and fails when a key one of them can emit is missing here — so a new
/// knob is a build-time decision, not a silent leak and not a silent loss.
///
/// The rule of thumb when classifying a new key: does it describe **what to make** (travels)
/// or **what this machine can afford to make it with** (does not)?
pub const ADVANCED_KEY_RULES: &[AdvancedKeyRule] = &[
    allow("flux3_layout", AdvancedShape::Flux3Layout, "Authored FLUX.3 soft-guidance boxes; public row fields only, never worker provenance."),
    allow("flux3_reference_ids", AdvancedShape::StringList, "Ordered attachment-instance identities for FLUX.3 layout replay; not asset paths."),
    allow("mossImageEdit", AdvancedShape::MossImageEditOptions, "Authored native edit reference edge and cache policy, emitted and restored by Studio."),
    allow_from("mossImageEditExecutionReport", AdvancedShape::MossImageEditReport, AdvancedKeySource::Server, "Actual edit adapter and ordered reference identity; imported receipts are scrubbed before execution."),
    allow_from("mossImageAdvanced", AdvancedShape::MossImageAdvanced, AdvancedKeySource::MossEditBuilder, "Edit-only authored tensor identities/shape, negative batch and output kind. Local asset IDs are removed; exact artifacts must be reacquired."),
    allow_from("promptBatch", AdvancedShape::PromptBatch, AdvancedKeySource::MossEditBuilder, "The full bounded native edit text batch, retained with exact tensor row bindings for replay."),
    allow_from("mossImageSampleIndex", AdvancedShape::SampleIndex, AdvancedKeySource::Server, "Output ordinal within the original native edit batch; display provenance only, scrubbed before a new execution."),
    allow("mossImage", AdvancedShape::MossImageOptions, "Requested MOSS mode and author controls; canonical native wire object."),
    deny("mossOrdinarySettings", "Inactive ordinary UI choices are local restore metadata, not the accelerated image's requested generation settings."),
    allow_from("mossImageExecutionReport", AdvancedShape::MossImageReport, AdvancedKeySource::Server, "Actual completed provider receipt; local adapter location is removed before sharing. Imported reports are display only and scrubbed before a new execution."),
    allow(
        "resolution",
        AdvancedShape::Scalar,
        "The authored output geometry label (\"1024x1024\") the studio control was set to.",
    ),
    allow(
        "structuredPrompt",
        AdvancedShape::StructuredPrompt,
        "Authored prompt content for structured-caption models; reduced to its scalar fields.",
    ),
    allow("sampler", AdvancedShape::Scalar, "Authored sampler choice."),
    allow(
        "scheduler",
        AdvancedShape::Scalar,
        "Authored scheduler choice.",
    ),
    allow(
        "schedulerShift",
        AdvancedShape::Scalar,
        "Authored time-shift (mu) for the curated schedule.",
    ),
    allow(
        "cfgInterval",
        AdvancedShape::NumberPair,
        "Authored classifier-free-guidance interval [start, end] in model time (sc-25681, \
         Iris-3B): which denoise steps guidance runs on — changes the picture, names nothing local.",
    ),
    allow(
        "captionOverflow",
        AdvancedShape::Scalar,
        "Authored caption-overflow policy (warn / error / silent, sc-25681): whether an over-long \
         prompt is truncated or refused — authored intent about the prompt, not a hardware budget.",
    ),
    allow(
        "steps",
        AdvancedShape::Scalar,
        "Authored step-count override.",
    ),
    allow(
        "guidanceScale",
        AdvancedShape::Scalar,
        "Authored guidance override.",
    ),
    allow(
        "guidanceMethod",
        AdvancedShape::Scalar,
        "Authored guidance method (CFG / CFG++).",
    ),
    allow(
        "enhancePrompt",
        AdvancedShape::Scalar,
        "Authored caption-upsampling opt-in — changes the prompt the model sees.",
    ),
    allow(
        "transparentBackground",
        AdvancedShape::Scalar,
        "Authored transparency (sc-24113): the model decodes four channels and the render carries a \
         real alpha channel instead of being composited onto white. WHAT TO MAKE, not what this \
         machine can afford — a cut-out is the picture the author asked for, and replaying their \
         recipe onto an opaque render would silently produce a different image. It is not a memory \
         accommodation and names nothing local (no id, path or preset). Emitted only for a model \
         that advertises `supportsAlphaOutput`; a replay onto a model that does not is refused by \
         name at the worker rather than downgraded.",
    ),
    allow(
        "usePid",
        AdvancedShape::Scalar,
        "Authored decoder choice: PiD changes the produced image, and is the output's \
         non-commercial marker. Unlike a quant tier it is not a memory accommodation.",
    ),
    allow(
        "decoder",
        AdvancedShape::Scalar,
        "Authored alternate terminal decoder id. Native is omitted; replay must preserve an explicit experimental decoder choice.",
    ),
    allow(
        "pidTarget",
        AdvancedShape::Scalar,
        "Authored PiD output tier (2k / 4k) — output geometry, not a hardware budget.",
    ),
    allow(
        "ipAdapterScale",
        AdvancedShape::Scalar,
        "Authored reference strength.",
    ),
    allow(
        "controlnetConditioningScale",
        AdvancedShape::Scalar,
        "Authored identity-structure strength (InstantID).",
    ),
    allow(
        "trueCfgScale",
        AdvancedShape::Scalar,
        "Authored variation strength.",
    ),
    allow(
        "strength",
        AdvancedShape::Scalar,
        "Authored img2img strength.",
    ),
    allow(
        "textStyleGain",
        AdvancedShape::Scalar,
        "Authored Krea text-style tap-reweight gain.",
    ),
    allow(
        "viewAngle",
        AdvancedShape::Scalar,
        "Authored head-angle label.",
    ),
    allow(
        "poses",
        AdvancedShape::Poses,
        "Authored pose selection. The numeric coordinate arrays the worker renders travel \
         (keypoints, hands, face); the pose-library ids that named them do not.",
    ),
    allow(
        "faceRestore",
        AdvancedShape::Scalar,
        "Authored face-restoration opt-in.",
    ),
    allow(
        "controlMode",
        AdvancedShape::Scalar,
        "Authored control type (canny / depth / …). The control IMAGE rides as an input shape.",
    ),
    allow(
        "controlScale",
        AdvancedShape::Scalar,
        "Authored control-lock strength.",
    ),
    allow("appearance", AdvancedShape::Appearance, "Selected public Camera/Film/Look/Lighting catalog IDs."),
    allow("appearancePromptBatch", AdvancedShape::PromptBatch, "Raw pre-appearance native batch, distinct from the worker-owned effective promptBatch."),
    allow("appearancePrompt", AdvancedShape::Scalar, "Raw pre-appearance prose, for composition exactly once on replay."),
    allow("appearanceNegativePrompt", AdvancedShape::Scalar, "Authored negative text before recipe contributions."),
    allow(
        "styleId",
        AdvancedShape::Scalar,
        "The catalog style the user picked.",
    ),
    allow(
        "stylePrompt",
        AdvancedShape::Scalar,
        "The raw pre-style prompt, so a reader sees what was actually typed.",
    ),
    allow_from(
        "cnScale",
        AdvancedShape::Scalar,
        AdvancedKeySource::DetailBuilder,
        "Authored tile-ControlNet strength for the Detail pass — one of the two knobs that \
         builder exposes, and the sibling of the allow-listed `strength`. Without it a shared \
         detail-refined image describes half of what produced it (sc-15948).",
    ),
    allow_from(
        "angleSet",
        AdvancedShape::Scalar,
        AdvancedKeySource::CharacterAngleExtras,
        "Authored turnaround request: it makes the worker emit one image per view angle \
         regardless of `count`, so it decides WHAT IS MADE. Dropping it made a shared angle-set \
         image replay as a single image (sc-15948).",
    ),
    allow_from(
        "systemMessage",
        AdvancedShape::Scalar,
        AdvancedKeySource::InterleaveBuilder,
        "The interleave system prompt the user typed, and the exact text the model saw — the same \
         class as `prompt` and `stylePrompt`, so it is PROSE (see `PROSE_KEYS`): bounded and \
         control-stripped, but never dropped for naming a directory. It is only sent when edited \
         away from the worker's own default, so it is authored content by construction.",
    ),
    allow_from(
        "imageGuidanceScale",
        AdvancedShape::Scalar,
        AdvancedKeySource::InterleaveBuilder,
        "Authored reference-guidance strength for an interleaved document (engine \
         `img_cfg_scale`) — the same authored-strength class as `ipAdapterScale` and \
         `controlScale`, and only emitted when the run actually grounds on reference frames.",
    ),
    deny_from(
        "keypointCollectionId",
        AdvancedKeySource::CharacterAngleExtras,
        "A local Key Point Library collection id. It resolves to nothing on another install — the \
         same class as `recipePresetId` and `controlImage` (sc-15948).",
    ),
    allow(
        "phases",
        AdvancedShape::Phases,
        "Authored multi-phase denoise schedule; LoRA references are indices into this \
         request's own list, not ids.",
    ),
    deny(
        "flashAttn",
        "A backend kernel toggle — describes what this install's attention path can do, \
         not what to make.",
    ),
    deny(
        "mlxQuantize",
        "Quant tier: a memory accommodation for THIS machine. The receiving install picks \
         its own.",
    ),
    deny(
        "mlxQuantizeExplicit",
        "Marks a deliberate tier pick on this install; meaningless and misleading elsewhere.",
    ),
    deny(
        "convRot",
        "The INT8-ConvRot tier selector — the same hardware-budget class as `mlxQuantize`.",
    ),
    deny(
        "quantTier",
        "Named out explicitly by sc-15946: install-specific and fingerprinting.",
    ),
    deny(
        "imageReference",
        "Diagnostic context for a local image reference: project/asset ids, display name and \
         media path. Reference requirements travel in inputs, not this machine's labels or paths.",
    ),
    deny(
        "controlImage",
        "A local asset id. The fact that a control image is needed rides in `inputs` instead.",
    ),
    deny(
        "controlWeights",
        "A trained-overlay id plus the resolved weights PATH the API stamps onto it.",
    ),
    deny_server(
        "recipePresetId",
        "A local preset id stamped by the API; it resolves to nothing on another install.",
    ),
    deny_server(
        "presetMissingLoras",
        "Local LoRA ids the API could not resolve on THIS install.",
    ),
    // -----------------------------------------------------------------------
    // The VIDEO arm (sc-15956)
    // -----------------------------------------------------------------------
    //
    // Same rule of thumb, same table. What is new is a third question the image lane never had to
    // ask: a video knob can be neither generation intent nor a hardware budget, but a disclosure
    // about a PERSON. `selectedPersonTrack` and `replacementModeLabel` are that, and they are
    // denied for a reason the other denials do not carry.
    allow_from(
        "motion",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored camera-motion preset (`static`, `slow push-in`, `handheld`) off a closed menu. \
         It conditions the generation, so it decides what is made.",
    ),
    allow_from(
        "ltxPipeline",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX pipeline selector. It picks which denoise path runs, so two values produce \
         two different clips from one prompt — output, not budget.",
    ),
    allow_from(
        "distilledVariant",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX distilled-checkpoint variant. A different checkpoint is a different model \
         for replay purposes — the same class as `usePid`, which is also a decoder choice that \
         changes the output rather than a memory accommodation.",
    ),
    allow_from(
        "transformerVariant",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX-2.5 transformer choice (`distilled` or `dev`). It changes the checkpoint, \
         schedule, guidance, and refinement recipe, so dropping it changes the rendered clip.",
    ),
    allow_from(
        "vaeDecoder",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX-2.5 Conv-versus-DiffVAE decoder choice. The selected decoder changes the \
         actual reconstruction path and its rendered detail.",
    ),
    allow_from(
        "autoDuration",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX-2.5 duration-head opt-in. It decides whether the prompt or an explicit \
         duration supplies the frame count, so it is generation intent rather than budget.",
    ),
    allow_from(
        "autoDurationMinSeconds",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored lower bound for LTX-2.5's duration prediction. It constrains the resulting clip \
         length and must travel with the opt-in.",
    ),
    allow_from(
        "autoDurationMaxSeconds",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored upper bound for LTX-2.5's duration prediction. It constrains the resulting clip \
         length and must travel with the opt-in.",
    ),
    allow_from(
        "temporalUpsampleRounds",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX-2.5 temporal-refinement count. Each round runs a real x2 DFR refinement and \
         changes both the output frame count and motion, so it is replay-critical intent.",
    ),
    allow_from(
        "textEncoderModel",
        AdvancedShape::Scalar,
        AdvancedKeySource::StudioBuilder,
        "Authored text-encoder pick. It changes what the model SEES of the prompt, so a replay \
         without it is a different run. Only the opaque selection id travels; the server-owned \
         source path does not. A missing choice stays authored and fails closed on replay instead \
         of silently substituting the bundled encoder.",
    ),
    allow_from(
        "lightning",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored Wan2.2 A14B fast-4-step toggle. It swaps in a distilled recipe and overrides the \
         step count, so it visibly changes the clip; it is a speed/quality choice the user made, \
         not a tier this machine could afford.",
    ),
    allow_from(
        "videoCfgGuidanceScale",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX native CFG scale — the video lane's `guidanceScale`.",
    ),
    allow_from(
        "videoStgGuidanceScale",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX spatiotemporal-guidance scale.",
    ),
    allow_from(
        "videoRescaleScale",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored LTX guidance-rescale factor.",
    ),
    allow_from(
        "videoConditioningStrength",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored source-clip conditioning strength (extend, bridge, Krea Realtime v2v) — the \
         video lane's `strength`, and the same authored-strength class as `ipAdapterScale`.",
    ),
    allow_from(
        "bridgeRightVideoConditioningStrength",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored right-clip conditioning strength for a bridge — the sibling of \
         `videoConditioningStrength`, and dropping it would make a shared bridge replay lopsided.",
    ),
    allow_from(
        "veda",
        AdvancedShape::H3VedaSettings,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored MiniMax-H3 Veda sparsity and full-attention controls. The complete closed wire \
         object changes the denoising recipe and must travel together; malformed or unknown \
         members are refused rather than silently producing a different recipe.",
    ),
    allow_from(
        "timelineAction",
        AdvancedShape::Scalar,
        AdvancedKeySource::TimelineExtendBuilder,
        "Which timeline operation produced the clip (`extend` / `replace` / `bridge`) — a closed \
         three-value vocabulary that names the generation MODE, with no id in it. Its companion \
         `timelineContext` is denied, so this travels alone: \"this was an extend\" is true and \
         useful on any install, where \"an extend of item 4f2c… on timeline 91ab…\" is neither.",
    ),
    deny_from(
        "durationHint",
        AdvancedKeySource::VideoStudioBuilder,
        "Model-catalog prose the studio echoes back into the payload (\"Recommended: 5s or \
         less.\"), not a knob anybody set. It is not intent and it is not a value the run used — \
         the receiving install renders its own from its own catalog, and a stale hint from someone \
         else's build would be worse than none.",
    ),
    deny_from(
        "precision",
        AdvancedKeySource::VideoStudioBuilder,
        "LTX weight precision (`fp8` / `bf16`): what THIS machine can afford to hold the weights \
         in. The same hardware-budget class as `mlxQuantize`, and the receiving install picks its \
         own.",
    ),
    deny_from(
        "quantization",
        AdvancedKeySource::VideoStudioBuilder,
        "The torch lane's quant tier — `mlxQuantize` for a different backend, and denied for the \
         identical reason.",
    ),
    deny_from(
        "selectedPersonTrack",
        AdvancedKeySource::VideoStudioBuilder,
        "The WHOLE person-track record, not an id: a user-typed `name` that is routinely a real \
         person's name, a `sourceDisplayName` that is the original imported FILENAME, install-local \
         asset ids, and a `frames[].mask` array of filesystem PATHS. It is simultaneously every \
         class this allow-list excludes, and it is not gated on mode — a stale selection rides a \
         plain text_to_video payload. Denied on privacy first and on shape second (sc-15956).",
    ),
    deny_from(
        "replacementModeLabel",
        AdvancedKeySource::VideoStudioBuilder,
        "The display label for a person-replacement mode (\"Face Only\", \"Full Person, Keep \
         Outfit\"). Not an id, not a path, and denied anyway: a shared video must not disclose \
         that it was made by replacing a specific person, still less which mode of replacement. \
         Its top-level twin `replacementMode` is withheld by the builder for the same reason — see \
         `build_video_workflow_share_from`. There is no replay value to weigh against it, because \
         a recipient has no access to the track this describes.",
    ),
    deny_from(
        "timelineContext",
        AdvancedKeySource::TimelineExtendBuilder,
        "Where in the LOCAL timeline to write the result: `timelineId`, `itemId`, `trackId`, \
         `sourceAssetId` and friends are install-local UUIDs, and `timelineName` is the user's own \
         typed name for a project structure that is nobody else's business. The same class as \
         `recipePresetId`, with a free-text field on top. `timelineAction` carries the part that \
         travels.",
    ),
];

/// The rule for `key`, if it is classified.
#[must_use]
pub fn advanced_key_rule(key: &str) -> Option<&'static AdvancedKeyRule> {
    ADVANCED_KEY_RULES.iter().find(|rule| rule.key == key)
}

// ---------------------------------------------------------------------------
// Path shapes
// ---------------------------------------------------------------------------

/// True when `value` looks like a filesystem location — absolute OR relative.
///
/// Belt to the allow-list's braces: no allow-listed `advanced` key and no classified top-level
/// label legitimately holds a path, so any value that looks like one is dropped even from a key
/// that is otherwise in. The story's line is "every filesystem path without exception", so this
/// deliberately errs toward dropping: a false positive costs one label, a false negative ships
/// the user's home directory — and therefore their name — inside every copy of the image.
///
/// Recognized:
///
/// * a backslash **anywhere** — every Windows location has one, absolute (`C:\Users\…`),
///   relative (`models\weights\x`), traversing (`..\..\Users\…`), UNC (`\\server\share`),
///   drive-relative (`c:secret\file`) or environment-expanded (`%USERPROFILE%\x`);
/// * a POSIX absolute (`/etc/passwd`) and `~` / `~user` home expansions;
/// * a bare drive prefix `X:` at a token boundary, with or without a following separator, so
///   `C:foo` is caught and the URL scheme in `https://…` is not;
/// * a `..` traversal segment and a leading `./`;
/// * a multi-segment relative POSIX path (`assets/images/x.png`) — three or more segments and
///   no URL scheme. Two segments is a Hugging Face repo id (`acme/mira`) and stays, and a slash
///   written with a space beside it is a human list separator, not a path separator (see
///   [`relative_tree_segments`]);
/// * `file://`, including percent-encoded (`file%3A%2F%2FD%3A%2Fx`).
///
/// Deliberately NOT applied to the six authored prose fields (`prompt`, `negativePrompt`,
/// `stylePrompt`, `systemMessage`, and the structured prompt's `intent` / `runtimePrompt` — the
/// four that live under an `advanced` key are the `PROSE_KEYS` constant below): those are what the
/// user typed, and silently mangling a prompt because it mentions a directory would be worse than
/// the leak it prevents — the user authored it and can see it. `systemMessage` is in that set, so
/// an interleave system prompt naming a directory travels like any other authored text. That
/// exemption is pinned by `authored_prose_travels_verbatim_even_when_it_names_a_path` in the
/// integration tests, and against the document by
/// `the_doc_lists_exactly_the_path_exempt_prose_fields`.
#[must_use]
pub fn is_path_shaped(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return false;
    }
    if looks_like_a_location(trimmed) {
        return true;
    }
    // Percent-encoding is not a disguise: `file%3A%2F%2FD%3A%2Fx` is `file://D:/x`. Decoded
    // repeatedly, bounded, so a double-encoded `%252F` cannot hide one round deeper.
    let mut decoded = trimmed.to_owned();
    for _ in 0..PERCENT_DECODE_ROUNDS {
        let next = percent_decode_separators(&decoded);
        if next == decoded {
            break;
        }
        decoded = next;
        if looks_like_a_location(decoded.trim()) {
            return true;
        }
    }
    false
}

/// How many times [`is_path_shaped`] re-decodes percent escapes before giving up. Two is enough
/// for `%252F`; the bound is what keeps a pathological value from spinning.
const PERCENT_DECODE_ROUNDS: usize = 3;

/// The path-shape rules themselves, run once on the raw value and once on its
/// separator-decoded form (see [`is_path_shaped`]).
fn looks_like_a_location(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    if lower.contains("file://") {
        return true;
    }
    // A backslash anywhere. Nothing this guard protects legitimately contains one.
    if value.contains('\\') {
        return true;
    }
    if value.starts_with('/') {
        return true;
    }
    // `~/models/x` and `~michael/x` alike.
    if value.starts_with('~') && value.contains('/') {
        return true;
    }
    // A drive prefix at a token boundary. A SINGLE letter before the colon is what separates a
    // drive (`c:`) from a URL scheme (`https:`), and no trailing separator is required — the
    // drive-relative `C:foo` resolves against the process's per-drive cwd and is still a path.
    let is_drive_token = |token: &str| {
        let bytes = token.as_bytes();
        bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
    };
    if lower
        .split(|character: char| !character.is_ascii_alphanumeric() && character != ':')
        .any(is_drive_token)
    {
        return true;
    }
    let segments: Vec<&str> = value.split('/').collect();
    if segments.contains(&"..") {
        return true;
    }
    if segments.len() > 1 && segments[0] == "." {
        return true;
    }
    // A relative POSIX tree. Two segments is an HF repo id and stays; a URL is not a location on
    // THIS machine, so `https://host/a/b/c` is left alone.
    !lower.contains("://") && relative_tree_segments(value) >= 3
}

/// How many PATH segments `value` has, counting only slashes that could be path separators.
///
/// A filesystem separator is never written with a space beside it; a human list separator almost
/// always is. `"PiD 1.5 Decoder (FLUX.1 / Boogu / Chroma / Z-Image)"` — a real
/// `config/manifests/builtin.models.jsonc` display name — is one label, not a four-deep tree, and
/// the same free-label class travels for real in `loras[].name`: without this a LoRA the user
/// named `"Ghibli watercolor / soft light / pastel"` would be dropped from their own share with
/// no signal to anyone.
///
/// Narrowed at the SLASH, deliberately not at the segment: ignoring any segment that contains
/// whitespace would also stop catching `Documents/Secret Project/render 1.png`, which is a real
/// relative path and does leak a name. Only a slash with whitespace on one side or the other
/// stops separating.
fn relative_tree_segments(value: &str) -> usize {
    let characters: Vec<char> = value.chars().collect();
    let is_space = |index: usize| {
        characters
            .get(index)
            .is_some_and(|next: &char| next.is_whitespace())
    };
    let mut segments = 0;
    let mut segment_len = 0_usize;
    for (index, character) in characters.iter().enumerate() {
        let separates =
            *character == '/' && !(index > 0 && is_space(index - 1)) && !is_space(index + 1);
        if separates {
            // Empty segments do not count, so `a//b` is two and a leading `/` opens none —
            // exactly what the `split('/').filter(non-empty)` this replaced did.
            segments += usize::from(segment_len > 0);
            segment_len = 0;
        } else {
            segment_len += 1;
        }
    }
    segments + usize::from(segment_len > 0)
}

/// Decode only the percent escapes that matter to [`looks_like_a_location`], so an encoded
/// separator cannot walk a path past the guard. Not a general URL decoder and not meant to be.
fn percent_decode_separators(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut out = String::with_capacity(value.len());
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '%' && index + 2 < chars.len() {
            let decoded = match (
                chars[index + 1].to_ascii_lowercase(),
                chars[index + 2].to_ascii_lowercase(),
            ) {
                ('3', 'a') => Some(':'),
                ('2', 'f') => Some('/'),
                ('5', 'c') => Some('\\'),
                ('7', 'e') => Some('~'),
                ('2', 'e') => Some('.'),
                // `%25` is the escape for `%` itself — the one that makes `%252F` a
                // double-encoded separator. Without it the second decode round has nothing left
                // to do and `%252F` walks straight through.
                ('2', '5') => Some('%'),
                _ => None,
            };
            if let Some(character) = decoded {
                out.push(character);
                index += 3;
                continue;
            }
        }
        out.push(chars[index]);
        index += 1;
    }
    out
}

/// The longest a classified label may be. A bound on every free string an outside envelope can
/// put under a key we declare — a model slug, a style id, a LoRA display name, the producer
/// block. Generous for anything legitimate and finite for anything else.
const LABEL_MAX_CHARS: usize = 200;

/// One non-prose label, reduced: trimmed, bounded, and dropped when it is empty, holds a
/// control character, or is path-shaped.
///
/// The single guard [`build_workflow_share`] and [`parse_workflow_share`] both run, so the two
/// directions cannot drift apart: an envelope that arrives from outside is reduced by exactly
/// the rules that built ours.
fn shareable_label(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.chars().count() > LABEL_MAX_CHARS
        || trimmed.chars().any(char::is_control)
        || is_path_shaped(trimmed)
    {
        return None;
    }
    Some(trimmed.to_owned())
}

/// The longest an authored prose field may be, **in bytes** — `prompt`, `negativePrompt`,
/// `stylePrompt`, `systemMessage` and the structured prompt's `intent` / `runtimePrompt`.
///
/// Bytes, not characters, and that is the whole point of the number. The bound used to be 20,000
/// CHARACTERS, which is 20 kB of English and 80 kB of any 4-byte scalar — so six prose slots of
/// U+1F600 were 480 kB of prose in one envelope, four times the "~120 kB worst case" the old
/// comment claimed and enough on its own to make a 3 kB PNG a 480 kB row in the asset index. A
/// character count cannot bound a serialized size, and a serialized size is what
/// [`WORKFLOW_SHARE_MAX_BYTES`] has to compose out of.
///
/// 16 KiB is the widest UTF-8 encoding of the API's own ceiling on this field class:
/// `MAX_PROMPT_CHARS` in `apps/rust-api/src/lib.rs` rejects a `prompt` or `negativePrompt` over
/// 4,000 characters, and 4,000 characters is at most 16,000 bytes. So a prompt this app accepted
/// cannot be truncated here, in any script — but a non-Latin prompt does get fewer CHARACTERS than a
/// Latin one out of the same allowance (5,461 for 3-byte CJK, 4,096 for 4-byte scalars), which is
/// the cost of bounding the thing that is actually persisted. All four are above the API's own
/// 4,000, so the bound the user can hit is still the API's, not this one.
///
/// The four `advanced` slots have no per-field validator of their own — they are bounded only by
/// `MAX_ADVANCED_JSON_BYTES`, the API's 64 KiB ceiling on the whole serialized `advanced` map — so
/// they take the same 16 KiB. `runtimePrompt`, the longest of them, is a serialized structured
/// prompt running to a few kilobytes at its most verbose.
const PROSE_MAX_BYTES: usize = 16 * 1024;

/// Authored prose, reduced. Unlike [`shareable_label`] this keeps what the user typed —
/// including a path, which is the deliberate exemption [`is_path_shaped`] documents — and only
/// bounds it.
///
/// On the BUILD side these strings are the user's own and this is a no-op. On the PARSE side
/// they are attacker-chosen, which is the epic's trust boundary: the reader (sc-15952) renders
/// them, so a value carrying ANSI escapes, a bidi override or several megabytes of text must not
/// arrive intact. Newlines and tabs survive — a multi-line prompt is normal and mangling it would
/// be the very harm the prose exemption exists to prevent — but every other control character and
/// every invisible formatting character is dropped.
///
/// Truncation is per CHARACTER against a BYTE budget, so the result can never be split UTF-8: a
/// character is pushed only if it fits whole, and the loop stops at the first that does not. A
/// `String` cannot hold invalid UTF-8 anyway — what this avoids is the byte-slicing form of the
/// same bound, which panics on a multi-byte boundary rather than producing invalid output.
pub(crate) fn shareable_prose(value: &str) -> String {
    let mut stripped = String::new();
    for character in value.chars().filter(|&character| {
        matches!(character, '\n' | '\t')
            || !(character.is_control() || is_invisible_formatting(character))
    }) {
        if stripped.len() + character.len_utf8() > PROSE_MAX_BYTES {
            break;
        }
        stripped.push(character);
    }
    stripped.trim().to_owned()
}

/// Non-`Cf` characters whose defined filler/blank glyph is rendered as empty space in ordinary
/// text despite not being Unicode whitespace.
///
/// This is deliberately an explicit set, not an `Lo` / `So` category rule: those categories hold
/// genuine CJK, emoji and symbol content. A character belongs here only when its own named
/// filler/blank semantics render it blank, it is not whitespace, and it is outside the `Cc` / `Cf`
/// classes the general rules already remove.
const DISPLAY_BLANK_CHARACTERS: &[char] = &[
    '\u{115F}',  // HANGUL CHOSEONG FILLER
    '\u{1160}',  // HANGUL JUNGSEONG FILLER
    '\u{2800}',  // BRAILLE PATTERN BLANK
    '\u{3164}',  // HANGUL FILLER
    '\u{FFA0}',  // HALFWIDTH HANGUL FILLER
    '\u{13441}', // EGYPTIAN HIEROGLYPH FULL BLANK
    '\u{13442}', // EGYPTIAN HIEROGLYPH HALF BLANK
];

/// True for a character that can invisibly change how surrounding text is perceived: Unicode's
/// `Cf` (format) class, the `Zl`/`Zp` line and paragraph separators, plus the narrowly selected
/// non-`Cf` display blanks in [`DISPLAY_BLANK_CHARACTERS`].
///
/// [`char::is_control`] is `Cc` and only `Cc`, which is why [`shareable_prose`] cannot stop at it.
/// `Cc` does catch ANSI escapes (ESC is `Cc`), so this is not terminal injection — it is display
/// spoofing, and in a feature whose entire pitch is "trust this recipe a stranger sent you" that
/// is the more relevant one. U+202E RIGHT-TO-LEFT OVERRIDE makes a rendered prompt say something
/// other than what is stored; U+200B and U+FEFF are the same trick without the reversal, splitting
/// a word the reader believes is whole. Both survived into a recorded envelope before this guard.
///
/// The ranges are the whole `Cf` class for Unicode 16.0 rather than the handful demonstrated,
/// because a class is what closes a class. Enumerated instead of taken from a crate because `std`
/// exposes no general-category API and pulling a Unicode table in for one predicate is a poor
/// trade: `Cf` additions are rare, additive, and — being new invisible formatting characters — are
/// new display tricks, not new prompt content.
///
/// Variation selectors (U+FE00..U+FE0F) are `Mn`, not `Cf`, and are deliberately NOT here: emoji
/// presentation is prompt content. The tag characters at U+E0020..U+E007F are `Cf` and do go, which
/// costs subdivision-flag emoji (🏴󠁧󠁢󠁳󠁣󠁴󠁿) in a prompt — the right side of that trade, since the same
/// range is the standard way to smuggle hidden text through a display.
fn is_invisible_formatting(character: char) -> bool {
    DISPLAY_BLANK_CHARACTERS.contains(&character)
        || matches!(
            character,
            '\u{00AD}'
            | '\u{0600}'..='\u{0605}'
            | '\u{061C}'
            | '\u{06DD}'
            | '\u{070F}'
            | '\u{0890}'..='\u{0891}'
            | '\u{08E2}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            // Zl / Zp: a line or paragraph separator is not a newline the renderer agreed to.
            | '\u{2028}'..='\u{2029}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{110BD}'
            | '\u{110CD}'
            | '\u{13430}'..='\u{1343F}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0001}'
            | '\u{E0020}'..='\u{E007F}'
        )
}

/// A Hugging Face repo id (`owner/name`) and nothing else — never a path, never a URL.
///
/// Load-bearing beyond tidiness: a repo id is joined into a cache directory name
/// (`models--owner--name`), and sc-15952's "install the missing LoRA" action acts on whatever
/// this returns. `.` and `-` and `_` are legal INSIDE a segment, so a segment must additionally
/// start with an alphanumeric — otherwise `../x` reads as the perfectly-shaped repo id `..`/`x`.
fn hf_repo_id(value: &str) -> Option<String> {
    let repo = crate::hf_repo_id::namespaced(value).ok()?;
    (repo.chars().count() <= LABEL_MAX_CHARS).then(|| repo.to_owned())
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// What the envelope takes from the produced IMAGE rather than from the job payload.
///
/// Exists because sc-15948 embeds at the worker's write seam, where no [`Asset`] has been built
/// yet — the worker writes the PNG and reports flat facts, and the API turns those into the
/// sidecar afterwards. Rather than have the worker fabricate an `Asset` (whose `mode` and
/// `adapter` are closed enums it would have to guess at), both callers name the same handful of
/// per-image values and go through the same builder.
///
/// Every field here is a FALLBACK for the payload except `seed`, which always wins: the payload
/// carries the batch's base `seed` and its whole `seeds` list, and only the writer of one file
/// knows which of those rendered it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowAssetFacts {
    pub mode: String,
    pub model: String,
    pub prompt: String,
    pub negative_prompt: String,
    /// The seed of THIS image, never the batch base.
    pub seed: i64,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

impl WorkflowAssetFacts {
    /// The facts an already-built sidecar carries.
    #[must_use]
    pub fn from_asset(asset: &Asset) -> Self {
        Self {
            mode: asset.recipe.mode.as_str().to_owned(),
            model: asset.recipe.model.clone(),
            prompt: asset.recipe.prompt.clone(),
            negative_prompt: asset.recipe.negative_prompt.clone(),
            seed: asset.recipe.seed.unwrap_or_default(),
            width: asset.file.width,
            height: asset.file.height,
        }
    }
}

/// Build the sanitized envelope for one generated asset.
///
/// `job_payload` is the job row's `payload_json` — the exact `ImageJobRequest` the API stored,
/// NOT the asset's `recipe`. The recipe is lossy: `sourceAssetId`, `referenceAssetIds`,
/// `maskAssetId`, `fitMode` and `upscale` are split between `lineage` and the untyped
/// `rawAdapterSettings` (see `crate::project_store::build_image_sidecar_parts`).
///
/// `asset` supplies the two things the payload cannot: the media kind, and the seed of THIS
/// image — the payload carries the whole batch's `seeds`, and only the sidecar knows which one
/// produced the file being shared.
#[must_use]
pub fn build_workflow_share(asset: &Asset, job_payload: &JsonObject) -> WorkflowShare {
    build_workflow_share_from(&WorkflowAssetFacts::from_asset(asset), job_payload)
}

/// [`build_workflow_share`] against [`WorkflowAssetFacts`] instead of a built sidecar — the entry
/// point the worker's write seam uses (sc-15948).
///
/// The one implementation both forms share, so the sanitizer, the payload-over-facts precedence
/// and the allow-list cannot differ between "embedded when the image was written" and "rebuilt
/// from a sidecar". `build_workflow_share_is_the_facts_builder` in
/// `crates/sceneworks-core/tests/workflow_share.rs` pins the delegation.
#[must_use]
pub fn build_workflow_share_from(
    facts: &WorkflowAssetFacts,
    job_payload: &JsonObject,
) -> WorkflowShare {
    build_share_of_kind(WORKFLOW_KIND_IMAGE, facts, job_payload)
}

/// [`build_workflow_share_from`] for the VIDEO lane (sc-15956) — the same builder, the same single
/// reducer, and a [`WORKFLOW_KIND_VIDEO`] marker.
///
/// Ungated, like its image sibling: [`embeddable_video_workflow_share`] is the form a write seam
/// must call.
///
/// # What a video envelope carries that an image one does not
///
/// `durationSeconds`, `fps` and `quality` — the three knobs off the studio's own menus that decide
/// what gets made. `fitMode` was already a top-level field and is shared.
///
/// # What it deliberately does NOT carry
///
/// **The person-replacement fields, `personTrackId` and `replacementMode`, are withheld — always,
/// not by preference.** A shared video must not disclose that it was made by replacing a specific
/// person. `personTrackId` is an install-local id that resolves to nothing elsewhere, which alone
/// would put it in the same class as `controlImage`; `replacementMode` is not an id at all and is
/// withheld for the stronger reason, which is that the pair together tell a stranger the clip is a
/// face replacement and which mode of one. That is a fact about a real person who did not choose to
/// share it, and no replay value justifies it — a recipient replaying this recipe has no access to
/// the track anyway, so the field could only ever inform, never reproduce.
///
/// Withheld SILENTLY rather than through `omitted`, which is the one place this contract's
/// "a stated absence beats an invisible one" rule is deliberately inverted: `omitted:
/// ["personTrackId"]` would announce the replacement to every reader while withholding only the id,
/// which is the disclosure the withholding exists to prevent. `OMITTED_FIELDS` is for collections
/// too large to record, where naming the gap costs nothing.
///
/// The clip ids (`sourceClipAssetIds`, `bridgeRightClipAssetId`, `referenceClipAssetId`) are
/// withheld as ids and recorded as SHAPE through [`INPUT_KIND_SOURCE_CLIP`] /
/// [`INPUT_KIND_REFERENCE_CLIP`], exactly as the image lane treats `sourceAssetId`.
#[must_use]
pub fn build_video_workflow_share_from(
    facts: &WorkflowAssetFacts,
    job_payload: &JsonObject,
) -> WorkflowShare {
    build_share_of_kind(WORKFLOW_KIND_VIDEO, facts, job_payload)
}

fn build_share_of_kind(
    kind: &str,
    facts: &WorkflowAssetFacts,
    job_payload: &JsonObject,
) -> WorkflowShare {
    let string_field = |key: &str| {
        job_payload
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    let u32_field = |key: &str| {
        job_payload
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    };

    let mode = string_field("mode").unwrap_or_else(|| facts.mode.clone());
    let model = string_field("model").unwrap_or_else(|| facts.model.clone());
    let prompt = string_field("prompt").unwrap_or_else(|| facts.prompt.clone());
    let negative_prompt =
        string_field("negativePrompt").unwrap_or_else(|| facts.negative_prompt.clone());

    // The produced image's seed — not the payload's. `payload.seed` is the batch BASE and
    // `payload.seeds` is the whole batch; only whoever wrote one file knows which one rendered
    // it. The batch list never travels: the other images are not this share's business.
    let advanced_edit = crate::moss_image::is_edit(job_payload)
        && job_payload
            .get("advanced")
            .and_then(|v| v.get(crate::moss_image_advanced::KEY))
            .is_some();
    let seed = if job_payload
        .get("advanced")
        .and_then(|v| v.get("mossImageAdvanced"))
        .and_then(|v| v.get("noise"))
        .is_some_and(|v| !v.is_null())
    {
        None
    } else if advanced_edit {
        job_payload
            .get("seed")
            .and_then(Value::as_i64)
            .or_else(|| {
                job_payload
                    .get("advanced")
                    .and_then(|v| v.get(crate::moss_image::EDIT_REPORT_KEY))
                    .and_then(|v| v.pointer("/base/samples/0/seed"))
                    .and_then(Value::as_i64)
            })
            .or(Some(facts.seed))
    } else {
        Some(facts.seed)
    };

    // Raw on purpose: `reduce_workflow_share` runs `sanitize_advanced`, and sanitizing here as
    // well would be the same pass twice — harmless only for as long as every rule stays
    // idempotent, which is not a property worth depending on.
    let advanced = job_payload
        .get("advanced")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    // Every value goes out through the SAME reducer an incoming envelope comes in through
    // (`reduce_workflow_share`), so the build side cannot grow a field the parse side does not
    // guard — which is exactly how the top-level labels below escaped the path check before.
    let is_video = kind == WORKFLOW_KIND_VIDEO;
    // FLUX.3 reference dimensions are authored OPTIONS. Falling back to the produced crop would
    // turn automatic/one-sided sizing into a different request (and an odd crop may not be on-grid).
    let reference_output = (!is_video && crate::image_request::flux3::is_model(&model))
        .then(|| crate::image_request::flux3::parse(job_payload).ok())
        .flatten()
        .filter(|intent| !intent.asset_ids.is_empty());
    let (width, height) = match reference_output {
        Some(intent) => (intent.width, intent.height),
        None => (
            u32_field("width").or(facts.width),
            u32_field("height").or(facts.height),
        ),
    };
    let moss_edit = crate::moss_image::is_moss(&model) && mode == "edit_image";
    let invalid_moss_references = moss_edit
        && crate::moss_image::edit_reference_slots(job_payload)
            .is_ok_and(|ids| ids.is_empty() || ids.len() > 4);
    let invalid_moss_references = invalid_moss_references
        || (moss_edit && crate::moss_image::edit_reference_slots(job_payload).is_err());
    let inputs = describe_inputs(job_payload, moss_edit);
    reduce_workflow_share(WorkflowShare {
        kind: kind.to_owned(),
        schema_version: WORKFLOW_SHARE_SCHEMA_VERSION,
        producer: WorkflowProducer::default(),
        mode,
        model,
        // Only the worker write seam may add a trusted checkpoint digest. A client payload cannot
        // claim one merely by spelling an internal-looking field.
        model_hash: None,
        prompt,
        negative_prompt,
        seed,
        width,
        height,
        count: u32_field("count"),
        // Video-only, and read off the payload rather than off the encoded file: these are the ASK
        // (see the field docs). Gated on the kind so an image envelope cannot grow a `fps` from a
        // payload key that happened to be there.
        duration_seconds: is_video
            .then(|| job_payload.get("duration").and_then(Value::as_f64))
            .flatten(),
        fps: is_video.then(|| u32_field("fps")).flatten(),
        quality: is_video.then(|| string_field("quality")).flatten(),
        style_preset: string_field("stylePreset"),
        style_id: string_field("styleId"),
        fit_mode: string_field("fitMode"),
        upscale: sanitize_upscale(job_payload.get("upscale")),
        loras: sanitize_loras(job_payload.get("loras")),
        inputs,
        advanced,
        // Nothing to carry in: the marker is derived by `reduce_workflow_share` from what the
        // sanitizer actually dropped, in this direction exactly as in the other.
        omitted: if invalid_moss_references {
            vec![OMITTED_INPUTS.to_owned()]
        } else {
            Vec::new()
        },
    })
}

/// The envelope to EMBED for one generated image, or `None` when there is none to embed.
///
/// The write seam's entry point, and the only one that runs the recording gate — so a writer cannot
/// produce a file its own reader would refuse with [`WorkflowShareError::TooLarge`]. `None` means
/// "write the file exactly as today", which is a shape the seams already have: the worker's
/// `workflow_source` already collapses "the user opted out" and "there is no payload to describe"
/// into the same `Option`, and this is a third reason with the same handling.
///
/// Unreachable for a request that came through the API, and that is the intent rather than an
/// accident: `MAX_PROMPT_CHARS` and `MAX_ADVANCED_JSON_BYTES` bound the payload well under
/// [`WORKFLOW_SHARE_MAX_BYTES`] (see the derivation there), and
/// `no_real_request_is_truncated_by_the_collection_bounds` in
/// `crates/sceneworks-core/tests/workflow_share.rs` measures the widest real requests against it.
/// It is here so that the ceiling is a property of the envelope rather than of the reader.
#[must_use]
pub fn embeddable_workflow_share(
    facts: &WorkflowAssetFacts,
    job_payload: &JsonObject,
) -> Option<WorkflowShare> {
    let share = build_workflow_share_from(facts, job_payload);
    within_recording_ceiling(&share).ok().map(|()| share)
}

/// [`embeddable_workflow_share`] for the VIDEO lane (sc-15956): the gated builder a video write
/// seam must call, and the only video form that runs the recording ceiling.
///
/// The ceiling is the same number and the same measurement — the serialized envelope — because it
/// is a bound on what gets PERSISTED, not on what a container can hold. An MP4's tag store would
/// take far more than 160 KiB; that is not a reason to record more, and letting the two lanes
/// diverge would mean an envelope that round-trips through a video and is refused by the PNG
/// reader. See [`crate::workflow_mp4::workflow_metadata_size`] for what the container adds on top.
#[must_use]
pub fn embeddable_video_workflow_share(
    facts: &WorkflowAssetFacts,
    job_payload: &JsonObject,
) -> Option<WorkflowShare> {
    let share = build_video_workflow_share_from(facts, job_payload);
    within_recording_ceiling(&share).ok().map(|()| share)
}

/// Input images by shape. Reads the ids only to count them; no id ever reaches the envelope.
fn describe_inputs(job_payload: &JsonObject, moss_edit: bool) -> Vec<WorkflowInput> {
    let has_id = |key: &str| {
        job_payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    let id_list_len = |key: &str| {
        job_payload
            .get(key)
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter(|value| value.as_str().is_some_and(|value| !value.trim().is_empty()))
                    .count()
            })
            .unwrap_or(0)
    };

    let mut inputs = Vec::new();
    // `lastFrameAssetId` is the video lane's end-frame target: a STILL, like `sourceAssetId`, so it
    // counts here rather than as a clip. An image payload never carries it, so this is a no-op for
    // the image lane.
    let sources = usize::from(!moss_edit && has_id("sourceAssetId"))
        + usize::from(has_id("lastFrameAssetId"));
    if sources > 0 {
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_SOURCE.to_owned(),
            count: u32::try_from(sources).unwrap_or(u32::MAX),
            control_mode: None,
        });
    }
    let references = if moss_edit {
        crate::moss_image::edit_reference_slots(job_payload).map_or(0, |ids| ids.len())
    } else {
        usize::from(has_id("referenceAssetId")) + id_list_len("referenceAssetIds")
    };
    if references > 0 {
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_REFERENCE.to_owned(),
            count: u32::try_from(references).unwrap_or(u32::MAX),
            control_mode: None,
        });
    }
    if has_id("maskAssetId") {
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_MASK.to_owned(),
            count: 1,
            control_mode: None,
        });
    }
    // The video lane's clip ids (sc-15956), counted exactly as the image lane counts its stills.
    // `lastFrameAssetId` is a STILL, not a clip — it is the end-frame an i2v run targets — so it
    // joins the `source` count above rather than this one.
    let source_clips = usize::from(has_id("sourceClipAssetId"))
        + usize::from(has_id("bridgeRightClipAssetId"))
        + id_list_len("sourceClipAssetIds");
    if source_clips > 0 {
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_SOURCE_CLIP.to_owned(),
            count: u32::try_from(source_clips).unwrap_or(u32::MAX),
            control_mode: None,
        });
    }
    if has_id("referenceClipAssetId") {
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_REFERENCE_CLIP.to_owned(),
            count: 1,
            control_mode: None,
        });
    }
    // The audio references (sc-17160), counted exactly as the still and clip references are.
    // Without their own kind a shared multi-modal recipe would describe "9 reference images +
    // 2 source clips" and say nothing about the audio it also needs — the recipient sees a
    // complete-looking input list and reproduces an under-conditioned render.
    let reference_audio = id_list_len("referenceAudioAssetIds");
    if reference_audio > 0 {
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_REFERENCE_AUDIO.to_owned(),
            count: u32::try_from(reference_audio).unwrap_or(u32::MAX),
            control_mode: None,
        });
    }
    let advanced = job_payload.get("advanced").and_then(Value::as_object);
    let control_image = advanced
        .and_then(|advanced| advanced.get("controlImage"))
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    if control_image {
        let control_mode = advanced
            .and_then(|advanced| advanced.get("controlMode"))
            .and_then(Value::as_str)
            .and_then(shareable_label);
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_CONTROL.to_owned(),
            count: 1,
            control_mode,
        });
    }
    inputs
}

fn sanitize_upscale(value: Option<&Value>) -> Option<WorkflowUpscale> {
    let upscale = value?.as_object()?;
    let enabled = upscale
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !enabled {
        return None;
    }
    Some(WorkflowUpscale {
        enabled: true,
        factor: upscale
            .get("factor")
            .and_then(Value::as_u64)
            .and_then(|factor| u32::try_from(factor).ok()),
        engine: upscale
            .get("engine")
            .and_then(Value::as_str)
            .and_then(shareable_label),
        softness: upscale.get("softness").and_then(Value::as_f64),
        scale: upscale
            .get("scale")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0),
        input_sizing: upscale
            .get("inputSizing")
            .and_then(Value::as_str)
            .filter(|value| matches!(*value, "budgeted" | "original"))
            .map(str::to_owned),
        color_fix: upscale.get("colorFix").and_then(Value::as_bool),
    })
}

fn sanitize_loras(value: Option<&Value>) -> Vec<WorkflowLora> {
    let Some(entries) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|entry| {
            // The catalog entry's `source` is `{ provider, repo, file }` for a Hugging Face
            // LoRA and `{ provider: "local", path }` for an imported one. Only the repo id
            // travels — never `path`, never `file`, never `installedPath`/`sourcePath`.
            let repo = entry
                .get("source")
                .and_then(Value::as_object)
                .filter(|source| {
                    source.get("provider").and_then(Value::as_str) == Some("huggingface")
                })
                .and_then(|source| source.get("repo"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            // `reduce_lora` is the shared guard — `hf_repo_id` on the repo, the path check on
            // the name — so a repo id that arrives in an envelope is validated exactly as one
            // read out of a job payload is.
            reduce_lora(WorkflowLora {
                name: entry.get("name").and_then(Value::as_str).map(str::to_owned),
                weight: entry.get("weight").and_then(Value::as_f64),
                repo,
                // Deliberately never read from the request. Only the worker write seam may attach
                // a digest after it resolves the same file inference loads.
                hash: None,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Value-level reduction (shared by build and parse)
// ---------------------------------------------------------------------------

/// The input kinds, in the order [`describe_inputs`] emits them. An `inputs[].kind` outside this
/// set is not something a reader can act on, so it does not survive a parse.
///
/// The last two are the video lane's (sc-15956), and they arrive here for the same reason the first
/// four did: a video recipe's `sourceClipAssetIds` / `referenceClipAssetId` are exactly the
/// identifier class the allow-list excludes, so they become SHAPE — "this recipe needs two source
/// clips" — instead of dangling local UUIDs.
pub const INPUT_KINDS: &[&str] = &[
    INPUT_KIND_SOURCE,
    INPUT_KIND_REFERENCE,
    INPUT_KIND_MASK,
    INPUT_KIND_CONTROL,
    INPUT_KIND_SOURCE_CLIP,
    INPUT_KIND_REFERENCE_CLIP,
    INPUT_KIND_REFERENCE_AUDIO,
];

// ---------------------------------------------------------------------------
// Collection bounds and the recording ceiling
// ---------------------------------------------------------------------------
//
// [`LABEL_MAX_CHARS`] and [`PROSE_MAX_BYTES`] bound every individual STRING an envelope can carry
// and the caps below bound how MANY of them it can carry. The only ceiling before either existed
// was sc-15947's 1 MiB cap on the decompressed chunk text, which a compressible envelope turns
// into enormous leverage: what an attacker pays for is the COMPRESSED chunk and what we pay is the
// decompressed envelope, persisted into the sidecar, the `assets.asset_json` row and every
// `list_assets` response for that asset (sc-14797 / sc-14798) forever.
//
// PER-FIELD BOUNDS DO NOT COMPOSE, and that is the lesson this section was rewritten around. The
// first attempt was per-collection caps, and each new measurement found a new way to spend what the
// caps left: 200,000 `null` coordinates cost nothing against a budget that counted only numbers;
// 200,000 empty arrays cost nothing against either; six prose slots of a 4-byte scalar were 480 kB
// under a bound written in characters. Measured at that point, from a real import: a 3,070-byte PNG
// became a 480,321-byte `extra.importedWorkflow`, and a 4,043-byte PNG became 720,347 bytes — worse
// than the 111x the caps were introduced to remove.
//
// So there are two kinds of bound here, and the second is the one that closes the class:
//
// * the per-collection caps, each derived from the validator that already limits the thing, so an
//   envelope claiming more did not come from a run here; and
// * [`WORKFLOW_SHARE_MAX_BYTES`], ONE ceiling on the serialized envelope, checked after every
//   per-field rule has run. It composes by construction — it is a bound on the thing we actually
//   persist — so a new field or a new leak inside an existing one cannot walk around it the way
//   each new vector walked around the caps.
//
// Everything here runs in ONE place both directions share, so the write side cannot drift from the
// read side.

/// The most LoRAs an envelope may name — [`crate::lora_family::MAX_JOB_LORAS`] exactly.
///
/// Not a chosen number: it is the hard per-job total the generation path REJECTS above
/// (`apps/rust-api/src/lib.rs`, the worker's `image_jobs` guard) and the recipe-preset
/// normalizer's own cap ("Recipe presets can include at most 10 LoRAs"). A run that applied more
/// than this could not have happened here, so an envelope claiming it did not come from us.
const MAX_SHARE_LORAS: usize = crate::lora_family::MAX_JOB_LORAS;

/// The most input descriptors an envelope may carry — one per kind, and the kinds are a closed
/// vocabulary.
///
/// Exact rather than generous: [`describe_inputs`] emits at most one entry per [`INPUT_KINDS`]
/// member (a multi-reference run is `{ kind: "reference", count: N }`, one entry with a count, not
/// N entries). So [`INPUT_KINDS`]`.len()` is not a budget, it is the shape.
const MAX_SHARE_INPUTS: usize = INPUT_KINDS.len();

/// The most multi-phase phases an envelope may carry.
///
/// The worker's `MAX_MULTIPHASE_PHASES` and the web's `MULTIPHASE_MAX_PHASES`, which are the same
/// 8 and are the values the request is validated against before it ever runs. `sceneworks-worker`
/// depends on this crate, so the constant cannot be imported; `the_phase_cap_matches_the_worker`
/// in `crates/sceneworks-core/tests/workflow_share.rs` reads both files and fails if they drift.
const MAX_SHARE_PHASES: usize = 8;

/// The most pose entries an envelope may carry.
///
/// Derived directly from [`crate::image_request::MAX_JOB_POSES`], the request contract enforced by
/// the API and mirrored by the shared web picker. A valid newly submitted job therefore cannot
/// first discover this limit when its workflow is embedded in an output image. The sanitizer still
/// retains its over-cap omission marker for untrusted or legacy envelopes.
///
/// This bounds the ENTRY count only; [`MAX_SHARE_POSE_SLOTS`] is what bounds the volume.
const MAX_SHARE_POSES: usize = crate::image_request::MAX_JOB_POSES;

/// The most coordinate SLOTS the whole `advanced.poses` array may carry, across every entry and
/// field: a number, or a `null` standing in for one.
///
/// Slots rather than numbers, and that distinction is a fixed bug rather than a nicety. The budget
/// counted `Value::Number` only, while [`is_coordinate_tree`] accepted `Value::Null` as a
/// coordinate — so 200,000 nulls under one `keypoints` key cost nothing against a 6,144-number
/// budget and serialized to 1,000,027 bytes. Nulls are counted because they ARE coordinate slots
/// (the worker's `normalize_points` fills a missing one), not because they are hostile. Empty
/// arrays were the same hole from the other side and are refused outright by
/// [`is_coordinate_tree`]: an array with nothing in it is not a point.
///
/// A budget rather than a per-entry cap, because that is what admits the real cases while still
/// bounding the worst one. One sanitized pose is at most 18 body keypoints + 42 hand + 68 face =
/// 128 points (the worker's `normalize_keypoints` / `normalize_hands` / `normalize_face` truncate
/// to exactly those counts), and a point is `[x, y]` or `[x, y, confidence]` — so 384 slots, and
/// this budget is 16 full whole-body skeletons' worth. 16 is also twice
/// [`crate::image_request::MAX_COUNT`], the payload-sanity ceiling on how many images one image
/// job may produce, which a pose set is the pose lane's version of.
///
/// Checked against the real cases rather than asserted: all 46 built-in poses are body keypoints
/// only (18 points, no hands, no face), so selecting the entire library spends 46 x 36 = 1,656
/// slots — 27% of the budget. What it refuses is dozens of poses each carrying full hands and face,
/// and what backstops IT — because 6,144 slots is ~147 kB of JSON at the widest an `f64` serializes,
/// which is over the whole envelope's allowance — is [`WORKFLOW_SHARE_MAX_BYTES`].
const MAX_SHARE_POSE_SLOTS: usize = 16 * (18 + 42 + 68) * 3;

/// The ceiling on the SERIALIZED envelope — the one bound that composes, and the reason the
/// per-collection caps above are no longer the last line.
///
/// Derived from the widest envelope this app can legitimately produce, which is an arithmetic sum
/// of validators rather than an estimate. Every term is a hard ceiling something else already
/// enforces:
///
/// | term | bytes | where it comes from |
/// |------|-------|---------------------|
/// | `prompt` + `negativePrompt` | 32,000 | `MAX_PROMPT_CHARS` (4,000) x 4 bytes, twice |
/// | `advanced` | 65,536 | `MAX_ADVANCED_JSON_BYTES`, the API's ceiling on the serialized map |
/// | `loras` | 16,500 | [`MAX_SHARE_LORAS`] x (name + repo at [`LABEL_MAX_CHARS`] x 4 + weight) |
/// | labels + `upscale` + `producer` + scalars | ~15,000 | [`LABEL_MAX_CHARS`] x 4, per field |
/// | | **~122,200** | |
///
/// 160 KiB leaves 34% headroom over that, which the measured real cases make look enormous: the
/// golden fixture is 1,239 bytes, a Krea multi-phase recipe at the validators' own ceiling is
/// ~2 kB, and a long CJK prompt is ~2 kB — all two orders of magnitude under. What the ceiling
/// refuses is the composition the caps could not: the sanitizer's own per-field bounds still admit
/// ~400 kB in total (six 16 KiB prose slots, ~40 allow-listed scalar labels, and a 6,144-slot pose
/// budget), and this is what stops that from being persisted.
///
/// Over the ceiling degrades to NO WORKFLOW — [`WorkflowShareError::TooLarge`] on the read side, a
/// `None` from [`embeddable_workflow_share`] on the write side — rather than to a partial record.
/// Shedding the biggest field instead would mean recording a recipe missing exactly the part that
/// made it too large, and the reader has no way to know which one that was.
pub const WORKFLOW_SHARE_MAX_BYTES: usize = 160 * 1024;

/// How many bytes an envelope serializes to: the unit [`WORKFLOW_SHARE_MAX_BYTES`] is spent in, and
/// the same measurement the sidecar row and the `list_assets` payload will pay.
fn workflow_share_bytes(share: &WorkflowShare) -> usize {
    // A serialization failure is not reachable for this type (no map with non-string keys, no
    // non-finite float — `reduce_workflow_share` filters those). Treating it as "over the ceiling"
    // is still the right fallback: an envelope we cannot serialize is one we cannot record.
    serde_json::to_string(share).map_or(usize::MAX, |text| text.len())
}

/// The recording gate, run by BOTH directions.
///
/// One function rather than a check at each end, for the same reason every other guard in this file
/// is one function: a ceiling the reader enforces and the writer does not is a writer that produces
/// files its own reader refuses.
fn within_recording_ceiling(share: &WorkflowShare) -> Result<(), WorkflowShareError> {
    let bytes = workflow_share_bytes(share);
    if bytes > WORKFLOW_SHARE_MAX_BYTES {
        return Err(WorkflowShareError::TooLarge {
            bytes,
            limit: WORKFLOW_SHARE_MAX_BYTES,
        });
    }
    Ok(())
}

/// One bounded collection: kept whole, or dropped whole. Never truncated.
///
/// The truncate-or-drop question, answered the same way [`shareable_label`] answers it for an
/// over-long label — by dropping, because a collection's MEMBERSHIP is its identity in exactly the
/// way a slug's spelling is. "The first 5 of these 8,000 LoRAs" is not the recipe that made the
/// image, and sc-15952 offers what survives here to the user as "Use this recipe": a plausible
/// subset is worse than a stated absence, because a reader cannot tell it is a subset.
///
/// A stated absence, now: the original form of this argument claimed a reader "can tell an
/// absence", and it cannot — see [`OMITTED_FIELDS`], which is what makes the drop legible.
///
/// [`shareable_prose`] truncates instead, and the difference is the point: prose still means what
/// it said after its tail is cut, and a list does not.
///
/// Takes the length rather than the values, so an over-cap array is refused without being cloned
/// first — the 8,000-entry case allocated 8,000 `Value`s to decide it wanted none of them.
fn over_cap(len: usize, max: usize) -> bool {
    len > max
}

/// Reduce every VALUE in an envelope to what the contract allows.
///
/// The typed struct reduces at KEY granularity only — it drops fields we do not declare, which
/// is not the same as trusting the ones we do. An envelope that arrived from outside carries
/// attacker-chosen strings under keys we *do* declare, and `loras[].repo` is the sharpest edge:
/// a Hugging Face repo id is joined into a cache path (`models--owner--name`), so sc-15952's
/// "install the missing LoRA" action would otherwise inherit whatever traversal string the file
/// contained. Both directions run this one function so the guards cannot drift apart.
fn reduce_workflow_share(share: WorkflowShare) -> WorkflowShare {
    let WorkflowShare {
        kind,
        schema_version,
        producer,
        mode,
        model,
        model_hash,
        prompt,
        negative_prompt,
        seed,
        width,
        height,
        count,
        duration_seconds,
        fps,
        quality,
        style_preset,
        style_id,
        fit_mode,
        upscale,
        loras,
        inputs,
        advanced,
        omitted,
    } = share;

    // Every collection's loss is derived by COMPARING what came in with what the sanitizer let out,
    // rather than reported by the rules themselves. One derivation instead of a marker push at every
    // `return None`, and it cannot go stale against a rule it does not know about: a new reason to
    // drop `advanced.poses` is a new reason this records it.
    let declared_loras = loras.len();
    let declared_inputs = inputs.len();
    let loras: Vec<WorkflowLora> = if over_cap(declared_loras, MAX_SHARE_LORAS) {
        // Bounded BEFORE reduction, not after: an envelope that declares 8,000 LoRAs is
        // disqualified by declaring them, and capping the survivors instead would let 7,995 junk
        // entries carry 5 real ones through.
        Vec::new()
    } else {
        loras.into_iter().filter_map(reduce_lora).collect()
    };
    let inputs: Vec<WorkflowInput> = if over_cap(declared_inputs, MAX_SHARE_INPUTS) {
        Vec::new()
    } else {
        inputs.into_iter().filter_map(reduce_input).collect()
    };
    let sanitized_advanced = sanitize_advanced(&advanced);

    let mut lost: Vec<String> = reduce_omitted(omitted);
    if loras.len() < declared_loras {
        lost.push(OMITTED_LORAS.to_owned());
    }
    if inputs.len() < declared_inputs {
        lost.push(OMITTED_INPUTS.to_owned());
    }
    lost.extend(advanced_omissions(&advanced, &sanitized_advanced));

    WorkflowShare {
        kind,
        schema_version,
        producer: reduce_producer(producer),
        mode: shareable_label(&mode).unwrap_or_default(),
        model: shareable_label(&model).unwrap_or_default(),
        model_hash: model_hash.as_deref().and_then(normalize_sha256),
        // Authored prose: exempt from the PATH check on purpose (see `is_path_shaped`), but not
        // from the other two bounds — on the way IN this is a stranger's text, and sc-15952
        // renders it.
        prompt: shareable_prose(&prompt),
        negative_prompt: shareable_prose(&negative_prompt),
        seed,
        width,
        height,
        count,
        // The video knobs, reduced on the way IN exactly as on the way out. A non-finite
        // `durationSeconds` from a stranger's file would serialize as `null` and re-parse as a
        // different envelope, so it is dropped here rather than carried — the same treatment
        // `upscale.softness` already gets. `quality` is a menu label and goes through the same
        // bound-and-path check as every other label.
        duration_seconds: duration_seconds.filter(|value| value.is_finite()),
        fps,
        quality: quality.as_deref().and_then(shareable_label),
        style_preset: style_preset.as_deref().and_then(shareable_label),
        style_id: style_id.as_deref().and_then(shareable_label),
        fit_mode: fit_mode.as_deref().and_then(shareable_label),
        upscale: upscale.map(|upscale| WorkflowUpscale {
            enabled: upscale.enabled,
            factor: upscale.factor,
            engine: upscale.engine.as_deref().and_then(shareable_label),
            softness: upscale.softness.filter(|value| value.is_finite()),
            scale: upscale
                .scale
                .filter(|value| value.is_finite() && *value > 0.0),
            input_sizing: upscale
                .input_sizing
                .filter(|value| matches!(value.as_str(), "budgeted" | "original")),
            color_fix: upscale.color_fix,
        }),
        loras,
        inputs,
        advanced: sanitized_advanced,
        omitted: reduce_omitted(lost),
    }
}

/// Which `advanced` collections went missing between the raw map and the sanitized one.
///
/// Read off the two maps rather than pushed by the rules, so it stays true when a rule changes: a
/// non-empty array that the allow-list, the entry cap, the slot budget or a shape check reduced to
/// nothing is a loss whatever caused it, and an input that was already empty is an absence rather
/// than an omission.
fn advanced_omissions(raw: &JsonObject, sanitized: &JsonObject) -> Vec<String> {
    let mut lost = Vec::new();
    let declared = |key: &str| {
        raw.get(key)
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty())
    };
    for (key, name) in [("poses", OMITTED_POSES), ("phases", OMITTED_PHASES)] {
        if declared(key) && !sanitized.contains_key(key) {
            lost.push(name.to_owned());
        }
    }
    // Phase LoRA schedules are the substance of a Krea multi-phase recipe, and a phase's `loras`
    // going missing is invisible in exactly the way the whole marker exists for: over the cap the
    // key is omitted, so the phase reads as "applies no LoRAs". Counted rather than index-matched
    // because a malformed phase does not occupy a slot in the output the way a malformed pose does.
    let with_loras = |map: &JsonObject| -> usize {
        map.get("phases")
            .and_then(Value::as_array)
            .map(|phases| {
                phases
                    .iter()
                    .filter(|phase| {
                        phase
                            .get("loras")
                            .and_then(Value::as_array)
                            .is_some_and(|loras| !loras.is_empty())
                    })
                    .count()
            })
            .unwrap_or(0)
    };
    if sanitized.contains_key("phases") && with_loras(sanitized) < with_loras(raw) {
        lost.push(OMITTED_PHASE_LORAS.to_owned());
    }
    lost
}

/// One LoRA reference, reduced. An entry left with nothing to say is dropped entirely.
fn reduce_lora(lora: WorkflowLora) -> Option<WorkflowLora> {
    let reduced = WorkflowLora {
        name: lora.name.as_deref().and_then(shareable_label),
        weight: lora.weight.filter(|weight| weight.is_finite()),
        // `owner/name` and nothing else. This is the value sc-15952 turns into a cache path.
        repo: lora.repo.as_deref().and_then(hf_repo_id),
        hash: lora.hash.as_deref().and_then(normalize_sha256),
    };
    (reduced.name.is_some()
        || reduced.weight.is_some()
        || reduced.repo.is_some()
        || reduced.hash.is_some())
    .then_some(reduced)
}

/// Build one LoRA record from worker-resolved facts while retaining the portable repository hint
/// from the original request. `hash` is trusted only by call-site placement: the ordinary request
/// sanitizer above never reads a digest, so a client cannot forge one by naming the same field.
#[must_use]
pub fn trusted_lora_for_share(
    raw: &Value,
    name: String,
    weight: f64,
    hash: Option<String>,
) -> Option<WorkflowLora> {
    let repo = raw
        .get("source")
        .and_then(Value::as_object)
        .filter(|source| source.get("provider").and_then(Value::as_str) == Some("huggingface"))
        .and_then(|source| source.get("repo"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    reduce_lora(WorkflowLora {
        name: Some(name),
        weight: Some(weight),
        repo,
        hash,
    })
}

/// One input descriptor, reduced. A kind outside [`INPUT_KINDS`] is dropped — the shape list is
/// a closed vocabulary, not free text.
fn reduce_input(input: WorkflowInput) -> Option<WorkflowInput> {
    if !INPUT_KINDS.contains(&input.kind.as_str()) {
        return None;
    }
    Some(WorkflowInput {
        control_mode: input.control_mode.as_deref().and_then(shareable_label),
        kind: input.kind,
        count: input.count,
    })
}

/// The producer block, bounded. In an envelope from outside these are three free strings, so a
/// name that is not a plain label, a URL that is not `http(s)`, or a version that is not strict
/// `MAJOR.MINOR.PATCH` is reduced to empty rather than echoed to the user as provenance.
fn reduce_producer(producer: WorkflowProducer) -> WorkflowProducer {
    WorkflowProducer {
        name: shareable_label(&producer.name).unwrap_or_default(),
        url: shareable_label(&producer.url)
            .filter(|url| is_web_url(url))
            .unwrap_or_default(),
        version: shareable_label(&producer.version)
            .filter(|version| is_strict_semver(version))
            .unwrap_or_default(),
    }
}

/// An `http`/`https` URL with no whitespace in it. Deliberately narrow: the producer URL is the
/// one URL the envelope carries, and anything else there is provenance nobody vouched for.
fn is_web_url(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.starts_with("https://") || lower.starts_with("http://"))
        && !value.chars().any(char::is_whitespace)
}

/// Strict `MAJOR.MINOR.PATCH` — the same shape `PRODUCER_VERSION` is asserted to have, applied
/// to a version that arrived from somewhere else.
fn is_strict_semver(value: &str) -> bool {
    let mut parts = value.split('.');
    let (Some(major), Some(minor), Some(patch), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    [major, minor, patch].iter().all(|part| {
        !part.is_empty()
            && part.len() <= 6
            && part.bytes().all(|byte| byte.is_ascii_digit())
            && (*part == "0" || !part.starts_with('0'))
    })
}

// ---------------------------------------------------------------------------
// The sanitizer
// ---------------------------------------------------------------------------

/// Reduce a request's `advanced` map to its allow-listed, shape-checked subset.
///
/// Unclassified keys are dropped, not passed through — that is the whole design (see
/// [`ADVANCED_KEY_RULES`]).
#[must_use]
pub fn sanitize_advanced(advanced: &JsonObject) -> JsonObject {
    let mut out = JsonObject::new();
    for (key, value) in advanced {
        let Some(rule) = advanced_key_rule(key) else {
            continue;
        };
        if rule.disposition != AdvancedDisposition::Allow {
            continue;
        }
        let sanitized = match rule.shape {
            AdvancedShape::Scalar => sanitize_scalar(key, value),
            AdvancedShape::StructuredPrompt => sanitize_structured_prompt(value),
            AdvancedShape::Appearance => sanitize_appearance(value),
            AdvancedShape::Poses => sanitize_poses(value),
            AdvancedShape::Phases => sanitize_phases(value),
            AdvancedShape::H3VedaSettings => sanitize_h3_veda_settings(value),
            AdvancedShape::MossImageOptions => sanitize_moss_options(value),
            AdvancedShape::MossImageReport => sanitize_moss_execution_report(value),
            AdvancedShape::MossImageEditOptions => {
                let mut advanced = JsonObject::new();
                advanced.insert(crate::moss_image::EDIT_OPTIONS_KEY.into(), value.clone());
                crate::moss_image::edit_options(&advanced)
                    .ok()
                    .and_then(|v| serde_json::to_value(v).ok())
            }
            AdvancedShape::MossImageEditReport => sanitize_moss_edit_execution_report(value),
            AdvancedShape::MossImageAdvanced => crate::moss_image_advanced::share_options(value),
            AdvancedShape::PromptBatch => value
                .as_array()
                .filter(|p| {
                    !p.is_empty()
                        && p.len() <= 8
                        && p.iter().all(|v| {
                            v.as_str()
                                .is_some_and(|s| !s.is_empty() && s.chars().count() <= 4000)
                        })
                })
                .map(|values| {
                    Value::Array(
                        values
                            .iter()
                            .map(|value| {
                                Value::String(shareable_prose(
                                    value.as_str().expect("validated prose"),
                                ))
                            })
                            .collect(),
                    )
                }),
            AdvancedShape::SampleIndex => {
                value.as_u64().filter(|index| *index < 8).map(Value::from)
            }
            AdvancedShape::NumberPair => sanitize_number_pair(value),
            AdvancedShape::Flux3Layout => sanitize_flux3_layout(value),
            AdvancedShape::StringList => sanitize_reference_ids(value),
        };
        if let Some(sanitized) = sanitized {
            out.insert(key.clone(), sanitized);
        }
    }
    out
}

fn sanitize_reference_ids(value: &Value) -> Option<Value> {
    let values = value.as_array()?;
    if values.len() > 10 {
        return None;
    }
    let ids = values
        .iter()
        .map(|v| v.as_str().filter(|s| !s.is_empty() && !is_path_shaped(s)))
        .collect::<Option<Vec<_>>>()?;
    if ids.iter().collect::<std::collections::HashSet<_>>().len() != ids.len() {
        return None;
    }
    Some(value.clone())
}

fn sanitize_flux3_layout(value: &Value) -> Option<Value> {
    let rows = value.get("boxes")?.as_array()?;
    let identities = rows
        .iter()
        .filter_map(|r| {
            r.get("reference_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    crate::image_request::flux3::validate_layout(value, 10, Some(&identities)).ok()?;
    let mut boxes = Vec::new();
    for row in rows {
        let mut public = JsonObject::new();
        for key in [
            "id",
            "desc",
            "bbox",
            "from",
            "reference_id",
            "src_bbox",
            "tgt_bbox",
            "bbox_space",
            "src_space",
            "tgt_space",
        ] {
            if let Some(v) = row.get(key) {
                if key != "desc" && v.as_str().is_some_and(is_path_shaped) {
                    return None;
                }
                public.insert(key.to_owned(), v.clone());
            }
        }
        boxes.push(Value::Object(public));
    }
    Some(serde_json::json!({"version":1,"boxes":boxes}))
}

fn sanitize_moss_options(value: &Value) -> Option<Value> {
    use gen_core::moss_image::{MossImageOptions, MossImageResolved, MossImageVariant};
    let options: MossImageOptions = serde_json::from_value(value.clone()).ok()?;
    MossImageResolved::resolve(
        MossImageVariant::Sft,
        &gen_core::GenerationRequest {
            moss_image: Some(options.clone()),
            ..Default::default()
        },
    )
    .ok()?;
    serde_json::to_value(options).ok()
}
/// Keep the native validated receipt while excluding this machine's optional adapter location.
/// The source receipt still identifies the exact upstream adapter bytes by repository/revision/hash.
pub fn sanitize_moss_execution_report(value: &Value) -> Option<Value> {
    use gen_core::moss_image::MossImageExecutionReport;
    let report: MossImageExecutionReport = serde_json::from_value(value.clone()).ok()?;
    let mut data = report.into_data();
    for adapter in &mut data.adapter_reports {
        adapter.adapter_path = std::path::PathBuf::from("lora.safetensors");
    }
    serde_json::to_value(MossImageExecutionReport::new(data).ok()?).ok()
}

/// Native validation protects report identities; only local filesystem locations are scrubbed.
pub fn sanitize_moss_edit_execution_report(value: &Value) -> Option<Value> {
    let mut report: gen_core::moss_image_edit::MossImageEditExecutionReport =
        serde_json::from_value(value.clone()).ok()?;
    report.adapter_path = std::path::PathBuf::from("adapter_model.safetensors");
    let base = serde_json::to_value(&report.base).ok()?;
    report.base = serde_json::from_value(sanitize_moss_execution_report(&base)?).ok()?;
    serde_json::to_value(report).ok()
}

/// Preserve a complete, valid Veda control object. The API's shared H3 validator owns the wire
/// grammar and `deny_unknown_fields`; this boundary adds a small byte ceiling before parsing, then
/// keeps the original strings intact so ranges and separators replay exactly as authored.
fn sanitize_h3_veda_settings(value: &Value) -> Option<Value> {
    const MAX_CONTROL_TEXT_BYTES: usize = PROSE_MAX_BYTES;
    const MAX_H3_STEPS: u32 = 200;

    let object = value.as_object()?;
    if object.values().any(|member| {
        member
            .as_str()
            .is_some_and(|text| text.len() > MAX_CONTROL_TEXT_BYTES)
    }) {
        return None;
    }
    let mut advanced = JsonObject::new();
    advanced.insert("veda".to_owned(), value.clone());
    crate::h3_veda::requested_h3_veda("minimax_h3", &advanced, MAX_H3_STEPS).ok()?;
    Some(Value::Object(object.clone()))
}

/// The prose keys inside a sanitized value that are what the user typed, and so are exempt
/// from the path-shape guard (see [`is_path_shaped`]).
///
/// Exempt from the PATH check only. They are still bounded and control-stripped by
/// [`shareable_prose`], because on the parse side they are a stranger's strings.
const PROSE_KEYS: &[&str] = &[
    "appearancePrompt",
    "appearanceNegativePrompt",
    "stylePrompt",
    "intent",
    "runtimePrompt",
    // The interleave system prompt (sc-15948). Prose rather than a label because the user typed
    // it into a textarea and it is the literal instruction the model received; bounding it as a
    // slug would silently drop a system message that happened to mention a directory, which is
    // the exact silent-loss failure this epic's lint exists to prevent.
    "systemMessage",
];

fn sanitize_appearance(value: &Value) -> Option<Value> {
    let mut selection: crate::appearance::AppearanceSelection =
        serde_json::from_value(value.clone()).ok()?;
    selection.camera = selection.camera.as_deref().and_then(shareable_label);
    selection.film = selection.film.as_deref().and_then(shareable_label);
    selection.lighting = selection.lighting.as_deref().and_then(shareable_label);
    selection.looks = selection
        .looks
        .iter()
        .filter_map(|id| shareable_label(id))
        .take(32)
        .collect();
    serde_json::to_value(selection).ok()
}

fn sanitize_scalar(key: &str, value: &Value) -> Option<Value> {
    match value {
        Value::String(text) => {
            if PROSE_KEYS.contains(&key) {
                return Some(Value::String(shareable_prose(text)));
            }
            // Everything else under a scalar key is a slug — the same class as a top-level
            // label, so it gets the same bound rather than only the path check.
            shareable_label(text).map(Value::String)
        }
        Value::Number(_) | Value::Bool(_) => Some(value.clone()),
        // An object or array under a scalar key is the smuggling channel this guard exists to
        // close: it is how a path reaches the file through a key that is otherwise allowed.
        _ => None,
    }
}

fn sanitize_number_pair(value: &Value) -> Option<Value> {
    let pair = value.as_array().filter(|pair| pair.len() == 2)?;
    pair.iter()
        .all(|number| number.as_f64().is_some_and(f64::is_finite))
        .then(|| value.clone())
}

fn sanitize_structured_prompt(value: &Value) -> Option<Value> {
    let recipe = value.as_object()?;
    // `caption` is the model-authored structured JSON — a free-form nested object, so it
    // cannot be allow-listed key by key. Nothing is lost: `runtimePrompt` is its serialized
    // form and is exactly what the top-level `prompt` already carries.
    const FIELDS: &[&str] = &[
        "version",
        "intent",
        "magicPromptBackend",
        "edited",
        "runtimePrompt",
    ];
    let mut out = JsonObject::new();
    for field in FIELDS {
        if let Some(field_value) = recipe.get(*field) {
            if let Some(sanitized) = sanitize_scalar(field, field_value) {
                out.insert((*field).to_owned(), sanitized);
            }
        }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

/// The pose fields the worker's `parse_poses` actually reads
/// (`crates/sceneworks-worker/src/image_jobs/base.rs`). All three are pure coordinate arrays
/// with nothing identifying in them, and `hands` / `face` change the rendered skeleton — and so
/// the image — exactly as `keypoints` does. Dropping them would cost reproduction fidelity for
/// zero privacy gain. What does NOT travel is the pose-library `id` that named the entry, and
/// anything else the picker attached to it.
///
/// Public so `docs/workflow-share-envelope.md` can be pinned to it: the document names these three
/// in prose twice, and `workflow_share_doc.rs` asserts both mentions against this slice in both
/// directions.
pub const POSE_FIELDS: &[&str] = &["keypoints", "hands", "face"];

fn sanitize_poses(value: &Value) -> Option<Value> {
    let poses = value.as_array()?;
    // Checked against the length before anything is cloned: the 8,000-entry case used to allocate
    // 8,000 `Value`s to decide it wanted none of them, which is work done precisely in the case the
    // guard exists to make cheap.
    if over_cap(poses.len(), MAX_SHARE_POSES) {
        return None;
    }
    // The entry count is load-bearing (poses replace `count` variations), so an entry whose
    // coordinates are missing or malformed still occupies its slot as an empty object.
    let sanitized: Vec<Value> = poses
        .iter()
        .map(|pose| {
            let mut out = JsonObject::new();
            if let Some(pose) = pose.as_object() {
                for field in POSE_FIELDS {
                    // An array, and a coordinate tree: a bare number or a bare `null` under
                    // `keypoints` is not a skeleton, and letting one through would record a
                    // positive claim about a pose nobody made.
                    if let Some(points) = pose
                        .get(*field)
                        .filter(|value| value.is_array() && is_coordinate_tree(value))
                    {
                        out.insert((*field).to_owned(), points.clone());
                    }
                }
            }
            Value::Object(out)
        })
        .collect();
    // Entry count is not volume: a coordinate tree is otherwise unbounded in both length and depth,
    // so 64 poses of a million coordinates each would walk straight through the cap above. Budgeted
    // across the WHOLE array (see `MAX_SHARE_POSE_SLOTS`) rather than per entry, so a large set of
    // plain body skeletons — which is what the shipped library is — still travels whole.
    let slots: usize = sanitized.iter().map(count_coordinate_slots).sum();
    if slots > MAX_SHARE_POSE_SLOTS {
        return None;
    }
    (!sanitized.is_empty()).then_some(Value::Array(sanitized))
}

/// True when `value` is made only of coordinate slots — numbers, `null`s standing in for a missing
/// one, and NON-EMPTY arrays of them. Anything with a string in it is not keypoints and does not
/// travel.
///
/// The emptiness rule is a closed hole rather than pedantry: `[]` is vacuously "all coordinates",
/// so 200,000 empty arrays under one `keypoints` key passed this check and cost nothing against the
/// slot budget, serializing to 600,027 bytes. An array with nothing in it is not a point, so it is
/// not a coordinate tree.
fn is_coordinate_tree(value: &Value) -> bool {
    match value {
        Value::Number(_) | Value::Null => true,
        Value::Array(values) => !values.is_empty() && values.iter().all(is_coordinate_tree),
        _ => false,
    }
}

/// How many coordinate slots a value's tree holds, at any depth. The unit
/// [`MAX_SHARE_POSE_SLOTS`] is spent in — a coordinate is one slot whether it arrived as
/// `[[x, y]]` or `[[[x, y]]]` (both forms are legal for `hands`), and a `null` is a slot too,
/// because it is a coordinate the worker's `normalize_points` fills in.
fn count_coordinate_slots(value: &Value) -> usize {
    match value {
        Value::Number(_) | Value::Null => 1,
        Value::Array(values) => values.iter().map(count_coordinate_slots).sum(),
        Value::Object(map) => map.values().map(count_coordinate_slots).sum(),
        _ => 0,
    }
}

fn sanitize_phases(value: &Value) -> Option<Value> {
    let phases = value.as_array()?;
    if over_cap(phases.len(), MAX_SHARE_PHASES) {
        return None;
    }
    let sanitized: Vec<Value> = phases
        .iter()
        .filter_map(Value::as_object)
        .map(|phase| {
            let mut out = JsonObject::new();
            for field in ["steps", "guidance"] {
                if let Some(number) = phase.get(field).filter(|value| value.is_number()) {
                    out.insert(field.to_owned(), number.clone());
                }
            }
            // Phase LoRA references are indices into THIS request's own `loras` list, so they
            // carry no id and stay meaningful next to the sanitized `loras` above — and they take
            // the same bound, for the same reason: a phase cannot reference more LoRAs than a job
            // is allowed to have.
            //
            // The key is written only when there is a schedule to write. An unconditional insert
            // turned an over-cap schedule into `"loras": []`, which is not an absence — it is the
            // positive claim "this phase applies no LoRAs", the exact plausible-and-unfalsifiable
            // record the drop doctrine above exists to refuse. `advanced_omissions` names it.
            if let Some(loras) = phase.get("loras").and_then(Value::as_array) {
                if !over_cap(loras.len(), MAX_SHARE_LORAS) {
                    let entries: Vec<Value> = loras
                        .iter()
                        .filter_map(Value::as_object)
                        .filter_map(|lora| {
                            let mut entry = JsonObject::new();
                            for field in ["index", "weight"] {
                                if let Some(number) =
                                    lora.get(field).filter(|value| value.is_number())
                                {
                                    entry.insert(field.to_owned(), number.clone());
                                }
                            }
                            entry.contains_key("index").then_some(Value::Object(entry))
                        })
                        .collect();
                    if !entries.is_empty() {
                        out.insert("loras".to_owned(), Value::Array(entries));
                    }
                }
            }
            Value::Object(out)
        })
        .collect();
    (!sanitized.is_empty()).then_some(Value::Array(sanitized))
}

// ---------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------

/// Parse an envelope out of a JSON string (a PNG text chunk, a dropped file).
///
/// # Errors
/// Returns a [`WorkflowShareError`] describing what is wrong, in a sentence fit to show a
/// user. Never panics and never silently defaults.
pub fn parse_workflow_share_json(text: &str) -> Result<WorkflowShare, WorkflowShareError> {
    let value: Value = serde_json::from_str(text)
        .map_err(|error| WorkflowShareError::InvalidJson(error.to_string()))?;
    parse_workflow_share(&value)
}

/// Parse an envelope out of already-decoded JSON.
///
/// The version gate runs BEFORE the body is deserialized, and it branches on `schemaVersion`
/// alone — `producer.version` is recorded, never interpreted. A file from a newer contract
/// gets [`WorkflowShareError::UnsupportedSchemaVersion`] naming both versions, so the user is
/// told to update rather than shown a field-level parse failure.
///
/// # Errors
/// See [`WorkflowShareError`].
pub fn parse_workflow_share(value: &Value) -> Result<WorkflowShare, WorkflowShareError> {
    let object = value.as_object().ok_or(WorkflowShareError::NotAnObject)?;

    let marker = object
        .get(WORKFLOW_SHARE_MARKER_KEY)
        .ok_or(WorkflowShareError::MissingMarker)?;
    let kind = marker
        .as_str()
        .ok_or_else(|| WorkflowShareError::Malformed {
            field: WORKFLOW_SHARE_MARKER_KEY.to_owned(),
            detail: "must be a workflow-kind string".to_owned(),
        })?;
    if !WORKFLOW_KINDS.contains(&kind) {
        return Err(WorkflowShareError::UnsupportedKind {
            found: kind.to_owned(),
            supported: WORKFLOW_KINDS.join(", "),
        });
    }

    // Present-but-wrong and absent are different problems with different fixes, so they get
    // different sentences: telling someone a field they can see in the file is "missing" sends
    // them looking in the wrong place. An explicit `null` counts as absent, which is what a
    // writer that omitted the field usually means.
    let schema_version = match object.get("schemaVersion") {
        None | Some(Value::Null) => return Err(WorkflowShareError::MissingSchemaVersion),
        Some(present) => present
            .as_u64()
            .and_then(|version| u32::try_from(version).ok())
            .ok_or_else(|| WorkflowShareError::Malformed {
                field: "schemaVersion".to_owned(),
                detail: "must be a whole number (the contract version the file was written with)"
                    .to_owned(),
            })?,
    };
    if schema_version > WORKFLOW_SHARE_SCHEMA_VERSION {
        return Err(WorkflowShareError::UnsupportedSchemaVersion {
            file: schema_version,
            supported: WORKFLOW_SHARE_SCHEMA_VERSION,
        });
    }

    let share: WorkflowShare =
        serde_json::from_value(value.clone()).map_err(|error| WorkflowShareError::Malformed {
            field: field_from_serde_error(&error),
            detail: error.to_string(),
        })?;
    // An envelope that arrived from outside is reduced on the way IN by the same function that
    // reduced ours on the way OUT — at VALUE granularity, not only key granularity. Dropping
    // the keys we do not declare says nothing about the strings under the keys we do.
    let reduced = reduce_workflow_share(share);
    // Last, on the reduced envelope, and refusing the WHOLE thing rather than shedding a field:
    // what gets persisted is this value, and an envelope over the ceiling is one whose recipe we
    // cannot record without also recording the payload attached to it. See
    // `WORKFLOW_SHARE_MAX_BYTES` — this is the bound the per-field caps could not compose into.
    within_recording_ceiling(&reduced)?;
    Ok(reduced)
}

/// Best-effort field name out of a serde error, so the message names something the user can
/// look for rather than only a byte offset.
fn field_from_serde_error(error: &serde_json::Error) -> String {
    let text = error.to_string();
    text.split('`')
        .nth(1)
        .map_or_else(|| "workflow".to_owned(), str::to_owned)
}
