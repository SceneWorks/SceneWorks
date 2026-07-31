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
//! parses each builder named in [`ADVANCED_BUILDERS`] and fails the build when a key is neither
//! allow-listed nor explicitly denied — a new knob can neither silently leak nor silently vanish.
//!
//! sc-15946 shipped that lint against ONE builder on the premise that it was the only one. It was
//! not: sc-15948 found a second (`buildDetailJobBody`, whose `cnScale` was being dropped) and then
//! two more (the character lane's `angleSet`, the interleave lane's `systemMessage` /
//! `imageGuidanceScale`). A lint that is bolted on per discovery closes no class, so
//! [`ADVANCED_BUILDERS`] is a registry the lint ENUMERATES, and the same test walks
//! `apps/web/src` for anything that builds an `advanced` map and refuses to pass while one is in
//! neither the registry nor [`DEFERRED_ADVANCED_BUILDERS`].
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

use serde::{Deserialize, Serialize};
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

/// A LoRA the run applied, reduced to what another install can act on: the display name, the
/// weight, and the Hugging Face repo id when the catalog entry resolved to one. No local id,
/// no installed path, no source path.
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
    /// **Video lane**: the quality preset the run was submitted at (`draft` / `standard` / …).
    /// A named tier off a menu, not a hardware budget — the receiving install has the same menu.
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
/// `loras` carries `skip_serializing_if = "Vec::is_empty"`, so a 6-LoRA envelope whose LoRAs were
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
/// Every variant except [`Self::Server`] names exactly one entry in [`ADVANCED_BUILDERS`], and
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
    /// Stamped onto `advanced` by the API after the request arrives (recipe-preset resolution
    /// in `apps/rust-api/src/generation.rs`). Classified here for the record; the lint does
    /// not expect to find them in the JS, and no [`ADVANCED_BUILDERS`] entry claims it.
    Server,
}

impl AdvancedKeySource {
    /// Every variant, so the lint can assert the registry covers them all. A new variant that
    /// nobody registered fails [`ADVANCED_BUILDERS`]'s round-trip test.
    pub const ALL: &'static [Self] = &[
        Self::StudioBuilder,
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
        Self::Server,
    ];
}

/// The JS shape a registered builder writes its `advanced` map in, and therefore which extractor
/// in `crates/sceneworks-core/tests/workflow_share.rs` can read it.
///
/// Named in the registry rather than sniffed, because a wrong guess is the failure mode this whole
/// lint exists to prevent: an extractor that quietly understands nothing returns an empty key set
/// and passes. Each extractor panics with instructions when the file stops matching its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedBuilderShape {
    /// `return { key, ...(cond ? { key } : {}) }` — the conditional-spread object literal
    /// `buildImageJobAdvanced` is built from. Needs the brace-aware scanner.
    ReturnedObject,
    /// `advanced: { a, b }` — a flat object literal in a returned payload, no spreads, no
    /// nesting (`buildDetailJobBody`).
    FlatAdvancedLiteral,
    /// `const advanced = { … }` followed by `advanced.key = …` assignments. Both halves are
    /// emits, and a spread in the initializer must be declared in
    /// [`AdvancedBuilder::spread_of`].
    AssignedObject,
    /// `advancedExtras: { … }` — a controller's contribution, merged into a
    /// [`Self::AssignedObject`] builder's initializer by spread.
    ExtrasLiteral,
    /// The builder posts NO `advanced` map at all (`buildUpscaleJobBody`). Registered anyway,
    /// because it feeds an embedding lane: the lint asserts the function stays empty of
    /// `advanced`, so the day someone adds a knob there it has to be classified.
    NoAdvancedMap,
    /// `advanced: { key, ...(cond ? { key } : {}) }` — an `advanced:` literal WITH conditional
    /// spreads, in a payload passed to a call rather than returned (sc-15956).
    ///
    /// The video builders' shape, and the one the four existing arms between them could not read:
    /// [`Self::FlatAdvancedLiteral`] asserts the literal has no spread and no nesting, and
    /// [`Self::ReturnedObject`] looks for a `return {`. Uses the same brace-aware
    /// `scan_object_literal` [`Self::ReturnedObject`] does, and honours
    /// [`AdvancedBuilder::spread_of`] the way [`Self::AssignedObject`] does — so a spread of
    /// somebody else's map has to name the builder that accounts for it.
    SpreadAdvancedLiteral,
}

/// One JS builder whose `advanced` keys reach an embedding lane.
///
/// The registry the coverage lint enumerates. `path` + `function` are read out of the repo, so a
/// rename fails loudly rather than silently pointing the lint at nothing.
#[derive(Debug, Clone, Copy)]
pub struct AdvancedBuilder {
    /// The tag rules use to name this builder. Unique across the registry.
    pub source: AdvancedKeySource,
    /// Repo-relative path of the file that defines it.
    pub path: &'static str,
    /// The JS function whose body the extractor reads.
    pub function: &'static str,
    pub shape: AdvancedBuilderShape,
    /// The embedding lane this builder's payload ends up written by, so a reader can see why the
    /// keys matter. Prose, asserted only to be non-trivial.
    pub lane: &'static str,
    /// Keys the extractor MUST still find. A floor against an extractor that has quietly stopped
    /// understanding the file — the failure mode that makes a green lint worthless.
    pub anchors: &'static [&'static str],
    /// The smallest number of keys the extractor may report before the lint calls itself broken.
    /// Set well under the real count; it is a floor, not a census.
    pub minimum_keys: usize,
    /// Identifiers spread into an [`AdvancedBuilderShape::AssignedObject`] initializer, whose keys
    /// are accounted for by another registry entry. Declared so a NEW spread of something nobody
    /// classified fails the lint instead of vanishing into it.
    pub spread_of: &'static [&'static str],
}

/// Every builder whose `advanced` keys travel inside a shared image (sc-15948).
///
/// The lanes, as of sc-15948: `POST /api/v1/image/jobs` (`text_to_image` / `edit_image` /
/// `character_image`), `POST /api/v1/image/detail/jobs`, `POST /api/v1/image/interleave/jobs`, and
/// the standalone `image_upscale` job. All four write their PNG through a seam that embeds.
///
/// **Adding a builder here is what turns the lint on for it.** Adding a builder to the WEB and not
/// to this table fails `every_advanced_builder_in_the_web_app_is_accounted_for`, which walks
/// `apps/web/src` and refuses any `advanced`-map producer that is in neither this registry nor
/// [`DEFERRED_ADVANCED_BUILDERS`].
pub const ADVANCED_BUILDERS: &[AdvancedBuilder] = &[
    AdvancedBuilder {
        source: AdvancedKeySource::StudioBuilder,
        path: "apps/web/src/imageJobAdvanced.js",
        function: "buildImageJobAdvanced",
        shape: AdvancedBuilderShape::ReturnedObject,
        lane:
            "POST /api/v1/image/jobs — text_to_image, edit_image and character_image, written by \
               `write_image_asset`",
        anchors: &["resolution", "sampler", "steps", "styleId", "poses"],
        minimum_keys: 25,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::EditBuilder,
        path: "apps/web/src/imageJobs.js",
        function: "buildEditJobBody",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/image/jobs — edit_image from the Image Editor, written by \
               `write_image_asset`",
        anchors: &["guidanceScale"],
        minimum_keys: 1,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::DetailBuilder,
        path: "apps/web/src/imageJobs.js",
        function: "buildDetailJobBody",
        shape: AdvancedBuilderShape::FlatAdvancedLiteral,
        lane: "POST /api/v1/image/detail/jobs, written by `image_jobs/detail.rs`",
        anchors: &["strength", "cnScale"],
        minimum_keys: 2,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::CharacterBuilder,
        path: "apps/web/src/components/CharacterAdvancedOptions.jsx",
        function: "buildAdvanced",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/image/jobs — character_image, written by `write_image_asset`",
        anchors: &["ipAdapterScale", "steps", "usePid"],
        // The angle-set / pose-library controllers' `advancedExtras`, each registered below.
        minimum_keys: 9,
        spread_of: &["base"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::CharacterAngleExtras,
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "useAngleController",
        shape: AdvancedBuilderShape::ExtrasLiteral,
        lane: "POST /api/v1/image/jobs — character_image (angle set), written by \
               `write_image_asset`",
        anchors: &["angleSet"],
        minimum_keys: 2,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::CharacterPoseExtras,
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "usePoseController",
        shape: AdvancedBuilderShape::ExtrasLiteral,
        lane: "POST /api/v1/image/jobs — character_image (pose library), written by \
               `write_image_asset`",
        anchors: &["poses", "faceRestore"],
        minimum_keys: 3,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::InterleaveBuilder,
        path: "apps/web/src/screens/DocumentStudio.jsx",
        function: "submit",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/image/interleave/jobs, written by `sensenova_jobs.rs` through \
               `write_image_asset`",
        anchors: &["systemMessage", "imageGuidanceScale"],
        minimum_keys: 2,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::UpscaleBuilder,
        path: "apps/web/src/imageJobs.js",
        function: "buildUpscaleJobBody",
        shape: AdvancedBuilderShape::NoAdvancedMap,
        lane: "POST /api/v1/jobs type=image_upscale, written by `single_child_asset.rs`",
        anchors: &[],
        minimum_keys: 0,
        spread_of: &[],
    },
    // -----------------------------------------------------------------------
    // The VIDEO builders (sc-15956), promoted out of `DEFERRED_ADVANCED_BUILDERS`
    // -----------------------------------------------------------------------
    //
    // All six moved in the SAME change as the video write seams below, because the back-check that
    // every entry here is named by an embedding seam makes the two halves inseparable — see
    // `WORKFLOW_WRITE_SEAMS`.
    AdvancedBuilder {
        source: AdvancedKeySource::VideoStudioBuilder,
        path: "apps/web/src/screens/VideoStudio.jsx",
        function: "submit",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — the Video Studio, written by \
               `video_jobs/mod.rs::encode_media`",
        anchors: &[
            "motion",
            "selectedPersonTrack",
            "replacementModeLabel",
            "lightning",
            "videoConditioningStrength",
        ],
        minimum_keys: 20,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::TimelineBaseBuilder,
        path: "apps/web/src/components/editor/useEditorGeneration.js",
        function: "buildBasePayload",
        shape: AdvancedBuilderShape::AssignedObject,
        lane: "POST /api/v1/jobs type=video_generate — the timeline editor's shared video payload \
               (queueTimelineVideoJob), written by `video_jobs/mod.rs::encode_media`",
        anchors: &["resolution", "motion", "lightning"],
        minimum_keys: 6,
        spread_of: &[],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::TimelineExtendBuilder,
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "extendSelectedClip",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — timeline extend, written by \
               `video_jobs/mod.rs::encode_media`",
        anchors: &["timelineAction", "timelineContext"],
        minimum_keys: 2,
        spread_of: &["base.advanced"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::TimelineReplaceBuilder,
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "replaceSelectedItem",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — timeline replace, written by \
               `video_jobs/mod.rs::encode_media`",
        anchors: &["timelineAction", "timelineContext"],
        minimum_keys: 2,
        spread_of: &["base.advanced"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::TimelineBridgeBuilder,
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "bridgeGap",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — timeline bridge, written by \
               `video_jobs/mod.rs::encode_media`",
        anchors: &["timelineAction", "timelineContext"],
        minimum_keys: 2,
        spread_of: &["base.advanced"],
    },
    AdvancedBuilder {
        source: AdvancedKeySource::SimpleVideoBuilder,
        path: "apps/web/src/simple/simpleJobs.js",
        function: "buildSimpleVideoRequest",
        shape: AdvancedBuilderShape::SpreadAdvancedLiteral,
        lane: "POST /api/v1/jobs type=video_generate — the Simple shell's video request, written \
               by `video_jobs/mod.rs::encode_media`",
        anchors: &["resolution"],
        minimum_keys: 1,
        spread_of: &[],
    },
];

/// A builder that produces an `advanced` map the lint has SEEN and deliberately does not
/// classify, because its lane does not embed yet.
///
/// This list exists so "unaccounted for" and "accounted for as out of scope" are different
/// states. Making one of these lanes embed means moving its entry into [`ADVANCED_BUILDERS`] —
/// at which point every key it emits must be classified, which is the point.
#[derive(Debug, Clone, Copy)]
pub struct DeferredAdvancedBuilder {
    pub path: &'static str,
    pub function: &'static str,
    /// Why it is not classified. Two categories, and the lint accepts nothing else, so a deferral
    /// cannot be a shrug:
    ///
    /// * **Awaiting classification** — names the `sc-` story that owns turning its lane on.
    /// * **Permanently exempt** — starts with [`PERMANENT_EXEMPTION`] and says why no story will
    ///   ever own it, plus what would have to change for that to stop being true.
    pub reason: &'static str,
}

/// The prefix that marks a [`DeferredAdvancedBuilder::reason`] as a PERMANENT exemption rather
/// than a story-owned deferral.
///
/// A deferral normally names the story that will classify its keys. Some `advanced` maps are not
/// waiting on anybody: they are a different namespace on a lane that embeds nothing, so there is no
/// story to name and inventing a fake id would be worse than saying so. The lint requires a reason
/// with this prefix to state what would have to change (an embedding write seam) and to name
/// [`ADVANCED_BUILDERS`] as where the entry moves if it ever does.
pub const PERMANENT_EXEMPTION: &str = "PERMANENT EXEMPTION:";

/// One permanent exemption for an `advanced` map that is not a generation payload at all.
///
/// **The six video builders that used to live here were promoted by sc-15956** — the story this
/// list was built for — and their 17 unclassified keys are now rules in [`ADVANCED_KEY_RULES`].
/// The mechanism worked exactly as designed: making a video seam embed was impossible until every
/// one of those keys had an `allow`/`deny` decision, and two of them (`selectedPersonTrack`,
/// `timelineContext`) turned out to be object-shaped id-and-free-text carriers that would have
/// travelled silently under the pre-sc-16113 arrangement.
///
/// Note the count. This list's own comment said "~15 keys"; the real number was 17, because two
/// (`timelineAction`, `timelineContext`) are emitted only by the three `EditorScreen.jsx` actions
/// and no one had read those builders. A deferral list records that a decision is owed, not what
/// the decision costs — and that gap is the argument for the lint being a discovery scan rather
/// than a hand-maintained tally.
///
/// # What enforces that, and what does not (sc-16113)
///
/// This list used to claim, in this comment, that "nothing about a video lane can start embedding
/// while its keys sit in this list". Nothing did. A reviewer added a `write_workflow_chunk` +
/// `embeddable_workflow_share` call to a real video-lane PNG write in
/// `crates/sceneworks-worker/src/video_jobs/seedvr2.rs` and every lint stayed green, because the
/// sweep over `apps/web/src` accepts membership in EITHER registry and the write-seam lint read
/// four hard-coded worker files.
///
/// What is enforced now, by `every_worker_write_seam_declares_the_lane_it_embeds_for` in
/// `crates/sceneworks-core/tests/workflow_share.rs`:
///
/// * every function in `crates/sceneworks-worker/src` that the scan can see an envelope reach —
///   it calls a name on the core write surface, or names a `WorkflowShare` (or a worker struct
///   holding one) in its own signature, or calls a worker function that does — must have a
///   [`WORKFLOW_WRITE_SEAMS`] entry, and an undeclared one fails the build. The seams are found by
///   walking the crate, not by a file list, and `use … as …` renames of those names are resolved
///   before the walk;
/// * a seam declared [`SeamDisposition::Embeds`] must name the web builders that feed it, and a
///   named builder that is in THIS list fails the build, naming the seam, the lane and the builder;
/// * a seam cannot dodge that by declaring [`SeamDisposition::Declines`]. A decline is checked
///   POSITIVELY against the source: it may not call `write_workflow_chunk` with anything but a
///   literal `None`, may not name `WorkflowShare` in its body at all, may not accept one through
///   its signature, and must fill every share-carrying struct field with `None` — by initializer
///   OR by later assignment. Where the envelope came from does not matter, so parsing one back out
///   of another file and writing it on is an embed like any other.
///
/// So making a video lane embed now requires EITHER moving its builder into [`ADVANCED_BUILDERS`],
/// which is what forces every key it emits to be classified, or writing a builder list into
/// [`WORKFLOW_WRITE_SEAMS`] that is not true. The lint cannot tell those two apart — it has no way
/// to know which web builder feeds a Rust seam — so what it guarantees is that the statement had
/// to be written at all, and that the honest version of it fails the build. Review is what catches
/// the dishonest one.
pub const DEFERRED_ADVANCED_BUILDERS: &[DeferredAdvancedBuilder] = &[DeferredAdvancedBuilder {
    path: "apps/web/src/training/trainingConfig.js",
    function: "trainingConfigSnapshot",
    reason: "PERMANENT EXEMPTION: the Training Studio's `advanced` is a DIFFERENT namespace \
                 from the image payload's — trainer hyperparameters (networkType, lrScheduler, \
                 sampleSteps, decomposeFactor) on a training job, not generation intent for an \
                 image. No training write seam EMBEDS a workflow, nothing sanitizes these keys \
                 through `ADVANCED_KEY_RULES`, and no story is waiting to classify them: this is \
                 not a deferral with an owner, it is deliberately outside the allow-list. What \
                 would have to change is a training write seam that embeds. Then, and only then, \
                 this entry moves into `ADVANCED_BUILDERS` and every key it emits needs an \
                 allow/deny decision.",
}];

/// A (file, function) reference to one entry in [`ADVANCED_BUILDERS`] or
/// [`DEFERRED_ADVANCED_BUILDERS`] (sc-16113).
///
/// The same key those two registries are already joined on by
/// `every_advanced_builder_in_the_web_app_is_accounted_for`, so a seam names a builder exactly the
/// way the sweep does. A reference that resolves to neither registry fails the lint; a reference
/// that resolves to the DEFERRED one fails it too, and that is the whole point of the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebBuilderRef {
    /// Repo-relative path of the JS/JSX file that defines the builder.
    pub path: &'static str,
    /// The builder function's name.
    pub function: &'static str,
}

/// What one worker seam does about the workflow chunk (sc-16113).
///
/// Four states, not two, because "writes no chunk", "writes whatever its caller decided" and
/// "an envelope reaches it but it writes no file" are three different claims and only one of them
/// ends the lane question. Each is checked against the seam's own source, so the disposition is a
/// declaration the lint verifies rather than one it trusts.
#[derive(Debug, Clone, Copy)]
pub enum SeamDisposition {
    /// An envelope leaves this function into a file or into a share-carrying field. Must name at
    /// least one builder, and every one of them must be in [`ADVANCED_BUILDERS`] — a reference to
    /// [`DEFERRED_ADVANCED_BUILDERS`] is exactly the state this registry exists to refuse.
    ///
    /// Where the envelope came from is not part of the test: building one, parsing one back out of
    /// another file, or cloning one held elsewhere are all embedding, because what reaches the
    /// written file is the same either way.
    Embeds(&'static [WebBuilderRef]),
    /// Writes an envelope its CALLER built, and obtains none of its own. The lane question belongs
    /// to the callers, and they are seams in their own right: a function that hands a
    /// `WorkflowShare` to this one names it, which is what the discovery scan reads. The reason
    /// says why the decision is not made here.
    ///
    /// Verified: it must accept a share through its signature, must reach the write surface (a
    /// function that reaches no writer is [`SeamDisposition::Inert`], not a conduit), and must not
    /// build, parse or name a `WorkflowShare` of its own — sourcing one is deciding a lane.
    Conduit(&'static str),
    /// Writes the file with no envelope at all, deliberately. The reason must say why that asset
    /// has no generation recipe to record.
    ///
    /// Verified positively, not by the absence of a builder call: it may not call
    /// `write_workflow_chunk` with anything but a literal `None`, may not name `WorkflowShare` in
    /// its body, may not accept one through its signature, and must fill every share-carrying
    /// struct field with `None` — by initializer or by later assignment. So declining is not a way
    /// to embed without classifying a lane, whatever the envelope's provenance.
    Declines(&'static str),
    /// A `WorkflowShare` reaches this function's signature and goes nowhere: it writes no file,
    /// builds no envelope and calls nothing on the write surface. A logging or validation helper
    /// that takes a spec is this, and calling it a [`SeamDisposition::Conduit`] would be a false
    /// statement — a conduit writes.
    ///
    /// Verified: it must carry a share, and it must call nothing the scan can see reach a write.
    /// The moment it calls a writer it stops being inert and owes a lane.
    Inert(&'static str),
}

/// One place in `sceneworks-worker` where a [`WorkflowShare`] can reach a written file (sc-16113).
///
/// The registry that ties "this lane has an embedding write seam" to "its builder is in
/// [`ADVANCED_BUILDERS`]". `path` + `function` are read out of the repo, so a rename or a move
/// fails loudly, and the seams themselves are DISCOVERED by scanning the worker crate rather than
/// listed here — an entry nobody can find is as much a failure as a seam nobody declared.
#[derive(Debug, Clone, Copy)]
pub struct WorkflowWriteSeam {
    /// Repo-relative path of the file that contains the seam.
    pub path: &'static str,
    /// The Rust function whose body touches the envelope.
    pub function: &'static str,
    /// Prose: which product lane's files this seam writes, so a reader can see why the builders
    /// below are the ones that feed it. Asserted only to be non-trivial.
    pub lane: &'static str,
    /// What it does about the chunk, checked against the source.
    pub disposition: SeamDisposition,
}

/// Every worker seam a [`WorkflowShare`] can pass through, and the web builders behind each one.
///
/// **The gate this whole epic's deferral list was supposed to be.** Adding an embedding call to a
/// lane whose builder sits in [`DEFERRED_ADVANCED_BUILDERS`] now fails the build twice over: once
/// because the new seam has no entry here, and again if the entry names the deferred builder.
///
/// Kept honest in both directions by `every_worker_write_seam_declares_the_lane_it_embeds_for`: a
/// discovered seam with no entry fails, and an entry whose function no longer touches an envelope
/// fails too.
///
/// # Ordering, for whoever wires a new lane (sc-15956)
///
/// The back-check that every [`ADVANCED_BUILDERS`] entry is named by some embedding seam means the
/// two halves of a new lane cannot be split across two PRs: classifying the ~15 video keys on its
/// own would leave `VideoStudio.jsx::submit` registered with no seam behind it, which fails. Move
/// the builder up and add the seam entry in the SAME change. The failure is loud either way, but
/// it is worth knowing before the split is attempted.
///
/// Likewise `a_seam_that_embeds_for_a_deferred_builder_fails_the_build` hard-codes
/// `VideoStudio.jsx::submit` as its deferred example, so promoting that builder turns that
/// mutation proof into a "did not panic" failure. Re-point it at whatever is still deferred then;
/// the proof is that SOME deferred builder is refused, not that one particular one is.
pub const WORKFLOW_WRITE_SEAMS: &[WorkflowWriteSeam] = &[
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "write_image_asset",
        lane: "POST /api/v1/image/jobs (text_to_image / edit_image / character_image) and \
               POST /api/v1/image/interleave/jobs — the one funnel every generated image is \
               written through",
        disposition: SeamDisposition::Embeds(IMAGE_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "upscaled_workflow_share",
        lane: "POST /api/v1/image/jobs — the inline-upscale sub-step of a generate job, whose \
               envelope is the generation's payload with the APPLIED pass overlaid",
        disposition: SeamDisposition::Embeds(GENERATE_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "write_upscaled_asset",
        lane: "POST /api/v1/image/jobs — the inline-upscaled variant's own PNG",
        disposition: SeamDisposition::Embeds(GENERATE_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "detail_workflow_share",
        lane: "POST /api/v1/image/detail/jobs — the standalone detail pass's own envelope",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildDetailJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs.rs",
        function: "standalone_upscale_workflow_share",
        lane: "POST /api/v1/jobs type=image_upscale — the standalone upscale job's own envelope",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildUpscaleJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/image_jobs/detail.rs",
        function: "run_image_detail_job",
        lane: "POST /api/v1/image/detail/jobs — writes the refined PNG (macOS-gated; the scan is \
               textual, so this seam is read on every platform)",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildDetailJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/upscale_jobs.rs",
        function: "run_image_upscale_job",
        lane: "POST /api/v1/jobs type=image_upscale — hands the standalone upscale's envelope to \
               the shared single-child writer",
        disposition: SeamDisposition::Embeds(&[WebBuilderRef {
            path: "apps/web/src/imageJobs.js",
            function: "buildUpscaleJobBody",
        }]),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/single_child_asset.rs",
        function: "write_single_child_asset",
        lane: "The shared one-PNG-child writer: the standalone upscale and the smart-select mask \
               both come through here",
        disposition: SeamDisposition::Conduit(
            "It writes the `SingleChildAssetSpec::workflow` its caller decided on and builds no \
             envelope of its own, so the lane question belongs to the callers — and every caller \
             is a seam here in its own right, because handing a `WorkflowShare` in is what the \
             discovery scan reads.",
        ),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/video_jobs/mod.rs",
        function: "video_workflow_metadata",
        lane: "POST /api/v1/jobs type=video_generate — the ONE funnel every generated clip is \
               encoded through (`encode_media`), whatever engine, route or studio produced it",
        disposition: SeamDisposition::Embeds(VIDEO_JOB_BUILDERS),
    },
    WorkflowWriteSeam {
        path: "crates/sceneworks-worker/src/segment_jobs.rs",
        function: "run_image_segment_job",
        lane: "POST /api/v1/jobs type=image_segment — the smart-select mask",
        disposition: SeamDisposition::Declines(
            "A smart-select mask is not a generated image: it is a binary selection derived from a \
             box prompt and a concept string, with no generation recipe to replay. It is also \
             grayscale, and the chunk writer encodes RGB8, so embedding would triple its size and \
             change its colour type.",
        ),
    },
];

/// Every builder behind a generated clip (sc-15956).
///
/// All six feed ONE seam, because all six post `type=video_generate` and every video job funnels
/// through `encode_media` — the same property that let sc-12371 measure clip length in one place.
const VIDEO_JOB_BUILDERS: &[WebBuilderRef] = &[
    WebBuilderRef {
        path: "apps/web/src/screens/VideoStudio.jsx",
        function: "submit",
    },
    WebBuilderRef {
        path: "apps/web/src/components/editor/useEditorGeneration.js",
        function: "buildBasePayload",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "extendSelectedClip",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "replaceSelectedItem",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "bridgeGap",
    },
    WebBuilderRef {
        path: "apps/web/src/simple/simpleJobs.js",
        function: "buildSimpleVideoRequest",
    },
];

/// The builders behind every image the generate lane writes (`POST /api/v1/image/jobs`).
///
/// Named once because three seams share it: the base render, the inline-upscale envelope, and the
/// upscaled variant's own PNG.
const GENERATE_JOB_BUILDERS: &[WebBuilderRef] = &[
    WebBuilderRef {
        path: "apps/web/src/imageJobAdvanced.js",
        function: "buildImageJobAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/imageJobs.js",
        function: "buildEditJobBody",
    },
    WebBuilderRef {
        path: "apps/web/src/components/CharacterAdvancedOptions.jsx",
        function: "buildAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "useAngleController",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "usePoseController",
    },
];

/// [`GENERATE_JOB_BUILDERS`] plus the interleave lane, which reaches `write_image_asset` through
/// `sensenova_jobs.rs` threading its `workflow_source` into the same `ImagePlan`.
///
/// The interleave lane has no inline-upscale sub-step, so it is not in [`GENERATE_JOB_BUILDERS`].
const IMAGE_JOB_BUILDERS: &[WebBuilderRef] = &[
    WebBuilderRef {
        path: "apps/web/src/imageJobAdvanced.js",
        function: "buildImageJobAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/imageJobs.js",
        function: "buildEditJobBody",
    },
    WebBuilderRef {
        path: "apps/web/src/components/CharacterAdvancedOptions.jsx",
        function: "buildAdvanced",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "useAngleController",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/characterPanels.jsx",
        function: "usePoseController",
    },
    WebBuilderRef {
        path: "apps/web/src/screens/DocumentStudio.jsx",
        function: "submit",
    },
];

/// How an allowed value is reduced. Anything that does not match its shape is dropped rather
/// than passed through — that is what stops an object (and any path inside it) from being
/// smuggled under an allow-listed scalar key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedShape {
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
    /// The Krea multi-phase denoise list, reduced to `{ steps, guidance, loras:[{index, weight}] }`.
    Phases,
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
/// in [`ADVANCED_BUILDERS`] and fails when a key one of them can emit is missing here — so a new
/// knob is a build-time decision, not a silent leak and not a silent loss.
///
/// The rule of thumb when classifying a new key: does it describe **what to make** (travels)
/// or **what this machine can afford to make it with** (does not)?
pub const ADVANCED_KEY_RULES: &[AdvancedKeyRule] = &[
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
        "usePid",
        AdvancedShape::Scalar,
        "Authored decoder choice: PiD changes the produced image, and is the output's \
         non-commercial marker. Unlike a quant tier it is not a memory accommodation.",
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
        "textEncoderModel",
        AdvancedShape::Scalar,
        AdvancedKeySource::VideoStudioBuilder,
        "Authored text-encoder pick. It changes what the model SEES of the prompt, so a replay \
         without it is a different run. A catalog-global model slug, not an install-local id — the \
         same class as `styleId` and unlike `controlWeights`, which carries a resolved path.",
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
fn shareable_prose(value: &str) -> String {
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

/// True for a character that takes no width of its own but changes how the text AROUND it reads:
/// Unicode's `Cf` (format) class plus the `Zl`/`Zp` line and paragraph separators.
///
/// [`char::is_control`] is `Cc` and only `Cc`, which is why [`shareable_prose`] cannot stop at it.
/// `Cc` does catch ANSI escapes (ESC is `Cc`), so this is not terminal injection — it is display
/// spoofing, and in a feature whose entire pitch is "trust this recipe a stranger sent you" that
/// is the more relevant one. U+202E RIGHT-TO-LEFT OVERRIDE makes a rendered prompt say something
/// other than what is stored; U+200B and U+FEFF are the same trick without the reversal, splitting
/// a word the reader believes is whole. Both survived into a recorded envelope before this guard.
///
/// The ranges are the whole `Cf` class (Unicode 16.0 — 16.0 added no `Cf`, so the list is exact for
/// 15.1 as well) rather than the handful demonstrated,
/// because a class is what closes a class. Enumerated instead of taken from a crate because `std`
/// exposes no general-category API and pulling a Unicode table in for one predicate is a poor
/// trade: `Cf` additions are rare, additive, and — being new invisible formatting characters — are
/// new display tricks, not new prompt content.
///
/// Variation selectors (U+FE00..U+FE0F) are `Mn`, not `Cf`, and are deliberately NOT here: emoji
/// presentation is prompt content. The tag characters at U+E0020..U+E007F are `Cf` and do go, which
/// costs subdivision-flag emoji (🏴󠁧󠁢󠁳󠁣󠁴󠁿) in a prompt — the right side of that trade, since the same
/// range is the standard way to smuggle hidden text through a display.
///
/// FOLLOW-UP (not this story): `Cf` is not the whole display-spoof class. U+3164 HANGUL FILLER,
/// U+115F / U+1160 (the jamo fillers), U+2800 BRAILLE PATTERN BLANK and U+FFA0 HALFWIDTH HANGUL
/// FILLER all render blank and are `Lo` / `So`, so they survive this guard. Same trick, different
/// general category — a separate decision from "close the `Cf` class", because `Lo` is where real
/// prompt content lives and a range list there needs its own justification.
fn is_invisible_formatting(character: char) -> bool {
    matches!(
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
    let trimmed = value.trim();
    if trimmed.chars().count() > LABEL_MAX_CHARS {
        return None;
    }
    let mut segments = trimmed.split('/');
    let (Some(owner), Some(name), None) = (segments.next(), segments.next(), segments.next())
    else {
        return None;
    };
    let segment_ok = |segment: &str| {
        segment
            .chars()
            .next()
            .is_some_and(|first| first.is_ascii_alphanumeric())
            && segment.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
            })
    };
    (segment_ok(owner) && segment_ok(name)).then(|| trimmed.to_owned())
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
            seed: asset.recipe.seed,
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
    let seed = Some(facts.seed);

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
    reduce_workflow_share(WorkflowShare {
        kind: kind.to_owned(),
        schema_version: WORKFLOW_SHARE_SCHEMA_VERSION,
        producer: WorkflowProducer::default(),
        mode,
        model,
        prompt,
        negative_prompt,
        seed,
        width: u32_field("width").or(facts.width),
        height: u32_field("height").or(facts.height),
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
        inputs: describe_inputs(job_payload),
        advanced,
        // Nothing to carry in: the marker is derived by `reduce_workflow_share` from what the
        // sanitizer actually dropped, in this direction exactly as in the other.
        omitted: Vec::new(),
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
fn describe_inputs(job_payload: &JsonObject) -> Vec<WorkflowInput> {
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
    let sources = usize::from(has_id("sourceAssetId")) + usize::from(has_id("lastFrameAssetId"));
    if sources > 0 {
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_SOURCE.to_owned(),
            count: u32::try_from(sources).unwrap_or(u32::MAX),
            control_mode: None,
        });
    }
    let references = usize::from(has_id("referenceAssetId")) + id_list_len("referenceAssetIds");
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
/// normalizer's own cap ("Recipe presets can include at most 5 LoRAs"). A run that applied more
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
/// The one collection with NO upstream validator to inherit, and the review that added it said so:
/// the pose lane renders one image per pose and nothing in the worker's `pose_entries` or in
/// `apps/web/src` clamps how many a user may select. So the derivation is the library they are
/// selected FROM — `apps/web/public/poses/index.json` ships 46 poses — and 64 clears an
/// all-of-the-library selection with room for user-created Key Point Library entries on top.
///
/// Because there is no upstream ceiling, this cap can fire on OUR OWN WRITE SIDE: a user who
/// selects 65 poses gets an envelope with no `advanced.poses` in it. That is why
/// [`OMITTED_FIELDS`] exists and is emitted on the build side too — a 70-pose selection that is
/// not recorded says so, instead of arriving as an absence indistinguishable from "no poses".
/// Clamping the selection at the source is the better fix and is not this story's to make.
///
/// This bounds the ENTRY count only; [`MAX_SHARE_POSE_SLOTS`] is what bounds the volume.
const MAX_SHARE_POSES: usize = 64;

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
/// | `loras` | 8,250 | [`MAX_SHARE_LORAS`] x (name + repo at [`LABEL_MAX_CHARS`] x 4 + weight) |
/// | labels + `upscale` + `producer` + scalars | ~15,000 | [`LABEL_MAX_CHARS`] x 4, per field |
/// | | **~113,900** | |
///
/// 160 KiB leaves 44% headroom over that, which the measured real cases make look enormous: the
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
    };
    (reduced.name.is_some() || reduced.weight.is_some() || reduced.repo.is_some())
        .then_some(reduced)
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
            AdvancedShape::Poses => sanitize_poses(value),
            AdvancedShape::Phases => sanitize_phases(value),
        };
        if let Some(sanitized) = sanitized {
            out.insert(key.clone(), sanitized);
        }
    }
    out
}

/// The prose keys inside a sanitized value that are what the user typed, and so are exempt
/// from the path-shape guard (see [`is_path_shaped`]).
///
/// Exempt from the PATH check only. They are still bounded and control-stripped by
/// [`shareable_prose`], because on the parse side they are a stranger's strings.
const PROSE_KEYS: &[&str] = &[
    "stylePrompt",
    "intent",
    "runtimePrompt",
    // The interleave system prompt (sc-15948). Prose rather than a label because the user typed
    // it into a textarea and it is the literal instruction the model received; bounding it as a
    // slug would silently drop a system message that happened to mention a directory, which is
    // the exact silent-loss failure this epic's lint exists to prevent.
    "systemMessage",
];

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The keys of a JSON object, sorted, so an assertion about WHICH keys survived the
    /// allow-list cannot accidentally assert the order they came out in.
    ///
    /// `serde_json::Map` is a `BTreeMap` (sorted iteration) with the `preserve_order` feature
    /// off and an `IndexMap` (insertion order) with it on — and Cargo unifies features across
    /// every crate being built, so the SAME map iterates differently under `cargo test -p
    /// sceneworks-core` than under the workspace build the `parity` job runs (`sceneworks-worker`
    /// → `sceneworks-gen-core` → `core-llm` turns `preserve_order` on). An assertion written as an
    /// ordered `Vec` therefore passes locally and fails in CI while the sanitizer is correct in
    /// both. Sorting here makes these assertions say what they mean — the key SET — under either
    /// configuration.
    fn sorted_keys(object: &JsonObject) -> Vec<&str> {
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    fn asset_fixture() -> Asset {
        serde_json::from_value(json!({
            "schemaVersion": 1,
            "id": "asset_123",
            "projectId": "project_abc",
            "generationSetId": "genset_xyz",
            "type": "image",
            "displayName": "Mist over hills #1",
            "createdAt": "2026-07-29T13:00:00Z",
            "file": {
                "path": "assets/images/2026-07-29_z_image_turbo_mist_0001.png",
                "mimeType": "image/png",
                "width": 1024,
                "height": 1024,
                "duration": null,
                "fps": null
            },
            "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
            "recipe": {
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "mist over hills",
                "negativePrompt": "blurry",
                "seed": 4242,
                "loras": [],
                "stylePreset": "cinematic",
                "normalizedSettings": {},
                "rawAdapterSettings": {}
            },
            "lineage": {
                "parents": [],
                "sourceAssetId": null,
                "sourceTimestamp": null,
                "jobId": "job_999"
            }
        }))
        .expect("asset fixture parses")
    }

    fn payload_fixture() -> JsonObject {
        json!({
            "projectId": "project_abc",
            "projectName": "Michael's Secret Film",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "negativePrompt": "blurry",
            "model": "z_image_turbo",
            "count": 4,
            "seed": 990001,
            "seeds": [990001, 990002, 990003, 990004],
            "width": 1024,
            "height": 1024,
            "stylePreset": "cinematic",
            "fitMode": "crop",
            "advanced": { "steps": 8, "sampler": "euler" }
        })
        .as_object()
        .cloned()
        .expect("payload fixture is an object")
    }

    #[test]
    fn producer_version_is_the_workspace_version() {
        let share = build_workflow_share(&asset_fixture(), &payload_fixture());
        assert_eq!(share.producer.name, PRODUCER_NAME);
        assert_eq!(share.producer.url, PRODUCER_URL);
        assert_eq!(share.producer.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn seed_comes_from_the_rendered_asset_not_the_batch_base() {
        let share = build_workflow_share(&asset_fixture(), &payload_fixture());
        assert_eq!(share.seed, Some(4242));
        let encoded = serde_json::to_string(&share).expect("serializes");
        assert!(
            !encoded.contains("seeds"),
            "the batch seed list must not travel"
        );
        assert!(
            !encoded.contains("990002"),
            "another image's seed must not travel"
        );
    }

    #[test]
    fn unclassified_advanced_keys_are_dropped() {
        let advanced = json!({
            "steps": 8,
            "aBrandNewKnobNobodyClassified": "leaky",
            "quantTier": "nvfp4",
            "mlxQuantize": 4,
            "flashAttn": false,
            "controlWeights": { "overlayId": "overlay_1" }
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        assert_eq!(sorted_keys(&sanitized), vec!["steps"]);
    }

    #[test]
    fn scalar_keys_reject_smuggled_objects_and_paths() {
        let advanced = json!({
            "sampler": { "path": "C:\\Users\\Michael\\samplers\\x.json" },
            "scheduler": "/home/michael/schedules/x",
            "steps": 8
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        assert_eq!(sorted_keys(&sanitized), vec!["steps"]);
    }

    #[test]
    fn structured_prompt_keeps_scalars_and_drops_the_free_form_caption() {
        let advanced = json!({
            "structuredPrompt": {
                "version": 1,
                "intent": "a lighthouse",
                "caption": { "compositional_deconstruction": { "subject": "lighthouse" } },
                "magicPromptBackend": null,
                "edited": true,
                "runtimePrompt": "{\"subject\":\"lighthouse\"}"
            }
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        let recipe = sanitized["structuredPrompt"].as_object().expect("object");
        assert!(recipe.contains_key("intent"));
        assert!(recipe.contains_key("runtimePrompt"));
        assert!(recipe.contains_key("edited"));
        assert!(!recipe.contains_key("caption"));
        // `magicPromptBackend: null` is not a scalar we carry.
        assert!(!recipe.contains_key("magicPromptBackend"));
    }

    #[test]
    fn poses_keep_keypoints_and_drop_library_ids() {
        let advanced = json!({
            "poses": [
                { "id": "pose_local_uuid", "keypoints": [[0.1, 0.2], [0.3, 0.4]] },
                { "id": "pose_other" }
            ]
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        let poses = sanitized["poses"].as_array().expect("array");
        assert_eq!(poses.len(), 2, "pose count is load-bearing");
        assert!(poses[0]["keypoints"].is_array());
        assert!(poses[0].get("id").is_none());
        assert!(poses[1].as_object().expect("object").is_empty());
        let encoded = serde_json::to_string(&sanitized).expect("serializes");
        assert!(!encoded.contains("pose_local_uuid"));
    }

    #[test]
    fn phases_keep_numeric_schedule_and_lora_indices() {
        let advanced = json!({
            "phases": [
                { "steps": 4, "guidance": 1.0, "loras": [{ "index": 0, "weight": 0.8 }], "note": "x" },
                { "steps": 4, "loras": [{ "id": "lora_local_uuid" }] }
            ]
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        let phases = sanitized["phases"].as_array().expect("array");
        assert_eq!(phases[0]["loras"][0]["index"], json!(0));
        assert!(phases[0].get("note").is_none());
        // The second phase named a LoRA by local id and nothing else, so there is no schedule left
        // to record — and the key is ABSENT rather than `[]`. sc-15949 review: an empty array is not
        // an absence, it is the positive claim "this phase applies no LoRAs", which for a Krea
        // multi-phase recipe is the substance of the thing. `advanced_omissions` names the loss;
        // `an_over_cap_phase_lora_schedule_is_omitted_not_emptied` covers that half.
        assert!(
            phases[1].get("loras").is_none(),
            "{:?}",
            phases[1].get("loras")
        );
        let encoded = serde_json::to_string(&sanitized).expect("serializes");
        assert!(!encoded.contains("lora_local_uuid"));
    }

    #[test]
    fn loras_keep_name_weight_and_hf_repo_only() {
        let payload = json!({
            "mode": "text_to_image",
            "prompt": "p",
            "model": "z_image_turbo",
            "loras": [
                {
                    "id": "lora_local_uuid",
                    "name": "Mira Character LoRA",
                    "weight": 0.8,
                    "installedPath": "C:\\Users\\Michael\\loras\\mira",
                    "sourcePath": "/mnt/data/mira.safetensors",
                    "source": { "provider": "huggingface", "repo": "acme/mira", "file": "v1.safetensors" }
                },
                {
                    "id": "lora_local_2",
                    "name": "Local Import",
                    "weight": 0.5,
                    "source": { "provider": "local", "path": "loras/local.safetensors" }
                }
            ]
        })
        .as_object()
        .cloned()
        .expect("object");
        let share = build_workflow_share(&asset_fixture(), &payload);
        assert_eq!(share.loras.len(), 2);
        assert_eq!(share.loras[0].repo.as_deref(), Some("acme/mira"));
        assert_eq!(share.loras[0].weight, Some(0.8));
        assert_eq!(share.loras[1].repo, None);
        let encoded = serde_json::to_string(&share).expect("serializes");
        for leak in [
            "lora_local_uuid",
            "installedPath",
            "sourcePath",
            "v1.safetensors",
            "loras/local.safetensors",
        ] {
            assert!(!encoded.contains(leak), "{leak} leaked into the envelope");
        }
    }

    #[test]
    fn upscale_travels_only_when_enabled() {
        let mut payload = payload_fixture();
        payload.insert(
            "upscale".to_owned(),
            json!({ "enabled": false, "factor": 2, "engine": "seedvr2" }),
        );
        assert!(build_workflow_share(&asset_fixture(), &payload)
            .upscale
            .is_none());
        payload.insert(
            "upscale".to_owned(),
            json!({ "enabled": true, "factor": 2, "engine": "seedvr2", "softness": 0.25 }),
        );
        let upscale = build_workflow_share(&asset_fixture(), &payload)
            .upscale
            .expect("upscale travels");
        assert_eq!(upscale.factor, Some(2));
        assert_eq!(upscale.engine.as_deref(), Some("seedvr2"));
        assert_eq!(upscale.softness, Some(0.25));
    }

    #[test]
    fn is_path_shaped_matches_the_four_path_families() {
        for path in [
            "/home/michael/x.png",
            "C:\\Users\\Michael\\x.png",
            "engine loaded from D:/models/x",
            "\\\\fileserver\\share\\x.png",
            "file:///D:/x.png",
            "~/models/x.png",
        ] {
            assert!(is_path_shaped(path), "{path} should read as a path");
        }
        for safe in [
            "euler",
            "1024x1024",
            "acme/mira",
            "https://github.com/SceneWorks/SceneWorks",
            "2k",
            "",
        ] {
            assert!(!is_path_shaped(safe), "{safe} should not read as a path");
        }
    }

    /// The guard used to test only the FIRST character for `\` and only recognized a drive
    /// letter when a separator followed it, so every one of these travelled verbatim out of an
    /// allow-listed key — and the relative-Windows ones carry the OS username. The story's line
    /// is "every filesystem path without exception", so each of these is pinned here.
    #[test]
    fn is_path_shaped_catches_relative_traversing_and_drive_relative_locations() {
        for path in [
            // Relative Windows, no leading separator.
            "Users\\Michael\\Desktop\\secret.png",
            "models\\weights\\x.safetensors",
            // Traversal, both separators.
            "..\\..\\Users\\Michael",
            "../../etc/passwd",
            "./local/thing",
            // Drive-relative: a drive prefix with NO separator after it.
            "C:foo",
            "c:secret\\file",
            // Environment expansion.
            "%USERPROFILE%\\x",
            // A relative POSIX tree.
            "assets/images/x.png",
            // `~user` as well as `~/`.
            "~michael/x",
            // Percent-encoded `file://D:/x`, single- and double-encoded.
            "file%3A%2F%2FD%3A%2Fx",
            "FILE%3a%2F%2FD%3A%2fx",
            "file%253A%252F%252FD%253A%252Fx",
            "Users%5CMichael%5CDesktop%5Cx.png",
        ] {
            assert!(is_path_shaped(path), "{path:?} should read as a path");
        }
    }

    /// The other half of the guard: a check this aggressive must not eat legitimate labels.
    /// Everything here is a real value the envelope carries.
    #[test]
    fn is_path_shaped_keeps_legitimate_labels() {
        for safe in [
            "euler",
            "dpmpp_2m",
            "beta",
            "cfg_pp",
            "1024x1024",
            "2k",
            // Hugging Face repo ids are `owner/name` — two segments, and they stay.
            "acme/mira",
            "acme/foggy-coast",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "https://github.com/SceneWorks/SceneWorks",
            "https://huggingface.co/acme/mira/tree/main",
            "text_to_image",
            "krea_2_turbo",
            "z_image_turbo",
            "seedvr2",
            "noir_bloom",
            "cinematic",
            "crop",
            "canny",
            "three_quarter_left",
            "",
        ] {
            assert!(!is_path_shaped(safe), "{safe:?} should not read as a path");
        }
    }

    /// A slash with a space beside it is how a person separates a LIST, and the tally read three
    /// of those as a three-deep tree. `config/manifests/builtin.models.jsonc` is the proof it
    /// happens for real — the PiD decoder display names below are shipped values. Those three do
    /// not travel (the envelope carries the model SLUG), but `loras[].name` is exactly the same
    /// class of free display label and DOES travel, so this was silent data loss waiting for a
    /// user who names a LoRA the way people name things.
    #[test]
    fn a_slash_written_with_a_space_beside_it_is_a_list_separator() {
        for label in [
            // Shipped display names, verbatim from the builtin model manifest.
            "PiD 1.5 Decoder (FLUX.1 / Boogu / Chroma / Z-Image)",
            "PiD 1.5 Decoder (FLUX.2 / Lens / Ideogram 4)",
            "PiD Decoder (SDXL / RealVisXL / Kolors)",
            // The class that actually travels: a LoRA the user named themselves.
            "Ghibli watercolor / soft light / pastel",
            "Realism / Photoreal",
            "A/B test",
            "acme/mira",
            "https://huggingface.co/acme/mira/tree/main",
        ] {
            assert!(!is_path_shaped(label), "{label:?} is a label, not a path");
        }
        // Narrowed at the SLASH and not at the segment, so a real relative path keeps tripping
        // it even when its directories have spaces in their names.
        for path in [
            "assets/images/x.png",
            "models/weights/x.safetensors",
            "../../etc/passwd",
            "Documents/Secret Project/render 1.png",
            "a/b/c",
        ] {
            assert!(is_path_shaped(path), "{path:?} should still read as a path");
        }
        // The tally itself: empty segments never counted and still do not.
        assert_eq!(relative_tree_segments("a//b"), 2);
        assert_eq!(relative_tree_segments("/a/b/"), 2);
        assert_eq!(relative_tree_segments("A / B / C"), 1);
    }

    /// The field the narrowing is FOR. A dropped `loras[].name` is invisible to everyone: the
    /// user picked those words and no signal says they went missing.
    #[test]
    fn a_lora_named_with_a_spaced_slash_list_still_travels() {
        const NAME: &str = "Ghibli watercolor / soft light / pastel";
        let mut payload = payload_fixture();
        payload.insert(
            "loras".to_owned(),
            json!([
                { "name": NAME, "weight": 0.7, "source": { "provider": "huggingface", "repo": "acme/mira" } },
                { "name": "loras/local/x.safetensors", "weight": 0.4 }
            ]),
        );
        let share = build_workflow_share(&asset_fixture(), &payload);
        assert_eq!(share.loras[0].name.as_deref(), Some(NAME));
        // The other direction is untouched: a real relative tree in the same field still goes.
        assert_eq!(share.loras[1].name, None);

        // And it survives the trust boundary, so a shared image reloads with the name intact.
        let envelope = serde_json::to_value(&share).expect("serializes");
        let parsed = parse_workflow_share(&envelope).expect("parses");
        assert_eq!(parsed.loras[0].name.as_deref(), Some(NAME));
    }

    #[test]
    fn prose_keys_carry_what_the_user_typed_even_when_it_names_a_path() {
        // The deliberate exemption, pinned. `stylePrompt` and the structured prompt's `intent` /
        // `runtimePrompt` are the same class as the top-level `prompt`, which the story puts IN:
        // silently mangling authored text because it mentions a directory would be worse than the
        // leak it prevents, and the user can see what they typed.
        let advanced = json!({
            "stylePrompt": "C:\\Users\\Michael\\Desktop\\secret_project\\brief.txt",
            "structuredPrompt": {
                "intent": "/home/michael/clients/acme/nda.md",
                "runtimePrompt": "see ..\\..\\briefs\\acme.json"
            },
            // The interleave system prompt joined this class in sc-15948 — same reason.
            "systemMessage": "Ground every panel on ..\\..\\briefs\\acme.json",
            "sampler": "C:\\Users\\Michael\\samplers\\euler.json"
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        assert_eq!(
            sanitized["stylePrompt"],
            json!("C:\\Users\\Michael\\Desktop\\secret_project\\brief.txt")
        );
        let recipe = sanitized["structuredPrompt"].as_object().expect("object");
        assert_eq!(recipe["intent"], json!("/home/michael/clients/acme/nda.md"));
        assert_eq!(
            recipe["runtimePrompt"],
            json!("see ..\\..\\briefs\\acme.json")
        );
        assert_eq!(
            sanitized["systemMessage"],
            json!("Ground every panel on ..\\..\\briefs\\acme.json")
        );
        // A NON-prose key next to them is still guarded — the exemption is per key, not global.
        assert!(!sanitized.contains_key("sampler"));
        assert_eq!(
            PROSE_KEYS,
            ["stylePrompt", "intent", "runtimePrompt", "systemMessage"]
        );
    }

    /// The prose exemption is from the PATH check and from nothing else. A prompt is
    /// user-authored on the way out and attacker-chosen on the way in — the epic's trust
    /// boundary — and sc-15952 renders it, so a value carrying terminal escapes or megabytes of
    /// text must not arrive intact.
    #[test]
    fn prose_from_outside_is_control_stripped_and_bounded() {
        let hostile = |prose: &str| {
            json!({
                "sceneworksWorkflow": "image",
                "schemaVersion": 1,
                "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": prose,
                "negativePrompt": prose,
                "advanced": {
                    "stylePrompt": prose,
                    "structuredPrompt": { "intent": prose, "runtimePrompt": prose }
                }
            })
        };
        let prose_fields = |share: &WorkflowShare| {
            let recipe = share.advanced["structuredPrompt"]
                .as_object()
                .expect("structuredPrompt object");
            vec![
                share.prompt.clone(),
                share.negative_prompt.clone(),
                share.advanced["stylePrompt"]
                    .as_str()
                    .expect("stylePrompt string")
                    .to_owned(),
                recipe["intent"].as_str().expect("intent").to_owned(),
                recipe["runtimePrompt"]
                    .as_str()
                    .expect("runtimePrompt")
                    .to_owned(),
            ]
        };

        // Control characters: the ESC that starts an ANSI sequence, a NUL and a BEL all go.
        // Newlines and tabs stay — a multi-line prompt is normal, and mangling one would be the
        // very harm the prose exemption exists to prevent.
        let share = parse_workflow_share(&hostile(
            "  a lighthouse\u{1b}[31m in fog\u{0}\u{7}\r\n\tsecond line  ",
        ))
        .expect("parses");
        for value in prose_fields(&share) {
            assert_eq!(value, "a lighthouse[31m in fog\n\tsecond line");
            assert!(
                !value
                    .chars()
                    .any(|c| c.is_control() && c != '\n' && c != '\t'),
                "a control character survived: {value:?}"
            );
        }

        // Several megabytes of prose is not a prompt, it is a payload.
        let absurd = "x".repeat(4 * 1024 * 1024);
        let share = parse_workflow_share(&hostile(&absurd)).expect("parses");
        for value in prose_fields(&share) {
            assert_eq!(value.len(), PROSE_MAX_BYTES);
        }

        // The same field in BYTES, which is the review's finding: a character count bounds nothing
        // that is measured in bytes, and every persisted copy of this envelope is. Four megabytes of
        // a 4-byte scalar used to come back as 20,000 characters — 80 kB, four times what the old
        // comment claimed the whole envelope's prose could be.
        let absurd = "\u{1F600}".repeat(1024 * 1024);
        let share = parse_workflow_share(&hostile(&absurd)).expect("parses");
        for value in prose_fields(&share) {
            assert!(
                value.len() <= PROSE_MAX_BYTES,
                "{} bytes, over the {PROSE_MAX_BYTES} byte bound",
                value.len()
            );
            // Truncated on a CHARACTER boundary: 16,384 is a multiple of 4, so this lands exactly,
            // and every scalar is intact. `String` cannot hold a split sequence, so the assertion
            // that matters is that no character was lost to a partial one.
            assert_eq!(value.chars().count(), PROSE_MAX_BYTES / 4);
            assert!(value.chars().all(|character| character == '\u{1F600}'));
        }

        // A 3-byte script does not divide the budget evenly, which is the case that would split a
        // sequence if the bound were applied to bytes rather than to whole characters.
        let cjk = "霧".repeat(100_000);
        let share = parse_workflow_share(&hostile(&cjk)).expect("parses");
        for value in prose_fields(&share) {
            assert_eq!(value.chars().count(), PROSE_MAX_BYTES / 3);
            assert_eq!(value.len(), (PROSE_MAX_BYTES / 3) * 3);
            assert!(value.chars().all(|character| character == '霧'));
        }

        // The number itself: 16 KiB is the widest UTF-8 encoding of the API's own 4,000-character
        // ceiling on this field class (`MAX_PROMPT_CHARS` in `apps/rust-api/src/lib.rs`), so a
        // prompt this app accepted cannot be truncated here in ANY script.
        assert_eq!(PROSE_MAX_BYTES, 16 * 1024);
        const _: () = assert!(PROSE_MAX_BYTES >= 4_000 * 4);
        // And prose-sized rather than label-sized: the two must never be conflated.
        const _: () = assert!(PROSE_MAX_BYTES > LABEL_MAX_CHARS * 4);
    }

    /// The other half: a bound this blunt must not touch anything real. A prompt far longer than
    /// anyone types, with the newlines and tabs a structured recipe carries, comes back byte for
    /// byte — on the way out AND on the way back in.
    #[test]
    fn a_realistic_long_prompt_survives_byte_for_byte() {
        let prompt = format!(
            "{}\n\tshot on 35mm, volumetric light",
            "a lighthouse in heavy fog, cinematic, rain on the lens, ".repeat(30)
        );
        assert!(prompt.chars().count() > 1_500, "the fixture must be long");
        assert!(prompt.len() < PROSE_MAX_BYTES);

        let mut payload = payload_fixture();
        payload.insert("prompt".to_owned(), json!(prompt));
        payload.insert("negativePrompt".to_owned(), json!(prompt));
        payload.insert(
            "advanced".to_owned(),
            json!({
                "stylePrompt": prompt,
                "structuredPrompt": { "intent": prompt, "runtimePrompt": prompt }
            }),
        );
        let share = build_workflow_share(&asset_fixture(), &payload);
        assert_eq!(share.prompt, prompt);
        assert_eq!(share.negative_prompt, prompt);
        assert_eq!(share.advanced["stylePrompt"], json!(prompt));

        let envelope = serde_json::to_value(&share).expect("serializes");
        let parsed = parse_workflow_share(&envelope).expect("parses");
        assert_eq!(parsed.prompt, prompt);
        assert_eq!(parsed.negative_prompt, prompt);
        assert_eq!(parsed.advanced["stylePrompt"], json!(prompt));
        let recipe = parsed.advanced["structuredPrompt"]
            .as_object()
            .expect("structuredPrompt object");
        assert_eq!(recipe["intent"], json!(prompt));
        assert_eq!(recipe["runtimePrompt"], json!(prompt));
    }

    #[test]
    fn top_level_labels_are_path_guarded_like_advanced_ones() {
        let mut payload = payload_fixture();
        for (key, value) in [
            ("mode", "C:\\modes\\evil"),
            ("model", "C:\\models\\evil"),
            ("stylePreset", "\\\\server\\styles\\x"),
            ("styleId", "../../etc/passwd"),
            ("fitMode", "/etc/passwd"),
        ] {
            payload.insert(key.to_owned(), json!(value));
        }
        payload.insert(
            "upscale".to_owned(),
            json!({ "enabled": true, "engine": "E:\\engines\\seedvr2" }),
        );
        let share = build_workflow_share(&asset_fixture(), &payload);
        assert_eq!(share.mode, "");
        assert_eq!(share.model, "");
        assert_eq!(share.style_preset, None);
        assert_eq!(share.style_id, None);
        assert_eq!(share.fit_mode, None);
        assert_eq!(share.upscale.as_ref().expect("upscale").engine, None);
        let encoded = serde_json::to_string(&share).expect("serializes");
        for leak in ["evil", "server", "etc/passwd", "engines"] {
            assert!(!encoded.contains(leak), "{leak} leaked into the envelope");
        }
    }

    #[test]
    fn poses_keep_every_numeric_coordinate_array_the_worker_renders() {
        // `hands` and `face` change the rendered skeleton exactly as `keypoints` does and hold
        // nothing but coordinates, so dropping them would cost fidelity for no privacy gain.
        let advanced = json!({
            "poses": [{
                "id": "pose_local_uuid",
                "label": "Standing, arms out",
                "keypoints": [[0.1, 0.2], [0.3, 0.4]],
                "hands": [[[0.5, 0.6]], [[0.7, 0.8]]],
                "face": [[0.9, 1.0]],
                "sourcePath": "C:\\poses\\a.json"
            }]
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        let pose = sanitized["poses"][0].as_object().expect("object");
        assert_eq!(sorted_keys(pose), vec!["face", "hands", "keypoints"]);
        let encoded = serde_json::to_string(&sanitized).expect("serializes");
        for leak in ["pose_local_uuid", "Standing", "sourcePath", "C:\\\\poses"] {
            assert!(!encoded.contains(leak), "{leak} leaked into the envelope");
        }

        // A non-numeric `hands`/`face` is not coordinates and does not travel.
        let smuggled = json!({
            "poses": [{ "keypoints": [[0.1, 0.2]], "hands": "C:\\hands\\a.json", "face": { "path": "/x" } }]
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&smuggled);
        let pose = sanitized["poses"][0].as_object().expect("object");
        assert_eq!(sorted_keys(pose), vec!["keypoints"]);
    }

    #[test]
    fn parse_rejects_a_missing_marker() {
        let error = parse_workflow_share(&json!({ "schemaVersion": 1 }))
            .expect_err("no marker must not parse");
        assert_eq!(error, WorkflowShareError::MissingMarker);
        assert!(error.to_string().contains("sceneworksWorkflow"));
    }

    #[test]
    fn parse_rejects_non_objects_and_bad_json() {
        assert_eq!(
            parse_workflow_share(&json!([1, 2, 3])).expect_err("array"),
            WorkflowShareError::NotAnObject
        );
        assert!(matches!(
            parse_workflow_share_json("{not json").expect_err("bad json"),
            WorkflowShareError::InvalidJson(_)
        ));
    }

    #[test]
    fn parse_names_both_versions_on_an_unknown_schema_version() {
        let future = json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": WORKFLOW_SHARE_SCHEMA_VERSION + 7,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "9.9.9" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "p"
        });
        let error = parse_workflow_share(&future).expect_err("future version must not parse");
        assert_eq!(
            error,
            WorkflowShareError::UnsupportedSchemaVersion {
                file: WORKFLOW_SHARE_SCHEMA_VERSION + 7,
                supported: WORKFLOW_SHARE_SCHEMA_VERSION,
            }
        );
        let message = error.to_string();
        assert!(message.contains(&(WORKFLOW_SHARE_SCHEMA_VERSION + 7).to_string()));
        assert!(message.contains(&WORKFLOW_SHARE_SCHEMA_VERSION.to_string()));
        assert!(message.contains("Update SceneWorks"));
    }

    #[test]
    fn parse_reports_a_malformed_body_instead_of_panicking() {
        let malformed = json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": ["not", "a", "string"]
        });
        assert!(matches!(
            parse_workflow_share(&malformed).expect_err("malformed body"),
            WorkflowShareError::Malformed { .. }
        ));
    }

    /// An unknown kind is refused, and the two known ones are not.
    ///
    /// `video` was the example here until sc-15956 made it a lane. That is the whole point of the
    /// gate: a build refuses a kind it does not understand rather than presenting it as one it
    /// does, and an OLDER build hits this same branch for a video envelope.
    #[test]
    fn parse_rejects_another_workflow_kind() {
        let envelope = |kind: &str| {
            json!({
                "sceneworksWorkflow": kind,
                "schemaVersion": 1,
                "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
                "mode": "image_to_video",
                "model": "wan_5b",
                "prompt": "p"
            })
        };
        for known in WORKFLOW_KINDS {
            assert!(
                parse_workflow_share(&envelope(known)).is_ok(),
                "`{known}` is a kind this build reads"
            );
        }
        assert!(matches!(
            parse_workflow_share(&envelope("hologram")).expect_err("an unknown kind"),
            WorkflowShareError::UnsupportedKind { .. }
        ));
    }

    #[test]
    fn parse_re_sanitizes_advanced_from_an_outside_writer() {
        let hostile = json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "p",
            "advanced": { "steps": 8, "controlWeights": { "path": "C:\\evil" } },
            "somethingNew": "dropped"
        });
        let share = parse_workflow_share(&hostile).expect("parses");
        assert_eq!(sorted_keys(&share.advanced), vec!["steps"]);
        let encoded = serde_json::to_string(&share).expect("serializes");
        assert!(!encoded.contains("somethingNew"));
    }

    /// Dropping unknown KEYS is only half the job: every string under a key we DO declare is
    /// attacker-chosen in a file that came from a stranger. `loras[].repo` is the sharpest edge
    /// — sc-15952 joins a repo id into a Hugging Face cache path — so a traversal string there
    /// must not survive the parse. Nothing here is echoed back out.
    #[test]
    fn parse_reduces_a_hostile_envelope_instead_of_echoing_it() {
        let hostile = json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": {
                "name": "SceneWorks (build C:\\Users\\Michael\\src)",
                "url": "file:///D:/exfil.html",
                "version": "0.8.1-dirty-MICHAELS-PC"
            },
            "mode": "..\\..\\Users\\Michael",
            "model": "C:\\models\\evil",
            "prompt": "a lighthouse",
            "stylePreset": "\\\\server\\styles\\x",
            "styleId": "../../etc/passwd",
            "fitMode": "/etc/passwd",
            "upscale": { "enabled": true, "factor": 2, "engine": "E:\\engines\\seedvr2" },
            "loras": [
                { "name": "Foggy Coast", "weight": 0.65, "repo": "../../../etc/passwd" },
                { "name": "C:\\Users\\Michael\\loras\\x", "repo": "acme/mira" },
                { "name": "..\\..\\Users\\Michael\\loras", "weight": 0.4, "repo": "acme/../../etc" },
                { "name": "/etc/passwd", "repo": "../x" }
            ],
            "inputs": [
                { "kind": "source", "count": 1 },
                { "kind": "C:\\Users\\Michael", "count": 1 },
                { "kind": "control", "count": 1, "controlMode": "/etc/passwd" }
            ],
            "advanced": { "steps": 8, "sampler": "C:\\Users\\Michael\\euler.json" }
        });
        let share = parse_workflow_share(&hostile).expect("a hostile envelope still parses");

        assert_eq!(share.mode, "");
        assert_eq!(share.model, "");
        assert_eq!(share.style_preset, None);
        assert_eq!(share.style_id, None);
        assert_eq!(share.fit_mode, None);
        assert_eq!(share.upscale.as_ref().expect("upscale").engine, None);

        // `repo` is validated as `owner/name`, exactly as it is on build. The fourth entry had
        // nothing left to say once its name and repo were dropped, so it does not travel at all.
        assert_eq!(share.loras.len(), 3);
        assert_eq!(share.loras[0].repo, None);
        assert_eq!(share.loras[0].name.as_deref(), Some("Foggy Coast"));
        assert_eq!(share.loras[1].repo.as_deref(), Some("acme/mira"));
        assert_eq!(share.loras[1].name, None);
        assert_eq!(share.loras[2].repo, None);
        assert_eq!(share.loras[2].name, None);
        assert_eq!(share.loras[2].weight, Some(0.4));

        // `.` and `-` are legal inside an HF segment, so `../x` is the shape a naive
        // `owner/name` check waves through. A segment must start with an alphanumeric.
        for rejected in [
            "../x",
            "acme/..",
            "./x",
            "../../../etc/passwd",
            "acme/../../etc",
            "/acme/mira",
            "acme",
            "",
        ] {
            assert_eq!(hf_repo_id(rejected), None, "{rejected:?} is not a repo id");
        }
        for accepted in [
            "acme/mira",
            "acme/foggy-coast",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ] {
            assert_eq!(hf_repo_id(accepted).as_deref(), Some(accepted));
        }

        // An `inputs[].kind` outside the closed vocabulary is not something a reader can act on.
        let kinds: Vec<&str> = share
            .inputs
            .iter()
            .map(|input| input.kind.as_str())
            .collect();
        assert_eq!(kinds, vec![INPUT_KIND_SOURCE, INPUT_KIND_CONTROL]);
        assert_eq!(share.inputs[1].control_mode, None);

        // The producer block is provenance nobody vouched for; a name, URL or version that is
        // not the shape we publish is reduced to empty rather than shown to the user as fact.
        assert_eq!(share.producer.name, "");
        assert_eq!(share.producer.url, "");
        assert_eq!(share.producer.version, "");

        assert_eq!(sorted_keys(&share.advanced), vec!["steps"]);

        let encoded = serde_json::to_string(&share).expect("serializes");
        for leak in [
            "Michael", "C:\\\\", "E:\\\\", "etc", "file://", "server", "evil", "exfil", "dirty",
        ] {
            assert!(
                !encoded.contains(leak),
                "{leak} survived the parse: {encoded}"
            );
        }
    }

    /// A well-formed producer block from another build travels intact — the reduction bounds the
    /// block, it does not erase it (the whole point of recording the producer is a bug report
    /// that says which build wrote the file).
    #[test]
    fn parse_keeps_a_well_formed_producer_block_from_another_build() {
        let other = json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "99.12.0" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "p"
        });
        let share = parse_workflow_share(&other).expect("parses");
        assert_eq!(share.producer.name, PRODUCER_NAME);
        assert_eq!(share.producer.url, PRODUCER_URL);
        assert_eq!(share.producer.version, "99.12.0");
    }

    #[test]
    fn parse_distinguishes_a_malformed_schema_version_from_a_missing_one() {
        let envelope = |version: Value| {
            let mut object = JsonObject::new();
            object.insert(
                WORKFLOW_SHARE_MARKER_KEY.to_owned(),
                json!(WORKFLOW_KIND_IMAGE),
            );
            if !version.is_null() {
                object.insert("schemaVersion".to_owned(), version);
            }
            Value::Object(object)
        };

        // Absent (and an explicit null, which a writer that omitted the field usually means).
        for missing in [Value::Null, json!(null)] {
            assert_eq!(
                parse_workflow_share(&envelope(missing)).expect_err("no schemaVersion"),
                WorkflowShareError::MissingSchemaVersion
            );
        }
        assert_eq!(
            parse_workflow_share(&json!({
                "sceneworksWorkflow": "image",
                "schemaVersion": null
            }))
            .expect_err("null schemaVersion"),
            WorkflowShareError::MissingSchemaVersion
        );

        // Present but the wrong shape — a different problem with a different fix, so a
        // different sentence. Telling someone a field they can SEE is missing sends them
        // looking in the wrong place.
        for malformed in [json!("1"), json!(-1), json!(1.5), json!([1]), json!({})] {
            let error = parse_workflow_share(&envelope(malformed.clone()))
                .expect_err("malformed schemaVersion");
            assert_eq!(
                error,
                WorkflowShareError::Malformed {
                    field: "schemaVersion".to_owned(),
                    detail:
                        "must be a whole number (the contract version the file was written with)"
                            .to_owned(),
                },
                "{malformed} should read as malformed, not missing"
            );
            let message = error.to_string();
            assert!(message.contains("schemaVersion"), "{message}");
            assert!(
                !message.contains("missing"),
                "a present-but-wrong version must not be reported as missing: {message}"
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // Collection bounds
    // -----------------------------------------------------------------------------------------

    /// Every cap, against the contract it is derived from. The numbers are load-bearing, so a
    /// change to either side has to be a decision rather than a drift.
    #[test]
    fn every_collection_cap_is_the_contract_it_came_from() {
        assert_eq!(MAX_SHARE_LORAS, crate::lora_family::MAX_JOB_LORAS);
        assert_eq!(MAX_SHARE_LORAS, 5);
        assert_eq!(MAX_SHARE_INPUTS, INPUT_KINDS.len());
        // Four image kinds plus the video lane's two clip kinds (sc-15956).
        assert_eq!(MAX_SHARE_INPUTS, 6);
        // The worker's `MAX_MULTIPHASE_PHASES`; pinned against its source by
        // `the_phase_cap_matches_the_multi_phase_validators` in tests/workflow_share.rs, which can
        // read the file this crate cannot import.
        assert_eq!(MAX_SHARE_PHASES, 8);
        // The pose budget is 16 whole-body skeletons, and 16 is twice the payload-sanity ceiling on
        // images per job — the thing a pose set is the pose lane's version of. If that ceiling
        // moves, the budget is a decision to re-make, not a number to follow.
        assert_eq!(MAX_SHARE_POSE_SLOTS, 6_144);
        assert_eq!(
            MAX_SHARE_POSE_SLOTS,
            2 * crate::image_request::MAX_COUNT as usize * (18 + 42 + 68) * 3
        );
    }

    /// The bound that did not exist: an envelope with 8,000 entries in a collection. Asserted in
    /// BOTH directions from the same fixture, because a bound only one direction runs is how the
    /// write and read sides drift apart.
    #[test]
    fn an_over_cap_collection_is_dropped_in_both_directions() {
        let many_loras: Vec<Value> = (0..8_000)
            .map(|index| json!({ "name": format!("lora {index}"), "weight": 0.5 }))
            .collect();
        let many_inputs: Vec<Value> = (0..8_000)
            .map(|_| json!({ "kind": "reference", "count": 1 }))
            .collect();

        // Parse: a file from a stranger.
        let parsed = parse_workflow_share(&json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "a lighthouse",
            "loras": many_loras.clone(),
            "inputs": many_inputs.clone(),
            "advanced": {
                "steps": 8,
                "poses": (0..8_000).map(|_| json!({})).collect::<Vec<Value>>(),
                "phases": (0..8_000).map(|_| json!({ "steps": 4 })).collect::<Vec<Value>>(),
            }
        }))
        .expect("a bounded envelope still parses — the recipe is not the collections");
        assert!(parsed.loras.is_empty(), "{:?}", parsed.loras);
        assert!(parsed.inputs.is_empty(), "{:?}", parsed.inputs);
        assert!(parsed.advanced.get("poses").is_none());
        assert!(parsed.advanced.get("phases").is_none());
        // Dropped, not truncated: a subset would be offered to the user as the recipe that made
        // the image, and they could not tell it was a subset.
        assert_eq!(parsed.advanced["steps"], json!(8), "the rest still travels");
        assert_eq!(parsed.prompt, "a lighthouse");

        // Build: the same collections arriving on a job payload.
        let mut payload = payload_fixture();
        payload.insert("loras".to_owned(), Value::Array(many_loras));
        payload.insert(
            "advanced".to_owned(),
            json!({
                "steps": 8,
                "poses": (0..8_000).map(|_| json!({ "keypoints": [[0.1, 0.2]] })).collect::<Vec<Value>>(),
                "phases": (0..8_000).map(|_| json!({ "steps": 4 })).collect::<Vec<Value>>(),
            }),
        );
        let built = build_workflow_share(&asset_fixture(), &payload);
        assert!(built.loras.is_empty(), "{:?}", built.loras);
        assert!(built.advanced.get("poses").is_none());
        assert!(built.advanced.get("phases").is_none());
        assert_eq!(built.advanced["steps"], json!(8));
    }

    /// The other half of every bound: exactly at the cap must survive, whole, both ways. This is
    /// the assertion that would fail if a cap were set below anything legitimate.
    #[test]
    fn a_collection_exactly_at_its_cap_survives_both_directions() {
        let loras: Vec<Value> = (0..MAX_SHARE_LORAS)
            .map(|index| json!({ "name": format!("Lora {index}"), "weight": 0.5, "source": { "repo": format!("acme/lora-{index}") } }))
            .collect();
        let phases: Vec<Value> = (0..MAX_SHARE_PHASES)
            .map(|index| json!({ "steps": index + 1, "guidance": 3.5, "loras": [{ "index": 0, "weight": 0.4 }] }))
            .collect();
        // 16 full whole-body skeletons is the pose budget exactly: 18 + 42 + 68 points of [x, y].
        let skeleton = |points: usize| -> Value {
            Value::Array(
                (0..points)
                    .map(|point| json!([point as f64 / 100.0, 0.5]))
                    .collect(),
            )
        };
        let poses: Vec<Value> = (0..16)
            .map(|_| {
                json!({
                    "keypoints": skeleton(18),
                    "hands": [skeleton(21), skeleton(21)],
                    "face": skeleton(68),
                })
            })
            .collect();

        let mut payload = payload_fixture();
        payload.insert("loras".to_owned(), Value::Array(loras));
        payload.insert("sourceAssetId".to_owned(), json!("asset_source"));
        payload.insert(
            "referenceAssetIds".to_owned(),
            json!(["asset_a", "asset_b", "asset_c"]),
        );
        payload.insert("maskAssetId".to_owned(), json!("asset_mask"));
        payload.insert(
            "advanced".to_owned(),
            json!({
                "steps": 8,
                "controlImage": "asset_control",
                "controlMode": "canny",
                "poses": poses,
                "phases": phases,
            }),
        );

        let built = build_workflow_share(&asset_fixture(), &payload);
        assert_eq!(built.loras.len(), MAX_SHARE_LORAS);
        // One entry per kind — the widest `inputs` an IMAGE payload can emit, which is the
        // four image kinds. The two video clip kinds need a video payload to appear, so the
        // cap is above what this fixture reaches; `clip_ids_become_shape_descriptors` in
        // tests/workflow_mp4.rs is where the other two are exercised.
        assert_eq!(built.inputs.len(), 4);
        assert!(built.inputs.len() <= MAX_SHARE_INPUTS);
        assert_eq!(
            built.advanced["phases"].as_array().expect("phases").len(),
            MAX_SHARE_PHASES
        );
        let built_poses = built.advanced["poses"].as_array().expect("poses");
        assert_eq!(built_poses.len(), 16);
        assert_eq!(
            built_poses
                .iter()
                .map(count_coordinate_slots)
                .sum::<usize>(),
            MAX_SHARE_POSE_SLOTS * 2 / 3,
            "16 skeletons of [x, y] pairs is two thirds of a budget sized for [x, y, confidence]"
        );

        // And back in through the reader, unchanged.
        let envelope = serde_json::to_value(&built).expect("serializes");
        let parsed = parse_workflow_share(&envelope).expect("parses");
        assert_eq!(parsed, built);
    }

    /// The entry cap is not a volume cap. 64 poses is under [`MAX_SHARE_POSES`], so only the
    /// numeric budget can catch a pose carrying a million coordinates — and it is the whole reason
    /// the budget exists as well as the count.
    #[test]
    fn a_pose_that_is_a_payload_rather_than_a_skeleton_is_dropped() {
        let flood: Vec<Value> = (0..4_000).map(|index| json!([index, 0.5])).collect();
        let advanced = json!({ "steps": 8, "poses": [{ "keypoints": flood }] })
            .as_object()
            .cloned()
            .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        assert!(
            sanitized.get("poses").is_none(),
            "one pose held {} numbers, over the {MAX_SHARE_POSE_SLOTS} budget",
            8_000
        );
        assert_eq!(sanitized["steps"], json!(8));

        // The realistic large case is NOT refused: every one of the 46 built-in poses
        // (`apps/web/public/poses/index.json`) is 18 body keypoints with no hands and no face, so
        // selecting the whole library spends 1,656 of the 6,144 numbers.
        let library: Vec<Value> = (0..46)
            .map(|_| {
                json!({ "keypoints": (0..18).map(|point| json!([f64::from(point) / 20.0, 0.5])).collect::<Vec<Value>>() })
            })
            .collect();
        let advanced = json!({ "poses": library })
            .as_object()
            .cloned()
            .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        let poses = sanitized["poses"]
            .as_array()
            .expect("the whole library travels");
        assert_eq!(poses.len(), 46);
        assert_eq!(
            poses.iter().map(count_coordinate_slots).sum::<usize>(),
            46 * 36
        );
    }

    // -----------------------------------------------------------------------------------------
    // Volume that is not a number (sc-15949 review)
    // -----------------------------------------------------------------------------------------

    /// A `null` is a coordinate SLOT and costs one, and an empty array is not a coordinate at all.
    ///
    /// The budget counted `Value::Number` while the shape check accepted `Value::Null`, so the two
    /// disagreed about what a coordinate is and the gap between them was free volume: 200,000 nulls
    /// under one `keypoints` key survived and serialized to a megabyte. Empty arrays were the same
    /// hole from the other side — `[]` is vacuously "all coordinates" — at 600 kB.
    #[test]
    fn a_null_costs_a_slot_and_an_empty_array_is_not_a_coordinate() {
        // 1. Nulls, over the budget: the field is dropped, and the size the envelope would have
        //    carried is asserted rather than described.
        let nulls: Vec<Value> = (0..200_000).map(|_| Value::Null).collect();
        let unbounded = serde_json::to_string(&json!([{ "keypoints": nulls.clone() }]))
            .expect("serializes")
            .len();
        assert!(
            unbounded > 900_000,
            "the fixture must be a payload to be worth guarding: {unbounded} bytes"
        );
        let advanced = json!({ "steps": 8, "poses": [{ "keypoints": nulls }] })
            .as_object()
            .cloned()
            .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        assert!(sanitized.get("poses").is_none(), "{:?}", sanitized);
        assert_eq!(sanitized["steps"], json!(8), "the rest still travels");

        // 2. Empty arrays: refused by the shape check, so the field never occupies the budget at all.
        let empties: Vec<Value> = (0..200_000).map(|_| json!([])).collect();
        let advanced = json!({ "steps": 8, "poses": [{ "keypoints": empties }] })
            .as_object()
            .cloned()
            .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        let poses = sanitized["poses"].as_array().expect("the slot is kept");
        assert_eq!(poses.len(), 1, "a pose entry still occupies its slot");
        assert!(
            poses[0].as_object().expect("object").is_empty(),
            "an array with nothing in it is not a point: {:?}",
            poses[0]
        );
        assert!(!is_coordinate_tree(&json!([])));
        assert!(!is_coordinate_tree(&json!([[0.1, 0.2], []])));

        // 3. Nulls WITHIN the budget still travel: they are the missing-coordinate form the worker's
        //    `normalize_points` fills in, so counting them is a bound and not a ban.
        let advanced = json!({ "poses": [{ "keypoints": [[0.1, null], [null, null]] }] })
            .as_object()
            .cloned()
            .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        assert_eq!(
            sanitized["poses"][0]["keypoints"],
            json!([[0.1, null], [null, null]])
        );
        assert_eq!(count_coordinate_slots(&sanitized["poses"]), 4);

        // 4. And a bare scalar under a pose field is not a skeleton: `keypoints: 5` used to pass the
        //    shape check and record a positive claim about a pose nobody made.
        for shape in [json!(5), Value::Null, json!("keypoints")] {
            let advanced = json!({ "poses": [{ "keypoints": shape }] })
                .as_object()
                .cloned()
                .expect("object");
            let sanitized = sanitize_advanced(&advanced);
            assert!(
                sanitized["poses"][0]
                    .as_object()
                    .expect("object")
                    .is_empty(),
                "{:?} travelled as a keypoint set",
                sanitized["poses"][0]
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // The recording ceiling (sc-15949 review)
    // -----------------------------------------------------------------------------------------

    /// The bound that composes: per-field bounds, all satisfied, adding up to an envelope nobody
    /// would record.
    ///
    /// Every value here is legal on its own — six prose slots inside [`PROSE_MAX_BYTES`], every
    /// allow-listed scalar inside [`LABEL_MAX_CHARS`], a pose set exactly at
    /// [`MAX_SHARE_POSE_SLOTS`] — and their sum is ~220 kB. That is the shape each round of
    /// per-field caps kept missing, and the ceiling is what closes it rather than another cap.
    #[test]
    fn an_envelope_over_the_recording_ceiling_is_no_workflow_at_all() {
        let prose = "z".repeat(PROSE_MAX_BYTES);
        let label = "L".repeat(LABEL_MAX_CHARS);
        let point = json!([0.123_456_789_012_345_67, 0.987_654_321_098_765_4, 0.55]);
        let skeleton: Vec<Value> = (0..(18 + 42 + 68)).map(|_| point.clone()).collect();
        let mut advanced = JsonObject::new();
        for key in [
            "sampler",
            "scheduler",
            "guidanceMethod",
            "pidTarget",
            "controlMode",
            "styleId",
            "angleSet",
            "resolution",
        ] {
            advanced.insert(key.to_owned(), json!(label));
        }
        advanced.insert("stylePrompt".to_owned(), json!(prose));
        advanced.insert("systemMessage".to_owned(), json!(prose));
        advanced.insert(
            "structuredPrompt".to_owned(),
            json!({ "intent": prose, "runtimePrompt": prose }),
        );
        advanced.insert(
            "poses".to_owned(),
            json!((0..16)
                .map(|_| json!({ "keypoints": skeleton.clone() }))
                .collect::<Vec<Value>>()),
        );
        let envelope = json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": prose,
            "negativePrompt": prose,
            "advanced": Value::Object(advanced),
        });

        let error = parse_workflow_share(&envelope)
            .expect_err("an envelope over the recording ceiling must not be recorded");
        let WorkflowShareError::TooLarge { bytes, limit } = error else {
            panic!("wrong error: {error}");
        };
        assert_eq!(limit, WORKFLOW_SHARE_MAX_BYTES);
        assert!(
            bytes > WORKFLOW_SHARE_MAX_BYTES,
            "{bytes} is not over {WORKFLOW_SHARE_MAX_BYTES}"
        );
        // Every per-field bound held: this is a composition failure, not a field failure.
        assert!(
            bytes < 4 * WORKFLOW_SHARE_MAX_BYTES,
            "{bytes} — the per-field bounds should have kept this to a few hundred kB"
        );
        // NO workflow, not a partial one. Shedding the biggest field would record a recipe missing
        // exactly the part that made it too large, and nothing in the file would say which.
        let message = WorkflowShareError::TooLarge { bytes, limit }.to_string();
        assert!(message.contains("no recipe was read"), "{message}");

        // The same envelope minus the poses is under the ceiling and travels whole, so the ceiling is
        // a bound on the sum and not a refusal of prose.
        let mut smaller = envelope.clone();
        smaller["advanced"]
            .as_object_mut()
            .expect("object")
            .remove("poses");
        let share = parse_workflow_share(&smaller).expect("under the ceiling");
        assert_eq!(share.prompt.len(), PROSE_MAX_BYTES);
        assert!(
            workflow_share_bytes(&share) <= WORKFLOW_SHARE_MAX_BYTES,
            "{}",
            workflow_share_bytes(&share)
        );
    }

    /// The ceiling's figure, against the validators it is derived from.
    #[test]
    fn the_recording_ceiling_clears_every_upstream_validator() {
        assert_eq!(WORKFLOW_SHARE_MAX_BYTES, 160 * 1024);
        // The arithmetic maximum a legitimate envelope can reach: two prompt fields at the API's
        // 4,000-character cap in the widest encoding, the API's own ceiling on the serialized
        // `advanced` map, and the bounded labels/LoRAs around them. See the constant's own table.
        const LEGITIMATE_MAX: usize = 2 * 4_000 * 4 + 64 * 1024 + 8_250 + 15_000;
        const _: () = assert!(LEGITIMATE_MAX < WORKFLOW_SHARE_MAX_BYTES);
        // With real headroom, rather than a number that only just clears the sum.
        const _: () = assert!(LEGITIMATE_MAX * 4 / 3 < WORKFLOW_SHARE_MAX_BYTES);
        // And the write side runs the same gate as the read side, so a writer cannot produce a file
        // its own reader refuses.
        let payload = payload_fixture();
        let facts = WorkflowAssetFacts::from_asset(&asset_fixture());
        let embedded = embeddable_workflow_share(&facts, &payload).expect("a real envelope embeds");
        assert_eq!(embedded, build_workflow_share_from(&facts, &payload));
        assert!(workflow_share_bytes(&embedded) < 4 * 1024, "{embedded:?}");
    }

    /// The write side's half: an envelope the ceiling refuses is not embedded, rather than embedded
    /// for our own reader to refuse on the way back in.
    #[test]
    fn the_write_side_refuses_what_the_read_side_would() {
        let prose = "z".repeat(PROSE_MAX_BYTES);
        let point = json!([0.123_456_789_012_345_67, 0.987_654_321_098_765_4, 0.55]);
        let skeleton: Vec<Value> = (0..(18 + 42 + 68)).map(|_| point.clone()).collect();
        let mut payload = payload_fixture();
        payload.insert("prompt".to_owned(), json!(prose));
        payload.insert("negativePrompt".to_owned(), json!(prose));
        payload.insert(
            "advanced".to_owned(),
            json!({
                "stylePrompt": prose,
                "systemMessage": prose,
                "structuredPrompt": { "intent": prose, "runtimePrompt": prose },
                "poses": (0..16).map(|_| json!({ "keypoints": skeleton.clone() })).collect::<Vec<Value>>(),
            }),
        );
        let facts = WorkflowAssetFacts::from_asset(&asset_fixture());
        assert!(
            embeddable_workflow_share(&facts, &payload).is_none(),
            "the write seam must embed nothing rather than a file our own reader refuses"
        );
        // And the reader agrees about the same envelope, which is the property one shared gate buys.
        let built = build_workflow_share_from(&facts, &payload);
        let envelope = serde_json::to_value(&built).expect("serializes");
        assert!(matches!(
            parse_workflow_share(&envelope),
            Err(WorkflowShareError::TooLarge { .. })
        ));
    }

    // -----------------------------------------------------------------------------------------
    // The omission marker (sc-15949 review)
    // -----------------------------------------------------------------------------------------

    /// A dropped collection says so, in both directions.
    ///
    /// The premise the drop doctrine rested on — "a reader can tell an absence" — is false: `loras`
    /// carries `skip_serializing_if = "Vec::is_empty"`, so this test starts by proving that a 6-LoRA
    /// envelope whose LoRAs were dropped and a genuinely LoRA-free one would otherwise be BYTE
    /// IDENTICAL, and then that the marker is what tells them apart.
    #[test]
    fn a_dropped_collection_is_self_describing_in_both_directions() {
        let with_loras = json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "a lighthouse",
            "loras": (0..6).map(|index| json!({ "name": format!("Lora {index}"), "weight": 0.5 })).collect::<Vec<Value>>(),
        });
        let mut without_loras = with_loras.clone();
        without_loras
            .as_object_mut()
            .expect("object")
            .remove("loras");

        let dropped = parse_workflow_share(&with_loras).expect("parses");
        let never_had_any = parse_workflow_share(&without_loras).expect("parses");
        assert!(dropped.loras.is_empty());
        assert!(never_had_any.loras.is_empty());
        assert_eq!(dropped.omitted, vec![OMITTED_LORAS.to_owned()]);
        assert!(never_had_any.omitted.is_empty());
        assert_ne!(
            serde_json::to_string(&dropped).expect("serializes"),
            serde_json::to_string(&never_had_any).expect("serializes"),
            "without the marker these two serialize identically, which is the whole finding"
        );

        // The BUILD side too, which is where it matters most: `MAX_SHARE_POSES` has no upstream
        // validator, so our own writer can drop a 70-pose selection — and this is what makes that
        // visible instead of silent.
        let mut payload = payload_fixture();
        payload.insert(
            "advanced".to_owned(),
            json!({
                "steps": 8,
                "poses": (0..70).map(|_| json!({ "keypoints": [[0.1, 0.2]] })).collect::<Vec<Value>>(),
            }),
        );
        let built = build_workflow_share(&asset_fixture(), &payload);
        assert!(built.advanced.get("poses").is_none());
        assert_eq!(built.omitted, vec![OMITTED_POSES.to_owned()]);
        assert_eq!(built.advanced["steps"], json!(8));

        // And it survives its own round trip, so a shared file carries the knowledge onward.
        let envelope = serde_json::to_value(&built).expect("serializes");
        assert_eq!(parse_workflow_share(&envelope).expect("parses"), built);
    }

    /// The marker is a closed vocabulary, so it adds no surface of its own.
    #[test]
    fn the_omission_marker_admits_only_field_names_it_defined() {
        let hostile = json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "a lighthouse",
            "omitted": [
                "loras",
                "loras",
                "advanced.poses",
                "C:\\Users\\Victim\\secrets.txt",
                "../../etc/passwd",
                "\u{1b}[31mred",
                "x".repeat(100_000),
                "everything",
            ],
        });
        let share = parse_workflow_share(&hostile).expect("parses");
        // Sorted, deduplicated, and nothing that is not a name this contract defined.
        assert_eq!(
            share.omitted,
            vec![OMITTED_POSES.to_owned(), OMITTED_LORAS.to_owned()]
                .into_iter()
                .collect::<std::collections::BTreeSet<String>>()
                .into_iter()
                .collect::<Vec<String>>()
        );
        for name in &share.omitted {
            assert!(
                OMITTED_FIELDS.contains(&name.as_str()),
                "`{name}` is not in the vocabulary"
            );
        }
        // Bounded by construction: the vocabulary IS the bound, so no cap has to compose with it.
        assert!(share.omitted.len() <= OMITTED_FIELDS.len());
        assert!(
            serde_json::to_string(&share.omitted)
                .expect("serializes")
                .len()
                < 200
        );
        // Every name is a plain field path — nothing here can be a label, a path or free text.
        for name in OMITTED_FIELDS {
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '[' | ']')),
                "{name}"
            );
        }
    }

    /// An over-cap phase LoRA schedule omits the KEY, and says so.
    ///
    /// `"loras": []` is not an absence: it is the positive claim "this phase applies no LoRAs", and
    /// the phase schedule is the substance of a Krea multi-phase recipe. This is the same
    /// plausible-and-unfalsifiable record the drop doctrine refuses everywhere else.
    #[test]
    fn an_over_cap_phase_lora_schedule_is_omitted_not_emptied() {
        let advanced = json!({
            "phases": [
                { "steps": 4, "loras": (0..6).map(|index| json!({ "index": index, "weight": 0.5 })).collect::<Vec<Value>>() },
                { "steps": 6, "loras": [{ "index": 0, "weight": 0.4 }] },
            ]
        })
        .as_object()
        .cloned()
        .expect("object");
        let sanitized = sanitize_advanced(&advanced);
        let phases = sanitized["phases"].as_array().expect("phases");
        assert!(
            phases[0].get("loras").is_none(),
            "an over-cap schedule became {:?}, which reads as `applies no LoRAs`",
            phases[0].get("loras")
        );
        assert_eq!(
            phases[0]["steps"],
            json!(4),
            "the phase itself still travels"
        );
        assert_eq!(
            phases[1]["loras"].as_array().expect("array").len(),
            1,
            "the phase that was within the cap keeps its schedule"
        );
        assert_eq!(
            advanced_omissions(&advanced, &sanitized),
            vec![OMITTED_PHASE_LORAS.to_owned()]
        );

        // Through the reducer, in both directions, so the marker is on the envelope and not only in
        // a helper's return value.
        let mut payload = payload_fixture();
        payload.insert("advanced".to_owned(), Value::Object(advanced));
        let built = build_workflow_share(&asset_fixture(), &payload);
        assert_eq!(built.omitted, vec![OMITTED_PHASE_LORAS.to_owned()]);
        let envelope = serde_json::to_value(&built).expect("serializes");
        assert_eq!(parse_workflow_share(&envelope).expect("parses"), built);

        // A phase that declared NO schedule is an absence rather than an omission, so the marker
        // stays quiet — it has to mean something when it does appear.
        let quiet = json!({ "phases": [{ "steps": 4 }, { "steps": 6, "loras": [] }] })
            .as_object()
            .cloned()
            .expect("object");
        assert!(advanced_omissions(&quiet, &sanitize_advanced(&quiet)).is_empty());
    }

    /// sc-15949 review: `char::is_control` is `Cc` only, so a bidi override and the zero-width
    /// family were surviving into a recorded envelope — where sc-15952 renders them.
    #[test]
    fn prose_strips_invisible_formatting_as_well_as_control_characters() {
        // U+202E RIGHT-TO-LEFT OVERRIDE / U+202D LEFT-TO-RIGHT OVERRIDE reverse how the rest of
        // the line reads; U+200B and U+FEFF split a word invisibly; U+2028 / U+2029 are line and
        // paragraph separators the renderer never agreed to.
        let hostile = "a light\u{202e}house\u{202d} in\u{200b} fo\u{feff}g\u{2028}second\u{2029}third\u{200f}\u{00ad}\u{2060}";
        let cleaned = shareable_prose(hostile);
        assert_eq!(cleaned, "a lighthouse in fogsecondthird");
        for smuggled in [
            '\u{202e}', '\u{202d}', '\u{200b}', '\u{feff}', '\u{2028}', '\u{2029}', '\u{200f}',
            '\u{00ad}', '\u{2060}',
        ] {
            assert!(
                !cleaned.contains(smuggled),
                "U+{:04X} survived: {cleaned:?}",
                smuggled as u32
            );
        }

        // Through the real parse path, in the field a stranger's file actually fills.
        let share = parse_workflow_share(&json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": hostile,
            "advanced": { "systemMessage": hostile }
        }))
        .expect("parses");
        assert_eq!(share.prompt, "a lighthouse in fogsecondthird");
        assert_eq!(
            share.advanced["systemMessage"],
            json!("a lighthouse in fogsecondthird")
        );

        // What must NOT change: the newlines and tabs a multi-line prompt is made of, and the
        // emoji variation selectors that are prompt content rather than a display trick.
        let real = "a lighthouse\n\tshot on 35mm \u{2764}\u{fe0f}, rain \u{1f327}\u{fe0f}";
        assert_eq!(shareable_prose(real), real);
    }
}
