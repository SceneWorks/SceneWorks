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
                "This is a `{found}` SceneWorks workflow; this reader handles `{supported}` workflows."
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
    /// Why it is not classified, naming the story that owns turning it on. The lint asserts this
    /// names a `sc-` story, so a deferral cannot be a shrug.
    pub reason: &'static str,
}

/// The VIDEO builders (sc-15956 owns the video envelope and these keys).
///
/// `VideoStudio.jsx` alone emits ~15 keys nothing here classifies. That is fine only for as long
/// as no video write seam embeds: the moment sc-15956 threads a workflow into a video write, its
/// builder moves up into [`ADVANCED_BUILDERS`] and every one of those keys needs an
/// `allow`/`deny` decision. Nothing about a video lane can start embedding while its keys sit in
/// this list — that is what this list is for.
pub const DEFERRED_ADVANCED_BUILDERS: &[DeferredAdvancedBuilder] = &[
    DeferredAdvancedBuilder {
        path: "apps/web/src/screens/VideoStudio.jsx",
        function: "submit",
        reason: "The Video Studio's own ~15-knob advanced map. No video write seam embeds a \
                 workflow yet; sc-15956 owns the video envelope and must classify these keys \
                 before it does.",
    },
    DeferredAdvancedBuilder {
        path: "apps/web/src/components/editor/useEditorGeneration.js",
        function: "buildBasePayload",
        reason: "The timeline editor's shared video payload (queueTimelineVideoJob). Video lane — \
                 sc-15956 classifies it when video images start embedding.",
    },
    DeferredAdvancedBuilder {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "extendSelectedClip",
        reason: "Timeline video action: `buildBasePayload`'s advanced plus timelineAction / \
                 timelineContext. Video lane — sc-15956.",
    },
    DeferredAdvancedBuilder {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "replaceSelectedItem",
        reason: "Timeline video action: `buildBasePayload`'s advanced plus timelineAction / \
                 timelineContext. Video lane — sc-15956.",
    },
    DeferredAdvancedBuilder {
        path: "apps/web/src/screens/EditorScreen.jsx",
        function: "bridgeGap",
        reason: "Timeline video action: `buildBasePayload`'s advanced plus timelineAction / \
                 timelineContext. Video lane — sc-15956.",
    },
    DeferredAdvancedBuilder {
        path: "apps/web/src/simple/simpleJobs.js",
        function: "buildSimpleVideoRequest",
        reason: "The Simple shell's VIDEO request. Its image sibling \
                 (`buildSimpleImageRequest`) delegates to `buildImageJobRequest` and so is \
                 covered by the studio builder; this one is a video lane — sc-15956.",
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
/// Deliberately NOT applied to the authored prose fields (`prompt`, `negativePrompt`,
/// `stylePrompt`, the structured prompt's `intent` / `runtimePrompt`): those are what the user
/// typed, and silently mangling a prompt because it mentions a directory would be worse than
/// the leak it prevents — the user authored it and can see it. That exemption is pinned by
/// `authored_prose_travels_verbatim_even_when_it_names_a_path` in the integration tests.
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

/// The longest an authored prose field may be — `prompt`, `negativePrompt`, `stylePrompt` and
/// the structured prompt's `intent` / `runtimePrompt`.
///
/// Distinct from and 100× [`LABEL_MAX_CHARS`] on purpose: a label is a slug and prose is
/// paragraphs. 20,000 characters is roughly 3,000 words. The longest thing this field class
/// legitimately holds is `runtimePrompt`, the serialized structured-prompt recipe, which runs to
/// a few kilobytes at its most verbose; a hand-typed prompt is two orders of magnitude shorter.
/// So the cap cannot truncate anything real, while still bounding what an envelope from a
/// stranger can hand sc-15952 to render: five prose fields × 20 k is ~100 kB worst case, which
/// keeps a PNG text chunk a text chunk rather than a payload.
const PROSE_MAX_CHARS: usize = 20_000;

/// Authored prose, reduced. Unlike [`shareable_label`] this keeps what the user typed —
/// including a path, which is the deliberate exemption [`is_path_shaped`] documents — and only
/// bounds it.
///
/// On the BUILD side these strings are the user's own and this is a no-op. On the PARSE side
/// they are attacker-chosen, which is the epic's trust boundary: the reader (sc-15952) renders
/// them, so a value carrying ANSI escapes or several megabytes of text must not arrive intact.
/// Newlines and tabs survive — a multi-line prompt is normal and mangling it would be the very
/// harm the prose exemption exists to prevent — but every other control character is dropped.
fn shareable_prose(value: &str) -> String {
    let stripped: String = value
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .take(PROSE_MAX_CHARS)
        .collect();
    stripped.trim().to_owned()
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
    reduce_workflow_share(WorkflowShare {
        kind: WORKFLOW_KIND_IMAGE.to_owned(),
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
        style_preset: string_field("stylePreset"),
        style_id: string_field("styleId"),
        fit_mode: string_field("fitMode"),
        upscale: sanitize_upscale(job_payload.get("upscale")),
        loras: sanitize_loras(job_payload.get("loras")),
        inputs: describe_inputs(job_payload),
        advanced,
    })
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
    if has_id("sourceAssetId") {
        inputs.push(WorkflowInput {
            kind: INPUT_KIND_SOURCE.to_owned(),
            count: 1,
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

/// The four input kinds, in the order [`describe_inputs`] emits them. An `inputs[].kind`
/// outside this set is not something a reader can act on, so it does not survive a parse.
pub const INPUT_KINDS: &[&str] = &[
    INPUT_KIND_SOURCE,
    INPUT_KIND_REFERENCE,
    INPUT_KIND_MASK,
    INPUT_KIND_CONTROL,
];

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
        style_preset,
        style_id,
        fit_mode,
        upscale,
        loras,
        inputs,
        advanced,
    } = share;
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
        style_preset: style_preset.as_deref().and_then(shareable_label),
        style_id: style_id.as_deref().and_then(shareable_label),
        fit_mode: fit_mode.as_deref().and_then(shareable_label),
        upscale: upscale.map(|upscale| WorkflowUpscale {
            enabled: upscale.enabled,
            factor: upscale.factor,
            engine: upscale.engine.as_deref().and_then(shareable_label),
            softness: upscale.softness.filter(|value| value.is_finite()),
        }),
        loras: loras.into_iter().filter_map(reduce_lora).collect(),
        inputs: inputs.into_iter().filter_map(reduce_input).collect(),
        advanced: sanitize_advanced(&advanced),
    }
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
const POSE_FIELDS: &[&str] = &["keypoints", "hands", "face"];

fn sanitize_poses(value: &Value) -> Option<Value> {
    let poses = value.as_array()?;
    // The entry count is load-bearing (poses replace `count` variations), so an entry whose
    // coordinates are missing or malformed still occupies its slot as an empty object.
    let sanitized: Vec<Value> = poses
        .iter()
        .map(|pose| {
            let mut out = JsonObject::new();
            if let Some(pose) = pose.as_object() {
                for field in POSE_FIELDS {
                    if let Some(numbers) = pose.get(*field).filter(|value| is_numeric_tree(value)) {
                        out.insert((*field).to_owned(), numbers.clone());
                    }
                }
            }
            Value::Object(out)
        })
        .collect();
    (!sanitized.is_empty()).then_some(Value::Array(sanitized))
}

/// True when `value` is made only of numbers, nulls and arrays of them — the shape pose
/// keypoints have. Anything with a string in it is not keypoints and does not travel.
fn is_numeric_tree(value: &Value) -> bool {
    match value {
        Value::Number(_) | Value::Null => true,
        Value::Array(values) => values.iter().all(is_numeric_tree),
        _ => false,
    }
}

fn sanitize_phases(value: &Value) -> Option<Value> {
    let phases = value.as_array()?;
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
            // carry no id and stay meaningful next to the sanitized `loras` above.
            if let Some(loras) = phase.get("loras").and_then(Value::as_array) {
                let entries: Vec<Value> = loras
                    .iter()
                    .filter_map(Value::as_object)
                    .filter_map(|lora| {
                        let mut entry = JsonObject::new();
                        for field in ["index", "weight"] {
                            if let Some(number) = lora.get(field).filter(|value| value.is_number())
                            {
                                entry.insert(field.to_owned(), number.clone());
                            }
                        }
                        entry.contains_key("index").then_some(Value::Object(entry))
                    })
                    .collect();
                out.insert("loras".to_owned(), Value::Array(entries));
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
    if kind != WORKFLOW_KIND_IMAGE {
        return Err(WorkflowShareError::UnsupportedKind {
            found: kind.to_owned(),
            supported: WORKFLOW_KIND_IMAGE.to_owned(),
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
    Ok(reduce_workflow_share(share))
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
        assert_eq!(phases[1]["loras"].as_array().expect("array").len(), 0);
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
            assert_eq!(value.chars().count(), PROSE_MAX_CHARS);
        }

        // The bound is prose-sized, not label-sized: the two must never be conflated.
        assert_eq!(PROSE_MAX_CHARS, 20_000);
        assert_eq!(PROSE_MAX_CHARS, LABEL_MAX_CHARS * 100);
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
        assert!(prompt.chars().count() < PROSE_MAX_CHARS);

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

    #[test]
    fn parse_rejects_another_workflow_kind() {
        let video = json!({
            "sceneworksWorkflow": "video",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": PRODUCER_URL, "version": "0.8.1" },
            "mode": "image_to_video",
            "model": "wan_5b",
            "prompt": "p"
        });
        assert!(matches!(
            parse_workflow_share(&video).expect_err("video envelope"),
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
}
