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
//! whenever someone adds a knob to `apps/web/src/imageJobAdvanced.js`. A deny-list leaks every
//! future field by default, so [`ADVANCED_KEY_RULES`] classifies every key the builder can emit
//! and anything unclassified is dropped. `crates/sceneworks-core/tests/workflow_share.rs`
//! parses that JS file and fails the build when a key is neither allow-listed nor explicitly
//! denied — a new knob can neither silently leak nor silently vanish.
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
    /// `schemaVersion` is missing or not a non-negative integer.
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

/// Where a key comes from, so the coverage lint knows which rules it can hold the JS builder
/// accountable for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedKeySource {
    /// Emitted by `buildImageJobAdvanced` in `apps/web/src/imageJobAdvanced.js`. The coverage
    /// lint asserts these match the JS source exactly, in both directions.
    StudioBuilder,
    /// Stamped onto `advanced` by the API after the request arrives (recipe-preset resolution
    /// in `apps/rust-api/src/generation.rs`). Classified here for the record; the lint does
    /// not expect to find them in the JS.
    Server,
}

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
    /// The pose-library selection, reduced to keypoints. Pose library ids do not travel.
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
/// The load-bearing table. `crates/sceneworks-core/tests/workflow_share.rs` parses
/// `apps/web/src/imageJobAdvanced.js` and fails when a key the builder can emit is missing
/// here — so a new knob is a compile-time decision, not a silent leak and not a silent loss.
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
        "Authored pose selection. Keypoints travel; the pose-library ids that named them do not.",
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

/// True when `value` looks like a filesystem location.
///
/// Belt to the allow-list's braces: no allow-listed `advanced` key legitimately holds a path,
/// so any value that looks like one is dropped even from a key that is otherwise in. Covers
/// POSIX absolutes, Windows drive letters (anywhere in the string, which catches
/// `"loaded from C:\\Users\\…"`), UNC shares, `file://` URLs and `~` expansions.
///
/// Deliberately NOT applied to the authored prose fields (`prompt`, `negativePrompt`,
/// `stylePrompt`, the structured prompt's `intent` / `runtimePrompt`): those are what the user
/// typed, and silently mangling a prompt because it mentions a directory would be worse than
/// the leak it prevents — the user authored it and can see it.
#[must_use]
pub fn is_path_shaped(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.to_ascii_lowercase().contains("file://") {
        return true;
    }
    if trimmed.starts_with('/') || trimmed.starts_with('\\') {
        return true;
    }
    if trimmed.starts_with("~/") || trimmed.starts_with("~\\") {
        return true;
    }
    // A drive-letter prefix anywhere: `C:\`, `d:/`.
    let bytes = trimmed.as_bytes();
    for index in 0..bytes.len().saturating_sub(2) {
        let letter = bytes[index];
        if !letter.is_ascii_alphabetic() {
            continue;
        }
        // Only when the letter starts a token, so `https://x` is not read as a drive.
        if index > 0 && (bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b':') {
            continue;
        }
        if bytes[index + 1] == b':' && (bytes[index + 2] == b'\\' || bytes[index + 2] == b'/') {
            return true;
        }
    }
    false
}

/// A Hugging Face repo id (`owner/name`) and nothing else — never a path, never a URL.
fn hf_repo_id(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let mut segments = trimmed.split('/');
    let (Some(owner), Some(name), None) = (segments.next(), segments.next(), segments.next())
    else {
        return None;
    };
    let segment_ok = |segment: &str| {
        !segment.is_empty()
            && segment.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
            })
    };
    (segment_ok(owner) && segment_ok(name)).then(|| trimmed.to_owned())
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

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

    let mode = string_field("mode").unwrap_or_else(|| asset.recipe.mode.as_str().to_owned());
    let model = string_field("model").unwrap_or_else(|| asset.recipe.model.clone());
    let prompt = string_field("prompt").unwrap_or_else(|| asset.recipe.prompt.clone());
    let negative_prompt =
        string_field("negativePrompt").unwrap_or_else(|| asset.recipe.negative_prompt.clone());

    // The sidecar's seed — not the payload's. `payload.seed` is the batch BASE and
    // `payload.seeds` is the whole batch; only the sidecar knows which one rendered the file
    // being shared. The batch list never travels: the other images are not this share's
    // business. (`Recipe::seed` is a required field, so there is nothing to fall back to.)
    let seed = Some(asset.recipe.seed);

    let advanced = sanitize_advanced(
        job_payload
            .get("advanced")
            .and_then(Value::as_object)
            .unwrap_or(&Map::new()),
    );

    WorkflowShare {
        kind: WORKFLOW_KIND_IMAGE.to_owned(),
        schema_version: WORKFLOW_SHARE_SCHEMA_VERSION,
        producer: WorkflowProducer::default(),
        mode,
        model,
        prompt,
        negative_prompt,
        seed,
        width: u32_field("width").or(asset.file.width),
        height: u32_field("height").or(asset.file.height),
        count: u32_field("count"),
        style_preset: string_field("stylePreset").filter(|value| !value.is_empty()),
        style_id: string_field("styleId").filter(|value| !value.is_empty()),
        fit_mode: string_field("fitMode").filter(|value| !value.is_empty()),
        upscale: sanitize_upscale(job_payload.get("upscale")),
        loras: sanitize_loras(job_payload.get("loras")),
        inputs: describe_inputs(job_payload),
        advanced,
    }
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
            .filter(|value| !value.trim().is_empty())
            .filter(|value| !is_path_shaped(value))
            .map(str::to_owned);
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
            .filter(|engine| !engine.is_empty() && !is_path_shaped(engine))
            .map(str::to_owned),
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
        .map(|entry| {
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
                .and_then(hf_repo_id);
            WorkflowLora {
                name: entry
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|name| !name.is_empty() && !is_path_shaped(name))
                    .map(str::to_owned),
                weight: entry.get("weight").and_then(Value::as_f64),
                repo,
            }
        })
        .filter(|lora| lora.name.is_some() || lora.weight.is_some() || lora.repo.is_some())
        .collect()
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
const PROSE_KEYS: &[&str] = &["stylePrompt", "intent", "runtimePrompt"];

fn sanitize_scalar(key: &str, value: &Value) -> Option<Value> {
    match value {
        Value::String(text) => {
            if !PROSE_KEYS.contains(&key) && is_path_shaped(text) {
                return None;
            }
            Some(Value::String(text.clone()))
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

fn sanitize_poses(value: &Value) -> Option<Value> {
    let poses = value.as_array()?;
    // The entry count is load-bearing (poses replace `count` variations), so an entry whose
    // keypoints are missing or malformed still occupies its slot as an empty object.
    let sanitized: Vec<Value> = poses
        .iter()
        .map(|pose| {
            let keypoints = pose
                .as_object()
                .and_then(|pose| pose.get("keypoints"))
                .filter(|keypoints| is_numeric_tree(keypoints));
            match keypoints {
                Some(keypoints) => {
                    let mut out = JsonObject::new();
                    out.insert("keypoints".to_owned(), keypoints.clone());
                    Value::Object(out)
                }
                None => Value::Object(JsonObject::new()),
            }
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

    let schema_version = object
        .get("schemaVersion")
        .ok_or(WorkflowShareError::MissingSchemaVersion)?
        .as_u64()
        .and_then(|version| u32::try_from(version).ok())
        .ok_or(WorkflowShareError::MissingSchemaVersion)?;
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
    // An envelope that arrived from outside is sanitized on the way IN as well: a hostile or
    // simply older writer's `advanced` is reduced by the same allow-list the builder uses.
    Ok(WorkflowShare {
        advanced: sanitize_advanced(&share.advanced),
        ..share
    })
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
        assert_eq!(sanitized.keys().collect::<Vec<_>>(), vec!["steps"]);
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
        assert_eq!(sanitized.keys().collect::<Vec<_>>(), vec!["steps"]);
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
        assert_eq!(share.advanced.keys().collect::<Vec<_>>(), vec!["steps"]);
        let encoded = serde_json::to_string(&share).expect("serializes");
        assert!(!encoded.contains("somethingNew"));
    }
}
