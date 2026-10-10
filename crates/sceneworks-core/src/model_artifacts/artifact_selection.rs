//! Data-driven selection of a job's exact artifact requirement closure.
//!
//! Manifests declaratively describe every artifact a model can load (primary weights per tier,
//! hard co-requisites, per-platform rows). This module reduces one manifest entry plus the
//! request's own selection data (worker platform, requested tier) to the exact
//! [`ExternalArtifactRequirement`] closure that one runtime load will open. It is shared by the
//! API's submission preflight/catalog listing and the worker's pre-loader guard, so both judge
//! the same closure through one code path — never a per-route or per-model reimplementation.
//!
//! Completeness rules that reviews pinned down and this module owns:
//! - the closure is computed for the **selected** platform, never the API host's platform;
//! - the closure covers the **selected** variant/tier only — sibling-variant receipts are never
//!   unioned into (or out of) the answer;
//! - receipt-recorded exact files take precedence, with declared exact (non-glob) manifest files
//!   as the fallback so never-downloaded installs still carry a checkable identity.

use super::external_library::ExternalArtifactRequirement;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The `<safe>` in the API/worker's managed `models/<safe>` download directory. Byte-identical to
/// the API (`apps/rust-api/src/lib.rs::safe_download_dir`) and the worker
/// (`sceneworks_worker::paths::safe_download_dir`); receipts written by either are read here.
pub fn safe_download_dir(value: &str) -> String {
    let mut output = String::new();
    let mut in_replacement = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            output.push(character);
            in_replacement = false;
        } else if !in_replacement {
            output.push_str("__");
            in_replacement = true;
        }
    }
    let output = output.trim_matches('_').to_owned();
    if output.is_empty() {
        "download".to_owned()
    } else {
        output
    }
}

/// True when a download entry names an artifact this contract can resolve (a Hugging Face repo).
pub fn is_supported_model_download(download: &Value) -> bool {
    download.get("provider").and_then(Value::as_str) == Some("huggingface")
        && download
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(|repo| !repo.is_empty())
}

/// The null SHA a DECLARED-but-unpublished download row carries as its `revision` (sc-24112).
///
/// Schema-valid 40-hex, so the manifest still type-checks, and unmistakably not a commit — the same
/// trick git itself uses for "no object". It is rigidly paired with
/// [`is_pending_artifact_download`]: the manifest audit fails closed on a null SHA without the flag
/// AND on the flag with a real SHA, so pinning the real revision and dropping the flag is one edit
/// that cannot be half-done.
pub const PENDING_ARTIFACT_REVISION: &str = "0000000000000000000000000000000000000000";

/// True when a download row is DECLARED but its artifact is not published yet (sc-24112).
///
/// A pending row is enumerated by the catalog — the tier axis and its picker are real before the
/// bytes exist, which is what lets the memory ladder, the fit gates and the UI be built and tested
/// against the tier that is coming — but it is never queued for download, never counted as
/// installable, and never the `default`. Without this, a user picking the tier would queue a fetch
/// of a revision that does not resolve and see an opaque download failure.
pub fn is_pending_artifact_download(download: &Value) -> bool {
    download.get("pendingArtifact").and_then(Value::as_bool) == Some(true)
}

/// A tier row's `localDerivation` block (sc-22998): the tier is derived on the user's machine from
/// the `from_variant` original rather than downloaded, and the derivation must reproduce exactly
/// `weights_file` of `weights_bytes` / `weights_sha256`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalDerivation {
    pub from_variant: String,
    pub conversion: String,
    pub weights_file: String,
    pub weights_bytes: u64,
    pub weights_sha256: String,
}

/// The row's [`LocalDerivation`], if it declares one. A malformed block (a missing field) is `None`
/// here and a schema/audit failure upstream; a caller that must not treat such a row as an ordinary
/// download checks for the raw `localDerivation` key instead (see [`declares_local_derivation`]).
pub fn local_derivation(download: &Value) -> Option<LocalDerivation> {
    let block = download.get("localDerivation")?;
    let text = |key: &str| {
        block
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    Some(LocalDerivation {
        from_variant: text("fromVariant")?,
        conversion: text("conversion")?,
        weights_file: text("weightsFile")?,
        weights_bytes: block.get("weightsBytes").and_then(Value::as_u64)?,
        weights_sha256: text("weightsSha256")?.to_ascii_lowercase(),
    })
}

/// True when the row carries a `localDerivation` block at all, well-formed or not. Such a row is
/// never an ordinary download: its `files` name the ORIGINAL, not the tier.
pub fn declares_local_derivation(download: &Value) -> bool {
    download.get("localDerivation").is_some()
}

fn safe_path_segment(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()
        && value != "."
        && value != ".."
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
    .then_some(value)
}

/// Where a locally derived tier snapshot lives (sc-22998):
/// `<data_dir>/models/derived/<model id>/<variant>/<conversion>/`. The conversion is part of the
/// path because a different conversion writes different bytes — a snapshot from another conversion
/// is a different artifact, never this tier. `None` when any segment is not a plain path segment.
/// The deriver (the worker's audio-lane preparer, sc-22999) writes here; the catalog only reads.
pub fn local_derivation_snapshot_dir(
    data_dir: &Path,
    model_id: &str,
    variant: &str,
    derivation: &LocalDerivation,
) -> Option<PathBuf> {
    Some(
        data_dir
            .join("models")
            .join("derived")
            .join(safe_path_segment(model_id)?)
            .join(safe_path_segment(variant)?)
            .join(safe_path_segment(&derivation.conversion)?),
    )
}

/// What is at a derived tier's snapshot location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DerivedSnapshotState {
    /// Nothing has been derived there.
    Absent,
    /// The weights file matches the pinned size and SHA-256.
    Verified,
    /// Something is there, but it is not the pinned derivation (why).
    Invalid(String),
}

type DigestKey = (PathBuf, u64, Option<std::time::SystemTime>);

/// SHA-256 of `path`, memoized per (path, size, mtime) for the life of the process, so a catalog
/// read does not re-hash a multi-GB derived weights file that has not changed.
fn memoized_file_sha256(path: &Path, len: u64) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    static CACHE: std::sync::OnceLock<std::sync::Mutex<BTreeMap<DigestKey, String>>> =
        std::sync::OnceLock::new();
    let modified = std::fs::metadata(path)?.modified().ok();
    let key = (path.to_path_buf(), len, modified);
    let cache = CACHE.get_or_init(Default::default);
    if let Some(hit) = cache.lock().ok().and_then(|map| map.get(&key).cloned()) {
        return Ok(hit);
    }
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = format!("{:x}", hasher.finalize());
    if let Ok(mut map) = cache.lock() {
        map.insert(key, digest.clone());
    }
    Ok(digest)
}

/// Judge the derived snapshot at `dir` against `derivation`'s pinned weights identity. Only the
/// weights file is checked here; the engine re-verifies the whole snapshot (every copied file and
/// its conversion manifest) when it loads it.
pub fn derived_snapshot_state(dir: &Path, derivation: &LocalDerivation) -> DerivedSnapshotState {
    let Some(file_name) = safe_path_segment(&derivation.weights_file) else {
        return DerivedSnapshotState::Invalid(format!(
            "weights file name '{}' is not a plain file name",
            derivation.weights_file
        ));
    };
    if !dir.is_dir() {
        return DerivedSnapshotState::Absent;
    }
    let weights = dir.join(file_name);
    let Ok(metadata) = std::fs::metadata(&weights) else {
        return DerivedSnapshotState::Invalid(format!("{file_name} is missing"));
    };
    if !metadata.is_file() || metadata.len() != derivation.weights_bytes {
        return DerivedSnapshotState::Invalid(format!(
            "{file_name} is {} bytes, the pinned derivation is {}",
            metadata.len(),
            derivation.weights_bytes
        ));
    }
    match memoized_file_sha256(&weights, metadata.len()) {
        Ok(digest) if digest == derivation.weights_sha256 => DerivedSnapshotState::Verified,
        Ok(digest) => DerivedSnapshotState::Invalid(format!(
            "{file_name} hashes to {digest}, the pinned derivation is {}",
            derivation.weights_sha256
        )),
        Err(error) => DerivedSnapshotState::Invalid(format!("{file_name} is unreadable: {error}")),
    }
}

/// True when a download entry is a co-requisite dependency (sc-9696): fetched ALONGSIDE the
/// primary download rather than as a pick-one alternate.
pub fn is_co_requisite_download(download: &Value) -> bool {
    download.get("coRequisite").and_then(Value::as_bool) == Some(true)
}

/// Whether a co-requisite row is scoped to ONE quant tier (sc-14980).
pub fn co_requisite_variant(download: &Value) -> Option<String> {
    download
        .get("variant")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
}

/// Drop download rows whose `platforms` list excludes `os`. Rows without a `platforms` key apply
/// everywhere. A manifest with no platform-scoped row at all is left untouched.
pub fn retain_downloads_for_os(model: &mut Value, os: &str) {
    let Some(downloads) = model.get_mut("downloads").and_then(Value::as_array_mut) else {
        return;
    };
    if !downloads
        .iter()
        .any(|entry| entry.get("platforms").is_some())
    {
        return;
    }
    downloads.retain(
        |entry| match entry.get("platforms").and_then(Value::as_array) {
            Some(platforms) => platforms.iter().any(|p| p.as_str() == Some(os)),
            None => true,
        },
    );
}

/// The model's canonical (default-or-first) primary download entry.
pub fn model_download(model: &Value) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let mut fallback = None;
    for download in downloads {
        if !is_supported_model_download(download) || is_co_requisite_download(download) {
            continue;
        }
        fallback.get_or_insert(download);
        if download.get("default").and_then(Value::as_bool) == Some(true) {
            return Some(download.clone());
        }
    }
    fallback.cloned()
}

/// The primary download entry whose `variant` matches `variant` (case-insensitive).
pub fn model_download_for_variant(model: &Value, variant: &str) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let wanted = variant.trim().to_ascii_lowercase();
    downloads
        .iter()
        .find(|download| {
            is_supported_model_download(download)
                && !is_co_requisite_download(download)
                && download
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(|value| value.trim().to_ascii_lowercase())
                    .as_deref()
                    == Some(wanted.as_str())
        })
        .cloned()
}

/// Every provider-supported co-requisite download row of `model`.
pub fn model_co_requisite_downloads(model: &Value) -> Vec<Value> {
    model
        .get("downloads")
        .and_then(Value::as_array)
        .map(|downloads| {
            downloads
                .iter()
                .filter(|download| {
                    is_co_requisite_download(download) && is_supported_model_download(download)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Every co-requisite row that applies to `variant` (sc-14980) — tier-agnostic rows always apply,
/// a tier-scoped row only to its own tier — INCLUDING every option of every choice group
/// ([`co_requisite_choice`]). Install-state checks read this, because any installed option of a
/// group satisfies it; what an install QUEUES is [`model_co_requisite_downloads_for_selection`].
pub fn model_co_requisite_downloads_for_variant_all_options(
    model: &Value,
    variant: Option<&str>,
) -> Vec<Value> {
    let wanted = variant.map(|value| value.trim().to_ascii_lowercase());
    model_co_requisite_downloads(model)
        .into_iter()
        .filter(
            |download| match (co_requisite_variant(download), wanted.as_deref()) {
                (None, _) => true,
                (Some(_), None) => true,
                (Some(row), Some(wanted)) => row == wanted,
            },
        )
        .collect()
}

/// The co-requisite downloads that apply to `variant` (sc-14980), with every choice group
/// ([`co_requisite_choice`], sc-22998) narrowed to its DEFAULT option. A model without choice rows
/// gets exactly [`model_co_requisite_downloads_for_variant_all_options`].
///
/// A group whose default is malformed (none, or several — the manifest audit forbids both) keeps
/// every option rather than silently dropping a dependency the load may need.
pub fn model_co_requisite_downloads_for_variant(
    model: &Value,
    variant: Option<&str>,
) -> Vec<Value> {
    let rows = model_co_requisite_downloads_for_variant_all_options(model, variant);
    let defaults = default_choice_options(&rows);
    rows.into_iter()
        .filter(|row| match co_requisite_choice(row) {
            None => true,
            Some(choice) => match defaults.get(&choice.group) {
                Some(Some(option)) => *option == choice.option,
                Some(None) | None => true,
            },
        })
        .collect()
}

/// One option of a user choice among co-requisites (sc-22998): YuE2's decoder is the group
/// `decoder` with the options `standard` (default) and `legacy`. Declared by a co-requisite row's
/// `choice` block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoRequisiteChoice {
    pub group: String,
    pub option: String,
    pub default: bool,
}

/// The choice a co-requisite row is an option of, if any.
pub fn co_requisite_choice(download: &Value) -> Option<CoRequisiteChoice> {
    if !is_co_requisite_download(download) {
        return None;
    }
    let choice = download.get("choice")?.as_object()?;
    let group = choice.get("group")?.as_str()?.trim();
    let option = choice.get("option")?.as_str()?.trim();
    if group.is_empty() || option.is_empty() {
        return None;
    }
    Some(CoRequisiteChoice {
        group: group.to_owned(),
        option: option.to_owned(),
        default: choice.get("default").and_then(Value::as_bool) == Some(true),
    })
}

/// Per group: `Some(option)` when exactly one option is the default, `None` when the declaration
/// is malformed (no default, or several).
fn default_choice_options(rows: &[Value]) -> BTreeMap<String, Option<String>> {
    let mut defaults: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for choice in rows.iter().filter_map(co_requisite_choice) {
        let entry = defaults.entry(choice.group).or_default();
        if choice.default {
            entry.push(choice.option);
        }
    }
    defaults
        .into_iter()
        .map(|(group, mut options)| {
            let option = (options.len() == 1).then(|| options.remove(0));
            (group, option)
        })
        .collect()
}

/// Why a requested co-requisite choice cannot be honoured. Never answered by substituting another
/// option: the caller refuses the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoRequisiteChoiceError {
    /// The model declares no choice group of this name.
    UnknownGroup {
        group: String,
        declared: Vec<String>,
    },
    /// The group declares no such option.
    UnknownOption {
        group: String,
        option: String,
        declared: Vec<String>,
    },
    /// The request left the group unset and the manifest does not declare exactly one default.
    NoSingleDefault { group: String },
}

impl std::fmt::Display for CoRequisiteChoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownGroup { group, declared } => write!(
                f,
                "this model has no '{group}' choice (it declares: {})",
                if declared.is_empty() {
                    "none".to_owned()
                } else {
                    declared.join(", ")
                }
            ),
            Self::UnknownOption {
                group,
                option,
                declared,
            } => write!(
                f,
                "'{option}' is not a '{group}' option of this model (options: {})",
                declared.join(", ")
            ),
            Self::NoSingleDefault { group } => write!(
                f,
                "the '{group}' choice declares no single default option; name one explicitly"
            ),
        }
    }
}

impl std::error::Error for CoRequisiteChoiceError {}

/// Resolve every choice group the model's co-requisites for `variant` declare to exactly one
/// option: the requested one, else the group's default. `requested` keys and values are matched
/// exactly (trimmed); an unknown group or option is an error, never a fallback to another option.
pub fn resolve_co_requisite_choices(
    model: &Value,
    variant: Option<&str>,
    requested: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, CoRequisiteChoiceError> {
    let rows = model_co_requisite_downloads_for_variant_all_options(model, variant);
    let mut options: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for choice in rows.iter().filter_map(co_requisite_choice) {
        options.entry(choice.group).or_default().push(choice.option);
    }
    for (group, option) in requested {
        let group = group.trim();
        let Some(declared) = options.get(group) else {
            return Err(CoRequisiteChoiceError::UnknownGroup {
                group: group.to_owned(),
                declared: options.keys().cloned().collect(),
            });
        };
        if !declared.iter().any(|candidate| candidate == option.trim()) {
            return Err(CoRequisiteChoiceError::UnknownOption {
                group: group.to_owned(),
                option: option.trim().to_owned(),
                declared: declared.clone(),
            });
        }
    }
    let defaults = default_choice_options(&rows);
    options
        .keys()
        .map(|group| {
            let requested_option = requested
                .iter()
                .find(|(key, _)| key.trim() == group)
                .map(|(_, option)| option.trim().to_owned());
            match requested_option.or_else(|| defaults.get(group).cloned().flatten()) {
                Some(option) => Ok((group.clone(), option)),
                None => Err(CoRequisiteChoiceError::NoSingleDefault {
                    group: group.clone(),
                }),
            }
        })
        .collect()
}

/// How much of a co-requisite row's snapshot is on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoRequisitePresence {
    /// Every declared file is cached.
    Installed,
    /// Some, but not all, declared files are cached: a started install of this option.
    Incomplete,
    /// Nothing of it is cached.
    Absent,
}

/// The co-requisite rows whose state gates an install (sc-22998): every non-choice row, plus ONE row
/// per choice group. That row is, in order: the first INSTALLED option (the group is satisfied);
/// else the first INCOMPLETE option — the one the user already started, so a repair completes it
/// rather than fetching a different decoder; else the group's default option (what a fresh install
/// fetches). A group with a malformed default and nothing on disk keeps every option, so nothing is
/// reported satisfied that is not. Order is preserved.
pub fn co_requisite_rows_gating_install(
    rows: Vec<Value>,
    presence: impl Fn(&Value) -> CoRequisitePresence,
) -> Vec<Value> {
    let mut installed: BTreeMap<String, String> = BTreeMap::new();
    let mut started: BTreeMap<String, String> = BTreeMap::new();
    for row in &rows {
        if let Some(choice) = co_requisite_choice(row) {
            match presence(row) {
                CoRequisitePresence::Installed => {
                    installed.entry(choice.group).or_insert(choice.option);
                }
                CoRequisitePresence::Incomplete => {
                    started.entry(choice.group).or_insert(choice.option);
                }
                CoRequisitePresence::Absent => {}
            }
        }
    }
    let defaults = default_choice_options(&rows);
    rows.into_iter()
        .filter(|row| match co_requisite_choice(row) {
            None => true,
            Some(choice) => match installed
                .get(&choice.group)
                .or_else(|| started.get(&choice.group))
            {
                Some(option) => *option == choice.option,
                None => match defaults.get(&choice.group) {
                    Some(Some(option)) => *option == choice.option,
                    Some(None) | None => true,
                },
            },
        })
        .collect()
}

/// The option of each choice group an install/REPAIR request that names no choice should fetch
/// (sc-22998): the group's [`co_requisite_rows_gating_install`] row — an installed or partially
/// installed option wins over the default, so repairing a half-downloaded legacy decoder completes
/// it instead of fetching the standard one. Groups the request names keep the request's option.
pub fn co_requisite_choices_for_repair(
    model: &Value,
    variant: Option<&str>,
    requested: &BTreeMap<String, String>,
    presence: impl Fn(&Value) -> CoRequisitePresence,
) -> BTreeMap<String, String> {
    let mut choices = requested.clone();
    for row in co_requisite_rows_gating_install(
        model_co_requisite_downloads_for_variant_all_options(model, variant),
        presence,
    ) {
        if let Some(choice) = co_requisite_choice(&row) {
            if !requested.keys().any(|group| group.trim() == choice.group) {
                choices.entry(choice.group).or_insert(choice.option);
            }
        }
    }
    choices
}

/// The co-requisite downloads an install of `variant` with the resolved `choices` queues: every
/// non-choice row for the tier, plus exactly the chosen option of each group. `choices` must come
/// from [`resolve_co_requisite_choices`]; a group it does not name keeps no option.
pub fn model_co_requisite_downloads_for_selection(
    model: &Value,
    variant: Option<&str>,
    choices: &BTreeMap<String, String>,
) -> Vec<Value> {
    model_co_requisite_downloads_for_variant_all_options(model, variant)
        .into_iter()
        .filter(|row| match co_requisite_choice(row) {
            None => true,
            Some(choice) => choices.get(&choice.group) == Some(&choice.option),
        })
        .collect()
}

fn string_array_field(payload: &Value, field: &str) -> Vec<String> {
    payload
        .get(field)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn pattern_matches(pattern: &str, value: &str) -> bool {
    let (pattern, value) = if cfg!(windows) {
        (pattern.to_ascii_lowercase(), value.to_ascii_lowercase())
    } else {
        (pattern.to_owned(), value.to_owned())
    };
    glob::Pattern::new(&pattern).is_ok_and(|pattern| pattern.matches(&value))
}

fn allow_pattern_matches(path: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    patterns
        .iter()
        .any(|pattern| pattern_matches(pattern, path))
}

/// The tier/variant the request actually selected, from the request's own data: an explicit
/// `variant`, an advanced `quantTier`, or the advanced `mlxQuantize` bit width. This is
/// request-shape vocabulary, never a model-name mapping.
pub fn requested_runtime_variant(payload: &Map<String, Value>) -> Option<String> {
    if let Some(variant) = payload
        .get("variant")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(variant.to_ascii_lowercase());
    }
    let advanced = payload.get("advanced").and_then(Value::as_object)?;
    if let Some(tier) = advanced
        .get("quantTier")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(tier.to_ascii_lowercase());
    }
    let bits = advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse::<i64>().ok())
    })?;
    Some(
        match bits {
            1..=4 => "q4",
            5..=8 => "q8",
            _ => "bf16",
        }
        .to_owned(),
    )
}

/// Reduce a manifest entry to the exact primary tier and hard co-requisites a worker on
/// `platform` will load. The public catalog may retain every installable tier, but source
/// availability is a runtime decision and must never union receipts from sibling variants or
/// filter co-requisites by the API host OS.
pub fn selected_model_artifact_closure(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
) -> Value {
    select_model_artifact_closure(model, platform, requested_variant, None)
}

/// [`selected_model_artifact_closure`] with each co-requisite choice group narrowed to the
/// option `choices` names (sc-22998) instead of its default. `choices` must come from
/// [`resolve_co_requisite_choices`]; a group it leaves out contributes no option.
pub fn selected_model_artifact_closure_with_choices(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
    choices: &BTreeMap<String, String>,
) -> Value {
    select_model_artifact_closure(model, platform, requested_variant, Some(choices))
}

fn select_model_artifact_closure(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
    choices: Option<&BTreeMap<String, String>>,
) -> Value {
    let mut selected = model.clone();
    retain_downloads_for_os(&mut selected, platform);
    let primary = requested_variant
        .and_then(|variant| model_download_for_variant(&selected, variant))
        .or_else(|| model_download(&selected));
    let selected_variant = primary
        .as_ref()
        .and_then(|download| download.get("variant"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let co_requisites = match choices {
        None => model_co_requisite_downloads_for_variant(&selected, selected_variant.as_deref()),
        Some(choices) => model_co_requisite_downloads_for_selection(
            &selected,
            selected_variant.as_deref(),
            choices,
        ),
    };
    let mut downloads = primary.into_iter().collect::<Vec<_>>();
    downloads.extend(
        co_requisites
            .into_iter()
            .filter(|download| download.get("required").and_then(Value::as_str) != Some("soft")),
    );
    if let Some(object) = selected.as_object_mut() {
        object.insert("downloads".to_owned(), Value::Array(downloads));
    }
    selected
}

struct ReceiptFileSet {
    files: Vec<String>,
    revision: Option<String>,
    variant: Option<String>,
}

fn receipt_entries(managed_path: &Path) -> Vec<Value> {
    let Ok(bytes) = std::fs::read(managed_path.join(".sceneworks-download-complete.json")) else {
        return Vec::new();
    };
    let Ok(receipt) = serde_json::from_slice::<Value>(&bytes) else {
        return Vec::new();
    };
    receipt
        .get("receipts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![receipt])
}

fn receipt_file_sets(
    managed_path: &Path,
    repo: &str,
    model_id: Option<&str>,
) -> Vec<ReceiptFileSet> {
    receipt_entries(managed_path)
        .into_iter()
        .filter_map(|entry| {
            if entry.get("repo").and_then(Value::as_str) != Some(repo) {
                return None;
            }
            // Shared repos back multiple catalog cards. A model-specific receipt protects only
            // the card that produced it; receipts predating modelId remain generic.
            if let (Some(expected), Some(actual)) =
                (model_id, entry.get("modelId").and_then(Value::as_str))
            {
                if actual != expected {
                    return None;
                }
            }
            let files = entry
                .get("resolvedFiles")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let revision = entry
                .get("snapshotRevision")
                .and_then(Value::as_str)
                .filter(|revision| !revision.trim().is_empty())
                .map(str::to_owned);
            let variant = entry
                .get("variant")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            (!files.is_empty()).then_some(ReceiptFileSet {
                files,
                revision,
                variant,
            })
        })
        .collect()
}

/// Requirements recovered from durable install receipts for the (already tier/platform-selected)
/// closure `model`. Receipts record the exact resolved file set of each completed install, so
/// this is the strongest install identity available: it survives a disconnected library without
/// re-reading a single artifact byte.
pub fn receipt_requirements_for_model(
    model: &Value,
    data_dir: &Path,
) -> Vec<ExternalArtifactRequirement> {
    let model_id = model.get("id").and_then(Value::as_str);
    let mut requirements = Vec::new();
    let downloads = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| is_supported_model_download(download));
    for download in downloads {
        if is_co_requisite_download(download)
            && download.get("required").and_then(Value::as_str) == Some("soft")
        {
            continue;
        }
        let Some(repo) = download.get("repo").and_then(Value::as_str) else {
            continue;
        };
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        let receipt_model_id = (!is_co_requisite_download(download))
            .then_some(model_id)
            .flatten();
        let variant = download
            .get("variant")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("default")
            .to_owned();
        let declared_files = string_array_field(download, "files");
        for receipt in receipt_file_sets(&managed, repo, receipt_model_id) {
            // A receipt belongs to this row only when its recorded variant matches, or — for
            // legacy variant-less receipts — when its files fall inside this row's declared
            // patterns. Sibling-tier receipts never leak into the selected closure.
            if receipt
                .variant
                .as_deref()
                .is_some_and(|receipt_variant| receipt_variant != variant)
                || (receipt.variant.is_none()
                    && !declared_files.is_empty()
                    && !receipt
                        .files
                        .iter()
                        .any(|file| allow_pattern_matches(file, &declared_files)))
            {
                continue;
            }
            let requirement = ExternalArtifactRequirement {
                repository: repo.to_owned(),
                revision: receipt.revision,
                variant: variant.clone(),
                files: receipt.files.into_iter().map(PathBuf::from).collect(),
                is_primary: !is_co_requisite_download(download),
            };
            if !requirements.contains(&requirement) {
                requirements.push(requirement);
            }
        }
    }
    requirements.sort_by(|left, right| {
        (&left.repository, &left.revision, &left.variant, &left.files).cmp(&(
            &right.repository,
            &right.revision,
            &right.variant,
            &right.files,
        ))
    });
    requirements
}

/// Requirements the manifest itself declares exactly: an immutable revision and a concrete
/// (non-glob) file list. This is the fallback identity for installs that predate receipts.
pub fn declared_exact_requirements_for_model(model: &Value) -> Vec<ExternalArtifactRequirement> {
    let mut requirements = Vec::new();
    for download in model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| is_supported_model_download(download))
        .filter(|download| download.get("required").and_then(Value::as_str) != Some("soft"))
    {
        let Some(repository) = download.get("repo").and_then(Value::as_str) else {
            continue;
        };
        let Some(revision) = download
            .get("revision")
            .and_then(Value::as_str)
            .filter(|revision| {
                revision.len() == 40
                    && revision
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
        else {
            continue;
        };
        let files = string_array_field(download, "files");
        if files.is_empty()
            || files.iter().any(|file| {
                file.bytes()
                    .any(|byte| matches!(byte, b'*' | b'?' | b'[' | b']'))
            })
        {
            continue;
        }
        requirements.push(ExternalArtifactRequirement {
            repository: repository.to_owned(),
            revision: Some(revision.to_owned()),
            variant: download
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_owned(),
            files: files.into_iter().map(PathBuf::from).collect(),
            is_primary: !is_co_requisite_download(download),
        });
    }
    requirements
}

/// One selected requirement closure plus the strength of its install evidence. Only
/// receipt-backed closures prove a completed installation; a declared-exact closure carries a
/// checkable identity but proves nothing about install state, so it must never produce the
/// typed "installed — external library unavailable" condition on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedRequirements {
    pub requirements: Vec<ExternalArtifactRequirement>,
    /// True when at least one requirement came from a durable download receipt.
    pub receipt_backed: bool,
}

/// The exact selected requirement closure for one (already tier/platform-selected) manifest
/// closure: receipt-backed requirements first, declared-exact fallbacks for rows without a
/// receipt, canonically ordered with the primary first. Returns an empty closure when the model
/// carries no checkable install identity (no receipt and no exact declaration) or when the
/// closure would not contain exactly one primary.
pub fn selected_requirements_for_closure(
    selected: &Value,
    data_dir: &Path,
) -> SelectedRequirements {
    let mut requirements = receipt_requirements_for_model(selected, data_dir);
    let receipt_backed = !requirements.is_empty();
    for declared in declared_exact_requirements_for_model(selected) {
        if !requirements.iter().any(|requirement| {
            requirement.repository == declared.repository
                && requirement.variant == declared.variant
                && requirement.is_primary == declared.is_primary
        }) {
            requirements.push(declared);
        }
    }
    if requirements
        .iter()
        .filter(|requirement| requirement.is_primary)
        .count()
        != 1
    {
        return SelectedRequirements {
            requirements: Vec::new(),
            receipt_backed: false,
        };
    }
    requirements.sort_by(|left, right| {
        (
            !left.is_primary,
            &left.repository,
            &left.variant,
            &left.files,
        )
            .cmp(&(
                !right.is_primary,
                &right.repository,
                &right.variant,
                &right.files,
            ))
    });
    SelectedRequirements {
        requirements,
        receipt_backed,
    }
}

/// The schema version of [`LocalCacheEligibility`], so a client can branch on shape rather than
/// on the presence of individual keys.
pub const LOCAL_CACHE_ELIGIBILITY_SCHEMA_VERSION: u32 = 1;

/// How much of a model the resolved model cache can ever serve from a local copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalCacheCoverage {
    /// Everything the selected closure needs can be held locally.
    Full,
    /// The primary can be held locally, but part of the declared closure never enters it, so a
    /// request that needs the excluded part still reads from the source library.
    Partial,
    /// Nothing can be served locally, however much of it is copied.
    None,
}

/// Why a model's local copy cannot cover it. Typed so the UI branches on the reason instead of
/// parsing prose, which is the same discipline `ModelAvailability` established.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalCacheExclusion {
    /// The model declares `required: "soft"` co-requisites. Closure selection drops them
    /// (`selected_model_artifact_closure`, `receipt_requirements_for_model`,
    /// `declared_exact_requirements_for_model`), so they are never promoted and never served
    /// locally, while the primary is — which is exactly how a "local copy" badge over-promises.
    OptionalComponentsExcluded,
    /// A requirement carries no recorded snapshot revision, which makes its WHOLE repository
    /// unserveable from the local tier: with no revision there is no pair to compare coverage
    /// against. Promotion can still build the bundle, so such a model can occupy cache bytes it
    /// will never be served from.
    UnpinnedRevision,
}

/// What a model's local copy can and cannot cover, and why (sc-19712 F-5).
///
/// The epic's acceptance criteria require unsupported artifact classes to be identified in the
/// product rather than silently bypassed. Before this, a model with soft co-requisites or a
/// revision-less requirement showed the same "local copy" affordance as a fully cacheable one and
/// the exclusion was visible only as a `resolved_cache_local_tier_not_selected` log line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LocalCacheEligibility {
    pub schema_version: u32,
    pub coverage: LocalCacheCoverage,
    /// `None` exactly when `coverage` is [`LocalCacheCoverage::Full`].
    pub reason: Option<LocalCacheExclusion>,
    /// A short sentence naming what is excluded. Supporting copy, never the thing to branch on.
    pub detail: Option<String>,
}

impl LocalCacheEligibility {
    fn full() -> Self {
        Self {
            schema_version: LOCAL_CACHE_ELIGIBILITY_SCHEMA_VERSION,
            coverage: LocalCacheCoverage::Full,
            reason: None,
            detail: None,
        }
    }

    fn excluded(coverage: LocalCacheCoverage, reason: LocalCacheExclusion, detail: String) -> Self {
        Self {
            schema_version: LOCAL_CACHE_ELIGIBILITY_SCHEMA_VERSION,
            coverage,
            reason: Some(reason),
            detail: Some(detail),
        }
    }
}

/// True when the model declares at least one optional (`required: "soft"`) co-requisite download.
/// These are filtered out of every requirement closure this module builds, so they can never be
/// promoted — the check is deliberately made against the RAW entry rather than a selected closure,
/// because by the time selection has run the evidence is already gone.
pub fn declares_optional_co_requisites(model: &Value) -> bool {
    model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|download| {
            is_co_requisite_download(download)
                && download.get("required").and_then(Value::as_str) == Some("soft")
        })
}

/// The one judgement of what a model's local copy can cover, shared by every surface that shows it.
///
/// Ordering matters: an unpinned revision beats an excluded optional component, because it is the
/// stronger statement — a model can be excluded on both counts (a real install of `qwen_image` is),
/// and reporting "some components are excluded" for something that can serve nothing at all would
/// still over-promise.
pub fn local_cache_eligibility_for_model(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
    data_dir: &Path,
) -> LocalCacheEligibility {
    let selected = selected_requirements_for_model(model, platform, requested_variant, data_dir);
    if let Some(unpinned) = selected
        .requirements
        .iter()
        .find(|requirement| requirement.revision.is_none())
    {
        return LocalCacheEligibility::excluded(
            LocalCacheCoverage::None,
            LocalCacheExclusion::UnpinnedRevision,
            format!(
                "No snapshot revision was recorded for {}, so no local copy of this model can be \
                 used and it will always load from the model library.",
                unpinned.repository
            ),
        );
    }
    if declares_optional_co_requisites(model) {
        return LocalCacheEligibility::excluded(
            LocalCacheCoverage::Partial,
            LocalCacheExclusion::OptionalComponentsExcluded,
            "Optional components of this model are never copied locally, so a request that needs \
             one still reads from the model library."
                .to_owned(),
        );
    }
    LocalCacheEligibility::full()
}

/// One-call composition: select the exact closure for (`platform`, `requested_variant`), then
/// compute its requirement list. This is the function the API seam and the worker guard share.
pub fn selected_requirements_for_model(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
    data_dir: &Path,
) -> SelectedRequirements {
    let selected = selected_model_artifact_closure(model, platform, requested_variant);
    selected_requirements_for_closure(&selected, data_dir)
}

/// [`selected_requirements_for_model`] for a request that names co-requisite choices
/// (sc-22998: a YuE2 job that decodes with the legacy VAE). `choices` must come from
/// [`resolve_co_requisite_choices`].
pub fn selected_requirements_for_model_with_choices(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
    choices: &BTreeMap<String, String>,
    data_dir: &Path,
) -> SelectedRequirements {
    let selected =
        selected_model_artifact_closure_with_choices(model, platform, requested_variant, choices);
    selected_requirements_for_closure(&selected, data_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const REV_A: &str = "0123456789abcdef0123456789abcdef01234567";
    const REV_B: &str = "89abcdef0123456789abcdef0123456789abcdef";

    fn write_receipts(data_dir: &Path, repo: &str, receipts: Value) {
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({ "receipts": receipts })).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn receipt_requirements_preserve_exact_variant_file_closures() {
        let temp = tempfile::tempdir().unwrap();
        write_receipts(
            temp.path(),
            "owner/matrix",
            json!([
                {"repo":"owner/matrix", "modelId":"matrix", "variant":"q4",
                 "resolvedFiles":["q4/model.safetensors"], "snapshotRevision": REV_A},
                {"repo":"owner/matrix", "modelId":"matrix", "variant":"q8",
                 "resolvedFiles":["q8/model.safetensors"], "snapshotRevision": REV_B}
            ]),
        );
        let model = json!({
            "id": "matrix",
            "downloads": [
                {"provider":"huggingface", "repo":"owner/matrix", "variant":"q4", "default":true, "files":["q4/*"]},
                {"provider":"huggingface", "repo":"owner/matrix", "variant":"q8", "files":["q8/*"]}
            ]
        });

        // The un-selected model still yields one requirement per installed variant.
        let all = receipt_requirements_for_model(&model, temp.path());
        assert_eq!(all.len(), 2);

        // Selecting q4 must judge ONLY the q4 closure: exactly one requirement, primary, and the
        // sibling q8 receipt must not appear (a model-wide union would mark q4 incomplete when
        // q8 was removed, or vice versa).
        let q4 =
            selected_requirements_for_model(&model, std::env::consts::OS, Some("q4"), temp.path());
        assert!(q4.receipt_backed);
        let q4 = q4.requirements;
        assert_eq!(q4.len(), 1);
        assert!(q4[0].is_primary);
        assert_eq!(q4[0].variant, "q4");
        assert_eq!(q4[0].files, [PathBuf::from("q4/model.safetensors")]);
        assert_eq!(q4[0].revision.as_deref(), Some(REV_A));

        // Removing the q8 receipt leaves the q4 selection untouched.
        write_receipts(
            temp.path(),
            "owner/matrix",
            json!([
                {"repo":"owner/matrix", "modelId":"matrix", "variant":"q4",
                 "resolvedFiles":["q4/model.safetensors"], "snapshotRevision": REV_A}
            ]),
        );
        let q4_after =
            selected_requirements_for_model(&model, std::env::consts::OS, Some("q4"), temp.path());
        assert_eq!(q4, q4_after.requirements);
        assert!(q4_after.receipt_backed);
    }

    #[test]
    fn selected_closure_uses_worker_platform_and_matching_tier_corequisites() {
        let model = json!({
            "id": "cross-platform",
            "downloads": [
                {"provider":"huggingface", "repo":"owner/mac", "variant":"q4", "default":true,
                 "files":["q4/*"], "platforms":["macos"]},
                {"provider":"huggingface", "repo":"owner/windows", "variant":"q4", "default":true,
                 "files":["q4/*"], "platforms":["windows", "linux"]},
                {"provider":"huggingface", "repo":"owner/candle-component", "variant":"q4",
                 "coRequisite":true, "files":["encoder.safetensors"], "platforms":["windows", "linux"]},
                {"provider":"huggingface", "repo":"owner/wrong-tier", "variant":"q8",
                 "coRequisite":true, "files":["encoder.safetensors"], "platforms":["windows", "linux"]}
            ]
        });
        let repositories = |entry: &Value| {
            entry["downloads"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|download| download["repo"].as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        };
        let mac = selected_model_artifact_closure(&model, "macos", Some("q4"));
        assert_eq!(repositories(&mac), vec!["owner/mac".to_owned()]);
        // The same manifest evaluated for a Windows worker must include the Candle-only
        // co-requisite the macOS host does not load — API-host platform filtering is the defect
        // this function exists to prevent.
        let windows = selected_model_artifact_closure(&model, "windows", Some("q4"));
        assert_eq!(
            repositories(&windows),
            vec![
                "owner/windows".to_owned(),
                "owner/candle-component".to_owned()
            ]
        );
    }

    #[test]
    fn declared_exact_requirements_skip_globs_soft_rows_and_mutable_revisions() {
        let model = json!({
            "id": "declared",
            "downloads": [
                {"provider":"huggingface", "repo":"owner/exact", "revision": REV_A,
                 "files":["model.safetensors"]},
                {"provider":"huggingface", "repo":"owner/glob", "revision": REV_A, "files":["q4/*"]},
                {"provider":"huggingface", "repo":"owner/mutable", "revision":"main",
                 "files":["model.safetensors"]},
                {"provider":"huggingface", "repo":"owner/soft", "revision": REV_A,
                 "coRequisite": true, "required":"soft", "files":["extra.safetensors"]}
            ]
        });
        let declared = declared_exact_requirements_for_model(&model);
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].repository, "owner/exact");
        assert_eq!(declared[0].revision.as_deref(), Some(REV_A));
    }

    #[test]
    fn closure_without_exactly_one_primary_is_empty_not_partial() {
        let temp = tempfile::tempdir().unwrap();
        // Only a co-requisite receipt exists: no primary identity means no closure at all —
        // a partial closure would let an unavailable primary read as ready.
        write_receipts(
            temp.path(),
            "owner/encoder",
            json!([
                {"repo":"owner/encoder", "resolvedFiles":["encoder.safetensors"],
                 "snapshotRevision": REV_A}
            ]),
        );
        let model = json!({
            "id": "primaryless",
            "downloads": [
                {"provider":"huggingface", "repo":"owner/primary", "files":["model.safetensors"]},
                {"provider":"huggingface", "repo":"owner/encoder", "coRequisite": true,
                 "files":["encoder.safetensors"]}
            ]
        });
        let selected =
            selected_requirements_for_model(&model, std::env::consts::OS, None, temp.path());
        assert!(selected.requirements.is_empty());
        assert!(!selected.receipt_backed);
    }

    #[test]
    fn requested_runtime_variant_reads_request_data_only() {
        let payload = |value: Value| value.as_object().unwrap().clone();
        assert_eq!(
            requested_runtime_variant(&payload(json!({"variant": "Q8"}))),
            Some("q8".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"advanced": {"quantTier": "bf16"}}))),
            Some("bf16".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"advanced": {"mlxQuantize": 4}}))),
            Some("q4".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"advanced": {"mlxQuantize": "8"}}))),
            Some("q8".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"advanced": {"mlxQuantize": 16}}))),
            Some("bf16".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"model": "x"}))),
            None
        );
    }

    /// The LIVE builtin YuE2 entry (sc-22998). Selection is judged on the real manifest row, not a
    /// copy of its shape.
    fn builtin_yue2() -> Value {
        let (_, contents) = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc is embedded");
        let manifest: Value =
            serde_json::from_str(&crate::jsonc::strip_jsonc_comments(contents)).expect("parses");
        manifest["models"]
            .as_array()
            .expect("models")
            .iter()
            .find(|model| model["id"] == "yue2")
            .expect("yue2 is in the builtin catalog")
            .clone()
    }

    fn repos(rows: &[Value]) -> Vec<&str> {
        rows.iter()
            .filter_map(|row| row.get("repo").and_then(Value::as_str))
            .collect()
    }

    #[test]
    fn yue2_install_queues_only_the_chosen_decoder() {
        // Base generation = YuE2-3B (+ its tokenizer, same repo) + ONE decoder. Mutation that reds
        // this: drop the `choice` filter from `model_co_requisite_downloads_for_variant` (both
        // decoders queue), or move `default: true` to the legacy row.
        let yue2 = builtin_yue2();
        for variant in ["bf16", "q8", "q4"] {
            let default_rows = model_co_requisite_downloads_for_variant(&yue2, Some(variant));
            assert_eq!(repos(&default_rows), ["m-a-p/YuE2-Vae"], "{variant}");
            assert_eq!(default_rows[0]["componentId"], "vae");

            let none = BTreeMap::new();
            let resolved = resolve_co_requisite_choices(&yue2, Some(variant), &none).unwrap();
            assert_eq!(
                resolved,
                BTreeMap::from([("decoder".to_owned(), "standard".to_owned())])
            );

            let legacy = BTreeMap::from([("decoder".to_owned(), "legacy".to_owned())]);
            let resolved = resolve_co_requisite_choices(&yue2, Some(variant), &legacy).unwrap();
            let rows = model_co_requisite_downloads_for_selection(&yue2, Some(variant), &resolved);
            assert_eq!(repos(&rows), ["m-a-p/YuE2-Vae-legacy"], "{variant}");
            assert_eq!(rows[0]["componentId"], "vae_legacy");

            // Install state reads every option: either decoder can satisfy the group.
            let all = model_co_requisite_downloads_for_variant_all_options(&yue2, Some(variant));
            assert_eq!(repos(&all), ["m-a-p/YuE2-Vae", "m-a-p/YuE2-Vae-legacy"]);
        }
    }

    #[test]
    fn either_installed_decoder_satisfies_the_install_and_a_missing_group_reports_its_default() {
        // Mutation that reds this: gate on every option (a legacy-only install reads incomplete) or
        // on none (a decoder-less install reads complete).
        let yue2 = builtin_yue2();
        let rows = || model_co_requisite_downloads_for_variant_all_options(&yue2, Some("q4"));
        let gating = |installed_repo: Option<&str>| {
            co_requisite_rows_gating_install(rows(), |row| {
                if installed_repo.is_some_and(|repo| row["repo"] == repo) {
                    CoRequisitePresence::Installed
                } else {
                    CoRequisitePresence::Absent
                }
            })
        };
        assert_eq!(repos(&gating(None)), ["m-a-p/YuE2-Vae"]);
        assert_eq!(
            repos(&gating(Some("m-a-p/YuE2-Vae-legacy"))),
            ["m-a-p/YuE2-Vae-legacy"]
        );
        assert_eq!(repos(&gating(Some("m-a-p/YuE2-Vae"))), ["m-a-p/YuE2-Vae"]);
    }

    #[test]
    fn a_derived_snapshot_is_verified_only_by_its_pinned_size_and_digest() {
        // Mutations that red this: drop the size check or the SHA-256 comparison in
        // `derived_snapshot_state`, or let `local_derivation_snapshot_dir` accept `..`.
        use sha2::{Digest, Sha256};
        let temp = tempfile::tempdir().unwrap();
        let weights = b"derived weights";
        let yue2 = builtin_yue2();
        let q4 = model_download_for_variant(&yue2, "q4").expect("q4 row");
        let mut derivation = local_derivation(&q4).expect("q4 declares a derivation");
        assert_eq!(derivation.from_variant, "bf16");
        assert_eq!(
            local_derivation(&model_download_for_variant(&yue2, "bf16").unwrap()),
            None
        );
        derivation.weights_bytes = weights.len() as u64;
        derivation.weights_sha256 = format!("{:x}", Sha256::digest(weights));

        let dir = local_derivation_snapshot_dir(temp.path(), "yue2", "q4", &derivation).unwrap();
        assert_eq!(
            dir,
            temp.path()
                .join("models/derived/yue2/q4")
                .join(&derivation.conversion)
        );
        assert_eq!(
            local_derivation_snapshot_dir(temp.path(), "..", "q4", &derivation),
            None
        );
        assert_eq!(
            derived_snapshot_state(&dir, &derivation),
            DerivedSnapshotState::Absent
        );

        std::fs::create_dir_all(&dir).unwrap();
        assert!(matches!(
            derived_snapshot_state(&dir, &derivation),
            DerivedSnapshotState::Invalid(_)
        ));
        std::fs::write(dir.join("model.safetensors"), b"short").unwrap();
        assert!(matches!(
            derived_snapshot_state(&dir, &derivation),
            DerivedSnapshotState::Invalid(why) if why.contains("bytes")
        ));
        std::fs::write(dir.join("model.safetensors"), vec![b'x'; weights.len()]).unwrap();
        assert!(matches!(
            derived_snapshot_state(&dir, &derivation),
            DerivedSnapshotState::Invalid(why) if why.contains("hashes to")
        ));
        std::fs::write(dir.join("model.safetensors"), weights).unwrap();
        assert_eq!(
            derived_snapshot_state(&dir, &derivation),
            DerivedSnapshotState::Verified
        );
    }

    #[test]
    fn a_partially_installed_legacy_decoder_is_what_the_repair_completes() {
        // Legacy half-downloaded, standard absent: the gating row (what the catalog reports missing)
        // and the repair's choice are LEGACY, not the default. Mutation that reds this: drop the
        // `started` fallback in `co_requisite_rows_gating_install` (standard comes back).
        let yue2 = builtin_yue2();
        let presence = |row: &Value| {
            if row["repo"] == "m-a-p/YuE2-Vae-legacy" {
                CoRequisitePresence::Incomplete
            } else {
                CoRequisitePresence::Absent
            }
        };
        let rows = model_co_requisite_downloads_for_variant_all_options(&yue2, Some("bf16"));
        assert_eq!(
            repos(&co_requisite_rows_gating_install(rows, presence)),
            ["m-a-p/YuE2-Vae-legacy"]
        );
        let choices =
            co_requisite_choices_for_repair(&yue2, Some("bf16"), &BTreeMap::new(), presence);
        assert_eq!(
            choices,
            BTreeMap::from([("decoder".to_owned(), "legacy".to_owned())])
        );
        // An explicit request still wins over what is on disk.
        let explicit = BTreeMap::from([("decoder".to_owned(), "standard".to_owned())]);
        assert_eq!(
            co_requisite_choices_for_repair(&yue2, Some("bf16"), &explicit, presence),
            explicit
        );
        // An INSTALLED standard beats a started legacy: the group is already satisfied.
        let both = |row: &Value| {
            if row["repo"] == "m-a-p/YuE2-Vae" {
                CoRequisitePresence::Installed
            } else {
                CoRequisitePresence::Incomplete
            }
        };
        let rows = model_co_requisite_downloads_for_variant_all_options(&yue2, Some("bf16"));
        assert_eq!(
            repos(&co_requisite_rows_gating_install(rows, both)),
            ["m-a-p/YuE2-Vae"]
        );
    }

    #[test]
    fn a_job_choosing_the_legacy_decoder_selects_exactly_that_decoder() {
        // The seam a YuE2 job's guard uses (sc-22999). Mutation that reds this: ignore `choices` in
        // `select_model_artifact_closure` (the default standard decoder comes back).
        let yue2 = builtin_yue2();
        let legacy = BTreeMap::from([("decoder".to_owned(), "legacy".to_owned())]);
        let closure = selected_model_artifact_closure_with_choices(&yue2, "macos", None, &legacy);
        assert_eq!(
            repos(closure["downloads"].as_array().unwrap()),
            ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae-legacy"]
        );
    }

    #[test]
    fn an_unknown_decoder_choice_is_refused_never_replaced() {
        let yue2 = builtin_yue2();
        let request = |group: &str, option: &str| {
            resolve_co_requisite_choices(
                &yue2,
                Some("bf16"),
                &BTreeMap::from([(group.to_owned(), option.to_owned())]),
            )
        };
        assert_eq!(
            request("decoder", "fp16"),
            Err(CoRequisiteChoiceError::UnknownOption {
                group: "decoder".to_owned(),
                option: "fp16".to_owned(),
                declared: vec!["standard".to_owned(), "legacy".to_owned()],
            })
        );
        assert!(matches!(
            request("vocoder", "standard"),
            Err(CoRequisiteChoiceError::UnknownGroup { .. })
        ));
        // A model with no choice groups refuses any requested choice rather than ignoring it.
        let plain = json!({"id": "plain", "downloads": [
            {"provider": "huggingface", "repo": "owner/plain"}
        ]});
        assert!(matches!(
            resolve_co_requisite_choices(
                &plain,
                None,
                &BTreeMap::from([("decoder".to_owned(), "legacy".to_owned())])
            ),
            Err(CoRequisiteChoiceError::UnknownGroup { .. })
        ));
    }

    #[test]
    fn a_malformed_choice_default_keeps_every_option_instead_of_dropping_one() {
        let mut yue2 = builtin_yue2();
        for row in yue2["downloads"].as_array_mut().unwrap() {
            if let Some(choice) = row.get_mut("choice") {
                choice.as_object_mut().unwrap().remove("default");
            }
        }
        let rows = model_co_requisite_downloads_for_variant(&yue2, Some("bf16"));
        assert_eq!(repos(&rows), ["m-a-p/YuE2-Vae", "m-a-p/YuE2-Vae-legacy"]);
        assert_eq!(
            resolve_co_requisite_choices(&yue2, Some("bf16"), &BTreeMap::new()),
            Err(CoRequisiteChoiceError::NoSingleDefault {
                group: "decoder".to_owned()
            })
        );
    }

    #[test]
    fn every_yue2_tier_selects_the_upstream_original_and_never_a_rehost_or_cover_dependency() {
        // The download-selection resolution point the worker guard and the API share. A q8 / q4
        // request must fetch the exact bf16 original (the tier is derived locally): same repo,
        // revision and files. Mutation that reds this: point a tier row at a `SceneWorks/…` re-host,
        // or add a SheetSage2 / MERT row to `downloads`.
        let yue2 = builtin_yue2();
        let bf16 = model_download_for_variant(&yue2, "bf16").expect("bf16 row");
        for platform in ["macos", "windows", "linux"] {
            for variant in ["bf16", "q8", "q4"] {
                let closure = selected_model_artifact_closure(&yue2, platform, Some(variant));
                let downloads = closure["downloads"].as_array().unwrap();
                let primary = &downloads[0];
                assert_eq!(primary["variant"], variant);
                for key in ["repo", "revision", "files"] {
                    assert_eq!(primary[key], bf16[key], "{platform}/{variant}/{key}");
                }
                assert_eq!(repos(downloads), ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]);
                for repo in repos(downloads) {
                    assert!(repo.starts_with("m-a-p/YuE2"), "{repo}");
                    assert!(
                        !repo.contains("SheetSage") && !repo.contains("MERT"),
                        "{repo}"
                    );
                }
            }
        }
    }
}
