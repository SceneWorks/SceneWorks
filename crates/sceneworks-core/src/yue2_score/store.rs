//! Immutable per-project YuE2 score versions, renders and listening comparisons.
//!
//! Layout under the project directory (all JSON, one file per record, never rewritten):
//!
//! - `yue2/versions/<id>.json` — a score version: request, ABC score, edit (operation, brief,
//!   contract, invariant report) and provenance. A root version comes from an import, a plan or a
//!   transcription; every edit is a NEW version whose `parentVersionId` names its source.
//! - `yue2/renders/<id>.json` — one rendering of a version (written by the render job, sc-22999):
//!   the request and score hash it rendered, the version's edit brief, model/decoder identity,
//!   the audio asset and both truncation flags.
//! - `yue2/comparisons/<id>.json` — a persisted A/B listening comparison of two versions (and
//!   optionally one render of each) with the symbolic differences between them.
//!
//! Records are write-once: a create refuses an existing path, and reads verify the stored score
//! hash so a hand-edited or corrupted file is reported instead of being trusted.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::abc::{inspect, section_inspections, sha256_hex, Score};
use super::compare::{describe_differences, ChangeContract};
use super::ops::{apply_operation, ScoreEditOperation};
use super::{
    check_text, parse_bounded, validate_request, SongRequest, Yue2ScoreError, MAX_BRIEF_CHARS,
    REGENERATION_NOTICE,
};
use crate::project_store::ProjectStore;
use crate::store_util::{is_safe_id, lock_project_files, random_hex, write_json};
use crate::time::utc_now;

pub const VERSION_SCHEMA: &str = "sceneworks.yue2.scoreVersion.v1";
pub const RENDER_SCHEMA: &str = "sceneworks.yue2.scoreRender.v1";
pub const COMPARISON_SCHEMA: &str = "sceneworks.yue2.listeningComparison.v1";
/// Engine identity on every record, so a YuE2 record can never be mistaken for YuE1.
pub const ENGINE: &str = "yue2";

const VERSION_PREFIX: &str = "yue2v_";
const RENDER_PREFIX: &str = "yue2r_";
const COMPARISON_PREFIX: &str = "yue2c_";
const MAX_IDENTITY_CHARS: usize = 200;
const MAX_FAILURE_CHARS: usize = 2_000;
const MAX_EFFECTIVE_SETTINGS_BYTES: usize = 64 * 1024;

type Result<T> = std::result::Result<T, Yue2ScoreError>;

fn bad(message: impl Into<String>) -> Yue2ScoreError {
    Yue2ScoreError::BadRequest(message.into())
}

// ---------------------------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------------------------

/// Who made a record and through which surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Actor {
    User,
    Agent,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    #[default]
    Api,
    Mcp,
    Ui,
    Worker,
}

/// Where a root version's score came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    Plan,
    Transcription,
    Job,
    Asset,
    External,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceReference {
    pub kind: SourceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Provenance {
    pub actor: Actor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    #[serde(default)]
    pub channel: Channel,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceReference>,
}

impl Provenance {
    fn validate(&self) -> Result<()> {
        if let Some(name) = &self.agent_name {
            if name.trim().is_empty() {
                return Err(bad("provenance.agentName must not be blank"));
            }
            check_text("provenance.agentName", name, MAX_IDENTITY_CHARS)?;
        }
        if let Some(SourceReference { id: Some(id), .. }) = &self.source {
            if !is_safe_id(id) || id.len() > MAX_IDENTITY_CHARS {
                return Err(bad("provenance.source.id must be a plain identifier"));
            }
        }
        Ok(())
    }
}

/// How a version's score entered SceneWorks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VersionOrigin {
    Import,
    Plan,
    Transcription,
    Edit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScoreRecord {
    pub abc: String,
    pub sha256: String,
    /// Compact inspection summary at creation (bpm, measures, notes, chords, sections).
    pub summary: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditRecord {
    pub source_version_id: String,
    pub source_score_sha256: String,
    /// The operation as submitted (`replace_score` keeps its ABC hash, not a second copy).
    pub operation: Value,
    pub brief: String,
    pub contract: ChangeContract,
    /// The [`super::InvariantReport`] that admitted the edit.
    pub invariants: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScoreVersionRecord {
    pub schema: String,
    pub id: String,
    pub project_id: String,
    pub engine: String,
    pub created_at: String,
    pub parent_version_id: Option<String>,
    pub root_version_id: String,
    pub origin: VersionOrigin,
    pub request: SongRequest,
    /// [`super::request_sha256`] of `request`: the identity a render must report back.
    pub request_sha256: String,
    pub score: ScoreRecord,
    pub edit: Option<EditRecord>,
    pub provenance: Provenance,
    pub render_notice: String,
}

/// Both upstream truncation flags: the ABC plan and the semantic token stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Truncation {
    pub abc: bool,
    pub semantic: bool,
}

impl Truncation {
    pub fn any(self) -> bool {
        self.abc || self.semantic
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ComponentIdentity {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

impl ComponentIdentity {
    fn validate(&self, field: &str) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(bad(format!("{field}.id must not be blank")));
        }
        check_text(&format!("{field}.id"), &self.id, MAX_IDENTITY_CHARS)?;
        if let Some(revision) = &self.revision {
            check_text(&format!("{field}.revision"), revision, MAX_IDENTITY_CHARS)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderStatus {
    Completed,
    Failed,
}

/// What the render job reports when it finishes (sc-22999 fills this in).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RenderInput {
    pub status: RenderStatus,
    /// The score hash the job actually rendered; must equal the version's.
    pub score_sha256: String,
    /// [`super::request_sha256`] of the request the job actually rendered; must equal the
    /// version's `requestSha256`.
    pub request_sha256: String,
    /// Required — there is no default truncation state.
    pub truncated: Truncation,
    pub model: ComponentIdentity,
    #[serde(default)]
    pub decoder: Option<ComponentIdentity>,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub audio_asset_id: Option<String>,
    #[serde(default)]
    pub effective_settings: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderRecord {
    pub schema: String,
    pub id: String,
    pub project_id: String,
    pub engine: String,
    pub created_at: String,
    pub version_id: String,
    pub parent_version_id: Option<String>,
    pub root_version_id: String,
    pub status: RenderStatus,
    pub request: SongRequest,
    pub request_sha256: String,
    /// The exact ABC score rendered (a self-contained receipt, like upstream's `score.abc`).
    pub score_abc: String,
    pub score_sha256: String,
    pub edit_brief: Option<String>,
    pub truncated: Truncation,
    pub model: ComponentIdentity,
    pub decoder: Option<ComponentIdentity>,
    pub job_id: Option<String>,
    pub audio_asset_id: Option<String>,
    pub effective_settings: Option<Value>,
    pub error: Option<String>,
    pub provenance: Provenance,
    /// Always true: YuE2 renders are complete regenerations, never waveform edits.
    pub whole_recording_regenerated: bool,
    pub render_notice: String,
}

/// One side of a comparison.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComparisonSide {
    pub version_id: String,
    pub parent_version_id: Option<String>,
    pub origin: VersionOrigin,
    pub request: SongRequest,
    pub request_sha256: String,
    pub score_abc: String,
    pub score_sha256: String,
    pub edit_operation: Option<String>,
    pub edit_brief: Option<String>,
    pub render: Option<ListeningEntry>,
}

/// The render a listener hears for one side.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListeningEntry {
    pub render_id: String,
    pub audio_asset_id: String,
    pub truncated: Truncation,
    pub model: ComponentIdentity,
    pub decoder: Option<ComponentIdentity>,
    pub job_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lineage {
    Same,
    BDerivesFromA,
    ADerivesFromB,
    SharedRoot,
    Unrelated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComparisonRecord {
    pub schema: String,
    pub id: String,
    pub project_id: String,
    pub engine: String,
    pub created_at: String,
    pub a: ComparisonSide,
    pub b: ComparisonSide,
    pub lineage: Lineage,
    /// Whether the two versions carry identical musical content and request fields.
    pub identical_music_and_request: bool,
    /// Every symbolic/request difference, A → B (parsed events, not text).
    pub symbolic_differences: Vec<String>,
    /// Chord changes A → B.
    pub harmony_changes: Value,
    /// When B is A's direct edit: the invariant report that admitted it.
    pub edit_invariants: Option<Value>,
    pub warnings: Vec<String>,
    pub notes: Option<String>,
    pub provenance: Provenance,
    pub render_notice: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionSummary {
    pub id: String,
    pub created_at: String,
    pub parent_version_id: Option<String>,
    pub root_version_id: String,
    pub origin: VersionOrigin,
    pub edit_operation: Option<String>,
    pub edit_brief: Option<String>,
    pub score_sha256: String,
    pub cot: super::Cot,
    pub summary: Value,
    pub render_count: usize,
    pub provenance: Provenance,
}

/// A listing plus any record files that could not be read (reported, never silently dropped).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Listing<T> {
    pub items: Vec<T>,
    pub unreadable: Vec<String>,
    pub render_notice: &'static str,
}

/// A version with its renders.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionDetail {
    pub version: ScoreVersionRecord,
    pub renders: Vec<RenderRecord>,
}

/// Input for a root version.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CreateVersionInput {
    pub abc: String,
    pub request: SongRequest,
    pub origin: VersionOrigin,
    pub provenance: Provenance,
}

/// Input for an edit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EditInput {
    pub operation: ScoreEditOperation,
    /// The bounded, human-readable change brief (required: every edit says what and why).
    pub brief: String,
    pub provenance: Provenance,
}

/// Input for a comparison.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ComparisonInput {
    pub version_a: String,
    pub version_b: String,
    #[serde(default)]
    pub render_a: Option<String>,
    #[serde(default)]
    pub render_b: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    pub provenance: Provenance,
}

// ---------------------------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------------------------

/// Per-project record access. Obtain one through [`ProjectStore::yue2_score_store`].
pub struct Yue2ScoreStore {
    project_path: PathBuf,
    project_id: String,
}

fn summarize(score: &Score) -> Value {
    let inspection = inspect(score);
    serde_json::json!({
        "bpm": inspection.bpm,
        "unitLength": inspection.unit_length,
        "measures": score.bar_count(),
        "durationQuarters": inspection.duration_quarters,
        "nominalDurationSeconds": inspection.nominal_duration_seconds,
        "soundingNotes": {
            "Vocal": inspection.voices.vocal.sounding_notes,
            "Ins": inspection.voices.ins.sounding_notes,
        },
        "chordCount": inspection.voices.vocal.chords.len(),
        "sections": section_inspections(score),
    })
}

/// A stored score/request snapshot must still hash to its recorded identities.
fn verify_snapshot(
    what: &str,
    abc: &str,
    score_sha256: &str,
    request: &SongRequest,
    request_sha256: &str,
) -> Result<()> {
    if sha256_hex(abc) != score_sha256 {
        return Err(Yue2ScoreError::Conflict(format!(
            "{what} is corrupt: its score ABC no longer matches the recorded sha256"
        )));
    }
    if super::request_sha256(request) != request_sha256 {
        return Err(Yue2ScoreError::Conflict(format!(
            "{what} is corrupt: its request no longer matches the recorded sha256"
        )));
    }
    Ok(())
}

fn validate_record_id(id: &str, prefix: &str, what: &str) -> Result<()> {
    if !id.starts_with(prefix) || !is_safe_id(id) || id.len() > 64 {
        return Err(Yue2ScoreError::NotFound(format!("{what} not found")));
    }
    Ok(())
}

impl Yue2ScoreStore {
    pub fn new(project_path: impl Into<PathBuf>, project_id: impl Into<String>) -> Self {
        Self {
            project_path: project_path.into(),
            project_id: project_id.into(),
        }
    }

    fn dir(&self, kind: &str) -> PathBuf {
        self.project_path.join("yue2").join(kind)
    }

    fn create_record<T: Serialize>(&self, kind: &str, id: &str, record: &T) -> Result<()> {
        let _guard = lock_project_files(&self.project_path);
        let path = self.dir(kind).join(format!("{id}.json"));
        if path.exists() {
            return Err(Yue2ScoreError::Conflict(format!(
                "{id} already exists; records are never overwritten"
            )));
        }
        write_json(&path, record)?;
        Ok(())
    }

    fn read_record<T: for<'de> Deserialize<'de>>(&self, kind: &str, id: &str) -> Result<T> {
        let path = self.dir(kind).join(format!("{id}.json"));
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Yue2ScoreError::NotFound(format!("{id} not found")));
            }
            Err(error) => return Err(error.into()),
        };
        Ok(serde_json::from_str(&text)?)
    }

    fn record_ids(&self, kind: &str, prefix: &str) -> Result<Vec<String>> {
        let dir = self.dir(kind);
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
                if stem.starts_with(prefix) && is_safe_id(stem) {
                    ids.push(stem.to_owned());
                }
            }
        }
        ids.sort();
        Ok(ids)
    }

    fn checked_version(&self, record: ScoreVersionRecord, id: &str) -> Result<ScoreVersionRecord> {
        if record.id != id || record.project_id != self.project_id || record.engine != ENGINE {
            return Err(Yue2ScoreError::Conflict(format!(
                "score version file {id} does not describe {id} in this project"
            )));
        }
        if sha256_hex(&record.score.abc) != record.score.sha256 {
            return Err(Yue2ScoreError::Conflict(format!(
                "score version {id} is corrupt: its ABC no longer matches the recorded sha256"
            )));
        }
        if super::request_sha256(&record.request) != record.request_sha256 {
            return Err(Yue2ScoreError::Conflict(format!(
                "score version {id} is corrupt: its request no longer matches the recorded sha256"
            )));
        }
        Ok(record)
    }

    /// Read one version (integrity-checked).
    pub fn get_version_record(&self, id: &str) -> Result<ScoreVersionRecord> {
        validate_record_id(id, VERSION_PREFIX, "score version")?;
        let record: ScoreVersionRecord = self.read_record("versions", id)?;
        self.checked_version(record, id)
    }

    /// A version plus its renders.
    pub fn get_version(&self, id: &str) -> Result<VersionDetail> {
        let version = self.get_version_record(id)?;
        let renders = self
            .list_renders()?
            .items
            .into_iter()
            .filter(|render| render.version_id == id)
            .collect();
        Ok(VersionDetail { version, renders })
    }

    /// Parsed score of a stored version.
    pub fn version_score(&self, id: &str) -> Result<(ScoreVersionRecord, Score)> {
        let record = self.get_version_record(id)?;
        let score = parse_bounded(&record.score.abc)?;
        Ok((record, score))
    }

    fn new_version(
        &self,
        score: &Score,
        request: SongRequest,
        origin: VersionOrigin,
        parent: Option<&ScoreVersionRecord>,
        edit: Option<EditRecord>,
        provenance: Provenance,
    ) -> Result<ScoreVersionRecord> {
        let id = format!("{VERSION_PREFIX}{}", random_hex(12)?);
        Ok(ScoreVersionRecord {
            schema: VERSION_SCHEMA.to_owned(),
            root_version_id: parent
                .map(|parent| parent.root_version_id.clone())
                .unwrap_or_else(|| id.clone()),
            id,
            project_id: self.project_id.clone(),
            engine: ENGINE.to_owned(),
            created_at: utc_now(),
            parent_version_id: parent.map(|parent| parent.id.clone()),
            origin,
            request_sha256: super::request_sha256(&request),
            request,
            score: ScoreRecord {
                abc: score.text.clone(),
                sha256: score.sha256(),
                summary: summarize(score),
            },
            edit,
            provenance,
            render_notice: REGENERATION_NOTICE.to_owned(),
        })
    }

    /// Create a root version from an imported, planned or transcribed score.
    pub fn create_version(&self, input: CreateVersionInput) -> Result<ScoreVersionRecord> {
        if input.origin == VersionOrigin::Edit {
            return Err(bad(
                "origin \"edit\" is reserved for versions made through an edit operation",
            ));
        }
        input.provenance.validate()?;
        let score = parse_bounded(&input.abc)?;
        validate_request(&input.request, &score)?;
        let record = self.new_version(
            &score,
            input.request,
            input.origin,
            None,
            None,
            input.provenance,
        )?;
        self.create_record("versions", &record.id, &record)?;
        Ok(record)
    }

    /// Check an edit against `source_id` without storing it (`dryRun`), or store it as a new
    /// version linked to its source.
    pub fn apply_edit(
        &self,
        source_id: &str,
        input: EditInput,
        dry_run: bool,
    ) -> Result<ScoreVersionRecord> {
        if input.brief.trim().is_empty() {
            return Err(bad(
                "brief must describe the requested change and what it must preserve",
            ));
        }
        check_text("brief", &input.brief, MAX_BRIEF_CHARS)?;
        input.provenance.validate()?;
        let (source, source_score) = self.version_score(source_id)?;
        let outcome = apply_operation(&source_score, &source.request, &input.operation)?;
        let edit = EditRecord {
            source_version_id: source.id.clone(),
            source_score_sha256: source.score.sha256.clone(),
            operation: input.operation.record(),
            brief: input.brief,
            contract: outcome.contract,
            invariants: serde_json::to_value(&outcome.report)?,
        };
        let record = self.new_version(
            &outcome.score,
            outcome.request,
            VersionOrigin::Edit,
            Some(&source),
            Some(edit),
            input.provenance,
        )?;
        if !dry_run {
            self.create_record("versions", &record.id, &record)?;
        }
        Ok(record)
    }

    /// All versions, oldest first.
    pub fn list_versions(&self) -> Result<Listing<VersionSummary>> {
        let renders = self.list_renders()?;
        let mut items = Vec::new();
        let mut unreadable = renders.unreadable.clone();
        for id in self.record_ids("versions", VERSION_PREFIX)? {
            match self.get_version_record(&id) {
                Ok(version) => items.push(VersionSummary {
                    render_count: renders
                        .items
                        .iter()
                        .filter(|render| render.version_id == version.id)
                        .count(),
                    id: version.id,
                    created_at: version.created_at,
                    parent_version_id: version.parent_version_id,
                    root_version_id: version.root_version_id,
                    origin: version.origin,
                    edit_operation: version.edit.as_ref().and_then(|edit| {
                        edit.operation
                            .get("op")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    }),
                    edit_brief: version.edit.map(|edit| edit.brief),
                    score_sha256: version.score.sha256,
                    cot: version.request.cot,
                    summary: version.score.summary,
                    provenance: version.provenance,
                }),
                Err(error) => unreadable.push(format!("versions/{id}.json: {error}")),
            }
        }
        // Timestamps have one-second resolution, so a source and its edit can share one; lineage
        // depth breaks the tie so a parent always lists before its children.
        let parents: std::collections::HashMap<String, Option<String>> = items
            .iter()
            .map(|item| (item.id.clone(), item.parent_version_id.clone()))
            .collect();
        let depth = |id: &str| {
            let mut depth = 0usize;
            let mut current = parents.get(id).cloned().flatten();
            while let Some(parent) = current {
                depth += 1;
                if depth > parents.len() {
                    break;
                }
                current = parents.get(&parent).cloned().flatten();
            }
            depth
        };
        items
            .sort_by_cached_key(|item| (item.created_at.clone(), depth(&item.id), item.id.clone()));
        Ok(Listing {
            items,
            unreadable,
            render_notice: REGENERATION_NOTICE,
        })
    }

    /// Record one rendering of a version. Crate-private: the public entry point is
    /// [`ProjectStore::record_yue2_render`], which also verifies the audio asset.
    pub(crate) fn record_render(
        &self,
        version_id: &str,
        input: RenderInput,
    ) -> Result<RenderRecord> {
        let version = self.get_version_record(version_id)?;
        input.provenance.validate()?;
        input.model.validate("model")?;
        if let Some(decoder) = &input.decoder {
            decoder.validate("decoder")?;
        }
        if input.score_sha256 != version.score.sha256 {
            return Err(Yue2ScoreError::Conflict(format!(
                "the render reports score {} but version {version_id} has score {}; a render \
                 must come from this version's exact score",
                input.score_sha256, version.score.sha256
            )));
        }
        if input.request_sha256 != version.request_sha256 {
            return Err(Yue2ScoreError::Conflict(format!(
                "the render reports request {} but version {version_id} has request {}; a render \
                 must use this version's exact style, lyrics, cot, seed and cfgScale",
                input.request_sha256, version.request_sha256
            )));
        }
        for (field, value) in [
            ("jobId", &input.job_id),
            ("audioAssetId", &input.audio_asset_id),
        ] {
            if let Some(value) = value {
                if !is_safe_id(value) || value.len() > MAX_IDENTITY_CHARS {
                    return Err(bad(format!("{field} must be a plain identifier")));
                }
            }
        }
        match input.status {
            RenderStatus::Completed => {
                if input.audio_asset_id.is_none() {
                    return Err(bad("a completed render must name its audio asset"));
                }
                if input.decoder.is_none() {
                    return Err(bad("a completed render must name its decoder"));
                }
                if input.error.is_some() {
                    return Err(bad("a completed render cannot carry an error"));
                }
            }
            RenderStatus::Failed => {
                if input.audio_asset_id.is_some() {
                    return Err(bad(
                        "a failed render has no recording; it cannot name an audio asset",
                    ));
                }
                match &input.error {
                    Some(error) if !error.trim().is_empty() => {
                        check_text("error", error, MAX_FAILURE_CHARS)?;
                    }
                    _ => return Err(bad("a failed render must say why it failed")),
                }
            }
        }
        if let Some(settings) = &input.effective_settings {
            if !settings.is_object() {
                return Err(bad("effectiveSettings must be a JSON object"));
            }
            if serde_json::to_vec(settings)?.len() > MAX_EFFECTIVE_SETTINGS_BYTES {
                return Err(bad(format!(
                    "effectiveSettings exceeds {MAX_EFFECTIVE_SETTINGS_BYTES} bytes"
                )));
            }
        }
        let record = RenderRecord {
            schema: RENDER_SCHEMA.to_owned(),
            id: format!("{RENDER_PREFIX}{}", random_hex(12)?),
            project_id: self.project_id.clone(),
            engine: ENGINE.to_owned(),
            created_at: utc_now(),
            version_id: version.id.clone(),
            parent_version_id: version.parent_version_id.clone(),
            root_version_id: version.root_version_id.clone(),
            status: input.status,
            request: version.request.clone(),
            request_sha256: version.request_sha256.clone(),
            score_abc: version.score.abc.clone(),
            score_sha256: version.score.sha256.clone(),
            edit_brief: version.edit.as_ref().map(|edit| edit.brief.clone()),
            truncated: input.truncated,
            model: input.model,
            decoder: input.decoder,
            job_id: input.job_id,
            audio_asset_id: input.audio_asset_id,
            effective_settings: input.effective_settings,
            error: input.error,
            provenance: input.provenance,
            whole_recording_regenerated: true,
            render_notice: REGENERATION_NOTICE.to_owned(),
        };
        self.create_record("renders", &record.id, &record)?;
        Ok(record)
    }

    pub fn get_render(&self, id: &str) -> Result<RenderRecord> {
        validate_record_id(id, RENDER_PREFIX, "render")?;
        let record: RenderRecord = self.read_record("renders", id)?;
        if record.id != id || record.project_id != self.project_id {
            return Err(Yue2ScoreError::Conflict(format!(
                "render file {id} does not describe {id} in this project"
            )));
        }
        verify_snapshot(
            &format!("render {id}"),
            &record.score_abc,
            &record.score_sha256,
            &record.request,
            &record.request_sha256,
        )?;
        Ok(record)
    }

    pub fn list_renders(&self) -> Result<Listing<RenderRecord>> {
        let mut items = Vec::new();
        let mut unreadable = Vec::new();
        for id in self.record_ids("renders", RENDER_PREFIX)? {
            match self.get_render(&id) {
                Ok(render) => items.push(render),
                Err(error) => unreadable.push(format!("renders/{id}.json: {error}")),
            }
        }
        items.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        Ok(Listing {
            items,
            unreadable,
            render_notice: REGENERATION_NOTICE,
        })
    }

    fn ancestors(&self, version: &ScoreVersionRecord) -> Result<Vec<String>> {
        let mut chain = Vec::new();
        let mut current = version.parent_version_id.clone();
        while let Some(id) = current {
            if chain.contains(&id) {
                return Err(Yue2ScoreError::Conflict(format!(
                    "version lineage loops at {id}"
                )));
            }
            let parent = self.get_version_record(&id)?;
            chain.push(id);
            current = parent.parent_version_id;
        }
        Ok(chain)
    }

    fn listening_entry(
        &self,
        version: &ScoreVersionRecord,
        render_id: Option<&str>,
        side: &str,
        warnings: &mut Vec<String>,
    ) -> Result<Option<ListeningEntry>> {
        let Some(render_id) = render_id else {
            warnings.push(format!(
                "side {side} has no render: this comparison is symbolic only for {side}; render \
                 the version to listen"
            ));
            return Ok(None);
        };
        let render = self.get_render(render_id)?;
        if render.version_id != version.id {
            return Err(bad(format!(
                "render {render_id} belongs to version {}, not {}",
                render.version_id, version.id
            )));
        }
        let (RenderStatus::Completed, Some(audio_asset_id)) =
            (render.status, render.audio_asset_id)
        else {
            return Err(bad(format!(
                "render {render_id} did not complete; it has no recording to compare"
            )));
        };
        if render.truncated.any() {
            warnings.push(format!(
                "side {side} render {render_id} is TRUNCATED (abc: {}, semantic: {}); compare \
                 complete recordings before judging the edit",
                render.truncated.abc, render.truncated.semantic
            ));
        }
        Ok(Some(ListeningEntry {
            render_id: render.id,
            audio_asset_id,
            truncated: render.truncated,
            model: render.model,
            decoder: render.decoder,
            job_id: render.job_id,
        }))
    }

    /// Persist an A/B listening comparison between two versions (and optionally one render each).
    pub fn create_comparison(&self, input: ComparisonInput) -> Result<ComparisonRecord> {
        input.provenance.validate()?;
        if let Some(notes) = &input.notes {
            check_text("notes", notes, MAX_BRIEF_CHARS)?;
        }
        let (a, a_score) = self.version_score(&input.version_a)?;
        let (b, b_score) = self.version_score(&input.version_b)?;
        let mut warnings = Vec::new();
        let render_a = self.listening_entry(&a, input.render_a.as_deref(), "A", &mut warnings)?;
        let render_b = self.listening_entry(&b, input.render_b.as_deref(), "B", &mut warnings)?;
        let (a_ancestors, b_ancestors) = (self.ancestors(&a)?, self.ancestors(&b)?);
        let lineage = if a.id == b.id {
            Lineage::Same
        } else if b_ancestors.contains(&a.id) {
            Lineage::BDerivesFromA
        } else if a_ancestors.contains(&b.id) {
            Lineage::ADerivesFromB
        } else if a.root_version_id == b.root_version_id {
            Lineage::SharedRoot
        } else {
            Lineage::Unrelated
        };
        let differences = describe_differences(&a_score, &a.request, &b_score, &b.request);
        let edit_invariants = b
            .edit
            .as_ref()
            .filter(|edit| edit.source_version_id == a.id)
            .map(|edit| edit.invariants.clone());
        let side = |version: &ScoreVersionRecord, render: Option<ListeningEntry>| ComparisonSide {
            version_id: version.id.clone(),
            parent_version_id: version.parent_version_id.clone(),
            origin: version.origin,
            request: version.request.clone(),
            request_sha256: version.request_sha256.clone(),
            score_abc: version.score.abc.clone(),
            score_sha256: version.score.sha256.clone(),
            edit_operation: version.edit.as_ref().and_then(|edit| {
                edit.operation
                    .get("op")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
            edit_brief: version.edit.as_ref().map(|edit| edit.brief.clone()),
            render,
        };
        let record = ComparisonRecord {
            schema: COMPARISON_SCHEMA.to_owned(),
            id: format!("{COMPARISON_PREFIX}{}", random_hex(12)?),
            project_id: self.project_id.clone(),
            engine: ENGINE.to_owned(),
            created_at: utc_now(),
            a: side(&a, render_a),
            b: side(&b, render_b),
            lineage,
            identical_music_and_request: differences.matches,
            symbolic_differences: differences.violations,
            harmony_changes: serde_json::to_value(&differences.harmony_changes)?,
            edit_invariants,
            warnings,
            notes: input.notes,
            provenance: input.provenance,
            render_notice: REGENERATION_NOTICE.to_owned(),
        };
        self.create_record("comparisons", &record.id, &record)?;
        Ok(record)
    }

    pub fn get_comparison(&self, id: &str) -> Result<ComparisonRecord> {
        validate_record_id(id, COMPARISON_PREFIX, "comparison")?;
        let record: ComparisonRecord = self.read_record("comparisons", id)?;
        if record.id != id || record.project_id != self.project_id {
            return Err(Yue2ScoreError::Conflict(format!(
                "comparison file {id} does not describe {id} in this project"
            )));
        }
        for (label, side) in [("A", &record.a), ("B", &record.b)] {
            verify_snapshot(
                &format!("comparison {id} side {label}"),
                &side.score_abc,
                &side.score_sha256,
                &side.request,
                &side.request_sha256,
            )?;
        }
        Ok(record)
    }

    pub fn list_comparisons(&self) -> Result<Listing<ComparisonRecord>> {
        let mut items = Vec::new();
        let mut unreadable = Vec::new();
        for id in self.record_ids("comparisons", COMPARISON_PREFIX)? {
            match self.get_comparison(&id) {
                Ok(record) => items.push(record),
                Err(error) => unreadable.push(format!("comparisons/{id}.json: {error}")),
            }
        }
        items.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        Ok(Listing {
            items,
            unreadable,
            render_notice: REGENERATION_NOTICE,
        })
    }

    pub fn project_path(&self) -> &Path {
        &self.project_path
    }
}

impl ProjectStore {
    /// Record a render of a YuE2 score version after verifying that a completed render's audio
    /// asset exists in the project and is audio. The entry point for the render job (sc-22999).
    pub fn record_yue2_render(
        &self,
        project_id: &str,
        version_id: &str,
        input: RenderInput,
    ) -> Result<RenderRecord> {
        let store = self.yue2_score_store(project_id)?;
        if let Some(asset_id) = &input.audio_asset_id {
            let asset = self
                .get_asset(project_id, asset_id)
                .map_err(|error| match error {
                    crate::project_store::ProjectStoreError::NotFound(_) => bad(format!(
                        "audio asset {asset_id} does not exist in this project"
                    )),
                    other => Yue2ScoreError::Store(other),
                })?;
            if asset.get("type").and_then(Value::as_str) != Some("audio") {
                return Err(bad(format!("asset {asset_id} is not an audio asset")));
            }
        }
        store.record_render(version_id, input)
    }
}
