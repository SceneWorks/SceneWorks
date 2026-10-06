//! Recording transcriptions (sc-23002, epic 22988 AT2): the SceneWorks record of a `transcribe`
//! job's review artifact, and the import of its scores as score versions.
//!
//! The worker transcribes with the native SheetSage2 + MERT-v2-FullSong port
//! (`candle_audio_sheetsage2`), persists the engine's review artifact into
//! `yue2/transcriptions/<runId>/` (the artifact is written file by file with `transcription.json`
//! last, so a manifest means a complete artifact), re-opens it with `ReviewArtifact::open` — which
//! re-derives every file and manifest field from the persisted tokens and model input — and unloads
//! the transcriber. [`Yue2ScoreStore::import_transcription`] then runs here, on the API side:
//!
//! 1. the manifest must hash to the SHA-256 the worker reported, name the engine's schema, and
//!    record the source recording's own SHA-256 (the recording the job was queued with);
//! 2. every transcribed score is read against its recorded digest; the modes the review marks
//!    cover-ready are imported as root score versions (origin `transcription`) linked to the
//!    transcription and the recording ([`TranscriptionLink`]) and carrying the job's usage policy.
//!    A mode the review refuses (an empty melody, an ABC that could not be built) is NOT imported
//!    and its reason is recorded; a score SceneWorks' native ABC dialect cannot hold is recorded as
//!    that mode's import error. Nothing is dropped silently.
//! 3. the record keeps what the reviewer needs — melody / chords / beat / key / structure summary,
//!    warnings, octave evidence, readiness per mode, the export files — plus the device the
//!    transcription ran on, the replay report and the transcriber's unload receipt.
//!
//! Re-running an import is idempotent (the terminal side effect may be re-run by the recovery
//! sweep): an existing record is returned, and a version already imported for a mode is reused.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::abc::sha256_hex;
use super::jobs::{is_transcription_dir, TRANSCRIPTION_ID_PREFIX};
use super::store::{
    CreateVersionInput, Provenance, ScoreVersionRecord, TranscriptionLink, VersionLineage,
    VersionOrigin, Yue2ScoreStore,
};
use super::{Cot, SongRequest, Yue2ScoreError, DEFAULT_SEED};
use crate::time::utc_now;

pub const TRANSCRIPTION_SCHEMA: &str = "sceneworks.yue2.transcription.v1";
/// The engine's review-artifact manifest (`candle_audio_sheetsage2::review::MANIFEST`).
pub const ARTIFACT_MANIFEST: &str = "transcription.json";
/// The engine's manifest schema (`candle_audio_sheetsage2::review::SCHEMA`).
pub const ARTIFACT_SCHEMA: &str = "sceneworks-sheetsage2-transcription-v1";
/// The full score, with chord symbols (upstream's `score.abc`).
pub const FULL_SCORE: &str = "score.abc";
/// The melody-only score (`candle_audio_sheetsage2::review::MELODY_SCORE`).
pub const MELODY_SCORE: &str = "score_melody.abc";
/// Engine identity on every transcription record.
pub const ENGINE: &str = "sheetsage2";

type Result<T> = std::result::Result<T, Yue2ScoreError>;

fn bad(message: impl Into<String>) -> Yue2ScoreError {
    Yue2ScoreError::BadRequest(message.into())
}

fn conflict(message: impl Into<String>) -> Yue2ScoreError {
    Yue2ScoreError::Conflict(message.into())
}

/// Whether one cover mode can use the transcription, as the review decided it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModeReadiness {
    pub ready: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Readiness {
    pub melody: ModeReadiness,
    pub full: ModeReadiness,
}

/// The score version imported for each mode (none for a refused mode).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedVersions {
    pub melody: Option<String>,
    pub full: Option<String>,
}

/// One file of the review artifact a reviewer can download (MIDI, LAB, ABC, JSON exports).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportFile {
    pub path: String,
    pub sha256: String,
    /// `midi`, `lab`, `abc`, `json`, `text` or `data`.
    pub kind: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionRecord {
    pub schema: String,
    pub id: String,
    pub project_id: String,
    pub engine: String,
    pub created_at: String,
    pub job_id: String,
    pub source_audio_asset_id: String,
    /// SHA-256 of the recording's media file the job transcribed.
    pub recording_sha256: String,
    /// Project-relative review-artifact directory.
    pub artifact_dir: String,
    /// SHA-256 of the artifact's `transcription.json`.
    pub manifest_sha256: String,
    /// The device the transcriber ran on (`cpu`, `metal`, `cuda`), as the artifact records it.
    pub device: String,
    /// The artifact's source identity (model input digest, duration, conversion, original digest).
    pub source: Value,
    pub settings: Value,
    /// The pinned closure identity (repos, revisions, weight and config digests, code revision).
    pub closure: Value,
    /// Melody voices, chords, keys, sections, bars, warnings, diagnostics, cover readiness.
    pub review: Value,
    /// Per-voice spectral octave evidence (`null` when there was no vocal note).
    pub octave_evidence: Value,
    /// Why a score could not be built, per mode (`{full, melody}`).
    pub abc_errors: Value,
    pub decode_warnings: Value,
    pub warnings: Vec<Value>,
    pub readiness: Readiness,
    pub versions: ImportedVersions,
    /// A cover-ready mode whose score SceneWorks could not import, with the reason.
    pub version_errors: BTreeMap<String, String>,
    pub exports: Vec<ExportFile>,
    /// The worker's `ReviewArtifact::open` replay report.
    pub replay: Value,
    /// The transcriber's unload receipt (`released`, `parameterBytes`, `liveModelsAfter`).
    pub unload: Value,
    pub usage_policy: Value,
    pub provenance: Provenance,
}

/// What the worker reported for a finished `transcribe` job.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptionImport {
    pub transcription_id: String,
    pub job_id: String,
    pub source_audio_asset_id: String,
    pub recording_sha256: String,
    pub artifact_dir: String,
    pub manifest_sha256: String,
    pub replay: Value,
    pub unload: Value,
    pub usage_policy: Value,
    pub provenance: Provenance,
}

/// A transcription and its two scores, read back against their recorded digests.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionDetail {
    pub transcription: TranscriptionRecord,
    /// `{full, melody}`: the transcribed ABC of each mode (`null` when the artifact has none).
    pub scores: BTreeMap<String, Option<String>>,
}

fn validate_transcription_id(id: &str) -> Result<()> {
    if !id.starts_with(TRANSCRIPTION_ID_PREFIX)
        || !crate::store_util::is_safe_id(id)
        || id.len() > 64
    {
        return Err(Yue2ScoreError::NotFound(
            "transcription not found".to_owned(),
        ));
    }
    Ok(())
}

/// An artifact-relative path is a plain relative path (no traversal, no absolute component).
fn safe_artifact_rel(rel: &str) -> bool {
    !rel.is_empty()
        && Path::new(rel)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn export_kind(path: &str) -> &'static str {
    match Path::new(path).extension().and_then(|ext| ext.to_str()) {
        Some("mid") => "midi",
        Some("lab") => "lab",
        Some("abc") => "abc",
        Some("json") => "json",
        Some("txt" | "tsv") => "text",
        _ => "data",
    }
}

fn mode_readiness(review: &Value, mode: &str) -> Result<ModeReadiness> {
    let cover = review
        .pointer(&format!("/cover/{mode}"))
        .ok_or_else(|| conflict(format!("the artifact records no {mode} cover readiness")))?;
    let ready = cover
        .get("ready")
        .and_then(Value::as_bool)
        .ok_or_else(|| conflict(format!("the artifact's {mode} readiness is malformed")))?;
    Ok(ModeReadiness {
        ready,
        reason: cover
            .get("reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

/// The review artifact on disk, opened against the manifest digest a record (or the worker)
/// recorded.
struct Artifact {
    dir: PathBuf,
    manifest: Value,
}

impl Artifact {
    fn open(project_path: &Path, artifact_dir: &str, manifest_sha256: &str) -> Result<Self> {
        if !is_transcription_dir(artifact_dir) {
            return Err(bad(format!(
                "{artifact_dir} is not a project transcription directory"
            )));
        }
        let dir = project_path.join(artifact_dir);
        let path = dir.join(ARTIFACT_MANIFEST);
        let bytes = fs::read(&path).map_err(|error| {
            conflict(format!(
                "the transcription artifact {artifact_dir} has no readable {ARTIFACT_MANIFEST} \
                 ({error})"
            ))
        })?;
        let digest = format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes));
        if digest != manifest_sha256 {
            return Err(conflict(format!(
                "{artifact_dir}/{ARTIFACT_MANIFEST} changed since the transcription recorded it \
                 (SHA-256 {digest}, recorded {manifest_sha256})"
            )));
        }
        let manifest: Value = serde_json::from_slice(&bytes)?;
        if manifest.get("schema").and_then(Value::as_str) != Some(ARTIFACT_SCHEMA) {
            return Err(conflict(format!(
                "{artifact_dir} is not a SheetSage2 review artifact (schema {})",
                manifest.get("schema").unwrap_or(&Value::Null)
            )));
        }
        Ok(Self { dir, manifest })
    }

    fn artifacts(&self) -> Result<Vec<(String, String)>> {
        let files = self
            .manifest
            .get("artifacts")
            .and_then(Value::as_object)
            .ok_or_else(|| conflict("the transcription manifest lists no artifacts"))?;
        files
            .iter()
            .map(|(rel, digest)| {
                let digest = digest
                    .as_str()
                    .ok_or_else(|| conflict(format!("artifact {rel} has no digest")))?;
                if !safe_artifact_rel(rel) {
                    return Err(conflict(format!(
                        "artifact path {rel} is not a plain relative path"
                    )));
                }
                Ok((rel.clone(), digest.to_owned()))
            })
            .collect()
    }

    /// Read `rel`, re-hashing it against its recorded digest (a file replaced after the
    /// transcription is refused, never served).
    fn read(&self, rel: &str) -> Result<Vec<u8>> {
        let digest = self
            .manifest
            .pointer("/artifacts")
            .and_then(|files| files.get(rel))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Yue2ScoreError::NotFound(format!("{rel} is not part of this transcription"))
            })?;
        if !safe_artifact_rel(rel) {
            return Err(Yue2ScoreError::NotFound(format!(
                "{rel} is not part of this transcription"
            )));
        }
        let path = rel
            .split('/')
            .fold(self.dir.clone(), |p, part| p.join(part));
        let bytes = fs::read(&path)?;
        let actual = format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes));
        if actual != digest {
            return Err(conflict(format!(
                "{rel} does not match its recorded SHA-256; the transcription artifact was altered"
            )));
        }
        Ok(bytes)
    }

    fn score(&self, file: &str) -> Result<Option<String>> {
        let listed = self
            .manifest
            .pointer("/artifacts")
            .and_then(|files| files.get(file))
            .is_some();
        if !listed {
            return Ok(None);
        }
        let bytes = self.read(file)?;
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| conflict(format!("{file} is not UTF-8")))
    }
}

impl Yue2ScoreStore {
    /// Record a finished transcription and import its cover-ready scores (see the
    /// [module docs](self)). Idempotent per transcription id.
    pub fn import_transcription(&self, input: TranscriptionImport) -> Result<TranscriptionRecord> {
        validate_transcription_id(&input.transcription_id)?;
        if let Ok(existing) = self.get_transcription(&input.transcription_id) {
            // A retry gets a new job id but retains the same run and recording. Never let an
            // existing record silently satisfy a different asset (even when its bytes match),
            // nor a different artifact under the same transcription id.
            if existing.source_audio_asset_id != input.source_audio_asset_id
                || existing.recording_sha256 != input.recording_sha256
                || existing.artifact_dir != input.artifact_dir
                || existing.manifest_sha256 != input.manifest_sha256
            {
                return Err(conflict(format!(
                    "transcription {} already belongs to another recording or artifact",
                    input.transcription_id
                )));
            }
            return Ok(existing);
        }
        let artifact = Artifact::open(
            self.project_path(),
            &input.artifact_dir,
            &input.manifest_sha256,
        )?;
        let manifest = &artifact.manifest;
        let original = manifest
            .pointer("/source/original_sha256")
            .and_then(Value::as_str);
        if original != Some(input.recording_sha256.as_str()) {
            return Err(conflict(format!(
                "the transcription records source recording {original:?}, but the job was queued \
                 with recording {}",
                input.recording_sha256
            )));
        }
        let review = manifest
            .get("review")
            .cloned()
            .ok_or_else(|| conflict("the transcription manifest has no review"))?;
        let readiness = Readiness {
            melody: mode_readiness(&review, "melody")?,
            full: mode_readiness(&review, "full")?,
        };
        let mut versions = ImportedVersions::default();
        let mut version_errors = BTreeMap::new();
        for (mode, cot, file) in [
            ("melody", Cot::Melody, MELODY_SCORE),
            ("full", Cot::Full, FULL_SCORE),
        ] {
            let ready = if mode == "melody" {
                readiness.melody.ready
            } else {
                readiness.full.ready
            };
            if !ready {
                continue;
            }
            match self.import_score(&input, &artifact, cot, file) {
                Ok(id) => {
                    if mode == "melody" {
                        versions.melody = Some(id);
                    } else {
                        versions.full = Some(id);
                    }
                }
                Err(error) => {
                    version_errors.insert(mode.to_owned(), error.to_string());
                }
            }
        }
        let exports = artifact
            .artifacts()?
            .into_iter()
            .map(|(path, sha256)| ExportFile {
                kind: export_kind(&path).to_owned(),
                path,
                sha256,
            })
            .collect();
        let device = manifest
            .pointer("/closure/device")
            .and_then(Value::as_str)
            .ok_or_else(|| conflict("the transcription manifest records no device"))?
            .to_owned();
        let field = |key: &str| manifest.get(key).cloned().unwrap_or(Value::Null);
        let record = TranscriptionRecord {
            schema: TRANSCRIPTION_SCHEMA.to_owned(),
            id: input.transcription_id.clone(),
            project_id: self.project_id().to_owned(),
            engine: ENGINE.to_owned(),
            created_at: utc_now(),
            job_id: input.job_id,
            source_audio_asset_id: input.source_audio_asset_id,
            recording_sha256: input.recording_sha256,
            artifact_dir: input.artifact_dir,
            manifest_sha256: input.manifest_sha256,
            device,
            source: field("source"),
            settings: field("settings"),
            closure: field("closure"),
            warnings: review
                .get("warnings")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            review,
            octave_evidence: field("octave_evidence"),
            abc_errors: field("abc_error"),
            decode_warnings: field("decode_warnings"),
            readiness,
            versions,
            version_errors,
            exports,
            replay: input.replay,
            unload: input.unload,
            usage_policy: input.usage_policy,
            provenance: input.provenance,
        };
        self.create_record("transcriptions", &record.id, &record)?;
        Ok(record)
    }

    fn import_score(
        &self,
        input: &TranscriptionImport,
        artifact: &Artifact,
        cot: Cot,
        file: &str,
    ) -> Result<String> {
        // Reuse a version this transcription already imported for the mode (a re-run import).
        if let Some(existing) = self.list_versions()?.items.into_iter().find(|version| {
            version.transcription.as_ref().is_some_and(|link| {
                link.transcription_id == input.transcription_id
                    && link.mode == cot
                    && version.parent_version_id.is_none()
            })
        }) {
            return Ok(existing.id);
        }
        let abc = artifact
            .score(file)?
            .ok_or_else(|| conflict(format!("the artifact has no {file}")))?;
        let link = TranscriptionLink {
            transcription_id: input.transcription_id.clone(),
            job_id: input.job_id.clone(),
            source_audio_asset_id: input.source_audio_asset_id.clone(),
            manifest_sha256: input.manifest_sha256.clone(),
            mode: cot,
            transcribed_score_sha256: sha256_hex(&abc),
        };
        let record: ScoreVersionRecord = self.create_version_with_lineage(
            CreateVersionInput {
                abc,
                // A transcription has a score but no words or style: the reviewer adds them (or
                // the cover job supplies them) before anything is generated.
                request: SongRequest {
                    style: String::new(),
                    lyrics: String::new(),
                    cot,
                    seed: DEFAULT_SEED,
                    cfg_scale: None,
                },
                origin: VersionOrigin::Transcription,
                provenance: input.provenance.clone(),
            },
            VersionLineage {
                transcription: Some(link),
                usage_policy: Some(input.usage_policy.clone()),
            },
        )?;
        Ok(record.id)
    }

    /// One transcription record.
    pub fn get_transcription(&self, id: &str) -> Result<TranscriptionRecord> {
        validate_transcription_id(id)?;
        let record: TranscriptionRecord = self.read_record("transcriptions", id)?;
        if record.id != id || record.project_id != self.project_id() {
            return Err(conflict(format!(
                "transcription file {id} does not describe {id} in this project"
            )));
        }
        Ok(record)
    }

    /// A transcription with both transcribed scores, each read against its recorded digest.
    pub fn get_transcription_detail(&self, id: &str) -> Result<TranscriptionDetail> {
        let record = self.get_transcription(id)?;
        let artifact = Artifact::open(
            self.project_path(),
            &record.artifact_dir,
            &record.manifest_sha256,
        )?;
        let mut scores = BTreeMap::new();
        scores.insert("full".to_owned(), artifact.score(FULL_SCORE)?);
        scores.insert("melody".to_owned(), artifact.score(MELODY_SCORE)?);
        Ok(TranscriptionDetail {
            transcription: record,
            scores,
        })
    }

    /// One file of a transcription's review artifact (an export), checked against its digest.
    pub fn transcription_file(&self, id: &str, rel: &str) -> Result<(ExportFile, Vec<u8>)> {
        let record = self.get_transcription(id)?;
        let export = record
            .exports
            .iter()
            .find(|export| export.path == rel)
            .cloned()
            .ok_or_else(|| {
                Yue2ScoreError::NotFound(format!("{rel} is not part of transcription {id}"))
            })?;
        let artifact = Artifact::open(
            self.project_path(),
            &record.artifact_dir,
            &record.manifest_sha256,
        )?;
        let bytes = artifact.read(rel)?;
        Ok((export, bytes))
    }

    /// Every transcription, oldest first; unreadable records are reported, never dropped.
    pub fn list_transcriptions(&self) -> Result<super::store::Listing<TranscriptionRecord>> {
        let mut items = Vec::new();
        let mut unreadable = Vec::new();
        for id in self.record_ids("transcriptions", TRANSCRIPTION_ID_PREFIX)? {
            match self.get_transcription(&id) {
                Ok(record) => items.push(record),
                Err(error) => unreadable.push(format!("transcriptions/{id}.json: {error}")),
            }
        }
        items.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        Ok(super::store::Listing {
            items,
            unreadable,
            render_notice: super::REGENERATION_NOTICE,
        })
    }
}

#[cfg(test)]
#[path = "transcriptions_tests.rs"]
mod tests;
