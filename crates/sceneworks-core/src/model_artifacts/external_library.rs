//! Durable identity and live availability for an operator-owned model source library.
//!
//! Download receipts describe immutable model installs. This module deliberately keeps the
//! physical library binding in a separate app-owned ledger so a disconnected scan cannot erase or
//! rewrite installation provenance. Runtime admission carries a serialized [`ModelResolution`],
//! but workers always re-probe its exact path and physical identity before constructing a loader.

use super::{ArtifactLocation, ResolvedModelArtifact};
use crate::hf_home::safe_repo_dir_name;
use crate::store_util::{atomic_write, random_hex};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const EXTERNAL_LIBRARY_CONTRACT_VERSION: u32 = 1;
pub const EXTERNAL_LIBRARY_UNAVAILABLE_CODE: &str = "external_model_library_unavailable";
const LEDGER_FILE: &str = ".sceneworks-external-library-state.json";
const LEDGER_LOCK: &str = ".sceneworks-external-library-state.lock";
const VALIDATED_CLOSURES_FILE: &str = ".sceneworks-external-library-closures.json";
const SESSION_DIR: &str = ".sceneworks-external-source-sessions";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalLibraryError(pub String);

impl std::fmt::Display for ExternalLibraryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ExternalLibraryError {}

impl From<std::io::Error> for ExternalLibraryError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAvailability {
    LocalReady,
    ExternalReady,
    InstalledExternalUnavailable,
    Incomplete,
    Missing,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalLibraryProbeStatus {
    Available,
    Unavailable,
    IdentityMismatch,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExternalLibraryPhysicalIdentity {
    /// Stable native volume identity. On macOS this is the full volume UUID rather than a
    /// mount-session device number, so unplugging and reconnecting the expected disk preserves the
    /// binding while mounting a different disk at the same path fails closed.
    pub volume_id: String,
    pub directory_id: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExternalLibraryBinding {
    pub schema_version: u32,
    pub configured_path: PathBuf,
    pub canonical_path: PathBuf,
    pub physical_identity: ExternalLibraryPhysicalIdentity,
    pub bound_at: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExternalArtifactRequirement {
    pub repository: String,
    /// Immutable snapshot revision when the receipt carries one. Legacy receipts may leave this
    /// unset; adoption then requires exactly one snapshot to satisfy the full recorded file set.
    pub revision: Option<String>,
    pub variant: String,
    pub files: Vec<PathBuf>,
    /// Exactly one requirement in a runtime closure identifies the selected primary artifact.
    /// Co-requisites may share repository/variant/file shapes, so callers must not infer this role.
    pub is_primary: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ValidatedExternalClosures {
    schema_version: u32,
    closures: Vec<Vec<ExternalArtifactRequirement>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExternalLibraryProbe {
    pub status: ExternalLibraryProbeStatus,
    pub observed_path: Option<PathBuf>,
    pub observed_identity: Option<ExternalLibraryPhysicalIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelResolution {
    pub schema_version: u32,
    pub availability: ModelAvailability,
    pub configured_library_path: PathBuf,
    pub expected_library: Option<ExternalLibraryBinding>,
    pub requirements: Vec<ExternalArtifactRequirement>,
    pub local_artifact: Option<ResolvedModelArtifact>,
}

impl ModelResolution {
    pub fn external_ready(
        configured_library_path: PathBuf,
        expected_library: ExternalLibraryBinding,
        requirements: Vec<ExternalArtifactRequirement>,
    ) -> Result<Self, ExternalLibraryError> {
        validate_requirements_shape(&requirements)?;
        Ok(Self {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            availability: ModelAvailability::ExternalReady,
            configured_library_path,
            expected_library: Some(expected_library),
            requirements,
            local_artifact: None,
        })
    }

    pub fn unavailable(
        configured_library_path: PathBuf,
        expected_library: Option<ExternalLibraryBinding>,
        requirements: Vec<ExternalArtifactRequirement>,
    ) -> Self {
        Self {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            availability: ModelAvailability::InstalledExternalUnavailable,
            configured_library_path,
            expected_library,
            requirements,
            local_artifact: None,
        }
    }

    pub fn not_ready(
        availability: ModelAvailability,
        configured_library_path: PathBuf,
        requirements: Vec<ExternalArtifactRequirement>,
    ) -> Result<Self, ExternalLibraryError> {
        if !matches!(
            availability,
            ModelAvailability::Incomplete | ModelAvailability::Missing
        ) {
            return Err(ExternalLibraryError(
                "not-ready resolution must be incomplete or missing".to_owned(),
            ));
        }
        Ok(Self {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            availability,
            configured_library_path,
            expected_library: None,
            requirements,
            local_artifact: None,
        })
    }

    pub fn local_ready(artifact: ResolvedModelArtifact) -> Result<Self, ExternalLibraryError> {
        artifact
            .validate()
            .map_err(|error| ExternalLibraryError(error.to_string()))?;
        if !matches!(&artifact.location, ArtifactLocation::ResolvedLocal { .. }) {
            return Err(ExternalLibraryError(
                "local-ready resolution must use the app-owned resolved tier".to_owned(),
            ));
        }
        Ok(Self {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            availability: ModelAvailability::LocalReady,
            configured_library_path: PathBuf::new(),
            expected_library: None,
            requirements: Vec::new(),
            local_artifact: Some(artifact),
        })
    }

    pub fn validate(&self) -> Result<(), ExternalLibraryError> {
        if self.schema_version != EXTERNAL_LIBRARY_CONTRACT_VERSION {
            return Err(ExternalLibraryError(
                "unsupported external-library contract version".to_owned(),
            ));
        }
        match self.availability {
            ModelAvailability::LocalReady => {
                if !self.configured_library_path.as_os_str().is_empty()
                    || self.expected_library.is_some()
                    || !self.requirements.is_empty()
                {
                    return Err(ExternalLibraryError(
                        "local-ready resolution contains external source state".to_owned(),
                    ));
                }
                let artifact = self.local_artifact.as_ref().ok_or_else(|| {
                    ExternalLibraryError("local-ready resolution has no artifact".into())
                })?;
                if !matches!(&artifact.location, ArtifactLocation::ResolvedLocal { .. }) {
                    return Err(ExternalLibraryError(
                        "local-ready resolution must use the app-owned resolved tier".to_owned(),
                    ));
                }
                artifact
                    .validate()
                    .map_err(|error| ExternalLibraryError(error.to_string()))
            }
            ModelAvailability::ExternalReady => {
                if self.local_artifact.is_some() {
                    return Err(ExternalLibraryError(
                        "external-ready resolution contains a local artifact".to_owned(),
                    ));
                }
                let binding = self.expected_library.as_ref().ok_or_else(|| {
                    ExternalLibraryError(
                        "external-ready resolution has no library binding".to_owned(),
                    )
                })?;
                if binding.schema_version != EXTERNAL_LIBRARY_CONTRACT_VERSION
                    || binding.configured_path != self.configured_library_path
                {
                    return Err(ExternalLibraryError(
                        "external-ready resolution does not match its library binding".to_owned(),
                    ));
                }
                validate_requirements_shape(&self.requirements)
            }
            ModelAvailability::InstalledExternalUnavailable => {
                if self.local_artifact.is_some() {
                    return Err(ExternalLibraryError(
                        "unavailable external resolution contains a local artifact".to_owned(),
                    ));
                }
                if let Some(binding) = &self.expected_library {
                    if binding.schema_version != EXTERNAL_LIBRARY_CONTRACT_VERSION
                        || binding.configured_path != self.configured_library_path
                    {
                        return Err(ExternalLibraryError(
                            "unavailable resolution does not match its library binding".to_owned(),
                        ));
                    }
                }
                if self.requirements.is_empty() {
                    Ok(())
                } else {
                    validate_requirements_shape(&self.requirements)
                }
            }
            ModelAvailability::Incomplete | ModelAvailability::Missing => {
                if self.expected_library.is_some() || self.local_artifact.is_some() {
                    return Err(ExternalLibraryError(
                        "not-ready resolution contains a ready artifact source".to_owned(),
                    ));
                }
                if self.requirements.is_empty() {
                    Ok(())
                } else {
                    validate_requirements_shape(&self.requirements)
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternalLibraryBindingStore {
    models_dir: PathBuf,
}

impl ExternalLibraryBindingStore {
    pub fn new(data_dir: &Path) -> Result<Self, ExternalLibraryError> {
        let models_dir = data_dir.join("models");
        std::fs::create_dir_all(&models_dir)?;
        let metadata = std::fs::symlink_metadata(&models_dir)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(ExternalLibraryError(
                "model directory is not a regular directory".to_owned(),
            ));
        }
        Ok(Self { models_dir })
    }

    pub fn load(&self) -> Result<Option<ExternalLibraryBinding>, ExternalLibraryError> {
        let _lock = self.lock_shared()?;
        self.read_unlocked()
    }

    /// Exact immutable closures that were previously validated on the bound physical library.
    /// This is a binding ledger, not an install receipt: it never changes receipt files and is used
    /// only to preserve typed installed identity while the source cannot be mounted.
    pub fn validated_closures(
        &self,
    ) -> Result<Vec<Vec<ExternalArtifactRequirement>>, ExternalLibraryError> {
        let _lock = self.lock_shared()?;
        self.read_validated_closures_unlocked()
    }

    pub fn probe_bound(
        &self,
        configured_path: &Path,
        binding: &ExternalLibraryBinding,
    ) -> ExternalLibraryProbe {
        probe_binding(configured_path, binding)
    }

    /// Bind a previously-unbound legacy library only after every exact receipt requirement is
    /// present. Existing bindings are never replaced: a new device at the same path stays unavailable.
    pub fn bind_or_probe_validated(
        &self,
        configured_path: &Path,
        requirements: &[ExternalArtifactRequirement],
    ) -> Result<(ExternalLibraryBinding, ExternalLibraryProbe), ExternalLibraryError> {
        validate_requirements_shape(requirements)?;
        let _lock = self.lock_exclusive()?;
        if let Some(binding) = self.read_unlocked()? {
            let probe = probe_binding(configured_path, &binding);
            if probe.status == ExternalLibraryProbeStatus::Available {
                validate_requirements_at_root(&binding.canonical_path, requirements)?;
                self.remember_validated_closure_unlocked(requirements)?;
            }
            return Ok((binding, probe));
        }

        let configured_path = absolute_lexical(configured_path)?;
        let canonical_before = std::fs::canonicalize(&configured_path).map_err(|error| {
            ExternalLibraryError(format!(
                "configured external model library {} is unavailable: {error}",
                configured_path.display()
            ))
        })?;
        ensure_regular_directory(&canonical_before)?;
        let identity_before = physical_identity(&canonical_before)?;
        validate_requirements_at_root(&canonical_before, requirements)?;
        let canonical_after = std::fs::canonicalize(&configured_path)?;
        let identity_after = physical_identity(&canonical_after)?;
        if canonical_before != canonical_after || identity_before != identity_after {
            return Err(ExternalLibraryError(
                "external model library changed while it was being validated".to_owned(),
            ));
        }
        let binding = ExternalLibraryBinding {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            configured_path,
            canonical_path: canonical_before,
            physical_identity: identity_before,
            bound_at: now_seconds()?,
        };
        self.write_unlocked(&binding)?;
        self.remember_validated_closure_unlocked(requirements)?;
        Ok((
            binding.clone(),
            ExternalLibraryProbe {
                status: ExternalLibraryProbeStatus::Available,
                observed_path: Some(binding.canonical_path.clone()),
                observed_identity: Some(binding.physical_identity.clone()),
            },
        ))
    }

    /// Deliberate, user-driven relocation of the model source library (sc-19709).
    ///
    /// [`Self::bind_or_probe_validated`] never replaces a binding — that is what makes a different
    /// physical volume at the same path fail closed. Relocation is the one explicit escape hatch:
    /// the operator names a new library root, and the binding is replaced ONLY after that root
    /// physically carries every closure this install previously validated. Download receipts and
    /// the validated-closure ledger are never touched, so relocation never redownloads anything and
    /// never loses installed state — it only re-points the durable identity at where the library
    /// now lives.
    pub fn relocate_binding(
        &self,
        library_root: &Path,
    ) -> Result<ExternalLibraryBinding, LibraryRelocationError> {
        let _lock = self
            .lock_exclusive()
            .map_err(LibraryRelocationError::Failed)?;
        let canonical_before = canonical_library_directory(library_root)?;
        if !has_repository_layout(&canonical_before) {
            return Err(LibraryRelocationError::Rejected(
                LibraryRelocationRejection::NotAModelLibrary,
            ));
        }
        let closures = self
            .read_validated_closures_unlocked()
            .map_err(LibraryRelocationError::Failed)?;
        let mut missing = Vec::new();
        for closure in &closures {
            if validate_requirements_at_root(&canonical_before, closure).is_err() {
                missing.extend(
                    closure
                        .iter()
                        .map(|requirement| requirement.repository.clone()),
                );
            }
        }
        if !missing.is_empty() {
            missing.sort();
            missing.dedup();
            return Err(LibraryRelocationError::Rejected(
                LibraryRelocationRejection::MissingInstalledModels {
                    repositories: missing,
                },
            ));
        }
        let identity_before = physical_identity(&canonical_before).map_err(|error| {
            LibraryRelocationError::Rejected(LibraryRelocationRejection::IdentityUnavailable {
                detail: error.0,
            })
        })?;
        // Same TOCTOU discipline as the initial bind: the closure walk touches many files and can
        // race an unmount, so the exact identity must still hold after validation.
        let canonical_after = canonical_library_directory(library_root)?;
        let identity_after = physical_identity(&canonical_after).map_err(|error| {
            LibraryRelocationError::Rejected(LibraryRelocationRejection::IdentityUnavailable {
                detail: error.0,
            })
        })?;
        if canonical_before != canonical_after || identity_before != identity_after {
            return Err(LibraryRelocationError::Failed(ExternalLibraryError(
                "external model library changed while it was being validated".to_owned(),
            )));
        }
        let binding = ExternalLibraryBinding {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            configured_path: absolute_lexical(library_root)
                .map_err(LibraryRelocationError::Failed)?,
            canonical_path: canonical_before,
            physical_identity: identity_before,
            bound_at: now_seconds().map_err(LibraryRelocationError::Failed)?,
        };
        self.write_unlocked(&binding)
            .map_err(LibraryRelocationError::Failed)?;
        Ok(binding)
    }

    pub fn probe_resolution(
        &self,
        resolution: &ModelResolution,
    ) -> Result<ExternalLibraryProbe, ExternalLibraryError> {
        resolution.validate()?;
        if resolution.availability == ModelAvailability::LocalReady {
            return Ok(ExternalLibraryProbe {
                status: ExternalLibraryProbeStatus::Available,
                observed_path: resolution
                    .local_artifact
                    .as_ref()
                    .map(|artifact| artifact.location.root().to_path_buf()),
                observed_identity: None,
            });
        }
        let binding = resolution.expected_library.as_ref().ok_or_else(|| {
            ExternalLibraryError("external resolution has no expected library binding".to_owned())
        })?;
        let durable_binding = self.load()?.ok_or_else(|| {
            ExternalLibraryError("external resolution has no durable library binding".to_owned())
        })?;
        if binding != &durable_binding {
            return Ok(ExternalLibraryProbe {
                status: ExternalLibraryProbeStatus::IdentityMismatch,
                observed_path: Some(durable_binding.canonical_path),
                observed_identity: Some(durable_binding.physical_identity),
            });
        }
        let probe_before = probe_binding(&resolution.configured_library_path, binding);
        if probe_before.status == ExternalLibraryProbeStatus::Available {
            validate_requirements_at_root(&binding.canonical_path, &resolution.requirements)?;
            // Validation walks multiple files and may race an unmount/remount. Identity is the
            // authority, so prove the exact binding a second time before admitting the loader.
            let probe_after = probe_binding(&resolution.configured_library_path, binding);
            if probe_after.status != ExternalLibraryProbeStatus::Available
                || probe_after.observed_path != probe_before.observed_path
                || probe_after.observed_identity != probe_before.observed_identity
            {
                return Ok(probe_after);
            }
        }
        Ok(probe_before)
    }

    fn ledger_path(&self) -> PathBuf {
        self.models_dir.join(LEDGER_FILE)
    }

    fn validated_closures_path(&self) -> PathBuf {
        self.models_dir.join(VALIDATED_CLOSURES_FILE)
    }

    fn read_validated_closures_unlocked(
        &self,
    ) -> Result<Vec<Vec<ExternalArtifactRequirement>>, ExternalLibraryError> {
        let path = self.validated_closures_path();
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(ExternalLibraryError(
                "validated external-library closure ledger is not a regular file".to_owned(),
            ));
        }
        let ledger: ValidatedExternalClosures = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|error| {
                ExternalLibraryError(format!("invalid validated closure ledger: {error}"))
            })?;
        if ledger.schema_version != EXTERNAL_LIBRARY_CONTRACT_VERSION {
            return Err(ExternalLibraryError(
                "unsupported validated closure ledger version".to_owned(),
            ));
        }
        for closure in &ledger.closures {
            validate_requirements_shape(closure)?;
        }
        Ok(ledger.closures)
    }

    fn remember_validated_closure_unlocked(
        &self,
        requirements: &[ExternalArtifactRequirement],
    ) -> Result<(), ExternalLibraryError> {
        let mut closures = self.read_validated_closures_unlocked()?;
        let closure = canonical_requirement_closure(requirements);
        if !closures.contains(&closure) {
            closures.push(closure);
            closures.sort_by(|left, right| format!("{left:?}").cmp(&format!("{right:?}")));
            let ledger = ValidatedExternalClosures {
                schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
                closures,
            };
            let mut bytes = serde_json::to_vec_pretty(&ledger)
                .map_err(|error| ExternalLibraryError(error.to_string()))?;
            bytes.push(b'\n');
            atomic_write(&self.validated_closures_path(), &bytes)
                .map_err(|error| ExternalLibraryError(error.to_string()))?;
        }
        Ok(())
    }

    fn read_unlocked(&self) -> Result<Option<ExternalLibraryBinding>, ExternalLibraryError> {
        let path = self.ledger_path();
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(ExternalLibraryError(
                "external-library binding ledger is not a regular file".to_owned(),
            ));
        }
        let binding: ExternalLibraryBinding = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|error| ExternalLibraryError(format!("invalid binding ledger: {error}")))?;
        if binding.schema_version != EXTERNAL_LIBRARY_CONTRACT_VERSION {
            return Err(ExternalLibraryError(
                "unsupported external-library binding version".to_owned(),
            ));
        }
        Ok(Some(binding))
    }

    fn write_unlocked(&self, binding: &ExternalLibraryBinding) -> Result<(), ExternalLibraryError> {
        let mut bytes = serde_json::to_vec_pretty(binding)
            .map_err(|error| ExternalLibraryError(error.to_string()))?;
        bytes.push(b'\n');
        atomic_write(&self.ledger_path(), &bytes)
            .map_err(|error| ExternalLibraryError(error.to_string()))
    }

    fn lock_shared(&self) -> Result<File, ExternalLibraryError> {
        let file = self.open_lock()?;
        FileExt::lock_shared(&file)?;
        Ok(file)
    }

    fn lock_exclusive(&self) -> Result<File, ExternalLibraryError> {
        let file = self.open_lock()?;
        FileExt::lock_exclusive(&file)?;
        Ok(file)
    }

    fn open_lock(&self) -> Result<File, ExternalLibraryError> {
        let path = self.models_dir.join(LEDGER_LOCK);
        if std::fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err(ExternalLibraryError(
                "external-library binding lock is a symlink".to_owned(),
            ));
        }
        Ok(OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?)
    }
}

/// Durable, operation-owned source session. The session's staging directory is the only path its
/// cleanup may remove; source-library bytes and unrelated partial roots are never scanned or deleted.
#[derive(Debug)]
pub struct ExternalSourceSession {
    session_id: String,
    session_root: PathBuf,
    staging_root: PathBuf,
    complete: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SourceSessionRecord<'a> {
    schema_version: u32,
    session_id: &'a str,
    resolution: &'a ModelResolution,
}

impl ExternalSourceSession {
    pub fn begin(
        data_dir: &Path,
        resolution: &ModelResolution,
    ) -> Result<Self, ExternalLibraryError> {
        if resolution.availability != ModelAvailability::ExternalReady {
            return Err(ExternalLibraryError(
                "source session requires an external-ready resolution".to_owned(),
            ));
        }
        let store = ExternalLibraryBindingStore::new(data_dir)?;
        let probe = store.probe_resolution(resolution)?;
        if probe.status != ExternalLibraryProbeStatus::Available {
            return Err(ExternalLibraryError(
                EXTERNAL_LIBRARY_UNAVAILABLE_CODE.to_owned(),
            ));
        }
        let sessions = store.models_dir.join(SESSION_DIR);
        std::fs::create_dir_all(&sessions)?;
        ensure_regular_directory(&sessions)?;
        let session_id = random_hex(16).map_err(|error| ExternalLibraryError(error.to_string()))?;
        let session_root = sessions.join(&session_id);
        std::fs::create_dir(&session_root)?;
        let staging_root = session_root.join("staging");
        std::fs::create_dir(&staging_root)?;
        let record = SourceSessionRecord {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            session_id: &session_id,
            resolution,
        };
        let mut bytes = serde_json::to_vec_pretty(&record)
            .map_err(|error| ExternalLibraryError(error.to_string()))?;
        bytes.push(b'\n');
        atomic_write(&session_root.join("session.json"), &bytes)
            .map_err(|error| ExternalLibraryError(error.to_string()))?;
        Ok(Self {
            session_id,
            session_root,
            staging_root,
            complete: false,
        })
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn staging_root(&self) -> &Path {
        &self.staging_root
    }

    pub fn mark_success(mut self) -> Result<(), ExternalLibraryError> {
        self.cleanup()?;
        self.complete = true;
        Ok(())
    }

    pub fn cancel(mut self) -> Result<(), ExternalLibraryError> {
        self.cleanup()?;
        self.complete = true;
        Ok(())
    }

    fn cleanup(&self) -> Result<(), ExternalLibraryError> {
        let parent = self.session_root.parent().ok_or_else(|| {
            ExternalLibraryError("source session has no managed parent".to_owned())
        })?;
        if !is_lower_hex_32(&self.session_id)
            || self
                .session_root
                .file_name()
                .and_then(|value| value.to_str())
                != Some(self.session_id.as_str())
        {
            return Err(ExternalLibraryError(
                "invalid source session ownership".to_owned(),
            ));
        }
        ensure_regular_directory(parent)?;
        ensure_regular_directory(&self.session_root)?;
        let canonical_parent = std::fs::canonicalize(parent)?;
        let canonical_session = std::fs::canonicalize(&self.session_root)?;
        if canonical_session.parent() != Some(canonical_parent.as_path()) {
            return Err(ExternalLibraryError(
                "source session escaped its managed parent".to_owned(),
            ));
        }
        // First move the exact owned directory entry within its managed parent. After this atomic
        // rename, reuse of the public session path cannot redirect cleanup. This story's staging
        // contract is flat, so remove only known regular-file entries; unexpected nested directories
        // are retained for explicit recovery instead of recursively traversed.
        let cleanup_root = parent.join(format!(".cleanup-{}", self.session_id));
        if std::fs::symlink_metadata(&cleanup_root).is_ok() {
            return Err(ExternalLibraryError(
                "source session cleanup destination already exists".to_owned(),
            ));
        }
        std::fs::rename(&self.session_root, &cleanup_root)?;
        cleanup_owned_source_session(&cleanup_root)?;
        Ok(())
    }
}

fn cleanup_owned_source_session(session_root: &Path) -> Result<(), ExternalLibraryError> {
    let staging = session_root.join("staging");
    let staging_metadata = std::fs::symlink_metadata(&staging)?;
    if metadata_is_reparse(&staging_metadata) {
        // The owned staging path is always a directory. On Windows both directory symlinks and
        // junctions are reparse points and must be unlinked with remove_dir; this removes the
        // reparse entry itself without walking its external target.
        std::fs::remove_dir(&staging)?;
    } else if staging_metadata.file_type().is_symlink() {
        std::fs::remove_file(&staging)?;
    } else if staging_metadata.is_dir() {
        for entry in std::fs::read_dir(&staging)? {
            let entry = entry?;
            let metadata = entry.file_type()?;
            if metadata.is_file() || metadata.is_symlink() {
                std::fs::remove_file(entry.path())?;
            } else {
                return Err(ExternalLibraryError(
                    "source session staging contains an unmanaged nested directory".to_owned(),
                ));
            }
        }
        std::fs::remove_dir(&staging)?;
    } else {
        return Err(ExternalLibraryError(
            "source session staging is not a managed directory".to_owned(),
        ));
    }

    let record = session_root.join("session.json");
    match std::fs::symlink_metadata(&record) {
        Ok(metadata) if metadata.is_file() || metadata.file_type().is_symlink() => {
            std::fs::remove_file(&record)?;
        }
        Ok(_) => {
            return Err(ExternalLibraryError(
                "source session record is not a regular file".to_owned(),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    std::fs::remove_dir(session_root)?;
    Ok(())
}

impl Drop for ExternalSourceSession {
    fn drop(&mut self) {
        if !self.complete {
            let _ = self.cleanup();
        }
    }
}

/// Typed, machine-readable reason a candidate model-library root was rejected for relocation
/// (sc-19709). The desktop prompt renders its guidance from this discriminant — a client must never
/// have to parse an error string, and a raw filesystem error must never reach the user.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum LibraryRelocationRejection {
    /// The chosen path does not exist, is not a directory, or could not be opened.
    NotADirectory,
    /// The chosen directory carries no Hugging Face `models--<repo>` layout at all — the classic
    /// "picked an unrelated folder" case.
    NotAModelLibrary,
    /// The layout is right, but models this install recorded as present are not in that library.
    /// Accepting it would silently orphan installed weights, so it fails closed.
    MissingInstalledModels { repositories: Vec<String> },
    /// The library validates, but the app cannot express it as a Hugging Face cache home: the
    /// configured root is always `<HF_HOME>/hub`, so the operator must choose the folder that
    /// CONTAINS `hub` (or the `hub` folder itself).
    HubDirectoryExpected,
    /// The volume's durable physical identity could not be read, so no binding can be written.
    IdentityUnavailable { detail: String },
}

/// Relocation failures split into "the operator can fix this by choosing differently"
/// ([`LibraryRelocationError::Rejected`]) and "the app failed" ([`LibraryRelocationError::Failed`]).
/// Only the former is ever rendered to a user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibraryRelocationError {
    Rejected(LibraryRelocationRejection),
    Failed(ExternalLibraryError),
}

impl std::fmt::Display for LibraryRelocationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(rejection) => write!(formatter, "{rejection:?}"),
            Self::Failed(error) => formatter.write_str(&error.0),
        }
    }
}

/// A validated relocation choice: the Hugging Face hub root SceneWorks binds, plus the `HF_HOME`
/// value that resolves to it. The two are always related as `library_root == hf_home/hub`
/// because that is exactly what [`crate::hf_home::huggingface_hub_cache_dir`] computes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RelocationTarget {
    pub library_root: PathBuf,
    pub hf_home: PathBuf,
}

/// Map a folder the operator picked to the pair of paths relocation needs.
///
/// Two shapes are accepted, in this order: the picked folder is a Hugging Face cache home (it has a
/// `hub` child carrying the repository layout), or the picked folder IS the hub root (named `hub`).
/// A directory that carries the layout but cannot be addressed as `<HF_HOME>/hub` is rejected with
/// [`LibraryRelocationRejection::HubDirectoryExpected`] rather than silently binding a root the app
/// could never resolve again after a restart.
pub fn resolve_relocation_target(
    picked: &Path,
) -> Result<RelocationTarget, LibraryRelocationRejection> {
    let picked = absolute_lexical(picked).map_err(|_| LibraryRelocationRejection::NotADirectory)?;
    let canonical =
        std::fs::canonicalize(&picked).map_err(|_| LibraryRelocationRejection::NotADirectory)?;
    if !canonical.is_dir() {
        return Err(LibraryRelocationRejection::NotADirectory);
    }
    let hub = picked.join("hub");
    if std::fs::canonicalize(&hub)
        .map(|path| has_repository_layout(&path))
        .unwrap_or(false)
    {
        return Ok(RelocationTarget {
            library_root: hub,
            hf_home: picked,
        });
    }
    if !has_repository_layout(&canonical) {
        return Err(LibraryRelocationRejection::NotAModelLibrary);
    }
    let is_hub = picked
        .file_name()
        .is_some_and(|name| name.eq_ignore_ascii_case("hub"));
    let parent = picked.parent().map(Path::to_path_buf);
    match (is_hub, parent) {
        (true, Some(home)) => Ok(RelocationTarget {
            library_root: picked,
            hf_home: home,
        }),
        _ => Err(LibraryRelocationRejection::HubDirectoryExpected),
    }
}

/// Does `root` look like a Hugging Face hub cache at all? One `models--<repo>` directory is enough:
/// this only separates "an unrelated folder" from "a model library", never "the RIGHT library" —
/// that judgement belongs to the recorded closures.
fn has_repository_layout(root: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with("models--"))
            && entry.file_type().is_ok_and(|kind| kind.is_dir())
    })
}

fn canonical_library_directory(root: &Path) -> Result<PathBuf, LibraryRelocationError> {
    let canonical = std::fs::canonicalize(root)
        .map_err(|_| LibraryRelocationError::Rejected(LibraryRelocationRejection::NotADirectory))?;
    ensure_regular_directory(&canonical)
        .map_err(|_| LibraryRelocationError::Rejected(LibraryRelocationRejection::NotADirectory))?;
    Ok(canonical)
}

/// Live status of the configured model source library (sc-19709). The desktop prompt's
/// "Connect drive and retry" re-probe reads exactly this — one cheap, write-free identity probe
/// rather than a catalog rebuild.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelSourceLibraryStatus {
    pub schema_version: u32,
    pub configured_library_path: PathBuf,
    pub expected_library: Option<ExternalLibraryBinding>,
    pub probe_status: ExternalLibraryProbeStatus,
    /// The single boolean a retry gates on: the expected library is verified present.
    pub available: bool,
}

/// Probe the configured library once, write-free. With a durable binding this is an identity
/// probe; without one, availability is simply whether the configured root is a readable directory
/// (nothing is installed there yet, so there is no identity to prove).
pub fn probe_model_source_library(
    data_dir: &Path,
    configured_library: &Path,
) -> ModelSourceLibraryStatus {
    let binding = ExternalLibraryBindingStore::new(data_dir)
        .ok()
        .and_then(|store| store.load().ok())
        .flatten();
    let (probe_status, available) = match &binding {
        Some(binding) => {
            let status = probe_binding(configured_library, binding).status;
            let available = status == ExternalLibraryProbeStatus::Available;
            (status, available)
        }
        None => {
            let present = configured_library.is_dir();
            (
                if present {
                    ExternalLibraryProbeStatus::Available
                } else {
                    ExternalLibraryProbeStatus::Unavailable
                },
                present,
            )
        }
    };
    ModelSourceLibraryStatus {
        schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
        configured_library_path: configured_library.to_path_buf(),
        expected_library: binding,
        probe_status,
        available,
    }
}

/// The typed payload the API attaches to an `external_model_library_unavailable` rejection
/// (sc-19709). The desktop prompt names the model and the expected library location from THESE
/// fields — never by parsing `detail`, and never by re-deriving availability client-side.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExternalLibraryUnavailableContext {
    pub schema_version: u32,
    pub availability: ModelAvailability,
    pub model_id: String,
    pub model_name: Option<String>,
    pub configured_library_path: PathBuf,
    pub expected_library_path: Option<PathBuf>,
    pub expected_volume_id: Option<String>,
}

impl ExternalLibraryUnavailableContext {
    pub fn from_resolution(
        model_id: impl Into<String>,
        model_name: Option<String>,
        resolution: &ModelResolution,
    ) -> Self {
        Self {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            availability: resolution.availability.clone(),
            model_id: model_id.into(),
            model_name,
            configured_library_path: resolution.configured_library_path.clone(),
            expected_library_path: resolution
                .expected_library
                .as_ref()
                .map(|binding| binding.canonical_path.clone()),
            expected_volume_id: resolution
                .expected_library
                .as_ref()
                .map(|binding| binding.physical_identity.volume_id.clone()),
        }
    }
}

pub fn probe_binding(
    configured_path: &Path,
    binding: &ExternalLibraryBinding,
) -> ExternalLibraryProbe {
    let configured_path = match absolute_lexical(configured_path) {
        Ok(path) => path,
        Err(_) => return unknown_probe(),
    };
    if configured_path != binding.configured_path {
        return ExternalLibraryProbe {
            status: ExternalLibraryProbeStatus::IdentityMismatch,
            observed_path: None,
            observed_identity: None,
        };
    }
    let canonical = match std::fs::canonicalize(&configured_path) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return ExternalLibraryProbe {
                status: ExternalLibraryProbeStatus::Unavailable,
                observed_path: None,
                observed_identity: None,
            }
        }
        Err(_) => return unknown_probe(),
    };
    let identity = match physical_identity(&canonical) {
        Ok(identity) => identity,
        Err(_) => return unknown_probe(),
    };
    let status = if canonical == binding.canonical_path && identity == binding.physical_identity {
        ExternalLibraryProbeStatus::Available
    } else {
        ExternalLibraryProbeStatus::IdentityMismatch
    };
    ExternalLibraryProbe {
        status,
        observed_path: Some(canonical),
        observed_identity: Some(identity),
    }
}

pub fn validate_requirements_at_root(
    root: &Path,
    requirements: &[ExternalArtifactRequirement],
) -> Result<(), ExternalLibraryError> {
    validate_requirements_shape(requirements)?;
    let canonical_root = std::fs::canonicalize(root)?;
    for requirement in requirements {
        let repo_name = safe_repo_dir_name(&requirement.repository)
            .ok_or_else(|| ExternalLibraryError("invalid external repository".to_owned()))?;
        let repo_root = canonical_root.join(format!("models--{repo_name}"));
        let canonical_repo = std::fs::canonicalize(&repo_root).map_err(|error| {
            ExternalLibraryError(format!(
                "external repository {} is unavailable: {error}",
                requirement.repository
            ))
        })?;
        if !canonical_repo.starts_with(&canonical_root) {
            return Err(ExternalLibraryError(
                "external repository escaped its configured library".to_owned(),
            ));
        }
        let snapshots = if let Some(revision) = requirement.revision.as_deref() {
            vec![canonical_repo.join("snapshots").join(revision)]
        } else {
            std::fs::read_dir(canonical_repo.join("snapshots"))?
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    entry
                        .file_type()
                        .ok()
                        .filter(|kind| kind.is_dir())
                        .map(|_| entry.path())
                })
                .collect()
        };
        let matching = snapshots
            .into_iter()
            .filter(|snapshot| {
                requirement.files.iter().all(|relative| {
                    let candidate = snapshot.join(relative);
                    let Ok(canonical) = std::fs::canonicalize(candidate) else {
                        return false;
                    };
                    canonical.is_file() && canonical.starts_with(&canonical_repo)
                })
            })
            .count();
        if matching != 1 {
            return Err(ExternalLibraryError(format!(
                "external artifact {} has {matching} matching snapshots; expected exactly one",
                requirement.repository
            )));
        }
    }
    Ok(())
}

fn validate_requirements_shape(
    requirements: &[ExternalArtifactRequirement],
) -> Result<(), ExternalLibraryError> {
    if requirements.is_empty() {
        return Err(ExternalLibraryError(
            "external artifact requirements are empty".to_owned(),
        ));
    }
    if requirements
        .iter()
        .filter(|requirement| requirement.is_primary)
        .count()
        != 1
    {
        return Err(ExternalLibraryError(
            "external artifact closure must contain exactly one primary requirement".to_owned(),
        ));
    }
    for requirement in requirements {
        if safe_repo_dir_name(&requirement.repository).is_none()
            || requirement.variant.trim().is_empty()
            || requirement.files.is_empty()
        {
            return Err(ExternalLibraryError(
                "external artifact requirement is incomplete".to_owned(),
            ));
        }
        if let Some(revision) = requirement.revision.as_deref() {
            if revision.len() != 40
                || !revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(ExternalLibraryError(
                    "external artifact revision is not immutable lowercase hex".to_owned(),
                ));
            }
        }
        if requirement.files.iter().any(|path| {
            path.as_os_str().is_empty()
                || path
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
        }) {
            return Err(ExternalLibraryError(
                "external artifact contains an unsafe relative file".to_owned(),
            ));
        }
    }
    Ok(())
}

fn ensure_regular_directory(path: &Path) -> Result<(), ExternalLibraryError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() || metadata_is_reparse(&metadata) {
        return Err(ExternalLibraryError(format!(
            "{} is not a regular directory",
            path.display()
        )));
    }
    Ok(())
}

fn absolute_lexical(path: &Path) -> Result<PathBuf, ExternalLibraryError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component)
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(ExternalLibraryError(
                        "configured library path escapes root".to_owned(),
                    ));
                }
            }
        }
    }
    Ok(normalized)
}

#[cfg(target_os = "macos")]
fn physical_identity(path: &Path) -> Result<ExternalLibraryPhysicalIdentity, ExternalLibraryError> {
    use std::os::unix::fs::MetadataExt;

    let directory = File::open(path)?;
    let metadata = directory.metadata()?;
    // SceneWorks forbids unsafe code in core. `diskutil -plist` is Apple's supported native volume
    // identity surface and returns the persistent VolumeUUID for the exact existing path (unlike
    // st_dev, which is only a mount-session number). No nearest-parent fallback is permitted.
    let df = macos_command_output(
        "df",
        std::process::Command::new("/bin/df")
            .arg("-P")
            .arg(path)
            .output(),
    )?;
    if !df.status.success() {
        return Err(ExternalLibraryError(format!(
            "df could not identify external-library mount: {}",
            String::from_utf8_lossy(&df.stderr).trim()
        )));
    }
    let df_stdout = String::from_utf8(df.stdout)
        .map_err(|error| ExternalLibraryError(format!("df output was not UTF-8: {error}")))?;
    let fields = df_stdout
        .lines()
        .last()
        .map(str::split_whitespace)
        .map(Iterator::collect::<Vec<_>>)
        .unwrap_or_default();
    if fields.len() < 6 {
        return Err(ExternalLibraryError(
            "df returned no exact external-library mount point".to_owned(),
        ));
    }
    let mount_point = fields[5..].join(" ");
    let output = macos_command_output(
        "diskutil",
        std::process::Command::new("/usr/sbin/diskutil")
            .args(["info", "-plist"])
            .arg(&mount_point)
            .output(),
    )?;
    let compact =
        macos_volume_uuid_from_diskutil(output.status.success(), &output.stdout, &output.stderr)?;
    Ok(ExternalLibraryPhysicalIdentity {
        volume_id: format!("macos-volume:{compact}"),
        directory_id: metadata.ino(),
    })
}

#[cfg(target_os = "macos")]
fn macos_command_output(
    command: &str,
    output: std::io::Result<std::process::Output>,
) -> Result<std::process::Output, ExternalLibraryError> {
    output.map_err(|error| {
        ExternalLibraryError(format!(
            "{command} is unavailable for external-library volume identity: {error}"
        ))
    })
}

#[cfg(target_os = "macos")]
fn macos_volume_uuid_from_diskutil(
    success: bool,
    stdout: &[u8],
    stderr: &[u8],
) -> Result<String, ExternalLibraryError> {
    if !success {
        return Err(ExternalLibraryError(format!(
            "diskutil could not identify external-library volume: {}",
            String::from_utf8_lossy(stderr).trim()
        )));
    }
    let plist = String::from_utf8(stdout.to_vec())
        .map_err(|error| ExternalLibraryError(format!("diskutil plist was not UTF-8: {error}")))?;
    let key = "<key>VolumeUUID</key>";
    let after_key = plist
        .split_once(key)
        .map(|(_, value)| value)
        .ok_or_else(|| {
            ExternalLibraryError("diskutil returned no persistent VolumeUUID".to_owned())
        })?;
    let after_key = after_key.trim_start();
    let start = after_key.strip_prefix("<string>").ok_or_else(|| {
        ExternalLibraryError("diskutil VolumeUUID had no string value".to_owned())
    })?;
    let end = start.find("</string>").ok_or_else(|| {
        ExternalLibraryError("diskutil VolumeUUID string was unterminated".to_owned())
    })?;
    let uuid = start[..end].trim().to_ascii_lowercase();
    let compact = uuid.replace('-', "");
    if compact.len() != 32
        || !compact
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ExternalLibraryError(
            "diskutil returned an invalid persistent VolumeUUID".to_owned(),
        ));
    }
    Ok(compact)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn physical_identity(path: &Path) -> Result<ExternalLibraryPhysicalIdentity, ExternalLibraryError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(path)?;
    Ok(ExternalLibraryPhysicalIdentity {
        volume_id: format!("unix-dev:{:016x}", metadata.dev()),
        directory_id: metadata.ino(),
    })
}

#[cfg(windows)]
fn physical_identity(path: &Path) -> Result<ExternalLibraryPhysicalIdentity, ExternalLibraryError> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    let handle = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    let information = winapi_util::file::information(&handle)
        .map_err(|error| ExternalLibraryError(error.to_string()))?;
    Ok(ExternalLibraryPhysicalIdentity {
        volume_id: format!("windows-volume:{:016x}", information.volume_serial_number()),
        directory_id: information.file_index(),
    })
}

#[cfg(not(any(unix, windows)))]
fn physical_identity(
    _path: &Path,
) -> Result<ExternalLibraryPhysicalIdentity, ExternalLibraryError> {
    Err(ExternalLibraryError(
        "physical external-library identity is unsupported on this platform".to_owned(),
    ))
}

#[cfg(windows)]
fn metadata_is_reparse(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    metadata.file_attributes() & 0x0000_0400 != 0
}

#[cfg(not(windows))]
fn metadata_is_reparse(_metadata: &std::fs::Metadata) -> bool {
    false
}

/// The one canonical ordering for a requirement closure: primary first, then by repository,
/// revision, variant and files. Both the validated-closures ledger writer and its readers sort
/// through this function, so ledger equality can never diverge from write-time ordering.
fn canonical_requirement_closure(
    requirements: &[ExternalArtifactRequirement],
) -> Vec<ExternalArtifactRequirement> {
    let mut closure = requirements.to_vec();
    closure.sort_by(|left, right| {
        (
            !left.is_primary,
            &left.repository,
            &left.revision,
            &left.variant,
            &left.files,
        )
            .cmp(&(
                !right.is_primary,
                &right.repository,
                &right.revision,
                &right.variant,
                &right.files,
            ))
    });
    closure
}

/// True when the validated-closures ledger records exactly this closure — the durable proof that
/// it once validated in full on the bound physical library. This is what keeps a receipt-less
/// legacy install typed as installed-but-unavailable across a disconnect instead of degrading to
/// `Missing` (which would silently re-enter the download path).
fn closure_has_validated_ledger_record(
    store: &ExternalLibraryBindingStore,
    requirements: &[ExternalArtifactRequirement],
) -> bool {
    let Ok(closures) = store.validated_closures() else {
        return false;
    };
    let canonical = canonical_requirement_closure(requirements);
    closures.contains(&canonical)
}

fn unknown_probe() -> ExternalLibraryProbe {
    ExternalLibraryProbe {
        status: ExternalLibraryProbeStatus::Unknown,
        observed_path: None,
        observed_identity: None,
    }
}

fn is_lower_hex_32(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn now_seconds() -> Result<u64, ExternalLibraryError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| ExternalLibraryError(error.to_string()))
}

/// An app-owned resolved-local artifact that fully covers `requirements`: the primary identity
/// matches the primary requirement exactly (repository, immutable revision when recorded, and
/// variant), and every requirement's file set is present in the artifact closure. Completeness is
/// judged against the exact selected closure only — sibling variants of the same model never widen
/// or narrow the answer.
pub fn local_artifact_for_requirements(
    artifacts: &[ResolvedModelArtifact],
    requirements: &[ExternalArtifactRequirement],
) -> Option<ResolvedModelArtifact> {
    if requirements.is_empty() {
        return None;
    }
    artifacts.iter().find_map(|artifact| {
        if !matches!(&artifact.location, ArtifactLocation::ResolvedLocal { .. }) {
            return None;
        }
        let primary_identity_matches = requirements
            .iter()
            .find(|requirement| requirement.is_primary)
            .is_some_and(|primary| {
                artifact.identity.repository == primary.repository
                    && primary
                        .revision
                        .as_deref()
                        .map_or(true, |revision| artifact.identity.revision == revision)
                    && artifact.identity.variant == primary.variant
                    && artifact.closure.members.iter().any(|member| {
                        member.role == super::ArtifactMemberRole::Primary
                            && member.source == artifact.identity
                    })
            });
        let covers = primary_identity_matches
            && requirements.iter().all(|requirement| {
                let available_files = artifact
                    .closure
                    .members
                    .iter()
                    .filter(|member| {
                        member.source.repository == requirement.repository
                            && requirement
                                .revision
                                .as_deref()
                                .map_or(true, |expected| member.source.revision == expected)
                            && member.source.variant == requirement.variant
                    })
                    .flat_map(|member| {
                        member
                            .files
                            .iter()
                            .map(|file| member.source_subpath.join(&file.relative_path))
                    })
                    .collect::<std::collections::HashSet<_>>();
                requirement
                    .files
                    .iter()
                    .all(|file| available_files.contains(file))
            });
        covers.then(|| artifact.clone())
    })
}

/// The single availability resolver. Catalog listing, submission preflight, and the worker's
/// pre-loader guard all judge one exact requirement closure through this function; no route or
/// model carries its own availability logic.
///
/// The decision ladder is uniform for every model:
/// 1. an app-owned resolved-local artifact that covers the exact closure → `LocalReady`;
/// 2. an empty closure (no checkable install identity) → `Missing` — the caller's established
///    download/on-demand behavior is preserved, never a typed disconnect;
/// 3. the bound (or bindable) physical library validates every requirement → `ExternalReady`;
/// 4. the library is physically present but the closure does not validate → `Incomplete`;
/// 5. the durable binding cannot be proven present (disconnected, or a different physical volume
///    now occupies the configured path) → `InstalledExternalUnavailable`. The install receipts and
///    binding ledger are never mutated on this path, so reconnecting restores `ExternalReady`.
///
/// `receipt_backed` is the strength of the closure's install evidence. The typed
/// `InstalledExternalUnavailable` state requires PROOF the model was installed: a durable download
/// receipt, or this exact closure recorded in the validated-closures ledger (written whenever the
/// closure validated on the bound physical library — the receipt-less legacy-install path). A
/// binding alone proves only that SOME library was bound, never that THIS model was installed on
/// it, so a declared-exact closure with neither receipts nor a ledger record resolves `Missing`
/// even while the bound volume is disconnected — a manifest declaration must not manufacture a
/// typed disconnect for a model that was never installed.
pub fn resolve_model_availability(
    data_dir: &Path,
    configured_library: &Path,
    requirements: &[ExternalArtifactRequirement],
    receipt_backed: bool,
    local_artifacts: &[ResolvedModelArtifact],
) -> ModelResolution {
    if let Some(artifact) = local_artifact_for_requirements(local_artifacts, requirements) {
        if let Ok(resolution) = ModelResolution::local_ready(artifact) {
            return resolution;
        }
    }
    if requirements.is_empty() {
        return ModelResolution {
            schema_version: EXTERNAL_LIBRARY_CONTRACT_VERSION,
            availability: ModelAvailability::Missing,
            configured_library_path: configured_library.to_path_buf(),
            expected_library: None,
            requirements: Vec::new(),
            local_artifact: None,
        };
    }
    let not_installed = || {
        ModelResolution::not_ready(
            ModelAvailability::Missing,
            configured_library.to_path_buf(),
            requirements.to_vec(),
        )
        .unwrap_or_else(|_| {
            ModelResolution::unavailable(
                configured_library.to_path_buf(),
                None,
                requirements.to_vec(),
            )
        })
    };
    let Ok(store) = ExternalLibraryBindingStore::new(data_dir) else {
        if !receipt_backed {
            return not_installed();
        }
        return ModelResolution::unavailable(
            configured_library.to_path_buf(),
            None,
            requirements.to_vec(),
        );
    };
    let installed_evidence =
        || receipt_backed || closure_has_validated_ledger_record(&store, requirements);
    match store.bind_or_probe_validated(configured_library, requirements) {
        Ok((binding, probe)) if probe.status == ExternalLibraryProbeStatus::Available => {
            ModelResolution::external_ready(
                configured_library.to_path_buf(),
                binding.clone(),
                requirements.to_vec(),
            )
            .unwrap_or_else(|_| {
                ModelResolution::unavailable(
                    configured_library.to_path_buf(),
                    Some(binding),
                    requirements.to_vec(),
                )
            })
        }
        Ok((binding, _)) if installed_evidence() => ModelResolution::unavailable(
            configured_library.to_path_buf(),
            Some(binding),
            requirements.to_vec(),
        ),
        // The bound volume is disconnected, but nothing proves THIS model was ever installed on
        // it (no receipt, no validated-closure record): it is a missing model, and its
        // established install path must stay reachable.
        Ok(_) => not_installed(),
        Err(_) => {
            let existing_binding = store.load().ok().flatten();
            let library_is_physically_present = existing_binding.as_ref().map_or_else(
                || configured_library.is_dir(),
                |binding| {
                    probe_binding(configured_library, binding).status
                        == ExternalLibraryProbeStatus::Available
                },
            );
            if library_is_physically_present {
                ModelResolution::not_ready(
                    ModelAvailability::Incomplete,
                    configured_library.to_path_buf(),
                    requirements.to_vec(),
                )
                .unwrap_or_else(|_| {
                    ModelResolution::unavailable(
                        configured_library.to_path_buf(),
                        existing_binding.clone(),
                        requirements.to_vec(),
                    )
                })
            } else if !installed_evidence() {
                // No install proof (no receipt, no validated-closure record): nothing was ever
                // installed here regardless of whether some binding exists. The declared manifest
                // identity alone must not manufacture a typed disconnect.
                not_installed()
            } else {
                // Receipt/ledger identity is durable even while the configured source cannot be
                // opened. Keep it installed-but-unavailable; never rewrite receipts or download.
                ModelResolution::unavailable(
                    configured_library.to_path_buf(),
                    existing_binding,
                    requirements.to_vec(),
                )
            }
        }
    }
}

#[cfg(test)]
#[path = "external_library_tests.rs"]
mod tests;
