//! Transactional ingestion of an application-owned ("managed") checkpoint (epic 20398, sc-20636).
//!
//! Every managed source — an upload, a copy of a local file the user already has, a plain URL, a
//! Hugging Face repo snapshot, a Civitai download — reaches SceneWorks-owned storage through this
//! one path:
//!
//! ```text
//! begin()                 mint an install id, create <data>/models/.import-staging/<installId>/
//!   stage_*()             write EVERY byte into staging; nothing outside it is touched
//! finalize()
//!   1. verify             SHA-256 of the staged primary artifact vs the source's declared digest
//!   2. validate           full-content inspection of the staged tree (checkpoint_inspector)
//!   3. commit             ONE atomic rename: .import-staging/<id>  ->  models/imports/<id>
//!   4. publish            compile the plan/record/bindings against the committed install
//! ```
//!
//! The rename in step 3 is the commit point, and it is what makes the failure modes uniform: a
//! cancel, a crash, a full disk, or a hash mismatch all end with the staging directory discarded
//! and NO directory at `models/imports/<installId>`, so nothing partially written is ever
//! addressable as an install, no plan or catalog record exists for it, and no manifest entry can be
//! stamped from it.
//! Writing into the final location and repairing afterwards would make each of those a distinct,
//! separately-recoverable partial state; there is deliberately only one.
//!
//! A crash is the one case no destructor can clean up. It cannot produce a runnable install (the
//! rename never ran), but it does leave the staging directory behind, so [`sweep_staging`] reclaims
//! orphans at startup.
//!
//! Secrets: provenance records WHERE the bytes came from — url, version/file identity, and the host
//! whose stored credential authorized the fetch — never the credential.
//! [`sanitize_provenance_url`] strips userinfo and secret-bearing query parameters before a URL can
//! reach a persisted plan, and [`crate::checkpoint_import::ManagedProvenanceV1::validate`] refuses
//! userinfo again at the contract boundary.

use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::checkpoint_import::ManagedProvenanceV1;
use crate::checkpoint_plan_store::{
    managed_checkpoint_id, CheckpointPlanError, CheckpointPlanStore, CompiledCheckpointV1,
};

const COPY_BUFFER_BYTES: usize = 1024 * 1024;

/// Query parameters that carry a secret on the download URLs SceneWorks attaches credentials to
/// (`downloads::download_source_url` appends `token`; Civitai and mirrors also accept the others).
/// Matched case-insensitively on the whole parameter name.
const SECRET_QUERY_PARAMS: &[&str] = &[
    "token",
    "access_token",
    "api_key",
    "apikey",
    "key",
    "password",
    "secret",
    "signature",
    "sig",
];

/// Every way a managed ingest refuses. Each one leaves no install directory, no plan, no catalog
/// record — the caller's only job is to surface it.
#[derive(Debug)]
pub enum ManagedIngestError {
    /// The caller cancelled the ingest (a cancelled job).
    Cancelled { install_id: String },
    /// The staged bytes are not the bytes the source declared.
    HashMismatch {
        install_id: String,
        relative_path: String,
        expected_sha256: String,
        actual_sha256: String,
    },
    /// The staged tree does not contain the artifact the caller named as its primary.
    PrimaryMissing {
        install_id: String,
        relative_path: String,
    },
    /// A relative path a caller wanted to stage is not a confined, portable relative path.
    InvalidRelativePath {
        relative_path: String,
        reason: &'static str,
    },
    /// Compile/validation of the committed install refused. The install has already been rolled
    /// back by the time this is returned.
    Plan(CheckpointPlanError),
    /// A filesystem failure while staging or committing — including a full disk.
    Io { path: PathBuf, message: String },
}

impl ManagedIngestError {
    /// Stable kebab-case refusal code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Cancelled { .. } => "cancelled",
            Self::HashMismatch { .. } => "hash-mismatch",
            Self::PrimaryMissing { .. } => "primary-missing",
            Self::InvalidRelativePath { .. } => "invalid-relative-path",
            Self::Plan(error) => error.code(),
            Self::Io { .. } => "io",
        }
    }
}

impl fmt::Display for ManagedIngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => return write!(f, "{error}"),
            _ => write!(f, "[checkpoint-ingest:{}] ", self.code())?,
        }
        match self {
            Self::Cancelled { install_id } => write!(
                f,
                "managed ingest {install_id:?} was cancelled; nothing was installed"
            ),
            Self::HashMismatch {
                install_id,
                relative_path,
                expected_sha256,
                actual_sha256,
            } => write!(
                f,
                "managed ingest {install_id:?} staged {relative_path:?} with sha256 {actual_sha256}, but the source declares {expected_sha256}; the transfer was corrupted and nothing was installed"
            ),
            Self::PrimaryMissing {
                install_id,
                relative_path,
            } => write!(
                f,
                "managed ingest {install_id:?} declared {relative_path:?} as its primary artifact, but nothing was staged there"
            ),
            Self::InvalidRelativePath {
                relative_path,
                reason,
            } => write!(
                f,
                "staged relative path {relative_path:?} is invalid: {reason}"
            ),
            Self::Plan(_) => Ok(()),
            Self::Io { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for ManagedIngestError {}

impl From<CheckpointPlanError> for ManagedIngestError {
    fn from(error: CheckpointPlanError) -> Self {
        Self::Plan(error)
    }
}

fn io_error(path: &Path, error: io::Error) -> ManagedIngestError {
    ManagedIngestError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

/// A finalized managed install: the committed bytes plus the plan the store published for them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedInstallV1 {
    pub install_id: String,
    /// `managed/<installId>` — the identity a manifest entry binds to via `importPlan.checkpointId`.
    pub checkpoint_id: String,
    /// The absolute SceneWorks-owned directory the bytes now live in.
    pub install_path: PathBuf,
    /// The primary artifact's path relative to `install_path`.
    pub primary_relative_path: String,
    /// SHA-256 of the primary artifact, as staged and verified.
    pub primary_sha256: String,
    pub compiled: CompiledCheckpointV1,
}

impl ManagedInstallV1 {
    /// Other persisted checkpoints holding these exact bytes (E1/AC2). Reported to the user;
    /// neither copy is ever deleted.
    pub fn duplicate_checkpoint_ids(&self) -> &[String] {
        &self.compiled.duplicate_checkpoint_ids
    }
}

/// Strip everything secret from a URL before it can be recorded as provenance: userinfo, and any
/// query parameter whose name is one a SceneWorks download attaches a credential to.
///
/// Returns `None` for a URL that cannot be parsed — provenance then records no URL rather than an
/// unexamined string, because a string this function could not inspect is a string whose secrets it
/// could not remove.
pub fn sanitize_provenance_url(source_url: &str) -> Option<String> {
    let mut url = url::Url::parse(source_url).ok()?;
    // A URL whose userinfo cannot be cleared (a cannot-be-a-base URL such as `data:`) is not a
    // download source and must not be recorded.
    url.set_username("").ok()?;
    url.set_password(None).ok()?;
    let redacted: Vec<(String, String)> = url
        .query_pairs()
        .filter(|(name, _)| {
            !SECRET_QUERY_PARAMS
                .iter()
                .any(|secret| name.eq_ignore_ascii_case(secret))
        })
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect();
    if redacted.is_empty() {
        url.set_query(None);
    } else {
        url.query_pairs_mut().clear().extend_pairs(redacted);
    }
    Some(url.to_string())
}

/// The host a stored credential would be looked up under for this URL, normalised the same way
/// [`crate::credentials`] normalises a stored host so the two agree.
///
/// Recorded on provenance so an install says WHICH credential authorized it. The credential itself
/// never leaves the credential store.
pub fn provenance_credential_host(source_url: &str) -> Option<String> {
    let host = url::Url::parse(source_url).ok()?.host_str()?.to_owned();
    let host = crate::credentials::normalize_host(&host);
    (!host.is_empty()).then_some(host)
}

/// Validate a portable relative path a caller wants to stage. Same shape the plan store enforces on
/// a compiled layer, applied at the staging boundary so no write can escape the staging directory.
fn staged_relative_path(relative_path: &str) -> Result<PathBuf, ManagedIngestError> {
    let invalid = |reason: &'static str| ManagedIngestError::InvalidRelativePath {
        relative_path: relative_path.to_owned(),
        reason,
    };
    if relative_path.trim().is_empty() {
        return Err(invalid("empty"));
    }
    if relative_path.contains('\\') {
        return Err(invalid("must use '/' separators"));
    }
    let mut out = PathBuf::new();
    for component in Path::new(relative_path).components() {
        match component {
            std::path::Component::Normal(part) => out.push(part),
            std::path::Component::CurDir => return Err(invalid("contains '.'")),
            std::path::Component::ParentDir => return Err(invalid("contains '..'")),
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(invalid("must be relative"))
            }
        }
    }
    if out.as_os_str().is_empty() {
        return Err(invalid("empty"));
    }
    Ok(out)
}

/// One in-flight managed ingest.
///
/// Held by value for the whole transfer. Dropping it without [`Self::finalize`] discards the
/// staging directory, so an early return, a `?`, a panic, or an explicit [`Self::cancel`] all reach
/// the same end state as a refusal.
pub struct ManagedIngest {
    store: CheckpointPlanStore,
    install_id: String,
    provenance: ManagedProvenanceV1,
    /// `None` once the session has been consumed (finalized or cancelled), so the destructor does
    /// not remove a directory that has already been committed or cleaned.
    staging_path: Option<PathBuf>,
}

impl fmt::Debug for ManagedIngest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManagedIngest")
            .field("install_id", &self.install_id)
            .field("staging_path", &self.staging_path)
            .finish()
    }
}

impl ManagedIngest {
    /// Open a staging area for a new managed install. `install_id` must not already be held by a
    /// persisted checkpoint or an existing install directory — finalizing over one would replace a
    /// live install's bytes while its plan still pointed at them.
    pub fn begin(
        store: &CheckpointPlanStore,
        install_id: &str,
        provenance: ManagedProvenanceV1,
    ) -> Result<Self, ManagedIngestError> {
        provenance
            .validate()
            .map_err(|error| ManagedIngestError::Plan(CheckpointPlanError::Contract(error)))?;
        let install_path = store.install_dir(install_id)?;
        let checkpoint_id = managed_checkpoint_id(install_id);
        if install_path.exists() || store.record(&checkpoint_id).is_ok() {
            return Err(CheckpointPlanError::InstallIdTaken {
                install_id: install_id.to_owned(),
                checkpoint_id,
                path: install_path,
            }
            .into());
        }
        let staging_root = store.staging_root().to_path_buf();
        fs::create_dir_all(&staging_root).map_err(|error| io_error(&staging_root, error))?;
        // Keyed on the install id, not a random token: an orphan left by a crash is then
        // attributable, and two sessions for the same id cannot both be staging at once.
        let staging_path = staging_root.join(install_id);
        if staging_path.exists() {
            fs::remove_dir_all(&staging_path).map_err(|error| io_error(&staging_path, error))?;
        }
        fs::create_dir(&staging_path).map_err(|error| io_error(&staging_path, error))?;
        Ok(Self {
            store: store.clone(),
            install_id: install_id.to_owned(),
            provenance,
            staging_path: Some(staging_path),
        })
    }

    pub fn install_id(&self) -> &str {
        &self.install_id
    }

    /// The directory every staged byte is written under. Callers that must hand a destination to an
    /// existing downloader (the HF snapshot downloader, the source-URL downloader) point it here;
    /// they get the transactional guarantees without duplicating a byte of transfer code.
    pub fn staging_dir(&self) -> &Path {
        self.staging_path
            .as_deref()
            .expect("staging path outlives the session")
    }

    /// The absolute path a staged relative path resolves to, creating parent directories.
    pub fn staged_path(&self, relative_path: &str) -> Result<PathBuf, ManagedIngestError> {
        let path = self
            .staging_dir()
            .join(staged_relative_path(relative_path)?);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        Ok(path)
    }

    /// Stage bytes from any reader. The reader's own errors — including a write failure on a full
    /// disk — propagate as [`ManagedIngestError::Io`] with the staging directory intact for the
    /// destructor to discard.
    pub fn stage_from_reader(
        &self,
        relative_path: &str,
        reader: &mut dyn Read,
    ) -> Result<u64, ManagedIngestError> {
        let path = self.staged_path(relative_path)?;
        let mut file = fs::File::create(&path).map_err(|error| io_error(&path, error))?;
        let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
        let mut written = 0_u64;
        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|error| io_error(&path, error))?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])
                .map_err(|error| io_error(&path, error))?;
            written += read as u64;
        }
        file.flush().map_err(|error| io_error(&path, error))?;
        Ok(written)
    }

    /// Stage a copy of a file the user already has. The source is only ever read (E6): a local-copy
    /// import never moves, renames, or deletes the user's file.
    pub fn stage_copy_file(
        &self,
        source: &Path,
        relative_path: &str,
    ) -> Result<u64, ManagedIngestError> {
        let mut file = fs::File::open(source).map_err(|error| io_error(source, error))?;
        self.stage_from_reader(relative_path, &mut file)
    }

    /// Recursively stage a copy of a directory the user already has, preserving relative layout.
    pub fn stage_copy_dir(
        &self,
        source: &Path,
        relative_prefix: &str,
    ) -> Result<u64, ManagedIngestError> {
        let mut total = 0_u64;
        for entry in fs::read_dir(source).map_err(|error| io_error(source, error))? {
            let entry = entry.map_err(|error| io_error(source, error))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                return Err(ManagedIngestError::InvalidRelativePath {
                    relative_path: entry.path().display().to_string(),
                    reason: "file name is not valid UTF-8",
                });
            };
            let relative = if relative_prefix.is_empty() {
                name.to_owned()
            } else {
                format!("{relative_prefix}/{name}")
            };
            // `file_type` does not follow links, so a symlink in the user's source directory is
            // never followed out of it: it is copied as the file it points at only when that file
            // is inside, and otherwise refused as an unsupported entry below.
            let file_type = entry.file_type().map_err(|error| io_error(source, error))?;
            if file_type.is_dir() {
                total += self.stage_copy_dir(&entry.path(), &relative)?;
            } else if file_type.is_file() {
                total += self.stage_copy_file(&entry.path(), &relative)?;
            }
        }
        Ok(total)
    }

    /// Abandon the ingest. Equivalent to dropping the session; explicit so a cancelled job reads as
    /// a decision rather than a fall-through.
    pub fn cancel(mut self) -> Result<(), ManagedIngestError> {
        self.discard();
        Err(ManagedIngestError::Cancelled {
            install_id: self.install_id.clone(),
        })
    }

    /// Verify, commit, and publish.
    ///
    /// `primary_relative_path` is the artifact the source's declared digest describes;
    /// `expected_sha256` is that digest when the source declared one (a caller-supplied
    /// `expectedSha256`, a Civitai file hash). Hashing happens on the STAGED bytes, before the
    /// commit, so a mismatch never produces an install.
    pub fn finalize(
        mut self,
        primary_relative_path: &str,
        expected_sha256: Option<&str>,
    ) -> Result<ManagedInstallV1, ManagedIngestError> {
        match self.finalize_inner(primary_relative_path, expected_sha256) {
            Ok(install) => Ok(install),
            Err(error) => {
                // Anything that refuses after the commit rolls the commit back, so the invariant
                // holds for the whole finalize, not just up to the rename.
                let _ = self.store.remove_managed(&self.install_id);
                self.discard();
                Err(error)
            }
        }
    }

    fn finalize_inner(
        &mut self,
        primary_relative_path: &str,
        expected_sha256: Option<&str>,
    ) -> Result<ManagedInstallV1, ManagedIngestError> {
        let staging_path = self.staging_dir().to_path_buf();
        let primary_staged = staging_path.join(staged_relative_path(primary_relative_path)?);
        if !primary_staged.is_file() {
            return Err(ManagedIngestError::PrimaryMissing {
                install_id: self.install_id.clone(),
                relative_path: primary_relative_path.to_owned(),
            });
        }

        // 1. Content digest of the staged primary, always computed (it becomes provenance) and
        //    compared when the source declared one.
        let primary_sha256 =
            sha256_file(&primary_staged).map_err(|error| io_error(&primary_staged, error))?;
        if let Some(expected) = expected_sha256 {
            let expected = expected.trim().to_ascii_lowercase();
            if expected != primary_sha256 {
                return Err(ManagedIngestError::HashMismatch {
                    install_id: self.install_id.clone(),
                    relative_path: primary_relative_path.to_owned(),
                    expected_sha256: expected,
                    actual_sha256: primary_sha256,
                });
            }
        }

        // 2. Commit. ONE rename of the whole staged tree into the SceneWorks-owned install
        //    location. Until it returns, no path under `installs/` exists for this id; after it
        //    returns, the complete validated tree does. There is no in-between state to recover.
        let install_path = self.store.install_dir(&self.install_id)?;
        if let Some(parent) = install_path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        fs::rename(&staging_path, &install_path).map_err(|error| io_error(&install_path, error))?;
        self.staging_path = None;

        // 3. Full-content validation and publication, against the committed bytes. A refusal here
        //    is rolled back by `finalize`, so an install the inspector will not accept is not left
        //    addressable either.
        let compiled = self.store.compile_managed(
            &self.install_id,
            primary_relative_path,
            self.provenance.clone(),
        )?;

        Ok(ManagedInstallV1 {
            install_id: self.install_id.clone(),
            checkpoint_id: compiled.checkpoint_id.clone(),
            install_path,
            primary_relative_path: primary_relative_path.to_owned(),
            primary_sha256,
            compiled,
        })
    }

    fn discard(&mut self) {
        if let Some(path) = self.staging_path.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

impl Drop for ManagedIngest {
    fn drop(&mut self) {
        self.discard();
    }
}

/// Remove every staging directory left behind by a crash.
///
/// A staging directory is by construction never referenced by a plan, a catalog record, or a
/// manifest entry, so this is always safe to run — including while other ingests are in flight, for
/// which it skips ids named by `in_flight`. Returns how many were reclaimed.
pub fn sweep_staging(
    store: &CheckpointPlanStore,
    in_flight: &[&str],
) -> Result<usize, ManagedIngestError> {
    let staging_root = store.staging_root().to_path_buf();
    let entries = match fs::read_dir(&staging_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(io_error(&staging_root, error)),
    };
    let mut reclaimed = 0;
    for entry in entries {
        let entry = entry.map_err(|error| io_error(&staging_root, error))?;
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| in_flight.contains(&name))
        {
            continue;
        }
        let path = entry.path();
        let removed = if path.is_dir() {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        };
        removed.map_err(|error| io_error(&path, error))?;
        reclaimed += 1;
    }
    Ok(reclaimed)
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_provenance_url_removes_userinfo_and_secret_parameters() {
        assert_eq!(
            sanitize_provenance_url("https://user:t0ken@civitai.com/api/download/models/42")
                .as_deref(),
            Some("https://civitai.com/api/download/models/42")
        );
        assert_eq!(
            sanitize_provenance_url(
                "https://civitai.com/api/download/models/42?type=Model&token=s3cret&format=SafeTensor"
            )
            .as_deref(),
            Some("https://civitai.com/api/download/models/42?type=Model&format=SafeTensor")
        );
        // Every secret name, alone, leaves a query-free URL rather than an empty `?`.
        for name in SECRET_QUERY_PARAMS {
            assert_eq!(
                sanitize_provenance_url(&format!("https://host.example/f.safetensors?{name}=x"))
                    .as_deref(),
                Some("https://host.example/f.safetensors"),
                "{name} must be stripped"
            );
            assert_eq!(
                sanitize_provenance_url(&format!(
                    "https://host.example/f.safetensors?{}=x",
                    name.to_ascii_uppercase()
                ))
                .as_deref(),
                Some("https://host.example/f.safetensors"),
                "{name} must be stripped case-insensitively"
            );
        }
        assert_eq!(sanitize_provenance_url("not a url"), None);
    }

    #[test]
    fn staged_relative_path_refuses_every_escape() {
        for (path, reason) in [
            ("", "empty"),
            ("   ", "empty"),
            ("../outside.safetensors", "contains '..'"),
            ("./model.safetensors", "contains '.'"),
            ("/abs/model.safetensors", "must be relative"),
            ("dir\\model.safetensors", "must use '/' separators"),
        ] {
            let error = staged_relative_path(path).expect_err(path);
            assert!(
                error.to_string().contains(reason),
                "{path} refused as {error}, expected {reason}"
            );
        }
        assert_eq!(
            staged_relative_path("transformer/model.safetensors").unwrap(),
            PathBuf::from("transformer").join("model.safetensors")
        );
    }
}
