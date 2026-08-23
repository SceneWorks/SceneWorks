//! Persisted checkpoint import plans and the approved-root primitives they resolve against
//! (epic 20398, sc-20634).
//!
//! The store owns `<data_dir>/checkpoints/`:
//!
//! * `approved-roots.json` — [`ApprovedRootsV1`]: the only place a linked library's absolute path
//!   lives. Plans and catalog records carry `rootId + relativePath`, never the path (E6).
//! * `inventory.json` — the [`CheckpointInventoryV1`] of catalog records (plan reference + summary).
//! * `plans/<planId>.json` — the immutable, complete [`ImportPlanV1`] each record references.
//!
//! and, outside that directory, `<data_dir>/models/imports/<installId>/` — the SceneWorks-owned
//! bytes of a MANAGED install (sc-20636, [`MANAGED_INSTALLS_RELATIVE_DIR`]). The only tree this
//! store ever deletes bytes from, and never a place a linked library lives.
//! * `bindings/<planId>.json` — per-layer filesystem stamps captured when the plan compiled, so a
//!   later resolve can skip re-hashing an unchanged multi-gigabyte source and must re-hash (and
//!   refuse on mismatch) as soon as the entry looks different.
//!
//! Compile = inspect (full-content, [`crate::checkpoint_inspector`]) → persist. Resolve = load the
//! record + plan, re-validate the record against the plan, resolve every layer's root, and verify
//! the bytes on disk are still the bytes the plan was compiled from. Every refusal is a typed
//! [`CheckpointPlanError`] raised before any loader is constructed.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::checkpoint_import::{
    CheckpointCatalogRecordV1, CheckpointImportContractError, CheckpointInventoryV1, ImportLayerV1,
    ImportPlanV1, ManagedProvenanceV1, SourceLocatorV1, CHECKPOINT_IMPORT_CONTRACT_VERSION,
};
use crate::checkpoint_inspector::{
    inspect_checkpoint, CheckpointDiagnosticSeverityV1, CheckpointDiagnosticV1,
    CheckpointInspectionRequestV1,
};

/// Directory under the application data dir that holds every file this store owns.
pub const CHECKPOINTS_DIR: &str = "checkpoints";
pub const APPROVED_ROOTS_FILE: &str = "approved-roots.json";
pub const INVENTORY_FILE: &str = "inventory.json";
pub const PLANS_DIR: &str = "plans";
/// The prefix every inspector-emitted `plan_id` carries; see [`validate_plan_id`].
pub const PLAN_ID_PREFIX: &str = "checkpoint-plan-";
pub const BINDINGS_DIR: &str = "bindings";
/// Every finalized managed install lives at `<data_dir>/models/imports/<installId>/` — the tree
/// SceneWorks has always installed imported models into, now addressed by install id.
///
/// It is deliberately NOT under `<data_dir>/checkpoints/`: the plan documents are metadata the
/// store rewrites freely, while these are the model bytes every existing consumer (`paths.model`,
/// the catalog's install-state sweep, the external-library runtime's anchor, the delete route)
/// already resolves. Keeping them in place is what lets managed ownership become transactional
/// without relocating anyone's models.
///
/// The whole subtree is SceneWorks-owned: it is the only tree
/// [`CheckpointPlanStore::remove_managed`] deletes bytes from, its location is derived from the
/// data dir rather than supplied by a caller, and a linked library is never under it (E6).
pub const MANAGED_INSTALLS_RELATIVE_DIR: &str = "models/imports";
/// Staging area for in-flight ingests. A sibling of the install root rather than a child, so a
/// half-written tree is never inside the directory consumers enumerate, and the commit is still a
/// rename within one filesystem.
pub const MANAGED_STAGING_RELATIVE_DIR: &str = "models/.import-staging";

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

/// One approved linked-library root: an opaque stable id and the absolute directory it names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovedRootV1 {
    pub root_id: String,
    pub path: PathBuf,
}

/// The persisted approved-root set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovedRootsV1 {
    pub schema_version: u32,
    pub roots: Vec<ApprovedRootV1>,
}

impl Default for ApprovedRootsV1 {
    fn default() -> Self {
        Self {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            roots: Vec::new(),
        }
    }
}

impl ApprovedRootsV1 {
    pub fn validate(&self) -> Result<(), CheckpointPlanError> {
        if self.schema_version != CHECKPOINT_IMPORT_CONTRACT_VERSION {
            return Err(CheckpointPlanError::Corrupt {
                what: APPROVED_ROOTS_FILE.to_owned(),
                message: format!(
                    "approved-roots schema version {} is unsupported; recompile/rescan required",
                    self.schema_version
                ),
            });
        }
        let mut ids = std::collections::BTreeSet::new();
        for root in &self.roots {
            if root.root_id.trim().is_empty() || !root.path.is_absolute() {
                return Err(CheckpointPlanError::Corrupt {
                    what: APPROVED_ROOTS_FILE.to_owned(),
                    message: format!(
                        "approved root {:?} must have a non-blank id and an absolute path",
                        root.root_id
                    ),
                });
            }
            if !ids.insert(root.root_id.as_str()) {
                return Err(CheckpointPlanError::Corrupt {
                    what: APPROVED_ROOTS_FILE.to_owned(),
                    message: format!("approved root id {:?} is declared twice", root.root_id),
                });
            }
        }
        Ok(())
    }

    pub fn get(&self, root_id: &str) -> Option<&ApprovedRootV1> {
        self.roots.iter().find(|root| root.root_id == root_id)
    }
}

/// Filesystem stamp of one layer's source entry, captured at compile time. Cheap to re-take; any
/// difference forces a full re-hash before the source may be used.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceStampV1 {
    pub size_bytes: u64,
    pub modified_nanos: Option<i64>,
    #[cfg(unix)]
    pub device: u64,
    #[cfg(unix)]
    pub inode: u64,
    #[cfg(unix)]
    pub changed_nanos: i64,
    #[cfg(windows)]
    pub creation_time: u64,
    #[cfg(windows)]
    pub last_write_time: u64,
}

impl SourceStampV1 {
    fn of(path: &Path) -> std::io::Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt as _;
        #[cfg(windows)]
        use std::os::windows::fs::MetadataExt as _;
        let metadata = fs::metadata(path)?;
        Ok(Self {
            size_bytes: metadata.len(),
            modified_nanos: metadata.modified().ok().and_then(|modified| {
                modified
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .and_then(|duration| i64::try_from(duration.as_nanos()).ok())
            }),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            changed_nanos: metadata
                .ctime()
                .saturating_mul(1_000_000_000)
                .saturating_add(metadata.ctime_nsec()),
            #[cfg(windows)]
            creation_time: metadata.creation_time(),
            #[cfg(windows)]
            last_write_time: metadata.last_write_time(),
        })
    }
}

/// The persisted per-plan binding: one stamp per layer id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceBindingsV1 {
    pub schema_version: u32,
    pub plan_id: String,
    pub stamps: BTreeMap<String, SourceStampV1>,
}

/// A freshly compiled and persisted checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledCheckpointV1 {
    pub checkpoint_id: String,
    pub record: CheckpointCatalogRecordV1,
    pub plan: ImportPlanV1,
    /// Every OTHER persisted checkpoint whose plan carries this plan's semantic digest — the same
    /// bytes already known to the store under a different ownership or location (E1/AC2). Reported,
    /// never acted on: neither copy is deleted, neither compile is refused, because a user may
    /// legitimately keep a linked library copy and a managed one.
    pub duplicate_checkpoint_ids: Vec<String>,
}

/// One plan layer resolved to the verified bytes on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedLayerV1 {
    pub layer: ImportLayerV1,
    /// The absolute path a loader opens: `<approved root>/<relativePath>`.
    pub path: PathBuf,
    /// Whether this resolve re-hashed the file (stamp differed) rather than trusting the stamp.
    pub rehashed: bool,
}

/// A persisted checkpoint resolved and verified for loading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCheckpointV1 {
    pub checkpoint_id: String,
    pub record: CheckpointCatalogRecordV1,
    pub plan: ImportPlanV1,
    pub layers: Vec<ResolvedLayerV1>,
}

impl ResolvedCheckpointV1 {
    pub fn family(&self) -> &str {
        &self.plan.family
    }

    /// The resolved layers carrying `role`, in layer-id order.
    pub fn layers_with_role<'a>(
        &'a self,
        role: &'a str,
    ) -> impl Iterator<Item = &'a ResolvedLayerV1> {
        self.layers
            .iter()
            .filter(move |layer| layer.layer.role == role)
    }
}

/// Every way a plan compile or resolve refuses. `code()` is the stable machine id; `Display`
/// carries it plus the actionable detail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointPlanError {
    /// The root id is not in the approved set.
    UnknownRoot {
        root_id: String,
    },
    /// The root is approved but its directory is not there (unmounted drive, moved library).
    RootUnavailable {
        root_id: String,
        path: PathBuf,
    },
    /// The path cannot be approved as a root.
    RootNotApprovable {
        path: PathBuf,
        reason: String,
    },
    /// The relative path is not a confined, portable relative path.
    InvalidRelativePath {
        relative_path: String,
        reason: String,
    },
    /// Full-content inspection did not produce a runnable inventory (unknown family, malformed
    /// container, missing sidecar, …); the diagnostics are the inspector's typed findings.
    UnrunnableSource {
        checkpoint_id: String,
        diagnostics: Vec<CheckpointDiagnosticV1>,
    },
    /// No persisted record for this checkpoint id.
    UnknownCheckpoint {
        checkpoint_id: String,
    },
    /// The record exists but its plan document is missing.
    MissingPlan {
        checkpoint_id: String,
        plan_id: String,
    },
    /// The persisted plan no longer matches its catalog record (edited or swapped on disk).
    PlanTampered {
        checkpoint_id: String,
        reason: String,
    },
    /// A layer's source file is not where the plan says.
    SourceMissing {
        checkpoint_id: String,
        relative_path: String,
        path: PathBuf,
    },
    /// A layer's bytes are not the bytes the plan was compiled from (modified, replaced, or the
    /// root was retargeted at a different library).
    SourceDrifted {
        checkpoint_id: String,
        relative_path: String,
        expected_sha256: String,
        actual_sha256: String,
    },
    /// The plan uses a locator kind this store does not resolve.
    UnsupportedLocator {
        checkpoint_id: String,
        layer_id: String,
        kind: &'static str,
    },
    /// A persisted or caller-supplied `install_id` is not the shape [`validate_install_id`]
    /// accepts, so it may not be used as a directory name. Same choke-point reasoning as
    /// [`Self::InvalidPlanId`]: install ids are read back from a user-writable `inventory.json`
    /// and then address a path.
    InvalidInstallId {
        install_id: String,
        reason: &'static str,
    },
    /// The managed install directory this plan was compiled into is gone (deleted out from under
    /// the store, or the data dir was moved).
    InstallUnavailable {
        install_id: String,
        path: PathBuf,
    },
    /// A managed install id is already taken by a different persisted checkpoint. Finalizing over
    /// it would replace one install's bytes with another's while the first's plan still pointed at
    /// them, so the ingest refuses instead.
    InstallIdTaken {
        install_id: String,
        checkpoint_id: String,
        path: PathBuf,
    },
    /// A layer whose locator kind disagrees with the ownership the operation is compiling. A
    /// managed compile that produced a linked layer (or the reverse) would persist a plan the
    /// resolve path reads against the wrong root.
    LocatorOwnershipMismatch {
        checkpoint_id: String,
        layer_id: String,
        expected: &'static str,
    },
    /// A layer's path is confined-relative but its resolved target lands outside the approved root
    /// (a symlink inside the library pointing elsewhere). The inspector refuses the same shape
    /// (`CheckpointDiagnosticCodeV1::PathEscapesRoot`); the store refuses it again at resolve time
    /// because the link can be planted after the plan was compiled.
    PathEscapesRoot {
        checkpoint_id: String,
        relative_path: String,
        root_path: PathBuf,
        resolved_path: PathBuf,
    },
    /// A persisted `plan_id` is not the shape the inspector emits, so it may not be used as a
    /// filename component. Read back from `inventory.json`, it would otherwise let a crafted
    /// record read or delete a document outside the store.
    InvalidPlanId {
        plan_id: String,
        reason: &'static str,
    },
    /// The approved-root catalog binds this root id to a different directory than the one being
    /// approved: two directories collided on the truncated root id.
    RootIdCollision {
        root_id: String,
        existing_path: PathBuf,
        path: PathBuf,
    },
    /// A contract-level validation failure on a persisted document.
    Contract(CheckpointImportContractError),
    /// A persisted document is unreadable or malformed.
    Corrupt {
        what: String,
        message: String,
    },
    Io {
        path: PathBuf,
        message: String,
    },
}

impl CheckpointPlanError {
    /// Stable kebab-case refusal code for telemetry and callers that branch on the class.
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnknownRoot { .. } => "unknown-root",
            Self::RootUnavailable { .. } => "root-unavailable",
            Self::RootNotApprovable { .. } => "root-not-approvable",
            Self::InvalidRelativePath { .. } => "invalid-relative-path",
            Self::UnrunnableSource { .. } => "unrunnable-source",
            Self::UnknownCheckpoint { .. } => "unknown-checkpoint",
            Self::MissingPlan { .. } => "missing-plan",
            Self::PlanTampered { .. } => "plan-tampered",
            Self::SourceMissing { .. } => "source-missing",
            Self::SourceDrifted { .. } => "source-drifted",
            Self::UnsupportedLocator { .. } => "unsupported-locator",
            Self::InvalidInstallId { .. } => "invalid-install-id",
            Self::InstallUnavailable { .. } => "install-unavailable",
            Self::InstallIdTaken { .. } => "install-id-taken",
            Self::LocatorOwnershipMismatch { .. } => "locator-ownership-mismatch",
            Self::PathEscapesRoot { .. } => "path-escapes-root",
            Self::InvalidPlanId { .. } => "invalid-plan-id",
            Self::RootIdCollision { .. } => "root-id-collision",
            Self::Contract(_) => "contract",
            Self::Corrupt { .. } => "corrupt",
            Self::Io { .. } => "io",
        }
    }
}

impl fmt::Display for CheckpointPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[checkpoint-plan:{}] ", self.code())?;
        match self {
            Self::UnknownRoot { root_id } => write!(f, "root {root_id:?} is not an approved root"),
            Self::RootUnavailable { root_id, path } => write!(
                f,
                "approved root {root_id:?} at {} is not available (unmounted or moved)",
                path.display()
            ),
            Self::RootNotApprovable { path, reason } => {
                write!(f, "{} cannot be approved as a root: {reason}", path.display())
            }
            Self::InvalidRelativePath {
                relative_path,
                reason,
            } => write!(f, "relative path {relative_path:?} is invalid: {reason}"),
            Self::UnrunnableSource {
                checkpoint_id,
                diagnostics,
            } => {
                write!(
                    f,
                    "checkpoint {checkpoint_id:?} is not runnable: {} diagnostic(s)",
                    diagnostics.len()
                )?;
                for diagnostic in diagnostics {
                    write!(
                        f,
                        "; {:?} {}{}",
                        diagnostic.code,
                        diagnostic
                            .relative_path
                            .as_deref()
                            .map(|path| format!("{path}: "))
                            .unwrap_or_default(),
                        diagnostic.message
                    )?;
                }
                Ok(())
            }
            Self::UnknownCheckpoint { checkpoint_id } => {
                write!(f, "no persisted plan for checkpoint {checkpoint_id:?}")
            }
            Self::MissingPlan {
                checkpoint_id,
                plan_id,
            } => write!(
                f,
                "checkpoint {checkpoint_id:?} references plan {plan_id:?} which is not persisted"
            ),
            Self::PlanTampered {
                checkpoint_id,
                reason,
            } => write!(
                f,
                "persisted plan for checkpoint {checkpoint_id:?} does not match its record: {reason}"
            ),
            Self::SourceMissing {
                checkpoint_id,
                relative_path,
                path,
            } => write!(
                f,
                "checkpoint {checkpoint_id:?} layer {relative_path:?} is missing at {}",
                path.display()
            ),
            Self::SourceDrifted {
                checkpoint_id,
                relative_path,
                expected_sha256,
                actual_sha256,
            } => write!(
                f,
                "checkpoint {checkpoint_id:?} layer {relative_path:?} changed since its plan was compiled (expected sha256 {expected_sha256}, found {actual_sha256}); rescan required"
            ),
            Self::UnsupportedLocator {
                checkpoint_id,
                layer_id,
                kind,
            } => write!(
                f,
                "checkpoint {checkpoint_id:?} layer {layer_id:?} uses a {kind} locator this store does not resolve"
            ),
            Self::InvalidInstallId { install_id, reason } => write!(
                f,
                "managed install id {install_id:?} is not a usable directory name: {reason}"
            ),
            Self::InstallUnavailable { install_id, path } => write!(
                f,
                "managed install {install_id:?} is not present at {}; its SceneWorks-owned copy was removed outside the app",
                path.display()
            ),
            Self::InstallIdTaken {
                install_id,
                checkpoint_id,
                path,
            } => write!(
                f,
                "managed install id {install_id:?} is already in use: {} exists (checkpoint {checkpoint_id:?}). Remove that install, or import under a different id.",
                path.display()
            ),
            Self::LocatorOwnershipMismatch {
                checkpoint_id,
                layer_id,
                expected,
            } => write!(
                f,
                "checkpoint {checkpoint_id:?} layer {layer_id:?} compiled to a locator that is not {expected}"
            ),
            Self::PathEscapesRoot {
                checkpoint_id,
                relative_path,
                root_path,
                resolved_path,
            } => write!(
                f,
                "checkpoint {checkpoint_id:?} layer {relative_path:?} resolves to {}, which is outside its approved root {}",
                resolved_path.display(),
                root_path.display()
            ),
            Self::InvalidPlanId { plan_id, reason } => write!(
                f,
                "plan id {plan_id:?} is not a usable document name: {reason}"
            ),
            Self::RootIdCollision {
                root_id,
                existing_path,
                path,
            } => write!(
                f,
                "root id {root_id:?} is already bound to {}, not {}",
                existing_path.display(),
                path.display()
            ),
            Self::Contract(error) => write!(f, "{error}"),
            Self::Corrupt { what, message } => write!(f, "{what} is corrupt: {message}"),
            Self::Io { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for CheckpointPlanError {}

impl From<CheckpointImportContractError> for CheckpointPlanError {
    fn from(error: CheckpointImportContractError) -> Self {
        Self::Contract(error)
    }
}

fn io_error(path: &Path, error: std::io::Error) -> CheckpointPlanError {
    CheckpointPlanError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

/// Checkpoint identity for a linked source: root id + relative path, nothing physical.
pub fn linked_checkpoint_id(root_id: &str, relative_path: &str) -> String {
    format!("linked/{root_id}/{relative_path}")
}

/// Checkpoint identity for a managed source: the install id, nothing physical. A managed install
/// holds exactly one checkpoint, so unlike [`linked_checkpoint_id`] the relative path is not part
/// of the identity — the install id already names it.
pub fn managed_checkpoint_id(install_id: &str) -> String {
    format!("managed/{install_id}")
}

/// The install id inside a `managed/<installId>` checkpoint id, if it is one.
pub fn managed_install_id(checkpoint_id: &str) -> Option<&str> {
    checkpoint_id
        .strip_prefix("managed/")
        .filter(|install_id| validate_install_id(install_id).is_ok())
}

/// The shape every managed install id must have before it can address a directory.
///
/// An install id is chosen by whoever opens the ingest — for a model import, the sanitized model id
/// that already names the install directory — and is read back out of a user-writable
/// `inventory.json` before being joined onto the install root. So this is the choke point: every
/// read, write, and delete of an install directory goes through
/// [`CheckpointPlanStore::install_dir`], which validates first.
pub fn validate_install_id(install_id: &str) -> Result<(), CheckpointPlanError> {
    let invalid = |reason: &'static str| CheckpointPlanError::InvalidInstallId {
        install_id: install_id.to_owned(),
        reason,
    };
    if install_id.is_empty() || install_id.len() > 128 {
        return Err(invalid("must be 1..=128 characters"));
    }
    // The accepted set is exactly what `model_artifacts::safe_download_dir` emits, so a model id
    // that already names an install directory today keeps naming it.
    if !install_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(invalid("must be ASCII alphanumeric, '_', '-', or '.'"));
    }
    // `.` is allowed inside a name (`model.v2`) but never as the leading character and never
    // doubled: that is what keeps `.`, `..`, and hidden directories unreachable.
    if install_id.starts_with('.') || install_id.starts_with('-') {
        return Err(invalid("must not start with '.' or '-'"));
    }
    if install_id.contains("..") {
        return Err(invalid("must not contain '..'"));
    }
    Ok(())
}

/// Deterministic opaque id for an approved root. Derived from the canonical path's digest so
/// re-approving the same directory is idempotent; the path itself never appears in the id, and a
/// later relink keeps the id while rebinding the path.
pub fn derive_root_id(canonical_path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sceneworks.checkpoint.approved-root.v1\0");
    hasher.update(canonical_path.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    format!("root-{:x}", digest)[..21].to_owned()
}

fn validate_portable_relative_path(relative_path: &str) -> Result<PathBuf, CheckpointPlanError> {
    let invalid = |reason: &str| CheckpointPlanError::InvalidRelativePath {
        relative_path: relative_path.to_owned(),
        reason: reason.to_owned(),
    };
    if relative_path.trim().is_empty() {
        return Err(invalid("empty"));
    }
    if relative_path.contains('\\') {
        return Err(invalid("must use '/' separators"));
    }
    let path = Path::new(relative_path);
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => return Err(invalid("contains '.'")),
            Component::ParentDir => return Err(invalid("contains '..'")),
            Component::RootDir | Component::Prefix(_) => return Err(invalid("must be relative")),
        }
    }
    if out.as_os_str().is_empty() {
        return Err(invalid("empty"));
    }
    Ok(out)
}

/// The `plan_id` shape the inspector emits (`checkpoint_inspector.rs`, `checkpoint-plan-{sha256:x}`).
/// Every persisted plan id becomes a filename component, and it is read back from a user-writable
/// `inventory.json`, so it is validated before it can address a path.
fn validate_plan_id(plan_id: &str) -> Result<(), CheckpointPlanError> {
    let invalid = |reason: &'static str| CheckpointPlanError::InvalidPlanId {
        plan_id: plan_id.to_owned(),
        reason,
    };
    let Some(digest) = plan_id.strip_prefix(PLAN_ID_PREFIX) else {
        return Err(invalid("must start with 'checkpoint-plan-'"));
    };
    if digest.is_empty() || digest.len() > 64 {
        return Err(invalid("digest must be 1..=64 characters"));
    }
    if !digest
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid("digest must be lowercase hex"));
    }
    Ok(())
}

/// Join a validated relative path under an approved root and confine the result to that root.
///
/// `validate_portable_relative_path` only rejects lexical escapes (`..`, absolute, `\`); a symlink
/// *inside* the library pointing outside it is still a lexically clean relative path. So the joined
/// path is canonicalized and required to stay under the canonical root — the same confinement the
/// inspector applies at discovery time (`checkpoint_inspector.rs`), re-applied at resolve time
/// because a link can be planted after the plan was compiled.
///
/// A path that does not exist is returned as the plain join: existence is the caller's refusal to
/// type (`SourceMissing`), not this function's.
fn confined_root_join(
    checkpoint_id: &str,
    root_path: &Path,
    relative_path: &str,
) -> Result<PathBuf, CheckpointPlanError> {
    let joined = root_path.join(validate_portable_relative_path(relative_path)?);
    let canonical_target = match fs::canonicalize(&joined) {
        Ok(canonical) => canonical,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(joined),
        Err(error) => return Err(io_error(&joined, error)),
    };
    let canonical_root = fs::canonicalize(root_path).map_err(|error| io_error(root_path, error))?;
    if !canonical_target.starts_with(&canonical_root) {
        return Err(CheckpointPlanError::PathEscapesRoot {
            checkpoint_id: checkpoint_id.to_owned(),
            relative_path: relative_path.to_owned(),
            root_path: canonical_root,
            resolved_path: canonical_target,
        });
    }
    Ok(canonical_target)
}

/// The locator kind as the store names it in refusals: the discriminator, never the payload.
fn locator_kind(locator: &SourceLocatorV1) -> &'static str {
    match locator {
        SourceLocatorV1::Linked { .. } => "linked",
        SourceLocatorV1::Managed { .. } => "managed",
    }
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// The store handle. Cheap to construct; every operation reads the current on-disk state.
#[derive(Clone, Debug)]
pub struct CheckpointPlanStore {
    root: PathBuf,
    /// Both derived from the data dir in [`Self::open`], never supplied by a caller: the tree
    /// `remove_managed` may delete from cannot be steered by any argument.
    installs_root: PathBuf,
    staging_root: PathBuf,
}

impl CheckpointPlanStore {
    pub fn open(data_dir: &Path) -> Self {
        Self {
            root: data_dir.join(CHECKPOINTS_DIR),
            installs_root: data_dir.join(MANAGED_INSTALLS_RELATIVE_DIR),
            staging_root: data_dir.join(MANAGED_STAGING_RELATIVE_DIR),
        }
    }

    /// The document root: `<data_dir>/checkpoints`.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The managed-install byte root: `<data_dir>/models/imports`.
    pub fn installs_root(&self) -> &Path {
        &self.installs_root
    }

    /// The staging root in-flight ingests write into: `<data_dir>/models/.import-staging`.
    pub fn staging_root(&self) -> &Path {
        &self.staging_root
    }

    fn approved_roots_path(&self) -> PathBuf {
        self.root.join(APPROVED_ROOTS_FILE)
    }

    fn inventory_path(&self) -> PathBuf {
        self.root.join(INVENTORY_FILE)
    }

    /// The plan document path for `plan_id`. Validating here rather than at the call sites makes
    /// this the choke point: no read, write, or delete can address a document without the id
    /// having passed [`validate_plan_id`] first.
    fn plan_path(&self, plan_id: &str) -> Result<PathBuf, CheckpointPlanError> {
        validate_plan_id(plan_id)?;
        Ok(self.root.join(PLANS_DIR).join(format!("{plan_id}.json")))
    }

    fn bindings_path(&self, plan_id: &str) -> Result<PathBuf, CheckpointPlanError> {
        validate_plan_id(plan_id)?;
        Ok(self.root.join(BINDINGS_DIR).join(format!("{plan_id}.json")))
    }

    fn write_atomic(&self, path: &Path, payload: &[u8]) -> Result<(), CheckpointPlanError> {
        crate::store_util::atomic_write(path, payload).map_err(|error| CheckpointPlanError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })
    }

    fn read_json<T: for<'de> Deserialize<'de>>(
        &self,
        path: &Path,
        what: &str,
    ) -> Result<Option<T>, CheckpointPlanError> {
        let payload = match fs::read(path) {
            Ok(payload) => payload,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error(path, error)),
        };
        serde_json::from_slice(&payload)
            .map(Some)
            .map_err(|error| CheckpointPlanError::Corrupt {
                what: what.to_owned(),
                message: error.to_string(),
            })
    }

    // ---- approved roots -------------------------------------------------------------------

    pub fn approved_roots(&self) -> Result<ApprovedRootsV1, CheckpointPlanError> {
        let roots: ApprovedRootsV1 = self
            .read_json(&self.approved_roots_path(), APPROVED_ROOTS_FILE)?
            .unwrap_or_default();
        roots.validate()?;
        Ok(roots)
    }

    fn save_approved_roots(&self, roots: &ApprovedRootsV1) -> Result<(), CheckpointPlanError> {
        roots.validate()?;
        let mut payload =
            serde_json::to_vec_pretty(roots).map_err(|error| CheckpointPlanError::Corrupt {
                what: APPROVED_ROOTS_FILE.to_owned(),
                message: error.to_string(),
            })?;
        payload.push(b'\n');
        self.write_atomic(&self.approved_roots_path(), &payload)
    }

    /// Approve an existing directory as a linked-library root. Idempotent for the same canonical
    /// directory; the returned root carries the id plans will reference.
    pub fn approve_root(&self, path: &Path) -> Result<ApprovedRootV1, CheckpointPlanError> {
        if !path.is_absolute() {
            return Err(CheckpointPlanError::RootNotApprovable {
                path: path.to_path_buf(),
                reason: "root path must be absolute".to_owned(),
            });
        }
        let canonical =
            fs::canonicalize(path).map_err(|error| CheckpointPlanError::RootNotApprovable {
                path: path.to_path_buf(),
                reason: format!("cannot canonicalize: {error}"),
            })?;
        if !canonical.is_dir() {
            return Err(CheckpointPlanError::RootNotApprovable {
                path: path.to_path_buf(),
                reason: "root must be an existing directory".to_owned(),
            });
        }
        let mut roots = self.approved_roots()?;
        let root_id = derive_root_id(&canonical);
        if let Some(existing) = roots.get(&root_id) {
            // Idempotent only for the SAME directory. `derive_root_id` truncates the digest, so a
            // matching id is not by itself proof of a matching path — without this check, approving
            // directory B could hand back directory A's binding and every plan compiled against it
            // would silently read A's files.
            if existing.path != canonical {
                return Err(CheckpointPlanError::RootIdCollision {
                    root_id,
                    existing_path: existing.path.clone(),
                    path: canonical,
                });
            }
            return Ok(existing.clone());
        }
        let root = ApprovedRootV1 {
            root_id,
            path: canonical,
        };
        roots.roots.push(root.clone());
        self.save_approved_roots(&roots)?;
        Ok(root)
    }

    // ---- managed installs -----------------------------------------------------------------

    /// The SceneWorks-owned directory a managed install occupies. Validates the id before it can
    /// address a path, so this is the only way any caller — including the ingest and
    /// [`Self::remove_managed`] — names an install directory.
    pub fn install_dir(&self, install_id: &str) -> Result<PathBuf, CheckpointPlanError> {
        validate_install_id(install_id)?;
        Ok(self.installs_root.join(install_id))
    }

    /// The install directory, required to exist. `InstallUnavailable` rather than a bare io error
    /// so a caller can tell "the user deleted our copy" apart from a broken data dir.
    pub fn resolve_install(&self, install_id: &str) -> Result<PathBuf, CheckpointPlanError> {
        let path = self.install_dir(install_id)?;
        if !path.is_dir() {
            return Err(CheckpointPlanError::InstallUnavailable {
                install_id: install_id.to_owned(),
                path,
            });
        }
        Ok(path)
    }

    /// Remove a managed install completely: its catalog record, plan and bindings documents, and
    /// the SceneWorks-owned bytes under `<data_dir>/models/imports/<installId>`.
    ///
    /// This is the ONLY method in this store that deletes checkpoint bytes, and it can only ever
    /// address a path under [`Self::installs_root`] (E6), which is derived from the data dir rather
    /// than supplied by anyone. A linked library is never under that tree, so no argument —
    /// including a crafted `inventory.json` — can steer it at a user's files;
    /// [`validate_install_id`] additionally refuses anything that is not a single non-traversing
    /// path component. Returns `false` when there was nothing to remove.
    pub fn remove_managed(&self, install_id: &str) -> Result<bool, CheckpointPlanError> {
        let install_path = self.install_dir(install_id)?;
        let removed_record = self.invalidate(&managed_checkpoint_id(install_id))?;
        let removed_bytes = match fs::remove_dir_all(&install_path) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(io_error(&install_path, error)),
        };
        Ok(removed_record || removed_bytes)
    }

    /// The live directory for an approved root id.
    pub fn resolve_root(&self, root_id: &str) -> Result<PathBuf, CheckpointPlanError> {
        let roots = self.approved_roots()?;
        let root = roots
            .get(root_id)
            .ok_or_else(|| CheckpointPlanError::UnknownRoot {
                root_id: root_id.to_owned(),
            })?;
        if !root.path.is_dir() {
            return Err(CheckpointPlanError::RootUnavailable {
                root_id: root_id.to_owned(),
                path: root.path.clone(),
            });
        }
        Ok(root.path.clone())
    }

    // ---- compile --------------------------------------------------------------------------

    /// Inspect `<root>/<relative_path>` as a linked checkpoint and persist its plan, catalog record,
    /// and source bindings. Identity is `rootId + relativePath`; an unrunnable inspection refuses
    /// with the inspector's typed diagnostics and persists nothing.
    pub fn compile_linked(
        &self,
        root_id: &str,
        relative_path: &str,
    ) -> Result<CompiledCheckpointV1, CheckpointPlanError> {
        let root_path = self.resolve_root(root_id)?;
        let relative = validate_portable_relative_path(relative_path)?;
        let checkpoint_id = linked_checkpoint_id(root_id, relative_path);
        let request = CheckpointInspectionRequestV1::linked(
            checkpoint_id.clone(),
            root_path.clone(),
            relative.clone(),
            root_id,
        )
        .map_err(|reason| CheckpointPlanError::InvalidRelativePath {
            relative_path: relative_path.to_owned(),
            reason,
        })?;
        self.compile(&request, &root_path, "linked")
    }

    /// Inspect `<installs>/<install_id>/<relative_path>` as an application-owned checkpoint and
    /// persist its plan, catalog record, and source bindings — the managed counterpart of
    /// [`Self::compile_linked`], into the same store, using the same inspector and the same
    /// documents (sc-20636).
    ///
    /// Identity is the install id alone, and the compiled layers carry
    /// `SourceLocatorV1::Managed { install_id, relative_path, sha256, provenance }`, so the
    /// resulting plan's SEMANTIC digest is byte-for-byte the digest the same checkpoint compiles to
    /// under a linked root (E1) while its source-binding identity is not.
    ///
    /// This does not copy, download, or validate transactionally — it compiles what is already
    /// finalized under the install directory. [`crate::checkpoint_ingest`] is what gets bytes there
    /// atomically; calling this directly on a half-written directory is exactly the partial install
    /// the ingest exists to prevent.
    pub fn compile_managed(
        &self,
        install_id: &str,
        relative_path: &str,
        provenance: ManagedProvenanceV1,
    ) -> Result<CompiledCheckpointV1, CheckpointPlanError> {
        let install_path = self.resolve_install(install_id)?;
        let relative = validate_portable_relative_path(relative_path)?;
        let checkpoint_id = managed_checkpoint_id(install_id);
        let request = CheckpointInspectionRequestV1::managed(
            checkpoint_id.clone(),
            install_path.clone(),
            relative.clone(),
            install_id,
            provenance,
        )
        .map_err(|reason| CheckpointPlanError::InvalidRelativePath {
            relative_path: relative_path.to_owned(),
            reason,
        })?;
        self.compile(&request, &install_path, "managed")
    }

    /// The ownership-independent half of a compile: inspect, take the runnable record and plan,
    /// stamp every layer, and publish. `root_path` is the directory the plan's relative paths are
    /// joined onto — an approved library root for linked, the install directory for managed — and
    /// `expected_kind` is the locator kind every compiled layer must carry, so a request that
    /// produced the wrong ownership refuses instead of persisting a plan the resolve path would
    /// read against the wrong root.
    fn compile(
        &self,
        request: &CheckpointInspectionRequestV1,
        root_path: &Path,
        expected_kind: &'static str,
    ) -> Result<CompiledCheckpointV1, CheckpointPlanError> {
        let checkpoint_id = request.checkpoint_id.clone();
        let inspection = inspect_checkpoint(request);
        let errors: Vec<CheckpointDiagnosticV1> = inspection
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == CheckpointDiagnosticSeverityV1::Error)
            .cloned()
            .collect();
        if !inspection.is_runnable() {
            return Err(CheckpointPlanError::UnrunnableSource {
                checkpoint_id,
                diagnostics: if errors.is_empty() {
                    inspection.diagnostics.clone()
                } else {
                    errors
                },
            });
        }
        let record = inspection
            .inventory
            .records
            .iter()
            .find(|record| record.checkpoint_id == checkpoint_id)
            .cloned()
            .ok_or_else(|| CheckpointPlanError::Corrupt {
                what: "inspection inventory".to_owned(),
                message: format!("runnable inspection carries no record for {checkpoint_id:?}"),
            })?;
        let plan = inspection
            .plans
            .iter()
            .find(|plan| plan.plan_id == record.plan.plan_id)
            .cloned()
            .ok_or_else(|| CheckpointPlanError::Corrupt {
                what: "inspection plans".to_owned(),
                message: format!(
                    "runnable inspection carries no plan {:?}",
                    record.plan.plan_id
                ),
            })?;
        record.validate_loaded_plan(&plan)?;

        // Stamps first: they describe the bytes the inspector just hashed.
        let mut stamps = BTreeMap::new();
        for layer in &plan.layers {
            if locator_kind(&layer.source) != expected_kind {
                return Err(CheckpointPlanError::LocatorOwnershipMismatch {
                    checkpoint_id,
                    layer_id: layer.layer_id.clone(),
                    expected: expected_kind,
                });
            }
            let path = self.layer_path(&checkpoint_id, root_path, layer)?;
            let stamp = SourceStampV1::of(&path).map_err(|error| io_error(&path, error))?;
            stamps.insert(layer.layer_id.clone(), stamp);
        }
        let bindings = SourceBindingsV1 {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            plan_id: plan.plan_id.clone(),
            stamps,
        };
        // Reported against the inventory as it stands BEFORE this record is published, so a
        // recompile of the same checkpoint never reports itself as its own duplicate.
        let duplicate_checkpoint_ids =
            self.duplicates_of(&record.summary.semantic_digest, &checkpoint_id)?;

        self.persist_plan(&plan)?;
        self.persist_bindings(&bindings)?;
        self.upsert_record(record.clone())?;
        Ok(CompiledCheckpointV1 {
            checkpoint_id,
            record,
            plan,
            duplicate_checkpoint_ids,
        })
    }

    /// Every persisted checkpoint other than `exclude_checkpoint_id` whose plan carries
    /// `semantic_digest`: the same bytes already known to the store under different ownership or in
    /// a different place. In checkpoint-id order, so the report is stable.
    pub fn duplicates_of(
        &self,
        semantic_digest: &str,
        exclude_checkpoint_id: &str,
    ) -> Result<Vec<String>, CheckpointPlanError> {
        Ok(self
            .inventory()?
            .records
            .into_iter()
            .filter(|record| {
                record.checkpoint_id != exclude_checkpoint_id
                    && record.summary.semantic_digest == semantic_digest
            })
            .map(|record| record.checkpoint_id)
            .collect())
    }

    fn layer_path(
        &self,
        checkpoint_id: &str,
        root_path: &Path,
        layer: &ImportLayerV1,
    ) -> Result<PathBuf, CheckpointPlanError> {
        let relative = match &layer.source {
            SourceLocatorV1::Linked { relative_path, .. }
            | SourceLocatorV1::Managed { relative_path, .. } => relative_path,
        };
        confined_root_join(checkpoint_id, root_path, relative)
    }

    fn persist_plan(&self, plan: &ImportPlanV1) -> Result<(), CheckpointPlanError> {
        let mut payload = plan.canonical_json()?.into_bytes();
        payload.push(b'\n');
        self.write_atomic(&self.plan_path(&plan.plan_id)?, &payload)
    }

    fn persist_bindings(&self, bindings: &SourceBindingsV1) -> Result<(), CheckpointPlanError> {
        let mut payload =
            serde_json::to_vec_pretty(bindings).map_err(|error| CheckpointPlanError::Corrupt {
                what: "source bindings".to_owned(),
                message: error.to_string(),
            })?;
        payload.push(b'\n');
        self.write_atomic(&self.bindings_path(&bindings.plan_id)?, &payload)
    }

    pub fn inventory(&self) -> Result<CheckpointInventoryV1, CheckpointPlanError> {
        match self.read_json::<CheckpointInventoryV1>(&self.inventory_path(), INVENTORY_FILE)? {
            Some(inventory) => Ok(inventory),
            None => Ok(CheckpointInventoryV1::new(Vec::new())?),
        }
    }

    fn save_inventory(&self, inventory: &CheckpointInventoryV1) -> Result<(), CheckpointPlanError> {
        let mut payload = inventory.canonical_json()?.into_bytes();
        payload.push(b'\n');
        self.write_atomic(&self.inventory_path(), &payload)
    }

    fn upsert_record(&self, record: CheckpointCatalogRecordV1) -> Result<(), CheckpointPlanError> {
        let inventory = self.inventory()?;
        let mut records = Vec::with_capacity(inventory.records.len() + 1);
        let mut superseded_plan = None;
        for existing in inventory.records {
            if existing.checkpoint_id == record.checkpoint_id {
                if existing.plan.plan_id != record.plan.plan_id {
                    superseded_plan = Some(existing.plan.plan_id.clone());
                }
            } else {
                records.push(existing);
            }
        }
        records.push(record);
        self.save_inventory(&CheckpointInventoryV1::new(records)?)?;
        // A re-compile of the same checkpoint replaces its previous plan; drop the orphaned
        // documents so the store never accumulates plans nothing references.
        if let Some(plan_id) = superseded_plan {
            self.remove_plan_documents(&plan_id)?;
        }
        Ok(())
    }

    fn remove_plan_documents(&self, plan_id: &str) -> Result<(), CheckpointPlanError> {
        for path in [self.plan_path(plan_id)?, self.bindings_path(plan_id)?] {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io_error(&path, error)),
            }
        }
        Ok(())
    }

    /// Remove a checkpoint's record, plan, and bindings. Never touches the linked source.
    pub fn invalidate(&self, checkpoint_id: &str) -> Result<bool, CheckpointPlanError> {
        let inventory = self.inventory()?;
        let Some(record) = inventory
            .records
            .iter()
            .find(|record| record.checkpoint_id == checkpoint_id)
            .cloned()
        else {
            return Ok(false);
        };
        let remaining: Vec<CheckpointCatalogRecordV1> = inventory
            .records
            .into_iter()
            .filter(|existing| existing.checkpoint_id != checkpoint_id)
            .collect();
        self.save_inventory(&CheckpointInventoryV1::new(remaining)?)?;
        self.remove_plan_documents(&record.plan.plan_id)?;
        Ok(true)
    }

    // ---- resolve --------------------------------------------------------------------------

    pub fn record(
        &self,
        checkpoint_id: &str,
    ) -> Result<CheckpointCatalogRecordV1, CheckpointPlanError> {
        self.inventory()?
            .records
            .into_iter()
            .find(|record| record.checkpoint_id == checkpoint_id)
            .ok_or_else(|| CheckpointPlanError::UnknownCheckpoint {
                checkpoint_id: checkpoint_id.to_owned(),
            })
    }

    pub fn plan(
        &self,
        checkpoint_id: &str,
        plan_id: &str,
    ) -> Result<ImportPlanV1, CheckpointPlanError> {
        self.read_json::<ImportPlanV1>(&self.plan_path(plan_id)?, "import plan")?
            .ok_or_else(|| CheckpointPlanError::MissingPlan {
                checkpoint_id: checkpoint_id.to_owned(),
                plan_id: plan_id.to_owned(),
            })
    }

    /// Load and verify a persisted checkpoint for loading: record ↔ plan agreement, every root
    /// approved and present, every layer present and byte-identical to the compiled plan.
    pub fn resolve(
        &self,
        checkpoint_id: &str,
    ) -> Result<ResolvedCheckpointV1, CheckpointPlanError> {
        let record = self.record(checkpoint_id)?;
        let plan = self.plan(checkpoint_id, &record.plan.plan_id)?;
        record
            .validate_loaded_plan(&plan)
            .map_err(|error| CheckpointPlanError::PlanTampered {
                checkpoint_id: checkpoint_id.to_owned(),
                reason: error.to_string(),
            })?;
        let mut bindings = self
            .read_json::<SourceBindingsV1>(&self.bindings_path(&plan.plan_id)?, "source bindings")?
            .unwrap_or_else(|| SourceBindingsV1 {
                schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
                plan_id: plan.plan_id.clone(),
                stamps: BTreeMap::new(),
            });
        if bindings.plan_id != plan.plan_id {
            return Err(CheckpointPlanError::PlanTampered {
                checkpoint_id: checkpoint_id.to_owned(),
                reason: format!(
                    "source bindings belong to plan {:?}, not {:?}",
                    bindings.plan_id, plan.plan_id
                ),
            });
        }

        let mut layers = Vec::with_capacity(plan.layers.len());
        let mut refreshed = false;
        for layer in &plan.layers {
            // The root a layer's relative path is joined onto is chosen by the LOCATOR, never by
            // the caller: a linked layer resolves under its approved library root, a managed layer
            // under its own SceneWorks-owned install directory. A managed layer naming an install
            // other than the checkpoint's own would resolve one install's plan against another
            // install's bytes, so it refuses rather than reading them.
            let (root_path, relative_path, fingerprint) = match &layer.source {
                SourceLocatorV1::Linked {
                    root_id,
                    relative_path,
                    fingerprint,
                    ..
                } => (self.resolve_root(root_id)?, relative_path, fingerprint),
                SourceLocatorV1::Managed {
                    install_id,
                    relative_path,
                    sha256,
                    ..
                } => {
                    if managed_install_id(checkpoint_id) != Some(install_id.as_str()) {
                        return Err(CheckpointPlanError::UnsupportedLocator {
                            checkpoint_id: checkpoint_id.to_owned(),
                            layer_id: layer.layer_id.clone(),
                            kind: "foreign-install managed",
                        });
                    }
                    (self.resolve_install(install_id)?, relative_path, sha256)
                }
            };
            let path = confined_root_join(checkpoint_id, &root_path, relative_path)?;
            let current = match SourceStampV1::of(&path) {
                Ok(stamp) => stamp,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Err(CheckpointPlanError::SourceMissing {
                        checkpoint_id: checkpoint_id.to_owned(),
                        relative_path: relative_path.clone(),
                        path,
                    })
                }
                Err(error) => return Err(io_error(&path, error)),
            };
            if !path.is_file() {
                return Err(CheckpointPlanError::SourceMissing {
                    checkpoint_id: checkpoint_id.to_owned(),
                    relative_path: relative_path.clone(),
                    path,
                });
            }
            let stamp_matches = bindings.stamps.get(&layer.layer_id) == Some(&current);
            let rehashed = !stamp_matches;
            if rehashed {
                let actual = sha256_file(&path).map_err(|error| io_error(&path, error))?;
                if &actual != fingerprint {
                    return Err(CheckpointPlanError::SourceDrifted {
                        checkpoint_id: checkpoint_id.to_owned(),
                        relative_path: relative_path.clone(),
                        expected_sha256: fingerprint.clone(),
                        actual_sha256: actual,
                    });
                }
                // Same bytes, new entry (touched, re-copied, relinked): refresh the stamp so the
                // next resolve is cheap again.
                let after = SourceStampV1::of(&path).map_err(|error| io_error(&path, error))?;
                bindings.stamps.insert(layer.layer_id.clone(), after);
                refreshed = true;
            }
            layers.push(ResolvedLayerV1 {
                layer: layer.clone(),
                path,
                rehashed,
            });
        }
        if refreshed {
            self.persist_bindings(&bindings)?;
        }
        Ok(ResolvedCheckpointV1 {
            checkpoint_id: checkpoint_id.to_owned(),
            record,
            plan,
            layers,
        })
    }
}
