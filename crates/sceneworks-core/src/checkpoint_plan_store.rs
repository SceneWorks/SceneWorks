//! Persisted checkpoint import plans and the approved-root primitives they resolve against
//! (epic 20398, sc-20634).
//!
//! The store owns `<data_dir>/checkpoints/`:
//!
//! * `approved-roots.json` — [`ApprovedRootsV1`]: the only place a linked library's absolute path
//!   lives. Plans and catalog records carry `rootId + relativePath`, never the path (E6).
//! * `inventory.json` — the [`CheckpointInventoryV1`] of catalog records (plan reference + summary).
//! * `plans/<planId>.json` — the immutable, complete [`ImportPlanV1`] each record references.
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
    ImportPlanV1, SourceLocatorV1, CHECKPOINT_IMPORT_CONTRACT_VERSION,
};
use crate::checkpoint_inspector::{
    discover_library_root, inspect_checkpoint, CheckpointCandidateV1, CheckpointDiagnosticCodeV1,
    CheckpointDiagnosticSeverityV1, CheckpointDiagnosticV1, CheckpointInspectionRequestV1,
};

/// Directory under the application data dir that holds every file this store owns.
pub const CHECKPOINTS_DIR: &str = "checkpoints";
pub const APPROVED_ROOTS_FILE: &str = "approved-roots.json";
pub const INVENTORY_FILE: &str = "inventory.json";
pub const PLANS_DIR: &str = "plans";
/// The prefix every inspector-emitted `plan_id` carries; see [`validate_plan_id`].
pub const PLAN_ID_PREFIX: &str = "checkpoint-plan-";
pub const BINDINGS_DIR: &str = "bindings";
/// The advisory file every read-modify-write of the store's SHARED documents serialises on.
///
/// `approved-roots.json` and `inventory.json` are each written atomically, but atomic write is not
/// atomic read-modify-write: the API process (root lifecycle) and the worker process (compile,
/// invalidate) both load a whole document, edit it, and write it back, so without a cross-process
/// lock the later writer silently discards whatever the other one committed in between. Same
/// pattern and same primitive as `resolved_cache::lock_metadata`.
pub const STORE_LOCK_FILE: &str = ".lock";

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

/// The longest user-chosen library label the store persists.
pub const MAX_ROOT_LABEL_BYTES: usize = 128;

/// One approved linked-library root: an opaque stable id, the absolute directory it names, and the
/// user's display label for it.
///
/// `label` is presentation only — nothing keys off it — and it is absent from roots written before
/// sc-20635, so it defaults to empty and [`Self::display_label`] falls back to the directory name.
/// The id and the label move independently: `rename_root` changes only the label, `relink_root`
/// changes only the path, and both keep every compiled plan's `rootId` valid.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovedRootV1 {
    pub root_id: String,
    pub path: PathBuf,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
}

impl ApprovedRootV1 {
    /// The label to show for this root: the user's own, else the directory's own name.
    pub fn display_label(&self) -> String {
        if !self.label.is_empty() {
            return self.label.clone();
        }
        self.path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.to_string_lossy().into_owned())
    }
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
        let mut paths = std::collections::BTreeSet::new();
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
            if !root.label.is_empty() {
                validate_root_label(&root.label)?;
            }
            if !ids.insert(root.root_id.as_str()) {
                return Err(CheckpointPlanError::Corrupt {
                    what: APPROVED_ROOTS_FILE.to_owned(),
                    message: format!("approved root id {:?} is declared twice", root.root_id),
                });
            }
            // Two ids bound to one directory is the shape `relink_root` would create if a later
            // `approve_root` of the relinked path derived a fresh id: the same files would then
            // compile twice under two identities and only one of them would ever be relinked.
            if !paths.insert(root.path.as_path()) {
                return Err(CheckpointPlanError::Corrupt {
                    what: APPROVED_ROOTS_FILE.to_owned(),
                    message: format!(
                        "approved directory {} is bound to two root ids",
                        root.path.display()
                    ),
                });
            }
        }
        Ok(())
    }

    pub fn get(&self, root_id: &str) -> Option<&ApprovedRootV1> {
        self.roots.iter().find(|root| root.root_id == root_id)
    }

    /// The approved root already bound to `canonical_path`, whatever its id.
    pub fn get_by_path(&self, canonical_path: &Path) -> Option<&ApprovedRootV1> {
        self.roots.iter().find(|root| root.path == canonical_path)
    }
}

/// A user-chosen library label: printable, bounded, and never a path component.
fn validate_root_label(label: &str) -> Result<(), CheckpointPlanError> {
    let invalid = |reason: &str| CheckpointPlanError::InvalidRootLabel {
        label: label.to_owned(),
        reason: reason.to_owned(),
    };
    if label.trim().is_empty() {
        return Err(invalid("must not be blank"));
    }
    if label.len() > MAX_ROOT_LABEL_BYTES {
        return Err(invalid("must be at most 128 bytes"));
    }
    if label.chars().any(char::is_control) {
        return Err(invalid("must not contain control characters"));
    }
    Ok(())
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
    /// The user-chosen library label cannot be persisted.
    InvalidRootLabel {
        label: String,
        reason: String,
    },
    /// A relink would bind this root id to a directory another approved root already owns.
    RootAlreadyApproved {
        root_id: String,
        existing_root_id: String,
        path: PathBuf,
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
    /// The plan uses a locator kind this store does not resolve (managed installs arrive with
    /// sc-20636).
    UnsupportedLocator {
        checkpoint_id: String,
        layer_id: String,
        kind: &'static str,
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
            Self::InvalidRootLabel { .. } => "invalid-root-label",
            Self::RootAlreadyApproved { .. } => "root-already-approved",
            Self::UnrunnableSource { .. } => "unrunnable-source",
            Self::UnknownCheckpoint { .. } => "unknown-checkpoint",
            Self::MissingPlan { .. } => "missing-plan",
            Self::PlanTampered { .. } => "plan-tampered",
            Self::SourceMissing { .. } => "source-missing",
            Self::SourceDrifted { .. } => "source-drifted",
            Self::UnsupportedLocator { .. } => "unsupported-locator",
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
            Self::InvalidRootLabel { label, reason } => {
                write!(f, "library label {label:?} is invalid: {reason}")
            }
            Self::RootAlreadyApproved {
                root_id,
                existing_root_id,
                path,
            } => write!(
                f,
                "root {root_id:?} cannot be relinked to {}: approved root {existing_root_id:?} already owns that directory",
                path.display()
            ),
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

/// Split a linked checkpoint id back into `(root_id, relative_path)`.
///
/// The id is the persisted identity, so every lifecycle action that has only an id — rescan,
/// status, per-root removal — recovers the root binding from it rather than from a stored absolute
/// path (E6). `None` for a managed id or any other shape. A root id never contains `/`
/// ([`derive_root_id`] emits `root-<hex>`), so the first separator is unambiguous.
pub fn parse_linked_checkpoint_id(checkpoint_id: &str) -> Option<(&str, &str)> {
    let rest = checkpoint_id.strip_prefix("linked/")?;
    let (root_id, relative_path) = rest.split_once('/')?;
    if root_id.is_empty() || relative_path.is_empty() {
        return None;
    }
    Some((root_id, relative_path))
}

/// Whether a persisted linked checkpoint can be selected right now, and if not, which lifecycle
/// action the user has to take (AC1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkedCheckpointStateV1 {
    /// Record, plan, approved root and every source byte verified: selectable.
    Ready,
    /// The approved root's directory is not there (unmounted volume, moved library). The plans
    /// stay valid; `relink_root` re-points the id at the library's new location.
    NeedsRelink,
    /// The root is present but this checkpoint's source is gone, changed, or its plan no longer
    /// matches its record. `rescan_checkpoint` recompiles it from the current bytes.
    NeedsRescan,
}

/// One persisted linked checkpoint and what the store currently thinks of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinkedCheckpointStatusV1 {
    pub checkpoint_id: String,
    pub root_id: String,
    pub relative_path: String,
    pub state: LinkedCheckpointStateV1,
    /// The typed refusal behind a non-`Ready` state, `[checkpoint-plan:<code>] …`.
    pub detail: Option<String>,
}

impl LinkedCheckpointStatusV1 {
    pub fn is_selectable(&self) -> bool {
        self.state == LinkedCheckpointStateV1::Ready
    }
}

/// One header-only candidate a library scan found, plus its compile state.
///
/// `selectable` is false for every candidate that has no `Ready` persisted plan: header evidence
/// is a hint, and only the full-content compile can promote it (E7).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinkedCandidateV1 {
    pub checkpoint_id: String,
    pub candidate: CheckpointCandidateV1,
    pub status: Option<LinkedCheckpointStatusV1>,
    pub selectable: bool,
}

/// The result of rescanning one approved root.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinkedRootScanV1 {
    pub root: ApprovedRootV1,
    /// False when the root directory is not there; every one of its checkpoints is Needs Relink.
    pub available: bool,
    pub candidates: Vec<LinkedCandidateV1>,
    /// Persisted checkpoints under this root the scan did not see as candidates: deleted, renamed,
    /// or now unreadable. They surface here rather than vanishing from the catalog.
    pub unmatched: Vec<LinkedCheckpointStatusV1>,
    pub diagnostics: Vec<CheckpointDiagnosticV1>,
}

/// What removing an approved root dropped. Only store-owned documents — never a library file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootRemovalV1 {
    pub root: ApprovedRootV1,
    pub removed_checkpoints: Vec<String>,
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

/// The one place the shape of a linked `relativePath` is defined.
///
/// Exposed so a caller can fail fast on a malformed path (an API request, a UI form) using the
/// SAME rules `compile_linked` will apply, rather than a second, drifting copy of them. It is a
/// lexical check only: whether the path's canonical target actually stays inside the root is
/// decided later by [`confined_root_join`], on the path that is opened.
pub fn validate_linked_relative_path(relative_path: &str) -> Result<(), CheckpointPlanError> {
    validate_portable_relative_path(relative_path).map(|_| ())
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
}

impl CheckpointPlanStore {
    pub fn open(data_dir: &Path) -> Self {
        Self {
            root: data_dir.join(CHECKPOINTS_DIR),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
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

    fn store_lock_path(&self) -> PathBuf {
        self.root.join(STORE_LOCK_FILE)
    }

    /// Take the store's exclusive cross-process lock for one read-modify-write.
    ///
    /// The returned handle IS the lock: dropping the file releases it.
    ///
    /// The lock is NOT reentrant — `fs2` locks an open file description, so a second acquisition on
    /// a fresh descriptor blocks even inside one process. Every mutator therefore takes it exactly
    /// once and never calls another locking mutator underneath it: `remove_root` does its own
    /// inventory rewrite rather than calling `invalidate` in a loop, which is also what collapses
    /// its N document writes into one.
    fn lock_store(&self) -> Result<fs::File, CheckpointPlanError> {
        fs::create_dir_all(&self.root).map_err(|error| io_error(&self.root, error))?;
        let path = self.store_lock_path();
        // A lock file that is a symlink would let a planted link move the lock (and the writes it
        // guards) somewhere else; the store owns this path, so anything but a regular file refuses.
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(CheckpointPlanError::Io {
                    path,
                    message: "checkpoint store lock is not a regular file".to_owned(),
                });
            }
        }
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| io_error(&path, error))?;
        fs2::FileExt::lock_exclusive(&file).map_err(|error| io_error(&path, error))?;
        Ok(file)
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

    /// Approve an existing directory as a linked-library root, labelled by its own directory name.
    pub fn approve_root(&self, path: &Path) -> Result<ApprovedRootV1, CheckpointPlanError> {
        self.approve_root_inner(path, None)
    }

    /// Approve an existing directory under a user-chosen label.
    pub fn approve_root_with_label(
        &self,
        path: &Path,
        label: &str,
    ) -> Result<ApprovedRootV1, CheckpointPlanError> {
        validate_root_label(label)?;
        self.approve_root_inner(path, Some(label))
    }

    /// Canonicalize and confirm a candidate root directory. Every lifecycle action that binds a
    /// path goes through here, so a symlinked or reparse-point root is resolved to its real
    /// directory ONCE and every later `confined_root_join` compares against that real directory.
    fn approvable_canonical_root(path: &Path) -> Result<PathBuf, CheckpointPlanError> {
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
        Ok(canonical)
    }

    fn approve_root_inner(
        &self,
        path: &Path,
        label: Option<&str>,
    ) -> Result<ApprovedRootV1, CheckpointPlanError> {
        let canonical = Self::approvable_canonical_root(path)?;
        let _guard = self.lock_store()?;
        let mut roots = self.approved_roots()?;
        // Path first, id second. After a `relink_root` a directory is bound to an id that
        // `derive_root_id` would NOT produce for it, and re-approving it must return that existing
        // binding rather than mint a second id for the same files.
        if let Some(existing) = roots.get_by_path(&canonical) {
            let existing_id = existing.root_id.clone();
            // Re-approving an already-approved directory under a NEW label is a relabel, not a
            // no-op: the caller asked for that label, and silently discarding it would leave the
            // library showing a name the user did not choose with nothing to say why.
            match label {
                Some(label) if existing.label != label => {
                    let root = roots
                        .roots
                        .iter_mut()
                        .find(|root| root.root_id == existing_id)
                        .expect("the root just found by path is in the list");
                    root.label = label.to_owned();
                    let relabelled = root.clone();
                    self.save_approved_roots(&roots)?;
                    return Ok(relabelled);
                }
                _ => return Ok(roots.get(&existing_id).expect("just found").clone()),
            }
        }
        let root_id = derive_root_id(&canonical);
        if let Some(existing) = roots.get(&root_id) {
            // `derive_root_id` truncates the digest, so a matching id is not by itself proof of a
            // matching path — and the path check above already returned for the same directory, so
            // reaching here means two different directories collided on one id.
            return Err(CheckpointPlanError::RootIdCollision {
                root_id,
                existing_path: existing.path.clone(),
                path: canonical,
            });
        }
        let root = ApprovedRootV1 {
            root_id,
            path: canonical,
            label: label.unwrap_or_default().to_owned(),
        };
        roots.roots.push(root.clone());
        self.save_approved_roots(&roots)?;
        Ok(root)
    }

    /// Relabel an approved root. Presentation only: the id, the path, and every compiled plan are
    /// untouched, and nothing under the library is read or written.
    pub fn rename_root(
        &self,
        root_id: &str,
        label: &str,
    ) -> Result<ApprovedRootV1, CheckpointPlanError> {
        validate_root_label(label)?;
        let _guard = self.lock_store()?;
        let mut roots = self.approved_roots()?;
        let root = roots
            .roots
            .iter_mut()
            .find(|root| root.root_id == root_id)
            .ok_or_else(|| CheckpointPlanError::UnknownRoot {
                root_id: root_id.to_owned(),
            })?;
        root.label = label.to_owned();
        let renamed = root.clone();
        self.save_approved_roots(&roots)?;
        Ok(renamed)
    }

    /// Re-point an approved root at the library's new location, keeping the root id so every
    /// persisted `Linked { rootId, relativePath }` stays valid (E6).
    ///
    /// The new directory is canonicalized and confirmed here; the plans are NOT trusted afterwards.
    /// A relink can legitimately be a move of the same library, but it can also be a retarget at a
    /// different one, so the caller's next `resolve`/`scan_root` re-verifies every layer's bytes
    /// against the fingerprint the plan was compiled from and refuses on drift.
    pub fn relink_root(
        &self,
        root_id: &str,
        path: &Path,
    ) -> Result<ApprovedRootV1, CheckpointPlanError> {
        let canonical = Self::approvable_canonical_root(path)?;
        let _guard = self.lock_store()?;
        let mut roots = self.approved_roots()?;
        if let Some(existing) = roots.get_by_path(&canonical) {
            if existing.root_id != root_id {
                return Err(CheckpointPlanError::RootAlreadyApproved {
                    root_id: root_id.to_owned(),
                    existing_root_id: existing.root_id.clone(),
                    path: canonical,
                });
            }
            return Ok(existing.clone());
        }
        let root = roots
            .roots
            .iter_mut()
            .find(|root| root.root_id == root_id)
            .ok_or_else(|| CheckpointPlanError::UnknownRoot {
                root_id: root_id.to_owned(),
            })?;
        root.path = canonical;
        let relinked = root.clone();
        self.save_approved_roots(&roots)?;
        Ok(relinked)
    }

    /// Forget an approved root and every checkpoint compiled from it.
    ///
    /// Removal is store-scoped by construction: the only paths this touches are
    /// `<data>/checkpoints/{approved-roots.json,inventory.json,plans/…,bindings/…}`. The library
    /// directory is never opened for writing, never enumerated for deletion, and its files are
    /// never referenced here at all (E6).
    ///
    /// The whole removal is ONE read-modify-write under the store lock, and the inventory is
    /// rewritten once rather than once per checkpoint. The previous per-checkpoint loop rewrote
    /// `inventory.json` N times with `?` in between and only then saved the root list, so a
    /// failure part-way left a root that still existed with some of its checkpoints already
    /// forgotten. Now the only interruptible window is between the two document writes, and it
    /// leaves a root with no checkpoints — a state a retry of `remove_root` converges from.
    pub fn remove_root(&self, root_id: &str) -> Result<RootRemovalV1, CheckpointPlanError> {
        let _guard = self.lock_store()?;
        let mut roots = self.approved_roots()?;
        let root = roots
            .get(root_id)
            .cloned()
            .ok_or_else(|| CheckpointPlanError::UnknownRoot {
                root_id: root_id.to_owned(),
            })?;
        let mut removed_checkpoints = Vec::new();
        let mut orphaned_plans = Vec::new();
        let mut remaining = Vec::new();
        for record in self.inventory()?.records {
            if parse_linked_checkpoint_id(&record.checkpoint_id)
                .is_some_and(|(id, _)| id == root_id)
            {
                removed_checkpoints.push(record.checkpoint_id.clone());
                orphaned_plans.push(record.plan.plan_id.clone());
            } else {
                remaining.push(record);
            }
        }
        self.save_inventory(&CheckpointInventoryV1::new(remaining)?)?;
        roots.roots.retain(|existing| existing.root_id != root_id);
        self.save_approved_roots(&roots)?;
        // Last, and only after both catalogs agree the checkpoints are gone: a failure here can
        // leave documents nothing references, never a record without its plan.
        for plan_id in orphaned_plans {
            self.remove_plan_documents(&plan_id)?;
        }
        Ok(RootRemovalV1 {
            root,
            removed_checkpoints,
        })
    }

    /// Every persisted checkpoint id whose identity names `root_id`, in inventory order.
    pub fn linked_checkpoint_ids_for_root(
        &self,
        root_id: &str,
    ) -> Result<Vec<String>, CheckpointPlanError> {
        Ok(self
            .inventory()?
            .records
            .into_iter()
            .filter(|record| {
                parse_linked_checkpoint_id(&record.checkpoint_id)
                    .is_some_and(|(id, _)| id == root_id)
            })
            .map(|record| record.checkpoint_id)
            .collect())
    }

    /// What the store currently thinks of one persisted linked checkpoint.
    ///
    /// This is [`Self::resolve`] reduced to its verdict: it runs the same record ↔ plan agreement,
    /// the same approved-root lookup, and the same per-layer confinement and byte verification, and
    /// classifies the typed refusal into the lifecycle action the user has to take. A missing root
    /// is Needs Relink; anything about the bytes is Needs Rescan.
    pub fn checkpoint_status(
        &self,
        checkpoint_id: &str,
    ) -> Result<LinkedCheckpointStatusV1, CheckpointPlanError> {
        let (root_id, relative_path) =
            parse_linked_checkpoint_id(checkpoint_id).ok_or_else(|| {
                CheckpointPlanError::UnknownCheckpoint {
                    checkpoint_id: checkpoint_id.to_owned(),
                }
            })?;
        // Prove the record exists before classifying: an id nobody compiled is not a lifecycle
        // state, it is an unknown checkpoint, and swallowing that into `NeedsRescan` would make
        // every typo look like a repairable library.
        self.record(checkpoint_id)?;
        let status =
            |state: LinkedCheckpointStateV1, detail: Option<String>| LinkedCheckpointStatusV1 {
                checkpoint_id: checkpoint_id.to_owned(),
                root_id: root_id.to_owned(),
                relative_path: relative_path.to_owned(),
                state,
                detail,
            };
        match self.resolve(checkpoint_id) {
            Ok(_) => Ok(status(LinkedCheckpointStateV1::Ready, None)),
            Err(
                error @ (CheckpointPlanError::UnknownRoot { .. }
                | CheckpointPlanError::RootUnavailable { .. }),
            ) => Ok(status(
                LinkedCheckpointStateV1::NeedsRelink,
                Some(error.to_string()),
            )),
            Err(
                error @ (CheckpointPlanError::SourceMissing { .. }
                | CheckpointPlanError::SourceDrifted { .. }
                | CheckpointPlanError::PathEscapesRoot { .. }
                | CheckpointPlanError::PlanTampered { .. }
                | CheckpointPlanError::MissingPlan { .. }
                | CheckpointPlanError::InvalidRelativePath { .. }
                | CheckpointPlanError::UnsupportedLocator { .. }),
            ) => Ok(status(
                LinkedCheckpointStateV1::NeedsRescan,
                Some(error.to_string()),
            )),
            // Corruption, I/O and contract failures are the store's own problem, not a library
            // lifecycle state; returning them keeps them from being papered over as "rescan".
            Err(error) => Err(error),
        }
    }

    /// Recompile one persisted linked checkpoint from the bytes on disk right now.
    ///
    /// Identity is preserved (`rootId + relativePath`); the plan, its layer fingerprints and the
    /// source stamps are replaced. `compile_linked` supersedes the previous plan documents through
    /// `upsert_record`, so a rescan leaves no orphaned plan behind.
    pub fn rescan_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<CompiledCheckpointV1, CheckpointPlanError> {
        let (root_id, relative_path) =
            parse_linked_checkpoint_id(checkpoint_id).ok_or_else(|| {
                CheckpointPlanError::UnknownCheckpoint {
                    checkpoint_id: checkpoint_id.to_owned(),
                }
            })?;
        self.compile_linked(root_id, relative_path)
    }

    /// Rescan an approved root: bounded header-only discovery of every candidate it holds, joined
    /// with the compile state of everything already persisted under it (AC1).
    ///
    /// Read-only with respect to the library. Discovery reads container headers, and each
    /// candidate's path is re-confined with [`confined_root_join`] — the same primitive a later
    /// compile uses to open it — so a candidate whose canonical target escapes the root is dropped
    /// with a `PathEscapesRoot` diagnostic instead of being offered (AC2).
    pub fn scan_root(&self, root_id: &str) -> Result<LinkedRootScanV1, CheckpointPlanError> {
        let roots = self.approved_roots()?;
        let root = roots
            .get(root_id)
            .cloned()
            .ok_or_else(|| CheckpointPlanError::UnknownRoot {
                root_id: root_id.to_owned(),
            })?;
        let mut persisted: BTreeMap<String, LinkedCheckpointStatusV1> = BTreeMap::new();
        let mut status_diagnostics = Vec::new();
        for checkpoint_id in self.linked_checkpoint_ids_for_root(root_id)? {
            match self.checkpoint_status(&checkpoint_id) {
                Ok(status) => {
                    persisted.insert(status.relative_path.clone(), status);
                }
                // One unreadable plan or bindings document is a fact about THAT checkpoint, not
                // about the library. Propagating it made a single corrupt file unlist every other
                // checkpoint under the root — including the ones a rescan could repair.
                Err(error) => status_diagnostics.push(CheckpointDiagnosticV1 {
                    severity: CheckpointDiagnosticSeverityV1::Error,
                    code: CheckpointDiagnosticCodeV1::Io,
                    relative_path: parse_linked_checkpoint_id(&checkpoint_id)
                        .map(|(_, relative_path)| relative_path.to_owned()),
                    message: error.to_string(),
                }),
            }
        }
        if !root.path.is_dir() {
            // An unavailable root has no candidates to offer and nothing to say about bytes: every
            // compiled checkpoint under it is Needs Relink, whatever its last known state was.
            let unmatched = persisted
                .into_values()
                .map(|status| LinkedCheckpointStatusV1 {
                    state: LinkedCheckpointStateV1::NeedsRelink,
                    ..status
                })
                .collect();
            status_diagnostics.insert(
                0,
                CheckpointDiagnosticV1 {
                    severity: CheckpointDiagnosticSeverityV1::Error,
                    code: CheckpointDiagnosticCodeV1::SourceNotFound,
                    relative_path: None,
                    message: format!(
                        "approved root {root_id:?} at {} is not available (unmounted or moved)",
                        root.path.display()
                    ),
                },
            );
            return Ok(LinkedRootScanV1 {
                diagnostics: status_diagnostics,
                root,
                available: false,
                candidates: Vec::new(),
                unmatched,
            });
        }
        let discovery = discover_library_root(&root.path);
        let mut diagnostics = status_diagnostics;
        diagnostics.extend(discovery.diagnostics);
        let mut candidates = Vec::with_capacity(discovery.candidates.len());
        for candidate in discovery.candidates {
            let checkpoint_id = linked_checkpoint_id(root_id, &candidate.relative_path);
            // Re-confine on the exact path a compile would open, not on the discovery string.
            if let Err(error) =
                confined_root_join(&checkpoint_id, &root.path, &candidate.relative_path)
            {
                diagnostics.push(CheckpointDiagnosticV1 {
                    severity: CheckpointDiagnosticSeverityV1::Error,
                    code: CheckpointDiagnosticCodeV1::PathEscapesRoot,
                    relative_path: Some(candidate.relative_path.clone()),
                    message: error.to_string(),
                });
                continue;
            }
            let status = persisted.remove(&candidate.relative_path);
            candidates.push(LinkedCandidateV1 {
                selectable: status
                    .as_ref()
                    .is_some_and(LinkedCheckpointStatusV1::is_selectable),
                checkpoint_id,
                candidate,
                status,
            });
        }
        let unmatched = persisted
            .into_values()
            .map(|status| match status.state {
                // The scan did not see it, so whatever the per-checkpoint verdict was, the source
                // is not where the plan says: rescan, not relink (the root itself is present).
                LinkedCheckpointStateV1::Ready => LinkedCheckpointStatusV1 {
                    state: LinkedCheckpointStateV1::NeedsRescan,
                    detail: Some(
                        "[checkpoint-plan:source-missing] the library scan no longer finds this \
                         checkpoint under its approved root"
                            .to_owned(),
                    ),
                    ..status
                },
                _ => status,
            })
            .collect();
        Ok(LinkedRootScanV1 {
            root,
            available: true,
            candidates,
            unmatched,
            diagnostics,
        })
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
        let inspection = inspect_checkpoint(&request);
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
            let path = self.layer_path(&checkpoint_id, &root_path, layer)?;
            let stamp = SourceStampV1::of(&path).map_err(|error| io_error(&path, error))?;
            stamps.insert(layer.layer_id.clone(), stamp);
        }
        let bindings = SourceBindingsV1 {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            plan_id: plan.plan_id.clone(),
            stamps,
        };

        self.persist_plan(&plan)?;
        self.persist_bindings(&bindings)?;
        self.upsert_record(record.clone())?;
        Ok(CompiledCheckpointV1 {
            checkpoint_id,
            record,
            plan,
        })
    }

    fn layer_path(
        &self,
        checkpoint_id: &str,
        root_path: &Path,
        layer: &ImportLayerV1,
    ) -> Result<PathBuf, CheckpointPlanError> {
        let relative = match &layer.source {
            SourceLocatorV1::Linked { relative_path, .. } => relative_path,
            SourceLocatorV1::Managed { .. } => {
                return Err(CheckpointPlanError::UnsupportedLocator {
                    checkpoint_id: checkpoint_id.to_owned(),
                    layer_id: layer.layer_id.clone(),
                    kind: "managed",
                })
            }
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
        let _guard = self.lock_store()?;
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
        let _guard = self.lock_store()?;
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
            let (root_id, relative_path, fingerprint) = match &layer.source {
                SourceLocatorV1::Linked {
                    root_id,
                    relative_path,
                    fingerprint,
                    ..
                } => (root_id, relative_path, fingerprint),
                SourceLocatorV1::Managed { .. } => {
                    return Err(CheckpointPlanError::UnsupportedLocator {
                        checkpoint_id: checkpoint_id.to_owned(),
                        layer_id: layer.layer_id.clone(),
                        kind: "managed",
                    })
                }
            };
            let root_path = self.resolve_root(root_id)?;
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
