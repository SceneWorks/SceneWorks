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
    inspect_checkpoint, CheckpointDiagnosticSeverityV1, CheckpointDiagnosticV1,
    CheckpointInspectionRequestV1,
};

/// Directory under the application data dir that holds every file this store owns.
pub const CHECKPOINTS_DIR: &str = "checkpoints";
pub const APPROVED_ROOTS_FILE: &str = "approved-roots.json";
pub const INVENTORY_FILE: &str = "inventory.json";
pub const PLANS_DIR: &str = "plans";
pub const BINDINGS_DIR: &str = "bindings";

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
    /// The plan uses a locator kind this store does not resolve (managed installs arrive with
    /// sc-20636).
    UnsupportedLocator {
        checkpoint_id: String,
        layer_id: String,
        kind: &'static str,
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

    fn plan_path(&self, plan_id: &str) -> PathBuf {
        self.root.join(PLANS_DIR).join(format!("{plan_id}.json"))
    }

    fn bindings_path(&self, plan_id: &str) -> PathBuf {
        self.root.join(BINDINGS_DIR).join(format!("{plan_id}.json"))
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
        Ok(root_path.join(validate_portable_relative_path(relative)?))
    }

    fn persist_plan(&self, plan: &ImportPlanV1) -> Result<(), CheckpointPlanError> {
        let mut payload = plan.canonical_json()?.into_bytes();
        payload.push(b'\n');
        self.write_atomic(&self.plan_path(&plan.plan_id), &payload)
    }

    fn persist_bindings(&self, bindings: &SourceBindingsV1) -> Result<(), CheckpointPlanError> {
        let mut payload =
            serde_json::to_vec_pretty(bindings).map_err(|error| CheckpointPlanError::Corrupt {
                what: "source bindings".to_owned(),
                message: error.to_string(),
            })?;
        payload.push(b'\n');
        self.write_atomic(&self.bindings_path(&bindings.plan_id), &payload)
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
        for path in [self.plan_path(plan_id), self.bindings_path(plan_id)] {
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
        self.read_json::<ImportPlanV1>(&self.plan_path(plan_id), "import plan")?
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
            .read_json::<SourceBindingsV1>(&self.bindings_path(&plan.plan_id), "source bindings")?
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
            let path = root_path.join(validate_portable_relative_path(relative_path)?);
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
