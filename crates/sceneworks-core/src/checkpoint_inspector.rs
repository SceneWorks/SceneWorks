//! Deterministic, CPU-only inspection of community checkpoint containers.
//!
//! Discovery and runnable validation are deliberately separate. Discovery performs a bounded
//! directory walk and reads only container descriptors (the safetensors JSON header or GGUF
//! magic). Runnable validation then parses descriptor contents strictly, validates every declared
//! tensor range, and streams every artifact through SHA-256. This keeps catalog discovery cheap
//! while ensuring an emitted [`CheckpointInventoryV1`] is bound to the exact bytes a loader will
//! consume. Inspection does not retain file handles or lock community-owned files: a later consumer
//! must verify each locator's recorded digest against the bytes it opens before loading them.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::base_weights::{classify_base_header, BaseWeightDetection};
use crate::checkpoint_import::{
    CheckpointCatalogRecordV1, CheckpointInventoryV1, ImportLayerV1, ImportPlanV1,
    ManagedProvenanceV1, SourceLocatorV1,
};
// Re-exported rather than defined here (sc-20651): the container verdict is now a FIELD of
// `ImportLayerV1`, so its type has to live in the contract module, which is a leaf and must not
// depend on this one. Every existing `checkpoint_inspector::CheckpointContainerV1` path keeps
// resolving through this re-export.
pub use crate::checkpoint_import::CheckpointContainerV1;
use crate::checkpoint_plan_store::portable_relative_path_parts;

const MAX_DISCOVERY_ENTRIES: usize = 4096;
const MAX_DISCOVERY_DEPTH: usize = 8;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;
const MAX_JSON_DESCRIPTOR_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GGUF_DESCRIPTOR_BYTES: u64 = 128 * 1024 * 1024;
const MAX_GGUF_ITEMS: u64 = 1_000_000;
const MAX_GGUF_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GGUF_ARRAY_DEPTH: usize = 16;
const HASH_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_AGGREGATE_TENSOR_NAMES: usize = 65_536;
const MAX_AGGREGATE_TENSOR_NAME_BYTES: usize = 32 * 1024 * 1024;
const MAX_AGGREGATE_EVIDENCE_STRING_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointLayoutV1 {
    SingleFile,
    FusedCheckpoint,
    ComponentDirectory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointDiagnosticSeverityV1 {
    Error,
    Warning,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointDiagnosticCodeV1 {
    SourceNotFound,
    PathEscapesRoot,
    DiscoveryLimitExceeded,
    NoWeightCandidates,
    UnsupportedContainer,
    HeaderTooLarge,
    TruncatedHeader,
    TruncatedData,
    DuplicateKey,
    MalformedMetadata,
    InvalidTensorRange,
    UnsupportedTensorType,
    MissingSidecar,
    IndexTensorMismatch,
    AmbiguousComponentRole,
    MissingFamilyEvidence,
    FamilyDialectConflict,
    SourceChangedDuringInspection,
    Io,
    ContractViolation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointDiagnosticV1 {
    pub severity: CheckpointDiagnosticSeverityV1,
    pub code: CheckpointDiagnosticCodeV1,
    pub relative_path: Option<String>,
    pub message: String,
}

impl CheckpointDiagnosticV1 {
    fn error(
        code: CheckpointDiagnosticCodeV1,
        relative_path: Option<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity: CheckpointDiagnosticSeverityV1::Error,
            code,
            relative_path,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointCandidateV1 {
    pub relative_path: String,
    pub container: CheckpointContainerV1,
    pub size_bytes: u64,
    pub header_role: Option<String>,
    pub header_family: Option<String>,
    pub quantization: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointDiscoveryV1 {
    pub layout: Option<CheckpointLayoutV1>,
    pub candidates: Vec<CheckpointCandidateV1>,
    pub descriptor_paths: Vec<String>,
    /// Every discovered file's ROOT-relative path mapped to its CHECKPOINT-relative one.
    ///
    /// Two names for the same file, and the seam depends on keeping them apart: the root-relative
    /// key is where the bytes are and is what a source locator records; the value is where the file
    /// sits inside the checkpoint and is what the plan's semantic identity is built from. They
    /// differ by the checkpoint's own depth under the root, and additionally whenever the
    /// checkpoint reaches a file through a symlink to somewhere else under the same root.
    #[serde(default)]
    pub internal_paths: BTreeMap<String, String>,
    pub diagnostics: Vec<CheckpointDiagnosticV1>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointOwnershipV1 {
    Linked {
        root_id: String,
    },
    Managed {
        install_id: String,
        provenance: ManagedProvenanceV1,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointInspectionRequestV1 {
    pub checkpoint_id: String,
    pub root_path: PathBuf,
    pub relative_path: PathBuf,
    pub ownership: CheckpointOwnershipV1,
}

impl CheckpointInspectionRequestV1 {
    pub fn linked(
        checkpoint_id: impl Into<String>,
        root_path: impl Into<PathBuf>,
        relative_path: &str,
        root_id: impl Into<String>,
    ) -> Result<Self, String> {
        Self::new(
            checkpoint_id.into(),
            root_path.into(),
            relative_path,
            CheckpointOwnershipV1::Linked {
                root_id: root_id.into(),
            },
        )
    }

    pub fn managed(
        checkpoint_id: impl Into<String>,
        root_path: impl Into<PathBuf>,
        relative_path: &str,
        install_id: impl Into<String>,
        provenance: ManagedProvenanceV1,
    ) -> Result<Self, String> {
        Self::new(
            checkpoint_id.into(),
            root_path.into(),
            relative_path,
            CheckpointOwnershipV1::Managed {
                install_id: install_id.into(),
                provenance,
            },
        )
    }

    /// `relative_path` is the PORTABLE `/`-separated document spelling, not a native path.
    ///
    /// That distinction is the whole reason this takes `&str` (sc-20651). Taking `impl Into<PathBuf>`
    /// and validating the resulting `Path` made the rule PLATFORM-DEPENDENT: on Windows
    /// `PathBuf::from("dir\\model.safetensors")` splits into two `Normal` components and normalises
    /// back to `dir/model.safetensors`, so the constructor accepted a spelling the contract
    /// (`SourceLocatorV1`) refuses on every platform, while Unix refused it. The windows-candle lane
    /// caught exactly that divergence.
    ///
    /// Deciding on the STRING, before any `std::path` semantics, is what makes the shared rule hold
    /// identically on both platforms. The native `PathBuf` is then BUILT from the validated parts —
    /// which is what the two production callers were doing themselves anyway.
    fn new(
        checkpoint_id: String,
        root_path: PathBuf,
        relative_path: &str,
        ownership: CheckpointOwnershipV1,
    ) -> Result<Self, String> {
        if checkpoint_id.trim().is_empty() {
            return Err("checkpoint id must not be blank".to_owned());
        }
        let Ok(relative_path) = portable_relative_path_parts(relative_path) else {
            return Err("checkpoint path must be a non-empty confined relative path".to_owned());
        };
        match &ownership {
            CheckpointOwnershipV1::Linked { root_id } if root_id.trim().is_empty() => {
                return Err("linked root id must not be blank".to_owned());
            }
            CheckpointOwnershipV1::Managed {
                install_id,
                provenance,
            } => {
                if install_id.trim().is_empty() {
                    return Err("managed install id must not be blank".to_owned());
                }
                provenance.validate().map_err(|error| error.to_string())?;
            }
            _ => {}
        }
        Ok(Self {
            checkpoint_id,
            root_path,
            relative_path,
            ownership,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointArtifactEvidenceV1 {
    pub relative_path: String,
    pub container: CheckpointContainerV1,
    pub role: Option<String>,
    pub family: Option<String>,
    pub dialect: Option<String>,
    /// SHA-256 from the authoritative final exact-byte pass.
    ///
    /// This is durable locator evidence, not a filesystem lock. Any consumer that opens the path
    /// later must hash the bytes it will load and reject a digest mismatch.
    pub sha256: String,
    pub size_bytes: u64,
    pub declared_tensor_bytes: Option<u64>,
    pub tensor_names: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointInspectionV1 {
    pub layout: Option<CheckpointLayoutV1>,
    pub fingerprint: Option<String>,
    pub inventory: CheckpointInventoryV1,
    pub plans: Vec<ImportPlanV1>,
    pub evidence: Vec<CheckpointArtifactEvidenceV1>,
    pub diagnostics: Vec<CheckpointDiagnosticV1>,
}

impl CheckpointInspectionV1 {
    pub fn is_runnable(&self) -> bool {
        !self.inventory.records.is_empty()
            && self
                .diagnostics
                .iter()
                .all(|item| item.severity != CheckpointDiagnosticSeverityV1::Error)
    }
}

/// Stable inspection boundary exposed for deterministic fault-injection tests.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointInspectionEventV1 {
    FirstExactBytePassComplete,
    SecondExactBytePassComplete,
}

#[derive(Clone)]
struct ResolvedInput {
    canonical_root: PathBuf,
    canonical_target: PathBuf,
}

/// One discovered file, in BOTH of the two names it has.
///
/// `canonical` is where the bytes actually are, which is what a source locator records and what a
/// confinement check is applied to. `internal` is where the file sits INSIDE the checkpoint — the
/// path the walk took to reach it, before any symlink was resolved — which is what the checkpoint's
/// semantic identity is built from. They differ whenever the checkpoint links to a file kept
/// elsewhere under the same root, and conflating them is what used to make the semantic digest
/// depend on where in the library the checkpoint was kept.
#[derive(Clone, Debug)]
struct DiscoveredFile {
    canonical: PathBuf,
    internal: String,
}

#[derive(Default)]
struct DiscoveryFiles {
    weights: Vec<DiscoveredFile>,
    descriptors: Vec<DiscoveredFile>,
    diagnostics: Vec<CheckpointDiagnosticV1>,
}

pub fn discover_checkpoint(request: &CheckpointInspectionRequestV1) -> CheckpointDiscoveryV1 {
    let Some(resolved) = resolve_input(request) else {
        return CheckpointDiscoveryV1 {
            layout: None,
            candidates: Vec::new(),
            descriptor_paths: Vec::new(),
            internal_paths: BTreeMap::new(),
            diagnostics: vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::SourceNotFound,
                path_to_portable_string(&request.relative_path),
                format!(
                    "checkpoint source '{}' does not exist or its root is unavailable",
                    request.relative_path.display()
                ),
            )],
        };
    };

    if !resolved
        .canonical_target
        .starts_with(&resolved.canonical_root)
    {
        return CheckpointDiscoveryV1 {
            layout: None,
            candidates: Vec::new(),
            descriptor_paths: Vec::new(),
            internal_paths: BTreeMap::new(),
            diagnostics: vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::PathEscapesRoot,
                path_to_portable_string(&request.relative_path),
                "checkpoint source resolves outside its declared root",
            )],
        };
    }

    let direct_file = resolved.canonical_target.is_file();
    let files = if direct_file {
        discover_direct_file(&resolved)
    } else if resolved.canonical_target.is_dir() {
        discover_directory(&resolved)
    } else {
        DiscoveryFiles {
            diagnostics: vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::UnsupportedContainer,
                relative_to_root(&resolved, &resolved.canonical_target),
                "checkpoint source is neither a regular file nor a directory",
            )],
            ..DiscoveryFiles::default()
        }
    };

    let mut diagnostics = files.diagnostics;
    let mut internal_paths = BTreeMap::new();
    let mut candidates = Vec::new();
    for file in files.weights {
        match discover_weight(&resolved, &file) {
            Ok(candidate) => {
                internal_paths.insert(candidate.relative_path.clone(), file.internal.clone());
                candidates.push(candidate);
            }
            Err(diagnostic) => diagnostics.push(diagnostic),
        }
    }
    candidates.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut descriptor_paths = Vec::new();
    for file in &files.descriptors {
        let Some(relative) = relative_to_root(&resolved, &file.canonical) else {
            continue;
        };
        internal_paths.insert(relative.clone(), file.internal.clone());
        descriptor_paths.push(relative);
    }
    descriptor_paths.sort();
    descriptor_paths.dedup();

    if candidates.is_empty() {
        diagnostics.push(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::NoWeightCandidates,
            path_to_portable_string(&request.relative_path),
            "no safetensors or GGUF checkpoint candidates were found",
        ));
    }

    let layout = if candidates.is_empty() {
        None
    } else if !direct_file {
        Some(CheckpointLayoutV1::ComponentDirectory)
    } else if candidates[0].container == CheckpointContainerV1::Safetensors
        && candidates[0].header_role.as_deref() == Some("checkpoint")
    {
        Some(CheckpointLayoutV1::FusedCheckpoint)
    } else {
        Some(CheckpointLayoutV1::SingleFile)
    };
    sort_diagnostics(&mut diagnostics);
    CheckpointDiscoveryV1 {
        layout,
        candidates,
        descriptor_paths,
        internal_paths,
        diagnostics,
    }
}

/// Bounded, header-only discovery of every checkpoint candidate an approved linked-library root
/// holds (epic 20398, sc-20635).
///
/// [`discover_checkpoint`] answers "what is inside THIS one checkpoint source"; a library root is
/// a directory of many unrelated checkpoints, so each discovered weight file is reported as its own
/// candidate keyed by its portable root-relative path — the exact `relativePath`
/// `CheckpointPlanStore::compile_linked` takes.
///
/// This is discovery, not validation: nothing here is selectable. Only the full-content compile
/// (which streams every byte through SHA-256) can promote a candidate to a runnable plan (E7), so
/// the caller renders these as visible-but-unselectable until it has a persisted record.
///
/// Every path is canonicalized and required to stay under the canonical root; a candidate that
/// escapes is dropped with a [`CheckpointDiagnosticCodeV1::PathEscapesRoot`] diagnostic rather than
/// being reported (fail closed, E6/E7).
pub fn discover_library_root(root_path: &Path) -> CheckpointDiscoveryV1 {
    let mut diagnostics = Vec::new();
    let canonical_root = match std::fs::canonicalize(root_path) {
        Ok(canonical) => canonical,
        Err(error) => {
            return CheckpointDiscoveryV1 {
                layout: None,
                candidates: Vec::new(),
                descriptor_paths: Vec::new(),
                internal_paths: BTreeMap::new(),
                diagnostics: vec![CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::SourceNotFound,
                    None,
                    format!(
                        "library root '{}' is unavailable: {error}",
                        root_path.display()
                    ),
                )],
            };
        }
    };
    if !canonical_root.is_dir() {
        return CheckpointDiscoveryV1 {
            layout: None,
            candidates: Vec::new(),
            descriptor_paths: Vec::new(),
            internal_paths: BTreeMap::new(),
            diagnostics: vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::UnsupportedContainer,
                None,
                format!("library root '{}' is not a directory", root_path.display()),
            )],
        };
    }
    let resolved = ResolvedInput {
        canonical_target: canonical_root.clone(),
        canonical_root,
    };
    let files = discover_directory(&resolved);
    diagnostics.extend(files.diagnostics);
    let mut internal_paths = BTreeMap::new();
    let mut candidates = Vec::new();
    for file in files.weights {
        match discover_weight(&resolved, &file) {
            Ok(candidate) => {
                internal_paths.insert(candidate.relative_path.clone(), file.internal.clone());
                candidates.push(candidate);
            }
            Err(diagnostic) => diagnostics.push(diagnostic),
        }
    }
    candidates.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut descriptor_paths = Vec::new();
    for file in &files.descriptors {
        let Some(relative) = relative_to_root(&resolved, &file.canonical) else {
            continue;
        };
        internal_paths.insert(relative.clone(), file.internal.clone());
        descriptor_paths.push(relative);
    }
    descriptor_paths.sort();
    descriptor_paths.dedup();
    sort_diagnostics(&mut diagnostics);
    CheckpointDiscoveryV1 {
        // A library root is a container of many checkpoints, not one checkpoint with a layout.
        layout: None,
        candidates,
        descriptor_paths,
        internal_paths,
        diagnostics,
    }
}

pub fn inspect_checkpoint(request: &CheckpointInspectionRequestV1) -> CheckpointInspectionV1 {
    inspect_checkpoint_with_hook(request, |_| {})
}

/// Test seam for mutating an artifact before the authoritative final exact-byte pass.
#[doc(hidden)]
pub fn inspect_checkpoint_with_hook(
    request: &CheckpointInspectionRequestV1,
    mut hook: impl FnMut(CheckpointInspectionEventV1),
) -> CheckpointInspectionV1 {
    let discovery = discover_checkpoint(request);
    let mut result = CheckpointInspectionV1 {
        layout: discovery.layout,
        fingerprint: None,
        inventory: empty_inventory(),
        plans: Vec::new(),
        evidence: Vec::new(),
        diagnostics: discovery.diagnostics.clone(),
    };
    let Some(resolved) = resolve_input(request) else {
        return result;
    };
    if !resolved
        .canonical_target
        .starts_with(&resolved.canonical_root)
    {
        return result;
    }

    let first_pass = inspect_discovered_pass(&resolved, &discovery);
    hook(CheckpointInspectionEventV1::FirstExactBytePassComplete);
    let second_pass = inspect_discovered_pass(&resolved, &discovery);
    hook(CheckpointInspectionEventV1::SecondExactBytePassComplete);
    let final_pass = inspect_discovered_pass(&resolved, &discovery);
    result.diagnostics.extend(final_pass.diagnostics);
    compare_exact_byte_passes(
        &first_pass.observations,
        &second_pass.observations,
        &mut result.diagnostics,
    );
    compare_exact_byte_passes(
        &second_pass.observations,
        &final_pass.observations,
        &mut result.diagnostics,
    );
    result.evidence = final_pass.evidence;
    let backbone_families = final_pass.backbone_families;

    if result.evidence.iter().all(|item| !item.sha256.is_empty()) && !result.evidence.is_empty() {
        result.fingerprint = Some(inspection_fingerprint(&result.evidence));
    }

    let already_has_family_diagnostic = result.diagnostics.iter().any(|item| {
        matches!(
            item.code,
            CheckpointDiagnosticCodeV1::MissingFamilyEvidence
                | CheckpointDiagnosticCodeV1::FamilyDialectConflict
        )
    });
    if backbone_families.is_empty()
        && !discovery.candidates.is_empty()
        && !already_has_family_diagnostic
    {
        result.diagnostics.push(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MissingFamilyEvidence,
            path_to_portable_string(&request.relative_path),
            "no checkpoint family or architecture evidence was found in the weight or descriptor contents",
        ));
    } else if backbone_families.len() > 1 {
        result.diagnostics.push(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::FamilyDialectConflict,
            path_to_portable_string(&request.relative_path),
            format!(
                "checkpoint artifacts disagree about the model family: {}",
                backbone_families
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }

    let final_discovery = discover_checkpoint(request);
    if final_discovery.layout != discovery.layout
        || final_discovery.candidates != discovery.candidates
        || final_discovery.descriptor_paths != discovery.descriptor_paths
    {
        result.diagnostics.push(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::SourceChangedDuringInspection,
            path_to_portable_string(&request.relative_path),
            "checkpoint files changed while inspection was in progress; retry against a stationary source",
        ));
    }
    revalidate_artifact_observations(&resolved, &final_pass.observations, &mut result.diagnostics);

    sort_diagnostics(&mut result.diagnostics);
    if result
        .diagnostics
        .iter()
        .any(|item| item.severity == CheckpointDiagnosticSeverityV1::Error)
    {
        return result;
    }

    let Some(fingerprint) = result.fingerprint.clone() else {
        return result;
    };
    let family = backbone_families
        .into_iter()
        .next()
        .expect("family was validated above");
    match compile_inventory(
        request,
        &discovery.internal_paths,
        &family,
        &fingerprint,
        &result.evidence,
    ) {
        Ok((inventory, plan)) => {
            result.inventory = inventory;
            result.plans.push(plan);
        }
        Err(error) => result.diagnostics.push(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::ContractViolation,
            path_to_portable_string(&request.relative_path),
            format!("validated checkpoint could not compile to the v1 contracts: {error}"),
        )),
    }
    sort_diagnostics(&mut result.diagnostics);
    result
}

#[derive(Default)]
struct InspectionPass {
    evidence: Vec<CheckpointArtifactEvidenceV1>,
    observations: Vec<ArtifactObservation>,
    diagnostics: Vec<CheckpointDiagnosticV1>,
    backbone_families: BTreeSet<String>,
    aggregate_tensor_names: usize,
    aggregate_tensor_name_bytes: usize,
    aggregate_evidence_string_bytes: usize,
    evidence_budget_exhausted: bool,
}

impl InspectionPass {
    fn retain_evidence(&mut self, evidence: CheckpointArtifactEvidenceV1) -> bool {
        let tensor_names = evidence.tensor_names.len();
        let tensor_name_bytes = evidence
            .tensor_names
            .iter()
            .try_fold(0_usize, |total, name| total.checked_add(name.len()));
        let evidence_string_bytes = [
            Some(evidence.relative_path.len()),
            evidence.role.as_ref().map(String::len),
            evidence.family.as_ref().map(String::len),
            evidence.dialect.as_ref().map(String::len),
            Some(evidence.sha256.len()),
            tensor_name_bytes,
        ]
        .into_iter()
        .flatten()
        .try_fold(0_usize, usize::checked_add);
        let next_tensor_names = self.aggregate_tensor_names.checked_add(tensor_names);
        let next_tensor_name_bytes =
            tensor_name_bytes.and_then(|bytes| self.aggregate_tensor_name_bytes.checked_add(bytes));
        let next_evidence_string_bytes = evidence_string_bytes
            .and_then(|bytes| self.aggregate_evidence_string_bytes.checked_add(bytes));
        let within_budget = next_tensor_names
            .zip(next_tensor_name_bytes)
            .zip(next_evidence_string_bytes)
            .is_some_and(|((names, name_bytes), evidence_bytes)| {
                names <= MAX_AGGREGATE_TENSOR_NAMES
                    && name_bytes <= MAX_AGGREGATE_TENSOR_NAME_BYTES
                    && evidence_bytes <= MAX_AGGREGATE_EVIDENCE_STRING_BYTES
            });
        if !within_budget {
            if !self.evidence_budget_exhausted {
                self.diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::DiscoveryLimitExceeded,
                    Some(evidence.relative_path),
                    format!(
                        "checkpoint inspection exceeded the aggregate evidence budget of {MAX_AGGREGATE_TENSOR_NAMES} tensor names, {MAX_AGGREGATE_TENSOR_NAME_BYTES} tensor-name UTF-8 bytes, or {MAX_AGGREGATE_EVIDENCE_STRING_BYTES} total evidence UTF-8 bytes"
                    ),
                ));
                self.evidence_budget_exhausted = true;
            }
            return false;
        }
        self.aggregate_tensor_names = next_tensor_names.expect("budget arithmetic checked above");
        self.aggregate_tensor_name_bytes =
            next_tensor_name_bytes.expect("budget arithmetic checked above");
        self.aggregate_evidence_string_bytes =
            next_evidence_string_bytes.expect("budget arithmetic checked above");
        self.evidence.push(evidence);
        true
    }
}

struct ArtifactObservation {
    relative_path: String,
    container: CheckpointContainerV1,
    canonical_path: PathBuf,
    snapshot: ArtifactSnapshot,
}

fn inspect_discovered_pass(
    resolved: &ResolvedInput,
    discovery: &CheckpointDiscoveryV1,
) -> InspectionPass {
    let mut pass = InspectionPass::default();
    let candidate_by_path = discovery
        .candidates
        .iter()
        .map(|candidate| (candidate.relative_path.clone(), candidate))
        .collect::<BTreeMap<_, _>>();
    let mut tensor_tables = BTreeMap::new();
    let mut indices = Vec::new();

    for candidate in &discovery.candidates {
        let Some(path) =
            resolve_confined_artifact(resolved, &candidate.relative_path, &mut pass.diagnostics)
        else {
            continue;
        };
        let path_role = infer_role_from_path(
            path.strip_prefix(&resolved.canonical_target)
                .unwrap_or(path.as_path()),
        );
        let prefix_limit = match candidate.container {
            CheckpointContainerV1::Safetensors => MAX_SAFETENSORS_HEADER_BYTES + 9,
            CheckpointContainerV1::Gguf => MAX_GGUF_DESCRIPTOR_BYTES + 1,
            CheckpointContainerV1::JsonDescriptor => unreachable!("weight candidate is JSON"),
        };
        let mut snapshot = match snapshot_file(&path, prefix_limit) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                pass.diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::Io,
                    Some(candidate.relative_path.clone()),
                    format!("failed to read checkpoint artifact: {error}"),
                ));
                continue;
            }
        };
        let validation = match candidate.container {
            CheckpointContainerV1::Safetensors => validate_safetensors(
                &snapshot.prefix,
                snapshot.size_bytes,
                Some(candidate.relative_path.clone()),
            ),
            CheckpointContainerV1::Gguf => validate_gguf(
                &snapshot.prefix,
                snapshot.size_bytes,
                Some(candidate.relative_path.clone()),
            ),
            CheckpointContainerV1::JsonDescriptor => unreachable!("weight candidate is JSON"),
        };
        let mut family = candidate.header_family.clone();
        let mut dialect = candidate.quantization.clone();
        let mut declared_tensor_bytes = None;
        let mut validated_role = candidate.header_role.clone();
        let mut tensor_names = Vec::new();
        match validation {
            Ok(validated) => {
                family = validated.family;
                dialect = validated.dialect;
                validated_role = validated.role;
                declared_tensor_bytes = validated.declared_tensor_bytes;
                tensor_names = validated.tensor_names;
            }
            Err(mut diagnostics) => pass.diagnostics.append(&mut diagnostics),
        }
        if candidate.container == CheckpointContainerV1::Gguf
            && path_role.is_none()
            && resolved.canonical_target.is_file()
        {
            validated_role = Some("checkpoint".to_owned());
        }
        let role = refine_multi_expert_role(
            reconcile_role(
                &candidate.relative_path,
                path_role,
                validated_role.as_deref(),
                &mut pass.diagnostics,
            ),
            family.as_deref(),
            &candidate.relative_path,
        );
        if matches!(
            role.as_deref(),
            Some("transformer" | "checkpoint" | TRANSFORMER_HIGH_ROLE | TRANSFORMER_LOW_ROLE)
        ) {
            if let Some(family) = &family {
                if !is_auxiliary_family(family) {
                    pass.backbone_families.insert(family.clone());
                }
            }
        }
        let evidence = CheckpointArtifactEvidenceV1 {
            relative_path: candidate.relative_path.clone(),
            container: candidate.container,
            role,
            family,
            dialect,
            sha256: snapshot.sha256.clone(),
            size_bytes: snapshot.size_bytes,
            declared_tensor_bytes,
            tensor_names,
        };
        let retained = pass.retain_evidence(evidence);
        if retained && candidate.container == CheckpointContainerV1::Safetensors {
            let tensor_names = &pass
                .evidence
                .last()
                .expect("retained evidence was just appended")
                .tensor_names;
            tensor_tables.insert(
                candidate.relative_path.clone(),
                tensor_names.iter().cloned().collect::<BTreeSet<_>>(),
            );
        }
        snapshot.prefix = Vec::new();
        pass.observations.push(ArtifactObservation {
            relative_path: candidate.relative_path.clone(),
            container: candidate.container,
            canonical_path: path,
            snapshot,
        });
    }

    for descriptor_path in &discovery.descriptor_paths {
        let Some(path) =
            resolve_confined_artifact(resolved, descriptor_path, &mut pass.diagnostics)
        else {
            continue;
        };
        let mut snapshot = match snapshot_file(&path, MAX_JSON_DESCRIPTOR_BYTES + 1) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                pass.diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::Io,
                    Some(descriptor_path.clone()),
                    format!("failed to read JSON descriptor: {error}"),
                ));
                continue;
            }
        };
        match validate_json_descriptor(
            resolved,
            &path,
            descriptor_path,
            &snapshot.prefix,
            snapshot.size_bytes,
            &mut pass.diagnostics,
        ) {
            Ok(descriptor) => {
                if descriptor.contributes_backbone_family {
                    if let Some(family) = &descriptor.family {
                        pass.backbone_families.insert(family.clone());
                    }
                }
                if let Some(index) = descriptor.index {
                    indices.push(index);
                }
                let _retained = pass.retain_evidence(CheckpointArtifactEvidenceV1 {
                    relative_path: descriptor_path.clone(),
                    container: CheckpointContainerV1::JsonDescriptor,
                    role: Some("descriptor".to_owned()),
                    family: descriptor.family,
                    dialect: descriptor.dialect,
                    sha256: snapshot.sha256.clone(),
                    size_bytes: snapshot.size_bytes,
                    declared_tensor_bytes: None,
                    tensor_names: Vec::new(),
                });
            }
            Err(diagnostic) => pass.diagnostics.push(diagnostic),
        }
        snapshot.prefix = Vec::new();
        pass.observations.push(ArtifactObservation {
            relative_path: descriptor_path.clone(),
            container: CheckpointContainerV1::JsonDescriptor,
            canonical_path: path,
            snapshot,
        });
    }

    validate_safetensors_indices(
        &discovery.internal_paths,
        &indices,
        &candidate_by_path,
        &tensor_tables,
        &mut pass.diagnostics,
    );
    pass.evidence
        .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    pass.observations
        .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    sort_diagnostics(&mut pass.diagnostics);
    pass
}

fn validate_safetensors_indices(
    internal_paths: &BTreeMap<String, String>,
    indices: &[SafetensorsIndexDeclaration],
    candidates: &BTreeMap<String, &CheckpointCandidateV1>,
    tensor_tables: &BTreeMap<String, BTreeSet<String>>,
    diagnostics: &mut Vec<CheckpointDiagnosticV1>,
) {
    // These messages describe the checkpoint's INTERNAL structure — which shard an index claims,
    // which shard a tensor is in — so they name shards the way the checkpoint does, not the way the
    // library root happens to. The lookup keys stay root-relative because that is what the
    // candidate and tensor tables are keyed on; only what the user reads is re-expressed. A shard
    // discovery never recorded keeps its root-relative name rather than losing its identity: this
    // is a rendering, and it must never turn a refusal into a blank.
    let shown = |shard: &str| {
        internal_paths
            .get(shard)
            .cloned()
            .unwrap_or_else(|| shard.to_owned())
    };
    let mut all_references = BTreeSet::new();
    let mut shard_owner = BTreeMap::<String, String>::new();
    for index in indices {
        let referenced_shards = index.weight_map.values().cloned().collect::<BTreeSet<_>>();
        for shard in &referenced_shards {
            all_references.insert(shard.clone());
            if let Some(previous) = shard_owner.insert(shard.clone(), index.descriptor_path.clone())
            {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::IndexTensorMismatch,
                    Some(index.descriptor_path.clone()),
                    format!(
                        "safetensors shard '{}' is claimed by both '{previous}' and '{}'",
                        shown(shard),
                        index.descriptor_path
                    ),
                ));
            }
            match candidates.get(shard) {
                Some(candidate)
                    if candidate.container == CheckpointContainerV1::Safetensors
                        && tensor_tables.contains_key(shard) => {}
                Some(_) => diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::IndexTensorMismatch,
                    Some(index.descriptor_path.clone()),
                    format!(
                        "safetensors index shard '{}' did not yield a valid safetensors tensor table",
                        shown(shard)
                    ),
                )),
                None => diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::MissingSidecar,
                    Some(index.descriptor_path.clone()),
                    format!(
                        "safetensors index shard '{}' was not discovered as an importable weight artifact",
                        shown(shard)
                    ),
                )),
            }
        }

        let mut tensor_locations = BTreeMap::<String, Vec<String>>::new();
        for shard in &referenced_shards {
            if let Some(tensors) = tensor_tables.get(shard) {
                for tensor in tensors {
                    tensor_locations
                        .entry(tensor.clone())
                        .or_default()
                        .push(shard.clone());
                }
            }
        }
        for (tensor, locations) in &tensor_locations {
            if locations.len() > 1 {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::IndexTensorMismatch,
                    Some(index.descriptor_path.clone()),
                    format!(
                        "tensor '{tensor}' exists in multiple indexed shards: {}",
                        locations
                            .iter()
                            .map(|shard| shown(shard))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                ));
            }
        }

        for (tensor, shard) in &index.weight_map {
            if !tensor_tables
                .get(shard)
                .is_some_and(|tensors| tensors.contains(tensor))
            {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::IndexTensorMismatch,
                    Some(index.descriptor_path.clone()),
                    format!(
                        "weight_map tensor '{tensor}' does not exist in its declared shard '{}'",
                        shown(shard)
                    ),
                ));
            }
        }
        for shard in &referenced_shards {
            let expected = index
                .weight_map
                .iter()
                .filter(|(_, declared_shard)| *declared_shard == shard)
                .map(|(tensor, _)| tensor.clone())
                .collect::<BTreeSet<_>>();
            if let Some(actual) = tensor_tables.get(shard) {
                for tensor in actual.difference(&expected) {
                    diagnostics.push(CheckpointDiagnosticV1::error(
                        CheckpointDiagnosticCodeV1::IndexTensorMismatch,
                        Some(index.descriptor_path.clone()),
                        format!(
                            "shard '{}' tensor '{tensor}' is missing from weight_map",
                            shown(shard)
                        ),
                    ));
                }
                for tensor in expected.difference(actual) {
                    diagnostics.push(CheckpointDiagnosticV1::error(
                        CheckpointDiagnosticCodeV1::IndexTensorMismatch,
                        Some(index.descriptor_path.clone()),
                        format!(
                            "weight_map contains extra tensor '{tensor}' for shard '{}'",
                            shown(shard)
                        ),
                    ));
                }
            }
        }
    }

    for candidate in candidates.values() {
        if looks_like_shard(&candidate.relative_path)
            && !all_references.contains(&candidate.relative_path)
        {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MissingSidecar,
                Some(candidate.relative_path.clone()),
                "sharded safetensors file is not named by any *.safetensors.index.json sidecar",
            ));
        }
    }
}

fn compare_exact_byte_passes(
    first: &[ArtifactObservation],
    second: &[ArtifactObservation],
    diagnostics: &mut Vec<CheckpointDiagnosticV1>,
) {
    let first_by_path = first
        .iter()
        .map(|item| (item.relative_path.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    let second_by_path = second
        .iter()
        .map(|item| (item.relative_path.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    let paths = first_by_path
        .keys()
        .chain(second_by_path.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    for path in paths {
        let stable = match (first_by_path.get(path), second_by_path.get(path)) {
            (Some(first), Some(second)) => {
                first.container == second.container
                    && first.canonical_path == second.canonical_path
                    && first.snapshot.sha256 == second.snapshot.sha256
                    && first.snapshot.size_bytes == second.snapshot.size_bytes
                    && first.snapshot.stable
                    && second.snapshot.stable
                    && first.snapshot.after == second.snapshot.before
            }
            _ => false,
        };
        if !stable {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::SourceChangedDuringInspection,
                Some(path.to_owned()),
                "checkpoint artifact identity or exact bytes changed between verification passes; retry against a stationary source",
            ));
        }
    }
}

fn revalidate_artifact_observations(
    resolved: &ResolvedInput,
    observations: &[ArtifactObservation],
    diagnostics: &mut Vec<CheckpointDiagnosticV1>,
) {
    for observation in observations {
        let joined = resolved
            .canonical_root
            .join(portable_to_path(&observation.relative_path));
        let unchanged = std::fs::canonicalize(&joined)
            .ok()
            .filter(|path| path == &observation.canonical_path)
            .and_then(|path| File::open(path).ok())
            .and_then(|file| file.metadata().ok())
            .map(|metadata| FileStamp::from_metadata(&metadata))
            .is_some_and(|stamp| stamp == observation.snapshot.after);
        if !unchanged {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::SourceChangedDuringInspection,
                Some(observation.relative_path.clone()),
                "checkpoint artifact identity or metadata changed after its final exact-byte pass; retry against a stationary source",
            ));
        }
    }
}

fn compile_inventory(
    request: &CheckpointInspectionRequestV1,
    internal_paths: &BTreeMap<String, String>,
    family: &str,
    fingerprint: &str,
    evidence: &[CheckpointArtifactEvidenceV1],
) -> Result<(CheckpointInventoryV1, ImportPlanV1), String> {
    let mut layers = Vec::with_capacity(evidence.len());
    for artifact in evidence {
        // SEMANTIC identity is checkpoint-relative; the LOCATOR below stays root-relative. The
        // internal path is the one discovery walked to, so a checkpoint that reaches a file through
        // a symlink still names it by where it sits INSIDE the checkpoint. No fallback: an artifact
        // discovery never saw is a contract violation, not a path to guess at.
        let internal_path = internal_paths
            .get(&artifact.relative_path)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "artifact {:?} has no discovered checkpoint-relative path",
                    artifact.relative_path
                )
            })?;
        let source = match &request.ownership {
            CheckpointOwnershipV1::Linked { root_id } => SourceLocatorV1::linked(
                root_id.clone(),
                artifact.relative_path.clone(),
                artifact.sha256.clone(),
            ),
            CheckpointOwnershipV1::Managed {
                install_id,
                provenance,
            } => SourceLocatorV1::managed(
                install_id.clone(),
                artifact.relative_path.clone(),
                artifact.sha256.clone(),
                provenance.clone(),
            ),
        }
        .map_err(|error| error.to_string())?;
        // Same rule as the internal path above, for the same reason: a role is what every
        // downstream consumer selects a layer BY (`transformer`, `text_encoder`, `vae`,
        // `transformer_high`/`_low`, `descriptor`), so inventing an `"artifact"` role would mint a
        // layer no adapter can bind and hand it to the plan as if it were resolved.
        //
        // Unreachable by construction today — `reconcile_role` records an
        // `AmbiguousComponentRole` ERROR for both of its `None` outcomes, and `inspect_checkpoint`
        // returns before compilation when any error diagnostic is present
        // (`a_role_conflict_refuses_and_compiles_no_plan` pins that). This is the local refusal for
        // the day a new evidence producer forgets the diagnostic: a contract violation, not a role
        // to guess at.
        let role = artifact.role.clone().ok_or_else(|| {
            format!(
                "artifact {:?} reached plan compilation without a resolved component role",
                artifact.relative_path
            )
        })?;
        layers.push(ImportLayerV1 {
            layer_id: format!("artifact:{internal_path}"),
            role,
            target_path: internal_path,
            // The container this evidence row was VALIDATED as, carried straight through. This is
            // the only place the verdict is available without re-reading the file: discovery
            // classified the header, `validate_safetensors` / `validate_gguf` parsed it, and the
            // result landed on the evidence. Downstream consumers must never re-derive it from the
            // extension — a `.safetensors` name on GGUF bytes is precisely what the header check
            // exists to catch.
            container: artifact.container,
            source,
        });
    }
    let mut plan_id_hasher = Sha256::new();
    plan_id_hasher.update(b"sceneworks.checkpoint.plan-id.v1\0");
    plan_id_hasher.update(request.checkpoint_id.as_bytes());
    plan_id_hasher.update(b"\0");
    plan_id_hasher.update(fingerprint.as_bytes());
    let plan_id = format!("checkpoint-plan-{:x}", plan_id_hasher.finalize());
    let plan = ImportPlanV1::new(plan_id, family, layers).map_err(|error| error.to_string())?;
    let record = CheckpointCatalogRecordV1::from_plan(request.checkpoint_id.clone(), &plan)
        .map_err(|error| error.to_string())?;
    let inventory = CheckpointInventoryV1::new(vec![record]).map_err(|error| error.to_string())?;
    Ok((inventory, plan))
}

fn resolve_input(request: &CheckpointInspectionRequestV1) -> Option<ResolvedInput> {
    let canonical_root = std::fs::canonicalize(&request.root_path).ok()?;
    let canonical_target =
        std::fs::canonicalize(canonical_root.join(&request.relative_path)).ok()?;
    Some(ResolvedInput {
        canonical_root,
        canonical_target,
    })
}

fn resolve_confined_artifact(
    resolved: &ResolvedInput,
    relative_path: &str,
    diagnostics: &mut Vec<CheckpointDiagnosticV1>,
) -> Option<PathBuf> {
    let joined = resolved
        .canonical_root
        .join(portable_to_path(relative_path));
    let canonical = match std::fs::canonicalize(&joined) {
        Ok(canonical) => canonical,
        Err(error) => {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::SourceChangedDuringInspection,
                Some(relative_path.to_owned()),
                format!("checkpoint artifact disappeared during inspection: {error}"),
            ));
            return None;
        }
    };
    if !canonical.starts_with(&resolved.canonical_root) {
        diagnostics.push(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::PathEscapesRoot,
            Some(relative_path.to_owned()),
            "checkpoint artifact resolves outside its declared root",
        ));
        return None;
    }
    Some(canonical)
}

fn discover_direct_file(resolved: &ResolvedInput) -> DiscoveryFiles {
    let path = resolved.canonical_target.clone();
    // A single-file checkpoint's internal path is its own name: the directory it happens to sit in
    // is where the user keeps it, not part of what it is.
    //
    // A name that is not valid UTF-8 has no portable internal path, so it REFUSES here exactly as
    // the directory walk does for its entries. The previous `unwrap_or_default()` produced an EMPTY
    // internal path instead — which is not a name, is not what discovery walked to, and feeds the
    // checkpoint's semantic identity, so two differently-named non-UTF-8 checkpoints digested
    // identically.
    let Some(internal) = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
    else {
        return DiscoveryFiles {
            diagnostics: vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::PathEscapesRoot,
                relative_to_root(resolved, &path),
                "checkpoint entry name is not valid UTF-8 and has no portable internal path",
            )],
            ..DiscoveryFiles::default()
        };
    };
    let file = DiscoveredFile {
        canonical: path.clone(),
        internal,
    };
    if is_json_descriptor(&path) {
        DiscoveryFiles {
            descriptors: vec![file],
            ..DiscoveryFiles::default()
        }
    } else {
        DiscoveryFiles {
            weights: vec![file],
            ..DiscoveryFiles::default()
        }
    }
}

fn discover_directory(resolved: &ResolvedInput) -> DiscoveryFiles {
    let mut result = DiscoveryFiles::default();
    // The third element is the walk's own path INSIDE the checkpoint, carried down rather than
    // recomputed from the canonical path: a directory reached through a symlink canonicalizes to
    // wherever the link led, and its name inside the checkpoint is not recoverable from there.
    let mut stack = vec![(resolved.canonical_target.clone(), String::new(), 0_usize)];
    let mut seen_dirs = HashSet::new();
    let mut visited_entries = 0_usize;

    while let Some((directory, internal_prefix, depth)) = stack.pop() {
        let Ok(canonical_directory) = std::fs::canonicalize(&directory) else {
            continue;
        };
        if !canonical_directory.starts_with(&resolved.canonical_root)
            || !seen_dirs.insert(canonical_directory.clone())
        {
            continue;
        }
        let Ok(read_dir) = std::fs::read_dir(&canonical_directory) else {
            result.diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::Io,
                relative_to_root(resolved, &canonical_directory),
                "checkpoint directory could not be read",
            ));
            continue;
        };
        for entry in read_dir {
            if visited_entries >= MAX_DISCOVERY_ENTRIES {
                result.diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::DiscoveryLimitExceeded,
                    relative_to_root(resolved, &resolved.canonical_target),
                    format!(
                        "checkpoint discovery exceeded its {MAX_DISCOVERY_ENTRIES}-entry bound"
                    ),
                ));
                return result;
            }
            visited_entries += 1;
            let Ok(entry) = entry else {
                result.diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::Io,
                    relative_to_root(resolved, &canonical_directory),
                    "checkpoint directory entry could not be read",
                ));
                continue;
            };
            let path = entry.path();
            if is_hidden_path(&path) {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                result.diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::PathEscapesRoot,
                    relative_to_root(resolved, &path),
                    "checkpoint entry name is not valid UTF-8 and has no portable internal path",
                ));
                continue;
            };
            let internal = if internal_prefix.is_empty() {
                name
            } else {
                format!("{internal_prefix}/{name}")
            };
            let Ok(canonical) = std::fs::canonicalize(&path) else {
                continue;
            };
            if !canonical.starts_with(&resolved.canonical_root) {
                result.diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::PathEscapesRoot,
                    relative_to_root(resolved, &path),
                    "checkpoint entry resolves outside its declared root",
                ));
                continue;
            }
            if canonical.is_dir() {
                if depth < MAX_DISCOVERY_DEPTH {
                    stack.push((canonical, internal, depth + 1));
                } else {
                    result.diagnostics.push(CheckpointDiagnosticV1::error(
                        CheckpointDiagnosticCodeV1::DiscoveryLimitExceeded,
                        relative_to_root(resolved, &canonical),
                        format!(
                            "checkpoint discovery exceeded its depth bound of {MAX_DISCOVERY_DEPTH}"
                        ),
                    ));
                }
                continue;
            }
            if is_weight_extension(&canonical) {
                result.weights.push(DiscoveredFile {
                    canonical,
                    internal,
                });
            } else if is_json_descriptor(&canonical) {
                result.descriptors.push(DiscoveredFile {
                    canonical,
                    internal,
                });
            }
        }
    }
    result.weights.sort_by(|left, right| {
        (&left.canonical, &left.internal).cmp(&(&right.canonical, &right.internal))
    });
    result.descriptors.sort_by(|left, right| {
        (&left.canonical, &left.internal).cmp(&(&right.canonical, &right.internal))
    });
    result
}

fn discover_weight(
    resolved: &ResolvedInput,
    file: &DiscoveredFile,
) -> Result<CheckpointCandidateV1, CheckpointDiagnosticV1> {
    let path = file.canonical.as_path();
    let relative_path = relative_to_root(resolved, path).ok_or_else(|| {
        CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::PathEscapesRoot,
            None,
            "checkpoint candidate has no confined portable relative path",
        )
    })?;
    let size_bytes = std::fs::metadata(path)
        .map(|item| item.len())
        .map_err(|error| {
            CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::Io,
                Some(relative_path.clone()),
                format!("checkpoint candidate metadata could not be read: {error}"),
            )
        })?;
    let mut magic = [0_u8; 4];
    let is_gguf = File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .is_ok()
        && &magic == b"GGUF";
    if is_gguf {
        return Ok(CheckpointCandidateV1 {
            relative_path,
            container: CheckpointContainerV1::Gguf,
            size_bytes,
            header_role: None,
            header_family: None,
            quantization: Some("gguf".to_owned()),
        });
    }
    let header = discover_safetensors_header(path, size_bytes).map_err(|error| {
        CheckpointDiagnosticV1::error(error.code, Some(relative_path.clone()), error.message)
    })?;
    match classify_base_header(&header) {
        BaseWeightDetection::Recognized(verdict) => Ok(CheckpointCandidateV1 {
            relative_path,
            container: CheckpointContainerV1::Safetensors,
            size_bytes,
            header_role: Some(verdict.component.as_str().to_owned()),
            header_family: verdict.family,
            quantization: Some(verdict.quant.as_str().to_owned()),
        }),
        BaseWeightDetection::Unrecognized { .. } => Ok(CheckpointCandidateV1 {
            relative_path,
            container: CheckpointContainerV1::Safetensors,
            size_bytes,
            header_role: None,
            header_family: None,
            quantization: None,
        }),
    }
}

struct SafetensorsDiscoveryError {
    code: CheckpointDiagnosticCodeV1,
    message: String,
}

fn discover_safetensors_header(
    path: &Path,
    file_len: u64,
) -> Result<Value, SafetensorsDiscoveryError> {
    let mut file = File::open(path).map_err(|error| SafetensorsDiscoveryError {
        code: CheckpointDiagnosticCodeV1::Io,
        message: format!("safetensors candidate could not be opened: {error}"),
    })?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|_| SafetensorsDiscoveryError {
            code: CheckpointDiagnosticCodeV1::TruncatedHeader,
            message: "safetensors candidate is truncated before its header length".to_owned(),
        })?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(SafetensorsDiscoveryError {
            code: CheckpointDiagnosticCodeV1::HeaderTooLarge,
            message: format!(
                "safetensors header length {header_len} is outside the discovery bound"
            ),
        });
    }
    if 8_u64
        .checked_add(header_len)
        .is_none_or(|end| end > file_len)
    {
        return Err(SafetensorsDiscoveryError {
            code: CheckpointDiagnosticCodeV1::TruncatedHeader,
            message: "safetensors descriptor bytes are truncated".to_owned(),
        });
    }
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|_| SafetensorsDiscoveryError {
            code: CheckpointDiagnosticCodeV1::TruncatedHeader,
            message: "safetensors descriptor bytes are truncated".to_owned(),
        })?;
    serde_json::from_slice(&header).map_err(|error| SafetensorsDiscoveryError {
        code: CheckpointDiagnosticCodeV1::MalformedMetadata,
        message: format!("safetensors descriptor is malformed JSON: {error}"),
    })
}

#[derive(Default)]
struct ValidatedArtifact {
    role: Option<String>,
    family: Option<String>,
    dialect: Option<String>,
    declared_tensor_bytes: Option<u64>,
    tensor_names: Vec<String>,
}

fn validate_safetensors(
    prefix: &[u8],
    file_len: u64,
    relative_path: Option<String>,
) -> Result<ValidatedArtifact, Vec<CheckpointDiagnosticV1>> {
    let Some(header_len_bytes) = prefix.get(..8) else {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::TruncatedHeader,
            relative_path,
            "safetensors file is truncated before its 8-byte header length",
        )]);
    };
    let header_len = u64::from_le_bytes(header_len_bytes.try_into().unwrap());
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::HeaderTooLarge,
            relative_path,
            format!(
                "safetensors header length {header_len} is outside the 1..={MAX_SAFETENSORS_HEADER_BYTES} bound"
            ),
        )]);
    }
    let Some(data_start) = 8_u64.checked_add(header_len) else {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::InvalidTensorRange,
            relative_path,
            "safetensors header length overflows the file address space",
        )]);
    };
    if data_start > file_len {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::TruncatedHeader,
            relative_path,
            format!(
                "safetensors header ends at byte {data_start}, beyond the {file_len}-byte file"
            ),
        )]);
    }
    let Some(header) = prefix.get(8..data_start as usize) else {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::TruncatedHeader,
            relative_path,
            "safetensors descriptor bytes are truncated",
        )]);
    };
    let value = strict_json(header).map_err(|error| {
        vec![CheckpointDiagnosticV1::error(
            error.code,
            relative_path.clone(),
            format!("invalid safetensors descriptor: {}", error.message),
        )]
    })?;
    let Some(entries) = value.as_object() else {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MalformedMetadata,
            relative_path,
            "safetensors descriptor must be a JSON object",
        )]);
    };

    let mut diagnostics = Vec::new();
    if let Some(metadata) = entries.get("__metadata__") {
        match metadata.as_object() {
            Some(metadata) if metadata.values().all(Value::is_string) => {}
            _ => diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path.clone(),
                "safetensors __metadata__ must be an object containing only string values",
            )),
        }
    }
    let data_len = file_len - data_start;
    let mut ranges = Vec::new();
    for (name, tensor) in entries {
        if name == "__metadata__" {
            continue;
        }
        if name.is_empty() {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path.clone(),
                "safetensors tensor names must not be empty",
            ));
            continue;
        }
        let Some(tensor) = tensor.as_object() else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path.clone(),
                format!("tensor '{name}' descriptor must be an object"),
            ));
            continue;
        };
        let Some(bits_per_element) = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .and_then(safetensors_dtype_bits)
        else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::UnsupportedTensorType,
                relative_path.clone(),
                format!("tensor '{name}' has a missing or unsupported dtype"),
            ));
            continue;
        };
        let Some(shape) = tensor.get("shape").and_then(Value::as_array) else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path.clone(),
                format!("tensor '{name}' has no valid shape array"),
            ));
            continue;
        };
        let element_count = shape.iter().try_fold(1_u64, |count, dimension| {
            count.checked_mul(dimension.as_u64()?)
        });
        let Some(expected_bits) =
            element_count.and_then(|count| count.checked_mul(bits_per_element))
        else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!("tensor '{name}' shape overflows its declared byte size"),
            ));
            continue;
        };
        if expected_bits % 8 != 0 {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!("tensor '{name}' contains {expected_bits} bits, which is not byte-aligned"),
            ));
            continue;
        }
        let expected_bytes = expected_bits / 8;
        let Some(offsets) = tensor.get("data_offsets").and_then(Value::as_array) else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path.clone(),
                format!("tensor '{name}' has no valid data_offsets pair"),
            ));
            continue;
        };
        let [start, end] = offsets.as_slice() else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path.clone(),
                format!("tensor '{name}' data_offsets must contain exactly two integers"),
            ));
            continue;
        };
        let (Some(start), Some(end)) = (start.as_u64(), end.as_u64()) else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path.clone(),
                format!("tensor '{name}' data_offsets must be unsigned integers"),
            ));
            continue;
        };
        if start > end || end.saturating_sub(start) != expected_bytes {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!(
                    "tensor '{name}' range [{start}, {end}) does not match its {expected_bytes}-byte dtype/shape"
                ),
            ));
        } else if end > data_len {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::TruncatedData,
                relative_path.clone(),
                format!(
                    "tensor '{name}' ends at data byte {end}, beyond the {data_len}-byte tensor body"
                ),
            ));
        } else {
            ranges.push((start, end, name.clone()));
        }
    }
    if ranges.is_empty() {
        diagnostics.push(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MalformedMetadata,
            relative_path.clone(),
            "safetensors descriptor declares no valid tensors",
        ));
    } else {
        ranges.sort_by_key(|(start, _, _)| *start);
        let mut expected_start = 0_u64;
        for (start, end, name) in &ranges {
            if *start != expected_start {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::InvalidTensorRange,
                    relative_path.clone(),
                    format!(
                        "tensor '{name}' begins at {start}; expected contiguous offset {expected_start}"
                    ),
                ));
            }
            expected_start = *end;
        }
        if expected_start < data_len {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!(
                    "safetensors descriptor accounts for {expected_start} bytes but the tensor body contains {data_len}"
                ),
            ));
        }
    }
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }
    let (role, family, dialect) = match classify_base_header(&value) {
        BaseWeightDetection::Recognized(verdict) => (
            Some(verdict.component.as_str().to_owned()),
            verdict.family,
            Some(verdict.quant.as_str().to_owned()),
        ),
        BaseWeightDetection::Unrecognized { .. } => (None, None, None),
    };
    let mut tensor_names = entries
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect::<Vec<_>>();
    tensor_names.sort();
    Ok(ValidatedArtifact {
        role,
        family,
        dialect,
        declared_tensor_bytes: Some(data_len),
        tensor_names,
    })
}

fn safetensors_dtype_bits(dtype: &str) -> Option<u64> {
    match dtype {
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Some(8),
        "U16" | "I16" | "F16" | "BF16" => Some(16),
        "U32" | "I32" | "F32" => Some(32),
        "U64" | "I64" | "F64" | "C64" => Some(64),
        _ => None,
    }
}

#[derive(Default)]
struct JsonDescriptorEvidence {
    family: Option<String>,
    dialect: Option<String>,
    contributes_backbone_family: bool,
    index: Option<SafetensorsIndexDeclaration>,
}

struct SafetensorsIndexDeclaration {
    descriptor_path: String,
    weight_map: BTreeMap<String, String>,
}

fn validate_json_descriptor(
    resolved: &ResolvedInput,
    path: &Path,
    relative_path: &str,
    raw: &[u8],
    size_bytes: u64,
    diagnostics: &mut Vec<CheckpointDiagnosticV1>,
) -> Result<JsonDescriptorEvidence, CheckpointDiagnosticV1> {
    if size_bytes > MAX_JSON_DESCRIPTOR_BYTES {
        return Err(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::HeaderTooLarge,
            Some(relative_path.to_owned()),
            format!(
                "JSON descriptor is {} bytes, above the {MAX_JSON_DESCRIPTOR_BYTES}-byte bound",
                size_bytes
            ),
        ));
    }
    let value = strict_json(raw).map_err(|error| {
        CheckpointDiagnosticV1::error(
            error.code,
            Some(relative_path.to_owned()),
            format!("invalid JSON descriptor: {}", error.message),
        )
    })?;
    let Some(object) = value.as_object() else {
        return Err(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MalformedMetadata,
            Some(relative_path.to_owned()),
            "JSON descriptor must contain an object at its root",
        ));
    };

    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let mut index = None;
    if filename.ends_with(".safetensors.index.json") {
        let Some(weight_map) = object.get("weight_map").and_then(Value::as_object) else {
            return Err(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                Some(relative_path.to_owned()),
                "safetensors index descriptor must contain an object weight_map",
            ));
        };
        if weight_map.is_empty() {
            return Err(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                Some(relative_path.to_owned()),
                "safetensors index weight_map must not be empty",
            ));
        }
        let mut canonical_weight_map = BTreeMap::new();
        for (tensor, shard) in weight_map {
            if tensor.is_empty() {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::MalformedMetadata,
                    Some(relative_path.to_owned()),
                    "safetensors index tensor names must not be empty",
                ));
                continue;
            }
            let Some(shard) = shard.as_str() else {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::MalformedMetadata,
                    Some(relative_path.to_owned()),
                    format!("safetensors index entry '{tensor}' must name a shard string"),
                ));
                continue;
            };
            let shard_path = Path::new(shard);
            if !safe_relative_path(shard_path) {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::PathEscapesRoot,
                    Some(relative_path.to_owned()),
                    format!("safetensors index entry '{tensor}' uses unsafe shard path '{shard}'"),
                ));
                continue;
            }
            if !shard_path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
            {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::MalformedMetadata,
                    Some(relative_path.to_owned()),
                    format!(
                        "safetensors index entry '{tensor}' must reference a .safetensors shard"
                    ),
                ));
                continue;
            }
            let absolute = path
                .parent()
                .unwrap_or(&resolved.canonical_target)
                .join(shard_path);
            if !absolute.is_file() {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::MissingSidecar,
                    Some(relative_path.to_owned()),
                    format!("safetensors index references missing shard '{shard}'"),
                ));
            } else if let Ok(canonical) = std::fs::canonicalize(&absolute) {
                if !canonical.starts_with(&resolved.canonical_root) {
                    diagnostics.push(CheckpointDiagnosticV1::error(
                        CheckpointDiagnosticCodeV1::PathEscapesRoot,
                        Some(relative_path.to_owned()),
                        format!(
                            "safetensors index shard '{shard}' resolves outside the declared root"
                        ),
                    ));
                } else if let Some(canonical_relative) = relative_to_root(resolved, &canonical) {
                    canonical_weight_map.insert(tensor.clone(), canonical_relative);
                }
            } else {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::Io,
                    Some(relative_path.to_owned()),
                    format!("safetensors index shard '{shard}' could not be resolved"),
                ));
            }
        }
        index = Some(SafetensorsIndexDeclaration {
            descriptor_path: relative_path.to_owned(),
            weight_map: canonical_weight_map,
        });
    }

    if filename == "model_index.json" {
        for (component_name, declaration) in object {
            if component_name.starts_with('_') || declaration.is_null() {
                continue;
            }
            let is_component = declaration
                .as_array()
                .is_some_and(|values| values.len() >= 2 && values[0].is_string());
            if !is_component {
                continue;
            }
            if !safe_relative_path(Path::new(component_name)) {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::PathEscapesRoot,
                    Some(relative_path.to_owned()),
                    format!("model_index.json declares unsafe component path '{component_name}'"),
                ));
                continue;
            }
            let component_path = path
                .parent()
                .unwrap_or(&resolved.canonical_target)
                .join(component_name);
            if !component_path.is_dir() {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::MissingSidecar,
                    Some(relative_path.to_owned()),
                    format!(
                        "model_index.json declares component '{component_name}' but its directory is missing"
                    ),
                ));
            } else if std::fs::canonicalize(&component_path)
                .ok()
                .is_none_or(|canonical| !canonical.starts_with(&resolved.canonical_root))
            {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::PathEscapesRoot,
                    Some(relative_path.to_owned()),
                    format!(
                        "model_index.json component '{component_name}' resolves outside the declared root"
                    ),
                ));
            } else if !component_path.join("config.json").is_file() {
                diagnostics.push(CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::MissingSidecar,
                    Some(relative_path.to_owned()),
                    format!("model_index.json component '{component_name}' is missing config.json"),
                ));
            }
        }
    }

    let mut dialects = Vec::new();
    for key in ["_class_name", "model_type", "architecture", "architectures"] {
        if let Some(value) = object.get(key) {
            collect_strings(value, &mut dialects);
        }
    }
    dialects.sort();
    dialects.dedup();
    let families = dialects
        .iter()
        .filter_map(|value| normalize_family(value))
        .collect::<BTreeSet<_>>();
    if families.len() > 1 {
        return Err(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::FamilyDialectConflict,
            Some(relative_path.to_owned()),
            format!(
                "JSON descriptor contains conflicting model-family evidence: {}",
                families.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
    }
    let family = families.into_iter().next();
    let relative = portable_to_path(relative_path);
    let contributes_backbone_family = filename == "model_index.json"
        || infer_role_from_path(relative.parent().unwrap_or(Path::new("")))
            .is_some_and(|role| matches!(role, "transformer" | "checkpoint"));
    Ok(JsonDescriptorEvidence {
        family,
        dialect: (!dialects.is_empty()).then(|| dialects.join(",")),
        contributes_backbone_family,
        index,
    })
}

fn collect_strings(value: &Value, output: &mut Vec<String>) {
    if let Some(value) = value.as_str() {
        output.push(value.to_owned());
    } else if let Some(values) = value.as_array() {
        for value in values {
            collect_strings(value, output);
        }
    }
}

fn normalize_family(value: &str) -> Option<String> {
    let lower = value.to_ascii_lowercase();
    for (needle, family) in [
        ("stable diffusion xl", "sdxl"),
        ("stablediffusionxl", "sdxl"),
        ("sdxl", "sdxl"),
        ("qwenimage", "qwen-image"),
        ("qwen_image", "qwen-image"),
        ("qwen-image", "qwen-image"),
        ("flux2", "flux2"),
        ("flux.2", "flux2"),
        ("flux", "flux"),
        ("wan", "wan-video"),
        ("z_image", "z-image"),
        ("z-image", "z-image"),
        ("mageflow", "mage-flow"),
        ("mage_flow", "mage-flow"),
        ("mage-flow", "mage-flow"),
        ("krea", "krea_2"),
        ("ltx", "ltx-video"),
        ("ideogram", "ideogram"),
        ("anima", "anima"),
        ("cosmos", "anima"),
    ] {
        if lower.contains(needle) {
            return Some(family.to_owned());
        }
    }
    None
}

fn is_auxiliary_family(family: &str) -> bool {
    matches!(family, "t5" | "qwen3" | "gemma")
}

fn reconcile_role(
    relative_path: &str,
    path_role: Option<&'static str>,
    header_role: Option<&str>,
    diagnostics: &mut Vec<CheckpointDiagnosticV1>,
) -> Option<String> {
    match (path_role, header_role) {
        (Some(path_role), Some(header_role)) if path_role != header_role => {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::AmbiguousComponentRole,
                Some(relative_path.to_owned()),
                format!(
                    "component path implies role '{path_role}' but its descriptor implies '{header_role}'"
                ),
            ));
            None
        }
        (Some(role), _) => Some(role.to_owned()),
        (_, Some(role)) => Some(role.to_owned()),
        (None, None) => {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::AmbiguousComponentRole,
                Some(relative_path.to_owned()),
                "component role is ambiguous: neither its path nor descriptor identifies transformer, text encoder, VAE, or fused checkpoint",
            ));
            None
        }
    }
}

/// The two plan-layer roles a MULTI-EXPERT backbone compiles to (epic 20398, sc-20644).
///
/// Underscored, like every other layer role this module emits (`transformer`, `text_encoder`,
/// `vae`, `checkpoint`) and like the `components[]` role spellings the catalog already uses for Wan.
/// The checkpoint ADAPTER's `component_topology` spells the same two roles with hyphens
/// (`transformer-high` / `transformer-low`), exactly as it spells `base-snapshot` for the
/// underscored `base_snapshot` component id — that split is pre-existing convention, not drift.
pub const TRANSFORMER_HIGH_ROLE: &str = "transformer_high";
pub const TRANSFORMER_LOW_ROLE: &str = "transformer_low";

/// Families whose backbone is TWO non-interchangeable experts rather than one transformer.
///
/// Checkpoint truth, and deliberately a closed list: Wan 2.2 is the only shipped family whose
/// ComfyUI distribution carries a high-noise and a low-noise expert selected per denoise step. The
/// list is what makes [`refine_multi_expert_role`] provably ADDITIVE — a checkpoint of any other
/// family is never even considered for the refinement, so no existing family's compiled plan can
/// change, whatever its files happen to be called.
const MULTI_EXPERT_FAMILIES: &[&str] = &["wan-video"];

/// Refine a `transformer` role into a specific EXPERT role, for a family that has more than one
/// backbone.
///
/// Wan 2.2's two experts are not interchangeable — they are selected per denoise step — so a plan
/// that recorded them as two `transformer` layers could not say which is which, and the routes that
/// consume a plan can only refuse it as an ambiguous primary. This is the vocabulary that lets a
/// plan name them.
///
/// **Additive by construction, in three independent ways.** The refinement applies only when (1) the
/// artifact's detected family is in [`MULTI_EXPERT_FAMILIES`], (2) the reconciled role is already
/// exactly `transformer`, and (3) the file name carries a delimited high/low-noise marker. A
/// single-expert Wan checkpoint (2.1, or a merged 2.2) keeps the plain `transformer` role, and no
/// other family reaches the marker test at all. The goldens prove the second half of that claim.
fn refine_multi_expert_role(
    role: Option<String>,
    family: Option<&str>,
    relative_path: &str,
) -> Option<String> {
    if role.as_deref() != Some("transformer") {
        return role;
    }
    if !family.is_some_and(|family| MULTI_EXPERT_FAMILIES.contains(&family)) {
        return role;
    }
    match expert_marker(relative_path) {
        Some(expert) => Some(expert.to_owned()),
        None => role,
    }
}

/// The expert a ComfyUI Wan file name declares, or `None`.
///
/// The three spellings the shipped redistributions actually use, all of which must be recognized:
///
/// | publisher | example | normalized token |
/// |---|---|---|
/// | ComfyUI | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | `_high_noise_` |
/// | QuantStack GGUF | `Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf` | `_highnoise_` |
/// | Kijai scaled-fp8 | `Wan2_2_T2V_HIGH_14B_fp8_e4m3fn_scaled.safetensors` | `_high_` |
///
/// The first form alone was matched originally, so a QuantStack or Kijai pair kept the plain
/// `transformer` role — two anonymous backbones, a missing-expert-role refusal, and (before the
/// video router propagated) a procedural stub render (sc-20644 review major 5).
///
/// Matching stays DELIMITED rather than substring: `_high_` matches, `_highest_` does not, and
/// `wan_highlights_14B` cannot be reclassified. A name carrying BOTH markers is still refused
/// (`None`) rather than resolved to whichever appears first — an ambiguous name is not evidence,
/// and a merged single-file `high_noise_and_low_noise` checkpoint is exactly that case.
fn expert_marker(relative_path: &str) -> Option<&'static str> {
    let name = Path::new(relative_path)
        .file_name()
        .and_then(|name| name.to_str())?
        .to_ascii_lowercase();
    // Normalize the separators these publishers use interchangeably, then match tokens bounded on
    // both sides.
    let normalized = format!("_{}_", name.replace(['-', '.', ' '], "_"));
    let marks = |side: &str| {
        // `_high_` / `_highnoise_` / `_high_noise_`, and the `low` equivalents. The optional
        // `noise` suffix is what covers QuantStack's concatenation and Kijai's omission without
        // loosening the boundary on either end.
        normalized.contains(&format!("_{side}_"))
            || normalized.contains(&format!("_{side}noise_"))
            || normalized.contains(&format!("_{side}_noise_"))
    };
    match (marks("high"), marks("low")) {
        (true, false) => Some(TRANSFORMER_HIGH_ROLE),
        (false, true) => Some(TRANSFORMER_LOW_ROLE),
        _ => None,
    }
}

fn infer_role_from_path(path: &Path) -> Option<&'static str> {
    let names = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(str::to_ascii_lowercase),
            _ => None,
        })
        .collect::<Vec<_>>();
    for name in names.iter().rev() {
        match name.as_str() {
            "transformer" | "transformers" | "unet" | "diffusion_models" => {
                return Some("transformer")
            }
            "text_encoder" | "text_encoders" | "text_encoder_2" => return Some("text_encoder"),
            "vae" => return Some("vae"),
            "checkpoints" => return Some("checkpoint"),
            _ => {}
        }
    }
    None
}

fn inspection_fingerprint(evidence: &[CheckpointArtifactEvidenceV1]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sceneworks.checkpoint.inspection.v1\0");
    for artifact in evidence {
        hasher.update((artifact.relative_path.len() as u64).to_le_bytes());
        hasher.update(artifact.relative_path.as_bytes());
        hasher.update([0]);
        hasher.update(artifact.sha256.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArtifactSnapshot {
    sha256: String,
    size_bytes: u64,
    prefix: Vec<u8>,
    before: FileStamp,
    after: FileStamp,
    stable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    modified: Option<std::time::SystemTime>,
    created: Option<std::time::SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    change_seconds: i64,
    #[cfg(unix)]
    change_nanoseconds: i64,
    #[cfg(windows)]
    creation_time: u64,
    #[cfg(windows)]
    last_write_time: u64,
}

impl FileStamp {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt as _;
        #[cfg(windows)]
        use std::os::windows::fs::MetadataExt as _;

        Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            created: metadata.created().ok(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            change_seconds: metadata.ctime(),
            #[cfg(unix)]
            change_nanoseconds: metadata.ctime_nsec(),
            #[cfg(windows)]
            creation_time: metadata.creation_time(),
            #[cfg(windows)]
            last_write_time: metadata.last_write_time(),
        }
    }
}

fn snapshot_file(path: &Path, prefix_limit: u64) -> std::io::Result<ArtifactSnapshot> {
    let mut file = File::open(path)?;
    let before = FileStamp::from_metadata(&file.metadata()?);
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut prefix = Vec::new();
    let mut size_bytes = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size_bytes = size_bytes.checked_add(read as u64).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "artifact size overflows u64",
            )
        })?;
        let remaining = prefix_limit.saturating_sub(prefix.len() as u64) as usize;
        prefix.extend_from_slice(&buffer[..read.min(remaining)]);
    }
    let after = FileStamp::from_metadata(&file.metadata()?);
    let stable = before == after && size_bytes == after.len;
    Ok(ArtifactSnapshot {
        sha256: format!("{:x}", hasher.finalize()),
        size_bytes,
        prefix,
        before,
        after,
        stable,
    })
}

fn empty_inventory() -> CheckpointInventoryV1 {
    CheckpointInventoryV1::new(Vec::new()).expect("empty v1 inventory is valid")
}

/// One rule set, defined once in the plan store, for every confined relative path in this seam.
///
/// The path is rendered to its portable `/`-separated form first so this is platform-correct on
/// Windows, where a native `PathBuf` separates with `\`.
fn safe_relative_path(path: &Path) -> bool {
    path_to_portable_string(path).is_some_and(|value| portable_relative_path_parts(&value).is_ok())
}

fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

fn is_weight_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("safetensors") || extension.eq_ignore_ascii_case("gguf")
        })
}

fn is_json_descriptor(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

fn looks_like_shard(relative_path: &str) -> bool {
    let lower = relative_path.to_ascii_lowercase();
    lower.ends_with(".safetensors")
        && lower.rsplit('/').next().is_some_and(|name| {
            let Some((left, right)) = name.rsplit_once("-of-") else {
                return false;
            };
            left.rsplit_once('-').is_some_and(|(_, index)| {
                !index.is_empty()
                    && index.chars().all(|character| character.is_ascii_digit())
                    && right.strip_suffix(".safetensors").is_some_and(|count| {
                        !count.is_empty()
                            && count.chars().all(|character| character.is_ascii_digit())
                    })
            })
        })
}

fn path_to_portable_string(path: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        let Component::Normal(value) = component else {
            return None;
        };
        parts.push(value.to_str()?.to_owned());
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

fn relative_to_root(resolved: &ResolvedInput, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(&resolved.canonical_root).ok()?;
    path_to_portable_string(relative)
}

fn portable_to_path(path: &str) -> PathBuf {
    path.split('/').collect()
}

fn sort_diagnostics(diagnostics: &mut Vec<CheckpointDiagnosticV1>) {
    diagnostics.sort_by(|left, right| {
        (
            left.severity,
            left.code,
            left.relative_path.as_deref(),
            left.message.as_str(),
        )
            .cmp(&(
                right.severity,
                right.code,
                right.relative_path.as_deref(),
                right.message.as_str(),
            ))
    });
    diagnostics.dedup();
}

struct StrictJsonError {
    code: CheckpointDiagnosticCodeV1,
    message: String,
}

fn strict_json(raw: &[u8]) -> Result<Value, StrictJsonError> {
    serde_json::from_slice::<StrictValue>(raw)
        .map(|value| value.0)
        .map_err(|error| {
            let message = error.to_string();
            let code = if message.contains("duplicate key") {
                CheckpointDiagnosticCodeV1::DuplicateKey
            } else {
                CheckpointDiagnosticCodeV1::MalformedMetadata
            };
            StrictJsonError { code, message }
        })
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .map(StrictValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value.to_owned())))
    }

    fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictValue>()? {
            values.push(value.0);
        }
        Ok(StrictValue(Value::Array(values)))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom(format!("duplicate key '{key}'")));
            }
            let value = map.next_value::<StrictValue>()?;
            values.insert(key, value.0);
        }
        Ok(StrictValue(Value::Object(values)))
    }
}

#[derive(Clone)]
enum GgufMetadataValue {
    Uint32(u32),
    String(String),
    Other,
}

#[derive(Clone)]
struct GgufTensorInfo {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

fn validate_gguf(
    prefix: &[u8],
    file_len: u64,
    relative_path: Option<String>,
) -> Result<ValidatedArtifact, Vec<CheckpointDiagnosticV1>> {
    let mut reader = ByteReader::new(prefix);
    if reader.take(4) != Some(b"GGUF".as_slice()) {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MalformedMetadata,
            relative_path,
            "GGUF container does not start with GGUF magic",
        )]);
    }
    let version = reader.u32().ok_or_else(|| {
        vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::TruncatedHeader,
            relative_path.clone(),
            "GGUF descriptor is truncated before its version",
        )]
    })?;
    if !matches!(version, 2 | 3) {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MalformedMetadata,
            relative_path,
            format!("unsupported GGUF version {version}; expected v2 or v3"),
        )]);
    }
    let tensor_count = reader.u64().ok_or_else(|| {
        vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::TruncatedHeader,
            relative_path.clone(),
            "GGUF descriptor is truncated before its tensor count",
        )]
    })?;
    let metadata_count = reader.u64().ok_or_else(|| {
        vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::TruncatedHeader,
            relative_path.clone(),
            "GGUF descriptor is truncated before its metadata count",
        )]
    })?;
    if tensor_count > MAX_GGUF_ITEMS || metadata_count > MAX_GGUF_ITEMS {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::HeaderTooLarge,
            relative_path,
            format!(
                "GGUF declares {tensor_count} tensors and {metadata_count} metadata entries; each is bounded at {MAX_GGUF_ITEMS}"
            ),
        )]);
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.string().map_err(|message| {
            vec![gguf_parse_diagnostic(
                relative_path.clone(),
                message,
                file_len,
                prefix.len(),
            )]
        })?;
        if !valid_gguf_metadata_key(&key) {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path,
                format!(
                    "GGUF metadata key '{key}' must be <=65535 ASCII bytes with lower_snake_case segments separated by dots"
                ),
            )]);
        }
        if metadata.contains_key(&key) {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::DuplicateKey,
                relative_path,
                format!("GGUF metadata contains duplicate key '{key}'"),
            )]);
        }
        let value_type = reader.u32().ok_or_else(|| {
            vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::TruncatedHeader,
                relative_path.clone(),
                format!("GGUF metadata key '{key}' has no value type"),
            )]
        })?;
        let value = reader.metadata_value(value_type, 0).map_err(|message| {
            vec![gguf_parse_diagnostic(
                relative_path.clone(),
                message,
                file_len,
                prefix.len(),
            )]
        })?;
        metadata.insert(key, value);
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    let mut tensor_names = BTreeSet::new();
    for _ in 0..tensor_count {
        let name = reader.string().map_err(|message| {
            vec![gguf_parse_diagnostic(
                relative_path.clone(),
                message,
                file_len,
                prefix.len(),
            )]
        })?;
        if name.is_empty() || name.len() > 64 {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path,
                format!("GGUF tensor name must contain 1..=64 UTF-8 bytes, found {name:?}"),
            )]);
        }
        if !tensor_names.insert(name.clone()) {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::DuplicateKey,
                relative_path,
                format!("GGUF descriptor contains duplicate tensor name '{name}'"),
            )]);
        }
        let dimensions = reader.u32().ok_or_else(|| {
            vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::TruncatedHeader,
                relative_path.clone(),
                format!("GGUF tensor '{name}' is missing its dimension count"),
            )]
        })?;
        if dimensions == 0 || dimensions > 4 {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path,
                format!("GGUF tensor '{name}' has invalid dimension count {dimensions}"),
            )]);
        }
        let mut shape = Vec::with_capacity(dimensions as usize);
        for _ in 0..dimensions {
            shape.push(reader.u64().ok_or_else(|| {
                vec![CheckpointDiagnosticV1::error(
                    CheckpointDiagnosticCodeV1::TruncatedHeader,
                    relative_path.clone(),
                    format!("GGUF tensor '{name}' has a truncated shape"),
                )]
            })?);
        }
        let tensor_type = reader.u32().ok_or_else(|| {
            vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::TruncatedHeader,
                relative_path.clone(),
                format!("GGUF tensor '{name}' has no tensor type"),
            )]
        })?;
        let offset = reader.u64().ok_or_else(|| {
            vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::TruncatedHeader,
                relative_path.clone(),
                format!("GGUF tensor '{name}' has no data offset"),
            )]
        })?;
        tensors.push(GgufTensorInfo {
            name,
            dimensions: shape,
            tensor_type,
            offset,
        });
    }
    let alignment = match metadata.get("general.alignment") {
        Some(GgufMetadataValue::Uint32(value)) => u64::from(*value),
        Some(_) => {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path,
                "GGUF general.alignment must have metadata type uint32",
            )])
        }
        None => 32,
    };
    if alignment < 8 || alignment % 8 != 0 {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MalformedMetadata,
            relative_path,
            format!("GGUF alignment {alignment} must be a non-zero uint32 multiple of 8"),
        )]);
    }
    let has_quantized_tensors = tensors.iter().any(|tensor| {
        ggml_type_layout(tensor.tensor_type).is_some_and(|(block_size, _)| block_size > 1)
    });
    match metadata.get("general.quantization_version") {
        Some(GgufMetadataValue::Uint32(version)) if matches!(*version, 1 | 2) => {}
        Some(GgufMetadataValue::Uint32(version)) => {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path,
                format!(
                    "GGUF general.quantization_version {version} is unsupported; expected 1 or 2"
                ),
            )])
        }
        Some(_) => {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path,
                "GGUF general.quantization_version must have metadata type uint32",
            )])
        }
        None if has_quantized_tensors => {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MalformedMetadata,
                relative_path,
                "quantized GGUF tensors require uint32 general.quantization_version metadata",
            )])
        }
        None => {}
    }
    let descriptor_end = reader.position() as u64;
    let padding = (alignment - (descriptor_end % alignment)) % alignment;
    let data_start = descriptor_end.checked_add(padding).ok_or_else(|| {
        vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::InvalidTensorRange,
            relative_path.clone(),
            "GGUF descriptor alignment overflows the file address space",
        )]
    })?;
    if data_start > MAX_GGUF_DESCRIPTOR_BYTES {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::HeaderTooLarge,
            relative_path,
            format!("GGUF descriptor ends at byte {data_start}, above the bound"),
        )]);
    }
    if data_start > file_len {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::TruncatedHeader,
            relative_path,
            format!("GGUF tensor data begins at {data_start}, beyond the {file_len}-byte file"),
        )]);
    }
    let data_len = file_len - data_start;
    let mut ranges = Vec::new();
    let mut diagnostics = Vec::new();
    for tensor in tensors {
        let Some((block_size, type_size)) = ggml_type_layout(tensor.tensor_type) else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::UnsupportedTensorType,
                relative_path.clone(),
                format!(
                    "GGUF tensor '{}' uses unsupported ggml tensor type {}",
                    tensor.name, tensor.tensor_type
                ),
            ));
            continue;
        };
        let elements = tensor
            .dimensions
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension));
        let Some(elements) = elements else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!("GGUF tensor '{}' dimensions overflow", tensor.name),
            ));
            continue;
        };
        let first_dimension = tensor.dimensions.first().copied().unwrap_or(0);
        if elements == 0 || first_dimension % block_size != 0 {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!(
                    "GGUF tensor '{}' has first dimension {first_dimension}, incompatible with block size {block_size}",
                    tensor.name,
                ),
            ));
            continue;
        }
        let byte_len = (elements / block_size).checked_mul(type_size);
        let end = byte_len.and_then(|length| tensor.offset.checked_add(length));
        let Some(end) = end else {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!("GGUF tensor '{}' range overflows u64", tensor.name),
            ));
            continue;
        };
        if tensor.offset % alignment != 0 {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!(
                    "GGUF tensor '{}' offset {} is not aligned to {alignment}",
                    tensor.name, tensor.offset
                ),
            ));
        }
        if end > data_len {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!(
                    "GGUF tensor '{}' range [{}, {end}) exceeds the {data_len}-byte data section",
                    tensor.name, tensor.offset
                ),
            ));
        }
        ranges.push((tensor.offset, end, tensor.name));
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = 0_u64;
    for (start, end, name) in &ranges {
        if *start < previous_end {
            diagnostics.push(CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::InvalidTensorRange,
                relative_path.clone(),
                format!("GGUF tensor '{name}' overlaps the preceding tensor range"),
            ));
        }
        previous_end = previous_end.max(*end);
    }
    if ranges.is_empty() {
        diagnostics.push(CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MalformedMetadata,
            relative_path.clone(),
            "GGUF descriptor declares no valid tensors",
        ));
    }
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(value))
            if !value.is_empty()
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit()) =>
        {
            value.clone()
        }
        _ => {
            return Err(vec![CheckpointDiagnosticV1::error(
                CheckpointDiagnosticCodeV1::MissingFamilyEvidence,
                relative_path,
                "GGUF general.architecture must be a non-empty lowercase ASCII [a-z0-9]+ string",
            )])
        }
    };
    // Fail CLOSED on an architecture nothing maps, exactly the way the safetensors/JSON path does:
    // there, `normalize_family` returning `None` leaves the checkpoint with no backbone family and
    // the inspection refuses with `MissingFamilyEvidence`. Falling back to the raw architecture
    // string instead invented a family name no adapter has ever heard of, so the checkpoint
    // inspected as Ready, was offered as selectable, and only failed at render time.
    let Some(family) = normalize_family(&architecture) else {
        return Err(vec![CheckpointDiagnosticV1::error(
            CheckpointDiagnosticCodeV1::MissingFamilyEvidence,
            relative_path,
            format!(
                "GGUF general.architecture '{architecture}' is not a model family this build can load"
            ),
        )]);
    };
    Ok(ValidatedArtifact {
        role: None,
        family: Some(family),
        dialect: Some(format!("gguf-v{version}")),
        declared_tensor_bytes: Some(previous_end),
        tensor_names: tensor_names.into_iter().collect(),
    })
}

fn valid_gguf_metadata_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 65_535
        && key.is_ascii()
        && key.split('.').all(|segment| {
            !segment.is_empty()
                && !segment.starts_with('_')
                && !segment.ends_with('_')
                && !segment.contains("__")
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        })
}

fn gguf_parse_diagnostic(
    relative_path: Option<String>,
    message: String,
    file_len: u64,
    captured_bytes: usize,
) -> CheckpointDiagnosticV1 {
    let code = if message.contains("exceeds")
        || (message.contains("truncated") && file_len > captured_bytes as u64)
    {
        CheckpointDiagnosticCodeV1::HeaderTooLarge
    } else if message.contains("truncated") {
        CheckpointDiagnosticCodeV1::TruncatedHeader
    } else {
        CheckpointDiagnosticCodeV1::MalformedMetadata
    };
    CheckpointDiagnosticV1::error(
        code,
        relative_path,
        format!("invalid GGUF descriptor: {message}"),
    )
}

fn ggml_type_layout(tensor_type: u32) -> Option<(u64, u64)> {
    match tensor_type {
        0 => Some((1, 4)),   // F32
        1 => Some((1, 2)),   // F16
        2 => Some((32, 18)), // Q4_0
        3 => Some((32, 20)), // Q4_1
        6 => Some((32, 22)), // Q5_0
        7 => Some((32, 24)), // Q5_1
        8 => Some((32, 34)), // Q8_0
        9 => Some((32, 36)), // Q8_1
        10 => Some((256, 84)),
        11 => Some((256, 110)),
        12 => Some((256, 144)),
        13 => Some((256, 176)),
        14 => Some((256, 210)),
        15 => Some((256, 292)),
        16 => Some((256, 66)),
        17 => Some((256, 74)),
        18 => Some((256, 98)),
        19 => Some((256, 50)),
        20 => Some((32, 18)),
        21 => Some((256, 110)),
        22 => Some((256, 82)),
        23 => Some((256, 136)),
        24 => Some((1, 1)), // I8
        25 => Some((1, 2)), // I16
        26 => Some((1, 4)), // I32
        27 => Some((1, 8)), // I64
        28 => Some((1, 8)), // F64
        29 => Some((256, 56)),
        30 => Some((1, 2)),    // BF16
        34 => Some((256, 54)), // TQ1_0
        35 => Some((256, 66)), // TQ2_0
        39 => Some((32, 17)),  // MXFP4
        40 => Some((64, 36)),  // NVFP4
        41 => Some((128, 18)), // Q1_0
        42 => Some((64, 18)),  // Q2_0
        _ => None,
    }
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.position.checked_add(count)?;
        let value = self.bytes.get(self.position..end)?;
        self.position = end;
        Some(value)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(*self.take(1)?.first()?)
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn string(&mut self) -> Result<String, String> {
        let length = self
            .u64()
            .ok_or_else(|| "truncated string length".to_owned())?;
        if length > MAX_GGUF_STRING_BYTES {
            return Err(format!(
                "string length {length} exceeds the {MAX_GGUF_STRING_BYTES}-byte bound"
            ));
        }
        let bytes = self
            .take(length as usize)
            .ok_or_else(|| format!("truncated {length}-byte string"))?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| "string is not valid UTF-8".to_owned())
    }

    fn metadata_value(
        &mut self,
        value_type: u32,
        depth: usize,
    ) -> Result<GgufMetadataValue, String> {
        if depth > MAX_GGUF_ARRAY_DEPTH {
            return Err(format!(
                "metadata array nesting exceeds depth {MAX_GGUF_ARRAY_DEPTH}"
            ));
        }
        match value_type {
            0 => self.u8().map(|_| GgufMetadataValue::Other),
            1 => self.u8().map(|_| GgufMetadataValue::Other),
            2 => self.u16().map(|_| GgufMetadataValue::Other),
            3 => self.u16().map(|_| GgufMetadataValue::Other),
            4 => self.u32().map(GgufMetadataValue::Uint32),
            5 | 6 => self.u32().map(|_| GgufMetadataValue::Other),
            7 => {
                let value = self
                    .u8()
                    .ok_or_else(|| "truncated boolean metadata value".to_owned())?;
                if value > 1 {
                    return Err(format!(
                        "boolean metadata value must be 0 or 1, found {value}"
                    ));
                }
                return Ok(GgufMetadataValue::Other);
            }
            8 => return self.string().map(GgufMetadataValue::String),
            9 => {
                let element_type = self
                    .u32()
                    .ok_or_else(|| "truncated metadata array element type".to_owned())?;
                let count = self
                    .u64()
                    .ok_or_else(|| "truncated metadata array count".to_owned())?;
                if count > MAX_GGUF_ITEMS {
                    return Err(format!(
                        "metadata array count {count} exceeds the {MAX_GGUF_ITEMS} bound"
                    ));
                }
                for _ in 0..count {
                    self.metadata_value(element_type, depth + 1)?;
                }
                return Ok(GgufMetadataValue::Other);
            }
            10 => self.u64().map(|_| GgufMetadataValue::Other),
            11 | 12 => self.u64().map(|_| GgufMetadataValue::Other),
            _ => return Err(format!("unsupported GGUF metadata value type {value_type}")),
        }
        .ok_or_else(|| "truncated metadata value".to_owned())
    }
}
