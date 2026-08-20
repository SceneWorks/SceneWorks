//! Versioned, backend-neutral checkpoint-import contracts.
//!
//! Inventory rows retain a compact [`ImportPlanSummaryV1`] plus a stable
//! [`ImportPlanReferenceV1`]. The immutable, complete [`ImportPlanV1`] is a
//! separate atomically published document, so catalog records never duplicate
//! individual layer locations.

use std::collections::BTreeSet;

use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

/// Wire version shared by all checkpoint-import contracts in this module.
pub const CHECKPOINT_IMPORT_CONTRACT_VERSION: u32 = 1;

const LOCATOR_SEMANTIC_DOMAIN: &str = "sceneworks.checkpoint-import.v1.locator-semantic";
const LOCATOR_BINDING_DOMAIN: &str = "sceneworks.checkpoint-import.v1.locator-binding";
const PLAN_SEMANTIC_DOMAIN: &str = "sceneworks.checkpoint-import.v1.plan-semantic";
const PLAN_BINDING_DOMAIN: &str = "sceneworks.checkpoint-import.v1.plan-binding";

/// Error returned when a checkpoint-import document is malformed or too new.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointImportContractError(pub String);

impl std::fmt::Display for CheckpointImportContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CheckpointImportContractError {}

fn invalid(message: impl Into<String>) -> CheckpointImportContractError {
    CheckpointImportContractError(message.into())
}

fn validate_version(found: u32) -> Result<(), CheckpointImportContractError> {
    if found != CHECKPOINT_IMPORT_CONTRACT_VERSION {
        return Err(invalid(format!(
            "checkpoint-import schema version {found} is unsupported; recompile/rescan required"
        )));
    }
    Ok(())
}

fn validate_nonempty(value: &str, label: &str) -> Result<(), CheckpointImportContractError> {
    if value.trim().is_empty() {
        return Err(invalid(format!("{label} must not be empty")));
    }
    Ok(())
}

fn validate_relative_path(value: &str, label: &str) -> Result<(), CheckpointImportContractError> {
    validate_nonempty(value, label)?;
    if value.contains('\\')
        || value.contains(':')
        || value.starts_with('/')
        || value.bytes().any(|byte| byte.is_ascii_control())
        || value
            .split('/')
            .any(|component| matches!(component, "" | "." | ".."))
    {
        return Err(invalid(format!(
            "{label} must be a portable confined relative path"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<(), CheckpointImportContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be a lowercase SHA-256 hex digest"
        )));
    }
    Ok(())
}

fn validate_sha256_prefixed(value: &str, label: &str) -> Result<(), CheckpointImportContractError> {
    let Some(digest) = value.strip_prefix("sha256:") else {
        return Err(invalid(format!("{label} must use the sha256: prefix")));
    };
    validate_sha256(digest, label)
}

/// A domain-separated SHA-256 identity over a struct with a fixed field order.
/// No map values participate in these contracts, so insertion order cannot affect
/// canonical bytes or a digest.
fn identity<T: Serialize>(domain: &str, value: &T) -> String {
    let payload = serde_json::to_vec(value).expect("checkpoint-import identity payload serializes");
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(payload);
    format!("sha256:{:x}", hasher.finalize())
}

fn canonical_json<T: Serialize>(value: &T) -> Result<String, CheckpointImportContractError> {
    serde_json::to_string(value).map_err(|error| invalid(error.to_string()))
}

/// Provenance retained for an application-owned installed checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ManagedProvenanceV1 {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
}

impl ManagedProvenanceV1 {
    fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_nonempty(&self.source, "managed provenance source")?;
        if let Some(reference) = &self.reference {
            validate_nonempty(reference, "managed provenance reference")?;
        }
        Ok(())
    }
}

/// The physical location of one source file.
///
/// [`Self::semantic_identity`] deliberately excludes physical ownership and
/// location; [`Self::source_binding_identity`] includes both.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum SourceLocatorV1 {
    Linked {
        schema_version: u32,
        root_id: String,
        relative_path: String,
        fingerprint: String,
    },
    Managed {
        schema_version: u32,
        install_id: String,
        relative_path: String,
        sha256: String,
        provenance: ManagedProvenanceV1,
    },
}

#[derive(Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum SourceLocatorV1Wire {
    Linked {
        schema_version: u32,
        root_id: String,
        relative_path: String,
        fingerprint: String,
    },
    Managed {
        schema_version: u32,
        install_id: String,
        relative_path: String,
        sha256: String,
        provenance: ManagedProvenanceV1,
    },
}

impl From<SourceLocatorV1Wire> for SourceLocatorV1 {
    fn from(value: SourceLocatorV1Wire) -> Self {
        match value {
            SourceLocatorV1Wire::Linked {
                schema_version,
                root_id,
                relative_path,
                fingerprint,
            } => Self::Linked {
                schema_version,
                root_id,
                relative_path,
                fingerprint,
            },
            SourceLocatorV1Wire::Managed {
                schema_version,
                install_id,
                relative_path,
                sha256,
                provenance,
            } => Self::Managed {
                schema_version,
                install_id,
                relative_path,
                sha256,
                provenance,
            },
        }
    }
}

impl<'de> Deserialize<'de> for SourceLocatorV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let locator = Self::from(SourceLocatorV1Wire::deserialize(deserializer)?);
        locator.validate().map_err(D::Error::custom)?;
        Ok(locator)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LocatorSemanticV1<'a> {
    schema_version: u32,
    content_digest: &'a str,
}

impl SourceLocatorV1 {
    pub fn linked(
        root_id: impl Into<String>,
        relative_path: impl Into<String>,
        fingerprint: impl Into<String>,
    ) -> Result<Self, CheckpointImportContractError> {
        let locator = Self::Linked {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            root_id: root_id.into(),
            relative_path: relative_path.into(),
            fingerprint: fingerprint.into(),
        };
        locator.validate()?;
        Ok(locator)
    }

    pub fn managed(
        install_id: impl Into<String>,
        relative_path: impl Into<String>,
        sha256: impl Into<String>,
        provenance: ManagedProvenanceV1,
    ) -> Result<Self, CheckpointImportContractError> {
        let locator = Self::Managed {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            install_id: install_id.into(),
            relative_path: relative_path.into(),
            sha256: sha256.into(),
            provenance,
        };
        locator.validate()?;
        Ok(locator)
    }

    pub fn content_digest(&self) -> &str {
        match self {
            Self::Linked { fingerprint, .. } => fingerprint,
            Self::Managed { sha256, .. } => sha256,
        }
    }

    /// Locator-independent identity of the source bytes.
    pub fn semantic_identity(&self) -> String {
        identity(
            LOCATOR_SEMANTIC_DOMAIN,
            &LocatorSemanticV1 {
                schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
                content_digest: self.content_digest(),
            },
        )
    }

    /// Identity of the source bytes and its physical owner, path, and provenance.
    pub fn source_binding_identity(&self) -> String {
        identity(LOCATOR_BINDING_DOMAIN, self)
    }

    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(self)
    }

    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        match self {
            Self::Linked {
                schema_version,
                root_id,
                relative_path,
                fingerprint,
            } => {
                validate_version(*schema_version)?;
                validate_nonempty(root_id, "linked root id")?;
                validate_relative_path(relative_path, "linked relative path")?;
                validate_sha256(fingerprint, "linked fingerprint")
            }
            Self::Managed {
                schema_version,
                install_id,
                relative_path,
                sha256,
                provenance,
            } => {
                validate_version(*schema_version)?;
                validate_nonempty(install_id, "managed install id")?;
                validate_relative_path(relative_path, "managed relative path")?;
                validate_sha256(sha256, "managed SHA-256")?;
                provenance.validate()
            }
        }
    }
}

/// One logical layer of an import plan. The source remains separate from its
/// logical shape, so linked and managed copies can compile to the same plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportLayerV1 {
    pub layer_id: String,
    pub role: String,
    pub target_path: String,
    pub source: SourceLocatorV1,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImportLayerV1Wire {
    layer_id: String,
    role: String,
    target_path: String,
    source: SourceLocatorV1,
}

impl From<ImportLayerV1Wire> for ImportLayerV1 {
    fn from(value: ImportLayerV1Wire) -> Self {
        Self {
            layer_id: value.layer_id,
            role: value.role,
            target_path: value.target_path,
            source: value.source,
        }
    }
}

impl<'de> Deserialize<'de> for ImportLayerV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let layer = Self::from(ImportLayerV1Wire::deserialize(deserializer)?);
        layer.validate().map_err(D::Error::custom)?;
        Ok(layer)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SemanticLayerV1<'a> {
    layer_id: &'a str,
    role: &'a str,
    target_path: &'a str,
    source_semantic_identity: String,
}

impl ImportLayerV1 {
    fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_nonempty(&self.layer_id, "import layer id")?;
        validate_nonempty(&self.role, "import layer role")?;
        validate_relative_path(&self.target_path, "import layer target path")?;
        self.source.validate()
    }

    fn semantic_form(&self) -> SemanticLayerV1<'_> {
        SemanticLayerV1 {
            layer_id: &self.layer_id,
            role: &self.role,
            target_path: &self.target_path,
            source_semantic_identity: self.source.semantic_identity(),
        }
    }
}

/// The complete, immutable loading plan stored separately from catalog rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportPlanV1 {
    pub schema_version: u32,
    pub plan_id: String,
    pub family: String,
    pub layers: Vec<ImportLayerV1>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImportPlanV1Wire {
    schema_version: u32,
    plan_id: String,
    family: String,
    layers: Vec<ImportLayerV1>,
}

impl From<ImportPlanV1Wire> for ImportPlanV1 {
    fn from(value: ImportPlanV1Wire) -> Self {
        Self {
            schema_version: value.schema_version,
            plan_id: value.plan_id,
            family: value.family,
            layers: value.layers,
        }
    }
}

impl<'de> Deserialize<'de> for ImportPlanV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let plan = Self::from(ImportPlanV1Wire::deserialize(deserializer)?);
        plan.validate().map_err(D::Error::custom)?;
        Ok(plan)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SemanticPlanV1<'a> {
    schema_version: u32,
    plan_id: &'a str,
    family: &'a str,
    layers: Vec<SemanticLayerV1<'a>>,
}

impl ImportPlanV1 {
    pub fn new(
        plan_id: impl Into<String>,
        family: impl Into<String>,
        mut layers: Vec<ImportLayerV1>,
    ) -> Result<Self, CheckpointImportContractError> {
        layers.sort_by(|left, right| left.layer_id.cmp(&right.layer_id));
        let plan = Self {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            plan_id: plan_id.into(),
            family: family.into(),
            layers,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Content and logical-routing identity, deliberately excluding all physical
    /// source owners, paths, and managed provenance.
    pub fn semantic_digest(&self) -> String {
        identity(PLAN_SEMANTIC_DOMAIN, &self.semantic_form())
    }

    /// Identity that also binds each layer's exact physical owner, path, and provenance.
    pub fn source_binding_identity(&self) -> String {
        identity(PLAN_BINDING_DOMAIN, &self.canonicalized())
    }

    pub fn plan_reference(&self) -> ImportPlanReferenceV1 {
        ImportPlanReferenceV1 {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            plan_id: self.plan_id.clone(),
            semantic_digest: self.semantic_digest(),
            source_binding_identity: self.source_binding_identity(),
        }
    }

    pub fn summary(&self) -> ImportPlanSummaryV1 {
        ImportPlanSummaryV1 {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            family: self.family.clone(),
            layer_count: self.layers.len() as u32,
            layer_roles: self.layers.iter().map(|layer| layer.role.clone()).collect(),
            semantic_digest: self.semantic_digest(),
        }
    }

    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(&self.canonicalized())
    }

    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        validate_nonempty(&self.plan_id, "import plan id")?;
        validate_nonempty(&self.family, "import plan family")?;
        if self.layers.is_empty() {
            return Err(invalid("import plan must contain at least one layer"));
        }
        let mut previous = None;
        for layer in &self.layers {
            layer.validate()?;
            if previous.as_deref() >= Some(layer.layer_id.as_str()) {
                return Err(invalid(
                    "import plan layers must be uniquely sorted by layer id",
                ));
            }
            previous = Some(layer.layer_id.clone());
        }
        Ok(())
    }

    fn semantic_form(&self) -> SemanticPlanV1<'_> {
        let mut layers: Vec<_> = self
            .layers
            .iter()
            .map(ImportLayerV1::semantic_form)
            .collect();
        layers.sort_by(|left, right| left.layer_id.cmp(right.layer_id));
        SemanticPlanV1 {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            plan_id: &self.plan_id,
            family: &self.family,
            layers,
        }
    }

    fn canonicalized(&self) -> Self {
        let mut plan = self.clone();
        plan.layers
            .sort_by(|left, right| left.layer_id.cmp(&right.layer_id));
        plan
    }
}

/// Stable handle to a separately atomically stored [`ImportPlanV1`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportPlanReferenceV1 {
    pub schema_version: u32,
    pub plan_id: String,
    pub semantic_digest: String,
    pub source_binding_identity: String,
}

/// Short spelling for APIs that conventionally call this a plan reference.
pub type ImportPlanRefV1 = ImportPlanReferenceV1;

impl<'de> Deserialize<'de> for ImportPlanReferenceV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            schema_version: u32,
            plan_id: String,
            semantic_digest: String,
            source_binding_identity: String,
        }
        let wire = Wire::deserialize(deserializer)?;
        let reference = Self {
            schema_version: wire.schema_version,
            plan_id: wire.plan_id,
            semantic_digest: wire.semantic_digest,
            source_binding_identity: wire.source_binding_identity,
        };
        reference.validate().map_err(D::Error::custom)?;
        Ok(reference)
    }
}

impl ImportPlanReferenceV1 {
    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(self)
    }
    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        validate_nonempty(&self.plan_id, "import plan reference id")?;
        validate_sha256_prefixed(&self.semantic_digest, "import plan semantic digest")?;
        validate_sha256_prefixed(
            &self.source_binding_identity,
            "import plan source-binding identity",
        )
    }
}

/// Small catalog-facing projection of a plan; it intentionally contains no layers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportPlanSummaryV1 {
    pub schema_version: u32,
    pub family: String,
    pub layer_count: u32,
    pub layer_roles: Vec<String>,
    pub semantic_digest: String,
}

impl<'de> Deserialize<'de> for ImportPlanSummaryV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            schema_version: u32,
            family: String,
            layer_count: u32,
            layer_roles: Vec<String>,
            semantic_digest: String,
        }
        let wire = Wire::deserialize(deserializer)?;
        let summary = Self {
            schema_version: wire.schema_version,
            family: wire.family,
            layer_count: wire.layer_count,
            layer_roles: wire.layer_roles,
            semantic_digest: wire.semantic_digest,
        };
        summary.validate().map_err(D::Error::custom)?;
        Ok(summary)
    }
}

impl ImportPlanSummaryV1 {
    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(self)
    }
    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        validate_nonempty(&self.family, "import plan summary family")?;
        if self.layer_count == 0 || self.layer_count as usize != self.layer_roles.len() {
            return Err(invalid(
                "import plan summary layer count must match its non-empty roles",
            ));
        }
        for role in &self.layer_roles {
            validate_nonempty(role, "import plan summary layer role")?;
        }
        validate_sha256_prefixed(&self.semantic_digest, "import plan summary semantic digest")
    }
}

/// A catalog row with a plan reference and compact projection, never per-layer data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CheckpointCatalogRecordV1 {
    pub checkpoint_id: String,
    pub plan: ImportPlanReferenceV1,
    pub summary: ImportPlanSummaryV1,
}

impl<'de> Deserialize<'de> for CheckpointCatalogRecordV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            checkpoint_id: String,
            plan: ImportPlanReferenceV1,
            summary: ImportPlanSummaryV1,
        }
        let wire = Wire::deserialize(deserializer)?;
        let record = Self {
            checkpoint_id: wire.checkpoint_id,
            plan: wire.plan,
            summary: wire.summary,
        };
        record.validate().map_err(D::Error::custom)?;
        Ok(record)
    }
}

impl CheckpointCatalogRecordV1 {
    pub fn from_plan(checkpoint_id: impl Into<String>, plan: &ImportPlanV1) -> Self {
        Self {
            checkpoint_id: checkpoint_id.into(),
            plan: plan.plan_reference(),
            summary: plan.summary(),
        }
    }
    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        canonical_json(self)
    }
    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_nonempty(&self.checkpoint_id, "checkpoint id")?;
        self.plan.validate()?;
        self.summary.validate()?;
        if self.plan.semantic_digest != self.summary.semantic_digest {
            return Err(invalid(
                "catalog plan reference and summary semantic digests differ",
            ));
        }
        Ok(())
    }
}

/// Canonical inventory of catalog records used by discovery and routing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CheckpointInventoryV1 {
    pub schema_version: u32,
    pub records: Vec<CheckpointCatalogRecordV1>,
}

impl<'de> Deserialize<'de> for CheckpointInventoryV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            schema_version: u32,
            records: Vec<CheckpointCatalogRecordV1>,
        }
        let wire = Wire::deserialize(deserializer)?;
        let inventory = Self {
            schema_version: wire.schema_version,
            records: wire.records,
        };
        inventory.validate().map_err(D::Error::custom)?;
        Ok(inventory)
    }
}

impl CheckpointInventoryV1 {
    pub fn new(
        mut records: Vec<CheckpointCatalogRecordV1>,
    ) -> Result<Self, CheckpointImportContractError> {
        records.sort_by(|left, right| left.checkpoint_id.cmp(&right.checkpoint_id));
        let inventory = Self {
            schema_version: CHECKPOINT_IMPORT_CONTRACT_VERSION,
            records,
        };
        inventory.validate()?;
        Ok(inventory)
    }

    pub fn canonical_json(&self) -> Result<String, CheckpointImportContractError> {
        self.validate()?;
        let mut inventory = self.clone();
        inventory
            .records
            .sort_by(|left, right| left.checkpoint_id.cmp(&right.checkpoint_id));
        canonical_json(&inventory)
    }

    pub fn validate(&self) -> Result<(), CheckpointImportContractError> {
        validate_version(self.schema_version)?;
        let mut ids = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if !ids.insert(&record.checkpoint_id) {
                return Err(invalid(
                    "checkpoint inventory contains duplicate checkpoint ids",
                ));
            }
        }
        Ok(())
    }
}
