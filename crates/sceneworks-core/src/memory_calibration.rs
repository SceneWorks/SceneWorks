//! Typed, fail-closed reader for the promoted memory-calibration evidence bundle.
//!
//! The canonical bundle is compiled into `sceneworks-core`. This keeps the same evidence bytes
//! available to desktop and remote workers without relying on the repository-only `docs/` path at
//! runtime. Gate wiring belongs to SC-15611; this module only parses and classifies evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

/// Schema v4 adds the required per-record `loadShape` axis. Older packaged evidence remains
/// parseable but stale, so consumers fall back until it is recollected under the typed shape ABI.
pub const MEMORY_CALIBRATION_SCHEMA_VERSION: u32 = 4;
pub const MEMORY_CALIBRATION_HARNESS_VERSION: &str = "sceneworks-memory-v4";
/// ABI paired by the manifest/query side of the reader.
///
/// The evidence schema deliberately stays at v2: callers must supply the manifest's ABI together
/// with its fingerprint. Exact source revisions remain captured provenance; the ABI/fingerprint
/// pair owns SceneWorks invalidation without rewriting the already-promoted producer contract.
pub const MEMORY_CALIBRATION_ABI: u32 = 2;
pub const PACKAGED_MEMORY_CALIBRATION_EVIDENCE: &str =
    include_str!("../../../docs/generated/memory-calibration-evidence.json");

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvidenceBundle {
    pub schema_version: u32,
    pub harness_version: String,
    pub records: Vec<EvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvidenceRecord {
    pub id: String,
    pub logical_case_id: String,
    pub status: RecordStatus,
    pub evidence_scope: EvidenceScope,
    pub backend: Backend,
    pub repositories: Repositories,
    pub hardware: Hardware,
    pub artifact: Artifact,
    pub target: Target,
    pub fixture: String,
    pub strategy: Strategy,
    pub load_shape: LoadShapeKey,
    pub sweep: Sweep,
    pub scenarios: Vec<Scenario>,
    pub predicted_peak_bytes: RequiredNullable<PredictedPeakBytes>,
    pub observed_memory: RequiredNullable<PhaseMetrics>,
    pub quality: Quality,
    pub negative_mutation: RequiredNullable<NegativeMutation>,
    pub loadability: Loadability,
    pub diagnostics: Option<Diagnostics>,
    pub calibration_fingerprint: String,
    pub captured_at: String,
    pub harness_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadShapeKey {
    EagerMaterialization,
    DeferredMaterialization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordStatus {
    Complete,
    Gated,
    NegativeComplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceScope {
    Authoritative,
    Candidate,
    Fixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Mlx,
    Candle,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Repositories {
    pub scene_works: GitState,
    pub inference: GitState,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GitState {
    pub revision: String,
    pub dirty: bool,
    pub matrix_source_revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Hardware {
    Cuda(CudaHardware),
    Mlx(MlxHardware),
}

impl Hardware {
    fn memory_bytes(&self) -> u64 {
        match self {
            Self::Cuda(hardware) => hardware.memory_bytes,
            Self::Mlx(hardware) => hardware.memory_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CudaHardware {
    pub probe: String,
    pub memory_bytes: u64,
    pub device_id: String,
    pub name: String,
    pub compute_capability: String,
    pub driver_version: String,
    pub runtime_version: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MlxHardware {
    pub probe: String,
    pub memory_bytes: u64,
    pub model: String,
    pub chip: String,
    pub os_version: String,
    pub metal_device: String,
    pub mlx_memory_limit_bytes: u64,
    pub wired_limit_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Artifact {
    pub repository: String,
    pub resolved_revision: String,
    pub variant: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Target {
    pub model_id: String,
    pub provider: String,
    pub tier: String,
    pub mode: String,
    pub overlay: String,
    pub geometry: Geometry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Geometry {
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    pub frames: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Strategy {
    pub rung: StrategyRung,
    pub engaged_rungs: Vec<StrategyRung>,
    pub parameters: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyRung {
    Resident,
    StagedResidency,
    BoundedDecode,
    BoundedAttention,
    BoundedTransformerResidency,
}

impl StrategyRung {
    pub const ALL: [Self; 5] = [
        Self::Resident,
        Self::StagedResidency,
        Self::BoundedDecode,
        Self::BoundedAttention,
        Self::BoundedTransformerResidency,
    ];
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sweep {
    pub axes: Vec<SweepAxis>,
    pub cases: Vec<SweepCase>,
    pub range_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SweepAxis {
    pub parameter: String,
    pub tested_values: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SweepCase {
    pub parameters: Map<String, Value>,
    pub result: SweepResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SweepResult {
    Passed,
    Failed,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Scenario {
    pub name: ScenarioName,
    pub result: ScenarioResult,
    pub reason: Option<String>,
    pub predicted_bytes: Option<u64>,
    pub effective_budget_bytes: Option<u64>,
    pub cleanup_verified: Option<bool>,
    pub warm_follow_up_passed: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioName {
    ExactFit,
    UnknownBudget,
    StaleEvidence,
    WarmRepeat,
    Cancel,
    Error,
    Loadability,
    Overlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioResult {
    Passed,
    Failed,
    NotApplicable,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PredictedPeakBytes {
    pub conditioning: u64,
    pub denoise: u64,
    pub decode: u64,
    pub overall: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PhaseMetrics {
    pub conditioning: Phase,
    pub denoise: Phase,
    pub decode: Phase,
    pub overall: Phase,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Phase {
    pub active_bytes: u64,
    pub allocator_bytes: u64,
    pub device_bytes: u64,
    pub wired_bytes: u64,
    pub reclaimable_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Quality {
    pub contract: Option<String>,
    pub identical_inputs: Option<bool>,
    pub identical_latents: Option<bool>,
    pub result: Option<QualityResult>,
    pub maximum_error: Option<f64>,
    pub mean_error: Option<f64>,
    pub maximum_error_threshold: Option<f64>,
    pub mean_error_threshold: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityResult {
    Passed,
    Failed,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NegativeMutation {
    pub parameters: Option<Map<String, Value>>,
    pub measured: Option<bool>,
    pub result: Option<NegativeMutationResult>,
    pub maximum_error: Option<f64>,
    pub mean_error: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NegativeMutationResult {
    FailedAsExpected,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Loadability {
    pub result: LoadabilityResult,
    pub resolved_path_fingerprint: RequiredNullable<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadabilityResult {
    Passed,
    Failed,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Diagnostics {
    pub adapter: String,
    pub execution: DiagnosticsExecution,
    pub blockers: Vec<String>,
    pub measurements: Vec<DiagnosticMeasurement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticsExecution {
    Executed,
    GatedBeforeExecution,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticMeasurement {
    pub name: String,
    pub unit: String,
    pub value: u64,
}

/// A required JSON field which may explicitly be `null`.
#[derive(Debug, Clone, PartialEq)]
pub enum RequiredNullable<T> {
    Null,
    Value(T),
}

impl<'de, T> Deserialize<'de> for RequiredNullable<T>
where
    T: DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(|value| match value {
            Some(value) => Self::Value(value),
            None => Self::Null,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaleBundleReason {
    SchemaVersion { found: Option<u64> },
    HarnessVersion { found: Option<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BundleLoad {
    Ready(EvidenceBundle),
    Stale(StaleBundleReason),
}

#[derive(Debug)]
pub enum BundleLoadError {
    Json(serde_json::Error),
    Invalid(String),
}

impl fmt::Display for BundleLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid memory-calibration JSON: {error}"),
            Self::Invalid(message) => {
                write!(formatter, "invalid memory-calibration evidence: {message}")
            }
        }
    }
}

impl std::error::Error for BundleLoadError {}

pub fn load_bundle(source: &str) -> Result<BundleLoad, BundleLoadError> {
    let raw: Value = serde_json::from_str(source).map_err(BundleLoadError::Json)?;
    let schema_version = raw.get("schemaVersion").and_then(Value::as_u64);
    if schema_version.is_some()
        && schema_version != Some(u64::from(MEMORY_CALIBRATION_SCHEMA_VERSION))
    {
        return Ok(BundleLoad::Stale(StaleBundleReason::SchemaVersion {
            found: schema_version,
        }));
    }
    let harness_version = raw
        .get("harnessVersion")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if harness_version.is_some()
        && harness_version.as_deref() != Some(MEMORY_CALIBRATION_HARNESS_VERSION)
    {
        return Ok(BundleLoad::Stale(StaleBundleReason::HarnessVersion {
            found: harness_version,
        }));
    }
    if let Some(found) = raw
        .get("records")
        .and_then(Value::as_array)
        .and_then(|records| {
            records.iter().find_map(|record| {
                record
                    .get("harnessVersion")
                    .and_then(Value::as_str)
                    .filter(|found| *found != MEMORY_CALIBRATION_HARNESS_VERSION)
                    .map(|found| Some(found.to_owned()))
            })
        })
    {
        return Ok(BundleLoad::Stale(StaleBundleReason::HarnessVersion {
            found,
        }));
    }

    let bundle: EvidenceBundle = serde_json::from_value(raw).map_err(BundleLoadError::Json)?;
    validate_bundle(&bundle).map_err(BundleLoadError::Invalid)?;
    Ok(BundleLoad::Ready(bundle))
}

/// Load the exact bundle compiled into the product.
pub fn load_packaged_bundle() -> Result<BundleLoad, BundleLoadError> {
    load_bundle(PACKAGED_MEMORY_CALIBRATION_EVIDENCE)
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationBinding {
    pub abi: u32,
    pub load_shape: LoadShapeKey,
    pub fingerprint: String,
    pub scene_works_revision: String,
    pub matrix_source_revision: String,
    pub inference_revision: String,
    pub artifact_repository: String,
    pub artifact_resolved_revision: String,
    pub artifact_variant: String,
    pub resolved_path_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceQuery {
    pub backend: Backend,
    pub model_id: String,
    pub provider: String,
    pub tier: String,
    pub mode: String,
    pub overlay: String,
    pub geometry: Geometry,
    pub rung: StrategyRung,
    pub parameters: Map<String, Value>,
    pub calibration: CalibrationBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleEvidenceReason {
    CalibrationAbi,
    LoadShape,
    CalibrationFingerprint,
    InferenceRevision,
    ArtifactRepository,
    ArtifactResolvedRevision,
    ArtifactVariant,
    ResolvedPathFingerprint,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceVerdict<'a> {
    Verified(&'a EvidenceRecord),
    Stale(StaleEvidenceReason),
    OutOfEnvelope,
    Unknown,
}

impl EvidenceBundle {
    /// Find exact, authoritative evidence for a prospective gate decision.
    ///
    /// Geometry and strategy parameters are contained only by exact equality with the record target
    /// and a passed executed sweep case. No substring, list-inclusion, interpolation, or pixel-area
    /// approximation is accepted.
    pub fn evidence_for(&self, query: &EvidenceQuery) -> EvidenceVerdict<'_> {
        let mut candidates = self
            .records
            .iter()
            .filter(|record| {
                record.status == RecordStatus::Complete
                    && record.evidence_scope == EvidenceScope::Authoritative
                    && record.backend == query.backend
                    && record.target.model_id == query.model_id
                    && record.target.provider == query.provider
                    && record.target.tier == query.tier
                    && record.target.mode == query.mode
                    && record.target.overlay == query.overlay
                    && record.strategy.rung == query.rung
            })
            .peekable();
        if candidates.peek().is_none() {
            return EvidenceVerdict::Unknown;
        }
        if query.calibration.abi != MEMORY_CALIBRATION_ABI {
            return EvidenceVerdict::Stale(StaleEvidenceReason::CalibrationAbi);
        }

        let mut saw_current_identity = false;
        let mut stale = None;
        for record in candidates {
            let mismatch = if record.load_shape != query.calibration.load_shape {
                Some(StaleEvidenceReason::LoadShape)
            } else if record.calibration_fingerprint != query.calibration.fingerprint {
                Some(StaleEvidenceReason::CalibrationFingerprint)
            } else if record.repositories.inference.revision != query.calibration.inference_revision
            {
                Some(StaleEvidenceReason::InferenceRevision)
            } else if record.artifact.repository != query.calibration.artifact_repository {
                Some(StaleEvidenceReason::ArtifactRepository)
            } else if record.artifact.resolved_revision
                != query.calibration.artifact_resolved_revision
            {
                Some(StaleEvidenceReason::ArtifactResolvedRevision)
            } else if record.artifact.variant != query.calibration.artifact_variant {
                Some(StaleEvidenceReason::ArtifactVariant)
            } else if !matches!(
                &record.loadability.resolved_path_fingerprint,
                RequiredNullable::Value(value)
                    if value == &query.calibration.resolved_path_fingerprint
            ) {
                Some(StaleEvidenceReason::ResolvedPathFingerprint)
            } else {
                None
            };
            if let Some(reason) = mismatch {
                stale.get_or_insert(reason);
                continue;
            }
            saw_current_identity = true;

            let exact_geometry = record.target.geometry == query.geometry;
            let passed_exact_case = record.sweep.range_verified
                && record.sweep.cases.iter().any(|case| {
                    case.result == SweepResult::Passed && case.parameters == query.parameters
                });
            if exact_geometry && passed_exact_case {
                return EvidenceVerdict::Verified(record);
            }
        }

        if saw_current_identity {
            EvidenceVerdict::OutOfEnvelope
        } else if let Some(reason) = stale {
            EvidenceVerdict::Stale(reason)
        } else {
            EvidenceVerdict::Unknown
        }
    }
}

/// Memory quantities used by admission after a verified MLX cell has been selected.
///
/// `peak_bytes` is the larger of the producer's predicted whole-request peak and the measured,
/// non-reclaimable wired high-water mark. `foreign_reserve_bytes` is the part of unified memory the
/// captured MLX limits deliberately left outside the process. Keeping these terms separate prevents
/// the dedicated-VRAM allocator slack used by Candle from being mistaken for macOS/foreign demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxAdmissionEnvelope {
    pub peak_bytes: u64,
    pub observed_non_reclaimable_wired_bytes: u64,
    pub foreign_reserve_bytes: u64,
}

impl MlxAdmissionEnvelope {
    /// Smallest physical unified-memory size that can satisfy this exact cell.
    ///
    /// This is the shared runtime/UI bridge: the worker compares exact evidence against the same
    /// additive requirement that a later web consumer may display. Saturation fails conservative
    /// for corrupt or impossible producer values.
    pub fn required_host_bytes(self) -> u64 {
        self.peak_bytes.saturating_add(self.foreign_reserve_bytes)
    }

    pub fn fits_host_bytes(self, host_bytes: u64) -> bool {
        self.required_host_bytes() <= host_bytes
    }
}

impl EvidenceRecord {
    /// Derive the verified MLX admission envelope from counters carried by the evidence contract.
    ///
    /// No constant is introduced here. The usable process ceiling is the smaller of the captured
    /// MLX memory and wired limits; the remainder of physical unified memory is foreign resident
    /// demand. Reclaimable allocator bytes are not charged as wired residency.
    pub fn mlx_admission_envelope(&self) -> Option<MlxAdmissionEnvelope> {
        let Hardware::Mlx(hardware) = &self.hardware else {
            return None;
        };
        let RequiredNullable::Value(predicted) = &self.predicted_peak_bytes else {
            return None;
        };
        let RequiredNullable::Value(observed) = &self.observed_memory else {
            return None;
        };
        let process_ceiling = hardware
            .mlx_memory_limit_bytes
            .min(hardware.wired_limit_bytes)
            .min(hardware.memory_bytes);
        let foreign_reserve_bytes = hardware.memory_bytes.saturating_sub(process_ceiling);
        let non_reclaimable_wired = observed
            .overall
            .wired_bytes
            .saturating_sub(observed.overall.reclaimable_bytes);
        Some(MlxAdmissionEnvelope {
            peak_bytes: predicted.overall.max(non_reclaimable_wired),
            observed_non_reclaimable_wired_bytes: non_reclaimable_wired,
            foreign_reserve_bytes,
        })
    }
}

fn validate_bundle(bundle: &EvidenceBundle) -> Result<(), String> {
    for record in &bundle.records {
        validate_record(record)?;
    }
    Ok(())
}

fn validate_record(record: &EvidenceRecord) -> Result<(), String> {
    require_nonempty(&record.id, "record.id")?;
    require_nonempty(&record.logical_case_id, "record.logicalCaseId")?;
    if !has_prefixed_hex(&record.id, "imc-", 20) {
        return Err("record.id must be imc- plus 20 lowercase hex digits".to_owned());
    }
    if !has_prefixed_hex(&record.logical_case_id, "implan-", 20) {
        return Err(format!(
            "{} logicalCaseId must be implan- plus 20 lowercase hex digits",
            record.id
        ));
    }
    if record.harness_version != MEMORY_CALIBRATION_HARNESS_VERSION {
        return Err(format!("{} has a stale harnessVersion", record.id));
    }
    validate_git_state(&record.repositories.scene_works, true, &record.id)?;
    validate_git_state(&record.repositories.inference, false, &record.id)?;
    validate_hardware(&record.hardware, &record.id)?;
    if !matches!(
        (record.backend, &record.hardware),
        (Backend::Mlx, Hardware::Mlx(_)) | (Backend::Candle, Hardware::Cuda(_))
    ) {
        return Err(format!(
            "{} backend does not agree with its hardware contract arm",
            record.id
        ));
    }
    for (value, field) in [
        (&record.artifact.repository, "artifact.repository"),
        (
            &record.artifact.resolved_revision,
            "artifact.resolvedRevision",
        ),
        (&record.artifact.variant, "artifact.variant"),
        (&record.target.model_id, "target.modelId"),
        (&record.target.provider, "target.provider"),
        (&record.target.tier, "target.tier"),
        (&record.target.mode, "target.mode"),
        (&record.target.overlay, "target.overlay"),
        (&record.fixture, "fixture"),
        (&record.calibration_fingerprint, "calibrationFingerprint"),
        (&record.captured_at, "capturedAt"),
    ] {
        require_nonempty(value, &format!("{}.{}", record.id, field))?;
    }
    if !is_rfc3339_datetime(&record.captured_at) {
        return Err(format!("{} capturedAt is not RFC 3339", record.id));
    }
    if record
        .quality
        .contract
        .as_deref()
        .is_some_and(str::is_empty)
        || [
            record.quality.maximum_error,
            record.quality.mean_error,
            record.quality.maximum_error_threshold,
            record.quality.mean_error_threshold,
        ]
        .into_iter()
        .flatten()
        .any(|value| value < 0.0)
    {
        return Err(format!("{} quality fields violate the schema", record.id));
    }
    if let RequiredNullable::Value(mutation) = &record.negative_mutation {
        if [mutation.maximum_error, mutation.mean_error]
            .into_iter()
            .flatten()
            .any(|value| value < 0.0)
        {
            return Err(format!(
                "{} negativeMutation metrics must be nonnegative",
                record.id
            ));
        }
    }
    if [
        record.target.geometry.width,
        record.target.geometry.height,
        record.target.geometry.batch,
        record.target.geometry.frames,
    ]
    .contains(&0)
    {
        return Err(format!("{} target geometry must be positive", record.id));
    }
    let engaged = record
        .strategy
        .engaged_rungs
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let canonical = StrategyRung::ALL
        .into_iter()
        .filter(|rung| engaged.contains(rung))
        .collect::<Vec<_>>();
    if engaged.len() != record.strategy.engaged_rungs.len()
        || canonical != record.strategy.engaged_rungs
        || !engaged.contains(&StrategyRung::Resident)
        || !engaged.contains(&record.strategy.rung)
    {
        return Err(format!(
            "{} strategy.engagedRungs must be a unique canonical set containing resident and the selected rung",
            record.id
        ));
    }
    validate_sweep(&record.sweep, &record.id)?;
    if let Some(diagnostics) = &record.diagnostics {
        require_nonempty(
            &diagnostics.adapter,
            &format!("{}.diagnostics.adapter", record.id),
        )?;
        for blocker in &diagnostics.blockers {
            require_nonempty(blocker, &format!("{}.diagnostics.blockers", record.id))?;
        }
        for measurement in &diagnostics.measurements {
            require_nonempty(
                &measurement.name,
                &format!("{}.diagnostics.measurements.name", record.id),
            )?;
            require_nonempty(
                &measurement.unit,
                &format!("{}.diagnostics.measurements.unit", record.id),
            )?;
        }
    }

    match record.status {
        RecordStatus::Complete => validate_complete(record),
        RecordStatus::NegativeComplete => validate_negative_complete(record),
        RecordStatus::Gated => Ok(()),
    }
}

fn validate_git_state(state: &GitState, require_matrix: bool, id: &str) -> Result<(), String> {
    if !(7..=40).contains(&state.revision.len()) || !is_lower_hex(&state.revision) {
        return Err(format!("{id} has an invalid repository revision"));
    }
    if require_matrix {
        let revision = state
            .matrix_source_revision
            .as_deref()
            .ok_or_else(|| format!("{id} is missing sceneWorks.matrixSourceRevision"))?;
        let suffix = revision
            .strip_prefix("source-tree:")
            .ok_or_else(|| format!("{id} has an invalid matrixSourceRevision"))?;
        if suffix.is_empty() || !is_lower_hex(suffix) {
            return Err(format!("{id} has an invalid matrixSourceRevision"));
        }
    }
    Ok(())
}

fn validate_hardware(hardware: &Hardware, id: &str) -> Result<(), String> {
    match hardware {
        Hardware::Cuda(hardware) => {
            if hardware.memory_bytes == 0 {
                return Err(format!("{id} hardware.memoryBytes must be positive"));
            }
            for value in [
                &hardware.probe,
                &hardware.device_id,
                &hardware.name,
                &hardware.compute_capability,
                &hardware.driver_version,
                &hardware.runtime_version,
            ] {
                require_nonempty(value, &format!("{id}.hardware"))?;
            }
        }
        Hardware::Mlx(hardware) => {
            if hardware.memory_bytes == 0
                || hardware.mlx_memory_limit_bytes == 0
                || hardware.wired_limit_bytes == 0
            {
                return Err(format!("{id} MLX hardware limits must be positive"));
            }
            for value in [
                &hardware.probe,
                &hardware.model,
                &hardware.chip,
                &hardware.os_version,
                &hardware.metal_device,
            ] {
                require_nonempty(value, &format!("{id}.hardware"))?;
            }
        }
    }
    Ok(())
}

fn validate_sweep(sweep: &Sweep, id: &str) -> Result<(), String> {
    if sweep.cases.is_empty() {
        return Err(format!("{id} sweep cases must be nonempty"));
    }
    let mut names = BTreeSet::new();
    for axis in &sweep.axes {
        require_nonempty(&axis.parameter, &format!("{id}.sweep.axis.parameter"))?;
        if axis.tested_values.is_empty()
            || axis
                .tested_values
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != axis.tested_values.len()
            || !names.insert(&axis.parameter)
        {
            return Err(format!("{id} sweep axes and tested values must be unique"));
        }
    }
    for (index, case) in sweep.cases.iter().enumerate() {
        if sweep.cases[..index]
            .iter()
            .any(|prior| prior.parameters == case.parameters)
        {
            return Err(format!("{id} sweep cases must be unique"));
        }
    }
    Ok(())
}

fn validate_complete(record: &EvidenceRecord) -> Result<(), String> {
    if record.repositories.scene_works.dirty || record.repositories.inference.dirty {
        return Err(format!(
            "{} complete evidence has a dirty repository",
            record.id
        ));
    }
    if !record.sweep.range_verified {
        return Err(format!(
            "{} complete sweep is not range-verified",
            record.id
        ));
    }
    if !record.sweep.cases.iter().any(|case| {
        case.result == SweepResult::Passed && case.parameters == record.strategy.parameters
    }) {
        return Err(format!(
            "{} exact strategy parameters are not a passed sweep case",
            record.id
        ));
    }
    for axis in &record.sweep.axes {
        let actual = record
            .sweep
            .cases
            .iter()
            .filter(|case| case.result == SweepResult::Passed)
            .map(|case| {
                case.parameters
                    .get(&axis.parameter)
                    .and_then(Value::as_u64)
                    .ok_or_else(|| {
                        format!(
                            "{} passed sweep case lacks integer axis {}",
                            record.id, axis.parameter
                        )
                    })
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let declared = axis.tested_values.iter().copied().collect::<BTreeSet<_>>();
        if actual.is_empty() || actual != declared {
            return Err(format!(
                "{} axis {} is not derived from passed sweep cases",
                record.id, axis.parameter
            ));
        }
    }
    let predicted = required_value(
        &record.predicted_peak_bytes,
        &record.id,
        "predictedPeakBytes",
    )?;
    if predicted.overall
        < predicted
            .conditioning
            .max(predicted.denoise)
            .max(predicted.decode)
    {
        return Err(format!(
            "{} predicted overall does not cover phase peaks",
            record.id
        ));
    }
    let observed = required_value(&record.observed_memory, &record.id, "observedMemory")?;
    validate_phase_metrics(observed, &record.id)?;
    if observed.overall.device_bytes > record.hardware.memory_bytes() {
        return Err(format!(
            "{} observed device memory exceeds hardware",
            record.id
        ));
    }
    if let Hardware::Mlx(hardware) = &record.hardware {
        if observed.overall.wired_bytes > hardware.wired_limit_bytes {
            return Err(format!(
                "{} observed wired memory exceeds hardware",
                record.id
            ));
        }
    }

    let scenarios: BTreeMap<_, _> = record
        .scenarios
        .iter()
        .map(|scenario| (scenario.name, scenario))
        .collect();
    let required = [
        ScenarioName::ExactFit,
        ScenarioName::UnknownBudget,
        ScenarioName::StaleEvidence,
        ScenarioName::WarmRepeat,
        ScenarioName::Cancel,
        ScenarioName::Error,
        ScenarioName::Loadability,
        ScenarioName::Overlay,
    ];
    if record.scenarios.len() != required.len() || scenarios.len() != required.len() {
        return Err(format!(
            "{} complete scenarios must be unique and exhaustive",
            record.id
        ));
    }
    for name in required {
        let scenario = scenarios
            .get(&name)
            .ok_or_else(|| format!("{} is missing scenario {name:?}", record.id))?;
        let valid = if name == ScenarioName::Overlay {
            matches!(
                scenario.result,
                ScenarioResult::Passed | ScenarioResult::NotApplicable
            )
        } else {
            scenario.result == ScenarioResult::Passed
        };
        if !valid {
            return Err(format!("{} scenario {name:?} did not pass", record.id));
        }
    }
    for name in [ScenarioName::Cancel, ScenarioName::Error] {
        let scenario = scenarios[&name];
        if scenario.cleanup_verified != Some(true) || scenario.warm_follow_up_passed != Some(true) {
            return Err(format!(
                "{} scenario {name:?} lacks cleanup proof",
                record.id
            ));
        }
    }
    let exact = scenarios[&ScenarioName::ExactFit];
    if exact.predicted_bytes.is_none() || exact.predicted_bytes != exact.effective_budget_bytes {
        return Err(format!(
            "{} exact_fit must exercise predicted == effective budget",
            record.id
        ));
    }
    if scenarios[&ScenarioName::Overlay].result == ScenarioResult::NotApplicable
        && scenarios[&ScenarioName::Overlay]
            .reason
            .as_deref()
            .map_or(true, str::is_empty)
    {
        return Err(format!(
            "{} overlay not_applicable requires a reason",
            record.id
        ));
    }

    let quality = &record.quality;
    if (quality.identical_inputs != Some(true) && quality.identical_latents != Some(true))
        || quality.result != Some(QualityResult::Passed)
    {
        return Err(format!(
            "{} complete quality evidence did not pass",
            record.id
        ));
    }
    let (maximum_error, mean_error, maximum_threshold, mean_threshold) =
        require_complete_quality_fields(quality, &record.id)?;
    if maximum_error > maximum_threshold || mean_error > mean_threshold {
        return Err(format!("{} quality thresholds were exceeded", record.id));
    }
    let mutation = required_value(&record.negative_mutation, &record.id, "negativeMutation")?;
    if mutation.measured != Some(true)
        || mutation.result != Some(NegativeMutationResult::FailedAsExpected)
        || mutation.parameters.is_none()
        || mutation.maximum_error.is_none()
        || mutation.mean_error.is_none()
    {
        return Err(format!(
            "{} complete negative mutation is incomplete",
            record.id
        ));
    }
    if mutation
        .maximum_error
        .is_some_and(|value| value <= maximum_threshold)
        && mutation
            .mean_error
            .is_some_and(|value| value <= mean_threshold)
    {
        return Err(format!(
            "{} negative mutation did not breach a quality threshold",
            record.id
        ));
    }
    if record.loadability.result != LoadabilityResult::Passed
        || !matches!(
            &record.loadability.resolved_path_fingerprint,
            RequiredNullable::Value(value) if !value.is_empty()
        )
    {
        return Err(format!("{} complete loadability did not pass", record.id));
    }
    Ok(())
}

fn validate_negative_complete(record: &EvidenceRecord) -> Result<(), String> {
    let (maximum_threshold, mean_threshold) =
        require_quality_thresholds(&record.quality, &record.id)?;
    let mutation = required_value(&record.negative_mutation, &record.id, "negativeMutation")?;
    if mutation.measured != Some(true)
        || mutation.result != Some(NegativeMutationResult::FailedAsExpected)
        || mutation.parameters.is_none()
        || mutation.maximum_error.is_none()
        || mutation.mean_error.is_none()
    {
        return Err(format!(
            "{} negative_complete mutation is incomplete",
            record.id
        ));
    }
    if mutation.parameters.as_ref() != Some(&record.strategy.parameters) {
        return Err(format!(
            "{} negative_complete parameters do not match its strategy",
            record.id
        ));
    }
    if mutation
        .maximum_error
        .is_some_and(|value| value <= maximum_threshold)
        && mutation
            .mean_error
            .is_some_and(|value| value <= mean_threshold)
    {
        return Err(format!(
            "{} negative_complete did not breach a quality threshold",
            record.id
        ));
    }
    Ok(())
}

fn require_quality_thresholds(quality: &Quality, id: &str) -> Result<(f64, f64), String> {
    match (
        quality.maximum_error_threshold,
        quality.mean_error_threshold,
    ) {
        (Some(maximum_threshold), Some(mean_threshold))
            if quality
                .contract
                .as_deref()
                .is_some_and(|value| !value.is_empty()) =>
        {
            Ok((maximum_threshold, mean_threshold))
        }
        _ => Err(format!("{id} quality threshold evidence is incomplete")),
    }
}

fn require_complete_quality_fields(
    quality: &Quality,
    id: &str,
) -> Result<(f64, f64, f64, f64), String> {
    let (maximum_threshold, mean_threshold) = require_quality_thresholds(quality, id)?;
    match (quality.maximum_error, quality.mean_error) {
        (Some(maximum), Some(mean)) => Ok((maximum, mean, maximum_threshold, mean_threshold)),
        _ => Err(format!("{id} complete quality measurements are incomplete")),
    }
}

fn validate_phase_metrics(metrics: &PhaseMetrics, id: &str) -> Result<(), String> {
    let phases = [&metrics.conditioning, &metrics.denoise, &metrics.decode];
    for phase in phases {
        if phase.allocator_bytes < phase.active_bytes
            || phase.device_bytes < phase.active_bytes
            || phase.wired_bytes < phase.active_bytes
            || phase.reclaimable_bytes > phase.allocator_bytes
        {
            return Err(format!("{id} phase metrics are internally inconsistent"));
        }
    }
    for (overall, maximum) in [
        (
            metrics.overall.active_bytes,
            phases
                .iter()
                .map(|phase| phase.active_bytes)
                .max()
                .unwrap_or(0),
        ),
        (
            metrics.overall.allocator_bytes,
            phases
                .iter()
                .map(|phase| phase.allocator_bytes)
                .max()
                .unwrap_or(0),
        ),
        (
            metrics.overall.device_bytes,
            phases
                .iter()
                .map(|phase| phase.device_bytes)
                .max()
                .unwrap_or(0),
        ),
        (
            metrics.overall.wired_bytes,
            phases
                .iter()
                .map(|phase| phase.wired_bytes)
                .max()
                .unwrap_or(0),
        ),
        (
            metrics.overall.reclaimable_bytes,
            phases
                .iter()
                .map(|phase| phase.reclaimable_bytes)
                .max()
                .unwrap_or(0),
        ),
    ] {
        if overall < maximum {
            return Err(format!(
                "{id} overall phase metrics do not cover phase peaks"
            ));
        }
    }
    Ok(())
}

fn required_value<'a, T>(
    value: &'a RequiredNullable<T>,
    id: &str,
    field: &str,
) -> Result<&'a T, String> {
    match value {
        RequiredNullable::Value(value) => Ok(value),
        RequiredNullable::Null => Err(format!("{id} complete {field} cannot be null")),
    }
}

fn require_nonempty(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty() {
        Err(format!("{field} must be nonempty"))
    } else {
        Ok(())
    }
}

fn has_prefixed_hex(value: &str, prefix: &str, digits: usize) -> bool {
    value
        .strip_prefix(prefix)
        .is_some_and(|suffix| suffix.len() == digits && is_lower_hex(suffix))
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_rfc3339_datetime(value: &str) -> bool {
    let Some((date, time_and_offset)) = value.split_once('T') else {
        return false;
    };
    let mut date_fields = date.split('-');
    let (Some(year), Some(month), Some(day), None) = (
        date_fields.next(),
        date_fields.next(),
        date_fields.next(),
        date_fields.next(),
    ) else {
        return false;
    };
    let (Ok(year), Ok(month), Ok(day)) = (
        year.parse::<i32>(),
        month.parse::<u32>(),
        day.parse::<u32>(),
    ) else {
        return false;
    };
    if date.len() != 10 || !(1..=12).contains(&month) {
        return false;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let max_day = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if !(1..=max_day).contains(&day) {
        return false;
    }

    let (time, valid_offset) = if let Some(time) = time_and_offset.strip_suffix('Z') {
        (time, true)
    } else {
        let Some(offset_index) = time_and_offset
            .char_indices()
            .skip(1)
            .find_map(|(index, character)| matches!(character, '+' | '-').then_some(index))
        else {
            return false;
        };
        let (time, offset) = time_and_offset.split_at(offset_index);
        let valid = offset.len() == 6
            && offset.as_bytes()[3] == b':'
            && offset[1..3].parse::<u32>().is_ok_and(|hours| hours <= 23)
            && offset[4..6]
                .parse::<u32>()
                .is_ok_and(|minutes| minutes <= 59);
        (time, valid)
    };
    if !valid_offset {
        return false;
    }
    let (clock, fraction_valid) = match time.split_once('.') {
        Some((clock, fraction)) => (
            clock,
            !fraction.is_empty() && fraction.bytes().all(|byte| byte.is_ascii_digit()),
        ),
        None => (time, true),
    };
    if !fraction_valid || clock.len() != 8 {
        return false;
    }
    let mut clock_fields = clock.split(':');
    let (Some(hour), Some(minute), Some(second), None) = (
        clock_fields.next(),
        clock_fields.next(),
        clock_fields.next(),
        clock_fields.next(),
    ) else {
        return false;
    };
    matches!(
        (
            hour.parse::<u32>(),
            minute.parse::<u32>(),
            second.parse::<u32>()
        ),
        (Ok(0..=23), Ok(0..=59), Ok(0..=59))
    )
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map, Value};

    use super::{
        load_bundle, load_packaged_bundle, Backend, BundleLoad, BundleLoadError,
        CalibrationBinding, EvidenceBundle, EvidenceQuery, EvidenceVerdict, Geometry, LoadShapeKey,
        StaleBundleReason, StaleEvidenceReason, StrategyRung, MEMORY_CALIBRATION_ABI,
    };

    fn phase(value: u64) -> Value {
        json!({
            "activeBytes": value,
            "allocatorBytes": value + 10,
            "deviceBytes": value + 20,
            "wiredBytes": value + 30,
            "reclaimableBytes": 0
        })
    }

    fn complete_record() -> Value {
        json!({
            "id": "imc-aaaaaaaaaaaaaaaaaaaa",
            "logicalCaseId": "implan-bbbbbbbbbbbbbbbbbbbb",
            "status": "complete",
            "evidenceScope": "authoritative",
            "backend": "candle",
            "repositories": {
                "sceneWorks": {
                    "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "dirty": false,
                    "matrixSourceRevision": "source-tree:1111111"
                },
                "inference": {
                    "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "dirty": false
                }
            },
            "hardware": {
                "probe": "nvidia-smi",
                "memoryBytes": 50000,
                "deviceId": "0",
                "name": "Fixture CUDA",
                "computeCapability": "9.0",
                "driverVersion": "999.1",
                "runtimeVersion": "12.8"
            },
            "artifact": {
                "repository": "SceneWorks/fixture",
                "resolvedRevision": "cccccccccccccccccccccccccccccccccccccccc",
                "variant": "q4"
            },
            "target": {
                "modelId": "fixture_model",
                "provider": "fixture_provider",
                "tier": "q4",
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            },
            "fixture": "fixture-seed42",
            "strategy": {
                "rung": "bounded_decode",
                "engagedRungs": ["resident", "bounded_decode"],
                "parameters": { "decodeTileEdge": 512, "decodeOverlap": 128 }
            },
            "loadShape": "eager_materialization",
            "sweep": {
                "axes": [{ "parameter": "decodeTileEdge", "testedValues": [384, 512] }],
                "cases": [
                    {
                        "parameters": { "decodeTileEdge": 384, "decodeOverlap": 128 },
                        "result": "passed"
                    },
                    {
                        "parameters": { "decodeTileEdge": 512, "decodeOverlap": 128 },
                        "result": "passed"
                    },
                    {
                        "parameters": { "decodeTileEdge": 256, "decodeOverlap": 32 },
                        "result": "failed"
                    }
                ],
                "rangeVerified": true
            },
            "scenarios": [
                {
                    "name": "exact_fit", "result": "passed",
                    "predictedBytes": 200, "effectiveBudgetBytes": 200
                },
                { "name": "unknown_budget", "result": "passed" },
                { "name": "stale_evidence", "result": "passed" },
                { "name": "warm_repeat", "result": "passed" },
                {
                    "name": "cancel", "result": "passed",
                    "cleanupVerified": true, "warmFollowUpPassed": true
                },
                {
                    "name": "error", "result": "passed",
                    "cleanupVerified": true, "warmFollowUpPassed": true
                },
                { "name": "loadability", "result": "passed" },
                { "name": "overlay", "result": "not_applicable", "reason": "no overlay" }
            ],
            "predictedPeakBytes": {
                "conditioning": 100, "denoise": 200, "decode": 150, "overall": 200
            },
            "observedMemory": {
                "conditioning": phase(100),
                "denoise": phase(200),
                "decode": phase(150),
                "overall": phase(200)
            },
            "quality": {
                "contract": "tolerance",
                "identicalLatents": true,
                "result": "passed",
                "maximumError": 0.01,
                "meanError": 0.001,
                "maximumErrorThreshold": 0.08,
                "meanErrorThreshold": 0.01
            },
            "negativeMutation": {
                "parameters": { "decodeTileEdge": 256, "decodeOverlap": 32 },
                "measured": true,
                "result": "failed_as_expected",
                "maximumError": 0.09,
                "meanError": 0.02
            },
            "loadability": {
                "result": "passed",
                "resolvedPathFingerprint": "fixture@resolved:q4"
            },
            "diagnostics": {
                "adapter": "fixture",
                "execution": "executed",
                "blockers": [],
                "measurements": [{ "name": "peak", "unit": "bytes", "value": 200 }]
            },
            "calibrationFingerprint": "fixture-formula-v2",
            "capturedAt": "2026-07-28T12:00:00Z",
            "harnessVersion": "sceneworks-memory-v4"
        })
    }

    fn bundle(record: Value) -> String {
        json!({
            "schemaVersion": 4,
            "harnessVersion": "sceneworks-memory-v4",
            "records": [record]
        })
        .to_string()
    }

    fn loaded_bundle() -> EvidenceBundle {
        match load_bundle(&bundle(complete_record())).expect("valid fixture") {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("unexpected stale fixture: {reason:?}"),
        }
    }

    fn mlx_record(total: u64, memory_limit: u64, wired_limit: u64) -> Value {
        let mut record = complete_record();
        record["backend"] = json!("mlx");
        record["hardware"] = json!({
            "probe": "mlx-rs",
            "memoryBytes": total,
            "model": "Fixture Mac",
            "chip": "Fixture Silicon",
            "osVersion": "26.0",
            "metalDevice": "Fixture GPU",
            "mlxMemoryLimitBytes": memory_limit,
            "wiredLimitBytes": wired_limit
        });
        record
    }

    #[test]
    fn mlx_admission_envelope_derives_foreign_demand_and_keeps_observed_distinct() {
        let gib = 1024_u64.pow(3);
        let small = match load_bundle(&bundle(mlx_record(8 * gib, 6 * gib, 7 * gib)))
            .expect("valid small-host evidence")
        {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("unexpected stale fixture: {reason:?}"),
        };
        let small = small.records[0]
            .mlx_admission_envelope()
            .expect("MLX complete record");
        assert_eq!(small.foreign_reserve_bytes, 2 * gib);
        assert_eq!(small.observed_non_reclaimable_wired_bytes, 230);
        assert_eq!(small.peak_bytes, 230);
        let required = 2 * gib + 230;
        assert_eq!(small.required_host_bytes(), required);
        assert!(small.fits_host_bytes(required), "exact equality fits");
        assert!(
            !small.fits_host_bytes(required - 1),
            "one byte below the exact host requirement must fail"
        );

        let mid = match load_bundle(&bundle(mlx_record(32 * gib, 24 * gib, 20 * gib)))
            .expect("valid mid-host evidence")
        {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("unexpected stale fixture: {reason:?}"),
        };
        let mid = mid.records[0]
            .mlx_admission_envelope()
            .expect("MLX complete record");
        assert_eq!(mid.foreign_reserve_bytes, 12 * gib);
        assert_ne!(small.foreign_reserve_bytes, mid.foreign_reserve_bytes);
        assert_eq!(
            mid.observed_non_reclaimable_wired_bytes, 230,
            "observed telemetry is not overwritten by a predicted/envelope maximum"
        );
    }

    fn exact_query() -> EvidenceQuery {
        EvidenceQuery {
            backend: Backend::Candle,
            model_id: "fixture_model".to_owned(),
            provider: "fixture_provider".to_owned(),
            tier: "q4".to_owned(),
            mode: "text_to_image".to_owned(),
            overlay: "none".to_owned(),
            geometry: Geometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            rung: StrategyRung::BoundedDecode,
            parameters: Map::from_iter([
                ("decodeTileEdge".to_owned(), json!(512)),
                ("decodeOverlap".to_owned(), json!(128)),
            ]),
            calibration: CalibrationBinding {
                abi: MEMORY_CALIBRATION_ABI,
                load_shape: LoadShapeKey::EagerMaterialization,
                fingerprint: "fixture-formula-v2".to_owned(),
                scene_works_revision: "a".repeat(40),
                matrix_source_revision: "source-tree:1111111".to_owned(),
                inference_revision: "b".repeat(40),
                artifact_repository: "SceneWorks/fixture".to_owned(),
                artifact_resolved_revision: "c".repeat(40),
                artifact_variant: "q4".to_owned(),
                resolved_path_fingerprint: "fixture@resolved:q4".to_owned(),
            },
        }
    }

    #[test]
    fn real_packaged_bundle_loads_promoted_records_and_unrelated_target_stays_unknown() {
        assert_eq!(
            load_packaged_bundle().expect("compiled bundle must parse"),
            BundleLoad::Stale(StaleBundleReason::SchemaVersion { found: Some(3) })
        );
    }

    #[test]
    fn schema_and_harness_drift_are_stale_but_bad_json_is_an_error() {
        assert_eq!(
            load_bundle(
                r#"{"schemaVersion":2,"harnessVersion":"sceneworks-memory-v3","records":[]}"#
            )
            .expect("version drift is not a parse failure"),
            BundleLoad::Stale(StaleBundleReason::SchemaVersion { found: Some(2) })
        );
        assert_eq!(
            load_bundle(r#"{"schemaVersion":4,"harnessVersion":"old","records":[]}"#)
                .expect("harness drift is not a parse failure"),
            BundleLoad::Stale(StaleBundleReason::HarnessVersion {
                found: Some("old".to_owned())
            })
        );
        let mut missing_load_shape = complete_record();
        missing_load_shape
            .as_object_mut()
            .expect("fixture object")
            .remove("loadShape");
        assert!(matches!(
            load_bundle(&bundle(missing_load_shape)),
            Err(BundleLoadError::Json(_))
        ));
        let mut record_stale = complete_record();
        record_stale["harnessVersion"] = json!("old-record");
        assert_eq!(
            load_bundle(&bundle(record_stale)).expect("record harness drift is stale"),
            BundleLoad::Stale(StaleBundleReason::HarnessVersion {
                found: Some("old-record".to_owned())
            })
        );
        assert!(matches!(
            load_bundle(r#"{"harnessVersion":"sceneworks-memory-v4","records":[]}"#),
            Err(BundleLoadError::Json(_))
        ));
        let mut missing_record_harness = complete_record();
        missing_record_harness
            .as_object_mut()
            .expect("fixture object")
            .remove("harnessVersion");
        assert!(matches!(
            load_bundle(&bundle(missing_record_harness)),
            Err(BundleLoadError::Json(_))
        ));
        let mut legacy_without_composition = complete_record();
        legacy_without_composition["strategy"]
            .as_object_mut()
            .expect("strategy object")
            .remove("engagedRungs");
        assert!(matches!(
            load_bundle(&bundle(legacy_without_composition)),
            Err(BundleLoadError::Json(_))
        ));
        let mut noncanonical_composition = complete_record();
        noncanonical_composition["strategy"]["engagedRungs"] =
            json!(["bounded_decode", "resident"]);
        assert!(matches!(
            load_bundle(&bundle(noncanonical_composition)),
            Err(BundleLoadError::Invalid(message)) if message.contains("canonical set")
        ));
        let mut wrong_typed_record_harness = complete_record();
        wrong_typed_record_harness["harnessVersion"] = json!(3);
        assert!(matches!(
            load_bundle(&bundle(wrong_typed_record_harness)),
            Err(BundleLoadError::Json(_))
        ));
        assert!(matches!(
            load_bundle(&bundle(json!(7))),
            Err(BundleLoadError::Json(_))
        ));
        assert!(matches!(load_bundle("{"), Err(BundleLoadError::Json(_))));
    }

    #[test]
    fn both_hardware_contract_arms_parse_and_unknown_fields_fail_closed() {
        let mut mlx = complete_record();
        mlx["hardware"] = json!({
            "probe": "system_profiler",
            "memoryBytes": 128000,
            "model": "Mac16,5",
            "chip": "Apple M4 Max",
            "osVersion": "15.7",
            "metalDevice": "Apple M4 Max",
            "mlxMemoryLimitBytes": 96000,
            "wiredLimitBytes": 80000
        });
        mlx["backend"] = json!("mlx");
        assert!(matches!(
            load_bundle(&bundle(mlx)),
            Ok(BundleLoad::Ready(_))
        ));

        let mut extra = complete_record();
        extra["target"]["futureField"] = json!(true);
        assert!(matches!(
            load_bundle(&bundle(extra)),
            Err(BundleLoadError::Json(_))
        ));

        let mut mismatch = complete_record();
        mismatch["backend"] = json!("mlx");
        assert!(matches!(
            load_bundle(&bundle(mismatch)),
            Err(BundleLoadError::Invalid(_))
        ));
    }

    #[test]
    fn conditional_complete_contract_is_enforced_by_the_reader() {
        for (field, value) in [
            ("predictedPeakBytes", Value::Null),
            ("observedMemory", Value::Null),
            ("negativeMutation", Value::Null),
        ] {
            let mut record = complete_record();
            record[field] = value;
            assert!(matches!(
                load_bundle(&bundle(record)),
                Err(BundleLoadError::Invalid(_))
            ));
        }
        let mut dirty = complete_record();
        dirty["repositories"]["sceneWorks"]["dirty"] = json!(true);
        assert!(matches!(
            load_bundle(&bundle(dirty)),
            Err(BundleLoadError::Invalid(_))
        ));
    }

    #[test]
    fn schema_minimal_negative_complete_quality_loads() {
        let mut record = complete_record();
        record["status"] = json!("negative_complete");
        record["predictedPeakBytes"] = Value::Null;
        record["observedMemory"] = Value::Null;
        record["quality"] = json!({
            "contract": "tolerance",
            "maximumErrorThreshold": 0.08,
            "meanErrorThreshold": 0.01
        });
        let strategy_parameters = record["strategy"]["parameters"].clone();
        record["negativeMutation"]["parameters"] = strategy_parameters;

        assert!(
            matches!(load_bundle(&bundle(record)), Ok(BundleLoad::Ready(_))),
            "negative_complete requires thresholds, not positive-case measured quality"
        );
    }

    #[test]
    fn exact_envelope_verifies_and_nonmatching_records_are_unknown() {
        let bundle = loaded_bundle();
        assert!(matches!(
            bundle.evidence_for(&exact_query()),
            EvidenceVerdict::Verified(_)
        ));

        let mut query = exact_query();
        query.model_id = "other".to_owned();
        assert_eq!(bundle.evidence_for(&query), EvidenceVerdict::Unknown);
    }

    #[test]
    fn fingerprint_inference_and_abi_mutations_are_stale_but_source_revisions_are_provenance() {
        let bundle = loaded_bundle();

        let mut fingerprint = exact_query();
        fingerprint.calibration.fingerprint.push_str("-mutated");
        assert_eq!(
            bundle.evidence_for(&fingerprint),
            EvidenceVerdict::Stale(StaleEvidenceReason::CalibrationFingerprint)
        );

        let mut scene_works = exact_query();
        scene_works.calibration.scene_works_revision = "c".repeat(40);
        assert!(matches!(
            bundle.evidence_for(&scene_works),
            EvidenceVerdict::Verified(_)
        ));

        let mut inference = exact_query();
        inference.calibration.inference_revision = "d".repeat(40);
        assert_eq!(
            bundle.evidence_for(&inference),
            EvidenceVerdict::Stale(StaleEvidenceReason::InferenceRevision)
        );

        let mut matrix = exact_query();
        matrix.calibration.matrix_source_revision = "source-tree:2222222".to_owned();
        assert!(matches!(
            bundle.evidence_for(&matrix),
            EvidenceVerdict::Verified(_)
        ));

        let mut artifact_repository = exact_query();
        artifact_repository.calibration.artifact_repository = "other/repo".to_owned();
        assert_eq!(
            bundle.evidence_for(&artifact_repository),
            EvidenceVerdict::Stale(StaleEvidenceReason::ArtifactRepository)
        );

        let mut artifact_revision = exact_query();
        artifact_revision.calibration.artifact_resolved_revision = "d".repeat(40);
        assert_eq!(
            bundle.evidence_for(&artifact_revision),
            EvidenceVerdict::Stale(StaleEvidenceReason::ArtifactResolvedRevision)
        );

        let mut artifact_variant = exact_query();
        artifact_variant.calibration.artifact_variant = "q8".to_owned();
        assert_eq!(
            bundle.evidence_for(&artifact_variant),
            EvidenceVerdict::Stale(StaleEvidenceReason::ArtifactVariant)
        );

        let mut path = exact_query();
        path.calibration.resolved_path_fingerprint = "different".to_owned();
        assert_eq!(
            bundle.evidence_for(&path),
            EvidenceVerdict::Stale(StaleEvidenceReason::ResolvedPathFingerprint)
        );

        let mut abi = exact_query();
        abi.calibration.abi += 1;
        assert_eq!(
            bundle.evidence_for(&abi),
            EvidenceVerdict::Stale(StaleEvidenceReason::CalibrationAbi)
        );
    }

    #[test]
    fn geometry_and_sweep_require_exact_containment() {
        let bundle = loaded_bundle();

        let mut geometry = exact_query();
        geometry.geometry.width = 512;
        assert_eq!(
            bundle.evidence_for(&geometry),
            EvidenceVerdict::OutOfEnvelope
        );

        let mut parameters = exact_query();
        parameters
            .parameters
            .insert("decodeTileEdge".to_owned(), json!([384, 512]));
        assert_eq!(
            bundle.evidence_for(&parameters),
            EvidenceVerdict::OutOfEnvelope
        );

        let mut unexecuted = exact_query();
        unexecuted
            .parameters
            .insert("decodeTileEdge".to_owned(), json!(640));
        assert_eq!(
            bundle.evidence_for(&unexecuted),
            EvidenceVerdict::OutOfEnvelope
        );
    }
}
