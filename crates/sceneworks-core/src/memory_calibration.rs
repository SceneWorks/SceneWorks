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

/// Schema v4 adds the required per-record `loadShape` axis. v3 bundles (including the currently
/// packaged production evidence, whose records were measured before the shape was typed) load as
/// `BundleLoad::Stale` and every consumer falls back to the legacy admission path until the
/// evidence is re-collected under the new harness.
pub const MEMORY_CALIBRATION_SCHEMA_VERSION: u32 = 4;
pub const MEMORY_CALIBRATION_HARNESS_VERSION: &str = "sceneworks-memory-v5";
/// ABI paired by the manifest/query side of the reader.
///
/// Callers must supply the manifest's ABI together with its fingerprint. Exact source revisions
/// remain captured provenance; the ABI/fingerprint pair owns SceneWorks invalidation without
/// rewriting the already-promoted producer contract.
///
/// ABI 3 tracks `gen_core::MEMORY_CALIBRATION_ABI`: calibration identities, run
/// contexts, evidence keys, and therefore these receipts are keyed by the typed materialization
/// [`LoadShapeKey`] plus exact request reference cardinality. Older records are intentionally
/// stale because neither edit cardinality nor eager/deferred materialization is interchangeable.
/// A worker-side lockstep test asserts this constant equals gen-core's.
pub const MEMORY_CALIBRATION_ABI: u32 = 3;
/// The per-provider inference compile-closure digests, compiled in (sc-17774).
///
/// ONE mechanism, applied identically to every model. Before this, each lane carried its own frozen
/// revision constant — `KREA_CONTROL_INFERENCE_REVISION`, `KREA_TURBO_INFERENCE_REVISION`,
/// `INFERENCE_CONTRACT_REVISION`, `INFERENCE_RUNTIME_REVISION` — and `flux2_dev` additionally had a
/// hand-audited one-shot compatibility hatch. All of them are gone; every lane now asks this table
/// for the provider it is actually admitting.
pub const PACKAGED_INFERENCE_PROVIDER_CLOSURES: &str =
    include_str!("../../../config/inference-provider-closures.json");

/// The live closure digest for one `(backend, provider)` lane, or `None` when it is not declared.
///
/// Keyed by BOTH because a provider id is not unique: `krea_2_turbo_control` exists on mlx
/// (`mlx-gen-krea`) and on candle (`candle-gen-krea`), which are different code paths that must
/// never be compared against each other.
///
/// `None` is a real answer and callers must fail closed on it rather than admitting: an undeclared
/// lane means nobody derived what code its measurements were taken against.
pub fn packaged_closure_digest(backend: &str, provider: &str) -> Option<String> {
    serde_json::from_str::<Value>(PACKAGED_INFERENCE_PROVIDER_CLOSURES)
        .ok()?
        .get("providers")?
        .get(format!("{backend}:{provider}"))?
        .get("digest")?
        .as_str()
        .map(str::to_owned)
}

pub const PACKAGED_MEMORY_CALIBRATION_EVIDENCE: &str =
    include_str!("../../../docs/generated/memory-calibration-evidence.json");

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvidenceBundle {
    pub schema_version: u32,
    pub harness_version: String,
    #[serde(default)]
    pub source_sessions: Vec<SourceSession>,
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
    /// Exact intra-phase materialization shape this record was measured under (schema v4 /
    /// calibration ABI 2). Mirrors `gen_core::LoadShape`; this crate keeps its own spelling
    /// because it deliberately has no gen-core dependency.
    pub load_shape: LoadShapeKey,
    pub sweep: Sweep,
    pub scenarios: Vec<Scenario>,
    pub predicted_peak_bytes: RequiredNullable<PredictedPeakBytes>,
    pub observed_memory: RequiredNullable<ObservedMemory>,
    pub quality: Quality,
    pub negative_mutation: RequiredNullable<NegativeMutation>,
    pub loadability: Loadability,
    pub diagnostics: Option<Diagnostics>,
    pub derivation: Option<EvidenceDerivation>,
    pub source_provenance: Option<SourceProvenance>,
    pub calibration_fingerprint: String,
    pub captured_at: String,
    pub harness_version: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceSession {
    pub id: String,
    pub kind: SourceSessionKind,
    pub command: String,
    pub source_path: String,
    pub captured_at: String,
    pub repositories: Repositories,
    pub hardware: SourceHardware,
    pub target: Option<SourceTarget>,
    pub stdout_sha256: String,
    pub inputs: Vec<SourceInput>,
    pub outputs: Vec<SourceOutput>,
    pub claims: Vec<SourceClaim>,
    pub result: SourceResult,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceInput {
    pub role: SourceInputRole,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub repository: String,
    pub resolved_revision: String,
    pub variant: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceInputRole {
    Base,
    Control,
    Adapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceSessionKind {
    PhysicalCuda,
    PhysicalMlx,
    UnitTest,
    StaticAnalysis,
    Comparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceProvenance {
    PhysicalMlxV1,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceHardware {
    pub probe: String,
    pub memory_bytes: u64,
    /// Source-session hardware is the schema's extensible `hardwareBase`, not one of the closed
    /// production-record hardware arms. Preserve backend-specific probe receipts (CUDA identity,
    /// driver/runtime versions, or future MLX metadata) while validation continues to require the
    /// two portable base fields. Keeping the extensions flattened mirrors JSON Schema's deliberate
    /// lack of `additionalProperties: false` on `hardwareBase` without loosening `SourceSession`.
    #[serde(flatten, default)]
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceTarget {
    pub tier: String,
    pub mode: String,
    pub overlay: String,
    pub rung: StrategyRung,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceOutput {
    pub role: Option<SourceOutputRole>,
    pub path: String,
    pub sha256: String,
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceOutputRole {
    Request,
    SelectedRgb,
    ReferenceRgb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceClaim {
    Memory,
    Quality,
    NegativeMutation,
    Lifecycle,
    Loadability,
    Overlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceResult {
    Passed,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvidenceDerivation {
    pub memory: DerivationRef,
    pub quality: DerivationRef,
    pub negative_mutation: DerivationRef,
    pub lifecycle: DerivationRef,
    pub loadability: DerivationRef,
    pub overlay: DerivationRef,
    pub justification: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DerivationRef {
    pub kind: DerivationKind,
    pub source_session_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivationKind {
    Direct,
    ConservativeUpperBound,
    IdenticalComponent,
    SharedImplementation,
}

/// Materialization shape a record was measured under. The two spellings are the persisted-JSON
/// forms of `gen_core::LoadShape`'s variants.
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
    RuntimeComplete,
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
    /// Digest of the measured provider's inference compile closure at `revision` (sc-17774).
    ///
    /// This — not `revision` — is the currency term. `revision` stays as capture provenance so a
    /// record can still say where it came from, but comparing it was what demoted every provider's
    /// measurements on any inference commit, including commits to a different model entirely.
    ///
    /// Optional in the type so a bundle can be parsed and diagnosed; [`EvidenceBundle::evidence_for`]
    /// refuses a record without one rather than falling back to revision equality.
    pub closure_digest: Option<String>,
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
    pub inventory_sha256: Option<String>,
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
#[serde(untagged)]
pub enum PredictedPeakBytes {
    Full(FullPredictedPeakBytes),
    RuntimeOverall(RuntimePredictedPeakBytes),
}

impl PredictedPeakBytes {
    pub fn overall(&self) -> u64 {
        match self {
            Self::Full(value) => value.overall,
            Self::RuntimeOverall(value) => value.overall,
        }
    }

    pub fn full(&self) -> Option<&FullPredictedPeakBytes> {
        match self {
            Self::Full(value) => Some(value),
            Self::RuntimeOverall(_) => None,
        }
    }

    pub fn full_mut(&mut self) -> Option<&mut FullPredictedPeakBytes> {
        match self {
            Self::Full(value) => Some(value),
            Self::RuntimeOverall(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FullPredictedPeakBytes {
    pub conditioning: u64,
    pub denoise: u64,
    pub decode: u64,
    pub overall: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimePredictedPeakBytes {
    pub overall: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ObservedMemory {
    Full(PhaseMetrics),
    RuntimeOverall(RuntimeObservedMemory),
}

impl ObservedMemory {
    pub fn full(&self) -> Option<&PhaseMetrics> {
        match self {
            Self::Full(value) => Some(value),
            Self::RuntimeOverall(_) => None,
        }
    }

    pub fn full_mut(&mut self) -> Option<&mut PhaseMetrics> {
        match self {
            Self::Full(value) => Some(value),
            Self::RuntimeOverall(_) => None,
        }
    }

    pub fn overall_device_or_active_bytes(&self) -> u64 {
        match self {
            Self::Full(value) => value.overall.device_bytes,
            Self::RuntimeOverall(value) => value.overall.active_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeObservedMemory {
    pub overall: RuntimeObservedOverall,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeObservedOverall {
    pub active_bytes: u64,
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
    pub root_mean_square_error: Option<f64>,
    pub maximum_error_threshold: Option<f64>,
    pub mean_error_threshold: Option<f64>,
    pub root_mean_square_error_threshold: Option<f64>,
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
    /// Materialization shape the manifest's opt-in claims its receipts were measured under.
    /// Compared record-by-record; a mismatch is [`StaleEvidenceReason::LoadShape`].
    pub load_shape: LoadShapeKey,
    pub fingerprint: String,
    pub scene_works_revision: String,
    pub matrix_source_revision: String,
    /// Capture provenance only — NEVER compared (sc-17774). See [`Self::inference_closure_digest`].
    pub inference_revision: String,
    /// The provider compile-closure digest this calibration is in force for (sc-17774).
    ///
    /// One mechanism for every model: a record is current exactly when the closure of the provider
    /// it measured is unchanged. A change to any other model's code path cannot move this value, so
    /// it cannot demote this calibration.
    pub inference_closure_digest: String,
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
    /// The measured provider's own inference compile closure moved (sc-17774).
    ///
    /// Replaces the former `InferenceRevision`, which fired whenever the inference pin moved at all
    /// — including for a commit to an unrelated model.
    InferenceClosure,
    /// A record carried no closure digest, so currency cannot be decided (sc-17774).
    ///
    /// Separate from [`Self::InferenceClosure`] on purpose: "we could not tell" must never be
    /// reported as "the code changed", and neither may silently fall back to revision equality.
    MissingClosureDigest,
    /// A current Qwen q4/bf16 record omitted the physical MLX receipt required by SC-18353.
    PhysicalMlxProvenance,
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
                matches!(
                    record.status,
                    RecordStatus::Complete | RecordStatus::RuntimeComplete
                ) && record.evidence_scope == EvidenceScope::Authoritative
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
            } else if record.repositories.inference.closure_digest.is_none() {
                Some(StaleEvidenceReason::MissingClosureDigest)
            } else if record.backend == Backend::Mlx
                && record.target.model_id == "qwen_image"
                && matches!(record.target.tier.as_str(), "q4" | "bf16")
                && record.repositories.inference.closure_digest.as_deref()
                    == Some(query.calibration.inference_closure_digest.as_str())
                && record.source_provenance != Some(SourceProvenance::PhysicalMlxV1)
            {
                Some(StaleEvidenceReason::PhysicalMlxProvenance)
            } else if record.repositories.inference.closure_digest.as_deref()
                != Some(query.calibration.inference_closure_digest.as_str())
            {
                // sc-17774: the provider's own compile closure, not the inference pin. The pin
                // comparison this replaces demoted every calibrated provider on any inference
                // commit — 2812 of 2812 non-merge commits over the 90 days to `fbb00d6b`.
                Some(StaleEvidenceReason::InferenceClosure)
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
    /// Physical unified memory on the capture host. The foreign reserve is a HOST-policy share of
    /// this capacity, not model memory: carrying its absolute 128 GiB-host byte count onto a 48 GiB
    /// host would reserve memory that does not exist there.
    pub capture_host_bytes: u64,
    pub foreign_reserve_bytes: u64,
}

impl MlxAdmissionEnvelope {
    /// Smallest physical unified-memory size that can satisfy this exact cell under the captured
    /// host-reserve policy.
    ///
    /// Below the capture host, runtime preserves the capture's foreign-reserve *ratio*. Solving
    /// `peak + ceil(reserve * host / capture) <= host` gives
    /// `ceil(peak * capture / (capture - reserve))`. At and above the capture host, runtime keeps
    /// the captured reserve absolute instead of speculating that foreign demand grows forever.
    ///
    /// This is the shared runtime/UI bridge. Intermediate products use `u128`; corrupt,
    /// impossible, or unrepresentable evidence fails conservatively.
    pub fn required_host_bytes(self) -> u64 {
        self.required_host_bytes_for_peak(self.peak_bytes)
    }

    /// The static minimum host boundary for an alternate effective peak, such as a stale-widened
    /// peak. This keeps the reserve policy identical while allowing the caller's evidence grading
    /// policy to remain authoritative.
    pub fn required_host_bytes_for_peak(self, peak_bytes: u64) -> u64 {
        if self.capture_host_bytes == 0 {
            return u64::MAX;
        }

        let absolute_requirement = peak_bytes.saturating_add(self.foreign_reserve_bytes);
        if self.foreign_reserve_bytes >= self.capture_host_bytes {
            return absolute_requirement;
        }

        let denominator = u128::from(self.capture_host_bytes - self.foreign_reserve_bytes);
        let numerator = u128::from(peak_bytes) * u128::from(self.capture_host_bytes);
        let proportional_requirement =
            numerator.saturating_add(denominator.saturating_sub(1)) / denominator;
        let proportional_requirement = u64::try_from(proportional_requirement).unwrap_or(u64::MAX);

        if proportional_requirement <= self.capture_host_bytes {
            proportional_requirement
        } else {
            absolute_requirement
        }
    }

    pub fn fits_host_bytes(self, host_bytes: u64) -> bool {
        self.required_host_bytes() <= host_bytes
    }

    /// Foreign host-policy reserve normalized to `host_bytes`, rounded up so a smaller live host is
    /// never given more process memory than the capture host's process-ceiling ratio permits.
    ///
    /// MLX still receives the resulting absolute process ceiling at runtime. This only translates
    /// the capture host's reserve ratio; it does not weaken the allocator limit or reinterpret the
    /// model peak.
    pub fn foreign_reserve_for_host_bytes(self, host_bytes: u64) -> u64 {
        if self.capture_host_bytes == 0 {
            return u64::MAX;
        }
        // A measurement cannot prove that foreign demand grows on a larger machine. Preserve its
        // captured absolute reserve there; normalization exists to avoid carrying an impossible
        // large-host reserve onto a smaller host.
        if host_bytes >= self.capture_host_bytes {
            return self.foreign_reserve_bytes;
        }
        let numerator = u128::from(self.foreign_reserve_bytes) * u128::from(host_bytes);
        let denominator = u128::from(self.capture_host_bytes);
        let scaled = numerator.saturating_add(denominator.saturating_sub(1)) / denominator;
        u64::try_from(scaled).unwrap_or(u64::MAX)
    }

    pub fn required_host_bytes_for(self, host_bytes: u64) -> u64 {
        self.peak_bytes
            .saturating_add(self.foreign_reserve_for_host_bytes(host_bytes))
    }

    pub fn fits_scaled_host_bytes(self, host_bytes: u64) -> bool {
        self.required_host_bytes_for(host_bytes) <= host_bytes
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
        let predicted = predicted.full()?;
        let RequiredNullable::Value(observed) = &self.observed_memory else {
            return None;
        };
        let observed = observed.full()?;
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
            capture_host_bytes: hardware.memory_bytes,
            foreign_reserve_bytes,
        })
    }
}

fn validate_bundle(bundle: &EvidenceBundle) -> Result<(), String> {
    let mut sessions = BTreeMap::new();
    let mut inventory_inputs = BTreeMap::<(SourceInputRole, String), &SourceInput>::new();
    for session in &bundle.source_sessions {
        validate_source_session(session)?;
        if sessions.insert(session.id.as_str(), session).is_some() {
            return Err(format!("duplicate source session {}", session.id));
        }
        if session.kind == SourceSessionKind::StaticAnalysis && session.target.is_none() {
            for input in &session.inputs {
                let key = (input.role, input.variant.clone());
                if let Some(existing) = inventory_inputs.insert(key, input) {
                    if existing != input {
                        return Err(format!(
                            "inventory sessions disagree on exact {:?}/{} input identity",
                            input.role, input.variant
                        ));
                    }
                }
            }
        }
    }
    for record in &bundle.records {
        validate_record(record)?;
        validate_derivation(record, &sessions, &inventory_inputs)?;
    }
    Ok(())
}

fn validate_source_session(session: &SourceSession) -> Result<(), String> {
    if !has_prefixed_hex(&session.id, "ims-", 20) {
        return Err("source session id must be ims- plus 20 lowercase hex digits".to_owned());
    }
    require_nonempty(&session.command, "sourceSession.command")?;
    require_nonempty(&session.source_path, "sourceSession.sourcePath")?;
    if !is_normalized_calibration_path(&session.source_path)
        || !session.source_path.ends_with(".log")
    {
        return Err(format!("{} has an invalid sourcePath", session.id));
    }
    require_nonempty(&session.hardware.probe, "sourceSession.hardware.probe")?;
    if session.hardware.memory_bytes == 0 {
        return Err(format!(
            "{} hardware.memoryBytes must be positive",
            session.id
        ));
    }
    if !is_rfc3339_datetime(&session.captured_at) {
        return Err(format!("{} capturedAt is not RFC 3339", session.id));
    }
    if !is_sha256(&session.stdout_sha256) {
        return Err(format!(
            "{} stdoutSha256 must be lowercase SHA-256",
            session.id
        ));
    }
    if session.claims.is_empty()
        || session
            .claims
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != session.claims.len()
    {
        return Err(format!("{} claims must be nonempty and unique", session.id));
    }
    validate_git_state(&session.repositories.scene_works, false, &session.id)?;
    validate_git_state(&session.repositories.inference, false, &session.id)?;
    for input in &session.inputs {
        require_nonempty(&input.path, "sourceSession.inputs.path")?;
        require_nonempty(&input.repository, "sourceSession.inputs.repository")?;
        require_nonempty(
            &input.resolved_revision,
            "sourceSession.inputs.resolvedRevision",
        )?;
        require_nonempty(&input.variant, "sourceSession.inputs.variant")?;
        if input.bytes == 0 || !is_sha256(&input.sha256) {
            return Err(format!("{} has an invalid source input", session.id));
        }
    }
    let requires_exact_inputs = session.claims.iter().any(|claim| {
        matches!(
            claim,
            SourceClaim::Loadability | SourceClaim::Quality | SourceClaim::NegativeMutation
        )
    });
    if requires_exact_inputs && session.inputs.is_empty() {
        return Err(format!(
            "{} artifact claims require exact inputs",
            session.id
        ));
    }
    if let Some(target) = &session.target {
        let has_base = session
            .inputs
            .iter()
            .any(|input| input.role == SourceInputRole::Base && input.variant == target.tier);
        let has_overlay = match target.overlay.as_str() {
            "control" => session
                .inputs
                .iter()
                .any(|input| input.role == SourceInputRole::Control),
            "lora" => session
                .inputs
                .iter()
                .any(|input| input.role == SourceInputRole::Adapter),
            _ => true,
        };
        if requires_exact_inputs && (!has_base || !has_overlay) {
            return Err(format!(
                "{} artifact claim is missing its exact tier/overlay inputs",
                session.id
            ));
        }
    }
    let mut physical_output_roles = BTreeSet::new();
    let mut physical_output_paths = BTreeSet::new();
    let source_directory = session
        .source_path
        .rsplit_once('/')
        .map(|(parent, _)| parent);
    if session.kind == SourceSessionKind::PhysicalMlx && session.outputs.len() != 3 {
        return Err(format!(
            "{} physical MLX session requires exactly request, selected_rgb, and reference_rgb outputs",
            session.id
        ));
    }
    for output in &session.outputs {
        require_nonempty(&output.path, "sourceSession.outputs.path")?;
        if session.kind == SourceSessionKind::PhysicalMlx {
            if !is_normalized_calibration_path(&output.path) {
                return Err(format!(
                    "{} physical MLX output has an invalid repository path",
                    session.id
                ));
            }
            let role = output
                .role
                .ok_or_else(|| format!("{} physical MLX output is missing its role", session.id))?;
            let bytes = output.bytes.ok_or_else(|| {
                format!(
                    "{} physical MLX output is missing its byte count",
                    session.id
                )
            })?;
            if bytes == 0 {
                return Err(format!(
                    "{} physical MLX output byte count must be positive",
                    session.id
                ));
            }
            if !physical_output_roles.insert(role) {
                return Err(format!(
                    "{} physical MLX session repeats an output role",
                    session.id
                ));
            }
            if !physical_output_paths.insert(output.path.as_str()) {
                return Err(format!(
                    "{} physical MLX session repeats an output path",
                    session.id
                ));
            }
            let output_directory = output.path.rsplit_once('/').map(|(parent, _)| parent);
            if output_directory != source_directory {
                return Err(format!(
                    "{} physical MLX outputs must share the source directory",
                    session.id
                ));
            }
            match role {
                SourceOutputRole::Request => {
                    let expected = format!(
                        "{}/{}.request.json",
                        source_directory.unwrap_or_default(),
                        session.id
                    );
                    if output.path != expected {
                        return Err(format!(
                            "{} request receipt must be named from the session id",
                            session.id
                        ));
                    }
                }
                SourceOutputRole::SelectedRgb | SourceOutputRole::ReferenceRgb => {
                    physical_mlx_rgb_metadata(output)?;
                }
            }
        }
        if !is_sha256(&output.sha256) {
            return Err(format!(
                "{} output sha256 must be lowercase SHA-256",
                session.id
            ));
        }
    }
    if session.kind == SourceSessionKind::PhysicalMlx
        && physical_output_roles
            != BTreeSet::from([
                SourceOutputRole::Request,
                SourceOutputRole::SelectedRgb,
                SourceOutputRole::ReferenceRgb,
            ])
    {
        return Err(format!(
            "{} physical MLX session must contain request, selected_rgb, and reference_rgb outputs",
            session.id
        ));
    }
    if session.kind == SourceSessionKind::PhysicalMlx {
        let expected_source = format!(
            "{}/{}.log",
            source_directory.unwrap_or_default(),
            session.id
        );
        if session.source_path != expected_source {
            return Err(format!(
                "{} physical MLX sourcePath must be named from the session id",
                session.id
            ));
        }
    }
    Ok(())
}

fn physical_mlx_rgb_metadata(output: &SourceOutput) -> Result<(String, u32, u32), String> {
    let role = match output.role {
        Some(SourceOutputRole::SelectedRgb) => "selected_rgb",
        Some(SourceOutputRole::ReferenceRgb) => "reference_rgb",
        _ => return Err("physical MLX RGB receipt has a non-RGB role".to_owned()),
    };
    let file_name = output.path.rsplit('/').next().unwrap_or_default();
    let stem = file_name
        .strip_suffix(".rgb")
        .ok_or_else(|| format!("physical MLX {role} receipt must end in .rgb"))?;
    if stem.len() < 27 || !has_prefixed_hex(&stem[..27], "implan-", 20) {
        return Err(format!(
            "physical MLX {role} receipt must begin with its logical case id"
        ));
    }
    let logical_case_id = stem[..27].to_owned();
    let remainder = stem[27..]
        .strip_prefix(&format!("-{role}-"))
        .ok_or_else(|| format!("physical MLX {role} receipt path has the wrong role"))?;
    let (dimensions, content_sha256) = remainder
        .split_once('-')
        .ok_or_else(|| format!("physical MLX {role} receipt path is missing its digest"))?;
    let (width, height) = dimensions
        .split_once('x')
        .ok_or_else(|| format!("physical MLX {role} receipt path is missing dimensions"))?;
    let width = width
        .parse::<u32>()
        .map_err(|_| format!("physical MLX {role} receipt width is invalid"))?;
    let height = height
        .parse::<u32>()
        .map_err(|_| format!("physical MLX {role} receipt height is invalid"))?;
    let expected_bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| format!("physical MLX {role} receipt dimensions overflow"))?;
    if width == 0 || height == 0 || output.bytes != Some(expected_bytes) {
        return Err(format!(
            "physical MLX {role} receipt byte count does not match its dimensions"
        ));
    }
    if !is_sha256(content_sha256) || content_sha256 != output.sha256 {
        return Err(format!(
            "physical MLX {role} receipt digest does not match its content-addressed path"
        ));
    }
    Ok((logical_case_id, width, height))
}

fn validate_physical_mlx_outputs_against_record(
    record: &EvidenceRecord,
    session: &SourceSession,
) -> Result<(), String> {
    for output in &session.outputs {
        if matches!(
            output.role,
            Some(SourceOutputRole::SelectedRgb | SourceOutputRole::ReferenceRgb)
        ) {
            let (logical_case_id, width, height) = physical_mlx_rgb_metadata(output)?;
            if logical_case_id != record.logical_case_id
                || width != record.target.geometry.width
                || height != record.target.geometry.height
            {
                return Err(format!(
                    "{} physical MLX RGB receipt does not match its logical case geometry",
                    record.id
                ));
            }
        }
    }
    Ok(())
}

fn is_normalized_calibration_path(value: &str) -> bool {
    value.starts_with("docs/calibration/")
        && value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn validate_derivation(
    record: &EvidenceRecord,
    sessions: &BTreeMap<&str, &SourceSession>,
    inventory_inputs: &BTreeMap<(SourceInputRole, String), &SourceInput>,
) -> Result<(), String> {
    let requires_z_image_provenance = record.backend == Backend::Candle
        && record.target.model_id == "z_image"
        && record.evidence_scope == EvidenceScope::Authoritative
        && record.status == RecordStatus::Complete;
    let is_authoritative_qwen_mlx = record.backend == Backend::Mlx
        && record.target.model_id == "qwen_image"
        && record.evidence_scope == EvidenceScope::Authoritative
        && record.status == RecordStatus::Complete;
    if record.source_provenance.is_some() && !is_authoritative_qwen_mlx {
        return Err(format!(
            "{} sourceProvenance is valid only for complete authoritative Qwen MLX evidence",
            record.id
        ));
    }
    let requires_qwen_mlx_provenance = is_authoritative_qwen_mlx
        && record.source_provenance == Some(SourceProvenance::PhysicalMlxV1);
    let requires_provenance = requires_z_image_provenance || requires_qwen_mlx_provenance;
    if requires_qwen_mlx_provenance && record.artifact.inventory_sha256.is_none() {
        return Err(format!(
            "{} authoritative Qwen MLX evidence requires an exact artifact inventory",
            record.id
        ));
    }
    let Some(derivation) = &record.derivation else {
        return if requires_provenance {
            Err(format!("{} requires source-session derivation", record.id))
        } else {
            Ok(())
        };
    };
    require_nonempty(&derivation.justification, "record.derivation.justification")?;
    let dimensions = [
        (SourceClaim::Memory, &derivation.memory),
        (SourceClaim::Quality, &derivation.quality),
        (SourceClaim::NegativeMutation, &derivation.negative_mutation),
        (SourceClaim::Lifecycle, &derivation.lifecycle),
        (SourceClaim::Loadability, &derivation.loadability),
        (SourceClaim::Overlay, &derivation.overlay),
    ];
    let mut derivation_session_ids = BTreeSet::new();
    for (claim, reference) in dimensions {
        if reference.source_session_ids.is_empty() {
            return Err(format!("{} has an empty derivation source list", record.id));
        }
        let unique = reference.source_session_ids.iter().collect::<BTreeSet<_>>();
        if unique.len() != reference.source_session_ids.len() {
            return Err(format!("{} repeats a derivation source", record.id));
        }
        for id in &reference.source_session_ids {
            derivation_session_ids.insert(id.as_str());
            let session = sessions
                .get(id.as_str())
                .ok_or_else(|| format!("{} references missing source session {id}", record.id))?;
            if !session.claims.contains(&claim) {
                return Err(format!(
                    "{} source session {id} does not claim {claim:?}",
                    record.id
                ));
            }
            if requires_qwen_mlx_provenance && session.kind != SourceSessionKind::PhysicalMlx {
                return Err(format!(
                    "{} authoritative Qwen MLX derivation requires a physical_mlx source session",
                    record.id
                ));
            }
            if requires_provenance {
                validate_source_inputs_against_record(
                    record,
                    session,
                    claim,
                    inventory_inputs,
                    requires_qwen_mlx_provenance,
                )?;
            }
            if let Some(target) = &session.target {
                if matches!(
                    claim,
                    SourceClaim::Memory | SourceClaim::Quality | SourceClaim::Overlay
                ) && target.tier != record.target.tier
                {
                    return Err(format!(
                        "{} cannot derive {claim:?} across precision tiers from {id}",
                        record.id
                    ));
                }
                if claim == SourceClaim::Memory && target.rung != record.strategy.rung {
                    return Err(format!(
                        "{} cannot derive memory across rungs from {id}",
                        record.id
                    ));
                }
                if matches!(reference.kind, DerivationKind::Direct)
                    && matches!(claim, SourceClaim::Quality | SourceClaim::Overlay)
                    && target.overlay != record.target.overlay
                {
                    return Err(format!(
                        "{} direct {claim:?} source {id} has the wrong overlay",
                        record.id
                    ));
                }
            }
        }
    }
    if requires_provenance {
        let mut loadability_inputs = BTreeMap::<SourceInputRole, &SourceInput>::new();
        for id in &derivation.loadability.source_session_ids {
            let session = sessions
                .get(id.as_str())
                .ok_or_else(|| format!("{} references missing source session {id}", record.id))?;
            for input in &session.inputs {
                if let Some(existing) = loadability_inputs.insert(input.role, input) {
                    if existing != input {
                        return Err(format!(
                            "{} loadability sources disagree on exact {:?} input identity",
                            record.id, input.role
                        ));
                    }
                }
            }
        }
    }
    if requires_qwen_mlx_provenance && derivation_session_ids.len() != 1 {
        return Err(format!(
            "{} authoritative Qwen MLX claims must share one physical capture session",
            record.id
        ));
    }
    if requires_qwen_mlx_provenance {
        let session_id = derivation_session_ids
            .iter()
            .next()
            .expect("physical MLX derivation has exactly one session");
        let session = sessions
            .get(session_id)
            .expect("physical MLX derivation session was resolved above");
        validate_physical_mlx_outputs_against_record(record, session)?;
    }
    if !matches!(
        derivation.memory.kind,
        DerivationKind::Direct | DerivationKind::ConservativeUpperBound
    ) {
        return Err(format!(
            "{} has an invalid memory derivation kind",
            record.id
        ));
    }
    if derivation.loadability.kind != DerivationKind::Direct {
        return Err(format!(
            "{} loadability must be directly sourced",
            record.id
        ));
    }
    Ok(())
}

fn validate_source_inputs_against_record(
    record: &EvidenceRecord,
    session: &SourceSession,
    claim: SourceClaim,
    inventory_inputs: &BTreeMap<(SourceInputRole, String), &SourceInput>,
    allow_physical_mlx_inventory: bool,
) -> Result<(), String> {
    let fingerprint = match &record.loadability.resolved_path_fingerprint {
        RequiredNullable::Value(value) => value.as_str(),
        RequiredNullable::Null => "",
    };
    for input in &session.inputs {
        let inventory_key = (input.role, input.variant.clone());
        if let Some(expected_input) = inventory_inputs.get(&inventory_key) {
            if **expected_input != *input {
                return Err(format!(
                    "{} source session {} input differs from its exact inventory identity",
                    record.id, session.id
                ));
            }
        } else if !allow_physical_mlx_inventory || session.kind != SourceSessionKind::PhysicalMlx {
            return Err(format!(
                "{} source session {} has no canonical {:?}/{} inventory",
                record.id, session.id, input.role, input.variant
            ));
        }
        if input.role == SourceInputRole::Base {
            if input.repository != record.artifact.repository
                || input.resolved_revision != record.artifact.resolved_revision
            {
                return Err(format!(
                    "{} source session {} base input does not match record artifact identity",
                    record.id, session.id
                ));
            }
            let expected_tier = session
                .target
                .as_ref()
                .map_or(record.target.tier.as_str(), |target| target.tier.as_str());
            if input.variant != expected_tier {
                return Err(format!(
                    "{} source session {} base input has the wrong tier variant",
                    record.id, session.id
                ));
            }
            if allow_physical_mlx_inventory
                && record.artifact.inventory_sha256.as_deref() != Some(input.sha256.as_str())
            {
                return Err(format!(
                    "{} source session {} base input does not match the record artifact inventory",
                    record.id, session.id
                ));
            }
        }

        let exact_overlay_source = session
            .target
            .as_ref()
            .map_or(true, |target| target.overlay == record.target.overlay)
            && matches!(
                claim,
                SourceClaim::Quality | SourceClaim::Loadability | SourceClaim::Overlay
            );
        if !exact_overlay_source {
            continue;
        }
        let token = match input.role {
            SourceInputRole::Base => format!(
                "{}@{}:{}",
                input.repository, input.resolved_revision, input.variant
            ),
            SourceInputRole::Control => {
                format!("+{}@{}", input.repository, input.resolved_revision)
            }
            SourceInputRole::Adapter => format!("+lora@{}", input.resolved_revision),
        };
        if !fingerprint.contains(&token) {
            return Err(format!(
                "{} source session {} input is absent from the record artifact fingerprint",
                record.id, session.id
            ));
        }
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
            record.quality.root_mean_square_error,
            record.quality.maximum_error_threshold,
            record.quality.mean_error_threshold,
            record.quality.root_mean_square_error_threshold,
        ]
        .into_iter()
        .flatten()
        .any(|value| value < 0.0)
    {
        return Err(format!("{} quality fields violate the schema", record.id));
    }
    if record
        .artifact
        .inventory_sha256
        .as_deref()
        .is_some_and(|value| value.len() != 64 || !is_lower_hex(value))
    {
        return Err(format!("{} artifact inventorySha256 is invalid", record.id));
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
        RecordStatus::RuntimeComplete => validate_runtime_complete(record),
        RecordStatus::NegativeComplete => validate_negative_complete(record),
        RecordStatus::Gated => Ok(()),
    }
}

fn validate_runtime_complete(record: &EvidenceRecord) -> Result<(), String> {
    if record.target.overlay != "none" {
        return Err(format!(
            "{} runtime-complete evidence must target the base-only none overlay",
            record.id
        ));
    }
    if record.repositories.scene_works.dirty || record.repositories.inference.dirty {
        return Err(format!(
            "{} runtime-complete evidence has a dirty repository",
            record.id
        ));
    }
    let sole_case = record.sweep.cases.first();
    if !record.sweep.range_verified
        || record.sweep.cases.len() != 1
        || sole_case.map_or(true, |case| {
            case.result != SweepResult::Passed || case.parameters != record.strategy.parameters
        })
    {
        return Err(format!(
            "{} runtime-complete evidence needs exactly one passed case matching its strategy parameters",
            record.id
        ));
    }
    let predicted = required_value(
        &record.predicted_peak_bytes,
        &record.id,
        "predictedPeakBytes",
    )?;
    if let Some(predicted) = predicted.full() {
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
    }
    let observed = required_value(&record.observed_memory, &record.id, "observedMemory")?;
    if let Some(observed) = observed.full() {
        validate_phase_metrics(observed, &record.id)?;
    }
    if observed.overall_device_or_active_bytes() > record.hardware.memory_bytes() {
        return Err(format!(
            "{} observed device memory exceeds hardware",
            record.id
        ));
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
            "{} runtime-complete scenarios must be unique and exhaustive",
            record.id
        ));
    }
    for name in [
        ScenarioName::ExactFit,
        ScenarioName::UnknownBudget,
        ScenarioName::StaleEvidence,
        ScenarioName::Loadability,
    ] {
        if scenarios.get(&name).map(|scenario| scenario.result) != Some(ScenarioResult::Passed) {
            return Err(format!("{} scenario {name:?} must pass", record.id));
        }
    }
    for name in [
        ScenarioName::WarmRepeat,
        ScenarioName::Cancel,
        ScenarioName::Error,
    ] {
        let scenario = scenarios
            .get(&name)
            .ok_or_else(|| format!("{} is missing scenario {name:?}", record.id))?;
        if scenario.result != ScenarioResult::NotRun
            || scenario.reason.as_deref().map_or(true, str::is_empty)
        {
            return Err(format!(
                "{} scenario {name:?} must remain explicitly not_run",
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
    let overlay = scenarios[&ScenarioName::Overlay];
    if overlay.result != ScenarioResult::NotApplicable
        || overlay.reason.as_deref().map_or(true, str::is_empty)
    {
        return Err(format!(
            "{} runtime-complete evidence must be base-only",
            record.id
        ));
    }
    if !matches!(record.negative_mutation, RequiredNullable::Null) {
        return Err(format!(
            "{} unexecuted negative mutation must remain null",
            record.id
        ));
    }
    let quality = &record.quality;
    if quality.identical_inputs != Some(true) || quality.result != Some(QualityResult::Passed) {
        return Err(format!(
            "{} runtime-complete quality evidence did not pass",
            record.id
        ));
    }
    let (maximum_error, mean_error, maximum_threshold, mean_threshold) =
        require_complete_quality_fields(quality, &record.id)?;
    if maximum_error > maximum_threshold || mean_error > mean_threshold {
        return Err(format!("{} quality thresholds were exceeded", record.id));
    }
    let rmse = quality
        .root_mean_square_error
        .ok_or_else(|| format!("{} is missing quality.rootMeanSquareError", record.id))?;
    let rmse_threshold = quality.root_mean_square_error_threshold.ok_or_else(|| {
        format!(
            "{} is missing quality.rootMeanSquareErrorThreshold",
            record.id
        )
    })?;
    if rmse > rmse_threshold {
        return Err(format!("{} RMSE threshold was exceeded", record.id));
    }
    if record.loadability.result != LoadabilityResult::Passed
        || !matches!(
            &record.loadability.resolved_path_fingerprint,
            RequiredNullable::Value(value) if !value.is_empty()
        )
    {
        return Err(format!(
            "{} runtime-complete loadability did not pass",
            record.id
        ));
    }
    Ok(())
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
    let predicted = predicted.full().ok_or_else(|| {
        format!(
            "{} complete evidence requires full predicted phase telemetry",
            record.id
        )
    })?;
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
    let observed = observed.full().ok_or_else(|| {
        format!(
            "{} complete evidence requires full observed phase telemetry",
            record.id
        )
    })?;
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

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && is_lower_hex(value)
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
    use std::collections::{BTreeMap, BTreeSet};

    use serde_json::{json, Map, Value};

    use super::{
        load_bundle, load_packaged_bundle, Backend, BundleLoad, BundleLoadError,
        CalibrationBinding, EvidenceBundle, EvidenceQuery, EvidenceVerdict, Geometry, LoadShapeKey,
        MlxAdmissionEnvelope, ObservedMemory, PredictedPeakBytes, RecordStatus, RequiredNullable,
        SourceSessionKind, StaleBundleReason, StaleEvidenceReason, StrategyRung,
        MEMORY_CALIBRATION_ABI, PACKAGED_MEMORY_CALIBRATION_EVIDENCE,
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
                    "dirty": false,
                    "closureDigest": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
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
            "harnessVersion": "sceneworks-memory-v5"
        })
    }

    fn bundle(record: Value) -> String {
        json!({
            "schemaVersion": 4,
            "harnessVersion": "sceneworks-memory-v5",
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

    #[test]
    fn physical_mlx_source_sessions_load_and_bind_authoritative_qwen_inventory() {
        let mut record = mlx_record(
            128 * 1024 * 1024 * 1024,
            120 * 1024 * 1024 * 1024,
            96 * 1024 * 1024 * 1024,
        );
        record["target"]["modelId"] = json!("qwen_image");
        record["target"]["provider"] = json!("qwen_image");
        record["sourceProvenance"] = json!("physical_mlx_v1");
        record["artifact"]["inventorySha256"] = json!("d".repeat(64));
        record["loadability"]["resolvedPathFingerprint"] =
            json!(format!("SceneWorks/fixture@{}:q4", "c".repeat(40)));
        let session_id = format!("ims-{}", "1".repeat(20));
        let logical_case_id = record["logicalCaseId"]
            .as_str()
            .expect("fixture logical case id")
            .to_owned();
        let direct = || json!({ "kind": "direct", "sourceSessionIds": [session_id.clone()] });
        record["derivation"] = json!({
            "memory": direct(),
            "quality": direct(),
            "negativeMutation": direct(),
            "lifecycle": direct(),
            "loadability": direct(),
            "overlay": direct(),
            "justification": "exact physical MLX capture",
        });
        let source_session = json!({
            "id": session_id.clone(),
            "kind": "physical_mlx",
            "command": "memory-mlx-adapter",
            "sourcePath": format!("docs/calibration/sc-test/{session_id}.log"),
            "capturedAt": "2026-08-01T12:00:00Z",
            "repositories": record["repositories"].clone(),
            "hardware": record["hardware"].clone(),
            "target": {
                "tier": "q4", "mode": "text_to_image", "overlay": "none",
                "rung": "bounded_decode"
            },
            "stdoutSha256": "2".repeat(64),
            "inputs": [{
                "role": "base",
                "path": "/fixture/q4",
                "bytes": 1234,
                "sha256": "d".repeat(64),
                "repository": "SceneWorks/fixture",
                "resolvedRevision": "c".repeat(40),
                "variant": "q4"
            }],
            "outputs": [
                {
                    "role": "request",
                    "path": format!("docs/calibration/sc-test/{session_id}.request.json"),
                    "sha256": "3".repeat(64),
                    "bytes": 1024
                },
                {
                    "role": "selected_rgb",
                    "path": format!(
                        "docs/calibration/sc-test/{logical_case_id}-selected_rgb-1024x1024-{}.rgb",
                        "4".repeat(64)
                    ),
                    "sha256": "4".repeat(64),
                    "bytes": 1024 * 1024 * 3
                },
                {
                    "role": "reference_rgb",
                    "path": format!(
                        "docs/calibration/sc-test/{logical_case_id}-reference_rgb-1024x1024-{}.rgb",
                        "5".repeat(64)
                    ),
                    "sha256": "5".repeat(64),
                    "bytes": 1024 * 1024 * 3
                }
            ],
            "claims": ["memory", "quality", "negative_mutation", "lifecycle", "loadability", "overlay"],
            "result": "passed"
        });
        let document = json!({
            "schemaVersion": 4,
            "harnessVersion": "sceneworks-memory-v5",
            "sourceSessions": [source_session],
            "records": [record]
        });
        let loaded = match load_bundle(&document.to_string()).expect("physical MLX bundle parses") {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => {
                panic!("unexpected stale physical MLX fixture: {reason:?}")
            }
        };
        assert_eq!(
            loaded.source_sessions[0].kind,
            SourceSessionKind::PhysicalMlx
        );

        let mut missing_receipts = document.clone();
        missing_receipts["sourceSessions"][0]["outputs"] = json!([]);
        assert!(matches!(
            load_bundle(&missing_receipts.to_string()),
            Err(BundleLoadError::Invalid(_))
        ));

        let mut duplicate_role = document.clone();
        duplicate_role["sourceSessions"][0]["outputs"][2]["role"] = json!("selected_rgb");
        assert!(matches!(
            load_bundle(&duplicate_role.to_string()),
            Err(BundleLoadError::Invalid(_))
        ));

        let mut swapped_roles = document.clone();
        swapped_roles["sourceSessions"][0]["outputs"][0]["role"] = json!("selected_rgb");
        swapped_roles["sourceSessions"][0]["outputs"][1]["role"] = json!("request");
        assert!(matches!(
            load_bundle(&swapped_roles.to_string()),
            Err(BundleLoadError::Invalid(_))
        ));

        let mut mismatched_digest = document.clone();
        mismatched_digest["sourceSessions"][0]["outputs"][1]["sha256"] = json!("6".repeat(64));
        assert!(matches!(
            load_bundle(&mismatched_digest.to_string()),
            Err(BundleLoadError::Invalid(_))
        ));

        let mut mismatched_bytes = document.clone();
        mismatched_bytes["sourceSessions"][0]["outputs"][1]["bytes"] = json!(1);
        assert!(matches!(
            load_bundle(&mismatched_bytes.to_string()),
            Err(BundleLoadError::Invalid(_))
        ));

        let mut wrong_logical_case = document.clone();
        wrong_logical_case["sourceSessions"][0]["outputs"][1]["path"] = json!(format!(
            "docs/calibration/sc-test/implan-{}-selected_rgb-1024x1024-{}.rgb",
            "0".repeat(20),
            "4".repeat(64)
        ));
        assert!(matches!(
            load_bundle(&wrong_logical_case.to_string()),
            Err(BundleLoadError::Invalid(_))
        ));

        let mut missing_derivation = document;
        missing_derivation["records"][0]
            .as_object_mut()
            .expect("record object")
            .remove("derivation");
        assert!(matches!(
            load_bundle(&missing_derivation.to_string()),
            Err(BundleLoadError::Invalid(_))
        ));
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
        assert_eq!(small.capture_host_bytes, 8 * gib);
        assert_eq!(small.observed_non_reclaimable_wired_bytes, 230);
        assert_eq!(small.peak_bytes, 230);
        let required = 307;
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

        assert_eq!(
            mid.foreign_reserve_for_host_bytes(8 * gib),
            3 * gib,
            "a 12/32 capture reserve is a 3/8 reserve on an 8 GiB live host"
        );
        assert_eq!(
            mid.foreign_reserve_for_host_bytes(1),
            1,
            "capacity normalization rounds up rather than granting an extra process byte"
        );
        assert_eq!(mid.required_host_bytes_for(8 * gib), 3 * gib + 230);
        assert_eq!(
            mid.required_host_bytes(),
            368,
            "ceil(230 * 32 GiB / (32 - 12) GiB) is the true static boundary"
        );
        assert_eq!(
            mid.foreign_reserve_for_host_bytes(64 * gib),
            12 * gib,
            "capture evidence does not speculate that foreign demand grows on a larger host"
        );

        let qwen = MlxAdmissionEnvelope {
            peak_bytes: 46_305_116_160,
            observed_non_reclaimable_wired_bytes: 44_056_333_980,
            capture_host_bytes: 137_438_953_472,
            foreign_reserve_bytes: 50_394_282_940,
        };
        assert_eq!(qwen.required_host_bytes(), 73_113_341_306);
        assert!(qwen.fits_scaled_host_bytes(73_113_341_306));
        assert!(!qwen.fits_scaled_host_bytes(73_113_341_305));
        assert_eq!(
            qwen.required_host_bytes_for_peak(48_620_371_968),
            76_769_008_371,
            "a stale-widened peak gets its own proportional minimum rather than adding a live-host reserve"
        );

        let capture_branch = MlxAdmissionEnvelope {
            peak_bytes: 7 * gib,
            observed_non_reclaimable_wired_bytes: 7 * gib,
            capture_host_bytes: 8 * gib,
            foreign_reserve_bytes: 2 * gib,
        };
        assert_eq!(
            capture_branch.required_host_bytes(),
            9 * gib,
            "when the proportional solution exceeds the capture host, the absolute-reserve branch owns the minimum"
        );
        let invalid_capture = MlxAdmissionEnvelope {
            capture_host_bytes: 0,
            ..capture_branch
        };
        assert_eq!(invalid_capture.required_host_bytes(), u64::MAX);
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
                inference_closure_digest: "d".repeat(64),
                artifact_repository: "SceneWorks/fixture".to_owned(),
                artifact_resolved_revision: "c".repeat(40),
                artifact_variant: "q4".to_owned(),
                resolved_path_fingerprint: "fixture@resolved:q4".to_owned(),
            },
        }
    }

    #[test]
    fn packaged_bundle_uses_the_current_schema_before_entry_calibration_fans_out() {
        // SC-15817 migrates the packaged protocol before the per-entry calibration stories run.
        // Existing MLX measurements remain available as history under their truthful load shapes;
        // their old inference revisions cannot become a current fit. SC-15510 adds four eager and
        // one deferred current-pin Z-Image records without rewriting that historical provenance.
        // SC-15823 then adds ten base-only runtime-complete FLUX.1 records (eight eager, two
        // deferred) without promoting them to Full completion. SC-15833 adds five deferred FLUX.2
        // runtime records and seven physical sessions without replacing any prior source receipt.
        // SC-18218 adds four eager MLX FLUX.2-dev runtime records across the q4/q8 tiers and
        // 768/1024 geometries.
        // SC-18353 adds thirteen physical MLX source sessions for the exact deferred Qwen bf16/q4
        // captures, without replacing the historical Qwen evidence they supersede for admission.
        // SC-16915 re-collects the MLX qwen_image and krea_2_turbo_control evidence at pin
        // a4f409ae under ABI 3, adding seventeen records (14 eager, 3 deferred) and leaving the
        // superseded 7fbcb4a2/1244b82f/96b13b66 rows in place as history — a receipt cannot be
        // re-dated onto a pin it never ran against (sc-16482).
        let bundle = match load_packaged_bundle().expect("compiled bundle must parse") {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("packaged bundle must be current: {reason:?}"),
        };
        let preserved_session_ids = BTreeSet::from([
            "ims-4b4ab770efa632199d23",
            "ims-4fbfb599c1fc3e3e9dfb",
            "ims-5cf99e7d2a0b88e1dfcf",
            "ims-689c72239ec5bb84594f",
            "ims-68cd302c4d981863ae34",
            "ims-6ba27c6bb1b02924f919",
            "ims-6d120db7e473577a8666",
            "ims-7e019daeae73957fa26c",
            "ims-7e8d2d3865ddc7416364",
            "ims-80d540a194d518ccd289",
            "ims-864721b19f3af847b3b0",
            "ims-ae9e9a0008dea92bd123",
            "ims-b11bcf06f6f086d942c5",
            "ims-bd6bf873c3afa366ebbc",
            "ims-d0895f08dc090ac204c5",
            "ims-d498c23a453aae2d8f8b",
        ]);
        let flux2_sessions = BTreeMap::from([
            (
                "ims-35c1264644b37f2f655b",
                (
                    "docs/calibration/sc-15833/base-q4-resident.log",
                    StrategyRung::Resident,
                    "text_to_image",
                    "none",
                ),
            ),
            (
                "ims-c232abf85a9aa537fc14",
                (
                    "docs/calibration/sc-15833/base-q4-staged_residency.log",
                    StrategyRung::StagedResidency,
                    "text_to_image",
                    "none",
                ),
            ),
            (
                "ims-c18f56bfccc12f00acfd",
                (
                    "docs/calibration/sc-15833/base-q4-bounded_decode.log",
                    StrategyRung::BoundedDecode,
                    "text_to_image",
                    "none",
                ),
            ),
            (
                "ims-dd71e09b38731b5a6c92",
                (
                    "docs/calibration/sc-15833/base-q4-bounded_attention.log",
                    StrategyRung::BoundedAttention,
                    "text_to_image",
                    "none",
                ),
            ),
            (
                "ims-450a73e0b9599f0bf598",
                (
                    "docs/calibration/sc-15833/base-q4-bounded_transformer_residency.log",
                    StrategyRung::BoundedTransformerResidency,
                    "text_to_image",
                    "none",
                ),
            ),
            (
                "ims-7c281ce84d1447b7a533",
                (
                    "docs/calibration/sc-15833/edit-q4-resident.log",
                    StrategyRung::Resident,
                    "image_to_image",
                    "none",
                ),
            ),
            (
                "ims-714a8c8533b53fddfbe6",
                (
                    "docs/calibration/sc-15833/control-q4-resident.log",
                    StrategyRung::Resident,
                    "text_to_image",
                    "control",
                ),
            ),
        ]);
        let qwen_sessions = BTreeMap::from([
            (
                "ims-0a88e8a2c2458d260e67",
                ("bf16", StrategyRung::BoundedDecode),
            ),
            (
                "ims-0ef338a58c51e4817de8",
                ("bf16", StrategyRung::BoundedDecode),
            ),
            (
                "ims-13fca8fa2f40f7c3190c",
                ("bf16", StrategyRung::BoundedDecode),
            ),
            (
                "ims-740649850057d6213fab",
                ("bf16", StrategyRung::BoundedDecode),
            ),
            (
                "ims-7950e071813e8805705c",
                ("bf16", StrategyRung::BoundedDecode),
            ),
            (
                "ims-a266e9aff977a2d60775",
                ("bf16", StrategyRung::StagedResidency),
            ),
            ("ims-adfab948e327840be555", ("bf16", StrategyRung::Resident)),
            (
                "ims-afcf19e7b205488da74d",
                ("bf16", StrategyRung::BoundedDecode),
            ),
            (
                "ims-b4923bb54b289b41ca28",
                ("bf16", StrategyRung::BoundedAttention),
            ),
            (
                "ims-c181f62841b2d13bc390",
                ("bf16", StrategyRung::BoundedDecode),
            ),
            (
                "ims-d42d5acce36df4155e30",
                ("q4", StrategyRung::BoundedAttention),
            ),
            (
                "ims-e9d3e4897ee17296f42d",
                ("q4", StrategyRung::BoundedTransformerResidency),
            ),
            (
                "ims-f6e39653ec1d973f41cf",
                ("bf16", StrategyRung::BoundedTransformerResidency),
            ),
        ]);
        let actual_session_ids = bundle
            .source_sessions
            .iter()
            .map(|session| session.id.as_str())
            .collect::<BTreeSet<_>>();
        let expected_session_ids = preserved_session_ids
            .iter()
            .copied()
            .chain(flux2_sessions.keys().copied())
            .chain(qwen_sessions.keys().copied())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_session_ids, expected_session_ids);
        assert_eq!(
            bundle.source_sessions.len(),
            preserved_session_ids.len() + flux2_sessions.len() + qwen_sessions.len()
        );
        for session in &bundle.source_sessions {
            let Some((path, rung, mode, overlay)) = flux2_sessions.get(session.id.as_str()) else {
                continue;
            };
            assert_eq!(session.kind, SourceSessionKind::PhysicalCuda);
            assert_eq!(&session.source_path, path);
            let target = session.target.as_ref().expect("SC-15833 target receipt");
            assert_eq!(target.tier, "q4");
            assert_eq!(target.rung, *rung);
            assert_eq!(target.mode, *mode);
            assert_eq!(target.overlay, *overlay);
        }
        for session in &bundle.source_sessions {
            let Some((tier, rung)) = qwen_sessions.get(session.id.as_str()) else {
                continue;
            };
            assert_eq!(session.kind, SourceSessionKind::PhysicalMlx);
            assert_eq!(
                session.source_path,
                format!("docs/calibration/sc-18353/{}.log", session.id)
            );
            let target = session.target.as_ref().expect("SC-18353 target receipt");
            assert_eq!(target.tier, *tier);
            assert_eq!(target.rung, *rung);
            assert_eq!(target.mode, "text_to_image");
            assert_eq!(target.overlay, "none");
            assert_eq!(
                session
                    .hardware
                    .extensions
                    .get("chip")
                    .and_then(Value::as_str),
                Some("Apple M5 Max")
            );
            assert_eq!(session.hardware.memory_bytes, 137_438_953_472);
        }
        assert!(bundle.source_sessions.iter().all(|session| {
            session.kind == SourceSessionKind::PhysicalMlx
                || (session.hardware.extensions.contains_key("deviceId")
                    && session
                        .hardware
                        .extensions
                        .contains_key("computeCapability")
                    && session.hardware.extensions.contains_key("driverVersion")
                    && session.hardware.extensions.contains_key("runtimeVersion"))
        }));
        let complete_count = bundle
            .records
            .iter()
            .filter(|record| record.status == RecordStatus::Complete)
            .count();
        assert_eq!(complete_count, 65);
        let runtime_keys = bundle
            .records
            .iter()
            .filter(|record| record.status == RecordStatus::RuntimeComplete)
            .map(|record| (record.target.model_id.as_str(), record.strategy.rung))
            .collect::<BTreeSet<_>>();
        let expected_runtime_keys = ["flux_schnell", "flux_dev", "flux2_dev"]
            .into_iter()
            .flat_map(|model_id| {
                StrategyRung::ALL
                    .into_iter()
                    .map(move |rung| (model_id, rung))
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(runtime_keys, expected_runtime_keys);
        let runtime_complete_count = bundle
            .records
            .iter()
            .filter(|record| record.status == RecordStatus::RuntimeComplete)
            .count();
        assert_eq!(runtime_complete_count, 19);
        assert_eq!(
            bundle.records.len(),
            complete_count + runtime_complete_count
        );
        assert_eq!(
            bundle
                .records
                .iter()
                .filter(|record| record.load_shape == LoadShapeKey::EagerMaterialization)
                .count(),
            54
        );
        assert_eq!(
            bundle
                .records
                .iter()
                .filter(|record| record.load_shape == LoadShapeKey::DeferredMaterialization)
                .count(),
            30
        );
    }

    #[test]
    fn source_hardware_accepts_schema_extensions_without_opening_source_sessions() {
        let mut extended: Value = serde_json::from_str(PACKAGED_MEMORY_CALIBRATION_EVIDENCE)
            .expect("packaged evidence JSON");
        extended["sourceSessions"][0]["hardware"]["futureProbeMetadata"] =
            json!({ "tool": "next-generation-probe", "version": 2 });
        let bundle = match load_bundle(&extended.to_string()).expect("hardware extension parses") {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("extended bundle must be current: {reason:?}"),
        };
        assert_eq!(
            bundle.source_sessions[0].hardware.extensions["futureProbeMetadata"],
            json!({ "tool": "next-generation-probe", "version": 2 })
        );

        extended["sourceSessions"][0]["unexpectedSessionField"] = json!(true);
        assert!(matches!(
            load_bundle(&extended.to_string()),
            Err(BundleLoadError::Json(_))
        ));
    }

    #[test]
    fn runtime_complete_requires_one_exact_passed_sweep_case() {
        fn runtime_record(raw: &mut Value) -> &mut Value {
            raw["records"]
                .as_array_mut()
                .expect("records array")
                .iter_mut()
                .find(|record| record["status"] == "runtime_complete")
                .expect("packaged runtime-complete record")
        }

        let mut extra: Value = serde_json::from_str(PACKAGED_MEMORY_CALIBRATION_EVIDENCE)
            .expect("packaged evidence JSON");
        runtime_record(&mut extra)["sweep"]["cases"]
            .as_array_mut()
            .expect("sweep cases")
            .push(json!({ "parameters": { "unexpected": 1 }, "result": "passed" }));
        assert!(matches!(
            load_bundle(&extra.to_string()),
            Err(BundleLoadError::Invalid(message))
                if message.contains("exactly one passed case matching its strategy parameters")
        ));

        let mut mismatch: Value = serde_json::from_str(PACKAGED_MEMORY_CALIBRATION_EVIDENCE)
            .expect("packaged evidence JSON");
        runtime_record(&mut mismatch)["sweep"]["cases"][0]["parameters"] =
            json!({ "unexpected": 1 });
        assert!(matches!(
            load_bundle(&mismatch.to_string()),
            Err(BundleLoadError::Invalid(message))
                if message.contains("exactly one passed case matching its strategy parameters")
        ));
    }

    #[test]
    fn runtime_complete_accepts_overall_only_telemetry_but_complete_requires_full_phases() {
        fn runtime_record(raw: &mut Value) -> &mut Value {
            raw["records"]
                .as_array_mut()
                .expect("records array")
                .iter_mut()
                .find(|record| record["status"] == "runtime_complete")
                .expect("packaged runtime-complete record")
        }

        let mut sparse: Value = serde_json::from_str(PACKAGED_MEMORY_CALIBRATION_EVIDENCE)
            .expect("packaged evidence JSON");
        let record = runtime_record(&mut sparse);
        let record_id = record["id"].as_str().expect("record id").to_owned();
        let predicted = record["predictedPeakBytes"]["overall"]
            .as_u64()
            .expect("predicted overall");
        let observed = record["observedMemory"]["overall"]["deviceBytes"]
            .as_u64()
            .expect("observed device overall");
        record["predictedPeakBytes"] = json!({ "overall": predicted });
        record["observedMemory"] = json!({ "overall": { "activeBytes": observed } });

        let bundle = match load_bundle(&sparse.to_string()).expect("sparse telemetry parses") {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("sparse bundle must be current: {reason:?}"),
        };
        let record = bundle
            .records
            .iter()
            .find(|record| record.id == record_id)
            .expect("sparse record survives");
        assert!(matches!(
            record.predicted_peak_bytes,
            RequiredNullable::Value(PredictedPeakBytes::RuntimeOverall(_))
        ));
        assert!(matches!(
            record.observed_memory,
            RequiredNullable::Value(ObservedMemory::RuntimeOverall(_))
        ));

        let mut overclaimed = sparse.clone();
        runtime_record(&mut overclaimed)["status"] = json!("complete");
        assert!(matches!(
            load_bundle(&overclaimed.to_string()),
            Err(BundleLoadError::Invalid(message))
                if message.contains("complete evidence requires full predicted phase telemetry")
        ));

        let mut partial = sparse;
        runtime_record(&mut partial)["predictedPeakBytes"] =
            json!({ "conditioning": 1, "overall": predicted });
        assert!(matches!(
            load_bundle(&partial.to_string()),
            Err(BundleLoadError::Json(_))
        ));
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
        assert!(
            matches!(
                load_bundle(&bundle(missing_load_shape)),
                Err(BundleLoadError::Json(_))
            ),
            "a v4 record without its measured loadShape must fail to parse, not default"
        );
        let mut record_stale = complete_record();
        record_stale["harnessVersion"] = json!("old-record");
        assert_eq!(
            load_bundle(&bundle(record_stale)).expect("record harness drift is stale"),
            BundleLoad::Stale(StaleBundleReason::HarnessVersion {
                found: Some("old-record".to_owned())
            })
        );
        assert!(matches!(
            load_bundle(r#"{"harnessVersion":"sceneworks-memory-v5","records":[]}"#),
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

        // sc-17774: the inference REVISION is provenance and moving it changes nothing. Before this
        // change the same mutation returned `Stale`, which is why an inference commit to any model
        // demoted every model's measurements.
        let mut inference = exact_query();
        inference.calibration.inference_revision = "d".repeat(40);
        assert!(matches!(
            bundle.evidence_for(&inference),
            EvidenceVerdict::Verified(_)
        ));

        // The provider's own compile closure is the term that decides currency, and it is not blind.
        let mut closure = exact_query();
        closure.calibration.inference_closure_digest = "e".repeat(64);
        assert_eq!(
            bundle.evidence_for(&closure),
            EvidenceVerdict::Stale(StaleEvidenceReason::InferenceClosure)
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
    fn current_qwen_q4_evidence_without_physical_mlx_provenance_is_stale() {
        let mut record = mlx_record(
            128 * 1024 * 1024 * 1024,
            120 * 1024 * 1024 * 1024,
            96 * 1024 * 1024 * 1024,
        );
        record["target"]["modelId"] = json!("qwen_image");
        record["target"]["provider"] = json!("qwen_image");
        record["target"]["tier"] = json!("q4");
        record["loadability"]["resolvedPathFingerprint"] = json!("fixture@resolved:q4");
        let bundle =
            match load_bundle(&bundle(record)).expect("legacy Qwen receipt remains history") {
                BundleLoad::Ready(bundle) => bundle,
                BundleLoad::Stale(reason) => panic!("unexpected stale bundle envelope: {reason:?}"),
            };
        let mut query = exact_query();
        query.backend = Backend::Mlx;
        query.model_id = "qwen_image".to_owned();
        query.provider = "qwen_image".to_owned();

        assert_eq!(
            bundle.evidence_for(&query),
            EvidenceVerdict::Stale(StaleEvidenceReason::PhysicalMlxProvenance)
        );

        query.calibration.inference_closure_digest = "e".repeat(64);
        assert_eq!(
            bundle.evidence_for(&query),
            EvidenceVerdict::Stale(StaleEvidenceReason::InferenceClosure),
            "pre-provenance records remain ordinary history once their captured closure is stale"
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
