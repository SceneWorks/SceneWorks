//! Structured, model-backed catalog analysis (sc-14957).
//!
//! The durable contract is deliberately record-local: every analyzer stores its
//! model identity, analyzer version, exact thresholds, terminal status, bounded
//! error, and confidence beside its objective output. A catalog-wide version
//! summary is useful for status screens, but selective re-analysis is decided
//! from each record so a crash cannot make incomplete rows look current.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use sceneworks_core::catalog_store::{
    Catalog, CatalogError, CatalogProcessingLease, CatalogRecord, CatalogRegistry, NewCatalogRecord,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const ANALYSIS_LOCK_FILE: &str = ".catalog-structured-analysis.lock";
const FETCH_LOCK_FILE: &str = ".catalog-image-fetch.lock";
const STRUCTURED_ANALYSIS_KEY: &str = "structured";
const STRUCTURED_SCHEMA_VERSION: u32 = 2;
const MAX_ERROR_CHARS: usize = 2_048;
const MAX_PAGE_SIZE: u32 = 1_000;

pub const PERSON_ANALYZER_VERSION: &str = "catalog-person-v1";
pub const FACE_ANALYZER_VERSION: &str = "catalog-face-v1";
pub const POSE_ANALYZER_VERSION: &str = "catalog-pose-v1";
pub const DERIVED_ANALYZER_VERSION: &str = "catalog-body-geometry-v1";

pub type CatalogAnalysisResult<T> = Result<T, CatalogAnalysisError>;

#[derive(Debug)]
pub enum CatalogAnalysisError {
    Catalog(CatalogError),
    Io(std::io::Error),
    InvalidConfig(String),
    Busy(String),
}

impl std::fmt::Display for CatalogAnalysisError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Catalog(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::InvalidConfig(detail) | Self::Busy(detail) => formatter.write_str(detail),
        }
    }
}

impl std::error::Error for CatalogAnalysisError {}

impl From<CatalogError> for CatalogAnalysisError {
    fn from(error: CatalogError) -> Self {
        Self::Catalog(error)
    }
}

impl From<std::io::Error> for CatalogAnalysisError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyzerStatus {
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzerMetadata {
    pub analyzer_version: String,
    pub model_id: String,
    pub model_version: String,
    pub thresholds: Value,
    pub status: AnalyzerStatus,
    pub error: Option<String>,
    pub confidence: Option<f64>,
}

impl AnalyzerMetadata {
    fn succeeded(spec: &AnalyzerSpec, thresholds: Value, confidence: Option<f64>) -> Self {
        Self {
            analyzer_version: spec.analyzer_version.clone(),
            model_id: spec.model_id.clone(),
            model_version: spec.model_version.clone(),
            thresholds,
            status: AnalyzerStatus::Succeeded,
            error: None,
            confidence: confidence.map(unit_interval),
        }
    }

    fn failed(spec: &AnalyzerSpec, thresholds: Value, error: impl std::fmt::Display) -> Self {
        Self {
            analyzer_version: spec.analyzer_version.clone(),
            model_id: spec.model_id.clone(),
            model_version: spec.model_version.clone(),
            thresholds,
            status: AnalyzerStatus::Failed,
            error: Some(bounded_error(error)),
            confidence: None,
        }
    }

    fn is_current(&self, spec: &AnalyzerSpec, thresholds: &Value) -> bool {
        self.status == AnalyzerStatus::Succeeded
            && self.analyzer_version == spec.analyzer_version
            && self.model_id == spec.model_id
            && self.model_version == spec.model_version
            && &self.thresholds == thresholds
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzerSpec {
    pub analyzer_version: String,
    pub model_id: String,
    pub model_version: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogAnalysisThresholds {
    pub person_min_confidence: f64,
    pub face_min_confidence: f64,
    /// RTMW SimCC scores are not probabilities and can exceed one.
    pub pose_min_keypoint_confidence: f64,
    pub prominent_frame_fraction: f64,
    pub frame_edge_margin: f64,
    pub min_pose_coverage: f64,
}

impl Default for CatalogAnalysisThresholds {
    fn default() -> Self {
        Self {
            person_min_confidence: 0.25,
            face_min_confidence: 0.65,
            pose_min_keypoint_confidence: 0.3,
            prominent_frame_fraction: 0.2,
            frame_edge_margin: 0.01,
            min_pose_coverage: 0.72,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogAnalysisConfig {
    pub person: AnalyzerSpec,
    pub face: AnalyzerSpec,
    pub pose: AnalyzerSpec,
    pub derived: AnalyzerSpec,
    pub thresholds: CatalogAnalysisThresholds,
    pub page_size: u32,
}

impl Default for CatalogAnalysisConfig {
    fn default() -> Self {
        Self {
            person: AnalyzerSpec {
                analyzer_version: PERSON_ANALYZER_VERSION.to_owned(),
                model_id: "yolo11m-person".to_owned(),
                model_version: default_person_model_version(),
            },
            face: AnalyzerSpec {
                analyzer_version: FACE_ANALYZER_VERSION.to_owned(),
                model_id: "scrfd-10g-antelopev2".to_owned(),
                model_version: default_face_model_version(),
            },
            pose: AnalyzerSpec {
                analyzer_version: POSE_ANALYZER_VERSION.to_owned(),
                model_id: "rtmw-dw-x-l/ort".to_owned(),
                model_version: default_pose_model_version(),
            },
            derived: AnalyzerSpec {
                analyzer_version: DERIVED_ANALYZER_VERSION.to_owned(),
                model_id: "deterministic-body-geometry".to_owned(),
                model_version: "1".to_owned(),
            },
            thresholds: CatalogAnalysisThresholds::default(),
            page_size: 250,
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn default_person_model_version() -> String {
    crate::person_jobs::DET_REVISION.to_owned()
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
fn default_person_model_version() -> String {
    "3cffbaccca4f239ae6301d8b66ba721401ecfa8d".to_owned()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn default_face_model_version() -> String {
    crate::image_jobs::INSTANTID_MLX_REVISION.to_owned()
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
fn default_face_model_version() -> String {
    "bca0cacf8e5e04529bb2b326a521361b02be84fd".to_owned()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn default_pose_model_version() -> String {
    format!(
        "{}+{}",
        crate::pose_jobs::DET_ZIP_SHA256,
        crate::pose_jobs::POSE_ZIP_SHA256
    )
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
fn default_pose_model_version() -> String {
    "a000224fd8ba283202bc62d4a5fcdfe353adb9f468777dbac1ea2ada2093adde+a87e1af41a0a067776dba7d46e1c21c8f6e9f18e247e0e606718dd1f31e96ffd".to_owned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyzerStage {
    Person,
    Face,
    Pose,
    Derived,
    VisionLanguage,
}

/// Canonical execution ordering shared with optional semantic analysis: cheap,
/// objective detectors and their pure derivation always precede a VLM.
pub fn ordered_analyzer_stages(
    requested: impl IntoIterator<Item = AnalyzerStage>,
) -> Vec<AnalyzerStage> {
    let mut requested = requested.into_iter().collect::<Vec<_>>();
    requested.sort_unstable();
    requested.dedup();
    requested
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub confidence: f64,
}

impl NormalizedBox {
    fn area(self) -> f64 {
        self.width * self.height
    }

    fn right(self) -> f64 {
        self.x + self.width
    }

    fn bottom(self) -> f64 {
        self.y + self.height
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedKeypoint {
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PersonObservation {
    pub boxes: Vec<NormalizedBox>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceObservation {
    pub boxes: Vec<NormalizedBox>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PosePersonObservation {
    pub body_keypoints: Vec<NormalizedKeypoint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PoseObservation {
    pub people: Vec<PosePersonObservation>,
}

/// Inference seam for deterministic tests and the real cached model adapter.
/// Implementations must return observations only; policy and persistence remain
/// centralized in this module.
pub trait CatalogAnalyzers {
    fn analyze_people(
        &mut self,
        image_path: &Path,
        min_confidence: f64,
    ) -> Result<PersonObservation, String>;

    fn analyze_faces(
        &mut self,
        image_path: &Path,
        min_confidence: f64,
    ) -> Result<FaceObservation, String>;

    fn analyze_pose(
        &mut self,
        image_path: &Path,
        min_keypoint_confidence: f64,
    ) -> Result<PoseObservation, String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonAnalysis {
    pub analyzer: AnalyzerMetadata,
    pub count: u32,
    pub max_confidence: Option<f64>,
    pub boxes: Vec<NormalizedBox>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceAnalysis {
    pub analyzer: AnalyzerMetadata,
    pub count: u32,
    pub max_confidence: Option<f64>,
    pub boxes: Vec<NormalizedBox>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PosePersonAnalysis {
    pub body_keypoints: Vec<NormalizedKeypoint>,
    pub detected_keypoints: u32,
    pub total_keypoints: u32,
    pub coverage: f64,
    pub mean_keypoint_confidence: Option<f64>,
    pub bounding_box: Option<NormalizedBox>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PoseAnalysis {
    pub analyzer: AnalyzerMetadata,
    pub person_count: u32,
    pub max_coverage: f64,
    pub people: Vec<PosePersonAnalysis>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CropState {
    Uncropped,
    Cropped,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DerivedBodyAnalysis {
    pub analyzer: AnalyzerMetadata,
    pub full_body_visible: bool,
    pub crop_state: CropState,
    pub subject_frame_fraction: Option<f64>,
    pub pose_coverage: Option<f64>,
    pub prominent: bool,
    pub exactly_one_person: bool,
    pub qualified_single_full_body: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StructuredCatalogAnalysis {
    pub schema_version: u32,
    /// SHA-256 of the exact image bytes observed by every analyzer in this
    /// result. Record IDs and paths are not content identities: a fetch retry or
    /// operator repair may replace bytes in place.
    pub input_fingerprint: String,
    pub person: PersonAnalysis,
    pub face: FaceAnalysis,
    pub pose: PoseAnalysis,
    pub derived: DerivedBodyAnalysis,
    pub execution_order: Vec<AnalyzerStage>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogAnalysisReport {
    pub records_examined: u64,
    pub records_updated: u64,
    pub records_skipped_unavailable: u64,
    pub analyzer_failures: u64,
    pub analyzer_runs: BTreeMap<String, u64>,
}

struct CatalogAnalysisLease {
    _processing: Option<CatalogProcessingLease>,
    _files: Vec<File>,
}

impl CatalogAnalysisLease {
    fn acquire(catalog: &Catalog) -> CatalogAnalysisResult<Self> {
        let processing =
            CatalogProcessingLease::try_acquire(catalog).map_err(|error| match error {
                CatalogError::Conflict(_) => CatalogAnalysisError::Busy(format!(
                    "Catalog {} already has active processing.",
                    catalog.descriptor().id
                )),
                error => CatalogAnalysisError::Catalog(error),
            })?;
        Self::acquire_with_processing(catalog, Some(processing))
    }

    #[allow(dead_code)]
    fn acquire_under_processing_lease(
        catalog: &Catalog,
        _processing: &CatalogProcessingLease,
    ) -> CatalogAnalysisResult<Self> {
        Self::acquire_with_processing(catalog, None)
    }

    fn acquire_with_processing(
        catalog: &Catalog,
        processing: Option<CatalogProcessingLease>,
    ) -> CatalogAnalysisResult<Self> {
        let mut files = Vec::new();
        for name in [FETCH_LOCK_FILE, ANALYSIS_LOCK_FILE] {
            let path = catalog.root().join(name);
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&path)?;
            let contended = fs2::lock_contended_error().raw_os_error();
            match file.try_lock_exclusive() {
                Ok(()) => files.push(file),
                Err(error) if error.raw_os_error() == contended => {
                    return Err(CatalogAnalysisError::Busy(format!(
                        "Catalog {} already has active fetch or analysis work.",
                        catalog.descriptor().id
                    )));
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(Self {
            _processing: processing,
            _files: files,
        })
    }
}

/// Analyze available records in one attached catalog. Catalog identity is
/// resolved exclusively through the registry; no caller-provided root is used.
///
/// Individual inference failures are persisted on that analyzer and do not
/// fabricate empty-success observations or abort other analyzers.
///
/// This is the reusable structured stage consumed by the catalog pipeline:
/// callers provision model paths once, invoke this pass, then schedule optional
/// VLM/embedding work only for records whose persisted objective fields survive
/// their requested filters.
pub fn analyze_attached_catalog<A: CatalogAnalyzers>(
    registry: &CatalogRegistry,
    catalog_id: &str,
    config: &CatalogAnalysisConfig,
    analyzers: &mut A,
) -> CatalogAnalysisResult<CatalogAnalysisReport> {
    analyze_attached_catalog_with_lease(registry, catalog_id, config, analyzers, None)
}

/// Pipeline-only sibling that keeps the caller's shared processing lease alive
/// across fetch, structured analysis, and survivor-only semantic analysis.
#[allow(dead_code)]
pub(crate) fn analyze_attached_catalog_under_processing_lease<A: CatalogAnalyzers>(
    registry: &CatalogRegistry,
    catalog_id: &str,
    config: &CatalogAnalysisConfig,
    analyzers: &mut A,
    processing: &CatalogProcessingLease,
) -> CatalogAnalysisResult<CatalogAnalysisReport> {
    analyze_attached_catalog_with_lease(registry, catalog_id, config, analyzers, Some(processing))
}

fn analyze_attached_catalog_with_lease<A: CatalogAnalyzers>(
    registry: &CatalogRegistry,
    catalog_id: &str,
    config: &CatalogAnalysisConfig,
    analyzers: &mut A,
    processing: Option<&CatalogProcessingLease>,
) -> CatalogAnalysisResult<CatalogAnalysisReport> {
    validate_config(config)?;
    let mut catalog = registry.open_attached(catalog_id)?;
    let _lease = match processing {
        Some(processing) => {
            CatalogAnalysisLease::acquire_under_processing_lease(&catalog, processing)?
        }
        None => CatalogAnalysisLease::acquire(&catalog)?,
    };
    let mut report = CatalogAnalysisReport::default();
    let mut cursor = None;

    loop {
        let page = catalog.page_records_after(cursor, config.page_size)?;
        if page.records.is_empty() {
            break;
        }
        for mut record in page.records {
            report.records_examined = report.records_examined.saturating_add(1);
            let structured = match resolve_catalog_image(&catalog, &record)
                .and_then(|path| image_fingerprint(&path).map(|fingerprint| (path, fingerprint)))
            {
                Ok((image_path, input_fingerprint)) => analyze_record(
                    &record,
                    &image_path,
                    &input_fingerprint,
                    config,
                    analyzers,
                    &mut report,
                ),
                Err(error) => {
                    report.records_skipped_unavailable =
                        report.records_skipped_unavailable.saturating_add(1);
                    let unavailable = unavailable_input_analysis(config, error);
                    (previous_structured_analysis(&record).as_ref() != Some(&unavailable))
                        .then_some(unavailable)
                }
            };
            if let Some(structured) = structured {
                persist_structured_analysis(&mut record, &structured)?;
                let update = NewCatalogRecord {
                    id: record.id,
                    image_path: record.image_path,
                    thumbnail_path: record.thumbnail_path,
                    embedding_path: record.embedding_path,
                    artifact_path: record.artifact_path,
                    metadata: record.metadata,
                };
                catalog.append_records(&[update])?;
                report.records_updated = report.records_updated.saturating_add(1);
            }
        }
        cursor = page.next_cursor;
        if cursor.is_none() {
            break;
        }
    }

    let mut state = catalog.contract_state()?;
    for (name, spec) in [
        ("person", &config.person),
        ("face", &config.face),
        ("pose", &config.pose),
        ("derived_body", &config.derived),
    ] {
        state.analyzer_versions.insert(
            name.to_owned(),
            format!(
                "{}:{}@{}",
                spec.analyzer_version, spec.model_id, spec.model_version
            ),
        );
    }
    catalog.set_contract_state(&state)?;
    Ok(report)
}

fn unavailable_input_analysis(
    config: &CatalogAnalysisConfig,
    error: impl std::fmt::Display,
) -> StructuredCatalogAnalysis {
    let error = format!("catalog image is unavailable: {error}");
    let person = failed_person(
        &config.person,
        json!({"minConfidence": config.thresholds.person_min_confidence}),
        error.clone(),
    );
    let face = failed_face(
        &config.face,
        json!({"minConfidence": config.thresholds.face_min_confidence}),
        error.clone(),
    );
    let pose = failed_pose(
        &config.pose,
        json!({"minKeypointConfidence": config.thresholds.pose_min_keypoint_confidence}),
        error,
    );
    let derived = derive_body_analysis(
        &person,
        &pose,
        &config.derived,
        json!({
            "prominentFrameFraction": config.thresholds.prominent_frame_fraction,
            "frameEdgeMargin": config.thresholds.frame_edge_margin,
            "minPoseCoverage": config.thresholds.min_pose_coverage,
            "poseMinKeypointConfidence": config.thresholds.pose_min_keypoint_confidence,
        }),
        &config.thresholds,
    );
    StructuredCatalogAnalysis {
        schema_version: STRUCTURED_SCHEMA_VERSION,
        input_fingerprint: "unavailable".to_owned(),
        person,
        face,
        pose,
        derived,
        execution_order: Vec::new(),
    }
}

fn analyze_record<A: CatalogAnalyzers>(
    record: &CatalogRecord,
    image_path: &Path,
    input_fingerprint: &str,
    config: &CatalogAnalysisConfig,
    analyzers: &mut A,
    report: &mut CatalogAnalysisReport,
) -> Option<StructuredCatalogAnalysis> {
    let previous = previous_structured_analysis(record);
    let input_is_current = previous
        .as_ref()
        .is_some_and(|value| value.input_fingerprint == input_fingerprint);
    let person_thresholds = json!({
        "minConfidence": config.thresholds.person_min_confidence,
    });
    let face_thresholds = json!({
        "minConfidence": config.thresholds.face_min_confidence,
    });
    let pose_thresholds = json!({
        "minKeypointConfidence": config.thresholds.pose_min_keypoint_confidence,
    });
    let derived_thresholds = json!({
        "prominentFrameFraction": config.thresholds.prominent_frame_fraction,
        "frameEdgeMargin": config.thresholds.frame_edge_margin,
        "minPoseCoverage": config.thresholds.min_pose_coverage,
        "poseMinKeypointConfidence": config.thresholds.pose_min_keypoint_confidence,
    });

    let rerun_person = !input_is_current
        || previous.as_ref().map_or(true, |value| {
            !value
                .person
                .analyzer
                .is_current(&config.person, &person_thresholds)
        });
    let rerun_face = !input_is_current
        || previous.as_ref().map_or(true, |value| {
            !value
                .face
                .analyzer
                .is_current(&config.face, &face_thresholds)
        });
    let rerun_pose = !input_is_current
        || previous.as_ref().map_or(true, |value| {
            !value
                .pose
                .analyzer
                .is_current(&config.pose, &pose_thresholds)
        });
    let rerun_derived = rerun_person
        || rerun_pose
        || previous.as_ref().map_or(true, |value| {
            !value
                .derived
                .analyzer
                .is_current(&config.derived, &derived_thresholds)
        });
    if !rerun_person && !rerun_face && !rerun_pose && !rerun_derived {
        return None;
    }

    let mut execution_order = Vec::new();
    let person = if rerun_person {
        execution_order.push(AnalyzerStage::Person);
        increment_run(report, "person");
        match analyzers.analyze_people(image_path, config.thresholds.person_min_confidence) {
            Ok(observation) => person_analysis(observation, &config.person, person_thresholds),
            Err(error) => {
                report.analyzer_failures = report.analyzer_failures.saturating_add(1);
                failed_person(&config.person, person_thresholds, error)
            }
        }
    } else {
        previous
            .as_ref()
            .expect("prior person result")
            .person
            .clone()
    };
    let face = if rerun_face {
        execution_order.push(AnalyzerStage::Face);
        increment_run(report, "face");
        match analyzers.analyze_faces(image_path, config.thresholds.face_min_confidence) {
            Ok(observation) => face_analysis(observation, &config.face, face_thresholds),
            Err(error) => {
                report.analyzer_failures = report.analyzer_failures.saturating_add(1);
                failed_face(&config.face, face_thresholds, error)
            }
        }
    } else {
        previous.as_ref().expect("prior face result").face.clone()
    };
    let pose = if rerun_pose {
        execution_order.push(AnalyzerStage::Pose);
        increment_run(report, "pose");
        match analyzers.analyze_pose(image_path, config.thresholds.pose_min_keypoint_confidence) {
            Ok(observation) => pose_analysis(
                observation,
                &config.pose,
                pose_thresholds,
                config.thresholds.pose_min_keypoint_confidence,
            ),
            Err(error) => {
                report.analyzer_failures = report.analyzer_failures.saturating_add(1);
                failed_pose(&config.pose, pose_thresholds, error)
            }
        }
    } else {
        previous.as_ref().expect("prior pose result").pose.clone()
    };
    let derived = if rerun_derived {
        execution_order.push(AnalyzerStage::Derived);
        increment_run(report, "derived");
        derive_body_analysis(
            &person,
            &pose,
            &config.derived,
            derived_thresholds,
            &config.thresholds,
        )
    } else {
        previous
            .as_ref()
            .expect("prior derived result")
            .derived
            .clone()
    };

    Some(StructuredCatalogAnalysis {
        schema_version: STRUCTURED_SCHEMA_VERSION,
        input_fingerprint: input_fingerprint.to_owned(),
        person,
        face,
        pose,
        derived,
        execution_order,
    })
}

fn person_analysis(
    mut observation: PersonObservation,
    spec: &AnalyzerSpec,
    thresholds: Value,
) -> PersonAnalysis {
    sanitize_boxes(&mut observation.boxes);
    let max_confidence = observation
        .boxes
        .iter()
        .map(|value| value.confidence)
        .reduce(f64::max);
    PersonAnalysis {
        analyzer: AnalyzerMetadata::succeeded(spec, thresholds, max_confidence),
        count: observation.boxes.len().min(u32::MAX as usize) as u32,
        max_confidence,
        boxes: observation.boxes,
    }
}

fn failed_person(spec: &AnalyzerSpec, thresholds: Value, error: String) -> PersonAnalysis {
    PersonAnalysis {
        analyzer: AnalyzerMetadata::failed(spec, thresholds, error),
        count: 0,
        max_confidence: None,
        boxes: Vec::new(),
    }
}

fn face_analysis(
    mut observation: FaceObservation,
    spec: &AnalyzerSpec,
    thresholds: Value,
) -> FaceAnalysis {
    sanitize_boxes(&mut observation.boxes);
    let max_confidence = observation
        .boxes
        .iter()
        .map(|value| value.confidence)
        .reduce(f64::max);
    FaceAnalysis {
        analyzer: AnalyzerMetadata::succeeded(spec, thresholds, max_confidence),
        count: observation.boxes.len().min(u32::MAX as usize) as u32,
        max_confidence,
        boxes: observation.boxes,
    }
}

fn failed_face(spec: &AnalyzerSpec, thresholds: Value, error: String) -> FaceAnalysis {
    FaceAnalysis {
        analyzer: AnalyzerMetadata::failed(spec, thresholds, error),
        count: 0,
        max_confidence: None,
        boxes: Vec::new(),
    }
}

fn pose_analysis(
    observation: PoseObservation,
    spec: &AnalyzerSpec,
    thresholds: Value,
    min_confidence: f64,
) -> PoseAnalysis {
    let people = observation
        .people
        .into_iter()
        .map(|person| pose_person_analysis(person, min_confidence))
        .collect::<Vec<_>>();
    let max_coverage = people
        .iter()
        .map(|person| person.coverage)
        .reduce(f64::max)
        .unwrap_or(0.0);
    PoseAnalysis {
        analyzer: AnalyzerMetadata::succeeded(spec, thresholds, Some(max_coverage)),
        person_count: people.len().min(u32::MAX as usize) as u32,
        max_coverage,
        people,
    }
}

fn failed_pose(spec: &AnalyzerSpec, thresholds: Value, error: String) -> PoseAnalysis {
    PoseAnalysis {
        analyzer: AnalyzerMetadata::failed(spec, thresholds, error),
        person_count: 0,
        max_coverage: 0.0,
        people: Vec::new(),
    }
}

fn pose_person_analysis(
    mut observation: PosePersonObservation,
    min_confidence: f64,
) -> PosePersonAnalysis {
    for point in &mut observation.body_keypoints {
        point.x = unit_interval(point.x);
        point.y = unit_interval(point.y);
        point.confidence = finite_nonnegative(point.confidence);
    }
    let total = observation.body_keypoints.len();
    let detected = observation
        .body_keypoints
        .iter()
        .filter(|point| point.confidence >= min_confidence)
        .count();
    let mean_keypoint_confidence = (detected > 0).then(|| {
        observation
            .body_keypoints
            .iter()
            .filter(|point| point.confidence >= min_confidence)
            .map(|point| point.confidence)
            .sum::<f64>()
            / detected as f64
    });
    let coverage = if total == 0 {
        0.0
    } else {
        detected as f64 / total as f64
    };
    let bounding_box = keypoint_box(&observation.body_keypoints, min_confidence);
    PosePersonAnalysis {
        body_keypoints: observation.body_keypoints,
        detected_keypoints: detected.min(u32::MAX as usize) as u32,
        total_keypoints: total.min(u32::MAX as usize) as u32,
        coverage,
        mean_keypoint_confidence,
        bounding_box,
    }
}

fn derive_body_analysis(
    person: &PersonAnalysis,
    pose: &PoseAnalysis,
    spec: &AnalyzerSpec,
    thresholds: Value,
    config: &CatalogAnalysisThresholds,
) -> DerivedBodyAnalysis {
    if person.analyzer.status != AnalyzerStatus::Succeeded
        || pose.analyzer.status != AnalyzerStatus::Succeeded
    {
        return DerivedBodyAnalysis {
            analyzer: AnalyzerMetadata::failed(
                spec,
                thresholds,
                "person and pose analyzers must succeed before body geometry can be derived",
            ),
            full_body_visible: false,
            crop_state: CropState::Unknown,
            subject_frame_fraction: None,
            pose_coverage: None,
            prominent: false,
            exactly_one_person: false,
            qualified_single_full_body: false,
        };
    }

    let exactly_one_person = person.count == 1;
    let subject = exactly_one_person.then(|| person.boxes[0]);
    let subject_frame_fraction = subject.map(NormalizedBox::area);
    let prominent =
        subject_frame_fraction.is_some_and(|fraction| fraction >= config.prominent_frame_fraction);
    let matched_pose = subject.and_then(|subject| best_matching_pose(subject, &pose.people));
    let pose_coverage = matched_pose.map(|value| value.coverage);
    let full_body_visible = matched_pose.is_some_and(|value| {
        full_body_keypoints_visible(&value.body_keypoints, config.pose_min_keypoint_confidence)
            && value.coverage >= config.min_pose_coverage
    });
    let touches_edge = subject.is_some_and(|value| {
        value.x <= config.frame_edge_margin
            || value.y <= config.frame_edge_margin
            || value.right() >= 1.0 - config.frame_edge_margin
            || value.bottom() >= 1.0 - config.frame_edge_margin
    });
    let crop_state = if touches_edge {
        CropState::Cropped
    } else if full_body_visible {
        CropState::Uncropped
    } else {
        CropState::Unknown
    };
    let qualified_single_full_body =
        exactly_one_person && prominent && full_body_visible && crop_state == CropState::Uncropped;
    let confidence = subject
        .map(|value| value.confidence)
        .zip(pose_coverage)
        .map(|(person_confidence, coverage)| person_confidence.min(coverage));
    DerivedBodyAnalysis {
        analyzer: AnalyzerMetadata::succeeded(spec, thresholds, confidence),
        full_body_visible,
        crop_state,
        subject_frame_fraction,
        pose_coverage,
        prominent,
        exactly_one_person,
        qualified_single_full_body,
    }
}

fn full_body_keypoints_visible(points: &[NormalizedKeypoint], min_confidence: f64) -> bool {
    if points.len() < 18 {
        return false;
    }
    let visible = |index: usize| points[index].confidence >= min_confidence;
    let head = [0, 14, 15, 16, 17].into_iter().any(visible);
    head && visible(1)
        && visible(8)
        && visible(9)
        && visible(10)
        && visible(11)
        && visible(12)
        && visible(13)
}

fn best_matching_pose(
    subject: NormalizedBox,
    poses: &[PosePersonAnalysis],
) -> Option<&PosePersonAnalysis> {
    poses
        .iter()
        .filter_map(|pose| pose.bounding_box.map(|bbox| (box_iou(subject, bbox), pose)))
        .filter(|(iou, _)| *iou > 0.0)
        .max_by(|(left, _), (right, _)| left.total_cmp(right))
        .map(|(_, pose)| pose)
}

fn box_iou(left: NormalizedBox, right: NormalizedBox) -> f64 {
    let x1 = left.x.max(right.x);
    let y1 = left.y.max(right.y);
    let x2 = left.right().min(right.right());
    let y2 = left.bottom().min(right.bottom());
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = left.area() + right.area() - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn keypoint_box(points: &[NormalizedKeypoint], min_confidence: f64) -> Option<NormalizedBox> {
    let mut visible = points
        .iter()
        .filter(|point| point.confidence >= min_confidence);
    let first = visible.next()?;
    let (mut x1, mut y1, mut x2, mut y2) = (first.x, first.y, first.x, first.y);
    for point in visible {
        x1 = x1.min(point.x);
        y1 = y1.min(point.y);
        x2 = x2.max(point.x);
        y2 = y2.max(point.y);
    }
    Some(NormalizedBox {
        x: x1,
        y: y1,
        width: (x2 - x1).max(0.0),
        height: (y2 - y1).max(0.0),
        confidence: 1.0,
    })
}

fn previous_structured_analysis(record: &CatalogRecord) -> Option<StructuredCatalogAnalysis> {
    let value = record
        .metadata
        .get("analysis")?
        .get(STRUCTURED_ANALYSIS_KEY)?
        .clone();
    serde_json::from_value(value)
        .ok()
        .filter(|analysis: &StructuredCatalogAnalysis| {
            analysis.schema_version == STRUCTURED_SCHEMA_VERSION
        })
}

fn persist_structured_analysis(
    record: &mut CatalogRecord,
    structured: &StructuredCatalogAnalysis,
) -> CatalogAnalysisResult<()> {
    let metadata = record.metadata.as_object_mut().ok_or_else(|| {
        CatalogAnalysisError::InvalidConfig(format!(
            "catalog record {} metadata must be an object",
            record.id
        ))
    })?;
    let analysis = metadata
        .entry("analysis".to_owned())
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| {
            CatalogAnalysisError::InvalidConfig(format!(
                "catalog record {} analysis metadata must be an object",
                record.id
            ))
        })?;
    analysis.insert(
        STRUCTURED_ANALYSIS_KEY.to_owned(),
        serde_json::to_value(structured).map_err(CatalogError::from)?,
    );
    analysis.insert("personCount".to_owned(), json!(structured.person.count));
    analysis.insert("faceCount".to_owned(), json!(structured.face.count));
    analysis.insert(
        "fullBody".to_owned(),
        json!(structured.derived.full_body_visible),
    );
    analysis.insert(
        "cropState".to_owned(),
        serde_json::to_value(structured.derived.crop_state).map_err(CatalogError::from)?,
    );
    analysis.insert(
        "subjectFrameFraction".to_owned(),
        json!(structured.derived.subject_frame_fraction),
    );
    analysis.insert(
        "poseCoverage".to_owned(),
        json!(structured.derived.pose_coverage),
    );
    analysis.insert("prominent".to_owned(), json!(structured.derived.prominent));
    analysis.insert(
        "qualifiedSingleFullBody".to_owned(),
        json!(structured.derived.qualified_single_full_body),
    );
    Ok(())
}

fn resolve_catalog_image(catalog: &Catalog, record: &CatalogRecord) -> std::io::Result<PathBuf> {
    let root = std::fs::canonicalize(catalog.root())?;
    let path = std::fs::canonicalize(catalog.root().join(&record.image_path))?;
    if !path.starts_with(&root) || !path.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "catalog image path escapes the catalog root or is not a file",
        ));
    }
    Ok(path)
}

fn image_fingerprint(path: &Path) -> std::io::Result<String> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn validate_config(config: &CatalogAnalysisConfig) -> CatalogAnalysisResult<()> {
    if config.page_size == 0 || config.page_size > MAX_PAGE_SIZE {
        return Err(CatalogAnalysisError::InvalidConfig(format!(
            "pageSize must be between 1 and {MAX_PAGE_SIZE}"
        )));
    }
    for (name, spec) in [
        ("person", &config.person),
        ("face", &config.face),
        ("pose", &config.pose),
        ("derived", &config.derived),
    ] {
        if spec.analyzer_version.trim().is_empty()
            || spec.model_id.trim().is_empty()
            || spec.model_version.trim().is_empty()
            || spec.analyzer_version.len() > 256
            || spec.model_id.len() > 256
            || spec.model_version.len() > 512
        {
            return Err(CatalogAnalysisError::InvalidConfig(format!(
                "{name} analyzer identity is invalid"
            )));
        }
    }
    let thresholds = &config.thresholds;
    for (name, value) in [
        ("personMinConfidence", thresholds.person_min_confidence),
        ("faceMinConfidence", thresholds.face_min_confidence),
        (
            "prominentFrameFraction",
            thresholds.prominent_frame_fraction,
        ),
        ("frameEdgeMargin", thresholds.frame_edge_margin),
        ("minPoseCoverage", thresholds.min_pose_coverage),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(CatalogAnalysisError::InvalidConfig(format!(
                "{name} must be finite and between zero and one"
            )));
        }
    }
    if thresholds.frame_edge_margin > 0.25 {
        return Err(CatalogAnalysisError::InvalidConfig(
            "frameEdgeMargin cannot exceed 0.25".to_owned(),
        ));
    }
    if !thresholds.pose_min_keypoint_confidence.is_finite()
        || thresholds.pose_min_keypoint_confidence < 0.0
    {
        return Err(CatalogAnalysisError::InvalidConfig(
            "poseMinKeypointConfidence must be finite and non-negative".to_owned(),
        ));
    }
    Ok(())
}

fn sanitize_boxes(boxes: &mut Vec<NormalizedBox>) {
    boxes.retain_mut(|value| {
        value.x = unit_interval(value.x);
        value.y = unit_interval(value.y);
        value.width = value.width.max(0.0).min(1.0 - value.x);
        value.height = value.height.max(0.0).min(1.0 - value.y);
        value.confidence = unit_interval(value.confidence);
        value.width.is_finite()
            && value.height.is_finite()
            && value.width > 0.0
            && value.height > 0.0
    });
    boxes.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
}

fn unit_interval(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn bounded_error(error: impl std::fmt::Display) -> String {
    error.to_string().chars().take(MAX_ERROR_CHARS).collect()
}

fn increment_run(report: &mut CatalogAnalysisReport, analyzer: &str) {
    let count = report.analyzer_runs.entry(analyzer.to_owned()).or_default();
    *count = count.saturating_add(1);
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub struct ModelBackedCatalogAnalyzerPaths {
    pub person_detector: PathBuf,
    pub face_weights_directory: PathBuf,
    pub pose_detector: PathBuf,
    pub pose_model: PathBuf,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub struct ModelBackedCatalogAnalyzers {
    paths: ModelBackedCatalogAnalyzerPaths,
    face_detector: Option<crate::face_analysis_jobs::CatalogFaceDetector>,
    face_load_error: Option<String>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl ModelBackedCatalogAnalyzers {
    pub fn new(paths: ModelBackedCatalogAnalyzerPaths) -> Self {
        Self {
            paths,
            face_detector: None,
            face_load_error: None,
        }
    }

    fn face_detector(&mut self) -> Result<&crate::face_analysis_jobs::CatalogFaceDetector, String> {
        if self.face_detector.is_none() && self.face_load_error.is_none() {
            match crate::face_analysis_jobs::CatalogFaceDetector::load(
                &self.paths.face_weights_directory,
            ) {
                Ok(detector) => self.face_detector = Some(detector),
                Err(error) => self.face_load_error = Some(error.to_string()),
            }
        }
        if let Some(error) = &self.face_load_error {
            return Err(error.clone());
        }
        self.face_detector
            .as_ref()
            .ok_or_else(|| "face detector failed to initialize".to_owned())
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl CatalogAnalyzers for ModelBackedCatalogAnalyzers {
    fn analyze_people(
        &mut self,
        image_path: &Path,
        min_confidence: f64,
    ) -> Result<PersonObservation, String> {
        let result = crate::person_jobs::detect_people_blocking(
            self.paths.person_detector.clone(),
            image_path.to_path_buf(),
            min_confidence as f32,
        )
        .map_err(|error| error.to_string())?;
        let width = f64::from(result.width);
        let height = f64::from(result.height);
        let boxes = result
            .detections
            .into_iter()
            .map(|detection| NormalizedBox {
                x: f64::from(detection.x1) / width,
                y: f64::from(detection.y1) / height,
                width: f64::from(detection.x2 - detection.x1) / width,
                height: f64::from(detection.y2 - detection.y1) / height,
                confidence: f64::from(detection.score),
            })
            .collect();
        Ok(PersonObservation { boxes })
    }

    fn analyze_faces(
        &mut self,
        image_path: &Path,
        min_confidence: f64,
    ) -> Result<FaceObservation, String> {
        let (width, height, detections) = self
            .face_detector()?
            .detect(image_path)
            .map_err(|error| error.to_string())?;
        let width = f64::from(width);
        let height = f64::from(height);
        let boxes = detections
            .into_iter()
            .filter(|detection| f64::from(detection.confidence) >= min_confidence)
            .map(|detection| NormalizedBox {
                x: f64::from(detection.bbox[0]) / width,
                y: f64::from(detection.bbox[1]) / height,
                width: f64::from(detection.bbox[2] - detection.bbox[0]) / width,
                height: f64::from(detection.bbox[3] - detection.bbox[1]) / height,
                confidence: f64::from(detection.confidence),
            })
            .collect();
        Ok(FaceObservation { boxes })
    }

    fn analyze_pose(
        &mut self,
        image_path: &Path,
        _min_keypoint_confidence: f64,
    ) -> Result<PoseObservation, String> {
        let result = crate::pose_jobs::detect_catalog_poses_blocking(
            &self.paths.pose_detector,
            &self.paths.pose_model,
            image_path,
        )
        .map_err(|error| error.to_string())?;
        Ok(PoseObservation {
            people: result
                .people
                .into_iter()
                .map(|person| PosePersonObservation {
                    body_keypoints: person
                        .body_keypoints
                        .into_iter()
                        .map(|point| NormalizedKeypoint {
                            x: f64::from(point[0]),
                            y: f64::from(point[1]),
                            confidence: f64::from(point[2]),
                        })
                        .collect(),
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use image::RgbImage;
    use sceneworks_core::catalog_store::CatalogRecordFilter;
    use tempfile::TempDir;

    use super::*;

    struct Fixture {
        _temporary: TempDir,
        registry: CatalogRegistry,
        catalog_id: String,
        root: PathBuf,
    }

    impl Fixture {
        fn new(name: &str, records: &[&str]) -> Self {
            let temporary = tempfile::tempdir().expect("temporary directory");
            let registry = CatalogRegistry::new(temporary.path().join("state"));
            let root = temporary.path().join(name);
            let mut catalog = registry
                .create_catalog(&root, name)
                .expect("catalog creates");
            std::fs::create_dir_all(root.join("images")).expect("image directory");
            let records = records
                .iter()
                .map(|id| {
                    RgbImage::new(8, 8)
                        .save(root.join("images").join(format!("{id}.png")))
                        .expect("fixture image writes");
                    NewCatalogRecord {
                        id: (*id).to_owned(),
                        image_path: format!("images/{id}.png"),
                        thumbnail_path: None,
                        embedding_path: None,
                        artifact_path: None,
                        metadata: json!({
                            "analysis": {
                                "visionLanguage": {"status": "not_requested"}
                            }
                        }),
                    }
                })
                .collect::<Vec<_>>();
            catalog.append_records(&records).expect("records append");
            let catalog_id = catalog.descriptor().id.clone();
            drop(catalog);
            Self {
                _temporary: temporary,
                registry,
                catalog_id,
                root,
            }
        }

        fn record(&self, id: &str) -> CatalogRecord {
            self.registry
                .open_attached(&self.catalog_id)
                .expect("catalog opens")
                .page_records(0, 100)
                .expect("records read")
                .into_iter()
                .find(|record| record.id == id)
                .expect("record exists")
        }
    }

    #[derive(Default)]
    struct FakeAnalyzers {
        calls: Vec<(AnalyzerStage, String, f64)>,
        fail: BTreeSet<(AnalyzerStage, String)>,
    }

    impl FakeAnalyzers {
        fn record_id(path: &Path) -> String {
            path.file_stem()
                .and_then(|value| value.to_str())
                .expect("fixture file stem")
                .to_owned()
        }

        fn should_fail(&self, stage: AnalyzerStage, id: &str) -> bool {
            self.fail.contains(&(stage, id.to_owned()))
        }

        fn called_stages(&self) -> Vec<AnalyzerStage> {
            self.calls.iter().map(|(stage, _, _)| *stage).collect()
        }
    }

    impl CatalogAnalyzers for FakeAnalyzers {
        fn analyze_people(
            &mut self,
            image_path: &Path,
            min_confidence: f64,
        ) -> Result<PersonObservation, String> {
            let id = Self::record_id(image_path);
            self.calls
                .push((AnalyzerStage::Person, id.clone(), min_confidence));
            if self.should_fail(AnalyzerStage::Person, &id) {
                return Err(format!("person model failed for {id}"));
            }
            let primary = match id.as_str() {
                "cropped" => NormalizedBox {
                    x: 0.0,
                    y: 0.05,
                    width: 0.7,
                    height: 0.94,
                    confidence: 0.96,
                },
                _ => NormalizedBox {
                    x: 0.2,
                    y: 0.04,
                    width: 0.6,
                    height: 0.9,
                    confidence: 0.97,
                },
            };
            let mut boxes = vec![primary];
            if id == "crowd" {
                boxes.push(NormalizedBox {
                    x: 0.72,
                    y: 0.25,
                    width: 0.2,
                    height: 0.5,
                    confidence: 0.82,
                });
            }
            Ok(PersonObservation { boxes })
        }

        fn analyze_faces(
            &mut self,
            image_path: &Path,
            min_confidence: f64,
        ) -> Result<FaceObservation, String> {
            let id = Self::record_id(image_path);
            self.calls
                .push((AnalyzerStage::Face, id.clone(), min_confidence));
            if self.should_fail(AnalyzerStage::Face, &id) {
                return Err(format!("face model failed for {id}"));
            }
            Ok(FaceObservation {
                boxes: vec![NormalizedBox {
                    x: 0.4,
                    y: 0.1,
                    width: 0.18,
                    height: 0.18,
                    confidence: 0.91,
                }],
            })
        }

        fn analyze_pose(
            &mut self,
            image_path: &Path,
            min_keypoint_confidence: f64,
        ) -> Result<PoseObservation, String> {
            let id = Self::record_id(image_path);
            self.calls
                .push((AnalyzerStage::Pose, id.clone(), min_keypoint_confidence));
            if self.should_fail(AnalyzerStage::Pose, &id) {
                return Err(format!("pose model failed for {id}"));
            }
            let mut points = full_body_points();
            if id == "missing-ankle" {
                points[13].confidence = 0.0;
            }
            let mut people = vec![PosePersonObservation {
                body_keypoints: points,
            }];
            if id == "crowd" {
                people.push(PosePersonObservation {
                    body_keypoints: full_body_points()
                        .into_iter()
                        .map(|mut point| {
                            point.x = 0.8 + (point.x - 0.5) * 0.25;
                            point.y = 0.3 + point.y * 0.5;
                            point
                        })
                        .collect(),
                });
            }
            Ok(PoseObservation { people })
        }
    }

    fn full_body_points() -> Vec<NormalizedKeypoint> {
        [
            (0.50, 0.08), // nose
            (0.50, 0.20), // neck
            (0.60, 0.22),
            (0.65, 0.38),
            (0.68, 0.53),
            (0.40, 0.22),
            (0.35, 0.38),
            (0.32, 0.53),
            (0.57, 0.52), // right hip
            (0.58, 0.70),
            (0.59, 0.91), // right ankle
            (0.43, 0.52), // left hip
            (0.42, 0.70),
            (0.41, 0.91), // left ankle
            (0.54, 0.07),
            (0.46, 0.07),
            (0.58, 0.09),
            (0.42, 0.09),
        ]
        .into_iter()
        .map(|(x, y)| NormalizedKeypoint {
            x,
            y,
            confidence: 0.95,
        })
        .collect()
    }

    fn structured(record: &CatalogRecord) -> StructuredCatalogAnalysis {
        previous_structured_analysis(record).expect("structured analysis persisted")
    }

    #[test]
    fn objective_fields_support_the_single_uncropped_full_body_filter() {
        let fixture = Fixture::new(
            "objective",
            &["qualified", "cropped", "crowd", "missing-ankle"],
        );
        let mut analyzers = FakeAnalyzers::default();
        let report = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &CatalogAnalysisConfig::default(),
            &mut analyzers,
        )
        .expect("analysis succeeds");
        assert_eq!(report.records_updated, 4);
        assert_eq!(
            analyzers.called_stages(),
            [
                AnalyzerStage::Person,
                AnalyzerStage::Face,
                AnalyzerStage::Pose,
                AnalyzerStage::Person,
                AnalyzerStage::Face,
                AnalyzerStage::Pose,
                AnalyzerStage::Person,
                AnalyzerStage::Face,
                AnalyzerStage::Pose,
                AnalyzerStage::Person,
                AnalyzerStage::Face,
                AnalyzerStage::Pose,
            ]
        );

        let qualified = fixture.record("qualified");
        let result = structured(&qualified);
        assert_eq!(result.person.count, 1);
        assert_eq!(result.face.count, 1);
        assert_eq!(result.pose.people[0].detected_keypoints, 18);
        assert_eq!(result.derived.crop_state, CropState::Uncropped);
        assert!(result.derived.full_body_visible);
        assert!(result.derived.prominent);
        assert!(result.derived.qualified_single_full_body);
        assert_eq!(
            qualified.metadata["analysis"]["visionLanguage"]["status"],
            json!("not_requested"),
            "structured persistence preserves unrelated analysis fields"
        );

        assert_eq!(
            structured(&fixture.record("cropped")).derived.crop_state,
            CropState::Cropped
        );
        assert!(
            !structured(&fixture.record("crowd"))
                .derived
                .exactly_one_person
        );
        assert!(
            !structured(&fixture.record("missing-ankle"))
                .derived
                .full_body_visible
        );

        let catalog = fixture
            .registry
            .open_attached(&fixture.catalog_id)
            .expect("catalog opens");
        let page = catalog
            .query_records_after(
                None,
                10,
                &[CatalogRecordFilter {
                    field: "analysis.qualifiedSingleFullBody".to_owned(),
                    values: vec!["1".to_owned()],
                }],
            )
            .expect("structured filter works");
        assert_eq!(
            page.records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["qualified"]
        );
    }

    #[test]
    fn reanalysis_is_selective_by_status_version_and_exact_thresholds() {
        let fixture = Fixture::new("selective", &["qualified"]);
        let base = CatalogAnalysisConfig::default();
        let mut first = FakeAnalyzers::default();
        analyze_attached_catalog(&fixture.registry, &fixture.catalog_id, &base, &mut first)
            .expect("initial analysis");

        let mut unchanged = FakeAnalyzers::default();
        let unchanged_report = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &base,
            &mut unchanged,
        )
        .expect("unchanged analysis");
        assert_eq!(unchanged_report.records_updated, 0);
        assert!(unchanged.calls.is_empty());

        let mut face_changed = base.clone();
        face_changed.thresholds.face_min_confidence = 0.7;
        let mut face = FakeAnalyzers::default();
        let report = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &face_changed,
            &mut face,
        )
        .expect("face threshold analysis");
        assert_eq!(
            report.analyzer_runs,
            BTreeMap::from([("face".to_owned(), 1)])
        );
        assert_eq!(face.called_stages(), [AnalyzerStage::Face]);

        let mut person_changed = face_changed.clone();
        person_changed.person.model_version = "new-yolo-revision".to_owned();
        let mut person = FakeAnalyzers::default();
        let report = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &person_changed,
            &mut person,
        )
        .expect("person version analysis");
        assert_eq!(
            report.analyzer_runs,
            BTreeMap::from([("derived".to_owned(), 1), ("person".to_owned(), 1)])
        );
        assert_eq!(person.called_stages(), [AnalyzerStage::Person]);
    }

    #[test]
    fn replacing_image_bytes_reruns_every_analyzer_and_preserves_lifecycle_verdict() {
        let fixture = Fixture::new("replacement", &["qualified"]);
        let mut catalog = fixture
            .registry
            .open_attached(&fixture.catalog_id)
            .expect("catalog opens");
        let mut record = catalog.page_records(0, 1).expect("record reads").remove(0);
        record.metadata["analysis"]["accepted"] = json!(false);
        catalog
            .append_records(&[NewCatalogRecord {
                id: record.id,
                image_path: record.image_path,
                thumbnail_path: record.thumbnail_path,
                embedding_path: record.embedding_path,
                artifact_path: record.artifact_path,
                metadata: record.metadata,
            }])
            .expect("lifecycle verdict updates");
        drop(catalog);

        let config = CatalogAnalysisConfig::default();
        let mut initial = FakeAnalyzers::default();
        analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &config,
            &mut initial,
        )
        .expect("initial analysis");
        let first = structured(&fixture.record("qualified"));

        RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3]))
            .save(fixture.root.join("images/qualified.png"))
            .expect("replacement image writes");
        let mut replacement = FakeAnalyzers::default();
        let report = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &config,
            &mut replacement,
        )
        .expect("replacement analysis");
        assert_eq!(
            report.analyzer_runs,
            BTreeMap::from([
                ("derived".to_owned(), 1),
                ("face".to_owned(), 1),
                ("person".to_owned(), 1),
                ("pose".to_owned(), 1),
            ])
        );
        assert_eq!(
            replacement.called_stages(),
            [
                AnalyzerStage::Person,
                AnalyzerStage::Face,
                AnalyzerStage::Pose,
            ]
        );
        let updated = fixture.record("qualified");
        assert_ne!(
            first.input_fingerprint,
            structured(&updated).input_fingerprint
        );
        assert_eq!(
            updated.metadata["analysis"]["accepted"],
            json!(false),
            "structured qualification must not own the shared lifecycle verdict"
        );
        assert!(
            updated.metadata["analysis"]["qualifiedSingleFullBody"]
                .as_bool()
                .expect("structured qualification"),
            "the independent structured filter is still searchable"
        );
    }

    #[test]
    fn unavailable_image_invalidates_a_previously_searchable_qualification() {
        let fixture = Fixture::new("unavailable", &["qualified"]);
        let config = CatalogAnalysisConfig::default();
        analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &config,
            &mut FakeAnalyzers::default(),
        )
        .expect("initial analysis");
        assert!(
            structured(&fixture.record("qualified"))
                .derived
                .qualified_single_full_body
        );

        std::fs::remove_file(fixture.root.join("images/qualified.png"))
            .expect("fixture image removes");
        let report = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &config,
            &mut FakeAnalyzers::default(),
        )
        .expect("unavailable image is recorded");
        assert_eq!(report.records_skipped_unavailable, 1);
        assert_eq!(report.records_updated, 1);

        let record = fixture.record("qualified");
        let result = structured(&record);
        assert_eq!(result.input_fingerprint, "unavailable");
        assert_eq!(result.person.analyzer.status, AnalyzerStatus::Failed);
        assert_eq!(result.face.analyzer.status, AnalyzerStatus::Failed);
        assert_eq!(result.pose.analyzer.status, AnalyzerStatus::Failed);
        assert_eq!(result.derived.analyzer.status, AnalyzerStatus::Failed);
        assert!(!result.derived.qualified_single_full_body);
        assert_eq!(
            record.metadata["analysis"]["qualifiedSingleFullBody"],
            json!(false)
        );
    }

    #[test]
    fn failed_analyzers_are_honest_bounded_and_retried() {
        let fixture = Fixture::new("failures", &["qualified"]);
        let mut failing = FakeAnalyzers::default();
        failing
            .fail
            .insert((AnalyzerStage::Face, "qualified".to_owned()));
        let report = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &CatalogAnalysisConfig::default(),
            &mut failing,
        )
        .expect("other analyzers continue");
        assert_eq!(report.analyzer_failures, 1);
        let result = structured(&fixture.record("qualified"));
        assert_eq!(result.face.analyzer.status, AnalyzerStatus::Failed);
        assert_eq!(result.face.count, 0);
        assert!(result.face.analyzer.confidence.is_none());
        assert!(result.face.analyzer.error.as_deref().is_some_and(|error| {
            error.contains("face model failed") && error.len() <= MAX_ERROR_CHARS
        }));
        assert!(
            result.derived.qualified_single_full_body,
            "face availability does not redefine person/body geometry"
        );

        let mut recovered = FakeAnalyzers::default();
        let report = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &CatalogAnalysisConfig::default(),
            &mut recovered,
        )
        .expect("failed analyzer retries");
        assert_eq!(recovered.called_stages(), [AnalyzerStage::Face]);
        assert_eq!(report.records_updated, 1);
        assert_eq!(
            structured(&fixture.record("qualified"))
                .face
                .analyzer
                .status,
            AnalyzerStatus::Succeeded
        );
    }

    #[test]
    fn catalogs_remain_isolated_even_when_record_ids_match() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let mut ids = Vec::new();
        for name in ["first", "second"] {
            let root = temporary.path().join(name);
            let mut catalog = registry
                .create_catalog(&root, name)
                .expect("catalog creates");
            std::fs::create_dir_all(root.join("images")).expect("image directory");
            RgbImage::new(4, 4)
                .save(root.join("images/shared.png"))
                .expect("fixture image");
            catalog
                .append_records(&[NewCatalogRecord {
                    id: "shared".to_owned(),
                    image_path: "images/shared.png".to_owned(),
                    thumbnail_path: None,
                    embedding_path: None,
                    artifact_path: None,
                    metadata: json!({}),
                }])
                .expect("record appends");
            ids.push(catalog.descriptor().id.clone());
        }
        let mut analyzers = FakeAnalyzers::default();
        analyze_attached_catalog(
            &registry,
            &ids[0],
            &CatalogAnalysisConfig::default(),
            &mut analyzers,
        )
        .expect("first catalog analysis");

        let first = registry
            .open_attached(&ids[0])
            .unwrap()
            .page_records(0, 1)
            .unwrap()
            .remove(0);
        let second = registry
            .open_attached(&ids[1])
            .unwrap()
            .page_records(0, 1)
            .unwrap()
            .remove(0);
        assert!(previous_structured_analysis(&first).is_some());
        assert!(previous_structured_analysis(&second).is_none());
        assert!(registry
            .open_attached(&ids[1])
            .unwrap()
            .contract_state()
            .unwrap()
            .analyzer_versions
            .is_empty());
    }

    #[test]
    fn optional_vision_language_stage_is_always_last() {
        assert_eq!(
            ordered_analyzer_stages([
                AnalyzerStage::VisionLanguage,
                AnalyzerStage::Pose,
                AnalyzerStage::Person,
                AnalyzerStage::Derived,
                AnalyzerStage::Face,
                AnalyzerStage::Person,
            ]),
            [
                AnalyzerStage::Person,
                AnalyzerStage::Face,
                AnalyzerStage::Pose,
                AnalyzerStage::Derived,
                AnalyzerStage::VisionLanguage,
            ]
        );
    }

    #[test]
    fn invalid_thresholds_are_rejected_before_model_or_catalog_mutation() {
        let fixture = Fixture::new("invalid", &["qualified"]);
        let mut config = CatalogAnalysisConfig::default();
        config.thresholds.frame_edge_margin = f64::NAN;
        let mut analyzers = FakeAnalyzers::default();
        assert!(matches!(
            analyze_attached_catalog(
                &fixture.registry,
                &fixture.catalog_id,
                &config,
                &mut analyzers
            ),
            Err(CatalogAnalysisError::InvalidConfig(_))
        ));
        assert!(analyzers.calls.is_empty());
        assert!(previous_structured_analysis(&fixture.record("qualified")).is_none());
    }

    #[test]
    fn shared_processing_lease_blocks_analysis_before_inference_or_mutation() {
        let fixture = Fixture::new("processing-contention", &["qualified"]);
        let catalog = fixture
            .registry
            .open_attached(&fixture.catalog_id)
            .expect("catalog opens");
        let _processing =
            CatalogProcessingLease::try_acquire(&catalog).expect("shared processing lease");
        let mut analyzers = FakeAnalyzers::default();
        let error = analyze_attached_catalog(
            &fixture.registry,
            &fixture.catalog_id,
            &CatalogAnalysisConfig::default(),
            &mut analyzers,
        )
        .expect_err("active scanner or fetch must block analysis");
        assert!(matches!(error, CatalogAnalysisError::Busy(_)));
        assert!(analyzers.calls.is_empty());
        assert!(previous_structured_analysis(&fixture.record("qualified")).is_none());
    }
}
