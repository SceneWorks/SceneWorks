//! Dataset Catalog lifecycle and bounded query HTTP contracts.
//!
//! Catalog-id operations resolve exclusively through [`CatalogRegistry`], whose
//! state lives under `settings.config_dir`. Only create/attach accept a
//! user-selected absolute path; no request can substitute a root for an attached
//! catalog id.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path as FsPath, PathBuf};

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, Response, StatusCode};
use axum::Json;
use sceneworks_core::catalog_store::{
    AttachedCatalog, Catalog, CatalogAnalyzerConfig, CatalogAnalyzerSettings, CatalogContractState,
    CatalogError, CatalogFacet, CatalogProcessingControl, CatalogProcessingDesiredState,
    CatalogProcessingLease, CatalogProcessingProgress, CatalogProcessingState, CatalogRecord,
    CatalogRecordFilter, CatalogRecordReview, CatalogRegistry, CatalogReviewDecision,
    CatalogSavedView, CatalogSearchRequest, CatalogSourceConfig, CatalogStorageAccounting,
};
use sceneworks_core::contracts::{JobSnapshot, JobType};
use sceneworks_core::jobs_store::CreateJob;
use sceneworks_core::training::{CaptionSource, TrainingDataset, TrainingDatasetStatus};
use sceneworks_core::training_store::{
    CaptionInput, ExternalTrainingDatasetItemInput, TrainingDatasetCreateInput,
    TrainingDatasetItemInput,
};
use sceneworks_worker::catalog_parquet_scanner::{
    validate_catalog_parquet_scan_plan_with_cancel, AttachedCatalogParquetScanDriver,
    CatalogParquetScanError, CatalogParquetScanOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::catalog_scan_supervisor::CatalogScanSpawn;
use crate::{project_call, ApiError, ApiJson, AppState};

const CATALOG_API_CONTRACT_VERSION: u32 = 1;
const DEFAULT_QUERY_LIMIT: u32 = 50;
const MAX_QUERY_LIMIT: u32 = 200;
const DEFAULT_FACET_LIMIT: u32 = 50;
const MAX_FACET_LIMIT: u32 = 100;
const SCHEDULED_SCAN_PASS_ROWS: u64 = 25_000;
const PARQUET_SCAN_CHECKPOINT_KEY: &str = "scanner.parquet.checkpoint.v1";
const CATALOG_SCAN_START_MAX_BACKOFF: std::time::Duration = std::time::Duration::from_millis(250);
pub(crate) const CATALOG_SCAN_MAX_CONTENTION_RETRIES: u8 = 6;
const CATALOG_PREFLIGHT_ADMISSION_WAIT: std::time::Duration = std::time::Duration::from_millis(250);
const CATALOG_PREFLIGHT_EXECUTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const MAX_MATERIALIZED_ITEMS: u32 = 500;
const MAX_MATERIALIZATION_CANDIDATES: usize = 10_000;
const MAX_MATERIALIZED_IMAGE_BYTES: usize = 32 * 1024 * 1024;
const MATERIALIZATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);
const MATERIALIZATION_CANDIDATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
const MAX_SAFE_INTEGER_SEED: u64 = 9_007_199_254_740_991;
const MAX_ANALYZER_PROVENANCE_STRING_BYTES: usize = 512;
const MAX_ANALYZER_PROVENANCE_STAGES: usize = 16;
const MAX_CATALOG_CAPTION_COMPONENT_BYTES: usize = 2_048;
const MAX_CATALOG_GENERATED_TAGS: usize = 32;
const MAX_CATALOG_GENERATED_TAG_BYTES: usize = 128;
const MAX_MATERIALIZED_CAPTION_BYTES: usize = 8_192;

#[cfg(test)]
struct CatalogCacheStageTestHook {
    claimed: std::sync::atomic::AtomicBool,
    before_stage: std::sync::Arc<std::sync::Barrier>,
    release_stage: std::sync::Arc<std::sync::Barrier>,
    stage_completed: std::sync::Arc<std::sync::Barrier>,
    timeout_observed: std::sync::Arc<std::sync::Barrier>,
    before_project_copy: std::sync::Arc<std::sync::Barrier>,
    release_project_copy: std::sync::Arc<std::sync::Barrier>,
}

#[cfg(test)]
fn catalog_cache_stage_test_hook(
) -> &'static std::sync::Mutex<Option<std::sync::Arc<CatalogCacheStageTestHook>>> {
    static HOOK: std::sync::OnceLock<
        std::sync::Mutex<Option<std::sync::Arc<CatalogCacheStageTestHook>>>,
    > = std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
pub(crate) struct CatalogCacheStageTestControl {
    hook: std::sync::Arc<CatalogCacheStageTestHook>,
}

#[cfg(test)]
impl CatalogCacheStageTestControl {
    async fn wait(barrier: std::sync::Arc<std::sync::Barrier>) {
        tokio::task::spawn_blocking(move || barrier.wait())
            .await
            .expect("catalog materialization test barrier joins");
    }

    pub(crate) async fn wait_before_stage(&self) {
        Self::wait(self.hook.before_stage.clone()).await;
    }

    pub(crate) async fn wait_for_timeout(&self) {
        Self::wait(self.hook.timeout_observed.clone()).await;
    }

    pub(crate) async fn wait_before_project_copy(&self) {
        Self::wait(self.hook.before_project_copy.clone()).await;
    }

    pub(crate) async fn release_stage(&self) {
        Self::wait(self.hook.release_stage.clone()).await;
    }

    pub(crate) async fn wait_for_stage_completion(&self) {
        Self::wait(self.hook.stage_completed.clone()).await;
    }

    pub(crate) async fn release_project_copy(&self) {
        Self::wait(self.hook.release_project_copy.clone()).await;
    }
}

#[cfg(test)]
impl Drop for CatalogCacheStageTestControl {
    fn drop(&mut self) {
        let mut current = catalog_cache_stage_test_hook()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if current
            .as_ref()
            .is_some_and(|hook| std::sync::Arc::ptr_eq(hook, &self.hook))
        {
            *current = None;
        }
    }
}

#[cfg(test)]
pub(crate) fn install_catalog_cache_stage_test_hook() -> CatalogCacheStageTestControl {
    let barrier = || std::sync::Arc::new(std::sync::Barrier::new(2));
    let hook = std::sync::Arc::new(CatalogCacheStageTestHook {
        claimed: std::sync::atomic::AtomicBool::new(false),
        before_stage: barrier(),
        release_stage: barrier(),
        stage_completed: barrier(),
        timeout_observed: barrier(),
        before_project_copy: barrier(),
        release_project_copy: barrier(),
    });
    *catalog_cache_stage_test_hook()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook.clone());
    CatalogCacheStageTestControl { hook }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CreateCatalogRequest {
    name: String,
    path: PathBuf,
    #[serde(default)]
    source_config: Option<CatalogSourceConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AttachCatalogRequest {
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CatalogQueryRequest {
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default = "default_query_limit")]
    limit: u32,
    #[serde(default)]
    filters: Vec<CatalogRecordFilter>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CatalogFacetsRequest {
    fields: Vec<String>,
    #[serde(default)]
    filters: Vec<CatalogRecordFilter>,
    #[serde(default = "default_facet_limit")]
    limit_per_facet: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CatalogCurationFacetsRequest {
    query: CatalogSearchRequest,
    fields: Vec<String>,
    #[serde(default = "default_facet_limit")]
    limit_per_facet: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CatalogRecordReviewRequest {
    decision: CatalogReviewDecision,
    #[serde(default)]
    rejection_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SaveCatalogViewRequest {
    name: String,
    query: CatalogSearchRequest,
    #[serde(default)]
    expected_revision: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct DeleteCatalogViewRequest {
    expected_revision: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CatalogControlRequest {
    expected_revision: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UpdateCatalogAnalyzerConfigRequest {
    expected_revision: u64,
    settings: CatalogAnalyzerSettings,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RunCatalogAnalysisRequest {
    expected_analyzer_config_revision: u64,
    #[serde(default)]
    requested_gpu: String,
    #[serde(default = "default_catalog_analysis_batch_size")]
    batch_size: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct MaterializeCatalogRequest {
    project_id: String,
    name: String,
    requested_count: u32,
    seed: u64,
    deduplication_policy: String,
    query: CatalogSearchRequest,
    #[serde(default)]
    include_generated_caption: bool,
    #[serde(default)]
    include_generated_tags: bool,
    idempotency_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MaterializationSkip {
    record_id: String,
    reason: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MaterializeCatalogResponse {
    dataset: TrainingDataset,
    requested_count: u32,
    materialized_count: u32,
    selected_record_ids: Vec<String>,
    skipped: Vec<MaterializationSkip>,
    reused_existing: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogCountsResponse {
    record_count: u64,
    candidate_count: u64,
    processed_count: u64,
    accepted_count: u64,
    rejected_count: u64,
    error_count: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogStorageResponse {
    database_bytes: u64,
    manifest_bytes: u64,
    artifact_bytes: u64,
    total_bytes: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogResponse {
    contract_version: u32,
    id: String,
    name: String,
    path: PathBuf,
    attached_at: String,
    created_at: Option<String>,
    schema_version: Option<u32>,
    availability: &'static str,
    source_config: Option<CatalogSourceConfig>,
    analyzer_versions: BTreeMap<String, String>,
    checkpoints: BTreeMap<String, Value>,
    counts: Option<CatalogCountsResponse>,
    storage: Option<CatalogStorageResponse>,
    processing: CatalogProcessingProgress,
    processing_control: CatalogProcessingControl,
    analyzer_config: CatalogAnalyzerConfig,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogQueryResponse {
    contract_version: u32,
    items: Vec<CatalogRecord>,
    next_cursor: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogCurationQueryResponse {
    contract_version: u32,
    items: Vec<CatalogRecord>,
    reviews: Vec<CatalogRecordReview>,
    next_cursor: Option<String>,
    total_count: Option<u64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogFacetsResponse {
    contract_version: u32,
    facets: Vec<CatalogFacet>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogLifecycleResponse {
    contract_version: u32,
    id: String,
    detached: bool,
    deleted_on_disk: bool,
}

#[derive(Clone)]
struct CatalogScanPlan {
    source: PathBuf,
    options: CatalogParquetScanOptions,
}

pub(crate) async fn list_catalogs(
    State(state): State<AppState>,
) -> Result<Json<Vec<CatalogResponse>>, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let responses = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        registry
            .list()?
            .into_iter()
            .map(|attached| match registry.open_attached(&attached.id) {
                Ok(catalog) => catalog_response(&attached, &catalog),
                Err(_) => Ok(unavailable_catalog_response(attached)),
            })
            .collect::<Result<Vec<_>, CatalogError>>()
    })
    .await?;
    Ok(Json(responses))
}

pub(crate) async fn create_catalog(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<CreateCatalogRequest>,
) -> Result<(StatusCode, Json<CatalogResponse>), ApiError> {
    require_absolute_catalog_path(&request.path)?;
    validated_scan_plan(&state, request.source_config.as_ref()).await?;
    let config_dir = state.settings.config_dir.clone();
    let scheduling_config_dir = config_dir.clone();
    let (response, scan_plan) = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let catalog = registry.create_catalog(&request.path, request.name)?;
        let attached = registry.get(&catalog.descriptor().id)?;
        let contract_state = CatalogContractState {
            source_config: request.source_config,
            ..CatalogContractState::default()
        };
        if let Err(error) = catalog.set_contract_state(&contract_state) {
            let catalog_id = catalog.descriptor().id.clone();
            catalog.close();
            let _ = registry.delete_on_disk(&catalog_id);
            return Err(error);
        }
        let contract_state = catalog.contract_state()?;
        let scan_plan = scan_plan(contract_state.source_config.as_ref())?;
        Ok((catalog_response(&attached, &catalog)?, scan_plan))
    })
    .await?;
    if let Some(scan_plan) = scan_plan {
        schedule_catalog_scan(
            &state,
            scheduling_config_dir,
            response.id.clone(),
            scan_plan,
        )
        .await;
    }
    Ok((StatusCode::CREATED, Json(response)))
}

pub(crate) async fn attach_catalog(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<AttachCatalogRequest>,
) -> Result<Json<CatalogResponse>, ApiError> {
    require_absolute_catalog_path(&request.path)?;
    let config_dir = state.settings.config_dir.clone();
    let response = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let attached = registry.attach(request.path)?;
        let catalog = registry.open_attached(&attached.id)?;
        catalog_response(&attached, &catalog)
    })
    .await?;
    Ok(Json(response))
}

pub(crate) async fn get_catalog(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
) -> Result<Json<CatalogResponse>, ApiError> {
    Ok(Json(
        load_catalog_response(state, catalog_id, CatalogStorageMode::Exact).await?,
    ))
}

pub(crate) async fn get_catalog_status(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
) -> Result<Json<CatalogResponse>, ApiError> {
    // Artifact bytes remain exact on detail/list responses. This polling route
    // omits storage instead of recursively walking a potentially huge catalog
    // or returning a stale cached total.
    Ok(Json(
        load_catalog_response(state, catalog_id, CatalogStorageMode::Omit).await?,
    ))
}

pub(crate) async fn query_catalog(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<CatalogQueryRequest>,
) -> Result<Json<CatalogQueryResponse>, ApiError> {
    if request.limit == 0 || request.limit > MAX_QUERY_LIMIT {
        return Err(ApiError::bad_request(format!(
            "Catalog query limit must be between 1 and {MAX_QUERY_LIMIT}"
        )));
    }
    let cursor = parse_cursor(request.cursor.as_deref())?;
    let config_dir = state.settings.config_dir.clone();
    let response = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let catalog = registry.open_attached(&catalog_id)?;
        let page = catalog.query_records_after(cursor, request.limit, &request.filters)?;
        Ok(CatalogQueryResponse {
            contract_version: CATALOG_API_CONTRACT_VERSION,
            items: page.records,
            next_cursor: page.next_cursor.map(|cursor| cursor.to_string()),
        })
    })
    .await?;
    Ok(Json(response))
}

pub(crate) async fn curate_catalog(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<CatalogSearchRequest>,
) -> Result<Json<CatalogCurationQueryResponse>, ApiError> {
    if request.limit == 0 || request.limit > MAX_QUERY_LIMIT {
        return Err(ApiError::bad_request(format!(
            "Catalog query limit must be between 1 and {MAX_QUERY_LIMIT}"
        )));
    }
    let config_dir = state.settings.config_dir.clone();
    let response = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let catalog = registry.open_attached(&catalog_id)?;
        let page = catalog.search_records(&request)?;
        let record_ids = page
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        Ok(CatalogCurationQueryResponse {
            contract_version: CATALOG_API_CONTRACT_VERSION,
            reviews: catalog.record_reviews(&record_ids)?,
            items: page.records,
            next_cursor: page.next_cursor,
            total_count: page.total_count,
        })
    })
    .await?;
    Ok(Json(response))
}

pub(crate) async fn materialize_catalog_results(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(mut request): ApiJson<MaterializeCatalogRequest>,
) -> Result<(StatusCode, Json<MaterializeCatalogResponse>), ApiError> {
    validate_materialization_request(&request)?;
    let deduplicate = match request.deduplication_policy.as_str() {
        "content_and_perceptual" => true,
        "none" => false,
        _ => {
            return Err(ApiError::bad_request(
                "deduplicationPolicy must be content_and_perceptual or none",
            ))
        }
    };
    request.query.cursor = None;
    request.query.limit = MAX_QUERY_LIMIT;
    request.query.sample_seed = Some(request.seed);
    request.query.deduplicate = deduplicate;
    request.query.include_total = false;
    canonicalize_search_query(&mut request.query);
    request.name = request.name.trim().to_owned();
    let request_fingerprint = materialization_request_fingerprint(&catalog_id, &request)?;
    if let Some(existing) = project_call(state.clone(), {
        let project_id = request.project_id.clone();
        let idempotency_key = request.idempotency_key.clone();
        move |store| store.get_catalog_materialization(&project_id, &idempotency_key)
    })
    .await?
    {
        return Ok((
            StatusCode::OK,
            Json(materialization_response_from_dataset(
                existing,
                true,
                &request_fingerprint,
            )?),
        ));
    }

    let config_dir = state.settings.config_dir.clone();
    let registry = CatalogRegistry::new(config_dir);
    let catalog = registry
        .open_attached(&catalog_id)
        .map_err(ApiError::from)?;
    let _lifecycle_lease = CatalogProcessingLease::try_acquire(&catalog).map_err(ApiError::from)?;
    let descriptor = catalog.descriptor().clone();
    let contract = catalog.contract_state().map_err(ApiError::from)?;
    let analyzer_config = catalog.analyzer_config().map_err(ApiError::from)?;
    let catalog_generation = catalog.content_generation().map_err(ApiError::from)?;
    let catalog_root = fs::canonicalize(catalog.root())
        .map_err(|error| ApiError::conflict(format!("Catalog storage is unavailable: {error}")))?;
    let canonical_query = canonical_json(
        serde_json::to_value(&request.query)
            .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    let catalog_snapshot = canonical_json(serde_json::json!({
        "id": descriptor.id,
        "createdAt": descriptor.created_at,
        "schemaVersion": descriptor.schema_version,
        "contract": &contract,
        "analyzerConfig": &analyzer_config,
    }));
    let staging = tempfile::tempdir().map_err(|error| {
        ApiError::internal(format!("Could not create dataset staging: {error}"))
    })?;
    let candidate_limit = (request.requested_count as usize)
        .saturating_mul(20)
        .max(request.requested_count as usize)
        .min(MAX_MATERIALIZATION_CANDIDATES);
    let mut selected = Vec::with_capacity(request.requested_count as usize);
    let mut selected_ids = Vec::with_capacity(request.requested_count as usize);
    let mut selected_versions = Vec::with_capacity(request.requested_count as usize);
    let mut skipped = Vec::new();
    let mut actual_hashes = BTreeSet::new();
    let mut cursor = None;
    let mut examined = 0_usize;
    let deadline = tokio::time::Instant::now() + MATERIALIZATION_TIMEOUT;

    'pages: loop {
        request.query.cursor = cursor.take();
        let page = catalog
            .search_records(&request.query)
            .map_err(ApiError::from)?;
        for record in page.records {
            if examined >= candidate_limit {
                break 'pages;
            }
            examined += 1;
            // A timed-out spawn_blocking verifier cannot be canceled. Use the
            // monotonic attempt number so its eventual write can never replace
            // a later accepted candidate.
            let staging_token = examined;
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break 'pages;
            }
            let acquired = tokio::time::timeout(
                remaining.min(materialization_candidate_timeout()),
                acquire_materialization_source(
                    &catalog_root,
                    staging.path(),
                    staging_token,
                    &record,
                ),
            )
            .await;
            let acquired = match acquired {
                Ok(result) => result,
                Err(_) => {
                    signal_catalog_cache_timeout_test_hook();
                    Err(ApiError::conflict("Catalog record image fetch timed out"))
                }
            };
            let acquired = match acquired {
                Ok(acquired) => acquired,
                Err(_) => {
                    skipped.push(MaterializationSkip {
                        record_id: record.id,
                        reason: "unavailable".to_owned(),
                    });
                    continue;
                }
            };
            if deduplicate && !actual_hashes.insert(acquired.content_hash.clone()) {
                skipped.push(MaterializationSkip {
                    record_id: record.id,
                    reason: "duplicate_content".to_owned(),
                });
                continue;
            }
            let caption = catalog_caption(
                &record.metadata,
                request.include_generated_caption,
                request.include_generated_tags,
            );
            let item_id = format!("item_{:04}", selected.len() + 1);
            selected_ids.push(record.id.clone());
            selected_versions.push(serde_json::json!({
                "recordId": &record.id,
                "recordUpdatedAt": &record.updated_at,
                "contentHash": &acquired.content_hash,
                "analyzerProvenance": catalog_analyzer_provenance(&record.metadata),
            }));
            let mut item_extra = BTreeMap::new();
            item_extra.insert(
                "catalogSource".to_owned(),
                catalog_source_snapshot(&record.metadata),
            );
            selected.push(ExternalTrainingDatasetItemInput {
                item: TrainingDatasetItemInput {
                    id: Some(item_id),
                    asset_id: None,
                    path: None,
                    display_name: Some(record.id.clone()),
                    caption: Some(CaptionInput {
                        text: caption,
                        source: Some(CaptionSource::Imported),
                        trigger_words: Vec::new(),
                    }),
                    width: Some(acquired.width),
                    height: Some(acquired.height),
                },
                source_path: acquired.path,
                expected_content_hash: acquired.content_hash,
                expected_width: acquired.width,
                expected_height: acquired.height,
                extra: item_extra,
            });
            if selected.len() == request.requested_count as usize {
                break 'pages;
            }
        }
        let Some(next) = page.next_cursor else {
            break;
        };
        cursor = Some(next);
    }
    if catalog.content_generation().map_err(ApiError::from)? != catalog_generation {
        return Err(ApiError::conflict(
            "Catalog changed while materialization was selecting records; retry the request",
        ));
    }
    drop(catalog);
    drop(_lifecycle_lease);

    if selected.len() != request.requested_count as usize {
        return Err(ApiError::conflict(format!(
            "Only {} verified images were available after deterministically examining {examined} candidates; requested {}",
            selected.len(),
            request.requested_count
        )));
    }

    let selected_snapshot = selected_ids.clone();
    let skipped_snapshot = skipped.clone();
    let materialization_fingerprint = hex_sha256(
        &serde_json::to_vec(&canonical_json(serde_json::json!({
            "catalog": &catalog_snapshot,
            "catalogGeneration": catalog_generation,
            "requestFingerprint": &request_fingerprint,
            "query": &canonical_query,
            "selectedRecords": &selected_versions,
        })))
        .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    let requested_count = request.requested_count;
    let project_id = request.project_id.clone();
    let idempotency_key = request.idempotency_key.clone();
    let mut extra = BTreeMap::new();
    extra.insert(
        "catalogMaterialization".to_owned(),
        serde_json::json!({
            "contractVersion": 1,
            "idempotencyKey": idempotency_key,
            "requestFingerprint": &request_fingerprint,
            "catalog": {
                "id": catalog_id,
                "schemaVersion": descriptor.schema_version,
                "generation": catalog_generation,
            },
            "materializationFingerprint": materialization_fingerprint,
            "query": canonical_query,
            "requestedCount": requested_count,
            "materializedCount": selected_snapshot.len(),
            "selectedRecordIds": selected_snapshot,
            "selectedRecordVersions": selected_versions,
            "skipped": skipped_snapshot,
            "analyzerVersions": contract.analyzer_versions,
            "seed": request.seed,
            "deduplicationPolicy": request.deduplication_policy,
            "captionOptions": {
                "source": true,
                "generatedCaption": request.include_generated_caption,
                "generatedTags": request.include_generated_tags,
            },
        }),
    );
    let create = TrainingDatasetCreateInput {
        name: request.name,
        modality: None,
        status: Some(TrainingDatasetStatus::Draft),
        character_id: None,
        items: selected.iter().map(|value| value.item.clone()).collect(),
    };
    let response_fingerprint = request_fingerprint.clone();
    let (dataset, reused_existing) = project_call(state, move |store| {
        run_before_project_copy_test_hook();
        let _staging_guard = staging;
        store.create_training_dataset_from_external(
            &project_id,
            create,
            selected,
            extra,
            &idempotency_key,
        )
    })
    .await?;
    let status = if reused_existing {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    };
    Ok((
        status,
        Json(materialization_response_from_dataset(
            dataset,
            reused_existing,
            &response_fingerprint,
        )?),
    ))
}

struct AcquiredMaterializationSource {
    path: PathBuf,
    width: u32,
    height: u32,
    content_hash: String,
}

async fn acquire_materialization_source(
    catalog_root: &FsPath,
    staging_root: &FsPath,
    index: usize,
    record: &CatalogRecord,
) -> Result<AcquiredMaterializationSource, ApiError> {
    if let Some(relative) = record
        .metadata
        .pointer("/acquisition/cachePath")
        .and_then(Value::as_str)
    {
        let catalog_root = catalog_root.to_owned();
        let staging_root = staging_root.to_owned();
        let relative = relative.to_owned();
        let record = record.clone();
        let cached = tokio::task::spawn_blocking(move || {
            verified_cached_source(&catalog_root, &staging_root, index, &relative, &record)
        })
        .await
        .map_err(|error| ApiError::internal(format!("Catalog cache verifier failed: {error}")))?;
        if let Ok(source) = cached {
            return Ok(source);
        }
    }
    let url = record
        .metadata
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::conflict("Catalog record has no available image source"))?;
    let verified = sceneworks_worker::catalog_image_fetch::fetch_verified_catalog_image(
        url,
        MAX_MATERIALIZED_IMAGE_BYTES,
    )
    .await
    .map_err(|_| ApiError::conflict("Catalog record image was unavailable"))?;
    verify_record_content_hash(record, &verified.content_hash)?;
    let path = staging_root.join(format!("candidate-{index}.{}", verified.extension));
    tokio::fs::write(&path, &verified.bytes)
        .await
        .map_err(|error| ApiError::internal(format!("Could not stage catalog image: {error}")))?;
    Ok(AcquiredMaterializationSource {
        path,
        width: verified.width,
        height: verified.height,
        content_hash: verified.content_hash,
    })
}

fn verified_cached_source(
    catalog_root: &FsPath,
    staging_root: &FsPath,
    index: usize,
    relative: &str,
    record: &CatalogRecord,
) -> Result<AcquiredMaterializationSource, ApiError> {
    let path = FsPath::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ApiError::conflict("Catalog cache path was invalid"));
    }
    let canonical = fs::canonicalize(catalog_root.join(path))
        .map_err(|_| ApiError::conflict("Catalog cache image was unavailable"))?;
    if !canonical.starts_with(catalog_root) || !canonical.is_file() {
        return Err(ApiError::conflict("Catalog cache path escaped its catalog"));
    }
    let byte_count = fs::metadata(&canonical)
        .map_err(|_| ApiError::conflict("Catalog cache image was unavailable"))?
        .len();
    if byte_count == 0 || byte_count > MAX_MATERIALIZED_IMAGE_BYTES as u64 {
        return Err(ApiError::conflict(
            "Catalog cache image exceeded the materialization limit",
        ));
    }
    let bytes = fs::read(&canonical)
        .map_err(|_| ApiError::conflict("Catalog cache image could not be read"))?;
    let verified = sceneworks_worker::catalog_image_fetch::verify_catalog_image_bytes(
        bytes,
        MAX_MATERIALIZED_IMAGE_BYTES,
    )
    .map_err(|_| ApiError::conflict("Catalog cache image format was invalid"))?;
    let content_hash = verified.content_hash;
    verify_record_content_hash(record, &content_hash)?;
    #[cfg(test)]
    let claimed_test_hook = claim_catalog_cache_stage_test_hook();
    #[cfg(test)]
    if let Some(hook) = &claimed_test_hook {
        hook.before_stage.wait();
        hook.release_stage.wait();
    }
    let staged = staging_root.join(format!("candidate-{index}.{}", verified.extension));
    let write_result = fs::write(&staged, &verified.bytes);
    #[cfg(test)]
    if let Some(hook) = claimed_test_hook {
        hook.stage_completed.wait();
    }
    write_result
        .map_err(|error| ApiError::internal(format!("Could not stage catalog image: {error}")))?;
    Ok(AcquiredMaterializationSource {
        path: staged,
        width: verified.width,
        height: verified.height,
        content_hash,
    })
}

fn verify_record_content_hash(record: &CatalogRecord, actual: &str) -> Result<(), ApiError> {
    if record
        .metadata
        .pointer("/acquisition/contentHash")
        .and_then(Value::as_str)
        .is_some_and(|expected| expected != actual)
    {
        return Err(ApiError::conflict(
            "Catalog record image failed content verification",
        ));
    }
    Ok(())
}

fn materialization_candidate_timeout() -> std::time::Duration {
    #[cfg(test)]
    if catalog_cache_stage_test_hook()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
    {
        return std::time::Duration::from_millis(100);
    }
    MATERIALIZATION_CANDIDATE_TIMEOUT
}

#[cfg(test)]
fn claim_catalog_cache_stage_test_hook() -> Option<std::sync::Arc<CatalogCacheStageTestHook>> {
    use std::sync::atomic::Ordering;

    let hook = catalog_cache_stage_test_hook()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()?;
    hook.claimed
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .ok()
        .map(|_| hook)
}

#[cfg(test)]
fn signal_catalog_cache_timeout_test_hook() {
    if let Some(hook) = catalog_cache_stage_test_hook()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
    {
        hook.timeout_observed.wait();
    }
}

#[cfg(not(test))]
fn signal_catalog_cache_timeout_test_hook() {}

#[cfg(test)]
fn run_before_project_copy_test_hook() {
    if let Some(hook) = catalog_cache_stage_test_hook()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
    {
        hook.before_project_copy.wait();
        hook.release_project_copy.wait();
    }
}

#[cfg(not(test))]
fn run_before_project_copy_test_hook() {}

fn materialization_response_from_dataset(
    dataset: TrainingDataset,
    reused_existing: bool,
    expected_request_fingerprint: &str,
) -> Result<MaterializeCatalogResponse, ApiError> {
    let provenance = dataset
        .extra
        .get("catalogMaterialization")
        .ok_or_else(|| ApiError::internal("Materialized dataset provenance was missing"))?;
    if provenance.get("requestFingerprint").and_then(Value::as_str)
        != Some(expected_request_fingerprint)
    {
        return Err(ApiError::conflict(
            "idempotencyKey was already used for a different catalog materialization request",
        ));
    }
    let requested_count = provenance
        .get("requestedCount")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| ApiError::internal("Materialized dataset requestedCount was invalid"))?;
    let selected_record_ids = provenance
        .get("selectedRecordIds")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::internal("Materialized dataset selectedRecordIds was invalid"))?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                ApiError::internal("Materialized dataset selectedRecordIds was invalid")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let skipped = serde_json::from_value(
        provenance
            .get("skipped")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new())),
    )
    .map_err(|_| ApiError::internal("Materialized dataset skipped provenance was invalid"))?;
    let materialized_count = provenance
        .get("materializedCount")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| ApiError::internal("Materialized dataset materializedCount was invalid"))?;
    Ok(MaterializeCatalogResponse {
        materialized_count,
        dataset,
        requested_count,
        selected_record_ids,
        skipped,
        reused_existing,
    })
}

fn catalog_caption(metadata: &Value, generated_caption: bool, generated_tags: bool) -> String {
    let mut parts = Vec::new();
    if let Some(source) = metadata
        .get("caption")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        parts.push(truncate_utf8(source, MAX_CATALOG_CAPTION_COMPONENT_BYTES));
    }
    if generated_caption {
        for pointer in [
            "/analysis/visionLanguage/caption",
            "/analysis/generatedCaption",
            "/generatedCaption",
        ] {
            if let Some(caption) = metadata
                .pointer(pointer)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                let caption = truncate_utf8(caption, MAX_CATALOG_CAPTION_COMPONENT_BYTES);
                if !parts.iter().any(|value| value == &caption) {
                    parts.push(caption);
                }
                break;
            }
        }
    }
    if generated_tags {
        if let Some(tags) = metadata
            .pointer("/analysis/visionLanguage/tags")
            .and_then(Value::as_array)
        {
            let tags = tags
                .iter()
                .take(MAX_CATALOG_GENERATED_TAGS)
                .filter_map(|tag| {
                    tag.get("value")
                        .and_then(Value::as_str)
                        .or_else(|| tag.as_str())
                })
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| truncate_utf8(value, MAX_CATALOG_GENERATED_TAG_BYTES))
                .collect::<Vec<_>>();
            if !tags.is_empty() {
                parts.push(tags.join(", "));
            }
        }
    }
    truncate_utf8(&parts.join("\n"), MAX_MATERIALIZED_CAPTION_BYTES)
}

fn catalog_source_snapshot(metadata: &Value) -> Value {
    let source_caption = metadata.get("caption").and_then(Value::as_str);
    let generated_caption = [
        "/analysis/visionLanguage/caption",
        "/analysis/generatedCaption",
        "/generatedCaption",
    ]
    .into_iter()
    .find_map(|pointer| metadata.pointer(pointer).and_then(Value::as_str));
    let (source_caption, source_truncation) =
        bounded_text_snapshot(source_caption, MAX_CATALOG_CAPTION_COMPONENT_BYTES);
    let (generated_caption, generated_truncation) =
        bounded_text_snapshot(generated_caption, MAX_CATALOG_CAPTION_COMPONENT_BYTES);
    let (generated_tags, tag_truncation) = bounded_tag_snapshot(
        metadata
            .pointer("/analysis/visionLanguage/tags")
            .and_then(Value::as_array)
            .map(Vec::as_slice),
    );
    let mut truncation = serde_json::Map::new();
    if let Some(value) = source_truncation {
        truncation.insert("sourceCaption".to_owned(), value);
    }
    if let Some(value) = generated_truncation {
        truncation.insert("generatedCaption".to_owned(), value);
    }
    if let Some(value) = tag_truncation {
        truncation.insert("generatedTags".to_owned(), value);
    }
    serde_json::json!({
        "sourceCaption": source_caption,
        "generatedCaption": generated_caption,
        "generatedTags": generated_tags,
        "truncation": truncation,
    })
}

fn bounded_text_snapshot(value: Option<&str>, max_bytes: usize) -> (Value, Option<Value>) {
    let Some(value) = value else {
        return (Value::Null, None);
    };
    let bounded = truncate_utf8(value, max_bytes);
    let truncation = (bounded.len() != value.len()).then(|| {
        serde_json::json!({
            "truncated": true,
            "originalBytes": value.len(),
            "sha256": hex_sha256(value.as_bytes()),
        })
    });
    (Value::String(bounded), truncation)
}

fn bounded_tag_snapshot(tags: Option<&[Value]>) -> (Value, Option<Value>) {
    let Some(tags) = tags else {
        return (Value::Array(Vec::new()), None);
    };
    let mut bounded = Vec::new();
    let mut truncated_values = Vec::new();
    for (index, tag) in tags.iter().take(MAX_CATALOG_GENERATED_TAGS).enumerate() {
        let Some(value) = tag
            .get("value")
            .and_then(Value::as_str)
            .or_else(|| tag.as_str())
        else {
            continue;
        };
        let bounded_value = truncate_utf8(value, MAX_CATALOG_GENERATED_TAG_BYTES);
        if bounded_value.len() != value.len() {
            truncated_values.push(serde_json::json!({
                "index": index,
                "originalBytes": value.len(),
                "sha256": hex_sha256(value.as_bytes()),
            }));
        }
        let mut snapshot = serde_json::Map::new();
        snapshot.insert("value".to_owned(), Value::String(bounded_value));
        if let Some(confidence @ Value::Number(_)) = tag.get("confidence") {
            snapshot.insert("confidence".to_owned(), confidence.clone());
        }
        bounded.push(Value::Object(snapshot));
    }
    let truncated = tags.len() > MAX_CATALOG_GENERATED_TAGS || !truncated_values.is_empty();
    let truncation = truncated.then(|| {
        serde_json::json!({
            "truncated": true,
            "originalCount": tags.len(),
            "maxCount": MAX_CATALOG_GENERATED_TAGS,
            "truncatedValues": truncated_values,
        })
    });
    (Value::Array(bounded), truncation)
}

fn catalog_analyzer_provenance(metadata: &Value) -> Value {
    let mut provenance = serde_json::Map::new();
    if let Some(structured) = metadata
        .pointer("/analysis/structured")
        .and_then(Value::as_object)
    {
        let mut snapshot = serde_json::Map::new();
        copy_scalar(structured, "schemaVersion", &mut snapshot);
        copy_bounded_string(structured, "inputFingerprint", &mut snapshot);
        if let Some(stages) = structured.get("executionOrder").and_then(Value::as_array) {
            snapshot.insert(
                "executionOrder".to_owned(),
                Value::Array(
                    stages
                        .iter()
                        .take(MAX_ANALYZER_PROVENANCE_STAGES)
                        .filter_map(Value::as_str)
                        .map(|value| Value::String(bounded_provenance_string(value)))
                        .collect(),
                ),
            );
        }
        let analyzers: serde_json::Map<String, Value> = ["person", "face", "pose", "derived"]
            .into_iter()
            .filter_map(|stage| {
                bounded_analyzer_snapshot(
                    structured
                        .get(stage)
                        .and_then(|value| value.get("analyzer")),
                )
                .map(|value| (stage.to_owned(), value))
            })
            .collect();
        if !analyzers.is_empty() {
            snapshot.insert("analyzers".to_owned(), Value::Object(analyzers));
        }
        if !snapshot.is_empty() {
            provenance.insert("structured".to_owned(), Value::Object(snapshot));
        }
    }
    for (source_key, output_key) in [
        ("visionLanguage", "visionLanguage"),
        ("semanticEmbedding", "semanticEmbedding"),
    ] {
        if let Some(snapshot) =
            bounded_analyzer_snapshot(metadata.pointer(&format!("/analysis/{source_key}")))
        {
            provenance.insert(output_key.to_owned(), snapshot);
        }
    }
    Value::Object(provenance)
}

fn bounded_analyzer_snapshot(value: Option<&Value>) -> Option<Value> {
    let value = value?.as_object()?;
    let mut snapshot = serde_json::Map::new();
    for key in [
        "analyzerVersion",
        "modelId",
        "modelVersion",
        "modelRevision",
        "runtime",
        "runtimeVersion",
        "runtimeRevision",
        "backend",
        "provider",
        "taxonomyVersion",
        "inputFingerprint",
        "status",
    ] {
        copy_bounded_string(value, key, &mut snapshot);
    }
    copy_scalar(value, "schemaVersion", &mut snapshot);
    copy_scalar(value, "confidence", &mut snapshot);
    (!snapshot.is_empty()).then_some(Value::Object(snapshot))
}

fn copy_bounded_string(
    source: &serde_json::Map<String, Value>,
    key: &str,
    target: &mut serde_json::Map<String, Value>,
) {
    if let Some(value) = source.get(key).and_then(Value::as_str) {
        target.insert(
            key.to_owned(),
            Value::String(bounded_provenance_string(value)),
        );
    }
}

fn copy_scalar(
    source: &serde_json::Map<String, Value>,
    key: &str,
    target: &mut serde_json::Map<String, Value>,
) {
    if let Some(value @ (Value::Bool(_) | Value::Number(_))) = source.get(key) {
        target.insert(key.to_owned(), value.clone());
    }
}

fn bounded_provenance_string(value: &str) -> String {
    truncate_utf8(value, MAX_ANALYZER_PROVENANCE_STRING_BYTES)
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    let mut end = value.len().min(max_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

fn validate_materialization_request(request: &MaterializeCatalogRequest) -> Result<(), ApiError> {
    if request.project_id.trim().is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if request.name.trim().is_empty() || request.name.chars().count() > 120 {
        return Err(ApiError::bad_request(
            "name must contain 1 to 120 characters",
        ));
    }
    if !(1..=MAX_MATERIALIZED_ITEMS).contains(&request.requested_count) {
        return Err(ApiError::bad_request(format!(
            "requestedCount must be between 1 and {MAX_MATERIALIZED_ITEMS}"
        )));
    }
    if request.seed > MAX_SAFE_INTEGER_SEED {
        return Err(ApiError::bad_request(format!(
            "seed must be no greater than {MAX_SAFE_INTEGER_SEED}"
        )));
    }
    if request.idempotency_key.trim().is_empty() || request.idempotency_key.chars().count() > 128 {
        return Err(ApiError::bad_request(
            "idempotencyKey must contain 1 to 128 characters",
        ));
    }
    Ok(())
}

fn materialization_request_fingerprint(
    catalog_id: &str,
    request: &MaterializeCatalogRequest,
) -> Result<String, ApiError> {
    let value = canonical_json(serde_json::json!({
        "catalogId": catalog_id,
        "projectId": request.project_id,
        "name": request.name,
        "requestedCount": request.requested_count,
        "seed": request.seed,
        "deduplicationPolicy": request.deduplication_policy,
        "query": request.query,
        "includeGeneratedCaption": request.include_generated_caption,
        "includeGeneratedTags": request.include_generated_tags,
    }));
    let bytes =
        serde_json::to_vec(&value).map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(hex_sha256(&bytes))
}

fn canonicalize_search_query(query: &mut CatalogSearchRequest) {
    query
        .filters
        .sort_by(|left, right| left.field.cmp(&right.field));
    for filter in &mut query.filters {
        filter.values.sort();
        filter.values.dedup();
    }
}

fn canonical_json(value: Value) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, canonical_json(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_json).collect()),
        other => other,
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(crate) async fn review_catalog_record(
    State(state): State<AppState>,
    Path((catalog_id, record_id)): Path<(String, String)>,
    ApiJson(request): ApiJson<CatalogRecordReviewRequest>,
) -> Result<Json<Option<CatalogRecordReview>>, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let review = catalog_call(move || {
        CatalogRegistry::new(config_dir)
            .open_attached(&catalog_id)?
            .set_record_review(&record_id, request.decision, request.rejection_reason)
    })
    .await?;
    Ok(Json(review))
}

pub(crate) async fn list_catalog_saved_views(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
) -> Result<Json<Vec<CatalogSavedView>>, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let views = catalog_call(move || {
        CatalogRegistry::new(config_dir)
            .open_attached(&catalog_id)?
            .saved_views()
    })
    .await?;
    Ok(Json(views))
}

pub(crate) async fn create_catalog_saved_view(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<SaveCatalogViewRequest>,
) -> Result<(StatusCode, Json<CatalogSavedView>), ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let view = catalog_call(move || {
        CatalogRegistry::new(config_dir)
            .open_attached(&catalog_id)?
            .save_view(None, request.name, request.query, None)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(view)))
}

pub(crate) async fn update_catalog_saved_view(
    State(state): State<AppState>,
    Path((catalog_id, view_id)): Path<(String, String)>,
    ApiJson(request): ApiJson<SaveCatalogViewRequest>,
) -> Result<Json<CatalogSavedView>, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let view = catalog_call(move || {
        CatalogRegistry::new(config_dir)
            .open_attached(&catalog_id)?
            .save_view(
                Some(&view_id),
                request.name,
                request.query,
                request.expected_revision,
            )
    })
    .await?;
    Ok(Json(view))
}

pub(crate) async fn delete_catalog_saved_view(
    State(state): State<AppState>,
    Path((catalog_id, view_id)): Path<(String, String)>,
    ApiJson(request): ApiJson<DeleteCatalogViewRequest>,
) -> Result<StatusCode, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    catalog_call(move || {
        CatalogRegistry::new(config_dir)
            .open_attached(&catalog_id)?
            .delete_saved_view(&view_id, request.expected_revision)
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}

pub(crate) async fn catalog_record_thumbnail(
    State(state): State<AppState>,
    Path((catalog_id, record_id)): Path<(String, String)>,
) -> Result<Response<Body>, ApiError> {
    const MAX_THUMBNAIL_BYTES: u64 = 32 * 1024 * 1024;
    let config_dir = state.settings.config_dir.clone();
    let (bytes, content_type) = catalog_call(move || {
        let catalog = CatalogRegistry::new(config_dir).open_attached(&catalog_id)?;
        let record = catalog.record_by_id(&record_id)?;
        let relative = record
            .thumbnail_path
            .as_deref()
            .or_else(|| {
                record
                    .metadata
                    .pointer("/acquisition/thumbnailPath")
                    .and_then(Value::as_str)
            })
            .ok_or_else(|| CatalogError::NotFound("Catalog thumbnail was not found".to_owned()))?;
        let root = std::fs::canonicalize(catalog.root())?;
        let path = std::fs::canonicalize(catalog.root().join(relative)).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                CatalogError::NotFound("Catalog thumbnail was not found".to_owned())
            } else {
                CatalogError::Io(error)
            }
        })?;
        if !path.starts_with(&root) || !path.is_file() {
            return Err(CatalogError::InvalidCatalog(
                "Catalog thumbnail escapes its catalog".to_owned(),
            ));
        }
        let metadata = std::fs::metadata(&path)?;
        if metadata.len() > MAX_THUMBNAIL_BYTES {
            return Err(CatalogError::InvalidCatalog(
                "Catalog thumbnail exceeds its size limit".to_owned(),
            ));
        }
        let content_type = match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("jpg" | "jpeg") => "image/jpeg",
            Some("png") => "image/png",
            Some("webp") => "image/webp",
            Some("gif") => "image/gif",
            _ => "application/octet-stream",
        };
        Ok((std::fs::read(path)?, content_type))
    })
    .await?;
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CACHE_CONTROL, "private, max-age=300")
        .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff")
        .body(Body::from(bytes))
        .map_err(|error| ApiError::internal(format!("Catalog thumbnail response failed: {error}")))
}

pub(crate) async fn catalog_facets(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<CatalogFacetsRequest>,
) -> Result<Json<CatalogFacetsResponse>, ApiError> {
    if request.limit_per_facet == 0 || request.limit_per_facet > MAX_FACET_LIMIT {
        return Err(ApiError::bad_request(format!(
            "Catalog facet limit must be between 1 and {MAX_FACET_LIMIT}"
        )));
    }
    let config_dir = state.settings.config_dir.clone();
    let response = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let catalog = registry.open_attached(&catalog_id)?;
        Ok(CatalogFacetsResponse {
            contract_version: CATALOG_API_CONTRACT_VERSION,
            facets: catalog.facet_counts(
                &request.fields,
                &request.filters,
                request.limit_per_facet,
            )?,
        })
    })
    .await?;
    Ok(Json(response))
}

pub(crate) async fn catalog_curation_facets(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<CatalogCurationFacetsRequest>,
) -> Result<Json<CatalogFacetsResponse>, ApiError> {
    if request.limit_per_facet == 0 || request.limit_per_facet > MAX_FACET_LIMIT {
        return Err(ApiError::bad_request(format!(
            "Catalog facet limit must be between 1 and {MAX_FACET_LIMIT}"
        )));
    }
    let config_dir = state.settings.config_dir.clone();
    let response = catalog_call(move || {
        let catalog = CatalogRegistry::new(config_dir).open_attached(&catalog_id)?;
        Ok(CatalogFacetsResponse {
            contract_version: CATALOG_API_CONTRACT_VERSION,
            facets: catalog.search_facet_counts(
                &request.query,
                &request.fields,
                request.limit_per_facet,
            )?,
        })
    })
    .await?;
    Ok(Json(response))
}

pub(crate) async fn detach_catalog(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
) -> Result<Json<CatalogLifecycleResponse>, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let detached =
        catalog_call(move || CatalogRegistry::new(config_dir).detach(&catalog_id)).await?;
    Ok(Json(CatalogLifecycleResponse {
        contract_version: CATALOG_API_CONTRACT_VERSION,
        id: detached.id,
        detached: true,
        deleted_on_disk: false,
    }))
}

pub(crate) async fn delete_catalog_on_disk(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
) -> Result<Json<CatalogLifecycleResponse>, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let deleted =
        catalog_call(move || CatalogRegistry::new(config_dir).delete_on_disk(&catalog_id)).await?;
    Ok(Json(CatalogLifecycleResponse {
        contract_version: CATALOG_API_CONTRACT_VERSION,
        id: deleted.id,
        detached: true,
        deleted_on_disk: true,
    }))
}

pub(crate) async fn update_catalog_analyzer_config(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<UpdateCatalogAnalyzerConfigRequest>,
) -> Result<Json<CatalogResponse>, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let response = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let catalog = registry.open_attached(&catalog_id)?;
        match catalog.request_analyzer_config(request.expected_revision, request.settings) {
            Ok(_) => {}
            Err(CatalogError::Conflict(_)) => {
                return Err(CatalogError::Conflict(
                    "Catalog analyzer configuration changed; refresh and retry".to_owned(),
                ));
            }
            Err(error) => return Err(error),
        }
        let attached = registry.get(&catalog_id)?;
        catalog_response(&attached, &catalog)
    })
    .await
    .map_err(|error| {
        if error.status == StatusCode::CONFLICT {
            ApiError {
                status: StatusCode::CONFLICT,
                detail: "Catalog analyzer configuration changed; refresh and retry".to_owned(),
                code: Some("catalog_analyzer_config_conflict"),
            }
        } else {
            error
        }
    })?;
    Ok(Json(response))
}

pub(crate) async fn run_catalog_analysis(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<RunCatalogAnalysisRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if request.batch_size == 0 || request.batch_size > 64 {
        return Err(ApiError::bad_request(
            "Catalog analysis batchSize must be between 1 and 64",
        ));
    }
    let config_dir = state.settings.config_dir.clone();
    let validation_id = catalog_id.clone();
    let expected_revision = request.expected_analyzer_config_revision;
    let (catalog_name, analyzer_config) = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let catalog = registry.open_attached(&validation_id)?;
        let analyzer_config = catalog.analyzer_config()?;
        if analyzer_config.revision != expected_revision {
            return Err(CatalogError::Conflict(
                "Catalog analyzer configuration changed; refresh and retry".to_owned(),
            ));
        }
        if (analyzer_config.settings.vision_analysis_enabled
            || analyzer_config.settings.semantic_embeddings_enabled)
            && !analyzer_config.settings.structured_analysis_enabled
        {
            return Err(CatalogError::InvalidCatalog(
                "Vision analysis and semantic embeddings require structured analysis".to_owned(),
            ));
        }
        if CatalogProcessingLease::is_active(&catalog)? {
            return Err(CatalogError::Conflict(
                "Catalog processing is already active".to_owned(),
            ));
        }
        Ok((catalog.descriptor().name.clone(), analyzer_config))
    })
    .await?;
    let mut model_manifest_entries = Vec::new();
    if analyzer_config.settings.vision_analysis_enabled {
        model_manifest_entries.push(
            crate::models::resolve_model_manifest_entry(&state, "vision_caption_qwen3vl_8b")
                .await?,
        );
    }
    if analyzer_config.settings.semantic_embeddings_enabled {
        model_manifest_entries
            .push(crate::models::resolve_model_manifest_entry(&state, "clip_vit_l14").await?);
    }
    let requested_gpu = crate::requested_gpu_or_auto(request.requested_gpu);
    let batch_size = request.batch_size;
    let mut payload = serde_json::json!({
        "provider": "catalog",
        "kind": "catalog_analysis",
        "catalogId": catalog_id,
        "catalogName": catalog_name,
        "analyzerConfigRevision": analyzer_config.revision,
        "structuredAnalysisEnabled": analyzer_config.settings.structured_analysis_enabled,
        "batchSize": batch_size,
        "modelManifestEntries": model_manifest_entries,
    })
    .as_object()
    .cloned()
    .expect("catalog analysis payload object");
    crate::models::ensure_runtime_model_sources(&state, &JobType::CatalogAnalysis, &mut payload)
        .await?;
    let job = crate::store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type: JobType::CatalogAnalysis,
            project_id: None,
            project_name: None,
            payload,
            requested_gpu,
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
    })
    .await?;
    crate::publish(&state, "job.updated", &job);
    crate::publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn pause_catalog(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<CatalogControlRequest>,
) -> Result<Json<CatalogResponse>, ApiError> {
    request_processing_state(
        state,
        catalog_id,
        request.expected_revision,
        CatalogProcessingDesiredState::Paused,
        true,
    )
    .await
}

pub(crate) async fn resume_catalog(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
    ApiJson(request): ApiJson<CatalogControlRequest>,
) -> Result<Json<CatalogResponse>, ApiError> {
    request_processing_state(
        state,
        catalog_id,
        request.expected_revision,
        CatalogProcessingDesiredState::Running,
        false,
    )
    .await
}

async fn request_processing_state(
    state: AppState,
    catalog_id: String,
    expected_revision: u64,
    desired_state: CatalogProcessingDesiredState,
    require_active_processor: bool,
) -> Result<Json<CatalogResponse>, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let scan_plan = if desired_state == CatalogProcessingDesiredState::Running {
        let plan_config_dir = config_dir.clone();
        let plan_catalog_id = catalog_id.clone();
        let source_config = catalog_call(move || {
            CatalogRegistry::new(plan_config_dir)
                .open_attached(&plan_catalog_id)?
                .contract_state()
                .map(|state| state.source_config)
        })
        .await?;
        validated_scan_plan(&state, source_config.as_ref()).await?
    } else {
        None
    };
    let scheduling_config_dir = config_dir.clone();
    let response_scan_plan = scan_plan.clone();
    let (response, scan_plan) = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let attached = registry.get(&catalog_id)?;
        let catalog = registry.open_attached(&catalog_id)?;
        let inactive_lease = match CatalogProcessingLease::try_acquire(&catalog) {
            Ok(lease) => {
                lease.reconcile_interrupted(&catalog)?;
                Some(lease)
            }
            Err(CatalogError::Conflict(_)) => None,
            Err(error) => return Err(error),
        };
        let active = inactive_lease.is_none();
        if active != require_active_processor {
            return Err(CatalogError::Conflict(if require_active_processor {
                "Catalog has no active processor to pause".to_owned()
            } else {
                "Catalog processor has not stopped yet".to_owned()
            }));
        }
        catalog.request_processing_control(expected_revision, desired_state)?;
        Ok((catalog_response(&attached, &catalog)?, response_scan_plan))
    })
    .await?;
    if desired_state == CatalogProcessingDesiredState::Running {
        state
            .catalog_scan_invalid_recovery_reported
            .lock()
            .await
            .remove(&response.id);
    }
    if let Some(scan_plan) = scan_plan {
        schedule_catalog_scan(
            &state,
            scheduling_config_dir,
            response.id.clone(),
            scan_plan,
        )
        .await;
    }
    Ok(Json(response))
}

async fn load_catalog_response(
    state: AppState,
    catalog_id: String,
    storage_mode: CatalogStorageMode,
) -> Result<CatalogResponse, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    let mut invalid_reported = state.catalog_scan_invalid_recovery_reported.lock().await;
    let recovery_allowed = !invalid_reported.contains(&catalog_id);
    let recovery_catalog_id = catalog_id.clone();
    let (response, recovery) = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let attached = registry.get(&catalog_id)?;
        let catalog = registry.open_attached(&catalog_id)?;
        let recovery = if recovery_allowed {
            persisted_scan_recovery(&catalog)?
        } else {
            PersistedScanRecovery::None
        };
        Ok((
            catalog_response_unreconciled(&attached, &catalog, storage_mode)?,
            recovery,
        ))
    })
    .await?;
    if matches!(recovery, PersistedScanRecovery::Invalid) {
        invalid_reported.insert(recovery_catalog_id.clone());
    }
    if let PersistedScanRecovery::Schedule(plan) = recovery {
        schedule_catalog_scan(
            &state,
            state.settings.config_dir.clone(),
            recovery_catalog_id,
            *plan,
        )
        .await;
    }
    Ok(response)
}

#[derive(Clone, Copy)]
enum CatalogStorageMode {
    Exact,
    Omit,
}

fn catalog_response(
    attached: &AttachedCatalog,
    catalog: &Catalog,
) -> Result<CatalogResponse, CatalogError> {
    match CatalogProcessingLease::try_acquire(catalog) {
        Ok(lease) => {
            lease.reconcile_interrupted(catalog)?;
        }
        Err(CatalogError::Conflict(_)) => {}
        Err(error) => return Err(error),
    }
    catalog_response_unreconciled(attached, catalog, CatalogStorageMode::Exact)
}

fn catalog_response_unreconciled(
    attached: &AttachedCatalog,
    catalog: &Catalog,
    storage_mode: CatalogStorageMode,
) -> Result<CatalogResponse, CatalogError> {
    let contract_state = catalog.contract_state()?;
    let storage = match storage_mode {
        CatalogStorageMode::Exact => Some(catalog.storage_accounting()?),
        CatalogStorageMode::Omit => None,
    };
    let record_count = match storage {
        Some(storage) => storage.record_count,
        None => catalog.record_count()?,
    };
    Ok(CatalogResponse {
        contract_version: CATALOG_API_CONTRACT_VERSION,
        id: attached.id.clone(),
        name: catalog.descriptor().name.clone(),
        path: catalog.root().to_path_buf(),
        attached_at: attached.attached_at.clone(),
        created_at: Some(catalog.descriptor().created_at.clone()),
        schema_version: Some(catalog.descriptor().schema_version),
        availability: "available",
        source_config: contract_state.source_config,
        analyzer_versions: contract_state.analyzer_versions,
        checkpoints: contract_state.checkpoints,
        counts: Some(counts_response(record_count, &contract_state.processing)),
        storage: storage.map(storage_response),
        processing: contract_state.processing,
        processing_control: catalog.processing_control()?,
        analyzer_config: catalog.analyzer_config()?,
    })
}

enum PersistedScanRecovery {
    None,
    Schedule(Box<CatalogScanPlan>),
    Invalid,
}

fn persisted_scan_recovery(catalog: &Catalog) -> Result<PersistedScanRecovery, CatalogError> {
    let _lease = match CatalogProcessingLease::try_acquire(catalog) {
        Ok(lease) => lease,
        Err(CatalogError::Conflict(_)) => return Ok(PersistedScanRecovery::None),
        Err(error) => return Err(error),
    };
    let control = catalog.processing_control()?;
    let contract = catalog.contract_state()?;
    if control.desired_state != CatalogProcessingDesiredState::Running {
        _lease.reconcile_interrupted(catalog)?;
        return Ok(PersistedScanRecovery::None);
    }
    if contract.processing.state == CatalogProcessingState::Completed {
        return Ok(PersistedScanRecovery::None);
    }
    if contract.processing.state == CatalogProcessingState::Failed
        && !contract
            .processing
            .message
            .as_deref()
            .is_some_and(|message| message.contains("interrupted"))
    {
        // A real scanner failure is terminal until the operator explicitly
        // resumes it. Only durable interruption failures represent orphaned
        // work that status discovery is allowed to restart automatically.
        return Ok(PersistedScanRecovery::None);
    }
    if contract.source_config.is_none()
        && !contract
            .checkpoints
            .contains_key(PARQUET_SCAN_CHECKPOINT_KEY)
    {
        // Catalogs without scan provenance may be driven by another processor.
        // A Parquet checkpoint is the durable evidence that a missing source
        // configuration represents a broken persisted scan plan.
        _lease.reconcile_interrupted(catalog)?;
        return Ok(PersistedScanRecovery::None);
    }
    let plan = if contract
        .source_config
        .as_ref()
        .is_some_and(|source| source.options.get("maxRows").is_some())
    {
        Err(CatalogError::InvalidCatalog(
            "maxRows is scheduler-managed and cannot be supplied".to_owned(),
        ))
    } else {
        scan_plan(contract.source_config.as_ref())
    };
    match plan {
        Ok(Some(plan)) => Ok(PersistedScanRecovery::Schedule(Box::new(plan))),
        Ok(None) => {
            publish_invalid_persisted_scan(
                catalog,
                contract.processing,
                "Catalog scan was interrupted and cannot restart because its source configuration is missing",
            )?;
            Ok(PersistedScanRecovery::Invalid)
        }
        Err(CatalogError::InvalidCatalog(detail)) => {
            publish_invalid_persisted_scan(
                catalog,
                contract.processing,
                &format!("Catalog scan cannot restart: {detail}"),
            )?;
            Ok(PersistedScanRecovery::Invalid)
        }
        Err(error) => Err(error),
    }
}

fn publish_invalid_persisted_scan(
    catalog: &Catalog,
    mut progress: CatalogProcessingProgress,
    detail: &str,
) -> Result<(), CatalogError> {
    progress.state = CatalogProcessingState::Failed;
    progress.message = Some(format!("{detail}; repair the catalog source and resume"));
    progress.updated_at = sceneworks_core::catalog_store::catalog_timestamp_now();
    catalog.set_processing_progress(&progress)
}

async fn validated_scan_plan(
    state: &AppState,
    source_config: Option<&CatalogSourceConfig>,
) -> Result<Option<CatalogScanPlan>, ApiError> {
    if source_config.is_some_and(|source| source.options.get("maxRows").is_some()) {
        return Err(ApiError::bad_request(
            "maxRows is scheduler-managed and cannot be supplied",
        ));
    }
    let plan = scan_plan(source_config).map_err(actionable_scan_plan_error)?;
    if let Some(validation_plan) = plan.clone() {
        #[cfg(test)]
        let admission_wait = test_duration_override(
            &state.catalog_scan_preflight_admission_timeout_ms,
            CATALOG_PREFLIGHT_ADMISSION_WAIT,
        );
        #[cfg(not(test))]
        let admission_wait = CATALOG_PREFLIGHT_ADMISSION_WAIT;
        let permit = tokio::time::timeout(
            admission_wait,
            state.catalog_scan_preflight_slots.clone().acquire_owned(),
        )
        .await
        .map_err(|_| {
            ApiError::service_unavailable(
                "Parquet validation is busy; retry after another catalog preflight finishes",
            )
        })?
        .map_err(|_| ApiError::service_unavailable("Parquet validation is shutting down"))?;

        #[cfg(test)]
        let preflight_delay = state.catalog_scan_preflight_delay_ms_once.clone();
        #[cfg(test)]
        let preflight_started = state.catalog_scan_preflight_started.clone();
        #[cfg(test)]
        let preflight_ticks = state.catalog_scan_preflight_test_ticks.clone();
        #[cfg(test)]
        let execution_timeout = test_duration_override(
            &state.catalog_scan_preflight_execution_timeout_ms,
            CATALOG_PREFLIGHT_EXECUTION_TIMEOUT,
        );
        #[cfg(not(test))]
        let execution_timeout = CATALOG_PREFLIGHT_EXECUTION_TIMEOUT;

        let cancellation = tokio_util::sync::CancellationToken::new();
        let cancel_on_drop = CancelOnDrop(cancellation.clone());
        let validation_cancellation = cancellation.clone();
        let mut validation_task = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            #[cfg(test)]
            {
                let delay = preflight_delay.swap(0, std::sync::atomic::Ordering::SeqCst);
                if delay > 0 {
                    preflight_started.notify_waiters();
                    let deadline =
                        std::time::Instant::now() + std::time::Duration::from_millis(delay);
                    while std::time::Instant::now() < deadline {
                        if validation_cancellation.is_cancelled() {
                            return Err(CatalogParquetScanError::Interrupted(
                                "Parquet validation was canceled before it completed.".to_owned(),
                            ));
                        }
                        preflight_ticks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
            }
            validate_catalog_parquet_scan_plan_with_cancel(
                &validation_plan.source,
                &validation_plan.options,
                || validation_cancellation.is_cancelled(),
            )
        });
        let validation = match tokio::time::timeout(execution_timeout, &mut validation_task).await {
            Ok(joined) => joined.map_err(|error| {
                ApiError::internal(format!("Catalog preflight task failed: {error}"))
            })?,
            Err(_) => {
                cancellation.cancel();
                return Err(ApiError::service_unavailable(
                    "Parquet validation timed out; reduce the source shard count and retry",
                ));
            }
        };
        drop(cancel_on_drop);
        validation.map_err(actionable_scanner_validation_error)?;
    }
    Ok(plan)
}

struct CancelOnDrop(tokio_util::sync::CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

#[cfg(test)]
fn test_duration_override(
    override_ms: &std::sync::atomic::AtomicU64,
    default: std::time::Duration,
) -> std::time::Duration {
    match override_ms.swap(0, std::sync::atomic::Ordering::SeqCst) {
        0 => default,
        milliseconds => std::time::Duration::from_millis(milliseconds),
    }
}

fn actionable_scan_plan_error(error: CatalogError) -> ApiError {
    ApiError::bad_request(match error {
        CatalogError::InvalidCatalog(detail) => detail,
        _ => "Catalog source configuration is invalid".to_owned(),
    })
}

fn actionable_scanner_validation_error(error: CatalogParquetScanError) -> ApiError {
    match error {
        CatalogParquetScanError::InvalidSource(detail) => ApiError::bad_request(detail),
        CatalogParquetScanError::Interrupted(detail) => ApiError::service_unavailable(detail),
        _ => ApiError::bad_request("Parquet source could not be read with the supplied options"),
    }
}

fn scan_plan(
    source_config: Option<&CatalogSourceConfig>,
) -> Result<Option<CatalogScanPlan>, CatalogError> {
    let Some(source_config) = source_config else {
        return Ok(None);
    };
    if source_config.kind != "parquet" {
        return Err(CatalogError::InvalidCatalog(
            "Automatic catalog processing currently supports Parquet sources only".to_owned(),
        ));
    }
    let [source] = source_config.paths.as_slice() else {
        return Err(CatalogError::InvalidCatalog(
            "Parquet catalog processing requires exactly one source path".to_owned(),
        ));
    };
    if !source.is_absolute() {
        return Err(CatalogError::InvalidCatalog(
            "Parquet catalog source path must be absolute".to_owned(),
        ));
    }
    let options =
        serde_json::from_value::<CatalogParquetScanOptions>(source_config.options.clone())
            .map_err(|_| {
                CatalogError::InvalidCatalog("Parquet catalog scan options are invalid".to_owned())
            })?;
    Ok(Some(CatalogScanPlan {
        source: source.clone(),
        options,
    }))
}

/// Starts an idempotent bounded scanner loop owned by AppState.
///
/// The supervisor uses the catalog id as a single-flight key, requests
/// cooperative cancellation on shutdown, and drains the task before server
/// teardown returns.
async fn schedule_catalog_scan(
    state: &AppState,
    config_dir: PathBuf,
    catalog_id: String,
    mut plan: CatalogScanPlan,
) {
    plan.options.max_rows = Some(SCHEDULED_SCAN_PASS_ROWS);
    #[cfg(test)]
    let before_start = state.catalog_scan_before_driver_start_once.clone();
    #[cfg(test)]
    let stop_after_pass = state.catalog_scan_stop_after_pass_once.clone();
    #[cfg(test)]
    let before_terminal_exit = state.catalog_scan_before_terminal_exit_once.clone();
    #[cfg(test)]
    let terminal_exit_reached = state.catalog_scan_terminal_exit_reached.clone();
    #[cfg(test)]
    let terminal_exit_release = state.catalog_scan_terminal_exit_release.clone();
    #[cfg(test)]
    let injected_sqlite_busy_failures = state.catalog_scan_injected_sqlite_busy_failures.clone();
    #[cfg(test)]
    let contention_backoff_started = state.catalog_scan_contention_backoff_started.clone();
    let work_slots = state.catalog_scan_work_slots.clone();
    let log_catalog_id = catalog_id.clone();
    let rejected_config_dir = config_dir.clone();
    let rejected_catalog_id = catalog_id.clone();
    let spawned = state
        .catalog_scan_supervisor
        .spawn(catalog_id.clone(), move |cancellation| async move {
            #[cfg(test)]
            let start_barrier = { before_start.lock().take() };
            #[cfg(test)]
            if let Some(barrier) = start_barrier {
                tokio::select! {
                    _ = barrier.wait() => {}
                    _ = cancellation.cancelled() => {
                        reconcile_cancelled_catalog_scan(
                            config_dir.clone(),
                            catalog_id.clone(),
                        ).await;
                        return;
                    },
                }
            }
            let mut retry_delay = std::time::Duration::from_millis(10);
            let mut contention_retries = 0_u8;
            'driver_session: loop {
                let mut driver = loop {
                if cancellation.is_cancelled() {
                    reconcile_cancelled_catalog_scan(config_dir.clone(), catalog_id.clone()).await;
                    return;
                }
                let pass_config_dir = config_dir.clone();
                let pass_catalog_id = catalog_id.clone();
                let start = catalog_scan_blocking(work_slots.clone(), &cancellation, move || {
                    AttachedCatalogParquetScanDriver::try_start(
                        &CatalogRegistry::new(pass_config_dir),
                        &pass_catalog_id,
                    )
                })
                .await;
                match start {
                    Ok(Some(Ok(driver))) => break driver,
                    Ok(Some(Err(error))) if error.is_retryable_contention() => {
                        if contention_retries >= CATALOG_SCAN_MAX_CONTENTION_RETRIES {
                            publish_catalog_scan_contention_exhausted(
                                config_dir.clone(),
                                catalog_id.clone(),
                            )
                            .await;
                            return;
                        }
                        contention_retries = contention_retries.saturating_add(1);
                        tokio::select! {
                            _ = tokio::time::sleep(retry_delay) => {}
                            _ = cancellation.cancelled() => {
                                reconcile_cancelled_catalog_scan(
                                    config_dir.clone(),
                                    catalog_id.clone(),
                                ).await;
                                return;
                            },
                        }
                        retry_delay = retry_delay
                            .saturating_mul(2)
                            .min(CATALOG_SCAN_START_MAX_BACKOFF);
                    }
                    Ok(Some(Err(CatalogParquetScanError::Busy(_)))) => {
                        let retry_config_dir = config_dir.clone();
                        let retry_catalog_id = catalog_id.clone();
                        let should_retry =
                            catalog_scan_blocking(work_slots.clone(), &cancellation, move || {
                                let registry = CatalogRegistry::new(retry_config_dir);
                                let catalog = registry.open_attached(&retry_catalog_id)?;
                                let desired = catalog.processing_control()?.desired_state;
                                let actual = catalog.contract_state()?.processing.state;
                                Ok::<_, CatalogError>(
                                    desired == CatalogProcessingDesiredState::Running
                                        && actual != CatalogProcessingState::Completed,
                                )
                            })
                            .await;
                        match should_retry {
                            Ok(Some(Ok(true))) => {}
                            Ok(Some(Ok(false))) => {
                                #[cfg(test)]
                                catalog_scan_terminal_test_barrier(
                                    &before_terminal_exit,
                                    &terminal_exit_reached,
                                    &terminal_exit_release,
                                    &cancellation,
                                )
                                .await;
                                if cancellation.is_cancelled() {
                                    reconcile_cancelled_catalog_scan(
                                        config_dir.clone(),
                                        catalog_id.clone(),
                                    )
                                    .await;
                                }
                                return;
                            }
                            Ok(Some(Err(CatalogError::NotFound(_)))) => return,
                            Ok(None) => {
                                reconcile_cancelled_catalog_scan(
                                    config_dir.clone(),
                                    catalog_id.clone(),
                                )
                                .await;
                                return;
                            }
                            Ok(Some(Err(error))) => {
                                tracing::error!(
                                    event = "catalog_parquet_scan_retry_check_failed",
                                    catalog_id,
                                    error = %error,
                                    "catalog Parquet scanner stopped after a terminal retry check"
                                );
                                return;
                            }
                            Err(error) => {
                                tracing::error!(
                                    event = "catalog_parquet_scan_task_failed",
                                    catalog_id,
                                    error = %error,
                                    "catalog Parquet scanner retry check stopped"
                                );
                                return;
                            }
                        }
                        tokio::select! {
                            _ = tokio::time::sleep(retry_delay) => {}
                            _ = cancellation.cancelled() => {
                                reconcile_cancelled_catalog_scan(
                                    config_dir.clone(),
                                    catalog_id.clone(),
                                ).await;
                                return;
                            },
                        }
                        retry_delay = retry_delay
                            .saturating_mul(2)
                            .min(CATALOG_SCAN_START_MAX_BACKOFF);
                    }
                    Ok(Some(Err(error))) => {
                        tracing::error!(
                            event = "catalog_parquet_scan_start_failed",
                            catalog_id,
                            error = %error,
                            "catalog Parquet scanner could not start"
                        );
                        return;
                    }
                    Ok(None) => {
                        reconcile_cancelled_catalog_scan(config_dir.clone(), catalog_id.clone())
                            .await;
                        return;
                    }
                    Err(error) => {
                        tracing::error!(
                            event = "catalog_parquet_scan_task_failed",
                            catalog_id,
                            error = %error,
                            "catalog Parquet scanner start task stopped"
                        );
                        return;
                    }
                }
                };
                loop {
                let pass_source = plan.source.clone();
                let pass_options = plan.options.clone();
                let pass_cancellation = cancellation.clone();
                #[cfg(test)]
                let pass_injected_sqlite_busy_failures = injected_sqlite_busy_failures.clone();
                let result = catalog_scan_blocking(work_slots.clone(), &cancellation, move || {
                    #[cfg(test)]
                    if pass_injected_sqlite_busy_failures
                        .fetch_update(
                            std::sync::atomic::Ordering::SeqCst,
                            std::sync::atomic::Ordering::SeqCst,
                            |remaining| remaining.checked_sub(1),
                        )
                        .is_ok()
                    {
                        return (
                            driver,
                            Err(CatalogParquetScanError::Catalog(CatalogError::Sqlite(
                                rusqlite::Error::SqliteFailure(
                                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
                                    Some("injected scheduler contention".to_owned()),
                                ),
                            ))),
                        );
                    }
                    let result = driver.scan_pass_with_cancel(&pass_source, &pass_options, || {
                        pass_cancellation.is_cancelled()
                    });
                    (driver, result)
                })
                .await;
                match result {
                    Err(error) => {
                        tracing::error!(
                            event = "catalog_parquet_scan_task_failed",
                            catalog_id,
                            error = %error,
                            "catalog Parquet scanner task stopped"
                        );
                        break;
                    }
                    Ok(Some((next_driver, Ok(report)))) => {
                        driver = next_driver;
                        contention_retries = 0;
                        retry_delay = std::time::Duration::from_millis(10);
                        if report.paused || report.checkpoint.complete {
                            drop(driver);
                            #[cfg(test)]
                            catalog_scan_terminal_test_barrier(
                                &before_terminal_exit,
                                &terminal_exit_reached,
                                &terminal_exit_release,
                                &cancellation,
                            )
                            .await;
                            if cancellation.is_cancelled() {
                                reconcile_cancelled_catalog_scan(
                                    config_dir.clone(),
                                    catalog_id.clone(),
                                )
                                .await;
                            }
                            break 'driver_session;
                        }
                        #[cfg(test)]
                        if stop_after_pass.swap(false, std::sync::atomic::Ordering::SeqCst) {
                            break 'driver_session;
                        }
                        if cancellation.is_cancelled() {
                            driver.publish_interrupted();
                            break 'driver_session;
                        }
                    }
                    Ok(Some((_next_driver, Err(CatalogParquetScanError::Interrupted(error))))) => {
                        tracing::info!(
                            event = "catalog_parquet_scan_interrupted",
                            catalog_id,
                            error,
                            "catalog Parquet scanner stopped for server shutdown"
                        );
                        break 'driver_session;
                    }
                    Ok(Some((next_driver, Err(error)))) if error.is_retryable_contention() => {
                        tracing::debug!(
                            event = "catalog_parquet_scan_contention_retry",
                            catalog_id,
                            error = %error,
                            "catalog Parquet scanner will retry transient SQLite contention"
                        );
                        if contention_retries >= CATALOG_SCAN_MAX_CONTENTION_RETRIES {
                            if let Err(publish_error) = next_driver.publish_contention_exhausted() {
                                tracing::error!(
                                    event = "catalog_parquet_scan_contention_exhaustion_publish_failed",
                                    catalog_id,
                                    error = %publish_error,
                                    "catalog Parquet scanner could not publish contention exhaustion"
                                );
                            }
                            break 'driver_session;
                        }
                        contention_retries = contention_retries.saturating_add(1);
                        drop(next_driver);
                        #[cfg(test)]
                        contention_backoff_started.notify_waiters();
                        if cancellation.is_cancelled() {
                            reconcile_cancelled_catalog_scan(
                                config_dir.clone(),
                                catalog_id.clone(),
                            )
                            .await;
                            break 'driver_session;
                        }
                        tokio::select! {
                            _ = tokio::time::sleep(retry_delay) => {}
                            _ = cancellation.cancelled() => {
                                reconcile_cancelled_catalog_scan(
                                    config_dir.clone(),
                                    catalog_id.clone(),
                                ).await;
                                break 'driver_session;
                            },
                        }
                        retry_delay = retry_delay
                            .saturating_mul(2)
                            .min(CATALOG_SCAN_START_MAX_BACKOFF);
                        continue 'driver_session;
                    }
                    Ok(Some((_next_driver, Err(error)))) => {
                        tracing::error!(
                            event = "catalog_parquet_scan_failed",
                            catalog_id,
                            error = %error,
                            "catalog Parquet scanner stopped"
                        );
                        break 'driver_session;
                    }
                    Ok(None) => {
                        reconcile_cancelled_catalog_scan(config_dir.clone(), catalog_id.clone())
                            .await;
                        break 'driver_session;
                    }
                }
            }
            }
        })
        .await;
    match spawned {
        CatalogScanSpawn::Started => {}
        CatalogScanSpawn::RestartQueued => {
            tracing::debug!(
                event = "catalog_parquet_scan_restart_queued",
                catalog_id = log_catalog_id,
                "catalog Parquet scan restart was handed off to the live generation"
            );
        }
        CatalogScanSpawn::ShuttingDown => {
            reconcile_cancelled_catalog_scan(rejected_config_dir, rejected_catalog_id).await;
            tracing::warn!(
                event = "catalog_parquet_scan_rejected_during_shutdown",
                catalog_id = log_catalog_id,
                "catalog Parquet scan was not started because the server is shutting down"
            );
        }
    }
}

async fn publish_catalog_scan_contention_exhausted(config_dir: PathBuf, catalog_id: String) {
    let log_catalog_id = catalog_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        let registry = CatalogRegistry::new(config_dir);
        let catalog = registry.open_attached(&catalog_id)?;
        let _lease = CatalogProcessingLease::try_acquire(&catalog)?;
        let mut progress = catalog.contract_state()?.processing;
        progress.state = CatalogProcessingState::Failed;
        progress.message = Some(
            "Catalog scan stopped after repeated database contention; restart it to resume."
                .to_owned(),
        );
        progress.updated_at = sceneworks_core::catalog_store::catalog_timestamp_now();
        catalog.set_processing_progress(&progress)
    })
    .await;
    match result {
        Ok(Ok(())) => {
            tracing::error!(
                event = "catalog_parquet_scan_contention_exhausted",
                catalog_id = log_catalog_id,
                "catalog Parquet scanner exhausted bounded SQLite contention retries"
            );
        }
        Ok(Err(error)) => {
            tracing::error!(
                event = "catalog_parquet_scan_contention_exhaustion_publish_failed",
                catalog_id = log_catalog_id,
                error = %error,
                "catalog Parquet scanner exhausted retries and could not publish terminal state"
            );
        }
        Err(error) => {
            tracing::error!(
                event = "catalog_parquet_scan_contention_exhaustion_task_failed",
                catalog_id = log_catalog_id,
                error = %error,
                "catalog Parquet scanner exhaustion publisher stopped"
            );
        }
    }
}

async fn reconcile_cancelled_catalog_scan(config_dir: PathBuf, catalog_id: String) {
    let log_catalog_id = catalog_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        let registry = CatalogRegistry::new(config_dir);
        let catalog = match registry.open_attached(&catalog_id) {
            Ok(catalog) => catalog,
            Err(CatalogError::NotFound(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        let _lease = match CatalogProcessingLease::try_acquire(&catalog) {
            Ok(lease) => lease,
            Err(CatalogError::Conflict(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        let control = catalog.processing_control()?;
        let mut progress = catalog.contract_state()?.processing;
        if control.desired_state != CatalogProcessingDesiredState::Running
            || matches!(
                progress.state,
                CatalogProcessingState::Paused
                    | CatalogProcessingState::Completed
                    | CatalogProcessingState::Failed
            )
        {
            // A paused processor already published a clean durable boundary.
            // If shutdown races with its queued successor, preserve that state
            // and desired=running so a new AppState can discover and resume it.
            return Ok(false);
        }
        progress.state = CatalogProcessingState::Failed;
        progress.message = Some(
            "Catalog scan was interrupted before it could continue; restart it to resume"
                .to_owned(),
        );
        progress.updated_at = sceneworks_core::catalog_store::catalog_timestamp_now();
        catalog.set_processing_progress(&progress)?;
        Ok(true)
    })
    .await;
    match result {
        Ok(Ok(true)) => {
            tracing::info!(
                event = "catalog_parquet_scan_prestart_interrupted",
                catalog_id = log_catalog_id,
                "catalog Parquet scan cancellation was published for public restart"
            );
        }
        Ok(Ok(false)) => {}
        Ok(Err(error)) => {
            tracing::error!(
                event = "catalog_parquet_scan_interruption_publish_failed",
                catalog_id = log_catalog_id,
                error = %error,
                "catalog Parquet scan cancellation could not be published"
            );
        }
        Err(error) => {
            tracing::error!(
                event = "catalog_parquet_scan_interruption_task_failed",
                catalog_id = log_catalog_id,
                error = %error,
                "catalog Parquet scan cancellation publisher stopped"
            );
        }
    }
}

#[cfg(test)]
async fn catalog_scan_terminal_test_barrier(
    enabled: &std::sync::atomic::AtomicBool,
    reached: &tokio::sync::Notify,
    release: &tokio::sync::Notify,
    cancellation: &CancellationToken,
) {
    if enabled.swap(false, std::sync::atomic::Ordering::SeqCst) {
        reached.notify_waiters();
        tokio::select! {
            _ = release.notified() => {}
            _ = cancellation.cancelled() => {}
        }
    }
}

async fn catalog_scan_blocking<T, F>(
    slots: std::sync::Arc<Semaphore>,
    cancellation: &CancellationToken,
    operation: F,
) -> Result<Option<T>, tokio::task::JoinError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let permit = tokio::select! {
        permit = slots.acquire_owned() => match permit {
            Ok(permit) => permit,
            Err(_) => return Ok(None),
        },
        _ = cancellation.cancelled() => return Ok(None),
    };
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        operation()
    })
    .await
    .map(Some)
}

fn unavailable_catalog_response(attached: AttachedCatalog) -> CatalogResponse {
    let processing = CatalogProcessingProgress {
        message: Some("Catalog files are unavailable or invalid".to_owned()),
        ..CatalogProcessingProgress::default()
    };
    CatalogResponse {
        contract_version: CATALOG_API_CONTRACT_VERSION,
        id: attached.id,
        name: attached.name,
        path: attached.path,
        attached_at: attached.attached_at,
        created_at: None,
        schema_version: None,
        availability: "unavailable",
        source_config: None,
        analyzer_versions: BTreeMap::new(),
        checkpoints: BTreeMap::new(),
        counts: None,
        storage: None,
        processing,
        processing_control: CatalogProcessingControl::default(),
        analyzer_config: CatalogAnalyzerConfig::default(),
    }
}

fn counts_response(
    record_count: u64,
    processing: &CatalogProcessingProgress,
) -> CatalogCountsResponse {
    CatalogCountsResponse {
        record_count,
        candidate_count: processing.candidate_count,
        processed_count: processing.processed_count,
        accepted_count: processing.accepted_count,
        rejected_count: processing.rejected_count,
        error_count: processing.error_count,
    }
}

fn storage_response(storage: CatalogStorageAccounting) -> CatalogStorageResponse {
    CatalogStorageResponse {
        database_bytes: storage.database_bytes,
        manifest_bytes: storage.manifest_bytes,
        artifact_bytes: storage.artifact_bytes,
        total_bytes: storage.total_bytes,
    }
}

fn require_absolute_catalog_path(path: &std::path::Path) -> Result<(), ApiError> {
    if !path.is_absolute() {
        return Err(ApiError::bad_request(
            "Catalog paths must be absolute user-selected locations",
        ));
    }
    Ok(())
}

fn parse_cursor(cursor: Option<&str>) -> Result<Option<i64>, ApiError> {
    cursor
        .map(|cursor| {
            cursor
                .parse::<i64>()
                .ok()
                .filter(|cursor| *cursor >= 0)
                .ok_or_else(|| ApiError::bad_request("Catalog cursor is invalid"))
        })
        .transpose()
}

async fn catalog_call<T, F>(operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, CatalogError> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| ApiError::internal(format!("Catalog task failed: {error}")))?
        .map_err(ApiError::from)
}

const fn default_query_limit() -> u32 {
    DEFAULT_QUERY_LIMIT
}

const fn default_facet_limit() -> u32 {
    DEFAULT_FACET_LIMIT
}

const fn default_catalog_analysis_batch_size() -> u32 {
    16
}

#[cfg(test)]
mod scan_admission_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    fn materialization_record(expected_hash: Option<&str>) -> CatalogRecord {
        CatalogRecord {
            id: "record".to_owned(),
            image_path: "images/record.png".to_owned(),
            thumbnail_path: None,
            embedding_path: None,
            artifact_path: None,
            metadata: serde_json::json!({
                "acquisition": {"contentHash": expected_hash}
            }),
            created_at: "2026-07-27T00:00:00Z".to_owned(),
            updated_at: "2026-07-27T00:00:00Z".to_owned(),
        }
    }

    #[test]
    fn remote_fallback_must_match_the_catalog_acquisition_hash() {
        assert!(
            verify_record_content_hash(&materialization_record(Some("expected")), "changed")
                .is_err()
        );
        assert!(
            verify_record_content_hash(&materialization_record(Some("expected")), "expected")
                .is_ok()
        );
        assert!(verify_record_content_hash(&materialization_record(None), "downloaded").is_ok());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn global_scan_admission_bounds_distinct_catalog_work_and_preserves_db_capacity() {
        let slots = Arc::new(Semaphore::new(2));
        let release = Arc::new(AtomicBool::new(false));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let cancellation = CancellationToken::new();
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let slots = slots.clone();
            let release = release.clone();
            let active = active.clone();
            let peak = peak.clone();
            let cancellation = cancellation.clone();
            tasks.push(tokio::spawn(async move {
                catalog_scan_blocking(slots, &cancellation, move || {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    while !release.load(Ordering::SeqCst) {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    active.fetch_sub(1, Ordering::SeqCst);
                })
                .await
                .expect("blocking task joins")
            }));
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            while active.load(Ordering::SeqCst) != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("two scan slots fill");
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(slots.available_permits(), 0);

        let status_result = tokio::time::timeout(
            Duration::from_millis(100),
            tokio::task::spawn_blocking(|| 42_u64),
        )
        .await
        .expect("unrelated catalog DB work retains blocking-pool capacity")
        .expect("status task joins");
        assert_eq!(status_result, 42);

        release.store(true, Ordering::SeqCst);
        for task in tasks {
            assert!(task.await.expect("admission task joins").is_some());
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
    }
}
