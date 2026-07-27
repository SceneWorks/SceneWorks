//! Dataset Catalog lifecycle and bounded query HTTP contracts.
//!
//! Catalog-id operations resolve exclusively through [`CatalogRegistry`], whose
//! state lives under `settings.config_dir`. Only create/attach accept a
//! user-selected absolute path; no request can substitute a root for an attached
//! catalog id.

use std::collections::BTreeMap;
use std::path::PathBuf;

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
use sceneworks_worker::catalog_parquet_scanner::{
    validate_catalog_parquet_scan_plan_with_cancel, AttachedCatalogParquetScanDriver,
    CatalogParquetScanError, CatalogParquetScanOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::catalog_scan_supervisor::CatalogScanSpawn;
use crate::{ApiError, ApiJson, AppState};

const CATALOG_API_CONTRACT_VERSION: u32 = 1;
const DEFAULT_QUERY_LIMIT: u32 = 50;
const MAX_QUERY_LIMIT: u32 = 200;
const DEFAULT_FACET_LIMIT: u32 = 50;
const MAX_FACET_LIMIT: u32 = 100;
const SCHEDULED_SCAN_PASS_ROWS: u64 = 25_000;
const PARQUET_SCAN_CHECKPOINT_KEY: &str = "scanner.parquet.checkpoint.v1";
const CATALOG_SCAN_START_MAX_BACKOFF: std::time::Duration = std::time::Duration::from_millis(250);
const CATALOG_PREFLIGHT_ADMISSION_WAIT: std::time::Duration = std::time::Duration::from_millis(250);
const CATALOG_PREFLIGHT_EXECUTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

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
    let requested_gpu = crate::requested_gpu_or_auto(request.requested_gpu);
    let batch_size = request.batch_size;
    let payload = serde_json::json!({
        "provider": "catalog",
        "kind": "catalog_analysis",
        "catalogId": catalog_id,
        "catalogName": catalog_name,
        "analyzerConfigRevision": analyzer_config.revision,
        "batchSize": batch_size,
    })
    .as_object()
    .cloned()
    .expect("catalog analysis payload object");
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
                let result = catalog_scan_blocking(work_slots.clone(), &cancellation, move || {
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
                            break;
                        }
                        #[cfg(test)]
                        if stop_after_pass.swap(false, std::sync::atomic::Ordering::SeqCst) {
                            break;
                        }
                        if cancellation.is_cancelled() {
                            driver.publish_interrupted();
                            break;
                        }
                    }
                    Ok(Some((_next_driver, Err(CatalogParquetScanError::Interrupted(error))))) => {
                        tracing::info!(
                            event = "catalog_parquet_scan_interrupted",
                            catalog_id,
                            error,
                            "catalog Parquet scanner stopped for server shutdown"
                        );
                        break;
                    }
                    Ok(Some((_next_driver, Err(error)))) => {
                        tracing::error!(
                            event = "catalog_parquet_scan_failed",
                            catalog_id,
                            error = %error,
                            "catalog Parquet scanner stopped"
                        );
                        break;
                    }
                    Ok(None) => {
                        reconcile_cancelled_catalog_scan(config_dir.clone(), catalog_id.clone())
                            .await;
                        break;
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
                CatalogProcessingState::Completed | CatalogProcessingState::Failed
            )
        {
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
