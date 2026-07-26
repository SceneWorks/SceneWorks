//! Dataset Catalog lifecycle and bounded query HTTP contracts.
//!
//! Catalog-id operations resolve exclusively through [`CatalogRegistry`], whose
//! state lives under `settings.config_dir`. Only create/attach accept a
//! user-selected absolute path; no request can substitute a root for an attached
//! catalog id.

use std::collections::BTreeMap;
use std::path::PathBuf;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use sceneworks_core::catalog_store::{
    AttachedCatalog, Catalog, CatalogContractState, CatalogError, CatalogFacet,
    CatalogProcessingControl, CatalogProcessingDesiredState, CatalogProcessingLease,
    CatalogProcessingProgress, CatalogRecord, CatalogRecordFilter, CatalogRegistry,
    CatalogSourceConfig, CatalogStorageAccounting,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ApiError, ApiJson, AppState};

const CATALOG_API_CONTRACT_VERSION: u32 = 1;
const DEFAULT_QUERY_LIMIT: u32 = 50;
const MAX_QUERY_LIMIT: u32 = 200;
const DEFAULT_FACET_LIMIT: u32 = 50;
const MAX_FACET_LIMIT: u32 = 100;

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
pub(crate) struct CatalogControlRequest {
    expected_revision: u64,
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
    let config_dir = state.settings.config_dir.clone();
    let response = catalog_call(move || {
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
        catalog_response(&attached, &catalog)
    })
    .await?;
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
    Ok(Json(load_catalog_response(state, catalog_id).await?))
}

pub(crate) async fn get_catalog_status(
    State(state): State<AppState>,
    Path(catalog_id): Path<String>,
) -> Result<Json<CatalogResponse>, ApiError> {
    Ok(Json(load_catalog_response(state, catalog_id).await?))
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
    let response = catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let attached = registry.get(&catalog_id)?;
        let catalog = registry.open_attached(&catalog_id)?;
        let active = CatalogProcessingLease::is_active(&catalog)?;
        if active != require_active_processor {
            return Err(CatalogError::Conflict(if require_active_processor {
                "Catalog has no active processor to pause".to_owned()
            } else {
                "Catalog processor has not stopped yet".to_owned()
            }));
        }
        catalog.request_processing_control(expected_revision, desired_state)?;
        catalog_response(&attached, &catalog)
    })
    .await?;
    Ok(Json(response))
}

async fn load_catalog_response(
    state: AppState,
    catalog_id: String,
) -> Result<CatalogResponse, ApiError> {
    let config_dir = state.settings.config_dir.clone();
    catalog_call(move || {
        let registry = CatalogRegistry::new(config_dir);
        let attached = registry.get(&catalog_id)?;
        let catalog = registry.open_attached(&catalog_id)?;
        catalog_response(&attached, &catalog)
    })
    .await
}

fn catalog_response(
    attached: &AttachedCatalog,
    catalog: &Catalog,
) -> Result<CatalogResponse, CatalogError> {
    let contract_state = catalog.contract_state()?;
    let storage = catalog.storage_accounting()?;
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
        counts: Some(counts_response(
            storage.record_count,
            &contract_state.processing,
        )),
        storage: Some(storage_response(storage)),
        processing: contract_state.processing,
        processing_control: catalog.processing_control()?,
    })
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
