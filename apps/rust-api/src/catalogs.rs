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
    CatalogProcessingProgress, CatalogProcessingState, CatalogRecord, CatalogRecordFilter,
    CatalogRegistry, CatalogSourceConfig, CatalogStorageAccounting,
};
use sceneworks_worker::catalog_parquet_scanner::{
    validate_catalog_parquet_scan_plan_with_cancel, AttachedCatalogParquetScanDriver,
    CatalogParquetScanError, CatalogParquetScanOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::catalog_scan_supervisor::CatalogScanSpawn;
use crate::{ApiError, ApiJson, AppState};

const CATALOG_API_CONTRACT_VERSION: u32 = 1;
const DEFAULT_QUERY_LIMIT: u32 = 50;
const MAX_QUERY_LIMIT: u32 = 200;
const DEFAULT_FACET_LIMIT: u32 = 50;
const MAX_FACET_LIMIT: u32 = 100;
const SCHEDULED_SCAN_PASS_ROWS: u64 = 25_000;
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
    match CatalogProcessingLease::try_acquire(catalog) {
        Ok(lease) => {
            lease.reconcile_interrupted(catalog)?;
        }
        Err(CatalogError::Conflict(_)) => {}
        Err(error) => return Err(error),
    }
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
    let log_catalog_id = catalog_id.clone();
    let spawned = state
        .catalog_scan_supervisor
        .spawn(catalog_id.clone(), move |cancellation| async move {
            #[cfg(test)]
            let start_barrier = { before_start.lock().take() };
            #[cfg(test)]
            if let Some(barrier) = start_barrier {
                tokio::select! {
                    _ = barrier.wait() => {}
                    _ = cancellation.cancelled() => return,
                }
            }
            let mut retry_delay = std::time::Duration::from_millis(10);
            let mut driver = loop {
                if cancellation.is_cancelled() {
                    return;
                }
                let pass_config_dir = config_dir.clone();
                let pass_catalog_id = catalog_id.clone();
                let start = tokio::task::spawn_blocking(move || {
                    AttachedCatalogParquetScanDriver::try_start(
                        &CatalogRegistry::new(pass_config_dir),
                        &pass_catalog_id,
                    )
                })
                .await;
                match start {
                    Ok(Ok(driver)) => break driver,
                    Ok(Err(CatalogParquetScanError::Busy(_))) => {
                        let retry_config_dir = config_dir.clone();
                        let retry_catalog_id = catalog_id.clone();
                        let should_retry = tokio::task::spawn_blocking(move || {
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
                            Ok(Ok(true)) => {}
                            Ok(Ok(false)) | Ok(Err(CatalogError::NotFound(_))) => return,
                            Ok(Err(error)) => {
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
                            _ = cancellation.cancelled() => return,
                        }
                        retry_delay = retry_delay
                            .saturating_mul(2)
                            .min(CATALOG_SCAN_START_MAX_BACKOFF);
                    }
                    Ok(Err(error)) => {
                        tracing::error!(
                            event = "catalog_parquet_scan_start_failed",
                            catalog_id,
                            error = %error,
                            "catalog Parquet scanner could not start"
                        );
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
                let result = tokio::task::spawn_blocking(move || {
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
                    Ok((next_driver, Ok(report))) => {
                        driver = next_driver;
                        if report.paused || report.checkpoint.complete {
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
                    Ok((_next_driver, Err(CatalogParquetScanError::Interrupted(error)))) => {
                        tracing::info!(
                            event = "catalog_parquet_scan_interrupted",
                            catalog_id,
                            error,
                            "catalog Parquet scanner stopped for server shutdown"
                        );
                        break;
                    }
                    Ok((_next_driver, Err(error))) => {
                        tracing::error!(
                            event = "catalog_parquet_scan_failed",
                            catalog_id,
                            error = %error,
                            "catalog Parquet scanner stopped"
                        );
                        break;
                    }
                }
            }
        })
        .await;
    match spawned {
        CatalogScanSpawn::Started => {}
        CatalogScanSpawn::AlreadyRunning => {
            tracing::debug!(
                event = "catalog_parquet_scan_deduplicated",
                catalog_id = log_catalog_id,
                "catalog Parquet scan already has a supervised task"
            );
        }
        CatalogScanSpawn::ShuttingDown => {
            tracing::warn!(
                event = "catalog_parquet_scan_rejected_during_shutdown",
                catalog_id = log_catalog_id,
                "catalog Parquet scan was not started because the server is shutting down"
            );
        }
    }
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
