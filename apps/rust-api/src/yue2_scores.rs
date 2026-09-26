//! HTTP surface for YuE2 score inspection, bounded edits, render records and listening
//! comparisons (sc-22997, epic 22988).
//!
//! Thin handlers over `sceneworks_core::yue2_score`: every edit goes through the core's bounded
//! operation set, dialect re-parse and invariant check, and lands as a NEW immutable version.
//! The MCP tools (`crates/sceneworks-mcp`) call these same routes, so a person (the UI) and an
//! agent share one behaviour surface.
//!
//! Error mapping: notation outside the supported dialect → 422 `yue2_unsupported_notation`; an
//! edit that changes something its contract fixes → 422 `yue2_invariant_violation` with the full
//! invariant report as `context`; malformed input → 400; unknown record → 404; a render whose
//! score hash differs from its version → 409.

use super::*;

use sceneworks_core::yue2_score::store::{
    ComparisonInput, ComparisonRecord, CreateVersionInput, EditInput, Listing, Provenance,
    RenderInput, RenderRecord, ScoreVersionRecord, VersionDetail, VersionSummary,
};
use sceneworks_core::yue2_score::{
    inspect, parse_bounded, ScoreEditOperation, ScoreInspection, Yue2ScoreError,
    REGENERATION_NOTICE,
};

pub(crate) const YUE2_UNSUPPORTED_NOTATION: &str = "yue2_unsupported_notation";
pub(crate) const YUE2_INVARIANT_VIOLATION: &str = "yue2_invariant_violation";

fn yue2_error(error: Yue2ScoreError) -> ApiError {
    match error {
        Yue2ScoreError::Notation(abc) => ApiError::typed(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "Unsupported YuE2 ABC notation: {abc}. The score is outside the supported native \
                 dialect (not necessarily invalid under the full ABC standard); rewrite it in the \
                 native dialect."
            ),
            YUE2_UNSUPPORTED_NOTATION,
            json!({ "message": abc.message }),
        ),
        Yue2ScoreError::Invariant(report) => match serde_json::to_value(&*report) {
            Ok(context) => ApiError::typed(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "The edit changes what its contract fixes: {}",
                    report.violations.join("; ")
                ),
                YUE2_INVARIANT_VIOLATION,
                context,
            ),
            Err(error) => {
                ApiError::internal(format!("invariant report could not be serialized: {error}"))
            }
        },
        Yue2ScoreError::BadRequest(detail) => ApiError::bad_request(detail),
        Yue2ScoreError::NotFound(detail) => ApiError {
            status: StatusCode::NOT_FOUND,
            detail,
            code: None,
            context: None,
        },
        Yue2ScoreError::Conflict(detail) => ApiError::conflict(detail),
        Yue2ScoreError::Store(error) => error.into(),
    }
}

async fn yue2_call<T, F>(state: AppState, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(Arc<ProjectStore>) -> Result<T, Yue2ScoreError> + Send + 'static,
{
    let store = state.project_store.clone();
    tokio::task::spawn_blocking(move || operation(store))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(yue2_error)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Yue2InspectRequest {
    abc: String,
}

/// Stateless inspection of any ABC text: exact events or a clear dialect rejection.
pub(crate) async fn inspect_yue2_score(
    ApiJson(payload): ApiJson<Yue2InspectRequest>,
) -> Result<Json<ScoreInspection>, ApiError> {
    let score = parse_bounded(&payload.abc).map_err(yue2_error)?;
    Ok(Json(inspect(&score)))
}

pub(crate) async fn list_yue2_score_versions(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Listing<VersionSummary>>, ApiError> {
    Ok(Json(
        yue2_call(state, move |store| {
            store.yue2_score_store(&project_id)?.list_versions()
        })
        .await?,
    ))
}

pub(crate) async fn create_yue2_score_version(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<CreateVersionInput>,
) -> Result<(StatusCode, Json<ScoreVersionRecord>), ApiError> {
    let record = yue2_call(state, move |store| {
        store.yue2_score_store(&project_id)?.create_version(payload)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(record)))
}

pub(crate) async fn get_yue2_score_version(
    State(state): State<AppState>,
    Path((project_id, version_id)): Path<(String, String)>,
) -> Result<Json<VersionDetail>, ApiError> {
    Ok(Json(
        yue2_call(state, move |store| {
            store
                .yue2_score_store(&project_id)?
                .get_version(&version_id)
        })
        .await?,
    ))
}

pub(crate) async fn inspect_yue2_score_version(
    State(state): State<AppState>,
    Path((project_id, version_id)): Path<(String, String)>,
) -> Result<Json<ScoreInspection>, ApiError> {
    Ok(Json(
        yue2_call(state, move |store| {
            let (_, score) = store
                .yue2_score_store(&project_id)?
                .version_score(&version_id)?;
            Ok(inspect(&score))
        })
        .await?,
    ))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Yue2EditRequest {
    operation: ScoreEditOperation,
    brief: String,
    provenance: Provenance,
    /// Check the edit and return the would-be version without storing it.
    #[serde(default)]
    dry_run: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Yue2EditResponse {
    dry_run: bool,
    version: ScoreVersionRecord,
    render_notice: &'static str,
}

pub(crate) async fn edit_yue2_score_version(
    State(state): State<AppState>,
    Path((project_id, version_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<Yue2EditRequest>,
) -> Result<(StatusCode, Json<Yue2EditResponse>), ApiError> {
    let dry_run = payload.dry_run;
    let version = yue2_call(state, move |store| {
        store.yue2_score_store(&project_id)?.apply_edit(
            &version_id,
            EditInput {
                operation: payload.operation,
                brief: payload.brief,
                provenance: payload.provenance,
            },
            dry_run,
        )
    })
    .await?;
    let status = if dry_run {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    };
    Ok((
        status,
        Json(Yue2EditResponse {
            dry_run,
            version,
            render_notice: REGENERATION_NOTICE,
        }),
    ))
}

pub(crate) async fn list_yue2_score_renders(
    State(state): State<AppState>,
    Path((project_id, version_id)): Path<(String, String)>,
) -> Result<Json<Vec<RenderRecord>>, ApiError> {
    Ok(Json(
        yue2_call(state, move |store| {
            Ok(store
                .yue2_score_store(&project_id)?
                .get_version(&version_id)?
                .renders)
        })
        .await?,
    ))
}

/// Record a finished (or failed) rendering of a version — the render job's write path
/// (sc-22999). Verifies the audio asset and the score hash the job rendered.
pub(crate) async fn record_yue2_score_render(
    State(state): State<AppState>,
    Path((project_id, version_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<RenderInput>,
) -> Result<(StatusCode, Json<RenderRecord>), ApiError> {
    let record = yue2_call(state, move |store| {
        store.record_yue2_render(&project_id, &version_id, payload)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(record)))
}

pub(crate) async fn list_yue2_comparisons(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Listing<ComparisonRecord>>, ApiError> {
    Ok(Json(
        yue2_call(state, move |store| {
            store.yue2_score_store(&project_id)?.list_comparisons()
        })
        .await?,
    ))
}

pub(crate) async fn create_yue2_comparison(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<ComparisonInput>,
) -> Result<(StatusCode, Json<ComparisonRecord>), ApiError> {
    let record = yue2_call(state, move |store| {
        store
            .yue2_score_store(&project_id)?
            .create_comparison(payload)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(record)))
}

pub(crate) async fn get_yue2_comparison(
    State(state): State<AppState>,
    Path((project_id, comparison_id)): Path<(String, String)>,
) -> Result<Json<ComparisonRecord>, ApiError> {
    Ok(Json(
        yue2_call(state, move |store| {
            store
                .yue2_score_store(&project_id)?
                .get_comparison(&comparison_id)
        })
        .await?,
    ))
}
