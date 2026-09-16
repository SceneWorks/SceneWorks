use super::*;

use sceneworks_core::film_plan::{
    validate_plan_against_pack, validate_plan_structure, validate_reference_pack, PlanDiagnostic,
    RunRecord,
};
use sceneworks_core::film_workspace::{FilmDraft, FilmRunLocator};

use crate::film_harness::{ControllerLease, HttpTransport, RunControl, RunOptions};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CreateFilmDraftRequest {
    pub title: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FilmRunView {
    pub locator: FilmRunLocator,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record: Option<RunRecord>,
    pub controller_active: bool,
}

pub(crate) async fn list_film_drafts(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<FilmDraft>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.list_film_drafts(&project_id)).await?,
    ))
}

pub(crate) async fn create_film_draft(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<CreateFilmDraftRequest>,
) -> Result<(StatusCode, Json<FilmDraft>), ApiError> {
    let draft_id = format!("film_{}", Uuid::new_v4().simple());
    let draft = project_call(state, move |store| {
        store.create_film_draft(&project_id, &draft_id, &payload.title)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(draft)))
}

pub(crate) async fn get_film_draft(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
) -> Result<Json<FilmDraft>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_film_draft(&project_id, &draft_id)
        })
        .await?,
    ))
}

pub(crate) async fn update_film_draft(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(draft): ApiJson<FilmDraft>,
) -> Result<Json<FilmDraft>, ApiError> {
    validate_planning_selection(&draft)?;
    Ok(Json(
        project_call(state, move |store| {
            store.save_film_draft(&project_id, &draft_id, draft)
        })
        .await?,
    ))
}

fn validate_planning_selection(draft: &FilmDraft) -> Result<(), ApiError> {
    match draft.planning.provider.as_str() {
        "prompt_refiner" => {
            if draft.planning.model_id.is_some() {
                return Err(ApiError::bad_request(
                    "The built-in prompt refiner does not take a separate planner modelId",
                ));
            }
        }
        "native" => {
            if draft.planning.model_id.as_deref()
                != Some(sceneworks_core::film_workspace::QWEN36_FILM_PLANNER_MODEL_ID)
            {
                return Err(ApiError::bad_request(
                    "Native film planning requires modelId film_planner_qwen3_6_27b",
                ));
            }
        }
        "openai_compatible" => {
            if draft
                .planning
                .connection_id
                .as_deref()
                .is_none_or(|id| id.trim().is_empty())
            {
                return Err(ApiError::bad_request(
                    "OpenAI-compatible film planning requires a saved connectionId",
                ));
            }
            if draft
                .planning
                .model_id
                .as_deref()
                .is_none_or(|model| model.trim().is_empty())
            {
                return Err(ApiError::bad_request(
                    "OpenAI-compatible film planning requires a planner modelId",
                ));
            }
        }
        _ => {
            return Err(ApiError::bad_request(
                "planning.provider must be prompt_refiner, native, or openai_compatible",
            ));
        }
    }
    if !matches!(
        draft.planning.thinking_mode.as_str(),
        "disabled" | "enabled" | "auto"
    ) {
        return Err(ApiError::bad_request(
            "planning.thinkingMode must be disabled, enabled, or auto",
        ));
    }
    Ok(())
}

pub(crate) async fn create_film_run(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
) -> Result<(StatusCode, Json<FilmRunView>), ApiError> {
    let draft = project_call(state.clone(), {
        let project_id = project_id.clone();
        let draft_id = draft_id.clone();
        move |store| store.get_film_draft(&project_id, &draft_id)
    })
    .await?;
    let mut findings = validate_plan_structure(&draft.production_plan);
    findings.extend(validate_reference_pack(&draft.reference_pack));
    findings.extend(validate_plan_against_pack(
        &draft.production_plan,
        &draft.reference_pack,
    ));
    if !findings.is_empty() {
        return Err(invalid_film_document(findings));
    }

    let locator_id = format!("filmrun_{}", Uuid::new_v4().simple());
    let locator = project_call(state, move |store| {
        store.create_film_run(&project_id, &locator_id, &draft_id)
    })
    .await?;
    Ok((
        StatusCode::CREATED,
        Json(FilmRunView {
            locator,
            record: None,
            controller_active: false,
        }),
    ))
}

pub(crate) async fn get_film_run(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
) -> Result<Json<FilmRunView>, ApiError> {
    Ok(Json(load_run_view(state, project_id, run_id).await?))
}

pub(crate) async fn start_film_run(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
) -> Result<(StatusCode, Json<FilmRunView>), ApiError> {
    let mut view = load_run_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    if view.record.is_some() && !view.controller_active {
        return Ok((StatusCode::OK, Json(view)));
    }
    let files = project_call(state.clone(), {
        let project_id = project_id.clone();
        let run_id = run_id.clone();
        move |store| store.film_run_files(&project_id, &run_id)
    })
    .await?;
    let lease = ControllerLease::acquire(&files.directory, format!("api:{run_id}"))
        .map_err(|error| ApiError::conflict(error.to_string()))?;
    view.controller_active = true;

    let base_url = state.settings.mcp_api_url.clone();
    let token = state.settings.access_token.clone();
    let project_id_for_run = project_id.clone();
    tokio::spawn(async move {
        let transport = match HttpTransport::new(&base_url, Some(token)) {
            Ok(transport) => transport,
            Err(error) => {
                tracing::error!(project_id, run_id, %error, "film run transport failed");
                return;
            }
        };
        let options = RunOptions {
            plan_path: files.plan,
            reference_pack_path: files.reference_pack,
            compiled_path: None,
            project_id: Some(project_id_for_run),
            shot_ids: None,
            out_dir: files.directory,
            poll_interval: Duration::from_secs(2),
            export: false,
            require_installed: true,
        };
        if let Err(error) = crate::film_harness::run_with_control_and_lease(
            &transport,
            &options,
            &RunControl::watching(&options.out_dir),
            lease,
        )
        .await
        {
            tracing::error!(project_id, run_id, %error, "film run stopped");
        }
    });

    Ok((StatusCode::ACCEPTED, Json(view)))
}

async fn load_run_view(
    state: AppState,
    project_id: String,
    run_id: String,
) -> Result<FilmRunView, ApiError> {
    let (locator, files) = project_call(state, move |store| {
        let locator = store.get_film_run(&project_id, &run_id)?;
        let files = store.film_run_files(&project_id, &run_id)?;
        Ok((locator, files))
    })
    .await?;
    let record = match crate::film_harness::read_run_record(&files.directory) {
        Ok(record) => Some(record),
        Err(crate::film_harness::HarnessError::Refused(_)) => None,
        Err(error) => return Err(ApiError::internal(error.to_string())),
    };
    Ok(FilmRunView {
        locator,
        record,
        controller_active: files
            .directory
            .join(crate::film_harness::CONTROLLER_LOCK_FILE)
            .exists(),
    })
}

fn invalid_film_document(findings: Vec<PlanDiagnostic>) -> ApiError {
    let detail = findings
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    ApiError::bad_request(format!("Film draft is not ready to render: {detail}"))
}
