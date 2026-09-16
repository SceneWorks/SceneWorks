use super::*;

use sceneworks_core::film_compile::production_plan_sha256;
use sceneworks_core::film_plan::{
    self, validate_reference_pack, PlanDiagnostic, ReferencePack, RunRecord,
};
use sceneworks_core::film_workspace::{FilmDraft, FilmRunLocator};

use crate::film_harness::{
    preflight_documents, ControllerLease, FilmDocumentPreflight, HarnessError, HttpTransport,
    RunControl, RunOptions,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CreateFilmDraftRequest {
    pub title: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ReferencePackUpdate {
    pub draft_revision: u32,
    pub reference_pack: ReferencePack,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FilmPreflightRequest {
    #[serde(default)]
    pub selected_shot_ids: Vec<String>,
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

pub(crate) async fn get_reference_pack(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
) -> Result<Json<ReferencePack>, ApiError> {
    let draft = project_call(state, move |store| {
        store.get_film_draft(&project_id, &draft_id)
    })
    .await?;
    Ok(Json(draft.reference_pack))
}

pub(crate) async fn update_reference_pack(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<ReferencePackUpdate>,
) -> Result<Json<FilmDraft>, ApiError> {
    let findings = validate_reference_pack(&payload.reference_pack);
    if !findings.is_empty() {
        return Err(invalid_film_document(findings));
    }
    Ok(Json(
        project_call(state, move |store| {
            let mut draft = store.get_film_draft(&project_id, &draft_id)?;
            if draft.revision != payload.draft_revision {
                return Err(ProjectStoreError::BadRequest(format!(
                    "Film draft revision conflict: expected {}, got {}",
                    draft.revision, payload.draft_revision
                )));
            }
            draft.reference_pack = payload.reference_pack;
            store.save_film_draft(&project_id, &draft_id, draft)
        })
        .await?,
    ))
}

pub(crate) async fn add_film_reference(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<FilmReferenceInput>,
) -> Result<(StatusCode, Json<FilmDraft>), ApiError> {
    let draft = project_call(state, move |store| {
        store.add_film_reference(&project_id, &draft_id, payload)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(draft)))
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
        _ => {
            return Err(ApiError::bad_request(
                "planning.provider must be prompt_refiner or native",
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
    ApiJson(payload): ApiJson<Option<FilmPreflightRequest>>,
) -> Result<(StatusCode, Json<FilmRunView>), ApiError> {
    let payload = payload.unwrap_or_default();
    let draft = project_call(state.clone(), {
        let project_id = project_id.clone();
        let draft_id = draft_id.clone();
        move |store| store.get_film_draft(&project_id, &draft_id)
    })
    .await?;
    let selected = effective_selection(&draft, payload.selected_shot_ids)?;
    let mut findings =
        film_plan::validate_all(&draft.production_plan, &draft.reference_pack, None, None);
    if let Some(compiled) = draft.compiled_plan.as_ref() {
        let sha = production_plan_sha256(&draft.production_plan)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        findings.extend(compiled.staleness_findings(&draft.production_plan, &sha));
    }
    if !findings.is_empty() {
        return Err(invalid_film_document(findings));
    }

    let locator_id = format!("filmrun_{}", Uuid::new_v4().simple());
    let compiled = draft.compiled_plan;
    let locator = project_call(state, move |store| {
        store.create_film_run(&project_id, &locator_id, &draft_id, selected, compiled)
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

pub(crate) async fn preflight_film_draft(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<Option<FilmPreflightRequest>>,
) -> Result<Json<FilmDocumentPreflight>, ApiError> {
    let payload = payload.unwrap_or_default();
    let draft = project_call(state.clone(), move |store| {
        store.get_film_draft(&project_id, &draft_id)
    })
    .await?;
    let selected = effective_selection(&draft, payload.selected_shot_ids)?;
    Ok(Json(run_preflight(&state, &draft, &selected).await?))
}

fn effective_selection(draft: &FilmDraft, requested: Vec<String>) -> Result<Vec<String>, ApiError> {
    if let Some(id) = requested.iter().find(|id| {
        !draft
            .production_plan
            .shots
            .iter()
            .any(|shot| &shot.id == *id)
    }) {
        return Err(invalid_film_document(vec![PlanDiagnostic::plan(
            "selection",
            format!("selected shot {id:?} is not in this plan"),
        )]));
    }
    let selected = draft
        .production_plan
        .shots
        .iter()
        .filter(|shot| requested.is_empty() || requested.contains(&shot.id))
        .map(|shot| shot.id.clone())
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(ApiError::bad_request("Select at least one shot to render"));
    }
    Ok(selected)
}

async fn run_preflight(
    state: &AppState,
    draft: &FilmDraft,
    selected: &[String],
) -> Result<FilmDocumentPreflight, ApiError> {
    let transport = HttpTransport::new(
        &state.settings.mcp_api_url,
        Some(state.settings.access_token.clone()),
    )
    .map_err(|error| ApiError::internal(error.to_string()))?;
    preflight_documents(
        &transport,
        &draft.production_plan,
        &draft.reference_pack,
        draft.compiled_plan.clone(),
        Some(selected),
        true,
    )
    .await
    .map_err(|error| match error {
        HarnessError::Validation(findings) => invalid_film_document(findings),
        other => ApiError::internal(other.to_string()),
    })
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
    let selected_shot_ids = view.locator.selected_shot_ids.clone();
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
            compiled_path: files.compiled.is_file().then_some(files.compiled),
            project_id: Some(project_id_for_run),
            shot_ids: (!selected_shot_ids.is_empty()).then_some(selected_shot_ids),
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
    let store_state = state.clone();
    let saved_project_id = project_id.clone();
    let (locator, files) = project_call(state, move |store| {
        let locator = store.get_film_run(&project_id, &run_id)?;
        let files = store.film_run_files(&project_id, &run_id)?;
        Ok((locator, files))
    })
    .await?;
    let mut record = match crate::film_harness::read_run_record(&files.directory) {
        Ok(record) => Some(record),
        Err(crate::film_harness::HarnessError::Refused(_)) => None,
        Err(error) => return Err(ApiError::internal(error.to_string())),
    };
    if let Some(record) = record.as_mut() {
        if let Some(timeline_id) = record.timeline.as_ref().map(|t| t.timeline_id.clone()) {
            let saved = project_call(store_state, move |store| {
                store.get_timeline(&saved_project_id, &timeline_id)
            })
            .await?;
            crate::film_harness::reconcile_saved_cut(record, &saved);
        }
    }
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
