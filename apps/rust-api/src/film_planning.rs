use super::*;

use sceneworks_core::film_compile::{CompiledPlan, PlannerExecutionRecord};
use sceneworks_core::film_plan::{PlanDiagnostic, ProductionPlan};
use sceneworks_core::film_planner::{DurationWindow, ProductionBrief, RequiredBeat};
use sceneworks_core::film_workspace::{
    parse_film_script, FilmBriefDocument, FilmRenderRegime, QWEN36_FILM_PLANNER_MODEL_ID,
    QWEN36_FILM_PLANNER_REPO,
};

use crate::film_harness::{ControllerLease, HarnessError, HttpTransport};
use crate::film_planner::{PlannerOptions, SceneWorksLlm, DEFAULT_LLM_JOB_TIMEOUT};
use crate::film_planner_connections::{find_connection, resolve_connection_credential};
use crate::openai_planner::{OpenAiPlannerLlm, OpenAiPlannerOptions};

const OPERATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ParseFilmScriptRequest {
    #[serde(default)]
    script: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct StartFilmPlanningRequest {
    #[serde(default)]
    max_repair_rounds: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ApplyFilmPlanningRequest {
    operation_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FilmPlanningOperation {
    schema_version: u32,
    id: String,
    project_id: String,
    draft_id: String,
    draft_revision: u32,
    status: String,
    stage: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    progress: Option<f64>,
    provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    planner_model_id: Option<String>,
    planner_model: String,
    video_model_id: String,
    thinking_mode: String,
    #[serde(default = "default_max_repair_rounds")]
    max_repair_rounds: u32,
    #[serde(default)]
    refine_prompts: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active_job_id: Option<String>,
    #[serde(default)]
    job_ids: Vec<String>,
    #[serde(default)]
    findings: Vec<PlanDiagnostic>,
    #[serde(default)]
    executions: Vec<PlannerExecutionRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_plan: Option<ProductionPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compiled: Option<CompiledPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FilmPlannerAvailability {
    providers: Vec<FilmPlannerProviderAvailability>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FilmPlannerProviderAvailability {
    provider: &'static str,
    label: &'static str,
    model_id: &'static str,
    available: bool,
    install_state: String,
    install_path: Option<String>,
    auto_download: bool,
    detail: String,
}

pub(crate) async fn parse_film_draft_script(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<ParseFilmScriptRequest>,
) -> Result<Json<FilmBriefDocument>, ApiError> {
    let draft = project_call(state, move |store| {
        store.get_film_draft(&project_id, &draft_id)
    })
    .await?;
    let script = if payload.script.trim().is_empty() {
        draft.original_script.as_str()
    } else {
        payload.script.as_str()
    };
    if script.trim().is_empty() {
        return Err(ApiError::bad_request(
            "Paste prose or screenplay text before extracting beats and dialogue",
        ));
    }
    Ok(Json(parse_film_script(script)))
}

pub(crate) async fn film_planner_availability(
    State(state): State<AppState>,
    Path((_project_id, _draft_id)): Path<(String, String)>,
) -> Result<Json<FilmPlannerAvailability>, ApiError> {
    let catalog = crate::models::model_catalog(&state).await?;
    let row = |id: &str| catalog.iter().find(|entry| entry["id"] == id);
    let provider = |provider, label, model_id, detail: &str| {
        let entry = row(model_id);
        let install_state = entry
            .and_then(|entry| entry.get("installState"))
            .and_then(Value::as_str)
            .unwrap_or("missing")
            .to_owned();
        FilmPlannerProviderAvailability {
            provider,
            label,
            model_id,
            available: install_state == "installed",
            install_path: entry
                .and_then(|entry| entry.get("installedPath"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            auto_download: false,
            install_state,
            detail: detail.to_owned(),
        }
    };
    Ok(Json(FilmPlannerAvailability {
        providers: vec![
            provider(
                "prompt_refiner",
                "Built-in prompt refiner",
                "prompt_refine_anubis_8b",
                "Default local planner. It does not require Qwen3.6-27B.",
            ),
            provider(
                "native",
                "Qwen3.6-27B",
                QWEN36_FILM_PLANNER_MODEL_ID,
                "Optional large native planner. Select and install it explicitly before use.",
            ),
            FilmPlannerProviderAvailability {
                provider: "openai_compatible",
                label: "Saved OpenAI-compatible connection",
                model_id: "",
                available: true,
                install_path: None,
                auto_download: false,
                install_state: "configured_separately".to_owned(),
                detail: "Uses Chat Completions through a saved backend connection. The video model remains separate.".to_owned(),
            },
        ],
    }))
}

pub(crate) async fn get_film_planning_operation(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
) -> Result<Json<FilmPlanningOperation>, ApiError> {
    let root = planning_root(state, &project_id, &draft_id).await?;
    Ok(Json(read_operation(&root.join("latest.json"))?))
}

pub(crate) async fn start_film_planning(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<StartFilmPlanningRequest>,
) -> Result<(StatusCode, Json<FilmPlanningOperation>), ApiError> {
    let mut draft = project_call(state.clone(), {
        let project_id = project_id.clone();
        let draft_id = draft_id.clone();
        move |store| store.get_film_draft(&project_id, &draft_id)
    })
    .await?;
    if draft.original_script.trim().is_empty() {
        return Err(ApiError::bad_request(
            "Paste prose or screenplay text before planning",
        ));
    }
    let render_options = crate::films::apply_selected_render_regime(&state, &mut draft).await?;
    if draft.render_regime == Some(FilmRenderRegime::RecommendedTurbo)
        && !render_options.recommended_turbo.available
    {
        return Err(ApiError::bad_request(format!(
            "The recommended Turbo render regime is unavailable ({}); choose Full quality or install a compatible adapter",
            crate::films::turbo_unavailable_code(
                render_options.recommended_turbo.unavailable_reason
            )
        )));
    }
    let root = planning_root(state.clone(), &project_id, &draft_id).await?;
    std::fs::create_dir_all(root.join("operations"))
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let lease = ControllerLease::acquire(&root, format!("api:planning:{draft_id}"))
        .map_err(|error| ApiError::conflict(error.to_string()))?;
    let _ = std::fs::remove_file(root.join(crate::film_harness::CANCEL_SENTINEL_FILE));

    let structured = if draft.structured_brief.beats.is_empty() {
        parse_film_script(&draft.original_script)
    } else {
        draft.structured_brief.clone()
    };
    if structured.beats.is_empty() {
        return Err(ApiError::bad_request(
            "The script produced no editable beats; add a beat before planning",
        ));
    }
    let external_connection = if draft.planning.provider == "openai_compatible" {
        Some(find_connection(
            &state,
            draft.planning.connection_id.as_deref().ok_or_else(|| {
                ApiError::bad_request("Choose a saved OpenAI-compatible connection")
            })?,
        )?)
    } else {
        None
    };
    let (planner_model_id, planner_model) = if external_connection.is_some() {
        let model = draft
            .planning
            .model_id
            .as_deref()
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| ApiError::bad_request("Choose or enter an external planner model ID"))?;
        (Some(model.to_owned()), model.to_owned())
    } else {
        let (id, model) = planner_identity(&draft.planning.provider)?;
        (id.map(str::to_owned), model.to_owned())
    };
    if draft.planning.provider == "native" {
        let selected = draft.planning.model_id.as_deref().unwrap_or_default();
        if selected != QWEN36_FILM_PLANNER_MODEL_ID {
            return Err(ApiError::bad_request(
                "The native film planner currently supports only film_planner_qwen3_6_27b",
            ));
        }
    }

    let operation_id = format!("filmplan_{}", Uuid::new_v4().simple());
    let operation_dir = root.join("operations").join(&operation_id);
    std::fs::create_dir_all(&operation_dir)
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let now = sceneworks_core::time::utc_now();
    let mut operation = FilmPlanningOperation {
        schema_version: OPERATION_SCHEMA_VERSION,
        id: operation_id.clone(),
        project_id: project_id.clone(),
        draft_id: draft_id.clone(),
        draft_revision: draft.revision,
        status: "running".to_owned(),
        stage: "preflight".to_owned(),
        progress: None,
        provider: draft.planning.provider.clone(),
        planner_model_id,
        planner_model: planner_model.clone(),
        video_model_id: draft.production_plan.model.id.clone(),
        thinking_mode: draft.planning.thinking_mode.clone(),
        max_repair_rounds: payload.max_repair_rounds.unwrap_or(2),
        refine_prompts: draft.planning.refine_prompts,
        active_job_id: None,
        job_ids: Vec::new(),
        findings: Vec::new(),
        executions: Vec::new(),
        candidate_plan: None,
        compiled: None,
        detail: Some(if external_connection.is_some() {
            "Checking the selected external connection and target video model.".to_owned()
        } else {
            "Checking the selected local planner and video model.".to_owned()
        }),
        created_at: now.clone(),
        updated_at: now,
    };

    if draft.planning.provider == "native" {
        let catalog = crate::models::model_catalog(&state).await?;
        let installed = catalog.iter().any(|entry| {
            entry["id"] == QWEN36_FILM_PLANNER_MODEL_ID && entry["installState"] == "installed"
        });
        if !installed {
            operation.status = "failed".to_owned();
            operation.stage = "unavailable".to_owned();
            operation.findings.push(PlanDiagnostic::plan(
                "planning.model",
                "Qwen3.6-27B is not installed. Install the optional Film Planner model explicitly, or select the built-in prompt refiner.",
            ));
            operation.detail =
                Some("The draft and its current shot plan were not changed.".to_owned());
            write_latest_operation(&root, &operation)?;
            drop(lease);
            return Ok((StatusCode::ACCEPTED, Json(operation)));
        }
    }

    let external_credential = if let Some(connection) = external_connection.as_ref() {
        let credential = resolve_connection_credential(&state, connection).await?;
        if connection.credential_host.is_some() && credential.is_none() {
            operation.status = "failed".to_owned();
            operation.stage = "authentication".to_owned();
            operation.findings.push(PlanDiagnostic::plan(
                "planning.connection",
                "The selected connection names a credential, but that credential is missing from the host secret facility.",
            ));
            operation.detail = Some(
                "The draft and current shot plan were preserved; no provider request was sent."
                    .to_owned(),
            );
            write_latest_operation(&root, &operation)?;
            drop(lease);
            return Ok((StatusCode::ACCEPTED, Json(operation)));
        }
        credential
    } else {
        None
    };

    let stage_result = project_call(state.clone(), {
        let project_id = project_id.clone();
        let draft_id = draft_id.clone();
        let operation_id = operation_id.clone();
        let draft_revision = draft.revision;
        move |store| {
            store.stage_film_planning_references(
                &project_id,
                &draft_id,
                draft_revision,
                &operation_id,
            )
        }
    })
    .await;
    if let Err(error) = stage_result {
        let _ = std::fs::remove_dir_all(&operation_dir);
        return Err(error);
    }
    let brief = production_brief(&draft, &structured);
    write_json_file(&operation_dir.join("brief.json"), &brief)?;
    write_json_file(
        &operation_dir.join("references.json"),
        &draft.reference_pack,
    )?;
    write_latest_operation(&root, &operation)?;

    let base_url = state.settings.mcp_api_url.clone();
    let token = state.settings.access_token.clone();
    let latest_path = root.join("latest.json");
    let callback_path = latest_path.clone();
    let callback_operation_id = operation_id.clone();
    let on_job_created = std::sync::Arc::new(move |job_id: &str| {
        let _ = update_operation(&callback_path, &callback_operation_id, |operation| {
            if operation.status != "running" {
                return;
            }
            operation.stage = "generating".to_owned();
            operation.detail =
                Some("The selected local planner is generating a candidate plan.".to_owned());
            operation.active_job_id = Some(job_id.to_owned());
            operation.job_ids.push(job_id.to_owned());
        });
    });
    let progress_path = latest_path.clone();
    let progress_operation_id = operation_id.clone();
    let on_job_progress = std::sync::Arc::new(move |job_id: &str, progress: f64| {
        let _ = update_operation(&progress_path, &progress_operation_id, |operation| {
            if operation.status == "running" && operation.active_job_id.as_deref() == Some(job_id) {
                operation.progress = Some(progress);
            }
        });
    });
    let cancel_path = root.join(crate::film_harness::CANCEL_SENTINEL_FILE);
    let cancel_requested = std::sync::Arc::new(move || cancel_path.exists());
    let external_started_path = latest_path.clone();
    let external_started_operation_id = operation_id.clone();
    let on_external_request_started = std::sync::Arc::new(move || {
        mark_external_planner_waiting(&external_started_path, &external_started_operation_id);
    });
    let options = PlannerOptions {
        brief_path: operation_dir.join("brief.json"),
        reference_pack_path: operation_dir.join("references.json"),
        out_dir: operation_dir,
        max_repair_rounds: operation.max_repair_rounds,
        refine_prompts: operation.refine_prompts,
        prompt_guide_path: None,
        require_installed: true,
        require_local_planner: external_connection.is_none(),
        send_reference_pixels: external_connection.is_some()
            && draft.planning.send_reference_pixels,
        api_url: base_url.clone(),
        force: false,
        poll_interval: Duration::from_millis(350),
        job_timeout: DEFAULT_LLM_JOB_TIMEOUT,
    };
    let model_override = (draft.planning.provider == "native").then(|| planner_model.clone());
    let thinking_mode = draft.planning.thinking_mode.clone();
    let source_script = draft.original_script.clone();
    let external_model = planner_model;
    let external_send_reference_pixels = draft.planning.send_reference_pixels;
    let external_client = state.http_client.clone();
    tokio::spawn(async move {
        let result = async {
            let transport = HttpTransport::new(&base_url, Some(token))?;
            if let Some(connection) = external_connection {
                let llm = OpenAiPlannerLlm::new(
                    external_client,
                    connection,
                    external_credential,
                    OpenAiPlannerOptions {
                        model: external_model,
                        thinking_mode,
                        source_script,
                        send_reference_pixels: external_send_reference_pixels,
                    },
                    cancel_requested,
                )?
                .on_request_started(on_external_request_started);
                crate::film_planner::generate(&transport, &llm, &options).await
            } else {
                let llm =
                    SceneWorksLlm::new(&transport, options.poll_interval, options.job_timeout)
                        .with_planner_model(model_override, thinking_mode)
                        .on_job_created(on_job_created)
                        .on_job_progress(on_job_progress)
                        .cancel_requested(cancel_requested);
                crate::film_planner::generate(&transport, &llm, &options).await
            }
        }
        .await;
        finish_planning_operation(&root, &latest_path, &operation_id, result);
        drop(lease);
    });

    Ok((StatusCode::ACCEPTED, Json(operation)))
}

fn mark_external_planner_waiting(latest_path: &FsPath, operation_id: &str) {
    let _ = update_operation(latest_path, operation_id, |operation| {
        if operation.status != "running" {
            return;
        }
        operation.stage = "planning".to_owned();
        operation.progress = None;
        operation.detail = Some(
            "Waiting for the selected external planner to return a candidate plan.".to_owned(),
        );
    });
}

pub(crate) async fn cancel_film_planning(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
) -> Result<Json<FilmPlanningOperation>, ApiError> {
    let root = planning_root(state.clone(), &project_id, &draft_id).await?;
    let path = root.join("latest.json");
    let mut operation = read_operation(&path)?;
    if !matches!(operation.status.as_str(), "running" | "canceling") {
        return Ok(Json(operation));
    }
    std::fs::write(
        root.join(crate::film_harness::CANCEL_SENTINEL_FILE),
        b"cancel\n",
    )
    .map_err(|error| ApiError::internal(error.to_string()))?;
    if let Some(job_id) = operation.active_job_id.clone() {
        let _ = crate::jobs::cancel_job(State(state), Path(job_id)).await;
    }
    operation.status = "canceling".to_owned();
    operation.stage = "canceling".to_owned();
    operation.detail =
        Some("Cancellation requested; waiting for the planning request to stop.".to_owned());
    operation.updated_at = sceneworks_core::time::utc_now();
    write_latest_operation(&root, &operation)?;
    Ok(Json(operation))
}

pub(crate) async fn apply_film_planning_candidate(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<ApplyFilmPlanningRequest>,
) -> Result<Json<sceneworks_core::film_workspace::FilmDraft>, ApiError> {
    let root = planning_root(state.clone(), &project_id, &draft_id).await?;
    let operation = read_operation(&root.join("latest.json"))?;
    if operation.id != payload.operation_id || operation.status != "ready" {
        return Err(ApiError::conflict(
            "That planning candidate is no longer ready to apply",
        ));
    }
    let mut candidate = operation
        .candidate_plan
        .ok_or_else(|| ApiError::internal("Ready planning operation has no candidate plan"))?;
    let mut compiled = operation
        .compiled
        .ok_or_else(|| ApiError::internal("Ready planning operation has no compiled plan"))?;
    let saved = project_call(state, move |store| {
        let mut draft = store.get_film_draft(&project_id, &draft_id)?;
        if draft.revision != operation.draft_revision {
            return Err(sceneworks_core::project_store::ProjectStoreError::BadRequest(format!(
                "Film draft changed after planning started (candidate revision {}, current revision {}); regenerate before replacing the edited plan",
                operation.draft_revision, draft.revision
            )));
        }
        // The store owns the next revision. Make the candidate and its compile current before the
        // normal save so it can preserve the refined prompts across that revision bump.
        candidate.id = draft.id.clone();
        candidate.title = draft.title.clone();
        candidate.version = draft.revision;
        compiled.plan_id = candidate.id.clone();
        compiled.plan_version = candidate.version;
        compiled.plan_sha256 = sceneworks_core::film_compile::production_plan_sha256(&candidate)?;
        draft.production_plan = candidate;
        draft.compiled_plan = Some(compiled);
        store.save_film_draft(&project_id, &draft_id, draft)
    })
    .await?;
    Ok(Json(saved))
}

fn planner_identity(provider: &str) -> Result<(Option<&'static str>, &'static str), ApiError> {
    match provider {
        "prompt_refiner" => Ok((None, "TheDrummer/Anubis-Mini-8B-v1")),
        "native" => Ok((Some(QWEN36_FILM_PLANNER_MODEL_ID), QWEN36_FILM_PLANNER_REPO)),
        other => Err(ApiError::bad_request(format!(
            "Unsupported planning provider {other:?}; choose prompt_refiner or native"
        ))),
    }
}

fn production_brief(
    draft: &sceneworks_core::film_workspace::FilmDraft,
    structured: &FilmBriefDocument,
) -> ProductionBrief {
    let required_beats = structured
        .beats
        .iter()
        .map(|beat| {
            let lines = structured
                .dialogue
                .iter()
                .filter(|line| line.beat_id == beat.id)
                .map(|line| format!("{}: {}", line.speaker, line.text))
                .collect::<Vec<_>>();
            RequiredBeat {
                id: beat.id.clone(),
                summary: if lines.is_empty() {
                    beat.summary.clone()
                } else {
                    format!("{} Dialogue: {}", beat.summary, lines.join(" / "))
                },
                required_roles: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    let target = structured.target_total_seconds.max(5.1667);
    let mut limits = draft.production_plan.limits.clone();
    if limits.planner_max_memory_gb.is_none() {
        limits.planner_max_memory_gb = Some(limits.max_memory_gb);
    }
    ProductionBrief {
        schema_version: sceneworks_core::film_planner::BRIEF_SCHEMA_VERSION,
        id: draft.id.clone(),
        version: draft.revision,
        title: draft.title.clone(),
        synopsis: if structured.synopsis.trim().is_empty() {
            draft.brief.clone()
        } else {
            structured.synopsis.clone()
        },
        style_notes: structured.style_notes.clone(),
        target_total_seconds: DurationWindow {
            min: (target * 0.75).max(0.1),
            max: target * 1.25,
        },
        required_beats,
        model: draft.production_plan.model.clone(),
        limits,
        max_shots: structured
            .beats
            .len()
            .clamp(1, sceneworks_core::film_planner::MAX_PLANNER_SHOTS),
        prefer_quality: draft.effective_render_regime() == FilmRenderRegime::Quality,
    }
}

fn default_max_repair_rounds() -> u32 {
    crate::film_planner::DEFAULT_MAX_REPAIR_ROUNDS
}

fn finish_planning_operation(
    root: &FsPath,
    latest_path: &FsPath,
    operation_id: &str,
    result: Result<crate::film_planner::PlannerArtifacts, HarnessError>,
) {
    let canceled = root
        .join(crate::film_harness::CANCEL_SENTINEL_FILE)
        .exists();
    let _ = update_operation(latest_path, operation_id, |operation| match result {
        Ok(artifacts) => {
            operation.status = "ready".to_owned();
            operation.stage = "review".to_owned();
            operation.progress = Some(1.0);
            operation.active_job_id = None;
            operation.executions = artifacts
                .compiled
                .planner
                .as_ref()
                .map(|planner| planner.executions.clone())
                .unwrap_or_default();
            operation.candidate_plan = Some(artifacts.plan);
            operation.compiled = Some(artifacts.compiled);
            operation.detail = Some(
                "Candidate ready. Review it, then explicitly replace the current edited plan."
                    .to_owned(),
            );
        }
        Err(error) => {
            match &error {
                HarnessError::PlannerResponse { execution, .. } => {
                    operation.executions.push((**execution).clone());
                }
                HarnessError::PlannerValidation { executions, .. } => {
                    operation.executions = executions.clone();
                }
                HarnessError::PlannerExecutionFailure { executions, .. } => {
                    operation.executions = executions.clone();
                }
                _ => {}
            }
            operation.status = if canceled { "canceled" } else { "failed" }.to_owned();
            operation.stage = if canceled { "canceled" } else { "failed" }.to_owned();
            operation.active_job_id = None;
            operation.findings = findings_from_error(error);
            operation.detail = Some(
                "The draft and current shot plan were preserved. You can edit them manually or retry."
                    .to_owned(),
            );
        }
    });
}

/// Reacquire crashed planning controllers and replay their exact durable job ids. Completed jobs
/// rebuild deterministic planner state, queued/running jobs are polled in place, and jobs marked
/// interrupted by API startup are terminal before one replacement request is dispatched.
pub(crate) fn spawn_film_planning_startup_reconciliation(
    state: AppState,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let projects = match project_call(state.clone(), |store| store.list_projects()).await {
            Ok(projects) => projects,
            Err(error) => {
                tracing::warn!(?error, "film planning recovery could not list projects");
                return;
            }
        };
        for project in projects {
            let drafts = match project_call(state.clone(), {
                let project_id = project.id.clone();
                move |store| store.list_film_drafts(&project_id)
            })
            .await
            {
                Ok(drafts) => drafts,
                Err(error) => {
                    tracing::warn!(project_id = %project.id, ?error, "film planning recovery could not list drafts");
                    continue;
                }
            };
            for draft in drafts {
                let root = PathBuf::from(&project.path)
                    .join("films/planning")
                    .join(&draft.id);
                let latest_path = root.join("latest.json");
                let operation = match read_operation(&latest_path) {
                    Ok(operation)
                        if matches!(operation.status.as_str(), "running" | "canceling") =>
                    {
                        operation
                    }
                    _ => continue,
                };
                let lease = match ControllerLease::acquire(
                    &root,
                    format!("startup-adopt:planning:{}", draft.id),
                ) {
                    Ok(lease) => lease,
                    Err(HarnessError::Refused(_)) => continue,
                    Err(error) => {
                        tracing::warn!(draft_id = %draft.id, %error, "film planning recovery lease failed");
                        continue;
                    }
                };
                let state = state.clone();
                tokio::spawn(async move {
                    if operation.status == "canceling"
                        || root
                            .join(crate::film_harness::CANCEL_SENTINEL_FILE)
                            .exists()
                    {
                        reconcile_canceled_planning(&state, &root, &latest_path, &operation).await;
                        drop(lease);
                        return;
                    }
                    if operation.provider == "openai_compatible" {
                        interrupt_external_planning(&latest_path, &operation);
                        drop(lease);
                        return;
                    }
                    let operation_dir = root.join("operations").join(&operation.id);
                    let options = PlannerOptions {
                        brief_path: operation_dir.join("brief.json"),
                        reference_pack_path: operation_dir.join("references.json"),
                        out_dir: operation_dir,
                        max_repair_rounds: operation.max_repair_rounds,
                        refine_prompts: operation.refine_prompts,
                        prompt_guide_path: None,
                        require_installed: true,
                        require_local_planner: true,
                        send_reference_pixels: false,
                        api_url: state.settings.mcp_api_url.clone(),
                        force: false,
                        poll_interval: Duration::from_millis(350),
                        job_timeout: DEFAULT_LLM_JOB_TIMEOUT,
                    };
                    let callback_path = latest_path.clone();
                    let callback_operation_id = operation.id.clone();
                    let on_job_created = std::sync::Arc::new(move |job_id: &str| {
                        let _ = update_operation(
                            &callback_path,
                            &callback_operation_id,
                            |current| {
                                if current.status != "running" {
                                    return;
                                }
                                current.stage = "generating".to_owned();
                                current.active_job_id = Some(job_id.to_owned());
                                if !current.job_ids.iter().any(|known| known == job_id) {
                                    current.job_ids.push(job_id.to_owned());
                                }
                                current.detail = Some(
                                    "Recovered planning dispatched a replacement for an interrupted local job."
                                        .to_owned(),
                                );
                            },
                        );
                    });
                    let progress_path = latest_path.clone();
                    let progress_operation_id = operation.id.clone();
                    let on_job_progress =
                        std::sync::Arc::new(move |job_id: &str, progress: f64| {
                            let _ = update_operation(
                                &progress_path,
                                &progress_operation_id,
                                |current| {
                                    if current.status == "running"
                                        && current.active_job_id.as_deref() == Some(job_id)
                                    {
                                        current.progress = Some(progress);
                                    }
                                },
                            );
                        });
                    let cancel_path = root.join(crate::film_harness::CANCEL_SENTINEL_FILE);
                    let cancel_requested = std::sync::Arc::new(move || cancel_path.exists());
                    let result = async {
                        let transport = HttpTransport::new(
                            &state.settings.mcp_api_url,
                            Some(state.settings.access_token.clone()),
                        )?;
                        let model_override = (operation.provider == "native")
                            .then(|| operation.planner_model.clone());
                        let llm = SceneWorksLlm::new(
                            &transport,
                            options.poll_interval,
                            options.job_timeout,
                        )
                        .with_planner_model(model_override, operation.thinking_mode.clone())
                        .on_job_created(on_job_created)
                        .on_job_progress(on_job_progress)
                        .cancel_requested(cancel_requested)
                        .adopt_jobs(operation.job_ids.clone());
                        crate::film_planner::generate(&transport, &llm, &options).await
                    }
                    .await;
                    finish_planning_operation(&root, &latest_path, &operation.id, result);
                    drop(lease);
                });
            }
        }
    })
}

fn interrupt_external_planning(latest_path: &FsPath, operation: &FilmPlanningOperation) {
    let _ = update_operation(latest_path, &operation.id, |current| {
        current.status = "interrupted".to_owned();
        current.stage = "interrupted".to_owned();
        current.active_job_id = None;
        if !current.findings.iter().any(|finding| {
            finding.field == "planning.operation"
                && finding.message.contains("durable remote job retrieval")
        }) {
            current.findings.push(PlanDiagnostic::plan(
                "planning.operation",
                "The external Chat Completions request was interrupted by API restart. This provider has no durable remote job retrieval, so SceneWorks did not resend the paid request. Review the preserved operation and explicitly retry planning.",
            ));
        }
        current.detail = Some(
            "External planning was interrupted. Draft inputs, job history, findings, and any candidate artifacts were preserved; retry is explicit."
                .to_owned(),
        );
    });
}

async fn reconcile_canceled_planning(
    state: &AppState,
    root: &FsPath,
    latest_path: &FsPath,
    operation: &FilmPlanningOperation,
) {
    if let (Some(job_id), Ok(transport)) = (
        operation.active_job_id.as_deref(),
        HttpTransport::new(
            &state.settings.mcp_api_url,
            Some(state.settings.access_token.clone()),
        ),
    ) {
        let _ = crate::film_harness::expect_ok_on(
            &transport,
            "POST",
            &format!("/api/v1/jobs/{job_id}/cancel"),
            None,
        )
        .await;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        while tokio::time::Instant::now() < deadline {
            match crate::film_harness::expect_ok_on(
                &transport,
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                None,
            )
            .await
            {
                Ok(snapshot)
                    if matches!(
                        snapshot.get("status").and_then(Value::as_str),
                        Some("completed" | "failed" | "canceled" | "interrupted")
                    ) =>
                {
                    break;
                }
                _ => tokio::time::sleep(Duration::from_millis(250)).await,
            }
        }
    }
    let _ = update_operation(latest_path, &operation.id, |current| {
        current.status = "canceled".to_owned();
        current.stage = "canceled".to_owned();
        current.active_job_id = None;
        current.detail = Some(
            "Planning canceled after bounded reconciliation; completed jobs and artifacts were preserved."
                .to_owned(),
        );
    });
    let _ = std::fs::remove_file(root.join(crate::film_harness::CANCEL_SENTINEL_FILE));
}

async fn planning_root(
    state: AppState,
    project_id: &str,
    draft_id: &str,
) -> Result<PathBuf, ApiError> {
    if draft_id.is_empty()
        || draft_id.len() > 128
        || !draft_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(ApiError::bad_request("Invalid film draft ID"));
    }
    Ok(project_path_for_id(state, project_id)
        .await?
        .join("films/planning")
        .join(draft_id))
}

fn findings_from_error(error: HarnessError) -> Vec<PlanDiagnostic> {
    match error {
        HarnessError::Validation(findings) | HarnessError::PlannerValidation { findings, .. } => {
            findings
        }
        HarnessError::PlannerExecutionFailure { source, .. } => findings_from_error(*source),
        other => vec![PlanDiagnostic::plan(
            "planning.operation",
            other.to_string(),
        )],
    }
}

fn read_operation(path: &FsPath) -> Result<FilmPlanningOperation, ApiError> {
    let bytes = std::fs::read(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ApiError {
                status: StatusCode::NOT_FOUND,
                detail: "No planning operation exists for this draft".to_owned(),
                context: None,
                code: None,
            }
        } else {
            ApiError::internal(error.to_string())
        }
    })?;
    serde_json::from_slice(&bytes).map_err(|error| ApiError::internal(error.to_string()))
}

fn write_json_file<T: Serialize>(path: &FsPath, value: &T) -> Result<(), ApiError> {
    let bytes =
        serde_json::to_vec_pretty(value).map_err(|error| ApiError::internal(error.to_string()))?;
    std::fs::write(path, bytes).map_err(|error| ApiError::internal(error.to_string()))
}

fn write_latest_operation(
    root: &FsPath,
    operation: &FilmPlanningOperation,
) -> Result<(), ApiError> {
    std::fs::create_dir_all(root).map_err(|error| ApiError::internal(error.to_string()))?;
    let path = root.join("latest.json");
    let temp = root.join(format!(
        ".latest-{}-{}.tmp",
        operation.id,
        uuid::Uuid::new_v4().simple()
    ));
    write_json_file(&temp, operation)?;
    std::fs::rename(temp, path).map_err(|error| ApiError::internal(error.to_string()))
}

fn update_operation(
    path: &FsPath,
    operation_id: &str,
    mutate: impl FnOnce(&mut FilmPlanningOperation),
) -> Result<(), ApiError> {
    let mut operation = read_operation(path)?;
    if operation.id != operation_id {
        return Ok(());
    }
    mutate(&mut operation);
    operation.updated_at = sceneworks_core::time::utc_now();
    write_latest_operation(path.parent().expect("latest path has parent"), &operation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::film_workspace::{FilmDraft, FilmRenderRegime};

    #[test]
    fn external_orphan_is_preserved_and_interrupted_without_constructing_a_transport() {
        let temp = tempfile::tempdir().expect("tempdir");
        let draft = FilmDraft::manual_one_shot("project_1", "film_1", "Courier");
        let original_finding = PlanDiagnostic::plan("planning.input", "Keep this finding");
        let operation = FilmPlanningOperation {
            schema_version: OPERATION_SCHEMA_VERSION,
            id: "planning_external_1".to_owned(),
            project_id: "project_1".to_owned(),
            draft_id: draft.id.clone(),
            draft_revision: draft.revision,
            status: "running".to_owned(),
            stage: "generating".to_owned(),
            progress: Some(0.4),
            provider: "openai_compatible".to_owned(),
            planner_model_id: None,
            planner_model: "external-model".to_owned(),
            video_model_id: draft.production_plan.model.id.clone(),
            thinking_mode: "disabled".to_owned(),
            max_repair_rounds: 2,
            refine_prompts: false,
            active_job_id: Some("remote_request_1".to_owned()),
            job_ids: vec!["remote_request_1".to_owned()],
            findings: vec![original_finding.clone()],
            executions: Vec::new(),
            candidate_plan: Some(draft.production_plan.clone()),
            compiled: None,
            detail: Some("Remote request was in flight".to_owned()),
            created_at: "2026-09-16T00:00:00Z".to_owned(),
            updated_at: "2026-09-16T00:00:00Z".to_owned(),
        };
        write_latest_operation(temp.path(), &operation).expect("write operation");
        let latest_path = temp.path().join("latest.json");

        // This recovery branch has no transport or planner parameter: an external orphan can only
        // be made terminal and actionable, never silently re-dispatched as a paid request.
        interrupt_external_planning(&latest_path, &operation);
        interrupt_external_planning(&latest_path, &operation);

        let recovered = read_operation(&latest_path).expect("read recovered operation");
        assert_eq!(recovered.status, "interrupted");
        assert_eq!(recovered.stage, "interrupted");
        assert_eq!(recovered.active_job_id, None);
        assert_eq!(recovered.job_ids, operation.job_ids);
        assert_eq!(recovered.candidate_plan, operation.candidate_plan);
        assert!(recovered.findings.contains(&original_finding));
        let recovery_findings = recovered
            .findings
            .iter()
            .filter(|finding| finding.field == "planning.operation")
            .collect::<Vec<_>>();
        assert_eq!(recovery_findings.len(), 1);
        assert!(recovery_findings[0].message.contains("explicitly retry"));
        assert!(recovered
            .detail
            .as_deref()
            .unwrap_or_default()
            .contains("retry is explicit"));
    }

    #[test]
    fn failed_external_response_keeps_input_and_sanitized_execution() {
        let temp = tempfile::tempdir().expect("tempdir");
        let draft = FilmDraft::manual_one_shot("project_1", "film_1", "Courier");
        let operation = FilmPlanningOperation {
            schema_version: OPERATION_SCHEMA_VERSION,
            id: "planning_external_failed".to_owned(),
            project_id: "project_1".to_owned(),
            draft_id: draft.id.clone(),
            draft_revision: draft.revision,
            status: "running".to_owned(),
            stage: "generating".to_owned(),
            progress: None,
            provider: "openai_compatible".to_owned(),
            planner_model_id: None,
            planner_model: "external-model".to_owned(),
            video_model_id: draft.production_plan.model.id.clone(),
            thinking_mode: "disabled".to_owned(),
            max_repair_rounds: 2,
            refine_prompts: false,
            active_job_id: None,
            job_ids: Vec::new(),
            findings: Vec::new(),
            executions: Vec::new(),
            candidate_plan: Some(draft.production_plan.clone()),
            compiled: None,
            detail: Some("External request is active".to_owned()),
            created_at: "2026-09-17T00:00:00Z".to_owned(),
            updated_at: "2026-09-17T00:00:00Z".to_owned(),
        };
        write_latest_operation(temp.path(), &operation).expect("write operation");
        let latest_path = temp.path().join("latest.json");
        let execution = PlannerExecutionRecord {
            provider: "openai_compatible".to_owned(),
            model: "external-model".to_owned(),
            backend: Some("fixture".to_owned()),
            target_video_model_id: draft.production_plan.model.id.clone(),
            thinking_mode: "disabled".to_owned(),
            max_output_tokens: Some(4096),
            reference_pixels_sent: Some(true),
            duration_seconds: Some(12.5),
            finish_reason: Some("length".to_owned()),
            failure_code: Some("no_textual_plan_content".to_owned()),
            thinking: Some("separate reasoning".to_owned()),
            usage: Some(sceneworks_core::film_compile::PlannerUsageRecord {
                input_tokens: Some(17),
                output_tokens: Some(4096),
                total_tokens: Some(4113),
            }),
            ..PlannerExecutionRecord::default()
        };
        let prior_execution = PlannerExecutionRecord {
            model: "external-model-prior-malformed".to_owned(),
            finish_reason: Some("stop".to_owned()),
            failure_code: None,
            ..execution.clone()
        };

        finish_planning_operation(
            temp.path(),
            &latest_path,
            &operation.id,
            Err(HarnessError::PlannerExecutionFailure {
                source: Box::new(HarnessError::PlannerResponse {
                    detail: "The external planner response has no textual plan content".to_owned(),
                    execution: Box::new(execution.clone()),
                }),
                executions: vec![prior_execution.clone(), execution.clone()],
            }),
        );

        let failed = read_operation(&latest_path).expect("read failed operation");
        assert_eq!(failed.status, "failed");
        assert_eq!(failed.stage, "failed");
        assert_eq!(failed.provider, "openai_compatible");
        assert!(
            failed.job_ids.is_empty(),
            "no native fallback job was created"
        );
        assert_eq!(failed.candidate_plan, operation.candidate_plan);
        assert_eq!(failed.compiled, operation.compiled);
        assert_eq!(failed.executions, vec![prior_execution, execution]);
        assert_eq!(failed.findings.len(), 1);
        assert!(failed.findings[0]
            .message
            .contains("no textual plan content"));
        assert!(failed.detail.as_deref().unwrap().contains("retry"));
    }

    #[tokio::test]
    async fn malformed_external_repairs_keep_every_execution_and_the_current_plan() {
        use axum::extract::State;
        use axum::routing::post;
        use axum::{Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};

        use crate::tests::film_harness::{planner_options, Harness};

        #[derive(Clone, Default)]
        struct MalformedFixture {
            calls: Arc<AtomicUsize>,
            requests: Arc<Mutex<Vec<Value>>>,
        }

        async fn malformed_response(
            State(fixture): State<MalformedFixture>,
            Json(body): Json<Value>,
        ) -> Json<Value> {
            let attempt = fixture.calls.fetch_add(1, Ordering::SeqCst) + 1;
            fixture.requests.lock().unwrap().push(body);
            Json(json!({
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": "{\"shots\":[",
                        "reasoning_content": format!("bounded reasoning {attempt}")
                    }
                }],
                "usage": {
                    "prompt_tokens": 100 + attempt,
                    "completion_tokens": 10 + attempt,
                    "total_tokens": 110 + (2 * attempt)
                },
                "provider_debug": {"credential": "must-not-survive"}
            }))
        }

        let fixture = MalformedFixture::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind malformed fixture");
        let address = listener.local_addr().expect("fixture address");
        let app = Router::new()
            .route("/v1/chat/completions", post(malformed_response))
            .with_state(fixture.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve fixture");
        });

        let harness = Harness::start(false, vec![]).await;
        let root = tempfile::tempdir().expect("operation root");
        let draft = FilmDraft::manual_one_shot("project_1", "film_1", "Existing cut");
        let operation = FilmPlanningOperation {
            schema_version: OPERATION_SCHEMA_VERSION,
            id: "planning_external_malformed".to_owned(),
            project_id: "project_1".to_owned(),
            draft_id: draft.id.clone(),
            draft_revision: draft.revision,
            status: "running".to_owned(),
            stage: "planning".to_owned(),
            progress: None,
            provider: "openai_compatible".to_owned(),
            planner_model_id: Some("external-model".to_owned()),
            planner_model: "external-model".to_owned(),
            video_model_id: "minimax_h3".to_owned(),
            thinking_mode: "enabled".to_owned(),
            max_repair_rounds: 2,
            refine_prompts: false,
            active_job_id: None,
            job_ids: Vec::new(),
            findings: Vec::new(),
            executions: Vec::new(),
            candidate_plan: Some(draft.production_plan.clone()),
            compiled: None,
            detail: Some("External request is active".to_owned()),
            created_at: "2026-09-17T00:00:00Z".to_owned(),
            updated_at: "2026-09-17T00:00:00Z".to_owned(),
        };
        write_latest_operation(root.path(), &operation).expect("write operation");
        let latest_path = root.path().join("latest.json");

        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            crate::film_planner_connections::FilmPlannerConnection {
                schema_version: 1,
                id: "malformed-fixture".to_owned(),
                label: "Malformed fixture".to_owned(),
                base_url: format!("http://{address}/v1"),
                credential_host: None,
                supports_model_listing: false,
                supports_image_input: true,
                timeout_seconds: 5,
                max_output_tokens: 3072,
            },
            None,
            OpenAiPlannerOptions {
                model: "external-model".to_owned(),
                thinking_mode: "enabled".to_owned(),
                source_script: "A courier enters with a parcel.".to_owned(),
                send_reference_pixels: true,
            },
            Arc::new(|| false),
        )
        .expect("planner adapter");
        let mut options = planner_options(&harness, "malformed-external");
        options.out_dir = root.path().join("artifacts");
        options.require_local_planner = false;
        options.send_reference_pixels = true;
        options.max_repair_rounds = 2;

        let result = crate::film_planner::generate(&harness.transport, &llm, &options).await;
        assert!(
            matches!(result, Err(HarnessError::PlannerValidation { .. })),
            "repair exhaustion must retain planner execution provenance: {result:?}"
        );
        finish_planning_operation(root.path(), &latest_path, &operation.id, result);

        let failed = read_operation(&latest_path).expect("read failed operation");
        assert_eq!(failed.status, "failed");
        assert_eq!(failed.stage, "failed");
        assert_eq!(
            failed.candidate_plan, operation.candidate_plan,
            "bounded planner failure must preserve the current edited plan"
        );
        assert_eq!(failed.compiled, None);
        assert_eq!(failed.executions.len(), 3, "initial call plus two repairs");
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 3);
        assert!(
            !options.plan_path().exists(),
            "no refused plan becomes current"
        );
        assert!(options.out_dir.join("planner-rejected.txt").is_file());
        assert!(failed
            .findings
            .iter()
            .any(|finding| finding.field == "planner.repair"));

        for (index, execution) in failed.executions.iter().enumerate() {
            let attempt = index as u64 + 1;
            assert_eq!(execution.provider, "openai_compatible");
            assert_eq!(execution.model, "external-model");
            assert_eq!(execution.backend.as_deref(), Some("malformed-fixture"));
            assert_eq!(execution.target_video_model_id, "minimax_h3");
            assert_eq!(execution.thinking_mode, "enabled");
            assert_eq!(execution.max_output_tokens, Some(3072));
            assert_eq!(execution.reference_pixels_sent, Some(true));
            assert!(execution.duration_seconds.is_some_and(|value| value >= 0.0));
            assert_eq!(execution.finish_reason.as_deref(), Some("stop"));
            assert_eq!(execution.failure_code, None);
            assert_eq!(
                execution.thinking.as_deref(),
                Some(format!("bounded reasoning {attempt}").as_str())
            );
            let usage = execution.usage.as_ref().expect("usage survives");
            assert_eq!(usage.input_tokens, Some(100 + attempt));
            assert_eq!(usage.output_tokens, Some(10 + attempt));
            assert_eq!(usage.total_tokens, Some(110 + (2 * attempt)));
        }
        let requests = fixture.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests
            .iter()
            .all(|request| request.to_string().contains("image_url")));
        let persisted = std::fs::read_to_string(&latest_path).expect("operation readable");
        assert!(!persisted.contains("must-not-survive"));
        server.abort();
    }

    #[tokio::test]
    async fn external_operation_reports_planning_while_provider_response_is_blocked() {
        use axum::extract::State;
        use axum::routing::post;
        use axum::{Json, Router};
        use std::sync::Arc;
        use tokio::sync::Notify;

        #[derive(Clone)]
        struct BlockedFixture {
            entered: Arc<Notify>,
            release: Arc<Notify>,
        }

        async fn blocked_response(
            State(fixture): State<BlockedFixture>,
            Json(_body): Json<Value>,
        ) -> Json<Value> {
            fixture.entered.notify_one();
            fixture.release.notified().await;
            Json(json!({"choices": [{"message": {"content": "{}"}}]}))
        }

        let fixture = BlockedFixture {
            entered: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fixture");
        let address = listener.local_addr().expect("fixture address");
        let app = Router::new()
            .route("/v1/chat/completions", post(blocked_response))
            .with_state(fixture.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve fixture");
        });

        let temp = tempfile::tempdir().expect("tempdir");
        let operation = FilmPlanningOperation {
            schema_version: OPERATION_SCHEMA_VERSION,
            id: "planning_external_blocked".to_owned(),
            project_id: "project_1".to_owned(),
            draft_id: "film_1".to_owned(),
            draft_revision: 1,
            status: "running".to_owned(),
            stage: "preflight".to_owned(),
            progress: None,
            provider: "openai_compatible".to_owned(),
            planner_model_id: Some("external-model".to_owned()),
            planner_model: "external-model".to_owned(),
            video_model_id: "minimax_h3".to_owned(),
            thinking_mode: "disabled".to_owned(),
            max_repair_rounds: 2,
            refine_prompts: false,
            active_job_id: None,
            job_ids: Vec::new(),
            findings: Vec::new(),
            executions: Vec::new(),
            candidate_plan: None,
            compiled: None,
            detail: Some(
                "Checking the selected external connection and target video model.".to_owned(),
            ),
            created_at: "2026-09-17T00:00:00Z".to_owned(),
            updated_at: "2026-09-17T00:00:00Z".to_owned(),
        };
        write_latest_operation(temp.path(), &operation).expect("write operation");
        let latest_path = temp.path().join("latest.json");
        let callback_path = latest_path.clone();
        let callback_operation_id = operation.id.clone();
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            crate::film_planner_connections::FilmPlannerConnection {
                schema_version: 1,
                id: "blocked-fixture".to_owned(),
                label: "Blocked fixture".to_owned(),
                base_url: format!("http://{address}/v1"),
                credential_host: None,
                supports_model_listing: false,
                supports_image_input: false,
                timeout_seconds: 5,
                max_output_tokens: 4096,
            },
            None,
            OpenAiPlannerOptions {
                model: "external-model".to_owned(),
                thinking_mode: "disabled".to_owned(),
                source_script: "A courier enters.".to_owned(),
                send_reference_pixels: false,
            },
            Arc::new(|| false),
        )
        .expect("planner adapter")
        .on_request_started(Arc::new(move || {
            mark_external_planner_waiting(&callback_path, &callback_operation_id);
        }));
        let request = crate::film_planner::LlmRequest {
            task: Some(crate::film_planner::FILM_PLAN_TASK.to_owned()),
            prompt: "Return a plan".to_owned(),
            model_id: Some("minimax_h3".to_owned()),
            workflow: "video".to_owned(),
            guide: None,
            reference_images: Vec::new(),
        };
        let response =
            tokio::spawn(
                async move { crate::film_planner::PlannerLlm::complete(&llm, request).await },
            );

        tokio::time::timeout(Duration::from_secs(1), fixture.entered.notified())
            .await
            .expect("provider request entered blocked fixture");
        let waiting = read_operation(&latest_path).expect("read waiting operation");
        assert_eq!(waiting.status, "running");
        assert_eq!(waiting.stage, "planning");
        assert_eq!(waiting.progress, None);
        assert_eq!(
            waiting.detail.as_deref(),
            Some("Waiting for the selected external planner to return a candidate plan.")
        );

        fixture.release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), response)
            .await
            .expect("fixture response completes")
            .expect("planner task joins")
            .expect("planner response succeeds");
        server.abort();
    }

    #[test]
    fn structured_dialogue_is_part_of_shared_planner_brief_without_changing_video_identity() {
        let mut draft = FilmDraft::manual_one_shot("project_1", "film_1", "Courier");
        draft.original_script = "MARA\nPut it down.".to_owned();
        let structured = parse_film_script(&draft.original_script);
        let brief = production_brief(&draft, &structured);
        assert_eq!(brief.model.id, "minimax_h3");
        assert!(brief.required_beats[0]
            .summary
            .contains("MARA: Put it down."));
        assert_eq!(draft.original_script, "MARA\nPut it down.");
    }

    #[test]
    fn planner_brief_pins_the_saved_render_regime_outside_the_llm_answer() {
        let mut draft = FilmDraft::manual_one_shot("project_1", "film_1", "Courier");
        draft.original_script = "A courier enters.".to_owned();
        draft.production_plan.model.loras = vec!["minimax_h3_turbo_4step_v01".to_owned()];
        let structured = parse_film_script(&draft.original_script);
        let recommended = production_brief(&draft, &structured);
        assert!(!recommended.prefer_quality);
        assert_eq!(recommended.model.loras, vec!["minimax_h3_turbo_4step_v01"]);

        draft.render_regime = Some(FilmRenderRegime::Quality);
        draft.production_plan.model.loras.clear();
        let quality = production_brief(&draft, &structured);
        assert!(quality.prefer_quality);
        assert!(quality.model.loras.is_empty());
    }
}
