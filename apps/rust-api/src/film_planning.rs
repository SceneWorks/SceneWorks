use super::*;

use sceneworks_core::film_compile::{CompiledPlan, PlannerExecutionRecord};
use sceneworks_core::film_plan::{PlanDiagnostic, ProductionPlan};
use sceneworks_core::film_planner::{DurationWindow, ProductionBrief, RequiredBeat};
use sceneworks_core::film_workspace::{
    parse_film_script, FilmBriefDocument, QWEN36_FILM_PLANNER_MODEL_ID, QWEN36_FILM_PLANNER_REPO,
};

use crate::film_harness::{ControllerLease, HarnessError, HttpTransport};
use crate::film_planner::{PlannerOptions, SceneWorksLlm, DEFAULT_LLM_JOB_TIMEOUT};

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
    let draft = project_call(state.clone(), {
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
    let (planner_model_id, planner_model) = planner_identity(&draft.planning.provider)?;
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
        planner_model_id: planner_model_id.map(str::to_owned),
        planner_model: planner_model.to_owned(),
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
        detail: Some("Checking the selected local planner and video model.".to_owned()),
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
    let options = PlannerOptions {
        brief_path: operation_dir.join("brief.json"),
        reference_pack_path: operation_dir.join("references.json"),
        out_dir: operation_dir,
        max_repair_rounds: operation.max_repair_rounds,
        refine_prompts: operation.refine_prompts,
        prompt_guide_path: None,
        require_installed: true,
        api_url: base_url.clone(),
        force: false,
        poll_interval: Duration::from_millis(350),
        job_timeout: DEFAULT_LLM_JOB_TIMEOUT,
    };
    let model_override = (draft.planning.provider == "native").then(|| planner_model.to_owned());
    let thinking_mode = draft.planning.thinking_mode.clone();
    tokio::spawn(async move {
        let result = async {
            let transport = HttpTransport::new(&base_url, Some(token))?;
            let llm = SceneWorksLlm::new(&transport, options.poll_interval, options.job_timeout)
                .with_planner_model(model_override, thinking_mode)
                .on_job_created(on_job_created)
                .on_job_progress(on_job_progress)
                .cancel_requested(cancel_requested);
            crate::film_planner::generate(&transport, &llm, &options).await
        }
        .await;
        finish_planning_operation(&root, &latest_path, &operation_id, result);
        drop(lease);
    });

    Ok((StatusCode::ACCEPTED, Json(operation)))
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
        Some("Cancellation requested; waiting for the native decode to stop.".to_owned());
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
    let candidate = operation
        .candidate_plan
        .ok_or_else(|| ApiError::internal("Ready planning operation has no candidate plan"))?;
    let saved = project_call(state, move |store| {
        let mut draft = store.get_film_draft(&project_id, &draft_id)?;
        if draft.revision != operation.draft_revision {
            return Err(sceneworks_core::project_store::ProjectStoreError::BadRequest(format!(
                "Film draft changed after planning started (candidate revision {}, current revision {}); regenerate before replacing the edited plan",
                operation.draft_revision, draft.revision
            )));
        }
        draft.production_plan = candidate;
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
        prefer_quality: false,
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
                    let operation_dir = root.join("operations").join(&operation.id);
                    let options = PlannerOptions {
                        brief_path: operation_dir.join("brief.json"),
                        reference_pack_path: operation_dir.join("references.json"),
                        out_dir: operation_dir,
                        max_repair_rounds: operation.max_repair_rounds,
                        refine_prompts: operation.refine_prompts,
                        prompt_guide_path: None,
                        require_installed: true,
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
        HarnessError::Validation(findings) => findings,
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
    use sceneworks_core::film_workspace::FilmDraft;

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
}
