use super::*;

use sceneworks_core::film_plan::{RunRecord, RunState};
use sceneworks_core::film_workspace::FilmRunLocator;

use crate::film_harness::{ControllerLease, HttpTransport, ResumeOptions};
use crate::films::{load_run_view, FilmRunView};

pub(crate) async fn list_film_runs(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<FilmRunView>>, ApiError> {
    let locators = project_call(state.clone(), {
        let project_id = project_id.clone();
        move |store| store.list_film_runs(&project_id)
    })
    .await?;
    let mut views = Vec::with_capacity(locators.len());
    for locator in locators {
        views.push(load_run_view(state.clone(), project_id.clone(), locator.id).await?);
    }
    Ok(Json(views))
}

pub(crate) async fn get_film_run_progress(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
) -> Result<Json<FilmRunView>, ApiError> {
    Ok(Json(load_run_view(state, project_id, run_id).await?))
}

pub(crate) async fn resume_film_run(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
) -> Result<(StatusCode, Json<FilmRunView>), ApiError> {
    let mut view = load_run_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    let record = view
        .record
        .as_ref()
        .ok_or_else(|| ApiError::conflict("Film run has not started; use its start operation"))?;
    if !record.is_resumable() {
        return Err(ApiError::conflict(actionable_stop(record)));
    }
    let files = project_call(state.clone(), {
        let project_id = project_id.clone();
        let run_id = run_id.clone();
        move |store| store.film_run_files(&project_id, &run_id)
    })
    .await?;
    let lease = ControllerLease::acquire(&files.directory, format!("api-resume:{run_id}"))
        .map_err(|error| ApiError::conflict(error.to_string()))?;
    view.controller_active = true;
    view.controller_owner = Some(format!("api-resume:{run_id}"));
    view.controller_interrupted = false;
    spawn_resume(state, project_id, run_id, files.directory, lease);
    Ok((StatusCode::ACCEPTED, Json(view)))
}

pub(crate) async fn cancel_film_run(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
) -> Result<(StatusCode, Json<FilmRunView>), ApiError> {
    let view = load_run_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    if view.record.is_none() {
        return Err(ApiError::conflict(
            "Film run has not started; there is nothing to cancel",
        ));
    }
    if !view.controller_active {
        return Err(ApiError::conflict(
            "Film run has no active controller. Resume it to reconcile its saved jobs, or leave its completed assets unchanged.",
        ));
    }
    let directory = project_call(state, move |store| {
        Ok(store.film_run_files(&project_id, &run_id)?.directory)
    })
    .await?;
    crate::film_harness::request_cancel(&directory)
        .map_err(|error| ApiError::conflict(error.to_string()))?;
    Ok((StatusCode::ACCEPTED, Json(view)))
}

fn actionable_stop(record: &sceneworks_core::film_plan::RunRecord) -> String {
    record
        .stop
        .as_ref()
        .map(|stop| format!("Film run cannot resume: {}. {}", stop.reason, stop.detail))
        .unwrap_or_else(|| format!("Film run cannot resume after {:?}", record.outcome))
}

fn spawn_resume(
    state: AppState,
    project_id: String,
    run_id: String,
    directory: std::path::PathBuf,
    lease: ControllerLease,
) {
    spawn_resume_with_install_requirement(state, project_id, run_id, directory, lease, true, None);
}

type StartupResumeResults = tokio::sync::mpsc::UnboundedSender<Result<RunRecord, String>>;

fn spawn_resume_with_install_requirement(
    state: AppState,
    project_id: String,
    run_id: String,
    directory: std::path::PathBuf,
    lease: ControllerLease,
    require_installed: bool,
    completion: Option<StartupResumeResults>,
) {
    tokio::spawn(async move {
        let transport = match HttpTransport::new(
            &state.settings.mcp_api_url,
            Some(state.settings.access_token.clone()),
        ) {
            Ok(transport) => transport,
            Err(error) => {
                tracing::error!(project_id, run_id, %error, "film resume transport failed");
                if let Some(completion) = completion {
                    let _ = completion.send(Err(error.to_string()));
                }
                return;
            }
        };
        let mut options = ResumeOptions::new(directory);
        options.poll_interval = Duration::from_secs(2);
        options.export = false;
        options.require_installed = require_installed;
        let result = crate::film_harness::resume_with_lease(&transport, &options, lease)
            .await
            .map_err(|error| error.to_string());
        if let Err(error) = &result {
            tracing::error!(project_id, run_id, %error, "film resume stopped");
        }
        if let Some(completion) = completion {
            let _ = completion.send(result);
        }
    });
}

/// Adopt only records that say a controller was active when the process stopped. Resumable
/// operator stops remain idle until an explicit resume. Advisory leases make this safe after a
/// crash and refuse adoption when another API or CLI process still owns the run.
pub(crate) fn spawn_film_startup_reconciliation(state: AppState) -> tokio::task::JoinHandle<()> {
    spawn_film_startup_reconciliation_with_install_requirement(state, true, None)
}

#[cfg(test)]
pub(crate) struct FilmStartupReconciliationTestHandle {
    pub(crate) scan: tokio::task::JoinHandle<()>,
    pub(crate) controller_results: tokio::sync::mpsc::UnboundedReceiver<Result<RunRecord, String>>,
}

#[cfg(test)]
pub(crate) fn spawn_film_startup_reconciliation_for_fake_worker(
    state: AppState,
) -> FilmStartupReconciliationTestHandle {
    // The fake worker deliberately owns no multi-gigabyte model installation. Production startup
    // always enters through `spawn_film_startup_reconciliation` above and keeps this guard on.
    let (completion, controller_results) = tokio::sync::mpsc::unbounded_channel();
    FilmStartupReconciliationTestHandle {
        scan: spawn_film_startup_reconciliation_with_install_requirement(
            state,
            false,
            Some(completion),
        ),
        controller_results,
    }
}

fn spawn_film_startup_reconciliation_with_install_requirement(
    state: AppState,
    require_installed: bool,
    completion: Option<StartupResumeResults>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let projects = match project_call(state.clone(), |store| store.list_projects()).await {
            Ok(projects) => projects,
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "film startup reconciliation could not list projects"
                );
                return;
            }
        };
        for project in projects {
            let project_id = project.id.clone();
            let locators: Vec<FilmRunLocator> = match project_call(state.clone(), {
                let project_id = project_id.clone();
                move |store| store.list_film_runs(&project_id)
            })
            .await
            {
                Ok(locators) => locators,
                Err(error) => {
                    tracing::warn!(
                        project_id,
                        ?error,
                        "film startup reconciliation could not list runs"
                    );
                    continue;
                }
            };
            for locator in locators {
                let run_id = locator.id;
                let files = match project_call(state.clone(), {
                    let project_id = project_id.clone();
                    let run_id = run_id.clone();
                    move |store| store.film_run_files(&project_id, &run_id)
                })
                .await
                {
                    Ok(files) => files,
                    Err(error) => {
                        tracing::warn!(
                            project_id,
                            run_id,
                            ?error,
                            "film startup reconciliation could not open run"
                        );
                        continue;
                    }
                };
                let Ok(record) = crate::film_harness::read_run_record(&files.directory) else {
                    continue;
                };
                if record.state != RunState::Running {
                    continue;
                }
                let lease = match ControllerLease::acquire(
                    &files.directory,
                    format!("startup-adopt:{run_id}"),
                ) {
                    Ok(lease) => lease,
                    Err(crate::film_harness::HarnessError::Refused(_)) => continue,
                    Err(error) => {
                        tracing::warn!(project_id, run_id, %error, "film startup reconciliation lease failed");
                        continue;
                    }
                };
                spawn_resume_with_install_requirement(
                    state.clone(),
                    project_id.clone(),
                    run_id,
                    files.directory,
                    lease,
                    require_installed,
                    completion.clone(),
                );
            }
        }
    })
}
