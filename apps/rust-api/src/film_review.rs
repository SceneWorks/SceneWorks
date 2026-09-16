use super::*;

use std::path::{Component, Path as FsPath};

use sceneworks_core::film_plan::RunRecord;
use sceneworks_core::film_review::{ObservedState, ASSISTIVE_NOTICE};
use sceneworks_core::project_store::ProjectStoreError;

use crate::film_harness::review::{self, Decision, ReviewOptions, VqaVision};
use crate::film_harness::{
    ControllerLease, EditOptions, HarnessError, HttpTransport, ResumeOptions, RunControl,
    TimelineEdit,
};
use crate::films::{load_run_view, FilmRunView};

const TERMINAL_ATTEMPT_STATUSES: &[&str] = &[
    "completed",
    "failed",
    "canceled",
    "canceled_by_operator",
    "interrupted",
    "timed_out",
    "rejected",
];

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FilmReviewView {
    run: FilmRunView,
    take_assets: Vec<Value>,
    observations: Vec<ObservedState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    review_timeline_id: Option<String>,
    selections: Vec<FilmTakeSelection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    action_disabled_reason: Option<String>,
    assistive_notice: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FilmTakeSelection {
    shot_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    record_selected_attempt: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    record_selected_asset_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeline_attempt: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeline_asset_id: Option<String>,
    state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    trim_conflict: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FilmReviewDecisionRequest {
    shot_id: String,
    decision: String,
    #[serde(default)]
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FilmReviewSwapRequest {
    shot_id: String,
    asset_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FilmReviewMutationRequest {
    shot_id: String,
    #[serde(default)]
    reason: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FilmReviewAnalyzeRequest {
    #[serde(default)]
    shot_ids: Vec<String>,
}

pub(crate) async fn get_film_review(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
) -> Result<Json<FilmReviewView>, ApiError> {
    Ok(Json(load_review_view(state, project_id, run_id).await?))
}

pub(crate) async fn decide_film_take(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<FilmReviewDecisionRequest>,
) -> Result<Json<FilmReviewView>, ApiError> {
    let view = load_review_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    ensure_mutation_available(&view)?;
    ensure_saved_cut_selection(&view, &payload.shot_id)?;
    let decision = match payload.decision.as_str() {
        "accept" => Decision::Accept,
        "reject" => Decision::Reject,
        _ => return Err(ApiError::bad_request("decision must be accept or reject")),
    };
    let directory = run_directory(state.clone(), &project_id, &run_id).await?;
    review::decide_take(
        &directory,
        &payload.shot_id,
        decision,
        payload.reason.trim(),
    )
    .map_err(harness_error)?;
    Ok(Json(load_review_view(state, project_id, run_id).await?))
}

pub(crate) async fn swap_film_take(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<FilmReviewSwapRequest>,
) -> Result<Json<FilmReviewView>, ApiError> {
    let view = load_review_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    ensure_mutation_available(&view)?;
    let directory = run_directory(state.clone(), &project_id, &run_id).await?;
    let options = EditOptions {
        run_record_path: directory.join(crate::film_harness::RUN_RECORD_FILE),
        export: false,
        poll_interval: Duration::from_secs(2),
    };
    let transport = transport(&state)?;
    crate::film_harness::edit_timeline(
        &transport,
        &options,
        TimelineEdit::SwapTake {
            shot_id: payload.shot_id,
            asset_id: payload.asset_id,
        },
    )
    .await
    .map_err(harness_error)?;
    Ok(Json(load_review_view(state, project_id, run_id).await?))
}

pub(crate) async fn replace_film_take(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<FilmReviewMutationRequest>,
) -> Result<(StatusCode, Json<FilmReviewView>), ApiError> {
    spawn_take_mutation(state, project_id, run_id, payload, false).await
}

pub(crate) async fn repair_film_take(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<FilmReviewMutationRequest>,
) -> Result<(StatusCode, Json<FilmReviewView>), ApiError> {
    spawn_take_mutation(state, project_id, run_id, payload, true).await
}

pub(crate) async fn analyze_film_take(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<FilmReviewAnalyzeRequest>,
) -> Result<(StatusCode, Json<FilmReviewView>), ApiError> {
    let before = load_review_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    ensure_mutation_available(&before)?;
    let directory = run_directory(state.clone(), &project_id, &run_id).await?;
    let owner = format!("api-review:{run_id}");
    let lease = ControllerLease::acquire(&directory, owner).map_err(harness_error)?;
    let control = RunControl::watching(&directory);
    let mut options = ReviewOptions::new(directory.clone());
    options.review_plan_path = Some(directory.join("review.jsonc"));
    options.shot_ids = payload.shot_ids;
    options.poll_interval = Duration::from_secs(2);
    options.control = control.clone();
    let transport = transport(&state)?;
    let limits = review::review_limits(&directory, options.review_plan_path.as_deref())
        .map_err(harness_error)?;
    let vision = VqaVision::new(&transport, options.poll_interval, control);
    vision.preflight(limits).await.map_err(harness_error)?;
    vision.preflight_model().await.map_err(harness_error)?;
    let active = load_review_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    tokio::spawn(async move {
        let vision = VqaVision::new(&transport, options.poll_interval, options.control.clone());
        if let Err(error) = review::review_with_lease(&transport, &options, &vision, lease).await {
            tracing::error!(project_id, run_id, %error, "film review stopped");
        }
    });
    Ok((StatusCode::ACCEPTED, Json(active)))
}

async fn spawn_take_mutation(
    state: AppState,
    project_id: String,
    run_id: String,
    payload: FilmReviewMutationRequest,
    repair: bool,
) -> Result<(StatusCode, Json<FilmReviewView>), ApiError> {
    let before = load_review_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    ensure_mutation_available(&before)?;
    let directory = run_directory(state.clone(), &project_id, &run_id).await?;
    let action = if repair { "repair" } else { "replace" };
    let lease = ControllerLease::acquire(&directory, format!("api-{action}:{run_id}"))
        .map_err(harness_error)?;
    let active = load_review_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    tokio::spawn(async move {
        let transport = match transport(&state) {
            Ok(transport) => transport,
            Err(error) => {
                tracing::error!(
                    project_id,
                    run_id,
                    ?error,
                    "film take mutation transport failed"
                );
                return;
            }
        };
        let mut options = ResumeOptions::new(directory);
        options.poll_interval = Duration::from_secs(2);
        options.export = false;
        let reason = payload.reason.trim();
        let result = if repair {
            review::request_repair_with_lease(&transport, &options, &payload.shot_id, reason, lease)
                .await
        } else {
            crate::film_harness::replace_take_with_lease(
                &transport,
                &options,
                &payload.shot_id,
                reason,
                lease,
            )
            .await
        };
        if let Err(error) = result {
            tracing::error!(project_id, run_id, %error, action, "film take mutation stopped");
        }
    });
    Ok((StatusCode::ACCEPTED, Json(active)))
}

fn transport(state: &AppState) -> Result<HttpTransport, ApiError> {
    HttpTransport::new(
        &state.settings.mcp_api_url,
        Some(state.settings.access_token.clone()),
    )
    .map_err(|error| ApiError::internal(error.to_string()))
}

async fn run_directory(
    state: AppState,
    project_id: &str,
    run_id: &str,
) -> Result<std::path::PathBuf, ApiError> {
    let project_id = project_id.to_owned();
    let run_id = run_id.to_owned();
    project_call(state, move |store| {
        Ok(store.film_run_files(&project_id, &run_id)?.directory)
    })
    .await
}

fn ensure_mutation_available(view: &FilmReviewView) -> Result<(), ApiError> {
    match view.action_disabled_reason.as_deref() {
        Some(reason) => Err(ApiError::conflict(reason)),
        None => Ok(()),
    }
}

fn ensure_saved_cut_selection(view: &FilmReviewView, shot_id: &str) -> Result<(), ApiError> {
    let selection = view
        .selections
        .iter()
        .find(|selection| selection.shot_id == shot_id)
        .ok_or_else(|| ApiError::bad_request(format!("Run has no shot {shot_id}")))?;
    if selection.state == "aligned" {
        Ok(())
    } else {
        Err(ApiError::conflict(format!(
            "Shot {shot_id} is {}. Use a recorded take in the saved cut before accepting or rejecting it.",
            selection.state.replace('_', " ")
        )))
    }
}

fn harness_error(error: HarnessError) -> ApiError {
    match error {
        HarnessError::Validation(findings) => ApiError::bad_request(
            findings
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; "),
        ),
        HarnessError::Refused(detail) => ApiError::conflict(detail),
        other => ApiError::internal(other.to_string()),
    }
}

pub(crate) async fn load_review_view(
    state: AppState,
    project_id: String,
    run_id: String,
) -> Result<FilmReviewView, ApiError> {
    let run = load_run_view(state.clone(), project_id.clone(), run_id.clone()).await?;
    let record = run.record.as_ref().ok_or_else(|| {
        ApiError::conflict("Film run has not started; render a take before review")
    })?;
    let asset_ids = record
        .shots
        .iter()
        .flat_map(|shot| shot.attempts.iter())
        .filter_map(|attempt| attempt.take.as_ref().map(|take| take.asset_id.clone()))
        .collect::<std::collections::BTreeSet<_>>();
    let timeline_id = record
        .timeline
        .as_ref()
        .map(|timeline| timeline.timeline_id.clone());
    let review_name = format!("film-harness review ({run_id})");
    let lookup_project = project_id.clone();
    let lookup_run = run_id.clone();
    let (directory, take_assets, review_timeline_id, saved_timeline) =
        project_call(state, move |store| {
            let directory = store
                .film_run_files(&lookup_project, &lookup_run)?
                .directory;
            let take_assets = asset_ids
                .iter()
                .filter_map(
                    |asset_id| match store.get_asset(&lookup_project, asset_id) {
                        Ok(asset) => Some(Ok(asset)),
                        Err(ProjectStoreError::NotFound(_)) => None,
                        Err(error) => Some(Err(error)),
                    },
                )
                .collect::<Result<Vec<_>, _>>()?;
            let review_timeline_id = store
                .list_timelines(&lookup_project)?
                .into_iter()
                .find(|timeline| timeline.name == review_name)
                .map(|timeline| timeline.id);
            let saved_timeline = timeline_id
                .as_deref()
                .map(|id| store.get_timeline(&lookup_project, id))
                .transpose()?;
            Ok((directory, take_assets, review_timeline_id, saved_timeline))
        })
        .await?;
    let observations = read_observations(&directory, record)?;
    let selections = record
        .shots
        .iter()
        .map(|shot| take_selection(record, shot, saved_timeline.as_ref()))
        .collect();
    let action_disabled_reason = mutation_disabled_reason(&run);
    Ok(FilmReviewView {
        run,
        take_assets,
        observations,
        review_timeline_id,
        selections,
        action_disabled_reason,
        assistive_notice: ASSISTIVE_NOTICE,
    })
}

fn mutation_disabled_reason(run: &FilmRunView) -> Option<String> {
    if run.controller_active {
        return Some(format!(
            "{} is active. Wait for it to stop or cancel it before changing takes.",
            run.controller_owner
                .as_deref()
                .unwrap_or("A film operation")
        ));
    }
    run.record.as_ref().and_then(|record| {
        record.shots.iter().find_map(|shot| {
            shot.attempts.iter().find_map(|attempt| {
                (!TERMINAL_ATTEMPT_STATUSES.contains(&attempt.status.as_str())).then(|| {
                    format!(
                        "Shot {} attempt {} is still {}. Resume the run to settle it before changing takes.",
                        shot.shot_id, attempt.attempt, attempt.status
                    )
                })
            })
        })
    })
}

fn read_observations(
    directory: &FsPath,
    record: &RunRecord,
) -> Result<Vec<ObservedState>, ApiError> {
    let mut observations = Vec::new();
    for summary in record.shots.iter().flat_map(|shot| shot.reviews.iter()) {
        let relative = FsPath::new(&summary.record_path);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            || relative
                .components()
                .next()
                .and_then(|component| match component {
                    Component::Normal(name) => name.to_str(),
                    _ => None,
                })
                != Some(review::REVIEWS_DIR)
        {
            return Err(ApiError::internal(format!(
                "Review record path {:?} is not contained in the run",
                summary.record_path
            )));
        }
        observations.push(
            review::read_observed_state(&directory.join(relative))
                .map_err(|error| ApiError::internal(error.to_string()))?,
        );
    }
    Ok(observations)
}

fn take_selection(
    record: &RunRecord,
    shot: &sceneworks_core::film_plan::ShotRunRecord,
    saved_timeline: Option<&Value>,
) -> FilmTakeSelection {
    let (timeline_asset_id, trim_conflict) =
        saved_cut_selection(&record.run_id, &shot.shot_id, saved_timeline);
    let timeline_attempt = timeline_asset_id.as_deref().and_then(|asset_id| {
        shot.attempts.iter().find_map(|attempt| {
            attempt
                .take
                .as_ref()
                .filter(|take| take.asset_id == asset_id)
                .map(|_| attempt.attempt)
        })
    });
    let record_selected_asset_id = shot
        .selected()
        .and_then(|attempt| attempt.take.as_ref())
        .map(|take| take.asset_id.clone());
    let state = if trim_conflict.is_some() {
        "trim_conflict"
    } else if timeline_asset_id.is_none() {
        "not_in_saved_cut"
    } else if record_selected_asset_id == timeline_asset_id {
        "aligned"
    } else if shot.selected_attempt.is_none()
        && timeline_attempt.is_some_and(|attempt| {
            shot.attempts
                .iter()
                .find(|candidate| candidate.attempt == attempt)
                .and_then(|candidate| candidate.rejection.as_ref())
                .is_some()
        })
    {
        "rejected_take_in_saved_cut"
    } else if timeline_attempt.is_none() {
        "external_asset_in_saved_cut"
    } else {
        "saved_cut_differs"
    };
    FilmTakeSelection {
        shot_id: shot.shot_id.clone(),
        record_selected_attempt: shot.selected_attempt,
        record_selected_asset_id,
        timeline_attempt,
        timeline_asset_id,
        state: state.to_owned(),
        trim_conflict,
    }
}

fn saved_cut_selection(
    run_id: &str,
    shot_id: &str,
    saved_timeline: Option<&Value>,
) -> (Option<String>, Option<Value>) {
    let timeline_item = saved_timeline.and_then(|timeline| {
        timeline
            .get("tracks")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .flat_map(|track| {
                track
                    .get("items")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .find(|item| {
                item.pointer("/filmHarness/runId").and_then(Value::as_str) == Some(run_id)
                    && item.pointer("/filmHarness/shotId").and_then(Value::as_str) == Some(shot_id)
            })
    });
    let timeline_asset_id = timeline_item
        .and_then(|item| item.get("assetId"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let trim_conflict = saved_timeline
        .and_then(|timeline| {
            timeline
                .pointer(&format!(
                    "/filmAssembly/runs/{}/trimConflicts/{}",
                    escape_pointer(run_id),
                    escape_pointer(shot_id)
                ))
                .cloned()
        })
        .filter(|value| !value.is_null());
    (timeline_asset_id, trim_conflict)
}

fn escape_pointer(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_trim_conflict_keeps_saved_asset_as_current_selection() {
        let timeline = json!({
            "tracks": [{"items": [{
                "assetId": "asset-old",
                "filmHarness": {"runId": "run/1", "shotId": "SH~010"}
            }]}],
            "filmAssembly": {"runs": {"run/1": {"trimConflicts": {"SH~010": {
                "currentAssetId": "asset-old",
                "incomingAssetId": "asset-new"
            }}}}}
        });

        let (current, conflict) = saved_cut_selection("run/1", "SH~010", Some(&timeline));
        assert_eq!(current.as_deref(), Some("asset-old"));
        assert_eq!(conflict.unwrap()["incomingAssetId"], "asset-new");
    }
}
