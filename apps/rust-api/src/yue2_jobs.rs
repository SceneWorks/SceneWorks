//! YuE2 song jobs (sc-22999, epic 22988): the submission route, execution-time eligibility, the
//! server-side licence acknowledgment and the terminal side effects that put a finished run into
//! the project library.
//!
//! # Submission — `POST /api/v1/projects/:project_id/yue2/jobs`
//!
//! The body is [`Yue2JobSpec`]. The route
//!
//! 1. validates it with `yue2_score::jobs::validate_request` (typed 400 `yue2_invalid_combination`
//!    / `yue2_invalid_value` / `yue2_missing_field`, 422 `yue2_unsupported_notation`);
//! 2. refuses recording transcription with the catalog's recorded block (403 `component_blocked`,
//!    reason + unblock condition) — the SheetSage2 / MERT port is gated on an owner licensing
//!    decision and linked into no bundle;
//! 3. checks eligibility ([`yue2_eligibility`]): a declared commercial use is refused with the
//!    verdict's pointer to YuE1 (403 `commercial_use_refused`), and the licence must be
//!    acknowledged — asserted now (and recorded) or recorded earlier for the current terms (403
//!    `license_acknowledgment_required`);
//! 4. resolves the kind's inputs from the project (a plan or song job's published run, a score
//!    version and its digests) and refuses what is missing, incomplete or foreign (404 / 409
//!    `yue2_source_unavailable`);
//! 5. queues one `audio_generate` job per take (`count`), each with its own run id and seed
//!    `seed + i`. The server claims at most one active job per GPU, so a batch renders serially on
//!    the admitted GPU.
//!
//! The stored payload carries the validated `yue2` block (envelope cleared, server fields set), the
//! manifest entry, the usage policy granted at submission and the declared `commercialUse`.
//!
//! # Execution — `GET /api/v1/jobs/:job_id/yue2-eligibility`
//!
//! The worker asks this before it resolves any weight: the same eligibility, re-evaluated against
//! the catalog and the acknowledgment store **as they are now**, not as they were at submission.
//! A withdrawn acknowledgment, changed licence terms or a commercial declaration refuses the job.
//!
//! # Terminal side effects ([`apply_yue2_side_effects`])
//!
//! After the worker's audio asset is persisted: a plan (or a planned song) becomes a sc-22997 score
//! version (origin `plan`, source = the job); a score-version render — completed or failed — is
//! recorded with `record_yue2_render` (score / request digests, truncation, model, decoder, job,
//! audio asset, effective settings, provenance). Both are idempotent per job, so the recovery sweep
//! can re-run them.

use super::*;

use crate::models::{
    model_catalog_snapshot, COMPONENT_BLOCKED_CODE, LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE,
};

use sceneworks_core::license_acknowledgments::{self as acks, LicenseAcknowledgment};
use sceneworks_core::model_usage_policy::{
    commercial_use_verdict, conditional_component_downloads, CommercialUseError,
    CommercialUseVerdict, ConditionalComponentsError,
};
use sceneworks_core::yue2_score::jobs::{
    self as yue2, BatchMember, RunSource, Sources, VersionSource, Yue2JobError, Yue2JobKind,
    Yue2JobSpec,
};
use sceneworks_core::yue2_score::store::{
    Actor, Channel, ComponentIdentity, CreateVersionInput, Provenance, RenderInput, RenderStatus,
    SourceKind, SourceReference, Truncation, VersionOrigin,
};

/// A commercial-use declaration refused for a noncommercial model.
pub(crate) const COMMERCIAL_USE_REFUSED_CODE: &str = "commercial_use_refused";
/// A plan / run / version a job needs is missing, incomplete or not this project's.
pub(crate) const YUE2_SOURCE_UNAVAILABLE_CODE: &str = "yue2_source_unavailable";
/// Transcription is not blocked by the catalog but no linked bundle can run it.
pub(crate) const YUE2_TRANSCRIPTION_UNAVAILABLE_CODE: &str = "yue2_transcription_unavailable";
/// A symbolic-song model sent to the generic audio route.
pub(crate) const YUE2_SONG_ROUTE_REQUIRED_CODE: &str = "yue2_song_route_required";
/// The route a symbolic-song model is submitted through.
pub(crate) const YUE2_JOBS_ROUTE: &str = "/api/v1/projects/:project_id/yue2/jobs";

/// Serializes read-modify-write of the acknowledgment file within this process (the API is its
/// only writer).
static ACK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn spec_error(error: Yue2JobError) -> ApiError {
    let status = if error.code == crate::yue2_scores::YUE2_UNSUPPORTED_NOTATION {
        StatusCode::UNPROCESSABLE_ENTITY
    } else {
        StatusCode::BAD_REQUEST
    };
    ApiError::typed(
        status,
        format!("YuE2 job: {error}"),
        error.code,
        json!({ "field": error.field }),
    )
}

fn source_unavailable(detail: impl Into<String>) -> ApiError {
    ApiError::typed(
        StatusCode::CONFLICT,
        detail,
        YUE2_SOURCE_UNAVAILABLE_CODE,
        json!({}),
    )
}

fn ack_error(error: acks::AcknowledgmentError) -> ApiError {
    ApiError::internal(error.to_string())
}

// ---------------------------------------------------------------------------------------------
// Licence acknowledgments.
// ---------------------------------------------------------------------------------------------

async fn ack_call<T, F>(state: &AppState, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(&std::path::Path) -> Result<T, acks::AcknowledgmentError> + Send + 'static,
{
    let path = acks::acknowledgments_file(&state.settings.config_dir);
    tokio::task::spawn_blocking(move || {
        let _guard = ACK_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        operation(&path)
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))?
    .map_err(ack_error)
}

/// Record that the user accepted `entry`'s current licence terms.
pub(crate) async fn record_license_acknowledgment(
    state: &AppState,
    entry: &Value,
    channel: &'static str,
) -> Result<LicenseAcknowledgment, ApiError> {
    let entry = entry.clone();
    let at = now_rfc3339();
    ack_call(state, move |path| {
        acks::record_acknowledgment(path, &entry, channel, &at)
    })
    .await
}

async fn current_license_acknowledgment(
    state: &AppState,
    entry: &Value,
) -> Result<Option<LicenseAcknowledgment>, ApiError> {
    let entry = entry.clone();
    ack_call(state, move |path| {
        acks::current_acknowledgment(path, &entry)
    })
    .await
}

async fn catalog_entry(state: &AppState, model_id: &str) -> Result<Value, ApiError> {
    model_catalog_snapshot(state)
        .await?
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
        .cloned()
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: format!("Model '{model_id}' is not in this catalog."),
            context: None,
            code: None,
        })
}

fn requires_license_acknowledgment(entry: &Value) -> bool {
    entry
        .get("requiresLicenseAcknowledgment")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn acknowledgment_view(entry: &Value, ack: Option<&LicenseAcknowledgment>) -> Value {
    json!({
        "modelId": entry.get("id").cloned().unwrap_or(Value::Null),
        "required": requires_license_acknowledgment(entry),
        "termsSha256": acks::terms_sha256(entry),
        "acknowledged": ack.is_some(),
        "acknowledgment": ack,
    })
}

/// `GET /api/v1/models/:model_id/license-acknowledgment` — the current acknowledgment, if any.
pub(crate) async fn get_license_acknowledgment(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let entry = catalog_entry(&state, &model_id).await?;
    let ack = current_license_acknowledgment(&state, &entry).await?;
    Ok(Json(acknowledgment_view(&entry, ack.as_ref())))
}

/// `PUT /api/v1/models/:model_id/license-acknowledgment` — the user accepts the current terms.
pub(crate) async fn put_license_acknowledgment(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let entry = catalog_entry(&state, &model_id).await?;
    let ack = record_license_acknowledgment(&state, &entry, "api").await?;
    Ok(Json(acknowledgment_view(&entry, Some(&ack))))
}

/// `DELETE /api/v1/models/:model_id/license-acknowledgment` — withdraw it. Queued jobs that
/// depended on it are refused when they reach execution.
pub(crate) async fn delete_license_acknowledgment(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let entry = catalog_entry(&state, &model_id).await?;
    let id = model_id.clone();
    let removed = ack_call(&state, move |path| acks::revoke_acknowledgment(path, &id)).await?;
    let mut view = acknowledgment_view(&entry, None);
    view["revoked"] = json!(removed);
    Ok(Json(view))
}

// ---------------------------------------------------------------------------------------------
// Eligibility.
// ---------------------------------------------------------------------------------------------

/// The usage policy a job runs under, persisted into every output's provenance and carried by
/// exports: the model's experimental / noncommercial declaration, its licence terms, the refusal a
/// commercial route gives it (with the pointer to its commercially eligible alternative) and the
/// acknowledgment the run relied on.
pub(crate) fn usage_policy(
    entry: &Value,
    verdict: &CommercialUseVerdict,
    ack: Option<&LicenseAcknowledgment>,
) -> Value {
    let flag = |key: &str| entry.get(key).and_then(Value::as_bool).unwrap_or(false);
    json!({
        "schema": "sceneworks.usagePolicy.v1",
        "modelId": entry.get("id").cloned().unwrap_or(Value::Null),
        "experimental": flag("experimental"),
        "nonCommercial": flag("nonCommercial"),
        "intendedUse": if matches!(verdict, CommercialUseVerdict::Refused { .. }) {
            "noncommercial"
        } else {
            "unrestricted"
        },
        "license": {
            "license": entry.get("license").cloned().unwrap_or(Value::Null),
            "url": entry.get("licenseUrl").cloned().unwrap_or(Value::Null),
            "notice": entry.get("licenseNotice").cloned().unwrap_or(Value::Null),
            "termsSha256": acks::terms_sha256(entry),
        },
        "commercialUse": verdict,
        "licenseAcknowledgment": ack,
    })
}

/// Whether `model_id` may run now for a job that declares (or not) a commercial use, and the
/// policy it runs under. `assert_ack` records the caller's acknowledgment first (submission only).
pub(crate) async fn yue2_eligibility(
    state: &AppState,
    model_id: &str,
    commercial_use: bool,
    assert_ack: bool,
) -> Result<Value, ApiError> {
    let catalog = model_catalog_snapshot(state).await?;
    let verdict = commercial_use_verdict(&catalog, model_id).map_err(|error| match error {
        CommercialUseError::UnknownModel(_) => ApiError {
            status: StatusCode::NOT_FOUND,
            detail: format!("Model '{model_id}' is not in this catalog."),
            context: None,
            code: None,
        },
        CommercialUseError::AmbiguousModel(_) => ApiError::conflict(error.to_string()),
    })?;
    let entry = catalog
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
        .cloned()
        .unwrap_or_else(|| json!({}));
    if commercial_use {
        if let CommercialUseVerdict::Refused {
            reason,
            alternatives,
            note,
            ..
        } = &verdict
        {
            let pointer = if alternatives.is_empty() {
                String::new()
            } else {
                format!(" Use {} instead.", alternatives.join(", "))
            };
            return Err(ApiError::typed(
                StatusCode::FORBIDDEN,
                format!(
                    "Model '{model_id}' may not be used for a commercial purpose: {reason}.{pointer}{}",
                    note.as_deref().map(|n| format!(" {n}")).unwrap_or_default()
                ),
                COMMERCIAL_USE_REFUSED_CODE,
                serde_json::to_value(&verdict).unwrap_or(Value::Null),
            ));
        }
    }
    let ack = if requires_license_acknowledgment(&entry) {
        if assert_ack {
            record_license_acknowledgment(state, &entry, "yue2_job").await?;
        }
        match current_license_acknowledgment(state, &entry).await? {
            Some(ack) => Some(ack),
            None => {
                return Err(ApiError::typed(
                    StatusCode::FORBIDDEN,
                    format!(
                        "Model '{model_id}' requires accepting its license (current terms) before \
                         it runs. Accept it on the Models screen, or send \
                         `licenseAcknowledged: true`."
                    ),
                    LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE,
                    json!({ "modelId": model_id, "termsSha256": acks::terms_sha256(&entry) }),
                ));
            }
        }
    } else {
        None
    };
    Ok(usage_policy(&entry, &verdict, ack.as_ref()))
}

/// `GET /api/v1/jobs/:job_id/yue2-eligibility` — the worker's execution-time check for a queued
/// YuE2 job, evaluated against the catalog and acknowledgments as they are now.
pub(crate) async fn get_job_yue2_eligibility(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let job = store_call(state.clone(), move |store, _| store.get_job(&job_id)).await?;
    if job.job_type != JobType::AudioGenerate
        || job
            .payload
            .get(yue2::PAYLOAD_KEY)
            .is_none_or(Value::is_null)
    {
        return Err(ApiError::bad_request(format!(
            "Job {} is not a YuE2 job.",
            job.id
        )));
    }
    // The worker always loads the `yue2` provider for a job with a `yue2` block, so eligibility
    // is always YuE2's — never whatever `model` the payload names (a replay could have changed it).
    let model_id = yue2::MODEL_ID.to_owned();
    let commercial = job
        .payload
        .get("commercialUse")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let policy = yue2_eligibility(&state, &model_id, commercial, false).await?;
    Ok(Json(json!({ "eligible": true, "usagePolicy": policy })))
}

/// The refusal for recording transcription: the catalog's recorded block when there is one.
fn transcription_refusal(entry: &Value) -> ApiError {
    match conditional_component_downloads(entry, "cover") {
        Err(ConditionalComponentsError::Blocked { blocked, .. }) => {
            let reasons: Vec<Value> = blocked
                .iter()
                .map(|(component_id, reason, unblock)| {
                    json!({"componentId": component_id, "reason": reason, "unblock": unblock})
                })
                .collect();
            ApiError::typed(
                StatusCode::FORBIDDEN,
                format!(
                    "YuE2 recording transcription is blocked: {}. Start a cover from a reviewed \
                     score (cover.score or cover.versionId) instead.",
                    blocked
                        .iter()
                        .map(|(_, reason, _)| reason.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
                COMPONENT_BLOCKED_CODE,
                json!({ "purpose": "cover", "blocked": reasons }),
            )
        }
        Err(error) => ApiError::typed(
            StatusCode::NOT_IMPLEMENTED,
            format!("YuE2 recording transcription is unavailable: {error}"),
            YUE2_TRANSCRIPTION_UNAVAILABLE_CODE,
            json!({ "purpose": "cover" }),
        ),
        Ok(_) => ApiError::typed(
            StatusCode::NOT_IMPLEMENTED,
            "YuE2 recording transcription is not linked into this build's runtime; start a cover \
             from a reviewed score instead.",
            YUE2_TRANSCRIPTION_UNAVAILABLE_CODE,
            json!({ "purpose": "cover" }),
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// Submission.
// ---------------------------------------------------------------------------------------------

fn random_seed() -> u64 {
    (Uuid::new_v4().as_u128() as u64) & ((1u64 << 63) - 1)
}

/// A completed YuE2 job of this project and the run it published.
async fn completed_run(
    state: &AppState,
    project_id: &str,
    job_id: &str,
    field: &str,
    accept: &[&str],
) -> Result<RunSource, ApiError> {
    let id = job_id.to_owned();
    let job = store_call(state.clone(), move |store, _| store.get_job(&id))
        .await
        .map_err(|_| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: format!("{field}: job {job_id} does not exist."),
            context: None,
            code: Some(YUE2_SOURCE_UNAVAILABLE_CODE),
        })?;
    if job.project_id.as_deref() != Some(project_id) {
        return Err(source_unavailable(format!(
            "{field}: job {job_id} belongs to another project."
        )));
    }
    if job.payload.get(yue2::PAYLOAD_KEY).is_none() {
        return Err(source_unavailable(format!(
            "{field}: job {job_id} is not a YuE2 job."
        )));
    }
    if job.status != JobStatus::Completed {
        return Err(source_unavailable(format!(
            "{field}: job {job_id} has not completed ({:?}).",
            job.status
        )));
    }
    let run = job
        .result
        .get("yue2")
        .and_then(|yue2| yue2.get("run"))
        .ok_or_else(|| {
            source_unavailable(format!("{field}: job {job_id} published no YuE2 run."))
        })?;
    let text = |key: &str| run.get(key).and_then(Value::as_str).map(str::to_owned);
    let kind = text("kind").unwrap_or_default();
    if !accept.contains(&kind.as_str()) {
        return Err(source_unavailable(format!(
            "{field}: job {job_id} published a `{kind}` run; this job needs one of {accept:?}."
        )));
    }
    let (Some(run_dir), Some(identity)) = (text("dir"), text("identity")) else {
        return Err(source_unavailable(format!(
            "{field}: job {job_id}'s run record is incomplete."
        )));
    };
    Ok(RunSource {
        job_id: job_id.to_owned(),
        run_dir,
        identity,
        plan_identity: text("planIdentity"),
    })
}

async fn version_source(
    state: &AppState,
    project_id: &str,
    version_id: &str,
) -> Result<(VersionSource, String), ApiError> {
    let (project, version) = (project_id.to_owned(), version_id.to_owned());
    let record = crate::yue2_scores::yue2_call(state.clone(), move |store| {
        store
            .yue2_score_store(&project)?
            .get_version_record(&version)
    })
    .await?;
    Ok((
        VersionSource {
            id: record.id,
            score_sha256: record.score.sha256,
            request_sha256: record.request_sha256,
        },
        record.score.abc,
    ))
}

async fn resolve_sources(
    state: &AppState,
    project_id: &str,
    spec: &mut Yue2JobSpec,
) -> Result<Sources, ApiError> {
    let mut sources = Sources::default();
    match spec.kind {
        Yue2JobKind::FromPlan => {
            let job_id = spec.plan_job_id.clone().unwrap_or_default();
            // Every published run holds its exact plan: a plan-only run, or any song run.
            sources.plan = Some(
                completed_run(
                    state,
                    project_id,
                    &job_id,
                    "planJobId",
                    &["plan", "song", "cached_decode"],
                )
                .await?,
            );
            if sources
                .plan
                .as_ref()
                .is_some_and(|plan| plan.plan_identity.is_none())
            {
                return Err(source_unavailable(format!(
                    "planJobId: job {job_id} recorded no plan identity."
                )));
            }
        }
        Yue2JobKind::Decode => {
            let job_id = spec.source_job_id.clone().unwrap_or_default();
            sources.source_run = Some(
                completed_run(
                    state,
                    project_id,
                    &job_id,
                    "sourceJobId",
                    &["song", "cached_decode"],
                )
                .await?,
            );
        }
        Yue2JobKind::RenderVersion => {
            let version_id = spec.version_id.clone().unwrap_or_default();
            sources.version = Some(version_source(state, project_id, &version_id).await?.0);
        }
        Yue2JobKind::Cover => {
            if let Some(version_id) = spec.cover.as_ref().and_then(|c| c.version_id.clone()) {
                let (source, abc) = version_source(state, project_id, &version_id).await?;
                if let Some(cover) = spec.cover.as_mut() {
                    cover.score = Some(abc);
                }
                sources.cover_version = Some(source);
            }
        }
        _ => {}
    }
    Ok(sources)
}

/// `POST /api/v1/projects/:project_id/yue2/jobs` — see the [module docs](self).
pub(crate) async fn create_yue2_jobs(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(mut spec): ApiJson<Yue2JobSpec>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    yue2::validate_request(&spec).map_err(spec_error)?;
    let project = project_call(state.clone(), {
        let project_id = project_id.clone();
        move |store| store.get_project(&project_id)
    })
    .await?;
    let entry = resolve_model_manifest_entry(&state, yue2::MODEL_ID).await?;
    if entry.get("id").and_then(Value::as_str) != Some(yue2::MODEL_ID) {
        return Err(ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "YuE2 is not in this catalog.".to_owned(),
            context: None,
            code: None,
        });
    }
    if yue2::requests_transcription(&spec) {
        return Err(transcription_refusal(&entry));
    }
    let commercial_use = spec.commercial_use;
    let usage_policy = yue2_eligibility(
        &state,
        yue2::MODEL_ID,
        commercial_use,
        spec.license_acknowledged,
    )
    .await?;
    let sources = resolve_sources(&state, &project_id, &mut spec).await?;

    let count = spec.count.unwrap_or(1);
    let base_seed = matches!(
        spec.kind,
        Yue2JobKind::Create | Yue2JobKind::Plan | Yue2JobKind::Cover
    )
    .then(|| spec.seed.unwrap_or_else(random_seed));
    let batch_id = (count > 1).then(|| format!("yue2batch_{}", Uuid::new_v4().simple()));
    let requested_gpu = requested_gpu_or_auto(spec.requested_gpu.take().unwrap_or_default());
    let project_name = spec.project_name.take().or(Some(project.name));
    spec.count = None;
    spec.commercial_use = false;
    spec.license_acknowledged = false;
    let has_sources = sources != Sources::default();

    let mut jobs = Vec::with_capacity(count as usize);
    for index in 0..count {
        let mut take = spec.clone();
        take.seed = base_seed.map(|seed| (seed + u64::from(index)) & ((1u64 << 63) - 1));
        take.run_id = Some(format!(
            "{}{}",
            yue2::RUN_ID_PREFIX,
            Uuid::new_v4().simple()
        ));
        take.batch = batch_id.as_ref().map(|id| BatchMember {
            id: id.clone(),
            index,
            count,
        });
        take.sources = has_sources.then(|| sources.clone());
        // The stored block is exactly what the worker will accept.
        yue2::validate_for_execution(&take).map_err(spec_error)?;
        let mut payload = JsonObject::new();
        payload.insert("projectId".to_owned(), json!(project_id));
        payload.insert("projectName".to_owned(), json!(project_name));
        payload.insert("model".to_owned(), json!(yue2::MODEL_ID));
        payload.insert("modelManifestEntry".to_owned(), entry.clone());
        payload.insert(
            "prompt".to_owned(),
            json!(take.style.clone().unwrap_or_default()),
        );
        payload.insert(
            yue2::PAYLOAD_KEY.to_owned(),
            serde_json::to_value(&take).map_err(|e| ApiError::internal(e.to_string()))?,
        );
        payload.insert("usagePolicy".to_owned(), usage_policy.clone());
        payload.insert("commercialUse".to_owned(), json!(commercial_use));
        let job = create_generation_job(
            state.clone(),
            JobType::AudioGenerate,
            Some(project_id.clone()),
            project_name.clone(),
            payload,
            requested_gpu.clone(),
        )
        .await?;
        jobs.push(public_job_snapshot(job));
    }
    Ok((
        StatusCode::CREATED,
        Json(json!({ "jobs": jobs, "batchId": batch_id })),
    ))
}

/// Canonicalize a replayed (retried / duplicated) `audio_generate` payload: `persisted` is the
/// original job's payload, `merged` the payload the replay will enqueue.
///
/// * A `yue2` block can be neither added (a generic audio job must not become a YuE2 job that
///   skipped the YuE2 route's eligibility and source resolution) nor removed.
/// * A YuE2 replay keeps `model = yue2` (the worker always loads the `yue2` provider, and
///   eligibility is always evaluated for it), its block must still be one the worker will run,
///   its manifest entry is re-resolved from the catalog, its submission-time usage policy is kept
///   and a commercial declaration can only be added, never withdrawn.
/// * A replay without a block must not name a symbolic-song model: the generic audio route
///   refuses those, and so does its replay.
pub(crate) async fn canonicalize_replayed_audio_payload(
    state: &AppState,
    persisted: &JsonObject,
    merged: &mut JsonObject,
) -> Result<(), ApiError> {
    let block_of = |payload: &JsonObject| {
        payload
            .get(yue2::PAYLOAD_KEY)
            .filter(|value| !value.is_null())
            .cloned()
    };
    let refused = |field: &str, detail: String| {
        ApiError::typed(
            StatusCode::BAD_REQUEST,
            format!("YuE2 job: {detail}"),
            yue2::INVALID_COMBINATION,
            json!({ "field": field }),
        )
    };
    match (block_of(persisted), block_of(merged)) {
        (None, Some(_)) => Err(refused(
            "yue2",
            "a yue2 block cannot be added to a job that was not submitted through the YuE2 \
             route"
                .into(),
        )),
        (Some(_), None) => Err(refused(
            "yue2",
            "a YuE2 job's yue2 block cannot be removed".into(),
        )),
        (Some(_), Some(block)) => {
            let model = merged.get("model").and_then(Value::as_str).unwrap_or("");
            if model != yue2::MODEL_ID {
                return Err(refused(
                    "model",
                    format!("a YuE2 job always runs `{}`, not `{model}`", yue2::MODEL_ID),
                ));
            }
            let spec: Yue2JobSpec = serde_json::from_value(block).map_err(|error| {
                ApiError::typed(
                    StatusCode::BAD_REQUEST,
                    format!("YuE2 job: the yue2 block is malformed: {error}"),
                    yue2::INVALID_VALUE,
                    json!({ "field": "yue2" }),
                )
            })?;
            yue2::validate_for_execution(&spec).map_err(spec_error)?;
            let entry = resolve_model_manifest_entry(state, yue2::MODEL_ID).await?;
            merged.insert("modelManifestEntry".to_owned(), entry);
            match persisted.get("usagePolicy") {
                Some(policy) => merged.insert("usagePolicy".to_owned(), policy.clone()),
                None => merged.remove("usagePolicy"),
            };
            let declared = |payload: &JsonObject| {
                payload
                    .get("commercialUse")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            };
            let commercial = declared(persisted) || declared(merged);
            merged.insert("commercialUse".to_owned(), json!(commercial));
            Ok(())
        }
        (None, None) => {
            let model = merged
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let entry = resolve_model_manifest_entry(state, &model).await?;
            refuse_symbolic_song_on_audio_route(&model, &entry)?;
            if let Some(stored) = merged.get("modelManifestEntry") {
                refuse_symbolic_song_on_audio_route(&model, stored)?;
            }
            Ok(())
        }
    }
}

/// The generic audio route refuses a symbolic-song model: its submission, eligibility and source
/// resolution live on [`YUE2_JOBS_ROUTE`], and a job that skipped them must not queue.
pub(crate) fn refuse_symbolic_song_on_audio_route(
    model_id: &str,
    entry: &Value,
) -> Result<(), ApiError> {
    let symbolic = entry
        .get("audio")
        .and_then(|audio| audio.get("supportsSymbolicSong"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if symbolic {
        return Err(ApiError::typed(
            StatusCode::BAD_REQUEST,
            format!(
                "Model {model_id} is a symbolic-plan song model; submit it through \
                 POST {YUE2_JOBS_ROUTE}."
            ),
            YUE2_SONG_ROUTE_REQUIRED_CODE,
            json!({ "route": YUE2_JOBS_ROUTE }),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Terminal side effects.
// ---------------------------------------------------------------------------------------------

fn job_provenance(job_id: &str) -> Provenance {
    Provenance {
        actor: Actor::User,
        agent_name: None,
        channel: Channel::Worker,
        source: Some(SourceReference {
            kind: SourceKind::Job,
            id: Some(job_id.to_owned()),
        }),
    }
}

fn yue2_block(job: &JobSnapshot) -> Option<Yue2JobSpec> {
    job.payload
        .get(yue2::PAYLOAD_KEY)
        .and_then(|value| serde_json::from_value(value.clone()).ok())
}

/// The model identity a render record names: the pinned primary download of the catalog entry.
fn model_identity(entry: &Value) -> ComponentIdentity {
    let primary = entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| row.get("coRequisite").and_then(Value::as_bool) != Some(true))
        });
    ComponentIdentity {
        id: primary
            .and_then(|row| row.get("repo"))
            .and_then(Value::as_str)
            .unwrap_or(yue2::MODEL_ID)
            .to_owned(),
        revision: primary
            .and_then(|row| row.get("revision"))
            .and_then(Value::as_str)
            .map(str::to_owned),
    }
}

/// A component identity the worker recorded under `field`: `Ok(None)` when it recorded none (a job
/// that failed before it resolved its weights), an error when it recorded one that is not a
/// `{id, revision}` identity — never silently replaced by another identity.
fn identity_from(block: &JsonObject, field: &str) -> Result<Option<ComponentIdentity>, ApiError> {
    match block.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => serde_json::from_value(value.clone())
            .map(Some)
            .map_err(|error| {
                ApiError::internal(format!(
                "the worker's `{field}` identity is not a component identity ({error}): {value}"
            ))
            }),
    }
}

/// The truncation the worker OBSERVED: `Ok(None)` when it recorded none (the run never
/// published), an error when the record is malformed. Never a default.
fn observed_truncation(block: &JsonObject) -> Result<Option<Truncation>, ApiError> {
    match block.get("truncated") {
        None | Some(Value::Null) => Ok(None),
        Some(value) => serde_json::from_value(value.clone())
            .map(Some)
            .map_err(|error| {
                ApiError::internal(format!(
                    "the worker's truncation record is malformed ({error}): {value}"
                ))
            }),
    }
}

/// Apply the YuE2 terminal side effects of `job` into `result` (see the [module docs](self)).
/// No-op for any other job. A failure is recorded in `result.yue2.sideEffectErrors` rather than
/// failing the terminal handoff, so an unparseable truncated plan never wedges the recovery loop.
pub(crate) async fn apply_yue2_side_effects(
    state: &AppState,
    job: &JobSnapshot,
    result: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(spec) = yue2_block(job) else {
        return Ok(());
    };
    let Some(project_id) = job.project_id.clone() else {
        return Ok(());
    };
    let mut block = result
        .get("yue2")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut errors: Vec<String> = Vec::new();
    match job.status {
        JobStatus::Completed => {
            if spec.kind == Yue2JobKind::RenderVersion {
                if block.get("renderRecordId").is_none() {
                    match record_render(state, job, &spec, &project_id, &block, result).await {
                        Ok(id) => {
                            block.insert("renderRecordId".to_owned(), json!(id));
                        }
                        Err(error) => errors.push(format!("render record: {}", error.detail)),
                    }
                }
            } else if let Some(abc) = block
                .get("score")
                .and_then(|score| score.get("abc"))
                .and_then(Value::as_str)
                .map(str::to_owned)
            {
                let abc_truncated = block
                    .get("truncated")
                    .and_then(|t| t.get("abc"))
                    .and_then(Value::as_bool)
                    == Some(true);
                if abc_truncated {
                    // A plan cut off by its token budget is not a reviewable score: it is kept in
                    // the run (and on the result) but never becomes a score version.
                    block.insert("scoreVersionSkipped".to_owned(), json!("abc_truncated"));
                } else if block.get("scoreVersionId").is_none() {
                    match create_plan_version(state, job, &project_id, &block, abc).await {
                        Ok(Some(id)) => {
                            block.insert("scoreVersionId".to_owned(), json!(id));
                        }
                        Ok(None) => {}
                        Err(error) => errors.push(format!("score version: {}", error.detail)),
                    }
                }
            }
        }
        JobStatus::Failed if spec.kind == Yue2JobKind::RenderVersion => {
            if block.get("renderRecordId").is_none() {
                match record_render(state, job, &spec, &project_id, &block, result).await {
                    Ok(id) => {
                        block.insert("renderRecordId".to_owned(), json!(id));
                    }
                    Err(error) => errors.push(format!("render record: {}", error.detail)),
                }
            }
        }
        _ => return Ok(()),
    }
    if !errors.is_empty() {
        block.insert("sideEffectErrors".to_owned(), json!(errors));
    }
    result.insert("yue2".to_owned(), Value::Object(block));
    Ok(())
}

async fn create_plan_version(
    state: &AppState,
    job: &JobSnapshot,
    project_id: &str,
    block: &JsonObject,
    abc: String,
) -> Result<Option<String>, ApiError> {
    let Some(request) = block.get("request").cloned() else {
        return Ok(None);
    };
    let Ok(request) =
        serde_json::from_value::<sceneworks_core::yue2_score::SongRequest>(request.clone())
    else {
        // `cot = off` (no score) or a request shape sc-22997 cannot hold: nothing to version.
        return Ok(None);
    };
    let (project, job_id) = (project_id.to_owned(), job.id.clone());
    crate::yue2_scores::yue2_call(state.clone(), move |store| {
        let scores = store.yue2_score_store(&project)?;
        // Idempotent per job: the recovery sweep may re-run this.
        if let Some(existing) = scores.list_versions()?.items.into_iter().find(|version| {
            version.provenance.source.as_ref().is_some_and(|source| {
                source.kind == SourceKind::Job && source.id.as_deref() == Some(job_id.as_str())
            })
        }) {
            return Ok(Some(existing.id));
        }
        let record = scores.create_version(CreateVersionInput {
            abc,
            request,
            origin: VersionOrigin::Plan,
            provenance: job_provenance(&job_id),
        })?;
        Ok(Some(record.id))
    })
    .await
}

async fn record_render(
    state: &AppState,
    job: &JobSnapshot,
    spec: &Yue2JobSpec,
    project_id: &str,
    block: &JsonObject,
    result: &JsonObject,
) -> Result<String, ApiError> {
    let version = spec
        .sources
        .as_ref()
        .and_then(|sources| sources.version.clone())
        .ok_or_else(|| ApiError::bad_request("the render job carries no score version"))?;
    let completed = job.status == JobStatus::Completed;
    let entry = job
        .payload
        .get("modelManifestEntry")
        .cloned()
        .unwrap_or(Value::Null);
    // What the worker actually rendered (its own digests of the version it read); a failed job
    // that never read the version reports the digests it was queued with.
    let rendered = |key: &str, fallback: &str| {
        block
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or(fallback)
            .to_owned()
    };
    let truncated = observed_truncation(block)?;
    // Absent only when the job failed before it resolved its weights: then the identity the job
    // was queued for (the catalog's pinned primary download) is what it would have rendered with.
    let model = match identity_from(block, "model")? {
        Some(model) => model,
        None => model_identity(&entry),
    };
    let decoder = if completed {
        identity_from(block, "decoder")?
    } else {
        None
    };
    let input = RenderInput {
        status: if completed {
            RenderStatus::Completed
        } else {
            RenderStatus::Failed
        },
        score_sha256: rendered("renderedScoreSha256", &version.score_sha256),
        request_sha256: rendered("renderedRequestSha256", &version.request_sha256),
        truncated,
        model,
        decoder,
        job_id: Some(job.id.clone()),
        audio_asset_id: if completed {
            result
                .get("assetIds")
                .and_then(Value::as_array)
                .and_then(|ids| ids.first())
                .and_then(Value::as_str)
                .map(str::to_owned)
        } else {
            None
        },
        effective_settings: block.get("effectiveSettings").cloned(),
        error: if completed {
            None
        } else {
            Some(
                job.error
                    .clone()
                    .filter(|e| !e.trim().is_empty())
                    .unwrap_or_else(|| job.message.clone())
                    .chars()
                    .take(2000)
                    .collect(),
            )
        },
        provenance: job_provenance(&job.id),
    };
    let (project, job_id, version_id) = (project_id.to_owned(), job.id.clone(), version.id);
    crate::yue2_scores::yue2_call(state.clone(), move |store| {
        let scores = store.yue2_score_store(&project)?;
        if let Some(existing) = scores
            .list_renders()?
            .items
            .into_iter()
            .find(|render| render.job_id.as_deref() == Some(job_id.as_str()))
        {
            return Ok(existing.id);
        }
        Ok(store.record_yue2_render(&project, &version_id, input)?.id)
    })
    .await
}

// ---------------------------------------------------------------------------------------------
// Exports.
// ---------------------------------------------------------------------------------------------

/// Every usage policy an asset carries, as `{assetId, policy}` entries: its own
/// (`extra.usagePolicy`, a YuE2 output) and the ones it inherited from what it was made from
/// (`extra.usagePolicies`, a derived asset or an export — already in that shape, flattened so a
/// derivative of a derivative still names the original asset).
fn asset_usage_policies(asset_id: &str, asset: &Value) -> Vec<Value> {
    let extra = asset.get("extra");
    let own = extra
        .and_then(|extra| extra.get("usagePolicy"))
        .filter(|policy| !policy.is_null())
        .map(|policy| json!({ "assetId": asset_id, "policy": policy }));
    let inherited = extra
        .and_then(|extra| extra.get("usagePolicies"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| entry.get("policy").is_some_and(|policy| !policy.is_null()))
        .cloned();
    own.into_iter().chain(inherited).collect()
}

/// The usage policies of `asset_ids` in `project_id`, deduplicated. A missing asset carries none;
/// any other read failure is an error (never an empty policy set).
fn usage_policies_of(
    store: &ProjectStore,
    project_id: &str,
    asset_ids: impl IntoIterator<Item = String>,
) -> Result<Vec<Value>, ProjectStoreError> {
    let mut policies: Vec<Value> = Vec::new();
    for id in asset_ids {
        let asset = match store.get_asset(project_id, &id) {
            Ok(asset) => asset,
            Err(ProjectStoreError::NotFound(_)) => continue,
            Err(error) => return Err(error),
        };
        for policy in asset_usage_policies(&id, &asset) {
            if !policies.contains(&policy) {
                policies.push(policy);
            }
        }
    }
    Ok(policies)
}

/// The distinct usage policies of the assets a timeline document places — their own and the ones
/// they inherited — read from the project library.
pub(crate) async fn timeline_usage_policies(
    state: &AppState,
    project_id: &str,
    document: &Value,
) -> Result<Vec<Value>, ApiError> {
    let mut asset_ids = std::collections::BTreeSet::new();
    collect_asset_ids(document, &mut asset_ids);
    let project = project_id.to_owned();
    project_call(state.clone(), move |store| {
        usage_policies_of(&store, &project, asset_ids)
    })
    .await
}

/// The asset ids a generated-asset fact names as its inputs: `parents`, and every
/// `...AssetId` / `...AssetIds` field (source, reference, edit and stem inputs), excluding the
/// fact's own `assetId`.
fn fact_input_asset_ids(fact: &Value) -> Vec<String> {
    let mut ids = Vec::new();
    let Some(object) = fact.as_object() else {
        return ids;
    };
    for (key, value) in object {
        let is_input = key == "parents"
            || (key != "assetId" && (key.ends_with("AssetId") || key.ends_with("AssetIds")));
        if !is_input {
            continue;
        }
        match value {
            Value::String(id) if !id.trim().is_empty() => ids.push(id.clone()),
            Value::Array(items) => ids.extend(
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|id| !id.trim().is_empty())
                    .map(str::to_owned),
            ),
            _ => {}
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

/// Propagate usage policies onto generated assets derived from policy-bearing inputs (sc-22999,
/// epic E2): an asset made from a YuE2 output — an audio edit or extension of it, a voice clone
/// from it, a stem, a mux — inherits the source's policy as `extra.usagePolicies`, so neither
/// the library nor an export can launder the noncommercial restriction by deriving from it.
/// Runs on every reported asset write before it is persisted.
pub(crate) fn inherit_usage_policies(
    store: &ProjectStore,
    project_id: &str,
    asset_writes: &mut [Value],
) -> Result<(), ProjectStoreError> {
    for fact in asset_writes.iter_mut() {
        let inputs = fact_input_asset_ids(fact);
        if inputs.is_empty() {
            continue;
        }
        let inherited = usage_policies_of(store, project_id, inputs)?;
        if inherited.is_empty() {
            continue;
        }
        let Some(object) = fact.as_object_mut() else {
            continue;
        };
        let extra = object.entry("extra").or_insert_with(|| json!({}));
        let Some(extra) = extra.as_object_mut() else {
            continue;
        };
        let merged = extra
            .entry("usagePolicies")
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Some(list) = merged.as_array_mut() {
            for policy in inherited {
                if !list.contains(&policy) {
                    list.push(policy);
                }
            }
        }
    }
    Ok(())
}

fn collect_asset_ids(value: &Value, out: &mut std::collections::BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if key == "assetId" {
                    if let Some(id) = child.as_str() {
                        out.insert(id.to_owned());
                    }
                }
                collect_asset_ids(child, out);
            }
        }
        Value::Array(items) => items.iter().for_each(|item| collect_asset_ids(item, out)),
        _ => {}
    }
}

/// Refuse a commercial export that places a noncommercial asset, naming the asset and the
/// alternative its model's verdict points to.
pub(crate) fn refuse_commercial_export(policies: &[Value]) -> Result<(), ApiError> {
    let restricted: Vec<&Value> = policies
        .iter()
        .filter(|entry| {
            entry
                .pointer("/policy/commercialUse/verdict")
                .and_then(Value::as_str)
                == Some("refused")
        })
        .collect();
    if restricted.is_empty() {
        return Ok(());
    }
    let names: Vec<String> = restricted
        .iter()
        .map(|entry| {
            format!(
                "{} ({})",
                entry["assetId"].as_str().unwrap_or("?"),
                entry
                    .pointer("/policy/modelId")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
            )
        })
        .collect();
    Err(ApiError::typed(
        StatusCode::FORBIDDEN,
        format!(
            "This export is declared commercial, but it places noncommercial assets: {}. Replace \
             them with assets from a commercially eligible model (see each policy's \
             commercialUse.alternatives).",
            names.join(", ")
        ),
        COMMERCIAL_USE_REFUSED_CODE,
        json!({ "assets": restricted }),
    ))
}

/// Stamp an export job's inherited usage policies onto the asset it produces
/// (`extra.usagePolicies`), so the exported file's provenance says what it may be used for.
pub(crate) fn stamp_export_usage_policies(
    job_type: &JobType,
    payload: &JsonObject,
    asset_writes: &mut [Value],
) {
    if !matches!(job_type, JobType::TimelineExport) {
        return;
    }
    let Some(policies) = payload
        .get("usagePolicies")
        .and_then(Value::as_array)
        .filter(|policies| !policies.is_empty())
    else {
        return;
    };
    for fact in asset_writes.iter_mut() {
        let Some(object) = fact.as_object_mut() else {
            continue;
        };
        let extra = object.entry("extra").or_insert_with(|| json!({}));
        if let Some(extra) = extra.as_object_mut() {
            extra.insert("usagePolicies".to_owned(), json!(policies));
        }
    }
}
