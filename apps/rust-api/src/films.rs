use super::*;

use sceneworks_core::film_compile::production_plan_sha256;
use sceneworks_core::film_plan::{
    self, validate_reference_pack, PlanDiagnostic, ProductionPlan, ReferencePack, RunRecord,
};
use sceneworks_core::film_planner::{capabilities_for, PlannerCapabilities};
use sceneworks_core::film_workspace::{
    FilmDraft, FilmRecommendedTurbo, FilmRenderChoice, FilmRenderOptions, FilmRenderRegime,
    FilmRunLocator, FilmTurboUnavailableReason,
};
use sceneworks_core::project_store::FilmSoundInput;

use crate::film_harness::{
    finish_explicit_export, preflight_documents, start_explicit_export, ControllerLease,
    FilmDocumentPreflight, HarnessError, HttpTransport, ResumeOptions, RunControl, RunOptions,
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
    pub expected_draft_revision: Option<u32>,
    #[serde(default)]
    pub selected_shot_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FilmRenderOptionsRequest {
    pub draft_revision: u32,
    pub production_plan: ProductionPlan,
    pub reference_pack: ReferencePack,
    #[serde(default)]
    pub render_regime: Option<FilmRenderRegime>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FilmRunView {
    pub locator: FilmRunLocator,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record: Option<RunRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_operation: Option<crate::film_harness::ActionOperation>,
    pub controller_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub controller_owner: Option<String>,
    pub controller_interrupted: bool,
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
    let mut draft = FilmDraft::manual_one_shot(&project_id, &draft_id, &payload.title);
    let options = apply_selected_render_regime(&state, &mut draft).await?;
    if !options.recommended_turbo.available {
        // A new draft can only default to a recipe this host can actually run. Persist quality as
        // the honest effective choice; the render-options response still carries the exact reason
        // Turbo was unavailable, and installing an adapter never rewrites an existing draft.
        draft.render_regime = Some(FilmRenderRegime::Quality);
        apply_selected_render_regime(&state, &mut draft).await?;
    }
    let draft = project_call(state, move |store| {
        store.create_film_draft_document(&project_id, draft)
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
    ApiJson(mut draft): ApiJson<FilmDraft>,
) -> Result<Json<FilmDraft>, ApiError> {
    validate_planning_selection(&draft)?;
    apply_selected_render_regime(&state, &mut draft).await?;
    Ok(Json(
        project_call(state, move |store| {
            store.save_film_draft(&project_id, &draft_id, draft)
        })
        .await?,
    ))
}

pub(crate) async fn get_film_render_options(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
) -> Result<Json<FilmRenderOptions>, ApiError> {
    let draft = project_call(state.clone(), move |store| {
        store.get_film_draft(&project_id, &draft_id)
    })
    .await?;
    Ok(Json(resolve_film_render_options(&state, &draft).await?))
}

pub(crate) async fn preview_film_render_options(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<FilmRenderOptionsRequest>,
) -> Result<Json<FilmRenderOptions>, ApiError> {
    let mut draft = project_call(state.clone(), move |store| {
        store.get_film_draft(&project_id, &draft_id)
    })
    .await?;
    if draft.revision != payload.draft_revision {
        return Err(ApiError::conflict(format!(
            "Film draft revision conflict: expected {}, got {}",
            draft.revision, payload.draft_revision
        )));
    }
    draft.production_plan = payload.production_plan;
    draft.reference_pack = payload.reference_pack;
    // Absence is the legacy contract: preserve the caller's exact adapters and step override as
    // Custom. It must never opt an older draft into Turbo merely because Turbo is now available.
    draft.render_regime = payload.render_regime;
    Ok(Json(resolve_film_render_options(&state, &draft).await?))
}

pub(crate) async fn resolve_film_render_options(
    state: &AppState,
    draft: &FilmDraft,
) -> Result<FilmRenderOptions, ApiError> {
    let selected_regime = draft.effective_render_regime();
    let custom_steps = draft
        .production_plan
        .model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.steps)
        .and_then(|steps| u32::try_from(steps).ok())
        .filter(|steps| *steps > 0);
    let effective_from_plan = |default_steps: Option<u32>| FilmRenderChoice {
        adapter_ids: draft.production_plan.model.loras.clone(),
        effective_steps: custom_steps
            .or_else(|| {
                let mut steps = draft.production_plan.model.loras.iter().filter_map(|id| {
                    sceneworks_core::minimax_h3_turbo::turbo_recipe_for_lora_id(id)
                        .map(|recipe| recipe.steps)
                });
                let first = steps.next()?;
                steps.all(|step| step == first).then_some(first)
            })
            .or(default_steps),
    };

    let models = crate::models::model_catalog(state).await?;
    let Some(base) = models.iter().find(|entry| {
        entry.get("id").and_then(Value::as_str) == Some(draft.production_plan.model.id.as_str())
    }) else {
        return Ok(unavailable_render_options(
            selected_regime,
            FilmTurboUnavailableReason::ModelUnavailable,
            effective_from_plan(None),
            None,
        ));
    };
    if base.get("installState").and_then(Value::as_str) != Some("installed") {
        let default_steps = model_default_steps(base);
        return Ok(unavailable_render_options(
            selected_regime,
            FilmTurboUnavailableReason::ModelUnavailable,
            effective_from_plan(default_steps),
            default_steps,
        ));
    }
    let Some(base) = base.as_object() else {
        return Err(ApiError::internal(
            "Film model catalog entry is not an object",
        ));
    };
    let mut capabilities = capabilities_for(
        &draft.production_plan.model,
        base,
        film_plan::ModelLane::for_current_platform(),
    );
    let default_steps = capabilities.default_steps;
    let reference_requested = reference_partition_requested(&draft.reference_pack);
    if reference_requested {
        if let Some(reference_id) = film_plan::reference_partition_for(&capabilities.model_id) {
            let reference = models.iter().find(|entry| {
                entry.get("id").and_then(Value::as_str) == Some(reference_id)
                    && entry.get("installState").and_then(Value::as_str) == Some("installed")
            });
            let Some(reference) = reference.and_then(Value::as_object) else {
                return Ok(unavailable_render_options(
                    selected_regime,
                    FilmTurboUnavailableReason::IncompletePartitionCoverage,
                    effective_from_plan(default_steps),
                    default_steps,
                ));
            };
            capabilities = capabilities.with_reference_partition(reference);
        }
    }
    capabilities = capabilities.narrowed_to_pack(&draft.reference_pack);
    let installed_lora_ids = crate::loras::lora_catalog(state, Some(&draft.project_id))
        .await?
        .into_iter()
        .filter(|entry| entry.get("installState").and_then(Value::as_str) == Some("installed"))
        .filter_map(|entry| entry.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<Vec<_>>();
    let resolutions = render_resolutions(&draft.production_plan, &capabilities);
    let recommended_turbo = recommended_turbo_for(&capabilities, &installed_lora_ids, &resolutions);
    Ok(FilmRenderOptions {
        selected_regime,
        recommended_turbo,
        quality: FilmRenderChoice {
            adapter_ids: Vec::new(),
            effective_steps: default_steps,
        },
        effective: effective_from_plan(default_steps),
    })
}

fn recommended_turbo_for(
    capabilities: &PlannerCapabilities,
    installed_lora_ids: &[String],
    resolutions: &[String],
) -> FilmRecommendedTurbo {
    if resolutions.is_empty() {
        return unavailable_turbo(FilmTurboUnavailableReason::IncompatibleResolution);
    }
    let expected_partitions = 1 + usize::from(capabilities.offers_references());
    let mut recommended: Option<Vec<sceneworks_core::film_planner::PlannerTurboLora>> = None;
    for resolution in resolutions {
        let mut for_resolution = capabilities.clone();
        for_resolution.default_resolution = Some(resolution.clone());
        let offers = for_resolution
            .with_installed_turbo_loras(installed_lora_ids)
            .turbo_loras;
        if offers.len() < expected_partitions {
            return unavailable_turbo(if offers.is_empty() {
                FilmTurboUnavailableReason::NoInstalledCompatibleAdapter
            } else {
                FilmTurboUnavailableReason::IncompletePartitionCoverage
            });
        }
        if recommended.as_ref().is_some_and(|current| {
            current
                .iter()
                .map(|offer| &offer.id)
                .ne(offers.iter().map(|offer| &offer.id))
        }) {
            return unavailable_turbo(FilmTurboUnavailableReason::IncompatibleResolution);
        }
        recommended = Some(offers);
    }
    let recommended = recommended.unwrap_or_default();
    let effective_steps = recommended
        .first()
        .map(|offer| offer.steps)
        .filter(|first| recommended.iter().all(|offer| offer.steps == *first));
    FilmRecommendedTurbo {
        available: !recommended.is_empty() && effective_steps.is_some(),
        adapter_ids: recommended.iter().map(|offer| offer.id.clone()).collect(),
        effective_steps,
        unavailable_reason: None,
    }
}

fn unavailable_turbo(reason: FilmTurboUnavailableReason) -> FilmRecommendedTurbo {
    FilmRecommendedTurbo {
        available: false,
        adapter_ids: Vec::new(),
        effective_steps: None,
        unavailable_reason: Some(reason),
    }
}

pub(crate) async fn apply_selected_render_regime(
    state: &AppState,
    draft: &mut FilmDraft,
) -> Result<FilmRenderOptions, ApiError> {
    let options = resolve_film_render_options(state, draft).await?;
    match draft.render_regime {
        Some(FilmRenderRegime::RecommendedTurbo) if options.recommended_turbo.available => {
            draft.production_plan.model.loras = options.recommended_turbo.adapter_ids.clone();
            clear_step_override(&mut draft.production_plan);
        }
        Some(FilmRenderRegime::Quality) => {
            draft.production_plan.model.loras.clear();
            clear_step_override(&mut draft.production_plan);
        }
        Some(FilmRenderRegime::RecommendedTurbo) | Some(FilmRenderRegime::Custom) | None => {}
    }
    resolve_film_render_options(state, draft).await
}

fn unavailable_render_options(
    selected_regime: FilmRenderRegime,
    reason: FilmTurboUnavailableReason,
    effective: FilmRenderChoice,
    default_steps: Option<u32>,
) -> FilmRenderOptions {
    FilmRenderOptions {
        selected_regime,
        recommended_turbo: FilmRecommendedTurbo {
            available: false,
            adapter_ids: Vec::new(),
            effective_steps: None,
            unavailable_reason: Some(reason),
        },
        quality: FilmRenderChoice {
            adapter_ids: Vec::new(),
            effective_steps: default_steps,
        },
        effective,
    }
}

fn model_default_steps(entry: &Value) -> Option<u32> {
    entry
        .get("defaults")
        .and_then(|defaults| defaults.get("steps"))
        .and_then(Value::as_u64)
        .and_then(|steps| u32::try_from(steps).ok())
}

/// Does this pack ask the workspace for the family's REFERENCE partition?
///
/// Only an approved, image-backed role of a [`film_plan::BINDABLE_REFERENCE_KINDS`] kind does. A
/// DESCRIBED-ONLY role (sc-24025) never does however right its kind reads: it supplies no image
/// and cannot be bound, so asking for `minimax_h3_ref` on its account would have the workspace
/// demand a second 18 GB DiT — and report the render unavailable when it is not installed — for a
/// role that only ever reaches the model as text.
fn reference_partition_requested(pack: &ReferencePack) -> bool {
    pack.references.iter().any(|reference| {
        reference.approved
            && reference.file().is_some()
            && film_plan::BINDABLE_REFERENCE_KINDS.contains(&reference.kind.as_str())
    })
}

fn render_resolutions(plan: &ProductionPlan, capabilities: &PlannerCapabilities) -> Vec<String> {
    let fallback = plan
        .model
        .resolution
        .clone()
        .or_else(|| capabilities.default_resolution.clone());
    let mut resolutions = Vec::new();
    for shot in &plan.shots {
        if let Some(resolution) = shot.resolution.as_ref().or(fallback.as_ref()) {
            if film_plan::parse_resolution(resolution).is_none() {
                return Vec::new();
            }
            if !resolutions.contains(resolution) {
                resolutions.push(resolution.clone());
            }
        }
    }
    if plan.shots.is_empty() {
        if let Some(resolution) = fallback {
            if film_plan::parse_resolution(&resolution).is_none() {
                return Vec::new();
            }
            resolutions.push(resolution);
        }
    }
    resolutions
}

fn clear_step_override(plan: &mut ProductionPlan) {
    if let Some(advanced) = plan.model.advanced.as_mut() {
        advanced.steps = None;
        if advanced.reference_image_short_edge.is_none() {
            plan.model.advanced = None;
        }
    }
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

pub(crate) async fn add_film_sound(
    State(state): State<AppState>,
    Path((project_id, draft_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<FilmSoundInput>,
) -> Result<(StatusCode, Json<FilmDraft>), ApiError> {
    let draft = project_call(state, move |store| {
        store.add_film_sound(&project_id, &draft_id, payload)
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

#[cfg(test)]
type FilmPinBarrier = (
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
);
#[cfg(test)]
static FILM_PIN_BARRIERS: std::sync::LazyLock<
    parking_lot::Mutex<std::collections::HashMap<String, FilmPinBarrier>>,
> = std::sync::LazyLock::new(Default::default);

#[cfg(test)]
pub(crate) fn film_pin_barrier(
    draft_id: &str,
) -> (
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Sender<()>,
) {
    let (validated_tx, validated_rx) = tokio::sync::oneshot::channel();
    let (resume_tx, resume_rx) = tokio::sync::oneshot::channel();
    FILM_PIN_BARRIERS
        .lock()
        .insert(draft_id.to_owned(), (validated_tx, resume_rx));
    (validated_rx, resume_tx)
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
    if payload
        .expected_draft_revision
        .is_some_and(|expected| expected != draft.revision)
    {
        return Err(ApiError::conflict("Film draft revision conflict: the saved draft changed; reload and preflight before rendering"));
    }
    let selected = effective_selection(&draft, payload.selected_shot_ids)?;
    let mut findings =
        film_plan::validate_all(&draft.production_plan, &draft.reference_pack, None, None);
    findings.extend(render_regime_findings(&state, &draft).await?);
    if let Some(compiled) = draft.compiled_plan.as_ref() {
        let sha = production_plan_sha256(&draft.production_plan)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        findings.extend(compiled.staleness_findings(&draft.production_plan, &sha));
    }
    if !findings.is_empty() {
        return Err(invalid_film_document(findings));
    }

    #[cfg(test)]
    {
        let barrier = FILM_PIN_BARRIERS.lock().remove(&draft_id);
        if let Some((validated, resume)) = barrier {
            let _ = validated.send(());
            let _ = resume.await;
        }
    }
    let locator_id = format!("filmrun_{}", Uuid::new_v4().simple());
    let revision = draft.revision;
    let locator = project_call(state, move |store| {
        store.create_film_run_at_revision(&project_id, &locator_id, &draft_id, revision, selected)
    })
    .await?;
    Ok((
        StatusCode::CREATED,
        Json(FilmRunView {
            locator,
            record: None,
            action_operation: None,
            controller_active: false,
            controller_owner: None,
            controller_interrupted: false,
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
    if payload
        .expected_draft_revision
        .is_some_and(|expected| expected != draft.revision)
    {
        return Err(ApiError::conflict("Film draft revision conflict: the saved draft changed; reload and preflight before rendering"));
    }
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
    let base_url = state.settings.mcp_api_url.clone();
    let token = state.settings.access_token.clone();
    let transport = HttpTransport::new(&base_url, Some(token.clone()))
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut preflight = preflight_documents(
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
    })?;
    let regime_findings = render_regime_findings(state, draft).await?;
    if !regime_findings.is_empty() {
        preflight.valid = false;
        preflight.compiled = None;
        preflight.findings.extend(regime_findings);
    }
    preflight.draft_revision = Some(draft.revision);
    Ok(preflight)
}

async fn render_regime_findings(
    state: &AppState,
    draft: &FilmDraft,
) -> Result<Vec<PlanDiagnostic>, ApiError> {
    let options = resolve_film_render_options(state, draft).await?;
    let mut findings = Vec::new();
    match draft.render_regime {
        Some(FilmRenderRegime::RecommendedTurbo) => {
            if !options.recommended_turbo.available {
                findings.push(PlanDiagnostic::plan(
                    "renderRegime",
                    format!(
                        "recommended Turbo is unavailable: {}. Choose Full quality or install a compatible adapter",
                        turbo_unavailable_code(options.recommended_turbo.unavailable_reason)
                    ),
                ));
            } else if draft.production_plan.model.loras != options.recommended_turbo.adapter_ids
                || draft
                    .production_plan
                    .model
                    .advanced
                    .as_ref()
                    .and_then(|advanced| advanced.steps)
                    .is_some()
            {
                findings.push(PlanDiagnostic::plan(
                    "renderRegime",
                    "the saved recommended Turbo selection is stale for this model, resolution, or reference conditioning; save the draft to apply the displayed adapter recipe",
                ));
            }
        }
        Some(FilmRenderRegime::Quality) => {
            if !draft.production_plan.model.loras.is_empty()
                || draft
                    .production_plan
                    .model
                    .advanced
                    .as_ref()
                    .and_then(|advanced| advanced.steps)
                    .is_some()
            {
                findings.push(PlanDiagnostic::plan(
                    "renderRegime",
                    "Full quality requires no adapters or step override; save the draft to apply the quality recipe",
                ));
            }
        }
        Some(FilmRenderRegime::Custom) | None => {}
    }
    Ok(findings)
}

pub(crate) fn turbo_unavailable_code(reason: Option<FilmTurboUnavailableReason>) -> &'static str {
    match reason {
        Some(FilmTurboUnavailableReason::ModelUnavailable) => "model_unavailable",
        Some(FilmTurboUnavailableReason::NoInstalledCompatibleAdapter) => {
            "no_installed_compatible_adapter"
        }
        Some(FilmTurboUnavailableReason::IncompletePartitionCoverage) => {
            "incomplete_partition_coverage"
        }
        Some(FilmTurboUnavailableReason::IncompatibleResolution) => "incompatible_resolution",
        None => "unknown",
    }
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
    let lease = ControllerLease::acquire_for_api(
        &files.directory,
        format!("api:{run_id}"),
        state.film_controller_shutdown.clone(),
    )
    .map_err(|error| ApiError::conflict(error.to_string()))?;
    view.controller_active = true;

    let base_url = state.settings.mcp_api_url.clone();
    let token = state.settings.access_token.clone();
    let project_id_for_run = project_id.clone();
    let selected_shot_ids = view.locator.selected_shot_ids.clone();
    let transport = HttpTransport::new(&base_url, Some(token))
        .map_err(|error| ApiError::internal(error.to_string()))?;
    tokio::spawn(async move {
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

pub(crate) async fn export_film_run(
    State(state): State<AppState>,
    Path((project_id, run_id)): Path<(String, String)>,
) -> Result<(StatusCode, Json<FilmRunView>), ApiError> {
    let files = project_call(state.clone(), {
        let project_id = project_id.clone();
        let run_id = run_id.clone();
        move |store| store.film_run_files(&project_id, &run_id)
    })
    .await?;
    let lease =
        ControllerLease::acquire_new_action(&files.directory, format!("api-export:{run_id}"))
            .map_err(|error| ApiError::conflict(error.to_string()))?;
    let base_url = state.settings.mcp_api_url.clone();
    let token = state.settings.access_token.clone();
    let transport = HttpTransport::new(&base_url, Some(token.clone()))
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut options = ResumeOptions::new(files.directory.clone());
    options.poll_interval = Duration::from_secs(2);
    let (record, task) = start_explicit_export(&transport, &options)
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let controller_owner = format!("api-export:{run_id}");
    let locator =
        project_call(state, move |store| store.get_film_run(&project_id, &run_id)).await?;
    let action_operation = crate::film_harness::read_action_operation(&options.out_dir)
        .map_err(|error| ApiError::internal(error.to_string()))?;
    tokio::spawn(async move {
        let _lease = lease;
        let transport = match HttpTransport::new(&base_url, Some(token)) {
            Ok(transport) => transport,
            Err(error) => {
                tracing::error!(%error, "film export transport failed");
                return;
            }
        };
        if let Err(error) = finish_explicit_export(&transport, &options, &task).await {
            tracing::error!(job_id = task.job_id, %error, "film export stopped");
        }
    });
    Ok((
        StatusCode::ACCEPTED,
        Json(FilmRunView {
            locator,
            record: Some(record),
            action_operation,
            controller_active: true,
            controller_owner: Some(controller_owner),
            controller_interrupted: false,
        }),
    ))
}

pub(crate) async fn load_run_view(
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
    let controller_active = ControllerLease::is_active(&files.directory)
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let controller_owner = std::fs::read_to_string(
        files
            .directory
            .join(crate::film_harness::CONTROLLER_LOCK_FILE),
    )
    .ok()
    .and_then(|metadata| {
        metadata
            .lines()
            .find_map(|line| line.strip_prefix("owner=").map(str::to_owned))
    });
    Ok(FilmRunView {
        locator,
        record,
        action_operation: crate::film_harness::read_action_operation(&files.directory)
            .map_err(|error| ApiError::internal(error.to_string()))?,
        controller_active,
        controller_interrupted: !controller_active && controller_owner.is_some(),
        controller_owner,
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

#[cfg(test)]
mod render_options_tests {
    use super::*;
    use sceneworks_core::film_plan::PlanModel;

    fn entry(id: &str, reference: bool) -> serde_json::Map<String, Value> {
        json!({
            "id": id,
            "type": "video",
            "capabilities": if reference {
                json!(["reference_to_video"])
            } else {
                json!(["text_to_video", "image_to_video", "first_last_frame"])
            },
            "defaults": { "fps": 24, "resolution": "1344x768", "steps": 50 },
            "limits": {
                "resolutions": ["1344x768", "576x320"],
                "maxReferenceAssets": if reference { 9 } else { 0 }
            }
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    fn capabilities(resolution: &str) -> PlannerCapabilities {
        capabilities_for(
            &PlanModel {
                id: "minimax_h3".to_owned(),
                tier: Some("q4".to_owned()),
                loras: Vec::new(),
                fps: Some(24),
                resolution: Some(resolution.to_owned()),
                advanced: None,
            },
            &entry("minimax_h3", false),
            film_plan::ModelLane::Mlx,
        )
    }

    fn installed() -> Vec<String> {
        [
            "minimax_h3_turbo_4step_768p",
            "minimax_h3_turbo_8step",
            "minimax_h3_turbo_4step_v01",
            "minimax_h3_ref2v_turbo_4step",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    /// A pack of DESCRIBED-ONLY roles selects the BASE partition (sc-24025). Its roles reach the
    /// model as prompt text and nothing else, so the reference DiT is never required — and a
    /// workspace that asked for it would report the render unavailable on a machine that has only
    /// the base checkpoint installed.
    #[test]
    fn a_described_only_pack_does_not_request_the_reference_partition() {
        let pack = |references: Value| -> ReferencePack {
            serde_json::from_value(json!({
                "schemaVersion": film_plan::REFERENCE_PACK_SCHEMA_VERSION,
                "id": "described-refs",
                "version": 1,
                "references": references
            }))
            .expect("the pack parses")
        };

        // Every bindable KIND, described-only: the kind is right and the image is missing.
        let described = pack(json!([
            { "role": "courier", "kind": "character", "description": "Blue jacket." },
            { "role": "red_parcel", "kind": "prop", "description": "Red box." },
            { "role": "workshop", "kind": "location", "description": "The workshop." }
        ]));
        assert!(
            !reference_partition_requested(&described),
            "no image is supplied, so no reference DiT is needed"
        );

        // ONE image-backed bindable role is all it takes.
        let mixed = pack(json!([
            { "role": "courier", "kind": "character", "description": "Blue jacket." },
            { "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png",
              "description": "Red box." }
        ]));
        assert!(reference_partition_requested(&mixed));

        // An image-backed role of a kind that may never be BOUND does not request it either.
        let unbindable = pack(json!([
            { "role": "house_style", "kind": "style", "file": "references/house_style.png",
              "description": "House look." }
        ]));
        assert!(!reference_partition_requested(&unbindable));

        // Nor does an UNAPPROVED image-backed one: it is never resolved into a conditioning slot.
        let unapproved = pack(json!([
            { "role": "courier", "kind": "character", "file": "references/courier.png",
              "description": "Blue jacket.", "approved": false }
        ]));
        assert!(!reference_partition_requested(&unapproved));
    }

    #[test]
    fn recommendation_matches_canvas_and_required_partitions() {
        let low = recommended_turbo_for(
            &capabilities("576x320"),
            &installed(),
            &["576x320".to_owned()],
        );
        assert!(low.available);
        assert_eq!(low.adapter_ids, vec!["minimax_h3_turbo_4step_v01"]);
        assert_eq!(low.effective_steps, Some(4));

        let high = recommended_turbo_for(
            &capabilities("1344x768"),
            &installed(),
            &["1344x768".to_owned()],
        );
        assert_eq!(high.adapter_ids, vec!["minimax_h3_turbo_4step_768p"]);

        let mixed_caps =
            capabilities("576x320").with_reference_partition(&entry("minimax_h3_ref", true));
        let mixed = recommended_turbo_for(&mixed_caps, &installed(), &["576x320".to_owned()]);
        assert_eq!(
            mixed.adapter_ids,
            vec!["minimax_h3_turbo_4step_v01", "minimax_h3_ref2v_turbo_4step"]
        );
        assert_eq!(mixed.effective_steps, Some(4));
    }

    #[test]
    fn recommendation_explains_missing_partition_and_mixed_canvas() {
        let mixed_caps =
            capabilities("576x320").with_reference_partition(&entry("minimax_h3_ref", true));
        let partial = recommended_turbo_for(
            &mixed_caps,
            &["minimax_h3_turbo_4step_v01".to_owned()],
            &["576x320".to_owned()],
        );
        assert_eq!(
            partial.unavailable_reason,
            Some(FilmTurboUnavailableReason::IncompletePartitionCoverage)
        );

        let incompatible = recommended_turbo_for(
            &capabilities("576x320"),
            &installed(),
            &["576x320".to_owned(), "1344x768".to_owned()],
        );
        assert_eq!(
            incompatible.unavailable_reason,
            Some(FilmTurboUnavailableReason::IncompatibleResolution)
        );

        let missing = recommended_turbo_for(&capabilities("576x320"), &[], &["576x320".to_owned()]);
        assert_eq!(
            missing.unavailable_reason,
            Some(FilmTurboUnavailableReason::NoInstalledCompatibleAdapter)
        );
    }

    #[test]
    fn resolution_recipe_follows_effective_shot_canvases() {
        let mut plan = FilmDraft::manual_one_shot("project_1", "film_1", "Film").production_plan;
        plan.model.resolution = Some("576x320".to_owned());
        plan.shots[0].resolution = Some("1344x768".to_owned());
        let caps = capabilities("576x320");
        assert_eq!(
            render_resolutions(&plan, &caps),
            vec!["1344x768"],
            "an unused plan fallback must not create a false mixed-recipe conflict"
        );

        let mut fallback_shot = plan.shots[0].clone();
        fallback_shot.id = "SH020".to_owned();
        fallback_shot.resolution = None;
        plan.shots.push(fallback_shot);
        assert_eq!(
            render_resolutions(&plan, &caps),
            vec!["1344x768", "576x320"]
        );
    }
}
