use super::*;

#[derive(Debug, Clone)]
pub(crate) struct DownloadContext {
    pub(super) repo: String,
    pub(super) files: Vec<String>,
    pub(super) fallback_size_bytes: Option<u64>,
}

pub(crate) async fn create_model_download_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelDownloadRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let model = require_model_by_id(&state, &model_id).await?;
    // License-acknowledgment gate (sc-17227), enforced HERE and not only in the web client.
    //
    // The web client refuses an unacknowledged download at `createModelDownloadJob`, but that code
    // only runs in the client. This endpoint is reachable from a browser on another machine (the
    // remote-access lane, epic 4484), from a workflow envelope's suggested action, and from curl,
    // and no client-side check binds any of those.
    //
    // Scoped to `requiresLicenseAcknowledgment` — NOT to `gated`. A gated model's download fails at
    // Hugging Face with a 401 without a saved credential, so an unacknowledged fetch never lands its
    // weights; adding the requirement there would 4xx every existing gated download whose client
    // predates the field, for no gain. A `requiresLicenseAcknowledgment` model's repo is PUBLIC —
    // nothing upstream refuses it — so this rejection is the only thing between the request and the
    // weights, which is why the flag defaults to `false` and must be asserted, not assumed.
    if model_requires_license_acknowledgment(&model)
        && !payload.license_acknowledged
        && crate::yue2_jobs::current_license_acknowledgment(&state, &model)
            .await?
            .is_none()
    {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            detail: format!(
                "Model '{model_id}' requires accepting its license before download. Accept the \
                 license on the Models screen, or send `licenseAcknowledged: true` to assert that \
                 the user has accepted it."
            ),
            code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
            context: None,
        });
    }
    if let Some(component_id) = payload.component_id.as_deref() {
        ensure_model_not_cleanup_only(&model)?;
        if !payload.choices.is_empty() {
            return Err(ApiError::bad_request(
                "Imported component repair takes no co-requisite choices",
            ));
        }
        let base_id = model
            .get("baseModel")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let base = find_model_by_id(&state, base_id).await?.ok_or_else(|| {
            ApiError::bad_request("Imported component repair requires its compatible base")
        })?;
        let download = bound_import_component_download(
            &model,
            &base,
            component_id,
            payload.variant.as_deref(),
            &state.settings.data_dir,
            false,
        )
        .map_err(ApiError::bad_request)?;
        if payload.license_acknowledged {
            for entry in [&model, &base] {
                if model_requires_license_acknowledgment(entry) {
                    crate::yue2_jobs::record_license_acknowledgment(&state, entry, "download")
                        .await?;
                }
            }
        }
        ensure_inherited_component_licenses_acknowledged(
            &state,
            model.as_object().expect("validated import"),
            payload.license_acknowledged,
        )
        .await?;
        ensure_license_acknowledged_for_source_inner(
            &state,
            &[download.get("repo").and_then(Value::as_str)],
            None,
            payload.license_acknowledged,
            base.get("id").and_then(Value::as_str),
        )
        .await?;
        let mut job_payload = build_model_download_job_payload(
            &model,
            &model_id,
            &download,
            None,
            false,
            payload.license_acknowledged,
            &state.settings.data_dir,
        )?;
        job_payload.insert("componentId".to_owned(), json!(component_id));
        job_payload.insert(
            "importComponentTier".to_owned(),
            model["importComponentTier"].clone(),
        );
        let binding = catalog_download_binding(
            &model_id,
            base["id"].as_str().expect("validated base"),
            "importComponent",
            &download,
            &job_payload,
            None,
            Some(component_id),
        );
        let job = queue_catalog_download(
            state,
            job_payload,
            requested_gpu_or_auto(payload.requested_gpu),
            binding,
        )
        .await?;
        return Ok((StatusCode::CREATED, Json(public_job_snapshot(job))));
    }
    ensure_model_downloadable(&model)?;
    // sc-22999: the acceptance is recorded server-side for the entry's current terms, so a queued
    // YuE2 job can re-check it at execution and a withdrawn acknowledgment refuses the job.
    if model_requires_license_acknowledgment(&model) && payload.license_acknowledged {
        crate::yue2_jobs::record_license_acknowledgment(&state, &model, "download").await?;
    }
    // Tier selection (sc-8508): an explicit `variant` installs that quant tier's download entry; an
    // absent variant installs the default tier (back-compat). A variant the model doesn't advertise
    // is a 400 rather than a silent wrong-tier install.
    let download = match payload
        .variant
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        Some(variant) => model_download_for_variant(&model, variant).ok_or_else(|| {
            ApiError::bad_request(format!(
                "Model does not define a '{variant}' download variant"
            ))
        })?,
        None => model_download(&model).ok_or_else(|| {
            ApiError::bad_request("Model does not define a Hugging Face download")
        })?,
    };
    let selected_download = download.clone();
    // sc-24112: a DECLARED-but-unpublished tier is refused HERE, at the one place a download job is
    // created, rather than by hiding the row — the catalog must keep enumerating it so the tier
    // axis, the memory ladder and the picker are all real before the bytes exist. Its revision is
    // the null SHA, so without this the job would queue a fetch that cannot resolve and the user
    // would see an opaque download failure instead of the reason.
    if is_pending_artifact_download(&download) {
        return Err(ApiError::bad_request(format!(
            "Model '{model_id}' declares the '{}' tier but its artifact is not published yet; \
             install a different tier.",
            download
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("selected")
        )));
    }
    // sc-22998 / sc-22999: a locally derived tier (YuE2 q8 / q4) is not an artifact anyone can
    // download — it is produced on this machine from the `fromVariant` original. Its row's `files`
    // name the ORIGINAL, so the job fetches (or re-verifies) that original under the ORIGINAL's
    // variant (its receipt is the original's, never the tier's) and carries the derivation for the
    // worker's audio-lane preparer to run afterwards. The tier reads installed only once its
    // snapshot verifies against the pinned weights (`derived_snapshot_state`).
    let derivation =
        if sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(
            &download,
        ) {
            let variant = download
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("selected")
                .to_owned();
            let derivation =
                sceneworks_core::model_artifacts::artifact_selection::local_derivation(&download)
                    .ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "Model '{model_id}': the '{variant}' tier declares a malformed \
                         localDerivation block."
                    ))
                })?;
            Some((variant, derivation))
        } else {
            None
        };
    let download = match &derivation {
        Some((variant, derivation)) => model_download_for_variant(&model, &derivation.from_variant)
            .filter(|original| {
                !sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(
                    original,
                )
            })
            .ok_or_else(|| {
                ApiError::bad_request(format!(
                    "Model '{model_id}': the '{variant}' tier derives from '{}', which this \
                     catalog does not define as a download.",
                    derivation.from_variant
                ))
            })?,
        None => download,
    };
    // The selected `download` is always the primary/tier entry — `model_download` and
    // `model_download_for_variant` skip co-requisites (sc-9696), so a co-requisite can never be
    // installed as if it were the model itself.
    let requested_variant = match &derivation {
        Some((_, derivation)) => Some(derivation.from_variant.as_str()),
        None => payload
            .variant
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty()),
    };

    // Only the SELECTED tier's co-requisites (sc-14980). Mage-Flow's shared text encoder exists as
    // three per-tier subtrees; fetching all of them would pull 16.1 GB of text encoder for a q4
    // install that needs 2.51 GB. Tier-agnostic co-requisites (every other family) are unaffected —
    // they carry no `variant` and always apply. Read the tier off the resolved `download` rather than
    // the request so the default-tier install (no explicit `variant`) picks up its tier too.
    let selected_variant = download
        .get("variant")
        .and_then(Value::as_str)
        .map(str::to_owned);
    // Choice groups (sc-22998): of a group's options — YuE2's standard / legacy decoder — only the
    // one the request names (else the manifest default) is queued; the others stay add-ons. An
    // unknown group or option is refused rather than answered with a different option.
    //
    // The request's choices are validated first (an unknown group or option is refused), then any
    // group the request leaves unset takes the option already partly on disk before the default, so
    // a REPAIR of a half-downloaded legacy decoder completes it instead of fetching the standard one.
    sceneworks_core::model_artifacts::artifact_selection::resolve_co_requisite_choices(
        &model,
        selected_variant.as_deref(),
        &payload.choices,
    )
    .map_err(|error| ApiError::bad_request(format!("Model '{model_id}': {error}")))?;
    let repair_choices =
        sceneworks_core::model_artifacts::artifact_selection::co_requisite_choices_for_repair(
            &model,
            selected_variant.as_deref(),
            &payload.choices,
            |download| co_requisite_presence(&state.settings.data_dir, download),
        );
    let choices =
        sceneworks_core::model_artifacts::artifact_selection::resolve_co_requisite_choices(
            &model,
            selected_variant.as_deref(),
            &repair_choices,
        )
        .map_err(|error| ApiError::bad_request(format!("Model '{model_id}': {error}")))?;
    let co_requisites =
        sceneworks_core::model_artifacts::artifact_selection::model_co_requisite_downloads_for_selection(
            &model,
            selected_variant.as_deref(),
            &choices,
        );

    // The REPO-keyed half of the same gate (sc-17227). The check above is keyed on the catalog id
    // in the PATH, so it fires only when the entry that id names declares
    // `requiresLicenseAcknowledgment` itself. Every other door — `POST /api/v1/jobs`,
    // `/models/import`, `/loras/import` — is keyed on the repo the request will FETCH, resolved
    // against `license_acknowledgment_repo_index`. That asymmetry left two doors onto one set of
    // weights: an entry that does not carry the flag but whose `downloads` name a repo a flagged
    // entry declares would have been fetched here while the generic queue answered 403 for the same
    // repo. Shared co-requisite rows are exactly that shape, and the manifest already uses it.
    //
    // Unreachable in the shipped catalog today — every restricted repo reference lives inside an
    // entry that carries the flag, and the manifest audit's
    // `test_every_entry_naming_a_license_gated_repo_carries_the_flag_itself` keeps it that way —
    // but that is a property of the current manifest, not of this route. An
    // ADDITION, not a replacement: the id check above keeps its own refusal text, which names the
    // model the caller asked for rather than the repo that supplies it.
    let queued_repos: Vec<String> = std::iter::once(&download)
        .chain(co_requisites.iter())
        .filter_map(|entry| entry.get("repo").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let queued_repos: Vec<Option<&str>> = queued_repos
        .iter()
        .map(|repo| Some(repo.as_str()))
        .collect();
    ensure_license_acknowledged_for_source_inner(
        &state,
        &queued_repos,
        None,
        payload.license_acknowledged,
        Some(&model_id),
    )
    .await?;

    let mut job_payload = build_model_download_job_payload(
        &model,
        &model_id,
        &download,
        requested_variant,
        true,
        payload.license_acknowledged,
        &state.settings.data_dir,
    )?;
    if let Some((variant, derivation)) = &derivation {
        job_payload.insert(
            LOCAL_DERIVATION_PAYLOAD_KEY.to_owned(),
            json!({
                "variant": variant,
                "fromVariant": derivation.from_variant,
                "conversion": derivation.conversion,
                "weightsFile": derivation.weights_file,
                "weightsBytes": derivation.weights_bytes,
                "weightsSha256": derivation.weights_sha256,
            }),
        );
    }

    // Co-requisites (sc-9696): dependencies that must install ALONGSIDE the primary — e.g. the PiD
    // decoder's shared gemma-2-2b-it caption encoder, or 10Eros's cond_safe distill LoRA. Without
    // them the feature silently no-ops (for PiD, `resolve_pid_weights` falls back to the native VAE
    // with no error). The catalog already filtered `downloads` to this OS, so every co-requisite
    // here applies. Each is queued as its own ModelDownload job (the worker is one-repo-per-job);
    // the catalog reports the entry installed only once all of them are cached
    // (`install_state_for`). `include_family: false` because a co-requisite (e.g. a text encoder)
    // is a different artifact than the model's primary checkpoint and must not be reconciled
    // against the model's family.
    let requested_gpu = requested_gpu_or_auto(payload.requested_gpu);
    for co_requisite in &co_requisites {
        if co_requisite_satisfied_by_exact_legacy_rename(&state.settings.data_dir, co_requisite) {
            continue;
        }
        let co_payload = build_model_download_job_payload(
            &model,
            &model_id,
            co_requisite,
            None,
            false,
            payload.license_acknowledged,
            &state.settings.data_dir,
        )?;
        let binding = catalog_download_binding(
            &model_id,
            &model_id,
            "download",
            co_requisite,
            &co_payload,
            None,
            None,
        );
        queue_catalog_download(state.clone(), co_payload, requested_gpu.clone(), binding).await?;
    }

    // The primary job is the one returned to the caller (its id is what the download UI tracks).
    let mut binding = catalog_download_binding(
        &model_id,
        &model_id,
        "download",
        &download,
        &job_payload,
        None,
        None,
    );
    binding["selectedDownload"] = selected_download;
    let job = queue_catalog_download(state, job_payload, requested_gpu, binding).await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

/// A purpose the entry declares no conditional components for (sc-23002).
pub(crate) const CONDITIONAL_COMPONENTS_NOT_DECLARED_CODE: &str =
    "conditional_components_not_declared";

/// `POST /api/v1/models/:model_id/conditional-components/:purpose/download` (sc-23002) — acquire the
/// pinned components a model needs only for `purpose` (YuE2's SheetSage2 + MERT-v2-FullSong cover
/// closure). They are never `downloads[]` rows, so no install reaches them; this is the one door, and
/// it applies the entry's gates exactly as the model install does:
///
/// * a `blocked` component refuses the whole purpose (403 `component_blocked`, reason + unblock);
/// * an entry that requires a licence acknowledgment refuses without one (403
///   `license_acknowledgment_required`) and records it server-side when asserted, for the entry's
///   CURRENT terms — the same acknowledgment YuE2 jobs re-check at execution;
/// * the repo-keyed half of the gate runs over the repos it will queue, as on every other door.
///
/// Each component not already installed is queued as its own `model_download` job (the worker is
/// one repo per job) at its pinned revision and file list, carrying the acknowledgment so a retry
/// re-validates. Components already on disk are reported, not re-fetched.
pub(crate) async fn create_conditional_components_download_jobs(
    State(state): State<AppState>,
    Path((model_id, purpose)): Path<(String, String)>,
    ApiJson(payload): ApiJson<ConditionalComponentsDownloadRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    use sceneworks_core::model_usage_policy::{
        conditional_component_downloads, ConditionalComponentsError,
    };
    let model = require_model_by_id(&state, &model_id).await?;
    let rows = conditional_component_downloads(&model, &purpose).map_err(|error| match error {
        ConditionalComponentsError::NotDeclared { .. } => ApiError::typed(
            StatusCode::NOT_FOUND,
            format!("Model '{model_id}': {error}."),
            CONDITIONAL_COMPONENTS_NOT_DECLARED_CODE,
            json!({ "purpose": purpose }),
        ),
        ConditionalComponentsError::Blocked { ref blocked, .. } => ApiError::typed(
            StatusCode::FORBIDDEN,
            format!("Model '{model_id}': {error}"),
            COMPONENT_BLOCKED_CODE,
            json!({
                "purpose": purpose,
                "blocked": blocked.iter().map(|(component_id, reason, unblock)| json!({
                    "componentId": component_id, "reason": reason, "unblock": unblock,
                })).collect::<Vec<_>>(),
            }),
        ),
    })?;
    if model_requires_license_acknowledgment(&model) {
        if !payload.license_acknowledged
            && crate::yue2_jobs::current_license_acknowledgment(&state, &model)
                .await?
                .is_none()
        {
            return Err(ApiError {
                status: StatusCode::FORBIDDEN,
                detail: format!(
                    "Model '{model_id}' requires accepting its license before its '{purpose}' \
                     components download. Accept the license on the Models screen, or send \
                     `licenseAcknowledged: true` to assert that the user has accepted it."
                ),
                code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
                context: None,
            });
        }
        if payload.license_acknowledged {
            crate::yue2_jobs::record_license_acknowledgment(&state, &model, "download").await?;
        }
    }
    let repos: Vec<String> = rows
        .iter()
        .filter_map(|row| row.get("repo").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let repo_refs: Vec<Option<&str>> = repos.iter().map(|repo| Some(repo.as_str())).collect();
    ensure_license_acknowledged_for_source_inner(
        &state,
        &repo_refs,
        None,
        payload.license_acknowledged,
        Some(&model_id),
    )
    .await?;
    let requested_gpu = requested_gpu_or_auto(payload.requested_gpu);
    let mut jobs = Vec::new();
    let mut components = Vec::new();
    for row in &rows {
        let component_id = row.get("componentId").cloned().unwrap_or(Value::Null);
        let presence = if model_id.starts_with("moss_image_") {
            pinned_component_presence(&state.settings.data_dir, row)
        } else {
            co_requisite_presence(&state.settings.data_dir, row)
        };
        if presence == CoRequisitePresence::Installed {
            components.push(json!({
                "componentId": component_id,
                "repo": row.get("repo"),
                "revision": row.get("revision"),
                "status": "installed",
            }));
            continue;
        }
        let job_payload = build_model_download_job_payload(
            &model,
            &model_id,
            row,
            None,
            false,
            payload.license_acknowledged,
            &state.settings.data_dir,
        )?;
        let binding = catalog_download_binding(
            &model_id,
            &model_id,
            "conditional",
            row,
            &job_payload,
            Some(&purpose),
            None,
        );
        let job =
            queue_catalog_download(state.clone(), job_payload, requested_gpu.clone(), binding)
                .await?;
        components.push(json!({
            "componentId": component_id,
            "repo": row.get("repo"),
            "revision": row.get("revision"),
            "status": "queued",
            "jobId": job.id,
        }));
        jobs.push(public_job_snapshot(job));
    }
    Ok((
        StatusCode::CREATED,
        Json(json!({ "purpose": purpose, "jobs": jobs, "components": components })),
    ))
}

/// Stamp each conditional component row of a catalog entry with its install state (sc-23002) and
/// summarize each purpose — `conditionalPurposes.<purpose>.installState` is `installed` only when
/// every component the purpose needs is, `incomplete` when any is partly on disk, else `missing`,
/// and `blocked` when any carries a block — so a client can tell whether a cover from a recording
/// can run without guessing from file names.
pub(super) fn apply_conditional_component_state(object: &mut JsonObject, data_dir: &FsPath) {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let Some(rows) = object
        .get_mut("conditionalComponents")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    let mut purposes: std::collections::BTreeMap<String, (bool, bool, bool)> =
        std::collections::BTreeMap::new();
    for row in rows.iter_mut() {
        let presence = if row
            .get("componentId")
            .and_then(Value::as_str)
            .is_some_and(|id| id.starts_with("moss-dmd-") || id == "moss-image-edit")
        {
            pinned_component_presence(data_dir, row)
        } else {
            co_requisite_presence(data_dir, row)
        };
        let blocked = row.get("blocked").is_some();
        let state = match presence {
            CoRequisitePresence::Installed => "installed",
            CoRequisitePresence::Incomplete => "incomplete",
            CoRequisitePresence::Absent => "missing",
        };
        for purpose in row
            .get("requiredFor")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let entry = purposes
                .entry(purpose.to_owned())
                .or_insert((true, false, false));
            entry.0 &= presence == CoRequisitePresence::Installed;
            entry.1 |= presence == CoRequisitePresence::Incomplete;
            entry.2 |= blocked;
        }
        if let Some(row) = row.as_object_mut() {
            row.insert("installState".to_owned(), json!(state));
        }
    }
    let summary: JsonObject = purposes
        .into_iter()
        .map(|(purpose, (installed, incomplete, blocked))| {
            let state = if installed {
                "installed"
            } else if incomplete {
                "incomplete"
            } else {
                "missing"
            };
            (
                purpose,
                json!({ "installState": state, "blocked": blocked }),
            )
        })
        .collect();
    object.insert("conditionalPurposes".to_owned(), Value::Object(summary));
}

/// The install state of `model`'s `purpose` components right now (sc-23002): `Ok(())` when every
/// one is installed; otherwise the component ids that are not.
pub(crate) fn conditional_components_missing(
    data_dir: &FsPath,
    model: &Value,
    purpose: &str,
) -> Result<Vec<Value>, sceneworks_core::model_usage_policy::ConditionalComponentsError> {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let rows =
        sceneworks_core::model_usage_policy::conditional_component_downloads(model, purpose)?;
    Ok(rows
        .into_iter()
        .filter(|row| co_requisite_presence(data_dir, row) != CoRequisitePresence::Installed)
        .map(|row| {
            json!({
                "componentId": row.get("componentId"),
                "repo": row.get("repo"),
                "revision": row.get("revision"),
            })
        })
        .collect())
}

/// Source validity is independent of aggregate resident-component readiness. The store reuses its
/// sealed source stamps and rehashes only changed/racy entries, as at the existing install probe.
pub(super) fn imported_source_present(entry: &JsonObject, data_dir: &FsPath) -> bool {
    let Some(checkpoint_id) = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)
    else {
        return false;
    };
    let Ok(resolved) = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data_dir)
        .resolve(checkpoint_id)
    else {
        return false;
    };
    if sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family(entry) {
        sceneworks_core::minimax_h3_import::verify_entry_assembly(entry, &resolved.plan).is_ok()
    } else {
        entry
            .get("importPlan")
            .and_then(|binding| binding.get("planId"))
            .and_then(Value::as_str)
            .is_none_or(|plan_id| resolved.plan.plan_id == plan_id)
    }
}

/// Stable row identity, including legacy resident companions without a component name.
pub(super) fn imported_component_id(row: &Value, index: usize) -> String {
    row.get("componentId")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("resident-{index}"))
}

/// Unknown identifiers and alternate tiers never fall back to the primary DiT download. Resolve
/// against BOTH the persisted closure and the compatible base's authorized resident rows.
pub(super) fn bound_import_component_download(
    model: &Value,
    base: &Value,
    component_id: &str,
    requested_variant: Option<&str>,
    data_dir: &FsPath,
    allow_installed: bool,
) -> Result<Value, String> {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let entry = model
        .as_object()
        .ok_or("Imported component repair requires an import")?;
    if !sceneworks_core::minimax_h3_import::entry_is_imported_minimax_h3(entry)
        || !imported_source_present(entry, data_dir)
    {
        return Err(
            "Repair the imported source by relinking, rescanning or re-importing it first".into(),
        );
    }
    let tier = sceneworks_core::minimax_h3_import::selected_import_component_tier(entry)?;
    if requested_variant.is_some_and(|requested| requested != tier) {
        return Err("Component repair must retain the imported assembly's selected tier".into());
    }
    let rows = model
        .get("downloads")
        .and_then(Value::as_array)
        .ok_or("The import has no bound resident components")?;
    let candidates: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(index, row)| imported_component_id(row, *index) == component_id)
        .collect();
    if candidates.len() != 1 {
        return Err("Unknown or ambiguous imported resident component".into());
    }
    let (_, row) = candidates[0];
    let expected = sceneworks_core::minimax_h3_import::import_component_downloads(
        entry,
        base,
        Some(tier),
        std::env::consts::OS,
    )?;
    if !is_co_requisite_download(row) || !expected.contains(row) {
        return Err(
            "The bound component no longer matches the compatible base; re-import the checkpoint"
                .into(),
        );
    }
    if !allow_installed
        && imported_component_presence(data_dir, row) == CoRequisitePresence::Installed
    {
        return Err("This imported resident component is already installed".into());
    }
    Ok(row.clone())
}

pub(super) fn ensure_model_downloadable(model: &Value) -> Result<(), ApiError> {
    ensure_model_not_cleanup_only(model)?;
    if model.get("downloadable").and_then(Value::as_bool) == Some(false) {
        return Err(ApiError::bad_request(
            "This model is not available for download.",
        ));
    }
    Ok(())
}

pub(super) fn ensure_model_not_cleanup_only(model: &Value) -> Result<(), ApiError> {
    if model.get("platformCleanupOnly").and_then(Value::as_bool) == Some(true) {
        return Err(ApiError::bad_request(
            "This model is retained only so its existing local files can be removed. No download, conversion, or generation route is available on this platform.",
        ));
    }
    Ok(())
}

/// Build the worker `ModelDownload` job payload for one `download` entry of `model`. Factored out
/// (sc-9696) so the primary download and each co-requisite share identical payload shaping.
/// `explicit_variant` records a request-selected quant tier (falling back to the entry's own
/// `variant`); `include_family` forwards the model's declared family for the worker's post-download
/// family reconcile (sc-1663) — pass `false` for co-requisites, whose weights are a different artifact
/// than the model's primary checkpoint.
pub(super) const MEMORY_CALIBRATION_PROVENANCE_REQUIRED: &str =
    "memoryCalibrationProvenanceRequired";

pub(super) fn catalog_requires_memory_calibration_provenance(model: &Value) -> bool {
    model
        .get("mlx")
        .and_then(|mlx| mlx.get("calibrations"))
        .and_then(Value::as_array)
        .is_some_and(|calibrations| !calibrations.is_empty())
}

pub(super) fn insert_memory_calibration_provenance_requirement(
    job_payload: &mut JsonObject,
    model: &Value,
    primary_artifact: bool,
) {
    job_payload.insert(
        MEMORY_CALIBRATION_PROVENANCE_REQUIRED.to_owned(),
        Value::Bool(primary_artifact && catalog_requires_memory_calibration_provenance(model)),
    );
}

pub(super) fn build_model_download_job_payload(
    model: &Value,
    model_id: &str,
    download: &Value,
    explicit_variant: Option<&str>,
    include_family: bool,
    license_acknowledged: bool,
    data_dir: &FsPath,
) -> Result<JsonObject, ApiError> {
    let repo = required_string_field(download, "repo")?.to_owned();
    let files = download
        .get("files")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut job_payload = JsonObject::new();
    job_payload.insert("modelId".to_owned(), Value::String(model_id.to_owned()));
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(model_id)
                .to_owned(),
        ),
    );
    // The cold install-time digest is authorized only by the server's catalog entry. The typed
    // client request cannot supply or override this flag, and co-requisites stay inert because the
    // calibration artifact identity is the primary checkpoint.
    insert_memory_calibration_provenance_requirement(&mut job_payload, model, include_family);
    // Preserve legacy explicit assertions verbatim. A download authorized by the
    // server store must recheck that store on retry/duplicate, rather than minting
    // a permanent assertion that could survive withdrawal or changed terms.
    if license_acknowledged {
        job_payload.insert(
            LICENSE_ACKNOWLEDGED_PAYLOAD_KEY.to_owned(),
            Value::Bool(true),
        );
    }
    job_payload.insert(
        "provider".to_owned(),
        Value::String(required_string_field(download, "provider")?.to_owned()),
    );
    job_payload.insert("repo".to_owned(), Value::String(repo.clone()));
    job_payload.insert("files".to_owned(), json!(files));
    if let Some(identity) =
        sceneworks_core::managed_file::ManagedFileIdentity::from_download(download)
            .map_err(ApiError::bad_request)?
    {
        identity.stamp(&mut job_payload);
    }

    // Forward an explicit pinned `revision` (sc-13541) so the worker fetches the exact commit SHA the
    // runtime resolver reads, not `main`. The worker defaults to `main` when this is absent, so
    // omitting `revision` on an entry keeps its behavior unchanged. Required for companion weights a
    // provider resolves via a pinned-SHA `hf_get_pinned` (chatterbox_tts's ve/perth co-requisites): the
    // resolver refuses a non-SHA revision and reads `snapshots/<sha>/`, so a main-branch predownload
    // would land in the wrong snapshot and offline generation would fail.
    if let Some(revision) = download
        .get("revision")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        job_payload.insert("revision".to_owned(), Value::String(revision.to_owned()));
    }
    // Record which quant tier this job installs (sc-8508) so the download record + per-variant
    // install tracking agree on the tier. Falls back to the selected entry's own `variant` when the
    // request omitted one (the default tier may still be a labeled variant).
    if let Some(variant) = explicit_variant
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            download
                .get("variant")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
    {
        job_payload.insert("variant".to_owned(), Value::String(variant));
    }
    // Forward the catalog-declared family so the worker can re-verify the downloaded
    // weights match it (parity with model import). The catalog is project-curated, but
    // a mis-declared family would otherwise silently mismatch downstream adapter
    // selection; the worker reconciles and fails on a confident conflict (sc-1663).
    if include_family {
        if let Some(family) = model.get("family").and_then(Value::as_str) {
            if !family.trim().is_empty() {
                job_payload.insert("family".to_owned(), Value::String(family.to_owned()));
            }
        }
    }
    job_payload.insert(
        "targetDir".to_owned(),
        Value::String(
            data_dir
                .join("models")
                .join(safe_download_dir(&repo))
                .display()
                .to_string(),
        ),
    );
    if let Some(identity) = sceneworks_core::managed_file::authoritative_job_identity(&job_payload)
        .map_err(ApiError::bad_request)?
    {
        identity.stamp(&mut job_payload);
    }
    Ok(job_payload)
}

/// Best-effort credential host for a gated model when the manifest entry doesn't
/// set `credentialHost` explicitly: an explicit per-download `credentialHost`,
/// else the well-known host for the provider (`huggingface` ⇒ `huggingface.co`),
/// else the host of a `sourceUrl`. Normalized (scheme/path stripped, lower-cased)
/// to match how credentials are keyed in the store.
pub(super) fn derive_credential_host(model: &serde_json::Map<String, Value>) -> Option<String> {
    let downloads = model.get("downloads")?.as_array()?;
    for download in downloads {
        if let Some(host) = download
            .get("credentialHost")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
        {
            return Some(host);
        }
        if download.get("provider").and_then(Value::as_str) == Some("huggingface") {
            return Some("huggingface.co".to_owned());
        }
        if let Some(host) = download
            .get("sourceUrl")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
        {
            return Some(host);
        }
    }
    None
}

pub(crate) fn model_download_context(model: &Value) -> Result<Option<DownloadContext>, ApiError> {
    let Some(download) = model_download(model) else {
        return Ok(None);
    };
    Ok(Some(DownloadContext {
        repo: required_string_field(&download, "repo")?.to_owned(),
        files: string_array_field(&download, "files"),
        fallback_size_bytes: manifest_download_size_bytes(model, &download),
    }))
}

async fn queue_catalog_download(
    state: AppState,
    payload: JsonObject,
    requested_gpu: String,
    binding: Value,
) -> Result<JobSnapshot, ApiError> {
    create_generation_job_with_binding(
        state,
        JobType::ModelDownload,
        None,
        None,
        payload,
        requested_gpu,
        (None, Some(binding)),
    )
    .await
}

fn catalog_download_binding(
    model_id: &str,
    owner_id: &str,
    kind: &str,
    row: &Value,
    payload: &JsonObject,
    purpose: Option<&str>,
    component: Option<&str>,
) -> Value {
    json!({"modelId": model_id, "ownerId": owner_id, "kind": kind, "row": row,
        "payload": payload, "purpose": purpose, "component": component})
}

pub(crate) async fn validate_catalog_download_replay(
    state: &AppState,
    binding: &Value,
    payload: &JsonObject,
) -> Result<(), ApiError> {
    let invalid = || {
        ApiError::bad_request(
            "The saved catalog download no longer matches its authorized selection",
        )
    };
    // A patch cannot turn an accepted selection into a different worker operation.
    if !sceneworks_core::jobs_store::download_binding_matches_payload(binding, payload) {
        return Err(invalid());
    }
    let model_id = binding
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(invalid)?;
    let owner_id = binding
        .get("ownerId")
        .and_then(Value::as_str)
        .ok_or_else(invalid)?;
    let row = binding.get("row").ok_or_else(invalid)?;
    let catalog = model_catalog(state).await?;
    let model = catalog
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some(model_id))
        .ok_or_else(invalid)?;
    let owner = catalog
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some(owner_id))
        .ok_or_else(invalid)?;
    let valid = match binding.get("kind").and_then(Value::as_str) {
        Some("download") if model_id == owner_id => model
            .get("downloads")
            .and_then(Value::as_array)
            .is_some_and(|rows| {
                rows.contains(row)
                    && binding
                        .get("selectedDownload")
                        .is_none_or(|selected| rows.contains(selected))
            }),
        Some("conditional") if model_id == owner_id => {
            let purpose = binding
                .get("purpose")
                .and_then(Value::as_str)
                .ok_or_else(invalid)?;
            sceneworks_core::model_usage_policy::conditional_component_downloads(model, purpose)
                .map_err(|_| invalid())?
                .contains(row)
        }
        Some("importComponent") => {
            if model.get("baseModel").and_then(Value::as_str) != Some(owner_id) {
                return Err(invalid());
            }
            let component = binding
                .get("component")
                .and_then(Value::as_str)
                .ok_or_else(invalid)?;
            let current = bound_import_component_download(
                model,
                owner,
                component,
                None,
                &state.settings.data_dir,
                true,
            )
            .map_err(|_| invalid())?;
            ensure_inherited_component_licenses_acknowledged(
                state,
                model.as_object().ok_or_else(invalid)?,
                false,
            )
            .await?;
            &current == row
        }
        _ => false,
    };
    if !valid {
        return Err(invalid());
    }
    for entry in [model, owner] {
        if model_requires_license_acknowledgment(entry)
            && crate::yue2_jobs::current_license_acknowledgment(state, entry)
                .await?
                .is_none()
        {
            return Err(ApiError {
                status: StatusCode::FORBIDDEN,
                detail: "Current model license acknowledgment is required".to_owned(),
                code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
                context: None,
            });
        }
    }
    let repos = LICENSE_GATED_REPO_PAYLOAD_KEYS
        .iter()
        .map(|key| payload.get(*key).and_then(Value::as_str))
        .collect::<Vec<_>>();
    ensure_license_acknowledged_for_source_inner(
        state,
        &repos,
        payload.get("sourceUrl").and_then(Value::as_str),
        false,
        Some(owner_id),
    )
    .await
}
