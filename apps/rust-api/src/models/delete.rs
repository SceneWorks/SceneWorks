use super::*;

/// Distinct source repositories a model owns. Co-requisite downloads are excluded: they are
/// shared dependencies rather than this model's own source, and resolved-cache entries are
/// selected by *primary* provenance.
pub(crate) fn owned_source_repositories(model: &Value) -> Vec<String> {
    let mut repositories = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| !is_co_requisite_download(entry))
        .filter_map(|entry| entry.get("repo").and_then(Value::as_str))
        .filter(|repo| !repo.trim().is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    repositories.sort();
    repositories.dedup();
    repositories
}

/// Reconciles the resolved-model hot cache with a source that has just been deleted (sc-19710).
///
/// Without this, entries whose authoritative source is gone can never be reclaimed: automatic
/// retention refuses to evict an artifact it cannot re-verify as a second copy, so an orphaned
/// entry would be held `SourceUnverified` forever — exactly the stranded state the epic forbids.
///
/// Reconciliation failures never fail the delete, whose files are already removed; they are logged
/// and returned as warnings so the orphan is surfaced rather than silently stranded.
pub(super) async fn reconcile_deleted_source(
    state: &AppState,
    selectors: Vec<sceneworks_core::model_artifacts::resolved_cache::SourceLifecycleSelector>,
) -> Vec<String> {
    use sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore;

    if selectors.is_empty() {
        return Vec::new();
    }
    let data_dir = state.settings.data_dir.clone();
    // Reconcile whenever a managed store exists, even if the policy was since disabled: entries
    // materialized while it was enabled must still be reconciled rather than orphaned. Never
    // create the store as a side effect of a delete.
    if !data_dir.join("models").join("resolved").exists() {
        return Vec::new();
    }
    let now = sceneworks_core::time::now_unix_seconds().max(0) as u64;
    let outcome = tokio::task::spawn_blocking(move || {
        let store = ResolvedCacheStore::open(&data_dir)?;
        let mut warnings = Vec::new();
        for selector in selectors {
            let report = store.reconcile_removed_source(&selector, now)?;
            for deferred in report.deferred {
                warnings.push(format!(
                    "A cached copy of {} is still in use and will be reclaimed after it is released.",
                    selector.repository
                ));
                tracing::info!(
                    repository = %selector.repository,
                    cache_key = %deferred.cache_key,
                    hold = %deferred.hold,
                    "resolved-cache reconciliation deferred an in-use entry"
                );
            }
            for stranded in report.unmatched_unreadable {
                warnings.push(format!(
                    "A damaged resolved-cache entry ({stranded}) could not be matched to {} and was left in place.",
                    selector.repository
                ));
            }
        }
        Ok::<_, sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheError>(warnings)
    })
    .await;
    match outcome {
        Ok(Ok(warnings)) => warnings,
        Ok(Err(error)) => {
            tracing::warn!(error = %error, "resolved-cache reconciliation failed after a source delete");
            vec![format!(
                "Cached copies of this model could not be reconciled and may still occupy disk: {error}"
            )]
        }
        Err(error) => {
            tracing::warn!(error = %error, "resolved-cache reconciliation task failed");
            vec![
                "Cached copies of this model could not be reconciled and may still occupy disk."
                    .to_owned(),
            ]
        }
    }
}

/// Drop the plan-store documents and cached derivatives a deleted LINKED model owned.
///
/// Keyed on `importPlan.checkpointId` — the same discriminator the scheduler and the worker's
/// route selection use — so a model that is not plan-backed does nothing here. Never touches the
/// linked library and never removes the approved root: only this checkpoint's record, plan,
/// bindings and derivatives.
///
/// Failures become warnings rather than failing the delete: the manifest entry is already gone by
/// this point, so refusing here would leave the catalog and the plan store disagreeing with no way
/// to retry.
pub(super) async fn forget_linked_checkpoint(state: &AppState, entry: &Value) -> Vec<String> {
    let Some(checkpoint_id) = entry
        .as_object()
        .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
        .map(str::to_owned)
    else {
        return Vec::new();
    };
    let manifest = state
        .settings
        .config_dir
        .join("manifests/user.models.jsonc");
    match load_manifest_entries(state, &manifest, "models").await {
        Ok(entries)
            if entries.iter().any(|entry| {
                entry
                    .as_object()
                    .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
                    == Some(checkpoint_id.as_str())
            }) =>
        {
            return Vec::new()
        }
        Ok(_) => {}
        Err(error) => {
            return vec![format!(
            "Retained the checkpoint plan because remaining model ownership could not be read: {}",
            error.detail
        )]
        }
    }
    let data_dir = state.settings.data_dir.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore::open(&data_dir)?
            .invalidate_checkpoint(&checkpoint_id)
    })
    .await;
    match outcome {
        Ok(Ok(_)) => Vec::new(),
        Ok(Err(error)) => {
            tracing::warn!(error = %error, "forgetting a linked checkpoint failed after a model delete");
            vec![format!(
                "This model's import plan and cached derivatives could not be removed: {error}"
            )]
        }
        Err(error) => {
            tracing::warn!(error = %error, "the linked-checkpoint cleanup task failed");
            vec!["This model's import plan and cached derivatives could not be removed.".to_owned()]
        }
    }
}

pub(crate) async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    Query(query): Query<CatalogDeleteQuery>,
) -> Result<Json<Value>, ApiError> {
    let permanent = query.permanent.unwrap_or(false);
    let catalogs = JobCatalogSnapshot::default();
    let model = catalogs
        .models(&state)
        .await?
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .cloned()
        .ok_or_else(|| model_not_found("Model not found"))?;
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    // Peek (not remove) the manifest entry so that if moving the files to the OS
    // trash fails we can leave the catalog untouched and prompt for confirmation.
    let manifest_entry = load_manifest_entries(&state, &manifest_path, "models")
        .await?
        .into_iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id.as_str()));
    let cleanup_source = manifest_entry.as_ref().unwrap_or(&model);
    // Keep a cross-process source guard through the actual trash/delete operation, not merely a
    // preflight. Dense imports have no derivative lease, but are just as in-use as normalized ones.
    let source_guard = cleanup_source
        .as_object()
        .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
        .map(|checkpoint_id| {
            sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(
                &state.settings.data_dir,
            )
            .lock_source_for_removal(checkpoint_id)
        })
        .transpose()
        .map_err(|error| ApiError {
            status: StatusCode::CONFLICT,
            detail: error.to_string(),
            context: None,
            code: Some("checkpoint_source_in_use"),
        })?;
    let allowed_roots = vec![
        state.settings.data_dir.join("models"),
        sceneworks_core::hf_home::model_source_library(&state.settings.data_dir)
            .root()
            .to_owned(),
    ];
    let removal = match remove_whole_model_artifacts(
        catalogs.models(&state).await?,
        &model_id,
        cleanup_source,
        &state.settings.data_dir,
        &allowed_roots,
        permanent,
    )
    .await
    {
        Ok(removal) => {
            if !removal.removed_paths.is_empty() {
                // Invalidate immediately after the filesystem mutation. Later
                // trash-failure and manifest/error returns must not leave the
                // warm install-state snapshot describing paths already moved.
                state.model_catalog_cache.invalidate();
            }
            removal
        }
        Err(error) => {
            // Artifact removal is intentionally incremental. An inspection or
            // unlink error may follow successful earlier removals whose result
            // cannot be recovered from the error, so conservatively refresh.
            state.model_catalog_cache.invalidate();
            return Err(error);
        }
    };
    drop(source_guard);
    // Some owned files could not reach the OS trash and nothing was permanently
    // deleted. Leave the registry entry in place and ask the client to confirm.
    if !permanent && !removal.trash_failed_paths.is_empty() {
        return Ok(Json(json!({
            "id": model_id,
            "kind": "model",
            "trashUnavailable": true,
            "trashFailedPaths": removal.trash_failed_paths,
            "removedManifestEntry": false,
            "removedLocalArtifacts": !removal.removed_paths.is_empty(),
            "removedPaths": removal.removed_paths,
            "retainedPaths": removal.retained_paths,
        })));
    }
    let removed_entry =
        remove_catalog_manifest_entry(&state, &manifest_path, "models", &model_id).await?;
    if removed_entry.is_none() && removal.removed_paths.is_empty() {
        return Err(ApiError::bad_request(
            "Built-in model catalog entries are read-only unless local files are installed",
        ));
    }
    let mut warnings =
        catalog_delete_warnings(&state, "model", &model_id, None, Some(&catalogs)).await?;
    // A plan-backed (linked) model's bytes are not ours, but its plan documents and derivatives
    // are. Without this the manifest entry went away and the record, plan, bindings and every
    // cached derivative stayed behind forever — reachable by nothing, reclaimable by nothing.
    // E6 still holds exactly: this drops SceneWorks-owned documents, never a library file, and
    // never the approved root itself (other checkpoints under it stay valid).
    warnings.extend(forget_linked_checkpoint(&state, cleanup_source).await);
    // The source is gone; reclaim (or surface) every resolved-cache entry that depended on it.
    warnings.extend(
        reconcile_deleted_source(
            &state,
            owned_source_repositories(cleanup_source)
                .into_iter()
                .map(|repository| {
                    sceneworks_core::model_artifacts::resolved_cache::SourceLifecycleSelector {
                        repository,
                        revision: None,
                        tier: None,
                    }
                })
                .collect(),
        )
        .await,
    );
    // ---- checkpoint teardown (epic 20398) ---------------------------------------------------
    // `remove_whole_model_artifacts` removed the BYTES. A plan-backed entry also owns documents
    // under `<data>/checkpoints/` — catalog record, plan, bindings — which that sweep never sees.
    // Left behind for a MANAGED install they make the id permanently un-reimportable: the manifest
    // entry is gone, so this route's `existing_ids` check passes on the next import, and the
    // worker's `ManagedIngest::begin` then refuses forever with `InstallIdTaken` naming a path that
    // no longer exists.
    //
    // Deliberately one block that discriminates on ownership, because the two ownerships need
    // opposite treatment and only ONE of them is this story's.
    let plan_checkpoint_id = cleanup_source
        .as_object()
        .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
        .map(str::to_owned);
    match plan_checkpoint_id.as_deref().map(|checkpoint_id| {
        (
            checkpoint_id,
            sceneworks_core::checkpoint_plan_store::managed_install_id(checkpoint_id),
        )
    }) {
        // MANAGED: SceneWorks owns the install tree, so the record, plan and bindings go with it.
        Some((checkpoint_id, Some(install_id))) => {
            let data_dir = state.settings.data_dir.clone();
            let owned_install_id = install_id.to_owned();
            let owned_checkpoint_id = checkpoint_id.to_owned();
            // Through the DERIVATIVE store, not `CheckpointPlanStore::remove_managed` directly:
            // the plan store's own removal drops the record and `remove_dir_all`s the install tree
            // with nothing asking about derivatives first, so a leased or pinned derivative was
            // orphaned — its source bytes and plan gone, and its `derivedFrom` checkpoint id (the
            // only handle any later invalidation has) no longer resolving to anything. The
            // derivative store removes derivatives FIRST and refuses the whole teardown if one is
            // still in use, which is the same ordering the linked path above already had
            // (sc-20651 feature-end review).
            //
            // But a REFUSAL cannot be honoured HERE, and only here: by this point
            // `remove_whole_model_artifacts` has already deleted the install bytes and the manifest
            // entry is already gone. Holding the plan record back for the derivative's sake would
            // therefore strand the record with no bytes under it, and `ManagedIngest::begin` would
            // refuse this install id FOREVER with `InstallIdTaken` naming a path that no longer
            // exists — the model becomes permanently un-reimportable. So the record is cleared
            // regardless, and the derivative that could not be reclaimed is NAMED in a warning:
            // it is orphaned either way once the bytes are gone, and a named orphan is one the user
            // can find and evict, while a silent one is not. Every other caller of
            // `remove_managed_install` still gets the refuse-first behaviour, which is correct for
            // them: their bytes are still there to hold.
            let outcome = tokio::task::spawn_blocking(move || {
                managed_checkpoint_teardown(&data_dir, &owned_install_id, &owned_checkpoint_id)
            })
            .await;
            let report = match outcome {
                Ok(report) => report,
                Err(error) => ManagedTeardownReport {
                    failure: Some(error.to_string()),
                    orphaned_derivatives: Vec::new(),
                    fallback_failure: Some(error.to_string()),
                },
            };
            // The files are already gone; a failure here cannot un-delete the model, so it is
            // surfaced rather than raised. The warning names the checkpoint because that id is the
            // only handle on the stranded record.
            if let Some(failure) = report.failure {
                tracing::warn!(
                    checkpoint_id = %checkpoint_id,
                    error = %failure,
                    orphaned_derivatives = report.orphaned_derivatives.len(),
                    cleared = report.fallback_failure.is_none(),
                    "managed checkpoint teardown failed after a model delete"
                );
                if let Some(fallback_failure) = report.fallback_failure {
                    warnings.push(format!(
                        "The managed checkpoint {checkpoint_id} could not be removed ({fallback_failure}); re-importing this model id may be refused until it is cleared."
                    ));
                } else {
                    let orphans = if report.orphaned_derivatives.is_empty() {
                        "none".to_owned()
                    } else {
                        report.orphaned_derivatives.join(", ")
                    };
                    warnings.push(format!(
                        "The managed checkpoint {checkpoint_id} was removed, but its cached derivatives could not be reclaimed ({failure}); orphaned resolved-cache entries: {orphans}. Unpin or release them to free that space."
                    ));
                }
            }
        }
        // LINKED (or any non-managed checkpoint id): the bytes are the user's own library copy,
        // which this route never owned, so `remove_managed` must not be reached — it would be the
        // wrong operation on the wrong tree. Invalidating a linked entry's plan and the derivatives
        // compiled from it is sc-20635's work; this arm is where it lands.
        Some((_, None)) => {}
        None => {}
    }
    let policy = if removed_entry.is_some() {
        "Removed the model registry entry and SceneWorks-owned local model files."
    } else {
        "Built-in model catalog entries are retained; SceneWorks-owned local model files were removed."
    };
    Ok(Json(json!({
        "id": model_id,
        "kind": "model",
        "trashed": !permanent,
        "removedManifestEntry": removed_entry.is_some(),
        "removedLocalArtifacts": !removal.removed_paths.is_empty(),
        "removedPaths": removal.removed_paths,
        "retainedPaths": removal.retained_paths,
        "warnings": warnings,
        "policy": policy,
    })))
}

/// What the managed checkpoint teardown did after a model delete, when it did not simply succeed.
pub(super) struct ManagedTeardownReport {
    /// The refusal that stopped the derivatives-first teardown. `None` when it succeeded outright,
    /// which is the only case that produces no warning.
    pub(super) failure: Option<String>,
    /// The derivative cache keys still present after that refusal — the entries the delete could
    /// not reclaim, named so the user can find them.
    pub(super) orphaned_derivatives: Vec<String>,
    /// The record-clearing fallback's OWN refusal. `Some` means the plan record really is stranded
    /// and the install id may stay un-reimportable, which is a materially different warning.
    pub(super) fallback_failure: Option<String>,
}

/// Tear down a managed install after its bytes and its manifest entry are already gone.
///
/// Derivatives first (so a derivative in use is reclaimed cleanly when it can be), then — if that
/// refuses — the plan record UNCONDITIONALLY, because leaving it behind is what poisons the install
/// id forever. Blocking throughout; callers must run it off the async runtime.
pub(super) fn managed_checkpoint_teardown(
    data_dir: &std::path::Path,
    install_id: &str,
    checkpoint_id: &str,
) -> ManagedTeardownReport {
    use sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore;

    let store = CheckpointDerivativeStore::open(data_dir);
    let failure = match &store {
        Ok(store) => match store.remove_managed_install(install_id) {
            Ok(_) => {
                return ManagedTeardownReport {
                    failure: None,
                    orphaned_derivatives: Vec::new(),
                    fallback_failure: None,
                }
            }
            Err(error) => error.to_string(),
        },
        Err(error) => error.to_string(),
    };
    let orphaned_derivatives = store
        .as_ref()
        .map(|store| surviving_derivatives(store, checkpoint_id))
        .unwrap_or_default();
    let fallback_failure =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data_dir)
            .remove_managed(install_id)
            .err()
            .map(|error| error.to_string());
    ManagedTeardownReport {
        failure: Some(failure),
        orphaned_derivatives,
        fallback_failure,
    }
}

/// The derivative cache keys still recorded as produced from `checkpoint_id`.
///
/// Read AFTER a refused teardown, so it names exactly the entries that were not reclaimed. An
/// enumeration failure yields an empty list rather than an error: this only enriches a warning that
/// is already being raised, and losing the warning would be worse than losing the names.
pub(super) fn surviving_derivatives(
    store: &sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore,
    checkpoint_id: &str,
) -> Vec<String> {
    store
        .cache()
        .enumerate()
        .map(|entries| {
            entries
                .into_iter()
                .filter_map(|summary| {
                    let metadata = summary.metadata?;
                    (metadata.production.is_derived()
                        && metadata.derived_from.as_deref() == Some(checkpoint_id))
                    .then_some(metadata.cache_key)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The file scopes `model` OWNS inside `repo` — the union of its PRIMARY (non-co-requisite) download
/// entries that point at `repo`. Co-requisite rows are deliberately excluded; see the comment on the
/// filter below for why that is what makes this "owned" rather than merely "referenced".
///
/// `None` when ANY of those primaries declares no `files` filter: that is a claim on the WHOLE repo,
/// which cannot be expressed as a scoped removal, so the caller keeps the blanket path removal rather
/// than silently reclaiming less than the user asked for.
pub(super) fn model_repo_file_scopes(model: &Value, repo: &str) -> Option<Vec<String>> {
    let mut scopes = Vec::new();
    for download in model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        // Co-requisites are EXCLUDED, and that is what makes this "owned" rather than "referenced".
        // A co-requisite may be shared by several models and is never removed by a model delete
        // (the `coRequisite` schema note says so outright). sc-19573 made this load-bearing: both
        // MiniMax-H3 entries now co-require the OTHER's DiT partition from the shared repo, so
        // counting co-requisite rows here would make an entry claim its sibling's subtree as its own
        // — and, symmetrically in `other_entries_repo_file_scopes`, retain its own. The two sets
        // would cover everything, `selected && !retained` would never hold, and a whole-model delete
        // would silently reclaim nothing. `delete_model_variant`'s `retained_files` already applies
        // exactly this filter for exactly this reason.
        .filter(|entry| !is_co_requisite_download(entry))
        .filter(|entry| entry.get("repo").and_then(Value::as_str) == Some(repo))
    {
        let files = string_array_field(download, "files");
        if files.is_empty() {
            return None;
        }
        for file in files {
            if !scopes.contains(&file) {
                scopes.push(file);
            }
        }
    }
    (!scopes.is_empty()).then_some(scopes)
}

/// The file scopes every catalog entry OTHER than `model_id` claims inside `repo` (sc-19078).
///
/// Thirteen catalog groups put two or more entries in ONE Hugging Face repo, and a whole-model delete
/// resolves that repo's cache path ([`model_artifact_paths`]) — so removing one entry took the SIBLING
/// entry's bytes with it. For most of those groups the two entries name the same `files`, so the
/// removal at least matched what both wanted. MiniMax-H3 is the group where it becomes destructive:
/// `minimax_h3` owns `{tier}/transformer` and `minimax_h3_ref` owns `{tier}/transformer_ref` inside
/// `SceneWorks/minimax-h3-mlx` — DIFFERENT weights, up to 66.3 GB per tier — so deleting the
/// text-to-video entry wiped an installed reference model the user never asked to remove.
///
/// Co-requisite rows are included: a sibling's shared component living in the same repo is still bytes
/// that sibling needs. Nothing here is conditioned on the sibling being INSTALLED — an installed
/// sibling is exactly the case that matters, and for a sibling that is absent every one of these
/// patterns matches no file on disk, so retaining them costs the delete nothing.
///
/// The two kinds are returned SEPARATELY, because the caller may only ever subtract from one of them.
/// See [`SiblingRepoScopes`].
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct SiblingRepoScopes {
    /// Scopes a sibling entry claims as its OWN primary weights. These are the sc-19078 subject and
    /// are NEVER subtracted from: the sibling is a separate installed model, and unlinking bytes it
    /// names as its own primary download is precisely the data loss this function exists to prevent.
    pub(super) primaries: Vec<String>,
    /// Scopes a sibling entry claims only as a CO-REQUISITE — a shared component, or (MiniMax-H3)
    /// the deleted entry's own partition that the sibling's engine also opens. Overlap with the
    /// deleted entry's own primaries is subtracted from THIS half only, so that "delete this model"
    /// still frees the model's own weights instead of no-opping.
    pub(super) co_requisites: Vec<String>,
}

impl SiblingRepoScopes {
    pub(super) fn is_empty(&self) -> bool {
        self.primaries.is_empty() && self.co_requisites.is_empty()
    }

    /// The retained set for [`remove_tier_artifacts`]: every sibling primary, untouched, plus the
    /// co-requisite scopes that do NOT overlap `own_files`.
    ///
    /// Subtracting the overlap from the co-requisite half ONLY is the whole point of the split
    /// (sc-19573 review). Subtracting it from the union instead collapses the retained set to `[]`
    /// for every group whose sibling names the SAME primary `files` — `flux_dev`/`pulid_flux_dev`,
    /// `z_image_turbo`/`z_image_edit`, `bernini`/`bernini_image`, `realvisxl`/`instantid_realvisxl`,
    /// `qwen_image_edit_2511`/`_lightning`, `ideogram_4`/`ideogram_4_turbo` — and strips the shared
    /// text-encoder/VAE out of the `anima_*` trio. `remove_tier_artifacts`'s `selected && !retained`
    /// would then unlink the sibling's blobs with `permanent=true`: tens of GB, unrecoverable without
    /// re-download, i.e. exactly the sc-19078 defect re-introduced.
    pub(super) fn retained_files(&self, own_files: &[String]) -> Vec<String> {
        let mut retained = self.primaries.clone();
        for file in &self.co_requisites {
            if !own_files.contains(file) && !retained.contains(file) {
                retained.push(file.clone());
            }
        }
        retained
    }
}

pub(super) fn other_entries_repo_file_scopes(
    catalog: &[Value],
    model_id: &str,
    repo: &str,
) -> SiblingRepoScopes {
    let mut scopes = SiblingRepoScopes::default();
    for entry in catalog
        .iter()
        .filter(|entry| entry.get("id").and_then(Value::as_str) != Some(model_id))
    {
        for download in entry
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|download| download.get("repo").and_then(Value::as_str) == Some(repo))
        {
            let bucket = if is_co_requisite_download(download) {
                &mut scopes.co_requisites
            } else {
                &mut scopes.primaries
            };
            for file in string_array_field(download, "files") {
                if !bucket.contains(&file) {
                    bucket.push(file);
                }
            }
        }
    }
    scopes
}

/// Remove a whole model's owned artifacts for [`delete_model`], keeping a shared-repo sibling's bytes.
///
/// The default path is unchanged: every path in [`model_artifact_paths`] is removed wholesale, which
/// is right for the ~80 catalog entries that own their download repo outright. When the primary repo is
/// ALSO claimed by another catalog entry, the repo's two storage locations (the app-managed mirror dir
/// and the Hugging Face hub cache) are removed SCOPED instead — via the same
/// [`remove_tier_artifacts`] machinery the per-tier delete uses, with the sibling's declared files as
/// the retained set — so this entry's own subtrees and their exclusive blobs go and the sibling's stay.
/// Every other artifact path (a manifest `paths.model`, an imported `source.path`) is entry-exclusive
/// and still removed wholesale.
///
/// An entry that declares NO file scope inside a shared repo ([`model_repo_file_scopes`] → `None`)
/// keeps the blanket removal: it claims the whole repo, so there is no honest narrower scope, and
/// today's behavior is preserved rather than quietly reclaiming nothing.
pub(super) async fn remove_whole_model_artifacts(
    catalog: &[Value],
    model_id: &str,
    cleanup_source: &Value,
    data_dir: &FsPath,
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<ArtifactRemoval, ApiError> {
    let default_repo = model_download(cleanup_source).and_then(|download| {
        download
            .get("repo")
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    let mut removal = remove_default_repo_artifacts(
        catalog,
        model_id,
        cleanup_source,
        default_repo.as_deref(),
        data_dir,
        allowed_roots,
        permanent,
    )
    .await?;
    // A model may own primaries in MORE than one repository on this host: the CUDA `minimax_h3`
    // row serves q4/q8 from the rehost and its dense bf16 tier from upstream `transformer/`.
    // [`model_artifact_paths`] names only the default primary's repository, so those other tiers
    // were never reclaimed. Their repositories also hold co-requisites (here the shared component
    // floor), which a model delete never removes, so they are always removed SCOPED to this
    // entry's own primary files.
    for repo in owned_source_repositories(cleanup_source)
        .into_iter()
        .filter(|repo| default_repo.as_deref() != Some(repo.as_str()))
    {
        let Some(own_files) = model_repo_file_scopes(cleanup_source, &repo) else {
            continue;
        };
        let mut retained_files =
            other_entries_repo_file_scopes(catalog, model_id, &repo).retained_files(&own_files);
        for download in cleanup_source
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|download| {
                is_co_requisite_download(download)
                    && download.get("repo").and_then(Value::as_str) == Some(repo.as_str())
            })
        {
            for file in string_array_field(download, "files") {
                if !retained_files.contains(&file) {
                    retained_files.push(file);
                }
            }
        }
        let scoped = remove_tier_artifacts(
            huggingface_repo_cache_path(data_dir, &repo),
            Some(data_dir.join("models").join(safe_download_dir(&repo))),
            &own_files,
            &retained_files,
            allowed_roots,
            permanent,
        )
        .await?;
        removal.removed_paths.extend(scoped.removed_paths);
        removal.retained_paths.extend(scoped.retained_paths);
        removal.trash_failed_paths.extend(scoped.trash_failed_paths);
    }
    Ok(removal)
}

/// The default primary repository's share of [`remove_whole_model_artifacts`], plus every other
/// path [`model_artifact_paths`] names.
async fn remove_default_repo_artifacts(
    catalog: &[Value],
    model_id: &str,
    cleanup_source: &Value,
    default_repo: Option<&str>,
    data_dir: &FsPath,
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<ArtifactRemoval, ApiError> {
    let all_paths = model_artifact_paths(cleanup_source, data_dir);
    let shared = default_repo.and_then(|repo| {
        let siblings = other_entries_repo_file_scopes(catalog, model_id, repo);
        let own = model_repo_file_scopes(cleanup_source, repo)?;
        (!siblings.is_empty()).then(|| (repo.to_owned(), own, siblings))
    });
    let Some((repo, own_files, sibling_scopes)) = shared else {
        return remove_owned_artifacts(all_paths, allowed_roots, permanent).await;
    };
    // sc-19573 — a sibling may co-require THIS entry's own subtree, and then the retained set would
    // cover everything the delete selected: `selected && !retained` would never hold and the user's
    // explicit "delete this model" would silently reclaim zero bytes.
    //
    // Both MiniMax-H3 entries are in that shape now — each co-requires the other's DiT partition,
    // because the engine opens both on every load. Removing the overlap resolves it in the honest
    // direction: the delete does what it says, and the sibling entry's install state drops to
    // incomplete + `repairAvailable`, which is the truth (it can no longer load) and is re-fetchable
    // in one click. Retaining instead would leave a user who asked to free 56 GB with a no-op and no
    // explanation.
    //
    // The subtraction applies to the CO-REQUISITE half ONLY ([`SiblingRepoScopes::retained_files`]).
    // A sibling's PRIMARY scopes are never subtracted — six catalog groups pair two entries that name
    // the IDENTICAL primary `files` in one repo, and subtracting there would empty the retained set
    // and let this delete unlink the sibling's own weights.
    //
    // Computed AFTER the `is_empty` gate above, so an entry whose only sibling claim is an overlap
    // still takes the SCOPED path rather than falling back to the blanket whole-repo removal that
    // sc-19078 exists to prevent.
    let sibling_files = sibling_scopes.retained_files(&own_files);

    let managed_dir = data_dir.join("models").join(safe_download_dir(&repo));
    let repo_cache = huggingface_repo_cache_path(data_dir, &repo);
    // Everything that is NOT the shared repo's storage: still this entry's alone, still removed whole.
    let exclusive = all_paths
        .into_iter()
        .filter(|path| {
            path != &managed_dir && repo_cache.as_ref().is_none_or(|cache| path != cache)
        })
        .collect::<Vec<_>>();
    let mut removal = remove_owned_artifacts(exclusive, allowed_roots, permanent).await?;
    let scoped = remove_tier_artifacts(
        repo_cache,
        Some(managed_dir),
        &own_files,
        &sibling_files,
        allowed_roots,
        permanent,
    )
    .await?;
    removal.removed_paths.extend(scoped.removed_paths);
    removal.retained_paths.extend(scoped.retained_paths);
    removal.trash_failed_paths.extend(scoped.trash_failed_paths);
    Ok(removal)
}

/// Delete ONE installed quant tier of a model and reclaim its disk, leaving the other tiers
/// (and the catalog entry) intact (sc-12024, epic 8506). The counterpart to per-tier download
/// (sc-8509): a user who fetched q8 to A/B against q4 can drop the unused tier without nuking
/// the whole model. Unlike `delete_model` — which removes the whole repo dir AND the registry
/// entry — this is scoped to the tier's `files` and never touches the manifest, so the model
/// stays catalogued and re-downloadable; deleting the last remaining tier simply flips it back
/// to not-installed. Unlike `delete_model` this deletes PERMANENTLY (never the OS trash): a tier is
/// many loose HF-cache blobs + snapshot symlinks, and trashing them one-by-one drove a macOS
/// per-file permission-prompt loop ("you don't have permission to access some of the items") — and a
/// tier isn't restorable from loose trashed blobs anyway (sc-12088). Same ownership guard as
/// `delete_model` (`<data>/models` + the HF hub cache).
pub(crate) async fn delete_model_variant(
    State(state): State<AppState>,
    Path((model_id, variant)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let variant = variant.trim().to_ascii_lowercase();
    let model = require_model_by_id(&state, &model_id).await?;
    let data_dir = &state.settings.data_dir;
    let allowed_roots = vec![
        data_dir.join("models"),
        sceneworks_core::hf_home::model_source_library(data_dir)
            .root()
            .to_owned(),
    ];
    // A tier lives in one of two storage shapes. Download-matrix models (`hasVariantMatrix`) keep the
    // tier as a `files`-filtered slice of a shared HF cache repo (sc-12024); convert-at-install
    // models (Anima) keep it as a real `<converted>/<tier>/` dir emitted by one convert job
    // (sc-12025). Resolve whichever this model uses; a variant that is neither has nothing to delete.
    let derived = model_download_for_variant(&model, &variant).filter(|download| {
        sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(download)
    });
    let removal_result = if let Some(download) = derived {
        // sc-22998: a locally derived tier owns ONLY its derived snapshot. Its row's `files` are the
        // original's, which belong to the `fromVariant` tier and are never touched here.
        let snapshot = sceneworks_core::model_artifacts::artifact_selection::local_derivation(
            &download,
        )
        .and_then(|derivation| {
            sceneworks_core::model_artifacts::artifact_selection::local_derivation_snapshot_dir(
                data_dir,
                &model_id,
                &variant,
                &derivation,
            )
        })
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "Tier '{variant}' declares a malformed local derivation"
            ))
        })?;
        if !snapshot.is_dir() {
            return Err(ApiError::bad_request(format!(
                "Tier '{variant}' is not installed"
            )));
        }
        let reclaimable = converted_tier_real_bytes(&snapshot);
        remove_owned_artifacts(vec![snapshot], &allowed_roots, true)
            .await
            .map(|removal| TierRemoval {
                reclaimed_bytes: if removal.removed_paths.is_empty() {
                    0
                } else {
                    reclaimable
                },
                removed_paths: removal.removed_paths,
                retained_paths: removal.retained_paths,
                trash_failed_paths: removal.trash_failed_paths,
            })
    } else if let Some(download) = model_download_for_variant(&model, &variant) {
        let scope = tier_delete_scope(&model, &download, &variant)?;
        let repo_cache = huggingface_repo_cache_path(data_dir, &scope.repo);
        let managed_dir = Some(data_dir.join("models").join(safe_download_dir(&scope.repo)));
        // Always permanent (skip the OS trash) — see the fn doc (sc-12088).
        remove_tier_artifacts(
            repo_cache,
            managed_dir,
            &scope.files,
            &scope.retained_files,
            &allowed_roots,
            true,
        )
        .await
    } else if model_has_convert_tier(&model, &variant) {
        // Convert-at-install: the tier is a real dir under the converted MLX tree. Prefer the
        // catalog's resolved `mlxConvertedPath`; fall back to the canonical convert output dir.
        let converted = model
            .get("mlxConvertedPath")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("models").join("mlx").join(&model_id));
        // Always permanent (skip the OS trash) — see the fn doc (sc-12088).
        remove_converted_tier(converted.join(&variant), &allowed_roots, true).await
    } else {
        return Err(ApiError::bad_request(format!(
            "Model does not advertise a '{variant}' quant tier"
        )));
    };
    let removal = match removal_result {
        Ok(removal) => removal,
        Err(error) => {
            // Tier removal is also incremental (snapshot links, blobs, then
            // managed paths). Refresh conservatively if any later step errors.
            state.model_catalog_cache.invalidate();
            return Err(error);
        }
    };
    // Invalidate before every post-removal early return. A successful removal
    // may have changed installState/variantStates even when the route later
    // decides there was no independently reclaimable logical tier.
    state.model_catalog_cache.invalidate();
    let shared_logical_tier = removal.removed_paths.is_empty()
        && !removal.retained_paths.is_empty()
        && model_download_for_variant(&model, &variant).is_some();
    if removal.removed_paths.is_empty() && !shared_logical_tier {
        return Err(ApiError::bad_request(format!(
            "Tier '{variant}' is not installed"
        )));
    }
    // A single-tier delete only orphans resolved-cache entries built from that tier; entries for
    // the model's other tiers stay valid, so the selector is scoped by tier.
    let cache_warnings = reconcile_deleted_source(
        &state,
        owned_source_repositories(&model)
            .into_iter()
            .map(|repository| {
                sceneworks_core::model_artifacts::resolved_cache::SourceLifecycleSelector {
                    repository,
                    revision: None,
                    tier: Some(variant.clone()),
                }
            })
            .collect(),
    )
    .await;
    Ok(Json(json!({
        "id": model_id,
        "variant": variant,
        "kind": "model-variant",
        "warnings": cache_warnings,
        // Permanent delete: no OS trash, no undo (sc-12088).
        "trashed": false,
        // A tier delete NEVER removes the registry entry: the model stays in the catalog so the
        // tier can be re-downloaded. Emitted false so the web keeps the model card in place.
        "removedManifestEntry": false,
        "removedLocalArtifacts": !removal.removed_paths.is_empty(),
        "reclaimedBytes": removal.reclaimed_bytes,
        "reclaimedLabel": format_bytes(removal.reclaimed_bytes),
        "removedPaths": removal.removed_paths,
        "retainedPaths": removal.retained_paths,
    })))
}

/// What one per-tier delete removes: the repo it lives in, the globs that select its files, and
/// the globs a sibling tier in the SAME repo still owns.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TierDeleteScope {
    pub(crate) repo: String,
    pub(crate) files: Vec<String>,
    pub(crate) retained_files: Vec<String>,
}

/// Whether a tier row with NO `files` scope is still reclaimable on its own (sc-24112): no other
/// download row of the same model — tier, alternate or co-requisite — names its `repo`, so that
/// repo's whole snapshot IS this tier. `qwen_image_2_1`'s bf16 tier is the released upstream
/// snapshot, the only row on `Qwen/Qwen-Image-2.1`, while its q8/q4 live in the SceneWorks re-host;
/// deleting it can strand nothing. A scope-less row on a repo a sibling shares is still the model,
/// not a slice of it, and stays refused.
pub(crate) fn is_sole_repo_tier(model: &Value, download: &Value) -> bool {
    // Only a DECLARED tier: an untagged single-download row is the "default" pseudo-variant, which
    // the per-tier route cannot address at all.
    let tagged = download
        .get("variant")
        .and_then(Value::as_str)
        .is_some_and(|variant| !variant.trim().is_empty());
    let Some(repo) = download
        .get("repo")
        .and_then(Value::as_str)
        .filter(|_| tagged)
    else {
        return false;
    };
    let sharers = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| entry.get("repo").and_then(Value::as_str) == Some(repo))
        .count();
    sharers == 1
}

/// Scope a per-tier delete of `variant` (its catalog row is `download`).
///
/// A row with a `files` scope deletes that slice and protects every path a sibling logical tier in
/// the same repo still references (Mage-Flow's overlapping load-time tiers reclaim zero bytes
/// rather than corrupting the snapshot the remaining choices use). A row with NO scope is refused
/// ("delete the whole model instead") unless it is the sole row on its repo
/// ([`is_sole_repo_tier`]), in which case the whole repo snapshot is the tier and every file in it
/// goes.
pub(crate) fn tier_delete_scope(
    model: &Value,
    download: &Value,
    variant: &str,
) -> Result<TierDeleteScope, ApiError> {
    let repo = download
        .get("repo")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let files = string_array_field(download, "files");
    if files.is_empty() {
        if is_sole_repo_tier(model, download) {
            return Ok(TierDeleteScope {
                repo,
                files: vec!["*".to_owned()],
                retained_files: Vec::new(),
            });
        }
        return Err(ApiError::bad_request(format!(
            "Tier '{variant}' has no file scope; delete the whole model instead"
        )));
    }
    let retained_files = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| {
            !is_co_requisite_download(entry)
                // sc-22998: a locally derived tier's `files` name the ORIGINAL it is derived from;
                // it does not own them, so deleting the original must not "retain" them for it.
                && !sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(
                    entry,
                )
                && entry.get("repo").and_then(Value::as_str) == Some(repo.as_str())
                && entry
                    .get("variant")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.eq_ignore_ascii_case(variant))
        })
        .flat_map(|entry| string_array_field(entry, "files"))
        .collect::<Vec<_>>();
    Ok(TierDeleteScope {
        repo,
        files,
        retained_files,
    })
}
/// Result of removing a single quant tier's on-disk artifacts (sc-12024).
#[derive(Default)]
pub(crate) struct TierRemoval {
    /// Paths (tier symlinks/files + their exclusive blobs) moved to the OS trash or unlinked.
    pub(crate) removed_paths: Vec<String>,
    /// Paths left in place because they are not inside a SceneWorks-owned root.
    pub(crate) retained_paths: Vec<String>,
    /// Owned paths that could NOT be moved to the OS trash (recycle bin disabled, unsupported
    /// volume, …). Nothing was deleted for these; the caller prompts before a permanent delete.
    pub(crate) trash_failed_paths: Vec<String>,
    /// Bytes actually reclaimed — the summed size of the data-bearing files/blobs removed.
    pub(crate) reclaimed_bytes: u64,
}

/// Remove ONE quant tier's artifacts from a download-matrix model's storage, reclaiming disk.
///
/// A download-matrix model keeps every tier (bf16/q8/q4) in ONE shared Hugging Face hub-cache
/// repo: the real bytes live in `blobs/<etag>` and each tier's files are relative SYMLINKS into
/// `blobs/` (`download_snapshot_into_cache`, crates/sceneworks-worker/src/downloads.rs). Deleting
/// the tier's snapshot symlinks alone frees nothing — the blobs behind them must go too. This
/// walks every snapshot revision under the repo cache (and the app-managed mirror dir, where a
/// turnkey install lands real files), selects the files matching the tier's `files` globs, and
/// removes those directory entries PLUS the blobs they resolve to — while PROTECTING any blob
/// still referenced by a retained tier (a shared etag). Emptied tier/snapshot dirs, and the whole
/// repo cache dir once no tier's payload remains, are pruned (best-effort; only ever unlinks EMPTY
/// dirs, so a surviving tier is never touched). `reclaimed_bytes` reflects only what actually left
/// disk. An empty `tier_files` is a no-op — the caller must never scope a delete to "everything".
pub(crate) async fn remove_tier_artifacts(
    repo_cache: Option<PathBuf>,
    managed_dir: Option<PathBuf>,
    tier_files: &[String],
    retained_files: &[String],
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<TierRemoval, ApiError> {
    if tier_files.is_empty() {
        return Ok(TierRemoval::default());
    }
    // The directories to scan: every snapshot revision under the HF repo cache, plus the managed
    // mirror dir. Each is scanned independently and files are matched RELATIVE to their scan dir.
    let mut scan_dirs: Vec<PathBuf> = Vec::new();
    if let Some(repo_cache) = repo_cache.as_ref() {
        if huggingface_repo_cache_exists(repo_cache) {
            scan_dirs.extend(huggingface_snapshot_dirs(repo_cache));
        }
    }
    if let Some(managed_dir) = managed_dir.as_ref() {
        if managed_dir.is_dir() {
            scan_dirs.push(managed_dir.clone());
        }
    }

    // Split every file under the scan dirs into this tier's entries vs the retained rest. A
    // retained file's real data (blob) must survive even if a tier symlink shares its etag.
    let mut tier_entries: Vec<PathBuf> = Vec::new();
    let mut retained_reals: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    for dir in &scan_dirs {
        for rel in snapshot_files(dir) {
            let abs = dir.join(&rel);
            let selected = tier_files
                .iter()
                .any(|pattern| pattern_matches(pattern, &rel));
            let retained = retained_files
                .iter()
                .any(|pattern| pattern_matches(pattern, &rel));
            if selected && !retained {
                tier_entries.push(abs);
            } else {
                if let Ok(real) = tokio::fs::canonicalize(&abs).await {
                    retained_reals.insert(real);
                }
            }
        }
    }

    let mut manifest_retained = Vec::new();
    for dir in &scan_dirs {
        for rel in snapshot_files(dir) {
            if tier_files
                .iter()
                .any(|pattern| pattern_matches(pattern, &rel))
                && retained_files
                    .iter()
                    .any(|pattern| pattern_matches(pattern, &rel))
            {
                manifest_retained.push(dir.join(rel).display().to_string());
            }
        }
    }

    // Build the ordered removal plan: unlink the tier's directory ENTRIES first (so a symlink
    // still resolves to its blob for the ownership check), THEN the blobs those symlinks resolve
    // to (skipping any shared with a retained tier). `data_sizes` records the byte size of every
    // data-bearing path so reclaimed bytes reflect exactly what leaves disk.
    let mut ordered: Vec<PathBuf> = Vec::new();
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let mut data_sizes: std::collections::HashMap<PathBuf, u64> = std::collections::HashMap::new();
    for entry in &tier_entries {
        // A real file (managed mirror) IS its own data holder; a symlink's bytes live in its blob.
        if let Ok(link_meta) = tokio::fs::symlink_metadata(entry).await {
            if !link_meta.file_type().is_symlink() && link_meta.is_file() {
                data_sizes.insert(entry.clone(), link_meta.len());
            }
        }
        if seen.insert(entry.clone()) {
            ordered.push(entry.clone());
        }
    }
    for entry in &tier_entries {
        // Resolve the snapshot entry to the blob it references and remove that blob unless a retained
        // tier shares it. On macOS/Linux the entry is a SYMLINK so `canonicalize` yields the blob. On
        // Windows the HF cache uses HARDLINKS, which `canonicalize` does NOT resolve to a different
        // path (`real == entry`), so the blob's second name under blobs/ is not reclaimed here — the
        // Windows hardlink reverse-map is tracked in sc-12038. macOS/Linux (primary targets) reclaim
        // fully; the unix `variant_delete_tests` cover it.
        if let Ok(real) = tokio::fs::canonicalize(entry).await {
            if &real != entry && !retained_reals.contains(&real) {
                if let Ok(meta) = tokio::fs::metadata(&real).await {
                    data_sizes.entry(real.clone()).or_insert(meta.len());
                }
                if seen.insert(real.clone()) {
                    ordered.push(real);
                }
            }
        }
    }

    let mut removal = remove_owned_artifacts(ordered, allowed_roots, permanent).await?;
    removal.retained_paths.extend(manifest_retained);
    let reclaimed_bytes = removal
        .removed_paths
        .iter()
        .filter_map(|path| data_sizes.get(FsPath::new(path)))
        .sum();
    let tier_removal = TierRemoval {
        removed_paths: removal.removed_paths,
        retained_paths: removal.retained_paths,
        trash_failed_paths: removal.trash_failed_paths,
        reclaimed_bytes,
    };

    // Best-effort tidy once the removal itself succeeded: drop now-empty tier/snapshot dirs, and
    // the whole repo cache dir once no tier's payload remains (otherwise only the tiny refs/
    // skeleton would linger). Only ever removes EMPTY dirs.
    if tier_removal.trash_failed_paths.is_empty() {
        if let Some(repo_cache) = repo_cache.as_ref() {
            prune_empty_repo_cache(repo_cache).await;
        }
        if let Some(managed_dir) = managed_dir.as_ref() {
            remove_empty_dirs(managed_dir).await;
        }
    }

    Ok(tier_removal)
}

/// Recursively remove empty subdirectories under `dir`, then `dir` itself if it ends up empty.
/// Best-effort (ignores errors) and only ever unlinks EMPTY directories, so a sibling tier's
/// surviving files can never be removed by it.
pub(super) async fn remove_empty_dirs(dir: &FsPath) {
    let dir = dir.to_path_buf();
    let _ = tokio::task::spawn_blocking(move || remove_empty_dirs_on_disk(&dir)).await;
}

pub(super) fn remove_empty_dirs_on_disk(dir: &FsPath) {
    let mut pending = vec![(dir.to_path_buf(), false)];
    while let Some((path, visited)) = pending.pop() {
        if visited {
            let _ = std::fs::remove_dir(&path);
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        let children = entries
            .map_while(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        pending.push((path, true));
        for child in children.into_iter().rev() {
            if child.is_dir() {
                pending.push((child, false));
            }
        }
    }
}

/// Prune a download-matrix repo cache after tier removal. The whole walk is
/// blocking and best effort; no filesystem work runs on the async worker.
pub(super) async fn prune_empty_repo_cache(repo_cache: &FsPath) {
    let repo_cache = repo_cache.to_path_buf();
    let _ = tokio::task::spawn_blocking(move || prune_empty_repo_cache_on_disk(&repo_cache)).await;
}

pub(super) fn prune_empty_repo_cache_on_disk(repo_cache: &FsPath) {
    let Some(source_root) = repo_cache.parent() else {
        return;
    };
    let Ok(source_library) =
        sceneworks_core::model_artifacts::ArtifactSourceLibrary::new(source_root)
    else {
        return;
    };
    let Ok(snapshots_root) = source_library.snapshots_root_from_repository_root(repo_cache) else {
        return;
    };
    remove_empty_dirs_on_disk(&snapshots_root);
    remove_empty_dirs_on_disk(&repo_cache.join("blobs"));
    let has_blobs = std::fs::read_dir(repo_cache.join("blobs"))
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);
    let has_snapshot_files = huggingface_snapshot_dirs(repo_cache)
        .iter()
        .any(|snapshot| !snapshot_files(snapshot).is_empty());
    if !has_blobs && !has_snapshot_files {
        let _ = std::fs::remove_dir_all(repo_cache);
    }
}

/// Whether `model` advertises `variant` as a convert-at-install tier — i.e. it appears in the
/// catalog's `mlxTiers` (the on-disk convert-output tiers of a converted MLX model, sc-10730).
pub(super) fn model_has_convert_tier(model: &Value, variant: &str) -> bool {
    model
        .get("mlxTiers")
        .and_then(Value::as_array)
        .is_some_and(|tiers| {
            tiers
                .iter()
                .filter_map(Value::as_str)
                .any(|tier| tier.eq_ignore_ascii_case(variant))
        })
}

/// Remove ONE convert-at-install tier dir (`<converted>/<tier>/`) and reclaim its disk (sc-12025).
/// Convert-at-install models (Anima) emit every tier from one convert job as a real per-tier dir
/// holding a packed DiT plus SYMLINKS to the shared dense TE/VAE (whose targets live OUTSIDE the tier
/// dir). Removing the tier dir frees only the packed DiT + the symlink entries — never the shared
/// source, which the other tiers still reference — so `reclaimed_bytes` counts only the real
/// (non-symlink) files under the tier. When this was the LAST tier with weights, the whole converted
/// dir is dropped so the model cleanly reverts to "needs conversion" rather than lingering as a bare
/// `model_index.json` marker.
pub(super) async fn remove_converted_tier(
    tier_dir: PathBuf,
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<TierRemoval, ApiError> {
    if !tier_dir.is_dir() {
        return Ok(TierRemoval::default());
    }
    let reclaimable = converted_tier_real_bytes(&tier_dir);
    let removal = remove_owned_artifacts(vec![tier_dir.clone()], allowed_roots, permanent).await?;
    let removed = removal
        .removed_paths
        .iter()
        .any(|path| FsPath::new(path) == tier_dir);
    // If no sibling tier retains weights, drop the whole converted dir (marker included) so the model
    // reverts to a clean not-converted state instead of a bare marker. Best-effort.
    if removed && removal.trash_failed_paths.is_empty() {
        if let Some(converted) = tier_dir.parent() {
            let any_tier_left = ["bf16", "q8", "q4"]
                .iter()
                .any(|tier| tier_subdir_has_weights(&converted.join(tier)));
            if !any_tier_left {
                let _ = tokio::fs::remove_dir_all(converted).await;
            }
        }
    }
    Ok(TierRemoval {
        removed_paths: removal.removed_paths,
        retained_paths: removal.retained_paths,
        trash_failed_paths: removal.trash_failed_paths,
        reclaimed_bytes: if removed { reclaimable } else { 0 },
    })
}

/// Sum the bytes of the REAL (non-symlink) files under a converted tier dir — the packed DiT — so a
/// tier delete reports only what it actually frees. The shared TE/VAE are symlinks to a source
/// outside the tier dir; following them would over-count disk that a tier delete does not reclaim.
pub(super) fn converted_tier_real_bytes(tier_dir: &FsPath) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![tier_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if meta.file_type().is_symlink() {
                continue; // shared TE/VAE — its target lives outside the tier dir; not reclaimed
            }
            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    total
}
