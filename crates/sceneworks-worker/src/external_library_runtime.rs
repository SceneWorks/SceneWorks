//! The worker's single pre-loader model-source guard.
//!
//! Every job dispatched by `run_utility_job` passes through [`RuntimeSourceGuard::begin`] before
//! its handler constructs a loader. The guard reads the job's generic model carriers
//! (`modelManifestEntry`, `baseModelManifestEntry`, `modelManifestEntries` — declarative manifest
//! data, never per-model wiring), reduces each to the exact requirement closure for THIS worker's
//! platform and the request's selected tier via
//! [`sceneworks_core::model_artifacts::artifact_selection`], and judges it through the one shared
//! resolver. There is deliberately no per-route or per-model availability code anywhere else in
//! this crate:
//!
//! - `local_ready` / `external_ready` proceed (an external source opens an operation-owned
//!   session whose physical identity was proven immediately before loader construction);
//! - `installed_external_unavailable` fails typed — installed state is preserved, receipts are
//!   never rewritten, and no silent re-download is attempted;
//! - `incomplete` / `missing` preserve the established on-demand download behavior of the
//!   handler (a model that was never installed keeps its existing install path).
//!
//! On a later load failure the guard re-probes the exact bound source and remaps the error to the
//! typed unavailable class only when the source is now provably absent or a different physical
//! volume — an engine defect while the source remains available is preserved verbatim, and a raw
//! loader ENOENT is never what a disconnect surfaces as.

use crate::{emit_event_value, JsonObject, Settings, WorkerError, WorkerResult};
use sceneworks_core::contracts::JobType;
use sceneworks_core::file_lock::FileLock;
use sceneworks_core::model_artifacts::artifact_selection::{
    is_co_requisite_download, requested_runtime_variant, resident_base_component_downloads,
    selected_portable_model_artifact_closure, selected_portable_requirements_for_model,
    selected_resident_base_closure_for_row,
};
#[cfg(test)]
use sceneworks_core::model_artifacts::artifact_selection::{
    selected_model_artifact_closure, selected_requirements_for_model,
};
pub(crate) use sceneworks_core::model_artifacts::artifact_selection::{
    ResidentBaseShape, PLAN_BACKED_RESIDENT_BASE_MODELS,
};
use sceneworks_core::model_artifacts::external_library::{
    resolve_model_availability, ExternalArtifactRequirement, ExternalSourceSession,
    ModelAvailability, ModelResolution,
};
use sceneworks_core::model_artifacts::local_preference::{
    hub_cache_member_destination, prefer_portable_snapshots, wait_for_portable_snapshots,
    ActiveLocalArtifacts, LocalArtifactOverlayEntry, PortableOverlayWaitCancellation,
};
use sceneworks_core::model_artifacts::portable_binding::{
    enumerate_portable_array_slots, LocalArtifactFile, LocalArtifactMember, PortableArtifactFile,
    PortableArtifactMember, PortableCheckpointPlan, PortableModelBinding, PortableRouteBinding,
    PortableSourceIdentity,
};
use sceneworks_core::model_artifacts::resolved_cache::{ResolvedCacheLease, ResolvedCacheStore};
use sceneworks_core::model_artifacts::ModelArtifactResolver;
use sceneworks_core::model_artifacts::{
    ArtifactContractError, ArtifactIdentity, ArtifactMemberRole, ArtifactSourceLibrary,
};
use sceneworks_core::portable_sources::{
    portable_aux_source_shape, required_portable_aux_source_ids, PortableAuxSourceShape,
};
use sceneworks_core::verified_source_cache::{SourceVerification, VerifiedSourceCache};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use tracing::Level;

#[derive(Debug)]
pub(crate) struct RuntimeSourceGuard {
    /// Worker-only request copy. Every user-supplied binding hint is removed before dispatch;
    /// a verified route may replace it with its frozen tier and later with pinned local paths.
    dispatch_payload: JsonObject,
    portable_scope: Option<ActiveLocalArtifacts>,
    _portable_snapshot: Option<PortableAttemptSnapshot>,
    plan_layers: Option<std::sync::Arc<VerifiedPlanLayers>>,
    verified_revision_sources: Vec<VerifiedRevisionSource>,
    resolutions: Vec<ModelResolution>,
    sessions: Vec<ExternalSourceSession>,
    /// The ONE process-wide local-tier preference scope for this job, holding exactly the
    /// `(repository, revision)` pairs the guard selected. Declared BEFORE the leases so field drop
    /// order stops serving local paths first and only then releases the leases protecting those
    /// bytes — the reverse of acquisition.
    ///
    /// One scope rather than one per model is load-bearing, not tidiness: the coverage decision is
    /// made across the WHOLE job, so a per-model scope could install a bundle for one model that a
    /// sibling model must not be served from, and installing them one at a time leaves a narrower
    /// bundle serving the process while a later model falls back.
    local_scope: Option<ActiveLocalArtifacts>,
    /// Cross-process cache leases (one per locally served artifact) held for the whole job. Each
    /// holds the entry's shared artifact lock, so an evictor taking that lock exclusively — even
    /// non-blockingly, and even when it re-verifies under the lock — cannot remove bytes a load is
    /// reading. Released on drop; `finish_success` crosses the promotion boundary first.
    cache_leases: Vec<ResolvedCacheLease>,
    /// sc-19706: the closures this job served from the SOURCE tier, recorded here so
    /// [`Self::finish_success`] can hand them to the promotion intake without doing any work.
    ///
    /// This is the whole of the trigger's job-path cost: a `Vec` built from values the guard had
    /// already computed for admission. Building the promotion candidate needs the source library
    /// walked, every closure member stat-ed and canonicalized, and a reservation taken — all of it
    /// belongs on the idle drain (`crate::resolved_cache_promotion`), never in the job's critical
    /// path, so nothing here touches the filesystem.
    pending_promotions: Vec<crate::resolved_cache_promotion::PendingPromotion>,
}

impl RuntimeSourceGuard {
    #[cfg(test)]
    pub(crate) fn begin(
        job_type: &JobType,
        payload: &JsonObject,
        settings: &Settings,
    ) -> WorkerResult<Self> {
        Self::begin_with_binding(job_type, payload, settings, None)
    }

    #[cfg(test)]
    pub(crate) fn begin_with_binding(
        job_type: &JobType,
        payload: &JsonObject,
        settings: &Settings,
        binding: Option<&PortableRouteBinding>,
    ) -> WorkerResult<Self> {
        Self::begin_with_binding_cancelable(job_type, payload, settings, binding, None)
    }

    pub(crate) fn begin_with_binding_cancelable(
        job_type: &JobType,
        payload: &JsonObject,
        settings: &Settings,
        binding: Option<&PortableRouteBinding>,
        cancellation: Option<&PortableOverlayWaitCancellation>,
    ) -> WorkerResult<Self> {
        let explicit_download = payload
            .get("modelArtifactOperation")
            .and_then(|value| value.get("kind"))
            .and_then(serde_json::Value::as_str)
            == Some("explicit_download");
        if explicit_download && !matches!(job_type, JobType::ModelDownload | JobType::LoraDownload)
        {
            return Err(WorkerError::InvalidPayload(
                "explicit-download artifact operation is invalid for this job type".to_owned(),
            ));
        }
        let model_free_dry_run = matches!(job_type, JobType::LoraTrain | JobType::ControlTraining)
            && payload.get("dryRun").and_then(serde_json::Value::as_bool) == Some(true);
        // sc-24829: a trigger-only caption job writes the trigger words and loads no captioner, so
        // the API sends it without model carriers (the identical exemption on that side).
        let model_free_trigger_only_caption = matches!(job_type, JobType::TrainingCaption)
            && payload.get("mode").and_then(serde_json::Value::as_str) == Some("triggerOnly");
        let model_free_catalog_analysis = matches!(job_type, JobType::CatalogAnalysis)
            && [
                "structuredAnalysisEnabled",
                "visionAnalysisEnabled",
                "semanticEmbeddingsEnabled",
            ]
            .iter()
            .all(|key| payload.get(*key).and_then(serde_json::Value::as_bool) == Some(false));
        if explicit_download
            || model_free_dry_run
            || model_free_trigger_only_caption
            || model_free_catalog_analysis
        {
            if binding.is_some() {
                return Err(WorkerError::InvalidPayload(
                    "a model-free operation cannot carry a model source binding".to_owned(),
                ));
            }
            return Ok(Self::none(payload));
        }

        let mut planning_payload = payload.clone();
        let mut verified_converted_slots = std::collections::BTreeSet::new();
        if binding.is_some() {
            // The claim's manifest was assembled on the API host. Its physical paths cannot
            // establish this worker's installation, including during the legacy source pass.
            let slots: Vec<_> = payload_model_slots(payload)
                .into_iter()
                .filter_map(|(slot, entry)| {
                    entry
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .map(|id| (slot, id.to_owned()))
                })
                .collect();
            for (slot, id) in slots {
                let model_binding =
                    binding.and_then(|route| route.models.iter().find(|model| model.slot == slot));
                let primary = model_binding.and_then(|model| {
                    model
                        .members
                        .iter()
                        .find(|member| member.role == ArtifactMemberRole::Primary)
                });
                let registered_id = model_binding
                    .filter(|_| primary.is_some_and(|member| matches!(member.source, PortableSourceIdentity::Digest { .. })))
                    .map(|model| {
                        sceneworks_core::model_artifacts::portable_binding::registered_primary_digest_catalog_id(
                            model,
                            &serde_json::Value::Object(payload.clone()),
                        )
                    })
                    .transpose()
                    .map_err(|error| invalid_binding(error.to_string()))?
                    .unwrap_or_else(|| id.clone());
                if let Some(local) = primary
                    .map(|member| {
                        worker_catalog_member_path(
                            settings,
                            "user.models.jsonc",
                            "models",
                            &registered_id,
                            None,
                            member,
                        )
                    })
                    .transpose()?
                    .flatten()
                {
                    if primary.is_some_and(|member| {
                        member.role == ArtifactMemberRole::Primary
                            && matches!(member.source, PortableSourceIdentity::Digest { .. })
                            && crate::portable_source_claims::is_converted_builtin_model_id(&id)
                    }) {
                        // The registered fixed output root has a valid producer receipt and
                        // exactly the frozen bytes. Its publisher download was an install-time
                        // input, not a runtime source requirement for this route.
                        verified_converted_slots.insert(slot.clone());
                    }
                    let entry = model_slot_entry_mut(&mut planning_payload, &slot)
                        .and_then(serde_json::Value::as_object_mut)
                        .ok_or_else(|| invalid_binding("worker model slot disappeared"))?;
                    entry.insert("modelPath".to_owned(), json!(local));
                    entry.insert("installedPath".to_owned(), json!(local));
                    entry
                        .entry("paths".to_owned())
                        .or_insert_with(|| json!({}))
                        .as_object_mut()
                        .ok_or_else(|| invalid_binding("worker model paths are invalid"))?
                        .insert("model".to_owned(), json!(local));
                    // Provenance is descriptive and never opened by a loader. The API host's
                    // source.path cannot decide this worker's local eligibility.
                    if let Some(source) = entry
                        .get_mut("source")
                        .and_then(serde_json::Value::as_object_mut)
                    {
                        source.remove("path");
                    }
                }
                if let Some(model_binding) = model_binding {
                    for member in model_binding.members.iter().filter(|member| {
                        member.role == ArtifactMemberRole::RequiredComponent
                            && matches!(member.source, PortableSourceIdentity::Digest { .. })
                            && !model_binding.checkpoint_plan.as_ref().is_some_and(|plan| {
                                checkpoint_plan_layer_id(member, &slot).is_some_and(|layer_id| {
                                    plan.layers.iter().any(|layer| layer.layer_id == layer_id)
                                })
                            })
                    }) {
                        let Some(component_id) = member.component_id.as_deref() else {
                            continue;
                        };
                        let key = portable_component_locator_key(&slot, component_id);
                        let Some(local) = worker_catalog_member_path(
                            settings,
                            "user.models.jsonc",
                            "models",
                            &id,
                            Some(key),
                            member,
                        )?
                        else {
                            continue;
                        };
                        let local = local
                            .to_str()
                            .ok_or_else(|| invalid_binding("worker catalog path is not UTF-8"))?;
                        let entry = model_slot_entry_mut(&mut planning_payload, &slot)
                            .and_then(serde_json::Value::as_object_mut)
                            .ok_or_else(|| invalid_binding("worker model slot disappeared"))?;
                        if !set_existing_entry_locator(entry, key, local) {
                            return Err(invalid_binding("worker component locator is absent"));
                        }
                    }
                }
            }
        }
        let model_entries = payload_model_slots(&planning_payload);
        // Fail closed on an EMPTY model carrier set for every model-backed route. Routes that
        // genuinely load no model bytes are declared in the exhaustive table below — a job type
        // absent from that allowlist cannot pass without carriers.
        if job_requires_typed_model_source(job_type) && model_entries.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "{} job carries no model manifest entries for the typed model-source guard",
                job_type.as_str()
            )));
        }

        let configured_library = sceneworks_core::hf_home::model_source_library(&settings.data_dir)
            .root()
            .to_path_buf();
        // Every VALID app-owned bundle, read once for this job. Empty when the resolved cache is
        // disabled or uninitialized, and — by construction of the provider — never containing an
        // entry that is torn, unverifiable, or in a shape the runtime cannot serve, so a partial
        // local copy can only ever move the load back to the source tier.
        let local_scan = local_resolved_artifacts(settings);
        // A bundle whose SHAPE the local tier cannot serve is the one rejection class a user can
        // act on, so it is named here rather than dropped at the scan.
        for rejection in &local_scan.rejections {
            emit_event_value(
                Level::WARN,
                json!({
                    "event": "resolved_cache_local_tier_unsupported",
                    "cacheKey": rejection.cache_key,
                    "repository": rejection.repository,
                    "revision": rejection.revision,
                    "reason": rejection.reason,
                }),
            );
        }
        let local_artifacts = local_scan.artifacts;
        // PASS 1 — resolve every model entry WITH local candidates, keeping the selected closure
        // beside each resolution. Nothing is leased and nothing is served yet: which pairs may be
        // served at all cannot be known until every entry in the job has been resolved.
        let mut plans: Vec<EntryPlan> = Vec::new();
        let mut converted_co_requirements = Vec::new();
        for (slot, entry) in &model_entries {
            let requested_variant = bound_runtime_variant(
                job_type,
                payload,
                slot,
                entry,
                binding.and_then(|route| route.models.iter().find(|model| &model.slot == slot)),
            )?;
            let has_hf_downloads = entry
                .get("downloads")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|downloads| {
                    downloads.iter().any(|download| {
                        sceneworks_core::model_artifacts::artifact_selection::is_supported_model_download(
                            download,
                        )
                    })
                });
            if !has_hf_downloads {
                // Imported/app-owned and explicitly configured external-root models are the only
                // entries outside the HF-library contract; they must prove their confinement.
                if entry_is_provably_non_hf_local(entry, settings)? {
                    // sc-22329: a plan-backed imported checkpoint is the transformer only — its
                    // resident base components (text encoder, VAE, tokenizer) load from the
                    // family's installed base model, which this entry does not carry. Give the
                    // base its own plan so it participates in local-tier selection exactly like an
                    // HF-managed entry's closure; without one, no scope is installed and the base
                    // streams from the source library on every cold load even while the resolved
                    // cache holds it. Every degradation returns `None`, which is the pre-sc-22329
                    // behavior (the route's own typed refusal still owns the not-installed case).
                    if let Some(plan) = plan_backed_resident_base_plan(
                        entry,
                        settings,
                        &configured_library,
                        &local_artifacts,
                    ) {
                        plans.push(plan);
                    }
                    continue;
                }
                return Err(WorkerError::InvalidPayload(format!(
                    "model entry '{}' names no supported artifact source and no confined local path",
                    entry
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>")
                )));
            }
            // This worker's OWN platform and the request's selected tier define the closure —
            // never the API host's platform, never a sibling variant's receipts.
            let selected = selected_portable_requirements_for_model(
                entry,
                crate::portable_runtime_backend(settings),
                std::env::consts::OS,
                requested_variant.as_deref(),
                payload,
                &settings.data_dir,
            )
            .map_err(invalid_binding)?;
            if verified_converted_slots.contains(slot) {
                // The converted output replaces the publisher primary. A co-requisite-only
                // ModelResolution is invalid (the source contract requires one primary), so
                // keep its exact required files for portable route verification and let the
                // frozen Revision members provide their own guarded source availability.
                converted_co_requirements.extend(
                    selected
                        .requirements
                        .into_iter()
                        .filter(|requirement| !requirement.is_primary),
                );
                continue;
            }
            let resolution = resolve_model_availability(
                &settings.data_dir,
                &configured_library,
                &selected.requirements,
                selected.receipt_backed,
                &local_artifacts,
            );
            plans.push(EntryPlan {
                requirements: selected.requirements,
                receipt_backed: selected.receipt_backed,
                resolution,
                resident_base: false,
            });
        }

        // PASSES 2 and 3 — decide which `(repository, revision)` pairs the whole job may serve
        // locally, take the leases behind exactly those, and re-resolve every model that did not
        // get its ENTIRE closure served. This is the only layer that can see every model a job
        // loads, and therefore the only layer that can answer the coverage question at all.
        let (mut local_scope, cache_leases) =
            select_local_tier(settings, &configured_library, &mut plans);

        // sc-19706 — the PRODUCER trigger, recorded here and fired only on success. Read AFTER
        // pass 3 so it reflects the tier that will really serve: a model that fell back from the
        // local tier is external-ready now, and its own closure is exactly what is missing from
        // the cache. Only `ExternalReady` qualifies — a local-ready model is already promoted,
        // and incomplete/missing/unavailable models have nothing installed to copy.
        let pending_promotions = crate::resolved_cache_promotion::pending_for_plans(
            settings,
            &configured_library,
            plans.iter().filter_map(|plan| {
                (plan.resolution.availability == ModelAvailability::ExternalReady)
                    .then_some(plan.requirements.as_slice())
            }),
        );

        // The job's CARRIED models decide admission; a resident base (sc-22329) may only ever ADD
        // a local-tier serve. So the two are separated here — deduped against each other, carried
        // first — and only the carried set runs the admitting loop below. A resolution a carried
        // entry also produced keeps its carried (fatal) semantics: a resident-base plan must never
        // be able to SUPPRESS a refusal a real carrier earned.
        let mut resolutions: Vec<ModelResolution> = Vec::new();
        let mut resident_base_only: Vec<ModelResolution> = Vec::new();
        for plan in plans.iter().filter(|plan| !plan.resident_base) {
            if !resolutions.contains(&plan.resolution) {
                resolutions.push(plan.resolution.clone());
            }
        }
        for plan in plans.iter().filter(|plan| plan.resident_base) {
            if !resolutions.contains(&plan.resolution)
                && !resident_base_only.contains(&plan.resolution)
            {
                resident_base_only.push(plan.resolution.clone());
            }
        }

        let mut sessions = Vec::new();
        for resolution in &resolutions {
            resolution.validate().map_err(|error| {
                WorkerError::InvalidPayload(format!("invalid model source resolution: {error}"))
            })?;
            match resolution.availability {
                // Already leased and served from the app-owned tier above; the configured source
                // library is deliberately not consulted, opened, or probed for it.
                ModelAvailability::LocalReady => {}
                ModelAvailability::InstalledExternalUnavailable => {
                    return Err(unavailable(
                        "The configured external model library holding this installed model is \
                         disconnected or no longer has the expected physical identity. Reconnect \
                         it and retry; the installation itself is preserved.",
                    ));
                }
                // A model that was never installed (or whose install is incomplete while the
                // library is provably present) keeps its established on-demand install path.
                ModelAvailability::Incomplete | ModelAvailability::Missing => {}
                ModelAvailability::ExternalReady => {
                    match ExternalSourceSession::begin(&settings.data_dir, resolution) {
                        Ok(session) => sessions.push(session),
                        Err(error) => {
                            // The session probe re-proves path + physical identity and re-walks
                            // the closure. Distinguish "components vanished while the library
                            // stayed provably present" from a disconnect mid-admission.
                            // The requirements came from an external-ready resolution, so the
                            // install evidence is already proven durable.
                            let recheck = resolve_model_availability(
                                &settings.data_dir,
                                &configured_library,
                                &resolution.requirements,
                                true,
                                &[],
                            );
                            if recheck.availability
                                == ModelAvailability::InstalledExternalUnavailable
                            {
                                return Err(unavailable(
                                    "The external model library disconnected while the model \
                                     source was being admitted. Reconnect it and retry.",
                                ));
                            }
                            return Err(WorkerError::InvalidPayload(format!(
                                "model source components could not be validated: {error}"
                            )));
                        }
                    }
                }
            }
        }
        // sc-22329 — a resident base is ADVISORY: it exists so the family's own resolver can be
        // answered from a leased bundle, and its whole effect was already had by
        // `select_local_tier` above. Nothing here may refuse, and no external session is opened
        // for it. That is what keeps AC2 true: with the base uninstalled, torn, or its library
        // disconnected, the guard admits exactly as it did before this table existed and the
        // family route raises its own tagged `[checkpoint-plan:missing-component]` refusal — the
        // string callers and tests key on. An invalid resolution is dropped rather than reported,
        // for the same reason.
        resident_base_only.retain(|resolution| resolution.validate().is_ok());
        emit_source_tiers(&resolutions);
        emit_source_tiers(&resident_base_only);
        resolutions.extend(resident_base_only);
        let mut dispatch_payload = payload.clone();
        dispatch_payload.remove("__portableBoundTier");
        if let Some(advanced) = dispatch_payload
            .get_mut("advanced")
            .and_then(serde_json::Value::as_object_mut)
        {
            advanced.remove("__portableBoundTier");
            advanced.remove("__portableAuxPaths");
            advanced.remove("__portableFrozenSources");
            advanced.remove("__portableGemmaFiles");
        }
        if let Some(plan) = dispatch_payload
            .get_mut("plan")
            .and_then(serde_json::Value::as_object_mut)
        {
            plan.remove("__portableAuxPaths");
        }
        let mut portable_snapshot = None;
        let mut portable_scope = None;
        let mut plan_layers = None;
        let mut verified_revision_sources = Vec::new();
        if let Some(binding) = binding {
            verify_portable_route(
                binding,
                job_type,
                payload,
                settings,
                &plans,
                &converted_co_requirements,
                cancellation,
            )?;
            let prepared = prepare_portable_snapshot(
                binding,
                job_type,
                payload,
                settings,
                &mut dispatch_payload,
                cancellation,
            )?;
            if let Some(prepared) = prepared {
                local_scope = None;
                if !prepared.overlays.is_empty() {
                    let scope = match cancellation {
                        Some(token) => wait_for_portable_snapshots(prepared.overlays, token),
                        None => prefer_portable_snapshots(prepared.overlays),
                    };
                    portable_scope = Some(scope.map_err(|_| {
                        if cancellation.is_some_and(PortableOverlayWaitCancellation::is_cancelled) {
                            WorkerError::Canceled("portable overlay wait canceled".to_owned())
                        } else {
                            invalid_binding("frozen snapshot overlay is unavailable")
                        }
                    })?);
                }
                portable_snapshot = Some(prepared.snapshot);
                plan_layers = prepared.plan_layers;
            }
            rebind_revision_primary_paths(binding, settings, &mut dispatch_payload)?;
            rewrite_training_plan_source(binding, settings, &mut dispatch_payload)?;
            verified_revision_sources = revision_primary_handoffs(binding, &dispatch_payload)?;
            dispatch_payload
                .entry("advanced".to_owned())
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .ok_or_else(|| invalid_binding("advanced options are not an object"))?
                .insert("__portableFrozenSources".to_owned(), json!(true));
            if *job_type == JobType::ModelConvert
                && payload.get("converter").and_then(serde_json::Value::as_str) == Some("ltx_video")
            {
                let files = binding
                    .models
                    .iter()
                    .flat_map(|model| &model.members)
                    .filter(|member| {
                        matches!(&member.source,
                        PortableSourceIdentity::Revision { identity }
                            if identity.repository == "SceneWorks/ltx-2.3-mlx")
                    })
                    .flat_map(|member| &member.files)
                    .filter(|file| file.relative_path.starts_with("gemma"))
                    .cloned()
                    .collect::<Vec<_>>();
                if !files.is_empty() {
                    dispatch_payload
                        .get_mut("advanced")
                        .and_then(serde_json::Value::as_object_mut)
                        .expect("private advanced object was inserted")
                        .insert("__portableGemmaFiles".to_owned(), json!(files));
                }
            }
            if let Some(tier) = binding
                .models
                .iter()
                .find(|model| model.slot == "modelManifestEntry")
                .and_then(|model| model.tier.as_ref())
            {
                dispatch_payload
                    .entry("advanced".to_owned())
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .ok_or_else(|| invalid_binding("advanced options are not an object"))?
                    .insert(
                        "__portableBoundTier".to_owned(),
                        serde_json::Value::String(tier.clone()),
                    );
            }
        }
        Ok(Self {
            dispatch_payload,
            portable_scope,
            _portable_snapshot: portable_snapshot,
            plan_layers,
            verified_revision_sources,
            resolutions,
            sessions,
            local_scope,
            cache_leases,
            pending_promotions,
        })
    }

    fn none(payload: &JsonObject) -> Self {
        let mut dispatch_payload = payload.clone();
        dispatch_payload.remove("__portableBoundTier");
        if let Some(advanced) = dispatch_payload
            .get_mut("advanced")
            .and_then(serde_json::Value::as_object_mut)
        {
            advanced.remove("__portableBoundTier");
            advanced.remove("__portableAuxPaths");
            advanced.remove("__portableFrozenSources");
            advanced.remove("__portableGemmaFiles");
        }
        if let Some(plan) = dispatch_payload
            .get_mut("plan")
            .and_then(serde_json::Value::as_object_mut)
        {
            plan.remove("__portableAuxPaths");
        }
        Self {
            dispatch_payload,
            portable_scope: None,
            _portable_snapshot: None,
            plan_layers: None,
            verified_revision_sources: Vec::new(),
            resolutions: Vec::new(),
            sessions: Vec::new(),
            local_scope: None,
            cache_leases: Vec::new(),
            pending_promotions: Vec::new(),
        }
    }

    pub(crate) fn dispatch_payload(&self) -> &JsonObject {
        &self.dispatch_payload
    }

    pub(crate) fn plan_layers(&self) -> Option<std::sync::Arc<VerifiedPlanLayers>> {
        self.plan_layers.clone()
    }

    pub(crate) fn verified_revision_sources(&self) -> Vec<VerifiedRevisionSource> {
        self.verified_revision_sources.clone()
    }

    pub(crate) fn finish_success(mut self) -> WorkerResult<()> {
        for session in self.sessions.drain(..) {
            session
                .mark_success()
                .map_err(|error| WorkerError::Io(std::io::Error::other(error.to_string())))?;
        }
        // Stop serving local paths before the leases protecting them are released.
        self.portable_scope = None;
        self.local_scope = None;
        for lease in self.cache_leases.drain(..) {
            // The promotion boundary for an artifact that just served a real load. Its own
            // entry is already published, so this only records the successful use.
            let _candidate = lease.mark_success();
        }
        // sc-19706 — the only place a real install becomes a promotion candidate. A job that
        // reached here loaded and ran to completion against the configured source library, which
        // is the exact evidence promotion needs: the bytes exist, they are valid, and something
        // actually used them. Pushing is a lock-and-move into a bounded in-memory queue and is
        // deliberately the last thing the job does — the drain that turns these into bundles runs
        // on the blocking pool when the worker next claims nothing.
        crate::resolved_cache_promotion::record_pending(std::mem::take(
            &mut self.pending_promotions,
        ));
        Ok(())
    }

    /// Mid-load failure classification: re-probe every externally sourced model and suppress the
    /// raw loader error ONLY when the exact bound source is now provably unavailable. A load
    /// failure with the source still present is a real defect and is preserved verbatim.
    pub(crate) fn classify_failure(
        &self,
        settings: &Settings,
        original: WorkerError,
    ) -> WorkerError {
        let configured_library = sceneworks_core::hf_home::model_source_library(&settings.data_dir)
            .root()
            .to_path_buf();
        let source_became_unavailable = self.resolutions.iter().any(|resolution| {
            resolution.availability == ModelAvailability::ExternalReady
                && resolve_model_availability(
                    &settings.data_dir,
                    &configured_library,
                    &resolution.requirements,
                    true,
                    &[],
                )
                .availability
                    == ModelAvailability::InstalledExternalUnavailable
        });
        if source_became_unavailable {
            unavailable(
                "The external model library disconnected or changed during model use; reconnect \
                 it and retry. The original load failure was suppressed because the model's \
                 source is provably unavailable.",
            )
        } else {
            original
        }
    }

    #[cfg(test)]
    pub(crate) fn resolutions(&self) -> &[ModelResolution] {
        &self.resolutions
    }
}

/// Private content-addressed bundle used only for digest-backed local sources, worker claims and
/// checkpoint-plan layers; pinned Hugging Face members are verified in place and never copied. The
/// shared route lease remains held through dispatch, including loader reads, and prevents
/// cooperative eviction (which needs the lease exclusively); other attempts on the same route read
/// concurrently. `None` when the route needed no bundle.
#[derive(Debug)]
struct PortableAttemptSnapshot {
    _lease: Option<std::sync::Arc<FileLock>>,
}

/// Written into every route-bundle directory this layout creates. A bundle without it was built by
/// the earlier layout that also copied pinned Hugging Face members; nothing reads those any more,
/// so retention reclaims them first, whatever the size and count targets.
const PORTABLE_BUNDLE_LAYOUT_MARKER: &str = ".layout-v4";

struct PreparedPortableSnapshot {
    snapshot: PortableAttemptSnapshot,
    overlays: Vec<LocalArtifactOverlayEntry>,
    plan_layers: Option<std::sync::Arc<VerifiedPlanLayers>>,
}

/// In-process handoff from the byte guard to checkpoint-plan loaders. Paths point into the
/// guard's immutable, leased attempt bundle; they never enter a job payload or public carrier.
/// The guard builds this only after copy-then-verify, and keeps the lease until dispatch ends.
#[derive(Debug, Clone)]
pub(crate) struct VerifiedPlanLayers {
    by_checkpoint: std::collections::BTreeMap<String, std::collections::BTreeMap<String, PathBuf>>,
    plans: std::collections::BTreeMap<String, PortableCheckpointPlan>,
    _lease: std::sync::Arc<FileLock>,
}

impl VerifiedPlanLayers {
    fn new(lease: std::sync::Arc<FileLock>) -> Self {
        Self {
            by_checkpoint: Default::default(),
            plans: Default::default(),
            _lease: lease,
        }
    }

    #[cfg(all(test, native_lane))]
    pub(crate) fn new_for_test(lease: std::sync::Arc<FileLock>) -> Self {
        Self::new(lease)
    }

    #[cfg(native_lane)]
    pub(crate) fn lease(&self) -> std::sync::Arc<FileLock> {
        self._lease.clone()
    }

    pub(crate) fn add_plan(
        &mut self,
        checkpoint_id: &str,
        plan: PortableCheckpointPlan,
    ) -> WorkerResult<()> {
        plan.validate()
            .map_err(|error| invalid_binding(error.to_string()))?;
        if self.plans.insert(checkpoint_id.to_owned(), plan).is_some() {
            return Err(invalid_binding("duplicate private checkpoint plan"));
        }
        Ok(())
    }

    #[cfg(native_lane)]
    pub(crate) fn resolve_checkpoint(
        &self,
        checkpoint_id: &str,
        check: &dyn Fn() -> io::Result<()>,
    ) -> Result<
        sceneworks_core::checkpoint_plan_store::ResolvedCheckpointV1,
        sceneworks_core::checkpoint_plan_store::CheckpointPlanError,
    > {
        let _lease = self.lease();
        use sceneworks_core::checkpoint_import::{
            CheckpointCatalogRecordV1, ImportLayerV1, ImportPlanV1, SourceLocatorV1,
        };
        use sceneworks_core::checkpoint_plan_store::{
            CheckpointPlanError, ResolvedCheckpointV1, ResolvedLayerV1,
        };
        let refuse = |reason: &str| CheckpointPlanError::PlanTampered {
            checkpoint_id: checkpoint_id.to_owned(),
            reason: reason.to_owned(),
        };
        let checked = || {
            check().map_err(|error| CheckpointPlanError::Io {
                path: PathBuf::new(),
                message: error.to_string(),
            })
        };
        checked()?;
        let portable = self
            .plans
            .get(checkpoint_id)
            .ok_or_else(|| refuse("the guard did not lease this checkpoint plan"))?;
        portable
            .validate()
            .map_err(|_| refuse("portable checkpoint plan is invalid"))?;
        let paths = self
            .by_checkpoint
            .get(checkpoint_id)
            .ok_or_else(|| refuse("checkpoint plan layers are absent"))?;
        if paths.len() != portable.layers.len() {
            return Err(refuse(
                "private plan layer set differs from the frozen plan",
            ));
        }
        let layers = portable
            .layers
            .iter()
            .map(|layer| {
                let digest = layer
                    .sha256
                    .strip_prefix("sha256:")
                    .ok_or_else(|| refuse("checkpoint layer digest is invalid"))?;
                Ok(ImportLayerV1 {
                    layer_id: layer.layer_id.clone(),
                    role: layer.role.clone(),
                    target_path: layer.target_path.clone(),
                    container: layer.container,
                    source: SourceLocatorV1::linked("portable", "artifact", digest)
                        .map_err(|_| refuse("checkpoint layer locator is invalid"))?,
                })
            })
            .collect::<Result<Vec<_>, CheckpointPlanError>>()?;
        let mut plan = ImportPlanV1::new("portable", &portable.family, layers)
            .map_err(|_| refuse("checkpoint plan structure is invalid"))?;
        plan.context = portable.context.clone();
        if plan
            .semantic_digest()
            .map_err(|_| refuse("checkpoint plan digest is invalid"))?
            != portable.semantic_digest
        {
            return Err(refuse(
                "checkpoint plan semantics differ from the frozen plan",
            ));
        }
        let record = CheckpointCatalogRecordV1::from_plan(checkpoint_id, &plan)
            .map_err(|_| refuse("checkpoint catalog projection is invalid"))?;
        let mut resolved_layers = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            checked()?;
            let path = paths
                .get(&layer.layer_id)
                .ok_or_else(|| refuse("private checkpoint layer is missing"))?;
            resolved_layers.push(ResolvedLayerV1 {
                layer: layer.clone(),
                path: path.clone(),
                rehashed: false,
            });
        }
        Ok(ResolvedCheckpointV1 {
            checkpoint_id: checkpoint_id.to_owned(),
            record,
            plan,
            layers: resolved_layers,
        })
    }

    pub(crate) fn insert(
        &mut self,
        checkpoint_id: &str,
        layer_id: &str,
        path: PathBuf,
    ) -> WorkerResult<()> {
        let layers = self
            .by_checkpoint
            .entry(checkpoint_id.to_owned())
            .or_default();
        if layers.insert(layer_id.to_owned(), path).is_some() {
            return Err(invalid_binding("duplicate private checkpoint plan layer"));
        }
        Ok(())
    }
}

const PORTABLE_CACHE_MAX_ENTRIES: usize = 8;
const PORTABLE_CACHE_MAX_BYTES: u64 = 256 * 1024 * 1024 * 1024;

// On non-Unix the readonly bit must be cleared before removal (Windows semantics);
// the Unix branches use explicit modes instead.
#[cfg_attr(not(unix), allow(clippy::permissions_set_readonly_false))]
fn remove_private_snapshot(root: &Path) -> io::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    let metadata = std::fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "private snapshot contains a link",
        ));
    }
    if metadata.is_file() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o600))?;
        }
        #[cfg(not(unix))]
        {
            let mut permissions = metadata.permissions();
            permissions.set_readonly(false);
            std::fs::set_permissions(root, permissions)?;
        }
        return std::fs::remove_file(root);
    }
    if !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "private snapshot contains a non-file",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))?;
    }
    for entry in std::fs::read_dir(root)? {
        remove_private_snapshot(&entry?.path())?;
    }
    std::fs::remove_dir(root)
}

/// A resolved-cache preference can contain a complete hard closure while a selected soft
/// companion lives only in the configured HF source tier. Choose per frozen member, never per
/// repository, so a partial preferred snapshot cannot hide that companion.
fn discover_frozen_member_snapshot(
    settings: &Settings,
    identity: &ArtifactIdentity,
    files: &[PortableArtifactFile],
) -> WorkerResult<PathBuf> {
    let preferred = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
    if let Ok((_, selected)) =
        preferred.discover_snapshot(&identity.repository, Some(&identity.revision))
    {
        let selected_complete = files
            .iter()
            .all(|file| selected.join(&file.relative_path).is_file());
        if selected_complete
            && (selected.starts_with(preferred.root())
                || snapshot_matches_frozen_files(&selected, files))
        {
            return Ok(selected);
        }
    }
    let source = ArtifactSourceLibrary::new(preferred.root())
        .map_err(|error| invalid_binding(error.to_string()))?;
    let source_snapshot = source
        .discover_snapshot(&identity.repository, Some(&identity.revision))
        .map_err(|_| invalid_binding("frozen source snapshot is unavailable"))?
        .1;
    if files
        .iter()
        .any(|file| !source_snapshot.join(&file.relative_path).is_file())
    {
        return Err(invalid_binding(
            "a frozen member is missing from every authorized source tier",
        ));
    }
    Ok(source_snapshot)
}

/// Wildcard checks prefer the authoritative source snapshot. An active local preference can
/// contain only another job's frozen files and hide additional files selected by this job's
/// manifest glob. If the original source has gone away, a verified local snapshot remains usable.
fn discover_frozen_selector_snapshot<'a>(
    settings: &Settings,
    members: impl Iterator<Item = &'a PortableArtifactMember>,
    repository: &str,
    revision: &str,
    variant: &str,
    primary: bool,
) -> WorkerResult<PathBuf> {
    let selected = members
        .filter(|member| {
            matches!(&member.source, PortableSourceIdentity::Revision { identity }
                if identity.repository == repository && identity.revision == revision
                    && identity.variant == variant)
                && (member.role == ArtifactMemberRole::Primary) == primary
        })
        .collect::<Vec<_>>();
    let identity = selected
        .first()
        .and_then(|member| match &member.source {
            PortableSourceIdentity::Revision { identity } => Some(identity),
            _ => None,
        })
        .ok_or_else(|| invalid_binding("wildcard source has no frozen member"))?;
    let preferred = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
    let source = ArtifactSourceLibrary::new(preferred.root())
        .map_err(|error| invalid_binding(error.to_string()))?;
    if let Ok((_, snapshot)) =
        source.discover_snapshot(&identity.repository, Some(&identity.revision))
    {
        return Ok(snapshot);
    }
    let files = selected
        .iter()
        .flat_map(|member| member.files.iter().cloned())
        .collect::<Vec<_>>();
    discover_frozen_member_snapshot(settings, identity, &files)
}

fn snapshot_matches_frozen_files(snapshot: &Path, files: &[PortableArtifactFile]) -> bool {
    files.iter().all(|file| {
        let path = snapshot.join(&file.relative_path);
        let Ok(input) = File::open(path) else {
            return false;
        };
        let Ok((digest, size)) = sceneworks_core::hashing::sha256_reader_retry(input) else {
            return false;
        };
        size == file.size_bytes && format!("sha256:{digest}") == file.sha256
    })
}

/// One file of a pinned Hugging Face (Revision) member, resolved in its own hub snapshot.
struct InPlaceRevisionFile {
    real: PathBuf,
    expected: PortableArtifactFile,
}

/// The generic progress label for in-place source verification; never a provider name.
const IN_PLACE_VERIFICATION_COMPONENT: &str = "model files";

#[cfg(test)]
thread_local! {
    static HASHED_SOURCE_FILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Resolve every frozen file of a Revision member inside `snapshot` without copying it. A
/// relative path that is not a plain descendant, a file whose real path leaves the snapshot's own
/// hub repository directory (`models--<repo>`: its `snapshots/` links and `blobs/`), or a size
/// that differs from the ledger is refused here, before any byte is read.
fn in_place_revision_files(
    snapshot: &Path,
    files: &[PortableArtifactFile],
) -> WorkerResult<Vec<InPlaceRevisionFile>> {
    let repository_dir = snapshot
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| invalid_binding("frozen source snapshot has no hub repository"))?;
    let repository_real = std::fs::canonicalize(repository_dir)
        .map_err(|_| invalid_binding("frozen source snapshot is unavailable"))?;
    files
        .iter()
        .map(|file| {
            if file.relative_path.as_os_str().is_empty()
                || !file
                    .relative_path
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
            {
                return Err(invalid_binding("a frozen member path escapes its snapshot"));
            }
            let real = std::fs::canonicalize(snapshot.join(&file.relative_path))
                .map_err(|_| invalid_binding("a frozen member is missing from its snapshot"))?;
            if !real.starts_with(&repository_real) {
                return Err(invalid_binding(
                    "a frozen member resolves outside its hub repository",
                ));
            }
            let metadata = std::fs::metadata(&real).map_err(WorkerError::Io)?;
            if !metadata.is_file() || metadata.len() != file.size_bytes {
                return Err(invalid_binding(
                    "a frozen member differs from its pinned ledger",
                ));
            }
            Ok(InPlaceRevisionFile {
                real,
                expected: file.clone(),
            })
        })
        .collect()
}

/// This host's hash-once verdict store ([`sceneworks_core::verified_source_cache`]).
fn verified_source_cache(settings: &Settings) -> VerifiedSourceCache {
    VerifiedSourceCache::new(
        settings
            .data_dir
            .join("models/portable-attempts/.verified-sources"),
    )
}

/// Run `body` with a load-progress context that reports `Verifying model files — read / total`
/// over `files` (real path, size), so a long first verification is never silent.
fn with_verification_progress<T>(
    files: &[(PathBuf, u64)],
    body: impl FnOnce(&gen_core::LoadActivityContext) -> T,
) -> T {
    let progress = gen_core::LoadActivityContext::capture().with_components(
        files
            .iter()
            .map(|(path, _)| (path.clone(), IN_PLACE_VERIFICATION_COMPONENT.to_owned())),
    );
    progress.register_files(files.iter().cloned());
    let _installed = progress.install();
    body(&progress)
}

fn verification_canceled(cancellation: Option<&PortableOverlayWaitCancellation>) -> bool {
    cancellation.is_some_and(PortableOverlayWaitCancellation::is_cancelled)
}

/// Verify one canonical file against its frozen ledger entry, hashing it at most once per host
/// while its filesystem identity is unchanged. A mismatch is `InvalidData`; cancellation is
/// `Interrupted`.
fn verify_source_once(
    cache: &VerifiedSourceCache,
    progress: &gen_core::LoadActivityContext,
    real: &Path,
    expected: &PortableArtifactFile,
    cancellation: Option<&PortableOverlayWaitCancellation>,
) -> io::Result<()> {
    let check = || {
        if verification_canceled(cancellation) {
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "frozen source verification canceled",
            ))
        } else {
            Ok(())
        }
    };
    match cache.verify(real, &expected.sha256, &check)? {
        SourceVerification::Reused => {
            let done = progress.file(
                gen_core::LoadActivityPhase::Verifying,
                real,
                Some(expected.size_bytes),
            );
            done.update(expected.size_bytes);
            done.finish();
        }
        SourceVerification::Hashed => {
            #[cfg(test)]
            HASHED_SOURCE_FILES.with(|count| count.set(count.get() + 1));
        }
    }
    Ok(())
}

/// Verify in-place Revision files against their frozen SHA-256 ledger (normally a recorded
/// verdict from the route check moments earlier, so no byte is read twice).
fn verify_in_place_revision_files(
    settings: &Settings,
    files: &[InPlaceRevisionFile],
    cancellation: Option<&PortableOverlayWaitCancellation>,
) -> WorkerResult<()> {
    if files.is_empty() {
        return Ok(());
    }
    let cache = verified_source_cache(settings);
    let sized = files
        .iter()
        .map(|file| (file.real.clone(), file.expected.size_bytes))
        .collect::<Vec<_>>();
    with_verification_progress(&sized, |progress| {
        for file in files {
            verify_source_once(&cache, progress, &file.real, &file.expected, cancellation)
                .map_err(|error| {
                    if verification_canceled(cancellation) {
                        WorkerError::Canceled("frozen source verification canceled".to_owned())
                    } else if error.kind() == io::ErrorKind::InvalidData {
                        invalid_binding("a frozen member differs from its pinned ledger")
                    } else {
                        WorkerError::Io(error)
                    }
                })?;
        }
        Ok(())
    })
}

fn selected_portable_aux_ids(
    job_type: &JobType,
    payload: &JsonObject,
    entry: &serde_json::Value,
    backend: &str,
) -> WorkerResult<Vec<String>> {
    let mut ids = required_portable_aux_source_ids(job_type, payload, entry, backend)
        .map_err(invalid_binding)?;
    if matches!(job_type, JobType::LoraTrain | JobType::ControlTraining) {
        let plan: sceneworks_core::training::TrainingPlan = serde_json::from_value(
            payload
                .get("plan")
                .cloned()
                .ok_or_else(|| invalid_binding("training plan is absent"))?,
        )
        .map_err(|error| invalid_binding(error.to_string()))?;
        if entry.get("id").and_then(serde_json::Value::as_str)
            == Some(plan.target.base_model.as_str())
        {
            let engine =
                crate::portable_trainer_engine_id_for(&plan.target.kernel, &plan.target.base_model)
                    .ok_or_else(|| invalid_binding("training engine is unavailable"))?;
            ids.extend(
                sceneworks_core::portable_sources::selected_training_aux_models(&plan, engine)
                    .map_err(invalid_binding)?
                    .into_iter()
                    .map(|model| model.id.to_owned()),
            );
        }
    }
    ids.sort();
    ids.dedup();
    Ok(ids)
}

fn selected_runtime_variant(
    job_type: &JobType,
    payload: &JsonObject,
    model_id: &str,
) -> WorkerResult<Option<String>> {
    if !matches!(job_type, JobType::LoraTrain | JobType::ControlTraining) {
        return Ok(requested_runtime_variant(payload));
    }
    let target = payload
        .get("plan")
        .and_then(|plan| plan.get("target"))
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| invalid_binding("training plan has no canonical target"))?;
    let base = target
        .get("baseModel")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_binding("training plan has no base model"))?;
    if base != model_id {
        return Ok(requested_runtime_variant(payload));
    }
    let kernel = target
        .get("kernel")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_binding("training plan has no native kernel"))?;
    Ok(Some(
        if base == "ltx_2_5" || kernel == "ltx_mlx_lora" {
            "q4"
        } else {
            "bf16"
        }
        .to_owned(),
    ))
}

/// The tier this request runs. An image request that names no tier runs the worker's implicit
/// standard tier, which the API froze into the admitted binding (`slot_requested_variant` in
/// rust-api `portable_admission.rs`); the request itself carries no tier, so the binding's own tier
/// is the selection. It applies only where that implicit rule applies (an image job's
/// `modelManifestEntry` slot of a standard quant-tier model) and only to a bf16/q8/q4 tier; the
/// selected closure is then verified against the binding like any explicit tier.
fn bound_runtime_variant(
    job_type: &JobType,
    payload: &JsonObject,
    slot: &str,
    entry: &serde_json::Value,
    binding: Option<&PortableModelBinding>,
) -> WorkerResult<Option<String>> {
    let model_id = entry
        .get("id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if let Some(variant) = selected_runtime_variant(job_type, payload, model_id)? {
        return Ok(Some(variant));
    }
    if slot != "modelManifestEntry"
        || !matches!(
            job_type,
            JobType::ImageGenerate | JobType::ImageEdit | JobType::ImageDetail
        )
    {
        return Ok(None);
    }
    let standard_tier_layout = sceneworks_core::registry::builtin_model_spec(model_id)
        .and_then(|spec| spec.product.fields.get("mlx"))
        .or_else(|| entry.get("mlx"))
        .and_then(|mlx| mlx.get("standardTierLayout"))
        .and_then(serde_json::Value::as_bool)
        == Some(true);
    Ok(binding
        .filter(|binding| standard_tier_layout && binding.model_id == model_id)
        .and_then(|binding| binding.tier.clone())
        .filter(|tier| matches!(tier.as_str(), "bf16" | "q8" | "q4")))
}

fn prepare_portable_snapshot(
    route: &PortableRouteBinding,
    job_type: &JobType,
    payload: &JsonObject,
    settings: &Settings,
    dispatch_payload: &mut JsonObject,
    cancellation: Option<&PortableOverlayWaitCancellation>,
) -> WorkerResult<Option<PreparedPortableSnapshot>> {
    let model_slots = payload_model_slots(payload);
    let selected_aux = route
        .models
        .iter()
        .map(|model| {
            let entry = model_slots
                .iter()
                .find(|(slot, _)| slot == &model.slot)
                .map(|(_, entry)| *entry)
                .ok_or_else(|| invalid_binding("snapshot model slot is absent"))?;
            selected_portable_aux_ids(job_type, payload, entry, &route.backend)
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    if route.models.is_empty() {
        return Ok(None);
    }
    // Pinned Hugging Face (Revision) members are verified and served IN PLACE from this worker's
    // own hub snapshot: they are content-addressed and never copied, and the loader reads the
    // hub's own layout (snapshot links into the repository's `blobs/`), which is what the
    // inference shard-path check accepts. Only the rest (digest-backed local sources, worker
    // claims and checkpoint-plan layers) is copied into a private, verified, leased route bundle.
    //
    // One revision can resolve in two roots (a resolved-cache snapshot holds the hard closure, the
    // source library a soft companion). A selected auxiliary has its own per-component loader
    // path and is served from its own root. When the members a loader reaches through the one
    // snapshot directory of that revision span two roots, those members fall back to the
    // copy-then-verify bundle, the only arrangement in which a pinned member is copied.
    let mut plan_layers_by_member = Vec::new();
    let mut revision_snapshots_by_member = Vec::new();
    let mut directory_roots: std::collections::BTreeMap<(String, String), Vec<PathBuf>> =
        std::collections::BTreeMap::new();
    let mut needs_bundle = false;
    for (model_index, model) in route.models.iter().enumerate() {
        let entry = model_slots
            .iter()
            .find(|(slot, _)| slot == &model.slot)
            .map(|(_, entry)| *entry)
            .ok_or_else(|| invalid_binding("snapshot model slot is absent"))?;
        let mut layers = Vec::new();
        let mut snapshots = Vec::new();
        for member in &model.members {
            let plan_layer = selected_checkpoint_plan_layer(member, model, entry, settings)?;
            needs_bundle |= plan_layer.is_some()
                || !matches!(member.source, PortableSourceIdentity::Revision { .. });
            let snapshot = if let PortableSourceIdentity::Revision { identity } = &member.source {
                let snapshot = discover_frozen_member_snapshot(settings, identity, &member.files)?;
                let is_selected_aux = member
                    .component_id
                    .as_ref()
                    .is_some_and(|id| selected_aux[model_index].contains(id));
                if plan_layer.is_none() && !is_selected_aux {
                    let roots = directory_roots
                        .entry((identity.repository.clone(), identity.revision.clone()))
                        .or_default();
                    if !roots.contains(&snapshot) {
                        roots.push(snapshot.clone());
                    }
                }
                Some(snapshot)
            } else {
                None
            };
            layers.push(plan_layer);
            snapshots.push(snapshot);
        }
        needs_bundle |= model.checkpoint_plan.is_some();
        debug_assert_eq!(plan_layers_by_member.len(), model_index);
        plan_layers_by_member.push(layers);
        revision_snapshots_by_member.push(snapshots);
    }
    let copied_revisions = directory_roots
        .iter()
        .filter(|(_, roots)| roots.len() > 1)
        .map(|(key, _)| key.clone())
        .collect::<std::collections::BTreeSet<_>>();
    needs_bundle |= !copied_revisions.is_empty();
    let cache_root = settings.data_dir.join("models/portable-attempts");
    create_private_directory(&settings.data_dir.join("models"))?;
    create_private_directory(&cache_root)?;
    ensure_private_directory(&cache_root)?;
    // A supported route may itself exceed the retention target. It remains leased while active;
    // prune removes other entries, and a later route can evict this one after the lease ends.
    let route_dir = cache_root.join(portable_cache_route_digest(route)?);
    create_private_directory(&route_dir)?;
    ensure_private_directory(&route_dir)?;
    // Readers share the route lease for as long as they use the bundle; an evictor needs it
    // exclusively, so it never removes a bundle an attempt is reading. A published bundle is
    // immutable, so concurrent attempts on one route (GPUs sharing `data_dir`) read it at once.
    let route_lease = std::sync::Arc::new(
        FileLock::shared(
            sceneworks_core::file_lock::open_regular_lock_file(&route_dir.join(".lease"))
                .map_err(WorkerError::Io)?,
        )
        .map_err(WorkerError::Io)?,
    );
    std::fs::write(route_dir.join(PORTABLE_BUNDLE_LAYOUT_MARKER), b"").map_err(WorkerError::Io)?;
    let bundle = needs_bundle.then(|| (route_dir.clone(), route_lease.clone()));
    let published = bundle
        .as_ref()
        .map(|(parent, _)| parent.join("bundle"))
        .unwrap_or_default();
    // Materialization is exclusive: one writer builds the staging tree and publishes it by
    // rename, so a reader only ever sees a complete, verified bundle or none. The writer lock is
    // held only until publication, never through dispatch.
    let materializing = match &bundle {
        Some((parent, _)) if !published.exists() => {
            let writer =
                FileLock::open_exclusive(&parent.join(".materialize")).map_err(WorkerError::Io)?;
            (!published.exists()).then_some(writer)
        }
        _ => None,
    };
    let needs_copy = materializing.is_some();
    let staging = bundle
        .as_ref()
        .map(|(parent, _)| parent.join(format!("staging-{}", uuid::Uuid::new_v4())))
        .unwrap_or_default();
    if let (true, Some((parent, _))) = (needs_copy, &bundle) {
        // A crashed first materialization leaves only a uniquely named staging tree; it never
        // becomes a load candidate. The writer lock makes cleanup exclusive with another writer.
        for entry in std::fs::read_dir(parent).map_err(WorkerError::Io)? {
            let entry = entry.map_err(WorkerError::Io)?;
            if entry.file_name().to_string_lossy().starts_with("staging-") {
                remove_private_snapshot(&entry.path()).map_err(WorkerError::Io)?;
            }
        }
        create_private_directory(&staging)?;
    }
    let mut allowed_roots = crate::paths::allowed_model_roots(settings)?;
    let mut rewrites = Vec::new();
    // (slot, member, selected path, selected auxiliary) for in-place Revision members.
    let mut in_place = Vec::new();
    let mut in_place_files = Vec::new();
    // The one snapshot directory each in-place revision is served from, and the files in it.
    let mut in_place_roots: std::collections::BTreeMap<(String, String), (PathBuf, Vec<PathBuf>)> =
        std::collections::BTreeMap::new();
    for (model_index, model) in route.models.iter().enumerate() {
        let entry = model_slots
            .iter()
            .find(|(slot, _)| slot == &model.slot)
            .map(|(_, entry)| *entry)
            .ok_or_else(|| invalid_binding("snapshot model slot is absent"))?;
        let required_ids = &selected_aux[model_index];
        for (member_index, member) in model.members.iter().enumerate() {
            let plan_layer = plan_layers_by_member[model_index][member_index].take();
            if plan_layer.is_some() && member.files.len() != 1 {
                return Err(invalid_binding(
                    "checkpoint plan layer is not one frozen file",
                ));
            }
            let is_selected_aux = member
                .component_id
                .as_ref()
                .is_some_and(|id| required_ids.contains(id));
            let revision_snapshot = revision_snapshots_by_member[model_index][member_index].take();
            let selected = if let Some(snapshot) = &revision_snapshot {
                if !is_selected_aux {
                    snapshot.clone()
                } else {
                    let id = member
                        .component_id
                        .as_deref()
                        .ok_or_else(|| invalid_binding("selected auxiliary has no id"))?;
                    match portable_aux_source_shape(id, entry)
                        .ok_or_else(|| invalid_binding("selected auxiliary has no loader shape"))?
                    {
                        PortableAuxSourceShape::File => {
                            if member.files.len() != 1 {
                                return Err(invalid_binding(
                                    "file-shaped auxiliary has multiple files",
                                ));
                            }
                            snapshot.join(&member.files[0].relative_path)
                        }
                        PortableAuxSourceShape::Directory => {
                            selected_revision_aux_directory(snapshot, &member.files, id)?
                        }
                    }
                }
            } else if let Some(layer) = &plan_layer {
                layer.source_path.clone()
            } else {
                selected_local_artifact_path(member, model, entry, payload, settings, job_type)?
            };
            if matches!(member.source, PortableSourceIdentity::WorkerClaim { .. })
                || plan_layer.is_some()
            {
                allowed_roots.push(selected.parent().unwrap_or(&selected).to_path_buf());
            }
            let selected_is_file = selected.is_file();
            if !selected_is_file && !selected.is_dir() {
                return Err(invalid_binding("snapshot source is unavailable"));
            }
            if is_selected_aux {
                let id = member
                    .component_id
                    .as_deref()
                    .expect("selected auxiliary has an id");
                let shape = portable_aux_source_shape(id, entry)
                    .ok_or_else(|| invalid_binding("selected auxiliary has no loader shape"))?;
                if selected_is_file != (shape == PortableAuxSourceShape::File) {
                    return Err(invalid_binding(
                        "selected auxiliary loader shape differs from its binding",
                    ));
                }
            }
            if let (Some(snapshot), None, PortableSourceIdentity::Revision { identity }) =
                (&revision_snapshot, &plan_layer, &member.source)
            {
                let key = (identity.repository.clone(), identity.revision.clone());
                if is_selected_aux || !copied_revisions.contains(&key) {
                    in_place_files.extend(in_place_revision_files(snapshot, &member.files)?);
                    if !is_selected_aux {
                        let (_, files) = in_place_roots
                            .entry(key)
                            .or_insert_with(|| (snapshot.clone(), Vec::new()));
                        files.extend(member.files.iter().map(|file| file.relative_path.clone()));
                    }
                    in_place.push((
                        model.slot.clone(),
                        member.clone(),
                        selected,
                        is_selected_aux,
                    ));
                    continue;
                }
            }
            let target_root = if let PortableSourceIdentity::Revision { identity } = &member.source
            {
                staging.join("hf").join(
                    hub_cache_member_destination(
                        &identity.repository,
                        &identity.revision,
                        Path::new(""),
                    )
                    .map_err(|error| invalid_binding(error.to_string()))?,
                )
            } else if let Some(layer) = &plan_layer {
                staging
                    .join(format!("model{model_index}/checkpoint-plan"))
                    .join(layer.target_path.parent().unwrap_or(Path::new("")))
            } else {
                staging.join(format!("model{model_index}/member{member_index}"))
            };
            let plan_target_name = plan_layer
                .as_ref()
                .map(|layer| {
                    layer
                        .target_path
                        .file_name()
                        .map(PathBuf::from)
                        .ok_or_else(|| invalid_binding("checkpoint plan layer has no target file"))
                })
                .transpose()?;
            if needs_copy {
                create_private_tree(&target_root)?;
                for file in &member.files {
                    let source = if let Some(snapshot) = &revision_snapshot {
                        snapshot.join(&file.relative_path)
                    } else if selected_is_file {
                        selected.clone()
                    } else {
                        selected.join(&file.relative_path)
                    };
                    sceneworks_core::model_artifacts::confine_artifact_path(
                        &source,
                        &allowed_roots,
                    )
                    .map_err(|_| invalid_binding("frozen source is outside an authorized root"))?;
                    let target = target_root
                        .join(plan_target_name.as_deref().unwrap_or(&file.relative_path));
                    create_private_tree(target.parent().expect("portable file has a parent"))?;
                    if revision_snapshot.is_some() && target.exists() {
                        continue;
                    }
                    let mut input = File::open(&source).map_err(WorkerError::Io)?;
                    let mut output = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&target)
                        .map_err(WorkerError::Io)?;
                    io::copy(&mut input, &mut output).map_err(WorkerError::Io)?;
                    output.sync_all().map_err(WorkerError::Io)?;
                }
            }
            rewrites.push((
                model.slot.clone(),
                member.clone(),
                if let Some(snapshot) = &revision_snapshot {
                    selected
                        .strip_prefix(snapshot)
                        .expect("selected revision source is inside snapshot")
                        .to_path_buf()
                } else if let Some(target_name) = &plan_target_name {
                    target_name.clone()
                } else if selected_is_file {
                    member.files[0].relative_path.clone()
                } else {
                    PathBuf::new()
                },
                is_selected_aux,
                target_root
                    .strip_prefix(&staging)
                    .expect("target is beneath staging")
                    .to_path_buf(),
                plan_layer.map(|layer| (layer.checkpoint_id, layer.layer_id)),
            ));
        }
    }
    // In-place Revision bytes are verified against the frozen ledger before any loader sees a
    // path, hashing each file at most once per host while its filesystem identity is unchanged.
    verify_in_place_revision_files(settings, &in_place_files, cancellation)?;
    // Conversions publish regular copies of what they borrow from this route; record which hub
    // files it verified so publication can recognize them however a converter names them.
    record_route_frozen_sources(&route_dir, &in_place_files)?;
    if needs_copy {
        // A source can change after the first guard check while it is being copied. Never
        // publish those bytes as a reusable bundle until the complete staging tree matches the
        // frozen ledger. A failed attempt remains private and the next attempt can rebuild it.
        publish_verified_snapshot(&staging, &published, || {
            private_tree_bytes(&staging)?;
            for (_, member, selected_relative, _, relative_root, plan_layer) in &rewrites {
                verify_private_snapshot_member(
                    &staging,
                    member,
                    selected_relative,
                    relative_root,
                    plan_layer.is_some(),
                    &allowed_roots,
                )?;
            }
            Ok(())
        })?;
        // Freeze before other writers or readers can observe the bundle as published by this
        // writer; later readers only re-verify it.
        freeze_private_snapshot(&published)?;
    }
    drop(materializing);
    let mut verified_plan_layers = None;
    if let Some((_, lease)) = &bundle {
        private_tree_bytes(&published)?;
        // Copy-then-verify closes the race where the original changes after the first source
        // check. The loader receives only these verified, private paths, never the original locator.
        let mut layers = VerifiedPlanLayers::new(lease.clone());
        for model in &route.models {
            if let Some(plan) = &model.checkpoint_plan {
                let entry = model_slots
                    .iter()
                    .find(|(slot, _)| slot == &model.slot)
                    .map(|(_, entry)| *entry)
                    .ok_or_else(|| invalid_binding("snapshot model slot is absent"))?;
                let checkpoint_id = entry
                    .pointer("/importPlan/checkpointId")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| invalid_binding("checkpoint plan has no job checkpoint ID"))?;
                layers.add_plan(checkpoint_id, plan.clone())?;
            }
        }
        for (slot, member, selected_relative, is_selected_aux, relative_root, plan_layer) in
            &rewrites
        {
            if !needs_copy {
                verify_private_snapshot_member(
                    &published,
                    member,
                    selected_relative,
                    relative_root,
                    plan_layer.is_some(),
                    &allowed_roots,
                )?;
            }
            let target_root = published.join(relative_root);
            let selected = target_root.join(selected_relative);
            route_loader_locator(job_type, dispatch_payload, slot, member, &selected)?;
            if let Some((checkpoint_id, layer_id)) = plan_layer {
                layers.insert(checkpoint_id, layer_id, selected)?;
            } else if *is_selected_aux
                || !matches!(member.source, PortableSourceIdentity::Revision { .. })
            {
                rewrite_loader_locator(
                    dispatch_payload,
                    slot,
                    member,
                    &selected,
                    *is_selected_aux,
                )?;
            }
        }
        freeze_private_snapshot(&published)?;
        verified_plan_layers = Some(layers);
    }
    for (slot, member, selected, is_selected_aux) in &in_place {
        route_loader_locator(job_type, dispatch_payload, slot, member, selected)?;
        if *is_selected_aux {
            rewrite_loader_locator(dispatch_payload, slot, member, selected, true)?;
        }
    }
    let mut by_source: std::collections::BTreeMap<(String, String), LocalArtifactOverlayEntry> =
        std::collections::BTreeMap::new();
    for ((repository, revision), (snapshot_root, files)) in in_place_roots {
        by_source.insert(
            (repository.clone(), revision.clone()),
            LocalArtifactOverlayEntry {
                repository,
                revision,
                snapshot_root,
                files,
            },
        );
    }
    for (_, member, _, is_selected_aux, _, plan_layer) in &rewrites {
        let PortableSourceIdentity::Revision { identity } = &member.source else {
            continue;
        };
        if *is_selected_aux || plan_layer.is_some() {
            continue;
        }
        let entry = by_source
            .entry((identity.repository.clone(), identity.revision.clone()))
            .or_insert_with(|| LocalArtifactOverlayEntry {
                repository: identity.repository.clone(),
                revision: identity.revision.clone(),
                snapshot_root: published.join("hf").join(
                    hub_cache_member_destination(
                        &identity.repository,
                        &identity.revision,
                        Path::new(""),
                    )
                    .expect("a copied revision has a hub destination"),
                ),
                files: Vec::new(),
            });
        entry
            .files
            .extend(member.files.iter().map(|file| file.relative_path.clone()));
    }
    let mut overlays = by_source.into_values().collect::<Vec<_>>();
    for entry in &mut overlays {
        entry.files.sort();
        entry.files.dedup();
    }
    std::fs::write(route_dir.join(".last-used"), b"").map_err(WorkerError::Io)?;
    prune_portable_snapshots(Some(&route_dir), settings)?;
    let plan_layers = verified_plan_layers
        .filter(|layers| !layers.plans.is_empty())
        .map(std::sync::Arc::new);
    Ok(Some(PreparedPortableSnapshot {
        snapshot: PortableAttemptSnapshot {
            _lease: Some(route_lease),
        },
        overlays,
        plan_layers,
    }))
}

/// Every hub file this route verified in place, for conversion publication (see
/// [`frozen_route_source`]). Rewritten on every attempt; removed with the route by retention.
const ROUTE_FROZEN_SOURCES: &str = ".frozen-sources.json";

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RouteFrozenSource {
    real_path: PathBuf,
    sha256: String,
    size_bytes: u64,
}

fn record_route_frozen_sources(
    route_dir: &Path,
    files: &[InPlaceRevisionFile],
) -> WorkerResult<()> {
    let record = files
        .iter()
        .map(|file| RouteFrozenSource {
            real_path: file.real.clone(),
            sha256: file.expected.sha256.clone(),
            size_bytes: file.expected.size_bytes,
        })
        .collect::<Vec<_>>();
    let body = serde_json::to_vec(&record).map_err(|error| WorkerError::Io(error.into()))?;
    sceneworks_core::fsutil::write_atomic(&route_dir.join(ROUTE_FROZEN_SOURCES), &body)
        .map_err(WorkerError::Io)
}

/// Test support: record `reals` (read and hashed here) as a route's in-place frozen sources.
/// Its only caller links hub files with Unix symlinks, so it shares that test's `unix` gate.
#[cfg(all(test, unix))]
pub(crate) fn test_record_route_frozen_sources(route_dir: &Path, reals: &[&Path]) {
    std::fs::create_dir_all(route_dir).unwrap();
    let files = reals
        .iter()
        .map(|real| {
            let bytes = std::fs::read(real).unwrap();
            InPlaceRevisionFile {
                real: std::fs::canonicalize(real).unwrap(),
                expected: PortableArtifactFile {
                    relative_path: real.file_name().unwrap().into(),
                    size_bytes: bytes.len() as u64,
                    sha256: format!("sha256:{:x}", Sha256::digest(&bytes)),
                },
            }
        })
        .collect::<Vec<_>>();
    record_route_frozen_sources(route_dir, &files).unwrap();
}

/// The frozen ledger (SHA-256, size) of hub file `real` when a portable route under
/// `attempts_root` (`models/portable-attempts`) verified it in place. Converters canonicalize what
/// they borrow, so a converted layout links a route's hub file by its real path.
pub(crate) fn frozen_route_source(
    attempts_root: &Path,
    real: &Path,
) -> WorkerResult<Option<(String, u64)>> {
    let Ok(routes) = std::fs::read_dir(attempts_root) else {
        return Ok(None);
    };
    for route in routes.flatten() {
        let Ok(body) = std::fs::read(route.path().join(ROUTE_FROZEN_SOURCES)) else {
            continue;
        };
        let record: Vec<RouteFrozenSource> = serde_json::from_slice(&body)
            .map_err(|_| invalid_binding("a route's frozen source record is unreadable"))?;
        if let Some(source) = record.into_iter().find(|source| source.real_path == real) {
            return Ok(Some((source.sha256, source.size_bytes)));
        }
    }
    Ok(None)
}

/// Utility jobs whose loader takes one directory path receive the verified primary's path.
fn route_loader_locator(
    job_type: &JobType,
    dispatch_payload: &mut JsonObject,
    slot: &str,
    member: &PortableArtifactMember,
    selected: &Path,
) -> WorkerResult<()> {
    if matches!(
        job_type,
        JobType::DatasetAnalysis | JobType::TrainingCaption
    ) && slot == "modelManifestEntries/0"
        && member.role == ArtifactMemberRole::Primary
    {
        dispatch_payload.insert(
            "modelNameOrPath".to_owned(),
            serde_json::Value::String(selected.to_string_lossy().into_owned()),
        );
    }
    Ok(())
}

fn publish_verified_snapshot(
    staging: &Path,
    published: &Path,
    verify: impl FnOnce() -> WorkerResult<()>,
) -> WorkerResult<()> {
    if let Err(error) = verify() {
        remove_private_snapshot(staging).map_err(WorkerError::Io)?;
        return Err(error);
    }
    std::fs::rename(staging, published).map_err(WorkerError::Io)
}

fn verify_private_snapshot_member(
    snapshot_root: &Path,
    member: &PortableArtifactMember,
    selected_relative: &Path,
    relative_root: &Path,
    is_plan_layer: bool,
    allowed_roots: &[PathBuf],
) -> WorkerResult<()> {
    let target_root = snapshot_root.join(relative_root);
    let mapped = LocalArtifactMember {
        role: member.role.clone(),
        component_id: member.component_id.clone(),
        source: member.source.clone(),
        tier: member.tier.clone(),
        files: member
            .files
            .iter()
            .map(|file| LocalArtifactFile {
                relative_path: file.relative_path.clone(),
                path: if is_plan_layer {
                    target_root.join(selected_relative)
                } else {
                    target_root.join(&file.relative_path)
                },
            })
            .collect(),
    };
    let captured = PortableArtifactMember::capture(&mapped, allowed_roots)
        .map_err(|_| invalid_binding("private snapshot verification failed"))?;
    if captured != *member {
        return Err(invalid_binding(
            "private snapshot differs from admitted bytes",
        ));
    }
    Ok(())
}

fn common_file_parent(
    files: &[sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile],
) -> WorkerResult<PathBuf> {
    let mut parent = files
        .first()
        .and_then(|file| file.relative_path.parent())
        .ok_or_else(|| invalid_binding("selected auxiliary has no files"))?
        .to_path_buf();
    while !files
        .iter()
        .all(|file| file.relative_path.starts_with(&parent))
    {
        if !parent.pop() {
            return Ok(PathBuf::new());
        }
    }
    Ok(parent)
}

fn selected_revision_aux_directory(
    snapshot: &Path,
    files: &[sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile],
    source_id: &str,
) -> WorkerResult<PathBuf> {
    if source_id == "ltx_convert_upscaler" {
        // The converter joins the authored repository-relative filename to this directory.
        // Rebase at the snapshot root even when that filename has nested parents.
        Ok(snapshot.to_path_buf())
    } else {
        Ok(snapshot.join(common_file_parent(files)?))
    }
}

fn portable_cache_route_digest(route: &PortableRouteBinding) -> WorkerResult<String> {
    let encoded = serde_json::to_vec(route)
        .map_err(|error| invalid_binding(format!("cannot encode frozen route: {error}")))?;
    let mut digest = Sha256::new();
    digest.update(b"portable-route-cache-v4\0");
    digest.update(encoded);
    Ok(format!("{:x}", digest.finalize()))
}

/// Target at most eight route bundles and 256 GiB of reusable copies. The active entry always
/// remains even when its frozen closure alone exceeds the target; a later route can evict it.
fn prune_portable_snapshots(current: Option<&Path>, settings: &Settings) -> WorkerResult<()> {
    let root = settings.data_dir.join("models/portable-attempts");
    let _prune_lock =
        FileLock::open_exclusive(&root.join(".prune.lock")).map_err(WorkerError::Io)?;
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(&root).map_err(WorkerError::Io)? {
        let path = entry.map_err(WorkerError::Io)?.path();
        if !path.is_dir()
            || Some(path.as_path()) == current
            || !(path.join("bundle").exists()
                || path.join(ROUTE_FROZEN_SOURCES).exists()
                || path.join(LEGACY_ROUTE_VIEWS).exists())
        {
            continue;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.len() == 64 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        {
            let used = std::fs::metadata(path.join(".last-used"))
                .and_then(|metadata| metadata.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            let obsolete = !path.join(PORTABLE_BUNDLE_LAYOUT_MARKER).exists();
            entries.push((!obsolete, used, path));
        }
    }
    // Obsolete-layout bundles first (never read again), then least recently used.
    entries.sort_by_key(|(current_layout, used, _)| (*current_layout, *used));
    let mut total = match current {
        Some(current) => route_entry_bytes(current)?,
        None => 0,
    };
    let mut count = usize::from(current.is_some());
    for (_, _, path) in &entries {
        total = total.saturating_add(route_entry_bytes(path)?);
        count += 1;
    }
    for (current_layout, _, path) in entries {
        if current_layout
            && count <= PORTABLE_CACHE_MAX_ENTRIES
            && total <= PORTABLE_CACHE_MAX_BYTES
        {
            break;
        }
        let Ok(lease) = FileLock::try_open_exclusive(&path.join(".lease")) else {
            continue;
        };
        let bytes = route_entry_bytes(&path)?;
        remove_private_snapshot(&path.join("bundle")).map_err(WorkerError::Io)?;
        std::fs::remove_file(path.join(ROUTE_FROZEN_SOURCES)).ok();
        // Leftover of the dropped link-view layout: a tree of links into the hub. Removing it
        // never follows a link.
        match std::fs::remove_dir_all(path.join(LEGACY_ROUTE_VIEWS)) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => {
                return Err(WorkerError::Io(error));
            }
            _ => {}
        }
        // Keep the lock file and directory: a concurrent claimant may already be waiting on it.
        std::fs::remove_file(path.join(".last-used")).ok();
        drop(lease);
        total = total.saturating_sub(bytes);
        count -= 1;
    }
    Ok(())
}

/// A route directory left by the dropped link-view layout; retention reclaims it.
const LEGACY_ROUTE_VIEWS: &str = "views";

/// The bytes a route entry holds: its copied bundle. In-place members hold none here; lock,
/// marker and record files are not counted.
fn route_entry_bytes(route_dir: &Path) -> WorkerResult<u64> {
    let bundle = route_dir.join("bundle");
    if bundle.exists() {
        private_tree_bytes(&bundle)
    } else {
        Ok(0)
    }
}

pub(crate) fn private_tree_bytes(path: &Path) -> WorkerResult<u64> {
    let metadata = std::fs::symlink_metadata(path).map_err(WorkerError::Io)?;
    if metadata.file_type().is_symlink() {
        return Err(invalid_binding("private snapshot contains a link"));
    }
    if metadata.is_file() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.nlink() != 1 {
                return Err(invalid_binding(
                    "private snapshot file has a hard-link alias",
                ));
            }
        }
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Err(invalid_binding("private snapshot contains a non-file"));
    }
    let mut size = 0u64;
    for entry in std::fs::read_dir(path).map_err(WorkerError::Io)? {
        size = size.saturating_add(private_tree_bytes(&entry.map_err(WorkerError::Io)?.path())?);
    }
    Ok(size)
}

fn create_private_directory(path: &Path) -> WorkerResult<()> {
    let _created = match std::fs::create_dir(path) {
        Ok(()) => true,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => false,
        Err(error) => return Err(WorkerError::Io(error)),
    };
    let metadata = std::fs::symlink_metadata(path).map_err(WorkerError::Io)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid_binding(
            "private snapshot parent is not a regular directory",
        ));
    }
    #[cfg(unix)]
    if _created {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .map_err(WorkerError::Io)?;
    }
    Ok(())
}

fn ensure_private_directory(path: &Path) -> WorkerResult<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(WorkerError::Io)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid_binding("portable cache directory is not regular"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                .map_err(WorkerError::Io)?;
            let tightened = std::fs::symlink_metadata(path).map_err(WorkerError::Io)?;
            if !tightened.is_dir()
                || tightened.file_type().is_symlink()
                || tightened.permissions().mode() & 0o077 != 0
            {
                return Err(invalid_binding("portable cache directory is not private"));
            }
        }
    }
    Ok(())
}

fn create_private_tree(path: &Path) -> WorkerResult<()> {
    if path.exists() {
        return create_private_directory(path);
    }
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            create_private_tree(parent)?;
        }
    }
    create_private_directory(path)
}

fn freeze_private_snapshot(path: &Path) -> WorkerResult<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(WorkerError::Io)?;
    if metadata.file_type().is_symlink() {
        return Err(invalid_binding("private snapshot contains a link"));
    }
    if metadata.is_dir() {
        for entry in std::fs::read_dir(path).map_err(WorkerError::Io)? {
            freeze_private_snapshot(&entry.map_err(WorkerError::Io)?.path())?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o500))
                .map_err(WorkerError::Io)?;
        }
    } else if metadata.is_file() {
        let mut permissions = metadata.permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(path, permissions).map_err(WorkerError::Io)?;
    } else {
        return Err(invalid_binding("private snapshot contains a non-file"));
    }
    Ok(())
}

fn rewrite_loader_locator(
    payload: &mut JsonObject,
    model_slot: &str,
    member: &PortableArtifactMember,
    selected: &Path,
    is_selected_aux: bool,
) -> WorkerResult<()> {
    if member.component_id.as_deref() == Some("selected_text_encoder") {
        if model_slot != "modelManifestEntry" {
            return Err(invalid_binding(
                "selected text encoder has the wrong model slot",
            ));
        }
        let resolution = payload
            .get("modelManifestEntry")
            .and_then(|entry| entry.get("resolvedTextEncoder"))
            .ok_or_else(|| invalid_binding("selected text encoder resolution disappeared"))?;
        let is_file = match resolution
            .get("sourceKind")
            .and_then(serde_json::Value::as_str)
        {
            Some("file") => true,
            Some("directory") => false,
            _ => {
                return Err(invalid_binding(
                    "selected text encoder source kind disappeared",
                ))
            }
        };
        let loader_path = if is_file {
            let original = resolution
                .get("path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid_binding("selected text encoder path disappeared"))?;
            let name = Path::new(original)
                .file_name()
                .ok_or_else(|| invalid_binding("selected text encoder file has no name"))?;
            if !member
                .files
                .iter()
                .any(|file| file.relative_path == Path::new(name))
            {
                return Err(invalid_binding(
                    "selected text encoder file is outside its frozen closure",
                ));
            }
            selected.join(name)
        } else {
            selected.to_path_buf()
        };
        let loader_path = crate::paths::normalize_absolute_path(&loader_path)?;
        let source = if is_file {
            gen_core::WeightsSource::File(loader_path.clone())
        } else {
            gen_core::WeightsSource::Dir(loader_path.clone())
        };
        let worker_id =
            crate::text_encoder_selection::opaque_id(&source).map_err(WorkerError::Io)?;
        payload
            .get_mut("advanced")
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| invalid_binding("selected text encoder has no advanced options"))?
            .insert("textEncoderModel".to_owned(), json!(worker_id));
        let resolution = payload
            .get_mut("modelManifestEntry")
            .and_then(|entry| entry.get_mut("resolvedTextEncoder"))
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| invalid_binding("selected text encoder resolution disappeared"))?;
        resolution.insert("selectionId".to_owned(), json!(worker_id));
        resolution.insert("path".to_owned(), json!(loader_path));
        return Ok(());
    }
    let path = selected
        .to_str()
        .ok_or_else(|| invalid_binding("private snapshot path is not UTF-8"))?
        .to_owned();
    if is_selected_aux || matches!(member.source, PortableSourceIdentity::WorkerClaim { .. }) {
        let source_id = member
            .component_id
            .as_ref()
            .ok_or_else(|| invalid_binding("auxiliary source has no ID"))?;
        let advanced = payload
            .entry("advanced".to_owned())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| invalid_binding("advanced options are not an object"))?;
        if source_id == "strict_control" {
            if let Some(weights) = advanced
                .get_mut("controlWeights")
                .and_then(serde_json::Value::as_object_mut)
            {
                // Some legacy strict-control loaders read this field directly. Give them the
                // verified attempt copy, never the API host's recorded physical locator.
                weights.insert("path".to_owned(), json!(path.clone()));
            }
        }
        advanced
            .entry("__portableAuxPaths".to_owned())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| invalid_binding("private auxiliary path map is not an object"))?
            .insert(source_id.clone(), json!(path));
        if let Some(plan) = payload
            .get_mut("plan")
            .and_then(serde_json::Value::as_object_mut)
        {
            plan.entry("__portableAuxPaths".to_owned())
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .ok_or_else(|| {
                    invalid_binding("private training auxiliary path map is not an object")
                })?
                .insert(source_id.clone(), json!(path));
        }
        return Ok(());
    }
    if matches!(
        member.role,
        ArtifactMemberRole::Lora | ArtifactMemberRole::Control
    ) {
        let slot = member
            .component_id
            .as_deref()
            .ok_or_else(|| invalid_binding("bound adapter has no slot"))?;
        let mut root = serde_json::Value::Object(std::mem::take(payload));
        let record = root
            .pointer_mut(&format!("/{slot}"))
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| invalid_binding("bound adapter slot disappeared before dispatch"))?;
        // The actual route may read any of these legacy locator spellings. Repoint every one in
        // this private copy so a secondary resolver cannot reopen the original after verification.
        for key in ["modelPath", "installedPath", "sourcePath", "path"] {
            record.insert(key.to_owned(), serde_json::Value::String(path.clone()));
        }
        if let Some(source) = record
            .get_mut("source")
            .and_then(serde_json::Value::as_object_mut)
        {
            source.insert("path".to_owned(), serde_json::Value::String(path));
        }
        *payload = root
            .as_object_mut()
            .map(std::mem::take)
            .ok_or_else(|| invalid_binding("private payload root changed shape"))?;
        return Ok(());
    }
    let owns_advanced_model_path = model_slot == "modelManifestEntry"
        || (model_slot == "modelManifestEntries/0"
            && ["modelManifestEntry", "baseModelManifestEntry"]
                .into_iter()
                .all(|slot| payload.get(slot).is_none_or(serde_json::Value::is_null)));
    if member.role == ArtifactMemberRole::Primary && owns_advanced_model_path {
        if let Some(advanced) = payload
            .get_mut("advanced")
            .and_then(serde_json::Value::as_object_mut)
        {
            if advanced.contains_key("modelPath") {
                advanced.insert(
                    "modelPath".to_owned(),
                    serde_json::Value::String(path.clone()),
                );
            }
        }
    }
    let entry = model_slot_entry_mut(payload, model_slot)
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| invalid_binding("bound model slot disappeared before dispatch"))?;
    let changed = if member.role == ArtifactMemberRole::Primary {
        for key in ["modelPath", "installedPath"] {
            if entry.contains_key(key) {
                entry.insert(key.to_owned(), serde_json::Value::String(path.clone()));
            }
        }
        if let Some(paths) = entry
            .get_mut("paths")
            .and_then(serde_json::Value::as_object_mut)
        {
            paths.insert("model".to_owned(), serde_json::Value::String(path.clone()));
        } else {
            entry.insert("paths".to_owned(), json!({"model": path}));
        }
        true
    } else {
        let id = member
            .component_id
            .as_deref()
            .ok_or_else(|| invalid_binding("bound component has no id"))?;
        set_existing_entry_locator(entry, portable_component_locator_key(model_slot, id), &path)
    };
    if !changed {
        return Err(invalid_binding("bound loader locator cannot be rewritten"));
    }
    rewrite_matching_plan_path(payload, model_slot, &path)?;
    Ok(())
}

fn rewrite_matching_plan_path(
    payload: &mut JsonObject,
    model_slot: &str,
    selected: &str,
) -> WorkerResult<()> {
    let Some(plan) = payload
        .get_mut("plan")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Ok(());
    };
    let Some(target) = plan
        .get_mut("target")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Err(invalid_binding("training plan has no target"));
    };
    if model_slot == "baseModelManifestEntry" {
        target.insert("baseModelPath".to_owned(), json!(selected));
    }
    Ok(())
}

fn rebind_revision_primary_paths(
    route: &PortableRouteBinding,
    settings: &Settings,
    dispatch: &mut JsonObject,
) -> WorkerResult<()> {
    for model in &route.models {
        let primary = model
            .members
            .iter()
            .find(|member| member.role == ArtifactMemberRole::Primary)
            .ok_or_else(|| invalid_binding("bound model has no primary source"))?;
        let PortableSourceIdentity::Revision { identity } = &primary.source else {
            continue;
        };
        let (_, snapshot) = sceneworks_core::hf_home::model_source_library(&settings.data_dir)
            .discover_snapshot(&identity.repository, Some(&identity.revision))
            .map_err(|error| invalid_binding(error.to_string()))?;
        let selected = if let Some(tier) = model
            .tier
            .as_deref()
            .filter(|tier| snapshot.join(tier).is_dir())
        {
            snapshot.join(tier)
        } else {
            snapshot
        };
        rewrite_loader_locator(dispatch, &model.slot, primary, &selected, false)?;
    }
    Ok(())
}

fn rewrite_training_plan_source(
    route: &PortableRouteBinding,
    settings: &Settings,
    payload: &mut JsonObject,
) -> WorkerResult<()> {
    let Some(target) = payload.get("plan").and_then(|plan| plan.get("target")) else {
        return Ok(());
    };
    let model_id = target
        .get("baseModel")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_binding("training plan has no base model identity"))?;
    let model = route
        .models
        .iter()
        .find(|model| model.slot == "baseModelManifestEntry" && model.model_id == model_id)
        .or_else(|| route.models.iter().find(|model| model.model_id == model_id))
        .ok_or_else(|| invalid_binding("training base model is not bound"))?;
    let primary = model
        .members
        .iter()
        .find(|member| member.role == ArtifactMemberRole::Primary)
        .ok_or_else(|| invalid_binding("training base model has no primary source"))?;
    let selected = match &primary.source {
        PortableSourceIdentity::Revision { identity } => {
            let source_library = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
            let (_, snapshot) = source_library
                .discover_snapshot(&identity.repository, Some(&identity.revision))
                .map_err(|error| invalid_binding(error.to_string()))?;
            let tier = model.tier.as_deref().unwrap_or("");
            if !tier.is_empty() && snapshot.join(tier).is_dir() {
                snapshot.join(tier)
            } else {
                snapshot
            }
        }
        PortableSourceIdentity::Digest { .. } => {
            let entry = model_slot_entry_mut(payload, &model.slot)
                .ok_or_else(|| invalid_binding("training model slot disappeared"))?;
            entry
                .get("paths")
                .and_then(|paths| paths.get("model"))
                .or_else(|| entry.get("modelPath"))
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from)
                .ok_or_else(|| {
                    invalid_binding("training source was not rebound to a private bundle")
                })?
        }
        PortableSourceIdentity::WorkerClaim { .. } => {
            return Err(invalid_binding(
                "training primary cannot use an auxiliary worker claim",
            ));
        }
    };
    let selected = selected
        .to_str()
        .ok_or_else(|| invalid_binding("training source path is not UTF-8"))?;
    payload
        .get_mut("plan")
        .and_then(|plan| plan.get_mut("target"))
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| invalid_binding("training target disappeared"))?
        .insert("baseModelPath".to_owned(), json!(selected));
    Ok(())
}

fn model_slot_entry_mut<'a>(
    payload: &'a mut JsonObject,
    slot: &str,
) -> Option<&'a mut serde_json::Value> {
    if let Some(index) = slot.strip_prefix("modelManifestEntries/") {
        return payload
            .get_mut("modelManifestEntries")?
            .as_array_mut()?
            .get_mut(index.parse::<usize>().ok()?);
    }
    payload.get_mut(slot)
}

/// A worker-local proof of one pinned primary source selected by the portable guard. This is
/// built only after route verification and copy-then-verify, and is carried in Settings while the
/// guard keeps the private snapshot leased. A job payload cannot create one.
#[derive(Clone, Debug)]
pub(crate) struct VerifiedRevisionSource {
    pub(crate) model_id: String,
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) path: PathBuf,
}

fn revision_primary_handoffs(
    route: &PortableRouteBinding,
    dispatch: &JsonObject,
) -> WorkerResult<Vec<VerifiedRevisionSource>> {
    route
        .models
        .iter()
        .filter_map(|model| {
            let primary = model
                .members
                .iter()
                .find(|member| member.role == ArtifactMemberRole::Primary)?;
            let PortableSourceIdentity::Revision { identity } = &primary.source else {
                return None;
            };
            Some((model, identity))
        })
        .map(|(model, identity)| {
            let (_, entry) = payload_model_slots(dispatch)
                .into_iter()
                .find(|(slot, _)| slot == &model.slot)
                .ok_or_else(|| invalid_binding("verified revision model slot disappeared"))?;
            let path = entry
                .pointer("/paths/model")
                .or_else(|| entry.get("modelPath"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid_binding("verified revision primary has no loader path"))?;
            Ok(VerifiedRevisionSource {
                model_id: model.model_id.clone(),
                repository: identity.repository.clone(),
                revision: identity.revision.clone(),
                path: PathBuf::from(path),
            })
        })
        .collect()
}

/// Rebind every model in the claimed route to this worker's installation before dispatch.
fn verify_portable_route(
    route: &PortableRouteBinding,
    job_type: &JobType,
    payload: &JsonObject,
    settings: &Settings,
    plans: &[EntryPlan],
    converted_co_requirements: &[ExternalArtifactRequirement],
    cancellation: Option<&PortableOverlayWaitCancellation>,
) -> WorkerResult<()> {
    route
        .validate()
        .map_err(|error| invalid_binding(error.to_string()))?;
    if route.backend != crate::portable_runtime_backend(settings)
        || route.platform != std::env::consts::OS
    {
        return Err(invalid_binding(
            "the claimed route differs from this worker's backend or platform",
        ));
    }
    let slots = payload_model_slots(payload);
    let expected_slots: Vec<_> = slots.iter().map(|(slot, _)| slot.as_str()).collect();
    let bound_slots: Vec<_> = route
        .models
        .iter()
        .map(|model| model.slot.as_str())
        .collect();
    if expected_slots != bound_slots {
        return Err(invalid_binding(
            "the admitted model slot set differs from the claim",
        ));
    }
    let claim = serde_json::Value::Object(payload.clone());
    let mut selected_adapter_slots =
        enumerate_portable_array_slots(&claim, &["loras", "lokrs", "controls"])
            .into_iter()
            .filter_map(|(slot, record)| {
                let field = slot.rsplit('/').nth(1)?;
                if field == "controls" {
                    return record
                        .get("modelPath")
                        .or_else(|| record.get("installedPath"))
                        .or_else(|| record.get("path"))
                        .and_then(serde_json::Value::as_str)
                        .map(|_| slot);
                }
                let pure_phase_reference = record.get("index").is_some()
                    && record.get("id").is_none()
                    && sceneworks_core::lora_family::adapter_record_path(record).is_none();
                (!pure_phase_reference).then_some(slot)
            })
            .collect::<Vec<_>>();
    selected_adapter_slots.sort();
    let mut bound_adapter_slots = route
        .models
        .iter()
        .flat_map(|model| &model.members)
        .filter(|member| {
            matches!(
                member.role,
                ArtifactMemberRole::Lora | ArtifactMemberRole::Control
            )
        })
        .filter_map(|member| {
            member
                .component_id
                .as_deref()
                .filter(|id| is_portable_array_slot(id))
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    bound_adapter_slots.sort();
    if selected_adapter_slots != bound_adapter_slots {
        return Err(invalid_binding(
            "the admitted adapter slot set differs from the claim",
        ));
    }
    for (model, (_, entry)) in route.models.iter().zip(slots) {
        if matches!(
            job_type,
            JobType::DatasetAnalysis | JobType::TrainingCaption
        ) && model.slot == "modelManifestEntries/0"
        {
            let primary = model
                .members
                .iter()
                .find(|member| member.role == ArtifactMemberRole::Primary)
                .ok_or_else(|| invalid_binding("utility model has no admitted primary"))?;
            if let PortableSourceIdentity::Revision { identity } = &primary.source {
                let selected = payload
                    .get("modelNameOrPath")
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|selection| !selection.is_empty());
                if selected.is_some_and(|selection| selection != identity.repository) {
                    return Err(invalid_binding(
                        "utility model selection differs from the frozen publisher source",
                    ));
                }
            }
        }
        let selected_encoder = matches!(
            job_type,
            JobType::ImageGenerate | JobType::ImageEdit | JobType::ImageDetail
        ) && model.slot == "modelManifestEntry"
            && payload
                .get("advanced")
                .and_then(|advanced| advanced.get("textEncoderModel"))
                .and_then(serde_json::Value::as_str)
                .is_some_and(|id| id != "default" && !id.is_empty());
        let bound_encoder = model
            .members
            .iter()
            .filter(|member| member.component_id.as_deref() == Some("selected_text_encoder"))
            .collect::<Vec<_>>();
        if selected_encoder != (bound_encoder.len() == 1)
            || bound_encoder.len() > 1
            || bound_encoder.iter().any(|member| {
                member.role != ArtifactMemberRole::OptionalComponent
                    || !matches!(member.source, PortableSourceIdentity::Digest { .. })
            })
        {
            return Err(invalid_binding(
                "selected text encoder closure differs from the admitted route",
            ));
        }
        let required = selected_portable_aux_ids(job_type, payload, entry, &route.backend)?;
        for source_id in &required {
            let admitted = model
                .members
                .iter()
                .find(|member| member.component_id.as_deref() == Some(source_id.as_str()));
            let Some(admitted) = admitted else {
                return Err(invalid_binding(format!(
                    "the admitted route omits selected auxiliary source {source_id}"
                )));
            };
            if source_id == "iris_depth_parent"
                && !matches!(
                    &admitted.source,
                    PortableSourceIdentity::WorkerClaim { source_id, .. }
                        if matches!(source_id.as_str(), "iris_depth_export" | "iris_generation_text_encoder")
                )
            {
                return Err(invalid_binding(
                    "Iris depth parent has no authorized source",
                ));
            }
        }
        verify_portable_binding(
            model,
            entry,
            payload,
            settings,
            plans,
            &required,
            job_type,
            cancellation,
        )?;
    }
    // The complete route must cover each selected HF requirement, including the resident base of
    // an imported checkpoint. A valid primary cannot hide an omitted protected component.
    for (requirement, resident_base) in plans
        .iter()
        .flat_map(|plan| {
            plan.requirements
                .iter()
                .map(move |requirement| (requirement, plan.resident_base))
        })
        .chain(
            converted_co_requirements
                .iter()
                .map(|requirement| (requirement, false)),
        )
    {
        if requirement.is_primary
            && matches!(
                job_type,
                JobType::DatasetAnalysis | JobType::TrainingCaption
            )
            && route.models.iter().any(|model| {
                model.slot == "modelManifestEntries/0"
                    && model.members.iter().any(|member| {
                        member.role == ArtifactMemberRole::Primary
                            && matches!(member.source, PortableSourceIdentity::Digest { .. })
                    })
            })
        {
            // The caller selected a different complete utility snapshot. Its Digest and
            // current worker claim replace only the catalog primary, never co-requisites.
            continue;
        }
        let covered = route
            .models
            .iter()
            .flat_map(|model| &model.members)
            .any(|member| {
                let PortableSourceIdentity::Revision { identity } = &member.source else {
                    return false;
                };
                identity.repository == requirement.repository
                    && Some(identity.revision.as_str()) == requirement.revision.as_deref()
                    && identity.variant == requirement.variant
                    && (if resident_base {
                        member.role == ArtifactMemberRole::RequiredComponent
                            && member
                                .component_id
                                .as_deref()
                                .is_some_and(|id| id.starts_with("resident_base/downloads/"))
                    } else {
                        requirement.is_primary == (member.role == ArtifactMemberRole::Primary)
                    })
                    && requirement.files.iter().all(|selector| {
                        glob::Pattern::new(&selector.to_string_lossy()).is_ok_and(|pattern| {
                            member
                                .files
                                .iter()
                                .any(|file| pattern.matches_path(&file.relative_path))
                        })
                    })
            });
        if !covered {
            return Err(invalid_binding(
                "the admitted route omits a selected model component",
            ));
        }
        // A wildcard declaration is a request to load the whole matching directory subset.
        // Merely proving one frozen file matched would let an extra local file be loaded outside
        // the binding. Compare the full selected local set with the captured concrete paths.
        for selector in &requirement.files {
            let selector_text = selector
                .to_str()
                .ok_or_else(|| invalid_binding("artifact selector is not portable UTF-8"))?;
            if !selector_text.contains(['*', '?', '[']) {
                continue;
            }
            let revision = requirement.revision.as_deref().ok_or_else(|| {
                invalid_binding("a wildcard model source has no immutable revision")
            })?;
            let snapshot = discover_frozen_selector_snapshot(
                settings,
                route.models.iter().flat_map(|model| &model.members),
                &requirement.repository,
                revision,
                &requirement.variant,
                requirement.is_primary,
            )?;
            let mut local_files = recursive_snapshot_matches(&snapshot, selector_text)?;
            let matcher = glob::Pattern::new(selector_text)
                .map_err(|error| invalid_binding(error.to_string()))?;
            let mut frozen_files = route
                .models
                .iter()
                .flat_map(|model| &model.members)
                .filter(|member| {
                    matches!(&member.source, PortableSourceIdentity::Revision { identity }
                        if identity.repository == requirement.repository
                            && identity.revision == revision
                            && identity.variant == requirement.variant)
                        && requirement.is_primary == (member.role == ArtifactMemberRole::Primary)
                })
                .flat_map(|member| &member.files)
                .filter(|file| matcher.matches(&file.relative_path.to_string_lossy()))
                .map(|file| file.relative_path.clone())
                .collect::<Vec<_>>();
            local_files.sort();
            local_files.dedup();
            frozen_files.sort();
            frozen_files.dedup();
            if local_files != frozen_files {
                return Err(invalid_binding(
                    "the selected local artifact file set differs from the admitted closure",
                ));
            }
        }
    }
    Ok(())
}

fn is_portable_array_slot(id: &str) -> bool {
    let mut segments = id.rsplit('/');
    let Some(index) = segments.next() else {
        return false;
    };
    let Some(field) = segments.next() else {
        return false;
    };
    matches!(field, "loras" | "lokrs" | "controls") && index.parse::<usize>().is_ok()
}

/// Verify one named model slot against its own manifest entry, then hash the actual selected
/// worker paths while the route's source lease is held.
#[allow(clippy::too_many_arguments)]
fn verify_portable_binding(
    binding: &PortableModelBinding,
    entry: &serde_json::Value,
    payload: &JsonObject,
    settings: &Settings,
    plans: &[EntryPlan],
    selected_aux_ids: &[String],
    job_type: &JobType,
    cancellation: Option<&PortableOverlayWaitCancellation>,
) -> WorkerResult<()> {
    binding
        .validate()
        .map_err(|error| invalid_binding(error.to_string()))?;
    let selected_model = entry.get("id").and_then(serde_json::Value::as_str);
    if selected_model != Some(binding.model_id.as_str()) {
        return Err(invalid_binding(
            "the selected model differs from its admitted identity",
        ));
    }
    let backend = crate::portable_runtime_backend(settings);
    if binding.backend != backend {
        return Err(invalid_binding(
            "the worker backend differs from the admitted route",
        ));
    }

    let requirements: Vec<_> = plans
        .iter()
        .flat_map(|plan| {
            plan.requirements
                .iter()
                .map(move |requirement| (requirement, plan.resident_base))
        })
        .collect();
    let requested_variant =
        bound_runtime_variant(job_type, payload, &binding.slot, entry, Some(binding))?;
    let mut selected_manifest = selected_portable_model_artifact_closure(
        entry,
        binding.backend.as_str(),
        std::env::consts::OS,
        requested_variant.as_deref(),
        payload,
    )
    .map_err(invalid_binding)?;
    if entry
        .get("downloads")
        .and_then(serde_json::Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        if let Some(base) = selected_resident_base_closure_for_entry(entry, std::env::consts::OS) {
            selected_manifest
                .as_object_mut()
                .ok_or_else(|| invalid_binding("selected model manifest is invalid"))?
                .insert(
                    "downloads".to_owned(),
                    serde_json::Value::Array(resident_base_component_downloads(&base)),
                );
        }
    }
    // Raw ModelConvert admission can freeze the immutable source privately while the synthetic
    // public manifest entry only carries repo/file hints. The binding is the source authority;
    // require its primary repo to agree with the converter's actual sourceRepo before using it.
    let raw_conversion_primary = *job_type == JobType::ModelConvert
        && binding.slot == "modelManifestEntries/0"
        && payload.get("sourceRepo").and_then(serde_json::Value::as_str)
            .is_some_and(|repo| binding.model_id == format!("model_convert:{repo}"))
        && binding.members.iter().find(|member| member.role == ArtifactMemberRole::Primary).is_some_and(|primary| {
            matches!(&primary.source, PortableSourceIdentity::Revision { identity }
                if payload.get("sourceRepo").and_then(serde_json::Value::as_str) == Some(identity.repository.as_str())
                    && entry.get("downloads").and_then(serde_json::Value::as_array).is_some_and(|downloads| downloads.iter().any(|download|
                        download.get("repo").and_then(serde_json::Value::as_str) == Some(identity.repository.as_str())
                            && download.get("revision").is_none()
                    ))
            )
        });
    let mut local = Vec::with_capacity(binding.members.len());
    for member in &binding.members {
        let files = match &member.source {
            PortableSourceIdentity::Revision { identity } => {
                let matching: Vec<_> = requirements
                    .iter()
                    .copied()
                    .filter(|(requirement, resident_base)| {
                        requirement.repository == identity.repository
                            && requirement.revision.as_deref() == Some(identity.revision.as_str())
                            && requirement.variant == identity.variant
                            && (if *resident_base {
                                member.role == ArtifactMemberRole::RequiredComponent
                                    && member.component_id.as_deref().is_some_and(|id| {
                                        id.starts_with("resident_base/downloads/")
                                    })
                            } else {
                                requirement.is_primary
                                    == (member.role == ArtifactMemberRole::Primary)
                            })
                    })
                    .collect();
                let selected_auxiliary = member
                    .component_id
                    .as_ref()
                    .is_some_and(|id| selected_aux_ids.contains(id));
                let selected_manifest = selected_manifest_row_covers_member(
                    member,
                    &selected_manifest,
                    raw_conversion_primary,
                );
                if !selected_auxiliary
                    && !selected_manifest
                    && (matching.is_empty()
                        || member.files.iter().any(|file| {
                            !matching.iter().any(|(requirement, _)| {
                                requirement.files.iter().any(|selector| {
                                    glob::Pattern::new(&selector.to_string_lossy()).is_ok_and(
                                        |pattern| {
                                            pattern.matches(&file.relative_path.to_string_lossy())
                                        },
                                    )
                                })
                            })
                        }))
                {
                    return Err(invalid_binding(
                        "the claimed artifact revision or file closure differs from the selected model",
                    ));
                }
                let snapshot = discover_frozen_member_snapshot(settings, identity, &member.files)?;
                member
                    .files
                    .iter()
                    .map(|file| LocalArtifactFile {
                        relative_path: file.relative_path.clone(),
                        path: snapshot.join(&file.relative_path),
                    })
                    .collect()
            }
            PortableSourceIdentity::Digest { .. } | PortableSourceIdentity::WorkerClaim { .. } => {
                let selected = if let Some(layer) =
                    selected_checkpoint_plan_layer(member, binding, entry, settings)?
                {
                    layer.source_path
                } else {
                    selected_local_artifact_path(
                        member, binding, entry, payload, settings, job_type,
                    )?
                };
                let is_file = selected.is_file();
                if !is_file && !selected.is_dir() {
                    return Err(invalid_binding(
                        "the selected local artifact is unavailable",
                    ));
                }
                if is_file && member.files.len() != 1 {
                    return Err(invalid_binding(
                        "a single local artifact file cannot bind a multi-file closure",
                    ));
                }
                member
                    .files
                    .iter()
                    .map(|file| LocalArtifactFile {
                        relative_path: file.relative_path.clone(),
                        path: if is_file {
                            selected.clone()
                        } else {
                            selected.join(&file.relative_path)
                        },
                    })
                    .collect()
            }
        };
        local.push(LocalArtifactMember {
            role: member.role.clone(),
            component_id: member.component_id.clone(),
            source: member.source.clone(),
            tier: member.tier.clone(),
            files,
        });
    }
    let primary = binding
        .members
        .iter()
        .find(|member| member.role == ArtifactMemberRole::Primary)
        .expect("portable binding validates exactly one primary");
    let selected = selected_portable_requirements_for_model(
        entry,
        binding.backend.as_str(),
        std::env::consts::OS,
        requested_variant.as_deref(),
        payload,
        &settings.data_dir,
    )
    .map_err(invalid_binding)?;
    let manifest_variant = selected_manifest
        .get("downloads")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .find(|row| !is_co_requisite_download(row))
        .map(|row| {
            row.get("variant")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("default")
                .to_owned()
        });
    let plan_backed_local_primary = binding.checkpoint_plan.is_some()
        && matches!(primary.source, PortableSourceIdentity::Digest { .. })
        && entry
            .pointer("/importPlan/checkpointId")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|id| !id.is_empty())
        && selected
            .requirements
            .iter()
            .all(|requirement| !requirement.is_primary)
        && selected_manifest
            .get("downloads")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|rows| rows.iter().all(is_co_requisite_download));
    let selected_variant = selected
        .requirements
        .iter()
        .find(|requirement| requirement.is_primary)
        .map(|requirement| requirement.variant.clone())
        .or(manifest_variant)
        .or_else(|| requested_variant.clone())
        .unwrap_or_else(|| {
            // A verified imported checkpoint remains the primary source even when its catalog
            // downloads contain pinned publisher companions. Those rows do not select a primary
            // tier; the API freezes the local plan-backed Digest as variant `local`.
            if plan_backed_local_primary
                || entry
                    .get("downloads")
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(Vec::is_empty)
            {
                "local".to_owned()
            } else {
                "default".to_owned()
            }
        });
    if binding.variant != selected_variant || primary.tier != binding.tier {
        return Err(invalid_binding(
            "the selected model tier differs from its admitted identity",
        ));
    }
    let mut allowed_roots = crate::paths::allowed_model_roots(settings)?;
    for member in &binding.members {
        let plan_layer = selected_checkpoint_plan_layer(member, binding, entry, settings)?;
        if matches!(member.source, PortableSourceIdentity::WorkerClaim { .. })
            || plan_layer.is_some()
        {
            let selected = if let Some(layer) = plan_layer {
                layer.source_path
            } else {
                selected_local_artifact_path(member, binding, entry, payload, settings, job_type)?
            };
            allowed_roots.push(selected.parent().unwrap_or(&selected).to_path_buf());
        }
    }
    // Hash each frozen file at most once per host while it is unchanged; the in-place pass in
    // `prepare_portable_snapshot` then reuses this verdict.
    let cache = verified_source_cache(settings);
    let sized = local
        .iter()
        .flat_map(|member| &member.files)
        .filter_map(|file| {
            let real = std::fs::canonicalize(&file.path).ok()?;
            let size = std::fs::metadata(&real).ok()?.len();
            Some((real, size))
        })
        .collect::<Vec<_>>();
    with_verification_progress(&sized, |progress| {
        binding.verify_local_with(&local, &allowed_roots, &|path, expected| {
            verify_source_once(&cache, progress, path, expected, cancellation).map_err(|error| {
                ArtifactContractError(if error.kind() == io::ErrorKind::InvalidData {
                    "artifact file bytes differ from admitted identity".to_owned()
                } else {
                    format!("artifact file hash failed: {error}")
                })
            })
        })
    })
    .map_err(|error| {
        if verification_canceled(cancellation) {
            WorkerError::Canceled("frozen source verification canceled".to_owned())
        } else {
            invalid_binding(error.to_string())
        }
    })?;
    let verified_derivative_primary =
        matches!(primary.source, PortableSourceIdentity::Digest { .. })
            && (crate::portable_source_claims::is_converted_builtin_model_id(&binding.model_id)
                || matches!(
                    job_type,
                    JobType::DatasetAnalysis | JobType::TrainingCaption
                ) && binding.slot == "modelManifestEntries/0");
    verify_manifest_globs(
        binding,
        &selected_manifest,
        settings,
        verified_derivative_primary,
    )
}

fn selected_manifest_row_covers_member(
    member: &PortableArtifactMember,
    selected: &serde_json::Value,
    raw_conversion_primary: bool,
) -> bool {
    let PortableSourceIdentity::Revision { identity } = &member.source else {
        return false;
    };
    selected
        .get("downloads")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|row| row.get("required").and_then(serde_json::Value::as_str) != Some("soft"))
        .any(|row| {
            let raw_primary = raw_conversion_primary
                && member.role == ArtifactMemberRole::Primary
                && row.get("revision").is_none()
                && row.get("files").is_none();
            if row.get("repo").and_then(serde_json::Value::as_str)
                != Some(identity.repository.as_str())
                || !raw_primary
                    && row.get("revision").and_then(serde_json::Value::as_str)
                        != Some(identity.revision.as_str())
                || !raw_primary
                    && row
                        .get("variant")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("default")
                        != identity.variant
                || (member.role == ArtifactMemberRole::Primary) != !is_co_requisite_download(row)
            {
                return false;
            }
            if raw_primary {
                return true;
            }
            let Some(selectors) = row.get("files").and_then(serde_json::Value::as_array) else {
                return false;
            };
            selectors.is_empty()
                || member.files.iter().all(|file| {
                    selectors
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .any(|selector| {
                            glob::Pattern::new(selector).is_ok_and(|pattern| {
                                pattern.matches(&file.relative_path.to_string_lossy())
                            })
                        })
                })
        })
}

/// Match the same relative file tree that admission's `glob::Pattern::matches` sees. A filesystem
/// glob of `snapshot/q4/*` stops at a directory, while the admitted selector includes nested
/// files such as `q4/transformer/model.safetensors`.
fn recursive_snapshot_matches(snapshot: &Path, selector: &str) -> WorkerResult<Vec<PathBuf>> {
    let pattern = glob::Pattern::new(selector)
        .map_err(|_| invalid_binding("artifact selector is invalid"))?;
    // Limit the walk to the literal directory prefix. A selected q4/* closure must not depend on
    // whether an unrelated q8 sibling contains a broken link or a non-file entry.
    let mut prefix = PathBuf::new();
    for component in Path::new(selector).components() {
        let std::path::Component::Normal(segment) = component else {
            return Err(invalid_binding(
                "artifact selector is not repository-relative",
            ));
        };
        let segment = segment
            .to_str()
            .ok_or_else(|| invalid_binding("artifact selector is not UTF-8"))?;
        if segment.contains(['*', '?', '[']) {
            break;
        }
        prefix.push(segment);
    }
    let selected_root = snapshot.join(prefix);
    if !selected_root.exists() {
        return Ok(Vec::new());
    }
    if !selected_root.is_dir() || selected_root.is_symlink() {
        return Err(invalid_binding("selected artifact subtree is invalid"));
    }
    let mut selected = Vec::new();
    let mut pending = vec![selected_root];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)
            .map_err(|_| invalid_binding("artifact snapshot is unavailable"))?
        {
            let entry = entry.map_err(|_| invalid_binding("artifact snapshot is unavailable"))?;
            let path = entry.path();
            let kind = entry
                .file_type()
                .map_err(|_| invalid_binding("artifact snapshot is unavailable"))?;
            if kind.is_dir() {
                pending.push(path);
                continue;
            }
            let relative = path
                .strip_prefix(snapshot)
                .map_err(|_| invalid_binding("artifact snapshot is invalid"))?;
            if kind.is_file() || kind.is_symlink() && path.is_file() {
                if pattern.matches(&relative.to_string_lossy()) {
                    selected.push(relative.to_path_buf());
                }
            } else if kind.is_symlink() && path.is_dir()
                || pattern.matches(&relative.to_string_lossy())
            {
                return Err(invalid_binding("artifact snapshot has a non-file entry"));
            }
        }
    }
    Ok(selected)
}

fn verify_manifest_globs(
    binding: &PortableModelBinding,
    selected: &serde_json::Value,
    settings: &Settings,
    verified_derivative_primary: bool,
) -> WorkerResult<()> {
    for download in selected
        .get("downloads")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| {
            download.get("required").and_then(serde_json::Value::as_str) != Some("soft")
                && (!verified_derivative_primary || is_co_requisite_download(download))
        })
    {
        let Some(repo) = download.get("repo").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let Some(revision) = download.get("revision").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let variant = download
            .get("variant")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        let primary = !is_co_requisite_download(download);
        let Some(files) = download.get("files").and_then(serde_json::Value::as_array) else {
            continue;
        };
        // Admission expands the legacy empty selector to `*` before freezing its exact
        // pinned ledger. Verify the same whole-snapshot set on this worker.
        let selectors = if files.is_empty() {
            vec!["*"]
        } else {
            files
                .iter()
                .filter_map(serde_json::Value::as_str)
                .filter(|selector| selector.contains(['*', '?', '[']))
                .collect()
        };
        for selector in selectors {
            let snapshot = discover_frozen_selector_snapshot(
                settings,
                binding.members.iter(),
                repo,
                revision,
                variant,
                primary,
            )?;
            let mut actual = recursive_snapshot_matches(&snapshot, selector)?;
            let mut frozen = binding.members.iter()
                .filter(|member| {
                    matches!(&member.source, PortableSourceIdentity::Revision { identity }
                        if identity.repository == repo && identity.revision == revision && identity.variant == variant)
                        && (member.role == ArtifactMemberRole::Primary) == primary
                })
                .flat_map(|member| &member.files)
                .filter(|file| sceneworks_core::model_artifacts::artifact_selection::pattern_matches(
                    selector, &file.relative_path.to_string_lossy()))
                .map(|file| file.relative_path.clone())
                .collect::<Vec<_>>();
            actual.sort();
            actual.dedup();
            frozen.sort();
            frozen.dedup();
            if actual != frozen {
                return Err(invalid_binding(
                    "the selected local artifact file set differs from the admitted closure",
                ));
            }
        }
    }
    Ok(())
}

fn selected_local_artifact_path(
    member: &PortableArtifactMember,
    model: &PortableModelBinding,
    entry: &serde_json::Value,
    payload: &JsonObject,
    settings: &Settings,
    job_type: &JobType,
) -> WorkerResult<PathBuf> {
    let model_slot = model.slot.as_str();
    if member.component_id.as_deref() == Some("selected_text_encoder") {
        if model_slot != "modelManifestEntry" {
            return Err(invalid_binding(
                "selected text encoder has the wrong model slot",
            ));
        }
        let resolution = entry
            .get("resolvedTextEncoder")
            .ok_or_else(|| invalid_binding("selected text encoder has no private resolution"))?;
        let selected_id = payload
            .get("advanced")
            .and_then(|advanced| advanced.get("textEncoderModel"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid_binding("selected text encoder has no authored id"))?;
        if resolution
            .get("selectionId")
            .and_then(serde_json::Value::as_str)
            != Some(selected_id)
        {
            return Err(invalid_binding(
                "selected text encoder id differs from private resolution",
            ));
        }
        let is_file = match resolution
            .get("sourceKind")
            .and_then(serde_json::Value::as_str)
        {
            Some("file") => true,
            Some("directory") => false,
            _ => return Err(invalid_binding("selected text encoder has no source kind")),
        };
        let model_id = entry
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid_binding("selected text encoder has no model identity"))?;
        let claim_model_id =
            sceneworks_core::model_artifacts::portable_binding::selected_text_encoder_claim_model_id(
                model_id, entry,
            )
            .ok_or_else(|| invalid_binding("selected text encoder has no registered model scope"))?;
        return crate::portable_source_claims::registered_text_encoder_path(
            settings,
            &claim_model_id,
            is_file,
            &member.files,
        )?
        .map(|(root, _)| root)
        .ok_or_else(|| invalid_binding("registered selected text encoder bytes are unavailable"));
    }
    if let PortableSourceIdentity::WorkerClaim { source_id, .. } = &member.source {
        let (claim, selected) = crate::portable_source_claims::capture_selected_source(
            source_id, settings, payload,
        )?
        .ok_or_else(|| invalid_binding("worker-advertised auxiliary source is unavailable"))?;
        claim
            .matches_worker_claim_member(member)
            .map_err(|error| invalid_binding(error.to_string()))?;
        return Ok(selected);
    }
    let utility_selected = if member.role == ArtifactMemberRole::Primary
        && model_slot == "modelManifestEntries/0"
        && matches!(
            job_type,
            JobType::DatasetAnalysis | JobType::TrainingCaption
        ) {
        if matches!(member.source, PortableSourceIdentity::Digest { .. }) {
            let selection = payload
                .get("modelNameOrPath")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid_binding("utility model selection is missing"))?;
            crate::portable_source_claims::selected_app_managed_exact_source(
                settings,
                selection,
                &member.files,
            )?
        } else {
            payload
                .get("modelNameOrPath")
                .and_then(serde_json::Value::as_str)
                .and_then(|selection| {
                    crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, selection)
                        .or_else(|| {
                            let path = PathBuf::from(selection);
                            path.is_dir().then_some(path)
                        })
                })
        }
    } else {
        None
    };
    let local_catalog = if utility_selected.is_some() {
        None
    } else if member.role == ArtifactMemberRole::Primary {
        let id = sceneworks_core::model_artifacts::portable_binding::registered_primary_digest_catalog_id(
            model,
            &serde_json::Value::Object(payload.clone()),
        )
        .map_err(|error| invalid_binding(error.to_string()))?;
        worker_catalog_member_path(settings, "user.models.jsonc", "models", &id, None, member)?
    } else if member.role == ArtifactMemberRole::Lora {
        let claim = serde_json::Value::Object(payload.clone());
        member
            .component_id
            .as_deref()
            .and_then(|slot| {
                enumerate_portable_array_slots(&claim, &["loras", "lokrs"])
                    .into_iter()
                    .find(|(candidate, _)| candidate == slot)
                    .map(|(_, record)| record)
            })
            .and_then(|record| record.get("id"))
            .and_then(serde_json::Value::as_str)
            .and_then(|id| {
                worker_lora_catalog_member_path(settings, payload, id, member).transpose()
            })
            .transpose()?
    } else if member.role == ArtifactMemberRole::Control {
        if member.component_id.as_deref() == Some("strict_control") {
            let overlay_id = payload
                .get("advanced")
                .and_then(|advanced| advanced.get("controlWeights"))
                .and_then(|weights| weights.get("overlayId"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid_binding("registered control overlay has no ID"))?;
            return crate::portable_source_claims::registered_control_overlay_path(
                settings,
                payload,
                overlay_id,
                &member.files,
            )?
            .ok_or_else(|| invalid_binding("registered control overlay bytes are unavailable"));
        }
        let claim = serde_json::Value::Object(payload.clone());
        member
            .component_id
            .as_deref()
            .and_then(|slot| {
                enumerate_portable_array_slots(&claim, &["controls"])
                    .into_iter()
                    .find(|(candidate, _)| candidate == slot)
                    .map(|(_, record)| record)
            })
            .and_then(|record| record.get("id"))
            .and_then(serde_json::Value::as_str)
            .and_then(|id| {
                worker_catalog_member_path(
                    settings,
                    "user.models.jsonc",
                    "models",
                    id,
                    None,
                    member,
                )
                .transpose()
            })
            .transpose()?
    } else {
        let model_id = entry.get("id").and_then(serde_json::Value::as_str);
        let locator_key = member
            .component_id
            .as_deref()
            .map(|id| portable_component_locator_key(model_slot, id));
        if let (Some(model_id), Some(locator_key)) = (model_id, locator_key) {
            worker_catalog_member_path(
                settings,
                "user.models.jsonc",
                "models",
                model_id,
                Some(locator_key),
                member,
            )?
        } else {
            None
        }
    };
    let raw = if let Some(selected) = utility_selected {
        Some(selected)
    } else if let Some(local) = local_catalog {
        Some(local)
    } else if member.role == ArtifactMemberRole::Primary {
        (model_slot == "modelManifestEntry")
            .then(|| {
                payload
                    .get("advanced")
                    .and_then(|advanced| advanced.get("modelPath"))
            })
            .flatten()
            .or_else(|| entry.get("modelPath"))
            .or_else(|| entry.get("paths").and_then(|paths| paths.get("model")))
            .or_else(|| entry.get("installedPath"))
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from)
    } else if matches!(
        member.role,
        ArtifactMemberRole::Lora | ArtifactMemberRole::Control
    ) {
        let id = member
            .component_id
            .as_deref()
            .ok_or_else(|| invalid_binding("an adapter has no admitted component id"))?;
        let claim = serde_json::Value::Object(payload.clone());
        enumerate_portable_array_slots(&claim, &["loras", "lokrs", "controls"])
            .into_iter()
            .find(|(slot, _)| slot == id)
            .and_then(|(_, selected)| {
                if member.role == ArtifactMemberRole::Lora {
                    return sceneworks_core::lora_family::adapter_record_path(selected);
                }
                selected
                    .get("modelPath")
                    .and_then(serde_json::Value::as_str)
                    .map(PathBuf::from)
                    .or_else(|| {
                        selected
                            .get("installedPath")
                            .or_else(|| selected.get("path"))
                            .or_else(|| selected.get("sourcePath"))
                            .and_then(serde_json::Value::as_str)
                            .map(PathBuf::from)
                    })
            })
    } else {
        let id = member
            .component_id
            .as_deref()
            .ok_or_else(|| invalid_binding("a local component has no admitted component id"))?;
        entry_locator_path(entry, portable_component_locator_key(model_slot, id))
    }
    .ok_or_else(|| invalid_binding("the admitted local artifact has no worker locator"))?;
    crate::paths::normalize_app_managed_model_entry_path(settings, &raw, "portable model binding")
}

struct SelectedCheckpointPlanLayer {
    checkpoint_id: String,
    layer_id: String,
    target_path: PathBuf,
    source_path: PathBuf,
}

fn checkpoint_plan_layer_id<'a>(
    member: &'a PortableArtifactMember,
    model_slot: &str,
) -> Option<&'a str> {
    member
        .component_id
        .as_deref()?
        .strip_prefix(&format!("{model_slot}/importPlan/layers/"))
        .filter(|id| !id.is_empty())
}

fn selected_checkpoint_plan_layer(
    member: &PortableArtifactMember,
    model: &PortableModelBinding,
    entry: &serde_json::Value,
    settings: &Settings,
) -> WorkerResult<Option<SelectedCheckpointPlanLayer>> {
    let Some(plan) = &model.checkpoint_plan else {
        return Ok(None);
    };
    let layer_id = if member.role == ArtifactMemberRole::Primary {
        plan.primary_layer_id.as_deref()
    } else {
        checkpoint_plan_layer_id(member, &model.slot)
    };
    let Some(layer_id) = layer_id else {
        return Ok(None);
    };
    let layer = plan
        .layers
        .iter()
        .find(|layer| layer.layer_id == layer_id)
        .ok_or_else(|| invalid_binding("frozen checkpoint layer is absent from semantic plan"))?;
    let checkpoint_id = entry
        .pointer("/importPlan/checkpointId")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_binding("checkpoint plan layer has no job checkpoint ID"))?;
    let [expected] = member.files.as_slice() else {
        return Err(invalid_binding(
            "checkpoint plan layer must freeze one file",
        ));
    };
    let model_id = entry
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_binding("checkpoint plan layer has no model ID"))?;
    // Registered content claims are model-scoped, so an independently imported worker copy may
    // have another root ID, plan ID or filename. Match the frozen bytes on this host.
    let source_path =
        crate::portable_source_claims::registered_model_content_path(settings, model_id, expected)?
            .ok_or_else(|| invalid_binding("worker checkpoint plan layer bytes are unavailable"))?;
    let target = layer.target_path.as_str();
    sceneworks_core::checkpoint_plan_store::portable_relative_path_parts(target)
        .map_err(|_| invalid_binding("checkpoint plan layer target is invalid"))?;
    Ok(Some(SelectedCheckpointPlanLayer {
        checkpoint_id: checkpoint_id.to_owned(),
        layer_id: layer_id.to_owned(),
        target_path: PathBuf::from(target),
        source_path,
    }))
}

fn portable_component_locator_key<'a>(model_slot: &str, component_id: &'a str) -> &'a str {
    component_id
        .strip_prefix(model_slot)
        .and_then(|value| value.strip_prefix('/'))
        .unwrap_or(component_id)
}

fn entry_locator_path(entry: &serde_json::Value, key: &str) -> Option<PathBuf> {
    let selected = match key {
        "modelPath" | "installedPath" => entry.get(key),
        _ if key.starts_with("paths/") => entry.get("paths")?.get(key.strip_prefix("paths/")?),
        _ if key.starts_with("components/") => {
            let index = key.strip_prefix("components/")?.parse::<usize>().ok()?;
            entry.get("components")?.as_array()?.get(index)?.get("path")
        }
        // Pre-portable jobs may carry a raw component id. Keep their exact existing locator.
        _ => entry
            .get("components")
            .and_then(serde_json::Value::as_array)
            .and_then(|components| {
                components.iter().find(|component| {
                    component.get("id").and_then(serde_json::Value::as_str) == Some(key)
                })
            })
            .and_then(|component| component.get("path"))
            .or_else(|| entry.get("paths").and_then(|paths| paths.get(key))),
    };
    selected
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
}

fn set_existing_entry_locator(
    entry: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &str,
) -> bool {
    let selected = match key {
        "modelPath" | "installedPath" => entry.get_mut(key),
        _ if key.starts_with("paths/") => entry
            .get_mut("paths")
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|paths| paths.get_mut(key.strip_prefix("paths/").unwrap_or_default())),
        _ if key.starts_with("components/") => key
            .strip_prefix("components/")
            .and_then(|index| index.parse::<usize>().ok())
            .and_then(|index| entry.get_mut("components")?.as_array_mut()?.get_mut(index))
            .and_then(|component| component.get_mut("path")),
        _ => {
            let component = entry
                .get_mut("components")
                .and_then(serde_json::Value::as_array_mut)
                .and_then(|components| {
                    components.iter_mut().find(|component| {
                        component.get("id").and_then(serde_json::Value::as_str) == Some(key)
                    })
                })
                .and_then(|component| component.get_mut("path"));
            if component.is_some() {
                component
            } else {
                entry
                    .get_mut("paths")
                    .and_then(serde_json::Value::as_object_mut)
                    .and_then(|paths| paths.get_mut(key))
            }
        }
    };
    if let Some(value) = selected.filter(|value| value.is_string()) {
        *value = serde_json::Value::String(path.to_owned());
        true
    } else {
        false
    }
}

fn worker_catalog_member_path(
    settings: &Settings,
    manifest_name: &str,
    collection: &str,
    id: &str,
    component_id: Option<&str>,
    member: &PortableArtifactMember,
) -> WorkerResult<Option<PathBuf>> {
    if collection == "models"
        && id.starts_with(sceneworks_core::external_base_models::EXTERNAL_ID_PREFIX)
    {
        return crate::portable_source_claims::registered_external_base_path(
            settings,
            id,
            component_id,
            &member.files,
        );
    }
    if manifest_name == "user.models.jsonc"
        && collection == "models"
        && component_id.is_none()
        && member.role == ArtifactMemberRole::Primary
        && matches!(&member.source, PortableSourceIdentity::Digest { .. })
        && crate::portable_source_claims::is_converted_builtin_model_id(id)
    {
        return crate::portable_source_claims::registered_converted_builtin_model_path(
            settings,
            id,
            &member.files,
        )
        .map(Some)
        .ok_or_else(|| invalid_binding("converted builtin receipt or bytes are unavailable"));
    }
    if settings.config_dir.as_os_str().is_empty() {
        return Ok(None);
    }
    let manifest = settings.config_dir.join("manifests").join(manifest_name);
    worker_catalog_member_path_at(settings, &manifest, collection, id, component_id, member)
}

fn worker_lora_catalog_member_path(
    settings: &Settings,
    payload: &JsonObject,
    id: &str,
    member: &PortableArtifactMember,
) -> WorkerResult<Option<PathBuf>> {
    if id.starts_with("character_lora_") {
        let Some(project_id) = payload.get("projectId").and_then(serde_json::Value::as_str) else {
            return Err(invalid_binding(
                "registered character LoRA has no project ID",
            ));
        };
        let Some(character_id) = payload
            .get("characterId")
            .and_then(serde_json::Value::as_str)
        else {
            return Err(invalid_binding(
                "registered character LoRA has no character ID",
            ));
        };
        let Some(project) = crate::paths::project_path_for_payload(settings, payload)
            .map_err(|_| invalid_binding("worker project source is unavailable"))?
        else {
            return Err(invalid_binding("registered character LoRA has no project"));
        };
        return crate::portable_source_claims::registered_character_lora_path(
            settings,
            &project,
            project_id,
            character_id,
            id,
            &member.files,
        )
        .and_then(|path| {
            path.map(Some)
                .ok_or_else(|| invalid_binding("registered character LoRA bytes are unavailable"))
        });
    }
    if id.starts_with(sceneworks_core::external_loras::EXTERNAL_ID_PREFIX) {
        return crate::portable_source_claims::registered_external_lora_path(
            settings,
            id,
            &member.files,
        );
    }
    if let Some(global) =
        worker_catalog_member_path(settings, "user.loras.jsonc", "loras", id, None, member)?
    {
        return Ok(Some(global));
    }
    let project = crate::paths::project_path_for_payload(settings, payload)
        .map_err(|_| invalid_binding("worker project source is unavailable"))?;
    if let Some(project) = project {
        let manifest = project.join("loras/manifest.jsonc");
        if let Some(path) =
            worker_catalog_member_path_at(settings, &manifest, "loras", id, None, member)?
        {
            return Ok(Some(path));
        }
    }
    Ok(crate::portable_source_claims::registered_builtin_lora_path(
        settings,
        id,
        &member.files,
    ))
}

fn worker_catalog_member_path_at(
    settings: &Settings,
    manifest: &Path,
    collection: &str,
    id: &str,
    component_id: Option<&str>,
    member: &PortableArtifactMember,
) -> WorkerResult<Option<PathBuf>> {
    let raw = match std::fs::read_to_string(manifest) {
        Ok(raw) => raw,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(invalid_binding("worker catalog is unavailable")),
    };
    let parsed: serde_json::Value = if collection == "models" {
        // Shared normalizer: a future-version model manifest is refused, never read as current.
        sceneworks_core::model_manifest_version::parse_user_model_manifest(&raw)
            .map_err(|_| invalid_binding("worker model catalog version is unsupported"))?
    } else {
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&raw))
            .map_err(|_| invalid_binding("worker catalog is invalid"))?
    };
    let matches = parsed
        .get(collection)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid_binding("worker catalog collection is invalid"))?
        .iter()
        .filter(|entry| entry.get("id").and_then(serde_json::Value::as_str) == Some(id))
        .collect::<Vec<_>>();
    for entry in matches {
        let candidates = if collection == "loras" {
            sceneworks_core::lora_family::adapter_record_path(entry)
                .map(|path| vec![(String::new(), path)])
                .unwrap_or_default()
        } else {
            crate::portable_source_claims::registered_model_locators(entry)
        };
        for (key, selected) in candidates {
            if component_id.is_some_and(|required| {
                key != required
                    && !(collection == "models"
                        && !required.contains('/')
                        && (key.strip_prefix("paths/") == Some(required)
                            || key
                                .strip_prefix("components/")
                                .and_then(|index| index.parse::<usize>().ok())
                                .and_then(|index| entry.get("components")?.as_array()?.get(index))
                                .and_then(|component| component.get("id"))
                                .and_then(serde_json::Value::as_str)
                                == Some(required)))
            }) {
                continue;
            }
            let normalized = if collection == "loras" {
                crate::portable_source_claims::normalize_catalog_lora_path(
                    settings, manifest, &selected,
                )
            } else {
                crate::paths::normalize_app_managed_model_entry_path(
                    settings,
                    &selected,
                    "portable catalog source",
                )
            };
            let Ok(normalized) = normalized else {
                continue;
            };
            let roots = crate::paths::allowed_model_roots(settings).unwrap_or_default();
            if crate::portable_source_claims::capture_registered_files_with_roots(
                &normalized,
                &roots,
            )
            .is_ok_and(|files| files == member.files)
            {
                return Ok(Some(normalized));
            }
        }
    }
    if collection == "models"
        && parsed
            .get(collection)
            .and_then(serde_json::Value::as_array)
            .is_some_and(|entries| {
                entries
                    .iter()
                    .any(|entry| entry.get("id").and_then(serde_json::Value::as_str) == Some(id))
            })
    {
        return Err(invalid_binding(
            "registered source bytes or location differ from the frozen binding",
        ));
    }
    Ok(None)
}

fn invalid_binding(message: impl Into<String>) -> WorkerError {
    WorkerError::InvalidPayload(format!("portable model binding: {}", message.into()))
}

/// One model entry's pass-1 outcome: the resolution, plus the SELECTED closure that produced it.
///
/// The closure is kept here and never re-read off the resolution, because
/// `ModelResolution::local_ready` deliberately carries `requirements: Vec::new()` — a local-ready
/// resolution names an artifact, not a source-library closure. Coverage and the pass-3 fallback
/// both need the real requirement list, so losing it to the resolution would make every
/// locally-resolved model look like it required nothing at all.
struct EntryPlan {
    requirements: Vec<ExternalArtifactRequirement>,
    receipt_backed: bool,
    resolution: ModelResolution,
    /// True for the synthetic resident-base plan (sc-22329) rather than a model the job CARRIES.
    /// It may only ever add a local-tier serve: it never opens an external session and never
    /// refuses admission. See the split above `let mut sessions`.
    resident_base: bool,
}

/// WHICH repositories of a resident base model the family's route actually opens — and therefore
/// the only ones whose coverage may decide whether the base is served locally.
///
/// This is not a preference. `select_local_tier` is all-or-nothing across a model's pairs, so a
/// closure naming a repository the route never reads cannot merely cost a cache miss: it BLOCKS
/// the pairs the route does read whenever the extra repository is uncached, and enqueues its
/// bytes for promotion on top.
/// The `(tier, shape, builtin catalog entry)` this build resolves a resident base for `family`
/// through, or `None` when the family has no resident base.
///
/// The indirection exists for the tests: the production rows name REAL repositories, and a test
/// that publishes a bundle for one of those would be visible to any concurrently running test
/// resolving the same repository — `discover_snapshot`'s `unique_local_snapshot` fallback is
/// repo-keyed and IGNORES the revision, so a unique fixture revision does not isolate it (see the
/// fixture-identity note at the top of this file's test module).
fn resident_base_row(family: &str) -> Option<(String, ResidentBaseShape, serde_json::Value)> {
    #[cfg(test)]
    if let Some(row) = tests::resident_base_override(family) {
        return Some(row);
    }
    let row = PLAN_BACKED_RESIDENT_BASE_MODELS
        .iter()
        .find(|row| row.family == family)?;
    let entry = crate::training_jobs::builtin_model_manifest_entry(row.base_model_id)?;
    Some((row.tier.to_owned(), row.shape, entry))
}

fn selected_resident_base_closure_for_entry(
    entry: &serde_json::Value,
    platform: &str,
) -> Option<serde_json::Value> {
    let family = entry.get("family")?.as_str()?.trim();
    let (tier, shape, base_entry) = resident_base_row(family)?;
    selected_resident_base_closure_for_row(&base_entry, &tier, shape, platform)
}

/// The synthetic [`EntryPlan`] for a provably-non-HF-local entry's resident base model, or `None`
/// when the entry has none (no declared family, a family without a resident base, or a base whose
/// installed closure cannot be recovered) — in which case the guard behaves exactly as it did
/// before sc-22329 and the family route's own typed refusal keeps owning the not-installed case.
///
/// The requirements are built by the same [`selected_requirements_for_closure`] machinery every
/// HF-carried entry goes through — so receipt matching, coverage, leasing, disconnect typing, tier
/// reporting and promotion intake all behave identically to a job that had carried the base model
/// explicitly. What differs per family is only WHICH of the base entry's downloads that machinery
/// is pointed at, which is [`ResidentBaseShape`]'s whole job: a closure naming a repository the
/// route never opens would block the ones it does (pass 3 is all-or-nothing across a model's
/// pairs) and enqueue bytes nothing reads for promotion.
fn plan_backed_resident_base_plan(
    entry: &serde_json::Value,
    settings: &Settings,
    configured_library: &Path,
    local_artifacts: &[sceneworks_core::model_artifacts::ResolvedModelArtifact],
) -> Option<EntryPlan> {
    use sceneworks_core::model_artifacts::artifact_selection::selected_requirements_for_closure;

    let family = entry
        .get("family")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())?;
    let (tier, shape, base_entry) = resident_base_row(family)?;
    // This worker's OWN platform, and the family's fixed base tier — deliberately not the
    // request's runtime variant (see [`PlanBackedResidentBase::tier`]).
    let closure =
        selected_resident_base_closure_for_row(&base_entry, &tier, shape, std::env::consts::OS)?;
    let selected = selected_requirements_for_closure(&closure, &settings.data_dir);
    if selected.requirements.is_empty() {
        return None;
    }
    let resolution = resolve_model_availability(
        &settings.data_dir,
        configured_library,
        &selected.requirements,
        selected.receipt_backed,
        local_artifacts,
    );
    Some(EntryPlan {
        requirements: selected.requirements,
        receipt_backed: selected.receipt_backed,
        resolution,
        resident_base: true,
    })
}

/// Every VALID app-owned resolved artifact on this host, plus the entries rejected for a shape the
/// local tier cannot serve. Empty when the resolved cache is switched off. Read-only: enumeration
/// never creates a cache session and never stamps usage, so judging availability cannot look like
/// using a model.
fn local_resolved_artifacts(
    settings: &Settings,
) -> sceneworks_core::model_artifacts::resolved_cache::LocalArtifactScan {
    if !settings.resolved_cache_policy().enabled {
        return Default::default();
    }
    ResolvedCacheStore::valid_local_artifacts(&settings.data_dir)
}

/// Decide the job's local tier, take the leases behind it, and install the one preference scope.
///
/// # The rule
///
/// > A `(repository, revision)` pair is served locally **iff** one leased bundle holds a SUPERSET
/// > of the union of file requirements every model entry in this job needs from that pair.
/// > Otherwise the pair is served to NOBODY and every model needing it re-resolves against the
/// > authoritative source tier.
///
/// # Why it has to be here
///
/// The preference seam is directory-level: it answers `(repository, revision)` with a snapshot
/// path, and once that path is out it cannot re-ask "does this bundle hold what THIS caller
/// needs". So the question must be settled before any path is handed out, by the only layer that
/// can see every model the job will load — this one. A per-model decision is precisely the defect:
/// a job carrying `owner/matrix` q4 (bundled) and q8 (not bundled) would admit q4, install the
/// overlay for `owner/matrix@REV`, and then resolve the q8 load into the q4 bundle root, which has
/// no `q8/` subtree at all.
///
/// # The deliberate consequence
///
/// In that same q4+q8 job the q4 model ALSO falls back to the source tier, because the union for
/// the shared pair includes q8's files and no bundle covers it. That is the correct conservative
/// answer, not a gap to engineer around: partial service of a pair is exactly what cannot be made
/// safe at a directory-level seam.
///
/// A requirement carrying `revision: None` (a legacy receipt with no recorded snapshot) makes the
/// WHOLE repository unserveable — there is no pair to compare coverage against, so no bundle can
/// be shown to hold what that requirement needs.
fn select_local_tier(
    settings: &Settings,
    configured_library: &std::path::Path,
    plans: &mut [EntryPlan],
) -> (Option<ActiveLocalArtifacts>, Vec<ResolvedCacheLease>) {
    use std::collections::{BTreeMap, BTreeSet};

    let re_resolve_to_source = |plans: &mut [EntryPlan]| {
        for plan in plans.iter_mut() {
            if plan.resolution.availability == ModelAvailability::LocalReady {
                plan.resolution = resolve_model_availability(
                    &settings.data_dir,
                    configured_library,
                    &plan.requirements,
                    plan.receipt_backed,
                    &[],
                );
            }
        }
    };

    if !plans
        .iter()
        .any(|plan| plan.resolution.availability == ModelAvailability::LocalReady)
    {
        return (None, Vec::new());
    }

    // PASS 2a — the union of file requirements per pair, across ALL resolutions, local AND
    // non-local. The non-local sibling is what poisons a pair: it is a model that will read this
    // very snapshot from the source tier, so a bundle that does not hold its files cannot be
    // allowed to answer for the pair at all.
    let mut required: BTreeMap<(String, String), BTreeSet<PathBuf>> = BTreeMap::new();
    let mut unserveable_repositories: BTreeSet<String> = BTreeSet::new();
    for plan in plans.iter() {
        for requirement in &plan.requirements {
            let Some(revision) = requirement.revision.as_ref() else {
                unserveable_repositories.insert(requirement.repository.clone());
                continue;
            };
            required
                .entry((requirement.repository.clone(), revision.clone()))
                .or_default()
                .extend(requirement.files.iter().cloned());
        }
    }

    // PASS 2b — candidate entries, read from the SCANNED artifacts. NOTHING is leased yet, and
    // that is deliberate: `acquire_complete` stamps `last_used_at` and takes the entry's shared
    // artifact lock through the BLOCKING `FileExt::lock_shared`. Leasing a candidate that coverage
    // then rejects would both mark an entry this job never reads as recently used — skewing
    // retention's LRU against the entries that really served — and stall the whole guard behind an
    // evictor holding a lock on a bundle no load here will ever touch.
    let mut candidates: Vec<(String, LocalArtifactOverlayEntry)> = Vec::new();
    // cache key -> repository, the only thing the lease needs from the artifact.
    let mut candidate_repositories: BTreeMap<String, String> = BTreeMap::new();
    for plan in plans.iter() {
        if plan.resolution.availability != ModelAvailability::LocalReady {
            continue;
        }
        let Some(artifact) = plan.resolution.local_artifact.as_ref() else {
            continue;
        };
        let Ok(cache_key) = artifact.cache_key() else {
            continue;
        };
        if candidate_repositories.contains_key(&cache_key) {
            continue;
        }
        match sceneworks_core::model_artifacts::local_preference::overlay_entries_for_artifact(
            artifact,
        ) {
            Ok(entries) => {
                candidate_repositories
                    .insert(cache_key.clone(), artifact.identity.repository.clone());
                candidates.extend(entries.into_iter().map(|entry| (cache_key.clone(), entry)));
            }
            Err(error) => {
                // The scan already reports this class, so reaching it here means the artifact the
                // resolver chose differs from what the scan offered. Named the same way either way.
                emit_event_value(
                    Level::WARN,
                    json!({
                        "event": "resolved_cache_local_tier_unsupported",
                        "cacheKey": cache_key,
                        "repository": artifact.identity.repository,
                        "revision": artifact.identity.revision,
                        "reason": error.to_string(),
                    }),
                );
            }
        }
    }

    // PASS 2c — tentatively select at most one serving entry per pair, and only on full coverage.
    let mut tentative: Vec<(String, LocalArtifactOverlayEntry)> = Vec::new();
    for ((repository, revision), files) in &required {
        if unserveable_repositories.contains(repository) {
            continue;
        }
        let files: Vec<PathBuf> = files.iter().cloned().collect();
        let Some((cache_key, entry)) = candidates.iter().find(|(_, entry)| {
            entry.repository == *repository
                && entry.revision == *revision
                && sceneworks_core::model_artifacts::local_preference::entry_covers(entry, &files)
        }) else {
            continue;
        };
        tentative.push((cache_key.clone(), entry.clone()));
    }

    // PASS 2d — lease EXACTLY the winners, and re-prove every selection against the LEASED copy.
    // The lease re-reads and re-validates the published entry under its shared artifact lock, so a
    // selection that no longer matches that copy was made against a stale scan; its pair is dropped
    // (the models needing it fall back in pass 3) rather than served on stale evidence.
    let mut leases: Vec<ResolvedCacheLease> = Vec::new();
    let mut selected_entries: Vec<LocalArtifactOverlayEntry> = Vec::new();
    let mut served_pairs: BTreeSet<(String, String)> = BTreeSet::new();
    let winners: BTreeSet<String> = tentative.iter().map(|(key, _)| key.clone()).collect();
    for cache_key in &winners {
        let Some(repository) = candidate_repositories.get(cache_key) else {
            continue;
        };
        let Some(lease) = lease_local_artifact(settings, cache_key, repository) else {
            continue;
        };
        let Ok(leased) =
            sceneworks_core::model_artifacts::local_preference::overlay_entries_for_artifact(
                lease.artifact(),
            )
        else {
            continue;
        };
        let mut serves_a_pair = false;
        for (_, entry) in tentative.iter().filter(|(key, _)| key == cache_key) {
            // IDENTICAL, not merely present: the coverage decision was made about this exact
            // snapshot root and file set, and anything else is a different bundle.
            if !leased.contains(entry) {
                continue;
            }
            served_pairs.insert((entry.repository.clone(), entry.revision.clone()));
            selected_entries.push(entry.clone());
            serves_a_pair = true;
        }
        if serves_a_pair {
            leases.push(lease);
        }
    }

    // PASS 3 — a model stays LocalReady only if EVERY pair in its closure was selected. Any other
    // model re-resolves against the source tier, keeping every typed disconnect/incomplete
    // guarantee instead of proceeding on a promise the runtime cannot keep.
    for plan in plans.iter_mut() {
        if plan.resolution.availability != ModelAvailability::LocalReady {
            continue;
        }
        let fully_served = plan.requirements.iter().all(|requirement| {
            requirement.revision.as_ref().is_some_and(|revision| {
                served_pairs.contains(&(requirement.repository.clone(), revision.clone()))
            })
        });
        if fully_served {
            continue;
        }
        emit_event_value(
            Level::INFO,
            json!({
                "event": "resolved_cache_local_tier_not_selected",
                "repository": plan
                    .requirements
                    .iter()
                    .find(|requirement| requirement.is_primary)
                    .map(|requirement| requirement.repository.clone())
                    .unwrap_or_default(),
                "reason": "no leased bundle covers every file this job needs from one of the \
                           model's (repository, revision) pairs",
            }),
        );
        plan.resolution = resolve_model_availability(
            &settings.data_dir,
            configured_library,
            &plan.requirements,
            plan.receipt_backed,
            &[],
        );
    }

    if selected_entries.is_empty() {
        return (None, Vec::new());
    }
    // ONE scope holding EXACTLY the selected entries. A pair in it is serveable to every consumer
    // in the process by construction, including a model that fell back above for a DIFFERENT pair
    // — the union that admitted this pair already covered that model's needs from it.
    match sceneworks_core::model_artifacts::local_preference::prefer_local_snapshots(
        selected_entries,
    ) {
        Ok(scope) => (Some(scope), leases),
        Err(error) => {
            // FAIL CLOSED, and actually close: nothing is installed, every lease is dropped, and
            // every locally resolved model goes back to the source tier. Installing a partial
            // scope here is the exact defect this function exists to prevent.
            emit_event_value(
                Level::WARN,
                json!({
                    "event": "resolved_cache_local_tier_not_selected",
                    "reason": error.to_string(),
                }),
            );
            re_resolve_to_source(plans);
            (None, Vec::new())
        }
    }
}

/// Take the runtime lease on one locally resolved artifact, or `None` when it cannot be leased
/// after all (concurrently evicted, no longer complete). The lease is what keeps the bytes behind a
/// served path from being evicted mid-load, so it is always acquired BEFORE the scope that serves
/// those bytes is installed.
fn lease_local_artifact(
    settings: &Settings,
    cache_key: &str,
    repository: &str,
) -> Option<ResolvedCacheLease> {
    let store = cache_store(settings)?;
    let resolver = ModelArtifactResolver::new(sceneworks_core::hf_home::model_source_library(
        &settings.data_dir,
    ));
    // The store takes the entry's SHARED artifact lock before it re-reads and re-validates the
    // published entry under it, then stamps usage. An evictor that reaches the entry first holds
    // the exclusive lock, so this waits; an evictor that arrives afterwards finds the lock held
    // and must leave the entry alone.
    store
        .acquire_complete(cache_key, &resolver, repository)
        .map_err(|error| {
            // sc-21534: the load boundary is now the ONE surface that detects a content-altered
            // bundle (every listing and lookup judges on paths and sizes), and its refusal names
            // the mismatched file. Falling back to the source tier is the right recovery, but
            // swallowing the verdict would leave no trace of WHY the local bytes were refused.
            tracing::warn!(
                cache_key,
                repository,
                error = %error,
                "resolved-cache entry refused at the load boundary; serving from the source tier"
            );
        })
        .ok()
        .flatten()
}

/// One long-lived cache session per data directory. Opening a session takes a durable session lock
/// and writes session records, so a fresh session per job would churn the store; the worker claims
/// one job at a time and keeps this handle for the process lifetime instead.
fn cache_store(settings: &Settings) -> Option<ResolvedCacheStore> {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    static STORES: OnceLock<Mutex<BTreeMap<std::path::PathBuf, ResolvedCacheStore>>> =
        OnceLock::new();
    let stores = STORES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut stores = stores
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(store) = stores.get(&settings.data_dir) {
        return Some(store.clone());
    }
    let store = ResolvedCacheStore::open(&settings.data_dir).ok()?;
    stores.insert(settings.data_dir.clone(), store.clone());
    Some(store)
}

/// Report which tier will serve each admitted model. This is the runtime's answer to "where did
/// this load's weights come from", emitted once per job before any loader runs.
fn emit_source_tiers(resolutions: &[ModelResolution]) {
    for resolution in resolutions {
        let tier = match resolution.availability {
            ModelAvailability::LocalReady => "resolved_local",
            ModelAvailability::ExternalReady => "source_library",
            _ => continue,
        };
        let (repository, revision) = match &resolution.local_artifact {
            Some(artifact) => (
                artifact.identity.repository.clone(),
                artifact.identity.revision.clone(),
            ),
            None => resolution
                .requirements
                .iter()
                .find(|requirement| requirement.is_primary)
                .map(|requirement| {
                    (
                        requirement.repository.clone(),
                        requirement.revision.clone().unwrap_or_default(),
                    )
                })
                .unwrap_or_default(),
        };
        emit_event_value(
            Level::INFO,
            json!({
                "event": "model_source_tier_selected",
                "tier": tier,
                "repository": repository,
                "revision": revision,
            }),
        );
    }
}

/// Allowlist AT THE SEAM: only routes that cannot open model/HF bytes may run without model
/// carriers. `JobType` is `#[non_exhaustive]`, so this match cannot force a compile error for a
/// new variant; instead the wildcard arm classifies anything unlisted as model-backed — an
/// unclassified route therefore defaults to FAIL CLOSED (requires carriers) and can only become
/// model-free by an explicit entry in the allowlist above.
fn job_requires_typed_model_source(job_type: &JobType) -> bool {
    match job_type {
        JobType::Placeholder
        | JobType::FrameExtract
        | JobType::TimelineExport
        | JobType::DatasetParquetImport
        | JobType::ModelImport
        | JobType::LoraImport => false,
        JobType::ImageGenerate
        | JobType::ImageEdit
        | JobType::ImageVqa
        | JobType::ImageInterleave
        | JobType::VideoGenerate
        | JobType::VideoExtend
        | JobType::VideoBridge
        | JobType::PersonDetect
        | JobType::PersonTrack
        | JobType::PersonReplace
        | JobType::AudioGenerate
        | JobType::PoseDetect
        | JobType::KpsExtract
        | JobType::ImageUpscale
        | JobType::ImageDetail
        | JobType::ImageSegment
        | JobType::ImageDepth
        | JobType::VideoUpscale
        | JobType::ModelDownload
        | JobType::ModelConvert
        | JobType::LoraDownload
        | JobType::LoraTrain
        | JobType::ControlTraining
        | JobType::TrainingCaption
        | JobType::DatasetAnalysis
        | JobType::CatalogAnalysis
        | JobType::DatasetUpscale
        | JobType::DatasetFaceAnalysis
        | JobType::DatasetSubjectMask
        | JobType::FaceLikenessCompare
        | JobType::PromptRefine
        | JobType::Unknown(_) => true,
        _ => true,
    }
}

fn payload_model_slots(payload: &JsonObject) -> Vec<(String, &serde_json::Value)> {
    let mut slots = Vec::new();
    for key in ["modelManifestEntry", "baseModelManifestEntry"] {
        if let Some(entry) = payload.get(key).filter(|entry| !entry.is_null()) {
            slots.push((key.to_owned(), entry));
        }
    }
    if let Some(entries) = payload
        .get("modelManifestEntries")
        .and_then(serde_json::Value::as_array)
    {
        for (index, entry) in entries.iter().enumerate() {
            if !entry.is_null() {
                slots.push((format!("modelManifestEntries/{index}"), entry));
            }
        }
    }
    slots.sort_by(|left, right| left.0.cmp(&right.0));
    slots
}

/// Imported/app-owned and explicitly configured external-root models are the only model entries
/// that may omit the HF artifact contract. They must have no download descriptors at all and
/// every concrete path they expose must pass the worker's existing app-managed confinement
/// check. Deliberately per-entry: one valid primary carrier cannot hide an unconfined auxiliary.
///
/// "Concrete path" means the fields that really name a location — `modelPath`, `installedPath`,
/// every value of the `paths` locator map, `source.path`, and each component's `path` — and
/// nothing else. Provenance fields that merely *describe* where the model came from are not
/// filesystem inputs and are not confined (sc-20524).
///
/// The two kinds of locator are confined DIFFERENTLY, and deliberately so. A LOADER locator
/// (`modelPath` / `installedPath` / `paths.*` / `components[].path`) is a string some consumer will
/// re-resolve verbatim, and every one of those consumers anchors a relative path on the process cwd
/// (`crate::paths::normalize_absolute_path`, `model_artifacts::canonical_or_absolute`), so this
/// guard must judge exactly the same string the loader will open — otherwise the two halves of the
/// two-boundary defense (`apps/rust-api/src/jobs.rs`) stop talking about the same file. Only
/// `source.path` — provenance, which nothing ever opens — is anchored on the data dir, because that
/// is the base the model-import job wrote it against.
fn entry_is_provably_non_hf_local(
    entry: &serde_json::Value,
    settings: &Settings,
) -> WorkerResult<bool> {
    if entry
        .get("downloads")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|downloads| !downloads.is_empty())
    {
        return Ok(false);
    }
    let mut loader_paths = Vec::new();
    for key in ["modelPath", "installedPath"] {
        if let Some(path) = entry.get(key).and_then(serde_json::Value::as_str) {
            loader_paths.push(path);
        }
    }
    // `paths` is a locator map — every value in it is a place on disk (`{ "model": … }`).
    loader_paths.extend(
        entry
            .get("paths")
            .and_then(serde_json::Value::as_object)
            .into_iter()
            .flat_map(|object| object.values())
            .filter_map(serde_json::Value::as_str),
    );
    loader_paths.extend(
        entry
            .get("components")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|component| component.get("path"))
            .filter_map(serde_json::Value::as_str),
    );
    // `source`, by contrast, is a PROVENANCE record, and only its `path` names a location:
    // `provider` is a literal ("local", "huggingface", "url"), `repo` is a repository id or
    // null, `url` is a URL. sc-20524 — confining every value in this object as a filesystem
    // path resolved `"local"` against the process cwd and rejected EVERY user-imported model
    // ("non-HF model source must be inside an app-managed directory") before any loader ran.
    let source_path = entry
        .get("source")
        .and_then(|source| source.get("path"))
        .and_then(serde_json::Value::as_str);
    if loader_paths.is_empty() && source_path.is_none() {
        return Ok(false);
    }
    // A loader locator is judged exactly as its consumer will re-resolve it: raw, cwd-anchored
    // when relative. Same string, same base, same verdict on both sides of the boundary.
    for path in loader_paths {
        crate::paths::normalize_app_managed_model_path(settings, path, LOCATOR_LABEL)?;
    }
    if let Some(path) = source_path {
        confine_provenance_path(settings, path)?;
    }
    Ok(true)
}

const LOCATOR_LABEL: &str = "non-HF model source";

/// Confine the ONE provenance locator, `source.path`.
///
/// Nothing opens this field — it records where an imported model came from — so it is the one
/// locator whose base this guard may choose, and the model-import job writes it relative to the
/// data dir (`models/imports/<name>`), never to the worker's cwd. Anchoring it there is what lets
/// an imported entry be judged against the base it was actually written against.
///
/// The invariant this preserves: no locator that resolves outside every allowed root is admitted,
/// under data-dir anchoring of `source.path`. Anchoring changes which path a relative
/// `source.path` names; it never skips the check. A `..` (or a symlink) that walks back out of the
/// data dir still lands outside every root and is still rejected, which is what keeps a forged
/// "local" entry out.
///
/// Windows has two spellings that are neither absolute nor anchorable, and `PathBuf::join` silently
/// does the wrong thing with both: `Path::is_absolute` is FALSE for the root-relative
/// `\Windows\System32` (`data_dir.join(..)` keeps only the drive prefix, so nothing of the data dir
/// survives), and false for the drive-relative `D:models` (`join` replaces the data dir outright,
/// leaving a path resolved against that drive's per-process cwd). Neither has any meaning relative
/// to the data dir, so both are refused here rather than confined as a path the anchoring never
/// produced.
fn confine_provenance_path(settings: &Settings, raw: &str) -> WorkerResult<()> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "{LOCATOR_LABEL} is required."
        )));
    }
    let candidate = Path::new(trimmed);
    let anchored = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else if candidate.components().any(|component| {
        matches!(
            component,
            std::path::Component::Prefix(_) | std::path::Component::RootDir
        )
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "{LOCATOR_LABEL} must be an absolute path or relative to the app data directory: {trimmed}"
        )));
    } else {
        settings.data_dir.join(candidate)
    };
    // Path-valued so the anchored form is never round-tripped through `to_string_lossy`: a
    // non-UTF-8 component would become U+FFFD and the check would judge a different path.
    crate::paths::normalize_app_managed_model_entry_path(settings, &anchored, LOCATOR_LABEL)?;
    Ok(())
}

fn unavailable(detail: impl Into<String>) -> WorkerError {
    WorkerError::ExternalLibraryUnavailable(detail.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::model_artifacts::external_library::EXTERNAL_LIBRARY_UNAVAILABLE_CODE;
    use sceneworks_core::model_artifacts::portable_binding::{
        canonical_authored_options_sha256, portable_closure_sha256, PortableArtifactMember,
    };
    use sceneworks_core::model_artifacts::ArtifactSourceLibrary;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    /// Review finding 7: the worker's model-catalog binding reader refuses a future-version
    /// `user.models.jsonc` instead of resolving members from it as if it were current.
    #[test]
    fn worker_model_catalog_reader_refuses_a_future_manifest_version() {
        let temp = tempfile::tempdir().unwrap();
        let mut settings = Settings::for_test(temp.path().join("worker"));
        settings.config_dir = temp.path().join("worker-config");
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        let model = temp.path().join("model.safetensors");
        std::fs::write(&model, b"weights").unwrap();
        let files = vec![
            sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile {
                relative_path: "model.safetensors".into(),
                size_bytes: 7,
                sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
            },
        ];
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest {
                sha256: portable_closure_sha256(&files).unwrap(),
            },
            tier: None,
            files,
        };
        // The same catalog at the current version resolves cleanly (no matching row); the
        // future version is refused before any row is interpreted.
        for (version, refused) in [(2, false), (3, true)] {
            std::fs::write(
                &manifest,
                json!({"schemaVersion":version,"models":[{"id":"other","modelPath":model}]})
                    .to_string(),
            )
            .unwrap();
            let resolved = worker_catalog_member_path_at(
                &settings, &manifest, "models", "later", None, &member,
            );
            assert_eq!(resolved.is_err(), refused, "schemaVersion {version}");
        }
    }

    // FILE-UNIQUE fixture identities, deliberately. Some of these tests install the PROCESS-WIDE
    // local-tier preference overlay, and the overlay is keyed on `(repository, revision)` — so a
    // fixture repository or revision shared with an unrelated test elsewhere in this crate
    // (`model_jobs.rs`, `image_jobs/tests.rs`, `flux1_control_candle.rs` all used the same
    // `owner/...` names and the same `0123456789ab…` revision) lets a live overlay here answer for
    // a snapshot that test expects to read from its own temp library. The suite runs in parallel,
    // so that is a rare, load-dependent, entirely genuine false red in a file that never mentions
    // the resolved cache. The `guardfx/` prefix and the `e19706…` revisions appear nowhere else.
    const REV_A: &str = "e19706aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const REV_B: &str = "e19706bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn settings(data_dir: PathBuf) -> Settings {
        Settings::for_test(data_dir)
    }

    #[test]
    fn selected_encoder_loader_receives_only_the_worker_private_path_and_id() {
        let temp = tempfile::tempdir().unwrap();
        let api_file = temp.path().join("api/model.safetensors");
        let private_root = temp.path().join("worker/private");
        std::fs::create_dir_all(&private_root).unwrap();
        std::fs::write(private_root.join("model.safetensors"), b"weights").unwrap();
        std::fs::write(private_root.join("config.json"), b"config").unwrap();
        let api_id = crate::text_encoder_selection::opaque_id(&gen_core::WeightsSource::File(
            api_file.clone(),
        ))
        .unwrap();
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::OptionalComponent,
            component_id: Some("selected_text_encoder".to_owned()),
            source: PortableSourceIdentity::Digest {
                sha256: format!("sha256:{}", "a".repeat(64)),
            },
            tier: None,
            files: vec![
                sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile {
                    relative_path: PathBuf::from("config.json"),
                    size_bytes: 6,
                    sha256: format!("sha256:{}", "b".repeat(64)),
                },
                sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile {
                    relative_path: PathBuf::from("model.safetensors"),
                    size_bytes: 7,
                    sha256: format!("sha256:{}", "c".repeat(64)),
                },
            ],
        };
        let mut payload = json!({
            "advanced":{"textEncoderModel":api_id},
            "modelManifestEntry":{"resolvedTextEncoder":{
                "selectionId":api_id,"sourceKind":"file","path":api_file
            }}
        })
        .as_object()
        .unwrap()
        .clone();
        rewrite_loader_locator(
            &mut payload,
            "modelManifestEntry",
            &member,
            &private_root,
            false,
        )
        .unwrap();
        let worker_file = private_root.join("model.safetensors");
        let normalized_worker_file = crate::paths::normalize_absolute_path(&worker_file).unwrap();
        let worker_id = crate::text_encoder_selection::opaque_id(&gen_core::WeightsSource::File(
            normalized_worker_file.clone(),
        ))
        .unwrap();
        assert_ne!(
            api_id, worker_id,
            "different roots have different UI option ids"
        );
        assert_eq!(payload["advanced"]["textEncoderModel"], worker_id);
        assert_eq!(
            payload["modelManifestEntry"]["resolvedTextEncoder"]["selectionId"],
            worker_id
        );
        assert_eq!(
            payload["modelManifestEntry"]["resolvedTextEncoder"]["path"],
            json!(normalized_worker_file)
        );
        assert!(!Value::Object(payload)
            .to_string()
            .contains(temp.path().join("api").to_string_lossy().as_ref()));
    }

    #[test]
    fn ltx_convert_upscaler_keeps_repository_relative_filename_under_snapshot_root() {
        let snapshot = Path::new("/snapshot");
        for name in [
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "weights/upscaler.safetensors",
        ] {
            let files = vec![PortableArtifactFile {
                relative_path: name.into(),
                size_bytes: 1,
                sha256: format!("sha256:{}", "a".repeat(64)),
            }];
            let selected =
                selected_revision_aux_directory(snapshot, &files, "ltx_convert_upscaler").unwrap();
            assert_eq!(selected, snapshot);
            assert_eq!(selected.join(name), snapshot.join(name));
        }
    }

    #[test]
    fn converted_builtin_primary_rebinds_without_user_model_record() {
        let temp = TempDir::new().unwrap();
        let mut settings = settings(temp.path().join("data"));
        settings.config_dir = temp.path().join("config");
        let id = "flux2_klein_9b_true_v2";
        let root = settings.data_dir.join("models/mlx").join(id);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model_index.json"), b"{}").unwrap();
        std::fs::write(root.join("model.safetensors"), b"converted weights").unwrap();
        crate::write_produced_artifact_receipt(&root, "convert-job", "source/repo", id).unwrap();
        let files = crate::portable_source_claims::capture_registered_files(&root).unwrap();
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest {
                sha256:
                    sceneworks_core::model_artifacts::portable_binding::portable_closure_sha256(
                        &files,
                    )
                    .unwrap(),
            },
            tier: None,
            files,
        };
        assert_eq!(
            worker_catalog_member_path(&settings, "user.models.jsonc", "models", id, None, &member)
                .unwrap()
                .as_deref(),
            Some(std::fs::canonicalize(&root).unwrap().as_path())
        );
        std::fs::remove_file(root.join(".sceneworks-produced-artifact.json")).unwrap();
        assert!(worker_catalog_member_path(
            &settings,
            "user.models.jsonc",
            "models",
            id,
            None,
            &member
        )
        .is_err());
    }

    #[test]
    fn converted_builtin_digest_replaces_only_its_publisher_runtime_requirement() {
        let temp = TempDir::new().unwrap();
        let mut settings = settings(temp.path().join("data"));
        settings.config_dir = temp.path().join("config");
        settings.gpu_id = "mlx".to_owned();
        let id = "flux2_klein_9b_true_v2";
        let root = settings.data_dir.join("models/mlx").join(id);
        let entry = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .unwrap()["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == id)
            .unwrap()
            .clone();
        let payload = json!({"model":id,"modelManifestEntry":entry})
            .as_object()
            .unwrap()
            .clone();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model_index.json"), b"{}").unwrap();
        std::fs::write(root.join("model.safetensors"), b"converted weights").unwrap();
        crate::write_produced_artifact_receipt(&root, "convert-job", "source/repo", id).unwrap();
        let files = crate::portable_source_claims::capture_registered_files(&root).unwrap();
        let binding = PortableRouteBinding::new(
            "mlx",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                id,
                "mlx",
                "default",
                Some("default".to_owned()),
                canonical_authored_options_sha256(&json!({})).unwrap(),
                vec![PortableArtifactMember {
                    role: ArtifactMemberRole::Primary,
                    component_id: None,
                    source: PortableSourceIdentity::Digest {
                        sha256: sceneworks_core::model_artifacts::portable_binding::portable_closure_sha256(&files).unwrap(),
                    },
                    tier: Some("default".to_owned()),
                    files,
                }],
            )
            .unwrap()],
        )
        .unwrap();

        let library = temp.path().join("empty-hf");
        std::fs::create_dir_all(&library).unwrap();
        with_library(&library, || {
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings,
                Some(&binding),
            )
            .expect("converted output must load without the publisher conversion input installed");
            let selected = guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                .as_str()
                .unwrap();
            assert_eq!(
                std::fs::read(Path::new(selected).join("model.safetensors")).unwrap(),
                b"converted weights"
            );
            drop(guard);

            std::fs::remove_file(root.join(".sceneworks-produced-artifact.json")).unwrap();
            assert!(
                RuntimeSourceGuard::begin_with_binding(
                    &JobType::ImageGenerate,
                    &payload,
                    &settings,
                    Some(&binding)
                )
                .is_err(),
                "missing producer receipt must refuse before loading"
            );
            crate::write_produced_artifact_receipt(&root, "convert-job", "source/repo", id)
                .unwrap();
            std::fs::remove_file(root.join("model.safetensors")).unwrap();
            assert!(
                RuntimeSourceGuard::begin_with_binding(
                    &JobType::ImageGenerate,
                    &payload,
                    &settings,
                    Some(&binding)
                )
                .is_err(),
                "missing converted bytes must refuse before loading"
            );
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn converted_eros_keeps_both_runtime_co_requisites_bound() {
        use sceneworks_core::model_artifacts::ArtifactIdentity;

        let temp = TempDir::new().unwrap();
        let mut settings = settings(temp.path().join("data"));
        settings.config_dir = temp.path().join("config");
        settings.gpu_id = "mlx".to_owned();
        std::fs::create_dir_all(settings.data_dir.join("models")).unwrap();
        let id = "ltx_2_3_eros";
        let root = settings.data_dir.join("models/mlx").join(id);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model_index.json"), b"{}").unwrap();
        std::fs::write(root.join("model.safetensors"), b"converted weights").unwrap();
        crate::write_produced_artifact_receipt(&root, "convert-job", "source/repo", id).unwrap();
        let files = crate::portable_source_claims::capture_registered_files(&root).unwrap();
        let mut members = vec![PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest {
                sha256:
                    sceneworks_core::model_artifacts::portable_binding::portable_closure_sha256(
                        &files,
                    )
                    .unwrap(),
            },
            tier: Some("default".to_owned()),
            files,
        }];
        let entry = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .unwrap()["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == id)
            .unwrap()
            .clone();
        let selected = selected_model_artifact_closure(&entry, std::env::consts::OS, None);
        let library = temp.path().join("hf");
        let mut co_files = Vec::new();
        for (index, download) in selected["downloads"].as_array().unwrap().iter().enumerate() {
            if !is_co_requisite_download(download) {
                continue;
            }
            let repo = download["repo"].as_str().unwrap();
            let revision = download["revision"].as_str().unwrap();
            let selector = download["files"][0].as_str().unwrap();
            let file = selector.replace('*', "model.safetensors");
            seed_snapshot(&library, repo, revision, &file);
            let path = library
                .join(format!(
                    "models--{}",
                    sceneworks_core::hf_home::safe_repo_dir_name(repo).unwrap()
                ))
                .join("snapshots")
                .join(revision)
                .join(&file);
            co_files.push(path);
            members.push(PortableArtifactMember {
                role: ArtifactMemberRole::CoRequisite,
                component_id: Some(format!("modelManifestEntry/downloads/{index}")),
                source: PortableSourceIdentity::Revision {
                    identity: ArtifactIdentity::pinned(repo, revision, "default").unwrap(),
                },
                tier: Some("default".to_owned()),
                files: vec![PortableArtifactFile {
                    relative_path: file.into(),
                    size_bytes: 7,
                    sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
                }],
            });
        }
        assert_eq!(
            co_files.len(),
            2,
            "Eros must retain both catalog co-requisites"
        );
        let route = PortableRouteBinding::new(
            "mlx",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                id,
                "mlx",
                "default",
                Some("default".to_owned()),
                canonical_authored_options_sha256(&json!({})).unwrap(),
                members,
            )
            .unwrap()],
        )
        .unwrap();
        let payload = json!({"model":id,"mode":"text_to_video","modelManifestEntry":entry})
            .as_object()
            .unwrap()
            .clone();
        with_library(&library, || {
            std::fs::remove_file(&co_files[0]).unwrap();
            assert!(
                RuntimeSourceGuard::begin_with_binding(
                    &JobType::VideoGenerate,
                    &payload,
                    &settings,
                    Some(&route),
                )
                .is_err(),
                "missing Gemma co-requisite must refuse"
            );
            std::fs::write(&co_files[0], b"weights").unwrap();
            std::fs::write(&co_files[1], b"changed").unwrap();
            assert!(
                RuntimeSourceGuard::begin_with_binding(
                    &JobType::VideoGenerate,
                    &payload,
                    &settings,
                    Some(&route),
                )
                .is_err(),
                "substituted distill LoRA must refuse"
            );
            std::fs::write(&co_files[1], b"weights").unwrap();
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::VideoGenerate,
                &payload,
                &settings,
                Some(&route),
            )
            .expect("converted Eros and both frozen co-requisites are available");
            let selected = guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                .as_str()
                .unwrap();
            assert_eq!(
                std::fs::read(Path::new(selected).join("model.safetensors")).unwrap(),
                b"converted weights"
            );
        });
    }

    fn local_binding(
        slot: &str,
        model: &str,
        file: &Path,
        component: Option<(&str, &Path)>,
    ) -> PortableModelBinding {
        let roots = vec![file
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()];
        let capture = |role, id: Option<&str>, source_file: &Path| {
            let bytes = std::fs::read(source_file).unwrap();
            let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
            let member = LocalArtifactMember {
                role,
                component_id: id.map(str::to_owned),
                source: PortableSourceIdentity::Digest { sha256: digest },
                tier: None,
                files: vec![LocalArtifactFile {
                    relative_path: source_file.file_name().unwrap().into(),
                    path: source_file.to_path_buf(),
                }],
            };
            PortableArtifactMember::capture(&member, &roots).unwrap()
        };
        let mut members = vec![capture(ArtifactMemberRole::Primary, None, file)];
        if let Some((id, path)) = component {
            members.push(capture(
                ArtifactMemberRole::RequiredComponent,
                Some(id),
                path,
            ));
        }
        PortableModelBinding::new(
            slot,
            model,
            "cpu",
            "local",
            None,
            canonical_authored_options_sha256(&json!({})).unwrap(),
            members,
        )
        .unwrap()
    }

    fn training_payload(base_model: &str, kernel: &str, model_path: &Path) -> JsonObject {
        json!({
            "modelManifestEntry":{"id":base_model,"downloads":[],"paths":{"model":model_path}},
            "plan":{
                "schemaVersion":1,"planVersion":1,"jobId":"job",
                "target":{"targetId":"target","kernel":kernel,"family":"training","modality":"image","outputKind":"lora","baseModel":base_model,"baseModelPath":model_path},
                "dataset":{"datasetId":"dataset","datasetVersion":1,"rootPath":"/dataset","items":[]},
                "config":{"rank":16,"alpha":32,"learningRate":0.0001,"steps":10,"batchSize":1,"gradientAccumulation":1,"resolution":512,"saveEvery":10,"seed":1,"optimizer":"adamw8bit","advanced":{}},
                "output":{"loraId":"lora","outputDir":"/output","fileName":"lora.safetensors","format":"safetensors","triggerWords":[]},
                "provenance":{"datasetId":"dataset","datasetVersion":1,"targetId":"target","baseModel":base_model,"configSnapshot":{},"outputLoraId":"lora","sourceJobId":"job","createdAt":"2026-11-07T00:00:00Z"}
            }
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn download_free_registered_model_uses_admitted_local_variant() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/demo/model.safetensors");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"registered local model").unwrap();
        let payload = json!({
            "model":"demo",
            "modelManifestEntry":{"id":"demo","downloads":[],"paths":{"model":model}}
        })
        .as_object()
        .unwrap()
        .clone();
        let mut model_binding = local_binding("modelManifestEntry", "demo", &model, None);
        model_binding.variant = "local".to_owned();
        let route =
            PortableRouteBinding::new("cpu", std::env::consts::OS, vec![model_binding]).unwrap();
        RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings(data),
            Some(&route),
        )
        .unwrap();
    }

    #[test]
    fn managed_h3_import_uses_local_primary_variant_with_pinned_companions() {
        use sceneworks_core::model_artifacts::portable_binding::{
            portable_closure_sha256, PortableCheckpointPlan,
        };

        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
            &data,
            "imported-h3-guard",
        );
        let mut settings = settings(data);
        settings.gpu_id = "mlx".to_owned();
        settings.config_dir = temp.path().join("config");
        let catalog = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(catalog.parent().unwrap()).unwrap();
        std::fs::write(
            &catalog,
            json!({"models":[imported.entry.clone()]}).to_string(),
        )
        .unwrap();
        let entry = Value::Object(imported.entry.clone());
        let payload = json!({
            "model":"imported-h3-guard",
            "mode":"text_to_video",
            "modelManifestEntry":entry,
        })
        .as_object()
        .unwrap()
        .clone();
        let slot = "modelManifestEntry";
        let source = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
        let primary_files =
            crate::portable_source_claims::capture_registered_files(&imported.install_dir).unwrap();
        assert_eq!(
            crate::paths::normalize_app_managed_model_entry_path(
                &settings,
                Path::new(entry.pointer("/paths/model").unwrap().as_str().unwrap()),
                "portable catalog source",
            )
            .ok()
            .and_then(|path| crate::portable_source_claims::capture_registered_files(&path).ok()),
            Some(primary_files.clone()),
            "the worker catalog locator must resolve the admitted primary closure",
        );
        let mut members = vec![PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest {
                sha256: portable_closure_sha256(&primary_files).unwrap(),
            },
            tier: None,
            files: primary_files,
        }];
        let resolved =
            sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir)
                .resolve(&imported.checkpoint_id)
                .unwrap();
        for layer in resolved.layers {
            let bytes = std::fs::read(&layer.path).unwrap();
            let sha256 = format!("sha256:{:x}", Sha256::digest(&bytes));
            members.push(PortableArtifactMember {
                role: ArtifactMemberRole::RequiredComponent,
                component_id: Some(format!("{slot}/importPlan/layers/{}", layer.layer.layer_id)),
                source: PortableSourceIdentity::Digest {
                    sha256: sha256.clone(),
                },
                tier: None,
                files: vec![PortableArtifactFile {
                    relative_path: "artifact".into(),
                    size_bytes: bytes.len() as u64,
                    sha256,
                }],
            });
        }
        for (index, download) in entry["downloads"].as_array().unwrap().iter().enumerate() {
            let repo = download["repo"].as_str().unwrap();
            let revision = download["revision"].as_str().unwrap();
            let variant = download["variant"].as_str().unwrap_or("default");
            let (_, snapshot) = source.discover_snapshot(repo, Some(revision)).unwrap();
            let all = crate::portable_source_claims::capture_registered_files(&snapshot).unwrap();
            let files = all
                .into_iter()
                .filter(|file| {
                    download["files"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|selector| {
                            glob::Pattern::new(selector.as_str().unwrap())
                                .unwrap()
                                .matches(&file.relative_path.to_string_lossy())
                        })
                })
                .collect::<Vec<_>>();
            assert!(!files.is_empty(), "publisher companion selector {index}");
            members.push(PortableArtifactMember {
                role: ArtifactMemberRole::CoRequisite,
                component_id: Some(format!("{slot}/downloads/{index}")),
                source: PortableSourceIdentity::Revision {
                    identity: ArtifactIdentity::pinned(repo, revision, variant).unwrap(),
                },
                tier: Some(variant.to_owned()),
                files,
            });
        }
        let plan = PortableCheckpointPlan::from_import_plan(&resolved.plan, None).unwrap();
        let model = PortableModelBinding::new(
            slot,
            "imported-h3-guard",
            "mlx",
            "local",
            None,
            canonical_authored_options_sha256(&json!({})).unwrap(),
            members,
        )
        .unwrap()
        .with_checkpoint_plan(plan)
        .unwrap();
        let route = PortableRouteBinding::new("mlx", std::env::consts::OS, vec![model]).unwrap();
        let mut wrong_variant = route.clone();
        wrong_variant.models[0].variant = "default".to_owned();
        assert!(
            RuntimeSourceGuard::begin_with_binding(
                &JobType::VideoGenerate,
                &payload,
                &settings,
                Some(&wrong_variant),
            )
            .is_err(),
            "a plan does not authorize changing the admitted local variant"
        );
        let selected = imported.install_dir.join("config.json");
        let original = std::fs::read(&selected).unwrap();
        std::fs::write(&selected, b"substituted config").unwrap();
        assert!(
            RuntimeSourceGuard::begin_with_binding(
                &JobType::VideoGenerate,
                &payload,
                &settings,
                Some(&route),
            )
            .is_err(),
            "changed imported primary bytes refuse before loader dispatch"
        );
        std::fs::write(&selected, original).unwrap();
        RuntimeSourceGuard::begin_with_binding(
            &JobType::VideoGenerate,
            &payload,
            &settings,
            Some(&route),
        )
        .expect("unchanged imported primary and pinned companions pass the guard");
    }

    #[test]
    fn training_base_variant_follows_admitted_bf16_and_ltx_q4_tiers() {
        for (base_model, kernel, tier) in [
            ("z_image_turbo", "z_image_lora", "bf16"),
            ("ltx_2_5", "ltx_mlx_lora", "q4"),
        ] {
            let temp = TempDir::new().unwrap();
            let data = temp.path().join("data");
            let model = data.join("models/imports/base/model.safetensors");
            std::fs::create_dir_all(model.parent().unwrap()).unwrap();
            std::fs::write(&model, b"training base").unwrap();
            let payload = training_payload(base_model, kernel, &model);
            let mut base = local_binding("modelManifestEntry", base_model, &model, None);
            base.variant = tier.to_owned();
            base.tier = Some(tier.to_owned());
            base.members[0].tier = Some(tier.to_owned());
            let route = PortableRouteBinding::new("cpu", std::env::consts::OS, vec![base]).unwrap();
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::LoraTrain,
                &payload,
                &settings(data),
                Some(&route),
            )
            .unwrap();
            assert_ne!(
                guard.dispatch_payload()["plan"]["target"]["baseModelPath"],
                payload["plan"]["target"]["baseModelPath"]
            );
        }
    }

    #[test]
    fn training_download_selection_ignores_generation_default_tier() {
        use sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile;
        use sceneworks_core::model_artifacts::ArtifactIdentity;

        let temp = TempDir::new().unwrap();
        let model = temp.path().join("models/base/model.safetensors");
        let payload = training_payload("z_image_turbo", "z_image_lora", &model);
        let entry = json!({"id":"z_image_turbo","downloads":[
            {"provider":"huggingface","repo":"guardfx/training-base","revision":REV_A,
             "variant":"q4","default":true,"files":["q4/model.safetensors"]},
            {"provider":"huggingface","repo":"guardfx/training-base","revision":REV_A,
             "variant":"bf16","files":["bf16/model.safetensors"]}
        ]});
        let selected =
            selected_runtime_variant(&JobType::LoraTrain, &payload, "z_image_turbo").unwrap();
        assert_eq!(selected.as_deref(), Some("bf16"));
        let closure =
            selected_model_artifact_closure(&entry, std::env::consts::OS, selected.as_deref());
        assert_eq!(closure["downloads"][0]["variant"], "bf16");
        let requirements = selected_requirements_for_model(
            &entry,
            std::env::consts::OS,
            selected.as_deref(),
            temp.path(),
        );
        assert_eq!(requirements.requirements[0].variant, "bf16");
        assert_eq!(
            requirements.requirements[0].files[0],
            PathBuf::from("bf16/model.safetensors")
        );
        let data = temp.path().join("data");
        let library = temp.path().join("hf");
        with_library(&library, || {
            let source = sceneworks_core::hf_home::model_source_library(&data)
                .repository_root("guardfx/training-base")
                .unwrap()
                .join("snapshots")
                .join(REV_A)
                .join("bf16/model.safetensors");
            std::fs::create_dir_all(source.parent().unwrap()).unwrap();
            std::fs::write(&source, b"bf16 training base").unwrap();
            let mut payload = training_payload("z_image_turbo", "z_image_lora", &source);
            payload.insert("modelManifestEntry".to_owned(), entry.clone());
            let member = PortableArtifactMember {
                role: ArtifactMemberRole::Primary,
                component_id: None,
                source: PortableSourceIdentity::Revision {
                    identity: ArtifactIdentity::pinned("guardfx/training-base", REV_A, "bf16")
                        .unwrap(),
                },
                tier: Some("bf16".to_owned()),
                files: vec![PortableArtifactFile {
                    relative_path: "bf16/model.safetensors".into(),
                    size_bytes: 18,
                    sha256: format!("sha256:{:x}", Sha256::digest(b"bf16 training base")),
                }],
            };
            let base = PortableModelBinding::new(
                "modelManifestEntry",
                "z_image_turbo",
                "cpu",
                "bf16",
                Some("bf16".to_owned()),
                canonical_authored_options_sha256(&json!({})).unwrap(),
                vec![member],
            )
            .unwrap();
            let route = PortableRouteBinding::new("cpu", std::env::consts::OS, vec![base]).unwrap();
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::LoraTrain,
                &payload,
                &settings(data),
                Some(&route),
            )
            .unwrap();
            let rebound = guard.dispatch_payload()["plan"]["target"]["baseModelPath"]
                .as_str()
                .unwrap();
            assert_ne!(Path::new(rebound), source);
            assert_eq!(
                std::fs::read(Path::new(rebound).join("model.safetensors")).unwrap(),
                b"bf16 training base"
            );
        });
    }

    #[test]
    fn selected_taef1_training_auxiliary_uses_its_pinned_directory() {
        use sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile;
        use sceneworks_core::model_artifacts::ArtifactIdentity;
        use sceneworks_core::training::depth_anchoring::TAEF1_MODEL;

        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/zimage/model.safetensors");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"base").unwrap();
        let library = temp.path().join("hf");
        with_library(&library, || {
            let source = sceneworks_core::hf_home::model_source_library(&data)
                .repository_root(TAEF1_MODEL.repo)
                .unwrap()
                .join("snapshots")
                .join(TAEF1_MODEL.revision)
                .join(TAEF1_MODEL.file);
            std::fs::create_dir_all(source.parent().unwrap()).unwrap();
            std::fs::write(&source, b"taef1 bytes").unwrap();
            let mut base = local_binding("modelManifestEntry", "z_image_turbo", &model, None);
            base.variant = "local".to_owned();
            base.members.push(PortableArtifactMember {
                role: ArtifactMemberRole::CoRequisite,
                component_id: Some(TAEF1_MODEL.id.to_owned()),
                source: PortableSourceIdentity::Revision {
                    identity: ArtifactIdentity::pinned(
                        TAEF1_MODEL.repo,
                        TAEF1_MODEL.revision,
                        "default",
                    )
                    .unwrap(),
                },
                tier: None,
                files: vec![PortableArtifactFile {
                    relative_path: TAEF1_MODEL.file.into(),
                    size_bytes: 11,
                    sha256: format!("sha256:{:x}", Sha256::digest(b"taef1 bytes")),
                }],
            });
            let route = PortableRouteBinding::new("cpu", std::env::consts::OS, vec![base]).unwrap();
            let payload = json!({
                "modelManifestEntry":{"id":"z_image_turbo","downloads":[],"paths":{"model":model}},
                "plan":{
                    "schemaVersion":1,"planVersion":1,"jobId":"job",
                    "target":{"targetId":"target","kernel":"z_image_lora","family":"zimage","modality":"image","outputKind":"lora","baseModel":"z_image_turbo","baseModelPath":model},
                    "dataset":{"datasetId":"dataset","datasetVersion":1,"rootPath":"/dataset","items":[]},
                    "config":{"rank":16,"alpha":32,"learningRate":0.0001,"steps":10,"batchSize":1,"gradientAccumulation":1,"resolution":512,"saveEvery":10,"seed":1,"optimizer":"adamw8bit","advanced":{"depthAnchoringWeight":0.1,"depthAnchoringModel":"small"}},
                    "output":{"loraId":"lora","outputDir":"/output","fileName":"lora.safetensors","format":"safetensors","triggerWords":[]},
                    "provenance":{"datasetId":"dataset","datasetVersion":1,"targetId":"target","baseModel":"z_image_turbo","configSnapshot":{},"outputLoraId":"lora","sourceJobId":"job","createdAt":"2026-11-07T00:00:00Z"}
                }
            })
            .as_object()
            .unwrap()
            .clone();
            let mut dispatch = payload.clone();
            let prepared = prepare_portable_snapshot(
                &route,
                &JobType::LoraTrain,
                &payload,
                &settings(data),
                &mut dispatch,
                None,
            )
            .unwrap();
            assert!(prepared.is_some());
            let selected = dispatch["plan"]["__portableAuxPaths"][TAEF1_MODEL.id]
                .as_str()
                .unwrap();
            assert!(Path::new(selected).is_dir());
            assert_eq!(
                std::fs::read(Path::new(selected).join(TAEF1_MODEL.file)).unwrap(),
                b"taef1 bytes"
            );
            // Verified in place: the loader's directory links to the pinned hub file itself.
            assert_eq!(
                std::fs::canonicalize(Path::new(selected).join(TAEF1_MODEL.file)).unwrap(),
                std::fs::canonicalize(&source).unwrap()
            );
        });
    }

    #[test]
    fn failed_staging_verification_does_not_poison_reusable_bundle() {
        let temp = TempDir::new().unwrap();
        let original = temp.path().join("data/models/imports/model.safetensors");
        std::fs::create_dir_all(original.parent().unwrap()).unwrap();
        std::fs::write(&original, b"frozen").unwrap();
        let frozen = local_binding("modelManifestEntry", "demo", &original, None)
            .members
            .remove(0);
        let staging = temp.path().join("staging");
        let published = temp.path().join("bundle");
        std::fs::create_dir(&staging).unwrap();
        let copied = staging.join("model.safetensors");
        std::fs::write(&copied, b"substituted").unwrap();
        let verify = |root: &Path| {
            verify_private_snapshot_member(
                root,
                &frozen,
                Path::new(""),
                Path::new(""),
                false,
                &[temp.path().to_path_buf()],
            )
        };
        let error =
            publish_verified_snapshot(&staging, &published, || verify(&staging)).unwrap_err();
        assert!(error
            .to_string()
            .contains("private snapshot verification failed"));
        assert!(
            !published.exists(),
            "invalid copied bytes must not be published"
        );
        assert!(
            !staging.exists(),
            "failed staging must be removable for retry"
        );
        std::fs::create_dir(&staging).unwrap();
        std::fs::write(staging.join("model.safetensors"), b"frozen").unwrap();
        publish_verified_snapshot(&staging, &published, || verify(&staging)).unwrap();
        assert_eq!(
            std::fs::read(published.join("model.safetensors")).unwrap(),
            b"frozen"
        );
    }

    #[test]
    fn local_multifile_components_rebind_synthetic_keys_across_hosts() {
        let temp = TempDir::new().unwrap();
        let api = temp.path().join("api/models");
        let worker = temp.path().join("worker/models");
        for root in [&api, &worker] {
            std::fs::create_dir_all(root).unwrap();
        }
        let files = [
            ("model", b"primary".as_slice()),
            ("vae", b"decoder".as_slice()),
            ("encoder", b"encoder".as_slice()),
        ];
        for (name, bytes) in files {
            std::fs::write(api.join(format!("{name}.safetensors")), bytes).unwrap();
            std::fs::write(worker.join(format!("{name}.safetensors")), bytes).unwrap();
        }
        let mut settings = Settings::for_test(temp.path().join("worker"));
        settings.config_dir = temp.path().join("worker-config");
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({"models":[{
                "id":"multi-local", "modelPath":worker.join("model.safetensors"),
                "paths":{"vae":worker.join("vae.safetensors")},
                "components":[{"id":"encoder", "path":worker.join("encoder.safetensors")}]
            }]}))
            .unwrap(),
        )
        .unwrap();
        let capture = |role, component_id: Option<&str>, path: PathBuf| {
            let bytes = std::fs::read(&path).unwrap();
            let member = LocalArtifactMember {
                role,
                component_id: component_id.map(str::to_owned),
                source: PortableSourceIdentity::Digest {
                    sha256: format!("sha256:{:x}", Sha256::digest(&bytes)),
                },
                tier: None,
                files: vec![LocalArtifactFile {
                    relative_path: path.file_name().unwrap().into(),
                    path,
                }],
            };
            PortableArtifactMember::capture(&member, &[api.clone(), worker.clone()]).unwrap()
        };
        let members = vec![
            capture(
                ArtifactMemberRole::Primary,
                None,
                api.join("model.safetensors"),
            ),
            capture(
                ArtifactMemberRole::RequiredComponent,
                Some("modelManifestEntry/paths/vae"),
                api.join("vae.safetensors"),
            ),
            capture(
                ArtifactMemberRole::RequiredComponent,
                Some("modelManifestEntry/components/0"),
                api.join("encoder.safetensors"),
            ),
        ];
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            "multi-local",
            "cpu",
            "local",
            None,
            canonical_authored_options_sha256(&json!({})).unwrap(),
            members,
        )
        .unwrap();
        let route = PortableRouteBinding::new("cpu", std::env::consts::OS, vec![model]).unwrap();
        let payload = json!({"model":"multi-local", "modelManifestEntry":{
            "id":"multi-local", "modelPath":api.join("model.safetensors"),
            "source":{"provider":"local","path":api.join("model.safetensors")},
            "paths":{"vae":api.join("vae.safetensors")},
            "components":[{"id":"encoder", "path":api.join("encoder.safetensors")}],
            "downloads":[]
        }})
        .as_object()
        .unwrap()
        .clone();
        let guard = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&route),
        )
        .unwrap();
        let dispatch = &guard.dispatch_payload()["modelManifestEntry"];
        for (pointer, expected) in [
            ("/modelPath", b"primary".as_slice()),
            ("/paths/vae", b"decoder".as_slice()),
            ("/components/0/path", b"encoder".as_slice()),
        ] {
            let selected = dispatch.pointer(pointer).and_then(Value::as_str).unwrap();
            assert_eq!(std::fs::read(selected).unwrap(), expected);
            assert!(!selected.starts_with(api.to_str().unwrap()));
        }
    }

    #[test]
    fn duplicate_catalog_id_selects_the_frozen_file_ledger() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        let wrong = data.join("models/wrong");
        let right = data.join("models/right");
        std::fs::create_dir_all(&wrong).unwrap();
        std::fs::create_dir_all(&right).unwrap();
        std::fs::write(wrong.join("model.safetensors"), b"wrong").unwrap();
        std::fs::write(right.join("model.safetensors"), b"right").unwrap();
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({"models":[
                {"id":"shared","modelPath":wrong},
                {"id":"shared","modelPath":right}
            ]}))
            .unwrap(),
        )
        .unwrap();
        let files = crate::portable_source_claims::capture_registered_files(&right).unwrap();
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest {
                sha256: files[0].sha256.clone(),
            },
            tier: None,
            files,
        };
        assert_eq!(
            worker_catalog_member_path(
                &settings,
                "user.models.jsonc",
                "models",
                "shared",
                None,
                &member
            )
            .unwrap(),
            Some(std::fs::canonicalize(right).unwrap())
        );
    }

    #[test]
    fn relative_lora_catalog_member_uses_global_and_project_scope() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        for (manifest, relative, selected) in [
            (
                settings.config_dir.join("manifests/user.loras.jsonc"),
                "loras/global",
                data.join("loras/global"),
            ),
            (
                data.join("projects/demo/loras/manifest.jsonc"),
                "loras/project",
                data.join("projects/demo/loras/project"),
            ),
        ] {
            std::fs::create_dir_all(&selected).unwrap();
            std::fs::write(selected.join("adapter.safetensors"), b"adapter").unwrap();
            std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
            std::fs::write(
                &manifest,
                serde_json::to_vec(&json!({"loras":[{"id":"trained","source":{"path":relative}}]}))
                    .unwrap(),
            )
            .unwrap();
            let files = crate::portable_source_claims::capture_registered_files(&selected).unwrap();
            let member = PortableArtifactMember {
                role: ArtifactMemberRole::Lora,
                component_id: None,
                source: PortableSourceIdentity::Digest {
                    sha256: files[0].sha256.clone(),
                },
                tier: None,
                files,
            };
            assert_eq!(
                worker_catalog_member_path_at(
                    &settings, &manifest, "loras", "trained", None, &member,
                )
                .unwrap(),
                Some(selected.canonicalize().unwrap())
            );
        }
    }

    #[test]
    fn portable_binding_rebinds_same_bytes_under_another_root_and_refuses_substitution() {
        let temp = TempDir::new().unwrap();
        let first = temp
            .path()
            .join("first/models/imports/demo/model.safetensors");
        let second = temp
            .path()
            .join("second/models/imports/demo/model.safetensors");
        std::fs::create_dir_all(first.parent().unwrap()).unwrap();
        std::fs::create_dir_all(second.parent().unwrap()).unwrap();
        std::fs::write(&first, b"same checkpoint").unwrap();
        std::fs::write(&second, b"same checkpoint").unwrap();
        let binding = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![local_binding("modelManifestEntry", "m", &first, None)],
        )
        .unwrap();
        let settings = settings(temp.path().join("second"));
        let route_digest = portable_cache_route_digest(&binding).unwrap();
        let cache_entry = settings
            .data_dir
            .join("models/portable-attempts")
            .join(route_digest);
        let interrupted = cache_entry.join("staging-interrupted");
        std::fs::create_dir_all(&interrupted).unwrap();
        std::fs::write(interrupted.join("partial"), b"incomplete").unwrap();
        let payload = json!({
            "model":"m", "modelManifestEntry":{"id":"m", "downloads":[],
                "paths":{"model":second}}
        })
        .as_object()
        .unwrap()
        .clone();
        let guard = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&binding),
        )
        .unwrap();
        assert!(!interrupted.exists(), "stale staging copy must be removed");
        let dispatched = guard
            .dispatch_payload()
            .get("modelManifestEntry")
            .and_then(|entry| entry.get("paths"))
            .and_then(|paths| paths.get("model"))
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from)
            .unwrap();
        assert_ne!(dispatched, second);
        assert_eq!(std::fs::read(&dispatched).unwrap(), b"same checkpoint");
        std::fs::write(&second, b"evil checkpoint").unwrap();
        assert_eq!(std::fs::read(&dispatched).unwrap(), b"same checkpoint");
        drop(guard);
        assert!(
            dispatched.exists(),
            "content-addressed copy must be reusable"
        );
        std::fs::write(&second, b"same checkpoint").unwrap();
        let reused = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&binding),
        )
        .unwrap();
        let reused_path = reused.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
            .as_str()
            .unwrap();
        assert_eq!(Path::new(reused_path), dispatched);
        drop(reused);
        std::fs::write(&second, b"evil checkpoint").unwrap();
        let error = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&binding),
        )
        .unwrap_err();
        assert!(error.to_string().contains("bytes differ"), "{error}");
        std::fs::remove_file(&second).unwrap();
        assert!(RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&binding)
        )
        .is_err());
    }

    #[test]
    fn portable_binding_checks_the_advanced_override_the_loader_would_open() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let selected = data.join("models/imports/demo/model.safetensors");
        let substituted = data.join("models/imports/other/model.safetensors");
        std::fs::create_dir_all(selected.parent().unwrap()).unwrap();
        std::fs::create_dir_all(substituted.parent().unwrap()).unwrap();
        std::fs::write(&selected, b"admitted checkpoint").unwrap();
        std::fs::write(&substituted, b"admittex checkpoint").unwrap();
        let mut model = local_binding("modelManifestEntry", "m", &selected, None);
        model.variant = "local".to_owned();
        let binding = PortableRouteBinding::new("cpu", std::env::consts::OS, vec![model]).unwrap();
        let payload = json!({"model":"m", "advanced":{"modelPath":substituted},
            "modelManifestEntry":{"id":"m", "downloads":[], "paths":{"model":selected}}})
        .as_object()
        .unwrap()
        .clone();
        let error = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings(data),
            Some(&binding),
        )
        .unwrap_err();
        assert!(error.to_string().contains("bytes differ"), "{error}");
    }

    #[test]
    fn publisher_byte_advanced_override_rebinds_to_verified_revision_snapshot() {
        let temp = TempDir::new().unwrap();
        let (settings, library, mut payload) = installed_model(&temp);
        let api_host_path = temp.path().join("api-host/models/publisher-copy");
        payload.insert("advanced".to_owned(), json!({"modelPath": api_host_path}));
        let binding = portable_installed_model_binding();
        with_library(&library, || {
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings,
                Some(&binding),
            )
            .expect("admitted publisher bytes must rebind despite the API-host path being absent");
            let dispatched = guard.dispatch_payload();
            let selected = dispatched["advanced"]["modelPath"].as_str().unwrap();
            assert_ne!(Path::new(selected), api_host_path);
            assert_eq!(dispatched["modelManifestEntry"]["paths"]["model"], selected);
            assert_eq!(
                std::fs::read(Path::new(selected).join("model.safetensors")).unwrap(),
                b"weights"
            );
            let source = library
                .join("models--guardfx--model/snapshots")
                .join(REV_A)
                .join("model.safetensors");
            assert_eq!(
                std::fs::canonicalize(Path::new(selected).join("model.safetensors")).unwrap(),
                std::fs::canonicalize(&source).unwrap(),
                "the pinned snapshot is verified in place, not copied"
            );
            drop(guard);
            std::fs::write(source, b"changed").unwrap();
            let error = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings,
                Some(&binding),
            )
            .expect_err("changed source bytes must be refused");
            assert!(
                error
                    .to_string()
                    .contains("bytes differ from admitted identity"),
                "{error}"
            );
        });
    }

    #[test]
    fn array_primary_revision_rewrites_advanced_loader_override() {
        let binding = portable_installed_model_binding();
        let primary = binding.models[0]
            .members
            .iter()
            .find(|member| member.role == ArtifactMemberRole::Primary)
            .unwrap();
        let mut payload = json!({
            "advanced": {"modelPath": "/api-host/publisher-copy"},
            "modelManifestEntries": [{"id": "m", "paths": {"model": "/api-host/publisher-copy"}}]
        })
        .as_object()
        .unwrap()
        .clone();
        let private = Path::new("/worker/private/verified");
        rewrite_loader_locator(
            &mut payload,
            "modelManifestEntries/0",
            primary,
            private,
            false,
        )
        .unwrap();
        assert_eq!(payload["advanced"]["modelPath"], private.to_str().unwrap());
        assert_eq!(
            payload["modelManifestEntries"][0]["paths"]["model"],
            private.to_str().unwrap()
        );
    }

    #[test]
    fn portable_binding_rebinds_training_plan_path_to_verified_worker_copy() {
        assert_eq!(
            crate::portable_trainer_engine_id_for("sdxl_lora", "sdxl"),
            Some("sdxl"),
            "the cross-host training fixture must select a real native trainer"
        );
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("worker");
        let source = data.join("models/imports/base/model.safetensors");
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"frozen training base").unwrap();
        let mut model = local_binding("baseModelManifestEntry", "sdxl", &source, None);
        model.variant = "bf16".to_owned();
        let binding = PortableRouteBinding::new("cpu", std::env::consts::OS, vec![model]).unwrap();
        let payload = json!({
            "baseModelManifestEntry":{"id":"sdxl", "downloads":[], "paths":{"model":source}},
            "plan":{
                "schemaVersion":1, "planVersion":1, "jobId":"job-1",
                "target":{"targetId":"base_lora", "kernel":"sdxl_lora", "family":"sdxl",
                    "modality":"image", "outputKind":"lora", "baseModel":"sdxl",
                    "baseModelPath":"/other-host/models/base"},
                "dataset":{"datasetId":"ds", "datasetVersion":1,
                    "rootPath":"/other-host/dataset", "items":[]},
                "config":{"rank":16, "alpha":32, "learningRate":0.0001, "steps":1,
                    "batchSize":1, "gradientAccumulation":1, "resolution":512,
                    "saveEvery":1, "seed":1, "optimizer":"adamw8bit", "advanced":{}},
                "output":{"loraId":"lora", "outputDir":"/other-host/output",
                    "fileName":"lora.safetensors", "format":"safetensors", "triggerWords":[]},
                "provenance":{"datasetId":"ds", "datasetVersion":1, "targetId":"base_lora",
                    "baseModel":"sdxl", "configSnapshot":{}, "outputLoraId":"lora",
                    "sourceJobId":"job-1", "createdAt":"2026-11-07T00:00:00Z"}
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let guard = RuntimeSourceGuard::begin_with_binding(
            &JobType::LoraTrain,
            &payload,
            &settings(data),
            Some(&binding),
        )
        .unwrap();
        let rebound = guard.dispatch_payload()["plan"]["target"]["baseModelPath"]
            .as_str()
            .unwrap();
        assert_ne!(rebound, "/other-host/models/base");
        assert_eq!(std::fs::read(rebound).unwrap(), b"frozen training base");
        assert_eq!(
            payload["plan"]["target"]["baseModelPath"],
            "/other-host/models/base"
        );
    }

    #[test]
    fn portable_cache_eviction_respects_active_lease_and_entry_bound() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let root = settings.data_dir.join("models/portable-attempts");
        std::fs::create_dir_all(&root).unwrap();
        let mut entries = Vec::new();
        for index in 0..10 {
            let path = root.join(format!("{index:064x}"));
            std::fs::create_dir_all(path.join("bundle")).unwrap();
            std::fs::write(path.join("bundle/file"), b"one").unwrap();
            std::fs::write(path.join(".last-used"), b"").unwrap();
            std::fs::write(path.join(PORTABLE_BUNDLE_LAYOUT_MARKER), b"").unwrap();
            entries.push(path);
        }
        let active = FileLock::open_exclusive(&entries[1].join(".lease")).unwrap();
        prune_portable_snapshots(Some(&entries[0]), &settings).unwrap();
        assert!(entries[0].join("bundle/file").exists());
        assert!(entries[1].join("bundle/file").exists());
        assert_eq!(
            entries
                .iter()
                .filter(|path| path.join("bundle/file").exists())
                .count(),
            8
        );
        drop(active);
    }

    /// A published route bundle is immutable, so a second attempt on the same frozen route (a
    /// second GPU worker sharing `data_dir`) reads it while the first attempt still holds its
    /// lease. Eviction stays excluded for as long as any attempt reads the bundle.
    #[test]
    fn published_route_bundle_serves_a_second_attempt_while_the_first_reads_it() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/demo/model.safetensors");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"model").unwrap();
        let settings = settings(data);
        let payload =
            json!({"modelManifestEntry":{"id":"m", "downloads":[], "paths":{"model":model}}})
                .as_object()
                .unwrap()
                .clone();
        let binding = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![local_binding("modelManifestEntry", "m", &model, None)],
        )
        .unwrap();
        let selected = |guard: &RuntimeSourceGuard| {
            guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        let first = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&binding),
        )
        .unwrap();
        let first_path = selected(&first);
        let root = settings.data_dir.join("models/portable-attempts");
        assert!(Path::new(&first_path).starts_with(&root));

        let (sender, receiver) = std::sync::mpsc::channel();
        {
            let (payload, settings, binding) = (payload.clone(), settings.clone(), binding.clone());
            std::thread::spawn(move || {
                let second = RuntimeSourceGuard::begin_with_binding(
                    &JobType::ImageGenerate,
                    &payload,
                    &settings,
                    Some(&binding),
                )
                .map(|guard| selected(&guard));
                let _ = sender.send(second);
            });
        }
        let second_path = receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the second attempt must not wait for the first attempt's read lease")
            .unwrap();
        assert_eq!(second_path, first_path, "both attempts read one bundle");

        // While the first attempt reads the bundle, an evictor cannot take it.
        let entry = Path::new(&first_path)
            .ancestors()
            .find(|path| path.parent() == Some(root.as_path()))
            .unwrap()
            .to_path_buf();
        assert!(FileLock::try_open_exclusive(&entry.join(".lease")).is_err());
        drop(first);
        assert!(FileLock::try_open_exclusive(&entry.join(".lease")).is_ok());
    }

    #[test]
    fn portable_cache_keeps_one_over_target_active_route() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let root = settings.data_dir.join("models/portable-attempts");
        let current = root.join(format!("{:064x}", 1));
        let old = root.join(format!("{:064x}", 2));
        for path in [&current, &old] {
            std::fs::create_dir_all(path.join("bundle")).unwrap();
            std::fs::write(path.join(".last-used"), b"").unwrap();
            std::fs::write(path.join(PORTABLE_BUNDLE_LAYOUT_MARKER), b"").unwrap();
        }
        let sparse = std::fs::File::create(current.join("bundle/weights")).unwrap();
        sparse.set_len(PORTABLE_CACHE_MAX_BYTES + 1).unwrap();
        std::fs::write(old.join("bundle/old"), b"old").unwrap();
        prune_portable_snapshots(Some(&current), &settings).unwrap();
        assert!(current.join("bundle/weights").exists());
        assert!(!old.join("bundle/old").exists());
    }

    /// SW-7033 Tier 2: the earlier layout copied pinned Hugging Face members into route bundles
    /// (an LTX-2.5 route held ~115 GB). Nothing reads those bundles any more, so retention reclaims
    /// them first even under the size and count targets, from a route that needs no bundle at all;
    /// a current-layout bundle and a leased obsolete one stay.
    #[test]
    fn obsolete_layout_bundles_are_reclaimed_under_the_retention_targets() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let root = settings.data_dir.join("models/portable-attempts");
        let make = |index: u8, marked: bool| {
            let path = root.join(format!("{index:064x}"));
            std::fs::create_dir_all(path.join("bundle/hf")).unwrap();
            std::fs::write(path.join("bundle/hf/weights"), b"copied").unwrap();
            std::fs::write(path.join(".last-used"), b"").unwrap();
            if marked {
                std::fs::write(path.join(PORTABLE_BUNDLE_LAYOUT_MARKER), b"").unwrap();
            }
            path
        };
        let obsolete = make(1, false);
        let leased_obsolete = make(2, false);
        let current_layout = make(3, true);
        let reading = FileLock::open_exclusive(&leased_obsolete.join(".lease")).unwrap();
        prune_portable_snapshots(None, &settings).unwrap();
        assert!(!obsolete.join("bundle").exists());
        assert!(leased_obsolete.join("bundle/hf/weights").exists());
        assert!(current_layout.join("bundle/hf/weights").exists());
        drop(reading);
        prune_portable_snapshots(None, &settings).unwrap();
        assert!(!leased_obsolete.join("bundle").exists());
        assert!(current_layout.join("bundle/hf/weights").exists());
    }

    #[test]
    fn portable_worker_claim_auxiliary_uses_verified_private_path() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/demo/model.safetensors");
        let auxiliary = data.join("cache/ipadapter-flux/ip_adapter.safetensors");
        for path in [&model, &auxiliary] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        }
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&auxiliary, b"selected adapter").unwrap();
        let settings = settings(data);
        let payload =
            json!({"modelManifestEntry":{"id":"m", "downloads":[], "paths":{"model":model}},
            "advanced":{"__portableAuxPaths":{"flux_ip_adapter":"/untrusted/path"}}})
            .as_object()
            .unwrap()
            .clone();
        let (claim, _) = crate::portable_source_claims::capture_selected_source(
            "flux_ip_adapter",
            &settings,
            &payload,
        )
        .unwrap()
        .unwrap();
        let mut model_binding = local_binding("modelManifestEntry", "m", &model, None);
        model_binding.members.push(PortableArtifactMember {
            role: ArtifactMemberRole::CoRequisite,
            component_id: Some("flux_ip_adapter".to_owned()),
            source: PortableSourceIdentity::WorkerClaim {
                source_id: claim.source_id.clone(),
                provenance: claim.provenance.clone(),
                closure_sha256: claim.closure_sha256.clone(),
            },
            tier: None,
            files: claim.files.clone(),
        });
        let binding =
            PortableRouteBinding::new("cpu", std::env::consts::OS, vec![model_binding]).unwrap();
        let guard = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&binding),
        )
        .unwrap();
        let selected = guard.dispatch_payload()["advanced"]["__portableAuxPaths"]
            ["flux_ip_adapter"]
            .as_str()
            .unwrap();
        assert_ne!(selected, "/untrusted/path");
        assert_ne!(selected, auxiliary.to_str().unwrap());
        std::fs::write(&auxiliary, b"changed adapter!").unwrap();
        assert_eq!(std::fs::read(selected).unwrap(), b"selected adapter");
        assert_eq!(
            payload["advanced"]["__portableAuxPaths"]["flux_ip_adapter"],
            "/untrusted/path"
        );
    }

    #[test]
    fn portable_binding_refuses_omitted_route_selected_auxiliary() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/flux/model.safetensors");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"model").unwrap();
        let binding = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![local_binding(
                "modelManifestEntry",
                "flux_dev",
                &model,
                None,
            )],
        )
        .unwrap();
        let payload = json!({
            "model":"flux_dev", "mode":"character_image", "referenceAssetId":"reference",
            "modelManifestEntry":{"id":"flux_dev", "downloads":[], "paths":{"model":model}}
        })
        .as_object()
        .unwrap()
        .clone();
        let error = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings(data),
            Some(&binding),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("omits selected auxiliary source"),
            "{error}"
        );
    }

    #[test]
    fn portable_revision_auxiliary_is_verified_in_place_into_loader_path_map() {
        use sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile;
        use sceneworks_core::model_artifacts::ArtifactIdentity;

        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/flux/model.safetensors");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"model").unwrap();
        let external = temp.path().join("external-hf");
        with_library(&external, || {
            let library = sceneworks_core::hf_home::model_source_library(&data);
            let repo = "guardfx/strict-control";
            let file = "diffusion_pytorch_model.safetensors";
            let expected = library
                .repository_root(repo)
                .unwrap()
                .join("snapshots")
                .join(REV_A)
                .join(file);
            std::fs::create_dir_all(expected.parent().unwrap()).unwrap();
            std::fs::write(&expected, b"weights").unwrap();
            let source = library
                .discover_snapshot(repo, Some(REV_A))
                .unwrap()
                .1
                .join(file);
            let partial = data.join("models/resolved-test/partial-snapshot");
            std::fs::create_dir_all(&partial).unwrap();
            std::fs::write(partial.join("base.safetensors"), b"base only").unwrap();
            let scope =
                sceneworks_core::model_artifacts::local_preference::prefer_local_snapshots(vec![
                    LocalArtifactOverlayEntry {
                        repository: repo.to_owned(),
                        revision: REV_A.to_owned(),
                        snapshot_root: partial.clone(),
                        files: vec!["base.safetensors".into()],
                    },
                ])
                .unwrap();
            let selected = discover_frozen_member_snapshot(
                &settings(data.clone()),
                &ArtifactIdentity::pinned(repo, REV_A, "default").unwrap(),
                &[PortableArtifactFile {
                    relative_path: file.into(),
                    size_bytes: 7,
                    sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
                }],
            )
            .unwrap();
            assert_eq!(selected.join(file), source);
            std::fs::write(partial.join(file), b"corrupt").unwrap();
            let selected = discover_frozen_member_snapshot(
                &settings(data.clone()),
                &ArtifactIdentity::pinned(repo, REV_A, "default").unwrap(),
                &[PortableArtifactFile {
                    relative_path: file.into(),
                    size_bytes: 7,
                    sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
                }],
            )
            .unwrap();
            assert_eq!(
                selected.join(file),
                source,
                "bad preferred bytes must fall back"
            );
            drop(scope);
            let mut base = local_binding("modelManifestEntry", "flux_dev", &model, None);
            base.members.push(PortableArtifactMember {
                role: ArtifactMemberRole::Control,
                component_id: Some("strict_control".to_owned()),
                source: PortableSourceIdentity::Revision {
                    identity: ArtifactIdentity::pinned(repo, REV_A, "default").unwrap(),
                },
                tier: None,
                files: vec![PortableArtifactFile {
                    relative_path: file.into(),
                    size_bytes: 7,
                    sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
                }],
            });
            let binding =
                PortableRouteBinding::new("cpu", std::env::consts::OS, vec![base]).unwrap();
            assert!(crate::utility_dispatch::missing_portable_revision_files(
                &binding,
                &settings(data.clone()),
            )
            .is_empty());
            std::fs::remove_file(&source).unwrap();
            let cold = crate::utility_dispatch::missing_portable_revision_files(
                &binding,
                &settings(data.clone()),
            );
            assert_eq!(
                cold[&(repo.to_owned(), REV_A.to_owned())],
                vec![file.to_owned()]
            );
            std::fs::write(&source, b"weights").unwrap();
            let payload = json!({
                "model":"flux_dev", "advanced":{"poses":[{"id":"pose"}]},
                "modelManifestEntry":{"id":"flux_dev", "downloads":[], "paths":{"model":model}}
            })
            .as_object()
            .unwrap()
            .clone();
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings(data.clone()),
                Some(&binding),
            )
            .unwrap();
            let mapped = guard.dispatch_payload()["advanced"]["__portableAuxPaths"]
                ["strict_control"]
                .as_str()
                .unwrap();
            assert_eq!(
                std::fs::canonicalize(mapped).unwrap(),
                std::fs::canonicalize(&source).unwrap(),
                "the pinned auxiliary is verified in place, not copied"
            );
            drop(guard);
            std::fs::write(&source, b"changed").unwrap();
            let error = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings(data),
                Some(&binding),
            )
            .expect_err("changed auxiliary bytes must be refused");
            assert!(
                error
                    .to_string()
                    .contains("bytes differ from admitted identity"),
                "{error}"
            );
        });
    }

    #[test]
    fn portable_binding_refuses_a_substituted_protected_component() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/demo/model.safetensors");
        let encoder = data.join("models/components/encoder.safetensors");
        let changed = data.join("models/components/changed.safetensors");
        for file in [&model, &encoder, &changed] {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        }
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&encoder, b"protected encoder").unwrap();
        std::fs::write(&changed, b"protfcted encoder").unwrap();
        let binding = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![local_binding(
                "modelManifestEntry",
                "m",
                &model,
                Some(("text_encoder", &encoder)),
            )],
        )
        .unwrap();
        let payload = json!({"model":"m", "modelManifestEntry":{
            "id":"m", "downloads":[], "paths":{"model":model},
            "components":[{"id":"text_encoder", "path":changed}]
        }})
        .as_object()
        .unwrap()
        .clone();
        let error = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings(data),
            Some(&binding),
        )
        .unwrap_err();
        assert!(error.to_string().contains("bytes differ"), "{error}");
    }

    #[test]
    fn portable_binding_verifies_every_model_slot() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/demo/model.safetensors");
        let base = data.join("models/imports/base/base.safetensors");
        for file in [&model, &base] {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, b"original").unwrap();
        }
        let binding = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![
                local_binding("modelManifestEntry", "m", &model, None),
                local_binding("baseModelManifestEntry", "base", &base, None),
            ],
        )
        .unwrap();
        let payload = json!({"model":"m",
            "modelManifestEntry":{"id":"m", "downloads":[], "paths":{"model":model}},
            "baseModelManifestEntry":{"id":"base", "downloads":[], "paths":{"model":base}}
        })
        .as_object()
        .unwrap()
        .clone();
        let settings = settings(data);
        RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&binding),
        )
        .unwrap();
        std::fs::write(&base, b"replaced").unwrap();
        assert!(RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&binding)
        )
        .is_err());
    }

    #[test]
    fn portable_component_uses_worker_catalog_path_across_hosts() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("worker");
        let model = data.join("models/imports/demo/model.safetensors");
        let component = data.join("models/components/encoder.safetensors");
        let api_component = temp
            .path()
            .join("api/models/components/encoder.safetensors");
        for file in [&model, &component, &api_component] {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        }
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&component, b"same encoder").unwrap();
        std::fs::write(&api_component, b"same encoder").unwrap();
        let route = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![local_binding(
                "modelManifestEntry",
                "m",
                &model,
                Some(("text_encoder", &component)),
            )],
        )
        .unwrap();
        let mut settings = settings(data);
        settings.config_dir = temp.path().join("worker-config");
        let catalog = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(catalog.parent().unwrap()).unwrap();
        std::fs::write(
            &catalog,
            serde_json::to_vec(&json!({"models":[{
                "id":"m", "paths":{"model":model},
                "components":[{"id":"text_encoder","path":component}]
            }]}))
            .unwrap(),
        )
        .unwrap();
        let payload = json!({"model":"m", "modelManifestEntry":{
            "id":"m", "downloads":[], "paths":{"model":model},
            "components":[{"id":"text_encoder","path":api_component}]
        }})
        .as_object()
        .unwrap()
        .clone();
        let guard = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&route),
        )
        .unwrap();
        let mapped = guard.dispatch_payload()["modelManifestEntry"]["components"][0]["path"]
            .as_str()
            .unwrap();
        assert_eq!(std::fs::read(mapped).unwrap(), b"same encoder");
        assert_ne!(mapped, api_component.to_str().unwrap());
    }

    #[test]
    fn portable_binding_uses_adapter_slot_even_when_ids_repeat() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/demo/model.safetensors");
        let first = data.join("loras/first.safetensors");
        let second = data.join("loras/second.safetensors");
        for file in [&model, &first, &second] {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        }
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&first, b"wrong adapter").unwrap();
        std::fs::write(&second, b"admitted adapter").unwrap();
        let primary = local_binding("modelManifestEntry", "m", &model, None);
        let adapter = |slot: &str, file: &Path| {
            let bytes = std::fs::read(file).unwrap();
            PortableArtifactMember::capture(
                &LocalArtifactMember {
                    role: ArtifactMemberRole::Lora,
                    component_id: Some(slot.to_owned()),
                    source: PortableSourceIdentity::Digest {
                        sha256: format!("sha256:{:x}", Sha256::digest(&bytes)),
                    },
                    tier: None,
                    files: vec![LocalArtifactFile {
                        relative_path: file.file_name().unwrap().into(),
                        path: file.to_path_buf(),
                    }],
                },
                std::slice::from_ref(&data),
            )
            .unwrap()
        };
        let mut members = primary.members;
        members.push(adapter("loras/0", &first));
        members.push(adapter("loras/1", &second));
        let binding = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                "m",
                "cpu",
                "local",
                None,
                canonical_authored_options_sha256(&json!({})).unwrap(),
                members,
            )
            .unwrap()],
        )
        .unwrap();
        let payload = json!({"model":"m", "modelManifestEntry":{"id":"m", "downloads":[], "paths":{"model":model}},
            "loras":[{"id":"same", "path":first}, {"id":"same", "path":second}]
        }).as_object().unwrap().clone();
        RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings(data),
            Some(&binding),
        )
        .unwrap();
        let mut omitted = binding.clone();
        omitted.models[0]
            .members
            .retain(|member| member.component_id.as_deref() != Some("loras/0"));
        assert!(RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings(temp.path().join("data")),
            Some(&omitted)
        )
        .is_err());
    }

    #[test]
    fn portable_lora_resolves_worker_catalog_under_a_different_root() {
        let temp = TempDir::new().unwrap();
        let api_adapter = temp.path().join("api/loras/adapter.safetensors");
        let data = temp.path().join("worker");
        let worker_adapter = data.join("loras/adapter.safetensors");
        let model = data.join("models/imports/demo/model.safetensors");
        for file in [&api_adapter, &worker_adapter, &model] {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        }
        std::fs::write(&api_adapter, b"same adapter").unwrap();
        std::fs::write(&worker_adapter, b"same adapter").unwrap();
        std::fs::write(&model, b"model").unwrap();
        let mut binding = local_binding("modelManifestEntry", "m", &model, None);
        let adapter = PortableArtifactMember::capture(
            &LocalArtifactMember {
                role: ArtifactMemberRole::Lora,
                component_id: Some("loras/0".to_owned()),
                source: PortableSourceIdentity::Digest {
                    sha256: format!("sha256:{:x}", Sha256::digest(b"same adapter")),
                },
                tier: None,
                files: vec![LocalArtifactFile {
                    relative_path: "adapter.safetensors".into(),
                    path: api_adapter.clone(),
                }],
            },
            &[temp.path().join("api")],
        )
        .unwrap();
        binding.members.push(adapter);
        let route = PortableRouteBinding::new("cpu", std::env::consts::OS, vec![binding]).unwrap();
        let mut settings = settings(data);
        settings.config_dir = temp.path().join("worker-config");
        let catalog = settings.config_dir.join("manifests/user.loras.jsonc");
        std::fs::create_dir_all(catalog.parent().unwrap()).unwrap();
        std::fs::write(
            &catalog,
            serde_json::to_vec(&json!({"loras":[{
                "id":"adapter-id", "installedPath":worker_adapter
            }]}))
            .unwrap(),
        )
        .unwrap();
        let payload = json!({"model":"m", "modelManifestEntry":{
            "id":"m", "downloads":[], "paths":{"model":model}},
            "loras":[{"id":"adapter-id", "installedPath":api_adapter}]
        })
        .as_object()
        .unwrap()
        .clone();
        let guard = RuntimeSourceGuard::begin_with_binding(
            &JobType::ImageGenerate,
            &payload,
            &settings,
            Some(&route),
        )
        .unwrap();
        let dispatched = guard.dispatch_payload()["loras"][0]["installedPath"]
            .as_str()
            .unwrap();
        assert_eq!(std::fs::read(dispatched).unwrap(), b"same adapter");
        assert_ne!(dispatched, api_adapter.to_str().unwrap());
        assert_ne!(dispatched, worker_adapter.to_str().unwrap());
    }

    #[test]
    fn raw_model_convert_uses_private_frozen_revision_without_receipt() {
        use sceneworks_core::model_artifacts::ArtifactIdentity;
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let repo = "guardfx/raw-convert";
        std::fs::create_dir_all(data.join("models")).unwrap();
        seed_snapshot(&library, repo, REV_A, "native.safetensors");
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Revision {
                identity: ArtifactIdentity::pinned(repo, REV_A, "default").unwrap(),
            },
            tier: None,
            files: vec![PortableArtifactFile {
                relative_path: "native.safetensors".into(),
                size_bytes: 7,
                sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
            }],
        };
        let route = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntries/0",
                "model_convert:guardfx/raw-convert",
                "cpu",
                "default",
                None,
                canonical_authored_options_sha256(&json!({})).unwrap(),
                vec![member],
            )
            .unwrap()],
        )
        .unwrap();
        let payload = json!({
            "modelId":"raw-convert", "sourceRepo":repo, "sourceFile":"native.safetensors",
            "converter":"supra2_img",
            "modelManifestEntries":[{"id":"model_convert:guardfx/raw-convert","downloads":[{"provider":"huggingface","repo":repo}]}]
        }).as_object().unwrap().clone();
        with_library(&library, || {
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ModelConvert,
                &payload,
                &settings(data.clone()),
                Some(&route),
            )
            .unwrap();
            let snapshot = sceneworks_core::hf_home::model_source_library(&data)
                .discover_snapshot(repo, None)
                .unwrap()
                .1;
            assert!(snapshot.join("native.safetensors").is_file());
            assert!(
                guard.dispatch_payload()["advanced"]["__portableFrozenSources"].as_bool()
                    == Some(true)
            );
            drop(guard);
            let source = library
                .join("models--guardfx--raw-convert/snapshots")
                .join(REV_A)
                .join("native.safetensors");
            std::fs::write(&source, b"changed").unwrap();
            let second = temp.path().join("second-data");
            std::fs::create_dir_all(second.join("models")).unwrap();
            assert!(RuntimeSourceGuard::begin_with_binding(
                &JobType::ModelConvert,
                &payload,
                &settings(second.clone()),
                Some(&route)
            )
            .is_err());
            std::fs::remove_file(&source).unwrap();
            assert!(RuntimeSourceGuard::begin_with_binding(
                &JobType::ModelConvert,
                &payload,
                &settings(second.clone()),
                Some(&route)
            )
            .is_err());
            let decision = crate::utility_dispatch::begin_source_guard_before_fetch(
                &JobType::ModelConvert,
                &payload,
                &settings(second.clone()),
                Some(&route),
            )
            .unwrap();
            let crate::utility_dispatch::SourceGuardBeforeFetch::Missing(missing) = decision else {
                panic!("cold frozen revision should request its exact missing file")
            };
            assert_eq!(
                missing[&(repo.to_owned(), REV_A.to_owned())],
                vec!["native.safetensors".to_owned()]
            );
            std::fs::remove_dir(source.parent().unwrap()).unwrap();
            let decision = crate::utility_dispatch::begin_source_guard_before_fetch(
                &JobType::ModelConvert,
                &payload,
                &settings(second),
                Some(&route),
            )
            .unwrap();
            assert!(matches!(
                decision,
                crate::utility_dispatch::SourceGuardBeforeFetch::Missing(_)
            ));
        });
    }

    #[test]
    fn empty_manifest_selector_binds_entire_pinned_snapshot() {
        use sceneworks_core::model_artifacts::ArtifactIdentity;

        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let repo = "guardfx/whole-snapshot";
        std::fs::create_dir_all(data.join("models")).unwrap();
        for file in ["model.safetensors", "tokenizer/config.json"] {
            seed_snapshot(&library, repo, REV_A, file);
        }
        let files = ["model.safetensors", "tokenizer/config.json"]
            .into_iter()
            .map(|relative_path| PortableArtifactFile {
                relative_path: relative_path.into(),
                size_bytes: 7,
                sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
            })
            .collect();
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Revision {
                identity: ArtifactIdentity::pinned(repo, REV_A, "default").unwrap(),
            },
            tier: Some("default".to_owned()),
            files,
        };
        let route = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                "whole-snapshot",
                "cpu",
                "default",
                Some("default".to_owned()),
                canonical_authored_options_sha256(&json!({})).unwrap(),
                vec![member],
            )
            .unwrap()],
        )
        .unwrap();
        let payload = json!({"model":"whole-snapshot","modelManifestEntry":{
            "id":"whole-snapshot", "downloads":[{"provider":"huggingface",
                "repo":repo,"revision":REV_A,"variant":"default","default":true,"files":[]}]
        }})
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings(data.clone()),
                Some(&route),
            )
            .unwrap();
            seed_snapshot(&library, repo, REV_A, "unbound-extra.bin");
            let error = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings(data.clone()),
                Some(&route),
            )
            .unwrap_err();
            assert!(error.to_string().contains("file set differs"), "{error}");
        });
    }

    /// A running utility conversion may overlay only its own files from a shared revision.
    /// Another conversion must verify its different wildcard closure from the complete source
    /// before it queues for the first job's portable overlay.
    #[test]
    fn concurrent_portable_revision_verifies_before_waiting_on_partial_overlay() {
        use sceneworks_core::model_artifacts::ArtifactIdentity;

        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let repo = "guardfx/shared-conversion";
        let first_file = "base/model.safetensors";
        let second_file = "turbo/model.safetensors";
        std::fs::create_dir_all(data.join("models")).unwrap();
        seed_snapshot(&library, repo, REV_A, first_file);
        seed_snapshot(&library, repo, REV_A, second_file);
        let partial = temp.path().join("first-private-snapshot");
        std::fs::create_dir_all(partial.join("base")).unwrap();
        std::fs::write(partial.join(first_file), b"weights").unwrap();
        let second_member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Revision {
                identity: ArtifactIdentity::pinned(repo, REV_A, "default").unwrap(),
            },
            tier: Some("default".to_owned()),
            files: vec![PortableArtifactFile {
                relative_path: second_file.into(),
                size_bytes: 7,
                sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
            }],
        };
        let route = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                "turbo",
                "cpu",
                "default",
                Some("default".to_owned()),
                canonical_authored_options_sha256(&json!({})).unwrap(),
                vec![second_member],
            )
            .unwrap()],
        )
        .unwrap();
        let payload = json!({"model":"turbo","modelManifestEntry":{
            "id":"turbo","downloads":[{"provider":"huggingface","repo":repo,
                "revision":REV_A,"variant":"default","default":true,"files":["turbo/*"]}]
        }})
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            let first = prefer_portable_snapshots(vec![LocalArtifactOverlayEntry {
                repository: repo.to_owned(),
                revision: REV_A.to_owned(),
                snapshot_root: partial.clone(),
                files: vec![first_file.into()],
            }])
            .unwrap();
            assert!(!partial.join(second_file).exists());
            let worker_settings = settings(data.clone());
            verify_portable_route(
                &route,
                &JobType::ImageGenerate,
                &payload,
                &worker_settings,
                &[],
                &[],
                None,
            )
            .expect("verification must use the complete source while a partial overlay is active");

            let (result_tx, result_rx) = std::sync::mpsc::channel();
            let waiter = std::thread::spawn(move || {
                result_tx
                    .send(RuntimeSourceGuard::begin_with_binding_cancelable(
                        &JobType::ImageGenerate,
                        &payload,
                        &worker_settings,
                        Some(&route),
                        Some(&PortableOverlayWaitCancellation::default()),
                    ))
                    .unwrap();
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(result_rx.try_recv().is_err());
            drop(first);
            let second = result_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap()
                .unwrap();
            drop(second);
            waiter.join().unwrap();
        });
    }

    #[test]
    fn cold_wildcard_manifest_matches_nested_frozen_files_without_receipt() {
        use sceneworks_core::model_artifacts::ArtifactIdentity;
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let repo = "guardfx/nested-glob";
        std::fs::create_dir_all(data.join("models")).unwrap();
        for file in [
            "q4/transformer/model.safetensors",
            "q4/text_encoder/model.safetensors",
        ] {
            seed_snapshot(&library, repo, REV_A, file);
        }
        #[cfg(unix)]
        {
            let unrelated = library
                .join("models--guardfx--nested-glob/snapshots")
                .join(REV_A)
                .join("q8");
            std::fs::create_dir_all(&unrelated).unwrap();
            std::os::unix::fs::symlink("absent.safetensors", unrelated.join("broken.safetensors"))
                .unwrap();
        }
        let files = [
            "q4/transformer/model.safetensors",
            "q4/text_encoder/model.safetensors",
        ]
        .into_iter()
        .map(|relative_path| PortableArtifactFile {
            relative_path: relative_path.into(),
            size_bytes: 7,
            sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
        })
        .collect();
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Revision {
                identity: ArtifactIdentity::pinned(repo, REV_A, "q4").unwrap(),
            },
            tier: Some("q4".to_owned()),
            files,
        };
        let route = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                "nested-glob",
                "cpu",
                "q4",
                Some("q4".to_owned()),
                canonical_authored_options_sha256(&json!({})).unwrap(),
                vec![member],
            )
            .unwrap()],
        )
        .unwrap();
        let payload = json!({"model":"nested-glob","modelManifestEntry":{
            "id":"nested-glob","downloads":[{"provider":"huggingface","repo":repo,
                "revision":REV_A,"variant":"q4","default":true,"files":["q4/*"]}]
        }})
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings(data.clone()),
                Some(&route),
            )
            .unwrap();
            let mapped = guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                .as_str()
                .unwrap();
            assert!(Path::new(mapped)
                .join("transformer/model.safetensors")
                .is_file());
            drop(guard);
            seed_snapshot(&library, repo, REV_A, "q4/extra/unbound.safetensors");
            assert!(RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings(data.clone()),
                Some(&route)
            )
            .is_err());
        });
    }

    #[test]
    fn portable_binding_refuses_extra_file_selected_by_manifest_glob() {
        use sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile;
        use sceneworks_core::model_artifacts::ArtifactIdentity;

        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/glob", REV_A, "q4/a.safetensors");
        seed_snapshot(&library, "guardfx/glob", REV_A, "q4/b.safetensors");
        write_receipts(
            &data,
            "guardfx/glob",
            json!([{
                "repo":"guardfx/glob", "modelId":"m", "variant":"q4",
                "resolvedFiles":["q4/a.safetensors"], "snapshotRevision":REV_A
            }]),
        );
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Revision {
                identity: ArtifactIdentity::pinned("guardfx/glob", REV_A, "q4").unwrap(),
            },
            tier: Some("q4".to_owned()),
            files: vec![PortableArtifactFile {
                relative_path: "q4/a.safetensors".into(),
                size_bytes: 7,
                sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
            }],
        };
        let route = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                "m",
                "cpu",
                "q4",
                Some("q4".to_owned()),
                canonical_authored_options_sha256(&json!({})).unwrap(),
                vec![member],
            )
            .unwrap()],
        )
        .unwrap();
        let payload = json!({"model":"m", "variant":"q4", "modelManifestEntry":{
            "id":"m", "downloads":[{"provider":"huggingface", "repo":"guardfx/glob",
                "revision":REV_A, "variant":"q4", "files":["q4/*"]}]
        }})
        .as_object()
        .unwrap()
        .clone();
        let partial = data.join("partial-snapshot");
        std::fs::create_dir_all(partial.join("q4")).unwrap();
        std::fs::write(partial.join("q4/a.safetensors"), b"weights").unwrap();
        let _serialized = overlay_guard();
        with_library(&library, || {
            let scope =
                sceneworks_core::model_artifacts::local_preference::prefer_local_snapshots(vec![
                    LocalArtifactOverlayEntry {
                        repository: "guardfx/glob".to_owned(),
                        revision: REV_A.to_owned(),
                        snapshot_root: partial,
                        files: vec!["q4/a.safetensors".into()],
                    },
                ])
                .unwrap();
            let selected = selected_model_artifact_closure(
                &payload["modelManifestEntry"],
                std::env::consts::OS,
                Some("q4"),
            );
            let error =
                verify_manifest_globs(&route.models[0], &selected, &settings(data.clone()), false)
                    .unwrap_err();
            assert!(error.to_string().contains("file set differs"), "{error}");
            drop(scope);
            let error = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings(data.clone()),
                Some(&route),
            )
            .unwrap_err();
            assert!(error.to_string().contains("file set differs"), "{error}");
            std::fs::remove_file(
                library
                    .join("models--guardfx--glob/snapshots")
                    .join(REV_A)
                    .join("q4/b.safetensors"),
            )
            .unwrap();
            let mut overridden = payload.clone();
            overridden.insert(
                "advanced".to_owned(),
                json!({"modelPath":"/unverified/alternate.safetensors"}),
            );
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &overridden,
                &settings(data),
                Some(&route),
            )
            .expect("a publisher path hint rebinds to the frozen private snapshot");
            let rebound = guard.dispatch_payload()["advanced"]["modelPath"]
                .as_str()
                .unwrap();
            assert_ne!(rebound, "/unverified/alternate.safetensors");
            assert_eq!(
                guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"],
                rebound
            );
            assert_eq!(
                std::fs::read(Path::new(rebound).join("a.safetensors")).unwrap(),
                b"weights"
            );
            drop(guard);
            let mut remote_hint = payload.clone();
            remote_hint.get_mut("modelManifestEntry").unwrap()["paths"] =
                json!({"model":"/api-host/models/q4"});
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &remote_hint,
                &settings(temp.path().join("data")),
                Some(&route),
            )
            .unwrap();
            let selected = guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                .as_str()
                .unwrap();
            assert!(
                std::fs::canonicalize(Path::new(selected).join("a.safetensors"))
                    .unwrap()
                    .starts_with(std::fs::canonicalize(&library).unwrap())
            );
            assert_ne!(selected, "/api-host/models/q4");
            assert_eq!(
                std::fs::read(Path::new(selected).join("a.safetensors")).unwrap(),
                b"weights"
            );
            std::fs::write(
                library
                    .join("models--guardfx--glob/snapshots")
                    .join(REV_A)
                    .join("q4/a.safetensors"),
                b"changed",
            )
            .unwrap();
            drop(guard);
            let error = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &remote_hint,
                &settings(temp.path().join("data")),
                Some(&route),
            )
            .expect_err("changed HF bytes must be refused");
            assert!(
                error
                    .to_string()
                    .contains("bytes differ from admitted identity"),
                "{error}"
            );
        });
    }

    /// Pin the configured source library to an explicit external dir for the whole body.
    /// `huggingface_hub_cache_dir` reads `HF_HUB_CACHE` before `data_dir`, and other tests in
    /// this crate mutate those vars — the shared `test_env` lock serializes against them, and the
    /// explicit dir models the external-drive scenario directly.
    fn with_library<T>(library: &Path, body: impl FnOnce() -> T) -> T {
        crate::test_env::temp_env_vars(
            &[("HF_HUB_CACHE", library.to_str().expect("utf-8 temp path"))],
            body,
        )
    }

    fn seed_snapshot(library: &Path, repo: &str, revision: &str, file: &str) {
        let safe = sceneworks_core::hf_home::safe_repo_dir_name(repo).unwrap();
        let snapshot = library
            .join(format!("models--{safe}"))
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        if let Some(parent) = snapshot.join(file).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(snapshot.join(file), b"weights").unwrap();
    }

    fn write_receipts(data_dir: &Path, repo: &str, receipts: Value) {
        let managed = data_dir
            .join("models")
            .join(sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(repo));
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({ "receipts": receipts })).unwrap(),
        )
        .unwrap();
    }

    /// One installed single-variant model: manifest entry + receipt + library snapshot.
    fn installed_model(temp: &TempDir) -> (Settings, PathBuf, JsonObject) {
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/model", REV_A, "model.safetensors");
        write_receipts(
            &data,
            "guardfx/model",
            json!([{ "repo": "guardfx/model", "modelId": "m",
                     "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
        );
        let payload = json!({
            "model": "m",
            "modelManifestEntry": {
                "id": "m",
                "downloads": [{ "provider": "huggingface", "repo": "guardfx/model",
                                "revision": REV_A, "files": ["model.safetensors"] }]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        (settings(data), library, payload)
    }

    fn portable_installed_model_binding() -> PortableRouteBinding {
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Revision {
                identity: ArtifactIdentity::pinned("guardfx/model", REV_A, "default").unwrap(),
            },
            tier: None,
            files: vec![PortableArtifactFile {
                relative_path: "model.safetensors".into(),
                size_bytes: 7,
                sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
            }],
        };
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            "m",
            "cpu",
            "default",
            None,
            canonical_authored_options_sha256(&json!({})).unwrap(),
            vec![member],
        )
        .unwrap();
        PortableRouteBinding::new("cpu", std::env::consts::OS, vec![model]).unwrap()
    }

    /// Older than the verified-source stamp's racy window, so a recorded verdict may stand in
    /// for a hash.
    fn settle_mtime(path: &Path) {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(10))
            .unwrap();
    }

    fn in_place_hashed_files() -> usize {
        HASHED_SOURCE_FILES.with(std::cell::Cell::get)
    }

    #[test]
    fn a_pinned_revision_is_verified_in_place_and_hashed_once_per_unchanged_blob() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        let binding = portable_installed_model_binding();
        let source = library
            .join("models--guardfx--model/snapshots")
            .join(REV_A)
            .join("model.safetensors");
        settle_mtime(&source);
        let begin = || {
            RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings,
                Some(&binding),
            )
        };
        with_library(&library, || {
            let start = in_place_hashed_files();
            let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let seen = events.clone();
            let observer =
                gen_core::LoadActivityObserver::new(move |event| seen.lock().unwrap().push(event));
            let guard = begin().unwrap();
            drop(observer);
            let loaded = guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                .as_str()
                .unwrap();
            assert_eq!(
                std::fs::canonicalize(Path::new(loaded).join("model.safetensors")).unwrap(),
                std::fs::canonicalize(&source).unwrap(),
                "the loader reads the worker's own hub snapshot"
            );
            let attempts = settings.data_dir.join("models/portable-attempts");
            assert!(
                std::fs::read_dir(&attempts)
                    .unwrap()
                    .flatten()
                    .all(|entry| !entry.path().join("bundle").exists()),
                "a Revision-only route copies nothing"
            );
            let verifying = events
                .lock()
                .unwrap()
                .iter()
                .flatten()
                .rfind(|activity| activity.phase == gen_core::LoadActivityPhase::Verifying)
                .cloned()
                .expect("in-place verification reports progress");
            assert_eq!(verifying.component, IN_PLACE_VERIFICATION_COMPONENT);
            assert_eq!(verifying.component_total_bytes, Some(7));
            assert_eq!(verifying.component_bytes_read, 7);
            assert_eq!(in_place_hashed_files(), start + 1);
            drop(guard);

            let guard = begin().unwrap();
            assert_eq!(
                in_place_hashed_files(),
                start + 1,
                "an unchanged blob is not hashed again"
            );
            drop(guard);

            // Same size, different bytes: the stamp changes, so the blob is hashed and refused.
            std::fs::write(&source, b"WEIGHTS").unwrap();
            settle_mtime(&source);
            let error = begin().expect_err("changed bytes must be refused");
            assert!(
                error
                    .to_string()
                    .contains("bytes differ from admitted identity"),
                "{error}"
            );
            // Rewriting the pinned bytes is a new file identity: hashed again, then accepted.
            std::fs::write(&source, b"weights").unwrap();
            settle_mtime(&source);
            drop(begin().unwrap());
            assert_eq!(in_place_hashed_files(), start + 2);
        });
    }

    fn tiered_route(repo: &str, tier: &str) -> PortableRouteBinding {
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Revision {
                identity: ArtifactIdentity::pinned(repo, REV_A, tier).unwrap(),
            },
            tier: Some(tier.to_owned()),
            files: vec![PortableArtifactFile {
                relative_path: format!("{tier}/a.safetensors").into(),
                size_bytes: 7,
                sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
            }],
        };
        PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                "m",
                "cpu",
                tier,
                Some(tier.to_owned()),
                canonical_authored_options_sha256(&json!({})).unwrap(),
                vec![member],
            )
            .unwrap()],
        )
        .unwrap()
    }

    /// Codex review of PR #3379: the API freezes an untiered standard-tier image request's
    /// implicit tier into the binding (not the catalog's default q4 download); the worker must
    /// verify the request against that bound tier and load exactly its files, even when other
    /// tiers are installed beside it.
    #[test]
    fn an_untiered_standard_tier_image_runs_its_bound_tier_through_verification_and_loading() {
        let repo = "guardfx/tiers";
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let tiers = ["q4", "q8", "bf16"];
        for tier in tiers {
            seed_snapshot(&library, repo, REV_A, &format!("{tier}/a.safetensors"));
        }
        write_receipts(
            &data,
            repo,
            json!(tiers
                .iter()
                .map(|tier| json!({
                    "repo": repo, "modelId": "m", "variant": tier,
                    "resolvedFiles": [format!("{tier}/a.safetensors")],
                    "snapshotRevision": REV_A
                }))
                .collect::<Vec<_>>()),
        );
        let payload = json!({"model": "m", "modelManifestEntry": {
            "id": "m",
            "mlx": {"standardTierLayout": true},
            "downloads": tiers.iter().map(|tier| json!({
                "provider": "huggingface", "repo": repo, "revision": REV_A,
                "variant": tier, "files": [format!("{tier}/a.safetensors")]
            })).collect::<Vec<_>>()
        }})
        .as_object()
        .unwrap()
        .clone();
        let snapshot = library.join("models--guardfx--tiers/snapshots").join(REV_A);
        let _serialized = overlay_guard();
        with_library(&library, || {
            for job_type in [JobType::ImageGenerate, JobType::ImageEdit] {
                for bound in ["q8", "bf16"] {
                    let route = tiered_route(repo, bound);
                    let guard = RuntimeSourceGuard::begin_with_binding(
                        &job_type,
                        &payload,
                        &settings(data.clone()),
                        Some(&route),
                    )
                    .unwrap_or_else(|error| {
                        panic!("{job_type:?} untiered request bound to {bound}: {error}")
                    });
                    let dispatch = guard.dispatch_payload();
                    assert_eq!(dispatch["advanced"]["__portableBoundTier"], bound);
                    let model = Path::new(
                        dispatch["modelManifestEntry"]["paths"]["model"]
                            .as_str()
                            .unwrap(),
                    );
                    assert_eq!(model.file_name().unwrap(), bound);
                    assert_eq!(
                        std::fs::canonicalize(model.join("a.safetensors")).unwrap(),
                        std::fs::canonicalize(snapshot.join(bound).join("a.safetensors")).unwrap()
                    );
                    // The loader is served the hub snapshot itself, where every tier is installed;
                    // the bound tier is what tier resolution selects among them.
                    let root = model.parent().unwrap();
                    assert_eq!(
                        std::fs::canonicalize(root).unwrap(),
                        std::fs::canonicalize(&snapshot).unwrap()
                    );
                    for other in tiers.iter().filter(|tier| **tier != bound) {
                        assert!(root.join(other).exists(), "{other} is installed beside it");
                    }
                    assert_eq!(
                        sceneworks_core::image_request::ImageRequest::from_payload(dispatch)
                            .quant_bits(),
                        Some(if bound == "bf16" { 0 } else { 8 }),
                        "every bits-driven tier resolver reads the bound tier"
                    );
                    #[cfg(native_lane)]
                    assert_eq!(
                        crate::image_jobs::standard_tier_subdir_for_test(
                            root,
                            &sceneworks_core::image_request::ImageRequest::from_payload(dispatch),
                        ),
                        root.join(bound),
                        "the engine's implicit tier resolution lands on the bound tier"
                    );
                }
            }
        });
    }

    /// Codex review of PR #3379 (4558615108): a resolved-cache snapshot can hold a model's hard
    /// closure while a soft companion of the same revision exists only in the source library. Both
    /// are reached through the revision's one snapshot directory, so that route falls back to the
    /// copy-then-verify bundle: one directory holding each file from its own verified root.
    #[test]
    fn one_revision_split_across_two_roots_is_served_from_a_verified_copy() {
        let repo = "guardfx/split";
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        for file in [
            "model.safetensors",
            "companion.safetensors",
            "other.safetensors",
        ] {
            seed_snapshot(&library, repo, REV_A, file);
        }
        let resolved = data.join("models/resolved-test/hard-closure");
        std::fs::create_dir_all(&resolved).unwrap();
        std::fs::write(resolved.join("model.safetensors"), b"weights").unwrap();
        let file = |name: &str| PortableArtifactFile {
            relative_path: name.into(),
            size_bytes: 7,
            sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
        };
        let identity = ArtifactIdentity::pinned(repo, REV_A, "default").unwrap();
        let members = vec![
            PortableArtifactMember {
                role: ArtifactMemberRole::Primary,
                component_id: None,
                source: PortableSourceIdentity::Revision {
                    identity: identity.clone(),
                },
                tier: None,
                files: vec![file("model.safetensors")],
            },
            PortableArtifactMember {
                role: ArtifactMemberRole::CoRequisite,
                component_id: Some("modelManifestEntry/downloads/1".to_owned()),
                source: PortableSourceIdentity::Revision { identity },
                tier: None,
                files: vec![file("companion.safetensors")],
            },
        ];
        let route = PortableRouteBinding::new(
            "cpu",
            std::env::consts::OS,
            vec![PortableModelBinding::new(
                "modelManifestEntry",
                "m",
                "cpu",
                "default",
                None,
                canonical_authored_options_sha256(&json!({})).unwrap(),
                members,
            )
            .unwrap()],
        )
        .unwrap();
        let payload = json!({"model": "m", "modelManifestEntry": {"id": "m", "downloads": []}})
            .as_object()
            .unwrap()
            .clone();
        let _serialized = overlay_guard();
        with_library(&library, || {
            let _resolved_cache =
                sceneworks_core::model_artifacts::local_preference::prefer_local_snapshots(vec![
                    LocalArtifactOverlayEntry {
                        repository: repo.to_owned(),
                        revision: REV_A.to_owned(),
                        snapshot_root: resolved.clone(),
                        files: vec!["model.safetensors".into()],
                    },
                ])
                .unwrap();
            let mut dispatch = payload.clone();
            let prepared = prepare_portable_snapshot(
                &route,
                &JobType::ImageGenerate,
                &payload,
                &settings(data.clone()),
                &mut dispatch,
                None,
            )
            .expect("both members pass ledger verification in their own roots")
            .unwrap();
            assert_eq!(prepared.overlays.len(), 1);
            let served = &prepared.overlays[0].snapshot_root;
            assert_eq!(
                prepared.overlays[0].files,
                vec![
                    PathBuf::from("companion.safetensors"),
                    PathBuf::from("model.safetensors")
                ]
            );
            assert!(served.starts_with(data.join("models/portable-attempts")));
            for name in ["model.safetensors", "companion.safetensors"] {
                let file = served.join(name);
                assert!(
                    std::fs::symlink_metadata(&file).unwrap().is_file(),
                    "{name}"
                );
                assert_eq!(std::fs::read(file).unwrap(), b"weights");
            }
            assert!(
                !served.join("other.safetensors").exists(),
                "an unfrozen file is never copied"
            );
        });
    }

    /// A route left by the dropped link-view layout (no current layout marker, a `views/` tree of
    /// links into the hub) is reclaimed by the existing retention prune, which removes the links
    /// and never what they point at.
    #[cfg(unix)]
    #[test]
    fn retention_reclaims_a_legacy_view_route_and_never_its_hub_files() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let hub = temp
            .path()
            .join("hub/models--guardfx--model/snapshots")
            .join(REV_A);
        std::fs::create_dir_all(&hub).unwrap();
        std::fs::write(hub.join("model.safetensors"), b"weights").unwrap();
        let legacy = data
            .join("models/portable-attempts")
            .join("a".repeat(64))
            .join("views/0/hf/models--guardfx--model/snapshots")
            .join(REV_A);
        std::fs::create_dir_all(&legacy).unwrap();
        std::os::unix::fs::symlink(
            hub.join("model.safetensors"),
            legacy.join("model.safetensors"),
        )
        .unwrap();
        let route = data.join("models/portable-attempts").join("a".repeat(64));
        std::fs::write(route.join(".layout-v3"), b"").unwrap();
        prune_portable_snapshots(None, &settings(data)).unwrap();
        assert!(!route.join("views").exists());
        assert_eq!(
            std::fs::read(hub.join("model.safetensors")).unwrap(),
            b"weights"
        );
    }

    /// The path the guard hands a loader for a pinned Hugging Face member passes inference's own
    /// shard-path check (`snapshot_shard_roots` + `resolve_safetensors_shards`): the hub layout's
    /// snapshot links into the repository's `blobs/` are authorized, so an in-place member never
    /// fails at load. No model compute.
    #[cfg(unix)]
    #[test]
    fn a_pinned_member_served_in_place_passes_the_inference_shard_check() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        let repository = library.join("models--guardfx--model");
        let snapshot = repository.join("snapshots").join(REV_A);
        std::fs::create_dir_all(repository.join("blobs")).unwrap();
        std::fs::rename(
            snapshot.join("model.safetensors"),
            repository.join("blobs/weights-blob"),
        )
        .unwrap();
        std::os::unix::fs::symlink(
            "../../blobs/weights-blob",
            snapshot.join("model.safetensors"),
        )
        .unwrap();
        let binding = portable_installed_model_binding();
        with_library(&library, || {
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings,
                Some(&binding),
            )
            .unwrap();
            let dir = Path::new(
                guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                    .as_str()
                    .unwrap(),
            );
            let roots = gen_core::safetensors_shards::snapshot_shard_roots(dir).unwrap();
            let shards =
                gen_core::safetensors_shards::resolve_safetensors_shards(dir, "model", &roots)
                    .expect("the inference shard check accepts the served member");
            assert_eq!(
                shards,
                vec![std::fs::canonicalize(repository.join("blobs/weights-blob")).unwrap()]
            );
        });
    }

    /// Codex review of PR #3379: two in-process jobs on one pinned revision overlap. The first
    /// job's process-wide overlay hands the second job's discovery the first job's view; the
    /// second job resolves those entries through the view's ledger to their own sources.
    #[test]
    fn two_overlapping_guards_on_one_revision_both_bind() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        let binding = portable_installed_model_binding();
        let source = library
            .join("models--guardfx--model/snapshots")
            .join(REV_A)
            .join("model.safetensors");
        let _serialized = overlay_guard();
        with_library(&library, || {
            let begin = || {
                RuntimeSourceGuard::begin_with_binding(
                    &JobType::ImageGenerate,
                    &payload,
                    &settings,
                    Some(&binding),
                )
            };
            let first = begin().expect("the first job binds");
            let second = begin().expect("an overlapping job on the same revision binds");
            for guard in [&first, &second] {
                let loaded = Path::new(
                    guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                        .as_str()
                        .unwrap(),
                )
                .join("model.safetensors");
                assert_eq!(
                    std::fs::canonicalize(&loaded).unwrap(),
                    std::fs::canonicalize(&source).unwrap()
                );
            }
            drop(first);
            assert_eq!(
                std::fs::read(
                    Path::new(
                        second.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                            .as_str()
                            .unwrap()
                    )
                    .join("model.safetensors")
                )
                .unwrap(),
                b"weights"
            );
        });
    }

    fn in_place_ledger(path: &str, size_bytes: u64) -> PortableArtifactFile {
        PortableArtifactFile {
            relative_path: path.into(),
            size_bytes,
            sha256: format!("sha256:{:x}", Sha256::digest(b"weights")),
        }
    }

    #[test]
    fn in_place_revision_refuses_an_escaping_missing_or_resized_member() {
        let temp = TempDir::new().unwrap();
        let snapshot = temp
            .path()
            .join("hub/models--guardfx--model/snapshots")
            .join(REV_A);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();
        std::fs::write(temp.path().join("hub/outside.safetensors"), b"weights").unwrap();
        assert!(
            in_place_revision_files(&snapshot, &[in_place_ledger("model.safetensors", 7)]).is_ok()
        );
        for (path, size, refusal) in [
            ("../../../outside.safetensors", 7, "escapes its snapshot"),
            ("./model.safetensors", 7, "escapes its snapshot"),
            ("", 7, "escapes its snapshot"),
            ("absent.safetensors", 7, "missing from its snapshot"),
            ("model.safetensors", 8, "differs from its pinned ledger"),
        ] {
            let error = in_place_revision_files(&snapshot, &[in_place_ledger(path, size)])
                .err()
                .unwrap_or_else(|| panic!("{path:?} must be refused"));
            assert!(error.to_string().contains(refusal), "{path:?}: {error}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn in_place_revision_follows_its_own_blob_links_and_refuses_links_outside_its_repository() {
        let temp = TempDir::new().unwrap();
        let repository = temp.path().join("hub/models--guardfx--model");
        let snapshot = repository.join("snapshots").join(REV_A);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(repository.join("blobs")).unwrap();
        std::fs::write(repository.join("blobs/own"), b"weights").unwrap();
        std::os::unix::fs::symlink("../../blobs/own", snapshot.join("model.safetensors")).unwrap();
        let sibling = temp.path().join("hub/models--guardfx--other/blobs");
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(sibling.join("theirs"), b"weights").unwrap();
        std::os::unix::fs::symlink(sibling.join("theirs"), snapshot.join("sibling.safetensors"))
            .unwrap();
        let outside = temp.path().join("outside.safetensors");
        std::fs::write(&outside, b"weights").unwrap();
        std::os::unix::fs::symlink(&outside, snapshot.join("leak.safetensors")).unwrap();

        let own =
            in_place_revision_files(&snapshot, &[in_place_ledger("model.safetensors", 7)]).unwrap();
        assert_eq!(
            own[0].real,
            std::fs::canonicalize(repository.join("blobs/own")).unwrap()
        );
        for path in ["sibling.safetensors", "leak.safetensors"] {
            let error = in_place_revision_files(&snapshot, &[in_place_ledger(path, 7)])
                .err()
                .unwrap_or_else(|| panic!("{path} must be refused"));
            assert!(
                error.to_string().contains("outside its hub repository"),
                "{path}: {error}"
            );
        }
    }

    #[test]
    fn utility_selection_uses_verified_worker_bytes_and_refuses_substitution() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        let selected = data.join("models/selected-clip");
        let substituted = data.join("models/other-clip");
        std::fs::create_dir_all(&selected).unwrap();
        std::fs::create_dir_all(&substituted).unwrap();
        std::fs::write(selected.join("model.safetensors"), b"selected").unwrap();
        std::fs::write(substituted.join("model.safetensors"), b"other").unwrap();
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({"models":[{"id":"clip_vit_l14","modelPath":selected}]}))
                .unwrap(),
        )
        .unwrap();
        let files = crate::portable_source_claims::capture_registered_files(&selected).unwrap();
        let digest = portable_closure_sha256(&files).unwrap();
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest { sha256: digest },
            tier: Some("default".to_owned()),
            files,
        };
        let model = PortableModelBinding::new(
            "modelManifestEntries/0",
            "clip_vit_l14",
            "cpu",
            "default",
            Some("default".to_owned()),
            canonical_authored_options_sha256(&json!({})).unwrap(),
            vec![member],
        )
        .unwrap();
        let route = PortableRouteBinding::new("cpu", std::env::consts::OS, vec![model]).unwrap();
        let mut payload = json!({
            "kind":"dataset_analysis", "embedder":"clip_vit_l14",
            "modelNameOrPath":"/api-host/selected-clip",
            "modelManifestEntries":[{"id":"clip_vit_l14","downloads":[{
                "provider":"huggingface", "repo":"openai/clip-vit-large-patch14",
                "revision":REV_A,"files":["model.safetensors"]
            }]}]
        })
        .as_object()
        .unwrap()
        .clone();
        let guard = RuntimeSourceGuard::begin_with_binding(
            &JobType::DatasetAnalysis,
            &payload,
            &settings,
            Some(&route),
        )
        .unwrap();
        let loaded = guard.dispatch_payload()["modelNameOrPath"]
            .as_str()
            .unwrap();
        assert!(Path::new(loaded).starts_with(data.join("models/portable-attempts")));
        assert_eq!(
            std::fs::read(Path::new(loaded).join("model.safetensors")).unwrap(),
            b"selected"
        );

        payload.insert("modelNameOrPath".to_owned(), json!(substituted));
        assert!(RuntimeSourceGuard::begin_with_binding(
            &JobType::DatasetAnalysis,
            &payload,
            &settings,
            Some(&route),
        )
        .is_err());
    }

    #[test]
    fn caption_revision_rebinds_top_level_locator_and_refuses_changed_selection() {
        let temp = TempDir::new().unwrap();
        let (settings, library, mut payload) = installed_model(&temp);
        let entry = payload.remove("modelManifestEntry").unwrap();
        payload.insert("kind".to_owned(), json!("training_caption"));
        payload.insert("modelNameOrPath".to_owned(), json!("guardfx/model"));
        payload.insert("modelManifestEntries".to_owned(), json!([entry]));
        let mut route = portable_installed_model_binding();
        route.models[0].slot = "modelManifestEntries/0".to_owned();
        with_library(&library, || {
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::TrainingCaption,
                &payload,
                &settings,
                Some(&route),
            )
            .unwrap();
            let loaded = guard.dispatch_payload()["modelNameOrPath"]
                .as_str()
                .unwrap();
            // The loader directory is the route's private view; its file is the verified HF file.
            assert!(
                std::fs::canonicalize(Path::new(loaded).join("model.safetensors"))
                    .unwrap()
                    .starts_with(std::fs::canonicalize(&library).unwrap())
            );
            assert_eq!(
                std::fs::read(Path::new(loaded).join("model.safetensors")).unwrap(),
                b"weights"
            );
            payload.insert("modelNameOrPath".to_owned(), json!("guardfx/other"));
            assert!(RuntimeSourceGuard::begin_with_binding(
                &JobType::TrainingCaption,
                &payload,
                &settings,
                Some(&route),
            )
            .is_err());
        });
    }

    #[test]
    fn cold_portable_revision_yields_worker_only_primary_handoff() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        let receipt = settings
            .data_dir
            .join("models")
            .join(
                sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(
                    "guardfx/model",
                ),
            )
            .join(".sceneworks-download-complete.json");
        std::fs::remove_file(receipt).unwrap();
        assert!(settings
            .verified_revision_source("m", "guardfx/model", REV_A)
            .is_none());
        with_library(&library, || {
            let guard = RuntimeSourceGuard::begin_with_binding(
                &JobType::ImageGenerate,
                &payload,
                &settings,
                Some(&portable_installed_model_binding()),
            )
            .expect("portable verification serves a cold source without an install receipt");
            let handoffs = guard.verified_revision_sources();
            assert_eq!(handoffs.len(), 1);
            assert_eq!(handoffs[0].model_id, "m");
            assert_eq!(handoffs[0].repository, "guardfx/model");
            assert_eq!(handoffs[0].revision, REV_A);
            assert_eq!(
                handoffs[0].path,
                PathBuf::from(
                    guard.dispatch_payload()["modelManifestEntry"]["paths"]["model"]
                        .as_str()
                        .unwrap()
                )
            );
            assert_eq!(
                std::fs::read(handoffs[0].path.join("model.safetensors")).unwrap(),
                b"weights"
            );
        });
    }

    // The resident-base rows this THREAD resolves, replacing the production table for the body of
    // a test. Thread-local, so it cannot leak into a concurrently running test; see
    // `super::resident_base_row` for why a test must not publish a bundle for a production
    // repository.
    thread_local! {
        static RESIDENT_BASE_ROWS: std::cell::RefCell<
            Vec<(String, String, super::ResidentBaseShape, Value)>,
        > = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// The override [`super::resident_base_row`] consults first.
    pub(super) fn resident_base_override(
        family: &str,
    ) -> Option<(String, super::ResidentBaseShape, Value)> {
        RESIDENT_BASE_ROWS.with(|rows| {
            rows.borrow()
                .iter()
                .find(|(candidate, ..)| candidate == family)
                .map(|(_, tier, shape, entry)| (tier.clone(), *shape, entry.clone()))
        })
    }

    /// Point one family at a synthetic base entry for the body, then restore. RAII so a panicking
    /// assertion cannot leave the override installed for the next test on this thread.
    struct ResidentBaseRowGuard;

    impl ResidentBaseRowGuard {
        fn install(
            family: &str,
            tier: &str,
            shape: super::ResidentBaseShape,
            entry: Value,
        ) -> Self {
            RESIDENT_BASE_ROWS.with(|rows| {
                rows.borrow_mut()
                    .push((family.to_owned(), tier.to_owned(), shape, entry))
            });
            Self
        }
    }

    impl Drop for ResidentBaseRowGuard {
        fn drop(&mut self) {
            RESIDENT_BASE_ROWS.with(|rows| rows.borrow_mut().clear());
        }
    }

    /// Serialize every test that installs a local-tier preference scope. The scope is PROCESS-WIDE
    /// by design (model paths resolve on engine threads far below the job task), so two of these
    /// bodies running concurrently would read each other's overlay.
    ///
    /// This mutex is the ONLY thing serializing them. `.cargo/config.toml` sets no
    /// `RUST_TEST_THREADS`, so this suite really does run in parallel and the interleaving is live
    /// rather than hypothetical — an earlier version of this comment claimed the harness already
    /// serialized the suite, which was false and would have made dropping the mutex look safe.
    fn overlay_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Enable the same policy field production reads; the original fixture remains disabled.
    fn with_local_cache<T>(
        settings: &Settings,
        library: &Path,
        body: impl FnOnce(Settings) -> T,
    ) -> T {
        let _serialized = overlay_guard();
        let mut settings = settings.clone();
        settings.resolved_cache.enabled = true;
        crate::test_env::temp_env_vars(
            &[
                ("HF_HUB_CACHE", library.to_str().expect("utf-8 temp path")),
                (
                    sceneworks_core::model_artifacts::resolved_cache::RESOLVED_CACHE_ENABLED_ENV,
                    "1",
                ),
            ],
            || body(settings),
        )
    }

    /// Publish a real bundle through the PRODUCTION materializer: the source snapshot is copied,
    /// verified and atomically published exactly as a promotion would do it, so these tests read
    /// back the same on-disk shape the runtime will meet.
    fn publish_bundle(
        data_dir: &Path,
        library: &Path,
        repository: &str,
        revision: &str,
        variant: &str,
        files: &[&str],
    ) -> sceneworks_core::model_artifacts::ResolvedModelArtifact {
        let destination =
            sceneworks_core::model_artifacts::local_preference::hub_cache_member_destination(
                repository,
                revision,
                Path::new(""),
            )
            .unwrap();
        publish_bundle_at(
            data_dir,
            library,
            repository,
            revision,
            variant,
            files,
            destination,
        )
    }

    /// `publish_bundle`, but with the bundle-relative member destination chosen by the caller — the
    /// one property that decides whether the published bundle is a shape the local tier can serve.
    fn publish_bundle_at(
        data_dir: &Path,
        library: &Path,
        repository: &str,
        revision: &str,
        variant: &str,
        files: &[&str],
        destination: PathBuf,
    ) -> sceneworks_core::model_artifacts::ResolvedModelArtifact {
        use sceneworks_core::model_artifacts::resolved_cache::{
            MaterializationCancellation, MaterializationOutcome, ResolvedCacheMaterializer,
        };
        use sceneworks_core::model_artifacts::{
            ArtifactAvailability, ArtifactFile, ArtifactIdentity, ArtifactMemberRole,
            ResolvedBundleClosure, ResolvedBundleMember,
        };

        let identity = ArtifactIdentity::pinned(repository, revision, variant).unwrap();
        let closure = ResolvedBundleClosure::new(vec![ResolvedBundleMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: identity.clone(),
            tier: Some(variant.to_owned()),
            source_subpath: PathBuf::new(),
            destination,
            files: files
                .iter()
                .map(|file| ArtifactFile::new(*file).unwrap())
                .collect(),
        }])
        .unwrap();
        let resolver = ModelArtifactResolver::new(ArtifactSourceLibrary::new(library).unwrap());
        let source = resolver
            .resolve_source(identity, closure, ArtifactAvailability::Available)
            .unwrap();
        let candidate = sceneworks_core::model_artifacts::PromotionCandidate {
            cache_key: source.cache_key().unwrap(),
            artifact: source,
        };
        let store = ResolvedCacheStore::open(data_dir).unwrap();
        let outcome = ResolvedCacheMaterializer::new(store)
            .materialize(
                &candidate,
                library,
                repository,
                &MaterializationCancellation::default(),
            )
            .unwrap();
        match outcome {
            MaterializationOutcome::Published(metadata) => metadata.artifact,
            other => panic!("bundle was not published: {other:?}"),
        }
    }

    /// The walking skeleton: an installed model with a published bundle admits from the app-owned
    /// tier, the SHARED snapshot resolver every image/video/audio/utility loader uses returns the
    /// bundle path, an evictor cannot take the entry while the job holds it, and finishing the job
    /// restores the source tier.
    #[test]
    fn a_published_bundle_serves_the_load_and_is_lease_protected() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let cache_key = artifact.cache_key().unwrap();
            let bundle_root = artifact.location.root().to_path_buf();

            // Before the guard runs, the shared resolver reads the configured source library.
            let source_snapshot =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("source snapshot");
            assert!(source_snapshot.starts_with(&library));

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );

            // The one shared snapshot resolver — every model-consuming runtime funnels through it
            // — now answers with the app-owned bundle.
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("leased local snapshot");
            assert_eq!(
                loaded,
                bundle_root.join(format!("models--guardfx--model/snapshots/{REV_A}"))
            );
            assert!(loaded.join("model.safetensors").is_file());
            // And so does the pinned-component resolver used by the utility/co-requisite lanes.
            assert_eq!(
                crate::model_jobs::huggingface_pinned_snapshot_dir(
                    &settings.data_dir,
                    "guardfx/model",
                    REV_A
                ),
                Some(loaded.clone())
            );

            // An evictor takes the entry's artifact lock exclusively; the live lease denies it.
            let evictor = ResolvedCacheStore::open(&settings.data_dir).unwrap();
            let candidate = sceneworks_core::model_artifacts::PromotionCandidate {
                cache_key: cache_key.clone(),
                artifact: ModelArtifactResolver::new(ArtifactSourceLibrary::new(&library).unwrap())
                    .resolve_source(
                        artifact.identity.clone(),
                        artifact.closure.clone(),
                        sceneworks_core::model_artifacts::ArtifactAvailability::Available,
                    )
                    .unwrap(),
            };
            assert!(matches!(
                evictor
                    .reserve(&candidate, &library, "guardfx/model")
                    .unwrap(),
                sceneworks_core::model_artifacts::resolved_cache::ReservationOutcome::Contended
            ));

            guard.finish_success().unwrap();
            // Usage was recorded for a real load, and the entry survives untouched.
            let metadata = evictor.lookup_complete(&cache_key).unwrap().unwrap();
            assert!(metadata.last_used_at.is_some());
            // With the job finished the source tier serves again.
            assert_eq!(
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model"),
                Some(source_snapshot)
            );
        });
    }

    /// A job that carries several models (a primary plus utility/co-requisite carriers) leases and
    /// serves EVERY one of them from the local tier — admission is per artifact, so one model can
    /// never silently drag another back onto the source tier.
    #[test]
    fn every_model_a_job_carries_is_served_from_its_own_leased_bundle() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        for repo in ["guardfx/primary", "guardfx/utility"] {
            seed_snapshot(&library, repo, REV_A, "model.safetensors");
            write_receipts(
                &data,
                repo,
                json!([{ "repo": repo, "modelId": repo,
                         "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
            );
        }
        let settings = settings(data);
        let entry = |repo: &str| {
            json!({
                "id": repo,
                "downloads": [{ "provider": "huggingface", "repo": repo, "revision": REV_A,
                                "files": ["model.safetensors"] }]
            })
        };
        let payload = json!({
            "modelManifestEntry": entry("guardfx/primary"),
            "modelManifestEntries": [entry("guardfx/utility")]
        })
        .as_object()
        .unwrap()
        .clone();
        with_local_cache(&settings, &library, |settings| {
            for repo in ["guardfx/primary", "guardfx/utility"] {
                publish_bundle(
                    &settings.data_dir,
                    &library,
                    repo,
                    REV_A,
                    "default",
                    &["model.safetensors"],
                );
            }
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(guard.resolutions().len(), 2);
            assert!(guard
                .resolutions()
                .iter()
                .all(|resolution| resolution.availability == ModelAvailability::LocalReady));
            for repo in ["guardfx/primary", "guardfx/utility"] {
                let loaded = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo)
                    .expect("leased local snapshot");
                assert!(loaded.join("model.safetensors").is_file());
                assert!(!loaded.starts_with(&library));
            }
            guard.finish_success().unwrap();
        });
    }

    /// The runtimes whose model directory is a concrete path resolved BEFORE the load — training
    /// base models, captioners, analyzers (the `ManagedModelPath` entrypoint) — and the
    /// receipt-provenance lane both read the leased bundle too, so no model-consuming surface is
    /// left on the source tier while a lease is active.
    #[test]
    fn payload_resolved_and_receipt_paths_are_served_from_the_leased_bundle() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let bundle_snapshot = artifact
                .location
                .root()
                .join(format!("models--guardfx--model/snapshots/{REV_A}"));
            let source_snapshot = ArtifactSourceLibrary::new(&library)
                .unwrap()
                .repository_root("guardfx/model")
                .unwrap()
                .join("snapshots")
                .join(REV_A);

            // Without a lease the authoritative source path survives untouched.
            let before = crate::paths::normalize_app_managed_model_path(
                &settings,
                source_snapshot.to_str().unwrap(),
                "base model",
            )
            .unwrap();
            assert!(before.ends_with(format!("snapshots/{REV_A}")));
            assert!(!before.starts_with(artifact.location.root()));

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            let during = crate::paths::normalize_app_managed_model_path(
                &settings,
                source_snapshot.to_str().unwrap(),
                "base model",
            )
            .unwrap();
            assert_eq!(during, bundle_snapshot);
            // The receipt lane proves its install against the source and then hands the loader the
            // leased local path.
            assert_eq!(
                crate::model_jobs::huggingface_receipt_weights_dir(
                    &settings.data_dir,
                    "guardfx/model",
                    Some("m"),
                    None
                ),
                Some(bundle_snapshot)
            );
            guard.finish_success().unwrap();

            // And the redirect is gone with the lease.
            assert_eq!(
                crate::paths::normalize_app_managed_model_path(
                    &settings,
                    source_snapshot.to_str().unwrap(),
                    "base model",
                )
                .unwrap(),
                before
            );
        });
    }

    /// The epic's headline outcome: with the configured library disconnected, a complete local
    /// bundle still admits and still resolves — without the runtime ever reaching a downloader or
    /// recreating the configured library root.
    #[test]
    fn a_disconnected_library_still_serves_a_complete_bundle_without_downloading() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            std::fs::rename(&library, temp.path().join("detached")).unwrap();

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("leased local snapshot while disconnected");
            assert!(loaded.join("model.safetensors").is_file());
            let resolved_root =
                std::fs::canonicalize(settings.data_dir.join("models").join("resolved")).unwrap();
            assert!(loaded.starts_with(&resolved_root), "{}", loaded.display());
            // Nothing recreated the configured library or a parallel download destination: the
            // preference path never reaches the downloader.
            assert!(!library.exists());
            guard.finish_success().unwrap();
        });
    }

    #[test]
    fn portable_prefetch_uses_complete_bundle_before_considering_disconnected_source() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            std::fs::rename(&library, temp.path().join("detached")).unwrap();
            let route = portable_installed_model_binding();
            assert!(
                !crate::utility_dispatch::missing_portable_revision_files(&route, &settings)
                    .is_empty(),
                "the old precheck would have attempted an offline fetch"
            );
            let decision = crate::utility_dispatch::begin_source_guard_before_fetch(
                &JobType::ImageGenerate,
                &payload,
                &settings,
                Some(&route),
            )
            .unwrap();
            let crate::utility_dispatch::SourceGuardBeforeFetch::Ready(guard) = decision else {
                panic!("complete leased bundle must not request a download")
            };
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );
            assert!(
                !library.exists(),
                "prefetch must not recreate the disconnected source"
            );
            guard.finish_success().unwrap();
        });
    }

    #[test]
    fn portable_prefetch_preserves_installed_external_unavailable_refusal() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            std::fs::rename(&library, temp.path().join("detached")).unwrap();
            let route = portable_installed_model_binding();
            let error = crate::utility_dispatch::begin_source_guard_before_fetch(
                &JobType::ImageGenerate,
                &payload,
                &settings,
                Some(&route),
            )
            .unwrap_err();
            assert!(
                matches!(error, WorkerError::ExternalLibraryUnavailable(_)),
                "{error}"
            );
            assert!(
                !library.exists(),
                "refusal must not recreate the source library"
            );
        });
    }

    /// Each invalid-entry case runs against a bundle that WOULD have served the load, with exactly
    /// one property broken, and asserts the control first — so a case can only pass because the
    /// broken property is what moved the load back to the authoritative source tier.
    fn invalid_local_entry_falls_back(break_it: impl FnOnce(&Path, &Path)) {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let bundle = artifact.location.root().to_path_buf();
            let source_snapshot =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("source snapshot");

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the intact bundle must serve this load"
            );
            drop(guard);

            break_it(&settings.data_dir, &bundle);

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady,
                "an invalid local entry must fall back to the authoritative source"
            );
            assert_eq!(
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model"),
                Some(source_snapshot),
                "and the load must read the source library, not the invalid bundle"
            );
        });
    }

    fn published_file(bundle: &Path) -> PathBuf {
        bundle.join(format!(
            "models--guardfx--model/snapshots/{REV_A}/model.safetensors"
        ))
    }

    #[test]
    fn a_torn_local_entry_never_wins() {
        invalid_local_entry_falls_back(|_, bundle| {
            std::fs::write(published_file(bundle), b"").unwrap();
        });
    }

    #[test]
    fn a_partial_local_entry_never_wins() {
        invalid_local_entry_falls_back(|_, bundle| {
            std::fs::remove_file(published_file(bundle)).unwrap();
        });
    }

    #[test]
    fn an_unverifiable_local_entry_never_wins() {
        invalid_local_entry_falls_back(|data_dir, _| {
            let entries = data_dir.join("models/resolved/entries");
            for entry in std::fs::read_dir(&entries).unwrap().flatten() {
                for slot in 0..=1 {
                    let path = entry.path().join(format!("metadata.{slot}.json"));
                    if path.exists() {
                        std::fs::write(&path, b"{\"not\":\"a journal\"}").unwrap();
                    }
                }
            }
        });
    }

    /// A bundle of a DIFFERENT revision or a DIFFERENT tier than the one this request selected is
    /// not this request's artifact: the selected closure keeps its source tier and the local entry
    /// is left alone.
    #[test]
    fn a_wrong_revision_or_wrong_tier_bundle_is_not_used() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q4/model.safetensors");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q8/model.safetensors");
        write_receipts(
            &data,
            "guardfx/matrix",
            json!([
                { "repo": "guardfx/matrix", "modelId": "m", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "guardfx/matrix", "modelId": "m", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_A }
            ]),
        );
        let settings = settings(data);
        let entry = json!({
            "id": "m",
            "downloads": [
                { "provider": "huggingface", "repo": "guardfx/matrix", "variant": "q4",
                  "default": true, "files": ["q4/*"] },
                { "provider": "huggingface", "repo": "guardfx/matrix", "variant": "q8",
                  "files": ["q8/*"] }
            ]
        });
        let payload = |variant: &str| {
            json!({ "modelManifestEntry": entry, "variant": variant })
                .as_object()
                .unwrap()
                .clone()
        };
        with_local_cache(&settings, &library, |settings| {
            // Only the q4 tier is published locally.
            let q4_bundle = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/matrix",
                REV_A,
                "q4",
                &["q4/model.safetensors"],
            );
            let q4_root = q4_bundle.location.root().to_path_buf();
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q4"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the q4 selection IS served by the q4 bundle"
            );
            drop(guard);

            // The q8 selection must NOT be served by the q4 bundle even though both share one
            // repository and one immutable revision.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q8"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            // AND — the assertion this test used to skip by calling `drop(guard)` first — the PATH
            // layer must agree. Availability said "source tier"; the bug lived one layer down,
            // where the process-wide overlay would still redirect `guardfx/matrix@REV_A` into the q4
            // bundle root, which holds no `q8/` subtree at all. With the guard STILL HELD:
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/matrix")
                    .expect("source snapshot");
            assert!(
                loaded.starts_with(&library),
                "the q8 load must read the source library, not the q4 bundle: {}",
                loaded.display()
            );
            assert!(!loaded.starts_with(&q4_root), "{}", loaded.display());
            assert!(
                loaded.join("q8/model.safetensors").is_file(),
                "and the path it was handed must actually hold the q8 weights"
            );
            drop(guard);

            // A bundle of another revision of the same model is likewise not this model.
            seed_snapshot(&library, "guardfx/other", REV_B, "model.safetensors");
            let other_bundle = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/other",
                REV_B,
                "default",
                &["model.safetensors"],
            );
            let other = json!({
                "modelManifestEntry": {
                    "id": "other",
                    "downloads": [{ "provider": "huggingface", "repo": "guardfx/other",
                                    "revision": REV_A, "files": ["model.safetensors"] }]
                }
            })
            .as_object()
            .unwrap()
            .clone();
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &other, &settings).unwrap();
            assert_ne!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "a bundle of a different revision must never satisfy a pinned request"
            );
            // Path layer, guard still held: the REV_B bundle must not answer for a REV_A request.
            assert!(crate::model_jobs::huggingface_pinned_snapshot_dir(
                &settings.data_dir,
                "guardfx/other",
                REV_A
            )
            .is_none_or(|path| !path.starts_with(other_bundle.location.root())));
            drop(guard);
        });
    }

    /// The DELIBERATE consequence of deciding coverage across the whole job: when a job carries
    /// both the q4 and the q8 selection of one model and only q4 is bundled, the q4 model ALSO
    /// falls back to the source tier.
    ///
    /// That is the correct conservative answer, not a gap. The two selections share one
    /// `(repository, revision)` pair, and the preference seam is directory-level: it answers that
    /// pair with ONE snapshot path and can never re-ask "which of you is asking" afterwards. Since
    /// no bundle holds the union of both selections' files, the pair is served to NOBODY. Serving
    /// q4 alone is exactly the defect — it would put the q8 load inside the q4 bundle root.
    #[test]
    fn a_pair_one_sibling_selection_does_not_cover_is_served_to_neither() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q4/model.safetensors");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q8/model.safetensors");
        write_receipts(
            &data,
            "guardfx/matrix",
            json!([
                { "repo": "guardfx/matrix", "modelId": "q4model", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "guardfx/matrix", "modelId": "q8model", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_A }
            ]),
        );
        let settings = settings(data);
        let entry = |id: &str, variant: &str, glob: &str| {
            json!({
                "id": id,
                "downloads": [{ "provider": "huggingface", "repo": "guardfx/matrix",
                                "variant": variant, "default": true, "files": [glob] }]
            })
        };
        // ONE job carrying BOTH selections of the same repository+revision.
        let payload = json!({
            "modelManifestEntry": entry("q4model", "q4", "q4/*"),
            "modelManifestEntries": [entry("q8model", "q8", "q8/*")]
        })
        .as_object()
        .unwrap()
        .clone();

        with_local_cache(&settings, &library, |settings| {
            let q4_bundle = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/matrix",
                REV_A,
                "q4",
                &["q4/model.safetensors"],
            );
            let q4_root = q4_bundle.location.root().to_path_buf();

            // Control: ALONE, the q4 selection is served from the bundle. This is what makes the
            // assertion below about the union rather than about the bundle being unusable.
            let alone = json!({ "modelManifestEntry": entry("q4model", "q4", "q4/*") })
                .as_object()
                .unwrap()
                .clone();
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &alone, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the q4 selection alone IS served locally"
            );
            drop(guard);

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(guard.resolutions().len(), 2);
            assert!(
                guard
                    .resolutions()
                    .iter()
                    .all(|resolution| resolution.availability != ModelAvailability::LocalReady),
                "neither selection may be served when no bundle covers their shared pair: {:?}",
                guard
                    .resolutions()
                    .iter()
                    .map(|resolution| &resolution.availability)
                    .collect::<Vec<_>>()
            );
            // The load-bearing half, at the path layer with the guard held: NOTHING resolves into
            // the q4 bundle, so the q8 load cannot land in a root that has no `q8/` subtree.
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/matrix")
                    .expect("source snapshot");
            assert!(!loaded.starts_with(&q4_root), "{}", loaded.display());
            assert!(loaded.starts_with(&library), "{}", loaded.display());
            assert!(loaded.join("q4/model.safetensors").is_file());
            assert!(loaded.join("q8/model.safetensors").is_file());
        });
    }

    #[test]
    fn repairing_a_legacy_revision_enables_runtime_local_loading_without_the_source() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let repo = "guardfx/revision-repair";
        seed_snapshot(&library, repo, REV_A, "model.safetensors");
        let mut receipt = json!({"repo":repo,"modelId":"repair", "resolvedFiles":["model.safetensors"],"snapshotRevision":null});
        write_receipts(&data, repo, json!([receipt]));
        let payload=json!({"modelManifestEntry":{"id":"repair","downloads":[{"provider":"huggingface","repo":repo,"files":["model.safetensors"]}]}}).as_object().unwrap().clone();
        let settings = settings(data.clone());
        with_local_cache(&settings, &library, |settings| {
            let artifact = publish_bundle(
                &data,
                &library,
                repo,
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert!(guard
                .resolutions()
                .iter()
                .all(|r| r.availability != ModelAvailability::LocalReady));
            drop(guard);
            receipt["snapshotRevision"] =
                json!(sceneworks_core::download_receipt::recover_revision(
                    &receipt,
                    &library.join("models--guardfx--revision-repair")
                )
                .unwrap());
            write_receipts(&data, repo, json!([receipt]));
            std::fs::rename(&library, temp.path().join("unplugged")).unwrap();
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );
            let loaded = crate::model_jobs::huggingface_snapshot_dir(&data, repo).unwrap();
            assert!(loaded.starts_with(artifact.location.root()));
            assert!(std::fs::File::open(loaded.join("model.safetensors")).is_ok());
            drop(guard);
        });
    }

    /// A requirement whose receipt carries NO `snapshotRevision` makes the WHOLE repository
    /// unserveable, not merely that one requirement.
    ///
    /// A legacy receipt yields `revision: None` (`artifact_selection.rs` maps a missing
    /// `snapshotRevision` straight through), and there is then no pair to compare coverage against
    /// — no bundle can be *shown* to hold what that requirement needs. Leaving the pinned sibling
    /// served anyway is not safe: the unpinned model re-resolves to the source tier, and its
    /// unpinned load walks `refs/main` into the overlay, which would hand it a bundle whose file
    /// set was never in the union.
    #[test]
    fn a_requirement_with_no_recorded_revision_makes_its_whole_repository_unserveable() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/shared", REV_A, "pinned.safetensors");
        seed_snapshot(&library, "guardfx/shared", REV_A, "legacy.safetensors");
        // `refs/main` is what an unpinned resolve reads — the walk that would reach the overlay.
        let repo_root = library.join("models--guardfx--shared");
        std::fs::create_dir_all(repo_root.join("refs")).unwrap();
        std::fs::write(repo_root.join("refs").join("main"), REV_A).unwrap();

        let pinned_receipt = json!({ "repo": "guardfx/shared", "modelId": "pinned",
                                     "resolvedFiles": ["pinned.safetensors"],
                                     "snapshotRevision": REV_A });
        // The legacy receipt: same repository, NO `snapshotRevision`.
        let legacy_receipt = json!({ "repo": "guardfx/shared", "modelId": "legacy",
                                     "resolvedFiles": ["legacy.safetensors"] });
        let pinned_entry = json!({
            "id": "pinned",
            "downloads": [{ "provider": "huggingface", "repo": "guardfx/shared",
                            "revision": REV_A, "files": ["pinned.safetensors"] }]
        });
        // The legacy carrier declares NO revision, so nothing overrides its receipt's `None` with a
        // declared-exact pin — this is what makes `revision: None` actually reach the guard.
        let legacy_entry = json!({
            "id": "legacy",
            "downloads": [{ "provider": "huggingface", "repo": "guardfx/shared",
                            "files": ["legacy.safetensors"] }]
        });

        // Control: the pinned selection ALONE is served locally. Without this the assertion below
        // could pass merely because the bundle was unusable.
        write_receipts(&data, "guardfx/shared", json!([pinned_receipt.clone()]));
        let settings = settings(data.clone());
        let alone = json!({ "modelManifestEntry": pinned_entry.clone() })
            .as_object()
            .unwrap()
            .clone();
        with_local_cache(&settings, &library, |settings| {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/shared",
                REV_A,
                "default",
                &["pinned.safetensors"],
            );
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &alone, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the pinned selection alone IS served locally"
            );
            drop(guard);
        });

        // Now add the legacy carrier on the SAME repository.
        write_receipts(
            &data,
            "guardfx/shared",
            json!([pinned_receipt, legacy_receipt]),
        );
        let payload = json!({
            "modelManifestEntry": pinned_entry,
            "modelManifestEntries": [legacy_entry]
        })
        .as_object()
        .unwrap()
        .clone();

        with_local_cache(&settings, &library, |settings| {
            let bundle_root = {
                let scan = sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore::valid_local_artifacts(
                    &settings.data_dir,
                );
                scan.artifacts[0].location.root().to_path_buf()
            };
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert!(
                guard
                    .resolutions()
                    .iter()
                    .all(|resolution| resolution.availability != ModelAvailability::LocalReady),
                "an unrevisioned requirement poisons the whole repository: {:?}",
                guard
                    .resolutions()
                    .iter()
                    .map(|resolution| &resolution.availability)
                    .collect::<Vec<_>>()
            );
            // The load-bearing half, at the PATH layer with the guard held: neither the pinned nor
            // the unpinned resolve may reach the bundle.
            let pinned = crate::model_jobs::huggingface_pinned_snapshot_dir(
                &settings.data_dir,
                "guardfx/shared",
                REV_A,
            )
            .expect("source snapshot");
            assert!(!pinned.starts_with(&bundle_root), "{}", pinned.display());
            assert!(pinned.starts_with(&library), "{}", pinned.display());
            // The unpinned walk (`refs/main`) is the one the poison exists to stop.
            let unpinned =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/shared")
                    .expect("source snapshot");
            assert!(
                !unpinned.starts_with(&bundle_root),
                "an unpinned load must never walk refs/main into a bundle whose files were never \
                 in the union: {}",
                unpinned.display()
            );
            assert!(unpinned.join("legacy.safetensors").is_file());
        });
    }

    /// Run `body` with `tracing` captured and return everything it emitted.
    fn captured_events<T>(body: impl FnOnce() -> T) -> (T, String) {
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        // Concurrent subscriber-less tests also drive these callsites; without the floor their
        // first hit can cache `Interest::never` and silently empty this capture (see test_env).
        crate::test_env::install_tracing_interest_floor();

        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .without_time()
            .finish();
        let value = tracing::subscriber::with_default(subscriber, body);
        let logged = String::from_utf8_lossy(
            &capture
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
        .into_owned();
        (value, logged)
    }

    /// A published, verified bundle whose LAYOUT the shared snapshot resolvers cannot be handed is
    /// reported as the local-tier-unsupported class and the model stays on the source tier.
    ///
    /// This class was previously unreachable for exactly the case it names: the scan dropped
    /// layout-unsupported entries with a bare `continue` before the guard ever saw them, so nothing
    /// could emit it. The event is what tells a user their local copy will never serve.
    #[test]
    fn an_unsupported_bundle_shape_is_reported_and_the_model_stays_on_the_source_tier() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            // Published and complete, but stored FLAT rather than in the source-library layout.
            publish_bundle_at(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
                PathBuf::new(),
            );

            let (guard, logged) = captured_events(|| {
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap()
            });

            assert!(
                logged.contains("resolved_cache_local_tier_unsupported"),
                "the unsupported shape must be reported: {logged}"
            );
            assert!(
                logged.contains("source-library layout"),
                "and it must name WHY: {logged}"
            );
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady,
                "an unsupported shape must never serve the load"
            );
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("source snapshot");
            assert!(loaded.starts_with(&library), "{}", loaded.display());
        });
    }

    /// "Where did this load's weights come from" is answered once per job, before any loader runs,
    /// for BOTH tiers — and the coverage refusal is reported as its OWN class rather than borrowing
    /// the unsupported-shape label, which describes a different fault entirely.
    #[test]
    fn the_serving_tier_is_reported_for_every_admitted_model() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        for repo in ["guardfx/bundled", "guardfx/external"] {
            seed_snapshot(&library, repo, REV_A, "model.safetensors");
            write_receipts(
                &data,
                repo,
                json!([{ "repo": repo, "modelId": repo,
                         "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
            );
        }
        let settings = settings(data);
        let entry = |repo: &str| {
            json!({
                "id": repo,
                "downloads": [{ "provider": "huggingface", "repo": repo, "revision": REV_A,
                                "files": ["model.safetensors"] }]
            })
        };
        let payload = json!({
            "modelManifestEntry": entry("guardfx/bundled"),
            "modelManifestEntries": [entry("guardfx/external")]
        })
        .as_object()
        .unwrap()
        .clone();

        with_local_cache(&settings, &library, |settings| {
            // Only ONE of the two models is bundled locally, so one job reports both tiers. They
            // are different repositories, so neither poisons the other's pair.
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/bundled",
                REV_A,
                "default",
                &["model.safetensors"],
            );

            let (guard, logged) = captured_events(|| {
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap()
            });

            assert!(
                logged.contains("model_source_tier_selected"),
                "every admitted model must report its tier: {logged}"
            );
            assert!(
                logged.contains("resolved_local"),
                "the bundled model must report the app-owned tier: {logged}"
            );
            assert!(
                logged.contains("source_library"),
                "the unbundled model must report the source tier: {logged}"
            );
            assert!(
                logged.contains("guardfx/bundled") && logged.contains("guardfx/external"),
                "each report must name its repository: {logged}"
            );
            // The unsupported-shape class describes a bundle that cannot be served at all; nothing
            // here is unsupported, so borrowing that label would be a lie.
            assert!(
                !logged.contains("resolved_cache_local_tier_unsupported"),
                "an ordinary source-tier model is not an unsupported shape: {logged}"
            );

            let tiers: Vec<_> = guard
                .resolutions()
                .iter()
                .map(|resolution| resolution.availability.clone())
                .collect();
            assert!(tiers.contains(&ModelAvailability::LocalReady), "{tiers:?}");
            assert!(
                tiers.contains(&ModelAvailability::ExternalReady),
                "{tiers:?}"
            );
            guard.finish_success().unwrap();
        });
    }

    /// A coverage refusal is reported as its own class. It is not an unsupported shape — the bundle
    /// is perfectly well-formed; it simply does not hold everything THIS job needs from the pair.
    #[test]
    fn a_coverage_refusal_is_reported_as_its_own_class_not_as_an_unsupported_shape() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q4/model.safetensors");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q8/model.safetensors");
        write_receipts(
            &data,
            "guardfx/matrix",
            json!([
                { "repo": "guardfx/matrix", "modelId": "q4model", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "guardfx/matrix", "modelId": "q8model", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_A }
            ]),
        );
        let settings = settings(data);
        let entry = |id: &str, variant: &str, glob: &str| {
            json!({
                "id": id,
                "downloads": [{ "provider": "huggingface", "repo": "guardfx/matrix",
                                "variant": variant, "default": true, "files": [glob] }]
            })
        };
        let payload = json!({
            "modelManifestEntry": entry("q4model", "q4", "q4/*"),
            "modelManifestEntries": [entry("q8model", "q8", "q8/*")]
        })
        .as_object()
        .unwrap()
        .clone();

        with_local_cache(&settings, &library, |settings| {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/matrix",
                REV_A,
                "q4",
                &["q4/model.safetensors"],
            );
            let (_guard, logged) = captured_events(|| {
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap()
            });
            assert!(
                logged.contains("resolved_cache_local_tier_not_selected"),
                "the coverage refusal must be reported: {logged}"
            );
            assert!(
                !logged.contains("resolved_cache_local_tier_unsupported"),
                "a well-formed bundle that merely does not cover the job is NOT an unsupported \
                 shape: {logged}"
            );
        });
    }

    /// The resolved cache is opt-in: with it switched off no entry is read, no session is opened,
    /// and every runtime keeps reading the configured source library byte for byte.
    #[test]
    fn a_disabled_cache_never_prefers_a_local_bundle() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
        });
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            assert!(crate::model_jobs::huggingface_snapshot_dir(
                &settings.data_dir,
                "guardfx/model"
            )
            .expect("source snapshot")
            .starts_with(&library));
        });
    }

    /// sc-19706 — THE test the reopened story exists for: production populates the cache.
    ///
    /// Nothing here publishes a bundle by hand. A job resolves against the configured source
    /// library, runs, and finishes; the trigger records the closure; the idle drain builds the
    /// candidate and materializes it; and a SECOND guard pass over the SAME payload resolves the
    /// model from the app-owned tier, with the shared snapshot resolver answering from the bundle.
    /// That whole chain is the epic's user-visible value, and before this story none of it ran
    /// outside a test that published the bundle itself.
    ///
    /// It also pins the two-stage split: the bundle must NOT exist when the job finishes. If
    /// `finish_success` ever materialized inline, the first assertion below fails — which is the
    /// only way to catch source-library I/O silently moving onto the job path.
    #[test]
    fn a_successful_source_tier_job_promotes_itself_and_the_next_job_loads_from_the_bundle() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&settings, &library, |settings| {
            // FIRST JOB — nothing is cached, so it is served by the configured source library.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            let source_snapshot =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("source snapshot");
            assert!(source_snapshot.starts_with(&library));

            guard.finish_success().unwrap();

            // The job path did the queueing and NOTHING else: no bundle, no store, no copy.
            assert!(
                crate::resolved_cache_promotion::work_pending(&settings.data_dir),
                "a successful source-tier load must record a promotion candidate"
            );
            assert!(
                !settings
                    .data_dir
                    .join("models")
                    .join("resolved")
                    .join("entries")
                    .is_dir(),
                "finish_success must not materialize inline — that is source-library I/O on the \
                 job path"
            );

            // IDLE DRAIN — what the worker's poll loop runs on the blocking pool when a claim
            // returns nothing.
            crate::resolved_cache_promotion::drain_intake(&settings);
            assert!(
                !crate::resolved_cache_promotion::work_pending(&settings.data_dir),
                "the drain consumes the intake"
            );

            let store = ResolvedCacheStore::open(&settings.data_dir).unwrap();
            let published = store
                .enumerate()
                .unwrap()
                .into_iter()
                .filter_map(|entry| entry.metadata)
                .find(|metadata| {
                    metadata.state
                        == sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheEntryState::Complete
                })
                .expect("the drain published a complete bundle");
            assert_eq!(published.artifact.identity.repository, "guardfx/model");
            assert_eq!(published.artifact.identity.revision, REV_A);

            // SECOND JOB — same payload, same worker, nothing else changed. It must now be served
            // from the app-owned tier.
            let second =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                second.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "the bundle production just wrote must serve the next load"
            );
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("leased local snapshot");
            // Compared against the published bundle root rather than a `starts_with` on
            // `data_dir`: the store records the CANONICAL root, and on macOS the temp dir's
            // `/var` -> `/private/var` symlink makes a lexical prefix check fail on a path that is
            // in fact inside the bundle.
            assert_eq!(
                loaded,
                published
                    .artifact
                    .location
                    .root()
                    .join(format!("models--guardfx--model/snapshots/{REV_A}")),
                "the shared snapshot resolver must answer from the bundle"
            );
            assert!(loaded.join("model.safetensors").is_file());
            second.finish_success().unwrap();

            // A LOCALLY served job records NOTHING. This is the qualification rule asserted in the
            // widening direction, which the `Complete == 1` check below cannot do: relaxing the
            // filter to record local-ready closures too would still leave exactly one complete
            // entry, because the redundant candidate is admitted as `AlreadyComplete`. Only the
            // intake being empty distinguishes "never recorded" from "recorded and then declined",
            // and the difference is a whole idle turn plus a full source-library closure walk
            // burned on a bundle that is already published.
            assert!(
                !crate::resolved_cache_promotion::work_pending(&settings.data_dir),
                "a job served from the local tier is already promoted and must record nothing"
            );

            // And the already-published entry is not promoted a second time: the drain admits it
            // as already complete and copies nothing.
            crate::resolved_cache_promotion::drain_intake(&settings);
            assert_eq!(
                store
                    .enumerate()
                    .unwrap()
                    .iter()
                    .filter(|entry| entry.state
                        == sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheEntryState::Complete)
                    .count(),
                1
            );
        });
    }

    #[test]
    fn installed_model_admits_as_external_ready_with_a_source_session() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(guard.resolutions().len(), 1);
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            guard.finish_success().unwrap();
            let sessions = settings
                .data_dir
                .join("models/.sceneworks-external-source-sessions");
            assert_eq!(std::fs::read_dir(sessions).unwrap().count(), 0);
        });
    }

    #[test]
    fn disconnected_library_fails_typed_and_receipts_survive_untouched() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let receipt_path = settings.data_dir.join("models").join(
                sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(
                    "guardfx/model",
                ),
            );
            let receipt_before =
                std::fs::read(receipt_path.join(".sceneworks-download-complete.json")).unwrap();
            // Bind first while connected, then disconnect the volume.
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap()
                .finish_success()
                .unwrap();
            std::fs::rename(&library, temp.path().join("detached")).unwrap();

            let error = RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap_err();
            assert!(matches!(error, WorkerError::ExternalLibraryUnavailable(_)));
            assert!(error
                .to_string()
                .contains(EXTERNAL_LIBRARY_UNAVAILABLE_CODE));
            // Installed state is never lost: receipts are byte-identical after the disconnect.
            assert_eq!(
                std::fs::read(receipt_path.join(".sceneworks-download-complete.json")).unwrap(),
                receipt_before
            );

            // Reconnect: the same payload admits again under the original binding.
            std::fs::rename(temp.path().join("detached"), &library).unwrap();
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
        });
    }

    #[test]
    fn component_removal_while_connected_is_incomplete_never_the_disconnect_class() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap()
                .finish_success()
                .unwrap();
            let safe = sceneworks_core::hf_home::safe_repo_dir_name("guardfx/model").unwrap();
            std::fs::remove_file(
                library
                    .join(format!("models--{safe}"))
                    .join("snapshots")
                    .join(REV_A)
                    .join("model.safetensors"),
            )
            .unwrap();
            // The library is provably present; the missing component makes the install stale, and
            // stale keeps its established on-demand install/repair semantics — it must NOT read
            // as a library disconnect.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Incomplete
            );
        });
    }

    #[test]
    fn model_routes_fail_closed_without_carriers_and_no_model_routes_pass() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        with_library(&library, || {
            let empty = JsonObject::new();
            for model_backed in [
                JobType::ImageGenerate,
                JobType::DatasetUpscale,
                JobType::PromptRefine,
                JobType::CatalogAnalysis,
            ] {
                let error =
                    RuntimeSourceGuard::begin(&model_backed, &empty, &settings).unwrap_err();
                assert!(
                    matches!(error, WorkerError::InvalidPayload(_)),
                    "{model_backed:?} must fail closed on an empty carrier set"
                );
            }
            for model_free in [
                JobType::Placeholder,
                JobType::FrameExtract,
                JobType::TimelineExport,
                JobType::DatasetParquetImport,
            ] {
                RuntimeSourceGuard::begin(&model_free, &empty, &settings).unwrap();
            }
        });
    }

    #[test]
    fn completeness_is_judged_on_the_selected_tier_not_a_variant_union() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        // q4 fully installed; q8 receipt exists but its library snapshot was pruned.
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q4/model.safetensors");
        write_receipts(
            &data,
            "guardfx/matrix",
            json!([
                { "repo": "guardfx/matrix", "modelId": "m", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "guardfx/matrix", "modelId": "m", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_B }
            ]),
        );
        let settings = settings(data);
        let entry = json!({
            "id": "m",
            "downloads": [
                { "provider": "huggingface", "repo": "guardfx/matrix", "variant": "q4",
                  "default": true, "files": ["q4/*"] },
                { "provider": "huggingface", "repo": "guardfx/matrix", "variant": "q8",
                  "files": ["q8/*"] }
            ]
        });
        let payload = |variant: &str| {
            json!({ "modelManifestEntry": entry, "variant": variant })
                .as_object()
                .unwrap()
                .clone()
        };
        with_library(&library, || {
            // The selected q4 tier is complete even though the q8 sibling receipt cannot
            // validate — a model-wide union of variant receipts would wrongly demote this to
            // incomplete.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q4"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            assert_eq!(guard.resolutions()[0].requirements.len(), 1);
            assert_eq!(guard.resolutions()[0].requirements[0].variant, "q4");
            drop(guard);
            // And the pruned q8 selection reads incomplete — the q4 install must not mask it.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q8"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Incomplete
            );
        });
    }

    #[test]
    fn requirements_use_this_workers_platform_closure_not_another_hosts() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let this_os = std::env::consts::OS;
        let other_os = if this_os == "macos" {
            "windows"
        } else {
            "macos"
        };
        seed_snapshot(&library, "guardfx/native", REV_A, "model.safetensors");
        write_receipts(
            &data,
            "guardfx/native",
            json!([{ "repo": "guardfx/native", "modelId": "m",
                     "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
        );
        // The other platform's primary and co-requisite are NOT installed anywhere. If the guard
        // used another host's platform closure (or no platform filter), admission would fail.
        let payload = json!({
            "modelManifestEntry": {
                "id": "m",
                "downloads": [
                    { "provider": "huggingface", "repo": "guardfx/native", "revision": REV_A,
                      "files": ["model.safetensors"], "platforms": [this_os] },
                    { "provider": "huggingface", "repo": "guardfx/foreign", "revision": REV_B,
                      "files": ["foreign.safetensors"], "platforms": [other_os] },
                    { "provider": "huggingface", "repo": "guardfx/foreign-corequisite",
                      "revision": REV_B, "coRequisite": true,
                      "files": ["encoder.safetensors"], "platforms": [other_os] }
                ]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let settings = settings(data);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            let repositories = guard.resolutions()[0]
                .requirements
                .iter()
                .map(|requirement| requirement.repository.as_str())
                .collect::<Vec<_>>();
            assert_eq!(repositories, ["guardfx/native"]);
        });
    }

    #[test]
    fn terminal_remap_occurs_only_when_the_exact_source_probe_proves_disconnect() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            let preserved = guard.classify_failure(
                &settings,
                WorkerError::Engine("real loader defect".to_owned()),
            );
            assert!(
                matches!(preserved, WorkerError::Engine(ref value) if value == "real loader defect"),
                "an engine failure with the source present must be preserved verbatim"
            );

            std::fs::rename(&library, temp.path().join("detached")).unwrap();
            let remapped = guard.classify_failure(
                &settings,
                WorkerError::Engine("No such file or directory (os error 2)".to_owned()),
            );
            assert!(matches!(
                remapped,
                WorkerError::ExternalLibraryUnavailable(_)
            ));
            assert!(
                !remapped.to_string().contains("os error 2"),
                "a proven disconnect must never surface as a raw ENOENT"
            );
            drop(guard);
            let sessions = settings
                .data_dir
                .join("models/.sceneworks-external-source-sessions");
            assert_eq!(
                std::fs::read_dir(sessions).unwrap().count(),
                0,
                "failure cleanup removes only the operation-owned source session"
            );
        });
    }

    #[test]
    fn never_installed_model_keeps_its_on_demand_install_path() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        std::fs::create_dir_all(&library).unwrap();
        let settings = settings(data);
        // Manifest present, but no receipt and glob files (no exact declaration): no durable
        // install identity, so availability is Missing and the job proceeds to its established
        // download path rather than failing typed.
        let payload = json!({
            "modelManifestEntry": {
                "id": "m",
                "downloads": [{ "provider": "huggingface", "repo": "guardfx/new",
                                "revision": REV_A, "files": ["q4/*"] }]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Missing
            );
        });
    }

    #[test]
    fn explicit_download_is_exempt_only_for_download_job_types() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let payload = json!({
            "modelArtifactOperation": { "schemaVersion": 1, "kind": "explicit_download" }
        })
        .as_object()
        .unwrap()
        .clone();
        RuntimeSourceGuard::begin(&JobType::ModelDownload, &payload, &settings).unwrap();
        RuntimeSourceGuard::begin(&JobType::LoraDownload, &payload, &settings).unwrap();
        assert!(RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).is_err());
    }

    #[test]
    fn only_confined_download_free_local_entries_may_use_the_non_hf_exception() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let imported = settings.data_dir.join("models/imports/custom");
        std::fs::create_dir_all(&imported).unwrap();
        std::fs::write(imported.join("model.safetensors"), b"weights").unwrap();
        let local = json!({
            "modelManifestEntry": {
                "id": "imported",
                "downloads": [],
                "paths": { "model": imported }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let outside = temp.path().join("outside/model.safetensors");
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        std::fs::write(&outside, b"outside").unwrap();
        let escaped = json!({
            "modelManifestEntry": {
                "id": "forged-local",
                "downloads": [],
                "paths": { "model": outside }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let bare = json!({ "modelManifestEntry": { "id": "bare" } })
            .as_object()
            .unwrap()
            .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &local, &settings).unwrap();
            assert!(
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &escaped, &settings).is_err()
            );
            // No downloads AND no paths proves nothing: fail closed.
            assert!(RuntimeSourceGuard::begin(&JobType::ImageGenerate, &bare, &settings).is_err());
        });
    }

    /// The FULL shape the model-import job writes (`apps/rust-api/src/models.rs`): an absolute
    /// `paths.model` install dir PLUS a `source` PROVENANCE block carrying a data-dir-relative
    /// `path`, a `provider` literal, a `repo` that is null for a file import, and a `url`.
    fn imported_entry(data_dir: &Path, id: &str) -> Value {
        imported_entry_with_family(data_dir, id, "krea_2")
    }

    /// [`imported_entry`] with the declared plan family chosen by the caller.
    fn imported_entry_with_family(data_dir: &Path, id: &str, family: &str) -> Value {
        let relative = format!("models/imports/{id}");
        let install = data_dir.join(&relative);
        std::fs::create_dir_all(&install).unwrap();
        std::fs::write(install.join("model.safetensors"), b"weights").unwrap();
        json!({
            "id": id,
            "name": id,
            "type": "image",
            "family": family,
            "downloads": [],
            "source": {
                "provider": "local",
                "repo": Value::Null,
                "path": relative,
                "url": "https://example.invalid/kreamania.safetensors",
            },
            "files": ["model.safetensors"],
            "paths": { "model": install },
        })
    }

    /// sc-20524 — a `source` block is provenance, not a path set. `provider` ("local"), `repo`
    /// and `url` are identifiers; only `source.path` names a location, and it is written
    /// data-dir-relative. Confining every `source` value as a filesystem path resolved `"local"`
    /// against the process cwd, so EVERY user-imported model failed the pre-loader guard with
    /// "non-HF model source must be inside an app-managed directory".
    #[test]
    fn imported_entry_source_provenance_is_not_confined_as_a_path_set() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let payload = json!({
            "model": "kreamania_v1",
            "modelManifestEntry": imported_entry(&settings.data_dir, "kreamania_v1"),
        })
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
        });
    }

    /// The whole reported population: every imported KreaMania checkpoint must clear the guard.
    #[test]
    fn every_imported_kreamania_checkpoint_clears_the_non_hf_guard() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        with_library(&library, || {
            for id in [
                "kreamania_v1",
                "kreamania_v2",
                "kreamania_v3",
                "kreamania_v4_int8",
                "kreamania_v5",
                "kreamania_v6",
            ] {
                let payload = json!({
                    "model": id,
                    "modelManifestEntry": imported_entry(&settings.data_dir, id),
                })
                .as_object()
                .unwrap()
                .clone();
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                    .unwrap_or_else(|error| panic!("{id} must clear the guard: {error}"));
            }
        });
    }

    /// sc-22329 fixtures: an installed resident base, and the imported entry that pairs with it.
    ///
    /// The family and its base entry are SYNTHETIC (`guardfx/` repositories, `e22329…` revisions),
    /// installed through the thread-local row override rather than the production table. That is
    /// required, not tidiness: `discover_snapshot`'s `unique_local_snapshot` fallback is keyed on
    /// the REPOSITORY and ignores the revision, so a bundle published here for a real repository
    /// would be a candidate answer for any concurrently running test that resolves that repository
    /// against its own temp library — the exact cross-test bleed the note at the top of this
    /// module exists to prevent, and one a unique revision does not close.
    const REV_C: &str = "e22329cccccccccccccccccccccccccccccccccc";

    /// An imported job whose declared family has a resident base, plus that base's install
    /// receipts and source snapshot. `shape` picks which repository the family's route reads:
    /// `FullTier` installs the base's own tier, `ComponentsOnly` a separate components repo whose
    /// receipt carries NO `modelId` (the shape a sibling catalog card's download writes).
    fn resident_base_fixture(
        temp: &TempDir,
        shape: super::ResidentBaseShape,
    ) -> (Settings, PathBuf, JsonObject, &'static str, &'static str) {
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let (read_repo, file) = match shape {
            super::ResidentBaseShape::FullTier => (
                "guardfx/resident-base",
                "bf16/text_encoder/model.safetensors",
            ),
            super::ResidentBaseShape::ComponentsOnly => (
                "guardfx/resident-components",
                "bf16/text_encoder/model.safetensors",
            ),
        };
        seed_snapshot(&library, read_repo, REV_C, file);
        write_receipts(
            &data,
            read_repo,
            match shape {
                // A primary row's receipts are filtered by `modelId`.
                super::ResidentBaseShape::FullTier => json!([{
                    "repo": read_repo, "modelId": "guardfx_base", "variant": "bf16",
                    "resolvedFiles": [file], "snapshotRevision": REV_C }]),
                // A co-requisite's are not — and must not be, because whichever sibling card the
                // user downloaded wrote this one.
                super::ResidentBaseShape::ComponentsOnly => json!([{
                    "repo": read_repo, "variant": "bf16",
                    "resolvedFiles": [file], "snapshotRevision": REV_C }]),
            },
        );
        let settings = settings(data);
        let payload = json!({
            "model": "guardfx_import",
            "modelManifestEntry":
                imported_entry_with_family(&settings.data_dir, "guardfx_import", "guardfx-family"),
        })
        .as_object()
        .unwrap()
        .clone();
        (settings, library, payload, read_repo, file)
    }

    /// The synthetic builtin catalog entry the override hands back for `guardfx-family`. Its
    /// `ComponentsOnly` form mirrors the real Mage shape: a large primary the route never opens,
    /// plus per-component co-requisite rows in a DIFFERENT repository, declared with globs and a
    /// `subdir`, one row per component.
    fn resident_base_entry(shape: super::ResidentBaseShape) -> Value {
        match shape {
            super::ResidentBaseShape::FullTier => json!({
                "id": "guardfx_base",
                "downloads": [
                    { "provider": "huggingface", "repo": "guardfx/resident-base",
                      "variant": "bf16", "files": ["bf16/*"] }
                ],
            }),
            super::ResidentBaseShape::ComponentsOnly => json!({
                "id": "guardfx_base",
                "downloads": [
                    { "provider": "huggingface", "repo": "guardfx/resident-primary",
                      "variant": "bf16", "files": ["bf16/*"] },
                    { "provider": "huggingface", "repo": "guardfx/resident-components",
                      "variant": "bf16", "coRequisite": true,
                      "files": ["bf16/text_encoder/*"], "subdir": "bf16/text_encoder" },
                    { "provider": "huggingface", "repo": "guardfx/resident-components",
                      "variant": "bf16", "coRequisite": true,
                      "files": ["bf16/vae/*"], "subdir": "bf16/vae" }
                ],
            }),
        }
    }

    /// sc-22329 — the story's headline outcome, at the guard seam: a plan-backed imported
    /// checkpoint's resident BASE model participates in local-tier selection, so the shared
    /// snapshot resolver its family route reads the base through
    /// (`huggingface_snapshot_dir` → `resolve_huggingface_snapshot_dir`) answers with the leased
    /// bundle instead of streaming the base off the source library on every cold load. The
    /// imported entry itself carries no downloads, so before this the guard `continue`d past it
    /// and no scope was ever installed.
    ///
    /// Run for BOTH resident-base shapes, because they reach the closure by different routes:
    /// `FullTier` through the base entry's primary download, `ComponentsOnly` through its
    /// component co-requisites alone.
    #[test]
    fn an_imported_checkpoints_resident_base_is_served_from_the_leased_bundle() {
        for shape in [
            super::ResidentBaseShape::FullTier,
            super::ResidentBaseShape::ComponentsOnly,
        ] {
            let temp = TempDir::new().unwrap();
            let (settings, library, payload, read_repo, file) = resident_base_fixture(&temp, shape);
            with_local_cache(&settings, &library, |settings| {
                let _rows = ResidentBaseRowGuard::install(
                    "guardfx-family",
                    "bf16",
                    shape,
                    resident_base_entry(shape),
                );
                let artifact = publish_bundle(
                    &settings.data_dir,
                    &library,
                    read_repo,
                    REV_C,
                    "bf16",
                    &[file],
                );
                let safe = read_repo.replace('/', "--");
                let bundle_snapshot = artifact
                    .location
                    .root()
                    .join(format!("models--{safe}/snapshots/{REV_C}"));

                // Control: before the guard, the base resolves to the source library — the exact
                // path the story's 30-second cold loads were reading.
                let source_snapshot =
                    crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, read_repo)
                        .expect("source snapshot");
                assert!(source_snapshot.starts_with(&library), "{shape:?}");

                let guard = RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                    .unwrap();
                assert_eq!(guard.resolutions().len(), 1, "{shape:?}");
                assert_eq!(
                    guard.resolutions()[0].availability,
                    ModelAvailability::LocalReady,
                    "{shape:?}"
                );
                assert_eq!(
                    crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, read_repo)
                        .expect("leased local snapshot"),
                    bundle_snapshot,
                    "{shape:?}"
                );
                assert!(bundle_snapshot.join(file).is_file(), "{shape:?}");
                guard.finish_success().unwrap();
                assert_eq!(
                    crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, read_repo),
                    Some(source_snapshot),
                    "{shape:?}"
                );

                // Mutation control: the identical payload with the declared family REMOVED must
                // produce no plan at all — proving the served path above came from the family
                // row, not from some other admission of the entry.
                let mut no_family = payload.clone();
                no_family
                    .get_mut("modelManifestEntry")
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove("family");
                let guard =
                    RuntimeSourceGuard::begin(&JobType::ImageGenerate, &no_family, &settings)
                        .unwrap();
                assert!(guard.resolutions().is_empty(), "{shape:?}");
                assert!(
                    crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, read_repo)
                        .expect("source snapshot without a family")
                        .starts_with(&library),
                    "{shape:?}"
                );
                guard.finish_success().unwrap();
            });
        }
    }

    /// sc-22329 — a `ComponentsOnly` family's closure must name ONLY the repository its resolver
    /// reads. The base entry's own primary is large and never opened by an imported checkpoint
    /// (the checkpoint IS the transformer), and naming it would be worse than useless: pass 3 is
    /// all-or-nothing across a model's pairs, so an uncached primary would ALSO stop the
    /// components pair from being served, and the primary's bytes would be queued for promotion.
    ///
    /// Failing mutation (run): give the `guardfx-family` row `ResidentBaseShape::FullTier` — the
    /// primary joins the closure, nothing covers it, and the components stop being served.
    #[test]
    fn a_components_only_resident_base_ignores_the_base_models_own_primary() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload, read_repo, file) =
            resident_base_fixture(&temp, super::ResidentBaseShape::ComponentsOnly);
        // The primary exists in the library and is NOT cached — the ordinary state for a user who
        // imported a fine-tune and never downloaded the base's own transformer.
        seed_snapshot(
            &library,
            "guardfx/resident-primary",
            REV_C,
            "bf16/transformer/model.safetensors",
        );
        with_local_cache(&settings, &library, |settings| {
            let _rows = ResidentBaseRowGuard::install(
                "guardfx-family",
                "bf16",
                super::ResidentBaseShape::ComponentsOnly,
                resident_base_entry(super::ResidentBaseShape::ComponentsOnly),
            );
            publish_bundle(
                &settings.data_dir,
                &library,
                read_repo,
                REV_C,
                "bf16",
                &[file],
            );
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "the components pair must serve even though the base's primary is uncached"
            );
            assert!(!crate::model_jobs::huggingface_snapshot_dir(
                &settings.data_dir,
                "guardfx/resident-primary"
            )
            .expect("the primary still resolves")
            .starts_with(settings.data_dir.join("models").join("resolved")));
            guard.finish_success().unwrap();
        });
    }

    /// One resident-base degradation: a label, and the mutation that puts the installed base into
    /// that state. The mutation takes only the source library — every case below is a change to
    /// what the library holds, not to the app data dir.
    type ResidentBaseDegradation = (&'static str, fn(&Path));

    /// sc-22329 AC2 — a resident base may only ever ADD a local-tier serve. Each degradation here
    /// is a state the guard must admit exactly as it did before the table existed, so the family
    /// route's own tagged `[checkpoint-plan:missing-component]` refusal keeps owning it. Two of
    /// these reach code that DOES refuse for a carried model, which is the whole point: an
    /// advisory plan must not borrow that power.
    #[test]
    fn a_resident_base_never_refuses_admission() {
        // (label, prepare) — each leaves the base in a state a carried model would fail on.
        let cases: [ResidentBaseDegradation; 3] = [
            // Not installed at all: no receipts, nothing in the library.
            ("uninstalled", |_| {}),
            // Installed and receipted, but the library holding it is gone. A CARRIED model in
            // this state is `InstalledExternalUnavailable` and the guard refuses typed.
            ("library disconnected", |library| {
                std::fs::remove_dir_all(library).unwrap();
            }),
            // Installed, library present, but the snapshot's files are gone underneath it — the
            // state that fails `ExternalSourceSession::begin` for a carried model.
            ("components vanished", |library| {
                let root = library.join("models--guardfx--resident-base");
                std::fs::remove_dir_all(&root).unwrap();
                std::fs::create_dir_all(root.join("snapshots")).unwrap();
            }),
        ];
        for (label, prepare) in cases {
            let temp = TempDir::new().unwrap();
            let (settings, library, payload, ..) =
                resident_base_fixture(&temp, super::ResidentBaseShape::FullTier);
            with_local_cache(&settings, &library, |settings| {
                let _rows = ResidentBaseRowGuard::install(
                    "guardfx-family",
                    "bf16",
                    super::ResidentBaseShape::FullTier,
                    resident_base_entry(super::ResidentBaseShape::FullTier),
                );
                prepare(&library);
                let guard = RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                    .unwrap_or_else(|error| {
                        panic!("a resident base must never refuse admission ({label}): {error}")
                    });
                // Nothing was served locally, and nothing opened an external session for it.
                assert!(
                    guard.local_scope.is_none(),
                    "{label} must install no local scope"
                );
                assert!(
                    guard.sessions.is_empty(),
                    "{label} must open no external source session"
                );
                guard.finish_success().unwrap();
            });
        }
    }

    /// The companion to the case above: a bundle that exists but does NOT cover the closure. The
    /// base falls back to the source tier (pass 3), and admission still succeeds.
    #[test]
    fn a_resident_base_whose_bundle_does_not_cover_the_closure_falls_back_without_refusing() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload, read_repo, file) =
            resident_base_fixture(&temp, super::ResidentBaseShape::FullTier);
        // A SECOND installed file the receipt names, absent from the bundle published below.
        write_receipts(
            &settings.data_dir,
            read_repo,
            json!([{ "repo": read_repo, "modelId": "guardfx_base", "variant": "bf16",
                     "resolvedFiles": [file, "bf16/vae/model.safetensors"],
                     "snapshotRevision": REV_C }]),
        );
        seed_snapshot(&library, read_repo, REV_C, "bf16/vae/model.safetensors");
        with_local_cache(&settings, &library, |settings| {
            let _rows = ResidentBaseRowGuard::install(
                "guardfx-family",
                "bf16",
                super::ResidentBaseShape::FullTier,
                resident_base_entry(super::ResidentBaseShape::FullTier),
            );
            publish_bundle(
                &settings.data_dir,
                &library,
                read_repo,
                REV_C,
                "bf16",
                &[file],
            );
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert!(
                guard.local_scope.is_none(),
                "partial coverage must not serve"
            );
            assert!(
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, read_repo)
                    .expect("source snapshot")
                    .starts_with(&library),
                "the base must read from the source tier when the bundle is short a file"
            );
            guard.finish_success().unwrap();
        });
    }

    /// sc-22329 fail-open, through the PRODUCTION table rather than the override: a real imported
    /// KreaMania entry whose `krea_2` base is not installed admits with no plan and no error.
    #[test]
    fn an_imported_entry_without_an_installed_base_admits_with_no_plan() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let payload = json!({
            "model": "kreamania_v5",
            "modelManifestEntry": imported_entry(&settings.data_dir, "kreamania_v5"),
        })
        .as_object()
        .unwrap()
        .clone();
        with_local_cache(&settings, &library, |settings| {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert!(guard.resolutions().is_empty());
            guard.finish_success().unwrap();
        });
    }

    /// The real manifest written on Windows carries `paths.model` in VERBATIM (`\\?\C:\...`)
    /// form — the model-import job canonicalizes the install dir before recording it. That is
    /// the exact user-reported shape (kreamania_v5/v6), so pin it verbatim.
    #[cfg(windows)]
    #[test]
    fn imported_entry_with_verbatim_install_path_clears_the_guard() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let mut entry = imported_entry(&settings.data_dir, "kreamania_v5");
        let install = settings.data_dir.join("models/imports/kreamania_v5");
        let verbatim = std::fs::canonicalize(&install).unwrap();
        assert!(
            verbatim.to_string_lossy().starts_with(r"\\?\"),
            "canonicalize must produce the verbatim form this test exists to pin"
        );
        entry["paths"]["model"] = json!(verbatim);
        let payload = json!({ "model": "kreamania_v5", "modelManifestEntry": entry })
            .as_object()
            .unwrap()
            .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap_or_else(
                |error| panic!("verbatim install path must clear the guard: {error}"),
            );
        });
    }

    /// Reading only `source.path` must not become "read nothing": a `source.path` that escapes
    /// the app-managed roots is still a rejection, and a benign-looking `source` block never
    /// launders an entry whose real `paths` point outside.
    #[test]
    fn source_path_is_still_confined_and_cannot_launder_an_outside_entry() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let outside = temp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("model.safetensors"), b"outside").unwrap();
        let carrier = |entry: Value| {
            json!({ "modelManifestEntry": entry })
                .as_object()
                .unwrap()
                .clone()
        };
        // `..` out of the data dir through the relative locator.
        let escaping_relative = carrier(json!({
            "id": "escaping-relative",
            "downloads": [],
            "source": { "provider": "local", "path": "../outside" },
        }));
        // An absolute `source.path` outside every root.
        let escaping_absolute = carrier(json!({
            "id": "escaping-absolute",
            "downloads": [],
            "source": { "provider": "local", "path": outside },
        }));
        // Provenance that says "local" while the actual weights live outside.
        let laundered = carrier(json!({
            "id": "laundered",
            "downloads": [],
            "source": { "provider": "local", "repo": Value::Null, "path": "models/imports/ok" },
            "paths": { "model": outside },
        }));
        with_library(&library, || {
            for (label, payload) in [
                ("escaping relative source.path", &escaping_relative),
                ("absolute outside source.path", &escaping_absolute),
                ("outside paths.model", &laundered),
            ] {
                // Assert the CONFINEMENT rejection specifically. `is_err()` alone would stay green
                // if a future change stopped reading these locators at all: the locator set would
                // go empty, the entry would fail the "no supported artifact source" branch, and the
                // test would pass while the mechanism it exists to pin had been removed.
                assert_confinement_rejection(
                    label,
                    RuntimeSourceGuard::begin(&JobType::ImageGenerate, payload, &settings),
                );
            }
        });
    }

    #[track_caller]
    fn assert_confinement_rejection(label: &str, outcome: WorkerResult<RuntimeSourceGuard>) {
        match outcome {
            Err(WorkerError::InvalidPayload(message))
                if message.contains("must be inside an app-managed directory") =>
            {
                assert!(
                    message.starts_with("non-HF model source"),
                    "{label} must be rejected by the non-HF locator guard, got: {message}"
                );
            }
            Err(error) => panic!("{label} must fail confinement, not something else: {error}"),
            Ok(_) => panic!("{label} must still be rejected"),
        }
    }

    /// Windows spells two path forms that `Path::is_absolute` calls relative while `PathBuf::join`
    /// refuses to anchor: root-relative (`\Windows\System32` — join keeps only the data dir's drive
    /// prefix) and drive-relative (`D:models` — join replaces the data dir outright). Neither means
    /// anything relative to the data dir, so `source.path` must refuse both rather than confine a
    /// path the anchoring never produced. On Unix both strings are ordinary single-component
    /// filenames and genuinely do anchor, so this is Windows-only by construction.
    #[cfg(windows)]
    #[test]
    fn windows_root_relative_and_drive_relative_source_paths_are_refused() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let drive_relative = format!(
            "{}models",
            settings
                .data_dir
                .components()
                .next()
                .map(|component| component.as_os_str().to_string_lossy().into_owned())
                .expect("a Windows temp dir carries a drive prefix")
        );
        with_library(&library, || {
            for path in [r"\Windows\System32", drive_relative.as_str()] {
                let payload = json!({
                    "modelManifestEntry": {
                        "id": "windows-relative",
                        "downloads": [],
                        "source": { "provider": "local", "path": path },
                    }
                })
                .as_object()
                .unwrap()
                .clone();
                match RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings) {
                    Err(WorkerError::InvalidPayload(message)) => assert!(
                        message.contains("relative to the app data directory"),
                        "{path} must be refused as unanchorable, got: {message}"
                    ),
                    Err(error) => panic!("{path} must be refused as unanchorable: {error}"),
                    Ok(_) => panic!("{path} must not be admitted"),
                }
            }
        });
    }

    /// A LOADER locator must be judged as the loader will re-resolve it. Every consumer of
    /// `modelPath` / `installedPath` / `paths.*` / `components[].path` re-reads the raw string and
    /// anchors a relative one on the process cwd (`paths::normalize_absolute_path`,
    /// `model_artifacts::canonical_or_absolute`), so anchoring these on the data dir here would
    /// make the guard prove a different path than the one that gets opened — admitting a locator
    /// that names nothing (`totally/made/up/never/created`) and "proving confined" an unexpanded
    /// template literal by planting it under the data dir. Only `source.path`, which nothing opens,
    /// is data-dir anchored.
    #[test]
    fn relative_loader_locators_are_judged_where_the_loader_will_resolve_them() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        // Anchored on the data dir both of these look like plausible children of an allowed root;
        // anchored where the loader anchors them (the worker's cwd) neither is confined.
        with_library(&library, || {
            for locator in ["totally/made/up/never/created", "${HF_CACHE}/models/model"] {
                let payload = json!({
                    "modelManifestEntry": {
                        "id": "relative-locator",
                        "downloads": [],
                        "paths": { "model": locator },
                    }
                })
                .as_object()
                .unwrap()
                .clone();
                assert_confinement_rejection(
                    locator,
                    RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings),
                );
            }
        });
    }

    /// Confinement posture for the two neighbouring classes is unchanged: an HF-backed entry is
    /// judged by its downloads (its `source` provenance is never confined at all), and an entry
    /// under a declared external root still passes on its real paths.
    #[test]
    fn hf_backed_and_external_root_entries_keep_their_posture() {
        let temp = TempDir::new().unwrap();
        let (settings, library, mut payload) = installed_model(&temp);
        let outside = temp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        // Provenance on an HF-backed entry never reaches the non-HF confinement branch.
        payload["modelManifestEntry"]["source"] =
            json!({ "provider": "huggingface", "repo": "guardfx/model", "path": outside });
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
        });

        let comfy = temp.path().join("comfy/models/checkpoints");
        std::fs::create_dir_all(&comfy).unwrap();
        let mut external = settings;
        external.external_model_roots = vec![temp.path().join("comfy")];
        let payload = json!({
            "modelManifestEntry": {
                "id": "comfy",
                "downloads": [],
                "paths": { "model": comfy },
            }
        })
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &external).unwrap();
        });
    }

    #[test]
    fn training_dry_run_is_model_free_but_a_real_run_is_not() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let dry = json!({ "dryRun": true }).as_object().unwrap().clone();
        RuntimeSourceGuard::begin(&JobType::LoraTrain, &dry, &settings).unwrap();
        RuntimeSourceGuard::begin(&JobType::ControlTraining, &dry, &settings).unwrap();
        let real = JsonObject::new();
        assert!(RuntimeSourceGuard::begin(&JobType::LoraTrain, &real, &settings).is_err());
    }

    // sc-24829: only a trigger-only caption job is model-free; a default or subject-only caption
    // job still loads JoyCaption and must carry its model source.
    #[test]
    fn trigger_only_caption_is_model_free_but_other_caption_modes_are_not() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let trigger_only = json!({ "mode": "triggerOnly" })
            .as_object()
            .unwrap()
            .clone();
        RuntimeSourceGuard::begin(&JobType::TrainingCaption, &trigger_only, &settings).unwrap();
        for mode in ["default", "subjectOnly"] {
            let captioned = json!({ "mode": mode }).as_object().unwrap().clone();
            assert!(
                RuntimeSourceGuard::begin(&JobType::TrainingCaption, &captioned, &settings)
                    .is_err(),
                "{mode} caption jobs load a model"
            );
        }
        // The exemption is keyed on the caption job type, not on the bare field.
        assert!(RuntimeSourceGuard::begin(&JobType::LoraTrain, &trigger_only, &settings).is_err());
    }
}
