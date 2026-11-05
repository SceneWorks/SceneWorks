//! Plan/catalog preflight and compilation against the API host.

use super::*;

/// Bytes per GB in every memory comparison the harness makes: GiB. `GET /api/v1/host-capabilities`
/// reports `memoryGb` as the worker's `memoryTotalMb / 1024`, and the manifests' `minMemoryGb` are
/// written in the same base, so `limits.maxMemoryGb` is a GiB budget and an observed peak in bytes
/// has to be divided by 1024^3 to be compared with it.
pub(crate) const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

/// One attempt's observed memory peak, and which signal it came from.
#[derive(Debug, Clone, Default)]
pub(super) struct MemoryObservation {
    /// The peak in GiB — what `limits.maxMemoryGb` is compared against.
    pub(super) gb: Option<f64>,
    /// The peak as a percentage of host memory, when the source expressed one.
    pub(super) pct: Option<f64>,
    pub(super) source: Option<String>,
}

/// Read an attempt's peak memory, preferring the real production signal.
///
/// `GenerationMetrics.peakMemoryBytes` (MLX `get_peak_memory` on macOS, the nvidia-smi high-water
/// mark on candle) is the only measured peak a shipped worker reports: EVERY `ProgressRequest` in
/// `sceneworks-worker` sets `peakGpuMemoryPct: None`, so the job snapshot's field is always null on
/// a real render and is kept here only as a last resort for a future worker that fills it.
pub(super) fn memory_observation(
    metrics: Option<&Value>,
    view: &JobView,
    host_memory_gb: Option<f64>,
) -> MemoryObservation {
    // A zero peak is "nothing was measured", not "this run used no memory" — the export job's own
    // metrics row on a real run is exactly that — so it falls through to the next source.
    let number = |value: Option<&Value>| {
        value
            .and_then(Value::as_f64)
            .filter(|number| number.is_finite() && *number > 0.0)
    };
    let metrics_pct = number(metrics.and_then(|metrics| metrics.get("peakMemoryPct")));
    if let Some(bytes) = number(metrics.and_then(|metrics| metrics.get("peakMemoryBytes"))) {
        return MemoryObservation {
            gb: Some(bytes / BYTES_PER_GB),
            pct: metrics_pct,
            source: Some("metrics.peakMemoryBytes".to_owned()),
        };
    }
    if let (Some(pct), Some(host)) = (metrics_pct, host_memory_gb) {
        return MemoryObservation {
            gb: Some(host * pct / 100.0),
            pct: Some(pct),
            source: Some("metrics.peakMemoryPct".to_owned()),
        };
    }
    match (view.peak_gpu_memory_pct, host_memory_gb) {
        (Some(pct), Some(host)) => MemoryObservation {
            gb: Some(host * pct / 100.0),
            pct: Some(pct),
            source: Some("job.peakGpuMemoryPct".to_owned()),
        },
        (Some(pct), None) => MemoryObservation {
            gb: None,
            pct: Some(pct),
            source: Some("job.peakGpuMemoryPct".to_owned()),
        },
        _ => MemoryObservation::default(),
    }
}

/// What the run learned about the host and the worker that will render. Shared with the planner
/// (sc-22713), which judges a plan against the same host rather than against this process.
#[derive(Debug, Clone, Default)]
pub(crate) struct HostFacts {
    /// The API HOST's platform (`std::env::consts::OS` spelling), from
    /// `GET /api/v1/host-capabilities`. `--api` may point at another machine, so this — not
    /// `cfg!(target_os)` — decides the lane whose `minMemoryGb` the plan is checked against, the
    /// platform the route's own reachability gate is judged on, and the hardware the record claims.
    pub(super) platform: Option<String>,
    pub(super) host_memory_gb: Option<f64>,
    pub(super) video_worker_id: Option<String>,
    pub(super) video_gpu_name: Option<String>,
    pub(super) export_worker: bool,
    /// Worker rows that advertise `video_generate` / `timeline_export` but are NOT live, so a
    /// refusal can name the stale row that would have fooled a capability-only check.
    pub(super) stale_video_workers: Vec<String>,
    pub(super) stale_export_workers: Vec<String>,
}

/// Worker statuses under which a registered row can actually claim a job. A row that still
/// advertises a capability with `status: "offline"` is a worker that left; counting it would queue
/// work nothing will ever claim — the sc-22714 smoke did exactly that against a data dir seeded
/// from an earlier run, and the review preflight learned the lesson first. Every preflight in the
/// harness (run, planner, review) now shares [`live_worker_advertising`] (sc-22715).
pub const LIVE_STATUSES: &[&str] = &["idle", "busy"];

/// The registered workers advertising `capability`, split into the first LIVE one and the rows
/// that advertise it but are not live (`"<id> (<status>)"`, for a refusal to name).
pub(crate) struct WorkerAdvert<'a> {
    pub(crate) live: Option<&'a Value>,
    pub(crate) stale: Vec<String>,
}

/// One rule for "is there a worker that will claim this job": a row counts only while its status
/// is one of [`LIVE_STATUSES`]. Shared by the run preflight (`video_generate`, `timeline_export`),
/// the planner preflight (`prompt_refine`) and the review preflight (`image_vqa`), so no preflight
/// can drift back to counting by capability alone.
pub(crate) fn live_worker_advertising<'a>(
    workers: &'a Value,
    capability: &str,
) -> WorkerAdvert<'a> {
    let advertises = |worker: &Value| {
        worker
            .get("capabilities")
            .and_then(Value::as_array)
            .is_some_and(|caps| caps.iter().any(|cap| cap == capability))
    };
    let live = |worker: &Value| {
        worker
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| LIVE_STATUSES.contains(&status))
    };
    let rows: Vec<&Value> = workers.as_array().into_iter().flatten().collect();
    WorkerAdvert {
        live: rows
            .iter()
            .copied()
            .find(|worker| advertises(worker) && live(worker)),
        stale: rows
            .iter()
            .filter(|worker| advertises(worker) && !live(worker))
            .map(|worker| {
                format!(
                    "{} ({})",
                    worker.get("id").and_then(Value::as_str).unwrap_or("?"),
                    worker.get("status").and_then(Value::as_str).unwrap_or("?")
                )
            })
            .collect(),
    }
}

/// The phrase a refusal appends when stale rows advertise the capability nobody live does.
pub(crate) fn stale_workers_detail(stale: &[String]) -> String {
    if stale.is_empty() {
        String::new()
    } else {
        format!(
            " ({} advertise(s) it but is not live: {})",
            stale.len(),
            stale.join(", ")
        )
    }
}

impl HostFacts {
    /// The lane the render host reads its memory minimum from, falling back to this process's own
    /// platform only when the API reports none.
    pub(crate) fn lane(&self) -> ModelLane {
        match self.platform.as_deref() {
            Some(platform) => ModelLane::for_platform(platform),
            None => ModelLane::for_current_platform(),
        }
    }

    pub(crate) fn platform_or_local(&self) -> &str {
        self.platform.as_deref().unwrap_or(std::env::consts::OS)
    }

    /// The memory the API host reports, in GiB, when any registered worker reports one.
    pub(crate) fn host_memory_gb(&self) -> Option<f64> {
        self.host_memory_gb
    }
}

pub(super) async fn discover_host(client: &Client<'_>) -> Result<HostFacts, HarnessError> {
    let workers = client.expect_ok("GET", "/api/v1/workers", None).await?;
    let mut facts = HostFacts::default();
    // Live rows only (sc-22715): a stale `offline` row still advertising `video_generate` is not
    // a GPU worker, and a run that counted it would dispatch a render nothing claims.
    let video = live_worker_advertising(&workers, "video_generate");
    if let Some(worker) = video.live {
        facts.video_worker_id = worker.get("id").and_then(Value::as_str).map(str::to_owned);
        facts.video_gpu_name = worker
            .get("gpuName")
            .and_then(Value::as_str)
            .map(str::to_owned);
    }
    facts.stale_video_workers = video.stale;
    let export = live_worker_advertising(&workers, "timeline_export");
    facts.export_worker = export.live.is_some();
    facts.stale_export_workers = export.stale;
    let host = client
        .expect_ok("GET", "/api/v1/host-capabilities", None)
        .await?;
    facts.platform = host
        .get("platform")
        .and_then(Value::as_str)
        .filter(|platform| !platform.trim().is_empty())
        .map(str::to_owned);
    facts.host_memory_gb = host
        .get("memoryGb")
        .and_then(Value::as_f64)
        .or_else(|| host.get("unifiedMemoryGb").and_then(Value::as_f64))
        .or_else(|| host.get("gpuMemoryGb").and_then(Value::as_f64));
    Ok(facts)
}

/// The catalog entry for `model_id` as `/api/v1/models` serves it (manifest fields plus the
/// derived install state), or `None` when the catalog has no such model.
pub(super) async fn resolve_model_entry(
    client: &Client<'_>,
    model_id: &str,
) -> Result<Option<JsonObject<String, Value>>, HarnessError> {
    let catalog = client.expect_ok("GET", "/api/v1/models", None).await?;
    Ok(catalog
        .as_array()
        .into_iter()
        .flatten()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
        .and_then(Value::as_object)
        .cloned())
}

/// The catalog entries one plan's shots may dispatch as (sc-23402): the declared model, plus the
/// family's reference partition when some shot of THIS plan binds reference roles.
///
/// It owns the entries because `Prepared` outlives the request that fetched them;
/// [`PlanCatalog::entries`] is the borrowed view every validator and the compiler read.
pub(crate) struct PlanCatalog {
    pub(super) model_id: String,
    pub(super) base: Option<JsonObject<String, Value>>,
    /// The reference partition this plan needs, when it needs one. `Some(id)` with a `None`
    /// `reference` is a partition the catalog does not serve — named per shot by the validator.
    pub(super) reference_id: Option<String>,
    pub(super) reference: Option<JsonObject<String, Value>>,
}

impl PlanCatalog {
    /// The borrowed view, or `None` when the plan's own model is not in the catalog at all (which
    /// [`model_entry_findings`] has already reported).
    pub(crate) fn entries(&self) -> Option<ModelEntries<'_>> {
        let base = self.base.as_ref()?;
        Some(ModelEntries::with_reference_partition(
            &self.model_id,
            base,
            self.reference_id.as_deref().zip(self.reference.as_ref()),
        ))
    }

    pub(crate) fn base_entry(&self) -> Option<&JsonObject<String, Value>> {
        self.base.as_ref()
    }

    /// The family's reference partition as the catalog actually SERVES it (sc-23405), or `None`
    /// when this catalog was not asked for one or the API has no such entry.
    ///
    /// This is the fact the planner's capability envelope widens on: the reference entry's own
    /// `capabilities` and `limits.maxReferenceAssets` are what a reference shot dispatches against,
    /// and an envelope built from a partition the catalog does not serve would hand the planner a
    /// mode whose every use is refused by `missing_partition_finding` on each repair round.
    pub(crate) fn reference_entry(&self) -> Option<(&str, &JsonObject<String, Value>)> {
        self.reference_id.as_deref().zip(self.reference.as_ref())
    }
}

/// The entry-level gate — catalog presence, video type, install state, platform reachability.
///
/// The base partition is always gated. The reference partition is gated only when
/// `gate_reference` says THIS invocation will load it (sc-23402): the reference DiT is a separate
/// 18.78 GB download with its own install state, so a run that discovers it uninstalled at
/// dispatch has already spent the base checkpoint's load — but references are OPTIONAL (E1), and a
/// caller that will never load those weights must not be asked to have them on disk.
///
/// `validate`/`run` pass whether a SELECTED shot resolves to it. The planner (sc-23405) has no
/// draft yet, so it passes the PACK-AND-CATALOG gate instead —
/// `PlannerCapabilities::offers_references`, true only when the catalog serves a reference
/// partition AND the chosen pack approves at least one reference the mode could bind
/// ([`sceneworks_core::film_plan::BINDABLE_REFERENCE_KINDS`]). That is exactly the condition under
/// which the envelope offers `reference_to_video` at all, so the weights are demanded only when
/// the plan the planner is allowed to draft could need them.
///
/// A partition the catalog does not serve at all is deliberately NOT reported here: the per-shot
/// validator names it together with the shot that needs it, which is the actionable form.
pub(crate) fn catalog_entry_findings(
    catalog: &PlanCatalog,
    tier: Option<&str>,
    require_installed: bool,
    facts: &HostFacts,
    gate_reference: bool,
) -> Vec<PlanDiagnostic> {
    let mut findings = model_entry_findings(
        &catalog.model_id,
        catalog.base_entry(),
        tier,
        require_installed,
        facts,
    );
    if !gate_reference {
        return findings;
    }
    if let (Some(reference_id), Some(reference)) =
        (catalog.reference_id.as_deref(), catalog.reference.as_ref())
    {
        findings.extend(model_entry_findings(
            reference_id,
            Some(reference),
            tier,
            require_installed,
            facts,
        ));
    }
    findings
}

/// Whether some SELECTED shot resolves to the family's reference partition — which is what decides
/// whether that partition's install state and platform reachability are gated (sc-23402 review).
///
/// Plan-level validation of the DOCUMENT stays whole-plan: a `--shots SH020` run is still refused
/// for a malformed SH010, and the reference entry is still resolved so SH010's declared caps can be
/// judged. Only the install gate follows the selection, because only the selected shots are
/// dispatched and only their partitions are ever loaded.
pub(super) fn selection_needs_reference(
    plan: &ProductionPlan,
    selection: Option<&[String]>,
) -> bool {
    plan.shots
        .iter()
        .filter(|shot| match selection {
            Some(ids) => ids.iter().any(|id| id == &shot.id),
            None => true,
        })
        .any(|shot| !shot.conditioning.reference_roles.is_empty())
}

/// Resolve the catalog entries a plan or brief on `model_id` may dispatch as, against a bare
/// transport. `include_reference` asks for the family's reference partition too — what the PLANNER
/// needs, since the draft it is about to produce may bind reference roles.
pub(crate) async fn plan_catalog_for(
    transport: &dyn ApiTransport,
    model_id: &str,
    include_reference: bool,
) -> Result<PlanCatalog, HarnessError> {
    let control = RunControl::new();
    let client = Client {
        transport,
        control: &control,
    };
    resolve_catalog(&client, model_id, include_reference).await
}

/// The ids `GET /api/v1/loras` reports INSTALLED on the API host (sc-23406).
///
/// On the transport rather than off the local catalog for the same reason the model entries are:
/// `--api` may point at another machine, and which adapters are on ITS disk is the only answer
/// that decides whether a plan naming one will enqueue.
pub(crate) async fn installed_lora_ids(
    transport: &dyn ApiTransport,
) -> Result<Vec<String>, HarnessError> {
    let control = RunControl::new();
    let client = Client {
        transport,
        control: &control,
    };
    let catalog = client.expect_ok("GET", "/api/v1/loras", None).await?;
    Ok(catalog
        .as_array()
        .into_iter()
        .flatten()
        .filter(|lora| lora.get("installState").and_then(Value::as_str) == Some("installed"))
        .filter_map(|lora| lora.get("id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect())
}

/// The family's reference partition when THIS plan needs it: some shot binds reference roles and
/// the declared model's family splits reference conditioning into a second catalog entry. A plan
/// whose shots bind none never resolves it, so a run that needs only the base checkpoint neither
/// demands the reference weights be installed nor prices their memory.
pub(super) fn plan_reference_partition(plan: &ProductionPlan) -> Option<String> {
    if !plan
        .shots
        .iter()
        .any(|shot| !shot.conditioning.reference_roles.is_empty())
    {
        return None;
    }
    film_plan::reference_partition_for(&plan.model.id).map(str::to_owned)
}

/// Resolve every catalog entry `plan` may dispatch as.
pub(super) async fn resolve_plan_catalog(
    client: &Client<'_>,
    plan: &ProductionPlan,
) -> Result<PlanCatalog, HarnessError> {
    resolve_catalog(
        client,
        &plan.model.id,
        plan_reference_partition(plan).is_some(),
    )
    .await
}

pub(super) async fn resolve_catalog(
    client: &Client<'_>,
    model_id: &str,
    include_reference: bool,
) -> Result<PlanCatalog, HarnessError> {
    let base = resolve_model_entry(client, model_id).await?;
    let reference_id = include_reference
        .then(|| film_plan::reference_partition_for(model_id))
        .flatten()
        .map(str::to_owned);
    let reference = match reference_id.as_deref() {
        Some(id) => resolve_model_entry(client, id).await?,
        None => None,
    };
    Ok(PlanCatalog {
        model_id: model_id.to_owned(),
        base,
        reference_id,
        reference,
    })
}

/// Whether the catalog reports the requested tier (or, with no tier named, the model) installed.
pub(crate) fn model_tier_installed(entry: &JsonObject<String, Value>, tier: Option<&str>) -> bool {
    if let Some(tier) = tier {
        if let Some(variants) = entry.get("variants").and_then(Value::as_array) {
            if let Some(variant) = variants
                .iter()
                .find(|variant| variant.get("variant").and_then(Value::as_str) == Some(tier))
            {
                return variant
                    .get("installed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
            }
        }
    }
    entry.get("installState").and_then(Value::as_str) == Some("installed")
}

/// The manifest download row that carries the requested tier's primary weights, for the record.
pub(super) fn primary_weights(
    entry: &JsonObject<String, Value>,
    tier: Option<&str>,
) -> Option<Value> {
    let downloads = entry.get("downloads").and_then(Value::as_array)?;
    let is_primary = |row: &&Value| {
        !row.get("coRequisite")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    };
    let row = match tier {
        Some(tier) => downloads
            .iter()
            .filter(is_primary)
            .find(|row| row.get("variant").and_then(Value::as_str) == Some(tier)),
        None => downloads
            .iter()
            .filter(is_primary)
            .find(|row| row.get("default").and_then(Value::as_bool).unwrap_or(false))
            .or_else(|| downloads.iter().find(is_primary)),
    }?;
    Some(json!({
        "provider": row.get("provider"),
        "repo": row.get("repo"),
        "revision": row.get("revision"),
        "variant": row.get("variant"),
        "files": row.get("files"),
    }))
}

/// [`primary_weights`] for EVERY partition the SELECTED shots resolve to, keyed by catalog model id
/// (sc-23402 review).
///
/// `ModelRecord::weights` names the declared model's download row only, so on a mixed run nothing
/// recorded the `transformer_ref` rows that produced the reference takes. Keyed by partition so a
/// reader can pair a take's `model` with the files behind it. A partition the catalog does not
/// serve contributes no entry — the run is already refused for it by name.
pub(super) fn partition_weights(
    plan: &ProductionPlan,
    entries: &ModelEntries<'_>,
    tier: Option<&str>,
    selection: &[String],
) -> BTreeMap<String, Value> {
    let mut weights = BTreeMap::new();
    let selected = ProductionPlan {
        shots: plan
            .shots
            .iter()
            .filter(|shot| selection.iter().any(|id| id == &shot.id))
            .cloned()
            .collect(),
        ..plan.clone()
    };
    for partition in entries.partitions_used(&selected) {
        let (id, entry) = entries.resolve_shot_partition_entry(&partition);
        if let Some(row) = entry.and_then(|entry| primary_weights(entry, tier)) {
            weights.insert(id.to_owned(), row);
        }
    }
    weights
}

/// The compiled requests this run dispatches: the document beside the plan when there is one, else
/// the plan compiled in memory with its authored prompts (the hand-authored path). Either way the
/// job bodies come from [`CompiledRequest::to_job_body`], so what a reviewer reads in
/// `compiled.json` is what the API receives.
pub(super) fn compiled_for_run(
    plan: &ProductionPlan,
    pack: &ReferencePack,
    entries: &ModelEntries<'_>,
    lane: ModelLane,
    plan_sha256: &str,
    supplied: Option<CompiledPlan>,
) -> Result<CompiledPlan, HarnessError> {
    match supplied {
        Some(compiled) => {
            // Two questions, in order: were these requests compiled from THIS plan, and do they
            // still say what compiling it would say? The second is what keeps a hand-edited
            // `compiled.json` — the document every dispatched field but the prompt is read from —
            // from reaching the route unjudged, since `validate_all` only ever reads the plan.
            let mut findings = compiled.staleness_findings(plan, plan_sha256, pack);
            if findings.is_empty() {
                findings = compiled.conformance_findings(plan, pack, entries, lane);
            }
            if findings.is_empty() {
                Ok(compiled)
            } else {
                Err(HarnessError::Validation(findings))
            }
        }
        None => compile_plan(
            plan,
            pack,
            &CompileInputs {
                entries,
                lane: lane.manifest_key(),
                plan_sha256,
                compiled_at: &utc_now(),
                refined_prompts: &BTreeMap::new(),
            },
        )
        .map_err(HarnessError::Validation),
    }
}

/// Shared in-memory preflight for the Film workspace and the file-oriented harness. It resolves
/// the same live catalog, host lane, worker reachability, LoRA inventory and compiler used by an
/// actual run, so the browser cannot present a second, approximate render contract.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FilmDocumentPreflight {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_revision: Option<u32>,
    pub valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<PlannerCapabilities>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compiled: Option<CompiledPlan>,
    pub findings: Vec<PlanDiagnostic>,
}

pub(crate) async fn preflight_documents(
    transport: &dyn ApiTransport,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    supplied: Option<CompiledPlan>,
    selection: Option<&[String]>,
    require_installed: bool,
) -> Result<FilmDocumentPreflight, HarnessError> {
    let mut findings = film_plan::validate_all(plan, pack, None, None);
    findings.extend(selection_findings(plan, selection));
    if !findings.is_empty() {
        return Ok(FilmDocumentPreflight {
            draft_revision: None,
            valid: false,
            capabilities: None,
            compiled: None,
            findings,
        });
    }

    let control = RunControl::new();
    let client = Client {
        transport,
        control: &control,
    };
    let facts = discover_host(&client).await?;
    let catalog = resolve_plan_catalog(&client, plan).await?;
    findings.extend(model_findings(
        plan,
        pack,
        &catalog,
        require_installed,
        &facts,
        selection,
    ));
    if findings.is_empty() {
        findings.extend(host_findings(plan, &facts, false));
    }

    let lane = facts
        .platform
        .as_deref()
        .map(ModelLane::for_platform)
        .unwrap_or_else(ModelLane::for_current_platform);
    let installed = installed_lora_ids(transport).await?;
    let capabilities = catalog.base_entry().map(|base| {
        let mut capabilities = capabilities_for(&plan.model, base, lane);
        if let Some((_, reference)) = catalog.reference_entry() {
            capabilities = capabilities.with_reference_partition(reference);
        }
        capabilities
            .narrowed_to_pack(pack)
            .with_installed_turbo_loras(&installed)
    });
    if !findings.is_empty() {
        return Ok(FilmDocumentPreflight {
            draft_revision: None,
            valid: false,
            capabilities,
            compiled: None,
            findings,
        });
    }

    let plan_sha256 = sceneworks_core::film_compile::production_plan_sha256(plan)
        .map_err(|error| HarnessError::Io(error.to_string()))?;
    let entries = catalog
        .entries()
        .expect("model findings are empty only with a catalog entry");
    match compiled_for_run(plan, pack, &entries, lane, &plan_sha256, supplied) {
        Ok(compiled) => Ok(FilmDocumentPreflight {
            draft_revision: None,
            valid: true,
            capabilities,
            compiled: Some(compiled),
            findings: Vec::new(),
        }),
        Err(HarnessError::Validation(findings)) => Ok(FilmDocumentPreflight {
            draft_revision: None,
            valid: false,
            capabilities,
            compiled: None,
            findings,
        }),
        Err(error) => Err(error),
    }
}

/// Read, validate and (optionally) check the documents against the live catalog without creating
/// anything. What `run` does before its first write, exposed for the `validate` subcommand.
pub async fn validate(
    transport: Option<&dyn ApiTransport>,
    options: &RunOptions,
) -> Result<(ProductionPlan, ReferencePack), HarnessError> {
    let plan = film_plan::read_plan_file(&options.plan_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let pack = film_plan::read_reference_pack_file(&options.reference_pack_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let pack_dir = options
        .reference_pack_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut findings = film_plan::validate_all(&plan, &pack, Some(&pack_dir), None);
    findings.extend(selection_findings(&plan, options.shot_ids.as_deref()));
    // Compiled requests are checked against the plan they claim: a plan edited after the compile
    // would otherwise dispatch the prompts it no longer holds.
    let compiled = read_compiled_for(options)?;
    if let Some((compiled, path)) = compiled.as_ref() {
        let plan_bytes = std::fs::read(&options.plan_path)?;
        let mut stale = compiled.staleness_findings(&plan, &sha256_hex(&plan_bytes), &pack);
        if !stale.is_empty() {
            stale.insert(0, compiled_document_header(path));
            findings.append(&mut stale);
        }
    }
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    if let Some(transport) = transport {
        let control = RunControl::default();
        let client = Client {
            transport,
            control: &control,
        };
        // The host comes FIRST: its platform decides which lane's `minMemoryGb` the plan is
        // checked against and which platform the route's reachability gate is judged on, so the
        // model checks cannot run before it is known.
        let facts = discover_host(&client).await?;
        let catalog = resolve_plan_catalog(&client, &plan).await?;
        let mut findings = model_findings(
            &plan,
            &pack,
            &catalog,
            options.require_installed,
            &facts,
            options.shot_ids.as_deref(),
        );
        if findings.is_empty() {
            findings.extend(host_findings(&plan, &facts, options.export));
        }
        // The document that is DISPATCHED, judged against the installed capabilities — not just
        // the plan it was compiled from. `--compiled FILE` takes a document from any path and
        // `execute_run` reads every field but the prompt straight out of it, so this is the only
        // place a hand-edited request meets the model's declared menus (sc-22713 review).
        if findings.is_empty() {
            if let (Some((compiled, path)), Some(entries)) = (compiled.as_ref(), catalog.entries())
            {
                let mut conformance =
                    compiled.conformance_findings(&plan, &pack, &entries, facts.lane());
                if !conformance.is_empty() {
                    findings.push(compiled_document_header(path));
                    findings.append(&mut conformance);
                }
            }
        }
        if !findings.is_empty() {
            return Err(HarnessError::Validation(findings));
        }
    }
    Ok((plan, pack))
}

/// The finding that says the findings after it are about the compiled document, not the plan.
pub(super) fn compiled_document_header(path: &Path) -> PlanDiagnostic {
    PlanDiagnostic::plan(
        "compiled",
        format!(
            "the compiled requests at {} do not match this plan; every finding below is about \
             that document",
            path.display()
        ),
    )
}

/// The compiled requests this run should use, with the path they came from: the explicit
/// `--compiled` document, else a `compiled.json` sitting beside the plan, else nothing.
pub(super) fn read_compiled_for(
    options: &RunOptions,
) -> Result<Option<(CompiledPlan, PathBuf)>, HarnessError> {
    let path = match &options.compiled_path {
        Some(path) => path.clone(),
        None => {
            let sibling = options
                .plan_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
                .join("compiled.json");
            if !sibling.is_file() {
                return Ok(None);
            }
            sibling
        }
    };
    let compiled = crate::film_planner::read_compiled_file(&path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    Ok(Some((compiled, path)))
}

pub(super) fn selection_findings(
    plan: &ProductionPlan,
    selection: Option<&[String]>,
) -> Vec<PlanDiagnostic> {
    let Some(selection) = selection else {
        return Vec::new();
    };
    let mut findings = Vec::new();
    if selection.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "selection",
            "at least one shot id must be selected",
        ));
    }
    for id in selection {
        if !plan.shots.iter().any(|shot| &shot.id == id) {
            findings.push(PlanDiagnostic::plan(
                "selection",
                format!(
                    "selected shot {id:?} is not in plan {:?} (shots: {})",
                    plan.id,
                    plan.shots
                        .iter()
                        .map(|shot| shot.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
    }
    findings
}

/// The route's own platform-reachability gate, run before dispatch instead of being discovered as
/// a 400 at enqueue: `POST /api/v1/video/jobs` calls exactly this function
/// ([`crate::generation::ensure_video_model_available_on_platform`]) on the resolved manifest
/// entry, so a plan this validator passes cannot be refused for the reason it checks.
pub(super) fn platform_reachability_finding(
    model_id: &str,
    entry: &JsonObject<String, Value>,
    platform: &str,
) -> Option<PlanDiagnostic> {
    let value = Value::Object(entry.clone());
    crate::generation::ensure_video_model_available_on_platform(model_id, &value, platform)
        .err()
        .map(|error| PlanDiagnostic::plan("model.id", error.detail))
}

/// The route's reference-payload gate, run before dispatch on the payload shape each shot will
/// produce. Placeholder ids stand in for the asset ids the import has not created yet: they are
/// non-blank and untrimmed-free by construction, so the only questions this can answer are the ones
/// that depend on the PLAN — how many references a shot carries, and the model/mode spelling.
pub(super) fn reference_payload_findings(
    plan: &ProductionPlan,
    pack: &ReferencePack,
    entries: &ModelEntries<'_>,
) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    for shot in &plan.shots {
        // The gate runs on the payload this shot will ACTUALLY produce, which on a split family
        // names the resolved partition and carries that entry's caps (sc-23402). A shot whose
        // partition is not in the catalog is already a finding from the validator.
        let (partition, partition_entry) = entries.resolve_shot(shot);
        let Some(partition_entry) = partition_entry else {
            continue;
        };
        let value = Value::Object(partition_entry.clone());
        let mut payload = JsonObject::new();
        payload.insert("model".to_owned(), json!(partition.model_id));
        payload.insert("mode".to_owned(), json!(shot.conditioning.mode));
        // Match CompiledRequest::job_body's keyframe carriers too. This gate now checks source
        // presence as well as reference counts; omitting a declared first frame would reject a
        // valid image-to-video plan before its assets can be imported.
        if shot.conditioning.first_frame_role.is_some() {
            payload.insert("sourceAssetId".to_owned(), json!("placeholder_first_frame"));
        }
        if shot.conditioning.last_frame_role.is_some() {
            payload.insert(
                "lastFrameAssetId".to_owned(),
                json!("placeholder_last_frame"),
            );
        }
        // One placeholder per PICTURE, not per role (sc-24024): roles sharing a pack file are
        // supplied as one image, so a shot binding ten roles across nine files posts nine ids. The
        // count comes from `shot_reference_pictures` — the same function the dispatch builds the
        // real list with — so this gate judges the length the route will actually receive.
        let pictures = sceneworks_core::film_compile::shot_reference_pictures(
            &shot.conditioning.reference_roles,
            pack,
        );
        payload.insert(
            "referenceAssetIds".to_owned(),
            Value::Array(
                (0..pictures.len())
                    .map(|index| json!(format!("placeholder_reference_{index}")))
                    .collect(),
            ),
        );
        if let Err(error) = crate::validate_video_reference_asset_ids_payload(&payload, &value) {
            findings.push(PlanDiagnostic::shot(
                &shot.id,
                "conditioning.referenceRoles",
                error.detail,
            ));
        }
    }
    findings
}

/// Findings about the catalog ENTRY: present, a video model, installed, and reachable on the render
/// host's platform. Shared with the planner (sc-22713), which must refuse an absent, uninstalled or
/// unreachable model before it spends a decode on a plan that could never be dispatched — rather
/// than keeping a second copy of these rules.
pub(crate) fn model_entry_findings(
    model_id: &str,
    entry: Option<&JsonObject<String, Value>>,
    tier: Option<&str>,
    require_installed: bool,
    facts: &HostFacts,
) -> Vec<PlanDiagnostic> {
    let Some(entry) = entry else {
        return vec![PlanDiagnostic::plan(
            "model.id",
            format!("{model_id:?} is not in this API's model catalog"),
        )];
    };
    let mut findings = Vec::new();
    if entry.get("type").and_then(Value::as_str) != Some("video") {
        findings.push(PlanDiagnostic::plan(
            "model.id",
            format!("{model_id:?} is not a video model"),
        ));
    }
    if require_installed && !model_tier_installed(entry, tier) {
        findings.push(PlanDiagnostic::plan(
            "model.tier",
            format!(
                "{model_id}{} is not installed on this host (catalog installState is not \
                 \"installed\"); download it in the Model Manager first",
                tier.map(|tier| format!(" tier {tier}")).unwrap_or_default()
            ),
        ));
    }
    findings.extend(platform_reachability_finding(
        model_id,
        entry,
        facts.platform_or_local(),
    ));
    findings
}

/// Takes the PACK because two of the rules below count the images a shot SUPPLIES rather than the
/// roles it names, and only the pack says which of a shot's roles share one file (sc-24024).
pub(super) fn model_findings(
    plan: &ProductionPlan,
    pack: &ReferencePack,
    catalog: &PlanCatalog,
    require_installed: bool,
    facts: &HostFacts,
    selection: Option<&[String]>,
) -> Vec<PlanDiagnostic> {
    let mut findings = catalog_entry_findings(
        catalog,
        plan.model.tier.as_deref(),
        require_installed,
        facts,
        selection_needs_reference(plan, selection),
    );
    let Some(entries) = catalog.entries() else {
        return findings;
    };
    findings.extend(film_plan::validate_plan_against_model(
        plan,
        pack,
        &entries,
        facts.lane(),
    ));
    findings.extend(reference_payload_findings(plan, pack, &entries));
    findings
}

pub(super) fn host_findings(
    plan: &ProductionPlan,
    facts: &HostFacts,
    export: bool,
) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if facts.video_worker_id.is_none() {
        findings.push(PlanDiagnostic::plan(
            "model.id",
            format!(
                "no live registered worker advertises video_generate{}; start the GPU worker \
                 (SCENEWORKS_WORKER_ONLY=1) and wait for it to register, or clear a stale worker \
                 row that is shadowing it",
                stale_workers_detail(&facts.stale_video_workers)
            ),
        ));
    }
    if export && !facts.export_worker {
        findings.push(PlanDiagnostic::plan(
            "export",
            format!(
                "no live registered worker advertises timeline_export{}; run the API with \
                 SCENEWORKS_RUN_UTILITY_INPROCESS=1 or start a utility worker",
                stale_workers_detail(&facts.stale_export_workers)
            ),
        ));
    }
    match facts.host_memory_gb {
        Some(host) if plan.limits.max_memory_gb > host => findings.push(PlanDiagnostic::plan(
            "limits.maxMemoryGb",
            format!(
                "budget {} GB exceeds the {host:.1} GB the registered worker reports for this host",
                plan.limits.max_memory_gb
            ),
        )),
        Some(_) => {}
        None => findings.push(PlanDiagnostic::plan(
            "limits.maxMemoryGb",
            "no registered worker reports host memory, so the memory budget cannot be checked \
             before dispatch",
        )),
    }
    findings
}

/// The model entry, host facts and fps a run needs once the documents themselves are valid.
pub(super) struct Prepared {
    pub(super) catalog: PlanCatalog,
    pub(super) facts: HostFacts,
    pub(super) fps: u32,
}

impl Prepared {
    /// The plan's declared model entry. Present by construction: `prepare` only builds a
    /// `Prepared` once the model findings are empty, which needs the entry.
    pub(super) fn base_entry(&self) -> &JsonObject<String, Value> {
        self.catalog
            .base_entry()
            .expect("model findings are empty only with an entry")
    }

    pub(super) fn entries(&self) -> ModelEntries<'_> {
        self.catalog
            .entries()
            .expect("model findings are empty only with an entry")
    }
}

/// Resolve the catalog entry and the host facts and judge the plan against both. `Ok(Err(findings))`
/// is a refusal: the caller writes a `rejected` record and creates nothing.
///
/// `selection` is the shots THIS controller will dispatch (`None` = the whole plan). It scopes the
/// reference partition's install gate only — the plan itself is still judged whole (sc-23402).
pub(super) async fn prepare(
    client: &Client<'_>,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    export: bool,
    require_installed: bool,
    selection: Option<&[String]>,
) -> Result<Result<Prepared, Vec<PlanDiagnostic>>, HarnessError> {
    let facts = discover_host(client).await?;
    let catalog = resolve_plan_catalog(client, plan).await?;
    let mut findings = model_findings(plan, pack, &catalog, require_installed, &facts, selection);
    if findings.is_empty() {
        findings.extend(host_findings(plan, &facts, export));
    }
    if !findings.is_empty() {
        return Ok(Err(findings));
    }
    let prepared = Prepared {
        catalog,
        facts,
        fps: 0,
    };
    let fps =
        film_plan::plan_fps(plan, prepared.base_entry()).expect("validated against the model");
    Ok(Ok(Prepared { fps, ..prepared }))
}
