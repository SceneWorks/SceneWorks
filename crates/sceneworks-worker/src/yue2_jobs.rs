//! YuE2 song jobs (sc-22999, epic 22988) — the worker half of `POST /api/v1/projects/:id/yue2/jobs`.
//!
//! A YuE2 job is an `audio_generate` job whose payload carries a `yue2` block
//! (`sceneworks_core::yue2_score::jobs::Yue2JobSpec`). It runs through the registered `yue2`
//! provider's [`Generator::generate_with_report`] — never through `generate`, whose single audio
//! output cannot carry a plan-only result, the published run record or the truncation warnings.
//!
//! # One job, in order
//!
//! 1. **Re-validate** the stored block (`validate_for_execution`): a retried or duplicated payload
//!    is held to the same rules as a fresh one.
//! 2. **Eligibility at execution** (`GET /api/v1/jobs/:id/yue2-eligibility`): the licence
//!    acknowledgment must be current and a declared commercial use is refused — evaluated against
//!    the catalog and acknowledgment store as they are *now*. Nothing is loaded before this passes.
//! 3. **Resolve** the tier's weights (the pinned `bf16` original, or a locally derived `q8` / `q4`
//!    snapshot that verifies against its pins), the staged decoders, and the kind's inputs: a saved
//!    plan or source run is verified against the identity its producing job recorded; a score
//!    version is re-read and must still have the digests it was queued with.
//! 4. **Generate** on a blocking thread with `audio.artifacts { dir: <project>/yue2/runs/<runId>,
//!    resume: true }` under an exclusive OS claim on `<run dir>.claim` ([`RunClaim`]): the stable
//!    run id makes a retried job resume its own checkpoints (a duplicate gets a fresh run id), and
//!    the engine checks every recorded identity before it reuses anything. The shared keepalive watcher
//!    ([`run_blocking_with_heartbeat`]) keeps the worker heartbeat alive through long AR phases and
//!    trips the request's cancel flag — the engine's own cancel hook — on a user cancel. Engine
//!    progress (per-token plan / semantic steps, acoustic ODE steps, decoding) is coalesced onto the
//!    job progress channel.
//! 5. **Publish**: the audio becomes a project library asset whose sidecar carries the full
//!    provenance (`extra.yue2`) and usage policy (`extra.usagePolicy`); the result carries the
//!    run record, truncation, warnings and effective settings the API's terminal side effects turn
//!    into a score version (plans) or a render record (score-version renders).
//!
//! A failure posts the job `failed` **with** that provenance (what ran, on what, under which
//! policy, and the run directory it left for a resume). A cancel after this job's engine started
//! working in the run removes the run's unpublished `.partial` working directory and nothing else —
//! a source run, a plan, a published run and every original stay untouched; an earlier cancel
//! leaves a prior attempt's checkpoints for the next retry.

use super::*;

use gen_core::{
    AudioArtifacts, AudioParams, CancelFlag, GenerationMemory, GenerationOutput, GenerationReport,
    GenerationRequest, Generator, LoadSpec, OffloadPolicy as GenOffloadPolicy, Precision, Progress,
    Quant, SavedPlan, SongCover, SongCoverMode, SongCoverVoice, SongDecoder, SongParams,
    SongPlanning, TokenSampling as GenTokenSampling, WeightsSource,
};
use sceneworks_core::model_artifacts::artifact_selection::{
    derived_snapshot_state, local_derivation, local_derivation_snapshot_dir, DerivedSnapshotState,
};
use sceneworks_core::yue2_score::jobs::{
    self as contract, ComputePrecision, CoverKeep, CoverMode, Decoder, OffloadPolicy, Planning,
    Tier, TokenSampling, Yue2JobKind, Yue2JobSpec,
};
use sceneworks_core::yue2_score::store::ScoreVersionRecord;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::video_jobs::{write_wav_pcm16, AudioTrack};

const CANCEL_MESSAGE: &str = "YuE2 job canceled by user.";
/// Suffix of a run's unpublished working directory (`candle_audio_yue2::run::PARTIAL_SUFFIX`).
pub(crate) const PARTIAL_SUFFIX: &str = ".partial";
/// The claim a run holds on its working directory (`candle_audio_yue2::run::LOCK_FILE`).
pub(crate) const RUN_LOCK_FILE: &str = ".yue2-run.lock";
/// Engine progress is per token; posts are coalesced to at most one per interval.
const PROGRESS_POST_INTERVAL: Duration = Duration::from_millis(500);

/// Whether an `audio_generate` payload is a YuE2 job.
pub(crate) fn is_yue2_job(payload: &JsonObject) -> bool {
    payload.contains_key(contract::PAYLOAD_KEY)
}

/// Run a YuE2 job with the runtime's audio registry.
pub(crate) async fn run_yue2_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    run_yue2_job_using(api, settings, job, crate::inference_runtime::load_audio).await
}

/// What is known about the job so far — persisted with a failure so a failed run keeps its
/// provenance.
#[derive(Default)]
struct JobRecord {
    spec: Option<Yue2JobSpec>,
    usage_policy: Option<Value>,
    effective: Option<Value>,
    run_rel: Option<String>,
    run_dir: Option<PathBuf>,
    /// The resolved weights: `(model, decoder)` identities and the tier.
    identities: Option<(Value, Option<Value>, &'static str)>,
    /// Set by the blocking task the moment it hands the request to the engine — the only point
    /// after which the run's working directory can be this job's.
    engine_started: Arc<AtomicBool>,
    /// This job's exclusive claim on its run directory, held until the job has cleaned up.
    claim: Option<RunClaim>,
}

impl JobRecord {
    fn failed_result(&self, job: &JobSnapshot, error: &WorkerError) -> JsonObject {
        let spec = self.spec.as_ref();
        let partial = self
            .run_dir
            .as_ref()
            .map(|dir| partial_dir(dir))
            .filter(|dir| dir.is_dir());
        let block = json!({
            "status": "failed",
            "kind": spec.map(|s| s.kind.as_str()),
            "runId": spec.and_then(|s| s.run_id.clone()),
            "run": self.run_rel.as_ref().map(|dir| json!({
                "dir": dir,
                // A failed run leaves its checkpoints in the working directory; a retry of this
                // job (same run id) resumes the ones whose identities still match.
                "partial": partial.is_some(),
            })),
            "error": error.to_string(),
            // What it was running on, when it got that far. Truncation is deliberately absent:
            // a failed run never observed it.
            "model": self.identities.as_ref().map(|(model, _, _)| model.clone()),
            "decoder": self.identities.as_ref().and_then(|(_, decoder, _)| decoder.clone()),
            "tier": self.identities.as_ref().map(|(_, _, tier)| *tier),
            "effectiveSettings": self.effective,
            "sources": spec.and_then(|s| s.sources.clone()),
            "batch": spec.and_then(|s| s.batch.clone()),
            "usagePolicy": self.usage_policy.clone().or_else(|| job.payload.get("usagePolicy").cloned()),
        });
        let mut result = JsonObject::new();
        result.insert("yue2".to_owned(), block);
        result
    }
}

/// [`run_yue2_job`] with the generator loader injected, so a test drives the real job path —
/// eligibility, resolution, progress, cancel, publication — against a stub [`Generator`].
pub(crate) async fn run_yue2_job_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<()> {
    let mut record = JobRecord::default();
    match execute(api, settings, job, load_generator, &mut record).await {
        Ok(()) => Ok(()),
        Err(WorkerError::Canceled(message)) => {
            // A canceled job cleans its temporary work: the run's unpublished working directory —
            // only when this job's engine worked in it (it started generating while this job held
            // the run's claim). A cancel before that leaves a prior attempt's resumable
            // checkpoints alone; a published run, a source run or an original is never touched.
            if record.claim.is_some() && record.engine_started.load(Ordering::SeqCst) {
                if let Some(dir) = &record.run_dir {
                    remove_partial(dir);
                }
            }
            drop(record.claim.take());
            Err(WorkerError::Canceled(message))
        }
        Err(error) => {
            let result = record.failed_result(job, &error);
            let mut payload = progress_payload(
                JobStatus::Failed,
                ProgressStage::Failed,
                1.0,
                "YuE2 job failed.",
                Some(error.to_string()),
                Some(result),
                None,
            );
            payload.backend = Some(backend_label(&settings.gpu_id).to_owned());
            update_job(api, &job.id, payload).await?;
            Ok(())
        }
    }
}

fn yue2_progress(
    status: JobStatus,
    stage: ProgressStage,
    fraction: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    let mut payload = progress_payload(status, stage, fraction, message, None, result, None);
    payload.backend = Some(backend.to_owned());
    payload
}

fn parse_spec(payload: &JsonObject) -> WorkerResult<Yue2JobSpec> {
    let value = payload
        .get(contract::PAYLOAD_KEY)
        .cloned()
        .ok_or_else(|| WorkerError::InvalidPayload("the job carries no yue2 block".into()))?;
    let spec: Yue2JobSpec = serde_json::from_value(value)
        .map_err(|error| WorkerError::InvalidPayload(format!("yue2 block: {error}")))?;
    contract::validate_for_execution(&spec)
        .map_err(|error| WorkerError::InvalidPayload(format!("[{}] {error}", error.code)))?;
    Ok(spec)
}

async fn execute(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
    record: &mut JobRecord,
) -> WorkerResult<()> {
    let spec = parse_spec(&job.payload)?;
    record.spec = Some(spec.clone());
    let run_id = spec.run_id.clone().unwrap_or_default();
    let project_id = job
        .payload
        .get("projectId")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| WorkerError::InvalidPayload("projectId is required.".into()))?
        .to_owned();
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&project_id)?;
    let project_path = PathBuf::from(project.path);
    let run_rel = contract::run_dir(&run_id);
    let run_dir = project_path.join(&run_rel);
    record.run_rel = Some(run_rel.clone());
    record.run_dir = Some(run_dir.clone());
    let backend = backend_label(&settings.gpu_id).to_owned();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        yue2_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.02,
            "Checking YuE2 eligibility.",
            None,
            &backend,
        ),
    )
    .await?;

    // Eligibility at execution: before any weight is resolved or read.
    let usage_policy = check_eligibility(api, &job.id).await?;
    record.usage_policy = Some(usage_policy.clone());
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    let entry = job
        .payload
        .get("modelManifestEntry")
        .cloned()
        .unwrap_or_else(|| json!({}));
    if spec.kind == Yue2JobKind::Transcribe {
        return Err(transcription_blocked(&entry));
    }
    let load = resolve_load(settings, &entry, &spec)?;
    record.identities = Some((
        load.model_identity.clone(),
        load.decoder_identity.clone(),
        load.tier.as_str(),
    ));
    let inputs = resolve_inputs(api, &spec, &project_path, &project_id).await?;
    record.effective = Some(effective_settings(&spec, &load, &inputs, None));
    // Exclusive for the whole job: no other worker (this host or another sharing the data dir)
    // can run, resume or clean this run while it is held.
    record.claim = Some(RunClaim::acquire(&run_dir)?);
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    // Whole-render memory admission (sc-23001): price THIS request as THIS load will run it, before
    // anything loads. A render that cannot fit is refused here with the binding stage, the
    // shortfall and what would fit; one that fits carries a memory block where every control the
    // job set is honoured as sent and every control it left unset is the gate's choice (so a job
    // that sets only `stageResidency` still gets a decode tile that fits), and the lease holds its
    // residency until the generator is dropped.
    let cancel = CancelFlag::new();
    let mut request = build_request(&spec, &inputs, &run_dir, cancel.clone());
    let tier = crate::yue2_admission::Yue2Tier::from_key(load.tier.as_str()).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("yue2: {} is not a YuE2 tier", load.tier.as_str()))
    })?;
    let controls = spec.memory.unwrap_or_default();
    let pins = crate::yue2_admission::Yue2Pins::from_controls(
        controls.stage_residency,
        spec.offload_policy
            .map(|policy| policy == OffloadPolicy::Sequential),
        controls.chunk_attention,
        controls.attention_chunk_size,
        controls.tile_vae_decode,
        controls.decode_tile_edge,
    )
    .map_err(|why| WorkerError::InvalidPayload(format!("yue2: {why}")))?;
    let admitted = crate::yue2_admission::check(
        contract::MODEL_ID,
        &entry,
        &request,
        crate::yue2_admission::Yue2LoadFacts::of(tier, &load.spec),
        Some(pins),
        &settings.gpu_id,
    )
    .await?;
    request.memory = Some(admitted.memory);

    update_job(
        api,
        &job.id,
        yue2_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.05,
            "Loading YuE2.",
            None,
            &backend,
        ),
    )
    .await?;
    let report = generate(
        api,
        settings,
        job,
        &spec,
        load.spec.clone(),
        request,
        cancel,
        admitted.lease,
        record.engine_started.clone(),
        load_generator,
    )
    .await?;

    // The run is published: finish publishing it. (A cancel that arrives now would leave a
    // published run no job owns; the finished work is kept instead.)
    update_job(
        api,
        &job.id,
        yue2_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.9,
            "Saving the YuE2 run.",
            None,
            &backend,
        ),
    )
    .await?;
    let published = Published::read(&report, &run_dir, &run_rel)?;
    let effective = effective_settings(&spec, &load, &inputs, published.config.as_ref());
    record.effective = Some(effective.clone());
    let mut block = provenance_block(
        &spec,
        &load,
        &inputs,
        &published,
        &report,
        effective,
        &usage_policy,
    );
    let (result, message) = if spec.kind.renders_audio() {
        let track = match report.output {
            Some(GenerationOutput::Audio(track)) => track,
            _ => {
                return Err(WorkerError::Engine(
                    "yue2: the render returned no audio track".into(),
                ))
            }
        };
        let result = publish_audio(&spec, &project_path, track, &mut block, &usage_policy).await?;
        (
            result,
            completion_message("Generated a YuE2 song", &published),
        )
    } else {
        let mut result = JsonObject::new();
        result.insert("yue2".to_owned(), Value::Object(block));
        (
            result,
            completion_message("Published the YuE2 plan", &published),
        )
    };
    update_job(
        api,
        &job.id,
        yue2_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &message,
            Some(result),
            &backend,
        ),
    )
    .await?;
    drop(record.claim.take());
    Ok(())
}

/// The completion message: a truncated run says so, naming the phase that hit its budget.
fn completion_message(done: &str, published: &Published) -> String {
    let truncated = published.truncated();
    let phases: Vec<&str> = ["abc", "semantic"]
        .into_iter()
        .filter(|phase| truncated.get(*phase).and_then(Value::as_bool) == Some(true))
        .map(|phase| {
            if phase == "abc" {
                "score"
            } else {
                "semantic tokens"
            }
        })
        .collect();
    if phases.is_empty() {
        format!("{done}.")
    } else {
        format!(
            "{done} (truncated: {} hit the token budget).",
            phases.join(" and ")
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Eligibility and blocked transcription.
// ---------------------------------------------------------------------------------------------

async fn check_eligibility(api: &ApiClient, job_id: &str) -> WorkerResult<Value> {
    let answer: Value = api
        .get_json(&format!("/api/v1/jobs/{job_id}/yue2-eligibility"))
        .await
        .map_err(|error| match error {
            // The refusal keeps its typed code in the job's error.
            WorkerError::Api {
                status,
                detail,
                code,
            } if status == StatusCode::FORBIDDEN || status == StatusCode::NOT_FOUND => {
                WorkerError::InvalidPayload(format!(
                    "YuE2 is not eligible to run now [{}]: {detail}",
                    code.as_deref().unwrap_or("refused")
                ))
            }
            other => other,
        })?;
    if answer.get("eligible").and_then(Value::as_bool) != Some(true) {
        return Err(WorkerError::InvalidPayload(
            "YuE2 is not eligible to run now: the eligibility check did not grant it".into(),
        ));
    }
    Ok(answer.get("usagePolicy").cloned().unwrap_or(Value::Null))
}

fn transcription_blocked(entry: &Value) -> WorkerError {
    let detail = match sceneworks_core::model_usage_policy::conditional_component_downloads(
        entry, "cover",
    ) {
        Err(sceneworks_core::model_usage_policy::ConditionalComponentsError::Blocked {
            blocked,
            ..
        }) => blocked
            .iter()
            .map(|(id, reason, unblock)| format!("{id}: {reason} (unblock: {unblock})"))
            .collect::<Vec<_>>()
            .join("; "),
        Err(error) => error.to_string(),
        Ok(_) => "no linked runtime bundle carries the transcription crate".into(),
    };
    WorkerError::InvalidPayload(format!(
        "[component_blocked] YuE2 recording transcription is blocked: {detail}"
    ))
}

// ---------------------------------------------------------------------------------------------
// Load resolution.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct LoadPlan {
    spec: LoadSpec,
    tier: Tier,
    decoder: Option<Decoder>,
    staged_decoders: Vec<Decoder>,
    model_identity: Value,
    decoder_identity: Option<Value>,
}

fn primary_rows(entry: &Value) -> Vec<&Value> {
    entry
        .get("downloads")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter(|row| row.get("coRequisite").and_then(Value::as_bool) != Some(true))
                .collect()
        })
        .unwrap_or_default()
}

fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|v| !v.is_empty())
}

fn tier_of(name: &str) -> Option<Tier> {
    match name {
        "bf16" => Some(Tier::Bf16),
        "q8" => Some(Tier::Q8),
        "q4" => Some(Tier::Q4),
        _ => None,
    }
}

fn resolve_load(settings: &Settings, entry: &Value, spec: &Yue2JobSpec) -> WorkerResult<LoadPlan> {
    let rows = primary_rows(entry);
    let tier = match spec.tier {
        Some(tier) => tier,
        None => rows
            .iter()
            .find(|row| row.get("default").and_then(Value::as_bool) == Some(true))
            .and_then(|row| text(row, "variant"))
            .and_then(tier_of)
            .unwrap_or(Tier::Bf16),
    };
    let row = rows
        .iter()
        .find(|row| text(row, "variant") == Some(tier.as_str()))
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "yue2: the catalog declares no '{}' tier",
                tier.as_str()
            ))
        })?;
    let repo = text(row, "repo").unwrap_or_default();
    let revision = text(row, "revision").unwrap_or_default();
    // `{id, revision}` — the score store's component identity; the tier is in the effective
    // settings.
    let model_identity = json!({ "id": repo, "revision": revision });
    let (weights, quant) = match local_derivation(row) {
        None => {
            let dir = crate::model_jobs::huggingface_pinned_snapshot_dir(
                &settings.data_dir,
                repo,
                revision,
            )
            .filter(|dir| dir.join("config.json").is_file())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "yue2: the '{}' weights ({repo}@{revision}) are not installed; install YuE2 \
                     first",
                    tier.as_str()
                ))
            })?;
            (dir, None)
        }
        Some(derivation) => {
            let dir = local_derivation_snapshot_dir(
                &settings.data_dir,
                contract::MODEL_ID,
                tier.as_str(),
                &derivation,
            )
            .ok_or_else(|| WorkerError::InvalidPayload("yue2: unsafe derivation path".into()))?;
            match derived_snapshot_state(&dir, &derivation) {
                DerivedSnapshotState::Verified => {}
                DerivedSnapshotState::Absent => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "yue2: the '{}' tier has not been derived on this machine; install it \
                         (it is derived locally from the bf16 original)",
                        tier.as_str()
                    )))
                }
                DerivedSnapshotState::Invalid(why) => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "yue2: the derived '{}' tier does not verify ({why}); reinstall it",
                        tier.as_str()
                    )))
                }
            }
            let quant = match tier {
                Tier::Q8 => Some(Quant::Q8),
                Tier::Q4 => Some(Quant::Q4),
                Tier::Bf16 => None,
            };
            (dir, quant)
        }
    };
    let mut load = LoadSpec::new(WeightsSource::Dir(weights));
    let mut staged = Vec::new();
    let mut decoder_identities: BTreeMap<&str, Value> = BTreeMap::new();
    for decoder in [Decoder::Standard, Decoder::Legacy] {
        let Some(row) = entry
            .get("downloads")
            .and_then(Value::as_array)
            .and_then(|rows| {
                rows.iter()
                    .find(|row| text(row, "componentId") == Some(decoder.component_id()))
            })
        else {
            continue;
        };
        let (repo, revision) = (
            text(row, "repo").unwrap_or_default(),
            text(row, "revision").unwrap_or_default(),
        );
        if let Some(dir) =
            crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision)
                .filter(|dir| dir.join("config.json").is_file() || dir_has_files(dir))
        {
            load = load.with_component(decoder.component_id(), WeightsSource::Dir(dir));
            staged.push(decoder);
            decoder_identities.insert(
                decoder.as_str(),
                json!({ "id": repo, "revision": revision }),
            );
        }
    }
    if staged.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "yue2: no decoder (m-a-p/YuE2-Vae or m-a-p/YuE2-Vae-legacy) is installed".into(),
        ));
    }
    let decoder = spec
        .kind
        .renders_audio()
        .then(|| spec.decoder.unwrap_or(Decoder::Standard));
    if let Some(decoder) = decoder {
        if !staged.contains(&decoder) {
            return Err(WorkerError::InvalidPayload(format!(
                "yue2: the {} decoder is not installed; install it (choice decoder = {})",
                decoder.as_str(),
                decoder.as_str()
            )));
        }
    }
    if let Some(quant) = quant {
        load = load.with_quant(quant);
    }
    if spec.precision == Some(ComputePrecision::Fp32) {
        load.precision = Precision::Fp32;
    }
    if spec.offload_policy == Some(OffloadPolicy::Sequential) {
        load = load.with_offload_policy(GenOffloadPolicy::Sequential);
    }
    Ok(LoadPlan {
        spec: load,
        tier,
        decoder,
        staged_decoders: staged,
        model_identity,
        decoder_identity: decoder.and_then(|d| decoder_identities.get(d.as_str()).cloned()),
    })
}

fn dir_has_files(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|mut entries| entries.next().is_some())
}

// ---------------------------------------------------------------------------------------------
// Inputs.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Inputs {
    /// A saved plan to restore: its directory and recorded plan identity.
    plan: Option<(PathBuf, String)>,
    /// A completed run whose latents are decoded.
    cached: Option<PathBuf>,
    /// The score version a render reads.
    version: Option<ScoreVersionRecord>,
    /// The cover's reviewed score.
    cover_score: Option<String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Verify a source run against the identity its producing job recorded, before anything reads it.
fn verify_source_run(dir: &Path, identity: &str) -> WorkerResult<Value> {
    if !dir.is_dir() {
        return Err(WorkerError::InvalidPayload(format!(
            "yue2: the source run {} is missing",
            dir.display()
        )));
    }
    crate::inference_runtime::verify_yue2_run(dir, Some(identity)).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "yue2: the source run does not verify against its recorded identity: {error}"
        ))
    })
}

async fn resolve_inputs(
    api: &ApiClient,
    spec: &Yue2JobSpec,
    project_path: &Path,
    project_id: &str,
) -> WorkerResult<Inputs> {
    let sources = spec.sources.clone().unwrap_or_default();
    let mut inputs = Inputs::default();
    match spec.kind {
        Yue2JobKind::FromPlan => {
            let plan = sources
                .plan
                .expect("validated: fromPlan carries sources.plan");
            let dir = project_path.join(&plan.run_dir);
            let result = verify_source_run(&dir, &plan.identity)?;
            let recorded = result
                .get("plan_identity")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let expected = plan.plan_identity.clone().unwrap_or_default();
            if recorded != expected {
                return Err(WorkerError::InvalidPayload(format!(
                    "yue2: the saved plan's identity {recorded} is not the recorded {expected}"
                )));
            }
            inputs.plan = Some((dir, expected));
        }
        Yue2JobKind::Decode => {
            let source = sources
                .source_run
                .expect("validated: decode carries sources.sourceRun");
            let dir = project_path.join(&source.run_dir);
            verify_source_run(&dir, &source.identity)?;
            inputs.cached = Some(dir);
        }
        Yue2JobKind::RenderVersion => {
            let queued = sources
                .version
                .expect("validated: renderVersion carries sources.version");
            let detail: Value = api
                .get_json(&format!(
                    "/api/v1/projects/{project_id}/yue2/score-versions/{}",
                    queued.id
                ))
                .await?;
            let version: ScoreVersionRecord =
                serde_json::from_value(detail.get("version").cloned().unwrap_or(Value::Null))
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "yue2: unreadable score version: {error}"
                        ))
                    })?;
            // The version must be the one the job was queued for, byte for byte.
            let score_sha = sha256_hex(version.score.abc.as_bytes());
            let request_sha = sceneworks_core::yue2_score::request_sha256(&version.request);
            if score_sha != queued.score_sha256
                || version.score.sha256 != queued.score_sha256
                || request_sha != queued.request_sha256
                || version.request_sha256 != queued.request_sha256
            {
                return Err(WorkerError::InvalidPayload(format!(
                    "yue2: score version {} no longer has the score/request digests this job was \
                     queued with",
                    queued.id
                )));
            }
            inputs.version = Some(version);
        }
        Yue2JobKind::Cover => {
            let cover = spec.cover.clone().unwrap_or_default();
            let score = cover.score.unwrap_or_default();
            if let Some(queued) = &sources.cover_version {
                if sha256_hex(score.as_bytes()) != queued.score_sha256 {
                    return Err(WorkerError::InvalidPayload(format!(
                        "yue2: the cover score is not score version {}'s reviewed score",
                        queued.id
                    )));
                }
            }
            inputs.cover_score = Some(score);
        }
        _ => {}
    }
    Ok(inputs)
}

// ---------------------------------------------------------------------------------------------
// The request.
// ---------------------------------------------------------------------------------------------

fn sampling(s: &TokenSampling) -> GenTokenSampling {
    GenTokenSampling {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        repetition_penalty: s.repetition_penalty,
        penalty_window: s.penalty_window,
        min_tokens: s.min_tokens,
        max_tokens: s.max_tokens,
    }
}

fn planning(p: Planning) -> SongPlanning {
    match p {
        Planning::Full => SongPlanning::Full,
        Planning::Melody => SongPlanning::Melody,
        Planning::Off => SongPlanning::Off,
    }
}

fn song_decoder(d: Decoder) -> SongDecoder {
    match d {
        Decoder::Standard => SongDecoder::Standard,
        Decoder::Legacy => SongDecoder::Legacy,
    }
}

/// The request's memory controls. `stage_residency` is decided by the load's offload policy when
/// the request leaves it unset: a present memory block otherwise forces the AR weights resident.
fn memory(spec: &Yue2JobSpec) -> Option<GenerationMemory> {
    let m = spec.memory?;
    Some(GenerationMemory {
        stage_residency: m
            .stage_residency
            .unwrap_or(spec.offload_policy == Some(OffloadPolicy::Sequential)),
        chunk_attention: m.chunk_attention.unwrap_or(false),
        attention_chunk_size: m.attention_chunk_size,
        tile_vae_decode: m.tile_vae_decode.unwrap_or(false),
        decode_tile_edge: m.decode_tile_edge,
        ..Default::default()
    })
}

fn build_request(
    spec: &Yue2JobSpec,
    inputs: &Inputs,
    run_dir: &Path,
    cancel: CancelFlag,
) -> GenerationRequest {
    let mut song = SongParams::default();
    let mut prompt = spec.style.clone().unwrap_or_default();
    let mut lyrics = spec.lyrics.clone();
    let mut seed = spec.seed;
    let mut guidance = spec.cfg_scale.map(|cfg| cfg as f32);
    match spec.kind {
        Yue2JobKind::Create | Yue2JobKind::Plan => {
            song.planning = spec.planning.map(planning);
            song.score = spec.score.clone();
            song.score_sampling = spec.score_sampling.as_ref().map(sampling);
            song.plan_only = spec.kind == Yue2JobKind::Plan;
        }
        Yue2JobKind::FromPlan => {
            song.plan = inputs.plan.as_ref().map(|(dir, identity)| SavedPlan {
                dir: dir.clone(),
                identity: Some(identity.clone()),
            });
        }
        Yue2JobKind::Cover => {
            let cover = spec.cover.clone().unwrap_or_default();
            song.cover = Some(SongCover {
                mode: match cover.mode {
                    Some(CoverMode::Melody) => SongCoverMode::Melody,
                    _ => SongCoverMode::Full,
                },
                score: inputs.cover_score.clone().unwrap_or_default(),
                keep: cover.keep.map(|keep| match keep {
                    CoverKeep::Both => SongCoverVoice::Both,
                    CoverKeep::Vocal => SongCoverVoice::Vocal,
                    CoverKeep::Instrumental => SongCoverVoice::Instrumental,
                }),
                translated_from: cover.translated_from,
            });
        }
        Yue2JobKind::RenderVersion => {
            if let Some(version) = &inputs.version {
                prompt = version.request.style.clone();
                lyrics = Some(version.request.lyrics.clone());
                seed = Some(version.request.seed);
                guidance = version.request.cfg_scale.map(|cfg| cfg as f32);
                song.planning = Some(match version.request.cot {
                    sceneworks_core::yue2_score::Cot::Full => SongPlanning::Full,
                    sceneworks_core::yue2_score::Cot::Melody => SongPlanning::Melody,
                });
                song.score = Some(version.score.abc.clone());
            }
        }
        Yue2JobKind::Decode => {
            song.cached_latents = inputs.cached.clone();
        }
        Yue2JobKind::Transcribe => {}
    }
    if spec.kind != Yue2JobKind::Plan {
        song.semantic_sampling = spec.semantic_sampling.as_ref().map(sampling);
        song.decoder = spec.decoder.map(song_decoder);
    }
    GenerationRequest {
        prompt,
        seed,
        steps: spec.steps,
        guidance,
        audio: Some(AudioParams {
            lyrics,
            song: Some(song),
            artifacts: Some(AudioArtifacts {
                dir: run_dir.to_path_buf(),
                resume: true,
            }),
            ..Default::default()
        }),
        memory: memory(spec),
        cancel,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Progress.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Plan,
    Semantic,
    Acoustic,
}

/// Maps the provider's progress onto one monotone job fraction and a stage message. The provider
/// emits `Step` per sampled token of the plan and semantic phases (against each phase's
/// `max_tokens`), then per acoustic ODE step, then `Decoding`; each phase counts from 1, so a
/// phase change is seen as the count restarting (or the total changing). A resumed run reuses
/// finished phases silently, so the first phase is recognised by its total.
#[derive(Clone, Debug)]
pub(crate) struct Yue2Progress {
    phases: Vec<Phase>,
    current: Option<usize>,
    last: Option<(u32, u32)>,
    abc_total: u32,
    semantic_total: u32,
    fraction: f64,
    plan_only: bool,
}

impl Yue2Progress {
    pub(crate) fn new(spec: &Yue2JobSpec) -> Self {
        let sampled_plan = matches!(spec.kind, Yue2JobKind::Create | Yue2JobKind::Plan)
            && spec.score.is_none()
            && spec.planning != Some(Planning::Off);
        let phases = match spec.kind {
            Yue2JobKind::Plan => vec![Phase::Plan],
            Yue2JobKind::Decode | Yue2JobKind::Transcribe => Vec::new(),
            _ if sampled_plan => vec![Phase::Plan, Phase::Semantic, Phase::Acoustic],
            _ => vec![Phase::Semantic, Phase::Acoustic],
        };
        Self {
            phases,
            current: None,
            last: None,
            abc_total: spec
                .score_sampling
                .and_then(|s| s.max_tokens)
                .unwrap_or(contract::ABC_TOKEN_DEFAULTS.1),
            semantic_total: spec
                .semantic_sampling
                .and_then(|s| s.max_tokens)
                .unwrap_or(contract::SEMANTIC_TOKEN_DEFAULTS.1),
            fraction: 0.05,
            plan_only: spec.kind == Yue2JobKind::Plan,
        }
    }

    fn band(&self, phase: Phase) -> (f64, f64) {
        if self.plan_only {
            return (0.10, 0.88);
        }
        match phase {
            Phase::Plan => (0.10, 0.30),
            Phase::Semantic => (0.30, 0.65),
            Phase::Acoustic => (0.65, 0.85),
        }
    }

    fn position(&self, phase: Phase) -> Option<usize> {
        self.phases.iter().position(|p| *p == phase)
    }

    /// The job fraction and message after `progress`.
    pub(crate) fn observe(&mut self, progress: Progress) -> (f64, String) {
        let message = match progress {
            Progress::Loading(_) => {
                self.fraction = self.fraction.max(0.08);
                "Loading YuE2 weights.".to_owned()
            }
            Progress::Decoding => {
                self.fraction = self.fraction.max(0.86);
                "Decoding audio.".to_owned()
            }
            Progress::Step { current, total } => {
                if self.phases.is_empty() {
                    return (self.fraction, "Decoding audio.".to_owned());
                }
                let index = match (self.current, self.last) {
                    (Some(index), Some((last_current, last_total)))
                        if current < last_current || total != last_total =>
                    {
                        (index + 1).min(self.phases.len() - 1)
                    }
                    (Some(index), _) => index,
                    (None, _) => {
                        if self.phases[0] == Phase::Plan && total == self.abc_total {
                            0
                        } else if let Some(semantic) = self
                            .position(Phase::Semantic)
                            .filter(|_| total == self.semantic_total)
                        {
                            semantic
                        } else {
                            self.position(Phase::Acoustic).unwrap_or(0)
                        }
                    }
                };
                self.current = Some(index);
                self.last = Some((current, total));
                let phase = self.phases[index];
                let (lo, hi) = self.band(phase);
                let done = if total == 0 {
                    0.0
                } else {
                    f64::from(current.min(total)) / f64::from(total)
                };
                self.fraction = self.fraction.max(lo + (hi - lo) * done);
                match phase {
                    Phase::Plan => format!("Planning the score: token {current} of {total}."),
                    Phase::Semantic => {
                        format!("Generating semantic tokens: {current} of {total}.")
                    }
                    Phase::Acoustic => format!("Acoustic synthesis: step {current} of {total}."),
                }
            }
        };
        (self.fraction, message)
    }
}

// ---------------------------------------------------------------------------------------------
// Generation.
// ---------------------------------------------------------------------------------------------

fn classify(context: &str, error: gen_core::Error) -> WorkerError {
    if matches!(error, gen_core::Error::Canceled) {
        WorkerError::Canceled(CANCEL_MESSAGE.to_owned())
    } else {
        crate::classify_engine_error(context, error)
    }
}

#[allow(clippy::too_many_arguments)]
async fn generate(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    spec: &Yue2JobSpec,
    load: LoadSpec,
    request: GenerationRequest,
    cancel: CancelFlag,
    lease: crate::yue2_admission::Yue2Lease,
    engine_started: Arc<AtomicBool>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<GenerationReport> {
    let (tx, mut rx) = tokio::sync::watch::channel::<Option<(f64, String)>>(None);
    let mut tracker = Yue2Progress::new(spec);
    let handle = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<GenerationReport> {
            // The engine's cancel hook is `request.cancel`; it is checked at the load boundary
            // too, so a cancel during a cold load never starts generation.
            // Declared before the generator so it drops AFTER it (sc-23001): the admitted
            // residency is released only once the model's memory is — on success, error or cancel.
            let mut lease = lease;
            if cancel.is_cancelled() {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            let generator = load_generator(contract::MODEL_ID, &load)
                .map_err(|error| classify("YuE2 model load failed", error))?;
            if cancel.is_cancelled() {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            let mut on_progress = |progress: Progress| {
                lease.observe(&progress);
                let update = tracker.observe(progress);
                // The pump may be gone; generation never depends on its progress sink.
                tx.send_replace(Some(update));
            };
            engine_started.store(true, Ordering::SeqCst);
            generator
                .generate_with_report(&request, &mut on_progress)
                .map_err(|error| classify("YuE2 generation failed", error))
        })
    };
    let pump = {
        let api = api.clone();
        let job_id = job.id.clone();
        let backend = backend_label(&settings.gpu_id).to_owned();
        let cancel = cancel.clone();
        tokio::spawn(async move {
            let mut posts = 0usize;
            let mut last_post: Option<Instant> = None;
            while rx.changed().await.is_ok() {
                if cancel.is_cancelled() {
                    break;
                }
                if let Some(at) = last_post {
                    let wait = PROGRESS_POST_INTERVAL.saturating_sub(at.elapsed());
                    if !wait.is_zero() {
                        tokio::time::sleep(wait).await;
                    }
                }
                let latest = rx.borrow_and_update().clone();
                if let Some((fraction, message)) = latest {
                    // A 409 after a concurrent cancel is expected; the watcher owns the terminal
                    // status.
                    let _ = update_job(
                        &api,
                        &job_id,
                        yue2_progress(
                            JobStatus::Running,
                            ProgressStage::Generating,
                            fraction,
                            &message,
                            None,
                            &backend,
                        ),
                    )
                    .await;
                    posts += 1;
                    last_post = Some(Instant::now());
                }
            }
            posts
        })
    };
    let result = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        CANCEL_MESSAGE,
        "YuE2 generation",
        no_cancel_ack(),
        handle,
    )
    .await;
    let _ = pump.await;
    result
}

// ---------------------------------------------------------------------------------------------
// Run directories.
// ---------------------------------------------------------------------------------------------

pub(crate) fn partial_dir(dir: &Path) -> PathBuf {
    let mut name = dir
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(PARTIAL_SUFFIX);
    dir.with_file_name(name)
}

/// Remove the run's unpublished working directory. Only ever `<run dir>.partial` of this job's own
/// run id, whose name the validated run id fixes.
fn remove_partial(run_dir: &Path) {
    let partial = partial_dir(run_dir);
    if partial.is_dir() {
        if let Err(error) = std::fs::remove_dir_all(&partial) {
            tracing::warn!(path = %partial.display(), %error, "could not remove a canceled YuE2 run's working directory");
        }
    }
}

/// This job's exclusive claim on its run directory: an OS advisory lock (`flock` / `LockFileEx`,
/// through the workspace's `fs2` guard) on the sidecar `<run dir>.claim`, held for the whole job.
///
/// The lock belongs to the open file, so it is released the moment its holder is gone — a crashed
/// worker, a container restarted as pid 1, a recycled pid — and it is honoured across processes
/// and pid namespaces sharing the data directory. While it is held, nothing else can be running
/// this run, so the engine's own `.yue2-run.lock` in the working directory can only be a stale
/// leftover of a dead attempt and is removed.
pub(crate) struct RunClaim {
    _lock: sceneworks_core::file_lock::FileLock,
}

impl RunClaim {
    pub(crate) fn path(run_dir: &Path) -> PathBuf {
        let mut name = run_dir
            .file_name()
            .map(|n| n.to_os_string())
            .unwrap_or_default();
        name.push(".claim");
        run_dir.with_file_name(name)
    }

    pub(crate) fn acquire(run_dir: &Path) -> WorkerResult<Self> {
        let path = Self::path(run_dir);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        let lock = sceneworks_core::file_lock::FileLock::try_exclusive(file).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "yue2: the run {} is claimed by another active job ({error}); it cannot run or \
                 resume while that job holds it",
                run_dir.display()
            ))
        })?;
        let engine_lock = partial_dir(run_dir).join(RUN_LOCK_FILE);
        match std::fs::remove_file(&engine_lock) {
            Ok(()) => tracing::info!(
                path = %engine_lock.display(),
                "removed a stale YuE2 engine claim left by a dead attempt"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(WorkerError::Io(error)),
        }
        Ok(Self { _lock: lock })
    }
}

// ---------------------------------------------------------------------------------------------
// Publication.
// ---------------------------------------------------------------------------------------------

/// What the published run directory records.
struct Published {
    run_rel: String,
    kind: String,
    identity: String,
    result: Value,
    request: Option<Value>,
    config: Option<Value>,
    score_abc: Option<String>,
}

fn read_json_file(path: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

impl Published {
    fn read(report: &GenerationReport, run_dir: &Path, run_rel: &str) -> WorkerResult<Self> {
        let record = report.artifacts.as_ref().ok_or_else(|| {
            WorkerError::Engine("yue2: the render published no run record".into())
        })?;
        let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => a == b,
        };
        if !same(&record.dir, run_dir) {
            return Err(WorkerError::Engine(format!(
                "yue2: the run was published at {} instead of {}",
                record.dir.display(),
                run_dir.display()
            )));
        }
        let result = read_json_file(&run_dir.join("result.json")).ok_or_else(|| {
            WorkerError::Engine("yue2: the published run has no readable result.json".into())
        })?;
        if result.get("identity").and_then(Value::as_str) != Some(record.identity.as_str()) {
            return Err(WorkerError::Engine(
                "yue2: the published record's identity is not its result.json's".into(),
            ));
        }
        let config = read_json_file(&run_dir.join("config.json")).or_else(|| {
            read_json_file(&run_dir.join("provenance.json")).and_then(|p| p.get("config").cloned())
        });
        Ok(Self {
            run_rel: run_rel.to_owned(),
            kind: record.kind.clone(),
            identity: record.identity.clone(),
            request: read_json_file(&run_dir.join("request.json")),
            config,
            score_abc: std::fs::read_to_string(run_dir.join("score.abc")).ok(),
            result,
        })
    }

    fn truncated(&self) -> Value {
        let flag = |key: &str| {
            self.result
                .pointer(&format!("/truncated/{key}"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        };
        json!({ "abc": flag("abc"), "semantic": flag("semantic") })
    }

    /// The run's request in sc-22997's `SongRequest` shape, when it has a score to version.
    fn score_request(&self) -> Option<Value> {
        let request = self.request.as_ref()?;
        let cot = request.get("cot").and_then(Value::as_str)?;
        if !matches!(cot, "full" | "melody") {
            return None;
        }
        let mut out = json!({
            "style": request.get("style").cloned().unwrap_or(json!("")),
            "lyrics": request.get("lyrics").cloned().unwrap_or(json!("")),
            "cot": cot,
            "seed": request.get("seed").cloned().unwrap_or(json!(contract_default_seed())),
        });
        if let Some(cfg) = request.get("cfg_scale").filter(|v| !v.is_null()) {
            out["cfgScale"] = cfg.clone();
        }
        Some(out)
    }
}

fn contract_default_seed() -> u64 {
    sceneworks_core::yue2_score::DEFAULT_SEED
}

/// Every setting the run was made with: what the job asked for and what it resolved to, plus the
/// engine's own effective configuration (`config.json`) once published.
fn effective_settings(
    spec: &Yue2JobSpec,
    load: &LoadPlan,
    inputs: &Inputs,
    engine_config: Option<&Value>,
) -> Value {
    let version_request = inputs.version.as_ref().map(|v| &v.request);
    json!({
        "kind": spec.kind.as_str(),
        "tier": load.tier.as_str(),
        "decoder": load.decoder.map(Decoder::as_str),
        "stagedDecoders": load.staged_decoders.iter().map(|d| d.as_str()).collect::<Vec<_>>(),
        "precision": spec.precision.map(|p| match p {
            ComputePrecision::Default => "default",
            ComputePrecision::Fp32 => "fp32",
        }).unwrap_or("default"),
        "offloadPolicy": spec.offload_policy.map(|p| match p {
            OffloadPolicy::Resident => "resident",
            OffloadPolicy::Sequential => "sequential",
        }).unwrap_or("resident"),
        "memory": spec.memory,
        "style": version_request.map(|r| r.style.clone()).or_else(|| spec.style.clone()),
        "lyrics": version_request.map(|r| r.lyrics.clone()).or_else(|| spec.lyrics.clone()),
        "seed": version_request.map(|r| r.seed).or(spec.seed),
        "cfgScale": version_request.and_then(|r| r.cfg_scale).or(spec.cfg_scale),
        "steps": spec.steps,
        "planning": version_request
            .map(|r| r.cot.as_str().to_owned())
            .or_else(|| spec.planning.map(|p| match p {
                Planning::Full => "full".to_owned(),
                Planning::Melody => "melody".to_owned(),
                Planning::Off => "off".to_owned(),
            })),
        "scoreSampling": spec.score_sampling,
        "semanticSampling": spec.semantic_sampling,
        "cover": spec.cover.as_ref().map(|c| json!({
            "mode": c.mode,
            "keep": c.keep,
            "translated": c.translated_from.is_some(),
        })),
        "engineConfig": engine_config,
    })
}

#[allow(clippy::too_many_arguments)]
fn provenance_block(
    spec: &Yue2JobSpec,
    load: &LoadPlan,
    inputs: &Inputs,
    published: &Published,
    report: &GenerationReport,
    effective: Value,
    usage_policy: &Value,
) -> JsonObject {
    let mut block = json!({
        "status": "completed",
        "kind": spec.kind.as_str(),
        "runId": spec.run_id,
        "run": {
            "dir": published.run_rel,
            "kind": published.kind,
            "identity": published.identity,
            "planIdentity": published.result.get("plan_identity"),
        },
        "truncated": published.truncated(),
        "warnings": report.warnings.iter().map(|w| json!({"code": w.code, "message": w.message})).collect::<Vec<_>>(),
        "effectiveSettings": effective,
        "request": published.score_request(),
        "model": load.model_identity,
        "decoder": load.decoder_identity,
        "weights": published.result.get("weights"),
        "engineLicense": published.result.get("license"),
        "sources": spec.sources,
        "batch": spec.batch,
        "usagePolicy": usage_policy,
    })
    .as_object()
    .cloned()
    .expect("json! object literal");
    if matches!(spec.kind, Yue2JobKind::Create | Yue2JobKind::Plan) {
        if let (Some(abc), Some(_)) = (&published.score_abc, published.score_request()) {
            block.insert(
                "score".to_owned(),
                json!({ "abc": abc, "sha256": sha256_hex(abc.as_bytes()) }),
            );
        }
    }
    if let Some(version) = &inputs.version {
        block.insert("versionId".to_owned(), json!(version.id));
        block.insert(
            "renderedScoreSha256".to_owned(),
            json!(sha256_hex(version.score.abc.as_bytes())),
        );
        block.insert(
            "renderedRequestSha256".to_owned(),
            json!(sceneworks_core::yue2_score::request_sha256(
                &version.request
            )),
        );
    }
    block
}

async fn publish_audio(
    spec: &Yue2JobSpec,
    project_path: &Path,
    track: gen_core::AudioTrack,
    block: &mut JsonObject,
    usage_policy: &Value,
) -> WorkerResult<JsonObject> {
    let sample_rate = track.sample_rate.max(1);
    let channels = track.channels.max(1);
    let sample_count = track.samples.len();
    if sample_count == 0 || track.samples.iter().all(|s| *s == 0.0) {
        return Err(WorkerError::Engine(format!(
            "yue2: the render produced no audible samples ({sample_count} samples); refusing to \
             register an empty clip"
        )));
    }
    let duration = sample_count as f64 / (f64::from(sample_rate) * f64::from(channels));
    let genset_id = format!("genset_{}", Uuid::new_v4().simple());
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let style = block
        .get("effectiveSettings")
        .and_then(|e| e.get("style"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let slug = slugify(&style, "song", Some(42));
    let media_rel = format!(
        "assets/audios/{genset_id}/{}_yue2_{}_{slug}.wav",
        &created_at[..10],
        spec.kind.as_str().to_ascii_lowercase()
    );
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let wav = AudioTrack {
        samples: track.samples,
        sample_rate,
        channels,
    };
    {
        let media_path = media_path.clone();
        tokio::task::spawn_blocking(move || write_wav_pcm16(&wav, &media_path))
            .await
            .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;
    }
    let truncated = block.get("truncated").cloned().unwrap_or_else(|| json!({}));
    let is_truncated = truncated.get("abc").and_then(Value::as_bool) == Some(true)
        || truncated.get("semantic").and_then(Value::as_bool) == Some(true);
    let title: String = style.chars().take(48).collect();
    let display_name = match (title.trim().is_empty(), is_truncated) {
        (true, false) => "YuE2 song".to_owned(),
        (true, true) => "YuE2 song (truncated)".to_owned(),
        (false, false) => title.trim().to_owned(),
        (false, true) => format!("{} (truncated)", title.trim()),
    };
    let effective = block
        .get("effectiveSettings")
        .cloned()
        .unwrap_or(Value::Null);
    let seed = effective.get("seed").cloned().unwrap_or(Value::Null);
    let mut provenance = block.clone();
    // The score text rides the score version the API creates, not every asset sidecar.
    provenance.remove("score");
    let fact = json!({
        "type": "audio",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "audio/wav",
        "duration": duration,
        "sampleRate": sample_rate,
        "channels": channels,
        "family": contract::MODEL_ID,
        "displayName": display_name,
        "createdAt": created_at,
        "mode": format!("yue2_{}", spec.kind.as_str()),
        "model": contract::MODEL_ID,
        "adapter": contract::MODEL_ID,
        "prompt": style,
        "lyrics": effective.get("lyrics").cloned(),
        "seed": seed,
        "steps": spec.steps,
        "guidance": effective.get("cfgScale").cloned(),
        "rawAdapterSettings": effective,
        "extra": {
            "yue2": Value::Object(provenance),
            "usagePolicy": usage_policy,
        },
    });
    let mut result = json!({
        "generationSetId": genset_id,
        "expectedCount": 1,
        "adapter": contract::MODEL_ID,
        "model": contract::MODEL_ID,
        "generationSet": {
            "id": genset_id,
            "mode": format!("yue2_{}", spec.kind.as_str()),
            "model": contract::MODEL_ID,
            "prompt": style,
            "count": 1,
            "createdAt": created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal");
    result.insert("yue2".to_owned(), Value::Object(block.clone()));
    Ok(result)
}

#[cfg(test)]
mod tests;
