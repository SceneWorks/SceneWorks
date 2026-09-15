//! Driving a LOCAL planner to produce a production plan, and compiling that plan into
//! model-specific requests (epic 22708, sc-22713).
//!
//! The planner is not a new inference dependency. It drives the LLM seam SceneWorks already ships —
//! `POST /api/v1/prompts/refine`, the `prompt_refine` job, the native `TextLlm` provider the worker
//! resolves (MLX on macOS, candle on the Windows/CUDA build) — through the same [`ApiTransport`] the
//! rest of the harness uses. Two tasks ride that one seam:
//!
//! * `task: "film_plan"` — the brief, the approved reference roles and the installed model's
//!   capability envelope in, one strict JSON plan out ([`sceneworks_core::film_planner`]);
//! * the ordinary `rewrite` task with `modelId` set to the plan's model — which is what selects the
//!   MiniMax-H3 prompt-refinement asset already in the worker — once per shot, to turn each planned
//!   prompt into the text the engine actually receives.
//!
//! Everything the model returns is treated as untrusted text: it is parsed strictly (unknown fields
//! refused), checked against the brief's required beats and the model's declared menus, and either
//! accepted whole or fed back as findings for a BOUNDED number of repair rounds. On exhaustion the
//! planner fails with the findings and writes the last draft out for inspection. No round drops a
//! beat, shortens the film or rounds a duration to make a finding disappear.
//!
//! There is no hosted path and no fallback: [`local_only_findings`] refuses before the first token
//! if the environment carries a hosted-LLM credential or endpoint, or if the API being driven is
//! not on this machine or this private network.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use sceneworks_core::film_compile::{
    compile_plan, CompileInputs, CompiledPlan, PlannerCostRecord, COMPILED_PLAN_SCHEMA_VERSION,
};
use sceneworks_core::film_plan::{self, ModelLane, PlanDiagnostic, ProductionPlan, ReferencePack};
use sceneworks_core::film_planner::{
    brief_pack_findings, build_planner_request, build_repair_request, capabilities_for,
    draft_to_plan, parse_planner_output, plan_coverage_findings, read_brief_file, validate_brief,
    validate_generated_plan, PlannerCapabilities, ProductionBrief,
};
use sceneworks_core::time::utc_now;
use serde_json::{json, Map as JsonObject, Value};
use tokio::time::Instant;

use crate::film_harness::{sha256_hex, ApiTransport, HarnessError, HostFacts, PlanCatalog};

/// Repair rounds the planner takes by default when a draft is refused. Small on purpose: a local
/// 8B refiner that has not satisfied the validator in two corrections is not converging, and every
/// round costs a full decode. The ceiling below bounds whatever the caller asks for.
pub const DEFAULT_MAX_REPAIR_ROUNDS: u32 = 2;

/// Hard ceiling on repair rounds, whatever `--max-repair-rounds` says. The loop is finite by
/// construction: at most `1 + MAX_REPAIR_ROUNDS_CEILING` model calls for the plan itself.
pub const MAX_REPAIR_ROUNDS_CEILING: u32 = 5;

/// How long one LLM job may take before the planner gives up on it and cancels. A planning decode
/// on a local 8B model is minutes, not hours; without a bound a stuck job would hang the planner.
pub const DEFAULT_LLM_JOB_TIMEOUT: Duration = Duration::from_secs(20 * 60);

/// The `task` discriminator the worker classifies as the film-plan task.
pub const FILM_PLAN_TASK: &str = "film_plan";

/// Environment variables that would point an LLM at somebody else's hardware. Their mere presence
/// refuses the run: the epic's premise is that this pipeline is local, and a planner that silently
/// used a hosted key would invalidate every measurement taken from it.
pub const HOSTED_LLM_ENV_VARS: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "COHERE_API_KEY",
    "FAL_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "HAILUO_API_KEY",
    "MINIMAX_API_KEY",
    "MISTRAL_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "REPLICATE_API_TOKEN",
    "SCENEWORKS_LLM_BASE_URL",
    "SCENEWORKS_PLANNER_LLM_URL",
    "SCENEWORKS_REMOTE_LLM_URL",
    "TOGETHER_API_KEY",
];

/// One request to the local LLM seam.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// The `prompt_refine` task discriminator (`film_plan`, or empty for the rewrite task).
    pub task: Option<String>,
    pub prompt: String,
    /// The TARGET model the answer is for — what selects the model-keyed refinement asset.
    pub model_id: Option<String>,
    pub workflow: String,
    /// The model's own prompt guide, forwarded exactly as Video Studio's "Refine" button forwards
    /// it. The worker appends it to the rewrite's system turn under `# Model prompt guide`; with
    /// none, the rewrite runs on the guide-less system prompt.
    pub guide: Option<String>,
}

/// One completed LLM request: the text, and what it cost (sc-22715). `job_id` and
/// `peak_memory_bytes` are `None` for a backend that runs no job (a scripted fake); the real seam
/// fills both from the job it created and the metrics block the worker posted for it.
#[derive(Debug, Clone, Default)]
pub struct LlmReply {
    pub text: String,
    pub job_id: Option<String>,
    pub elapsed_seconds: f64,
    pub peak_memory_bytes: Option<u64>,
}

pub type LlmFuture<'a> = Pin<Box<dyn Future<Output = Result<LlmReply, HarnessError>> + Send + 'a>>;

/// The planner's only dependency on a language model. Implemented over the SceneWorks LLM seam for
/// real runs and by a scripted fake in tests, so every rule in this module is exercised against
/// well-formed, malformed, beat-dropping and out-of-envelope replies without a GPU.
pub trait PlannerLlm: Send + Sync {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_>;
}

/// The running total of what the planner's LLM work cost, folded into
/// [`sceneworks_core::film_compile::PlannerCostRecord`] when `compiled.json` is written.
#[derive(Debug, Clone, Default)]
struct PlannerCost {
    job_ids: Vec<String>,
    elapsed_seconds: f64,
    peak_memory_bytes: Option<u64>,
}

impl PlannerCost {
    fn record(&mut self, reply: &LlmReply) {
        if let Some(job_id) = &reply.job_id {
            self.job_ids.push(job_id.clone());
        }
        self.elapsed_seconds += reply.elapsed_seconds;
        if let Some(peak) = reply.peak_memory_bytes {
            self.peak_memory_bytes = Some(self.peak_memory_bytes.map_or(peak, |max| max.max(peak)));
        }
    }

    fn into_record(self, repair_rounds: u32, budget_gb: Option<f64>) -> PlannerCostRecord {
        PlannerCostRecord {
            job_ids: self.job_ids,
            elapsed_seconds: self.elapsed_seconds,
            peak_memory_bytes: self.peak_memory_bytes,
            repair_rounds,
            planner_max_memory_gb: budget_gb,
        }
    }
}

/// [`PlannerLlm`] over the shipped `prompt_refine` seam: create the job through
/// `POST /api/v1/prompts/refine`, poll it, and return `result.refinedPrompt`.
pub struct SceneWorksLlm<'a> {
    transport: &'a dyn ApiTransport,
    poll_interval: Duration,
    job_timeout: Duration,
}

impl<'a> SceneWorksLlm<'a> {
    pub fn new(
        transport: &'a dyn ApiTransport,
        poll_interval: Duration,
        job_timeout: Duration,
    ) -> Self {
        Self {
            transport,
            poll_interval,
            job_timeout,
        }
    }
}

impl PlannerLlm for SceneWorksLlm<'_> {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_> {
        Box::pin(async move {
            let started = Instant::now();
            let mut body = json!({
                "prompt": request.prompt,
                "workflow": request.workflow,
            });
            if let Some(task) = request.task.as_deref() {
                body["task"] = json!(task);
            }
            if let Some(model_id) = request.model_id.as_deref() {
                body["modelId"] = json!(model_id);
            }
            if let Some(guide) = request
                .guide
                .as_deref()
                .filter(|guide| !guide.trim().is_empty())
            {
                body["guide"] = json!(guide);
            }
            let created = crate::film_harness::expect_ok_on(
                self.transport,
                "POST",
                "/api/v1/prompts/refine",
                Some(body),
            )
            .await?;
            let job_id = created
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    HarnessError::Transport(format!("refine job response has no id: {created}"))
                })?
                .to_owned();
            let deadline = Instant::now() + self.job_timeout;
            loop {
                let snapshot = crate::film_harness::expect_ok_on(
                    self.transport,
                    "GET",
                    &format!("/api/v1/jobs/{job_id}"),
                    None,
                )
                .await?;
                let status = snapshot
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                match status {
                    "completed" => {
                        let text = snapshot
                            .get("result")
                            .and_then(|result| result.get("refinedPrompt"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        if text.trim().is_empty() {
                            return Err(HarnessError::Transport(format!(
                                "refine job {job_id} completed with no refinedPrompt"
                            )));
                        }
                        // The job's metrics block (`GET /api/v1/jobs/:id/metrics`) carries the
                        // peak the worker's probe measured for this decode. It is POSTed after
                        // the terminal progress, so give it the same grace a render's metrics
                        // get; a worker that measured nothing posts none, and that is `None`,
                        // never an error — cost telemetry must not fail a plan that generated.
                        let peak_memory_bytes =
                            crate::film_harness::job_peak_memory_bytes(self.transport, &job_id)
                                .await;
                        return Ok(LlmReply {
                            text,
                            job_id: Some(job_id),
                            elapsed_seconds: started.elapsed().as_secs_f64(),
                            peak_memory_bytes,
                        });
                    }
                    "failed" | "canceled" | "interrupted" => {
                        let detail = snapshot
                            .get("error")
                            .and_then(Value::as_str)
                            .filter(|error| !error.trim().is_empty())
                            .or_else(|| snapshot.get("message").and_then(Value::as_str))
                            .unwrap_or("no detail");
                        return Err(HarnessError::Transport(format!(
                            "refine job {job_id} {status}: {detail}"
                        )));
                    }
                    _ => {}
                }
                if Instant::now() >= deadline {
                    let _ = crate::film_harness::expect_ok_on(
                        self.transport,
                        "POST",
                        &format!("/api/v1/jobs/{job_id}/cancel"),
                        None,
                    )
                    .await;
                    return Err(HarnessError::Transport(format!(
                        "refine job {job_id} did not finish within {}s (last status {status}); it \
                         was canceled",
                        self.job_timeout.as_secs()
                    )));
                }
                tokio::time::sleep(self.poll_interval).await;
            }
        })
    }
}

/// Refuse anything that would take generation off this machine. `env` is the environment lookup
/// (`std::env::var` in production), injected so the rule is testable without mutating the process.
pub fn local_only_findings(
    api_url: &str,
    env: &dyn Fn(&str) -> Option<String>,
) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    for name in HOSTED_LLM_ENV_VARS {
        if env(name).is_some_and(|value| !value.trim().is_empty()) {
            findings.push(PlanDiagnostic::plan(
                "planner.local",
                format!(
                    "{name} is set in this environment. The local filmmaking harness has no hosted \
                     model path and will not run beside a hosted credential or endpoint; unset it \
                     and retry."
                ),
            ));
        }
    }
    if let Some(message) = non_local_api_reason(api_url) {
        findings.push(PlanDiagnostic::plan("planner.local", message));
    }
    findings
}

/// Why `api_url` is not a SceneWorks API on this machine or this private network, if it is not.
fn non_local_api_reason(api_url: &str) -> Option<String> {
    let trimmed = api_url.trim();
    let (scheme, rest) = trimmed.split_once("://")?;
    if !matches!(scheme.to_ascii_lowercase().as_str(), "http" | "https") {
        return Some(format!(
            "the planner drives the SceneWorks API over http(s); {trimmed:?} is not"
        ));
    }
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('@')
        .next()
        .unwrap_or_default();
    let host = match authority.strip_prefix('[') {
        // IPv6 literal.
        Some(rest) => rest.split(']').next().unwrap_or_default().to_owned(),
        None => authority
            .rsplit_once(':')
            .map(|(host, _)| host.to_owned())
            .unwrap_or_else(|| authority.to_owned()),
    };
    if host.is_empty() {
        return Some(format!("{trimmed:?} names no host"));
    }
    if is_local_host(&host) {
        return None;
    }
    Some(format!(
        "{host:?} is not this machine or a private-network host. The planner only drives a \
         SceneWorks API running on local hardware, so generation cannot leave it."
    ))
}

/// Whether `host` is loopback, a private/link-local address, an `.local` name, or a single-label
/// hostname — the addresses a SceneWorks API on the user's own hardware answers on.
fn is_local_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") {
        return true;
    }
    if let Ok(address) = host.parse::<std::net::IpAddr>() {
        return match address {
            std::net::IpAddr::V4(v4) => {
                v4.is_loopback()
                    || v4.is_private()
                    || v4.is_link_local()
                    // 100.64.0.0/10, the shared address space tailnets and CGNAT use.
                    || (v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1]))
            }
            std::net::IpAddr::V6(v6) => {
                v6.is_loopback()
                    // fc00::/7 unique-local and fe80::/10 link-local.
                    || (v6.segments()[0] & 0xfe00) == 0xfc00
                    || (v6.segments()[0] & 0xffc0) == 0xfe80
            }
        };
    }
    // A single-label hostname ("studio-mac") is resolved on the local network, not the internet.
    !host.contains('.')
}

/// What one planning run needs.
#[derive(Debug, Clone)]
pub struct PlannerOptions {
    pub brief_path: PathBuf,
    pub reference_pack_path: PathBuf,
    /// Where `plan.json` and `compiled.json` are written.
    pub out_dir: PathBuf,
    /// Repair rounds after the first attempt, clamped to [`MAX_REPAIR_ROUNDS_CEILING`].
    pub max_repair_rounds: u32,
    /// Run each planned prompt through the model's own prompt refinement when compiling.
    pub refine_prompts: bool,
    /// The model's prompt guide to forward on each rewrite (`--prompt-guide FILE`). `None` falls
    /// back to the guide the catalog entry names, when that file is on disk beside this checkout.
    pub prompt_guide_path: Option<PathBuf>,
    /// Refuse a model the catalog does not report installed.
    pub require_installed: bool,
    /// The API base URL, for the local-only check. Empty skips it (in-process tests).
    pub api_url: String,
    /// Overwrite an existing `plan.json` whose content differs — off by default so a generated plan
    /// the user has since edited is never silently replaced.
    pub force: bool,
    pub poll_interval: Duration,
    pub job_timeout: Duration,
}

impl PlannerOptions {
    pub fn rounds(&self) -> u32 {
        self.max_repair_rounds.min(MAX_REPAIR_ROUNDS_CEILING)
    }

    pub fn plan_path(&self) -> PathBuf {
        self.out_dir.join("plan.json")
    }

    pub fn compiled_path(&self) -> PathBuf {
        self.out_dir.join("compiled.json")
    }
}

/// What a planning run produced.
#[derive(Debug, Clone)]
pub struct PlannerArtifacts {
    pub plan: ProductionPlan,
    pub plan_path: PathBuf,
    pub compiled: CompiledPlan,
    pub compiled_path: PathBuf,
    /// Repair rounds actually taken (0 when the first draft validated).
    pub repair_rounds: u32,
}

/// Resolve the model entry and the capability envelope the planner will be held to, refusing the
/// same way `film-harness validate` does when the model is absent, not installed or not a video
/// model.
async fn resolve_envelope(
    transport: &dyn ApiTransport,
    brief: &ProductionBrief,
    pack: &ReferencePack,
    require_installed: bool,
    facts: &HostFacts,
) -> Result<(PlanCatalog, PlannerCapabilities), HarnessError> {
    // The reference partition is RESOLVED unconditionally here (sc-23402): the draft this envelope
    // is about to produce does not exist yet, so whether any shot will bind reference roles is not
    // knowable from the draft, and having the entry in hand costs one catalog read.
    let catalog = crate::film_harness::plan_catalog_for(transport, &brief.model.id, true).await?;
    // The lane follows the API HOST's platform, not this process's: `--api` may be another machine.
    let lane = facts.lane();
    // Whether this planning run will offer reference conditioning at all is decided from TWO facts
    // and neither of them is install state (sc-23405):
    //
    //   * the catalog SERVES the family's reference partition — an envelope built on an entry the
    //     API does not hold would offer a mode whose every use is refused per shot; and
    //   * the pack approves at least one reference the shots could bind — references are OPTIONAL
    //     (E1), and a user who supplies none gets the phase-1 envelope and the base path.
    //
    // Install state is deliberately NOT one of them: what the planner writes must not depend on
    // which weights happen to be on this disk, or the same brief and pack would produce a
    // different film on two machines. It is GATED below instead, exactly as `validate` and `run`
    // gate the partition a selected shot resolves to — a refusal in seconds, naming the partition,
    // rather than twenty-five minutes of decoding a plan that could never render. A host that
    // wants the plan anyway passes `--skip-install-check`.
    let caps = {
        let base = catalog.base_entry();
        let widened = base.map(|entry| {
            let caps = capabilities_for(&brief.model, entry, lane);
            match catalog.reference_entry() {
                Some((_, reference)) => caps.with_reference_partition(reference),
                None => caps,
            }
        });
        widened.map(|caps| caps.narrowed_to_pack(pack))
    };
    let gate_reference = caps
        .as_ref()
        .is_some_and(PlannerCapabilities::offers_references);
    // The SAME entry-level gate the dispatch path runs — catalog presence, video type, install
    // state, and the route's own platform-reachability check — rather than a second copy of it.
    let findings = crate::film_harness::catalog_entry_findings(
        &catalog,
        brief.model.tier.as_deref(),
        require_installed,
        facts,
        gate_reference,
    );
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let caps = caps.expect("findings are empty only with an entry");
    Ok((catalog, caps))
}

/// Refuse before the first decode when no registered worker can run an LLM job. Without this the
/// planner would sit on a queued job until its timeout on a host whose build advertises no native
/// `prompt_refine` provider (a candle-less Linux worker, for instance).
async fn refiner_findings(
    transport: &dyn ApiTransport,
) -> Result<Vec<PlanDiagnostic>, HarnessError> {
    let workers =
        crate::film_harness::expect_ok_on(transport, "GET", "/api/v1/workers", None).await?;
    // A LIVE row only (sc-22715), through the same rule the run and the review preflights use: a
    // stale `offline` row still advertising `prompt_refine` would queue a decode nothing claims.
    let advert = crate::film_harness::live_worker_advertising(&workers, "prompt_refine");
    Ok(if advert.live.is_some() {
        Vec::new()
    } else {
        vec![PlanDiagnostic::plan(
            "planner.llm",
            format!(
                "no live registered worker advertises prompt_refine{}, so there is no local model \
                 to plan with; start the worker (SCENEWORKS_WORKER_ONLY=1) and wait for it to \
                 register, or clear a stale worker row that is shadowing it",
                crate::film_harness::stale_workers_detail(&advert.stale)
            ),
        )]
    })
}

/// The planner's declared memory budget against the API HOST's reported memory (sc-22715), the
/// same shape as `host_findings` for a plan's `limits.maxMemoryGb`: a budget the host cannot meet
/// is refused before the first token, and a host that reports no memory at all is refused too —
/// an unchecked ceiling is not a checked one.
fn planner_memory_findings(brief: &ProductionBrief, facts: &HostFacts) -> Vec<PlanDiagnostic> {
    let Some(budget) = brief.limits.planner_max_memory_gb else {
        // `validate_brief` has already refused an undeclared budget.
        return Vec::new();
    };
    match facts.host_memory_gb() {
        Some(host) if budget > host => vec![PlanDiagnostic::plan(
            "limits.plannerMaxMemoryGb",
            format!(
                "planner budget {budget} GB exceeds the {host:.1} GB the registered worker reports \
                 for this host"
            ),
        )],
        Some(_) => Vec::new(),
        None => vec![PlanDiagnostic::plan(
            "limits.plannerMaxMemoryGb",
            "no registered worker reports host memory, so the planner memory budget cannot be \
             checked before the first decode",
        )],
    }
}

/// Refuse a command that would dispatch work against a hosted endpoint or beside a hosted-LLM
/// credential (E1, sc-22715). `plan` and `compile` always checked this; every other command that
/// reaches the API — run, resume, replace-take, review, review-eval, an edit with `--export` —
/// goes through the same rule now, from the one place the binary builds its transport.
pub fn local_only_guard(api_url: &str) -> Result<(), HarnessError> {
    let findings = local_only_findings(api_url, &|name| std::env::var(name).ok());
    if findings.is_empty() {
        Ok(())
    } else {
        Err(HarnessError::Validation(findings))
    }
}

/// Read the brief and the reference pack, refuse anything that is not local, and check both
/// documents before a token is generated.
async fn prepare(
    transport: &dyn ApiTransport,
    options: &PlannerOptions,
) -> Result<
    (
        ProductionBrief,
        ReferencePack,
        PlanCatalog,
        PlannerCapabilities,
        HostFacts,
    ),
    HarnessError,
> {
    let brief = read_brief_file(&options.brief_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let pack = film_plan::read_reference_pack_file(&options.reference_pack_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let mut findings = Vec::new();
    if !options.api_url.trim().is_empty() {
        findings.extend(local_only_findings(&options.api_url, &|name| {
            std::env::var(name).ok()
        }));
    }
    findings.extend(validate_brief(&brief));
    findings.extend(film_plan::validate_reference_pack(&pack));
    // A beat that requires a role this pack does not approve can never be satisfied by any draft,
    // so it is caught here rather than after `1 + rounds` decodes of trying.
    findings.extend(brief_pack_findings(&brief, &pack));
    findings.extend(film_plan::validate_reference_pack_files(
        &pack,
        &pack_dir(&options.reference_pack_path),
    ));
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let facts = crate::film_harness::host_facts_for(transport).await?;
    let (catalog, caps) =
        resolve_envelope(transport, &brief, &pack, options.require_installed, &facts).await?;
    let findings = brief_model_findings(
        &brief,
        catalog
            .base_entry()
            .expect("resolve_envelope refuses without an entry"),
        facts.lane(),
    );
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    // The refiner first, then its memory: with no worker at all the answer is "nothing can plan",
    // not "nothing reports memory" — the second is a consequence of the first.
    let findings = refiner_findings(transport).await?;
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let findings = planner_memory_findings(&brief, &facts);
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    Ok((brief, pack, catalog, caps, facts))
}

/// The PLAN-level half of [`film_plan::validate_plan_against_model`], run against the brief's own
/// model block before the first decode.
///
/// Those rules — an fps off the model's declared menu, a plan resolution off its resolution menu, a
/// `limits.maxMemoryGb` below the lane's `minMemoryGb` — are properties of the brief, not of
/// anything the planner writes, and the brief dictates all three to every draft. Left to fire on
/// the first draft they would cost `1 + rounds` full local decodes and then blame the planner for
/// an input it was never allowed to change. A plan with no shots is exactly the plan-level subset
/// of that validator, so this is the same rules on the same code, not a second copy of them.
fn brief_model_findings(
    brief: &ProductionBrief,
    entry: &JsonObject<String, Value>,
    lane: ModelLane,
) -> Vec<PlanDiagnostic> {
    let probe = ProductionPlan {
        schema_version: film_plan::PLAN_SCHEMA_VERSION,
        id: brief.id.clone(),
        version: brief.version,
        title: brief.title.clone(),
        synopsis: brief.synopsis.clone(),
        model: brief.model.clone(),
        limits: brief.limits.clone(),
        // Inert here: the rules this probe runs read `model` and `limits` only.
        sound: film_plan::PlanSound::default(),
        shots: Vec::new(),
    };
    // The probe has NO shots, so no shot can resolve to a partition: a single-entry view is the
    // whole truth for the plan-level rules this runs.
    film_plan::validate_plan_against_model(
        &probe,
        &film_plan::ModelEntries::single(&brief.model.id, entry),
        lane,
    )
}

fn pack_dir(pack_path: &Path) -> PathBuf {
    pack_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Generate a plan from the brief, repair it up to the declared bound, compile it, and write both
/// documents under `out_dir`.
pub async fn generate(
    transport: &dyn ApiTransport,
    llm: &dyn PlannerLlm,
    options: &PlannerOptions,
) -> Result<PlannerArtifacts, HarnessError> {
    let (brief, pack, catalog, caps, facts) = prepare(transport, options).await?;
    let entries = catalog
        .entries()
        .expect("prepare refuses without a catalog entry");
    let rounds = options.rounds();
    let mut request = build_planner_request(&brief, &pack, &caps);
    let mut last_reply;
    let mut round = 0_u32;
    let mut cost = PlannerCost::default();
    let plan = loop {
        let reply = llm
            .complete(LlmRequest {
                task: Some(FILM_PLAN_TASK.to_owned()),
                prompt: request.clone(),
                model_id: Some(brief.model.id.clone()),
                workflow: "video".to_owned(),
                // No guide: the film-plan task's system turn is the plan contract, not the
                // model's prompt-writing guide, and the whole capability envelope is already in
                // the request the planner composes.
                guide: None,
            })
            .await?;
        cost.record(&reply);
        let reply = reply.text;
        last_reply = reply.clone();
        let findings = match parse_planner_output(&reply) {
            Ok(draft) => {
                let plan = draft_to_plan(&brief, &draft);
                let findings = validate_generated_plan(
                    &brief,
                    &draft,
                    &plan,
                    &pack,
                    Some(&pack_dir(&options.reference_pack_path)),
                    Some((&entries, facts.lane())),
                );
                if findings.is_empty() {
                    break plan;
                }
                findings
            }
            Err(error) => vec![PlanDiagnostic::plan(
                "planner.output",
                format!(
                    "the planner's answer is not a plan document: {error}. Answer with one JSON \
                     object in the required format and nothing else."
                ),
            )],
        };
        if round >= rounds {
            // Bounded: the loop stops at the declared round count and fails with the findings that
            // are still outstanding. The refused draft is written out so the failure is diagnosable
            // and a human can repair it by hand.
            let rejected = options.out_dir.join("planner-rejected.txt");
            std::fs::create_dir_all(&options.out_dir)?;
            std::fs::write(
                &rejected,
                format!(
                    "# planner output refused after {} round(s) of repair\n# {}\n\n{last_reply}\n\n\
                     # findings\n{}\n",
                    round,
                    utc_now(),
                    findings
                        .iter()
                        .map(|finding| format!("- {finding}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                ),
            )?;
            let mut findings = findings;
            findings.push(PlanDiagnostic::plan(
                "planner.repair",
                format!(
                    "the planner did not produce a valid plan within {rounds} repair round(s); the \
                     refused answer is at {}",
                    rejected.display()
                ),
            ));
            return Err(HarnessError::Validation(findings));
        }
        round += 1;
        request = build_repair_request(&brief, &caps, &last_reply, &findings, round, rounds);
    };

    let (plan_path, plan_bytes) = write_generated_plan(options, &plan)?;
    copy_brief_beside_plan(options)?;
    let (compiled, compiled_path) = compile_and_write(
        llm,
        options,
        &plan,
        &pack,
        &entries,
        facts.lane(),
        &plan_bytes,
        cost,
        round,
    )
    .await?;
    Ok(PlannerArtifacts {
        plan,
        plan_path,
        compiled,
        compiled_path,
        repair_rounds: round,
    })
}

/// Compile an existing (possibly hand-edited) plan into requests, rewriting `compiled.json` only.
/// This is the second half of the human-correction loop: generate, edit `plan.json`, recompile,
/// validate, run.
pub async fn compile_existing(
    transport: &dyn ApiTransport,
    llm: &dyn PlannerLlm,
    options: &PlannerOptions,
    plan_path: &Path,
) -> Result<PlannerArtifacts, HarnessError> {
    let plan = film_plan::read_plan_file(plan_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let pack = film_plan::read_reference_pack_file(&options.reference_pack_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let mut findings = Vec::new();
    if !options.api_url.trim().is_empty() {
        findings.extend(local_only_findings(&options.api_url, &|name| {
            std::env::var(name).ok()
        }));
    }
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let facts = crate::film_harness::host_facts_for(transport).await?;
    // Both partitions, when this plan's shots need the reference one (sc-23402).
    let catalog = crate::film_harness::plan_catalog_for(
        transport,
        &plan.model.id,
        plan.shots
            .iter()
            .any(|shot| !shot.conditioning.reference_roles.is_empty()),
    )
    .await?;
    // Compile has no shot selection: every shot of this plan is compiled, so the reference
    // partition is gated exactly when some shot resolves to it (the same condition that decided
    // whether to resolve it at all, above).
    let mut findings = crate::film_harness::catalog_entry_findings(
        &catalog,
        plan.model.tier.as_deref(),
        options.require_installed,
        &facts,
        plan.shots
            .iter()
            .any(|shot| !shot.conditioning.reference_roles.is_empty()),
    );
    if findings.is_empty() {
        let entries = catalog
            .entries()
            .expect("entry present when findings empty");
        findings.extend(film_plan::validate_all(
            &plan,
            &pack,
            Some(&pack_dir(&options.reference_pack_path)),
            Some((&entries, facts.lane())),
        ));
    }
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let entries = catalog
        .entries()
        .expect("entry present when findings empty");
    // A brief is not required to recompile a hand-authored plan. When one is available AND the plan
    // carries the beat ids a generated plan records, the coverage is re-checked by identity so a
    // hand edit cannot quietly delete a required beat.
    if let Some(brief) =
        sibling_brief(plan_path, &options.brief_path).map_err(HarnessError::Validation)?
    {
        let coverage = plan_coverage_findings(&brief, &plan);
        if !coverage.is_empty() {
            return Err(HarnessError::Validation(coverage));
        }
    }
    if options.refine_prompts {
        let findings = refiner_findings(transport).await?;
        if !findings.is_empty() {
            return Err(HarnessError::Validation(findings));
        }
    }
    let plan_bytes = std::fs::read(plan_path)?;
    let (compiled, compiled_path) = compile_and_write(
        llm,
        options,
        &plan,
        &pack,
        &entries,
        facts.lane(),
        &plan_bytes,
        PlannerCost::default(),
        0,
    )
    .await?;
    Ok(PlannerArtifacts {
        plan,
        plan_path: plan_path.to_path_buf(),
        compiled,
        compiled_path,
        repair_rounds: 0,
    })
}

/// The brief to re-check an edited plan against: the one the caller named, else one sitting beside
/// the plan.
///
/// A brief the caller NAMED is a demand for the coverage check, so a malformed one, a stale
/// `schemaVersion` or a typo'd path that happens to exist is an error rather than a silent skip:
/// "the brief could not be read" and "coverage verified" must not look the same. The silent `None`
/// is kept only for the DISCOVERED sibling, where the absence of a usable brief is the ordinary
/// hand-authored case rather than a mistake.
fn sibling_brief(
    plan_path: &Path,
    brief_path: &Path,
) -> Result<Option<ProductionBrief>, Vec<PlanDiagnostic>> {
    if brief_path.is_file() {
        let brief = read_brief_file(brief_path).map_err(|finding| vec![finding])?;
        let findings = validate_brief(&brief);
        if !findings.is_empty() {
            return Err(findings);
        }
        return Ok(Some(brief));
    }
    let Some(dir) = plan_path.parent() else {
        return Ok(None);
    };
    Ok(["brief.json", "brief.jsonc"]
        .iter()
        .map(|name| dir.join(name))
        .find(|candidate| candidate.is_file())
        .and_then(|candidate| read_brief_file(&candidate).ok()))
}

/// Copy the brief into the output directory as `brief.json`, beside the plan it produced.
///
/// This is what makes the documented correction loop (`plan --out DIR`, edit `DIR/plan.json`,
/// `compile --plan DIR/plan.json --out DIR`) re-check beat coverage without the user having to
/// remember `--brief`: [`sibling_brief`] looks for exactly this file. Without it the recompile that
/// AC3 asks a human to run would accept an edit that deletes a required beat, which is the one
/// thing the brief is a contract about. The bytes are copied verbatim (JSONC and all) so the brief
/// the plan answers to travels with it.
fn copy_brief_beside_plan(options: &PlannerOptions) -> Result<(), HarnessError> {
    let destination = options.out_dir.join("brief.json");
    // Planning straight into the brief's own directory must not rewrite the brief with itself.
    if same_file(&options.brief_path, &destination) {
        return Ok(());
    }
    std::fs::write(&destination, std::fs::read(&options.brief_path)?)?;
    Ok(())
}

/// Whether two paths name the same existing file (canonicalized, so `./a` and `a` are one file).
fn same_file(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

/// Write the freshly generated plan, refusing to clobber one the user has since edited — that file
/// IS the correction surface. Returns the exact bytes written, which are what the compiled document
/// is hashed against.
fn write_generated_plan(
    options: &PlannerOptions,
    plan: &ProductionPlan,
) -> Result<(PathBuf, Vec<u8>), HarnessError> {
    std::fs::create_dir_all(&options.out_dir)?;
    let plan_json = serde_json::to_string_pretty(&plan)
        .map_err(|error| HarnessError::Io(error.to_string()))?
        + "\n";
    let plan_path = options.plan_path();
    if let Ok(existing) = std::fs::read_to_string(&plan_path) {
        if existing != plan_json && !options.force {
            return Err(HarnessError::Io(format!(
                "{} already exists and differs from the plan just generated; pass --force to \
                 replace it, or write to a different --out directory",
                plan_path.display()
            )));
        }
    }
    std::fs::write(&plan_path, &plan_json)?;
    Ok((plan_path, plan_json.into_bytes()))
}

/// Refine each shot's prompt (when asked), compile, and write `compiled.json`. The plan itself is
/// never written here: `compile` runs against the plan the user edited, byte for byte, and hashes
/// exactly those bytes so the harness can detect a later edit.
#[allow(clippy::too_many_arguments)]
async fn compile_and_write(
    llm: &dyn PlannerLlm,
    options: &PlannerOptions,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    entries: &film_plan::ModelEntries<'_>,
    lane: ModelLane,
    plan_bytes: &[u8],
    mut cost: PlannerCost,
    repair_rounds: u32,
) -> Result<(CompiledPlan, PathBuf), HarnessError> {
    std::fs::create_dir_all(&options.out_dir)?;
    let mut refined = BTreeMap::new();
    if options.refine_prompts {
        // Read once, not once per shot: the guide is the same for every rewrite in this compile.
        // The guide is a property of the FAMILY the plan declares, so it comes off the base entry
        // even when some shots dispatch on the reference partition (sc-23402).
        let guide = resolve_prompt_guide(options, entries.base_entry())?;
        for shot in &plan.shots {
            // The model-keyed refinement asset is selected by `modelId`, and the guide is the same
            // `guide` field Video Studio forwards — so with a guide resolved this is the rewrite
            // the "Refine" button runs for this model, and with none it is that rewrite MINUS its
            // model prompt guide (`--prompt-guide FILE`, or a guide on disk where the catalog entry
            // names it).
            let reply = llm
                .complete(LlmRequest {
                    task: None,
                    prompt: shot.prompt.clone(),
                    model_id: Some(plan.model.id.clone()),
                    workflow: "video".to_owned(),
                    guide: guide.clone(),
                })
                .await?;
            cost.record(&reply);
            refined.insert(shot.id.clone(), reply.text);
        }
    }
    let mut compiled = compile_plan(
        plan,
        pack,
        &CompileInputs {
            entries,
            lane: lane.manifest_key(),
            plan_sha256: &sha256_hex(plan_bytes),
            compiled_at: &utc_now(),
            refined_prompts: &refined,
        },
    )
    .map_err(HarnessError::Validation)?;
    // What the LLM work cost, persisted beside what it produced (sc-22715). A compile that ran no
    // LLM at all (`--no-refine` over a hand-authored plan) records nothing rather than zeros that
    // would read as a measured cost.
    if !cost.job_ids.is_empty() || repair_rounds > 0 {
        compiled.planner = Some(cost.into_record(repair_rounds, plan.limits.planner_max_memory_gb));
    }
    let compiled_path = options.compiled_path();
    std::fs::write(
        &compiled_path,
        serde_json::to_string_pretty(&compiled)
            .map_err(|error| HarnessError::Io(error.to_string()))?
            + "\n",
    )?;
    Ok((compiled, compiled_path))
}

/// The prompt guide to forward on every rewrite of this compile.
///
/// `--prompt-guide FILE` wins and is an ERROR when unreadable — a guide the caller named and did
/// not get is not the same run as one they never asked for. Otherwise the catalog entry's own
/// `ui.promptGuide.path` is resolved against this checkout's web assets, so a local run gets the
/// guide with no flag; the rust-api serves that path only in an `embed-web` build, so it cannot be
/// fetched from `--api` in general and is read from disk or not at all.
fn resolve_prompt_guide(
    options: &PlannerOptions,
    entry: &JsonObject<String, Value>,
) -> Result<Option<String>, HarnessError> {
    if let Some(path) = options.prompt_guide_path.as_deref() {
        let text = std::fs::read_to_string(path).map_err(|error| {
            HarnessError::Validation(vec![PlanDiagnostic::plan(
                "planner.promptGuide",
                format!(
                    "cannot read the prompt guide at {}: {error}",
                    path.display()
                ),
            )])
        })?;
        return Ok(Some(text));
    }
    Ok(declared_prompt_guide_path(entry).and_then(|path| std::fs::read_to_string(path).ok()))
}

/// Where this checkout keeps the web asset a catalog entry's `ui.promptGuide.path` names, if the
/// file is there. `path` is a URL path under `/prompt-guides/`, so it is taken apart component by
/// component and refused unless every one is a plain name — it comes from a manifest, and a
/// manifest is data.
fn declared_prompt_guide_path(entry: &JsonObject<String, Value>) -> Option<PathBuf> {
    let declared = entry
        .get("ui")
        .and_then(Value::as_object)?
        .get("promptGuide")
        .and_then(Value::as_object)?
        .get("path")
        .and_then(Value::as_str)?
        .trim_start_matches('/');
    let relative = Path::new(declared);
    if declared.is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return None;
    }
    // Vite copies `apps/web/public` into `apps/web/dist`, so either is the same guide; both are
    // resolved against the working directory, which is the checkout for the documented invocation.
    ["apps/web/public", "apps/web/dist"]
        .iter()
        .map(|root| Path::new(root).join(relative))
        .find(|candidate| candidate.is_file())
}

/// Read a compiled request document (strict: unknown fields refused).
pub fn read_compiled_file(path: &Path) -> Result<CompiledPlan, PlanDiagnostic> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        PlanDiagnostic::plan(
            "compiled",
            format!("cannot read {}: {error}", path.display()),
        )
    })?;
    serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text)).map_err(|error| {
        PlanDiagnostic::plan(
            "compiled",
            format!(
                "{}: {error} (this build reads compiled plan schema \
                 {COMPILED_PLAN_SCHEMA_VERSION})",
                path.display()
            ),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosted_credentials_and_remote_apis_are_refused_before_any_generation() {
        let none = |_: &str| None;
        assert!(local_only_findings("http://127.0.0.1:8000", &none).is_empty());
        assert!(local_only_findings("http://localhost:8000/", &none).is_empty());
        assert!(local_only_findings("http://192.168.1.40:8000", &none).is_empty());
        assert!(local_only_findings("http://studio-mac:8000", &none).is_empty());
        assert!(local_only_findings("https://studio.local:8443", &none).is_empty());
        assert!(local_only_findings("http://[::1]:8000", &none).is_empty());

        for hosted in [
            "https://api.openai.com/v1",
            "https://example.com:8000",
            "http://8.8.8.8:8000",
        ] {
            let findings = local_only_findings(hosted, &none);
            assert_eq!(findings.len(), 1, "{hosted}: {findings:?}");
            assert!(
                findings[0].message.contains("not this machine"),
                "{hosted}: {findings:?}"
            );
        }

        for name in ["OPENAI_API_KEY", "SCENEWORKS_PLANNER_LLM_URL"] {
            let env = |asked: &str| (asked == name).then(|| "set".to_owned());
            let findings = local_only_findings("http://127.0.0.1:8000", &env);
            assert_eq!(findings.len(), 1, "{findings:?}");
            assert!(findings[0].message.contains(name), "{findings:?}");
        }
        // An empty value is not a configured endpoint.
        let env = |_: &str| Some("   ".to_owned());
        assert!(local_only_findings("http://127.0.0.1:8000", &env).is_empty());
    }

    /// E1 (sc-22715): the guard every dispatching command runs from the one place the binary
    /// builds its transport, and the binary has exactly that one place.
    #[test]
    fn every_dispatching_command_builds_its_transport_through_the_local_only_guard() {
        // Env is process-global, so this asserts on the URL half; the env half is
        // `hosted_credentials_and_remote_apis_are_refused_before_any_generation` above.
        let error = local_only_guard("https://api.openai.com/v1").expect_err("hosted is refused");
        assert!(
            format!("{error}").contains("not this machine or a private-network host"),
            "{error}"
        );
        // A loopback API passes unless the environment carries a hosted credential, which this
        // process's tests never set.
        if HOSTED_LLM_ENV_VARS
            .iter()
            .all(|name| std::env::var(name).is_err())
        {
            local_only_guard("http://127.0.0.1:8000").expect("loopback is local");
        }
        // The binary: ONE constructor for its HTTP transport, and that constructor calls the
        // guard first. Counting `HttpTransport::new(` is what keeps a future command from
        // building an unguarded transport of its own.
        let source = include_str!("bin/film-harness.rs");
        assert_eq!(
            source.matches("HttpTransport::new(").count(),
            1,
            "film-harness must build its HttpTransport in exactly one place (`guarded_transport`)"
        );
        let guarded = source
            .split("fn guarded_transport(")
            .nth(1)
            .expect("guarded_transport exists");
        let body = guarded
            .split("HttpTransport::new(")
            .next()
            .expect("the transport is built inside guarded_transport");
        assert!(
            body.contains("local_only_guard("),
            "guarded_transport must run the local-only guard BEFORE building the transport"
        );
    }

    #[test]
    fn repair_rounds_are_clamped_to_the_declared_ceiling() {
        let options = |rounds| PlannerOptions {
            brief_path: PathBuf::from("brief.json"),
            reference_pack_path: PathBuf::from("references.json"),
            out_dir: PathBuf::from("out"),
            max_repair_rounds: rounds,
            refine_prompts: true,
            prompt_guide_path: None,
            require_installed: true,
            api_url: String::new(),
            force: false,
            poll_interval: Duration::from_secs(1),
            job_timeout: DEFAULT_LLM_JOB_TIMEOUT,
        };
        assert_eq!(options(0).rounds(), 0);
        assert_eq!(options(2).rounds(), 2);
        assert_eq!(options(99).rounds(), MAX_REPAIR_ROUNDS_CEILING);
        assert_eq!(options(1).plan_path(), PathBuf::from("out/plan.json"));
        assert_eq!(
            options(1).compiled_path(),
            PathBuf::from("out/compiled.json")
        );
    }

    #[test]
    fn an_edited_plan_that_deletes_a_beat_is_reported_but_a_hand_authored_one_is_not() {
        let brief: ProductionBrief = serde_json::from_value(serde_json::json!({
            "schemaVersion": 1,
            "id": "courier",
            "version": 1,
            "title": "Courier",
            "synopsis": "s",
            "targetTotalSeconds": { "min": 5.0, "max": 60.0 },
            "requiredBeats": [
                { "id": "arrival", "summary": "The courier arrives." },
                { "id": "discovery", "summary": "The recipient discovers the parcel." }
            ],
            "model": { "id": "minimax_h3" },
            "limits": { "maxRunSeconds": 60, "maxShotSeconds": 60, "maxAttemptsPerShot": 1, "maxMemoryGb": 1.0 }
        }))
        .unwrap();
        let plan: ProductionPlan = serde_json::from_value(serde_json::json!({
            "schemaVersion": 1,
            "id": "courier",
            "version": 1,
            "title": "Courier",
            "model": { "id": "minimax_h3" },
            "limits": { "maxRunSeconds": 60, "maxShotSeconds": 60, "maxAttemptsPerShot": 1, "maxMemoryGb": 1.0 },
            "shots": [{
                "id": "SH010", "beatId": "arrival",
                "beat": "The courier arrives at the door.", "framing": "wide",
                "prompt": "p", "targetDurationSeconds": 5.0, "startState": "a", "endState": "b",
                "conditioning": { "mode": "text_to_video" }
            }]
        }))
        .unwrap();
        let findings = plan_coverage_findings(&brief, &plan);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].message.contains("\"discovery\""),
            "{findings:?}"
        );

        // A hand-authored plan carries no beat ids and answers to no brief.
        let mut hand_authored = plan.clone();
        hand_authored.shots[0].beat_id = None;
        assert!(plan_coverage_findings(&brief, &hand_authored).is_empty());
    }
}
