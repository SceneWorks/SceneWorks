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
    compile_plan, CompileInputs, CompiledPlan, COMPILED_PLAN_SCHEMA_VERSION,
};
use sceneworks_core::film_plan::{self, ModelLane, PlanDiagnostic, ProductionPlan, ReferencePack};
use sceneworks_core::film_planner::{
    build_planner_request, build_repair_request, capabilities_for, draft_to_plan,
    parse_planner_output, plan_coverage_findings, read_brief_file, validate_brief,
    validate_generated_plan, PlannerCapabilities, ProductionBrief,
};
use sceneworks_core::time::utc_now;
use serde_json::{json, Map as JsonObject, Value};
use tokio::time::Instant;

use crate::film_harness::{sha256_hex, ApiTransport, HarnessError, HostFacts};

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
}

pub type LlmFuture<'a> = Pin<Box<dyn Future<Output = Result<String, HarnessError>> + Send + 'a>>;

/// The planner's only dependency on a language model. Implemented over the SceneWorks LLM seam for
/// real runs and by a scripted fake in tests, so every rule in this module is exercised against
/// well-formed, malformed, beat-dropping and out-of-envelope replies without a GPU.
pub trait PlannerLlm: Send + Sync {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_>;
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
                        return Ok(text);
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
    require_installed: bool,
    facts: &HostFacts,
) -> Result<(JsonObject<String, Value>, PlannerCapabilities), HarnessError> {
    let entry = crate::film_harness::model_entry_for(transport, &brief.model.id).await?;
    // The SAME entry-level gate the dispatch path runs — catalog presence, video type, install
    // state, and the route's own platform-reachability check — rather than a second copy of it.
    let findings = crate::film_harness::model_entry_findings(
        &brief.model.id,
        entry.as_ref(),
        brief.model.tier.as_deref(),
        require_installed,
        facts,
    );
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let entry = entry.expect("findings are empty only with an entry");
    // The lane follows the API HOST's platform, not this process's: `--api` may be another machine.
    let caps = capabilities_for(&brief.model, &entry, facts.lane());
    Ok((entry, caps))
}

/// Refuse before the first decode when no registered worker can run an LLM job. Without this the
/// planner would sit on a queued job until its timeout on a host whose build advertises no native
/// `prompt_refine` provider (a candle-less Linux worker, for instance).
async fn refiner_findings(
    transport: &dyn ApiTransport,
) -> Result<Vec<PlanDiagnostic>, HarnessError> {
    let workers =
        crate::film_harness::expect_ok_on(transport, "GET", "/api/v1/workers", None).await?;
    let advertised = workers
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|worker| worker.get("capabilities").and_then(Value::as_array))
        .flatten()
        .filter_map(Value::as_str)
        .any(|capability| capability == "prompt_refine");
    Ok(if advertised {
        Vec::new()
    } else {
        vec![PlanDiagnostic::plan(
            "planner.llm",
            "no registered worker advertises prompt_refine, so there is no local model to plan \
             with; start the worker (SCENEWORKS_WORKER_ONLY=1) and wait for it to register",
        )]
    })
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
        JsonObject<String, Value>,
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
    findings.extend(film_plan::validate_reference_pack_files(
        &pack,
        &pack_dir(&options.reference_pack_path),
    ));
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let facts = crate::film_harness::host_facts_for(transport).await?;
    let (entry, caps) =
        resolve_envelope(transport, &brief, options.require_installed, &facts).await?;
    let findings = refiner_findings(transport).await?;
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    Ok((brief, pack, entry, caps, facts))
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
    let (brief, pack, entry, caps, facts) = prepare(transport, options).await?;
    let rounds = options.rounds();
    let mut request = build_planner_request(&brief, &pack, &caps);
    let mut last_reply;
    let mut round = 0_u32;
    let plan = loop {
        let reply = llm
            .complete(LlmRequest {
                task: Some(FILM_PLAN_TASK.to_owned()),
                prompt: request.clone(),
                model_id: Some(brief.model.id.clone()),
                workflow: "video".to_owned(),
            })
            .await?;
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
                    Some((&entry, facts.lane())),
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
        request = build_repair_request(&brief, &last_reply, &findings, round, rounds);
    };

    let (plan_path, plan_bytes) = write_generated_plan(options, &plan)?;
    let (compiled, compiled_path) = compile_and_write(
        llm,
        options,
        &plan,
        &pack,
        &entry,
        facts.lane(),
        &plan_bytes,
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
    let entry = crate::film_harness::model_entry_for(transport, &plan.model.id).await?;
    let mut findings = crate::film_harness::model_entry_findings(
        &plan.model.id,
        entry.as_ref(),
        plan.model.tier.as_deref(),
        options.require_installed,
        &facts,
    );
    if findings.is_empty() {
        let entry = entry.as_ref().expect("entry present when findings empty");
        findings.extend(film_plan::validate_all(
            &plan,
            &pack,
            Some(&pack_dir(&options.reference_pack_path)),
            Some((entry, facts.lane())),
        ));
    }
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let entry = entry.expect("entry present when findings empty");
    // A brief is not required to recompile a hand-authored plan. When one is available AND the plan
    // carries the beat ids a generated plan records, the coverage is re-checked by identity so a
    // hand edit cannot quietly delete a required beat.
    if let Some(brief) = sibling_brief(plan_path, &options.brief_path) {
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
        &entry,
        facts.lane(),
        &plan_bytes,
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
fn sibling_brief(plan_path: &Path, brief_path: &Path) -> Option<ProductionBrief> {
    if brief_path.is_file() {
        return read_brief_file(brief_path).ok();
    }
    let dir = plan_path.parent()?;
    ["brief.json", "brief.jsonc"]
        .iter()
        .map(|name| dir.join(name))
        .find(|candidate| candidate.is_file())
        .and_then(|candidate| read_brief_file(&candidate).ok())
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
async fn compile_and_write(
    llm: &dyn PlannerLlm,
    options: &PlannerOptions,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    entry: &JsonObject<String, Value>,
    lane: ModelLane,
    plan_bytes: &[u8],
) -> Result<(CompiledPlan, PathBuf), HarnessError> {
    std::fs::create_dir_all(&options.out_dir)?;
    let mut refined = BTreeMap::new();
    if options.refine_prompts {
        for shot in &plan.shots {
            // The model-keyed refinement asset is selected by `modelId`, so this is the SAME
            // rewrite Video Studio's "Refine" button runs for this model — not a second prompt
            // pipeline.
            let text = llm
                .complete(LlmRequest {
                    task: None,
                    prompt: shot.prompt.clone(),
                    model_id: Some(plan.model.id.clone()),
                    workflow: "video".to_owned(),
                })
                .await?;
            refined.insert(shot.id.clone(), text);
        }
    }
    let compiled = compile_plan(
        plan,
        pack,
        &CompileInputs {
            model_entry: entry,
            lane: lane.manifest_key(),
            plan_sha256: &sha256_hex(plan_bytes),
            compiled_at: &utc_now(),
            refined_prompts: &refined,
        },
    )
    .map_err(HarnessError::Validation)?;
    let compiled_path = options.compiled_path();
    std::fs::write(
        &compiled_path,
        serde_json::to_string_pretty(&compiled)
            .map_err(|error| HarnessError::Io(error.to_string()))?
            + "\n",
    )?;
    Ok((compiled, compiled_path))
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

    #[test]
    fn repair_rounds_are_clamped_to_the_declared_ceiling() {
        let options = |rounds| PlannerOptions {
            brief_path: PathBuf::from("brief.json"),
            reference_pack_path: PathBuf::from("references.json"),
            out_dir: PathBuf::from("out"),
            max_repair_rounds: rounds,
            refine_prompts: true,
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
