//! Compiling a [`ProductionPlan`] into the model-specific requests the harness dispatches
//! (epic 22708, sc-22713).
//!
//! A plan says what the film is; a [`CompiledPlan`] says exactly what will be asked of the model.
//! It is a separate, versioned, diffable document for one reason: the prompt the engine sees is not
//! the prompt the plan holds — it has been through the model's own prompt refinement — and a POC
//! whose output cannot be explained is not evidence. With the compiled document beside the plan,
//! every dispatched request can be read, diffed against the previous version and corrected before
//! anything renders.
//!
//! The compiled request is also the ONLY place a video job body is built. [`CompiledRequest::
//! to_job_body`] is what the driver posts to `/api/v1/video/jobs`, so what a reviewer reads in
//! `compiled.json` and what the API receives cannot drift: the difference between them is only the
//! run-scoped ids and the reference roles resolved to imported asset ids.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonObject, Value};

use crate::film_plan::{
    shot_resolution, ModelLane, PlanDiagnostic, ProductionPlan, ReferencePack, Shot,
};
use crate::MAX_PROMPT_CHARS;

/// Schema version of [`CompiledPlan`] documents this module reads and writes.
pub const COMPILED_PLAN_SCHEMA_VERSION: u32 = 1;

/// Where a compiled request's prompt text came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSource {
    /// The plan's prompt, verbatim. What a hand-authored plan compiles to, and what `--no-refine`
    /// produces.
    Authored,
    /// The plan's prompt after the model's own prompt refinement (the `prompt_refine` seam). The
    /// authored text is kept beside it so the rewrite is reviewable.
    Refined,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompiledModel {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    pub fps: u32,
    pub lane: String,
}

/// One shot, resolved to everything the video route needs except the run-scoped ids and the asset
/// ids the reference roles import to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompiledRequest {
    pub shot_id: String,
    pub beat: String,
    pub mode: String,
    pub model: String,
    /// The prompt the engine will receive.
    pub prompt: String,
    pub prompt_source: PromptSource,
    /// The plan's own prompt, kept when `prompt` was refined so the rewrite can be reviewed and
    /// reverted by editing the plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    pub duration_seconds: f64,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame_role: Option<String>,
    #[serde(default)]
    pub reference_roles: Vec<String>,
    /// Declared continuity intent, carried through to the job so a take's provenance records it.
    /// It never becomes conditioning: the frames this request is conditioned on are the canonical
    /// roles above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_from_shot_id: Option<String>,
    #[serde(default)]
    pub continuity_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompiledPlan {
    pub schema_version: u32,
    pub plan_id: String,
    pub plan_version: u32,
    /// SHA-256 of the plan document these requests were compiled from. A plan edited after the
    /// compile no longer matches, and the harness refuses to dispatch stale requests.
    pub plan_sha256: String,
    pub reference_pack_id: String,
    pub reference_pack_version: u32,
    pub compiled_at: String,
    pub model: CompiledModel,
    pub requests: Vec<CompiledRequest>,
    /// What the planner's LLM decodes cost to produce this document (sc-22715): the jobs it
    /// created through the `prompt_refine` seam, their wall-clock, and the peak memory their
    /// metrics blocks reported. Absent on a plan compiled with no LLM at all (`--no-refine` over a
    /// hand-authored plan), present on every generated or refined one, so the planner's cost is
    /// persisted beside the requests it produced rather than only printed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner: Option<PlannerCostRecord>,
}

/// The cost of the planner's LLM work, persisted into `compiled.json` (sc-22715).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannerCostRecord {
    /// Every `prompt_refine` job this document's planning and refinement created, in order: the
    /// plan draft, each repair round, then one rewrite per shot.
    pub job_ids: Vec<String>,
    /// Wall-clock the LLM jobs took end to end, summed.
    pub elapsed_seconds: f64,
    /// Highest `peakMemoryBytes` any of those jobs' metrics blocks reported; `None` when no job
    /// reported one (a worker whose probe measured nothing posts no block).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_memory_bytes: Option<u64>,
    /// Repair rounds actually taken for the plan draft (0 when the first draft validated, and 0 on
    /// a `compile` of an existing plan).
    pub repair_rounds: u32,
    /// The budget the brief declared for those decodes (`limits.plannerMaxMemoryGb`), so the
    /// record carries the bound beside the measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_max_memory_gb: Option<f64>,
}

/// `advanced.mlxQuantize` for a tier — the shared convention the MLX lanes read.
pub fn mlx_quantize_for_tier(tier: &str) -> Value {
    match tier {
        "bf16" => json!(0),
        "q8" => json!(8),
        _ => json!(4),
    }
}

/// Inputs the compile needs beyond the plan itself.
pub struct CompileInputs<'a> {
    /// The model's catalog entry, for fps and geometry defaults.
    pub model_entry: &'a JsonObject<String, Value>,
    pub lane: &'a str,
    pub plan_sha256: &'a str,
    pub compiled_at: &'a str,
    /// Refined prompt text per shot id. A shot with no entry compiles its authored prompt.
    pub refined_prompts: &'a BTreeMap<String, String>,
}

/// Compile every shot of `plan`. Returns findings instead of a document when a shot cannot be
/// expressed as a request — an unresolvable geometry, or a refined prompt that is empty or longer
/// than the route accepts. Nothing is truncated or substituted: an unusable rewrite is a finding
/// naming the shot, so the fix is to re-run the refinement or edit the plan, never to ship a
/// silently shortened prompt.
pub fn compile_plan(
    plan: &ProductionPlan,
    pack: &ReferencePack,
    inputs: &CompileInputs<'_>,
) -> Result<CompiledPlan, Vec<PlanDiagnostic>> {
    let mut findings = Vec::new();
    let Some(fps) = crate::film_plan::plan_fps(plan, inputs.model_entry) else {
        return Err(vec![PlanDiagnostic::plan(
            "model.fps",
            format!(
                "{} declares no default fps; set model.fps in the plan before compiling",
                plan.model.id
            ),
        )]);
    };
    let mut requests = Vec::with_capacity(plan.shots.len());
    for shot in &plan.shots {
        match compile_shot(plan, shot, inputs, fps) {
            Ok(request) => requests.push(request),
            Err(mut shot_findings) => findings.append(&mut shot_findings),
        }
    }
    if !findings.is_empty() {
        return Err(findings);
    }
    Ok(CompiledPlan {
        schema_version: COMPILED_PLAN_SCHEMA_VERSION,
        plan_id: plan.id.clone(),
        plan_version: plan.version,
        plan_sha256: inputs.plan_sha256.to_owned(),
        reference_pack_id: pack.id.clone(),
        reference_pack_version: pack.version,
        compiled_at: inputs.compiled_at.to_owned(),
        model: CompiledModel {
            id: plan.model.id.clone(),
            tier: plan.model.tier.clone(),
            fps,
            lane: inputs.lane.to_owned(),
        },
        requests,
        planner: None,
    })
}

fn compile_shot(
    plan: &ProductionPlan,
    shot: &Shot,
    inputs: &CompileInputs<'_>,
    fps: u32,
) -> Result<CompiledRequest, Vec<PlanDiagnostic>> {
    let Some((width, height)) = shot_resolution(plan, shot, inputs.model_entry) else {
        return Err(vec![PlanDiagnostic::shot(
            &shot.id,
            "resolution",
            format!(
                "{} declares no default resolution; set one on the plan or the shot",
                plan.model.id
            ),
        )]);
    };
    let (prompt, prompt_source, authored) = match inputs.refined_prompts.get(&shot.id) {
        Some(refined) => {
            let trimmed = refined.trim();
            let length = trimmed.chars().count();
            if trimmed.is_empty() || length > MAX_PROMPT_CHARS {
                return Err(vec![PlanDiagnostic::shot(
                    &shot.id,
                    "prompt",
                    format!(
                        "the refined prompt is {length} characters, outside the 1-{MAX_PROMPT_CHARS} \
                         the video route accepts; re-run the refinement or compile with --no-refine \
                         (the authored prompt is not silently substituted)"
                    ),
                )]);
            }
            (
                trimmed.to_owned(),
                PromptSource::Refined,
                Some(shot.prompt.clone()),
            )
        }
        None => (shot.prompt.clone(), PromptSource::Authored, None),
    };
    Ok(CompiledRequest {
        shot_id: shot.id.clone(),
        beat: shot.beat.clone(),
        mode: shot.conditioning.mode.clone(),
        model: plan.model.id.clone(),
        prompt,
        prompt_source,
        authored_prompt: authored,
        negative_prompt: shot.negative_prompt.clone(),
        duration_seconds: shot.target_duration_seconds,
        fps,
        width,
        height,
        seed: shot.seed,
        first_frame_role: shot.conditioning.first_frame_role.clone(),
        last_frame_role: shot.conditioning.last_frame_role.clone(),
        reference_roles: shot.conditioning.reference_roles.clone(),
        chain_from_shot_id: shot.conditioning.chain_from_shot_id.clone(),
        continuity_roles: shot.continuity_roles.clone(),
    })
}

/// The run-scoped facts a compiled request needs to become a job body.
pub struct DispatchContext<'a> {
    pub project_id: &'a str,
    pub run_id: &'a str,
    pub plan_id: &'a str,
    pub plan_version: u32,
    pub attempt: u32,
    pub tier: Option<&'a str>,
    /// The key this attempt dispatches under, stamped into the job's `filmHarness` provenance so a
    /// controller that died between the POST and the record write finds its OWN job instead of
    /// enqueuing a second render for the same attempt (sc-22711). `None` leaves it out.
    pub idempotency_key: Option<&'a str>,
    /// Reference role -> imported asset id.
    pub role_assets: &'a BTreeMap<String, String>,
}

/// The reference assets one request is conditioned on, as the run record stores them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedConditioning {
    pub first_frame_asset_id: Option<String>,
    pub last_frame_asset_id: Option<String>,
    pub reference_asset_ids: Vec<String>,
}

impl CompiledRequest {
    /// Resolve this request's reference roles against the imported assets. A role with no asset is
    /// a finding — the run never dispatches a keyframe shot with its keyframe quietly missing.
    pub fn resolve_conditioning(
        &self,
        role_assets: &BTreeMap<String, String>,
    ) -> Result<ResolvedConditioning, Vec<PlanDiagnostic>> {
        let mut findings = Vec::new();
        let mut resolve = |field: &str, role: &str| -> Option<String> {
            match role_assets.get(role) {
                Some(asset) => Some(asset.clone()),
                None => {
                    findings.push(PlanDiagnostic::shot(
                        &self.shot_id,
                        field,
                        format!("reference role {role:?} was never imported as an asset"),
                    ));
                    None
                }
            }
        };
        let first = self
            .first_frame_role
            .as_deref()
            .and_then(|role| resolve("conditioning.firstFrameRole", role));
        let last = self
            .last_frame_role
            .as_deref()
            .and_then(|role| resolve("conditioning.lastFrameRole", role));
        let references: Vec<String> = self
            .reference_roles
            .iter()
            .filter_map(|role| resolve("conditioning.referenceRoles", role))
            .collect();
        if findings.is_empty() {
            Ok(ResolvedConditioning {
                first_frame_asset_id: first,
                last_frame_asset_id: last,
                reference_asset_ids: references,
            })
        } else {
            Err(findings)
        }
    }

    /// The `POST /api/v1/video/jobs` body for one attempt. The single place a video job body is
    /// built, for the generated and the hand-authored path alike.
    pub fn to_job_body(&self, context: &DispatchContext<'_>) -> Result<Value, Vec<PlanDiagnostic>> {
        let assets = self.resolve_conditioning(context.role_assets)?;
        Ok(self.to_job_body_with(context, &assets))
    }

    /// [`to_job_body`](Self::to_job_body) with the conditioning already resolved.
    pub fn to_job_body_with(
        &self,
        context: &DispatchContext<'_>,
        assets: &ResolvedConditioning,
    ) -> Value {
        let mut advanced = JsonObject::new();
        if let Some(tier) = context.tier {
            advanced.insert("mlxQuantize".to_owned(), mlx_quantize_for_tier(tier));
        }
        let mut provenance = json!({
            "runId": context.run_id,
            "planId": context.plan_id,
            "planVersion": context.plan_version,
            "shotId": self.shot_id,
            "attempt": context.attempt,
        });
        if let Some(key) = context.idempotency_key {
            provenance["idempotencyKey"] = json!(key);
        }
        if self.prompt_source == PromptSource::Refined {
            provenance["promptSource"] = json!("refined");
        }
        if let Some(chain) = &self.chain_from_shot_id {
            // Recorded so a take's provenance says which shot it was meant to continue. It is not
            // conditioning: the keyframe/reference slots above are the only anchors.
            provenance["chainFromShotId"] = json!(chain);
        }
        if !self.continuity_roles.is_empty() {
            // The canonical roles this shot was written to depict. Recorded for the same reason as
            // the chain and with the same status — traceability, never conditioning — so a take's
            // provenance can say which approved references it was meant to carry even on a model
            // whose declared `maxReferenceAssets` is 0 and whose shots are all `text_to_video`.
            provenance["continuityRoles"] = json!(self.continuity_roles);
        }
        advanced.insert("filmHarness".to_owned(), provenance);
        let mut body = json!({
            "projectId": context.project_id,
            "mode": self.mode,
            "model": self.model,
            "prompt": self.prompt,
            "duration": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "fitMode": "crop",
            "requestedGpu": "auto",
            "advanced": advanced,
        });
        if let Some(negative) = self.negative_prompt.as_deref() {
            body["negativePrompt"] = json!(negative);
        }
        // Attempt `n` renders at `seed + (n - 1)` (sc-22715). The MLX render is deterministic for
        // a seed — two runs of the fixture's SH010 at seed 22710 were pixel-identical — so a
        // replacement that kept the plan's seed would re-render the very take it rejects. The
        // plan's seed is still attempt 1, the derivation is recorded (`filmHarness.seed`, and the
        // take's recipe), and it is the only thing about the dispatched request that varies by
        // attempt: the prompt, the geometry and the conditioning are the compiled request's.
        if let Some(seed) = self.seed {
            let attempt_seed = seed.wrapping_add(i64::from(context.attempt.saturating_sub(1)));
            body["seed"] = json!(attempt_seed);
            body["advanced"]["filmHarness"]["seed"] = json!(attempt_seed);
        }
        if let Some(first) = &assets.first_frame_asset_id {
            body["sourceAssetId"] = json!(first);
        }
        if let Some(last) = &assets.last_frame_asset_id {
            body["lastFrameAssetId"] = json!(last);
        }
        if !assets.reference_asset_ids.is_empty() {
            body["referenceAssetIds"] = json!(assets.reference_asset_ids);
        }
        body
    }
}

impl CompiledPlan {
    /// The compiled request for `shot_id`, if the plan compiled one.
    pub fn request(&self, shot_id: &str) -> Option<&CompiledRequest> {
        self.requests
            .iter()
            .find(|request| request.shot_id == shot_id)
    }

    /// Findings that make these requests unusable for `plan`: a different plan, a different
    /// version, an edit since the compile, or a shot the compile does not cover. This is what keeps
    /// a hand-edited plan from being dispatched with stale prompts.
    pub fn staleness_findings(
        &self,
        plan: &ProductionPlan,
        plan_sha256: &str,
    ) -> Vec<PlanDiagnostic> {
        let mut findings = Vec::new();
        if self.schema_version != COMPILED_PLAN_SCHEMA_VERSION {
            findings.push(PlanDiagnostic::plan(
                "compiled.schemaVersion",
                format!(
                    "unsupported compiled plan schema version {} (this build reads \
                     {COMPILED_PLAN_SCHEMA_VERSION})",
                    self.schema_version
                ),
            ));
            return findings;
        }
        if self.plan_id != plan.id {
            findings.push(PlanDiagnostic::plan(
                "compiled.planId",
                format!(
                    "these requests were compiled from plan {:?}, not {:?}",
                    self.plan_id, plan.id
                ),
            ));
        }
        if self.plan_sha256 != plan_sha256 {
            findings.push(PlanDiagnostic::plan(
                "compiled.planSha256",
                format!(
                    "the plan has changed since it was compiled (plan v{} is {}, the requests were \
                     compiled from {}); re-run `film-harness compile`",
                    plan.version,
                    &plan_sha256[..plan_sha256.len().min(12)],
                    &self.plan_sha256[..self.plan_sha256.len().min(12)]
                ),
            ));
        }
        for shot in &plan.shots {
            if self.request(&shot.id).is_none() {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "compiled",
                    "no compiled request for this shot; re-run `film-harness compile`",
                ));
            }
        }
        findings
    }

    /// Findings that make these requests a different ASK than the plan describes.
    ///
    /// [`staleness_findings`](Self::staleness_findings) proves the requests were compiled from
    /// THIS plan; this proves they still say what compiling it now would say. It matters because
    /// the compiled document — not the plan — is what becomes the job body: `--compiled FILE`
    /// accepts one from any path, and every dispatched field but the prompt is taken from it. A
    /// hand-edited `durationSeconds` the engine would silently snap onto its frame lattice, a
    /// swapped `model`, a `mode` the checkpoint does not declare or a `negativePrompt` on a model
    /// that has none would otherwise reach the route unjudged, because the document validators only
    /// ever read the plan.
    ///
    /// Only `prompt`, `promptSource` and `authoredPrompt` may differ from a fresh compile: those
    /// three ARE the compile's output (the model's own rewrite), and everything else is a
    /// transcription of the plan. The expected request is produced by the compiler itself rather
    /// than by a second list of rules, so the two cannot drift.
    pub fn conformance_findings(
        &self,
        plan: &ProductionPlan,
        entry: &JsonObject<String, Value>,
        lane: ModelLane,
    ) -> Vec<PlanDiagnostic> {
        let mut findings = Vec::new();
        if self.model.id != plan.model.id {
            findings.push(PlanDiagnostic::plan(
                "compiled.model.id",
                format!(
                    "the compiled requests target model {:?}, but the plan renders through {:?}",
                    self.model.id, plan.model.id
                ),
            ));
        }
        if self.model.tier != plan.model.tier {
            findings.push(PlanDiagnostic::plan(
                "compiled.model.tier",
                format!(
                    "the compiled requests declare tier {:?}, but the plan asks for {:?}",
                    self.model.tier, plan.model.tier
                ),
            ));
        }
        if self.model.lane != lane.manifest_key() {
            findings.push(PlanDiagnostic::plan(
                "compiled.model.lane",
                format!(
                    "these requests were compiled for the {:?} lane, but this run's host renders \
                     on {:?}; re-run `film-harness compile` against this host",
                    self.model.lane,
                    lane.manifest_key()
                ),
            ));
        }
        let Some(fps) = crate::film_plan::plan_fps(plan, entry) else {
            findings.push(PlanDiagnostic::plan(
                "model.fps",
                format!(
                    "{} declares no default fps; set model.fps in the plan before dispatching",
                    plan.model.id
                ),
            ));
            return findings;
        };
        if self.model.fps != fps {
            findings.push(PlanDiagnostic::plan(
                "compiled.model.fps",
                format!(
                    "the compiled requests declare {} fps, but the plan renders at {fps}",
                    self.model.fps
                ),
            ));
        }
        let empty = BTreeMap::new();
        let inputs = CompileInputs {
            model_entry: entry,
            lane: lane.manifest_key(),
            plan_sha256: &self.plan_sha256,
            compiled_at: &self.compiled_at,
            refined_prompts: &empty,
        };
        for shot in &plan.shots {
            // A shot with no request at all is `staleness_findings`' finding, not a second one.
            let Some(request) = self.request(&shot.id) else {
                continue;
            };
            match compile_shot(plan, shot, &inputs, fps) {
                Ok(expected) => findings.extend(request_differences(request, &expected)),
                Err(mut shot_findings) => findings.append(&mut shot_findings),
            }
        }
        findings
    }
}

/// Every field of `actual` that a fresh compile would have written differently, except the three
/// the compile itself produces (`prompt`, `promptSource`, `authoredPrompt`).
///
/// `expected` is destructured exhaustively on purpose: a field added to [`CompiledRequest`] fails
/// to compile here until it is either compared or deliberately exempted, so the guarantee this
/// function states cannot quietly narrow as the request grows.
fn request_differences(
    actual: &CompiledRequest,
    expected: &CompiledRequest,
) -> Vec<PlanDiagnostic> {
    let CompiledRequest {
        shot_id,
        beat,
        mode,
        model,
        prompt: _,
        prompt_source: _,
        authored_prompt: _,
        negative_prompt,
        duration_seconds,
        fps,
        width,
        height,
        seed,
        first_frame_role,
        last_frame_role,
        reference_roles,
        chain_from_shot_id,
        continuity_roles,
    } = expected;
    let mut findings = Vec::new();
    let mut differ = |field: &str, found: String, planned: String| {
        if found != planned {
            findings.push(PlanDiagnostic::shot(
                shot_id,
                field,
                format!(
                    "the compiled request asks for {found}, but the plan says {planned}; re-run \
                     `film-harness compile` (only the prompt may differ from the plan)"
                ),
            ));
        }
    };
    differ("compiled.model", quoted(&actual.model), quoted(model));
    differ("compiled.beat", quoted(&actual.beat), quoted(beat));
    differ("compiled.mode", quoted(&actual.mode), quoted(mode));
    differ(
        "compiled.durationSeconds",
        format!("{}s", actual.duration_seconds),
        format!("{duration_seconds}s"),
    );
    differ(
        "compiled.fps",
        format!("{} fps", actual.fps),
        format!("{fps} fps"),
    );
    differ(
        "compiled.resolution",
        format!("{}x{}", actual.width, actual.height),
        format!("{width}x{height}"),
    );
    differ(
        "compiled.seed",
        format!("{:?}", actual.seed),
        format!("{seed:?}"),
    );
    differ(
        "compiled.negativePrompt",
        format!("{:?}", actual.negative_prompt),
        format!("{negative_prompt:?}"),
    );
    differ(
        "compiled.firstFrameRole",
        format!("{:?}", actual.first_frame_role),
        format!("{first_frame_role:?}"),
    );
    differ(
        "compiled.lastFrameRole",
        format!("{:?}", actual.last_frame_role),
        format!("{last_frame_role:?}"),
    );
    differ(
        "compiled.referenceRoles",
        format!("{:?}", actual.reference_roles),
        format!("{reference_roles:?}"),
    );
    differ(
        "compiled.chainFromShotId",
        format!("{:?}", actual.chain_from_shot_id),
        format!("{chain_from_shot_id:?}"),
    );
    differ(
        "compiled.continuityRoles",
        format!("{:?}", actual.continuity_roles),
        format!("{continuity_roles:?}"),
    );
    findings
}

fn quoted(value: &str) -> String {
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film_plan::{parse_plan, parse_reference_pack};

    fn plan_text() -> String {
        serde_json::to_string(&json!({
            "schemaVersion": 1,
            "id": "courier-workshop",
            "version": 2,
            "title": "Courier",
            "model": { "id": "minimax_h3", "tier": "q4", "fps": 24, "resolution": "576x320" },
            "limits": { "maxRunSeconds": 3600, "maxShotSeconds": 1800, "maxAttemptsPerShot": 1, "maxMemoryGb": 96 },
            "shots": [
                {
                    "id": "SH010", "beat": "enter", "framing": "wide", "prompt": "a courier enters",
                    "targetDurationSeconds": 5.1667, "startState": "empty", "endState": "courier inside",
                    "seed": 7, "conditioning": { "mode": "text_to_video" },
                    "continuityRoles": ["courier"]
                },
                {
                    "id": "SH020", "beat": "place", "framing": "medium", "prompt": "places the parcel",
                    "targetDurationSeconds": 5.875, "startState": "courier inside", "endState": "parcel on table",
                    "conditioning": {
                        "mode": "image_to_video",
                        "firstFrameRole": "workshop_plate",
                        "chainFromShotId": "SH010"
                    },
                    "continuityRoles": ["courier", "red_parcel"]
                }
            ]
        }))
        .unwrap()
    }

    fn pack() -> ReferencePack {
        parse_reference_pack(
            &json!({
                "schemaVersion": 1,
                "id": "courier-refs",
                "version": 3,
                "references": [
                    { "role": "courier", "kind": "character", "file": "references/courier.png" },
                    { "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png" },
                    { "role": "workshop_plate", "kind": "plate", "file": "references/plate.png" }
                ]
            })
            .to_string(),
        )
        .unwrap()
    }

    fn entry() -> JsonObject<String, Value> {
        json!({
            "id": "minimax_h3",
            "defaults": { "fps": 24, "resolution": "1344x768" },
            "limits": { "resolutions": ["1344x768", "576x320"] }
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    fn role_assets() -> BTreeMap<String, String> {
        [
            ("courier", "asset_courier"),
            ("red_parcel", "asset_parcel"),
            ("workshop_plate", "asset_plate"),
        ]
        .into_iter()
        .map(|(role, asset)| (role.to_owned(), asset.to_owned()))
        .collect()
    }

    fn compiled(refined: BTreeMap<String, String>) -> CompiledPlan {
        let plan = parse_plan(&plan_text()).unwrap();
        compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                model_entry: &entry(),
                lane: "mlx",
                plan_sha256: "abc123def456",
                compiled_at: "2026-09-13T00:00:00Z",
                refined_prompts: &refined,
            },
        )
        .expect("compiles")
    }

    #[test]
    fn a_plan_compiles_to_one_request_per_shot_inside_the_declared_geometry() {
        let compiled = compiled(BTreeMap::new());
        assert_eq!(compiled.requests.len(), 2);
        assert_eq!(compiled.plan_version, 2);
        assert_eq!(compiled.reference_pack_version, 3);
        assert_eq!(compiled.model.fps, 24);
        let first = compiled.request("SH010").unwrap();
        assert_eq!(first.prompt, "a courier enters");
        assert_eq!(first.prompt_source, PromptSource::Authored);
        assert_eq!(first.authored_prompt, None);
        assert_eq!((first.width, first.height), (576, 320));
        let second = compiled.request("SH020").unwrap();
        assert_eq!(second.first_frame_role.as_deref(), Some("workshop_plate"));
        assert_eq!(second.chain_from_shot_id.as_deref(), Some("SH010"));
        // Round-trips through JSON with no unknown fields.
        let text = serde_json::to_string(&compiled).unwrap();
        let back: CompiledPlan = serde_json::from_str(&text).unwrap();
        assert_eq!(back, compiled);
    }

    #[test]
    fn a_later_attempt_renders_at_the_plans_seed_offset_by_its_attempt_number() {
        let compiled = compiled(BTreeMap::new());
        let assets = role_assets();
        let body_for = |attempt: u32| {
            let context = DispatchContext {
                project_id: "proj_1",
                run_id: "run_abc",
                plan_id: "courier-workshop",
                plan_version: 2,
                attempt,
                tier: Some("q4"),
                idempotency_key: None,
                role_assets: &assets,
            };
            compiled
                .request("SH010")
                .unwrap()
                .to_job_body(&context)
                .unwrap()
        };
        // Attempt 1 is the plan's own seed; the replacement (attempt 2) must not be the same
        // render, because the MLX pipeline is deterministic for a seed.
        assert_eq!(body_for(1)["seed"], 7);
        assert_eq!(body_for(1)["advanced"]["filmHarness"]["seed"], 7);
        assert_eq!(body_for(2)["seed"], 8);
        assert_eq!(body_for(2)["advanced"]["filmHarness"]["seed"], 8);
        assert_eq!(body_for(5)["seed"], 11);
        // Nothing else about the request varies by attempt.
        let (one, two) = (body_for(1), body_for(2));
        for key in ["prompt", "mode", "duration", "fps", "width", "height"] {
            assert_eq!(one[key], two[key], "{key} must not vary by attempt");
        }
        // A request with no seed leaves the field out at every attempt.
        let mut unseeded = compiled.clone();
        unseeded.requests[0].seed = None;
        let context = DispatchContext {
            project_id: "proj_1",
            run_id: "run_abc",
            plan_id: "courier-workshop",
            plan_version: 2,
            attempt: 3,
            tier: Some("q4"),
            idempotency_key: None,
            role_assets: &assets,
        };
        let body = unseeded
            .request("SH010")
            .unwrap()
            .to_job_body(&context)
            .unwrap();
        assert!(body.get("seed").is_none());
        assert!(body["advanced"]["filmHarness"].get("seed").is_none());
    }

    #[test]
    fn the_job_body_is_the_compiled_request_with_roles_resolved() {
        let compiled = compiled(BTreeMap::new());
        let assets = role_assets();
        let context = DispatchContext {
            project_id: "proj_1",
            run_id: "run_abc",
            plan_id: "courier-workshop",
            plan_version: 2,
            attempt: 1,
            tier: Some("q4"),
            idempotency_key: Some("run_abc:SH010:a1"),
            role_assets: &assets,
        };
        let body = compiled
            .request("SH010")
            .unwrap()
            .to_job_body(&context)
            .unwrap();
        assert_eq!(body["projectId"], "proj_1");
        assert_eq!(body["mode"], "text_to_video");
        assert_eq!(body["prompt"], "a courier enters");
        assert_eq!(body["duration"], 5.1667);
        assert_eq!(body["fps"], 24);
        assert_eq!(body["width"], 576);
        assert_eq!(body["height"], 320);
        assert_eq!(body["fitMode"], "crop");
        assert_eq!(body["seed"], 7);
        assert_eq!(body["advanced"]["mlxQuantize"], 4);
        assert_eq!(body["advanced"]["filmHarness"]["shotId"], "SH010");
        assert_eq!(body["advanced"]["filmHarness"]["planVersion"], 2);
        // The replay key rides the dispatched payload: it is how a controller that died between the
        // POST and the record write finds its own job instead of enqueuing a second render.
        assert_eq!(
            body["advanced"]["filmHarness"]["idempotencyKey"],
            "run_abc:SH010:a1"
        );
        assert!(body.get("sourceAssetId").is_none());
        assert!(body["advanced"]["filmHarness"]
            .get("promptSource")
            .is_none());

        // The declared continuity roles ride the provenance too, so a take can say which approved
        // references it was meant to depict even when the model takes no reference conditioning.
        assert_eq!(
            body["advanced"]["filmHarness"]["continuityRoles"],
            json!(["courier"])
        );

        let body = compiled
            .request("SH020")
            .unwrap()
            .to_job_body(&context)
            .unwrap();
        assert_eq!(body["sourceAssetId"], "asset_plate");
        assert_eq!(body["advanced"]["filmHarness"]["chainFromShotId"], "SH010");
        assert_eq!(
            body["advanced"]["filmHarness"]["continuityRoles"],
            json!(["courier", "red_parcel"])
        );
        assert!(body.get("referenceAssetIds").is_none());

        // A shot that declares no continuity roles claims none in its provenance.
        let mut bare = compiled.request("SH010").unwrap().clone();
        bare.continuity_roles.clear();
        let body = bare.to_job_body(&context).unwrap();
        assert!(body["advanced"]["filmHarness"]
            .get("continuityRoles")
            .is_none());

        // A role that was never imported refuses the dispatch instead of sending a body with a
        // missing keyframe.
        let empty = BTreeMap::new();
        let context = DispatchContext {
            role_assets: &empty,
            ..context
        };
        let findings = compiled
            .request("SH020")
            .unwrap()
            .to_job_body(&context)
            .expect_err("missing asset refuses");
        assert_eq!(findings.len(), 1);
        assert!(
            findings[0].message.contains("never imported"),
            "{findings:?}"
        );
    }

    #[test]
    fn a_refined_prompt_replaces_the_text_and_keeps_the_authored_one() {
        let refined = [(
            "SH010".to_owned(),
            "  integrated_multimodal_description: a courier enters a warm workshop  ".to_owned(),
        )]
        .into_iter()
        .collect();
        let compiled = compiled(refined);
        let first = compiled.request("SH010").unwrap();
        assert_eq!(first.prompt_source, PromptSource::Refined);
        assert_eq!(
            first.prompt,
            "integrated_multimodal_description: a courier enters a warm workshop"
        );
        assert_eq!(first.authored_prompt.as_deref(), Some("a courier enters"));
        // The untouched shot still compiles its authored prompt.
        assert_eq!(
            compiled.request("SH020").unwrap().prompt_source,
            PromptSource::Authored
        );
        let assets = role_assets();
        let body = first
            .to_job_body(&DispatchContext {
                project_id: "p",
                run_id: "r",
                plan_id: "courier-workshop",
                plan_version: 2,
                attempt: 1,
                tier: None,
                idempotency_key: None,
                role_assets: &assets,
            })
            .unwrap();
        assert_eq!(body["advanced"]["filmHarness"]["promptSource"], "refined");
        assert!(body["advanced"].get("mlxQuantize").is_none());
    }

    #[test]
    fn an_unusable_refined_prompt_is_a_finding_not_a_silent_fallback() {
        let plan = parse_plan(&plan_text()).unwrap();
        for bad in ["   ", &"x".repeat(MAX_PROMPT_CHARS + 1)] {
            let refined: BTreeMap<String, String> =
                [("SH010".to_owned(), bad.to_owned())].into_iter().collect();
            let findings = compile_plan(
                &plan,
                &pack(),
                &CompileInputs {
                    model_entry: &entry(),
                    lane: "mlx",
                    plan_sha256: "abc",
                    compiled_at: "now",
                    refined_prompts: &refined,
                },
            )
            .expect_err("refuses");
            assert_eq!(findings.len(), 1, "{findings:?}");
            assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));
            assert!(findings[0].message.contains("the video route accepts"));
        }
    }

    #[test]
    fn stale_compiled_requests_are_refused_against_the_plan_they_claim() {
        let plan = parse_plan(&plan_text()).unwrap();
        let compiled = compiled(BTreeMap::new());
        assert!(compiled
            .staleness_findings(&plan, "abc123def456")
            .is_empty());
        let findings = compiled.staleness_findings(&plan, "999999999999");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0]
                .message
                .contains("has changed since it was compiled"),
            "{findings:?}"
        );

        let mut compiled = compiled;
        compiled
            .requests
            .retain(|request| request.shot_id != "SH020");
        let findings = compiled.staleness_findings(&plan, "abc123def456");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].shot_id.as_deref(), Some("SH020"));

        compiled.plan_id = "other".to_owned();
        assert!(compiled
            .staleness_findings(&plan, "abc123def456")
            .iter()
            .any(|finding| finding.field == "compiled.planId"));
    }

    #[test]
    fn a_hand_edited_request_is_refused_even_when_it_pins_the_right_plan() {
        let plan = parse_plan(&plan_text()).unwrap();
        let entry = entry();
        let clean = compiled(BTreeMap::new());
        assert!(clean
            .conformance_findings(&plan, &entry, ModelLane::Mlx)
            .is_empty());
        // The compile's own output is exempt: a refined prompt is why the document exists.
        let refined = compiled(
            [("SH010".to_owned(), "a rewritten courier".to_owned())]
                .into_iter()
                .collect(),
        );
        assert!(refined
            .conformance_findings(&plan, &entry, ModelLane::Mlx)
            .is_empty());

        // Every other field is a transcription of the plan, and an edit to one is named.
        // (field the finding must name, the hand edit, a fragment of the message it must carry)
        type TamperCase = (&'static str, fn(&mut CompiledRequest), &'static str);
        let cases: Vec<TamperCase> = vec![
            (
                "compiled.durationSeconds",
                |request| request.duration_seconds = 9.0,
                "9s",
            ),
            (
                "compiled.mode",
                |request| request.mode = "image_to_video".to_owned(),
                "image_to_video",
            ),
            (
                "compiled.model",
                |request| request.model = "ltx_2_5".to_owned(),
                "ltx_2_5",
            ),
            ("compiled.fps", |request| request.fps = 30, "30 fps"),
            (
                "compiled.resolution",
                |request| request.width = 1344,
                "1344x320",
            ),
            ("compiled.seed", |request| request.seed = Some(8), "Some(8)"),
            (
                "compiled.negativePrompt",
                |request| request.negative_prompt = Some("blurry".to_owned()),
                "blurry",
            ),
            (
                "compiled.firstFrameRole",
                |request| request.first_frame_role = Some("workshop_plate".to_owned()),
                "workshop_plate",
            ),
            (
                "compiled.lastFrameRole",
                |request| request.last_frame_role = Some("workshop_plate".to_owned()),
                "workshop_plate",
            ),
            (
                "compiled.referenceRoles",
                |request| request.reference_roles = vec!["courier".to_owned()],
                "courier",
            ),
            (
                "compiled.chainFromShotId",
                |request| request.chain_from_shot_id = Some("SH020".to_owned()),
                "SH020",
            ),
            (
                "compiled.continuityRoles",
                |request| request.continuity_roles.clear(),
                "[]",
            ),
            (
                "compiled.beat",
                |request| request.beat = "exit".to_owned(),
                "exit",
            ),
        ];
        for (field, edit, expected) in cases {
            let mut tampered = clean.clone();
            edit(&mut tampered.requests[0]);
            let findings = tampered.conformance_findings(&plan, &entry, ModelLane::Mlx);
            assert_eq!(findings.len(), 1, "{field}: {findings:?}");
            assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"), "{field}");
            assert_eq!(findings[0].field, field, "{findings:?}");
            assert!(
                findings[0].message.contains(expected),
                "{field}: {findings:?}"
            );
        }

        // The document's own model block is checked too: a swapped model, tier, fps or lane.
        let mut tampered = clean.clone();
        tampered.model.id = "ltx_2_5".to_owned();
        tampered.model.tier = Some("q8".to_owned());
        tampered.model.fps = 30;
        let fields: Vec<String> = tampered
            .conformance_findings(&plan, &entry, ModelLane::Candle)
            .into_iter()
            .map(|finding| finding.field)
            .collect();
        assert!(
            fields.contains(&"compiled.model.id".to_owned())
                && fields.contains(&"compiled.model.tier".to_owned())
                && fields.contains(&"compiled.model.fps".to_owned())
                && fields.contains(&"compiled.model.lane".to_owned()),
            "{fields:?}"
        );

        // A shot the compile does not cover is `staleness_findings`' report, not a second one here.
        let mut short = clean;
        short.requests.retain(|request| request.shot_id != "SH020");
        assert!(short
            .conformance_findings(&plan, &entry, ModelLane::Mlx)
            .is_empty());
    }

    #[test]
    fn tier_maps_to_the_shared_mlx_quantize_convention() {
        assert_eq!(mlx_quantize_for_tier("q4"), json!(4));
        assert_eq!(mlx_quantize_for_tier("q8"), json!(8));
        assert_eq!(mlx_quantize_for_tier("bf16"), json!(0));
    }
}
