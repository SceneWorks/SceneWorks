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
    shot_resolution, PlanDiagnostic, ProductionPlan, ReferencePack, Shot, MAX_PROMPT_CHARS,
};

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
        if self.prompt_source == PromptSource::Refined {
            provenance["promptSource"] = json!("refined");
        }
        if let Some(chain) = &self.chain_from_shot_id {
            // Recorded so a take's provenance says which shot it was meant to continue. It is not
            // conditioning: the keyframe/reference slots above are the only anchors.
            provenance["chainFromShotId"] = json!(chain);
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
        if let Some(seed) = self.seed {
            body["seed"] = json!(seed);
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
        assert!(body.get("sourceAssetId").is_none());
        assert!(body["advanced"]["filmHarness"]
            .get("promptSource")
            .is_none());

        let body = compiled
            .request("SH020")
            .unwrap()
            .to_job_body(&context)
            .unwrap();
        assert_eq!(body["sourceAssetId"], "asset_plate");
        assert_eq!(body["advanced"]["filmHarness"]["chainFromShotId"], "SH010");
        assert!(body.get("referenceAssetIds").is_none());

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
    fn tier_maps_to_the_shared_mlx_quantize_convention() {
        assert_eq!(mlx_quantize_for_tier("q4"), json!(4));
        assert_eq!(mlx_quantize_for_tier("q8"), json!(8));
        assert_eq!(mlx_quantize_for_tier("bf16"), json!(0));
    }
}
