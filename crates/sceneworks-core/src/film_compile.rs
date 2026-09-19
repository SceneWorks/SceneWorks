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
use sha2::{Digest, Sha256};

use crate::film_plan::{
    is_reference_partition_id, plan_lora_payload_entries, plan_loras_for_partition,
    shot_resolution, ModelEntries, ModelLane, PlanDiagnostic, ProductionPlan, ReferenceEntry,
    ReferencePack, Shot,
};
use crate::minimax_h3_turbo::resolve_turbo_recipe;
use crate::video_request::effective_reference_image_short_edge;
use crate::MAX_PROMPT_CHARS;

/// Schema version of [`CompiledPlan`] documents this module reads and writes.
///
/// **2** (sc-23402): `CompiledRequest::model` is the RESOLVED partition id rather than the plan's
/// declared family model, and `partitionReason` says why. The document's semantics changed, so a v1
/// document is refused BY VERSION. Without the bump `partitionReason`'s `#[serde(default)]` would
/// let a v1 document parse with an empty reason and then fail [`request_differences`] as
/// hand-edited, which blames the operator for a schema migration. The remedy either way is
/// `film-harness compile`.
/// **3** (sc-23406): a request carries the LoRAs it dispatches with and the step count it renders
/// at. Both are DERIVED fields [`request_differences`] compares, so a v2 document read under this
/// build would default them to "none / unknown" and then be blamed as hand-edited — the same
/// migration trap the v2 bump above exists to avoid. The remedy is the same one line:
/// `film-harness compile`.
/// **4** (sc-24023): a reference request's prompt now LEADS with compiler-written binding sentences
/// naming each reference's `<Picture N>`, and the inserted text is recorded separately in
/// `insertedText`. A v3 document's prompt has no binding sentences and no `insertedText` key, so
/// reading one under this build would default the field to empty and then blame the operator for a
/// hand edit through [`request_differences`] — the same migration trap as the two bumps above. The
/// remedy is the same one line: `film-harness compile`.
pub const COMPILED_PLAN_SCHEMA_VERSION: u32 = 4;

/// Serialize a production plan exactly as the project store and harness persist it, then hash
/// those bytes. Keeping this beside the compiler prevents the editor preflight and CLI harness
/// from inventing competing definitions of a "current" compiled document.
pub fn production_plan_sha256(plan: &ProductionPlan) -> Result<String, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(plan)?;
    bytes.push(b'\n');
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// How far apart one shot's successive attempts are seeded (sc-22715).
///
/// Attempt `n` of a shot renders at `seed + (n - 1) * ATTEMPT_SEED_STRIDE`, so the seed a run
/// dispatches identifies a (shot, attempt) pair rather than only a shot. Anything smaller than the
/// gap a plan leaves between its own per-shot seeds makes two different renders share a seed: the
/// shipped courier fixture seeds its six shots 22710..22715, so at a stride of 1 SH020's second
/// attempt and SH030's first were the same number.
pub const ATTEMPT_SEED_STRIDE: i64 = 1000;

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

/// One role bound to a reference picture, with the pack entry it names.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundReferenceRole<'a> {
    pub role: String,
    /// The pack entry for `role`, when the pack declares one. `None` only for a plan that never
    /// passed [`crate::film_plan::validate_plan_against_pack`], which refuses an undeclared role.
    pub entry: Option<&'a ReferenceEntry>,
}

/// One reference picture a shot dispatches: the 1-based number the engine labels it with, and the
/// pack roles it carries.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferencePicture<'a> {
    /// The `N` in `<Picture N>`, and the 1-based position of this picture's asset in the
    /// dispatched `referenceAssetIds`.
    pub number: u32,
    /// The roles bound to this picture, in the shot's own order. Exactly one today.
    pub roles: Vec<BoundReferenceRole<'a>>,
}

impl ReferencePicture<'_> {
    /// The role whose imported asset this picture dispatches — the first one bound to it.
    pub fn dispatch_role(&self) -> Option<&str> {
        self.roles.first().map(|bound| bound.role.as_str())
    }
}

/// THE order a shot's reference images are supplied in, and therefore both the `<Picture N>` the
/// engine labels each one with and the position that image's asset takes in the dispatched
/// `referenceAssetIds` (sc-24023).
///
/// One function, called by the compiler that writes the binding sentences AND by the dispatcher
/// that builds `referenceAssetIds`, because the two numbers are the same number: a prompt that says
/// "the courier is the person shown in `<Picture 2>`" while the courier's asset is dispatched first
/// binds the model to the wrong image, and nothing downstream can detect it. The MiniMax-H3 text
/// encoder labels the supplied assets `<Picture 1>`, `<Picture 2>`, … in supply order, so the
/// numbering is positional and there is no id to check it against.
///
/// Today every bound role has its own file, so a picture carries exactly one role and the order is
/// the shot's own role order.
pub fn shot_reference_pictures<'a>(
    reference_roles: &[String],
    pack: &'a ReferencePack,
) -> Vec<ReferencePicture<'a>> {
    reference_roles
        .iter()
        .enumerate()
        .map(|(index, role)| ReferencePicture {
            number: u32::try_from(index + 1).unwrap_or(u32::MAX),
            roles: vec![BoundReferenceRole {
                role: role.clone(),
                entry: pack.references.iter().find(|entry| &entry.role == role),
            }],
        })
        .collect()
}

/// A kind of text the COMPILER writes into a prompt, recorded so a reviewer can see exactly what
/// was added and why (sc-24023).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsertedTextKind {
    /// One sentence per bound reference role, giving that reference a job in the prompt by naming
    /// the `<Picture N>` the engine will label its image with.
    ReferenceBinding,
}

/// Text the compiler wrote into [`CompiledRequest::prompt`], kept beside the authored prompt so the
/// document says exactly what was added rather than only that the prompt differs (sc-24023).
///
/// Every insertion LEADS the prompt, in the order of this list, because the engine presents the
/// reference media before the text: the binding a picture needs should be the first thing said
/// about it. Inserted text is written AFTER the model's own refine rewrite, so the refiner can
/// never paraphrase a `<Picture N>` into something the engine does not label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InsertedText {
    pub kind: InsertedTextKind,
    pub text: String,
}

/// The noun a binding sentence calls a bound reference, by its pack kind.
///
/// Only [`crate::film_plan::BINDABLE_REFERENCE_KINDS`] reach a `conditioning.referenceRoles` slot —
/// `validate_plan_against_pack` refuses the other two — so the fallback is for a role the pack does
/// not declare at all, which no validated plan has.
fn bound_reference_noun(kind: &str) -> &'static str {
    match kind {
        "character" => "person",
        "prop" => "object",
        "location" => "place",
        _ => "subject",
    }
}

/// `red_parcel` -> `red parcel`: the role as a prompt reads it.
fn role_phrase(role: &str) -> String {
    role.replace(['_', '-'], " ")
}

/// A pack-authored description as compiler-owned text may repeat it: every `\r`, `\n`, `\t` and run
/// of spaces collapsed to a single space, and the ends trimmed (sc-24023).
///
/// THE one place a description is normalized, because every kind of inserted text repeats one — the
/// reference binding sentences here, and the audio bindings a later story adds — and a description
/// that carried a newline or a control run would otherwise land raw in the dispatched prompt inside
/// the one field [`CompiledPlan::conformance_findings`] treats as the compiler's own authored text
/// and therefore never re-reads.
///
/// WHITESPACE only. `<`, `>` and genuine control characters are refused at the document boundary by
/// [`crate::film_plan::reference_pack_findings`], because a description that forges `<Picture 3>` is
/// an authoring mistake to name rather than something to silently rewrite.
pub fn normalized_description(description: &str) -> String {
    description.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The binding sentences for one shot: one per bound role, in picture order, each naming the
/// `<Picture N>` that role's image will be labelled with and repeating the pack's own description
/// of it verbatim (sc-24023).
///
/// Plain declarative sentences and nothing else — no emphasis, no imperatives, no restating of the
/// shot. The prompt guide's rule is that a reference needs a job ("the woman from `<Picture 1>`");
/// this is that job, stated once per reference.
fn reference_binding_text(pictures: &[ReferencePicture<'_>]) -> Option<InsertedText> {
    let mut sentences: Vec<String> = Vec::new();
    for picture in pictures {
        for bound in &picture.roles {
            let kind = bound.entry.map_or("", |entry| entry.kind.as_str());
            let mut sentence = format!(
                "The {} is the {} shown in <Picture {}>.",
                role_phrase(&bound.role),
                bound_reference_noun(kind),
                picture.number
            );
            // The pack's description is the author's own words about that image, so it is repeated
            // rather than paraphrased into the sentence above — but its WHITESPACE is normalized
            // first: the sentence is one line of a prompt, and a description carrying a newline or a
            // tab run would put a raw control run into the dispatched text (sc-24023).
            let description = bound
                .entry
                .map(|entry| normalized_description(&entry.description))
                .unwrap_or_default();
            if !description.is_empty() {
                sentence.push(' ');
                sentence.push_str(&description);
                if !description.ends_with(['.', '!', '?']) {
                    sentence.push('.');
                }
            }
            sentences.push(sentence);
        }
    }
    (!sentences.is_empty()).then(|| InsertedText {
        kind: InsertedTextKind::ReferenceBinding,
        text: sentences.join(" "),
    })
}

/// Everything the compiler writes into a shot's prompt, in the order it leads with.
///
/// The one place an insertion kind is produced: a later kind of compiler-owned text is another
/// entry pushed here, and needs no change to the placement, the record or the conformance check.
fn inserted_text_for_shot(pictures: &[ReferencePicture<'_>]) -> Vec<InsertedText> {
    reference_binding_text(pictures).into_iter().collect()
}

/// The prompt the engine receives: the inserted text, in order, leading the refined or authored
/// prompt.
fn apply_inserted_text(prompt: &str, inserted: &[InsertedText]) -> String {
    if inserted.is_empty() {
        return prompt.to_owned();
    }
    let mut composed = String::new();
    for piece in inserted {
        composed.push_str(piece.text.trim());
        composed.push(' ');
    }
    composed.push_str(prompt);
    composed
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
    /// The catalog model id this request DISPATCHES as: the partition the shot resolved to, which
    /// on a split family (MiniMax-H3's `minimax_h3` / `minimax_h3_ref`) is not the plan's declared
    /// model (sc-23402). [`CompiledRequest::to_job_body_with`] writes exactly this into the job
    /// body's `model`, so `compiled.json` and the route agree by construction.
    pub model: String,
    /// The short edge this request's image references are encoded at, in pixels, when the plan asked
    /// for one (`model.advanced.referenceImageShortEdge`, sc-23402).
    ///
    /// Written ONLY for a request that resolved to the family's reference partition: the base
    /// partition encodes no reference, so carrying the knob there would dispatch a field the
    /// checkpoint has nothing to apply it to. Absent means the engine's own default, which
    /// [`CompiledRequest::effective_reference_image_short_edge`] resolves for the record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image_short_edge: Option<u32>,
    /// The catalog LoRA ids this request dispatches with, resolved PER PARTITION from the plan's
    /// one `model.loras` list (sc-23406).
    ///
    /// A step-distill adapter declares the partitions it was distilled for, so the same plan-level
    /// list produces the ref2v turbo on a `minimax_h3_ref` request and the fl2v turbo on a
    /// `minimax_h3` one — and neither on a request whose partition has no compatible entry, which
    /// is an empty list rather than a silent substitution.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<String>,
    /// `model.advanced.steps`, when the plan set one — the value DISPATCHED as `advanced.steps`.
    /// `None` leaves the step count to the recipe or the model default, which is what
    /// [`CompiledRequest::effective_steps`] records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// The model-evaluation count this request will actually render at: `steps` above, else the
    /// selected turbo recipe's own count, else the partition's declared `defaults.steps`.
    ///
    /// Resolved HERE rather than on read because only the compile holds all three inputs at once —
    /// the plan's override, the partition this shot resolved to, and that partition's catalog
    /// entry. The attempt record copies it, so the document and the record state one number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_steps: Option<u32>,
    /// The video sigma shift the selected turbo recipe imposes, when one applies to this request's
    /// partition. Absent in the base regime, where the engine's own constant governs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turbo_scheduler_shift: Option<f64>,
    /// Why this request resolved to `model` and not the family's other partition
    /// ([`crate::film_plan::ShotPartition::reason`]). Derived, like every other field but the
    /// prompt: [`CompiledPlan::conformance_findings`] refuses a hand-edited one.
    #[serde(default)]
    pub partition_reason: String,
    /// The prompt the engine will receive: [`Self::inserted_text`], in order, leading the authored
    /// or refined text.
    pub prompt: String,
    pub prompt_source: PromptSource,
    /// The plan's own prompt, kept when `prompt` was refined so the rewrite can be reviewed and
    /// reverted by editing the plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored_prompt: Option<String>,
    /// What the COMPILER wrote into `prompt`, per kind, separately from the authored text
    /// (sc-24023). A reviewer reads this to see exactly what was added without diffing two
    /// paragraphs, and it is DERIVED like every other field but the prompt itself:
    /// [`CompiledPlan::conformance_findings`] refuses a hand-edited one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inserted_text: Vec<InsertedText>,
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
    /// One entry per LLM call, preserving the actual planner checkpoint separately from the target
    /// video model. Thinking is kept out of the accepted plan text and stored only in its own field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub executions: Vec<PlannerExecutionRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannerExecutionRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    pub target_video_model_id: String,
    pub thinking_mode: String,
    /// Effective request bound advertised to an OpenAI-compatible planner. Native executions omit
    /// this because their token bound belongs to the worker job payload instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Effective per-request wall-clock bound, retained even for failed dispatch attempts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_timeout_seconds: Option<u64>,
    /// Effective sampler temperature when explicitly controlled by the adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Whether approved reference pixels were included in the attempted request payload.
    /// This is not an acknowledgement of server receipt after a network failure. Kept separate
    /// from reference roles so the external-data boundary remains explicit in provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_pixels_sent: Option<bool>,
    /// Wall-clock time spent waiting for this individual planner response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Sanitized provider completion reason, when the compatible endpoint reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Stable failure classification for a provider response that could not become plan text.
    /// Human-readable, actionable detail remains on the planning operation finding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Provider-reported token counts. Optional because OpenAI-compatible servers are allowed to
    /// omit usage, but when present the sanitized counts travel with the plan provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<PlannerUsageRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannerUsageRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

/// The model-evaluation count a catalog entry declares as its default (`defaults.steps`) — what
/// the engine renders at when nothing names a count (sc-23406).
fn default_steps(entry: &JsonObject<String, Value>) -> Option<u32> {
    entry
        .get("defaults")
        .and_then(Value::as_object)
        .and_then(|defaults| defaults.get("steps"))
        .and_then(Value::as_u64)
        .and_then(|steps| u32::try_from(steps).ok())
        .filter(|steps| *steps > 0)
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
    /// The catalog entries the plan's shots resolve against: the declared model, plus the family's
    /// reference partition when the catalog serves one. Each shot's geometry defaults come from the
    /// entry it will actually dispatch as (sc-23402).
    pub entries: &'a ModelEntries<'a>,
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
    let Some(fps) = crate::film_plan::plan_fps(plan, inputs.entries.base_entry()) else {
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
        match compile_shot(plan, shot, pack, inputs, fps) {
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
    pack: &ReferencePack,
    inputs: &CompileInputs<'_>,
    fps: u32,
) -> Result<CompiledRequest, Vec<PlanDiagnostic>> {
    // Which of the family's checkpoints this shot dispatches as, decided ONCE here and carried into
    // the request, the job body and the attempt record (sc-23402). A shot that binds no reference
    // roles stays on the plan's declared model: references are optional input, never a requirement.
    let (partition, partition_entry) = inputs.entries.resolve_shot(shot);
    let Some(partition_entry) = partition_entry else {
        return Err(vec![PlanDiagnostic::shot(
            &shot.id,
            "conditioning.referenceRoles",
            format!(
                "{} is not in this API's model catalog, so this shot cannot be compiled ({})",
                partition.model_id, partition.reason
            ),
        )]);
    };
    let Some((width, height)) = shot_resolution(plan, shot, partition_entry) else {
        return Err(vec![PlanDiagnostic::shot(
            &shot.id,
            "resolution",
            format!(
                "{} declares no default resolution; set one on the plan or the shot",
                partition.model_id
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
    // THE INSERTION STEP (sc-24023). It runs HERE — after the refine rewrite has already been
    // chosen above — because the refiner is a language model: text handed to it comes back
    // paraphrased, and a paraphrased `<Picture 2>` is a binding to an image the engine never
    // labelled that way. Writing it afterwards makes the compiler, not the model, the author of
    // every word the engine reads that the plan did not write.
    let inserted_text = inserted_text_for_shot(&shot_reference_pictures(
        &shot.conditioning.reference_roles,
        pack,
    ));
    let prompt = apply_inserted_text(&prompt, &inserted_text);
    let length = prompt.chars().count();
    if length > MAX_PROMPT_CHARS {
        return Err(vec![PlanDiagnostic::shot(
            &shot.id,
            "prompt",
            format!(
                "this shot's prompt is {length} characters once the compiler's binding sentences \
                 lead it, outside the 1-{MAX_PROMPT_CHARS} the video route accepts; shorten the \
                 shot's prompt or its references' descriptions (nothing is silently truncated)"
            ),
        )]);
    }
    // A reference-only knob reaches a reference-only request (sc-23402). The plan declares it once
    // on the family; the shots that resolve to the base partition encode no reference, so the field
    // is not written onto them and never reaches their job body or their attempt record.
    let reference_image_short_edge = plan
        .model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.reference_image_short_edge)
        .filter(|_| is_reference_partition_id(&partition.model_id));
    // The plan's one LoRA list, resolved against the partition this shot ACTUALLY dispatches as
    // (sc-23406). A shot whose partition has no compatible entry gets none — the empty list is the
    // record of that, and `effective_steps` below then resolves to the model's own default.
    let loras: Vec<String> = plan_loras_for_partition(&plan.model.loras, &partition.model_id)
        .into_iter()
        .map(|lora| lora.id.clone())
        .collect();
    // Resolved through the SAME resolver the worker calls, on the same payload shape, so the
    // schedule this document promises is the schedule the engine runs. A conflict is refused by
    // `validate_plan_structure` before a compile is attempted; reaching it here means the plan was
    // compiled anyway, and a finding beats compiling a request with two schedules in it.
    let payload_loras = plan_lora_payload_entries(&plan.model.loras, &partition.model_id);
    let recipe = match resolve_turbo_recipe(&partition.model_id, &payload_loras) {
        Ok(recipe) => recipe,
        Err(error) => {
            return Err(vec![PlanDiagnostic::shot(&shot.id, "model.loras", error)]);
        }
    };
    let steps = plan
        .model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.steps)
        .and_then(|steps| u32::try_from(steps).ok())
        .filter(|steps| *steps > 0);
    let effective_steps = steps
        .or_else(|| recipe.map(|recipe| recipe.steps))
        .or_else(|| default_steps(partition_entry));
    let turbo_scheduler_shift = recipe.map(|recipe| f64::from(recipe.video_shift));
    Ok(CompiledRequest {
        shot_id: shot.id.clone(),
        beat: shot.beat.clone(),
        mode: shot.conditioning.mode.clone(),
        model: partition.model_id,
        reference_image_short_edge,
        loras,
        steps,
        effective_steps,
        turbo_scheduler_shift,
        partition_reason: partition.reason,
        prompt,
        prompt_source,
        authored_prompt: authored,
        inserted_text,
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
    /// The approved pack the roles below were imported from. Dispatch reads it for ONE thing: the
    /// reference order ([`shot_reference_pictures`]), so the position an asset takes in
    /// `referenceAssetIds` is decided by the same function that numbered the `<Picture N>` in the
    /// prompt (sc-24023).
    pub pack: &'a ReferencePack,
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
    /// The reference-image short edge this request will actually render at, for the attempt record
    /// (sc-23402) — or `None` for a request that encodes no reference at all.
    ///
    /// `Some(requested)`, `Some(default)` and `None` are three different facts: a reference request
    /// that named no value still renders at the engine's default, and recording that number is what
    /// makes a run comparable against one that lowered it. A base-partition request has no reference
    /// to size, so it records nothing rather than a number that never applied.
    ///
    /// The default comes from [`effective_reference_image_short_edge`], the local twin of gen-core's
    /// `effective_reference_image_short_edge` (this crate has no gen-core dependency), so the
    /// recorded value cannot drift from the value the engine resolved.
    pub fn effective_reference_image_short_edge(&self) -> Option<u32> {
        is_reference_partition_id(&self.model)
            .then(|| effective_reference_image_short_edge(self.reference_image_short_edge))
    }

    /// Resolve this request's reference roles against the imported assets. A role with no asset is
    /// a finding — the run never dispatches a keyframe shot with its keyframe quietly missing.
    ///
    /// The reference list is built by walking [`shot_reference_pictures`] — the same function the
    /// compiler numbered this request's `<Picture N>` with — so the position of an asset here and
    /// the number in the prompt are one decision rather than two that agree by coincidence
    /// (sc-24023).
    pub fn resolve_conditioning(
        &self,
        pack: &ReferencePack,
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
        let pictures = shot_reference_pictures(&self.reference_roles, pack);
        let references: Vec<String> = pictures
            .iter()
            .filter_map(|picture| picture.dispatch_role())
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
        let assets = self.resolve_conditioning(context.pack, context.role_assets)?;
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
        if let Some(steps) = self.steps {
            // The plan-level override, dispatched on the same `advanced` convention the Video
            // Studio uses and read by the same worker branch (`minimax_h3_sampling`), where it
            // wins over a selected recipe's own count.
            advanced.insert("steps".to_owned(), json!(steps));
        }
        if let Some(edge) = self.reference_image_short_edge {
            // The same `advanced` convention as the tier (sc-23402): a request axis the engine reads
            // off the job, not a document axis. Only ever present on a reference-partition request,
            // because `compile_shot` is the only thing that writes the field.
            advanced.insert("referenceImageShortEdge".to_owned(), json!(edge));
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
        if !self.partition_reason.is_empty() {
            // The dispatched body says which of the family's checkpoints it asked for AND why
            // (sc-23402): `model` above is the resolved id, and this is the sentence that explains
            // it, so a job read back on its own carries the same two facts as the attempt record.
            provenance["partitionReason"] = json!(self.partition_reason);
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
        if !self.loras.is_empty() {
            // `{ id, weight }` per entry — the exact shape `generationStudio.jsx` posts for a
            // studio selection, so the route's `hydrate_lora_spec` hydrates it from the catalog the
            // same way and the worker's `resolve_turbo_recipe` reads the same ids. The weight is
            // the catalog's own `defaultWeight`, which is what the route would have filled in for
            // an id alone; sending it explicitly keeps the body readable beside a studio job.
            body["loras"] = json!(plan_lora_payload_entries(&self.loras, &self.model));
        }
        if let Some(negative) = self.negative_prompt.as_deref() {
            body["negativePrompt"] = json!(negative);
        }
        // Attempt `n` renders at `seed + (n - 1) * ATTEMPT_SEED_STRIDE` (sc-22715). The MLX render
        // is deterministic for a seed — two runs of the fixture's SH010 at seed 22710 were
        // pixel-identical — so a replacement that kept the plan's seed would re-render the very
        // take it rejects. The plan's seed is still attempt 1, the derivation is recorded
        // (`filmHarness.seed`, and the take's recipe), and it is the only thing about the
        // dispatched request that varies by attempt: the prompt, the geometry and the conditioning
        // are the compiled request's.
        //
        // The STRIDE is what keeps a seed unique per (shot, attempt) within a run. A stride of 1
        // collides with the plan's own per-shot seed spacing — the shipped fixture numbers its
        // shots 22710, 22711, … 22715, so SH020's second attempt and SH030's first were both
        // seed 22712, and the 2026-09-14 evaluation dispatched 22712, 22714 and 22715 twice each
        // in one run. A shot's attempts are therefore spaced far enough apart that no plausible
        // plan puts two shots inside one shot's attempt range.
        if let Some(seed) = self.seed {
            let attempt_seed = seed.wrapping_add(
                i64::from(context.attempt.saturating_sub(1)).wrapping_mul(ATTEMPT_SEED_STRIDE),
            );
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
                     {COMPILED_PLAN_SCHEMA_VERSION}); re-run `film-harness compile` to rewrite \
                     these requests",
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
    ///
    /// `insertedText` is NOT in that exemption (sc-24023): the compiler's own sentences are derived
    /// from the plan and the pack, a fresh compile reproduces them exactly, and a hand-edited
    /// `<Picture N>` would bind the model to the wrong image with nothing downstream able to tell.
    pub fn conformance_findings(
        &self,
        plan: &ProductionPlan,
        pack: &ReferencePack,
        entries: &ModelEntries<'_>,
        lane: ModelLane,
    ) -> Vec<PlanDiagnostic> {
        let entry = entries.base_entry();
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
            entries,
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
            match compile_shot(plan, shot, pack, &inputs, fps) {
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
        reference_image_short_edge,
        loras,
        steps,
        effective_steps,
        turbo_scheduler_shift,
        partition_reason,
        prompt: _,
        prompt_source: _,
        authored_prompt: _,
        inserted_text,
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
    differ(
        "compiled.partitionReason",
        quoted(&actual.partition_reason),
        quoted(partition_reason),
    );
    differ(
        "compiled.referenceImageShortEdge",
        format!("{:?}", actual.reference_image_short_edge),
        format!("{reference_image_short_edge:?}"),
    );
    differ(
        "compiled.loras",
        format!("{:?}", actual.loras),
        format!("{loras:?}"),
    );
    differ(
        "compiled.steps",
        format!("{:?}", actual.steps),
        format!("{steps:?}"),
    );
    differ(
        "compiled.effectiveSteps",
        format!("{:?}", actual.effective_steps),
        format!("{effective_steps:?}"),
    );
    differ(
        "compiled.turboSchedulerShift",
        format!("{:?}", actual.turbo_scheduler_shift),
        format!("{turbo_scheduler_shift:?}"),
    );
    differ(
        "compiled.insertedText",
        format!("{:?}", actual.inserted_text),
        format!("{inserted_text:?}"),
    );
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
            // `steps` mirrors the shipped catalog: it is what the engine renders at when nothing
            // names a count, and therefore what `effective_steps` records in the base regime.
            "defaults": { "fps": 24, "resolution": "1344x768", "steps": 50 },
            "limits": { "resolutions": ["1344x768", "576x320"] }
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    fn reference_entry() -> JsonObject<String, Value> {
        json!({
            "id": "minimax_h3_ref",
            "defaults": { "fps": 24, "resolution": "1344x768", "steps": 50 },
            "limits": { "resolutions": ["1344x768", "576x320"], "maxReferenceAssets": 9 }
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    /// The fixture pack, borrowed for a [`DispatchContext`]'s lifetime.
    fn pack_ref() -> &'static ReferencePack {
        static PACK: std::sync::OnceLock<ReferencePack> = std::sync::OnceLock::new();
        PACK.get_or_init(pack)
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
        let entry = entry();
        compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &ModelEntries::single("minimax_h3", &entry),
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

    /// The attempt offset is a STRIDE, not `+1` (sc-22715). The shipped courier fixture seeds its
    /// shots one apart (22710..22715), so a stride of 1 made SH020's second attempt and SH030's
    /// first the same seed — the 2026-09-14 evaluation run dispatched 22712, 22714 and 22715 twice
    /// each. The seed a run dispatches has to identify the (shot, attempt) pair it came from.
    #[test]
    fn a_later_attempt_renders_at_the_plans_seed_offset_by_its_attempt_number() {
        let compiled = compiled(BTreeMap::new());
        let assets = role_assets();
        // Every attempt of a shot whose plan seed is `base`, dispatched through the real body
        // builder. Taking it from `to_job_body` rather than computing it here is the point: the
        // derivation is what is under test, not the constant.
        let seeds_from = |base: i64| -> Vec<i64> {
            let mut request = compiled.request("SH010").unwrap().clone();
            request.seed = Some(base);
            (1..=8u32)
                .map(|attempt| {
                    let context = DispatchContext {
                        project_id: "proj_1",
                        run_id: "run_abc",
                        plan_id: "courier-workshop",
                        plan_version: 2,
                        attempt,
                        tier: Some("q4"),
                        idempotency_key: None,
                        pack: pack_ref(),
                        role_assets: &assets,
                    };
                    request.to_job_body(&context).unwrap()["seed"]
                        .as_i64()
                        .expect("a seeded request dispatches a seed")
                })
                .collect()
        };
        let body_for = |attempt: u32| {
            let context = DispatchContext {
                project_id: "proj_1",
                run_id: "run_abc",
                plan_id: "courier-workshop",
                plan_version: 2,
                attempt,
                tier: Some("q4"),
                idempotency_key: None,
                pack: pack_ref(),
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
        assert_eq!(body_for(2)["seed"], 1007);
        assert_eq!(body_for(2)["advanced"]["filmHarness"]["seed"], 1007);
        assert_eq!(body_for(5)["seed"], 4007);
        // A plan that numbers its shots one apart — which every fixture and the evaluation plan do
        // (22710…22715) — must not have one shot's later attempts land on the next shot's renders.
        // At an offset of `n − 1` they did: SH020-a2, SH030-a1 and three more pairs were the same
        // seed in one run, so a dispatched seed no longer said which render it belonged to.
        let (first, next) = (seeds_from(22710), seeds_from(22711));
        for seed in &first {
            assert!(
                !next.contains(seed),
                "seed {seed} is dispatched for two different shots of the same run: {first:?} vs \
                 {next:?}"
            );
        }
        assert_eq!(
            first
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            first.len(),
            "no shot repeats a seed across its own attempts: {first:?}"
        );
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
            pack: pack_ref(),
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
            pack: pack_ref(),
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
            pack: pack_ref(),
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
                pack: pack_ref(),
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
                    entries: &ModelEntries::single("minimax_h3", &entry()),
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

    /// sc-23402 review. A `compiled.json` written by a PRE-STORY build is refused by SCHEMA
    /// VERSION, not blamed on the operator as a hand edit.
    ///
    /// Such a document has no `partitionReason` key at all. `#[serde(default)]` reads it back as
    /// `""`, which `request_differences` would report as `compiled.partitionReason` — "the
    /// compiled request asks for …, but the plan says …", a tampering message — so a phase-1 run
    /// directory could no longer be resumed and the refusal named the wrong cause. The version
    /// bump to 2 is what makes the first finding the true one, and it names the remedy.
    #[test]
    fn a_pre_story_compiled_document_is_refused_by_schema_version_not_as_tampered() {
        let plan = parse_plan(&plan_text()).unwrap();
        let entry = entry();
        let entries = ModelEntries::single("minimax_h3", &entry);

        // Exactly what a v1 document on disk deserializes to: version 1, and the key absent.
        let mut v1 = compiled(BTreeMap::new());
        v1.schema_version = 1;
        for request in &mut v1.requests {
            request.partition_reason = String::new();
        }
        let round_tripped: CompiledPlan =
            serde_json::from_value(serde_json::to_value(&v1).unwrap()).unwrap();
        assert_eq!(round_tripped.schema_version, 1);
        assert!(round_tripped.requests[0].partition_reason.is_empty());

        let findings = round_tripped.staleness_findings(&plan, "abc123def456");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].field, "compiled.schemaVersion", "{findings:?}");
        assert!(
            findings[0].message.contains("schema version 1")
                && findings[0].message.contains("film-harness compile"),
            "the refusal must name the version AND the remedy: {findings:?}"
        );
        assert!(
            !findings
                .iter()
                .any(|finding| finding.field == "compiled.partitionReason"),
            "a schema migration must never be reported as a hand edit: {findings:?}"
        );

        // And this is the finding the bump replaced: at the CURRENT version the same empty
        // `partitionReason` is (correctly) a tampering report, which is why v1 had to be refused
        // by version rather than left to fall through to conformance.
        let mut current = round_tripped;
        current.schema_version = COMPILED_PLAN_SCHEMA_VERSION;
        assert!(current.staleness_findings(&plan, "abc123def456").is_empty());
        assert!(
            current
                .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
                .iter()
                .any(|finding| finding.field == "compiled.partitionReason"),
            "without the bump a v1 document lands here instead"
        );
    }

    #[test]
    fn a_hand_edited_request_is_refused_even_when_it_pins_the_right_plan() {
        let plan = parse_plan(&plan_text()).unwrap();
        let entry = entry();
        let entries = ModelEntries::single("minimax_h3", &entry);
        let clean = compiled(BTreeMap::new());
        assert!(clean
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());
        // The compile's own output is exempt: a refined prompt is why the document exists.
        let refined = compiled(
            [("SH010".to_owned(), "a rewritten courier".to_owned())]
                .into_iter()
                .collect(),
        );
        assert!(refined
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
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
            (
                "compiled.partitionReason",
                |request| request.partition_reason = "because I said so".to_owned(),
                "because I said so",
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
            let findings = tampered.conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx);
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
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Candle)
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
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());
    }

    /// A mixed plan: SH010 binds two reference roles, SH020 binds none (sc-23402, AC1).
    fn mixed_plan_text() -> String {
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
                    "conditioning": {
                        "mode": "reference_to_video",
                        // Both BINDABLE kinds (character, prop): a `plate` is refused in this
                        // slot by `validate_plan_against_pack`.
                        "referenceRoles": ["courier", "red_parcel"]
                    },
                    "continuityRoles": ["courier"]
                },
                {
                    "id": "SH020", "beat": "place", "framing": "medium", "prompt": "places the parcel",
                    "targetDurationSeconds": 5.875, "startState": "courier inside", "endState": "parcel on table",
                    "conditioning": { "mode": "text_to_video" },
                    "continuityRoles": ["courier", "red_parcel"]
                }
            ]
        }))
        .unwrap()
    }

    /// The mixed plan with `model.advanced.referenceImageShortEdge` set, compiled against both
    /// partitions.
    fn mixed_compiled_with_short_edge(edge: Option<u32>) -> CompiledPlan {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        if let Some(edge) = edge {
            document["model"]["advanced"] = json!({ "referenceImageShortEdge": edge });
        }
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &ModelEntries::with_reference_partition(
                    "minimax_h3",
                    &base,
                    Some(("minimax_h3_ref", &reference)),
                ),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("the mixed plan compiles")
    }

    fn short_edge_context<'a>(assets: &'a BTreeMap<String, String>) -> DispatchContext<'a> {
        DispatchContext {
            project_id: "proj_1",
            run_id: "run_abc",
            plan_id: "courier-workshop",
            plan_version: 2,
            attempt: 1,
            tier: Some("q4"),
            idempotency_key: Some("run_abc:SH010:a1"),
            pack: pack_ref(),
            role_assets: assets,
        }
    }

    /// The mixed plan with a turbo selection and, optionally, a `model.advanced.steps` override.
    fn mixed_compiled_with_turbo(loras: Value, steps: Option<i64>) -> CompiledPlan {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["model"]["loras"] = loras;
        if let Some(steps) = steps {
            document["model"]["advanced"] = json!({ "steps": steps });
        }
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &ModelEntries::with_reference_partition(
                    "minimax_h3",
                    &base,
                    Some(("minimax_h3_ref", &reference)),
                ),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("the turbo plan compiles")
    }

    /// 🔴 sc-23406. ONE plan-level LoRA list, resolved PER PARTITION — into the compiled request,
    /// into the dispatched body, and into the schedule each request records.
    ///
    /// Every assertion here is a silent failure if it goes the other way: the ref2v adapter on the
    /// base checkpoint folds cleanly at the wrong quality, and an fl2v adapter on the reference one
    /// does the same in the other direction (sc-19563). The step count and the shift are asserted
    /// as VALUES rather than as "not the default", because 4/12.0 is what the catalog declares for
    /// both of these files and 50/absent is what the base regime is.
    #[test]
    fn the_plans_loras_resolve_per_partition_into_the_request_and_the_body() {
        let compiled = mixed_compiled_with_turbo(
            json!(["minimax_h3_ref2v_turbo_4step", "minimax_h3_turbo_4step_v01"]),
            None,
        );
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();
        assert_eq!(referenced.model, "minimax_h3_ref");
        assert_eq!(referenced.loras, vec!["minimax_h3_ref2v_turbo_4step"]);
        assert_eq!(plain.model, "minimax_h3");
        assert_eq!(plain.loras, vec!["minimax_h3_turbo_4step_v01"]);
        // The schedule each one will actually run, from the catalog's own declaration.
        assert_eq!(referenced.effective_steps, Some(4));
        assert_eq!(referenced.turbo_scheduler_shift, Some(12.0));
        assert_eq!(plain.effective_steps, Some(4));
        assert_eq!(plain.turbo_scheduler_shift, Some(12.0));
        assert_eq!(referenced.steps, None, "no plan-level override was set");

        // The body: `{ id, weight }`, the shape `generationStudio.jsx` posts, on the partition it
        // belongs to and NOWHERE else.
        let assets = role_assets();
        let context = short_edge_context(&assets);
        let referenced_body = referenced.to_job_body(&context).expect("SH010 body");
        let plain_body = plain.to_job_body(&context).expect("SH020 body");
        assert_eq!(
            referenced_body["loras"],
            json!([{ "id": "minimax_h3_ref2v_turbo_4step", "weight": 1.0 }])
        );
        assert_eq!(
            plain_body["loras"],
            json!([{ "id": "minimax_h3_turbo_4step_v01", "weight": 1.0 }])
        );
        assert!(
            referenced_body["advanced"].get("steps").is_none(),
            "no override ⇒ no advanced.steps; the recipe governs: {}",
            referenced_body["advanced"]
        );

        // A plan that names ONLY the ref2v adapter leaves the base shot with none — an empty list,
        // recorded as such, rather than the ref2v file quietly attaching to the wrong checkpoint.
        let compiled = mixed_compiled_with_turbo(json!(["minimax_h3_ref2v_turbo_4step"]), None);
        let plain = compiled.request("SH020").unwrap();
        assert!(plain.loras.is_empty());
        assert_eq!(
            plain.effective_steps,
            Some(50),
            "no recipe applies, so the model's declared default governs"
        );
        assert_eq!(plain.turbo_scheduler_shift, None);
        let body = plain.to_job_body(&context).expect("SH020 body");
        assert!(
            body.get("loras").is_none(),
            "an empty list writes no field: {body}"
        );
    }

    /// `model.advanced.steps` overrides the recipe's own count — the plan-level twin of the knob
    /// `minimax_h3_sampling` already honours — and rides `advanced.steps` on every partition,
    /// including the one no accelerator reached.
    #[test]
    fn a_plan_level_steps_override_wins_over_the_recipe_and_rides_the_body() {
        let compiled = mixed_compiled_with_turbo(json!(["minimax_h3_ref2v_turbo_4step"]), Some(6));
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();
        assert_eq!(referenced.steps, Some(6));
        assert_eq!(
            referenced.effective_steps,
            Some(6),
            "the override wins over the recipe's 4"
        );
        assert_eq!(
            referenced.turbo_scheduler_shift,
            Some(12.0),
            "the SHIFT is not overridable: a distilled checkpoint keeps its trained shift"
        );
        assert_eq!(
            plain.effective_steps,
            Some(6),
            "the override wins over the model's 50 as well"
        );
        let assets = role_assets();
        let context = short_edge_context(&assets);
        for request in [referenced, plain] {
            let body = request.to_job_body(&context).expect("body");
            assert_eq!(body["advanced"]["steps"], json!(6), "{}", request.shot_id);
        }
    }

    /// A plan that declares no LoRAs compiles exactly as it did before the field existed, and a
    /// hand edit to any of the four new derived fields is caught as a hand edit.
    #[test]
    fn a_plan_without_loras_compiles_unchanged_and_the_derived_fields_are_conformance_checked() {
        let compiled = mixed_compiled_with_turbo(json!([]), None);
        for request in &compiled.requests {
            assert!(request.loras.is_empty(), "{}", request.shot_id);
            assert_eq!(request.steps, None, "{}", request.shot_id);
            assert_eq!(request.effective_steps, Some(50), "{}", request.shot_id);
            assert_eq!(request.turbo_scheduler_shift, None, "{}", request.shot_id);
        }
        let document = serde_json::to_value(&compiled).expect("serializes");
        assert!(
            document["requests"][0].get("loras").is_none()
                && document["requests"][0].get("turboSchedulerShift").is_none(),
            "absent values write no fields: {}",
            document["requests"][0]
        );

        // Conformance: each derived field is compared, so a hand-edited compiled document is
        // refused by the field that was edited rather than dispatched.
        let mut plan_document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        plan_document["model"]["loras"] = json!(["minimax_h3_ref2v_turbo_4step"]);
        let plan = parse_plan(&plan_document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        let clean = mixed_compiled_with_turbo(json!(["minimax_h3_ref2v_turbo_4step"]), None);
        type Tamper = fn(&mut CompiledRequest);
        let cases: Vec<(&str, Tamper)> = vec![
            ("compiled.loras", |request| request.loras.clear()),
            ("compiled.steps", |request| request.steps = Some(9)),
            ("compiled.effectiveSteps", |request| {
                request.effective_steps = Some(50)
            }),
            ("compiled.turboSchedulerShift", |request| {
                request.turbo_scheduler_shift = None
            }),
        ];
        for (field, edit) in cases {
            let mut tampered = clean.clone();
            edit(&mut tampered.requests[0]);
            let fields: Vec<String> = tampered
                .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
                .into_iter()
                .map(|finding| finding.field)
                .collect();
            assert!(fields.contains(&field.to_owned()), "{field}: {fields:?}");
        }
    }

    /// sc-23402. The plan declares the short edge ONCE on the family; it reaches only the request
    /// that resolved to the reference partition, and only that request's job body.
    #[test]
    fn the_reference_short_edge_reaches_only_the_reference_partitions_request() {
        let compiled = mixed_compiled_with_short_edge(Some(1536));
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();
        assert_eq!(referenced.model, "minimax_h3_ref");
        assert_eq!(referenced.reference_image_short_edge, Some(1536));
        assert_eq!(plain.model, "minimax_h3");
        assert_eq!(
            plain.reference_image_short_edge, None,
            "the base partition encodes no reference, so it carries no short edge"
        );

        let assets = role_assets();
        let context = short_edge_context(&assets);
        let body = referenced.to_job_body(&context).unwrap();
        assert_eq!(body["advanced"]["referenceImageShortEdge"], json!(1536));
        let body = plain.to_job_body(&context).unwrap();
        assert!(
            body["advanced"].get("referenceImageShortEdge").is_none(),
            "{}",
            body["advanced"]
        );

        // The document survives a conformance check, and a hand-edited value is caught by name —
        // the compiled document, not the plan, is what becomes the job body.
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["model"]["advanced"] = json!({ "referenceImageShortEdge": 1536 });
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        assert!(compiled
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());
        let mut tampered = compiled.clone();
        tampered.requests[0].reference_image_short_edge = Some(1024);
        let fields: Vec<String> = tampered
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .into_iter()
            .map(|finding| finding.field)
            .collect();
        assert_eq!(
            fields,
            vec!["compiled.referenceImageShortEdge".to_owned()],
            "{fields:?}"
        );
    }

    /// A plan that names no short edge compiles and dispatches exactly what it did before sc-23402,
    /// and the EFFECTIVE value a reference attempt records is the engine's own default — 2048, the
    /// same number `sceneworks_gen_core::effective_reference_image_short_edge` resolves (this crate
    /// has no gen-core dependency, so the rule is applied locally).
    #[test]
    fn an_absent_short_edge_dispatches_nothing_and_records_the_default_2048() {
        let compiled = mixed_compiled_with_short_edge(None);
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();
        assert_eq!(referenced.reference_image_short_edge, None);
        assert_eq!(plain.reference_image_short_edge, None);

        let assets = role_assets();
        let context = short_edge_context(&assets);
        for request in [referenced, plain] {
            let body = request.to_job_body(&context).unwrap();
            assert!(
                body["advanced"].get("referenceImageShortEdge").is_none(),
                "an absent knob dispatches no key: {}",
                body["advanced"]
            );
        }

        assert_eq!(
            referenced.effective_reference_image_short_edge(),
            Some(2048),
            "a reference request with no value still renders at the engine's default"
        );
        assert_eq!(
            plain.effective_reference_image_short_edge(),
            None,
            "a base-partition request records no short edge at all"
        );
        let asked = mixed_compiled_with_short_edge(Some(1024));
        assert_eq!(
            asked
                .request("SH010")
                .unwrap()
                .effective_reference_image_short_edge(),
            Some(1024),
            "a requested value is recorded verbatim"
        );
        assert_eq!(
            asked
                .request("SH020")
                .unwrap()
                .effective_reference_image_short_edge(),
            None
        );
    }

    #[test]
    fn each_shot_compiles_to_the_partition_its_own_conditioning_needs() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = ModelEntries::with_reference_partition(
            "minimax_h3",
            &base,
            Some(("minimax_h3_ref", &reference)),
        );
        let compiled = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &entries,
                lane: "mlx",
                plan_sha256: "abc123def456",
                compiled_at: "2026-09-14T00:00:00Z",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("compiles");
        // The plan still declares the FAMILY once; only the requests differ.
        assert_eq!(compiled.model.id, "minimax_h3");

        let referenced = compiled.request("SH010").unwrap();
        assert_eq!(referenced.model, "minimax_h3_ref");
        assert_eq!(referenced.mode, "reference_to_video");
        assert_eq!(
            referenced.reference_roles,
            vec!["courier".to_owned(), "red_parcel".to_owned()]
        );
        assert!(
            referenced.partition_reason.contains("minimax_h3_ref")
                && referenced.partition_reason.contains("2 reference role"),
            "{}",
            referenced.partition_reason
        );

        let plain = compiled.request("SH020").unwrap();
        assert_eq!(plain.model, "minimax_h3");
        assert_eq!(plain.mode, "text_to_video");
        assert!(plain.reference_roles.is_empty());
        assert!(
            plain.partition_reason.contains("no reference roles"),
            "{}",
            plain.partition_reason
        );

        // Both partitions reach the route under their own id, in role ORDER, and the reason rides
        // the payload beside it.
        let assets = role_assets();
        let context = DispatchContext {
            project_id: "proj_1",
            run_id: "run_abc",
            plan_id: "courier-workshop",
            plan_version: 2,
            attempt: 1,
            tier: Some("q4"),
            idempotency_key: Some("run_abc:SH010:a1"),
            pack: pack_ref(),
            role_assets: &assets,
        };
        let body = referenced.to_job_body(&context).unwrap();
        assert_eq!(body["model"], "minimax_h3_ref");
        assert_eq!(body["mode"], "reference_to_video");
        assert_eq!(
            body["referenceAssetIds"],
            json!(["asset_courier", "asset_parcel"])
        );
        assert_eq!(
            body["advanced"]["filmHarness"]["partitionReason"],
            json!(referenced.partition_reason)
        );
        let body = plain.to_job_body(&context).unwrap();
        assert_eq!(body["model"], "minimax_h3");
        assert!(body.get("referenceAssetIds").is_none());

        // Re-compiling the same plan says the same thing, so a conformance check on this document
        // is clean — and a document whose reference shot was re-pointed at the base checkpoint is
        // not.
        assert!(compiled
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());
        let mut tampered = compiled.clone();
        tampered.requests[0].model = "minimax_h3".to_owned();
        let fields: Vec<String> = tampered
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .into_iter()
            .map(|finding| finding.field)
            .collect();
        assert_eq!(fields, vec!["compiled.model".to_owned()], "{fields:?}");
    }

    #[test]
    fn a_reference_shot_refuses_to_compile_when_the_partition_is_not_in_the_catalog() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let findings = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &ModelEntries::single("minimax_h3", &base),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect_err("refuses");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));
        assert!(
            findings[0].message.contains("minimax_h3_ref")
                && findings[0]
                    .message
                    .contains("not in this API's model catalog"),
            "{findings:?}"
        );
    }

    /// The mixed plan's `role_assets`, borrowed for a [`DispatchContext`]'s lifetime.
    fn mixed_entries<'a>(
        base: &'a JsonObject<String, Value>,
        reference: &'a JsonObject<String, Value>,
    ) -> ModelEntries<'a> {
        ModelEntries::with_reference_partition(
            "minimax_h3",
            base,
            Some(("minimax_h3_ref", reference)),
        )
    }

    /// sc-24023, E2. A reference shot's prompt LEADS with one binding sentence per bound role, and
    /// the `<Picture N>` each sentence names is the 1-based position that role's asset takes in the
    /// dispatched `referenceAssetIds` — both read off [`shot_reference_pictures`], which is why
    /// they cannot disagree. A shot that resolved to the base checkpoint says nothing about
    /// pictures, because it sends none.
    ///
    /// The numbering is positional and unverifiable downstream: the MiniMax-H3 text encoder labels
    /// the assets it is handed `<Picture 1>`, `<Picture 2>`, … in supply order, so a prompt that
    /// binds the courier to `<Picture 2>` while the courier's asset is dispatched first renders a
    /// confidently wrong film with nothing to flag.
    #[test]
    fn a_reference_shots_prompt_leads_with_a_binding_sentence_numbered_as_the_asset_is_dispatched()
    {
        let compiled = mixed_compiled_with_short_edge(None);
        let referenced = compiled.request("SH010").unwrap();
        let plain = compiled.request("SH020").unwrap();

        // One insertion, of one kind, holding one sentence per bound role in picture order.
        assert_eq!(referenced.inserted_text.len(), 1, "{referenced:?}");
        assert_eq!(
            referenced.inserted_text[0].kind,
            InsertedTextKind::ReferenceBinding
        );
        assert_eq!(
            referenced.inserted_text[0].text,
            "The courier is the person shown in <Picture 1>. The red parcel is the object shown in \
             <Picture 2>."
        );
        // It LEADS: the engine presents the pictures before the text, so the binding is the first
        // thing the prompt says, and the authored text survives verbatim behind it.
        assert_eq!(
            referenced.prompt,
            "The courier is the person shown in <Picture 1>. The red parcel is the object shown in \
             <Picture 2>. a courier enters"
        );
        assert!(
            referenced
                .prompt
                .starts_with(&referenced.inserted_text[0].text),
            "{}",
            referenced.prompt
        );

        // THE criterion: N == the 1-based position in the dispatched list. Read out of the job body
        // this compiled request builds rather than recomputed here, so the assertion is against the
        // shipped list builder. This is the COMPILED REQUEST's body, not the harness's own dispatch:
        // the harness resolves through the same `resolve_conditioning` and posts `to_job_body_with`,
        // and the end-to-end assertion in `apps/rust-api/src/tests/film_harness.rs` is what pins the
        // agreement on the body a real run actually sent.
        let assets = role_assets();
        let context = short_edge_context(&assets);
        let body = referenced.to_job_body(&context).expect("SH010 body");
        let dispatched: Vec<String> = body["referenceAssetIds"]
            .as_array()
            .expect("a reference request dispatches its assets")
            .iter()
            .map(|id| id.as_str().expect("asset ids are strings").to_owned())
            .collect();
        assert_eq!(dispatched, vec!["asset_courier", "asset_parcel"]);
        for (index, role) in referenced.reference_roles.iter().enumerate() {
            let number = index + 1;
            assert!(
                referenced.inserted_text[0]
                    .text
                    .contains(&format!("{} is the", role_phrase(role))),
                "{role} is never named: {}",
                referenced.inserted_text[0].text
            );
            let sentence_at = referenced.inserted_text[0]
                .text
                .find(&format!("{} is the", role_phrase(role)))
                .expect("the role is named");
            let picture_at = referenced.inserted_text[0].text[sentence_at..]
                .find(&format!("<Picture {number}>"))
                .map(|at| at + sentence_at);
            assert!(
                picture_at.is_some(),
                "{role} must be bound to <Picture {number}>, the position its asset takes in \
                 {dispatched:?}: {}",
                referenced.inserted_text[0].text
            );
            assert_eq!(
                dispatched[index],
                *assets.get(role).expect("every role imported"),
                "{role} is dispatched at position {number}"
            );
        }

        // The base partition sends no pictures, so it claims none and its prompt is the plan's.
        assert!(plain.inserted_text.is_empty(), "{plain:?}");
        assert_eq!(plain.prompt, "places the parcel");
        let body = plain.to_job_body(&context).expect("SH020 body");
        assert!(body.get("referenceAssetIds").is_none(), "{body}");
        assert!(
            !body["prompt"]
                .as_str()
                .expect("a prompt is dispatched")
                .contains("<Picture"),
            "{body}"
        );

        // The document round-trips with the new field and nothing unknown.
        let text = serde_json::to_string(&compiled).unwrap();
        let back: CompiledPlan = serde_json::from_str(&text).unwrap();
        assert_eq!(back, compiled);
        let document: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            document["requests"][0]["insertedText"][0]["kind"],
            json!("reference_binding")
        );
        assert!(
            document["requests"][1].get("insertedText").is_none(),
            "a shot with no insertion writes no field: {}",
            document["requests"][1]
        );
    }

    /// sc-24023, E5. The binding sentences are written AFTER the refine rewrite, so the refiner
    /// cannot paraphrase a `<Picture N>` into a label the engine never applies — and the record
    /// keeps the three texts apart: the plan's, the model's rewrite, and the compiler's own.
    #[test]
    fn the_binding_sentences_are_written_after_the_refine_rewrite_and_recorded_apart_from_it() {
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["model"]["advanced"] = json!({ "referenceImageShortEdge": 1536 });
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let refined: BTreeMap<String, String> = [(
            "SH010".to_owned(),
            "integrated_multimodal_description: a courier steps into a warm workshop".to_owned(),
        )]
        .into_iter()
        .collect();
        let compiled = compile_plan(
            &plan,
            &pack(),
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &refined,
            },
        )
        .expect("the refined mixed plan compiles");
        let referenced = compiled.request("SH010").unwrap();
        assert_eq!(referenced.prompt_source, PromptSource::Refined);
        // The rewrite is carried through UNTOUCHED, behind the bindings.
        assert_eq!(
            referenced.prompt,
            "The courier is the person shown in <Picture 1>. The red parcel is the object shown in \
             <Picture 2>. integrated_multimodal_description: a courier steps into a warm workshop"
        );
        // The authored prompt is the PLAN's, with no compiler text in it: the insertion happened
        // after the rewrite, not before it, so neither recorded text has been polluted.
        assert_eq!(
            referenced.authored_prompt.as_deref(),
            Some("a courier enters")
        );
        assert!(
            !referenced
                .authored_prompt
                .as_deref()
                .unwrap()
                .contains("<Picture"),
            "the authored prompt must stay the plan's own words"
        );
        assert_eq!(referenced.inserted_text.len(), 1);
        assert!(
            !referenced.inserted_text[0]
                .text
                .contains("courier steps into"),
            "the inserted text is the compiler's alone: {}",
            referenced.inserted_text[0].text
        );
    }

    /// The sentence a role gets is built from its pack entry: the KIND chooses the noun, and the
    /// author's own DESCRIPTION is repeated verbatim rather than paraphrased.
    #[test]
    fn the_pack_kind_chooses_the_noun_and_the_description_is_repeated_verbatim() {
        let described = parse_reference_pack(
            &json!({
                "schemaVersion": 1,
                "id": "described",
                "version": 1,
                "references": [
                    { "role": "courier", "kind": "character", "file": "references/a.png",
                      "description": "Blue jacket, carries the parcel." },
                    { "role": "red_parcel", "kind": "prop", "file": "references/b.png",
                      "description": "Small bright red cardboard parcel" },
                    { "role": "workshop_location", "kind": "location", "file": "references/c.png",
                      "description": "Cluttered woodworking workshop, door camera-left." }
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["shots"][0]["conditioning"]["referenceRoles"] =
            json!(["courier", "red_parcel", "workshop_location"]);
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let compiled = compile_plan(
            &plan,
            &described,
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("compiles");
        assert_eq!(
            compiled.request("SH010").unwrap().inserted_text[0].text,
            "The courier is the person shown in <Picture 1>. Blue jacket, carries the parcel. \
             The red parcel is the object shown in <Picture 2>. Small bright red cardboard parcel. \
             The workshop location is the place shown in <Picture 3>. Cluttered woodworking \
             workshop, door camera-left."
        );
    }

    /// sc-24023. A description's WHITESPACE is normalized before it is appended: the binding
    /// sentence is one line of a dispatched prompt, and a pack description that wrapped over several
    /// lines would otherwise put a raw newline and tab run into the prompt — inside `insertedText`,
    /// the one field `conformance_findings` reads as the compiler's own authored text and therefore
    /// never checks. Only the joined whole was ever trimmed, never each description.
    #[test]
    fn a_multi_line_description_is_collapsed_to_one_line_before_it_reaches_the_prompt() {
        let wrapped = parse_reference_pack(
            &json!({
                "schemaVersion": 1,
                "id": "wrapped",
                "version": 1,
                "references": [
                    { "role": "courier", "kind": "character", "file": "references/a.png",
                      "description": "  Blue jacket,\r\n\tcarries   the parcel.  " },
                    { "role": "red_parcel", "kind": "prop", "file": "references/b.png",
                      "description": "Small\nred parcel" }
                ]
            })
            .to_string(),
        )
        .unwrap();
        let mut document: Value = serde_json::from_str(&mixed_plan_text()).unwrap();
        document["shots"][0]["conditioning"]["referenceRoles"] = json!(["courier", "red_parcel"]);
        let plan = parse_plan(&document.to_string()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let compiled = compile_plan(
            &plan,
            &wrapped,
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect("compiles");
        let request = compiled.request("SH010").unwrap();
        assert_eq!(
            request.inserted_text[0].text,
            "The courier is the person shown in <Picture 1>. Blue jacket, carries the parcel. \
             The red parcel is the object shown in <Picture 2>. Small red parcel."
        );
        // And the prompt the engine receives carries no control run either.
        assert!(
            !request.prompt.contains(['\n', '\r', '\t']),
            "{:?}",
            request.prompt
        );
    }

    /// The compiler's own sentences are a DERIVED field: a hand-edited `insertedText` — the one
    /// edit that could repoint a binding at the wrong picture without changing anything else the
    /// document declares — is refused by name, exactly like a swapped model or duration.
    #[test]
    fn a_hand_edited_inserted_text_is_refused_like_every_other_derived_field() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = mixed_entries(&base, &reference);
        let clean = mixed_compiled_with_short_edge(None);
        assert!(clean
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .is_empty());

        let mut tampered = clean.clone();
        tampered.requests[0].inserted_text[0].text =
            "The courier is the person shown in <Picture 2>.".to_owned();
        let findings = tampered.conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].field, "compiled.insertedText", "{findings:?}");
        assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));

        // Deleting the insertion entirely is the same refusal: an unbound reference prompt is not
        // what compiling this plan produces.
        let mut emptied = clean;
        emptied.requests[0].inserted_text.clear();
        let fields: Vec<String> = emptied
            .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
            .into_iter()
            .map(|finding| finding.field)
            .collect();
        assert_eq!(
            fields,
            vec!["compiled.insertedText".to_owned()],
            "{fields:?}"
        );
    }

    /// sc-24023, E6. A `compiled.json` written by the PREVIOUS schema version is refused BY
    /// VERSION, not blamed on the operator as a hand edit — the same rule the v2 and v3 bumps
    /// established. Such a document has no `insertedText` key and a prompt with no binding
    /// sentences, which conformance would otherwise report as tampering.
    #[test]
    fn a_previous_version_compiled_document_is_refused_by_schema_version_not_as_tampered() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let entries = mixed_entries(&base, &reference);

        // The BUMP itself. 3 is the version a `compiled.json` written before this story carries,
        // and its requests have no `insertedText` and no binding sentences; leaving the constant
        // at 3 would let such a document pass `staleness_findings` and then be blamed at
        // conformance for a hand edit it never made. The number is named, not derived, because it
        // is the specific document on disk that has to be refused — leave the constant at 3 and
        // this document stops producing a finding at all.
        let mut v3 = mixed_compiled_with_short_edge(None);
        v3.schema_version = 3;
        for request in &mut v3.requests {
            request.inserted_text.clear();
        }
        let findings = v3.staleness_findings(&plan, "abc");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].field, "compiled.schemaVersion", "{findings:?}");

        // Exactly what the previous version on disk deserializes to: the old number, and the key
        // absent.
        let mut previous = mixed_compiled_with_short_edge(None);
        previous.schema_version = COMPILED_PLAN_SCHEMA_VERSION - 1;
        for request in &mut previous.requests {
            request.inserted_text.clear();
        }
        let round_tripped: CompiledPlan =
            serde_json::from_value(serde_json::to_value(&previous).unwrap()).unwrap();
        assert!(round_tripped.requests[0].inserted_text.is_empty());

        let findings = round_tripped.staleness_findings(&plan, "abc");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].field, "compiled.schemaVersion", "{findings:?}");
        assert!(
            findings[0].message.contains(&format!(
                "schema version {}",
                COMPILED_PLAN_SCHEMA_VERSION - 1
            )) && findings[0].message.contains("film-harness compile"),
            "the refusal must name the version AND the remedy: {findings:?}"
        );

        // And this is the finding the bump replaced: at the CURRENT version the same missing
        // insertion is (correctly) a tampering report.
        let mut current = round_tripped;
        current.schema_version = COMPILED_PLAN_SCHEMA_VERSION;
        assert!(current.staleness_findings(&plan, "abc").is_empty());
        assert!(
            current
                .conformance_findings(&plan, &pack(), &entries, ModelLane::Mlx)
                .iter()
                .any(|finding| finding.field == "compiled.insertedText"),
            "without the bump a previous-version document lands here instead"
        );
    }

    /// A prompt that no longer fits once the bindings lead it is a FINDING naming the shot, not a
    /// silently truncated prompt — the same rule an unusable refinement already obeys.
    #[test]
    fn a_prompt_that_no_longer_fits_once_the_bindings_lead_it_is_refused() {
        let long = parse_reference_pack(
            &json!({
                "schemaVersion": 1,
                "id": "long",
                "version": 1,
                "references": [
                    { "role": "courier", "kind": "character", "file": "references/a.png",
                      "description": "x".repeat(MAX_PROMPT_CHARS) },
                    { "role": "red_parcel", "kind": "prop", "file": "references/b.png" }
                ]
            })
            .to_string(),
        )
        .unwrap();
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let base = entry();
        let reference = reference_entry();
        let findings = compile_plan(
            &plan,
            &long,
            &CompileInputs {
                entries: &mixed_entries(&base, &reference),
                lane: "mlx",
                plan_sha256: "abc",
                compiled_at: "now",
                refined_prompts: &BTreeMap::new(),
            },
        )
        .expect_err("refuses");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));
        assert!(
            findings[0].message.contains("binding sentences")
                && findings[0].message.contains("the video route accepts"),
            "{findings:?}"
        );
    }

    #[test]
    fn tier_maps_to_the_shared_mlx_quantize_convention() {
        assert_eq!(mlx_quantize_for_tier("q4"), json!(4));
        assert_eq!(mlx_quantize_for_tier("q8"), json!(8));
        assert_eq!(mlx_quantize_for_tier("bf16"), json!(0));
    }

    #[test]
    fn production_plan_hash_matches_the_persisted_pretty_document() {
        let plan = parse_plan(&mixed_plan_text()).unwrap();
        let mut persisted = serde_json::to_vec_pretty(&plan).unwrap();
        persisted.push(b'\n');
        let expected = format!("{:x}", Sha256::digest(&persisted));
        assert_eq!(production_plan_sha256(&plan).unwrap(), expected);
    }
}
