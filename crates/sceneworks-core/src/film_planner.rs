//! Turning a brief into a [`ProductionPlan`](crate::film_plan::ProductionPlan) with a LOCAL
//! planner (epic 22708, sc-22713).
//!
//! This module is the pure half of the planner: it owns the brief document, the capability
//! envelope the installed model actually declares, the request text handed to the local LLM, the
//! strict parse of what comes back, and the findings that decide whether a draft is a plan or a
//! repair round. It performs no I/O beyond reading the two documents it is given and knows nothing
//! about jobs, routes or transports — the driver (`rust-api` `film_planner`) supplies the model
//! replies, so every rule below is testable against a scripted reply with no API and no GPU.
//!
//! Three properties the rest of the harness relies on:
//!
//! * **The output is the SAME document the hand-authored path uses.** A generated plan is a
//!   [`ProductionPlan`] that `film-harness validate` accepts unchanged; there is no second schema
//!   and no planner-only field the driver has to interpret.
//! * **The brief's beats are a contract.** Every [`RequiredBeat`] must be covered by at least one
//!   shot. A draft that drops one is a finding, never a shorter plan — and the finding is fed back
//!   verbatim to the next repair round, so a bounded repair loop can fix it or fail loudly.
//! * **Nothing is coerced to fit.** Timing off the model's declared menu, a mode it does not
//!   declare, a reference role the pack does not contain: all findings. The planner never rounds a
//!   duration, never silently drops a reference and never invents a role.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::film_plan::{
    parse_resolution, validate_all, ModelEntries, ModelLane, PlanDiagnostic, PlanLimits,
    PlanLoraEntry, PlanModel, PlanSound, ProductionPlan, ReferencePack, Shot, ShotConditioning,
    ShotDependency, BINDABLE_REFERENCE_KINDS, PLAN_SCHEMA_VERSION, SHOT_CONDITIONING_MODES,
};
use crate::jsonc::strip_jsonc_comments;
use crate::minimax_h3_turbo::TurboRecipe;
use crate::video_request::{default_fps, default_resolution, reference_caps};

/// Schema version of [`ProductionBrief`] documents this module reads.
pub const BRIEF_SCHEMA_VERSION: u32 = 1;

/// Hard ceiling on the shots one planner draft may contain, whatever the brief asks for. A finite
/// bound on the planner's output is what keeps a runaway reply from becoming a runaway run (E5);
/// the brief's own `maxShots` may only narrow it.
pub const MAX_PLANNER_SHOTS: usize = 24;

/// The user's input: what the film is, how long it should run, and the beats it must contain. The
/// approved reference pack is the planner's other input and stays a separate document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProductionBrief {
    pub schema_version: u32,
    /// Id of the plan this brief produces (`[A-Za-z0-9_-]{1,64}`).
    pub id: String,
    pub version: u32,
    pub title: String,
    pub synopsis: String,
    /// Free-text direction: look, palette, camera language, sound. Handed to the planner verbatim.
    #[serde(default)]
    pub style_notes: String,
    /// Total running time the sequence should land in, in seconds.
    pub target_total_seconds: DurationWindow,
    /// Beats the film must contain, in narrative order. Every one must map to at least one shot.
    pub required_beats: Vec<RequiredBeat>,
    /// The model the plan renders through, exactly as a hand-authored plan declares it.
    pub model: PlanModel,
    /// The limits the generated plan declares.
    pub limits: PlanLimits,
    /// Ceiling on the number of shots, never above [`MAX_PLANNER_SHOTS`].
    #[serde(default = "default_max_shots")]
    pub max_shots: usize,
    /// Keep the model's FULL step count: do not offer the planner the installed step-distill
    /// accelerators, and strip any the draft names (sc-23406).
    ///
    /// The turbo recipes are the default because the alternative is a several-hour render for a
    /// minute of film, which is not a default anyone chooses on purpose. This is the brief-level
    /// opt-out for the case where the extra quality is worth the wall-clock — one boolean on the
    /// brief rather than a per-shot knob, because the schedule is a property of the render regime,
    /// not of a shot.
    #[serde(default)]
    pub prefer_quality: bool,
}

fn default_max_shots() -> usize {
    MAX_PLANNER_SHOTS
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DurationWindow {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RequiredBeat {
    /// Stable id (`[A-Za-z0-9_-]{1,64}`) the planner tags each shot with, so coverage is checked by
    /// identity rather than by matching prose.
    pub id: String,
    pub summary: String,
    /// Approved pack roles this beat is ABOUT: the subject, the prop the story turns on, whoever
    /// has to be on screen for the beat to have happened. The shots covering the beat must bind
    /// every one of them, so "the beat is covered" cannot be satisfied by a shot of the right
    /// length in the right place that leaves the parcel out of the delivery (sc-22713).
    ///
    /// Empty means "the brief makes no claim", which is the pre-sc-22713 behaviour and what a brief
    /// written before this field says.
    #[serde(default)]
    pub required_roles: Vec<String>,
}

/// Read and parse a brief document (JSONC tolerated).
pub fn read_brief_file(path: &Path) -> Result<ProductionBrief, PlanDiagnostic> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        PlanDiagnostic::plan("brief", format!("cannot read {}: {error}", path.display()))
    })?;
    parse_brief(&text)
        .map_err(|error| PlanDiagnostic::plan("brief", format!("{}: {error}", path.display())))
}

/// Parse a brief from JSON/JSONC text. Unknown fields are refused.
pub fn parse_brief(text: &str) -> Result<ProductionBrief, String> {
    serde_json::from_str(&strip_jsonc_comments(text)).map_err(|error| error.to_string())
}

/// Structural findings on the brief alone — checked before a single token is generated.
pub fn validate_brief(brief: &ProductionBrief) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if brief.schema_version != BRIEF_SCHEMA_VERSION {
        findings.push(PlanDiagnostic::plan(
            "brief.schemaVersion",
            format!(
                "unsupported brief schema version {} (this build reads {BRIEF_SCHEMA_VERSION})",
                brief.schema_version
            ),
        ));
    }
    if !crate::film_plan::is_safe_plan_id(&brief.id) {
        findings.push(PlanDiagnostic::plan(
            "brief.id",
            "brief id must be 1-64 characters of [A-Za-z0-9_-]",
        ));
    }
    if brief.version == 0 {
        findings.push(PlanDiagnostic::plan(
            "brief.version",
            "brief version must be >= 1",
        ));
    }
    if brief.title.trim().is_empty() {
        findings.push(PlanDiagnostic::plan("brief.title", "a title is required"));
    }
    if brief.synopsis.trim().is_empty() {
        findings.push(PlanDiagnostic::plan(
            "brief.synopsis",
            "a synopsis is required; it is what the planner writes the film from",
        ));
    }
    let window = brief.target_total_seconds;
    if !window.min.is_finite() || !window.max.is_finite() || window.min <= 0.0 {
        findings.push(PlanDiagnostic::plan(
            "brief.targetTotalSeconds",
            "the target running time must be finite seconds > 0",
        ));
    } else if window.max < window.min {
        findings.push(PlanDiagnostic::plan(
            "brief.targetTotalSeconds",
            format!(
                "max ({}) is below min ({}); the window is empty",
                window.max, window.min
            ),
        ));
    }
    if brief.required_beats.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "brief.requiredBeats",
            "a brief needs at least one required beat; beats are what the plan is checked against",
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, beat) in brief.required_beats.iter().enumerate() {
        if !crate::film_plan::is_safe_plan_id(&beat.id) {
            findings.push(PlanDiagnostic::plan(
                format!("brief.requiredBeats[{index}].id"),
                format!(
                    "beat id {:?} must be 1-64 characters of [A-Za-z0-9_-]",
                    beat.id
                ),
            ));
        } else if !seen.insert(beat.id.as_str()) {
            findings.push(PlanDiagnostic::plan(
                format!("brief.requiredBeats[{index}].id"),
                format!("duplicate beat id {:?}", beat.id),
            ));
        }
        if beat.summary.trim().is_empty() {
            findings.push(PlanDiagnostic::plan(
                format!("brief.requiredBeats[{index}].summary"),
                "a beat needs a summary",
            ));
        }
        let mut roles = BTreeSet::new();
        for role in &beat.required_roles {
            if !crate::film_plan::is_safe_plan_id(role) {
                findings.push(PlanDiagnostic::plan(
                    format!("brief.requiredBeats[{index}].requiredRoles"),
                    format!("role {role:?} must be 1-64 characters of [A-Za-z0-9_-]"),
                ));
            } else if !roles.insert(role.as_str()) {
                findings.push(PlanDiagnostic::plan(
                    format!("brief.requiredBeats[{index}].requiredRoles"),
                    format!("role {role:?} is listed twice"),
                ));
            }
        }
    }
    // The planner's own memory budget is a declared bound like every other limit (E5, sc-22715):
    // `1 + rounds` full local decodes plus one rewrite per shot run on the same host the plan will
    // later render on, and nothing checks that host before the first token unless the brief says
    // what the decodes are allowed.
    if brief.limits.planner_max_memory_gb.is_none() {
        findings.push(PlanDiagnostic::plan(
            "brief.limits.plannerMaxMemoryGb",
            "a brief must declare limits.plannerMaxMemoryGb — the memory the planner's LLM decodes \
             are allowed on the API host; it is checked against the host before the first token",
        ));
    }
    if brief.max_shots == 0 || brief.max_shots > MAX_PLANNER_SHOTS {
        findings.push(PlanDiagnostic::plan(
            "brief.maxShots",
            format!("maxShots must be between 1 and {MAX_PLANNER_SHOTS}"),
        ));
    } else if brief.max_shots < brief.required_beats.len() {
        findings.push(PlanDiagnostic::plan(
            "brief.maxShots",
            format!(
                "maxShots ({}) is below the {} required beats, so a covering plan cannot exist",
                brief.max_shots,
                brief.required_beats.len()
            ),
        ));
    }
    findings
}

// ---------------------------------------------------------------------------------------------
// Capability envelope
// ---------------------------------------------------------------------------------------------

/// What the INSTALLED model declares it can do, read off its catalog entry. The planner is told
/// only this — never a capability from a model card, an upstream repo or a hosted API — so a draft
/// that is inside the envelope is dispatchable and one that is not is refused with the same
/// diagnostics `film-harness validate` would produce.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlannerCapabilities {
    pub model_id: String,
    /// Conditioning modes the entry declares AND this schema can express.
    pub modes: Vec<String>,
    /// The declared clip-length menu, in seconds. Empty when the model declares no menu.
    pub durations: Vec<f64>,
    /// Frames per second the plan will render at.
    pub fps: Option<u32>,
    /// The declared resolution menu, `WxH`. Empty when the model declares no menu.
    pub resolutions: Vec<String>,
    /// The resolution every shot uses unless it overrides it (plan default, else model default).
    pub default_resolution: Option<String>,
    /// Model evaluations used when neither a turbo recipe nor an authored step override applies.
    pub default_steps: Option<u32>,
    /// `limits.maxReferenceAssets` — zero means the checkpoint has no reference conditioning.
    pub max_reference_images: usize,
    pub supports_negative_prompt: bool,
    /// `<lane>.minMemoryGb`, when declared.
    pub min_memory_gb: Option<f64>,
    /// The INSTALLED step-distill accelerators this plan may declare, one per partition the plan
    /// can dispatch as (sc-23406).
    ///
    /// Install state IS a filter here, unlike the reference partition's: a plan naming an adapter
    /// whose weights are not on this disk 400s at enqueue, so offering one would be an
    /// unreachable instruction. Empty means the base regime — nothing installed, or the brief
    /// opted out with `preferQuality`.
    pub turbo_loras: Vec<PlannerTurboLora>,
    /// Every LoRA id this host reports INSTALLED, as `GET /api/v1/loras` returned them — a superset
    /// of [`Self::turbo_loras`], which is the subset the envelope chose to offer.
    ///
    /// Kept because "installed" and "offered" answer different questions and only one of them is a
    /// rule for the planner: a BRIEF may declare an adapter the envelope did not offer (a second
    /// recipe for a partition, or a non-accelerator LoRA), and that is the author's call, not a
    /// draft's mistake. What it may not be is absent from this disk, since that 400s at enqueue —
    /// so [`lora_offer_findings`] judges a brief-declared id against this list and a
    /// draft-contributed one against the offers.
    pub installed_lora_ids: Vec<String>,
}

/// One installed accelerator the planner may declare, as the envelope states it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlannerTurboLora {
    /// The catalog id — the exact string the draft must write.
    pub id: String,
    pub name: String,
    /// The partition it attaches to, so the envelope can say one per checkpoint rather than
    /// offering a list the model has to pair up itself.
    pub model_id: String,
    /// Model evaluations the recipe renders at, against the model's own default.
    pub steps: u32,
}

/// Read the capability envelope for `plan_model` out of its catalog entry.
pub fn capabilities_for(
    plan_model: &PlanModel,
    entry: &Map<String, Value>,
    lane: ModelLane,
) -> PlannerCapabilities {
    let limits = entry.get("limits").and_then(Value::as_object);
    let modes = entry
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|mode| SHOT_CONDITIONING_MODES.contains(mode))
        .map(str::to_owned)
        .collect();
    let durations = limits
        .and_then(|limits| limits.get("durations"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_f64)
        .collect();
    let resolutions: Vec<String> = limits
        .and_then(|limits| limits.get("resolutions"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let default_resolution = plan_model
        .resolution
        .clone()
        .filter(|value| parse_resolution(value).is_some())
        .or_else(|| default_resolution(entry).map(|(width, height)| format!("{width}x{height}")));
    PlannerCapabilities {
        model_id: plan_model.id.clone(),
        modes,
        durations,
        fps: plan_model.fps.or_else(|| default_fps(entry)),
        resolutions,
        default_resolution,
        default_steps: entry
            .get("defaults")
            .and_then(Value::as_object)
            .and_then(|defaults| defaults.get("steps"))
            .and_then(Value::as_u64)
            .and_then(|steps| u32::try_from(steps).ok()),
        max_reference_images: reference_caps(entry).images,
        supports_negative_prompt: entry
            .get("video")
            .and_then(Value::as_object)
            .and_then(|video| video.get("supportsNegativePrompt"))
            .and_then(Value::as_bool)
            .unwrap_or(true),
        min_memory_gb: crate::film_plan::model_min_memory_gb(entry, lane),
        turbo_loras: Vec::new(),
        installed_lora_ids: Vec::new(),
    }
}

impl PlannerCapabilities {
    /// Widen this envelope to the family's REFERENCE partition (sc-23405).
    ///
    /// A split family — MiniMax-H3 is the shipped one — serves `reference_to_video` from a SECOND
    /// catalog entry with its own `capabilities` and its own `limits.maxReferenceAssets`, and the
    /// base entry declares neither. Built from the base entry alone the envelope can never produce
    /// a reference-binding shot, which is where sc-23402 deliberately left it; the caps the planner
    /// is now held to are the ones the shot will ACTUALLY dispatch against, read off the partition
    /// that will render it rather than off the entry that cannot.
    ///
    /// The modes UNION rather than replace: the base checkpoint's `text_to_video` and keyframe
    /// modes stay legal on a plan that also has reference shots, because the partition is resolved
    /// per shot and a plan may mix them (`plan.ref.jsonc`).
    pub fn with_reference_partition(mut self, entry: &Map<String, Value>) -> Self {
        for mode in entry
            .get("capabilities")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|mode| SHOT_CONDITIONING_MODES.contains(mode))
        {
            if !self.modes.iter().any(|declared| declared == mode) {
                self.modes.push(mode.to_owned());
            }
        }
        self.max_reference_images = self.max_reference_images.max(reference_caps(entry).images);
        self
    }

    /// Narrow this envelope to what THIS reference pack can actually fill (E1).
    ///
    /// References are OPTIONAL input: a user may supply a pack that approves none, and then the
    /// base path runs. A `reference_to_video` shot needs at least one approved reference role to
    /// bind, so on such a pack the mode and the cap come off the envelope entirely rather than
    /// being offered and then refused a decode later — the planner emits the modes it emitted
    /// before this story, and the plan resolves to the base checkpoint throughout.
    ///
    /// "Can fill" is counted over [`BINDABLE_REFERENCE_KINDS`] only. An approved `style` or `plate`
    /// is not something a `reference_to_video` shot may bind — a style is a look rather than a
    /// subject, and a plate is placed through the keyframe slots — so a pack approving only those
    /// two fills no reference shot and narrows exactly as an empty pack does.
    pub fn narrowed_to_pack(mut self, pack: &ReferencePack) -> Self {
        if pack
            .references
            .iter()
            .any(|entry| entry.approved && BINDABLE_REFERENCE_KINDS.contains(&entry.kind.as_str()))
        {
            return self;
        }
        self.modes.retain(|mode| mode != "reference_to_video");
        self.max_reference_images = 0;
        self
    }

    /// Offer the INSTALLED step-distill accelerators this plan may declare (sc-23406).
    ///
    /// `installed_lora_ids` is the host's answer — the ids `GET /api/v1/loras` reports installed —
    /// and the catalog's own `modelIds` allowlist decides which partition each one attaches to, so
    /// the envelope can never offer an adapter onto the checkpoint it was not distilled for. At
    /// most one recipe per partition is offered, because a render has one schedule and a list the
    /// planner has to choose from is a list it can choose two from.
    ///
    /// # Which one, when a partition has several
    ///
    /// By RULE, never by position. The route hands its ids back sorted by `(scope, family, name)`
    /// (`apps/rust-api/src/loras.rs`), which on the shipped catalog puts
    /// `minimax_h3_turbo_4step_768p` ahead of `minimax_h3_turbo_4step_v01` for no reason anyone
    /// chose — taking the first match would have made a display-name sort decide the film's video
    /// shift. Per partition, in order:
    ///
    /// 1. **Schedule parity.** The accelerator whose recipe is the one already chosen for the other
    ///    partition in use ([`TurboRecipe::recipe_eq`]). A mixed film dispatches both partitions,
    ///    and sampling its two halves on two different schedules is the thing
    ///    `plan.v2.turbo.jsonc`'s header exists to avoid.
    /// 2. **Training canvas.** Among recipes that declare a training short edge, choose the nearest
    ///    one to this plan's short edge. This keeps the 768p recipe on 768p plans and the 544p
    ///    recipe on the shipped 576x320 film canvas. An undeclared canvas remains generic.
    /// 3. **Fewest steps, then catalog order.** Break equal canvas distances by the fewest model
    ///    evaluations, then by catalog order — never by the route's display-name sort.
    ///
    /// The REFERENCE partition is resolved first when it is offered, because it is the constrained
    /// end: exactly one shipped adapter distils the reference path, so resolving it first gives the
    /// base partition a parity anchor to match, rather than the other way round.
    ///
    /// The reference partition is offered only when this envelope offers reference conditioning at
    /// all ([`Self::offers_references`]) — on a run with no reference pack no shot can resolve
    /// there, and an adapter for a partition the film never dispatches is a second id for the
    /// planner to copy and nothing more.
    pub fn with_installed_turbo_loras(mut self, installed_lora_ids: &[String]) -> Self {
        self.installed_lora_ids = installed_lora_ids.to_vec();
        // Catalog order, not the caller's: the tiebreak below is the catalog's and the route's
        // sort must not reach it. Membership is the only thing the host's list decides.
        let installed: Vec<String> = crate::film_plan::builtin_plan_loras()
            .iter()
            .map(|lora| lora.id.clone())
            .filter(|id| installed_lora_ids.contains(id))
            .collect();
        let partitions: Vec<String> = crate::film_plan::plan_partition_ids(&self.model_id)
            .into_iter()
            .filter(|id| {
                !crate::film_plan::is_reference_partition_id(id) || self.offers_references()
            })
            .collect();
        // Reference partition FIRST, base against it; emitted below in the declared partition
        // order so the contract's array does not reorder itself with the pack.
        let mut resolution_order = partitions.clone();
        resolution_order.sort_by_key(|id| !crate::film_plan::is_reference_partition_id(id));
        let plan_short_edge = self
            .default_resolution
            .as_deref()
            .and_then(parse_resolution)
            .map(|(width, height)| width.min(height));
        let mut chosen: Vec<(String, PlannerTurboLora)> = Vec::new();
        let mut anchor: Option<&'static TurboRecipe> = None;
        for partition in &resolution_order {
            let candidates: Vec<(&'static PlanLoraEntry, &'static TurboRecipe)> =
                crate::film_plan::plan_loras_for_partition(&installed, partition)
                    .into_iter()
                    .filter_map(|lora| {
                        crate::minimax_h3_turbo::turbo_recipe_for_lora_id(&lora.id)
                            .map(|recipe| (lora, recipe))
                    })
                    .collect();
            let recipes: Vec<&TurboRecipe> = candidates.iter().map(|(_, recipe)| *recipe).collect();
            let Some(index) = Self::choose_turbo_offer(&recipes, anchor, plan_short_edge) else {
                continue;
            };
            let (lora, recipe) = candidates[index];
            anchor = Some(recipe);
            chosen.push((
                partition.clone(),
                PlannerTurboLora {
                    id: lora.id.clone(),
                    name: lora.name.clone(),
                    model_id: partition.clone(),
                    steps: recipe.steps,
                },
            ));
        }
        self.turbo_loras = partitions
            .iter()
            .filter_map(|partition| {
                chosen
                    .iter()
                    .find(|(id, _)| id == partition)
                    .map(|(_, offer)| offer.clone())
            })
            .collect();
        self
    }

    /// Which of one partition's installed accelerators to offer: the index into `recipes`, which
    /// are in CATALOG order, or `None` when the partition has none.
    ///
    /// Pure over the recipes so the rule can be exercised without a catalog, and so the three tiers
    /// are one readable list rather than three passes through
    /// [`PlannerCapabilities::with_installed_turbo_loras`]. The tiers are documented there; this is
    /// their whole implementation.
    fn choose_turbo_offer(
        recipes: &[&TurboRecipe],
        anchor: Option<&TurboRecipe>,
        plan_short_edge: Option<u32>,
    ) -> Option<usize> {
        if let Some(anchor) = anchor {
            if let Some(index) = recipes.iter().position(|recipe| recipe.recipe_eq(anchor)) {
                return Some(index);
            }
        }
        if let Some(edge) = plan_short_edge {
            if let Some((index, _)) = recipes
                .iter()
                .enumerate()
                .filter_map(|(index, recipe)| {
                    recipe.training_short_edge.map(|training_edge| {
                        (index, (training_edge.abs_diff(edge), recipe.steps, index))
                    })
                })
                .min_by_key(|(_, key)| *key)
            {
                return Some(index);
            }
        }
        // `min_by_key` keeps the FIRST minimum, and `recipes` is in catalog order, so the tiebreak
        // is the catalog's.
        recipes
            .iter()
            .enumerate()
            .min_by_key(|(_, recipe)| recipe.steps)
            .map(|(index, _)| index)
    }

    /// Whether this envelope offers reference conditioning at all — the one question the contract,
    /// the mode guidance and the driver's install gate all branch on, so they cannot disagree.
    pub fn offers_references(&self) -> bool {
        self.max_reference_images > 0 && self.modes.iter().any(|mode| mode == "reference_to_video")
    }

    /// The envelope as the planner is shown it: one line per axis, values only, no prose the model
    /// could read as optional.
    pub fn as_prompt_section(&self) -> String {
        let mut lines = vec![format!("Model: {}", self.model_id)];
        lines.push(format!(
            "Allowed conditioning modes (use no others): {}",
            if self.modes.is_empty() {
                "none".to_owned()
            } else {
                self.modes.join(", ")
            }
        ));
        if self.durations.is_empty() {
            lines.push(
                "Allowed targetDurationSeconds: any positive value this model accepts.".to_owned(),
            );
        } else {
            lines.push(format!(
                "Allowed targetDurationSeconds (copy one of these EXACTLY, they are not \
                 roundable): {}",
                self.durations
                    .iter()
                    .map(|value| format!("{value}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if let Some(fps) = self.fps {
            lines.push(format!(
                "Frame rate: {fps} fps for every shot (fixed; do not write an fps field)."
            ));
        }
        match (&self.default_resolution, self.resolutions.is_empty()) {
            (Some(default), false) => lines.push(format!(
                "Resolution: every shot renders at {default}. Omit the resolution field; the only \
                 other legal values are {}.",
                self.resolutions.join(", ")
            )),
            (Some(default), true) => lines.push(format!(
                "Resolution: every shot renders at {default}. Omit the resolution field."
            )),
            (None, _) => lines.push("Resolution: omit the resolution field.".to_owned()),
        }
        lines.push(if !self.offers_references() {
            "Reference conditioning: THIS CHECKPOINT HAS NONE. referenceRoles must be empty on \
             every shot, and the reference_to_video mode is unavailable. Keyframe modes \
             (image_to_video, first_last_frame) take a first/last frame role instead; a shot never \
             mixes keyframes with references."
                .to_owned()
        } else {
            format!(
                "Reference conditioning: at most {} reference roles on a reference_to_video shot, \
                 and an approved reference pack is available — so USE IT. Every shot binds, in \
                 referenceRoles, the approved roles that are on screen in it: start from the list \
                 its beat says it MUST show (copy that list whole — a beat may name two props and \
                 no location, so never swap one of its roles for a different kind) and add any \
                 other approved role in frame. That is what makes the same character, the same \
                 object and the same place appear in every shot instead of a new one each time. \
                 ONLY character, prop and location roles are bound this way: a style role or a \
                 plate role is NEVER listed in referenceRoles — a style belongs in the prose and a \
                 plate goes in a keyframe slot. A shot never mixes keyframe roles with reference \
                 roles — they are different conditioning tasks.",
                self.max_reference_images
            )
        });
        if self.turbo_loras.is_empty() {
            lines.push(
                "Accelerators: none are installed for this model, so write no loras field. Every \
                 shot renders at the model's full step count."
                    .to_owned(),
            );
        } else {
            lines.push(format!(
                "Accelerators (step-distill LoRAs), INSTALLED and ON BY DEFAULT: {}. Declare every \
                 one of these ids in the top-level loras array — they are what make this film take \
                 minutes instead of hours, and each one is distilled for the checkpoint named \
                 beside it, so the ones that do not apply to a given shot are simply not used. \
                 Never invent an id, and never write more than these.",
                self.turbo_loras
                    .iter()
                    .map(|lora| format!(
                        "{} ({} steps on {})",
                        lora.id, lora.steps, lora.model_id
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !self.supports_negative_prompt {
            lines.push(
                "Negative prompts: this model has none. Never write a negativePrompt field; state \
                 everything positively in the prompt instead."
                    .to_owned(),
            );
        }
        // Which mode to REACH FOR, not merely which are legal. Without this the planner satisfies
        // the slot shape the cheapest way it can see — every shot first_last_frame with the same
        // plate at both ends, which asks each clip to finish exactly where it began (sc-22713).
        //
        // With an approved pack in hand the default INVERTS (sc-23405): reference conditioning is
        // the whole reason the pack exists, and a text_to_video shot in a reference film is a shot
        // that quietly reinvents whoever is in it. The system turn defers to this line rather than
        // naming a default of its own, so the two cannot contradict each other.
        if self.offers_references() {
            lines.push(
                "Choosing a mode: reference_to_video is the DEFAULT and the right answer for every \
                 shot that shows an approved character, prop or location — which is nearly every \
                 shot. Use text_to_video only for a shot that shows none of them. image_to_video \
                 and first_last_frame BEGIN (and end) on one specific approved plate and take no \
                 referenceRoles at all; reach for them only when a shot must start on an exact \
                 frame."
                    .to_owned(),
            );
        } else if self.modes.iter().any(|mode| mode == "first_last_frame") {
            lines.push(
                "Choosing a mode: text_to_video is the default and the right answer for most \
                 shots. Use image_to_video when a shot should BEGIN on a specific approved plate. \
                 Use first_last_frame only when you have TWO DIFFERENT approved roles for the \
                 first and the last frame — the same role in both slots is refused, because it \
                 asks the shot to end exactly where it started."
                    .to_owned(),
            );
        } else if self.modes.iter().any(|mode| mode == "image_to_video") {
            lines.push(
                "Choosing a mode: text_to_video is the default and the right answer for most \
                 shots. Use image_to_video when a shot should BEGIN on a specific approved plate."
                    .to_owned(),
            );
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------------------------
// The draft the planner returns
// ---------------------------------------------------------------------------------------------

/// One shot as the planner writes it: the plan's own shot fields plus the id of the brief beat it
/// covers. Unknown fields are refused, so an invented field is a finding rather than silent data
/// loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DraftShot {
    pub id: String,
    /// The [`RequiredBeat::id`] this shot covers.
    pub beat_id: String,
    pub beat: String,
    pub framing: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    pub target_duration_seconds: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    pub start_state: String,
    pub end_state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialogue: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sound: Option<String>,
    pub conditioning: ShotConditioning,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default)]
    pub continuity_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannerDraft {
    /// Catalog LoRA ids the plan renders with — the accelerators the envelope offered (sc-23406).
    ///
    /// Declared once for the whole film rather than per shot, because it is the render REGIME, not
    /// a shot property: [`crate::film_plan::plan_loras_for_partition`] then routes each id to the
    /// partitions it was distilled for. A draft that writes none renders at the model's full step
    /// count, which is the pre-story behaviour and what `brief.preferQuality` asks for.
    #[serde(default)]
    pub loras: Vec<String>,
    pub shots: Vec<DraftShot>,
}

/// Isolate the outermost `{ … }` span of a model reply, normalise the shapes a local planner
/// reliably gets slightly wrong, and parse the result strictly.
///
/// Strict where it matters: an unknown or misspelled field is still refused outright, because a
/// field the schema does not know is data the plan would silently lose. Lenient where the model's
/// mistake carries no ambiguity ([`normalize_draft`]).
///
/// A failure names the FIELD PATH (`shots[3].startState`) rather than a line and column. The
/// consumer of this message is a local 8B model being asked to correct itself: it never sees the
/// text it emitted as numbered lines, so "at line 10 column 20" is unusable and cost the sc-22713
/// smoke all three of its decodes.
pub fn parse_planner_output(text: &str) -> Result<PlannerDraft, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("the planner returned nothing".to_owned());
    }
    let start = trimmed
        .find('{')
        .ok_or_else(|| "the planner returned no JSON object".to_owned())?;
    let end = trimmed
        .rfind('}')
        .ok_or_else(|| "the planner's JSON object is not closed".to_owned())?;
    if end <= start {
        return Err("the planner's JSON object is not closed".to_owned());
    }
    let mut value: Value = serde_json::from_str(&trimmed[start..=end])
        .map_err(|error| format!("the JSON does not parse: {error}"))?;
    normalize_draft(&mut value);
    serde_path_to_error::deserialize(value).map_err(|error| {
        let path = error.path().to_string();
        let inner = error.inner().to_string();
        if path.is_empty() || path == "." {
            inner
        } else {
            format!("{path}: {inner}")
        }
    })
}

/// The unambiguous shape repairs applied to a draft before it is parsed.
///
/// Each one exists because the local planner made it on a real run and there is exactly one thing
/// it could have meant, so bouncing it would cost a full decode to be told what a `match` arm can
/// say for free. Nothing here changes WHAT the plan says — only how it is spelled:
///
/// * `startState`/`endState` written as a role→description object, or as a list of phrases,
///   become one string. An object's keys are sorted (see [`flatten_to_prose`] for why explicitly),
///   so the same draft always normalises to the same plan and the plan's sha256 is reproducible.
/// * an optional field written as the empty string — `dialogue: ""`, `chainFromShotId: ""`,
///   `firstFrameRole: ""` — is REMOVED, which is what the contract asks for ("omit an optional
///   field rather than writing null, \"\" or a placeholder") and what the model meant. Left in
///   place, `chainFromShotId: ""` is a chain to a shot named "" and `firstFrameRole: ""` a
///   reference role that is not in any pack — two findings for one placeholder.
/// * a keyframe slot on a mode that takes none — `firstFrameRole` on a `text_to_video` shot — is
///   REMOVED. The mode is the shot's actual statement about its conditioning; the slots are
///   subordinate to it, and one the mode cannot use conditions nothing. Every draft of the
///   sc-22713 smoke filled both slots on every `text_to_video` shot and all three rounds were
///   refused for it and nothing else — the model was copying the shape of the contract's own
///   template, which now shows the slot-free form.
///
/// Anything else stays wrong and is reported. In particular a NON-EMPTY `referenceRoles` on a
/// keyframe mode is never dropped: unlike a keyframe slot, which names one frame, that is a list of
/// subjects the shot claims to be conditioned on, and silently deleting it would change what the
/// plan asks for.
fn normalize_draft(value: &mut Value) {
    let Some(shots) = value
        .get_mut("shots")
        .and_then(|shots| shots.as_array_mut())
    else {
        return;
    };
    for shot in shots {
        let Some(shot) = shot.as_object_mut() else {
            continue;
        };
        for field in ["startState", "endState", "beat", "framing", "prompt"] {
            if let Some(entry) = shot.get_mut(field) {
                if let Some(flattened) = flatten_to_prose(entry) {
                    *entry = Value::String(flattened);
                }
            }
        }
        for field in [
            "dialogue",
            "sound",
            "negativePrompt",
            "resolution",
            "beatId",
        ] {
            drop_if_blank(shot, field);
        }
        if let Some(conditioning) = shot
            .get_mut("conditioning")
            .and_then(|value| value.as_object_mut())
        {
            for field in ["firstFrameRole", "lastFrameRole", "chainFromShotId"] {
                drop_if_blank(conditioning, field);
            }
            let mode = conditioning
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if mode != "image_to_video" && mode != "first_last_frame" {
                conditioning.remove("firstFrameRole");
            }
            if mode != "first_last_frame" {
                conditioning.remove("lastFrameRole");
            }
            // An EMPTY list on a mode that takes no references is the same empty statement the
            // schema's own default is; a non-empty one is left to be reported.
            if conditioning
                .get("referenceRoles")
                .and_then(Value::as_array)
                .is_some_and(|roles| roles.is_empty())
            {
                conditioning.remove("referenceRoles");
            }
        }
    }
}

/// A string-valued field the model filled with a blank placeholder is removed outright. `beatId` is
/// deliberately included even though it is required: "" is not a beat id, and `beatId` missing is a
/// finding that names the field, where `beatId: ""` is one that names a beat nobody declared.
fn drop_if_blank(object: &mut Map<String, Value>, field: &str) {
    let blank = match object.get(field) {
        Some(Value::String(text)) => text.trim().is_empty(),
        Some(Value::Null) => true,
        _ => false,
    };
    if blank {
        object.remove(field);
    }
}

/// One line of prose for a value the schema wants as a string but the model wrote as an object or a
/// list. `None` for a value that is already a string (or anything else, which stays a type error).
///
/// An object's keys are sorted HERE rather than taken in `Map` order. `serde_json`'s map is a
/// `BTreeMap` or an `IndexMap` depending on whether anything in the build graph turns on
/// `preserve_order` — which this workspace's graph does — so relying on its iteration order would
/// make the normalised text, and with it the plan's sha256, depend on a transitive feature flag.
/// Sorting here is the only way the same draft normalises to the same plan in every build.
fn flatten_to_prose(value: &Value) -> Option<String> {
    match value {
        Value::Object(fields) => Some({
            // Collected through a BTreeMap so the order is key-sorted by construction.
            let entries: BTreeMap<&str, &Value> = fields
                .iter()
                .map(|(key, field)| (key.as_str(), field))
                .collect();
            entries
                .into_iter()
                .filter_map(|(key, field)| {
                    let text = match field {
                        Value::String(text) => text.trim().to_owned(),
                        other => other.to_string(),
                    };
                    (!text.is_empty()).then(|| format!("{key}: {text}"))
                })
                .collect::<Vec<_>>()
                .join("; ")
        }),
        Value::Array(items) => Some(
            items
                .iter()
                .map(|item| match item {
                    Value::String(text) => text.trim().to_owned(),
                    other => other.to_string(),
                })
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("; "),
        ),
        _ => None,
    }
}

/// Assemble a [`ProductionPlan`] from the brief and a parsed draft. Everything outside the shots —
/// id, version, title, model, limits — comes from the brief, so the planner cannot widen a limit or
/// swap the model.
///
/// The planner writes sound INTENT only (each shot's `dialogue` and `sound` prose, below); it never
/// places sound. Resolving intent into a placed bed or a [`Shot::dialogue_clip`] needs a reference
/// pack to resolve roles against, and a brief does not carry one — so the generated plan takes
/// [`PlanSound::default`] (no beds, generated clip audio muted, the policy that cannot double a
/// line) and leaves every clip unset, which `validate_plan_structure` and
/// `validate_plan_against_pack` both accept. Placing sound is an edit to the generated plan.
pub fn draft_to_plan(brief: &ProductionBrief, draft: &PlannerDraft) -> ProductionPlan {
    ProductionPlan {
        schema_version: PLAN_SCHEMA_VERSION,
        id: brief.id.clone(),
        version: brief.version,
        title: brief.title.clone(),
        synopsis: brief.synopsis.clone(),
        // The model block is the BRIEF's, so the planner can neither widen a limit nor swap the
        // checkpoint. `loras` is the one axis it may state, and only when the brief left it open:
        // a brief that declared its own selection keeps it, and `preferQuality` clears the field
        // outright rather than trusting the draft to have honoured an instruction (sc-23406).
        model: {
            let mut model = brief.model.clone();
            if brief.prefer_quality {
                model.loras.clear();
            } else if model.loras.is_empty() {
                model.loras = draft.loras.clone();
            }
            model
        },
        limits: brief.limits.clone(),
        sound: PlanSound::default(),
        shots: draft
            .shots
            .iter()
            .map(|shot| Shot {
                id: shot.id.clone(),
                beat_id: Some(shot.beat_id.clone()),
                beat: shot.beat.clone(),
                framing: shot.framing.clone(),
                prompt: shot.prompt.clone(),
                negative_prompt: shot.negative_prompt.clone(),
                target_duration_seconds: shot.target_duration_seconds,
                resolution: shot.resolution.clone(),
                start_state: shot.start_state.clone(),
                end_state: shot.end_state.clone(),
                dialogue: shot.dialogue.clone(),
                sound: shot.sound.clone(),
                generated_audio: None,
                dialogue_clip: None,
                conditioning: shot.conditioning.clone(),
                seed: shot.seed,
                continuity_roles: shot.continuity_roles.clone(),
                // A declared chain IS a continuity dependency in sc-22711's vocabulary — "this
                // shot's start state is that shot's end state" — so a generated plan carries the
                // edge rather than leaving a planner-written film with nothing to flag when a take
                // it continues is replaced. The planner is not asked for edges: this is the one it
                // already stated, restated where `replace-take` reads it.
                depends_on: shot
                    .conditioning
                    .chain_from_shot_id
                    .iter()
                    .map(|shot_id| ShotDependency {
                        shot_id: shot_id.clone(),
                        kind: "continuity".to_owned(),
                        note: "declared by conditioning.chainFromShotId".to_owned(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Beat coverage: every required beat maps to at least one shot, every shot names a beat the brief
/// declares, and the shots that cover the beats run in the brief's narrative order.
///
/// This is the check that makes "never silently drops a required narrative beat" mechanical: a
/// dropped beat is a finding with the beat's id and summary in it, which the repair round hands
/// straight back to the planner.
pub fn coverage_findings(brief: &ProductionBrief, draft: &PlannerDraft) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let declared: BTreeMap<&str, &RequiredBeat> = brief
        .required_beats
        .iter()
        .map(|beat| (beat.id.as_str(), beat))
        .collect();
    let mut covered: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, shot) in draft.shots.iter().enumerate() {
        match declared.get(shot.beat_id.as_str()) {
            Some(_) => covered.entry(&shot.beat_id).or_default().push(index),
            None => findings.push(PlanDiagnostic::shot(
                &shot.id,
                "beatId",
                format!(
                    "{:?} is not a beat in this brief (beats: {})",
                    shot.beat_id,
                    declared.keys().copied().collect::<Vec<_>>().join(", ")
                ),
            )),
        }
    }
    for beat in &brief.required_beats {
        if !covered.contains_key(beat.id.as_str()) {
            findings.push(PlanDiagnostic::plan(
                "shots",
                format!(
                    "required beat {:?} ({}) is covered by no shot; add a shot for it — a beat is \
                     never dropped to make the plan fit",
                    beat.id,
                    beat.summary.trim()
                ),
            ));
        }
    }
    // The beats the brief declares are in narrative order; the shots that cover them must be too,
    // or the film tells the story out of sequence.
    let mut last_position: Option<(usize, &str)> = None;
    for shot in &draft.shots {
        let Some(position) = brief
            .required_beats
            .iter()
            .position(|beat| beat.id == shot.beat_id)
        else {
            continue;
        };
        if let Some((previous, previous_id)) = last_position {
            if position < previous {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "beatId",
                    format!(
                        "beat {:?} comes before {previous_id:?} in the brief but its shot comes \
                         after; keep the shots in the brief's narrative order",
                        shot.beat_id
                    ),
                ));
            }
        }
        last_position = Some((position, &shot.beat_id));
    }
    findings
}

/// Beat coverage of a PLAN (rather than a draft), checked through the `beatId` each generated shot
/// carries. This is what a hand edit is re-checked against: a plan with no beat ids at all was
/// hand-authored and has no brief to answer to, but once a shot claims a beat, every beat the brief
/// declares must still be claimed by one.
pub fn plan_coverage_findings(
    brief: &ProductionBrief,
    plan: &ProductionPlan,
    // The pack the roles are read against, so the repair hint cannot name a role the pack-aware
    // validator would then refuse (sc-23406).
    pack: &ReferencePack,
) -> Vec<PlanDiagnostic> {
    let claimed: BTreeSet<&str> = plan
        .shots
        .iter()
        .filter_map(|shot| shot.beat_id.as_deref())
        .collect();
    if claimed.is_empty() {
        return Vec::new();
    }
    let declared: BTreeSet<&str> = brief
        .required_beats
        .iter()
        .map(|beat| beat.id.as_str())
        .collect();
    let mut findings: Vec<PlanDiagnostic> = brief
        .required_beats
        .iter()
        .filter(|beat| !claimed.contains(beat.id.as_str()))
        .map(|beat| {
            PlanDiagnostic::plan(
                "shots",
                format!(
                    "required beat {:?} ({}) is covered by no shot in this plan; restore a shot \
                     for it, or update the brief if the beat is genuinely gone",
                    beat.id,
                    beat.summary.trim()
                ),
            )
        })
        .collect();
    for shot in &plan.shots {
        if let Some(beat_id) = shot.beat_id.as_deref() {
            if !declared.contains(beat_id) {
                findings.push(PlanDiagnostic::shot(
                    &shot.id,
                    "beatId",
                    format!(
                        "{beat_id:?} is not a beat in this brief (beats: {})",
                        declared.iter().copied().collect::<Vec<_>>().join(", ")
                    ),
                ));
            }
        }
    }
    // A hand edit can also hollow a beat out without deleting it — the shot stays, the subject
    // stops being named — so the roles are re-checked on the same pass as the beats.
    findings.extend(role_coverage_findings(brief, plan, pack));
    findings
}

/// Every approved pack role one shot binds: the roles it declares it depicts, plus whatever its
/// conditioning slots name. A role bound in either place is on screen by the plan's own account.
fn bound_roles(shot: &Shot) -> BTreeSet<&str> {
    let mut roles: BTreeSet<&str> = shot
        .continuity_roles
        .iter()
        .map(String::as_str)
        .chain(shot.conditioning.reference_roles.iter().map(String::as_str))
        .collect();
    roles.extend(shot.conditioning.first_frame_role.as_deref());
    roles.extend(shot.conditioning.last_frame_role.as_deref());
    roles
}

/// Beat ROLE coverage: for every beat that declares `requiredRoles`, the shots covering that beat
/// must between them bind each one.
///
/// Beat coverage alone only asks that a beat has a shot. It cannot tell a delivery from a shot of
/// an empty workbench, which is exactly what the first real-LLM draft produced: the parcel bound on
/// two of six shots, the courier absent from their own departure (sc-22713). The brief is where
/// "this beat is ABOUT the parcel" belongs, because only the brief knows the story.
///
/// The repair hint is built against the PACK, not against whatever the draft happened to name: a
/// role the draft hallucinated, or an approved `style`/`plate` role that is legal in
/// `continuityRoles` but refused as a `reference_to_video` subject, would otherwise be handed back
/// as "write [...]" while [`crate::film_plan::validate_plan_against_pack`] refuses the very array
/// it named — a copy-only repairer can never converge on that (sc-23406).
pub fn role_coverage_findings(
    brief: &ProductionBrief,
    plan: &ProductionPlan,
    pack: &ReferencePack,
) -> Vec<PlanDiagnostic> {
    let approved: BTreeSet<&str> = pack
        .references
        .iter()
        .filter(|entry| entry.approved)
        .map(|entry| entry.role.as_str())
        .collect();
    // A role the pack approves AND whose kind may be BOUND as a reference_to_video subject.
    let bindable = |role: &str| {
        pack.references.iter().any(|entry| {
            entry.approved
                && entry.role == role
                && BINDABLE_REFERENCE_KINDS.contains(&entry.kind.as_str())
        })
    };
    let mut findings = Vec::new();
    for beat in &brief.required_beats {
        if beat.required_roles.is_empty() {
            continue;
        }
        let covering: Vec<&Shot> = plan
            .shots
            .iter()
            .filter(|shot| shot.beat_id.as_deref() == Some(beat.id.as_str()))
            .collect();
        if covering.is_empty() {
            // "No shot at all" is `coverage_findings`' report; do not say it twice.
            continue;
        }
        let bound: BTreeSet<&str> = covering.iter().flat_map(|shot| bound_roles(shot)).collect();
        let missing: Vec<&str> = beat
            .required_roles
            .iter()
            .map(String::as_str)
            .filter(|role| !bound.contains(role))
            .collect();
        if !missing.is_empty() {
            // The repair is handed over as the exact array to write, not as a description of it:
            // the beat's list first, then whatever the named shot already binds, so copying it
            // adds the missing roles and drops nothing. The real planner reproduced every literal
            // array it was shown and re-derived every list it was described (sc-23406).
            let shot = covering[0];
            let mut corrected: Vec<String> = beat.required_roles.clone();
            for role in bound_roles(shot) {
                // Only what the pack approves: a role the draft invented is not a binding the
                // validator would accept, so it is not a binding the hint may ask for.
                if approved.contains(role) && !corrected.iter().any(|known| known == role) {
                    corrected.push(role.to_owned());
                }
            }
            // One array is written to both lists, so `referenceRoles` may be named only when every
            // role in it is bindable there. A beat that itself requires a style or a plate is an
            // unsatisfiable binding, never a droppable one: that case is told about
            // `continuityRoles` alone, which is where the coverage is read back.
            let names_reference_roles = shot.conditioning.mode == "reference_to_video"
                && beat
                    .required_roles
                    .iter()
                    .all(|role| bindable(role.as_str()));
            if names_reference_roles {
                corrected.retain(|role| bindable(role.as_str()));
            }
            let bound_now = bound_roles(shot);
            findings.push(PlanDiagnostic::shot(
                &shot.id,
                "continuityRoles",
                format!(
                    "beat {:?} must show {} but {} binds {}, missing {}; write {} into {}'s \
                     continuityRoles{} and describe {} in its prompt",
                    beat.id,
                    beat.required_roles.join(", "),
                    shot.id,
                    if bound_now.is_empty() {
                        "none of them".to_owned()
                    } else {
                        format!(
                            "only {}",
                            bound_now.iter().copied().collect::<Vec<_>>().join(", ")
                        )
                    },
                    missing.join(", "),
                    roles_json_array(&corrected),
                    shot.id,
                    if names_reference_roles {
                        " AND its referenceRoles"
                    } else {
                        ""
                    },
                    missing.join(", ")
                ),
            ));
        }
    }
    findings
}

/// Findings about the brief read against the reference pack it will be planned with: a beat cannot
/// require a role the pack does not approve, because no draft could ever satisfy it. Checked before
/// the first decode, so an unsatisfiable brief is a refusal rather than `1 + rounds` decodes that
/// were never going to converge.
pub fn brief_pack_findings(brief: &ProductionBrief, pack: &ReferencePack) -> Vec<PlanDiagnostic> {
    let approved: BTreeSet<&str> = pack
        .references
        .iter()
        .filter(|entry| entry.approved)
        .map(|entry| entry.role.as_str())
        .collect();
    let mut findings = Vec::new();
    for (index, beat) in brief.required_beats.iter().enumerate() {
        for role in &beat.required_roles {
            if !approved.contains(role.as_str()) {
                findings.push(PlanDiagnostic::plan(
                    format!("brief.requiredBeats[{index}].requiredRoles"),
                    format!(
                        "beat {:?} requires role {role:?}, which reference pack {:?} does not \
                         approve (approved: {})",
                        beat.id,
                        pack.id,
                        approved.iter().copied().collect::<Vec<_>>().join(", ")
                    ),
                ));
            }
        }
    }
    findings
}

/// Findings about the shape of the draft itself: how many shots it has and how long they add up to.
pub fn shape_findings(brief: &ProductionBrief, plan: &ProductionPlan) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let ceiling = brief.max_shots.min(MAX_PLANNER_SHOTS);
    if plan.shots.len() > ceiling {
        findings.push(PlanDiagnostic::plan(
            "shots",
            format!(
                "{} shots exceed the {ceiling} this brief allows; cover the beats in fewer shots \
                 rather than dropping one",
                plan.shots.len()
            ),
        ));
    }
    let total: f64 = plan
        .shots
        .iter()
        .map(|shot| shot.target_duration_seconds)
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .sum();
    let window = brief.target_total_seconds;
    let window_usable =
        window.min.is_finite() && window.max.is_finite() && window.max >= window.min && total > 0.0;
    if window_usable && (total < window.min || total > window.max) {
        findings.push(PlanDiagnostic::plan(
            "shots",
            format!(
                "the shots total {total:.4}s, outside the brief's {}-{}s window; change how many \
                 shots cover a beat or which legal clip length each one uses — the lengths \
                 themselves are not adjustable",
                window.min, window.max
            ),
        ));
    }
    findings
}

/// Findings on the LoRAs a draft declared, judged against what the envelope actually OFFERED
/// (sc-23406).
///
/// [`crate::film_plan::validate_plan_structure`] already refuses an id no shipped catalog carries.
/// This is the host-shaped half it cannot see: an id that IS in the catalog but is not installed
/// here, or one the envelope did not offer for any partition this plan uses. Both would 400 at
/// enqueue after the whole film had been planned, so they are repair-round findings naming the id —
/// the repair round hands the planner the offered ids back, which is what turns a refusal into a
/// repair.
///
/// # Only what the DRAFT contributed is judged against the offers
///
/// [`draft_to_plan`] keeps a brief's own `model.loras` and takes the draft's only when the brief
/// left the field open, so `plan.model.loras` can hold ids the planner never wrote. The offer
/// policy is a rule FOR THE PLANNER — at most one recipe per partition, chosen by the envelope —
/// and holding an author's own selection to it produced a finding no repair round could clear: the
/// draft cannot withdraw an id it never wrote, so every round re-emitted the same finding until the
/// run gave up. A brief-declared id is still judged on everything that is not the offer policy: it
/// must be INSTALLED here (below), and the document rules — family, `modelIds`, one recipe per
/// partition — are [`crate::film_plan::validate_plan_structure`]'s, which judges the whole list.
pub fn lora_offer_findings(
    caps: &PlannerCapabilities,
    brief: &ProductionBrief,
    plan: &ProductionPlan,
) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    let brief_has_custom_steps = brief
        .model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.steps)
        .is_some();
    if !brief.prefer_quality
        && brief.model.loras.is_empty()
        && !brief_has_custom_steps
        && plan.model.loras.is_empty()
        && !caps.turbo_loras.is_empty()
    {
        findings.push(PlanDiagnostic::plan(
            "model.loras",
            format!(
                "the planner omitted the default installed Turbo recipe; write exactly {}",
                caps.turbo_loras
                    .iter()
                    .map(|offer| offer.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    for id in &plan.model.loras {
        if caps.turbo_loras.iter().any(|offer| offer.id == *id) {
            continue;
        }
        if brief.model.loras.contains(id) {
            if !caps.installed_lora_ids.contains(id) {
                findings.push(PlanDiagnostic::plan(
                    "model.loras",
                    format!(
                        "{id:?} is declared by the brief but is not installed on this host, so \
                         every shot would be refused at enqueue; install it, or drop it from the \
                         brief's model.loras"
                    ),
                ));
            }
            continue;
        }
        findings.push(PlanDiagnostic::plan(
            "model.loras",
            format!(
                "{id:?} is not an accelerator this host offers; write {}",
                if caps.turbo_loras.is_empty() {
                    "no loras field at all".to_owned()
                } else {
                    format!(
                        "only {}",
                        caps.turbo_loras
                            .iter()
                            .map(|offer| offer.id.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            ),
        ));
    }
    findings
}

/// Every finding a generated plan is judged on: the document-level checks the hand-authored path
/// runs ([`validate_all`]), plus the planner-only ones — beat coverage, shot count and total
/// running time. The order is deliberate: coverage first, because a dropped beat is the failure
/// that must never be repaired away by shortening the film.
pub fn validate_generated_plan(
    brief: &ProductionBrief,
    draft: &PlannerDraft,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    pack_dir: Option<&Path>,
    model_entry: Option<(&ModelEntries<'_>, ModelLane)>,
    // The envelope the draft was produced against, when one is in hand — the only thing that
    // knows which accelerators this host offered (sc-23406).
    caps: Option<&PlannerCapabilities>,
) -> Vec<PlanDiagnostic> {
    let mut findings = coverage_findings(brief, draft);
    findings.extend(role_coverage_findings(brief, plan, pack));
    findings.extend(shape_findings(brief, plan));
    if let Some(caps) = caps {
        findings.extend(lora_offer_findings(caps, brief, plan));
    }
    findings.extend(validate_all(plan, pack, pack_dir, model_entry));
    findings
}

// ---------------------------------------------------------------------------------------------
// The request the local planner is given
// ---------------------------------------------------------------------------------------------

/// The user turn for the first planning round: the brief, the approved reference roles, the
/// capability envelope, and the exact JSON contract. Deterministic for a given input, so a test can
/// assert what the planner is told rather than what it happened to answer.
/// Approved role ids as the JSON array a draft writes: `["courier", "red_parcel"]`.
///
/// The planner is handed its role lists in this form wherever one is a requirement, because a
/// literal array is the one thing the local 8B planner copies faithfully — it reproduced the
/// `loras` array byte for byte in every real run — while a list it had to DERIVE from prose came
/// back as one character, one prop and one place, whatever the beat actually named (sc-23406).
pub fn roles_json_array(roles: &[String]) -> String {
    format!(
        "[{}]",
        roles
            .iter()
            .map(|role| format!("{role:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The beat's `requiredRoles` as the instruction the planner is given, on the beat line of the
/// first round and of every repair round: the exact array to write, and where.
///
/// `referenceRoles` is named only when the envelope offers reference conditioning; on a keyframe
/// or text-only envelope the same list goes into `continuityRoles` alone, which is where
/// [`role_coverage_findings`] reads it back.
fn required_roles_instruction(beat: &RequiredBeat, caps: &PlannerCapabilities) -> String {
    format!(
        "write {} into the covering shot's continuityRoles{}, copied whole (add other approved \
         roles in frame, never drop one of these)",
        roles_json_array(&beat.required_roles),
        if caps.offers_references() {
            " AND its referenceRoles"
        } else {
            ""
        }
    )
}

pub fn build_planner_request(
    brief: &ProductionBrief,
    pack: &ReferencePack,
    caps: &PlannerCapabilities,
) -> String {
    let mut out = String::new();
    out.push_str("# Brief\n\n");
    out.push_str(&format!("Title: {}\n", brief.title.trim()));
    out.push_str(&format!("Synopsis: {}\n", brief.synopsis.trim()));
    if !brief.style_notes.trim().is_empty() {
        out.push_str(&format!("Style: {}\n", brief.style_notes.trim()));
    }
    out.push_str(&format!(
        "Total running time: between {} and {} seconds across at most {} shots.\n",
        brief.target_total_seconds.min,
        brief.target_total_seconds.max,
        brief.max_shots.min(MAX_PLANNER_SHOTS)
    ));

    out.push_str("\n# Required beats (in order)\n\n");
    out.push_str(
        "Every beat below must be covered by at least one shot, and the shots must run in this \
         order. Tag each shot with the beat id it covers. Never drop a beat to make the running \
         time fit.\n\n",
    );
    for beat in &brief.required_beats {
        out.push_str(&format!("- {}: {}", beat.id, beat.summary.trim()));
        if !beat.required_roles.is_empty() {
            // Named as a requirement, not as colour: this is checked mechanically after the reply.
            out.push_str(&format!(
                " [this beat MUST show these roles — {} and describe each of them in its \
                 prompt.]",
                required_roles_instruction(beat, caps)
            ));
        }
        out.push('\n');
    }

    out.push_str("\n# Approved reference roles\n\n");
    let has_approved_roles = pack.references.iter().any(|entry| entry.approved);
    if has_approved_roles {
        out.push_str(
            "These are the only reference roles that exist. Name them exactly; never invent one. \
             Every shot lists in continuityRoles the approved roles it depicts, so the sequence \
             stays anchored to these approved references rather than to whatever the previous \
             shot happened to end on.\n\n",
        );
    } else {
        out.push_str(
            "This reference pack approves NO roles. Write continuityRoles: [] on EVERY shot. \
             Characters, props and locations named only in the brief or prompt are prose, not \
             role ids; never invent role ids for them.\n\n",
        );
    }
    if caps.offers_references() {
        out.push_str(
            "This model can also CONDITION on these references directly. Bind the ones a shot \
             shows in that shot's referenceRoles as well — see the conditioning rules below.\n\n",
        );
    }
    for entry in &pack.references {
        if !entry.approved {
            continue;
        }
        out.push_str(&format!(
            "- {} ({}): {}",
            entry.role,
            entry.kind,
            entry.description.trim()
        ));
        // A role sharing its image with another is named as sharing it (sc-24024). Binding both
        // costs the shot ONE reference image rather than two, and the locator is how the planner
        // knows the two roles are different subjects in one photograph rather than two pictures.
        let sharing: Vec<&str> = pack
            .references
            .iter()
            .filter(|other| other.approved && other.file == entry.file && other.role != entry.role)
            .map(|other| other.role.as_str())
            .collect();
        if let Some(locator) = entry.locator() {
            out.push_str(&format!(" It is {locator} in its image."));
        }
        if !sharing.is_empty() {
            out.push_str(&format!(
                " That image also shows {} — binding them together costs one reference image, \
                 not {}.",
                sharing.join(", "),
                sharing.len() + 1
            ));
        }
        out.push('\n');
    }

    out.push_str("\n# What this model can actually do\n\n");
    out.push_str(&caps.as_prompt_section());

    out.push_str("\n\n# Output contract\n\n");
    out.push_str(&plan_json_contract(caps, pack));
    out
}

/// The user turn for a repair round: what to change, the beats that must survive the change, the
/// draft being corrected, and the contract. Nothing is summarised — the planner is corrected with
/// the same text a human running `film-harness validate` would read.
///
/// The ORDER is load-bearing. The first version of this prompt led with the findings and then
/// handed over the whole rejected draft; on the real 8B planner all three rounds came back byte
/// for byte identical (sc-22713), because the last thing the model read was a complete, fluent
/// answer to the question and copying it is the likeliest continuation. So the draft is framed as
/// raw material rather than as an answer, and the findings are repeated after it as the last thing
/// read before the contract.
pub fn build_repair_request(
    brief: &ProductionBrief,
    pack: &ReferencePack,
    caps: &PlannerCapabilities,
    previous_output: &str,
    findings: &[PlanDiagnostic],
    round: u32,
    max_rounds: u32,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# Repair round {round} of {max_rounds}\n\nYour previous answer was REJECTED. Below are \
         the changes it needs, the beats it must still cover, and the text to correct. Return the \
         WHOLE plan as one JSON object with those changes made. Returning the same answer again \
         fails the round: every finding names a field, and every field it names must come back \
         different.\n\n"
    ));
    out.push_str("# Changes to make\n\n");
    for finding in findings {
        out.push_str(&format!("- {finding}\n"));
    }
    out.push_str("\n# Beats the corrected plan must still cover (in order)\n\n");
    out.push_str(
        "Fixing a finding never removes a beat or drops a shot's subject. When a duration finding \
         says the total is outside the brief's window, change legal shot durations or the number \
         of shots while retaining every beat.\n\n",
    );
    for beat in &brief.required_beats {
        out.push_str(&format!("- {}: {}", beat.id, beat.summary.trim()));
        if !beat.required_roles.is_empty() {
            out.push_str(&format!(
                " [MUST show — {}]",
                required_roles_instruction(beat, caps)
            ));
        }
        out.push('\n');
    }
    out.push_str(
        "\n# The text to correct (this is a DRAFT with the faults listed above, not an answer)\n\n",
    );
    out.push_str(previous_output.trim());
    out.push_str("\n\n# Before you answer, re-read these and fix each one\n\n");
    for finding in findings {
        out.push_str(&format!("- {finding}\n"));
    }
    out.push_str("\n# Output contract\n\n");
    out.push_str(&plan_json_contract(caps, pack));
    out.push_str("\n\n# Final repair checklist — apply every item now\n\n");
    if !pack.references.iter().any(|entry| entry.approved) {
        out.push_str(
            "- This pack approves no roles: write continuityRoles: [] on EVERY shot and do not \
             invent role ids from prose nouns.\n",
        );
    }
    for finding in findings {
        out.push_str(&format!("- {finding}\n"));
    }
    out.push_str(
        "Return the whole corrected JSON object. Do not return the draft above unchanged.\n",
    );
    out
}

/// The placeholder [`PLAN_JSON_CONTRACT`]'s worked example carries where its `targetDurationSeconds`
/// goes. It is filled from the ENVELOPE, never hard-coded: the example used to read `5.1667` —
/// MiniMax-H3's shortest clip — which on any other model is a value the same contract's own rule
/// ("copy one of the allowed durations EXACTLY") forbids, and a planner shown a forbidden value in
/// the one filled example it gets was being taught the wrong thing (AT4, sc-22715).
pub const EXAMPLE_DURATION_PLACEHOLDER: &str = "{{EXAMPLE_DURATION}}";

/// Where the worked example's `conditioning` object goes. Filled from the envelope for the same
/// reason the duration is (sc-23405): the one filled shot a planner is shown is the strongest
/// instruction in the contract, so on a pack-and-partition that make `reference_to_video` the
/// default it must not show a `text_to_video` shot — that teaches the opposite of what the
/// envelope's own mode guidance just said.
pub const EXAMPLE_CONDITIONING_PLACEHOLDER: &str = "{{EXAMPLE_CONDITIONING}}";

/// Where the output shape, standing rule and filled example describe continuity roles. These are
/// filled from the approved pack rather than from the model envelope: a script-only film with no
/// approved references must be shown empty arrays, not example role ids it cannot legally copy.
pub const CONTINUITY_FIELD_PLACEHOLDER: &str = "{{CONTINUITY_FIELD}}";
/// See [`CONTINUITY_FIELD_PLACEHOLDER`].
pub const CONTINUITY_RULE_PLACEHOLDER: &str = "{{CONTINUITY_RULE}}";
/// See [`CONTINUITY_FIELD_PLACEHOLDER`].
pub const EXAMPLE_CONTINUITY_PLACEHOLDER: &str = "{{EXAMPLE_CONTINUITY}}";

/// Where the per-envelope reference RULE goes: the standing instruction to bind approved roles on
/// every shot, or nothing at all when no reference conditioning is on offer.
pub const REFERENCE_RULE_PLACEHOLDER: &str = "{{REFERENCE_RULE}}";

/// Where the top-level `loras` line of the answer's shape goes, and the rule that governs it. Both
/// are filled from the envelope (sc-23406): a host with nothing installed must not be shown a field
/// it would then be refused for writing.
pub const LORA_FIELD_PLACEHOLDER: &str = "{{LORA_FIELD}}";
/// See [`LORA_FIELD_PLACEHOLDER`].
pub const LORA_RULE_PLACEHOLDER: &str = "{{LORA_RULE}}";

/// The rule inserted at [`REFERENCE_RULE_PLACEHOLDER`] when the envelope offers references.
const REFERENCE_BINDING_RULE: &str = "\n- An approved reference pack is available, so EVERY shot \
that shows an approved character, prop or location uses \"mode\": \"reference_to_video\" and lists \
those roles in referenceRoles — the beat's MUST-show list first, copied whole, then any other \
approved role in frame. A shot that binds nothing renders a stranger in a room nobody approved. \
referenceRoles and continuityRoles are not alternatives: a bound role is still named in \
continuityRoles, so on a reference_to_video shot the two lists are normally identical.
- A style role and a plate role are NEVER listed in referenceRoles: only character, prop and \
location roles are bound as reference subjects. A style is a look — say it in the prompt. A plate \
is a literal frame — put it in firstFrameRole or lastFrameRole. Either one may still be named in \
continuityRoles.";

/// The contract with its worked example on THIS model's envelope: the first allowed duration when
/// the model declares a menu, else a plain round number the "any positive value" rule admits, and
/// the conditioning form the envelope makes the default.
pub fn plan_json_contract(caps: &PlannerCapabilities, pack: &ReferencePack) -> String {
    let example = caps
        .durations
        .first()
        .map(|value| format!("{value}"))
        .unwrap_or_else(|| "6".to_owned());
    let (conditioning, reference_rule) = if caps.offers_references() {
        (
            // The same three roles the example's continuityRoles name: the one filled shot must
            // not model a referenceRoles list that is a SUBSET of what the shot depicts.
            "{ \"mode\": \"reference_to_video\", \"referenceRoles\": [\"mechanic\", \
             \"customer\", \"brass_key\"] }",
            REFERENCE_BINDING_RULE,
        )
    } else {
        ("{ \"mode\": \"text_to_video\" }", "")
    };
    let (lora_field, lora_rule) = if caps.turbo_loras.is_empty() {
        (String::new(), String::new())
    } else {
        let ids = caps
            .turbo_loras
            .iter()
            .map(|lora| format!("\"{}\"", lora.id))
            .collect::<Vec<_>>()
            .join(", ");
        (
            format!("  \"loras\": [{ids}],\n"),
            format!(
                "\n- loras is the top-level list of installed accelerators, copied EXACTLY as \
                 shown: [{ids}]. Write it. Omitting it, or naming any other id, makes this film \
                 render at the full step count — hours instead of minutes."
            ),
        )
    };
    let (continuity_field, continuity_rule, example_continuity) =
        if pack.references.iter().any(|entry| entry.approved) {
            (
                "[\"<the approved roles this shot depicts>\"]",
                "- Every shot needs at least one approved role in continuityRoles, and a shot \
                 lists every approved role that is on screen in it. Start from the array its beat \
                 says it MUST show and copy that array whole — it is checked role by role after \
                 you answer, and a beat that names two props and no location means exactly that, \
                 not one character, one prop and one place.",
                "[\"mechanic\", \"customer\", \"brass_key\"]",
            )
        } else {
            (
                "[]",
                "- This reference pack approves NO roles. Write continuityRoles: [] on EVERY \
                 shot. Characters, props and locations named only in the brief or prompt are \
                 prose, not role ids; never invent role ids for them.",
                "[]",
            )
        };
    PLAN_JSON_CONTRACT
        .replace(EXAMPLE_DURATION_PLACEHOLDER, &example)
        .replace(EXAMPLE_CONDITIONING_PLACEHOLDER, conditioning)
        .replace(CONTINUITY_FIELD_PLACEHOLDER, continuity_field)
        .replace(CONTINUITY_RULE_PLACEHOLDER, continuity_rule)
        .replace(EXAMPLE_CONTINUITY_PLACEHOLDER, example_continuity)
        .replace(REFERENCE_RULE_PLACEHOLDER, reference_rule)
        .replace(LORA_FIELD_PLACEHOLDER, &lora_field)
        .replace(LORA_RULE_PLACEHOLDER, &lora_rule)
}

/// The JSON contract both the first round and every repair round end with. Kept as one constant so
/// the two rounds cannot drift. It is a TEMPLATE: [`plan_json_contract`] fills the worked example's
/// duration from the envelope, so it is never sent raw.
pub const PLAN_JSON_CONTRACT: &str = "\
Answer with ONE JSON object and nothing else — no prose, no markdown fence, no commentary:

{
{{LORA_FIELD}}  \"shots\": [
    {
      \"id\": \"SH010\",
      \"beatId\": \"<a beat id from the list above>\",
      \"beat\": \"<one sentence: what this shot accomplishes in the story>\",
      \"framing\": \"<shot size, angle, camera movement>\",
      \"prompt\": \"<what the model should render, as a single paragraph>\",
      \"targetDurationSeconds\": <one of the allowed durations, copied exactly>,
      \"startState\": \"<the world at the first frame>\",
      \"endState\": \"<the world at the last frame>\",
      \"sound\": \"<the diegetic sound of this shot>\",
      \"dialogue\": \"<a spoken line, or omit the field>\",
      \"conditioning\": { \"mode\": \"<one of the allowed modes>\" },
      \"seed\": <an integer, optional>,
      \"continuityRoles\": {{CONTINUITY_FIELD}}
    }
  ]
}

Rules:
- Shot ids ascend in tens: SH010, SH020, SH030 ...
- Every field name is spelled exactly as above. An extra or misspelled field is rejected outright.
- Omit an optional field rather than writing null, \"\" or a placeholder.{{LORA_RULE}}
- beat, framing, prompt, startState and endState are STRINGS — one piece of prose each. Never an \
object, never a list, and never a role-by-role breakdown.
- startState and endState describe WHAT THE CAMERA SEES in the first and the last frame of this \
shot: who is in frame, where they are, and where the objects that matter are. They are different \
from each other — if they are not, the shot has no action in it.
- conditioning carries NOTHING BUT \"mode\" unless the mode itself takes a slot. A text_to_video \
shot's conditioning object is exactly { \"mode\": \"text_to_video\" } — writing firstFrameRole, \
lastFrameRole, referenceRoles or chainFromShotId on it is a fault, not extra detail. The four \
slot-bearing forms, and there are no others:
    image_to_video:      { \"mode\": \"image_to_video\", \"firstFrameRole\": \"<a role>\" }
    first_last_frame:    { \"mode\": \"first_last_frame\", \"firstFrameRole\": \"<a role>\", \
\"lastFrameRole\": \"<a DIFFERENT role>\" }
    reference_to_video:  { \"mode\": \"reference_to_video\", \"referenceRoles\": [\"<a role>\"] }
    any of the above, continuing the shot just before it: add \
\"chainFromShotId\": \"<the previous shot's id>\"
- A shot never carries keyframe roles and reference roles together; they are different \
conditioning tasks.
- first_last_frame needs TWO DIFFERENT roles. The same role in both slots is refused.
- chainFromShotId names the shot IMMEDIATELY BEFORE this one, or is omitted. It is never a \
substitute for continuityRoles: a chained shot still names the approved roles it depicts.
{{CONTINUITY_RULE}}{{REFERENCE_RULE}}

One filled shot, for shape only. It is from a DIFFERENT film: copy the spelling and the level of \
detail, never the content.

{
  \"id\": \"SH020\",
  \"beatId\": \"handover\",
  \"beat\": \"The mechanic hands the key across the counter and the customer takes it.\",
  \"framing\": \"Medium two-shot, eye level, locked off\",
  \"prompt\": \"A cramped garage office in hard noon light. A mechanic in an oil-stained shirt \
slides a brass key across a steel counter; the customer's hand closes around it and lifts it away. \
Dust turns in the light from the roller door behind them. The camera does not move.\",
  \"targetDurationSeconds\": {{EXAMPLE_DURATION}},
  \"startState\": \"The mechanic stands behind the counter with the brass key flat under their \
palm; the customer waits opposite with both hands at their sides.\",
  \"endState\": \"The counter is empty and the customer holds the brass key at chest height; the \
mechanic's hand is withdrawn.\",
  \"sound\": \"key scraping on steel, a compressor cycling somewhere off screen\",
  \"conditioning\": {{EXAMPLE_CONDITIONING}},
  \"continuityRoles\": {{EXAMPLE_CONTINUITY}}
}";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film_plan::REFERENCE_PACK_SCHEMA_VERSION;
    use serde_json::json;

    fn brief_json() -> Value {
        json!({
            "schemaVersion": 1,
            "id": "courier-workshop",
            "version": 1,
            "title": "Courier at the Workshop",
            "synopsis": "A courier leaves a red parcel; the recipient finds it.",
            "styleNotes": "Warm late-afternoon light, film grain.",
            "targetTotalSeconds": { "min": 30.0, "max": 60.0 },
            "requiredBeats": [
                { "id": "arrival", "summary": "The courier arrives at the workshop." },
                { "id": "delivery", "summary": "The parcel is left on the bench." },
                { "id": "discovery", "summary": "The recipient finds the parcel." }
            ],
            "model": { "id": "minimax_h3", "tier": "q4", "fps": 24, "resolution": "576x320" },
            "limits": { "maxRunSeconds": 7200, "maxShotSeconds": 2700, "maxAttemptsPerShot": 1, "maxMemoryGb": 96, "plannerMaxMemoryGb": 24 },
            "maxShots": 8
        })
    }

    fn brief() -> ProductionBrief {
        serde_json::from_value(brief_json()).expect("brief parses")
    }

    fn pack() -> ReferencePack {
        serde_json::from_value(json!({
            "schemaVersion": REFERENCE_PACK_SCHEMA_VERSION,
            "id": "courier-refs",
            "version": 1,
            "references": [
                { "role": "courier", "kind": "character", "file": "references/courier.png", "description": "Blue jacket." },
                { "role": "red_parcel", "kind": "prop", "file": "references/red_parcel.png", "description": "Red box." },
                { "role": "workbench_table", "kind": "prop", "file": "references/workbench_table.png", "description": "Scarred bench." },
                { "role": "workshop_location", "kind": "location", "file": "references/workshop_location.png", "description": "The workshop." },
                { "role": "workshop_plate", "kind": "plate", "file": "references/workshop_plate.png", "description": "Wide plate." },
                { "role": "house_style", "kind": "style", "file": "references/house_style.png", "description": "House look." },
                { "role": "draft_look", "kind": "style", "file": "references/draft.png", "description": "Not approved.", "approved": false }
            ]
        }))
        .expect("pack parses")
    }

    /// The single-entry view of a catalog entry, for the tests whose plans use one partition.
    fn single_entries(entry: &Map<String, Value>) -> ModelEntries<'_> {
        ModelEntries::single("minimax_h3", entry)
    }

    fn model_entry() -> Map<String, Value> {
        json!({
            "id": "minimax_h3",
            "type": "video",
            "capabilities": ["text_to_video", "image_to_video", "first_last_frame"],
            "video": { "supportsGuidance": false, "supportsNegativePrompt": false },
            "defaults": { "duration": 5.1667, "fps": 24, "resolution": "1344x768" },
            "limits": {
                "durations": [5.1667, 5.875, 14.375],
                "hardMinDuration": 5.1667,
                "hardMaxDuration": 14.375,
                "fps": [24],
                "maxPixels": 1032192,
                "resolutions": ["1344x768", "576x320"],
                "maxReferenceAssets": 0
            },
            "mlx": { "minMemoryGb": 64 }
        })
        .as_object()
        .cloned()
        .expect("object")
    }

    /// MiniMax-H3's REFERENCE partition as the catalog serves it: `reference_to_video` only, nine
    /// reference images, the same geometry and duration menus as the base entry (sc-23405).
    fn reference_entry() -> Map<String, Value> {
        let mut entry = model_entry();
        entry["id"] = json!("minimax_h3_ref");
        entry["capabilities"] = json!(["reference_to_video"]);
        entry["limits"]["maxReferenceAssets"] = json!(9);
        entry
    }

    /// A pack that approves nothing — the user who supplied no references (E1).
    fn pack_without_references() -> ReferencePack {
        serde_json::from_value(json!({
            "schemaVersion": REFERENCE_PACK_SCHEMA_VERSION,
            "id": "courier-refs",
            "version": 1,
            "references": [
                { "role": "draft_look", "kind": "style", "file": "references/draft.png", "description": "Not approved.", "approved": false }
            ]
        }))
        .expect("pack parses")
    }

    fn draft_shot(id: &str, beat_id: &str) -> Value {
        json!({
            "id": id,
            "beatId": beat_id,
            "beat": "something happens",
            "framing": "wide static",
            "prompt": "A workshop in warm light.",
            "targetDurationSeconds": 14.375,
            "startState": "empty",
            "endState": "not empty",
            "sound": "room tone",
            "conditioning": { "mode": "text_to_video" },
            "continuityRoles": ["courier", "red_parcel"]
        })
    }

    fn good_draft() -> PlannerDraft {
        serde_json::from_value(json!({
            "shots": [
                draft_shot("SH010", "arrival"),
                draft_shot("SH020", "delivery"),
                draft_shot("SH030", "discovery")
            ]
        }))
        .expect("draft parses")
    }

    fn messages(findings: &[PlanDiagnostic]) -> Vec<String> {
        findings.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn a_well_formed_draft_becomes_a_plan_the_hand_authored_validator_accepts() {
        let brief = brief();
        let draft = good_draft();
        let plan = draft_to_plan(&brief, &draft);
        assert_eq!(plan.id, "courier-workshop");
        assert_eq!(plan.model.id, "minimax_h3");
        assert_eq!(plan.limits.max_memory_gb, 96.0);
        let findings = validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            None,
        );
        assert!(findings.is_empty(), "{:?}", messages(&findings));
        // 3 x 14.375 = 43.125s, inside the 30-60s window.
        assert!(shape_findings(&brief, &plan).is_empty());
    }

    // ── sc-23406: the turbo recipes the envelope offers and the draft declares ───────────────

    /// All FOUR shipped MiniMax-H3 accelerators, in the order `GET /api/v1/loras` hands them back.
    ///
    /// That route sorts by `(scope, family, name)` (`apps/rust-api/src/loras.rs`) and all four
    /// share a scope and a family, so the display NAME alone orders them — which is why this list
    /// leads with the ref2v adapter and puts the 768p file ahead of the v0.1 one. A test driven
    /// from a hand-picked order would assert an outcome the route never produces; the guard below
    /// pins this literal to the shipped catalog's own names rather than to that reasoning.
    fn route_ordered_turbo_ids() -> Vec<String> {
        [
            "minimax_h3_ref2v_turbo_4step",
            "minimax_h3_turbo_4step_768p",
            "minimax_h3_turbo_4step_v01",
            "minimax_h3_turbo_8step",
        ]
        .iter()
        .map(|id| (*id).to_owned())
        .collect()
    }

    /// 🔴 The order above IS `list_loras`'s: sorting the shipped catalog's own display names
    /// reproduces it. Without this the offer tests would be driven from a literal nobody checked.
    #[test]
    fn the_route_hands_the_accelerators_back_in_display_name_order() {
        let mut by_name = route_ordered_turbo_ids();
        by_name.sort_by_key(|id| {
            let entry = crate::film_plan::builtin_plan_lora(id).expect("a shipped id");
            // Every one of the four is scope `builtin`, family `minimax-h3`; the name breaks it.
            (entry.family.clone(), entry.name.clone())
        });
        assert_eq!(by_name, route_ordered_turbo_ids());
    }

    /// The envelope built on what this host has INSTALLED: one accelerator per partition, paired
    /// by the catalog's own `modelIds` allowlist rather than by anything the test asserts twice.
    ///
    /// Reference conditioning is offered (the widened, pack-narrowed shape a mixed film plans
    /// against), because the reference partition's accelerator is offered only when it is.
    fn turbo_caps() -> PlannerCapabilities {
        capabilities_for(&brief().model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack())
            .with_installed_turbo_loras(&route_ordered_turbo_ids())
    }

    /// 🔴 The envelope offers at most ONE accelerator per partition, paired to the checkpoint its
    /// catalog entry names — and the contract and the prompt section both say so with the ids the
    /// draft must copy.
    ///
    /// The four-installed / two-offered shape is the point: THREE fl2v adapters are installed and
    /// only one is offered, because offering two would let a draft declare two schedules for one
    /// checkpoint — the conflict `validate_plan_structure` then refuses, after a full planning run.
    ///
    /// Which one is the rule's answer, not the list's: the route hands its ids back sorted by
    /// display name, so a first-match implementation offers `minimax_h3_turbo_4step_768p` (video
    /// shift 6.0) beside a ref2v adapter at shift 12.0 and samples the two halves of one film on
    /// two schedules. Schedule parity with the reference partition picks
    /// `minimax_h3_turbo_4step_v01`, which is what `plan.v2.turbo.jsonc`'s header says the shipped
    /// plan uses.
    #[test]
    fn the_envelope_offers_one_installed_accelerator_per_partition() {
        let caps = turbo_caps();
        let offered: Vec<(&str, &str, u32)> = caps
            .turbo_loras
            .iter()
            .map(|lora| (lora.id.as_str(), lora.model_id.as_str(), lora.steps))
            .collect();
        assert_eq!(
            offered,
            vec![
                ("minimax_h3_turbo_4step_v01", "minimax_h3", 4),
                ("minimax_h3_ref2v_turbo_4step", "minimax_h3_ref", 4),
            ]
        );

        let section = caps.as_prompt_section();
        assert!(
            section.contains("minimax_h3_turbo_4step_v01 (4 steps on minimax_h3)")
                && section.contains("minimax_h3_ref2v_turbo_4step (4 steps on minimax_h3_ref)")
                && section.contains("ON BY DEFAULT"),
            "{section}"
        );
        let contract = plan_json_contract(&caps, &pack());
        assert!(
            contract.contains(
                "\"loras\": [\"minimax_h3_turbo_4step_v01\", \"minimax_h3_ref2v_turbo_4step\"]"
            ),
            "the answer's shape carries the ids to copy: {contract}"
        );

        // The choice is the RULE's, not the list's: permuting what the host reports changes
        // nothing. (Reversed, a first-match implementation offers the 8-step adapter instead.)
        let mut reversed = route_ordered_turbo_ids();
        reversed.reverse();
        let permuted = capabilities_for(&brief().model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack())
            .with_installed_turbo_loras(&reversed);
        assert_eq!(permuted.turbo_loras, caps.turbo_loras);

        // A host with nothing installed is shown NO loras field and no rule about one — an
        // instruction it could only be refused for following.
        let bare = capabilities_for(&brief().model, &model_entry(), ModelLane::Mlx)
            .with_installed_turbo_loras(&[]);
        assert!(bare.turbo_loras.is_empty());
        assert!(
            bare.as_prompt_section().contains("none are installed"),
            "{}",
            bare.as_prompt_section()
        );
        assert!(
            !plan_json_contract(&bare, &pack()).contains("\"loras\""),
            "{}",
            plan_json_contract(&bare, &pack())
        );
    }

    /// 🔴 The reference partition's accelerator is offered only when this envelope offers
    /// reference conditioning at all.
    ///
    /// An envelope narrowed to a pack that approves nothing — or built where the catalog serves no
    /// reference partition — can produce no shot that dispatches there, so the ref2v adapter is an
    /// id the planner would copy into a list that reaches no shot, and a second name to get wrong.
    ///
    /// With no reference partition there is also no parity anchor, so the base offer follows the
    /// plan canvas. The shipped 576x320 film canvas is nearer the 544p v0.1 recipe than the 768p
    /// recipe and therefore keeps the same base adapter used by a mixed film.
    #[test]
    fn the_reference_accelerator_is_offered_only_when_references_are() {
        let offered = |caps: PlannerCapabilities| -> Vec<(String, String)> {
            caps.with_installed_turbo_loras(&route_ordered_turbo_ids())
                .turbo_loras
                .iter()
                .map(|lora| (lora.id.clone(), lora.model_id.clone()))
                .collect()
        };
        let base_only = capabilities_for(&brief().model, &model_entry(), ModelLane::Mlx);
        assert!(!base_only.offers_references());
        let base_offer = vec![(
            "minimax_h3_turbo_4step_v01".to_owned(),
            "minimax_h3".to_owned(),
        )];
        assert_eq!(offered(base_only), base_offer);
        // The tiebreak is the CATALOG's order, not the host's: reversing what the host reports
        // must not move a tie (both 4-step files) onto the other file.
        let mut reversed = route_ordered_turbo_ids();
        reversed.reverse();
        assert_eq!(
            capabilities_for(&brief().model, &model_entry(), ModelLane::Mlx)
                .with_installed_turbo_loras(&reversed)
                .turbo_loras
                .iter()
                .map(|lora| (lora.id.clone(), lora.model_id.clone()))
                .collect::<Vec<_>>(),
            base_offer
        );

        // The reference partition IS served here, but the pack approves nothing bindable, so the
        // envelope narrows out of reference conditioning and the offer goes with it.
        let narrowed = capabilities_for(&brief().model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack_without_references());
        assert!(!narrowed.offers_references());
        assert_eq!(
            offered(narrowed)
                .iter()
                .map(|(_, partition)| partition.clone())
                .collect::<Vec<_>>(),
            vec!["minimax_h3".to_owned()]
        );
    }

    /// 🔴 The three tiers of the offer rule, driven directly so each one is asserted where it
    /// DECIDES rather than only where the shipped catalog happens to exercise it.
    ///
    /// The canvas tier exercises both an exact match and a nearest-canvas choice.
    #[test]
    fn the_offer_rule_prefers_parity_then_canvas_then_fewest_steps() {
        let recipe = |id: &str, steps: u32, video_shift: f32, edge: Option<u32>| TurboRecipe {
            lora_id: id.to_owned(),
            name: id.to_owned(),
            steps,
            video_shift,
            audio_shift: 3.0,
            training_short_edge: edge,
        };
        let fast_768 = recipe("fast_768", 4, 6.0, Some(768));
        let fast_544 = recipe("fast_544", 4, 12.0, Some(544));
        let slow_320 = recipe("slow_320", 8, 12.0, Some(320));
        let candidates = [&fast_768, &fast_544, &slow_320];

        // 1. Parity with the partition already resolved wins, even though the anchor's match is
        //    neither first nor the fewest-steps answer's equal.
        let anchor = recipe("ref", 8, 12.0, None);
        assert_eq!(
            PlannerCapabilities::choose_turbo_offer(&candidates, Some(&anchor), Some(768)),
            Some(2),
            "the 8-step schedule the other partition runs"
        );
        // 2. No parity: the declared canvas matching the plan's short edge, over fewer steps.
        assert_eq!(
            PlannerCapabilities::choose_turbo_offer(&candidates, None, Some(320)),
            Some(2)
        );
        // 3. Nearest canvas wins; equal distances then use steps and catalog order.
        assert_eq!(
            PlannerCapabilities::choose_turbo_offer(&candidates, None, Some(544)),
            Some(1)
        );
        assert_eq!(
            PlannerCapabilities::choose_turbo_offer(&candidates, None, None),
            Some(0)
        );
        assert_eq!(
            PlannerCapabilities::choose_turbo_offer(&[], None, None),
            None
        );
    }

    /// A scripted draft that declares the offered accelerators produces a plan carrying them, and
    /// that plan validates — the default path, which is what most runs take.
    #[test]
    fn a_draft_declaring_the_offered_accelerators_becomes_a_turbo_plan() {
        let brief = brief();
        let draft: PlannerDraft = serde_json::from_value(json!({
            "loras": ["minimax_h3_turbo_4step_v01", "minimax_h3_ref2v_turbo_4step"],
            "shots": [
                draft_shot("SH010", "arrival"),
                draft_shot("SH020", "delivery"),
                draft_shot("SH030", "discovery")
            ]
        }))
        .expect("draft parses");
        let plan = draft_to_plan(&brief, &draft);
        assert_eq!(
            plan.model.loras,
            vec!["minimax_h3_turbo_4step_v01", "minimax_h3_ref2v_turbo_4step"],
            "the declared regime reaches the plan's model block"
        );
        let findings = validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            Some(&turbo_caps()),
        );
        assert!(findings.is_empty(), "{:?}", messages(&findings));
    }

    #[test]
    fn omitting_the_default_turbo_recipe_is_repairable_but_explicit_regimes_are_not() {
        let caps = turbo_caps();
        let draft = good_draft();
        let default_brief = brief();
        let plan = draft_to_plan(&default_brief, &draft);
        let findings = messages(&lora_offer_findings(&caps, &default_brief, &plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("omitted the default installed Turbo recipe")
                && findings[0].contains("minimax_h3_turbo_4step_v01")
                && findings[0].contains("minimax_h3_ref2v_turbo_4step"),
            "{}",
            findings[0]
        );

        let mut quality_document = brief_json();
        quality_document["preferQuality"] = json!(true);
        let quality_brief: ProductionBrief =
            serde_json::from_value(quality_document).expect("quality brief parses");
        let quality_plan = draft_to_plan(&quality_brief, &draft);
        assert!(
            lora_offer_findings(&caps, &quality_brief, &quality_plan).is_empty(),
            "Full quality explicitly opts out of Turbo"
        );

        let mut custom_document = brief_json();
        custom_document["model"]["advanced"] = json!({"steps": 7});
        let custom_brief: ProductionBrief =
            serde_json::from_value(custom_document).expect("custom brief parses");
        let custom_plan = draft_to_plan(&custom_brief, &draft);
        assert!(
            lora_offer_findings(&caps, &custom_brief, &custom_plan).is_empty(),
            "a custom step count explicitly opts out of the default recipe"
        );
    }

    /// 🔴 A draft naming an accelerator this host does NOT offer is refused BY NAME, and the
    /// refusal hands back the ids that are offered — which is what turns it into a repair rather
    /// than a dead end.
    ///
    /// Two shapes, because they fail in different places: an id no catalog carries at all (the
    /// document validator's refusal) and an id the catalog carries, and this host even has
    /// installed, but the envelope did not offer (the envelope's, which the document validator
    /// cannot see).
    #[test]
    fn a_draft_naming_an_unoffered_accelerator_is_refused_by_name() {
        let brief = brief();
        let caps = turbo_caps();
        // Installed here, and NOT offered: it is the second recipe on the base partition, so a
        // draft that writes it is writing an id the envelope never showed it.
        let draft: PlannerDraft = serde_json::from_value(json!({
            "loras": ["minimax_h3_turbo_4step_768p"],
            "shots": [
                draft_shot("SH010", "arrival"),
                draft_shot("SH020", "delivery"),
                draft_shot("SH030", "discovery")
            ]
        }))
        .expect("draft parses");
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            Some(&caps),
        ));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("model.loras")
                && findings[0].contains("minimax_h3_turbo_4step_768p")
                && findings[0].contains("minimax_h3_ref2v_turbo_4step"),
            "the refusal names the id AND what to write instead: {}",
            findings[0]
        );

        // An INVENTED id is refused by the document validator too, so it is caught even on a run
        // with no envelope in hand.
        let draft: PlannerDraft = serde_json::from_value(json!({
            "loras": ["minimax_h3_turbo_fastest"],
            "shots": [
                draft_shot("SH010", "arrival"),
                draft_shot("SH020", "delivery"),
                draft_shot("SH030", "discovery")
            ]
        }))
        .expect("draft parses");
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            Some(&caps),
        ));
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("minimax_h3_turbo_fastest")),
            "{findings:?}"
        );
    }

    /// 🔴 An adapter the BRIEF declared is judged on install state, not on the offer policy.
    ///
    /// The offer policy is a rule for the planner — one recipe per partition, the envelope's pick —
    /// and the draft never wrote this id, so a finding against it is one no repair round can clear:
    /// the planner would be asked, round after round, to withdraw something it did not write, until
    /// `generate` gave up. What the brief may NOT do is name an adapter this host has not
    /// installed, because that 400s at enqueue whoever chose it.
    #[test]
    fn a_brief_declared_accelerator_is_judged_on_install_state_not_on_the_offer() {
        let mut document = brief_json();
        // Installed on this host (all four are) but NOT offered: a second recipe for the base
        // partition, which is exactly the shape the offer policy withholds from the planner.
        document["model"]["loras"] = json!(["minimax_h3_turbo_8step"]);
        let author_brief: ProductionBrief = serde_json::from_value(document).expect("brief parses");
        let draft = good_draft();
        let plan = draft_to_plan(&author_brief, &draft);
        assert_eq!(
            plan.model.loras,
            vec!["minimax_h3_turbo_8step"],
            "the brief's own selection survives the draft"
        );
        let findings = messages(&validate_generated_plan(
            &author_brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            Some(&turbo_caps()),
        ));
        assert!(findings.is_empty(), "{findings:?}");

        // Not installed, though: the envelope's own installed list is what says so, and the
        // refusal names the id.
        let uninstalled = capabilities_for(&brief().model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack())
            .with_installed_turbo_loras(&["minimax_h3_ref2v_turbo_4step".to_owned()]);
        let findings = messages(&validate_generated_plan(
            &author_brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            Some(&uninstalled),
        ));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("minimax_h3_turbo_8step") && findings[0].contains("not installed"),
            "{}",
            findings[0]
        );
    }

    /// `brief.preferQuality` keeps the 50-step path: the accelerators are not offered, and a draft
    /// that declares them anyway has them STRIPPED rather than honoured.
    ///
    /// Stripped, not refused, because the brief is the authority on the regime and a repair round
    /// spent teaching an 8B model to withhold a field it was never shown is a decode wasted on
    /// nothing.
    #[test]
    fn prefer_quality_keeps_the_full_step_path() {
        let mut document = brief_json();
        document["preferQuality"] = json!(true);
        let quality_brief: ProductionBrief =
            serde_json::from_value(document).expect("brief parses");
        assert!(quality_brief.prefer_quality);
        let draft: PlannerDraft = serde_json::from_value(json!({
            "loras": ["minimax_h3_turbo_4step_v01"],
            "shots": [
                draft_shot("SH010", "arrival"),
                draft_shot("SH020", "delivery"),
                draft_shot("SH030", "discovery")
            ]
        }))
        .expect("draft parses");
        let plan = draft_to_plan(&quality_brief, &draft);
        assert!(
            plan.model.loras.is_empty(),
            "preferQuality clears the regime: {:?}",
            plan.model.loras
        );
        // And the default brief does NOT strip it, so the assertion above is about the flag.
        assert_eq!(
            draft_to_plan(&brief(), &draft).model.loras,
            vec!["minimax_h3_turbo_4step_v01"]
        );
    }

    #[test]
    fn malformed_planner_output_is_refused_rather_than_coerced() {
        assert!(parse_planner_output("").is_err());
        assert!(parse_planner_output("I cannot help with that.").is_err());
        assert!(parse_planner_output("{\"shots\": [").is_err());
        // An unknown field is refused outright, not dropped.
        let error = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": {"mode": "text_to_video"}, "cameraLens": "35mm"}]}"#,
        )
        .expect_err("unknown field refused");
        assert!(error.contains("cameraLens"), "{error}");
        // A missing required field is refused.
        let error = parse_planner_output(r#"{"shots": [{"id": "SH010"}]}"#)
            .expect_err("incomplete shot refused");
        assert!(error.contains("beatId"), "{error}");
        // Prose around a well-formed object still parses — the object is isolated.
        let draft = parse_planner_output(&format!(
            "Sure! Here is the plan:\n{}\nLet me know if you want changes.",
            serde_json::to_string(&good_draft()).unwrap()
        ))
        .expect("isolated object parses");
        assert_eq!(draft.shots.len(), 3);
    }

    #[test]
    fn a_parse_failure_names_the_field_path_the_planner_can_act_on() {
        // The real sc-22713 smoke failure: `startState` as an object on a shot whose fields are
        // otherwise fine. The 8B planner never sees its answer as numbered lines, so "line 10
        // column 20" told it nothing and all three decodes came back identical.
        let error = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": {"mode": "text_to_video"}},
                {"id": "SH020", "beatId": "delivery", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s",
                "endState": 12, "conditioning": {"mode": "text_to_video"}}]}"#,
        )
        .expect_err("a wrong type is refused");
        assert!(
            error.contains("shots[1].endState") && error.contains("expected a string"),
            "{error}"
        );
        assert!(!error.contains("line 1"), "{error}");
    }

    #[test]
    fn a_structurally_close_draft_is_normalised_rather_than_bounced() {
        // Verbatim shapes from the refused real-LLM draft (run2-planner-refused): the states as
        // role -> description objects, and `dialogue`/`chainFromShotId` as empty-string
        // placeholders the contract asks to be omitted.
        let draft = parse_planner_output(
            r#"{"shots": [{
                "id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f", "prompt": "p",
                "targetDurationSeconds": 5.1667,
                "startState": {"workshop_plate": "empty workshop", "courier": "not yet in frame"},
                "endState": ["the courier fills the doorway", "the parcel is against their chest"],
                "dialogue": "", "sound": "  ",
                "conditioning": {
                    "mode": "text_to_video", "firstFrameRole": "", "lastFrameRole": "",
                    "referenceRoles": [], "chainFromShotId": ""
                },
                "continuityRoles": ["courier"]
            }]}"#,
        )
        .expect("a near-miss draft parses");
        let shot = &draft.shots[0];
        // Deterministic: keys SORTED, joined into one line. Not `Map` order — `serde_json`'s map is
        // a BTreeMap or an IndexMap depending on whether anything in the build graph enables
        // `preserve_order`, so the same draft would otherwise normalise to a different plan (and a
        // different plan sha256) in a different build. The reversed spelling below is the same
        // object written the other way round and must normalise identically.
        assert_eq!(
            shot.start_state,
            "courier: not yet in frame; workshop_plate: empty workshop"
        );
        let reversed = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667,
                "startState": {"workshop_plate": "empty workshop", "courier": "not yet in frame"},
                "endState": "e", "conditioning": {"mode": "text_to_video"}}]}"#,
        )
        .expect("parses");
        let forwards = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667,
                "startState": {"courier": "not yet in frame", "workshop_plate": "empty workshop"},
                "endState": "e", "conditioning": {"mode": "text_to_video"}}]}"#,
        )
        .expect("parses");
        assert_eq!(reversed.shots[0].start_state, forwards.shots[0].start_state);
        assert_eq!(
            shot.end_state,
            "the courier fills the doorway; the parcel is against their chest"
        );
        // A blank placeholder is an omission, not a value: `chainFromShotId: ""` would otherwise be
        // a chain to a shot named "" and `firstFrameRole: ""` a role no pack can contain.
        assert_eq!(shot.dialogue, None);
        assert_eq!(shot.sound, None);
        assert_eq!(shot.conditioning.chain_from_shot_id, None);
        assert_eq!(shot.conditioning.first_frame_role, None);
        assert_eq!(shot.conditioning.last_frame_role, None);
        // Normalisation never invents: a state object with nothing in it stays empty, and the
        // ordinary structural validator reports it as the missing field it is.
        let draft = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": {}, "endState": "e",
                "conditioning": {"mode": "text_to_video"}}]}"#,
        )
        .expect("an empty object flattens");
        assert_eq!(draft.shots[0].start_state, "");

        // The fault EVERY draft of the sc-22713 smoke made on EVERY shot, through all three
        // rounds: keyframe slots filled on a text_to_video shot. The mode is what the shot says
        // about its conditioning; a slot the mode cannot use conditions nothing.
        let draft = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": {
                    "mode": "text_to_video", "firstFrameRole": "workshop_plate",
                    "lastFrameRole": "courier", "referenceRoles": []
                },
                "continuityRoles": ["courier"]}]}"#,
        )
        .expect("a text_to_video shot with filled keyframe slots parses");
        assert_eq!(draft.shots[0].conditioning.first_frame_role, None);
        assert_eq!(draft.shots[0].conditioning.last_frame_role, None);

        // A slot the mode DOES take is untouched, and so is the second slot's absence.
        let draft = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": {
                    "mode": "image_to_video", "firstFrameRole": "workshop_plate",
                    "lastFrameRole": "courier"
                },
                "continuityRoles": ["courier"]}]}"#,
        )
        .expect("an image_to_video shot parses");
        assert_eq!(
            draft.shots[0].conditioning.first_frame_role.as_deref(),
            Some("workshop_plate")
        );
        assert_eq!(draft.shots[0].conditioning.last_frame_role, None);

        // A NON-EMPTY referenceRoles list is a claim about what the shot depicts, so it survives
        // normalisation and is reported by the validator instead of being deleted.
        let draft = parse_planner_output(
            r#"{"shots": [{"id": "SH010", "beatId": "arrival", "beat": "b", "framing": "f",
                "prompt": "p", "targetDurationSeconds": 5.1667, "startState": "s", "endState": "e",
                "conditioning": { "mode": "text_to_video", "referenceRoles": ["courier"] },
                "continuityRoles": ["courier"]}]}"#,
        )
        .expect("parses");
        assert_eq!(draft.shots[0].conditioning.reference_roles, ["courier"]);
        let plan = draft_to_plan(&brief(), &draft);
        assert!(
            messages(&crate::film_plan::validate_plan_structure(&plan))
                .iter()
                .any(|message| message.contains("does not take reference roles")),
            "the surviving list is reported"
        );
    }

    #[test]
    fn a_beat_that_names_required_roles_is_not_covered_by_a_shot_that_omits_them() {
        let mut brief = brief();
        brief.required_beats[1].required_roles =
            vec!["red_parcel".to_owned(), "courier".to_owned()];
        let draft = good_draft();
        let mut plan = draft_to_plan(&brief, &draft);
        // The shot for the beat is present and names one of the two roles.
        plan.shots[1].continuity_roles = vec!["courier".to_owned()];
        let findings = messages(&role_coverage_findings(&brief, &plan, &pack()));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("beat \"delivery\"") && findings[0].contains("red_parcel"),
            "{findings:?}"
        );
        // Only the role that is actually missing is reported as missing; `courier` appears in the
        // message only as part of what the beat is about and of what the shot already binds.
        assert!(findings[0].contains("missing red_parcel;"), "{findings:?}");
        assert!(findings[0].contains("binds only courier,"), "{findings:?}");

        // Bound in a conditioning slot rather than continuityRoles still counts as on screen.
        plan.shots[1].conditioning.mode = "image_to_video".to_owned();
        plan.shots[1].conditioning.first_frame_role = Some("red_parcel".to_owned());
        assert!(role_coverage_findings(&brief, &plan, &pack()).is_empty());

        // A beat with no declared roles makes no claim, which is what every pre-sc-22713 brief is.
        brief.required_beats[1].required_roles.clear();
        plan.shots[1].conditioning.first_frame_role = None;
        plan.shots[1].conditioning.mode = "text_to_video".to_owned();
        assert!(role_coverage_findings(&brief, &plan, &pack()).is_empty());
    }

    #[test]
    fn a_brief_cannot_require_a_role_the_pack_does_not_approve() {
        let clean = brief();
        let mut brief = brief();
        brief.required_beats[0].required_roles = vec![
            "courier".to_owned(),
            "draft_look".to_owned(),
            "wolf".to_owned(),
        ];
        let findings = messages(&brief_pack_findings(&brief, &pack()));
        assert_eq!(findings.len(), 2, "{findings:?}");
        // `draft_look` is in the pack but unapproved, so it is as unusable as one that is absent.
        assert!(
            findings.iter().any(|m| m.contains("\"draft_look\"")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("\"wolf\"")),
            "{findings:?}"
        );
        assert!(brief_pack_findings(&clean, &pack()).is_empty());
    }

    #[test]
    fn a_dropped_beat_is_a_finding_that_names_the_beat() {
        let brief = brief();
        let mut draft = good_draft();
        draft.shots.retain(|shot| shot.beat_id != "delivery");
        let findings = messages(&coverage_findings(&brief, &draft));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("required beat \"delivery\"")
                && findings[0].contains("The parcel is left on the bench."),
            "{findings:?}"
        );

        // An invented beat id is named too.
        let mut draft = good_draft();
        draft.shots[1].beat_id = "montage".to_owned();
        let findings = messages(&coverage_findings(&brief, &draft));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH020] beatId") && m.contains("montage")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("required beat \"delivery\"")),
            "{findings:?}"
        );

        // Beats covered out of the brief's order are a finding.
        let mut draft = good_draft();
        draft.shots.swap(0, 1);
        let findings = messages(&coverage_findings(&brief, &draft));
        assert!(
            findings.iter().any(|m| m.contains("narrative order")),
            "{findings:?}"
        );
    }

    #[test]
    fn impossible_timing_and_oversized_drafts_are_findings_not_silent_coercion() {
        let brief = brief();
        let mut draft = good_draft();
        // Three of the shortest rung is 15.5s — under the brief's 30s floor.
        for shot in &mut draft.shots {
            shot.target_duration_seconds = 5.1667;
        }
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&shape_findings(&brief, &plan));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("30-60s window"), "{findings:?}");
        // The durations themselves are untouched — nothing was rounded to fit.
        assert_eq!(plan.shots[0].target_duration_seconds, 5.1667);

        // A duration off the model's menu is a model finding, never a coercion.
        let mut draft = good_draft();
        draft.shots[0].target_duration_seconds = 12.0;
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            None,
        ));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH010] targetDurationSeconds") && m.contains("menu")),
            "{findings:?}"
        );

        let mut draft = good_draft();
        for index in 0..10 {
            let mut extra = draft.shots[0].clone();
            extra.id = format!("SH1{index}0");
            draft.shots.push(extra);
        }
        let plan = draft_to_plan(&brief, &draft);
        assert!(
            messages(&shape_findings(&brief, &plan))
                .iter()
                .any(|m| m.contains("exceed the 8 this brief allows")),
            "{:?}",
            messages(&shape_findings(&brief, &plan))
        );
    }

    #[test]
    fn unsupported_conditioning_from_the_planner_is_refused_against_the_installed_entry() {
        let brief = brief();
        let mut draft = good_draft();
        draft.shots[1].conditioning = ShotConditioning {
            mode: "reference_to_video".to_owned(),
            first_frame_role: None,
            last_frame_role: None,
            reference_roles: vec!["courier".to_owned()],
            chain_from_shot_id: None,
        };
        draft.shots[2].negative_prompt = Some("blurry".to_owned());
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            None,
        ));
        // The reference shot resolves to the family's reference partition (sc-23402), and only the
        // base entry is installed here — so it is refused by name rather than dispatched at the
        // base checkpoint, which serves no reference conditioning at all.
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH020] conditioning.referenceRoles")
                    && m.contains("minimax_h3_ref")
                    && m.contains("not in this API's model catalog")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("[SH030] negativePrompt")),
            "{findings:?}"
        );

        // An unapproved role cannot be conditioned on. The presence of other references in the
        // pack does not independently require this shot to invent a continuity binding.
        let mut draft = good_draft();
        draft.shots[0].conditioning = ShotConditioning {
            mode: "image_to_video".to_owned(),
            first_frame_role: Some("draft_look".to_owned()),
            last_frame_role: None,
            reference_roles: Vec::new(),
            chain_from_shot_id: None,
        };
        draft.shots[0].continuity_roles = Vec::new();
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            None,
        ));
        assert!(
            findings.iter().any(|m| m.contains("not approved")),
            "{findings:?}"
        );
        assert_eq!(findings.len(), 1, "{findings:?}");
    }

    #[test]
    fn the_capability_envelope_reports_only_what_the_entry_declares() {
        let brief = brief();
        let caps = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        assert_eq!(
            caps.modes,
            vec!["text_to_video", "image_to_video", "first_last_frame"]
        );
        assert_eq!(caps.fps, Some(24));
        assert_eq!(caps.default_resolution.as_deref(), Some("576x320"));
        assert_eq!(caps.max_reference_images, 0);
        assert!(!caps.supports_negative_prompt);
        assert_eq!(caps.min_memory_gb, Some(64.0));
        let section = caps.as_prompt_section();
        assert!(section.contains("5.1667, 5.875, 14.375"), "{section}");
        assert!(section.contains("THIS CHECKPOINT HAS NONE"), "{section}");
        assert!(
            section.contains("Never write a negativePrompt"),
            "{section}"
        );
        // reference_to_video is not declared, so it is never offered.
        assert!(!section.contains("Allowed conditioning modes (use no others): text_to_video, image_to_video, first_last_frame, reference_to_video"));

        // A model WITH references says so instead.
        let mut entry = model_entry();
        entry["capabilities"] = json!(["text_to_video", "reference_to_video"]);
        entry["limits"]["maxReferenceAssets"] = json!(3);
        entry["video"] = json!({ "supportsNegativePrompt": true });
        let caps = capabilities_for(&brief.model, &entry, ModelLane::Mlx);
        assert_eq!(caps.max_reference_images, 3);
        let section = caps.as_prompt_section();
        assert!(section.contains("at most 3 reference roles"), "{section}");
        assert!(!section.contains("negativePrompt"), "{section}");
    }

    /// sc-23405 AC2. The envelope the planner is held to is widened to the family's REFERENCE
    /// partition — the entry a reference shot actually dispatches against — and then narrowed to
    /// what the supplied pack can fill. Both halves matter: without the first the planner can never
    /// write a reference shot (where sc-23402 left it); without the second a user who supplied no
    /// references is offered a mode no draft of theirs could satisfy.
    #[test]
    fn the_envelope_widens_to_the_reference_partition_and_narrows_to_the_pack() {
        let brief = brief();
        let base = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        assert!(!base.offers_references());

        let widened = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack());
        // The base modes SURVIVE the widening: the partition is resolved per shot, so a plan may
        // mix a reference shot with a text-to-video one.
        assert_eq!(
            widened.modes,
            vec![
                "text_to_video",
                "image_to_video",
                "first_last_frame",
                "reference_to_video"
            ]
        );
        assert_eq!(widened.max_reference_images, 9);
        assert!(widened.offers_references());
        // Nothing else moved: the geometry, the menu and the memory floor are still the base
        // entry's, which is what the two partitions agree on.
        assert_eq!(widened.durations, base.durations);
        assert_eq!(widened.default_resolution, base.default_resolution);
        assert_eq!(widened.min_memory_gb, base.min_memory_gb);

        let section = widened.as_prompt_section();
        assert!(section.contains("at most 9 reference roles"), "{section}");
        assert!(
            section.contains("Every shot binds, in referenceRoles"),
            "{section}"
        );
        assert!(
            section.contains("reference_to_video is the DEFAULT"),
            "{section}"
        );

        // The SAME model and the SAME partition, with a pack that approves nothing: the mode and
        // the cap come straight back off, and the planner is shown the phase-1 envelope.
        let narrowed = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack_without_references());
        assert_eq!(narrowed.modes, base.modes);
        assert_eq!(narrowed.max_reference_images, 0);
        assert!(!narrowed.offers_references());
        let section = narrowed.as_prompt_section();
        assert!(section.contains("THIS CHECKPOINT HAS NONE"), "{section}");
        assert!(
            !section.contains("reference_to_video is the DEFAULT"),
            "{section}"
        );
    }

    /// sc-23405 review. "The pack can fill a reference shot" counts only the kinds a
    /// `reference_to_video` shot may BIND ([`BINDABLE_REFERENCE_KINDS`]).
    ///
    /// A pack may approve a style and a plate and still approve no SUBJECT: Ref2VA binds every image
    /// as a thing to depict, a style is a look rather than a thing, and a plate is placed through the
    /// keyframe slots instead. Counting either as "fillable" offers the planner a mode whose every
    /// use would be refused a decode later — the narrowing has to agree with what a shot can bind,
    /// not merely with whether the pack has any approved row at all.
    #[test]
    fn a_pack_approving_only_a_style_and_a_plate_fills_no_reference_shot() {
        let brief = brief();
        let base = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        let style_and_plate_only: ReferencePack = serde_json::from_value(json!({
            "schemaVersion": REFERENCE_PACK_SCHEMA_VERSION,
            "id": "courier-refs",
            "version": 1,
            "references": [
                { "role": "house_style", "kind": "style", "file": "references/style.png", "description": "Approved look." },
                { "role": "workshop_plate", "kind": "plate", "file": "references/plate.png", "description": "Approved plate." }
            ]
        }))
        .expect("pack parses");
        // Both rows are APPROVED — the narrowing must turn on the KIND, not on approval.
        assert!(style_and_plate_only
            .references
            .iter()
            .all(|entry| entry.approved));

        let narrowed = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&style_and_plate_only);
        assert_eq!(
            narrowed.modes, base.modes,
            "an approved style and an approved plate are not subjects a reference shot can bind, \
             so the envelope is the phase-1 one"
        );
        assert_eq!(narrowed.max_reference_images, 0);
        assert!(!narrowed.offers_references());

        // The control: add ONE approved bindable row to the same pack and it widens again, so what
        // this asserts is the kind filter and not a narrowing that never widens.
        let mut with_a_subject = style_and_plate_only;
        with_a_subject.references.push(
            serde_json::from_value(json!({
                "role": "courier", "kind": "character", "file": "references/courier.png",
                "description": "Blue jacket."
            }))
            .expect("entry parses"),
        );
        let widened = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&with_a_subject);
        assert!(widened.offers_references());
        assert_eq!(widened.max_reference_images, 9);
    }

    /// sc-23405. The worked example and the standing reference rule follow the ENVELOPE, for the
    /// same reason the duration does (AT4, sc-22715): the one filled shot a planner is shown is the
    /// strongest instruction in the contract, so on an envelope whose default is
    /// `reference_to_video` it must not model a `text_to_video` shot.
    #[test]
    fn the_contracts_worked_example_and_reference_rule_follow_the_envelope() {
        let brief = brief();
        let with_references = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack());
        let contract = plan_json_contract(&with_references, &pack());
        assert!(
            contract.contains(
                "\"conditioning\": { \"mode\": \"reference_to_video\", \"referenceRoles\": \
                 [\"mechanic\", \"customer\", \"brass_key\"] },"
            ),
            "{contract}"
        );
        assert!(
            contract.contains("uses \"mode\": \"reference_to_video\" and lists"),
            "{contract}"
        );

        let without = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        let contract = plan_json_contract(&without, &pack());
        assert!(
            contract.contains("\"conditioning\": { \"mode\": \"text_to_video\" },"),
            "{contract}"
        );
        assert!(
            !contract.contains("An approved reference pack is available"),
            "{contract}"
        );

        // No placeholder ever reaches the planner unfilled, on either envelope or in either round.
        for caps in [&with_references, &without] {
            for text in [
                build_planner_request(&brief, &pack(), caps),
                build_repair_request(&brief, &pack(), caps, "{}", &[], 1, 1),
            ] {
                for placeholder in [
                    EXAMPLE_DURATION_PLACEHOLDER,
                    EXAMPLE_CONDITIONING_PLACEHOLDER,
                    CONTINUITY_FIELD_PLACEHOLDER,
                    CONTINUITY_RULE_PLACEHOLDER,
                    EXAMPLE_CONTINUITY_PLACEHOLDER,
                    REFERENCE_RULE_PLACEHOLDER,
                ] {
                    assert!(
                        !text.contains(placeholder),
                        "{placeholder} survived: {text}"
                    );
                }
            }
        }
    }

    /// A script-only film has prose nouns but no approved role ids. Its initial prompt and every
    /// repair round must therefore model empty continuity arrays all the way through the shape,
    /// standing rule and filled example. The repair's actionable findings are repeated after that
    /// long shared contract so a model cannot simply echo the refused draft again.
    #[test]
    fn an_empty_pack_contract_and_repair_never_teach_invented_continuity_roles() {
        let brief = brief();
        let pack = pack_without_references();
        let caps = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack);
        let request = build_planner_request(&brief, &pack, &caps);
        let contract = plan_json_contract(&caps, &pack);

        assert!(
            request.contains(
                "This reference pack approves NO roles. Write continuityRoles: [] on EVERY shot."
            ),
            "{request}"
        );
        assert_eq!(
            contract.matches("\"continuityRoles\": []").count(),
            2,
            "the output shape and filled example both model the legal empty array: {contract}"
        );
        assert!(
            !contract.contains("\"continuityRoles\": [\"mechanic\"")
                && !contract.contains("Every shot needs at least one approved role"),
            "the different-film example must not invent role ids for this pack: {contract}"
        );

        let findings = vec![
            PlanDiagnostic::shot(
                "SH010",
                "continuityRoles",
                "role \"courier\" is not approved by this film's reference pack",
            ),
            PlanDiagnostic::plan(
                "shots.duration",
                "total duration 13.1667s is above the brief maximum 12.91675s",
            ),
        ];
        let repair = build_repair_request(&brief, &pack, &caps, "{\"shots\": []}", &findings, 1, 2);
        assert!(
            !repair.contains("shortens the film"),
            "the repair cannot forbid the duration correction it requests: {repair}"
        );
        assert!(
            repair.contains(
                "change legal shot durations or the number of shots while retaining every beat"
            ),
            "{repair}"
        );
        let contract_position = repair.find("# Output contract").expect("shared contract");
        let checklist_position = repair
            .rfind("# Final repair checklist")
            .expect("final checklist");
        assert!(contract_position < checklist_position, "{repair}");
        for finding in &findings {
            let finding_position = repair
                .rfind(&finding.to_string())
                .expect("finding is repeated in the final checklist");
            assert!(checklist_position < finding_position, "{repair}");
        }
        assert!(
            repair.trim_end().ends_with(
                "Return the whole corrected JSON object. Do not return the draft above unchanged."
            ),
            "the exact repair checklist must be the model's final instruction: {repair}"
        );
        for placeholder in [
            EXAMPLE_DURATION_PLACEHOLDER,
            EXAMPLE_CONDITIONING_PLACEHOLDER,
            CONTINUITY_FIELD_PLACEHOLDER,
            CONTINUITY_RULE_PLACEHOLDER,
            EXAMPLE_CONTINUITY_PLACEHOLDER,
            REFERENCE_RULE_PLACEHOLDER,
            LORA_FIELD_PLACEHOLDER,
            LORA_RULE_PLACEHOLDER,
        ] {
            assert!(
                !repair.contains(placeholder),
                "{placeholder} survived: {repair}"
            );
        }
    }

    /// sc-23405 AC2, the enforcement half. `requiredRoles` is checked exactly as before: a role
    /// bound only in `referenceRoles` COVERS its beat (it is on screen by the plan's own account),
    /// and a role bound nowhere is a finding that names it, which is what a repair round is handed.
    #[test]
    fn a_required_role_is_covered_by_a_reference_binding_and_named_when_it_is_bound_nowhere() {
        let brief: ProductionBrief = {
            let mut value = brief_json();
            value["requiredBeats"][0]["requiredRoles"] = json!(["courier", "red_parcel"]);
            serde_json::from_value(value).expect("brief parses")
        };
        let bound = |roles: Value, continuity: Value| {
            let mut shot = draft_shot("SH010", "arrival");
            shot["conditioning"] = json!({ "mode": "reference_to_video", "referenceRoles": roles });
            shot["continuityRoles"] = continuity;
            let draft: PlannerDraft = serde_json::from_value(json!({
                "shots": [shot, draft_shot("SH020", "delivery"), draft_shot("SH030", "discovery")]
            }))
            .expect("draft parses");
            draft_to_plan(&brief, &draft)
        };

        // Bound ONLY as conditioning: covered.
        let plan = bound(json!(["courier", "red_parcel"]), json!(["courier"]));
        assert!(
            role_coverage_findings(&brief, &plan, &pack()).is_empty(),
            "{:?}",
            messages(&role_coverage_findings(&brief, &plan, &pack()))
        );

        // The parcel bound nowhere at all: a finding that names it.
        let plan = bound(json!(["courier"]), json!(["courier"]));
        let findings = messages(&role_coverage_findings(&brief, &plan, &pack()));
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].contains("[SH010]") && findings[0].contains("red_parcel"),
            "{findings:?}"
        );
    }

    /// The repairer the real local planner turned out to be (sc-23406): it copies a literal array
    /// it is handed and re-derives anything it is merely described. Applies every
    /// `write [...] into SHnnn's continuityRoles[ AND its referenceRoles]` instruction in
    /// `findings` to `draft` — and nothing else — returning how many it found to apply.
    fn copy_arrays_from_findings(draft: &mut PlannerDraft, findings: &[PlanDiagnostic]) -> usize {
        let mut applied = 0;
        for finding in findings {
            let message = finding.to_string();
            let Some(start) = message.find("write [") else {
                continue;
            };
            let array = &message[start + "write ".len()..];
            let Some(end) = array.find(']') else {
                continue;
            };
            let roles: Vec<String> =
                serde_json::from_str(&array[..=end]).expect("the hint is a JSON array");
            let rest = &array[end + 1..];
            let Some(shot_at) = rest.find("into ") else {
                continue;
            };
            let shot_id: String = rest[shot_at + "into ".len()..]
                .chars()
                .take_while(char::is_ascii_alphanumeric)
                .collect();
            let Some(shot) = draft.shots.iter_mut().find(|shot| shot.id == shot_id) else {
                continue;
            };
            shot.continuity_roles.clone_from(&roles);
            if rest.contains("AND its referenceRoles") {
                shot.conditioning.reference_roles = roles;
            }
            applied += 1;
        }
        applied
    }

    /// 🔴 sc-23406. The violation the real planner produced on the default (turbo-offered) brief,
    /// twice: a beat that must show `courier, red_parcel, workbench_table` covered by a shot
    /// binding `workshop_location, courier, red_parcel` in BOTH lists — one character, one prop,
    /// one place, the location standing in for the second prop. The finding must hand back the
    /// exact array to write and both fields to write it into, because that is what the planner
    /// copies; a scripted repairer that does only that comes back clean. Drop the array from the
    /// hint and this draft is never repaired.
    #[test]
    fn the_real_planners_role_substitution_is_repaired_by_copying_the_findings_array() {
        let mut brief = brief();
        brief.required_beats[1].required_roles = vec![
            "courier".to_owned(),
            "red_parcel".to_owned(),
            "workbench_table".to_owned(),
        ];
        let mut delivery = draft_shot("SH020", "delivery");
        delivery["conditioning"] = json!({
            "mode": "reference_to_video",
            "referenceRoles": ["workshop_location", "courier", "red_parcel"]
        });
        delivery["continuityRoles"] = json!(["workshop_location", "courier", "red_parcel"]);
        let draft: PlannerDraft = serde_json::from_value(json!({
            "shots": [draft_shot("SH010", "arrival"), delivery, draft_shot("SH030", "discovery")]
        }))
        .expect("draft parses");

        let plan = draft_to_plan(&brief, &draft);
        let findings = role_coverage_findings(&brief, &plan, &pack());
        let texts = messages(&findings);
        assert_eq!(texts.len(), 1, "{texts:?}");

        // The repair FIRST: a planner that only copies arrays clears the finding in one round.
        let mut repaired_draft = draft.clone();
        assert_eq!(
            copy_arrays_from_findings(&mut repaired_draft, &findings),
            1,
            "the finding carries no array to copy: {}",
            texts[0]
        );
        let repaired = draft_to_plan(&brief, &repaired_draft);
        assert!(
            role_coverage_findings(&brief, &repaired, &pack()).is_empty(),
            "{:?}",
            messages(&role_coverage_findings(&brief, &repaired, &pack()))
        );
        assert_eq!(
            repaired_draft.shots[1].conditioning.reference_roles,
            [
                "courier",
                "red_parcel",
                "workbench_table",
                "workshop_location"
            ],
            "the reference binding follows the continuity list on a reference_to_video shot"
        );
        assert!(
            texts[0].contains(
                "write [\"courier\", \"red_parcel\", \"workbench_table\", \"workshop_location\"] \
                 into SH020's continuityRoles AND its referenceRoles"
            ) && texts[0].contains("missing workbench_table;")
                && texts[0].contains("describe workbench_table in its prompt"),
            "{}",
            texts[0]
        );

        // A shot that is not reference-conditioned is told about continuityRoles alone: telling it
        // to write referenceRoles would hand it the fault `validate_plan_structure` refuses.
        let mut text_only = draft_shot("SH020", "delivery");
        text_only["continuityRoles"] = json!(["courier"]);
        let draft: PlannerDraft = serde_json::from_value(json!({
            "shots": [draft_shot("SH010", "arrival"), text_only, draft_shot("SH030", "discovery")]
        }))
        .expect("draft parses");
        let texts = messages(&role_coverage_findings(
            &brief,
            &draft_to_plan(&brief, &draft),
            &pack(),
        ));
        assert_eq!(texts.len(), 1, "{texts:?}");
        assert!(
            texts[0].contains(
                "write [\"courier\", \"red_parcel\", \"workbench_table\"] into SH020's \
                 continuityRoles and describe"
            ) && !texts[0].contains("referenceRoles"),
            "{}",
            texts[0]
        );
    }

    /// 🔴 sc-23406 review. The repair hint is built against the PACK, so it can never instruct a
    /// binding the pack-aware validator then refuses:
    ///
    /// (a) a role the draft HALLUCINATED into `continuityRoles` is not echoed back into the array
    ///     to write — `validate_plan_against_pack` reports it "not in reference pack", so a
    ///     copy-only repairer handed it back could never converge;
    /// (b) an APPROVED `style` or `plate` role — legal in `continuityRoles`, refused as a
    ///     `reference_to_video` subject — is dropped from the array whenever the message names
    ///     `referenceRoles`.
    ///
    /// In both cases the copy-only repairer comes back clean in ONE round, on the role check AND on
    /// the pack validator.
    #[test]
    fn the_repair_hint_never_names_a_role_the_pack_validator_would_refuse() {
        let mut brief = brief();
        brief.required_beats[1].required_roles =
            vec!["courier".to_owned(), "red_parcel".to_owned()];
        // A delivery shot that shows only the courier, with `extra` declared in continuityRoles.
        let drafted = |conditioning: Value, extra: Value| -> PlannerDraft {
            let mut delivery = draft_shot("SH020", "delivery");
            delivery["conditioning"] = conditioning;
            delivery["continuityRoles"] = extra;
            serde_json::from_value(json!({
                "shots": [draft_shot("SH010", "arrival"), delivery, draft_shot("SH030", "discovery")]
            }))
            .expect("draft parses")
        };
        // Every role finding the pack validator raises about a role name or a bound kind.
        let pack_role_faults = |plan: &ProductionPlan| -> Vec<String> {
            messages(&crate::film_plan::validate_plan_against_pack(plan, &pack()))
                .into_iter()
                .filter(|message| {
                    message.contains("is not in reference pack")
                        || message.contains("may be BOUND")
                        || message.contains("is not approved for conditioning")
                })
                .collect()
        };

        let reference_to_video =
            json!({ "mode": "reference_to_video", "referenceRoles": ["courier"] });
        let text_to_video = json!({ "mode": "text_to_video" });
        for (case, conditioning, extra, tail) in [
            // (a) The hallucinated role, on a shot whose hint names continuityRoles ALONE — the
            // only place the pack-approval filter is the thing doing the work.
            (
                "a hallucinated role, continuity only",
                text_to_video.clone(),
                json!(["courier", "wolf"]),
                "into SH020's continuityRoles and describe",
            ),
            (
                "a hallucinated role, bound shot",
                reference_to_video.clone(),
                json!(["courier", "wolf"]),
                "into SH020's continuityRoles AND its referenceRoles",
            ),
            // (b) The approved style and plate, legal in continuityRoles and refused as subjects.
            (
                "an approved style and plate",
                reference_to_video.clone(),
                json!(["courier", "house_style", "workshop_plate"]),
                "into SH020's continuityRoles AND its referenceRoles",
            ),
        ] {
            let draft = drafted(conditioning, extra);
            let findings = role_coverage_findings(&brief, &draft_to_plan(&brief, &draft), &pack());
            let texts = messages(&findings);
            assert_eq!(texts.len(), 1, "{case}: {texts:?}");
            // The array to write is the beat's two roles and nothing else: `wolf` is not in the
            // pack, and a style or a plate is not a reference subject.
            assert!(
                texts[0].contains(&format!("write [\"courier\", \"red_parcel\"] {tail}")),
                "{case}: {}",
                texts[0]
            );
            // Only the ARRAY is an instruction; "binds only courier, wolf" is the description of
            // what is wrong and may still name it.
            let start = texts[0].find("write [").expect("the hint carries an array");
            let array = &texts[0][start..texts[0][start..].find(']').expect("closed") + start + 1];
            for refused in ["wolf", "house_style", "workshop_plate"] {
                assert!(
                    !array.contains(refused),
                    "{case}: the array to write names {refused:?}: {array}"
                );
            }

            // One round of a repairer that does nothing but copy the array clears both checks.
            let mut repaired_draft = draft.clone();
            assert_eq!(
                copy_arrays_from_findings(&mut repaired_draft, &findings),
                1,
                "{case}: the finding carries no array to copy: {}",
                texts[0]
            );
            let repaired = draft_to_plan(&brief, &repaired_draft);
            assert!(
                role_coverage_findings(&brief, &repaired, &pack()).is_empty(),
                "{case}: {:?}",
                messages(&role_coverage_findings(&brief, &repaired, &pack()))
            );
            assert!(
                pack_role_faults(&repaired).is_empty(),
                "{case}: {:?}",
                pack_role_faults(&repaired)
            );
        }

        // A beat that itself REQUIRES a style role is a different case: the role cannot be dropped
        // (coverage would never be satisfied) and cannot be bound, so the hint names
        // `continuityRoles` alone — which is where the coverage is read back — and converges there.
        let mut style_brief = brief.clone();
        style_brief.required_beats[1].required_roles =
            vec!["courier".to_owned(), "house_style".to_owned()];
        let draft = drafted(reference_to_video, json!(["courier"]));
        let findings =
            role_coverage_findings(&style_brief, &draft_to_plan(&style_brief, &draft), &pack());
        let texts = messages(&findings);
        assert_eq!(texts.len(), 1, "{texts:?}");
        assert!(
            texts[0].contains(
                "write [\"courier\", \"house_style\"] into SH020's continuityRoles and describe"
            ) && !texts[0].contains("referenceRoles"),
            "{}",
            texts[0]
        );
        let mut repaired_draft = draft.clone();
        assert_eq!(
            copy_arrays_from_findings(&mut repaired_draft, &findings),
            1,
            "{}",
            texts[0]
        );
        let repaired = draft_to_plan(&style_brief, &repaired_draft);
        assert!(
            role_coverage_findings(&style_brief, &repaired, &pack()).is_empty(),
            "{:?}",
            messages(&role_coverage_findings(&style_brief, &repaired, &pack()))
        );
        assert!(
            pack_role_faults(&repaired).is_empty(),
            "{:?}",
            pack_role_faults(&repaired)
        );
    }

    /// 🔴 sc-23406 review. The reference rule now tells the planner to bind "any other approved
    /// role in frame", and an approved `style` or `plate` is such a role — so both the envelope
    /// section and the output contract (the request AND every repair round) must say that a style
    /// or a plate is never bound in `referenceRoles`.
    #[test]
    fn the_prompts_say_a_style_or_plate_role_is_never_bound_in_reference_roles() {
        let brief = brief();
        let caps = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack());
        let envelope = caps.as_prompt_section();
        assert!(
            envelope.contains(
                "ONLY character, prop and location roles are bound this way: a style role or a \
                 plate role is NEVER listed in referenceRoles"
            ),
            "{envelope}"
        );
        let request = build_planner_request(&brief, &pack(), &caps);
        let repair = build_repair_request(&brief, &pack(), &caps, "{}", &[], 1, 2);
        for (name, text) in [("request", &request), ("repair", &repair)] {
            assert!(
                text.contains(
                    "- A style role and a plate role are NEVER listed in referenceRoles: only \
                     character, prop and location roles are bound as reference subjects."
                ),
                "the {name} prompt drops the style/plate clause: {text}"
            );
        }
        // An envelope with no reference conditioning is told nothing about binding at all.
        let bare = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        assert!(
            !plan_json_contract(&bare, &pack()).contains("NEVER listed in referenceRoles"),
            "{}",
            plan_json_contract(&bare, &pack())
        );
    }

    /// 🔴 sc-23406. Every place the planner reads a required-role list — the beat lines of the
    /// first round and of every repair round — hands it the literal array to copy, names
    /// `referenceRoles` only on an envelope that offers references, and nowhere describes the list
    /// as one character, one prop and one place. The accelerator rule sits with the other
    /// top-level-field rules, so the last rules before the worked example are the role rules; and
    /// the example binds exactly the roles it depicts.
    #[test]
    fn the_planner_is_handed_every_required_role_list_as_the_array_to_copy() {
        let mut brief = brief();
        brief.required_beats[1].required_roles = vec![
            "courier".to_owned(),
            "red_parcel".to_owned(),
            "workbench_table".to_owned(),
        ];
        let with_references = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx)
            .with_reference_partition(&reference_entry())
            .narrowed_to_pack(&pack());
        let request = build_planner_request(&brief, &pack(), &with_references);
        assert!(
            request.contains(
                "- delivery: The parcel is left on the bench. [this beat MUST show these roles — \
                 write [\"courier\", \"red_parcel\", \"workbench_table\"] into the covering \
                 shot's continuityRoles AND its referenceRoles, copied whole"
            ),
            "{request}"
        );
        let repair = build_repair_request(&brief, &pack(), &with_references, "{}", &[], 1, 2);
        assert!(
            repair.contains(
                "[MUST show — write [\"courier\", \"red_parcel\", \"workbench_table\"] into the \
                 covering shot's continuityRoles AND its referenceRoles"
            ),
            "{repair}"
        );

        let bare = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        let bare_request = build_planner_request(&brief, &pack(), &bare);
        assert!(
            bare_request.contains(
                "write [\"courier\", \"red_parcel\", \"workbench_table\"] into the covering \
                 shot's continuityRoles, copied whole"
            ) && !bare_request.contains("AND its referenceRoles"),
            "{bare_request}"
        );

        for text in [
            request.as_str(),
            repair.as_str(),
            &with_references.as_prompt_section(),
        ] {
            assert!(
                !text.contains("then the place")
                    && !text.contains("the location it happens in")
                    && !text.contains("the prop the beat turns on, the location"),
                "the one-of-each-kind template must not reach the planner: {text}"
            );
        }

        let turbo = with_references.with_installed_turbo_loras(&route_ordered_turbo_ids());
        let contract = plan_json_contract(&turbo, &pack());
        let lora_rule = contract
            .find("- loras is the top-level list")
            .expect("the accelerator rule is in the contract");
        let role_rule = contract
            .find("- Every shot needs at least one approved role")
            .expect("the role rule is in the contract");
        let reference_rule = contract
            .find("- An approved reference pack is available")
            .expect("the reference rule is in the contract");
        assert!(
            lora_rule < role_rule && role_rule < reference_rule,
            "the role rules are the last thing read before the example: {contract}"
        );
        assert!(
            contract.contains(
                "\"referenceRoles\": [\"mechanic\", \"customer\", \"brass_key\"] },\n  \
                 \"continuityRoles\": [\"mechanic\", \"customer\", \"brass_key\"]"
            ),
            "{contract}"
        );
    }

    /// sc-23406. An off-menu duration is corrected with the values on either side of it, which the
    /// planner copies, rather than with the whole menu alone, from which it interpolated.
    #[test]
    fn an_off_menu_duration_is_corrected_with_its_nearest_allowed_values() {
        let brief = brief();
        let mut draft = good_draft();
        draft.shots[1].target_duration_seconds = 6.0;
        let plan = draft_to_plan(&brief, &draft);
        let findings = messages(&validate_generated_plan(
            &brief,
            &draft,
            &plan,
            &pack(),
            None,
            Some((&single_entries(&model_entry()), ModelLane::Mlx)),
            None,
        ));
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("[SH020] targetDurationSeconds")
                    && finding.contains("write 5.875 or 14.375 instead")),
            "{findings:?}"
        );
    }

    #[test]
    fn the_planner_request_carries_the_beats_roles_and_envelope_and_never_an_unapproved_role() {
        let brief = brief();
        let pack = pack();
        let caps = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        let request = build_planner_request(&brief, &pack, &caps);
        for beat in &brief.required_beats {
            assert!(request.contains(&beat.id), "{request}");
            assert!(request.contains(beat.summary.trim()), "{request}");
        }
        assert!(request.contains("courier (character)"), "{request}");
        assert!(request.contains("workshop_plate (plate)"), "{request}");
        assert!(
            !request.contains("draft_look"),
            "an unapproved role must never be offered to the planner"
        );
        assert!(request.contains("576x320"), "{request}");
        assert!(
            request.contains(&plan_json_contract(&caps, &pack)),
            "{request}"
        );
        assert!(
            !request.contains(EXAMPLE_DURATION_PLACEHOLDER),
            "the contract template must never reach the planner unfilled: {request}"
        );
        assert!(request.contains("at most 8 shots"), "{request}");
    }

    #[test]
    fn a_repair_request_returns_every_finding_and_restates_the_beats() {
        let brief = brief();
        let caps = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        let findings = vec![
            PlanDiagnostic::plan("shots", "required beat \"delivery\" is covered by no shot"),
            PlanDiagnostic::shot("SH010", "targetDurationSeconds", "6s is not on the menu"),
        ];
        let request =
            build_repair_request(&brief, &pack(), &caps, "{\"shots\": []}", &findings, 1, 2);
        assert!(request.contains("Repair round 1 of 2"), "{request}");
        for finding in &findings {
            assert!(request.contains(&finding.to_string()), "{request}");
        }
        assert!(request.contains("arrival:"), "{request}");
        assert!(request.contains("{\"shots\": []}"), "{request}");
        assert!(
            request.contains(&plan_json_contract(&caps, &pack())),
            "{request}"
        );
    }

    /// AT4 (sc-22715): the contract's ONE worked example copies a duration off the envelope it is
    /// sent with. On the H3 entry that is still 5.1667; on a model with a different menu the
    /// example must show that model's first allowed value, and never H3's — which the contract's
    /// own "copy one of these EXACTLY" rule would forbid.
    #[test]
    fn the_contracts_worked_example_takes_its_duration_from_the_envelope() {
        let brief = brief();
        let h3 = capabilities_for(&brief.model, &model_entry(), ModelLane::Mlx);
        assert!(
            plan_json_contract(&h3, &pack()).contains("\"targetDurationSeconds\": 5.1667,"),
            "{}",
            plan_json_contract(&h3, &pack())
        );

        // An LTX-2.5-shaped envelope: whole-second clips, none of them 5.1667.
        let mut entry = model_entry();
        entry["limits"]["durations"] = json!([4, 6, 8, 10, 12, 15]);
        entry["defaults"]["duration"] = json!(6);
        let ltx = capabilities_for(&brief.model, &entry, ModelLane::Mlx);
        let contract = plan_json_contract(&ltx, &pack());
        assert!(
            contract.contains("\"targetDurationSeconds\": 4,"),
            "the example must be the first allowed duration: {contract}"
        );
        assert!(
            !contract.contains("5.1667"),
            "H3's clip length must not leak into another model's contract: {contract}"
        );
        let request = build_planner_request(&brief, &pack(), &ltx);
        assert!(
            request.contains("\"targetDurationSeconds\": 4,"),
            "{request}"
        );
        assert!(!request.contains("5.1667"), "{request}");
        let repair = build_repair_request(&brief, &pack(), &ltx, "{}", &[], 1, 1);
        assert!(repair.contains("\"targetDurationSeconds\": 4,"), "{repair}");

        // No declared menu at all: a plain round number the "any positive value" rule admits.
        entry["limits"]["durations"] = json!([]);
        let open = capabilities_for(&brief.model, &entry, ModelLane::Mlx);
        assert!(
            plan_json_contract(&open, &pack()).contains("\"targetDurationSeconds\": 6,"),
            "{}",
            plan_json_contract(&open, &pack())
        );
    }

    #[test]
    fn brief_findings_cover_versions_windows_beats_and_ceilings() {
        let mut value = brief_json();
        value["schemaVersion"] = json!(9);
        value["targetTotalSeconds"] = json!({ "min": 60.0, "max": 30.0 });
        value["requiredBeats"][1]["id"] = json!("arrival");
        value["maxShots"] = json!(2);
        let broken: ProductionBrief = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_brief(&broken));
        assert!(
            findings.iter().any(|m| m.contains("schema version 9")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("the window is empty")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|m| m.contains("duplicate beat id")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|m| m.contains("below the 3 required beats")),
            "{findings:?}"
        );
        assert!(validate_brief(&brief()).is_empty());
        // The planner's memory budget is a declared bound (sc-22715): a brief without one is
        // refused by name, and a non-positive one is refused by the shared limits rule.
        let mut value = brief_json();
        value["limits"]
            .as_object_mut()
            .unwrap()
            .remove("plannerMaxMemoryGb");
        let undeclared: ProductionBrief = serde_json::from_value(value).unwrap();
        let findings = messages(&validate_brief(&undeclared));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("limits.plannerMaxMemoryGb")),
            "{findings:?}"
        );
        let mut value = brief_json();
        value["limits"]["plannerMaxMemoryGb"] = json!(0);
        let zero: ProductionBrief = serde_json::from_value(value).unwrap();
        let findings = messages(&crate::film_plan::validate_limits(&zero.limits));
        assert!(
            findings
                .iter()
                .any(|m| m.contains("planner memory budget must be")),
            "{findings:?}"
        );
        // An unknown field in a brief is refused on parse, like every other document here.
        let mut value = brief_json();
        value["tone"] = json!("wistful");
        assert!(parse_brief(&value.to_string())
            .expect_err("unknown field refused")
            .contains("tone"));
    }
}
