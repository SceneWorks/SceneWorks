//! The MiniMax-H3 turbo sampling recipe (epic 17137, sc-18726) — the ONE resolver that turns a
//! selected step-distill adapter into the `(steps, video shift, audio shift)` triple a job actually
//! renders with.
//!
//! # Why this exists at all
//!
//! sc-18725 put four `lightx2v/Minimax-h3-Turbo` adapters in the catalog with `role: accelerator`,
//! and that marker is **weight-load only**: selecting one stacked a 692M-parameter residual and
//! still ran the base 50 steps at the base video shift 12.0 — slower than the base render (the LoRA
//! is pure overhead at 50 steps) *and* off-distribution, because a distilled checkpoint sampled on
//! the schedule it was distilled AWAY from is not the model anyone trained. sc-18729 measured what
//! the right recipe buys: **12.6 min against 2.42 h** at 1344x768 (11.57x), with a matched-geometry
//! floor of 7.05x.
//!
//! # Why the recipe is per FILE and lives in the catalog
//!
//! The four published files disagree with each other:
//!
//! | adapter | NFE | video shift | audio shift |
//! | --- | --- | --- | --- |
//! | `minimax_h3_turbo_4step_768p` | 4 | **6.0** | 3.0 |
//! | `minimax_h3_turbo_8step` | **8** | 12.0 | 3.0 |
//! | `minimax_h3_turbo_4step_v01` | 4 | 12.0 | 3.0 |
//! | `minimax_h3_ref2v_turbo_4step` | 4 | 12.0 | 3.0 |
//!
//! No pair of axes separates them, so no family-wide constant is right for more than one, and — the
//! part that rules out the SCAIL-2 approach — the files are **format-identical**, so a
//! `has_diff_patch_keys`-style sniff cannot tell them apart either. The discriminator that exists is
//! the catalog id, so the recipe is declared per catalog entry (`sampling` in
//! `builtin.loras.jsonc`) and resolved from the id here.
//!
//! Resolution reads the **embedded** builtin manifest rather than the payload's own `sampling`
//! block. A job payload is caller-supplied: reading a recipe out of it would let any HTTP client
//! name its own step count and shift under an adapter that was not distilled for them, bypassing
//! `steps_limit_error` entirely. Reading it from the id means the worst a forged payload can do is
//! name a real adapter, which is identical to selecting it.
//!
//! # The step-count contract — decided ONCE, here
//!
//! `steps` is the **number of model evaluations (NFE)**, everywhere: `advanced.steps`,
//! `limits.hardMinSteps`, `defaults.steps: 50`, the `sampling.steps` declared above, and the
//! engine's `req.steps`. `mlx-gen-minimax-h3` appends the terminal sigma = 0 grid point ITSELF
//! (`JointSchedule::with_shifts(evaluations + 1, ..)`), so there is **no +1 anywhere in
//! SceneWorks** and every number above is comparable to every other one without adjustment. sc-18726
//! flagged this as the design call that makes "every step count in the product off by one" if it is
//! made twice; [`crate::video_request::steps_limit_error`] is the gate that reads the same unit, and
//! `minimax_h3_turbo_steps_are_model_evaluations_not_grid_points` is the guard that pins it.
//!
//! # The audio shift is DECLARED, RECORDED and GUARDED — but not a request knob
//!
//! MiniMax-H3 denoises video and audio against two separate sigma schedules, and the audio one is a
//! real axis of the recipe, so it is declared per variant and recorded on the asset as
//! `effectiveAudioSchedulerShift`. It is not forwarded, because there is nothing to forward it to:
//! `GenerationRequest` carries one `scheduler_shift` and the engine applies it to VIDEO ONLY,
//! holding audio at its own `AUDIO_SIGMA_SHIFT` constant — deliberately, since every published
//! variant trains at 3.0 and applying one shift to both modalities is a known defect class.
//! [`ENGINE_AUDIO_SIGMA_SHIFT`] mirrors that constant and
//! `every_turbo_variant_declares_the_audio_shift_the_engine_is_fixed_at` fails the moment a declared
//! variant diverges from it — which turns "the engine cannot express this" from a silent wrong
//! render into a red test at the moment the catalog gains such a variant.

use std::sync::OnceLock;

use serde_json::Value;

use crate::video_request::is_minimax_h3_model;

/// The audio sigma shift `mlx-gen-minimax-h3` is hard-wired at (`AUDIO_SIGMA_SHIFT`), and therefore
/// the only audio shift a SceneWorks job can actually render at.
///
/// Mirrored here rather than imported because `sceneworks-core` builds on every platform and the
/// MLX engine crate is macOS-only — and because the mirror is the point: a declared variant that
/// disagrees with this number is a variant this app cannot render correctly, and the guard that
/// compares the two is the only thing that would say so.
pub const ENGINE_AUDIO_SIGMA_SHIFT: f32 = 3.0;

/// One published turbo adapter's sampling recipe, as declared on its catalog entry.
#[derive(Debug, Clone, PartialEq)]
pub struct TurboRecipe {
    /// The catalog LoRA id this recipe belongs to — the discriminator, since the files themselves
    /// are format-identical.
    pub lora_id: String,
    /// Display name, for the message a conflicting selection produces.
    pub name: String,
    /// Model evaluations (NFE). See the module header: no +1 lives on this side.
    pub steps: u32,
    /// The video flow-matching sigma shift → `GenerationRequest::scheduler_shift`.
    pub video_shift: f32,
    /// The audio sigma shift. Recorded on the asset; the engine is fixed at
    /// [`ENGINE_AUDIO_SIGMA_SHIFT`].
    pub audio_shift: f32,
    /// The SHORT EDGE of the canvas this file was distilled at (`sampling.trainingShortEdge`), when
    /// the catalog declares one — 768 for a 768p file, 544 for a 544p one.
    ///
    /// The shipped entries declare this from the upstream model-specs table. An undeclared canvas
    /// remains generic for user or future entries; it is never inferred from a filename.
    pub training_short_edge: Option<u32>,
}

static TURBO_RECIPES: OnceLock<Vec<TurboRecipe>> = OnceLock::new();

/// Every `family: minimax-h3` accelerator in the EMBEDDED builtin LoRA catalog that declares a
/// `sampling` block, in catalog order.
///
/// Derived from the shipped manifest rather than retyped, so adding a fifth published variant is a
/// manifest edit and nothing else — the studio control, the worker recipe and the guards below all
/// read this one list. Invalid embedded catalogs fail closed instead of supplying empty policy.
pub fn turbo_recipes() -> &'static [TurboRecipe] {
    TURBO_RECIPES.get_or_init(|| {
        parse_turbo_recipes(include_str!(
            "../../../config/manifests/builtin.loras.jsonc"
        ))
    })
}

pub fn parse_turbo_recipes(contents: &str) -> Vec<TurboRecipe> {
    let stripped = crate::jsonc::strip_jsonc_comments(contents);
    let Ok(manifest) = serde_json::from_str::<Value>(&stripped) else {
        return Vec::new();
    };
    turbo_recipes_from_manifest(&manifest)
}

pub fn turbo_recipes_from_manifest(manifest: &Value) -> Vec<TurboRecipe> {
    let Some(loras) = manifest.get("loras").and_then(Value::as_array) else {
        return Vec::new();
    };
    loras
        .iter()
        .filter(|lora| lora.get("family").and_then(Value::as_str) == Some("minimax-h3"))
        .filter(|lora| lora.get("role").and_then(Value::as_str) == Some("accelerator"))
        .filter_map(|lora| {
            let sampling = lora.get("sampling")?.as_object()?;
            let steps = u32::try_from(sampling.get("steps")?.as_u64()?).ok()?;
            let video_shift = sampling.get("schedulerShift")?.as_f64()? as f32;
            let audio_shift = sampling.get("audioSchedulerShift")?.as_f64()? as f32;
            let lora_id = lora.get("id")?.as_str()?.to_owned();
            let name = lora
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&lora_id)
                .to_owned();
            let training_short_edge = sampling
                .get("trainingShortEdge")
                .and_then(Value::as_u64)
                .and_then(|edge| u32::try_from(edge).ok())
                .filter(|edge| *edge > 0);
            (steps > 0 && video_shift > 0.0 && audio_shift > 0.0).then_some(TurboRecipe {
                lora_id,
                name,
                steps,
                video_shift,
                audio_shift,
                training_short_edge,
            })
        })
        .collect()
}

/// The recipe declared for catalog LoRA `id`, or `None` when the id names no turbo adapter.
pub fn turbo_recipe_for_lora_id(id: &str) -> Option<&'static TurboRecipe> {
    turbo_recipes().iter().find(|recipe| recipe.lora_id == id)
}

/// The turbo recipe a MiniMax-H3 request runs under, resolved from the job's own selected LoRAs.
///
/// `Ok(None)` is the base regime — the 50-step / shift-12.0 checkpoint default, byte-identical to
/// what shipped before this story, which is what makes "turbo off" a real state rather than a
/// different set of numbers.
///
/// This is the SINGLE resolver sc-18727 asks for: the Video Studio's turbo control and the generic
/// LoRA picker are two ways of putting the same catalog id in `request.loras`, and both land here.
/// Nothing else may derive the recipe, or the two entry points can disagree.
///
/// Two adapters carrying DIFFERENT recipes is an error, not a silent first-wins: they are
/// contradictory sampling regimes and picking one would render at a schedule the user can see they
/// did not choose. Two selections of the SAME recipe (possible via a project-scoped copy of a
/// builtin id) resolve normally.
///
/// Returns `Ok(None)` unconditionally for a non-MiniMax-H3 model, so a turbo file cross-selected
/// onto some other family loads as the plain residual it is rather than reshaping that family's
/// schedule.
pub fn resolve_turbo_recipe(
    model: &str,
    loras: &[Value],
) -> Result<Option<&'static TurboRecipe>, String> {
    if !is_minimax_h3_model(model) {
        return Ok(None);
    }
    let mut resolved: Option<&'static TurboRecipe> = None;
    for lora in loras {
        let Some(id) = lora.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(recipe) = turbo_recipe_for_lora_id(id) else {
            continue;
        };
        match resolved {
            None => resolved = Some(recipe),
            Some(current) if current.lora_id == recipe.lora_id => {}
            Some(current) if current.recipe_eq(recipe) => {}
            Some(current) => {
                return Err(format!(
                    "{model}: \"{}\" and \"{}\" are both step-distill accelerators and ask for \
                     different schedules ({} steps at video shift {} vs {} steps at video shift \
                     {}). A render has one schedule — select one of them.",
                    current.name,
                    recipe.name,
                    current.steps,
                    current.video_shift,
                    recipe.steps,
                    recipe.video_shift
                ));
            }
        }
    }
    Ok(resolved)
}

impl TurboRecipe {
    /// Whether two recipes ask for the same schedule (ignoring identity).
    ///
    /// This is also the planner's test for schedule PARITY across a mixed film's two partitions
    /// (sc-23406): one notion of "the same schedule", shared with the refusal above, so the
    /// envelope cannot offer a pair the worker would then call contradictory.
    pub fn recipe_eq(&self, other: &Self) -> bool {
        self.steps == other.steps
            && self.video_shift.to_bits() == other.video_shift.to_bits()
            && self.audio_shift.to_bits() == other.audio_shift.to_bits()
    }
}
