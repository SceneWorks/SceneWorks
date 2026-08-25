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
}

static TURBO_RECIPES: OnceLock<Vec<TurboRecipe>> = OnceLock::new();

/// Every `family: minimax-h3` accelerator in the EMBEDDED builtin LoRA catalog that declares a
/// `sampling` block, in catalog order.
///
/// Derived from the shipped manifest rather than retyped, so adding a fifth published variant is a
/// manifest edit and nothing else — the studio control, the worker recipe and the guards below all
/// read this one list. A malformed manifest yields an EMPTY list rather than a panic: the catalog is
/// compile-time embedded and validated by the tests below, so an empty list at runtime can only mean
/// a build that shipped a broken manifest, and degrading to "no turbo available" beats taking the
/// worker down on a video job.
pub fn turbo_recipes() -> &'static [TurboRecipe] {
    TURBO_RECIPES.get_or_init(|| {
        let contents = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.loras.jsonc")
            .map(|(_, contents)| *contents)
            .unwrap_or("");
        parse_turbo_recipes(contents)
    })
}

/// Parse the turbo recipes out of a `builtin.loras.jsonc` body. Split out so the tests can drive it
/// on synthetic catalogs as well as the shipped one.
fn parse_turbo_recipes(contents: &str) -> Vec<TurboRecipe> {
    let stripped = crate::jsonc::strip_jsonc_comments(contents);
    let Ok(manifest) = serde_json::from_str::<Value>(&stripped) else {
        return Vec::new();
    };
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
            (steps > 0 && video_shift > 0.0 && audio_shift > 0.0).then_some(TurboRecipe {
                lora_id,
                name,
                steps,
                video_shift,
                audio_shift,
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
    fn recipe_eq(&self, other: &Self) -> bool {
        self.steps == other.steps
            && self.video_shift.to_bits() == other.video_shift.to_bits()
            && self.audio_shift.to_bits() == other.audio_shift.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_request::steps_limit_error;
    use serde_json::json;

    fn lora(id: &str) -> Value {
        json!({ "id": id })
    }

    /// The shipped catalog carries EXACTLY the four published turbo adapters, each with the recipe
    /// upstream's model-specs table publishes.
    ///
    /// Every triple is spelled out rather than derived, because the whole point of the per-entry
    /// declaration is that the four disagree: an implementation that returned one constant for all
    /// of them, or that read the shift off the family, fails at least two arms here.
    #[test]
    fn the_shipped_catalog_declares_every_published_turbo_variant() {
        let recipes = turbo_recipes();
        let actual: Vec<(&str, u32, f32, f32)> = recipes
            .iter()
            .map(|recipe| {
                (
                    recipe.lora_id.as_str(),
                    recipe.steps,
                    recipe.video_shift,
                    recipe.audio_shift,
                )
            })
            .collect();
        assert_eq!(
            actual,
            vec![
                ("minimax_h3_turbo_4step_768p", 4, 6.0, 3.0),
                ("minimax_h3_turbo_8step", 8, 12.0, 3.0),
                ("minimax_h3_turbo_4step_v01", 4, 12.0, 3.0),
                ("minimax_h3_ref2v_turbo_4step", 4, 12.0, 3.0),
            ],
            "the shipped turbo recipes"
        );
        // The set is not uniform on EITHER axis — the property that makes a per-entry declaration
        // necessary rather than decorative. Asserted so a future edit that flattens the catalog to
        // one recipe cannot pass the arm above by rewriting it.
        assert!(
            recipes.iter().any(|r| r.steps == 4) && recipes.iter().any(|r| r.steps == 8),
            "the published set spans two step counts"
        );
        assert!(
            recipes.iter().any(|r| r.video_shift == 6.0)
                && recipes.iter().any(|r| r.video_shift == 12.0),
            "the published set spans two video shifts"
        );
    }

    /// 🔴 The declared audio shift must equal the ONE the engine can render at.
    ///
    /// `mlx-gen-minimax-h3` reads `req.scheduler_shift` as the VIDEO shift and holds audio at its own
    /// `AUDIO_SIGMA_SHIFT = 3.0` with no request-level override. So a catalog variant declaring any
    /// other audio shift would be recorded on the asset as what the user asked for and rendered at
    /// 3.0 anyway — a silent divergence with no shape and no error. This is the guard that makes
    /// that loud: it fails at the moment such a variant is added, which is the moment the engine
    /// needs a second shift knob.
    #[test]
    fn every_turbo_variant_declares_the_audio_shift_the_engine_is_fixed_at() {
        for recipe in turbo_recipes() {
            assert_eq!(
                recipe.audio_shift, ENGINE_AUDIO_SIGMA_SHIFT,
                "{}: the engine has no audio-shift request knob, so a variant declaring {} would \
                 render at {ENGINE_AUDIO_SIGMA_SHIFT} regardless. Adding one needs an engine-side \
                 audio shift on GenerationRequest first.",
                recipe.lora_id, recipe.audio_shift
            );
        }
    }

    /// Every recipe's step count must clear the paired model's own step gate, or selecting the
    /// adapter would build a job `steps_limit_error` refuses — a control that 400s the render it
    /// exists to start.
    ///
    /// Reads the SHIPPED model manifest (not a fixture) and the SAME gate the API and worker call,
    /// so a manifest edit that tightens `limits.hardMinSteps` above 4, or that pins
    /// `limits.steps: [50]`, goes red here rather than at enqueue.
    #[test]
    fn every_turbo_recipe_step_count_clears_the_model_step_gate() {
        let contents = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("builtin.models.jsonc is embedded");
        let stripped = crate::jsonc::strip_jsonc_comments(contents);
        let manifest: Value = serde_json::from_str(&stripped).expect("the model catalog parses");
        let models = manifest["models"].as_array().expect("a models array");
        for model_id in ["minimax_h3", "minimax_h3_ref"] {
            let entry = models
                .iter()
                .find(|model| model["id"].as_str() == Some(model_id))
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("{model_id} is in the shipped catalog"));
            for recipe in turbo_recipes() {
                assert_eq!(
                    steps_limit_error(model_id, recipe.steps, entry),
                    None,
                    "{model_id} refuses {}'s {} steps",
                    recipe.lora_id,
                    recipe.steps
                );
            }
        }
    }

    /// 🔴 sc-18726's "decide the +1 once" call, pinned as a value rather than a comment.
    ///
    /// `steps` is model EVALUATIONS on this side of the seam. The engine builds
    /// `evaluations + 1` sigma grid points itself, so a SceneWorks-side +1 would render 5 and 9
    /// evaluations for the "4-step" and "8-step" adapters. Asserting the declared numbers are the
    /// PUBLISHED NFE — not NFE+1 — is what keeps a future "fix" from adding the engine's increment a
    /// second time.
    #[test]
    fn minimax_h3_turbo_steps_are_model_evaluations_not_grid_points() {
        let four =
            turbo_recipe_for_lora_id("minimax_h3_turbo_4step_768p").expect("the 768p recipe");
        let eight = turbo_recipe_for_lora_id("minimax_h3_turbo_8step").expect("the 8-step recipe");
        assert_eq!(four.steps, 4, "the file upstream calls 4-step runs 4 NFE");
        assert_eq!(eight.steps, 8, "the file upstream calls 8-step runs 8 NFE");
    }

    /// The resolver: no accelerator ⇒ the base regime, one ⇒ its own recipe, and the recipe follows
    /// the ID rather than the position in the list.
    #[test]
    fn resolves_the_selected_variants_own_recipe() {
        assert_eq!(resolve_turbo_recipe("minimax_h3", &[]), Ok(None));
        assert_eq!(
            resolve_turbo_recipe("minimax_h3", &[lora("some_style_lora")]),
            Ok(None),
            "a plain LoRA leaves the base regime alone"
        );
        // A NON-DEFAULT pair on purpose: 4 differs from the model's `defaults.steps: 50` and 6.0
        // from the engine's own 12.0, so this cannot pass against an implementation that returns
        // the defaults.
        let turbo = resolve_turbo_recipe("minimax_h3", &[lora("minimax_h3_turbo_4step_768p")])
            .expect("one accelerator resolves")
            .expect("a recipe");
        assert_eq!((turbo.steps, turbo.video_shift), (4, 6.0));
        // The 8-step file, selected SECOND, must still win its own recipe — the resolver keys on the
        // id, not on the order or the count.
        let eight = resolve_turbo_recipe(
            "minimax_h3",
            &[lora("some_style_lora"), lora("minimax_h3_turbo_8step")],
        )
        .expect("resolves")
        .expect("a recipe");
        assert_eq!((eight.steps, eight.video_shift), (8, 12.0));
    }

    /// Two accelerators asking for different schedules is refused by name, not silently first-wins.
    #[test]
    fn two_conflicting_accelerators_are_refused_by_name() {
        let error = resolve_turbo_recipe(
            "minimax_h3",
            &[
                lora("minimax_h3_turbo_4step_768p"),
                lora("minimax_h3_turbo_8step"),
            ],
        )
        .expect_err("two different schedules cannot both govern one render");
        assert!(
            error.contains("MiniMax-H3 Turbo (4-step, 768p)")
                && error.contains("MiniMax-H3 Turbo (8-step)"),
            "the refusal names both adapters: {error}"
        );
        // The same id twice is not a conflict.
        assert!(resolve_turbo_recipe(
            "minimax_h3",
            &[
                lora("minimax_h3_turbo_8step"),
                lora("minimax_h3_turbo_8step"),
            ],
        )
        .is_ok());
        // Nor are two DISTINCT adapters that happen to declare the same schedule — the 544p 4-step
        // pair, which is exactly the shape a Ref2VA job selecting both would produce.
        assert!(resolve_turbo_recipe(
            "minimax_h3_ref",
            &[
                lora("minimax_h3_turbo_4step_v01"),
                lora("minimax_h3_ref2v_turbo_4step"),
            ],
        )
        .is_ok());
    }

    /// A turbo file cross-selected onto some OTHER family must not reshape that family's schedule.
    #[test]
    fn a_turbo_adapter_on_a_non_minimax_model_changes_nothing() {
        assert_eq!(
            resolve_turbo_recipe("wan_2_2_t2v_14b", &[lora("minimax_h3_turbo_4step_768p")]),
            Ok(None)
        );
    }

    /// The parser reads the declared block and drops a malformed one rather than inventing values —
    /// the difference between "this adapter has no recipe" (loads as a plain residual) and "this
    /// adapter renders at a number nobody declared".
    #[test]
    fn a_malformed_or_absent_sampling_block_yields_no_recipe() {
        let catalog = r#"{
          "schemaVersion": 1,
          "loras": [
            { "id": "no_block", "family": "minimax-h3", "role": "accelerator" },
            { "id": "zero_steps", "family": "minimax-h3", "role": "accelerator",
              "sampling": { "steps": 0, "schedulerShift": 6.0, "audioSchedulerShift": 3.0 } },
            { "id": "no_shift", "family": "minimax-h3", "role": "accelerator",
              "sampling": { "steps": 4, "audioSchedulerShift": 3.0 } },
            { "id": "not_an_accelerator", "family": "minimax-h3",
              "sampling": { "steps": 4, "schedulerShift": 6.0, "audioSchedulerShift": 3.0 } },
            { "id": "other_family", "family": "wan-video", "role": "accelerator",
              "sampling": { "steps": 4, "schedulerShift": 6.0, "audioSchedulerShift": 3.0 } },
            { "id": "good", "family": "minimax-h3", "role": "accelerator",
              "sampling": { "steps": 8, "schedulerShift": 12.0, "audioSchedulerShift": 3.0 } }
          ]
        }"#;
        let recipes = parse_turbo_recipes(catalog);
        assert_eq!(
            recipes
                .iter()
                .map(|r| r.lora_id.as_str())
                .collect::<Vec<_>>(),
            vec!["good"]
        );
        assert!(parse_turbo_recipes("not json at all").is_empty());
    }
}
