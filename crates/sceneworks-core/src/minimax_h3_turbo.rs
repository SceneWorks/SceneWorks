pub use sceneworks_options::minimax_h3_turbo::*;
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
        let manifest = crate::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("valid embedded model catalog");
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

    #[test]
    fn shipped_turbo_recipes_declare_their_published_training_canvases() {
        let edge = |id| {
            turbo_recipe_for_lora_id(id)
                .unwrap_or_else(|| panic!("{id} recipe"))
                .training_short_edge
        };
        assert_eq!(edge("minimax_h3_turbo_4step_768p"), Some(768));
        assert_eq!(edge("minimax_h3_turbo_8step"), Some(544));
        assert_eq!(edge("minimax_h3_turbo_4step_v01"), Some(544));
        assert_eq!(edge("minimax_h3_ref2v_turbo_4step"), Some(544));
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
