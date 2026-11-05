use super::*;

fn mage_planned(provider: &str, model_id: &str, tier: &str, steps: u32) -> Value {
    let slug = provider.replace('_', "-");
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": model_id,
                "tier": tier,
                "mode": if provider.contains("edit") { "edit_image" } else { "text_to_image" },
                "overlay": "none",
                "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
            },
            "backend": "candle",
            "loadShape": if tier == "bf16" { "deferred_materialization" } else { "eager_materialization" },
            "strategy": {
                "rung": "staged_residency",
                "engagedRungs": ["resident", "staged_residency"],
                "parameters": {}
            },
            "calibrationFingerprint":
                format!("mage-flow-cuda-{slug}-{tier}-shared-ladder-v3"),
            "fixture": format!("{slug}-candle-{tier}-768-seed{MAGE_SEED}-step{steps}"),
        }
    })
}

/// Every Mage member resolves an arm, a crossed `(provider, modelId)` pair is refused by name,
/// and a non-Mage provider yields `None` rather than being swept into this family.
#[test]
fn the_arm_is_resolved_from_the_provider_and_model_pair() {
    for arm in MAGE_ARMS {
        let request = mage_planned(arm.provider, arm.provider, "q4", arm.steps);
        assert_eq!(mage_arm(&request).unwrap(), Some(arm));
    }
    let crossed = mage_planned("mage_flow_edit", "mage_flow_edit_turbo", "q4", 2);
    assert!(mage_arm(&crossed)
        .unwrap_err()
        .contains("does not implement"));
    let foreign = mage_planned(QWEN_ID, QWEN_ID, "q4", 2);
    assert_eq!(mage_arm(&foreign).unwrap(), None);
}

/// The three gates that carry the shared refusal phrase all name every Mage member, and the
/// five-rung loader refuses one by name rather than trying to open it with a single root.
/// `adapterCapturableProviders` INTERSECTS those gates, so a member missing from any one of them
/// disappears from the derived capturable set.
#[test]
fn every_mage_member_is_named_by_every_dispatch_gate() {
    for arm in MAGE_ARMS {
        let request = mage_planned(arm.provider, arm.provider, "q4", arm.steps);
        assert_eq!(plain_execution_path(&request).unwrap(), arm.execution_path);
        assert_eq!(
            still_calibration_label(&request).unwrap(),
            arm.still_calibration
        );
        assert!(routes_to_five_rung_reference(&request).unwrap());
        assert_eq!(five_rung_evidence_story(arm.provider), "sc-22733");
        assert!(mage_registry_bypass(arm.provider).contains("load_mage_generator"));
    }
}

/// A plan row declaring the OTHER mode for a member is refused.
#[test]
fn a_plan_declaring_the_wrong_mode_for_a_member_is_refused() {
    for arm in MAGE_ARMS {
        let mut request = mage_planned(arm.provider, arm.provider, "q4", arm.steps);
        let wrong = if arm.edit {
            "text_to_image"
        } else {
            "edit_image"
        };
        request["planned"]["target"]["mode"] = json!(wrong);
        assert_eq!(
            validate_mage_mode(&request, arm).unwrap_err(),
            format!(
                "{} renders {:?}, but the plan declares mode {wrong:?}",
                arm.provider,
                mage_mode(arm)
            )
        );
    }
}

/// No two rows share an artifact, an env family, a fixture prefix or an execution path.
#[test]
fn every_mage_arm_row_is_unique_on_every_identifying_axis() {
    assert_eq!(MAGE_ARMS.len(), 6);
    for axis in [
        MAGE_ARMS.map(|arm| arm.provider),
        MAGE_ARMS.map(|arm| arm.repository),
        MAGE_ARMS.map(|arm| arm.repository_env),
        MAGE_ARMS.map(|arm| arm.fixture_prefix),
        MAGE_ARMS.map(|arm| arm.execution_path),
        MAGE_ARMS.map(|arm| arm.still_calibration),
    ] {
        let mut sorted = axis.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            6,
            "an identifying axis is shared by two Mage arms"
        );
    }
    for arm in MAGE_ARMS {
        assert_eq!(arm.edit, arm.provider.contains("edit"));
        let distilled = arm.provider.ends_with("turbo");
        assert_eq!(arm.steps, if distilled { 4 } else { 2 });
        assert_eq!(arm.guidance, if distilled { 1.0 } else { 5.0 });
    }
}

/// The fixture binds member, tier, edge, seed and the recipe's step count.
#[test]
fn the_fixture_binds_the_member_the_tier_and_the_step_count() {
    let arm = MAGE_ARMS[5];
    let good = mage_planned(arm.provider, arm.provider, "bf16", arm.steps);
    validate_mage_fixture(&good, arm, "bf16").unwrap();
    assert!(validate_mage_fixture(&good, arm, "q4")
        .unwrap_err()
        .contains("must start with"));
    let wrong_steps = mage_planned(arm.provider, arm.provider, "bf16", 2);
    assert!(validate_mage_fixture(&wrong_steps, arm, "bf16")
        .unwrap_err()
        .contains("4-step calibration request"));
    let mut wrong_seed = good.clone();
    wrong_seed["planned"]["fixture"] =
        json!(format!("{}-bf16-768-seed16402-step4", arm.fixture_prefix));
    assert!(validate_mage_fixture(&wrong_seed, arm, "bf16")
        .unwrap_err()
        .contains("does not match the seed"));
}

/// The composed `LoadSpec` is the shape the WORKER loads at the candle anchor rung: `Sequential`
/// (the `staged_residency` composition), the PLAN's per-tier load shape — deferred on bf16
/// (the Applied BTR declaration), eager on q4/q8 (the refused one; see [`mage_load_spec`]) —
/// the planned tier's quant on the spec (`resolved_quant` reads it directly), and both shared
/// components staged from the components snapshot's own tier directory.
///
/// The shape is asserted against the committed plan's 18 `mage_flow*:*:candle` rows, read
/// through the same `mage_planned_load_shape` a capture uses, rather than a literal: the
/// worker's `memory_route_registry` test pins those rows to the registry's own evaluation over
/// the real manifest entries, so this closes the chain arm == plan == registry. The per-tier
/// split is ALSO asserted literally, so a plan regenerated all-eager or all-deferred cannot
/// carry this test along with it.
#[test]
fn the_load_spec_is_the_shape_the_worker_loads() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let anchors = plan["anchors"].as_object().expect("anchors object");
    for tier in ["bf16", "q4", "q8"] {
        let root = PathBuf::from("/hub/models--SceneWorks--Mage-Flow/snapshots/abc").join(tier);
        let components =
            PathBuf::from("/hub/models--SceneWorks--Mage-Flow-Components-mlx/snapshots/def");
        let expected_shape = if tier == "bf16" {
            LoadShape::DeferredMaterialization
        } else {
            LoadShape::EagerMaterialization
        };
        for arm in MAGE_ARMS {
            let key = format!("{}:{tier}:candle", arm.provider);
            let request = json!({ "planned": { "loadShape": anchors[&key]["loadShape"].clone() } });
            let load_shape =
                mage_planned_load_shape(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
            assert_eq!(
                load_shape, expected_shape,
                "{key}: the plan must bind the shape the worker loads on this tier"
            );
            let spec = mage_load_spec(tier, load_shape, root.clone(), &components).unwrap();
            assert_eq!(spec.weights, WeightsSource::Dir(root.clone()));
            assert_eq!(spec.offload_policy, OffloadPolicy::Sequential);
            assert_eq!(
                spec.load_shape, load_shape,
                "{key}: the arm must load the shape the plan (and the worker) binds"
            );
        }
        let spec = mage_load_spec(tier, expected_shape, root.clone(), &components).unwrap();
        assert_eq!(spec.quantize, numeric_tier(tier).unwrap().quant);
        for component in [
            protocol::MAGE_COMPONENT_TEXT_ENCODER,
            protocol::MAGE_COMPONENT_VAE,
        ] {
            let Some(WeightsSource::Dir(dir)) = spec.components.get(component) else {
                panic!("the {component} component was not staged");
            };
            assert_eq!(dir, &components.join(tier).join(component));
        }
    }
}

/// Every candle Mage plan row names the per-(member, tier) production identity the loaded
/// generator publishes (inference PR 953's `-v3` table), the 18 identities are distinct, no row
/// carries the retired tier-free `-v2` string or an MLX identity, and the pre-load check
/// refuses a row that does — before any env or weights are touched.
#[test]
fn every_planned_mage_candle_row_names_the_production_identity_and_is_checked_before_the_load() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let anchors = plan["anchors"].as_object().expect("anchors object");
    let mut identities = std::collections::BTreeSet::new();
    let mut seen = std::collections::BTreeSet::new();
    for arm in MAGE_ARMS {
        for tier in ["bf16", "q4", "q8"] {
            let key = format!("{}:{tier}:candle", arm.provider);
            let row = &anchors[&key];
            let expected = mage_calibration_fingerprint(arm, tier);
            assert_eq!(
                row["calibrationFingerprint"].as_str(),
                Some(expected.as_str()),
                "{key}"
            );
            assert!(expected.ends_with("-shared-ladder-v3"), "{expected}");
            assert!(expected.contains(&format!("-{tier}-")), "{expected}");
            assert!(!expected.contains("-mlx-"), "{expected}");
            assert!(
                identities.insert(expected),
                "{key}: identity shared with another cell"
            );
            seen.insert(key.clone());
            // The checked-in row passes the pre-load check; a retired or foreign string fails
            // it by name.
            let request = json!({ "planned": {
                "target": { "provider": arm.provider, "modelId": arm.provider, "tier": tier },
                "calibrationFingerprint": row["calibrationFingerprint"].clone(),
            }});
            validate_mage_plan_identity(&request, arm, tier).unwrap();
            assert!(
                mage_planned_load_shape(&json!({ "planned": { "loadShape": "streamed" } }))
                    .unwrap_err()
                    .contains("unsupported planned.loadShape")
            );
            for stale in [
                format!(
                    "mage-flow-cuda-shared-ladder-provider-abi-v2-{}",
                    arm.provider.replace('_', "-")
                ),
                format!(
                    "mage-flow-{}-{tier}-mlx-shared-ladder-v1",
                    arm.provider.replace('_', "-")
                ),
            ] {
                let mut wrong = request.clone();
                wrong["planned"]["calibrationFingerprint"] = json!(stale);
                let error = validate_mage_plan_identity(&wrong, arm, tier).unwrap_err();
                assert!(error.contains("calibration mismatch"), "{key}: {error}");
                assert!(error.contains(&stale), "{key}: {error}");
            }
        }
    }
    let expected: std::collections::BTreeSet<String> = MAGE_ARMS
        .iter()
        .flat_map(|arm| {
            ["bf16", "q4", "q8"]
                .iter()
                .map(move |tier| format!("{}:{tier}:candle", arm.provider))
        })
        .collect();
    assert_eq!(seen, expected);
    assert_eq!(identities.len(), 18);
}

/// The edit members condition on exactly one reference, the text-to-image members on none, and
/// only the undistilled members carry a negative prompt (the distilled ones run CFG off).
#[test]
fn only_the_edit_members_carry_a_reference() {
    for arm in MAGE_ARMS {
        let capabilities = mage_capabilities(arm);
        let request = mage_generation_request(arm, &capabilities, 768, 768);
        assert_eq!(request.conditioning.len(), usize::from(arm.edit));
        if arm.edit {
            let Conditioning::Reference { image, strength } = &request.conditioning[0] else {
                panic!("{} must condition on a Reference", arm.provider);
            };
            assert_eq!((image.width, image.height), (768, 768));
            assert_eq!(
                *strength, None,
                "Mage's edit_references never reads a strength lever"
            );
        }
        assert_eq!(request.steps, Some(arm.steps));
        // Both knobs follow the loaded descriptor's capability, never the table's scale
        // (sc-22738) — see `a_distilled_mage_route_sends_no_guidance_and_no_negative_prompt`.
        assert_eq!(request.guidance.is_some(), capabilities.supports_guidance);
        assert_eq!(
            request.negative_prompt.is_some(),
            capabilities.supports_negative_prompt
        );
        assert_eq!(request.seed, Some(MAGE_SEED));
    }
}

/// The capability set the pinned engine publishes for one member: `candle-gen-mage`
/// `src/lib.rs:132-148` takes both flags per generator and `:939-940` derives the edit ones as
/// `!matches!(variant, MageEditVariant::EditTurbo)` — i.e. the two distilled `*_turbo` ids
/// advertise neither knob, exactly as the MLX twin does.
fn mage_capabilities(arm: MageArm) -> Capabilities {
    let guided = !arm.provider.ends_with("turbo");
    Capabilities {
        supports_guidance: guided,
        supports_negative_prompt: guided,
        ..Capabilities::default()
    }
}

/// sc-22738. A DISTILLED MAGE ROUTE SENDS NO GUIDANCE AND NO NEGATIVE PROMPT.
///
/// The MLX twin of this arm had all six `mage_flow*_turbo` cells refused at REQUEST time with
/// `unsupported: mage_flow_turbo: guidance is not supported` — the shared floor
/// (`gen-core/src/generator.rs:3216-3221`) rejects a `guidance` on a provider whose
/// `supports_guidance` is false. This lane built the same request off the same table, so it
/// carried the same defect; both now read the loaded descriptor, as production does
/// (`image_jobs/base.rs` `resolve_guidance` :4632, `resolve_negative_prompt` :4870).
///
/// MUTATIONS THIS PINS. Restoring `guidance: Some(arm.guidance)` reds the first assertion;
/// restoring `negative_prompt: (arm.guidance > 1.0).then(…)`, which reads the TABLE, reds the
/// second; withholding both unconditionally reds the third and fourth.
#[test]
fn a_distilled_mage_route_sends_no_guidance_and_no_negative_prompt() {
    let distilled = Capabilities::default();
    let guided = Capabilities {
        supports_guidance: true,
        supports_negative_prompt: true,
        ..Capabilities::default()
    };
    for arm in MAGE_ARMS {
        let refused = mage_generation_request(arm, &distilled, 768, 768);
        assert_eq!(
            refused.guidance, None,
            "{} must send no guidance to a provider that does not accept one",
            arm.provider
        );
        assert_eq!(
            refused.negative_prompt, None,
            "{} must send no negative prompt to a provider that does not accept one",
            arm.provider
        );
        let accepted = mage_generation_request(arm, &guided, 768, 768);
        assert_eq!(
            accepted.guidance,
            Some(arm.guidance),
            "{} must send its declared scale where the provider accepts one",
            arm.provider
        );
        assert!(
            accepted.negative_prompt.is_some(),
            "{} must send its negative prompt where the provider accepts one",
            arm.provider
        );
    }
    let distilled_ids = MAGE_ARMS
        .into_iter()
        .filter(|arm| arm.guidance == 1.0)
        .map(|arm| arm.provider)
        .collect::<Vec<_>>();
    assert_eq!(distilled_ids, ["mage_flow_turbo", "mage_flow_edit_turbo"]);
}
