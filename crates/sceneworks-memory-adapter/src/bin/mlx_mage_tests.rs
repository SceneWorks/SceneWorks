use super::*;

fn mage_planned(provider: &str, model_id: &str, tier: &str, steps: u32) -> Value {
    let mode = if provider.contains("edit") {
        "edit_image"
    } else {
        "text_to_image"
    };
    let slug = provider.replace('_', "-");
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": model_id,
                "tier": tier,
                "mode": mode,
                "overlay": "none",
                "geometry": { "width": MAGE_EDGE, "height": MAGE_EDGE, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": "deferred_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": format!("mage-flow-{slug}-{tier}-mlx-shared-ladder-v1"),
            "fixture": format!("{slug}-mlx-{tier}-{}-seed{MAGE_SEED}-step{steps}", MAGE_EDGE),
        }
    })
}

/// The capability set the PINNED engine publishes for one member, restated so the request
/// builder can be driven without weights: `mlx-gen-mage` `src/model.rs:866-868` sets
/// `supports_negative_prompt` and `supports_guidance` alike to `!variant.is_distilled()`, and
/// `is_distilled` is exactly `Turbo | EditTurbo` (`model.rs:747-749`) — the two `*_turbo`
/// catalog ids. Every other capability stays at its conservative default; nothing here reads
/// one.
fn mage_capabilities(arm: MageArm) -> Capabilities {
    let guided = !arm.provider.ends_with("turbo");
    Capabilities {
        supports_guidance: guided,
        supports_negative_prompt: guided,
        ..Capabilities::default()
    }
}

/// sc-22738. THE RECEIPT CARRIES THE OVERLAY SCENARIO THE PROTOCOL SETTLES.
///
/// All eighteen `mage_flow*:*:mlx` cells rendered and were then refused with "provider fragment
/// is missing the required overlay scenario": [`protocol::settle_plain_overlay_scenario`]
/// REPLACES an existing `overlay` entry and refuses a fragment that has none, and this arm's
/// scenario list was the only one in this binary that declared no placeholder. Run over the
/// real list, weights-free — the render is what made the omission expensive to find.
///
/// MUTATIONS THIS PINS. Deleting the `overlay` entry from [`mage_scenarios`] reds the first
/// assertion with the production refusal verbatim. Authoring the verdict in the list instead of
/// letting the settler derive it reds the second, whose reason text only the settler writes.
/// Relaxing the settler so a material overlay is excused reds the third.
#[test]
fn the_mage_receipt_settles_the_overlay_scenario_from_the_declared_target() {
    for arm in MAGE_ARMS {
        let request = mage_planned(arm.provider, arm.provider, "bf16", arm.steps);
        let mut fragment = json!({ "scenarios": mage_scenarios(1_024) });
        protocol::settle_plain_overlay_scenario(&request, &mut fragment, arm.execution_path)
            .unwrap_or_else(|error| {
                panic!("{} must settle its overlay scenario: {error}", arm.provider)
            });
        let overlay = fragment["scenarios"]
            .as_array()
            .expect("the settled fragment still carries a scenario array")
            .iter()
            .find(|scenario| scenario["name"] == "overlay")
            .expect("the settled overlay scenario")
            .clone();
        assert_eq!(overlay["result"], "not_applicable");
        let reason = overlay["reason"].as_str().expect("a settled reason");
        assert!(
            reason.contains(arm.execution_path)
                && reason.contains("no second resident network to measure"),
            "the verdict must be the settler's, derived from the plan row, not authored in \
                 the list: {reason}"
        );
        // A material overlay still fails closed rather than acquiring `not_applicable`.
        let mut overlaid = request.clone();
        overlaid["planned"]["target"]["overlay"] = json!("lora");
        let mut fragment = json!({ "scenarios": mage_scenarios(1_024) });
        assert!(
            protocol::settle_plain_overlay_scenario(&overlaid, &mut fragment, arm.execution_path)
                .is_err(),
            "{} must refuse a declared LoRA overlay it does not execute",
            arm.provider
        );
    }
}

/// sc-22738. A DISTILLED MAGE ROUTE SENDS NO GUIDANCE AND NO NEGATIVE PROMPT.
///
/// The six `mage_flow*_turbo:*:mlx` cells were refused at REQUEST time — before any render —
/// with `unsupported: mage_flow_turbo: guidance is not supported`. The shared engine floor
/// (`gen-core/src/generator.rs:3216-3221`) rejects a `guidance` on a provider whose
/// `Capabilities::supports_guidance` is false, and `mlx-gen-mage` `src/model.rs:866-868`
/// publishes both that flag and `supports_negative_prompt` as `!variant.is_distilled()`.
/// Production asks the loaded descriptor the same question (`image_jobs/base.rs`
/// `resolve_guidance` :4632, `resolve_negative_prompt` :4870, both read by `generate_stream`
/// :8609/:8659) and sends neither knob; this arm now does the same.
///
/// MUTATIONS THIS PINS. Restoring `guidance: Some(arm.guidance)` — passing the scale on a
/// distilled route — reds the first assertion under the engine's own refusal condition.
/// Restoring `negative_prompt: (arm.guidance > 1.0).then(…)`, which reads the TABLE rather than
/// the descriptor, reds the second. Withholding both unconditionally reds the third and fourth,
/// so the fix cannot degenerate into dropping the knobs everywhere.
#[test]
fn a_distilled_mage_route_sends_no_guidance_and_no_negative_prompt() {
    let distilled = Capabilities::default();
    let guided = Capabilities {
        supports_guidance: true,
        supports_negative_prompt: true,
        ..Capabilities::default()
    };
    for arm in MAGE_ARMS {
        let refused = mage_request(arm, &distilled, MAGE_EDGE, MAGE_EDGE, MAGE_SEED);
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
        // The knobs are withheld on CAPABILITY, never on the member being an edit route or on
        // the fixed prompt — both requests below are the same member.
        let accepted = mage_request(arm, &guided, MAGE_EDGE, MAGE_EDGE, MAGE_SEED);
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
        assert_eq!(refused.steps, accepted.steps);
        assert_eq!(refused.seed, accepted.seed);
    }
    // The engine's rule is `supports_guidance: !variant.is_distilled()`, and `is_distilled` is
    // exactly `Turbo | EditTurbo` (`model.rs:747-749`) — so the two members production withholds
    // the knobs from are the two `*_turbo` ids. Bound here so a seventh member cannot be added
    // on the guided default by accident.
    let distilled_ids = MAGE_ARMS
        .into_iter()
        .filter(|arm| arm.guidance == 1.0)
        .map(|arm| arm.provider)
        .collect::<Vec<_>>();
    assert_eq!(distilled_ids, ["mage_flow_turbo", "mage_flow_edit_turbo"]);
    for arm in MAGE_ARMS {
        assert_eq!(
            arm.provider.ends_with("turbo"),
            arm.steps == 4,
            "{} disagrees with the engine's distilled recipe",
            arm.provider
        );
    }
}

/// Every registered Mage id reaches the arm, and a foreign one is refused BY NAME rather than
/// misrouted into it (the sc-18104 defect class).
#[test]
fn run_dispatches_every_mage_id_and_still_refuses_a_foreign_one() {
    for arm in MAGE_ARMS {
        let error = run(&mage_planned(arm.provider, arm.provider, "q4", arm.steps))
            .expect_err("the weights-free plan cannot complete a real capture");
        assert!(
            !error.contains("does not implement provider"),
            "{} was not dispatched into the Mage arm: {error}",
            arm.provider
        );
    }
    for provider in ["mage_flow_lora", "mage_flow_edit_rl"] {
        let error = run(&mage_planned(provider, provider, "q4", 2)).unwrap_err();
        assert_eq!(
            error,
            format!("MLX five-rung calibration does not implement provider {provider:?}")
        );
    }
}

/// sc-22738. A measured peak ABOVE the engine's own modelled total is a FINDING, never a
/// refusal, and the finding states the delta.
///
/// The numbers are tonight's measured MLX re-capture on this Mac at 768x768 bf16, where the
/// old refusal killed every one of the 18 `mage_flow*:*:mlx` anchors after a complete render:
/// the plain route peaked 0.6 GB above the model and the edit route 7.3 GB above the SAME
/// modelled total, because `generation_peak_gb` scales only its VAE decode term with geometry
/// and has no reference-image term at all.
///
/// MUTATIONS THIS PINS. Dropping the delta from the blocker string reds the third and sixth
/// assertions. Dropping the measured peak, the modelled total, the geometry, the tier or the
/// route reds one of the others. Widening the helper to fire on an equal or under-measured
/// peak reds the last two. Restoring the refusal at the CALL SITE is pinned separately, by
/// [`the_model_shortfall_reaches_the_record_and_never_refuses`] — this test alone could not
/// see it, because the render it happens after needs real weights.
#[test]
fn a_peak_above_the_engines_model_is_recorded_as_a_finding_with_its_delta() {
    let arm = MAGE_ARMS
        .into_iter()
        .find(|arm| arm.provider == "mage_flow_edit")
        .expect("the edit route is a registered Mage arm");
    let measured = 25_635_586_048_u64;
    let modelled = 18_328_139_520_u64;
    let finding = mage_model_shortfall_blocker(arm, "bf16", 768, 768, measured, modelled)
        .expect("a peak above the modelled total is a finding");
    assert!(
        finding.contains(&measured.to_string()),
        "the finding must state the measured peak: {finding}"
    );
    assert!(
        finding.contains(&modelled.to_string()),
        "the finding must state the modelled total: {finding}"
    );
    assert!(
        finding.contains(&(measured - modelled).to_string()),
        "the finding must state the delta: {finding}"
    );
    assert!(
        finding.contains("768x768"),
        "the finding must state the geometry: {finding}"
    );
    assert!(
        finding.contains("bf16") && finding.contains("mage_flow_edit"),
        "the finding must state the tier and the route: {finding}"
    );
    // The plain route's much smaller shortfall at the same geometry and tier is a DIFFERENT
    // number in the record, not a shared "exceeded" sentence.
    let plain = MAGE_ARMS
        .into_iter()
        .find(|arm| arm.provider == "mage_flow")
        .expect("the plain route is a registered Mage arm");
    let plain_finding =
        mage_model_shortfall_blocker(plain, "bf16", 768, 768, 18_924_699_648, modelled)
            .expect("a peak above the modelled total is a finding");
    assert!(
        plain_finding.contains(&(18_924_699_648_u64 - modelled).to_string())
            && plain_finding != finding,
        "each route files its own delta: {plain_finding}"
    );
    // A peak the model DID cover files nothing.
    assert_eq!(
        mage_model_shortfall_blocker(arm, "bf16", 768, 768, modelled, modelled),
        None
    );
    assert_eq!(
        mage_model_shortfall_blocker(arm, "bf16", 768, 768, modelled - 1, modelled),
        None
    );
}

/// sc-22738. The shortfall FINDING reaches the record, and the arm has no exceedance refusal.
///
/// Read the way the sibling receipt guard in this file reads the arm — the exceedance only
/// happens after a measured render, which needs real weights, so the property is pinned over
/// the arm's source. What it pins is exactly the mutation the fix reverses: restoring
/// `if predicted > required_total_peak_bytes { return Err(…) }` reds the second assertion, and
/// computing the finding but never extending it into `lifecycle_blockers` (so the record
/// carries the peak with no statement that the model missed it) reds the first and third.
#[test]
fn the_model_shortfall_reaches_the_record_and_never_refuses() {
    let source = crate::mlx_source_for_tests();
    let start = source
        .find("\nfn run_mage_provider(")
        .expect("the arm still exists");
    let body = &source[start
        ..start
            + source[start..]
                .find("\n}\n")
                .expect("the arm's body closes")];
    assert!(
        body.contains("let model_shortfall_blocker = mage_model_shortfall_blocker("),
        "the arm must compute the shortfall finding"
    );
    assert!(
        body.contains("lifecycle_blockers.extend(model_shortfall_blocker);"),
        "the shortfall finding must be recorded on the record's blockers, not dropped"
    );
    assert!(
        body.contains("\"engineModelledTotalBytes\"")
            && body.contains("\"engineModelShortfallBytes\""),
        "the record must carry the modelled total and the shortfall machine-readably"
    );
    assert!(
        !body.contains("exceeds the pinned provider's own modelled total"),
        "a measured peak above the engine's model is a finding, never a refusal"
    );
    // The `>` comparison against the modelled total must not reappear as a control-flow gate.
    assert!(
        !body.contains("if predicted > required_total_peak_bytes"),
        "the exceedance must not gate the capture"
    );
}

/// sc-22738. The one peak check that survives as a REFUSAL, because it proves the capture is
/// invalid rather than saying something about the engine's model: MLX active memory is
/// absolute, so a render cannot peak below the weights the provider itself declares resident.
///
/// MUTATIONS THIS PINS. Deleting the check, or narrowing it so an under-weight capture stops
/// refusing, reds the first assertion; dropping either figure or the route from the refusal
/// reds the second; relaxing the comparison to `<=` reds the third. The last pins that this
/// check does NOT reclassify the shortfall case — every measured Mage peak tonight is far
/// above the declared weights.
#[test]
fn a_peak_below_the_declared_weights_is_still_a_refusal() {
    let arm = MAGE_ARMS
        .into_iter()
        .find(|arm| arm.provider == "mage_flow")
        .expect("the plain route is a registered Mage arm");
    let error = mage_peak_below_weights_error(arm, "bf16", 1_000, 16_000_000_000)
        .expect("a peak below the declared weights is a refusal");
    assert!(
        error.contains("1000") && error.contains("16000000000") && error.contains("mage_flow"),
        "the refusal must state both figures and the route: {error}"
    );
    assert_eq!(
        mage_peak_below_weights_error(arm, "bf16", 16_000_000_000, 16_000_000_000),
        None
    );
    assert_eq!(
        mage_peak_below_weights_error(arm, "bf16", 18_924_699_648, 16_000_000_000),
        None
    );
}

/// The six checkpoints are architecturally identical, so a plan whose `modelId` names a
/// different member than its `provider` must be refused before anything is loaded — nothing
/// downstream could notice that one variant's peaks had been recorded as another's.
#[test]
fn a_crossed_provider_and_model_pair_is_refused_by_name() {
    let request = mage_planned("mage_flow_turbo", "mage_flow_base", "q4", 4);
    let error = mage_arm(&request).unwrap_err();
    assert_eq!(
        error,
        "the MLX Mage-Flow arm does not implement provider \"mage_flow_turbo\" for model \
             \"mage_flow_base\""
    );
}

/// Every arm's table row is self-consistent, and no two rows share an artifact, an env family, a
/// fixture prefix or an execution path — the four things that keep two members' captures apart.
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
        MAGE_ARMS.map(|arm| arm.slug),
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
        // The distilled members run the distill's own four-step CFG-off recipe.
        let distilled = arm.provider.ends_with("turbo");
        assert_eq!(arm.steps, if distilled { 4 } else { 2 });
        assert_eq!(arm.guidance, if distilled { 1.0 } else { 5.0 });
        assert!(arm.repository_env.ends_with("_REPOSITORY"));
        assert!(arm.revision_env.ends_with("_REVISION"));
        assert!(arm.root_env.ends_with("_ROOT"));
    }
}

/// The fixture binds member, tier, edge and the recipe's step count. A fixture naming another
/// member's prefix, another tier, or another step count is refused.
#[test]
fn the_fixture_binds_the_member_the_tier_and_the_step_count() {
    let arm = MAGE_ARMS[2];
    let good = mage_planned(arm.provider, arm.provider, "q8", arm.steps);
    assert_eq!(planned_mage_seed(&good, arm, "q8").unwrap(), MAGE_SEED);
    // Another member's fixture prefix.
    let crossed = mage_planned(MAGE_ARMS[1].provider, arm.provider, "q8", arm.steps);
    let mut request = good.clone();
    request["planned"]["fixture"] = crossed["planned"]["fixture"].clone();
    assert!(planned_mage_seed(&request, arm, "q8")
        .unwrap_err()
        .contains("must start with"));
    // Another tier.
    assert!(planned_mage_seed(&good, arm, "q4")
        .unwrap_err()
        .contains("must start with"));
    // Another step count.
    let wrong_steps = mage_planned(arm.provider, arm.provider, "q8", 2);
    let error = planned_mage_seed(&wrong_steps, arm, "q8").unwrap_err();
    assert!(
        error.contains("4-step calibration request"),
        "the refusal must name the recipe this member renders, got: {error}"
    );
}

/// A plan row declaring the OTHER mode for a member is refused before any environment work: the
/// record's `mode` comes from the plan while the reference comes from the arm, so an unchecked
/// mismatch would label a reference-conditioned render reference-free.
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
        let error = run(&request).expect_err("a crossed mode must be refused");
        assert_eq!(
            error,
            format!(
                "{} renders {:?}, but the plan declares mode {wrong:?}",
                arm.provider,
                mage_mode(arm)
            )
        );
    }
}

/// The still-geometry guard fires under each member's own label, before any environment work.
#[test]
fn every_mage_member_refuses_a_multi_frame_geometry_under_its_own_label() {
    for arm in MAGE_ARMS {
        for frames in [0_u64, 2, 145] {
            let mut request = mage_planned(arm.provider, arm.provider, "q4", arm.steps);
            request["planned"]["target"]["geometry"]["frames"] = json!(frames);
            let error = run(&request).expect_err("a video geometry must be refused");
            assert_eq!(
                error,
                format!(
                    "{} requires geometry.frames == 1, got {frames}",
                    arm.still_calibration
                )
            );
        }
    }
}

fn mage_temp_root(fixtures: &crate::adapter_test_support::Fixtures, label: &str) -> PathBuf {
    fixtures.path(label)
}

fn staged_snapshot(
    fixtures: &crate::adapter_test_support::Fixtures,
    repository: &str,
    revision: &str,
    tier: &str,
    components: bool,
) -> PathBuf {
    let root = mage_temp_root(fixtures, &repository.replace('/', "-"))
        .join(format!("models--{}", repository.replace('/', "--")))
        .join("snapshots")
        .join(revision);
    if components {
        for component in [
            protocol::MAGE_COMPONENT_TEXT_ENCODER,
            protocol::MAGE_COMPONENT_VAE,
        ] {
            std::fs::create_dir_all(root.join(tier).join(component)).unwrap();
        }
        root
    } else {
        let tier_root = root.join(tier);
        std::fs::create_dir_all(tier_root.join("transformer")).unwrap();
        tier_root
    }
}

const MAGE_TEST_REVISION: &str = "5f6455818d8ca80ce780e9c01b9e0de1d8c5f9db";
const MAGE_TEST_COMPONENTS_REVISION: &str = "c936de2a107ee8d0869137e73943f6414f23adaa";

fn mage_source(
    fixtures: &crate::adapter_test_support::Fixtures,
    arm: MageArm,
    tier: &str,
) -> MageArtifactSource {
    MageArtifactSource {
        repository: arm.repository.to_owned(),
        revision: MAGE_TEST_REVISION.to_owned(),
        root: staged_snapshot(fixtures, arm.repository, MAGE_TEST_REVISION, tier, false),
        components_repository: protocol::MAGE_COMPONENTS_REPOSITORY.to_owned(),
        components_revision: MAGE_TEST_COMPONENTS_REVISION.to_owned(),
        components_root: staged_snapshot(
            fixtures,
            protocol::MAGE_COMPONENTS_REPOSITORY,
            MAGE_TEST_COMPONENTS_REVISION,
            tier,
            true,
        ),
    }
}

fn resident_selection(tier: &str) -> MemorySelection {
    MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: MemoryStrategyParameters::default(),
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: match tier {
                "q4" => Some(Quant::Q4),
                "q8" => Some(Quant::Q8),
                _ => None,
            },
            component_precision_floors: &[],
        },
    }
}

/// Every MLX Mage plan row names the per-(member, tier) production identity the loaded
/// generator publishes (inference PR 953's table), the 18 identities are distinct, no row
/// names a weights-free conformance string or the retired single string, every row binds the
/// deferred shape the worker loads, and the pre-load check refuses a row that names anything
/// else — before any env or weights are touched.
///
/// It also binds [`MAGE_EDGE`] and [`MAGE_SEED`] to those rows: the arm reads geometry and seed
/// from the PLAN, so the two constants are only true of the campaign while every checked-in row
/// declares them, and `planned_mage_seed` must accept each row as written.
#[test]
fn every_planned_mage_mlx_row_names_the_production_identity_and_is_checked_before_the_load() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let anchors = plan["anchors"].as_object().expect("anchors object");
    let mut identities = std::collections::BTreeSet::new();
    let mut seen = std::collections::BTreeSet::new();
    for arm in MAGE_ARMS {
        for tier in ["bf16", "q4", "q8"] {
            let key = format!("{}:{tier}:mlx", arm.provider);
            let row = &anchors[&key];
            let expected = mage_calibration_fingerprint(arm, tier);
            assert_eq!(
                row["calibrationFingerprint"].as_str(),
                Some(expected.as_str()),
                "{key}"
            );
            assert!(!is_mage_weights_free_fingerprint(&expected), "{expected}");
            assert!(expected.contains(&format!("-{tier}-mlx-")), "{expected}");
            assert!(
                identities.insert(expected),
                "{key}: identity shared with another cell"
            );
            assert_eq!(
                row["loadShape"].as_str(),
                Some(protocol::LOAD_SHAPE_DEFERRED),
                "{key}: the worker loads Mage deferred on MLX"
            );
            assert_eq!(
                row["geometry"]["width"].as_u64(),
                Some(u64::from(MAGE_EDGE)),
                "{key}: every Mage anchor is planned at MAGE_EDGE²"
            );
            assert_eq!(
                row["geometry"]["height"].as_u64(),
                Some(u64::from(MAGE_EDGE)),
                "{key}: every Mage anchor is planned at MAGE_EDGE²"
            );
            assert_eq!(
                row["fixture"].as_str(),
                Some(
                    format!(
                        "{}-{tier}-{MAGE_EDGE}-seed{MAGE_SEED}-step{}",
                        arm.fixture_prefix, arm.steps
                    )
                    .as_str()
                ),
                "{key}: the row's fixture must be the member/tier/edge/seed/step spelling"
            );
            // …and `planned_mage_seed` must accept the row exactly as it is written, so the
            // spelling asserted above is the one the arm parses rather than a parallel claim.
            let planned_row = json!({ "planned": {
                "target": { "provider": arm.provider, "modelId": arm.provider, "tier": tier,
                            "geometry": row["geometry"].clone() },
                "fixture": row["fixture"].clone(),
            }});
            assert_eq!(
                planned_mage_seed(&planned_row, arm, tier).unwrap(),
                MAGE_SEED,
                "{key}"
            );
            seen.insert(key.clone());
            let request = json!({ "planned": {
                "target": { "provider": arm.provider, "modelId": arm.provider, "tier": tier },
                "calibrationFingerprint": row["calibrationFingerprint"].clone(),
            }});
            validate_mage_plan_identity(&request, arm, tier).unwrap();
            for stale in [
                "mage-flow-mlx-shared-ladder-2026-08-03-v1".to_owned(),
                format!("mage-flow-mlx-registry-behavior-v1-{}-{tier}", arm.slug),
                format!("mage-flow-cuda-{}-{tier}-shared-ladder-v3", arm.slug),
            ] {
                assert!(
                    is_mage_weights_free_fingerprint(&stale) || stale.contains("-cuda-"),
                    "{stale}"
                );
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
                .map(move |tier| format!("{}:{tier}:mlx", arm.provider))
        })
        .collect();
    assert_eq!(seen, expected);
    assert_eq!(identities.len(), 18);
}

/// The composed `LoadSpec` is the shape the WORKER loads at the planned rung: the variant's own
/// tier root, both shared components staged EXPLICITLY (the variant rehost has no
/// `text_encoder/` or `vae/` sibling for the loader's flat-layout fallback to find), the planned
/// tier's quant, and the offload policy and load shape the caller passed through unchanged
/// (the capture passes the plan's deferred shape; the arguments here are the eager/resident
/// pair so the pass-through itself is what is asserted).
#[test]
fn the_load_spec_binds_the_tier_root_both_components_and_the_planned_quant() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    for arm in MAGE_ARMS {
        for tier in ["bf16", "q4", "q8"] {
            let source = mage_source(fixtures, arm, tier);
            let selection = resident_selection(tier);
            let MageArtifact { root, spec } = mage_load_spec(
                arm,
                tier,
                &source,
                &selection,
                OffloadPolicy::Resident,
                LoadShape::EagerMaterialization,
            )
            .unwrap();
            assert_eq!(
                spec.weights,
                WeightsSource::Dir(root.clone()),
                "the DiT root is the variant's own tier directory"
            );
            assert!(root.ends_with(tier));
            assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
            assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);
            assert_eq!(spec.quantize, selection.tier.quant);
            for component in [
                protocol::MAGE_COMPONENT_TEXT_ENCODER,
                protocol::MAGE_COMPONENT_VAE,
            ] {
                let Some(WeightsSource::Dir(dir)) = spec.components.get(component) else {
                    panic!("{} did not stage the {component} component", arm.provider);
                };
                assert!(
                    dir.ends_with(std::path::Path::new(tier).join(component)),
                    "the {component} component must come from the components snapshot's own \
                         {tier} directory, got {}",
                    dir.display()
                );
                // And never from the variant rehost, which does not ship it.
                assert!(!dir.starts_with(&root));
            }
        }
    }
}

/// A stale export of ANOTHER tier cannot satisfy this plan: the root suffix is checked against
/// the planned tier before the load is paid for (the sc-17097 defect class).
#[test]
fn another_tiers_root_cannot_satisfy_the_planned_tier() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let arm = MAGE_ARMS[0];
    let mut source = mage_source(fixtures, arm, "q4");
    source.root = staged_snapshot(fixtures, arm.repository, MAGE_TEST_REVISION, "q8", false);
    let error = mage_load_spec(
        arm,
        "q4",
        &source,
        &resident_selection("q4"),
        OffloadPolicy::Resident,
        LoadShape::EagerMaterialization,
    )
    .unwrap_err();
    assert!(error.contains("q4"), "{error}");
}

/// Each member binds its OWN rehost. A components root pointing at a variant repository — or a
/// variant root pointing at a sibling's — is refused by repository name.
#[test]
fn each_member_refuses_another_repositorys_root() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let arm = MAGE_ARMS[3];
    let mut crossed = mage_source(fixtures, arm, "q4");
    crossed.repository = MAGE_ARMS[4].repository.to_owned();
    let error = mage_load_spec(
        arm,
        "q4",
        &crossed,
        &resident_selection("q4"),
        OffloadPolicy::Resident,
        LoadShape::EagerMaterialization,
    )
    .unwrap_err();
    assert!(error.contains(arm.repository), "{error}");

    let mut crossed_components = mage_source(fixtures, arm, "q4");
    crossed_components.components_repository = arm.repository.to_owned();
    let error = mage_load_spec(
        arm,
        "q4",
        &crossed_components,
        &resident_selection("q4"),
        OffloadPolicy::Resident,
        LoadShape::EagerMaterialization,
    )
    .unwrap_err();
    assert!(
        error.contains(protocol::MAGE_COMPONENTS_REPOSITORY),
        "{error}"
    );
}

/// The edit members condition on exactly one reference and the text-to-image members on none —
/// the one thing that makes an edit capture measure the edit path (`edit_references` refuses a
/// reference-free edit request outright).
#[test]
fn only_the_edit_members_carry_a_reference_and_only_the_undistilled_a_negative_prompt() {
    for arm in MAGE_ARMS {
        let capabilities = mage_capabilities(arm);
        let request = mage_request(arm, &capabilities, MAGE_EDGE, MAGE_EDGE, MAGE_SEED);
        assert_eq!(request.conditioning.len(), usize::from(arm.edit));
        if arm.edit {
            assert!(matches!(
                request.conditioning[0],
                Conditioning::Reference { .. }
            ));
        }
        assert_eq!(request.steps, Some(arm.steps));
        // Both knobs follow the pinned engine's published capability for this member, never the
        // table's scale (sc-22738) — see
        // `a_distilled_mage_route_sends_no_guidance_and_no_negative_prompt`.
        assert_eq!(request.guidance.is_some(), capabilities.supports_guidance);
        assert_eq!(
            request.negative_prompt.is_some(),
            capabilities.supports_negative_prompt
        );
        assert_eq!(request.seed, Some(MAGE_SEED));
    }
}

/// The run context is the coordinate the provider's own route gate admits this member under
/// (`model.rs` `request_context_error`), and never the opposite one.
#[test]
fn the_run_context_carries_each_members_own_mode_and_reference_count() {
    for arm in MAGE_ARMS {
        let calibration = MemoryCalibrationIdentity::new(
            "mage-flow-mlx-shared-ladder-2026-08-03-v1",
            LoadShape::EagerMaterialization,
        );
        let context = mage_run_context(
            arm,
            resident_selection("q4"),
            &calibration,
            MAGE_EDGE,
            MAGE_EDGE,
            1_024,
            1,
            0,
        );
        let expected = if arm.edit {
            MemoryMode::Edit
        } else {
            MemoryMode::TextToImage
        };
        assert_eq!(context.mode, expected);
        assert_eq!(context.has_reference, arm.edit);
        assert_eq!(context.geometry.reference_count, u32::from(arm.edit));
        assert_eq!(context.overlay, None);
        assert!(!context.has_phases);
        assert!(!context.use_pid);
        assert_eq!(context.load_shape, LoadShape::EagerMaterialization);
    }
}

/// The pinned provider's own weights-free memory registration, per member, in `MAGE_ARMS`
/// order. Pairing them by index is only safe because the test below re-asserts each pair's
/// `provider_id` against the arm row, so a re-ordered table cannot silently drive one member's
/// context through another member's predicate.
const MAGE_MEMORY_REGISTRATIONS: [(
    mlx_gen_mage::MageVariant,
    &mlx_gen::gen_core::MemoryRegistration,
); 6] = [
    (
        mlx_gen_mage::MageVariant::Rl,
        &mlx_gen_mage::model::MEMORY_REGISTRATION,
    ),
    (
        mlx_gen_mage::MageVariant::Base,
        &mlx_gen_mage::model::MEMORY_REGISTRATION_BASE,
    ),
    (
        mlx_gen_mage::MageVariant::Turbo,
        &mlx_gen_mage::model::MEMORY_REGISTRATION_TURBO,
    ),
    (
        mlx_gen_mage::MageVariant::Edit,
        &mlx_gen_mage::model::MEMORY_REGISTRATION_EDIT,
    ),
    (
        mlx_gen_mage::MageVariant::EditBase,
        &mlx_gen_mage::model::MEMORY_REGISTRATION_EDIT_BASE,
    ),
    (
        mlx_gen_mage::MageVariant::EditTurbo,
        &mlx_gen_mage::model::MEMORY_REGISTRATION_EDIT_TURBO,
    ),
];

/// Refusals that come from the LIVE MLX wired-memory budget rather than from the request.
///
/// `mlx-gen-mage`'s resident rung resolves `production_safe_budget_gb()` from the process's
/// current MLX limits, which sibling arms in this binary deliberately move (`set_memory_limit`)
/// and a busy host shrinks — the 768² q4 cell has been observed refused at a 7.30 GB live
/// budget in a shared test process. That is process state, not a caller claim, so the baseline
/// tolerates exactly these three reasons and nothing else.
fn is_live_budget_refusal(reason: &str) -> bool {
    reason.contains("unified-memory limit is unavailable")
        || reason.contains("max-buffer probe is unavailable")
        || reason.contains("exceeding the safe wired-memory budget")
}

fn mage_registration_spec(
    fixtures: &crate::adapter_test_support::Fixtures,
    tier: &str,
) -> LoadSpec {
    let spec = LoadSpec::new(WeightsSource::Dir(mage_temp_root(fixtures, "registration")));
    match resident_selection(tier).tier.quant {
        Some(quant) => spec.with_quant(quant),
        None => spec,
    }
}

/// The request context every Mage capture is admitted under, built exactly as
/// [`run_mage_provider`] builds it, for one member and tier against a weights-free contract.
fn mage_admitted_context(
    arm: MageArm,
    variant: mlx_gen_mage::MageVariant,
    tier: &str,
) -> (mlx_gen::gen_core::MemoryProviderContract, MemoryRunContext) {
    let quant = resident_selection(tier).tier.quant;
    let contract = mlx_gen_mage::model::memory_strategy_contract(arm.provider, quant);
    let mut selection = resident_selection(tier);
    selection.tier.component_precision_floors = active_component_floors(
        mlx_gen_mage::descriptor_for(variant)
            .capabilities
            .component_precision_floors,
        quant,
    );
    let calibration = contract
        .calibration
        .clone()
        .expect("the weights-free Mage contract publishes a conformance identity");
    let (required, credit) = mage_admission_budget(&contract, quant, MAGE_EDGE, MAGE_EDGE);
    let context = mage_run_context(
        arm,
        selection,
        &calibration,
        MAGE_EDGE,
        MAGE_EDGE,
        required
            .saturating_add(credit)
            .saturating_add(1_000_000_000),
        required.saturating_sub(credit),
        credit,
    );
    (contract, context)
}

/// sc-22738: the capture's request context is admitted by the PINNED provider's own safety
/// predicate, for all six members at all three tiers, and each of the three things the campaign
/// got wrong is refused by name.
///
/// The campaign's every Mage anchor died here, not in a render: the arm declared the
/// placeholder peak of `1` against a committed snapshot of `0`, which asks Mage to credit its
/// entire model as already-resident memory nobody had charged, and the q4 cells additionally
/// declared a bare Q4 tier against a provider that holds its text encoder and transformer head
/// at Q8. Both are caller-side claims, so both are asserted here — weights-free, through the
/// provider's own registered predicate rather than a restatement of it.
#[test]
fn every_member_and_tier_is_admitted_by_the_pinned_providers_own_safety_predicate() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    for (arm, (variant, registration)) in MAGE_ARMS.iter().zip(MAGE_MEMORY_REGISTRATIONS) {
        assert_eq!(
            registration.provider_id, arm.provider,
            "the registration table is out of order with MAGE_ARMS"
        );
        for tier in ["bf16", "q4", "q8"] {
            let spec = mage_registration_spec(fixtures, tier);
            let (contract, context) = mage_admitted_context(*arm, variant, tier);
            let key = format!("{}:{tier}", arm.provider);
            // Every REQUEST-context rule must admit this pair. The resident rung then resolves
            // the host's live wired-memory budget, and that number is not a property of the
            // request: it is MLX process state, which sibling arms in this same test binary
            // move with `set_memory_limit`, so an assertion on it would pass or fail by test
            // order. The three mutations below fire inside the request-context rules — ahead of
            // that branch — which is what keeps this baseline honest.
            match (registration.safety_check)(&spec, &contract, &context) {
                MemorySafetyDecision::Accept => {}
                MemorySafetyDecision::Reject { reason } => assert!(
                    is_live_budget_refusal(&reason),
                    "{key}: the pinned provider refused the capture's own request context: \
                         {reason}"
                ),
            }

            // The peak/credit pair the campaign got wrong: a placeholder peak asks the
            // provider to credit everything it did not see against an uncharged snapshot.
            let mut placeholder = context.clone();
            placeholder.predicted_peak_bytes = 1;
            placeholder.budget.committed_bytes = 0;
            let MemorySafetyDecision::Reject { reason } =
                (registration.safety_check)(&spec, &contract, &placeholder)
            else {
                panic!("{key}: a placeholder peak of 1 was admitted");
            };
            assert!(
                reason.contains("is inconsistent with provider total"),
                "{key}: {reason}"
            );

            // The tier the campaign got wrong: the selected tier must carry exactly the
            // promotions the loaded provider applies — neither fewer nor more.
            let declared = mlx_gen_mage::descriptor_for(variant)
                .capabilities
                .component_precision_floors;
            let mut crossed_floors = context.clone();
            crossed_floors.selection.tier.component_precision_floors =
                if context.selection.tier.component_precision_floors.is_empty() {
                    declared
                } else {
                    &[]
                };
            let MemorySafetyDecision::Reject { reason } =
                (registration.safety_check)(&spec, &contract, &crossed_floors)
            else {
                panic!("{key}: a tier carrying the wrong component precision floors was admitted");
            };
            assert!(
                reason.contains("does not match loaded tier"),
                "{key}: {reason}"
            );

            // And the per-variant route gate, on the predicate rather than on the arm's own
            // table: an edit member never admits a text-to-image request, or the reverse.
            let mut crossed_mode = context.clone();
            crossed_mode.mode = if arm.edit {
                MemoryMode::TextToImage
            } else {
                MemoryMode::Edit
            };
            crossed_mode.has_reference = !arm.edit;
            crossed_mode.geometry.reference_count = u32::from(!arm.edit);
            let MemorySafetyDecision::Reject { reason } =
                (registration.safety_check)(&spec, &contract, &crossed_mode)
            else {
                panic!("{key}: the opposite request mode was admitted");
            };
            assert!(
                reason.contains("does not match") && reason.contains(arm.provider),
                "{key}: {reason}"
            );
        }
    }
}

/// sc-22738 — every Mage route is graded on all three declared lifecycle phases, each cut on a
/// boundary the pinned engine emits on the production path (inference #963 at e16c6a55e:
/// `Loading(Renderer)` after the prompt encode on the Resident warm arm, `Progress::Decoding`
/// from the decode closure). The pin before it bounded neither edge and this arm filed an
/// overall-only ceiling with the phases declared unattributed; that relaxation is gone, and a
/// zero in any phase is again a refusal. Read the way the sibling guards in this file read an
/// arm — the measured render needs real weights.
#[test]
fn every_mage_route_cuts_every_declared_phase_on_an_engine_boundary() {
    let source = crate::mlx_source_for_tests();
    let start = source
        .find("\nfn run_mage_provider(")
        .expect("the arm still exists");
    let body = &source[start
        ..start
            + source[start..]
                .find("\n}\n")
                .expect("the arm's body closes")];
    for cut in [
        "Progress::Loading(LoadPhase::Renderer) => {",
        "Progress::Decoding => {",
        "if [conditioning.active, denoise.active, decode.active].contains(&0) {",
        "(\"conditioningActivePeak\", \"bytes\", conditioning.active)",
        "(\"denoiseActivePeak\", \"bytes\", denoise.active)",
        "(\"decodeActivePeak\", \"bytes\", decode.active)",
    ] {
        assert!(
            body.contains(cut),
            "the strict three-phase capture lost {cut}"
        );
    }
    for relaxation in [
        "Progress::Step { current: 1, .. } =>",
        "PhaseUnattributed",
        "overall_only_json",
    ] {
        assert!(
            !body.contains(relaxation),
            "the Mage arm must not grade on {relaxation} again"
        );
    }
}

/// sc-22738 — the render's informational coherence note is DEFERRED, so a capture that fails
/// after the render is named on stderr by its own failure. The runner reads the first stderr
/// line, and eighteen Mage refusals were reported as a coherence statistic because this line
/// went out the moment `generate` returned.
#[test]
fn the_coherence_note_is_deferred_rather_than_printed_at_the_render() {
    let source = crate::mlx_source_for_tests();
    let start = source
        .find("fn report_gpu_view_retries(")
        .expect("the reporter still exists");
    let body = &source[start
        ..start
            + source[start..]
                .find("\n}\n")
                .expect("the reporter's body closes")];
    assert!(
        body.contains("protocol::defer_note("),
        "the coherence note must be deferred"
    );
    assert!(
        !body.contains("eprintln!"),
        "the coherence note must not be written to stderr at the render"
    );
    // And the deferred buffer is flushed on both terminals: `main` emits it on the success path,
    // and `protocol::fail` emits it AFTER the failure line — the ordering that makes the
    // runner's "first stderr line" name the real failure.
    assert!(source.contains("protocol::flush_deferred_notes();"));
    let protocol_source = include_str!("../lib.rs");
    let fail = protocol_source
        .find("pub fn fail(")
        .expect("the failure terminal still exists");
    let failure_line = protocol_source[fail..]
        .find("eprintln!(\"memory-strategy provider adapter: {}\"")
        .expect("the failure line");
    let flush = protocol_source[fail..]
        .find("flush_deferred_notes();")
        .expect("the notes are flushed on the failure path");
    assert!(
        failure_line < flush,
        "the failure line must be written BEFORE the deferred notes"
    );
}
