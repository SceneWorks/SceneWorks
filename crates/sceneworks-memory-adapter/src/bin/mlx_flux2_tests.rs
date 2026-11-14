use super::*;
use mlx_gen::gen_core::MemoryStrategySupport;

/// sc-22738: a warm-repeat retention over the clean warm bound is RECORDED, not refused — and
/// the recording has to survive into the blocker list, because that list is the only place the
/// overage is named. Both halves are asserted here: the bound still discriminates (a reading
/// inside it adds nothing, one byte over on either axis adds a finding), and the finding
/// actually reaches the blockers the record ships.
///
/// The over-bound case is the real `flux2_klein_9b:q8:mlx` reading from the 2026-09-06 MLX
/// walk, scaled nowhere: the campaign's own numbers, so a future reader can match the finding
/// text against the log line that motivated this.
#[test]
fn a_flux2_warm_retention_over_the_bound_is_recorded_as_a_blocker_not_refused() {
    let scope = "scope blocker";
    let clean = AllocatorState {
        active: 1_000,
        cache: 200,
    };
    let bounds = LifecycleMemoryBounds::from_clean_warm(10_000, clean);
    assert_eq!(bounds.tolerance_bytes, 200);

    // Inside the bound on both axes: the standing scope blocker and nothing else.
    let within = flux2_lifecycle_blockers(
        "flux2_klein_9b",
        scope,
        bounds,
        AllocatorState {
            active: 1_200,
            cache: 400,
        },
    );
    assert_eq!(
        within,
        vec![scope.to_owned()],
        "a retention inside the bound must add no finding"
    );

    // One byte over on EITHER axis is a finding, so neither axis can be dropped unnoticed.
    for over in [
        AllocatorState {
            active: 1_201,
            cache: 400,
        },
        AllocatorState {
            active: 1_200,
            cache: 401,
        },
    ] {
        let blockers = flux2_lifecycle_blockers("flux2_klein_9b", scope, bounds, over);
        assert_eq!(
            blockers.len(),
            2,
            "a retention above the bound must be recorded beside the scope blocker: {blockers:?}"
        );
        assert_eq!(blockers[0], scope, "the scope blocker must stay first");
        let finding = &blockers[1];
        assert!(
            finding.contains("flux2_klein_9b"),
            "the finding must name the cell: {finding}"
        );
        // The tolerance is asserted through its OWN phrase, not a bare "200": the clean
        // cleanup's cache is also 200, so a loose substring check stayed green when the
        // tolerance was replaced by a literal zero (caught by mutation M4).
        assert!(
            finding.contains("active: 1000") && finding.contains("plus 200 bytes"),
            "the finding must carry the clean warm cleanup and the tolerance: {finding}"
        );
    }

    // The measured klein q8 event itself: 5,208,457,240 bytes over a 709,354,919-byte
    // tolerance. It is recorded, and the capture is NOT refused — the whole point of sc-22738.
    let measured = LifecycleMemoryBounds::from_clean_warm(
        35_467_745_964,
        AllocatorState {
            active: 24_981_289_616,
            cache: 0,
        },
    );
    assert_eq!(measured.tolerance_bytes, 709_354_919);
    let blockers = flux2_lifecycle_blockers(
        "flux2_klein_9b",
        scope,
        measured,
        AllocatorState {
            active: 30_189_746_856,
            cache: 0,
        },
    );
    assert_eq!(
        blockers.len(),
        2,
        "the measured klein q8 retention is a finding"
    );
    assert!(
        blockers[1].contains("active: 30189746856")
            && blockers[1].contains("active: 24981289616")
            && blockers[1].contains("plus 709354919 bytes"),
        "{:?}",
        blockers[1]
    );

    // ...and the clean rerun of that same anchor, which retained nothing, stays quiet.
    assert_eq!(
        flux2_lifecycle_blockers(
            "flux2_klein_9b",
            scope,
            measured,
            AllocatorState {
                active: 24_981_289_616,
                cache: 0,
            },
        ),
        vec![scope.to_owned()],
    );
}

/// sc-18808 added the still geometry: `validate_flux2_target` now refuses a non-still target
/// alongside a foreign provider, so a request that omits the axis entirely can no longer reach
/// the rung gate this module's tests are aiming at.
fn minimal_request(provider: &str, rung: &str) -> Value {
    minimal_request_for(provider, provider, rung)
}

/// The same shape with the CATALOG model id spelled independently of the provider — the axis
/// the two klein models differ on (sc-22727).
fn minimal_request_for(provider: &str, model_id: &str, rung: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": model_id,
                "overlay": "none",
                "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
            },
            "strategy": { "rung": rung, "parameters": {} }
        }
    })
}

/// The per-arm twin of `validate_z_image_batch`'s provider guard (sc-18104), extended to the
/// `(provider, modelId)` pair by sc-22727: dispatch routes by provider name today, and two
/// catalog models share one provider id, so a misrouted target must be refused by name INSIDE
/// the arm, and the refusal must not read like a missing-field complaint.
#[test]
fn the_flux2_arm_refuses_a_foreign_provider_or_model_by_name() {
    for (provider, model_id) in [
        ("z_image_turbo", "z_image_turbo"),
        ("qwen_image", "qwen_image"),
        ("flux2_dev_edit", "flux2_dev_edit"),
        ("flux2_klein_9b_kv_edit", "flux2_klein_9b_kv_edit"),
        // The engine's third klein artifact route: a real catalog model on this provider that
        // this arm does NOT serve (its snapshot is an assembled convert dir, not a tiered
        // rehost), so it must be refused rather than measured as the base klein.
        ("flux2_klein_9b", "flux2_klein_9b_true_v2"),
        // The KV artifact must never be served by the base klein plan, or vice versa: the two
        // differ ONLY in this field.
        ("flux2_dev", "flux2_klein_9b"),
    ] {
        let error = run_flux2(&minimal_request_for(provider, model_id, "resident"))
            .expect_err("a foreign (provider, modelId) pair must not reach a FLUX.2 contract");
        assert_eq!(
                error,
                format!(
                    "the MLX FLUX.2 arm does not implement provider {provider:?} for model {model_id:?}"
                )
            );
        assert!(
            !error.contains("must be a string") && !error.contains("fingerprint"),
            "refusal came from deeper in the arm, so the guard let it through: {error}"
        );
    }
    // And the guard is not a blanket refusal: every shipped member resolves.
    for (provider, model_id, expected) in [
        (FLUX2_PROVIDER, "flux2_dev", FLUX2_DEV_ARM),
        (FLUX2_KLEIN_PROVIDER, "flux2_klein_9b", FLUX2_KLEIN_ARM),
        (
            FLUX2_KLEIN_PROVIDER,
            "flux2_klein_9b_kv",
            FLUX2_KLEIN_KV_ARM,
        ),
    ] {
        let arm = flux2_arm(&minimal_request_for(provider, model_id, "resident")).unwrap();
        assert_eq!(arm, expected, "{provider}/{model_id}");
        assert_eq!(arm.model_id, model_id);
    }
    assert!(
        flux2_arm(&json!({ "planned": { "target": { "provider": FLUX2_PROVIDER } } }))
            .unwrap_err()
            .contains("planned.target.modelId")
    );
}

/// Every member carries its OWN artifact family, execution path, refusal label, diagnostics
/// slug and fixture prefix. Two members sharing any of them would make their records
/// indistinguishable, and two members sharing an env family would let one artifact satisfy the
/// other's plan (sc-22727).
#[test]
fn every_flux2_member_is_distinguishable_from_every_other() {
    let arms = [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM];
    for field in [
        arms.map(|arm| arm.model_id),
        arms.map(|arm| arm.execution_path),
        arms.map(|arm| arm.still_calibration),
        arms.map(|arm| arm.repository_env),
        arms.map(|arm| arm.revision_env),
        arms.map(|arm| arm.root_env),
        arms.map(|arm| arm.expected_repository),
        arms.map(|arm| arm.fixture_slug),
        arms.map(|arm| arm.slug),
    ] {
        let mut unique = field.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), arms.len(), "collision in {field:?}");
    }
    // The KV rehost and the base rehost are separate repositories, and neither is the dev one.
    assert_eq!(
        FLUX2_KLEIN_ARM.expected_repository,
        protocol::FLUX2_KLEIN_REPOSITORY
    );
    assert_eq!(
        FLUX2_KLEIN_KV_ARM.expected_repository,
        protocol::FLUX2_KLEIN_KV_REPOSITORY
    );
    // But they DO share the engine provider id: that is the whole reason `modelId` is the
    // discriminator rather than `provider`.
    assert_eq!(FLUX2_KLEIN_ARM.provider, FLUX2_KLEIN_KV_ARM.provider);
}

/// The Dev capture measures unscoped resident renders and warm repeats. It must refuse every
/// other rung even though the provider now supports staging, rather than mislabel resident
/// measurements as staged evidence. Each rejected rung must be named back. The
/// klein ladder publishes five rungs, so it is NOT refused here — its selection is settled by
/// the pinned contract instead.
#[test]
fn the_flux2_dev_capture_refuses_rungs_its_resident_measurement_does_not_execute() {
    assert!(flux2_request(1024, 1024, FLUX2_SEED).memory.is_none());
    for rung in [
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ] {
        let error = run_flux2(&minimal_request(FLUX2_PROVIDER, rung))
            .expect_err("a non-resident rung must be refused");
        assert!(
            error.contains(rung) && error.contains("resident"),
            "refusal must name the rung and the resident capture boundary: {error}"
        );
        let klein = run_flux2(&minimal_request_for(
            FLUX2_KLEIN_PROVIDER,
            "flux2_klein_9b",
            rung,
        ))
        .expect_err("the minimal klein request is still incomplete");
        assert!(
            !klein.contains("captures only the resident strategy"),
            "the klein ladder must not borrow the dev capture's resident-only refusal: {klein}"
        );
    }
    let resident = run_flux2(&minimal_request(FLUX2_PROVIDER, "resident"))
        .expect_err("the minimal resident request is still incomplete");
    assert!(
        !resident.contains("not capturable"),
        "the resident rung itself must pass the rung gate: {resident}"
    );
}

#[test]
fn flux2_fixture_is_bound_to_member_tier_geometry_and_step_count() {
    let request = json!({
        "planned": { "fixture": "flux2-dev-mlx-q4-768-seed18218-step2" }
    });
    assert_eq!(
        planned_flux2_seed(&request, FLUX2_DEV_ARM, "q4", 768).unwrap(),
        18218
    );
    assert!(planned_flux2_seed(&request, FLUX2_DEV_ARM, "q8", 768)
        .unwrap_err()
        .contains("must start with"));
    assert!(planned_flux2_seed(&request, FLUX2_DEV_ARM, "q4", 1024)
        .unwrap_err()
        .contains("must start with"));
    // A dev fixture must not satisfy a klein plan, and the two klein members must not satisfy
    // each other's: the family segment is part of the binding (sc-22727).
    for arm in [FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
        assert!(planned_flux2_seed(&request, arm, "q4", 768)
            .unwrap_err()
            .contains("must start with"));
    }
    let klein = json!({
        "planned": { "fixture": "flux2-klein-9b-mlx-bf16-768-seed22727-step2" }
    });
    assert_eq!(
        planned_flux2_seed(&klein, FLUX2_KLEIN_ARM, "bf16", 768).unwrap(),
        FLUX2_KLEIN_SEED
    );
    assert!(planned_flux2_seed(&klein, FLUX2_KLEIN_KV_ARM, "bf16", 768)
        .unwrap_err()
        .contains("must start with"));
    let kv = json!({
        "planned": { "fixture": "flux2-klein-9b-kv-mlx-bf16-768-seed22727-step2" }
    });
    assert_eq!(
        planned_flux2_seed(&kv, FLUX2_KLEIN_KV_ARM, "bf16", 768).unwrap(),
        FLUX2_KLEIN_SEED
    );
    assert!(planned_flux2_seed(&kv, FLUX2_KLEIN_ARM, "bf16", 768)
        .unwrap_err()
        .contains("must start with"));
    let three_step = json!({
        "planned": { "fixture": "flux2-dev-mlx-q4-768-seed18218-step3" }
    });
    assert!(planned_flux2_seed(&three_step, FLUX2_DEV_ARM, "q4", 768)
        .unwrap_err()
        .contains("two-step"));
}

/// sc-17097's lesson applied to this arm: the tier is derived from the planned target, never a
/// hardcoded q4 (sc-22727 widened this arm from q4/q8 to the full shipped ladder). WHERE the
/// tier reaches the loader is per member: the dev route folds it through `LoadSpec::quantize`
/// (bf16 is the dense base and carries none — the worker's `tier_to_quant` shape), while the
/// klein turnkey rehosts declare their tier in the snapshot path and REFUSE a spec that also
/// carries a quant — "flux2 Klein turnkey tiers require BF16 execution with
/// LoadSpec.quantize=None" (`mlx-gen-flux2/src/artifact_inventory.rs`), which the sc-22727
/// proof capture hit for real before this flag existed.
#[test]
fn flux2_load_spec_carries_the_planned_tier_the_way_each_member_takes_it() {
    for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
        for (tier, folded_quant) in [
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
            ("bf16", None),
        ] {
            let request = json!({
                "planned": {
                    "strategy": { "rung": "resident", "parameters": {} },
                    "target": { "tier": tier }
                }
            });
            let selection = planned_selection(&request).unwrap();
            // The PLANNED tier is derived identically on every member; only whether it is
            // handed to the loader differs.
            assert_eq!(selection.tier.quant, folded_quant, "planned tier {tier}");
            let spec = flux2_spec(
                arm,
                PathBuf::from(format!("/tmp/{}-{tier}", arm.slug)),
                &selection,
            );
            let expected = if arm.tier_quant_reaches_the_loader {
                folded_quant
            } else {
                None
            };
            assert_eq!(spec.quantize, expected, "{} tier {tier}", arm.model_id);
            assert_eq!(spec.load_shape, arm.load_shape);
            // The catalog model id reaches the engine as `resolved_route`: it is what
            // `KleinArtifactInventory::validate_resolved_route` refuses a cross-variant
            // artifact with (mlx-gen-flux2/src/artifact_inventory.rs).
            assert_eq!(spec.resolved_route.as_deref(), Some(arm.model_id));
        }
    }
    // Which member takes the fold, stated as data so the loop above cannot pass vacuously by
    // every member happening to answer the same way.
    assert_eq!(
        [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM]
            .map(|arm| arm.tier_quant_reaches_the_loader),
        [true, false, false],
        "only the dev route folds the planned tier into LoadSpec::quantize"
    );
}

/// Every FLUX.2 arm loads the shape of the RUNG IT PRICES: `Resident` + `EagerMaterialization`,
/// no quant on the klein turnkeys.
///
/// sc-22727 rewrote this. The klein arms used to load `Sequential` + `DeferredMaterialization`
/// because that was the only shape the pinned engine published a calibration identity under —
/// `klein_contract_for` set `calibration` iff `klein_streamable(spec)`. Inference PR 948
/// decoupled the two: the identity is now a property of the admitted artifact and its tier and
/// is published for every load shape, while `klein_streamable` still gates rung 4 alone. The
/// streamable shape is therefore no longer required for an identity — and it is not RENDERABLE
/// at this rung either: a deferred klein transformer refuses without an explicit block window
/// (`mlx-gen-flux2/src/transformer.rs`), which is where the sc-22727 proof capture died, 123 s
/// in, after clearing the whole contract-and-load path. The predicate is transcribed here term
/// by term (it is crate-private at the pin) so the arms can assert they are OUTSIDE it.
#[test]
fn flux2_specs_are_the_resident_rungs_shape_and_are_deliberately_not_streamable() {
    let klein_streamable = |spec: &LoadSpec| {
        spec.offload_policy == OffloadPolicy::Sequential
            && spec.load_shape == LoadShape::DeferredMaterialization
            && spec.quantize.is_none()
            && spec.adapters.is_empty()
            && spec.control.is_none()
            && spec.extra_controls.is_empty()
            && spec.ip_adapter.is_none()
            && spec.identity.is_none()
            && spec.text_encoder.is_none()
            && spec.components.is_empty()
            && matches!(spec.weights, WeightsSource::Dir(_))
    };
    for tier in ["q4", "q8", "bf16"] {
        let request = json!({
            "planned": {
                "strategy": { "rung": "resident", "parameters": {} },
                "target": { "tier": tier }
            }
        });
        let selection = planned_selection(&request).unwrap();
        for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
            let spec = flux2_spec(
                arm,
                PathBuf::from(format!("/tmp/{}-{tier}", arm.slug)),
                &selection,
            );
            assert_eq!(
                spec.offload_policy,
                OffloadPolicy::Resident,
                "{} {tier}",
                arm.model_id
            );
            assert_eq!(
                spec.load_shape,
                LoadShape::EagerMaterialization,
                "{} {tier}",
                arm.model_id
            );
            assert!(
                !klein_streamable(&spec),
                "{} {tier}: the resident rung must not load the rung-4 streamable shape: \
                     {spec:?}",
                arm.model_id
            );
        }
        // The klein turnkeys carry no load-time quant at all; the dev route folds the tier.
        for arm in [FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
            let spec = flux2_spec(
                arm,
                PathBuf::from(format!("/tmp/{}-{tier}", arm.slug)),
                &selection,
            );
            assert!(spec.quantize.is_none(), "{} {tier}", arm.model_id);
        }
    }
    // Stated as data, so the loop cannot pass by every member answering the same way.
    assert_eq!(
        [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM].map(|arm| arm.offload_policy),
        [
            OffloadPolicy::Resident,
            OffloadPolicy::Resident,
            OffloadPolicy::Resident
        ],
    );
}

/// The arm table is bound to the two documents the worker and the harness actually read.
///
/// * The shipped manifest, where the worker takes both facts from: `mlx.denseTextEncoderTier`
///   (`is_dense_te_tier` — its reason to load a klein tier with `Quant::None`, so the flag must
///   equal `!tier_quant_reaches_the_loader`), and the MLX `bounded_transformer_residency` row's
///   `requiredOffloadPolicy` under the plain provider (`apply_declared_mlx_load_policy_for_request`
///   — its reason to bind Sequential; a member with no such row stays Resident). Every rung
///   row's `fingerprint` is the WEIGHTS-FREE registry declaration
///   ([`Flux2Arm::registry_fingerprint`]) — sc-22727: on the klein routes that is no longer
///   the identity a load of real weights publishes, so the manifest is bound to the
///   declaration and the plan to the production identity, separately.
/// * The anchor plan, whose `loadShape` the harness checks the measured record against and
///   whose `calibrationFingerprint` `run_flux2` checks the pinned contract against: every MLX
///   row of every member must spell the member's shape and its per-tier PRODUCTION identity
///   ([`flux2_calibration_fingerprint`]).
///
/// Flip any arm flag without the manifest or the plan moving and this reds.
#[test]
fn the_flux2_arm_table_agrees_with_the_shipped_manifest_and_the_anchor_plan() {
    let manifest: Value =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("the shipped models manifest parses")
            .clone();
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut klein_identities = std::collections::BTreeMap::<String, String>::new();
    for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
        let entry = manifest["models"]
            .as_array()
            .expect("models")
            .iter()
            .find(|entry| entry["id"] == arm.model_id)
            .unwrap_or_else(|| panic!("{} is not a shipped model", arm.model_id));
        let mlx = &entry["mlx"];
        let dense_te = mlx["denseTextEncoderTier"] == json!(true);
        assert_eq!(
            arm.tier_quant_reaches_the_loader, !dense_te,
            "{}: the worker loads a dense-TE tier with Quant::None (is_dense_te_tier)",
            arm.model_id
        );
        let contract = &mlx["memoryStrategyContract"];
        assert_eq!(contract["provider"], arm.provider, "{}", arm.model_id);
        // The RESIDENT rung's row - the rung this anchor prices. sc-22727: it is deliberately
        // NOT the `bounded_transformer_residency` row. Both klein entries carry a BTR row with
        // `requiredOffloadPolicy: "sequential"`, but that row describes rung 4's load, and a
        // deferred klein transformer cannot render at all without a block window
        // (`mlx-gen-flux2/src/transformer.rs`, proved by the sc-22727 proof capture). The
        // resident rung declares no `requiredOffloadPolicy` at all, which is the worker's
        // Resident + Eager default, and that is what this arm loads.
        let resident_rows = contract["implementations"]
            .as_array()
            .expect("implementations")
            .iter()
            .filter(|row| {
                row["rung"] == "resident"
                    && row
                        .get("runtimeProvider")
                        .and_then(Value::as_str)
                        .unwrap_or(arm.provider)
                        == arm.provider
            })
            .collect::<Vec<_>>();
        let [resident_row] = resident_rows.as_slice() else {
            panic!(
                "{}: {} resident rows for one provider",
                arm.model_id,
                resident_rows.len()
            )
        };
        assert!(
            resident_row.get("requiredOffloadPolicy").is_none(),
            "{}: the resident rung declares no offload policy; it is the eager default",
            arm.model_id
        );
        assert_eq!(
            resident_row["fingerprint"], arm.registry_fingerprint,
            "{}: the manifest fingerprint is the weights-free registry declaration",
            arm.model_id
        );
        assert_eq!(
            arm.offload_policy,
            OffloadPolicy::Resident,
            "{}: the resident rung is a Resident load",
            arm.model_id
        );
        assert_eq!(
            arm.load_shape,
            LoadShape::EagerMaterialization,
            "{}: a Resident policy keeps the eager default",
            arm.model_id
        );
        let expected_shape = protocol::LOAD_SHAPE_EAGER;
        for tier in ["q4", "q8", "bf16"] {
            let key = format!("{}:{tier}:mlx", arm.model_id);
            let row = &plan["anchors"][&key];
            assert!(row.is_object(), "{key} is not a planned anchor");
            assert_eq!(row["loadShape"], expected_shape, "{key}");
            assert_eq!(row["provider"], arm.provider, "{key}");
            // The PRODUCTION identity, per tier — not the manifest's registry declaration.
            assert_eq!(
                row["calibrationFingerprint"],
                json!(flux2_calibration_fingerprint(arm, tier)),
                "{key}"
            );
            if arm.provider == FLUX2_KLEIN_PROVIDER {
                // sc-22727: a klein plan row still carrying the weights-free declaration would
                // bind a cell no load of real weights ever returns, and the six klein cells
                // are six DISTINCT measured cells — two artifact families times three tiers.
                assert_ne!(
                    flux2_calibration_fingerprint(arm, tier),
                    arm.registry_fingerprint,
                    "{key}: the registry declaration is not a production identity"
                );
                assert!(
                    klein_identities
                        .insert(flux2_calibration_fingerprint(arm, tier), key.clone())
                        .is_none(),
                    "{key}: this identity is already claimed by another klein cell"
                );
            }
        }
    }
    assert_eq!(
        klein_identities.len(),
        6,
        "six distinct FLUX.2 klein MLX identities: two artifact families times three tiers"
    );
}

/// A plan row spelling a load shape other than the member's worker shape is refused by name
/// before any environment or weight work — the harness would refuse the record afterwards, but
/// only after a full load.
#[test]
fn run_flux2_refuses_a_plan_whose_load_shape_is_not_the_members_worker_shape() {
    for (arm, wrong) in [
        (FLUX2_DEV_ARM, protocol::LOAD_SHAPE_DEFERRED),
        (FLUX2_KLEIN_ARM, protocol::LOAD_SHAPE_DEFERRED),
        (FLUX2_KLEIN_KV_ARM, protocol::LOAD_SHAPE_DEFERRED),
    ] {
        let mut request = minimal_request_for(arm.provider, arm.model_id, "resident");
        request["planned"]["loadShape"] = json!(wrong);
        let error = run_flux2(&request).expect_err("a crossed load shape must be refused");
        assert!(
            error.contains("worker load shape"),
            "{}: {error}",
            arm.model_id
        );
    }
}

/// The PLANNED tier must be carried by the root, and the member's OWN repository must be the
/// one bound — a q4 export cannot satisfy a q8 or bf16 plan, and the KV artifact cannot satisfy
/// a base-klein plan even though both load through `flux2_klein_9b`.
#[test]
fn flux2_root_must_carry_the_planned_tier_and_the_members_own_repository() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    const REVISION: &str = "acf05e8d5103838baba6a5e32dc91d6997a56023";
    for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
        let selection = resident_selection(Some(Quant::Q4));
        let q4_root = flux2_snapshot_root(fixtures, arm.expected_repository, REVISION, "q4");
        let error = flux2_load_spec_at(
            arm,
            "q8",
            &selection,
            arm.expected_repository.to_owned(),
            REVISION.to_owned(),
            q4_root.clone(),
        )
        .expect_err("a q8 plan must not be satisfied by a q4 root");
        assert!(
            error.ends_with(&format!("/snapshots/{REVISION}/q8")),
            "{}: {error}",
            arm.model_id
        );
        for tier in ["q4", "q8", "bf16"] {
            let root = flux2_snapshot_root(fixtures, arm.expected_repository, REVISION, tier);
            let (repository, revision, _) = flux2_load_spec_at(
                arm,
                tier,
                &selection,
                arm.expected_repository.to_owned(),
                REVISION.to_owned(),
                root,
            )
            .unwrap_or_else(|error| panic!("{}/{tier}: {error}", arm.model_id));
            assert_eq!(repository, arm.expected_repository);
            assert_eq!(revision, REVISION);
        }
        for other in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
            if other == arm {
                continue;
            }
            let error = flux2_load_spec_at(
                arm,
                "q4",
                &selection,
                other.expected_repository.to_owned(),
                REVISION.to_owned(),
                q4_root.clone(),
            )
            .expect_err("another member's repository must be refused");
            assert!(
                error.contains(arm.expected_repository),
                "{} vs {}: {error}",
                arm.model_id,
                other.model_id
            );
        }
    }
}

fn flux2_snapshot_root(
    fixtures: &crate::adapter_test_support::Fixtures,
    repository: &str,
    revision: &str,
    tier: &str,
) -> PathBuf {
    fixtures.snapshot(repository, revision, Some(tier))
}

#[test]
fn flux2_admission_context_is_reference_free_text_to_image() {
    let contract = mlx_gen_flux2::memory_strategy::registered_dev_t2i_contract(&weights_free_spec(
        Some(Quant::Q4),
    ))
    .unwrap();
    let calibration = contract.calibration.as_ref().unwrap();
    let context = flux2_admission_context(
        FLUX2_DEV_ARM,
        &resident_selection(Some(Quant::Q4)),
        calibration,
        (768, 768),
        Flux2AdmissionProbe {
            fingerprint: &calibration.fingerprint,
            total_bytes: 1_000_000,
            predicted_peak_bytes: 1_000_000,
        },
    );
    assert_eq!(context.mode, MemoryMode::TextToImage);
    assert!(!context.has_reference);
    assert_eq!(context.geometry.reference_count, 0);
    assert!(context.overlay.is_none());
}

#[test]
fn flux2_repeat_envelope_accepts_jitter_and_rejects_the_mandatory_mutation() {
    assert!(flux2_quality_passes(2.0 / 255.0, 0.5 / 255.0, 1.0 / 255.0));
    assert!(!flux2_quality_passes(4.0 / 255.0, 0.5 / 255.0, 1.0 / 255.0));
    assert!(!flux2_quality_passes(2.0 / 255.0, 1.5 / 255.0, 1.0 / 255.0));
    assert!(!flux2_quality_passes(2.0 / 255.0, 0.5 / 255.0, 2.0 / 255.0));

    let baseline = Image {
        width: 2,
        height: 1,
        pixels: vec![0, 63, 127, 128, 191, 255],
    };
    let mutated = qwen_negative_mutation(&baseline);
    let (maximum, mean, rms) = image_max_mean_rms_abs(&mutated, &baseline).unwrap();
    assert!(maximum >= 64.0 / 255.0);
    assert!(mean >= 64.0 / 255.0);
    assert!(rms >= 64.0 / 255.0);
    assert!(!flux2_quality_passes(maximum, mean, rms));
}

#[test]
fn rms_is_measured_from_the_same_pixels_as_max_and_mean() {
    let left = Image {
        width: 2,
        height: 1,
        pixels: vec![0, 0, 0, 0, 0, 0],
    };
    let right = Image {
        width: 2,
        height: 1,
        pixels: vec![51, 0, 0, 0, 0, 0],
    };
    let (maximum, mean, rms) = image_max_mean_rms_abs(&left, &right).unwrap();
    assert!((maximum - 0.2).abs() < 1e-9);
    assert!((mean - 0.2 / 6.0).abs() < 1e-9);
    assert!((rms - 0.2 / 6.0_f64.sqrt()).abs() < 1e-9);
}

/// `pub(super)` so the sibling `ltx_tests` module can pin its own provider's contract
/// shape through the SAME weights-free spec these FLUX.2 pins use (sc-18808 review).
pub(super) fn weights_free_spec(quant: Option<Quant>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization);
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    spec
}

fn resident_selection(quant: Option<Quant>) -> MemorySelection {
    MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: MemoryStrategyParameters::default(),
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant,
            component_precision_floors: &[],
        },
    }
}

/// Pins the arm's load-bearing premises to the PINNED provider crate, weights-free:
///
///   1. `flux2_dev` directly registers its own T2I contract;
///   2. that T2I contract supports Resident and StagedResidency, with deeper rungs `Missing`;
///   3. its calibration fingerprint is the exact string the plan entries pin.
///
/// If a pin bump changes any of these, this test reds and the arm must be revisited rather
/// than silently measuring under a different contract.
#[test]
fn the_pinned_flux2_t2i_contract_supports_staging_and_preserves_the_resident_plan() {
    let registry = mlx_gen_flux2::provider_registry().unwrap();
    let spec = weights_free_spec(Some(Quant::Q4));
    let contract = registry
        .memory_strategy_contract(FLUX2_PROVIDER, &spec)
        .unwrap()
        .expect("the pinned FLUX.2-dev T2I contract");
    assert_eq!(contract.provider_id, FLUX2_PROVIDER);
    for capability in &contract.strategies {
        if matches!(
            capability.strategy,
            MemoryStrategy::Resident | MemoryStrategy::StagedResidency
        ) {
            assert_eq!(
                capability.support,
                MemoryStrategySupport::Implemented,
                "{:?} must be supported",
                capability.strategy
            );
        } else {
            assert!(
                matches!(capability.support, MemoryStrategySupport::Missing),
                "{:?} is no longer Missing at the pin; revisit the capture capabilities",
                capability.strategy
            );
        }
    }
    assert_eq!(
        contract.calibration.as_ref().unwrap().fingerprint,
        FLUX2_CALIBRATION_FINGERPRINT,
        "the plan entries pin this fingerprint; regenerate them with the provider"
    );
    // The plan's `engagedRungs: ["resident"]` must equal the provider's engaged composition,
    // or `attested_strategy` fails every capture at plan/measured comparison time.
    assert_eq!(
        contract.engaged_composition(MemoryStrategy::Resident),
        vec![MemoryStrategy::Resident],
        "the plan entries pin engagedRungs [\"resident\"]; regenerate them with the provider"
    );
    let staged = MemorySelection {
        strategy: MemoryStrategy::StagedResidency,
        ..resident_selection(Some(Quant::Q4))
    };
    contract.validate_selection(&staged).unwrap();
    assert!(contract.generation_memory(&staged).unwrap().stage_residency);
}

/// sc-22727 moved the arm from the crate-local `mlx_gen_flux2::provider_registry()` onto the
/// PRODUCTION catalog the worker composes (`runtime_macos::catalog()`, E4). That is only a
/// safe move if the two resolve the SAME contract — `mlx-gen-catalog` calls the same
/// `register_providers`, but nothing in this repo asserted it, and a bundle that wrapped or
/// re-registered a provider would silently change what every FLUX.2 capture measures.
///
/// Weights-free: both sides are asked for the contract over one placeholder spec, so this
/// costs no GPU and no snapshot, and it covers every member the arm serves.
#[test]
fn the_production_catalog_resolves_the_same_flux2_contracts_as_the_provider_crate() {
    let crate_registry = mlx_gen_flux2::provider_registry().unwrap();
    let catalog = runtime_macos::catalog().expect("the production MLX catalog builds");
    for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
        let spec = weights_free_spec(if arm.tier_quant_reaches_the_loader {
            Some(Quant::Q4)
        } else {
            None
        });
        let direct = crate_registry.memory_strategy_contract(arm.provider, &spec);
        let production = catalog
            .media()
            .memory_strategy_contract(arm.provider, &spec);
        match (direct, production) {
            (Ok(direct), Ok(production)) => assert_eq!(
                direct, production,
                "{}: the production catalog resolves a different contract",
                arm.model_id
            ),
            // A weights-free spec is a placeholder, so a member whose contract wants a real
            // snapshot refuses on BOTH sides. The claim is that they agree, including on the
            // refusal text — that is what rules out a bundle-side wrapper.
            (Err(direct), Err(production)) => assert_eq!(
                direct.to_string(),
                production.to_string(),
                "{}: the two registries refuse for different reasons",
                arm.model_id
            ),
            (direct, production) => panic!(
                "{}: one registry answered and the other did not: {direct:?} vs {production:?}",
                arm.model_id
            ),
        }
    }
}

/// The admission-scenario legs the capture will run, exercised weights-free through the SAME
/// registered function the loaded T2I generator delegates to. Mutation-verified in both
/// directions: the accept leg proves the two rejects are not a blanket refusal, and the
/// route-gate leg proves the accept is not a blanket accept.
#[test]
fn flux2_admission_scenarios_accept_exact_fit_and_reject_unknown_stale_and_foreign_routes() {
    let registry = mlx_gen_flux2::provider_registry().unwrap();
    let spec = weights_free_spec(Some(Quant::Q4));
    let contract = registry
        .memory_strategy_contract(FLUX2_PROVIDER, &spec)
        .unwrap()
        .expect("the pinned FLUX.2-dev T2I contract");
    let calibration = contract.calibration.as_ref().unwrap();
    let selection = resident_selection(Some(Quant::Q4));
    let check = |context: &MemoryRunContext| {
        mlx_gen_flux2::memory_strategy::registered_dev_t2i_safety_check(&spec, &contract, context)
    };

    let exact = flux2_admission_context(
        FLUX2_DEV_ARM,
        &selection,
        calibration,
        (1024, 1024),
        Flux2AdmissionProbe {
            fingerprint: &calibration.fingerprint,
            total_bytes: 1_000_000,
            predicted_peak_bytes: 1_000_000,
        },
    );
    assert!(
        matches!(check(&exact), MemorySafetyDecision::Accept),
        "exact-fit admission must accept: {:?}",
        check(&exact)
    );

    let unknown = flux2_admission_context(
        FLUX2_DEV_ARM,
        &selection,
        calibration,
        (1024, 1024),
        Flux2AdmissionProbe {
            fingerprint: &calibration.fingerprint,
            total_bytes: 0,
            predicted_peak_bytes: 1,
        },
    );
    assert!(matches!(
        check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ));

    let stale = flux2_admission_context(
        FLUX2_DEV_ARM,
        &selection,
        calibration,
        (1024, 1024),
        Flux2AdmissionProbe {
            fingerprint: "stale-flux2-dev-fingerprint",
            total_bytes: 1_000_000,
            predicted_peak_bytes: 1,
        },
    );
    assert!(matches!(check(&stale), MemorySafetyDecision::Reject { .. }));

    // The route-gate mutation: the same fitting budget in an edit shape must be refused — the
    // T2I contract admits only reference-free text-to-image.
    let mut edit_shaped = flux2_admission_context(
        FLUX2_DEV_ARM,
        &selection,
        calibration,
        (1024, 1024),
        Flux2AdmissionProbe {
            fingerprint: &calibration.fingerprint,
            total_bytes: 1_000_000,
            predicted_peak_bytes: 1_000_000,
        },
    );
    edit_shaped.mode = MemoryMode::Edit;
    edit_shaped.has_reference = true;
    edit_shaped.geometry.reference_count = 2;
    assert!(matches!(
        check(&edit_shaped),
        MemorySafetyDecision::Reject { .. }
    ));

    // NVFP4 is the one tier the route gate refuses by name.
    let nvfp4_spec = weights_free_spec(Some(Quant::Nvfp4));
    let nvfp4 = flux2_admission_context(
        FLUX2_DEV_ARM,
        &resident_selection(Some(Quant::Nvfp4)),
        calibration,
        (1024, 1024),
        Flux2AdmissionProbe {
            fingerprint: &calibration.fingerprint,
            total_bytes: 1_000_000,
            predicted_peak_bytes: 1_000_000,
        },
    );
    assert!(matches!(
        mlx_gen_flux2::memory_strategy::registered_dev_t2i_safety_check(
            &nvfp4_spec,
            &contract,
            &nvfp4
        ),
        MemorySafetyDecision::Reject { .. }
    ));
}
