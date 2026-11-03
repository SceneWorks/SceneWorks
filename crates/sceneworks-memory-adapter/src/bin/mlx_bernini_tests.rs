use super::*;
use runtime_macos::providers::bernini::memory_strategy as engine;

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn advertised() -> (u32, u32) {
    engine::ADVERTISED_GEOMETRIES[0]
}

/// A plan row for one member at one tier on the engine's first advertised canvas.
fn request(arm: BerniniArm, tier: &str) -> Value {
    let (width, height) = advertised();
    let frames = arm.frames;
    json!({
        "hardware": {
            "memoryBytes": 137_438_953_472_u64,
            "wiredLimitBytes": 96_000_000_000_u64,
        },
        "planned": {
            "target": {
                "provider": BERNINI_PROVIDER,
                "modelId": arm.model_id,
                "tier": tier,
                "mode": arm.mode,
                "overlay": "none",
                "geometry": { "width": width, "height": height, "batch": 1, "frames": frames }
            },
            "backend": "mlx",
            "loadShape": protocol::LOAD_SHAPE_EAGER,
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": bernini_calibration_fingerprint(tier)
                .expect("a shipped Bernini tier"),
            "fixture": format!(
                "bernini-{}-mlx-{tier}-{width}x{height}-f{frames}-fps{BERNINI_FPS}-seed{BERNINI_SEED}",
                arm.slug
            )
        }
    })
}

/// The identity family is the ENGINE's, byte for byte: the provider id, the calibrated tier
/// and the route token are all read off the pinned crate, and the retained key is the
/// `(route, CALIBRATED_TIER)` member the format mints rather than a special case.
#[test]
fn the_bernini_identity_family_reproduces_the_engines_retained_key() {
    assert_eq!(
        BERNINI_PROVIDER,
        engine::FULL_ID,
        "the provider id is the engine's"
    );
    assert_eq!(BERNINI_CALIBRATED_TIER, engine::CALIBRATED_TIER);
    assert!(BERNINI_TIERS.contains(&BERNINI_CALIBRATED_TIER));
    let route = engine::calibration_route(engine::FULL_ID)
        .expect("the engine mints a route token for the full pipeline");
    let renderer_route = engine::calibration_route(engine::RENDERER_ID)
        .expect("the engine mints a route token for the renderer-only sibling");
    assert_ne!(
        route, renderer_route,
        "the route token discriminates the two provider ids"
    );
    let retained = engine::MEMORY_CALIBRATION_FINGERPRINT;
    assert_eq!(
        bernini_calibration_fingerprint(BERNINI_CALIBRATED_TIER).unwrap(),
        retained,
        "the calibrated cell must reproduce the engine's retained key byte-for-byte"
    );
    let (stem, version) = retained
        .rsplit_once("-mlx-dual-expert-ladder-")
        .unwrap_or_else(|| panic!("{retained} is not a Bernini dual-expert-ladder key"));
    assert_eq!(
        stem,
        format!("bernini-{route}-{BERNINI_CALIBRATED_TIER}"),
        "the retained key is the (route, calibrated tier) member of the family: {retained}"
    );
    let mut identities = std::collections::BTreeSet::new();
    for tier in BERNINI_TIERS {
        let minted = bernini_calibration_fingerprint(tier).unwrap();
        assert_eq!(
            minted,
            format!("bernini-{route}-{tier}-mlx-dual-expert-ladder-{version}"),
            "the {tier} cell must be the engine's stem, route and version with its own tier"
        );
        assert!(
            identities.insert(minted),
            "{tier} reuses another tier's identity"
        );
    }
    assert_eq!(identities.len(), BERNINI_TIERS.len());
}

/// A tier the family does not ship has no identity and is refused BY NAME — from the family
/// directly, and through the arm before any environment work.
#[test]
fn an_unshipped_tier_is_refused_by_name() {
    let error = bernini_calibration_fingerprint("fp8").unwrap_err();
    assert!(
        error.contains("\"fp8\"") && error.contains("bf16"),
        "{error}"
    );
    let mut request = request(BERNINI_VIDEO_ARM, "q4");
    request["planned"]["target"]["tier"] = json!("fp8");
    let error = run_bernini(&request).expect_err("an unshipped tier has no cell");
    assert!(error.contains("\"fp8\""), "{error}");
    assert!(!error.contains("SCENEWORKS_"), "{error}");
}

/// A plan row naming ANOTHER TIER's identity — a plausible member of this family — is refused
/// before any environment work, naming both keys, so a copied row cannot re-label one tier's
/// peaks as another's. Both members.
#[test]
fn a_plan_naming_another_tiers_identity_is_refused_before_any_environment_work() {
    // A q8 row carrying the CALIBRATED tier's retained key is the copy a reader is likeliest
    // to make, and the one a single-literal arm would wave through.
    for arm in BERNINI_ARMS {
        let mut request = request(arm, "q8");
        request["planned"]["calibrationFingerprint"] =
            json!(bernini_calibration_fingerprint(BERNINI_CALIBRATED_TIER).unwrap());
        let error = run_bernini(&request).expect_err("a q8 row may not carry the retained key");
        assert!(
            error.starts_with("plan/adapter calibration mismatch"),
            "{error}"
        );
        assert!(
            error.contains(&bernini_calibration_fingerprint("q8").unwrap()),
            "{error}"
        );
        assert!(
            error.contains(&bernini_calibration_fingerprint(BERNINI_CALIBRATED_TIER).unwrap()),
            "{error}"
        );
        assert!(!error.contains("SCENEWORKS_"), "{}: {error}", arm.model_id);
    }
}

/// sc-22738 — the video member's rendered frame count is the ENGINE's decoded depth, not the
/// requested count.
///
/// Bernini decodes through the same NON-causal Wan z16 VAE that made the campaign's SCAIL-2
/// anchor render 77 requested frames as 80 and then refused the capture after the render had
/// completed. This arm carried the identical equality against a 49-frame request, which the
/// engine decodes to 52. The expectation is bound to the engines' OWN symbols — `latent_shape`
/// for the latent depth (Bernini calls exactly that, through `PROVIDER_VAE_STRIDE`, which is
/// derived from this same `VAE_TILING`) and gen-core's `TilingConfig::plan` for `out_f` — so a
/// pin that changed either reds here in milliseconds instead of after a multi-hour render.
///
/// Mutations that fail this: returning `geometry.frames` from [`bernini_rendered_frame_count`];
/// flipping the causal branch in `protocol::vae_decoded_frame_count`.
#[test]
fn the_video_members_rendered_frame_count_is_the_engines_own_decoded_depth() {
    let (width, height) = advertised();
    let vae = bernini_engine_vae().expect("the pinned Bernini provider publishes a VAE");
    assert!(
        !vae.causal_temporal,
        "Bernini's z16 decode is non-causal at this pin"
    );
    let geometry = BerniniGeometry {
        width,
        height,
        frames: BERNINI_VIDEO_FRAMES,
    };
    let latent = mlx_gen_wan::pipeline::latent_shape(
        geometry.frames as usize,
        geometry.height,
        geometry.width,
        1,
        (
            vae.temporal_scale as usize,
            vae.spatial_scale as usize,
            vae.spatial_scale as usize,
        ),
    )
    .expect("the wan latent rule accepts Bernini's planned frame count");
    let engine_out_f = TilingConfig {
        spatial: None,
        temporal: None,
    }
    .plan(vae, latent[1], latent[2], latent[3])
    .out_f;
    assert_eq!(
        i64::from(bernini_rendered_frame_count(geometry).unwrap()),
        i64::from(engine_out_f)
    );
    // The concrete consequence, so the rule cannot silently become an identity: the non-causal
    // decode over-delivers by one temporal stride minus one.
    assert_eq!(
        bernini_rendered_frame_count(geometry).unwrap(),
        geometry.frames + 3
    );
}

/// The still member is the engine's own exception, and it is keyed on the engine's own
/// predicate. `bernini.rs`'s `generate_impl` takes the FIRST image of the decoded temporal
/// chunk when `frames == 1`, so a one-frame request renders exactly one frame even though the
/// same non-causal VAE decodes that single latent frame to four.
///
/// Mutation that fails this: dropping the `frames == 1` branch from
/// [`bernini_rendered_frame_count`] so the still member expects the untrimmed chunk.
#[test]
fn the_still_member_renders_the_one_frame_the_engine_returns() {
    let (width, height) = advertised();
    let vae = bernini_engine_vae().unwrap();
    let geometry = BerniniGeometry {
        width,
        height,
        frames: 1,
    };
    assert_eq!(bernini_rendered_frame_count(geometry).unwrap(), 1);
    // ... and the shared rule alone would NOT say 1 here, which is why the branch exists.
    assert_eq!(
        protocol::vae_decoded_frame_count(1, vae.temporal_scale, vae.causal_temporal).unwrap(),
        u32::try_from(vae.temporal_scale).unwrap()
    );
}

fn snapshot_root(tier: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir()
        .join(format!("sc-22737-bernini-{}-{nonce}", std::process::id()))
        .join(format!(
            "models--{}",
            protocol::BERNINI_REPOSITORY.replace('/', "--")
        ))
        .join("snapshots")
        .join(REVISION)
        .join(tier);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// The tier is the PLAN's, the root must carry it, and the `LoadSpec` binds the root with NO
/// load-time `quantize` (the tier is proven by the artifact's own sidecars) — a q4 plan against
/// a q8 export is refused NAMING the tier it wanted, the sc-17097 defect class.
#[test]
fn a_root_of_another_tier_is_refused_naming_the_planned_tier() {
    let q8_root = snapshot_root("q8");
    let error = bernini_load_spec_at(
        &request(BERNINI_VIDEO_ARM, "q4"),
        "q4",
        LoadShape::EagerMaterialization,
        protocol::BERNINI_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        q8_root.clone(),
    )
    .err()
    .expect("a q4 plan must not be satisfied by a q8 root");
    assert!(
        error.ends_with(&format!("/snapshots/{REVISION}/q4")),
        "{error}"
    );
    for tier in BERNINI_TIERS {
        let root = snapshot_root(tier);
        let artifact = bernini_load_spec_at(
            &request(BERNINI_IMAGE_ARM, tier),
            tier,
            LoadShape::EagerMaterialization,
            protocol::BERNINI_REPOSITORY.to_owned(),
            REVISION.to_owned(),
            root.clone(),
        )
        .unwrap_or_else(|error| panic!("{tier}: {error}"));
        assert_eq!(
            artifact.tier_root,
            std::fs::canonicalize(&root).unwrap(),
            "{tier}"
        );
        assert!(
            matches!(&artifact.spec.weights, WeightsSource::Dir(dir) if *dir == artifact.tier_root),
            "{tier}: the spec loads the resolved tier directory"
        );
        assert_eq!(
            artifact.spec.quantize, None,
            "{tier}: a pre-packed tier loads with quant = None"
        );
        assert_eq!(artifact.repository, protocol::BERNINI_REPOSITORY);
    }
    let error = bernini_load_spec_at(
        &request(BERNINI_VIDEO_ARM, "q4"),
        "q4",
        LoadShape::EagerMaterialization,
        "SceneWorks/not-bernini".to_owned(),
        REVISION.to_owned(),
        q8_root,
    )
    .err()
    .expect("a foreign repository is refused by name");
    assert!(error.contains("SceneWorks/not-bernini"), "{error}");
}

/// The video/still split is resolved from `(provider, modelId)` — one provider, two catalog
/// entries filed separately — and every other axis of the target is refused by name.
#[test]
fn the_member_is_resolved_from_provider_and_model_id() {
    let (width, height) = advertised();
    let (arm, geometry) = bernini_target(&request(BERNINI_VIDEO_ARM, "q4")).unwrap();
    assert_eq!(arm, BERNINI_VIDEO_ARM);
    assert_eq!(
        geometry,
        BerniniGeometry {
            width,
            height,
            frames: BERNINI_VIDEO_FRAMES
        }
    );
    let (arm, geometry) = bernini_target(&request(BERNINI_IMAGE_ARM, "q4")).unwrap();
    assert_eq!(arm, BERNINI_IMAGE_ARM);
    assert_eq!(
        geometry,
        BerniniGeometry {
            width,
            height,
            frames: 1
        }
    );

    // The member's frame count is its own: a one-frame video row and a 49-frame still row are
    // both refused naming the member and the count it renders.
    for (arm, frames) in [
        (BERNINI_VIDEO_ARM, 1),
        (BERNINI_IMAGE_ARM, BERNINI_VIDEO_FRAMES),
    ] {
        let mut request = request(arm, "q4");
        request["planned"]["target"]["geometry"]["frames"] = json!(frames);
        let error = bernini_target(&request).expect_err("the member's frame count is fixed");
        assert!(
            error.contains(arm.model_id) && error.contains(&format!("frames {frames}")),
            "{error}"
        );
    }
    // The mode is the member's evidence key, so swapping the two is refused.
    let mut swapped = request(BERNINI_IMAGE_ARM, "q4");
    swapped["planned"]["target"]["mode"] = json!(BERNINI_VIDEO_ARM.mode);
    let error = bernini_target(&swapped).expect_err("a still row is filed under its own mode");
    assert!(
        error.contains(BERNINI_IMAGE_MODEL_ID) && error.contains(BERNINI_VIDEO_ARM.mode),
        "{error}"
    );
    // A foreign provider and an unknown entry are refused by name — the entry refusal lists
    // exactly the two members this arm serves.
    let mut foreign = request(BERNINI_VIDEO_ARM, "q4");
    foreign["planned"]["target"]["provider"] = json!("wan2_2_t2v_14b");
    let error = bernini_target(&foreign).expect_err("a foreign provider is refused");
    assert!(error.contains("wan2_2_t2v_14b"), "{error}");
    let mut unknown = request(BERNINI_VIDEO_ARM, "q4");
    unknown["planned"]["target"]["modelId"] = json!("bernini_renderer");
    let error = bernini_target(&unknown).expect_err("an unknown entry is refused");
    assert!(
        error.contains("bernini_renderer")
            && error.contains(BERNINI_VIDEO_MODEL_ID)
            && error.contains(BERNINI_IMAGE_MODEL_ID),
        "{error}"
    );
    // A canvas outside the engine's advertised set is refused for either member.
    let mut off_canvas = request(BERNINI_VIDEO_ARM, "q4");
    off_canvas["planned"]["target"]["geometry"]["width"] = json!(width + 8);
    let error = bernini_target(&off_canvas).expect_err("the canvas is the engine's");
    assert!(error.contains("advertised geometries"), "{error}");
}

/// The shipped plan names, for every `mlx` row of either member, exactly the identity this
/// arm mints for the row's tier, under the member's own mode and frame count.
#[test]
fn every_planned_bernini_mlx_row_names_the_identity_this_arm_mints_for_its_tier() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan is valid JSON");
    let mut planned = std::collections::BTreeSet::new();
    for (key, row) in plan["anchors"].as_object().expect("anchors is an object") {
        let parts: Vec<&str> = key.split(':').collect();
        if parts.len() != 3 || parts[2] != "mlx" {
            continue;
        }
        let Some(arm) = BERNINI_ARMS.iter().find(|arm| arm.model_id == parts[0]) else {
            continue;
        };
        let tier = parts[1];
        assert_eq!(row["provider"].as_str(), Some(BERNINI_PROVIDER), "{key}");
        assert_eq!(row["mode"].as_str(), Some(arm.mode), "{key}");
        assert_eq!(
            row["geometry"]["frames"].as_u64(),
            Some(u64::from(arm.frames)),
            "{key}"
        );
        assert_eq!(
            row["calibrationFingerprint"]
                .as_str()
                .expect("fingerprint is a string"),
            bernini_calibration_fingerprint(tier).unwrap(),
            "{key} names an identity the pinned provider does not mint for tier {tier}"
        );
        assert!(
            planned.insert((arm.model_id, tier.to_owned())),
            "{key} is planned twice"
        );
    }
    let expected: std::collections::BTreeSet<(&str, String)> = BERNINI_ARMS
        .iter()
        .flat_map(|arm| BERNINI_TIERS.map(|tier| (arm.model_id, tier.to_owned())))
        .collect();
    assert_eq!(
        planned, expected,
        "every member has one planned MLX row per shipped tier"
    );
}

/// The WORKER's `bernini_memory_attempt`, constructed out of the worker's own source rather
/// than restated here (sc-22738).
///
/// A second hand-written copy of the predicate would agree with the arm by construction and
/// prove nothing — the whole claim under test is that the arm and PRODUCTION take the same
/// decision, so the production side has to come from production. `sceneworks-worker` is not a
/// dependency of this crate (and making the whole worker a dev-dependency to reach one private
/// `fn` would be a build-time cost out of all proportion), so the predicate is read off
/// `crates/sceneworks-worker/src/video_admission.rs` at test time — the same workspace-source
/// technique `every_cited_workspace_source_path_exists` uses. A worker that widens its admitted
/// route list therefore widens this test's expectation with it.
struct WorkerBerniniAttempt {
    /// The one catalog id the worker's predicate names.
    model_id: String,
    /// The video routes it admits without a reference.
    modes: Vec<String>,
    /// Whether any non-zero reference count alone makes it an attempt.
    any_reference: bool,
}

impl WorkerBerniniAttempt {
    fn read() -> Self {
        fn quoted(source: &str) -> Vec<String> {
            source
                .split('"')
                .skip(1)
                .step_by(2)
                .map(str::to_owned)
                .collect()
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("the workspace root is two levels above this crate");
        let path = root.join("crates/sceneworks-worker/src/video_admission.rs");
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let start = source
            .find("fn bernini_memory_attempt(")
            .expect("the worker still defines bernini_memory_attempt");
        let open = source[start..].find('{').expect("the predicate has a body") + start;
        let end = source[open..].find("\n}\n").expect("the body closes") + open;
        let body = &source[open..end];
        // The axes this extraction knows how to evaluate. A predicate that starts reading
        // another one must be re-read here rather than silently half-modelled.
        for axis in [
            "reference_shape",
            "lane",
            "tier",
            "overlay",
            "fps",
            "frames",
        ] {
            assert!(
                !body.contains(axis),
                "bernini_memory_attempt now reads `{axis}`, which this extraction does not \
                     model: {body}"
            );
        }
        let model_clause = &body[body
            .find("request.model_id ==")
            .expect("the predicate still binds a catalog id")..];
        let model_id = quoted(model_clause)
            .first()
            .cloned()
            .expect("the catalog id is a string literal");
        let mode_clause = &body[body
            .find("request.mode,")
            .expect("the predicate still matches on the request mode")..];
        let modes = quoted(&mode_clause[..mode_clause.find(')').expect("the match closes")]);
        let any_reference = body.contains("request.reference_count > 0");
        assert!(!model_id.is_empty());
        assert!(
            modes.len() > 1 && {
                let mut unique = modes.clone();
                unique.sort();
                unique.dedup();
                unique.len() == modes.len()
            },
            "the extracted route list is degenerate: {modes:?}"
        );
        assert!(
            any_reference,
            "the reference-count disjunct is gone from bernini_memory_attempt: {body}"
        );
        Self {
            model_id,
            modes,
            any_reference,
        }
    }

    fn decides(&self, model_id: &str, mode: &str, reference_count: u32) -> bool {
        model_id == self.model_id
            && (self.modes.iter().any(|admitted| admitted == mode)
                || (self.any_reference && reference_count > 0))
    }
}

/// sc-22738. THE decision this fix is about: for every planned `mlx:bernini*` cell the arm's
/// answer to "does this capture ask Bernini's admission gate anything?" must be the answer
/// PRODUCTION gives for the same request shape.
///
/// sc-22737 asked the gate on the anchor's own reference-free surface, which
/// `mlx-gen-bernini`'s route gate refuses by name, and all six anchors died with "admission
/// rejected a fitting probe budget" after a full dual-expert weight load. The remedy is not a
/// surface that makes the gate answer — production never presents one for this request — it is
/// to skip the probe exactly where production skips the context.
///
/// The agreement is checked well past today's six cells: over the worker's whole admitted route
/// list, both catalog members and both reference counts, so a worker that adds a route cannot
/// leave the arm's mirror behind.
#[test]
fn the_arms_admission_decision_is_the_workers_for_every_planned_bernini_cell() {
    let worker = WorkerBerniniAttempt::read();
    for arm in BERNINI_ARMS {
        for tier in BERNINI_TIERS {
            let label = format!("{}:{tier}:mlx", arm.model_id);
            let request = request(arm, tier);
            let (resolved, _) = bernini_target(&request).expect("the fixture is this arm's");
            let reference_count =
                bernini_reference_count(&request).expect("the fixture declares a count");
            assert_eq!(
                bernini_probes_admission(resolved.model_id, resolved.mode, reference_count),
                worker.decides(resolved.model_id, resolved.mode, reference_count),
                "{label}: the arm and the worker disagree about whether this request has an \
                     admission surface at all"
            );
            // And today that shared answer is NO — the fixture is reference-free on both
            // members, so the capture prices the resident load and asks the gate nothing.
            assert_eq!(reference_count, 0, "{label}");
            assert!(
                !bernini_probes_admission(resolved.model_id, resolved.mode, reference_count),
                "{label}: a reference-free Bernini capture must not probe admission"
            );
        }
    }
    // The mirror, not just the six cells: every route the worker admits, every route it does
    // not, both members, with and without a reference.
    let modes: Vec<&str> = worker
        .modes
        .iter()
        .map(String::as_str)
        .chain(["text_to_video", "text_to_image", "image_to_video"])
        .collect();
    for arm in BERNINI_ARMS {
        for mode in &modes {
            for reference_count in [0, 1, 4] {
                assert_eq!(
                    bernini_probes_admission(arm.model_id, mode, reference_count),
                    worker.decides(arm.model_id, mode, reference_count),
                    "{}/{mode}/refs={reference_count}: the arm's mirror of \
                         bernini_memory_attempt has drifted",
                    arm.model_id
                );
            }
        }
    }

    // …and the arm ACTS on that decision. The predicate above is only the answer; a body that
    // still called the provider's admission gate, or a record that still claimed the three
    // admission scenarios passed, would be the sc-22737 defect wearing the sc-22738 predicate.
    // The measured render needs real weights and cannot run here, so the body is read the way
    // `every_receipt_builder_is_bound_to_its_lane_prediction_wrapper` reads its bindings.
    let source = include_str!("mlx.rs");
    let start = source
        .find("\nfn run_bernini(")
        .expect("the arm still exists");
    let body = &source[start
        ..start
            + source[start..]
                .find("\n}\n")
                .expect("the arm's body closes")];
    assert!(
        !body.contains("safety_check"),
        "the Bernini arm asks the provider's admission gate something again; production asks \
             it nothing for a reference-free request"
    );
    assert!(
        body.contains("\"status\": \"gated\""),
        "a capture that ran no admission probe cannot file a runtime-activating record"
    );
    for scenario in ["exact_fit", "unknown_budget", "stale_evidence"] {
        assert!(
            body.contains(&format!(
                "{{ \"name\": \"{scenario}\", \"result\": \"not_run\", \"reason\": \
                     BERNINI_ADMISSION_BLOCKER }}"
            )),
            "the {scenario} scenario must be reported unexecuted, with the reason production \
                 gives for it"
        );
    }
    assert!(
        BERNINI_ADMISSION_BLOCKER.contains("reference-free Bernini has NO admission surface")
            && BERNINI_ADMISSION_BLOCKER.contains("prices the RESIDENT load"),
        "the record must state why it ran no probe and what it does price: \
             {BERNINI_ADMISSION_BLOCKER}"
    );
}

/// The load spec the arm builds for a tier, minus the snapshot, against a path that does not
/// exist: `weights_free_memory_strategy_contract` charges zero measured bytes and falls back to
/// the architecture defaults for an unreadable root, so this stays a CPU test with no weights
/// and no Metal.
fn probe_spec(selection: &MemorySelection) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(
        "/nonexistent/sc-22738/bernini",
    )))
    .with_offload_policy(OffloadPolicy::Resident)
    .with_load_shape(LoadShape::EagerMaterialization);
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    spec
}

/// sc-22738. The other half of the claim: the arm skips the probe because THIS REQUEST has no
/// admission surface, not because the seam is broken or unreachable.
///
/// A reference-carrying V2V request — one source clip, its `provider_video_mode:v2v` axis, the
/// single-phase trajectory Bernini actually runs, one of the frame counts both sides admit — is
/// a request production DOES build a context for (`bernini_memory_attempt` answers true) and
/// the engine's own `loaded_safety_check` accepts. The surface is spelled here, in the test,
/// rather than in the arm: it is the fixture of this seam exercise, and an arm that carried it
/// would be able to present it for a request that has no reference — which is exactly the
/// defect this story removes.
#[test]
fn a_reference_carrying_bernini_request_would_build_a_context_the_engine_admits() {
    /// The provider task axis the engine's route gate requires for V2V, and the axis
    /// `video_admission.rs#bernini_surface_is_exact` expects alongside `video_to_video`.
    const V2V_OVERLAY: &str = "provider_video_mode:v2v";
    const V2V_MODE: &str = "video_to_video";
    /// V2V carries exactly one source clip.
    const V2V_REFERENCE_COUNT: u32 = 1;
    /// One of the frame counts both gates admit (45 / 61 / 77).
    const V2V_FRAMES: u32 = 45;

    let worker = WorkerBerniniAttempt::read();
    assert!(
        worker.decides(BERNINI_VIDEO_MODEL_ID, V2V_MODE, V2V_REFERENCE_COUNT),
        "the worker must build a memory context for a reference-carrying V2V request, or the \
             arm's skip is not a property of the request"
    );

    let (width, height) = advertised();
    let selection =
        planned_selection(&request(BERNINI_VIDEO_ARM, "q4")).expect("the fixture plans a rung");
    let spec = probe_spec(&selection);
    let contract = engine::weights_free_memory_strategy_contract(engine::FULL_ID, &spec)
        .expect("the pinned crate publishes a weights-free contract");
    let calibration = contract
        .calibration
        .clone()
        .expect("the weights-free contract carries a calibration identity");
    let context = |overlay: Option<&str>| MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other(V2V_MODE.to_owned()),
        has_reference: true,
        use_pid: false,
        // Phase-resolved evidence is not a multi-phase request modifier; the worker's own
        // `MemoryRunContext` literal carries this false and Bernini runs a single-phase denoise.
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: V2V_FRAMES,
            reference_count: V2V_REFERENCE_COUNT,
        },
        reference_resources: None,
        overlay: overlay.map(str::to_owned),
        budget: MemoryBudget {
            total_bytes: 137_438_953_472,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22738@{}", protocol::INFERENCE_PIN),
    };
    let expert_blocks = engine::trunk_blocks() / 2;
    let check = |context: &MemoryRunContext| {
        engine::loaded_safety_check(engine::FULL_ID, &spec, &contract, context, expert_blocks)
    };
    assert!(
        matches!(
            check(&context(Some(V2V_OVERLAY))),
            MemorySafetyDecision::Accept
        ),
        "the engine refused a reference-carrying V2V request it admits in production: {:?}",
        check(&context(Some(V2V_OVERLAY)))
    );
    // The accept is not blanket: the same gate rejects the same request without the provider
    // task axis, and rejects the reference-free surface sc-22737 probed.
    assert!(
        matches!(check(&context(None)), MemorySafetyDecision::Reject { .. }),
        "a V2V request with no provider task axis must stay a rejection"
    );
    let mut reference_free = context(Some(V2V_OVERLAY));
    reference_free.has_reference = false;
    reference_free.geometry.reference_count = 0;
    reference_free.mode = MemoryMode::Other(BERNINI_VIDEO_ARM.mode.to_owned());
    reference_free.overlay = None;
    assert!(
        matches!(check(&reference_free), MemorySafetyDecision::Reject { .. }),
        "the reference-free surface this arm renders must stay outside Bernini admission — it \
             is why the arm asks the gate nothing"
    );
}
