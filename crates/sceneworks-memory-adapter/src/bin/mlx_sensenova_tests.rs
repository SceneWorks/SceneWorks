use super::*;

fn sensenova_planned(provider: &str, model_id: &str, tier: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": model_id,
                "tier": tier,
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": "eager_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": "unused"
        }
    })
}

/// The member is resolved from the PAIR, and every per-member field is unique across the
/// family — two members sharing an env family, a repository or a slug would let one member's
/// capture be recorded as another's.
#[test]
fn sensenova_arm_is_resolved_from_the_plans_provider_and_model_id() {
    for arm in SENSENOVA_FAMILY {
        let resolved = sensenova_arm(&sensenova_planned(arm.provider, arm.model_id, "q4")).unwrap();
        assert_eq!(resolved, arm);
    }
    for field in [
        SENSENOVA_FAMILY.map(|arm| arm.model_id).to_vec(),
        SENSENOVA_FAMILY.map(|arm| arm.slug).to_vec(),
        SENSENOVA_FAMILY.map(|arm| arm.repository_env).to_vec(),
        SENSENOVA_FAMILY.map(|arm| arm.revision_env).to_vec(),
        SENSENOVA_FAMILY.map(|arm| arm.root_env).to_vec(),
        SENSENOVA_FAMILY.map(|arm| arm.expected_repository).to_vec(),
        SENSENOVA_FAMILY.map(|arm| arm.execution_path).to_vec(),
        SENSENOVA_FAMILY.map(|arm| arm.still_calibration).to_vec(),
    ] {
        let mut unique = field.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), field.len(), "collision in {field:?}");
    }
    // A model id no member serves is refused BY NAME, never measured as its neighbour.
    for absent in ["sensenova_u1_8b_infographic_v4", "z_image", "qwen_image"] {
        let error = sensenova_arm(&sensenova_planned(SENSENOVA_PROVIDER, absent, "q4"))
            .expect_err("an unserved model id must be refused");
        assert!(error.contains(&format!("modelId {absent:?}")), "{error}");
    }
    // And so is a pair that crosses the quality/fast split: the distilled trio is a separate
    // engine over separate weights, not a mode of the quality one.
    let crossed = sensenova_arm(&sensenova_planned(
        SENSENOVA_FAST_PROVIDER,
        "sensenova_u1_8b",
        "q4",
    ))
    .expect_err("a crossed (provider, modelId) pair must be refused");
    assert!(crossed.contains("sensenova_u1_8b"), "{crossed}");
    assert!(crossed.contains(SENSENOVA_FAST_PROVIDER), "{crossed}");
}

/// The spec is the shape the WORKER loads: resident, eagerly materialized, the resolved route
/// bound, and the tier's own quant (none at bf16).
#[test]
fn sensenova_load_spec_binds_the_worker_shape_and_the_planned_tier() {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    for arm in SENSENOVA_FAMILY {
        for (tier, expected) in [
            ("bf16", None),
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
        ] {
            let repository = arm.expected_repository;
            let root = PathBuf::from(format!(
                "/cache/models--{}/snapshots/{revision}/{tier}",
                repository.replace('/', "--")
            ));
            let spec =
                sensenova_load_spec_at(arm, repository, revision, root.clone(), tier).unwrap();
            assert_eq!(spec.offload_policy, OffloadPolicy::Resident, "{repository}");
            assert_eq!(
                spec.load_shape,
                LoadShape::EagerMaterialization,
                "{repository}"
            );
            assert_eq!(spec.quantize, expected, "{repository}:{tier}");
            assert_eq!(spec.resolved_route.as_deref(), Some(arm.model_id));
            assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if dir == &root));
            assert!(spec.adapters.is_empty(), "SenseNova attaches no adapters");
            assert!(spec.components.is_empty(), "the distill LoRA is pre-merged");
            assert!(
                spec.text_encoder.is_none(),
                "the MoT backbone has no external TE"
            );
        }
        // A root at another tier cannot satisfy this plan, and neither can another repository.
        let root = PathBuf::from(format!(
            "/cache/models--{}/snapshots/{revision}/q4",
            arm.expected_repository.replace('/', "--")
        ));
        assert!(
            sensenova_load_spec_at(arm, arm.expected_repository, revision, root.clone(), "q8")
                .is_err(),
            "a q4 root must not satisfy a q8 plan"
        );
        assert!(
            sensenova_load_spec_at(arm, "sensenova/SenseNova-U1-8B-MoT", revision, root, "q4")
                .is_err(),
            "the upstream snapshot must not satisfy the tiered rehost plan"
        );
    }
}

/// Every SenseNova cell the committed plan declares resolves to an implemented member, and its
/// fixture satisfies the same binding a capture would apply. The expected set is DERIVED from
/// the family and the shipped tiers, never a count.
#[test]
fn every_planned_sensenova_mlx_cell_resolves_to_an_implemented_member() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut seen = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":mlx")
            || !matches!(provider, SENSENOVA_PROVIDER | SENSENOVA_FAST_PROVIDER)
        {
            continue;
        }
        seen.insert(key.clone());
        let model_id = key.split(':').next().unwrap();
        let tier = key.split(':').nth(1).unwrap();
        let request = json!({ "planned": {
            "target": {
                "provider": provider,
                "modelId": model_id,
                "tier": tier,
                "mode": entry["mode"].clone(),
                "overlay": entry["overlay"].clone(),
                "geometry": entry["geometry"].clone(),
            },
            "loadShape": entry["loadShape"].clone(),
            "fixture": entry["fixture"].clone(),
        }});
        let arm = sensenova_arm(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        validate_sensenova_fixture(&request, arm, tier)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(
            planned_load_shape(&request).unwrap(),
            LoadShape::EagerMaterialization,
            "{key}: the worker loads this family eagerly; deferred is the rung-4 shape"
        );
    }
    let expected: std::collections::BTreeSet<String> = SENSENOVA_FAMILY
        .iter()
        .flat_map(|arm| {
            ["bf16", "q4", "q8"]
                .iter()
                .map(move |tier| format!("{}:{tier}:mlx", arm.model_id))
        })
        .collect();
    assert_eq!(
        seen, expected,
        "every shipped SenseNova MLX cell, exactly once"
    );
}

/// Every SenseNova MLX plan row names the production identity its loaded generator publishes.
/// The two measured keys are the engine's own constants byte-for-byte, no row names the
/// weights-free conformance namespace, and every cell's identity is distinct — a shared string
/// would let one cell's anchor bind another's measured evidence.
#[test]
fn every_planned_sensenova_mlx_cell_names_the_identity_its_loaded_generator_publishes() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut identities = std::collections::BTreeMap::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":mlx")
            || !matches!(provider, SENSENOVA_PROVIDER | SENSENOVA_FAST_PROVIDER)
        {
            continue;
        }
        let model_id = key.split(':').next().unwrap();
        let tier = key.split(':').nth(1).unwrap();
        let arm = sensenova_arm(&sensenova_planned(provider, model_id, tier)).unwrap();
        let planned = entry["calibrationFingerprint"]
            .as_str()
            .unwrap_or_else(|| panic!("{key}: calibrationFingerprint must be a string"));
        assert!(
            !planned.starts_with("sensenova-static-registry-behavior"),
            "{key}: {planned} is the weights-free registry-behaviour identity, which no \
                 production load returns"
        );
        assert_eq!(
            planned,
            sensenova_calibration_fingerprint(arm, tier),
            "{key}: the plan row must name the loaded generator's production identity"
        );
        assert!(
            identities.insert(planned.to_owned(), key.clone()).is_none(),
            "{key}: {planned} is already claimed by {}",
            identities[planned]
        );
    }
    assert_eq!(
        identities.len(),
        SENSENOVA_FAMILY.len() * 3,
        "one distinct identity per (route, shipped tier)"
    );
    // The preserved measured keys, bound to the engine's constants so the table cannot drift.
    assert_eq!(
        identities
            [runtime_macos::providers::sensenova::memory_strategy::QUALITY_CALIBRATION_FINGERPRINT],
        "sensenova_u1_8b:q8:mlx"
    );
    assert_eq!(
        identities
            [runtime_macos::providers::sensenova::memory_strategy::FAST_CALIBRATION_FINGERPRINT],
        "sensenova_u1_8b_fast:q8:mlx"
    );
}

/// The anchor is the RESIDENT rung under the eager load shape. The pinned contract classifies
/// `StagedResidency` and `BoundedDecode` structurally not applicable on this family, and rung 4
/// keys off the deferred load shape, so neither is a capturable anchor here — both are refused
/// before any weights are opened.
#[test]
fn sensenova_refuses_a_non_resident_rung_and_a_deferred_load_shape() {
    let mut deferred = sensenova_planned(SENSENOVA_PROVIDER, "sensenova_u1_8b", "q8");
    deferred["planned"]["loadShape"] = json!("deferred_materialization");
    let error = run_sensenova(&deferred).expect_err("the deferred shape is not the anchor's");
    assert!(error.contains("eager_materialization"), "{error}");

    let mut staged = sensenova_planned(SENSENOVA_PROVIDER, "sensenova_u1_8b", "q8");
    staged["planned"]["strategy"] = json!({
        "rung": "staged_residency",
        "engagedRungs": ["resident", "staged_residency"],
        "parameters": {}
    });
    let error = run_sensenova(&staged).expect_err("staged residency is structurally N/A here");
    assert!(error.contains("StructurallyNotApplicable"), "{error}");
}

/// The fixture is bound to the member, the tier and the geometry edge.
#[test]
fn sensenova_fixture_is_bound_to_the_member_tier_and_edge() {
    let request = sensenova_planned(SENSENOVA_PROVIDER, "sensenova_u1_8b", "q4");
    let bind = |fixture: &str, arm, tier| {
        validate_sensenova_fixture(
            &json!({ "planned": {
                "fixture": fixture,
                "target": request["planned"]["target"].clone(),
            }}),
            arm,
            tier,
        )
    };
    assert!(bind(
        "quality-mlx-q4-1024-seed22734-step2",
        SENSENOVA_BASE_ARM,
        "q4"
    )
    .is_ok());
    for (fixture, expected) in [
        // another member's stem
        (
            "infographic-v2-mlx-q4-1024-seed22734-step2",
            "must start with",
        ),
        // another tier
        ("quality-mlx-q8-1024-seed22734-step2", "must start with"),
        // another edge
        ("quality-mlx-q4-768-seed22734-step2", "must start with"),
        // another seed
        ("quality-mlx-q4-1024-seed1-step2", "does not match"),
        // another step count
        ("quality-mlx-q4-1024-seed22734-step4", "two-step"),
    ] {
        let error = bind(fixture, SENSENOVA_BASE_ARM, "q4")
            .expect_err("the fixture must be bound to its cell");
        assert!(error.contains(expected), "{fixture}: {error}");
    }
}

// ---- sc-22738: the three phases the pinned engine bounds on every route -------------------

const CONDITIONING: PhaseMemory = PhaseMemory {
    active: 17_000_000_000,
    cache: 512,
};
const DENOISE: PhaseMemory = PhaseMemory {
    active: 18_000_000_000,
    cache: 1_024,
};
const DECODE: PhaseMemory = PhaseMemory {
    active: 16_500_000_000,
    cache: 256,
};

fn step(current: u32, total: u32) -> Progress {
    Progress::Step { current, total }
}

/// One request's stream on a single-loop route: `Loading(Renderer)` after the prefills,
/// one `Step` per completed denoise step, `Decoding` after the last.
fn single_loop_stream(steps: u32) -> Vec<Progress> {
    let mut stream = vec![Progress::Loading(LoadPhase::Renderer)];
    stream.extend((1..=steps).map(|current| step(current, steps)));
    stream.push(Progress::Decoding);
    stream
}

/// The pinned engine's progress stream per route, from the engine's own source at e34d7b46a
/// (`mlx-gen-sensenova/src/t2i.rs`: `StepReporter::open_denoise_phase` /
/// `open_decode_phase`, `request_phase_bounds`, `finish_interleave_progress`; pinned there by
/// `step_reporter_opens_each_phase_once_per_request` and
/// `interleave_run_opens_denoise_once_and_closes_with_one_decode_boundary`). Every route
/// opens each boundary exactly once per request; the interleave route folds the bar and
/// closes with one `Decoding` after the fill; a multi-image request opens denoise from its
/// first loop and decode from its last.
fn pinned_engine_streams() -> Vec<(&'static str, Vec<Progress>)> {
    vec![
        (
            "text_to_image (sensenova_u1_8b quality, `T2iModel::denoise`)",
            single_loop_stream(2),
        ),
        (
            "edit / it2i (`T2iModel::it2i_denoise`)",
            single_loop_stream(2),
        ),
        (
            "fast turnkey (sensenova_u1_8b_fast, distilled steps)",
            single_loop_stream(8),
        ),
        (
            "infographic v2 / v3 (same denoise loop under the infographic rehost)",
            single_loop_stream(3),
        ),
        (
            "interleave (model-driven, 2 of 3 images realized, folded bar + fill)",
            vec![
                Progress::Loading(LoadPhase::Renderer),
                step(1, 6),
                step(2, 6),
                step(3, 6),
                step(4, 6),
                step(6, 6),
                Progress::Decoding,
            ],
        ),
        (
            "multi-image request (count 2: the first loop opens denoise, the last opens decode)",
            vec![
                Progress::Loading(LoadPhase::Renderer),
                step(1, 2),
                step(2, 2),
                step(1, 2),
                step(2, 2),
                Progress::Decoding,
            ],
        ),
    ]
}

/// Fold a stream the way the arm does: the first close reads the conditioning peak, the
/// second the denoise peak. Returns the fold and how many times a phase closed.
fn fold(stream: &[Progress]) -> (SenseNovaPhases, u32) {
    let mut phases = SenseNovaPhases::default();
    let mut closes = 0_u32;
    for progress in stream {
        phases.observe(progress, || {
            closes += 1;
            if closes == 1 {
                CONDITIONING
            } else {
                DENOISE
            }
        });
    }
    (phases, closes)
}

#[test]
fn prompt_encoding_completion_does_not_close_or_invent_memory_phases() {
    for (_, stream) in pinned_engine_streams() {
        let with_completions: Vec<_> = std::iter::once(Progress::PromptEncodingComplete)
            .chain(
                stream
                    .iter()
                    .flat_map(|event| [*event, Progress::PromptEncodingComplete]),
            )
            .collect();
        let (baseline, baseline_closes) = fold(&stream);
        let (observed, closes) = fold(&with_completions);
        assert_eq!(closes, baseline_closes);
        assert_eq!(
            sensenova_phase_peaks(SENSENOVA_BASE_ARM, &observed, DECODE),
            sensenova_phase_peaks(SENSENOVA_BASE_ARM, &baseline, DECODE),
        );
    }
}

/// sc-22738 — every route's stream at e34d7b46a cuts all three declared phases on an engine
/// boundary, each phase closes exactly once, and the receipt is the full decomposition the
/// sibling still arms file.
#[test]
fn every_route_stream_cuts_all_three_declared_phases_on_an_engine_boundary() {
    use sceneworks_core::memory_calibration::PredictedPeakBytes;

    for (route, stream) in pinned_engine_streams() {
        let (phases, closes) = fold(&stream);
        assert_eq!(
            closes, 2,
            "{route}: conditioning and denoise each close once"
        );
        let peaks = sensenova_phase_peaks(SENSENOVA_BASE_ARM, &phases, DECODE)
            .unwrap_or_else(|error| panic!("{route}: {error}"));
        assert_eq!(
            peaks,
            SenseNovaPhasePeaks {
                conditioning: CONDITIONING,
                denoise: DENOISE,
                decode: DECODE,
            },
            "{route}"
        );
        let overall = PhaseMemory::overall(&[peaks.conditioning, peaks.denoise, peaks.decode]);
        let predicted = image_predicted_peak_bytes(peaks.conditioning, peaks.denoise, peaks.decode);
        assert_eq!(
            predicted.overall,
            predicted_phase_ceiling(overall, IMAGE_PREDICTED_PEAK_BASIS),
            "{route}"
        );
        let receipt: PredictedPeakBytes =
            serde_json::from_value(predicted.json()).expect("a four-key decomposition");
        assert!(
            receipt.full().is_some(),
            "{route}: the receipt is the full decomposition, not an overall-only ceiling"
        );
    }
}

/// sc-22738 (mutation) — drop, duplicate, misplace or add a boundary and the stream is refused
/// BY NAME; nothing is filed under a model the engine no longer emits. The e16c6a55e stream
/// (steps only) is one of the refused shapes.
#[test]
fn a_stream_missing_or_repeating_a_boundary_is_refused_not_filed() {
    for (route, stream) in pinned_engine_streams() {
        let without_loading: Vec<Progress> = stream
            .iter()
            .filter(|progress| !matches!(progress, Progress::Loading(_)))
            .cloned()
            .collect();
        let (phases, _) = fold(&without_loading);
        let error = sensenova_phase_peaks(SENSENOVA_BASE_ARM, &phases, DECODE)
            .expect_err("no conditioning/denoise edge");
        assert!(
            error.contains(
                "emitted 0 `Progress::Loading(LoadPhase::Renderer)`, 1 `Progress::Decoding`"
            ),
            "{route}: {error}"
        );
        assert!(error.contains("nothing is filed"), "{route}: {error}");

        let without_decoding: Vec<Progress> = stream
            .iter()
            .filter(|progress| !matches!(progress, Progress::Decoding))
            .cloned()
            .collect();
        let (phases, _) = fold(&without_decoding);
        let error = sensenova_phase_peaks(SENSENOVA_FAST_ARM, &phases, DECODE)
            .expect_err("denoise never closes");
        assert!(
            error.contains(
                "emitted 1 `Progress::Loading(LoadPhase::Renderer)`, 0 `Progress::Decoding`"
            ),
            "{route}: {error}"
        );
    }

    // The e16c6a55e engine: steps only.
    let (phases, closes) = fold(&[step(1, 2), step(2, 2)]);
    assert_eq!(closes, 0, "a step is not a boundary");
    let error = sensenova_phase_peaks(SENSENOVA_BASE_ARM, &phases, DECODE)
        .expect_err("the previous pin's stream bounds no phase");
    assert!(
        error
            .contains("emitted 0 `Progress::Loading(LoadPhase::Renderer)`, 0 `Progress::Decoding`"),
        "{error}"
    );

    // A duplicated boundary (a per-image reporter opening a phase again).
    let mut duplicated = single_loop_stream(2);
    duplicated.insert(2, Progress::Loading(LoadPhase::Renderer));
    let (phases, closes) = fold(&duplicated);
    assert_eq!(closes, 2, "the second `Loading(Renderer)` closes nothing");
    let error = sensenova_phase_peaks(SENSENOVA_BASE_ARM, &phases, DECODE)
        .expect_err("two conditioning edges");
    assert!(
        error
            .contains("emitted 2 `Progress::Loading(LoadPhase::Renderer)`, 1 `Progress::Decoding`"),
        "{error}"
    );
    let mut duplicated = single_loop_stream(2);
    duplicated.push(Progress::Decoding);
    let (phases, _) = fold(&duplicated);
    let error =
        sensenova_phase_peaks(SENSENOVA_BASE_ARM, &phases, DECODE).expect_err("two decode edges");
    assert!(
        error.contains("1 `Progress::Loading(LoadPhase::Renderer)`, 2 `Progress::Decoding`"),
        "{error}"
    );

    // A boundary the engine does not emit: the model may no longer cut the conditioning edge.
    let mut foreign = single_loop_stream(2);
    foreign.insert(0, Progress::Loading(LoadPhase::TextEncoder));
    let (phases, _) = fold(&foreign);
    let error = sensenova_phase_peaks(SENSENOVA_BASE_ARM, &phases, DECODE)
        .expect_err("an unexpected loading boundary");
    assert!(
        error.contains("and 1 other `Progress::Loading(_)` boundaries"),
        "{error}"
    );

    // The boundaries in the wrong order: counts are right, the edge was never cut.
    let (phases, closes) = fold(&[
        Progress::Decoding,
        step(1, 1),
        Progress::Loading(LoadPhase::Renderer),
    ]);
    assert_eq!(closes, 1);
    let error = sensenova_phase_peaks(SENSENOVA_BASE_ARM, &phases, DECODE)
        .expect_err("decode before the conditioning edge");
    assert!(error.contains("`Progress::Decoding` before"), "{error}");
}

/// A zero in any bounded phase is a refusal — never a filed number.
#[test]
fn a_zero_phase_is_refused_not_filed() {
    let (phases, _) = fold(&single_loop_stream(2));
    let error = sensenova_phase_peaks(
        SENSENOVA_BASE_ARM,
        &phases,
        PhaseMemory {
            active: 0,
            cache: 0,
        },
    )
    .expect_err("an empty decode phase");
    assert!(error.contains("zero active peak"), "{error}");
    assert!(error.contains("decode: 0 bytes"), "{error}");

    let mut phases = SenseNovaPhases::default();
    for progress in single_loop_stream(2) {
        phases.observe(&progress, || PhaseMemory {
            active: 0,
            cache: 0,
        });
    }
    let error = sensenova_phase_peaks(SENSENOVA_FAST_ARM, &phases, DECODE)
        .expect_err("empty conditioning and denoise phases");
    assert!(
        error.contains("conditioning: 0 bytes, denoise: 0 bytes"),
        "{error}"
    );
}

/// sc-22738 — the arm ACTS on the three-phase cut: the stream reaches it only through
/// `SenseNovaPhases::observe`, the receipt is the full four-key decomposition, the per-phase
/// peaks that mint an anchor are published, the identity preflight runs before the load, and
/// the e16c6a55e relaxation (an overall-only ceiling, phases declared unattributed, a cut on
/// the first `Step`) is gone. Read the way the sibling guards in this file read an arm — the
/// measured render needs real weights.
///
/// Mutations that fail this: cutting on `Progress::Step { current: 1, .. }` again; filing
/// `overall_only_json`; withholding `conditioningActivePeak` / `denoiseActivePeak` /
/// `decodeActivePeak`; declaring a `PhaseUnattributed` marker; dropping the preflight.
#[test]
fn the_sensenova_arm_cuts_every_declared_phase_on_an_engine_boundary() {
    let source = crate::mlx_source_for_tests();
    let start = source
        .find("\nfn run_sensenova(")
        .expect("the arm still exists");
    let body = &source[start
        ..start
            + source[start..]
                .find("\n}\n")
                .expect("the arm's body closes")];
    for cut in [
        "sensenova_identity_preflight(arm, tier, &spec, &expected_fingerprint)?;",
        "phases.observe(&progress, || {",
        "let decode = PhaseMemory::capture();",
        "} = sensenova_phase_peaks(arm, &phases, decode)?;",
        "let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);",
        "\"predictedPeakBytes\": predicted_peaks.json(),",
        "\"conditioning\": conditioning.json(),",
        "\"denoise\": denoise.json(),",
        "\"decode\": decode.json(),",
        "(\"conditioningActivePeak\", \"bytes\", conditioning.active)",
        "(\"denoiseActivePeak\", \"bytes\", denoise.active)",
        "(\"decodeActivePeak\", \"bytes\", decode.active)",
    ] {
        assert!(
            body.contains(cut),
            "the strict three-phase capture lost {cut}"
        );
    }
    // Any direct read of the stream inside the arm — a match arm, an `if let`, a reset keyed
    // on an event — is a cut the fold does not make; the only consumer of the stream is
    // `SenseNovaPhases::observe`.
    for relaxation in [
        "Progress::Step",
        "Progress::Loading",
        "Progress::Decoding",
        "PhaseUnattributed",
        "overall_only_json",
        "throughFirstDenoiseStep",
        "postFirstDenoiseStep",
        "SENSENOVA_ENGINE_BOUNDARIES",
        "SENSENOVA_BOUNDARY_INFERENCE_CHANGE",
        "\"observedMemory\": { \"overall\"",
    ] {
        assert!(
            !body.contains(relaxation),
            "the SenseNova arm must not grade on {relaxation} again"
        );
    }
}

// ---- sc-22738: the identity the pinned engine publishes ---------------------------------

fn preflight_root(arm: SenseNovaArm, tier: &str) -> (PathBuf, &'static str, &'static str) {
    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir()
        .join(format!("sc-22738-sensenova-{}-{nonce}", std::process::id()))
        .join(format!(
            "models--{}",
            arm.expected_repository.replace('/', "--")
        ))
        .join("snapshots")
        .join(REVISION)
        .join(tier);
    std::fs::create_dir_all(&root).unwrap();
    (root, arm.expected_repository, REVISION)
}

/// A minimal dense checkpoint (or shard) in the shape the engine's tier proof reads
/// (`resolved_artifact_tier`, memory_strategy.rs:733-808): one backbone decoder Linear
/// (`convert::is_backbone_linear`) with no `.scales` companion — a bf16 artifact.
fn dense_backbone_safetensors(tensor: &str) -> Vec<u8> {
    let header = format!(r#"{{"{tensor}":{{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}}}"#);
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0_u8; 8]);
    bytes
}

const SHARD_TENSORS: [(&str, &str); 2] = [
    (
        "model-00001-of-00002.safetensors",
        "language_model.model.layers.0.self_attn.q_proj.weight",
    ),
    (
        "model-00002-of-00002.safetensors",
        "language_model.model.layers.1.self_attn.k_proj.weight",
    ),
];

/// The layout the three dense quality rehosts ship, at two shards: every shard a readable
/// dense checkpoint, and `model.safetensors.index.json` whose `weight_map` names exactly them.
fn write_sharded_dense_root(root: &Path, indexed: &[(&str, &str)]) {
    for (shard, tensor) in SHARD_TENSORS {
        std::fs::write(root.join(shard), dense_backbone_safetensors(tensor)).unwrap();
    }
    let weight_map: Vec<String> = indexed
        .iter()
        .map(|(shard, tensor)| format!("\"{tensor}\":\"{shard}\""))
        .collect();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        format!(
            "{{\"metadata\":{{}},\"weight_map\":{{{}}}}}",
            weight_map.join(",")
        ),
    )
    .unwrap();
}

/// sc-22738 — at e34d7b46a (inference #966) an indexed shard set PINS: the preflight admits
/// the sharded dense root with the engine's own predicates, proves it dense, and the engine's
/// table names the identity the plan carries for it. At e16c6a55e the same root was refused
/// before the load (`verified_artifact` pinned only a single `model.safetensors`), which is
/// why the three quality bf16 cells never captured.
#[test]
fn an_indexed_shard_set_passes_the_preflight_and_publishes_the_identity() {
    use runtime_macos::providers::sensenova::memory_strategy as engine;
    let (root, repository, revision) = preflight_root(SENSENOVA_BASE_ARM, "bf16");
    write_sharded_dense_root(&root, &SHARD_TENSORS);
    let spec =
        sensenova_load_spec_at(SENSENOVA_BASE_ARM, repository, revision, root, "bf16").unwrap();
    let expected = sensenova_calibration_fingerprint(SENSENOVA_BASE_ARM, "bf16");
    assert_eq!(
        sensenova_identity_preflight(SENSENOVA_BASE_ARM, "bf16", &spec, &expected),
        Ok(())
    );
    let identity = engine::verified_artifact_identity(&spec).expect("an indexed shard set pins");
    assert_eq!(identity.len(), 64, "{identity}");
    assert_eq!(engine::resolved_artifact_tier(&spec).unwrap(), None);
    assert_eq!(
        engine::production_calibration_fingerprint(SENSENOVA_BASE_ARM.provider, &spec).as_deref(),
        Some(expected.as_str())
    );

    // The same shards under a q4 plan: the artifact-proven tier disagrees with the knob.
    let (root, repository, revision) = preflight_root(SENSENOVA_BASE_ARM, "q4");
    write_sharded_dense_root(&root, &SHARD_TENSORS);
    let spec =
        sensenova_load_spec_at(SENSENOVA_BASE_ARM, repository, revision, root, "q4").unwrap();
    let expected = sensenova_calibration_fingerprint(SENSENOVA_BASE_ARM, "q4");
    let error = sensenova_identity_preflight(SENSENOVA_BASE_ARM, "q4", &spec, &expected)
        .expect_err("a dense shard set cannot serve a q4 cell");
    assert!(error.contains("ARTIFACT-proven tier"), "{error}");
    assert!(error.contains("None") && error.contains("Q4"), "{error}");
}

/// sc-22738 — a tier root the pinned engine cannot pin is still refused BEFORE the load, with
/// the engine's own predicates, naming the layout and the engine site: an index that does not
/// account for the shards beside it, an unindexed multi-file layout, a single dense file under
/// a packed plan, an unreadable single file.
#[test]
fn a_root_the_pinned_engine_cannot_pin_is_refused_before_the_load_naming_the_layout() {
    // Two shards, an index naming only one of them.
    let (root, repository, revision) = preflight_root(SENSENOVA_BASE_ARM, "bf16");
    write_sharded_dense_root(&root, &SHARD_TENSORS[..1]);
    let spec =
        sensenova_load_spec_at(SENSENOVA_BASE_ARM, repository, revision, root, "bf16").unwrap();
    let expected = sensenova_calibration_fingerprint(SENSENOVA_BASE_ARM, "bf16");
    let error = sensenova_identity_preflight(SENSENOVA_BASE_ARM, "bf16", &spec, &expected)
        .expect_err("an index that does not account for the shards does not pin");
    for needle in [
            "refused before the load",
            "`verified_artifact` (mlx-gen-sensenova/src/memory_strategy.rs:656-669",
            "2 safetensors entries [model-00001-of-00002.safetensors, model-00002-of-00002.safetensors]",
            "the index does not account for exactly",
            expected.as_str(),
            "stays runnable",
        ] {
            assert!(error.contains(needle), "missing {needle:?} in {error}");
        }
    assert!(!error.contains("SCENEWORKS_"), "{error}");

    // Two shards and no index at all.
    let (root, repository, revision) = preflight_root(SENSENOVA_BASE_ARM, "bf16");
    write_sharded_dense_root(&root, &SHARD_TENSORS);
    std::fs::remove_file(root.join("model.safetensors.index.json")).unwrap();
    let spec =
        sensenova_load_spec_at(SENSENOVA_BASE_ARM, repository, revision, root, "bf16").unwrap();
    let error = sensenova_identity_preflight(SENSENOVA_BASE_ARM, "bf16", &spec, &expected)
        .expect_err("an unindexed multi-file root does not pin");
    assert!(error.contains("an unindexed multi-file layout"), "{error}");

    // The single dense `model.safetensors` the fast turnkeys ship: pinned, proven dense, and
    // the engine's own table names the identity the plan carries for it.
    let (root, repository, revision) = preflight_root(SENSENOVA_FAST_ARM, "bf16");
    std::fs::write(
        root.join("model.safetensors"),
        dense_backbone_safetensors(SHARD_TENSORS[0].1),
    )
    .unwrap();
    let spec =
        sensenova_load_spec_at(SENSENOVA_FAST_ARM, repository, revision, root, "bf16").unwrap();
    let expected = sensenova_calibration_fingerprint(SENSENOVA_FAST_ARM, "bf16");
    assert_eq!(
        sensenova_identity_preflight(SENSENOVA_FAST_ARM, "bf16", &spec, &expected),
        Ok(())
    );
    assert_eq!(
        runtime_macos::providers::sensenova::memory_strategy::production_calibration_fingerprint(
            SENSENOVA_FAST_ARM.provider,
            &spec
        )
        .as_deref(),
        Some(expected.as_str())
    );

    // The same dense file under a q4 plan: the artifact-proven tier disagrees with the knob,
    // and the engine keys the identity on the artifact.
    let (root, repository, revision) = preflight_root(SENSENOVA_FAST_ARM, "q4");
    std::fs::write(
        root.join("model.safetensors"),
        dense_backbone_safetensors(SHARD_TENSORS[0].1),
    )
    .unwrap();
    let spec =
        sensenova_load_spec_at(SENSENOVA_FAST_ARM, repository, revision, root, "q4").unwrap();
    let expected = sensenova_calibration_fingerprint(SENSENOVA_FAST_ARM, "q4");
    let error = sensenova_identity_preflight(SENSENOVA_FAST_ARM, "q4", &spec, &expected)
        .expect_err("a dense artifact cannot serve a q4 cell");
    assert!(error.contains("ARTIFACT-proven tier"), "{error}");
    assert!(error.contains("None") && error.contains("Q4"), "{error}");

    // A single file whose headers cannot be read fails closed on the tier proof.
    let (root, repository, revision) = preflight_root(SENSENOVA_BASE_ARM, "q8");
    std::fs::write(root.join("model.safetensors"), b"not a checkpoint").unwrap();
    let spec =
        sensenova_load_spec_at(SENSENOVA_BASE_ARM, repository, revision, root, "q8").unwrap();
    let expected = sensenova_calibration_fingerprint(SENSENOVA_BASE_ARM, "q8");
    let error = sensenova_identity_preflight(SENSENOVA_BASE_ARM, "q8", &spec, &expected)
        .expect_err("an unreadable artifact proves no tier");
    assert!(error.contains("`resolved_artifact_tier`"), "{error}");
}

/// sc-22738 — the identity NAME set the pinned engine tables for all eighteen cells is the set
/// the plan and this arm carry: no naming drift between PR #2734's SceneWorks side and the
/// engine's `production_calibration_fingerprint` (memory_strategy.rs:869-898) at e34d7b46a.
/// The table is the (route, tier) key; since inference #966 the sharded dense roots pin in
/// front of it too (`an_indexed_shard_set_passes_the_preflight_and_publishes_the_identity`).
#[test]
fn every_sensenova_cell_names_the_identity_the_pinned_engine_tables() {
    use runtime_macos::providers::sensenova::memory_strategy as engine;
    let revision = "0123456789abcdef0123456789abcdef01234567";
    for arm in SENSENOVA_FAMILY {
        for tier in ["bf16", "q4", "q8"] {
            let root = PathBuf::from(format!(
                "/cache/models--{}/snapshots/{revision}/{tier}",
                arm.expected_repository.replace('/', "--")
            ));
            let spec =
                sensenova_load_spec_at(arm, arm.expected_repository, revision, root, tier).unwrap();
            assert_eq!(
                engine::production_calibration_fingerprint(arm.provider, &spec).as_deref(),
                Some(sensenova_calibration_fingerprint(arm, tier).as_str()),
                "{}:{tier}",
                arm.model_id
            );
        }
    }
}
