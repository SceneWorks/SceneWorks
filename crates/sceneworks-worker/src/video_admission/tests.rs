//! sc-18814: the video gate reaches the shared ladder selector, and the constants
//! `sceneworks-core` transcribed from gen-core still match the pinned bundle.

use gen_core::{
    LoadShape, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryLifecycleCapabilities,
    MemoryParameterRanges, MemoryPhase, MemoryStrategyCapability, MemoryStrategySupport,
    MemoryWindowMaterialization, Precision, Quant, VaeTiling,
};
use sceneworks_core::video_request::{
    single_pass_decode_frame_cap, vae_full_res_channels, video_admission, VideoAdmission,
    VideoAdmissionGeometry, VideoDecodePass, VideoGeometryRole,
};

use super::*;

const GIB: u64 = 1024 * 1024 * 1024;

fn tier() -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant: Some(Quant::Q8),
        component_precision_floors: &[],
    }
}

/// A conformant contract whose implemented rungs are exactly `rungs` (plus the resident baseline).
///
/// `conformance_errors()` is asserted empty here rather than left to chance: a contract the shared
/// selector rejects as `Invalid` would make every assertion below pass as `Undecidable` without
/// ever exercising a selection — the hollow proof this epic has shipped four of.
fn fixture_contract(
    base_gib: u64,
    conditioning_gib: u64,
    rungs: &[MemoryStrategy],
) -> MemoryProviderContract {
    fixture_contract_with_realization(
        base_gib,
        conditioning_gib,
        rungs,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: false,
            cache_eviction: true,
        },
    )
}

/// [`fixture_contract`] on a caller-chosen backend realization, so the candle lane can be graded
/// against a provider whose `backend_id()` actually says `candle` — which is what
/// `memory_strategy::candidate_exclusion` compares `RequestScope.backend` against.
fn fixture_contract_with_realization(
    base_gib: u64,
    conditioning_gib: u64,
    rungs: &[MemoryStrategy],
    realization: MemoryBackendRealization,
) -> MemoryProviderContract {
    let mut contract =
        MemoryProviderContract::compatibility_default("video_admission_fixture", realization);
    contract.asset_facts.base_bytes = base_gib * GIB;
    contract.asset_facts.conditioning_bytes = conditioning_gib * GIB;
    contract.asset_facts.transformer_bytes = (base_gib - conditioning_gib) * GIB;
    contract.asset_facts.decoder_bytes = 0;
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.calibration = Some(MemoryCalibrationIdentity {
        abi: gen_core::MEMORY_CALIBRATION_ABI,
        fingerprint: "video-admission-fixture-v1".to_owned(),
        load_shape: LoadShape::EagerMaterialization,
    });
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        // Each rung's execution hook, so a contract that declares the rung `Implemented` is
        // conformant and the selection can be judged on its PREREQUISITES rather than bouncing
        // off a conformance error.
        decode_tiling: rungs.contains(&MemoryStrategy::BoundedDecode),
        attention_chunking: rungs.contains(&MemoryStrategy::BoundedAttention),
        transformer_window_materialization: rungs
            .contains(&MemoryStrategy::BoundedTransformerResidency),
    };
    contract.strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: if strategy == MemoryStrategy::Resident || rungs.contains(&strategy) {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            },
            // Each rung owns ONLY its own knobs; the contract's conformance rules reject a rung
            // that publishes another's.
            parameters: MemoryParameterRanges {
                decode_tile_edges: if strategy == MemoryStrategy::BoundedDecode {
                    vec![256, 512]
                } else {
                    Vec::new()
                },
                decode_overlaps: if strategy == MemoryStrategy::BoundedDecode {
                    vec![32, 64]
                } else {
                    Vec::new()
                },
                attention_chunk_sizes: if strategy == MemoryStrategy::BoundedAttention {
                    vec![1024, 4096]
                } else {
                    Vec::new()
                },
                transformer_window_sizes: if strategy == MemoryStrategy::BoundedTransformerResidency
                {
                    vec![1, 2]
                } else {
                    Vec::new()
                },
                transformer_window_components: Vec::new(),
            },
        })
        .collect();
    assert!(
        contract.conformance_errors().is_empty(),
        "the fixture contract must be conformant, else every selection below is vacuously \
         Unverified: {:?}",
        contract.conformance_errors()
    );
    contract
}

/// Move a contract AND its calibration identity onto one materialization shape. Both must agree:
/// the shared selector's estimate path runs `MemoryEvidence::optimized_eligibility`, which compares
/// the identity's shape, so a contract whose identity disagrees is excluded before its rung's
/// prerequisites are ever consulted — and the test would then pass for the wrong reason.
fn with_load_shape(
    mut contract: MemoryProviderContract,
    load_shape: LoadShape,
) -> MemoryProviderContract {
    contract.load_shape = load_shape;
    if let Some(identity) = contract.calibration.as_mut() {
        identity.load_shape = load_shape;
    }
    contract
}

fn budget(total_gb: f64) -> Option<Budget> {
    Some(Budget {
        available_gb: total_gb,
        reclaimable_gb: 0.0,
        total_gb,
        reserved_headroom_gb: 0.0,
    })
}

fn geometry(frames: u32, role: VideoGeometryRole) -> VideoAdmissionGeometry {
    VideoAdmissionGeometry {
        width: 1280,
        height: 704,
        frames,
        batch: 1,
        decode_pass: VideoDecodePass::SinglePass,
        role,
    }
}

// --------------------------------------------------------------------------------------------
// The transcription pin: `sceneworks-core` has no gen-core dependency, so its VAE constants are
// copied. A pin bump that moves one must be RED here, not silently wrong in the gate.
// --------------------------------------------------------------------------------------------

/// The number of `type: "video"` entries in the shipped `builtin.models.jsonc`, mirroring
/// `sceneworks-core`'s `EXPECTED_SHIPPED_VIDEO_COUNT` and `pinned_engine_geometry`'s
/// `EXPECTED_VIDEO_IDS`. Adding a video model without updating it trips
/// [`core_transcribes_the_pinned_vae_write_bounds`].
const EXPECTED_SHIPPED_VIDEO_COUNT: usize = 10;

/// Every video model id in the shipped manifest. The ONE list the transcription pin is driven
/// from, so a newly shipped family cannot be transcribed in core and left unpinned here.
///
/// `pinned_engine_geometry` has the identical helper but is `#[cfg]`-gated to the lanes that link
/// an engine bundle (macOS or `backend-candle`); this module compiles on all three, so it reads
/// the same manifest bytes rather than depending on that module.
fn shipped_video_model_ids() -> Vec<String> {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present in BUILTIN_MANIFESTS");
    let manifest: serde_json::Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
            .expect("builtin.models.jsonc parses as JSON");
    let ids: Vec<String> = manifest
        .get("models")
        .and_then(serde_json::Value::as_array)
        .expect("builtin.models.jsonc has a models array")
        .iter()
        .filter(|model| model.get("type").and_then(serde_json::Value::as_str) == Some("video"))
        .map(|model| {
            model
                .get("id")
                .and_then(serde_json::Value::as_str)
                .expect("every shipped video model declares an id")
                .to_owned()
        })
        .collect();
    assert_eq!(
        ids.len(),
        EXPECTED_SHIPPED_VIDEO_COUNT,
        "a video model was added/removed in builtin.models.jsonc — update \
         EXPECTED_SHIPPED_VIDEO_COUNT and map the new id to the VaeTiling its decoder actually \
         runs in `expected_vae_tiling` (sc-18814); do not let it go unpinned: {ids:?}"
    );
    ids
}

/// The `gen_core::VaeTiling` a shipped video family's decoder actually runs, or `None` for a
/// family `sceneworks-core` deliberately does not model.
///
/// **An unmapped id panics**, the same posture — and for the same reason — as
/// `pinned_engine_geometry::expected_max_pixels` (sc-12409): adding a video model is a deliberate
/// act that must derive its write bound from its own engine, never inherit one by default. The
/// panic is safe HERE and would not be inside `vae_full_res_channels` itself, which is reached for
/// arbitrary community model ids in production.
///
/// The assignments cite the decode path each engine takes; see `vae_full_res_channels`' doc for
/// the citations and for what these tests do NOT pin (sc-19117).
fn expected_vae_tiling(id: &str) -> Option<VaeTiling> {
    match id {
        "ltx_2_3" | "ltx_2_3_eros" => Some(VaeTiling::LTX),
        // The dense TI2V-5B is welded to the z48 vae22 (`mlx-gen-wan/src/pipeline.rs:235`).
        "wan_2_2" => Some(VaeTiling::WAN22),
        // The A14B grid and every Wan-derived renderer decode through the Wan2.1 z16 VAE.
        "wan_2_2_t2v_14b"
        | "wan_2_2_i2v_14b"
        | "wan_2_2_vace_fun_14b"
        | "bernini"
        | "scail2_14b"
        | "krea_realtime_14b" => Some(VaeTiling::WAN),
        // SVD's bound is `candle-gen-svd`'s PRIVATE `SVD_VAE_TILING` with no MLX sibling to pin a
        // second value against, so core reports it unmodelled rather than guessing.
        "svd" => None,
        other => panic!(
            "shipped video model {other:?} is not mapped to a pinned VaeTiling — read the \
             VaeTiling its decoder passes to `budgeted_plan` out of that engine's crate and add \
             it to `expected_vae_tiling`; do not blanket-apply a default (sc-18814)"
        ),
    }
}

#[test]
fn core_transcribes_the_pinned_vae_write_bounds() {
    let mut modelled = 0_usize;
    let mut unmodelled = 0_usize;
    for model in shipped_video_model_ids() {
        match expected_vae_tiling(&model) {
            Some(vae) => {
                assert_eq!(
                    vae_full_res_channels(&model),
                    Some(vae.full_res_channels as u32),
                    "{model}: core's transcribed channel count must equal the pinned \
                     gen_core::VaeTiling constant its decoder runs"
                );
                modelled += 1;
            }
            None => {
                assert_eq!(
                    vae_full_res_channels(&model),
                    None,
                    "{model} is deliberately unmodelled; core must report None rather than a \
                     number nothing can pin"
                );
                unmodelled += 1;
            }
        }
    }
    // Both arms are genuinely exercised, so neither is vacuous.
    assert_eq!(modelled + unmodelled, EXPECTED_SHIPPED_VIDEO_COUNT);
    assert!(modelled > 0 && unmodelled > 0);

    // The three constants are genuinely different, so the per-family loop above is not comparing
    // one value against itself.
    assert_ne!(
        VaeTiling::LTX.full_res_channels,
        VaeTiling::WAN22.full_res_channels
    );
    assert_ne!(
        VaeTiling::WAN22.full_res_channels,
        VaeTiling::WAN.full_res_channels
    );
}

#[test]
fn the_core_frame_cap_equals_gen_cores_writable_frame_cap() {
    for (width, height) in [(1280_u32, 704_u32), (768, 512), (512, 512), (1920, 1080)] {
        for (model, vae) in [
            ("ltx_2_3", VaeTiling::LTX),
            ("wan_2_2", VaeTiling::WAN22),
            ("krea_realtime_14b", VaeTiling::WAN),
        ] {
            let engine = vae.writable_frame_cap(height as i32, width as i32);
            assert_eq!(
                single_pass_decode_frame_cap(model, width, height),
                Some(u32::try_from(engine).unwrap()),
                "{model} @ {width}x{height}: core must agree with VaeTiling::writable_frame_cap"
            );
        }
    }
    // A family core deliberately does not model returns None rather than a wrong number.
    assert_eq!(single_pass_decode_frame_cap("svd", 1024, 576), None);
}

// --------------------------------------------------------------------------------------------
// The selector actually reaches `memory_strategy::select_strategy`.
// --------------------------------------------------------------------------------------------

fn select_once(
    contract: &MemoryProviderContract,
    budget: Option<Budget>,
    headroom_bytes: u64,
    geometry: VideoAdmissionGeometry,
) -> (
    VideoRungSelection,
    Vec<(VideoAdmissionGeometry, MemorySelection)>,
) {
    let mut selector = LadderVideoSelector::new(
        VideoRequestIdentity {
            route: "video_admission_fixture",
            lane: VideoLane::Mlx,
            tier: tier(),
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        contract,
        budget,
        headroom_bytes,
    );
    let verdict = selector.select(geometry);
    (verdict, selector.selections)
}

#[test]
fn a_roomy_host_selects_the_resident_rung_through_the_shared_selector() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let (verdict, selections) = select_once(
        &contract,
        budget(128.0),
        18 * GIB,
        geometry(241, VideoGeometryRole::Requested),
    );
    let VideoRungSelection::Selected { rung, .. } = verdict else {
        panic!("expected a selection, got {verdict:?}");
    };
    assert_eq!(rung, StrategyRung::Resident);
    assert_eq!(selections.len(), 1);
    // The frames the video request actually renders reached the evidence key — this is the seam
    // sc-18829's temporal term attaches to, and the image lane pins this field at 1.
    assert_eq!(selections[0].0.frames, 241);
}

#[test]
fn a_constrained_host_walks_down_the_ladder_instead_of_refusing() {
    // 20 GiB of weights (4 conditioning + 16 transformer) + 18 GiB headroom = 38 GiB resident,
    // widened by the 10% MLX estimate margin to 41.8. Staged drops the co-residency to
    // max(4, 16) = 16, i.e. 34 GiB → 37.4 widened. A 40 GiB host therefore refuses resident and
    // admits staged: the ladder's whole point.
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let (verdict, _) = select_once(
        &contract,
        budget(40.0),
        18 * GIB,
        geometry(241, VideoGeometryRole::Requested),
    );
    let VideoRungSelection::Selected { rung, .. } = verdict else {
        panic!("expected a selection, got {verdict:?}");
    };
    assert_eq!(rung, StrategyRung::StagedResidency);

    // The SAME host with a provider that implements no optimized rung has nowhere to go, so it
    // rejects — proving the line above is the ladder working and not simply a roomy budget.
    let flat = fixture_contract(20, 4, &[]);
    let (flat_verdict, _) = select_once(
        &flat,
        budget(40.0),
        18 * GIB,
        geometry(241, VideoGeometryRole::Requested),
    );
    assert!(
        matches!(flat_verdict, VideoRungSelection::Reject { .. }),
        "got {flat_verdict:?}"
    );
}

#[test]
fn no_budget_is_undecidable_rather_than_a_refusal() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let (verdict, _) = select_once(
        &contract,
        None,
        18 * GIB,
        geometry(241, VideoGeometryRole::Requested),
    );
    assert_eq!(verdict, VideoRungSelection::Undecidable);
}

// --------------------------------------------------------------------------------------------
// `admit_video_generation` — the funnel-facing entry point.
// --------------------------------------------------------------------------------------------

struct FixtureGenerator {
    descriptor: gen_core::ModelDescriptor,
    contract: Option<MemoryProviderContract>,
}

fn fixture_generator(contract: Option<MemoryProviderContract>) -> FixtureGenerator {
    FixtureGenerator {
        descriptor: gen_core::ModelDescriptor {
            id: "video_admission_fixture",
            family: "ltx",
            backend: "mlx",
            modality: gen_core::Modality::Video,
            capabilities: gen_core::Capabilities::default(),
            required_components: &[],
            control_kinds: None,
        },
        contract,
    }
}

impl gen_core::Generator for FixtureGenerator {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, _request: &gen_core::GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        _request: &gen_core::GenerationRequest,
        _on_progress: &mut dyn FnMut(gen_core::Progress),
    ) -> gen_core::Result<gen_core::GenerationOutput> {
        unreachable!("the admission gate never generates")
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        self.contract.as_ref()
    }
}

fn inputs<'a>(
    frames: u32,
    budget: Option<Budget>,
    headroom_bytes: u64,
) -> VideoAdmissionInputs<'a> {
    VideoAdmissionInputs {
        model_id: "ltx_2_3",
        route: "video_admission_fixture",
        lane: VideoLane::Mlx,
        tier: tier(),
        width: 1280,
        height: 704,
        frames,
        budget,
        headroom_bytes,
        expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
    }
}

#[test]
fn a_generator_with_no_contract_leaves_the_request_untouched() {
    let generator = fixture_generator(None);
    assert_eq!(
        admit_video_generation(&generator, inputs(241, budget(8.0), 18 * GIB)),
        VideoAdmissionOutcome::default(),
        "no contract means no declared rungs; the gate must fail open"
    );
}

#[test]
fn a_selected_resident_rung_leaves_the_request_byte_identical() {
    let generator = fixture_generator(Some(fixture_contract(
        20,
        4,
        &[MemoryStrategy::StagedResidency],
    )));
    let outcome = admit_video_generation(&generator, inputs(241, budget(128.0), 18 * GIB));
    assert_eq!(
        outcome,
        VideoAdmissionOutcome::default(),
        "resident engages nothing, so emitting explicit all-false knobs would replace the \
         provider's own defaults"
    );
}

#[test]
fn a_selected_optimized_rung_reaches_the_generation_request() {
    let generator = fixture_generator(Some(fixture_contract(
        20,
        4,
        &[MemoryStrategy::StagedResidency],
    )));
    let outcome = admit_video_generation(&generator, inputs(241, budget(40.0), 18 * GIB));
    let memory = outcome.memory.expect("staged residency was selected");
    assert!(memory.stage_residency, "{memory:?}");
    assert!(outcome.refusal.is_none());
}

/// A request above the single-pass cap grades the **cap geometry too**, through the real selector.
///
/// The name says what is actually checked. It deliberately does NOT claim the cap *binds* the
/// selection: with today's `floor_phase_peaks` the peak is geometry-blind — the same
/// weights+headroom floor for every geometry — so the two graded geometries cannot land on
/// different rungs no matter how the fixture is shaped, and a test named for the cap binding would
/// pass identically with the cap geometry never graded at all (sc-18814 review).
///
/// What IS provable here, and is: both geometries reach `select_strategy`, the second is the
/// 297-frame cap, and the request runs. The rung-level consequence of the cap is proved where it
/// can be — `sceneworks-core`'s `grading_only_the_request_would_have_understated_the_rung` and
/// `video_admission_selects_the_deepest_rung_the_graded_set_requires`, which drive the selector
/// seam directly. sc-18829's per-phase fit is what makes the peak geometry-dependent; this test's
/// `assert_ne!` on the two peaks is the tripwire that says so.
#[test]
fn a_request_above_the_cap_grades_the_cap_geometry_through_the_real_selector() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let mut selector = LadderVideoSelector::new(
        VideoRequestIdentity {
            route: "video_admission_fixture",
            lane: VideoLane::Mlx,
            tier: tier(),
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &contract,
        budget(40.0),
        18 * GIB,
    );
    // f305 at 1280x704 is past LTX's 297-frame single-pass cap, so both geometries are graded.
    let verdict = video_admission("ltx_2_3", VideoLane::Mlx, 1280, 704, 305, &mut selector);
    assert!(
        matches!(verdict, VideoAdmission::Admitted { .. }),
        "{verdict:?}"
    );
    assert_eq!(selector.selections.len(), 2, "{:?}", selector.selections);
    assert_eq!(selector.selections[0].0.frames, 305);
    assert_eq!(selector.selections[0].0.role, VideoGeometryRole::Requested);
    assert_eq!(selector.selections[1].0.frames, 297);
    assert_eq!(
        selector.selections[1].0.role,
        VideoGeometryRole::SinglePassDecodeCap
    );

    // And the reason the two cannot yet differ, as a live assertion rather than a comment, so it
    // goes RED the moment sc-18829 makes the peak geometry-dependent.
    assert_eq!(
        selector.selections[0].1.strategy, selector.selections[1].1.strategy,
        "`floor_phase_peaks` takes no geometry, so two graded geometries necessarily select the \
         same rung — which is precisely why this test cannot assert that the cap BINDS. When \
         sc-18829's per-phase fit makes this RED, re-derive the test to assert the binding rung \
         and re-run M21 (see the `peak_bytes()` comment in video_admission.rs)"
    );
}

/// The non-regression guard: inside the estimate-margin band the ladder still selects, but it must
/// NOT manufacture a refusal the pre-existing load gate would not have made.
#[test]
fn a_refusal_inside_the_estimate_margin_band_is_suppressed() {
    // 20 GiB weights + 18 GiB headroom = 38 GiB unwidened resident floor. Every implemented rung
    // is `Missing` beyond resident, so the ladder has nowhere to go; at 39 GiB the unwidened floor
    // FITS while 38 * 1.10 = 41.8 does not, which is exactly the band.
    let generator = fixture_generator(Some(fixture_contract(20, 4, &[])));
    let banded = admit_video_generation(&generator, inputs(241, budget(39.0), 18 * GIB));
    assert_eq!(
        banded,
        VideoAdmissionOutcome::default(),
        "a job whose unwidened floor still fits runs today, and must keep running"
    );

    // Below the unwidened floor the refusal IS emitted, so the suppression above is a band
    // property and not a blanket "never refuse".
    let refused = admit_video_generation(&generator, inputs(241, budget(30.0), 18 * GIB));
    let message = refused.refusal.expect("30 GiB cannot hold a 38 GiB floor");
    assert!(message.starts_with("ltx_2_3: "), "{message}");
    assert!(message.contains("1280x704 x 241 frames"), "{message}");
    assert!(refused.memory.is_none());
}

/// **The suppression is SCOPED to the shape it claims, and this is the test that says so.**
///
/// The guard exists to swallow one specific refusal: the estimate margin applied to a peak that IS
/// the weights+headroom floor. Comparing only "does the floor fit the budget" is correct today —
/// `floor_phase_peaks` makes every peak be that floor — and becomes a **planted OOM** the moment
/// sc-18829 lands a fitted per-phase peak: this epic's own measured LTX figures are ~94.3 GB at
/// decode against a ~38 GB weights floor, so on a host that fits 38 but not 94.3, an unscoped
/// guard would suppress a genuine all-rungs-reject and run the job resident into an OOM.
///
/// Driven at the pure predicate because the peak cannot yet be made to exceed the floor through
/// the real selector — that is the whole hazard. The numbers are the epic's measured ones so the
/// case under test is the real future one, not an invented shape.
#[test]
fn a_rejection_whose_peak_exceeds_the_floor_beyond_the_margin_is_not_suppressed() {
    const MARGIN: f64 = crate::ladder_margin_policy::MLX_ESTIMATE_MARGIN;
    // ~38 GB weights+headroom resident floor (20 GiB weights + 18 GiB headroom).
    let floor_bytes = 38 * GIB;
    let floor_gb = crate::memory_strategy::peak_bytes_to_gb(floor_bytes);
    let widened_floor_gb = crate::memory_strategy::peak_bytes_to_gb(
        crate::memory_strategy::widened_peak_bytes(floor_bytes, MARGIN),
    );
    // A host that comfortably holds the floor but not the fitted decode peak.
    let host_gb = 100.0;
    assert!(floor_gb < host_gb, "{floor_gb} vs {host_gb}");

    // sc-18829's shape: the rejected peak is the fitted DECODE peak, far above the floor.
    let fitted_decode_reject_gb = 94.3 * (1.0 + MARGIN);
    assert!(
        fitted_decode_reject_gb > host_gb,
        "the fixture must be a genuine rejection on this host: {fitted_decode_reject_gb} vs \
         {host_gb}"
    );
    assert!(
        !refusal_is_a_margin_artifact(fitted_decode_reject_gb, floor_bytes, MARGIN, Some(host_gb),),
        "a rejection whose peak exceeds the weights floor by more than the margin is REAL and \
         must survive — suppressing it runs the job into an OOM"
    );

    // Today's shape, on the identical floor/host/margin: the rejected peak IS the widened floor,
    // so the suppression still applies. Without this the assertion above could be satisfied by a
    // guard that never suppresses anything.
    assert!(
        refusal_is_a_margin_artifact(widened_floor_gb, floor_bytes, MARGIN, Some(host_gb)),
        "a rejection at exactly the widened floor is the margin artifact this guard exists for"
    );
    // And the boundary is where the doc says it is: one ULP past the widened floor survives.
    assert!(
        !refusal_is_a_margin_artifact(
            widened_floor_gb + f64::EPSILON * widened_floor_gb,
            floor_bytes,
            MARGIN,
            Some(host_gb),
        ),
        "the scope check is `<= widened floor`, so anything above it is out of scope"
    );
}

/// The second conjunct — the non-regression condition proper — is independent of the first.
#[test]
fn a_floor_that_does_not_fit_is_never_suppressed_and_no_budget_never_suppresses() {
    const MARGIN: f64 = crate::ladder_margin_policy::MLX_ESTIMATE_MARGIN;
    let floor_bytes = 38 * GIB;
    let floor_gb = crate::memory_strategy::peak_bytes_to_gb(floor_bytes);
    let widened_floor_gb = crate::memory_strategy::peak_bytes_to_gb(
        crate::memory_strategy::widened_peak_bytes(floor_bytes, MARGIN),
    );

    // In scope (the peak IS the floor) but the floor itself does not fit: a real refusal the
    // pre-existing load gate would also have made.
    assert!(!refusal_is_a_margin_artifact(
        widened_floor_gb,
        floor_bytes,
        MARGIN,
        Some(floor_gb - 1.0),
    ));
    // Exactly at the floor still fits — the comparison is `<=`, matching `mlx_fit_gate`'s.
    assert!(refusal_is_a_margin_artifact(
        widened_floor_gb,
        floor_bytes,
        MARGIN,
        Some(floor_gb),
    ));
    // No budget signal: never suppress. `select_strategy` cannot even produce a `Reject` without
    // one, so this is the never-block-without-evidence posture applied in the safe direction.
    assert!(!refusal_is_a_margin_artifact(
        widened_floor_gb,
        floor_bytes,
        MARGIN,
        None,
    ));
}

/// `estimate_evidence` gained a `backend` parameter for this story. The image lane must still
/// write `Mlx`, which was the hardcoded value before.
#[test]
fn the_image_lanes_estimate_evidence_still_keys_to_mlx() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: gen_core::MemoryStrategyParameters::default(),
        tier: tier(),
    };
    let evidence = crate::mlx_fit_gate::estimate_evidence(
        &contract,
        gen_core::MemoryBackend::Mlx,
        tier(),
        "text_to_image",
        None,
        MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        selection,
        1,
        None,
    );
    assert_eq!(evidence.key.backend, gen_core::MemoryBackend::Mlx);
    assert_eq!(evidence.key.geometry.frames, 1);

    // ...and the video lane can key to Candle, so the parameter is genuinely load-bearing rather
    // than a constant with extra steps.
    let candle = crate::mlx_fit_gate::estimate_evidence(
        &contract,
        gen_core::MemoryBackend::Candle,
        tier(),
        VIDEO_MODE_KEY,
        None,
        MemoryGeometry {
            width: 1280,
            height: 704,
            batch: 1,
            frames: 241,
            reference_count: 0,
        },
        selection,
        1,
        None,
    );
    assert_eq!(candle.key.backend, gen_core::MemoryBackend::Candle);
    assert_eq!(candle.key.geometry.frames, 241);
}

/// End-to-end through core's gate with the real selector: an unrouted family never reaches it.
#[test]
fn an_unrouted_family_never_reaches_the_shared_selector() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let mut selector = LadderVideoSelector::new(
        VideoRequestIdentity {
            route: "video_admission_fixture",
            lane: VideoLane::Candle,
            tier: tier(),
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &contract,
        budget(8.0),
        18 * GIB,
    );
    assert_eq!(
        video_admission(
            "krea_realtime_14b",
            VideoLane::Candle,
            1280,
            704,
            241,
            &mut selector
        ),
        VideoAdmission::NotRouted
    );
    assert!(selector.selections.is_empty());
}

/// M15's target. `frames` is what makes a video evidence cell distinguishable from every other
/// frame count at the same resolution, and it is the field sc-18829's temporal term multiplies.
#[test]
fn the_gen_core_geometry_carries_the_real_frame_count() {
    let mapped = video_memory_geometry(geometry(241, VideoGeometryRole::Requested));
    assert_eq!(mapped.frames, 241);
    assert_eq!(mapped.width, 1280);
    assert_eq!(mapped.height, 704);
    assert_eq!(mapped.batch, 1);
    // A different frame count maps to a different cell, so the assertion above is not satisfied by
    // any constant.
    assert_eq!(
        video_memory_geometry(geometry(305, VideoGeometryRole::Requested)).frames,
        305
    );
    // Degenerate zero frames floor to one rather than producing an unkeyable cell.
    assert_eq!(
        video_memory_geometry(geometry(0, VideoGeometryRole::Requested)).frames,
        1
    );
}

/// A CANDLE-lane request drives `LadderVideoSelector::select` end to end through core's gate.
///
/// The gap this closes: every other selection test above runs `VideoLane::Mlx`, so the candle half
/// of `memory_strategy::candidate_exclusion`'s `request.backend == contract.backend.backend_id()`
/// agreement (`memory_strategy.rs:446-457`) was never exercised — `LadderVideoSelector` puts
/// `lane.as_key()` on one side and the loaded provider's realization on the other, and nothing
/// checked they line up on candle. Also the only test that grades against
/// `CANDLE_ESTIMATE_MARGIN` rather than the MLX one.
///
/// Both directions are asserted: a candle lane against a candle contract SELECTS, and the same
/// candle lane against the MLX contract every other test uses is excluded — so the agreement is
/// shown to be load-bearing rather than incidentally satisfied.
#[test]
fn the_candle_lane_selects_end_to_end_against_a_candle_contract() {
    let candle_contract = fixture_contract_with_realization(
        20,
        4,
        &[MemoryStrategy::StagedResidency],
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    // 20 GiB weights + 18 GiB headroom = 38 GiB resident, widened by the 4% CANDLE estimate margin
    // to 39.52. Staged drops co-residency to max(4, 16) = 16, i.e. 34 GiB -> 35.36 widened. A 38 GiB
    // host therefore refuses resident and admits staged.
    let mut selector = LadderVideoSelector::new(
        VideoRequestIdentity {
            route: "video_admission_fixture",
            lane: VideoLane::Candle,
            tier: tier(),
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &candle_contract,
        budget(38.0),
        18 * GIB,
    );
    // `ltx_2_3` is candle-routed, so core's gate reaches the selector rather than short-circuiting.
    let verdict = video_admission("ltx_2_3", VideoLane::Candle, 1280, 704, 241, &mut selector);
    let VideoAdmission::Admitted { rung, .. } = verdict else {
        panic!("expected a candle-lane admission, got {verdict:?}");
    };
    assert_eq!(rung, StrategyRung::StagedResidency);
    assert_eq!(selector.selections.len(), 1);
    // The evidence really did key to candle, not to the MLX default.
    assert_eq!(
        LadderVideoSelector::new(
            VideoRequestIdentity {
                route: "video_admission_fixture",
                lane: VideoLane::Candle,
                tier: tier(),
                expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
            },
            &candle_contract,
            budget(38.0),
            18 * GIB,
        )
        .backend(),
        gen_core::MemoryBackend::Candle
    );

    // The SAME candle-lane request against an MLX-realization contract is excluded by the shared
    // `candidate_exclusion` backend agreement, so nothing is selected. That is what makes the
    // assertion above about the agreement holding rather than about 38 GiB being roomy.
    let mlx_contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let mut mismatched = LadderVideoSelector::new(
        VideoRequestIdentity {
            route: "video_admission_fixture",
            lane: VideoLane::Candle,
            tier: tier(),
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &mlx_contract,
        budget(38.0),
        18 * GIB,
    );
    assert_eq!(
        video_admission(
            "ltx_2_3",
            VideoLane::Candle,
            1280,
            704,
            241,
            &mut mismatched
        ),
        VideoAdmission::Undecidable,
        "a candle-lane request must not grade against an MLX provider's contract"
    );
    assert!(mismatched.selections.is_empty());
}

/// M16's target. The lane the gate runs on decides which backend its evidence keys to; collapsing
/// both lanes onto one backend would make a candle cell indistinguishable from an MLX one.
#[test]
fn each_lane_keys_its_evidence_to_its_own_backend() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let selector = |lane| {
        LadderVideoSelector::new(
            VideoRequestIdentity {
                route: "video_admission_fixture",
                lane,
                tier: tier(),
                expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
            },
            &contract,
            budget(128.0),
            18 * GIB,
        )
        .backend()
    };
    assert_eq!(selector(VideoLane::Mlx), gen_core::MemoryBackend::Mlx);
    assert_eq!(selector(VideoLane::Candle), gen_core::MemoryBackend::Candle);
}

/// A rung the provider cannot execute must never be selected for a video request.
///
/// This module deliberately restates none of that policy — `memory_strategy::candidate_exclusion`
/// runs `contract.validate_selection` on every candidate — so what is pinned here is that the
/// SHARED exclusion is genuinely in force on the video lane, not that a local copy of it exists.
/// `BoundedTransformerResidency` requires `LoadShape::DeferredMaterialization` (gen-core
/// `MemoryStrategy::requires`), so a provider declaring it `Implemented` under an eager load shape
/// must not reach it even on a host where it is the only rung that fits.
#[test]
fn a_rung_whose_prerequisite_is_unmet_is_not_offered() {
    let eager = with_load_shape(
        fixture_contract(20, 4, &[MemoryStrategy::BoundedTransformerResidency]),
        LoadShape::EagerMaterialization,
    );
    // 20 GiB weights + 18 GiB headroom = 38 GiB resident (41.8 widened); rung 4 sheds the whole
    // 16 GiB transformer, so its floor is 22 GiB (24.2 widened). A 30 GiB host can hold ONLY
    // rung 4 — which is exactly why offering it here would be the harm.
    let (verdict, selections) = select_once(
        &eager,
        budget(30.0),
        18 * GIB,
        geometry(241, VideoGeometryRole::Requested),
    );
    assert!(
        !matches!(verdict, VideoRungSelection::Selected { .. }),
        "rung 4 is unreachable under an eager load shape, so 30 GiB has nothing to select: \
         {verdict:?}"
    );
    assert!(selections.is_empty());

    // The SAME contract under the deferred shape the rung requires DOES reach it, so the
    // assertion above is about the prerequisite and not about 30 GiB being too small.
    let deferred = with_load_shape(
        fixture_contract(20, 4, &[MemoryStrategy::BoundedTransformerResidency]),
        LoadShape::DeferredMaterialization,
    );
    let (reachable, _) = select_once(
        &deferred,
        budget(30.0),
        18 * GIB,
        geometry(241, VideoGeometryRole::Requested),
    );
    assert!(
        matches!(
            reachable,
            VideoRungSelection::Selected {
                rung: StrategyRung::BoundedTransformerResidency,
                ..
            }
        ),
        "{reachable:?}"
    );
}

// --------------------------------------------------------------------------------------------
// The seam sc-18829 substitutes into: per-phase peaks, reduced to a scalar as late as possible.
// --------------------------------------------------------------------------------------------

/// The admission number is a MAX over phases, not an aggregate. sc-18810 measured every candidate
/// temporal form missing the aggregate by >= 10.26 GiB while landing at 0.019-0.44 GiB per phase,
/// so a phase-resolved prediction is the only accurate one.
#[test]
fn the_admission_peak_is_the_max_over_phases() {
    let peaks = PhasePeaks {
        conditioning_bytes: 32 * GIB,
        denoise_bytes: 20 * GIB,
        decode_bytes: 2 * GIB,
    };
    assert_eq!(peaks.peak_bytes(), 32 * GIB);
    // Each phase can be the max, so the reduction is not reading one fixed field.
    assert_eq!(
        PhasePeaks {
            conditioning_bytes: GIB,
            denoise_bytes: 40 * GIB,
            decode_bytes: 2 * GIB,
        }
        .peak_bytes(),
        40 * GIB
    );
    assert_eq!(
        PhasePeaks {
            conditioning_bytes: GIB,
            denoise_bytes: 2 * GIB,
            decode_bytes: 94 * GIB,
        }
        .peak_bytes(),
        94 * GIB
    );
}

/// Which phase binds is a property of the GEOMETRY. It genuinely varies — measured, text binds at
/// 11,904 latent tokens and decode at 14,080 inside one model's envelope — so nothing may cache
/// "the" binding phase for a model.
#[test]
fn the_binding_phase_varies_and_ties_resolve_to_the_later_phase() {
    let text_binds = PhasePeaks {
        conditioning_bytes: 33 * GIB,
        denoise_bytes: 21 * GIB,
        decode_bytes: 3 * GIB,
    };
    let decode_binds = PhasePeaks {
        conditioning_bytes: 33 * GIB,
        denoise_bytes: 21 * GIB,
        decode_bytes: 94 * GIB,
    };
    let denoise_binds = PhasePeaks {
        conditioning_bytes: 10 * GIB,
        denoise_bytes: 21 * GIB,
        decode_bytes: 3 * GIB,
    };
    assert_eq!(text_binds.binding_phase(), VideoBindingPhase::Conditioning);
    assert_eq!(denoise_binds.binding_phase(), VideoBindingPhase::Denoise);
    assert_eq!(decode_binds.binding_phase(), VideoBindingPhase::Decode);

    // Ties resolve to the LATER phase, matching `mlx_fit_gate::binding_phase` on the same triple.
    assert_eq!(
        PhasePeaks {
            conditioning_bytes: 21 * GIB,
            denoise_bytes: 21 * GIB,
            decode_bytes: 21 * GIB,
        }
        .binding_phase(),
        VideoBindingPhase::Decode
    );
}

/// The floor is phase-blind today, and its scalar is byte-identical to the pre-seam number — so
/// introducing the phase shape changed no prediction. That is the point: sc-18829 changes what the
/// three values ARE, not where the scalar is taken.
#[test]
fn the_floor_is_phase_uniform_and_its_peak_is_the_unchanged_scalar() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let engaged = contract.engaged_composition(MemoryStrategy::StagedResidency);
    let peaks = floor_phase_peaks(&contract, &engaged, 18 * GIB);
    let scalar = crate::mlx_fit_gate::estimate_floor_weights_bytes(&contract, &engaged)
        .saturating_add(18 * GIB);
    assert_eq!(peaks.peak_bytes(), scalar);
    assert_eq!(peaks.conditioning_bytes, scalar);
    assert_eq!(peaks.denoise_bytes, scalar);
    assert_eq!(peaks.decode_bytes, scalar);
    // Staged residency really does reduce the floor, so `scalar` is not a constant the assertions
    // above could match trivially.
    let resident = floor_phase_peaks(
        &contract,
        &contract.engaged_composition(MemoryStrategy::Resident),
        18 * GIB,
    );
    assert!(
        resident.peak_bytes() > peaks.peak_bytes(),
        "resident {} vs staged {}",
        resident.peak_bytes(),
        peaks.peak_bytes()
    );
    // And the headroom really is added, so the phase values are not the bare weights.
    assert_eq!(
        floor_phase_peaks(&contract, &engaged, 0).peak_bytes() + 18 * GIB,
        scalar
    );
}
