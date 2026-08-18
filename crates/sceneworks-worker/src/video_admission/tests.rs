//! sc-18814: the video gate reaches the shared ladder selector, and the constants
//! `sceneworks-core` transcribed from gen-core still match the pinned bundle.

use gen_core::{
    LoadShape, LoadSpec, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase, MemoryStrategyCapability,
    MemoryStrategySupport, MemoryWindowMaterialization, Precision, Quant, VaeTiling, WeightsSource,
};
use sceneworks_core::video_memory_curves::VideoCurveHullPoint;
use sceneworks_core::video_request::{
    single_pass_decode_frame_cap, vae_full_res_channels, video_admission, VideoAdmission,
    VideoAdmissionGeometry, VideoDecodePass, VideoGeometryRole,
};

use super::*;

const GIB: u64 = 1024 * 1024 * 1024;

#[test]
fn candle_post_load_snapshot_preserves_committed_pressure_under_a_cap() {
    let budget = candle_budget_from_total_free(24 * GIB, 10 * GIB, Some(16 * GIB))
        .expect("valid CUDA snapshot");
    assert_eq!(budget.total_bytes, 16 * GIB);
    assert_eq!(budget.committed_bytes, 14 * GIB);
    assert_eq!(budget.reclaimable_bytes, 0);
    assert_eq!(budget.reserved_headroom_bytes, 2 * GIB);
    assert!(candle_budget_from_total_free(24 * GIB, 25 * GIB, None).is_none());
}
const FITTED_CURVE_CLOSURE: &str =
    "87a27d5dcab7bfcbe962fb0cb6cd16a75e8e04f2c194bcaa0b14f633d4ff5db3";

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
    let mut contract = MemoryProviderContract::compatibility_default("ltx_2_3", realization);
    contract.asset_facts.base_bytes = base_gib * GIB;
    contract.asset_facts.conditioning_bytes = conditioning_gib * GIB;
    contract.asset_facts.transformer_bytes = (base_gib - conditioning_gib) * GIB;
    contract.asset_facts.decoder_bytes = 0;
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.calibration = Some(MemoryCalibrationIdentity {
        abi: gen_core::MEMORY_CALIBRATION_ABI,
        fingerprint: "sc-18808-ltx-2-3-mlx-t2v-staged-capture-v1".to_owned(),
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
                // Empty preserves the legacy route-blind parameter contract for this fixture.
                decode_geometry_policies: Vec::new(),
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
        decode_pass_frames: frames,
        batch: 1,
        decode_pass: VideoDecodePass::SinglePass,
        role,
    }
}

fn tiered_decode_profile(
    _lane: VideoLane,
    _provider_id: &str,
    _geometry: VideoAdmissionGeometry,
    selection: MemorySelection,
) -> Result<Option<ResolvedVideoDecodeProfile>, String> {
    let (working_set_bytes, evidence_revision) = match selection.strategy {
        MemoryStrategy::BoundedDecode => (4 * GIB, "video-provider-selected-decode-profile-v1"),
        _ => (35 * GIB, "video-provider-conservative-decode-profile-v1"),
    };
    Ok(Some(ResolvedVideoDecodeProfile {
        profile: VideoDecodeMemoryProfile::new(working_set_bytes, 0)
            .expect("fixture decode profile is internally consistent"),
        evidence_revision,
    }))
}

fn decoder_substitution_profile(
    _lane: VideoLane,
    _provider_id: &str,
    _geometry: VideoAdmissionGeometry,
    _selection: MemorySelection,
) -> Result<Option<ResolvedVideoDecodeProfile>, String> {
    Ok(Some(ResolvedVideoDecodeProfile {
        profile: VideoDecodeMemoryProfile::new(10 * GIB, 4 * GIB)
            .expect("fixture decode profile is internally consistent"),
        evidence_revision: "video-provider-conservative-decode-profile-v1",
    }))
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
fn expected_vae_tiling(id: &str, lane: VideoLane) -> Option<VaeTiling> {
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
        "svd" => match lane {
            VideoLane::Mlx => None,
            VideoLane::Candle => Some(VaeTiling {
                spatial_scale: 8,
                temporal_scale: 1,
                causal_temporal: false,
                full_res_channels: 256,
            }),
        },
        other => panic!(
            "shipped video model {other:?} is not mapped to a pinned VaeTiling — read the \
             VaeTiling its decoder passes to `budgeted_plan` out of that engine's crate and add \
             it to `expected_vae_tiling`; do not blanket-apply a default (sc-18814)"
        ),
    }
}

#[test]
fn core_transcribes_the_pinned_vae_write_bounds() {
    for lane in [VideoLane::Mlx, VideoLane::Candle] {
        let mut modelled = 0_usize;
        let mut unmodelled = 0_usize;
        for model in shipped_video_model_ids() {
            match expected_vae_tiling(&model, lane) {
                Some(vae) => {
                    assert_eq!(
                        vae_full_res_channels(&model, lane),
                        Some(vae.full_res_channels as u32),
                        "{lane:?}/{model}: core's transcribed channel count must equal the \
                         pinned gen_core::VaeTiling constant its decoder runs"
                    );
                    modelled += 1;
                }
                None => {
                    assert_eq!(
                        vae_full_res_channels(&model, lane),
                        None,
                        "{lane:?}/{model} is deliberately unmodelled; core must report None \
                         rather than a number nothing can pin"
                    );
                    unmodelled += 1;
                }
            }
        }
        assert_eq!(modelled + unmodelled, EXPECTED_SHIPPED_VIDEO_COUNT);
        assert!(modelled > 0);
        assert_eq!(unmodelled, usize::from(lane == VideoLane::Mlx));
    }

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

#[cfg(target_os = "macos")]
#[test]
fn mlx_svd_remains_unmodelled_in_core() {
    assert_eq!(vae_full_res_channels("svd", VideoLane::Mlx), None);
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_svd_transcription_is_pinned_to_the_provider_owned_runtime_authority() {
    let provider = runtime_cuda::vae_tiling("svd_xt")
        .expect("advance the SceneWorks inference pin to the sc-19104 Candle SVD authority");
    assert_eq!(
        provider,
        VaeTiling {
            spatial_scale: 8,
            temporal_scale: 1,
            causal_temporal: false,
            full_res_channels: 256,
        }
    );
    assert_eq!(
        vae_full_res_channels("svd", VideoLane::Candle),
        Some(provider.full_res_channels as u32)
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
                single_pass_decode_frame_cap(model, VideoLane::Mlx, width, height),
                Some(u32::try_from(engine).unwrap()),
                "{model} @ {width}x{height}: core must agree with VaeTiling::writable_frame_cap"
            );
        }
    }
    assert_eq!(
        single_pass_decode_frame_cap("svd", VideoLane::Mlx, 1024, 576),
        None
    );
    assert_eq!(
        single_pass_decode_frame_cap("svd", VideoLane::Candle, 1024, 576),
        Some(14)
    );
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
            model_id: "ltx_2_3",
            model_family: "ltx-video",
            route: "ltx_2_3",
            mode: "text_to_video",
            reference_count: 0,
            overlay: None,
            lane: VideoLane::Mlx,
            tier: tier(),
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        contract,
        budget,
        headroom_bytes,
        0,
    );
    let verdict = selector.select(geometry);
    (
        verdict,
        selector
            .selections
            .into_iter()
            .map(|candidate| (candidate.binding_geometry, candidate.selection))
            .collect(),
    )
}

/// The promoted sc-18810 curve used as a historical, structurally valid fixture. The fixture
/// contract above adopts the artifact's exact provider/fingerprint identity; mutating the bundle's
/// identity would correctly sever its immutable source-record handshake and make `evaluate` fail
/// closed before these selector tests reached the coefficient under test.
fn fixture_curve_bundle() -> VideoMemoryCurveBundle {
    let bundle = sceneworks_core::video_memory_curves::packaged_video_memory_curves()
        .expect("packaged video curve")
        .clone();
    assert_eq!(bundle.curves.len(), 1);
    assert_eq!(bundle.curves[0].closure_digest, FITTED_CURVE_CLOSURE);
    bundle
}

fn select_once_with_curves(
    contract: &MemoryProviderContract,
    curves: &VideoMemoryCurveBundle,
    budget: Option<Budget>,
    geometry: VideoAdmissionGeometry,
) -> VideoRungSelection {
    let mut selector = selector_with_curves(contract, Some(curves), budget);
    selector.select(geometry)
}

fn selector_with_curves<'a>(
    contract: &'a MemoryProviderContract,
    curves: Option<&'a VideoMemoryCurveBundle>,
    budget: Option<Budget>,
) -> LadderVideoSelector<'a> {
    LadderVideoSelector::with_curve_bundle(
        VideoRequestIdentity {
            model_id: "ltx_2_3",
            model_family: "ltx-video",
            route: "ltx_2_3",
            mode: "text_to_video",
            reference_count: 0,
            overlay: None,
            lane: VideoLane::Mlx,
            tier: tier(),
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: FITTED_CURVE_CLOSURE,
        },
        contract,
        budget,
        18 * GIB,
        0,
        curves,
    )
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
            id: "ltx_2_3",
            family: "ltx",
            backend: "mlx",
            modality: gen_core::Modality::Video,
            capabilities: gen_core::Capabilities::default(),
            required_components: &[],
            control_kinds: None,
            encoder_contract: None,
            denoiser_output_latent_space: None,
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
        model_family: "ltx-video",
        route: "ltx_2_3",
        mode: "text_to_video",
        reference_count: 0,
        overlay: None,
        lane: VideoLane::Mlx,
        tier: tier(),
        width: 1280,
        height: 704,
        frames,
        decode_chunk_size: None,
        fps: 24,
        runtime: budget.map(|budget| VideoRuntimeMemoryState {
            budget: gen_core::MemoryBudget {
                total_bytes: (budget.total_gb * GIB as f64).round() as u64,
                committed_bytes: ((budget.total_gb - budget.available_gb).max(0.0) * GIB as f64)
                    .round() as u64,
                reclaimable_bytes: (budget.reclaimable_gb * GIB as f64).round() as u64,
                reserved_headroom_bytes: (budget.reserved_headroom_gb * GIB as f64).round() as u64,
            },
            cache_state: gen_core::MemoryCacheState::Cold,
            load_policy: gen_core::OffloadPolicy::Resident,
            provider_resident_bytes: 0,
        }),
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
    let outcome =
        admit_video_generation_with_curves(&generator, inputs(241, budget(128.0), 18 * GIB), None);
    assert!(
        outcome.memory.is_none(),
        "resident preserves provider defaults"
    );
    assert!(outcome.refusal.is_none());
    let context = outcome
        .context
        .expect("contract-backed Resident still carries the provider handshake");
    assert_eq!(context.selection.strategy, MemoryStrategy::Resident);
    assert_eq!(context.mode.as_key(), "text_to_video");
    assert_eq!(context.geometry.frames, 241);
    assert!(!context.has_phases);
}

#[test]
fn a_selected_optimized_rung_reaches_the_generation_request() {
    let generator = fixture_generator(Some(fixture_contract(
        20,
        4,
        &[MemoryStrategy::StagedResidency],
    )));
    let outcome =
        admit_video_generation_with_curves(&generator, inputs(241, budget(40.0), 18 * GIB), None);
    let memory = outcome.memory.expect("staged residency was selected");
    assert!(memory.stage_residency, "{memory:?}");
    assert!(outcome.refusal.is_none());
}

#[test]
fn provider_profiles_make_bounded_decode_a_reachable_production_fallback() {
    let generator = fixture_generator(Some(fixture_contract(
        20,
        4,
        &[MemoryStrategy::BoundedDecode],
    )));
    // The conservative resident profile is 20 + 35 = 55 GiB, which cannot fit. The exact
    // bounded carrier retains the generic 20 + 18 = 38 GiB lower bound and fits after the shared
    // 10% estimate margin. Without consuming the selected profile, Resident would win first.
    let outcome = admit_video_generation_with_curves_and_profiles(
        &generator,
        inputs(241, budget(42.0), 18 * GIB),
        None,
        tiered_decode_profile,
    );
    let memory = outcome
        .memory
        .expect("the provider-selected bounded decode carrier must reach the request");
    assert!(memory.tile_vae_decode, "{memory:?}");
    assert_eq!(memory.decode_tile_edge, Some(256));
    assert_eq!(memory.decode_overlap, Some(32));
    let context = outcome
        .context
        .expect("selected profile carries run context");
    assert_eq!(context.selection.strategy, MemoryStrategy::BoundedDecode);
    assert_eq!(context.predicted_peak_bytes, 38 * GIB);
    assert_eq!(
        context.evidence_revision,
        "video-provider-selected-decode-profile-v1"
    );
    assert!(outcome.refusal.is_none());
}

#[test]
fn provider_profile_refusal_is_not_suppressed_by_the_smaller_generic_floor() {
    let generator = fixture_generator(Some(fixture_contract(20, 4, &[])));
    // The historical generic floor (38 GiB) fits this host, but the provider-owned resident
    // profile (55 GiB) does not. Refusal suppression must compare the exact profiled candidate,
    // not reconstruct the smaller generic floor after the selector returns.
    let outcome = admit_video_generation_with_curves_and_profiles(
        &generator,
        inputs(241, budget(39.0), 18 * GIB),
        None,
        tiered_decode_profile,
    );
    let refusal = outcome
        .refusal
        .expect("a provider profile that does not fit is a real refusal");
    assert!(refusal.contains("needs about 60.5 GB"), "{refusal}");
    assert!(outcome.memory.is_none());
    assert!(outcome.context.is_none());
}

#[test]
fn provider_profile_composition_and_warm_residency_are_each_accounted_once() {
    let mut contract = fixture_contract(20, 4, &[]);
    contract.asset_facts.transformer_bytes = 12 * GIB;
    contract.asset_facts.decoder_bytes = 4 * GIB;
    let generator = fixture_generator(Some(contract));
    let mut request = inputs(241, budget(30.0), 0);
    request.runtime = Some(VideoRuntimeMemoryState {
        budget: MemoryBudget {
            total_bytes: 30 * GIB,
            committed_bytes: 20 * GIB,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        cache_state: MemoryCacheState::Warm,
        load_policy: OffloadPolicy::Resident,
        provider_resident_bytes: 20 * GIB,
    });

    let outcome = admit_video_generation_with_curves_and_profiles(
        &generator,
        request,
        None,
        decoder_substitution_profile,
    );
    let context = outcome.context.expect("the 6 GiB incremental peak fits");
    // 20 GiB contract composition + (10 GiB decode profile - 4 GiB decoder already included)
    // = 26 GiB absolute peak. The post-load snapshot then credits the provider's 20 GiB exactly
    // once, leaving 6 GiB of incremental demand.
    assert_eq!(context.predicted_peak_bytes, 6 * GIB);
    assert_eq!(context.budget.committed_bytes, 20 * GIB);
    assert_eq!(
        context.evidence_revision,
        "video-provider-conservative-decode-profile-v1"
    );
    assert!(outcome.memory.is_none());
    assert!(outcome.refusal.is_none());
}

#[cfg(target_os = "macos")]
#[test]
fn packaged_mlx_wan_profiles_expose_conservative_and_selected_sources() {
    let geometry = VideoAdmissionGeometry {
        width: 480,
        height: 480,
        frames: 1,
        decode_pass_frames: 1,
        batch: 1,
        decode_pass: VideoDecodePass::SinglePass,
        role: VideoGeometryRole::Requested,
    };
    let resident = packaged_video_decode_profile(
        VideoLane::Mlx,
        "wan2_2_ti2v_5b",
        geometry,
        MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: tier(),
        },
    )
    .expect("conservative profile lookup succeeds")
    .expect("MLX Wan publishes a conservative decode profile");
    assert_eq!(
        resident.evidence_revision,
        "video-provider-conservative-decode-profile-v1"
    );

    let selected = packaged_video_decode_profile(
        VideoLane::Mlx,
        "wan2_2_ti2v_5b",
        geometry,
        MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: gen_core::MemoryStrategyParameters {
                decode_tile_edge: Some(448),
                decode_overlap: Some(64),
                ..Default::default()
            },
            tier: tier(),
        },
    )
    .expect("selected profile lookup succeeds")
    .expect("MLX Wan publishes the exact bounded-decode carrier profile");
    assert_eq!(
        selected.evidence_revision,
        "video-provider-selected-decode-profile-v1"
    );
    assert!(
        selected.profile.working_set_bytes() <= resident.profile.working_set_bytes(),
        "the selected bounded carrier must not exceed the conservative single-pass profile"
    );
}

#[test]
fn a_curve_cannot_be_relabelled_to_manufacture_bounded_decode_parameters() {
    let contract = fixture_contract(
        60,
        20,
        &[
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ],
    );
    let generator = fixture_generator(Some(contract));
    let mut curves = fixture_curve_bundle();
    curves.curves[0].rung = StrategyRung::BoundedDecode;
    let mut request = inputs(121, budget(40.0), 0);
    request.expected_closure_digest = FITTED_CURVE_CLOSURE;

    let outcome = admit_video_generation_with_curves(&generator, request, Some(&curves));
    assert!(
        outcome.memory.is_none(),
        "relabeling staged source records as bounded-decode evidence must not mint provider knobs"
    );
    assert!(
        outcome.context.is_none(),
        "the source-selector mismatch must fail closed before a lifecycle context is built"
    );
    assert!(
        outcome.refusal.is_some(),
        "the honest 60 GiB fallback cannot fit this 40 GiB fixture"
    );
}

#[test]
fn unsupported_video_surfaces_fail_open_before_contract_selection() {
    let generator = fixture_generator(Some(fixture_contract(
        20,
        4,
        &[MemoryStrategy::StagedResidency],
    )));
    let assert_open = |label: &str, request| {
        assert_eq!(
            admit_video_generation(&generator, request),
            VideoAdmissionOutcome::default(),
            "{label} must preserve legacy direct generation"
        );
    };

    let mut request = inputs(121, budget(128.0), 18 * GIB);
    request.mode = "image_to_video";
    request.reference_count = 1;
    assert_open("I2V", request);

    let mut request = inputs(121, budget(128.0), 18 * GIB);
    request.overlay = Some("enhancer");
    assert_open("enhancer", request);

    for fps in [0, 1, 23, 31, 60] {
        let mut request = inputs(121, budget(128.0), 18 * GIB);
        request.fps = fps;
        assert_open("out-of-envelope FPS", request);
    }

    let mut request = inputs(121, budget(128.0), 18 * GIB);
    request.runtime = None;
    assert_open("missing canonical post-load budget", request);
}

#[test]
fn provider_residency_is_credited_once_and_unrelated_memory_stays_charged() {
    let generator = fixture_generator(Some(fixture_contract(
        20,
        4,
        &[MemoryStrategy::StagedResidency],
    )));
    // Deliberately grow then shrink the unrelated portion across warm requests. A historical
    // pre-load baseline would turn the 27 -> 2 transition into bogus provider credit; the retained
    // cold-load delta must leave the incremental request prediction invariant in both directions.
    for unrelated_gib in [7, 27, 2, 19] {
        let mut request = inputs(241, budget(128.0), 18 * GIB);
        request.runtime = Some(VideoRuntimeMemoryState {
            budget: MemoryBudget {
                total_bytes: 128 * GIB,
                committed_bytes: (20 + unrelated_gib) * GIB,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            cache_state: MemoryCacheState::Warm,
            load_policy: OffloadPolicy::Resident,
            provider_resident_bytes: 20 * GIB,
        });

        let outcome = admit_video_generation_with_curves(&generator, request, None);
        let context = outcome.context.expect("resident selection carries context");
        // The provider's cold-load delta stays fixed while unrelated live pressure shrinks and
        // grows. Incremental demand remains 18 GiB; the live committed budget does not.
        assert_eq!(context.predicted_peak_bytes, 18 * GIB);
        assert_eq!(context.budget.committed_bytes, (20 + unrelated_gib) * GIB);
        assert_eq!(context.cache_state, MemoryCacheState::Warm);
        assert_eq!(context.geometry.frames, 241);
        assert!(
            outcome.memory.is_none(),
            "Resident preserves provider defaults"
        );
    }
}

#[test]
fn resident_attribution_above_a_modeled_rung_fails_closed() {
    let generator = fixture_generator(Some(fixture_contract(
        60,
        20,
        &[MemoryStrategy::StagedResidency],
    )));
    let curves = fixture_curve_bundle();
    let mut request = inputs(121, budget(79.0), 18 * GIB);
    request.expected_closure_digest = FITTED_CURVE_CLOSURE;
    request.runtime = Some(VideoRuntimeMemoryState {
        budget: MemoryBudget {
            total_bytes: 79 * GIB,
            committed_bytes: 60 * GIB,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        cache_state: MemoryCacheState::Warm,
        load_policy: OffloadPolicy::Resident,
        provider_resident_bytes: 60 * GIB,
    });

    let outcome = admit_video_generation_with_curves(&generator, request, Some(&curves));
    let refusal = outcome
        .refusal
        .expect("60 GiB attribution cannot be subtracted from a ~34 GiB fitted peak");
    assert!(refusal.contains("exceeds modeled total peak"), "{refusal}");
    assert!(outcome.memory.is_none());
    assert!(outcome.context.is_none());
}

/// A request above the single-pass cap grades the **cap geometry too**, through the real selector.
///
/// This fixture deliberately presents no current curve (the sentinel closure cannot match the
/// packaged one), so it pins the established fallback behavior: both geometries are graded and use
/// the identical weights-plus-headroom floor. `the_single_pass_cap_geometry_can_bind_the_graded_set`
/// separately proves the fitted path at the cap with an explicitly synthetic test-only envelope.
#[test]
fn a_request_above_the_cap_grades_the_cap_geometry_through_the_real_selector() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let mut selector = LadderVideoSelector::new(
        VideoRequestIdentity {
            model_id: "ltx_2_3",
            model_family: "ltx-video",
            route: "ltx_2_3",
            mode: "text_to_video",
            reference_count: 0,
            overlay: None,
            lane: VideoLane::Mlx,
            tier: tier(),
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &contract,
        budget(40.0),
        18 * GIB,
        0,
    );
    // f305 at 1280x704 is past LTX's 297-frame single-pass cap, so both geometries are graded.
    let verdict = video_admission(
        "ltx_2_3",
        VideoLane::Mlx,
        1280,
        704,
        305,
        None,
        &mut selector,
    );
    assert!(
        matches!(verdict, VideoAdmission::Admitted { .. }),
        "{verdict:?}"
    );
    assert_eq!(selector.selections.len(), 2, "{:?}", selector.selections);
    assert_eq!(selector.selections[0].binding_geometry.frames, 305);
    assert_eq!(
        selector.selections[0].binding_geometry.role,
        VideoGeometryRole::Requested
    );
    assert_eq!(selector.selections[1].binding_geometry.frames, 305);
    assert_eq!(
        selector.selections[1].binding_geometry.decode_pass_frames,
        297
    );
    assert_eq!(
        selector.selections[1].binding_geometry.estimate_frames(),
        297
    );
    assert_eq!(
        selector.selections[1].binding_geometry.role,
        VideoGeometryRole::SinglePassDecodeCap
    );

    // No current curve means both retain the exact historical floor decision.
    assert_eq!(
        selector.selections[0].selection.strategy, selector.selections[1].selection.strategy,
        "the no-current-curve path must retain its phase-uniform floor"
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
    let banded =
        admit_video_generation_with_curves(&generator, inputs(241, budget(39.0), 18 * GIB), None);
    assert_eq!(
        banded,
        VideoAdmissionOutcome::default(),
        "a job whose unwidened floor still fits runs today, and must keep running"
    );

    // Below the unwidened floor the refusal IS emitted, so the suppression above is a band
    // property and not a blanket "never refuse".
    let refused =
        admit_video_generation_with_curves(&generator, inputs(241, budget(30.0), 18 * GIB), None);
    let message = refused.refusal.expect("30 GiB cannot hold a 38 GiB floor");
    assert!(message.starts_with("ltx_2_3: "), "{message}");
    assert!(message.contains("1280x704 x 241 frames"), "{message}");
    assert!(refused.memory.is_none());
}

/// **The suppression is SCOPED to the shape it claims, and this is the test that says so.**
///
/// The guard exists to swallow one specific refusal: the estimate margin applied to a peak that IS
/// the weights+headroom floor. Comparing only "does the floor fit the budget" becomes a **planted
/// OOM** on a fitted per-phase peak: the measured LTX decode can sit far above the weights floor,
/// so on a host that fits the floor but not the phase peak, an unscoped guard would suppress a
/// genuine all-rungs-reject and run the job resident into an OOM.
///
/// The pure predicate pins the exact suppression boundary; the fitted selector tests below drive
/// both the floor and above-floor outcomes end to end.
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

    // The fitted shape: the rejected peak is the DECODE peak, far above the floor.
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

    // The fallback shape, on the identical floor/host/margin: the rejected peak IS the widened
    // floor, so the suppression still applies. Without this the assertion above could be satisfied
    // by a guard that never suppresses anything.
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
        "text_to_video",
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
            model_id: "ltx_2_3",
            model_family: "ltx-video",
            route: "ltx_2_3",
            mode: "text_to_video",
            reference_count: 0,
            overlay: None,
            lane: VideoLane::Candle,
            tier: tier(),
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &contract,
        budget(8.0),
        18 * GIB,
        0,
    );
    assert_eq!(
        video_admission(
            "krea_realtime_14b",
            VideoLane::Candle,
            1280,
            704,
            241,
            None,
            &mut selector
        ),
        VideoAdmission::NotRouted
    );
    assert!(selector.selections.is_empty());
}

/// Requested evidence identity keeps the exact output clip length, while a synthetic cap row uses
/// the interior estimate geometry. Decode chunking must not collapse distinct planned captures.
#[test]
fn the_gen_core_geometry_carries_the_role_aware_estimate_frame_count() {
    let mapped = video_memory_geometry(geometry(241, VideoGeometryRole::Requested), 0);
    assert_eq!(mapped.frames, 241);
    assert_eq!(mapped.width, 1280);
    assert_eq!(mapped.height, 704);
    assert_eq!(mapped.batch, 1);
    // A different frame count maps to a different cell, so the assertion above is not satisfied by
    // any constant.
    assert_eq!(
        video_memory_geometry(geometry(305, VideoGeometryRole::Requested), 0).frames,
        305
    );
    // Degenerate zero frames floor to one rather than producing an unkeyable cell.
    assert_eq!(
        video_memory_geometry(geometry(0, VideoGeometryRole::Requested), 0).frames,
        1
    );

    let chunked = VideoAdmissionGeometry {
        frames: 25,
        decode_pass_frames: 8,
        ..geometry(25, VideoGeometryRole::Requested)
    };
    assert_eq!(
        video_memory_geometry(chunked, 0).frames,
        25,
        "calibration/evidence identity must describe the exact 25-frame capture"
    );

    let shorter_same_chunk = VideoAdmissionGeometry {
        frames: 9,
        decode_pass_frames: 8,
        ..geometry(9, VideoGeometryRole::Requested)
    };
    assert_ne!(
        video_memory_geometry(shorter_same_chunk, 0),
        video_memory_geometry(chunked, 0),
        "f9/chunk8 and f25/chunk8 must remain distinct evidence/cache coordinates"
    );

    let cap = VideoAdmissionGeometry {
        frames: 25,
        decode_pass_frames: 14,
        role: VideoGeometryRole::SinglePassDecodeCap,
        ..geometry(25, VideoGeometryRole::Requested)
    };
    assert_eq!(
        video_memory_geometry(cap, 0).frames,
        14,
        "the synthetic cap row must evaluate and key the interior peak"
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
            model_id: "ltx_2_3",
            model_family: "ltx-video",
            route: "ltx_2_3",
            mode: "text_to_video",
            reference_count: 0,
            overlay: None,
            lane: VideoLane::Candle,
            tier: tier(),
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &candle_contract,
        budget(38.0),
        18 * GIB,
        0,
    );
    // `ltx_2_3` is candle-routed, so core's gate reaches the selector rather than short-circuiting.
    let verdict = video_admission(
        "ltx_2_3",
        VideoLane::Candle,
        1280,
        704,
        241,
        None,
        &mut selector,
    );
    let VideoAdmission::Admitted { rung, .. } = verdict else {
        panic!("expected a candle-lane admission, got {verdict:?}");
    };
    assert_eq!(rung, StrategyRung::StagedResidency);
    assert_eq!(selector.selections.len(), 1);
    // The evidence really did key to candle, not to the MLX default.
    assert_eq!(
        LadderVideoSelector::new(
            VideoRequestIdentity {
                model_id: "ltx_2_3",
                model_family: "ltx-video",
                route: "ltx_2_3",
                mode: "text_to_video",
                reference_count: 0,
                overlay: None,
                lane: VideoLane::Candle,
                tier: tier(),
                calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
                expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
            },
            &candle_contract,
            budget(38.0),
            18 * GIB,
            0,
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
            model_id: "ltx_2_3",
            model_family: "ltx-video",
            route: "ltx_2_3",
            mode: "text_to_video",
            reference_count: 0,
            overlay: None,
            lane: VideoLane::Candle,
            tier: tier(),
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &mlx_contract,
        budget(38.0),
        18 * GIB,
        0,
    );
    assert_eq!(
        video_admission(
            "ltx_2_3",
            VideoLane::Candle,
            1280,
            704,
            241,
            None,
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
                model_id: "ltx_2_3",
                model_family: "ltx-video",
                route: "ltx_2_3",
                mode: "text_to_video",
                reference_count: 0,
                overlay: None,
                lane,
                tier: tier(),
                calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
                expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
            },
            &contract,
            budget(128.0),
            18 * GIB,
            0,
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
// Per-phase peaks, reduced to a scalar as late as possible.
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

/// The fallback floor remains phase-blind, and its scalar is byte-identical to the pre-curve
/// number. Fitted curves change the three values, not where the scalar is taken.
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

// --------------------------------------------------------------------------------------------
// sc-18829 / sc-19020 — fitted cross curves at the real video-selector seam.
// --------------------------------------------------------------------------------------------

#[test]
fn fitted_frames_change_the_selected_outcome_inside_the_measured_hull() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let curves = fixture_curve_bundle();

    let at_121 = select_once_with_curves(
        &contract,
        &curves,
        budget(40.0),
        geometry(121, VideoGeometryRole::Requested),
    );
    assert!(
        matches!(
            at_121,
            VideoRungSelection::Selected {
                rung: StrategyRung::StagedResidency,
                ..
            }
        ),
        "the f121 fitted staged peak still fits 40 GiB: {at_121:?}"
    );

    let at_145 = select_once_with_curves(
        &contract,
        &curves,
        budget(40.0),
        geometry(145, VideoGeometryRole::Requested),
    );
    assert!(
        matches!(at_145, VideoRungSelection::Reject { .. }),
        "the cross term raises the f145 fitted peak past the same budget: {at_145:?}"
    );
}

#[test]
fn fitted_phase_laws_bind_by_exact_geometry_and_reduce_by_max() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let curves = fixture_curve_bundle();
    let selector = selector_with_curves(&contract, Some(&curves), budget(128.0));
    let engaged = contract.engaged_composition(MemoryStrategy::StagedResidency);

    let small = VideoAdmissionGeometry {
        width: 768,
        height: 512,
        frames: 121,
        decode_pass_frames: 121,
        batch: 1,
        decode_pass: VideoDecodePass::SinglePass,
        role: VideoGeometryRole::Requested,
    };
    let (small_peaks, small_basis, closure, curve_id, _) =
        fitted_or_floor_phase_peaks(&selector, small, MemoryStrategy::StagedResidency, &engaged);
    assert_eq!(small_basis, CandidateBasis::EstimateFittedCurve);
    assert_eq!(closure, FITTED_CURVE_CLOSURE);
    assert_eq!(
        curve_id,
        Some("ltx_2_3:ltx-video:ltx_2_3:mlx:q8:text_to_video:staged_residency:eager_materialization:b1:abi3:single_pass:87a27d5dcab7:sc-18808-ltx-2-3-mlx-t2v-staged-capture-v1")
    );
    assert_eq!(small_peaks.binding_phase(), VideoBindingPhase::Conditioning);
    assert_eq!(small_peaks.peak_bytes(), small_peaks.conditioning_bytes);

    let large = geometry(145, VideoGeometryRole::Requested);
    let (large_peaks, large_basis, ..) =
        fitted_or_floor_phase_peaks(&selector, large, MemoryStrategy::StagedResidency, &engaged);
    assert_eq!(large_basis, CandidateBasis::EstimateFittedCurve);
    assert_eq!(large_peaks.binding_phase(), VideoBindingPhase::Decode);
    assert_eq!(large_peaks.peak_bytes(), large_peaks.decode_bytes);
    assert!(large_peaks.decode_bytes > small_peaks.decode_bytes);
    assert!(large_peaks.denoise_bytes > small_peaks.denoise_bytes);
    assert!(large_peaks.conditioning_bytes > small_peaks.conditioning_bytes);
}

#[test]
fn mutating_the_ratified_cross_coefficient_changes_selector_outcome() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let original = fixture_curve_bundle();
    let geometry = geometry(145, VideoGeometryRole::Requested);
    let original_verdict = select_once_with_curves(&contract, &original, budget(41.0), geometry);
    assert!(
        matches!(original_verdict, VideoRungSelection::Reject { .. }),
        "the generated decode cross coefficient must bind: {original_verdict:?}"
    );

    let mut mutated = original.clone();
    mutated.curves[0].phases.decode.per_mpx_frame_gb = 0.0;
    let mutated_verdict = select_once_with_curves(&contract, &mutated, budget(41.0), geometry);
    assert!(
        matches!(
            mutated_verdict,
            VideoRungSelection::Selected {
                rung: StrategyRung::StagedResidency,
                ..
            }
        ),
        "removing the decode cross term must change the admission result: {mutated_verdict:?}"
    );
}

#[test]
fn mutating_a_phase_residual_changes_the_admission_decision() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let original = fixture_curve_bundle();
    let request = geometry(121, VideoGeometryRole::Requested);
    let original_verdict = select_once_with_curves(&contract, &original, budget(40.0), request);
    assert!(
        matches!(original_verdict, VideoRungSelection::Selected { .. }),
        "the shipped residual-bounded curve must fit the bracket: {original_verdict:?}"
    );

    let mut mutated = original.clone();
    mutated.curves[0].phases.decode.max_residual_gb += 12.0;
    let mutated_verdict = select_once_with_curves(&contract, &mutated, budget(40.0), request);
    assert!(
        matches!(mutated_verdict, VideoRungSelection::Reject { .. }),
        "the residual is admission evidence, not report-only metadata: {mutated_verdict:?}"
    );
}

#[test]
fn historical_q8_curve_fixture_is_tier_exact_while_q4_and_bf16_keep_an_honest_floor() {
    // Structural old-closure fixture only. The final SC-19109 provider fingerprint/closure must
    // fail this SC-18808 artifact closed until SC-18946 physically reseeds and refits it.
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let curves = fixture_curve_bundle();
    let request = geometry(145, VideoGeometryRole::Requested);
    for (quant, expected_basis) in [
        (Some(Quant::Q8), CandidateBasis::EstimateFittedCurve),
        (Some(Quant::Q4), CandidateBasis::EstimateFloor),
        (None, CandidateBasis::EstimateFloor),
    ] {
        let mut selector = selector_with_curves(&contract, Some(&curves), budget(128.0));
        selector.identity.tier.quant = quant;
        let engaged = contract.engaged_composition(MemoryStrategy::StagedResidency);
        let (_, basis, ..) = fitted_or_floor_phase_peaks(
            &selector,
            request,
            MemoryStrategy::StagedResidency,
            &engaged,
        );
        assert_eq!(basis, expected_basis, "checkpoint tier {quant:?}");
    }

    let generator = fixture_generator(Some(contract));
    for quant in [Some(Quant::Q8), Some(Quant::Q4), None] {
        for cache_state in [MemoryCacheState::Cold, MemoryCacheState::Warm] {
            let mut request = inputs(121, budget(40.0), 18 * GIB);
            request.tier.quant = quant;
            request.expected_closure_digest = FITTED_CURVE_CLOSURE;
            request.runtime.as_mut().unwrap().cache_state = cache_state;
            let outcome = admit_video_generation_with_curves(&generator, request, Some(&curves));
            assert!(
                outcome.refusal.is_none(),
                "tier {quant:?} {cache_state:?} must pass safety admission"
            );
            let context = outcome
                .context
                .expect("a selected tier carries safety context");
            assert_eq!(context.cache_state, cache_state);
            if quant == Some(Quant::Q8) {
                assert!(context.evidence_revision.contains("single_pass"));
            } else {
                assert_eq!(context.evidence_revision, "video-estimate-floor-v1");
            }
        }
    }
}

#[test]
fn checkpoint_bound_ltx_tiers_drive_curve_or_floor_safety_on_cold_and_warm_requests() {
    // This deliberately joins the two production seams the focused resolver test keeps separate:
    // the same tiny on-disk split manifest that determines the provider load determines the tier
    // submitted to admission, while `LoadSpec.quantize` remains absent as it does for LTX jobs.
    let checkpoint = tempfile::tempdir().expect("ltx split fixture");
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let generator = fixture_generator(Some(contract));
    let curves = fixture_curve_bundle();
    for (label, manifest, expected_quant, expected_revision) in [
        (
            "q8",
            Some(r#"{"quantized":true,"quantization_bits":8}"#),
            Some(Quant::Q8),
            "single_pass",
        ),
        (
            "q4",
            Some(r#"{"quantized":true,"quantization_bits":4}"#),
            Some(Quant::Q4),
            "video-estimate-floor-v1",
        ),
        ("bf16", None, None, "video-estimate-floor-v1"),
    ] {
        let split = checkpoint.path().join("split_model.json");
        match manifest {
            Some(raw) => std::fs::write(&split, raw).expect("write split manifest"),
            None if split.exists() => std::fs::remove_file(&split).expect("remove split manifest"),
            None => {}
        }
        let spec = LoadSpec::new(WeightsSource::Dir(checkpoint.path().to_owned()));
        assert_eq!(spec.quantize, None, "{label} has no request-side tier hint");
        let resolved = crate::mlx_fit_gate::resolved_video_numeric_tier("ltx_2_3", &spec)
            .unwrap_or_else(|error| panic!("{label} checkpoint tier resolves: {error}"));
        assert_eq!(resolved.quant, expected_quant, "{label}");

        for cache_state in [MemoryCacheState::Cold, MemoryCacheState::Warm] {
            let mut request = inputs(121, budget(40.0), 18 * GIB);
            request.tier = resolved;
            request.expected_closure_digest = FITTED_CURVE_CLOSURE;
            request.runtime.as_mut().unwrap().cache_state = cache_state;
            let outcome = admit_video_generation_with_curves(&generator, request, Some(&curves));
            assert!(
                outcome.refusal.is_none(),
                "{label}/{cache_state:?} must survive selector and provider safety"
            );
            let context = outcome
                .context
                .unwrap_or_else(|| panic!("{label}/{cache_state:?} carries safety context"));
            assert_eq!(context.cache_state, cache_state);
            if expected_quant == Some(Quant::Q8) {
                assert!(context.evidence_revision.contains(expected_revision));
            } else {
                assert_eq!(context.evidence_revision, expected_revision);
            }
        }
    }

    std::fs::write(
        checkpoint.path().join("split_model.json"),
        r#"{"quantized":true,"quantization_bits":8}"#,
    )
    .expect("write mismatch split manifest");
    let mismatched =
        LoadSpec::new(WeightsSource::Dir(checkpoint.path().to_owned())).with_quant(Quant::Q4);
    for cache_state in [MemoryCacheState::Cold, MemoryCacheState::Warm] {
        let error = crate::mlx_fit_gate::resolved_video_numeric_tier("ltx_2_3", &mismatched)
            .expect_err("an explicit q4 assertion cannot price a loaded q8 checkpoint")
            .to_string();
        assert!(
            error.contains("disagrees"),
            "mismatch must fail closed on {cache_state:?}: {error}"
        );
    }
}

fn assert_curve_mismatch_falls_back(
    label: &str,
    contract: &MemoryProviderContract,
    curves: &VideoMemoryCurveBundle,
    geometry: VideoAdmissionGeometry,
) {
    let expected = {
        let mut selector = selector_with_curves(contract, None, budget(40.0));
        selector.select(geometry)
    };
    let actual = select_once_with_curves(contract, curves, budget(40.0), geometry);
    assert_eq!(
        actual, expected,
        "{label}: an inapplicable fitted curve must preserve the prior floor decision"
    );
    assert!(
        matches!(
            actual,
            VideoRungSelection::Selected {
                rung: StrategyRung::StagedResidency,
                ..
            }
        ),
        "{label}: the fallback must be a real floor selection, not merely undecidable: {actual:?}"
    );
}

#[test]
fn every_identity_or_envelope_mismatch_falls_back_to_the_unchanged_floor() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let inside = geometry(145, VideoGeometryRole::Requested);

    let mut curves = fixture_curve_bundle();
    curves.curves[0].backend = VideoCurveBackend::Candle;
    assert_curve_mismatch_falls_back("foreign lane", &contract, &curves, inside);

    let mut curves = fixture_curve_bundle();
    curves.curves[0].closure_digest = "0".repeat(64);
    assert_curve_mismatch_falls_back("stale closure", &contract, &curves, inside);

    let mut curves = fixture_curve_bundle();
    curves.curves[0].calibration_abi += 1;
    assert_curve_mismatch_falls_back("foreign calibration ABI", &contract, &curves, inside);

    let curves = fixture_curve_bundle();
    let mut selector = selector_with_curves(&contract, Some(&curves), budget(40.0));
    selector.identity.calibration_abi += 1;
    let live_abi_mismatch = selector.select(inside);
    let mut no_curves = selector_with_curves(&contract, None, budget(40.0));
    assert_eq!(
        live_abi_mismatch,
        no_curves.select(inside),
        "a pinned gen-core ABI bump must stale the old curve even if the fixture contract's own \
         calibration identity still says 3"
    );

    let mut stale_contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    stale_contract.calibration.as_mut().unwrap().abi += 1;
    let selector = selector_with_curves(&stale_contract, Some(&curves), budget(40.0));
    let engaged = stale_contract.engaged_composition(MemoryStrategy::StagedResidency);
    let (_, basis, ..) =
        fitted_or_floor_phase_peaks(&selector, inside, MemoryStrategy::StagedResidency, &engaged);
    assert_eq!(
        basis,
        CandidateBasis::EstimateFloor,
        "a contract identity minted under another ABI must not match the pinned ABI merely because \
         its fingerprint string remained the same"
    );

    let mut curves = fixture_curve_bundle();
    curves.curves[0].calibration_fingerprint = "stale-fixture".to_owned();
    assert_curve_mismatch_falls_back("stale fingerprint", &contract, &curves, inside);

    let mut curves = fixture_curve_bundle();
    curves.curves[0].tier = "bf16".to_owned();
    assert_curve_mismatch_falls_back("unsupported tier", &contract, &curves, inside);

    let mut curves = fixture_curve_bundle();
    curves.curves[0].model_id = "ltx_2_3_eros".to_owned();
    assert_curve_mismatch_falls_back("unsupported model", &contract, &curves, inside);

    let mut curves = fixture_curve_bundle();
    curves.curves[0].model_family = "ltx-custom".to_owned();
    assert_curve_mismatch_falls_back("unsupported family", &contract, &curves, inside);

    let mut curves = fixture_curve_bundle();
    curves.curves[0].mode = "image_to_video".to_owned();
    assert_curve_mismatch_falls_back("unsupported mode", &contract, &curves, inside);

    let curves = fixture_curve_bundle();
    let mut tiled = inside;
    tiled.decode_pass = VideoDecodePass::Tiled;
    assert_curve_mismatch_falls_back("tiling discontinuity", &contract, &curves, tiled);

    assert_curve_mismatch_falls_back(
        "outside measured voxel hull",
        &contract,
        &curves,
        geometry(241, VideoGeometryRole::Requested),
    );

    let outside_area = VideoAdmissionGeometry {
        width: 1920,
        height: 1080,
        frames: 121,
        decode_pass_frames: 121,
        batch: 1,
        decode_pass: VideoDecodePass::SinglePass,
        role: VideoGeometryRole::Requested,
    };
    assert_curve_mismatch_falls_back(
        "outside measured area hull",
        &contract,
        &curves,
        outside_area,
    );
}

#[test]
fn the_single_pass_cap_geometry_can_bind_the_graded_set() {
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let mut curves = fixture_curve_bundle();
    // Structural wiring fixture only: the committed campaign did NOT measure 1280x704 f297. Extend
    // the test copy's convex hull to that cap so this test can prove core passes the cap's exact
    // geometry/regime into the fitted selector without claiming production evidence for it.
    curves.curves[0].measured_geometry_hull = vec![
        VideoCurveHullPoint {
            pixels: 393_216,
            voxels: 393_216 * 121,
        },
        VideoCurveHullPoint {
            pixels: 901_120,
            voxels: 901_120 * 121,
        },
        VideoCurveHullPoint {
            pixels: 901_120,
            voxels: 901_120 * 297,
        },
        VideoCurveHullPoint {
            pixels: 393_216,
            voxels: 393_216 * 361,
        },
    ];

    let requested = VideoAdmissionGeometry {
        decode_pass: VideoDecodePass::Tiled,
        ..geometry(305, VideoGeometryRole::Requested)
    };
    assert!(
        matches!(
            select_once_with_curves(&contract, &curves, budget(40.0), requested),
            VideoRungSelection::Selected {
                rung: StrategyRung::StagedResidency,
                ..
            }
        ),
        "the tiled request itself falls back to the 37.4 GiB staged floor"
    );

    let mut selector = selector_with_curves(&contract, Some(&curves), budget(40.0));
    let verdict = video_admission(
        "ltx_2_3",
        VideoLane::Mlx,
        1280,
        704,
        305,
        None,
        &mut selector,
    );
    let VideoAdmission::Refused { geometry, .. } = verdict else {
        panic!("the fitted f297 cap must reject the graded set: {verdict:?}");
    };
    assert_eq!(geometry.frames, 305);
    assert_eq!(geometry.decode_pass_frames, 297);
    assert_eq!(geometry.estimate_frames(), 297);
    assert_eq!(geometry.decode_pass, VideoDecodePass::SinglePass);
    assert_eq!(geometry.role, VideoGeometryRole::SinglePassDecodeCap);
    assert_eq!(
        selector.selections.len(),
        1,
        "only the tiled request selected"
    );
}

#[test]
fn same_rung_cap_binding_carries_cap_peak_but_actual_request_geometry() {
    let contract = fixture_contract(90, 45, &[MemoryStrategy::StagedResidency]);
    let generator = fixture_generator(Some(contract.clone()));
    let mut curves = fixture_curve_bundle();
    // Structural fixture only: extend the copy to the 297-frame cap. Production remains bounded by
    // the generated campaign hull and makes no claim at this geometry.
    curves.curves[0].measured_geometry_hull = vec![
        VideoCurveHullPoint {
            pixels: 393_216,
            voxels: 393_216 * 121,
        },
        VideoCurveHullPoint {
            pixels: 901_120,
            voxels: 901_120 * 121,
        },
        VideoCurveHullPoint {
            pixels: 901_120,
            voxels: 901_120 * 297,
        },
        VideoCurveHullPoint {
            pixels: 393_216,
            voxels: 393_216 * 361,
        },
    ];
    let cap = VideoAdmissionGeometry {
        width: 1280,
        height: 704,
        frames: 305,
        decode_pass_frames: 297,
        batch: 1,
        decode_pass: VideoDecodePass::SinglePass,
        role: VideoGeometryRole::SinglePassDecodeCap,
    };
    let expected_peak = {
        let selector = selector_with_curves(&contract, Some(&curves), budget(95.0));
        let engaged = contract.engaged_composition(MemoryStrategy::StagedResidency);
        fitted_or_floor_phase_peaks(&selector, cap, MemoryStrategy::StagedResidency, &engaged)
            .0
            .peak_bytes()
    };

    let mut request = inputs(305, budget(95.0), 0);
    request.expected_closure_digest = FITTED_CURVE_CLOSURE;
    let outcome = admit_video_generation_with_curves(&generator, request, Some(&curves));
    let context = outcome
        .context
        .expect("requested and cap geometries both select staged");
    assert_eq!(context.selection.strategy, MemoryStrategy::StagedResidency);
    assert_eq!(
        context.predicted_peak_bytes, expected_peak,
        "same-rung tie must retain the cap's higher raw fitted peak"
    );
    assert_eq!(
        context.geometry.frames, 305,
        "provider scope validates the actual request, never the binding cap geometry"
    );
    assert_eq!(context.geometry.width, 1280);
    assert_eq!(context.geometry.height, 704);
    assert_eq!(
        context.evidence_revision,
        "ltx_2_3:ltx-video:ltx_2_3:mlx:q8:text_to_video:staged_residency:eager_materialization:b1:abi3:single_pass:87a27d5dcab7:sc-18808-ltx-2-3-mlx-t2v-staged-capture-v1"
    );
    assert!(outcome.refusal.is_none());
}
