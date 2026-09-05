//! sc-18814: the video gate reaches the shared ladder selector, and the constants
//! `sceneworks-core` transcribed from gen-core still match the pinned bundle.

use gen_core::{
    Generator, LoadShape, LoadSpec, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase, MemoryResidentComponent,
    MemoryStrategyCapability, MemoryStrategySupport, MemoryWindowMaterialization, Precision, Quant,
    VaeTiling, WeightsSource,
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

/// The flat activation-headroom term every fixture below hands the selector (`inputs(.., 18 * GIB)`).
/// sc-22508 charges a floor's allowance against THIS term alone, so the fixtures state it once.
const FIXTURE_HEADROOM_GIB: u64 = 18;

/// A host budget derived FROM the per-term policy: a `peak_gib` GiB estimate-backed FLOOR whose
/// ACTIVATION term is `activation_gib` GiB, at the admitted ceiling `select_strategy` grades it at
/// (in integer bytes, exactly as the selector computes it), plus `slack_gb`. Fixtures use this to
/// sit a budget in a window — admit every candidate whose admitted peak is at or under this
/// floor's, refuse every wider one — without hardcoding the policy's arithmetic into magic floats
/// that rot when the allowance moves (sc-18094, re-termed by sc-22508).
///
/// The activation term is a PARAMETER, not the fixture headroom constant, because production
/// derives it from the peak it built: a generic floor's activation is the headroom, but a provider
/// decode profile raises the peak and its activation slice with it (`profiled - weights`).
fn mlx_widened_floor_gb(peak_gib: u64, activation_gib: u64, slack_gb: f64) -> f64 {
    crate::memory_strategy::peak_bytes_to_gb(crate::memory_strategy::floor_admitted_peak_bytes(
        gen_core::MemoryBackend::Mlx,
        peak_gib * GIB,
        Some(activation_gib * GIB),
    )) + slack_gb
}

/// [`mlx_widened_floor_gb`] for the common case: a GENERIC weights+headroom floor, whose activation
/// term is exactly the fixture headroom every `inputs(.., 18 * GIB)` hands the selector.
fn mlx_widened_gb(peak_gib: u64, slack_gb: f64) -> f64 {
    mlx_widened_floor_gb(peak_gib, FIXTURE_HEADROOM_GIB, slack_gb)
}

/// The ADMITTED ceiling of the fitted staged candidate at one geometry, derived end to end from
/// the same helpers production uses — the pattern
/// `mutating_the_ratified_cross_coefficient_changes_selector_outcome` established, shared so every
/// host window in this file moves with an allowance change instead of freezing a float
/// (epic 22505 feature-end fix round: the floor allowance re-derivation moved every
/// resident-floor-anchored window).
fn fitted_staged_admitted_gb(
    contract: &MemoryProviderContract,
    curves: &VideoMemoryCurveBundle,
    geometry: VideoAdmissionGeometry,
) -> f64 {
    let selector = selector_with_curves(contract, Some(curves), budget(0.0));
    let engaged = contract.engaged_composition(MemoryStrategy::StagedResidency);
    let (peaks, basis, ..) = fitted_or_floor_phase_peaks(
        &selector,
        geometry,
        MemoryStrategy::StagedResidency,
        &engaged,
    );
    assert_eq!(
        basis,
        CandidateBasis::EstimateFittedCurve,
        "the window is only meaningful while the curve is what decides"
    );
    crate::memory_strategy::peak_bytes_to_gb(crate::memory_strategy::admitted_peak_bytes(
        crate::ladder_margin_policy::AdmissionSubject {
            backend: gen_core::MemoryBackend::Mlx,
            basis,
            closure_is_stale: false,
            unmodeled_activation_bytes: None,
        },
        peaks.peak_bytes(),
    ))
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
const EXPECTED_SHIPPED_VIDEO_COUNT: usize = 13;

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
        "ltx_2_3" | "ltx_2_3_eros" | "ltx_2_5" => Some(VaeTiling::LTX),
        // The dense TI2V-5B is welded to the z48 vae22 (`mlx-gen-wan/src/pipeline.rs:235`).
        "wan_2_2" => Some(VaeTiling::WAN22),
        // The A14B grid and every Wan-derived renderer decode through the Wan2.1 z16 VAE.
        "wan_2_2_t2v_14b"
        | "wan_2_2_i2v_14b"
        | "wan_2_2_vace_fun_14b"
        | "bernini"
        | "scail2_14b"
        | "krea_realtime_14b" => Some(VaeTiling::WAN),
        // MiniMax-H3 remains deliberately unmodelled on both lanes. Its Candle provider uses the
        // reference-parity fixed 256/64 spatial tiler rather than gen-core's writable-element
        // VaeTiling planner; inventing a VaeTiling transcription would claim a decode cap the
        // provider does not consume.
        "minimax_h3" | "minimax_h3_ref" => None,
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
        // MiniMax-H3's pair is unmodelled on both lanes for the provider-specific reason above;
        // SVD is additionally unmodelled on MLX.
        assert_eq!(unmodelled, 2 + usize::from(lane == VideoLane::Mlx));
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
            reference_shape: "none",
            fps: 30,
            overlay: None,
            lane: VideoLane::Mlx,
            tier: tier(),
            transformer_variant: Some(Ltx25TransformerVariant::Distilled),
            decoder: Some(Ltx25Decoder::Conv),
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
    let mut bundle = sceneworks_core::video_memory_curves::packaged_video_memory_curves()
        .expect("packaged video curve")
        .clone();
    // Select the fixture BY its closure digest rather than asserting the packaged bundle happens to
    // hold exactly one curve and taking `[0]` (sc-20799 round 2). The bundle grows a curve every
    // time a new selector key is fitted; a positional read would then silently hand these tests a
    // different model's coefficients, and the `len() == 1` guard would have failed for a reason
    // that says nothing about what this fixture needs.
    let fixture = bundle
        .curves
        .iter()
        .find(|curve| curve.closure_digest == FITTED_CURVE_CLOSURE)
        .unwrap_or_else(|| {
            panic!("packaged video curves omit the sc-18810 fixture closure {FITTED_CURVE_CLOSURE}")
        })
        .clone();
    bundle.curves = vec![fixture];
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
            reference_shape: "none",
            fps: 30,
            overlay: None,
            lane: VideoLane::Mlx,
            tier: tier(),
            transformer_variant: Some(Ltx25TransformerVariant::Distilled),
            decoder: Some(Ltx25Decoder::Conv),
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
    // 20 GiB of weights (4 conditioning + 16 transformer) + 18 GiB headroom = 38 GiB resident.
    // Staged drops the co-residency to max(4, 16) = 16, i.e. a 34 GiB floor. sc-22508 admits each
    // at its peak plus the allowance on its ACTIVATION term alone (both carry the same 18 GiB
    // headroom), so the two admitted ceilings stay 4 GiB apart exactly as the raw floors are. A
    // host 0.5 GiB above the admitted staged floor therefore refuses resident and admits staged:
    // the ladder's whole point. Both ceilings come from `mlx_widened_gb`, so no number here rots.
    let host_gb = mlx_widened_gb(34, 0.5);
    assert!(host_gb < mlx_widened_gb(38, 0.0), "the window must exist");
    let contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    let (verdict, _) = select_once(
        &contract,
        budget(host_gb),
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
        budget(host_gb),
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
        reference_shape: "none",
        overlay: None,
        lane: VideoLane::Mlx,
        tier: tier(),
        transformer_variant: Some(Ltx25TransformerVariant::Distilled),
        decoder: Some(Ltx25Decoder::Conv),
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

fn bernini_inputs<'a>(overlay: Option<&'a str>) -> VideoAdmissionInputs<'a> {
    let mut request = inputs(45, budget(128.0), 0);
    request.model_id = "bernini";
    request.model_family = "bernini";
    request.route = "bernini";
    request.mode = "video_to_video";
    request.reference_count = 1;
    request.reference_shape = "video";
    request.width = 848;
    request.height = 480;
    request.fps = 16;
    request.overlay = overlay;
    request
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

/// sc-22512 (epic 22505, E8) split the old
/// `bernini_v2v_is_refused_for_missing_or_crossed_evidence` in two. Bernini was the ONE video lane
/// that turned a MISSING measurement into a refusal, while every other provider abstains on the
/// same branch. This test keeps the half that is decidable without any measurement — a request
/// outside the supported surface — and the companion below asserts the half that changed.
#[test]
fn bernini_v2v_is_refused_for_an_unsupported_request_surface() {
    let generator = fixture_generator(Some(fixture_contract(20, 4, &[MemoryStrategy::Resident])));
    let mut exact = inputs(45, budget(128.0), 0);
    exact.model_id = "bernini";
    exact.model_family = "bernini";
    exact.route = "bernini";
    exact.mode = "video_to_video";
    exact.reference_count = 1;
    exact.reference_shape = "video";
    exact.width = 848;
    exact.height = 480;
    exact.fps = 16;
    exact.overlay = Some("provider_video_mode:no_audio");
    let outcome = admit_video_generation(&generator, exact);
    assert!(outcome
        .refusal
        .as_deref()
        .is_some_and(|message| message.contains("exact surface")));

    let mut crossed = inputs(45, budget(128.0), 0);
    crossed.model_id = "bernini";
    crossed.model_family = "bernini";
    crossed.route = "bernini";
    crossed.mode = "video_to_video";
    crossed.reference_count = 1;
    crossed.reference_shape = "image";
    crossed.width = 640;
    crossed.height = 640;
    crossed.fps = 24;
    let outcome = admit_video_generation(&generator, crossed);
    assert!(outcome
        .refusal
        .as_deref()
        .is_some_and(|message| message.contains("exact surface")));

    // ...but a generator publishing NO contract fails open, even on the surface that refuses above.
    // `bernini_surface_is_exact` answers `false` for a missing contract because there is no declared
    // surface to compare against — that is absence, not an unsupported request, and every branch
    // below the evidence check fails open for exactly this case. Without `contract.is_some()` in the
    // predicate, this same request would be surface-refused and the gate would contradict itself.
    let contractless = fixture_generator(None);
    let mut exact = inputs(45, budget(128.0), 0);
    exact.model_id = "bernini";
    exact.model_family = "bernini";
    exact.route = "bernini";
    exact.mode = "video_to_video";
    exact.reference_count = 1;
    exact.reference_shape = "video";
    exact.width = 848;
    exact.height = 480;
    exact.fps = 16;
    exact.overlay = Some("provider_video_mode:no_audio");
    let outcome = admit_video_generation(&contractless, exact);
    assert_eq!(
        outcome,
        VideoAdmissionOutcome::default(),
        "a Bernini request against a contractless generator must FAIL OPEN, not be surface-refused: \
         {outcome:?}"
    );
}

/// sc-22512 (epic 22505, E8): a WELL-FORMED Bernini request on a coordinate nobody has calibrated
/// is no longer denied — the gate abstains, exactly as it already did for every other provider
/// reaching the same branch.
///
/// This is the leg that changed. The request below is on Bernini's exact supported surface (V2V,
/// FPS16, 45 frames, a public geometry, a supported tier, the `provider_video_mode:v2v` overlay
/// receipt) and the packaged corpus carries no evidence covering it. That used to produce
/// "Bernini video_to_video memory admission refused: no current calibrated evidence matches
/// route=…", so the absence of a measurement blocked the job outright rather than widening its
/// estimate. Absence never blocks; runtime catching (E6) is the failure posture.
///
/// The negative control is the sibling test above: an unsupported SURFACE still refuses, which is
/// decidable without any measurement at all. Without that pairing this test would pass just as well
/// against a gate that had been deleted outright.
#[test]
fn an_uncalibrated_but_well_formed_bernini_request_is_not_refused_for_missing_evidence() {
    let generator = fixture_generator(Some(fixture_contract(20, 4, &[MemoryStrategy::Resident])));
    let mut exact = inputs(45, budget(128.0), 0);
    exact.model_id = "bernini";
    exact.model_family = "bernini";
    exact.route = "bernini";
    exact.mode = "video_to_video";
    exact.reference_count = 1;
    exact.reference_shape = "video";
    exact.width = 848;
    exact.height = 480;
    exact.fps = 16;
    exact.overlay = Some("provider_video_mode:v2v");
    let outcome = admit_video_generation(&generator, exact);
    assert_eq!(
        outcome,
        VideoAdmissionOutcome::default(),
        "an uncalibrated coordinate on the supported surface must ABSTAIN, not refuse: {outcome:?}"
    );
}

#[test]
fn bernini_v2v_consumes_the_loaded_adapter_receipt_without_reconstructing_it() {
    const RECEIPT: &str = "adapters:[artifact=safetensors;path_hex=2f746d702f612e7361666574656e736f7273;digest=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;kind=Lora;scale_bits=3f000001;pass_scale_bits=3e800000/3f400000;expert=Some(High);verified_bytes=12345;stable=true]";
    let contract_with_receipt = |resident_bytes| {
        let mut contract = fixture_contract(20, 4, &[MemoryStrategy::Resident]);
        contract.asset_facts.overlay_bytes = resident_bytes;
        contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![MemoryFormulaVariable::OverlayBytes],
            resident_components: vec![MemoryResidentComponent {
                id: RECEIPT.to_owned(),
                kind: MemoryComponentKind::AdapterStack,
                resident_bytes,
                bounded_by: None,
                residency: MemoryComponentResidency::WholeRender,
            }],
        };
        contract
    };
    let exact_axis = bernini_adapter_receipt_axis(RECEIPT);
    let exact_overlay = format!("{exact_axis}+provider_video_mode:v2v");
    let exact = bernini_inputs(Some(&exact_overlay));

    for resident_bytes in [0, 12_345] {
        let contract = contract_with_receipt(resident_bytes);
        assert!(
            bernini_surface_is_exact(&exact, Some(&contract)),
            "dense-folded zero-byte and packed-additive receipts are both identity-bearing"
        );
    }

    // These mutations cover every receipt axis the provider sealed. The worker does not parse or
    // round any of them; it compares the complete typed provider receipt carried by the contract.
    for crossed in [
        RECEIPT.replace("scale_bits=3f000001", "scale_bits=3f000000"),
        RECEIPT.replace("pass_scale_bits=3e800000/3f400000", "pass_scale_bits=none"),
        RECEIPT.replace("kind=Lora", "kind=Lokr"),
        RECEIPT.replace("expert=Some(High)", "expert=Some(Low)"),
        RECEIPT.replace("verified_bytes=12345", "verified_bytes=12346"),
        RECEIPT.replace("digest=sha256:aa", "digest=sha256:ba"),
    ] {
        let contract = contract_with_receipt(12_345);
        let crossed_axis = bernini_adapter_receipt_axis(&crossed);
        let crossed_overlay = format!("{crossed_axis}+provider_video_mode:v2v");
        let crossed_request = bernini_inputs(Some(&crossed_overlay));
        assert!(
            !bernini_surface_is_exact(&crossed_request, Some(&contract)),
            "crossed receipt was accepted: {crossed}"
        );
    }
    assert!(!bernini_surface_is_exact(&exact, None));
}

fn r2v_conditioning() -> Vec<gen_core::Conditioning> {
    vec![gen_core::Conditioning::MultiReference {
        images: vec![
            gen_core::Image {
                width: 640,
                height: 360,
                pixels: vec![11; 640 * 360 * 3],
            },
            gen_core::Image {
                width: 360,
                height: 640,
                pixels: vec![29; 360 * 640 * 3],
            },
        ],
    }]
}

fn rv2v_conditioning() -> Vec<gen_core::Conditioning> {
    let mut conditioning = r2v_conditioning();
    let [gen_core::Conditioning::MultiReference { images }] = conditioning.as_mut_slice() else {
        unreachable!()
    };
    let images = std::mem::take(images);
    let frame = gen_core::Image {
        width: 848,
        height: 480,
        pixels: vec![7; 848 * 480 * 3],
    };
    vec![
        gen_core::Conditioning::VideoClip {
            frames: vec![frame; 45],
            frame_idx: 0,
            strength: 1.0,
        },
        gen_core::Conditioning::MultiReference { images },
    ]
}

fn mv2v_conditioning() -> Vec<gen_core::Conditioning> {
    let first = gen_core::Image {
        width: 848,
        height: 480,
        pixels: vec![7; 848 * 480 * 3],
    };
    let second = gen_core::Image {
        width: 848,
        height: 480,
        pixels: vec![13; 848 * 480 * 3],
    };
    [first, second]
        .into_iter()
        .map(|frame| gen_core::Conditioning::VideoClip {
            frames: vec![frame; 45],
            frame_idx: 0,
            strength: 1.0,
        })
        .collect()
}

#[test]
fn bernini_r2v_worker_receipts_bind_backend_specific_effective_shapes() {
    let conditioning = r2v_conditioning();
    let mlx = bernini_r2v_reference_receipt(VideoLane::Mlx, 848, 480, &conditioning).unwrap();
    assert_eq!(mlx, "bernini-r2v-references-v2:backend-mlx:source-preprocess-full-vae624-v1:count-2:0:native-640x360;vit-280x168;vae-624x352|1:native-360x640;vit-168x280;vae-352x624+bernini-r2v-request-seal-v1-cd11cf62ec83e85860e1790538062a88b39ae384d2956fd1dc54c0e45d6fa8f5");
    let candle = bernini_r2v_reference_receipt(VideoLane::Candle, 848, 480, &conditioning).unwrap();
    assert_eq!(candle, "bernini-r2v-references-v2:backend-candle:source-preprocess-full-vae624-v1:count-2:0:native-640x360;vit-280x168;vae-624x352|1:native-360x640;vit-168x280;vae-352x624+bernini-r2v-request-seal-v1-cd11cf62ec83e85860e1790538062a88b39ae384d2956fd1dc54c0e45d6fa8f5");

    let mut duplicate = r2v_conditioning();
    let [gen_core::Conditioning::MultiReference { images }] = duplicate.as_mut_slice() else {
        unreachable!()
    };
    images[1] = images[0].clone();
    assert!(
        bernini_r2v_reference_receipt(VideoLane::Mlx, 848, 480, &duplicate).is_ok(),
        "distinct public asset ids may legitimately resolve to identical pixels"
    );
}

#[test]
fn bernini_r2v_curve_identity_reuses_shapes_while_request_seal_binds_pixels() {
    let first = r2v_conditioning();
    let mut second = first.clone();
    let [gen_core::Conditioning::MultiReference { images }] = second.as_mut_slice() else {
        unreachable!()
    };
    images[0].pixels[0] ^= 1;

    let first_receipt = bernini_r2v_reference_receipt(VideoLane::Mlx, 848, 480, &first).unwrap();
    let second_receipt = bernini_r2v_reference_receipt(VideoLane::Mlx, 848, 480, &second).unwrap();
    assert_ne!(
        first_receipt, second_receipt,
        "the post-admission byte seal changes"
    );

    let first_overlay = format!("provider_video_mode:r2v+{first_receipt}");
    let second_overlay = format!("provider_video_mode:r2v+{second_receipt}");
    let first_curve = video_curve_overlay(Some(&first_overlay)).unwrap();
    let second_curve = video_curve_overlay(Some(&second_overlay)).unwrap();
    assert_eq!(
        first_curve, second_curve,
        "same ordered native/effective shapes must reuse one fitted curve"
    );
    assert!(!first_curve.contains("request-seal"));
}

#[test]
fn bernini_mv2v_curve_identity_excludes_only_the_request_seal() {
    let first = mv2v_conditioning();
    let mut changed_bytes = first.clone();
    let [gen_core::Conditioning::VideoClip { frames, .. }, ..] = changed_bytes.as_mut_slice()
    else {
        unreachable!()
    };
    frames[0].pixels[0] ^= 1;
    let mut reordered = first.clone();
    reordered.reverse();
    let mut different_surface = first.clone();
    different_surface.push(first[0].clone());

    let receipt = bernini_mv2v_clip_receipt(VideoLane::Mlx, 848, 480, &first).unwrap();
    let changed_bytes_receipt =
        bernini_mv2v_clip_receipt(VideoLane::Mlx, 848, 480, &changed_bytes).unwrap();
    let reordered_receipt =
        bernini_mv2v_clip_receipt(VideoLane::Mlx, 848, 480, &reordered).unwrap();
    let different_surface_receipt =
        bernini_mv2v_clip_receipt(VideoLane::Mlx, 848, 480, &different_surface).unwrap();
    assert_ne!(
        receipt, changed_bytes_receipt,
        "full receipt binds clip bytes"
    );
    assert_ne!(
        receipt, reordered_receipt,
        "full receipt binds ordered clips"
    );

    let curve = video_curve_overlay(Some(&format!("provider_video_mode:mv2v+{receipt}"))).unwrap();
    let changed_bytes_curve = video_curve_overlay(Some(&format!(
        "provider_video_mode:mv2v+{changed_bytes_receipt}"
    )))
    .unwrap();
    let reordered_curve = video_curve_overlay(Some(&format!(
        "provider_video_mode:mv2v+{reordered_receipt}"
    )))
    .unwrap();
    let different_surface_curve = video_curve_overlay(Some(&format!(
        "provider_video_mode:mv2v+{different_surface_receipt}"
    )))
    .unwrap();
    assert_eq!(
        curve, changed_bytes_curve,
        "request bytes are not a curve axis"
    );
    assert_eq!(
        curve, reordered_curve,
        "equal source surfaces reuse one curve"
    );
    assert_ne!(
        curve, different_surface_curve,
        "source count and source-ID schedule remain curve axes"
    );
    assert!(!curve.contains(BERNINI_MV2V_SEAL_DOMAIN));
}

#[test]
fn bernini_ads2v_receipt_binds_distinct_clip_roles_images_and_three_to_four_source_boundary() {
    let clip = |pixel| gen_core::Conditioning::VideoClip {
        frames: vec![
            gen_core::Image {
                width: 848,
                height: 480,
                pixels: vec![pixel; 848 * 480 * 3]
            };
            45
        ],
        frame_idx: 0,
        strength: 1.0,
    };
    let mut first = vec![
        clip(7),
        clip(13),
        gen_core::Conditioning::MultiReference {
            images: vec![gen_core::Image {
                width: 640,
                height: 360,
                pixels: vec![29; 640 * 360 * 3],
            }],
        },
    ];
    let receipt = bernini_ads2v_source_receipt(VideoLane::Mlx, 848, 480, &first).unwrap();
    assert!(receipt.contains("count-3:"));
    assert!(receipt.contains("source-ids-1,2,3"));
    assert!(
        receipt.contains("source-video:")
            && receipt.contains("reference-video:")
            && receipt.contains("image-1:")
    );
    let curve = video_curve_overlay(Some(&format!("provider_video_mode:ads2v+{receipt}"))).unwrap();
    assert!(!curve.contains(BERNINI_ADS2V_SEAL_DOMAIN));
    let mut swapped = first.clone();
    swapped.swap(0, 1);
    assert_ne!(
        receipt,
        bernini_ads2v_source_receipt(VideoLane::Mlx, 848, 480, &swapped).unwrap()
    );
    let gen_core::Conditioning::MultiReference { images } = &mut first[2] else {
        unreachable!()
    };
    images.push(gen_core::Image {
        width: 360,
        height: 640,
        pixels: vec![31; 360 * 640 * 3],
    });
    let four = bernini_ads2v_source_receipt(VideoLane::Candle, 848, 480, &first).unwrap();
    assert!(four.contains("count-4:") && four.contains("source-ids-1,1.666667,2.333333,3"));
}

#[test]
fn bernini_full_vae_shape_matches_providers_two_stage_bankers_rounding() {
    assert_eq!(bernini_full_vae_shape(24, 625), (32, 624));
}

#[test]
fn bernini_r2v_exact_surface_reaches_the_shared_selector_with_flattened_count() {
    let mut contract = fixture_contract(20, 4, &[MemoryStrategy::BoundedDecode]);
    contract.provider_id = "bernini".to_owned();
    let receipt =
        bernini_r2v_reference_receipt(VideoLane::Mlx, 848, 480, &r2v_conditioning()).unwrap();
    let overlay = format!("provider_video_mode:r2v+{receipt}");
    let generator = fixture_generator(Some(contract));
    for cache_state in [MemoryCacheState::Cold, MemoryCacheState::Warm] {
        let mut request = bernini_inputs(Some(&overlay));
        request.mode = "reference_to_video";
        request.reference_count = 2;
        request.reference_shape = "multi_image";
        request.runtime.as_mut().unwrap().cache_state = cache_state;
        assert!(bernini_surface_is_exact(
            &request,
            generator.memory_strategy_contract()
        ));

        let outcome = admit_video_generation_with_curves(&generator, request, None);
        assert!(
            outcome.context.is_some(),
            "shared selector context: {outcome:?}"
        );
        let context = outcome.context.unwrap();
        assert_eq!(context.mode.as_key(), "reference_to_video");
        assert_eq!(context.geometry.reference_count, 2);
        assert_eq!(context.overlay.as_deref(), Some(overlay.as_str()));
        assert_eq!(context.selection.strategy, MemoryStrategy::Resident);
        assert_eq!(context.cache_state, cache_state);
    }

    for (count, shape, mode) in [
        (1, "multi_image", "reference_to_video"),
        (2, "video", "reference_to_video"),
        (2, "multi_image", "video_to_video"),
    ] {
        let mut crossed = bernini_inputs(Some(&overlay));
        crossed.mode = mode;
        crossed.reference_count = count;
        crossed.reference_shape = shape;
        assert!(!bernini_surface_is_exact(
            &crossed,
            generator.memory_strategy_contract()
        ));
    }
}

#[test]
fn bernini_rv2v_seals_the_combined_source_tokens_and_distinguishes_clip_plus_image() {
    let contract = {
        let mut contract = fixture_contract(20, 4, &[MemoryStrategy::BoundedDecode]);
        contract.provider_id = "bernini".to_owned();
        contract
    };
    let conditioning = rv2v_conditioning();
    for (lane, packed_tokens, image_tokens) in [
        (VideoLane::Mlx, 12_012_u64, 858_u64),
        (VideoLane::Candle, 12_012_u64, 858_u64),
    ] {
        let receipt = bernini_r2v_reference_receipt(lane, 848, 480, &conditioning).unwrap();
        assert!(receipt.contains(&format!("count-2:packed-source-tokens-{packed_tokens}:")));
        assert!(receipt.contains("source-preprocess-full-vae624-v1"));
        assert!(receipt.contains("video-1:frames-45;native-848x480;vae-12x78x44;tokens-10296"));
        assert!(receipt.contains(&format!(";tokens-{image_tokens}")));

        let overlay = format!("provider_video_mode:rv2v+{receipt}");
        let exact = || {
            let mut request = bernini_inputs(Some(&overlay));
            request.lane = lane;
            request.mode = "reference_video_to_video";
            request.reference_count = 3;
            request.reference_shape = "video+multi_image";
            request
        };
        assert!(bernini_surface_is_exact(&exact(), Some(&contract)));

        let mut crossed = exact();
        crossed.reference_count = 2;
        assert!(!bernini_surface_is_exact(&crossed, Some(&contract)));
        crossed = exact();
        crossed.width = 1280;
        crossed.height = 720;
        assert!(!bernini_surface_is_exact(&crossed, Some(&contract)));
        crossed = exact();
        crossed.reference_shape = "multi_image";
        assert!(!bernini_surface_is_exact(&crossed, Some(&contract)));
        crossed = exact();
        crossed.mode = "reference_to_video";
        assert!(!bernini_surface_is_exact(&crossed, Some(&contract)));
        crossed = exact();
        let crossed_overlay = crossed.overlay.unwrap().replace(
            "source-preprocess-full-vae624-v1",
            "source-preprocess-renderer-output-v1",
        );
        crossed.overlay = Some(&crossed_overlay);
        assert!(
            !bernini_surface_is_exact(&crossed, Some(&contract)),
            "full Bernini must reject renderer preprocessing evidence"
        );
    }

    let r2v_receipt =
        bernini_r2v_reference_receipt(VideoLane::Mlx, 848, 480, &r2v_conditioning()).unwrap();
    assert!(!r2v_receipt.contains("video-1"));
    assert!(!r2v_receipt.contains("packed-source-tokens"));

    let receipt = bernini_r2v_reference_receipt(VideoLane::Mlx, 848, 480, &conditioning).unwrap();
    let overlay = format!("provider_video_mode:rv2v+{receipt}");
    let mut selector_contract = fixture_contract(20, 4, &[MemoryStrategy::BoundedDecode]);
    selector_contract.provider_id = "bernini".to_owned();
    let generator = fixture_generator(Some(selector_contract));
    for cache_state in [MemoryCacheState::Cold, MemoryCacheState::Warm] {
        let mut request = bernini_inputs(Some(&overlay));
        request.mode = "reference_video_to_video";
        request.reference_count = 3;
        request.reference_shape = "video+multi_image";
        request.runtime.as_mut().unwrap().cache_state = cache_state;
        let outcome = admit_video_generation_with_curves(&generator, request, None);
        assert!(outcome.refusal.is_none(), "{outcome:?}");
        let context = outcome.context.expect("exact RV2V selector context");
        assert_eq!(context.mode.as_key(), "reference_video_to_video");
        assert_eq!(context.geometry.reference_count, 3);
        assert_eq!(context.overlay.as_deref(), Some(overlay.as_str()));
        assert_eq!(context.selection.strategy, MemoryStrategy::Resident);
        assert_eq!(context.cache_state, cache_state);
    }
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
    // In the walk-down window: above the ADMITTED 34 GiB staged floor, below the admitted 38 GiB
    // resident floor (see `a_constrained_host_walks_down_the_ladder_instead_of_refusing`).
    let outcome = admit_video_generation_with_curves(
        &generator,
        inputs(241, budget(mlx_widened_gb(34, 0.5)), 18 * GIB),
        None,
    );
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
    // The conservative resident profile is 20 GiB of weights + 35 GiB of profiled activation =
    // 55 GiB, which cannot fit. The exact bounded carrier retains the generic 20 + 18 = 38 GiB
    // lower bound and fits a host 0.5 GiB above that floor's admitted ceiling. Without consuming
    // the selected profile, Resident would win first.
    //
    // The profiled peak's activation term is `55 - 20`, NOT the 18 GiB generic headroom: sc-22508
    // derives the term from the peak each candidate actually built.
    let host_gb = mlx_widened_gb(38, 0.5);
    assert!(
        host_gb < mlx_widened_floor_gb(55, 35, 0.0),
        "the window must exist"
    );
    let outcome = admit_video_generation_with_curves_and_profiles(
        &generator,
        inputs(241, budget(host_gb), 18 * GIB),
        None,
        tiered_decode_profile,
        false,
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
        false,
    );
    let refusal = outcome
        .refusal
        .expect("a provider profile that does not fit is a real refusal");
    // The reported need is the 55 GiB profiled peak plus the allowance on ITS activation term —
    // `55 - 20` GiB of profiled activation over 20 GiB of counted weights, not the 18 GiB generic
    // headroom — exactly as the selector admits it. A production term derived from the generic
    // headroom instead would under-report this by 17% of 17 GiB and reds here.
    let widened_profile = format!("needs about {:.1} GB", mlx_widened_floor_gb(55, 35, 0.0));
    assert!(refusal.contains(&widened_profile), "{refusal}");
    assert!(outcome.memory.is_none());
    assert!(outcome.context.is_none());
}

#[test]
fn provider_profile_composition_and_warm_residency_are_each_accounted_once() {
    let mut contract = fixture_contract(20, 4, &[]);
    contract.asset_facts.transformer_bytes = 12 * GIB;
    contract.asset_facts.decoder_bytes = 4 * GIB;
    let generator = fixture_generator(Some(contract));
    // The host: 20 GiB already committed, plus the ADMITTED 6 GiB incremental floor. After the
    // 20 GiB warm credit the counted-weights term is zero, so the whole 6 GiB residual is the
    // floor's activation slice and carries the full allocator-envelope allowance — computed from
    // the same helper production charges it with, so an allowance re-derivation moves this window
    // instead of silently flipping the verdict (epic 22505 feature-end fix round).
    let total_gb =
        20.0 + crate::memory_strategy::peak_bytes_to_gb(
            crate::memory_strategy::floor_admitted_peak_bytes(
                gen_core::MemoryBackend::Mlx,
                6 * GIB,
                Some(6 * GIB),
            ),
        ) + 0.5;
    let total_bytes = (total_gb * GIB as f64) as u64;
    let mut request = inputs(241, budget(total_gb), 0);
    request.runtime = Some(VideoRuntimeMemoryState {
        budget: MemoryBudget {
            total_bytes,
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
        false,
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
    // A 40 GiB host with this fixture's 18 GiB activation headroom: the honest staged fallback is
    // 40 weights + 18 headroom and the resident one 60 + 18, so no rung fits and the only way to
    // "admit" would be to mint bounded-decode knobs from the relabelled staged curve.
    let mut request = inputs(121, budget(40.0), FIXTURE_HEADROOM_GIB * GIB);
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

    let mut request = inputs(121, budget(128.0), 18 * GIB);
    request.fps = 30;
    request.expected_closure_digest = FITTED_CURVE_CLOSURE;
    request.overlay = Some("provider_video_mode:no_audio");
    assert_open("provider no-audio mode", request);

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
fn request_scoped_selection_refuses_crossed_identity_stale_and_out_of_envelope_evidence() {
    let generator = fixture_generator(Some(fixture_contract(
        20,
        4,
        &[MemoryStrategy::StagedResidency],
    )));
    let curves = fixture_curve_bundle();
    let admitted = |request| {
        admit_video_generation_with_curves_and_profiles(
            &generator,
            request,
            Some(&curves),
            no_video_decode_profile,
            true,
        )
    };
    let mut exact = inputs(121, budget(128.0), 18 * GIB);
    exact.expected_closure_digest = FITTED_CURVE_CLOSURE;
    exact.fps = 30;
    assert!(
        admitted(exact).context.is_some(),
        "exact sealed cell is selectable"
    );

    let mut crossed_family = inputs(121, budget(128.0), 18 * GIB);
    crossed_family.expected_closure_digest = FITTED_CURVE_CLOSURE;
    crossed_family.fps = 30;
    crossed_family.model_family = "ltx-alias";

    let mut crossed_route = inputs(121, budget(128.0), 18 * GIB);
    crossed_route.expected_closure_digest = FITTED_CURVE_CLOSURE;
    crossed_route.fps = 30;
    crossed_route.route = "ltx_2_3_alias";

    let mut crossed_mode = inputs(121, budget(128.0), 18 * GIB);
    crossed_mode.expected_closure_digest = FITTED_CURVE_CLOSURE;
    crossed_mode.fps = 30;
    crossed_mode.mode = "image_to_video";
    crossed_mode.reference_count = 1;
    crossed_mode.reference_shape = "image";

    let mut stale = inputs(121, budget(128.0), 18 * GIB);
    stale.expected_closure_digest = "stale-closure";
    stale.fps = 30;

    let mut outside_fps = inputs(121, budget(128.0), 18 * GIB);
    outside_fps.expected_closure_digest = FITTED_CURVE_CLOSURE;
    outside_fps.fps = 24;

    for (label, request) in [
        ("crossed family", crossed_family),
        ("crossed route", crossed_route),
        ("crossed mode/reference", crossed_mode),
        ("stale closure", stale),
        ("out-of-envelope FPS", outside_fps),
    ] {
        assert_eq!(
            admitted(request),
            VideoAdmissionOutcome::default(),
            "{label} must not borrow a different request-scoped curve"
        );
    }
}

#[test]
fn evidence_preflight_needs_no_runtime_and_skips_unsupported_requests() {
    let generator = fixture_generator(Some(fixture_contract(
        20,
        4,
        &[MemoryStrategy::StagedResidency],
    )));
    let mut exact = inputs(121, None, 18 * GIB);
    exact.expected_closure_digest = FITTED_CURVE_CLOSURE;
    exact.fps = 30;
    assert!(packaged_video_evidence_covers_request(&generator, &exact));

    exact.overlay = Some("provider_video_mode:no_audio");
    assert!(
        !packaged_video_evidence_covers_request(&generator, &exact),
        "no-audio must be rejected before a live runtime probe until separately evidenced"
    );

    exact.overlay = None;
    exact.mode = "image_to_video";
    exact.reference_count = 1;
    exact.reference_shape = "image";
    assert!(
        !packaged_video_evidence_covers_request(&generator, &exact),
        "unsupported requests must be rejected before a live runtime probe"
    );
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
            reference_shape: "none",
            fps: 30,
            overlay: None,
            lane: VideoLane::Mlx,
            tier: tier(),
            transformer_variant: Some(Ltx25TransformerVariant::Distilled),
            decoder: Some(Ltx25Decoder::Conv),
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: crate::mlx_fit_gate::UNCALIBRATED_CLOSURE,
        },
        &contract,
        // In the staged-not-resident window: above the widened 34 GiB staged floor, below the
        // widened 38 GiB resident floor, so both graded geometries land on the same staged rung.
        budget(mlx_widened_gb(34, 0.5)),
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
    // 20 GiB weights + 18 GiB headroom = 38 GiB raw resident floor. Every implemented rung is
    // `Missing` beyond resident, so the ladder has nowhere to go; at 39 GiB the raw floor FITS
    // while its admitted ceiling — 38 plus 17% of the 18 GiB activation term, `mlx_widened_gb(38,
    // 0.0)` — does not, which is exactly the band.
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
    // ~38 GB weights+headroom resident floor (20 GiB weights + 18 GiB headroom). sc-22508: the
    // admitted ceiling adds the headroom TERM once, not a percentage of the whole 38 GiB.
    let floor_bytes = 38 * GIB;
    let admitted_floor_bytes = crate::memory_strategy::floor_admitted_peak_bytes(
        gen_core::MemoryBackend::Mlx,
        floor_bytes,
        Some(FIXTURE_HEADROOM_GIB * GIB),
    );
    let floor_gb = crate::memory_strategy::peak_bytes_to_gb(floor_bytes);
    let widened_floor_gb = crate::memory_strategy::peak_bytes_to_gb(admitted_floor_bytes);
    assert_eq!(
        admitted_floor_bytes,
        floor_bytes
            + ((FIXTURE_HEADROOM_GIB * GIB) as f64
                * crate::ladder_margin_policy::FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE)
                .ceil() as u64,
        "the floor's allowance is charged on its activation term, not on its counted weights"
    );
    // A host that comfortably holds the floor but not the fitted decode peak.
    let host_gb = 100.0;
    assert!(floor_gb < host_gb, "{floor_gb} vs {host_gb}");

    // The fitted shape: the rejected peak is the DECODE peak, far above the floor. Derived from
    // the admitted floor plus explicit slack, so the "far above" claim stays true — and stays a
    // claim about the ALLOWANCE boundary — if the allowance moves.
    const DECODE_ABOVE_FLOOR_SLACK_GB: f64 = 78.0;
    let fitted_decode_reject_gb = widened_floor_gb + DECODE_ABOVE_FLOOR_SLACK_GB;
    assert!(
        fitted_decode_reject_gb > host_gb,
        "the fixture must be a genuine rejection on this host: {fitted_decode_reject_gb} vs \
         {host_gb}"
    );
    assert!(
        !refusal_is_a_margin_artifact(
            fitted_decode_reject_gb,
            floor_bytes,
            admitted_floor_bytes,
            Some(host_gb),
        ),
        "a rejection whose peak exceeds the weights floor by more than the allowance is REAL and \
         must survive — suppressing it runs the job into an OOM"
    );

    // The fallback shape, on the identical floor/host/allowance: the rejected peak IS the admitted
    // floor, so the suppression still applies. Without this the assertion above could be satisfied
    // by a guard that never suppresses anything.
    assert!(
        refusal_is_a_margin_artifact(
            widened_floor_gb,
            floor_bytes,
            admitted_floor_bytes,
            Some(host_gb)
        ),
        "a rejection at exactly the admitted floor is the margin artifact this guard exists for"
    );
    // And the boundary is where the doc says it is: one ULP past the admitted floor survives.
    assert!(
        !refusal_is_a_margin_artifact(
            widened_floor_gb + f64::EPSILON * widened_floor_gb,
            floor_bytes,
            admitted_floor_bytes,
            Some(host_gb),
        ),
        "the scope check is `<= admitted floor`, so anything above it is out of scope"
    );
}

/// The second conjunct — the non-regression condition proper — is independent of the first.
#[test]
fn a_floor_that_does_not_fit_is_never_suppressed_and_no_budget_never_suppresses() {
    let floor_bytes = 38 * GIB;
    let admitted_floor_bytes = crate::memory_strategy::floor_admitted_peak_bytes(
        gen_core::MemoryBackend::Mlx,
        floor_bytes,
        Some(FIXTURE_HEADROOM_GIB * GIB),
    );
    let floor_gb = crate::memory_strategy::peak_bytes_to_gb(floor_bytes);
    let widened_floor_gb = crate::memory_strategy::peak_bytes_to_gb(admitted_floor_bytes);

    // In scope (the peak IS the floor) but the floor itself does not fit: a real refusal the
    // pre-existing load gate would also have made.
    assert!(!refusal_is_a_margin_artifact(
        widened_floor_gb,
        floor_bytes,
        admitted_floor_bytes,
        Some(floor_gb - 1.0),
    ));
    // Exactly at the floor still fits — the comparison is `<=`, matching `mlx_fit_gate`'s.
    assert!(refusal_is_a_margin_artifact(
        widened_floor_gb,
        floor_bytes,
        admitted_floor_bytes,
        Some(floor_gb),
    ));
    // No budget signal: never suppress. `select_strategy` cannot even produce a `Reject` without
    // one, so this is the never-block-without-evidence posture applied in the safe direction.
    assert!(!refusal_is_a_margin_artifact(
        widened_floor_gb,
        floor_bytes,
        admitted_floor_bytes,
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
            reference_shape: "none",
            fps: 30,
            overlay: None,
            lane: VideoLane::Candle,
            tier: tier(),
            transformer_variant: Some(Ltx25TransformerVariant::Distilled),
            decoder: Some(Ltx25Decoder::Conv),
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
            reference_shape: "none",
            fps: 30,
            overlay: None,
            lane: VideoLane::Candle,
            tier: tier(),
            transformer_variant: Some(Ltx25TransformerVariant::Distilled),
            decoder: Some(Ltx25Decoder::Conv),
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
                reference_shape: "none",
                fps: 30,
                overlay: None,
                lane: VideoLane::Candle,
                tier: tier(),
                transformer_variant: Some(Ltx25TransformerVariant::Distilled),
                decoder: Some(Ltx25Decoder::Conv),
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
            reference_shape: "none",
            fps: 30,
            overlay: None,
            lane: VideoLane::Candle,
            tier: tier(),
            transformer_variant: Some(Ltx25TransformerVariant::Distilled),
            decoder: Some(Ltx25Decoder::Conv),
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
                reference_shape: "none",
                fps: 30,
                overlay: None,
                lane,
                tier: tier(),
                transformer_variant: Some(Ltx25TransformerVariant::Distilled),
                decoder: Some(Ltx25Decoder::Conv),
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
    // 20 GiB weights + 18 GiB headroom = a 38 GiB resident floor; rung 4 sheds the whole 16 GiB
    // transformer, so its floor is 22 GiB. Both carry the same 18 GiB activation term, so the
    // admitted ceilings (`mlx_widened_gb`) stay 16 GiB apart. A host 0.5 GiB above rung 4's
    // admitted floor can hold ONLY rung 4 — which is exactly why offering it here would be the
    // harm.
    let host_gb = mlx_widened_gb(22, 0.5);
    assert!(host_gb < mlx_widened_gb(38, 0.0), "the window must exist");
    let (verdict, selections) = select_once(
        &eager,
        budget(host_gb),
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
    // assertion above is about the prerequisite and not about the host being too small.
    let deferred = with_load_shape(
        fixture_contract(20, 4, &[MemoryStrategy::BoundedTransformerResidency]),
        LoadShape::DeferredMaterialization,
    );
    let (reachable, _) = select_once(
        &deferred,
        budget(host_gb),
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
    // The window, derived end to end from the same helpers production uses: the host sits between
    // the f121 and f145 FITTED admitted ceilings (so the frame count is the only thing that
    // changes between the two verdicts) and below the admitted resident floor (so resident never
    // fits and cannot mask either verdict).
    let f121_ceiling_gb = fitted_staged_admitted_gb(
        &contract,
        &curves,
        geometry(121, VideoGeometryRole::Requested),
    );
    let f145_ceiling_gb = fitted_staged_admitted_gb(
        &contract,
        &curves,
        geometry(145, VideoGeometryRole::Requested),
    );
    let host_gb = (f121_ceiling_gb + f145_ceiling_gb) / 2.0;
    assert!(
        f121_ceiling_gb < host_gb && host_gb < f145_ceiling_gb,
        "the frame count must move the admitted ceiling across the host: f121 {f121_ceiling_gb}, \
         host {host_gb}, f145 {f145_ceiling_gb}"
    );
    assert!(
        host_gb < mlx_widened_gb(38, 0.0),
        "the resident floor must not fit, or the frame count is not what decides"
    );

    let at_121 = select_once_with_curves(
        &contract,
        &curves,
        budget(host_gb),
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
        "the f121 fitted staged peak still fits {host_gb} GiB: {at_121:?}"
    );

    let at_145 = select_once_with_curves(
        &contract,
        &curves,
        budget(host_gb),
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
        Some("ltx_2_3:ltx-video:ltx_2_3:ltx_2_3:mlx:q8:distilled:conv:text_to_video:refnone-0:fps30:none:staged_residency:eager_materialization:b1:abi3:single_pass:87a27d5dcab7:sc-18808-ltx-2-3-mlx-t2v-staged-capture-v1")
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
    // A 40 GiB-weights contract, NOT the 20 GiB one the neighbouring tests use, and the choice is
    // load-bearing: at 20 GiB the staged scalar floor is 16 + 18 = 34 GiB, whose admitted ceiling
    // sits below the host this test needs, so the floor alone would admit and the curve
    // coefficient could not be what flips the verdict. At 40 GiB the two scalar floors (resident
    // 40 + 18, staged 36 + 18) are out of reach, leaving the fitted curve as the only decider.
    // `staged_floor_ceiling_gb` below asserts that rather than assuming it.
    let contract = fixture_contract(40, 4, &[MemoryStrategy::StagedResidency]);
    let original = fixture_curve_bundle();
    let geometry = geometry(145, VideoGeometryRole::Requested);
    let mut mutated = original.clone();
    mutated.curves[0].phases.decode.per_mpx_frame_gb = 0.0;

    // The window, derived end to end from the policy rather than written down as a float: the host
    // sits between the two FITTED admitted ceilings (the mutated one admits, the shipped one
    // rejects) and below the lower of the two SCALAR floor ceilings (so neither floor can decide).
    // Every bound comes from the same helpers production uses, so an allowance change moves the
    // window with it instead of silently making one arm vacuous.
    let fitted_ceiling_gb = |bundle: &VideoMemoryCurveBundle| -> f64 {
        let selector = selector_with_curves(&contract, Some(bundle), budget(0.0));
        let engaged = contract.engaged_composition(MemoryStrategy::StagedResidency);
        let (peaks, basis, ..) = fitted_or_floor_phase_peaks(
            &selector,
            geometry,
            MemoryStrategy::StagedResidency,
            &engaged,
        );
        assert_eq!(
            basis,
            CandidateBasis::EstimateFittedCurve,
            "the window is only meaningful while the curve is what decides"
        );
        crate::memory_strategy::peak_bytes_to_gb(crate::memory_strategy::admitted_peak_bytes(
            crate::ladder_margin_policy::AdmissionSubject {
                backend: gen_core::MemoryBackend::Mlx,
                basis,
                closure_is_stale: false,
                unmodeled_activation_bytes: None,
            },
            peaks.peak_bytes(),
        ))
    };
    let shipped_ceiling_gb = fitted_ceiling_gb(&original);
    let mutated_ceiling_gb = fitted_ceiling_gb(&mutated);
    let staged_floor_ceiling_gb = mlx_widened_gb(36 + FIXTURE_HEADROOM_GIB, 0.0);
    let host_gb = (shipped_ceiling_gb + mutated_ceiling_gb) / 2.0;
    assert!(
        mutated_ceiling_gb < host_gb && host_gb < shipped_ceiling_gb,
        "the coefficient must move the admitted ceiling across the host: \
         mutated {mutated_ceiling_gb}, host {host_gb}, shipped {shipped_ceiling_gb}"
    );
    assert!(
        host_gb < staged_floor_ceiling_gb,
        "neither scalar floor may fit, or the curve is not what decides: host {host_gb}, \
         staged floor ceiling {staged_floor_ceiling_gb}"
    );

    let original_verdict = select_once_with_curves(&contract, &original, budget(host_gb), geometry);
    assert!(
        matches!(original_verdict, VideoRungSelection::Reject { .. }),
        "the generated decode cross coefficient must bind: {original_verdict:?}"
    );

    let mutated_verdict = select_once_with_curves(&contract, &mutated, budget(host_gb), geometry);
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
    let mut grown = original.clone();
    grown.curves[0].phases.decode.max_residual_gb += 12.0;
    // The window, derived end to end from the helpers: the host sits between the shipped and the
    // residual-grown FITTED admitted ceilings — so the residual alone flips the verdict — and
    // below the admitted resident floor, so resident never fits.
    let shipped_ceiling_gb = fitted_staged_admitted_gb(&contract, &original, request);
    let grown_ceiling_gb = fitted_staged_admitted_gb(&contract, &grown, request);
    let host_gb = (shipped_ceiling_gb + grown_ceiling_gb) / 2.0;
    assert!(
        shipped_ceiling_gb < host_gb && host_gb < grown_ceiling_gb,
        "the residual must move the admitted ceiling across the host: shipped \
         {shipped_ceiling_gb}, host {host_gb}, grown {grown_ceiling_gb}"
    );
    assert!(
        host_gb < mlx_widened_gb(38, 0.0),
        "the resident floor must not fit, or the residual is not what decides"
    );
    let original_verdict = select_once_with_curves(&contract, &original, budget(host_gb), request);
    assert!(
        matches!(original_verdict, VideoRungSelection::Selected { .. }),
        "the shipped residual-bounded curve must fit the bracket: {original_verdict:?}"
    );

    let mutated_verdict = select_once_with_curves(&contract, &grown, budget(host_gb), request);
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
    // 0.7 GiB under the admitted 38 GiB resident floor (41.06): admits the q8 fitted f121 peak
    // (~39.8 at the recapture ceiling) and the q4/bf16 staged floor (34 + 17% of its 18 GiB
    // activation term = 37.06) while resident stays refused, so every tier lands on the staged
    // rung its basis assertion above describes.
    let host_gb = mlx_widened_gb(38, -0.7);
    for quant in [Some(Quant::Q8), Some(Quant::Q4), None] {
        for cache_state in [MemoryCacheState::Cold, MemoryCacheState::Warm] {
            let mut request = inputs(121, budget(host_gb), 18 * GIB);
            request.fps = 30;
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
            // Same staged-rung window as the historical-q8 test: admits the q8 fitted f121 peak
            // and the q4/bf16 staged floor at its admitted ceiling, refuses resident.
            let mut request = inputs(121, budget(mlx_widened_gb(38, -1.0)), 18 * GIB);
            request.fps = 30;
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
    // In the staged-floor window — above the ADMITTED 34 GiB staged floor, below the admitted
    // 38 GiB resident floor (both from `mlx_widened_gb`, which adds 17% of the shared 18 GiB
    // activation term) — so the floor decision being preserved is a real staged selection.
    let host_gb = mlx_widened_gb(34, 0.5);
    let expected = {
        let mut selector = selector_with_curves(contract, None, budget(host_gb));
        selector.select(geometry)
    };
    let actual = select_once_with_curves(contract, curves, budget(host_gb), geometry);
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
    let mut selector =
        selector_with_curves(&contract, Some(&curves), budget(mlx_widened_gb(34, 0.5)));
    selector.identity.calibration_abi += 1;
    let live_abi_mismatch = selector.select(inside);
    let mut no_curves = selector_with_curves(&contract, None, budget(mlx_widened_gb(34, 0.5)));
    assert_eq!(
        live_abi_mismatch,
        no_curves.select(inside),
        "a pinned gen-core ABI bump must stale the old curve even if the fixture contract's own \
         calibration identity still says 3"
    );

    let mut stale_contract = fixture_contract(20, 4, &[MemoryStrategy::StagedResidency]);
    stale_contract.calibration.as_mut().unwrap().abi += 1;
    let selector = selector_with_curves(
        &stale_contract,
        Some(&curves),
        budget(mlx_widened_gb(34, 0.5)),
    );
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
    curves.curves[0].transformer_variant = Ltx25TransformerVariant::Dev;
    assert_curve_mismatch_falls_back("crossed transformer pipeline", &contract, &curves, inside);

    let mut curves = fixture_curve_bundle();
    curves.curves[0].decoder = Ltx25Decoder::DiffVae;
    assert_curve_mismatch_falls_back("crossed decoder pipeline", &contract, &curves, inside);

    let curves = fixture_curve_bundle();
    let mut selector =
        selector_with_curves(&contract, Some(&curves), budget(mlx_widened_gb(34, 0.5)));
    selector.identity.transformer_variant = Some(Ltx25TransformerVariant::Dev);
    let crossed_request = selector.select(inside);
    let mut no_curves = selector_with_curves(&contract, None, budget(mlx_widened_gb(34, 0.5)));
    assert_eq!(
        crossed_request,
        no_curves.select(inside),
        "runtime request identity must not borrow a fitted curve from another transformer pipeline",
    );

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

    // In the staged-floor window: above the ADMITTED 34 GiB staged floor, below the admitted
    // 38 GiB resident floor (both from `mlx_widened_gb`) — and far below the fitted f297 cap
    // peak's admitted ceiling, which is what makes the cap row the binding refusal.
    let host_gb = mlx_widened_gb(34, 0.5);
    let requested = VideoAdmissionGeometry {
        decode_pass: VideoDecodePass::Tiled,
        ..geometry(305, VideoGeometryRole::Requested)
    };
    assert!(
        matches!(
            select_once_with_curves(&contract, &curves, budget(host_gb), requested),
            VideoRungSelection::Selected {
                rung: StrategyRung::StagedResidency,
                ..
            }
        ),
        "the tiled request itself falls back to the widened 34 GiB staged floor"
    );

    let mut selector = selector_with_curves(&contract, Some(&curves), budget(host_gb));
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
    // 40/20 rather than the historical 90/45 (epic 22505 feature-end fix round): the floor
    // allowance re-derivation raised every floor's admitted ceiling by ~3.1x its activation term,
    // and at 90/45 the REQUESTED row's staged floor (63 GiB raw) would out-demand the fitted cap
    // row and flip which row binds. At 32/16 the staged floor is 34 GiB raw, its admitted ceiling
    // sits back under the cap's, and the cap row is again the binding one — the premise this test
    // exists to exercise. The window assertions below check that ordering rather than assume it.
    let contract = fixture_contract(32, 16, &[MemoryStrategy::StagedResidency]);
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
    let (expected_peak, expected_basis) = {
        let selector = selector_with_curves(&contract, Some(&curves), budget(95.0));
        let engaged = contract.engaged_composition(MemoryStrategy::StagedResidency);
        let (peaks, basis, ..) =
            fitted_or_floor_phase_peaks(&selector, cap, MemoryStrategy::StagedResidency, &engaged);
        (peaks.peak_bytes(), basis)
    };

    // The basis this window is built on. Asserted, not assumed: the whole window below is computed
    // from the fitted-curve allowance, so a change that demoted this candidate to a floor would
    // otherwise leave the test passing against arithmetic that no longer describes production.
    assert_eq!(
        expected_basis,
        CandidateBasis::EstimateFittedCurve,
        "the cap candidate must be the fitted curve for this window to mean anything"
    );

    // 0.5 GiB above the larger of the two staged candidates' admitted ceilings — the fitted cap
    // row and the REQUESTED row (whose out-of-hull geometry falls to the staged floor, admitted
    // behind the floor's activation-term allowance) — and below the admitted resident floor, so
    // both geometries select STAGED and the same-rung tie is what is under test. sc-22508: each
    // allowance is named by its candidate's own basis. This mirror declares the term production
    // declares for that basis — `None` for a fitted curve, and for a floor the activation slice
    // of the peak itself, exactly as `LadderVideoSelector::select` derives it.
    let admitted_gb = |geometry: VideoAdmissionGeometry| -> f64 {
        let selector = selector_with_curves(&contract, Some(&curves), budget(95.0));
        let engaged = contract.engaged_composition(MemoryStrategy::StagedResidency);
        let (peaks, basis, ..) = fitted_or_floor_phase_peaks(
            &selector,
            geometry,
            MemoryStrategy::StagedResidency,
            &engaged,
        );
        crate::memory_strategy::peak_bytes_to_gb(crate::memory_strategy::admitted_peak_bytes(
            crate::ladder_margin_policy::AdmissionSubject {
                backend: gen_core::MemoryBackend::Mlx,
                basis,
                closure_is_stale: false,
                unmodeled_activation_bytes: matches!(basis, CandidateBasis::EstimateFloor).then(
                    || {
                        peaks.peak_bytes().saturating_sub(
                            crate::mlx_fit_gate::estimate_floor_weights_bytes(&contract, &engaged),
                        )
                    },
                ),
            },
            peaks.peak_bytes(),
        ))
    };
    let requested_ceiling_gb = admitted_gb(geometry(305, VideoGeometryRole::Requested));
    let cap_ceiling_gb = admitted_gb(cap);
    assert!(
        requested_ceiling_gb < cap_ceiling_gb,
        "the cap row must be the binding (most demanding) staged candidate, or the same-rung tie \
         below has nothing to retain: requested {requested_ceiling_gb}, cap {cap_ceiling_gb}"
    );
    let host_gb = cap_ceiling_gb.max(requested_ceiling_gb) + 0.5;
    // The resident floor is 32 GiB of weights plus this request's 18 GiB headroom = 50 GiB raw,
    // admitted behind the allowance on its 18 GiB activation term (`mlx_widened_gb`) — the
    // allowance is a fraction of the activation term, not a second copy of it.
    assert!(
        host_gb < mlx_widened_gb(32 + FIXTURE_HEADROOM_GIB, 0.0),
        "the staged-not-resident window must exist: {host_gb}"
    );

    let mut request = inputs(305, budget(host_gb), FIXTURE_HEADROOM_GIB * GIB);
    request.fps = 30;
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
        "ltx_2_3:ltx-video:ltx_2_3:ltx_2_3:mlx:q8:distilled:conv:text_to_video:refnone-0:fps30:none:staged_residency:eager_materialization:b1:abi3:single_pass:87a27d5dcab7:sc-18808-ltx-2-3-mlx-t2v-staged-capture-v1"
    );
    assert!(outcome.refusal.is_none());
}

// --------------------------------------------------------------------------------------------
// sc-22507 (epic 22505): anchor + analytic derivation. One measured anchor per
// (model, tier, lane) prices a never-measured (geometry, frames) cell; the selector admits from
// the derived estimate when it fits.
// --------------------------------------------------------------------------------------------

/// The calibration campaign the packaged LTX-2.5 anchors were extracted from. The anchor path
/// requires the contract to still name it, exactly as the fitted path requires its own fingerprint.
const LTX25_ANCHOR_FINGERPRINT: &str = "sc-18797-ltx-2-5-mlx-ladder-v1";

/// A conformant LTX-2.5 contract on the MLX lane whose identity matches the packaged anchors.
fn ltx25_fixture_contract(rungs: &[MemoryStrategy]) -> MemoryProviderContract {
    let mut contract = fixture_contract(20, 4, rungs);
    contract.provider_id = "ltx_2_5".to_owned();
    contract.calibration = Some(MemoryCalibrationIdentity {
        abi: gen_core::MEMORY_CALIBRATION_ABI,
        fingerprint: LTX25_ANCHOR_FINGERPRINT.to_owned(),
        load_shape: LoadShape::EagerMaterialization,
    });
    assert!(contract.conformance_errors().is_empty());
    contract
}

/// The pipeline cell the packaged `q8` anchor was measured on. The corpus measures no
/// `q8 distilled/*` cell at all, so `distilled` here is not an alternative — it is the
/// unmeasured-cell control below.
fn ltx25_identity(expected_closure_digest: &str) -> VideoRequestIdentity<'_> {
    VideoRequestIdentity {
        model_id: "ltx_2_5",
        model_family: "ltx-video",
        route: "ltx_2_5",
        mode: "text_to_video",
        reference_count: 0,
        reference_shape: "none",
        fps: 25,
        overlay: None,
        lane: VideoLane::Mlx,
        tier: tier(),
        transformer_variant: Some(Ltx25TransformerVariant::Dev),
        decoder: Some(Ltx25Decoder::DiffVae),
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        expected_closure_digest,
    }
}

/// The geometry deliberately absent from the retained corpus: 640x640 was measured only at
/// f145, and no record at any geometry was measured at 89 frames.
fn ltx25_unmeasured_geometry() -> VideoAdmissionGeometry {
    VideoAdmissionGeometry {
        width: 640,
        height: 640,
        frames: 89,
        decode_pass_frames: 89,
        batch: 1,
        decode_pass: VideoDecodePass::SinglePass,
        role: VideoGeometryRole::Requested,
    }
}

/// `None` when the packaged store carries no q8 dev/diffvae MLX anchor for LTX-2.5 (sc-22512,
/// E8/E9). Currency is deliberately not part of this helper: selector tests restamp the packaged
/// store to the live content-derived closure and must retain their anchor-path coverage when a pin
/// changes. Only the production-funnel test asks whether the packaged anchor itself is current.
fn ltx25_expected_derived_peaks() -> Option<sceneworks_core::memory_anchor::AnchorDerivedPhases> {
    let anchor = sceneworks_core::memory_anchor::packaged_memory_anchors()
        .expect("packaged anchors load")
        .anchor_for(
            "ltx_2_5",
            sceneworks_core::memory_anchor::AnchorBackend::Mlx,
            "q8",
            Ltx25TransformerVariant::Dev,
            Ltx25Decoder::DiffVae,
        )?;
    Some(
        anchor
            .derive_video_phase_peaks(sceneworks_core::memory_anchor::AnchorDeriveRequest {
                width: 640,
                height: 640,
                frames: 89,
                decode_tiled: false,
                transformer_windowed: false,
                deferred_materialization: false,
            })
            .expect("the unmeasured geometry is derivable"),
    )
}

#[test]
fn an_unmeasured_ltx25_geometry_is_admitted_from_the_anchor_derived_estimate() {
    let contract = ltx25_fixture_contract(&[]);
    let Some(expected) = ltx25_expected_derived_peaks() else {
        return;
    };
    let anchors = current_loader_anchor_store();

    let mut selector = LadderVideoSelector::new(
        ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE),
        &contract,
        budget(128.0),
        18 * GIB,
        0,
    )
    .with_anchor_store(Some(&anchors));
    let verdict = selector.select(ltx25_unmeasured_geometry());
    let VideoRungSelection::Selected { rung, .. } = verdict else {
        panic!("expected an anchor-derived selection, got {verdict:?}");
    };
    assert_eq!(rung, StrategyRung::Resident);
    assert_eq!(selector.selections.len(), 1);
    // The selected candidate carries the anchor derivation, not the weights+headroom floor: the
    // raw predicted peak is exactly the core derivation's max phase, and the evidence revision
    // names the anchor it came from.
    assert_eq!(
        selector.selections[0].predicted_peak_bytes,
        expected.peak_bytes()
    );
    assert_eq!(
        selector.selections[0].evidence_revision,
        "ltx_2_5:mlx:q8:dev:diffvae:sc-18797-ltx-2-5-mlx-ladder-v1:imc-7f8186376a9a3143ebee"
    );

    // Differential control: the SAME request without an anchor store falls back to the
    // phase-blind floor — a different peak and the floor's evidence label — proving the anchor
    // path, not the floor, carried the admission above.
    let mut floored = LadderVideoSelector::new(
        ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE),
        &contract,
        budget(128.0),
        18 * GIB,
        0,
    )
    .with_anchor_store(None);
    let floor_verdict = floored.select(ltx25_unmeasured_geometry());
    assert!(matches!(floor_verdict, VideoRungSelection::Selected { .. }));
    assert_ne!(
        floored.selections[0].predicted_peak_bytes,
        expected.peak_bytes()
    );
    assert_eq!(
        floored.selections[0].evidence_revision,
        "video-estimate-floor-v1"
    );
}

#[test]
fn the_production_funnel_admits_an_unmeasured_ltx25_geometry_from_the_anchor() {
    // The production entry point requires packaged request evidence before probing. LTX-2.5 has
    // no fitted curve, so this passes only because the packaged anchor store covers the request;
    // the admitted context must then carry the anchor-derived peak end-to-end.
    let generator = fixture_generator(Some(ltx25_fixture_contract(&[])));
    let Some(expected) = ltx25_expected_derived_peaks() else {
        return;
    };
    // The funnel reads the PACKAGED store, so its question needs the packaged anchor current.
    if !packaged_ltx25_anchor_is_current() {
        return;
    }
    let mut request = inputs(89, budget(128.0), 18 * GIB);
    request.model_id = "ltx_2_5";
    request.route = "ltx_2_5";
    request.transformer_variant = Some(Ltx25TransformerVariant::Dev);
    request.decoder = Some(Ltx25Decoder::DiffVae);
    request.width = 640;
    request.height = 640;
    request.fps = 25;
    let outcome = admit_video_generation(&generator, request);
    assert!(outcome.refusal.is_none(), "{:?}", outcome.refusal);
    let context = outcome
        .context
        .expect("the anchor-covered request must reach the ladder and select");
    assert_eq!(context.selection.strategy, MemoryStrategy::Resident);
    assert_eq!(context.predicted_peak_bytes, expected.peak_bytes());
    assert_eq!(
        context.evidence_revision,
        "ltx_2_5:mlx:q8:dev:diffvae:sc-18797-ltx-2-5-mlx-ladder-v1:imc-7f8186376a9a3143ebee"
    );

    // Control: the identical request under a model id with no anchor (and no fitted curve) is
    // not covered by packaged evidence, so the production gate stays failed open.
    let mut uncovered = inputs(89, budget(128.0), 18 * GIB);
    uncovered.model_id = "ltx_2_5_nonexistent";
    uncovered.route = "ltx_2_5";
    uncovered.transformer_variant = Some(Ltx25TransformerVariant::Dev);
    uncovered.decoder = Some(Ltx25Decoder::DiffVae);
    uncovered.width = 640;
    uncovered.height = 640;
    uncovered.fps = 25;
    assert_eq!(
        admit_video_generation(&generator, uncovered),
        VideoAdmissionOutcome::default(),
        "no packaged evidence must keep the historical fail-open behavior"
    );
}

#[test]
fn the_anchor_derived_estimate_admits_when_it_fits_and_refuses_when_it_does_not() {
    let contract = ltx25_fixture_contract(&[]);
    let Some(expected) = ltx25_expected_derived_peaks() else {
        return;
    };
    // sc-22508: an anchor-derived peak is FULLY PRICED by the derivation (coefficient uncertainty
    // inside the coefficients, the allocator envelope in `ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`), so
    // the selector adds nothing and the admitted ceiling IS the derived peak.
    let widened_gb = crate::memory_strategy::peak_bytes_to_gb(expected.peak_bytes());
    let anchors = current_loader_anchor_store();

    let mut fits = LadderVideoSelector::new(
        ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE),
        &contract,
        budget(widened_gb + 0.5),
        18 * GIB,
        0,
    )
    .with_anchor_store(Some(&anchors));
    assert!(matches!(
        fits.select(ltx25_unmeasured_geometry()),
        VideoRungSelection::Selected { .. }
    ));

    let mut refused = LadderVideoSelector::new(
        ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE),
        &contract,
        budget(widened_gb - 0.5),
        18 * GIB,
        0,
    )
    .with_anchor_store(Some(&anchors));
    assert!(matches!(
        refused.select(ltx25_unmeasured_geometry()),
        VideoRungSelection::Reject { .. }
    ));
}

/// The retained corpus measures no `q8 distilled/conv` cell. Since the epic 22505 feature-end fix
/// round (E2) that request no longer falls to the phase-blind floor: it derives from the
/// `q8 dev/diffvae` SIBLING anchor plus the BOUND component deltas — the distillation LoRA's and
/// the conv decoder's shipped file sizes — and the evidence revision names the sibling honestly.
/// A store with the deltas stripped still falls to the floor: no bound size, no derivation.
#[test]
fn an_unmeasured_pipeline_cell_derives_from_the_sibling_anchor_plus_the_bound_deltas() {
    let contract = ltx25_fixture_contract(&[]);
    let cell_identity = || {
        let mut identity = ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE);
        identity.transformer_variant = Some(Ltx25TransformerVariant::Distilled);
        identity.decoder = Some(Ltx25Decoder::Conv);
        identity
    };
    let store = sceneworks_core::memory_anchor::packaged_memory_anchors()
        .expect("the packaged anchor store loads");
    let closures = sceneworks_core::memory_anchor::packaged_anchor_loader_closures()
        .expect("the packaged loader closures load");
    let geometry = ltx25_unmeasured_geometry();
    let expected = store.derive_video_phase_peaks_for_cell(
        "ltx_2_5",
        sceneworks_core::memory_anchor::AnchorBackend::Mlx,
        "q8",
        Ltx25TransformerVariant::Distilled,
        Ltx25Decoder::Conv,
        sceneworks_core::memory_anchor::AnchorDeriveRequest {
            width: geometry.width,
            height: geometry.height,
            frames: geometry.estimate_frames(),
            decode_tiled: false,
            transformer_windowed: false,
            deferred_materialization: false,
        },
    );
    let expected = expected.expect("the sibling+delta fall-through prices the unmeasured cell");
    assert!(
        expected.delta_bytes > 0,
        "the derivation must cross an axis with a bound delta, or this test asks nothing"
    );

    let mut selector =
        LadderVideoSelector::new(cell_identity(), &contract, budget(128.0), 18 * GIB, 0);
    assert!(matches!(
        selector.select(geometry),
        VideoRungSelection::Selected { .. }
    ));
    if expected.anchor.is_current(closures) {
        assert_eq!(
            selector.selections[0].evidence_revision, expected.anchor.id,
            "the unmeasured cell must be priced from the sibling anchor, named honestly"
        );
        assert_eq!(
            selector.selections[0].predicted_peak_bytes,
            expected.phases.peak_bytes(),
            "the selected peak is the sibling+delta derivation, byte for byte"
        );
    } else {
        // Currency is allowed to be false BY DESIGN (sc-22511): a stale sibling demotes the cell
        // to the floor rather than pricing it, and that is the honest verdict at such a pin.
        assert_eq!(
            selector.selections[0].evidence_revision,
            "video-estimate-floor-v1"
        );
    }

    // The floor control: strip the component deltas and the fall-through refuses, so the request
    // keeps the phase-blind floor rather than a guessed size.
    let mut deltaless = store.clone();
    deltaless.component_deltas.clear();
    let mut selector =
        LadderVideoSelector::new(cell_identity(), &contract, budget(128.0), 18 * GIB, 0);
    selector.anchors = Some(&deltaless);
    assert!(matches!(
        selector.select(geometry),
        VideoRungSelection::Selected { .. }
    ));
    assert_eq!(
        selector.selections[0].evidence_revision, "video-estimate-floor-v1",
        "an axis with no bound delta must fall to the floor"
    );
    if let Some(anchor_derived) = ltx25_expected_derived_peaks() {
        assert_ne!(
            selector.selections[0].predicted_peak_bytes,
            anchor_derived.peak_bytes()
        );
    }
}

/// sc-22512 (epic 22505, AC1): a model the packaged anchor store has NO row for at all is not a
/// failure anywhere in the pipeline.
///
/// Both halves are proved here. (a) The coverage machinery classifies the zero-anchor model
/// GRACEFULLY: the packaged store loads and every lookup axis misses by returning `None`, never a
/// panic or an `Err`. (b) The selector still ADMITS the request — `Selected`, carrying the
/// conservative analytic estimate (`video-estimate-floor-v1`) at a peak that is provably NOT an
/// anchor-derived one. Absence never blocks; a measurement only ever sharpens the estimate.
///
/// The zero-anchor MODEL is the subject, not a suppressed store: `with_anchor_store(None)` only
/// proves the store-absent path, so it is asserted alongside as a control rather than instead of.
/// The final leg proves the admission is a real budget-sensitive estimate and not a rubber stamp.
#[test]
fn a_model_with_zero_anchors_is_classified_gracefully_and_admitted_from_the_analytic_estimate() {
    use sceneworks_core::memory_anchor::AnchorBackend;

    /// A provider/model id the packaged store carries no row for, by construction.
    const ZERO_ANCHOR_MODEL: &str = "sc22512_model_with_no_anchors";

    // (a) Coverage: the store answers about an entirely unknown model without failing.
    let store = sceneworks_core::memory_anchor::packaged_memory_anchors()
        .expect("the packaged anchor store loads");
    assert!(
        !store
            .anchors
            .iter()
            .any(|anchor| anchor.model_id == ZERO_ANCHOR_MODEL),
        "the fixture must genuinely have zero anchors, else this test proves nothing"
    );
    for backend in [AnchorBackend::Mlx, AnchorBackend::Candle] {
        for tier_key in ["bf16", "q8", "q4"] {
            for variant in [
                Ltx25TransformerVariant::Dev,
                Ltx25TransformerVariant::Distilled,
            ] {
                for decoder in [Ltx25Decoder::Conv, Ltx25Decoder::DiffVae] {
                    assert!(
                        store
                            .anchor_for(ZERO_ANCHOR_MODEL, backend, tier_key, variant, decoder)
                            .is_none(),
                        "a zero-anchor model must MISS gracefully on \
                         ({backend:?}, {tier_key}, {variant:?}, {decoder:?})"
                    );
                }
            }
        }
    }

    // (b) Admission: the same model is selected from the conservative analytic estimate.
    let mut contract = ltx25_fixture_contract(&[]);
    contract.provider_id = ZERO_ANCHOR_MODEL.to_owned();
    let zero_anchor_identity = || {
        let mut identity = ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE);
        // Model, route and provider move together: the request is coherently ABOUT a model the
        // store has never heard of, not an LTX-2.5 request wearing a foreign model id (which the
        // shared selector would exclude on the route/provider handshake, masking the question).
        identity.model_id = ZERO_ANCHOR_MODEL;
        identity.route = ZERO_ANCHOR_MODEL;
        identity
    };

    let mut selector = LadderVideoSelector::new(
        zero_anchor_identity(),
        &contract,
        budget(128.0),
        18 * GIB,
        0,
    );
    let verdict = selector.select(ltx25_unmeasured_geometry());
    assert!(
        matches!(verdict, VideoRungSelection::Selected { .. }),
        "a model with zero anchors must be ADMITTED under the analytic estimate, got {verdict:?}"
    );
    assert_eq!(
        selector.selections[0].evidence_revision, "video-estimate-floor-v1",
        "the admission must come from the conservative analytic estimate"
    );
    if let Some(anchor_derived) = ltx25_expected_derived_peaks() {
        assert_ne!(
            selector.selections[0].predicted_peak_bytes,
            anchor_derived.peak_bytes(),
            "an anchor-derived peak here would mean the zero-anchor model borrowed another \
             model's measurements"
        );
    }

    // Control: suppressing the store entirely reaches the SAME estimate, so the admission above is
    // the zero-anchor-model path and not an artefact of which store was consulted.
    let mut store_absent = LadderVideoSelector::new(
        zero_anchor_identity(),
        &contract,
        budget(128.0),
        18 * GIB,
        0,
    )
    .with_anchor_store(None);
    assert!(matches!(
        store_absent.select(ltx25_unmeasured_geometry()),
        VideoRungSelection::Selected { .. }
    ));
    assert_eq!(
        store_absent.selections[0].predicted_peak_bytes,
        selector.selections[0].predicted_peak_bytes
    );

    // The estimate is real: at its own predicted peak the request admits, and half a GiB under it
    // the very same request refuses. Read off the selection the selector actually made rather than
    // re-widened here — sc-22508 moved the margin into the derivation, so charging it again would
    // double-count. Same derivation as the sibling anchor test above, never a magic float.
    let widened_gb = crate::memory_strategy::peak_bytes_to_gb(
        crate::memory_strategy::floor_admitted_peak_bytes(
            gen_core::MemoryBackend::Mlx,
            selector.selections[0].predicted_peak_bytes,
            Some(FIXTURE_HEADROOM_GIB * GIB),
        ),
    );
    let mut fits = LadderVideoSelector::new(
        zero_anchor_identity(),
        &contract,
        budget(widened_gb + 0.5),
        18 * GIB,
        0,
    );
    assert!(matches!(
        fits.select(ltx25_unmeasured_geometry()),
        VideoRungSelection::Selected { .. }
    ));
    let mut refused = LadderVideoSelector::new(
        zero_anchor_identity(),
        &contract,
        budget(widened_gb - 0.5),
        18 * GIB,
        0,
    );
    assert!(
        matches!(
            refused.select(ltx25_unmeasured_geometry()),
            VideoRungSelection::Reject { .. }
        ),
        "the analytic estimate must stay budget-sensitive, or the admission above is a rubber stamp"
    );

    // (c) The PRODUCTION entry path, not just the selector. Everything above drives
    // `LadderVideoSelector` directly, which is one layer below what a real job calls, so on its own
    // it could stay green while `admit_video_generation` refused the same zero-anchor request at an
    // earlier branch. One leg through the production entry closes that gap: the same coherent
    // zero-anchor request must produce NO refusal.
    let mut zero_anchor_contract = ltx25_fixture_contract(&[]);
    zero_anchor_contract.provider_id = ZERO_ANCHOR_MODEL.to_owned();
    let generator = fixture_generator(Some(zero_anchor_contract));
    let mut request = inputs(89, budget(128.0), 0);
    request.model_id = ZERO_ANCHOR_MODEL;
    request.model_family = ZERO_ANCHOR_MODEL;
    request.route = ZERO_ANCHOR_MODEL;
    request.width = 640;
    request.height = 640;
    let outcome = admit_video_generation(&generator, request);
    assert_eq!(
        outcome.refusal, None,
        "a model with zero anchors must not be refused at the production admission entry: \
         {outcome:?}"
    );
}

/// Whether the packaged `q8 dev/diffvae` LTX-2.5 MLX anchor is current against its declared
/// loader closure. `false` is a DESIGNED state (sc-22511 E8/E9): a pin bump that moves the LTX-2.5
/// loader — sc-22414's coherence guard did exactly that — stales the packaged anchor, admission
/// demotes that cell to the analytic floor, and the render still runs. A test that needs the
/// PACKAGED anchor to carry a selection reports and steps aside on `false` rather than rebuilding
/// the pin-bump-forces-re-measurement gate one level down; a test that only needs AN anchor uses
/// [`current_loader_anchor_store`] and keeps its coverage at every pin.
fn packaged_ltx25_anchor_is_current() -> bool {
    let (Some(packaged), Some(closures)) = (
        sceneworks_core::memory_anchor::packaged_memory_anchors(),
        sceneworks_core::memory_anchor::packaged_anchor_loader_closures(),
    ) else {
        return false;
    };
    let current = packaged
        .anchors
        .iter()
        .filter(|anchor| anchor.model_id == "ltx_2_5")
        .any(|anchor| anchor.is_current(closures));
    if !current {
        eprintln!(
            "note: no packaged ltx_2_5 anchor is current against its declared loader closure — \
             the production funnel correctly prices from the analytic floor. Re-stamp or \
             re-measure; this is a designed state, not a failure."
        );
    }
    current
}

/// The packaged store with every anchor's currency key restamped to the LIVE declaration for its
/// `(model, backend)` — the inverse of [`staled_loader_anchor_store`]. Lets a selector test prove
/// the anchor-derived path carries a selection regardless of whether the measurement behind the
/// packaged anchor is current at this pin: the derivation under test is the same either way, and
/// currency itself is asserted separately (`a_foreign_identity_or_a_moved_loader_closure_…`).
fn current_loader_anchor_store() -> sceneworks_core::memory_anchor::MemoryAnchorStore {
    let closures = sceneworks_core::memory_anchor::packaged_anchor_loader_closures()
        .expect("the packaged loader closures load");
    let mut restamped: serde_json::Value =
        serde_json::from_str(sceneworks_core::memory_anchor::PACKAGED_MEMORY_ANCHORS)
            .expect("packaged anchors parse");
    for anchor in restamped["anchors"]
        .as_array_mut()
        .expect("the store carries anchors")
    {
        let model_id = anchor["modelId"].as_str().expect("anchor names its model");
        let backend = match anchor["backend"].as_str() {
            Some("mlx") => sceneworks_core::memory_anchor::AnchorBackend::Mlx,
            Some("candle") => sceneworks_core::memory_anchor::AnchorBackend::Candle,
            other => panic!("unexpected anchor backend {other:?}"),
        };
        if let Some(digest) = closures.digest_for(model_id, backend) {
            anchor["source"]["loaderClosureDigest"] = serde_json::json!(digest);
        }
    }
    sceneworks_core::memory_anchor::load_memory_anchors(&restamped.to_string())
        .expect("a restamped currency digest is still a well-formed store")
}

/// An anchor store whose anchors cite a loader closure that no longer matches the declaration —
/// i.e. the model's own loader source moved since the measurement (sc-22511).
fn staled_loader_anchor_store() -> sceneworks_core::memory_anchor::MemoryAnchorStore {
    let mut doctored: serde_json::Value =
        serde_json::from_str(sceneworks_core::memory_anchor::PACKAGED_MEMORY_ANCHORS)
            .expect("packaged anchors parse");
    for anchor in doctored["anchors"]
        .as_array_mut()
        .expect("the store carries anchors")
    {
        anchor["source"]["loaderClosureDigest"] = serde_json::json!("f".repeat(64));
    }
    sceneworks_core::memory_anchor::load_memory_anchors(&doctored.to_string())
        .expect("a doctored currency digest is still a well-formed store")
}

/// Every identity/currency axis the anchor derivation binds, exercised one mutation at a time.
///
/// `anchor_derived_phase_peaks` is called directly here because a foreign provider id also makes the
/// CONTRACT undecidable to the shared selector, which would mask the anchor guard behind an
/// unrelated (and equally safe) demotion. The end-to-end control below then proves one of these axes
/// really does land on the phase-blind floor through `select`.
///
/// CURRENCY IS THE LOADER CLOSURE, AND ONLY THAT (sc-22511, E9). The calibration axes that used to
/// appear in this list — an absent calibration identity, a moved ABI, a later campaign's
/// fingerprint — are asserted in the OPPOSITE direction below: none of them may demote an anchor
/// whose loader never moved.
#[test]
fn a_foreign_identity_or_a_moved_loader_closure_does_not_reach_the_anchor_derivation() {
    let baseline_contract = ltx25_fixture_contract(&[]);
    let derived = |identity: VideoRequestIdentity<'_>, contract: &MemoryProviderContract| {
        let selector = LadderVideoSelector::new(identity, contract, budget(128.0), 18 * GIB, 0);
        anchor_derived_phase_peaks(&selector, ltx25_unmeasured_geometry(), &[])
            .map(|(_, anchor_id)| anchor_id.to_owned())
    };

    // THE BASELINE PRESUPPOSES A CURRENT ANCHOR, and currency is allowed to be false BY DESIGN
    // (sc-22511 E8/E9): a pin bump that genuinely moves the LTX-2.5 loader stales this anchor,
    // admission demotes that cell to the conservative floor, and the render still runs. That is not
    // a defect in this test, and turning it into a red would rebuild the pin-bump-forces-
    // re-measurement gate one level down. So the designed state is REPORTED and the mutation sweep
    // — which asks what the derivation binds, and needs a reachable anchor to ask it — steps aside.
    // A CURRENT anchor that still fails to reach the derivation is a real defect and still fails.
    let baseline = derived(
        ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE),
        &baseline_contract,
    );
    let packaged = sceneworks_core::memory_anchor::packaged_memory_anchors()
        .expect("the packaged anchor store loads");
    let closures = sceneworks_core::memory_anchor::packaged_anchor_loader_closures()
        .expect("the packaged loader closures load");
    let ltx25_current = packaged
        .anchors
        .iter()
        .filter(|anchor| anchor.model_id == "ltx_2_5")
        .any(|anchor| anchor.is_current(closures));
    if !ltx25_current {
        eprintln!(
            "note: no packaged ltx_2_5 anchor is current against its declared loader closure — \
             the derivation is correctly unreachable and the identity sweep below has nothing to \
             bind. Re-stamp or re-measure; this is a designed state, not a failure."
        );
        assert_eq!(
            baseline, None,
            "a stale anchor must not reach the derivation"
        );
        return;
    }
    assert_eq!(
        baseline.as_deref(),
        Some("ltx_2_5:mlx:q8:dev:diffvae:sc-18797-ltx-2-5-mlx-ladder-v1:imc-7f8186376a9a3143ebee"),
        "the conformant baseline must reach the anchor, or every mutation below proves nothing"
    );

    // Identity axes: family, mode, route, and pipeline cell.
    for (label, mutate) in [
        (
            "foreign family",
            (|identity: &mut VideoRequestIdentity<'static>| identity.model_family = "ltx-video-x")
                as fn(&mut VideoRequestIdentity<'static>),
        ),
        ("foreign mode", |identity| identity.mode = "image_to_video"),
        ("foreign route", |identity| identity.route = "ltx_2_5_eros"),
        // "unmeasured decoder" (dev/conv) is NOT an unmeasured cell: it carries its own exact
        // anchor, whose TILED measured regime refuses this unbounded request — and an exact
        // anchor's regime refusal deliberately does not fall through to a sibling (the measured
        // cell's own evidence may not be bypassed by a delta).
        ("unmeasured decoder", |identity| {
            identity.decoder = Some(Ltx25Decoder::Conv)
        }),
        ("absent pipeline identity", |identity| {
            identity.transformer_variant = None
        }),
    ] {
        let mut identity = ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE);
        mutate(&mut identity);
        assert_eq!(
            derived(identity, &baseline_contract),
            None,
            "{label} must not reach the anchor derivation"
        );
    }

    // An UNMEASURED variant is no longer a refusal (epic 22505 feature-end fix round, E2): the
    // cell derives from the dev/diffvae SIBLING anchor plus the bound distillation-LoRA delta,
    // and the evidence names the sibling — never pretends to be a distilled measurement.
    let mut unmeasured_variant = ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE);
    unmeasured_variant.transformer_variant = Some(Ltx25TransformerVariant::Distilled);
    assert_eq!(
        derived(unmeasured_variant, &baseline_contract).as_deref(),
        Some("ltx_2_5:mlx:q8:dev:diffvae:sc-18797-ltx-2-5-mlx-ladder-v1:imc-7f8186376a9a3143ebee"),
        "the unmeasured variant must derive from its sibling anchor plus the bound delta"
    );

    // Contract axes: the provider that produced the measurement, and calibration currency.
    let mut foreign_provider = ltx25_fixture_contract(&[]);
    foreign_provider.provider_id = "ltx_2_5_community_rehost".to_owned();

    let mut absent_calibration = ltx25_fixture_contract(&[]);
    absent_calibration.calibration = None;

    let mut foreign_fingerprint = ltx25_fixture_contract(&[]);
    foreign_fingerprint.calibration = Some(MemoryCalibrationIdentity {
        abi: gen_core::MEMORY_CALIBRATION_ABI,
        fingerprint: "sc-99999-some-later-campaign-v1".to_owned(),
        load_shape: LoadShape::EagerMaterialization,
    });

    assert_eq!(
        derived(
            ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE),
            &foreign_provider
        ),
        None,
        "a foreign provider must not reach the anchor derivation"
    );

    // E9: the calibration axes are PROVENANCE. An absent calibration identity, a moved ABI and a
    // later campaign's fingerprint all leave an anchor whose loader never moved authoritative.
    for (label, contract, abi) in [
        (
            "no calibration identity",
            &absent_calibration,
            gen_core::MEMORY_CALIBRATION_ABI,
        ),
        (
            "later campaign fingerprint",
            &foreign_fingerprint,
            gen_core::MEMORY_CALIBRATION_ABI,
        ),
        (
            "moved calibration ABI",
            &baseline_contract,
            gen_core::MEMORY_CALIBRATION_ABI + 1,
        ),
    ] {
        let mut identity = ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE);
        identity.calibration_abi = abi;
        assert!(
            derived(identity, contract).is_some(),
            "{label} must NOT demote an anchor whose loader closure is unchanged"
        );
    }

    // THE currency axis: the model's own loader closure moved since the measurement.
    let staled = staled_loader_anchor_store();
    let selector = LadderVideoSelector::new(
        ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE),
        &baseline_contract,
        budget(128.0),
        18 * GIB,
        0,
    )
    .with_anchor_store(Some(&staled));
    assert_eq!(
        anchor_derived_phase_peaks(&selector, ltx25_unmeasured_geometry(), &[])
            .map(|(_, anchor_id)| anchor_id.to_owned()),
        None,
        "an anchor whose loader closure moved must not price a request"
    );

    // End-to-end control: that demotion really does land on the phase-blind floor.
    let mut selector = LadderVideoSelector::new(
        ltx25_identity(crate::mlx_fit_gate::UNCALIBRATED_CLOSURE),
        &baseline_contract,
        budget(128.0),
        18 * GIB,
        0,
    )
    .with_anchor_store(Some(&staled));
    assert!(matches!(
        selector.select(ltx25_unmeasured_geometry()),
        VideoRungSelection::Selected { .. }
    ));
    assert_eq!(
        selector.selections[0].evidence_revision, "video-estimate-floor-v1",
        "a moved loader closure must demote the anchor derivation to the floor"
    );

    // …and the evidence gate closes on exactly the same event, through the same seam.
    let mut request = inputs(89, budget(128.0), 18 * GIB);
    request.model_id = "ltx_2_5";
    request.route = "ltx_2_5";
    request.transformer_variant = Some(Ltx25TransformerVariant::Dev);
    request.decoder = Some(Ltx25Decoder::DiffVae);
    request.width = 640;
    request.height = 640;
    request.fps = 25;
    assert!(
        anchor_evidence_covers_request(
            sceneworks_core::memory_anchor::packaged_memory_anchors(),
            &baseline_contract,
            &request
        ),
        "the packaged anchors are current, so the gate must be open"
    );
    assert!(
        !anchor_evidence_covers_request(Some(&staled), &baseline_contract, &request),
        "a moved loader closure must not keep the anchor evidence gate open"
    );
    // The campaign fingerprint moving does not close it — E9 again, at the gate.
    assert!(
        anchor_evidence_covers_request(
            sceneworks_core::memory_anchor::packaged_memory_anchors(),
            &foreign_fingerprint,
            &request
        ),
        "a later campaign fingerprint must not close the anchor evidence gate"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-22667 (epic 22657 E3/E4): the contract's architecture facts reach the core law through
// `architecture_facts_from_contract`, axis by axis, `None` preserved.
// ---------------------------------------------------------------------------------------------

/// A conformant contract carrying exactly `facts` as its architecture block.
fn contract_stating(facts: gen_core::MemoryArchitectureFacts) -> MemoryProviderContract {
    let mut contract = fixture_contract(40, 5, &[MemoryStrategy::StagedResidency]);
    contract.architecture_facts = facts;
    contract
}

/// Every axis, with a setter on the gen-core block and a reader on the core facts, and a value
/// distinct per axis so a swapped translation cannot pass as a faithful one.
type ContractAxisSet = fn(&mut gen_core::MemoryArchitectureFacts, u32);
type CoreAxisGet = fn(&sceneworks_core::memory_anchor::ArchitectureFacts) -> Option<u32>;
const ARCHITECTURE_AXES: [(&str, u32, ContractAxisSet, CoreAxisGet); 8] = [
    (
        "attention_heads",
        30,
        |facts, value| facts.attention_heads = Some(value),
        |facts| facts.attention_heads,
    ),
    (
        "head_dim",
        128,
        |facts, value| facts.head_dim = Some(value),
        |facts| facts.head_dim,
    ),
    (
        "transformer_blocks",
        60,
        |facts, value| facts.transformer_blocks = Some(value),
        |facts| facts.transformer_blocks,
    ),
    (
        "patch_size",
        2,
        |facts, value| facts.patch_size = Some(value),
        |facts| facts.patch_size,
    ),
    (
        "latent_channels",
        16,
        |facts, value| facts.latent_channels = Some(value),
        |facts| facts.latent_channels,
    ),
    (
        "vae_spatial_scale",
        8,
        |facts, value| facts.vae_spatial_scale = Some(value),
        |facts| facts.vae_spatial_scale,
    ),
    (
        "vae_temporal_scale",
        4,
        |facts, value| facts.vae_temporal_scale = Some(value),
        |facts| facts.vae_temporal_scale,
    ),
    (
        "activation_dtype_width",
        3,
        |facts, value| facts.activation_dtype_width = Some(value),
        |facts| facts.activation_dtype_width,
    ),
];

/// One axis stated on the contract arrives on exactly that core axis, and on no other — the
/// seven others stay `None`. MUTATION: hard-wiring any translated axis to `None`, or crossing two
/// axes (`head_dim: attention_heads`), reds the arm named after it.
fn assert_axis_translates_alone(axis: &str) {
    let (label, value, set, get) = ARCHITECTURE_AXES
        .into_iter()
        .find(|(label, ..)| *label == axis)
        .unwrap_or_else(|| panic!("{axis} is not an architecture axis"));
    let mut stated = gen_core::MemoryArchitectureFacts::default();
    set(&mut stated, value);
    let translated = architecture_facts_from_contract(&contract_stating(stated));
    assert_eq!(
        get(&translated),
        Some(value),
        "{label}: the stated axis must arrive"
    );
    for (other, _, _, other_get) in ARCHITECTURE_AXES {
        if other != label {
            assert_eq!(
                other_get(&translated),
                None,
                "{label}: must not arrive on {other} — an unstated axis stays None"
            );
        }
    }
}

#[test]
fn architecture_facts_translate_attention_heads() {
    assert_axis_translates_alone("attention_heads");
}

#[test]
fn architecture_facts_translate_head_dim() {
    assert_axis_translates_alone("head_dim");
}

#[test]
fn architecture_facts_translate_transformer_blocks() {
    assert_axis_translates_alone("transformer_blocks");
}

#[test]
fn architecture_facts_translate_patch_size() {
    assert_axis_translates_alone("patch_size");
}

#[test]
fn architecture_facts_translate_latent_channels() {
    assert_axis_translates_alone("latent_channels");
}

#[test]
fn architecture_facts_translate_vae_spatial_scale() {
    assert_axis_translates_alone("vae_spatial_scale");
}

#[test]
fn architecture_facts_translate_vae_temporal_scale() {
    assert_axis_translates_alone("vae_temporal_scale");
}

#[test]
fn architecture_facts_translate_activation_dtype_width() {
    assert_axis_translates_alone("activation_dtype_width");
}

/// A contract stating every axis translates to every axis, each with its own value — the
/// exhaustive complement of the one-at-a-time arms, which is what catches a translation that
/// reads the right axis only because the others were absent.
#[test]
fn architecture_facts_translate_every_axis_together() {
    let mut stated = gen_core::MemoryArchitectureFacts::default();
    for (_, value, set, _) in ARCHITECTURE_AXES {
        set(&mut stated, value);
    }
    assert!(!stated.is_empty());
    let translated = architecture_facts_from_contract(&contract_stating(stated));
    for (label, value, _, get) in ARCHITECTURE_AXES {
        assert_eq!(get(&translated), Some(value), "{label}");
    }
    assert_eq!(
        translated,
        sceneworks_core::memory_anchor::ArchitectureFacts {
            attention_heads: Some(30),
            head_dim: Some(128),
            transformer_blocks: Some(60),
            patch_size: Some(2),
            latent_channels: Some(16),
            vae_spatial_scale: Some(8),
            vae_temporal_scale: Some(4),
            activation_dtype_width: Some(3),
        }
    );
}

/// A contract with every axis absent — the registry's weights-free surfaces, a single-file
/// import, a provider that has not adopted the block, and `compatibility_default` itself —
/// yields the default core facts: every axis `None`, none of them zero. The law then leaves every
/// residue unscaled (its `missing_facts_leave_residues_unscaled_and_never_shrink_the_estimate`),
/// which is the conservative reading and the reason zero-by-default would be a defect here.
/// MUTATION: `.unwrap_or(0)` (or `Some(0)`) on any translated axis reds both arms.
#[test]
fn a_contract_stating_no_architecture_facts_yields_the_default_core_facts() {
    let empty = gen_core::MemoryArchitectureFacts::default();
    assert!(empty.is_empty());
    let translated = architecture_facts_from_contract(&contract_stating(empty));
    assert_eq!(
        translated,
        sceneworks_core::memory_anchor::ArchitectureFacts::default()
    );
    for (label, _, _, get) in ARCHITECTURE_AXES {
        assert_eq!(
            get(&translated),
            None,
            "{label}: absent stays absent, never zero"
        );
    }
    // The compatibility default a not-yet-adopting provider publishes states none either.
    let compatibility = MemoryProviderContract::compatibility_default(
        "ltx_2_3",
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    assert_eq!(
        architecture_facts_from_contract(&compatibility),
        sceneworks_core::memory_anchor::ArchitectureFacts::default()
    );
}
