//! Byte/metadata-only regressions for the production H3 admission path. No tensors or inference.
use super::*;

fn request<'a>(
    frames: u32,
    veda: Option<&'a gen_core::VedaRequest>,
    total: u64,
) -> VideoAdmissionInputs<'a> {
    let mut request = inputs(frames, budget(total as f64 / GIB as f64), GIB);
    request.model_id = "minimax_h3";
    request.model_family = "minimax_h3";
    request.route = "minimax_h3";
    request.lane = VideoLane::Candle;
    request.transformer_variant = None;
    request.decoder = None;
    request.width = 1344;
    request.height = 768;
    let mut native = native_quant_request(&[]);
    native.shape.veda = veda;
    native.shape.width = request.width;
    native.shape.height = request.height;
    native.shape.frames = frames;
    request.native_quant_request = Some(native);
    request
}

#[test]
fn native_phase_floor_keeps_conditioning_and_decode_out_of_veda_denoise() {
    let mut contract = native_quant_contract(VideoLane::Candle, 2);
    // Deliberately make the conditioner dominate every DiT phase. Charging Veda there
    // would reject a separately fitting phase, even though those allocations never coexist.
    contract.asset_facts.conditioning_bytes = 200 * GIB;
    contract.asset_facts.base_bytes = 205 * GIB;
    let veda = gen_core::VedaRequest {
        enabled: true,
        ..Default::default()
    };
    let mut shape = native_quant_request(&[]);
    shape.shape.veda = Some(&veda);
    let geometry = geometry(345, VideoGeometryRole::Requested);
    let selection = selection_for(MemoryStrategy::Resident);
    let mut selector = native_quant_selector(&contract, shape, VideoLane::Candle);
    selector.headroom_bytes = GIB;
    let weights = native_cuda_phase_weights(&selector, geometry, selection)
        .unwrap()
        .unwrap();
    assert_eq!(weights.conditioning_bytes, 201 * GIB);
    assert!(weights.denoise_bytes < weights.conditioning_bytes);
    assert!(weights.decode_bytes < weights.conditioning_bytes);
    let sparse =
        native_quant_phase_floor(&contract, &selection, Some(shape), geometry, weights).unwrap();
    shape.shape.veda = None;
    let dense =
        native_quant_phase_floor(&contract, &selection, Some(shape), geometry, weights).unwrap();
    assert_eq!(sparse.conditioning_bytes, dense.conditioning_bytes);
    assert_eq!(sparse.decode_bytes, dense.decode_bytes);
    assert_eq!(
        sparse.peak_bytes(),
        dense.peak_bytes(),
        "only the simultaneously live denoise phase may grow"
    );
}

#[test]
fn native_cuda_wrapper_uses_same_headroom_and_requires_selected_context() {
    let mut contract = native_quant_contract(VideoLane::Candle, 2);
    contract.asset_facts.conditioning_bytes = 200 * GIB;
    contract.asset_facts.base_bytes = 205 * GIB;
    let generator = fixture_generator(Some(contract));
    let veda = gen_core::VedaRequest {
        enabled: true,
        ..Default::default()
    };
    let shape = gen_core::GenerationRequest {
        width: 1344,
        height: 768,
        frames: Some(345),
        veda: Some(veda.clone()),
        ..Default::default()
    };
    let bound =
        gen_core_h3::veda::workspace::for_request(&shape, gen_core::veda::workspace::Backend::Cuda)
            .unwrap()
            .unwrap();
    let observed = std::cell::Cell::new(None);
    let result = admit_with_veda_bound(
        &generator,
        request(345, Some(&veda), 512 * GIB),
        Some(&bound),
        GIB,
        |generator, input| {
            observed.set(Some(input.headroom_bytes));
            admit_video_generation_with_curves(generator, input, None)
        },
    )
    .unwrap();
    assert!(result.refusal.is_none(), "{:?}", result.refusal);
    assert!(result.context.is_some());
    assert_eq!(
        observed.get(),
        Some(GIB),
        "Veda is composed in denoise, never in global headroom"
    );
    assert!(
        admit_with_veda_bound(
            &generator,
            request(345, Some(&veda), 512 * GIB),
            Some(&bound),
            GIB,
            |_, _| VideoAdmissionOutcome::default()
        )
        .is_err(),
        "an undecided native source must not execute resident"
    );
    let mut missing = request(345, Some(&veda), 512 * GIB);
    missing.runtime = None;
    assert!(admit_with_veda_bound(
        &generator,
        missing,
        Some(&bound),
        GIB,
        |generator, input| admit_video_generation_with_curves(generator, input, None)
    )
    .is_err());
}

#[test]
fn current_cuda_budget_and_request_duration_are_regraded_for_cold_and_queued_jobs() {
    let generator = fixture_generator(Some(native_quant_contract(VideoLane::Candle, 2)));
    let admit = |frames, total, cache| {
        let mut input = request(frames, None, total);
        input.runtime.as_mut().unwrap().cache_state = cache;
        admit_with_veda_bound(&generator, input, None, GIB, |generator, input| {
            admit_video_generation_with_curves(generator, input, None)
        })
        .unwrap()
    };
    let short = admit(124, 512 * GIB, MemoryCacheState::Cold)
        .context
        .expect("short admitted");
    let long = admit(345, 512 * GIB, MemoryCacheState::Cold)
        .context
        .expect("long admitted");
    assert_eq!(short.geometry.frames, 124);
    assert_eq!(long.geometry.frames, 345);
    assert!(
        long.predicted_peak_bytes > short.predicted_peak_bytes,
        "124-frame catalog evidence must not price the long request"
    );
    let warm = admit(345, 512 * GIB, MemoryCacheState::Warm)
        .context
        .expect("warm admitted");
    assert_eq!(warm.predicted_peak_bytes, long.predicted_peak_bytes);
    let refused = admit(345, 1, MemoryCacheState::Warm)
        .refusal
        .expect("current smaller budget refuses");
    assert!(refused.contains("Limiting pool: GPU; phase:"), "{refused}");
    assert!(!refused.contains("smaller quant"));
}

#[test]
fn host_pool_exact_boundary_is_independent_and_unknown_is_not_zero() {
    let need = 123_456_789;
    assert!(admit_h3_host_bytes(need, Some(need)).is_ok());
    let refused = admit_h3_host_bytes(need, Some(need - 1))
        .unwrap_err()
        .to_string();
    assert!(refused.contains("host") && refused.contains("Free host RAM"));
    assert!(admit_h3_host_bytes(need, None).is_err());
}

#[test]
fn gpu_pool_exact_ceiling_fits_and_one_byte_less_refuses_the_same_request() {
    let contract = native_quant_contract(VideoLane::Candle, 2);
    let input = request(345, None, 512 * GIB);
    let mut selector = native_quant_selector(
        &contract,
        input.native_quant_request.unwrap(),
        VideoLane::Candle,
    );
    let geometry = VideoAdmissionGeometry {
        width: input.width,
        height: input.height,
        ..geometry(345, VideoGeometryRole::Requested)
    };
    let VideoRungSelection::Selected { needed_gb, .. } = selector.select(geometry) else {
        panic!("fixture must select")
    };
    let ceiling = (needed_gb * GIB as f64).round() as u64;
    let generator = fixture_generator(Some(contract));
    for cache in [MemoryCacheState::Cold, MemoryCacheState::Warm] {
        let admit = |bytes| {
            let mut input = request(345, None, bytes);
            input.runtime.as_mut().unwrap().cache_state = cache;
            admit_with_veda_bound(&generator, input, None, GIB, |generator, input| {
                admit_video_generation_with_curves(generator, input, None)
            })
            .unwrap()
        };
        assert!(
            admit(ceiling).context.is_some(),
            "exact GPU ceiling fits in {cache:?}"
        );
        let error = admit(ceiling - 1)
            .refusal
            .expect("one byte below the same selected ceiling refuses");
        assert!(error.contains("Limiting pool: GPU; phase:"), "{error}");
    }
}

#[test]
fn too_small_attention_row_excludes_only_that_candidate() {
    let mut contract = native_quant_contract(VideoLane::Candle, 2);
    let policy = fixture_contract_with_realization(
        6,
        1,
        &[
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ],
        contract.backend.clone(),
    );
    contract.strategies = policy.strategies;
    contract.lifecycle = policy.lifecycle;
    contract
        .strategies
        .iter_mut()
        .find(|cap| cap.strategy == MemoryStrategy::BoundedAttention)
        .unwrap()
        .parameters
        .attention_chunk_sizes = vec![1 << 20, 8 << 20, 32 << 20];
    let veda = gen_core::VedaRequest {
        enabled: true,
        ..Default::default()
    };
    let input = request(345, Some(&veda), 512 * GIB);
    let shape = input.native_quant_request.unwrap();
    let geometry = VideoAdmissionGeometry {
        width: input.width,
        height: input.height,
        ..geometry(345, VideoGeometryRole::Requested)
    };
    let invalid = video_estimate_selections(&contract, tier())
        .into_iter()
        .find(|selection| selection.parameters.attention_chunk_size == Some(1 << 20))
        .unwrap();
    assert!(native_quant_phase_floor(
        &contract,
        &invalid,
        Some(shape),
        geometry,
        PhasePeaks {
            conditioning_bytes: 0,
            denoise_bytes: 0,
            decode_bytes: 0
        }
    )
    .is_err());
    let generator = fixture_generator(Some(contract));
    let outcome = admit_with_veda_bound(&generator, input, None, GIB, |generator, input| {
        admit_video_generation_with_curves(generator, input, None)
    })
    .unwrap();
    assert!(outcome.refusal.is_none(), "{:?}", outcome.refusal);
    assert!(
        outcome.context.is_some(),
        "a valid declared attention domain survives the invalid row"
    );
}

#[test]
fn overflowing_source_overlay_refuses_instead_of_saturating_into_a_fit() {
    let mut contract = native_quant_contract(VideoLane::Candle, 2);
    contract.asset_facts.overlay_bytes = u64::MAX;
    let generator = fixture_generator(Some(contract));
    let outcome = admit_with_veda_bound(
        &generator,
        request(345, None, 512 * GIB),
        None,
        GIB,
        |generator, input| admit_video_generation_with_curves(generator, input, None),
    )
    .unwrap();
    assert!(outcome.context.is_none());
    let error = outcome
        .refusal
        .expect("overflow is a source admission failure");
    assert!(error.contains("overflow"), "{error}");
}

#[test]
fn complete_native_decode_graph_replaces_only_the_generic_decode_activation_floor() {
    let contract = native_quant_with_conditioning_contract(VideoLane::Candle);
    let shape = native_quant_request(&[]);
    let geometry = VideoAdmissionGeometry {
        width: 32,
        height: 32,
        ..geometry(124, VideoGeometryRole::Requested)
    };
    let selection = selection_for(MemoryStrategy::Resident);
    let mut selector = native_quant_selector(&contract, shape, VideoLane::Candle);
    selector.headroom_bytes = 128 * GIB;
    let weights = native_cuda_phase_weights(&selector, geometry, selection)
        .unwrap()
        .unwrap();
    let phased =
        native_quant_phase_floor(&contract, &selection, Some(shape), geometry, weights).unwrap();
    assert!(
        phased.decode_bytes > weights.decode_bytes,
        "the complete source workspace is still charged"
    );
    assert!(
        phased.decode_bytes < 128 * GIB,
        "an unrelated transformer fallback must not price the complete decoder graph"
    );
    assert_eq!(
        phased.conditioning_bytes, weights.conditioning_bytes,
        "the text-encoder fallback remains priced"
    );
    assert!(phased.conditioning_bytes >= 128 * GIB);
    let mut partial = contract.clone();
    partial
        .phase_facts
        .as_mut()
        .unwrap()
        .native_quant
        .as_mut()
        .unwrap()
        .conditioning = None;
    let mut selector = native_quant_selector(&partial, shape, VideoLane::Candle);
    selector.headroom_bytes = 128 * GIB;
    assert!(
        native_cuda_phase_weights(&selector, geometry, selection)
            .unwrap()
            .unwrap()
            .decode_bytes
            >= 128 * GIB,
        "a source with no complete decoder graph retains its conservative fallback"
    );
}
