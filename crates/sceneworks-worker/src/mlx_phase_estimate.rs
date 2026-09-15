//! Active working sets for MLX's actual phase schedule. Cache is reclaimable and is not
//! part of these peaks. Provider facts describe executed code; retained observations supply
//! workspace terms. These are estimates, never new measured calibration records.

use gen_core::{
    DecoderTilingRealization, DecoderWorkspaceFacts, MemoryGeometry, MemoryProviderContract,
    MemorySelection, MemoryStrategy, StagedWeightSchedule,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Workspace {
    pub conditioning: u64,
    pub denoise: u64,
    pub decode: u64,
}

/// Sum of the dual-CLIP constructor's activation tensors, retaining every layer's intermediates
/// instead of assuming reuse. Per layer: two norms (2D), Q/K/V (3D), attention output/projection
/// (2D), residuals (2D), MLP input/output (4D+4D+D), final output (D), rounded up to 20D.
/// Also charge full attention scores although fused SDPA does not materialize them. Compute at
/// f32, CFG batch two, for both the 12x768/12-head and 32x1280/20-head constructors.
pub(crate) fn dual_clip_workspace(windows: u32) -> u64 {
    let layer_states = 20_u64 * 77 * (12 * 768 + 32 * 1280);
    let scores = 77_u64 * 77 * (12 * 12 + 32 * 20);
    let outputs = 4_u64 * 77 * (768 + 1280);
    (layer_states + scores + outputs)
        .saturating_mul(4 * 2)
        .saturating_mul(u64::from(windows))
}

/// Retained layer-wise GroupNorm/convolution sweep: inference Actions 31918926833,
/// job 95095446324 (2026-08-16), 1024², f32 decoder computation, staged q4 ZImage.
/// Fit the global feature-map term and bounded convolution term from edges 256 and 512.
/// Other edges are held-out checks. Subtract only the decoder weights present in that sweep.
/// Printed peaks have three decimals; the upper endpoint includes the rounding interval.
/// Below the measured output size retain the global term rather than inventing its fixed/area
/// split. The model applies only to the identical four-stage decoder topology and compute width.
pub(crate) fn layerwise_decode_workspace(
    facts: &DecoderWorkspaceFacts,
    geometry: MemoryGeometry,
    edge: u32,
) -> Option<u64> {
    if facts.tiling != DecoderTilingRealization::LayerwiseConvolution
        || facts.activation_dtype_width != 4
        || facts.channels != [512, 512, 256, 128]
        || facts.input_channels != [512, 512, 512, 256]
        || facts.spatial_divisors != [8, 4, 2, 1]
        || geometry.frames != 1
        || geometry.batch != 1
        || !(256..=768).contains(&edge)
    {
        return None;
    }
    const GIB: f64 = 1_073_741_824.0;
    const REFERENCE_DECODER_BYTES: f64 = 97_583_622.0;
    let tile_slope = (6.129 - 4.457) * GIB / (512_f64.powi(2) - 256_f64.powi(2));
    let global = 4.4575 * GIB - REFERENCE_DECODER_BYTES - tile_slope * 256_f64.powi(2);
    let area_scale =
        (f64::from(geometry.width) * f64::from(geometry.height) / 1024_f64.powi(2)).max(1.0);
    Some((global * area_scale + tile_slope * f64::from(edge).powi(2)).ceil() as u64)
}

/// Phase weight lifetimes, including the nonstreamable trunk. A window is materialized only
/// during denoise; after its final evaluation all window blocks have been dropped. Two-stage
/// providers still retain the trunk and decoder through decode. Unknown streaming facts earn no
/// weight reduction, and TE-only streaming cannot reduce the denoiser's weights.
pub(crate) fn peak(
    contract: &MemoryProviderContract,
    selection: &MemorySelection,
    workspace: Workspace,
) -> Option<u64> {
    let phase = contract.phase_facts.as_ref()?;
    let facts = contract.asset_facts;
    if facts.base_bytes == 0 {
        return None;
    }
    let staged = contract.engages_selection(selection, MemoryStrategy::StagedResidency);
    let streamed = contract
        .engages_selection(selection, MemoryStrategy::BoundedTransformerResidency)
        && selection.parameters.window_component().includes_dit();
    let stream = phase
        .transformer_stream
        .as_ref()
        .filter(|stream| stream.peak_bytes(None) == facts.transformer_bytes)
        .filter(|_| streamed);
    let denoiser = stream.map_or(facts.transformer_bytes, |stream| {
        stream.peak_bytes(selection.parameters.transformer_window_size)
    });
    let trunk = stream.map_or(facts.transformer_bytes, |stream| stream.resident_bytes);
    // Keep every unclassified base byte and charge auxiliary weights across every phase.
    let remainder = facts.base_bytes.saturating_sub(
        facts
            .conditioning_bytes
            .saturating_add(facts.transformer_bytes)
            .saturating_add(facts.decoder_bytes),
    );
    let cond_weights = if staged {
        facts.conditioning_bytes
    } else {
        facts
            .conditioning_bytes
            .saturating_add(trunk)
            .saturating_add(facts.decoder_bytes)
    };
    // Three-stage means the denoiser is shed before decode; it does not prove the decoder
    // was unmaterialized during denoise (previews and load-time quantization can touch it).
    let den_weights = denoiser
        .saturating_add(if staged { 0 } else { facts.conditioning_bytes })
        .saturating_add(facts.decoder_bytes);
    let decode_weights = facts
        .decoder_bytes
        .saturating_add(if staged { 0 } else { facts.conditioning_bytes })
        .saturating_add(
            if staged && phase.staged_weights == StagedWeightSchedule::ThreeStage {
                0
            } else {
                trunk
            },
        );
    Some(
        cond_weights
            .saturating_add(workspace.conditioning)
            .max(den_weights.saturating_add(workspace.denoise))
            .max(decode_weights.saturating_add(workspace.decode))
            .saturating_add(remainder)
            .saturating_add(facts.overlay_bytes),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_lifetimes_and_streaming_scope_price_only_simultaneous_weights() {
        use gen_core::{
            MemoryAssetFacts, MemoryBackendRealization, MemoryNumericTier, MemoryPhaseFacts,
            MemoryStrategyParameters, MemoryStrategySupport, StreamedWeightFacts,
        };
        let mut contract = MemoryProviderContract::compatibility_default(
            "phase-test",
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        for cap in &mut contract.strategies {
            cap.support = MemoryStrategySupport::Implemented;
        }
        contract.asset_facts = MemoryAssetFacts {
            base_bytes: 9,
            conditioning_bytes: 2,
            transformer_bytes: 6,
            decoder_bytes: 1,
            overlay_bytes: 0,
        };
        contract.phase_facts = Some(MemoryPhaseFacts {
            architecture: None,
            staged_weights: StagedWeightSchedule::ThreeStage,
            transformer_stream: Some(StreamedWeightFacts {
                resident_bytes: 2,
                stacks: vec![vec![2, 2]],
            }),
            decoder_workspace: None,
        });
        let workspace = Workspace {
            conditioning: 1,
            denoise: 1,
            decode: 5,
        };
        let mut selection = MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q4),
                component_precision_floors: &[],
            },
            parameters: MemoryStrategyParameters {
                stage_residency: Some(true),
                transformer_window_size: Some(1),
                ..Default::default()
            },
        };
        assert_eq!(peak(&contract, &selection, workspace), Some(6));
        assert_eq!(
            peak(
                &contract,
                &selection,
                Workspace {
                    conditioning: 0,
                    denoise: 10,
                    decode: 0
                }
            ),
            Some(15),
            "three-stage retains decoder bytes during denoise"
        );
        let engaged = [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ];
        assert_eq!(
            crate::mlx_fit_gate::mlx_fallback_weights_bytes(
                &contract,
                &engaged,
                selection.parameters
            ),
            5
        );
        let mut te_only = selection.parameters;
        te_only.transformer_window_component = Some(gen_core::TransformerComponent::TextEncoder);
        assert_eq!(
            crate::mlx_fit_gate::mlx_fallback_weights_bytes(&contract, &engaged, te_only),
            7
        );
        let mut unknown = contract.clone();
        unknown.phase_facts = None;
        assert_eq!(
            crate::mlx_fit_gate::mlx_fallback_weights_bytes(
                &unknown,
                &engaged,
                selection.parameters
            ),
            7
        );

        contract.phase_facts.as_mut().unwrap().staged_weights = StagedWeightSchedule::TwoStage;
        assert_eq!(peak(&contract, &selection, workspace), Some(8));
        selection.parameters.stage_residency = None;
        assert_eq!(peak(&contract, &selection, workspace), Some(10));
        selection.parameters.transformer_window_component =
            Some(gen_core::TransformerComponent::TextEncoder);
        assert_eq!(peak(&contract, &selection, workspace), Some(14));
        selection.parameters.transformer_window_component = None;
        contract
            .phase_facts
            .as_mut()
            .unwrap()
            .transformer_stream
            .as_mut()
            .unwrap()
            .resident_bytes = 1;
        assert_eq!(
            peak(&contract, &selection, workspace),
            Some(14),
            "mismatching inventories cannot earn credit"
        );
    }

    #[test]
    fn clip_workload_bound_preserves_multiple_prompt_windows() {
        let one = dual_clip_workspace(1);
        assert!(one > 0 && one < 1_073_741_824);
        assert_eq!(dual_clip_workspace(3), 3 * one);
        assert_eq!(crate::mlx_fit_gate::clip_window_upper_bound("fox", ""), 1);
        assert_eq!(
            crate::mlx_fit_gate::clip_window_upper_bound("fox", &"x".repeat(151)),
            3
        );
    }

    #[test]
    fn retained_decode_sweep_is_bounded_without_charging_unrelated_weights() {
        let facts = DecoderWorkspaceFacts {
            tiling: DecoderTilingRealization::LayerwiseConvolution,
            activation_dtype_width: 4,
            channels: vec![512, 512, 256, 128],
            input_channels: vec![512, 512, 512, 256],
            spatial_divisors: vec![8, 4, 2, 1],
        };
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            frames: 1,
            batch: 1,
            reference_count: 0,
        };
        for (edge, observed) in [
            (256, 4.457),
            (320, 4.548),
            (384, 4.708),
            (448, 5.376),
            (512, 6.129),
            (640, 6.840),
            (768, 8.595),
        ] {
            let predicted =
                layerwise_decode_workspace(&facts, geometry, edge).unwrap() + 97_583_622;
            assert!(
                predicted as f64 / 1_073_741_824.0 >= observed,
                "edge {edge}"
            );
        }
        let mut unsupported = facts.clone();
        unsupported.activation_dtype_width = 2;
        assert!(layerwise_decode_workspace(&unsupported, geometry, 256).is_none());
        assert!(layerwise_decode_workspace(&facts, geometry, 128).is_none());
    }
}
