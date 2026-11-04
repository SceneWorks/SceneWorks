//! Backend-neutral construction of memory-selection metadata. No model execution lives here.
use std::sync::{Mutex, OnceLock};

use gen_core::{
    ComponentPrecisionFloor, MemoryConformanceState, MemoryEvidence, MemoryEvidenceDimensions,
    MemoryEvidenceKey, MemoryEvidenceVerdict, MemoryParityContract, MemoryParityResult,
    MemoryProviderContract, MemoryStrategy, MemoryStrategyParameters, Quant, TransformerComponent,
};

/// Retain exactly the declarations applying to the requested tier, in provider order.
///
/// `gen_core::effective_component_quant` applies floors individually; a descriptor is a table of
/// conditional substitutions, not an all-or-nothing group. Dense tiers have no quantized floors.
/// The pinned `MemoryNumericTier` requires a static slice. Mixed tables are interned once per
/// distinct matching subset (not once per request); uniform tables retain their original slice.
pub fn active_component_floors(
    declared: &'static [ComponentPrecisionFloor],
    selected: Option<Quant>,
) -> &'static [ComponentPrecisionFloor] {
    let Some(selected) = selected else { return &[] };
    let count = declared
        .iter()
        .filter(|floor| floor.applies_to(selected))
        .count();
    if count == 0 {
        return &[];
    }
    if count == declared.len() {
        return declared;
    }
    let active: Vec<_> = declared
        .iter()
        .copied()
        .filter(|floor| floor.applies_to(selected))
        .collect();
    static SUBSETS: OnceLock<Mutex<Vec<&'static [ComponentPrecisionFloor]>>> = OnceLock::new();
    let mut subsets = SUBSETS
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Some(existing) = subsets
        .iter()
        .find(|existing| **existing == active.as_slice())
    {
        return existing;
    }
    let subset = Box::leak(active.into_boxed_slice());
    subsets.push(subset);
    subset
}

/// Build unverified estimate metadata without discarding any lane-specific identity axis.
/// The caller owns the full key and calibration identity; measured records never use this helper.
pub fn unverified_estimate(
    key: MemoryEvidenceKey,
    calibration_abi: u32,
    calibration_fingerprint: String,
    sceneworks_revision: String,
    predicted_peak_bytes: u64,
) -> MemoryEvidence {
    MemoryEvidence {
        key,
        conformance: MemoryConformanceState::ImplementedUnverified,
        dimensions: MemoryEvidenceDimensions {
            static_implementation: MemoryEvidenceVerdict::Satisfied,
            declared_calibration: MemoryEvidenceVerdict::Missing,
            historical_verification: MemoryEvidenceVerdict::Missing,
            current_environment_verification: MemoryEvidenceVerdict::Missing,
            canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
            exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
        },
        calibration_abi,
        calibration_fingerprint,
        sceneworks_revision,
        inference_revision: crate::inference_provenance::INFERENCE_REVISION.to_owned(),
        harness_version: String::new(),
        predicted_peak_bytes,
        observed_peak_bytes: None,
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::NotRun,
    }
}

/// Existing provider/caller scope conventions, separate from the shared numeric minima.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformerScopeParameters {
    /// Candle and SenseNova preserve the provider's ordered representative, including explicit DiT.
    ProviderOrder,
    /// MLX video opts into scope selection only with adopted native-quant phase facts. DiT keeps
    /// its legacy implicit representation; the native selector subsequently enumerates all scopes.
    NativeQuant,
    /// The MLX image compatibility path retains its historical implicit DiT request.
    ImplicitDit,
}

/// Smallest declared numeric value per engaged knob. Unengaged knobs remain absent; an empty
/// required domain refuses the candidate. Scope conventions preserve each lane's evidence-key
/// representation and native phase adoption boundary without duplicating parameter construction.
pub fn smallest_parameters(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    scope: TransformerScopeParameters,
) -> Option<MemoryStrategyParameters> {
    let smallest = |strategy: MemoryStrategy,
                    pick: fn(&gen_core::MemoryParameterRanges) -> &Vec<u32>|
     -> Option<Option<u32>> {
        if !engaged.contains(&strategy) {
            return Some(None);
        }
        pick(&contract.capability(strategy)?.parameters)
            .iter()
            .copied()
            .min()
            .map(Some)
    };
    let transformer_window_component =
        if engaged.contains(&MemoryStrategy::BoundedTransformerResidency) {
            let components = &contract
                .capability(MemoryStrategy::BoundedTransformerResidency)?
                .parameters
                .transformer_window_components;
            match scope {
                TransformerScopeParameters::ProviderOrder => components.first().copied(),
                TransformerScopeParameters::NativeQuant
                    if contract
                        .phase_facts
                        .as_ref()
                        .and_then(|phase| phase.native_quant.as_ref())
                        .is_some() =>
                {
                    components
                        .iter()
                        .copied()
                        .min()
                        .filter(|component| *component != TransformerComponent::Dit)
                }
                _ => None,
            }
        } else {
            None
        };
    Some(MemoryStrategyParameters {
        stage_residency: None,
        decode_tile_edge: smallest(MemoryStrategy::BoundedDecode, |ranges| {
            &ranges.decode_tile_edges
        })?,
        decode_overlap: smallest(MemoryStrategy::BoundedDecode, |ranges| {
            &ranges.decode_overlaps
        })?,
        attention_chunk_size: smallest(MemoryStrategy::BoundedAttention, |ranges| {
            &ranges.attention_chunk_sizes
        })?,
        transformer_window_size: smallest(MemoryStrategy::BoundedTransformerResidency, |ranges| {
            &ranges.transformer_window_sizes
        })?,
        transformer_window_component,
    })
}

/// Translate the selected composition to the arithmetic law's request regime. The optional
/// parameters are already validated by the provider at admission. Missing values remain absent,
/// preserving the MLX law's historical unbounded interpretation; Candle's strict wrapper rejects
/// incomplete engaged knobs before invoking this mapping. The anchor law's transformer is the
/// denoiser: encoder-only windows keep it resident. Native phase allocation consumes the complete
/// selection separately and retains its encoder/window/format constraints.
pub fn request_regime(
    engaged: &[MemoryStrategy],
    parameters: &MemoryStrategyParameters,
) -> crate::memory_anchor::RequestRegime {
    use crate::memory_anchor::{DecodeTile, RequestRegime};
    RequestRegime {
        staged: engaged.contains(&MemoryStrategy::StagedResidency),
        decode_tile: engaged
            .contains(&MemoryStrategy::BoundedDecode)
            .then(|| {
                Some(DecodeTile {
                    edge: parameters.decode_tile_edge?,
                    overlap: parameters.decode_overlap?,
                })
            })
            .flatten(),
        attention_chunk_scores: engaged
            .contains(&MemoryStrategy::BoundedAttention)
            .then_some(parameters.attention_chunk_size)
            .flatten()
            .map(u64::from),
        transformer_window: (engaged.contains(&MemoryStrategy::BoundedTransformerResidency)
            && parameters.window_component().includes_dit())
        .then_some(parameters.transformer_window_size)
        .flatten(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{
        LoadShape, MemoryBackend, MemoryBackendRealization, MemoryGeometry, MemoryMode,
        MemoryNumericTier, MemoryParameterRanges, MemoryReferenceShape, MemorySelection,
        MemoryStrategyCapability, MemoryStrategySupport, MemoryWindowMaterialization, Precision,
        PrecisionFloorComponent,
    };

    const MIXED: &[ComponentPrecisionFloor] = &[
        ComponentPrecisionFloor {
            component: PrecisionFloorComponent::TextEncoder,
            selected_tier: Quant::Q4,
            resident_tier: Quant::Q8,
        },
        ComponentPrecisionFloor {
            component: PrecisionFloorComponent::TextEncoder,
            selected_tier: Quant::Nvfp4,
            resident_tier: Quant::Fp8Rowwise,
        },
        ComponentPrecisionFloor {
            component: PrecisionFloorComponent::TransformerHead,
            selected_tier: Quant::Q4,
            resident_tier: Quant::Q8,
        },
    ];

    #[test]
    fn component_floor_activation_filters_each_entry_and_interns_mixed_subsets() {
        for selected in [
            None,
            Some(Quant::Q4),
            Some(Quant::Q8),
            Some(Quant::Nvfp4),
            Some(Quant::Fp8Rowwise),
        ] {
            let expected: Vec<_> = MIXED
                .iter()
                .copied()
                .filter(|floor| selected.is_some_and(|tier| floor.applies_to(tier)))
                .collect();
            let actual = active_component_floors(MIXED, selected);
            assert_eq!(actual, expected);
            assert!(std::ptr::eq(
                actual,
                active_component_floors(MIXED, selected)
            ));
            assert!(active_component_floors(&[], selected).is_empty());
            if let Some(selected) = selected {
                for component in [
                    PrecisionFloorComponent::TextEncoder,
                    PrecisionFloorComponent::TransformerHead,
                ] {
                    assert_eq!(
                        gen_core::effective_component_quant(actual, component, selected),
                        gen_core::effective_component_quant(MIXED, component, selected)
                    );
                }
            }
        }
        assert!(std::ptr::eq(
            active_component_floors(&MIXED[..1], Some(Quant::Q4)),
            &MIXED[..1]
        ));
    }

    fn contract() -> MemoryProviderContract {
        let mut contract = MemoryProviderContract::compatibility_default(
            "fixture",
            MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        contract.load_shape = LoadShape::DeferredMaterialization;
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: MemoryStrategySupport::Implemented,
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: vec![512, 128, 256],
                        decode_overlaps: vec![64, 16],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                        attention_chunk_sizes: vec![2048, 512],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency => MemoryParameterRanges {
                        transformer_window_sizes: vec![8, 2, 4],
                        transformer_window_components: vec![
                            TransformerComponent::TextEncoder,
                            TransformerComponent::Both,
                        ],
                        ..Default::default()
                    },
                    _ => Default::default(),
                },
            })
            .collect();
        contract.lifecycle = gen_core::MemoryLifecycleCapabilities {
            phases: vec![
                gen_core::MemoryPhase::Conditioning,
                gen_core::MemoryPhase::Denoise,
                gen_core::MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        };
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        contract
    }

    fn tier() -> MemoryNumericTier {
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: active_component_floors(MIXED, Some(Quant::Q4)),
        }
    }

    #[test]
    fn smallest_declared_parameters_and_regime_preserve_engagement_and_scopes() {
        let mut contract = contract();
        for strategy in MemoryStrategy::ALL {
            let engaged = contract.engaged_composition(strategy);
            let parameters = smallest_parameters(
                &contract,
                &engaged,
                TransformerScopeParameters::ProviderOrder,
            )
            .unwrap();
            let selection = MemorySelection {
                strategy,
                parameters,
                tier: tier(),
            };
            contract.validate_selection(&selection).unwrap();
            let regime = request_regime(&engaged, &parameters);
            assert_eq!(
                regime.staged,
                engaged.contains(&MemoryStrategy::StagedResidency)
            );
            assert_eq!(
                parameters.decode_tile_edge,
                engaged
                    .contains(&MemoryStrategy::BoundedDecode)
                    .then_some(128)
            );
            assert_eq!(
                parameters.decode_overlap,
                engaged
                    .contains(&MemoryStrategy::BoundedDecode)
                    .then_some(16)
            );
            assert_eq!(
                parameters.attention_chunk_size,
                engaged
                    .contains(&MemoryStrategy::BoundedAttention)
                    .then_some(512)
            );
            assert_eq!(
                parameters.transformer_window_size,
                engaged
                    .contains(&MemoryStrategy::BoundedTransformerResidency)
                    .then_some(2)
            );
            assert_eq!(
                parameters.transformer_window_component,
                engaged
                    .contains(&MemoryStrategy::BoundedTransformerResidency)
                    .then_some(TransformerComponent::TextEncoder)
            );
            assert_eq!(
                regime.decode_tile.map(|tile| (tile.edge, tile.overlap)),
                parameters.decode_tile_edge.zip(parameters.decode_overlap)
            );
            assert_eq!(
                regime.attention_chunk_scores,
                parameters.attention_chunk_size.map(u64::from)
            );
            assert_eq!(
                regime.transformer_window, None,
                "encoder-only windows grant no denoiser credit in the anchor law"
            );
        }
        let engaged = contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency);
        assert_eq!(
            smallest_parameters(&contract, &engaged, TransformerScopeParameters::NativeQuant)
                .unwrap()
                .transformer_window_component,
            None
        );
        assert_eq!(
            smallest_parameters(&contract, &engaged, TransformerScopeParameters::ImplicitDit)
                .unwrap()
                .transformer_window_component,
            None
        );
        // Native formats keep the same declared scope: phase metadata is not a switch that may
        // silently turn a text-encoder window into a DiT window.
        contract.phase_facts = Some(gen_core::MemoryPhaseFacts {
            native_quant: Some(gen_core::h3_memory::NativeAllocationFacts {
                conditioning: None,
                conditioning_stream: None,
                conditioning_resident_bytes: 0,
                conditioning_load_bytes: 0,
                audio_decode_load_bytes: 0,
                video_decode_resident_bytes: 0,
                video_decode_load_bytes: 0,
                audio_decode_resident_bytes: 0,
                default_evaluations: 1,
                backend: gen_core::h3_memory::NativeBackend::Mlx { tile_columns: 32 },
                tasks: vec![],
            }),
            ..gen_core::MemoryPhaseFacts::staged(gen_core::StagedWeightSchedule::ThreeStage)
        });
        contract
            .strategies
            .iter_mut()
            .find(|cap| cap.strategy == MemoryStrategy::BoundedTransformerResidency)
            .unwrap()
            .parameters
            .transformer_window_components
            .reverse();
        assert_eq!(
            smallest_parameters(
                &contract,
                &engaged,
                TransformerScopeParameters::ProviderOrder
            )
            .unwrap()
            .transformer_window_component,
            Some(TransformerComponent::Both)
        );
        assert_eq!(
            smallest_parameters(&contract, &engaged, TransformerScopeParameters::NativeQuant)
                .unwrap()
                .transformer_window_component,
            Some(TransformerComponent::TextEncoder)
        );
        contract
            .strategies
            .iter_mut()
            .find(|cap| cap.strategy == MemoryStrategy::BoundedTransformerResidency)
            .unwrap()
            .parameters
            .transformer_window_components
            .reverse();
        let engaged = contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency);
        let parameters = smallest_parameters(
            &contract,
            &engaged,
            TransformerScopeParameters::ProviderOrder,
        )
        .unwrap();
        assert_eq!(
            parameters.transformer_window_component,
            Some(TransformerComponent::TextEncoder)
        );
        contract
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters,
                tier: tier(),
            })
            .unwrap();
        let window = contract
            .strategies
            .iter_mut()
            .find(|cap| cap.strategy == MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        window.parameters.transformer_window_components = vec![TransformerComponent::Dit];
        let dit_parameters = smallest_parameters(
            &contract,
            &engaged,
            TransformerScopeParameters::ProviderOrder,
        )
        .unwrap();
        assert_eq!(
            request_regime(&engaged, &dit_parameters).transformer_window,
            Some(2)
        );
        let both_parameters = MemoryStrategyParameters {
            transformer_window_component: Some(TransformerComponent::Both),
            ..dit_parameters
        };
        assert_eq!(
            request_regime(&engaged, &both_parameters).transformer_window,
            Some(2)
        );
        assert_eq!(
            smallest_parameters(
                &contract,
                &engaged,
                TransformerScopeParameters::ProviderOrder
            )
            .unwrap()
            .transformer_window_component,
            Some(TransformerComponent::Dit)
        );
        contract
            .strategies
            .iter_mut()
            .find(|cap| cap.strategy == MemoryStrategy::BoundedTransformerResidency)
            .unwrap()
            .parameters
            .transformer_window_sizes
            .clear();
        assert!(smallest_parameters(
            &contract,
            &engaged,
            TransformerScopeParameters::ProviderOrder
        )
        .is_none());
        assert_eq!(
            smallest_parameters(
                &contract,
                &[MemoryStrategy::Resident],
                TransformerScopeParameters::ProviderOrder
            ),
            Some(MemoryStrategyParameters::default())
        );
        contract
            .strategies
            .retain(|cap| cap.strategy != MemoryStrategy::BoundedAttention);
        assert!(smallest_parameters(
            &contract,
            &[MemoryStrategy::BoundedAttention],
            TransformerScopeParameters::ProviderOrder
        )
        .is_none());
    }

    #[test]
    fn estimate_constructor_preserves_the_complete_lane_key_and_claims_only_an_estimate() {
        for backend in [MemoryBackend::Mlx, MemoryBackend::Candle] {
            let contract = contract();
            let strategy = MemoryStrategy::BoundedTransformerResidency;
            let engaged = contract.engaged_composition(strategy);
            let parameters = smallest_parameters(
                &contract,
                &engaged,
                TransformerScopeParameters::ProviderOrder,
            )
            .unwrap();
            let key = MemoryEvidenceKey {
                model_family: "family".into(),
                resolved_route: "route/control".into(),
                backend,
                tier: tier(),
                load_shape: LoadShape::DeferredMaterialization,
                mode: MemoryMode::ImageToImage,
                reference_shape: MemoryReferenceShape::Video,
                overlay: Some("control+lora".into()),
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 768,
                    batch: 2,
                    frames: 25,
                    reference_count: 3,
                },
                frames_per_second: Some(24),
                strategy,
                engaged_composition: engaged,
                parameters,
            };
            let evidence = unverified_estimate(
                key.clone(),
                91,
                "fingerprint".into(),
                "caller-revision".into(),
                12345,
            );
            assert_eq!(evidence.key, key);
            assert_eq!(evidence.calibration_abi, 91);
            assert_eq!(evidence.calibration_fingerprint, "fingerprint");
            assert_eq!(evidence.sceneworks_revision, "caller-revision");
            assert_eq!(
                evidence.inference_revision,
                crate::inference_provenance::INFERENCE_REVISION
            );
            assert_eq!(
                evidence.conformance,
                MemoryConformanceState::ImplementedUnverified
            );
            assert_eq!(
                evidence.dimensions,
                MemoryEvidenceDimensions {
                    static_implementation: MemoryEvidenceVerdict::Satisfied,
                    declared_calibration: MemoryEvidenceVerdict::Missing,
                    historical_verification: MemoryEvidenceVerdict::Missing,
                    current_environment_verification: MemoryEvidenceVerdict::Missing,
                    canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
                    exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
                }
            );
            assert_eq!(evidence.predicted_peak_bytes, 12345);
            assert_eq!(evidence.observed_peak_bytes, None);
            assert!(evidence.harness_version.is_empty());
            assert_eq!(evidence.parity, MemoryParityContract::Exact);
            assert_eq!(evidence.parity_result, MemoryParityResult::NotRun);
        }
    }
}
