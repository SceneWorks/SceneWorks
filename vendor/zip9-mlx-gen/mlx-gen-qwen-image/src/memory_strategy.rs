//! Qwen-Image MLX shared memory-ladder contract (SC-15511, SC-16353).
//!
//! Rung 1 uses the existing staged `Residency` lifecycle; rung 2 drives the head-once/tail-tiled
//! Qwen VAE over its measured production tile ladder; rung 3 threads the shared MLX attention
//! planner through every one of the 60 joint-attention blocks; rung 4 uses the shared block-window
//! primitive from SC-16353. The provider contract is the only selector surface.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
#[cfg(test)]
use mlx_gen::gen_core::GenerationMemory;
use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite,
    MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

/// Load shape is a typed evidence-key axis; this content fingerprint remains shape-independent.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "qwen-image-mlx-shared-ladder-2026-08-01-v1";
/// The Wan terminal decoder has no promoted whole-request measurement. Give the composite its own
/// identity so native-Qwen evidence cannot be reused; SceneWorks must admit it through the
/// asset-facts + generic-headroom estimate path, where the estimate safety margin is applied.
pub const WAN_DECODER_CALIBRATION_FINGERPRINT: &str =
    "qwen-image-mlx-wan21-decoder-unmeasured-composite-2026-08-10-v1";

fn calibration_fingerprint(spec: &LoadSpec) -> &'static str {
    if spec.components.contains_key(mlx_gen::VAE_COMPONENT) {
        WAN_DECODER_CALIBRATION_FINGERPRINT
    } else {
        MEMORY_CALIBRATION_FINGERPRINT
    }
}

/// Native Qwen-VAE production tile ladder in output pixels, measured against the exact untiled
/// decode on the real bf16 VAE. SC-15511's same-process Metal A/B found overlap 96 increased the
/// incremental active peak by 12 MiB at both 448- and 256-pixel tiles versus overlap 64, so only 64
/// is shipped. A candidate is not a production range merely because it changes seam blending.
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 448, 384, 320, 256];
pub const DECODE_OVERLAP: u32 = 64;
pub const REJECTED_SUB_512_OVERLAP: u32 = 96;

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

/// The shared 64-Mi score-element budget used by the MLX rung-3 kernel.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
/// SC-16353 measured 1/2/4/8 plus the unbounded 60-block control at 1024² across Q4/Q8/BF16.
/// Only window 1 materially lowered the denoise counter versus the same-stream unbounded control;
/// windows above 1 were indistinguishable noise or worse and never moved the conditioning-bound
/// request peak. The all-covering 60-block arm is deliberately not publishable:
/// [`mlx_gen::block_residency::BlockPlan::is_bounded`] defines it as fully resident.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[TRANSFORMER_WINDOW_SIZE];

fn transformer_blocks(provider_id: &str) -> usize {
    if provider_id == crate::model_edit::MODEL_ID {
        crate::transformer::QwenTransformerConfig::qwen_image_edit().num_layers
    } else {
        crate::transformer::QwenTransformerConfig::qwen_image().num_layers
    }
}

pub(crate) fn is_streamable_spec(spec: &LoadSpec) -> bool {
    let matching_device_format = match (&spec.weights, spec.quantize) {
        (WeightsSource::Dir(root), None) => {
            matches!(
                mlx_gen::quant::packed_quant_bits(root, "transformer"),
                Ok(None)
            )
        }
        (WeightsSource::Dir(root), Some(quant)) => matches!(
            mlx_gen::quant::packed_quant_bits(root, "transformer"),
            Ok(Some(bits)) if bits == quant.bits()
        ),
        (WeightsSource::File(_), _) => false,
    };
    matching_device_format
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.precision, Precision::Bf16)
        && spec.pid.is_none()
}

/// Whether a provider may arm the Qwen transformer block stream at load time. Base and Edit own a
/// published transformer-window lifecycle; Control does not, because its separate five-block
/// branch remains unbounded.
pub(crate) fn should_arm_block_stream(provider_id: &str, spec: &LoadSpec) -> bool {
    provider_id != crate::model_control::MODEL_ID && is_streamable_spec(spec)
}

/// Resolve the physical DiT cadence for a request. Eligible deferred loads always use the stream.
/// SC-16353 found that reopening the view more often lowers the denoise-only counter but never the
/// whole-request peak at 1024². The published cadence is nevertheless genuinely bounded; the
/// all-covering plan remains an attribution control, never a selectable rung-4 parameter.
pub(crate) fn resolve_window_size(
    request: &GenerationRequest,
    contract: &MemoryProviderContract,
) -> mlx_gen::Result<Option<usize>> {
    let requested = request
        .memory
        .is_some_and(|memory| memory.stream_transformer_blocks);
    if requested && !contract.lifecycle.transformer_window_materialization {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{}: bounded transformer residency requires Sequential + DeferredMaterialization + directory weights",
            contract.provider_id
        )));
    }
    if !contract.lifecycle.transformer_window_materialization {
        return Ok(None);
    }
    if requested {
        let memory = request.memory.expect("requested checked");
        if memory.transformer_window_component.unwrap_or_default() != TransformerComponent::Dit {
            return Err(mlx_gen::Error::Unsupported(format!(
                "{}: only the DiT transformer window is implemented",
                contract.provider_id
            )));
        }
        return Ok(Some(
            memory
                .transformer_window_size
                .unwrap_or(TRANSFORMER_WINDOW_SIZE) as usize,
        ));
    }
    Ok(Some(transformer_blocks(&contract.provider_id)))
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let _ = crate::model::component_footprint(spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "qwen-image memory facts require a snapshot directory".to_owned(),
        ));
    };
    let encoder_contract = crate::active_encoder_contract();
    let selected_text_encoder = encoder_contract.source_for_load(spec, root)?;
    let language = selected_text_encoder.materialized_language_tensor_headers(&encoder_contract)?;
    let vision = if provider_id == crate::model_edit::MODEL_ID {
        let vision_source = encoder_contract
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
        Some(vision_source.materialized_vision_tensor_headers(
            &crate::active_vision_encoder_contract(),
            &encoder_contract,
        )?)
    } else {
        None
    };
    let conditioning_bytes =
        crate::model::projected_conditioning_headers(&language, vision.as_deref())?;
    let transformer_bytes = projected_component_bytes(&root.join("transformer"), spec.quantize)?;
    let native_decoder_bytes = projected_component_bytes(&root.join("vae"), None)?;
    // The alternate lane keeps the native VAE loaded (reference/img2img encoding still uses it) and
    // adds the standalone Wan decoder for the terminal decode. Price the composition, never substitute
    // donor bytes for the native decoder or borrow the native route's measured peak unchanged.
    let alternate_decoder_bytes = match spec.components.get(mlx_gen::VAE_COMPONENT) {
        Some(WeightsSource::Dir(path)) => {
            projected_safetensors_bytes(path, |_| ResidentProjection::Stored)?
        }
        Some(WeightsSource::File(path)) => spec.read_file_unchanged_if_prepared(path, |p| {
            projected_safetensors_bytes(p, |_| ResidentProjection::Stored)
        })?,
        None => 0,
    };
    let decoder_bytes = native_decoder_bytes.saturating_add(alternate_decoder_bytes);
    let overlay_bytes = match &spec.control {
        Some(WeightsSource::Dir(path)) | Some(WeightsSource::File(path)) => {
            projected_safetensors_bytes(path, |_| match spec.quantize {
                Some(quant) => ResidentProjection::GroupQuantized {
                    bits: quant.bits(),
                    group_size: crate::quant::GROUP_SIZE as usize,
                },
                None => ResidentProjection::Stored,
            })?
        }
        None => 0,
    };
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlay_bytes,
    )
}

fn projected_component_bytes(
    path: &std::path::Path,
    quant: Option<mlx_gen::Quant>,
) -> CoreResult<u64> {
    projected_safetensors_bytes(path, |_| match quant {
        Some(quant) => ResidentProjection::GroupQuantized {
            bits: quant.bits(),
            group_size: crate::quant::GROUP_SIZE as usize,
        },
        None => ResidentProjection::Stored,
    })
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    memory_strategy_contract_with_asset_facts(provider_id, spec, 0, 0, 0, 0)
}

/// Whether the registry surface this selector names ships a device-format transformer.
///
/// [`is_streamable_spec`] answers that by probing the packed-quant marker on disk, which a
/// weights-free witness has no directory to read. The selector already names the *resolved*
/// artifact tier, so an on-disk packed snapshot at that tier is exactly the shape it denotes and
/// the marker probe is satisfied by construction. Every other axis still comes from the spec.
fn surface_is_streamable(surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec) -> bool {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    matches!(
        surface.resolved_artifact_tier(),
        MemoryContractSurfaceTier::Bf16
            | MemoryContractSurfaceTier::Q4
            | MemoryContractSurfaceTier::Q8
    ) && matches!(surface.spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(surface.spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(surface.spec.precision, Precision::Bf16)
        && surface.spec.pid.is_none()
}

/// Declaration-equivalent contract resolved from the explicit surface selector.
///
/// Registered for the base and edit routes only. Control keeps the `LoadSpec` factory: its separate
/// unbounded five-block branch excludes it from rungs 3 and 4 at every tier regardless of selector.
pub(crate) fn weights_free_memory_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<MemoryProviderContract> {
    contract_with_asset_facts_and_streamability(
        provider_id,
        &surface.spec,
        surface_is_streamable(surface),
        0,
        0,
        0,
        0,
    )
}

fn memory_strategy_contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
    overlay_bytes: u64,
) -> CoreResult<MemoryProviderContract> {
    contract_with_asset_facts_and_streamability(
        provider_id,
        spec,
        is_streamable_spec(spec),
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlay_bytes,
    )
}

/// Pixels per latent unit on each spatial axis — the four-stage Qwen-Image autoencoder, whose
/// declared tiling geometry (`VaeTiling::QWEN_IMAGE`) pins the same x8.
const VAE_SPATIAL_SCALE: u32 = 8;

/// Architecture axes for one registered Qwen-Image route (epic SC-22657, E2).
///
/// This crate mirrors the reference `transformer/config.json` as
/// [`QwenTransformerConfig`](crate::transformer::QwenTransformerConfig); the edit variant differs
/// only by the `zero_cond_t` modulation flag, so all three routes publish the same shape. The
/// control route adds a five-block branch beside the trunk — an overlay, not a change to the trunk's
/// own depth — so `transformer_blocks` stays the base `num_layers`.
///
/// `latent_channels` is the DiT's `out_channels`, which is what the decoder consumes;
/// `in_channels` is the *packed* width (`out_channels * patch²`) and would overstate the axis by 4x.
///
/// `vae_temporal_scale` stays `None`: Qwen-Image decodes a singleton frame through an image
/// autoencoder, and a structurally absent axis is declared absent, never zero.
fn architecture_facts(provider_id: &str) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit = if provider_id == "qwen_image_edit" {
        crate::transformer::QwenTransformerConfig::qwen_image_edit()
    } else {
        crate::transformer::QwenTransformerConfig::qwen_image()
    };
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: mlx_gen::architecture_facts::axis(dit.patch_size),
        latent_channels: mlx_gen::architecture_facts::axis(dit.out_channels),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(VAE_SPATIAL_SCALE),
        vae_temporal_scale: None,
        // The pipeline casts the text embeddings and drives the DiT at `Dtype::Bfloat16`.
        activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
    }
}

#[allow(clippy::too_many_arguments)]
fn contract_with_asset_facts_and_streamability(
    provider_id: &str,
    spec: &LoadSpec,
    streamable: bool,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
    overlay_bytes: u64,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
    // The optional control route owns a separate five-block attention branch which is not yet
    // windowed by the shared block loader.  It may use the native tiled VAE, but must not inherit
    // the base/edit route's rung-3 or rung-4 claims merely because those providers share a crate.
    let has_unbounded_control_branch = provider_id == "qwen_image_control";
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.phase_facts = Some(mlx_gen::gen_core::MemoryPhaseFacts::staged(
        mlx_gen::gen_core::StagedWeightSchedule::TwoStage,
    ));
    contract.architecture_facts = architecture_facts(provider_id);
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::OverlayBytes,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    contract.formula = if overlay_bytes > 0 {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components: vec![MemoryResidentComponent {
                id: "control_branch".to_owned(),
                kind: MemoryComponentKind::ControlBranch,
                resident_bytes: overlay_bytes,
                bounded_by: None,
                residency: MemoryComponentResidency::WholeRender,
            }],
        }
    } else {
        MemoryFormulaKind::PhaseEnvelope { phases, variables }
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        calibration_fingerprint(spec),
        spec.load_shape,
    ));
    contract.asset_facts.base_bytes = conditioning_bytes
        .saturating_add(transformer_bytes)
        .saturating_add(decoder_bytes);
    contract.asset_facts.conditioning_bytes = conditioning_bytes;
    contract.asset_facts.transformer_bytes = transformer_bytes;
    contract.asset_facts.decoder_bytes = decoder_bytes;
    contract.asset_facts.overlay_bytes = overlay_bytes;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        decode_tiling: true,
        attention_chunking: !has_unbounded_control_branch,
        transformer_window_materialization: streamable && !has_unbounded_control_branch,
    };

    let mut implemented_scratch = vec![(
        MemoryStrategy::BoundedDecode,
        MemoryParameterRanges {
            decode_tile_edges: routes.published_edges(),
            decode_overlaps: routes.published_overlaps(),
            ..Default::default()
        },
    )];
    if !has_unbounded_control_branch {
        implemented_scratch.push((
            MemoryStrategy::BoundedAttention,
            MemoryParameterRanges {
                attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                ..Default::default()
            },
        ));
    }
    for (strategy, parameters) in implemented_scratch {
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == strategy)
            .expect("compatibility contract contains every strategy");
        capability.support = MemoryStrategySupport::Implemented;
        capability.parameters = parameters;
    }
    contract.pid_decode_routes = Some(mlx_gen::gen_core::MemoryPidDecodeRoutes {
        native: mlx_gen::gen_core::MemoryDecodeRouteDomain {
            tile_edges: routes.native_edges().to_vec(),
            tile_overlap: DECODE_OVERLAP,
        },
        pid: mlx_gen::gen_core::MemoryDecodeRouteDomain {
            tile_edges: mlx_gen_pid::DecodeRoutes::pid_edges(),
            tile_overlap: mlx_gen_pid::DecodeRoutes::pid_overlap(),
        },
    });

    if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
        let staged = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("compatibility contract contains every strategy");
        staged.support = MemoryStrategySupport::Implemented;
    }
    if streamable && !has_unbounded_control_branch {
        let transformer_blocks = transformer_blocks(provider_id);
        let plan = mlx_gen::block_residency::BlockPlan::new(
            transformer_blocks,
            TRANSFORMER_WINDOW_SIZE as usize,
        )?;
        if !plan.is_bounded() {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: published transformer window {} does not bound the {}-block stack",
                TRANSFORMER_WINDOW_SIZE, transformer_blocks
            )));
        }
        let rung = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("compatibility contract contains every strategy");
        rung.support = MemoryStrategySupport::Implemented;
        rung.parameters.transformer_window_sizes = TRANSFORMER_WINDOW_SIZES.to_vec();
        rung.parameters.transformer_window_components = vec![TransformerComponent::Dit];
    }
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    Ok(contract)
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision,
            quant,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(contract, spec.precision, spec.quantize, context)
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let (mode, has_reference) = if contract.provider_id.ends_with("_edit") {
        (mlx_gen::gen_core::MemoryMode::Edit, true)
    } else if contract.provider_id.ends_with("_control") {
        (mlx_gen::gen_core::MemoryMode::ImageToImage, true)
    } else {
        (mlx_gen::gen_core::MemoryMode::TextToImage, false)
    };
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let route = |use_pid| mlx_gen::gen_core::MemoryBehaviorRoute {
        mode: mode.clone(),
        reference_count: u32::from(has_reference),
        use_pid,
        has_phases: false,
        overlay: None,
    };
    let mut fixtures = vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(
        mlx_gen::gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            tier,
            route(false),
        )?,
    )];
    if contract.engages(strategy, MemoryStrategy::BoundedDecode) {
        fixtures.push(mlx_gen::gen_core::MemoryBehaviorFixture::new(
            mlx_gen::gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                route(true),
            )?,
        ));
    }
    Ok(fixtures)
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        spec.precision,
        spec.quantize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

#[cfg(test)]
fn qwen_generation_memory(
    contract: &MemoryProviderContract,
    selection: &mlx_gen::gen_core::MemorySelection,
) -> Option<GenerationMemory> {
    contract.generation_memory(selection)
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        precision,
        quant,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_request_with_cleanup(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = decode_routes(provider_id)?;
    let transformer_blocks = transformer_blocks(provider_id);
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        transformer_blocks,
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryNumericTier, MemorySelection, MemoryStrategyParameters, MemoryStrategySupport,
    };

    fn exact_language_headers() -> Vec<mlx_gen::gen_core::SafetensorsTensorHeader> {
        gen_core_testkit::encoder_contract_fixture_tensor_headers(crate::ENCODER_CONTRACT, None)
            .unwrap()
    }

    fn exact_vision_headers() -> Vec<mlx_gen::gen_core::SafetensorsTensorHeader> {
        crate::VISION_ENCODER_CONTRACT
            .expected_headers()
            .unwrap()
            .into_iter()
            .map(|(name, shape)| mlx_gen::gen_core::SafetensorsTensorHeader {
                name,
                data_bytes: shape
                    .iter()
                    .try_fold(2_u64, |bytes, &dimension| {
                        bytes.checked_mul(dimension as u64)
                    })
                    .unwrap(),
                dtype: mlx_gen::gen_core::weightsmeta::Dtype::F16,
                shape,
            })
            .collect()
    }

    fn spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("qwen-memory-spec");
        for component in ["text_encoder", "transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    fn write_control(path: &std::path::Path) {
        let mut header =
            br#"{"control.weight":{"dtype":"BF16","shape":[2,64],"data_offsets":[0,256]}}"#
                .to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 256]);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn empty_required_component_directory_cannot_be_reported_as_zero() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        for component in ["text_encoder", "transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert!(projected_component_bytes(&root.join("transformer"), spec.quantize).is_err());
        assert!(weights_free_memory_strategy_contract("qwen_image", &spec).is_ok());
    }

    /// AC (SC-22662): every registered Qwen-Image route publishes the axes of the trunk it runs,
    /// derived from this crate's own transformer config, and passes the shared facts check.
    #[test]
    fn architecture_facts_follow_the_crate_transformer_config() {
        let tmp = tempfile::tempdir().unwrap();
        let expected = mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: Some(24),
            head_dim: Some(128),
            transformer_blocks: Some(60),
            patch_size: Some(2),
            // `out_channels`, the decoder-side width; `in_channels` 64 is the 2x2-packed view.
            latent_channels: Some(16),
            vae_spatial_scale: Some(8),
            vae_temporal_scale: None,
            activation_dtype_width: Some(2),
        };
        for provider_id in ["qwen_image", "qwen_image_edit", "qwen_image_control"] {
            let contract = weights_free_memory_strategy_contract(provider_id, &spec(&tmp)).unwrap();
            assert_eq!(
                contract.architecture_facts, expected,
                "{provider_id} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        // The packed DiT input width the loader consumes is exactly `latent x patch²`, so the two
        // published axes cannot drift apart from the config they came from.
        let dit = crate::transformer::QwenTransformerConfig::qwen_image();
        assert_eq!(
            dit.out_channels * dit.patch_size * dit.patch_size,
            dit.in_channels
        );
        // The constant's doc claims it is the same x8 `VaeTiling::QWEN_IMAGE` pins; assert that pin
        // rather than leaving the two free to drift.
        assert_eq!(
            VAE_SPATIAL_SCALE,
            mlx_gen::gen_core::tiling::VaeTiling::QWEN_IMAGE.spatial_scale as u32
        );
    }

    #[test]
    fn sequential_deferred_directory_declares_the_exact_dit_window() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = weights_free_memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        assert!(contract.conformance_errors().is_empty());
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.support, MemoryStrategySupport::Implemented);
        let routes = decode_routes("qwen_image").unwrap();
        assert_eq!(
            decode.parameters.decode_tile_edges,
            routes.published_edges()
        );
        assert_eq!(
            decode.parameters.decode_overlaps,
            routes.published_overlaps()
        );
        let declared_routes = contract.pid_decode_routes.as_ref().unwrap();
        assert_eq!(declared_routes.native.tile_edges, DECODE_TILE_EDGES);
        assert_eq!(
            declared_routes.pid.tile_edges,
            mlx_gen_pid::DecodeRoutes::pid_edges()
        );
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(attention.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            attention.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        let rung = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            rung.parameters.transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
        assert_eq!(
            rung.parameters.transformer_window_components,
            vec![TransformerComponent::Dit]
        );
        assert!(contract.engages(
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategy::StagedResidency
        ));
    }

    #[test]
    fn selected_encoder_pricing_ignores_nested_safetensors_not_loaded_as_shards() {
        let language = exact_language_headers();
        let before = crate::model::projected_conditioning_headers(&language, None).unwrap();
        let mut authored = language.clone();
        authored.push(mlx_gen::gen_core::SafetensorsTensorHeader {
            name: "archive.ignored.weight".into(),
            dtype: mlx_gen::gen_core::weightsmeta::Dtype::F16,
            shape: vec![2, 64],
            data_bytes: 256,
        });
        let materialized = crate::ENCODER_CONTRACT
            .materialized_dense_language_tensor_headers(&authored)
            .unwrap();
        let after = crate::model::projected_conditioning_headers(&materialized, None).unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn edit_prices_selected_language_plus_base_vision_while_base_and_control_exclude_visual() {
        let language = exact_language_headers();
        let vision = exact_vision_headers();
        let base = crate::model::projected_conditioning_headers(&language, None).unwrap();
        let control = crate::model::projected_conditioning_headers(&language, None).unwrap();
        let edit = crate::model::projected_conditioning_headers(&language, Some(&vision)).unwrap();
        assert_eq!(
            control, base,
            "control must not price visual.* from a multimodal source"
        );
        assert!(
            edit > base,
            "edit must add its separately loaded vision tower"
        );
        let vision_bytes = mlx_gen::asset_facts::projected_tensor_headers_bytes(&vision, |_| {
            ResidentProjection::Stored
        })
        .unwrap();
        assert_eq!(edit, base.checked_add(vision_bytes).unwrap());

        let alternate = gen_core_testkit::encoder_contract_fixture_tensor_headers(
            crate::ENCODER_CONTRACT,
            None,
        )
        .unwrap();
        let selected_base = crate::model::projected_conditioning_headers(&alternate, None).unwrap();
        let selected_control =
            crate::model::projected_conditioning_headers(&alternate, None).unwrap();
        let selected_edit =
            crate::model::projected_conditioning_headers(&alternate, Some(&vision)).unwrap();
        assert_eq!(selected_control, selected_base);
        assert_eq!(
            selected_edit,
            selected_base.checked_add(vision_bytes).unwrap(),
            "the language-only alternate must compose with base vision rather than replace it"
        );
    }

    #[test]
    fn block_stream_load_shape_predicate_is_exact_and_excludes_control() {
        let tmp = tempfile::tempdir().unwrap();
        let eligible = spec(&tmp);
        assert!(is_streamable_spec(&eligible));
        assert!(should_arm_block_stream(crate::model::MODEL_ID, &eligible));
        assert!(should_arm_block_stream(
            crate::model_edit::MODEL_ID,
            &eligible
        ));
        assert!(!should_arm_block_stream(
            crate::model_control::MODEL_ID,
            &eligible
        ));

        let mut eager = eligible.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        let mut resident = eligible.clone();
        resident.offload_policy = OffloadPolicy::Resident;
        let mut file = eligible.clone();
        file.weights = WeightsSource::File("transformer.safetensors".into());
        let mut load_time_quant = eligible;
        load_time_quant.quantize = Some(mlx_gen::Quant::Q4);
        for ineligible in [eager, resident, file, load_time_quant] {
            assert!(!is_streamable_spec(&ineligible));
            assert!(!should_arm_block_stream(
                crate::model::MODEL_ID,
                &ineligible
            ));
        }
    }

    #[test]
    fn checked_decode_routes_keep_native_geometry_out_of_pid_requests() {
        let routes = decode_routes("qwen_image").unwrap();
        assert_eq!(routes.native_edges(), DECODE_TILE_EDGES);
        routes
            .validate(false, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .unwrap();
        let error = routes
            .validate(true, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .unwrap_err();
        assert!(error.contains("PiD overlay"));
        assert!(error.contains("not a"));
    }

    #[test]
    fn control_route_does_not_overstate_its_unbounded_side_branch() {
        let tmp = tempfile::tempdir().unwrap();
        let contract =
            weights_free_memory_strategy_contract("qwen_image_control", &spec(&tmp)).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing,
                "the separate five-block control branch is not yet bounded"
            );
        }
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn control_overlay_is_quant_projected_typed_and_excluded_from_base() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        for component in ["transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
            write_control(&root.join(component).join("model.safetensors"));
        }
        let control = root.join("control.safetensors");
        write_control(&control);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(mlx_gen::Quant::Q4)
            .with_control(WeightsSource::File(control.clone()));
        let conditioning =
            crate::model::projected_conditioning_headers(&exact_language_headers(), None).unwrap();
        let transformer = projected_safetensors_bytes(root.join("transformer"), |_| {
            ResidentProjection::GroupQuantized {
                bits: 4,
                group_size: crate::quant::GROUP_SIZE as usize,
            }
        })
        .unwrap();
        let decoder =
            projected_safetensors_bytes(root.join("vae"), |_| ResidentProjection::Stored).unwrap();
        let overlay =
            projected_safetensors_bytes(&control, |_| ResidentProjection::GroupQuantized {
                bits: 4,
                group_size: crate::quant::GROUP_SIZE as usize,
            })
            .unwrap();
        let contract = memory_strategy_contract_with_asset_facts(
            "qwen_image_control",
            &spec,
            conditioning,
            transformer,
            decoder,
            overlay,
        )
        .unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, 14_141_238_272);
        assert_eq!(contract.asset_facts.transformer_bytes, 72);
        assert_eq!(contract.asset_facts.decoder_bytes, 256);
        assert_eq!(contract.asset_facts.base_bytes, 14_141_238_600);
        assert_eq!(contract.asset_facts.overlay_bytes, 72);
        assert_eq!(contract.auxiliary_resident_bytes(), 72);
        assert!(matches!(
            contract.formula,
            MemoryFormulaKind::ComponentPhaseEnvelope { .. }
        ));
        assert!(contract.conformance_errors().is_empty());
    }

    #[test]
    fn alternate_decoder_is_additive_to_native_decoder_asset_facts() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("qwen-alternate-decoder");
        for component in ["transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
            write_control(&root.join(component).join("model.safetensors"));
        }
        let mut base_spec = LoadSpec::new(WeightsSource::Dir(root));
        let conditioning =
            crate::model::projected_conditioning_headers(&exact_language_headers(), None).unwrap();
        let native = memory_strategy_contract_with_asset_facts(
            "qwen_image",
            &base_spec,
            conditioning,
            256,
            256,
            0,
        )
        .unwrap();
        let donor = tmp.path().join("wan-vae.safetensors");
        write_control(&donor);
        base_spec = base_spec.with_component(mlx_gen::VAE_COMPONENT, WeightsSource::File(donor));
        let composite = memory_strategy_contract_with_asset_facts(
            "qwen_image",
            &base_spec,
            conditioning,
            256,
            512,
            0,
        )
        .unwrap();
        assert_eq!(
            composite.asset_facts.decoder_bytes,
            native.asset_facts.decoder_bytes + 256,
            "reference encoding retains the native VAE while terminal decode adds the donor"
        );
        assert_eq!(
            composite.asset_facts.base_bytes,
            native.asset_facts.base_bytes + 256
        );
        assert_eq!(
            composite.calibration.as_ref().unwrap().fingerprint,
            WAN_DECODER_CALIBRATION_FINGERPRINT,
            "native whole-request measurements must not authorize the composite decoder path"
        );
        assert_eq!(
            native.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
    }

    fn selection(strategy: MemoryStrategy) -> MemorySelection {
        let mut parameters = MemoryStrategyParameters::default();
        if matches!(
            strategy,
            MemoryStrategy::BoundedDecode
                | MemoryStrategy::BoundedAttention
                | MemoryStrategy::BoundedTransformerResidency
        ) {
            parameters.decode_tile_edge = Some(DECODE_TILE_EDGE);
            parameters.decode_overlap = Some(DECODE_OVERLAP);
        }
        if matches!(
            strategy,
            MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency
        ) {
            parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
        }
        if strategy == MemoryStrategy::BoundedTransformerResidency {
            parameters.transformer_window_size = Some(TRANSFORMER_WINDOW_SIZE);
            parameters.transformer_window_component = Some(TransformerComponent::Dit);
        }
        MemorySelection {
            strategy,
            parameters,
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        }
    }

    #[test]
    fn selections_translate_to_the_shared_cumulative_request_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = weights_free_memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        let resident = qwen_generation_memory(&contract, &selection(MemoryStrategy::Resident));
        assert_eq!(resident, None);

        let staged =
            qwen_generation_memory(&contract, &selection(MemoryStrategy::StagedResidency)).unwrap();
        assert!(staged.stage_residency);
        assert!(!staged.tile_vae_decode);
        assert!(!staged.chunk_attention);
        assert!(!staged.stream_transformer_blocks);

        let decode =
            qwen_generation_memory(&contract, &selection(MemoryStrategy::BoundedDecode)).unwrap();
        assert!(!decode.stage_residency);
        assert!(decode.tile_vae_decode);
        assert_eq!(decode.decode_tile_edge, Some(DECODE_TILE_EDGE));
        assert_eq!(decode.decode_overlap, Some(DECODE_OVERLAP));
        assert!(!decode.chunk_attention);

        let attention =
            qwen_generation_memory(&contract, &selection(MemoryStrategy::BoundedAttention))
                .unwrap();
        assert!(attention.tile_vae_decode);
        assert!(attention.chunk_attention);
        assert!(!attention.stage_residency);

        let streamed = qwen_generation_memory(
            &contract,
            &selection(MemoryStrategy::BoundedTransformerResidency),
        )
        .unwrap();
        assert!(
            streamed.stage_residency,
            "Qwen rung 4 explicitly requires rung 1"
        );
        assert!(streamed.tile_vae_decode);
        assert!(streamed.chunk_attention);
        assert!(streamed.stream_transformer_blocks);
        assert_eq!(
            streamed.transformer_window_component,
            Some(TransformerComponent::Dit)
        );
    }

    #[test]
    fn unpublished_parameters_are_rejected_instead_of_silently_coerced() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = weights_free_memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        let mut decode = selection(MemoryStrategy::BoundedDecode);
        assert!(
            contract.validate_selection(&decode).is_ok(),
            "{:?}",
            contract.validate_selection(&decode)
        );
        decode.parameters.decode_overlap = Some(REJECTED_SUB_512_OVERLAP);
        assert!(contract.validate_selection(&decode).is_err());

        let mut attention = selection(MemoryStrategy::BoundedAttention);
        assert!(
            contract.validate_selection(&attention).is_ok(),
            "{:?}",
            contract.validate_selection(&attention)
        );
        attention.parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE / 2);
        assert!(contract.validate_selection(&attention).is_err());
    }

    #[test]
    fn eager_and_resident_loads_do_not_advertise_rung_four() {
        let tmp = tempfile::tempdir().unwrap();
        let deferred_contract =
            weights_free_memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        let mut eager = spec(&tmp);
        eager.load_shape = LoadShape::EagerMaterialization;
        let eager_contract = weights_free_memory_strategy_contract("qwen_image", &eager).unwrap();
        assert_eq!(
            deferred_contract.calibration.as_ref().unwrap().fingerprint,
            eager_contract.calibration.as_ref().unwrap().fingerprint
        );
        assert_ne!(
            deferred_contract.calibration.as_ref().unwrap().load_shape,
            eager_contract.calibration.as_ref().unwrap().load_shape
        );
        let mut resident = spec(&tmp);
        resident.offload_policy = OffloadPolicy::Resident;
        for spec in [eager, resident] {
            let contract = weights_free_memory_strategy_contract("qwen_image", &spec).unwrap();
            assert!(matches!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .map(|c| &c.support),
                Some(MemoryStrategySupport::Missing)
            ));
            assert!(!contract.lifecycle.transformer_window_materialization);
        }
    }

    #[test]
    fn dense_load_time_quantization_does_not_advertise_rung_four() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"dtype":"bfloat16"}"#,
        )
        .unwrap();
        let dense_q8 = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(mlx_gen::Quant::Q8)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = weights_free_memory_strategy_contract("qwen_image", &dense_q8).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "per-window dense-to-Q8 conversion is not a device-format transfer"
        );

        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8}}"#,
        )
        .unwrap();
        let contract = weights_free_memory_strategy_contract("qwen_image", &dense_q8).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
    }

    #[test]
    fn single_file_source_is_rejected_by_the_family_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let mut file = spec(&tmp);
        file.weights = WeightsSource::File("/nonexistent/qwen.safetensors".into());
        assert!(memory_strategy_contract("qwen_image", &file).is_err());
    }

    #[test]
    fn rung_four_rejects_an_unpublished_window() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = weights_free_memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        let mut selection = selection(MemoryStrategy::BoundedTransformerResidency);
        assert!(
            contract.validate_selection(&selection).is_ok(),
            "{:?}",
            contract.validate_selection(&selection)
        );
        selection.parameters.transformer_window_size = Some(2);
        assert!(contract.validate_selection(&selection).is_err());
    }
}
