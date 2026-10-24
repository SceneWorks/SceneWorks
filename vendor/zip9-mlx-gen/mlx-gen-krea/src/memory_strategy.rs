//! Shared memory-strategy contract for the MLX Krea pose-control provider.
//!
//! This module declares only the quality-preserving mechanisms the provider actually executes.
//! Exact peak envelopes remain in SceneWorks' promoted calibration bundle.
//!
//! SC-15517's real q4 1024²/1-step pose-control A/B held staged residency plus the verified 512/64
//! decode in both arms. Adding 64 Mi-score attention to the resident seven-block pose branch and both
//! attention/windowing to the reopenable 28-block base DiT reduced request peak from 15.574 GiB to
//! 9.200 GiB (40.9%) with zero pixel delta. The overlay remains explicitly resident in
//! `resident_components`; only the base DiT advertises `TransformerComponent::Dit` windowing.
//!
//! # Declaration vs measurement (sc-18451)
//!
//! The two axes are independent here exactly as they are for the Lens and SD3.5 families:
//!
//! * **Support** — the rungs the pose-control engine can execute for this `(provider, LoadSpec)`.
//!   Derived from the loader's own predicates (`streamable_base_transformer` is the contract-side
//!   spelling of `crate::model::resolve_load_plan` plus the residency/materialization gates), never
//!   from the measured-route key.
//! * **Calibration identity** — whether a *measurement* backs the route. SC-15517 measured **q4 and
//!   nothing else** (SceneWorks' `krea_2_turbo_control` calibration plan is q4-only), so
//!   [`MEMORY_CALIBRATION_FINGERPRINT`] attaches to the q4 snapshot composition alone. The bf16 and
//!   q8 bases, the imported single-file DiT, and the additive Wan terminal decoder publish **no**
//!   calibration, which [`mlx_gen::gen_core::standard_memory_strategy_safety_check`] admits only
//!   under an explicit [`mlx_gen::gen_core::MemoryOptimizationAuthority::Estimated`] authority.
//!
//! Before sc-18451 the measured key was stamped unconditionally — on every tier, on the imported
//! file route, on the unmeasured Wan composite, and on the weights-free registry surface — so an
//! admission handshake against an unmeasured route succeeded under `Calibrated` authority. The
//! weights-free declaration surface now substitutes the source-owned
//! [`STATIC_BEHAVIOR_FINGERPRINT`], which is never a measurement and never appears on a contract
//! built from a real load.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
#[cfg(test)]
use mlx_gen::gen_core::MemoryGeometry;
use mlx_gen::gen_core::{
    Error as CoreError, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges,
    MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemoryRuntimeSemantics, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "krea-control-mlx-full-ladder-512-64-attn64m-window1-2026-08-03-v2";
/// Static, weights-free identity for the registry declaration walk. Never a production calibration.
///
/// A contract built from a real load leaves an unmeasured route's calibration `None` so admission has
/// to name an explicit estimate authority. The weights-free surface cannot do that: the shared
/// conformance walk builds its run context through
/// [`mlx_gen::gen_core::standard_memory_behavior_context`], which requires *some* identity. This
/// constant supplies one whose value is structural, keyed per resolved tier and residency policy so a
/// context assembled for one selector can never hand its handshake to another.
pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "krea-control-mlx-registry-behavior-v1";

/// The exact composition SC-15517 measured, and nothing else.
///
/// The evidence is a **q4** snapshot artifact (`SceneWorks/krea-2-turbo-mlx@d009674…:q4` plus the
/// pose overlay) decoded by the native Qwen VAE. Three neighbouring routes are deliberately excluded:
/// * an unmeasured **bf16 or q8** base — the pose branch even packs to a different tier there
///   (`crate::memory::control_branch_quant_bits`), so the envelope is not the measured one;
/// * the **imported single-file** DiT, whose dequantized-to-bf16 residency is a different load source
///   and has no promoted cell (the evidence matrix has no load-source axis);
/// * the additive **Wan terminal decoder**, an unmeasured composite with the base and control branch.
///
/// Each of those publishes no calibration at all rather than a fabricated identity, which is what
/// forces conservative asset-facts + headroom estimation instead of a handshake that silently passes.
///
/// **Low-rank adapters are an accepted axis of this identity, not an exclusion.** A q4 base carrying
/// a Raw-trained LoRA/LoKr keeps the measured key even though SC-15517's arms ran without one. That
/// is deliberate and matches the rest of the family: the base ladder's own
/// `crate::block_memory_strategy` fingerprint is likewise adapter-independent (its sc-16352 table
/// measured the q4 LoRA and LoKr arms against the same key), and SceneWorks selects a promoted record
/// by provider, tier, mode, overlay and geometry — never by adapter set — so an adapter-keyed split
/// here would strand every adapter render on an estimate for no evidentiary gain. The adapter's own
/// residency cost is already priced structurally, and a dense `.diff` patch is excluded from rung 4
/// separately because it mutates the resident base rather than because of the calibration identity.
fn measured_calibration(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<Option<MemoryCalibrationIdentity>> {
    if matches!(spec.weights, mlx_gen::WeightsSource::File(_))
        || spec.components.contains_key(mlx_gen::VAE_COMPONENT)
    {
        return Ok(None);
    }
    // The same seam `registered_safety_check` reads for the loaded tier, so the declared calibration
    // and the admitted tier cannot disagree: a prepacked q4 turnkey and a dense snapshot packed at
    // load both resolve to `Quant::Q4` here, and nothing else does.
    let measured =
        crate::model::effective_base_quant_tier(spec, provider_id)? == Some(mlx_gen::Quant::Q4);
    Ok(measured
        .then(|| MemoryCalibrationIdentity::new(MEMORY_CALIBRATION_FINGERPRINT, spec.load_shape)))
}

/// Per-selector static behavior identity for the weights-free declaration surface.
///
/// `MemoryProviderContract::conformance_errors` requires lowercase kebab tokens, so every component
/// spelled into the identity is already one.
fn static_behavior_identity(
    tier: &str,
    offload_policy: mlx_gen::OffloadPolicy,
    load_shape: mlx_gen::gen_core::LoadShape,
) -> MemoryCalibrationIdentity {
    let policy = match offload_policy {
        mlx_gen::OffloadPolicy::Resident => "resident",
        mlx_gen::OffloadPolicy::Sequential => "sequential",
    };
    MemoryCalibrationIdentity::new(
        format!("{STATIC_BEHAVIOR_FINGERPRINT}-{tier}-{policy}"),
        load_shape,
    )
}

/// The already-resolved artifact tier named by a registry surface selector.
fn selector_tier_token(tier: mlx_gen::gen_core::MemoryContractSurfaceTier) -> &'static str {
    match tier {
        mlx_gen::gen_core::MemoryContractSurfaceTier::Bf16 => "bf16",
        mlx_gen::gen_core::MemoryContractSurfaceTier::Q4 => "q4",
        mlx_gen::gen_core::MemoryContractSurfaceTier::Q8 => "q8",
        mlx_gen::gen_core::MemoryContractSurfaceTier::Nvfp4 => "nvfp4",
    }
}

/// The tier a bare weights-free `LoadSpec` names, for the fixture seam that has no selector.
///
/// Deliberately the same vocabulary [`selector_tier_token`] produces, so a dense bf16 witness gets
/// one identity whichever seam built it. The resolver refuses a non-bf16 activation precision, so
/// only this seam can ever see `fp32`, and it must not collapse onto the dense bf16 token.
fn spec_tier_token(spec: &LoadSpec) -> &'static str {
    match (spec.precision, spec.quantize) {
        (mlx_gen::Precision::Fp32, _) => "fp32",
        (_, None) => "bf16",
        (_, Some(mlx_gen::Quant::Q4)) => "q4",
        (_, Some(mlx_gen::Quant::Q8)) => "q8",
        (_, Some(mlx_gen::Quant::Nvfp4)) => "nvfp4",
    }
}
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 64;
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
/// Exact tile-edge domain admitted by current real-weight evidence. The 384 px candidate is
/// deliberately excluded: the clean 1024² sc-16099 run exceeded the established diffusion-latent
/// maximum-error threshold, so it must not inherit the 512 px calibration.
pub const DECODE_TILE_EDGES: [u32; 1] = [DECODE_TILE_EDGE];

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    if matches!(spec.weights, mlx_gen::WeightsSource::File(_)) {
        crate::model_control::validate_control_spec(spec)
            .map_err(|error| CoreError::Msg(error.to_string()))?;
        let base = mlx_gen::require_base_snapshot(spec, provider_id)?;
        return native_memory_strategy_contract_from_spec(provider_id, spec, base, false);
    }
    let (asset_facts, resident_components) = asset_facts(spec, provider_id)?;
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        asset_facts,
        resident_components,
        streamable_base_transformer(spec, provider_id)?,
        // A production contract never fabricates an identity: an unmeasured tier stays uncalibrated
        // so admission has to name an explicit estimate authority for it.
        measured_calibration(provider_id, spec)?,
    )
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
///
/// This is the fixture seam, reached with a caller-supplied witness `LoadSpec` rather than a registry
/// surface selector, so its behavior identity is keyed on that spec's own tier and residency axes.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        MemoryAssetFacts::default(),
        Vec::new(),
        streamable_base_transformer(spec, provider_id)?,
        Some(static_behavior_identity(
            spec_tier_token(spec),
            spec.offload_policy,
            spec.load_shape,
        )),
    )
}

/// Resolve the pose-control catalog surface from its explicit already-packed artifact tier.
///
/// The control route's weights-free witness is a nonexistent snapshot path, so the production
/// [`streamable_base_transformer`] predicate reads `packed_quant_bits` there, finds no marker, and
/// concludes that a Q4/Q8 witness must be packed *at load* — which withdraws rung 4 from exactly the
/// two prepacked tiers a shipped turnkey provides, and from q4 in particular, the only tier this
/// route's measured evidence covers (sc-15517). The selector names the already-resolved artifact
/// tier, so this resolver publishes the source-derived load eligibility from that instead, exactly as
/// [`crate::block_memory_strategy::weights_free_memory_strategy_surface_contract`] does for the four
/// base routes. It deliberately retains zero asset facts; proving the selected snapshot marker and
/// tensor inventory remains production contract construction's job.
pub(crate) fn weights_free_memory_strategy_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<MemoryProviderContract> {
    if provider_id != crate::KREA_2_TURBO_CONTROL_ID {
        return Err(CoreError::Unsupported(format!(
            "unknown Krea pose-control memory provider {provider_id}"
        )));
    }
    crate::block_memory_strategy::surface_selector_matches_spec(surface)?;
    crate::model_control::validate_control_load_axes(&surface.spec)
        .map_err(|error| CoreError::Msg(error.to_string()))?;
    let streamable = matches!(
        surface.resolved_artifact_tier(),
        mlx_gen::gen_core::MemoryContractSurfaceTier::Bf16
            | mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
            | mlx_gen::gen_core::MemoryContractSurfaceTier::Q8
    ) && matches!(
        surface.spec.offload_policy,
        mlx_gen::OffloadPolicy::Sequential
    ) && matches!(
        surface.spec.load_shape,
        mlx_gen::gen_core::LoadShape::DeferredMaterialization
    ) && !crate::model::adapters_have_diff_patch_for_spec(&surface.spec)?;
    memory_strategy_contract_with_asset_facts(
        provider_id,
        &surface.spec,
        MemoryAssetFacts::default(),
        Vec::new(),
        streamable,
        Some(static_behavior_identity(
            selector_tier_token(surface.resolved_artifact_tier()),
            surface.selector.offload_policy,
            surface.selector.load_shape,
        )),
    )
}

fn memory_strategy_contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    asset_facts: MemoryAssetFacts,
    resident_components: Vec<MemoryResidentComponent>,
    streamable_transformer: bool,
    calibration: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
    let staged_residency = matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential);
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
    Ok(MemoryProviderContract {
        phase_facts: Some(mlx_gen::gen_core::MemoryPhaseFacts::staged(
            mlx_gen::gen_core::StagedWeightSchedule::TwoStage,
        )),
        architecture_facts: crate::block_memory_strategy::architecture_facts(spec),
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::Resident
                    | MemoryStrategy::BoundedDecode
                    | MemoryStrategy::BoundedAttention => MemoryStrategySupport::Implemented,
                    MemoryStrategy::BoundedTransformerResidency if streamable_transformer => {
                        MemoryStrategySupport::Implemented
                    }
                    MemoryStrategy::StagedResidency if staged_residency => {
                        MemoryStrategySupport::Implemented
                    }
                    MemoryStrategy::StagedResidency
                    | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
                },
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: routes.native_edges().to_vec(),
                        decode_overlaps: vec![DECODE_OVERLAP],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                        attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency if streamable_transformer => {
                        MemoryParameterRanges {
                            transformer_window_sizes: vec![TRANSFORMER_WINDOW_SIZE],
                            transformer_window_components: vec![TransformerComponent::Dit],
                            ..Default::default()
                        }
                    }
                    _ => MemoryParameterRanges::default(),
                },
            })
            .collect(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: streamable_transformer
            .then_some((
                MemoryStrategy::BoundedTransformerResidency,
                mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            ))
            .into_iter()
            .collect(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: staged_residency,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: streamable_transformer,
        },
        formula: if resident_components.is_empty() {
            MemoryFormulaKind::PhaseEnvelope { phases, variables }
        } else {
            MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables,
                resident_components,
            }
        },
        calibration,
        asset_facts,
        runtime: MemoryRuntimeSemantics::default(),
    })
}

/// Exact contract for the community single-file DiT **pose-control** composition (the control twin of
/// `crate::block_memory_strategy::native_memory_strategy_contract`): the imported DiT is read from
/// `dit_file` (its native I8 projections are dequantized to bf16 at load, scale/descriptor tensors
/// consumed and dropped), the text encoder and VAE come from the resident base tier, and the pose
/// overlay loads dense — the native DiT is dense in memory, and a dense base carries a dense branch
/// (`crate::memory::control_branch_quant_bits(None)`), so the overlay bytes are its stored bytes.
/// Same `provider_id` as the snapshot control contract — the implementation, phase model, and
/// non-transformer components are the same, so promoted evidence is never orphaned onto a bespoke
/// "imported" provider identity. It carries **no calibration identity**, though: the evidence matrix
/// has no load-source axis, so a `Dir`-measured cell must not be reported as a `File` measurement
/// until this re-openable path is measured directly (sc-18451 — before it, the measured fingerprint
/// was stamped here and the handshake passed under `Calibrated` authority). That withholding is now
/// expressed twice, consistently: no calibration, and `streamable = false` for the registry contract.
/// The loader's reopenable mechanism stays available for its source smoke.
pub(crate) fn native_memory_strategy_contract_from_spec(
    provider_id: &str,
    spec: &LoadSpec,
    base_snapshot_dir: &std::path::Path,
    streamable: bool,
) -> CoreResult<MemoryProviderContract> {
    let dit_file = match &spec.weights {
        mlx_gen::WeightsSource::File(path) => path,
        mlx_gen::WeightsSource::Dir(path) => {
            return Err(CoreError::Msg(format!(
                "{provider_id}: native pose-control facts require a single-file DiT, not {}",
                path.display()
            )))
        }
    };
    let control = mlx_gen::require_control(spec, provider_id, "Krea 2 pose control overlay")?;
    let stored = |path: &std::path::Path, what: &str| -> CoreResult<u64> {
        projected_safetensors_bytes(path, |_| ResidentProjection::Stored).map_err(|error| {
            CoreError::Msg(format!(
                "{provider_id}: native {what} asset facts for '{}': {error}",
                path.display()
            ))
        })
    };
    let alternate_decoder_bytes = match spec.components.get(mlx_gen::VAE_COMPONENT) {
        Some(mlx_gen::WeightsSource::Dir(path)) => stored(path, "alternate Wan decoder")?,
        Some(mlx_gen::WeightsSource::File(path)) => {
            spec.read_file_unchanged_if_prepared(path, |p| stored(p, "alternate Wan decoder"))?
        }
        None => 0,
    };
    let selected_text_encoder =
        crate::model::runtime_encoder_contract().source_for_load(spec, base_snapshot_dir)?;
    let expected_language_bits =
        crate::model::native_text_encoder_expected_quant_bits(base_snapshot_dir)?;
    let conditioning_bytes = crate::model::selected_language_resident_bytes(
        &selected_text_encoder,
        expected_language_bits,
        provider_id,
    )?;
    let decoder_bytes =
        stored(&base_snapshot_dir.join("vae"), "base VAE")?.saturating_add(alternate_decoder_bytes);
    // Same declared logical shapes the load will use (sc-20644) — see
    // `block_memory_strategy::base_architecture_config`.
    let base_cfg =
        crate::block_memory_strategy::base_architecture_config(provider_id, base_snapshot_dir)?;
    let transformer_bytes = spec.read_file_unchanged_if_prepared(dit_file, |p| {
        crate::block_memory_strategy::native_dit_transformer_bytes(
            provider_id,
            p,
            spec.quantize,
            crate::native_remap::DeclaredLogicalShapes::from_base(base_cfg.as_ref()),
        )
    })?;
    let branch_bits =
        crate::memory::control_branch_quant_bits(spec.quantize.map(mlx_gen::Quant::bits));
    let projected_control = |path: &std::path::Path| {
        projected_safetensors_bytes(path, |tensor| {
            if let Some(bits) =
                branch_bits.filter(|_| crate::control::is_control_quant_target(&tensor.name))
            {
                ResidentProjection::GroupQuantized {
                    bits,
                    group_size: crate::quant::GROUP_SIZE as usize,
                }
            } else {
                ResidentProjection::Stored
            }
        })
        .map_err(|error| {
            CoreError::Msg(format!(
                "{provider_id}: native pose control overlay asset facts for '{}': {error}",
                path.display()
            ))
        })
    };
    let (control_path, overlay_bytes) = match control {
        mlx_gen::WeightsSource::Dir(path) => (path, projected_control(path)?),
        mlx_gen::WeightsSource::File(path) => (
            path,
            spec.read_file_unchanged_if_prepared(path, projected_control)?,
        ),
    };
    if overlay_bytes == 0 {
        return Err(CoreError::Msg(format!(
            "{provider_id}: pose control overlay '{}' contains no tensor bytes",
            control_path.display()
        )));
    }
    let resident_components = vec![MemoryResidentComponent {
        id: "pose_control_branch".to_owned(),
        kind: MemoryComponentKind::ControlBranch,
        resident_bytes: overlay_bytes,
        bounded_by: None,
        residency: MemoryComponentResidency::WholeRender,
    }];
    let asset_facts = MemoryAssetFacts {
        base_bytes: conditioning_bytes
            .saturating_add(transformer_bytes)
            .saturating_add(decoder_bytes),
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlay_bytes,
    };
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        asset_facts,
        resident_components,
        streamable,
        // The imported single-file route has no promoted measurement; see the doc comment above.
        None,
    )
}

/// Is an **authorized** calibration fault armed for `phase` on this request-scoped memory selection?
///
/// The shared request floor (`GenerationMemory::authorize_calibration_fault`) sets the pair; a
/// production selector leaves both controls unset, so this is a `false` for every ordinary render.
/// Callers use it to arm the extra materialization a fault boundary needs without perturbing an
/// ordinary render's laziness.
pub(crate) fn calibration_fault_armed(
    memory: Option<mlx_gen::gen_core::GenerationMemory>,
    phase: MemoryPhase,
) -> bool {
    memory.is_some_and(|memory| {
        memory.calibration_fault_harness_authorized && memory.calibration_error_phase == Some(phase)
    })
}

/// Request-local conformance fault at a physical phase boundary (SC-15449, sc-22738), keyed on a
/// request-scoped memory selection. Mirrors the FLUX/Chroma/Z-Image spelling so a lifecycle
/// certification sees the same error at the same boundaries on every MLX family.
///
/// # The boundary is the phase's EXIT, not its entry
///
/// Krea follows the FLUX/Chroma/Qwen convention: every call site raises the fault **after** the
/// phase's work has run and its produced tensor has been evaluated — the conditioning fault after the
/// text encode materializes the context and before any heavy component loads
/// (`mlx-gen-chroma/src/model.rs`, the `eval` of the embeds inside the text-phase closure), the
/// denoise fault after the sampler with an `eval` of the latent (chroma's `eval([&final_latents])`
/// then its `Denoise` fault), the decode fault after the VAE with the decoded output materialized
/// (`mlx-gen-flux/src/model.rs`, which force-evals its lazy decode before faulting).
///
/// That is not cosmetic. The SceneWorks adapter certifies an anchor by injecting an authorized fault
/// here and then reading the engine's retained bytes against its `allows_retained` bound, so the
/// phase's real allocation has to exist when the error is returned. An entry-convention fault (the
/// anima spelling) would hand the adapter the *previous* phase's heap and certify nothing. See
/// `denoise_exit_fault` / `decode_exit_fault` in `pipeline.rs` for the two render-side sites.
pub(crate) fn calibration_fault_for_memory(
    memory: Option<mlx_gen::gen_core::GenerationMemory>,
    phase: MemoryPhase,
    provider_id: &str,
) -> mlx_gen::Result<()> {
    if calibration_fault_armed(memory, phase) {
        return Err(mlx_gen::Error::Msg(format!(
            "{provider_id}: injected memory-strategy calibration error at {phase:?}"
        )));
    }
    Ok(())
}

/// [`calibration_fault_for_memory`] against a whole request — the spelling the model layer uses.
pub(crate) fn calibration_fault(
    req: &mlx_gen::GenerationRequest,
    phase: MemoryPhase,
    provider_id: &str,
) -> mlx_gen::Result<()> {
    calibration_fault_for_memory(req.memory, phase, provider_id)
}

/// Compatibility shim for the former bespoke imported-control entrypoint.
#[cfg(test)]
pub(crate) fn native_memory_strategy_contract(
    provider_id: &str,
    dit_file: &std::path::Path,
    base_snapshot_dir: &std::path::Path,
    control: &std::path::Path,
) -> CoreResult<MemoryProviderContract> {
    let spec = LoadSpec::new(mlx_gen::WeightsSource::File(dit_file.to_path_buf()))
        .with_component(
            mlx_gen::BASE_SNAPSHOT_COMPONENT,
            mlx_gen::WeightsSource::Dir(base_snapshot_dir.to_path_buf()),
        )
        .with_control(mlx_gen::WeightsSource::File(control.to_path_buf()));
    native_memory_strategy_contract_from_spec(provider_id, &spec, base_snapshot_dir, false)
}

fn streamable_base_transformer(spec: &LoadSpec, provider_id: &str) -> CoreResult<bool> {
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return Ok(false);
    };
    let plan = crate::model::resolve_load_plan(spec, root, provider_id)?;
    Ok(
        matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential)
            && matches!(
                spec.load_shape,
                mlx_gen::gen_core::LoadShape::DeferredMaterialization
            )
            && !crate::model::adapters_have_diff_patch_for_spec(spec)?
            && plan.load_time_quant_bits.is_none(),
    )
}

fn asset_facts(
    spec: &LoadSpec,
    provider_id: &str,
) -> CoreResult<(MemoryAssetFacts, Vec<MemoryResidentComponent>)> {
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "krea pose memory facts require a snapshot directory".to_owned(),
        ));
    };
    let project = |path: &std::path::Path, select: &dyn Fn(&str) -> bool| -> CoreResult<u64> {
        projected_safetensors_bytes(path, |tensor| {
            if let Some(quant) = spec.quantize.filter(|_| select(&tensor.name)) {
                ResidentProjection::GroupQuantized {
                    bits: quant.bits(),
                    group_size: crate::quant::GROUP_SIZE as usize,
                }
            } else {
                ResidentProjection::Stored
            }
        })
    };
    let selected_text_encoder =
        crate::model::runtime_encoder_contract().source_for_load(spec, root)?;
    let expected_language_bits = crate::model::effective_base_quant_bits(spec, root, provider_id)?;
    let conditioning_bytes = crate::model::selected_language_resident_bytes(
        &selected_text_encoder,
        expected_language_bits,
        provider_id,
    )?;
    let transformer_bytes = project(&root.join("transformer"), &|name| {
        crate::convert::is_transformer_quant_target(name)
    })?;
    let alternate_decoder_bytes = match spec.components.get(mlx_gen::VAE_COMPONENT) {
        Some(mlx_gen::WeightsSource::Dir(path)) => {
            projected_safetensors_bytes(path, |_| ResidentProjection::Stored)?
        }
        Some(mlx_gen::WeightsSource::File(path)) => spec
            .read_file_unchanged_if_prepared(path, |p| {
                projected_safetensors_bytes(p, |_| ResidentProjection::Stored)
            })?,
        None => 0,
    };
    let decoder_bytes =
        project(&root.join("vae"), &|_| false)?.saturating_add(alternate_decoder_bytes);
    let overlay_bytes = match &spec.control {
        Some(mlx_gen::WeightsSource::Dir(path)) => {
            let base_bits = crate::model::effective_base_quant_bits(spec, root, provider_id)?;
            let branch_bits = crate::memory::control_branch_quant_bits(base_bits);
            projected_safetensors_bytes(path, |_| match branch_bits {
                Some(bits) => ResidentProjection::GroupQuantized {
                    bits,
                    group_size: crate::quant::GROUP_SIZE as usize,
                },
                None => ResidentProjection::Stored,
            })?
        }
        Some(mlx_gen::WeightsSource::File(path)) => {
            let base_bits = crate::model::effective_base_quant_bits(spec, root, provider_id)?;
            let branch_bits = crate::memory::control_branch_quant_bits(base_bits);
            spec.read_file_unchanged_if_prepared(path, |p| {
                projected_safetensors_bytes(p, |_| match branch_bits {
                    Some(bits) => ResidentProjection::GroupQuantized {
                        bits,
                        group_size: crate::quant::GROUP_SIZE as usize,
                    },
                    None => ResidentProjection::Stored,
                })
            })?
        }
        None => 0,
    };
    let resident_components = (overlay_bytes > 0)
        .then(|| MemoryResidentComponent {
            id: "pose_control_branch".to_owned(),
            kind: MemoryComponentKind::ControlBranch,
            resident_bytes: overlay_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        })
        .into_iter()
        .collect();
    Ok((
        MemoryAssetFacts {
            base_bytes: conditioning_bytes
                .saturating_add(transformer_bytes)
                .saturating_add(decoder_bytes),
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes,
        },
        resident_components,
    ))
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    precision: mlx_gen::Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        // The Krea pose-control composition deliberately has no PiD decoder (its heavy bundle is the
        // base VAE plus the pose branch). Reject the flag explicitly instead of letting the residency
        // seam ignore it and execute a native decode under a PiD-labelled request.
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: PiD decode is not implemented for pose control",
                contract.provider_id
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    false,
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

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match crate::model::effective_base_quant_tier(spec, &contract.provider_id) {
        Ok(quant) => safety_check(contract, spec.precision, quant, context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let quant = crate::model::effective_base_quant_tier(spec, &contract.provider_id)?;
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::ImageToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some("pose-control".to_owned()),
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

pub fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        spec.precision,
        crate::model::effective_base_quant_tier(spec, provider_id)?,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: mlx_gen::Precision,
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
    precision: mlx_gen::Precision,
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
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::Krea2Config::turbo().num_layers,
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

    /// Every phase: the authorized pair refuses at exactly its own boundary and nowhere else.
    #[test]
    fn an_authorized_calibration_fault_refuses_at_exactly_its_named_phase() {
        for named in [
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ] {
            let mut memory = mlx_gen::gen_core::GenerationMemory::default();
            memory.authorize_calibration_fault(named);
            for phase in [
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ] {
                let result =
                    calibration_fault_for_memory(Some(memory), phase, crate::KREA_2_RAW_ID);
                if phase == named {
                    let error = result.expect_err("the named phase must refuse").to_string();
                    assert!(error.contains(&format!("{phase:?}")), "{error}");
                    assert!(error.contains(crate::KREA_2_RAW_ID), "{error}");
                } else {
                    assert!(result.is_ok(), "{phase:?} must not fire for {named:?}");
                }
                assert_eq!(calibration_fault_armed(Some(memory), phase), phase == named);
            }
        }
    }

    /// A phase selection WITHOUT the harness authorization is inert: an ordinary render proceeds.
    #[test]
    fn an_unauthorized_phase_selection_never_refuses_a_render() {
        for phase in [
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ] {
            let memory = mlx_gen::gen_core::GenerationMemory {
                calibration_error_phase: Some(phase),
                calibration_fault_harness_authorized: false,
                ..Default::default()
            };
            assert!(!calibration_fault_armed(Some(memory), phase));
            assert!(
                calibration_fault_for_memory(Some(memory), phase, crate::KREA_2_RAW_ID).is_ok(),
                "{phase:?} must render normally without harness authorization"
            );
        }
        assert!(
            calibration_fault_for_memory(None, MemoryPhase::Decode, crate::KREA_2_RAW_ID).is_ok()
        );
    }

    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemorySelection, MemoryStrategyParameters,
        Precision, Quant, WeightsSource,
    };

    /// A native-keyed imported DiT fixture (one dense bf16 tensor, 256 payload bytes) -- the plan
    /// compiler (sc-20385) refuses foreign keys, so the imported-file fixtures must carry a real
    /// Krea native key.
    fn write_native_dit(path: &std::path::Path) {
        let mut header = br#"{"model.diffusion_model.first.weight":{"dtype":"BF16","shape":[2,64],"data_offsets":[0,256]}}"#
            .to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 256]);
        std::fs::write(path, bytes).unwrap();
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

    fn write_snapshot(root: &std::path::Path) {
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_control(&dir.join("model.safetensors"));
        }
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::model::test_encoder_contract(),
        )
        .expect("validation-complete text encoder fixture");
    }

    #[test]
    fn native_control_contract_keeps_the_builtin_provider_identity_and_components() {
        // The imported single-file pose-control contract must resolve through the SAME provider
        // identity as the snapshot lane: same provider_id, same resident pose-branch component —
        // never a bespoke "imported" identity that would orphan promoted evidence. It publishes no
        // calibration, because the evidence matrix has no load-source axis and this route has never
        // been measured (sc-18451). Resident-only: no staged residency, no windowing.
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let dit = root.join("imported-dit.safetensors");
        write_native_dit(&dit);
        let overlay = root.join("control.safetensors");
        write_control(&overlay);

        let contract =
            native_memory_strategy_contract("krea_2_turbo_control", &dit, &root, &overlay).unwrap();
        assert_eq!(contract.provider_id, "krea_2_turbo_control");
        assert_eq!(
            contract.calibration, None,
            "an unmeasured load source must not inherit the Dir-measured identity"
        );
        let components = contract.resident_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].id, "pose_control_branch");
        assert_eq!(components[0].kind, MemoryComponentKind::ControlBranch);
        assert!(components[0].resident_bytes > 0);
        // The transformer term is the native FILE's projection, not the base tier's transformer dir;
        // the fixture stores one dense bf16 tensor, so projected == stored bytes.
        assert_eq!(contract.asset_facts.transformer_bytes, 256);
        assert_eq!(
            contract.asset_facts.base_bytes,
            contract.asset_facts.conditioning_bytes
                + contract.asset_facts.transformer_bytes
                + contract.asset_facts.decoder_bytes
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        // A missing / zero-byte overlay is refused loudly — the branch is a REQUIRED component.
        let missing = native_memory_strategy_contract(
            "krea_2_turbo_control",
            &dit,
            &root,
            &root.join("missing-control.safetensors"),
        )
        .expect_err("missing overlay must fail")
        .to_string();
        assert!(missing.contains("pose control overlay"), "got: {missing}");
    }

    #[test]
    fn alternate_decoder_is_additive_to_pose_control_decoder_facts() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("base");
        write_snapshot(&root);
        let overlay = tmp.path().join("control.safetensors");
        write_control(&overlay);
        let donor = tmp.path().join("wan-vae.safetensors");
        write_control(&donor);
        // Q4 is the tier SC-15517 measured, so this pair isolates the decoder axis: the ONLY
        // difference between the two contracts is the additive Wan terminal decoder.
        let spec = LoadSpec::new(WeightsSource::Dir(root))
            .with_quant(Quant::Q4)
            .with_control(WeightsSource::File(overlay));
        let native = memory_strategy_contract("krea_2_turbo_control", &spec).unwrap();
        let composite = memory_strategy_contract(
            "krea_2_turbo_control",
            &spec
                .clone()
                .with_component(mlx_gen::VAE_COMPONENT, WeightsSource::File(donor)),
        )
        .unwrap();
        assert_eq!(
            composite.asset_facts.decoder_bytes,
            native.asset_facts.decoder_bytes + 256
        );
        assert_eq!(
            composite.asset_facts.base_bytes,
            native.asset_facts.base_bytes + 256
        );
        assert_eq!(
            composite.calibration, None,
            "native control measurements must not authorize the composite decoder path"
        );
        assert_eq!(
            native.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        // The measured identity is q4-only: the same composition on a dense base is a different,
        // unmeasured envelope (the pose branch alone packs to a different tier there).
        let mut dense = spec;
        dense.quantize = None;
        assert_eq!(
            memory_strategy_contract("krea_2_turbo_control", &dense)
                .unwrap()
                .calibration,
            None
        );
    }

    #[test]
    fn registry_file_control_contract_uses_file_bytes_but_withholds_rung_four() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let dit = root.join("imported-dit.safetensors");
        write_native_dit(&dit);
        let overlay = root.join("control.safetensors");
        write_control(&overlay);
        let spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(mlx_gen::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(root))
            .with_control(WeightsSource::File(overlay))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);

        let contract = memory_strategy_contract("krea_2_turbo_control", &spec).unwrap();
        assert_eq!(contract.provider_id, "krea_2_turbo_control");
        assert_eq!(contract.asset_facts.transformer_bytes, 256);
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "the pinned File mechanism needs source-specific evidence before authorization"
        );
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn staged_residency_and_synchronized_release_require_a_sequential_control_load() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);

        let resident_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let resident = memory_strategy_contract("krea_2_turbo_control", &resident_spec).unwrap();
        assert_eq!(
            resident
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(!resident.lifecycle.synchronized_phase_release);
        assert_eq!(
            resident
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented,
            "bounded decode is independent of the component load policy"
        );

        let sequential_spec = resident_spec.with_offload_policy(mlx_gen::OffloadPolicy::Sequential);
        let sequential =
            memory_strategy_contract("krea_2_turbo_control", &sequential_spec).unwrap();
        assert_eq!(
            sequential
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert!(sequential.lifecycle.synchronized_phase_release);
        assert_eq!(
            sequential
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );

        let streamable_spec =
            sequential_spec.with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);
        let streamable =
            memory_strategy_contract("krea_2_turbo_control", &streamable_spec).unwrap();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                streamable.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?} must be executable on the deferred control composition"
            );
        }
        assert_eq!(
            streamable
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_components,
            vec![TransformerComponent::Dit],
            "the seven-block pose overlay stays explicitly resident; the reopenable base DiT is windowed"
        );
        assert_eq!(
            streamable.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
    }

    #[test]
    fn prepacked_q8_pose_without_an_override_accepts_only_the_actual_tier() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let contract = memory_strategy_contract("krea_2_turbo_control", &spec).unwrap();
        // A prepacked q8 pose base is a real, loadable route with NO promoted measurement: the tier
        // gate below still binds exactly, but nothing hands it the q4 evidence key.
        assert_eq!(contract.calibration, None);
        let context_for = |quant| MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                },
            },
            calibration_abi: mlx_gen::gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: String::new(),
            load_shape: spec.load_shape,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 512,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        };

        assert_eq!(
            registered_safety_check(&spec, &contract, &context_for(Some(Quant::Q8))),
            MemorySafetyDecision::Accept
        );
        for wrong in [None, Some(Quant::Q4)] {
            assert!(matches!(
                registered_safety_check(&spec, &contract, &context_for(wrong)),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("does not match loaded tier")
            ));
        }
    }

    #[test]
    fn q4_base_projects_pose_overlay_at_the_declared_q8_floor() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        let control = root.join("control.safetensors");
        write_control(&control);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(Quant::Q4)
            .with_control(WeightsSource::File(control));
        let contract = memory_strategy_contract("krea_2_turbo_control", &spec).unwrap();
        // Q8: 128 code bytes + two 2x1 bf16 tables (8 bytes). A uniform Q4 projection would be 72.
        assert_eq!(contract.asset_facts.overlay_bytes, 136);
        // The bounded sealed-source receipt retains every compact loaded layer, with projection
        // matrices at Q4 and embeddings/norms dense. The production 35-layer total is asserted from
        // pure production headers in model::tests.
        let headers = gen_core_testkit::encoder_contract_fixture_tensor_headers(
            crate::model::test_encoder_contract(),
            None,
        )
        .unwrap();
        let conditioning_bytes =
            mlx_gen::asset_facts::projected_tensor_headers_bytes(&headers, |tensor| {
                if crate::convert::is_text_encoder_quant_target(&tensor.name) {
                    ResidentProjection::GroupQuantized {
                        bits: 4,
                        group_size: crate::quant::GROUP_SIZE as usize,
                    }
                } else {
                    ResidentProjection::Stored
                }
            })
            .unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, conditioning_bytes);
        assert_eq!(contract.asset_facts.transformer_bytes, 256);
        assert_eq!(contract.asset_facts.decoder_bytes, 256);
        assert_eq!(contract.asset_facts.base_bytes, conditioning_bytes + 512);
        assert_eq!(contract.auxiliary_resident_bytes(), 136);
        assert!(contract.conformance_errors().is_empty());
    }

    #[test]
    fn empty_pose_base_component_cannot_be_reported_as_zero() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::remove_file(root.join("vae/model.safetensors")).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert!(memory_strategy_contract("krea_2_turbo_control", &spec).is_err());
        assert!(weights_free_memory_strategy_contract("krea_2_turbo_control", &spec).is_ok());
    }

    #[test]
    fn imported_pose_overlay_inventory_rejects_missing_empty_and_corrupt_files() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let dit = root.join("imported.safetensors");
        write_native_dit(&dit);
        let overlay = root.join("control.safetensors");
        let spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(
                mlx_gen::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(root.clone()),
            )
            .with_control(WeightsSource::File(overlay.clone()));

        for (case, replacement) in [
            ("empty", Some(Vec::new())),
            ("corrupt", Some(b"corrupt".to_vec())),
            ("missing", None),
        ] {
            match replacement {
                Some(bytes) => std::fs::write(&overlay, bytes).unwrap(),
                None => {
                    if overlay.exists() {
                        std::fs::remove_file(&overlay).unwrap();
                    }
                }
            }
            let error = memory_strategy_contract("krea_2_turbo_control", &spec)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("pose control") || error.contains("safetensors"),
                "{case}: {error}"
            );
            write_control(&overlay);
        }
    }

    #[test]
    fn imported_file_contract_matches_the_control_loader_for_every_typed_field() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let dit = root.join("imported.safetensors");
        write_native_dit(&dit);
        let overlay = root.join("control.safetensors");
        write_control(&overlay);
        let valid = LoadSpec::new(WeightsSource::File(dit))
            .with_component(
                mlx_gen::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(root.clone()),
            )
            .with_control(WeightsSource::File(overlay));

        let mut precision = valid.clone();
        precision.precision = Precision::Fp32;
        let mut extra_control = valid.clone();
        extra_control
            .extra_controls
            .push(WeightsSource::File(root.join("extra-control.safetensors")));
        let mut ip_adapter = valid.clone();
        ip_adapter.ip_adapter = Some(WeightsSource::Dir(root.join("ip-adapter")));
        let mut identity = valid.clone();
        identity.identity = Some(mlx_gen::gen_core::IdentityWeights::default());
        let mut text_encoder = valid.clone();
        let external_text_encoder = root.join("external-text");
        gen_core_testkit::write_encoder_contract_fixture(
            &external_text_encoder,
            crate::model::test_encoder_contract(),
        )
        .expect("validation-complete selected text encoder fixture");
        text_encoder.text_encoder = Some(WeightsSource::Dir(external_text_encoder));
        let mut pid = valid.clone();
        pid.pid = Some(mlx_gen::gen_core::PidWeights {
            checkpoint: WeightsSource::File(root.join("pid.safetensors")),
            gemma: WeightsSource::Dir(root.join("gemma")),
        });
        let mut unknown_component = valid.clone();
        unknown_component.components.insert(
            "unknown".into(),
            WeightsSource::File(root.join("unknown.safetensors")),
        );
        let mut missing_base = valid.clone();
        missing_base.components.clear();
        let accepted_adapter = valid.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            mlx_gen::AdapterKind::Lora,
        )]);
        let accepted_deferred = valid
            .clone()
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);

        for (case, spec, expected) in [
            ("valid", valid.clone(), true),
            ("adapter", accepted_adapter, true),
            ("deferred", accepted_deferred, true),
            ("precision", precision, false),
            ("quantize", valid.clone().with_quant(Quant::Q4), true),
            ("extra_control", extra_control, false),
            ("ip_adapter", ip_adapter, false),
            ("pid", pid, false),
            ("identity", identity, false),
            ("text_encoder", text_encoder, true),
            ("unknown_component", unknown_component, false),
            ("missing_base", missing_base, false),
        ] {
            let loader = crate::model_control::validate_control_spec(&spec).is_ok();
            let contract = memory_strategy_contract("krea_2_turbo_control", &spec).is_ok();
            assert_eq!(loader, expected, "loader validation for {case}");
            assert_eq!(contract, loader, "contract/loader parity for {case}");
        }
    }

    fn control_surface(
        tier: mlx_gen::gen_core::MemoryContractSurfaceTier,
        offload_policy: mlx_gen::OffloadPolicy,
        load_shape: mlx_gen::gen_core::LoadShape,
    ) -> mlx_gen::gen_core::MemoryContractSurfaceSpec {
        mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == tier
                    && surface.selector.offload_policy == offload_policy
                    && surface.selector.load_shape == load_shape
            })
            .expect("the MLX surface set publishes every tier x policy x shape selector")
    }

    fn prepacked_q4_surface() -> mlx_gen::gen_core::MemoryContractSurfaceSpec {
        control_surface(
            mlx_gen::gen_core::MemoryContractSurfaceTier::Q4,
            mlx_gen::OffloadPolicy::Sequential,
            mlx_gen::gen_core::LoadShape::DeferredMaterialization,
        )
    }

    fn respec(
        surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
        spec: LoadSpec,
    ) -> mlx_gen::gen_core::MemoryContractSurfaceSpec {
        mlx_gen::gen_core::MemoryContractSurfaceSpec {
            selector: surface.selector,
            spec,
        }
    }

    /// sc-18451: the pose-control route reached the registry with no selector-aware resolver, so its
    /// weights-free surface derived streamability by asking `packed_quant_bits` about a snapshot path
    /// that does not exist. Every already-packed tier therefore looked like a load-time quantize and
    /// rung 4 was withdrawn — including q4, the only tier this route's evidence covers. The selector
    /// names the resolved artifact tier, so all three prepacked tiers must now reach rung 4 on the
    /// residency/materialization shape the loader actually streams under.
    #[test]
    fn control_selector_surfaces_publish_every_prepacked_tier_at_rung_four() {
        let provider_id = crate::KREA_2_TURBO_CONTROL_ID;
        let mut identities = std::collections::BTreeSet::new();
        let mut streamable_selectors = std::collections::BTreeSet::new();
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            let contract =
                weights_free_memory_strategy_surface_contract(provider_id, &surface).unwrap();
            let expected = surface.selector.offload_policy == mlx_gen::OffloadPolicy::Sequential
                && surface.selector.load_shape
                    == mlx_gen::gen_core::LoadShape::DeferredMaterialization;
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                if expected {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                },
                "{}",
                surface.selector.id()
            );
            if expected {
                streamable_selectors.insert(surface.selector.id());
                assert_eq!(
                    contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
                    vec![
                        MemoryStrategy::Resident,
                        MemoryStrategy::StagedResidency,
                        MemoryStrategy::BoundedDecode,
                        MemoryStrategy::BoundedAttention,
                        MemoryStrategy::BoundedTransformerResidency,
                    ],
                    "{}",
                    surface.selector.id()
                );
            }
            assert_eq!(contract.provider_id, provider_id);
            assert_eq!(contract.asset_facts, Default::default());
            let fingerprint = &contract.calibration.as_ref().unwrap().fingerprint;
            assert!(
                fingerprint.starts_with(STATIC_BEHAVIOR_FINGERPRINT),
                "{}: {fingerprint}",
                surface.selector.id()
            );
            assert_ne!(
                fingerprint, MEMORY_CALIBRATION_FINGERPRINT,
                "the declaration surface must never publish the measured identity"
            );
            assert!(contract.conformance_errors().is_empty());
            identities.insert(format!(
                "{fingerprint}:{:?}",
                contract.calibration.as_ref().unwrap().load_shape
            ));
        }
        // Shape, not a population: every selector resolves to its own identity, and the tiers that
        // reach rung 4 are exactly the three shipped ones on the sequential+deferred shape.
        assert_eq!(
            identities.len(),
            mlx_gen::gen_core::mlx_memory_contract_surface_specs().len(),
            "one static behavior identity per selector"
        );
        assert_eq!(
            streamable_selectors,
            [
                "bf16:sequential:deferred",
                "q4:sequential:deferred",
                "q8:sequential:deferred"
            ]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
        );
    }

    /// The resolver is what recovers the prepacked tiers: the fixture seam has no selector and still
    /// reads the witness path, so it keeps reporting the withdrawn rung. Pinning both sides here is
    /// what makes the regression visible if the resolver registration is ever dropped again.
    #[test]
    fn control_surface_resolver_recovers_the_rung_the_fixture_seam_cannot_see() {
        let surface = prepacked_q4_surface();
        let rung = |contract: &MemoryProviderContract| {
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support
                .clone()
        };
        let fixture =
            weights_free_memory_strategy_contract(crate::KREA_2_TURBO_CONTROL_ID, &surface.spec)
                .unwrap();
        assert_eq!(rung(&fixture), MemoryStrategySupport::Missing);
        let resolved =
            weights_free_memory_strategy_surface_contract(crate::KREA_2_TURBO_CONTROL_ID, &surface)
                .unwrap();
        assert_eq!(rung(&resolved), MemoryStrategySupport::Implemented);
        assert!(resolved.lifecycle.transformer_window_materialization);
        assert_eq!(
            resolved
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_components,
            vec![TransformerComponent::Dit],
            "the pose branch stays resident; only the reopenable base DiT is windowed"
        );
        assert_eq!(
            resolved
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_sizes,
            vec![TRANSFORMER_WINDOW_SIZE]
        );
    }

    /// Each guard is mutated ALONE against the otherwise-valid q4 sequential deferred witness, so a
    /// removed guard cannot hide behind another one still rejecting.
    #[test]
    fn control_surface_resolver_fails_closed_on_each_mutated_axis() {
        let valid = prepacked_q4_surface();
        assert!(weights_free_memory_strategy_surface_contract(
            crate::KREA_2_TURBO_CONTROL_ID,
            &valid
        )
        .is_ok());

        assert!(
            weights_free_memory_strategy_surface_contract("krea_2_turbo", &valid).is_err(),
            "a base provider id must not be handed the pose-control ladder"
        );

        let mut tier_mismatch = respec(&valid, valid.spec.clone());
        tier_mismatch.spec.quantize = Some(Quant::Q8);
        let mut file_source = respec(&valid, valid.spec.clone());
        file_source.spec.weights = WeightsSource::File("/krea.safetensors".into());
        let mut policy_mismatch = respec(&valid, valid.spec.clone());
        policy_mismatch.spec.offload_policy = mlx_gen::OffloadPolicy::Resident;
        let mut shape_mismatch = respec(&valid, valid.spec.clone());
        shape_mismatch.spec.load_shape = mlx_gen::gen_core::LoadShape::EagerMaterialization;
        let mut precision = respec(&valid, valid.spec.clone());
        precision.spec.precision = Precision::Fp32;
        let mut unknown_component = respec(&valid, valid.spec.clone());
        unknown_component.spec.components.insert(
            "unknown".into(),
            WeightsSource::File("/unknown.safetensors".into()),
        );

        for (case, mutated) in [
            ("tier_mismatch", tier_mismatch),
            ("file_source", file_source),
            ("policy_mismatch", policy_mismatch),
            ("shape_mismatch", shape_mismatch),
            ("precision", precision),
            ("unknown_component", unknown_component),
        ] {
            assert!(
                weights_free_memory_strategy_surface_contract(
                    crate::KREA_2_TURBO_CONTROL_ID,
                    &mutated
                )
                .is_err(),
                "{case} must be refused by the resolver"
            );
        }

        // Supported compositions stay admissible, and a dense `.diff` patch is a TRUTHFUL
        // non-streamable contract rather than a resolver error: it mutates the resident base, so the
        // window cannot be rebuilt from the pristine snapshot.
        let tmp = tempfile::tempdir().unwrap();
        let wan_vae = respec(
            &valid,
            valid.spec.clone().with_component(
                mlx_gen::VAE_COMPONENT,
                WeightsSource::File(tmp.path().join("wan-vae.safetensors")),
            ),
        );
        assert_eq!(
            weights_free_memory_strategy_surface_contract(crate::KREA_2_TURBO_CONTROL_ID, &wan_vae)
                .unwrap()
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );

        let dense_diff = tmp.path().join("dense-diff.safetensors");
        let mut header = br#"{"transformer_blocks.0.attn.to_q.diff":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(0.0_f32.to_le_bytes());
        std::fs::write(&dense_diff, bytes).unwrap();
        let diff = respec(
            &valid,
            valid
                .spec
                .clone()
                .with_adapters(vec![mlx_gen::AdapterSpec::new(
                    dense_diff,
                    1.0,
                    mlx_gen::AdapterKind::Lora,
                )]),
        );
        assert_eq!(
            weights_free_memory_strategy_surface_contract(crate::KREA_2_TURBO_CONTROL_ID, &diff)
                .expect("a dense diff patch is a truthful contract, not a resolver error")
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
    }

    /// sc-18451: the measured SC-15517 identity belongs to the q4 snapshot route and nothing else.
    #[test]
    fn only_the_measured_q4_snapshot_route_carries_the_measured_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let prepacked_q4 = tmp.path().join("prepacked-q4");
        write_snapshot(&prepacked_q4);
        std::fs::write(
            prepacked_q4.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let dense = tmp.path().join("dense");
        write_snapshot(&dense);
        let overlay = tmp.path().join("control.safetensors");
        write_control(&overlay);
        let donor = tmp.path().join("wan-vae.safetensors");
        write_control(&donor);

        let control = |root: &std::path::Path| {
            LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
                .with_control(WeightsSource::File(overlay.clone()))
        };
        let fingerprint = |spec: &LoadSpec| {
            memory_strategy_contract(crate::KREA_2_TURBO_CONTROL_ID, spec)
                .unwrap()
                .calibration
                .map(|calibration| calibration.fingerprint)
        };

        // A shipped prepacked q4 turnkey and a dense snapshot packed at load are the same measured
        // tier, resolved through the same seam admission reads.
        assert_eq!(
            fingerprint(&control(&prepacked_q4)).as_deref(),
            Some(MEMORY_CALIBRATION_FINGERPRINT)
        );
        assert_eq!(
            fingerprint(&control(&dense).with_quant(Quant::Q4)).as_deref(),
            Some(MEMORY_CALIBRATION_FINGERPRINT)
        );
        for (case, spec) in [
            ("dense", control(&dense)),
            ("q8", control(&dense).with_quant(Quant::Q8)),
            (
                "wan_composite",
                control(&prepacked_q4)
                    .with_component(mlx_gen::VAE_COMPONENT, WeightsSource::File(donor.clone())),
            ),
        ] {
            assert_eq!(fingerprint(&spec), None, "{case} is an unmeasured route");
        }

        // The measured identity still separates the two materialization shapes it was captured
        // under, exactly as the base ladder does.
        let eager = control(&prepacked_q4);
        let deferred = eager
            .clone()
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);
        let identity = |spec: &LoadSpec| {
            memory_strategy_contract(crate::KREA_2_TURBO_CONTROL_ID, spec)
                .unwrap()
                .calibration
                .unwrap()
        };
        assert_eq!(
            identity(&eager).fingerprint,
            identity(&deferred).fingerprint
        );
        assert_ne!(identity(&eager).load_shape, identity(&deferred).load_shape);
    }
}
