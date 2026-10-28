//! Lens / Lens-Turbo MLX shared image-memory ladder.
//!
//! SC-15800's dense-bf16 text-encoder-only result remains an independent legacy measurement. The
//! full ladder uses a new identity and does not relabel that result as DiT, Both, Q4, decode, or
//! attention evidence.
//!
//! # Declaration vs measurement (SC-18605)
//!
//! Before SC-18605 this module derived the *whole* contract from the measured-route key: a load that
//! was not the exact measured `lens` Q4 route or the exact legacy dense `lens_turbo` route published
//! nothing but [`MemoryStrategy::Resident`]. The engine did not agree. `generate_memory_impl` gates
//! staged residency, tiled decode and chunked attention on nothing at all — they are request levers
//! available on every load — and gates the rung-4 block window purely on
//! `can_stream_text`/`can_stream_dit`. So ten of the twelve `lens` registry surfaces and eleven
//! of the twelve `lens_turbo` surfaces carried an executable ladder that no consumer could ever
//! select, because it was undeclared. That is the inverse of the usual defect: the mechanism was
//! reachable, the declaration was not.
//!
//! The contract therefore now has two independent axes, and every consumer must read both:
//!
//! * **Support** — which rungs the engine can execute for this exact `(provider, LoadSpec)`. Derived
//!   from the engine's own predicates, never from the measured-route key.
//! * **Calibration identity** — whether a *measurement* backs this route. The two measured keys are
//!   unchanged and still bind to their exact envelopes. Every other route publishes no production
//!   calibration, so [`mlx_gen::gen_core::standard_memory_strategy_safety_check`] admits it only
//!   under an explicit [`mlx_gen::gen_core::MemoryOptimizationAuthority::Estimated`] authority.
//!
//! The weights-free declaration surface consumed by the catalog inventory substitutes a static
//! behavior identity ([`STATIC_BEHAVIOR_FINGERPRINT`]) for the absent production calibration so the
//! registry conformance walk can build a run context. That identity is never a measurement and never
//! appears on a contract built from a real load.
//!
//! ## The one asymmetry, deliberately kept
//!
//! `lens_turbo` at `bf16:sequential:deferred` is the single surface covered by the SC-15800 legacy
//! text-encoder measurement. It keeps exactly that envelope — rung 4 scoped to
//! [`TransformerComponent::TextEncoder`], with rungs 1-3 unpublished — because broadening it would
//! relabel that measurement as decode, attention or `Both` evidence, which the module contract above
//! forbids. Its five uncalibrated deferred siblings publish the full structural ladder. Closing that
//! gap needs a new `lens_turbo` measurement, not a wider declaration.

use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    Result as CoreResult, TransformerComponent,
};
#[cfg(test)]
use mlx_gen::{gen_core::MemoryGeometry, GenerationRequest};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

pub const LEGACY_TEXT_ENCODER_FINGERPRINT: &str = "lens-text-encoder-window-2026-07-31-v1";
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "lens-mlx-shared-ladder-2026-08-03-v1";
/// Static, weights-free identity for the registry declaration walk. Never a production calibration.
///
/// A contract built from a real load leaves an unmeasured route's calibration `None` so admission
/// must name an explicit estimate authority. The weights-free surface cannot do that: the shared
/// conformance walk builds its run context through
/// [`mlx_gen::gen_core::standard_memory_behavior_context`], which requires *some* identity. This
/// constant supplies one whose value is structural, so an unmeasured declared rung is still walked
/// without any measured claim attaching to it.
pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "lens-mlx-registry-behavior-v1";
pub const TEXT_ENCODER_WINDOW: u32 = 1;
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 128;
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(provider_id, [DECODE_TILE_EDGE], DECODE_OVERLAP)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CalibrationRoute {
    FullQ4Lens,
    LegacyDenseLensTurboTextEncoder,
    Unmeasured,
}

/// The exact load shape for which the measured production rung is executable and beneficial.
pub(crate) fn is_streamable_spec(spec: &LoadSpec) -> bool {
    matches!(spec.weights, WeightsSource::Dir(_))
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.precision, Precision::Bf16)
        && spec.quantize.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none()
}

fn base_streamable(spec: &LoadSpec) -> Option<&std::path::Path> {
    if spec.load_shape != LoadShape::DeferredMaterialization
        || spec.precision != Precision::Bf16
        || spec.pid.is_some()
    {
        return None;
    }
    match &spec.weights {
        WeightsSource::Dir(root) => Some(root),
        WeightsSource::File(_) => None,
    }
}

/// Fail closed on an id this crate does not register.
///
/// This mattered less while an unmeasured route declared nothing but `Resident`. Now that an
/// unmeasured route publishes a real ladder, an unrecognized id must not be handed one: the rungs
/// below are claims about the Lens engine specifically, and nothing else is entitled to them.
fn validate_provider(provider_id: &str) -> CoreResult<()> {
    if matches!(
        provider_id,
        crate::registry::MODEL_ID_BASE | crate::registry::MODEL_ID_TURBO
    ) {
        Ok(())
    } else {
        Err(CoreError::Unsupported(format!(
            "unknown Lens provider {provider_id}"
        )))
    }
}

fn calibration_route(
    provider_id: &str,
    spec: &LoadSpec,
    text_streamable: bool,
    dit_streamable: bool,
) -> CalibrationRoute {
    if provider_id == "lens"
        && spec.quantize == Some(mlx_gen::Quant::Q4)
        && spec.adapters.is_empty()
        && text_streamable
        && dit_streamable
    {
        CalibrationRoute::FullQ4Lens
    } else if provider_id == "lens_turbo" && is_streamable_spec(spec) {
        CalibrationRoute::LegacyDenseLensTurboTextEncoder
    } else {
        CalibrationRoute::Unmeasured
    }
}

pub(crate) fn can_stream_text(spec: &LoadSpec) -> CoreResult<bool> {
    let Some(root) = base_streamable(spec) else {
        return Ok(false);
    };
    Ok(match spec.quantize {
        Some(quant) => {
            !mlx_gen::quant::needs_load_time_quant(root, "text_encoder", quant.bits(), "lens")?
        }
        None => true,
    })
}

pub(crate) fn can_stream_dit(spec: &LoadSpec) -> CoreResult<bool> {
    let Some(root) = base_streamable(spec) else {
        return Ok(false);
    };
    if !spec.adapters.is_empty() {
        return Ok(false);
    }
    Ok(match spec.quantize {
        Some(quant) => {
            !mlx_gen::quant::needs_load_time_quant(root, "transformer", quant.bits(), "lens")?
        }
        None => true,
    })
}

/// Rung-4 component scopes the engine can actually execute for one load.
///
/// `registry::resolve_transformer_windows` refuses a `TextEncoder`/`Both` window without a
/// streamable text encoder and a `Dit`/`Both` window without a streamable DiT, so this is that
/// gate's own predicate rather than a parallel list that can drift away from it.
fn structural_window_components(
    text_streamable: bool,
    dit_streamable: bool,
) -> Vec<TransformerComponent> {
    let mut components = Vec::new();
    if text_streamable && dit_streamable {
        components.push(TransformerComponent::Both);
    }
    if text_streamable {
        components.push(TransformerComponent::TextEncoder);
    }
    if dit_streamable {
        components.push(TransformerComponent::Dit);
    }
    components
}

/// Per-route static behavior identity for the weights-free declaration surface.
///
/// One shared string across every selector would let a context assembled for one route hand its
/// handshake to another. Keying the identity on the exact axes the contract shape depends on —
/// provider, numeric tier, offload policy, plus the load shape the identity already carries — keeps
/// the declaration walk fail-closed in the same way the measured identities are.
fn static_behavior_identity(provider_id: &str, spec: &LoadSpec) -> MemoryCalibrationIdentity {
    let precision = match spec.precision {
        Precision::Bf16 => "bf16",
        Precision::Fp32 => "fp32",
    };
    let quant = match spec.quantize {
        None => "dense",
        Some(mlx_gen::Quant::Q4) => "q4",
        Some(mlx_gen::Quant::Q8) => "q8",
        Some(mlx_gen::Quant::Nvfp4) => "nvfp4",
    };
    let policy = match spec.offload_policy {
        OffloadPolicy::Resident => "resident",
        OffloadPolicy::Sequential => "sequential",
    };
    // `MemoryProviderContract::conformance_errors` requires lowercase kebab tokens, and the provider
    // ids are snake_case, so `lens_turbo` has to be spelled `lens-turbo` here.
    let route = provider_id.replace('_', "-");
    MemoryCalibrationIdentity::new(
        format!("{STATIC_BEHAVIOR_FINGERPRINT}-{route}-{precision}-{quant}-{policy}"),
        spec.load_shape,
    )
}

/// The tier the snapshot on disk actually is, from the `quantization` marker
/// [`mlx_gen::quant::packed_quant_bits`] reads out of each component's `config.json`.
///
/// The outer `Option` is "is there an artifact to name at all": a `WeightsSource::File` import is
/// not a Lens snapshot tree, so nothing is provable and nothing is named. The inner `Option` is the
/// tier itself, `None` meaning dense/bf16 — the same spelling [`LoadSpec::quantize`] uses, so the
/// two can be compared directly.
///
/// The DiT and the text encoder must AGREE. A tree whose two halves declare different tiers is a
/// self-inconsistent snapshot, not a dense one, and it is an `Err` here rather than a silent `None`
/// so a caller that wants the hard failure can have it. [`production_calibration_identity`] does
/// not: it binds this with `.ok().flatten()`, because an unnameable artifact must cost the load
/// nothing.
fn resolved_artifact_tier(spec: &LoadSpec) -> mlx_gen::Result<Option<Option<mlx_gen::Quant>>> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(None);
    };
    let dit = mlx_gen::quant::packed_quant_bits(root, "transformer")?;
    let text = mlx_gen::quant::packed_quant_bits(root, "text_encoder")?;
    if dit != text {
        return Err(mlx_gen::Error::Msg(format!(
            "lens: snapshot {} declares transformer tier {dit:?} but text_encoder tier {text:?}; a \
             calibration identity cannot name a self-inconsistent tree",
            root.display()
        )));
    }
    match dit {
        None => Ok(Some(None)),
        Some(4) => Ok(Some(Some(mlx_gen::Quant::Q4))),
        Some(8) => Ok(Some(Some(mlx_gen::Quant::Q8))),
        Some(bits) => Err(mlx_gen::Error::Msg(format!(
            "lens: snapshot {} declares an unsupported packed tier Q{bits}",
            root.display()
        ))),
    }
}

/// A load whose resident set is the clean base tree and nothing else.
///
/// Every slot below adds a network beside the three the anchors priced, so an overlay-carrying load
/// is a different resident set that no clean-base measurement covers. Mirrors
/// `mlx-gen-flux/src/memory_strategy.rs`'s `clean_base` over the full `LoadSpec` overlay surface.
fn clean_base(spec: &LoadSpec) -> bool {
    spec.adapters.is_empty()
        && spec.pid.is_none()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.identity.is_none()
        && spec.text_encoder.is_none()
        && spec.components.is_empty()
}

/// Production calibration identity table of the clean Lens base routes, keyed on
/// (route, proven artifact tier) (sc-22732, epic sc-22723 E1/E4).
///
/// This is the TABLE, not the binding: only `production_calibration_identity` — which proves the
/// tier against the artifact on disk first — may turn one of these strings into a contract identity.
///
/// Twelve cells are measurable (two routes x three tiers x two lanes); this table owns the six MLX
/// ones. Each is its own string because a tier is a different resident set and one anchor cannot
/// price three of them. The `lens` q4 cell keeps the measured [`MEMORY_CALIBRATION_FINGERPRINT`]
/// byte-for-byte: it is the same (route, tier) cell that string already named, so preserving it
/// keeps the existing evidence bound rather than orphaning it.
///
/// Offload policy and load shape are deliberately NOT inputs.
/// [`MemoryCalibrationIdentity::load_shape`] carries the materialization axis, and this crate has no
/// Resident/Sequential asymmetry to encode: `spec.offload_policy` reaches no `capability.support`
/// arm, no lifecycle field and no formula variable, and
/// `every_deferred_selector_of_both_providers_declares_rung_four` pins rung-4 support to the load
/// shape alone across both policies.
///
/// [`LEGACY_TEXT_ENCODER_FINGERPRINT`] is deliberately NOT in this table — see
/// `production_calibration_identity`.
pub fn production_calibration_fingerprint(
    provider_id: &str,
    artifact_tier: Option<mlx_gen::Quant>,
) -> Option<String> {
    let tier = match artifact_tier {
        None => "bf16",
        Some(mlx_gen::Quant::Q4) => "q4",
        Some(mlx_gen::Quant::Q8) => "q8",
        Some(_) => return None,
    };
    match (provider_id, artifact_tier) {
        // The one measured MLX cell, preserved byte-for-byte at exactly the cell it measured.
        (crate::registry::MODEL_ID_BASE, Some(mlx_gen::Quant::Q4)) => {
            Some(MEMORY_CALIBRATION_FINGERPRINT.to_owned())
        }
        (crate::registry::MODEL_ID_BASE, _) => Some(format!("lens-{tier}-mlx-shared-ladder-v1")),
        (crate::registry::MODEL_ID_TURBO, _) => {
            Some(format!("lens-turbo-{tier}-mlx-shared-ladder-v1"))
        }
        _ => None,
    }
}

/// The identity a loaded Lens route publishes, bound to the artifact it opens.
///
/// Before sc-22732 the identity was a third consumer of [`CalibrationRoute`], so only two of the
/// twelve measurable cells ever carried one — `lens` q4 and `lens_turbo` dense — and both only under
/// `DeferredMaterialization`, because `calibration_route`'s streamability predicates are reachable
/// only there. A memory anchor reads `contract.calibration` off the loaded generator and refuses a
/// contract without one, so the other ten cells were unmeasurable. The rung DECLARATIONS still come
/// off [`CalibrationRoute`] exactly as before; only the identity moved off that gate.
///
/// Three fail-closed gates:
///
/// * a clean base load at bf16 execution precision ([`clean_base`]) — an overlay stack is a
///   different resident set that no anchor priced, and [`can_stream_dit`] already refuses it;
/// * the tier must be PROVEN from disk and must equal `spec.quantize`. A dense snapshot opened with
///   `quantize = Some(Q4)` requantizes at load — see `mlx_gen::quant::needs_load_time_quant` — and
///   no anchor measured that peak; and
/// * only THEN the legacy arm. [`LEGACY_TEXT_ENCODER_FINGERPRINT`] is the SC-15800 **dense**
///   text-encoder measurement, and the route that carries it publishes a NARROWED envelope — rungs
///   1-3 unpublished, rung 4 scoped to [`TransformerComponent::TextEncoder`]. It must therefore not
///   share the full-ladder `lens-turbo-bf16-…` key, and it stays reachable at `lens_turbo` +
///   [`is_streamable_spec`]. This is the ONE place a load-shape / offload-policy gate survives on an
///   identity in this crate, and it survives because the ladder it names is narrower, not because
///   the tier is.
///
/// The legacy arm sits AFTER the tier proof rather than ahead of it — as a short-circuit it is the
/// #950 blocker in mirror image. [`is_streamable_spec`] constrains the request KNOB
/// (`spec.quantize.is_none()`) but proves nothing about the tree, so ahead of the proof a PACKED
/// snapshot opened with `quantize = None` published the dense-measured string. Behind it,
/// `artifact_tier == spec.quantize == None` forces the artifact itself to be dense, which is what
/// SC-15800 measured.
///
/// Withholding is never fatal: `registry.rs` `?`-propagates [`memory_strategy_contract`] into the
/// LOAD, so [`resolved_artifact_tier`]'s error is bound with `.ok().flatten()` and costs the load
/// nothing.
fn production_calibration_identity(
    provider_id: &str,
    spec: &LoadSpec,
    calibration_route: CalibrationRoute,
) -> Option<MemoryCalibrationIdentity> {
    if !clean_base(spec) || spec.precision != Precision::Bf16 {
        return None;
    }
    let artifact_tier = resolved_artifact_tier(spec).ok().flatten()?;
    if artifact_tier != spec.quantize {
        return None;
    }
    if calibration_route == CalibrationRoute::LegacyDenseLensTurboTextEncoder {
        return Some(MemoryCalibrationIdentity::new(
            LEGACY_TEXT_ENCODER_FINGERPRINT,
            spec.load_shape,
        ));
    }
    production_calibration_fingerprint(provider_id, artifact_tier)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let text_streamable = can_stream_text(spec)?;
    let dit_streamable = can_stream_dit(spec)?;
    let footprint = crate::registry::component_footprint(spec)?;
    memory_strategy_contract_with_surface_facts(
        provider_id,
        spec,
        text_streamable,
        dit_streamable,
        footprint,
        // A production contract never fabricates an identity: an unmeasured route stays uncalibrated
        // so admission has to name an explicit estimate authority for it.
        None,
    )
}

/// Declaration-equivalent registry contract with no filesystem traversal.
///
/// SceneWorks catalog tiers are provisioned snapshot directories. Their q4/q8 subtrees are already
/// packed, so the weights-free surface treats a deferred directory as re-openable rather than
/// asking `needs_load_time_quant` to inspect a nonexistent witness path.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let base_streamable = matches!(spec.weights, WeightsSource::Dir(_))
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.precision == Precision::Bf16
        && spec.pid.is_none();
    memory_strategy_contract_with_surface_facts(
        provider_id,
        spec,
        base_streamable,
        base_streamable && spec.adapters.is_empty(),
        mlx_gen::gen_core::PerComponentBytes::default(),
        Some(static_behavior_identity(provider_id, spec)),
    )
}

/// Architecture axes shared by the two registered Lens routes (epic SC-22657, E2).
///
/// This crate mirrors the reference `transformer/config.json` as
/// [`LensDitConfig::lens`](crate::dit::transformer::LensDitConfig::lens); `lens` and `lens_turbo`
/// run the same DiT and differ only in steps and guidance, so they publish one set of axes.
///
/// The decoder is the FLUX.2 autoencoder Lens reuses: 32 latent channels over a x8 spatial scale,
/// which the DiT's own `out_channels` restates. `vae_temporal_scale` stays `None` — that
/// autoencoder is an image autoencoder with no temporal axis, and a structurally absent axis is
/// declared absent, never zero.
///
/// The activation width follows the spec's precision because the loader does:
/// `registry::resolve_root` maps `Precision::Bf16` to `Dtype::Bfloat16` and
/// `Precision::Fp32` to `Dtype::Float32`, and every component of the tree is opened at that one
/// dtype. `Precision::Fp32` is a served route — nothing on the load path refuses it — so publishing
/// the half width unconditionally under-declared the f32 route's denoise activations by exactly 2x,
/// and an under-price is the defect class epic SC-22657 forbids.
fn architecture_facts(precision: Precision) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit = crate::dit::transformer::LensDitConfig::lens();
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: mlx_gen::architecture_facts::axis(dit.patch_size),
        // `out_channels` is the decoder-side width; `in_channels` 128 is the 2x2-packed view.
        latent_channels: mlx_gen::architecture_facts::axis(dit.out_channels),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(VAE_SPATIAL_SCALE),
        vae_temporal_scale: None,
        activation_dtype_width: Some(activation_width(precision)),
    }
}

/// Bytes per activation element the loader opens this tree at (`registry::resolve_root`).
///
/// Shared by the declared axis above and by `registry::component_footprint`, so the declared
/// activation width and the declared asset bytes can never disagree about the same load.
pub(crate) fn activation_width(precision: Precision) -> u32 {
    match precision {
        Precision::Bf16 => mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH,
        Precision::Fp32 => mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH,
    }
}

/// Pixels per latent unit on each spatial axis for the FLUX.2 autoencoder Lens decodes through.
/// `pipeline::VAE_SCALE_FACTOR` is this times the DiT's 2x2 latent patch, which the tests pin.
const VAE_SPATIAL_SCALE: u32 = 8;

fn memory_strategy_contract_with_surface_facts(
    provider_id: &str,
    spec: &LoadSpec,
    text_streamable: bool,
    dit_streamable: bool,
    footprint: mlx_gen::gen_core::PerComponentBytes,
    unmeasured_identity: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    validate_provider(provider_id)?;
    let calibration_route = calibration_route(provider_id, spec, text_streamable, dit_streamable);
    let legacy_text_encoder_route =
        calibration_route == CalibrationRoute::LegacyDenseLensTurboTextEncoder;
    // Each measured route publishes exactly the envelope it measured; every other route publishes
    // what the engine can execute. Keeping this one list is what stops the two claims from drifting.
    let window_components = match calibration_route {
        CalibrationRoute::FullQ4Lens => vec![TransformerComponent::Both],
        CalibrationRoute::LegacyDenseLensTurboTextEncoder => {
            vec![TransformerComponent::TextEncoder]
        }
        CalibrationRoute::Unmeasured => {
            structural_window_components(text_streamable, dit_streamable)
        }
    };
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
        mlx_gen::gen_core::StagedWeightSchedule::ThreeStage,
    ));
    contract.architecture_facts = architecture_facts(spec.precision);
    let mut formula_variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    if !legacy_text_encoder_route {
        formula_variables.extend([
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
        ]);
    }
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables: formula_variables,
    };
    // The weights-free surface keeps its static identity on EVERY cell; a real load derives its
    // identity from the artifact it opens (sc-22732).
    contract.calibration = match unmeasured_identity {
        Some(identity) => Some(identity),
        None => production_calibration_identity(provider_id, spec, calibration_route),
    };
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae);
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        // The legacy text-encoder measurement covers neither lever, and this surface exists to
        // report that measurement truthfully rather than to advertise the engine's full reach.
        decode_tiling: !legacy_text_encoder_route,
        attention_chunking: !legacy_text_encoder_route,
        transformer_window_materialization: !window_components.is_empty(),
    };
    for capability in &mut contract.strategies {
        match capability.strategy {
            // `Residency::run_staged_request_scoped` takes `stage_residency` as a plain request
            // argument and both Lens residency owners are request-scoped for either offload policy,
            // so shedding the encoder before the denoise/decode phase is executable on every load.
            MemoryStrategy::Resident => {
                capability.support = MemoryStrategySupport::Implemented;
            }
            MemoryStrategy::StagedResidency if !legacy_text_encoder_route => {
                capability.support = MemoryStrategySupport::Implemented;
            }
            // `generate_memory_impl` builds the tiling config and the attention budget straight from
            // the request; neither is gated on tier, offload policy or materialization shape.
            MemoryStrategy::BoundedDecode if !legacy_text_encoder_route => {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.decode_tile_edges = vec![DECODE_TILE_EDGE];
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
            }
            MemoryStrategy::BoundedAttention if !legacy_text_encoder_route => {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
            }
            MemoryStrategy::BoundedTransformerResidency if !window_components.is_empty() => {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.transformer_window_sizes = vec![TEXT_ENCODER_WINDOW];
                capability.parameters.transformer_window_components = window_components.clone();
            }
            _ => {}
        }
    }
    if calibration_route == CalibrationRoute::FullQ4Lens {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    Ok(contract)
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.use_pid
            || contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode)
        {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(|reason| {
                    let detail = if context.use_pid {
                        "the Lens rung-4 calibration covers native VAE decode only, not the PiD/Gemma overlay"
                    } else {
                        "Lens decode route validation failed"
                    };
                    CoreError::Unsupported(format!(
                        "{}: {detail}: {reason}",
                        contract.provider_id
                    ))
                })?;
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: the Lens rung-4 calibration covers native VAE decode only, not the PiD/Gemma overlay",
                contract.provider_id
            )));
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
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            // Derived from the contract this fixture is built for, not from a measured-route key:
            // a fixture that claimed no phases while the selection engages staged residency would
            // be rejected by the very safety gate it exists to exercise.
            has_phases: contract.engages(strategy, MemoryStrategy::StagedResidency),
            overlay: None,
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
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
    let component = context.selection.parameters.window_component();
    let routes = decode_routes(provider_id)?;
    let transformer_blocks = match component {
        TransformerComponent::TextEncoder => crate::config::GptOssConfig::lens().num_layers,
        TransformerComponent::Dit | TransformerComponent::Both => {
            crate::dit::LensDitConfig::lens().num_layers
        }
    };
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
    // The admitted chunk comes from the selection the contract just validated. Keying this on the
    // measured fingerprint instead would leave every declared-but-unmeasured bounded-attention route
    // with no chunk bound, so `configure_attention` would refuse the rung the contract published.
    config.attention_chunk_size = contract
        .engages(context.selection.strategy, MemoryStrategy::BoundedAttention)
        .then_some(context.selection.parameters.attention_chunk_size)
        .flatten();
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
        MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategyParameters, MemoryStrategySupport, TransformerComponent,
        MEMORY_CALIBRATION_ABI,
    };
    use mlx_gen::{AdapterKind, AdapterSpec, PidWeights, Quant};
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture(tmp: &tempfile::TempDir, bits: Option<i32>) -> (std::path::PathBuf, LoadSpec) {
        let root = tmp.path().join(format!(
            "mlx_gen_lens_sc15800_{}",
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), [0_u8; 8]).unwrap();
        }
        for component in ["text_encoder", "transformer"] {
            let config = match bits {
                Some(bits) => {
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#)
                }
                None => r#"{"dtype":"bfloat16"}"#.to_owned(),
            };
            std::fs::write(root.join(component).join("config.json"), config).unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        (root, spec)
    }

    fn dense_legacy_spec(tmp: &tempfile::TempDir) -> (std::path::PathBuf, LoadSpec) {
        let (root, spec) = fixture(tmp, None);
        (root, spec.with_offload_policy(OffloadPolicy::Sequential))
    }

    fn packed_spec(
        tmp: &tempfile::TempDir,
        bits: i32,
        quant: Quant,
    ) -> (std::path::PathBuf, LoadSpec) {
        let (root, mut spec) = fixture(tmp, Some(bits));
        spec.quantize = Some(quant);
        (root, spec)
    }

    /// One route checked on both of the contract's independent claims.
    ///
    /// The identity claim names the (route, proven artifact tier) cell, or is absent when the load
    /// is not a clean base load at its artifact's own tier (sc-22732 moved it off `CalibrationRoute`
    /// so an anchor has something to bind on every measurable cell). The support claim is a separate
    /// question, and it must equal what the engine can execute: rungs 1-3 are unconditional request
    /// levers, and rung 4 is exactly the scopes `can_stream_text`/`can_stream_dit` allow.
    fn assert_structural_ladder(
        contract: &MemoryProviderContract,
        expected_window_components: &[TransformerComponent],
        expected_fingerprint: Option<&str>,
    ) {
        assert_eq!(
            contract
                .calibration
                .as_ref()
                .map(|identity| identity.fingerprint.as_str()),
            expected_fingerprint,
            "production calibration identity"
        );
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?} is an unconditional Lens request lever and must stay selectable"
            );
        }
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(
            window.support,
            if expected_window_components.is_empty() {
                MemoryStrategySupport::Missing
            } else {
                MemoryStrategySupport::Implemented
            },
            "rung 4 support must equal the engine's own streamability gate"
        );
        assert_eq!(
            window.parameters.transformer_window_components,
            expected_window_components
        );
    }

    #[test]
    fn exact_q4_lens_route_publishes_the_measured_full_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 4, Quant::Q4);
        let contract = memory_strategy_contract("lens", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.support, MemoryStrategySupport::Implemented);
        assert_eq!(decode.parameters.decode_tile_edges, vec![DECODE_TILE_EDGE]);
        assert_eq!(decode.parameters.decode_overlaps, vec![DECODE_OVERLAP]);
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(attention.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            attention.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(window.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            window.parameters.transformer_window_components,
            vec![TransformerComponent::Both]
        );
        let measured = rung_four_context().selection;
        contract
            .validate_selection(&measured)
            .expect("the measured Both scope must remain selectable");
        for unmeasured in [TransformerComponent::TextEncoder, TransformerComponent::Dit] {
            let mut selection = measured;
            selection.parameters.transformer_window_component = Some(unmeasured);
            let error = contract
                .validate_selection(&selection)
                .expect_err("an unmeasured component scope must remain unpublished");
            assert!(
                error.to_string().contains("transformer_window_component")
                    && error.to_string().contains("[Both]"),
                "the refusal must identify the unadvertised component: {error}"
            );
        }
        assert!(contract.lifecycle.decode_tiling);
        assert!(contract.lifecycle.attention_chunking);
        assert!(contract.lifecycle.transformer_window_materialization);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn legacy_dense_lens_turbo_te_only_identity_remains_separate() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = dense_legacy_spec(&tmp);
        let contract = memory_strategy_contract("lens_turbo", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            LEGACY_TEXT_ENCODER_FINGERPRINT
        );
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(window.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            window.parameters.transformer_window_components,
            vec![TransformerComponent::TextEncoder]
        );
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        let MemoryFormulaKind::PhaseEnvelope { variables, .. } = &contract.formula else {
            panic!("legacy Lens-Turbo calibration must retain its phase envelope")
        };
        assert!(!variables.contains(&MemoryFormulaVariable::DecodeTileArea));
        assert!(!variables.contains(&MemoryFormulaVariable::AttentionChunkSize));
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn provider_tier_and_load_shape_mutations_do_not_fan_out_evidence() {
        const EVERY_SCOPE: &[TransformerComponent] = &[
            TransformerComponent::Both,
            TransformerComponent::TextEncoder,
            TransformerComponent::Dit,
        ];

        let tmp = tempfile::tempdir().unwrap();
        let (q4_root, q4) = packed_spec(&tmp, 4, Quant::Q4);
        // The two providers are audited independently: `lens_turbo` at the exact tier that measures
        // `lens` inherits none of that evidence, but it is the same engine, so it keeps the reach.
        assert_structural_ladder(
            &memory_strategy_contract("lens_turbo", &q4).unwrap(),
            EVERY_SCOPE,
            Some("lens-turbo-q4-mlx-shared-ladder-v1"),
        );

        let (q8_root, q8) = packed_spec(&tmp, 8, Quant::Q8);
        assert_structural_ladder(
            &memory_strategy_contract("lens", &q8).unwrap(),
            EVERY_SCOPE,
            Some("lens-q8-mlx-shared-ladder-v1"),
        );

        // `lens` on the dense Sequential+Deferred shape is NOT the legacy `lens_turbo` cell, so it
        // publishes its own full-ladder bf16 key.
        let (dense_root, dense) = dense_legacy_spec(&tmp);
        assert_structural_ladder(
            &memory_strategy_contract("lens", &dense).unwrap(),
            EVERY_SCOPE,
            Some("lens-bf16-mlx-shared-ladder-v1"),
        );

        // Eager materialization: rungs 1-3 stay executable, but the block window has nothing left to
        // bound, which is the one shared prerequisite gen-core declares for rung 4.
        let mut eager = q4.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        assert_structural_ladder(
            &memory_strategy_contract("lens", &eager).unwrap(),
            &[],
            Some(MEMORY_CALIBRATION_FINGERPRINT),
        );

        // A merged adapter fixes the DiT weights at load, so only the encoder half can still stream.
        let mut adapted = q4.clone();
        adapted.adapters.push(AdapterSpec::new(
            q4_root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        assert_structural_ladder(
            &memory_strategy_contract("lens", &adapted).unwrap(),
            &[TransformerComponent::TextEncoder],
            None,
        );

        let mut pid = q4.clone();
        pid.pid = Some(PidWeights {
            checkpoint: WeightsSource::File(q4_root.join("pid.safetensors")),
            gemma: WeightsSource::Dir(q4_root.join("gemma")),
        });
        assert_structural_ladder(&memory_strategy_contract("lens", &pid).unwrap(), &[], None);

        for root in [q4_root, q8_root, dense_root] {
            std::fs::remove_dir_all(root).ok();
        }
    }

    /// Rung 4 is declared on every load the engine can actually window, for **both** providers.
    ///
    /// This is the SC-18605 repair stated as an assertion: before it, `lens` published rung 4 on the
    /// two measured Q4 deferred selectors only and `lens_turbo` on the one legacy dense selector
    /// only, while `resolve_transformer_windows` would have run the block window on all six deferred
    /// selectors of both. The declaration, not the mechanism, was the thing missing.
    #[test]
    fn every_deferred_selector_of_both_providers_declares_rung_four() {
        let tmp = tempfile::tempdir().unwrap();
        let mut declared = Vec::new();
        for provider_id in ["lens", "lens_turbo"] {
            for (bits, quant) in [
                (None, None),
                (Some(4), Some(Quant::Q4)),
                (Some(8), Some(Quant::Q8)),
            ] {
                for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
                    for load_shape in [
                        LoadShape::DeferredMaterialization,
                        LoadShape::EagerMaterialization,
                    ] {
                        let (root, mut spec) = fixture(&tmp, bits);
                        spec.quantize = quant;
                        spec = spec.with_offload_policy(policy).with_load_shape(load_shape);
                        let contract = memory_strategy_contract(provider_id, &spec).unwrap();
                        assert!(
                            contract.conformance_errors().is_empty(),
                            "{provider_id} {policy:?} {load_shape:?}"
                        );
                        let window = contract
                            .capability(MemoryStrategy::BoundedTransformerResidency)
                            .unwrap();
                        let implemented = window.support == MemoryStrategySupport::Implemented;
                        assert_eq!(
                            implemented,
                            load_shape == LoadShape::DeferredMaterialization,
                            "{provider_id} {quant:?} {policy:?} {load_shape:?}"
                        );
                        if implemented {
                            assert_eq!(
                                window.parameters.transformer_window_sizes,
                                vec![TEXT_ENCODER_WINDOW]
                            );
                            declared.push((provider_id, policy, quant));
                        }
                        std::fs::remove_dir_all(root).ok();
                    }
                }
            }
        }
        assert_eq!(
            declared.len(),
            12,
            "six deferred selectors per provider must declare rung 4: {declared:?}"
        );
    }

    /// The weights-free declaration surface substitutes a per-route static identity, never a
    /// measurement, and never lets one route's handshake stand in for another's.
    #[test]
    fn weights_free_declaration_identities_are_static_and_route_exact() {
        let mut seen = std::collections::BTreeSet::new();
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            for provider_id in ["lens", "lens_turbo"] {
                let contract =
                    weights_free_memory_strategy_contract(provider_id, &surface.spec).unwrap();
                assert!(
                    contract.conformance_errors().is_empty(),
                    "{provider_id} {}: {:?}",
                    surface.selector.id(),
                    contract.conformance_errors()
                );
                let fingerprint = &contract.calibration.as_ref().unwrap().fingerprint;
                assert!(
                    fingerprint.starts_with(STATIC_BEHAVIOR_FINGERPRINT),
                    "every declaration surface carries the static behavior identity: {fingerprint}"
                );
                assert!(
                    seen.insert(format!("{fingerprint}:{:?}", surface.spec.load_shape)),
                    "static behavior identity {fingerprint} is reused across routes"
                );
            }
        }
        // 24, not `2 * 12 - 3`: before sc-22732 three of the weights-free cells leaked a PRODUCTION
        // string (`lens` q4 under both policies, `lens_turbo` dense sequential), because the
        // measured arms of the `CalibrationRoute` match outranked the caller-supplied static
        // identity. The weights-free surface now keeps its own identity on every cell, so all
        // 2 providers x 12 selectors are static and distinct — the static key carries route,
        // precision, quant and policy, and the pair carries the load shape, which is exactly the
        // 3 tiers x 2 policies x 2 load shapes each selector set spans.
        assert_eq!(seen.len(), 2 * 12);
    }

    /// Every (route, tier) cell the six-cell MLX table names, at the production seam.
    const PRODUCTION_TIERS: [(&str, Option<i32>, Option<Quant>); 3] = [
        ("bf16", None, None),
        ("q4", Some(4), Some(Quant::Q4)),
        ("q8", Some(8), Some(Quant::Q8)),
    ];

    /// The two shapes an anchor capture drives: the worker's own Resident/Eager still-image shape,
    /// and the staged Sequential/Deferred one.
    const CAPTURE_SHAPES: [(OffloadPolicy, LoadShape); 2] = [
        (OffloadPolicy::Resident, LoadShape::EagerMaterialization),
        (
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization,
        ),
    ];

    fn tier_spec(
        root: &std::path::Path,
        quant: Option<Quant>,
        offload: OffloadPolicy,
        load_shape: LoadShape,
    ) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
        spec.quantize = quant;
        spec.with_offload_policy(offload)
            .with_load_shape(load_shape)
    }

    /// sc-22732 (epic sc-22723, E1 measurable / E4 production loader): all six MLX cells — two
    /// routes x three artifact tiers — publish their own production calibration identity through the
    /// production builder the loader calls, under the worker's Resident/Eager shape and under the
    /// staged Sequential/Deferred one. Ten of the twelve had none before, because the identity was a
    /// third consumer of `CalibrationRoute`, whose streamability predicates are reachable only under
    /// `DeferredMaterialization`.
    ///
    /// *Mutations this kills:* restoring the `CalibrationRoute`-driven `contract.calibration` match
    /// (every Resident/Eager cell, and every tier but `lens` q4 / `lens_turbo` dense, goes back to
    /// `None`); dropping the `{tier}` token from the table's format string (the six strings collapse
    /// to two and the distinctness assert reds); dropping the route token (they collapse to three);
    /// keying the string on the offload policy or the load shape (the two shapes below disagree);
    /// and deleting the `artifact_tier != spec.quantize` refusal (the wrong-quant loop reds, since a
    /// requantize-at-load peak is nobody's anchor).
    #[test]
    fn every_lens_tier_publishes_its_routes_production_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let mut published = std::collections::BTreeSet::new();
        for provider in ["lens", "lens_turbo"] {
            for (tier, bits, quant) in PRODUCTION_TIERS {
                let expected = production_calibration_fingerprint(provider, quant).unwrap();
                assert!(
                    expected.contains(tier) || expected == MEMORY_CALIBRATION_FINGERPRINT,
                    "{provider} {tier}: {expected}"
                );
                let (root, _) = fixture(&tmp, bits);
                for (offload, load_shape) in CAPTURE_SHAPES {
                    let spec = tier_spec(&root, quant, offload, load_shape);
                    let contract = memory_strategy_contract(provider, &spec).unwrap();
                    let identity = contract.calibration.as_ref().unwrap_or_else(|| {
                        panic!("{provider} {tier} {offload:?} {load_shape:?} publishes none")
                    });
                    // The one exception: `lens_turbo` dense on the streamable shape is the SC-15800
                    // legacy text-encoder cell, whose declared envelope is NARROWED (rungs 1-3
                    // unpublished, rung 4 scoped to the text encoder). It must not share the
                    // full-ladder `lens-turbo-bf16-…` key with the other three shapes.
                    let legacy = provider == "lens_turbo" && is_streamable_spec(&spec);
                    assert_eq!(
                        identity.fingerprint,
                        if legacy {
                            LEGACY_TEXT_ENCODER_FINGERPRINT
                        } else {
                            expected.as_str()
                        },
                        "{provider} {tier} {offload:?} {load_shape:?}"
                    );
                    assert_eq!(identity.load_shape, load_shape);
                    assert!(contract.conformance_errors().is_empty());
                }

                // The request knob never outranks the artifact: every tier the snapshot is NOT.
                for wrong in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                    if wrong == quant {
                        continue;
                    }
                    let spec = tier_spec(
                        &root,
                        wrong,
                        OffloadPolicy::Resident,
                        LoadShape::DeferredMaterialization,
                    );
                    // A packed snapshot opened at a FOREIGN packed tier never reaches the contract
                    // at all: `needs_load_time_quant` is a hard error on packed weights, which is a
                    // strictly stronger refusal than withholding. Every shape that does reach the
                    // contract — a dense snapshot under a quant knob, or a packed one opened dense —
                    // is a genuine requantize-at-load whose peak no anchor measured.
                    match memory_strategy_contract(provider, &spec) {
                        Ok(contract) => assert!(
                            contract.calibration.is_none(),
                            "{provider} {tier} must publish nothing for a {wrong:?} load quant"
                        ),
                        Err(error) => assert!(
                            error.to_string().contains("pre-quantized"),
                            "{provider} {tier} {wrong:?}: unexpected refusal {error}"
                        ),
                    }
                }
                assert!(
                    published.insert(expected.clone()),
                    "{provider} {tier} repeats another cell's identity: {expected}"
                );
                std::fs::remove_dir_all(root).ok();
            }
        }
        // Six distinct strings: two routes x three tiers. A route-only key collapses this to 3, a
        // tier-only key to 2.
        assert_eq!(published.len(), 6);
    }

    /// The two byte-for-byte preserved strings stay bound to the exact cells their measurements
    /// cover, the weights-free namespace stays disjoint from the whole production set, and an
    /// overlay-carrying load publishes nothing.
    ///
    /// *Mutations this kills:* widening the legacy arm by dropping `is_streamable_spec` (the legacy
    /// string appears on Resident/Eager `lens_turbo` cells too); folding
    /// `LEGACY_TEXT_ENCODER_FINGERPRINT` into the table (it would collide with the full-ladder
    /// `lens-turbo-bf16-…` key); moving `MEMORY_CALIBRATION_FINGERPRINT` off the `(lens, q4)` cell;
    /// handing the weights-free path a production identity (the disjointness loop reds); and
    /// dropping `clean_base` (the adapter and PiD loads publish a clean-base string).
    #[test]
    fn the_preserved_lens_fingerprints_are_reachable_at_exactly_one_cell_each() {
        let tmp = tempfile::tempdir().unwrap();
        let production: std::collections::BTreeSet<String> = ["lens", "lens_turbo"]
            .into_iter()
            .flat_map(|provider| {
                PRODUCTION_TIERS.into_iter().map(move |(_, _, quant)| {
                    production_calibration_fingerprint(provider, quant).unwrap()
                })
            })
            .chain([LEGACY_TEXT_ENCODER_FINGERPRINT.to_owned()])
            .collect();
        assert_eq!(production.len(), 7);

        // Walk every (route, tier, policy, load shape) production cell and record where each
        // preserved string is reachable.
        let mut legacy_cells = Vec::new();
        let mut measured_cells = Vec::new();
        for provider in ["lens", "lens_turbo"] {
            for (tier, bits, quant) in PRODUCTION_TIERS {
                let (root, _) = fixture(&tmp, bits);
                for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
                    for load_shape in [
                        LoadShape::DeferredMaterialization,
                        LoadShape::EagerMaterialization,
                    ] {
                        let spec = tier_spec(&root, quant, policy, load_shape);
                        let fingerprint = memory_strategy_contract(provider, &spec)
                            .unwrap()
                            .calibration
                            .unwrap()
                            .fingerprint;
                        if fingerprint == LEGACY_TEXT_ENCODER_FINGERPRINT {
                            legacy_cells.push((provider, tier, policy, load_shape));
                        }
                        if fingerprint == MEMORY_CALIBRATION_FINGERPRINT {
                            measured_cells.push((provider, tier, policy, load_shape));
                        }
                    }
                }
                std::fs::remove_dir_all(root).ok();
            }
        }
        assert_eq!(
            legacy_cells,
            vec![(
                "lens_turbo",
                "bf16",
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
            )],
            "the SC-15800 legacy text-encoder string covers exactly one narrowed envelope"
        );
        assert_eq!(
            measured_cells,
            vec![
                (
                    "lens",
                    "q4",
                    OffloadPolicy::Resident,
                    LoadShape::DeferredMaterialization
                ),
                (
                    "lens",
                    "q4",
                    OffloadPolicy::Resident,
                    LoadShape::EagerMaterialization
                ),
                (
                    "lens",
                    "q4",
                    OffloadPolicy::Sequential,
                    LoadShape::DeferredMaterialization
                ),
                (
                    "lens",
                    "q4",
                    OffloadPolicy::Sequential,
                    LoadShape::EagerMaterialization
                ),
            ],
            "the measured MLX string stays on the `(lens, q4)` cell and no other route or tier"
        );

        // sc-22732: the legacy arm sits BEHIND the tier proof, not ahead of it. The legacy route's
        // own predicate `is_streamable_spec` constrains the request KNOB (`quantize.is_none()`) and
        // proves nothing about the tree, so as a short-circuit it handed the SC-15800 DENSE
        // text-encoder string to a PACKED snapshot opened with `quantize = None` — the #950 blocker
        // in mirror image. Behind the proof, `artifact_tier != spec.quantize` withholds it.
        for (tier, bits) in [("q4", Some(4)), ("q8", Some(8))] {
            let (packed_root, _) = fixture(&tmp, bits);
            let dense_knob = tier_spec(
                &packed_root,
                None,
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
            );
            assert!(
                is_streamable_spec(&dense_knob),
                "{tier}: the legacy route's own predicate still admits this spec, which is why the \
                 tier proof has to run first"
            );
            assert!(
                memory_strategy_contract("lens_turbo", &dense_knob)
                    .unwrap()
                    .calibration
                    .is_none(),
                "a packed {tier} tree opened with quantize = None must not publish the dense \
                 SC-15800 identity"
            );
            std::fs::remove_dir_all(packed_root).ok();
        }

        // The weights-free namespace is disjoint from every production string.
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            for provider in ["lens", "lens_turbo"] {
                let fingerprint = weights_free_memory_strategy_contract(provider, &surface.spec)
                    .unwrap()
                    .calibration
                    .unwrap()
                    .fingerprint;
                assert!(
                    !production.contains(&fingerprint),
                    "weights-free {provider} {} leaks a production string: {fingerprint}",
                    surface.selector.id()
                );
            }
        }

        // An overlay stack is a different resident set than any clean-base anchor priced.
        let (root, _) = fixture(&tmp, Some(4));
        let base = tier_spec(
            &root,
            Some(Quant::Q4),
            OffloadPolicy::Resident,
            LoadShape::EagerMaterialization,
        );
        let mut adapted = base.clone();
        adapted.adapters.push(AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        let mut piloted = base.clone();
        piloted.pid = Some(PidWeights {
            checkpoint: WeightsSource::File(root.join("pid.safetensors")),
            gemma: WeightsSource::Dir(root.join("gemma")),
        });
        let mut external_text = base.clone();
        external_text.text_encoder = Some(WeightsSource::Dir(root.join("te")));
        for (label, spec) in [
            ("adapter", &adapted),
            ("pid", &piloted),
            ("external text encoder", &external_text),
        ] {
            assert!(
                memory_strategy_contract("lens", spec)
                    .unwrap()
                    .calibration
                    .is_none(),
                "an {label} load publishes no clean-base identity"
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    /// A snapshot whose two components declare different tiers WITHHOLDS the identity rather than
    /// failing the LOAD: `registry.rs` `?`-propagates `memory_strategy_contract` into the load, so an
    /// escaping error turns a loadable snapshot into a refused one.
    ///
    /// *Mutations this kills:* `resolved_artifact_tier(spec).ok().flatten()?` ->
    /// `.unwrap().flatten()?` (the error escapes and the contract call panics/fails); and dropping
    /// the `dit != text` agreement check (a half-packed tree silently borrows a tier's string).
    #[test]
    fn a_disagreeing_component_marker_withholds_the_identity_without_failing_the_load() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp, Some(4));
        // The DiT says Q8 while the text encoder still says Q4.
        std::fs::write(
            root.join("transformer").join("config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        assert!(
            resolved_artifact_tier(&LoadSpec::new(WeightsSource::Dir(root.clone()))).is_err(),
            "the tier proof keeps its hard error for callers that want it"
        );
        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            for provider in ["lens", "lens_turbo"] {
                let spec = tier_spec(
                    &root,
                    quant,
                    OffloadPolicy::Resident,
                    LoadShape::EagerMaterialization,
                );
                let contract = memory_strategy_contract(provider, &spec)
                    .expect("a self-inconsistent snapshot must not fail the load");
                assert!(
                    contract.calibration.is_none(),
                    "{provider} {quant:?} must publish no identity for a disagreeing tree"
                );
                assert!(contract.conformance_errors().is_empty());
            }
        }
        std::fs::remove_dir_all(root).ok();
    }

    /// AC (SC-22662): both registered Lens routes publish the axes of the one DiT they share,
    /// derived from this crate's own config constants, and pass the shared facts check.
    #[test]
    fn architecture_facts_follow_the_crate_dit_config() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp, None);
        for provider_id in ["lens", "lens_turbo"] {
            let contract = weights_free_memory_strategy_contract(provider_id, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(24),
                    head_dim: Some(64),
                    transformer_blocks: Some(48),
                    patch_size: Some(2),
                    latent_channels: Some(32),
                    vae_spatial_scale: Some(8),
                    vae_temporal_scale: None,
                    activation_dtype_width: Some(2),
                },
                "{provider_id} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        // The published pair reconstructs the pipeline's enforced stride, so neither axis can drift
        // alone: VAE scale x DiT patch IS `pipeline::VAE_SCALE_FACTOR`.
        let dit = crate::dit::transformer::LensDitConfig::lens();
        assert_eq!(
            VAE_SPATIAL_SCALE * dit.patch_size as u32,
            crate::pipeline::VAE_SCALE_FACTOR
        );
        std::fs::remove_dir_all(root).ok();
    }

    /// The declared activation width is the dtype `registry::resolve_root` opens the tree at, so the
    /// served `Precision::Fp32` route publishes the f32 width rather than the bf16 one.
    ///
    /// Mutation: restoring the unconditional `HALF_ACTIVATION_WIDTH` reds the `Some(4)` arm, and
    /// swapping the two arms of `activation_width` reds the `Some(2)` arm.
    #[test]
    fn the_declared_activation_width_follows_the_loaded_precision() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, base) = fixture(&tmp, None);
        for (precision, width) in [(Precision::Bf16, 2_u32), (Precision::Fp32, 4)] {
            let mut spec = base.clone();
            spec.precision = precision;
            for provider_id in ["lens", "lens_turbo"] {
                for contract in [
                    memory_strategy_contract(provider_id, &spec).unwrap(),
                    weights_free_memory_strategy_contract(provider_id, &spec).unwrap(),
                ] {
                    assert_eq!(
                        contract.architecture_facts.activation_dtype_width,
                        Some(width),
                        "{provider_id} {precision:?} activation width"
                    );
                }
            }
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn an_unregistered_provider_is_never_handed_a_lens_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 4, Quant::Q4);
        for provider_id in ["lens", "lens_turbo"] {
            assert!(memory_strategy_contract(provider_id, &spec).is_ok());
            assert!(weights_free_memory_strategy_contract(provider_id, &spec).is_ok());
        }
        for provider_id in ["lens_pro", "flux1_dev", "", "Lens"] {
            assert!(
                memory_strategy_contract(provider_id, &spec).is_err(),
                "{provider_id}"
            );
            assert!(
                weights_free_memory_strategy_contract(provider_id, &spec).is_err(),
                "{provider_id}"
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    /// Two admission truths, restated after sc-22732 made the Q8 cell nameable.
    ///
    /// The clean `(lens, q8)` base load now PUBLISHES an identity, so its rung 4 is admitted only
    /// against that exact handshake — an empty or stale fingerprint is refused under every
    /// authority, which is the anchor-binding property the epic needs. A load that still publishes
    /// no identity — here an adapter stack, whose resident set no clean-base anchor priced — keeps
    /// the original property: its declared rung is reachable only behind explicit estimate
    /// authority.
    #[test]
    fn published_identities_gate_admission_and_unnamed_loads_still_need_an_estimate() {
        use mlx_gen::gen_core::MemoryOptimizationAuthority;

        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 8, Quant::Q8);
        let contract = memory_strategy_contract("lens", &spec).unwrap();
        let published = contract.calibration.as_ref().unwrap().fingerprint.clone();
        assert_eq!(published, "lens-q8-mlx-shared-ladder-v1");

        let mut context = rung_four_context();
        context.selection.tier.quant = Some(Quant::Q8);
        context.calibration_fingerprint = published;
        for authority in [
            MemoryOptimizationAuthority::Calibrated,
            MemoryOptimizationAuthority::Estimated,
        ] {
            let mut admitted = context.clone();
            admitted.optimization_authority = authority;
            assert_eq!(
                safety_check(&contract, Precision::Bf16, Some(Quant::Q8), &admitted),
                MemorySafetyDecision::Accept,
                "{authority:?} must admit the Q8 rung against its own published handshake"
            );
        }
        for (label, fingerprint) in [("empty", String::new()), ("stale", "stale".to_owned())] {
            for authority in [
                MemoryOptimizationAuthority::Calibrated,
                MemoryOptimizationAuthority::Estimated,
                MemoryOptimizationAuthority::Resident,
            ] {
                let mut refused = context.clone();
                refused.optimization_authority = authority;
                refused.calibration_fingerprint = fingerprint.clone();
                assert!(
                    matches!(
                        safety_check(&contract, Precision::Bf16, Some(Quant::Q8), &refused),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "a {label} handshake must not be admitted under {authority:?}"
                );
            }
        }

        // An adapter stack publishes no identity, so only an explicit estimate reaches its rung —
        // scoped to the text encoder, the one half `can_stream_dit` still allows.
        let mut adapted = spec.clone();
        adapted.adapters.push(AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        let unnamed = memory_strategy_contract("lens", &adapted).unwrap();
        assert!(unnamed.calibration.is_none());
        let mut context = rung_four_context();
        context.selection.tier.quant = Some(Quant::Q8);
        context.selection.parameters.transformer_window_component =
            Some(TransformerComponent::TextEncoder);
        context.calibration_fingerprint = String::new();
        context.optimization_authority = MemoryOptimizationAuthority::Estimated;
        assert_eq!(
            safety_check(&unnamed, Precision::Bf16, Some(Quant::Q8), &context),
            MemorySafetyDecision::Accept,
            "a declared-but-unnamed rung must be reachable behind an estimate"
        );
        for authority in [
            MemoryOptimizationAuthority::Calibrated,
            MemoryOptimizationAuthority::Resident,
        ] {
            let mut refused = context.clone();
            refused.optimization_authority = authority;
            assert!(
                matches!(
                    safety_check(&unnamed, Precision::Bf16, Some(Quant::Q8), &refused),
                    MemorySafetyDecision::Reject { .. }
                ),
                "{authority:?} must not admit an unnamed Lens rung"
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    fn rung_four_context() -> MemoryRunContext {
        MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    stage_residency: None,
                    decode_tile_edge: Some(DECODE_TILE_EDGE),
                    decode_overlap: Some(DECODE_OVERLAP),
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    transformer_window_size: Some(TEXT_ENCODER_WINDOW),
                    transformer_window_component: Some(TransformerComponent::Both),
                },
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            load_shape: LoadShape::DeferredMaterialization,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: true,
            geometry: MemoryGeometry {
                width: 256,
                height: 256,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 1_000_000,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    #[test]
    fn selected_contract_scope_reaches_the_generation_request_and_pid_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 4, Quant::Q4);
        let contract = memory_strategy_contract("lens", &spec).unwrap();
        let context = rung_four_context();
        let mut scope = registered_begin_request("lens", &spec, &contract, &context)
            .unwrap()
            .unwrap();
        let mut request = GenerationRequest {
            width: 256,
            height: 256,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.expect("rung 4 configures request memory");
        assert!(memory.stage_residency);
        assert!(memory.tile_vae_decode);
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(TEXT_ENCODER_WINDOW));
        assert_eq!(
            memory.transformer_window_component,
            Some(TransformerComponent::Both)
        );
        // Registry conformance uses the same tensor-free scope cleanup exercised here; the Device
        // cleanup path is covered by the real-Metal runner.
        drop(scope);

        let mut unmeasured_native = context.clone();
        unmeasured_native.selection.parameters.decode_tile_edge = Some(640);
        assert!(matches!(
            safety_check(
                &contract,
                Precision::Bf16,
                Some(Quant::Q4),
                &unmeasured_native
            ),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut pid = context;
        pid.use_pid = true;
        assert!(matches!(
            safety_check(&contract, Precision::Bf16, Some(Quant::Q4), &pid),
            MemorySafetyDecision::Reject { reason } if reason.contains("native VAE decode only")
        ));
        std::fs::remove_dir_all(root).ok();
    }
}
