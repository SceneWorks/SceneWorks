//! SC-16352: request-scoped memory contract for the four base Krea 2 providers.
//!
//! Pose control has its own contract and seven-block control branch. This module covers the shared
//! 28-block base DiT used by Turbo, Raw, and their edit surfaces.
//!
//! The production domain is one block. Real `krea_2_turbo` weights at 512²/1 step measured the full
//! request (max of conditioning, denoise, decode) against an otherwise-identical Sequential + deferred
//! resident-stack attribution control:
//!
//! | tier / overlay | resident request | window 1 request | reduction |
//! |---|---:|---:|---:|
//! | q4 base | 11.928 GiB | 5.555 GiB | 53.4% |
//! | q8 base | 17.748 GiB | 5.715 GiB | 67.8% |
//! | bf16 base | 28.660 GiB | 8.383 GiB | 70.8% |
//! | q4 LoRA | 12.141 GiB | 5.768 GiB | 52.5% |
//! | q4 LoKr | 13.383 GiB | 7.010 GiB | 47.6% |
//!
//! Every windowed image was byte-identical to its resident control. Low-rank adapters are captured
//! and replayed per materialized block. A dense `.diff`/`.diff_b` patch is excluded at contract build
//! time because it mutates the resident base and cannot be reconstructed from the pristine snapshot.
//! The full-ladder rung-4 key includes `engaged_composition=[Resident, StagedResidency,
//! BoundedDecode, BoundedAttention, BoundedTransformerResidency]`: this loader reopens components
//! between phases, so block streaming is valid only when staged residency is active in the same
//! request.
//!
//! SC-15517 re-ran the complete q4 ladder at 1024²/1 step on the exact Turbo cache revision
//! `d009674080cc1bccf2b629d834c34bf5eccdb723`:
//!
//! | engaged composition | conditioning | denoise | decode | request |
//! |---|---:|---:|---:|---:|
//! | staged | 3.105 GiB | 9.668 GiB | 15.674 GiB | 15.674 GiB |
//! | + 512/64 bounded decode | 3.044 GiB | 9.668 GiB | 12.013 GiB | 12.013 GiB |
//! | + 64 Mi-score attention | 3.380 GiB | 9.409 GiB | 12.013 GiB | 12.013 GiB |
//! | + DiT window 1 | 3.400 GiB | 3.316 GiB | 5.640 GiB | **5.640 GiB** |
//!
//! The attention and block-window arms were pixel-identical to the tiled-decode arm. The independent
//! real-Qwen-VAE 512/64 seam test measured max float delta `1.0857e-2`, mean `2.8614e-4` against the
//! untiled decoder. The final ladder therefore reduces request peak by 64.0% without changing denoise
//! numerics; the bounded-decode comparison retains the existing Qwen spatial blend tolerance.
//!
//! The optional Qwen PiD route uses the student's separately measured 2048/256 output-pixel tiling
//! domain. On exact PiD revision `39d7b0a9003a3fc934d36d8b5658b2d8ea9c1231`, Gemma revision
//! `684c553b5b41a1c835989d89f62f585e6269a7de`, and the same q4 Krea revision, a 768→3072 multitile
//! A/B measured staged+tiled request peak 21.848 GiB and full decode+attention+window peak 15.475 GiB
//! (29.2% lower), with max/mean pixel delta zero. Native and PiD decode domains remain disjoint and
//! are validated against `use_pid` at both admission and request-scope configuration.

use mlx_gen::asset_facts::{
    projected_safetensors_bytes, projected_tensor_headers_bytes, ResidentProjection,
};
#[cfg(test)]
use mlx_gen::gen_core::MemoryGeometry;
use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

use crate::native_remap::DeclaredLogicalShapes;

pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 64;
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
/// The Wan terminal decoder has not been measured as part of Krea's whole-request ladder. A separate
/// identity prevents native-decoder evidence from being reused and forces conservative
/// asset-facts + headroom estimation (including SceneWorks' estimate margin).
pub const WAN_DECODER_CALIBRATION_FINGERPRINT: &str =
    "krea-2-mlx-wan21-decoder-unmeasured-composite-2026-08-10-v1";

/// Static, weights-free identity prefix for the registry declaration walk. Never a production
/// calibration.
///
/// A contract built from a real load leaves an unprovable route's calibration `None` so admission
/// has to name an explicit estimate authority. The weights-free surfaces cannot do that: the shared
/// conformance walk builds its run context through
/// [`mlx_gen::gen_core::standard_memory_behavior_context`], and the SceneWorks fit gate
/// (`every_planned_mlx_lane_resolves_a_weights_free_provider_contract`) requires *some* identity
/// whose `load_shape` matches the spec's. This prefix supplies one whose value is structural, keyed
/// per route, resolved tier, and residency policy so a context assembled for one selector can never
/// hand its handshake to another — and so it can never equal a production string.
pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "krea-2-mlx-registry-behavior-v1";

/// Production calibration identity table of the four registered base Krea 2 routes, keyed on
/// (route, tier).
///
/// `tier` is `bf16`, `q4`, or `q8` — the tier the base transformer ACTUALLY runs at, never the
/// request knob alone. Anything else, and any provider id outside the base family, is `None`.
///
/// **Every** base route — `krea_2_turbo` included — is keyed per (route, tier) as
/// `krea-2-<route>-<tier>-mlx-shared-ladder-v1`. Before sc-22735 all four routes published one
/// measured turbo string at every tier, so a `krea_2_raw` bf16 anchor was indistinguishable by
/// calibration identity from a `krea_2_turbo` q4 anchor, and the capture apparatus binds records
/// by exactly this string.
///
/// Turbo was carved out of the first sc-22735 pass on the argument that its three shipped plan rows
/// already declared the shared key. That argument does not survive the evidence: SceneWorks'
/// `config/memory-anchors.json` holds **no** measured `krea_2_turbo` MLX record at any tier (the one
/// turbo anchor in the catalog is `krea_2_turbo:candle:q4`, which is a Candle string owned by
/// `candle-gen-krea`). Three MLX plan rows sharing a string with nothing measured behind it is not
/// preserved evidence — it is an ambiguity with no offsetting gain, so the retired
/// `krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3` is retained nowhere.
///
/// Offload policy and load shape are deliberately not inputs: the identity names the artifact the
/// evidence was captured against, and [`MemoryCalibrationIdentity`]`::load_shape` carries the
/// materialization axis separately.
///
/// This is the table, not the binding. The tier here is a caller-supplied token; only the contract
/// builder — which proves the tier against the artifact's own packed marker before publishing —
/// may turn one of these strings into a contract identity.
pub fn production_calibration_fingerprint(provider_id: &str, tier: &str) -> Option<String> {
    if !matches!(tier, "bf16" | "q4" | "q8") {
        return None;
    }
    let route = match provider_id {
        crate::model::KREA_2_TURBO_ID => "turbo",
        crate::model::KREA_2_RAW_ID => "raw",
        crate::model::KREA_2_EDIT_ID => "edit",
        crate::model::KREA_2_TURBO_EDIT_ID => "turbo-edit",
        _ => return None,
    };
    Some(format!("krea-2-{route}-{tier}-mlx-shared-ladder-v1"))
}

/// The tier token proven from the ARTIFACT, never from `LoadSpec::quantize` alone.
///
/// The SceneWorks worker passes `LoadSpec::quantize = None` for the packed MLX turnkeys at every
/// tier (`mlx_load_quant_for_resolved_artifact`), so keying on the request knob would collapse all
/// three tiers onto one string. [`crate::model::effective_base_quant_tier`] resolves the load plan
/// and returns the tier the base actually runs at, read from `transformer/config.json`'s packed
/// marker — the same seam `registered_safety_check` reads — so the declared calibration and the
/// admitted tier cannot disagree.
///
/// Fails closed to `None`: an unreadable artifact, a packed-vs-requested mismatch, or an
/// unsupported quantization tier publishes no identity rather than fabricating one or failing the
/// contract build. `None` is the honest answer — admission then has to name an explicit estimate
/// authority.
fn proven_tier_token(provider_id: &str, spec: &LoadSpec) -> Option<&'static str> {
    match crate::model::effective_base_quant_tier(spec, provider_id) {
        Ok(None) => Some("bf16"),
        Ok(Some(mlx_gen::Quant::Q4)) => Some("q4"),
        Ok(Some(mlx_gen::Quant::Q8)) => Some("q8"),
        Ok(Some(_)) => None,
        Err(_) => None,
    }
}

/// The production identity a real load publishes, or `None` when no cell can be proven.
///
/// Precedence is unchanged from the pre-sc-22735 shape: an additive Wan terminal decoder
/// ([`mlx_gen::VAE_COMPONENT`]) is an unmeasured composite and keeps
/// [`WAN_DECODER_CALIBRATION_FINGERPRINT`], winning over the base table so native whole-request
/// evidence can never authorize the composite decode path.
///
/// A [`WeightsSource::File`] import publishes nothing: its dequantized-to-bf16 residency is a
/// different load source with no promoted cell, and the evidence matrix has no load-source axis.
fn production_calibration_identity(
    provider_id: &str,
    spec: &LoadSpec,
) -> Option<MemoryCalibrationIdentity> {
    if spec.components.contains_key(mlx_gen::VAE_COMPONENT) {
        return Some(MemoryCalibrationIdentity::new(
            WAN_DECODER_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        ));
    }
    if matches!(spec.weights, WeightsSource::File(_)) {
        return None;
    }
    let tier = proven_tier_token(provider_id, spec)?;
    production_calibration_fingerprint(provider_id, tier)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
}

/// Per-(route, tier, policy) static behavior identity for the two weights-free declaration seams.
///
/// `MemoryProviderContract::conformance_errors` requires lowercase kebab tokens with exactly one
/// `vN`, so every component spelled into the identity is already one and the route is the provider
/// id with `_` replaced by `-`.
fn static_behavior_identity(
    provider_id: &str,
    tier: &str,
    offload_policy: OffloadPolicy,
    load_shape: LoadShape,
) -> MemoryCalibrationIdentity {
    let policy = match offload_policy {
        OffloadPolicy::Resident => "resident",
        OffloadPolicy::Sequential => "sequential",
    };
    let route = provider_id.replace('_', "-");
    MemoryCalibrationIdentity::new(
        format!("{STATIC_BEHAVIOR_FINGERPRINT}-{route}-{tier}-{policy}"),
        load_shape,
    )
}

/// The already-resolved artifact tier named by a registry surface selector — the resolver seam's
/// tier source, which touches no filesystem.
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
/// one identity whichever seam built it, and — like its sibling in [`crate::memory_strategy`] — it
/// touches no filesystem. A non-bf16 activation precision must not collapse onto the dense bf16
/// token, so it is spelled out.
fn spec_tier_token(spec: &LoadSpec) -> &'static str {
    match (spec.precision, spec.quantize) {
        (Precision::Fp32, _) => "fp32",
        (_, None) => "bf16",
        (_, Some(mlx_gen::Quant::Q4)) => "q4",
        (_, Some(mlx_gen::Quant::Q8)) => "q8",
        (_, Some(mlx_gen::Quant::Nvfp4)) => "nvfp4",
    }
}

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(provider_id, [DECODE_TILE_EDGE], DECODE_OVERLAP)
}

#[cfg(test)]
pub(crate) fn is_streamable_spec(provider_id: &str, spec: &LoadSpec) -> CoreResult<bool> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(false);
    };
    let mut plan = crate::model::resolve_load_plan(spec, root, provider_id)?;
    plan.streamable_transformer = matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && !crate::model::adapters_have_diff_patch_for_spec(spec)?
        && plan.load_time_quant_bits.is_none();
    Ok(plan.streamable_transformer)
}

fn resolved_load_plan(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<crate::model::ResolvedLoadPlan> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "krea memory facts require a snapshot directory".to_owned(),
        ));
    };
    let mut plan = crate::model::resolve_load_plan(spec, root, provider_id)?;
    plan.streamable_transformer = matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && !crate::model::adapters_have_diff_patch_for_spec(spec)?
        && plan.load_time_quant_bits.is_none();
    Ok(plan)
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    if matches!(spec.weights, WeightsSource::File(_)) {
        crate::model::validate_native_krea_spec(spec, provider_id)
            .map_err(|error| CoreError::Msg(error.to_string()))?;
        let base = mlx_gen::require_base_snapshot(spec, provider_id)?;
        // The native loader is retained and smoke-tested as reopenable, but File has no promoted
        // rung-4 measurement. Keep authorization Missing until source-specific evidence exists.
        return native_memory_strategy_contract_from_spec(provider_id, spec, base, false);
    }
    Ok(memory_strategy_contract_with_plan(provider_id, spec)?.0)
}

pub(crate) fn memory_strategy_contract_with_plan(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<(MemoryProviderContract, crate::model::ResolvedLoadPlan)> {
    let _ = crate::model::component_footprint_for(provider_id, spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "krea memory facts require a snapshot directory".to_owned(),
        ));
    };
    let plan = resolved_load_plan(provider_id, spec)?;
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
    let alternate_decoder_bytes = match spec.components.get(mlx_gen::VAE_COMPONENT) {
        Some(WeightsSource::Dir(path)) => {
            projected_safetensors_bytes(path, |_| ResidentProjection::Stored)?
        }
        Some(WeightsSource::File(path)) => spec.read_file_unchanged_if_prepared(path, |p| {
            projected_safetensors_bytes(p, |_| ResidentProjection::Stored)
        })?,
        None => 0,
    };
    let selected_text_encoder =
        crate::model::runtime_encoder_contract().source_for_load(spec, root)?;
    let language_bytes = crate::model::selected_language_resident_bytes(
        &selected_text_encoder,
        plan.effective_quant.map(mlx_gen::Quant::bits),
        provider_id,
    )?;
    let mut vision_bytes = 0;
    if matches!(
        provider_id,
        crate::model::KREA_2_EDIT_ID | crate::model::KREA_2_TURBO_EDIT_ID
    ) {
        let language_contract = crate::model::runtime_encoder_contract();
        let vision = language_contract
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
        let vision_headers = vision.materialized_vision_tensor_headers(
            &crate::model::runtime_vision_encoder_contract(),
            &language_contract,
        )?;
        vision_bytes =
            projected_tensor_headers_bytes(&vision_headers, |_| ResidentProjection::Stored)?;
    }
    let selected_text_encoder_bytes =
        language_bytes.checked_add(vision_bytes).ok_or_else(|| {
            CoreError::Msg(format!(
                "{provider_id}: selected language plus builtin vision resident byte overflow"
            ))
        })?;
    let components = mlx_gen::PerComponentBytes {
        text_encoder: selected_text_encoder_bytes,
        dit: project(&root.join("transformer"), &|name| {
            crate::convert::is_transformer_quant_target(name)
        })?,
        // The native Qwen VAE remains resident for reference/edit encoding; Wan is an additive
        // terminal decoder, so the contract prices both rather than reusing native measurements.
        vae: project(&root.join("vae"), &|_| false)?.saturating_add(alternate_decoder_bytes),
    };
    Ok((
        memory_strategy_contract_with_components(
            provider_id,
            spec,
            components,
            plan.streamable_transformer,
            production_calibration_identity(provider_id, spec),
        )?,
        plan,
    ))
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
///
/// Publishes the source-owned [`static_behavior_identity`], never a production key: this seam
/// describes a *declaration*, not a measurement. Its tier comes from `spec.quantize` — the seam
/// carries no selector and must touch no filesystem.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let plan = resolved_load_plan(provider_id, spec)?;
    memory_strategy_contract_with_components(
        provider_id,
        spec,
        Default::default(),
        plan.streamable_transformer,
        Some(static_behavior_identity(
            provider_id,
            spec_tier_token(spec),
            spec.offload_policy,
            spec.load_shape,
        )),
    )
}

/// The witness self-check every Krea surface resolver runs first: the explicit resolved artifact tier
/// must agree with the witness `LoadSpec`, the source must be a provisioned snapshot directory, and
/// the selector's residency/materialization axes must be the ones the spec carries.
///
/// Shared with the pose-control resolver in [`crate::memory_strategy`] (sc-18451) so the two families
/// cannot drift into different notions of a well-formed witness.
pub(crate) fn surface_selector_matches_spec(
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<()> {
    let tier_matches = match surface.resolved_artifact_tier() {
        mlx_gen::gen_core::MemoryContractSurfaceTier::Bf16 => {
            surface.spec.precision == Precision::Bf16 && surface.spec.quantize.is_none()
        }
        mlx_gen::gen_core::MemoryContractSurfaceTier::Q4 => {
            surface.spec.quantize == Some(mlx_gen::Quant::Q4)
        }
        mlx_gen::gen_core::MemoryContractSurfaceTier::Q8 => {
            surface.spec.quantize == Some(mlx_gen::Quant::Q8)
        }
        mlx_gen::gen_core::MemoryContractSurfaceTier::Nvfp4 => false,
    };
    if tier_matches
        && matches!(surface.spec.weights, WeightsSource::Dir(_))
        && surface.selector.offload_policy == surface.spec.offload_policy
        && surface.selector.load_shape == surface.spec.load_shape
    {
        Ok(())
    } else {
        Err(CoreError::Msg(format!(
            "Krea memory surface selector '{}' does not match its weights-free LoadSpec",
            surface.selector.id()
        )))
    }
}

/// Resolve the finite catalog surface from its explicit already-packed artifact tier.
///
/// The generic Q4/Q8 witness carries `LoadSpec::quantize` only so its selector is self-checking. A
/// real Krea turnkey has a matching packed marker and therefore reaches production with
/// `load_time_quant_bits == None`; interpreting the synthetic witness as a dense source would falsely
/// withdraw both packed tiers. This resolver publishes only the source-derived load eligibility and
/// deliberately retains zero asset facts. Production contract construction remains responsible for
/// proving the selected snapshot marker and tensor inventory.
pub(crate) fn weights_free_memory_strategy_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<MemoryProviderContract> {
    surface_selector_matches_spec(surface)?;
    crate::model::validate_base_krea_load_axes(&surface.spec, provider_id)
        .map_err(|error| CoreError::Msg(error.to_string()))?;
    let streamable = matches!(
        surface.resolved_artifact_tier(),
        mlx_gen::gen_core::MemoryContractSurfaceTier::Bf16
            | mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
            | mlx_gen::gen_core::MemoryContractSurfaceTier::Q8
    ) && surface.spec.precision == Precision::Bf16
        && matches!(surface.spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(surface.spec.load_shape, LoadShape::DeferredMaterialization)
        && !crate::model::adapters_have_diff_patch_for_spec(&surface.spec)?;
    memory_strategy_contract_with_components(
        provider_id,
        &surface.spec,
        Default::default(),
        streamable,
        Some(static_behavior_identity(
            provider_id,
            selector_tier_token(surface.resolved_artifact_tier()),
            surface.spec.offload_policy,
            surface.spec.load_shape,
        )),
    )
}

/// Architecture axes shared by every registered Krea 2 route (epic SC-22657, E2).
///
/// [`Krea2Config::turbo`](crate::config::Krea2Config::turbo) is this crate's mirror of the published
/// `transformer/config.json`, and `Krea2Config::from_snapshot` parses that same file (falling back
/// to `turbo()` per key) at load; the five routes — Turbo, Raw, the two edit routes and the control
/// route — run one DiT and one VAE, so they publish one set of axes.
///
/// `latent_channels` is [`crate::vae::VAE_CHANNELS`], the decoder's own width; the DiT's
/// `in_channels` 64 is the 2x2-packed view of it. `vae_temporal_scale` stays `None`: Krea 2 is an
/// image model whose autoencoder has no temporal axis, and a structurally absent axis is declared
/// absent, never zero.
///
/// When `spec` names a materialized snapshot directory this re-runs `Krea2Config::from_snapshot` —
/// the loader's own `transformer/config.json` parse — so the published trunk axes are the
/// snapshot's rather than the preset's. On the weights-free surface there is nothing to read and
/// the preset, which that parser itself falls back to per key, is the honest answer.
///
/// SC-22667: the two cases are now separated. A *missing* key degrades to the preset because
/// `Krea2Config::from_snapshot` itself degrades per key, so the loader builds exactly that
/// geometry. An `Err` — an unreadable file, malformed JSON, or a config that fails `validate` —
/// does NOT: `load_transformer_with_stream` propagates it and refuses the load, so publishing the
/// turbo preset would describe a model this snapshot cannot produce. The trunk axes are declared
/// absent there instead, which is the rule this very file already states 250 lines down for the
/// sibling base-config projection ("a config that IS present but unreadable or invalid propagates
/// as an error — it is never degraded into `None`", and by the same token never into a preset).
pub(crate) fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit = match mlx_gen::architecture_facts::materialized_root(spec) {
        None => Some(crate::config::Krea2Config::turbo()),
        Some(root) => crate::config::Krea2Config::from_snapshot(root).ok(),
    };
    let Some(dit) = dit else {
        return mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: None,
            head_dim: None,
            transformer_blocks: None,
            patch_size: None,
            // The latent axes are the decoder's own crate constants, not config reads, so they
            // survive a refused trunk parse and the contract still declares a real axis.
            latent_channels: mlx_gen::architecture_facts::axis(crate::vae::VAE_CHANNELS),
            vae_spatial_scale: mlx_gen::architecture_facts::axis(crate::vae::VAE_COMPRESSION),
            vae_temporal_scale: None,
            activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
        };
    };
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.attention_head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: mlx_gen::architecture_facts::axis(dit.patch_size),
        latent_channels: mlx_gen::architecture_facts::axis(crate::vae::VAE_CHANNELS),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(crate::vae::VAE_COMPRESSION),
        vae_temporal_scale: None,
        // The loader gates on `Precision::Bf16` and the DiT computes there.
        activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
    }
}

/// `calibration` is supplied by the caller rather than derived here, because the seams differ in
/// kind: the two production callers pass [`production_calibration_identity`] (a measured or
/// per-(route, tier) key, or `None` when no cell can be proven), while the two weights-free seams
/// pass a [`static_behavior_identity`] that is never a measurement. Deriving one identity inside
/// this shared builder is exactly the sc-22735 defect — it republished the turbo measured key on
/// every route, every tier, and both declaration surfaces.
fn memory_strategy_contract_with_components(
    provider_id: &str,
    spec: &LoadSpec,
    components: mlx_gen::PerComponentBytes,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.phase_facts = Some(mlx_gen::gen_core::MemoryPhaseFacts::staged(
        mlx_gen::gen_core::StagedWeightSchedule::TwoStage,
    ));
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts(spec);
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.calibration = calibration;
    contract.asset_facts.base_bytes = components
        .text_encoder
        .saturating_add(components.dit)
        .saturating_add(components.vae);
    contract.asset_facts.conditioning_bytes = components.text_encoder;
    contract.asset_facts.transformer_bytes = components.dit;
    contract.asset_facts.decoder_bytes = components.vae;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        transformer_window_materialization: streamable,
        ..Default::default()
    };
    if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("compatibility contract contains every strategy")
            .support = MemoryStrategySupport::Implemented;
    }
    let bounded_decode = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .expect("compatibility contract contains every strategy");
    bounded_decode.support = MemoryStrategySupport::Implemented;
    bounded_decode.parameters.decode_tile_edges = routes.published_edges();
    bounded_decode.parameters.decode_overlaps = routes.published_overlaps();

    let bounded_attention = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedAttention)
        .expect("compatibility contract contains every strategy");
    bounded_attention.support = MemoryStrategySupport::Implemented;
    bounded_attention.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
    contract.lifecycle.decode_tiling = true;
    contract.lifecycle.attention_chunking = true;
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
    if streamable {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("compatibility contract contains every strategy");
        capability.support = MemoryStrategySupport::Implemented;
        capability.parameters.transformer_window_sizes = vec![TRANSFORMER_WINDOW_SIZE];
        capability.parameters.transformer_window_components = vec![TransformerComponent::Dit];
    }
    Ok(contract)
}

/// Exact contract for the supported community single-file DiT composition. The native I8 format is
/// dequantized projection-by-projection to bf16, while its scale/descriptor tensors are consumed and
/// dropped; the text encoder and VAE remain sourced from the resident base snapshot.
///
/// Keeping the snapshot and imported forms on the same provider id is intentional: the
/// implementation, phase model, and non-transformer components are the same. The promoted-memory
/// evidence matrix does not currently have a load-source axis, however. Consequently a published
/// snapshot (`Dir`) rung-4 cell must not be described as an imported-file measurement merely because
/// this contract can re-open a pinned `File`; the `File` route needs its own real-path measurement
/// before release evidence may claim that cell. The lower-level loader may still be reopened for its
/// story smoke; the public contract must pass `streamable = false` until that evidence exists.
///
/// sc-22735 makes that separation visible in the *calibration* too:
/// [`production_calibration_identity`] publishes no identity for a `File` import, so admission has
/// to name an explicit estimate authority instead of inheriting a snapshot key. An additive Wan
/// terminal decoder still publishes [`WAN_DECODER_CALIBRATION_FINGERPRINT`] on either source.
pub(crate) fn native_memory_strategy_contract_from_spec(
    provider_id: &str,
    spec: &LoadSpec,
    base_snapshot_dir: &std::path::Path,
    streamable: bool,
) -> CoreResult<MemoryProviderContract> {
    let stored = |path: &std::path::Path, what: &str| -> CoreResult<u64> {
        projected_safetensors_bytes(path, |_| ResidentProjection::Stored).map_err(|error| {
            CoreError::Msg(format!(
                "{provider_id}: native {what} asset facts for '{}': {error}",
                path.display()
            ))
        })
    };
    let dit_file = match &spec.weights {
        WeightsSource::File(path) => path,
        WeightsSource::Dir(path) => {
            return Err(CoreError::Msg(format!(
                "{provider_id}: native memory facts require a single-file DiT, not directory {}",
                path.display()
            )))
        }
    };
    let alternate_decoder_bytes = match spec.components.get(mlx_gen::VAE_COMPONENT) {
        Some(WeightsSource::Dir(path)) => stored(path, "alternate Wan decoder")?,
        Some(WeightsSource::File(path)) => {
            spec.read_file_unchanged_if_prepared(path, |p| stored(p, "alternate Wan decoder"))?
        }
        None => 0,
    };
    // The base tier owns the architecture config the imported file is loaded against, so admission
    // prices the DiT at the SAME declared logical shapes the load will use (sc-20644): otherwise an
    // MXFP8 layer would be priced at its 32-padded storage while the load unpads it, and this
    // projection is the one both native contracts read.
    let base_cfg = base_architecture_config(provider_id, base_snapshot_dir)?;
    let selected_text_encoder =
        crate::model::runtime_encoder_contract().source_for_load(spec, base_snapshot_dir)?;
    let expected_language_bits =
        crate::model::native_text_encoder_expected_quant_bits(base_snapshot_dir)?;
    let language_bytes = crate::model::selected_language_resident_bytes(
        &selected_text_encoder,
        expected_language_bits,
        provider_id,
    )?;
    let mut vision_bytes = 0;
    if matches!(
        provider_id,
        crate::model::KREA_2_EDIT_ID | crate::model::KREA_2_TURBO_EDIT_ID
    ) {
        let language_contract = crate::model::runtime_encoder_contract();
        let builtin = language_contract.validate_source_against_base(
            &WeightsSource::Dir(base_snapshot_dir.join("text_encoder")),
            base_snapshot_dir,
        )?;
        let headers = builtin.materialized_vision_tensor_headers(
            &crate::model::runtime_vision_encoder_contract(),
            &language_contract,
        )?;
        vision_bytes = projected_tensor_headers_bytes(&headers, |_| ResidentProjection::Stored)?;
    }
    let selected_text_encoder_bytes =
        language_bytes.checked_add(vision_bytes).ok_or_else(|| {
            CoreError::Msg(format!(
                "{provider_id}: selected language plus builtin vision resident byte overflow"
            ))
        })?;
    let components = mlx_gen::PerComponentBytes {
        text_encoder: selected_text_encoder_bytes,
        dit: spec.read_file_unchanged_if_prepared(dit_file, |p| {
            native_dit_transformer_bytes(
                provider_id,
                p,
                spec.quantize,
                DeclaredLogicalShapes::from_base(base_cfg.as_ref()),
            )
        })?,
        vae: stored(&base_snapshot_dir.join("vae"), "base VAE")?
            .saturating_add(alternate_decoder_bytes),
    };
    memory_strategy_contract_with_components(
        provider_id,
        spec,
        components,
        streamable,
        production_calibration_identity(provider_id, spec),
    )
}

/// Compatibility shim for the pre-registry native loader. New call sites carry the base snapshot in
/// `LoadSpec::components` and use [`native_memory_strategy_contract_from_spec`].
#[cfg(test)]
pub(crate) fn native_memory_strategy_contract(
    provider_id: &str,
    dit_file: &std::path::Path,
    base_snapshot_dir: &std::path::Path,
) -> CoreResult<MemoryProviderContract> {
    let spec = LoadSpec::new(WeightsSource::File(dit_file.to_path_buf())).with_component(
        mlx_gen::BASE_SNAPSHOT_COMPONENT,
        WeightsSource::Dir(base_snapshot_dir.to_path_buf()),
    );
    native_memory_strategy_contract_from_spec(provider_id, &spec, base_snapshot_dir, false)
}

/// The architecture config of the base tier an imported single file is loaded against — the one
/// [`crate::native_remap::KreaNativeToDiffusersMapping`] derives its declared logical shapes from
/// (sc-20644).
///
/// `Ok(None)` **only** when the base carries no `transformer/config.json` at all — a checked
/// absence, not a failed read. Admission runs against bases that predate/omit the transformer
/// config, and this projection must not start refusing them; with no config the plan simply
/// declares nothing, which is the pre-sc-20644 behaviour
/// ([`crate::native_remap::DeclaredLogicalShapes::NotInScope`]).
///
/// A config that IS present but unreadable or invalid propagates as an error — it is never
/// degraded into `None`, which would silently price MXFP8 at its 32-padded storage while the load
/// unpads it.
pub(crate) fn base_architecture_config(
    provider_id: &str,
    base_snapshot_dir: &std::path::Path,
) -> CoreResult<Option<crate::config::Krea2Config>> {
    if !base_snapshot_dir
        .join("transformer")
        .join("config.json")
        .is_file()
    {
        return Ok(None);
    }
    crate::config::Krea2Config::from_snapshot(base_snapshot_dir)
        .map(Some)
        .map_err(|error| {
            CoreError::Msg(format!(
                "{provider_id}: base architecture config for '{}': {error}",
                base_snapshot_dir.display()
            ))
        })
}

/// Resident bytes of a community single-file native DiT, priced from the compiled logical-weight
/// plan (sc-20385): each layer's planned dense-fallback residency — I8 codes and fp8 values
/// materialize to bf16 (MXFP8 unpadded to its logical shape), scale/descriptor companions are
/// consumed — with the optional load-time group quantization projected onto the quant-target
/// projections. A file the plan compiler refuses (unregistered format, malformed descriptor)
/// cannot be priced and fails closed here, exactly as the load itself would.
/// The SINGLE projection both native contracts read (the t2i one above and the pose-control one in
/// `crate::memory_strategy`), so the two can never disagree about what a native file costs resident.
pub(crate) fn native_dit_transformer_bytes(
    provider_id: &str,
    dit_file: &std::path::Path,
    quant: Option<mlx_gen::Quant>,
    shapes: crate::native_remap::DeclaredLogicalShapes<'_>,
) -> CoreResult<u64> {
    let prefix = crate::native_remap::detect_native_prefix(dit_file).map_err(|error| {
        CoreError::Msg(format!(
            "{provider_id}: native DiT asset facts for '{}': {error}",
            dit_file.display()
        ))
    })?;
    let plan = mlx_gen::logical_weights::plan_logical_weights(
        dit_file,
        &crate::native_remap::KreaNativeToDiffusersMapping::new(prefix, shapes),
    )
    .map_err(|error| {
        CoreError::Msg(format!(
            "{provider_id}: native DiT asset facts for '{}': {error}",
            dit_file.display()
        ))
    })?;
    // The plan's resident headers are logical-keyed (diffusers names) at the dense resident dtype,
    // so the quant-target predicate applies directly and `Stored` means "the planned resident
    // bytes", not the on-disk packing.
    // A plan whose residency has no per-element width (a GGUF container) cannot be priced this way
    // and refuses by name rather than emitting a header that contradicts its own byte count.
    let resident = plan.resident_tensor_headers().map_err(|error| {
        CoreError::Msg(format!(
            "{provider_id}: native DiT asset facts for '{}': {error}",
            dit_file.display()
        ))
    })?;
    mlx_gen::asset_facts::projected_tensor_headers_bytes(&resident, |tensor| {
        if let Some(quant) =
            quant.filter(|_| crate::convert::is_transformer_quant_target(&tensor.name))
        {
            ResidentProjection::GroupQuantized {
                bits: quant.bits(),
                group_size: crate::quant::GROUP_SIZE as usize,
            }
        } else {
            ResidentProjection::Stored
        }
    })
    .map_err(|error| {
        CoreError::Msg(format!(
            "{provider_id}: native DiT asset facts for '{}': {error}",
            dit_file.display()
        ))
    })
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        validate_memory_behavior_route(contract.provider_id.as_str(), context)?;
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            decode_routes(contract.provider_id.as_str())?
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

fn validate_memory_behavior_route(provider_id: &str, context: &MemoryRunContext) -> CoreResult<()> {
    if context.overlay.is_some() {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: base Krea memory routes use typed request axes and no overlay"
        )));
    }
    let route_matches = match provider_id {
        crate::model::KREA_2_TURBO_ID => matches!(
            (
                &context.mode,
                context.geometry.reference_count,
                context.has_phases
            ),
            (mlx_gen::gen_core::MemoryMode::TextToImage, 0, false)
                | (mlx_gen::gen_core::MemoryMode::ImageToImage, 1, false)
        ),
        crate::model::KREA_2_RAW_ID => {
            matches!(
                (
                    &context.mode,
                    context.geometry.reference_count,
                    context.has_phases
                ),
                (mlx_gen::gen_core::MemoryMode::TextToImage, 0, false)
                    | (mlx_gen::gen_core::MemoryMode::ImageToImage, 1, false)
            ) || matches!(
                (
                    &context.mode,
                    context.geometry.reference_count,
                    context.has_phases
                ),
                (mlx_gen::gen_core::MemoryMode::TextToImage, 0, true)
            ) && !context.use_pid
        }
        crate::model::KREA_2_EDIT_ID | crate::model::KREA_2_TURBO_EDIT_ID => matches!(
            (
                &context.mode,
                context.geometry.reference_count,
                context.has_phases
            ),
            (mlx_gen::gen_core::MemoryMode::Edit, 1 | 2, false)
        ),
        _ => false,
    };
    if !route_matches {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: unsupported base Krea memory route {:?} with {} references, use_pid={}, has_phases={}",
            context.mode,
            context.geometry.reference_count,
            context.use_pid,
            context.has_phases
        )));
    }
    Ok(())
}

pub(crate) fn registered_safety_check(
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

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let quant = crate::model::effective_base_quant_tier(spec, &contract.provider_id)?;
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    };
    let mut routes = match contract.provider_id.as_str() {
        crate::model::KREA_2_TURBO_ID | crate::model::KREA_2_RAW_ID => vec![
            (mlx_gen::gen_core::MemoryMode::TextToImage, 0, false, true),
            (mlx_gen::gen_core::MemoryMode::ImageToImage, 1, false, true),
        ],
        crate::model::KREA_2_EDIT_ID | crate::model::KREA_2_TURBO_EDIT_ID => vec![
            (mlx_gen::gen_core::MemoryMode::Edit, 1, false, true),
            (mlx_gen::gen_core::MemoryMode::Edit, 2, false, true),
        ],
        provider_id => {
            return Err(CoreError::Msg(format!(
                "unsupported base Krea memory behavior provider '{provider_id}'"
            )))
        }
    };
    if contract.provider_id == crate::model::KREA_2_RAW_ID {
        routes.push((mlx_gen::gen_core::MemoryMode::TextToImage, 0, true, false));
    }
    let mut fixtures = Vec::new();
    for (mode, reference_count, has_phases, permits_pid) in routes {
        let route = |use_pid| mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mode.clone(),
            reference_count,
            use_pid,
            has_phases,
            // Adapters, PiD, references, and phase presence are already typed request axes. Krea's
            // provider contract has no second string overlay identity for those same facts.
            overlay: None,
        };
        fixtures.push(executable_memory_behavior_fixture(
            mlx_gen::gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                route(false),
            )?,
        ));
        if permits_pid
            && contract.pid_decode_routes.is_some()
            && contract.engages(strategy, MemoryStrategy::BoundedDecode)
        {
            fixtures.push(executable_memory_behavior_fixture(
                mlx_gen::gen_core::standard_memory_behavior_context(
                    contract,
                    strategy,
                    tier,
                    route(true),
                )?,
            ));
        }
    }
    Ok(fixtures)
}

fn executable_memory_behavior_fixture(
    context: MemoryRunContext,
) -> mlx_gen::gen_core::MemoryBehaviorFixture {
    let mut fixture = mlx_gen::gen_core::MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free Krea memory behavior".to_owned();
    let reference = || mlx_gen::Image {
        width: 1,
        height: 1,
        pixels: vec![0, 0, 0],
    };
    fixture.request.conditioning = match (
        &fixture.context.mode,
        fixture.context.geometry.reference_count,
    ) {
        (mlx_gen::gen_core::MemoryMode::TextToImage, 0) => Vec::new(),
        (mlx_gen::gen_core::MemoryMode::ImageToImage, 1) => {
            vec![mlx_gen::Conditioning::Reference {
                image: reference(),
                strength: Some(1.0),
            }]
        }
        (mlx_gen::gen_core::MemoryMode::Edit, 1) => {
            vec![mlx_gen::Conditioning::Reference {
                image: reference(),
                strength: None,
            }]
        }
        (mlx_gen::gen_core::MemoryMode::Edit, 2) => {
            vec![mlx_gen::Conditioning::MultiReference {
                images: vec![reference(), reference()],
            }]
        }
        _ => unreachable!("provider-owned route validation constructs only executable fixtures"),
    };
    if fixture.context.has_phases {
        fixture.request.phases = Some(vec![
            mlx_gen::GenerationPhase {
                steps: 1,
                ..Default::default()
            },
            mlx_gen::GenerationPhase {
                steps: 1,
                ..Default::default()
            },
        ]);
    }
    fixture
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
        crate::model::effective_base_quant_tier(spec, provider_id)?,
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
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemorySelection, MemoryStrategy,
        MemoryStrategyParameters, MemoryStrategySupport,
    };
    use mlx_gen::{AdapterKind, AdapterSpec, Quant};
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn write_minimal_safetensors(path: &std::path::Path) {
        let mut header = br#"{"model.diffusion_model.first.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 2]);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_native_i8_safetensors(path: &std::path::Path) {
        // A real (tiny) int8-per-row layer: since sc-20385 the resident pricing compiles the
        // logical-weight plan, so the descriptor payload must be the genuine ComfyUI JSON, not
        // filler bytes.
        let descriptor = br#"{"format":"int8_tensorwise","per_row":true}"#;
        let mut header = format!(
            concat!(
                r#"{{"model.diffusion_model.blocks.0.attn.wq.weight":{{"dtype":"I8","shape":[2,64],"data_offsets":[0,128]}},"#,
                r#""model.diffusion_model.blocks.0.attn.wq.weight_scale":{{"dtype":"F32","shape":[2],"data_offsets":[128,136]}},"#,
                r#""model.diffusion_model.blocks.0.attn.wq.comfy_quant":{{"dtype":"U8","shape":[{len}],"data_offsets":[136,{end}]}}}}"#
            ),
            len = descriptor.len(),
            end = 136 + descriptor.len()
        )
        .into_bytes();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 136]);
        bytes.extend(descriptor);
        std::fs::write(path, bytes).unwrap();
    }

    /// **sc-20644: a config read that FAILS is never degraded into "no config declared".** Absence
    /// of `transformer/config.json` is a checked condition that yields `None` (the pre-sc-20644
    /// padded-shape behaviour, which this projection must keep accepting); a config that is present
    /// but malformed surfaces as a provider-scoped error, because silently returning `None` there
    /// would price an MXFP8 import at its 32-padded storage while the load unpads it.
    #[test]
    fn base_architecture_config_separates_an_absent_config_from_a_failed_read() {
        let tmp = tempfile::Builder::new()
            .prefix(&format!("krea-base-cfg-{}-", std::process::id()))
            .tempdir()
            .unwrap();

        // No transformer/config.json at all → the named "not in scope" state.
        assert_eq!(
            base_architecture_config("krea_2_turbo", tmp.path()).unwrap(),
            None
        );

        // Present and well-formed → the architecture it declares.
        let transformer = tmp.path().join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            r#"{"num_attention_heads":48,"attention_head_dim":128}"#,
        )
        .unwrap();
        assert_eq!(
            base_architecture_config("krea_2_turbo", tmp.path())
                .unwrap()
                .map(|cfg| cfg.hidden_size),
            Some(48 * 128)
        );

        // Present but unparseable → an error, NOT `None`.
        std::fs::write(transformer.join("config.json"), "{not json").unwrap();
        let error = base_architecture_config("krea_2_turbo", tmp.path())
            .expect_err("a malformed base config must surface, not degrade to `None`")
            .to_string();
        assert!(
            error.contains("krea_2_turbo") && error.contains("base architecture config"),
            "unexpected error: {error}"
        );

        // Present, parseable, but architecturally invalid → also an error.
        std::fs::write(
            transformer.join("config.json"),
            r#"{"num_attention_heads":48,"attention_head_dim":128,"num_key_value_heads":7}"#,
        )
        .unwrap();
        assert!(
            base_architecture_config("krea_2_turbo", tmp.path()).is_err(),
            "an invalid architecture must surface, not degrade to `None`"
        );
    }

    fn fixture(tmp: &tempfile::TempDir) -> (std::path::PathBuf, LoadSpec) {
        let root = tmp.path().join(format!(
            "mlx_gen_krea_sc16352_{}",
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::model::test_encoder_contract(),
            crate::model::test_vision_encoder_contract(),
        )
        .expect("validation-complete text encoder fixture");
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_quant(Quant::Q4);
        (root, spec)
    }

    fn write_diff_patch(path: &std::path::Path) {
        let header = br#"{"transformer_blocks.0.attn.to_q.diff":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::with_capacity(8 + header.len() + 4);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn base_selector_surfaces_publish_all_prepacked_tiers_and_fail_closed_on_mutation() {
        let providers = [
            crate::model::KREA_2_TURBO_ID,
            crate::model::KREA_2_RAW_ID,
            crate::model::KREA_2_EDIT_ID,
            crate::model::KREA_2_TURBO_EDIT_ID,
        ];
        for provider_id in providers {
            let mut implemented = 0;
            for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
                let contract =
                    weights_free_memory_strategy_surface_contract(provider_id, &surface).unwrap();
                let expected = surface.selector.offload_policy == OffloadPolicy::Sequential
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization;
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
                    "{provider_id}: {}",
                    surface.selector.id()
                );
                implemented += usize::from(expected);
                assert_eq!(contract.provider_id, provider_id);
                assert_eq!(contract.asset_facts, Default::default());
            }
            assert_eq!(
                implemented, 3,
                "{provider_id}: bf16, q4, and q8 must all be represented"
            );
        }

        let q4 = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.offload_policy == OffloadPolicy::Sequential
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .expect("q4 sequential deferred surface");

        let mut tier_mismatch = mlx_gen::gen_core::MemoryContractSurfaceSpec {
            selector: q4.selector,
            spec: q4.spec.clone(),
        };
        tier_mismatch.spec.quantize = Some(Quant::Q8);
        for provider_id in providers {
            assert!(
                weights_free_memory_strategy_surface_contract(provider_id, &tier_mismatch).is_err()
            );
        }

        let mut file_source = mlx_gen::gen_core::MemoryContractSurfaceSpec {
            selector: q4.selector,
            spec: q4.spec.clone(),
        };
        file_source.spec.weights = WeightsSource::File("/krea.safetensors".into());
        for provider_id in providers {
            assert!(
                weights_free_memory_strategy_surface_contract(provider_id, &file_source).is_err()
            );
        }

        let mut control = mlx_gen::gen_core::MemoryContractSurfaceSpec {
            selector: q4.selector,
            spec: q4.spec.clone(),
        };
        control.spec.control = Some(WeightsSource::File("/control.safetensors".into()));
        for provider_id in providers {
            assert!(weights_free_memory_strategy_surface_contract(provider_id, &control).is_err());
        }

        let mut unknown_component = mlx_gen::gen_core::MemoryContractSurfaceSpec {
            selector: q4.selector,
            spec: q4.spec.clone(),
        };
        unknown_component.spec.components.insert(
            "unknown".into(),
            WeightsSource::File("/unknown.safetensors".into()),
        );
        for provider_id in providers {
            assert!(
                weights_free_memory_strategy_surface_contract(provider_id, &unknown_component)
                    .is_err()
            );
        }

        let mut wan_vae = q4;
        wan_vae.spec.components.insert(
            mlx_gen::VAE_COMPONENT.into(),
            WeightsSource::File("/wan-vae.safetensors".into()),
        );
        for provider_id in providers {
            let wan_contract = weights_free_memory_strategy_surface_contract(provider_id, &wan_vae)
                .expect("supported Wan VAE component must remain selector-admissible");
            assert_eq!(
                wan_contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            );
            assert_eq!(wan_contract.asset_facts, Default::default());
        }
    }

    #[test]
    fn base_selector_preserves_supported_compositions_and_refuses_dense_diff_btr() {
        let tmp = tempfile::tempdir().unwrap();
        let low_rank = tmp.path().join("low-rank.safetensors");
        let dense_diff = tmp.path().join("dense-diff.safetensors");
        write_minimal_safetensors(&low_rank);
        write_diff_patch(&dense_diff);
        let q4 = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.offload_policy == OffloadPolicy::Sequential
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .expect("q4 sequential deferred surface");

        for provider_id in [
            crate::model::KREA_2_TURBO_ID,
            crate::model::KREA_2_RAW_ID,
            crate::model::KREA_2_EDIT_ID,
            crate::model::KREA_2_TURBO_EDIT_ID,
        ] {
            for kind in [AdapterKind::Lora, AdapterKind::Lokr] {
                let mut composed = mlx_gen::gen_core::MemoryContractSurfaceSpec {
                    selector: q4.selector,
                    spec: q4
                        .spec
                        .clone()
                        .with_adapters(vec![AdapterSpec::new(low_rank.clone(), 1.0, kind)])
                        .with_pid(
                            WeightsSource::File(tmp.path().join("pid.safetensors")),
                            WeightsSource::Dir(tmp.path().join("gemma")),
                        )
                        .with_text_encoder(WeightsSource::Dir(
                            tmp.path().join("external-text-encoder"),
                        )),
                };
                composed.spec.components.insert(
                    mlx_gen::VAE_COMPONENT.into(),
                    WeightsSource::File(tmp.path().join("wan-vae.safetensors")),
                );
                let contract =
                    weights_free_memory_strategy_surface_contract(provider_id, &composed)
                        .expect("low-rank + PiD + external TE + Wan VAE remains admissible");
                assert_eq!(
                    contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .support,
                    MemoryStrategySupport::Implemented,
                    "{provider_id} {kind:?}"
                );
            }

            let dense = mlx_gen::gen_core::MemoryContractSurfaceSpec {
                selector: q4.selector,
                spec: q4.spec.clone().with_adapters(vec![AdapterSpec::new(
                    dense_diff.clone(),
                    1.0,
                    AdapterKind::Lora,
                )]),
            };
            let contract = weights_free_memory_strategy_surface_contract(provider_id, &dense)
                .expect("dense diff is a truthful non-streamable contract, not a resolver error");
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing,
                "{provider_id}"
            );
        }
    }

    #[test]
    fn base_behavior_routes_are_executable_and_fail_closed_on_typed_mutation() {
        let q4 = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.offload_policy == OffloadPolicy::Sequential
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .expect("q4 sequential deferred surface");

        for provider_id in [
            crate::model::KREA_2_TURBO_ID,
            crate::model::KREA_2_RAW_ID,
            crate::model::KREA_2_EDIT_ID,
            crate::model::KREA_2_TURBO_EDIT_ID,
        ] {
            let contract = weights_free_memory_strategy_surface_contract(provider_id, &q4).unwrap();
            let fixtures = registered_valid_fixture(
                &q4.spec,
                &contract,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap();
            let descriptor = match provider_id {
                crate::model::KREA_2_TURBO_ID => crate::model::descriptor(),
                crate::model::KREA_2_RAW_ID => crate::model::raw_descriptor(),
                crate::model::KREA_2_EDIT_ID => crate::model::edit_descriptor(),
                crate::model::KREA_2_TURBO_EDIT_ID => crate::model::turbo_edit_descriptor(),
                _ => unreachable!(),
            };
            for fixture in &fixtures {
                assert_eq!(
                    registered_safety_check(&q4.spec, &contract, &fixture.context),
                    MemorySafetyDecision::Accept,
                    "{provider_id}: {:?}",
                    fixture.context
                );
                crate::model::validate_request(&descriptor, &fixture.request).unwrap_or_else(
                    |error| panic!("{provider_id} fixture is not executable: {error}"),
                );
                assert_eq!(
                    fixture.request.use_pid, fixture.context.use_pid,
                    "{provider_id}"
                );
                assert_eq!(
                    fixture.request.phases.is_some(),
                    fixture.context.has_phases,
                    "{provider_id}"
                );
            }

            let mut wrong_mode = fixtures[0].context.clone();
            wrong_mode.mode = if matches!(
                provider_id,
                crate::model::KREA_2_EDIT_ID | crate::model::KREA_2_TURBO_EDIT_ID
            ) {
                MemoryMode::ImageToImage
            } else {
                MemoryMode::Edit
            };
            wrong_mode.geometry.reference_count = 1;
            wrong_mode.has_reference = true;
            assert!(matches!(
                registered_safety_check(&q4.spec, &contract, &wrong_mode),
                MemorySafetyDecision::Reject { .. }
            ));

            let mut wrong_references = fixtures[0].context.clone();
            wrong_references.geometry.reference_count = 3;
            wrong_references.has_reference = true;
            assert!(matches!(
                registered_safety_check(&q4.spec, &contract, &wrong_references),
                MemorySafetyDecision::Reject { .. }
            ));

            let mut wrong_phases = if provider_id == crate::model::KREA_2_RAW_ID {
                fixtures
                    .iter()
                    .find(|fixture| {
                        fixture.context.mode == MemoryMode::ImageToImage
                            && !fixture.context.has_phases
                    })
                    .unwrap()
                    .context
                    .clone()
            } else {
                fixtures[0].context.clone()
            };
            wrong_phases.has_phases = true;
            assert!(matches!(
                registered_safety_check(&q4.spec, &contract, &wrong_phases),
                MemorySafetyDecision::Reject { .. }
            ));

            let mut overlay = fixtures[0].context.clone();
            overlay.overlay = Some("references:1".to_owned());
            assert!(matches!(
                registered_safety_check(&q4.spec, &contract, &overlay),
                MemorySafetyDecision::Reject { .. }
            ));
            assert!(registered_begin_request(provider_id, &q4.spec, &contract, &overlay).is_err());

            if provider_id == crate::model::KREA_2_RAW_ID {
                let mut pid_multiphase = fixtures
                    .iter()
                    .find(|fixture| fixture.context.has_phases)
                    .unwrap()
                    .context
                    .clone();
                pid_multiphase.use_pid = true;
                assert!(matches!(
                    registered_safety_check(&q4.spec, &contract, &pid_multiphase),
                    MemorySafetyDecision::Reject { .. }
                ));
            }
        }
    }

    /// The four registered base routes. Every calibration-identity expectation below is derived
    /// from this product rather than a frozen list of strings or a frozen count.
    const BASE_ROUTES: [&str; 4] = [
        crate::model::KREA_2_TURBO_ID,
        crate::model::KREA_2_RAW_ID,
        crate::model::KREA_2_EDIT_ID,
        crate::model::KREA_2_TURBO_EDIT_ID,
    ];
    const BASE_TIERS: [&str; 3] = ["bf16", "q4", "q8"];

    /// The full {4 routes} x {3 tiers} product of the production table.
    fn production_cells() -> Vec<(&'static str, &'static str, String)> {
        BASE_ROUTES
            .into_iter()
            .flat_map(|provider_id| {
                BASE_TIERS.into_iter().map(move |tier| {
                    (
                        provider_id,
                        tier,
                        production_calibration_fingerprint(provider_id, tier).unwrap_or_else(
                            || panic!("{provider_id}:{tier} must have a production cell"),
                        ),
                    )
                })
            })
            .collect()
    }

    fn write_transformer_quant_marker(root: &std::path::Path, bits: Option<u32>) {
        let json = match bits {
            Some(bits) => format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            None => "{}".to_owned(),
        };
        std::fs::write(root.join("transformer/config.json"), json).unwrap();
    }

    /// **sc-22735 (a)/(b)/(c): the collision this story exists to close.**
    ///
    /// Before this change all four base routes published the single measured turbo string at all
    /// three tiers, so a `krea_2_raw` bf16 anchor was indistinguishable by calibration identity
    /// from a `krea_2_turbo` q4 anchor. Every one of the twelve cells — turbo's three included —
    /// must now be pairwise distinct and match the documented format.
    #[test]
    fn every_base_route_and_tier_cell_publishes_its_own_production_string() {
        let cells = production_cells();
        assert_eq!(cells.len(), BASE_ROUTES.len() * BASE_TIERS.len());

        let mut per_tier = Vec::new();
        for (provider_id, tier, fingerprint) in &cells {
            mlx_gen::gen_core::validate_calibration_fingerprint(fingerprint)
                .unwrap_or_else(|reason| panic!("{provider_id}:{tier} fingerprint {reason}"));
            let route = provider_id
                .strip_prefix("krea_2_")
                .expect("base route ids are krea_2_*")
                .replace('_', "-");
            assert_eq!(
                *fingerprint,
                format!("krea-2-{route}-{tier}-mlx-shared-ladder-v1"),
                "{provider_id}:{tier}"
            );
            per_tier.push(fingerprint.clone());
        }

        assert_eq!(
            per_tier.len(),
            BASE_ROUTES.len() * BASE_TIERS.len(),
            "every route is keyed per tier"
        );
        let distinct: std::collections::BTreeSet<_> = per_tier.iter().collect();
        assert_eq!(
            distinct.len(),
            per_tier.len(),
            "no two (route, tier) cells may collide: {per_tier:?}"
        );
        // The retired shared key is retained by no cell: no MLX turbo record was ever measured
        // against it, so preserving it would only re-create the ambiguity.
        for fingerprint in &per_tier {
            assert_ne!(
                fingerprint, RETIRED_SHARED_MLX_FINGERPRINT,
                "the retired shared MLX key must not come back"
            );
        }
    }

    /// The string every base route used to publish at every tier, kept here and nowhere else so a
    /// re-collapse onto it reddens a named test rather than passing as a plausible identity.
    const RETIRED_SHARED_MLX_FINGERPRINT: &str =
        "krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3";

    /// (c) `krea_2_raw` never returns the turbo key at any tier — stated on its own so a
    /// provider-match mutation that folds raw back onto turbo reddens a named test.
    #[test]
    fn raw_never_publishes_the_turbo_key() {
        for tier in BASE_TIERS {
            let raw = production_calibration_fingerprint(crate::model::KREA_2_RAW_ID, tier)
                .expect("raw ships every tier");
            assert_ne!(raw, RETIRED_SHARED_MLX_FINGERPRINT, "{tier}");
            for turbo_tier in BASE_TIERS {
                assert_ne!(
                    raw,
                    production_calibration_fingerprint(crate::model::KREA_2_TURBO_ID, turbo_tier)
                        .expect("turbo ships every tier"),
                    "{tier} vs turbo {turbo_tier}"
                );
            }
            assert!(raw.starts_with("krea-2-raw-"), "{raw}");
        }
    }

    /// (f) an unknown provider id, and any tier outside the shipped ladder, get `None` from the
    /// table rather than a fabricated string.
    #[test]
    fn the_production_table_refuses_unknown_routes_and_tiers() {
        for tier in BASE_TIERS {
            assert_eq!(
                production_calibration_fingerprint("krea_2_unknown", tier),
                None
            );
            assert_eq!(
                production_calibration_fingerprint(
                    crate::model_control::KREA_2_TURBO_CONTROL_ID,
                    tier
                ),
                None,
                "the pose-control route owns its own identity in `crate::memory_strategy`"
            );
        }
        for tier in ["nvfp4", "fp32", "", "Q4"] {
            for provider_id in BASE_ROUTES {
                assert_eq!(
                    production_calibration_fingerprint(provider_id, tier),
                    None,
                    "{provider_id}:{tier}"
                );
            }
        }
    }

    /// **The load-bearing binding**: the tier in the published string is proven from the ARTIFACT's
    /// packed marker, not from `LoadSpec::quantize`. The SceneWorks worker passes
    /// `quantize = None` for these routes at every tier on MLX, so a `spec.quantize` key would
    /// collapse all three tiers onto one string.
    #[test]
    fn production_contracts_key_on_the_proven_artifact_tier_not_the_request_knob() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, base) = fixture(&tmp);
        let mut published = Vec::new();
        for (bits, tier) in [(Some(4), "q4"), (Some(8), "q8"), (None, "bf16")] {
            write_transformer_quant_marker(&root, bits);
            let mut spec = base.clone();
            // Exactly what the worker sends: no quant request at ANY tier.
            spec.quantize = None;
            for provider_id in BASE_ROUTES {
                assert_eq!(
                    proven_tier_token(provider_id, &spec),
                    Some(tier),
                    "{provider_id}: packed marker {bits:?}"
                );
                let contract = memory_strategy_contract(provider_id, &spec).unwrap();
                let identity = contract
                    .calibration
                    .as_ref()
                    .expect("a proven base cell publishes an identity");
                assert_eq!(
                    identity.fingerprint,
                    production_calibration_fingerprint(provider_id, tier).unwrap(),
                    "{provider_id}:{tier}"
                );
                assert_eq!(identity.load_shape, spec.load_shape);
                assert_eq!(identity.load_shape, contract.load_shape);
                assert!(contract.conformance_errors().is_empty(), "{provider_id}");
                published.push((provider_id, tier, identity.fingerprint.clone()));
            }
        }
        // The nine non-turbo cells that were built from real contracts are pairwise distinct.
        let non_turbo: Vec<_> = published
            .iter()
            .filter(|(provider_id, ..)| *provider_id != crate::model::KREA_2_TURBO_ID)
            .map(|(.., fingerprint)| fingerprint.clone())
            .collect();
        let distinct: std::collections::BTreeSet<_> = non_turbo.iter().collect();
        assert_eq!(distinct.len(), non_turbo.len(), "{non_turbo:?}");
        std::fs::remove_dir_all(root).ok();
    }

    /// (d) both weights-free seams publish a static-behavior identity — keyed per route, tier and
    /// residency policy — that is never any production string, and never `None` (the SceneWorks
    /// fit gate resolves the fixture registration for every planned MLX anchor and requires an
    /// identity whose `load_shape` equals the spec's).
    #[test]
    fn weights_free_seams_publish_a_static_behavior_identity_that_is_never_production() {
        let production: std::collections::BTreeSet<String> = production_cells()
            .into_iter()
            .map(|(.., fingerprint)| fingerprint)
            .chain([WAN_DECODER_CALIBRATION_FINGERPRINT.to_owned()])
            .collect();
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let surfaces = mlx_gen::gen_core::mlx_memory_contract_surface_specs();
        let expected_distinct: std::collections::BTreeSet<_> = surfaces
            .iter()
            .map(|surface| {
                (
                    selector_tier_token(surface.resolved_artifact_tier()),
                    matches!(surface.spec.offload_policy, OffloadPolicy::Sequential),
                )
            })
            .collect();

        let mut across_routes = std::collections::BTreeSet::new();
        for provider_id in BASE_ROUTES {
            let route = provider_id.replace('_', "-");

            // Fixture seam: tier from `spec.quantize` (the fixture carries Q4).
            let contract = weights_free_memory_strategy_contract(provider_id, &spec).unwrap();
            let identity = contract
                .calibration
                .as_ref()
                .expect("fixture seam identity");
            assert_eq!(
                identity.fingerprint,
                format!("{STATIC_BEHAVIOR_FINGERPRINT}-{route}-q4-sequential")
            );
            assert!(!production.contains(&identity.fingerprint));
            assert_eq!(identity.load_shape, spec.load_shape);
            assert_eq!(identity.load_shape, contract.load_shape);
            assert!(contract.conformance_errors().is_empty(), "{provider_id}");

            // Resolver seam: tier from the surface selector, no filesystem read.
            let mut seen = std::collections::BTreeSet::new();
            for surface in &surfaces {
                let contract =
                    weights_free_memory_strategy_surface_contract(provider_id, surface).unwrap();
                let identity = contract
                    .calibration
                    .as_ref()
                    .expect("resolver seam identity");
                let policy = match surface.spec.offload_policy {
                    OffloadPolicy::Resident => "resident",
                    OffloadPolicy::Sequential => "sequential",
                };
                assert_eq!(
                    identity.fingerprint,
                    format!(
                        "{STATIC_BEHAVIOR_FINGERPRINT}-{route}-{}-{policy}",
                        selector_tier_token(surface.resolved_artifact_tier())
                    ),
                    "{provider_id}: {}",
                    surface.selector.id()
                );
                assert!(
                    !production.contains(&identity.fingerprint),
                    "{provider_id}: {} republished a production key",
                    surface.selector.id()
                );
                assert_eq!(identity.load_shape, surface.spec.load_shape);
                assert_eq!(identity.load_shape, contract.load_shape);
                assert!(contract.conformance_errors().is_empty());
                seen.insert(identity.fingerprint.clone());
            }
            assert_eq!(
                seen.len(),
                expected_distinct.len(),
                "{provider_id}: one identity per (tier, policy) the surface catalog names"
            );
            across_routes.extend(seen);
        }
        assert_eq!(
            across_routes.len(),
            expected_distinct.len() * BASE_ROUTES.len(),
            "the route is part of the identity, so no two routes may share one"
        );
        std::fs::remove_dir_all(root).ok();
    }

    /// (e) fail closed to `None`, never fail the load: an imported single-file DiT has no promoted
    /// cell (the evidence matrix has no load-source axis) and an artifact whose tier cannot be
    /// resolved proves nothing — both publish no identity while the contract still builds.
    #[test]
    fn an_unprovable_route_publishes_no_calibration_and_still_builds_a_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, base) = fixture(&tmp);
        let native = root.join("single.safetensors");
        write_native_i8_safetensors(&native);
        let file = LoadSpec::new(WeightsSource::File(native.clone())).with_component(
            mlx_gen::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(root.clone()),
        );
        for provider_id in BASE_ROUTES {
            let contract = memory_strategy_contract(provider_id, &file)
                .unwrap_or_else(|error| panic!("{provider_id}: {error}"));
            assert_eq!(
                contract.calibration, None,
                "{provider_id}: an imported file must not inherit a snapshot key"
            );
            assert!(contract.conformance_errors().is_empty(), "{provider_id}");
        }

        // A packed-vs-requested mismatch and an unreadable marker both fail closed.
        write_transformer_quant_marker(&root, Some(8));
        let mut mismatched = base.clone();
        mismatched.quantize = Some(Quant::Q4);
        for provider_id in BASE_ROUTES {
            assert!(
                crate::model::effective_base_quant_tier(&mismatched, provider_id).is_err(),
                "{provider_id}"
            );
            assert_eq!(proven_tier_token(provider_id, &mismatched), None);
        }

        std::fs::write(root.join("transformer/config.json"), "{ malformed").unwrap();
        for provider_id in BASE_ROUTES {
            assert_eq!(
                proven_tier_token(provider_id, &base),
                None,
                "{provider_id}: an unreadable marker proves no tier"
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn identical_fingerprint_is_separated_by_typed_load_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, deferred_spec) = fixture(&tmp);
        let deferred = memory_strategy_contract("krea_2_turbo", &deferred_spec).unwrap();
        let mut eager_spec = deferred_spec;
        eager_spec.load_shape = LoadShape::EagerMaterialization;
        let eager = memory_strategy_contract("krea_2_turbo", &eager_spec).unwrap();
        assert_eq!(
            deferred.calibration.as_ref().unwrap().fingerprint,
            eager.calibration.as_ref().unwrap().fingerprint
        );
        assert_ne!(
            deferred.calibration.as_ref().unwrap().load_shape,
            eager.calibration.as_ref().unwrap().load_shape
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn alternate_decoder_is_additive_to_native_decoder_asset_facts() {
        let tmp = tempfile::tempdir().unwrap();
        let (_root, spec) = fixture(&tmp);
        let native = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        let donor = tmp.path().join("wan-vae.safetensors");
        write_minimal_safetensors(&donor);
        let composite = memory_strategy_contract(
            "krea_2_turbo",
            &spec
                .clone()
                .with_component(mlx_gen::VAE_COMPONENT, WeightsSource::File(donor)),
        )
        .unwrap();
        assert_eq!(
            composite.asset_facts.decoder_bytes,
            native.asset_facts.decoder_bytes + 2,
            "Krea keeps its Qwen VAE for conditioning and adds the Wan terminal decoder"
        );
        assert_eq!(
            composite.asset_facts.base_bytes,
            native.asset_facts.base_bytes + 2
        );
        assert_eq!(
            composite.calibration.as_ref().unwrap().fingerprint,
            WAN_DECODER_CALIBRATION_FINGERPRINT,
            "native whole-request measurements must not authorize the composite decoder path"
        );
        assert_eq!(
            native.calibration.as_ref().unwrap().fingerprint,
            production_calibration_fingerprint(crate::model::KREA_2_TURBO_ID, "q4").unwrap(),
            "the fixture packs the transformer at q4, so the native contract names the turbo q4 cell"
        );

        // sc-22735: the VAE-composite marker keeps precedence over the per-(route, tier) base
        // table on EVERY route, not just the one turbo cell that happens to share a string with
        // the measured key.
        let donor = tmp.path().join("wan-vae-shared.safetensors");
        write_minimal_safetensors(&donor);
        for provider_id in BASE_ROUTES {
            let composite = memory_strategy_contract(
                provider_id,
                &spec
                    .clone()
                    .with_component(mlx_gen::VAE_COMPONENT, WeightsSource::File(donor.clone())),
            )
            .unwrap();
            assert_eq!(
                composite.calibration.as_ref().unwrap().fingerprint,
                WAN_DECODER_CALIBRATION_FINGERPRINT,
                "{provider_id}"
            );
        }
    }

    fn resident_context(
        contract: &MemoryProviderContract,
        quant: Option<Quant>,
    ) -> MemoryRunContext {
        let calibration = contract.calibration.as_ref().unwrap();
        MemoryRunContext {
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
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
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
        }
    }

    #[test]
    fn prepacked_q4_without_an_override_binds_registration_to_the_actual_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, mut spec) = fixture(&tmp);
        spec.quantize = None;
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            registered_safety_check(
                &spec,
                &contract,
                &resident_context(&contract, Some(Quant::Q4))
            ),
            MemorySafetyDecision::Accept
        );
        for wrong in [None, Some(Quant::Q8)] {
            assert!(matches!(
                registered_safety_check(&spec, &contract, &resident_context(&contract, wrong)),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("does not match loaded tier")
            ));
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rung_four_declares_and_engages_staged_residency_in_the_same_request() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn base_family_declares_route_exact_decode_attention_domains() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        let routes = decode_routes("krea_2_turbo").unwrap();
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            decode.parameters.decode_tile_edges,
            routes.published_edges()
        );
        assert_eq!(
            decode.parameters.decode_overlaps,
            routes.published_overlaps()
        );
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(attention.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            attention.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );

        let pid_spec = spec.clone().with_pid(
            WeightsSource::File(root.join("pid.safetensors")),
            WeightsSource::Dir(root.clone()),
        );
        let pid_contract = memory_strategy_contract("krea_2_turbo", &pid_spec).unwrap();
        assert!(pid_contract.engages(
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedDecode
        ));

        let mut native_attention = resident_context(&pid_contract, Some(Quant::Q4));
        native_attention.selection.strategy = MemoryStrategy::BoundedAttention;
        native_attention.selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(DECODE_TILE_EDGE),
            decode_overlap: Some(DECODE_OVERLAP),
            attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
            ..Default::default()
        };
        assert_eq!(
            safety_check(
                &pid_contract,
                Precision::Bf16,
                Some(Quant::Q4),
                &native_attention
            ),
            MemorySafetyDecision::Accept,
            "loading PiD must not strip cumulative native Qwen bounded decode from use_pid=false"
        );

        let mut pid_attention = resident_context(&pid_contract, Some(Quant::Q4));
        pid_attention.use_pid = true;
        pid_attention.selection.strategy = MemoryStrategy::BoundedAttention;
        pid_attention.selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(mlx_gen_pid::DecodeRoutes::pid_edges()[0]),
            decode_overlap: Some(mlx_gen_pid::DecodeRoutes::pid_overlap()),
            attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
            ..Default::default()
        };
        assert_eq!(
            safety_check(
                &pid_contract,
                Precision::Bf16,
                Some(Quant::Q4),
                &pid_attention
            ),
            MemorySafetyDecision::Accept,
            "PiD must combine its own measured decode domain with bounded DiT attention"
        );

        let mut pid_window = pid_attention.clone();
        pid_window.selection.strategy = MemoryStrategy::BoundedTransformerResidency;
        pid_window.selection.parameters.transformer_window_size = Some(TRANSFORMER_WINDOW_SIZE);
        pid_window.selection.parameters.transformer_window_component =
            Some(TransformerComponent::Dit);
        assert_eq!(
            safety_check(&pid_contract, Precision::Bf16, Some(Quant::Q4), &pid_window),
            MemorySafetyDecision::Accept,
            "PiD decode tiling, bounded DiT attention, and block residency are independently verified request-scoped mechanisms"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sequential_deferred_snapshot_advertises_the_exact_dit_window() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        let staged = contract
            .capability(MemoryStrategy::StagedResidency)
            .unwrap();
        assert_eq!(staged.support, MemoryStrategySupport::Implemented);
        let rung = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            rung.parameters.transformer_window_sizes,
            [TRANSFORMER_WINDOW_SIZE]
        );
        assert_eq!(
            rung.parameters.transformer_window_components,
            [TransformerComponent::Dit]
        );
        std::fs::remove_dir_all(root).ok();
    }

    /// AC (SC-22662): every registered Krea 2 route — the four base routes here and the control
    /// route in `memory_strategy` — publishes the axes of the one DiT and VAE they share, derived
    /// from this crate's own config constants, and passes the shared facts conformance check.
    #[test]
    fn architecture_facts_follow_the_crate_dit_and_vae_constants() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let expected = mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: Some(48),
            head_dim: Some(128),
            transformer_blocks: Some(28),
            patch_size: Some(2),
            latent_channels: Some(16),
            vae_spatial_scale: Some(8),
            vae_temporal_scale: None,
            activation_dtype_width: Some(2),
        };
        for provider_id in [
            crate::model::KREA_2_TURBO_ID,
            crate::model::KREA_2_RAW_ID,
            crate::model::KREA_2_EDIT_ID,
            crate::model::KREA_2_TURBO_EDIT_ID,
        ] {
            let contract = weights_free_memory_strategy_contract(provider_id, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts, expected,
                "{provider_id} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        let control = crate::memory_strategy::weights_free_memory_strategy_contract(
            crate::model_control::KREA_2_TURBO_CONTROL_ID,
            &spec,
        )
        .unwrap();
        assert_eq!(control.architecture_facts, expected, "control route");
        gen_core_testkit::assert_memory_contract_facts_conform(&control);

        // The DiT's packed input width IS `latent x patch²`, so the two published axes cannot drift
        // apart from the config they came from.
        let dit = crate::config::Krea2Config::turbo();
        assert_eq!(
            crate::vae::VAE_CHANNELS as usize * dit.patch_size * dit.patch_size,
            dit.in_channels
        );
        std::fs::remove_dir_all(root).ok();
    }

    /// The `transformer/config.json` keys `Krea2Config::from_json` reads, emitted from a config
    /// value so the fixture cannot drift from the struct it mirrors.
    fn krea_transformer_config_json(cfg: &crate::config::Krea2Config) -> serde_json::Value {
        serde_json::json!({
            "in_channels": cfg.in_channels,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_kv_heads,
            "attention_head_dim": cfg.attention_head_dim,
            "num_layers": cfg.num_layers,
            "intermediate_size": cfg.intermediate_size,
            "norm_eps": cfg.norm_eps,
            "axes_dims_rope": cfg.axes_dims_rope,
            "rope_theta": cfg.rope_theta,
            "timestep_embed_dim": cfg.timestep_embed_dim,
            "num_text_layers": cfg.num_text_layers,
            "num_layerwise_text_blocks": cfg.num_layerwise_text_blocks,
            "num_refiner_text_blocks": cfg.num_refiner_text_blocks,
            "text_hidden_dim": cfg.text_hidden_dim,
            "text_intermediate_size": cfg.text_intermediate_size,
            "text_num_attention_heads": cfg.text_num_attention_heads,
            "text_num_key_value_heads": cfg.text_num_kv_heads,
        })
    }

    fn spec_for_transformer_config(dir: &std::path::Path, config: &serde_json::Value) -> LoadSpec {
        let transformer = dir.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(transformer.join("config.json"), config.to_string()).unwrap();
        LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(dir.to_path_buf()))
    }

    /// AC (SC-22662, review follow-up): on the **materialized** path the DiT axes are read out of
    /// the snapshot's own `transformer/config.json` — the file `Krea2Config::from_snapshot` parses
    /// at load — rather than published from the compile-time preset. The mirror fixture agrees
    /// with the weights-free path; a fixture whose `num_layers` is mutated publishes the mutated
    /// depth, which is what the unconditional `architecture_facts()` this replaced would fail.
    ///
    /// `num_layers` is the mutated key because it is the only trunk axis a snapshot can move on
    /// its own: `Krea2Config::validate` ties `attention_head_dim` to `sum(axes_dims_rope)` and to
    /// `text_hidden_dim`, so mutating the head width alone is rejected by the parser rather than
    /// published.
    #[test]
    fn materialized_dit_axes_come_from_the_snapshot_rather_than_the_preset() {
        let preset = crate::config::Krea2Config::turbo();
        let weights_free = LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));

        let mirror = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(&spec_for_transformer_config(
                mirror.path(),
                &krea_transformer_config_json(&preset)
            )),
            architecture_facts(&weights_free),
            "a snapshot mirroring the published config must publish the preset's axes"
        );

        let mutated_dir = tempfile::tempdir().unwrap();
        let mut mutated = krea_transformer_config_json(&preset);
        mutated["num_layers"] = serde_json::json!(7);
        let mutated_facts =
            architecture_facts(&spec_for_transformer_config(mutated_dir.path(), &mutated));
        assert_eq!(
            mutated_facts.transformer_blocks,
            Some(7),
            "the materialized path must publish the snapshot's depth, not the preset's"
        );
    }

    /// Feature-end review (SC-22667, E2): a materialized snapshot whose `transformer/config.json`
    /// is present but **unparseable or invalid** must declare its trunk axes absent rather than
    /// degrade into the turbo preset. `load_transformer_with_stream` propagates that same
    /// `Krea2Config::from_snapshot` error and refuses the load, so a preset published here would
    /// describe a model this snapshot cannot produce. A *missing key* is the other case and keeps
    /// the preset, because the parser itself defaults per key and the loader builds exactly that.
    ///
    /// This is the rule the same file already states for the sibling base-config projection: a
    /// present-but-unreadable config is never degraded.
    ///
    /// Mutation that fails this: restoring `.unwrap_or_else(crate::config::Krea2Config::turbo)` —
    /// the malformed fixture then publishes the preset's trunk axes as if they had been read off
    /// the snapshot.
    #[test]
    fn an_invalid_snapshot_config_declares_the_trunk_axes_absent() {
        let preset = crate::config::Krea2Config::turbo();
        let weights_free = LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        let declared = architecture_facts(&weights_free);

        let malformed = tempfile::tempdir().unwrap();
        let transformer = malformed.path().join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(transformer.join("config.json"), b"{not json").unwrap();
        let spec = LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
            malformed.path().to_path_buf(),
        ));
        let facts = architecture_facts(&spec);
        assert_eq!(facts.attention_heads, None);
        assert_eq!(facts.head_dim, None);
        assert_eq!(facts.transformer_blocks, None);
        assert_eq!(facts.patch_size, None);
        assert_ne!(
            facts, declared,
            "an unreadable trunk config must not publish the preset"
        );
        // The decoder's own crate constants survive, so a real architecture axis is still declared.
        assert_eq!(facts.latent_channels, declared.latent_channels);
        assert_eq!(facts.vae_spatial_scale, declared.vae_spatial_scale);
        assert!(facts.has_declared_architecture_axis());
        assert!(facts.zero_valued_axes().is_empty());

        // A snapshot that merely OMITS a key keeps the preset for it: `from_snapshot` defaults per
        // key and the loader builds exactly that geometry.
        let partial_dir = tempfile::tempdir().unwrap();
        let mut partial = krea_transformer_config_json(&preset);
        partial.as_object_mut().unwrap().remove("num_layers");
        assert_eq!(
            architecture_facts(&spec_for_transformer_config(partial_dir.path(), &partial))
                .transformer_blocks,
            declared.transformer_blocks,
            "an omitted key degrades to the preset in the loader too"
        );
    }

    #[test]
    fn missing_required_component_directory_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        std::fs::remove_dir_all(root.join("text_encoder")).unwrap();
        assert!(memory_strategy_contract("krea_2_turbo", &spec).is_err());
        assert!(weights_free_memory_strategy_contract("krea_2_turbo", &spec).is_ok());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn native_required_file_rejects_missing_empty_and_corrupt_sources() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp);
        let native = root.join("native.safetensors");
        let missing = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(missing.contains("native DiT asset facts"), "{missing}");
        std::fs::write(&native, []).unwrap();
        let empty = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(empty.contains("native DiT asset facts"), "{empty}");
        std::fs::write(&native, b"corrupt").unwrap();
        let corrupt = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(corrupt.contains("native DiT asset facts"), "{corrupt}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn imported_base_component_inventory_rejects_missing_empty_and_corrupt_files() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp);
        let native = root.join("native.safetensors");
        write_native_i8_safetensors(&native);
        let spec = LoadSpec::new(WeightsSource::File(native)).with_component(
            mlx_gen::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(root.clone()),
        );

        for component in ["text_encoder", "vae"] {
            let file = root.join(component).join("model.safetensors");
            for (case, replacement) in [
                ("empty", Some(Vec::new())),
                ("corrupt", Some(b"corrupt".to_vec())),
                ("missing", None),
            ] {
                match replacement {
                    Some(bytes) => std::fs::write(&file, bytes).unwrap(),
                    None => std::fs::remove_file(&file).unwrap(),
                }
                let error = memory_strategy_contract("krea_2_turbo", &spec)
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains(component) || error.contains("safetensors"),
                    "{component}/{case}: {error}"
                );
                if component == "text_encoder" {
                    gen_core_testkit::write_encoder_contract_fixture(
                        &root.join("text_encoder"),
                        crate::model::test_encoder_contract(),
                    )
                    .expect("restore validation-complete text encoder fixture");
                } else {
                    write_minimal_safetensors(&file);
                }
            }
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn low_rank_overlay_is_admissible_but_dense_diff_patch_is_not() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, mut spec) = fixture(&tmp);
        let low_rank = root.join("low-rank.safetensors");
        std::fs::write(&low_rank, [0_u8; 8]).unwrap();
        spec.adapters = vec![AdapterSpec::new(low_rank, 1.0, AdapterKind::Lora)];
        assert!(is_streamable_spec("krea_2_turbo", &spec).unwrap());

        let diff = root.join("dense-diff.safetensors");
        write_diff_patch(&diff);
        spec.adapters = vec![AdapterSpec::new(diff, 1.0, AdapterKind::Lora)];
        assert!(!is_streamable_spec("krea_2_turbo", &spec).unwrap());
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rung_four_resolves_load_time_quantization_instead_of_the_override_presence() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);

        for (quant, bits) in [(Quant::Q4, 4), (Quant::Q8, 8)] {
            std::fs::write(
                root.join("transformer/config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let mut packed = spec.clone();
            packed.quantize = Some(quant);
            assert!(
                is_streamable_spec("krea_2_turbo", &packed).unwrap(),
                "a matching prepacked Q{bits} override is a no-op and must remain streamable"
            );
            let (contract, plan) =
                memory_strategy_contract_with_plan("krea_2_turbo", &packed).unwrap();
            assert!(contract.lifecycle.transformer_window_materialization);
            assert_eq!(plan.effective_quant, Some(quant));
            assert_eq!(plan.load_time_quant_bits, None);
            assert!(plan.streamable_transformer);
        }

        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        assert!(
            !is_streamable_spec("krea_2_turbo", &spec).unwrap(),
            "a dense snapshot requiring per-window Q4 packing must not be streamable"
        );
        let dense = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert!(!dense.lifecycle.transformer_window_materialization);
        assert_eq!(
            dense
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );

        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        let mismatch = memory_strategy_contract("krea_2_turbo", &spec)
            .unwrap_err()
            .to_string();
        assert!(
            mismatch.contains("Q8") && mismatch.contains("Q4"),
            "{mismatch}"
        );

        let mut no_override = spec.clone();
        no_override.quantize = None;
        std::fs::write(root.join("transformer/config.json"), "{ malformed").unwrap();
        let eligibility_error = is_streamable_spec("krea_2_turbo", &no_override)
            .unwrap_err()
            .to_string();
        assert!(
            eligibility_error.contains("packed quant"),
            "{eligibility_error}"
        );
        let contract_error = memory_strategy_contract("krea_2_turbo", &no_override)
            .unwrap_err()
            .to_string();
        assert!(contract_error.contains("packed quant"), "{contract_error}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn file_contract_withholds_rung_four_but_lower_level_loader_remains_reopenable() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, base) = fixture(&tmp);
        let mut resident = base.clone();
        resident.offload_policy = OffloadPolicy::Resident;
        let mut eager = base.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        let native = root.join("single.safetensors");
        write_native_i8_safetensors(&native);
        let file = LoadSpec::new(WeightsSource::File(native))
            .with_component(
                mlx_gen::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(root.clone()),
            )
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        for spec in [resident, eager] {
            let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }
        let file_contract =
            native_memory_strategy_contract_from_spec("krea_2_turbo", &file, &root, true).unwrap();
        assert_eq!(
            file_contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented,
            "the registry File source is lstat-pinned and reopened for each transformer window"
        );
        let registered = memory_strategy_contract("krea_2_turbo", &file).unwrap();
        assert_eq!(
            registered
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "a reopenable implementation is not authorization without File-specific evidence"
        );
        assert!(!registered.lifecycle.transformer_window_materialization);
        assert!(
            crate::model::native_file_streamable(&file).unwrap(),
            "explicit Sequential + Deferred execution may use the pinned File stream seam even while automatic authorization stays Missing"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn imported_file_contract_matches_the_base_loader_for_every_typed_field() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp);
        let native = root.join("single.safetensors");
        write_native_i8_safetensors(&native);
        let valid = LoadSpec::new(WeightsSource::File(native)).with_component(
            mlx_gen::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(root.clone()),
        );

        let mut precision = valid.clone();
        precision.precision = Precision::Fp32;
        let mut control = valid.clone();
        control.control = Some(WeightsSource::File(root.join("control.safetensors")));
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
        let mut unknown_component = valid.clone();
        unknown_component.components.insert(
            "unknown".into(),
            WeightsSource::File(root.join("unknown.safetensors")),
        );
        let mut missing_base = valid.clone();
        missing_base.components.clear();
        let accepted_adapter = valid.clone().with_adapters(vec![AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        )]);
        let accepted_pid = valid.clone().with_pid(
            WeightsSource::File(root.join("pid.safetensors")),
            WeightsSource::Dir(root.join("gemma")),
        );
        let accepted_deferred = valid
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);

        for (case, spec, expected) in [
            ("valid", valid.clone(), true),
            ("adapter", accepted_adapter, true),
            ("pid", accepted_pid, true),
            ("deferred", accepted_deferred, true),
            ("precision", precision, false),
            ("quantize", valid.clone().with_quant(Quant::Q4), true),
            ("control", control, false),
            ("extra_control", extra_control, false),
            ("ip_adapter", ip_adapter, false),
            ("identity", identity, false),
            ("text_encoder", text_encoder, true),
            ("unknown_component", unknown_component, false),
            ("missing_base", missing_base, false),
        ] {
            let loader = crate::model::validate_native_krea_spec(&spec, "krea_2_turbo").is_ok();
            let contract = memory_strategy_contract("krea_2_turbo", &spec).is_ok();
            assert_eq!(loader, expected, "loader validation for {case}");
            assert_eq!(contract, loader, "contract/loader parity for {case}");
        }
    }

    #[test]
    fn native_i8_contract_counts_bf16_materialization_and_omits_source_companions() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp);
        let native = root.join("native.safetensors");
        write_native_i8_safetensors(&native);
        let contract = native_memory_strategy_contract("krea_2_turbo", &native, &root).unwrap();
        let conditioning_bytes = gen_core_testkit::encoder_contract_fixture_tensor_headers(
            crate::model::test_encoder_contract(),
            None,
        )
        .unwrap()
        .into_iter()
        .map(|header| header.data_bytes)
        .sum::<u64>();
        assert_eq!(contract.asset_facts.conditioning_bytes, conditioning_bytes);
        assert_eq!(contract.asset_facts.decoder_bytes, 2);
        assert_eq!(contract.asset_facts.transformer_bytes, 2 * 64 * 2);
        assert_eq!(contract.asset_facts.base_bytes, conditioning_bytes + 258);
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "native ConvRot materialization must retain the execution-only {strategy:?} rung"
            );
        }
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing,
                "the compatibility helper intentionally models the historical eager load; registry File specs carry the reopenable lifecycle"
            );
        }
        assert!(contract.conformance_errors().is_empty());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn edit_prices_the_materialized_vision_surface_while_t2i_excludes_it() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let (t2i, _) = memory_strategy_contract_with_plan(crate::model::KREA_2_TURBO_ID, &spec)
            .expect("t2i contract");
        let (edit, _) = memory_strategy_contract_with_plan(crate::model::KREA_2_EDIT_ID, &spec)
            .expect("edit contract");
        let language_contract = crate::model::test_encoder_contract();
        let vision = language_contract
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), &root)
            .unwrap();
        let vision_bytes = projected_tensor_headers_bytes(
            &vision
                .materialized_vision_tensor_headers(
                    &crate::model::test_vision_encoder_contract(),
                    &language_contract,
                )
                .unwrap(),
            |_| ResidentProjection::Stored,
        )
        .unwrap();
        assert_eq!(
            edit.asset_facts.conditioning_bytes - t2i.asset_facts.conditioning_bytes,
            vision_bytes
        );
    }
}
