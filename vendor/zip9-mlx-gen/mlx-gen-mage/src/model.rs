//! The six Mage-Flow variants, their [`ModelDescriptor`]s, and the explicit registration
//! constants and the registered RL [`Generator`] implementation.
//!
//! ## The variant matrix
//!
//! `microsoft` publishes **six** repositories whose `transformer/`, `vae/`, `scheduler/` and
//! `text_encoder/` JSON configs are byte-identical; only the transformer *weights* and the
//! README's default `steps`/`cfg` differ. **No config flag distinguishes a variant, or even
//! generation from editing** — the edit path differs purely by input-sequence assembly, so the
//! same backbone serves both. Turbo and Edit-Turbo are full distilled checkpoints, not LoRAs.
//!
//! | id | repo | task | steps | cfg |
//! | --- | --- | --- | --- | --- |
//! | `mage_flow` | `microsoft/Mage-Flow` (RL) | gen | 20 | 5.0 |
//! | `mage_flow_base` | `microsoft/Mage-Flow-Base` | gen | 30 | 5.0 |
//! | `mage_flow_turbo` | `microsoft/Mage-Flow-Turbo` | gen | 4 | 1.0 (off) |
//! | `mage_flow_edit` | `microsoft/Mage-Flow-Edit` | edit | 30 | 5.0 |
//! | `mage_flow_edit_base` | `microsoft/Mage-Flow-Edit-Base` | edit | 30 | 5.0 |
//! | `mage_flow_edit_turbo` | `microsoft/Mage-Flow-Edit-Turbo` | edit | 4 | 1.0 (off) |
//!
//! Each variant has its own id (rather than one id plus a switch), keeping the variant part
//! of the worker's model cache key.
//!
//! All generation and edit IDs are composed into the shipped platform catalog after their owning
//! stories validated the shared production paths and checkpoint-specific defaults.
//!
//! [`mlx_gen_catalog::provider_registry`]: https://docs.rs/mlx-gen-catalog

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core::{
    adapter_stack_resident_bytes, AdapterResidencyMode, Error as CoreError,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable, MemoryMode, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, Result as CoreResult,
};
#[cfg(test)]
use mlx_gen::gen_core::{GenerationMemory, MemoryGeometry, MemoryRunOutcome, MemorySelection};
use mlx_gen::{
    Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, Result,
    WeightsSource,
};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::config::{FAMILY, MAX_SIZE, MIN_SIZE, SIZE_MULTIPLE};
use crate::pipeline::MageComponentDirs;
use crate::{resolve_gs_key, GenerationSample};
use mlx_gen::residency::{Residency, StagedHeavy};

/// The weights-free registry-conformance identity family (sc-22733, epic sc-22723 E1/E4).
///
/// A registry surface has no artifact to prove a tier against, so it publishes
/// `<STATIC_BEHAVIOR_FINGERPRINT>-<provider>-<tier>` — a namespace that can never equal a
/// [`production_calibration_fingerprint`] string, so a conformance contract is never mistaken for
/// the identity of a load a memory anchor measured.
pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "mage-flow-mlx-registry-behavior-v1";

/// The label a tier carries inside a calibration string. `None` for a tier this family does not
/// ship, so an unshipped tier is unnameable rather than collapsed onto a neighbour.
pub fn calibration_tier_label(tier: Option<Quant>) -> Option<&'static str> {
    match tier {
        None => Some("bf16"),
        Some(Quant::Q4) => Some("q4"),
        Some(Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

/// Production calibration identity table of the six Mage-Flow routes, keyed on
/// (route, PROVEN artifact tier): `mage-flow-<route>-<tier>-mlx-shared-ladder-v1`, eighteen
/// distinct cells (sc-22733, epic sc-22723 E1/E4).
///
/// This is the TABLE, not the binding: only `production_calibration_identity` — which reads the
/// tier off the component directories the loader itself opens — may turn one of these strings into
/// a contract identity. The three tiers of one route are three different resident sets and one
/// anchor cannot price all three, which is why the tier is an axis of the key; the offload policy
/// and the load shape are deliberately NOT axes (Mage never reads `LoadSpec::offload_policy`, and
/// `MemoryCalibrationIdentity::load_shape` carries the materialization axis separately).
///
/// The pre-sc-22733 single string `mage-flow-mlx-shared-ladder-2026-08-03-v1` (SC-15509: 768²/one
/// step, q4 Edit 7.714→1.794 GiB and bf16 text-to-image 16.594→1.306 GiB under the cumulative
/// rung-4 request; staging byte-identical; bounded decode luma correlation 0.996013 / 0.991627;
/// clean-warm cancellation and decode-fault recovery within 2% of the clean rung-4 peak) is
/// RETIRED rather than folded in behind the tier proof: it was published unconditionally for six
/// routes × three tiers, no memory record retains it, and the artifacts it measured predate the
/// prepacked per-tier rehosts and the split text-encoder/VAE component layout every shipped load
/// opens today, so there is no cell of this table it could truthfully name.
pub fn production_calibration_fingerprint(
    provider_id: &str,
    artifact_tier: Option<Quant>,
) -> Option<String> {
    let variant = MageVariant::from_id(provider_id)?;
    let tier = calibration_tier_label(artifact_tier)?;
    Some(format!(
        "mage-flow-{}-{tier}-mlx-shared-ladder-v1",
        variant.route_label()
    ))
}

/// The weights-free registry-conformance identity for one (provider, tier) surface: the same route
/// and tier in the [`STATIC_BEHAVIOR_FINGERPRINT`] namespace. Never a production string.
pub fn weights_free_calibration_fingerprint(
    provider_id: &str,
    tier: Option<Quant>,
) -> Option<String> {
    let variant = MageVariant::from_id(provider_id)?;
    let tier = calibration_tier_label(tier)?;
    Some(format!(
        "{STATIC_BEHAVIOR_FINGERPRINT}-{}-{tier}",
        variant.route_label()
    ))
}

/// The packed tier a resolved component directory IS, read from the marker the loader itself
/// consults (`crate::pipeline::load_time_quant_bits` → `mlx_gen::quant::packed_quant_bits_at`, the
/// `quantization.bits` manifest the converter writes into `<component>/config.json`).
///
/// `Ok(None)` is a dense (bf16) component. An unreadable or malformed marker is an error, which the
/// identity binding turns into a WITHHELD identity rather than a failed load.
fn packed_component_tier(dir: &Path) -> Result<Option<Quant>> {
    Ok(match mlx_gen::quant::packed_quant_bits_at(dir)? {
        None => None,
        Some(4) => Some(Quant::Q4),
        Some(8) => Some(Quant::Q8),
        Some(bits) => {
            return Err(Error::Unsupported(format!(
                "mage_flow: {} declares an unshipped {bits}-bit packing",
                dir.display()
            )))
        }
    })
}

/// The identity a LOADED Mage route publishes, bound to the artifact it opens.
///
/// Two gates, both fail-closed and both withholding rather than failing the load
/// (`assemble` propagates `memory_strategy_contract_for_resolved_components`'s error, so an
/// escaping error would turn a loadable snapshot into a refused one):
///
/// * the DiT and the text encoder must carry the SAME packed marker — a split-layout load whose
///   two heavy components disagree on their tier is not a shipped cell; and
/// * `spec.quantize` must be the tier the artifact already is. This engine quantizes DENSE AT LOAD
///   (`crate::pipeline::load_heavy_components`), so a dense snapshot opened with a `Q4`/`Q8` knob —
///   an imported fine-tune, an upstream flat snapshot — is a genuine runtime requantization whose
///   load peak no anchor measured, and publishes nothing. A packed artifact opened at its own tier
///   is the checked no-op the worker performs (`load_time_quant_bits` returns `None`); a packed
///   artifact opened at ANY other knob never reaches this point, because the loader refuses it.
///
/// The request knob therefore never reaches a string on its own: for every load that succeeds, the
/// tier in the string is the tier on disk. Adapters are allowed: Mage installs them as forward-time
/// residuals and the shared contract builder prices the stack into the predicted peak as its own
/// resident component, so the base cell's identity is unchanged by them.
fn production_calibration_identity(
    provider_id: &str,
    spec: &LoadSpec,
    dirs: &MageComponentDirs,
) -> Option<MemoryCalibrationIdentity> {
    let artifact_tier = packed_component_tier(&dirs.transformer).ok()?;
    if packed_component_tier(&dirs.text_encoder).ok()? != artifact_tier {
        return None;
    }
    if artifact_tier != spec.quantize {
        return None;
    }
    production_calibration_fingerprint(provider_id, artifact_tier)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
}
/// Only the physically exercised 768→512 output-pixel tiling cell is publishable. Wider candidates
/// are intentionally absent until Mage-specific real-weight measurement exists for them.
pub const DECODE_TILE_EDGES: &[u32] = &[512];
pub const DECODE_OVERLAP: u32 = 256;
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];

/// Resolve the exact Mage route a memory declaration is being built for.
///
/// A declaration that cannot name its route is not a declaration: it would publish one of the six
/// ladders under an id no loader serves. Every contract entry point resolves through here so the
/// published surface and the loaded generator agree on which of the six checkpoints is in play.
pub fn variant_for(provider_id: &str) -> CoreResult<MageVariant> {
    MageVariant::from_id(provider_id)
        .ok_or_else(|| CoreError::Unsupported(format!("unknown Mage-Flow provider {provider_id}")))
}

/// Authenticate every [`LoadSpec`] axis a Mage memory route is allowed to carry.
///
/// Mage has no control branch, no IP-Adapter, no PiD decoder, no InstantID identity stack and no
/// externally supplied text encoder — [`load`] never reads those fields, so a spec that sets one is
/// asking for a route this engine does not implement. Declaring the ladder anyway would publish a
/// contract that admission could select and no load could honour, which is exactly the
/// declaration-without-reachability defect this route family exists to close. Adapters stay
/// allowed: they are forward-time residuals Mage genuinely installs, and the shared contract
/// builder already sizes the adapter stack into the predicted peak.
pub fn validate_load_contract(provider_id: &str, spec: &LoadSpec) -> CoreResult<MageVariant> {
    let variant = variant_for(provider_id)?;
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: Mage-Flow memory routes require a snapshot directory, not a single file"
        )));
    }
    if spec.precision != Precision::Bf16
        || !matches!(spec.quantize, None | Some(Quant::Q4) | Some(Quant::Q8))
    {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: Mage-Flow memory routes execute the bf16, Q4 and Q8 tiers only"
        )));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: Mage-Flow memory routes do not load control, IP-Adapter, PiD, identity \
             or external text-encoder components"
        )));
    }
    mlx_gen::gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, FAMILY)?;
    Ok(variant)
}

/// Build the eager Mage shared-memory contract. Every executable eager rung is declared here;
/// snapshot-backed language-model and DiT block windows additionally require the deferred load
/// shape exposed by [`memory_strategy_contract_for_spec`].
///
/// Handed no load identity at all, this publishes the weights-free conformance identity for
/// `tier` (sc-22733): there is no artifact here to prove a production tier against.
pub fn memory_strategy_contract(provider_id: &str, tier: Option<Quant>) -> MemoryProviderContract {
    let load_shape = mlx_gen::LoadShape::EagerMaterialization;
    memory_strategy_contract_with_adapters(
        provider_id,
        None,
        &[],
        Default::default(),
        load_shape,
        false,
        weights_free_calibration_fingerprint(provider_id, tier)
            .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, load_shape)),
    )
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
///
/// This is the legacy single-`LoadSpec` conformance probe. It derives streamability by probing the
/// caller's (absent) snapshot, so it can only ever witness the dense shape;
/// [`weights_free_memory_surface_contract`] is the authoritative finite surface for generated
/// capability facts because it reads the selector's already-resolved artifact tier instead.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    let streamable = streamable_spec(spec)?;
    // No artifact is opened on this probe, so the tier is the caller's `quantize` knob — which is
    // exactly why this is a conformance identity and never a production one.
    let calibration = weights_free_calibration_fingerprint(provider_id, spec.quantize)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    Ok(memory_strategy_contract_with_adapters(
        provider_id,
        Some(spec),
        &spec.adapters,
        Default::default(),
        spec.load_shape,
        streamable,
        calibration,
    ))
}

/// Whether the selector's already-resolved catalog artifact reaches Mage's rung-4 windows.
///
/// **The resolved tier is an output fact, not a load-time conversion request.** Every shipped Mage
/// tier is prepacked under `<variant snapshot>/<tier>/`, so a Q4 or Q8 install reaches production
/// with a matching component marker and [`crate::pipeline::load_time_quant_bits`] returns `None` —
/// the same dense-equivalent shape the bf16 tier presents. Deriving streamability by probing the
/// weights-free fixture path instead reports "needs load-time quantization" for Q4/Q8 and erases
/// both shipped tiers from the published ladder, which is precisely how Mage came to look like it
/// had no rung 4 at all.
///
/// **Mage never reads [`LoadSpec::offload_policy`]** — `assemble` builds a
/// [`Residency::request_scoped`] pipeline and staging/streaming are selected per request by
/// `GenerationRequest::memory`. Both offload policies therefore reach the same ladder, and
/// restricting the declaration to `Sequential` would under-report a rung the Resident-policy route
/// genuinely engages. The load shape is the real gate: only
/// [`mlx_gen::LoadShape::DeferredMaterialization`] leaves the snapshot reopenable for bounded
/// text/DiT residency.
fn surface_streamable(surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec) -> bool {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    matches!(
        surface.resolved_artifact_tier(),
        MemoryContractSurfaceTier::Bf16
            | MemoryContractSurfaceTier::Q4
            | MemoryContractSurfaceTier::Q8
    ) && surface.spec.load_shape == mlx_gen::LoadShape::DeferredMaterialization
        && surface.spec.adapters.is_empty()
        && matches!(surface.spec.weights, WeightsSource::Dir(_))
}

/// Reject a surface whose declared selector disagrees with the `LoadSpec` it ships.
///
/// The selector is what downstream capability facts record. If the two could drift, the published
/// tier would be a label rather than a fact about the artifact the route resolves.
fn surface_selector_matches_spec(
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<()> {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    let tier_matches = match surface.resolved_artifact_tier() {
        MemoryContractSurfaceTier::Bf16 => {
            surface.spec.precision == Precision::Bf16 && surface.spec.quantize.is_none()
        }
        MemoryContractSurfaceTier::Q4 => surface.spec.quantize == Some(Quant::Q4),
        MemoryContractSurfaceTier::Q8 => surface.spec.quantize == Some(Quant::Q8),
        MemoryContractSurfaceTier::Nvfp4 => false,
    };
    if tier_matches
        && surface.selector.offload_policy == surface.spec.offload_policy
        && surface.selector.load_shape == surface.spec.load_shape
    {
        Ok(())
    } else {
        Err(CoreError::Unsupported(format!(
            "Mage-Flow memory surface selector '{}' does not match its registry LoadSpec",
            surface.selector.id()
        )))
    }
}

/// Resolve one finite registry surface from the selector's explicit artifact tier.
///
/// This is the authoritative declaration seam for all six Mage routes: it publishes the complete
/// ladder, including rung 4, on exactly the selectors the engine reaches, without opening weights.
pub fn weights_free_memory_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<MemoryProviderContract> {
    surface_selector_matches_spec(surface)?;
    validate_load_contract(provider_id, &surface.spec)?;
    // The selector's already-resolved artifact tier is the tier this surface names
    // (`surface_selector_matches_spec` has just proven the spec agrees with it); the identity is
    // still the weights-free one, because no artifact was opened to measure anything.
    let tier = match surface.resolved_artifact_tier() {
        mlx_gen::gen_core::MemoryContractSurfaceTier::Bf16 => None,
        mlx_gen::gen_core::MemoryContractSurfaceTier::Q4 => Some(Quant::Q4),
        mlx_gen::gen_core::MemoryContractSurfaceTier::Q8 => Some(Quant::Q8),
        mlx_gen::gen_core::MemoryContractSurfaceTier::Nvfp4 => Some(Quant::Nvfp4),
    };
    let calibration = weights_free_calibration_fingerprint(provider_id, tier)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, surface.spec.load_shape));
    Ok(memory_strategy_contract_with_adapters(
        provider_id,
        Some(&surface.spec),
        &surface.spec.adapters,
        Default::default(),
        surface.spec.load_shape,
        surface_streamable(surface),
        calibration,
    ))
}

fn adapters_have_diff_patch(specs: &[mlx_gen::AdapterSpec]) -> bool {
    specs.iter().any(|spec| {
        mlx_gen::gen_core::weightsmeta::CheckpointMeta::from_file(&spec.path)
            .map(|meta| mlx_gen::adapters::loader::has_diff_patch_key_names(meta.keys()))
            .unwrap_or(false)
    })
}

fn streamable_spec(spec: &LoadSpec) -> CoreResult<bool> {
    if spec.load_shape != mlx_gen::LoadShape::DeferredMaterialization
        || adapters_have_diff_patch(&spec.adapters)
    {
        return Ok(false);
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(false);
    };
    let dirs = resolve_component_dirs(root, spec)?;
    streamable_resolved_components(spec, &dirs)
}

/// Whether the exact component directories selected by the loader can be reopened for bounded
/// text/transformer residency.  Keeping this seam directory-based is load-bearing for imported
/// fine-tunes: their `spec.weights` is already the transformer component directory, whereas the
/// ordinary snapshot resolver would incorrectly append another `transformer/` child.
fn streamable_resolved_components(spec: &LoadSpec, dirs: &MageComponentDirs) -> CoreResult<bool> {
    if spec.load_shape != mlx_gen::LoadShape::DeferredMaterialization
        || adapters_have_diff_patch(&spec.adapters)
    {
        return Ok(false);
    }
    let bits = spec.quantize.map(Quant::bits);
    Ok(
        crate::pipeline::load_time_quant_bits(&dirs.text_encoder, bits, "mage_flow")?.is_none()
            && crate::pipeline::load_time_quant_bits(&dirs.transformer, bits, "mage_flow")?
                .is_none(),
    )
}

/// Build the load-exact Mage contract. Mage installs every adapter as a forward-time residual after
/// quantization, so a fully sizeable stack is independently resident and is part of the predicted
/// peak. An unreadable stack stays undeclared; the consumer can distinguish that evidence gap from
/// an adapter-free load and fail closed.
pub fn memory_strategy_contract_for_spec(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "mage_flow memory facts require a snapshot directory".to_owned(),
        ));
    };
    let dirs = resolve_component_dirs(root, spec)?;
    memory_strategy_contract_for_resolved_components(provider_id, spec, &dirs)
}

/// Build the load-exact contract from the same already-resolved directories the executable loader
/// consumes.  This avoids reinterpreting an imported TransformerDirectory as a snapshot root.
fn memory_strategy_contract_for_resolved_components(
    provider_id: &str,
    spec: &LoadSpec,
    dirs: &MageComponentDirs,
) -> CoreResult<MemoryProviderContract> {
    // `assemble` builds the loaded generator's contract here, so authenticating the spec on this
    // path is what keeps a loaded Mage generator from exposing a ladder the declaration surface
    // would refuse.
    validate_load_contract(provider_id, spec)?;
    let project =
        |path: &Path, select: &dyn Fn(&str) -> bool, apply_floor: bool| -> CoreResult<u64> {
            projected_safetensors_bytes(path, |tensor| {
                let Some(quant) = spec.quantize else {
                    return ResidentProjection::Stored;
                };
                let Some(base) = tensor.name.strip_suffix(".weight") else {
                    return ResidentProjection::Stored;
                };
                if !select(base) {
                    return ResidentProjection::Stored;
                }
                ResidentProjection::GroupQuantized {
                    bits: if apply_floor {
                        crate::convert::quant_floor_bits(base, quant.bits())
                    } else {
                        quant.bits()
                    },
                    group_size: crate::quant::GROUP_SIZE as usize,
                }
            })
        };
    let components = mlx_gen::PerComponentBytes {
        text_encoder: project(&dirs.text_encoder, &crate::convert::is_te_target, true)?,
        dit: project(&dirs.transformer, &crate::convert::is_dit_target, true)?,
        vae: project(&dirs.vae, &|_| false, false)?,
    };
    let streamable = streamable_resolved_components(spec, dirs)?;
    Ok(memory_strategy_contract_with_adapters(
        provider_id,
        Some(spec),
        &spec.adapters,
        components,
        spec.load_shape,
        streamable,
        production_calibration_identity(provider_id, spec, dirs),
    ))
}

/// Architecture axes shared by all six registered Mage-Flow routes (epic SC-22657, E2).
///
/// [`MageFlowConfig::mage_flow`](crate::config::MageFlowConfig::mage_flow) is this crate's mirror of
/// the `transformer/config.json` shipped by every `microsoft/Mage-Flow*` repo, and
/// `MageFlowConfig::from_transformer_config_json` parses that same file at load; the six routes
/// (base/turbo x t2i/edit) differ only in weights and sampling profile, so they publish one set of
/// axes.
///
/// `patch_size` is 1 — Mage flattens token-per-latent-cell with no patchify, so this is a real
/// declared value and not a stand-in for an absent axis. `vae_temporal_scale` stays `None`:
/// Mage-Flow is an image model whose autoencoder has no temporal axis.
///
/// When `spec` names a materialized snapshot directory, [`dit_config`] re-runs the loader's own
/// `from_transformer_config_json` parse over that snapshot's `transformer/config.json`, so the
/// published trunk axes are the snapshot's rather than the preset's. `None` is the eager
/// [`memory_strategy_contract`] entry point, which is handed no load identity at all and therefore
/// has no snapshot to read: it publishes the preset, exactly as it did before.
///
/// SC-22667: a materialized snapshot whose config is **present but unparseable or partial** no
/// longer degrades into the preset. `MageTransformer::load` propagates that same parse failure —
/// every key is required precisely so a partial file cannot half-default into this model's numbers
/// — so a preset published here would describe a model this load will never build. A provider falls
/// back to a preset only where the LOADER falls back to that preset; otherwise the axis is declared
/// absent. The two latent axes below survive because they are the loader's own crate constants, not
/// config reads, so the contract still declares a real architecture axis.
fn architecture_facts(spec: Option<&LoadSpec>) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let Some(dit) = dit_config(spec) else {
        return mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: None,
            head_dim: None,
            transformer_blocks: None,
            patch_size: None,
            latent_channels: mlx_gen::architecture_facts::axis(crate::config::LATENT_CHANNELS),
            vae_spatial_scale: mlx_gen::architecture_facts::axis(
                crate::config::VAE_DOWNSAMPLE_FACTOR,
            ),
            vae_temporal_scale: None,
            activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
        };
    };
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_heads),
        // The exactness-gated helper, not `axis(dit.head_dim())`: `head_dim()` is a plain
        // `hidden_size / num_heads`, and a rounded quotient would invent a width a non-uniform
        // stack does not have.
        head_dim: mlx_gen::architecture_facts::head_dim(dit.hidden_size, dit.num_heads),
        // `DEPTH_SINGLE_BLOCKS` is 0: every block is dual-stream, so `depth` IS the trunk.
        transformer_blocks: mlx_gen::architecture_facts::axis(
            dit.depth.saturating_add(crate::config::DEPTH_SINGLE_BLOCKS),
        ),
        patch_size: mlx_gen::architecture_facts::axis(dit.patch_size),
        latent_channels: mlx_gen::architecture_facts::axis(crate::config::LATENT_CHANNELS),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(crate::config::VAE_DOWNSAMPLE_FACTOR),
        vae_temporal_scale: None,
        // `pipeline.rs` builds the latent tokens and loads the VAE at `Dtype::Bfloat16`.
        activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
    }
}

/// The trunk geometry a contract should describe: the materialized snapshot's own
/// `transformer/config.json` when one exists, else this crate's mirror of it.
///
/// Both snapshot layouts are accepted, because both are loadable: the pipeline's
/// `<root>/transformer/config.json`, and a caller-owned fine-tuned transformer directory whose
/// own `config.json` sits at the root (`is_finetuned_transformer_dir`).
///
/// `from_transformer_config_json` requires every consumed key and pins the ones this port
/// hardcodes, so a partial or divergent file declines rather than half-defaulting into this
/// model's numbers.
///
/// `None` means: a snapshot IS materialized and none of its accepted layouts yielded a config this
/// crate can parse. That is exactly the case `MageTransformer::load` refuses, so the caller declares
/// the trunk axes absent rather than substituting the preset (SC-22667). With no materialized
/// snapshot at all there is no load to describe and the preset is returned, which is what the
/// registry contract surface has always published.
fn dit_config(spec: Option<&LoadSpec>) -> Option<crate::config::MageFlowConfig> {
    let Some(root) = spec.and_then(mlx_gen::architecture_facts::materialized_root) else {
        return Some(crate::config::MageFlowConfig::mage_flow());
    };
    [root.join("transformer"), root.to_path_buf()]
        .into_iter()
        .find_map(|dir| {
            let json =
                std::fs::read_to_string(dir.join(crate::transformer::TRANSFORMER_CONFIG_FILE))
                    .ok()?;
            crate::config::MageFlowConfig::from_transformer_config_json(&json).ok()
        })
}

fn memory_strategy_contract_with_adapters(
    provider_id: &str,
    spec: Option<&LoadSpec>,
    adapters: &[mlx_gen::AdapterSpec],
    components: mlx_gen::PerComponentBytes,
    load_shape: mlx_gen::LoadShape,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let mut variables = vec![
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    let adapter_bytes = adapter_stack_resident_bytes(adapters, AdapterResidencyMode::Additive);
    contract.formula = if let Some(adapter_bytes) = adapter_bytes.filter(|bytes| *bytes > 0) {
        variables.push(MemoryFormulaVariable::OverlayBytes);
        contract.asset_facts.overlay_bytes = adapter_bytes;
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components: vec![MemoryResidentComponent {
                id: "adapter_stack".to_owned(),
                kind: MemoryComponentKind::AdapterStack,
                resident_bytes: adapter_bytes,
                bounded_by: None,
                residency: MemoryComponentResidency::WholeRender,
            }],
        }
    } else {
        MemoryFormulaKind::PhaseEnvelope { phases, variables }
    };
    contract.load_shape = load_shape;
    contract.phase_facts = Some(mlx_gen::gen_core::MemoryPhaseFacts::staged(
        mlx_gen::gen_core::StagedWeightSchedule::ThreeStage,
    ));
    contract.architecture_facts = architecture_facts(spec);
    // Decided by the caller, never here: a loaded route binds its identity to the artifact it
    // opened (`production_calibration_identity`), a weights-free surface publishes the conformance
    // family, and a withheld identity stays withheld (sc-22733).
    contract.calibration = calibration;
    // Mage's loaded resident generator uses sequential defaults internally. An explicit shared
    // Resident selection must therefore carry an all-disabled memory block to override them.
    contract.resident_request_memory = mlx_gen::gen_core::ResidentRequestMemory::ExplicitResident;
    contract.asset_facts.conditioning_bytes = components.text_encoder;
    contract.asset_facts.transformer_bytes = components.dit;
    contract.asset_facts.decoder_bytes = components.vae;
    contract.asset_facts.base_bytes = components
        .text_encoder
        .saturating_add(components.dit)
        .saturating_add(components.vae);
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident
            | MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedDecode
            | MemoryStrategy::BoundedAttention => {
                mlx_gen::gen_core::MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                mlx_gen::gen_core::MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency => {
                mlx_gen::gen_core::MemoryStrategySupport::Missing
            }
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedDecode => mlx_gen::gen_core::MemoryParameterRanges {
                decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                decode_overlaps: vec![DECODE_OVERLAP],
                ..Default::default()
            },
            MemoryStrategy::BoundedAttention => mlx_gen::gen_core::MemoryParameterRanges {
                attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                ..Default::default()
            },
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                mlx_gen::gen_core::MemoryParameterRanges {
                    transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                    transformer_window_components: vec![
                        mlx_gen::gen_core::TransformerComponent::Both,
                    ],
                    ..Default::default()
                }
            }
            _ => Default::default(),
        };
    }
    contract.lifecycle = mlx_gen::gen_core::MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: streamable,
    };
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    contract
}

/// Which published checkpoint a registered id serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MageVariant {
    /// `microsoft/Mage-Flow` — the Diffusion-NFT RL checkpoint, 20 steps.
    Rl,
    /// `microsoft/Mage-Flow-Base` — the training target, 30 steps.
    Base,
    /// `microsoft/Mage-Flow-Turbo` — Decoupled-DMD distilled, 4 steps, CFG off.
    Turbo,
    /// `microsoft/Mage-Flow-Edit` — instruction editing, 30 steps.
    Edit,
    /// `microsoft/Mage-Flow-Edit-Base` — instruction editing, 30 steps.
    EditBase,
    /// `microsoft/Mage-Flow-Edit-Turbo` — distilled instruction editing, 4 steps, CFG off.
    EditTurbo,
}

impl MageVariant {
    /// Registry id — always prefixed with [`FAMILY`].
    pub const fn id(self) -> &'static str {
        match self {
            Self::Rl => "mage_flow",
            Self::Base => "mage_flow_base",
            Self::Turbo => "mage_flow_turbo",
            Self::Edit => "mage_flow_edit",
            Self::EditBase => "mage_flow_edit_base",
            Self::EditTurbo => "mage_flow_edit_turbo",
        }
    }

    /// The upstream Hugging Face repository this variant's weights come from.
    pub const fn upstream_repo(self) -> &'static str {
        match self {
            Self::Rl => "microsoft/Mage-Flow",
            Self::Base => "microsoft/Mage-Flow-Base",
            Self::Turbo => "microsoft/Mage-Flow-Turbo",
            Self::Edit => "microsoft/Mage-Flow-Edit",
            Self::EditBase => "microsoft/Mage-Flow-Edit-Base",
            Self::EditTurbo => "microsoft/Mage-Flow-Edit-Turbo",
        }
    }

    /// The route label a calibration string carries: the registry id with `_` spelled `-`
    /// (`mage-flow`, `mage-flow-edit-turbo`, …). A `const` table rather than a runtime
    /// `replace`, so the six labels are visible in one place.
    pub const fn route_label(self) -> &'static str {
        match self {
            Self::Rl => "mage-flow",
            Self::Base => "mage-flow-base",
            Self::Turbo => "mage-flow-turbo",
            Self::Edit => "mage-flow-edit",
            Self::EditBase => "mage-flow-edit-base",
            Self::EditTurbo => "mage-flow-edit-turbo",
        }
    }

    /// `true` for the instruction-editing checkpoints, which consume reference images.
    pub const fn is_edit(self) -> bool {
        matches!(self, Self::Edit | Self::EditBase | Self::EditTurbo)
    }

    /// `true` for the Decoupled-DMD distilled checkpoints (4 steps, CFG off).
    pub const fn is_distilled(self) -> bool {
        matches!(self, Self::Turbo | Self::EditTurbo)
    }

    /// Published default step count, used when a request omits `steps`.
    pub const fn default_steps(self) -> u32 {
        match self {
            Self::Rl => 20,
            Self::Base | Self::Edit | Self::EditBase => 30,
            Self::Turbo | Self::EditTurbo => 4,
        }
    }

    /// Published default guidance scale. The distilled variants default to **1.0**, at which the
    /// reference builds no unconditional branch at all (`pipeline.py:326`, `:535`) — so CFG is
    /// genuinely off, not merely weightless.
    pub const fn default_cfg(self) -> f32 {
        if self.is_distilled() {
            1.0
        } else {
            5.0
        }
    }

    /// Every variant, in registration order.
    pub const ALL: [MageVariant; 6] = [
        Self::Rl,
        Self::Base,
        Self::Turbo,
        Self::Edit,
        Self::EditBase,
        Self::EditTurbo,
    ];

    /// Inverse of [`Self::id`]. Declaration surfaces are keyed by provider id, so every memory
    /// route resolves its variant here rather than accepting an unrecognised id and publishing a
    /// ladder nothing loads.
    pub fn from_id(provider_id: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|variant| variant.id() == provider_id)
    }
}

/// Every registered Mage-Flow id, in registration order.
pub const MODEL_IDS: [&str; 6] = [
    "mage_flow",
    "mage_flow_base",
    "mage_flow_turbo",
    "mage_flow_edit",
    "mage_flow_edit_base",
    "mage_flow_edit_turbo",
];

/// Maximum homogeneous output count exposed through the platform request surface.
pub const MAX_COUNT: u32 = 8;

/// Immutable upstream revision used to establish the Turbo checkpoint fingerprint.
pub const TURBO_SNAPSHOT_REVISION: &str = "8523c9d1ae3cbe2148241e4769c918d0ab158ef8";
/// Immutable upstream revision used to establish the Base checkpoint fingerprint.
pub const BASE_SNAPSHOT_REVISION: &str = "59a9cfd58cf6ecef28245852c6bdace3f12428a2";
/// Immutable upstream revision used to establish the Edit-Base checkpoint fingerprint.
pub const EDIT_BASE_SNAPSHOT_REVISION: &str = "8654a7bc0283ab2946385230b5b2eb944e0b76ea";
/// Immutable upstream revision used to establish the Edit-Turbo checkpoint fingerprint.
pub const EDIT_TURBO_SNAPSHOT_REVISION: &str = "14427bd7627d3a25436497a5939e1096f6a0d523";
/// Immutable upstream revision used to establish the primary Edit checkpoint fingerprint.
pub const EDIT_SNAPSHOT_REVISION: &str = "b01d524f86498b7dabcc4b3572c6d264d786a16e";
// Every identity tensor is a **bias**, deliberately (sc-14980). Biases are never quantized, so a
// bias fingerprint is byte-identical in the dense flat snapshot and in every pre-quantized
// `<tier>/transformer/` artifact — one pinned hash per variant verifies all three tiers, with no
// per-tier constant and no weakening of the check on the packed path.
//
// Turbo previously pinned `img_in.weight`, which the Q4/Q8 packs rewrite into u32 codes; that hash
// could not survive a tier artifact. `transformer_blocks.0.attn.add_k_proj.bias` replaces it and is
// strictly stronger: measured over all six published checkpoints its first 4096 bytes yield **six
// distinct** digests, so this one tensor discriminates every variant (`img_in.bias`, by contrast,
// collides Base with RL and Edit-Base with Edit).
const TURBO_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const BASE_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const EDIT_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const TURBO_IDENTITY_BYTES: usize = 4096;
const BASE_IDENTITY_BYTES: usize = 4096;
const EDIT_IDENTITY_BYTES: usize = 4096;
const TURBO_IDENTITY_SHA256: &str =
    "52d3e3d2bcbb655f4575b71757081da3406dd13e5c58ef73173e070ff1c4767f";
const BASE_IDENTITY_SHA256: &str =
    "c6597b08e4efe45f7bbb5d2470c68e7975d71ca26dce13a1fb34db18ca6a9e3e";
const EDIT_BASE_IDENTITY_SHA256: &str =
    "bb53a04c20e5df443bb093c3f24027f9391f6d65e3edd60ed96546b050db717b";
const EDIT_TURBO_IDENTITY_SHA256: &str =
    "d387be05845ea0e0fc6b2bec5c05bccb3808c25a0123d9e2b3459e2e7f9705df";
const EDIT_IDENTITY_SHA256: &str =
    "bd24b2009764136298499d60750ded8ebdfa7950981d116e9937588471b2ecab";

/// Build a variant's weights-free descriptor.
///
/// Capability fields that later stories own are left at their conservative `Default` (`false` /
/// empty) rather than pre-announced: quant tiers are sc-14046, LoRA/LoKr routing is sc-14057, and
/// the curated sampler/scheduler menus are sc-14041's once the flow-match loop exists. A
/// descriptor is a promise to the worker, so the scaffold promises only what it can point at in
/// the published configs.
pub fn descriptor_for(variant: MageVariant) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::MAGE_LATENT_SPACE),
        control_kinds: None,
        // The text encoder (8.875 GB) and VAE (0.345 GB) are BIT-IDENTICAL across all six Mage
        // variants — only the 8.232 GB DiT differs — so the SceneWorks mirrors host them once in a
        // shared components repo and stage them as caller-provisioned co-requisite dirs
        // (sc-14979): 58.65 GB for a full six-variant install instead of 105.04 GB. The DiT still
        // arrives as the base `WeightsSource::Dir`. A spec that stages neither falls back to the
        // flat published layout — see `resolve_component_dirs`.
        required_components: REQUIRED_COMPONENTS,
        id: variant.id(),
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Real CFG (`guidance_embed: false`) on the undistilled checkpoints; the distilled
            // ones run at cfg 1.0, where the reference never builds the negative branch.
            supports_negative_prompt: !variant.is_distilled(),
            supports_guidance: !variant.is_distilled(),
            conditioning: if variant.is_edit() {
                vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                ]
            } else {
                Vec::new()
            },
            // LoRA and LoKr both install through the one strict seam
            // ([`crate::adapters::apply_mage_adapters`] → `apply_adapters_strict`), applied in
            // [`assemble`] for EVERY variant — the adapter host is `MageTransformer`, which the
            // edit and generate variants share verbatim. Stated as engine capability, not product
            // exposure: which variants a user may attach an adapter to is decided by the catalog
            // manifest's `loraCompatibility` and the router, not here (sc-15328).
            supports_lora: true,
            supports_lokr: true,
            // Q4/Q8 tiers are sc-14046; `&[]` means dense-only, which is what the scaffold is.
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: crate::quant::COMPONENT_PRECISION_FLOORS,
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            // A platform request has one geometry/prompt and `count` independent seeds. The
            // pipeline additionally exposes heterogeneous geometry/prompt packs directly.
            max_count: MAX_COUNT,
            mac_only: true,
            ..Default::default()
        },
    }
}

/// Construct a Mage-Flow generator from a [`LoadSpec`].
///
/// `spec.adapters` carries LoRA/LoKr adapters to install on the DiT (sc-15328). They are applied
/// during assembly, AFTER the per-component tier quantization, through the strict shared seam
/// [`crate::adapters::apply_mage_adapters`] — stacked and mixed LoRA/LoKr, erroring rather than
/// silently dropping an unmatched target. An empty `adapters` is the unchanged no-adapter load.
pub fn load(variant: MageVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Unsupported(
            "mage_flow variants support bf16/Q4/Q8 checkpoints".into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "mage_flow expects a diffusers snapshot directory".into(),
            ))
        }
    };
    // A full Base fine-tune is the transformer component directory itself, not a published
    // diffusers snapshot root. The exact imported-source registry route targets `mage_flow_base`,
    // so dispatch that structural shape to the fine-tuned loader before published-checkpoint
    // identity validation. Published snapshots keep both files under `transformer/` and therefore
    // cannot collide with this gate.
    if variant == MageVariant::Base && is_finetuned_transformer_dir(root) {
        return load_finetuned(variant, spec);
    }
    match variant {
        MageVariant::Base => verify_checkpoint_identity(
            root,
            variant,
            BASE_SNAPSHOT_REVISION,
            BASE_IDENTITY_TENSOR,
            BASE_IDENTITY_BYTES,
            &[3072],
            BASE_IDENTITY_SHA256,
        )?,
        MageVariant::Turbo => verify_checkpoint_identity(
            root,
            variant,
            TURBO_SNAPSHOT_REVISION,
            TURBO_IDENTITY_TENSOR,
            TURBO_IDENTITY_BYTES,
            &[3072],
            TURBO_IDENTITY_SHA256,
        )?,
        MageVariant::EditBase => verify_checkpoint_identity(
            root,
            variant,
            EDIT_BASE_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_BASE_IDENTITY_SHA256,
        )?,
        MageVariant::Edit => verify_checkpoint_identity(
            root,
            variant,
            EDIT_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_IDENTITY_SHA256,
        )?,
        MageVariant::EditTurbo => verify_checkpoint_identity(
            root,
            variant,
            EDIT_TURBO_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_TURBO_IDENTITY_SHA256,
        )?,
        _ => {}
    }
    let dirs = resolve_component_dirs(root, spec)?;
    assemble(variant, spec, dirs)
}

fn is_finetuned_transformer_dir(root: &std::path::Path) -> bool {
    ["config.json", "diffusion_pytorch_model.safetensors"]
        .into_iter()
        .all(|name| {
            // Rehosted/Hugging Face snapshots commonly expose blob-backed symlink entries. The
            // worker has already confined the resolved transformer directory; follow each child to
            // verify the loader-visible object is a regular file instead of rejecting valid cache
            // layouts solely because their directory entry is a symlink.
            std::fs::metadata(root.join(name))
                .map(|metadata| metadata.is_file())
                .unwrap_or(false)
        })
}

/// Construct a Mage-Flow generator from a caller-owned **fine-tuned transformer** (sc-15036,
/// epic 14034 F6) — the artifact a full base fine-tune (sc-14056) writes.
///
/// Two things distinguish this from [`load`], and both are forced by what a fine-tune *is*:
///
/// 1. **`spec.weights` is the fine-tuned `transformer/` component directory itself** (a
///    `config.json` + `diffusion_pytorch_model.safetensors` pair, exactly what the trainer's
///    `save_full_checkpoint` emits), NOT a diffusers snapshot root. A training run produces the
///    DiT alone; it never re-emits the text encoder or VAE, so there is no snapshot root to point
///    at and no flat-layout sibling to fall back to. Both shared components must therefore be
///    caller-staged in [`LoadSpec::components`] — normally the installed base model's own
///    `text_encoder/` + `vae/`, which a fine-tune leaves untouched and is numerically paired with
///    by construction. A missing one is a typed error here rather than a mid-load "No such file".
/// 2. **The pinned-checkpoint identity verification is skipped.** That guard exists to catch one
///    *published* variant's snapshot staged under another published variant's id, and it works by
///    hashing a prefix of `transformer_blocks.0.attn.add_k_proj.bias`. A full fine-tune trains
///    every DiT weight including that bias, so the guard would reject the user's own trained
///    checkpoint **by construction** — it cannot distinguish "fine-tuned from Base" from "the
///    wrong published checkpoint", because the caller is the only one who knows. `variant` states
///    which published checkpoint the run started from, and with it the architecture, the
///    sampling regime (steps / CFG / distillation) and the edit-vs-generate input assembly the
///    fine-tune inherits.
///
/// Also reached by the exact `TransformerDirectory` imported-source registration. Ordinary
/// `mage_flow_base` snapshot loads remain unchanged because their config and weights live under
/// `transformer/`, while this shape has both files at the supplied root.
pub fn load_finetuned(variant: MageVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Unsupported(
            "mage_flow fine-tuned checkpoints load as bf16/Q4/Q8".into(),
        ));
    }
    // Unlike [`load`], a fine-tuned checkpoint keeps refusing adapters (sc-15328). A Mage adapter is
    // trained against, and its residual is calibrated for, the *published* base weights; a full
    // fine-tune moves every DiT weight (sc-15277 measured ~96% of `img_in.weight` changed), so
    // stacking one on top composes two independent deltas the pair was never fit for. Refused here,
    // loudly and terminally, rather than silently honoured — and the router must not queue the
    // combination in the first place.
    if !spec.adapters.is_empty() {
        return Err(Error::Unsupported(
            "mage_flow fine-tuned checkpoints cannot take LoRA/LoKr adapters: the adapter is fit \
             against the published base weights, which a full fine-tune has moved. Render the \
             adapter on the base model, or use the fine-tune without adapters."
                .into(),
        ));
    }
    let transformer = match &spec.weights {
        WeightsSource::Dir(dir) => dir.clone(),
        WeightsSource::File(file) => {
            return Err(Error::Msg(format!(
                "mage_flow: a fine-tuned checkpoint is a transformer DIRECTORY (config.json + \
                 diffusion_pytorch_model.safetensors), got the file {}",
                file.display()
            )))
        }
    };
    mlx_gen::gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, FAMILY)?;
    let staged = |id: &str| -> Result<std::path::PathBuf> {
        match spec.components.get(id) {
            Some(WeightsSource::Dir(dir)) => Ok(dir.clone()),
            Some(WeightsSource::File(file)) => Err(Error::Msg(format!(
                "mage_flow: the '{id}' component must be staged as a directory, got the file {}",
                file.display()
            ))),
            // No flat-layout fallback: a fine-tune dir has no component siblings, so silently
            // probing `<transformer>/text_encoder` would only turn a staging bug into a confusing
            // deep load failure.
            None => Err(Error::Msg(format!(
                "mage_flow: loading a fine-tuned transformer requires the '{id}' component to be \
                 staged from the installed base model — a training run produces the transformer \
                 alone"
            ))),
        }
    };
    let dirs = MageComponentDirs {
        transformer,
        text_encoder: staged(COMPONENT_TEXT_ENCODER)?,
        vae: staged(COMPONENT_VAE)?,
    };
    assemble(variant, spec, dirs)
}

/// Build the pipeline + generator from already-resolved component dirs — the half [`load`] and
/// [`load_finetuned`] share once each has decided *where* the components live.
fn assemble(
    variant: MageVariant,
    spec: &LoadSpec,
    dirs: MageComponentDirs,
) -> Result<Box<dyn Generator>> {
    // Compute the contract from the exact component directories selected above.  In particular, a
    // fine-tune's transformer is `spec.weights` itself, not `<spec.weights>/transformer`.
    let memory_strategy_contract =
        memory_strategy_contract_for_resolved_components(variant.id(), spec, &dirs)?;
    let part = if variant.is_edit() {
        crate::vae::VaePart::Both
    } else {
        crate::vae::VaePart::Decode
    };
    let text_dirs = dirs.clone();
    let heavy_dirs = dirs;
    let quant_bits = spec.quantize.map(Quant::bits);
    let adapters = spec.adapters.clone();
    let multimodal = variant.is_edit();
    let residency = Residency::request_scoped(
        move |streamable| {
            crate::pipeline::load_text_component(&text_dirs, quant_bits, multimodal, streamable)
        },
        move |_use_pid, streamable| {
            let loaded = crate::pipeline::load_heavy_components(
                &heavy_dirs,
                quant_bits,
                part,
                streamable,
                &adapters,
            )?;
            Ok(MageHeavyOwned {
                transformer: loaded.transformer,
                vae: loaded.vae,
            })
        },
    );
    Ok(Box::new(MageFlow {
        variant,
        descriptor: descriptor_for(variant),
        tier: spec.quantize,
        memory_strategy_contract,
        residency,
    }))
}

/// The caller-provisioned component ids Mage-Flow advertises (sc-14979).
pub const COMPONENT_TEXT_ENCODER: &str = "text_encoder";
/// The caller-provisioned VAE component id (sc-14979).
pub const COMPONENT_VAE: &str = "vae";
/// Both shared components, in descriptor order.
pub const REQUIRED_COMPONENTS: &[&str] = &[COMPONENT_TEXT_ENCODER, COMPONENT_VAE];

/// Resolve where each component's weights live for this load.
///
/// **Split layout (the SceneWorks mirrors, sc-14980/sc-14979).** `spec.weights` is the variant's
/// per-tier dir (`<variant snapshot>/<tier>/`), holding the DiT alone; the text encoder and VAE —
/// bit-identical across all six variants — are staged by the caller in [`LoadSpec::components`] as
/// exact component dirs resolved from the shared components mirror. Six installs cost 58.65 GB
/// instead of 105.04 GB.
///
/// **Flat layout (upstream snapshots, existing installs, arbitrary user paths).** No components are
/// staged and every component sits directly under `spec.weights`. This fallback is why the split is
/// not a breaking change: a repo/revision without tier subdirs, and every `#[ignore]`d real-weights
/// test that points at a raw `microsoft/Mage-Flow*` snapshot, keeps loading unchanged.
///
/// The two are distinguished per component, not globally, so a partially-staged spec is still
/// coherent. Unknown component ids are rejected rather than ignored.
pub(crate) fn resolve_component_dirs(root: &Path, spec: &LoadSpec) -> Result<MageComponentDirs> {
    mlx_gen::gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, FAMILY)?;
    let staged = |id: &str, fallback: &str| -> Result<std::path::PathBuf> {
        match spec.components.get(id) {
            Some(WeightsSource::Dir(dir)) => Ok(dir.clone()),
            Some(WeightsSource::File(file)) => Err(Error::Msg(format!(
                "mage_flow: the '{id}' component must be staged as a directory, got the file {}",
                file.display()
            ))),
            None => Ok(root.join(fallback)),
        }
    };
    Ok(MageComponentDirs {
        transformer: root.join("transformer"),
        text_encoder: staged(COMPONENT_TEXT_ENCODER, "text_encoder")?,
        vae: staged(COMPONENT_VAE, "vae")?,
    })
}

/// Verify bytes from a weight-bearing tensor, not a path or model-card label.
///
/// All Mage-Flow variants share byte-identical configs and tensor schemas, so those cannot detect
/// one checkpoint accidentally routed under another variant id. The caller supplies a tensor,
/// byte count, and hash pinned to the immutable upstream revision. Base deliberately uses an
/// attention bias because its `img_in.weight` prefix is byte-identical to RL's.
fn verify_checkpoint_identity(
    root: &Path,
    variant: MageVariant,
    revision: &str,
    tensor_name: &str,
    identity_bytes: usize,
    expected_shape: &[u64],
    expected_sha256: &str,
) -> Result<()> {
    let id = variant.id();
    let path = root
        .join("transformer")
        .join("diffusion_pytorch_model.safetensors");
    let mut file = std::fs::File::open(&path).map_err(|error| {
        Error::Msg(format!(
            "{id}: cannot open transformer checkpoint {}: {error}",
            path.display()
        ))
    })?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > 1_048_576 {
        return Err(Error::Msg(format!(
            "{id}: invalid safetensors header length {header_len}"
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let metadata: serde_json::Value = serde_json::from_slice(&header).map_err(|error| {
        Error::Msg(format!(
            "{id}: invalid safetensors header in {}: {error}",
            path.display()
        ))
    })?;
    let tensor = metadata
        .get(tensor_name)
        .ok_or_else(|| Error::Msg(format!("{id}: missing {tensor_name}")))?;
    if tensor.get("dtype").and_then(serde_json::Value::as_str) != Some("BF16")
        || tensor.get("shape").and_then(serde_json::Value::as_array)
            != Some(
                &expected_shape
                    .iter()
                    .map(|&dimension| serde_json::json!(dimension))
                    .collect::<Vec<_>>(),
            )
    {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} has the wrong dtype or shape"
        )));
    }
    let offsets = tensor
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Msg(format!("{id}: {tensor_name} has no data offsets")))?;
    let start = offsets.first().and_then(serde_json::Value::as_u64);
    let end = offsets.get(1).and_then(serde_json::Value::as_u64);
    let (Some(start), Some(end)) = (start, end) else {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} has invalid data offsets"
        )));
    };
    if end.saturating_sub(start) < identity_bytes as u64 {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} is too short for identity verification"
        )));
    }
    file.seek(SeekFrom::Start(8 + header_len + start))?;
    let mut bytes = vec![0u8; identity_bytes];
    file.read_exact(&mut bytes)?;
    let got = format!("{:x}", Sha256::digest(bytes));
    if got != expected_sha256 {
        return Err(Error::Msg(format!(
            "{id}: checkpoint fingerprint mismatch for {tensor_name} \
             (expected revision {revision}, got sha256 {got}); \
             another Mage-Flow checkpoint cannot serve the {id} id"
        )));
    }
    Ok(())
}

pub struct MageFlow {
    variant: MageVariant,
    descriptor: ModelDescriptor,
    tier: Option<Quant>,
    memory_strategy_contract: MemoryProviderContract,
    residency: Residency<crate::MageTextEncoder, MageHeavyOwned>,
}

pub(crate) struct MageHeavyOwned {
    transformer: crate::MageTransformer,
    vae: crate::MageVae,
}

pub(crate) struct MageLightOwned {
    vae: crate::MageVae,
}

pub(crate) struct MageDecodeView<'a> {
    vae: &'a crate::MageVae,
}

impl StagedHeavy for MageHeavyOwned {
    type Light = MageLightOwned;
    type DecodeView<'a> = MageDecodeView<'a>;

    fn shed_dit(self) -> Self::Light {
        MageLightOwned { vae: self.vae }
    }

    fn decode_view(&self) -> Self::DecodeView<'_> {
        MageDecodeView { vae: &self.vae }
    }

    fn light_view(light: &Self::Light) -> Self::DecodeView<'_> {
        MageDecodeView { vae: &light.vae }
    }
}

fn resident_request_scope(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<mlx_gen::request_scope::MlxRequestScopeCore> {
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::MageFlowConfig::mage_flow().depth,
        |_use_pid, edge, overlap| {
            if DECODE_TILE_EDGES.contains(&edge) && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(CoreError::Unsupported(format!(
                    "mage_flow: unsupported decode geometry {edge}/{overlap}"
                )))
            }
        },
    )?;
    let mut config = config;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = (context.selection.strategy
        == MemoryStrategy::BoundedTransformerResidency)
        .then_some(())
        .and(context.selection.parameters.transformer_window_size);
    Ok(mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(
        config, cleanup,
    ))
}

fn request_context_error(
    provider_id: &str,
    variant: MageVariant,
    tier: Option<Quant>,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> Option<String> {
    let expected_mode = if variant.is_edit() {
        MemoryMode::Edit
    } else {
        MemoryMode::TextToImage
    };
    let route_gate = || {
        if context.mode != expected_mode {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: request mode {:?} does not match {expected_mode:?}",
                context.mode
            )));
        }
        Ok(())
    };
    if let MemorySafetyDecision::Reject { reason } =
        mlx_gen::gen_core::standard_memory_strategy_safety_check(
            contract,
            context,
            Some(mlx_gen::gen_core::MemoryNumericTier {
                precision: Precision::Bf16,
                quant: tier,
                component_precision_floors: crate::quant::active_component_precision_floors(tier),
            }),
            Some(&route_gate),
        )
    {
        return Some(reason);
    }
    if context.budget.total_bytes == 0 {
        return Some(format!("{provider_id}: request budget is unavailable"));
    }
    let required_total_peak_bytes = ((crate::memory::generation_peak_gb(
        tier,
        context.geometry.width,
        context.geometry.height,
        context.geometry.batch,
    ) * 1_000_000_000.0)
        .round() as u64)
        .saturating_add(contract.auxiliary_resident_bytes());
    let maximum_resident_credit = contract.total_resident_bytes();
    let credited_resident_bytes =
        required_total_peak_bytes.saturating_sub(context.predicted_peak_bytes);
    if context.predicted_peak_bytes > required_total_peak_bytes
        || credited_resident_bytes > maximum_resident_credit
        || credited_resident_bytes > context.budget.committed_bytes
    {
        return Some(format!(
            "{provider_id}: caller peak {} is inconsistent with provider total {}, resident \
             envelope {}, and committed bytes {}",
            context.predicted_peak_bytes,
            required_total_peak_bytes,
            maximum_resident_credit,
            context.budget.committed_bytes
        ));
    }
    None
}

fn memory_strategy_safety_check_for(
    provider_id: &str,
    variant: MageVariant,
    tier: Option<Quant>,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    if let Some(reason) = request_context_error(provider_id, variant, tier, contract, context) {
        return MemorySafetyDecision::Reject { reason };
    }
    if context.selection.strategy != MemoryStrategy::Resident {
        return MemorySafetyDecision::Accept;
    }
    let safe_gb = match crate::memory::production_safe_budget_gb() {
        Ok(safe_gb) => safe_gb,
        Err(error) => {
            return MemorySafetyDecision::Reject {
                reason: error.to_string(),
            }
        }
    };
    match crate::memory::ensure_generation_fits(
        tier,
        context.geometry.width,
        context.geometry.height,
        context.geometry.batch,
        safe_gb,
    ) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

impl Generator for MageFlow {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.memory_strategy_contract)
    }

    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        memory_strategy_safety_check_for(
            self.descriptor.id,
            self.variant,
            self.tier,
            &self.memory_strategy_contract,
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &MemoryRunContext,
    ) -> CoreResult<Option<Box<dyn MemoryRequestScope + '_>>> {
        if let MemorySafetyDecision::Reject { reason } = self.memory_strategy_safety_check(context)
        {
            return Err(CoreError::Unsupported(reason));
        }
        Ok(Some(Box::new(resident_request_scope(
            self.descriptor.id,
            &self.memory_strategy_contract,
            context,
            mlx_gen::request_scope::MlxScopeCleanup::Device,
        )?)))
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_generation_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        if req.memory.is_none() {
            crate::memory::ensure_generation_fits(
                self.tier,
                req.width,
                req.height,
                req.count,
                crate::memory::production_safe_budget_gb()?,
            )?;
        }
        if req.cancel.is_cancelled() {
            return Err(mlx_gen::gen_core::Error::Canceled);
        }
        let steps = req.steps.unwrap_or(self.variant.default_steps());
        let cfg = req.guidance.unwrap_or(self.variant.default_cfg());
        let seed = req.seed.unwrap_or(0) as i64;
        let key = resolve_gs_key(None)?;
        let negative_prompt = req.negative_prompt.as_deref().unwrap_or(" ");
        let memory = req.memory.unwrap_or_default();
        let stage_residency = memory.stage_residency;
        let streamable = memory.stream_transformer_blocks;
        if self.variant.is_edit() {
            let references = edit_references(req)?;
            let mut images = Vec::with_capacity(req.count as usize);
            for index in 0..req.count {
                let run_seed = seed.wrapping_add(index as i64);
                let trace = self.residency.run_staged_request_scoped(
                    stage_residency,
                    streamable,
                    &req.cancel,
                    false,
                    on_progress,
                    |text| {
                        calibration_fault(req, MemoryPhase::Conditioning)?;
                        crate::pipeline::encode_edit_phase(
                            text,
                            &req.prompt,
                            negative_prompt,
                            &references,
                            cfg,
                            &req.cancel,
                        )
                    },
                    |encoded| match encoded {
                        Some(encoded) => crate::pipeline::materialize_edit_encoded(encoded),
                        None => Ok(()),
                    },
                    |heavy, encoded, progress| {
                        calibration_fault(req, MemoryPhase::Denoise)?;
                        crate::pipeline::denoise_edit_phase(
                            &heavy.transformer,
                            &heavy.vae,
                            encoded,
                            &references,
                            req.height,
                            req.width,
                            steps as usize,
                            cfg,
                            run_seed,
                            &key,
                            req.memory,
                            &req.cancel,
                            progress,
                        )
                    },
                    crate::pipeline::materialize_edit_denoised,
                    |view, denoised, progress| {
                        calibration_fault(req, MemoryPhase::Decode)?;
                        crate::pipeline::decode_edit_phase(
                            view.vae,
                            denoised,
                            req.memory,
                            &req.cancel,
                            progress,
                        )
                    },
                )?;
                if req.cancel.is_cancelled() {
                    return Err(CoreError::Canceled);
                }
                mlx_rs::transforms::eval([&trace.image_u8]).map_err(Error::from)?;
                images.push(Image {
                    width: req.width,
                    height: req.height,
                    pixels: trace
                        .image_u8
                        .try_as_slice::<u8>()
                        .map_err(|error| {
                            Error::Msg(format!(
                                "mage_flow edit: RGB8 output is not host-readable: {error}"
                            ))
                        })?
                        .to_vec(),
                });
            }
            return Ok(GenerationOutput::Images(images));
        }
        let samples = (0..req.count)
            .map(|index| GenerationSample {
                prompt: &req.prompt,
                negative_prompt,
                height: req.height,
                width: req.width,
                seed: seed.wrapping_add(index as i64),
            })
            .collect::<Vec<_>>();
        let traces = self
            .residency
            .run_staged_request_scoped(
                stage_residency,
                streamable,
                &req.cancel,
                false,
                on_progress,
                |text| {
                    calibration_fault(req, MemoryPhase::Conditioning)?;
                    crate::pipeline::encode_generation_phase(text, &samples, cfg, &req.cancel)
                },
                |encoded| match encoded {
                    Some(encoded) => crate::pipeline::materialize_generation_encoded(encoded),
                    None => Ok(()),
                },
                |heavy, encoded, progress| {
                    calibration_fault(req, MemoryPhase::Denoise)?;
                    crate::pipeline::denoise_generation_phase(
                        &heavy.transformer,
                        &samples,
                        encoded,
                        steps as usize,
                        cfg,
                        &key,
                        req.memory,
                        &req.cancel,
                        progress,
                    )
                },
                crate::pipeline::materialize_generation_denoised,
                |view, denoised, progress| {
                    calibration_fault(req, MemoryPhase::Decode)?;
                    crate::pipeline::decode_generation_phase(
                        view.vae,
                        denoised,
                        req.memory,
                        &req.cancel,
                        progress,
                    )
                },
            )?
            .samples;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled);
        }
        let mut images = Vec::with_capacity(traces.len());
        for trace in traces {
            mlx_rs::transforms::eval([&trace.image_u8]).map_err(Error::from)?;
            let pixels = trace
                .image_u8
                .try_as_slice::<u8>()
                .map_err(|e| {
                    Error::Msg(format!("mage_flow: RGB8 output is not host-readable: {e}"))
                })?
                .to_vec();
            images.push(Image {
                width: req.width,
                height: req.height,
                pixels,
            });
        }
        Ok(GenerationOutput::Images(images))
    }
}

fn calibration_fault(req: &GenerationRequest, phase: MemoryPhase) -> Result<()> {
    if req.memory.is_some_and(|memory| {
        memory.calibration_fault_harness_authorized && memory.calibration_error_phase == Some(phase)
    }) {
        return Err(Error::Msg(format!(
            "mage_flow: authorized calibration fault at {phase:?}"
        )));
    }
    Ok(())
}

fn edit_references(req: &GenerationRequest) -> Result<Vec<image::RgbImage>> {
    let mut images = Vec::new();
    for conditioning in &req.conditioning {
        match conditioning {
            Conditioning::Reference { image, .. } => images.push(image),
            Conditioning::MultiReference { images: refs } => images.extend(refs),
            _ => {}
        }
    }
    if images.is_empty() {
        return Err(Error::Msg(
            "mage_flow edit: Reference or MultiReference conditioning is required".into(),
        ));
    }
    images
        .into_iter()
        .map(|image| {
            image::RgbImage::from_raw(image.width, image.height, image.pixels.clone()).ok_or_else(
                || Error::Msg("mage_flow edit: reference image is not valid RGB8".into()),
            )
        })
        .collect()
}

fn validate_generation_request(
    descriptor: &ModelDescriptor,
    req: &GenerationRequest,
) -> mlx_gen::gen_core::Result<()> {
    descriptor
        .capabilities
        .validate_request(descriptor.id, req)?;
    if !req.width.is_multiple_of(REQUIRED_SIZE_MULTIPLE)
        || !req.height.is_multiple_of(REQUIRED_SIZE_MULTIPLE)
    {
        return Err(mlx_gen::gen_core::Error::Msg(format!(
            "mage_flow dimensions must be divisible by {REQUIRED_SIZE_MULTIPLE}"
        )));
    }
    Ok(())
}

/// Every side of a Mage-Flow request must be a multiple of this (the VAE's 16× downsample;
/// `patch_size == 1` adds no further stride). Re-exported at the model layer because SceneWorks
/// pins each advertised resolution bucket to an engine stride constant.
pub const REQUIRED_SIZE_MULTIPLE: u32 = SIZE_MULTIPLE;

/// Per-component on-disk footprint used by the worker's staged-residency fit gate.
///
/// Mage-Flow quantizes all three weight-bearing components, so the accounting must follow the
/// selected snapshot tree rather than a transformer-only approximation. Missing/unreadable
/// subdirectories contribute zero bytes here; checkpoint identity/load validation separately rejects
/// missing required components before generation.
///
/// **Resolves through [`resolve_component_dirs`], not the spec's weights root** (sc-15154). On the
/// SPLIT layout the root is the variant's per-tier dir and holds the DiT *alone* — the text encoder
/// and VAE are staged in [`LoadSpec::components`] from the shared mirror. Summing subdirs of the
/// root therefore reported the DiT's bytes as the whole model: 2.33 GB for a q4 tier whose real
/// install is 7.00 GB, and 8.23 GB for bf16's 17.46 GB. The worker's fit gate adds a flat activation
/// headroom to this number, so the shortfall surfaced as an over-budget message quoting a figure
/// that tracked neither the tier's weights nor its measured peak. The flat-layout fallback is
/// unchanged: with nothing staged, `resolve_component_dirs` returns `root/<component>` exactly as
/// before.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::gen_core::Error::Msg(
            "mage_flow: per-component footprint requires a snapshot directory, not a single \
             .safetensors file"
                .to_owned(),
        ));
    };
    let dirs = resolve_component_dirs(root, spec)?;
    Ok(mlx_gen::PerComponentBytes {
        text_encoder: mlx_gen::safetensors_path_bytes(dirs.text_encoder),
        dit: mlx_gen::safetensors_path_bytes(dirs.transformer),
        vae: mlx_gen::safetensors_path_bytes(dirs.vae),
    })
}

macro_rules! mage_registrations {
    ( $( $variant:ident => ( $descriptor_fn:ident, $load_fn:ident, $registration:ident ) ),+ $(,)? ) => {
        $(
            /// This variant's weights-free descriptor (see [`descriptor_for`]).
            pub fn $descriptor_fn() -> ModelDescriptor {
                descriptor_for(MageVariant::$variant)
            }

            fn $load_fn(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
                load(MageVariant::$variant, spec)
            }

            mlx_gen::register_generators! {
                pub const $registration = $descriptor_fn => $load_fn;
                footprint = component_footprint
            }
        )+

        /// The explicit registration constants, in variant order — the surface a catalog crate
        /// composes.
        pub const REGISTRATIONS: &[mlx_gen::registry::ModelRegistration] = &[ $($registration),+ ];
    };
}

mage_registrations! {
    Rl => (descriptor, load_rl, REGISTRATION),
    Base => (descriptor_base, load_base, REGISTRATION_BASE),
    Turbo => (descriptor_turbo, load_turbo, REGISTRATION_TURBO),
    Edit => (descriptor_edit, load_edit, REGISTRATION_EDIT),
    EditBase => (descriptor_edit_base, load_edit_base, REGISTRATION_EDIT_BASE),
    EditTurbo => (descriptor_edit_turbo, load_edit_turbo, REGISTRATION_EDIT_TURBO),
}

macro_rules! mage_memory_registration {
    ($name:ident, $behavior:ident, $variant:ident, $id:literal) => {
        pub const $name: mlx_gen::gen_core::MemoryRegistration =
            mlx_gen::gen_core::MemoryRegistration {
                provider_id: $id,
                contract: |spec| memory_strategy_contract_for_spec($id, spec),
                safety_check: |spec, contract, context| {
                    memory_strategy_safety_check_for(
                        $id,
                        MageVariant::$variant,
                        spec.quantize,
                        contract,
                        context,
                    )
                },
            };
        pub const $behavior: mlx_gen::gen_core::MemoryBehaviorRegistration =
            mlx_gen::gen_core::MemoryBehaviorRegistration {
                provider_id: $id,
                valid_fixtures: |spec, contract, strategy| {
                    registered_valid_fixture(MageVariant::$variant, spec, contract, strategy)
                },
                begin_request: |spec, contract, context| {
                    registered_begin_request($id, MageVariant::$variant, spec, contract, context)
                },
            };
    };
}

fn registered_valid_fixture(
    variant: MageVariant,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        mlx_gen::gen_core::MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: crate::quant::active_component_precision_floors(
                spec.quantize,
            ),
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: if variant.is_edit() {
                MemoryMode::Edit
            } else {
                MemoryMode::TextToImage
            },
            reference_count: u32::from(variant.is_edit()),
            use_pid: false,
            has_phases: contract.engages(strategy, MemoryStrategy::StagedResidency),
            overlay: None,
        },
    )?;
    let predicted_peak_bytes = ((crate::memory::generation_peak_gb(
        spec.quantize,
        context.geometry.width,
        context.geometry.height,
        context.geometry.batch,
    ) * 1_000_000_000.0)
        .round() as u64)
        .saturating_add(contract.auxiliary_resident_bytes());
    context.predicted_peak_bytes = predicted_peak_bytes;
    context.budget = mlx_gen::gen_core::MemoryBudget {
        total_bytes: predicted_peak_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

fn registered_begin_request(
    provider_id: &'static str,
    variant: MageVariant,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } =
        memory_strategy_safety_check_for(provider_id, variant, spec.quantize, contract, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    Ok(Some(Box::new(resident_request_scope(
        provider_id,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )?)))
}

mage_memory_registration!(
    MEMORY_REGISTRATION,
    MEMORY_BEHAVIOR_REGISTRATION,
    Rl,
    "mage_flow"
);
mage_memory_registration!(
    MEMORY_REGISTRATION_BASE,
    MEMORY_BEHAVIOR_REGISTRATION_BASE,
    Base,
    "mage_flow_base"
);
mage_memory_registration!(
    MEMORY_REGISTRATION_TURBO,
    MEMORY_BEHAVIOR_REGISTRATION_TURBO,
    Turbo,
    "mage_flow_turbo"
);
mage_memory_registration!(
    MEMORY_REGISTRATION_EDIT,
    MEMORY_BEHAVIOR_REGISTRATION_EDIT,
    Edit,
    "mage_flow_edit"
);
mage_memory_registration!(
    MEMORY_REGISTRATION_EDIT_BASE,
    MEMORY_BEHAVIOR_REGISTRATION_EDIT_BASE,
    EditBase,
    "mage_flow_edit_base"
);
mage_memory_registration!(
    MEMORY_REGISTRATION_EDIT_TURBO,
    MEMORY_BEHAVIOR_REGISTRATION_EDIT_TURBO,
    EditTurbo,
    "mage_flow_edit_turbo"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn finetuned_shape_accepts_blob_backed_symlink_entries() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let transformer = tmp.path().join("transformer");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(blobs.join("config"), b"{}").unwrap();
        std::fs::write(blobs.join("weights"), b"safetensors").unwrap();
        symlink(blobs.join("config"), transformer.join("config.json")).unwrap();
        symlink(
            blobs.join("weights"),
            transformer.join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();

        assert!(is_finetuned_transformer_dir(&transformer));
        std::fs::remove_file(blobs.join("weights")).unwrap();
        assert!(
            !is_finetuned_transformer_dir(&transformer),
            "a broken blob link must remain fail-closed"
        );
    }

    fn write_memory_safetensors(path: &Path, entries: &[(&str, &str, &[usize], usize)]) {
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in entries {
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend(json);
        file.resize(file.len() + offset, 0);
        std::fs::write(path, file).unwrap();
    }

    fn write_memory_snapshot(root: &Path) {
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_memory_safetensors(
                &dir.join("model.safetensors"),
                &[("probe", "BF16", &[1], 2)],
            );
        }
    }

    /// Stamp the packed-tier marker the converter writes (`<component>/config.json`
    /// `quantization.bits`) onto the two heavy components of a `write_memory_snapshot` root, which
    /// is what makes it a prepacked `q<bits>` tier to the loader (`pipeline::load_time_quant_bits`).
    fn mark_packed(root: &Path, bits: u32) {
        let marker = format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#);
        for component in ["text_encoder", "transformer"] {
            std::fs::write(root.join(component).join("config.json"), &marker).unwrap();
        }
    }

    /// The worker's production still-image spec over a split-layout snapshot: the tier knob on the
    /// spec, both shared components staged explicitly.
    fn tier_spec(root: &Path, quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_component(
                COMPONENT_TEXT_ENCODER,
                WeightsSource::Dir(root.join("text_encoder")),
            )
            .with_component(COMPONENT_VAE, WeightsSource::Dir(root.join("vae")));
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        spec
    }

    /// sc-22733 (epic sc-22723, E1 measurable / E4 production loader): every shipped Mage cell —
    /// six routes × three tiers — publishes its OWN production identity off the loaded contract,
    /// keyed on the route and the tier the artifact on disk IS, under both offload policies and both
    /// load shapes. Before this the crate published one string for all eighteen cells
    /// (`mage-flow-mlx-shared-ladder-2026-08-03-v1`), so no memory anchor could name the cell it
    /// measured.
    ///
    /// *Mutations this kills:* restoring the single const (eighteen strings collapse to one and the
    /// distinctness assert fails); dropping the tier from the key (collapse to six); keying on the
    /// offload policy or the load shape (the four specs per cell disagree); replaying one route's
    /// string on another; publishing the weights-free conformance string in production; and
    /// reading the tier off `spec.quantize` instead of the artifact — the last block opens a DENSE
    /// snapshot with a `Q4`/`Q8` knob (a runtime requantization) and must publish nothing, and a
    /// snapshot whose two heavy components disagree on their packing publishes nothing either.
    #[test]
    fn every_shipped_cell_publishes_its_own_production_identity_off_the_artifact_tier() {
        let mut published = std::collections::BTreeSet::new();
        for provider in MODEL_IDS {
            for (bits, quant) in [
                (None, None),
                (Some(4), Some(Quant::Q4)),
                (Some(8), Some(Quant::Q8)),
            ] {
                let tmp = tempfile::tempdir().unwrap();
                write_memory_snapshot(tmp.path());
                if let Some(bits) = bits {
                    mark_packed(tmp.path(), bits);
                }
                let expected = production_calibration_fingerprint(provider, quant).unwrap();
                let tier = calibration_tier_label(quant).unwrap();
                assert!(expected.contains(tier), "{provider} {tier}: {expected}");
                assert_ne!(
                    expected,
                    weights_free_calibration_fingerprint(provider, quant).unwrap()
                );
                assert_ne!(expected, "mage-flow-mlx-shared-ladder-2026-08-03-v1");
                for (offload, load_shape) in [
                    (
                        mlx_gen::OffloadPolicy::Resident,
                        mlx_gen::LoadShape::EagerMaterialization,
                    ),
                    (
                        mlx_gen::OffloadPolicy::Sequential,
                        mlx_gen::LoadShape::EagerMaterialization,
                    ),
                    (
                        mlx_gen::OffloadPolicy::Resident,
                        mlx_gen::LoadShape::DeferredMaterialization,
                    ),
                    (
                        mlx_gen::OffloadPolicy::Sequential,
                        mlx_gen::LoadShape::DeferredMaterialization,
                    ),
                ] {
                    let spec = tier_spec(tmp.path(), quant)
                        .with_offload_policy(offload)
                        .with_load_shape(load_shape);
                    let contract = memory_strategy_contract_for_spec(provider, &spec).unwrap();
                    let identity = contract.calibration.as_ref().unwrap_or_else(|| {
                        panic!("{provider} {tier} {offload:?} {load_shape:?} publishes none")
                    });
                    assert_eq!(identity.fingerprint, expected, "{provider} {tier}");
                    assert_eq!(identity.load_shape, load_shape);
                    assert_eq!(contract.load_shape, load_shape);
                    assert!(contract.conformance_errors().is_empty());
                }
                // The knob never outranks the artifact: every tier the snapshot is NOT.
                for wrong in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                    if wrong == quant {
                        continue;
                    }
                    let spec = tier_spec(tmp.path(), wrong);
                    // A packed artifact under the wrong knob is refused by the loader's own tier
                    // gate; only the dense-with-knob shape (runtime requantization) reaches a
                    // contract, and it must publish nothing.
                    if let Ok(contract) = memory_strategy_contract_for_spec(provider, &spec) {
                        assert!(
                            contract.calibration.is_none(),
                            "{provider} {tier} published an identity for a {wrong:?} knob"
                        );
                    }
                }
                assert!(
                    published.insert(expected.clone()),
                    "{provider} {tier} repeats another cell's identity: {expected}"
                );
            }
        }
        // Eighteen distinct strings: six routes × three tiers.
        assert_eq!(published.len(), MODEL_IDS.len() * 3);

        // Two heavy components that disagree on their packing are not a shipped cell.
        let tmp = tempfile::tempdir().unwrap();
        write_memory_snapshot(tmp.path());
        mark_packed(tmp.path(), 4);
        std::fs::write(tmp.path().join("text_encoder/config.json"), "{}").unwrap();
        let contract =
            memory_strategy_contract_for_spec("mage_flow", &tier_spec(tmp.path(), Some(Quant::Q4)))
                .unwrap();
        assert!(contract.calibration.is_none());
    }

    /// sc-22733: an unreadable tier marker WITHHOLDS the identity rather than failing the load —
    /// `assemble` propagates the contract builder's error, so an escaping error would turn a
    /// loadable snapshot into a refused one.
    ///
    /// *Mutation this kills:* `packed_component_tier(..).ok()?` → `.unwrap()` (the contract builder
    /// panics / the load fails instead of publishing a contract without an identity).
    #[test]
    fn an_unreadable_tier_marker_withholds_the_identity_without_failing_the_load() {
        let tmp = tempfile::tempdir().unwrap();
        write_memory_snapshot(tmp.path());
        mark_packed(tmp.path(), 4);
        std::fs::write(
            tmp.path().join("transformer/config.json"),
            r#"{"quantization":{"bits":"four"}}"#,
        )
        .unwrap();
        // The loader's own gate refuses this marker outright, so the contract never reaches a
        // quant knob; probe the identity binding directly with the dirs the loader would resolve.
        let dirs = MageComponentDirs {
            transformer: tmp.path().join("transformer"),
            text_encoder: tmp.path().join("text_encoder"),
            vae: tmp.path().join("vae"),
        };
        assert!(packed_component_tier(&dirs.transformer).is_err());
        assert!(production_calibration_identity(
            "mage_flow",
            &tier_spec(tmp.path(), Some(Quant::Q4)),
            &dirs
        )
        .is_none());
    }

    /// The registry-conformance surfaces publish the weights-free family, one string per
    /// (route, resolved tier), and never a production string — so a fixture contract can never be
    /// mistaken for measured evidence.
    ///
    /// *Mutation this kills:* handing `weights_free_memory_surface_contract` the production table.
    #[test]
    fn weights_free_surfaces_publish_the_conformance_family_per_tier() {
        use mlx_gen::gen_core::MemoryContractSurfaceTier;

        let mut published = std::collections::BTreeSet::new();
        for provider in MODEL_IDS {
            for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
                let quant = match surface.resolved_artifact_tier() {
                    MemoryContractSurfaceTier::Bf16 => None,
                    MemoryContractSurfaceTier::Q4 => Some(Quant::Q4),
                    MemoryContractSurfaceTier::Q8 => Some(Quant::Q8),
                    MemoryContractSurfaceTier::Nvfp4 => continue,
                };
                let contract = weights_free_memory_surface_contract(provider, &surface).unwrap();
                let identity = contract.calibration.as_ref().unwrap();
                assert_eq!(
                    identity.fingerprint,
                    weights_free_calibration_fingerprint(provider, quant).unwrap(),
                    "{provider} {}",
                    surface.selector.id()
                );
                assert!(identity
                    .fingerprint
                    .starts_with(STATIC_BEHAVIOR_FINGERPRINT));
                assert_ne!(
                    identity.fingerprint,
                    production_calibration_fingerprint(provider, quant).unwrap()
                );
                published.insert(identity.fingerprint.clone());
            }
            // The legacy single-spec probe and the eager entry point agree with the surface family.
            for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let expected = weights_free_calibration_fingerprint(provider, quant).unwrap();
                let spec = tier_spec(Path::new("/__sceneworks_memory_contract_surface__"), quant);
                let probe = weights_free_memory_strategy_contract(provider, &spec).unwrap();
                assert_eq!(probe.calibration.as_ref().unwrap().fingerprint, expected);
                let eager = memory_strategy_contract(provider, quant);
                assert_eq!(eager.calibration.as_ref().unwrap().fingerprint, expected);
            }
        }
        assert_eq!(published.len(), MODEL_IDS.len() * 3);
    }

    #[test]
    fn finetuned_contract_projects_the_supplied_transformer_directory_itself() {
        let tmp = tempfile::tempdir().unwrap();
        let transformer = tmp.path().join("trained-transformer");
        let text_encoder = tmp.path().join("shared-text-encoder");
        let vae = tmp.path().join("shared-vae");
        for dir in [&transformer, &text_encoder, &vae] {
            std::fs::create_dir_all(dir).unwrap();
            write_memory_safetensors(
                &dir.join("diffusion_pytorch_model.safetensors"),
                &[("probe", "BF16", &[1], 2)],
            );
        }
        std::fs::write(transformer.join("config.json"), "{}").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(transformer.clone()))
            .with_component(
                COMPONENT_TEXT_ENCODER,
                WeightsSource::Dir(text_encoder.clone()),
            )
            .with_component(COMPONENT_VAE, WeightsSource::Dir(vae.clone()));
        let dirs = MageComponentDirs {
            transformer: transformer.clone(),
            text_encoder,
            vae,
        };

        let contract =
            memory_strategy_contract_for_resolved_components("mage_flow_base", &spec, &dirs)
                .unwrap();

        assert_eq!(contract.asset_facts.transformer_bytes, 2);
        assert_eq!(contract.asset_facts.conditioning_bytes, 2);
        assert_eq!(contract.asset_facts.decoder_bytes, 2);
        assert!(!transformer.join("transformer").exists());
    }

    #[test]
    fn memory_strategy_contract_declares_the_executable_eager_ladder() {
        use mlx_gen::gen_core::{MemoryStrategySupport, MEMORY_CALIBRATION_ABI};

        let contract = memory_strategy_contract("mage_flow", Some(Quant::Q4));
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(contract.provider_id, "mage_flow");
        assert_eq!(
            contract
                .calibration
                .as_ref()
                .map(|identity| (identity.abi, identity.fingerprint.clone())),
            Some((
                MEMORY_CALIBRATION_ABI,
                weights_free_calibration_fingerprint("mage_flow", Some(Quant::Q4)).unwrap()
            ))
        );
        assert!(matches!(
            contract
                .capability(MemoryStrategy::Resident)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        ));
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert!(matches!(
                contract
                    .capability(strategy)
                    .map(|capability| &capability.support),
                Some(MemoryStrategySupport::Implemented)
            ));
        }
        assert!(matches!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Missing)
        ));
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );
        assert!(matches!(
            contract.backend,
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            }
        ));
    }

    #[test]
    fn deferred_contract_adds_snapshot_backed_text_and_dit_windows() {
        use mlx_gen::gen_core::{MemoryStrategySupport, TransformerComponent};
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_memory_snapshot(&root);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        let contract = memory_strategy_contract_for_spec("mage_flow", &spec).unwrap();
        let rung4 = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert!(matches!(rung4.support, MemoryStrategySupport::Implemented));
        assert_eq!(rung4.parameters.transformer_window_sizes, [1]);
        assert_eq!(
            rung4.parameters.transformer_window_components,
            [TransformerComponent::Both]
        );
        assert_eq!(
            contract.load_shape,
            mlx_gen::LoadShape::DeferredMaterialization
        );
        assert!(contract.conformance_errors().is_empty());
    }

    #[test]
    fn ladder_parameters_and_load_shape_fail_closed_under_mutation() {
        use mlx_gen::gen_core::{MemoryNumericTier, TransformerComponent};
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: crate::quant::COMPONENT_PRECISION_FLOORS,
        };
        let eager = memory_strategy_contract("mage_flow", Some(Quant::Q4));
        assert!(eager
            .representative_selection(MemoryStrategy::BoundedTransformerResidency, tier, false)
            .is_err());

        let mut deferred = eager.clone();
        deferred.load_shape = mlx_gen::LoadShape::DeferredMaterialization;
        let rung4 = deferred
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert!(matches!(
            rung4.support,
            mlx_gen::gen_core::MemoryStrategySupport::Missing
        ));

        let mut deferred = memory_strategy_contract_with_adapters(
            "mage_flow",
            None,
            &[],
            Default::default(),
            mlx_gen::LoadShape::DeferredMaterialization,
            true,
            None,
        );
        let mut selected = deferred
            .representative_selection(MemoryStrategy::BoundedTransformerResidency, tier, false)
            .unwrap();
        assert_eq!(
            deferred.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
        assert!(deferred.validate_selection(&selected).is_ok());
        selected.parameters.decode_tile_edge = Some(511);
        assert!(deferred.validate_selection(&selected).is_err());
        selected.parameters.decode_tile_edge = Some(DECODE_TILE_EDGES[0]);
        selected.parameters.attention_chunk_size = Some(1);
        assert!(deferred.validate_selection(&selected).is_err());
        selected.parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
        selected.parameters.transformer_window_size = Some(2);
        assert!(deferred.validate_selection(&selected).is_err());
        selected.parameters.transformer_window_size = Some(1);
        selected.parameters.transformer_window_component = Some(TransformerComponent::Dit);
        assert!(deferred.validate_selection(&selected).is_err());

        deferred.load_shape = mlx_gen::LoadShape::EagerMaterialization;
        assert!(deferred.validate_selection(&selected).is_err());
    }

    #[test]
    fn deferred_rung_four_fails_closed_for_load_time_quant_and_diff_patch_adapters() {
        use mlx_gen::gen_core::{MemoryStrategySupport, TransformerComponent};

        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_memory_snapshot(&root);
        for component in ["text_encoder", "transformer", "vae"] {
            std::fs::write(root.join(component).join("config.json"), "{}").unwrap();
        }
        let dense_q4 = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(Quant::Q4)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        let dense_contract = memory_strategy_contract_for_spec("mage_flow", &dense_q4).unwrap();
        assert_eq!(
            dense_contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(!dense_contract.lifecycle.transformer_window_materialization);

        let packed = r#"{"quantization":{"bits":4,"group_size":64}}"#;
        for component in ["text_encoder", "transformer"] {
            std::fs::write(root.join(component).join("config.json"), packed).unwrap();
        }
        let packed_contract = memory_strategy_contract_for_spec("mage_flow", &dense_q4).unwrap();
        let rung4 = packed_contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(rung4.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            rung4.parameters.transformer_window_components,
            [TransformerComponent::Both]
        );

        let adapter = root.join("diff-patch.safetensors");
        write_memory_safetensors(
            &adapter,
            &[("transformer_blocks.0.attn.to_q.diff", "BF16", &[1], 2)],
        );
        let diff_patch = dense_q4
            .clone()
            .with_adapters(vec![mlx_gen::AdapterSpec::new(
                adapter,
                1.0,
                mlx_gen::AdapterKind::Lora,
            )]);
        let diff_contract = memory_strategy_contract_for_spec("mage_flow", &diff_patch).unwrap();
        assert_eq!(
            diff_contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn deferred_ownership_guards_evict_resident_stacks_and_keep_architectural_depth() {
        let transformer = include_str!("transformer.rs");
        assert!(transformer.contains("self.blocks.clear();"));
        assert!(transformer.contains("if !self.blocks.is_empty()"));
        assert!(transformer.contains("BlockPlan::new(self.cfg.depth, size)"));

        let text = include_str!("text_encoder/encoder.rs");
        assert!(text.contains("self.layers.clear();"));
        assert!(text.contains("if !self.layers.is_empty()"));
        assert!(text.contains("BlockPlan::new(stream.cfg.num_layers, 1)"));
    }

    #[test]
    fn spec_contract_uses_projected_component_bytes_and_mage_q4_floors() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        for component in ["text_encoder", "transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        write_memory_safetensors(
            &root.join("transformer/model.safetensors"),
            &[
                ("norm_out.linear.weight", "BF16", &[2, 64], 256),
                ("blocks.0.proj.weight", "BF16", &[2, 64], 256),
            ],
        );
        write_memory_safetensors(
            &root.join("text_encoder/model.safetensors"),
            &[
                (
                    "model.visual.pos_embed.weight",
                    "BF16",
                    &[2304, 1024],
                    4_718_592,
                ),
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    "BF16",
                    &[2, 64],
                    256,
                ),
            ],
        );
        write_memory_safetensors(
            &root.join("vae/model.safetensors"),
            &[("norm.weight", "BF16", &[1], 2)],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        let contract = memory_strategy_contract_for_spec("mage_flow", &spec).unwrap();
        // The documented [2304,1024] vision position embedding stays dense bf16 because its loader
        // reads it directly. The adjacent LM projection is an actual target and takes Mage's Q8
        // text-layer floor. A projector that quantizes every packable rank-two weight reports the
        // old, invalid 1,327,104-byte Q4 position embedding instead of 4,718,592 bytes.
        assert_eq!(contract.asset_facts.conditioning_bytes, 4_718_592 + 136);
        assert_eq!(contract.asset_facts.conditioning_bytes - 136, 4_718_592);
        assert_ne!(contract.asset_facts.conditioning_bytes - 136, 1_327_104);
        assert_eq!(contract.asset_facts.transformer_bytes, 136 + 72);
        assert_eq!(contract.asset_facts.decoder_bytes, 2);
        assert_eq!(contract.asset_facts.base_bytes, 4_718_938);
        assert!(contract.conformance_errors().is_empty());
    }

    /// AC (SC-22662): every registered Mage-Flow route publishes the axes of the one NR-MMDiT they
    /// share, derived from this crate's own config constants, and passes the shared facts check.
    #[test]
    fn architecture_facts_follow_the_crate_transformer_config() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/mage-contract".into()));
        for provider_id in MODEL_IDS {
            let contract = weights_free_memory_strategy_contract(provider_id, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(24),
                    // 3072 / 24, derived by `MageFlowConfig::head_dim`.
                    head_dim: Some(128),
                    // 12 dual-stream blocks and no single-stream tail.
                    transformer_blocks: Some(12),
                    // Mage flattens token-per-latent-cell: a real 1, not an absent axis.
                    patch_size: Some(1),
                    latent_channels: Some(128),
                    vae_spatial_scale: Some(16),
                    vae_temporal_scale: None,
                    activation_dtype_width: Some(2),
                },
                "{provider_id} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    /// The `transformer/config.json` keys `from_transformer_config_json` requires, emitted from a
    /// config value so the fixture cannot drift from the struct it mirrors.
    fn transformer_config_json(cfg: &crate::config::MageFlowConfig) -> serde_json::Value {
        serde_json::json!({
            "in_channels": cfg.in_channels,
            "out_channels": cfg.out_channels,
            "context_in_dim": cfg.context_in_dim,
            "hidden_size": cfg.hidden_size,
            "num_heads": cfg.num_heads,
            "depth": cfg.depth,
            "axes_dim": cfg.axes_dim,
            "checkpoint": cfg.checkpoint,
            "patch_size": cfg.patch_size,
        })
    }

    fn spec_for_transformer_config(dir: &std::path::Path, config: &serde_json::Value) -> LoadSpec {
        let transformer = dir.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join(crate::transformer::TRANSFORMER_CONFIG_FILE),
            config.to_string(),
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
    }

    /// AC (SC-22662, review follow-up): on the **materialized** path the trunk axes are read out of
    /// the snapshot's own `transformer/config.json` — the file `MageTransformer::load` parses —
    /// rather than published from the compile-time preset. The mirror fixture agrees with the
    /// weights-free path; a fixture whose `depth` is mutated publishes the mutated block count,
    /// which is what the unconditional `architecture_facts()` this replaced would fail.
    ///
    /// `depth` is the mutated key because it is the trunk axis a snapshot can move on its own:
    /// `MageFlowConfig::validate` ties `hidden_size`/`num_heads` to `sum(axes_dim)`, so mutating
    /// the head width alone is rejected by the parser rather than published.
    #[test]
    fn materialized_trunk_axes_come_from_the_snapshot_rather_than_the_preset() {
        let preset = crate::config::MageFlowConfig::mage_flow();
        let weights_free = LoadSpec::new(WeightsSource::Dir("/nonexistent/mage-contract".into()));

        let mirror = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(Some(&spec_for_transformer_config(
                mirror.path(),
                &transformer_config_json(&preset)
            ))),
            architecture_facts(Some(&weights_free)),
            "a snapshot mirroring the published config must publish the preset's axes"
        );
        // The eager entry point carries no load identity at all and still publishes the preset.
        assert_eq!(
            architecture_facts(None),
            architecture_facts(Some(&weights_free))
        );

        let mutated_dir = tempfile::tempdir().unwrap();
        let mut mutated = transformer_config_json(&preset);
        mutated["depth"] = serde_json::json!(7);
        let mutated_facts = architecture_facts(Some(&spec_for_transformer_config(
            mutated_dir.path(),
            &mutated,
        )));
        assert_eq!(
            mutated_facts.transformer_blocks,
            Some(7),
            "the materialized path must publish the snapshot's depth, not the preset's"
        );
    }

    /// Feature-end review (SC-22667, E2): a materialized snapshot whose `transformer/config.json`
    /// is present but unparseable or partial must declare its trunk axes ABSENT, not fall back to
    /// the compile-time preset. `MageTransformer::load` propagates the same parse failure — every
    /// consumed key is required exactly so a partial file cannot half-default into this model's
    /// numbers — so a preset published here describes a model this load will never build. The rule
    /// is: fall back to a preset only where the LOADER falls back to that preset.
    ///
    /// Mutation that fails this: restoring
    /// `.unwrap_or_else(crate::config::MageFlowConfig::mage_flow)` in `dit_config` — both fixtures
    /// then publish the preset's `attention_heads`/`head_dim`/`transformer_blocks` as if they had
    /// been read off the snapshot.
    #[test]
    fn an_unparseable_snapshot_config_declares_the_trunk_axes_absent() {
        let preset = crate::config::MageFlowConfig::mage_flow();
        let declared = architecture_facts(None);

        // (1) Present but not JSON at all.
        let malformed = tempfile::tempdir().unwrap();
        let transformer = malformed.path().join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join(crate::transformer::TRANSFORMER_CONFIG_FILE),
            b"{not json",
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(malformed.path().to_path_buf()));

        // (2) Valid JSON, but missing a required key.
        let partial_dir = tempfile::tempdir().unwrap();
        let mut partial = transformer_config_json(&preset);
        partial
            .as_object_mut()
            .unwrap()
            .remove("depth")
            .expect("the fixture must have carried a depth to remove");
        let partial_spec = spec_for_transformer_config(partial_dir.path(), &partial);

        for (label, spec) in [("malformed", &spec), ("partial", &partial_spec)] {
            let facts = architecture_facts(Some(spec));
            assert_eq!(facts.attention_heads, None, "{label}");
            assert_eq!(facts.head_dim, None, "{label}");
            assert_eq!(facts.transformer_blocks, None, "{label}");
            assert_eq!(facts.patch_size, None, "{label}");
            assert_ne!(
                facts, declared,
                "{label}: an unreadable config must not publish the preset's trunk"
            );
            // The two latent axes are the loader's own crate constants rather than config reads, so
            // they survive and the contract still declares a real architecture axis.
            assert_eq!(facts.latent_channels, declared.latent_channels, "{label}");
            assert_eq!(
                facts.vae_spatial_scale, declared.vae_spatial_scale,
                "{label}"
            );
            assert!(facts.has_declared_architecture_axis(), "{label}");
            assert!(facts.zero_valued_axes().is_empty(), "{label}");
        }
    }

    #[test]
    fn empty_mage_component_directory_cannot_be_reported_as_zero() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_memory_snapshot(&root);
        std::fs::remove_file(root.join("text_encoder/model.safetensors")).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert!(memory_strategy_contract_for_spec("mage_flow", &spec).is_err());
        assert!(weights_free_memory_strategy_contract("mage_flow", &spec).is_ok());
    }

    #[test]
    fn adapter_contract_adds_load_exact_residency_and_preserves_missing_evidence() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_memory_snapshot(&root);
        let adapter = root.join("mage.safetensors");
        std::fs::write(&adapter, vec![0_u8; 4096]).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(vec![
            mlx_gen::AdapterSpec::new(adapter, 1.0, mlx_gen::AdapterKind::Lora),
        ]);
        let contract = memory_strategy_contract_for_spec("mage_flow", &spec).unwrap();

        assert!(contract.conformance_errors().is_empty());
        assert_eq!(contract.auxiliary_resident_bytes(), 4096);
        assert_eq!(contract.asset_facts.overlay_bytes, 4096);
        assert!(contract.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert_eq!(
            contract
                .predicted_peak_from_base(100)
                .predicted_peak_bytes(),
            4196
        );

        let missing = LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(vec![
            mlx_gen::AdapterSpec::new(
                root.join("missing.safetensors"),
                1.0,
                mlx_gen::AdapterKind::Lora,
            ),
        ]);
        let missing_contract = memory_strategy_contract_for_spec("mage_flow", &missing).unwrap();
        assert_eq!(missing_contract.auxiliary_resident_bytes(), 0);
        assert!(!missing_contract
            .formula
            .uses(MemoryFormulaVariable::OverlayBytes));
    }

    #[test]
    fn resident_safety_recomputes_peak_and_binds_calibration_identity() {
        use mlx_gen::gen_core::{
            MemoryBudget, MemoryCacheState, MemoryNumericTier, MemoryStrategyParameters,
            MEMORY_CALIBRATION_ABI,
        };

        let mismatch_root_tmp = tempfile::tempdir().unwrap();
        let mismatch_root = mismatch_root_tmp.path().to_path_buf();
        write_memory_snapshot(&mismatch_root);
        // A prepacked q4 snapshot at its own tier, so the loaded contract carries the q4 cell's
        // production identity (sc-22733) and the context below can bind to it.
        mark_packed(&mismatch_root, 4);
        let loaded_spec =
            LoadSpec::new(WeightsSource::Dir(mismatch_root.clone())).with_quant(Quant::Q4);
        let contract = memory_strategy_contract_for_spec("mage_flow", &loaded_spec).unwrap();
        let loaded_fingerprint = contract.calibration.as_ref().unwrap().fingerprint.clone();
        let required = (crate::memory::generation_peak_gb(Some(Quant::Q4), 512, 512, 1)
            * 1_000_000_000.0)
            .round() as u64;
        let valid = MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: crate::quant::COMPONENT_PRECISION_FLOORS,
                },
            },
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: loaded_fingerprint.clone(),
            load_shape: mlx_gen::LoadShape::EagerMaterialization,
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
                total_bytes: required + 1_000_000_000,
                committed_bytes: contract.asset_facts.base_bytes,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: required - contract.asset_facts.base_bytes,
            cache_state: MemoryCacheState::Warm,
            evidence_revision: "test".to_owned(),
        };
        let mismatched_spec =
            LoadSpec::new(WeightsSource::Dir(mismatch_root.clone())).with_quant(Quant::Q8);
        let registered = (MEMORY_REGISTRATION.safety_check)(&mismatched_spec, &contract, &valid);
        assert!(matches!(
            registered,
            MemorySafetyDecision::Reject { reason }
                if reason.contains("does not match loaded tier")
        ));
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &valid
        )
        .is_none());

        let mut wrong_identity = valid.clone();
        wrong_identity.calibration_fingerprint = "stale".to_owned();
        wrong_identity.mode = MemoryMode::Edit;
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &wrong_identity
        )
        .unwrap()
        .contains("calibration handshake mismatch"));

        let mut wrong_tier_and_mode = valid.clone();
        wrong_tier_and_mode.selection.tier.quant = Some(Quant::Q8);
        wrong_tier_and_mode.mode = MemoryMode::Edit;
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &wrong_tier_and_mode
        )
        .unwrap()
        .contains("does not match loaded tier"));

        let mut zero_zero = valid.clone();
        zero_zero.budget.total_bytes = 0;
        zero_zero.budget.committed_bytes = 0;
        zero_zero.predicted_peak_bytes = 0;
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &zero_zero
        )
        .unwrap()
        .contains("budget is unavailable"));

        let mut underreported = valid;
        underreported.predicted_peak_bytes = 0;
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &underreported
        )
        .unwrap()
        .contains("inconsistent"));

        let mut uncharged_resident_credit = underreported;
        uncharged_resident_credit.predicted_peak_bytes = required - contract.asset_facts.base_bytes;
        uncharged_resident_credit.budget.committed_bytes = 0;
        uncharged_resident_credit.budget.total_bytes =
            uncharged_resident_credit.predicted_peak_bytes;
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &uncharged_resident_credit
        )
        .unwrap()
        .contains("committed bytes"));
    }

    #[test]
    fn registered_receipts_bind_only_floors_active_for_the_loaded_tier() {
        for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
            let mut spec = LoadSpec::new(WeightsSource::Dir("/weights-free-mage".into()));
            spec.quantize = quant;
            let contract = weights_free_memory_strategy_contract("mage_flow", &spec).unwrap();
            let context = registered_valid_fixture(
                MageVariant::Rl,
                &spec,
                &contract,
                MemoryStrategy::StagedResidency,
            )
            .unwrap()
            .remove(0)
            .context;
            assert_eq!(
                context.selection.tier.component_precision_floors,
                crate::quant::active_component_precision_floors(quant),
                "fixture receipt must be tier-exact for {quant:?}"
            );
            assert_eq!(
                memory_strategy_safety_check_for(
                    "mage_flow",
                    MageVariant::Rl,
                    quant,
                    &contract,
                    &context,
                ),
                MemorySafetyDecision::Accept
            );

            if quant != Some(Quant::Q4) {
                let mut over_bound = context.clone();
                over_bound.selection.tier.component_precision_floors =
                    crate::quant::COMPONENT_PRECISION_FLOORS;
                assert!(matches!(
                    memory_strategy_safety_check_for(
                        "mage_flow",
                        MageVariant::Rl,
                        quant,
                        &contract,
                        &over_bound,
                    ),
                    MemorySafetyDecision::Reject { reason }
                        if reason.contains("does not match loaded tier")
                ));
            } else {
                let mut under_bound = context.clone();
                under_bound.selection.tier.component_precision_floors = &[];
                assert!(matches!(
                    memory_strategy_safety_check_for(
                        "mage_flow",
                        MageVariant::Rl,
                        quant,
                        &contract,
                        &under_bound,
                    ),
                    MemorySafetyDecision::Reject { reason }
                        if reason.contains("does not match loaded tier")
                ));
            }
        }
    }

    #[test]
    fn resident_scope_reapplies_request_state_after_cancel_cleanup() {
        let selection = MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: mlx_gen::gen_core::MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: crate::quant::COMPONENT_PRECISION_FLOORS,
            },
        };
        let geometry = MemoryGeometry {
            width: 1024,
            height: 768,
            batch: 3,
            frames: 1,
            reference_count: 0,
        };
        let contract = memory_strategy_contract("mage_flow", Some(Quant::Q4));
        let context = MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection,
            calibration_abi: mlx_gen::gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: contract.calibration.as_ref().unwrap().fingerprint.clone(),
            load_shape: mlx_gen::LoadShape::EagerMaterialization,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry,
            overlay: None,
            budget: mlx_gen::gen_core::MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: mlx_gen::gen_core::MemoryCacheState::Warm,
            evidence_revision: "test".to_owned(),
        };
        let make_scope = || {
            resident_request_scope(
                "mage_flow",
                &contract,
                &context,
                mlx_gen::request_scope::MlxScopeCleanup::None,
            )
            .unwrap()
        };
        assert_eq!(context.selection.strategy, MemoryStrategy::Resident);
        let mut canceled = make_scope();
        let mut first = GenerationRequest {
            prompt: "first".to_owned(),
            width: 1024,
            height: 768,
            count: 1,
            ..Default::default()
        };
        canceled.configure_request(&mut first).unwrap();
        assert_eq!(first.memory, Some(GenerationMemory::default()));
        let mut overflow = GenerationRequest {
            prompt: "overflow".to_owned(),
            width: 1024,
            height: 768,
            count: 4,
            ..Default::default()
        };
        assert!(canceled.configure_request(&mut overflow).is_err());
        assert!(canceled.configure_decode(1, 0, context.geometry).is_err());
        assert!(canceled.configure_attention(1).is_err());
        assert!(canceled.materialize_transformer_window(0, 1).is_err());
        canceled.finish(MemoryRunOutcome::Canceled).unwrap();
        assert!(canceled.finish(MemoryRunOutcome::Canceled).is_err());
        assert!(canceled.configure_request(&mut first).is_err());

        let mut warm = make_scope();
        let mut follow_up = GenerationRequest {
            prompt: "follow-up".to_owned(),
            width: 1024,
            height: 768,
            count: 1,
            ..Default::default()
        };
        warm.configure_request(&mut follow_up).unwrap();
        warm.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(
            warm.finish(MemoryRunOutcome::Complete).is_err(),
            "a warm follow-up owns fresh terminal state"
        );
    }

    #[test]
    fn actual_begin_hook_delegates_to_the_resident_scope_adopter() {
        let source = include_str!("model.rs");
        let begin = source
            .split_once("fn begin_memory_strategy_request(")
            .expect("Generator must retain its begin hook")
            .1
            .split_once("fn validate(")
            .expect("begin hook must remain bounded by validate")
            .0;
        assert!(
            begin.contains("resident_request_scope("),
            "the actual Generator begin hook bypassed the behaviorally tested shared-core adopter"
        );
    }

    /// sc-15154 — the footprint must follow the SPLIT layout's staged components, not the tier dir.
    ///
    /// The discriminating case: the same fake tier tree scanned with and without the components
    /// staged. A footprint that sums subdirs of `spec.weights` scores the split spec at the DiT's
    /// bytes alone and cannot tell the two specs apart, which is exactly what made the worker's
    /// over-budget message quote a figure unrelated to the tier's real install.
    #[test]
    fn the_footprint_counts_staged_components_not_just_the_tier_dir() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let write = |dir: std::path::PathBuf, bytes: usize| {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).unwrap();
            dir
        };
        // SPLIT: the variant tier dir holds the DiT; the shared mirror holds the TE + VAE.
        let tier = root.join("q4");
        write(tier.join("transformer"), 300);
        let te = write(root.join("shared/q4/text_encoder"), 700);
        let vae = write(root.join("shared/q4/vae"), 50);

        let split = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(tier.clone()))
            .with_component(COMPONENT_TEXT_ENCODER, mlx_gen::WeightsSource::Dir(te))
            .with_component(COMPONENT_VAE, mlx_gen::WeightsSource::Dir(vae));
        let got = component_footprint(&split).unwrap();
        assert_eq!(
            (got.dit, got.text_encoder, got.vae),
            (300, 700, 50),
            "the staged text encoder and VAE are part of what this tier loads"
        );

        // ...and the same spec with nothing staged sees only the DiT — the pre-fix behavior, kept
        // here so the assertion above is visibly about the staging and not about the tree.
        let unstaged = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(tier));
        let got = component_footprint(&unstaged).unwrap();
        assert_eq!((got.dit, got.text_encoder, got.vae), (300, 0, 0));

        // FLAT (upstream snapshots / legacy installs): every component under the root, nothing
        // staged. Unchanged by this fix.
        let flat = root.join("flat");
        write(flat.join("transformer"), 300);
        write(flat.join("text_encoder"), 700);
        write(flat.join("vae"), 50);
        let got = component_footprint(&mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(flat)))
            .unwrap();
        assert_eq!((got.dit, got.text_encoder, got.vae), (300, 700, 50));
    }

    /// A minimal but structurally valid safetensors file carrying exactly the Base identity tensor
    /// (`transformer_blocks.0.attn.add_k_proj.bias`, BF16, `[3072]`) filled with `fill` — enough for
    /// [`verify_checkpoint_identity`] to parse the header, seek, and hash. Nothing loads these
    /// weights; the tests below only exercise which *guard* fires.
    fn load_error(result: Result<Box<dyn Generator>>, context: &str) -> String {
        match result {
            Ok(_) => panic!("{context}"),
            Err(error) => error.to_string(),
        }
    }

    fn write_identity_only_checkpoint(path: &Path, fill: u8) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let payload = 3072 * 2; // BF16 [3072]
        let header = format!(
            "{{\"{BASE_IDENTITY_TENSOR}\":{{\"dtype\":\"BF16\",\"shape\":[3072],\"data_offsets\":[0,{payload}]}}}}"
        );
        let mut bytes = Vec::with_capacity(8 + header.len() + payload);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(std::iter::repeat_n(fill, payload));
        std::fs::write(path, bytes).unwrap();
    }

    /// sc-15036 — `load_finetuned` must get PAST the pinned-checkpoint identity guard that `load`
    /// enforces, because a full base fine-tune (sc-14056) rewrites every DiT weight *including*
    /// `transformer_blocks.0.attn.add_k_proj.bias`, so `load` rejects the user's own trained
    /// checkpoint by construction.
    ///
    /// Discriminating in both directions on ONE fabricated checkpoint:
    ///   * `load(Base, …)` must fail with the **fingerprint-mismatch** message — delete the guard
    ///     and this half fails;
    ///   * `load_finetuned(Base, …)` must fail with something else entirely (it reaches component
    ///     staging) — route `load_finetuned` back through the guard and this half fails.
    #[test]
    fn load_finetuned_bypasses_the_pinned_checkpoint_identity_guard() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let transformer = root.join("transformer");
        write_identity_only_checkpoint(
            &transformer.join("diffusion_pytorch_model.safetensors"),
            0x5a,
        );

        // `load` sees a snapshot ROOT and hashes `<root>/transformer/…`: the fill is not the pinned
        // Base fingerprint, so the guard fires.
        let published = load_error(
            load(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone())),
            ),
            "a checkpoint whose identity tensor moved must not load as published Base",
        );
        assert!(
            published.contains("checkpoint fingerprint mismatch"),
            "expected the identity guard to fire, got: {published}"
        );

        // `load_finetuned` is handed the SAME root — deliberately, so the mutation "delegate to
        // `load`" is caught: under it this call would report the fingerprint mismatch above. The
        // real entrypoint treats the path as the transformer dir itself and never opens
        // `<path>/transformer`, so it gets past identity and fails later, at the actual load.
        // Pid-keyed so the "this path does not exist" premise cannot be broken by a leftover from,
        // or a concurrent, second `cargo test` process sharing `$TMPDIR`.
        let staged_tmp = tempfile::tempdir().unwrap();
        let staged = staged_tmp.path().to_path_buf();
        let finetuned = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone()))
                    .with_component(COMPONENT_TEXT_ENCODER, WeightsSource::Dir(staged.clone()))
                    .with_component(COMPONENT_VAE, WeightsSource::Dir(staged)),
            ),
            "the fabricated checkpoint has no real components to load",
        );
        assert!(
            !finetuned.contains("checkpoint fingerprint"),
            "load_finetuned must not enforce the published-checkpoint fingerprint, got: {finetuned}"
        );
        // ...and the transformer dir it DID read is the one it was handed.
        assert!(
            transformer.is_dir(),
            "fixture sanity: the nested published-layout transformer dir exists"
        );
    }

    /// sc-15328 — `load` must ACCEPT `spec.adapters` (they install in [`assemble`] via
    /// [`crate::adapters::apply_mage_adapters`]), while [`load_finetuned`] must keep refusing them
    /// with a message that says why.
    ///
    /// Discriminating in both directions on one fixture, so neither half can pass vacuously:
    ///
    ///   * `load` carrying an adapter must get PAST the entry guard and fail on the *next* thing it
    ///     checks — the published-checkpoint fingerprint. Restore `|| !spec.adapters.is_empty()` to
    ///     `load`'s guard and this half fails, because the error becomes the `Unsupported` one.
    ///   * `load_finetuned` carrying the same adapter must fail on the adapter refusal
    ///     SPECIFICALLY — not on the missing components it would otherwise hit. Drop that guard and
    ///     this half fails.
    ///
    /// The adapter path is deliberately nonexistent: neither call may get far enough to read it,
    /// which is what makes the first half about the *guard* rather than about adapter loading.
    #[test]
    fn load_takes_adapters_while_a_fine_tuned_checkpoint_still_refuses_them() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_identity_only_checkpoint(
            &root
                .join("transformer")
                .join("diffusion_pytorch_model.safetensors"),
            0x5a,
        );
        let adapters = vec![mlx_gen::runtime::AdapterSpec::new(
            std::env::temp_dir().join(format!(
                "mage-adapter-never-read-{}.safetensors",
                std::process::id()
            )),
            0.8,
            mlx_gen::runtime::AdapterKind::Lora,
        )];

        let published = load_error(
            load(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(adapters.clone()),
            ),
            "the fabricated checkpoint is not the pinned Base, so this must still fail",
        );
        assert!(
            published.contains("checkpoint fingerprint mismatch"),
            "an adapter must no longer be refused at `load`'s entry guard — it should reach the \
             identity check like any other load, got: {published}"
        );

        let staged_tmp = tempfile::tempdir().unwrap();
        let staged = staged_tmp.path().to_path_buf();
        let finetuned = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone()))
                    .with_component(COMPONENT_TEXT_ENCODER, WeightsSource::Dir(staged.clone()))
                    .with_component(COMPONENT_VAE, WeightsSource::Dir(staged))
                    .with_adapters(adapters),
            ),
            "a fine-tuned checkpoint must not accept adapters",
        );
        assert!(
            finetuned.contains("cannot take LoRA/LoKr adapters"),
            "a fine-tune + adapter must be refused explicitly, and BEFORE the component staging it \
             would otherwise trip over, got: {finetuned}"
        );
    }

    /// sc-15328 — the descriptor is the engine's capability statement, and every Mage variant hosts
    /// adapters through the same `MageTransformer`. A variant that advertised neither would leave
    /// the app's `supports_adapters()` reading `false` for a model that demonstrably takes them.
    #[test]
    fn every_variant_advertises_lora_and_lokr() {
        for registration in REGISTRATIONS {
            let descriptor = (registration.descriptor)();
            assert!(
                descriptor.capabilities.supports_lora && descriptor.capabilities.supports_lokr,
                "{} must advertise supports_lora + supports_lokr: `assemble` installs both through \
                 `apply_mage_adapters` for every variant",
                descriptor.id
            );
        }
    }

    /// sc-15036 — the shared components are REQUIRED for a fine-tune and there is deliberately no
    /// flat-layout fallback: a training run emits the transformer alone, so probing
    /// `<transformer>/text_encoder` would turn a staging bug into a confusing deep load failure.
    /// Each missing id must be named. Also pins that a FILE weights source is refused (a fine-tune
    /// is a directory).
    #[test]
    fn load_finetuned_requires_both_shared_components_to_be_staged() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let dir = |name: &str| WeightsSource::Dir(root.join(name));

        let bare = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone())),
            ),
            "no components staged",
        );
        assert!(
            bare.contains(COMPONENT_TEXT_ENCODER),
            "the missing component must be named, got: {bare}"
        );

        let vae_only = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone()))
                    .with_component(COMPONENT_TEXT_ENCODER, dir("te")),
            ),
            "the VAE is still missing",
        );
        assert!(
            vae_only.contains(COMPONENT_VAE),
            "the missing VAE must be named, got: {vae_only}"
        );

        let as_file = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::File(
                    root.join("diffusion_pytorch_model.safetensors"),
                )),
            ),
            "a fine-tune is a transformer directory, not a single file",
        );
        assert!(
            as_file.contains("transformer DIRECTORY"),
            "expected the directory-shape refusal, got: {as_file}"
        );
    }

    /// sc-15036 real-weights end-to-end (epic 14034 F6): TRAIN a full base fine-tune, then RENDER
    /// with it through [`load_finetuned`], pairing the trained transformer with the base snapshot's
    /// own text encoder + VAE — the exact assembly the SceneWorks `mage_finetuned` worker lane
    /// performs.
    ///
    /// This is the claim the story exists to make true, so it is proved on real weights rather than
    /// asserted: before it, the checkpoint could not be loaded at all (the pinned-fingerprint guard
    /// rejects a retrained `add_k_proj.bias` by construction).
    ///
    /// The training step is deliberately GENTLE (4 steps at lr 1e-7, resolution 64) — this test is
    /// about the load + pairing seam, not convergence, and a gentle run is what makes the render a
    /// meaningful assertion. Measured on this checkpoint: at 10 steps / lr 1e-5 the run genuinely
    /// collapses the model onto its two-solid-swatch dataset and renders a FLAT FIELD, which would
    /// pass any "did we get pixels" check while telling you nothing about whether the trained
    /// transformer was correctly paired with the base's text encoder and VAE. At this budget the
    /// fine-tuned checkpoint still renders the base's own image, so the structure assertions below
    /// — dynamic range plus non-repeating rows, the same pair `base_real_weights.rs` uses — fail if
    /// the assembly is wrong in any way that degrades the decode.
    ///
    ///     MAGE_BASE_SNAPSHOT=<flat Mage-Flow-Base snapshot> \
    ///     MAGE_FINETUNE_RENDER_OUT=/tmp/finetuned.png \
    ///     cargo test -p mlx-gen-mage --lib finetune_then_render -- --ignored --nocapture
    #[test]
    #[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT) and an authorized Metal device"]
    fn finetune_then_render_through_load_finetuned() {
        use crate::transformer::{TRANSFORMER_CONFIG_FILE, TRANSFORMER_WEIGHTS_FILE};
        use mlx_gen::train::{TrainingConfig, TrainingItem, TrainingRequest};

        let Ok(root) = std::env::var("MAGE_BASE_SNAPSHOT") else {
            return;
        };
        let root = std::path::PathBuf::from(&root);
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();

        // --- train (tiny) ---
        let mut items = Vec::new();
        for (i, colour) in [[220u8, 60, 40], [40, 90, 210]].into_iter().enumerate() {
            let path = tmp.join(format!("swatch_{i}.png"));
            let mut im = image::RgbImage::new(96, 96);
            for px in im.pixels_mut() {
                *px = image::Rgb(colour);
            }
            im.save(&path).unwrap();
            items.push(TrainingItem::captioned(
                path,
                format!("a solid colour swatch {i}"),
            ));
        }
        let out_dir = tmp.join("finetune");
        let mut trainer =
            crate::training::load_trainer(&LoadSpec::new(WeightsSource::Dir(root.clone())))
                .unwrap();
        let output = trainer
            .train(
                &TrainingRequest {
                    items,
                    config: TrainingConfig {
                        full_finetune: true,
                        steps: 4,
                        resolution: 64,
                        learning_rate: 1e-7,
                        train_dtype: "f32".into(),
                        save_every: 0,
                        sample_every: 0,
                        seed: 7,
                        ..Default::default()
                    },
                    output_dir: out_dir.clone(),
                    file_name: "finetune.safetensors".into(),
                    trigger_words: vec![],
                    cancel: mlx_gen::CancelFlag::new(),
                },
                &mut |_| {},
            )
            .expect("the full fine-tune runs");
        drop(trainer);
        println!(
            "[sc-15036] trained {} steps, final loss {:.5}; checkpoint at {}",
            output.steps,
            output.final_loss,
            out_dir.display()
        );
        // The artifact really is a transformer component dir, not an adapter file.
        assert!(out_dir.join(TRANSFORMER_CONFIG_FILE).is_file());
        assert!(out_dir.join(TRANSFORMER_WEIGHTS_FILE).is_file());

        // --- render through the fine-tuned entrypoint ---
        // `spec.weights` is the trained transformer dir; the shared components come from the
        // INSTALLED base, exactly as the worker lane stages them.
        let spec = LoadSpec::new(WeightsSource::Dir(out_dir.clone()))
            .with_component(
                COMPONENT_TEXT_ENCODER,
                WeightsSource::Dir(root.join("text_encoder")),
            )
            .with_component(COMPONENT_VAE, WeightsSource::Dir(root.join("vae")));
        let model = match load_finetuned(MageVariant::Base, &spec) {
            Ok(model) => model,
            Err(error) => panic!("the fine-tuned checkpoint must load: {error}"),
        };

        let request = GenerationRequest {
            prompt: "a red apple on a wooden table, soft daylight".to_owned(),
            width: 512,
            height: 512,
            count: 1,
            seed: Some(11),
            steps: Some(20),
            guidance: Some(5.0),
            ..Default::default()
        };
        let out = model
            .generate(&request, &mut |_| {})
            .expect("the fine-tuned checkpoint renders");
        let GenerationOutput::Images(images) = out else {
            panic!("expected images");
        };
        let image = images.into_iter().next().expect("one image");
        assert_eq!((image.width, image.height), (512, 512));
        // Real STRUCTURE, not merely non-blank: full dynamic range and non-repeating rows. A flat
        // field (what a heavier fine-tune on this dataset legitimately produces, and also what a
        // mis-paired text encoder or a broken VAE decode produces) fails both.
        let (min, max) = image
            .pixels
            .iter()
            .fold((u8::MAX, u8::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(
            max.saturating_sub(min) >= 32,
            "the fine-tuned render has collapsed dynamic range: {min}..={max}"
        );
        let repeated_rows = image
            .pixels
            .chunks_exact(512 * 3)
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|rows| rows[0] == rows[1])
            .count();
        println!(
            "[sc-15036] fine-tuned render dynamic range {min}..={max}; repeated adjacent rows \
             {repeated_rows}/511"
        );
        assert!(
            repeated_rows < 51,
            "the fine-tuned render has {repeated_rows} repeated adjacent rows — the trained \
             transformer is not correctly paired with the base's text encoder / VAE"
        );

        if let Ok(png) = std::env::var("MAGE_FINETUNE_RENDER_OUT") {
            image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
                .expect("rgb buffer")
                .save(&png)
                .expect("png writes");
            println!("[sc-15036] wrote {png}");
        }
    }

    #[test]
    fn variant_table_matches_the_published_defaults() {
        // Pinned against the six model cards (epic sc-14034 ground-truth reference). Deliberately
        // asserts the *non-uniform* values — RL 20 vs Base 30 vs Turbo 4 — so a table that
        // collapsed to a single default could not pass.
        let table: Vec<(&str, bool, bool, u32, f32)> = MageVariant::ALL
            .iter()
            .map(|v| {
                (
                    v.id(),
                    v.is_edit(),
                    v.is_distilled(),
                    v.default_steps(),
                    v.default_cfg(),
                )
            })
            .collect();
        assert_eq!(
            table,
            vec![
                ("mage_flow", false, false, 20, 5.0),
                ("mage_flow_base", false, false, 30, 5.0),
                ("mage_flow_turbo", false, true, 4, 1.0),
                ("mage_flow_edit", true, false, 30, 5.0),
                ("mage_flow_edit_base", true, false, 30, 5.0),
                ("mage_flow_edit_turbo", true, true, 4, 1.0),
            ]
        );
    }

    #[test]
    fn registrations_cover_every_variant_in_order() {
        let ids: Vec<&str> = REGISTRATIONS
            .iter()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(ids, MODEL_IDS.to_vec());
        assert_eq!(
            ids,
            MageVariant::ALL.iter().map(|v| v.id()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn edit_variants_advertise_reference_conditioning_and_gen_variants_do_not() {
        let edit = descriptor_for(MageVariant::Edit);
        assert_eq!(
            edit.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        let gen = descriptor_for(MageVariant::Rl);
        assert!(gen.capabilities.conditioning.is_empty());
        // Distillation, not task, drives the CFG surface.
        assert!(gen.capabilities.supports_guidance);
        assert!(
            !descriptor_for(MageVariant::Turbo)
                .capabilities
                .supports_guidance
        );
        assert!(
            descriptor_for(MageVariant::Edit)
                .capabilities
                .supports_guidance
        );
    }

    #[test]
    fn every_edit_variant_enters_the_full_multimodal_loader() {
        let spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()));
        for variant in [
            MageVariant::Edit,
            MageVariant::EditBase,
            MageVariant::EditTurbo,
        ] {
            let err = load(variant, &spec)
                .err()
                .expect("missing edit snapshot must fail");
            assert!(
                !matches!(err, Error::Unsupported(_)),
                "{} must enter the multimodal component loader: {err}",
                variant.id(),
            );
        }
    }

    #[test]
    fn edit_variant_defaults_and_cfg_surfaces_are_exact() {
        let base = descriptor_edit_base();
        assert_eq!(base.id, "mage_flow_edit_base");
        assert_eq!(MageVariant::EditBase.default_steps(), 30);
        assert_eq!(MageVariant::EditBase.default_cfg(), 5.0);
        assert!(base.capabilities.supports_guidance);
        assert!(base.capabilities.supports_negative_prompt);

        let turbo = descriptor_edit_turbo();
        assert_eq!(turbo.id, "mage_flow_edit_turbo");
        assert_eq!(MageVariant::EditTurbo.default_steps(), 4);
        assert_eq!(MageVariant::EditTurbo.default_cfg(), 1.0);
        assert!(!crate::pipeline::uses_cfg(
            MageVariant::EditTurbo.default_cfg()
        ));
        assert!(!turbo.capabilities.supports_guidance);
        assert!(!turbo.capabilities.supports_negative_prompt);
    }

    #[test]
    #[ignore = "needs complete MAGE_EDIT_SNAPSHOT, MAGE_EDIT_BASE_SNAPSHOT, and MAGE_EDIT_TURBO_SNAPSHOT"]
    fn complete_edit_snapshots_are_config_identical_and_checkpoint_distinct() {
        let root = |name: &str| {
            std::path::PathBuf::from(
                std::env::var(name).unwrap_or_else(|_| panic!("set {name} to a complete snapshot")),
            )
        };
        let edit = root("MAGE_EDIT_SNAPSHOT");
        let base = root("MAGE_EDIT_BASE_SNAPSHOT");
        let turbo = root("MAGE_EDIT_TURBO_SNAPSHOT");
        for relative in [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/chat_template.json",
            "transformer/config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/preprocessor_config.json",
            "text_encoder/tokenizer.json",
            "text_encoder/tokenizer_config.json",
            "text_encoder/video_preprocessor_config.json",
            "text_encoder/vocab.json",
            "vae/config.json",
        ] {
            let expected = std::fs::read(edit.join(relative)).unwrap();
            assert_eq!(
                std::fs::read(base.join(relative)).unwrap(),
                expected,
                "Edit-Base {relative} must be byte-identical to Edit RL"
            );
            assert_eq!(
                std::fs::read(turbo.join(relative)).unwrap(),
                expected,
                "Edit-Turbo {relative} must be byte-identical to Edit RL"
            );
        }

        let check = |root: &Path, variant, revision, hash| {
            verify_checkpoint_identity(
                root,
                variant,
                revision,
                EDIT_IDENTITY_TENSOR,
                EDIT_IDENTITY_BYTES,
                &[3072],
                hash,
            )
        };
        check(
            &edit,
            MageVariant::Edit,
            EDIT_SNAPSHOT_REVISION,
            EDIT_IDENTITY_SHA256,
        )
        .unwrap();
        check(
            &base,
            MageVariant::EditBase,
            EDIT_BASE_SNAPSHOT_REVISION,
            EDIT_BASE_IDENTITY_SHA256,
        )
        .unwrap();
        check(
            &turbo,
            MageVariant::EditTurbo,
            EDIT_TURBO_SNAPSHOT_REVISION,
            EDIT_TURBO_IDENTITY_SHA256,
        )
        .unwrap();

        for wrong in [&edit, &turbo] {
            assert!(
                check(
                    wrong,
                    MageVariant::EditBase,
                    EDIT_BASE_SNAPSHOT_REVISION,
                    EDIT_BASE_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit-Base must reject RL and Turbo transformer weights"
            );
        }
        for wrong in [&base, &turbo] {
            assert!(
                check(
                    wrong,
                    MageVariant::Edit,
                    EDIT_SNAPSHOT_REVISION,
                    EDIT_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit must reject Base and Turbo transformer weights"
            );
        }
        for wrong in [&edit, &base] {
            assert!(
                check(
                    wrong,
                    MageVariant::EditTurbo,
                    EDIT_TURBO_SNAPSHOT_REVISION,
                    EDIT_TURBO_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit-Turbo must reject RL and Base transformer weights"
            );
        }
    }

    #[test]
    fn edit_reference_shapes_are_required_and_preserve_order() {
        let image = |byte| Image {
            width: 1,
            height: 1,
            pixels: vec![byte, byte, byte],
        };
        let request = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: image(1),
                    strength: None,
                },
                Conditioning::MultiReference {
                    images: vec![image(2), image(3)],
                },
            ],
            ..Default::default()
        };
        let refs = edit_references(&request).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0].as_raw(), &[1, 1, 1]);
        assert_eq!(refs[1].as_raw(), &[2, 2, 2]);
        assert_eq!(refs[2].as_raw(), &[3, 3, 3]);
        assert!(edit_references(&GenerationRequest::default()).is_err());
        let malformed = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0; 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        assert!(edit_references(&malformed).is_err());
    }

    #[test]
    fn rl_load_enters_the_real_snapshot_loader() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow".into()));
        let err = load(MageVariant::Rl, &spec)
            .err()
            .expect("missing snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "RL must not regress to the scaffold refusal: {err}"
        );
    }

    #[test]
    fn base_has_a_distinct_registration_and_enters_the_full_snapshot_loader() {
        assert_eq!(descriptor_base().id, "mage_flow_base");
        assert_eq!(
            MageVariant::Base.upstream_repo(),
            "microsoft/Mage-Flow-Base"
        );
        assert_eq!((REGISTRATION_BASE.descriptor)().id, "mage_flow_base");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow-base".into()));
        let err = load_base(&spec)
            .err()
            .expect("missing Base snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "Base must enter the same complete component-tree loader: {err}"
        );
    }

    #[test]
    fn base_platform_defaults_are_thirty_steps_with_real_cfg() {
        assert_eq!(MageVariant::Base.default_steps(), 30);
        assert_eq!(MageVariant::Base.default_cfg(), 5.0);
        let descriptor = descriptor_for(MageVariant::Base);
        assert!(descriptor.capabilities.supports_guidance);
        assert!(descriptor.capabilities.supports_negative_prompt);
        let request = GenerationRequest {
            prompt: "test".into(),
            negative_prompt: Some("artifact".into()),
            width: 1024,
            height: 1024,
            guidance: Some(5.0),
            ..Default::default()
        };
        validate_generation_request(&descriptor, &request).unwrap();
    }

    #[test]
    fn turbo_has_a_distinct_registration_and_enters_the_full_snapshot_loader() {
        assert_eq!(descriptor_turbo().id, "mage_flow_turbo");
        assert_eq!(
            MageVariant::Turbo.upstream_repo(),
            "microsoft/Mage-Flow-Turbo"
        );
        assert_eq!((REGISTRATION_TURBO.descriptor)().id, "mage_flow_turbo");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow-turbo".into()));
        let err = load_turbo(&spec)
            .err()
            .expect("missing Turbo snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "Turbo must enter the same complete component-tree loader: {err}"
        );
    }

    #[test]
    fn turbo_platform_defaults_are_four_steps_with_cfg_and_negative_prompt_off() {
        assert_eq!(MageVariant::Turbo.default_steps(), 4);
        assert_eq!(MageVariant::Turbo.default_cfg(), 1.0);
        let descriptor = descriptor_for(MageVariant::Turbo);
        assert!(!descriptor.capabilities.supports_guidance);
        assert!(!descriptor.capabilities.supports_negative_prompt);
        let plain = GenerationRequest {
            prompt: "test".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        validate_generation_request(&descriptor, &plain).unwrap();
        let mut negative = plain.clone();
        negative.negative_prompt = Some("must not be encoded".into());
        assert!(validate_generation_request(&descriptor, &negative).is_err());
        let mut guided = plain;
        guided.guidance = Some(2.0);
        assert!(validate_generation_request(&descriptor, &guided).is_err());
    }

    #[test]
    fn rl_platform_defaults_and_exact_native_sizes_validate() {
        assert_eq!(MageVariant::Rl.default_steps(), 20);
        assert_eq!(MageVariant::Rl.default_cfg(), 5.0);
        let descriptor = descriptor_for(MageVariant::Rl);
        assert_eq!(descriptor.capabilities.max_count, MAX_COUNT);
        for &(width, height) in &[
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (512, 2048),
            (2048, 512),
            (1232, 688),
        ] {
            let req = GenerationRequest {
                prompt: "test".into(),
                width,
                height,
                ..Default::default()
            };
            validate_generation_request(&descriptor, &req).unwrap();
        }
        for &(width, height) in &[(496, 512), (512, 2064), (513, 512)] {
            let req = GenerationRequest {
                prompt: "test".into(),
                width,
                height,
                ..Default::default()
            };
            assert!(
                validate_generation_request(&descriptor, &req).is_err(),
                "{width}x{height} must be rejected"
            );
        }
        let mut batch = GenerationRequest {
            prompt: "test".into(),
            width: 2048,
            height: 2048,
            count: MAX_COUNT,
            ..Default::default()
        };
        validate_generation_request(&descriptor, &batch).unwrap();
        batch.count += 1;
        assert!(validate_generation_request(&descriptor, &batch).is_err());
    }

    /// SC-18610: the published surface for **every** Mage route, on **every** shipped tier.
    ///
    /// Before this, the declaration derived streamability by probing the weights-free fixture path,
    /// so Q4 and Q8 — the tiers a constrained Mac actually installs — published rung 4 as `Missing`
    /// even though the engine implements it for them.
    #[test]
    fn every_mage_route_publishes_the_complete_ladder_on_every_shipped_tier() {
        use mlx_gen::gen_core::{MemoryContractSurfaceTier, MemoryStrategySupport};
        use std::collections::BTreeSet;

        let expected_rung_four: BTreeSet<&str> = [
            "bf16:resident:deferred",
            "bf16:sequential:deferred",
            "q4:resident:deferred",
            "q4:sequential:deferred",
            "q8:resident:deferred",
            "q8:sequential:deferred",
        ]
        .into_iter()
        .collect();

        for variant in MageVariant::ALL {
            let provider_id = variant.id();
            let surfaces = mlx_gen::gen_core::mlx_memory_contract_surface_specs();
            assert_eq!(surfaces.len(), 12, "{provider_id}");
            let mut rung_four = BTreeSet::new();
            for surface in &surfaces {
                let contract = weights_free_memory_surface_contract(provider_id, surface)
                    .unwrap_or_else(|error| {
                        panic!("{provider_id} {}: {error}", surface.selector.id())
                    });
                assert_eq!(contract.provider_id, provider_id);
                assert!(
                    contract.conformance_errors().is_empty(),
                    "{provider_id} {}",
                    surface.selector.id()
                );
                assert_eq!(
                    contract.asset_facts,
                    Default::default(),
                    "a weights-free surface must publish no measured bytes"
                );
                assert_eq!(contract.load_shape, surface.selector.load_shape);
                for strategy in [
                    MemoryStrategy::Resident,
                    MemoryStrategy::StagedResidency,
                    MemoryStrategy::BoundedDecode,
                    MemoryStrategy::BoundedAttention,
                ] {
                    assert_eq!(
                        contract.capability(strategy).unwrap().support,
                        MemoryStrategySupport::Implemented,
                        "{provider_id} {} {strategy:?}",
                        surface.selector.id()
                    );
                }
                let rung4 = contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap();
                if rung4.support == MemoryStrategySupport::Implemented {
                    assert_eq!(rung4.parameters.transformer_window_sizes, [1]);
                    assert_eq!(
                        rung4.parameters.transformer_window_components,
                        [mlx_gen::gen_core::TransformerComponent::Both]
                    );
                    assert!(contract.lifecycle.transformer_window_materialization);
                    rung_four.insert(surface.selector.id());
                } else {
                    assert_eq!(rung4.support, MemoryStrategySupport::Missing);
                    assert!(!contract.lifecycle.transformer_window_materialization);
                }
                // Every shipped MLX tier must be one of the three the engine executes; a fourth
                // would silently ride the `matches!` arm in `surface_streamable`.
                assert!(matches!(
                    surface.resolved_artifact_tier(),
                    MemoryContractSurfaceTier::Bf16
                        | MemoryContractSurfaceTier::Q4
                        | MemoryContractSurfaceTier::Q8
                ));
            }
            assert_eq!(
                rung_four.iter().copied().collect::<BTreeSet<_>>(),
                expected_rung_four,
                "{provider_id} must publish rung 4 on every shipped tier under both offload policies"
            );
        }
    }

    /// Mage never reads [`LoadSpec::offload_policy`]: staging and streaming are per-request flags on
    /// a [`Residency::request_scoped`] pipeline. The declaration must track the load shape, which is
    /// what actually decides whether the snapshot stays reopenable.
    #[test]
    fn surface_rung_four_tracks_load_shape_not_offload_policy() {
        use mlx_gen::gen_core::MemoryStrategySupport;

        let rung_four = |selector_id: &str| {
            let surface = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
                .into_iter()
                .find(|surface| surface.selector.id() == selector_id)
                .unwrap();
            weights_free_memory_surface_contract("mage_flow_edit", &surface)
                .unwrap()
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support
                .clone()
        };
        // Offload policy alone never moves the rung.
        assert_eq!(
            rung_four("q4:resident:deferred"),
            MemoryStrategySupport::Implemented
        );
        assert_eq!(
            rung_four("q4:sequential:deferred"),
            MemoryStrategySupport::Implemented
        );
        // Load shape alone always does.
        assert_eq!(
            rung_four("q4:resident:eager"),
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            rung_four("q4:sequential:eager"),
            MemoryStrategySupport::Missing
        );
    }

    /// Each unsupported axis is mutated on its own: asserting the whole set at once would prove the
    /// set is rejected without proving any individual guard fires.
    #[test]
    fn declaration_rejects_every_unsupported_route_and_load_axis_individually() {
        let base = || {
            LoadSpec::new(WeightsSource::Dir("/weights-free-mage".into()))
                .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
        };
        assert!(validate_load_contract("mage_flow", &base()).is_ok());
        assert!(validate_load_contract("mage_flow_not_a_route", &base()).is_err());
        assert!(validate_load_contract("flux1_dev", &base()).is_err());

        let mut cases: Vec<LoadSpec> =
            vec![
                LoadSpec::new(WeightsSource::File("/weights.safetensors".into()))
                    .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization),
            ];
        let mut fp32 = base();
        fp32.precision = Precision::Fp32;
        cases.push(fp32);
        cases.push(base().with_quant(Quant::Nvfp4));
        let mut control = base();
        control.control = Some(WeightsSource::File("/control.safetensors".into()));
        cases.push(control);
        let mut extra_control = base();
        extra_control
            .extra_controls
            .push(WeightsSource::File("/extra-control.safetensors".into()));
        cases.push(extra_control);
        let mut ip_adapter = base();
        ip_adapter.ip_adapter = Some(WeightsSource::Dir("/ip".into()));
        cases.push(ip_adapter);
        cases.push(base().with_pid(
            WeightsSource::File("/pid.safetensors".into()),
            WeightsSource::Dir("/gemma".into()),
        ));
        let mut identity = base();
        identity.identity = Some(Default::default());
        cases.push(identity);
        let mut text_encoder = base();
        text_encoder.text_encoder = Some(WeightsSource::Dir("/external-text".into()));
        cases.push(text_encoder);
        let mut unknown_component = base();
        unknown_component.components.insert(
            "unexpected".to_owned(),
            WeightsSource::Dir("/unexpected".into()),
        );
        cases.push(unknown_component);

        for spec in cases {
            assert!(
                validate_load_contract("mage_flow", &spec).is_err(),
                "unsupported axis must be refused by the declaration"
            );
            // The refusal is typed identically on the declaration, the finite surface, and the
            // production seam, so no path can publish a ladder another path would reject.
            assert!(weights_free_memory_strategy_contract("mage_flow", &spec).is_err());
            assert!(memory_strategy_contract_for_spec("mage_flow", &spec).is_err());
            let surface = mlx_gen::gen_core::MemoryContractSurfaceSpec {
                selector: mlx_gen::gen_core::MemoryContractSurfaceSelector {
                    tier: mlx_gen::gen_core::MemoryContractSurfaceTier::Bf16,
                    offload_policy: spec.offload_policy,
                    load_shape: spec.load_shape,
                },
                spec,
            };
            assert!(weights_free_memory_surface_contract("mage_flow", &surface).is_err());
        }
    }

    /// A selector that disagrees with its own `LoadSpec` would publish the tier as a label rather
    /// than a fact about the artifact the route resolves.
    #[test]
    fn surface_selector_must_agree_with_its_load_spec() {
        use mlx_gen::gen_core::{MemoryContractSurfaceSpec, MemoryContractSurfaceTier};

        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            let tier = surface.resolved_artifact_tier();
            for crossed_tier in [
                MemoryContractSurfaceTier::Bf16,
                MemoryContractSurfaceTier::Q4,
                MemoryContractSurfaceTier::Q8,
                MemoryContractSurfaceTier::Nvfp4,
            ] {
                if crossed_tier == tier {
                    continue;
                }
                let mut selector = surface.selector;
                selector.tier = crossed_tier;
                let crossed = MemoryContractSurfaceSpec {
                    selector,
                    spec: surface.spec.clone(),
                };
                assert!(
                    weights_free_memory_surface_contract("mage_flow", &crossed).is_err(),
                    "{tier:?} spec must not be published as {crossed_tier:?}"
                );
            }
            let mut selector = surface.selector;
            selector.offload_policy = match selector.offload_policy {
                mlx_gen::OffloadPolicy::Resident => mlx_gen::OffloadPolicy::Sequential,
                mlx_gen::OffloadPolicy::Sequential => mlx_gen::OffloadPolicy::Resident,
            };
            assert!(weights_free_memory_surface_contract(
                "mage_flow",
                &MemoryContractSurfaceSpec {
                    selector,
                    spec: surface.spec.clone(),
                }
            )
            .is_err());
            let mut selector = surface.selector;
            selector.load_shape = match selector.load_shape {
                mlx_gen::LoadShape::EagerMaterialization => {
                    mlx_gen::LoadShape::DeferredMaterialization
                }
                mlx_gen::LoadShape::DeferredMaterialization => {
                    mlx_gen::LoadShape::EagerMaterialization
                }
            };
            assert!(weights_free_memory_surface_contract(
                "mage_flow",
                &MemoryContractSurfaceSpec {
                    selector,
                    spec: surface.spec,
                }
            )
            .is_err());
        }
    }

    /// Declaration is not reachability. This drives the exact seam `assemble` uses to build a
    /// loaded generator's contract — for **all six** routes, over a prepacked Q4 snapshot shaped
    /// like the shipped `<variant>/<tier>/` install — and proves the loaded contract carries the
    /// same rung 4 the finite surface publishes, then executes the admitted selection into the
    /// tensor-neutral request controls the pipeline actually reads.
    #[test]
    fn every_loaded_mage_route_reaches_the_declared_rung_four() {
        use mlx_gen::gen_core::{MemoryNumericTier, MemoryStrategySupport};

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        write_memory_snapshot(&root);
        // Prepacked Q4 markers on the weight-bearing components: the shipped tier shape, in which
        // `load_time_quant_bits` is `None` and the snapshot stays reopenable.
        for component in ["text_encoder", "transformer"] {
            std::fs::write(
                root.join(component).join("config.json"),
                r#"{"quantization":{"bits":4,"group_size":64}}"#,
            )
            .unwrap();
        }
        std::fs::write(root.join("vae").join("config.json"), "{}").unwrap();

        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(Quant::Q4)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        let dirs = resolve_component_dirs(&root, &spec).unwrap();
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: crate::quant::active_component_precision_floors(Some(
                Quant::Q4,
            )),
        };

        for variant in MageVariant::ALL {
            let provider_id = variant.id();
            let loaded =
                memory_strategy_contract_for_resolved_components(provider_id, &spec, &dirs)
                    .unwrap();
            assert_eq!(loaded.provider_id, provider_id);
            assert_eq!(
                loaded
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented,
                "{provider_id} must reach the rung its surface declares"
            );
            // The declared surface and the loaded contract agree rung for rung.
            let surface = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
                .into_iter()
                .find(|surface| surface.selector.id() == "q4:sequential:deferred")
                .unwrap();
            let declared = weights_free_memory_surface_contract(provider_id, &surface).unwrap();
            for strategy in [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_eq!(
                    declared.capability(strategy).unwrap().support,
                    loaded.capability(strategy).unwrap().support,
                    "{provider_id} {strategy:?} declaration and loaded contract disagree"
                );
            }

            // Executable: the route's own registered behavior opens a scope for the rung and
            // resolves it into the request controls Mage's generate path reads.
            let mut fixture = registered_valid_fixture(
                variant,
                &spec,
                &loaded,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap()
            .remove(0);
            assert_eq!(
                fixture.context.mode,
                if variant.is_edit() {
                    MemoryMode::Edit
                } else {
                    MemoryMode::TextToImage
                },
                "{provider_id}"
            );
            assert_eq!(
                fixture.context.geometry.reference_count,
                u32::from(variant.is_edit()),
                "{provider_id}"
            );
            assert!(loaded
                .representative_selection(MemoryStrategy::BoundedTransformerResidency, tier, false)
                .is_ok());
            let mut scope =
                registered_begin_request(provider_id, variant, &spec, &loaded, &fixture.context)
                    .unwrap()
                    .unwrap_or_else(|| panic!("{provider_id} must open a rung-4 request scope"));
            scope.configure_request(&mut fixture.request).unwrap();
            let memory = fixture.request.memory.unwrap_or_else(|| {
                panic!("{provider_id} rung-4 scope must configure request memory")
            });
            assert!(memory.stream_transformer_blocks, "{provider_id}");
            assert!(memory.stage_residency, "{provider_id}");
            assert!(memory.tile_vae_decode, "{provider_id}");
            assert!(memory.chunk_attention, "{provider_id}");
            assert_eq!(memory.transformer_window_size, Some(1), "{provider_id}");

            // A sibling route's context must not authorize this one: the mode gate separates the
            // edit trio from the generate trio.
            let mut crossed = fixture.context.clone();
            crossed.mode = if variant.is_edit() {
                MemoryMode::TextToImage
            } else {
                MemoryMode::Edit
            };
            assert!(matches!(
                memory_strategy_safety_check_for(
                    provider_id,
                    variant,
                    Some(Quant::Q4),
                    &loaded,
                    &crossed
                ),
                MemorySafetyDecision::Reject { .. }
            ));
        }
    }
}
