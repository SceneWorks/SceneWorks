//! Shared memory-strategy contract for base SANA and SANA-Sprint (SC-16783, extended by SC-15523).
//!
//! The provider implements Resident, load-time staged residency, the measured DC-AE tiled decode
//! ladder, and — since SC-15523 — bounded attention over `attn2` and bounded transformer residency
//! over the 20-block Linear-DiT stack.
//!
//! ## Rung 3's scope is `attn2` only, and that is architecture, not convenience
//!
//! SANA has two attentions per block and only one of them has anything for a score budget to bound:
//!
//! | site | kernel | score domain |
//! |---|---|---|
//! | `attn1` (self) | ReLU-**linear**, `num = (V·Kᵀ)·Q` | none — the key axis is contracted into a `[B,H,hd,hd]` gram matrix |
//! | `attn2` (caption cross) | hand-rolled softmax SDPA, f32 | `[B, H, N, 300]`, explicitly materialized |
//!
//! So the rung is genuinely applicable (it is **not** `StructurallyNotApplicable`) and its reach is
//! genuinely one of the two sites. The key axis is fixed at 300 caption slots, so the domain grows
//! only with the token count `N = (edge/32)²` — 6.14 Mi f32 scores at 1024², which is why the
//! sibling families' 64 Mi operating point is inert here and SANA publishes its own budgets.
//!
//! **The exactness claim has a measured boundary.** Query-row chunking leaves each row's complete
//! k/v and both reductions untouched, but it changes the query GEMM's `M` dimension, and at `M = 1`
//! MLX dispatches a different (gemv) kernel whose accumulation order differs — measured at ~1e-6
//! relative on this provider. So rung 3 is bit-exact **over its published domain**, not
//! unconditionally: the narrowest chunk any published budget produces anywhere in the advertised
//! 256..=1024 range is **174 query rows**, and
//! `a_single_query_row_chunk_is_not_bit_exact_and_the_domain_cannot_reach_one` pins both halves of
//! that — the degenerate case is not exact, and the domain cannot reach it.
//!
//! ## Rung 4's availability is a per-LOAD fact
//!
//! A window rebuilds trunk blocks from the snapshot, so it needs a **re-openable source**
//! ([`WeightsSource::Dir`] — every registry load; SANA already refuses a single-file load) *and*
//! [`LoadShape::DeferredMaterialization`], which says the request wants those re-openable blocks
//! rather than a bulk-committed stack. `OffloadPolicy` is deliberately absent from that pair: phase
//! release and intra-phase block materialization are separate axes. The rung-1 edge is declared
//! separately as a per-provider `additional_prerequisites` entry, because
//! [`MemoryStrategy::engages`] does **not** make rung 4 engage rung 1 universally.

use mlx_gen::asset_facts::ResidentProjection;
use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, GenerationMemory,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    ResidentRequestMemory, Result as CoreResult, TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, WeightsSource};

use crate::pipeline::{DECODE_OVERLAP, DECODE_TILE_EDGE};

pub const DECODE_TILE_EDGES: &[u32] = &[512, 384, 256, 192];
pub const DECODE_TILE_EDGES_REJECTED: &[u32] = &[128, 96, 64];

/// Rung 3's published score budgets, in `[B, H, N, 300]` f32 elements per `attn2` call.
///
/// SANA's whole 1024² cross-attention domain is `1 · 20 · 1024 · 300 = 6.14 Mi` scores, so the
/// Z-Image family operating point ([`mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`], 64 Mi)
/// never chunks here — the shared planner is budget-agnostic precisely so a family can publish its
/// own. These three bracket the domain: at 1024² they chunk into 2, 3 and 6 calls respectively, and
/// the smallest still chunks at 512² (2 calls). None of them chunks the 256² floor, whose whole
/// domain is 384 Ki scores — deliberately, because 1.5 MiB of f32 scratch is not what this rung
/// exists to bound. `the_shared_64mi_budget_never_chunks_sana_and_the_published_ones_do` pins all of
/// that arithmetic.
pub const ATTENTION_CHUNK_SIZES: &[u32] = &[4_194_304, 2_097_152, 1_048_576];
/// The default budget — the tightest published bound, which is what the rung exists for.
pub const ATTENTION_CHUNK_SIZE: u32 = 1_048_576;
/// Budgets swept and REJECTED, kept with their rejection so the domain is not silently narrowed.
/// 64 Mi is the sibling families' constant and is inert at every advertised SANA geometry.
pub const ATTENTION_CHUNK_SIZES_REJECTED: &[u32] = &[67_108_864];

/// **Rung 4 is implemented, output-preserving, and WITHHELD — measured, not assumed (SC-15523).**
///
/// The window runs and it bounds what it claims to bound. From the safetensors `data_offsets`, all
/// 20 blocks are 93.59 MiB each and the stack is 1871.9 MiB, so a `window = 1` render holds
/// **93.59 MiB instead of 1871.9 MiB** of trunk weights; the wall clock confirms the mechanism
/// executes (651 → 929 ms/step, +42%, which is the 20-blocks-per-step re-materialization cost). The
/// image is byte-identical across five production latents.
///
/// It does **not move the REQUEST peak**, and this epic's rule is that bounding a phase which is not
/// the request peak is not a saving. Measured on `sana_1600m` q4 at 1024² (Sequential + Deferred):
///
/// | row | peak | vs the rung-3 control |
/// |---|---:|---:|
/// | rung-3 control | 2.9108 GiB | — |
/// | `window = 1` | 2.8602 GiB | −1.74% |
///
/// −1.74% is inside this cell's **positional** noise, and the noise is not random: eight identical
/// windowed requests produce a deterministic five-value cycle spanning **4.92%**
/// (2.8053 / 2.9108 / 2.8602 / 2.8270 / 2.9434, repeating). Under `SANA_WINDOW_PROBE_ORDER` the
/// peak follows the row's POSITION and not its cadence — cadence 10 run first and cadence 4 run
/// first both read exactly 2.8602 GiB. A genuine weight-residency effect cannot do that, so the
/// cadence column is withdrawn as evidence rather than published.
///
/// **Why, mechanically — and this half is MEASURED, not inferred from the byte table.**
/// `the_request_peak_bearing_phase_is_measured_not_assumed` sweeps the advertised edge range with
/// rungs 1-3 engaged. SANA's conditioning phase is geometry-INDEPENDENT (the CHI prompt pads to a
/// fixed 300 caption slots at every size) while denoise and decode both scale with `N = (edge/32)²`,
/// so the sweep separates them:
///
/// | edge | tokens | peak |
/// |---|---:|---:|
/// | 256² | 64 | 2.9108 GiB |
/// | 512² | 256 | 2.8602 GiB |
/// | 1024² | 1024 | 2.8270 GiB |
///
/// **−2.88% across a 16× token increase** — flat, every value a member of the five-cycle. A denoise-
/// or decode-borne peak cannot do that, and the flat value is above the 2.1596 GiB Gemma-2 weight
/// floor, so the component setting it is large enough to be the caption encoder rather than
/// something smaller that merely happens not to scale. The arithmetic agrees — after rungs 1-3 the
/// *windowed* denoise phase holds only ≈1.28 GiB (93.59 MiB trunk + 24.8 MiB non-block +
/// 1191.1 MiB dense-f32 DC-AE) — but the measurement is what settles it.
///
/// SC-15969's own survey noted that a TextEncoder-scoped window is structurally available for this
/// family; that scope, not the DiT one, is what would move SANA's peak — tracked as **sc-17859**,
/// its own story because the encoder lives in `mlx-gen-pid` and is shared with PiD and LTX, so
/// windowing it is a change to a component three families load.
///
/// The implementation is retained deliberately: it is correct, it is exercised by the weights-free
/// block-stream and windowed-forward tests, and it is the foundation the TextEncoder scope builds
/// on. Flipping this one constant re-publishes the rung when a measurement justifies it.
pub const TRANSFORMER_WINDOW_WITHHELD: bool = true;

/// Rung 4's block cadences over the 20-block `transformer_blocks` stack. Published only when
/// [`TRANSFORMER_WINDOW_WITHHELD`] is cleared.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 5, 10];
/// The default cadence — the tightest weight bound.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
/// The only component scope SANA windows. The Gemma-2 caption encoder is a uniform decoder stack and
/// is structurally windowable too (SC-15969's survey says so), but it lives in `mlx-gen-pid` and is
/// shared with PiD and LTX; nothing here is measured for it, so it stays unpublished and the
/// contract REFUSES it rather than letting an unmeasured scope be selected.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// The measured `2026-08-06` key of the **(`sana_1600m`, q4, Sequential)** cell — the one cell
/// `tests/memory_ladder_real_weights.rs` actually sweeps (`REPRESENTATIVE` = `sana_1600m`,
/// `DEFAULT_TIER` = `q4`). Retained byte-for-byte by [`production_calibration_fingerprint`].
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "sana-mlx-full-ladder-2026-08-06-v1-sequential";
/// The same measured cell under `OffloadPolicy::Resident`.
pub const RESIDENT_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "sana-mlx-full-ladder-2026-08-06-v1-resident";

/// The route the two retained strings above were measured on.
pub const CALIBRATED_ROUTE: &str = crate::MODEL_ID;
/// The tier the two retained strings above were measured on.
pub const CALIBRATED_TIER: &str = "q4";

/// Artifact-tier label of a SANA load: `bf16` for a dense snapshot, `q4`/`q8` for the shipped
/// packed tiers. `None` for a tier this family does not ship.
///
/// Both callers pass the tier they have *proven*: [`production_calibration_fingerprint`] answers
/// for the request knob and the production contract binds that answer to the artifact's own marker
/// through [`resolved_artifact_tier`].
pub fn calibration_tier_label(quant: Option<mlx_gen::Quant>) -> Option<&'static str> {
    match quant {
        None => Some("bf16"),
        Some(mlx_gen::Quant::Q4) => Some("q4"),
        Some(mlx_gen::Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

const fn policy_label(policy: OffloadPolicy) -> &'static str {
    match policy {
        OffloadPolicy::Sequential => "sequential",
        OffloadPolicy::Resident => "resident",
    }
}

fn route_label(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        crate::MODEL_ID => Some("base"),
        crate::SPRINT_MODEL_ID => Some("sprint"),
        _ => None,
    }
}

/// The tier of the artifact `spec` points at, read from the seam the SANA loader itself
/// packed-detects on: the `{base}.scales` companions inside the `transformer/` weight file(s)
/// (`crate::quant::lin` → [`mlx_gen::quant::lin`]), with the width inferred from the u32 codes /
/// scales shape ratio at `crate::quant::GROUP_SIZE` exactly as [`mlx_gen::quant::packed_bits`]
/// infers it at load. The SANA turnkey writes **no** `transformer/config.json` (`crate::convert`:
/// the config is the hard-coded `SanaTransformerConfig`, and the width self-describes via the
/// `.scales` shapes), so a JSON marker is never consulted — a probe on it read every shipped tier
/// as unreadable and published no identity at all.
///
/// `Ok(None)` is a dense (bf16) transformer. `Err` fails closed on anything that is not a
/// readable shipped tier: a root that is not a snapshot directory, a `transformer/` component that
/// is absent or carries no readable `.safetensors`, a packed base with no `.weight` codes, a codes /
/// scales ratio that is not an exact 4- or 8-bit pack, or two bases packed at different widths.
pub fn resolved_artifact_tier(spec: &LoadSpec) -> CoreResult<Option<mlx_gen::Quant>> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Unsupported(
            "SANA artifact tier: the load is not a snapshot directory".to_owned(),
        ));
    };
    let component = root.join("transformer");
    let fail = |detail: String| {
        CoreError::Unsupported(format!(
            "SANA artifact tier: {} — {detail}",
            component.display()
        ))
    };
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&component)
        .map_err(|error| fail(format!("no readable transformer weights ({error})")))?;
    if headers.is_empty() {
        return Err(fail(
            "no transformer weights to read a tier from".to_owned(),
        ));
    }
    let shapes: std::collections::HashMap<&str, &[usize]> = headers
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
        .collect();
    let mut width: Option<mlx_gen::Quant> = None;
    for tensor in &headers {
        let Some(base) = tensor.name.strip_suffix(".scales") else {
            continue;
        };
        let codes = shapes
            .get(format!("{base}.weight").as_str())
            .ok_or_else(|| fail(format!("`{base}.scales` has no `{base}.weight` codes")))?;
        let bits = packed_width_from_shapes(codes, &tensor.shape)
            .ok_or_else(|| fail(format!("`{base}` is not an exact 4- or 8-bit pack")))?;
        match width {
            None => width = Some(bits),
            Some(seen) if seen == bits => {}
            Some(seen) => {
                return Err(fail(format!(
                    "packed bases disagree on their width: `{base}` is {bits:?}, an earlier base \
                     is {seen:?}"
                )))
            }
        }
    }
    Ok(width)
}

/// [`mlx_gen::quant::packed_bits`] on header shapes alone: `scales` is `[out, in / GROUP_SIZE]` and
/// the u32 codes are `[out, in · bits / 32]`, so `bits = codes.cols · 32 / (scales.cols · GROUP_SIZE)`
/// when that division is exact and lands on a shipped width.
fn packed_width_from_shapes(codes: &[usize], scales: &[usize]) -> Option<mlx_gen::Quant> {
    let (&[out, code_cols], &[scale_rows, scale_cols]) = (codes, scales) else {
        return None;
    };
    if out != scale_rows {
        return None;
    }
    let group = usize::try_from(crate::quant::GROUP_SIZE).ok()?;
    let in_dim = scale_cols.checked_mul(group)?;
    let packed_width = code_cols.checked_mul(32)?;
    if in_dim == 0 || packed_width % in_dim != 0 {
        return None;
    }
    match packed_width / in_dim {
        4 => Some(mlx_gen::Quant::Q4),
        8 => Some(mlx_gen::Quant::Q8),
        _ => None,
    }
}

/// The tier a load ASKS for. SANA's tiers are prepacked and the loader packed-detects them, so the
/// worker may pass the resolved tier through `LoadSpec::quantize` (its shipped shape) or leave it
/// `None` and let the loader detect; both spellings load the same weights, so the requested tier is
/// the explicit one when it is there and the artifact's own packed width otherwise.
/// [`memory_strategy_contract`] then proves the answer against that width either way, so this can
/// never turn a dense snapshot into a packed claim on its own.
pub fn requested_tier(spec: &LoadSpec) -> Option<mlx_gen::Quant> {
    spec.quantize
        .or_else(|| resolved_artifact_tier(spec).ok().flatten())
}

/// Production calibration identity table of the SANA routes, keyed on
/// **(provider, tier, offload policy)** — sc-22731, epic sc-22723 E1/E4.
///
/// Before sc-22731 the key was the offload policy ALONE, so `sana_1600m` and `sana_sprint_1600m`
/// published the same string and so did all three shipped tiers of each: six cells sharing two
/// identities, which is exactly the sc-22511 false green — a Sprint q8 anchor binding base SANA's
/// q4 evidence. The provider and the tier are now in the key.
///
/// The **policy stays in the key** (unlike the FLUX.1 table, sc-22726): SANA's two strings are not
/// one measurement seen twice. Rung 4 is declared only on `Sequential` (`windowed` in
/// `contract_with_asset_facts`), so the two policies publish genuinely different ladders and each
/// carries its own retained measured string. `MemoryCalibrationIdentity::load_shape` continues to
/// carry the materialization axis separately, and the identity is independent of it.
///
/// This is the TABLE, not the binding: the tier here is [`requested_tier`], and only
/// [`memory_strategy_contract`] — which proves that tier against the artifact's own packed width
/// before publishing — may turn one of these strings into a production contract identity.
pub fn production_calibration_fingerprint(provider_id: &str, spec: &LoadSpec) -> Option<String> {
    let route = route_label(provider_id)?;
    let tier = calibration_tier_label(requested_tier(spec))?;
    let policy = policy_label(spec.offload_policy);
    Some(
        if provider_id == CALIBRATED_ROUTE && tier == CALIBRATED_TIER {
            match spec.offload_policy {
                OffloadPolicy::Sequential => MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
                OffloadPolicy::Resident => RESIDENT_MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            }
        } else {
            format!("sana-{route}-{tier}-mlx-full-ladder-v1-{policy}")
        },
    )
}

/// The weights-free registry-conformance identity: the same (provider, tier, policy) coordinate in
/// a namespace that can never collide with a production string, so a fixture contract can never be
/// mistaken for measured evidence of the cell it describes.
pub fn weights_free_calibration_fingerprint(provider_id: &str, spec: &LoadSpec) -> Option<String> {
    let route = route_label(provider_id)?;
    let tier = calibration_tier_label(requested_tier(spec))?;
    let policy = policy_label(spec.offload_policy);
    Some(format!(
        "sana-{route}-{tier}-mlx-weights-free-conformance-v1-{policy}"
    ))
}

/// The two catalog ids this contract serves.
///
/// A contract minted for any other id is a declaration no route can reach — the exact defect class
/// SC-18607 exists to close — so the id is authenticated rather than interpolated into the result.
fn is_known_provider(provider_id: &str) -> bool {
    matches!(provider_id, crate::MODEL_ID | crate::SPRINT_MODEL_ID)
}

/// **Fail closed on every load axis the loader itself refuses or silently discards (SC-18607).**
///
/// A memory contract is a claim about a load that can actually run. Before this gate the
/// declaration was minted from the load shape alone, so three separate shapes got a full published
/// ladder they could never execute:
///
/// 1. axes `crate::model::load_components` rejects outright — a non-`Bf16` precision override, a
///    LoRA/LoKr adapter set, and a single-file source;
/// 2. component slots SANA **never reads** — `control`, `extra_controls`, `ip_adapter`, `pid`,
///    `identity`, `text_encoder` and named `components`. The loader resolves every component from
///    the snapshot root, so a caller-provisioned overlay in one of those slots is discarded today;
///    declaring a ladder over it advertises a composition nothing executes;
/// 3. an unknown provider id.
///
/// `quantize` is deliberately NOT here: SANA's tiers are **packed-detected from disk**, so
/// `LoadSpec::quantize` is advisory and every tier resolves the same declared ladder — see
/// `the_declared_ladder_is_invariant_in_the_advisory_quantize_axis`.
fn validate_load_contract(provider_id: &str, spec: &LoadSpec) -> CoreResult<()> {
    if !is_known_provider(provider_id) {
        return Err(CoreError::Unsupported(format!(
            "unknown SANA provider {provider_id}"
        )));
    }
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: SANA memory routes require a snapshot directory \
             (transformer/ vae/ text_encoder/), not a single .safetensors file"
        )));
    }
    if spec.precision != mlx_gen::Precision::Bf16 {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: LoRA/LoKr adapters are not supported"
        )));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
    {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: SANA loads every component from the snapshot root; it has no control, \
             IP-adapter, PiD, identity, external text-encoder or named-component slot"
        )));
    }
    Ok(())
}

/// Whether THIS load can execute a rung-4 window. See the module doc — two independent load-time
/// facts, and `OffloadPolicy` is not one of them.
pub(crate) fn is_streamable(spec: &LoadSpec) -> bool {
    matches!(spec.weights, WeightsSource::Dir(_))
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && spec.adapters.is_empty()
}

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP as u32,
    )
}

/// Price the DC-AE decoder at its **resident** width rather than its stored one (the sc-15839
/// class), falling back to the stored sum when the component is not on disk.
///
/// [`crate::dc_ae::DcAeDecoder::from_weights`] and its encoder twin `as_dtype(Float32)` every conv
/// weight, bias and norm they read, on every load, unconditionally — the same shape as
/// `mlx_gen_sdxl::load_vae`'s `cast_all(Float32)`, which sc-15839 measured as a **2x underprice at
/// every tier**.
///
/// Every SANA tier shipped so far already stores the DC-AE in f32, so this projection is IDENTITY
/// today. That is exactly why it is written down instead of left implicit: a narrower VAE in some
/// future tier would be underpriced by 2x from the moment it shipped, and an underpriced decoder
/// reads as a memory regression in the engine rather than as a pricing bug in the contract. The
/// trunk and the Gemma-2 caption encoder stay [`ResidentProjection::Stored`] — both keep their
/// on-disk width (packed tiers load packed; the dtype casts in those two components are all on
/// activations, never on a retained weight).
fn resident_decoder_bytes(spec: &LoadSpec, stored: u64) -> u64 {
    let WeightsSource::Dir(root) = &spec.weights else {
        return stored;
    };
    let vae = root.join("vae");
    if !vae.is_dir() {
        return stored;
    }
    mlx_gen::asset_facts::projected_safetensors_bytes(&vae, |_| ResidentProjection::Float32)
        .unwrap_or(stored)
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    let components = crate::model::component_footprint(spec)?;
    contract_with_asset_facts(
        provider_id,
        spec,
        production_calibration_identity(provider_id, spec),
        components.text_encoder,
        components.dit,
        resident_decoder_bytes(spec, components.vae),
    )
}

/// The identity a PRODUCTION load publishes: the (provider, tier, policy) string from
/// [`production_calibration_fingerprint`], but only once the requested tier has been proven to be
/// the tier of the artifact on disk (sc-22731, the sc-22726 review rule).
///
/// A dense snapshot loaded with `quantize = Some(_)` would be a load-time requantization whose peak
/// no anchor measured, and a packed snapshot loaded with the OTHER packed tier requested is not a
/// shipped load either. Both publish `None` rather than the requested tier's string. A packed
/// snapshot loaded with `quantize = None` IS the packed load (the loader detects the width, and
/// [`requested_tier`] follows it), so it publishes that tier's string. An artifact whose width
/// cannot be read publishes `None` — fail closed.
fn production_calibration_identity(
    provider_id: &str,
    spec: &LoadSpec,
) -> Option<MemoryCalibrationIdentity> {
    if resolved_artifact_tier(spec).ok()? != requested_tier(spec) {
        return None;
    }
    production_calibration_fingerprint(provider_id, spec)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
}

/// Architecture axes for one registered SANA route (epic SC-22657, E2).
///
/// This crate mirrors the reference `transformer/config.json` as
/// [`SanaTransformerConfig`](crate::config::SanaTransformerConfig) and the reference DC-AE config as
/// [`DcAeConfig`](crate::config::DcAeConfig); the Sprint variant differs only in guidance embedding
/// and QK-norm, so both routes publish the same shape.
///
/// SANA's DC-AE is a **f32 deep-compression** autoencoder: its six `block_out_channels` stages give
/// the x32 spatial scale, not the x8 an ordinary four-stage AutoencoderKL gives. `patch_size` is 1 —
/// the DiT consumes the 32-channel latent token directly.
///
/// `vae_temporal_scale` stays `None`: SANA is an image model whose autoencoder has no temporal axis,
/// and a structurally absent axis is declared absent, never zero.
fn architecture_facts(provider_id: &str) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit = if provider_id == crate::SPRINT_MODEL_ID {
        crate::config::SanaTransformerConfig::sana_sprint_1600m()
    } else {
        crate::config::SanaTransformerConfig::sana_1600m()
    };
    let vae = crate::config::DcAeConfig::sana_f32c32();
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.attention_head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: mlx_gen::architecture_facts::axis(dit.patch_size),
        latent_channels: mlx_gen::architecture_facts::axis(vae.latent_channels),
        vae_spatial_scale: mlx_gen::architecture_facts::vae_spatial_scale_from_stages(
            vae.num_stages(),
        ),
        vae_temporal_scale: None,
        // The SANA transformer is dtype-preserving over an f32 latent, and its linear-attention
        // Q/K/V and RMSNorm seams are pinned f32 outright (`transformer.rs`).
        activation_dtype_width: Some(mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH),
    }
}

pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    let calibration = weights_free_calibration_fingerprint(provider_id, spec)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    contract_with_asset_facts(provider_id, spec, calibration, 0, 0, 0)
}

fn contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    calibration: Option<MemoryCalibrationIdentity>,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
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
    contract.load_shape = spec.load_shape;
    contract.phase_facts = Some(mlx_gen::gen_core::MemoryPhaseFacts {
        architecture: Some(
            mlx_gen::gen_core::ImagePipelineArchitecture::SanaLinearDit {
                classifier_free_guidance: provider_id == crate::model::MODEL_ID,
            },
        ),
        staged_weights: mlx_gen::gen_core::StagedWeightSchedule::ThreeStage,
        transformer_stream: None,
        decoder_workspace: Some(mlx_gen::gen_core::DecoderWorkspaceFacts {
            tiling: mlx_gen::gen_core::DecoderTilingRealization::WholeTail,
            activation_dtype_width: 4,
            channels: vec![1024, 1024, 512, 512, 256, 128],
            input_channels: vec![32, 1024, 1024, 512, 512, 256],
            spatial_divisors: vec![32, 16, 8, 4, 2, 1],
        }),
    });
    contract.architecture_facts = architecture_facts(provider_id);
    let staged = matches!(spec.offload_policy, OffloadPolicy::Sequential);
    // Rung 4 needs BOTH load-time facts AND rung 1, whose own availability IS the `Sequential`
    // policy — declaring it on a Resident load would advertise a composition its own declared
    // prerequisite could never satisfy. ONE binding, so the capability, the lifecycle hook and the
    // formula variable cannot disagree about whether a window can run on this load.
    let windowed = is_streamable(spec) && staged && !TRANSFORMER_WINDOW_WITHHELD;
    let mut variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
    ];
    if windowed {
        variables.push(MemoryFormulaVariable::TransformerWindowSize);
    }
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables,
    };
    contract.calibration = calibration;
    contract.asset_facts.conditioning_bytes = conditioning_bytes;
    contract.asset_facts.transformer_bytes = transformer_bytes;
    contract.asset_facts.decoder_bytes = decoder_bytes;
    contract.asset_facts.base_bytes = conditioning_bytes
        .saturating_add(transformer_bytes)
        .saturating_add(decoder_bytes);
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: windowed,
    };
    // Sequential loads ship the measured bounded-decode default. An explicit shared-contract
    // Resident selection must therefore write an all-disabled request block to override it.
    contract.resident_request_memory = ResidentRequestMemory::ExplicitResident;

    for strategy in [
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ] {
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == strategy)
            .expect("compatibility contract declares every rung");
        capability.support = match strategy {
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode | MemoryStrategy::BoundedAttention => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if windowed => {
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
        capability.parameters = match strategy {
            MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                decode_tile_edges: routes.native_edges().to_vec(),
                decode_overlaps: vec![DECODE_OVERLAP as u32],
                ..Default::default()
            },
            MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                attention_chunk_sizes: ATTENTION_CHUNK_SIZES.to_vec(),
                ..Default::default()
            },
            MemoryStrategy::BoundedTransformerResidency
                if capability.support == MemoryStrategySupport::Implemented =>
            {
                MemoryParameterRanges {
                    transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                    transformer_window_components: vec![TRANSFORMER_WINDOW_COMPONENT],
                    ..Default::default()
                }
            }
            _ => MemoryParameterRanges::default(),
        };
    }
    if windowed {
        // Rung 4 holds ~95 MiB of trunk weights instead of ~1.85 GiB, but that only moves the
        // REQUEST peak if the phases it does not bound have already been shed. `engages` does not
        // supply this edge (rung 4 does not engage rung 1 universally), so the provider declares it.
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    // The compatibility constructor supplies the required empty engagement-exclusion surface.
    debug_assert!(contract.default_engagement_exclusions.is_empty());
    Ok(contract)
}

/// Refuse a request-scoped memory selection whose parameters are outside the published domain —
/// **on the production `generate` path**, at the same layer for every rung.
///
/// A caller can set [`GenerationMemory`] on a request without going through
/// `begin_memory_strategy_request`, so admission-time validation alone leaves a hole through which
/// an unmeasured tile edge, score budget or block cadence reaches the engine and is silently
/// executed. Every published parameter is checked here, including rung 2's, so the three rungs are
/// enforced in one place rather than one of them being enforced twice and the others not at all.
///
/// The decode route goes through the same checked [`mlx_gen_pid::DecodeRoutes`] the admission gate
/// uses, so the native ladder and the PiD refusal cannot drift apart.
pub(crate) fn validate_request_memory(
    provider_id: &str,
    spec: &LoadSpec,
    memory: &GenerationMemory,
) -> CoreResult<()> {
    if memory.tile_vae_decode {
        decode_routes(provider_id)?
            .validate(
                false,
                Some(memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE as u32)),
                Some(memory.decode_overlap.unwrap_or(DECODE_OVERLAP as u32)),
            )
            .map_err(CoreError::Unsupported)?;
    }
    if memory.chunk_attention {
        let size = memory.attention_chunk_size.unwrap_or(ATTENTION_CHUNK_SIZE);
        if !ATTENTION_CHUNK_SIZES.contains(&size) {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: attention_chunk_size {size} is outside the published domain \
                 {ATTENTION_CHUNK_SIZES:?}"
            )));
        }
    }
    if memory.stream_transformer_blocks {
        if TRANSFORMER_WINDOW_WITHHELD {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: bounded transformer residency is implemented and measured, and \
                 WITHHELD because it does not move the request peak on this family (-1.74% inside \
                 a 4.92% positional cycle); see TRANSFORMER_WINDOW_WITHHELD for the numbers"
            )));
        }
        if !is_streamable(spec) {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: bounded transformer residency needs a re-openable snapshot \
                 directory loaded with DeferredMaterialization and no adapters"
            )));
        }
        // Rung 4's declared `EngagedInSameRequest` prerequisite, enforced rather than documented.
        // SANA's rung 1 is a LOAD-time mechanism (the `Sequential` policy drives the shared
        // `Residency` seam; sc-16783 measured and published it that way), so `is_streamable` above
        // plus the contract's `Sequential`-only rung-4 declaration already guarantee the mechanism
        // is running. This check is the other half: the SELECTION must say so too, or a caller
        // could compose a request the selector can never produce and get a peak nobody predicted.
        if !memory.stage_residency {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: bounded transformer residency requires staged residency engaged in \
                 the same request"
            )));
        }
        let window = memory
            .transformer_window_size
            .unwrap_or(TRANSFORMER_WINDOW_SIZE);
        if !TRANSFORMER_WINDOW_SIZES.contains(&window) {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: transformer_window_size {window} is outside the published domain \
                 {TRANSFORMER_WINDOW_SIZES:?}"
            )));
        }
        let component = memory
            .transformer_window_component
            .unwrap_or(TransformerComponent::Dit);
        if component != TRANSFORMER_WINDOW_COMPONENT {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: transformer_window_component {component:?} is outside the published \
                 domain [{TRANSFORMER_WINDOW_COMPONENT:?}]"
            )));
        }
    }
    Ok(())
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            if context.use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{}: SANA has no PiD decoder; mlx-gen-pid is used only for Gemma-2 conditioning",
                    contract.provider_id
                )));
            }
            decode_routes(&contract.provider_id)?
                .validate(
                    false,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !matches!(
        contract
            .capability(strategy)
            .map(|capability| &capability.support),
        Some(MemoryStrategySupport::Implemented)
    ) || !strategy.is_optimized()
    {
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
        MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: matches!(spec.offload_policy, OffloadPolicy::Sequential),
            overlay: None,
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_with_cleanup(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = decode_routes(provider_id)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::SanaTransformerConfig::sana_1600m().num_layers as usize,
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{provider_id}: SANA has no PiD decoder"
                )));
            }
            routes
                .validate(false, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = contract
        .engages(context.selection.strategy, MemoryStrategy::BoundedAttention)
        .then(|| {
            context
                .selection
                .parameters
                .attention_chunk_size
                .unwrap_or(ATTENTION_CHUNK_SIZE)
        });
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then(|| {
            context
                .selection
                .parameters
                .transformer_window_size
                .unwrap_or(TRANSFORMER_WINDOW_SIZE)
        });
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub fn declared_parameters() -> mlx_gen::gen_core::MemoryStrategyParameters {
    mlx_gen::gen_core::MemoryStrategyParameters {
        stage_residency: None,
        decode_tile_edge: Some(DECODE_TILE_EDGE as u32),
        decode_overlap: Some(DECODE_OVERLAP as u32),
        attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::MemoryStrategySupport;
    use mlx_gen::WeightsSource;

    fn spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/sana-contract".into()))
            .with_offload_policy(policy)
            .with_load_shape(LoadShape::EagerMaterialization)
    }

    fn streamable_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/sana-contract".into()))
            .with_offload_policy(policy)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    /// AC (SC-22662): both registered SANA routes publish the axes of the DiT and DC-AE this crate
    /// declares, and each contract passes the shared facts conformance check.
    #[test]
    fn architecture_facts_follow_the_crate_transformer_and_dcae_configs() {
        let spec = streamable_spec(OffloadPolicy::Sequential);
        for provider_id in [crate::MODEL_ID, crate::SPRINT_MODEL_ID] {
            let contract = weights_free_memory_strategy_contract(provider_id, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(70),
                    head_dim: Some(32),
                    transformer_blocks: Some(20),
                    // The DiT consumes the DC-AE latent token directly.
                    patch_size: Some(1),
                    latent_channels: Some(32),
                    // Six DC-AE stages => five halvings => x32, not the x8 of a plain image VAE.
                    vae_spatial_scale: Some(32),
                    vae_temporal_scale: None,
                    activation_dtype_width: Some(4),
                },
                "{provider_id} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    #[test]
    fn both_ids_publish_the_same_ladder_with_rung_four_withheld() {
        let spec = streamable_spec(OffloadPolicy::Sequential);
        let base = weights_free_memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        let sprint = weights_free_memory_strategy_contract(crate::SPRINT_MODEL_ID, &spec).unwrap();
        assert!(
            base.conformance_errors().is_empty(),
            "{:?}",
            base.conformance_errors()
        );
        for strategy in MemoryStrategy::ALL {
            assert_eq!(base.capability(strategy), sprint.capability(strategy));
        }
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                base.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?} must be implemented on the full-ladder route"
            );
        }
        // Rung 4: implemented, output-preserving, measured, and WITHHELD. The withdrawal is a
        // measured verdict at a stated cell, not a structural one, so it is one flippable constant.
        const { assert!(TRANSFORMER_WINDOW_WITHHELD) };
        assert_eq!(
            base.capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(!base.lifecycle.transformer_window_materialization);
        assert_eq!(
            base.resident_request_memory,
            ResidentRequestMemory::ExplicitResident
        );
        assert!(base.lifecycle.attention_chunking);
        // The rung-1 edge rung 4 WOULD carry is a provider declaration, never an inherited one: the
        // shared cost order deliberately does not drag rung 1 in. Pinned here so a future
        // re-publication cannot quietly rely on the shared order for it.
        assert!(
            !MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::StagedResidency),
            "the shared cost order must never be the source of rung 4's rung-1 edge"
        );
        assert!(base
            .requires(MemoryStrategy::BoundedTransformerResidency)
            .all(|prerequisite| !matches!(prerequisite, MemoryStrategyPrerequisite::Rung { .. })));
    }

    #[test]
    fn published_rung_three_and_four_domains_are_pinned() {
        assert_eq!(ATTENTION_CHUNK_SIZES, &[4_194_304, 2_097_152, 1_048_576]);
        assert_eq!(ATTENTION_CHUNK_SIZE, 1_048_576);
        assert_eq!(ATTENTION_CHUNK_SIZES_REJECTED, &[67_108_864]);
        assert_eq!(TRANSFORMER_WINDOW_SIZES, &[1, 2, 4, 5, 10]);
        assert_eq!(TRANSFORMER_WINDOW_SIZE, 1);
        assert_eq!(TRANSFORMER_WINDOW_COMPONENT, TransformerComponent::Dit);
        let contract = weights_free_memory_strategy_contract(
            crate::MODEL_ID,
            &streamable_spec(OffloadPolicy::Sequential),
        )
        .unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            ATTENTION_CHUNK_SIZES.to_vec()
        );
        // Rung 4 is withheld, so it publishes NO parameter domain — a withheld rung that still
        // advertised cadences would let a caller select one the contract refuses.
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert!(window.parameters.transformer_window_sizes.is_empty());
        assert!(window.parameters.transformer_window_components.is_empty());
        // The rejected sibling constant must stay out of the published domain in both directions.
        for rejected in ATTENTION_CHUNK_SIZES_REJECTED {
            assert!(!ATTENTION_CHUNK_SIZES.contains(rejected));
        }
    }

    /// **Rung 4's availability is a per-LOAD fact, and the two facts are independent.**
    ///
    /// An eager load and a single-file load can BOTH satisfy every other declaration, so a rung-4
    /// declaration keyed off anything else (the offload policy, the provider id, the tier) would
    /// pass a shape it cannot execute to the selector.
    #[test]
    fn rung_four_availability_reads_source_load_shape_and_the_staged_prerequisite() {
        let deferred_dir = streamable_spec(OffloadPolicy::Sequential);
        assert!(is_streamable(&deferred_dir));
        // The load-time predicate is kept correct even while the rung is withheld, so a future
        // re-publication does not have to re-derive it.

        let mut eager = deferred_dir.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        assert!(!is_streamable(&eager));

        let single_file =
            LoadSpec::new(WeightsSource::File("/nonexistent/sana.safetensors".into()))
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(LoadShape::DeferredMaterialization);
        assert!(!is_streamable(&single_file));

        let mut adapted = deferred_dir.clone();
        adapted.adapters.push(mlx_gen::AdapterSpec::new(
            "/nonexistent/lora.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        assert!(!is_streamable(&adapted));

        // SC-18607: a single-file source and an adapter set are not "rung 4 unavailable" — they are
        // loads SANA refuses outright, so the contract refuses to describe them AT ALL rather than
        // publishing rungs 1-3 over a composition no route can execute.
        for unloadable in [&single_file, &adapted] {
            assert!(weights_free_memory_strategy_contract(crate::MODEL_ID, unloadable).is_err());
            assert!(memory_strategy_contract(crate::SPRINT_MODEL_ID, unloadable).is_err());
        }

        // An EAGER directory load, by contrast, is a real declarable load — it simply cannot execute
        // a window, so it keeps rungs 1-3 and drops rung 4.
        {
            let contract = weights_free_memory_strategy_contract(crate::MODEL_ID, &eager).unwrap();
            assert!(contract.conformance_errors().is_empty());
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing,
                "a load that cannot stream must not advertise rung 4"
            );
            assert!(!contract.lifecycle.transformer_window_materialization);
            let MemoryFormulaKind::PhaseEnvelope { variables, .. } = &contract.formula else {
                panic!("SANA declares a phase envelope")
            };
            assert!(!variables.contains(&MemoryFormulaVariable::TransformerWindowSize));
            let _ = &deferred_dir;
            assert!(
                !contract
                    .requires(MemoryStrategy::BoundedTransformerResidency)
                    .any(|prerequisite| matches!(
                        prerequisite,
                        MemoryStrategyPrerequisite::Rung { .. }
                    )),
                "an unavailable rung must not carry a prerequisite edge"
            );
            // Rung 3 is load-shape independent: it bounds scratch, not residency.
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            );
        }

        // A Resident load can stream in principle, but rung 4's declared prerequisite is rung 1,
        // whose availability IS the Sequential policy — so declaring it would advertise a
        // composition the contract itself can never satisfy.
        let resident = weights_free_memory_strategy_contract(
            crate::MODEL_ID,
            &streamable_spec(OffloadPolicy::Resident),
        )
        .unwrap();
        assert!(resident.conformance_errors().is_empty());
        assert_eq!(
            resident
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        // The capability, the lifecycle hook and the formula variable are ONE binding: a contract
        // that withheld the rung but kept advertising the hook or the variable would tell a caller
        // its peak depends on a cadence it cannot select.
        assert!(!resident.lifecycle.transformer_window_materialization);
        let MemoryFormulaKind::PhaseEnvelope { variables, .. } = &resident.formula else {
            panic!("SANA declares a phase envelope")
        };
        assert!(!variables.contains(&MemoryFormulaVariable::TransformerWindowSize));
        assert!(variables.contains(&MemoryFormulaVariable::AttentionChunkSize));
    }

    /// **Every published parameter is refused outside its domain, on the production layer — for BOTH
    /// catalog entries.**
    ///
    /// SC-18607: the pair shares one provider module, and sharing code is explicitly not what makes
    /// an entry covered. Sprint carried no domain-enforcement assertion of its own before this, so a
    /// refusal keyed on the base id would have shipped Sprint an unenforced ladder.
    #[test]
    fn request_scoped_parameters_are_refused_outside_the_published_domain() {
        for provider in [crate::MODEL_ID, crate::SPRINT_MODEL_ID] {
            let spec = streamable_spec(OffloadPolicy::Sequential);
            let full = GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                chunk_attention: true,
                stream_transformer_blocks: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE as u32),
                decode_overlap: Some(DECODE_OVERLAP as u32),
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                transformer_window_component: Some(TransformerComponent::Dit),
                ..Default::default()
            };
            // Rung 4 is withheld, so the fully published SELECTION is rungs 1-3.
            let published = GenerationMemory {
                stream_transformer_blocks: false,
                transformer_window_size: None,
                transformer_window_component: None,
                ..full
            };
            validate_request_memory(provider, &spec, &published)
                .expect("the fully published selection must be accepted");
            // …and every rung-4 selection is refused by name, whatever its parameters.
            for window in TRANSFORMER_WINDOW_SIZES {
                let error = validate_request_memory(
                    provider,
                    &spec,
                    &GenerationMemory {
                        transformer_window_size: Some(*window),
                        ..full
                    },
                )
                .expect_err("a withheld rung must be refused");
                assert!(
                    error.to_string().contains("WITHHELD"),
                    "{provider}: the refusal must name the withdrawal, got: {error}"
                );
            }
            // Every published value in every domain is reachable.
            for edge in DECODE_TILE_EDGES {
                let mut memory = published;
                memory.decode_tile_edge = Some(*edge);
                validate_request_memory(provider, &spec, &memory).unwrap();
            }
            for size in ATTENTION_CHUNK_SIZES {
                let mut memory = published;
                memory.attention_chunk_size = Some(*size);
                validate_request_memory(provider, &spec, &memory).unwrap();
            }
            let mut refusals = Vec::new();
            let mut check = |label: &str, memory: GenerationMemory| {
                if validate_request_memory(provider, &spec, &memory).is_ok() {
                    refusals.push(label.to_owned());
                }
            };
            for rejected in DECODE_TILE_EDGES_REJECTED {
                check(
                    "decode edge",
                    GenerationMemory {
                        decode_tile_edge: Some(*rejected),
                        ..published
                    },
                );
            }
            for rejected in ATTENTION_CHUNK_SIZES_REJECTED.iter().chain(&[0, 7, 999]) {
                check(
                    "attention budget",
                    GenerationMemory {
                        attention_chunk_size: Some(*rejected),
                        ..published
                    },
                );
            }
            for rejected in [0_u32, 3, 6, 7, 11, 20, 21] {
                check(
                    "window cadence",
                    GenerationMemory {
                        transformer_window_size: Some(rejected),
                        ..full
                    },
                );
            }
            for component in [
                TransformerComponent::TextEncoder,
                TransformerComponent::Both,
            ] {
                check(
                    "window component",
                    GenerationMemory {
                        transformer_window_component: Some(component),
                        ..full
                    },
                );
            }
            check(
                "rung 4 without rung 1",
                GenerationMemory {
                    stage_residency: false,
                    ..full
                },
            );
            assert!(
                refusals.is_empty(),
                "{provider} silently admitted these out-of-domain selections: {refusals:?}"
            );

            // …and rung 4 is refused outright on a load that cannot stream it.
            let mut eager = spec.clone();
            eager.load_shape = LoadShape::EagerMaterialization;
            assert!(validate_request_memory(provider, &eager, &full).is_err());
            // …while a request that selects no rung is untouched by any of this.
            validate_request_memory(provider, &eager, &GenerationMemory::default()).unwrap();
            validate_request_memory(provider, &eager, &published).unwrap();
        }
    }

    /// **The declaration fails closed on every load axis the loader refuses or silently discards.**
    ///
    /// SC-18607's defect class is a ladder that is declared but that no route reaches. The loader
    /// refuses a precision override, an adapter set and a single-file source, and it resolves every
    /// component from the snapshot root — so a caller-provisioned control/PiD/identity/external-TE
    /// slot is discarded rather than honoured. Before this gate every one of those shapes still got
    /// the full published ladder from `weights_free_memory_strategy_contract`.
    ///
    /// Each axis is mutated **individually** against a known-good spec, so this proves each member
    /// of the gate rather than the set.
    #[test]
    fn the_declaration_fails_closed_on_every_load_axis_the_loader_refuses() {
        let good = streamable_spec(OffloadPolicy::Sequential);
        weights_free_memory_strategy_contract(crate::MODEL_ID, &good)
            .expect("the control spec must still be declarable");

        let mut cases: Vec<(&str, LoadSpec)> = Vec::new();
        cases.push((
            "single-file source",
            LoadSpec::new(WeightsSource::File("/nonexistent/sana.safetensors".into()))
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(LoadShape::DeferredMaterialization),
        ));
        let mut fp32 = good.clone();
        fp32.precision = mlx_gen::Precision::Fp32;
        cases.push(("precision override", fp32));
        let mut adapted = good.clone();
        adapted.adapters.push(mlx_gen::AdapterSpec::new(
            "/nonexistent/lora.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        cases.push(("adapter", adapted));
        let mut control = good.clone();
        control.control = Some(WeightsSource::File(
            "/nonexistent/control.safetensors".into(),
        ));
        cases.push(("control", control));
        let mut extra_control = good.clone();
        extra_control
            .extra_controls
            .push(WeightsSource::File("/nonexistent/extra.safetensors".into()));
        cases.push(("extra control", extra_control));
        let mut ip_adapter = good.clone();
        ip_adapter.ip_adapter = Some(WeightsSource::Dir("/nonexistent/ip".into()));
        cases.push(("ip adapter", ip_adapter));
        cases.push((
            "pid",
            good.clone().with_pid(
                WeightsSource::File("/nonexistent/pid.safetensors".into()),
                WeightsSource::Dir("/nonexistent/gemma".into()),
            ),
        ));
        let mut identity = good.clone();
        identity.identity = Some(Default::default());
        cases.push(("identity", identity));
        let mut text_encoder = good.clone();
        text_encoder.text_encoder = Some(WeightsSource::Dir("/nonexistent/external-text".into()));
        cases.push(("external text encoder", text_encoder));
        let mut component = good.clone();
        component.components.insert(
            "unexpected".to_owned(),
            WeightsSource::File("/nonexistent/unexpected.safetensors".into()),
        );
        cases.push(("named component", component));

        let mut admitted = Vec::new();
        for (label, spec) in &cases {
            for provider in [crate::MODEL_ID, crate::SPRINT_MODEL_ID] {
                if weights_free_memory_strategy_contract(provider, spec).is_ok() {
                    admitted.push(format!("{provider} weights-free {label}"));
                }
                if memory_strategy_contract(provider, spec).is_ok() {
                    admitted.push(format!("{provider} production {label}"));
                }
            }
        }
        // An unknown id is the same class: a contract minted for a route the catalog does not serve.
        for unknown in ["sana", "sana_1600m_turbo", "sana_sprint", ""] {
            if weights_free_memory_strategy_contract(unknown, &good).is_ok() {
                admitted.push(format!("unknown id '{unknown}'"));
            }
        }
        assert!(
            admitted.is_empty(),
            "these unloadable shapes were still declared a ladder: {admitted:?}"
        );
    }

    // ------------------------------------------------------------------------------------------
    // sc-22731 (epic sc-22723 E1/E4): the production calibration identity is per
    // (provider, tier, offload policy), bound to the artifact on disk.
    // ------------------------------------------------------------------------------------------

    /// Write one safetensors file by hand: `(name, dtype, shape)` entries over a zero-filled data
    /// region. Only the HEADER matters to the tier seam, exactly as it does to the loader's
    /// packed-detect.
    fn write_safetensors(path: &std::path::Path, entries: &[(String, &str, Vec<usize>)]) {
        let width = |dtype: &str| match dtype {
            "BF16" => 2,
            "U32" | "F32" => 4,
            other => panic!("fixture dtype {other}"),
        };
        let mut offset = 0usize;
        let mut fields = Vec::new();
        for (name, dtype, shape) in entries {
            let bytes = shape.iter().product::<usize>() * width(dtype);
            let shape = shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let end = offset + bytes;
            fields.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape}],\"data_offsets\":[{offset},{end}]}}"
            ));
            offset = end;
        }
        let mut json = format!("{{{}}}", fields.join(",")).into_bytes();
        while json.len() % 8 != 0 {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::write(path, bytes).unwrap();
    }

    /// The packed triple `crate::convert` writes for one Linear `{base}` of `in = 64` at
    /// `GROUP_SIZE` 64 — u32 codes `[out, 64·bits/32]`, bf16 scales/biases `[out, 1]` — or the
    /// dense bf16 `[out, 64]` weight when `bits` is `None`.
    fn linear_entries(
        base: &str,
        out: usize,
        bits: Option<usize>,
    ) -> Vec<(String, &'static str, Vec<usize>)> {
        match bits {
            Some(bits) => vec![
                (format!("{base}.weight"), "U32", vec![out, 64 * bits / 32]),
                (format!("{base}.scales"), "BF16", vec![out, 1]),
                (format!("{base}.biases"), "BF16", vec![out, 1]),
            ],
            None => vec![(format!("{base}.weight"), "BF16", vec![out, 64])],
        }
    }

    /// A snapshot root laid out the way `crate::convert` writes the shipped turnkey tiers: the three
    /// component directories SANA resolves, each holding its single weight file, with the trunk
    /// Linears and the Gemma-2 projections packed at `bits` (`None` = dense bf16) as real
    /// `.weight`/`.scales`/`.biases` triples, the convs and the DC-AE dense, and **no**
    /// `transformer/config.json` — the turnkey never writes one.
    fn sana_tier_root(bits: Option<usize>) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        for component in ["transformer", "vae", "text_encoder"] {
            std::fs::create_dir_all(tmp.path().join(component)).unwrap();
        }
        let mut transformer = linear_entries("transformer_blocks.0.attn1.to_q", 8, bits);
        transformer.extend(linear_entries("proj_out", 8, bits));
        transformer.push((
            "patch_embed.proj.weight".to_owned(),
            "BF16",
            vec![8, 32, 1, 1],
        ));
        write_safetensors(
            &tmp.path()
                .join("transformer/diffusion_pytorch_model.safetensors"),
            &transformer,
        );
        let mut te = linear_entries("model.layers.0.self_attn.q_proj", 8, bits);
        te.push(("model.embed_tokens.weight".to_owned(), "BF16", vec![16, 64]));
        write_safetensors(
            &tmp.path().join("text_encoder/gemma-2-2b-it.safetensors"),
            &te,
        );
        write_safetensors(
            &tmp.path().join("vae/diffusion_pytorch_model.safetensors"),
            &[(
                "decoder.conv_out.weight".to_owned(),
                "F32",
                vec![3, 8, 3, 3],
            )],
        );
        assert!(
            !tmp.path().join("transformer/config.json").exists(),
            "the fixture must mirror the turnkey, which writes no transformer/config.json"
        );
        tmp
    }

    fn tier_spec(root: &std::path::Path, quant: Option<mlx_gen::Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_offload_policy(OffloadPolicy::Resident)
            .with_load_shape(LoadShape::EagerMaterialization);
        spec.quantize = quant;
        spec
    }

    /// The six shipped MLX cells (two routes x three tiers) each publish their OWN production
    /// identity, on both offload policies and both load shapes, and the measured `2026-08-06`
    /// strings stay byte-identical on the (`sana_1600m`, q4) cell they were captured on.
    ///
    /// Mutation that fails this: restoring `calibration_fingerprint(policy)` as the key — all six
    /// cells collapse onto two strings, which is the sc-22511 false green (a Sprint q8 anchor
    /// binding base SANA's q4 evidence).
    #[test]
    fn every_shipped_sana_cell_publishes_its_own_per_tier_identity_on_both_policies() {
        let mut published = std::collections::BTreeSet::new();
        for provider in [crate::MODEL_ID, crate::SPRINT_MODEL_ID] {
            for (bits, quant) in [
                (Some(4), Some(mlx_gen::Quant::Q4)),
                (Some(8), Some(mlx_gen::Quant::Q8)),
                (None, None),
            ] {
                for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
                    for shape in [
                        LoadShape::EagerMaterialization,
                        LoadShape::DeferredMaterialization,
                    ] {
                        let root = sana_tier_root(bits);
                        let spec = {
                            let mut spec = tier_spec(root.path(), quant);
                            spec = spec.with_offload_policy(policy).with_load_shape(shape);
                            spec
                        };
                        let label = format!("{provider} {quant:?} {policy:?} {shape:?}");
                        assert_eq!(resolved_artifact_tier(&spec).unwrap(), quant, "{label}");
                        let contract = memory_strategy_contract(provider, &spec).unwrap();
                        let phase = contract.phase_facts.as_ref().unwrap();
                        assert_eq!(
                            phase.architecture,
                            Some(
                                mlx_gen::gen_core::ImagePipelineArchitecture::SanaLinearDit {
                                    classifier_free_guidance: provider == crate::MODEL_ID,
                                }
                            ),
                            "{label}"
                        );
                        let decoder = phase.decoder_workspace.as_ref().unwrap();
                        assert_eq!(
                            decoder.channels,
                            crate::config::DcAeConfig::sana_f32c32()
                                .block_out_channels
                                .iter()
                                .rev()
                                .map(|c| *c as u32)
                                .collect::<Vec<_>>()
                        );
                        assert_eq!(decoder.activation_dtype_width, 4);

                        let identity = contract
                            .calibration
                            .as_ref()
                            .unwrap_or_else(|| panic!("{label}: no production identity"));
                        assert_eq!(identity.load_shape, shape, "{label}");
                        assert_eq!(
                            Some(identity.fingerprint.clone()),
                            production_calibration_fingerprint(provider, &spec),
                            "{label}"
                        );
                        // The weights-free conformance identity is a different namespace, so a
                        // fixture contract can never be read as evidence of this cell.
                        let fixture =
                            weights_free_memory_strategy_contract(provider, &spec).unwrap();
                        assert_ne!(
                            fixture.calibration.as_ref().unwrap().fingerprint,
                            identity.fingerprint,
                            "{label}"
                        );
                        published.insert(identity.fingerprint.clone());
                    }
                }
            }
        }
        assert_eq!(
            published.len(),
            2 * 3 * 2,
            "two (provider, tier, policy) cells share one identity: {published:?}"
        );
        // The retained measured cell, byte-for-byte.
        let q4 = sana_tier_root(Some(4));
        for (policy, expected) in [
            (OffloadPolicy::Sequential, MEMORY_CALIBRATION_FINGERPRINT),
            (
                OffloadPolicy::Resident,
                RESIDENT_MEMORY_CALIBRATION_FINGERPRINT,
            ),
        ] {
            assert_eq!(
                production_calibration_fingerprint(
                    CALIBRATED_ROUTE,
                    &tier_spec(q4.path(), Some(mlx_gen::Quant::Q4)).with_offload_policy(policy),
                )
                .as_deref(),
                Some(expected),
            );
        }
        assert_eq!(CALIBRATED_ROUTE, "sana_1600m");
        assert_eq!(CALIBRATED_TIER, "q4");
    }

    /// **The worker's real MLX SANA load publishes the retained measured identity over a
    /// convert-shaped q4 root.** SceneWorks loads SANA as `quantize = Some(Q4)` + `Resident` +
    /// `EagerMaterialization` (manifest `mlx.quantize: 4`) over a turnkey that carries NO
    /// `transformer/config.json`; the tier must come off the `.scales` shapes the loader itself
    /// packed-detects on, or every shipped render publishes nothing.
    ///
    /// Mutation that fails this: restoring the `transformer/config.json` existence check in
    /// `resolved_artifact_tier` — the convert-shaped root reads as unreadable and the contract's
    /// `calibration` is `None`.
    #[test]
    fn the_workers_real_q4_resident_eager_load_publishes_the_measured_resident_identity() {
        let q4 = sana_tier_root(Some(4));
        let spec = tier_spec(q4.path(), Some(mlx_gen::Quant::Q4));
        assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
        assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(
            resolved_artifact_tier(&spec).unwrap(),
            Some(mlx_gen::Quant::Q4)
        );
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract.calibration,
            Some(MemoryCalibrationIdentity::new(
                "sana-mlx-full-ladder-2026-08-06-v1-resident".to_owned(),
                LoadShape::EagerMaterialization,
            ))
        );
        // The q8 and dense turnkeys resolve off the same seam.
        for (bits, quant) in [(Some(8), Some(mlx_gen::Quant::Q8)), (None, None)] {
            let root = sana_tier_root(bits);
            let spec = tier_spec(root.path(), quant);
            assert_eq!(resolved_artifact_tier(&spec).unwrap(), quant, "{bits:?}");
        }
    }

    /// `quantize = None` over a packed turnkey is the same load as `quantize = Some(tier)` — the
    /// loader packed-detects either way — so [`requested_tier`] follows the artifact and the cell
    /// publishes its own tier's string, not the dense one and not nothing.
    ///
    /// Mutation that fails this: dropping the `or_else(resolved_artifact_tier)` fallback in
    /// `requested_tier` — the request reads as bf16, the binding sees `Some(Q4) != None`, and the
    /// contract publishes no identity.
    #[test]
    fn a_quantize_none_load_over_a_packed_root_publishes_the_artifacts_tier() {
        for (bits, quant, tier) in [(4, mlx_gen::Quant::Q4, "q4"), (8, mlx_gen::Quant::Q8, "q8")] {
            for provider in [crate::MODEL_ID, crate::SPRINT_MODEL_ID] {
                let root = sana_tier_root(Some(bits));
                let spec = tier_spec(root.path(), None);
                assert_eq!(requested_tier(&spec), Some(quant), "{provider} {tier}");
                let explicit = tier_spec(root.path(), Some(quant));
                let published = memory_strategy_contract(provider, &spec)
                    .unwrap()
                    .calibration
                    .unwrap_or_else(|| panic!("{provider} {tier}: no identity on quantize=None"));
                assert_eq!(
                    Some(published.fingerprint.clone()),
                    production_calibration_fingerprint(provider, &explicit),
                    "{provider} {tier}"
                );
                assert!(
                    published.fingerprint.contains(&format!("-{tier}-"))
                        || published.fingerprint == RESIDENT_MEMORY_CALIBRATION_FINGERPRINT
                );
                assert_eq!(
                    weights_free_calibration_fingerprint(provider, &spec),
                    weights_free_calibration_fingerprint(provider, &explicit),
                    "{provider} {tier}: the weights-free key must follow the artifact too"
                );
            }
        }
    }

    /// The tier in the published string is the tier of the artifact on disk, never the request knob
    /// alone: a dense snapshot asked for at q4 would be a load-time requantization no anchor
    /// measured, and a packed snapshot asked for the other packed tier is no shipped load either.
    /// A transformer whose packed width cannot be read fails closed.
    ///
    /// Mutation that fails this: deleting the `resolved_artifact_tier(spec) != requested_tier(spec)`
    /// refusal in `production_calibration_identity` — every mismatched cell publishes the requested
    /// tier's string over another tier's weights.
    #[test]
    fn the_production_identity_is_withheld_when_the_request_and_the_artifact_disagree() {
        for (bits, requested) in [
            (None, Some(mlx_gen::Quant::Q4)),
            (None, Some(mlx_gen::Quant::Q8)),
            (Some(4), Some(mlx_gen::Quant::Q8)),
            (Some(8), Some(mlx_gen::Quant::Q4)),
        ] {
            for provider in [crate::MODEL_ID, crate::SPRINT_MODEL_ID] {
                let root = sana_tier_root(bits);
                let spec = tier_spec(root.path(), requested);
                let label = format!("{provider} artifact {bits:?} requested {requested:?}");
                assert!(
                    production_calibration_fingerprint(provider, &spec).is_some(),
                    "{label}: the table still answers for the request knob"
                );
                assert!(
                    memory_strategy_contract(provider, &spec)
                        .unwrap()
                        .calibration
                        .is_none(),
                    "{label}: published an identity over another tier's weights"
                );
            }
        }
        // A root whose transformer weights cannot be read proves no tier at all: no `transformer/`
        // (the old weightless declaration root), a `transformer/` with no weight file, a packed
        // base whose codes/scales ratio is not a shipped width, and two bases packed differently.
        let empty = tempfile::tempdir().unwrap();
        let bare = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(bare.path().join("transformer")).unwrap();
        let unshipped = sana_tier_root(Some(4));
        write_safetensors(
            &unshipped
                .path()
                .join("transformer/diffusion_pytorch_model.safetensors"),
            &linear_entries("proj_out", 8, Some(2)),
        );
        let mixed = sana_tier_root(Some(4));
        let mut entries = linear_entries("proj_out", 8, Some(4));
        entries.extend(linear_entries(
            "transformer_blocks.0.attn1.to_q",
            8,
            Some(8),
        ));
        write_safetensors(
            &mixed
                .path()
                .join("transformer/diffusion_pytorch_model.safetensors"),
            &entries,
        );
        for (label, root) in [
            ("no transformer/", empty.path()),
            ("transformer/ without weights", bare.path()),
            ("2-bit pack", unshipped.path()),
            ("mixed q4/q8 pack", mixed.path()),
        ] {
            for requested in [None, Some(mlx_gen::Quant::Q4)] {
                let spec = tier_spec(root, requested);
                assert!(
                    resolved_artifact_tier(&spec).is_err(),
                    "{label} {requested:?}"
                );
                assert!(
                    memory_strategy_contract(crate::MODEL_ID, &spec)
                        .unwrap()
                        .calibration
                        .is_none(),
                    "{label} {requested:?}: published an identity over unreadable weights"
                );
            }
        }
    }

    /// **The declared ladder is invariant in the advisory `quantize` axis — asserted, not assumed.**
    ///
    /// Sibling families register a selector-aware surface resolver because their `LoadSpec::quantize`
    /// means "pack this dense source at load time", so the resolved artifact tier and the request are
    /// genuinely different facts. SANA's tiers are **packed-detected from the on-disk `.scales`**
    /// (sc-8489), so `quantize` never changes what rungs are declared. Rather than import another
    /// family's axis, that invariance is pinned here: if a future tier ever moves a rung, this goes
    /// red and the resolver becomes the right answer.
    ///
    /// **The calibration IDENTITY is deliberately not in that invariance (sc-22731).** The rungs a
    /// tier can run are the same; the memory each one costs is not, and an identity shared across
    /// tiers is what let a q8 anchor bind q4's evidence. The identity is asserted to MOVE with the
    /// tier here, so the invariance claim above can never be widened back over it by accident.
    #[test]
    fn the_declared_ladder_is_invariant_in_the_advisory_quantize_axis() {
        for provider in [crate::MODEL_ID, crate::SPRINT_MODEL_ID] {
            for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
                for shape in [
                    LoadShape::EagerMaterialization,
                    LoadShape::DeferredMaterialization,
                ] {
                    let base = LoadSpec::new(WeightsSource::Dir("/nonexistent/sana-tier".into()))
                        .with_offload_policy(policy)
                        .with_load_shape(shape);
                    let dense = weights_free_memory_strategy_contract(provider, &base).unwrap();
                    for quant in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
                        let mut packed = base.clone();
                        packed.quantize = Some(quant);
                        let contract =
                            weights_free_memory_strategy_contract(provider, &packed).unwrap();
                        for strategy in MemoryStrategy::ALL {
                            assert_eq!(
                                contract.capability(strategy),
                                dense.capability(strategy),
                                "{provider} {policy:?}/{shape:?} {quant:?} moved {strategy:?}"
                            );
                        }
                        assert_ne!(
                            contract.calibration, dense.calibration,
                            "{provider} {policy:?}/{shape:?}: {quant:?} must not share the dense \
                             tier's identity"
                        );
                        assert_eq!(
                            contract.lifecycle.transformer_window_materialization,
                            dense.lifecycle.transformer_window_materialization
                        );
                        assert_eq!(
                            contract.lifecycle.synchronized_phase_release,
                            dense.lifecycle.synchronized_phase_release
                        );
                        assert_eq!(
                            contract.resident_request_memory,
                            dense.resident_request_memory
                        );
                    }
                }
            }
        }
    }

    /// **The three sc-15839 defect classes, pre-checked against this provider.**
    ///
    /// The epic requires a new rung to answer them rather than inherit a sibling's answer, and two
    /// of the three are answered by a *property* rather than by code, which is exactly the kind of
    /// answer that rots silently if it is only written in prose.
    ///
    /// 1. **F32 materialization underpricing** — [`resident_decoder_bytes`] projects the DC-AE at
    ///    its resident width; its own doc carries the reasoning.
    /// 2. **A resident overlay priced as zero** — SANA has none (no ControlNet branch, and
    ///    `mlx-gen-pid` is used only for Gemma-2 conditioning, never as a decoder overlay), so
    ///    `overlay_bytes` is zero and there are no auxiliary resident components. Asserted so that a
    ///    future overlay cannot be added while the contract keeps saying the request costs nothing
    ///    extra.
    /// 3. **Unconstrained batch geometry** — the formula declares `BatchCount`, the descriptor caps
    ///    `max_count`, and `validate_request` refuses above it. A memory contract whose peak scales
    ///    with a batch nothing bounds is a contract that cannot predict its own worst case.
    #[test]
    fn the_sc15839_pricing_defect_classes_are_answered_for_this_provider() {
        let contract = weights_free_memory_strategy_contract(
            crate::MODEL_ID,
            &streamable_spec(OffloadPolicy::Sequential),
        )
        .unwrap();
        assert_eq!(contract.asset_facts.overlay_bytes, 0);
        assert_eq!(contract.auxiliary_resident_bytes(), 0);
        assert!(
            contract.resident_components().is_empty(),
            "SANA declares no auxiliary resident component; an overlay must revisit the pricing"
        );
        let MemoryFormulaKind::PhaseEnvelope { variables, .. } = &contract.formula else {
            panic!("SANA declares a phase envelope")
        };
        assert!(variables.contains(&MemoryFormulaVariable::BatchCount));

        // The batch the formula scales with is bounded by the descriptor, and the bound is enforced.
        let descriptor = crate::model::descriptor();
        assert_eq!(descriptor.capabilities.max_count, 8);
        let over_batch = mlx_gen::GenerationRequest {
            prompt: "a red fox".into(),
            width: 1024,
            height: 1024,
            count: descriptor.capabilities.max_count + 1,
            ..Default::default()
        };
        assert!(
            crate::model::validate_request(&descriptor, &over_batch).is_err(),
            "an unbounded batch would make the declared BatchCount term unpredictable"
        );
    }

    /// **An evidence record built the way the harness builds one actually serializes.**
    ///
    /// `MemoryEvidenceLogRecord::to_json_line` refuses a record that fails any of a dozen
    /// validations — a malformed calibration fingerprint, a declared/observed identity mismatch, a
    /// non-canonical engaged composition. Every one of those is a property of THIS provider's
    /// contract, and discovering one of them from a real-weight run is discovering it after the
    /// expensive part. This asserts the shape without weights so the sweep cannot be wasted, and it
    /// covers both the Resident-load identity and the Sequential one, which differ.
    #[test]
    fn the_evidence_record_this_provider_produces_is_serializable() {
        use mlx_gen::gen_core::{
            MemoryBackend, MemoryEvidenceKey, MemoryEvidenceLogRecord, MemoryGeometry,
            MemoryParityContract, MemoryParityResult, MemoryStrategyParameters,
        };

        for (policy, shape, strategy) in [
            (
                OffloadPolicy::Resident,
                LoadShape::EagerMaterialization,
                MemoryStrategy::Resident,
            ),
            (
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
                MemoryStrategy::BoundedAttention,
            ),
        ] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/sana-evidence".into()))
                .with_offload_policy(policy)
                .with_load_shape(shape);
            let contract = weights_free_memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
            let record = MemoryEvidenceLogRecord {
                key: MemoryEvidenceKey {
                    model_family: "sana".to_owned(),
                    resolved_route: crate::MODEL_ID.to_owned(),
                    backend: MemoryBackend::Mlx,
                    tier: MemoryNumericTier {
                        precision: spec.precision,
                        quant: spec.quantize,
                        component_precision_floors: &[],
                    },
                    load_shape: spec.load_shape,
                    mode: mlx_gen::gen_core::MemoryMode::TextToImage,
                    reference_shape: mlx_gen::gen_core::MemoryReferenceShape::None,
                    overlay: None,
                    geometry: MemoryGeometry {
                        width: 1024,
                        height: 1024,
                        batch: 1,
                        frames: 1,
                        reference_count: 0,
                    },
                    frames_per_second: None,
                    strategy,
                    engaged_composition: contract.engaged_composition(strategy),
                    parameters: MemoryStrategyParameters::default(),
                },
                declared_calibration: MemoryCalibrationIdentity::new(
                    weights_free_calibration_fingerprint(crate::MODEL_ID, &spec)
                        .expect("the base route at a shipped tier has a conformance identity"),
                    spec.load_shape,
                ),
                observed_calibration: contract.calibration.clone().unwrap(),
                predicted_peak_bytes: 5_000_000_000,
                observed_peak_bytes: 5_000_000_000,
                inference_revision: "a".repeat(40),
                sceneworks_revision: "b".repeat(40),
                model_revision: "c".repeat(40),
                model_inventory_sha256: "d".repeat(64),
                harness_version: "inference-sana-memory-ladder-v1".to_owned(),
                output_sha256: "e".repeat(64),
                parity: MemoryParityContract::Exact,
                // `NotRun` is honest here: this weights-free smoke only proves the record
                // serializes — it renders nothing, compares nothing, and its line is never emitted
                // as evidence. The real-weight ladder harness earns `Passed` per row by asserting
                // each optimized row's sha against its rung-0 row (sc-17861).
                parity_result: MemoryParityResult::NotRun,
            };
            record.to_json_line().unwrap_or_else(|error| {
                panic!("{policy:?}/{shape:?} evidence record must serialize: {error}")
            });
        }
    }

    #[test]
    fn decode_domain_and_rejection_set_are_pinned() {
        assert_eq!(DECODE_TILE_EDGES, &[512, 384, 256, 192]);
        assert_eq!(DECODE_TILE_EDGES_REJECTED, &[128, 96, 64]);
        assert_eq!(DECODE_TILE_EDGE, 192);
        assert_eq!(DECODE_OVERLAP, 48);
        let routes = decode_routes(crate::MODEL_ID).unwrap();
        routes.validate(false, Some(192), Some(48)).unwrap();
        assert!(routes.validate(false, Some(128), Some(48)).is_err());
        assert!(routes.validate(true, Some(192), Some(48)).is_err());
    }

    #[test]
    fn calibration_identity_is_split_by_offload_policy() {
        let resident =
            weights_free_memory_strategy_contract(crate::MODEL_ID, &spec(OffloadPolicy::Resident))
                .unwrap();
        let sequential = weights_free_memory_strategy_contract(
            crate::MODEL_ID,
            &spec(OffloadPolicy::Sequential),
        )
        .unwrap();
        assert_ne!(resident.calibration, sequential.calibration);
        assert_eq!(
            resident
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            sequential
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert!(resident.pid_decode_routes.is_none());
        assert!(sequential.pid_decode_routes.is_none());
    }
}
