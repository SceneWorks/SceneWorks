//! MLX unified-memory pre-load fit-gate (epic 10834 Phase 0 sc-10835; Phase 1 residency selection
//! sc-10839).
//!
//! The unified-memory sibling of the candle `vram_gate.rs` (epic 10765, sc-10766/sc-10821). Before an
//! MLX generation loads, predict the model's whole-model peak footprint and compare it against the
//! machine's unified-memory budget. Three outcomes, mirroring the candle gate:
//!  - **Fits** (or no signal) → load resident (the warm, cross-job path).
//!  - **Won't fit resident, but the provider supports sequential component residency (sc-10839) and
//!    the staged max-single-component peak WILL fit** → select [`OffloadPolicy::Sequential`] so the
//!    engine drops the text encoder(s) before the DiT loads, bounding peak to the largest component.
//!  - **Won't fit even staged** → reject with an actionable message instead of a SIGKILL / Metal-OOM
//!    mid-render.
//!
//! Also honors [`MLX_MEMORY_CAP_ENV`], which emulates a smaller Mac on big hardware so the sequential
//! residency selection can be validated on the dev box's 128 GB machine without 16/32 GB hardware.
//!
//! ## Why this budgets on PREDICTED (on-disk) bytes, not live allocator deltas
//! MLX materializes weights lazily on first forward, so `get_active_memory()` reads ~0 right after
//! `load` — a post-load delta would see nothing. And a wired-memory overcommit SIGKILLs the process
//! rather than returning a catchable error, so we cannot "load and catch the OOM." This is therefore
//! a pre-load ADMISSION check keyed off the summed on-disk component weight bytes plus a fixed
//! headroom, never a post-allocation accounting number — the same conclusion the candle gate reached
//! for a different (caching-allocator) reason.
//!
//! Complements FLUX.2-dev edit's provider-owned multi-reference safety policy (`image_jobs/flux2.rs`):
//! that policy gates one activation-bound edit path; this gates base weight-fit for every MLX model.
//!
//! The pure decision logic is cross-platform and unit-tested on every lane; only the live
//! `sysctl hw.memsize` probe is macOS-only (it returns `None` elsewhere, so the gate no-ops).

#[path = "mlx_fit_gate/calibration.rs"]
mod calibration;
pub(crate) use calibration::*;

#[path = "mlx_fit_gate/request.rs"]
mod request;
pub(crate) use request::*;

#[path = "mlx_fit_gate/residency.rs"]
mod residency;
pub(crate) use residency::*;

#[path = "mlx_fit_gate/evidence.rs"]
mod evidence;
pub(crate) use evidence::*;

#[path = "mlx_fit_gate/video.rs"]
mod video;
#[cfg_attr(not(native_lane), allow(unused_imports))]
pub(crate) use video::*;

#[path = "mlx_fit_gate/full_finetune.rs"]
mod full_finetune;
pub use full_finetune::*;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use gen_core::{
    GenerationMemory, LoadShapeDeclarationResult, LoadSpec, MemoryBackend,
    MemoryBackendRealization, MemoryBudget, MemoryCacheState, MemoryConformanceState,
    MemoryDecodeArtifactIdentity, MemoryDecodeGeometryPolicy, MemoryDecodePolicyQuery,
    MemoryDecodeQualityDisposition, MemoryDecodeQualityFixture, MemoryDecodeQualityRuntimeIdentity,
    MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryGeometry, MemoryMode,
    MemoryNumericTier, MemoryOptimizationAuthority, MemoryParityContract, MemoryParityResult,
    MemoryProviderContract, MemoryReferenceShape, MemoryRunContext, MemorySelection,
    MemoryStrategy, OffloadPolicy, PerComponentBytes, Precision, Quant, TransformerComponent,
    WeightsSource,
};
#[cfg(test)]
use gen_core::{MemoryEvidenceVerdict, StagedResidencyAvailability};
use sceneworks_core::memory_calibration::{
    Backend as CalibrationBackend, CalibrationBinding, EvidenceBundle, EvidenceMismatchReason,
    EvidenceQuery, EvidenceVerdict, Geometry as CalibrationGeometry, LoadShapeKey, StrategyRung,
};
use serde::Deserialize;
use serde_json::{Map as JsonObject, Value};

use crate::fit_gate::resolve_offload;
pub(crate) use crate::fit_gate::FitDecision;
use crate::memory_strategy::memory_mode_from_mode_key;
use crate::model_jobs::ResolvedArtifactProvenance;
use crate::{WorkerError, WorkerResult};

use crate::memory_route_registry::MLX_REQUEST_EVIDENCE_REVISION as REQUEST_EVIDENCE_REVISION;
#[cfg(test)]
use sceneworks_core::inference_provenance::INFERENCE_REVISION;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct DecodeQualityFixtureWire {
    seed: u64,
    production_latent_provenance: String,
    production_latent_sha256: String,
    dense_output_sha256: String,
    tiled_output_sha256: String,
    observed_error: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct DecodeQualityGeometryWire {
    width: u32,
    height: u32,
    batch: u32,
    frames: u32,
    reference_count: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct DecodeQualityArtifactWire {
    repository: String,
    revision: String,
    variant: String,
    fingerprint: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
enum DecodeQualityDispositionWire {
    Admitted,
    Refused { reason: String },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct DecodeGeometryPolicyWire {
    quality_abi: u32,
    family: String,
    resolved_route: String,
    backend: String,
    tier: String,
    load_shape: String,
    artifact: DecodeQualityArtifactWire,
    implementation_fingerprint: String,
    mode: String,
    overlay: Option<String>,
    geometry: DecodeQualityGeometryWire,
    use_pid: bool,
    tile_edge: u32,
    overlap: u32,
    metric: String,
    maximum_error: u32,
    fixtures: Vec<DecodeQualityFixtureWire>,
    production_evidence_sha256: String,
    disposition: DecodeQualityDispositionWire,
}

/// Typed load-time disposition for a packaged quality table. A refused binding removes semantic
/// decode authority only; dense decode and independent higher memory rungs remain available.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) enum DecodeQualityBindingDecision {
    Legacy,
    Bound {
        artifact: MemoryDecodeArtifactIdentity,
        row_count: usize,
    },
    Refused {
        reason: String,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct DecodeQualityPolicyBinding {
    pub(crate) policies: Vec<MemoryDecodeGeometryPolicy>,
    /// True whenever the manifest declared semantic decode scope, including a typed refusal that
    /// strips every row before provider construction. This prevents a refused table from becoming
    /// indistinguishable from a genuinely legacy route.
    pub(crate) authoritative: bool,
    pub(crate) runtime_identity: Option<MemoryDecodeQualityRuntimeIdentity>,
    pub(crate) decision: DecodeQualityBindingDecision,
}

/// Load-invariant inputs used to estimate each request without putting geometry or strategy in the
/// generator cache key.
#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct MlxRequestPlan {
    engine_id: &'static str,
    model_id: String,
    tier: MemoryNumericTier,
    asset_bytes: u64,
    /// Control checkpoint bytes already folded into `asset_bytes` by the legacy on-disk estimator.
    /// An adopting contract replaces this raw source size with its load-exact typed residency.
    folded_control_bytes: u64,
    /// Adapter source bytes included by the legacy estimator. An adopting contract replaces this
    /// raw file size with its load-exact typed residency. Dense Wan adapters are folded into base
    /// weights and therefore contribute zero here; packed Wan residuals remain additive.
    folded_adapter_bytes: u64,
    activation_headroom_bytes: u64,
    /// The resolution-INDEPENDENT slice of `activation_headroom_bytes` (sc-16195). Always
    /// `<= activation_headroom_bytes`, so the area term below can never go negative.
    fixed_reserve_bytes: u64,
    calibration: MlxCalibrationConfig,
    /// Terminal declaration refusals retain safe eager execution, but may not consume any optimized
    /// provider strategy later in the cache-aware selector. NotEvaluated preserves every legacy
    /// route; Applied preserves the exact declaration-owned provider contract.
    load_shape_declaration_result: LoadShapeDeclarationResult,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct MlxCalibrationBinding {
    query: CalibrationBinding,
    provider: String,
    tier: String,
    mode: String,
    overlay: String,
    geometry: CalibrationGeometry,
    rung: StrategyRung,
    parameters: JsonObject<String, Value>,
    selection_parameters: gen_core::MemoryStrategyParameters,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct MlxCalibrationSet {
    bindings: Vec<MlxCalibrationBinding>,
    resolved: ResolvedArtifactProvenance,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
enum MlxCalibrationConfig {
    Absent,
    Valid(MlxCalibrationSet),
    /// The manifest declares bindings but the resolver could not establish immutable artifact
    /// provenance, so no binding may be trusted for this request. Distinct from [`Self::Absent`]
    /// so telemetry can tell "this model declares no evidence" from "this INSTALL cannot prove
    /// which artifact it holds" — the two demand different follow-up.
    Unproven,
    Invalid(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
enum AdmissionPath {
    Evidence,
    Legacy,
}

/// Why a request left the Evidence path for the legacy/estimate selector. Every reason is a
/// structural or identity fact about the packaged evidence and the install; none is measurement
/// currency (sc-22738) — a binding whose provider closure has moved since capture is routed exactly
/// as one whose closure has not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
enum LegacyAdmissionReason {
    PackagedEmpty,
    NoBinding,
    NoRecord,
    OutOfEnvelope,
    /// The manifest binding cites a calibration campaign the packaged record does not carry.
    FingerprintMismatch,
    /// The manifest binding describes different artifact bytes, ABI, or materialization shape
    /// than the packaged record — or the loaded provider's calibration identity does not match the
    /// candidates the route assembled.
    IdentityMismatch,
    /// The manifest opted in but the install could not prove its artifact identity.
    NoProvenance,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct VerifiedAdmissionCandidate {
    evidence: MemoryEvidence,
    /// Reserve enforced on this live host and passed to MLX as an absolute process ceiling.
    foreign_reserve_bytes: u64,
    /// Actionable static host boundary under the captured reserve policy: the smallest host that
    /// satisfies the reserve policy at the measured peak.
    minimum_host_bytes: u64,
    record_id: String,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct VerifiedGeometryAlternative {
    geometry: CalibrationGeometry,
    calibration_abi: u32,
    calibration_fingerprint: String,
    /// The materialization shape the alternative was MEASURED under (sc-18101). Part of the
    /// calibration identity (`MemoryCalibrationIdentity::load_shape`), so the identity demotion
    /// must compare it here for the same reason it compares the abi and fingerprint: an
    /// alternative measured under a shape this load does not use is advice the very next request
    /// would not honour.
    load_shape: gen_core::LoadShape,
    strategy: MemoryStrategy,
    parameters: gen_core::MemoryStrategyParameters,
    engaged_composition: Vec<MemoryStrategy>,
}

/// One verified measured cell usable as the extrapolation basis for a fitted-curve estimate
/// (sc-18096): same provider, tier, mode, and overlay as the request, an artifact-current binding,
/// but a DIFFERENT geometry — the cell the request itself could not be admitted on.
///
/// The binding's provider closure is NOT a conjunct (sc-22738). Until this story a basis also had
/// to be closure-current, on the argument that a stale record's own cell was already paying the
/// recapture spread and an extrapolation from it would stack closure drift under a single
/// allowance. That argument priced closure drift as an uncertainty the runtime should charge for;
/// Michael's standing rule says the runtime charges nothing for currency — the measurement is
/// used as if valid, and a moved closure is the probe tooling's cue to re-capture.
///
/// Everything the extrapolation, the binding-phase constraint, and the loaded-provider identity
/// gate need is captured here, so the synthesis step never re-reads the bundle.
#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct MeasuredRungBasis {
    rung: StrategyRung,
    parameters: gen_core::MemoryStrategyParameters,
    engaged_composition: Vec<MemoryStrategy>,
    load_shape: gen_core::LoadShape,
    /// The calibration identity the basis binding was measured under. `synthesize_estimate_ladder`
    /// requires it to equal the LOADED contract's identity: a provider whose estimator drifted
    /// from the packaged records must not receive fitted candidates built from them (sc-18096
    /// review). This gate cannot be left to the `carries_verified_claim` demotion, which only
    /// fires when the route carries a verified claim (`Evidence` path or a named lower
    /// alternative) — bases ride on legacy routes where neither may hold.
    calibration_abi: u32,
    calibration_fingerprint: String,
    geometry: CalibrationGeometry,
    /// Per-phase predicted peaks from the measured record, in canonical phase order
    /// (conditioning, denoise, decode). The binding phase is their argmax.
    conditioning_peak_bytes: u64,
    denoise_peak_bytes: u64,
    decode_peak_bytes: u64,
    /// The measured admission envelope peak the extrapolated estimate scales.
    envelope_peak_bytes: u64,
    record_id: String,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct AdmissionRoute {
    path: AdmissionPath,
    fallback_reason: Option<LegacyAdmissionReason>,
    evidence: Vec<VerifiedAdmissionCandidate>,
    /// Measured cells available as fitted-curve estimate bases when the request itself is not
    /// covered (sc-18096). Populated only on legacy fallbacks of a calibrated provider whose
    /// artifact identity is current; empty on every covered (`Evidence`) route.
    estimate_bases: Vec<MeasuredRungBasis>,
    evidence_revision: Option<String>,
    process_limit_bytes: Option<u64>,
    lower_alternative: Option<VerifiedGeometryAlternative>,
}

/// Exact request axes that must be reconsidered for both cold and warm cache runs.
#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct MlxRequestInputs {
    pub width: u32,
    pub height: u32,
    /// Original job image count. This is a SCHEDULING quantity, not a memory one — see
    /// [`request_batch`] for why it must never reach a geometry's `batch`. Kept here so the
    /// over-budget message can quote the request the operator actually submitted.
    pub count: u32,
    /// Images ONE provider pass renders: `1` on every sequential lane, and `prompts × count` for a
    /// native prompt batch (sc-25681, `image.maxPromptBatchImages`) — the genuine batched pass
    /// [`request_batch`] admits against.
    pub native_batch: u32,
    pub mode: String,
    pub overlay: Option<String>,
    pub adapter_count: usize,
    pub has_reference: bool,
    pub reference_count: u32,
    /// Actual owned reference geometry/tokens, never a client numeric memory hint. The selected
    /// backend prices these from the current candidate LoadSpec before every probe/fit decision.
    pub reference_resources: Option<gen_core::MemoryReferenceResourceFacts>,
    pub use_pid: bool,
    pub has_phases: bool,
    /// Conservative count of CLIP windows for the actual positive/negative prompt pair.
    /// None means the caller has not supplied prompt demand; no single-window estimate is used.
    pub conditioning_windows: Option<u32>,
    /// Whether the request evaluates a true-CFG NEGATIVE prompt branch
    /// (`execution_planner::evaluates_negative_branch`). Read only by a provider that prices a
    /// prefix KV cache, whose cache is held once per prompt branch (epic 25899); `false` on every
    /// lane that never reaches one.
    pub negative_branch: bool,
    /// Whether the memory block admitted here is handed to ONE render at exactly this geometry and
    /// reference set, so a prefix KV cache may be priced onto it (epic 25899). `false` for a
    /// multi-pass render (hires fix: the first pass renders at the base size with the caller's own
    /// references and re-resolved CFG, under the same block) and on every lane that never reaches
    /// a cache provider.
    pub prefix_kv_cache_eligible: bool,
}

/// A live pricing context, not a cached request cost: every tier probe binds its actual spec.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct ReferencePricing<'a> {
    spec: &'a LoadSpec,
    registry: &'a gen_core::ProviderRegistry,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct MlxRequestEvaluation {
    pub memory: GenerationMemory,
    pub context: MemoryRunContext,
    /// Typed semantic-decode admissions and refusals considered for this exact request. Kept
    /// separate from memory evidence so no quality receipt can be mistaken for a measurement.
    pub decode_quality_decisions: Vec<DecodeQualityRequestDecision>,
    /// Request-scoped soft MLX ceiling derived from an exact verified cell. Legacy cells keep the
    /// process-global #1947 fallback. Applying this never changes the wired limit.
    pub process_limit_bytes: Option<u64>,
    /// What the admission decided about the provider's prefix KV cache (epic 25899): `None` for
    /// every provider that prices none. A `Cached` plan is already written onto [`Self::memory`]
    /// and counted in `context.predicted_peak_bytes`.
    pub prefix_kv_cache: Option<crate::execution_planner::PrefixKvCachePlan>,
}

/// Live hidden-width (3072) activation buffers of ONE FLUX3 `Block::forward` (pinned BFL
/// `dit.py` / native `dit.rs`), per token, counted from the tensors alive at its widest point
/// with NO intra-block reuse assumed: x 1, modulated x_mod 1, q/k/v projections 3, their
/// contiguous head-split copies 3, f32 RMSNorm+RoPE q/k 2, attention output 1, mlp_in
/// (2 x mlp_ratio 3) 6, silu(gate) 3, silu(gate)*up 3, mlp_out 1, attn_out 1, delta 1,
/// gate*delta 1, residual 1 = 28. Blocks run sequentially, so only one block's set is live.
pub(crate) const FLUX3_BLOCK_LIVE_HIDDEN_BUFFERS: u64 = 28;

/// One estimate-backed candidate synthesized for an implemented-but-unmeasured rung (sc-18096).
#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct SynthesizedEstimate {
    selection: MemorySelection,
    evidence: MemoryEvidence,
    basis: crate::memory_strategy::CandidateBasis,
    /// The ACTIVATION slice of `evidence.predicted_peak_bytes`, declared by the arm that built the
    /// peak (sc-22508). `Some` only on the weights+headroom floor arm, which composes the peak here
    /// as `estimate_floor_weights_bytes + generic_headroom_bytes` and therefore knows the split;
    /// the fitted-curve arm scales a measured envelope and has none to declare.
    unmodeled_activation_bytes: Option<u64>,
    /// Which activation term the peak was composed with — `Some` only on the weights+headroom
    /// floor arm. sc-25909 reads it to know whether the rung was charged the generic area-scaled
    /// headroom (and nothing request-specific) as its activation.
    activation_term: Option<ImageFloorActivationTerm>,
    decode_quality: DecodeQualityRequestDecision,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) enum DecodeQualityRequestDecision {
    NotDeclared,
    NotEngaged {
        strategy: MemoryStrategy,
    },
    Admitted {
        strategy: MemoryStrategy,
        tile_edge: u32,
        overlap: u32,
        evidence_sha256: String,
    },
    Refused {
        strategy: MemoryStrategy,
        tile_edge: Option<u32>,
        overlap: Option<u32>,
        evidence_sha256: Option<String>,
        reason: String,
    },
    Unmeasured {
        strategy: MemoryStrategy,
        reason: String,
    },
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct EstimateParameterCandidate {
    parameters: gen_core::MemoryStrategyParameters,
    decode_quality: DecodeQualityRequestDecision,
}

#[derive(Default)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct SynthesizedEstimateLadder {
    estimates: Vec<SynthesizedEstimate>,
    decode_quality_decisions: Vec<DecodeQualityRequestDecision>,
}

/// The canonical measurement phases, in ladder order, used for the binding-phase comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
enum EstimatePhase {
    Conditioning,
    Denoise,
    Decode,
}

/// Which ACTIVATION term the image floor composes [`image_floor_weights_bytes`] with — the input
/// that decides where rung 4's window slice is charged when the facts state no block count (see
/// the `transformer_blocks: None` regime on [`estimate_floor_weights_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) enum ImageFloorActivationTerm {
    /// The sc-22663 law's per-phase residue for the rung's regime
    /// ([`mlx_image_anchor_activation_residue`]): weights-free and window-free, so the weights
    /// term must carry every resident weight itself.
    LawResidue,
    /// `MlxRequestPlan::generic_headroom_bytes`: the pre-epic geometry-blind allowance, which the
    /// sc-18096 accounting defines as carrying rung 4's window slice.
    GenericHeadroom,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
enum MlxAdmissionPhase {
    TierProbe,
    Generation,
}

/// A budget refusal is distinct from invalid artifacts/contracts: only the former can try a
/// lower installed tier. Keep the original request-scoped diagnostic through tier selection.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) enum MlxRequestAdmission {
    Admitted(Box<MlxRequestEvaluation>),
    Rejected(WorkerError),
}

/// A family's flat 1024² allowance, together with how much of it is fixed OS/app reserve rather than
/// this request's area-dependent activation transient.
///
/// The two constants that fill this slot are NOT the same kind of quantity, and conflating them is a
/// live correctness bug rather than a naming wart:
///
/// * [`HEADROOM_GB`] (18) is a measured transient PLUS an OS/app reserve — sc-10863 built it as
///   14.04 + ~4.
/// * [`LENS_DENSE_HEADROOM_GB`] (29.88) is a measured transient ALONE — sc-11924 took it straight
///   from `peak − resident` on lens-turbo bf16, and the weight half of that family's error is
///   corrected separately via `materialized_expansion` in [`spec_component_bytes`]. There is no
///   reserve inside it to hold out.
///
/// sc-16195's split therefore cannot be applied blindly. Subtracting a 2 GiB fixed reserve from the
/// lens-dense allowance would take those 2 GiB out of its AREA term — `27.88·MP` becomes
/// `2 + 25.88·MP` — which is strictly LESS conservative above 1024², growing as `2·(MP−1)`: about
/// 2.5 GiB at 1152×2048 and 6 GiB at 2048², on a path that sc-11924 already records as
/// under-predicting. That is the opposite of what this story was for.
///
/// Carrying `os_reserve_gb` alongside the load-time total lets the request planner recover a bare
/// activation fallback without guessing: generic yields 18 − 4 = 14 GiB, while Lens dense yields
/// 29.88 − 0 = 29.88 GiB. The request estimator then adds the remaining 2 GiB OS/app reserve as a
/// separate fixed term. Thus Lens dense becomes `2 + 29.88·MP`, never the unsafe
/// `2 + 27.88·MP` split that would take part of its measured activation out of the area term.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HeadroomAllowance {
    /// Total GiB the load-time gate adds on top of summed component weights.
    total_gb: f64,
    /// How much of `total_gb` is fixed OS/app reserve. Zero when the constant was measured as a bare
    /// activation transient.
    os_reserve_gb: f64,
}

/// Bytes per binary gigabyte (GiB) — matches `gpu::total_unified_memory_gb`, which divides
/// `hw.memsize` by 1024³, and the epic's measured on-disk table. Shared with the candle gate
/// (sc-12306) so a "GB" in either lane's fit message means the same thing.
use crate::fit_gate::BYTES_PER_GIB;

/// A usable unified-memory budget for the machine, in GB. Single field (no free/total split): on
/// unified memory the whole pool is the budget, and current pressure is absorbed by [`HEADROOM_GB`]
/// rather than a live "free" reading that fluctuates with the OS.
///
/// Named `Mlx*` since sc-15804: this module also imports [`gen_core::MemoryBudget`], the shared
/// memory-strategy contract's request-scoped byte budget, which was `ImageMemoryBudget` until the
/// contract dropped its lane prefix. They are different things — this one is the static
/// whole-machine GB admission floor, that one is the live per-request byte envelope — and the
/// contract type owns the bare name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MlxMemoryBudget {
    pub total_gb: f64,
}

#[derive(Clone, Debug)]
struct ResolvedTextEncoderSource {
    path: PathBuf,
    direct_shard_bytes: u64,
}

/// The residency-selection outcome (sc-10839) — the pure decision, split from the [`LoadSpec`]/IO so
/// the whole three-way selection is deterministically unit-testable without the live probe or the
/// `MLX_MEMORY_CAP_ENV` global. See [`decide_residency`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ResidencyOutcome {
    /// Fits resident (or no signal) — load with everything co-resident (the warm cross-job path).
    Resident,
    /// Won't fit resident but the provider stages components and the staged peak fits — load with
    /// [`OffloadPolicy::Sequential`].
    Sequential,
    /// Won't fit even staged (or the provider can't stage) — reject. `staged_gb` is `Some` when the
    /// staged path was attempted and still overflows (so the message can name it).
    Reject {
        needed_gb: f64,
        available_gb: f64,
        staged_gb: Option<f64>,
    },
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "mlx_tier_admission_tests.rs"]
mod tier_admission_tests;

/// sc-24806: installed-artifact check of the tier chooser's declared-Sequential release. Runs the
/// production policy step and both pre-load admissions without loading tensors.
#[cfg(all(test, target_os = "macos"))]
mod declared_sequential_release_artifact_tests;
