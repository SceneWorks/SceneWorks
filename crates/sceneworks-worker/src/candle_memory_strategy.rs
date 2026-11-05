//! Candle image-provider adoption of the shared request-scoped memory selector.
//!
//! Static capability declarations never authorize an optimized request at a MEASURED grade. Exact
//! authoritative records from the packaged evidence bundle are the only measured optimized
//! candidates. Since sc-18097 (epic 18093 R1b, the candle mirror of sc-18096) every other
//! implemented optimized rung of a CERTIFIED artifact additionally carries a synthesized
//! ESTIMATE candidate — priced per rung through the image derivation law from the cell's measured
//! memory anchor where one exists (sc-22664, epic 22657 E4), else from the raw manifest
//! `vramGbByTier`/`sequentialPeakGb` rows, never a promised unmeasured saving — graded by the
//! shared selector behind the candle estimate margin
//! (`crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD`; CUDA OOM is a recoverable `Err`, so the
//! margin is looser than MLX's). The ladder's operational reserve is charged ONCE, against the
//! selector budget (`crate::vram_gate::ladder_reserve_gb`: the measured idle baseline plus a named
//! margin), and never inside a candidate's peak. Any eligible measured candidate at the same rung
//! supersedes the estimate, so measured-current admission is byte-for-byte unchanged; an
//! UNCERTIFIED artifact gets no floors at all (the manifest rows describe the certified bytes,
//! not an imported checkpoint) and keeps its resident-estimate-only behavior.

#[cfg(test)]
use crate::memory_route_registry::SD35_REQUEST_EVIDENCE_REVISION;
pub(crate) use crate::memory_route_registry::{
    KOLORS_REQUEST_EVIDENCE_REVISION, PULID_FLUX_REQUEST_EVIDENCE_REVISION,
    SDXL_REQUEST_EVIDENCE_REVISION,
};
#[path = "candle_memory_strategy/receipts.rs"]
mod receipts;
pub(crate) use receipts::*;

#[path = "candle_memory_strategy/estimates.rs"]
mod estimates;
pub(crate) use estimates::*;

#[path = "candle_memory_strategy/evidence.rs"]
mod evidence;
pub(crate) use evidence::*;

#[path = "candle_memory_strategy/selection.rs"]
mod selection;
pub(crate) use selection::*;

use gen_core::{
    GenerationMemory, LoadSpec, MemoryBackend, MemoryCacheState, MemoryConformanceState,
    MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryGeometry, MemoryMode,
    MemoryNumericTier, MemoryParityContract, MemoryParityResult, MemoryRunContext, MemorySelection,
    MemoryStrategy, Precision, Quant,
};
use sceneworks_core::memory_anchor::sdxl_cuda_memory_endpoints::SdxlCudaMemoryEndpoints;
use sceneworks_core::memory_calibration::{
    Backend as CalibrationBackend, CalibrationBinding, EvidenceQuery, EvidenceVerdict,
    Geometry as CalibrationGeometry, LoadShapeKey, QualityResult, RequiredNullable, StrategyRung,
};
use serde_json::{Map as JsonObject, Value};
use sha2::Digest;

use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};
use crate::vram_gate::VramBudget;
use crate::{WorkerError, WorkerResult};

const BYTES_PER_GIB: f64 = crate::fit_gate::BYTES_PER_GIB;
const CHROMA_ADAPTER_OVERLAY_PREFIX: &str = "chroma.adapters.ordered-additive.sha256:";
const IDEOGRAM_ADAPTER_OVERLAY_PREFIX: &str = "ideogram.adapters.ordered-additive.sha256:";
const IDEOGRAM_TURBO_ADAPTER_PREFIX: &str = "ideogram.turbo-time.sha256:";
const IDEOGRAM_PID_PREFIX: &str = "ideogram.pid.flux2.sha256:";
const IDEOGRAM_PHYSICAL_RECEIPT_PREFIX: &str = "ideogram.physical.sha256:";
const SANA_PHYSICAL_RECEIPT_PREFIX: &str = "sana.candle.dense.physical.sha256:";
const SD35_PHYSICAL_RECEIPT_PREFIX: &str = "sd3.5.candle.physical.sha256:";
const SD35_ADAPTER_RECEIPT_PREFIX: &str = "sd3.5.adapters.ordered-additive.sha256:";

const KOLORS_PHYSICAL_RECEIPT_PREFIX: &str = "kolors.physical.sha256:";
const KOLORS_ADAPTER_RECEIPT_PREFIX: &str = "kolors.adapters.ordered.sha256:";
const KOLORS_IP_RECEIPT_PREFIX: &str = "kolors.ip.sha256:";
const KOLORS_CONTROL_RECEIPT_PREFIX: &str = "kolors.control.sha256:";
const KOLORS_PID_RECEIPT_PREFIX: &str = "kolors.pid.sdxl.sha256:";
const SDXL_OVERLAY_RECEIPT_DOMAIN: &str = "sdxl-candle-overlay-assembly-v1";
const SDXL_OVERLAY_RECEIPT_PREFIX: &str = "sdxl.overlay.ordered.sha256:";

/// The output pixel count the manifest's candle rows were MEASURED at: `candle.vramMeasuredPixels`,
/// or the rows' documented 1024x1024 when the block states none (every shipped candle block
/// states it; the fallback is the geometry the rows have always been read as).
pub(crate) const MANIFEST_ROW_DEFAULT_MEASURED_PIXELS: u64 = 1_048_576;

pub(crate) struct CandleMemoryEvaluation {
    pub memory: Option<GenerationMemory>,
    pub context: MemoryRunContext,
    pub predicted_peak_gb: f64,
    /// A literal endpoint's observed whole-run peak. Admission uncertainty and reserve are
    /// reported separately; an intermediate derived configuration carries no measured claim.
    pub measured_endpoint_peak_bytes: Option<u64>,
    /// The exact lower-peak staged candidate for a receipt-priced provider. A warm cache entry may
    /// already have been loaded under `Sequential` even when this request's least-cost cold
    /// selection is Resident. In that case the cache policy requires the tighter loaded shape; this
    /// pre-admitted sibling lets the caller preserve that shape without inventing a post-load rung.
    pub warm_staged: Option<CandleWarmStagedEvaluation>,
    /// What priced the selected candidate (sc-22664, E7): a measured record, an anchor
    /// derivation, or a floor.
    pub basis: crate::memory_strategy::CandidateBasis,
    /// The selected rung's three derived phase peaks (before the runtime overlay charge) when the
    /// law priced it; `None` for a measured record, the resident live estimate, or an unscaled
    /// row. `predicted_peak_gb` is their max plus the overlay — the number the selector graded.
    pub phase_peaks: Option<sceneworks_core::memory_anchor::AnchorDerivedPhases>,
    /// The selector's own admission figures, so telemetry agrees with the selector by
    /// construction rather than by re-derivation.
    pub admitted: AdmittedBudget,
    /// What the admission decided about the provider's prefix KV cache (epic 25899): `None` for
    /// every provider that prices none. A `Cached` plan is already written onto [`Self::memory`]
    /// and counted in `context.predicted_peak_bytes` / [`Self::predicted_peak_gb`].
    pub prefix_kv_cache: Option<crate::execution_planner::PrefixKvCachePlan>,
}

/// The admission figures of one selected candle strategy (sc-22664).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AdmittedBudget {
    /// The admitted peak (raw peak plus the policy allowance), GiB.
    pub needed_gb: f64,
    /// The selector's effective budget: free minus the reserve, GiB.
    pub available_gb: f64,
    /// The operational reserve charged once against the budget (`vram_gate::ladder_reserve_gb`).
    pub reserve_gb: f64,
}

pub(crate) struct CandleWarmStagedEvaluation {
    pub memory: GenerationMemory,
    pub context: MemoryRunContext,
}

struct RequestModeBinding {
    mode: MemoryMode,
    /// Catalog/matrix axis used to find the entry-specific calibration binding.
    calibration_key: String,
    /// Typed gen-core key used by the exact request scope (`MemoryMode::as_key`).
    scope_key: String,
}

/// One synthesized estimate candidate of the candle ladder (sc-22664): the rung, its evidence
/// row, the basis that priced it, and — where the law priced it — the three derived phase peaks
/// the admission telemetry reports (E7).
#[derive(Clone, Debug)]
pub(crate) struct EstimateCandidate {
    pub selection: MemorySelection,
    pub evidence: MemoryEvidence,
    pub basis: crate::memory_strategy::CandidateBasis,
    /// The derived conditioning / denoise / decode peaks BEFORE the runtime overlay charge, when
    /// the candidate was priced through `MemoryAnchor::derive_phase_peaks` (from a measured anchor
    /// or from the manifest row as a pseudo-anchor). `None` for an unscaled row and for the
    /// resident clamp.
    pub phase_peaks: Option<sceneworks_core::memory_anchor::AnchorDerivedPhases>,
}

/// The anchor store and architecture facts the candle ladder prices from (sc-22664).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CandleLadderAnchors<'a> {
    pub store: Option<&'a sceneworks_core::memory_anchor::MemoryAnchorStore>,
    /// The facts the law scales its residues by. From the contract in production
    /// ([`architecture_facts_from_contract`], the one worker-edge seam shared with the MLX and
    /// video lanes, which translates the contract's own `architecture_facts` block axis by axis
    /// since sc-22667); a fixture may state the model's facts directly.
    ///
    /// [`architecture_facts_from_contract`]: crate::video_admission::architecture_facts_from_contract
    pub facts: sceneworks_core::memory_anchor::ArchitectureFacts,
}

#[cfg(test)]
mod tests;
