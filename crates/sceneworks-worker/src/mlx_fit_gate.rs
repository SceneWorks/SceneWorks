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
//! Generalizes the per-model `flux2_dev_edit_memory_guard` (`image_jobs/flux2.rs`): that one gates a
//! single activation-bound edit path; this gates the base weight-fit for every MLX image model.
//!
//! The pure decision logic is cross-platform and unit-tested on every lane; only the live
//! `sysctl hw.memsize` probe is macOS-only (it returns `None` elsewhere, so the gate no-ops).

use std::path::Path;
use std::sync::OnceLock;

use gen_core::{
    GenerationMemory, LoadSpec, MemoryBackendRealization, MemoryBudget, MemoryCacheState,
    MemoryConformanceState, MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey,
    MemoryEvidenceVerdict, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryParityContract,
    MemoryParityResult, MemoryProviderContract, MemoryRunContext, MemorySelection, MemoryStrategy,
    OffloadPolicy, TransformerComponent, WeightsSource,
};
use sceneworks_core::memory_calibration::{
    Backend as CalibrationBackend, BundleLoad, CalibrationBinding, EvidenceBundle, EvidenceQuery,
    EvidenceVerdict, Geometry as CalibrationGeometry, StaleEvidenceReason, StrategyRung,
};
use serde_json::{Map as JsonObject, Value};

use crate::fit_gate::resolve_offload;
pub(crate) use crate::fit_gate::FitDecision;
use crate::model_jobs::ResolvedArtifactProvenance;
use crate::{WorkerError, WorkerResult};

const REQUEST_EVIDENCE_REVISION: &str = "sc-15507-request-scope-v1";
const INFERENCE_CONTRACT_REVISION: &str = "1c4354b4b22d7f2cf5c4ea5fe17a83ab6c655e82";
const MAGE_CALIBRATION_FINGERPRINT: &str = "mage-flow-generation-peak-v1";

/// Load-invariant inputs used to estimate each request without putting geometry or strategy in the
/// generator cache key.
#[derive(Clone, Debug)]
pub(crate) struct MlxRequestPlan {
    engine_id: &'static str,
    model_id: String,
    tier: MemoryNumericTier,
    asset_bytes: u64,
    activation_headroom_bytes: u64,
    /// The resolution-INDEPENDENT slice of `activation_headroom_bytes` (sc-16195). Always
    /// `<= activation_headroom_bytes`, so the area term below can never go negative.
    fixed_reserve_bytes: u64,
    calibration: MlxCalibrationConfig,
}

impl MlxRequestPlan {
    pub(crate) fn for_spec_and_manifest(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
    ) -> Self {
        let (asset_bytes, _, headroom) = spec_component_bytes(engine_id, spec);
        let spec_tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
        };
        let (tier, calibration) = match manifest {
            Some(manifest) => match MlxCalibrationBinding::from_manifest(manifest) {
                Ok(Some(bindings)) => match resolved_artifact {
                    Some(resolved) => match resolved.fixed_artifact_tier.as_deref() {
                        Some(fixed_tier) => {
                            match numeric_tier_for_resolved(fixed_tier, spec_tier) {
                                Ok(tier) => (
                                    tier,
                                    MlxCalibrationConfig::Valid(MlxCalibrationSet {
                                        bindings,
                                        resolved,
                                    }),
                                ),
                                Err(reason) => (spec_tier, MlxCalibrationConfig::Invalid(reason)),
                            }
                        }
                        None => (
                            spec_tier,
                            MlxCalibrationConfig::Valid(MlxCalibrationSet { bindings, resolved }),
                        ),
                    },
                    None => (
                        spec_tier,
                        MlxCalibrationConfig::Invalid(
                            "the resolver supplied no immutable artifact provenance".to_owned(),
                        ),
                    ),
                },
                Ok(None) => (spec_tier, MlxCalibrationConfig::Absent),
                Err(reason) => (spec_tier, MlxCalibrationConfig::Invalid(reason)),
            },
            None => (spec_tier, MlxCalibrationConfig::Absent),
        };
        Self {
            engine_id,
            model_id: model_id.to_owned(),
            tier,
            asset_bytes,
            // The historical generic headroom includes the legacy 2 GiB unified reserve. Request
            // budgeting carries that reserve separately on Decision 2 fallback paths, so leave only
            // the activation allowance here. Exact evidence supplies its own measured envelope.
            activation_headroom_bytes: gib_to_bytes(
                (headroom.total_gb - crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB).max(0.0),
            ),
            // Whatever remains of this family's OS/app reserve once budgeting has taken the legacy
            // unified reserve out separately. ZERO for an allowance measured as a bare transient —
            // see [`HeadroomAllowance`] for why holding a reserve out of one of those would make the
            // estimator less conservative, not more.
            fixed_reserve_bytes: gib_to_bytes(
                (headroom.os_reserve_gb - crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB)
                    .max(0.0),
            ),
            calibration,
        }
    }

    /// Request peak for the generic/legacy path: resident weights + a FIXED OS/app reserve + the
    /// activation transient scaled by output area (sc-16195).
    ///
    /// The predecessor scaled the WHOLE `activation_headroom_bytes` by megapixels. That constant is
    /// `HEADROOM_GB` (a 1024²-only calibration) minus the legacy reserve, and its own doc comment
    /// decomposes it as *transient + a ~4 GiB macOS/app reserve*. Scaling the reserve was simply
    /// wrong: the OS's working set does not grow because this request renders 2048² instead of 1024².
    ///
    /// The AREA term, by contrast, is scaled exactly as before — because the sc-16195 sweep
    /// (`crate::resolution_sweep`, 7 tiers across 5 families × 5 cells, `docs/sc-16195/`) found the
    /// measured transient to be PROPORTIONAL to megapixels above 1024². Each tier's transient
    /// normalised by its OWN 1024² value, against the proportional target:
    ///
    /// | tier                     | 1024² GiB | ×1.50 area | ×2.25 area | ×4.00 area |
    /// |--------------------------|----------:|-----------:|-----------:|-----------:|
    /// | illustrious q8 (SDXL)    |     14.04 |      1.497 |      2.245 |      3.922 |
    /// | sdxl bf16 (SDXL, dense)  |     14.04 |      1.497 |      2.245 |      3.922 |
    /// | z_image_turbo q4 (DiT)   |     14.04 |      1.497 |      2.244 |      3.921 |
    /// | lens q4 (DiT)            |     14.04 |      1.497 |      2.245 |      3.922 |
    /// | qwen_image q8 (tiled VAE)|      7.66 |      1.498 |      2.246 |      3.994 |
    /// | krea_2_turbo q8 (tiled)  |      7.67 |      1.498 |      2.245 |      3.992 |
    /// | krea_2_turbo bf16 (tiled)|      7.67 |      1.498 |      2.245 |      3.992 |
    /// | **proportional target**  |         — |      1.500 |      2.250 |      4.000 |
    ///
    /// Maximum deviation from proportional across every cell above the anchor: **1.97%**.
    ///
    /// This matters more as a REFUTATION than as a confirmation. The story that filed this work
    /// reasoned from the only prior evidence — two sc-5567 points suggesting the transient was
    /// markedly SUBLINEAR (16× area → 3.8× memory, i.e. an exponent near 0.48) — and expected the fix
    /// to be a fitted sub-linear curve. Fitting that exponent would have predicted illustrious q8 at
    /// 2048² as 27.3 GiB against a measured 55.06 — a ~28 GiB UNDER-prediction on a gate whose
    /// permissive-side failure mode is an OS Jetsam SIGKILL. The area term is left linear because it
    /// measured linear, not because it was not examined.
    ///
    /// What the sweep did NOT settle is the per-family LEVEL: the 1024² column above splits 14.04 vs
    /// 7.66 on whether the family's VAE decode is tiled (sc-11747), and every family is currently
    /// charged 14. That is the larger error, and it is tracked as sc-16209 — reducing a family's
    /// anchor lowers a SIGKILL-guarding margin, which is a different kind of change from this one.
    ///
    /// The `.max(1.0)` floor on the scale is likewise kept: below 1024² the measured transient stops
    /// falling off proportionally (illustrious 0.305× and qwen 0.512× of their anchors at 0.25×
    /// area, both ABOVE the 0.25× a proportional term would predict), so the floor is the
    /// conservative reading of the data rather than a leftover.
    ///
    /// `batch` continues to multiply the area term only — a genuine batched pass renders more
    /// pixels, but it does not run more copies of macOS. See [`request_batch`]: on this lane it is
    /// always 1, because a multi-image job is a sequential loop (sc-16194).
    ///
    /// Families whose allowance carries no OS reserve (see [`HeadroomAllowance`]) hold
    /// `fixed_reserve_bytes == 0` and are therefore left EXACTLY as they were — the whole allowance
    /// stays in the area term.
    fn generic_total_peak_bytes(&self, geometry: MemoryGeometry) -> u64 {
        let megapixel_scale =
            (f64::from(geometry.width) * f64::from(geometry.height) / (1024.0 * 1024.0)).max(1.0);
        let request_scale = megapixel_scale * f64::from(geometry.batch.max(1));
        let fixed_reserve_bytes = self.fixed_reserve_bytes.min(self.activation_headroom_bytes);
        let area_bytes = self.activation_headroom_bytes - fixed_reserve_bytes;
        self.asset_bytes
            .saturating_add(fixed_reserve_bytes)
            .saturating_add(
                (area_bytes as f64 * request_scale)
                    .round()
                    .clamp(0.0, u64::MAX as f64) as u64,
            )
    }
}

#[derive(Clone, Debug)]
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
struct MlxCalibrationSet {
    bindings: Vec<MlxCalibrationBinding>,
    resolved: ResolvedArtifactProvenance,
}

#[derive(Clone, Debug)]
enum MlxCalibrationConfig {
    Absent,
    Valid(MlxCalibrationSet),
    Invalid(String),
}

impl MlxCalibrationBinding {
    /// Read the optional closed collection of exact-cell MLX bindings. A model may carry many tiers
    /// and many request cells per tier; request-time routing selects exactly one by resolved source
    /// plus mode/overlay/geometry. Duplicate selectors are rejected here as ambiguous opt-in.
    fn from_manifest(manifest: &JsonObject<String, Value>) -> Result<Option<Vec<Self>>, String> {
        let Some(calibrations) = manifest.get("mlx").and_then(|mlx| mlx.get("calibrations")) else {
            return Ok(None);
        };
        let calibrations = calibrations
            .as_array()
            .filter(|items| !items.is_empty())
            .ok_or_else(|| "mlx.calibrations must be a non-empty array".to_owned())?;
        let mut bindings = Vec::with_capacity(calibrations.len());
        for (index, calibration) in calibrations.iter().enumerate() {
            bindings.push(Self::parse(calibration, index)?);
        }
        for left in 0..bindings.len() {
            for right in (left + 1)..bindings.len() {
                if bindings[left].same_selector(&bindings[right]) {
                    return Err(format!(
                        "mlx.calibrations[{left}] and mlx.calibrations[{right}] are ambiguous duplicates"
                    ));
                }
            }
        }
        Ok(Some(bindings))
    }

    fn parse(calibration: &Value, index: usize) -> Result<Self, String> {
        const CALIBRATION_FIELDS: [&str; 16] = [
            "abi",
            "fingerprint",
            "sceneWorksRevision",
            "matrixSourceRevision",
            "inferenceRevision",
            "provider",
            "tier",
            "mode",
            "overlay",
            "geometry",
            "artifactRepository",
            "artifactResolvedRevision",
            "artifactVariant",
            "resolvedPathFingerprint",
            "rung",
            "parameters",
        ];
        let calibration = calibration
            .as_object()
            .ok_or_else(|| format!("mlx.calibrations[{index}] must be an object"))?;
        if let Some(field) = calibration
            .keys()
            .find(|field| !CALIBRATION_FIELDS.contains(&field.as_str()))
        {
            return Err(format!(
                "mlx.calibrations[{index}] contains unknown field {field:?}"
            ));
        }
        let text = |name| {
            calibration
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    format!("mlx.calibrations[{index}].{name} must be a non-empty string")
                })
        };
        let rung = match text("rung")?.as_str() {
            "resident" => StrategyRung::Resident,
            "staged_residency" => StrategyRung::StagedResidency,
            "bounded_decode" => StrategyRung::BoundedDecode,
            "bounded_attention" => StrategyRung::BoundedAttention,
            "bounded_transformer_residency" => StrategyRung::BoundedTransformerResidency,
            other => {
                return Err(format!(
                    "unsupported mlx.calibrations[{index}].rung {other:?}"
                ))
            }
        };
        let parameters = calibration
            .get("parameters")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| format!("mlx.calibrations[{index}].parameters must be an object"))?;
        let selection_parameters = parse_evidence_parameters(rung, &parameters)?;
        let geometry = calibration
            .get("geometry")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("mlx.calibrations[{index}].geometry must be an object"))?;
        const GEOMETRY_FIELDS: [&str; 4] = ["width", "height", "batch", "frames"];
        if let Some(field) = geometry
            .keys()
            .find(|field| !GEOMETRY_FIELDS.contains(&field.as_str()))
        {
            return Err(format!(
                "mlx.calibrations[{index}].geometry contains unknown field {field:?}"
            ));
        }
        let geometry_value = |name| {
            geometry
                .get(name)
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    format!("mlx.calibrations[{index}].geometry.{name} must be a positive u32")
                })
        };
        Ok(Self {
            query: CalibrationBinding {
                abi: calibration
                    .get("abi")
                    .and_then(Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| format!("mlx.calibrations[{index}].abi must be a u32"))?,
                fingerprint: text("fingerprint")?,
                scene_works_revision: text("sceneWorksRevision")?,
                matrix_source_revision: text("matrixSourceRevision")?,
                inference_revision: text("inferenceRevision")?,
                artifact_repository: text("artifactRepository")?,
                artifact_resolved_revision: text("artifactResolvedRevision")?,
                artifact_variant: text("artifactVariant")?,
                resolved_path_fingerprint: text("resolvedPathFingerprint")?,
            },
            provider: text("provider")?,
            tier: text("tier")?,
            mode: text("mode")?,
            overlay: text("overlay")?,
            geometry: CalibrationGeometry {
                width: geometry_value("width")?,
                height: geometry_value("height")?,
                batch: geometry_value("batch")?,
                frames: geometry_value("frames")?,
            },
            rung,
            parameters,
            selection_parameters,
        })
    }

    fn same_selector(&self, other: &Self) -> bool {
        self.query == other.query
            && self.provider == other.provider
            && self.tier == other.tier
            && self.mode == other.mode
            && self.overlay == other.overlay
            && self.geometry == other.geometry
            && self.rung == other.rung
            && self.parameters == other.parameters
    }
}

fn numeric_tier_for_resolved(
    tier: &str,
    fallback: MemoryNumericTier,
) -> Result<MemoryNumericTier, String> {
    let resolved = match tier {
        "q4" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
        },
        "q8" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q8),
        },
        "nvfp4" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Nvfp4),
        },
        "bf16" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: None,
        },
        "fp32" => MemoryNumericTier {
            precision: gen_core::Precision::Fp32,
            quant: None,
        },
        other => return Err(format!("unsupported resolver-supplied MLX tier {other:?}")),
    };
    if fallback.quant.is_some() && fallback != resolved {
        return Err(format!(
            "resolver tier {tier:?} conflicts with the LoadSpec numeric tier"
        ));
    }
    Ok(resolved)
}

fn plan_tier_key(tier: MemoryNumericTier) -> &'static str {
    match tier.quant {
        Some(gen_core::Quant::Q4) => "q4",
        Some(gen_core::Quant::Q8) => "q8",
        Some(gen_core::Quant::Nvfp4) => "nvfp4",
        None if tier.precision == gen_core::Precision::Fp32 => "fp32",
        None => "bf16",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmissionPath {
    Evidence,
    Legacy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyAdmissionReason {
    PackagedEmpty,
    NoBinding,
    NoRecord,
    OutOfEnvelope,
    StaleFingerprint,
    StaleIdentity,
    StaleBundle,
}

#[derive(Clone, Debug)]
struct VerifiedAdmissionCandidate {
    evidence: MemoryEvidence,
    foreign_reserve_bytes: u64,
    required_host_bytes: u64,
    record_id: String,
}

#[derive(Clone, Debug)]
struct VerifiedGeometryAlternative {
    geometry: CalibrationGeometry,
    calibration_abi: u32,
    calibration_fingerprint: String,
}

#[derive(Clone, Debug)]
struct AdmissionRoute {
    path: AdmissionPath,
    fallback_reason: Option<LegacyAdmissionReason>,
    evidence: Vec<VerifiedAdmissionCandidate>,
    evidence_revision: Option<String>,
    process_limit_bytes: Option<u64>,
    lower_alternative: Option<VerifiedGeometryAlternative>,
}

fn budget_for_admission(mut budget: MemoryBudget, admission: &AdmissionRoute) -> MemoryBudget {
    if let Some(process_limit_bytes) = admission.process_limit_bytes {
        budget.reserved_headroom_bytes = budget.total_bytes.saturating_sub(process_limit_bytes);
    }
    budget
}

/// Exact request axes that must be reconsidered for both cold and warm cache runs.
#[derive(Clone, Debug)]
pub(crate) struct MlxRequestInputs {
    pub width: u32,
    pub height: u32,
    /// Original job image count. This is a SCHEDULING quantity, not a memory one — see
    /// [`request_batch`] for why it must never reach a geometry's `batch`. Kept here so the
    /// over-budget message can quote the request the operator actually submitted.
    pub count: u32,
    pub mode: String,
    pub overlay: Option<String>,
    pub adapter_count: usize,
    pub has_reference: bool,
    pub use_pid: bool,
    pub has_phases: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct MlxRequestEvaluation {
    pub memory: GenerationMemory,
    pub context: MemoryRunContext,
    /// Request-scoped soft MLX ceiling derived from an exact verified cell. Legacy cells keep the
    /// process-global #1947 fallback. Applying this never changes the wired limit.
    pub process_limit_bytes: Option<u64>,
}

fn gib_to_bytes(gib: f64) -> u64 {
    (gib * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64
}

fn decimal_gb_to_bytes(gb: f64) -> u64 {
    (gb * 1_000_000_000.0).round().clamp(0.0, u64::MAX as f64) as u64
}

pub(crate) fn add_post_load_external_delta(baseline: u64, before: u64, after: u64) -> u64 {
    baseline.saturating_add(after.saturating_sub(before))
}

/// The batch dimension of ONE provider forward pass, which on this lane is always 1 — NOT the job's
/// image count.
///
/// A multi-image job is a SEQUENTIAL loop, not a batched pass: `image_jobs::base` expands the job
/// count into one seed per image and hands the vector to `image_jobs::stream::drive_gen_items`, which
/// calls the provider once per seed with a `GenerationRequest { count: 1, .. }` and releases MLX's
/// retained-buffer cache between items (`RequestCacheRelease`, a `Drop` guard, so cancel and error
/// exits release too — sc-5567). Peak unified memory is therefore a MAX over items, not a sum: the
/// resident weights plus ONE image's transient working set, whatever the count.
///
/// This function exists because charging `count` as a batch dimension is not a small over-estimate,
/// it is unbounded. Both consumers of `geometry.batch` multiply by it — the generic estimator's
/// `request_scale` and Mage's `generation_peak_gb` — so a 4-image 1152x2048 krea_2_turbo request was
/// quoted 33.22 GiB of weights + 16 GiB x 2.25 MP x 4 = 177.22 GiB against a 126.00 GiB budget. The
/// activation term ALONE (144 GiB) exceeded the whole budget, so that cell rejected every model at
/// every tier on a 128 GiB Mac before a single weight byte was counted.
///
/// If a provider ever renders a genuine batched pass, the fix is to thread THAT pass's batch size
/// here — not to reinstate the job count, which cannot describe serialized work.
fn request_batch(_inputs: &MlxRequestInputs) -> u32 {
    1
}

/// The provider-facing geometry of one forward pass. See [`request_batch`] for the `batch` rule.
fn request_geometry(inputs: &MlxRequestInputs) -> MemoryGeometry {
    MemoryGeometry {
        width: inputs.width,
        height: inputs.height,
        batch: request_batch(inputs),
        frames: 1,
    }
}

fn request_mode(mode: &str) -> (MemoryMode, &'static str) {
    match mode {
        "image_generation" | "text_to_image" => (MemoryMode::TextToImage, "text_to_image"),
        "character_image" | "image_to_image" => (MemoryMode::ImageToImage, "image_to_image"),
        "edit_image" => (MemoryMode::Edit, "edit"),
        _ => (MemoryMode::Other(mode.to_owned()), "other"),
    }
}

/// Translate a selection into the engine's per-rung engagement knobs.
///
/// SC-15805: this asks the contract which rungs the selection ENGAGES rather than re-deriving the
/// answer from `MemoryStrategy`'s numeric order. The order is a cost ordering, and the cumulative
/// default it expresses is defeasible — a rung the provider does not implement is not engaged, so a
/// provider publishing a verified cheaper composition no longer has its unimplemented rung's knob
/// switched on underneath it.
fn memory_for_selection(
    contract: &MemoryProviderContract,
    selection: MemorySelection,
) -> GenerationMemory {
    GenerationMemory {
        stage_residency: contract.engages(selection.strategy, MemoryStrategy::StagedResidency),
        tile_vae_decode: contract.engages(selection.strategy, MemoryStrategy::BoundedDecode),
        chunk_attention: contract.engages(selection.strategy, MemoryStrategy::BoundedAttention),
        stream_transformer_blocks: contract.engages(
            selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ),
        ..Default::default()
    }
}

fn resident_evidence(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    mode: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    predicted_peak_bytes: u64,
    calibration_fingerprint: Option<&str>,
) -> (MemorySelection, MemoryEvidence) {
    let selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier,
    };
    let evidence = MemoryEvidence {
        key: MemoryEvidenceKey {
            resolved_route: contract.provider_id.clone(),
            backend: "mlx".to_owned(),
            tier,
            mode: mode.to_owned(),
            overlay: overlay.map(str::to_owned),
            geometry,
            strategy: selection.strategy,
            engaged_composition: contract.engaged_composition(selection.strategy),
            parameters: selection.parameters,
        },
        conformance: MemoryConformanceState::ImplementedUnverified,
        dimensions: MemoryEvidenceDimensions {
            static_implementation: MemoryEvidenceVerdict::Satisfied,
            declared_calibration: MemoryEvidenceVerdict::Missing,
            historical_verification: MemoryEvidenceVerdict::Missing,
            current_environment_verification: MemoryEvidenceVerdict::Missing,
            canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
            exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
        },
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: calibration_fingerprint.unwrap_or_default().to_owned(),
        sceneworks_revision: REQUEST_EVIDENCE_REVISION.to_owned(),
        inference_revision: INFERENCE_CONTRACT_REVISION.to_owned(),
        harness_version: String::new(),
        predicted_peak_bytes,
        observed_peak_bytes: None,
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::NotRun,
    };
    (selection, evidence)
}

fn stale_fallback_reason(reason: StaleEvidenceReason) -> LegacyAdmissionReason {
    if reason == StaleEvidenceReason::CalibrationFingerprint {
        LegacyAdmissionReason::StaleFingerprint
    } else {
        LegacyAdmissionReason::StaleIdentity
    }
}

fn stronger_fallback_reason(
    current: LegacyAdmissionReason,
    candidate: LegacyAdmissionReason,
) -> LegacyAdmissionReason {
    let priority = |reason| match reason {
        LegacyAdmissionReason::NoRecord => 0,
        LegacyAdmissionReason::OutOfEnvelope => 1,
        LegacyAdmissionReason::StaleFingerprint => 2,
        LegacyAdmissionReason::StaleIdentity => 3,
        LegacyAdmissionReason::PackagedEmpty
        | LegacyAdmissionReason::NoBinding
        | LegacyAdmissionReason::StaleBundle => 4,
    };
    if priority(candidate) > priority(current) {
        candidate
    } else {
        current
    }
}

fn evidence_strategy(rung: StrategyRung) -> MemoryStrategy {
    match rung {
        StrategyRung::Resident => MemoryStrategy::Resident,
        StrategyRung::StagedResidency => MemoryStrategy::StagedResidency,
        StrategyRung::BoundedDecode => MemoryStrategy::BoundedDecode,
        StrategyRung::BoundedAttention => MemoryStrategy::BoundedAttention,
        StrategyRung::BoundedTransformerResidency => MemoryStrategy::BoundedTransformerResidency,
    }
}

fn parse_evidence_parameters(
    rung: StrategyRung,
    parameters: &JsonObject<String, Value>,
) -> Result<gen_core::MemoryStrategyParameters, String> {
    const KEYS: [&str; 5] = [
        "decodeTileEdge",
        "decodeOverlap",
        "attentionChunkSize",
        "transformerWindowSize",
        "transformerWindowComponent",
    ];
    if let Some(key) = parameters.keys().find(|key| !KEYS.contains(&key.as_str())) {
        return Err(format!("unknown MLX strategy parameter {key:?}"));
    }
    let integer = |key: &str, minimum: u32| -> Result<u32, String> {
        parameters
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value >= minimum)
            .ok_or_else(|| format!("{key} must be an integer >= {minimum}"))
    };
    let expected_numeric: &[(&str, u32)] = match rung {
        StrategyRung::Resident | StrategyRung::StagedResidency => &[],
        StrategyRung::BoundedDecode => &[("decodeTileEdge", 1), ("decodeOverlap", 0)],
        StrategyRung::BoundedAttention => &[
            ("decodeTileEdge", 1),
            ("decodeOverlap", 0),
            ("attentionChunkSize", 1),
        ],
        StrategyRung::BoundedTransformerResidency => &[
            ("decodeTileEdge", 1),
            ("decodeOverlap", 0),
            ("attentionChunkSize", 1),
            ("transformerWindowSize", 1),
        ],
    };
    for (key, _) in expected_numeric {
        if !parameters.contains_key(*key) {
            return Err(format!("{rung:?} requires {key}"));
        }
    }
    for key in KEYS[..4].iter().filter(|key| {
        !expected_numeric
            .iter()
            .any(|(expected, _)| expected == *key)
    }) {
        if parameters.contains_key(*key) {
            return Err(format!("{rung:?} forbids {key}"));
        }
    }
    let transformer_window_component = match parameters.get("transformerWindowComponent") {
        None => None,
        Some(Value::String(value)) if rung == StrategyRung::BoundedTransformerResidency => {
            Some(match value.as_str() {
                "dit" => TransformerComponent::Dit,
                "text_encoder" => TransformerComponent::TextEncoder,
                "both" => TransformerComponent::Both,
                other => return Err(format!("unsupported transformerWindowComponent {other:?}")),
            })
        }
        Some(_) if rung != StrategyRung::BoundedTransformerResidency => {
            return Err(format!("{rung:?} forbids transformerWindowComponent"))
        }
        Some(_) => {
            return Err("transformerWindowComponent must be dit, text_encoder, or both".to_owned())
        }
    };
    Ok(gen_core::MemoryStrategyParameters {
        decode_tile_edge: expected_numeric
            .iter()
            .find(|(key, _)| *key == "decodeTileEdge")
            .map(|(key, minimum)| integer(key, *minimum))
            .transpose()?,
        decode_overlap: expected_numeric
            .iter()
            .find(|(key, _)| *key == "decodeOverlap")
            .map(|(key, minimum)| integer(key, *minimum))
            .transpose()?,
        attention_chunk_size: expected_numeric
            .iter()
            .find(|(key, _)| *key == "attentionChunkSize")
            .map(|(key, minimum)| integer(key, *minimum))
            .transpose()?,
        transformer_window_size: expected_numeric
            .iter()
            .find(|(key, _)| *key == "transformerWindowSize")
            .map(|(key, minimum)| integer(key, *minimum))
            .transpose()?,
        transformer_window_component,
    })
}

/// Apply Decision 2 at the request seam: exact verified cells fail closed; every non-covering state
/// returns to the established legacy selector. The route is returned so tests and telemetry can
/// distinguish a normal empty-bundle transition from drift or an out-of-envelope request.
fn packaged_admission_route(
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
) -> WorkerResult<AdmissionRoute> {
    if let MlxCalibrationConfig::Invalid(reason) = &plan.calibration {
        return Err(WorkerError::InvalidPayload(format!(
            "{} has an invalid MLX calibration opt-in: {reason}",
            plan.model_id
        )));
    }
    let loaded = sceneworks_core::memory_calibration::load_packaged_bundle().map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "packaged memory-calibration evidence is invalid: {error}"
        ))
    })?;
    let bundle = match loaded {
        BundleLoad::Ready(bundle) => bundle,
        BundleLoad::Stale(_) => {
            return Ok(AdmissionRoute {
                path: AdmissionPath::Legacy,
                fallback_reason: Some(LegacyAdmissionReason::StaleBundle),
                evidence: Vec::new(),
                evidence_revision: None,
                process_limit_bytes: None,
                lower_alternative: None,
            });
        }
    };
    evidence_admission_route(&bundle, plan, inputs, mode_key, budget)
}

fn evidence_admission_route(
    bundle: &EvidenceBundle,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
) -> WorkerResult<AdmissionRoute> {
    if let MlxCalibrationConfig::Invalid(reason) = &plan.calibration {
        return Err(WorkerError::InvalidPayload(format!(
            "{} has an invalid MLX calibration opt-in: {reason}",
            plan.model_id
        )));
    }
    if bundle.records.is_empty() {
        return Ok(AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::PackagedEmpty),
            evidence: Vec::new(),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: None,
        });
    }
    let calibration = match &plan.calibration {
        MlxCalibrationConfig::Absent => {
            return Ok(AdmissionRoute {
                path: AdmissionPath::Legacy,
                fallback_reason: Some(LegacyAdmissionReason::NoBinding),
                evidence: Vec::new(),
                evidence_revision: None,
                process_limit_bytes: None,
                lower_alternative: None,
            })
        }
        MlxCalibrationConfig::Valid(calibration) => calibration,
        MlxCalibrationConfig::Invalid(_) => unreachable!("invalid opt-in rejected above"),
    };
    let identity_matches = calibration
        .bindings
        .iter()
        .filter(|binding| {
            binding.provider == plan.engine_id
                && binding.tier == plan_tier_key(plan.tier)
                && binding.query.artifact_repository == calibration.resolved.identity.repository
                && binding.query.artifact_resolved_revision
                    == calibration.resolved.identity.revision
                && binding.query.artifact_variant == calibration.resolved.identity.variant
                && binding.query.resolved_path_fingerprint
                    == calibration.resolved.identity.fingerprint
        })
        .collect::<Vec<_>>();
    if identity_matches.is_empty() {
        return Ok(AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::StaleIdentity),
            evidence: Vec::new(),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: None,
        });
    }
    // A measured cell is recorded per forward pass (`batch: 1`), so keying the lookup on the job's
    // image count made every count > 1 request miss its own calibration and fall to the generic
    // estimator's Legacy/OutOfEnvelope path. See `request_batch`.
    let request_cell_geometry = CalibrationGeometry {
        width: inputs.width,
        height: inputs.height,
        batch: request_batch(inputs),
        frames: 1,
    };
    let overlay = inputs.overlay.as_deref().unwrap_or("none");
    let matching = identity_matches
        .into_iter()
        .filter(|binding| {
            binding.mode == mode_key
                && binding.overlay == overlay
                && binding.geometry == request_cell_geometry
        })
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return Ok(AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::OutOfEnvelope),
            evidence: Vec::new(),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: verified_lower_alternative(
                bundle,
                calibration,
                plan,
                inputs,
                mode_key,
                budget,
            ),
        });
    }
    let mut evidence = Vec::new();
    let mut fallback_reason = LegacyAdmissionReason::NoRecord;
    for binding in matching {
        let query = EvidenceQuery {
            backend: CalibrationBackend::Mlx,
            model_id: plan.model_id.clone(),
            provider: binding.provider.clone(),
            tier: binding.tier.clone(),
            mode: mode_key.to_owned(),
            overlay: overlay.to_owned(),
            geometry: request_cell_geometry,
            rung: binding.rung,
            parameters: binding.parameters.clone(),
            calibration: binding.query.clone(),
        };
        match bundle.evidence_for(&query) {
            EvidenceVerdict::Verified(record) => {
                let envelope = record.mlx_admission_envelope().ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                    "{} has a verified MLX evidence cell without a complete MLX admission envelope",
                    plan.model_id
                ))
                })?;
                let memory_evidence = MemoryEvidence {
                    key: MemoryEvidenceKey {
                        resolved_route: plan.engine_id.to_owned(),
                        backend: "mlx".to_owned(),
                        tier: plan.tier,
                        mode: mode_key.to_owned(),
                        overlay: inputs.overlay.clone(),
                        geometry: request_geometry(inputs),
                        strategy: evidence_strategy(binding.rung),
                        engaged_composition: record
                            .strategy
                            .engaged_rungs
                            .iter()
                            .copied()
                            .map(evidence_strategy)
                            .collect(),
                        parameters: binding.selection_parameters,
                    },
                    conformance: MemoryConformanceState::Verified,
                    dimensions: MemoryEvidenceDimensions::VERIFIED,
                    calibration_abi: binding.query.abi,
                    calibration_fingerprint: binding.query.fingerprint.clone(),
                    sceneworks_revision: binding.query.scene_works_revision.clone(),
                    inference_revision: binding.query.inference_revision.clone(),
                    harness_version: record.harness_version.clone(),
                    predicted_peak_bytes: envelope.peak_bytes,
                    observed_peak_bytes: Some(envelope.observed_non_reclaimable_wired_bytes),
                    parity: MemoryParityContract::Exact,
                    parity_result: MemoryParityResult::Passed,
                };
                evidence.push(VerifiedAdmissionCandidate {
                    evidence: memory_evidence,
                    foreign_reserve_bytes: envelope.foreign_reserve_bytes,
                    required_host_bytes: envelope.required_host_bytes(),
                    record_id: record.id.clone(),
                });
            }
            EvidenceVerdict::Unknown => {}
            EvidenceVerdict::OutOfEnvelope => {
                fallback_reason =
                    stronger_fallback_reason(fallback_reason, LegacyAdmissionReason::OutOfEnvelope);
            }
            EvidenceVerdict::Stale(reason) => {
                fallback_reason =
                    stronger_fallback_reason(fallback_reason, stale_fallback_reason(reason));
            }
        }
    }
    if evidence.is_empty() {
        return Ok(AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(fallback_reason),
            evidence,
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: None,
        });
    }
    let lower_alternative =
        verified_lower_alternative(bundle, calibration, plan, inputs, mode_key, budget);
    Ok(AdmissionRoute {
        path: AdmissionPath::Evidence,
        fallback_reason: None,
        evidence,
        evidence_revision: None,
        process_limit_bytes: None,
        lower_alternative,
    })
}

/// Select the largest strictly lower, same-aspect geometry backed by a current exact record that
/// fits the live host boundary. This is the only source for a named refusal alternative: no formula,
/// interpolation, tier heuristic, or aspect-ratio rewrite is admitted.
fn verified_lower_alternative(
    bundle: &EvidenceBundle,
    calibration: &MlxCalibrationSet,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
) -> Option<VerifiedGeometryAlternative> {
    let overlay = inputs.overlay.as_deref().unwrap_or("none");
    let requested_width = u64::from(inputs.width);
    let requested_height = u64::from(inputs.height);
    calibration
        .bindings
        .iter()
        .filter(|binding| {
            binding.provider == plan.engine_id
                && binding.tier == plan_tier_key(plan.tier)
                && binding.mode == mode_key
                && binding.overlay == overlay
                && binding.query.artifact_repository == calibration.resolved.identity.repository
                && binding.query.artifact_resolved_revision
                    == calibration.resolved.identity.revision
                && binding.query.artifact_variant == calibration.resolved.identity.variant
                && binding.query.resolved_path_fingerprint
                    == calibration.resolved.identity.fingerprint
                && binding.geometry.batch == inputs.count.max(1)
                && binding.geometry.frames == 1
                && binding.geometry.width <= inputs.width
                && binding.geometry.height <= inputs.height
                && (binding.geometry.width < inputs.width
                    || binding.geometry.height < inputs.height)
                && u64::from(binding.geometry.width) * requested_height
                    == u64::from(binding.geometry.height) * requested_width
        })
        .filter_map(|binding| {
            let query = EvidenceQuery {
                backend: CalibrationBackend::Mlx,
                model_id: plan.model_id.clone(),
                provider: binding.provider.clone(),
                tier: binding.tier.clone(),
                mode: binding.mode.clone(),
                overlay: binding.overlay.clone(),
                geometry: binding.geometry,
                rung: binding.rung,
                parameters: binding.parameters.clone(),
                calibration: binding.query.clone(),
            };
            let EvidenceVerdict::Verified(record) = bundle.evidence_for(&query) else {
                return None;
            };
            let envelope = record.mlx_admission_envelope()?;
            let effective = budget.effective_bytes();
            (envelope.required_host_bytes() <= budget.total_bytes
                && envelope.peak_bytes <= effective)
                .then_some(VerifiedGeometryAlternative {
                    geometry: binding.geometry,
                    calibration_abi: binding.query.abi,
                    calibration_fingerprint: binding.query.fingerprint.clone(),
                })
        })
        .max_by_key(|alternative| {
            let geometry = alternative.geometry;
            (
                u64::from(geometry.width) * u64::from(geometry.height),
                geometry.width,
                geometry.height,
            )
        })
}

#[cfg(test)]
fn verified_lower_geometry(
    bundle: &EvidenceBundle,
    calibration: &MlxCalibrationSet,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
) -> Option<CalibrationGeometry> {
    verified_lower_alternative(bundle, calibration, plan, inputs, mode_key, budget)
        .map(|alternative| alternative.geometry)
}

/// Pure request selector used by production and unit/hardware seams. Additional provider evidence is
/// accepted explicitly; absent exact verified cells, only the resident baseline can be admitted.
#[allow(clippy::too_many_arguments)]
fn evaluate_request_with_budget(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
) -> WorkerResult<MlxRequestEvaluation> {
    evaluate_request_with_budget_using_bundle(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        additional_evidence,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_request_with_budget_using_bundle(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
    evidence_bundle: Option<&EvidenceBundle>,
) -> WorkerResult<MlxRequestEvaluation> {
    use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};

    let geometry = request_geometry(inputs);
    let (mode, mode_key) = request_mode(&inputs.mode);
    if plan.engine_id.starts_with("mage_flow") && inputs.adapter_count > 0 {
        return Err(WorkerError::InvalidPayload(format!(
            "{} request includes {} adapter(s), but Mage's paired memory calibration does not \
             include LoRA/LoKr tensors; refusing an unbounded MLX request",
            plan.engine_id, inputs.adapter_count
        )));
    }
    let mut fallback_contract;
    let contract = if let Some(contract) = generator.memory_strategy_contract() {
        contract
    } else {
        fallback_contract = MemoryProviderContract::compatibility_default(
            plan.engine_id,
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: false,
                cache_eviction: true,
            },
        );
        fallback_contract.asset_facts.base_bytes = plan.asset_bytes;
        &fallback_contract
    };
    let calibration_fingerprint = contract
        .calibration
        .as_ref()
        .map(|identity| identity.fingerprint.as_str());
    let calibration_abi = contract
        .calibration
        .as_ref()
        .map_or(0, |identity| identity.abi);
    let mut admission = match evidence_bundle {
        Some(bundle) => evidence_admission_route(bundle, plan, inputs, mode_key, budget)?,
        None => packaged_admission_route(plan, inputs, mode_key, budget)?,
    };
    let carries_verified_claim =
        admission.path == AdmissionPath::Evidence || admission.lower_alternative.is_some();
    if carries_verified_claim
        && !contract.calibration.as_ref().is_some_and(|identity| {
            admission.evidence.iter().all(|candidate| {
                identity.abi == candidate.evidence.calibration_abi
                    && identity.fingerprint == candidate.evidence.calibration_fingerprint
            }) && admission
                .lower_alternative
                .as_ref()
                .is_none_or(|alternative| {
                    identity.abi == alternative.calibration_abi
                        && identity.fingerprint == alternative.calibration_fingerprint
                })
        })
    {
        admission = AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::StaleIdentity),
            evidence: Vec::new(),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: None,
        };
    }
    // The selector and provider request context must use the same evidence-derived foreign demand
    // as the fail-closed precheck and the request-scoped MLX ceiling. Otherwise a covered request
    // with existing committed memory could be admitted against the legacy 2 GiB reserve even though
    // its record captured a larger non-process share.
    let mut budget = budget_for_admission(budget, &admission);
    tracing::info!(
        event = "mlx_memory_admission_path",
        route = plan.engine_id,
        model = plan.model_id,
        path = ?admission.path,
        fallback_reason = ?admission.fallback_reason,
        evidence_revision = admission.evidence_revision.as_deref().unwrap_or("none"),
        width = inputs.width,
        height = inputs.height,
        count = inputs.count.max(1),
        "selected MLX memory-admission path"
    );
    if plan.engine_id.starts_with("mage_flow")
        && !matches!(
            contract.calibration.as_ref(),
            Some(identity)
                if identity.abi == gen_core::MEMORY_CALIBRATION_ABI
                    && identity.fingerprint == MAGE_CALIBRATION_FINGERPRINT
        )
    {
        return Err(WorkerError::InvalidPayload(format!(
            "{} loaded provider calibration does not match ABI {} / fingerprint {}; refusing \
             request admission against an unpaired estimator",
            plan.engine_id,
            gen_core::MEMORY_CALIBRATION_ABI,
            MAGE_CALIBRATION_FINGERPRINT
        )));
    }
    // The estimator is a complete-pipeline peak, while the live budget is incremental from the
    // process's current state. Only bytes above the cache-recorded pre-load external baseline, and
    // no more than the provider-declared resident envelope, may be credited as already present.
    // Unrelated process allocations therefore remain charged on the available side.
    let attributable_resident_bytes = budget
        .committed_bytes
        .saturating_sub(external_committed_bytes)
        .min(contract.asset_facts.base_bytes);
    let evidence_peak_bytes = admission
        .evidence
        .iter()
        .map(|candidate| candidate.evidence.predicted_peak_bytes)
        .max();
    let modeled_peak_bytes = evidence_peak_bytes.unwrap_or(total_peak_bytes);
    if attributable_resident_bytes > modeled_peak_bytes {
        return Err(WorkerError::InvalidPayload(format!(
            "{} live resident attribution {} exceeds modeled total peak {}; refusing an \
             inconsistent MLX budget",
            plan.engine_id, attributable_resident_bytes, modeled_peak_bytes
        )));
    }
    let predicted_peak_bytes = if admission.path == AdmissionPath::Evidence {
        // Exact evidence describes the whole request peak. On a warm cache, remove only this
        // provider's already-resident assets from committed bytes so the full peak is charged once;
        // unrelated allocations remain committed. Do not rewrite the evidence record's peak.
        budget.committed_bytes = budget
            .committed_bytes
            .saturating_sub(attributable_resident_bytes);
        modeled_peak_bytes
    } else {
        total_peak_bytes - attributable_resident_bytes
    };
    let (resident_selection, resident) = resident_evidence(
        contract,
        plan.tier,
        mode_key,
        inputs.overlay.as_deref(),
        geometry,
        predicted_peak_bytes,
        calibration_fingerprint,
    );
    let mut selections = Vec::new();
    let mut evidence = Vec::new();
    if admission.path == AdmissionPath::Evidence {
        // A covered cell is authorized only by its exact verified ladder. Letting the generic
        // resident estimate or caller-supplied evidence run first would turn Evidence telemetry
        // into a legacy bypass.
        for candidate in &admission.evidence {
            let exact = &candidate.evidence;
            if attributable_resident_bytes > exact.predicted_peak_bytes {
                continue;
            }
            let candidate_budget = Budget {
                available_gb: budget.total_bytes.saturating_sub(budget.committed_bytes) as f64
                    / BYTES_PER_GIB,
                reclaimable_gb: budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
                total_gb: budget.total_bytes as f64 / BYTES_PER_GIB,
                reserved_headroom_gb: candidate.foreign_reserve_bytes as f64 / BYTES_PER_GIB,
            };
            if candidate_budget.effective_gb().is_some_and(|available| {
                exact.predicted_peak_bytes as f64 / BYTES_PER_GIB <= available
            }) {
                selections.push(MemorySelection {
                    strategy: exact.key.strategy,
                    parameters: exact.key.parameters,
                    tier: exact.key.tier,
                });
                evidence.push(exact);
            }
        }
        if evidence.is_empty() {
            let minimum_required_host = admission
                .evidence
                .iter()
                .map(|candidate| candidate.required_host_bytes)
                .min()
                .unwrap_or(0);
            let alternative = admission
                .lower_alternative
                .as_ref()
                .map(|alternative| {
                    format!(
                        "; current verified alternative: {}x{}",
                        alternative.geometry.width, alternative.geometry.height
                    )
                })
                .unwrap_or_default();
            return Err(WorkerError::InvalidPayload(format!(
                "{} request {}x{} count {} needs at least {:.2} GiB at its smallest verified \
                 MLX host boundary, but no exact candidate fits the live unified-memory budget\
                 {alternative}",
                plan.model_id,
                inputs.width,
                inputs.height,
                inputs.count.max(1),
                minimum_required_host as f64 / BYTES_PER_GIB,
            )));
        }
    } else {
        selections.reserve(1 + additional_evidence.len());
        evidence.reserve(1 + additional_evidence.len());
        selections.push(resident_selection);
        evidence.push(&resident);
        selections.extend(additional_evidence.iter().map(|item| MemorySelection {
            strategy: item.key.strategy,
            parameters: item.key.parameters,
            tier: item.key.tier,
        }));
        evidence.extend(additional_evidence);
    }
    let candidates = selections
        .iter()
        .zip(evidence)
        .map(|(selection, evidence)| Candidate {
            selection: *selection,
            evidence,
        })
        .collect::<Vec<_>>();
    let expected_inference_revision = admission
        .evidence
        .first()
        .map_or(INFERENCE_CONTRACT_REVISION, |candidate| {
            candidate.evidence.inference_revision.as_str()
        });
    let selection = crate::memory_strategy::select_strategy(
        RequestScope {
            resolved_route: plan.engine_id,
            backend: "mlx",
            tier: plan.tier,
            mode: mode_key,
            overlay: inputs.overlay.as_deref(),
            geometry,
            expected_inference_revision,
        },
        contract,
        Some(Budget {
            available_gb: budget.total_bytes.saturating_sub(budget.committed_bytes) as f64
                / BYTES_PER_GIB,
            reclaimable_gb: budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
            total_gb: budget.total_bytes as f64 / BYTES_PER_GIB,
            reserved_headroom_gb: if admission.path == AdmissionPath::Evidence {
                0.0
            } else {
                budget.reserved_headroom_bytes as f64 / BYTES_PER_GIB
            },
        }),
        &candidates,
    );
    let (selection, mut needed_gb, mut available_gb) = match selection {
        Selection::Selected {
            selection,
            needed_gb,
            available_gb,
        } => (selection, needed_gb, available_gb),
        Selection::Reject {
            needed_gb,
            available_gb,
        } => {
            let alternative = admission
                .lower_alternative
                .as_ref()
                .map(|alternative| {
                    format!(
                        "; current verified alternative: {}x{}",
                        alternative.geometry.width, alternative.geometry.height
                    )
                })
                .unwrap_or_default();
            return Err(WorkerError::InvalidPayload(format!(
                "{} request {}x{} count {} needs {:.2} GiB but only {:.2} GiB is safely available{}",
                plan.engine_id,
                inputs.width,
                inputs.height,
                inputs.count.max(1),
                needed_gb,
                available_gb,
                alternative,
            )));
        }
        Selection::Unverified { reason } => {
            let alternative = admission
                .lower_alternative
                .as_ref()
                .map(|alternative| {
                    format!(
                        "; current verified alternative: {}x{}",
                        alternative.geometry.width, alternative.geometry.height
                    )
                })
                .unwrap_or_default();
            let message = format!(
                "{} request {}x{} count {} has no safely verified MLX memory strategy ({reason:?}); \
                 refusing to enter MLX's process-terminating allocation path{alternative}",
                plan.engine_id,
                inputs.width,
                inputs.height,
                inputs.count.max(1),
            );
            return Err(WorkerError::InvalidPayload(message));
        }
    };
    let mut selected_record_id = None;
    let mut process_limit_bytes = admission.process_limit_bytes;
    let predicted_peak_bytes = if admission.path == AdmissionPath::Evidence {
        let index = admission
            .evidence
            .iter()
            .position(|candidate| {
                let evidence = &candidate.evidence;
                evidence.key.strategy == selection.strategy
                    && evidence.key.parameters == selection.parameters
                    && evidence.key.tier == selection.tier
            })
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{} selected a strategy without matching verified MLX evidence",
                    plan.engine_id
                ))
            })?;
        let candidate = &admission.evidence[index];
        let evidence = &candidate.evidence;
        let reserve = candidate.foreign_reserve_bytes;
        let selected_budget = Budget {
            available_gb: budget.total_bytes.saturating_sub(budget.committed_bytes) as f64
                / BYTES_PER_GIB,
            reclaimable_gb: budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
            total_gb: budget.total_bytes as f64 / BYTES_PER_GIB,
            reserved_headroom_gb: reserve as f64 / BYTES_PER_GIB,
        };
        needed_gb = evidence.predicted_peak_bytes.saturating_add(reserve) as f64 / BYTES_PER_GIB;
        available_gb = selected_budget.effective_gb().unwrap_or(0.0);
        budget.reserved_headroom_bytes = reserve;
        process_limit_bytes = Some(budget.total_bytes.saturating_sub(reserve));
        selected_record_id = Some(candidate.record_id.clone());
        evidence.predicted_peak_bytes
    } else {
        predicted_peak_bytes
    };
    tracing::info!(
        event = "memory_strategy_request_selected",
        route = plan.engine_id,
        backend = "mlx",
        tier = ?plan.tier,
        mode = mode_key,
        overlay = inputs.overlay.as_deref().unwrap_or("none"),
        width = inputs.width,
        height = inputs.height,
        count = inputs.count.max(1),
        cache_state = ?cache_state,
        load_policy = ?load_policy,
        strategy = ?selection.strategy,
        cache_eviction = contract.engages(
            selection.strategy,
            MemoryStrategy::StagedResidency,
        ),
        parameters = ?selection.parameters,
        evidence_record_id = selected_record_id.as_deref().unwrap_or("none"),
        predicted_peak_bytes,
        effective_budget_bytes = budget.effective_bytes(),
        needed_gb,
        available_gb,
        "selected request-scoped MLX memory strategy"
    );
    Ok(MlxRequestEvaluation {
        memory: memory_for_selection(contract, selection),
        process_limit_bytes,
        context: MemoryRunContext {
            selection,
            calibration_abi,
            calibration_fingerprint: calibration_fingerprint.unwrap_or_default().to_owned(),
            mode,
            has_reference: inputs.has_reference,
            use_pid: inputs.use_pid,
            has_phases: inputs.has_phases,
            geometry,
            overlay: inputs.overlay.clone(),
            budget,
            predicted_peak_bytes,
            cache_state,
            evidence_revision: selected_record_id
                .or(admission.evidence_revision)
                .unwrap_or_else(|| REQUEST_EVIDENCE_REVISION.to_owned()),
        },
    })
}

#[cfg(target_os = "macos")]
fn live_request_budget(engine_id: &str) -> WorkerResult<MemoryBudget> {
    // Reclaim only allocator-cache buffers from prior geometries; live arrays (the cached weights)
    // remain committed. This makes A→B→A independent of B's freed scratch without reloading A.
    mlx_rs::memory::clear_cache();
    let committed_bytes = mlx_rs::memory::get_active_memory() as u64;
    let reclaimable_bytes = mlx_rs::memory::get_cache_memory() as u64;
    let (total_bytes, reserved_headroom_bytes) = if engine_id.starts_with("mage_flow") {
        let safe_gb = runtime_macos::providers::mage::memory::production_safe_budget_gb()
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        (decimal_gb_to_bytes(safe_gb), 0)
    } else {
        let total_gib = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "MLX unified-memory budget is unavailable; refusing an unbounded request"
                        .to_owned(),
                )
            })?
            .total_gb;
        let reserve = crate::fit_gate::legacy_unified_reserve(total_gib);
        (gib_to_bytes(total_gib), gib_to_bytes(reserve.gb))
    };
    Ok(MemoryBudget {
        total_bytes,
        committed_bytes,
        reclaimable_bytes,
        reserved_headroom_bytes,
    })
}

#[cfg(target_os = "macos")]
fn request_total_peak_bytes(plan: &MlxRequestPlan, geometry: MemoryGeometry) -> u64 {
    if plan.engine_id.starts_with("mage_flow") {
        decimal_gb_to_bytes(runtime_macos::providers::mage::memory::generation_peak_gb(
            plan.tier.quant,
            geometry.width,
            geometry.height,
            geometry.batch,
        ))
    } else {
        plan.generic_total_peak_bytes(geometry)
    }
}

/// Evaluate one real MLX request after cache lookup and immediately before generation.
#[cfg(target_os = "macos")]
pub(crate) fn evaluate_request(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    external_committed_bytes: u64,
) -> WorkerResult<MlxRequestEvaluation> {
    let geometry = request_geometry(inputs);
    let budget = live_request_budget(plan.engine_id)?;
    let total_peak_bytes = request_total_peak_bytes(plan, geometry);
    evaluate_request_with_budget(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        &[],
    )
}

/// Whether `engine_id`'s provider drops components in phase order under [`OffloadPolicy::Sequential`]
/// — derived at query time from the engine's REGISTERED descriptor
/// [`Capabilities`](gen_core::Capabilities)`::supports_sequential_offload` bit, not a hand-maintained
/// allowlist (sc-10840, epic 10834).
///
/// Why the descriptor bit is the right source of truth: [`OffloadPolicy::Sequential`] is *advisory* —
/// a provider that has NOT wired the load→use→drop residency lifecycle silently treats it as
/// `Resident` (never an error), so predicting "it fits staged" and then holding everything resident
/// would SIGKILL. `supports_sequential_offload` is precisely the provider's own machine-readable
/// attestation that it wired that lifecycle (the gen-core discovery signal, sc-11126). Reading it
/// per-engine makes the gate self-maintaining: every family the mlx-gen Phase-1 fan-out wires
/// (sc-10840 — sd3/sana/flux/flux2/chroma/ideogram/kolors/anima/boogu/bernini alongside the earlier
/// sdxl/z-image/qwen/lens/krea families) is covered the moment its descriptor advertises the bit, with
/// no lockstep edit here. An engine that does not separate a text encoder (e.g. sensenova's fused MoT,
/// `footprint` te=0) leaves the bit `false` and is correctly never offered `Sequential` — a no-op that
/// would OOM.
///
/// This is a pre-load, weights-free registry lookup (`(descriptor)()` allocates no tensors), the same
/// query shape the worker already uses for family/guidance/quant capability advertisement and the
/// analogous `ProviderRegistry::footprint` size seam (sc-10894). An id with no registered generator — or a
/// registered one that does not advertise the bit — yields `false` (the safe default: never select a
/// residency policy the provider won't honor). Sees exactly the providers the selected runtime bundle
/// carries: MLX providers are explicitly anchored on macOS, while the CUDA bundle exposes its explicit
/// Candle catalog. The same query is shared by the MLX fit gate (sc-10840) and Candle fit gate
/// (sc-12130), so adding a truthful provider capability needs no worker allowlist edit.
pub(crate) fn engine_supports_sequential(engine_id: &str) -> bool {
    crate::inference_runtime::media()
        .generators()
        .find(|reg| (reg.descriptor)().id == engine_id)
        .is_some_and(|reg| (reg.descriptor)().capabilities.supports_sequential_offload)
}

/// Emulate a smaller Mac: force the total-unified-memory budget (GB). Set e.g.
/// `SCENEWORKS_MLX_MEMORY_CAP_GB=16` to make the gate treat this machine as a 16 GB Mac, so a model
/// that would OOM there is rejected (and, once Phase 1 lands, run under sequential residency) exactly
/// as on real small hardware. Unset / non-positive ⇒ use the real `sysctl hw.memsize` total.
pub(crate) const MLX_MEMORY_CAP_ENV: &str = "SCENEWORKS_MLX_MEMORY_CAP_GB";

/// Headroom (GiB) added on top of the summed on-disk component weights to cover the MLX Metal
/// activation transient during denoise/decode plus the OS + other apps drawing from the same unified
/// pool (the gate budgets against TOTAL physical RAM, so the OS share must come out of this headroom).
///
/// CALIBRATED (sc-10863) from real `get_peak_memory` footprints measured through
/// `footprint_measure.rs` (one tier per process; peak = load + one 1024² generation, RESIDENT
/// hold-all path, no memory cap). Measured `transient = peak − resident` and `headroom = peak − disk`:
///
/// | model            | disk GiB | resident | peak  | transient | headroom(peak−disk) |
/// |------------------|---------:|---------:|------:|----------:|--------------------:|
/// | illustrious q8   |     5.01 |     4.74 | 18.78 |     14.04 |               13.77 |
/// | lens q4          |    17.67 |    16.46 | 30.50 |     14.04 |               12.83 |
/// | qwen-image q8    |    35.90 |    33.45 | 41.11 |      7.66 |                5.20 |
/// | lens-turbo bf16  |    28.43 |    45.67 | 75.55 |     29.88 |               47.12 |
///
/// FINDING — the transient is NOT a function of on-disk weight bytes: qwen-image q8 has the LARGEST
/// weights (35.9 GiB) but the SMALLEST transient (7.66 — its VAE decode is tiled, sc-11747), while
/// illustrious q8 has the smallest weights but a 14 GiB transient. It is architecture- and
/// resolution-bound (dominated by the VAE decode + attention at the output resolution), so a
/// disk-SCALED predictor (`Σweights · k`) would over-reject the large-but-efficient models and
/// under-predict the small ones — the wrong shape. And the load-time gate cannot see the request
/// resolution (the generator is cached across resolutions), so a per-request `f(resolution)` term is
/// not threadable at this seam. A conservative CONSTANT is therefore the right structure.
///
/// 18 GiB = the max COMMON-CASE transient at 1024² (14.04, illustrious q8 & lens q4 — the three
/// resident≈disk tiers; lens-turbo's larger 29.88 transient is a separate architecture outlier, below)
/// plus a ~4 GiB macOS/app reserve. This replaces the provisional 10.0, which UNDER-predicted 3 of the
/// 4 measured tiers (illustrious 15.0<18.8, lens 27.7<30.5, lens-turbo 38.4<75.6) — i.e. was a latent
/// SIGKILL risk on Macs sized between the predicted and the real peak. All three resident≈disk tiers
/// are now covered with margin without over-rejecting a model that fits (illustrious q8: 5.01+18=23.0
/// still fits a 24 GiB Mac, where its real 18.8 GiB peak + OS does too).
///
/// NOT covered by this constant (surfaced sc-10863, tracked follow-ups — see the story): (1) the
/// lens-turbo bf16 OUTLIER, whose 47.12 GiB headroom (peak 75.55 − disk 28.43) is NOT one effect but
/// TWO that must be modeled together. It DECOMPOSES as (a) 17.24 GiB IN-MEMORY WEIGHT EXPANSION
/// (resident 45.67 − disk 28.43) — its mxfp4-on-disk gpt-oss text encoder expands loading to bf16
/// (45.67 = 1.61× disk 28.43), so `sum_safetensors_bytes` under-counts the in-memory weights — PLUS
/// (b) a 29.88 GiB ACTIVATION TRANSIENT (peak 75.55 − resident 45.67), which is architecture-bound (the
/// large gpt-oss encoder's activations) and ~2× the ~14 GiB the other three tiers show at the same
/// 1024². HEADROOM=18 covers the common-case transient (~14) + ~4 GiB OS/app reserve, but UNDER-predicts
/// this class by ~29 GiB even AFTER a weight-byte correction — because the 29.88 transient ALONE exceeds
/// 18 (75.55 − (28.43 + 18) = 29.12; the old provisional 10 under-predicted it by ~37: 75.55 −
/// (28.43 + 10) = 37.12). So correcting only the weight bytes is INSUFFICIENT: both the in-memory weight
/// size AND the outsized transient must be modeled for these tiers. (A blanket bf16 expansion factor
/// also can't fix the weight half — a bf16 tier whose encoder is bf16-on-disk would then be
/// over-rejected ~1.6× — so that fix needs per-family in-memory weight sizing plus a per-architecture
/// transient term, backed by bf16 measurements across models.) Tracked in sc-11924. (2) Output
/// RESOLUTION > 1024² grows the VAE-decode transient past 14 GiB — all four points are 1024², so 18 is
/// a 1024²-worst-case; a higher-res campaign is a follow-up.
const HEADROOM_GB: f64 = 18.0;
/// Lens dense/bf16's measured 1024² activation transient. Its gpt-oss encoder is the only current
/// MLX family whose architecture-bound transient exceeds the generic calibration (sc-11924).
const LENS_DENSE_HEADROOM_GB: f64 = 29.88;

/// The macOS/app share inside [`HEADROOM_GB`] — the part of that flat allowance that covers the OS and
/// other apps drawing from the same unified pool, as opposed to this request's activation transient.
///
/// This is the sc-10863 decomposition read back out and given a name: `HEADROOM_GB` 18 = the max
/// common-case measured 1024² transient (14.04, illustrious q8 & lens q4) + this ~4 GiB reserve. It
/// was previously only prose in that constant's doc comment, which is precisely how it came to be
/// multiplied by megapixels — the request estimator could not tell the two halves apart.
///
/// It applies to [`HEADROOM_GB`] ONLY. See [`HeadroomAllowance`] for why that distinction is load
/// bearing rather than pedantic.
const OS_APP_RESERVE_GB: f64 = 4.0;

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
/// Carrying `os_reserve_gb` alongside the total makes the lens-dense path an exact no-op (reserve 0 ⇒
/// the whole allowance stays in the area term) and keeps the generic path's 2 + 14 split, so the
/// change is provably non-regressive for every family rather than only for the measured ones.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HeadroomAllowance {
    /// Total GiB the load-time gate adds on top of summed component weights.
    total_gb: f64,
    /// How much of `total_gb` is fixed OS/app reserve. Zero when the constant was measured as a bare
    /// activation transient.
    os_reserve_gb: f64,
}

impl HeadroomAllowance {
    /// The sc-10863 calibration: a 1024² transient plus a macOS/app reserve.
    const GENERIC: Self = Self {
        total_gb: HEADROOM_GB,
        os_reserve_gb: OS_APP_RESERVE_GB,
    };
    /// sc-11924's lens-dense measurement: a bare activation transient, no reserve folded in.
    const LENS_DENSE: Self = Self {
        total_gb: LENS_DENSE_HEADROOM_GB,
        os_reserve_gb: 0.0,
    };
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

/// Read the small-Mac cap from the environment. `Some(gb)` only for a positive number.
pub(crate) fn mlx_memory_cap_gb() -> Option<f64> {
    parse_memory_cap(std::env::var(MLX_MEMORY_CAP_ENV).ok().as_deref())
}

/// Parse the cap value: a positive, finite float (GB), else `None`.
pub(crate) fn parse_memory_cap(raw: Option<&str>) -> Option<f64> {
    let value = raw?.trim().parse::<f64>().ok()?;
    (value.is_finite() && value > 0.0).then_some(value)
}

/// Resolve the budget: the emulation cap overrides the real probed total (emulating a smaller Mac);
/// otherwise the real total. `None` from both ⇒ no budget ⇒ the gate no-ops (`Unknown`).
pub(crate) fn resolve_budget(
    real_total_gb: Option<f64>,
    cap: Option<f64>,
) -> Option<MlxMemoryBudget> {
    cap.or(real_total_gb)
        .map(|total_gb| MlxMemoryBudget { total_gb })
}

/// Predicted whole-model peak (GiB) = summed component weight bytes + [`HEADROOM_GB`]. `None` when
/// `weight_bytes == 0` (nothing measured ⇒ no signal ⇒ never block).
#[cfg(test)]
pub(crate) fn predicted_peak_gb(weight_bytes: u64) -> Option<f64> {
    predicted_peak_gb_with_headroom(weight_bytes, HEADROOM_GB)
}

fn predicted_peak_gb_with_headroom(weight_bytes: u64, headroom_gb: f64) -> Option<f64> {
    (weight_bytes > 0).then(|| weight_bytes as f64 / BYTES_PER_GIB + headroom_gb)
}

/// Decide whether the predicted peak fits the budget. Missing either input ⇒ `Unknown` (never
/// block), exactly like the flux2 guard and the candle gate.
pub(crate) fn fit_decision(needed_gb: Option<f64>, budget: Option<MlxMemoryBudget>) -> FitDecision {
    let (Some(needed_gb), Some(budget)) = (needed_gb, budget) else {
        return FitDecision::Unknown;
    };
    if budget.total_gb + f64::EPSILON < needed_gb {
        FitDecision::TooBig {
            needed_gb,
            available_gb: budget.total_gb,
        }
    } else {
        FitDecision::Fits
    }
}

/// Predicted SEQUENTIAL peak (GiB) = the largest single working set + [`HEADROOM_GB`] (sc-10839). The
/// `Sequential` schedule drops the text encoder(s) before the DiT loads and keeps the tiny VAE
/// co-resident with the DiT, so the peak is `max(text-encoders, everything-else) + headroom` rather
/// than the resident sum. `everything-else = total − text_encoders` (the DiT + VAE + any control/IP).
/// `None` when nothing was measured (`total == 0`). When the text encoders are unmeasured
/// (`te_bytes == 0`) this equals the resident peak — no claimed saving, so the second-stage overflow
/// check then rejects exactly as the resident gate would (the safe fallback).
#[cfg(test)]
pub(crate) fn predicted_sequential_peak_gb(total_bytes: u64, te_bytes: u64) -> Option<f64> {
    predicted_sequential_peak_gb_with_headroom(total_bytes, te_bytes, HEADROOM_GB)
}

fn predicted_sequential_peak_gb_with_headroom(
    total_bytes: u64,
    te_bytes: u64,
    headroom_gb: f64,
) -> Option<f64> {
    if total_bytes == 0 {
        return None;
    }
    let rest_bytes = total_bytes.saturating_sub(te_bytes);
    let staged_max = te_bytes.max(rest_bytes);
    Some(staged_max as f64 / BYTES_PER_GIB + headroom_gb)
}

/// Second-stage gate for a [`FitDecision::Offload`] (sc-10839): sequential residency was selected
/// because the RESIDENT peak won't fit, on the promise that the staged working set will. If the
/// predicted staged peak ([`predicted_sequential_peak_gb`]) STILL exceeds the budget, return
/// `Some(needed_gb)` so the caller rejects before load with an actionable message instead of a
/// reactive Metal-OOM / SIGKILL. `None` (staged fits, or no budget) keeps the sequential run. Unlike
/// the candle gate — where the sequential peak is only sometimes measured — the MLX staged peak is
/// always derivable from the on-disk component split, so this check always applies.
pub(crate) fn sequential_overflow_gb(
    sequential_needed_gb: Option<f64>,
    budget: Option<MlxMemoryBudget>,
) -> Option<f64> {
    let (needed_gb, budget) = (sequential_needed_gb?, budget?);
    (budget.total_gb + f64::EPSILON < needed_gb).then_some(needed_gb)
}

/// Sum the on-disk bytes of every `.safetensors` weight file under `dir` (recursively), following
/// symlinks (the HF cache stores each shard as a symlink into `blobs/`). AppleDouble `._*` sidecars
/// are skipped — they masquerade as `.safetensors` and would double-count (and corrupt globs, per
/// the AppleDouble sidecar gotcha). Returns 0 if the directory is missing or holds no weights, which
/// the gate treats as "no signal".
pub(crate) fn sum_safetensors_bytes(dir: &Path) -> u64 {
    fn walk(
        dir: &Path,
        visited_dirs: &mut std::collections::HashSet<std::path::PathBuf>,
        total: &mut u64,
    ) {
        // `metadata` below intentionally follows file symlinks because Hugging Face
        // snapshots link shards into `blobs/`. Directory links are different: an
        // operator-provided model root may contain a link/junction cycle. Canonical
        // directory identity makes that traversal finite and also avoids double-counting
        // an aliased subtree.
        let Ok(canonical_dir) = std::fs::canonicalize(dir) else {
            return;
        };
        if !visited_dirs.insert(canonical_dir.clone()) {
            return;
        }
        let Ok(entries) = std::fs::read_dir(canonical_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // `metadata()` follows symlinks (HF blobs); `file_type()` on the DirEntry does not, so
            // resolve the target kind via `metadata` for symlinked shard files.
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                walk(&path, visited_dirs, total);
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".safetensors") && !name.starts_with("._") {
                *total += meta.len();
            }
        }
    }
    let mut total = 0;
    walk(dir, &mut std::collections::HashSet::new(), &mut total);
    total
}

/// On-disk `.safetensors` bytes of a [`WeightsSource`]: the recursive sum for a `Dir`, the file length
/// for a single-file `File`. Used to fold a separate control/overlay checkpoint ([`LoadSpec::control`])
/// into the fit total — its weights are not under the base `spec.weights` tree.
fn weights_source_bytes(src: &WeightsSource) -> u64 {
    match src {
        WeightsSource::Dir(dir) => sum_safetensors_bytes(dir),
        WeightsSource::File(file) => std::fs::metadata(file).map_or(0, |meta| meta.len()),
    }
}

/// Resolve the TEXT-ENCODER on-disk bytes for the staged split (sc-10894), preferring the provider-owned
/// per-component footprint over the `text_encoder*` subdir scan.
///
/// The subdir scan ([`sum_text_encoder_bytes`]) only recognizes the *diffusers* `text_encoder*` naming;
/// it returns **zero** for a family whose encoder lives elsewhere — boogu's `mllm/`, bernini's flat
/// `t5_encoder.safetensors`, anima's `text_encoders/` under a `split_files/` root — or that has no
/// separable encoder at all (sensenova's flat unified MoT). A zero text-encoder collapses the staged
/// (`max(te, rest)`) peak back to the resident peak, so no `Sequential` saving is ever selected. The
/// provider's `ProviderRegistry::footprint` computes the split from the exact subdirs *its own* loader resolves,
/// so it is authoritative per family. `footprint_te` is `Some` when the provider declared a footprint,
/// `None` otherwise (or the query errored) — in which case this falls back to the subdir scan, the
/// historical behavior. The whole-model `total` stays the recursive [`sum_safetensors_bytes`] sum, so
/// `rest = total − te` accounts for the DiT + VAE + anything else regardless of the footprint's own
/// dit/vae split (and keeps the sc-11006 control-branch folding intact).
pub(crate) fn resolve_text_encoder_bytes(footprint_te: Option<u64>, dir: &Path) -> u64 {
    footprint_te.unwrap_or_else(|| sum_text_encoder_bytes(dir))
}

/// Sum the on-disk `.safetensors` bytes of the model's TEXT-ENCODER component(s) — the phase-A
/// component the `Sequential` residency drops before the DiT loads (sc-10839). Matches the diffusers
/// snapshot's top-level `text_encoder` / `text_encoder_2` / `text_encoder_*` subdirs (SDXL has both
/// CLIP encoders; Z-Image the single Qwen encoder), reusing [`sum_safetensors_bytes`] per subdir so
/// the HF-cache symlink + AppleDouble handling is shared. `0` if the dir is missing or has no
/// recognizable text-encoder subdir — which makes the staged estimate fall back to the resident sum
/// (no claimed saving), the safe direction. Superseded, when a provider declares a footprint, by
/// [`resolve_text_encoder_bytes`] (sc-10894); still the fallback for providers that do not.
pub(crate) fn sum_text_encoder_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut total = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !std::fs::metadata(&path).is_ok_and(|meta| meta.is_dir()) {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "text_encoder" || name.starts_with("text_encoder_") {
            total += sum_safetensors_bytes(&path);
        }
    }
    total
}

/// Total unified memory (GiB) via a blocking `sysctl hw.memsize`, cached process-wide (physical RAM
/// never changes at runtime). The blocking sibling of `gpu::total_unified_memory_gb` — the gate runs
/// on the generator-cache thread, which is already blocking on the weight load, so a one-shot
/// subprocess probe there is free. `None` off macOS or when the probe fails ⇒ the gate no-ops (a
/// cached `None` is a deliberate fail-open, consistent with `Unknown` never blocking).
fn probe_total_unified_memory_gib() -> Option<f64> {
    probe_total_unified_memory_bytes().map(|bytes| bytes as f64 / BYTES_PER_GIB)
}

/// Total unified memory (bytes) — the raw probe [`probe_total_unified_memory_gib`] scales. Cached
/// process-wide behind one `OnceLock`, so the `sysctl` subprocess runs at most once however many
/// callers ask. Also read by [`crate::generator_cache::apply_gpu_memory_limit`] at worker startup,
/// which needs the byte figure rather than the GiB one.
pub(crate) fn probe_total_unified_memory_bytes() -> Option<u64> {
    static TOTAL_BYTES: OnceLock<Option<u64>> = OnceLock::new();
    *TOTAL_BYTES.get_or_init(sysctl_total_unified_memory_bytes)
}

#[cfg(target_os = "macos")]
fn sysctl_total_unified_memory_bytes() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

#[cfg(not(target_os = "macos"))]
fn sysctl_total_unified_memory_bytes() -> Option<u64> {
    None
}

// ---------------------------------------------------------------------------------------------
// Mochi 1: the FRAME-DEPENDENT decode fit gate (epic 1788 / sc-11992)
// ---------------------------------------------------------------------------------------------
//
// Why Mochi needs its own gate rather than riding the generic one above:
//
//  1. The generic gate is deliberately RESOLUTION-BLIND. `predicted_peak_gb` = Σweights +
//     HEADROOM_GB, where HEADROOM_GB is a 1024²-calibrated CONSTANT — the load-time seam cannot see
//     the request geometry (the generator is cached across resolutions, see the HEADROOM_GB note).
//     That structure is right for image models, whose transient is roughly request-independent once
//     calibrated. It is WRONG for Mochi: its AsymmVAE decode is UNTILED (`vae.decode(latents)`
//     materializes the whole clip — sc-12291), so the peak grows LINEARLY IN CLIP LENGTH. A 7-frame
//     and a 151-frame request differ by ~55 GiB on the same model. A constant cannot express that.
//
//  2. Mochi's `supports_sequential_offload: false` ⇒ the legacy admission override admits on WEIGHTS ALONE
//     (18.73 GiB fits almost any Mac), so the generic gate would happily admit a 151-frame job that
//     then needs ~79 GiB. The floor is correct for its own purpose (sc-12179: never wall-reject a
//     machine that used to work) but it is precisely what makes the generic gate unable to protect
//     Mochi.
//
//  3. This MUST be a pre-flight rejection, not a caught error. MLX's default error handler is
//     `exit(-1)` — a hard process kill (sc-12178/12179, GitHub #1544). An `exit(-1)` cannot be
//     mapped to a job error after the fact, so the only place to honor the epic's "actionable error,
//     not a crash" AC is BEFORE the decode allocates.
//
// The gate is therefore a per-REQUEST admission check (it sees frames + geometry), layered beside —
// not inside — the generic per-LOAD one, and it budgets on a DERIVED architectural formula rather
// than a calibration constant, because nothing has been measured on-device yet (B5/sc-11995 backfills
// `footprint.residentMemoryBytes`/`peakMemoryBytes`; sc-12291 tiles the decode and should then cut
// the frame term sharply).
//
// All three points above hold verbatim for the candle lane, which grew the same gate in sc-12306 —
// except (3), where a CUDA OOM is catchable rather than a SIGKILL, so the pre-flight buys an
// actionable message and minutes of un-wasted denoise rather than process survival. What remains here
// is MLX-SPECIFIC: the unified-memory budget probe, the typed legacy/evidence reserve, and the Mac-worded
// message. The shared ARITHMETIC lives in `crate::fit_gate` (`mochi_decode_peak_gb` /
// `mochi_needed_gb`); the candle half is `vram_gate::mochi_fit_error`. The on-disk scan
// (`mochi_resident_bytes`) is shared too — the hosted tier layout is one repo serving both lanes.

/// Predicted whole-generation peak (GiB) for an MLX Mochi request, in UNIFIED memory: the shared
/// backend-neutral arithmetic ([`crate::fit_gate::mochi_needed_gb`]) with the caller's unified reserve,
/// because on unified memory the OS draws from the same pool the model does.
///
/// The formula itself moved to [`crate::fit_gate`] when the candle video lane needed the identical
/// arithmetic (sc-12306) — every term in it is a fact about the MODEL (tensor shapes × the f32 dtype
/// both decoders pin), so duplicating it per lane would let the two gates disagree about one model.
/// The RESERVE is what stays lane-specific; see that function's note.
fn mochi_needed_gb(
    weight_bytes: u64,
    frames: u32,
    width: u32,
    height: u32,
    reserve_gb: f64,
) -> Option<f64> {
    crate::fit_gate::mochi_needed_gb(weight_bytes, frames, width, height, reserve_gb)
}

/// The pure Mochi admission decision: `Some(error)` when the predicted peak overflows the budget,
/// `None` to admit. Missing either signal (unmeasurable weights / no budget) admits — the gate never
/// blocks without evidence, exactly like [`fit_decision`].
pub(crate) fn mochi_fit_error(
    model_label: &str,
    weight_bytes: u64,
    frames: u32,
    width: u32,
    height: u32,
    budget: Option<MlxMemoryBudget>,
) -> Option<WorkerError> {
    let budget = budget?;
    let reserve = crate::fit_gate::legacy_unified_reserve(budget.total_gb);
    let needed_gb = mochi_needed_gb(weight_bytes, frames, width, height, reserve.gb)?;
    (budget.total_gb + f64::EPSILON < needed_gb).then(|| {
        mochi_too_big_error(
            model_label,
            needed_gb,
            budget.total_gb,
            frames,
            width,
            height,
            weight_bytes as f64 / BYTES_PER_GIB,
        )
    })
}

/// Build Mochi's actionable over-budget rejection. Follows the [`too_big_error`] convention — name
/// the model, explain the constraint ("unified memory"), state what it needs and what the machine
/// has — and adds the lever that is UNIQUE to Mochi: the clip length. The generic message's advice
/// ("choose a smaller quant tier, lower the resolution") is nearly useless here — Mochi has one
/// trained bucket (848×480) and the decode dwarfs the tier delta (q4→bf16 is ~11 GiB against a
/// ~60 GiB decode) — so the message leads with shortening the clip, which is the only lever that
/// moves the dominant term.
#[allow(clippy::too_many_arguments)]
fn mochi_too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    frames: u32,
    width: u32,
    height: u32,
    weights_gb: f64,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of unified memory to render a {frames}-frame \
         {width}x{height} clip (~{weights} GB of weights, held resident for the whole run, plus an \
         untiled VAE decode whose peak grows with clip length) but this machine has ~{available} \
         GB. Shorten the clip — the decode peak scales roughly linearly with duration — or run on a \
         Mac with more memory.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        weights = weights_gb.round() as i64,
    ))
}

/// Live pre-flight Mochi admission check (the seam the worker's Mochi generation arm calls before
/// loading). Resolves the same budget the generic gate uses — the real `sysctl hw.memsize` total,
/// overridable by [`MLX_MEMORY_CAP_ENV`] so a small Mac can be emulated in tests — and sums the
/// on-disk bytes the load will actually hold resident: the TIER dir (the AsymmDiT) plus the SHARED
/// `text_encoder/` + `vae/` siblings, which the provider resolves from the tier dir's PARENT
/// (`resolve_component_root`, mlx-gen-mochi/src/model.rs). Summing only the tier dir would miss the
/// ~9.7 GiB T5-XXL + VAE — over half the resident footprint.
///
/// `Ok(())` admits (including whenever there is no signal); `Err` is the actionable pre-crash
/// rejection.
pub(crate) fn mochi_fit_check(
    model_label: &str,
    tier_dir: &Path,
    frames: u32,
    width: u32,
    height: u32,
) -> WorkerResult<()> {
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    match mochi_fit_error(
        model_label,
        mochi_resident_bytes(tier_dir),
        frames,
        width,
        height,
        budget,
    ) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// The on-disk bytes a Mochi load holds resident: the tier dir (AsymmDiT) + the shared `text_encoder/`
/// and `vae/` components resolved from the tier dir's parent (the A6 shared-sibling layout). A
/// self-contained dir — the raw snapshot, where the components live under the dir itself — is summed
/// once, never double-counted, because the parent scan only adds SIBLING dirs of `tier_dir`.
///
/// **Shared by both lanes** (sc-12306), despite the module name: this describes the HOSTED REPO LAYOUT,
/// not MLX. `SceneWorks/mochi-1-mlx` serves candle too — A6's `.scales`-detect seam ingests the same
/// mlx-affine tiers 1:1 through the same `resolve_mochi_model_dir` — so the resident byte total is
/// identical off-Mac, and both providers set `supports_sequential_offload: false`, so both hold all
/// three components for the whole run. It stays here beside its `sum_safetensors_bytes` helper (which
/// has several other MLX callers) rather than moving to `fit_gate` with the arithmetic; the candle
/// video gate calls it through this path, exactly as `image_jobs/base.rs` already calls
/// `mlx_fit_gate::engine_supports_sequential` from the candle image lane (sc-12130).
pub(crate) fn mochi_resident_bytes(tier_dir: &Path) -> u64 {
    let mut total = sum_safetensors_bytes(tier_dir);
    if let Some(parent) = tier_dir.parent() {
        for component in ["text_encoder", "vae"] {
            let dir = parent.join(component);
            // Only count a shared sibling — a self-contained tier dir already summed its own
            // components above (`sum_safetensors_bytes` recurses).
            if dir.is_dir() && !tier_dir.join(component).is_dir() {
                total += sum_safetensors_bytes(&dir);
            }
        }
    }
    total
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

/// The pure residency decision: given the model's whole-model + text-encoder on-disk bytes, the
/// (possibly emulated) budget, and whether the provider stages components, choose Resident /
/// Sequential / Reject (sc-10839). No IO, no globals — the live [`apply_residency_policy`] resolves
/// those and delegates here.
#[cfg(test)]
pub(crate) fn decide_residency(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
) -> ResidencyOutcome {
    decide_residency_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        sequential_capable,
        HEADROOM_GB,
    )
}

fn decide_residency_with_headroom(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
    headroom_gb: f64,
) -> ResidencyOutcome {
    let peak = decide_residency_by_peak_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        sequential_capable,
        headroom_gb,
    );
    match peak {
        ResidencyOutcome::Reject { .. } => {
            legacy_admission_override(total_bytes, te_bytes, budget, sequential_capable)
                .unwrap_or(peak)
        }
        admitted => admitted,
    }
}

/// The PEAK-based residency decision (the pre-sc-12179 logic): compare the predicted whole-model peak
/// (`Σweights + HEADROOM_GB`) — and, for a sequential-capable provider, the staged max-component peak
/// — against the budget. This is the right signal for SELECTING Resident vs Sequential and for the
/// rejection message's `needed`/`staged` numbers, but it rejects too aggressively on small Macs
/// because the flat headroom bundles a pageable 1024² activation transient (sc-12179); the caller
/// folds in the Decision 2 legacy override before honoring a reject.
#[cfg(test)]
fn decide_residency_by_peak(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
) -> ResidencyOutcome {
    decide_residency_by_peak_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        sequential_capable,
        HEADROOM_GB,
    )
}

fn decide_residency_by_peak_with_headroom(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
    headroom_gb: f64,
) -> ResidencyOutcome {
    // Generic MLX has no provider-supplied memory-strategy contract or request-scoped evidence yet.
    // Enter the shared selector explicitly as ImplementedUnverified, then keep the established
    // cold-load gate. This adopts one selector API without manufacturing VERIFIED evidence or
    // copying optimized selection logic; request-aware providers can promote only after exposing
    // their own contract, fingerprint, and exact request evidence.
    let shared_observation = generic_mlx_shared_observation(total_bytes, budget, headroom_gb);
    debug_assert!(matches!(
        shared_observation,
        crate::memory_strategy::Selection::Selected {
            selection: gen_core::MemorySelection {
                strategy: gen_core::MemoryStrategy::Resident,
                ..
            },
            ..
        } | crate::memory_strategy::Selection::Reject { .. }
            | crate::memory_strategy::Selection::Unverified { .. }
    ));
    let resident = fit_decision(
        predicted_peak_gb_with_headroom(total_bytes, headroom_gb),
        budget,
    );
    match resolve_offload(resident, sequential_capable) {
        FitDecision::Fits | FitDecision::Unknown => ResidencyOutcome::Resident,
        FitDecision::Offload {
            needed_gb,
            available_gb,
        } => {
            let staged =
                predicted_sequential_peak_gb_with_headroom(total_bytes, te_bytes, headroom_gb);
            match sequential_overflow_gb(staged, budget) {
                Some(_) => ResidencyOutcome::Reject {
                    needed_gb,
                    available_gb,
                    staged_gb: staged,
                },
                None => ResidencyOutcome::Sequential,
            }
        }
        FitDecision::TooBig {
            needed_gb,
            available_gb,
        } => ResidencyOutcome::Reject {
            needed_gb,
            available_gb,
            staged_gb: None,
        },
    }
}

fn generic_mlx_shared_observation(
    total_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    headroom_gb: f64,
) -> crate::memory_strategy::Selection {
    use crate::memory_strategy::{Budget, Candidate, RequestScope};
    use gen_core::{
        MemoryBackendRealization, MemoryConformanceState, MemoryEvidence, MemoryEvidenceDimensions,
        MemoryEvidenceKey, MemoryEvidenceVerdict, MemoryGeometry, MemoryNumericTier,
        MemoryParityContract, MemoryParityResult, MemoryProviderContract, MemorySelection,
        MemoryStrategy,
    };

    let tier = MemoryNumericTier {
        precision: gen_core::Precision::Bf16,
        quant: None,
    };
    let geometry = MemoryGeometry {
        width: 1,
        height: 1,
        batch: 1,
        frames: 1,
    };
    let selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier,
    };
    let evidence = MemoryEvidence {
        key: MemoryEvidenceKey {
            resolved_route: "generic_mlx_cold_load".into(),
            backend: "mlx".into(),
            tier,
            mode: "image_generation".into(),
            overlay: Some("resolved_load_spec".into()),
            geometry,
            strategy: MemoryStrategy::Resident,
            engaged_composition: vec![MemoryStrategy::Resident],
            parameters: Default::default(),
        },
        conformance: MemoryConformanceState::ImplementedUnverified,
        dimensions: MemoryEvidenceDimensions {
            static_implementation: MemoryEvidenceVerdict::Satisfied,
            declared_calibration: MemoryEvidenceVerdict::Missing,
            historical_verification: MemoryEvidenceVerdict::Missing,
            current_environment_verification: MemoryEvidenceVerdict::Missing,
            canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
            exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
        },
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: String::new(),
        sceneworks_revision: "sc-15449-contract-v1".into(),
        inference_revision: "1c4354b4b22d7f2cf5c4ea5fe17a83ab6c655e82".into(),
        harness_version: String::new(),
        predicted_peak_bytes: total_bytes,
        observed_peak_bytes: None,
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::NotRun,
    };
    let contract = MemoryProviderContract::compatibility_default(
        "generic_mlx_cold_load",
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: false,
            cache_eviction: true,
        },
    );
    crate::memory_strategy::select_strategy(
        RequestScope {
            resolved_route: "generic_mlx_cold_load",
            backend: "mlx",
            tier,
            mode: "image_generation",
            overlay: Some("resolved_load_spec"),
            geometry,
            expected_inference_revision: "1c4354b4b22d7f2cf5c4ea5fe17a83ab6c655e82",
        },
        &contract,
        budget.map(|budget| Budget {
            available_gb: budget.total_gb,
            reclaimable_gb: 0.0,
            total_gb: budget.total_gb,
            reserved_headroom_gb: headroom_gb,
        }),
        &[Candidate {
            selection,
            evidence: &evidence,
        }],
    )
}

/// The largest single component's on-disk weight bytes (GiB) — the wired residency the `Sequential`
/// schedule holds at peak (text encoder(s) dropped before the DiT loads). WEIGHTS ONLY, no activation
/// headroom — contrast [`predicted_sequential_peak_gb`], which adds [`HEADROOM_GB`] for the peak
/// estimate. `rest = total − te` (the DiT + VAE + any folded control branch).
fn staged_weights_gb(total_bytes: u64, te_bytes: u64) -> f64 {
    let rest_bytes = total_bytes.saturating_sub(te_bytes);
    te_bytes.max(rest_bytes) as f64 / BYTES_PER_GIB
}

/// Decision 1 / Decision 2 transition override.
///
/// This deliberately replaces the old `weights_fit_floor` symbol while preserving its outcome for
/// requests which have no exact verified evidence. The settled transition rule requires those cells
/// to remain byte-for-byte legacy, including the 8 GiB q4 guard, until that cell's calibration story
/// opts it into evidence. Verified cells never call this function and genuinely fail closed above.
fn legacy_admission_override(
    total_bytes: u64,
    te_bytes: u64,
    budget: Option<MlxMemoryBudget>,
    sequential_capable: bool,
) -> Option<ResidencyOutcome> {
    let budget = budget?;
    let reserve = crate::fit_gate::legacy_unified_reserve(budget.total_gb);
    let ceiling_gb = budget.total_gb - reserve.gb;
    if sequential_capable {
        (staged_weights_gb(total_bytes, te_bytes) <= ceiling_gb)
            .then_some(ResidencyOutcome::Sequential)
    } else {
        (total_bytes as f64 / BYTES_PER_GIB <= ceiling_gb).then_some(ResidencyOutcome::Resident)
    }
}

/// Pre-load admission + residency-selection gate (sc-10835 Phase 0, sc-10839 Phase 1). Called on the
/// generator cache's cold-load path, before `crate::inference_runtime::load` allocates — never on a warm cache hit,
/// so an already-resident model is never re-gated. Resolves the budget + on-disk component bytes,
/// delegates the choice to [`decide_residency`], and returns the [`LoadSpec`] to load with:
///  - fits resident (or no signal / unmeasurable weights) ⇒ the spec unchanged (warm resident path);
///  - won't fit resident but the provider stages components and the staged peak fits ⇒ the spec with
///    [`OffloadPolicy::Sequential`] set (drop the text encoder(s) before the DiT loads);
///  - won't fit even staged ⇒ [`WorkerError::InvalidPayload`] with an actionable message.
///
/// `engine_id` is both the [`engine_supports_sequential`] key and the human-facing model name in the
/// rejection message.
pub(crate) fn apply_residency_policy(spec: LoadSpec, engine_id: &str) -> WorkerResult<LoadSpec> {
    // Respect an offload policy already chosen upstream (defensive: the MLX cache seam normally sees
    // the default `Resident`, but never downgrade a `Sequential` set by another gate).
    if spec.offload_policy == OffloadPolicy::Sequential {
        return Ok(spec);
    }
    match decide_residency_for_spec(engine_id, &spec) {
        ResidencyOutcome::Resident => Ok(spec),
        ResidencyOutcome::Sequential => {
            let (total_bytes, te_bytes, _) = spec_component_bytes(engine_id, &spec);
            tracing::info!(
                event = "mlx_sequential_residency_selected",
                engine = %engine_id,
                total_gb = (total_bytes as f64 / BYTES_PER_GIB).round() as i64,
                text_encoder_gb = (te_bytes as f64 / BYTES_PER_GIB).round() as i64,
            );
            Ok(spec.with_offload_policy(OffloadPolicy::Sequential))
        }
        ResidencyOutcome::Reject {
            needed_gb,
            available_gb,
            staged_gb,
        } => Err(too_big_error(engine_id, needed_gb, available_gb, staged_gb)),
    }
}

/// The `(total, text-encoder)` on-disk component bytes a `spec` loads (sc-10894 seam). The whole-model
/// sum plus the staged text-encoder split, preferring the provider-owned per-component footprint over
/// the `text_encoder*` subdir scan (which reads ZERO for boogu `mllm/`, bernini flat `t5_encoder`,
/// anima `text_encoders/`, etc.), and folding a separate `spec.control` (qwen_image_control's VACE
/// branch) into the HEAVY side so the staged split `rest = total − te` counts it on the DiT side.
fn spec_component_bytes(engine_id: &str, spec: &LoadSpec) -> (u64, u64, HeadroomAllowance) {
    let footprint = crate::inference_runtime::media()
        .footprint(engine_id, spec)
        .ok()
        .flatten();
    let footprint_te = footprint.map(|fp| fp.text_encoder);
    let mut headroom = HeadroomAllowance::GENERIC;
    let (mut total_bytes, te_bytes) = match &spec.weights {
        WeightsSource::Dir(dir) => {
            let mut total = sum_safetensors_bytes(dir);
            let te = resolve_text_encoder_bytes(footprint_te, dir);
            if matches!(engine_id, "lens" | "lens_turbo") {
                let disk_te = sum_safetensors_bytes(&dir.join("text_encoder"));
                let materialized_expansion = te.saturating_sub(disk_te);
                total = total.saturating_add(materialized_expansion);
                if spec.quantize.is_none() && packed_quant_bits(dir, "text_encoder").is_none() {
                    headroom = HeadroomAllowance::LENS_DENSE;
                }
            }
            (total, te)
        }
        // A single-file source has no diffusers component tree; honor a footprint TE if the provider
        // somehow computed one, else 0 (resident-or-reject only).
        WeightsSource::File(file) => (
            std::fs::metadata(file).map_or(0, |meta| meta.len()),
            footprint_te.unwrap_or(0),
        ),
    };
    if let Some(control) = &spec.control {
        total_bytes += weights_source_bytes(control);
    }
    // Caller-provisioned components (epic 13657) are staged from a DIFFERENT snapshot than
    // `spec.weights`, so the dir scan above cannot see them (sc-15154). Mage-Flow's per-tier dir
    // holds the DiT alone — its text encoder and VAE are bit-identical across the six variants and
    // hosted once in a shared mirror — so an unstaged sum scored a q4 edit install at 2.17 GiB
    // against a real 6.52 GiB, which both under-quoted the over-budget message and let the
    // the legacy override admit tiers that do not fit. A component resolved to a path INSIDE the
    // weights dir is skipped, because the scan already counted it.
    for source in spec.components.values() {
        let inside = match source {
            WeightsSource::Dir(path) | WeightsSource::File(path) => match &spec.weights {
                WeightsSource::Dir(root) => path.starts_with(root),
                WeightsSource::File(_) => false,
            },
        };
        if !inside {
            total_bytes = total_bytes.saturating_add(weights_source_bytes(source));
        }
    }
    (total_bytes, te_bytes, headroom)
}

fn packed_quant_bits(root: &std::path::Path, component: &str) -> Option<i64> {
    let config = std::fs::read(root.join(component).join("config.json")).ok()?;
    serde_json::from_slice::<serde_json::Value>(&config)
        .ok()?
        .get("quantization")?
        .get("bits")?
        .as_i64()
}

/// The residency outcome (Resident / Sequential / Reject) a `spec` would take against this machine's
/// unified-memory budget — the pure decision behind [`apply_residency_policy`], factored out so the
/// capability downtier (sc-10733) can evaluate a candidate tier's fit at the base.rs seam WITHOUT
/// building the final spec twice. Same budget + component-byte + sequential-capability inputs the live
/// gate uses, so the seam's downtier choice and the cache's admission never disagree.
pub(crate) fn decide_residency_for_spec(engine_id: &str, spec: &LoadSpec) -> ResidencyOutcome {
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    let (total_bytes, te_bytes, headroom) = spec_component_bytes(engine_id, spec);
    // The LOAD-time gate is resolution-blind, so it budgets the family's whole flat allowance and
    // has no use for the reserve split (sc-16195 changes only the request-scoped estimator).
    decide_residency_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        engine_supports_sequential(engine_id),
        headroom.total_gb,
    )
}

/// The residency outcome for a candidate tier's WEIGHTS DIR — a bare-`Dir` convenience over
/// [`decide_residency_for_spec`], kept for the live real-weights gate below.
///
/// The base.rs capability downtier (sc-10733) used to call this and now builds its probe spec with
/// `tier_probe_spec` instead, because a bare `Dir` under-counts any model whose components are
/// caller-staged from another snapshot (sc-15154).
#[cfg(test)]
pub(crate) fn residency_for_dir(
    engine_id: &str,
    weights_dir: &std::path::Path,
) -> ResidencyOutcome {
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.to_path_buf()));
    decide_residency_for_spec(engine_id, &spec)
}

/// The on-disk WEIGHT bytes (GiB) a `spec` loads — the weights half of the number
/// [`decide_residency_for_spec`] rejects on (`Σweights + `[`HEADROOM_GB`]).
///
/// For the reject message (sc-15154). The peak alone cannot be read: on a small budget the flat
/// headroom dominates it, so a tier whose real install is 7 GB is refused with a ~25 GB figure and
/// the number reads like the wrong tier's total. Naming both makes the split legible — and makes a
/// mis-scoped footprint visible instead of hiding inside a constant.
#[cfg(target_os = "macos")]
pub(crate) fn spec_weights_gb(engine_id: &str, spec: &LoadSpec) -> f64 {
    spec_component_bytes(engine_id, spec).0 as f64 / BYTES_PER_GIB
}

/// Build the actionable over-budget rejection. `staged_gb` is `Some` when sequential residency was
/// tried and its staged peak still overflows (so the message names both the resident and the staged
/// requirement — telling the user even one-component-at-a-time won't fit); `None` for a plain resident
/// reject on a non-staging provider. Split out so the message is testable without the live probe.
fn too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    staged_gb: Option<f64>,
) -> WorkerError {
    let staged_note = match staged_gb {
        Some(staged) => format!(
            " (~{} GB even loading one component at a time)",
            staged.round() as i64
        ),
        None => String::new(),
    };
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of unified memory{staged_note} (model weights plus \
         headroom for activations and the OS) but this machine has ~{available} GB. Choose a \
         smaller quant tier, lower the output resolution, or run on a Mac with more memory.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
    ))
}

// ---------------------------------------------------------------------------------------------
// Full base fine-tune training memory-envelope gate (epic 14034 Mage-Flow / sc-14056)
// ---------------------------------------------------------------------------------------------
//
// A LoRA/LoKr run freezes the base and trains a tiny adapter, so it rides the base-installed check and
// the generation-style residency of a frozen model. A **full base fine-tune** (sc-14056) is a
// different memory regime: it makes EVERY DiT weight trainable, so the resident state is the
// full-precision master weights PLUS the optimizer's per-parameter state PLUS the gradients, and —
// because gradient (activation) checkpointing is not yet ported (sc-14989) — the dense retained-graph
// backward holds the whole forward's activations at once. That envelope grows far beyond the on-disk
// weight bytes and, at production resolution, exceeds any consumer Mac.
//
// This gate is the training analogue of the generation `apply_residency_policy` admission check, and
// it REUSES this module's machinery per sc-14056: `sum_safetensors_bytes` for the dense DiT bytes,
// `resolve_budget` + the `sysctl hw.memsize` probe (+ `SCENEWORKS_MLX_MEMORY_CAP_ENV` emulation) for
// the platform-aware unified-memory budget, and the typed legacy unified reserve. It is a pre-flight
// admission check (an MLX overcommit SIGKILLs the process — it cannot be caught after the fact, the
// same reason the generation gate and the Mochi gate are pre-flight), so a full-tune this machine
// cannot hold is refused up front with an actionable message rather than an uncatchable mid-run kill.
//
// The envelope is deliberately CONSERVATIVE (it errs toward rejection): under-predicting is the
// dangerous direction — it would advertise a production-resolution full-tune as fitting when it will
// not, exactly the failure sc-14056 calls out. The state multiplier is exact arithmetic *given the
// configuration* — it is 8× at `gradient_accumulation == 1` and 10× above it, because the engine's
// accumulator holds one more full f32 gradient map for the length of the window. The activation terms
// are a documented pre-grad-checkpointing ESTIMATE, uncalibrated in both directions against MLX's
// fast-SDPA vjp; sc-15038 tracks measuring them on-device with `mlx::get_peak_memory` the way
// sc-14053 calibrated the quant envelopes.

/// Peak resident training state as a multiple of the dense DiT's **bf16 on-disk** bytes: f32 master
/// weights (2×) + AdamW first/second moments in f32 (4×) + f32 gradients (2×). The in-trace bf16
/// weight reconstruction is a transient freed before the optimizer step, so this 8× step-peak
/// dominates. This term is resolution-independent and is exact arithmetic (a 4B-parameter model needs
/// ~8× its bf16 bytes of state regardless of resolution), so it alone gates full fine-tunes off Macs
/// that are simply too small for the optimizer state.
///
/// It is the **`gradient_accumulation == 1`** figure — see
/// [`FULL_FINETUNE_ACCUM_MULTIPLIER`] for the extra map an accumulation window holds.
const FULL_FINETUNE_STATE_MULTIPLIER: f64 = 8.0;

/// Extra peak state, again as a multiple of the bf16 on-disk DiT bytes, held **only** when
/// `gradient_accumulation > 1`.
///
/// The engine's `accumulate_grads` (mlx-gen `train/lora.rs`) keeps a running accumulator that is a
/// FULL f32 gradient map (2× the bf16 bytes) alive for the whole window, *in addition to* the
/// per-micro-step gradient map already counted in [`FULL_FINETUNE_STATE_MULTIPLIER`] — both are live
/// simultaneously inside the summing loop. So the real step peak is 10×, not 8×, and it is the
/// accumulator that pushes a 4B full tune from ~61 GiB of state to ~77 GiB.
///
/// This was the omission that made the "exact arithmetic" claim wrong in the first cut of this gate
/// (sc-14056 review): `gradient_accumulation` is a user-exposed knob (Training Studio "Gradient
/// accumulation"), so a 64–80 GB Mac was predicted at ~63 GiB, admitted, and then hard-killed —
/// an MLX overcommit is a SIGKILL the worker cannot catch, which is the whole reason this gate is
/// pre-flight. `average_grads` afterwards holds at most the same two full maps (its input and its
/// output), so it needs no term of its own; 10× bounds both phases.
const FULL_FINETUNE_ACCUM_MULTIPLIER: f64 = 2.0;

/// Retained attention bytes per (query,key) token pair, summed over all heads and blocks, that the
/// dense backward keeps WITHOUT gradient checkpointing (sc-14989): Mage's `num_heads = 24` × `depth =
/// 12` × f32 (4 bytes) × a conservative ×3 retain factor for the score / softmax / scaled-score
/// intermediates. This term is QUADRATIC in the packed token count and is what makes a
/// production-resolution full-tune exceed even a large Mac until checkpointing lands.
const FULL_FINETUNE_ATTN_BYTES_PER_TOKEN_PAIR: f64 = 24.0 * 12.0 * 4.0 * 3.0;

/// Retained feature-map bytes per token, summed over the block stack: ≈24 retained intermediates ×
/// `hidden_size = 3072` × `depth = 12` × f32 (4 bytes). Linear in the packed token count.
const FULL_FINETUNE_FEATURE_BYTES_PER_TOKEN: f64 = 24.0 * 3072.0 * 12.0 * 4.0;

/// The packed token count a full fine-tune trains at `resolution`: the trainer buckets the edge to a
/// multiple of 32, divides by the 16× Mage-VAE stride, and — `patch_size == 1` — packs exactly the
/// square latent grid (`grid²`).
fn full_finetune_tokens(resolution: u32) -> f64 {
    let edge = (resolution / 32).max(1) * 32;
    let grid = (edge / 16) as f64;
    grid * grid
}

/// Predicted unified-memory peak (GiB) of a full base fine-tune of a `dit_bytes` (bf16 on-disk) DiT at
/// `resolution` with `gradient_accumulation` micro-steps per optimizer update, WITHOUT gradient
/// checkpointing. `None` when `dit_bytes == 0` (nothing installed → no signal → never block),
/// mirroring [`predicted_peak_gb`].
fn full_finetune_peak_gb(
    dit_bytes: u64,
    resolution: u32,
    gradient_accumulation: u32,
) -> Option<f64> {
    if dit_bytes == 0 {
        return None;
    }
    let dit_gb = dit_bytes as f64 / BYTES_PER_GIB;
    let tokens = full_finetune_tokens(resolution);
    // The accumulator map exists only when the window is longer than one micro-step; `0` and `1` both
    // mean "step every micro-step" to the engine, so neither pays the term.
    let state_multiplier = if gradient_accumulation > 1 {
        FULL_FINETUNE_STATE_MULTIPLIER + FULL_FINETUNE_ACCUM_MULTIPLIER
    } else {
        FULL_FINETUNE_STATE_MULTIPLIER
    };
    let state = dit_gb * state_multiplier;
    let attention = tokens * tokens * FULL_FINETUNE_ATTN_BYTES_PER_TOKEN_PAIR / BYTES_PER_GIB;
    let feature = tokens * FULL_FINETUNE_FEATURE_BYTES_PER_TOKEN / BYTES_PER_GIB;
    Some(state + attention + feature + crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB)
}

/// The pure full-fine-tune admission decision: `Some(message)` when the predicted peak overflows the
/// machine's unified-memory budget, `None` to admit. Missing either signal admits (never block without
/// evidence — the fit-gate invariant shared with [`fit_decision`] and [`mochi_fit_error`]).
fn full_finetune_fit_message(
    model_label: &str,
    dit_bytes: u64,
    resolution: u32,
    gradient_accumulation: u32,
    budget: Option<MlxMemoryBudget>,
) -> Option<String> {
    let (needed_gb, budget) = (
        full_finetune_peak_gb(dit_bytes, resolution, gradient_accumulation)?,
        budget?,
    );
    (budget.total_gb + f64::EPSILON < needed_gb).then(|| {
        full_finetune_too_big_message(
            model_label,
            needed_gb,
            budget.total_gb,
            resolution,
            gradient_accumulation,
        )
    })
}

/// The actionable over-budget rejection for a full base fine-tune — names the model, the constraint
/// (unified memory), what it needs and what the machine has, and the levers that actually move the
/// dominant terms: a lower training resolution, a LoRA adapter instead of a full fine-tune, or a bigger
/// Mac — and states plainly that a production-resolution full fine-tune needs gradient checkpointing
/// (sc-14989), which is not yet available.
fn full_finetune_too_big_message(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    resolution: u32,
    gradient_accumulation: u32,
) -> String {
    // Naming gradient accumulation only when it is actually costing memory keeps the advice
    // actionable: at accumulation 1 there is no accumulator map and "lower it" would be noise.
    let accum_lever = if gradient_accumulation > 1 {
        format!(
            " Gradient accumulation is {gradient_accumulation}, which holds an extra full-size \
             gradient buffer for the whole window — setting it to 1 removes that buffer."
        )
    } else {
        String::new()
    };
    format!(
        "A full base fine-tune of {model_label} at {resolution}px needs ~{needed} GB of unified \
         memory (the full-precision master weights, the optimizer state, and — without gradient \
         checkpointing — the whole retained backward graph) but this machine has ~{available} GB. \
         Lower the training resolution, train a LoRA adapter instead of a full fine-tune, or run on \
         a Mac with more memory.{accum_lever} A production-resolution full fine-tune needs gradient \
         (activation) checkpointing (sc-14989), which is not yet available.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
    )
}

/// Live, platform-aware **full base fine-tune** memory pre-flight for the training submit gate
/// (sc-14056). Sums the base model's dense DiT bytes (`<base_model_dir>/transformer`), resolves this
/// machine's unified-memory budget with the same `sysctl hw.memsize` probe + `SCENEWORKS_MLX_MEMORY_
/// CAP_GB` emulation the generation fit gate uses, and returns a clear rejection message when a full
/// fine-tune at `resolution` with `gradient_accumulation` micro-steps per update won't fit — or
/// `None` to permit. `gradient_accumulation` is load-bearing, not cosmetic: a window longer than one
/// micro-step keeps an extra full-size f32 gradient buffer resident, which is the difference between
/// admitting and refusing a 4B full tune on a 64–80 GB Mac. Permits (returns `None`) whenever there
/// is no signal: the base isn't installed (`transformer/` sums to 0 bytes — the base-installed gate
/// covers that case), or the platform probe yields no budget (off-macOS, where a full-tune would run
/// on a different backend with its own envelope). Only the full-fine-tune path needs this; a LoRA run
/// trains against a frozen base and is covered by the base-installed check.
///
/// Re-exported publicly from the crate root as `sceneworks_worker::full_finetune_memory_error` so the
/// rust-api training submit gate can call it alongside `training_base_model_status` /
/// `training_disk_space_error`.
pub fn full_finetune_memory_error(
    base_model_dir: &Path,
    resolution: u32,
    gradient_accumulation: u32,
    model_label: &str,
) -> Option<String> {
    let dit_bytes = sum_safetensors_bytes(&base_model_dir.join("transformer"));
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    full_finetune_fit_message(
        model_label,
        dit_bytes,
        resolution,
        gradient_accumulation,
        budget,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_evidence_is_keyed_by_the_live_contract_composition() {
        use gen_core::{MemoryPrerequisiteScope, MemoryStrategyPrerequisite};

        let mut contract = MemoryProviderContract::compatibility_default(
            "fixture_provider",
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        contract.additional_prerequisites.push((
            MemoryStrategy::Resident,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("staged-residency capability")
            .support = gen_core::MemoryStrategySupport::Implemented;
        let (_, evidence) = resident_evidence(
            &contract,
            MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q4),
            },
            "text_to_image",
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            1,
            None,
        );
        assert_eq!(
            evidence.key.engaged_composition,
            vec![MemoryStrategy::Resident, MemoryStrategy::StagedResidency,],
            "the test must distinguish live derivation from the old hardcoded resident-only key"
        );
    }

    struct RequestGenerator {
        descriptor: gen_core::ModelDescriptor,
        contract: Option<MemoryProviderContract>,
    }

    impl gen_core::Generator for RequestGenerator {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        fn validate(&self, _req: &gen_core::GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            _req: &gen_core::GenerationRequest,
            _on_progress: &mut dyn FnMut(gen_core::Progress),
        ) -> gen_core::Result<gen_core::GenerationOutput> {
            unreachable!("request selector tests do not execute tensors")
        }

        fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
            self.contract.as_ref()
        }
    }

    fn request_generator(contract: Option<MemoryProviderContract>) -> RequestGenerator {
        RequestGenerator {
            descriptor: gen_core::ModelDescriptor {
                id: "mage_flow",
                family: "test",
                backend: "mlx",
                modality: gen_core::Modality::Image,
                capabilities: gen_core::Capabilities::default(),
                required_components: &[],
            },
            contract,
        }
    }

    fn request_plan() -> MlxRequestPlan {
        MlxRequestPlan {
            engine_id: "mage_flow",
            model_id: "mage_flow".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q4),
            },
            asset_bytes: gib_to_bytes(6.0),
            // Deliberately ABOVE the 2 GiB fixed reserve so the area term is non-zero: a fixture
            // sitting exactly on the reserve would model resolution-blind and silently stop
            // exercising the sc-16195 scaling at all.
            activation_headroom_bytes: gib_to_bytes(6.0),
            fixed_reserve_bytes: gib_to_bytes(2.0),
            calibration: MlxCalibrationConfig::Absent,
        }
    }

    fn mage_request_contract() -> MemoryProviderContract {
        use gen_core::MemoryCalibrationIdentity;

        let mut contract = MemoryProviderContract::compatibility_default(
            "mage_flow",
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        contract.calibration = Some(MemoryCalibrationIdentity::new(MAGE_CALIBRATION_FINGERPRINT));
        contract.asset_facts.base_bytes = gib_to_bytes(6.0);
        contract
    }

    fn request_inputs(width: u32, height: u32, count: u32) -> MlxRequestInputs {
        MlxRequestInputs {
            width,
            height,
            count,
            mode: "edit_image".to_owned(),
            overlay: Some("references:2+mask+adapters:1".to_owned()),
            adapter_count: 0,
            has_reference: true,
            use_pid: false,
            has_phases: false,
        }
    }

    fn fixture_bundle() -> EvidenceBundle {
        match sceneworks_core::memory_calibration::load_bundle(include_str!(
            "../tests/fixtures/mlx-memory-calibration.json"
        ))
        .expect("valid MLX calibration fixture")
        {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("unexpected stale fixture: {reason:?}"),
        }
    }

    fn fixture_binding(tier: &str, variant: &str) -> MlxCalibrationBinding {
        let parameters = JsonObject::from_iter([
            ("decodeTileEdge".to_owned(), serde_json::json!(512)),
            ("decodeOverlap".to_owned(), serde_json::json!(128)),
        ]);
        fixture_binding_for(tier, variant, StrategyRung::BoundedDecode, parameters)
    }

    fn fixture_binding_for(
        tier: &str,
        variant: &str,
        rung: StrategyRung,
        parameters: JsonObject<String, Value>,
    ) -> MlxCalibrationBinding {
        MlxCalibrationBinding {
            query: CalibrationBinding {
                abi: sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
                fingerprint: "fixture-formula-v2".to_owned(),
                scene_works_revision: "a".repeat(40),
                matrix_source_revision: "source-tree:1111111".to_owned(),
                inference_revision: "b".repeat(40),
                artifact_repository: "SceneWorks/fixture".to_owned(),
                artifact_resolved_revision: "c".repeat(40),
                artifact_variant: variant.to_owned(),
                resolved_path_fingerprint: format!(
                    "SceneWorks/fixture@{}:{variant}",
                    "c".repeat(40)
                ),
            },
            provider: "fixture_provider".to_owned(),
            tier: tier.to_owned(),
            mode: "text_to_image".to_owned(),
            overlay: "none".to_owned(),
            geometry: CalibrationGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            rung,
            selection_parameters: parse_evidence_parameters(rung, &parameters)
                .expect("fixture parameters"),
            parameters,
        }
    }

    fn fixture_calibration_json(tier: &str, variant: &str) -> Value {
        serde_json::json!({
            "abi": sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
            "fingerprint": "fixture-formula-v2",
            "sceneWorksRevision": "a".repeat(40),
            "matrixSourceRevision": "source-tree:1111111",
            "inferenceRevision": "b".repeat(40),
            "provider": "fixture_provider",
            "tier": tier,
            "mode": "text_to_image",
            "overlay": "none",
            "geometry": {
                "width": 1024,
                "height": 1024,
                "batch": 1,
                "frames": 1
            },
            "artifactRepository": "SceneWorks/fixture",
            "artifactResolvedRevision": "c".repeat(40),
            "artifactVariant": variant,
            "resolvedPathFingerprint": format!(
                "SceneWorks/fixture@{}:{variant}",
                "c".repeat(40)
            ),
            "rung": "bounded_decode",
            "parameters": {
                "decodeTileEdge": 512,
                "decodeOverlap": 128
            }
        })
    }

    fn fixture_manifest(calibrations: Vec<Value>) -> JsonObject<String, Value> {
        serde_json::json!({ "mlx": { "calibrations": calibrations } })
            .as_object()
            .expect("manifest object")
            .clone()
    }

    fn fixture_spec(tier: gen_core::Quant, variant: &str) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(format!(
            "/cache/models--SceneWorks--fixture/snapshots/{}/{variant}/weights",
            "c".repeat(40)
        ))))
        .with_quant(tier)
    }

    fn fixture_provenance(tier: &str, variant: &str) -> ResolvedArtifactProvenance {
        ResolvedArtifactProvenance {
            identity: crate::model_jobs::ResolvedArtifactIdentity {
                repository: "SceneWorks/fixture".to_owned(),
                revision: "c".repeat(40),
                variant: variant.to_owned(),
                fingerprint: format!("SceneWorks/fixture@{}:{variant}", "c".repeat(40)),
            },
            fixed_artifact_tier: Some(tier.to_owned()),
        }
    }

    fn fixture_plan() -> MlxRequestPlan {
        let variant = "packed-q4";
        MlxRequestPlan {
            engine_id: "fixture_provider",
            model_id: "fixture_model".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q4),
            },
            asset_bytes: gib_to_bytes(3.0),
            // Deliberately ABOVE the 2 GiB fixed reserve so the area term is non-zero: a fixture
            // sitting exactly on the reserve would model resolution-blind and silently stop
            // exercising the sc-16195 scaling at all.
            activation_headroom_bytes: gib_to_bytes(6.0),
            fixed_reserve_bytes: gib_to_bytes(2.0),
            calibration: MlxCalibrationConfig::Valid(MlxCalibrationSet {
                bindings: vec![fixture_binding("q4", variant)],
                resolved: fixture_provenance("q4", variant),
            }),
        }
    }

    fn packaged_krea_plan() -> MlxRequestPlan {
        let bundle = match sceneworks_core::memory_calibration::load_packaged_bundle()
            .expect("packaged calibration bundle parses")
        {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => {
                panic!("packaged Krea evidence must be current: {reason:?}")
            }
        };
        let records = bundle
            .records
            .iter()
            .filter(|record| {
                matches!(record.backend, CalibrationBackend::Mlx)
                    && record.target.model_id == "krea_2_turbo"
                    && record.target.provider == "krea_2_turbo_control"
                    && record.target.tier == "q4"
                    && record.target.mode == "text_to_image"
                    && record.target.overlay == "control:1"
            })
            .collect::<Vec<_>>();
        assert_eq!(
            records.len(),
            2,
            "the packaged Krea contract has two exact cells"
        );
        let first = records[0];
        let resolved_path_fingerprint =
            |record: &sceneworks_core::memory_calibration::EvidenceRecord| {
                let sceneworks_core::memory_calibration::RequiredNullable::Value(fingerprint) =
                    &record.loadability.resolved_path_fingerprint
                else {
                    panic!("complete Krea record must carry a resolved path fingerprint");
                };
                fingerprint.clone()
            };
        let resolved = ResolvedArtifactProvenance {
            identity: crate::model_jobs::ResolvedArtifactIdentity {
                repository: first.artifact.repository.clone(),
                revision: first.artifact.resolved_revision.clone(),
                variant: first.artifact.variant.clone(),
                fingerprint: resolved_path_fingerprint(first),
            },
            fixed_artifact_tier: Some(first.target.tier.clone()),
        };
        let bindings = records
            .into_iter()
            .map(|record| MlxCalibrationBinding {
                query: CalibrationBinding {
                    abi: sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
                    fingerprint: record.calibration_fingerprint.clone(),
                    scene_works_revision: "sc-16099-contract-v1".to_owned(),
                    matrix_source_revision: record
                        .repositories
                        .scene_works
                        .matrix_source_revision
                        .clone()
                        .expect("current matrix source revision"),
                    inference_revision: record.repositories.inference.revision.clone(),
                    artifact_repository: record.artifact.repository.clone(),
                    artifact_resolved_revision: record.artifact.resolved_revision.clone(),
                    artifact_variant: record.artifact.variant.clone(),
                    resolved_path_fingerprint: resolved_path_fingerprint(record),
                },
                provider: record.target.provider.clone(),
                tier: record.target.tier.clone(),
                mode: record.target.mode.clone(),
                overlay: record.target.overlay.clone(),
                geometry: CalibrationGeometry {
                    width: record.target.geometry.width,
                    height: record.target.geometry.height,
                    batch: record.target.geometry.batch,
                    frames: record.target.geometry.frames,
                },
                rung: record.strategy.rung,
                selection_parameters: parse_evidence_parameters(
                    record.strategy.rung,
                    &record.strategy.parameters,
                )
                .expect("packaged Krea strategy parameters"),
                parameters: record.strategy.parameters.clone(),
            })
            .collect();
        MlxRequestPlan {
            engine_id: "krea_2_turbo_control",
            model_id: "krea_2_turbo".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q4),
            },
            asset_bytes: gib_to_bytes(30.0),
            activation_headroom_bytes: gib_to_bytes(2.0),
            fixed_reserve_bytes: 0,
            calibration: MlxCalibrationConfig::Valid(MlxCalibrationSet { bindings, resolved }),
        }
    }

    fn packaged_krea_generator() -> RequestGenerator {
        use gen_core::MemoryCalibrationIdentity;

        let mut generator = fixture_generator();
        generator.descriptor.id = "krea_2_turbo_control";
        let contract = generator.contract.as_mut().expect("fixture contract");
        contract.provider_id = "krea_2_turbo_control".to_owned();
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "krea-control-mlx-v4-q4-pose-bounded-decode-512-64",
        ));
        let bounded_decode = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .expect("bounded decode capability");
        bounded_decode.parameters.decode_overlaps = vec![64];
        generator
    }

    fn fixture_inputs(width: u32, height: u32) -> MlxRequestInputs {
        MlxRequestInputs {
            width,
            height,
            count: 1,
            mode: "text_to_image".to_owned(),
            overlay: None,
            adapter_count: 0,
            has_reference: false,
            use_pid: false,
            has_phases: false,
        }
    }

    fn fixture_generator() -> RequestGenerator {
        use gen_core::{
            MemoryCalibrationIdentity, MemoryLifecycleCapabilities, MemoryParameterRanges,
            MemoryPhase, MemoryStrategyCapability, MemoryStrategySupport,
        };

        let mut contract = MemoryProviderContract::compatibility_default(
            "fixture_provider",
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        contract.calibration = Some(MemoryCalibrationIdentity::new("fixture-formula-v2"));
        contract.asset_facts.base_bytes = gib_to_bytes(3.0);
        contract.lifecycle = MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: false,
        };
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: if matches!(
                    strategy,
                    MemoryStrategy::Resident
                        | MemoryStrategy::StagedResidency
                        | MemoryStrategy::BoundedDecode
                        | MemoryStrategy::BoundedAttention
                ) {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                },
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![128],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                        attention_chunk_sizes: vec![256],
                        ..Default::default()
                    },
                    _ => MemoryParameterRanges::default(),
                },
            })
            .collect();
        RequestGenerator {
            descriptor: gen_core::ModelDescriptor {
                id: "fixture_provider",
                family: "test",
                backend: "mlx",
                modality: gen_core::Modality::Image,
                capabilities: gen_core::Capabilities::default(),
                required_components: &[],
            },
            contract: Some(contract),
        }
    }

    fn fixture_budget(total_gib: f64) -> MemoryBudget {
        MemoryBudget {
            total_bytes: gib_to_bytes(total_gib),
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        }
    }

    fn fixture_ladder() -> (EvidenceBundle, MlxRequestPlan) {
        use sceneworks_core::memory_calibration::{PredictedPeakBytes, RequiredNullable};

        let mut bundle = fixture_bundle();
        let base = bundle.records.remove(0);
        let generator = fixture_generator();
        let contract = generator.contract.as_ref().expect("fixture contract");
        let rung = |rung, parameters: JsonObject<String, Value>, peak_gib: f64| {
            let mut record = base.clone();
            record.strategy.rung = rung;
            record.strategy.engaged_rungs = contract
                .engaged_composition(evidence_strategy(rung))
                .into_iter()
                .map(|strategy| match strategy {
                    MemoryStrategy::Resident => StrategyRung::Resident,
                    MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
                    MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
                    MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
                    MemoryStrategy::BoundedTransformerResidency => {
                        StrategyRung::BoundedTransformerResidency
                    }
                })
                .collect();
            record.strategy.parameters = parameters.clone();
            record.sweep.cases[0].parameters = parameters;
            if let RequiredNullable::Value(predicted) = &mut record.predicted_peak_bytes {
                *predicted = PredictedPeakBytes {
                    conditioning: predicted.conditioning.min(gib_to_bytes(peak_gib)),
                    denoise: gib_to_bytes(peak_gib),
                    decode: predicted.decode.min(gib_to_bytes(peak_gib)),
                    overall: gib_to_bytes(peak_gib),
                };
            }
            record
        };
        let resident_parameters = JsonObject::new();
        let decode_parameters = JsonObject::from_iter([
            ("decodeTileEdge".to_owned(), serde_json::json!(512)),
            ("decodeOverlap".to_owned(), serde_json::json!(128)),
        ]);
        let attention_parameters = JsonObject::from_iter([
            ("decodeTileEdge".to_owned(), serde_json::json!(512)),
            ("decodeOverlap".to_owned(), serde_json::json!(128)),
            ("attentionChunkSize".to_owned(), serde_json::json!(256)),
        ]);
        bundle.records = vec![
            rung(StrategyRung::Resident, resident_parameters.clone(), 7.0),
            rung(StrategyRung::BoundedDecode, decode_parameters.clone(), 5.0),
            rung(
                StrategyRung::BoundedAttention,
                attention_parameters.clone(),
                4.0,
            ),
        ];
        let mut plan = fixture_plan();
        let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
            panic!("fixture calibration");
        };
        calibration.bindings = vec![
            fixture_binding_for(
                "q4",
                "packed-q4",
                StrategyRung::Resident,
                resident_parameters,
            ),
            fixture_binding_for(
                "q4",
                "packed-q4",
                StrategyRung::BoundedDecode,
                decode_parameters,
            ),
            fixture_binding_for(
                "q4",
                "packed-q4",
                StrategyRung::BoundedAttention,
                attention_parameters,
            ),
        ];
        (bundle, plan)
    }

    #[test]
    fn verified_lower_geometry_requires_exact_current_identity_and_geometry() {
        let mut bundle = fixture_bundle();
        let mut lower_record = bundle.records.remove(0);
        lower_record.target.geometry = CalibrationGeometry {
            width: 768,
            height: 768,
            batch: 1,
            frames: 1,
        };
        bundle.records = vec![lower_record];

        let mut plan = fixture_plan();
        let mut lower_binding = fixture_binding("q4", "packed-q4");
        lower_binding.geometry = CalibrationGeometry {
            width: 768,
            height: 768,
            batch: 1,
            frames: 1,
        };
        let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
            panic!("fixture calibration");
        };
        calibration.bindings = vec![lower_binding.clone()];
        let inputs = fixture_inputs(1024, 1024);
        let MlxCalibrationConfig::Valid(calibration) = &plan.calibration else {
            panic!("fixture calibration");
        };
        assert_eq!(
            verified_lower_geometry(
                &bundle,
                calibration,
                &plan,
                &inputs,
                "text_to_image",
                fixture_budget(128.0),
            ),
            Some(CalibrationGeometry {
                width: 768,
                height: 768,
                batch: 1,
                frames: 1,
            })
        );

        let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
            panic!("fixture calibration");
        };
        calibration.bindings[0].query.fingerprint = "mutated".to_owned();
        let MlxCalibrationConfig::Valid(calibration) = &plan.calibration else {
            panic!("fixture calibration");
        };
        assert_eq!(
            verified_lower_geometry(
                &bundle,
                calibration,
                &plan,
                &inputs,
                "text_to_image",
                fixture_budget(128.0),
            ),
            None,
            "a fingerprint mutation must stop the refusal from naming the lower geometry"
        );

        let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
            panic!("fixture calibration");
        };
        calibration.bindings[0] = lower_binding;
        calibration.bindings[0].geometry.width = 640;
        let MlxCalibrationConfig::Valid(calibration) = &plan.calibration else {
            panic!("fixture calibration");
        };
        assert_eq!(
            verified_lower_geometry(
                &bundle,
                calibration,
                &plan,
                &inputs,
                "text_to_image",
                fixture_budget(128.0),
            ),
            None,
            "a geometry mutation must stop the refusal from naming the lower geometry"
        );
    }

    #[test]
    fn exact_infeasible_geometry_refuses_before_provider_and_names_only_verified_lower_geometry() {
        use sceneworks_core::memory_calibration::{PredictedPeakBytes, RequiredNullable};

        let mut bundle = fixture_bundle();
        let mut high_record = bundle.records.remove(0);
        let mut lower_record = high_record.clone();
        lower_record.target.geometry = CalibrationGeometry {
            width: 768,
            height: 768,
            batch: 1,
            frames: 1,
        };
        let RequiredNullable::Value(predicted) = &mut high_record.predicted_peak_bytes else {
            panic!("fixture predicted peak");
        };
        *predicted = PredictedPeakBytes {
            conditioning: predicted.conditioning,
            denoise: gib_to_bytes(6.0),
            decode: predicted.decode,
            overall: gib_to_bytes(6.0),
        };
        bundle.records = vec![high_record, lower_record];

        let mut plan = fixture_plan();
        let mut lower_binding = fixture_binding("q4", "packed-q4");
        lower_binding.geometry = CalibrationGeometry {
            width: 768,
            height: 768,
            batch: 1,
            frames: 1,
        };
        let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
            panic!("fixture calibration");
        };
        calibration.bindings.push(lower_binding);

        let error = evaluate_request_with_budget_using_bundle(
            &fixture_generator(),
            &plan,
            &fixture_inputs(1024, 1024),
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(8.0),
            gib_to_bytes(6.0),
            0,
            &[],
            Some(&bundle),
        )
        .expect_err("the 6 GiB high record plus its 3 GiB foreign reserve must refuse");
        let message = error.to_string();
        assert!(
            message.contains("smallest verified MLX host boundary"),
            "the refusal must come from the post-handshake exact evidence precheck: {message}"
        );
        assert!(
            message.contains("current verified alternative: 768x768"),
            "the refusal must name the lower exact record: {message}"
        );
    }

    #[test]
    fn packaged_krea_1024_refuses_before_render_and_names_only_a_fitting_current_cell() {
        use gen_core::MemoryCalibrationIdentity;

        let plan = packaged_krea_plan();
        let generator = packaged_krea_generator();
        let mut inputs = fixture_inputs(1024, 1024);
        inputs.overlay = Some("control:1".to_owned());

        for (budget_gib, expected) in [(128.0, "896x896"), (83.0, "768x768")] {
            let error = evaluate_request_with_budget(
                &generator,
                &plan,
                &inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                fixture_budget(budget_gib),
                gib_to_bytes(130.0),
                0,
                &[],
            )
            .expect_err("the independent legacy estimate must refuse before provider render");
            let message = error.to_string();
            assert!(
                message.contains(&format!("current verified alternative: {expected}")),
                "the largest exact current cell that fits {budget_gib} GiB must be named: {message}"
            );
        }

        let mut exact_896 = inputs.clone();
        exact_896.width = 896;
        exact_896.height = 896;
        let message = evaluate_request_with_budget(
            &generator,
            &plan,
            &exact_896,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(83.0),
            gib_to_bytes(130.0),
            0,
            &[],
        )
        .expect_err("the exact 896 cell exceeds 83 GiB including its captured foreign reserve")
        .to_string();
        assert!(
            message.contains("current verified alternative: 768x768"),
            "a packaged exact-cell refusal must retain its fitting lower record: {message}"
        );

        let mut mismatched = packaged_krea_generator();
        mismatched
            .contract
            .as_mut()
            .expect("Krea contract")
            .calibration = Some(MemoryCalibrationIdentity::new("mutated-loaded-provider"));
        let message = evaluate_request_with_budget(
            &mismatched,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(128.0),
            gib_to_bytes(130.0),
            0,
            &[],
        )
        .expect_err("the mismatched loaded provider still refuses on the independent estimate")
        .to_string();
        assert!(
            !message.contains("current verified alternative"),
            "a loaded-provider fingerprint mutation must suppress evidence-derived naming: {message}"
        );
    }

    #[test]
    fn manifest_reader_distinguishes_absent_valid_and_malformed_opt_in() {
        let spec = fixture_spec(gen_core::Quant::Q4, "packed-q4");
        let absent = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &spec,
            Some(&JsonObject::new()),
            None,
        );
        assert!(matches!(absent.calibration, MlxCalibrationConfig::Absent));

        let valid_manifest = fixture_manifest(vec![fixture_calibration_json("q4", "packed-q4")]);
        let valid = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &spec,
            Some(&valid_manifest),
            Some(fixture_provenance("q4", "packed-q4")),
        );
        let MlxCalibrationConfig::Valid(calibration) = valid.calibration else {
            panic!("well-formed opt-in must be valid");
        };
        assert_eq!(calibration.bindings.len(), 1);
        assert_eq!(calibration.resolved.identity.variant, "packed-q4");
        assert_eq!(valid.tier.quant, Some(gen_core::Quant::Q4));

        let unavailable = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &spec,
            Some(&valid_manifest),
            None,
        );
        assert!(matches!(
            unavailable.calibration,
            MlxCalibrationConfig::Invalid(_)
        ));
        assert!(packaged_admission_route(
            &unavailable,
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(8.0)
        )
        .expect_err("present opt-in without trusted resolver provenance must fail closed")
        .to_string()
        .contains("resolver supplied no immutable"));

        let mut malformed = fixture_calibration_json("q4", "packed-q4");
        malformed
            .as_object_mut()
            .expect("calibration object")
            .remove("fingerprint");
        let malformed_manifest = fixture_manifest(vec![malformed]);
        let malformed = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &spec,
            Some(&malformed_manifest),
            Some(fixture_provenance("q4", "packed-q4")),
        );
        assert!(matches!(
            malformed.calibration,
            MlxCalibrationConfig::Invalid(_)
        ));
        assert!(packaged_admission_route(
            &malformed,
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(8.0)
        )
        .expect_err("a malformed present opt-in must not collapse to packaged-empty legacy")
        .to_string()
        .contains("invalid MLX calibration opt-in"));
    }

    #[test]
    fn parameter_reader_is_closed_and_preserves_transformer_component() {
        let parameters = JsonObject::from_iter([
            ("decodeTileEdge".to_owned(), serde_json::json!(512)),
            ("decodeOverlap".to_owned(), serde_json::json!(128)),
            ("attentionChunkSize".to_owned(), serde_json::json!(256)),
            ("transformerWindowSize".to_owned(), serde_json::json!(4)),
            (
                "transformerWindowComponent".to_owned(),
                serde_json::json!("both"),
            ),
        ]);
        let parsed =
            parse_evidence_parameters(StrategyRung::BoundedTransformerResidency, &parameters)
                .expect("the complete exact parameter set is valid");
        assert_eq!(parsed.decode_tile_edge, Some(512));
        assert_eq!(parsed.decode_overlap, Some(128));
        assert_eq!(parsed.attention_chunk_size, Some(256));
        assert_eq!(parsed.transformer_window_size, Some(4));
        assert_eq!(
            parsed.transformer_window_component,
            Some(TransformerComponent::Both)
        );

        let mut unknown = parameters.clone();
        unknown.insert("unrecognized".to_owned(), serde_json::json!(1));
        assert!(
            parse_evidence_parameters(StrategyRung::BoundedTransformerResidency, &unknown)
                .expect_err("unknown parameters fail closed")
                .contains("unknown")
        );

        let mut malformed = parameters.clone();
        malformed.insert(
            "transformerWindowComponent".to_owned(),
            serde_json::json!(12),
        );
        assert!(
            parse_evidence_parameters(StrategyRung::BoundedTransformerResidency, &malformed)
                .expect_err("a non-string transformer component fails closed")
                .contains("transformerWindowComponent")
        );

        let mut unsupported = parameters;
        unsupported.insert(
            "transformerWindowComponent".to_owned(),
            serde_json::json!("vae"),
        );
        assert!(
            parse_evidence_parameters(StrategyRung::BoundedTransformerResidency, &unsupported)
                .expect_err("an unknown transformer component fails closed")
                .contains("unsupported")
        );
    }

    #[test]
    fn fallback_reason_priority_is_independent_of_binding_order() {
        let fold = |reasons: &[LegacyAdmissionReason]| {
            reasons
                .iter()
                .copied()
                .fold(LegacyAdmissionReason::NoRecord, stronger_fallback_reason)
        };
        let forward = [
            LegacyAdmissionReason::OutOfEnvelope,
            LegacyAdmissionReason::StaleFingerprint,
            LegacyAdmissionReason::StaleIdentity,
        ];
        let mut reversed = forward;
        reversed.reverse();
        assert_eq!(
            fold(&forward),
            LegacyAdmissionReason::StaleIdentity,
            "the strongest drift reason wins"
        );
        assert_eq!(
            fold(&forward),
            fold(&reversed),
            "reversing calibration binding order must not change the fallback reason"
        );
    }

    #[test]
    fn calibration_collection_supports_tiers_and_cells_but_rejects_duplicate_selectors() {
        let mut second_cell = fixture_calibration_json("q4", "packed-q4");
        second_cell["geometry"]["width"] = serde_json::json!(512);
        second_cell["geometry"]["height"] = serde_json::json!(512);
        let manifest = fixture_manifest(vec![
            fixture_calibration_json("q4", "packed-q4"),
            second_cell,
            fixture_calibration_json("q8", "packed-q8"),
        ]);
        let bindings = MlxCalibrationBinding::from_manifest(&manifest)
            .expect("valid collection")
            .expect("present collection");
        assert_eq!(bindings.len(), 3);
        assert_eq!(
            bindings
                .iter()
                .filter(|binding| binding.tier == "q4")
                .count(),
            2,
            "one tier may contain multiple exact request cells"
        );
        assert!(bindings.iter().any(|binding| binding.tier == "q8"));

        let duplicate = fixture_manifest(vec![
            fixture_calibration_json("q4", "packed-q4"),
            fixture_calibration_json("q4", "packed-q4"),
        ]);
        assert!(MlxCalibrationBinding::from_manifest(&duplicate)
            .expect_err("duplicate request selectors are ambiguous")
            .contains("ambiguous duplicates"));
        let mut resident = fixture_calibration_json("q4", "packed-q4");
        resident["rung"] = serde_json::json!("resident");
        resident["parameters"] = serde_json::json!({});
        let ladder = fixture_manifest(vec![fixture_calibration_json("q4", "packed-q4"), resident]);
        assert_eq!(
            MlxCalibrationBinding::from_manifest(&ladder)
                .expect("distinct rungs in one cell are a ladder, not duplicates")
                .expect("present ladder")
                .len(),
            2
        );

        let mut bundle = fixture_bundle();
        let mut second_cell_record = bundle.records[0].clone();
        second_cell_record.target.geometry.width = 512;
        second_cell_record.target.geometry.height = 512;
        bundle.records.push(second_cell_record);

        let q4_plan = MlxRequestPlan {
            calibration: MlxCalibrationConfig::Valid(MlxCalibrationSet {
                bindings,
                resolved: fixture_provenance("q4", "packed-q4"),
            }),
            ..fixture_plan()
        };
        assert_eq!(
            evidence_admission_route(
                &bundle,
                &q4_plan,
                &fixture_inputs(512, 512),
                "text_to_image",
                fixture_budget(8.0)
            )
            .expect("the second exact cell is independently selectable")
            .path,
            AdmissionPath::Evidence
        );
    }

    #[test]
    fn load_spec_tier_and_resolved_artifact_variant_are_independent() {
        let manifest = fixture_manifest(vec![
            fixture_calibration_json("q4", "packed-q4"),
            fixture_calibration_json("q8", "packed-q8"),
        ]);
        let q4 = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &fixture_spec(gen_core::Quant::Q4, "packed-q4"),
            Some(&manifest),
            Some(fixture_provenance("q4", "packed-q4")),
        );
        let MlxCalibrationConfig::Valid(q4_calibration) = &q4.calibration else {
            panic!("q4 binding");
        };
        assert_eq!(q4.tier.quant, Some(gen_core::Quant::Q4));
        assert_eq!(q4_calibration.resolved.identity.variant, "packed-q4");
        assert_eq!(
            evidence_admission_route(
                &fixture_bundle(),
                &q4,
                &fixture_inputs(1024, 1024),
                "text_to_image",
                fixture_budget(8.0)
            )
            .expect("q4 packed artifact is independently verified")
            .path,
            AdmissionPath::Evidence
        );

        let dense_load_q4 = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &LoadSpec::new(WeightsSource::Dir("/resolved/packed-q4".into())),
            Some(&manifest),
            Some(fixture_provenance("q4", "packed-q4")),
        );
        assert_eq!(
            dense_load_q4.tier.quant,
            Some(gen_core::Quant::Q4),
            "resolver tier must win when a packed transformer keeps LoadSpec.quantize=None"
        );
        let relabeled_packed_q4 = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &fixture_spec(gen_core::Quant::Q8, "packed-q4"),
            Some(&manifest),
            Some(fixture_provenance("q4", "packed-q4")),
        );
        assert!(
            matches!(
                relabeled_packed_q4.calibration,
                MlxCalibrationConfig::Invalid(_)
            ),
            "request quantization must not relabel a fixed packed-q4 artifact as q8"
        );

        let wrong_variant = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &fixture_spec(gen_core::Quant::Q4, "different-packed-q4"),
            Some(&manifest),
            Some(fixture_provenance("q4", "different-packed-q4")),
        );
        assert_eq!(
            evidence_admission_route(
                &fixture_bundle(),
                &wrong_variant,
                &fixture_inputs(1024, 1024),
                "text_to_image",
                fixture_budget(8.0)
            )
            .expect("unverified artifact variant uses legacy")
            .fallback_reason,
            Some(LegacyAdmissionReason::StaleIdentity)
        );

        let q8 = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &fixture_spec(gen_core::Quant::Q8, "packed-q8"),
            Some(&manifest),
            Some(fixture_provenance("q8", "packed-q8")),
        );
        let MlxCalibrationConfig::Valid(q8_calibration) = &q8.calibration else {
            panic!("q8 binding");
        };
        assert_eq!(q8.tier.quant, Some(gen_core::Quant::Q8));
        assert_eq!(q8_calibration.resolved.identity.variant, "packed-q8");

        let dense_load_q8 = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &LoadSpec::new(WeightsSource::Dir("/resolved/packed-q8".into())),
            Some(&manifest),
            Some(fixture_provenance("q8", "packed-q8")),
        );
        assert_eq!(
            dense_load_q8.tier.quant,
            Some(gen_core::Quant::Q8),
            "resolved packed q8 must not be mislabeled bf16 by a dense text-encoder LoadSpec"
        );

        let mut q8_bundle = fixture_bundle();
        let mut q8_record = q8_bundle.records[0].clone();
        q8_record.target.tier = "q8".to_owned();
        q8_record.artifact.variant = "packed-q8".to_owned();
        q8_record.loadability.resolved_path_fingerprint =
            sceneworks_core::memory_calibration::RequiredNullable::Value(format!(
                "SceneWorks/fixture@{}:packed-q8",
                "c".repeat(40)
            ));
        q8_bundle.records.push(q8_record);
        assert_eq!(
            evidence_admission_route(
                &q8_bundle,
                &q8,
                &fixture_inputs(1024, 1024),
                "text_to_image",
                fixture_budget(8.0)
            )
            .expect("q8 record is selected independently of q4")
            .path,
            AdmissionPath::Evidence
        );
    }

    #[tokio::test]
    async fn flat_default_receipt_keeps_identity_orthogonal_to_q4_q8_execution() {
        let data = tempfile::tempdir().expect("data dir");
        let hub = data.path().join("hub");
        let _env =
            crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().expect("hub path"))]);
        let repo = "SceneWorks/fixture";
        let revision = "c".repeat(40);
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(data.path(), repo)
            .expect("cache")
            .join("snapshots")
            .join(&revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        std::fs::write(snapshot.join("weights.safetensors"), b"flat-q8").expect("weights");
        let resolved_files = vec!["weights.safetensors".to_owned()];
        let stamp = crate::model_jobs::resolved_files_tree_stamp(&snapshot, &resolved_files)
            .expect("tree stamp");
        let payload = JsonObject::from_iter([
            ("modelId".to_owned(), serde_json::json!("fixture_model")),
            ("variant".to_owned(), serde_json::json!("default")),
            ("mlx".to_owned(), serde_json::json!({ "quantize": 8 })),
        ]);
        let resolved_tier = crate::model_jobs::download_payload_resolved_tier(&payload);
        assert_eq!(
            resolved_tier, None,
            "a flat dense artifact is not permanently labeled by its default runtime quantization"
        );
        let marker_dir = data
            .path()
            .join("models")
            .join(crate::paths::safe_download_dir("fixture_model"));
        crate::write_model_download_receipt(
            &marker_dir,
            &payload,
            repo,
            "job-flat-default",
            &resolved_files,
            Some(&revision),
            crate::imports::DownloadArtifactReceipt {
                resolved_tier: resolved_tier.as_deref(),
                tree_stamp: Some(&stamp),
            },
        )
        .await
        .expect("real receipt writer");
        let resolved = crate::model_jobs::huggingface_receipt_weights(
            data.path(),
            repo,
            Some("fixture_model"),
            Some("default"),
        )
        .expect("receipt resolver");
        assert_eq!(resolved.path, snapshot);
        let provenance = resolved.provenance.expect("trusted provenance");
        assert_eq!(provenance.fixed_artifact_tier.as_deref(), None);

        let calibration = |tier| {
            let mut calibration = fixture_calibration_json(tier, "default");
            calibration["artifactResolvedRevision"] = serde_json::json!(revision);
            calibration["artifactVariant"] = serde_json::json!("default");
            calibration["resolvedPathFingerprint"] =
                serde_json::json!(provenance.identity.fingerprint);
            calibration
        };
        let manifest = fixture_manifest(vec![calibration("q4"), calibration("q8")]);
        for quant in [gen_core::Quant::Q4, gen_core::Quant::Q8] {
            let plan = MlxRequestPlan::for_spec_and_manifest(
                "fixture_provider",
                "fixture_model",
                &LoadSpec::new(WeightsSource::Dir(resolved.path.clone())).with_quant(quant),
                Some(&manifest),
                Some(provenance.clone()),
            );
            assert_eq!(plan.tier.quant, Some(quant));
            assert!(matches!(plan.calibration, MlxCalibrationConfig::Valid(_)));
        }
    }

    #[test]
    fn decision_two_transition_paths_are_explicit_and_fail_closed_only_when_covered() {
        let bundle = fixture_bundle();
        let inputs = fixture_inputs(1024, 1024);

        let mut uncalibrated = fixture_plan();
        uncalibrated.calibration = MlxCalibrationConfig::Absent;
        let route = evidence_admission_route(
            &bundle,
            &uncalibrated,
            &inputs,
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("an uncalibrated model uses legacy");
        assert_eq!(route.path, AdmissionPath::Legacy);
        assert_eq!(
            route.fallback_reason,
            Some(LegacyAdmissionReason::NoBinding)
        );

        let uncovered = evidence_admission_route(
            &bundle,
            &fixture_plan(),
            &fixture_inputs(768, 768),
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("an uncovered geometry uses legacy");
        assert_eq!(uncovered.path, AdmissionPath::Legacy);
        assert_eq!(
            uncovered.fallback_reason,
            Some(LegacyAdmissionReason::OutOfEnvelope)
        );

        let mut drifted = fixture_plan();
        let MlxCalibrationConfig::Valid(calibration) = &mut drifted.calibration else {
            panic!("fixture calibration");
        };
        calibration.bindings[0].query.fingerprint = "different".to_owned();
        let drifted = evidence_admission_route(
            &bundle,
            &drifted,
            &inputs,
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("fingerprint drift uses legacy");
        assert_eq!(drifted.path, AdmissionPath::Legacy);
        assert_eq!(
            drifted.fallback_reason,
            Some(LegacyAdmissionReason::StaleFingerprint)
        );

        let covered = evidence_admission_route(
            &bundle,
            &fixture_plan(),
            &inputs,
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("the exact covered cell fits its captured safe envelope");
        assert_eq!(covered.path, AdmissionPath::Evidence);
        assert_eq!(
            covered.process_limit_bytes,
            None,
            "the process ceiling is candidate-specific and cannot be chosen before strategy selection"
        );
        assert_eq!(
            budget_for_admission(fixture_budget(8.0), &covered).reserved_headroom_bytes,
            0,
            "no sibling candidate reserve may be imposed before strategy selection"
        );
        let evidence = covered
            .evidence
            .first()
            .expect("verified selector evidence")
            .evidence
            .clone();
        assert_eq!(evidence.predicted_peak_bytes, gib_to_bytes(5.0));
        assert_eq!(
            evidence.observed_peak_bytes,
            Some(gib_to_bytes(4.0)),
            "observed telemetry remains the measured counter rather than the predicted maximum"
        );
        let evaluated = evaluate_request_with_budget_using_bundle(
            &fixture_generator(),
            &fixture_plan(),
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(8.0),
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
        )
        .expect("selected exact candidate");
        assert_eq!(evaluated.process_limit_bytes, Some(gib_to_bytes(5.0)));
        assert_eq!(
            evaluated.context.budget.reserved_headroom_bytes,
            gib_to_bytes(3.0)
        );

        let unfit = evaluate_request_with_budget_using_bundle(
            &fixture_generator(),
            &fixture_plan(),
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(6.0),
            gib_to_bytes(5.0),
            0,
            &[],
            Some(&bundle),
        )
        .expect_err("the exact covered 5 GiB cell must reject when only 3 GiB is safely available");
        assert!(unfit
            .to_string()
            .contains("smallest verified MLX host boundary"));
    }

    #[test]
    fn target_tier_and_artifact_variant_are_independent_identity_mutations() {
        let bundle = fixture_bundle();
        let inputs = fixture_inputs(1024, 1024);

        let mut wrong_tier = fixture_plan();
        wrong_tier.tier.quant = Some(gen_core::Quant::Q8);
        let MlxCalibrationConfig::Valid(calibration) = &mut wrong_tier.calibration else {
            panic!("fixture calibration");
        };
        calibration.resolved.fixed_artifact_tier = Some("q8".to_owned());
        let wrong_tier = evidence_admission_route(
            &bundle,
            &wrong_tier,
            &inputs,
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("target-tier mismatch falls back");
        assert_eq!(
            wrong_tier.fallback_reason,
            Some(LegacyAdmissionReason::StaleIdentity)
        );

        let mut wrong_artifact = fixture_plan();
        let MlxCalibrationConfig::Valid(calibration) = &mut wrong_artifact.calibration else {
            panic!("fixture calibration");
        };
        calibration.resolved.identity.variant = "different-packed-q4".to_owned();
        calibration.resolved.identity.fingerprint =
            format!("SceneWorks/fixture@{}:different-packed-q4", "c".repeat(40));
        let wrong_artifact = evidence_admission_route(
            &bundle,
            &wrong_artifact,
            &inputs,
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("artifact mismatch falls back as drift");
        assert_eq!(
            wrong_artifact.fallback_reason,
            Some(LegacyAdmissionReason::StaleIdentity)
        );

        let mut wrong_provider = fixture_plan();
        let MlxCalibrationConfig::Valid(calibration) = &mut wrong_provider.calibration else {
            panic!("fixture calibration");
        };
        calibration.bindings[0].provider = "fixture_model".to_owned();
        let wrong_provider = evidence_admission_route(
            &bundle,
            &wrong_provider,
            &inputs,
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("provider mismatch falls back as drift");
        assert_eq!(
            wrong_provider.fallback_reason,
            Some(LegacyAdmissionReason::StaleIdentity),
            "the binding provider must match the actual engine route, not the catalog model id"
        );
    }

    #[test]
    fn covered_cell_selects_exact_strategy_cold_and_warm_without_resident_bypass() {
        let bundle = fixture_bundle();
        let generator = fixture_generator();
        let plan = fixture_plan();
        let inputs = fixture_inputs(1024, 1024);
        let expected = MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: fixture_binding("q4", "packed-q4").selection_parameters,
            tier: plan.tier,
        };
        let exact = evidence_admission_route(
            &bundle,
            &plan,
            &inputs,
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("covered route")
        .evidence
        .into_iter()
        .next()
        .expect("exact evidence")
        .evidence;
        assert!(
            exact.validation_errors().is_empty(),
            "fixture evidence errors: {:?}",
            exact.validation_errors()
        );
        let contract = generator.contract.as_ref().expect("fixture contract");
        assert!(
            contract.validate_selection(&expected).is_ok(),
            "fixture contract rejection: {:?}",
            contract.validate_selection(&expected)
        );
        let evaluate = |cache_state, committed_bytes| {
            evaluate_request_with_budget_using_bundle(
                &generator,
                &plan,
                &inputs,
                cache_state,
                OffloadPolicy::Resident,
                MemoryBudget {
                    committed_bytes,
                    ..fixture_budget(8.0)
                },
                gib_to_bytes(4.0),
                0,
                &[],
                Some(&bundle),
            )
            .expect("the exact covered request must select its verified rung")
        };
        let cold = evaluate(MemoryCacheState::Cold, 0);
        let warm = evaluate(MemoryCacheState::Warm, gib_to_bytes(3.0));
        assert_eq!(cold.context.selection, expected);
        assert_eq!(warm.context.selection, expected);
        assert!(cold.memory.tile_vae_decode && warm.memory.tile_vae_decode);
        assert_eq!(cold.context.predicted_peak_bytes, gib_to_bytes(5.0));
        assert_eq!(
            warm.context.predicted_peak_bytes,
            cold.context.predicted_peak_bytes,
            "warm attribution credits resident assets in the budget without rewriting exact evidence"
        );
        assert_eq!(
            warm.context.budget.committed_bytes, 0,
            "the provider's three resident GiB are credited exactly once"
        );
    }

    #[test]
    fn verified_ladder_selects_resident_decode_then_attention_as_budget_tightens() {
        let (bundle, plan) = fixture_ladder();
        let generator = fixture_generator();
        let inputs = fixture_inputs(1024, 1024);
        for (total_gib, expected) in [
            (10.0, MemoryStrategy::Resident),
            (8.0, MemoryStrategy::BoundedDecode),
            (7.0, MemoryStrategy::BoundedAttention),
        ] {
            let evaluation = evaluate_request_with_budget_using_bundle(
                &generator,
                &plan,
                &inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                fixture_budget(total_gib),
                gib_to_bytes(4.0),
                0,
                &[],
                Some(&bundle),
            )
            .unwrap_or_else(|error| panic!("{total_gib} GiB ladder failed: {error}"));
            assert_eq!(evaluation.context.selection.strategy, expected);
            assert!(
                bundle
                    .records
                    .iter()
                    .any(|record| record.id == evaluation.context.evidence_revision),
                "telemetry must name the selected evidence record, not the whole candidate list"
            );
        }
    }

    #[test]
    fn candidate_specific_foreign_reserve_does_not_block_a_fitting_lower_rung() {
        use sceneworks_core::memory_calibration::Hardware;

        let (mut bundle, plan) = fixture_ladder();
        let set_reserve = |record: &mut sceneworks_core::memory_calibration::EvidenceRecord,
                           reserve_gib: f64| {
            let Hardware::Mlx(hardware) = &mut record.hardware else {
                panic!("MLX fixture hardware");
            };
            hardware.memory_bytes = gib_to_bytes(10.0);
            hardware.mlx_memory_limit_bytes = gib_to_bytes(10.0 - reserve_gib);
            hardware.wired_limit_bytes = hardware.memory_bytes;
        };
        set_reserve(&mut bundle.records[0], 4.0);
        set_reserve(&mut bundle.records[1], 2.0);
        set_reserve(&mut bundle.records[2], 2.0);

        let evaluation = evaluate_request_with_budget_using_bundle(
            &fixture_generator(),
            &plan,
            &fixture_inputs(1024, 1024),
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(10.0),
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
        )
        .expect("the bounded-decode candidate's own 5+2 GiB boundary fits");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::BoundedDecode,
            "resident 7+4 GiB must be excluded without imposing its reserve on bounded decode"
        );
        assert_eq!(
            evaluation.process_limit_bytes,
            Some(gib_to_bytes(8.0)),
            "request ceiling comes from the selected candidate's 2 GiB reserve"
        );
    }

    #[test]
    fn packaged_bundle_without_an_exact_record_is_a_normal_legacy_reason_not_drift() {
        let route = packaged_admission_route(
            &fixture_plan(),
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(8.0),
        )
        .expect("the promoted bundle is valid even when this fixture has no exact record");
        assert_eq!(route.path, AdmissionPath::Legacy);
        assert_eq!(route.fallback_reason, Some(LegacyAdmissionReason::NoRecord));
    }

    /// SC-15805: `memory_for_selection` is the live MLX memory-admission seam — the
    /// [`GenerationMemory`] it builds reaches the engine via `image_jobs/base.rs`. It asks the
    /// CONTRACT which rungs a selection engages instead of re-deriving the answer from
    /// `MemoryStrategy`'s numeric order, and that difference is only observable when a provider
    /// declares a cheaper rung unavailable.
    ///
    /// Every other test in this module uses [`mage_request_contract`], whose optimized rungs are all
    /// `Missing`, so it never selects past `Resident` and never sets these knobs at all — a test that
    /// passes because a field was never set is a false green. This one sets the field.
    ///
    /// Reverting `memory_for_selection` to `selection.strategy >= MemoryStrategy::BoundedDecode`
    /// (etc.) must turn this RED.
    #[test]
    fn an_unimplemented_rung_is_not_engaged_by_a_deeper_selection() {
        use gen_core::{MemoryParameterRanges, MemoryStrategyCapability, MemoryStrategySupport};

        let contract_with = |missing: MemoryStrategy| {
            let mut contract = mage_request_contract();
            contract.strategies = MemoryStrategy::ALL
                .into_iter()
                .map(|strategy| MemoryStrategyCapability {
                    strategy,
                    support: if strategy == missing {
                        MemoryStrategySupport::Missing
                    } else {
                        MemoryStrategySupport::Implemented
                    },
                    parameters: MemoryParameterRanges::default(),
                })
                .collect();
            contract
        };
        let deepest = MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: Default::default(),
            tier: request_plan().tier,
        };

        // Rung 2 unavailable: the deepest selection must not tile the decode.
        let memory = memory_for_selection(&contract_with(MemoryStrategy::BoundedDecode), deepest);
        assert!(
            !memory.tile_vae_decode,
            "BoundedDecode is declared Missing, so a BoundedTransformerResidency selection must \
             not switch decode tiling on underneath the provider: the ladder's numeric order is a \
             COST ordering, not a dependency"
        );
        // ...and the rungs the provider does declare are still engaged, so this is not the vacuous
        // "everything is false" green that would also pass if the function returned Default.
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);

        // Rung 3 unavailable: same property, one knob over.
        let memory =
            memory_for_selection(&contract_with(MemoryStrategy::BoundedAttention), deepest);
        assert!(
            !memory.chunk_attention,
            "BoundedAttention is declared Missing, so a deeper selection must not chunk attention"
        );
        assert!(memory.tile_vae_decode);
        assert!(memory.stream_transformer_blocks);

        // Positive control: with every rung implemented, the cumulative default still applies in
        // full. Without this, a `memory_for_selection` that returned `Default` unconditionally would
        // satisfy both assertions above.
        let all_implemented = {
            let mut contract = mage_request_contract();
            contract.strategies = MemoryStrategy::ALL
                .into_iter()
                .map(|strategy| MemoryStrategyCapability {
                    strategy,
                    support: MemoryStrategySupport::Implemented,
                    parameters: MemoryParameterRanges::default(),
                })
                .collect();
            contract
        };
        let memory = memory_for_selection(&all_implemented, deepest);
        assert!(
            !memory.stage_residency,
            "rung 4 must not evict the warm cache by implicitly engaging rung 1"
        );
        assert!(memory.tile_vae_decode);
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);

        let staged = memory_for_selection(
            &all_implemented,
            MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: request_plan().tier,
            },
        );
        assert!(
            staged.stage_residency,
            "an explicit rung-1 selection must reach GenerationMemory"
        );
        assert!(!staged.tile_vae_decode);
        assert!(!staged.chunk_attention);
        assert!(!staged.stream_transformer_blocks);
    }

    #[test]
    fn request_scope_reselects_a_b_a_without_fragmenting_request_axes() {
        let generator = request_generator(Some(mage_request_contract()));
        let plan = request_plan();
        let budget = MemoryBudget {
            total_bytes: gib_to_bytes(16.0),
            committed_bytes: gib_to_bytes(4.0),
            reclaimable_bytes: 0,
            reserved_headroom_bytes: gib_to_bytes(2.0),
        };
        let a = request_inputs(512, 512, 3);
        let b = request_inputs(1536, 1024, 3);
        let first_a = evaluate_request_with_budget(
            &generator,
            &plan,
            &a,
            MemoryCacheState::Cold,
            OffloadPolicy::Sequential,
            budget,
            gib_to_bytes(8.0),
            0,
            &[],
        )
        .unwrap();
        let selected_b = evaluate_request_with_budget(
            &generator,
            &plan,
            &b,
            MemoryCacheState::Warm,
            OffloadPolicy::Sequential,
            budget,
            gib_to_bytes(9.0),
            0,
            &[],
        )
        .unwrap();
        let second_a = evaluate_request_with_budget(
            &generator,
            &plan,
            &a,
            MemoryCacheState::Warm,
            OffloadPolicy::Sequential,
            budget,
            gib_to_bytes(8.0),
            0,
            &[],
        )
        .unwrap();

        // A 3-image job is three sequential forward passes, so the modeled batch stays 1 even though
        // `request_inputs` carried count 3 (see `request_batch`).
        assert_eq!(first_a.context.geometry.batch, 1);
        assert_eq!(first_a.context.mode, MemoryMode::Edit);
        assert_eq!(
            first_a.context.overlay.as_deref(),
            Some("references:2+mask+adapters:1")
        );
        assert_eq!(first_a.context.cache_state, MemoryCacheState::Cold);
        assert_eq!(selected_b.context.geometry.width, 1536);
        assert_eq!(selected_b.context.cache_state, MemoryCacheState::Warm);
        assert_eq!(first_a.context.selection, second_a.context.selection);
        assert_eq!(
            first_a.context.predicted_peak_bytes, second_a.context.predicted_peak_bytes,
            "the intervening geometry cannot poison a warm follow-up"
        );
    }

    /// A multi-image job is a sequential loop with an MLX cache release between items (sc-5567), so
    /// its peak is one image's working set — the estimator must be INVARIANT in the job count.
    ///
    /// The numbers are the real reported rejection: krea_2_turbo bf16 (33.22 GiB of safetensors) at
    /// 1152x2048 count 4 on a 128 GiB Mac (126.00 GiB after the 2 GiB legacy unified reserve). The
    /// count multiplier quoted 33.22 + 16 x 2.25 x 4 = 177.22 GiB — the activation term alone
    /// exceeded the whole budget, so this cell rejected every model at every tier.
    #[test]
    fn generic_request_peak_is_invariant_in_the_job_image_count() {
        let plan = MlxRequestPlan {
            engine_id: "krea_2_turbo",
            model_id: "krea_2_turbo".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
            },
            asset_bytes: 35_666_644_396,
            activation_headroom_bytes: gib_to_bytes(HEADROOM_GB - 2.0),
            fixed_reserve_bytes: gib_to_bytes(OS_APP_RESERVE_GB - 2.0),
            calibration: MlxCalibrationConfig::Absent,
        };
        // Go through `request_geometry` rather than hand-building a `batch: 1` geometry, so this
        // exercises the production count -> batch seam instead of asserting a value it supplies.
        let peak = |count: u32| {
            plan.generic_total_peak_bytes(request_geometry(&request_inputs(1152, 2048, count)))
        };

        let single = peak(1);
        for count in [2, 4, 8] {
            assert_eq!(
                peak(count),
                single,
                "count {count} is {count} sequential passes, not a batched one"
            );
        }

        let budget = gib_to_bytes(126.0);
        assert!(
            peak(4) <= budget,
            "the reported cell must admit: needed {:.2} GiB vs {:.2} GiB available",
            peak(4) as f64 / BYTES_PER_GIB,
            budget as f64 / BYTES_PER_GIB
        );

        // Non-constant control: resolution is still a real axis, so an estimator that stopped
        // scaling entirely would not pass by accident.
        assert!(
            peak(1) > plan.generic_total_peak_bytes(request_geometry(&request_inputs(512, 512, 1))),
            "the estimator must still grow with output resolution"
        );
    }

    /// sc-16195: the OS/app reserve inside the flat headroom is FIXED, so only the activation term
    /// may scale with output area.
    ///
    /// Every number here is pinned in absolute GiB rather than re-derived from the formula, so the
    /// test fails if the split moves — a test that recomputed `fixed + area * mp` would pass against
    /// any split, including the broken one.
    #[test]
    fn generic_request_peak_scales_the_area_term_but_never_the_os_reserve() {
        let asset_gb = 33.22_f64;
        let plan = MlxRequestPlan {
            engine_id: "krea_2_turbo",
            model_id: "krea_2_turbo".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
            },
            asset_bytes: gib_to_bytes(asset_gb),
            activation_headroom_bytes: gib_to_bytes(HEADROOM_GB - 2.0),
            fixed_reserve_bytes: gib_to_bytes(OS_APP_RESERVE_GB - 2.0),
            calibration: MlxCalibrationConfig::Absent,
        };
        let peak_gb = |width, height| {
            plan.generic_total_peak_bytes(request_geometry(&request_inputs(width, height, 1)))
                as f64
                / BYTES_PER_GIB
        };
        // The split: 16 GiB of allowance = a 2 GiB fixed remainder of the 4 GiB OS/app reserve (the
        // other 2 are carried separately as the legacy unified reserve) + a 14 GiB area term.
        let close = |actual: f64, expected: f64| (actual - expected).abs() < 1e-6;

        // 1024² is the calibration anchor and must be BIT-IDENTICAL to the pre-sc-16195 estimator —
        // the sweep re-derived the shape, it did not re-cut the safety margin.
        assert!(
            close(peak_gb(1024, 1024), asset_gb + 16.0),
            "1024² must be unchanged at asset + 16: got {:.4}",
            peak_gb(1024, 1024)
        );

        // Above 1024² only the 14 GiB area term scales. The old estimator scaled all 16.
        for (width, height, megapixels) in
            [(1024, 1536, 1.5), (1152, 2048, 2.25), (2048, 2048, 4.0)]
        {
            let expected = asset_gb + 2.0 + 14.0 * megapixels;
            assert!(
                close(peak_gb(width, height), expected),
                "{width}x{height}: expected asset + 2 + 14*{megapixels} = {expected:.4}, got {:.4}",
                peak_gb(width, height)
            );
            // Mutation guard: the pre-sc-16195 formula scaled the whole 16, so it is strictly
            // larger everywhere above the anchor. If the reserve ever starts scaling again, this
            // fires even if the arithmetic above were loosened.
            assert!(
                peak_gb(width, height) < asset_gb + 16.0 * megapixels,
                "{width}x{height} must model below the old whole-headroom scaling"
            );
        }

        // Below 1024² the scale is floored at 1.0: the measured transient stops falling off
        // proportionally down there (illustrious 0.305x and qwen 0.512x of their anchors at 0.25x
        // area, both above proportional), so the floor is the conservative reading.
        assert!(
            close(peak_gb(512, 512), asset_gb + 16.0),
            "sub-anchor requests stay floored at the 1024² allowance: got {:.4}",
            peak_gb(512, 512)
        );
    }

    /// sc-16195: a family whose allowance was measured as a BARE activation transient carries no OS
    /// reserve to hold out, so the split must be an exact no-op for it.
    ///
    /// This is the lens dense path ([`HeadroomAllowance::LENS_DENSE`], sc-11924). Holding a 2 GiB
    /// reserve out of its 27.88 GiB allowance would move 2 GiB from the AREA term into a constant —
    /// `2 + 25.88·MP` instead of `27.88·MP` — which is strictly LESS conservative above 1024², by
    /// `2·(MP−1)`. On a path sc-11924 already records as under-predicting, and whose permissive-side
    /// failure mode is an OS Jetsam SIGKILL, that would be a regression rather than a refinement.
    #[test]
    fn a_bare_transient_allowance_keeps_its_whole_area_term() {
        let asset_gb = 28.43_f64;
        let plan = |headroom: HeadroomAllowance| MlxRequestPlan {
            engine_id: "lens_turbo",
            model_id: "lens_turbo".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
            },
            asset_bytes: gib_to_bytes(asset_gb),
            activation_headroom_bytes: gib_to_bytes(
                headroom.total_gb - crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB,
            ),
            fixed_reserve_bytes: gib_to_bytes(
                (headroom.os_reserve_gb - crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB)
                    .max(0.0),
            ),
            calibration: MlxCalibrationConfig::Absent,
        };
        let dense = plan(HeadroomAllowance::LENS_DENSE);
        let peak_gb = |plan: &MlxRequestPlan, width, height| {
            plan.generic_total_peak_bytes(request_geometry(&request_inputs(width, height, 1)))
                as f64
                / BYTES_PER_GIB
        };
        // LENS_DENSE_HEADROOM_GB 29.88 − the 2 GiB legacy reserve budgeting carries separately.
        let allowance = LENS_DENSE_HEADROOM_GB - 2.0;
        for (width, height, megapixels) in [
            (1024, 1024, 1.0),
            (1024, 1536, 1.5),
            (1152, 2048, 2.25),
            (2048, 2048, 4.0),
        ] {
            let expected = asset_gb + allowance * megapixels;
            assert!(
                (peak_gb(&dense, width, height) - expected).abs() < 1e-6,
                "{width}x{height}: a bare-transient allowance must scale WHOLLY with area — \
                 expected {expected:.4}, got {:.4}",
                peak_gb(&dense, width, height)
            );
        }

        // Control: the generic allowance, which really does contain a reserve, must NOT behave that
        // way — otherwise this test would pass against a build that had removed the split entirely.
        let generic = plan(HeadroomAllowance::GENERIC);
        let generic_whole = asset_gb + (HEADROOM_GB - 2.0) * 2.25;
        assert!(
            peak_gb(&generic, 1152, 2048) < generic_whole - 1e-6,
            "the generic allowance must still hold its reserve out of the area term"
        );
    }

    #[test]
    fn request_scope_charges_external_committed_memory_and_never_accepts_zero_zero() {
        let generator = request_generator(Some(mage_request_contract()));
        let plan = request_plan();
        let inputs = request_inputs(512, 512, 1);
        let external = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Warm,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(10.0),
                committed_bytes: gib_to_bytes(8.0),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(8.0),
            gib_to_bytes(7.0),
            &[],
        );
        assert!(
            external.is_err(),
            "seven GiB of unrelated active memory must remain charged"
        );

        let overcommitted = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Warm,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(10.0),
                committed_bytes: gib_to_bytes(12.0),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(8.0),
            0,
            &[],
        );
        assert!(
            overcommitted.is_err(),
            "an overcommitted process must not pass through a saturated 0 <= 0 comparison"
        );
    }

    #[test]
    fn legacy_provider_credits_only_the_known_generator_assets_on_a_warm_run() {
        let plan = MlxRequestPlan {
            engine_id: "flux_dev",
            model_id: "flux_dev".to_owned(),
            tier: request_plan().tier,
            asset_bytes: gib_to_bytes(6.0),
            // Deliberately ABOVE the 2 GiB fixed reserve so the area term is non-zero: a fixture
            // sitting exactly on the reserve would model resolution-blind and silently stop
            // exercising the sc-16195 scaling at all.
            activation_headroom_bytes: gib_to_bytes(6.0),
            fixed_reserve_bytes: gib_to_bytes(2.0),
            calibration: MlxCalibrationConfig::Absent,
        };
        let selected = evaluate_request_with_budget(
            &request_generator(None),
            &plan,
            &request_inputs(512, 512, 1),
            MemoryCacheState::Warm,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(10.0),
                committed_bytes: gib_to_bytes(6.0),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(8.0),
            gib_to_bytes(1.0),
            &[],
        )
        .unwrap();

        assert_eq!(
            selected.context.predicted_peak_bytes,
            gib_to_bytes(3.0),
            "five attributable GiB are already resident; the unrelated GiB stays charged"
        );
    }

    #[test]
    fn post_load_external_delta_never_reclassifies_or_subtracts_resident_bytes() {
        assert_eq!(add_post_load_external_delta(10, 100, 125), 35);
        assert_eq!(
            add_post_load_external_delta(10, 125, 100),
            10,
            "allocator cleanup cannot reduce the cache-recorded external baseline"
        );
        assert_eq!(add_post_load_external_delta(u64::MAX - 1, 0, 10), u64::MAX);
    }

    #[test]
    fn mage_resident_path_requires_the_provider_calibration_handshake() {
        let error = evaluate_request_with_budget(
            &request_generator(None),
            &request_plan(),
            &request_inputs(512, 512, 1),
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(16.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(8.0),
            0,
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(MAGE_CALIBRATION_FINGERPRINT),
            "the production Resident path must compare the loaded provider fingerprint: {error}"
        );
    }

    #[test]
    fn mage_adapter_requests_fail_closed_outside_the_paired_calibration() {
        let mut inputs = request_inputs(512, 512, 1);
        inputs.adapter_count = 1;
        let error = evaluate_request_with_budget(
            &request_generator(Some(mage_request_contract())),
            &request_plan(),
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(128.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(8.0),
            0,
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("does not include LoRA/LoKr tensors"));
    }

    #[test]
    fn optimized_fingerprint_mismatch_fails_closed() {
        use gen_core::{
            MemoryCalibrationIdentity, MemoryParameterRanges, MemoryStrategyCapability,
            MemoryStrategyParameters, MemoryStrategySupport,
        };

        let plan = request_plan();
        let inputs = request_inputs(1024, 1024, 1);
        let mut contract = MemoryProviderContract::compatibility_default(
            "mage_flow",
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        contract.calibration = Some(MemoryCalibrationIdentity::new(MAGE_CALIBRATION_FINGERPRINT));
        contract.asset_facts.base_bytes = gib_to_bytes(6.0);
        contract.lifecycle.decode_tiling = true;
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                    MemoryStrategy::StagedResidency => {
                        MemoryStrategySupport::StructurallyNotApplicable {
                            reason: "single resident pipeline".to_owned(),
                        }
                    }
                    MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
                    _ => MemoryStrategySupport::Missing,
                },
                parameters: if strategy == MemoryStrategy::BoundedDecode {
                    MemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![32],
                        ..Default::default()
                    }
                } else {
                    Default::default()
                },
            })
            .collect();
        let selection = MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(32),
                ..Default::default()
            },
            tier: plan.tier,
        };
        let evidence = MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: "mage_flow".to_owned(),
                backend: "mlx".to_owned(),
                tier: plan.tier,
                mode: "edit".to_owned(),
                overlay: inputs.overlay.clone(),
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                },
                strategy: selection.strategy,
                engaged_composition: contract.engaged_composition(selection.strategy),
                parameters: selection.parameters,
            },
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions {
                static_implementation: MemoryEvidenceVerdict::Satisfied,
                declared_calibration: MemoryEvidenceVerdict::Satisfied,
                historical_verification: MemoryEvidenceVerdict::Satisfied,
                current_environment_verification: MemoryEvidenceVerdict::Satisfied,
                canonical_route_loadability: MemoryEvidenceVerdict::Satisfied,
                exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
            },
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: "wrong-provider-fingerprint".to_owned(),
            sceneworks_revision: REQUEST_EVIDENCE_REVISION.to_owned(),
            inference_revision: INFERENCE_CONTRACT_REVISION.to_owned(),
            harness_version: "test".to_owned(),
            predicted_peak_bytes: gib_to_bytes(5.0),
            observed_peak_bytes: Some(gib_to_bytes(5.0)),
            parity: MemoryParityContract::Exact,
            parity_result: MemoryParityResult::Passed,
        };
        let error = evaluate_request_with_budget(
            &request_generator(Some(contract)),
            &plan,
            &inputs,
            MemoryCacheState::Warm,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(10.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(12.0),
            0,
            &[evidence],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("FingerprintMismatch"),
            "mismatch must remain an unverified fail-closed result: {error}"
        );
    }

    /// The dense Mage-Flow-Base DiT (`transformer/diffusion_pytorch_model.safetensors`, bf16 on disk).
    const MAGE_DIT_BYTES: u64 = 8_231_536_784;

    #[test]
    fn full_finetune_tokens_track_the_square_latent_grid() {
        // edge buckets to a multiple of 32, /16 stride, patch_size 1 ⇒ grid² tokens.
        assert_eq!(full_finetune_tokens(1024), 4096.0); // 64²
        assert_eq!(full_finetune_tokens(64), 16.0); // 4²
        assert_eq!(full_finetune_tokens(512), 1024.0); // 32²
                                                       // Sub-32 buckets up to one 32-edge grid (never zero tokens).
        assert_eq!(full_finetune_tokens(16), 4.0); // edge 32 → grid 2 → 4
    }

    #[test]
    fn full_finetune_peak_is_no_signal_without_weights() {
        // Nothing installed ⇒ no signal ⇒ never block (the fit-gate invariant).
        assert_eq!(full_finetune_peak_gb(0, 1024, 1), None);
        assert!(full_finetune_peak_gb(MAGE_DIT_BYTES, 1024, 1).is_some());
    }

    #[test]
    fn full_finetune_rejects_production_resolution_even_on_a_large_mac() {
        // A 1024² full fine-tune of the 4B Mage DiT exceeds even a 128 GB Mac without gradient
        // checkpointing — the exact "don't advertise production-res as fitting" case (sc-14056).
        let msg = full_finetune_fit_message(
            "Mage-Flow Base",
            MAGE_DIT_BYTES,
            1024,
            1,
            Some(MlxMemoryBudget { total_gb: 128.0 }),
        )
        .expect("1024² full-tune must be rejected on 128 GB");
        assert!(
            msg.contains("sc-14989"),
            "message must point at grad-checkpointing: {msg}"
        );
        assert!(
            msg.contains("gradient"),
            "message must name the missing lever: {msg}"
        );
        assert!(
            msg.contains("1024px"),
            "message must name the resolution: {msg}"
        );
        assert!(
            msg.contains("Mage-Flow Base"),
            "message must name the model: {msg}"
        );
    }

    #[test]
    fn full_finetune_permits_a_tiny_resolution_on_a_large_mac() {
        // A tiny-resolution full-tune (the e2e regime) fits a 128 GB Mac — the gate permits it.
        assert_eq!(
            full_finetune_fit_message(
                "Mage-Flow Base",
                MAGE_DIT_BYTES,
                64,
                1,
                Some(MlxMemoryBudget { total_gb: 128.0 }),
            ),
            None
        );
    }

    #[test]
    fn full_finetune_rejects_any_resolution_on_a_small_mac() {
        // The optimizer state alone (~8× the bf16 DiT bytes ≈ 61 GB) exceeds a 32 GB Mac, so a 4B
        // full fine-tune is refused there even at the tiniest resolution — resolution-independent.
        assert!(full_finetune_fit_message(
            "Mage-Flow Base",
            MAGE_DIT_BYTES,
            64,
            1,
            Some(MlxMemoryBudget { total_gb: 32.0 }),
        )
        .is_some());
    }

    /// sc-14056 review — the gradient-accumulation term. The first cut of this gate called the 8×
    /// state multiplier "exact" while omitting the accumulator map `accumulate_grads` holds across
    /// the whole window whenever `gradient_accumulation > 1` (a user-exposed Training Studio knob).
    /// That under-prediction is the dangerous direction: a 64–80 GB Mac was predicted at ~63 GiB,
    /// ADMITTED, and then hard-killed — an MLX overcommit is a SIGKILL nothing downstream can catch.
    ///
    /// This test is written to DISCRIMINATE rather than to restate the formula: it pins a budget in
    /// the exact band where the two multipliers disagree, and asserts the accumulation-1 run is still
    /// permitted there while the accumulation-4 run — identical in every other respect — is refused.
    /// Delete the accumulation term and the second half fails; make the term unconditional and the
    /// first half fails.
    #[test]
    fn full_finetune_accumulation_window_rejects_where_a_single_step_run_fits() {
        // 8× the 7.665 GiB bf16 DiT ≈ 61.3 GiB of state; 10× ≈ 76.7 GiB. At a tiny resolution the
        // activation terms are negligible, so a 72 GB budget sits squarely between them — the band a
        // 64–80 GB Mac lands in, and the band the old formula got wrong.
        const BUDGET: Option<MlxMemoryBudget> = Some(MlxMemoryBudget { total_gb: 72.0 });

        let single = full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 1).expect("weights present");
        let windowed = full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 4).expect("weights present");
        assert!(
            windowed > single,
            "an accumulation window must cost MORE than a single-step run ({windowed} vs {single})"
        );

        assert_eq!(
            full_finetune_fit_message("Mage-Flow Base", MAGE_DIT_BYTES, 64, 1, BUDGET),
            None,
            "accumulation 1 still fits a 72 GB Mac at a tiny resolution — the gate must not become \
             unconditionally stricter"
        );
        let msg = full_finetune_fit_message("Mage-Flow Base", MAGE_DIT_BYTES, 64, 4, BUDGET)
            .expect(
                "accumulation 4 holds an extra full f32 gradient map and must now be REFUSED on a \
                 72 GB Mac — this is the run that previously got an uncatchable SIGKILL",
            );
        assert!(
            msg.contains("Gradient accumulation is 4"),
            "the rejection must name the accumulation lever it can actually act on: {msg}"
        );
        // …and must NOT mention it when accumulation is not what is costing the memory.
        let single_step_reject = full_finetune_fit_message(
            "Mage-Flow Base",
            MAGE_DIT_BYTES,
            64,
            1,
            Some(MlxMemoryBudget { total_gb: 32.0 }),
        )
        .expect("32 GB cannot hold the optimizer state at any resolution");
        assert!(
            !single_step_reject.contains("Gradient accumulation"),
            "at accumulation 1 there is no accumulator buffer, so the advice would be noise: \
             {single_step_reject}"
        );
    }

    /// `0` and `1` both mean "step every micro-step" to the engine, so neither may pay the
    /// accumulator term — a legacy payload that omits the key must not be gated more strictly than
    /// one that sets it to 1.
    #[test]
    fn full_finetune_accumulation_zero_and_one_are_the_same_envelope() {
        assert_eq!(
            full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 0),
            full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 1)
        );
        assert_ne!(
            full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 1),
            full_finetune_peak_gb(MAGE_DIT_BYTES, 64, 2)
        );
    }

    #[test]
    fn full_finetune_never_blocks_without_a_budget() {
        // No platform budget (off-macOS, or an unreadable probe) ⇒ never block, exactly like the
        // generation gate's `Unknown`.
        assert_eq!(
            full_finetune_fit_message("Mage-Flow Base", MAGE_DIT_BYTES, 1024, 1, None),
            None
        );
    }

    #[test]
    fn full_finetune_memory_error_no_signal_when_transformer_absent() {
        // The live entry point permits when the base isn't installed (transformer/ sums to 0) — the
        // base-installed gate is what reports "not installed".
        let empty =
            std::env::temp_dir().join(format!("mage_full_gate_{}_{}", std::process::id(), line!()));
        std::fs::create_dir_all(&empty).expect("mk dir");
        assert_eq!(
            full_finetune_memory_error(&empty, 1024, 1, "Mage-Flow Base"),
            None
        );
        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn parse_memory_cap_accepts_positive_numbers_only() {
        assert_eq!(parse_memory_cap(Some("16")), Some(16.0));
        assert_eq!(parse_memory_cap(Some("  32.5 ")), Some(32.5));
        assert_eq!(parse_memory_cap(Some("0")), None);
        assert_eq!(parse_memory_cap(Some("-8")), None);
        assert_eq!(parse_memory_cap(Some("nan")), None);
        assert_eq!(parse_memory_cap(Some("inf")), None);
        assert_eq!(parse_memory_cap(Some("abc")), None);
        assert_eq!(parse_memory_cap(Some("")), None);
        assert_eq!(parse_memory_cap(None), None);
    }

    #[test]
    fn resolve_budget_prefers_the_emulation_cap() {
        // Cap overrides the real total (emulate a smaller Mac on big hardware).
        assert_eq!(
            resolve_budget(Some(128.0), Some(16.0)),
            Some(MlxMemoryBudget { total_gb: 16.0 })
        );
        // No cap ⇒ the real total.
        assert_eq!(
            resolve_budget(Some(128.0), None),
            Some(MlxMemoryBudget { total_gb: 128.0 })
        );
        // A cap with no real reading still yields a budget (exercisable in a no-probe unit test).
        assert_eq!(
            resolve_budget(None, Some(16.0)),
            Some(MlxMemoryBudget { total_gb: 16.0 })
        );
        // No signal at all ⇒ no budget ⇒ gate no-ops.
        assert_eq!(resolve_budget(None, None), None);
    }

    #[test]
    fn predicted_peak_is_weights_plus_headroom_and_zero_is_no_signal() {
        // 20 GiB of weights ⇒ 20 + headroom.
        let bytes = 20 * 1024 * 1024 * 1024_u64;
        assert_eq!(predicted_peak_gb(bytes), Some(20.0 + HEADROOM_GB));
        // No measurable weights ⇒ no signal.
        assert_eq!(predicted_peak_gb(0), None);
    }

    /// NOTE (sc-15799): `45.67` and `28.43` below are the THIRD copy of the lens-turbo measurement —
    /// the other two are the `HEADROOM_GB` doc table above and the `lens_turbo` row in
    /// `config/tier-integrity.jsonc`, which declares the 17.24 GiB mxfp4 → bf16 upcast as a
    /// backend-capability exception. `scripts/check-tier-integrity.mjs` asserts all three numbers still
    /// appear in THIS file, so a re-measure (sc-16014) that moves one copy and not the others fails the
    /// `parity` lane with a message that names the reason, instead of turning this lane red for something
    /// unrelated.
    #[test]
    fn lens_dense_calibration_covers_the_measured_full_peak_without_blanket_inflation() {
        let gib = BYTES_PER_GIB;
        let lens_materialized = (45.67 * gib).ceil() as u64;
        let lens_peak = predicted_peak_gb_with_headroom(lens_materialized, LENS_DENSE_HEADROOM_GB)
            .expect("lens signal");
        assert!(lens_peak >= 75.55, "{lens_peak} GiB must cover 75.55 GiB");

        // A bf16-on-disk family keeps the generic calculation; no global dense-tier multiplier.
        let ordinary_bf16 = (28.43 * gib).ceil() as u64;
        let ordinary_peak = predicted_peak_gb(ordinary_bf16).expect("ordinary signal");
        assert!((ordinary_peak - 46.43).abs() < 1e-6);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn lens_provider_footprint_expands_only_the_dense_turnkey() {
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_sc11924_{}_{}",
            std::process::id(),
            line!()
        ));
        for (component, bytes) in [("text_encoder", 13), ("transformer", 11), ("vae", 3)] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).expect("component dir");
            std::fs::write(dir.join("model.safetensors"), vec![0; bytes]).expect("fixture");
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let (dense_total, dense_te, dense_headroom) = spec_component_bytes("lens_turbo", &spec);
        let expected_te = (30.07 * BYTES_PER_GIB).ceil() as u64;
        assert_eq!(dense_te, expected_te);
        assert_eq!(dense_total, expected_te + 14);
        assert_eq!(dense_headroom, HeadroomAllowance::LENS_DENSE);
        // sc-16195: a bare measured transient carries NO OS reserve, so the request estimator
        // must leave the whole allowance in its area term for this path.
        assert_eq!(dense_headroom.os_reserve_gb, 0.0);

        std::fs::write(
            root.join("text_encoder").join("config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .expect("packed marker");
        let (packed_total, packed_te, packed_headroom) = spec_component_bytes("lens_turbo", &spec);
        assert_eq!((packed_total, packed_te), (27, 13));
        assert_eq!(packed_headroom, HeadroomAllowance::GENERIC);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn fit_decision_rejects_only_a_genuine_overflow() {
        let budget = MlxMemoryBudget { total_gb: 16.0 };
        // qwen-image q8: ~36 GiB weights + 18 headroom (sc-10863) = 54 ⇒ too big for a 16 GB Mac.
        assert_eq!(
            fit_decision(predicted_peak_gb(36 * 1024 * 1024 * 1024_u64), Some(budget)),
            FitDecision::TooBig {
                needed_gb: 36.0 + HEADROOM_GB,
                available_gb: 16.0,
            }
        );
        // A ~3 GiB model fits a roomy budget (3 + 18 headroom = 21 < 32).
        assert_eq!(
            fit_decision(
                predicted_peak_gb(3 * 1024 * 1024 * 1024_u64),
                Some(MlxMemoryBudget { total_gb: 32.0 })
            ),
            FitDecision::Fits
        );
        // Exactly-fits is not a rejection: budget 46, need 46.
        assert_eq!(
            fit_decision(Some(46.0), Some(MlxMemoryBudget { total_gb: 46.0 })),
            FitDecision::Fits
        );
        // Missing either input ⇒ never block.
        assert_eq!(fit_decision(None, Some(budget)), FitDecision::Unknown);
        assert_eq!(fit_decision(Some(8.0), None), FitDecision::Unknown);
    }

    #[test]
    fn sum_safetensors_skips_appledouble_and_nonweights_and_recurses() {
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_sum_{}_{}",
            std::process::id(),
            line!()
        ));
        let te = root.join("text_encoder");
        let dit = root.join("transformer");
        std::fs::create_dir_all(&te).expect("mk te");
        std::fs::create_dir_all(&dit).expect("mk dit");
        std::fs::write(te.join("model.safetensors"), vec![0u8; 1000]).expect("te weights");
        std::fs::write(dit.join("diffusion.safetensors"), vec![0u8; 2000]).expect("dit weights");
        // AppleDouble sidecar + a non-weight file must NOT be counted.
        std::fs::write(te.join("._model.safetensors"), vec![0u8; 500]).expect("sidecar");
        std::fs::write(dit.join("config.json"), vec![0u8; 700]).expect("config");

        assert_eq!(sum_safetensors_bytes(&root), 3000);
        // Missing dir ⇒ 0 (no signal).
        assert_eq!(sum_safetensors_bytes(&root.join("nope")), 0);

        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-15154 — a SPLIT-layout tier's staged co-requisites are part of what it loads.
    ///
    /// Mage-Flow's per-tier dir holds the DiT alone; the text encoder and VAE are bit-identical
    /// across the six variants and staged from a shared mirror. Scanning only `spec.weights` scored a
    /// q4 edit install at the DiT's bytes, so the over-budget message quoted a peak derived from a
    /// third of the tier, and the permissive legacy override admitted budgets the tier does not
    /// fit. The pre-fix number is asserted alongside the fixed one — a test that only checked the new
    /// total could pass on a spec that stages nothing.
    #[test]
    fn a_staged_component_counts_toward_the_tier_that_loads_it() {
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_staged_{}_{}",
            std::process::id(),
            line!()
        ));
        let write = |dir: std::path::PathBuf, bytes: usize| {
            std::fs::create_dir_all(&dir).expect("mk dir");
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("write");
            dir
        };
        // The variant mirror's tier dir (DiT only) + the shared components mirror.
        let tier = root.join("q4");
        write(tier.join("transformer"), 3_000);
        let te = write(root.join("shared/q4/text_encoder"), 5_000);
        let vae = write(root.join("shared/q4/vae"), 1_000);

        // Nothing staged — the pre-fix reading, and still correct for a flat snapshot.
        let bare = LoadSpec::new(WeightsSource::Dir(tier.clone()));
        assert_eq!(spec_component_bytes("mage_flow_edit", &bare).0, 3_000);

        // Staged the way the load stages them.
        let staged = bare
            .clone()
            .with_component("text_encoder", WeightsSource::Dir(te))
            .with_component("vae", WeightsSource::Dir(vae));
        let (total, te_bytes, _) = spec_component_bytes("mage_flow_edit", &staged);
        println!(
            "mage_flow_edit split tier: bare total={} staged total={total} staged te={te_bytes}",
            spec_component_bytes("mage_flow_edit", &bare).0
        );
        assert_eq!(
            total, 9_000,
            "the staged text encoder and VAE are weights this tier holds resident"
        );
        // The TEXT-ENCODER split comes from the provider's own footprint at the pinned inference
        // revision, which must likewise resolve the STAGED dir and not `<tier>/text_encoder`. A zero
        // here would collapse the sequential schedule's `max(te, rest)` onto the whole model and
        // silently over-reject, so it is asserted rather than inferred from the total.
        //
        // macOS ONLY: the MLX media registry — and therefore any provider footprint — is compiled in
        // on macOS alone. Off-macOS `footprint()` yields `None`, `resolve_text_encoder_bytes` falls
        // back to the diffusers `text_encoder*` subdir scan of the TIER dir, and that is correctly 0
        // here. The `total` assertions above are registry-free and hold on every platform, which is
        // what this test is really for.
        #[cfg(target_os = "macos")]
        assert_eq!(
            te_bytes, 5_000,
            "mage_flow's per-component footprint must follow the staged component dirs"
        );

        // A component that resolves INSIDE the weights dir was already counted by the scan, so it
        // must not be added twice — the flat published layout, staged explicitly.
        let flat_te = write(tier.join("text_encoder"), 700);
        let flat = LoadSpec::new(WeightsSource::Dir(tier))
            .with_component("text_encoder", WeightsSource::Dir(flat_te));
        assert_eq!(spec_component_bytes("mage_flow_edit", &flat).0, 3_700);

        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-15154 — the Mage q4 admit boundary, from the REAL published install sizes.
    ///
    /// The epic-14034 acceptance run found emulated caps of **both 6 GB and 7 GB admitting** a q4
    /// edit job. That was not the memory model being wrong; it was this gate seeing 2.17 GiB of
    /// weights (the DiT alone) where the tier installs 6.52 GiB, so the permissive weights-fit floor
    /// (`Σstaged ≤ budget − legacy reserve`) cleared caps the tier cannot hold.
    ///
    /// Bytes are the manifest's `estimatedSizeBytes` for `mage_flow_edit` q4:
    /// DiT 2,326,294,167 + shared TE 4,331,077,508 + shared VAE 345,053,168 = 7,002,424,843
    /// (7.00 GB decimal / 6.52 GiB), against inference's measured `calibration_peak_gb(Q4)` of
    /// **7.868 GB** at 512² — the anchor sc-15071 re-derived and this branch's pin bump finally
    /// brings into the app. The two gates are different instruments (this one bounds WIRED weights
    /// under a staged schedule; mage's own `ensure_generation_fits` bounds the complete peak against
    /// MLX's live limit), but they must not disagree about direction: a 6 GB machine cannot hold a
    /// tier that peaks at 7.87 GB, and this gate must stop saying it can.
    #[test]
    fn the_mage_q4_admit_boundary_moves_with_the_real_install_size() {
        const DIT: u64 = 2_326_294_167;
        const TE: u64 = 4_331_077_508;
        const VAE: u64 = 345_053_168;
        let total = DIT + TE + VAE;
        let budget = |gb: f64| Some(MlxMemoryBudget { total_gb: gb });
        for gb in [6.0, 7.0] {
            println!(
                "q4 @ {gb} GB cap: fixed={:?}  pre-fix(DiT-only)={:?}",
                decide_residency(total, TE, budget(gb), true),
                decide_residency(DIT, 0, budget(gb), true),
            );
        }

        // 6 GB: below the 7.868 GB measured q4 peak ⇒ must refuse.
        assert!(
            matches!(
                decide_residency(total, TE, budget(6.0), true),
                ResidencyOutcome::Reject { .. }
            ),
            "a 6 GB Mac cannot hold a tier whose measured peak is 7.87 GB"
        );
        // 7 GB: the staged schedule's wired high-water is the text encoder (4.03 GiB), which fits
        // 7 − the legacy reserve. Admitting here is the transition override working as designed — a machine
        // that can hold the weights runs, paging the activation transient.
        assert_eq!(
            decide_residency(total, TE, budget(7.0), true),
            ResidencyOutcome::Sequential
        );

        // The pre-fix reading — the DiT alone, no text-encoder split — cleared BOTH caps. Asserted
        // so the numbers above are visibly about the footprint and not about the thresholds.
        assert_eq!(
            decide_residency(DIT, 0, budget(6.0), true),
            ResidencyOutcome::Sequential,
            "this is the acceptance-run symptom: 6 GB admitted because the gate saw a third of the \
             tier. If it now rejects, the floor moved and this test's premise must be re-derived."
        );
    }

    #[test]
    fn weights_source_bytes_counts_both_file_and_dir_control_checkpoints() {
        // The qwen_image_control VACE branch ships either as a single `.safetensors` File or as a Dir
        // of shards; both must be counted so `apply_residency_policy` folds the control branch into the
        // heavy side of the staged-peak split (else the DiT-phase working set is under-counted).
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_ctrl_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&root).expect("mk root");

        // Single-file control checkpoint ⇒ its file length.
        let file = root.join("control.safetensors");
        std::fs::write(&file, vec![0u8; 4096]).expect("control file");
        assert_eq!(
            weights_source_bytes(&WeightsSource::File(file.clone())),
            4096
        );

        // Dir control checkpoint ⇒ the recursive `.safetensors` sum (AppleDouble sidecars skipped).
        let dir = root.join("control_dir");
        std::fs::create_dir_all(&dir).expect("mk control dir");
        std::fs::write(dir.join("part-1.safetensors"), vec![0u8; 1000]).expect("shard 1");
        std::fs::write(dir.join("part-2.safetensors"), vec![0u8; 2000]).expect("shard 2");
        std::fs::write(dir.join("._part-1.safetensors"), vec![0u8; 999]).expect("sidecar");
        assert_eq!(weights_source_bytes(&WeightsSource::Dir(dir)), 3000);

        std::fs::remove_dir_all(&root).ok();
    }

    /// The gate derives sequential-capability from each engine's REGISTERED descriptor bit
    /// (`Capabilities::supports_sequential_offload`) rather than a hand-maintained allowlist (sc-10840,
    /// epic 10834). This exercises the LIVE registry, so it must see the force-linked `mlx_gen_*`
    /// providers — anchored (`use mlx_gen_* as _;` in `image_jobs`) only on macOS, the sole platform the
    /// MLX gate runs on. Off-Mac the image registry is empty, so this is macOS-gated exactly like the
    /// `engines.rs` descriptor sweeps. At the pinned mlx-gen `45428fa` every image engine advertises the
    /// bit, so every wired id resolves true through the shared registry query.
    #[cfg(target_os = "macos")]
    #[test]
    fn engine_supports_sequential_is_derived_from_the_registered_capability() {
        // The earlier-wired families (sdxl / z-image / qwen / lens / krea) still resolve true — proving
        // dropping the hardcoded allowlist introduced no regression for the already-covered engines.
        for id in [
            "sdxl",
            "z_image",
            "z_image_control",
            "z_image_turbo",
            "z_image_turbo_control",
            "qwen_image",
            "qwen_image_edit",
            "qwen_image_control",
            "lens",
            "lens_turbo",
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
            "krea_2_turbo_edit",
            "krea_2_turbo_control",
        ] {
            assert!(
                engine_supports_sequential(id),
                "{id}: earlier-wired family must stay sequential-capable"
            );
        }
        // The sc-10840 Phase-1 fan-out families are AUTO-covered by the capability query with no
        // allowlist edit here — the whole point of deriving from the descriptor bit. A newly-wired
        // engine (e.g. `sd3_5_large`) is sequential-capable the moment its provider advertises the bit.
        for id in [
            "sd3_5_large",
            "sd3_5_large_turbo",
            "sd3_5_medium",
            "sana_1600m",
            "sana_sprint_1600m",
            "flux1_schnell",
            "flux1_dev",
            "flux1_dev_control",
            "flux2_klein_9b",
            "flux2_klein_9b_edit",
            "flux2_klein_9b_kv_edit",
            "flux2_dev",
            "flux2_dev_control",
            "flux2_dev_edit",
            "chroma1_base",
            "chroma1_flash",
            "chroma1_hd",
            "ideogram_4",
            "ideogram_4_turbo",
            "kolors",
            "anima_base",
            "anima_aesthetic",
            "anima_turbo",
            "boogu_image",
            "boogu_image_turbo",
            "boogu_image_edit",
            "bernini",
        ] {
            assert!(
                engine_supports_sequential(id),
                "{id}: sc-10840 fan-out engine must be sequential-capable at mlx-gen 45428fa"
            );
        }
        // A REGISTERED engine that does NOT advertise the bit stays false: sensenova's encoder is fused
        // into a unified MoT (`footprint` te=0) — no separable text encoder to drop, so residency buys
        // nothing and Sequential would be a no-op that OOMs. This proves the query reads the descriptor
        // BIT, not mere registry membership.
        assert!(!engine_supports_sequential("sensenova_u1_8b"));
    }

    /// An id with no registered generator is never sequential-capable (the safe default: never select a
    /// residency policy the provider won't honor) — a cross-platform invariant.
    #[test]
    fn engine_supports_sequential_is_false_for_an_unregistered_id() {
        assert!(!engine_supports_sequential("no_such_engine_xyz"));
    }

    /// Candle's sc-12130 twin of the macOS registry sweep above. These are the generic generator ids that
    /// reach the Candle fit gate after route diversion. Krea edit is now true because sc-12129 landed;
    /// the bespoke pose-control provider is not a registered generator and remains outside this gate.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn candle_sequential_capability_is_derived_from_the_registered_descriptor() {
        for id in [
            "flux1_dev",
            "flux1_schnell",
            "flux2_dev",
            "flux2_klein_9b",
            "qwen_image",
            "z_image_turbo",
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
        ] {
            assert!(
                engine_supports_sequential(id),
                "{id}: wired Candle provider must advertise sequential residency"
            );
        }
        assert!(!engine_supports_sequential("krea_2_turbo_control"));
    }

    #[test]
    fn predicted_sequential_peak_is_largest_component_plus_headroom() {
        let gib = 1024 * 1024 * 1024_u64;
        // illustrious q8-class: total ~5 GiB, text encoders ~1 GiB ⇒ staged = max(1, 4) + headroom.
        let total = 5 * gib;
        let te = gib;
        assert_eq!(
            predicted_sequential_peak_gb(total, te),
            Some(4.0 + HEADROOM_GB)
        );
        // TE-dominant (lens-class): total 17, TE 13 ⇒ staged = max(13, 4) + headroom = 13 + headroom.
        assert_eq!(
            predicted_sequential_peak_gb(17 * gib, 13 * gib),
            Some(13.0 + HEADROOM_GB)
        );
        // Unmeasured text encoders ⇒ staged == resident sum (no claimed saving), the safe fallback.
        assert_eq!(
            predicted_sequential_peak_gb(20 * gib, 0),
            Some(20.0 + HEADROOM_GB)
        );
        // Nothing measured ⇒ no signal.
        assert_eq!(predicted_sequential_peak_gb(0, 0), None);
    }

    #[test]
    fn resolve_offload_rewrites_toobig_only_when_capable() {
        let too_big = FitDecision::TooBig {
            needed_gb: 46.0,
            available_gb: 16.0,
        };
        // Sequential-capable provider ⇒ Offload (carrying the resident numbers).
        assert_eq!(
            resolve_offload(too_big.clone(), true),
            FitDecision::Offload {
                needed_gb: 46.0,
                available_gb: 16.0,
            }
        );
        // Non-capable ⇒ still a reject.
        assert!(matches!(
            resolve_offload(too_big, false),
            FitDecision::TooBig { .. }
        ));
        // Fits / Unknown are never rewritten.
        assert_eq!(resolve_offload(FitDecision::Fits, true), FitDecision::Fits);
        assert_eq!(
            resolve_offload(FitDecision::Unknown, true),
            FitDecision::Unknown
        );
    }

    #[test]
    fn sequential_overflow_rejects_only_a_genuine_staged_overflow() {
        let budget = Some(MlxMemoryBudget { total_gb: 16.0 });
        // Staged still needs 20 > 16 ⇒ reject even sequentially.
        assert_eq!(sequential_overflow_gb(Some(20.0), budget), Some(20.0));
        // Staged fits (14 <= 16) ⇒ run sequentially, no reject.
        assert_eq!(sequential_overflow_gb(Some(14.0), budget), None);
        // Exactly-fits is not an overflow.
        assert_eq!(sequential_overflow_gb(Some(16.0), budget), None);
        // No staged estimate or no budget ⇒ best-effort run (no reject).
        assert_eq!(sequential_overflow_gb(None, budget), None);
        assert_eq!(sequential_overflow_gb(Some(20.0), None), None);
    }

    #[test]
    fn too_big_error_names_model_budget_and_optional_staged() {
        // Plain resident reject (non-staging provider): no staged note.
        let WorkerError::InvalidPayload(resident) = too_big_error("qwen-image", 46.0, 16.0, None)
        else {
            panic!("expected InvalidPayload");
        };
        assert!(
            resident.contains("qwen-image"),
            "names the model: {resident}"
        );
        assert!(resident.contains("unified memory"), "explains: {resident}");
        assert!(resident.contains("46"), "states what it needs: {resident}");
        assert!(resident.contains("16"), "states the budget: {resident}");
        assert!(
            !resident.contains("one component at a time"),
            "no staged note when not attempted: {resident}"
        );
        // Staged reject: the message also names the one-at-a-time requirement.
        let WorkerError::InvalidPayload(staged) = too_big_error("sdxl", 46.0, 16.0, Some(24.0))
        else {
            panic!("expected InvalidPayload");
        };
        assert!(
            staged.contains("one component at a time"),
            "names the staged path: {staged}"
        );
        assert!(
            staged.contains("24"),
            "states the staged requirement: {staged}"
        );
    }

    // The PEAK layer (`decide_residency_by_peak`) still selects Resident / Sequential / Reject exactly
    // as the pre-sc-12179 gate did — the Decision 2 legacy override is layered on top only for
    // unverified cells, so this proves the peak selection is intact.
    #[test]
    fn decide_residency_by_peak_picks_resident_sequential_or_reject_by_budget() {
        let gib = 1024 * 1024 * 1024_u64;
        // illustrious q8-class: total ~5 GiB (TE ~1, DiT+VAE ~4). With HEADROOM_GB=18 (sc-10863):
        // resident peak = 5+18 = 23; staged peak = max(1, 4)+18 = 22.
        let total = 5 * gib;
        let te = gib;

        // Roomy budget (128 GB Mac) ⇒ Resident (keep the warm path).
        assert_eq!(
            decide_residency_by_peak(total, te, Some(MlxMemoryBudget { total_gb: 128.0 }), true),
            ResidencyOutcome::Resident
        );
        // Budget between staged (22) and resident (23): resident won't fit, staged will, provider
        // stages ⇒ Sequential. This is the fit-gate SELECTING sequential residency.
        assert_eq!(
            decide_residency_by_peak(total, te, Some(MlxMemoryBudget { total_gb: 22.5 }), true),
            ResidencyOutcome::Sequential
        );
        // Same budget but a provider that can't stage ⇒ reject (never a silent Resident that OOMs).
        assert!(matches!(
            decide_residency_by_peak(total, te, Some(MlxMemoryBudget { total_gb: 22.5 }), false),
            ResidencyOutcome::Reject {
                staged_gb: None,
                ..
            }
        ));
        // Budget below even the staged peak (22) ⇒ reject, naming the staged requirement.
        assert!(matches!(
            decide_residency_by_peak(total, te, Some(MlxMemoryBudget { total_gb: 20.0 }), true),
            ResidencyOutcome::Reject {
                staged_gb: Some(_),
                ..
            }
        ));
        // No budget signal ⇒ Resident (never block).
        assert_eq!(
            decide_residency_by_peak(total, te, None, true),
            ResidencyOutcome::Resident
        );
        // Unmeasured weights ⇒ Resident (no signal).
        assert_eq!(
            decide_residency_by_peak(0, 0, Some(MlxMemoryBudget { total_gb: 8.0 }), true),
            ResidencyOutcome::Resident
        );
    }

    #[test]
    fn generic_mlx_adopts_shared_selector_without_an_optimized_claim() {
        let observation = generic_mlx_shared_observation(
            4 * 1024 * 1024 * 1024,
            Some(MlxMemoryBudget { total_gb: 32.0 }),
            HEADROOM_GB,
        );
        assert!(matches!(
            observation,
            crate::memory_strategy::Selection::Selected {
                selection: gen_core::MemorySelection {
                    strategy: gen_core::MemoryStrategy::Resident,
                    ..
                },
                ..
            }
        ));
    }

    /// Decision 1: replacing `weights_fit_floor` must preserve the settled Decision 2 legacy result.
    /// The renamed transition override uses the typed 2 GiB legacy reserve only for unverified cells.
    #[test]
    fn legacy_admission_override_preserves_small_mac_behavior() {
        let gib = 1024 * 1024 * 1024_u64;

        // SANA-class small model on an 8 GB Mac: total 2 GiB (TE 1, rest 1). Peak = 2+18 = 20 ≫ 8 and
        // staged peak = 1+18 = 19 ≫ 8, so the PEAK layer rejects outright...
        let (total, te) = (2 * gib, gib);
        let budget = Some(MlxMemoryBudget { total_gb: 8.0 });
        assert!(matches!(
            decide_residency_by_peak(total, te, budget, true),
            ResidencyOutcome::Reject { .. }
        ));
        // ...but the staged weights (1 GiB) fit 8 − 2 = 6, so legacy runs it Sequential instead of
        // walling off the machine. This is exactly the model that generated fine on 0.7.3.
        assert_eq!(
            decide_residency(total, te, budget, true),
            ResidencyOutcome::Sequential
        );
        // A non-staging provider whose whole 2 GiB weights fit 6 loads Resident best-effort (not reject).
        assert_eq!(
            decide_residency(total, te, budget, false),
            ResidencyOutcome::Resident
        );

        // A genuinely-too-big model on 8 GB still rejects: 40 GiB weights (staged max-component 30) can
        // NOT be held resident under any policy ⇒ the override returns None and the reject stands.
        let (big_total, big_te) = (40 * gib, 10 * gib);
        assert!(matches!(
            decide_residency(big_total, big_te, budget, true),
            ResidencyOutcome::Reject { .. }
        ));

        // The override never fabricates a decision without a budget signal.
        assert_eq!(
            decide_residency(total, te, None, true),
            ResidencyOutcome::Resident
        );
    }

    /// Decision 1's existing 8 GiB policy guard: until an exact cell opts into evidence, the settled
    /// transition keeps this real on-disk q4 footprint on legacy. This is a policy outcome only, not
    /// a model calibration or implementation claim. Measured
    /// tier layout: total 5.49 GiB (text_encoder 2.11, transformer 3.23, vae 0.15), so the largest
    /// single component the Sequential schedule ever holds wired is ~3.38 GiB. Against an 8 GB budget
    /// (legacy reserve 2 ⇒ 6 GiB weight budget) that admits with ~2.6 GiB of margin.
    #[test]
    fn unverified_q4_8gb_guard_remains_on_the_legacy_transition() {
        // Bytes rounded from the measured tier (…/z-image-turbo-mlx/…/q4), following HF-cache symlinks.
        let mib = 1024 * 1024_u64;
        let total = 5624 * mib; // ~5.49 GiB whole model
        let te = 2161 * mib; //    ~2.11 GiB Qwen text encoder (dropped first under Sequential)
        let budget = Some(MlxMemoryBudget { total_gb: 8.0 });

        // The flat-headroom peak layer would reject it (5.49 + 18 = 23.49 ≫ 8)...
        assert!(matches!(
            decide_residency_by_peak(total, te, budget, true),
            ResidencyOutcome::Reject { .. }
        ));
        // ...but z-image-turbo stages components, and its largest (transformer ≈ 3.38 GiB) fits the
        // 6 GiB weight budget ⇒ Sequential. This is the #1544 baseline that must keep working.
        assert_eq!(
            decide_residency(total, te, budget, true),
            ResidencyOutcome::Sequential
        );
    }

    #[test]
    fn sum_text_encoder_bytes_sums_only_text_encoder_subdirs() {
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_te_{}_{}",
            std::process::id(),
            line!()
        ));
        // SDXL-shaped tree: two CLIP encoders + the U-Net + VAE.
        for (sub, bytes) in [
            ("text_encoder", 1000usize),
            ("text_encoder_2", 3000),
            ("unet", 9000),
            ("vae", 400),
        ] {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).expect("mk subdir");
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("weights");
        }
        // Only the two text-encoder subdirs count (1000 + 3000); unet/vae are excluded.
        assert_eq!(sum_text_encoder_bytes(&root), 4000);
        // The whole-model sum includes everything.
        assert_eq!(sum_safetensors_bytes(&root), 13400);
        // Missing dir ⇒ 0.
        assert_eq!(sum_text_encoder_bytes(&root.join("nope")), 0);

        std::fs::remove_dir_all(&root).ok();
    }

    // HF cache stores each shard as a symlink into `blobs/`; the gate must follow those to the real
    // byte size. The synthetic test above uses plain files, so exercise the symlink layout here.
    #[cfg(unix)]
    #[test]
    fn sum_safetensors_follows_hf_cache_symlinks() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_symlink_{}_{}",
            std::process::id(),
            line!()
        ));
        let blobs = root.join("blobs");
        let snap = root.join("snapshots/hash/transformer");
        std::fs::create_dir_all(&blobs).expect("mk blobs");
        std::fs::create_dir_all(&snap).expect("mk snap");
        std::fs::write(blobs.join("weightblob"), vec![0u8; 4096]).expect("weight blob");
        std::fs::write(blobs.join("sidecarblob"), vec![0u8; 500]).expect("sidecar blob");
        symlink(blobs.join("weightblob"), snap.join("diffusion.safetensors")).expect("weight link");
        symlink(
            blobs.join("sidecarblob"),
            snap.join("._diffusion.safetensors"),
        )
        .expect("sidecar link");

        // Summing the SNAPSHOT dir follows the symlink to the 4096-byte blob and skips the `._`
        // sidecar; the `blobs/` dir is not under the snapshot, so nothing is double-counted.
        assert_eq!(sum_safetensors_bytes(&root.join("snapshots/hash")), 4096);

        std::fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn sum_safetensors_terminates_on_directory_symlink_cycles() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_cycle_{}_{}",
            std::process::id(),
            line!()
        ));
        let weights = root.join("weights");
        std::fs::create_dir_all(&weights).expect("mk weights");
        std::fs::write(weights.join("model.safetensors"), vec![0_u8; 4096]).expect("write weights");
        symlink(&root, weights.join("cycle")).expect("create directory cycle");

        assert_eq!(sum_safetensors_bytes(&root), 4096);

        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-10894: on a boogu-style snapshot (text encoder under `mllm/`, not `text_encoder*`), the
    /// subdir scan reads ZERO, but `resolve_text_encoder_bytes` PREFERS a provider footprint value when
    /// present and only falls back to the scan when it is `None`.
    #[test]
    fn resolve_text_encoder_prefers_footprint_over_subdir_scan() {
        let root = std::env::temp_dir().join(format!(
            "mlx_fit_gate_resolve_{}_{}",
            std::process::id(),
            line!()
        ));
        // Encoder under `mllm/`, DiT `transformer/`, VAE `vae/` — NO `text_encoder*` subdir.
        for (sub, bytes) in [("mllm", 1500usize), ("transformer", 9000), ("vae", 400)] {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).expect("mk subdir");
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("weights");
        }
        // The historical subdir scan finds no `text_encoder*` → 0 (the bug this seam fixes).
        assert_eq!(sum_text_encoder_bytes(&root), 0);
        // The whole-model sum still sees every component.
        assert_eq!(sum_safetensors_bytes(&root), 10900);
        // No footprint declared ⇒ fall back to the (zero) subdir scan.
        assert_eq!(resolve_text_encoder_bytes(None, &root), 0);
        // A provider footprint (the `mllm/` bytes) is preferred, even though the scan reads zero.
        assert_eq!(resolve_text_encoder_bytes(Some(1500), &root), 1500);

        std::fs::remove_dir_all(&root).ok();
    }

    /// #1544 baseline through the LIVE gate path on REAL weights (ignored — needs the model on disk +
    /// the force-linked registry). Drives `residency_for_dir` — the exact seam the worker's cold load
    /// uses — against the real z-image-turbo q4 tier under an emulated 8 GB Mac
    /// (`SCENEWORKS_MLX_MEMORY_CAP_GB=8`), so it exercises the real on-disk `.safetensors` scan, the
    /// provider footprint TE split, the registered `supports_sequential_offload` capability, AND the
    /// budget resolution together. Must come back Sequential, not Reject. Run explicitly (alone, since
    /// it sets a process env var):
    ///   cargo test -p sceneworks-worker --lib -- --ignored --nocapture z_image_turbo_q4_live_gate
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs z-image-turbo q4 weights on disk + the force-linked mlx-gen registry"]
    fn z_image_turbo_q4_live_gate_admits_under_an_emulated_8gb_cap() {
        // Resolve the q4 snapshot dir from the HF cache (HF_HOME or ~/.cache/huggingface).
        let hf_home = std::env::var("HF_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
                    .join(".cache/huggingface")
            });
        let snapshots = hf_home.join("hub/models--SceneWorks--z-image-turbo-mlx/snapshots");
        let Some(q4) = std::fs::read_dir(&snapshots).ok().and_then(|entries| {
            entries
                .flatten()
                .map(|e| e.path().join("q4"))
                .find(|p| p.is_dir())
        }) else {
            eprintln!(
                "SKIP: z-image-turbo q4 not found under {}",
                snapshots.display()
            );
            return;
        };

        // Emulate an 8 GB Mac through the same env the epic added for exactly this (sc-10835).
        // Through the crate-wide seam: `set_var` is process-global, and the hand-rolled
        // set/`remove_var` pair this replaces both skipped the lock AND left the cap set for the rest
        // of the process if the assertion below panicked between them (sc-12380).
        let outcome = crate::test_env::temp_env_var(MLX_MEMORY_CAP_ENV, "8", || {
            residency_for_dir("z_image_turbo", &q4)
        });

        eprintln!("live gate on {} @ 8 GB → {outcome:?}", q4.display());
        assert_eq!(
            outcome,
            ResidencyOutcome::Sequential,
            "z-image-turbo q4 (the #1544 0.7.3 baseline) must run on an 8 GB Mac, not be rejected"
        );
    }

    /// sc-10894 end-to-end: a non-zero footprint text encoder flips the residency decision from Reject to
    /// Sequential where the zero-reading subdir scan (the fallback) would reject. This is the whole point
    /// of the seam — the staged working set is only real when the text-encoder split is measured. Post
    /// sc-12179 the flip runs through the weights-fit floor: the measured TE lowers the staged WEIGHTS
    /// (the wired residency), which is what legacy admits against `budget − reserve`.
    #[test]
    fn footprint_text_encoder_flips_reject_to_sequential() {
        let gib = 1024 * 1024 * 1024_u64;
        // boogu-class: whole model 22 GiB (mllm 13 + transformer 8 + vae 1). No `text_encoder*` subdir,
        // so the subdir scan reads 0.
        let total = 22 * gib;
        // Budget where the staged WEIGHTS decide it: floor ceiling = 22 − 2 = 20 GiB. te=0 ⇒ staged
        // weights = 22 > 20 (reject); te=13 ⇒ staged weights = max(13, 9) = 13 ≤ 20 (Sequential).
        let budget = Some(MlxMemoryBudget { total_gb: 22.0 });

        // Fallback path (footprint None on a dir with no `text_encoder*`) → te = 0 → staged weights ==
        // whole model (22 GiB) > 20 → Reject even under the floor: one component IS the whole model.
        let te_fallback = resolve_text_encoder_bytes(None, std::path::Path::new("/nonexistent"));
        assert_eq!(te_fallback, 0);
        assert!(matches!(
            decide_residency(total, te_fallback, budget, true),
            ResidencyOutcome::Reject { .. }
        ));

        // Provider footprint (te = 13 GiB) → staged weights = max(13, 22 − 13 = 9) = 13 ≤ 20 → Sequential.
        let te_footprint =
            resolve_text_encoder_bytes(Some(13 * gib), std::path::Path::new("/ignored"));
        assert_eq!(te_footprint, 13 * gib);
        assert_eq!(
            decide_residency(total, te_footprint, budget, true),
            ResidencyOutcome::Sequential
        );
    }

    // -----------------------------------------------------------------------------------------
    // Mochi 1 frame-aware decode gate (epic 1788 / sc-11992)
    // -----------------------------------------------------------------------------------------

    /// Mochi's q4 resident weights (GiB): DiT 9.007 + T5-XXL bf16 8.871 + VAE 0.856 = 18.73 (the
    /// exact hosted bytes B1's manifest derivation pins). Used as the weight signal in the gate tests.
    const MOCHI_Q4_RESIDENT_BYTES: u64 = 9_670_883_602 + 9_524_669_250 + 919_551_200;

    // The pure decode arithmetic (`mochi_decode_peak_gb`: linearity in frames + pixels, the f32
    // anchor) is pinned in `crate::fit_gate`'s tests alongside the formula, which moved there in
    // sc-12306 when the candle video lane grew the same gate. What stays here is what is genuinely
    // MLX: the unified-memory budget shape, the typed unified reserve, and the Mac-worded message.

    /// THE gate behavior: on ONE fixed machine, a short Mochi clip is admitted and a long one is
    /// rejected. A frame-blind gate (the plausible wrong implementation — reusing `predicted_peak_gb`
    /// or dropping the frames term) cannot pass this: it would give both clips the same verdict.
    #[test]
    fn mochi_gate_admits_a_short_clip_and_rejects_a_long_one_on_the_same_mac() {
        // A 64 GB Mac — the machine B1's `mlx.minMemoryGb: 96` says is under-provisioned, and the one
        // the epic's crash report names.
        let mac_64 = Some(MlxMemoryBudget { total_gb: 64.0 });

        // 19 frames (the engine's own DEFAULT_FRAMES, ~0.6 s): 18.73 weights + 9.32 decode + 2 OS
        // ≈ 30.1 GiB ⇒ admitted. (All GiB — see `mochi_decode_peak_gb`.)
        assert!(
            mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 19, 848, 480, mac_64).is_none(),
            "a 19-frame clip fits a 64 GB Mac and must NOT be rejected"
        );

        // 151 frames (the shipped 5 s default): 18.73 + 60.56 + 2 ≈ 81.3 GiB ⇒ rejected BEFORE the
        // untiled decode can trip MLX's `exit(-1)`.
        assert!(
            mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 151, 848, 480, mac_64).is_some(),
            "a 151-frame clip needs ~81 GB and MUST be rejected on a 64 GB Mac — this is the \
             unmappable exit(-1) the gate exists to prevent"
        );

        // The same 151-frame clip is admitted on a 128 GB Mac — the gate rejects by BUDGET, it does
        // not blanket-ban the default duration.
        let mac_128 = Some(MlxMemoryBudget { total_gb: 128.0 });
        assert!(
            mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 151, 848, 480, mac_128).is_none(),
            "a 151-frame clip fits a 128 GB Mac"
        );
    }

    /// The rejection message must be self-contained + actionable, following the `too_big_error`
    /// convention (name the model, explain the constraint, state need vs budget) and additionally
    /// naming Mochi's one real lever: clip length.
    #[test]
    fn mochi_too_big_error_names_model_budget_and_the_clip_length_lever() {
        let mac_64 = Some(MlxMemoryBudget { total_gb: 64.0 });
        let Some(WorkerError::InvalidPayload(message)) =
            mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 151, 848, 480, mac_64)
        else {
            panic!("expected an InvalidPayload rejection");
        };
        assert!(message.contains("mochi_1"), "names the model: {message}");
        assert!(
            message.contains("unified memory"),
            "explains the constraint: {message}"
        );
        assert!(
            message.contains("81"),
            "states what it needs (~81 GB): {message}"
        );
        assert!(message.contains("64"), "states the budget: {message}");
        assert!(
            message.contains("151"),
            "names the clip length that drove it: {message}"
        );
        assert!(message.contains("848x480"), "names the geometry: {message}");
        assert!(
            message.contains("Shorten the clip"),
            "gives the actionable lever: {message}"
        );
    }

    /// No signal ⇒ never block, matching `fit_decision`/`predicted_peak_gb`. An unmeasurable model
    /// dir or a machine with no probe must not wall off generation.
    #[test]
    fn mochi_gate_never_blocks_without_a_signal() {
        let mac_64 = Some(MlxMemoryBudget { total_gb: 64.0 });
        // Weights unmeasurable (empty/missing dir ⇒ 0 bytes) ⇒ admit.
        assert!(mochi_fit_error("mochi_1", 0, 151, 848, 480, mac_64).is_none());
        assert!(mochi_needed_gb(
            0,
            151,
            848,
            480,
            crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB
        )
        .is_none());
        // No budget (off-Mac / probe failed) ⇒ admit.
        assert!(mochi_fit_error("mochi_1", MOCHI_Q4_RESIDENT_BYTES, 151, 848, 480, None).is_none());
    }

    /// `mochi_resident_bytes` must fold the SHARED `text_encoder/` + `vae/` siblings resolved from the
    /// tier dir's PARENT — they are over half the resident footprint, and summing only the tier dir
    /// would under-count by ~9.7 GiB and admit a job that then dies.
    #[test]
    fn mochi_resident_bytes_folds_the_shared_parent_siblings() {
        let root =
            std::env::temp_dir().join(format!("mochi_resident_{}_{}", std::process::id(), line!()));
        // The A6 installed layout: tier dirs as siblings of the shared components.
        for (sub, bytes) in [
            ("q4/transformer", 400_usize),
            ("text_encoder", 300),
            ("vae", 200),
            // A sibling tier that must NOT be counted into the q4 load.
            ("q8/transformer", 999),
        ] {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).unwrap();
        }

        let tier_dir = root.join("q4");
        assert_eq!(
            mochi_resident_bytes(&tier_dir),
            400 + 300 + 200,
            "tier transformer + the shared text_encoder/vae siblings, and NOT the q8 tier"
        );

        // A self-contained dir (the raw snapshot: components UNDER the dir) is summed once, not
        // double-counted via the parent scan.
        let flat = root.join("flat");
        for (sub, bytes) in [
            ("transformer", 400_usize),
            ("text_encoder", 300),
            ("vae", 200),
        ] {
            let dir = flat.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).unwrap();
        }
        assert_eq!(
            mochi_resident_bytes(&flat),
            900,
            "a self-contained snapshot counts its own components exactly once"
        );

        std::fs::remove_dir_all(&root).ok();
    }
}
