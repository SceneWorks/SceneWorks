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

use std::path::Path;
use std::sync::OnceLock;

use gen_core::{
    GenerationMemory, LoadSpec, MemoryBackendRealization, MemoryBudget, MemoryCacheState,
    MemoryConformanceState, MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey,
    MemoryEvidenceVerdict, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryParityContract,
    MemoryParityResult, MemoryProviderContract, MemoryRunContext, MemorySelection, MemoryStrategy,
    OffloadPolicy, PerComponentBytes, TransformerComponent, WeightsSource,
};
use sceneworks_core::memory_calibration::{
    Backend as CalibrationBackend, BundleLoad, CalibrationBinding, EvidenceBundle, EvidenceQuery,
    EvidenceVerdict, Geometry as CalibrationGeometry, LoadShapeKey, StaleEvidenceReason,
    StrategyRung,
};
use serde_json::{Map as JsonObject, Value};

use crate::fit_gate::resolve_offload;
pub(crate) use crate::fit_gate::FitDecision;
use crate::memory_strategy::memory_mode_from_mode_key;
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
}

impl MlxRequestPlan {
    pub(crate) fn for_spec_and_manifest(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
    ) -> Self {
        let media = crate::inference_runtime::media();
        let provider_footprint = media.footprint(engine_id, spec).ok().flatten();
        let activation_anchor_bytes = media.activation_memory_bytes_1024(engine_id).ok().flatten();
        Self::for_spec_and_manifest_with_provider_facts(
            engine_id,
            model_id,
            spec,
            manifest,
            resolved_artifact,
            provider_footprint,
            activation_anchor_bytes,
        )
    }

    /// The platform-neutral core of [`Self::for_spec_and_manifest`]. Production supplies both facts
    /// from the active provider registry; tests may inject the same provider-owned facts when the
    /// host deliberately links no MLX catalog (the default Linux workspace build). Keeping the
    /// filesystem accounting and plan construction here prevents an audit from replacing the live
    /// path with hand-written `asset + headroom` arithmetic.
    #[allow(clippy::too_many_arguments)]
    fn for_spec_and_manifest_with_provider_facts(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
        provider_footprint: Option<PerComponentBytes>,
        activation_anchor_bytes: Option<u64>,
    ) -> Self {
        let (asset_bytes, _, headroom) =
            spec_component_bytes_with_provider_footprint(engine_id, spec, provider_footprint);
        let folded_control_bytes = spec.control.as_ref().map_or(0, weights_source_bytes);
        let folded_adapter_bytes = adapter_source_bytes_for_gate(engine_id, spec);
        let declared_floors = declared_component_floors(engine_id);
        let spec_tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: active_component_floors(declared_floors, spec.quantize),
        };
        let (tier, calibration) = match manifest {
            Some(manifest) => match MlxCalibrationBinding::from_manifest(manifest) {
                Ok(Some(bindings)) => match resolved_artifact {
                    Some(resolved) => match resolved.fixed_artifact_tier.as_deref() {
                        Some(fixed_tier) => {
                            match numeric_tier_for_resolved(fixed_tier, spec_tier, declared_floors)
                            {
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
                    // The manifest opts in, but the resolver could not PROVE which artifact is on
                    // disk. That is a property of the install, not of the opt-in: a receipt written
                    // before download-time tree stamps existed (`backfill_current_receipt` never
                    // writes `artifactTreeStamp`) resolves a perfectly loadable snapshot and still
                    // yields no provenance. Refusing the request outright would strand every such
                    // install with no repair path, so this is a NON-COVERING state and routes to the
                    // established legacy selector — the conservative gate — exactly like every other
                    // one. `Invalid` stays reserved for a malformed opt-in, which is an authoring bug.
                    None => (spec_tier, MlxCalibrationConfig::Unproven),
                },
                Ok(None) => (spec_tier, MlxCalibrationConfig::Absent),
                Err(reason) => (spec_tier, MlxCalibrationConfig::Invalid(reason)),
            },
            None => (spec_tier, MlxCalibrationConfig::Absent),
        };
        // Request budgeting already removes the legacy 2 GiB unified reserve from the available
        // envelope. Keep the remainder of the 4 GiB OS/app reserve fixed, and source only the bare
        // 1024² activation term from the provider's exact-tier measurement. An absent measurement
        // retains the load allowance's conservative activation component.
        let fixed_reserve_bytes = gib_to_bytes(
            (OS_APP_RESERVE_GB - crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB).max(0.0),
        );
        let activation_anchor_bytes = activation_anchor_bytes
            .unwrap_or_else(|| gib_to_bytes((headroom.total_gb - headroom.os_reserve_gb).max(0.0)));
        Self {
            engine_id,
            model_id: model_id.to_owned(),
            tier,
            asset_bytes,
            folded_control_bytes,
            folded_adapter_bytes,
            // This field remains the combined request allowance so the generic formula can hold the
            // fixed slice out before scaling: fixed reserve + provider activation anchor.
            activation_headroom_bytes: activation_anchor_bytes.saturating_add(fixed_reserve_bytes),
            fixed_reserve_bytes,
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
    /// Provider anchors are bare activation measurements. The request planner adds the remaining
    /// fixed OS/app reserve separately, so even an anchor larger than the generic fallback (Lens
    /// dense) keeps its whole measured value in the area term.
    fn generic_total_peak_bytes(&self, geometry: MemoryGeometry) -> u64 {
        self.asset_bytes
            .saturating_add(self.generic_headroom_bytes(geometry))
    }

    /// The non-weights slice of [`Self::generic_total_peak_bytes`]: the fixed OS/app reserve plus
    /// the area-scaled activation transient. Factored out (sc-18096) so the estimate-floor
    /// candidates charge the exact same headroom convention as the resident baseline — same fixed
    /// slice, same anchor, same area law — with only their weights term differing per rung.
    fn generic_headroom_bytes(&self, geometry: MemoryGeometry) -> u64 {
        let megapixel_scale =
            (f64::from(geometry.width) * f64::from(geometry.height) / (1024.0 * 1024.0)).max(1.0);
        let request_scale = megapixel_scale * f64::from(geometry.batch.max(1));
        let fixed_reserve_bytes = self.fixed_reserve_bytes.min(self.activation_headroom_bytes);
        let area_bytes = self.activation_headroom_bytes - fixed_reserve_bytes;
        fixed_reserve_bytes.saturating_add(
            (area_bytes as f64 * request_scale)
                .round()
                .clamp(0.0, u64::MAX as f64) as u64,
        )
    }

    /// Convert the legacy whole-spec estimate into the base-only scalar expected by the additive
    /// component contract. `spec_component_bytes` folds raw control and resident adapter sources
    /// into the legacy scalar. A typed declaration replaces the corresponding raw source bytes with
    /// the provider's load-exact residency. Non-adopting providers retain the legacy scalar.
    ///
    /// A [`gen_core::MemoryComponentResidency::PrecomputedThenEvicted`] declaration deliberately
    /// does NOT move this figure (sc-19721). This is the resident BASELINE's peak: it is the one
    /// candidate that must always cover the widest instant of the whole pipeline, and the precompute
    /// instant — where the evictable sub-stack is fully materialized — is inside it. The eviction is
    /// a steady-state fact and reaches the floor candidates through
    /// [`estimate_floor_weights_bytes`], never this leg.
    fn contract_base_peak_bytes(
        &self,
        legacy_total_peak_bytes: u64,
        contract: &MemoryProviderContract,
    ) -> u64 {
        let declares_control_branch = contract
            .resident_components()
            .iter()
            .any(|component| component.kind == gen_core::MemoryComponentKind::ControlBranch);
        let declares_adapter_stack = contract
            .resident_components()
            .iter()
            .any(|component| component.kind == gen_core::MemoryComponentKind::AdapterStack);
        // Mage's request estimator is provider-specific and already returns the adapter-free base
        // peak, unlike the generic legacy estimator whose asset term contains external adapter
        // files. Do not normalize bytes that were never present or the provider's later additive
        // declaration would cancel to zero.
        let legacy_includes_adapter_sources = !self.engine_id.starts_with("mage_flow");
        legacy_total_peak_bytes
            .saturating_sub(if declares_control_branch {
                self.folded_control_bytes
            } else {
                0
            })
            .saturating_sub(
                if declares_adapter_stack && legacy_includes_adapter_sources {
                    self.folded_adapter_bytes
                } else {
                    0
                },
            )
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn declared_component_floors(engine_id: &str) -> &'static [gen_core::ComponentPrecisionFloor] {
    crate::inference_runtime::media_descriptor(engine_id)
        .map(|descriptor| descriptor.capabilities.component_precision_floors)
        .unwrap_or(&[])
}

#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
fn declared_component_floors(_: &str) -> &'static [gen_core::ComponentPrecisionFloor] {
    // The platform-neutral worker build has no active media registry. Its fit-gate decision remains
    // usable for pure contract tests, but there is no provider declaration to attach to the tier.
    &[]
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
    /// The manifest declares bindings but the resolver could not establish immutable artifact
    /// provenance, so no binding may be trusted for this request. Distinct from [`Self::Absent`]
    /// so telemetry can tell "this model declares no evidence" from "this INSTALL cannot prove
    /// which artifact it holds" — the two demand different follow-up.
    Unproven,
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
        const CALIBRATION_FIELDS: [&str; 19] = [
            "abi",
            "loadShape",
            "fingerprint",
            "sceneWorksRevision",
            "matrixSourceRevision",
            "inferenceRevision",
            "inferenceClosureDigest",
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
            "engagedRungs",
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
        let declared_engaged = match calibration.get("engagedRungs") {
            None => None,
            Some(Value::Array(values)) => Some(
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .and_then(crate::memory_strategy::rung_from_key)
                            .ok_or_else(|| {
                                format!(
                                    "mlx.calibrations[{index}].engagedRungs contains an unsupported rung {value}"
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Some(_) => {
                return Err(format!(
                    "mlx.calibrations[{index}].engagedRungs must be an array"
                ))
            }
        };
        let engaged_rungs =
            crate::memory_strategy::engaged_composition(rung, declared_engaged.as_deref())
                .map_err(|reason| format!("mlx.calibrations[{index}]: {reason}"))?;
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
        let abi = calibration
            .get("abi")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("mlx.calibrations[{index}].abi must be a u32"))?;
        // ABI 2 keys receipts by the typed materialization shape, so the opt-in must declare the
        // shape its receipts were measured under. ABI-1 opt-ins predate the axis; they are stale on
        // the ABI check alone, so the placeholder value is never compared.
        let load_shape = match calibration.get("loadShape").and_then(Value::as_str) {
            Some("eager_materialization") => LoadShapeKey::EagerMaterialization,
            Some("deferred_materialization") => LoadShapeKey::DeferredMaterialization,
            Some(other) => {
                return Err(format!(
                    "mlx.calibrations[{index}].loadShape {other:?} is not a known materialization shape"
                ))
            }
            None if abi >= 2 => {
                return Err(format!(
                    "mlx.calibrations[{index}].loadShape is required at calibration ABI {abi}"
                ))
            }
            None => LoadShapeKey::EagerMaterialization,
        };
        // The selected rung's DECLARED prerequisite graph is deliberately not re-checked here. At
        // the pinned contract revision rung 4's shared prerequisite is
        // `LoadShape::DeferredMaterialization`, and that axis is already owned by
        // `EvidenceBundle::evidence_for`, which degrades a load-shape mismatch to
        // `StaleEvidenceReason::LoadShape` and the legacy selector rather than rejecting the opt-in.
        // The rung-1 edge some providers add for rung 4 is realization-specific
        // (`MemoryProviderContract::additional_prerequisites`) and is enforced by
        // `validate_selection` against the provider contract, which no manifest reader holds.
        let selection_parameters = parse_evidence_parameters(rung, &engaged_rungs, &parameters)
            .map_err(|reason| format!("mlx.calibrations[{index}]: {reason}"))?;
        Ok(Self {
            query: CalibrationBinding {
                abi,
                load_shape,
                fingerprint: text("fingerprint")?,
                scene_works_revision: text("sceneWorksRevision")?,
                matrix_source_revision: text("matrixSourceRevision")?,
                inference_revision: text("inferenceRevision")?,
                inference_closure_digest: text("inferenceClosureDigest")?,
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
    declared_floors: &'static [gen_core::ComponentPrecisionFloor],
) -> Result<MemoryNumericTier, String> {
    let resolved = match tier {
        "q4" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
            component_precision_floors: active_component_floors(
                declared_floors,
                Some(gen_core::Quant::Q4),
            ),
        },
        "q8" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q8),
            component_precision_floors: &[],
        },
        "nvfp4" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Nvfp4),
            component_precision_floors: &[],
        },
        "bf16" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
        "fp32" => MemoryNumericTier {
            precision: gen_core::Precision::Fp32,
            quant: None,
            component_precision_floors: &[],
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

fn active_component_floors(
    declared: &'static [gen_core::ComponentPrecisionFloor],
    selected: Option<gen_core::Quant>,
) -> &'static [gen_core::ComponentPrecisionFloor] {
    match selected {
        Some(selected) if declared.iter().any(|floor| floor.applies_to(selected)) => declared,
        _ => &[],
    }
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
    /// The manifest opted in but the install could not prove its artifact identity.
    NoProvenance,
}

/// Sentinel for routes that carry no calibration record, so no closure can be current against them.
const UNCALIBRATED_CLOSURE: &str = "uncalibrated";

/// Resolves the LIVE compile-closure digest for one `(backend, provider)` lane (sc-17774).
///
/// Production passes `None` and the packaged `config/inference-provider-closures.json` answers. The
/// unit tests inject one so their synthetic lane resolves, because the gate must keep failing closed
/// on a lane nobody declared — an undeclared lane means nobody derived what code its measurements
/// were taken against, and admitting it would be exactly the false green this epic removes.
/// Declaring the fixture in the shipped config instead would put a permanent fiction in the one
/// artifact that has to stay trustworthy.
type ClosureDigestLookup<'a> = &'a dyn Fn(&str, &str) -> Option<String>;

#[derive(Clone, Debug)]
struct VerifiedAdmissionCandidate {
    evidence: MemoryEvidence,
    /// Reserve enforced on this live host and passed to MLX as an absolute process ceiling.
    foreign_reserve_bytes: u64,
    /// Actionable static host boundaries under the captured reserve policy. The stale value uses
    /// the selector's canonical widened peak rather than treating a current-host reserve sum as a
    /// portable recommendation.
    minimum_host_bytes: u64,
    stale_minimum_host_bytes: u64,
    record_id: String,
    /// The provider closure digest this candidate's binding was measured under (sc-17774).
    closure_digest: String,
}

#[derive(Clone, Debug)]
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
    engaged_composition: Vec<MemoryStrategy>,
}

/// One verified measured cell usable as the extrapolation basis for a fitted-curve estimate
/// (sc-18096): same provider, tier, mode, and overlay as the request, artifact-current AND
/// closure-current binding, but a DIFFERENT geometry — the cell the request itself could not be
/// admitted on.
///
/// Closure-current is a deliberate restriction, not an oversight: `MLX_ESTIMATE_MARGIN` (0.10)
/// was derived to cover extrapolation error on top of same-closure re-capture variance
/// (`crates/sceneworks-worker/src/ladder_margin_policy.rs`). A stale-closure record already
/// carries its own 0.05 drift allowance on the MEASURED path; stacking that drift under an
/// extrapolation would spend the estimate margin twice, and no derivation covers the sum — so a
/// stale record may keep serving its own cell behind the stale margin (sc-18095) but may not seed
/// an extrapolated estimate.
///
/// Everything the extrapolation, the binding-phase constraint, and the loaded-provider identity
/// gate need is captured here, so the synthesis step never re-reads the bundle.
#[derive(Clone, Debug)]
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
    pub reference_count: u32,
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
        reference_count: inputs.reference_count,
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
            backend: gen_core::MemoryBackend::Mlx,
            tier,
            load_shape: contract.load_shape,
            mode: memory_mode_from_mode_key(mode),
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
        | LegacyAdmissionReason::StaleBundle
        | LegacyAdmissionReason::NoProvenance => 4,
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

/// Read one binding's exact strategy parameters against the composition it ENGAGES (sc-17728).
///
/// The required set is derived from `engaged`, never from the selected rung's ordinal. Keying it on
/// the ordinal assumed the ladder is always cumulative, which made a provider that implements rung 4
/// while declaring rungs 2 and 3 `Missing` structurally unable to record evidence: it has no honest
/// decode or attention parameter to name. Strictness is unchanged in both directions — an engaged
/// rung must name its parameters, and a rung the composition does not engage must not.
fn parse_evidence_parameters(
    rung: StrategyRung,
    engaged: &[StrategyRung],
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
    let expected_numeric = crate::memory_strategy::required_numeric_parameters(engaged);
    let composition = engaged
        .iter()
        .map(|rung| crate::memory_strategy::rung_key(*rung))
        .collect::<Vec<_>>()
        .join(", ");
    for (key, _) in &expected_numeric {
        if !parameters.contains_key(*key) {
            return Err(format!("{rung:?} engaging [{composition}] requires {key}"));
        }
    }
    for key in KEYS[..4].iter().filter(|key| {
        !expected_numeric
            .iter()
            .any(|(expected, _)| expected == *key)
    }) {
        if parameters.contains_key(*key) {
            return Err(format!("{rung:?} engaging [{composition}] forbids {key}"));
        }
    }
    let engages_transformer = engaged.contains(&StrategyRung::BoundedTransformerResidency);
    // The component scope is validated separately from the numeric parameters: unlike a tile edge or
    // a window size it carries a meaningful DEFAULT (DiT-only), so an engaged rung 4 may leave it
    // unnamed and only an explicitly declared scope is checked. Naming it without engaging its
    // owning rung stays an error. This mirrors gen-core's `validate_selected_parameters`.
    let transformer_window_component = match parameters.get("transformerWindowComponent") {
        None => None,
        Some(Value::String(value)) if engages_transformer => Some(match value.as_str() {
            "dit" => TransformerComponent::Dit,
            "text_encoder" => TransformerComponent::TextEncoder,
            "both" => TransformerComponent::Both,
            other => return Err(format!("unsupported transformerWindowComponent {other:?}")),
        }),
        Some(_) if !engages_transformer => {
            return Err(format!(
                "{rung:?} engaging [{composition}] forbids transformerWindowComponent"
            ))
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
/// `expected_closure_digest` is the LIVE compile-closure digest for `("mlx", plan.engine_id)`
/// (sc-17774) — see [`evidence_admission_route`] for why it is a parameter rather than a lookup
/// performed here.
fn packaged_admission_route(
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
    expected_closure_digest: &str,
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
                estimate_bases: Vec::new(),
                evidence_revision: None,
                process_limit_bytes: None,
                lower_alternative: None,
            });
        }
    };
    evidence_admission_route(
        &bundle,
        plan,
        inputs,
        mode_key,
        budget,
        expected_closure_digest,
    )
}

/// `expected_closure_digest` is the LIVE compile-closure digest for `("mlx", plan.engine_id)`
/// (sc-17774). It is threaded in rather than resolved here so the caller's injected resolver reaches
/// this seam too — the synthetic test lanes are deliberately absent from the shipped closure config,
/// and re-deriving from the packaged table here would silently grade them against `None`.
fn evidence_admission_route(
    bundle: &EvidenceBundle,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
    expected_closure_digest: &str,
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
            estimate_bases: Vec::new(),
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
                estimate_bases: Vec::new(),
                evidence_revision: None,
                process_limit_bytes: None,
                lower_alternative: None,
            })
        }
        MlxCalibrationConfig::Unproven => {
            return Ok(AdmissionRoute {
                path: AdmissionPath::Legacy,
                fallback_reason: Some(LegacyAdmissionReason::NoProvenance),
                evidence: Vec::new(),
                estimate_bases: Vec::new(),
                evidence_revision: None,
                process_limit_bytes: None,
                lower_alternative: None,
            })
        }
        MlxCalibrationConfig::Valid(calibration) => calibration,
        MlxCalibrationConfig::Invalid(_) => unreachable!("invalid opt-in rejected above"),
    };
    // ARTIFACT identity only (sc-18096). Until this story the filter also required
    // `binding.query.inference_closure_digest == expected_closure_digest` — the `StaleIdentity`
    // pre-demotion that made the selector's stale-measured arm (sc-18095) production-unreachable
    // on this lane: a stale binding was routed to `AdmissionPath::Legacy` before any candidate
    // reached `select_strategy`. Currency is a signal, not a gate, so the closure conjunct is
    // retired here: a stale binding proceeds, its candidate carries the digest it was MEASURED
    // under (see `VerifiedAdmissionCandidate::closure_digest`), and the selector grades it behind
    // the widened stale-measured margin. The pre-18096 fear — "a stale binding admitted here has
    // no candidate left to fall back to and kills the request" — no longer holds: refusal now
    // happens only when nothing fits with margins, which is the honest outcome for a stale ladder
    // too.
    //
    // The ARTIFACT conjuncts stay: a binding for different bytes on disk (repository, revision,
    // variant, or path fingerprint) is not a stale measurement of this install, it is a
    // measurement of something else, and remains structurally excluded. `inference_revision`
    // likewise survives on the binding as capture provenance and is deliberately not compared
    // (sc-17774).
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
            estimate_bases: Vec::new(),
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
        .iter()
        .copied()
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
            estimate_bases: collect_estimate_bases(
                bundle,
                plan,
                &identity_matches,
                mode_key,
                overlay,
                request_cell_geometry,
                expected_closure_digest,
            ),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: verified_lower_alternative(
                bundle,
                calibration,
                plan,
                inputs,
                mode_key,
                budget,
                expected_closure_digest,
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
                // The engaged composition is NOT re-checked against the receipt here, and does not
                // need to be: `EvidenceBundle::evidence_for` already requires the binding's exact
                // parameter map to equal a passed sweep case's, so any composition difference among
                // the rungs that OWN parameters (2, 3 and 4) forces a parameter mismatch and never
                // reaches this arm. Rungs 0 and 1 own none, so a difference confined to them cannot
                // change which parameters the binding was obliged to name. The receipt stays
                // authoritative for the composition recorded on the evidence key below.
                let envelope = record.mlx_admission_envelope().ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                    "{} has a verified MLX evidence cell without a complete MLX admission envelope",
                    plan.model_id
                ))
                })?;
                let memory_evidence = MemoryEvidence {
                    key: MemoryEvidenceKey {
                        resolved_route: plan.engine_id.to_owned(),
                        backend: gen_core::MemoryBackend::Mlx,
                        tier: plan.tier,
                        load_shape: crate::memory_strategy::load_shape_from_receipt(
                            record.load_shape,
                        ),
                        mode: memory_mode_from_mode_key(mode_key),
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
                let foreign_reserve_bytes =
                    envelope.foreign_reserve_for_host_bytes(budget.total_bytes);
                let stale_peak_bytes = crate::memory_strategy::stale_widened_peak_bytes(
                    gen_core::MemoryBackend::Mlx,
                    envelope.peak_bytes,
                );
                evidence.push(VerifiedAdmissionCandidate {
                    evidence: memory_evidence,
                    foreign_reserve_bytes,
                    minimum_host_bytes: envelope.required_host_bytes(),
                    stale_minimum_host_bytes: envelope
                        .required_host_bytes_for_peak(stale_peak_bytes),
                    record_id: record.id.clone(),
                    closure_digest: binding.query.inference_closure_digest.clone(),
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
            estimate_bases: collect_estimate_bases(
                bundle,
                plan,
                &identity_matches,
                mode_key,
                overlay,
                request_cell_geometry,
                expected_closure_digest,
            ),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: None,
        });
    }
    let lower_alternative = verified_lower_alternative(
        bundle,
        calibration,
        plan,
        inputs,
        mode_key,
        budget,
        expected_closure_digest,
    );
    Ok(AdmissionRoute {
        path: AdmissionPath::Evidence,
        fallback_reason: None,
        evidence,
        estimate_bases: Vec::new(),
        evidence_revision: None,
        process_limit_bytes: None,
        lower_alternative,
    })
}

/// Collect the verified measured cells a fitted-curve estimate may extrapolate from (sc-18096):
/// artifact-current, closure-CURRENT bindings of this provider and tier whose mode and overlay
/// match the request but whose GEOMETRY does not, resolved to their own verified records at their
/// own geometry. The per-phase peaks ride along so the binding-phase constraint can be applied at
/// synthesis time. See [`MeasuredRungBasis`] for why a stale-closure record is not a legitimate
/// extrapolation basis even though it remains admissible for its own cell.
fn collect_estimate_bases(
    bundle: &EvidenceBundle,
    plan: &MlxRequestPlan,
    identity_matches: &[&MlxCalibrationBinding],
    mode_key: &str,
    overlay: &str,
    request_cell_geometry: CalibrationGeometry,
    expected_closure_digest: &str,
) -> Vec<MeasuredRungBasis> {
    identity_matches
        .iter()
        .filter(|binding| {
            binding.query.inference_closure_digest == expected_closure_digest
                && binding.mode == mode_key
                && binding.overlay == overlay
                && binding.geometry != request_cell_geometry
                // A phase curve extrapolates over output AREA; a different batch or frame count is
                // a different workload shape, not a scalable geometry.
                && binding.geometry.batch == request_cell_geometry.batch
                && binding.geometry.frames == request_cell_geometry.frames
        })
        .filter_map(|binding| {
            let query = EvidenceQuery {
                backend: CalibrationBackend::Mlx,
                model_id: plan.model_id.clone(),
                provider: binding.provider.clone(),
                tier: binding.tier.clone(),
                mode: mode_key.to_owned(),
                overlay: overlay.to_owned(),
                geometry: binding.geometry,
                rung: binding.rung,
                parameters: binding.parameters.clone(),
                calibration: binding.query.clone(),
            };
            let EvidenceVerdict::Verified(record) = bundle.evidence_for(&query) else {
                return None;
            };
            let envelope = record.mlx_admission_envelope()?;
            let predicted = match &record.predicted_peak_bytes {
                sceneworks_core::memory_calibration::RequiredNullable::Value(value) => {
                    value.full()?
                }
                _ => return None,
            };
            Some(MeasuredRungBasis {
                rung: binding.rung,
                parameters: binding.selection_parameters,
                engaged_composition: record
                    .strategy
                    .engaged_rungs
                    .iter()
                    .copied()
                    .map(evidence_strategy)
                    .collect(),
                load_shape: crate::memory_strategy::load_shape_from_receipt(record.load_shape),
                calibration_abi: binding.query.abi,
                calibration_fingerprint: binding.query.fingerprint.clone(),
                geometry: binding.geometry,
                conditioning_peak_bytes: predicted.conditioning,
                denoise_peak_bytes: predicted.denoise,
                decode_peak_bytes: predicted.decode,
                envelope_peak_bytes: envelope.peak_bytes,
                record_id: record.id.clone(),
            })
        })
        .collect()
}

/// One estimate-backed candidate synthesized for an implemented-but-unmeasured rung (sc-18096).
#[derive(Clone, Debug)]
struct SynthesizedEstimate {
    selection: MemorySelection,
    evidence: MemoryEvidence,
    basis: crate::memory_strategy::CandidateBasis,
}

const fn strategy_rung(strategy: MemoryStrategy) -> StrategyRung {
    match strategy {
        MemoryStrategy::Resident => StrategyRung::Resident,
        MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
        MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
        MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency => StrategyRung::BoundedTransformerResidency,
    }
}

/// The canonical measurement phases, in ladder order, used for the binding-phase comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EstimatePhase {
    Conditioning,
    Denoise,
    Decode,
}

/// The phase carrying the peak of a `(conditioning, denoise, decode)` triple. Ties resolve to the
/// LATER phase deterministically; the comparison below only ever contrasts two triples scaled by
/// the same per-phase rule, so tie handling cannot manufacture a flip on its own.
fn binding_phase(conditioning: u64, denoise: u64, decode: u64) -> EstimatePhase {
    let mut phase = EstimatePhase::Conditioning;
    let mut peak = conditioning;
    if denoise >= peak {
        phase = EstimatePhase::Denoise;
        peak = denoise;
    }
    if decode >= peak {
        phase = EstimatePhase::Decode;
    }
    phase
}

/// Fabricated evidence for a synthesized estimate candidate, following the
/// [`generic_mlx_shared_observation`] / [`resident_evidence`] pattern: `ImplementedUnverified`
/// conformance, no observed peak, parity not run — the record claims exactly what an estimate can
/// claim and nothing more. The selector's estimate-scoped eligibility wrap (sc-18096,
/// `memory_strategy::candidate_exclusion`) is what admits it.
#[allow(clippy::too_many_arguments)]
fn estimate_evidence(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    mode: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    selection: MemorySelection,
    predicted_peak_bytes: u64,
    calibration_fingerprint: Option<&str>,
) -> MemoryEvidence {
    MemoryEvidence {
        key: MemoryEvidenceKey {
            resolved_route: contract.provider_id.clone(),
            backend: gen_core::MemoryBackend::Mlx,
            tier,
            load_shape: contract.load_shape,
            mode: memory_mode_from_mode_key(mode),
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
    }
}

/// Bytes the provider declares it drops from INSIDE `asset_facts.transformer_bytes` before the
/// declaring phase reaches steady state — gen-core's
/// [`MemoryComponentKind::TransformerSubStack`] + [`MemoryComponentResidency::PrecomputedThenEvicted`]
/// pair (SC-18665), read through [`MemoryProviderContract::steady_state_transformer_bytes`].
///
/// **Zero for all 23 providers that have not adopted the sub-stack vocabulary**, because the
/// accessor returns `asset_facts.transformer_bytes` unchanged for them. That is what makes this
/// term safe to fold into a shared floor: a non-adopting contract is byte-identical to the
/// pre-sc-19721 arithmetic.
///
/// Deliberately NOT [`MemoryProviderContract::evicted_component_bytes`], which sums the drop over
/// EVERY declared component including auxiliary networks. Those bytes are not inside
/// `transformer_bytes`, so subtracting them from the transformer's own term would remove bytes the
/// transformer never held.
fn intra_transformer_evicted_bytes(contract: &MemoryProviderContract) -> u64 {
    contract
        .asset_facts
        .transformer_bytes
        .saturating_sub(contract.steady_state_transformer_bytes())
}

/// The floor's per-rung WEIGHTS term, derived only from the provider contract's own declarations
/// (sc-18096). Nothing here is a tuned coefficient:
///
/// * `StagedResidency` engaged ⇒ the co-residency drop the rung exists for: the resident working
///   set is the larger of the conditioning stack and everything else, exactly the
///   `staged_weights_gb` split the load-time gate has always used.
/// * `BoundedTransformerResidency` engaged ⇒ the transformer's declared bytes leave the resident
///   floor: the rung windows them, and the window slice plus scratch is carried by the headroom
///   term and the estimate margin, not by a guessed window fraction.
/// * Rungs 2 and 3 bound TRANSIENTS, not weights, so they take no weights reduction here — and
///   deliberately no transient reduction either, because no measured basis for one exists on an
///   unmeasured cell. Their floor equals rung 1's, which keeps them selectable without ever
///   promising an unmeasured saving.
/// * Auxiliary components (control branches, adapter stacks, …) stay resident unless the contract
///   itself declares them `bounded_by` a rung the composition engages.
/// * A declared intra-transformer eviction ([`intra_transformer_evicted_bytes`]) leaves the floor
///   only on the STAGED branch, and only down to the load-exact transformer (sc-19721). Both
///   restrictions are the same rule: **the drop lowers the steady state, not the peak.** The
///   declaring phase still holds the whole sub-stack at the precompute instant — MiniMax-H3's
///   denoise runs 64.56 GB → 38.70 GB *across* it — so the evicted bytes may only be removed from
///   bytes that are provably NOT co-resident with that instant.
///   * Without `StagedResidency` nothing is staged out of it: the conditioning stack, the
///     transformer and the decoder are all charged as one co-residency, and that co-residency
///     includes the instant. Removing anything there would under-charge it by the whole eviction —
///     the OOM direction, and the exact asymmetry the provider's `retained_bytes` declaration is
///     chosen to avoid.
///   * With it engaged, `heavy` is still a lumped charge for the transformer plus every later
///     phase's component (the decoder). Clamping the reduced lump at `transformer_bytes` — the
///     load-exact figure, sub-stack included — keeps the precompute instant covered while letting
///     the drop cancel against the later phases' bytes, which are not resident at that instant.
///     The reduction is therefore `min(evicted, base_bytes − conditioning − transformer)`, never
///     the raw eviction, and this leg is deliberately not an un-lumping of the staged phases: no
///     measured basis for that exists here (epic 18093 owns it).
///   * `BoundedTransformerResidency` still subtracts the load-exact `transformer_bytes`, because
///     that rung windows the whole transformer — the sub-stack is inside it, so a second deduction
///     would double-count the same bytes.
///
/// The auxiliary fold deliberately charges `resident_bytes`, not
/// `MemoryResidentComponent::steady_state_bytes`: an auxiliary network stands beside the base model
/// rather than inside a staged phase, so nothing here establishes that its widest instant is not
/// co-resident with the rest of the floor. No shipped provider declares an evicting auxiliary
/// component today, so the two readings are byte-identical; this comment records which one is meant
/// if one ever does.
fn estimate_floor_weights_bytes(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> u64 {
    let facts = contract.asset_facts;
    let conditioning = facts.conditioning_bytes;
    let staged = engaged.contains(&MemoryStrategy::StagedResidency);
    // The load-exact non-conditioning working set: what the transformer's own phase holds while the
    // evictable sub-stack is still materialized.
    let heavy_load_exact = facts.base_bytes.saturating_sub(conditioning);
    let mut heavy = if staged {
        heavy_load_exact
            .saturating_sub(intra_transformer_evicted_bytes(contract))
            .max(facts.transformer_bytes)
    } else {
        heavy_load_exact
    };
    if engaged.contains(&MemoryStrategy::BoundedTransformerResidency) {
        heavy = heavy.saturating_sub(facts.transformer_bytes);
    }
    let base = if staged {
        conditioning.max(heavy)
    } else {
        conditioning.saturating_add(heavy)
    };
    let auxiliary = contract
        .resident_components()
        .iter()
        .filter(|component| component.kind.is_auxiliary())
        .filter(|component| match component.bounded_by {
            Some(bounding) => !engaged.contains(&bounding),
            None => true,
        })
        .fold(0_u64, |total, component| {
            total.saturating_add(component.resident_bytes)
        });
    base.saturating_add(auxiliary)
}

/// The smallest declared value for every numeric knob the engaged composition requires — the most
/// deeply bounding parameters the provider publishes, which keeps the true runtime transient as
/// far below the floor's unreduced headroom charge as the provider allows. `None` when a required
/// knob has no declared range: such a selection cannot be validated, so no candidate is
/// synthesized for the rung.
fn estimate_floor_parameters(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> Option<gen_core::MemoryStrategyParameters> {
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
    Some(gen_core::MemoryStrategyParameters {
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
        transformer_window_component: None,
    })
}

/// Synthesize estimate-backed candidates for every optimized rung the provider contract marks
/// `Implemented` (sc-18096, epic 18093 R1a). Called only on legacy admission routes — a covered
/// cell is authorized by its exact measured ladder and gets no synthetic sibling.
///
/// Peak source per rung, in preference order:
///
/// 1. **Fitted curve** — a verified measured cell of the same provider/tier/mode/overlay at a
///    different geometry ([`MeasuredRungBasis`]), extrapolated over output area: the conditioning
///    peak is area-flat (text encoding does not grow with the render target) while denoise,
///    decode, and the admission envelope scale by the area ratio, floored at 1.0 so a
///    smaller-than-measured request never predicts below the measurement. Gated by
///    [`crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`]: if the
///    extrapolated triple's binding phase differs from the measured cell's, the fitted candidate
///    is NOT emitted (no per-phase variance re-derivation exists) and the rung falls back to the
///    floor, whose no-measured-basis path the constraint's scope sentence explicitly exempts.
/// 2. **Weights + headroom floor** — [`estimate_floor_weights_bytes`] plus the exact same
///    fixed-reserve + area-scaled headroom the resident baseline charges
///    ([`MlxRequestPlan::generic_headroom_bytes`]).
///
/// The MLX-conservative estimate margin is NOT applied here — the selector owns margin widening
/// (`memory_strategy::select_strategy`), exactly as it owns the sc-18095 stale widening.
#[allow(clippy::too_many_arguments)]
fn synthesize_estimate_ladder(
    contract: &MemoryProviderContract,
    plan: &MlxRequestPlan,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    calibration_fingerprint: Option<&str>,
    bases: &[MeasuredRungBasis],
) -> Vec<SynthesizedEstimate> {
    use crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE;
    use crate::memory_strategy::CandidateBasis;

    let request_area = f64::from(geometry.width) * f64::from(geometry.height);
    let mut synthesized = Vec::new();
    for strategy in MemoryStrategy::ALL {
        if strategy == MemoryStrategy::Resident {
            // The resident baseline candidate already exists on every legacy route.
            continue;
        }
        if !matches!(
            contract.capability(strategy).map(|cap| &cap.support),
            Some(gen_core::MemoryStrategySupport::Implemented)
        ) {
            continue;
        }
        let engaged = contract.engaged_composition(strategy);

        // 1. Fitted-curve basis: the closest measured geometry below the request, else the
        //    smallest above it (whose clamp-at-1.0 scaling degenerates to the measurement itself).
        //    The basis must have been measured under the LOADED provider's exact calibration
        //    identity: a drifted estimator invalidates the measured numbers as an extrapolation
        //    seed, and this is the only gate on legacy routes (the `carries_verified_claim`
        //    demotion never fires without a verified claim on the route). A contract with no
        //    calibration identity gets no fitted candidates at all — fail closed.
        //
        //    The load-shape conjunct compares CONTRACT shape only — deliberately, and unlike the
        //    Evidence-path filter's measured-candidate leg, which also compares `identity
        //    .load_shape` (sc-18251). An estimate-basis candidate is graded downstream by the
        //    estimate wrap of `optimized_eligibility`, which short-circuits at the conformance
        //    gate's `Unverified` BEFORE the identity load-shape comparison ever runs, so the
        //    identity's shape is never consulted for an estimate. Adding the conjunct here would
        //    be stricter than the gate this filter anticipates.
        let fitted = bases
            .iter()
            .filter(|basis| {
                basis.rung == strategy_rung(strategy)
                    && basis.load_shape == contract.load_shape
                    && basis.engaged_composition == engaged
                    && contract.calibration.as_ref().is_some_and(|identity| {
                        identity.abi == basis.calibration_abi
                            && identity.fingerprint == basis.calibration_fingerprint
                    })
            })
            .max_by_key(|basis| {
                let area = u64::from(basis.geometry.width) * u64::from(basis.geometry.height);
                let below = area as f64 <= request_area;
                // Rank every below-request basis above every above-request one; among "below" take
                // the largest area, among "above" the smallest.
                (below, if below { area as i128 } else { -(area as i128) })
            })
            .and_then(|basis| {
                let selection = MemorySelection {
                    strategy,
                    parameters: basis.parameters,
                    tier: plan.tier,
                };
                if contract.validate_selection(&selection).is_err() {
                    return None;
                }
                let measured_area =
                    f64::from(basis.geometry.width) * f64::from(basis.geometry.height);
                let scale = (request_area / measured_area).max(1.0);
                let scaled =
                    |bytes: u64| (bytes as f64 * scale).ceil().clamp(0.0, u64::MAX as f64) as u64;
                let measured_binding = binding_phase(
                    basis.conditioning_peak_bytes,
                    basis.denoise_peak_bytes,
                    basis.decode_peak_bytes,
                );
                let extrapolated_binding = binding_phase(
                    basis.conditioning_peak_bytes,
                    scaled(basis.denoise_peak_bytes),
                    scaled(basis.decode_peak_bytes),
                );
                if ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE
                    && extrapolated_binding != measured_binding
                {
                    // The pinned sc-18094 constraint: the corpus shows a 17.14% per-phase
                    // re-capture spread that no margin in the policy absorbs, so an extrapolation
                    // that moves the request peak onto a different phase than the one measured is
                    // refused rather than margined. The rung falls back to the floor path below.
                    tracing::info!(
                        route = contract.provider_id,
                        backend = "mlx",
                        ?strategy,
                        basis_record = basis.record_id,
                        measured_binding_phase = ?measured_binding,
                        extrapolated_binding_phase = ?extrapolated_binding,
                        "fitted-curve estimate rejected: extrapolation flips the binding phase \
                         (ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE)"
                    );
                    return None;
                }
                let predicted_peak_bytes = scaled(basis.envelope_peak_bytes);
                tracing::info!(
                    route = contract.provider_id,
                    backend = "mlx",
                    ?strategy,
                    basis_record = basis.record_id,
                    basis_geometry = format!("{}x{}", basis.geometry.width, basis.geometry.height),
                    raw_peak_bytes = predicted_peak_bytes,
                    area_scale = scale,
                    "synthesized fitted-curve estimate candidate from a measured cell"
                );
                Some(SynthesizedEstimate {
                    selection,
                    evidence: estimate_evidence(
                        contract,
                        plan.tier,
                        mode_key,
                        overlay,
                        geometry,
                        selection,
                        predicted_peak_bytes,
                        calibration_fingerprint,
                    ),
                    basis: CandidateBasis::EstimateFittedCurve,
                })
            });
        if let Some(candidate) = fitted {
            synthesized.push(candidate);
            continue;
        }

        // 2. Weights + headroom floor — no measured basis, so the binding-phase constraint does
        //    not gate it (scope sentence on the constraint's doc).
        let Some(parameters) = estimate_floor_parameters(contract, &engaged) else {
            continue;
        };
        let selection = MemorySelection {
            strategy,
            parameters,
            tier: plan.tier,
        };
        if contract.validate_selection(&selection).is_err() {
            continue;
        }
        let predicted_peak_bytes = estimate_floor_weights_bytes(contract, &engaged)
            .saturating_add(plan.generic_headroom_bytes(geometry));
        tracing::info!(
            route = contract.provider_id,
            backend = "mlx",
            ?strategy,
            raw_peak_bytes = predicted_peak_bytes,
            "synthesized weights+headroom floor estimate candidate"
        );
        synthesized.push(SynthesizedEstimate {
            selection,
            evidence: estimate_evidence(
                contract,
                plan.tier,
                mode_key,
                overlay,
                geometry,
                selection,
                predicted_peak_bytes,
                calibration_fingerprint,
            ),
            basis: CandidateBasis::EstimateFloor,
        });
    }
    synthesized
}

/// Select the largest strictly lower, same-aspect geometry backed by a current exact record that
/// fits the live host boundary. This is the only source for a named refusal alternative: no formula,
/// interpolation, tier heuristic, or aspect-ratio rewrite is admitted.
///
/// "Current" is `expected_closure_digest` (sc-17774), and this filter is the ONLY thing enforcing it
/// on this path: the alternative never becomes a `Candidate`, so `memory_strategy` never grades it.
/// The conjunct here used to be the inference pin, which named an alternative geometry the very next
/// request would refuse for the same staleness — advice the gate itself would not honour.
fn verified_lower_alternative(
    bundle: &EvidenceBundle,
    calibration: &MlxCalibrationSet,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
    expected_closure_digest: &str,
) -> Option<VerifiedGeometryAlternative> {
    let overlay = inputs.overlay.as_deref().unwrap_or("none");
    let requested_width = u64::from(inputs.width);
    let requested_height = u64::from(inputs.height);
    calibration
        .bindings
        .iter()
        .filter(|binding| {
            binding.query.inference_closure_digest == expected_closure_digest
                && binding.provider == plan.engine_id
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
            let strategy = evidence_strategy(record.strategy.rung);
            (envelope.fits_scaled_host_bytes(budget.total_bytes)
                && envelope.peak_bytes <= effective)
                .then_some(VerifiedGeometryAlternative {
                    geometry: binding.geometry,
                    calibration_abi: binding.query.abi,
                    calibration_fingerprint: binding.query.fingerprint.clone(),
                    load_shape: crate::memory_strategy::load_shape_from_receipt(record.load_shape),
                    strategy,
                    engaged_composition: record
                        .strategy
                        .engaged_rungs
                        .iter()
                        .copied()
                        .map(evidence_strategy)
                        .collect(),
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
    expected_closure_digest: &str,
) -> Option<CalibrationGeometry> {
    verified_lower_alternative(
        bundle,
        calibration,
        plan,
        inputs,
        mode_key,
        budget,
        expected_closure_digest,
    )
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
    closure_digests: Option<ClosureDigestLookup<'_>>,
) -> WorkerResult<MlxRequestEvaluation> {
    use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};

    // Component precision floors are a provider property, not a manifest guess. Bind them only
    // after the concrete generator is loaded, then use that tier for every evidence/cache identity
    // in this request so uniform-q4 measurements cannot authorize a mixed-precision provider.
    let declared_floors = generator
        .descriptor()
        .capabilities
        .component_precision_floors;
    let provider_floors = active_component_floors(declared_floors, plan.tier.quant);
    let mut effective_plan;
    let plan = if plan.tier.component_precision_floors == provider_floors {
        plan
    } else {
        effective_plan = plan.clone();
        effective_plan.tier.component_precision_floors = provider_floors;
        &effective_plan
    };

    let geometry = request_geometry(inputs);
    let (mode, mode_key) = request_mode(&inputs.mode);
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
        // PR #395 (asset-facts unification) requires base_bytes to equal the sum of the base
        // component bytes. The legacy fallback has only the summed on-disk figure — no per-component
        // split — so carry the whole sum on the transformer axis rather than fail conformance.
        fallback_contract.asset_facts.transformer_bytes = plan.asset_bytes;
        &fallback_contract
    };
    if plan.engine_id.starts_with("mage_flow")
        && inputs.adapter_count > 0
        && !contract.resident_components().iter().any(|component| {
            component.kind == gen_core::MemoryComponentKind::AdapterStack
                && component.resident_bytes > 0
        })
    {
        return Err(WorkerError::InvalidPayload(format!(
            "{} request includes {} adapter(s), but the loaded provider did not declare \
             load-exact adapter residency; refusing an unbounded MLX request",
            plan.engine_id, inputs.adapter_count
        )));
    }
    let calibration_fingerprint = contract
        .calibration
        .as_ref()
        .map(|identity| identity.fingerprint.as_str());
    let calibration_abi = contract
        .calibration
        .as_ref()
        .map_or(0, |identity| identity.abi);
    // sc-17774: the LIVE closure for the provider being admitted, resolved once and used by BOTH
    // currency seams — the admission filter that decides which bindings are still measurements of
    // this code, and the selector comparison below. The constant this replaces was read off the
    // first candidate's own evidence, so the gate compared candidates against themselves and could
    // never see a stale one. `unwrap_or_default` fails CLOSED — an undeclared provider yields an
    // empty expectation that no real 64-hex digest matches.
    let live_closure_digest = closure_digests
        .map_or_else(
            || sceneworks_core::memory_calibration::packaged_closure_digest("mlx", plan.engine_id),
            |lookup| lookup("mlx", plan.engine_id),
        )
        .unwrap_or_default();
    let mut admission = if plan.tier.component_precision_floors.is_empty() {
        match evidence_bundle {
            Some(bundle) => evidence_admission_route(
                bundle,
                plan,
                inputs,
                mode_key,
                budget,
                &live_closure_digest,
            )?,
            None => packaged_admission_route(plan, inputs, mode_key, budget, &live_closure_digest)?,
        }
    } else {
        // Persisted calibration bindings currently identify only the coarse tier token (for
        // example `q4`). They cannot distinguish uniform q4 from a provider whose descriptor keeps
        // selected components at q8, so they must not be relabeled with the live descriptor floors.
        // Until the on-disk evidence schema carries the floors explicitly, fail closed to the
        // provider's conservative resident estimate.
        AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::StaleIdentity),
            evidence: Vec::new(),
            estimate_bases: Vec::new(),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: None,
        }
    };
    // sc-18101: a candidate whose MATERIALIZATION SHAPE the loaded provider does not use is not a
    // measurement of this load, and must be dropped BEFORE the selector — but dropping it must not
    // take the route's usable siblings with it.
    //
    // `MemoryCalibrationIdentity` has three fields — `abi`, `fingerprint`, `load_shape` — and
    // gen-core's `optimized_eligibility` rejects an OPTIMIZED candidate whose `key.load_shape`
    // disagrees with either `contract.load_shape` or `identity.load_shape`, returning
    // `FingerprintMismatch`. (`Resident` is exempt: `optimized_eligibility` returns `Ok(())` for it
    // before the shape comparison, because a resident cell engages no optimized rung whose
    // materialization the shape could change. The filter below mirrors that exemption exactly —
    // see its own comment.) The demotion below compares only the first two, which left a hole with
    // nothing behind it: a cell
    // whose records were all captured under the other shape reached `AdmissionPath::Evidence`, lost
    // every candidate inside `select_strategy`, and refused the request outright with "no
    // structurally admissible MLX memory strategy" — because estimate synthesis runs only on the
    // Legacy route.
    //
    // That hole shipped. `mlx:qwen_image` q8 1024² was captured `eager_materialization`, while
    // `image_jobs::apply_measured_mlx_load_shape` forces `DeferredMaterialization` on every
    // `qwen_image` directory load, so the flagship q8 route hard-refused its most common geometry on
    // a 128 GiB machine. Measured on real weights at this commit and at the epic's base commit: the
    // base commit ADMITTED (its `evidence_admission_route` closure conjunct, retired by sc-18096,
    // pre-demoted the cell to Legacy long before eligibility ran), so this is a REGRESSION this epic
    // introduced. See `docs/epic-18093-end-to-end-validation-sc-18101.md`.
    //
    // Why a per-candidate FILTER and not a whole-route demotion: a route legitimately carries
    // bindings for several rungs captured under DIFFERENT shapes — `qwen_image` q8 ships a
    // `bounded_attention` cell measured eager and a `bounded_transformer_residency` cell measured
    // deferred. Demoting the whole route when any sibling mismatches would throw away the candidate
    // that DOES match and silently downgrade a calibrated request to an estimate, which is its own
    // regression. Filtering keeps every usable measurement and degrades only when nothing is left.
    //
    // sc-18251: the same hole class existed on the two structural legs the sc-18101 filter did not
    // mirror. A promoted binding whose captured `engaged_composition` disagrees with the live
    // contract's (an engagement edge grew or was excluded since capture), or whose parameters no
    // longer pass the live `contract.validate_selection` (a declared range narrowed), still entered
    // `AdmissionPath::Evidence`, lost every candidate inside `select_strategy`
    // (`CompositionMismatch` / `Invalid`), and hard-refused via `Selection::Unverified` — where
    // pre-epic code degraded to legacy first (the retired `StaleIdentity` closure pre-demotion
    // caught every drifted binding before eligibility ran). The filter now mirrors those legs too,
    // so composition drift and parameter-range narrowing degrade to the estimate ladder exactly
    // like a shape mismatch.
    if admission.path == AdmissionPath::Evidence {
        // A candidate is USABLE only if the selector could actually reach it: it must have been
        // measured under a shape the downstream eligibility gate will accept for its rung, its
        // captured engaged composition must still be the loaded contract's canonical set for that
        // rung, and the selection it authorizes must still pass the loaded contract's own
        // `validate_selection`. All three are properties of the LOADED PROVIDER, not of the
        // request, and all three are checked downstream where the only outcome left is a refusal
        // (`FingerprintMismatch`, `CompositionMismatch`, and `Invalid` respectively — and an
        // unimplemented rung silently, since `select_strategy` skips it without even recording an
        // exclusion, so that request dies with a bare `Missing`).
        //
        // The shape test MIRRORS `optimized_eligibility` rather than tightening it. That gate
        // short-circuits `Ok(())` for `MemoryStrategy::Resident` before it ever compares load
        // shapes, so a resident cell measured under the other shape is one the gate ACCEPTS.
        // Dropping it here would make this filter stricter than the thing it exists to anticipate,
        // and would silently discard a usable measurement — the same failure mode as the whole-route
        // demotion this filter deliberately avoids. Reachable in the shipped corpus: `qwen_image`
        // bf16 carries a resident binding captured eager against a production deferred load.
        let usable = |candidate: &VerifiedAdmissionCandidate| {
            let evidence = &candidate.evidence;
            // The two conjuncts mirror `optimized_eligibility`'s pair literally, but they are NOT
            // independently observable (sc-18251 review): gen-core's `conformance_errors` requires
            // `calibration.load_shape == contract.load_shape`, and `select_strategy` refuses every
            // candidate on a non-conformant contract with `Unverified(Invalid)` before grading a
            // single one — so on any contract where the pair could split, the request refuses no
            // matter what this filter does. The pair is effectively one comparison; both spellings
            // are kept so the filter stays a literal mirror of the gate. The premise is pinned by
            // `gen_core_forbids_a_contract_identity_load_shape_split`: if a pin bump ever legalizes
            // the split, that test reds and the identity conjunct needs its own fixture arm.
            let shape_agrees = !evidence.key.strategy.is_optimized()
                || contract.calibration.as_ref().is_some_and(|identity| {
                    evidence.key.load_shape == contract.load_shape
                        && evidence.key.load_shape == identity.load_shape
                });
            // sc-18251: COMPOSITION. `optimized_eligibility`'s structural prefix rejects a
            // candidate whose captured `engaged_composition` is not the loaded contract's canonical
            // set for its rung (`CompositionMismatch`) — and unlike the shape leg above there is NO
            // Resident exemption to mirror: both of the gate's composition checks run BEFORE its
            // `is_optimized` short-circuit, pinned against the pinned gen-core by
            // `gen_core_rejects_a_resident_cell_whose_composition_disagrees`. The gate's
            // canonical-form check (`Invalid` for an empty or unsorted captured set) needs no
            // separate conjunct: `contract.engaged_composition` walks `MemoryStrategy::ALL` in
            // order, so equality forces the captured set into canonical form — except when both
            // sides are empty, which requires the selected rung itself to be non-`Implemented`,
            // and `selection_valid` below drops exactly that candidate.
            let composition_agrees = evidence.key.engaged_composition
                == contract.engaged_composition(evidence.key.strategy);
            // sc-18251: SELECTION. `select_strategy` runs every candidate through the loaded
            // contract's own `validate_selection` (`memory_strategy::candidate_exclusion`),
            // excluding with `Invalid` a candidate whose rung the provider no longer declares
            // `Implemented`, whose prerequisite edges the live contract no longer satisfies, or
            // whose captured parameters fall outside the provider's live declared ranges — a
            // narrowed range strands the old measurement. The downstream check applies to every
            // rung, Resident included, so no exemption is mirrored here either. Consulting the
            // gate's own predicate (rather than a resembling re-implementation) also subsumes the
            // previous standalone `Implemented` conjunct: `validate_selection` fails for an
            // undeclared or non-`Implemented` rung before it looks at parameters.
            let selection_valid = contract
                .validate_selection(&MemorySelection {
                    strategy: evidence.key.strategy,
                    parameters: evidence.key.parameters,
                    tier: evidence.key.tier,
                })
                .is_ok();
            shape_agrees && composition_agrees && selection_valid
        };
        let retained = admission.evidence.iter().filter(|c| usable(c)).count();
        if retained != admission.evidence.len() {
            tracing::info!(
                route = plan.engine_id,
                backend = "mlx",
                dropped = admission.evidence.len() - retained,
                retained,
                contract_load_shape = ?contract.load_shape,
                "dropped measured candidates the loaded provider cannot serve (materialization \
                 shape, engaged composition, or selection validity)"
            );
            admission.evidence.retain(usable);
        }
        if admission.evidence.is_empty() {
            // Nothing measured survives. Degrade to the estimate ladder rather than refuse: a
            // calibration identity that does not match the loaded provider is a reason to stop
            // CLAIMING the measurement, not a reason to deny service. The estimate ladder is
            // strictly more capable than the pre-epic resident-only freeze it replaces.
            admission = AdmissionRoute {
                path: AdmissionPath::Legacy,
                fallback_reason: Some(LegacyAdmissionReason::StaleIdentity),
                evidence: Vec::new(),
                estimate_bases: Vec::new(),
                evidence_revision: None,
                process_limit_bytes: None,
                lower_alternative: None,
            };
        }
    }
    // A named refusal alternative is advice the next request must be able to honour, so it is held
    // to the same shape agreement — but it is only advice, so a mismatch drops the alternative
    // rather than the route.
    if admission
        .lower_alternative
        .as_ref()
        .is_some_and(|alternative| {
            contract.calibration.as_ref().map_or(true, |identity| {
                alternative.load_shape != identity.load_shape
            })
        })
    {
        admission.lower_alternative = None;
    }
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
                .map_or(true, |alternative| {
                    identity.abi == alternative.calibration_abi
                        && identity.fingerprint == alternative.calibration_fingerprint
                        && contract.engaged_composition(alternative.strategy)
                            == alternative.engaged_composition
                })
        })
    {
        admission = AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::StaleIdentity),
            evidence: Vec::new(),
            estimate_bases: Vec::new(),
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
    // The caller estimates the base-model pipeline. Let the provider's canonical contract seam add
    // any separately declared auxiliary networks before either fit selection or warm-cache credit.
    // Exact evidence already describes the whole request peak and therefore remains authoritative.
    let contract_base_peak_bytes = plan.contract_base_peak_bytes(total_peak_bytes, contract);
    let base_prediction = generator.predicted_memory_peak_from_base(contract_base_peak_bytes);
    let evidence_peak_bytes = admission
        .evidence
        .iter()
        .map(|candidate| candidate.evidence.predicted_peak_bytes)
        .max();
    let modeled_peak_bytes =
        evidence_peak_bytes.unwrap_or_else(|| base_prediction.predicted_peak_bytes());

    // The modeled peak is a complete-pipeline peak, while the live budget is incremental from the
    // process's current state. Only bytes above the cache-recorded pre-load external baseline, and
    // no more than the provider-declared total resident envelope, may be credited as already
    // present. Unrelated process allocations therefore remain charged on the available side.
    //
    // sc-19721: `total_resident_bytes()` stays the LOAD-EXACT envelope on a provider that declares
    // an eviction. It is a ceiling on a credit, and the credit is already floored by what the
    // process has actually committed — so a provider caught before its precompute-and-evict is
    // credited for what it really holds, while one caught after is bounded by `committed_bytes`
    // anyway. Lowering the ceiling to the steady state would only refuse credit that is genuinely
    // resident.
    let attributable_resident_bytes = budget
        .committed_bytes
        .saturating_sub(external_committed_bytes)
        .min(contract.total_resident_bytes());
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
        modeled_peak_bytes.saturating_sub(attributable_resident_bytes)
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
    // sc-18096: on a legacy route — the exact request cell has no measured evidence — synthesize
    // estimate-backed candidates for every implemented optimized rung, so the full ladder is
    // selectable behind the estimate margin instead of freezing to the resident baseline and
    // refusing. A covered (`Evidence`) route gets none: its measured ladder is authoritative.
    let synthesized_estimates = if admission.path == AdmissionPath::Legacy {
        synthesize_estimate_ladder(
            contract,
            plan,
            mode_key,
            inputs.overlay.as_deref(),
            geometry,
            calibration_fingerprint,
            &admission.estimate_bases,
        )
    } else {
        Vec::new()
    };
    let mut selections = Vec::new();
    let mut evidence = Vec::new();
    // Index-aligned with `evidence`, and pushed at the same sites. Each entry is the closure the
    // candidate was MEASURED under: a calibrated candidate carries its binding's digest, while the
    // resident baseline and caller-supplied `additional_evidence` are live estimates with no record
    // behind them and carry the live digest, because there is nothing there for currency to
    // invalidate.
    let mut candidate_digests: Vec<&str> = Vec::new();
    // Index-aligned basis axis (sc-18096): synthesized candidates carry their estimate basis, the
    // resident baseline is the rung-0 weights+headroom floor, and everything measured stays
    // `Measured`. Pushed at the same sites as `evidence` for the same fail-open reason as the
    // digests above.
    let mut candidate_bases: Vec<crate::memory_strategy::CandidateBasis> = Vec::new();
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
            // sc-18096: this pre-check runs against the candidate's CAPTURED foreign reserve,
            // which the selector's uniform zero-reserve Evidence budget cannot carry, so a stale
            // candidate must be graded here at the same widened ceiling the selector admits it at
            // — via the selector's own policy function, not a re-derived margin.
            let graded_peak_bytes = if candidate.closure_digest == live_closure_digest {
                exact.predicted_peak_bytes
            } else {
                crate::memory_strategy::stale_widened_peak_bytes(
                    gen_core::MemoryBackend::Mlx,
                    exact.predicted_peak_bytes,
                )
            };
            if candidate_budget
                .effective_gb()
                .is_some_and(|available| graded_peak_bytes as f64 / BYTES_PER_GIB <= available)
            {
                selections.push(MemorySelection {
                    strategy: exact.key.strategy,
                    parameters: exact.key.parameters,
                    tier: exact.key.tier,
                });
                evidence.push(exact);
                candidate_digests.push(candidate.closure_digest.as_str());
                candidate_bases.push(crate::memory_strategy::CandidateBasis::Measured);
            }
        }
        if evidence.is_empty() {
            let minimum_required_host = admission
                .evidence
                .iter()
                .map(|candidate| {
                    // Same stale-aware grading as the pre-check above, expressed as the smallest
                    // host that satisfies the reserve policy. The current-host enforced sum is a
                    // useful diagnostic but not a portable minimum: the reserve changes when the
                    // host capacity changes.
                    if candidate.closure_digest == live_closure_digest {
                        candidate.minimum_host_bytes
                    } else {
                        candidate.stale_minimum_host_bytes
                    }
                })
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
        use crate::memory_strategy::CandidateBasis;

        let capacity = 1 + additional_evidence.len() + synthesized_estimates.len();
        selections.reserve(capacity);
        evidence.reserve(capacity);
        candidate_digests.reserve(capacity);
        candidate_bases.reserve(capacity);
        selections.push(resident_selection);
        evidence.push(&resident);
        candidate_digests.push(live_closure_digest.as_str());
        // The resident baseline IS the rung-0 weights+headroom floor estimate (sc-18096): its
        // peak source is unchanged, but it is now graded behind the estimate margin like every
        // other unmeasured candidate instead of at its raw guess.
        candidate_bases.push(CandidateBasis::EstimateFloor);
        selections.extend(additional_evidence.iter().map(|item| MemorySelection {
            strategy: item.key.strategy,
            parameters: item.key.parameters,
            tier: item.key.tier,
        }));
        evidence.extend(additional_evidence);
        candidate_digests.extend(
            additional_evidence
                .iter()
                .map(|_| live_closure_digest.as_str()),
        );
        candidate_bases.extend(additional_evidence.iter().map(|_| CandidateBasis::Measured));
        for estimate in &synthesized_estimates {
            selections.push(estimate.selection);
            evidence.push(&estimate.evidence);
            candidate_digests.push(live_closure_digest.as_str());
            candidate_bases.push(estimate.basis);
        }
    }
    // The digests are carried from the push sites rather than recovered by searching
    // `admission.evidence` for a matching `MemoryEvidenceKey`. That search was how this read first,
    // and it FAILED OPEN: a miss fell back to the live digest, which is exactly the value the gate
    // compares against, so any candidate the search could not place became automatically current.
    // Keys are also not unique enough to be a lookup key in principle. Pushing the digest alongside
    // the evidence removes the failure mode instead of arguing it cannot happen.
    debug_assert_eq!(evidence.len(), candidate_digests.len());
    debug_assert_eq!(evidence.len(), candidate_bases.len());
    let candidates = selections
        .iter()
        .zip(evidence)
        .zip(candidate_digests.iter().zip(&candidate_bases))
        .map(
            |((selection, evidence), (closure_digest, basis))| Candidate {
                selection: *selection,
                evidence,
                closure_digest,
                basis: *basis,
            },
        )
        .collect::<Vec<_>>();
    let selection = crate::memory_strategy::select_strategy(
        RequestScope {
            resolved_route: plan.engine_id,
            backend: "mlx",
            tier: plan.tier,
            mode: mode_key,
            overlay: inputs.overlay.as_deref(),
            geometry,
            expected_closure_digest: &live_closure_digest,
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
        // sc-18096 retired the "no measured evidence ⇒ refuse" meaning of this arm: every
        // implemented rung of a legacy route now carries an estimate-backed candidate, so the
        // selector only lands here when the evidence is STRUCTURALLY invalid (contract
        // conformance errors, composition mismatch, an unresolvable budget, or a rung whose
        // declared parameters cannot form a valid selection). That remains a refusal — the other
        // permitted refusal, "no rung fits with margins", is the `Reject` arm above.
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
                "{} request {}x{} count {} has no structurally admissible MLX memory strategy \
                 ({reason:?}); refusing to enter MLX's process-terminating allocation \
                 path{alternative}",
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
        let graded_peak_bytes = if candidate.closure_digest == live_closure_digest {
            evidence.predicted_peak_bytes
        } else {
            crate::memory_strategy::stale_widened_peak_bytes(
                gen_core::MemoryBackend::Mlx,
                evidence.predicted_peak_bytes,
            )
        };
        needed_gb = graded_peak_bytes.saturating_add(reserve) as f64 / BYTES_PER_GIB;
        available_gb = selected_budget.effective_gb().unwrap_or(0.0);
        budget.reserved_headroom_bytes = reserve;
        process_limit_bytes = Some(budget.total_bytes.saturating_sub(reserve));
        selected_record_id = Some(candidate.record_id.clone());
        evidence.predicted_peak_bytes
    } else if let Some(estimate) = synthesized_estimates.iter().find(|estimate| {
        estimate.selection.strategy == selection.strategy
            && estimate.selection.parameters == selection.parameters
            && estimate.selection.tier == selection.tier
    }) {
        // sc-18096: a synthesized deep rung was selected. The run context's incremental demand is
        // that rung's raw estimate, not the resident baseline's — the whole point of the rung is a
        // smaller working set. Warm-resident credit applies the same way as the baseline arm.
        estimate
            .evidence
            .predicted_peak_bytes
            .saturating_sub(attributable_resident_bytes)
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
            load_shape: contract.load_shape,
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
/// (sc-10840 — sd3/sana/flux/flux2/chroma/ideogram/kolors/anima/boogu alongside the earlier
/// sdxl/z-image/qwen/lens/krea families) is covered the moment its descriptor advertises the bit, with
/// no lockstep edit here. Providers that stage unconditionally (for example Bernini) deliberately
/// leave the selectable-control bit `false`; an engine that cannot benefit from staging (for example
/// SenseNova's fused MoT, `footprint` te=0) does the same. Neither is offered a no-op `Sequential`
/// selection.
///
/// This is a pre-load, weights-free registry lookup (`(descriptor)()` allocates no tensors), the same
/// query shape the worker already uses for family/guidance/quant capability advertisement and the
/// analogous `ProviderRegistry::footprint` size seam (sc-10894). An id with no registered generator — or a
/// registered one that does not advertise the bit — yields `false` (the safe default: never select a
/// residency policy the provider won't honor). Sees exactly the providers the selected runtime bundle
/// carries: MLX providers are explicitly anchored on macOS, while the CUDA bundle exposes its explicit
/// Candle catalog. The same query is shared by the MLX fit gate (sc-10840) and Candle fit gate
/// (sc-12130), so adding a truthful provider capability needs no worker allowlist edit.
/// **Two descriptor bits, one question (sc-19721).** gen-core split the attestation in two, and this
/// gate's question is the disjunction:
///
/// * `supports_sequential_offload` — the SELECTABLE [`OffloadPolicy::Sequential`] is honoured.
/// * `unconditionally_engages_staged_residency` — the provider stages eligible components through a
///   load/use/drop lifecycle on EVERY generation, with or without a policy request.
///
/// Either one makes "peak is the dominant component, not the sum" true, which is the only thing this
/// gate reads the bit for. The pair is not redundant: MLX Bernini holds no component weights on the
/// generator at all, so it never had a selectable control to honour — it advertised
/// `supports_sequential_offload: true` purely to reach this gate, and inference corrected that to the
/// second bit. Reading only the first would have charged Bernini the SUM of the Qwen2.5-VL planner,
/// the UMT5-XXL encoder and the two MoE experts — a co-residency `generate_impl` never creates — and
/// refused a model that fits, silently, on a pin bump. The `false` default still means "no
/// attestation", so an unwired engine (sensenova's fused MoT) is still never offered `Sequential`.
///
/// Requesting `Sequential` from a provider that stages unconditionally but exposes no selectable
/// control is safe in the direction that matters: the policy is *advisory* (gen-core treats an
/// unwired request as `Resident`, never an error), and here the staged behaviour the prediction
/// assumes is what the provider physically does regardless.
pub(crate) fn engine_supports_sequential(engine_id: &str) -> bool {
    crate::inference_runtime::media()
        .generators()
        .find(|reg| (reg.descriptor)().id == engine_id)
        .is_some_and(|reg| {
            let capabilities = (reg.descriptor)().capabilities;
            capabilities.supports_sequential_offload
                || capabilities.unconditionally_engages_staged_residency
        })
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
/// size AND the outsized transient must be modeled for this bf16/MXFP4 source. (A blanket bf16 expansion factor
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

fn adapter_source_bytes_for_gate(engine_id: &str, spec: &LoadSpec) -> u64 {
    adapter_source_bytes_for_gate_where(engine_id, spec, |_| true)
}

fn external_adapter_source_bytes_for_gate(engine_id: &str, spec: &LoadSpec) -> u64 {
    adapter_source_bytes_for_gate_where(engine_id, spec, |path| match &spec.weights {
        WeightsSource::Dir(root) => !path.starts_with(root),
        WeightsSource::File(_) => true,
    })
}

fn adapter_source_bytes_for_gate_where(
    engine_id: &str,
    spec: &LoadSpec,
    include: impl Fn(&Path) -> bool,
) -> u64 {
    let source_bytes = spec.adapters.iter().fold(0_u64, |total, adapter| {
        if !include(&adapter.path) {
            return total;
        }
        let bytes = match std::fs::metadata(&adapter.path) {
            Ok(metadata) if metadata.is_dir() => sum_safetensors_bytes(&adapter.path),
            Ok(metadata)
                if adapter
                    .path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    == Some("safetensors") =>
            {
                metadata.len()
            }
            _ => 0,
        };
        total.saturating_add(bytes)
    });
    if matches!(engine_id, "wan_vace" | "wan2_2_vace_fun_14b") {
        return 0;
    }
    if !matches!(
        engine_id,
        "wan2_2_ti2v_5b" | "wan2_2_t2v_14b" | "wan2_2_i2v_14b"
    ) {
        return source_bytes;
    }

    // Wan dense loads fold factors into the mutable base before optional load-time quantization, so
    // the adapter files do not remain independently resident. A pre-packed snapshot declares its
    // quantization in the root config and installs adapters as forward-time residuals instead.
    let prepacked = spec.quantize.is_none()
        && matches!(&spec.weights, WeightsSource::Dir(root) if packed_quant_bits(root, "").is_some());
    if prepacked {
        source_bytes
    } else {
        0
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
            weights_floor_load_admission(total_bytes, te_bytes, budget, sequential_capable)
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
/// folds in the [`weights_floor_load_admission`] floor before honoring a reject.
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
        component_precision_floors: &[],
    };
    let geometry = MemoryGeometry {
        width: 1,
        height: 1,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier,
    };
    let evidence = MemoryEvidence {
        key: MemoryEvidenceKey {
            resolved_route: "generic_mlx_cold_load".into(),
            backend: gen_core::MemoryBackend::Mlx,
            tier,
            // The generic estimator models the historical bulk cold load; no generic route
            // defers transformer materialization.
            load_shape: gen_core::LoadShape::EagerMaterialization,
            mode: memory_mode_from_mode_key("image_generation"),
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
            // This route is a generic cold-load estimate with no calibration record behind it, so
            // there is no measured closure to be current against. Both sides carry the same
            // sentinel, which states that plainly instead of naming a revision nothing was measured
            // at (the constant here used to be a frozen inference SHA, which implied otherwise).
            expected_closure_digest: UNCALIBRATED_CLOSURE,
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
            closure_digest: UNCALIBRATED_CLOSURE,
            // The generic cold-load estimate is exactly a weights+headroom floor (sc-18096).
            basis: crate::memory_strategy::CandidateBasis::EstimateFloor,
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

/// LOAD-time weights-floor admission (formerly `legacy_admission_override`, and before that
/// `weights_fit_floor`).
///
/// sc-18096 retired this function's old "Decision 1 / Decision 2 transition freeze" meaning — the
/// rule that unmeasured cells "remain byte-for-byte legacy … until that cell's calibration story
/// opts it into evidence" is gone: the REQUEST-scoped gate now synthesizes estimate-backed
/// candidates for every implemented rung, so an unmeasured cell is no longer frozen to
/// resident/sequential behavior. What survives, byte-for-byte (including the 8 GiB q4 guard), is
/// the load-time floor itself: a spec whose bare staged weights fit under the legacy ceiling is
/// ADMITTED for loading even when the peak-based prediction (weights + flat headroom) rejects,
/// because bounding the render transient is the request gate's job, not the loader's. The
/// remaining honest reject is a spec whose staged WEIGHTS do not fit — the load materializes
/// those bytes before any request-time rung can bound anything, so no ladder admission could
/// rescue it.
fn weights_floor_load_admission(
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

fn with_selected_sequential_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    let spec = spec.with_offload_policy(OffloadPolicy::Sequential);
    if engine_id == "z_image_turbo" {
        spec.with_load_shape(gen_core::LoadShape::DeferredMaterialization)
    } else {
        spec
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
            let spec = with_selected_sequential_shape(engine_id, spec);
            // Z-Image's shipped rung-4 evidence was captured under an independent deferred loader
            // shape, while its lower rungs are eager. Production reaches both honestly by coupling
            // the deferred shape only to the cold-load branch that selected Sequential residency.
            // Resident hosts retain the eager shape and its four lower-rung bindings.
            Ok(spec)
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
///
/// sc-16014-resolution: rehosted-q4-q8. Lens q4/q8 use provider-detected MLX affine packs, so their
/// provider footprint stays disk-derived. The bf16 artifact remains MXFP4 and receives both its
/// measured materialization delta and the architecture-specific activation headroom. A genuinely
/// bf16-on-disk encoder receives the same activation headroom but no invented weight expansion.
fn spec_component_bytes(engine_id: &str, spec: &LoadSpec) -> (u64, u64, HeadroomAllowance) {
    let footprint = crate::inference_runtime::media()
        .footprint(engine_id, spec)
        .ok()
        .flatten();
    spec_component_bytes_with_provider_footprint(engine_id, spec, footprint)
}

/// Provider-injected core of [`spec_component_bytes`]. The live path obtains `footprint` from the
/// active registry; keeping the arithmetic independent of registry composition lets platform-neutral
/// tests exercise the exact Lens materialization and overlay accounting used on macOS.
fn spec_component_bytes_with_provider_footprint(
    engine_id: &str,
    spec: &LoadSpec,
    footprint: Option<PerComponentBytes>,
) -> (u64, u64, HeadroomAllowance) {
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
    // Read the actual adapter sources at the same pre-load seam as controls. Provider-specific
    // residency matters: packed Wan keeps additive residuals, while dense Wan folds them into the
    // base and adds zero independent bytes. Other providers conservatively retain the source bytes;
    // a typed component contract may replace them with a more exact resident measurement below.
    total_bytes =
        total_bytes.saturating_add(external_adapter_source_bytes_for_gate(engine_id, spec));
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

    /// sc-18237: every shipped Qwen binding must describe a load shape the production route can
    /// execute. Native Qwen is deliberately deferred under both Resident and Sequential policies;
    /// q8 and the SC-18353 bf16/q4 ladder coordinates were captured under that exact
    /// materialization contract, while the old eager BF16/Q4 records remain historical corpus
    /// entries only.
    /// This is deliberately mutation-sensitive: adding any eager binding, or reintroducing an
    /// uncaptured tier, makes the production-shape assertion red.
    #[test]
    fn shipped_qwen_bindings_are_producible_by_the_production_deferred_route() {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin model manifest parses");
        let qwen = manifest["models"]
            .as_array()
            .and_then(|models| models.iter().find(|model| model["id"] == "qwen_image"))
            .and_then(Value::as_object)
            .expect("shipped qwen_image manifest entry");
        let bindings = MlxCalibrationBinding::from_manifest(qwen)
            .expect("Qwen calibration bindings are valid")
            .expect("Qwen declares exact MLX calibration bindings");

        assert_eq!(
            bindings.len(),
            9,
            "the exact deferred q8, q4, and bf16 bindings"
        );
        assert!(
            !live_mlx_closure_digest("qwen_image").is_empty(),
            "qwen_image must be declared in config/inference-provider-closures.json; an undeclared \
             lane resolves to the fail-closed empty expectation, which would make every currency \
             comparison in this module discriminating for the wrong reason"
        );
        let declared = shipped_mlx_declared_closure_digest("qwen_image");
        assert!(bindings.iter().all(|binding| {
            binding.query.abi == sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI
                && binding.provider == "qwen_image"
                && ["bf16", "q4", "q8"].contains(&binding.tier.as_str())
                && binding.mode == "text_to_image"
                && binding.overlay == "none"
                && binding.geometry
                    == CalibrationGeometry {
                        width: 1024,
                        height: 1024,
                        batch: 1,
                        frames: 1,
                    }
                && binding.query.inference_closure_digest == declared
        }));
        assert!(bindings
            .iter()
            .all(|binding| { binding.query.load_shape == LoadShapeKey::DeferredMaterialization }));

        let rungs_for = |tier: &str| {
            let mut rungs = bindings
                .iter()
                .filter(|binding| binding.tier == tier)
                .map(|binding| format!("{:?}", binding.rung))
                .collect::<Vec<_>>();
            rungs.sort();
            rungs
        };
        assert_eq!(
            rungs_for("q8"),
            ["BoundedAttention", "BoundedTransformerResidency"]
        );
        assert_eq!(
            rungs_for("q4"),
            ["BoundedAttention", "BoundedTransformerResidency"]
        );
        assert_eq!(
            rungs_for("bf16"),
            [
                "BoundedAttention",
                "BoundedDecode",
                "BoundedTransformerResidency",
                "Resident",
                "StagedResidency",
            ]
        );
    }

    /// sc-18408: the audited-model set is DERIVED from the manifest — every model declaring
    /// `mlx.calibrations` is audited, so a new declaration (flux2_dev arrived via PR #2221 while
    /// the old hand list silently ignored it) is covered the moment it ships. Each binding's
    /// expected shape is COMPUTED by calling the production shaping functions for that lane —
    /// `image_jobs::apply_measured_mlx_load_shape` (the entry shaping that opts Qwen/Lens/plain
    /// Krea/plain SDXL into the deferred load-exact contract), then either
    /// `apply_residency_policy` (the generator-cache cold-load Resident branch; the fixture path
    /// has unmeasurable weights, which always admits Resident and leaves the spec unchanged) or
    /// `with_selected_sequential_shape` (the Sequential branch that couples Z-Image's deferred
    /// shape to rung 4). No arm hand-writes a shape or leans on the `LoadSpec` struct default, so
    /// a flipped production route reds here instead of staying green against a stale literal.
    /// macOS-only because the entry shaping IS the macOS MLX route; off-Mac there is no MLX
    /// cold-load path for a binding to describe.
    #[test]
    #[cfg(target_os = "macos")]
    fn every_shipped_audited_mlx_binding_has_a_producible_production_load_shape() {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin model manifest parses");
        let models = manifest["models"].as_array().expect("models array");

        let mut audited: Vec<String> = Vec::new();
        for model in models {
            let model = model.as_object().expect("model entry is an object");
            let model_id = model["id"].as_str().expect("model id");
            let Some(bindings) = MlxCalibrationBinding::from_manifest(model)
                .unwrap_or_else(|error| panic!("{model_id} bindings parse: {error}"))
            else {
                continue;
            };
            audited.push(model_id.to_owned());
            for binding in bindings {
                let provider = binding.provider.clone();
                let control_lane = provider == format!("{model_id}_control");
                assert!(
                    control_lane || provider == model_id,
                    "{model_id} declares an mlx.calibrations binding for provider {provider}, \
                     which this audit cannot map to a production lane of the model (base or \
                     `_control`) — teach the audit that lane's production route before shipping \
                     the binding, or its load shape goes unchecked"
                );

                // The lane's entry spec, characterized the way the production route builds it:
                // a directory-native load at the binding's tier, with the control overlay set for
                // a `_control` provider (which is exactly what keeps the entry shaping from
                // opting the base model's plain-text-to-image deferred contract into this lane).
                let mut spec =
                    LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("fixture")));
                spec = match binding.tier.as_str() {
                    "q4" => spec.with_quant(gen_core::Quant::Q4),
                    "q8" => spec.with_quant(gen_core::Quant::Q8),
                    "bf16" => spec,
                    other => panic!("{model_id} binding names unaudited tier {other}"),
                };
                if control_lane {
                    spec = spec.with_control(WeightsSource::File(std::path::PathBuf::from(
                        "fixture-control",
                    )));
                }
                let entry = crate::image_jobs::apply_measured_mlx_load_shape(&provider, spec);
                let expected = match binding.rung {
                    // Rung 4 is reached through the cold-load branch that selects Sequential
                    // residency; every lower rung rides the Resident branch.
                    StrategyRung::BoundedTransformerResidency => {
                        with_selected_sequential_shape(&provider, entry).load_shape
                    }
                    _ => {
                        apply_residency_policy(entry, &provider)
                            .expect("unmeasurable fixture weights always admit Resident")
                            .load_shape
                    }
                };
                let actual =
                    crate::memory_strategy::load_shape_from_receipt(binding.query.load_shape);
                assert_eq!(
                    actual, expected,
                    "{model_id} {provider} {:?} binding names a shape no production cold-load \
                     branch produces",
                    binding.rung
                );
            }
        }

        // A FLOOR, not a ceiling: new `mlx.calibrations` declarations are audited automatically
        // by the loop above. This guards the derivation itself — if the manifest loader ever
        // broke and reported "no bindings" for everything, the loop would audit nothing and pass
        // vacuously.
        for known in ["qwen_image", "z_image_turbo", "krea_2_turbo", "flux2_dev"] {
            assert!(
                audited.iter().any(|id| id == known),
                "{known} ships mlx.calibrations but the derived audit missed it — the manifest \
                 loader stopped seeing its bindings"
            );
        }
    }

    /// sc-18408 item (d): every MLX plan row must resolve through the shipped registry to a
    /// weights-free provider contract. This is deliberately derived from the plan rather than a
    /// hand-maintained provider list: adding a planned lane without registering its contract must
    /// fail in CI before the calibration adapter reaches a physical capture. Provider-owned
    /// contract fixtures avoid filesystem-shaped test doubles where providers expose them;
    /// SDXL and FLUX.2-dev intentionally fall back to their normal registrations, whose contract
    /// builders are themselves weights-free.
    #[test]
    #[cfg(target_os = "macos")]
    fn every_planned_mlx_lane_resolves_a_weights_free_provider_contract() {
        let plan: Value =
            serde_json::from_str(include_str!("../../../config/memory-calibration-plan.json"))
                .expect("memory calibration plan parses");
        let rows = plan["providers"].as_array().expect("plan providers array");
        let registry = crate::inference_runtime::media();
        let mut checked = 0_usize;

        for row in rows.iter().filter(|row| row["backend"] == "mlx") {
            let target = row["target"].as_object().expect("plan target object");
            let provider = target["provider"].as_str().expect("plan provider");
            let mode = target["mode"].as_str().expect("plan mode");
            let tier = target["tier"].as_str().expect("plan tier");
            let overlay = target["overlay"].as_str().expect("plan overlay");
            let load_shape = match row["loadShape"].as_str().expect("plan loadShape") {
                "eager_materialization" => gen_core::LoadShape::EagerMaterialization,
                "deferred_materialization" => gen_core::LoadShape::DeferredMaterialization,
                other => {
                    panic!("planned MLX lane {provider}/{mode} names unknown load shape {other}")
                }
            };

            let mut spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("fixture")))
                .with_load_shape(load_shape);
            spec = match tier {
                "q4" => spec.with_quant(gen_core::Quant::Q4),
                "q8" => spec.with_quant(gen_core::Quant::Q8),
                "bf16" => spec,
                other => panic!("planned MLX lane {provider}/{mode} names unknown tier {other}"),
            };
            match overlay {
                "none" => {}
                "control:1" => {
                    spec = spec.with_control(WeightsSource::File(std::path::PathBuf::from(
                        "fixture-control",
                    )));
                }
                other => panic!(
                    "planned MLX lane {provider}/{mode} names unmapped overlay {other}; teach the \
                     generic contract guard how production represents it"
                ),
            }

            let registration = registry
                .memory_strategy_registrations()
                .find(|registration| registration.provider_id == provider)
                .unwrap_or_else(|| {
                    panic!(
                        "planned MLX lane {provider}/{mode} has no memory-strategy registration in \
                         the shipped runtime registry"
                    )
                });
            let contract = match registry
                .memory_contract_fixture_registrations()
                .find(|fixture| fixture.provider_id == provider)
            {
                Some(fixture) => (fixture.contract)(&spec),
                None => (registration.contract)(&spec),
            }
            .unwrap_or_else(|error| {
                panic!(
                    "planned MLX lane {provider}/{mode} cannot build a weights-free memory \
                     contract: {error}"
                )
            });

            assert_eq!(
                contract.provider_id, provider,
                "planned MLX lane {provider}/{mode} resolved another provider's contract"
            );
            assert_eq!(
                contract.backend.backend_kind(),
                gen_core::MemoryBackend::Mlx,
                "planned MLX lane {provider}/{mode} resolved a non-MLX contract"
            );
            assert_eq!(
                contract.load_shape, load_shape,
                "planned MLX lane {provider}/{mode} contract does not preserve its load shape"
            );
            let calibration = contract.calibration.as_ref().unwrap_or_else(|| {
                panic!(
                    "planned MLX lane {provider}/{mode} resolves only an uncalibratable \
                     compatibility contract"
                )
            });
            assert_eq!(
                calibration.load_shape, load_shape,
                "planned MLX lane {provider}/{mode} calibration identity does not preserve its \
                 load shape"
            );
            checked += 1;
        }

        assert!(
            checked > 0,
            "the shipped plan must contain at least one MLX lane"
        );
    }

    /// SC-16915 acceptance: the SHIPPED manifest opt-in and the SHIPPED evidence bundle must agree
    /// well enough for a covered cell to take the calibrated path. Every other test in this module
    /// builds its own fixture manifest and fixture bundle, so all of them stay green even when the
    /// two real artefacts have drifted apart — which is exactly the state this story had to repair.
    /// This one reads both real files and nothing else.
    ///
    /// The manifest now ships only the q8 pair re-captured under the production deferred route.
    /// BF16/Q4 continue through the estimate ladder until production-shaped measurements exist.
    /// The count assertion makes a dropped q8 record fail rather than hiding behind its sibling.
    ///
    /// sc-17774 split this into the two questions it had been conflating. AGREEMENT — do the two
    /// shipped artefacts describe the same measurements — is graded at the closure they were both
    /// captured under, and is therefore true regardless of where the pin has since moved. CURRENCY
    /// is graded separately, at the live closure: since sc-18096 a moved closure no longer demotes
    /// the route — the ladder still reaches calibrated admission with each candidate carrying its
    /// measured digest, and the selector applies the widened stale-measured margin when that
    /// digest differs from the live one.
    #[test]
    fn shipped_qwen_manifest_and_packaged_evidence_agree_at_their_captured_closure() {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin.models.jsonc parses");
        let entry = manifest
            .get("models")
            .and_then(Value::as_array)
            .expect("models array")
            .iter()
            .find(|model| model.get("id").and_then(Value::as_str) == Some("qwen_image"))
            .and_then(Value::as_object)
            .expect("qwen_image manifest entry");
        let calibrations = entry
            .get("mlx")
            .and_then(|mlx| mlx.get("calibrations"))
            .and_then(Value::as_array)
            .expect("qwen_image declares mlx.calibrations");
        let declared = shipped_mlx_declared_closure_digest("qwen_image");
        let live = live_mlx_closure_digest("qwen_image");

        let tier = "q8";
        let quant = Some(gen_core::Quant::Q8);
        let expected_rungs = 2_usize;
        {
            // Take the request identity from the opt-in itself rather than restating it, so the
            // test cannot drift from the manifest it is checking.
            let binding = calibrations
                .iter()
                .find(|item| item.get("tier").and_then(Value::as_str) == Some(tier))
                .unwrap_or_else(|| panic!("{tier} is part of the shipped opt-in"));
            let text = |key: &str| {
                binding
                    .get(key)
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| panic!("{tier} binding is missing {key}"))
                    .to_owned()
            };
            let geometry = binding.get("geometry").expect("binding geometry");
            let dimension = |key: &str| {
                u32::try_from(
                    geometry
                        .get(key)
                        .and_then(Value::as_u64)
                        .unwrap_or_else(|| panic!("geometry is missing {key}")),
                )
                .expect("geometry fits u32")
            };

            let mut spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(format!(
                "/cache/models--SceneWorks--qwen-image-mlx/snapshots/x/{tier}"
            ))));
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            let plan = MlxRequestPlan::for_spec_and_manifest(
                "qwen_image",
                "qwen_image",
                &spec,
                Some(entry),
                Some(ResolvedArtifactProvenance {
                    identity: crate::model_jobs::ResolvedArtifactIdentity {
                        repository: text("artifactRepository"),
                        revision: text("artifactResolvedRevision"),
                        variant: text("artifactVariant"),
                        fingerprint: text("resolvedPathFingerprint"),
                    },
                    fixed_artifact_tier: Some(tier.to_owned()),
                }),
            );
            assert!(
                matches!(plan.calibration, MlxCalibrationConfig::Valid(_)),
                "{tier}: the shipped opt-in must parse as a valid binding set"
            );

            let mut inputs = fixture_inputs(dimension("width"), dimension("height"));
            inputs.overlay = None;
            // Graded at the closure both artefacts were captured under. This is the agreement claim
            // and it does not expire: whether the manifest opt-in and the bundle describe the same
            // measurements is a fact about the two files, not about the pin.
            let route = packaged_admission_route(
                &plan,
                &inputs,
                &text("mode"),
                fixture_budget(128.0),
                &declared,
            )
            .expect("a covered cell must not error");
            assert_eq!(
                route.path,
                AdmissionPath::Evidence,
                "{tier}: shipped manifest + shipped evidence must reach calibrated admission at \
                 the closure they were captured under; got fallback {:?}",
                route.fallback_reason
            );
            assert_eq!(
                route.evidence.len(),
                expected_rungs,
                "{tier}: every declared rung must resolve to a promoted record, not just one"
            );
            assert!(
                route
                    .evidence
                    .iter()
                    .all(|candidate| !candidate.record_id.is_empty()),
                "{tier}: each candidate names the exact record backing it"
            );

            // The currency claim, derived from the digest pair rather than hardcoded either way.
            // `mlx:qwen_image`'s closure covers `crates/media/mlx-gen`, which every MLX provider
            // depends on, so an edit for another model legitimately re-dates this ladder.
            // sc-18096: currency is a signal, not a gate — a superseded closure no longer demotes
            // the route. The ladder still reaches calibrated admission, its candidates carry the
            // digest they were MEASURED under, and the selector grades them behind the widened
            // stale-measured margin.
            let at_live = packaged_admission_route(
                &plan,
                &inputs,
                &text("mode"),
                fixture_budget(128.0),
                &live,
            )
            .expect("a moved provider closure widens the margin, it never errors");
            assert_eq!(
                at_live.path,
                AdmissionPath::Evidence,
                "{tier}: the ladder must reach calibrated admission current OR stale; got \
                 fallback {:?}",
                at_live.fallback_reason
            );
            assert!(
                at_live
                    .evidence
                    .iter()
                    .all(|candidate| candidate.closure_digest == declared),
                "{tier}: every candidate must carry the digest its binding was measured under, so \
                 the selector can grade its currency against the live closure"
            );
        }

        // Mutation check for the axis this story exists to restore. Asserting only the route above
        // is a FALSE GREEN for `loadShape`: the route matches whichever binding fits the request,
        // so corrupting one cell's shape just selects a different cell and still reaches Evidence.
        // Flip EVERY declared shape and the whole opt-in must stop matching — these q8 receipts say
        // deferred, and an eager claim is not interchangeable.
        //
        // Driven at `declared`, not at the live closure. Once the two diverge the live route is
        // ALREADY `Legacy`/`StaleIdentity` for currency reasons, so a mutation graded there proves
        // nothing about the load-shape axis — the assertion would pass with the mutation reverted.
        let q8 = calibrations
            .iter()
            .find(|item| item.get("tier").and_then(Value::as_str) == Some("q8"))
            .expect("q8 binding");
        let text = |key: &str| {
            q8.get(key)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("binding is missing {key}"))
                .to_owned()
        };
        let mut mutated = entry.clone();
        for calibration in mutated
            .get_mut("mlx")
            .and_then(|mlx| mlx.get_mut("calibrations"))
            .and_then(Value::as_array_mut)
            .expect("calibrations array")
        {
            let shape = calibration
                .get("loadShape")
                .and_then(Value::as_str)
                .expect("every binding declares a loadShape");
            calibration["loadShape"] = Value::from(match shape {
                "eager_materialization" => "deferred_materialization",
                _ => "eager_materialization",
            });
        }
        let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
            "/cache/models--SceneWorks--qwen-image-mlx/snapshots/x/q8",
        )))
        .with_quant(gen_core::Quant::Q8);
        let mut inputs = fixture_inputs(1024, 1024);
        inputs.overlay = None;
        let mutated_route = packaged_admission_route(
            &MlxRequestPlan::for_spec_and_manifest(
                "qwen_image",
                "qwen_image",
                &spec,
                Some(&mutated),
                Some(ResolvedArtifactProvenance {
                    identity: crate::model_jobs::ResolvedArtifactIdentity {
                        repository: text("artifactRepository"),
                        revision: text("artifactResolvedRevision"),
                        variant: text("artifactVariant"),
                        fingerprint: text("resolvedPathFingerprint"),
                    },
                    fixed_artifact_tier: Some("q8".to_owned()),
                }),
            ),
            &inputs,
            "text_to_image",
            fixture_budget(128.0),
            &declared,
        )
        .expect("a load-shape mismatch degrades, never errors");
        assert_eq!(
            mutated_route.path,
            AdmissionPath::Legacy,
            "an opt-in claiming the wrong materialization shape must NOT reach calibrated admission"
        );
        // The REASON matters as much as the path: `Legacy` alone would also be satisfied by the
        // mutated manifest failing to parse into bindings at all (`NoBinding`), which would make
        // this a test of malformed JSON rather than of the load-shape axis.
        assert_eq!(
            mutated_route.fallback_reason,
            Some(LegacyAdmissionReason::StaleIdentity),
            "the bindings must PARSE and then go stale on the shape, not fail to parse"
        );
    }

    /// The krea half of the same guarantee. `packaged_krea_plan` synthesizes its bindings FROM the
    /// evidence records, so it is current-by-construction and structurally cannot notice the
    /// shipped krea manifest disagreeing with the shipped bundle. krea_2_turbo_control is a covered
    /// provider of sc-16915, so it gets the same real-manifest × real-evidence route check qwen has
    /// — including the same sc-17774 split between agreement (at the captured closure, permanent)
    /// and currency (at the live closure, derived).
    #[test]
    fn shipped_krea_manifest_and_packaged_evidence_agree_at_their_captured_closure() {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin.models.jsonc parses");
        let entry = manifest
            .get("models")
            .and_then(Value::as_array)
            .expect("models array")
            .iter()
            .find(|model| model.get("id").and_then(Value::as_str) == Some("krea_2_turbo"))
            .and_then(Value::as_object)
            .expect("krea_2_turbo manifest entry");
        let calibrations = entry
            .get("mlx")
            .and_then(|mlx| mlx.get("calibrations"))
            .and_then(Value::as_array)
            .expect("krea_2_turbo declares mlx.calibrations");
        assert_eq!(
            calibrations.len(),
            2,
            "the shipped krea opt-in is the 768² and 896² pose-control pair"
        );
        let declared = shipped_mlx_declared_closure_digest("krea_2_turbo");
        let live = live_mlx_closure_digest("krea_2_turbo_control");

        for binding in calibrations {
            let text = |key: &str| {
                binding
                    .get(key)
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| panic!("krea binding is missing {key}"))
                    .to_owned()
            };
            let geometry = binding.get("geometry").expect("binding geometry");
            let dimension = |key: &str| {
                u32::try_from(
                    geometry
                        .get(key)
                        .and_then(Value::as_u64)
                        .expect("dimension"),
                )
                .expect("geometry fits u32")
            };
            let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
                "/cache/models--SceneWorks--krea-2-turbo-mlx/snapshots/x/q4",
            )))
            .with_quant(gen_core::Quant::Q4);
            // `packaged_admission_route` filters on `binding.provider == plan.engine_id`, so the
            // engine id is the PROVIDER (`krea_2_turbo_control`) and the model id is the catalog
            // entry (`krea_2_turbo`) — the order production uses in `image_jobs/krea_control.rs`,
            // which passes `KREA_CONTROL_ENGINE_ID` then `&request.model`. Swapping them yields
            // `StaleIdentity` and a silent legacy fallback.
            let plan = MlxRequestPlan::for_spec_and_manifest(
                "krea_2_turbo_control",
                "krea_2_turbo",
                &spec,
                Some(entry),
                Some(ResolvedArtifactProvenance {
                    identity: crate::model_jobs::ResolvedArtifactIdentity {
                        repository: text("artifactRepository"),
                        revision: text("artifactResolvedRevision"),
                        variant: text("artifactVariant"),
                        fingerprint: text("resolvedPathFingerprint"),
                    },
                    fixed_artifact_tier: Some(text("tier")),
                }),
            );
            assert!(
                matches!(plan.calibration, MlxCalibrationConfig::Valid(_)),
                "the shipped krea opt-in must parse as a valid binding set"
            );

            let mut inputs = fixture_inputs(dimension("width"), dimension("height"));
            inputs.overlay = Some(text("overlay"));
            inputs.has_reference = true;
            inputs.reference_count = 1;
            let route = packaged_admission_route(
                &plan,
                &inputs,
                &text("mode"),
                fixture_budget(128.0),
                &declared,
            )
            .expect("a covered krea cell must not error");
            assert_eq!(
                route.path,
                AdmissionPath::Evidence,
                "{}x{}: shipped krea manifest + evidence must reach calibrated admission at the \
                 closure they were captured under; got fallback {:?}",
                dimension("width"),
                dimension("height"),
                route.fallback_reason
            );

            // Currency, derived. `mlx:krea_2_turbo_control`'s closure spans `mlx-gen` and five
            // sibling provider crates, so it moves on work that has nothing to do with Krea.
            // sc-18096: a superseded closure no longer demotes — the ladder stays admissible with
            // its candidates carrying the measured digest for the selector's widened grading.
            let at_live = packaged_admission_route(
                &plan,
                &inputs,
                &text("mode"),
                fixture_budget(128.0),
                &live,
            )
            .expect("a moved provider closure widens the margin, it never errors");
            assert_eq!(
                at_live.path,
                AdmissionPath::Evidence,
                "{}x{}: the krea ladder must reach calibrated admission current OR stale; got \
                 fallback {:?}",
                dimension("width"),
                dimension("height"),
                at_live.fallback_reason
            );
            assert!(
                at_live
                    .evidence
                    .iter()
                    .all(|candidate| candidate.closure_digest == declared),
                "{}x{}: every candidate must carry the digest its binding was measured under",
                dimension("width"),
                dimension("height"),
            );
        }
    }

    /// SC-16915 acceptance, end to end on real weights (ignored — needs the qwen-image-mlx bf16
    /// turnkey, ~57 GB, and peaks around 59 GiB active).
    ///
    /// The non-gated `shipped_qwen_manifest_and_packaged_evidence_select_the_calibrated_path`
    /// proves the shipped manifest and bundle agree. This closes the story's remaining acceptance
    /// line — "calibrated admission selects verified rungs again on a real MLX render" — by driving
    /// the production seam `image_jobs/base.rs` calls per generation, against a LIVE loaded
    /// provider, and then rendering.
    ///
    /// The load-bearing assertion is `process_limit_bytes.is_some()`. That ceiling is derived only
    /// from an exact verified cell; a legacy admission leaves it `None` and falls back to the
    /// process-global limit. So it distinguishes "the request was admitted" — which the legacy
    /// estimator also does — from "an exact verified rung was selected", which is what regressed.
    ///
    ///   cargo test -p sceneworks-worker --lib -- --ignored --nocapture \
    ///     qwen_real_install_selects_a_verified_rung
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight MLX render; needs the qwen-image-mlx bf16 turnkey (~57 GB)"]
    fn qwen_real_install_selects_a_verified_rung_and_renders() {
        let hub = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
            .join(".cache/huggingface/hub");
        let snapshots = hub.join("models--SceneWorks--qwen-image-mlx/snapshots");
        let Some(bf16) = std::fs::read_dir(&snapshots).ok().and_then(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path().join("bf16"))
                .find(|path| path.join("model_index.json").is_file())
        }) else {
            eprintln!(
                "SKIP: no qwen-image-mlx bf16 turnkey under {}",
                snapshots.display()
            );
            return;
        };

        // The SHIPPED manifest entry, not a fixture: the opt-in under test is the one that ships.
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin.models.jsonc parses");
        let entry = manifest
            .get("models")
            .and_then(Value::as_array)
            .expect("models array")
            .iter()
            .find(|model| model.get("id").and_then(Value::as_str) == Some("qwen_image"))
            .and_then(Value::as_object)
            .expect("qwen_image manifest entry")
            .clone();

        // Provenance is derived from the snapshot actually being loaded, NOT copied from the app's
        // install receipt — that receipt can name a different variant than the one under test.
        // Reading the revision out of the resolved snapshot path and asserting it against the
        // manifest's declared identity is the stronger check anyway: it proves the bytes on disk
        // are the ones the opt-in names, which is the whole job of provenance.
        let revision = bf16
            .parent()
            .and_then(|snapshot| snapshot.file_name())
            .and_then(|name| name.to_str())
            .expect("snapshot directory is <hub>/models--.../snapshots/<revision>/bf16")
            .to_owned();
        let binding = entry
            .get("mlx")
            .and_then(|mlx| mlx.get("calibrations"))
            .and_then(Value::as_array)
            .expect("qwen_image declares mlx.calibrations")
            .iter()
            .find(|item| item.get("tier").and_then(Value::as_str) == Some("bf16"))
            .expect("a bf16 binding");
        let declared = |key: &str| {
            binding
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("binding is missing {key}"))
                .to_owned()
        };
        assert_eq!(
            revision,
            declared("artifactResolvedRevision"),
            "the on-disk snapshot must be the exact artifact the shipped opt-in names"
        );
        let provenance = ResolvedArtifactProvenance {
            identity: crate::model_jobs::ResolvedArtifactIdentity {
                repository: declared("artifactRepository"),
                revision,
                variant: declared("artifactVariant"),
                fingerprint: declared("resolvedPathFingerprint"),
            },
            fixed_artifact_tier: Some(declared("tier")),
        };

        let spec = LoadSpec::new(WeightsSource::Dir(bf16.clone()));
        let plan = MlxRequestPlan::for_spec_and_manifest(
            "qwen_image",
            "qwen_image",
            &spec,
            Some(&entry),
            Some(provenance),
        );
        let inputs = fixture_inputs(1024, 1024);

        // Cheap pre-load check, so a routing regression fails before a 57 GB load.
        let route = packaged_admission_route(
            &plan,
            &inputs,
            "text_to_image",
            fixture_budget(128.0),
            &live_mlx_closure_digest("qwen_image"),
        )
        .expect("covered cell must not error");
        assert_eq!(
            route.path,
            AdmissionPath::Evidence,
            "shipped opt-in must route to calibrated admission; got fallback {:?}",
            route.fallback_reason
        );

        eprintln!(
            "[acceptance] loading qwen_image bf16 from {}",
            bf16.display()
        );
        let generator =
            crate::inference_runtime::load("qwen_image", &spec).expect("load qwen_image");

        // THE production call: exactly what base.rs runs per generation, on a live generator.
        let evaluation = evaluate_request(
            &*generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            0,
        )
        .expect("a covered qwen cell must be admitted");
        eprintln!("[acceptance] gate admitted: {evaluation:?}");
        assert!(
            evaluation.process_limit_bytes.is_some(),
            "a verified rung supplies a request-scoped ceiling; `None` means this degraded to the \
             legacy estimator, which is exactly the regression this story repaired"
        );

        let request = gen_core::GenerationRequest {
            prompt: "a red fox resting beside a blue ceramic vase, studio photograph".to_owned(),
            width: 1024,
            height: 1024,
            count: 1,
            seed: Some(15511),
            steps: Some(2),
            memory: Some(evaluation.memory),
            ..Default::default()
        };
        let image = match generator
            .generate(&request, &mut |_| {})
            .expect("real qwen render must succeed under the calibrated ceiling")
        {
            gen_core::GenerationOutput::Images(mut images) => {
                assert_eq!(images.len(), 1, "one requested image");
                images.pop().expect("one image")
            }
            other => panic!("qwen_image must return images, got {other:?}"),
        };
        assert!(
            image.width == 1024 && image.height == 1024,
            "rendered geometry must match the verified cell"
        );
        eprintln!(
            "[acceptance] rendered {}x{} under a verified rung",
            image.width, image.height
        );
    }

    /// Stale admission (sc-18096) on a REAL shipped lane. Z-Image's ladder was measured at
    /// `d4802320`, and `mlx:z_image_turbo`'s closure has moved since, so this is the one shipped
    /// opt-in whose calibration is genuinely stale. Under sc-18095/18096 that ladder stays
    /// SELECTABLE — its candidates reach the selector carrying their measured digest and are
    /// graded behind the widened stale margin — while a binding that LIES about its digest still
    /// resolves to nothing.
    #[test]
    fn shipped_z_image_manifest_admits_historical_mlx_ladder_rungs_as_stale() {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin model manifest parses");
        let z_image = manifest["models"]
            .as_array()
            .and_then(|models| models.iter().find(|model| model["id"] == "z_image_turbo"))
            .and_then(Value::as_object)
            .expect("shipped z_image_turbo manifest entry");
        let bindings = MlxCalibrationBinding::from_manifest(z_image)
            .expect("Z-Image calibration bindings are valid")
            .expect("Z-Image declares exact MLX calibration bindings");

        assert_eq!(bindings.len(), 5);
        // The digest the SHIPPED EVIDENCE says this ladder was measured under, read from the bundle
        // rather than restated as a hex literal. A literal here is brittle in a way that matters:
        // `--restamp` legitimately re-derives every digest on a `CLOSURE_DIGEST_VERSION` change
        // without any measurement moving, and a pinned literal turns that no-op into a red test.
        // Reading it back is also STRICTER — it catches the manifest binding drifting away from the
        // record it claims, which a literal cannot see.
        let captured = packaged_bundle()
            .records
            .into_iter()
            .find(|record| {
                matches!(record.backend, CalibrationBackend::Mlx)
                    && record.target.provider == "z_image_turbo"
            })
            .and_then(|record| record.repositories.inference.closure_digest)
            .expect("the packaged bundle carries the historical Z-Image ladder with its digest");
        let live = live_mlx_closure_digest("z_image_turbo");
        assert_ne!(
            captured, live,
            "this test is about a lane whose closure HAS moved; if z_image_turbo were recaptured \
             current, the assertions below would be checking the opposite of their own name"
        );
        assert!(bindings.iter().all(|binding| {
            binding.query.abi == sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI
                && binding.provider == "z_image_turbo"
                && binding.tier == "q4"
                && binding.mode == "text_to_image"
                && binding.overlay == "none"
                && binding.geometry
                    == CalibrationGeometry {
                        width: 768,
                        height: 768,
                        batch: 1,
                        frames: 1,
                    }
                // Capture provenance, pinned as a literal because it is a fact about the shipped
                // opt-in that no pin bump may silently rewrite. It is NOT the currency term.
                && binding.query.inference_revision == "d48023204cd3a4f3f8eb060f79803dccaddcb482"
                // The currency term (sc-17774), graded against three independent artifacts: the
                // manifest binding must agree with the evidence bundle about what it measured, and
                // must disagree with the live closure config — which is why the route degrades.
                && binding.query.inference_closure_digest == captured
                && binding.query.inference_closure_digest != live
        }));
        assert!(bindings.iter().all(|binding| {
            binding.query.load_shape
                == if binding.rung == StrategyRung::BoundedTransformerResidency {
                    LoadShapeKey::DeferredMaterialization
                } else {
                    LoadShapeKey::EagerMaterialization
                }
        }));

        let resolved_binding = &bindings[0].query;
        let resolved = ResolvedArtifactProvenance {
            identity: crate::model_jobs::ResolvedArtifactIdentity {
                repository: resolved_binding.artifact_repository.clone(),
                revision: resolved_binding.artifact_resolved_revision.clone(),
                variant: resolved_binding.artifact_variant.clone(),
                fingerprint: resolved_binding.resolved_path_fingerprint.clone(),
            },
            fixed_artifact_tier: Some("q4".to_owned()),
        };
        let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
            "/packaged/z-image-turbo-mlx/q4",
        )))
        .with_quant(gen_core::Quant::Q4);
        let plan = MlxRequestPlan::for_spec_and_manifest(
            "z_image_turbo",
            "z_image_turbo",
            &spec,
            Some(z_image),
            Some(resolved),
        );
        let route = packaged_admission_route(
            &plan,
            &fixture_inputs(768, 768),
            "text_to_image",
            fixture_budget(128.0),
            &live_mlx_closure_digest("z_image_turbo"),
        )
        .expect("historical Z-Image evidence must route without an error");

        // sc-18096: the historical ladder is no longer pre-demoted to legacy. It reaches
        // calibrated admission with every candidate carrying the digest it was MEASURED under, so
        // the selector grades the whole ladder behind the widened stale-measured margin instead of
        // refusing the render outright.
        assert_eq!(
            route.path,
            AdmissionPath::Evidence,
            "the stale historical ladder must reach the selector; got fallback {:?}",
            route.fallback_reason
        );
        assert_eq!(
            route.evidence.len(),
            5,
            "every historical rung must resolve to its promoted record"
        );
        assert!(
            route
                .evidence
                .iter()
                .all(|candidate| candidate.closure_digest == captured),
            "each candidate must carry the captured digest so the selector can widen it"
        );

        // A manifest still cannot LAUNDER a stale ladder current by claiming the live digest.
        //
        // With the admission pre-filter's closure conjunct retired (sc-18096), what stops the
        // laundering is `EvidenceBundle::evidence_for` finding the RECORD still stamped with the
        // digest it was really measured under: a binding claiming the live digest no longer
        // matches its own record, so it resolves to NO candidates at all — a lie about identity
        // is worse off than the honest stale declaration above, which is exactly the incentive
        // the two comparisons must preserve. One asks "which measurement is this binding telling
        // the truth about?"; the currency question is now the selector's widened-margin grading.
        let mut laundered = z_image.clone();
        for calibration in laundered
            .get_mut("mlx")
            .and_then(|mlx| mlx.get_mut("calibrations"))
            .and_then(Value::as_array_mut)
            .expect("calibrations array")
        {
            calibration["inferenceClosureDigest"] = Value::from(live.clone());
        }
        let laundered_route = packaged_admission_route(
            &MlxRequestPlan::for_spec_and_manifest(
                "z_image_turbo",
                "z_image_turbo",
                &spec,
                Some(&laundered),
                Some(ResolvedArtifactProvenance {
                    identity: crate::model_jobs::ResolvedArtifactIdentity {
                        repository: resolved_binding.artifact_repository.clone(),
                        revision: resolved_binding.artifact_resolved_revision.clone(),
                        variant: resolved_binding.artifact_variant.clone(),
                        fingerprint: resolved_binding.resolved_path_fingerprint.clone(),
                    },
                    fixed_artifact_tier: Some("q4".to_owned()),
                }),
            ),
            &fixture_inputs(768, 768),
            "text_to_image",
            fixture_budget(128.0),
            &live,
        )
        .expect("a laundered opt-in degrades, it does not error");
        assert_eq!(
            laundered_route.path,
            AdmissionPath::Legacy,
            "a binding claiming the live digest must not reach calibrated admission on records \
             measured under another closure"
        );
        assert!(laundered_route.evidence.is_empty());
    }

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
                component_precision_floors: &[],
            },
            "text_to_image",
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
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
                control_kinds: None,
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
                component_precision_floors: &[],
            },
            asset_bytes: gib_to_bytes(6.0),
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
            // Deliberately ABOVE the 2 GiB fixed reserve so the area term is non-zero: a fixture
            // sitting exactly on the reserve would model resolution-blind and silently stop
            // exercising the sc-16195 scaling at all.
            activation_headroom_bytes: gib_to_bytes(6.0),
            fixed_reserve_bytes: gib_to_bytes(2.0),
            calibration: MlxCalibrationConfig::Absent,
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mage_q4_memory_identity_includes_descriptor_component_floors() {
        let source = WeightsSource::Dir(std::path::PathBuf::from("/nonexistent/mage-fixture"));
        let q4 = LoadSpec::new(source.clone()).with_quant(gen_core::Quant::Q4);
        let q4_plan =
            MlxRequestPlan::for_spec_and_manifest("mage_flow", "mage_flow", &q4, None, None);
        let advertised = crate::inference_runtime::media_descriptor("mage_flow")
            .unwrap()
            .capabilities
            .component_precision_floors;
        assert_eq!(q4_plan.tier.component_precision_floors, advertised);
        assert_eq!(advertised.len(), 2);

        let q8 = LoadSpec::new(source).with_quant(gen_core::Quant::Q8);
        let q8_plan =
            MlxRequestPlan::for_spec_and_manifest("mage_flow", "mage_flow", &q8, None, None);
        assert!(q8_plan.tier.component_precision_floors.is_empty());
        assert_ne!(q4_plan.tier, q8_plan.tier);
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
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            MAGE_CALIBRATION_FINGERPRINT,
            gen_core::LoadShape::EagerMaterialization,
        ));
        contract.asset_facts.base_bytes = gib_to_bytes(6.0);
        contract.asset_facts.transformer_bytes = gib_to_bytes(6.0);
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
            reference_count: 2,
            use_pid: false,
            has_phases: false,
        }
    }

    /// The fixture record carries its own synthetic revision and [`FIXTURE_CLOSURE_DIGEST`].
    ///
    /// This used to rewrite every record's `inference.revision` to the live Cargo pin, because
    /// currency was pin equality and the fixture would otherwise have read as stale on load. Under
    /// sc-17774 currency is the closure digest, which the fixture file already carries, so the
    /// rewrite is dead — and keeping it would restate the pin as a currency term in the one place
    /// every gate test builds its evidence from.
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
                load_shape: sceneworks_core::memory_calibration::LoadShapeKey::EagerMaterialization,
                fingerprint: "fixture-formula-v2".to_owned(),
                scene_works_revision: "a".repeat(40),
                matrix_source_revision: "source-tree:1111111".to_owned(),
                // Capture provenance only, and deliberately the fixture RECORD's own synthetic
                // revision rather than the live pin: nothing compares this field, and spelling the
                // pin here implied a currency that no longer exists (sc-17774).
                inference_revision: "b".repeat(40),
                inference_closure_digest: fixture_closure_digest(),
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
            selection_parameters: parse_evidence_parameters(
                rung,
                &crate::memory_strategy::default_engaged_composition(rung),
                &parameters,
            )
            .expect("fixture parameters"),
            parameters,
        }
    }

    /// The closure digest the synthetic `fixture_provider` lane is measured under.
    ///
    /// `fixture_provider` is not a real inference crate, so it is deliberately NOT in
    /// `config/inference-provider-closures.json` — the gate must keep refusing undeclared lanes.
    /// These tests inject [`fixture_closure_lookup`] instead, which answers for the fixture lane and
    /// defers to the packaged table for every real one.
    const FIXTURE_CLOSURE_DIGEST: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn fixture_closure_digest() -> String {
        FIXTURE_CLOSURE_DIGEST.to_owned()
    }

    /// What the gate resolves in production for one real MLX lane, including the fail-closed empty
    /// string for a lane nobody declared (`krea_2_turbo`, the base t2i route, is one). Spelled as a
    /// helper so a test names the provider whose currency it is asserting instead of a hex literal.
    fn live_mlx_closure_digest(provider: &str) -> String {
        sceneworks_core::memory_calibration::packaged_closure_digest("mlx", provider)
            .unwrap_or_default()
    }

    /// The injected resolver. Real lanes still resolve through the shipped table, so a test that
    /// uses `qwen_image` or `krea_2_turbo_control` is still graded against production config.
    fn fixture_closure_lookup(backend: &str, provider: &str) -> Option<String> {
        if backend == "mlx" && provider == "fixture_provider" {
            return Some(fixture_closure_digest());
        }
        sceneworks_core::memory_calibration::packaged_closure_digest(backend, provider)
    }

    /// The compile-closure digest the SHIPPED `mlx.calibrations` opt-in for `model_id` declares —
    /// the closure its bindings, and the packaged records behind them, were measured under.
    ///
    /// Deliberately NOT [`live_mlx_closure_digest`]. That one answers "what does the pinned
    /// inference tree compile to now"; this one answers "what did these measurements describe". The
    /// two are equal exactly while the opt-in is current, and the tests below DERIVE that verdict
    /// from the pair rather than assuming either side of it. They have to: the MLX providers share
    /// first-party crates (`crates/media/mlx-gen` is in every one of their closures), so an edit
    /// aimed at one model legitimately re-dates the others. That is the closure mechanism being
    /// conservative, not an opt-in that broke, and a test that hardcodes "current" turns it into a
    /// red build instead of a re-capture signal.
    ///
    /// Uniformity across the model's bindings is asserted rather than assumed: a split opt-in would
    /// let one stale row hide behind a current one and make every comparison below ambiguous.
    fn shipped_mlx_declared_closure_digest(model_id: &str) -> String {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin model manifest parses");
        let mut declared = manifest["models"]
            .as_array()
            .and_then(|models| models.iter().find(|model| model["id"] == model_id))
            .and_then(|model| model["mlx"]["calibrations"].as_array())
            .unwrap_or_else(|| panic!("{model_id} declares mlx.calibrations"))
            .iter()
            .map(|binding| {
                binding["inferenceClosureDigest"]
                    .as_str()
                    .unwrap_or_else(|| {
                        panic!("{model_id} calibration binding declares inferenceClosureDigest")
                    })
                    .to_owned()
            })
            .collect::<Vec<_>>();
        declared.sort();
        declared.dedup();
        assert_eq!(
            declared.len(),
            1,
            "{model_id}'s shipped bindings must all name ONE captured closure"
        );
        declared.remove(0)
    }

    /// [`fixture_closure_lookup`] with the Krea control lane pinned to the closure its PACKAGED
    /// records were captured under.
    ///
    /// `packaged_krea_1024_refuses_before_render_and_names_only_a_fitting_current_cell` is about how
    /// a refusal names the largest fitting exact cell — not about whether the shipped bundle is
    /// still current, which `a_moved_provider_closure_demotes_the_calibrated_ladder` owns. Reading
    /// currency from the live table made the two inseparable: a shared `mlx-gen` edit emptied the
    /// fixture and the naming behaviour silently went untested. The digest is read from the shipped
    /// opt-in, never restated as a literal, so this cannot drift into exercising a closure nothing
    /// was ever measured under.
    fn packaged_krea_closure_lookup(backend: &str, provider: &str) -> Option<String> {
        if backend == "mlx" && provider == "krea_2_turbo_control" {
            return Some(shipped_mlx_declared_closure_digest("krea_2_turbo"));
        }
        fixture_closure_lookup(backend, provider)
    }

    fn fixture_calibration_json(tier: &str, variant: &str) -> Value {
        serde_json::json!({
            "abi": sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
            "loadShape": "eager_materialization",
            "fingerprint": "fixture-formula-v2",
            "sceneWorksRevision": "a".repeat(40),
            "matrixSourceRevision": "source-tree:1111111",
            // Capture provenance only — see `fixture_binding_for`.
            "inferenceRevision": "b".repeat(40),
            // sc-17774: the fixture must present the digest the gate will look up for this lane,
            // otherwise every fit-ladder test silently becomes a currency-refusal test.
            "inferenceClosureDigest": fixture_closure_digest(),
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
                component_precision_floors: &[],
            },
            asset_bytes: gib_to_bytes(3.0),
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
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
            .expect("packaged bundle must parse")
        {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("packaged bundle must be current: {reason:?}"),
        };
        // Select the records that ARE current instead of rewriting stale ones into looking current,
        // which is what the deleted `packaged_bundle_migrated_to_v4_for_tests` shim did. The bundle
        // still carries the superseded 96b13b66 Krea cells as history, so without this currency
        // predicate the filter matches four records and the count assertion below fails.
        //
        // sc-17774: the predicate is the provider's own closure digest, not the inference pin. The
        // pin form separated the same two records only by coincidence — it also expired them on any
        // unrelated inference commit, so this fixture went unbuildable on every bump.
        //
        // The digest is the one the shipped opt-in DECLARES, not the live one. Selecting on live
        // made the fixture empty the moment a shared `mlx-gen` crate moved — a legitimate closure
        // move — and this fixture exists to exercise refusal NAMING, which needs two exact cells to
        // choose between. `packaged_krea_closure_lookup` feeds the same digest to the gate, and the
        // test asserts the live-closure refusal separately so the demotion is not merely bypassed.
        let captured = shipped_mlx_declared_closure_digest("krea_2_turbo");
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
                    && record.repositories.inference.closure_digest.as_deref()
                        == Some(captured.as_str())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            records.len(),
            2,
            "the packaged Krea contract has two exact cells at the closure the shipped \
             mlx:krea_2_turbo_control opt-in declares"
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
                    load_shape: record.load_shape,
                    fingerprint: record.calibration_fingerprint.clone(),
                    scene_works_revision: "sc-16099-contract-v1".to_owned(),
                    matrix_source_revision: record
                        .repositories
                        .scene_works
                        .matrix_source_revision
                        .clone()
                        .expect("current matrix source revision"),
                    inference_revision: record.repositories.inference.revision.clone(),
                    inference_closure_digest: record
                        .repositories
                        .inference
                        .closure_digest
                        .clone()
                        .expect("captured provider closure digest"),
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
                    &record.strategy.engaged_rungs,
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
                component_precision_floors: &[],
            },
            asset_bytes: gib_to_bytes(30.0),
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
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
        // Taken from the packaged record this fixture stands in for, never restated as a literal.
        // The previous literal `krea-control-mlx-v4-q4-pose-bounded-decode-512-64` outlived the
        // provider's move to the full-ladder identity, and a stale copy here fails the handshake
        // and reports the cell as `Missing` — which is how it presented before this was fixed.
        let captured = shipped_mlx_declared_closure_digest("krea_2_turbo");
        let record = packaged_bundle()
            .records
            .into_iter()
            .find(|record| {
                matches!(record.backend, CalibrationBackend::Mlx)
                    && record.target.provider == "krea_2_turbo_control"
                    && record.repositories.inference.closure_digest.as_deref()
                        == Some(captured.as_str())
            })
            .expect("the packaged bundle carries a Krea control record at the declared closure");
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            record.calibration_fingerprint,
            match record.load_shape {
                LoadShapeKey::DeferredMaterialization => {
                    gen_core::LoadShape::DeferredMaterialization
                }
                LoadShapeKey::EagerMaterialization => gen_core::LoadShape::EagerMaterialization,
            },
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
            reference_count: 0,
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
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "fixture-formula-v2",
            gen_core::LoadShape::EagerMaterialization,
        ));
        contract.asset_facts.base_bytes = gib_to_bytes(3.0);
        contract.asset_facts.transformer_bytes = gib_to_bytes(3.0);
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
                control_kinds: None,
            },
            contract: Some(contract),
        }
    }

    /// The real packaged bundle, as production loads it. sc-16915 re-collected the MLX evidence at
    /// the current pin, so the tests below no longer need the deleted
    /// `packaged_bundle_migrated_to_v4_for_tests` shim, which forced `schemaVersion` to 4, stamped
    /// every row `eager_materialization`, and rewrote every inference revision to the running one.
    /// That shim could not distinguish a genuinely current bundle from a stale one — its whole job
    /// was to erase the difference — so a regression that re-staled the evidence would not have
    /// failed a single test that used it.
    fn packaged_bundle() -> EvidenceBundle {
        match sceneworks_core::memory_calibration::load_packaged_bundle()
            .expect("packaged bundle must parse")
        {
            BundleLoad::Ready(bundle) => bundle,
            BundleLoad::Stale(reason) => panic!("packaged bundle must be current: {reason:?}"),
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
        use sceneworks_core::memory_calibration::RequiredNullable;

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
                let predicted = predicted
                    .full_mut()
                    .expect("fixture has full phase telemetry");
                predicted.conditioning = predicted.conditioning.min(gib_to_bytes(peak_gib));
                predicted.denoise = gib_to_bytes(peak_gib);
                predicted.decode = predicted.decode.min(gib_to_bytes(peak_gib));
                predicted.overall = gib_to_bytes(peak_gib);
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
                FIXTURE_CLOSURE_DIGEST,
            ),
            Some(CalibrationGeometry {
                width: 768,
                height: 768,
                batch: 1,
                frames: 1,
            })
        );

        // sc-17774: currency, on the one path `memory_strategy` never sees. The alternative is only
        // ever formatted into a refusal message, so nothing downstream would catch a stale one —
        // and naming a geometry the very next request refuses for the identical staleness is worse
        // than naming none. Same binding, same bundle, same budget: only the live closure moves.
        assert_eq!(
            verified_lower_geometry(
                &bundle,
                calibration,
                &plan,
                &inputs,
                "text_to_image",
                fixture_budget(128.0),
                &"a".repeat(64),
            ),
            None,
            "a moved provider closure must stop the refusal from naming the lower geometry"
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
                FIXTURE_CLOSURE_DIGEST,
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
                FIXTURE_CLOSURE_DIGEST,
            ),
            None,
            "a geometry mutation must stop the refusal from naming the lower geometry"
        );
    }

    #[test]
    fn exact_infeasible_geometry_refuses_before_provider_and_names_only_verified_lower_geometry() {
        use sceneworks_core::memory_calibration::RequiredNullable;

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
        let predicted = predicted
            .full_mut()
            .expect("fixture has full phase telemetry");
        predicted.denoise = gib_to_bytes(6.0);
        predicted.overall = gib_to_bytes(6.0);
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
            Some(&fixture_closure_lookup),
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
    fn packaged_krea_1024_admits_by_estimate_and_refuses_below_the_widened_margins() {
        use gen_core::MemoryCalibrationIdentity;

        let plan = packaged_krea_plan();
        // Realistic component facts so the floor arithmetic is meaningful: conditioning 8 GiB,
        // transformer 60 GiB, decoder 4 GiB. Floors at 1024² (headroom = 2 GiB fixture anchor):
        //   staged (rung 1)       max(8, 64) + 2 = 66 GiB  -> widened 72.6 GiB
        //   bounded decode floor  (8 + 64)  + 2 = 74 GiB  -> widened 81.4 GiB (never used: the
        //                                                   measured 896² basis supplies a fitted
        //                                                   curve instead)
        // Fitted bounded-decode estimate from the 896² record: envelope 38.563 GiB scaled by
        // 1024²/896² = 50.37 GiB, widened by the 0.10 MLX estimate margin to 55.41 GiB.
        let mut generator = packaged_krea_generator();
        {
            let facts = &mut generator
                .contract
                .as_mut()
                .expect("Krea contract")
                .asset_facts;
            facts.conditioning_bytes = gib_to_bytes(8.0);
            facts.transformer_bytes = gib_to_bytes(60.0);
            facts.decoder_bytes = gib_to_bytes(4.0);
            facts.base_bytes = gib_to_bytes(72.0);
        }
        let generator = generator;
        let mut inputs = fixture_inputs(1024, 1024);
        inputs.overlay = Some("control:1".to_owned());
        let evaluate = |generator: &RequestGenerator, budget_gib: f64| {
            evaluate_request_with_budget_using_bundle(
                generator,
                &plan,
                &inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                fixture_budget(budget_gib),
                gib_to_bytes(130.0),
                0,
                &[],
                Some(&packaged_bundle()),
                Some(&packaged_krea_closure_lookup),
            )
        };

        // sc-18096 headline: 1024² has no measured cell, and before this story it REFUSED here.
        // Now the estimate ladder admits it. At 128 GiB the cheapest fitting rung is the staged
        // floor; the run context is legacy-scoped (no record id, no request-scoped ceiling).
        let admitted = evaluate(&generator, 128.0)
            .expect("an unmeasured geometry must be admitted by the estimate ladder");
        assert_eq!(
            admitted.context.selection.strategy,
            MemoryStrategy::StagedResidency,
            "the cheapest fitting estimate rung must win: {:?}",
            admitted.context.selection
        );
        assert_eq!(admitted.process_limit_bytes, None);
        assert_eq!(
            admitted.context.evidence_revision,
            REQUEST_EVIDENCE_REVISION
        );

        // At 60 GiB the staged floor (72.6 widened) no longer fits, and the FITTED bounded-decode
        // estimate extrapolated from the measured 896² cell (55.41 GiB widened) is selected — with
        // the measured cell's own sweep parameters, which a floor synthesis (built from the
        // smallest declared ranges and a 81.4 GiB widened peak) could not produce at this budget.
        let fitted = evaluate(&generator, 60.0)
            .expect("the fitted-curve estimate must admit where the floors cannot");
        assert_eq!(
            fitted.context.selection.strategy,
            MemoryStrategy::BoundedDecode
        );
        assert_eq!(
            fitted.context.selection.parameters.decode_tile_edge,
            Some(512),
            "the fitted estimate must carry the measured basis' parameters"
        );
        assert_eq!(
            fitted.context.selection.parameters.decode_overlap,
            Some(64),
            "the fitted estimate must carry the measured basis' parameters"
        );

        // Below every widened estimate the request still refuses — margins are load-bearing. No
        // verified alternative fits 40 GiB (the 768² cell needs its captured foreign reserve too),
        // so none may be named.
        let message = evaluate(&generator, 40.0)
            .expect_err("below the widened estimates the refusal must remain")
            .to_string();
        assert!(
            message.contains("needs") && message.contains("safely available"),
            "the refusal keeps the actionable needs/available format: {message}"
        );
        assert!(
            !message.contains("current verified alternative"),
            "no verified cell fits 40 GiB, so none may be named: {message}"
        );

        let mut exact_896 = inputs.clone();
        exact_896.width = 896;
        exact_896.height = 896;
        let exact = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &exact_896,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(83.0),
            gib_to_bytes(130.0),
            0,
            &[],
            Some(&packaged_bundle()),
            Some(&packaged_krea_closure_lookup),
        )
        .expect("the exact 896 cell fits once its 128 GiB-host reserve is normalized to 83 GiB");
        assert_eq!(
            exact.context.selection.strategy,
            MemoryStrategy::BoundedDecode
        );
        assert_eq!(
            exact.context.evidence_revision, "imc-2cd840a85ce33b4f22a9",
            "host normalization must admit the exact verified cell, not silently demote to estimates"
        );

        // A loaded provider whose calibration identity DRIFTED from the packaged records must not
        // receive fitted-curve candidates built from them (sc-18096 review, major finding). The
        // mutated fingerprint is deliberately WELL-FORMED (`-v1` satisfies the contract's version
        // token conformance rule), so the refusal below is the work of the basis identity gate in
        // `synthesize_estimate_ladder`, not an accidental contract-conformance `Invalid`. And it
        // runs at 60 GiB, a budget where NO lower alternative is named (both packaged cells need
        // their ~47 GiB captured foreign reserve), so `carries_verified_claim` is FALSE and the
        // fingerprint demotion at the admission seam never fires — the synthesis-side gate is the
        // only thing standing between the drifted provider and the fitted candidate. The
        // unmutated generator ADMITS at 60 (the fitted arm above), so the refusal is exactly the
        // mutation's doing.
        let mut mismatched_generator = packaged_krea_generator();
        {
            let contract = mismatched_generator
                .contract
                .as_mut()
                .expect("Krea contract");
            contract.asset_facts = generator.contract.as_ref().expect("facts").asset_facts;
            contract.calibration = Some(MemoryCalibrationIdentity::new(
                "mutated-loaded-provider-v1",
                gen_core::LoadShape::EagerMaterialization,
            ));
        }
        assert!(
            mismatched_generator
                .contract
                .as_ref()
                .expect("Krea contract")
                .conformance_errors()
                .is_empty(),
            "the mutated fingerprint must be conformance-CLEAN so this arm exercises the basis \
             identity gate, not format validation"
        );
        let message = evaluate(&mismatched_generator, 60.0)
            .expect_err("a fingerprint-drifted provider loses the measured basis and refuses")
            .to_string();
        assert!(
            message.contains("needs") && message.contains("safely available"),
            "the drifted provider's refusal is the floors-only Reject — the fitted candidate was \
             suppressed by the basis identity gate: {message}"
        );
        assert!(
            !message.contains("current verified alternative"),
            "a loaded-provider fingerprint mutation must suppress evidence-derived naming: {message}"
        );
        // The gate is precisely scoped to the MEASURED basis: the floors derive from the
        // contract's own asset facts, not from any record, so the drifted provider still admits
        // where a floor fits.
        let floor_admitted = evaluate(&mismatched_generator, 128.0)
            .expect("a drifted provider keeps its no-measured-basis floor estimates");
        assert_eq!(
            floor_admitted.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );

        // Dropping the rung's Implemented support removes both the fitted candidate and its floor:
        // an unimplemented rung is never estimate-admissible.
        let mut unimplemented_generator = packaged_krea_generator();
        {
            let contract = unimplemented_generator
                .contract
                .as_mut()
                .expect("Krea contract");
            contract.asset_facts = generator.contract.as_ref().expect("facts").asset_facts;
            contract
                .strategies
                .iter_mut()
                .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
                .expect("bounded decode capability")
                .support = gen_core::MemoryStrategySupport::Missing;
        }
        let message = evaluate(&unimplemented_generator, 60.0)
            .expect_err("an unimplemented rung is never estimate-admissible")
            .to_string();
        assert!(
            !message.contains("current verified alternative"),
            "a loaded-provider composition mutation must suppress evidence-derived naming: {message}"
        );

        // At the LIVE closure the verdict forks on the digest pair, derived rather than
        // hardcoded: a fitted-curve estimate may extrapolate only from CLOSURE-CURRENT records
        // (see `MeasuredRungBasis` — the 0.10 estimate margin was derived over same-closure
        // re-capture variance and cannot also absorb closure drift). While the pose-control pair
        // is current the 60 GiB request admits the fitted rung; once the closure moves, the
        // records may keep serving their own measured cells behind the stale margin (sc-18095)
        // but may NOT seed an extrapolation, so the request refuses on floors alone.
        let live = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(60.0),
            gib_to_bytes(130.0),
            0,
            &[],
            Some(&packaged_bundle()),
            Some(&fixture_closure_lookup),
        );
        if shipped_mlx_declared_closure_digest("krea_2_turbo")
            == live_mlx_closure_digest("krea_2_turbo_control")
        {
            let admitted =
                live.expect("at a current closure the fitted estimate admits the 60 GiB request");
            assert_eq!(
                admitted.context.selection.strategy,
                MemoryStrategy::BoundedDecode
            );
        } else {
            let message = live
                .expect_err(
                    "a stale-closure record must not seed a fitted extrapolation; floors alone \
                     cannot fit 60 GiB",
                )
                .to_string();
            assert!(
                message.contains("needs") && message.contains("safely available"),
                "the stale-basis refusal is the floors-only Reject: {message}"
            );
        }
    }

    /// The fixture generator with the FULL ladder implemented, including rung 4 with its
    /// deferred-materialization prerequisite — the shape an unmeasured provider needs for the
    /// sc-18096 estimate ladder to reach every rung.
    fn full_ladder_generator() -> RequestGenerator {
        use gen_core::{MemoryCalibrationIdentity, MemoryStrategySupport};

        let mut generator = fixture_generator();
        let contract = generator.contract.as_mut().expect("fixture contract");
        contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "fixture-formula-v2",
            gen_core::LoadShape::DeferredMaterialization,
        ));
        contract.lifecycle.transformer_window_materialization = true;
        let rung4 = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("rung 4 capability");
        rung4.support = MemoryStrategySupport::Implemented;
        rung4.parameters.transformer_window_sizes = vec![2, 4];
        generator
    }

    /// sc-18096 acceptance: an UNMEASURED provider (no calibration opt-in, no evidence bundle)
    /// under a small emulated unified-memory budget — the `SCENEWORKS_MLX_MEMORY_CAP_GB` scenario,
    /// driven through the same pure seam the cap feeds — selects a DEEP rung instead of refusing,
    /// and the selection translates to the right engine knobs.
    ///
    /// Floor arithmetic (fixture facts: base 3 GiB all transformer, headroom 2 fixed + 4 area):
    ///   resident        9 GiB modeled -> widened  9.9
    ///   staged floor    3 + 6 = 9     -> widened  9.9
    ///   decode floor    3 + 6 = 9     -> widened  9.9   (bounds transients, not weights)
    ///   attention floor 3 + 6 = 9     -> widened  9.9
    ///   rung 4 floor    0 + 6 = 6     -> widened  6.6   (windowed transformer leaves residency)
    /// An 8 GiB budget therefore admits exactly one rung: BoundedTransformerResidency.
    #[test]
    fn unmeasured_provider_under_a_small_budget_selects_a_deep_estimate_rung() {
        let generator = full_ladder_generator();
        let mut plan = fixture_plan();
        plan.calibration = MlxCalibrationConfig::Absent;
        let inputs = fixture_inputs(1024, 1024);
        let evaluation = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(8.0),
            gib_to_bytes(9.0),
            0,
            &[],
        )
        .expect("the estimate ladder must admit the deep rung instead of refusing");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
            "only rung 4's floor fits an 8 GiB budget: {:?}",
            evaluation.context.selection
        );
        // The floor synthesizes the most deeply bounding declared parameters.
        assert_eq!(
            evaluation.context.selection.parameters.decode_tile_edge,
            Some(512)
        );
        assert_eq!(
            evaluation.context.selection.parameters.decode_overlap,
            Some(128)
        );
        assert_eq!(
            evaluation.context.selection.parameters.attention_chunk_size,
            Some(256)
        );
        assert_eq!(
            evaluation
                .context
                .selection
                .parameters
                .transformer_window_size,
            Some(2),
            "the smallest declared window is the most bounding"
        );
        // And the selection translates to the engine knobs the engaged composition names:
        // rung 4 engages bounded decode, bounded attention, and block streaming — but NOT staged
        // residency, which the shared cost order keeps explicit-selection-only.
        assert!(evaluation.memory.tile_vae_decode);
        assert!(evaluation.memory.chunk_attention);
        assert!(evaluation.memory.stream_transformer_blocks);
        assert!(!evaluation.memory.stage_residency);
        // Legacy-scoped telemetry: an estimate admission never claims a verified record or a
        // request-scoped process ceiling.
        assert_eq!(evaluation.process_limit_bytes, None);
        assert_eq!(
            evaluation.context.evidence_revision,
            REQUEST_EVIDENCE_REVISION
        );

        // Mutation arm: at 6 GiB even the rung-4 widened floor (6.6 GiB) overflows, and the
        // refusal is the honest Reject quoting the widened requirement — proving the estimate
        // margin is applied on this path (a zeroed margin would admit 6.0 <= 6.0).
        let error = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(6.0),
            gib_to_bytes(9.0),
            0,
            &[],
        )
        .expect_err("below every widened estimate the request must refuse")
        .to_string();
        assert!(
            error.contains("needs 6.60 GiB"),
            "the refusal must quote the WIDENED rung-4 floor: {error}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn shipped_plain_krea_without_a_binding_preserves_the_request_on_estimate_admission() {
        fn fixture_spec(root: &std::path::Path, policy: OffloadPolicy) -> LoadSpec {
            for component in ["text_encoder", "transformer", "vae"] {
                let directory = root.join(component);
                std::fs::create_dir_all(&directory).unwrap();
                let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
                let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
                bytes.extend_from_slice(header);
                bytes.extend_from_slice(&0_f32.to_le_bytes());
                std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
            }
            for component in ["text_encoder", "transformer"] {
                std::fs::write(
                    root.join(component).join("config.json"),
                    r#"{"quantization":{"bits":4,"group_size":64}}"#,
                )
                .unwrap();
            }
            LoadSpec::new(WeightsSource::Dir(root.to_owned()))
                .with_quant(gen_core::Quant::Q4)
                .with_offload_policy(policy)
                .with_load_shape(gen_core::LoadShape::DeferredMaterialization)
        }

        fn contract(root: &std::path::Path, policy: OffloadPolicy) -> MemoryProviderContract {
            let mut contract = crate::inference_runtime::media()
                .memory_strategy_contract("krea_2_turbo", &fixture_spec(root, policy))
                .unwrap()
                .expect("the shipped plain Krea registry contract");
            // Preserve the shipped contract, composition, parameters and load shape while making
            // the pure selector arithmetic legible: a 6 GiB base consists of a 1 GiB conditioner
            // and 5 GiB DiT. With 6 GiB of request headroom, only the windowed composition fits an
            // 8 GiB constrained host after the canonical 10% estimate margin.
            contract.asset_facts.base_bytes = gib_to_bytes(6.0);
            contract.asset_facts.conditioning_bytes = gib_to_bytes(1.0);
            contract.asset_facts.transformer_bytes = gib_to_bytes(5.0);
            contract.asset_facts.decoder_bytes = 0;
            contract
        }

        fn generator(contract: MemoryProviderContract) -> RequestGenerator {
            RequestGenerator {
                descriptor: gen_core::ModelDescriptor {
                    id: "krea_2_turbo",
                    family: "krea",
                    backend: "mlx",
                    modality: gen_core::Modality::Image,
                    capabilities: gen_core::Capabilities::default(),
                    required_components: &[],
                    control_kinds: None,
                },
                contract: Some(contract),
            }
        }

        let root = tempfile::tempdir().unwrap();
        let plan = MlxRequestPlan {
            engine_id: "krea_2_turbo",
            model_id: "krea_2_turbo".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q4),
                component_precision_floors: &[],
            },
            asset_bytes: gib_to_bytes(6.0),
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
            activation_headroom_bytes: gib_to_bytes(6.0),
            fixed_reserve_bytes: gib_to_bytes(2.0),
            calibration: MlxCalibrationConfig::Absent,
        };
        let inputs = fixture_inputs(1024, 1024);

        let resident = evaluate_request_with_budget(
            &generator(contract(root.path(), OffloadPolicy::Resident)),
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(20.0),
            gib_to_bytes(12.0),
            0,
            &[],
        )
        .expect("a roomy host must keep the exact plain Krea request on resident admission");
        assert_eq!(
            resident.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert_eq!(
            resident.context.load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );

        let constrained = evaluate_request_with_budget(
            &generator(contract(root.path(), OffloadPolicy::Sequential)),
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Sequential,
            fixture_budget(8.0),
            gib_to_bytes(12.0),
            0,
            &[],
        )
        .expect("the unmeasured shipped Krea contract must reach the deep estimate rung");
        assert_eq!(
            constrained.context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency
        );
        assert_eq!(
            constrained.context.selection.parameters.decode_tile_edge,
            Some(512)
        );
        assert_eq!(
            constrained.context.selection.parameters.decode_overlap,
            Some(64)
        );
        assert_eq!(
            constrained
                .context
                .selection
                .parameters
                .attention_chunk_size,
            Some(67_108_864)
        );
        assert_eq!(
            constrained
                .context
                .selection
                .parameters
                .transformer_window_size,
            Some(1)
        );
        assert!(constrained.memory.stage_residency);
        assert!(constrained.memory.tile_vae_decode);
        assert!(constrained.memory.chunk_attention);
        assert!(constrained.memory.stream_transformer_blocks);
        for evaluation in [&resident, &constrained] {
            assert_eq!(evaluation.context.mode, MemoryMode::TextToImage);
            assert!(!evaluation.context.has_reference);
            assert_eq!(evaluation.context.geometry.reference_count, 0);
            assert!(evaluation.context.overlay.is_none());
            assert!(!evaluation.context.use_pid);
            assert_eq!(
                evaluation.context.evidence_revision,
                REQUEST_EVIDENCE_REVISION
            );
            assert_eq!(evaluation.process_limit_bytes, None);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn shipped_plain_sdxl_without_a_binding_preserves_the_three_rung_estimate_path() {
        fn fixture_spec(root: &std::path::Path, policy: OffloadPolicy) -> LoadSpec {
            for component in ["text_encoder", "text_encoder_2", "unet", "vae"] {
                let directory = root.join(component);
                std::fs::create_dir_all(&directory).unwrap();
                let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
                let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
                bytes.extend_from_slice(header);
                bytes.extend_from_slice(&0_f32.to_le_bytes());
                std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
            }
            std::fs::write(
                root.join("unet").join("config.json"),
                r#"{"quantization":{"bits":4,"group_size":64}}"#,
            )
            .unwrap();
            LoadSpec::new(WeightsSource::Dir(root.to_owned()))
                .with_quant(gen_core::Quant::Q4)
                .with_offload_policy(policy)
                .with_load_shape(gen_core::LoadShape::DeferredMaterialization)
        }

        fn contract(root: &std::path::Path, policy: OffloadPolicy) -> MemoryProviderContract {
            let mut contract = crate::inference_runtime::media()
                .memory_strategy_contract("sdxl", &fixture_spec(root, policy))
                .unwrap()
                .expect("the shipped plain SDXL registry contract");
            contract.asset_facts.base_bytes = gib_to_bytes(6.0);
            contract.asset_facts.conditioning_bytes = gib_to_bytes(1.0);
            contract.asset_facts.transformer_bytes = gib_to_bytes(5.0);
            contract.asset_facts.decoder_bytes = 0;
            contract
        }

        fn generator(contract: MemoryProviderContract) -> RequestGenerator {
            RequestGenerator {
                descriptor: gen_core::ModelDescriptor {
                    id: "sdxl",
                    family: "sdxl",
                    backend: "mlx",
                    modality: gen_core::Modality::Image,
                    capabilities: gen_core::Capabilities::default(),
                    required_components: &[],
                    control_kinds: None,
                },
                contract: Some(contract),
            }
        }

        let root = tempfile::tempdir().unwrap();
        let plan = MlxRequestPlan {
            engine_id: "sdxl",
            model_id: "sdxl".to_owned(),
            tier: MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q4),
                component_precision_floors: &[],
            },
            asset_bytes: gib_to_bytes(6.0),
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
            activation_headroom_bytes: gib_to_bytes(6.0),
            fixed_reserve_bytes: gib_to_bytes(2.0),
            calibration: MlxCalibrationConfig::Absent,
        };
        let inputs = fixture_inputs(1024, 1024);

        let resident = evaluate_request_with_budget(
            &generator(contract(root.path(), OffloadPolicy::Resident)),
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(20.0),
            gib_to_bytes(12.0),
            0,
            &[],
        )
        .expect("a roomy host must retain exact SDXL resident admission");
        assert_eq!(
            resident.context.selection.strategy,
            MemoryStrategy::Resident
        );

        let constrained = evaluate_request_with_budget(
            &generator(contract(root.path(), OffloadPolicy::Sequential)),
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Sequential,
            fixture_budget(8.0),
            gib_to_bytes(12.0),
            0,
            &[],
        )
        .expect("unmeasured SDXL must reach its deepest implemented estimate rung");
        assert_eq!(
            constrained.context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency
        );
        assert_eq!(
            constrained
                .context
                .selection
                .parameters
                .transformer_window_size,
            Some(1),
            "the current selector chooses the smallest-memory cadence from the provider domain"
        );
        assert!(constrained.memory.stage_residency);
        assert!(constrained.memory.stream_transformer_blocks);
        assert!(!constrained.memory.tile_vae_decode);
        assert!(!constrained.memory.chunk_attention);
        for evaluation in [&resident, &constrained] {
            assert_eq!(evaluation.context.mode, MemoryMode::TextToImage);
            assert!(!evaluation.context.has_reference);
            assert_eq!(evaluation.context.geometry.reference_count, 0);
            assert!(evaluation.context.overlay.is_none());
            assert!(!evaluation.context.use_pid);
            assert_eq!(
                evaluation.context.load_shape,
                gen_core::LoadShape::DeferredMaterialization
            );
            assert_eq!(
                evaluation.context.evidence_revision,
                REQUEST_EVIDENCE_REVISION
            );
            assert_eq!(evaluation.process_limit_bytes, None);
        }
    }

    /// sc-18094/sc-18096: `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` is honored at the
    /// synthesis seam. An extrapolation that moves the request peak onto a DIFFERENT phase than
    /// the measured cell's is rejected — the rung falls back to the no-measured-basis floor,
    /// which the constraint's scope sentence exempts — while a same-binding-phase extrapolation
    /// produces the fitted candidate at the area-scaled envelope.
    #[test]
    fn fitted_estimates_honor_the_measured_binding_phase_constraint() {
        use crate::memory_strategy::CandidateBasis;

        let generator = fixture_generator();
        let contract = generator.contract.as_ref().expect("fixture contract");
        let plan = fixture_plan();
        // Measured cell at 1024²: the CONDITIONING phase binds (16 GiB text-encoder peak against
        // 12 GiB denoise and 5 GiB decode) — the exact shape of the corpus example behind the
        // constraint.
        let basis = MeasuredRungBasis {
            rung: StrategyRung::BoundedDecode,
            parameters: gen_core::MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(128),
                ..Default::default()
            },
            engaged_composition: contract.engaged_composition(MemoryStrategy::BoundedDecode),
            load_shape: contract.load_shape,
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: "fixture-formula-v2".to_owned(),
            geometry: CalibrationGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            conditioning_peak_bytes: gib_to_bytes(16.0),
            denoise_peak_bytes: gib_to_bytes(12.0),
            decode_peak_bytes: gib_to_bytes(5.0),
            envelope_peak_bytes: gib_to_bytes(20.0),
            record_id: "imc-binding-phase-fixture".to_owned(),
        };

        // 2048²: area scale 4.0 pushes denoise to 48 GiB past the flat 16 GiB conditioning peak —
        // the binding phase flips, the fitted candidate is refused, and the rung falls back to
        // the weights+headroom floor.
        let flipped = synthesize_estimate_ladder(
            contract,
            &plan,
            "text_to_image",
            None,
            MemoryGeometry {
                width: 2048,
                height: 2048,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            None,
            std::slice::from_ref(&basis),
        );
        let decode = flipped
            .iter()
            .find(|estimate| estimate.selection.strategy == MemoryStrategy::BoundedDecode)
            .expect("the rung must still be estimate-admissible via the floor");
        assert_eq!(
            decode.basis,
            CandidateBasis::EstimateFloor,
            "a binding-phase flip must refuse the FITTED candidate per \
             ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE"
        );

        // 1120² (area scale ~1.196): denoise extrapolates to ~14.4 GiB, still under the
        // conditioning peak — no flip, so the fitted candidate is emitted at the area-scaled
        // envelope with the measured cell's parameters.
        let request_geometry = MemoryGeometry {
            width: 1120,
            height: 1120,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let fitted = synthesize_estimate_ladder(
            contract,
            &plan,
            "text_to_image",
            None,
            request_geometry,
            None,
            std::slice::from_ref(&basis),
        );
        let decode = fitted
            .iter()
            .find(|estimate| estimate.selection.strategy == MemoryStrategy::BoundedDecode)
            .expect("a same-binding-phase extrapolation must be admissible");
        assert_eq!(decode.basis, CandidateBasis::EstimateFittedCurve);
        assert_eq!(decode.selection.parameters.decode_tile_edge, Some(512));
        assert_eq!(decode.selection.parameters.decode_overlap, Some(128));
        let scale = (1120.0 * 1120.0) / (1024.0 * 1024.0);
        let expected_peak = (gib_to_bytes(20.0) as f64 * scale).ceil() as u64;
        assert_eq!(
            decode.evidence.predicted_peak_bytes, expected_peak,
            "the fitted estimate's raw peak is the area-scaled measured envelope (the estimate \
             margin is applied later, by the selector)"
        );
    }

    #[test]
    fn manifest_reader_distinguishes_absent_valid_unproven_and_malformed_opt_in() {
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

        // An opt-in the resolver cannot back with immutable provenance is a NON-COVERING state, not
        // a malformed manifest: the install simply cannot prove which artifact it holds (a receipt
        // written before download-time tree stamps is the common cause). It must degrade to the
        // conservative legacy selector rather than refuse the request, which would strand the
        // install with no repair path.
        let unavailable = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &spec,
            Some(&valid_manifest),
            None,
        );
        assert!(matches!(
            unavailable.calibration,
            MlxCalibrationConfig::Unproven
        ));
        assert_eq!(
            packaged_admission_route(
                &unavailable,
                &fixture_inputs(1024, 1024),
                "text_to_image",
                fixture_budget(8.0),
                FIXTURE_CLOSURE_DIGEST,
            )
            .expect("an unproven opt-in must degrade, never refuse")
            .path,
            AdmissionPath::Legacy
        );

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
            fixture_budget(8.0),
            FIXTURE_CLOSURE_DIGEST,
        )
        .expect_err("a malformed present opt-in must not collapse to packaged-empty legacy")
        .to_string()
        .contains("invalid MLX calibration opt-in"));
    }

    #[test]
    fn parameter_reader_is_closed_and_preserves_transformer_component() {
        let cumulative = crate::memory_strategy::default_engaged_composition(
            StrategyRung::BoundedTransformerResidency,
        );
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
        let parsed = parse_evidence_parameters(
            StrategyRung::BoundedTransformerResidency,
            &cumulative,
            &parameters,
        )
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
        assert!(parse_evidence_parameters(
            StrategyRung::BoundedTransformerResidency,
            &cumulative,
            &unknown,
        )
        .expect_err("unknown parameters fail closed")
        .contains("unknown"));

        let mut malformed = parameters.clone();
        malformed.insert(
            "transformerWindowComponent".to_owned(),
            serde_json::json!(12),
        );
        assert!(parse_evidence_parameters(
            StrategyRung::BoundedTransformerResidency,
            &cumulative,
            &malformed,
        )
        .expect_err("a non-string transformer component fails closed")
        .contains("transformerWindowComponent"));

        let mut unsupported = parameters;
        unsupported.insert(
            "transformerWindowComponent".to_owned(),
            serde_json::json!("vae"),
        );
        assert!(parse_evidence_parameters(
            StrategyRung::BoundedTransformerResidency,
            &cumulative,
            &unsupported,
        )
        .expect_err("an unknown transformer component fails closed")
        .contains("unsupported"));
    }

    /// One calibration opt-in shaped like a real published binding, with the rung, declared
    /// composition, parameters and materialization shape under test.
    fn calibration_json_for(
        rung: &str,
        engaged: Option<&[&str]>,
        parameters: Value,
        load_shape: &str,
    ) -> Value {
        let mut value = fixture_calibration_json("q4", "packed-q4");
        let object = value.as_object_mut().expect("calibration object");
        object.insert("rung".to_owned(), serde_json::json!(rung));
        object.insert("parameters".to_owned(), parameters);
        object.insert("loadShape".to_owned(), serde_json::json!(load_shape));
        if let Some(engaged) = engaged {
            object.insert("engagedRungs".to_owned(), serde_json::json!(engaged));
        }
        value
    }

    const SDXL_SHAPED_COMPOSITION: [&str; 3] = [
        "resident",
        "staged_residency",
        "bounded_transformer_residency",
    ];

    /// sc-17728. SDXL (sc-15525) and Kolors (sc-15521) both land rungs 0/1/4 `Implemented` with
    /// rungs 2 and 3 `Missing` — measured and withheld, not unattempted. Their published rung-4
    /// composition is `[resident, staged_residency, bounded_transformer_residency]`, and there is no
    /// honest `decodeTileEdge` or `attentionChunkSize` to name because no selectable strategy on
    /// those providers sets one. Keying the required set on the rung ORDINAL made that shape
    /// structurally unrepresentable, so neither family could ever record authoritative evidence.
    #[test]
    fn a_rung_four_binding_records_evidence_without_the_rungs_its_provider_withholds() {
        for component in ["dit", "text_encoder", "both"] {
            let binding = MlxCalibrationBinding::parse(
                &calibration_json_for(
                    "bounded_transformer_residency",
                    Some(&SDXL_SHAPED_COMPOSITION),
                    serde_json::json!({
                        "transformerWindowSize": 1,
                        "transformerWindowComponent": component,
                    }),
                    "deferred_materialization",
                ),
                0,
            )
            .unwrap_or_else(|error| {
                panic!("the published SDXL/Kolors rung-4 shape must bind evidence: {error}")
            });
            assert_eq!(
                binding.selection_parameters.transformer_window_size,
                Some(1)
            );
            assert!(binding
                .selection_parameters
                .transformer_window_component
                .is_some());
            // The withheld rungs contribute nothing rather than a fabricated default.
            assert_eq!(binding.selection_parameters.decode_tile_edge, None);
            assert_eq!(binding.selection_parameters.decode_overlap, None);
            assert_eq!(binding.selection_parameters.attention_chunk_size, None);
        }
    }

    /// The Z-Image/Qwen shape: rung 4 over a cumulative scratch composition that never engages
    /// staged residency. At the pinned contract revision rung 4's SHARED prerequisite is
    /// `LoadShape::DeferredMaterialization`, and `MemoryStrategy::engages` states outright that rung
    /// 4 does not engage rung 1; a rung-1 edge is provider-specific (mlx-gen-anima declares one,
    /// mlx-gen-z-image does not). So the rung-1-free composition must keep binding.
    #[test]
    fn a_rung_four_binding_without_rung_one_is_legitimate_when_the_contract_says_so() {
        let binding = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_transformer_residency",
                Some(&[
                    "resident",
                    "bounded_decode",
                    "bounded_attention",
                    "bounded_transformer_residency",
                ]),
                serde_json::json!({
                    "decodeTileEdge": 512,
                    "decodeOverlap": 64,
                    "attentionChunkSize": 67_108_864_u64,
                    "transformerWindowSize": 1,
                }),
                "deferred_materialization",
            ),
            0,
        )
        .expect("the checked-in Z-Image rung-4 plan composition must bind");
        assert_eq!(binding.selection_parameters.decode_tile_edge, Some(512));
        assert_eq!(
            binding.selection_parameters.transformer_window_size,
            Some(1)
        );
        // The shared cost-order default is the same composition, so the shape also binds with no
        // declaration at all — the rung-1-free reading is not an artefact of declaring it.
        assert!(crate::memory_strategy::default_engaged_composition(
            StrategyRung::BoundedTransformerResidency
        )
        .iter()
        .all(|rung| *rung != StrategyRung::StagedResidency));
    }

    /// Fail-closed in the other direction: the fix derives the required set, it does not make fields
    /// optional. A rung the composition does not engage owns no parameter, so naming one is an
    /// error rather than harmless noise.
    #[test]
    fn naming_a_withheld_rungs_parameters_is_an_error() {
        for (parameters, needle) in [
            (
                serde_json::json!({
                    "transformerWindowSize": 1,
                    "decodeTileEdge": 512,
                    "decodeOverlap": 64,
                }),
                "forbids decodeTileEdge",
            ),
            (
                serde_json::json!({
                    "transformerWindowSize": 1,
                    "attentionChunkSize": 67_108_864_u64,
                }),
                "forbids attentionChunkSize",
            ),
        ] {
            let error = MlxCalibrationBinding::parse(
                &calibration_json_for(
                    "bounded_transformer_residency",
                    Some(&SDXL_SHAPED_COMPOSITION),
                    parameters,
                    "deferred_materialization",
                ),
                0,
            )
            .expect_err("a withheld rung's parameters must not be nameable");
            assert!(error.contains(needle), "{error}");
        }
        // The same rule the other way round: a rung-2 binding cannot name the transformer scope.
        let error = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_decode",
                None,
                serde_json::json!({
                    "decodeTileEdge": 512,
                    "decodeOverlap": 64,
                    "transformerWindowComponent": "dit",
                }),
                "deferred_materialization",
            ),
            0,
        )
        .expect_err("a rung-2 binding does not engage rung 4");
        assert!(
            error.contains("forbids transformerWindowComponent"),
            "{error}"
        );
    }

    /// An ENGAGED rung must still name every parameter it owns. This is the guard the fix had to
    /// keep: derived, not relaxed.
    #[test]
    fn an_engaged_rung_must_still_name_every_parameter_it_owns() {
        for (engaged, parameters, needle) in [
            (
                SDXL_SHAPED_COMPOSITION.to_vec(),
                serde_json::json!({ "transformerWindowComponent": "dit" }),
                "requires transformerWindowSize",
            ),
            (
                vec![
                    "resident",
                    "bounded_decode",
                    "bounded_transformer_residency",
                ],
                serde_json::json!({ "decodeOverlap": 64, "transformerWindowSize": 1 }),
                "requires decodeTileEdge",
            ),
            (
                vec![
                    "resident",
                    "bounded_attention",
                    "bounded_transformer_residency",
                ],
                serde_json::json!({ "transformerWindowSize": 1 }),
                "requires attentionChunkSize",
            ),
        ] {
            let error = MlxCalibrationBinding::parse(
                &calibration_json_for(
                    "bounded_transformer_residency",
                    Some(&engaged),
                    parameters,
                    "deferred_materialization",
                ),
                0,
            )
            .expect_err("an engaged rung must name its own parameters");
            assert!(error.contains(needle), "{error}");
        }
    }

    /// The declaration itself is closed. A binding cannot dodge a required parameter by publishing
    /// an incoherent composition.
    #[test]
    fn a_declared_composition_is_itself_closed() {
        for (engaged, needle) in [
            (
                vec!["staged_residency", "bounded_transformer_residency"],
                "must contain resident",
            ),
            (
                vec!["resident", "staged_residency"],
                "must contain resident and the selected rung",
            ),
            (
                vec![
                    "resident",
                    "bounded_transformer_residency",
                    "staged_residency",
                ],
                "canonical ladder order",
            ),
            (
                vec!["resident", "resident", "bounded_transformer_residency"],
                "unique set",
            ),
        ] {
            let error = MlxCalibrationBinding::parse(
                &calibration_json_for(
                    "bounded_transformer_residency",
                    Some(&engaged),
                    serde_json::json!({ "transformerWindowSize": 1 }),
                    "deferred_materialization",
                ),
                0,
            )
            .expect_err("an incoherent composition must fail closed");
            assert!(error.contains(needle), "{error}");
        }
        // A cheaper selection cannot claim to have engaged a costlier rung's mechanism.
        let error = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_decode",
                Some(&[
                    "resident",
                    "bounded_decode",
                    "bounded_transformer_residency",
                ]),
                serde_json::json!({ "decodeTileEdge": 512, "decodeOverlap": 64 }),
                "deferred_materialization",
            ),
            0,
        )
        .expect_err("a rung-2 selection cannot engage rung 4");
        assert!(error.contains("costlier than the selected rung"), "{error}");
    }

    /// Regression guard for the shape that already worked. A provider with the whole ladder
    /// implemented (Anima) publishes the cumulative composition, and a binding that declares no
    /// composition at all still gets exactly the pre-sc-17728 fixed required set.
    #[test]
    fn the_full_ladder_shape_still_validates_unchanged() {
        let cumulative = serde_json::json!({
            "decodeTileEdge": 512,
            "decodeOverlap": 128,
            "attentionChunkSize": 256,
            "transformerWindowSize": 4,
            "transformerWindowComponent": "both",
        });
        for engaged in [
            None,
            Some(
                [
                    "resident",
                    "bounded_decode",
                    "bounded_attention",
                    "bounded_transformer_residency",
                ]
                .as_slice(),
            ),
            // Anima's MLX provider appends a rung-1 edge to the shared graph, so its rung-4
            // composition additionally engages staged residency.
            Some(
                [
                    "resident",
                    "staged_residency",
                    "bounded_decode",
                    "bounded_attention",
                    "bounded_transformer_residency",
                ]
                .as_slice(),
            ),
        ] {
            let binding = MlxCalibrationBinding::parse(
                &calibration_json_for(
                    "bounded_transformer_residency",
                    engaged,
                    cumulative.clone(),
                    "deferred_materialization",
                ),
                0,
            )
            .expect("the full-ladder shape must keep binding");
            assert_eq!(binding.selection_parameters.decode_tile_edge, Some(512));
            assert_eq!(binding.selection_parameters.decode_overlap, Some(128));
            assert_eq!(binding.selection_parameters.attention_chunk_size, Some(256));
            assert_eq!(
                binding.selection_parameters.transformer_window_size,
                Some(4)
            );
            assert_eq!(
                binding.selection_parameters.transformer_window_component,
                Some(TransformerComponent::Both)
            );
        }
        // Omitting the composition keeps the cumulative default, so an omitted lower-rung parameter
        // is still rejected exactly as before.
        let error = MlxCalibrationBinding::parse(
            &calibration_json_for(
                "bounded_transformer_residency",
                None,
                serde_json::json!({ "transformerWindowSize": 4 }),
                "deferred_materialization",
            ),
            0,
        )
        .expect_err("the cumulative default still requires the lower rungs' parameters");
        assert!(error.contains("requires decodeTileEdge"), "{error}");
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
                fixture_budget(8.0),
                FIXTURE_CLOSURE_DIGEST,
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
                fixture_budget(8.0),
                FIXTURE_CLOSURE_DIGEST,
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
                fixture_budget(8.0),
                FIXTURE_CLOSURE_DIGEST,
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
                fixture_budget(8.0),
                FIXTURE_CLOSURE_DIGEST,
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
                ..Default::default()
            },
        )
        .await
        .expect("real receipt writer");
        let resolved = crate::model_jobs::huggingface_receipt_weights(
            data.path(),
            repo,
            Some("fixture_model"),
            Some("default"),
            crate::model_jobs::ProvenanceRepair::Skip,
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
            FIXTURE_CLOSURE_DIGEST,
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
            FIXTURE_CLOSURE_DIGEST,
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
            FIXTURE_CLOSURE_DIGEST,
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
            FIXTURE_CLOSURE_DIGEST,
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
            Some(&fixture_closure_lookup),
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
            Some(&fixture_closure_lookup),
        )
        .expect_err("the exact covered 5 GiB cell must reject when only 3 GiB is safely available");
        let unfit = unfit.to_string();
        assert!(unfit.contains("smallest verified MLX host boundary"));
        assert!(
            unfit.contains("needs at least 8.00 GiB"),
            "the refusal must quote the static proportional boundary, not the 7.25 GiB reserve sum evaluated only at this 6 GiB host: {unfit}"
        );
    }

    #[test]
    fn persisted_uniform_tier_evidence_cannot_authorize_component_precision_floors() {
        use gen_core::{ComponentPrecisionFloor, PrecisionFloorComponent};

        const FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
            component: PrecisionFloorComponent::TransformerHead,
            selected_tier: gen_core::Quant::Q4,
            resident_tier: gen_core::Quant::Q8,
        }];
        let bundle = fixture_bundle();
        let mut generator = fixture_generator();
        generator.descriptor.capabilities.component_precision_floors = FLOORS;
        let evaluated = evaluate_request_with_budget_using_bundle(
            &generator,
            &fixture_plan(),
            &fixture_inputs(1024, 1024),
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(8.0),
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
            Some(&fixture_closure_lookup),
        )
        .expect("a mixed-precision provider falls back to its conservative resident estimate");

        assert_eq!(evaluated.process_limit_bytes, None);
        assert_eq!(
            evaluated.context.selection.strategy,
            MemoryStrategy::Resident,
            "a coarse persisted q4 record must not be relabeled as mixed q4/q8 evidence"
        );
        assert_eq!(
            evaluated.context.selection.tier.component_precision_floors,
            FLOORS
        );
    }

    #[test]
    fn provider_floor_binding_stays_inactive_for_q8_and_bf16() {
        use gen_core::{ComponentPrecisionFloor, PrecisionFloorComponent};

        const Q4_ONLY_FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
            component: PrecisionFloorComponent::TransformerHead,
            selected_tier: gen_core::Quant::Q4,
            resident_tier: gen_core::Quant::Q8,
        }];
        let mut generator = fixture_generator();
        generator.descriptor.capabilities.component_precision_floors = Q4_ONLY_FLOORS;

        for (quant, label) in [(Some(gen_core::Quant::Q8), "q8"), (None, "bf16")] {
            let mut plan = fixture_plan();
            plan.tier.quant = quant;
            plan.calibration = MlxCalibrationConfig::Absent;
            let evaluated = evaluate_request_with_budget_using_bundle(
                &generator,
                &plan,
                &fixture_inputs(1024, 1024),
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                fixture_budget(8.0),
                gib_to_bytes(4.0),
                0,
                &[],
                None,
                Some(&fixture_closure_lookup),
            )
            .unwrap_or_else(|error| panic!("{label} provider binding failed: {error}"));

            assert!(
                evaluated
                    .context
                    .selection
                    .tier
                    .component_precision_floors
                    .is_empty(),
                "Q4-only component floors must not relabel a {label} request"
            );
        }
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
            FIXTURE_CLOSURE_DIGEST,
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
            FIXTURE_CLOSURE_DIGEST,
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
            FIXTURE_CLOSURE_DIGEST,
        )
        .expect("provider mismatch falls back as drift");
        assert_eq!(
            wrong_provider.fallback_reason,
            Some(LegacyAdmissionReason::StaleIdentity),
            "the binding provider must match the actual engine route, not the catalog model id"
        );
    }

    /// sc-18101 #0 REGRESSION GUARD: a load-shape mismatch must DEGRADE to the estimate ladder,
    /// never refuse the request.
    ///
    /// `MemoryCalibrationIdentity` carries three fields, and gen-core's `optimized_eligibility`
    /// rejects a candidate whose `key.load_shape` disagrees with the contract's — returning
    /// `FingerprintMismatch`. The identity demotion above `evaluate_request_with_budget_using_bundle`
    /// used to compare only `abi` and `fingerprint`, so a shape mismatch sailed past it into
    /// `AdmissionPath::Evidence`, lost its only candidate inside `select_strategy`, and refused with
    /// "no structurally admissible MLX memory strategy" — with no estimate ladder behind it, because
    /// synthesis runs only on the Legacy route.
    ///
    /// That hole shipped: `mlx:qwen_image` q8 1024² was captured `eager_materialization` while
    /// `image_jobs::apply_measured_mlx_load_shape` forces `DeferredMaterialization` on every
    /// `qwen_image` directory load, so the flagship q8 route hard-refused its most common geometry on
    /// a 128 GiB machine. Verified on real weights at this commit and at the epic's base commit
    /// (`docs/epic-18093-end-to-end-validation-sc-18101.md`): pre-epic it was ADMITTED, post-epic
    /// REFUSED, post-fix ADMITTED again at the same predicted peak.
    ///
    /// The fixture reproduces it minimally: the bundle's records are `eager`, the contract is moved
    /// to `deferred` with its fingerprint and abi UNCHANGED, so `load_shape` is the only field that
    /// disagrees. The mutation arm below puts the shape back and shows the same request reaches
    /// `AdmissionPath::Evidence`, which is what makes this test discriminate rather than pass
    /// vacuously.
    #[test]
    fn a_load_shape_mismatch_degrades_to_estimates_instead_of_refusing() {
        use gen_core::MemoryCalibrationIdentity;

        let bundle = fixture_bundle();
        let plan = fixture_plan();
        let inputs = fixture_inputs(1024, 1024);

        let mut generator = fixture_generator();
        let contract = generator.contract.as_mut().expect("fixture contract");
        let fingerprint = contract
            .calibration
            .as_ref()
            .expect("fixture calibration")
            .fingerprint
            .clone();
        // ONLY the materialization shape moves. Same provider, same artifact, same abi, same
        // formula fingerprint — exactly the qwen_image production shape.
        contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            fingerprint,
            gen_core::LoadShape::DeferredMaterialization,
        ));

        let admitted = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(64.0),
            gib_to_bytes(9.0),
            0,
            &[],
            Some(&bundle),
            Some(&|_backend: &str, _provider: &str| Some(FIXTURE_CLOSURE_DIGEST.to_owned())),
        )
        .expect(
            "a load-shape mismatch must degrade to the estimate ladder, not refuse the request",
        );
        // Served from a synthesized estimate, not from a measurement: no verified record was
        // selected, so there is no request-scoped process ceiling.
        assert_eq!(
            admitted.process_limit_bytes, None,
            "a degraded cell must not claim a verified record's request-scoped ceiling"
        );

        // Mutation arm: restore the shape the records were captured under and the SAME request
        // reaches calibrated admission. Without this the test would also pass if the gate refused
        // everything for an unrelated reason.
        let mut matched = fixture_generator();
        matched
            .contract
            .as_mut()
            .expect("fixture contract")
            .load_shape = gen_core::LoadShape::EagerMaterialization;
        let route = evidence_admission_route(
            &bundle,
            &plan,
            &inputs,
            "text_to_image",
            fixture_budget(64.0),
            FIXTURE_CLOSURE_DIGEST,
        )
        .expect("the matching-shape cell routes without error");
        assert_eq!(
            route.path,
            AdmissionPath::Evidence,
            "with the captured shape restored the cell must reach calibrated admission: {:?}",
            route.fallback_reason
        );
        // …and the filter must SPARE a candidate the loaded provider can serve while dropping its
        // unusable sibling. This is the arm that separates the shipped per-candidate filter from a
        // whole-route demotion, and it is the fix's central design decision: `qwen_image` q8 ships
        // one eager and one deferred binding on the same route, so demoting the whole route would
        // silently discard the matching measurement and serve an estimate instead. An earlier
        // iteration of this fix did exactly that.
        //
        // Asserting `process_limit_bytes.is_some()` is NOT sufficient here and was the gap a review
        // caught: with only one record in the bundle there is no sibling to spare, so
        // `evidence.clear()` — a literal whole-route demotion — passed. The bundle now carries a
        // second record on `bounded_transformer_residency` captured `deferred_materialization`, and
        // the assertion is on the surviving SET.
        let mut two_binding_plan = plan.clone();
        let MlxCalibrationConfig::Valid(calibration) = &mut two_binding_plan.calibration else {
            panic!("the fixture plan opts in to calibration");
        };
        let mut sibling = fixture_binding_for(
            "q4",
            "packed-q4",
            StrategyRung::BoundedTransformerResidency,
            JsonObject::from_iter([
                ("decodeTileEdge".to_owned(), serde_json::json!(512)),
                ("decodeOverlap".to_owned(), serde_json::json!(128)),
                ("attentionChunkSize".to_owned(), serde_json::json!(65536)),
                ("transformerWindowSize".to_owned(), serde_json::json!(1)),
            ]),
        );
        // The binding must declare the shape its RECORD was captured under, exactly as the shipped
        // `qwen_image` q8 pair does — `EvidenceBundle::evidence_for` matches on it.
        sibling.query.load_shape =
            sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
        calibration.bindings.push(sibling);

        let route = evidence_admission_route(
            &bundle,
            &two_binding_plan,
            &inputs,
            "text_to_image",
            fixture_budget(64.0),
            FIXTURE_CLOSURE_DIGEST,
        )
        .expect("both bindings route without error");
        assert_eq!(
            route.evidence.len(),
            2,
            "the fixture must present BOTH a shape-matching and a mismatching candidate, or this \
             arm cannot tell a filter from a whole-route demotion"
        );

        let calibrated = evaluate_request_with_budget_using_bundle(
            &matched,
            &two_binding_plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(64.0),
            gib_to_bytes(9.0),
            0,
            &[],
            Some(&bundle),
            Some(&|_backend: &str, _provider: &str| Some(FIXTURE_CLOSURE_DIGEST.to_owned())),
        )
        .expect("the matching-shape cell is admitted");
        assert!(
            calibrated.process_limit_bytes.is_some(),
            "a shape-matching measured cell must still be served from its RECORD (which supplies \
             the request-scoped ceiling), not degraded to an estimate"
        );
        // The survivor is the shape-matching rung, and it survived ALONE: `clear()` in place of
        // `retain()` empties this and degrades the route, which the selection below would then have
        // served from an estimate with no `process_limit_bytes`.
        assert_eq!(
            calibrated.context.selection.strategy,
            MemoryStrategy::BoundedDecode,
            "the eager `bounded_decode` cell is the one this eager contract can serve; selecting \
             anything else means the filter dropped it"
        );
        assert_eq!(
            calibrated
                .context
                .selection
                .parameters
                .transformer_window_size,
            None,
            "the deferred rung-4 sibling must not have been selected"
        );

        // Third arm: the Resident EXEMPTION. `optimized_eligibility` short-circuits `Ok(())` for a
        // non-optimized selection before it compares load shapes (pinned against gen-core by
        // `gen_core_accepts_a_resident_cell_whose_load_shape_disagrees`), so a resident cell measured
        // under the other shape is one the downstream gate ACCEPTS — and this filter must not be
        // stricter than the gate it anticipates. Dropping the exemption would silently discard a
        // usable measurement, which is the same failure mode as the whole-route demotion.
        //
        // The bundle carries a resident cell captured `deferred_materialization`; bind it alongside
        // the eager `bounded_decode` cell and select under the EAGER contract. If the exemption
        // holds, the resident record survives the filter and the selector takes it first (rung
        // order), serving a RECORD — `process_limit_bytes` is `Some`. Without the exemption it is
        // filtered out and `bounded_decode` is selected instead.
        let mut resident_plan = plan.clone();
        let MlxCalibrationConfig::Valid(resident_calibration) = &mut resident_plan.calibration
        else {
            panic!("the fixture plan opts in to calibration");
        };
        let mut resident_binding =
            fixture_binding_for("q4", "packed-q4", StrategyRung::Resident, JsonObject::new());
        resident_binding.query.load_shape =
            sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
        resident_calibration.bindings.push(resident_binding);

        let exempted = evaluate_request_with_budget_using_bundle(
            &matched,
            &resident_plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(64.0),
            gib_to_bytes(9.0),
            0,
            &[],
            Some(&bundle),
            Some(&|_backend: &str, _provider: &str| Some(FIXTURE_CLOSURE_DIGEST.to_owned())),
        )
        .expect("the resident cell is admitted");
        assert_eq!(
            exempted.context.selection.strategy,
            MemoryStrategy::Resident,
            "a RESIDENT cell whose load shape disagrees must survive the filter, because \
             `optimized_eligibility` accepts it; filtering it would be stricter than the gate"
        );
        assert!(
            exempted.process_limit_bytes.is_some(),
            "the exempted resident cell must be served from its RECORD, not an estimate"
        );

        // Fourth arm: RUNG SUPPORT, load-bearing in production — on the real `qwen_image` q8 cell
        // the probe records `dropped=2 retained=0`, and the second drop is this one, not the shape
        // one. Since sc-18251 the `Implemented` requirement is enforced through the
        // `validate_selection` conjunct of `usable` (which fails for an undeclared or
        // non-`Implemented` rung before it looks at parameters); the composition conjunct also
        // drops this cell (a `Missing` rung is absent from its own live composition), so this arm
        // pins the OUTCOME — degrade, not a bare-`Missing` refusal — while each conjunct's own
        // isolating fixture lives in the sc-18251 tests below.
        //
        // Bind ONLY the deferred rung-4 cell and select under the DEFERRED contract, whose
        // `BoundedTransformerResidency` support is `Missing`. The shape now AGREES, so only the
        // new conjuncts can drop it. They must: otherwise the candidate reaches `select_strategy`,
        // which skips an unimplemented rung WITHOUT recording an exclusion, leaving every rung
        // empty and refusing with a bare `Missing` and no log line naming a cause.
        let mut rung_plan = plan.clone();
        let MlxCalibrationConfig::Valid(rung_calibration) = &mut rung_plan.calibration else {
            panic!("the fixture plan opts in to calibration");
        };
        let mut rung4_only = fixture_binding_for(
            "q4",
            "packed-q4",
            StrategyRung::BoundedTransformerResidency,
            JsonObject::from_iter([
                ("decodeTileEdge".to_owned(), serde_json::json!(512)),
                ("decodeOverlap".to_owned(), serde_json::json!(128)),
                ("attentionChunkSize".to_owned(), serde_json::json!(65536)),
                ("transformerWindowSize".to_owned(), serde_json::json!(1)),
            ]),
        );
        rung4_only.query.load_shape =
            sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
        rung_calibration.bindings = vec![rung4_only];
        assert_eq!(
            generator
                .contract
                .as_ref()
                .and_then(
                    |contract| contract.capability(MemoryStrategy::BoundedTransformerResidency)
                )
                .map(|capability| &capability.support),
            Some(&gen_core::MemoryStrategySupport::Missing),
            "precondition: the fixture contract does not implement rung 4"
        );
        let unsupported_rung = evaluate_request_with_budget_using_bundle(
            &generator,
            &rung_plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(64.0),
            gib_to_bytes(9.0),
            0,
            &[],
            Some(&bundle),
            Some(&|_backend: &str, _provider: &str| Some(FIXTURE_CLOSURE_DIGEST.to_owned())),
        )
        .expect(
            "a measured cell on a rung the loaded contract does not implement must degrade to the \
             estimate ladder, not refuse with a bare `Missing`",
        );
        assert_eq!(
            unsupported_rung.process_limit_bytes, None,
            "nothing measured survived, so the request is served from an estimate"
        );
    }

    /// sc-18101: the PREMISE of the Resident exemption in the shape filter, pinned against
    /// gen-core rather than restated.
    ///
    /// The filter above skips its load-shape test for `MemoryStrategy::Resident` because
    /// `optimized_eligibility` does: that gate short-circuits `Ok(())` for a non-optimized
    /// selection before it ever compares load shapes, so a resident cell measured under the other
    /// shape is one the gate ACCEPTS. The filter exists to anticipate the gate, not to tighten it —
    /// dropping such a cell would silently discard a usable measurement, which is the failure mode
    /// the per-candidate filter exists to avoid.
    ///
    /// This test asserts the premise directly. If a pin bump ever makes gen-core reject a resident
    /// cell on load shape, this reds and the exemption must be revisited — which is the only way a
    /// mirrored predicate can be kept honest against a dependency it does not own.
    #[test]
    fn gen_core_accepts_a_resident_cell_whose_load_shape_disagrees() {
        use gen_core::MemoryCalibrationIdentity;

        let generator = fixture_generator();
        let contract = generator.contract.as_ref().expect("fixture contract");
        assert_eq!(
            contract.load_shape,
            gen_core::LoadShape::EagerMaterialization,
            "fixture precondition"
        );

        // A resident cell measured under the OTHER shape.
        let (selection, mut resident) = resident_evidence(
            contract,
            fixture_plan().tier,
            "text_to_image",
            None,
            request_geometry(&fixture_inputs(1024, 1024)),
            gib_to_bytes(4.0),
            Some("fixture-formula-v2"),
        );
        resident.key.load_shape = gen_core::LoadShape::DeferredMaterialization;
        resident.conformance = gen_core::MemoryConformanceState::Verified;
        resident.dimensions = gen_core::MemoryEvidenceDimensions::VERIFIED;
        assert!(
            !selection.strategy.is_optimized(),
            "the exemption is about the non-optimized rung"
        );
        assert_eq!(
            resident.optimized_eligibility(contract),
            Ok(()),
            "gen-core must still accept a RESIDENT cell whose load shape disagrees; the filter's \
             exemption is built on this"
        );

        // …and the same disagreement on an OPTIMIZED rung is what the gate rejects, which is what
        // the filter anticipates. Without this arm the assertion above could pass vacuously. The
        // optimized candidate is taken from the real admission route rather than hand-built, so it
        // is structurally valid in every dimension EXCEPT the one under test.
        let route = evidence_admission_route(
            &fixture_bundle(),
            &fixture_plan(),
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(64.0),
            FIXTURE_CLOSURE_DIGEST,
        )
        .expect("the fixture route resolves");
        let mut optimized = route
            .evidence
            .iter()
            .map(|candidate| candidate.evidence.clone())
            .find(|evidence| evidence.key.strategy.is_optimized())
            .expect("the fixture bundle carries an optimized cell");
        assert_eq!(
            optimized.optimized_eligibility(contract),
            Ok(()),
            "precondition: the optimized cell is eligible BEFORE the shape is disturbed"
        );
        optimized.key.load_shape = gen_core::LoadShape::DeferredMaterialization;
        assert_eq!(
            optimized.optimized_eligibility(contract),
            Err(gen_core::MemoryEvidenceVerdict::FingerprintMismatch),
            "an OPTIMIZED cell with the same shape disagreement must be rejected"
        );

        // Belt and braces: the identity really does carry the shape as a third field, which is the
        // thing the worker demotion above compares only two of.
        let identity = MemoryCalibrationIdentity::new(
            "fixture-formula-v2",
            gen_core::LoadShape::EagerMaterialization,
        );
        assert_eq!(
            identity.load_shape,
            gen_core::LoadShape::EagerMaterialization
        );
    }

    /// sc-18251 leg 1: COMPOSITION DRIFT must degrade to the estimate ladder, never refuse — and
    /// the filter must stay per-candidate, sparing a sibling whose composition still agrees.
    ///
    /// The live provider grows a realization prerequisite that engages rung 1 in every rung-2
    /// request, so the live composition for `BoundedDecode` becomes
    /// `[Resident, StagedResidency, BoundedDecode]` while the record was captured under
    /// `[resident, bounded_decode]`. Same shape, same abi, same fingerprint, and the captured
    /// parameters still pass `validate_selection` (rung 1 owns no parameters, and the new edge is
    /// satisfied by its own engagement) — the preconditions below pin that, so of the filter's
    /// three legs ONLY the composition one can drop this candidate. That is what makes the
    /// mutation "composition conjunct deleted" observable: the candidate then reaches
    /// `select_strategy`, is excluded with `CompositionMismatch`, and the route refuses with "no
    /// structurally admissible MLX memory strategy" instead of degrading — the exact pre-sc-18251
    /// hole, which the retired `StaleIdentity` closure pre-demotion used to mask by demoting every
    /// drifted binding to Legacy before eligibility ran.
    #[test]
    fn a_composition_drift_degrades_to_estimates_and_spares_agreeing_siblings() {
        use gen_core::{MemoryPrerequisiteScope, MemoryStrategyPrerequisite};

        let bundle = fixture_bundle();
        let plan = fixture_plan();
        let inputs = fixture_inputs(1024, 1024);

        let mut generator = fixture_generator();
        let contract = generator.contract.as_mut().expect("fixture contract");
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        let contract = generator.contract.as_ref().expect("fixture contract");
        assert!(
            contract.conformance_errors().is_empty(),
            "the drifted contract must stay structurally conformant, or the selector refuses \
             everything for an unrelated reason and the arm passes vacuously"
        );
        // The captured composition no longer matches the live one…
        let captured = fixture_binding("q4", "packed-q4");
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode
            ],
            "precondition: the realization edge moved the live rung-2 composition"
        );
        // …while the OTHER two legs still pass, so only the composition conjunct can drop it.
        let captured_selection = MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: captured.selection_parameters,
            tier: plan.tier,
        };
        assert!(
            contract.validate_selection(&captured_selection).is_ok(),
            "precondition: the captured parameters remain valid under the drifted contract, \
             isolating the composition leg"
        );

        let degraded = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(64.0),
            gib_to_bytes(9.0),
            0,
            &[],
            Some(&bundle),
            Some(&|_backend: &str, _provider: &str| Some(FIXTURE_CLOSURE_DIGEST.to_owned())),
        )
        .expect("a composition drift must degrade to the estimate ladder, not refuse the request");
        assert_eq!(
            degraded.process_limit_bytes, None,
            "a degraded cell must not claim a verified record's request-scoped ceiling"
        );

        // Per-candidate arm: alongside the drifted rung-2 binding, bind the bundle's resident cell
        // (whose composition `[resident]` is untouched by the rung-2 edge). The survivor must be
        // served from its RECORD — `evidence.clear()` in place of `retain(usable)` empties the
        // route and this becomes an estimate with no process ceiling.
        let mut sibling_plan = plan.clone();
        let MlxCalibrationConfig::Valid(calibration) = &mut sibling_plan.calibration else {
            panic!("the fixture plan opts in to calibration");
        };
        let mut resident_binding =
            fixture_binding_for("q4", "packed-q4", StrategyRung::Resident, JsonObject::new());
        // The bundle's resident record was captured deferred; the binding must declare the shape
        // its RECORD was captured under, and the Resident shape exemption keeps it usable.
        resident_binding.query.load_shape =
            sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
        calibration.bindings.push(resident_binding);

        let spared = evaluate_request_with_budget_using_bundle(
            &generator,
            &sibling_plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(64.0),
            gib_to_bytes(9.0),
            0,
            &[],
            Some(&bundle),
            Some(&|_backend: &str, _provider: &str| Some(FIXTURE_CLOSURE_DIGEST.to_owned())),
        )
        .expect("the agreeing resident sibling is admitted");
        assert_eq!(
            spared.context.selection.strategy,
            MemoryStrategy::Resident,
            "the resident cell whose composition still agrees must survive the filter"
        );
        assert!(
            spared.process_limit_bytes.is_some(),
            "the spared sibling must be served from its RECORD, not an estimate"
        );
    }

    /// sc-18251 leg 2: PARAMETER-RANGE NARROWING must degrade to the estimate ladder, never
    /// refuse — and the filter must stay per-candidate.
    ///
    /// The live provider narrows its declared `decode_tile_edges` from `[512]` to `[256]`, so the
    /// record captured at 512 no longer passes `contract.validate_selection`. Shape and
    /// composition still agree (narrowing a range changes neither), pinned below — so only the
    /// selection-validity conjunct can drop this candidate, which is what makes the mutation
    /// "selection conjunct deleted" observable: the candidate then reaches `select_strategy`, is
    /// excluded with `Invalid` by the same `validate_selection` call downstream, and the route
    /// refuses instead of degrading.
    #[test]
    fn a_narrowed_parameter_range_degrades_to_estimates_and_spares_valid_siblings() {
        let bundle = fixture_bundle();
        let plan = fixture_plan();
        let inputs = fixture_inputs(1024, 1024);

        let mut generator = fixture_generator();
        let contract = generator.contract.as_mut().expect("fixture contract");
        let bounded_decode = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .expect("bounded decode capability");
        bounded_decode.parameters.decode_tile_edges = vec![256];
        let contract = generator.contract.as_ref().expect("fixture contract");
        assert!(
            contract.conformance_errors().is_empty(),
            "the narrowed contract must stay structurally conformant"
        );
        let captured = fixture_binding("q4", "packed-q4");
        let captured_selection = MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: captured.selection_parameters,
            tier: plan.tier,
        };
        assert!(
            contract.validate_selection(&captured_selection).is_err(),
            "precondition: the captured tile edge fell outside the narrowed range"
        );
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            vec![MemoryStrategy::Resident, MemoryStrategy::BoundedDecode],
            "precondition: narrowing a parameter range does not move the composition, isolating \
             the selection leg"
        );

        let degraded = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(64.0),
            gib_to_bytes(9.0),
            0,
            &[],
            Some(&bundle),
            Some(&|_backend: &str, _provider: &str| Some(FIXTURE_CLOSURE_DIGEST.to_owned())),
        )
        .expect(
            "a parameter-range narrowing must degrade to the estimate ladder, not refuse the \
             request",
        );
        assert_eq!(
            degraded.process_limit_bytes, None,
            "a degraded cell must not claim a verified record's request-scoped ceiling"
        );

        // Per-candidate arm: the resident sibling owns no parameters, so no narrowing can strand
        // it; it must survive and be served from its record.
        let mut sibling_plan = plan.clone();
        let MlxCalibrationConfig::Valid(calibration) = &mut sibling_plan.calibration else {
            panic!("the fixture plan opts in to calibration");
        };
        let mut resident_binding =
            fixture_binding_for("q4", "packed-q4", StrategyRung::Resident, JsonObject::new());
        resident_binding.query.load_shape =
            sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
        calibration.bindings.push(resident_binding);

        let spared = evaluate_request_with_budget_using_bundle(
            &generator,
            &sibling_plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(64.0),
            gib_to_bytes(9.0),
            0,
            &[],
            Some(&bundle),
            Some(&|_backend: &str, _provider: &str| Some(FIXTURE_CLOSURE_DIGEST.to_owned())),
        )
        .expect("the still-valid resident sibling is admitted");
        assert_eq!(
            spared.context.selection.strategy,
            MemoryStrategy::Resident,
            "the resident cell whose parameters remain valid must survive the filter"
        );
        assert!(
            spared.process_limit_bytes.is_some(),
            "the spared sibling must be served from its RECORD, not an estimate"
        );
    }

    // Exact LFS object sizes for the two control checkpoints the production FLUX router can actually
    // select. The audit creates sparse files with these logical lengths so the production
    // `weights_source_bytes` seam prices the real overlay without downloading it.
    #[cfg(target_os = "macos")]
    const FLUX1_CONTROL_BYTES: u64 = 4_281_779_224;
    #[cfg(target_os = "macos")]
    const FLUX2_CONTROL_BYTES: u64 = 8_232_506_680;
    // The pinned Lens bf16 turnkeys share this exact three-shard MXFP4 encoder on disk. The provider
    // footprint query, not a copied resident-byte constant, supplies its load-exact expansion.
    #[cfg(target_os = "macos")]
    const LENS_BF16_TEXT_ENCODER_DISK_BYTES: u64 = 4_845_744_456 + 4_774_186_632 + 4_154_656_824;

    #[cfg(target_os = "macos")]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum ResidentOnlyAuditSurface {
        Base,
        Edit,
        StrictControl,
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct SourceBoundAuditSurface {
        manifest_id: String,
        surface: ResidentOnlyAuditSurface,
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct SourceBoundAuditCell {
        manifest_id: String,
        provider_id: &'static str,
        tier: String,
        surface: ResidentOnlyAuditSurface,
        base_asset_bytes: u64,
        control_bytes: u64,
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct SourceBoundShippedTier {
        tier: String,
        base_asset_bytes: u64,
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum SourceBoundManifestDisposition {
        GenericSelector,
        PulidIdentityExcluded,
        MageSplitComponentsExcluded,
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct SourceBoundManifestClassification {
        manifest_id: String,
        family: String,
        disposition: SourceBoundManifestDisposition,
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct SourceBoundExcludedCell {
        manifest_id: String,
        family: String,
        provider_id: &'static str,
        tier: String,
        base_asset_bytes: u64,
        disposition: SourceBoundManifestDisposition,
    }

    /// Story-scoped product families for the Resident-only estimate-band audit. Individual
    /// manifest entries and tiers are deliberately not enumerated: the shipped manifest and
    /// production routers expand this family scope into the exact candidate inventory.
    #[cfg(target_os = "macos")]
    const RESIDENT_ONLY_AUDIT_FAMILIES: &[&str] = &[
        "boogu",
        "chroma",
        "flux",
        "flux2-dev",
        "flux2-klein",
        "ideogram",
        "lens",
        "sd3",
    ];

    #[cfg(target_os = "macos")]
    fn source_bound_manifest_classifications(
        models: &[Value],
    ) -> Result<Vec<SourceBoundManifestClassification>, String> {
        use SourceBoundManifestDisposition::{
            GenericSelector, MageSplitComponentsExcluded, PulidIdentityExcluded,
        };

        let mut classifications = Vec::new();
        let mut unique = std::collections::BTreeSet::new();
        for model in models {
            if model["type"] != "image" || !model["mlx"].is_object() {
                continue;
            }
            let manifest_id = model["id"]
                .as_str()
                .ok_or_else(|| "shipped MLX image entry has no string id".to_owned())?;
            let family = model["family"].as_str().ok_or_else(|| {
                format!("shipped MLX image entry {manifest_id} has no string family")
            })?;
            let resolved = crate::engines::mlx_model(manifest_id);
            let is_pulid = crate::image_jobs::is_pulid_flux_model(manifest_id);
            let routed_as_mage = resolved
                .as_ref()
                .is_some_and(|model| model.adapter_label() == "mlx_mage");
            let in_mage_family = family == "mage-flow";
            let in_audit_family = RESIDENT_ONLY_AUDIT_FAMILIES.contains(&family);
            if !in_audit_family && !is_pulid && !in_mage_family && !routed_as_mage {
                continue;
            }
            if !unique.insert(manifest_id) {
                return Err(format!(
                    "shipped manifest declares duplicate classified MLX image id {manifest_id}"
                ));
            }
            if source_bound_shipped_tiers(model)?.is_empty() {
                return Err(format!(
                    "production MLX image route {manifest_id} has no auditable shipped q4/q8/bf16 tier"
                ));
            }
            let disposition = if is_pulid {
                if family != "flux" || resolved.is_some() {
                    return Err(format!(
                        "{manifest_id} PuLID bespoke classification drifted: expected family=flux and no MODEL_TABLE row"
                    ));
                }
                PulidIdentityExcluded
            } else if in_mage_family || routed_as_mage {
                if !in_mage_family || !routed_as_mage {
                    return Err(format!(
                        "{manifest_id} Mage split-component classification drifted: expected family=mage-flow and a production mlx_mage route"
                    ));
                }
                MageSplitComponentsExcluded
            } else if in_audit_family {
                if resolved.is_none() {
                    return Err(format!(
                        "{manifest_id} is in an audited family but has neither a production MODEL_TABLE route nor an explicit bespoke exclusion"
                    ));
                }
                GenericSelector
            } else {
                return Err(format!(
                    "{manifest_id} reached the source classifier without a disposition"
                ));
            };
            classifications.push(SourceBoundManifestClassification {
                manifest_id: manifest_id.to_owned(),
                family: family.to_owned(),
                disposition,
            });
        }
        if classifications.is_empty() {
            return Err("shipped manifest exposes no classified MLX image routes".to_owned());
        }
        Ok(classifications)
    }

    #[cfg(target_os = "macos")]
    fn source_bound_generic_manifest_ids(
        classifications: &[SourceBoundManifestClassification],
    ) -> impl Iterator<Item = &str> {
        classifications.iter().filter_map(|classification| {
            (classification.disposition == SourceBoundManifestDisposition::GenericSelector)
                .then_some(classification.manifest_id.as_str())
        })
    }

    #[cfg(target_os = "macos")]
    fn source_bound_audit_surfaces_from_manifest_ids<'a>(
        manifest_ids: impl IntoIterator<Item = &'a str>,
    ) -> Result<Vec<SourceBoundAuditSurface>, String> {
        use ResidentOnlyAuditSurface::{Base, Edit, StrictControl};

        let mut declared = std::collections::BTreeSet::new();
        let mut surfaces = Vec::new();
        for manifest_id in manifest_ids {
            if !declared.insert(manifest_id) {
                return Err(format!(
                    "duplicate source-bound audit manifest declaration {manifest_id}"
                ));
            }
            if crate::engines::mlx_model(manifest_id).is_none() {
                return Err(format!(
                    "{manifest_id} no longer resolves through production MODEL_TABLE and the pinned registry"
                ));
            }
            surfaces.push(SourceBoundAuditSurface {
                manifest_id: manifest_id.to_owned(),
                surface: Base,
            });
            if let Some(edit_engine_id) = crate::image_jobs::flux2_edit_engine_id(manifest_id) {
                if !crate::image_jobs::flux2_edit_uses_provider_memory_safety(edit_engine_id) {
                    surfaces.push(SourceBoundAuditSurface {
                        manifest_id: manifest_id.to_owned(),
                        surface: Edit,
                    });
                }
            }
            if crate::image_jobs::mlx_flux_strict_control_engine_id(manifest_id).is_some() {
                surfaces.push(SourceBoundAuditSurface {
                    manifest_id: manifest_id.to_owned(),
                    surface: StrictControl,
                });
            }
        }
        Ok(surfaces)
    }

    #[cfg(target_os = "macos")]
    fn source_bound_audit_surfaces(
        models: &[Value],
    ) -> Result<Vec<SourceBoundAuditSurface>, String> {
        let classifications = source_bound_manifest_classifications(models)?;
        source_bound_audit_surfaces_from_manifest_ids(source_bound_generic_manifest_ids(
            &classifications,
        ))
    }

    fn set_sparse_len(path: &Path, bytes: u64) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("sparse fixture parent");
        }
        std::fs::File::create(path)
            .and_then(|file| file.set_len(bytes))
            .expect("sparse fixture size");
    }

    #[cfg(target_os = "macos")]
    fn set_sparse_valid_safetensor(path: &Path, bytes: u64) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut data_bytes = bytes
            .checked_sub(128)
            .ok_or_else(|| format!("{bytes} bytes is too small for a safetensors fixture"))?;
        let header = loop {
            let mut header = format!(
                r#"{{"weight":{{"dtype":"U8","shape":[{data_bytes}],"data_offsets":[0,{data_bytes}]}}}}"#
            );
            while (8 + header.len()) % 8 != 0 {
                header.push(' ');
            }
            let next_data_bytes = bytes
                .checked_sub(8 + header.len() as u64)
                .ok_or_else(|| format!("{bytes} bytes is too small for its safetensors header"))?;
            if next_data_bytes == data_bytes {
                break header;
            }
            data_bytes = next_data_bytes;
        };
        use std::io::Write;
        let mut file = std::fs::File::create(path).map_err(|error| error.to_string())?;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|()| file.write_all(header.as_bytes()))
            .and_then(|()| file.set_len(bytes))
            .map_err(|error| error.to_string())
    }

    #[cfg(target_os = "macos")]
    fn source_bound_shipped_tiers(model: &Value) -> Result<Vec<SourceBoundShippedTier>, String> {
        let downloads = model["downloads"].as_array().ok_or_else(|| {
            format!(
                "{} has no shipped downloads",
                model["id"].as_str().unwrap_or("<unknown>")
            )
        })?;
        fn supports_macos(download: &Value) -> bool {
            download["platforms"].as_array().is_none_or(|platforms| {
                platforms
                    .iter()
                    .any(|platform| platform.as_str() == Some("macos"))
            })
        }
        fn supported_variant(download: &Value) -> Option<&str> {
            download["variant"]
                .as_str()
                .and_then(|variant| matches!(variant, "q4" | "q8" | "bf16").then_some(variant))
        }
        let has_explicit_tiers = downloads
            .iter()
            .any(|download| supports_macos(download) && supported_variant(download).is_some());
        let inferred_tier = match model["mlx"]["quantize"].as_u64() {
            Some(4) => "q4",
            Some(8) => "q8",
            None | Some(0) => "bf16",
            Some(other) => {
                return Err(format!(
                    "{} has unsupported shipped mlx.quantize tier {other}",
                    model["id"].as_str().unwrap_or("<unknown>")
                ));
            }
        };
        let mut tiers = std::collections::BTreeMap::<String, u64>::new();
        for download in downloads {
            if !supports_macos(download) {
                continue;
            }
            let tier = if has_explicit_tiers {
                let Some(tier) = supported_variant(download) else {
                    continue;
                };
                tier
            } else if download["variant"].as_str() == Some("training") {
                continue;
            } else {
                inferred_tier
            };
            let bytes = download["estimatedSizeBytes"]
                .as_u64()
                .or_else(|| download["footprint"]["diskSizeBytes"].as_u64())
                .ok_or_else(|| {
                    format!(
                        "{} {tier} needs an auditable shipped byte size",
                        model["id"].as_str().unwrap_or("<unknown>")
                    )
                })?;
            let total = tiers.entry(tier.to_owned()).or_default();
            *total = total.checked_add(bytes).ok_or_else(|| {
                format!(
                    "{} {tier} shipped byte total overflows u64",
                    model["id"].as_str().unwrap_or("<unknown>")
                )
            })?;
        }
        Ok(tiers
            .into_iter()
            .map(|(tier, base_asset_bytes)| SourceBoundShippedTier {
                tier,
                base_asset_bytes,
            })
            .collect())
    }

    /// Expand every explicitly excluded manifest entry into the same route/tier accounting shape
    /// as the generic-selector audit. These are exclusions from the generic sparse-`LoadSpec`
    /// fixture, not exclusions from source-truth accounting: every shipped tier remains pinned.
    #[cfg(target_os = "macos")]
    fn source_bound_excluded_inventory_from_classifications(
        models: &[Value],
        classifications: &[SourceBoundManifestClassification],
    ) -> Result<Vec<SourceBoundExcludedCell>, String> {
        use SourceBoundManifestDisposition::{
            GenericSelector, MageSplitComponentsExcluded, PulidIdentityExcluded,
        };

        let mut cells = std::collections::BTreeSet::new();
        for classification in classifications {
            if classification.disposition == GenericSelector {
                continue;
            }
            let model = models
                .iter()
                .find(|model| model["id"] == classification.manifest_id)
                .ok_or_else(|| {
                    format!(
                        "missing explicitly excluded {} manifest entry",
                        classification.manifest_id
                    )
                })?;
            let provider_id = match classification.disposition {
                PulidIdentityExcluded => {
                    if !crate::image_jobs::is_pulid_flux_model(&classification.manifest_id)
                        || classification.family != "flux"
                        || crate::engines::mlx_model(&classification.manifest_id).is_some()
                    {
                        return Err(format!(
                            "{} no longer matches the PuLID identity-path exclusion",
                            classification.manifest_id
                        ));
                    }
                    "pulid_flux"
                }
                MageSplitComponentsExcluded => {
                    let resolved = crate::engines::mlx_model(&classification.manifest_id)
                        .ok_or_else(|| {
                            format!(
                                "{} no longer resolves through the production Mage route",
                                classification.manifest_id
                            )
                        })?;
                    if classification.family != "mage-flow"
                        || resolved.adapter_label() != "mlx_mage"
                    {
                        return Err(format!(
                            "{} no longer matches the Mage split-component exclusion",
                            classification.manifest_id
                        ));
                    }
                    resolved.engine_id()
                }
                GenericSelector => unreachable!("generic entries were skipped above"),
            };
            if crate::inference_runtime::media_descriptor(provider_id).is_none() {
                return Err(format!(
                    "explicitly excluded provider {provider_id} is absent from the pinned registry"
                ));
            }
            for shipped in source_bound_shipped_tiers(model)? {
                let cell = SourceBoundExcludedCell {
                    manifest_id: classification.manifest_id.clone(),
                    family: classification.family.clone(),
                    provider_id,
                    tier: shipped.tier,
                    base_asset_bytes: shipped.base_asset_bytes,
                    disposition: classification.disposition,
                };
                if !cells.insert(cell.clone()) {
                    return Err(format!(
                        "source-bound excluded inventory resolves a duplicate cell {cell:?}"
                    ));
                }
            }
        }
        Ok(cells.into_iter().collect())
    }

    /// Revalidate each exclusion against current production routing and current manifest tier
    /// accounting. Exact equality with the source-derived inventory below prevents a caller from
    /// silently dropping an excluded model or tier before this classifier runs.
    #[cfg(target_os = "macos")]
    fn source_bound_classified_excluded_inventory(
        models: &[Value],
        source_inventory: &[SourceBoundExcludedCell],
    ) -> Result<Vec<SourceBoundExcludedCell>, String> {
        let classifications = source_bound_manifest_classifications(models)?;
        let canonical =
            source_bound_excluded_inventory_from_classifications(models, &classifications)?
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>();
        let mut classified = Vec::new();
        for candidate in source_inventory {
            if !canonical.contains(candidate) {
                return Err(format!(
                    "source-bound exclusion no longer matches production source/routing/accounting: {candidate:?}"
                ));
            }
            classified.push(candidate.clone());
        }
        Ok(classified)
    }

    #[cfg(target_os = "macos")]
    fn require_exact_source_bound_excluded_inventory(
        expected: &[SourceBoundExcludedCell],
        actual: &[SourceBoundExcludedCell],
    ) -> Result<(), String> {
        let expected = expected
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let actual = actual
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        if expected == actual {
            return Ok(());
        }
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
        Err(format!(
            "source-bound excluded inventory mismatch; missing={missing:?}; unexpected={unexpected:?}"
        ))
    }

    #[cfg(target_os = "macos")]
    fn source_bound_audit_provider(
        surface: &SourceBoundAuditSurface,
    ) -> Result<&'static str, String> {
        match surface.surface {
            ResidentOnlyAuditSurface::Base => crate::engines::mlx_model(&surface.manifest_id)
                .map(|resolved| resolved.engine_id())
                .ok_or_else(|| {
                    format!(
                        "{} no longer resolves through production MODEL_TABLE and the pinned registry",
                        surface.manifest_id
                    )
                }),
            ResidentOnlyAuditSurface::Edit => {
                crate::image_jobs::flux2_edit_engine_id(&surface.manifest_id).ok_or_else(|| {
                    format!(
                        "{} has no production FLUX.2 edit route",
                        surface.manifest_id
                    )
                })
            }
            ResidentOnlyAuditSurface::StrictControl => {
                crate::image_jobs::mlx_flux_strict_control_engine_id(&surface.manifest_id)
                    .ok_or_else(|| {
                        format!(
                            "{} has no production FLUX strict-control route",
                            surface.manifest_id
                        )
                    })
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn source_bound_control_bytes(
        surface: &SourceBoundAuditSurface,
        provider_id: &str,
    ) -> Result<u64, String> {
        if surface.surface != ResidentOnlyAuditSurface::StrictControl {
            return Ok(0);
        }
        match provider_id {
            "flux1_dev_control" => Ok(FLUX1_CONTROL_BYTES),
            "flux2_dev_control" => Ok(FLUX2_CONTROL_BYTES),
            other => Err(format!(
                "production strict-control provider {other} has no audited control checkpoint size"
            )),
        }
    }

    #[cfg(target_os = "macos")]
    fn audit_load_spec(
        provider_id: &str,
        tier: &str,
        base_asset_bytes: u64,
        control_bytes: u64,
    ) -> Result<(tempfile::TempDir, LoadSpec), String> {
        use gen_core::Quant;

        let fixture = tempfile::tempdir().map_err(|error| error.to_string())?;
        let weights = fixture.path().join("weights");
        if provider_id == "lens" && tier == "q4" {
            // The production worker opts this one route into the load-exact deferred contract.
            // Build the same component/config shape its registry contract inspects; a flat sparse
            // file would make the lookup fall back and falsely classify measured Lens q4 as
            // Resident-only.
            let text_encoder_bytes = base_asset_bytes / 3;
            let transformer_bytes = base_asset_bytes / 3;
            let vae_bytes = base_asset_bytes - text_encoder_bytes - transformer_bytes;
            for (component, bytes) in [
                ("text_encoder", text_encoder_bytes),
                ("transformer", transformer_bytes),
                ("vae", vae_bytes),
            ] {
                set_sparse_valid_safetensor(
                    &weights.join(component).join("model.safetensors"),
                    bytes,
                )?;
            }
            for component in ["text_encoder", "transformer"] {
                std::fs::write(
                    weights.join(component).join("config.json"),
                    r#"{"quantization":{"bits":4,"group_size":64}}"#,
                )
                .map_err(|error| error.to_string())?;
            }
        } else if matches!(provider_id, "lens" | "lens_turbo") && tier == "bf16" {
            if base_asset_bytes < LENS_BF16_TEXT_ENCODER_DISK_BYTES {
                return Err(format!(
                    "Lens bf16 asset {base_asset_bytes} is smaller than its source encoder"
                ));
            }
            set_sparse_len(
                &weights.join("text_encoder/model.safetensors"),
                LENS_BF16_TEXT_ENCODER_DISK_BYTES,
            );
            set_sparse_len(
                &weights.join("transformer/model.safetensors"),
                base_asset_bytes - LENS_BF16_TEXT_ENCODER_DISK_BYTES,
            );
            std::fs::write(
                weights.join("text_encoder/config.json"),
                r#"{"quantization_config":{"quant_method":"mxfp4"}}"#,
            )
            .map_err(|error| error.to_string())?;
        } else {
            set_sparse_len(&weights.join("model.safetensors"), base_asset_bytes);
        }
        let spec = match tier {
            "q4" => LoadSpec::new(WeightsSource::Dir(weights)).with_quant(Quant::Q4),
            "q8" => LoadSpec::new(WeightsSource::Dir(weights)).with_quant(Quant::Q8),
            "bf16" => LoadSpec::new(WeightsSource::Dir(weights)),
            other => return Err(format!("unsupported audited tier {other}")),
        };
        let spec = if control_bytes == 0 {
            spec
        } else {
            let control = fixture.path().join("control.safetensors");
            set_sparse_len(&control, control_bytes);
            spec.with_control(WeightsSource::File(control))
        };
        Ok((
            fixture,
            crate::image_jobs::apply_measured_mlx_load_shape(provider_id, spec),
        ))
    }

    #[cfg(target_os = "macos")]
    fn source_bound_contract(
        provider_id: &'static str,
        spec: &LoadSpec,
    ) -> Result<(MemoryProviderContract, bool), String> {
        crate::inference_runtime::media()
            .memory_strategy_contract(provider_id, spec)
            .map_err(|error| error.to_string())
            .map(|contract| {
                contract.map_or_else(
                    || {
                        (
                            MemoryProviderContract::compatibility_default(
                                provider_id,
                                MemoryBackendRealization::MlxMetal {
                                    bounded_wired_residency: true,
                                    lazy_or_mmap_materialization: true,
                                    explicit_evaluation_and_synchronization: true,
                                    cache_eviction: true,
                                },
                            ),
                            false,
                        )
                    },
                    |contract| (contract, true),
                )
            })
    }

    #[cfg(target_os = "macos")]
    fn implemented_optimized_strategies(contract: &MemoryProviderContract) -> Vec<MemoryStrategy> {
        MemoryStrategy::ALL
            .into_iter()
            .filter(|strategy| strategy.is_optimized())
            .filter(|strategy| {
                matches!(
                    contract
                        .capability(*strategy)
                        .map(|capability| &capability.support),
                    Some(gen_core::MemoryStrategySupport::Implemented)
                )
            })
            .collect()
    }

    /// The shipped manifest is the product-side binding of provider capability to a concrete tier.
    /// Provider contract construction may inspect the on-disk component tree, which this audit
    /// represents with sparse total-size files; use the manifest's explicit optimized-rung tier
    /// declaration to prevent that deliberately minimal filesystem fixture from turning an
    /// optimized clean route (Chroma/FLUX/Klein) into a fake Resident-only cell. A manifest may now
    /// also declare an exhaustive Resident-only contract (FLUX.2 q4/q8, SC-18218); that is evidence
    /// that the tier belongs in this audit, not a reason to exclude it.
    #[cfg(target_os = "macos")]
    fn manifest_declares_optimized_tier(model: &Value, tier: &str) -> bool {
        model["mlx"]["memoryStrategyContract"]["implementations"]
            .as_array()
            .is_some_and(|implementations| {
                implementations.iter().any(|implementation| {
                    implementation["rung"]
                        .as_str()
                        .is_some_and(|rung| rung != "resident")
                        && implementation["tiers"].as_array().is_some_and(|tiers| {
                            tiers.iter().any(|item| item.as_str() == Some(tier))
                        })
                })
            })
    }

    #[cfg(target_os = "macos")]
    fn source_bound_audit_inventory_from_surfaces(
        models: &[Value],
        source_surfaces: &[SourceBoundAuditSurface],
    ) -> Result<Vec<SourceBoundAuditCell>, String> {
        let mut surfaces = std::collections::BTreeSet::new();
        let mut cells = std::collections::BTreeSet::new();
        let mut cell_keys = std::collections::BTreeSet::new();
        for expected in source_surfaces {
            let model = models
                .iter()
                .find(|model| model["id"] == expected.manifest_id)
                .ok_or_else(|| {
                    format!("missing shipped {} manifest entry", expected.manifest_id)
                })?;
            if model["type"] != "image" {
                return Err(format!(
                    "{} is no longer a shipped image model",
                    expected.manifest_id
                ));
            }
            let provider_id = source_bound_audit_provider(expected)?;
            let control_bytes = source_bound_control_bytes(expected, provider_id)?;
            if !surfaces.insert((expected.manifest_id.clone(), expected.surface, provider_id)) {
                return Err(format!(
                    "source-bound resident-only inventory resolves a duplicate {:?} route {} ({provider_id})",
                    expected.surface, expected.manifest_id
                ));
            }
            if crate::inference_runtime::media_descriptor(provider_id).is_none() {
                return Err(format!(
                    "source-bound resident-only provider {provider_id} is absent from the pinned registry"
                ));
            }
            for shipped in source_bound_shipped_tiers(model)? {
                let tier = shipped.tier.as_str();
                let base_asset_bytes = shipped.base_asset_bytes;
                let key = (
                    expected.manifest_id.to_owned(),
                    provider_id,
                    tier.to_owned(),
                    expected.surface,
                );
                if !cell_keys.insert(key) {
                    return Err(format!(
                        "source-bound resident-only inventory resolves a duplicate cell {} ({provider_id}) {tier} {:?}",
                        expected.manifest_id, expected.surface
                    ));
                }
                cells.insert(SourceBoundAuditCell {
                    manifest_id: expected.manifest_id.to_owned(),
                    provider_id,
                    tier: tier.to_owned(),
                    surface: expected.surface,
                    base_asset_bytes,
                    control_bytes,
                });
            }
        }
        Ok(cells.into_iter().collect())
    }

    #[cfg(target_os = "macos")]
    fn require_exact_source_bound_inventory(
        expected: &[SourceBoundAuditCell],
        actual: &[SourceBoundAuditCell],
    ) -> Result<(), String> {
        let expected = expected
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let actual = actual
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        if expected == actual {
            return Ok(());
        }
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
        Err(format!(
            "source-bound audit inventory mismatch; missing={missing:?}; unexpected={unexpected:?}"
        ))
    }

    #[cfg(target_os = "macos")]
    fn source_bound_resident_only_cells_from_inventory(
        models: &[Value],
        source_inventory: &[SourceBoundAuditCell],
    ) -> Result<(Vec<SourceBoundAuditCell>, Vec<SourceBoundAuditCell>), String> {
        let mut classified = Vec::new();
        let mut cells = Vec::new();
        for candidate in source_inventory {
            let model = models
                .iter()
                .find(|model| model["id"] == candidate.manifest_id)
                .ok_or_else(|| {
                    format!("missing shipped {} manifest entry", candidate.manifest_id)
                })?;
            if candidate.surface != ResidentOnlyAuditSurface::StrictControl
                && manifest_declares_optimized_tier(model, &candidate.tier)
            {
                classified.push(candidate.clone());
                continue;
            }
            let (_fixture, spec) = audit_load_spec(
                candidate.provider_id,
                &candidate.tier,
                candidate.base_asset_bytes,
                candidate.control_bytes,
            )?;
            let (contract, _) =
                source_bound_contract(candidate.provider_id, &spec).map_err(|error| {
                    format!(
                        "{} ({}) {} {:?} contract lookup failed: {error}",
                        candidate.manifest_id,
                        candidate.provider_id,
                        candidate.tier,
                        candidate.surface
                    )
                })?;
            if implemented_optimized_strategies(&contract).is_empty() {
                cells.push(candidate.clone());
            }
            classified.push(candidate.clone());
        }
        Ok((classified, cells))
    }

    /// sc-18251 resident-only audit: derive every reachable candidate through the production base,
    /// edit, and FLUX strict-control routers, retain only tiers whose pinned loaded-provider contract
    /// is actually Resident-only, then drive those exact cells through the production request plan and
    /// selector. A base provider never receives an invented control checkpoint.
    #[cfg(target_os = "macos")]
    #[test]
    fn shipped_resident_only_mlx_estimate_band_audit_uses_the_production_budget_path() {
        let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
            include_str!("../../../config/manifests/builtin.models.jsonc"),
        ))
        .expect("builtin.models.jsonc parses");
        let models = manifest["models"].as_array().expect("manifest models");
        let classifications =
            source_bound_manifest_classifications(models).expect("source-bound classifications");
        assert_eq!(
            classifications.len(),
            26,
            "the source-derived generic-plus-excluded model inventory changed"
        );
        let generic_manifest_ids =
            source_bound_generic_manifest_ids(&classifications).collect::<Vec<_>>();
        assert_eq!(
            generic_manifest_ids.len(),
            19,
            "the generic-selector model inventory changed"
        );
        let surfaces = source_bound_audit_surfaces_from_manifest_ids(generic_manifest_ids)
            .expect("source-bound audit surfaces");
        let source_inventory = source_bound_audit_inventory_from_surfaces(models, &surfaces)
            .expect("source-bound candidate inventory");
        assert_eq!(
            source_inventory.len(),
            62,
            "the source-derived candidate inventory changed; update the recorded audit result"
        );
        let excluded_inventory =
            source_bound_excluded_inventory_from_classifications(models, &classifications)
                .expect("source-bound excluded inventory");
        let pulid_excluded = excluded_inventory
            .iter()
            .filter(|cell| {
                cell.disposition == SourceBoundManifestDisposition::PulidIdentityExcluded
            })
            .count();
        let mage_excluded = excluded_inventory
            .iter()
            .filter(|cell| {
                cell.disposition == SourceBoundManifestDisposition::MageSplitComponentsExcluded
            })
            .count();
        assert_eq!(pulid_excluded, 3, "PuLID must account for q4/q8/bf16");
        assert_eq!(
            mage_excluded, 18,
            "the six Mage variants must each account for q4/q8/bf16"
        );
        assert_eq!(
            source_inventory.len() + pulid_excluded,
            65,
            "every route/tier cell in the eight-family story scope must be classified"
        );
        let classified_excluded =
            source_bound_classified_excluded_inventory(models, &excluded_inventory)
                .expect("classified source-bound exclusions");
        require_exact_source_bound_excluded_inventory(&excluded_inventory, &classified_excluded)
            .expect("the executable audit must classify the exact excluded inventory");
        let (classified_inventory, cells) =
            source_bound_resident_only_cells_from_inventory(models, &source_inventory)
                .expect("source-bound resident-only inventory");
        require_exact_source_bound_inventory(&source_inventory, &classified_inventory).expect(
            "the executable audit must classify the exact source-bound candidate inventory",
        );
        assert_eq!(
            cells.len(),
            35,
            "the source-derived Resident-only inventory changed; update the recorded audit result"
        );
        let legacy_reserve_bytes = gib_to_bytes(crate::fit_gate::legacy_unified_reserve(48.0).gb);
        let hosts = [48_u64, 64, 96, 128];
        let mut flips = Vec::new();
        let mut audited_cells = std::collections::BTreeSet::new();

        for cell in &cells {
            assert!(
                audited_cells.insert((
                    cell.manifest_id.clone(),
                    cell.provider_id,
                    cell.tier.clone(),
                    cell.surface,
                )),
                "the executable audit must reject duplicate route-tier cells: {cell:?}"
            );
            let (_fixture, spec) = audit_load_spec(
                cell.provider_id,
                &cell.tier,
                cell.base_asset_bytes,
                cell.control_bytes,
            )
            .expect("source-bound audit spec");
            let media = crate::inference_runtime::media();
            let provider_footprint = media
                .footprint(cell.provider_id, &spec)
                .expect("source-bound provider footprint");
            let activation_anchor_bytes = media
                .activation_memory_bytes_1024(cell.provider_id)
                .expect("source-bound activation query");
            let plan = MlxRequestPlan::for_spec_and_manifest_with_provider_facts(
                cell.provider_id,
                &cell.manifest_id,
                &spec,
                None,
                None,
                provider_footprint,
                activation_anchor_bytes,
            );
            assert_eq!(
                plan.folded_control_bytes, cell.control_bytes,
                "{} ({}) {}: deleting production control-source accounting must make this \
                     audit red",
                cell.manifest_id, cell.provider_id, cell.tier
            );
            assert!(
                plan.asset_bytes >= cell.base_asset_bytes + cell.control_bytes,
                "{} ({}) {}: provider materialization cannot erase shipped base/control bytes",
                cell.manifest_id,
                cell.provider_id,
                cell.tier
            );
            let geometry = MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            };
            let raw_incremental_peak = plan.generic_headroom_bytes(geometry);
            let widened_incremental_peak = (raw_incremental_peak as f64
                * (1.0 + crate::ladder_margin_policy::MLX_ESTIMATE_MARGIN))
                .ceil()
                .clamp(0.0, u64::MAX as f64) as u64;
            // These are precisely the cells for which no optimized product contract applies. Model
            // their legacy Resident fallback with the same compatibility contract production uses
            // when a provider has no applicable adopted cell, and bind its aggregate base fact to
            // the source-derived load plan so post-load cache credit is exact.
            let mut contract = MemoryProviderContract::compatibility_default(
                cell.provider_id,
                MemoryBackendRealization::MlxMetal {
                    bounded_wired_residency: true,
                    lazy_or_mmap_materialization: true,
                    explicit_evaluation_and_synchronization: true,
                    cache_eviction: true,
                },
            );
            contract.asset_facts.base_bytes = plan.asset_bytes;
            contract.asset_facts.transformer_bytes = plan.asset_bytes;
            assert!(
                contract.conformance_errors().is_empty(),
                "audit contract must stay conformant for {cell:?}: {:?}",
                contract.conformance_errors()
            );
            let generator = RequestGenerator {
                descriptor: crate::inference_runtime::media_descriptor(cell.provider_id)
                    .expect("source-bound provider descriptor"),
                contract: Some(contract),
            };
            let legacy_total_peak = plan.generic_total_peak_bytes(geometry);

            for host_gib in hosts {
                let host_bytes = gib_to_bytes(host_gib as f64);
                // This is exactly the live legacy budget after the generator has loaded:
                // committed provider assets remain on the available side, while the modeled
                // peak receives the matching provider-resident cache credit.
                let available_incremental = host_bytes
                    .saturating_sub(plan.asset_bytes)
                    .saturating_sub(legacy_reserve_bytes);
                let admitted_before_margin = raw_incremental_peak <= available_incremental;
                let admitted_now = widened_incremental_peak <= available_incremental;
                if admitted_before_margin && !admitted_now {
                    flips.push((
                        cell.manifest_id.as_str(),
                        cell.provider_id,
                        cell.tier.as_str(),
                        host_gib,
                        plan.asset_bytes,
                        raw_incremental_peak,
                        widened_incremental_peak,
                    ));
                }

                // Exercise the production selector seam too; the arithmetic above names the
                // historical no-margin counterfactual, while this call proves the current side
                // uses cache credit + reserve + EstimateFloor margin together.
                let evaluated = evaluate_request_with_budget(
                    &generator,
                    &plan,
                    &fixture_inputs(1024, 1024),
                    MemoryCacheState::Cold,
                    OffloadPolicy::Resident,
                    MemoryBudget {
                        total_bytes: host_bytes,
                        committed_bytes: plan.asset_bytes,
                        reclaimable_bytes: 0,
                        reserved_headroom_bytes: legacy_reserve_bytes,
                    },
                    legacy_total_peak,
                    0,
                    &[],
                );
                assert_eq!(
                    evaluated.is_ok(),
                    admitted_now,
                    "production-path result drifted for {} ({}) {} on {host_gib} GiB: \
                         {evaluated:?}",
                    cell.manifest_id,
                    cell.provider_id,
                    cell.tier
                );
            }
        }

        assert_eq!(
            audited_cells.len(),
            cells.len(),
            "the executable budget walk must cover every unique source-derived cell exactly"
        );

        assert_eq!(
            flips
                .iter()
                .map(|(model, provider, tier, host, ..)| (*model, *provider, *tier, *host))
                .collect::<Vec<_>>(),
            vec![
                ("sd3_5_large", "sd3_5_large", "q8", 48),
                ("sd3_5_large_turbo", "sd3_5_large_turbo", "q8", 48,),
            ],
            "the source-bound resident-only audit changed; update the recorded result, \
             not only this expectation: {flips:?}"
        );
    }

    /// Mutation proof for the source-truth failure found in the prior audit: no control checkpoint
    /// may be injected into Chroma, FLUX.1 Schnell, or FLUX.2 Klein, while the two real Dev routes
    /// must resolve to their dedicated control providers and remain covered.
    #[cfg(target_os = "macos")]
    #[test]
    fn resident_only_audit_rejects_impossible_control_routes_and_keeps_real_ones() {
        let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
            include_str!("../../../config/manifests/builtin.models.jsonc"),
        ))
        .expect("builtin.models.jsonc parses");
        let models = manifest["models"].as_array().expect("manifest models");
        for fake in [
            "chroma1_base",
            "chroma1_flash",
            "chroma1_hd",
            "flux_schnell",
            "flux2_klein_9b",
            "flux2_klein_9b_kv",
            "flux2_klein_9b_true_v2",
        ] {
            let injected = SourceBoundAuditSurface {
                manifest_id: fake.to_owned(),
                surface: ResidentOnlyAuditSurface::StrictControl,
            };
            let error = source_bound_audit_inventory_from_surfaces(models, &[injected])
                .expect_err("an impossible strict-control route must fail closed");
            assert!(
                error.contains("has no production FLUX strict-control route"),
                "{fake} must fail at the production control router, got: {error}"
            );
        }

        let surfaces = source_bound_audit_surfaces(models).expect("production source surfaces");
        let source_inventory = source_bound_audit_inventory_from_surfaces(models, &surfaces)
            .expect("production source inventory");
        let (_, cells) = source_bound_resident_only_cells_from_inventory(models, &source_inventory)
            .expect("production resident-only cells");
        assert!(
            !cells.iter().any(|cell| {
                cell.manifest_id == "lens"
                    && cell.tier == "q4"
                    && cell.surface == ResidentOnlyAuditSurface::Base
            }),
            "base Lens q4 must remain on its production measured deferred-materialization contract"
        );
        assert!(
            !cells
                .iter()
                .any(|cell| cell.provider_id == "flux2_dev_edit"),
            "FLUX.2 Dev edit must remain on its provider-owned request-safety path"
        );
        assert!(cells.iter().any(|cell| {
            cell.manifest_id == "flux_dev"
                && cell.provider_id == "flux1_dev_control"
                && cell.surface == ResidentOnlyAuditSurface::StrictControl
                && cell.control_bytes == FLUX1_CONTROL_BYTES
        }));
        assert!(cells.iter().any(|cell| {
            cell.manifest_id == "flux2_dev"
                && cell.provider_id == "flux2_dev_control"
                && cell.surface == ResidentOnlyAuditSurface::StrictControl
                && cell.control_bytes == FLUX2_CONTROL_BYTES
        }));
        assert!(cells.iter().all(|cell| {
            cell.control_bytes == 0
                || matches!(cell.provider_id, "flux1_dev_control" | "flux2_dev_control")
        }));
    }

    /// Completeness mutation for the loophole found after the production-router rewrite. A
    /// zero-cell route is still part of the candidate inventory: replacing FLUX.1 Schnell with an
    /// already-declared Chroma entry must fail before deduplication, and simply deleting Schnell
    /// must fail exact source-inventory equality even though the 35 Resident-only cells and two
    /// flips are unchanged.
    #[cfg(target_os = "macos")]
    #[test]
    fn resident_only_audit_inventory_rejects_duplicate_and_zero_cell_drop_mutations() {
        let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
            include_str!("../../../config/manifests/builtin.models.jsonc"),
        ))
        .expect("builtin.models.jsonc parses");
        let models = manifest["models"].as_array().expect("manifest models");
        let classifications =
            source_bound_manifest_classifications(models).expect("source classifications");
        let manifest_ids = source_bound_generic_manifest_ids(&classifications).collect::<Vec<_>>();
        let expected_surfaces =
            source_bound_audit_surfaces_from_manifest_ids(manifest_ids.iter().copied())
                .expect("source surfaces");
        let expected_inventory =
            source_bound_audit_inventory_from_surfaces(models, &expected_surfaces)
                .expect("source inventory");
        let (_, expected_resident) =
            source_bound_resident_only_cells_from_inventory(models, &expected_inventory)
                .expect("source Resident-only inventory");
        assert!(
            expected_inventory
                .iter()
                .any(|cell| cell.manifest_id == "flux_schnell"),
            "the mutation requires FLUX.1 Schnell in the full source inventory"
        );
        assert!(
            expected_resident
                .iter()
                .all(|cell| cell.manifest_id != "flux_schnell"),
            "the mutation requires FLUX.1 Schnell to be a representative zero-cell route"
        );

        let mut duplicate_replacement = manifest_ids.clone();
        let schnell = duplicate_replacement
            .iter()
            .position(|manifest_id| *manifest_id == "flux_schnell")
            .expect("FLUX.1 Schnell source declaration");
        duplicate_replacement[schnell] = "chroma1_base";
        let duplicate_error =
            source_bound_audit_surfaces_from_manifest_ids(duplicate_replacement.iter().copied())
                .expect_err("a duplicate replacement must fail before set conversion");
        assert!(
            duplicate_error
                .contains("duplicate source-bound audit manifest declaration chroma1_base"),
            "duplicate replacement failed for the wrong reason: {duplicate_error}"
        );

        let dropped_ids = manifest_ids
            .iter()
            .copied()
            .filter(|manifest_id| *manifest_id != "flux_schnell")
            .collect::<Vec<_>>();
        let dropped_surfaces =
            source_bound_audit_surfaces_from_manifest_ids(dropped_ids.iter().copied())
                .expect("dropped source surfaces still resolve");
        let dropped_inventory =
            source_bound_audit_inventory_from_surfaces(models, &dropped_surfaces)
                .expect("dropped source inventory still resolves");
        let (_, dropped_resident) =
            source_bound_resident_only_cells_from_inventory(models, &dropped_inventory)
                .expect("dropped Resident-only inventory still resolves");
        assert_eq!(
            dropped_resident, expected_resident,
            "precondition: dropping a zero-cell route preserves the old Resident-only summary"
        );
        let drop_error =
            require_exact_source_bound_inventory(&expected_inventory, &dropped_inventory)
                .expect_err("exact source equality must reject a dropped zero-cell route");
        assert!(
            drop_error.contains("flux_schnell"),
            "zero-cell drop must name its missing source route: {drop_error}"
        );
    }

    /// Explicit-exclusion mutation proof. PuLID is inside the eight-family story scope but takes
    /// the identity-conditioned route outside MODEL_TABLE; Mage is outside that family scope but
    /// must remain named because its split component tree cannot be represented by the generic
    /// single-root sparse fixture. Neither exclusion may disappear or drift families silently.
    #[cfg(target_os = "macos")]
    #[test]
    fn resident_only_audit_exclusions_reject_pulid_and_mage_scope_mutations() {
        let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
            include_str!("../../../config/manifests/builtin.models.jsonc"),
        ))
        .expect("builtin.models.jsonc parses");
        let models = manifest["models"].as_array().expect("manifest models");
        let classifications =
            source_bound_manifest_classifications(models).expect("source classifications");
        let expected =
            source_bound_excluded_inventory_from_classifications(models, &classifications)
                .expect("source excluded inventory");

        let missing_pulid = classifications
            .iter()
            .filter(|item| item.manifest_id != "pulid_flux_dev")
            .cloned()
            .collect::<Vec<_>>();
        let actual = source_bound_excluded_inventory_from_classifications(models, &missing_pulid)
            .expect("the mutated exclusion list still resolves");
        let error = require_exact_source_bound_excluded_inventory(&expected, &actual)
            .expect_err("removing PuLID classification must fail exact equality");
        assert!(error.contains("pulid_flux_dev"), "wrong failure: {error}");

        let reclassified_pulid = classifications
            .iter()
            .cloned()
            .map(|mut item| {
                if item.manifest_id == "pulid_flux_dev" {
                    item.disposition = SourceBoundManifestDisposition::GenericSelector;
                }
                item
            })
            .collect::<Vec<_>>();
        let actual =
            source_bound_excluded_inventory_from_classifications(models, &reclassified_pulid)
                .expect("the mutated disposition list still resolves");
        let error = require_exact_source_bound_excluded_inventory(&expected, &actual)
            .expect_err("changing PuLID classification must fail exact equality");
        assert!(error.contains("pulid_flux_dev"), "wrong failure: {error}");

        let missing_mage = classifications
            .iter()
            .filter(|item| {
                item.disposition != SourceBoundManifestDisposition::MageSplitComponentsExcluded
            })
            .cloned()
            .collect::<Vec<_>>();
        let actual = source_bound_excluded_inventory_from_classifications(models, &missing_mage)
            .expect("the mutated exclusion list still resolves");
        let error = require_exact_source_bound_excluded_inventory(&expected, &actual)
            .expect_err("removing Mage scope must fail exact equality");
        assert!(error.contains("mage_flow"), "wrong failure: {error}");

        let mut pulid_family_drift = models.clone();
        pulid_family_drift
            .iter_mut()
            .find(|model| model["id"] == "pulid_flux_dev")
            .expect("PuLID manifest entry")["family"] = Value::String("not-flux".to_owned());
        let error = source_bound_manifest_classifications(&pulid_family_drift)
            .expect_err("PuLID family drift must fail classification");
        assert!(error.contains("PuLID bespoke classification drifted"));

        let mut mage_family_drift = models.clone();
        mage_family_drift
            .iter_mut()
            .find(|model| model["id"] == "mage_flow")
            .expect("Mage manifest entry")["family"] = Value::String("flux".to_owned());
        let error = source_bound_manifest_classifications(&mage_family_drift)
            .expect_err("Mage family drift must fail classification");
        assert!(error.contains("Mage split-component classification drifted"));
    }

    /// sc-18251: the PREMISE of applying the composition leg to Resident, pinned against gen-core
    /// rather than restated.
    ///
    /// The shape leg of the `usable` filter exempts `MemoryStrategy::Resident` because
    /// `optimized_eligibility` short-circuits `Ok(())` for a non-optimized selection before it
    /// compares load shapes. The composition leg deliberately does NOT mirror that exemption,
    /// because both of the gate's composition checks — canonical form (`Invalid`) and contract
    /// agreement (`CompositionMismatch`) — run BEFORE the short-circuit, so gen-core rejects a
    /// resident cell on either. If a pin bump ever moves those checks behind the Resident
    /// short-circuit, this test reds and the filter's legs must be re-derived.
    #[test]
    fn gen_core_rejects_a_resident_cell_whose_composition_disagrees() {
        let generator = fixture_generator();
        let contract = generator.contract.as_ref().expect("fixture contract");

        let (selection, mut resident) = resident_evidence(
            contract,
            fixture_plan().tier,
            "text_to_image",
            None,
            request_geometry(&fixture_inputs(1024, 1024)),
            gib_to_bytes(4.0),
            Some("fixture-formula-v2"),
        );
        resident.conformance = gen_core::MemoryConformanceState::Verified;
        resident.dimensions = gen_core::MemoryEvidenceDimensions::VERIFIED;
        assert!(
            !selection.strategy.is_optimized(),
            "the premise is about the non-optimized rung"
        );
        assert_eq!(
            resident.optimized_eligibility(contract),
            Ok(()),
            "precondition: the resident cell is eligible before the composition is disturbed"
        );

        // Canonical but disagreeing: the gate's contract-agreement check must reject even a
        // RESIDENT cell, which is why the filter's composition leg carries no Resident exemption.
        resident.key.engaged_composition =
            vec![MemoryStrategy::Resident, MemoryStrategy::BoundedDecode];
        assert_eq!(
            resident.optimized_eligibility(contract),
            Err(gen_core::MemoryEvidenceVerdict::CompositionMismatch),
            "gen-core rejects a RESIDENT cell whose composition disagrees; the filter's \
             no-exemption composition leg is built on this"
        );

        // The canonical-form prefix also runs before the Resident short-circuit.
        resident.key.engaged_composition = Vec::new();
        assert_eq!(
            resident.optimized_eligibility(contract),
            Err(gen_core::MemoryEvidenceVerdict::Invalid),
            "gen-core rejects a RESIDENT cell whose composition is non-canonical"
        );
    }

    /// sc-18251 (review addendum): why the shape leg's `identity.load_shape` conjunct has no
    /// isolating fixture of its own — the contract↔identity split it would need is impossible by
    /// construction, and THAT premise is what this test pins against the pinned gen-core.
    ///
    /// Every fixture moves `contract.load_shape` and `identity.load_shape` together, so deleting
    /// either single conjunct of
    /// `key.load_shape == contract.load_shape && key.load_shape == identity.load_shape` cannot be
    /// distinguished by those fixtures alone. An isolating arm was written and it demonstrated the
    /// impossibility empirically: gen-core's `conformance_errors` requires
    /// `calibration.load_shape == contract.load_shape`, and `select_strategy` refuses EVERY
    /// candidate on a non-conformant contract with `Unverified(Invalid)` before grading a single
    /// one, so a split contract refuses the request no matter which conjunct the filter carries.
    /// The pair is effectively one comparison, kept in both spellings only to mirror
    /// `optimized_eligibility` literally.
    ///
    /// If a pin bump ever legalizes the split, this test reds — and the identity conjunct then
    /// needs its own fixture arm, because it will have become independently load-bearing.
    #[test]
    fn gen_core_forbids_a_contract_identity_load_shape_split() {
        use gen_core::MemoryCalibrationIdentity;

        let generator = fixture_generator();
        let contract = generator.contract.as_ref().expect("fixture contract");
        assert!(
            contract.conformance_errors().is_empty(),
            "precondition: the agreeing fixture contract is conformant"
        );
        let fingerprint = contract
            .calibration
            .as_ref()
            .expect("fixture calibration")
            .fingerprint
            .clone();

        // Identity moves, contract stays.
        let mut identity_split = contract.clone();
        identity_split.calibration = Some(MemoryCalibrationIdentity::new(
            fingerprint.clone(),
            gen_core::LoadShape::DeferredMaterialization,
        ));
        assert!(
            identity_split
                .conformance_errors()
                .iter()
                .any(|error| error.contains("load shape")),
            "gen-core must reject a contract whose calibration identity shape disagrees with its \
             live load shape; the filter's fused shape conjuncts are built on this"
        );

        // Contract moves, identity stays.
        let mut contract_split = contract.clone();
        contract_split.load_shape = gen_core::LoadShape::DeferredMaterialization;
        assert!(
            contract_split
                .conformance_errors()
                .iter()
                .any(|error| error.contains("load shape")),
            "the split must be rejected in both directions"
        );

        // And a non-conformant contract never grades candidates at all — the refusal outruns the
        // filter, which is why no fixture can observe the identity conjunct alone.
        let selection = crate::memory_strategy::select_strategy(
            crate::memory_strategy::RequestScope {
                resolved_route: "fixture_provider",
                backend: "mlx",
                tier: fixture_plan().tier,
                mode: "text_to_image",
                overlay: None,
                geometry: request_geometry(&fixture_inputs(1024, 1024)),
                expected_closure_digest: FIXTURE_CLOSURE_DIGEST,
            },
            &identity_split,
            Some(crate::memory_strategy::Budget {
                available_gb: 64.0,
                reclaimable_gb: 0.0,
                total_gb: 64.0,
                reserved_headroom_gb: 0.0,
            }),
            &[],
        );
        assert_eq!(
            selection,
            crate::memory_strategy::Selection::Unverified {
                reason: gen_core::MemoryEvidenceVerdict::Invalid
            },
            "a split contract refuses everything before any candidate is graded"
        );
    }

    #[test]
    fn a_moved_provider_closure_admits_the_stale_ladder_behind_the_widened_margin() {
        // sc-18096 (scope addendum from sc-18095's review): the `StaleIdentity` pre-demotion in
        // `evidence_admission_route` is retired. A binding measured under a moved closure still
        // reaches `AdmissionPath::Evidence`; its candidate carries the digest it was MEASURED
        // under, and `select_strategy` grades it behind `MLX_STALE_MEASURED_MARGIN`. This is the
        // PRODUCTION-routing proof that the sc-18095 selector arm is reachable on the MLX lane —
        // not merely a selector unit test.
        //
        // Fixture arithmetic: the record's envelope peak is exactly 5 GiB with a 3 GiB captured
        // foreign reserve, so the widened requirement is 5 * 1.05 = 5.25 GiB against
        // `total - reserve` of effective budget.
        let bundle = fixture_bundle();
        let generator = fixture_generator();
        let plan = fixture_plan();
        let inputs = fixture_inputs(1024, 1024);
        let moved_digest = "a".repeat(64);
        let moved = |_backend: &str, _provider: &str| Some(moved_digest.clone());

        // The seam: the stale binding is still an identity match (same artifact bytes), so it
        // enters the Evidence path with its measured digest attached for the selector to grade.
        let stale = evidence_admission_route(
            &bundle,
            &plan,
            &inputs,
            "text_to_image",
            fixture_budget(9.0),
            &moved_digest,
        )
        .expect("a moved closure admits behind the widened margin, it does not error");
        assert_eq!(
            stale.path,
            AdmissionPath::Evidence,
            "a stale-only cell must reach the selector instead of being pre-demoted: {:?}",
            stale.fallback_reason
        );
        assert!(!stale.evidence.is_empty());
        assert!(
            stale
                .evidence
                .iter()
                .all(
                    |candidate| candidate.closure_digest == FIXTURE_CLOSURE_DIGEST
                        && candidate.closure_digest != moved_digest
                ),
            "each candidate must carry the digest it was MEASURED under, not the live one"
        );
        // Refusal advice stays current-only: a stale cell may serve widened numbers, but it is not
        // offered as a named "current verified alternative".
        assert!(stale.lower_alternative.is_none());

        // End to end at 9 GiB: effective budget is 9 - 3 (captured foreign reserve) = 6 GiB, the
        // widened 5.25 GiB fits, and the request keeps the exact verified rung INCLUDING its
        // request-scoped process ceiling.
        let admitted = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(9.0),
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
            Some(&moved),
        )
        .expect("a stale ladder that fits with the widened margin must admit");
        assert_eq!(
            admitted.context.selection.strategy,
            MemoryStrategy::BoundedDecode,
            "the stale measured rung itself must be selected"
        );
        assert!(
            admitted.process_limit_bytes.is_some(),
            "a stale exact cell still derives the request-scoped ceiling"
        );

        // The margin is APPLIED, not just plumbed (production-path mutation check): at 8.1 GiB the
        // effective budget is 5.1 GiB — the RAW 5 GiB peak fits, the widened 5.25 GiB does not. A
        // gate that stopped widening stale admission would admit here and flip this arm. The
        // refusal quotes the graded host requirement: widened 5.25 GiB + the 3 GiB captured
        // foreign reserve = 8.25 GiB.
        let error = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(8.1),
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
            Some(&moved),
        )
        .expect_err("the raw peak fits 5.1 GiB but the WIDENED stale peak must not")
        .to_string();
        assert!(
            error.contains("needs at least 8.25 GiB"),
            "the refusal must quote the widened stale host requirement: {error}"
        );

        // The control: the SAME request with the closure unmoved is graded at the raw peak, so the
        // 8.1 GiB budget that refused above admits — proving the refusal was the stale widening.
        let current = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(8.1),
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
            Some(&fixture_closure_lookup),
        )
        .expect("the unmoved closure must still admit the verified rung at the raw peak");
        assert!(
            current.process_limit_bytes.is_some(),
            "the unmoved closure must still reach the exact verified cell"
        );

        // A stale record serves its OWN cell (the arms above) but may not SEED an extrapolation:
        // at 768² — off the measured 1024² geometry — the moved-closure request gets no fitted
        // basis and refuses on floors alone (staged/decode/attention floors widen to 9.9 GiB
        // against 8 GiB), while the unmoved closure admits the fitted bounded-decode estimate
        // (clamped scale 1.0, envelope 5 GiB widened to 5.5) at the same budget. A gate that let
        // stale records seed extrapolations would admit BOTH and flip the first arm.
        let off_geometry = fixture_inputs(768, 768);
        let error = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &off_geometry,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(8.0),
            gib_to_bytes(12.0),
            0,
            &[],
            Some(&bundle),
            Some(&moved),
        )
        .expect_err("a stale-closure record must not seed a fitted extrapolation")
        .to_string();
        assert!(
            error.contains("needs") && error.contains("safely available"),
            "the stale-basis refusal is the floors-only Reject: {error}"
        );
        let fitted = evaluate_request_with_budget_using_bundle(
            &generator,
            &plan,
            &off_geometry,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(8.0),
            gib_to_bytes(12.0),
            0,
            &[],
            Some(&bundle),
            Some(&fixture_closure_lookup),
        )
        .expect("the CURRENT-closure record is a legitimate fitted basis at the same budget");
        assert_eq!(
            fitted.context.selection.strategy,
            MemoryStrategy::BoundedDecode,
            "the fitted estimate from the current-closure cell must admit: {:?}",
            fitted.context.selection
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
            FIXTURE_CLOSURE_DIGEST,
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
                Some(&fixture_closure_lookup),
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
                Some(&fixture_closure_lookup),
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

    /// SC-15808's reachability claim is structural: the Krea route enters the same admission path as
    /// every other MLX provider and has no provider-local three-rung ceiling. The synthetic rows here
    /// deliberately make no calibration claim; Krea's family/model stories still own implementation
    /// and real-weight evidence for rungs 3 and 4. This test proves those future verified rows are not
    /// hidden from `select_strategy` by route-specific admission code.
    #[test]
    fn krea_route_admission_reaches_later_rungs_when_verified_rows_exist() {
        use gen_core::{LoadShape, MemoryParameterRanges, MemoryStrategySupport};
        use sceneworks_core::memory_calibration::RequiredNullable;

        const KREA_CONTROL_ROUTE: &str = "krea_2_turbo_control";
        let (mut bundle, mut plan) = fixture_ladder();
        let mut generator = fixture_generator();
        generator.descriptor.id = KREA_CONTROL_ROUTE;
        let contract = generator.contract.as_mut().expect("fixture contract");
        contract.provider_id = KREA_CONTROL_ROUTE.to_owned();
        contract.load_shape = LoadShape::DeferredMaterialization;
        // ABI 2: the calibration identity, every receipt, and the binding must carry the same
        // materialization shape as the contract or eligibility fails closed.
        contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
            "fixture-formula-v2",
            LoadShape::DeferredMaterialization,
        ));
        for record in &mut bundle.records {
            record.load_shape =
                sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
        }
        contract.lifecycle.transformer_window_materialization = true;
        let transformer = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("transformer-residency capability");
        transformer.support = MemoryStrategySupport::Implemented;
        transformer.parameters = MemoryParameterRanges {
            transformer_window_sizes: vec![1],
            ..Default::default()
        };

        let transformer_parameters = JsonObject::from_iter([
            ("decodeTileEdge".to_owned(), serde_json::json!(512)),
            ("decodeOverlap".to_owned(), serde_json::json!(128)),
            ("attentionChunkSize".to_owned(), serde_json::json!(256)),
            ("transformerWindowSize".to_owned(), serde_json::json!(1)),
        ]);
        let mut transformer_record = bundle
            .records
            .last()
            .expect("attention fixture row")
            .clone();
        transformer_record.id = "imc-15808000000000000000".to_owned();
        transformer_record.logical_case_id = "implan-15808000000000000000".to_owned();
        transformer_record.strategy.rung = StrategyRung::BoundedTransformerResidency;
        transformer_record.strategy.engaged_rungs = contract
            .engaged_composition(MemoryStrategy::BoundedTransformerResidency)
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
        transformer_record.strategy.parameters = transformer_parameters.clone();
        transformer_record.sweep.cases[0].parameters = transformer_parameters.clone();
        if let RequiredNullable::Value(predicted) = &mut transformer_record.predicted_peak_bytes {
            let predicted = predicted
                .full_mut()
                .expect("fixture has full phase telemetry");
            predicted.conditioning = predicted.conditioning.min(gib_to_bytes(3.0));
            predicted.denoise = gib_to_bytes(3.0);
            predicted.decode = predicted.decode.min(gib_to_bytes(3.0));
            predicted.overall = gib_to_bytes(3.0);
        }
        if let RequiredNullable::Value(observed) = &mut transformer_record.observed_memory {
            let observed = observed
                .full_mut()
                .expect("fixture has full phase telemetry");
            observed.overall.active_bytes = gib_to_bytes(3.0);
            observed.overall.allocator_bytes = gib_to_bytes(3.0);
            observed.overall.device_bytes = gib_to_bytes(3.0);
            observed.overall.wired_bytes = gib_to_bytes(3.0);
            observed.overall.reclaimable_bytes = 0;
        }
        bundle.records.push(transformer_record);

        plan.engine_id = KREA_CONTROL_ROUTE;
        plan.model_id = "krea_2_turbo".to_owned();
        let MlxCalibrationConfig::Valid(calibration) = &mut plan.calibration else {
            panic!("fixture calibration");
        };
        calibration.bindings.push(fixture_binding_for(
            "q4",
            "packed-q4",
            StrategyRung::BoundedTransformerResidency,
            transformer_parameters,
        ));
        for binding in &mut calibration.bindings {
            binding.provider = KREA_CONTROL_ROUTE.to_owned();
            binding.query.load_shape =
                sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization;
        }
        for record in &mut bundle.records {
            record.target.provider = KREA_CONTROL_ROUTE.to_owned();
            record.target.model_id = "krea_2_turbo".to_owned();
        }

        // This test dresses FIXTURE bindings in a real lane's id, so the injected resolver has to
        // answer for that id too — the digests here are the fixture's, not the shipped Krea lane's.
        let krea_closure_lookup = |backend: &str, provider: &str| -> Option<String> {
            if backend == "mlx" && provider == KREA_CONTROL_ROUTE {
                return Some(fixture_closure_digest());
            }
            fixture_closure_lookup(backend, provider)
        };
        for (total_gib, expected) in [
            (7.0, MemoryStrategy::BoundedAttention),
            (6.0, MemoryStrategy::BoundedTransformerResidency),
        ] {
            let evaluation = evaluate_request_with_budget_using_bundle(
                &generator,
                &plan,
                &fixture_inputs(1024, 1024),
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                fixture_budget(total_gib),
                gib_to_bytes(4.0),
                0,
                &[],
                Some(&bundle),
                Some(&krea_closure_lookup),
            )
            .unwrap_or_else(|error| panic!("{total_gib} GiB Krea route failed: {error}"));
            assert_eq!(evaluation.context.selection.strategy, expected);
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
            Some(&fixture_closure_lookup),
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
    fn capture_host_reserve_scales_to_48_gib_without_erasing_the_stale_margin() {
        use sceneworks_core::memory_calibration::MlxAdmissionEnvelope;

        let capture_host = gib_to_bytes(128.0);
        let envelope = MlxAdmissionEnvelope {
            peak_bytes: gib_to_bytes(16.41),
            observed_non_reclaimable_wired_bytes: gib_to_bytes(15.0),
            capture_host_bytes: capture_host,
            foreign_reserve_bytes: gib_to_bytes(46.93),
        };
        let live_host = gib_to_bytes(48.0);
        assert!(
            envelope
                .peak_bytes
                .saturating_add(envelope.foreign_reserve_bytes)
                > live_host,
            "the old absolute 128 GiB-host reserve reproduces the false 48 GiB refusal"
        );
        assert!(
            envelope.required_host_bytes() <= live_host,
            "the true static boundary must agree that this candidate can fit below 48 GiB"
        );
        let live_reserve = envelope.foreign_reserve_for_host_bytes(live_host);
        let stale_peak = crate::memory_strategy::stale_widened_peak_bytes(
            gen_core::MemoryBackend::Mlx,
            envelope.peak_bytes,
        );
        assert!(
            stale_peak.saturating_add(live_reserve) <= live_host,
            "the stale widening remains charged after host-capacity normalization"
        );
        let process_limit = live_host.saturating_sub(live_reserve);
        assert!(
            stale_peak <= process_limit,
            "the request remains below the absolute MLX process limit used for OOM containment"
        );
        assert_eq!(
            live_reserve,
            18_896_513_925,
            "46.93 GiB reserved on 128 GiB scales, conservatively rounded up, to 17.59875 GiB on 48 GiB"
        );
    }

    #[test]
    fn packaged_bundle_without_an_exact_record_is_a_normal_no_record_reason() {
        // The packaged evidence is current for schema v4 / ABI 3. An uncovered fixture therefore
        // degrades to the legacy path with the precise `NoRecord` reason, not bundle drift.
        let route = packaged_admission_route(
            &fixture_plan(),
            &fixture_inputs(1024, 1024),
            "text_to_image",
            fixture_budget(8.0),
            FIXTURE_CLOSURE_DIGEST,
        )
        .expect("a current promoted bundle with no exact record degrades, never errors");
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
                component_precision_floors: &[],
            },
            asset_bytes: 35_666_644_396,
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
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

    /// sc-16209: the generic fallback must consume Krea's provider-owned, measured 1024²
    /// activation anchor instead of charging the unrelated 14 GiB worst-case family anchor.
    #[cfg(target_os = "macos")]
    #[test]
    fn krea_measured_anchor_admits_the_64_gib_motivating_cell() {
        let weights = tempfile::tempdir().expect("Krea weights fixture");
        std::fs::File::create(weights.path().join("fixture.safetensors"))
            .expect("Krea weights fixture file")
            .set_len(1024)
            .expect("Krea weights fixture size");
        let spec = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
        let mut plan = MlxRequestPlan::for_spec_and_manifest(
            "krea_2_turbo",
            "krea_2_turbo",
            &spec,
            None,
            None,
        );
        // Preserve the production-selected allowance while pinning the exact asset term from the
        // measured motivating cell. A tiny sparse fixture is enough to exercise descriptor lookup.
        plan.asset_bytes = gib_to_bytes(33.22);

        let peak_gb =
            plan.generic_total_peak_bytes(request_geometry(&request_inputs(1152, 2048, 1))) as f64
                / BYTES_PER_GIB;
        let expected_gb = 33.22 + 2.0 + 7.67 * 2.25;
        assert!(
            (peak_gb - expected_gb).abs() < 1e-5,
            "Krea's measured 7.67 GiB anchor should model {expected_gb:.4} GiB, got {peak_gb:.4}"
        );
        assert!(
            peak_gb <= 62.0,
            "a 64 GiB Mac's 62 GiB request budget must admit the measured Krea cell"
        );

        // Mutation guard: restoring the generic 14 GiB anchor through the production estimator
        // reproduces the original rejection.
        plan.activation_headroom_bytes = gib_to_bytes(16.0);
        plan.fixed_reserve_bytes = gib_to_bytes(2.0);
        let generic_peak_gb =
            plan.generic_total_peak_bytes(request_geometry(&request_inputs(1152, 2048, 1))) as f64
                / BYTES_PER_GIB;
        assert!(
            generic_peak_gb > 62.0,
            "the generic fallback must remain a rejecting estimate, got {generic_peak_gb:.4} GiB"
        );
    }

    /// An unmeasured route must retain the conservative generic allowance. Krea Turbo's measured
    /// family anchor cannot silently authorize the distinct Krea Raw graph.
    #[test]
    fn unmeasured_krea_route_retains_the_generic_activation_fallback() {
        let weights = tempfile::tempdir().expect("Krea weights fixture");
        std::fs::File::create(weights.path().join("fixture.safetensors"))
            .expect("Krea weights fixture file")
            .set_len(1024)
            .expect("Krea weights fixture size");
        let spec = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
        let plan =
            MlxRequestPlan::for_spec_and_manifest("krea_2_raw", "krea_2_raw", &spec, None, None);

        assert_eq!(plan.fixed_reserve_bytes, gib_to_bytes(2.0));
        assert_eq!(plan.activation_headroom_bytes, gib_to_bytes(16.0));
        assert_eq!(
            plan.generic_total_peak_bytes(request_geometry(&request_inputs(1152, 2048, 1))),
            plan.asset_bytes
                .saturating_add(gib_to_bytes(2.0))
                .saturating_add(gib_to_bytes(14.0 * 2.25))
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
                component_precision_floors: &[],
            },
            asset_bytes: gib_to_bytes(asset_gb),
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
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

    /// sc-16209: a family whose allowance was measured as a BARE activation transient keeps that
    /// whole measurement in the area term, with the request budget's remaining OS reserve added as
    /// a separate fixed term.
    ///
    /// This is the lens dense path ([`HeadroomAllowance::LENS_DENSE`], sc-11924). Subtracting the
    /// legacy 2 GiB request-budget reserve from its 29.88 GiB BARE activation measurement would
    /// under-predict above 1024² by `2·MP`. The safe decomposition is fixed 2 + 29.88·MP.
    #[test]
    fn a_bare_transient_allowance_keeps_its_whole_area_term() {
        let asset_gb = 28.43_f64;
        let plan = |headroom: HeadroomAllowance| {
            let fixed_reserve_bytes = gib_to_bytes(
                (OS_APP_RESERVE_GB - crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB).max(0.0),
            );
            let activation_anchor_bytes =
                gib_to_bytes((headroom.total_gb - headroom.os_reserve_gb).max(0.0));
            MlxRequestPlan {
                engine_id: "lens_turbo",
                model_id: "lens_turbo".to_owned(),
                tier: MemoryNumericTier {
                    precision: gen_core::Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
                asset_bytes: gib_to_bytes(asset_gb),
                folded_control_bytes: 0,
                folded_adapter_bytes: 0,
                activation_headroom_bytes: activation_anchor_bytes
                    .saturating_add(fixed_reserve_bytes),
                fixed_reserve_bytes,
                calibration: MlxCalibrationConfig::Absent,
            }
        };
        let dense = plan(HeadroomAllowance::LENS_DENSE);
        let peak_gb = |plan: &MlxRequestPlan, width, height| {
            plan.generic_total_peak_bytes(request_geometry(&request_inputs(width, height, 1)))
                as f64
                / BYTES_PER_GIB
        };
        for (width, height, megapixels) in [
            (1024, 1024, 1.0),
            (1024, 1536, 1.5),
            (1152, 2048, 2.25),
            (2048, 2048, 4.0),
        ] {
            let expected = asset_gb + 2.0 + LENS_DENSE_HEADROOM_GB * megapixels;
            assert!(
                (peak_gb(&dense, width, height) - expected).abs() < 1e-6,
                "{width}x{height}: a bare-transient allowance must scale WHOLLY with area — \
                 expected fixed 2 + {LENS_DENSE_HEADROOM_GB}*{megapixels} = {expected:.4}, got {:.4}",
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
            folded_control_bytes: 0,
            folded_adapter_bytes: 0,
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
    fn production_control_spec_replaces_raw_checkpoint_with_one_typed_residency() {
        use gen_core::{
            MemoryComponentKind, MemoryFormulaKind, MemoryFormulaVariable, MemoryResidentComponent,
        };
        use std::fs::File;

        const BASE_SOURCE_BYTES: u64 = 4_000;
        const CONTROL_SOURCE_BYTES: u64 = 2_000;
        const CONTROL_RESIDENT_BYTES: u64 = 1_000;
        let root_guard = tempfile::Builder::new()
            .prefix("sceneworks-mlx-fit-sc-16065-")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        let base = root.join("base.safetensors");
        let control = root.join("control.safetensors");
        File::create(&base)
            .expect("base fixture")
            .set_len(BASE_SOURCE_BYTES)
            .expect("base fixture size");
        File::create(&control)
            .expect("control fixture")
            .set_len(CONTROL_SOURCE_BYTES)
            .expect("control fixture size");
        let spec =
            LoadSpec::new(WeightsSource::File(base)).with_control(WeightsSource::File(control));
        let mut plan = MlxRequestPlan::for_spec_and_manifest(
            "fixture_provider",
            "fixture_model",
            &spec,
            None,
            None,
        );
        // Isolate asset accounting from the separately-tested activation allowance. Crucially, the
        // plan itself still comes from the production LoadSpec path that folds control into total.
        plan.activation_headroom_bytes = 0;
        plan.fixed_reserve_bytes = 0;
        assert_eq!(plan.asset_bytes, BASE_SOURCE_BYTES + CONTROL_SOURCE_BYTES);
        assert_eq!(plan.folded_control_bytes, CONTROL_SOURCE_BYTES);

        let mut generator = fixture_generator();
        {
            let contract = generator.contract.as_mut().expect("fixture contract");
            let phases = contract.lifecycle.phases.clone();
            contract.asset_facts.base_bytes = BASE_SOURCE_BYTES;
            contract.asset_facts.transformer_bytes = BASE_SOURCE_BYTES;
            contract.asset_facts.overlay_bytes = CONTROL_RESIDENT_BYTES;
            contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables: vec![
                    MemoryFormulaVariable::AssetBytes,
                    MemoryFormulaVariable::OverlayBytes,
                ],
                resident_components: vec![MemoryResidentComponent {
                    id: "fixture.control".to_owned(),
                    kind: MemoryComponentKind::ControlBranch,
                    resident_bytes: CONTROL_RESIDENT_BYTES,
                    bounded_by: None,
                    residency: gen_core::MemoryComponentResidency::WholeRender,
                }],
            };
        }
        let inputs = fixture_inputs(1024, 1024);
        let legacy_total_peak = plan.generic_total_peak_bytes(request_geometry(&inputs));
        assert_eq!(
            legacy_total_peak,
            BASE_SOURCE_BYTES + CONTROL_SOURCE_BYTES,
            "the production legacy estimate includes the raw control checkpoint"
        );

        let cold = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(1.0),
            legacy_total_peak,
            0,
            &[],
        )
        .expect("the normalized base plus one typed control branch fits");

        assert_eq!(
            cold.context.predicted_peak_bytes,
            BASE_SOURCE_BYTES + CONTROL_RESIDENT_BYTES,
            "the raw control source must be replaced, not double-counted"
        );
        let contract = generator.contract.as_ref().expect("fixture contract");
        // Ask the CONTRACT for the typed decomposition, not the run context (sc-17037).
        //
        // gen-core `8ffa211a` changed `MemoryRunContext::predicted_peak_breakdown` to ignore
        // its contract argument (the parameter is literally `_contract` now) and report the
        // whole scalar as unattributed. That is deliberate and correct: the same bump renamed
        // `MemoryBudget::fits(predicted_peak_bytes)` to `fits(incremental_live_demand_bytes)`,
        // so the context scalar is now INCREMENTAL demand after the caller has already credited
        // request-owned resident bytes — subtracting the provider's formula components from it
        // again would credit the control branch twice.
        //
        // `MemoryProviderContract::decompose_predicted_peak` is unchanged and is where typed
        // attribution still lives, which is what these four assertions are actually about. This
        // fixture zeroes activation headroom and fixed reserve, so the context scalar equals the
        // absolute peak here and the decomposition is exact.
        let breakdown = contract.decompose_predicted_peak(cold.context.predicted_peak_bytes);
        assert_eq!(
            breakdown.predicted_peak_bytes(),
            BASE_SOURCE_BYTES + CONTROL_RESIDENT_BYTES
        );
        assert_eq!(breakdown.unattributed_bytes, BASE_SOURCE_BYTES);
        assert_eq!(breakdown.components.len(), 1);
        assert_eq!(
            breakdown.components[0].kind,
            MemoryComponentKind::ControlBranch
        );
        // Pin the context-level behaviour too. Without this the next gen-core bump could quietly
        // reinstate contract-aware decomposition here — re-introducing exactly the double-credit
        // the upstream change removed — and nothing in this repo would notice: this test is the
        // ONLY caller of `predicted_peak_breakdown` in SceneWorks.
        let context_breakdown = cold.context.predicted_peak_breakdown(contract);
        assert_eq!(
            context_breakdown.unattributed_bytes,
            BASE_SOURCE_BYTES + CONTROL_RESIDENT_BYTES,
            "the run context reports incremental demand whole; it must not re-decompose it"
        );
        assert!(context_breakdown.components.is_empty());

        let warm = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Warm,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(1.0),
                committed_bytes: contract.total_resident_bytes(),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            legacy_total_peak,
            0,
            &[],
        )
        .expect("a fully warm base and control branch require no duplicate incremental bytes");
        assert_eq!(
            warm.context.predicted_peak_bytes, 0,
            "warm credit must remove the base and typed control exactly once"
        );

        let legacy = evaluate_request_with_budget(
            &fixture_generator(),
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            fixture_budget(1.0),
            legacy_total_peak,
            0,
            &[],
        )
        .expect("a non-adopting provider preserves the legacy whole-spec estimate");
        assert_eq!(legacy.context.predicted_peak_bytes, legacy_total_peak);
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
    fn mage_adapter_requests_require_and_consume_declared_residency() {
        use gen_core::{
            ComponentPrecisionFloor, MemoryComponentKind, MemoryFormulaKind, MemoryFormulaVariable,
            MemoryResidentComponent, PrecisionFloorComponent,
        };

        let mut inputs = request_inputs(512, 512, 1);
        inputs.adapter_count = 1;
        let root_guard = tempfile::Builder::new()
            .prefix("mage-request-adapter-plan-")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        let base = root.join("base.safetensors");
        let adapter = root.join("adapter.safetensors");
        std::fs::write(&base, vec![0_u8; 100]).unwrap();
        std::fs::write(&adapter, vec![0_u8; 5_750]).unwrap();
        let spec = LoadSpec::new(WeightsSource::File(base))
            .with_quant(gen_core::Quant::Q4)
            .with_adapters(vec![gen_core::AdapterSpec::new(
                adapter,
                1.0,
                gen_core::AdapterKind::Lora,
            )]);
        let plan =
            MlxRequestPlan::for_spec_and_manifest("mage_flow", "mage_flow", &spec, None, None);
        assert_eq!(plan.folded_adapter_bytes, 5_750);
        let error = evaluate_request_with_budget(
            &request_generator(Some(mage_request_contract())),
            &plan,
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
        assert!(error.contains("did not declare load-exact adapter residency"));

        const BASE_PEAK_GIB: f64 = 18.73;
        const ADAPTER_GIB: f64 = 5.75;
        let mut contract = mage_request_contract();
        let phases = contract.lifecycle.phases.clone();
        contract.asset_facts.overlay_bytes = gib_to_bytes(ADAPTER_GIB);
        contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::OverlayBytes,
            ],
            resident_components: vec![MemoryResidentComponent {
                id: "adapter_stack".to_owned(),
                kind: MemoryComponentKind::AdapterStack,
                resident_bytes: gib_to_bytes(ADAPTER_GIB),
                bounded_by: None,
                residency: gen_core::MemoryComponentResidency::WholeRender,
            }],
        };
        const PRECISION_FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
            component: PrecisionFloorComponent::TransformerHead,
            selected_tier: gen_core::Quant::Q4,
            resident_tier: gen_core::Quant::Q8,
        }];
        let mut generator = request_generator(Some(contract.clone()));
        generator.descriptor.capabilities.component_precision_floors = PRECISION_FLOORS;
        let evaluated = evaluate_request_with_budget(
            &generator,
            &plan,
            &inputs,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            MemoryBudget {
                total_bytes: gib_to_bytes(128.0),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            gib_to_bytes(BASE_PEAK_GIB),
            0,
            &[],
        )
        .expect("load-exact adapter residency narrows Mage's refusal");
        assert_eq!(
            evaluated.context.predicted_peak_bytes,
            gib_to_bytes(BASE_PEAK_GIB).saturating_add(gib_to_bytes(ADAPTER_GIB)),
            "the measured 18.73 GiB base must become 24.48 GiB with additive adapters"
        );
        assert_eq!(
            evaluated.context.selection.tier.component_precision_floors, PRECISION_FLOORS,
            "provider precision floors must participate in the selected evidence identity"
        );

        contract.asset_facts.overlay_bytes = 0;
        if let MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            ..
        } = &mut contract.formula
        {
            resident_components[0].resident_bytes = 0;
        }
        assert!(
            evaluate_request_with_budget(
                &request_generator(Some(contract)),
                &plan,
                &inputs,
                MemoryCacheState::Cold,
                OffloadPolicy::Resident,
                MemoryBudget {
                    total_bytes: gib_to_bytes(128.0),
                    committed_bytes: 0,
                    reclaimable_bytes: 0,
                    reserved_headroom_bytes: 0,
                },
                gib_to_bytes(BASE_PEAK_GIB),
                0,
                &[],
            )
            .is_err(),
            "mutation guard: zero adapter bytes must restore the refusal"
        );
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
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            MAGE_CALIBRATION_FINGERPRINT,
            gen_core::LoadShape::EagerMaterialization,
        ));
        contract.asset_facts.base_bytes = gib_to_bytes(6.0);
        contract.asset_facts.transformer_bytes = gib_to_bytes(6.0);
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
                backend: gen_core::MemoryBackend::Mlx,
                tier: plan.tier,
                load_shape: gen_core::LoadShape::EagerMaterialization,
                mode: memory_mode_from_mode_key("edit"),
                overlay: inputs.overlay.clone(),
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: inputs.reference_count,
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
        // sc-18096 repin. The mismatched record's 5 GiB raw peak fits the 10 GiB budget easily, so
        // ANY error here proves the record was excluded rather than authorizing the fit — the
        // fail-closed property this test owns. What changed: the bounded-decode rung now also
        // carries a synthesized floor estimate (weights 6 GiB + headroom 6 GiB, widened to 13.2),
        // so the refusal is the honest "no rung fits with margins" `Reject` quoting the floor's
        // widened requirement instead of an `Unverified`/`FingerprintMismatch` refusal.
        assert!(
            error.contains("needs 13.20 GiB"),
            "the mismatched record must not authorize the fit; the refusal must quote the \
             estimate floor instead: {error}"
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
        let empty_guard = tempfile::Builder::new()
            .prefix("mage_full_gate_")
            .tempdir()
            .expect("temp dir");
        let empty = empty_guard.path();
        assert_eq!(
            full_finetune_memory_error(empty, 1024, 1, "Mage-Flow Base"),
            None
        );
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

    /// NOTE (sc-16014): these numbers remain the bf16/MXFP4 calibration; they no longer describe q4/q8,
    /// whose re-hosted affine packs eliminated the tier-integrity exception. The ledger and this file
    /// share the `sc-16014-resolution: rehosted-q4-q8` marker so a future artifact change must reconcile
    /// both sources rather than reviving or erasing the exception silently.
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
    fn lens_provider_footprint_distinguishes_storage_format() {
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_sc11924_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        for (component, bytes) in [("text_encoder", 13), ("transformer", 11), ("vae", 3)] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).expect("component dir");
            std::fs::write(dir.join("model.safetensors"), vec![0; bytes]).expect("fixture");
        }
        std::fs::write(
            root.join("text_encoder").join("config.json"),
            r#"{"dtype":"bfloat16","quantization_config":{"quant_method":"mxfp4"}}"#,
        )
        .expect("mxfp4 marker");
        let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
        let (dense_total, dense_te, dense_headroom) = spec_component_bytes("lens_turbo", &spec);
        let expected_te = (30.07 * BYTES_PER_GIB).ceil() as u64;
        assert_eq!(dense_te, expected_te);
        assert_eq!(dense_total, expected_te + 14);
        assert_eq!(dense_headroom, HeadroomAllowance::LENS_DENSE);
        // sc-16195: a bare measured transient carries NO OS reserve, so the request estimator
        // must leave the whole allowance in its area term for this path.
        assert_eq!(dense_headroom.os_reserve_gb, 0.0);
        for engine_id in ["lens", "lens_turbo"] {
            let plan =
                MlxRequestPlan::for_spec_and_manifest(engine_id, engine_id, &spec, None, None);
            assert_eq!(plan.fixed_reserve_bytes, gib_to_bytes(2.0));
            assert_eq!(
                plan.activation_headroom_bytes,
                gib_to_bytes(2.0 + LENS_DENSE_HEADROOM_GB),
                "{engine_id} dense/MXFP4 must preserve the format-aware 29.88 GiB activation fallback"
            );
        }

        std::fs::write(
            root.join("text_encoder").join("config.json"),
            r#"{"dtype":"bfloat16"}"#,
        )
        .expect("bf16 marker");
        let (bf16_total, bf16_te, bf16_headroom) = spec_component_bytes("lens_turbo", &spec);
        assert_eq!(
            (bf16_total, bf16_te),
            (27, 13),
            "bf16-on-disk has no MXFP4 materialization delta"
        );
        assert_eq!(
            bf16_headroom,
            HeadroomAllowance::LENS_DENSE,
            "the Lens activation transient remains architecture-specific even without a weight upcast"
        );

        std::fs::write(
            root.join("text_encoder").join("config.json"),
            // The re-hosted q4/q8 configs retain this inherited MXFP4 marker. The load-bearing MLX
            // affine marker must win, or the fit gate would inflate the already-packed experts.
            r#"{"quantization":{"bits":8,"group_size":64},"quantization_config":{"quant_method":"mxfp4"}}"#,
        )
        .expect("packed marker with inherited MXFP4 metadata");
        let (packed_total, packed_te, packed_headroom) = spec_component_bytes("lens_turbo", &spec);
        assert_eq!((packed_total, packed_te), (27, 13));
        assert_eq!(packed_headroom, HeadroomAllowance::GENERIC);
        for engine_id in ["lens", "lens_turbo"] {
            let plan =
                MlxRequestPlan::for_spec_and_manifest(engine_id, engine_id, &spec, None, None);
            assert_eq!(plan.fixed_reserve_bytes, gib_to_bytes(2.0));
            assert_eq!(
                plan.activation_headroom_bytes,
                gib_to_bytes(16.0),
                "{engine_id} packed q4/q8 must preserve the generic 14 GiB activation fallback"
            );
        }
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
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_sum_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        let te = root.join("text_encoder");
        let dit = root.join("transformer");
        std::fs::create_dir_all(&te).expect("mk te");
        std::fs::create_dir_all(&dit).expect("mk dit");
        std::fs::write(te.join("model.safetensors"), vec![0u8; 1000]).expect("te weights");
        std::fs::write(dit.join("diffusion.safetensors"), vec![0u8; 2000]).expect("dit weights");
        // AppleDouble sidecar + a non-weight file must NOT be counted.
        std::fs::write(te.join("._model.safetensors"), vec![0u8; 500]).expect("sidecar");
        std::fs::write(dit.join("config.json"), vec![0u8; 700]).expect("config");

        assert_eq!(sum_safetensors_bytes(root), 3000);
        // Missing dir ⇒ 0 (no signal).
        assert_eq!(sum_safetensors_bytes(&root.join("nope")), 0);
    }

    /// Adapter factors are resident only when the provider keeps them as additive residuals.
    #[test]
    fn spec_adapter_bytes_distinguish_additive_and_folded_loads() {
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_adapters_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        let weights = root.join("weights");
        std::fs::create_dir_all(&weights).expect("weights dir");
        std::fs::write(weights.join("model.safetensors"), vec![0_u8; 1_000]).expect("base weights");
        let adapter = root.join("adapter.safetensors");
        std::fs::write(&adapter, vec![0_u8; 250]).expect("adapter weights");
        let adapter_spec = gen_core::AdapterSpec::new(adapter, 1.0, gen_core::AdapterKind::Lora);

        let additive = LoadSpec::new(WeightsSource::Dir(weights.clone()))
            .with_adapters(vec![adapter_spec.clone()]);
        assert_eq!(spec_component_bytes("mage_flow", &additive).0, 1_250);
        assert_eq!(
            MlxRequestPlan::for_spec_and_manifest("mage_flow", "mage_flow", &additive, None, None)
                .folded_adapter_bytes,
            250
        );

        let dense_wan = LoadSpec::new(WeightsSource::Dir(weights.clone()))
            .with_quant(gen_core::Quant::Q4)
            .with_adapters(vec![adapter_spec.clone()]);
        assert_eq!(
            spec_component_bytes("wan2_2_t2v_14b", &dense_wan).0,
            1_000,
            "dense Wan folds factors before load-time quantization"
        );
        assert_eq!(
            spec_component_bytes("wan_vace", &dense_wan).0,
            1_000,
            "Wan VACE always folds its dense adapter factors"
        );
        assert_eq!(
            spec_component_bytes("wan2_2_vace_fun_14b", &dense_wan).0,
            1_000,
            "Wan VACE-Fun folds each expert's factors before quantization"
        );

        std::fs::write(
            weights.join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .expect("packed marker");
        let packed_wan =
            LoadSpec::new(WeightsSource::Dir(weights)).with_adapters(vec![adapter_spec]);
        assert_eq!(
            spec_component_bytes("wan2_2_t2v_14b", &packed_wan).0,
            1_250,
            "pre-packed Wan retains adapter factors as additive residuals"
        );
    }

    /// sc-15154: a split-layout tier's staged co-requisites are part of what it loads. Mage-Flow's
    /// per-tier dir holds the DiT alone; the shared text encoder and VAE must still count.
    #[test]
    fn a_staged_component_counts_toward_the_tier_that_loads_it() {
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_staged_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
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
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_ctrl_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();

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
    }

    /// The gate derives sequential-capability from each engine's REGISTERED descriptor bit
    /// (`Capabilities::supports_sequential_offload`) rather than a hand-maintained allowlist (sc-10840,
    /// epic 10834). This exercises the LIVE registry, so it must see the force-linked `mlx_gen_*`
    /// providers — anchored (`use mlx_gen_* as _;` in `image_jobs`) only on macOS, the sole platform the
    /// MLX gate runs on. Off-Mac the image registry is empty, so this is macOS-gated exactly like the
    /// `engines.rs` descriptor sweeps. Selectable engines resolve true through the shared registry
    /// query, while a registered provider that stages unconditionally remains false for this specific
    /// request-selectable control.
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
        ] {
            assert!(
                engine_supports_sequential(id),
                "{id}: sc-10840 fan-out engine must advertise selectable sequential residency"
            );
        }
        // Bernini's provider is registered and physically stages every request, but it has no
        // Resident-warm mode and does not consume `OffloadPolicy`. Its descriptor must therefore
        // expose unconditional staging without falsely advertising the selectable Sequential control.
        let bernini = crate::inference_runtime::media()
            .generators()
            .find(|registration| (registration.descriptor)().id == "bernini")
            .expect("Bernini provider must be registered in the MLX runtime");
        let bernini_capabilities = (bernini.descriptor)().capabilities;
        assert!(!bernini_capabilities.supports_sequential_offload);
        assert!(bernini_capabilities.unconditionally_engages_staged_residency);
        // The DESCRIPTOR still does not advertise the selectable Sequential control — that is the
        // assertion directly above, and it is what keeps the knob off Bernini. `engine_supports_
        // sequential` is a different question: it feeds `decide_residency_with_headroom`, i.e. how
        // much memory to CHARGE. sc-19721 widened it to the disjunction because reading only
        // `supports_sequential_offload` made Bernini charge the SUM of planner + UMT5-XXL + both
        // experts — a co-residency `generate_impl` never creates. Unconditional staging is
        // sequential for accounting purposes even though it is not offerable as a control.
        assert!(engine_supports_sequential("bernini"));
        // A REGISTERED engine that does NOT advertise the bit stays false: sensenova's encoder is fused
        // into a unified MoT (`footprint` te=0) — no separable text encoder to drop, so residency buys
        // nothing and Sequential would be a no-op that OOMs. This proves the query reads the descriptor
        // BIT, not mere registry membership.
        assert!(!engine_supports_sequential("sensenova_u1_8b"));

        // sc-19721: WHICH engines reach `true` through the second bit rather than the first, pinned
        // as a set so the disjunction cannot quietly widen. Every one of these declares
        // `unconditionally_engages_staged_residency: true` and `supports_sequential_offload: false`
        // at the pinned revision: they stage physically on every generation and expose no selectable
        // control to honour. Bernini is here because inference moved it between the two bits, which
        // is what made the disjunction necessary.
        for id in ["bernini", "krea_realtime_14b", "ltx_2_3", "scail2_14b"] {
            let capabilities = crate::inference_runtime::media()
                .generators()
                .find(|reg| (reg.descriptor)().id == id)
                .map(|reg| (reg.descriptor)().capabilities)
                .unwrap_or_else(|| panic!("{id} is registered in the pinned bundle"));
            assert!(
                !capabilities.supports_sequential_offload,
                "{id}: this set is specifically the engines the FIRST bit does not cover"
            );
            assert!(
                capabilities.unconditionally_engages_staged_residency,
                "{id}: reaches the gate only through the unconditional-staging bit"
            );
            assert!(engine_supports_sequential(id));
        }
    }

    /// An id with no registered generator is never sequential-capable (the safe default: never select a
    /// residency policy the provider won't honor) — a cross-platform invariant.
    #[test]
    fn engine_supports_sequential_is_false_for_an_unregistered_id() {
        assert!(!engine_supports_sequential("no_such_engine_xyz"));
    }

    #[test]
    fn production_residency_policies_materialize_each_provider_under_its_bound_shape() {
        let eager = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("fixture")));
        let z_image = with_selected_sequential_shape("z_image_turbo", eager.clone());
        assert_eq!(z_image.offload_policy, OffloadPolicy::Sequential);
        assert_eq!(
            z_image.load_shape,
            gen_core::LoadShape::DeferredMaterialization,
            "the shipped Z-Image rung-4 binding must be producible by the production cold-load route"
        );

        let qwen_resident = eager
            .clone()
            .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
        let qwen_resident = apply_residency_policy(qwen_resident, "qwen_image")
            .expect("production deferred Qwen resident route");
        assert_eq!(qwen_resident.offload_policy, OffloadPolicy::Resident);
        assert_eq!(
            qwen_resident.load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );
        let qwen_sequential = with_selected_sequential_shape("qwen_image", qwen_resident.clone());
        assert_eq!(qwen_sequential.offload_policy, OffloadPolicy::Sequential);
        assert_eq!(
            qwen_sequential.load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );

        let krea = with_selected_sequential_shape("krea_2_turbo_control", eager);
        assert_eq!(krea.offload_policy, OffloadPolicy::Sequential);
        assert_eq!(
            krea.load_shape,
            gen_core::LoadShape::EagerMaterialization,
            "Krea's shipped bounded-decode bindings remain on its production eager route"
        );
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
            "z_image",
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

    /// The load-time weights floor (`weights_floor_load_admission`, formerly
    /// `legacy_admission_override`) must preserve its settled small-Mac outcomes byte-for-byte
    /// through sc-18096's retirement of the request-path freeze: a spec whose bare staged weights
    /// fit still loads, and one whose weights cannot be held resident under any policy still
    /// rejects at load.
    #[test]
    fn weights_floor_load_admission_preserves_small_mac_behavior() {
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

    /// Decision 1's existing 8 GiB policy guard, preserved byte-for-byte through sc-18096: the
    /// LOAD-time weights floor still admits this real on-disk q4 footprint (the request-scoped
    /// estimate ladder owns transient bounding, not the loader). This is a policy outcome only,
    /// not a model calibration or implementation claim. Measured
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
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_te_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
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
        assert_eq!(sum_text_encoder_bytes(root), 4000);
        // The whole-model sum includes everything.
        assert_eq!(sum_safetensors_bytes(root), 13400);
        // Missing dir ⇒ 0.
        assert_eq!(sum_text_encoder_bytes(&root.join("nope")), 0);
    }

    // HF cache stores each shard as a symlink into `blobs/`; the gate must follow those to the real
    // byte size. The synthetic test above uses plain files, so exercise the symlink layout here.
    #[cfg(unix)]
    #[test]
    fn sum_safetensors_follows_hf_cache_symlinks() {
        use std::os::unix::fs::symlink;
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_symlink_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
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
    }

    #[cfg(unix)]
    #[test]
    fn sum_safetensors_terminates_on_directory_symlink_cycles() {
        use std::os::unix::fs::symlink;
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_cycle_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        let weights = root.join("weights");
        std::fs::create_dir_all(&weights).expect("mk weights");
        std::fs::write(weights.join("model.safetensors"), vec![0_u8; 4096]).expect("write weights");
        symlink(root, weights.join("cycle")).expect("create directory cycle");

        assert_eq!(sum_safetensors_bytes(root), 4096);
    }

    /// sc-10894: on a boogu-style snapshot (text encoder under `mllm/`, not `text_encoder*`), the
    /// subdir scan reads ZERO, but `resolve_text_encoder_bytes` PREFERS a provider footprint value when
    /// present and only falls back to the scan when it is `None`.
    #[test]
    fn resolve_text_encoder_prefers_footprint_over_subdir_scan() {
        let root_guard = tempfile::Builder::new()
            .prefix("mlx_fit_gate_resolve_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        // Encoder under `mllm/`, DiT `transformer/`, VAE `vae/` — NO `text_encoder*` subdir.
        for (sub, bytes) in [("mllm", 1500usize), ("transformer", 9000), ("vae", 400)] {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).expect("mk subdir");
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).expect("weights");
        }
        // The historical subdir scan finds no `text_encoder*` → 0 (the bug this seam fixes).
        assert_eq!(sum_text_encoder_bytes(root), 0);
        // The whole-model sum still sees every component.
        assert_eq!(sum_safetensors_bytes(root), 10900);
        // No footprint declared ⇒ fall back to the (zero) subdir scan.
        assert_eq!(resolve_text_encoder_bytes(None, root), 0);
        // A provider footprint (the `mllm/` bytes) is preferred, even though the scan reads zero.
        assert_eq!(resolve_text_encoder_bytes(Some(1500), root), 1500);
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

    /// The REAL shipped `krea_2_turbo` manifest entry, for the two live sc-16482 tests below.
    ///
    /// `SCENEWORKS_KREA_MANIFEST_JSON` points at the entry extracted from
    /// `config/manifests/builtin.models.jsonc` (that file is JSONC; this crate has no JSONC reader,
    /// and hand-copying the entry into the test would let it drift from what actually ships):
    /// ```text
    /// node --input-type=module -e "
    ///   import { readFileSync, writeFileSync } from 'node:fs';
    ///   const { stripJsoncComments } = await import('./scripts/lib/jsonc.mjs');
    ///   const doc = JSON.parse(stripJsoncComments(readFileSync('config/manifests/builtin.models.jsonc','utf8')));
    ///   const list = Array.isArray(doc) ? doc : (doc.models ?? Object.values(doc).find(Array.isArray));
    ///   writeFileSync(process.env.OUT, JSON.stringify(list.find(m => m && m.id === 'krea_2_turbo')));
    /// "
    /// ```
    /// Returns `None` (with a SKIP note) rather than panicking, so `--ignored` on a machine without
    /// the setup reports honestly instead of failing.
    #[cfg(target_os = "macos")]
    fn real_krea_manifest_entry() -> Option<JsonObject<String, Value>> {
        let Ok(path) = std::env::var("SCENEWORKS_KREA_MANIFEST_JSON") else {
            eprintln!("SKIP: set SCENEWORKS_KREA_MANIFEST_JSON (see doc comment for extraction)");
            return None;
        };
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read SCENEWORKS_KREA_MANIFEST_JSON {path}: {error}"));
        let manifest: JsonObject<String, Value> =
            serde_json::from_slice(&bytes).expect("manifest entry object");
        assert_eq!(
            manifest.get("id").and_then(Value::as_str),
            Some("krea_2_turbo"),
            "SCENEWORKS_KREA_MANIFEST_JSON must be the krea_2_turbo entry"
        );
        Some(manifest)
    }

    /// sc-16482 on the REAL install (ignored — needs krea-2-turbo-mlx on disk and its actual receipt).
    ///
    /// Reproduces the reported failure end to end against the artifacts that produced it, rather than
    /// a fixture: the user's own `.sceneworks-download-complete.json` (three `backfilled: true`
    /// receipts, no `artifactTreeStamp`) plus the real `SceneWorks/krea-2-turbo-mlx` snapshot in the HF
    /// cache and the real `krea_2_turbo` manifest entry (2 calibrations, both `krea_2_turbo_control`).
    ///
    /// The receipt is COPIED into a temp data dir, so running this never mutates the real install.
    /// `HF_HUB_CACHE` still points at the real cache, so the snapshot and every stat'ed file are real.
    ///
    /// Before the fix this returned `Err("... invalid MLX calibration opt-in: the resolver supplied no
    /// immutable artifact provenance")`. Run explicitly:
    ///   SCENEWORKS_KREA_MANIFEST_JSON=<extracted entry>.json \
    ///     cargo test -p sceneworks-worker --lib -- --ignored --nocapture krea_real_install
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs the real krea-2-turbo-mlx install + SCENEWORKS_KREA_MANIFEST_JSON"]
    fn krea_real_install_backfilled_receipt_now_admits() {
        let home = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"));
        let real_marker = home
            .join("SceneWorks/data/models/SceneWorks__krea-2-turbo-mlx")
            .join(crate::INSTALL_MARKER);
        let Ok(receipt_bytes) = std::fs::read(&real_marker) else {
            eprintln!("SKIP: no real receipt at {}", real_marker.display());
            return;
        };
        let Some(manifest) = real_krea_manifest_entry() else {
            return;
        };
        let calibrations = manifest["mlx"]["calibrations"]
            .as_array()
            .expect("real manifest still declares mlx.calibrations");
        assert!(
            !calibrations.is_empty(),
            "this test is meaningless if the opt-in was removed"
        );

        // The real artifact, untouched.
        let hub = home.join(".cache/huggingface/hub");
        let snapshots = hub.join("models--SceneWorks--krea-2-turbo-mlx/snapshots");
        let Some(snapshot) = std::fs::read_dir(&snapshots)
            .ok()
            .and_then(|entries| entries.flatten().map(|e| e.path()).find(|p| p.is_dir()))
        else {
            eprintln!("SKIP: no krea snapshot under {}", snapshots.display());
            return;
        };
        let q8 = snapshot.join("q8");
        if !q8.is_dir() {
            eprintln!("SKIP: no q8 tier at {}", q8.display());
            return;
        }

        // Copy-on-read data dir: the receipt is real, the file we may stamp is a throwaway.
        let data = tempfile::tempdir().expect("data dir");
        let marker_dir = data
            .path()
            .join("models")
            .join(crate::paths::safe_download_dir(
                "SceneWorks/krea-2-turbo-mlx",
            ));
        std::fs::create_dir_all(&marker_dir).expect("marker dir");
        std::fs::write(marker_dir.join(crate::INSTALL_MARKER), &receipt_bytes).expect("copy");

        let outcome =
            crate::test_env::temp_env_var("HF_HUB_CACHE", hub.to_str().expect("hub path"), || {
                let unrepaired = crate::model_jobs::huggingface_receipt_weights(
                    data.path(),
                    "SceneWorks/krea-2-turbo-mlx",
                    Some("krea_2_turbo"),
                    Some("q8"),
                    crate::model_jobs::ProvenanceRepair::Skip,
                )
                .expect("the real backfilled receipt still resolves loadable weights");
                assert_eq!(unrepaired.path, q8, "resolves the real q8 tier");
                assert_eq!(
                    unrepaired.provenance, None,
                    "the real receipt carries no artifactTreeStamp — this IS the reported cause"
                );

                let repaired = crate::model_jobs::huggingface_receipt_weights(
                    data.path(),
                    "SceneWorks/krea-2-turbo-mlx",
                    Some("krea_2_turbo"),
                    Some("q8"),
                    crate::model_jobs::ProvenanceRepair::Allow,
                )
                .expect("resolves")
                .provenance
                .expect("repair must establish provenance for the real install");

                let spec =
                    LoadSpec::new(WeightsSource::Dir(q8.clone())).with_quant(gen_core::Quant::Q8);
                let plan = MlxRequestPlan::for_spec_and_manifest(
                    "krea_2_turbo",
                    "krea_2_turbo",
                    &spec,
                    Some(&manifest),
                    Some(repaired.clone()),
                );
                (
                    repaired,
                    packaged_admission_route(
                        &plan,
                        &fixture_inputs(1024, 1024),
                        "text_to_image",
                        fixture_budget(128.0),
                        // The base t2i lane is undeclared in the closure config, so production
                        // resolves the empty fail-closed expectation here. Reproduced, not papered
                        // over: every krea binding names `krea_2_turbo_control`.
                        &live_mlx_closure_digest("krea_2_turbo"),
                    ),
                )
            });

        let (provenance, route) = outcome;
        eprintln!(
            "real install → repo={} revision={} variant={} tier={:?}",
            provenance.identity.repository,
            provenance.identity.revision,
            provenance.identity.variant,
            provenance.fixed_artifact_tier
        );
        let route = route.expect("the reported generation must no longer be refused");
        eprintln!(
            "admission → path={:?} fallback={:?}",
            route.path, route.fallback_reason
        );
        // The base t2i lane has no bindings of its own (every calibration names
        // `krea_2_turbo_control`), so legacy is the CORRECT route here. What matters is that it is
        // reached by evaluation rather than by a refusal.
        assert_eq!(route.path, AdmissionPath::Legacy);
        assert_ne!(
            route.fallback_reason,
            Some(LegacyAdmissionReason::NoProvenance),
            "provenance was repaired, so the route must not still report it missing"
        );
    }

    /// sc-16482 FULL end-to-end on real weights (ignored — needs the krea-2-turbo-mlx q8 turnkey).
    ///
    /// The sibling test above proves the admission route stops refusing. This one closes the last gap:
    /// it drives the REAL production seam — `evaluate_request`, the call `image_jobs/base.rs` makes
    /// immediately before generation, against a live loaded generator — and then actually renders. The
    /// base `krea_2_turbo` text-to-image lane at q8/1024² is the exact configuration that was reported
    /// broken, so a written PNG is the end of the causal chain the bug report started.
    ///
    ///   SCENEWORKS_KREA_MANIFEST_JSON=<entry>.json KREA_BASE_OUT_DIR=<dir> \
    ///     cargo test -p sceneworks-worker --lib -- --ignored --nocapture krea_real_install_renders
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight MLX render; needs the krea-2-turbo-mlx q8 turnkey + SCENEWORKS_KREA_MANIFEST_JSON"]
    fn krea_real_install_renders_through_the_live_gate() {
        let home = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"));
        let hub = home.join(".cache/huggingface/hub");
        let snapshots = hub.join("models--SceneWorks--krea-2-turbo-mlx/snapshots");
        let Some(q8) = std::fs::read_dir(&snapshots).ok().and_then(|entries| {
            entries.flatten().map(|e| e.path().join("q8")).find(|p| {
                p.join("transformer/diffusion_pytorch_model.safetensors")
                    .is_file()
            })
        }) else {
            eprintln!("SKIP: no krea q8 turnkey under {}", snapshots.display());
            return;
        };
        let Some(manifest) = real_krea_manifest_entry() else {
            return;
        };
        let out_dir = std::path::PathBuf::from(crate::smoke_support::env_or(
            "KREA_BASE_OUT_DIR",
            "/tmp/krea_base_smoke",
        ));
        std::fs::create_dir_all(&out_dir).expect("out dir");

        // Real receipt, copied so the real install is never mutated.
        let data = tempfile::tempdir().expect("data dir");
        let marker_dir = data
            .path()
            .join("models")
            .join(crate::paths::safe_download_dir(
                "SceneWorks/krea-2-turbo-mlx",
            ));
        std::fs::create_dir_all(&marker_dir).expect("marker dir");
        std::fs::copy(
            home.join("SceneWorks/data/models/SceneWorks__krea-2-turbo-mlx")
                .join(crate::INSTALL_MARKER),
            marker_dir.join(crate::INSTALL_MARKER),
        )
        .expect("copy real receipt");

        let provenance =
            crate::test_env::temp_env_var("HF_HUB_CACHE", hub.to_str().expect("hub path"), || {
                crate::model_jobs::huggingface_receipt_weights(
                    data.path(),
                    "SceneWorks/krea-2-turbo-mlx",
                    Some("krea_2_turbo"),
                    Some("q8"),
                    crate::model_jobs::ProvenanceRepair::Allow,
                )
                .expect("resolves")
                .provenance
                .expect("repair establishes provenance for the real install")
            });

        let spec = LoadSpec::new(WeightsSource::Dir(q8.clone())).with_quant(gen_core::Quant::Q8);
        let plan = MlxRequestPlan::for_spec_and_manifest(
            "krea_2_turbo",
            "krea_2_turbo",
            &spec,
            Some(&manifest),
            Some(provenance),
        );
        eprintln!("[smoke] loading krea_2_turbo q8 from {}", q8.display());
        let generator =
            crate::inference_runtime::load("krea_2_turbo", &spec).expect("load krea_2_turbo");

        // THE production call: exactly what base.rs runs per generation, on a live generator.
        let evaluation = evaluate_request(
            &*generator,
            &plan,
            &fixture_inputs(1024, 1024),
            MemoryCacheState::Warm,
            OffloadPolicy::Resident,
            0,
        )
        .expect("the reported generation must be admitted, not refused");
        eprintln!("[smoke] gate admitted: {evaluation:?}");

        let request = gen_core::GenerationRequest {
            prompt: "a windswept basalt sea stack at dawn, low mist, long exposure water"
                .to_owned(),
            width: 1024,
            height: 1024,
            count: 1,
            seed: Some(20260801),
            steps: Some(8),
            // Krea Turbo is CFG-free (distilled).
            guidance: None,
            cancel: gen_core::CancelFlag::new(),
            ..Default::default()
        };
        let image = match generator
            .generate(&request, &mut |_| {})
            .expect("krea_2_turbo generate")
        {
            gen_core::GenerationOutput::Images(mut images) => images.pop().expect("one image"),
            other => panic!("expected Images, got {other:?}"),
        };
        let path = out_dir.join("krea_base_q8_1024.png");
        crate::smoke_support::save_png(&image, &path);
        let std_dev = crate::smoke_support::image_std(&image);
        eprintln!("[smoke] rendered {} (std {std_dev:.2})", path.display());
        assert_eq!((image.width, image.height), (1024, 1024));
        assert!(
            !crate::smoke_support::is_all_zero(&image),
            "render is entirely black"
        );
        assert!(
            std_dev > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
            "render looks degenerate (std {std_dev:.2})"
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
        let root_guard = tempfile::Builder::new()
            .prefix("mochi_resident_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
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
    }

    /// sc-19721 — the AdaLN exclusion, at the only consumer that can honour it.
    ///
    /// gen-core landed `MemoryComponentKind::TransformerSubStack` +
    /// `MemoryComponentResidency::PrecomputedThenEvicted` (SC-18665) with **zero** non-test callers,
    /// so the declaration moved no estimate by a byte. These grade the consumer, and they grade the
    /// half that is easy to get wrong in the OOM direction: the drop lowers the STEADY STATE, and
    /// the declaring phase still holds the whole sub-stack at the precompute instant.
    ///
    /// Every figure is MiniMax-H3's own declaration, tied to the pinned engine's public constants by
    /// [`the_h3_eviction_figures_are_the_pinned_engines_own`]. Nothing is asserted against a default:
    /// `evicted_component_bytes()` is asserted non-zero and equal to `resident − retained` first, so
    /// a contract that declared no component (or a build with the feature deleted) cannot pass by
    /// comparing zero with zero.
    mod adaln_exclusion_reaches_the_estimate_floor {
        use super::*;

        /// Qwen3-VL-32B, the dense conditioning stack — `mlx_gen_minimax_h3::TEXT_ENCODER_BYTES`.
        const TEXT_ENCODER_BYTES: u64 = 66_714_912_872;
        /// One bf16 33B DiT partition — `DIT_BF16_BYTES`. A render loads exactly one.
        const DIT_BF16_BYTES: u64 = 66_280_504_216;
        /// Video VAE + audio VAE: the decode-phase components, i.e. the part of the `heavy` lump
        /// that is NOT the transformer.
        const VAE_BYTES: u64 = 10_415_558_888 + 605_429_340;
        /// The 50-block `adaln_proj` stack at bf16 — `ADALN_EVICTED_BYTES`.
        const ADALN_RESIDENT_BF16_BYTES: u64 = 26_020_915_200;
        /// The same 50 projections on the packed q4 tier — `ADALN_EVICTED_Q4_BYTES`. **The lever is
        /// tier-scaled**, which is why q4 is graded here and not only bf16.
        const ADALN_RESIDENT_Q4_BYTES: u64 = 7_325_337_600;
        /// What the precompute keeps in the projections' place —
        /// `ADALN_MODULATION_TABLE_MAX_BYTES`. Deliberately **not** tier-scaled: the table's dtype is
        /// the compute dtype, not the tier's bit width. Applying one factor to both would be wrong at
        /// every tier but bf16, and this pair is what makes that visible.
        const ADALN_RETAINED_BYTES: u64 = 3_870_720_000;

        /// A packed conditioning tier (sc-19120 re-hosts one). Used as a STAND-IN so the DiT-side
        /// arithmetic is observable at all: with the dense encoder above, `max(conditioning, heavy)`
        /// is pinned by the conditioner at q4 and the whole exclusion is invisible at the floor —
        /// itself a finding, graded by [`q4_with_the_dense_conditioner_moves_nothing`].
        const PACKED_TEXT_ENCODER_BYTES: u64 = TEXT_ENCODER_BYTES / 4;

        const STAGED: &[MemoryStrategy] = &[MemoryStrategy::StagedResidency];
        const CO_RESIDENT: &[MemoryStrategy] = &[];
        const STAGED_PLUS_RUNG4: &[MemoryStrategy] = &[
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ];

        /// MiniMax-H3's contract shape: three base components, and one `TransformerSubStack`
        /// component inside `transformer_bytes` that is precomputed and evicted during Denoise.
        /// `evicting = false` builds the byte-identical contract a provider had before SC-18665
        /// existed — the control every delta below is measured against.
        fn h3_shaped_contract(
            conditioning_bytes: u64,
            dit_bytes: u64,
            adaln_resident_bytes: u64,
            evicting: bool,
        ) -> MemoryProviderContract {
            let mut contract = MemoryProviderContract::compatibility_default(
                "minimax_h3",
                MemoryBackendRealization::MlxMetal {
                    bounded_wired_residency: false,
                    lazy_or_mmap_materialization: true,
                    explicit_evaluation_and_synchronization: false,
                    cache_eviction: true,
                },
            );
            contract.asset_facts.conditioning_bytes = conditioning_bytes;
            contract.asset_facts.transformer_bytes = dit_bytes;
            contract.asset_facts.decoder_bytes = VAE_BYTES;
            contract.asset_facts.base_bytes = conditioning_bytes + dit_bytes + VAE_BYTES;
            let phases = contract.lifecycle.phases.clone();
            contract.formula = gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
                resident_components: vec![gen_core::MemoryResidentComponent {
                    id: "dit_adaln_proj_stack".to_owned(),
                    kind: gen_core::MemoryComponentKind::TransformerSubStack(
                        TransformerComponent::Dit,
                    ),
                    resident_bytes: adaln_resident_bytes,
                    bounded_by: None,
                    residency: if evicting {
                        gen_core::MemoryComponentResidency::PrecomputedThenEvicted {
                            precomputed_in: gen_core::MemoryPhase::Denoise,
                            retained_bytes: ADALN_RETAINED_BYTES,
                            evidence: "sc-19721 fixture mirroring \
                                       mlx-gen-minimax-h3::memory_strategy::adaln_component"
                                .to_owned(),
                        }
                    } else {
                        gen_core::MemoryComponentResidency::WholeRender
                    },
                }],
            };
            contract
        }

        /// The declared drop, read off the contract rather than recomputed — and proved non-zero and
        /// equal to the provider's own `resident − retained`, so none of the deltas below can be a
        /// zero-versus-zero pass.
        fn declared_eviction(contract: &MemoryProviderContract, adaln_resident: u64) -> u64 {
            let evicted = contract.evicted_component_bytes();
            assert_eq!(
                evicted,
                adaln_resident - ADALN_RETAINED_BYTES,
                "the fixture must declare the NET drop the provider declares: the precompute keeps \
                 a modulation table in the projections' place, and a gross declaration claims a \
                 saving the runtime does not deliver"
            );
            assert!(
                evicted > 0,
                "a zero drop would make every delta below vacuous"
            );
            assert_eq!(
                contract.steady_state_transformer_bytes(),
                contract.asset_facts.transformer_bytes - evicted,
                "the accessor under test must correct transformer_bytes by exactly that drop"
            );
            evicted
        }

        /// bf16 + the dense Qwen3-VL-32B conditioner: the shipped cell today.
        ///
        /// The lump `heavy = transformer + decoder` is 77.30 GB against a 66.71 GB conditioner, so
        /// the staged floor is the lump. The drop cancels against the decoder's bytes — which the
        /// denoise phase does not hold — until it hits the load-exact transformer, and there the
        /// clamp stops it. The conditioner is then the taller leg, so the floor lands on it.
        #[test]
        fn bf16_staged_floor_falls_to_the_conditioning_stack_and_never_below_the_transformer() {
            let legacy = h3_shaped_contract(
                TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES,
                ADALN_RESIDENT_BF16_BYTES,
                false,
            );
            let adopted = h3_shaped_contract(
                TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES,
                ADALN_RESIDENT_BF16_BYTES,
                true,
            );
            let evicted = declared_eviction(&adopted, ADALN_RESIDENT_BF16_BYTES);
            assert_eq!(
                legacy.evicted_component_bytes(),
                0,
                "the control declares WholeRender, so it drops nothing"
            );

            let before = estimate_floor_weights_bytes(&legacy, STAGED);
            let after = estimate_floor_weights_bytes(&adopted, STAGED);
            assert_eq!(
                before,
                DIT_BF16_BYTES + VAE_BYTES,
                "before: the staged floor is the transformer+decoder lump"
            );
            assert_eq!(
                after, TEXT_ENCODER_BYTES,
                "after: the lump falls below the conditioning stack, which becomes the binding leg"
            );
            assert!(
                before > after,
                "a configuration between {after} and {before} bytes was refused and now fits"
            );
            assert!(
                after >= DIT_BF16_BYTES,
                "the precompute instant holds the WHOLE DiT ({DIT_BF16_BYTES} B, sub-stack \
                 included). A floor below it under-charges that instant by up to {evicted} B — the \
                 OOM direction, and the exact asymmetry ADALN_MODULATION_TABLE_MAX_BYTES exists to \
                 avoid."
            );
        }

        /// bf16 + a packed conditioner: the clamp is the ONLY thing stopping the fall, and the floor
        /// lands exactly on the load-exact transformer. This is the cell that would go silently
        /// wrong if the eviction were subtracted raw.
        #[test]
        fn bf16_staged_floor_stops_exactly_at_the_load_exact_transformer() {
            let legacy = h3_shaped_contract(
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES,
                ADALN_RESIDENT_BF16_BYTES,
                false,
            );
            let adopted = h3_shaped_contract(
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES,
                ADALN_RESIDENT_BF16_BYTES,
                true,
            );
            let evicted = declared_eviction(&adopted, ADALN_RESIDENT_BF16_BYTES);
            assert!(
                evicted > VAE_BYTES,
                "this cell only means something while the drop is bigger than the decode-phase \
                 bytes it can cancel against"
            );

            let before = estimate_floor_weights_bytes(&legacy, STAGED);
            let after = estimate_floor_weights_bytes(&adopted, STAGED);
            assert_eq!(before, DIT_BF16_BYTES + VAE_BYTES);
            assert_eq!(
                after, DIT_BF16_BYTES,
                "the fall stops at the load-exact transformer — the precompute instant — so the \
                 reduction is the decode-phase bytes ({VAE_BYTES}), never the raw drop ({evicted})"
            );
            assert_eq!(before - after, VAE_BYTES);
        }

        /// q4 + a packed conditioner: neither clamp binds, so the floor falls by EXACTLY
        /// `evicted_component_bytes()`. Grading a second tier is not redundant — the resident side
        /// is tier-scaled (26.02 → 7.33 GB) while the retained table is not, so a single factor
        /// applied to both would be right at bf16 and wrong here.
        #[test]
        fn q4_staged_floor_falls_by_exactly_the_declared_eviction() {
            let legacy = h3_shaped_contract(
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES / 4,
                ADALN_RESIDENT_Q4_BYTES,
                false,
            );
            let adopted = h3_shaped_contract(
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES / 4,
                ADALN_RESIDENT_Q4_BYTES,
                true,
            );
            let evicted = declared_eviction(&adopted, ADALN_RESIDENT_Q4_BYTES);
            assert!(
                evicted < VAE_BYTES,
                "at q4 the drop is smaller than the decode-phase bytes, so the clamp does not bind \
                 and the whole declared exclusion reaches the floor"
            );
            assert_ne!(
                evicted,
                ADALN_RESIDENT_BF16_BYTES - ADALN_RETAINED_BYTES,
                "the exclusion must be tier-scaled; a q4 drop equal to bf16's is the error this \
                 cell exists to catch"
            );

            let before = estimate_floor_weights_bytes(&legacy, STAGED);
            let after = estimate_floor_weights_bytes(&adopted, STAGED);
            assert_eq!(
                before - after,
                evicted,
                "the whole declared exclusion reaches the fit gate at this cell"
            );
            assert_eq!(after, DIT_BF16_BYTES / 4 + VAE_BYTES - evicted);
        }

        /// q4 + the dense conditioner: the floor does NOT move, because the 66.71 GB conditioning
        /// stack is taller than the whole packed DiT pipeline. Pinned rather than left unsaid: it is
        /// why sc-19120's packed text encoder and this change are a pair, and why a q4 assertion
        /// against the shipped dense encoder would have looked like the fix doing nothing.
        #[test]
        fn q4_with_the_dense_conditioner_moves_nothing() {
            let legacy = h3_shaped_contract(
                TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES / 4,
                ADALN_RESIDENT_Q4_BYTES,
                false,
            );
            let adopted = h3_shaped_contract(
                TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES / 4,
                ADALN_RESIDENT_Q4_BYTES,
                true,
            );
            declared_eviction(&adopted, ADALN_RESIDENT_Q4_BYTES);
            assert_eq!(
                estimate_floor_weights_bytes(&legacy, STAGED),
                TEXT_ENCODER_BYTES
            );
            assert_eq!(
                estimate_floor_weights_bytes(&adopted, STAGED),
                TEXT_ENCODER_BYTES,
                "the dense conditioning stack still binds; the DiT-side win is real but invisible \
                 here"
            );
        }

        /// Without `StagedResidency` nothing is staged out of the precompute instant: the
        /// conditioner, the transformer and the decoder are charged as ONE co-residency, and that
        /// co-residency includes the instant. The floor must not move by a byte.
        #[test]
        fn a_co_resident_composition_takes_no_reduction_at_all() {
            for (conditioning, dit, adaln) in [
                (
                    TEXT_ENCODER_BYTES,
                    DIT_BF16_BYTES,
                    ADALN_RESIDENT_BF16_BYTES,
                ),
                (
                    PACKED_TEXT_ENCODER_BYTES,
                    DIT_BF16_BYTES,
                    ADALN_RESIDENT_BF16_BYTES,
                ),
                (
                    PACKED_TEXT_ENCODER_BYTES,
                    DIT_BF16_BYTES / 4,
                    ADALN_RESIDENT_Q4_BYTES,
                ),
            ] {
                let legacy = h3_shaped_contract(conditioning, dit, adaln, false);
                let adopted = h3_shaped_contract(conditioning, dit, adaln, true);
                declared_eviction(&adopted, adaln);
                assert_eq!(
                    estimate_floor_weights_bytes(&adopted, CO_RESIDENT),
                    estimate_floor_weights_bytes(&legacy, CO_RESIDENT),
                    "co-resident floor moved for conditioning={conditioning} dit={dit}: the \
                     precompute instant is inside this charge and would be under-charged"
                );
                assert_eq!(
                    estimate_floor_weights_bytes(&adopted, CO_RESIDENT),
                    conditioning + dit + VAE_BYTES
                );
            }
        }

        /// Rung 4 windows the WHOLE transformer, sub-stack included, so the eviction must not be
        /// deducted a second time on top of it.
        #[test]
        fn rung_four_deducts_the_transformer_once_not_twice() {
            let legacy = h3_shaped_contract(
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES,
                ADALN_RESIDENT_BF16_BYTES,
                false,
            );
            let adopted = h3_shaped_contract(
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES,
                ADALN_RESIDENT_BF16_BYTES,
                true,
            );
            declared_eviction(&adopted, ADALN_RESIDENT_BF16_BYTES);
            assert_eq!(
                estimate_floor_weights_bytes(&adopted, STAGED_PLUS_RUNG4),
                estimate_floor_weights_bytes(&legacy, STAGED_PLUS_RUNG4),
            );
            assert_eq!(
                estimate_floor_weights_bytes(&adopted, STAGED_PLUS_RUNG4),
                PACKED_TEXT_ENCODER_BYTES.max(VAE_BYTES),
                "the windowed transformer leaves the floor exactly once"
            );
        }

        /// The 23 providers that never adopted the sub-stack vocabulary must be byte-identical.
        /// `steady_state_transformer_bytes()` returns `asset_facts.transformer_bytes` unchanged for
        /// them, which is the property that makes this change safe to land at the shared floor.
        #[test]
        fn a_provider_that_declares_no_sub_stack_is_byte_identical() {
            let mut contract = MemoryProviderContract::compatibility_default(
                "fixture_provider",
                MemoryBackendRealization::MlxMetal {
                    bounded_wired_residency: false,
                    lazy_or_mmap_materialization: true,
                    explicit_evaluation_and_synchronization: false,
                    cache_eviction: true,
                },
            );
            contract.asset_facts.conditioning_bytes = gib_to_bytes(1.0);
            contract.asset_facts.transformer_bytes = gib_to_bytes(5.0);
            contract.asset_facts.decoder_bytes = gib_to_bytes(2.0);
            contract.asset_facts.base_bytes = gib_to_bytes(8.0);
            assert!(contract.resident_components().is_empty());
            assert_eq!(contract.evicted_component_bytes(), 0);
            assert_eq!(
                contract.steady_state_transformer_bytes(),
                contract.asset_facts.transformer_bytes
            );
            assert_eq!(intra_transformer_evicted_bytes(&contract), 0);

            assert_eq!(
                estimate_floor_weights_bytes(&contract, STAGED),
                gib_to_bytes(7.0),
                "staged: max(conditioning, transformer + decoder)"
            );
            assert_eq!(
                estimate_floor_weights_bytes(&contract, CO_RESIDENT),
                gib_to_bytes(8.0),
                "co-resident: the whole base"
            );
            assert_eq!(
                estimate_floor_weights_bytes(&contract, STAGED_PLUS_RUNG4),
                gib_to_bytes(2.0),
                "rung 4: max(conditioning, decoder) with the transformer windowed out"
            );
        }

        /// An eviction declared on an AUXILIARY network is not inside `transformer_bytes`, so it
        /// must not be subtracted from the transformer's term. This is the distinction between
        /// `steady_state_transformer_bytes()` and `evicted_component_bytes()`, and swapping one for
        /// the other is invisible on MiniMax-H3 (whose only component is the sub-stack) — which is
        /// exactly why it is graded on a contract that has both.
        #[test]
        fn an_evicting_auxiliary_component_does_not_move_the_transformer_term() {
            const CONTROL_RESIDENT: u64 = 4_000_000_000;
            const CONTROL_RETAINED: u64 = 1_000_000_000;
            let mut contract = h3_shaped_contract(
                PACKED_TEXT_ENCODER_BYTES,
                DIT_BF16_BYTES / 4,
                ADALN_RESIDENT_Q4_BYTES,
                true,
            );
            let sub_stack_only = estimate_floor_weights_bytes(&contract, STAGED);
            let intra = intra_transformer_evicted_bytes(&contract);

            if let gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
                resident_components,
                ..
            } = &mut contract.formula
            {
                resident_components.push(gen_core::MemoryResidentComponent {
                    id: "fixture.control".to_owned(),
                    kind: gen_core::MemoryComponentKind::ControlBranch,
                    resident_bytes: CONTROL_RESIDENT,
                    bounded_by: None,
                    residency: gen_core::MemoryComponentResidency::PrecomputedThenEvicted {
                        precomputed_in: gen_core::MemoryPhase::Denoise,
                        retained_bytes: CONTROL_RETAINED,
                        evidence: "sc-19721 fixture: an evicting auxiliary network".to_owned(),
                    },
                });
            } else {
                panic!("fixture declares a ComponentPhaseEnvelope formula");
            }

            assert_eq!(
                contract.evicted_component_bytes(),
                intra + (CONTROL_RESIDENT - CONTROL_RETAINED),
                "the contract-wide accessor now sums BOTH drops"
            );
            assert_eq!(
                intra_transformer_evicted_bytes(&contract),
                intra,
                "…but only the sub-stack's drop is inside transformer_bytes"
            );
            assert_eq!(
                estimate_floor_weights_bytes(&contract, STAGED),
                sub_stack_only + CONTROL_RESIDENT,
                "the auxiliary network adds its LOAD-EXACT residency and removes nothing from the \
                 transformer term; reading evicted_component_bytes() here would take {} B off a \
                 stack that never held them",
                CONTROL_RESIDENT - CONTROL_RETAINED
            );
        }

        /// The fixture figures above are the PINNED engine's own constants, not transcriptions that
        /// can drift from it. Gated to the lanes that have a provider bundle in scope, exactly like
        /// `pinned_engine_geometry`.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        #[test]
        fn the_h3_eviction_figures_are_the_pinned_engines_own() {
            use platform_runtime::providers::minimax_h3::memory_strategy as h3;
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            use runtime_cuda as platform_runtime;
            #[cfg(target_os = "macos")]
            use runtime_macos as platform_runtime;

            assert_eq!(TEXT_ENCODER_BYTES, h3::TEXT_ENCODER_BYTES);
            assert_eq!(DIT_BF16_BYTES, h3::DIT_BF16_BYTES);
            assert_eq!(VAE_BYTES, h3::VIDEO_VAE_BYTES + h3::AUDIO_VAE_BYTES);
            assert_eq!(ADALN_RESIDENT_BF16_BYTES, h3::ADALN_EVICTED_BYTES);
            // The tier-scaled sub-stack figures are declared by the MLX engine only: the candle
            // sibling's `memory_strategy` carries the bf16 constant plus a private
            // `resolved_adaln_bytes` that scales it from the staged tier, and publishes no q4
            // `pub const` to bind to. Nothing to tie on that lane, so the tie is macOS-only.
            #[cfg(target_os = "macos")]
            assert_eq!(ADALN_RESIDENT_Q4_BYTES, h3::ADALN_EVICTED_Q4_BYTES);
            assert_eq!(
                ADALN_RETAINED_BYTES,
                h3::ADALN_MODULATION_TABLE_MAX_BYTES,
                "the retained table is NOT tier-scaled; if this constant moves, every tier cell \
                 above changes and must be re-derived rather than re-stamped"
            );
        }
    }
}
