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
// Must remain identical to mlx-gen-mage's loaded provider contract. sc-22733 (inference #953,
// epic sc-22723) retired the single string `mage-flow-mlx-shared-ladder-2026-08-03-v1` for a
// table keyed on (route, PROVEN artifact tier) — `mage-flow-<route>-<tier>-mlx-shared-ladder-v1`,
// eighteen cells — because the three tiers of one route are three different resident sets and one
// anchor cannot price all three. The engine binds a cell only when the tier is proven off the
// `quantization.bits` marker of the component directories the loader itself opened (DiT and text
// encoder agreeing) and equals `LoadSpec::quantize`, so the plan's `tier.quant` IS the tier axis.
// The worker mirrors the table here rather than calling the engine because the Resident-path
// handshake below is compiled on every platform; the macOS test
// `mage_estimator_fingerprint_matches_the_linked_provider_contract` pins every cell to the linked
// engine's `production_calibration_fingerprint`, so a move on either side is red at test time.
// The prior generation-peak token predated the shared ladder and caused every exact Mage request
// to fail the provider/gate handshake before selection.
const MAGE_CALIBRATION_FINGERPRINT_SUFFIX: &str = "-mlx-shared-ladder-v1";

/// The route label inside a Mage-Flow calibration string, mirroring
/// `mlx_gen_mage::model::MageVariant::route_label` over the six registered ids. `None` for any id
/// the engine does not serve: a plan naming one could only reach an unpaired estimator.
fn mage_calibration_route_label(engine_id: &str) -> Option<&'static str> {
    Some(match engine_id {
        "mage_flow" => "mage-flow",
        "mage_flow_base" => "mage-flow-base",
        "mage_flow_turbo" => "mage-flow-turbo",
        "mage_flow_edit" => "mage-flow-edit",
        "mage_flow_edit_base" => "mage-flow-edit-base",
        "mage_flow_edit_turbo" => "mage-flow-edit-turbo",
        _ => return None,
    })
}

/// The tier label inside a Mage-Flow calibration string, mirroring
/// `mlx_gen_mage::model::calibration_tier_label`. `None` for a tier the family does not ship, so an
/// unshipped tier is unnameable rather than collapsed onto a neighbour.
fn mage_calibration_tier_label(quant: Option<gen_core::Quant>) -> Option<&'static str> {
    match quant {
        None => Some("bf16"),
        Some(gen_core::Quant::Q4) => Some("q4"),
        Some(gen_core::Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

/// The production calibration identity the loaded `mlx-gen-mage` route publishes for one
/// (route, tier) cell, or `None` when no cell exists for the pair.
fn mage_calibration_fingerprint(engine_id: &str, quant: Option<gen_core::Quant>) -> Option<String> {
    let route = mage_calibration_route_label(engine_id)?;
    let tier = mage_calibration_tier_label(quant)?;
    Some(format!(
        "mage-flow-{route}-{tier}{MAGE_CALIBRATION_FINGERPRINT_SUFFIX}"
    ))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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
struct DecodeQualityGeometryWire {
    width: u32,
    height: u32,
    batch: u32,
    frames: u32,
    reference_count: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DecodeQualityArtifactWire {
    repository: String,
    revision: String,
    variant: String,
    fingerprint: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum DecodeQualityDispositionWire {
    Admitted,
    Refused { reason: String },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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

impl DecodeGeometryPolicyWire {
    fn into_core(self, model_id: &str) -> WorkerResult<MemoryDecodeGeometryPolicy> {
        if self.resolved_route != model_id {
            return Err(WorkerError::InvalidPayload(format!(
                "decode-quality policy route {:?} cannot authorize model {:?}",
                self.resolved_route, model_id
            )));
        }
        let backend = match self.backend.as_str() {
            "mlx" => MemoryBackend::Mlx,
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "MLX decode-quality policy cannot use backend {other:?}"
                )))
            }
        };
        let tier = match self.tier.as_str() {
            "bf16" => MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            "q4" => MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
            "q8" => MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q8),
                component_precision_floors: &[],
            },
            "nvfp4" => MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Nvfp4),
                component_precision_floors: &[],
            },
            "fp32" => MemoryNumericTier {
                precision: Precision::Fp32,
                quant: None,
                component_precision_floors: &[],
            },
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "unsupported decode-quality tier {other:?}"
                )))
            }
        };
        let load_shape = match self.load_shape.as_str() {
            "eager_materialization" => gen_core::LoadShape::EagerMaterialization,
            "deferred_materialization" => gen_core::LoadShape::DeferredMaterialization,
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "unsupported decode-quality load shape {other:?}"
                )))
            }
        };
        let mode = match self.mode.as_str() {
            "text_to_image" => MemoryMode::TextToImage,
            "image_to_image" => MemoryMode::ImageToImage,
            "edit" => MemoryMode::Edit,
            other => MemoryMode::Other(other.to_owned()),
        };
        let disposition = match self.disposition {
            DecodeQualityDispositionWire::Admitted => MemoryDecodeQualityDisposition::Admitted,
            DecodeQualityDispositionWire::Refused { reason } => {
                MemoryDecodeQualityDisposition::Refused { reason }
            }
        };
        Ok(MemoryDecodeGeometryPolicy {
            quality_abi: self.quality_abi,
            family: self.family,
            resolved_route: self.resolved_route,
            backend,
            tier,
            load_shape,
            artifact: MemoryDecodeArtifactIdentity {
                repository: self.artifact.repository,
                revision: self.artifact.revision,
                variant: self.artifact.variant,
                fingerprint: self.artifact.fingerprint,
            },
            implementation_fingerprint: self.implementation_fingerprint,
            mode,
            overlay: self.overlay,
            geometry: MemoryGeometry {
                width: self.geometry.width,
                height: self.geometry.height,
                batch: self.geometry.batch,
                frames: self.geometry.frames,
                reference_count: self.geometry.reference_count,
            },
            use_pid: self.use_pid,
            tile_edge: self.tile_edge,
            overlap: self.overlap,
            metric: self.metric,
            maximum_error: self.maximum_error,
            fixtures: self
                .fixtures
                .into_iter()
                .map(|fixture| MemoryDecodeQualityFixture {
                    seed: fixture.seed,
                    production_latent_provenance: fixture.production_latent_provenance,
                    production_latent_sha256: fixture.production_latent_sha256,
                    dense_output_sha256: fixture.dense_output_sha256,
                    tiled_output_sha256: fixture.tiled_output_sha256,
                    observed_error: fixture.observed_error,
                })
                .collect(),
            production_evidence_sha256: self.production_evidence_sha256,
            disposition,
        })
    }
}

/// Typed load-time disposition for a packaged quality table. A refused binding removes semantic
/// decode authority only; dense decode and independent higher memory rungs remain available.
#[derive(Clone, Debug, PartialEq, Eq)]
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

pub(crate) fn manifest_declares_decode_quality_policies(
    manifest: &JsonObject<String, Value>,
) -> bool {
    manifest
        .get("mlx")
        .and_then(Value::as_object)
        .and_then(|mlx| mlx.get("memoryStrategyContract"))
        .and_then(|contract| contract.get("implementations"))
        .and_then(Value::as_array)
        .is_some_and(|implementations| {
            implementations.iter().any(|implementation| {
                implementation
                    .get("parameterRanges")
                    .and_then(|ranges| ranges.get("decodeGeometryPolicies"))
                    .and_then(Value::as_array)
                    .is_some_and(|rows| !rows.is_empty())
            })
        })
}

/// Read the semantic P9 table from the model's derived static contract. This does not read or
/// mutate the performance/calibration bundle: quality rows have their own ABI, hashes, and runtime
/// path. Missing policy preserves the historical provider contract and its estimated fallback.
pub(crate) fn decode_quality_policies_from_manifest(
    manifest: &JsonObject<String, Value>,
    model_id: &str,
) -> WorkerResult<Vec<MemoryDecodeGeometryPolicy>> {
    let Some(contract) = manifest
        .get("mlx")
        .and_then(Value::as_object)
        .and_then(|mlx| mlx.get("memoryStrategyContract"))
    else {
        return Ok(Vec::new());
    };
    let implementations = contract
        .get("implementations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "mlx.memoryStrategyContract.implementations must be an array".to_owned(),
            )
        })?;
    let mut policies = Vec::new();
    for implementation in implementations {
        let Some(ranges) = implementation.get("parameterRanges") else {
            continue;
        };
        let Some(rows) = ranges.get("decodeGeometryPolicies") else {
            continue;
        };
        if implementation.get("rung").and_then(Value::as_str) != Some("bounded_decode") {
            return Err(WorkerError::InvalidPayload(
                "decodeGeometryPolicies may be declared only by bounded_decode".to_owned(),
            ));
        }
        let rows = rows.as_array().ok_or_else(|| {
            WorkerError::InvalidPayload("decodeGeometryPolicies must be an array".to_owned())
        })?;
        for row in rows {
            let wire: DecodeGeometryPolicyWire =
                serde_json::from_value(row.clone()).map_err(|error| {
                    WorkerError::InvalidPayload(format!("invalid decode-quality policy: {error}"))
                })?;
            policies.push(wire.into_core(model_id)?);
        }
    }
    Ok(policies)
}

/// Bind packaged semantic rows to provenance resolved independently beside the exact weights tree.
/// An unavailable or mismatched identity refuses only the optimized decode table and is emitted as
/// a typed audit decision; it never turns a valid model install into a dense-decode load failure.
pub(crate) fn bind_decode_quality_policies_from_manifest(
    manifest: &JsonObject<String, Value>,
    model_id: &str,
    resolved_artifact: Option<&ResolvedArtifactProvenance>,
) -> WorkerResult<DecodeQualityPolicyBinding> {
    let policies = decode_quality_policies_from_manifest(manifest, model_id)?;
    if policies.is_empty() {
        return Ok(DecodeQualityPolicyBinding {
            policies,
            authoritative: false,
            runtime_identity: None,
            decision: DecodeQualityBindingDecision::Legacy,
        });
    }
    let Some(resolved_artifact) = resolved_artifact else {
        return Ok(DecodeQualityPolicyBinding {
            policies: Vec::new(),
            authoritative: true,
            runtime_identity: None,
            decision: DecodeQualityBindingDecision::Refused {
                reason: "decode-quality table has no independently resolved artifact identity"
                    .to_owned(),
            },
        });
    };
    let artifact = MemoryDecodeArtifactIdentity {
        repository: resolved_artifact.identity.repository.clone(),
        revision: resolved_artifact.identity.revision.clone(),
        variant: resolved_artifact.identity.variant.clone(),
        fingerprint: resolved_artifact.identity.fingerprint.clone(),
    };
    if let Some(foreign) = policies.iter().find(|policy| policy.artifact != artifact) {
        return Ok(DecodeQualityPolicyBinding {
            policies: Vec::new(),
            authoritative: true,
            runtime_identity: None,
            decision: DecodeQualityBindingDecision::Refused {
                reason: format!(
                    "decode-quality artifact {:?}@{}:{} ({}) does not match resolved artifact {:?}@{}:{} ({})",
                    foreign.artifact.repository,
                    foreign.artifact.revision,
                    foreign.artifact.variant,
                    foreign.artifact.fingerprint,
                    artifact.repository,
                    artifact.revision,
                    artifact.variant,
                    artifact.fingerprint,
                ),
            },
        });
    }
    let runtime_identity = MemoryDecodeQualityRuntimeIdentity {
        artifact: artifact.clone(),
    };
    let decision = DecodeQualityBindingDecision::Bound {
        artifact,
        row_count: policies.len(),
    };
    Ok(DecodeQualityPolicyBinding {
        policies,
        authoritative: true,
        runtime_identity: Some(runtime_identity),
        decision,
    })
}

pub(crate) fn attach_decode_quality_binding(
    mut spec: LoadSpec,
    binding: DecodeQualityPolicyBinding,
    route: &str,
) -> LoadSpec {
    tracing::info!(
        event = "mlx_decode_quality_binding",
        route,
        decision = ?binding.decision,
        "resolved semantic decode-quality load binding"
    );
    spec = spec
        .with_decode_geometry_policy_authority(binding.authoritative)
        .with_decode_geometry_policies(binding.policies);
    if let Some(identity) = binding.runtime_identity {
        spec = spec.with_decode_quality_runtime_identity(identity);
    }
    spec
}

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
    /// Terminal declaration refusals retain safe eager execution, but may not consume any optimized
    /// provider strategy later in the cache-aware selector. NotEvaluated preserves every legacy
    /// route; Applied preserves the exact declaration-owned provider contract.
    load_shape_declaration_result: LoadShapeDeclarationResult,
}

impl MlxRequestPlan {
    /// Bind the already-resolved artifact tier without reinterpreting `LoadSpec::quantize`.
    /// Prepacked Krea q4/q8 artifacts intentionally load with `quantize=None`; this explicit axis is
    /// therefore required for request/evidence identity even when no calibration provenance exists.
    pub(crate) fn with_resolved_artifact_tier(
        mut self,
        resolved_tier: Option<&str>,
    ) -> WorkerResult<Self> {
        let Some(resolved_tier) = resolved_tier else {
            return Err(WorkerError::InvalidPayload(format!(
                "{} has no resolved MLX artifact tier",
                self.model_id
            )));
        };
        self.tier = numeric_tier_for_resolved(
            resolved_tier,
            self.tier,
            declared_component_floors(self.engine_id),
        )
        .map_err(|reason| WorkerError::InvalidPayload(format!("{}: {reason}", self.model_id)))?;
        Ok(self)
    }

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

    /// Prepared-spec planner used by imported single-file routes. Provider queries and token
    /// validation are fallible contract operations here: failures propagate to the job instead of
    /// being silently converted into an unmeasured/raw-restat estimate.
    pub(crate) fn try_for_spec_and_manifest(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
    ) -> WorkerResult<Self> {
        let media = crate::inference_runtime::media();
        let provider_footprint = media
            .footprint(engine_id, spec)
            .map_err(|error| crate::classify_engine_error("MLX footprint query failed", error))?;
        let activation_anchor_bytes =
            media
                .activation_memory_bytes_1024(engine_id)
                .map_err(|error| {
                    crate::classify_engine_error("MLX activation-memory query failed", error)
                })?;
        let (asset_bytes, _, headroom) = spec_component_bytes_with_provider_footprint_checked(
            engine_id,
            spec,
            provider_footprint,
        )
        .map_err(|error| {
            crate::classify_engine_error("MLX prepared source accounting failed", error)
        })?;
        let file_sizes = prepared_file_sizes(spec).map_err(|error| {
            crate::classify_engine_error("MLX prepared source validation failed", error)
        })?;
        let folded_control_bytes = spec
            .control
            .as_ref()
            .map_or(0, |source| prepared_source_bytes(source, &file_sizes));
        let folded_adapter_source_bytes = spec.adapters.iter().fold(0_u64, |total, adapter| {
            total.saturating_add(prepared_source_bytes(
                &WeightsSource::File(adapter.path.clone()),
                &file_sizes,
            ))
        });
        let folded_adapter_bytes =
            adapter_resident_source_bytes(engine_id, spec, folded_adapter_source_bytes);
        Ok(Self::for_spec_and_manifest_with_accounting(
            engine_id,
            model_id,
            spec,
            manifest,
            resolved_artifact,
            asset_bytes,
            folded_control_bytes,
            folded_adapter_bytes,
            headroom,
            activation_anchor_bytes,
        ))
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
        Self::for_spec_and_manifest_with_accounting(
            engine_id,
            model_id,
            spec,
            manifest,
            resolved_artifact,
            asset_bytes,
            folded_control_bytes,
            folded_adapter_bytes,
            headroom,
            activation_anchor_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_spec_and_manifest_with_accounting(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
        asset_bytes: u64,
        folded_control_bytes: u64,
        folded_adapter_bytes: u64,
        headroom: HeadroomAllowance,
        activation_anchor_bytes: Option<u64>,
    ) -> Self {
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
            load_shape_declaration_result: spec.load_shape_declaration_result,
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
        let baseline = fixed_reserve_bytes.saturating_add(
            (area_bytes as f64 * request_scale)
                .round()
                .clamp(0.0, u64::MAX as f64) as u64,
        );
        let extra =
            flux3_cfg_additional_workspaces(self.engine_id, geometry.width, geometry.height);
        baseline.saturating_add(extra.0.max(extra.1))
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

#[cfg(native_lane)]
fn declared_component_floors(engine_id: &str) -> &'static [gen_core::ComponentPrecisionFloor] {
    crate::inference_runtime::media_descriptor(engine_id)
        .map(|descriptor| descriptor.capabilities.component_precision_floors)
        .unwrap_or(&[])
}

#[cfg(not(native_lane))]
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
        // `EvidenceMismatchReason::LoadShape` and the legacy selector rather than rejecting the opt-in.
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
    sceneworks_core::memory_contracts::active_component_floors(declared, selected)
}

pub(crate) fn plan_tier_key(tier: MemoryNumericTier) -> &'static str {
    match tier.quant {
        Some(gen_core::Quant::Q4) => "q4",
        Some(gen_core::Quant::Q8) => "q8",
        Some(gen_core::Quant::Nvfp4) => "nvfp4",
        Some(gen_core::Quant::Fp8Rowwise) => "fp8r",
        None if tier.precision == gen_core::Precision::Fp32 => "fp32",
        None => "bf16",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmissionPath {
    Evidence,
    Legacy,
}

/// Why a request left the Evidence path for the legacy/estimate selector. Every reason is a
/// structural or identity fact about the packaged evidence and the install; none is measurement
/// currency (sc-22738) — a binding whose provider closure has moved since capture is routed exactly
/// as one whose closure has not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
struct ReferencePricing<'a> {
    spec: &'a LoadSpec,
    registry: &'a gen_core::ProviderRegistry,
}

fn missing_reference_pricing(engine_id: &str) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{engine_id}: owned reference resources require the reference-aware memory admission entry point and its resolved LoadSpec; unpriced references cannot be admitted"
    ))
}

pub(crate) fn clip_window_upper_bound(prompt: &str, negative: &str) -> u32 {
    // Byte-level BPE can only merge the lowercased UTF-8 bytes. BOS/EOS leave 75 content
    // tokens per window. Count both CFG rows at their padded maximum, never sum sequential jobs.
    let bytes = prompt
        .to_lowercase()
        .len()
        .max(negative.to_lowercase().len());
    u32::try_from(bytes.div_ceil(75).max(1)).unwrap_or(u32::MAX)
}

/// Count the same untruncated CLIP tokens the loaded SDXL provider will window. The byte bound
/// remains a conservative fallback when tokenizer assets cannot be read during preparation.
#[cfg(target_os = "macos")]
pub(crate) fn clip_windows_for_spec(
    engine_id: &str,
    spec: &LoadSpec,
    prompt: &str,
    negative: &str,
) -> u32 {
    if engine_id == "sdxl" {
        use runtime_macos::providers::sdxl::{load_tokenizer, LDM_TOKENIZER_COMPONENT};
        let root = match &spec.weights {
            WeightsSource::Dir(root) => Some(root),
            WeightsSource::File(_) => match spec.components.get(LDM_TOKENIZER_COMPONENT) {
                Some(WeightsSource::Dir(root)) => Some(root),
                _ => None,
            },
        };
        let count = root
            .and_then(|root| load_tokenizer(root).ok())
            .and_then(|tokenizer| {
                let positive = tokenizer.tokenize(prompt).ok()?;
                let negative = tokenizer.tokenize(negative).ok()?;
                let tokens = positive.len().max(negative.len()).saturating_sub(2);
                u32::try_from(tokens.div_ceil(75).max(1)).ok()
            });
        if let Some(windows) = count {
            return windows;
        }
    }
    clip_window_upper_bound(prompt, negative)
}

#[derive(Clone, Debug)]
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

/// Frozen FLUX3 sampling.denoise always packs [unconditional, conditional], including scales
/// 0 and 1; the distilled flow map packs one. This is NOT output/job count or decoder batch.
pub(crate) fn flux3_conditioning_forward_batch(engine_id: &str) -> u64 {
    if engine_id == "flux3_klein_base" {
        2
    } else {
        1
    }
}

/// Live hidden-width (3072) activation buffers of ONE FLUX3 `Block::forward` (pinned BFL
/// `dit.py` / native `dit.rs`), per token, counted from the tensors alive at its widest point
/// with NO intra-block reuse assumed: x 1, modulated x_mod 1, q/k/v projections 3, their
/// contiguous head-split copies 3, f32 RMSNorm+RoPE q/k 2, attention output 1, mlp_in
/// (2 x mlp_ratio 3) 6, silu(gate) 3, silu(gate)*up 3, mlp_out 1, attn_out 1, delta 1,
/// gate*delta 1, residual 1 = 28. Blocks run sequentially, so only one block's set is live.
pub(crate) const FLUX3_BLOCK_LIVE_HIDDEN_BUFFERS: u64 = 28;

/// Source-derived retained conditioning/DiT allowances, NOT measurements. The source clamps text
/// at 4096, stacks 8 x 2560 encoder channels, uses 3072-wide blocks and 128-channel packed
/// latents. Each active row carries [`FLUX3_BLOCK_LIVE_HIDDEN_BUFFERS`] at f32 width even though
/// most are bf16 — a deliberate PROVISIONAL over-estimate (no buffer reuse, f32 everywhere) until
/// the epic-end measured anchor (sc-26243) replaces it; it is bounded by the source live set, not
/// a guessed multiplier, so it does not refuse renders the live set could fit. Attention scratch
/// and allocator reserve remain in the existing generic headroom; no weights, decoder, job count
/// or optional PUP copies are added here.
pub(crate) fn flux3_source_workspaces(engine_id: &str, width: u32, height: u32) -> (u64, u64) {
    if !matches!(engine_id, "flux3_klein" | "flux3_klein_base") {
        return (0, 0);
    }
    let rows = flux3_conditioning_forward_batch(engine_id);
    let image = u64::from(width.div_ceil(16)).saturating_mul(u64::from(height.div_ceil(16)));
    let conditioning = rows.saturating_mul(4096 * 20480 * 4);
    let projections = image
        .saturating_add(4096)
        .saturating_mul(3072 * 4 * FLUX3_BLOCK_LIVE_HIDDEN_BUFFERS)
        .saturating_mul(rows);
    let cfg_merge = if rows == 2 {
        image.saturating_mul(128 * 4 * 5)
    } else {
        0
    };
    (
        conditioning,
        projections
            .saturating_add(conditioning)
            .saturating_add(cfg_merge),
    )
}

fn flux3_cfg_additional_workspaces(engine_id: &str, width: u32, height: u32) -> (u64, u64) {
    if engine_id != "flux3_klein_base" {
        return (0, 0);
    }
    let base = flux3_source_workspaces(engine_id, width, height);
    let distilled = flux3_source_workspaces("flux3_klein", width, height);
    (
        base.0.saturating_sub(distilled.0),
        base.1.saturating_sub(distilled.1),
    )
}

fn flux3_cfg_workspace(
    engine_id: &str,
    geometry: MemoryGeometry,
    generic: u64,
) -> crate::mlx_phase_estimate::Workspace {
    let extra = flux3_cfg_additional_workspaces(engine_id, geometry.width, geometry.height);
    crate::mlx_phase_estimate::Workspace {
        conditioning: generic.saturating_add(extra.0),
        denoise: generic.saturating_add(extra.1),
        decode: generic,
    }
}

fn gib_to_bytes(gib: f64) -> u64 {
    sceneworks_core::memory_anchor::gib_to_bytes(gib)
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
/// A provider that renders a genuine batched pass threads THAT pass's batch size here instead of
/// the job count: the native prompt batch (sc-25681, Iris-3B `generate(prompts=[…])`) carries its
/// `prompts × count` images in [`MlxRequestInputs::native_batch`]. All of them are charged, not
/// just one `count` repetition's prompt rows: the engine renders the repetitions inside the same
/// provider call and keeps every finished image live until it returns, and no batched run has been
/// measured (sc-25684 owns that), so the declared estimate scales the whole activation term by the
/// image count — conservative by construction, never a batch admitted on a one-image budget.
fn request_batch(inputs: &MlxRequestInputs) -> u32 {
    inputs.native_batch.max(1)
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
        // sc-20799 round 2: unlike the video lane's reference-shape `"other"`, this `"other"` is
        // IDENTITY-BEARING and must stay. `mode_key` is matched against `binding.mode` when
        // resolving packaged evidence and is round-tripped by `memory_mode_from_mode_key`, whose
        // catch-all arm reconstructs `MemoryMode::Other("other")`. Renaming it (to "none" or
        // anything else) would silently repoint every non-standard image mode at a different
        // admission coordinate. The real asymmetry worth knowing about is that the typed value
        // keeps the concrete mode while the key collapses to a single bucket, so two distinct
        // non-standard modes share one evidence key; narrowing that is a calibration-corpus change,
        // not a rename, and is out of scope for this pass.
        _ => (MemoryMode::Other(mode.to_owned()), "other"),
    }
}

/// Provider-facing behavior identity for an exact routed request.
///
/// Krea Raw's public reference toggle deliberately remains the catalog's `image_generation`
/// coordinate, but the provider executes that one-reference request through its latent-init
/// image-to-image entrypoint. Keep that internal contract identity exact without inventing a new
/// public route or a generic CFG axis.
fn provider_request_mode(engine_id: &str, inputs: &MlxRequestInputs) -> (MemoryMode, &'static str) {
    if (engine_id == "flux1_dev_control"
        && matches!(
            inputs.mode.as_str(),
            "image_generation" | "text_to_image" | "style_variations" | "character_image"
        ))
        || (engine_id == "flux1_dev" && inputs.mode == "character_image")
    {
        (MemoryMode::ImageToImage, "image_to_image")
    } else if matches!(engine_id, "flux1_schnell" | "flux1_dev")
        && matches!(
            inputs.mode.as_str(),
            "image_generation" | "text_to_image" | "style_variations"
        )
    {
        (MemoryMode::TextToImage, "text_to_image")
    } else if matches!(engine_id, "flux2_klein_9b_edit" | "flux2_klein_9b_kv_edit")
        && (1..=8).contains(&inputs.reference_count)
    {
        (MemoryMode::Edit, "edit")
    } else if (engine_id == "flux2_klein_9b" && inputs.has_reference && inputs.reference_count == 1)
        || is_krea_base_dit_reference_route(engine_id, inputs)
    {
        (MemoryMode::ImageToImage, "image_to_image")
    } else {
        request_mode(&inputs.mode)
    }
}

fn is_krea_base_dit_provider(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Krea)
}

fn is_flux2_klein_provider(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Flux2Klein)
}

fn is_flux1_provider(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Flux1)
}

/// Provider-owned request overlay derived from the complete load identity.
///
/// FLUX.2 Klein authenticates every load-time composition at request scope, including conservative
/// eager requests whose BTR declaration refused the shape. Mirror the provider's closed axis order
/// from the actual `LoadSpec`; public labels such as `lora` and `references:1` are not evidence
/// identities. Krea's base-DiT family deliberately carries the same features on typed request/load
/// axes and always uses a provider overlay of None.
pub(crate) fn provider_overlay_for_load_spec(
    engine_id: &str,
    spec: &LoadSpec,
    public_overlay: Option<String>,
) -> Option<String> {
    if is_krea_base_dit_provider(engine_id) {
        return None;
    }
    if is_flux1_provider(engine_id) {
        let mut axes = Vec::new();
        if engine_id == "flux1_dev_control" || spec.control.is_some() {
            axes.push("control");
        }
        if !spec.extra_controls.is_empty() {
            axes.push("extra-controls");
        }
        if !spec.adapters.is_empty() {
            axes.push("adapters");
        }
        if spec.ip_adapter.is_some() {
            axes.push("ip-adapter");
        }
        if spec.pid.is_some() {
            axes.push("pid");
        }
        if spec.identity.is_some() {
            axes.push("identity");
        }
        if spec.text_encoder.is_some() {
            axes.push("external-text-encoder");
        }
        return (!axes.is_empty()).then(|| axes.join("-"));
    }
    if !is_flux2_klein_provider(engine_id) {
        return public_overlay;
    }
    let mut axes = Vec::new();
    if !spec.adapters.is_empty() {
        axes.push("adapters");
    }
    if spec.control.is_some() {
        axes.push("control");
    }
    if !spec.extra_controls.is_empty() {
        axes.push("extra-controls");
    }
    if spec.ip_adapter.is_some() {
        axes.push("ip-adapter");
    }
    if spec.identity.is_some() {
        axes.push("identity");
    }
    if spec.text_encoder.is_some() {
        axes.push("external-text-encoder");
    }
    if !spec.components.is_empty() {
        axes.push("components");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

fn is_krea_base_dit_reference_route(engine_id: &str, inputs: &MlxRequestInputs) -> bool {
    matches!(engine_id, "krea_2_raw" | "krea_2_turbo")
        && matches!(inputs.mode.as_str(), "image_generation" | "text_to_image")
        && inputs.has_reference
        && inputs.reference_count == 1
}

/// Enforce provider request identity after production has derived any load-dependent overlay from
/// the complete `LoadSpec`. Krea always normalizes to None. Klein inputs already carry the exact
/// provider overlay from [`provider_overlay_for_load_spec`]; reconstructing it here from only an
/// adapter count would erase text-encoder and component axes on safe eager fallback routes.
fn provider_request_inputs<'a>(
    engine_id: &str,
    inputs: &'a MlxRequestInputs,
) -> Cow<'a, MlxRequestInputs> {
    if !is_krea_base_dit_provider(engine_id) {
        return Cow::Borrowed(inputs);
    }
    if inputs.overlay.is_some() {
        let mut normalized = inputs.clone();
        normalized.overlay = None;
        Cow::Owned(normalized)
    } else {
        Cow::Borrowed(inputs)
    }
}

/// Translate a selection into the engine's per-rung engagement knobs.
///
/// SC-15805: this asks the contract which rungs the selection ENGAGES rather than re-deriving the
/// answer from `MemoryStrategy`'s numeric order. The order is a cost ordering, and the cumulative
/// default it expresses is defeasible — a rung the provider does not implement is not engaged, so a
/// provider publishing a verified cheaper composition no longer has its unimplemented rung's knob
/// switched on underneath it.
pub(crate) fn memory_for_selection(
    contract: &MemoryProviderContract,
    selection: MemorySelection,
) -> GenerationMemory {
    contract.generation_memory(&selection).unwrap_or_default()
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
    let evidence = sceneworks_core::memory_contracts::unverified_estimate(
        MemoryEvidenceKey {
            model_family: contract.provider_id.clone(),
            resolved_route: contract.provider_id.clone(),
            backend: gen_core::MemoryBackend::Mlx,
            tier,
            load_shape: contract.load_shape,
            mode: memory_mode_from_mode_key(mode),
            reference_shape: if geometry.reference_count == 0 {
                MemoryReferenceShape::None
            } else {
                MemoryReferenceShape::Image
            },
            overlay: overlay.map(str::to_owned),
            geometry,
            frames_per_second: None,
            strategy: selection.strategy,
            engaged_composition: contract.engaged_composition_for_selection(&selection),
            parameters: selection.parameters,
        },
        gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint.unwrap_or_default().to_owned(),
        REQUEST_EVIDENCE_REVISION.to_owned(),
        predicted_peak_bytes,
    );
    (selection, evidence)
}

fn mismatch_fallback_reason(reason: EvidenceMismatchReason) -> LegacyAdmissionReason {
    if reason == EvidenceMismatchReason::CalibrationFingerprint {
        LegacyAdmissionReason::FingerprintMismatch
    } else {
        LegacyAdmissionReason::IdentityMismatch
    }
}

fn stronger_fallback_reason(
    current: LegacyAdmissionReason,
    candidate: LegacyAdmissionReason,
) -> LegacyAdmissionReason {
    let priority = |reason| match reason {
        LegacyAdmissionReason::NoRecord => 0,
        LegacyAdmissionReason::OutOfEnvelope => 1,
        LegacyAdmissionReason::FingerprintMismatch => 2,
        LegacyAdmissionReason::IdentityMismatch => 3,
        LegacyAdmissionReason::PackagedEmpty
        | LegacyAdmissionReason::NoBinding
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
        stage_residency: None,
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
///
/// The packaged bundle is served whatever its version stamps say (sc-22738): a bundle that parses
/// and validates is the evidence, and there is no `StaleBundle` fallback any more — that arm
/// demoted every MLX request to the legacy estimate on a harness-version drift, which is a
/// re-capture signal for the probe tooling and not a runtime input.
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
    let bundle = sceneworks_core::memory_calibration::load_packaged_bundle().map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "packaged memory-calibration evidence is invalid: {error}"
        ))
    })?;
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
    // ARTIFACT identity only (sc-18096, sc-22738). The filter once also required
    // `binding.query.inference_closure_digest == <live closure>` — a pre-demotion that routed a
    // binding whose provider closure had moved to `AdmissionPath::Legacy` before any candidate
    // reached `select_strategy`; sc-18096 retired the pre-demotion but kept grading such a
    // candidate behind a widened "stale" margin, and sc-22738 retired the margin too. A binding
    // now proceeds and is graded at its measured peak regardless of the closure it was measured
    // under: currency is a re-capture signal for the probe tooling, never a runtime input.
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
            fallback_reason: Some(LegacyAdmissionReason::IdentityMismatch),
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
            transformer_variant: None,
            decoder: None,
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
                        model_family: plan.model_id.to_owned(),
                        resolved_route: plan.engine_id.to_owned(),
                        backend: gen_core::MemoryBackend::Mlx,
                        tier: plan.tier,
                        load_shape: crate::memory_strategy::load_shape_from_receipt(
                            record.load_shape,
                        ),
                        mode: memory_mode_from_mode_key(mode_key),
                        reference_shape: if inputs.reference_count == 0 {
                            MemoryReferenceShape::None
                        } else {
                            MemoryReferenceShape::Image
                        },
                        overlay: inputs.overlay.clone(),
                        geometry: request_geometry(inputs),
                        frames_per_second: None,
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
                evidence.push(VerifiedAdmissionCandidate {
                    evidence: memory_evidence,
                    foreign_reserve_bytes,
                    minimum_host_bytes: envelope.required_host_bytes(),
                    record_id: record.id.clone(),
                });
            }
            EvidenceVerdict::Unknown => {}
            EvidenceVerdict::OutOfEnvelope => {
                fallback_reason =
                    stronger_fallback_reason(fallback_reason, LegacyAdmissionReason::OutOfEnvelope);
            }
            EvidenceVerdict::Mismatch(reason) => {
                fallback_reason =
                    stronger_fallback_reason(fallback_reason, mismatch_fallback_reason(reason));
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
            ),
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
/// synthesis time. See [`MeasuredRungBasis`] for why the binding's closure is not a conjunct.
fn collect_estimate_bases(
    bundle: &EvidenceBundle,
    plan: &MlxRequestPlan,
    identity_matches: &[&MlxCalibrationBinding],
    mode_key: &str,
    overlay: &str,
    request_cell_geometry: CalibrationGeometry,
) -> Vec<MeasuredRungBasis> {
    identity_matches
        .iter()
        .filter(|binding| {
            binding.mode == mode_key
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
                transformer_variant: None,
                decoder: None,
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

/// The anchor store the image-MLX derivation reads. Production has exactly one — the packaged,
/// validated store; the `cfg(test)` override exists for the same reason as
/// `vram_gate::krea_anchor_store`'s: currency is the PACKAGED loader-closure declaration, which
/// no argument can reach, so tests that grade the derivation itself inject a store stamped at the
/// live declared digest.
fn mlx_image_anchor_store() -> Option<&'static sceneworks_core::memory_anchor::MemoryAnchorStore> {
    #[cfg(test)]
    if let Some(store) = tests::injected_image_anchor_store() {
        return Some(store);
    }
    sceneworks_core::memory_anchor::packaged_memory_anchors()
}

/// The anchor-derived admission peak for one image-MLX request (epic 22505 feature-end fix round,
/// E2/E7): the packaged measured anchor for this `(model, tier, mlx lane)` priced through
/// `MemoryAnchor::derive_mlx_image_phase_peaks`. The returned peak contains active allocations;
/// the selector adds recapture uncertainty separately.
///
/// Carries the FULL guard set the other anchor consumers carry, every conjunct fail-open to the
/// caller's floor:
///
/// * IDENTITY — model (the store lookup key), tier, backend lane (the lookup + the law), provider
///   (the loaded contract's id), route (the plan's engine), mode, and the pipeline axes (the law
///   refuses an anchor carrying any).
/// * REGIME — the contract's materialization shape must equal the anchor's, and the law itself
///   requires the anchor to be the eager unbounded resident composition (which upper-bounds every
///   optimized composition, so no per-rung regime conjunct is needed on the request side).
/// * OVERLAY / REFERENCES — the anchors were measured overlay-free with zero references on a
///   single frame at batch 1; a differently-conditioned surface keeps its floor.
/// * NOT currency (sc-22738) — the anchor's loader-closure digest is a re-capture signal for the
///   probe tooling; the anchor prices this request whether or not its loader has moved.
fn mlx_image_anchor_derived_peak(
    contract: &MemoryProviderContract,
    plan: &MlxRequestPlan,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
) -> Option<(u64, String)> {
    use sceneworks_core::memory_anchor::AnchorMlxImageDeriveRequest;

    let anchor = mlx_image_anchor_match(contract, plan, mode_key, overlay, geometry)?;
    let derived = anchor.derive_mlx_image_phase_peaks(
        AnchorMlxImageDeriveRequest {
            width: geometry.width,
            height: geometry.height,
        },
        crate::video_admission::anchor_component_bytes(contract.asset_facts),
    )?;
    Some((derived.peak_bytes(), anchor.id.clone()))
}

/// The measured image anchor this request may borrow from, or `None` — the identity, overlay,
/// reference, geometry-shape and currency conjuncts of the doc comment on
/// [`mlx_image_anchor_derived_peak`], factored out (sc-22665) so the anchor-derived arm and the
/// estimate floor's derived activation residue below grade the SAME anchor against the SAME
/// guards. A second inline copy is how the two arms would silently disagree about whether the
/// evidence covers the request.
///
/// The REGIME conjuncts are deliberately not here: each consumer's own law entry point states what
/// it needs of the anchor's measured regime (the lane shim wants the eager unbounded resident
/// composition; the residue path lets the law's own regime guard decide), and hoisting the
/// stricter of the two here would silently narrow the other.
fn mlx_image_anchor_match(
    contract: &MemoryProviderContract,
    plan: &MlxRequestPlan,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
) -> Option<&'static sceneworks_core::memory_anchor::MemoryAnchor> {
    use sceneworks_core::memory_anchor::{AnchorBackend, AnchorLoadShape};

    if overlay.is_some()
        || geometry.reference_count != 0
        || geometry.frames != 1
        || geometry.batch > 1
    {
        return None;
    }
    let store = mlx_image_anchor_store()?;
    let anchor =
        store.image_anchor_for(&plan.model_id, AnchorBackend::Mlx, plan_tier_key(plan.tier))?;
    let anchor_load_shape = match anchor.load_shape {
        AnchorLoadShape::EagerMaterialization => gen_core::LoadShape::EagerMaterialization,
        AnchorLoadShape::DeferredMaterialization => gen_core::LoadShape::DeferredMaterialization,
    };
    if anchor.route != plan.engine_id
        || anchor.provider != contract.provider_id
        || anchor.mode != mode_key
        || anchor.overlay.is_some()
        || anchor.reference_count != 0
        || anchor.underived_reason.is_some()
        || anchor_load_shape != contract.load_shape
    {
        return None;
    }
    Some(anchor)
}

/// The regime the graded candidate executes in, in the derivation law's vocabulary (sc-22665):
/// the engaged composition says which rungs are on, the selection's own parameters say how deeply
/// each one bounds. A rung that is not engaged contributes `None`, which is what makes the law
/// price it unbounded.
///
/// The parameters are the ones the candidate will actually run with — on the estimate floor those
/// are `estimate_floor_parameters`' most deeply bounding declared values, which is the same
/// selection `validate_selection` admits and the same one the provider would be handed.
fn request_regime(
    engaged: &[MemoryStrategy],
    parameters: gen_core::MemoryStrategyParameters,
) -> sceneworks_core::memory_anchor::RequestRegime {
    sceneworks_core::memory_contracts::request_regime(engaged, &parameters)
}

/// Active phase residue from a compatible anchor. Production MLX fallback calls this with the
/// unbounded resident regime: generic dense-score and whole-tail tile reductions do not describe
/// fused MLX attention or layer-wise decode. Exact executed reductions belong to phase facts.
/// Cache bytes never enter this term; the selector prices recapture uncertainty separately.
#[allow(clippy::too_many_arguments)]
fn mlx_image_anchor_activation_residue(
    contract: &MemoryProviderContract,
    plan: &MlxRequestPlan,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    engaged: &[MemoryStrategy],
    parameters: gen_core::MemoryStrategyParameters,
    facts: sceneworks_core::memory_anchor::ArchitectureFacts,
) -> Option<(u64, String)> {
    use sceneworks_core::memory_anchor::ImageDeriveRequest;

    let anchor = mlx_image_anchor_match(contract, plan, mode_key, overlay, geometry)?;
    let residues = anchor.derive_phase_activation_residues(
        &ImageDeriveRequest::new(
            geometry.width,
            geometry.height,
            request_regime(engaged, parameters),
        ),
        crate::video_admission::anchor_component_bytes(contract.asset_facts),
        facts,
    )?;
    // The ACTIVE-derived phase max, with no envelope byte in it — the quantity
    // `FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE` is defined as a multiple OF.
    let peak = residues
        .conditioning
        .max(residues.denoise)
        .max(residues.decode);
    Some((peak, anchor.id.clone()))
}

/// One estimate-backed candidate synthesized for an implemented-but-unmeasured rung (sc-18096).
#[derive(Clone, Debug)]
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
struct EstimateParameterCandidate {
    parameters: gen_core::MemoryStrategyParameters,
    decode_quality: DecodeQualityRequestDecision,
}

#[derive(Default)]
struct SynthesizedEstimateLadder {
    estimates: Vec<SynthesizedEstimate>,
    decode_quality_decisions: Vec<DecodeQualityRequestDecision>,
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
/// `backend` is a parameter rather than a constant because sc-18814 routes the VIDEO lane through
/// this same shape on both backends; every image caller here passes [`MemoryBackend::Mlx`], which
/// is what the field was hardcoded to before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_evidence(
    contract: &MemoryProviderContract,
    backend: gen_core::MemoryBackend,
    tier: MemoryNumericTier,
    mode: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    selection: MemorySelection,
    predicted_peak_bytes: u64,
    calibration_fingerprint: Option<&str>,
) -> MemoryEvidence {
    sceneworks_core::memory_contracts::unverified_estimate(
        MemoryEvidenceKey {
            model_family: contract.provider_id.clone(),
            resolved_route: contract.provider_id.clone(),
            backend,
            tier,
            load_shape: contract.load_shape,
            mode: memory_mode_from_mode_key(mode),
            reference_shape: if geometry.reference_count == 0 {
                MemoryReferenceShape::None
            } else {
                MemoryReferenceShape::Image
            },
            overlay: overlay.map(str::to_owned),
            geometry,
            frames_per_second: None,
            strategy: selection.strategy,
            engaged_composition: contract.engaged_composition_for_selection(&selection),
            parameters: selection.parameters,
        },
        gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint.unwrap_or_default().to_owned(),
        REQUEST_EVIDENCE_REVISION.to_owned(),
        predicted_peak_bytes,
    )
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

/// Historical floor arithmetic retained for Candle and compatibility callers. Production MLX
/// image/video fallbacks use `mlx_fallback_weights_bytes`, which requires exact stream facts.
/// The shared core implementation owns staged peak clamps and intra-transformer eviction rules.
pub(crate) fn estimate_floor_weights_bytes(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> u64 {
    floor_weights_bytes(contract, engaged, 0)
}

/// Which ACTIVATION term the image floor composes [`image_floor_weights_bytes`] with — the input
/// that decides where rung 4's window slice is charged when the facts state no block count (see
/// the `transformer_blocks: None` regime on [`estimate_floor_weights_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageFloorActivationTerm {
    /// The sc-22663 law's per-phase residue for the rung's regime
    /// ([`mlx_image_anchor_activation_residue`]): weights-free and window-free, so the weights
    /// term must carry every resident weight itself.
    LawResidue,
    /// `MlxRequestPlan::generic_headroom_bytes`: the pre-epic geometry-blind allowance, which the
    /// sc-18096 accounting defines as carrying rung 4's window slice.
    GenericHeadroom,
}

/// The IMAGE lane's floor weights term (sc-22667): [`estimate_floor_weights_bytes`] with rung 4's
/// resident window stated as the image law states it — `transformer x min(window, blocks) /
/// blocks` — WHEN the facts carry a block count. Without one the window share is unknowable and
/// the term depends on what carries the slice: under the generic headroom, the pre-epic
/// whole-transformer-out accounting (the headroom/allowance term carries it); under the law's
/// residue, the whole transformer stays resident (the law's own reading — the residue carries no
/// weights, so nothing else would). See the regimes on [`estimate_floor_weights_bytes`].
/// `transformer_window` is the selected `transformer_window_size`; `facts` are the contract's
/// architecture facts (`video_admission::architecture_facts_from_contract`). All three inputs are
/// ignored unless the composition engages `BoundedTransformerResidency`.
#[cfg(test)]
pub(crate) fn image_floor_weights_bytes(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    transformer_window: Option<u32>,
    facts: sceneworks_core::memory_anchor::ArchitectureFacts,
    activation: ImageFloorActivationTerm,
) -> u64 {
    let windows = engaged.contains(&MemoryStrategy::BoundedTransformerResidency);
    let resident_window = match (facts.transformer_blocks, activation) {
        _ if !windows => 0,
        // A block count, or the law's residue without one: the law's helper, which states the
        // share when it can and keeps the whole transformer when it cannot.
        (Some(_), _) | (None, ImageFloorActivationTerm::LawResidue) => {
            sceneworks_core::memory_anchor::windowed_transformer_bytes(
                contract.asset_facts.transformer_bytes,
                transformer_window,
                facts.transformer_blocks,
            )
        }
        // No block count under the generic headroom: the pre-epic accounting, zero resident
        // window — the headroom term carries the slice.
        (None, ImageFloorActivationTerm::GenericHeadroom) => 0,
    };
    floor_weights_bytes(contract, engaged, resident_window)
}

/// MLX fallback weights: only an exact loader inventory can grant streaming credit.
/// A component marked bounded is still materialized in windows; unknown auxiliary inventories
/// remain fully charged. Unlike the phase estimator this deliberately keeps later-phase weights.
pub(crate) fn mlx_fallback_weights_bytes(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    parameters: gen_core::MemoryStrategyParameters,
) -> u64 {
    let facts = contract.asset_facts;
    let streamed = engaged.contains(&MemoryStrategy::BoundedTransformerResidency)
        && parameters.window_component().includes_dit();
    let retained_transformer = contract
        .phase_facts
        .as_ref()
        .and_then(|phase| phase.transformer_stream.as_ref())
        .filter(|stream| streamed && stream.peak_bytes(None) == facts.transformer_bytes)
        .map_or(facts.transformer_bytes, |stream| {
            stream.peak_bytes(parameters.transformer_window_size)
        });
    let heavy = facts
        .base_bytes
        .saturating_sub(facts.conditioning_bytes)
        .saturating_sub(facts.transformer_bytes)
        .saturating_add(retained_transformer);
    let base = if engaged.contains(&MemoryStrategy::StagedResidency) {
        facts.conditioning_bytes.max(heavy)
    } else {
        facts.conditioning_bytes.saturating_add(heavy)
    };
    let auxiliary = contract
        .resident_components()
        .iter()
        .filter(|component| component.kind.is_auxiliary())
        .fold(0_u64, |sum, component| {
            sum.saturating_add(component.resident_bytes)
        });
    base.saturating_add(auxiliary.max(facts.overlay_bytes))
}

/// The shared arithmetic: `resident_window_bytes` is what rung 4 keeps resident of the
/// transformer once the rung has removed it (0 on the video lane, the law's share on the image
/// lane), clamped at the load-exact transformer so no lane can state more than it loads.
///
/// Since sc-22738 the arithmetic itself is `sceneworks_core::memory_anchor::floor_weights_bytes`,
/// which the memory adapter's LTX-2.3 admission calls with the same facts; this is the worker's
/// lift of the contract onto that function's inputs. The staged/rung-4 exclusivity and the
/// sc-19721 eviction clamp are documented on the shared function.
fn floor_weights_bytes(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    resident_window_bytes: u64,
) -> u64 {
    let facts = contract.asset_facts;
    let auxiliary_resident_bytes = contract
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
    sceneworks_core::memory_anchor::floor_weights_bytes(
        sceneworks_core::memory_anchor::FloorWeightsFacts {
            conditioning_bytes: facts.conditioning_bytes,
            base_bytes: facts.base_bytes,
            transformer_bytes: facts.transformer_bytes,
            intra_transformer_evicted_bytes: intra_transformer_evicted_bytes(contract),
            auxiliary_resident_bytes,
        },
        sceneworks_core::memory_anchor::FloorWeightsComposition {
            staged: engaged.contains(&MemoryStrategy::StagedResidency),
            bounded_transformer: engaged.contains(&MemoryStrategy::BoundedTransformerResidency),
            resident_window_bytes,
        },
    )
}

/// Video callers share the declared parameter builder; the image builder below layers the
/// provider's decode-quality authority over the same non-decode parameters.
pub(crate) fn estimate_floor_smallest_parameters(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> Option<gen_core::MemoryStrategyParameters> {
    sceneworks_core::memory_contracts::smallest_parameters(
        contract,
        engaged,
        sceneworks_core::memory_contracts::TransformerScopeParameters::NativeQuant,
    )
}

#[allow(clippy::too_many_arguments)]
fn estimate_floor_parameters(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    strategy: MemoryStrategy,
    tier: MemoryNumericTier,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    use_pid: bool,
) -> (
    Vec<EstimateParameterCandidate>,
    Vec<DecodeQualityRequestDecision>,
) {
    let non_decode: Vec<_> = engaged
        .iter()
        .copied()
        .filter(|rung| *rung != MemoryStrategy::BoundedDecode)
        .collect();
    let Some(base_parameters) = sceneworks_core::memory_contracts::smallest_parameters(
        contract,
        &non_decode,
        sceneworks_core::memory_contracts::TransformerScopeParameters::ImplicitDit,
    ) else {
        return (
            Vec::new(),
            vec![DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: if engaged.contains(&MemoryStrategy::BoundedAttention)
                    && contract
                        .capability(MemoryStrategy::BoundedAttention)
                        .is_none_or(|capability| {
                            capability.parameters.attention_chunk_sizes.is_empty()
                        }) {
                    "required attention chunk range is empty"
                } else {
                    "required transformer window range is empty"
                }
                .to_owned(),
            }],
        );
    };
    if !engaged.contains(&MemoryStrategy::BoundedDecode) {
        return (
            vec![EstimateParameterCandidate {
                parameters: base_parameters,
                decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
            }],
            Vec::new(),
        );
    }
    let Some(decode) = contract.capability(MemoryStrategy::BoundedDecode) else {
        return (
            Vec::new(),
            vec![DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "bounded decode capability is absent".to_owned(),
            }],
        );
    };
    if decode.parameters.decode_geometry_policies.is_empty() {
        if contract.decode_geometry_policy_authoritative {
            let refusal = DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "bounded decode has a declared quality scope but no applicable row for the loaded tier/load shape"
                    .to_owned(),
            };
            // The decode rung itself has no exact authority. A higher independent rung retains its
            // own estimate with decode omitted, so request-selection-aware engagement cannot
            // accidentally resurrect the route-blind tile domain.
            let candidates = (strategy != MemoryStrategy::BoundedDecode)
                .then_some(EstimateParameterCandidate {
                    parameters: base_parameters,
                    decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
                })
                .into_iter()
                .collect();
            return (candidates, vec![refusal]);
        }
        let pair = if let Some(routes) = &contract.pid_decode_routes {
            let route = if use_pid { &routes.pid } else { &routes.native };
            route
                .tile_edges
                .iter()
                .copied()
                .min()
                .zip(Some(route.tile_overlap))
        } else {
            sceneworks_core::memory_contracts::smallest_parameters(
                contract,
                engaged,
                sceneworks_core::memory_contracts::TransformerScopeParameters::ImplicitDit,
            )
            .and_then(|parameters| parameters.decode_tile_edge.zip(parameters.decode_overlap))
        };
        let Some((tile_edge, overlap)) = pair else {
            return (
                Vec::new(),
                vec![DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: None,
                    overlap: None,
                    evidence_sha256: None,
                    reason: "legacy bounded decode has no complete edge/overlap pair".to_owned(),
                }],
            );
        };
        let mut parameters = base_parameters;
        parameters.decode_tile_edge = Some(tile_edge);
        parameters.decode_overlap = Some(overlap);
        return (
            vec![EstimateParameterCandidate {
                parameters,
                decode_quality: DecodeQualityRequestDecision::NotDeclared,
            }],
            Vec::new(),
        );
    }

    let query = MemoryDecodePolicyQuery {
        tier,
        load_shape: contract.load_shape,
        mode_key,
        overlay,
        geometry,
        use_pid,
    };
    let rows = match contract.decode_geometry_policies_for_request(query) {
        Ok(rows) => rows,
        Err(error) => {
            return (
                Vec::new(),
                vec![DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: None,
                    overlap: None,
                    evidence_sha256: None,
                    reason: error.to_string(),
                }],
            )
        }
    };
    let mut rows = rows;
    rows.sort_by_key(|policy| (policy.tile_edge, policy.overlap));
    let mut candidates = Vec::new();
    let mut decisions = Vec::new();
    for policy in rows {
        match &policy.disposition {
            MemoryDecodeQualityDisposition::Admitted => {
                let mut parameters = base_parameters;
                parameters.decode_tile_edge = Some(policy.tile_edge);
                parameters.decode_overlap = Some(policy.overlap);
                candidates.push(EstimateParameterCandidate {
                    parameters,
                    decode_quality: DecodeQualityRequestDecision::Admitted {
                        strategy,
                        tile_edge: policy.tile_edge,
                        overlap: policy.overlap,
                        evidence_sha256: policy.production_evidence_sha256.clone(),
                    },
                });
            }
            MemoryDecodeQualityDisposition::Refused { reason } => {
                decisions.push(DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: Some(policy.tile_edge),
                    overlap: Some(policy.overlap),
                    evidence_sha256: Some(policy.production_evidence_sha256.clone()),
                    reason: reason.clone(),
                });
            }
        }
    }
    if candidates.is_empty() {
        if decisions.is_empty() {
            decisions.push(DecodeQualityRequestDecision::Unmeasured {
                strategy,
                reason: format!(
                    "no exact production-latent row for {}x{} load_shape={:?} mode={} overlay={overlay:?} use_pid={use_pid}",
                    geometry.width, geometry.height, contract.load_shape, mode_key,
                ),
            });
        }
        // Bounded decode itself cannot exist without an admitted pair. A higher independent rung
        // keeps its own parameters and omits decode so selection-aware engagement excludes rung 2.
        if strategy != MemoryStrategy::BoundedDecode {
            candidates.push(EstimateParameterCandidate {
                parameters: base_parameters,
                decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
            });
        }
    }
    (candidates, decisions)
}

/// Price active memory under the selected execution schedule. A resident anchor supplies
/// only activation residues; changing the load shape here does not relabel it as a measured
/// optimized run. Unsupported modes and auxiliary ownership retain the conservative fallback.
#[allow(clippy::too_many_arguments)]
fn mlx_selected_phase_peak(
    contract: &MemoryProviderContract,
    plan: &MlxRequestPlan,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    selection: &MemorySelection,
    conditioning_windows: Option<u32>,
) -> Option<u64> {
    use crate::mlx_phase_estimate::layerwise_decode_workspace;
    use sceneworks_core::memory_anchor::{AnchorBackend, AnchorMlxImageDeriveRequest};
    if mode_key != "text_to_image"
        || overlay.is_some()
        || geometry.reference_count != 0
        || geometry.frames != 1
        || geometry.batch != 1
    {
        return None;
    }
    let phase = contract.phase_facts.as_ref()?;
    let extra = flux3_cfg_additional_workspaces(plan.engine_id, geometry.width, geometry.height);
    let generic = plan
        .generic_headroom_bytes(geometry)
        .saturating_sub(plan.fixed_reserve_bytes.min(plan.activation_headroom_bytes))
        .saturating_sub(extra.0.max(extra.1));
    let mut workspace = flux3_cfg_workspace(plan.engine_id, geometry, generic);
    if let Some(anchor) = mlx_image_anchor_store()
        .and_then(|store| {
            store.image_anchor_for(&plan.model_id, AnchorBackend::Mlx, plan_tier_key(plan.tier))
        })
        .filter(|anchor| {
            anchor.route == plan.engine_id
                && anchor.provider == contract.provider_id
                && anchor.mode == mode_key
                && anchor.overlay.is_none()
                && anchor.reference_count == 0
        })
    {
        let components = crate::video_admission::anchor_component_bytes(contract.asset_facts);
        let residues = anchor.derive_mlx_image_activation_residues(
            AnchorMlxImageDeriveRequest {
                width: geometry.width,
                height: geometry.height,
            },
            components,
        );
        workspace.conditioning = residues[0].unwrap_or(workspace.conditioning);
        workspace.denoise = residues[1].unwrap_or(workspace.denoise);
        workspace.decode = residues[2].unwrap_or(workspace.decode);
        if residues[1].is_some()
            && phase.architecture == Some(gen_core::ImagePipelineArchitecture::ZImageDit)
        {
            // Fused SDPA scales with image tokens, without materializing a quadratic score matrix.
            let resident = components
                .conditioning
                .saturating_add(components.transformer)
                .saturating_add(components.decoder);
            let residue = anchor
                .phase_active_peak_bytes
                .denoise
                .checked_sub(resident)?;
            let scale = (f64::from(geometry.width) * f64::from(geometry.height)
                / (f64::from(anchor.geometry.width) * f64::from(anchor.geometry.height)))
            .max(1.0);
            workspace.denoise = (residue as f64 * scale).ceil() as u64;
        }
    }
    if phase.architecture == Some(gen_core::ImagePipelineArchitecture::SdxlUnetWithDualClip) {
        let windows = conditioning_windows.filter(|windows| *windows > 0)?;
        // Retained SDXL UNet-only seam: CFG batch 2, 77 tokens, 1024², dense fp16.
        // The exact selected reference tensor shapes total 5,134,927,368 bytes.
        let seam_workspace = (6.58685 * BYTES_PER_GIB).ceil() as u64 - 5_134_927_368;
        workspace.conditioning = crate::mlx_phase_estimate::dual_clip_workspace(windows);
        // Scaling the whole workspace by padded context windows is conservative: only the
        // cross-attention and carried conditioning actually grow with prompt length.
        workspace.denoise = ((seam_workspace as f64)
            * f64::from(windows)
            * (f64::from(geometry.width) * f64::from(geometry.height) / 1_048_576.0).max(1.0))
        .ceil() as u64;
    }
    if let Some(gen_core::ImagePipelineArchitecture::SanaLinearDit {
        classifier_free_guidance,
    }) = phase.architecture
    {
        if contract.engages_selection(selection, MemoryStrategy::BoundedDecode)
            && selection.parameters.decode_overlap == Some(48)
        {
            if let Some(derived) = phase.decoder_workspace.as_ref().and_then(|facts| {
                crate::mlx_phase_estimate::sana_workspace(
                    facts,
                    geometry,
                    selection.parameters.decode_tile_edge,
                    classifier_free_guidance,
                )
            }) {
                workspace = derived;
            }
        }
    }
    if contract.engages_selection(selection, MemoryStrategy::BoundedDecode) {
        if let (Some(decoder), Some(edge)) = (
            &phase.decoder_workspace,
            selection.parameters.decode_tile_edge,
        ) {
            if let Some(bytes) = layerwise_decode_workspace(decoder, geometry, edge) {
                workspace.decode = bytes;
            }
        }
    }
    crate::mlx_phase_estimate::peak(contract, selection, workspace)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn synthesize_estimate_ladder(
    contract: &MemoryProviderContract,
    plan: &MlxRequestPlan,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    use_pid: bool,
    calibration_fingerprint: Option<&str>,
    bases: &[MeasuredRungBasis],
) -> SynthesizedEstimateLadder {
    synthesize_estimate_ladder_with_conditioning(
        contract,
        plan,
        mode_key,
        overlay,
        geometry,
        use_pid,
        calibration_fingerprint,
        bases,
        None,
    )
}

/// Price implemented selections, including supported staging compositions. Compatible fitted
/// cells take precedence; provider phase facts and retained active observations then derive an
/// estimate. Unknown facts retain conservative weights and workspace. The selector owns margin
/// widening, and every tiled candidate still requires its exact decode-quality policy.
#[allow(clippy::too_many_arguments)]
fn synthesize_estimate_ladder_with_conditioning(
    contract: &MemoryProviderContract,
    plan: &MlxRequestPlan,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    use_pid: bool,
    calibration_fingerprint: Option<&str>,
    bases: &[MeasuredRungBasis],
    conditioning_windows: Option<u32>,
) -> SynthesizedEstimateLadder {
    use crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE;
    use crate::memory_strategy::CandidateBasis;

    let request_area = f64::from(geometry.width) * f64::from(geometry.height);
    let mut ladder = SynthesizedEstimateLadder::default();
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
        let declared_engaged = contract.engaged_composition(strategy);
        let (mut parameter_candidates, decisions) = estimate_floor_parameters(
            contract,
            &declared_engaged,
            strategy,
            plan.tier,
            mode_key,
            overlay,
            geometry,
            use_pid,
        );
        ladder.decode_quality_decisions.extend(decisions);

        if strategy != MemoryStrategy::StagedResidency
            && !declared_engaged.contains(&MemoryStrategy::StagedResidency)
            && matches!(
                contract
                    .capability(MemoryStrategy::StagedResidency)
                    .map(|cap| &cap.support),
                Some(gen_core::MemoryStrategySupport::Implemented)
            )
        {
            let staged = parameter_candidates
                .iter()
                .cloned()
                .map(|mut candidate| {
                    candidate.parameters.stage_residency = Some(true);
                    candidate
                })
                .collect::<Vec<_>>();
            parameter_candidates.extend(staged);
        }

        for parameter_candidate in parameter_candidates {
            let floor_selection = MemorySelection {
                strategy,
                parameters: parameter_candidate.parameters,
                tier: plan.tier,
            };
            let engaged = contract.engaged_composition_for_selection(&floor_selection);

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
                        && basis.parameters.decode_tile_edge
                            == parameter_candidate.parameters.decode_tile_edge
                        && basis.parameters.decode_overlap
                            == parameter_candidate.parameters.decode_overlap
                        && (contract
                            .phase_facts
                            .as_ref()
                            .and_then(|phase| phase.architecture)
                            != Some(gen_core::ImagePipelineArchitecture::SdxlUnetWithDualClip)
                            || conditioning_windows == Some(1))
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
                    let mut parameters = basis.parameters;
                    parameters.stage_residency = parameter_candidate.parameters.stage_residency;
                    parameters.decode_tile_edge = parameter_candidate.parameters.decode_tile_edge;
                    parameters.decode_overlap = parameter_candidate.parameters.decode_overlap;
                    let selection = MemorySelection {
                        strategy,
                        parameters,
                        tier: plan.tier,
                    };
                    if contract.validate_selection(&selection).is_err() {
                        return None;
                    }
                    let measured_area =
                        f64::from(basis.geometry.width) * f64::from(basis.geometry.height);
                    let scale = (request_area / measured_area).max(1.0);
                    let scaled = |bytes: u64| {
                        (bytes as f64 * scale).ceil().clamp(0.0, u64::MAX as f64) as u64
                    };
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
                        basis_geometry =
                            format!("{}x{}", basis.geometry.width, basis.geometry.height),
                        raw_peak_bytes = predicted_peak_bytes,
                        area_scale = scale,
                        "synthesized fitted-curve estimate candidate from a measured cell"
                    );
                    Some(SynthesizedEstimate {
                        selection,
                        evidence: estimate_evidence(
                            contract,
                            gen_core::MemoryBackend::Mlx,
                            plan.tier,
                            mode_key,
                            overlay,
                            geometry,
                            selection,
                            predicted_peak_bytes,
                            calibration_fingerprint,
                        ),
                        basis: CandidateBasis::EstimateFittedCurve,
                        unmodeled_activation_bytes: None,
                        activation_term: None,
                        decode_quality: parameter_candidate.decode_quality.clone(),
                    })
                });
            if let Some(candidate) = fitted {
                ladder
                    .decode_quality_decisions
                    .push(candidate.decode_quality.clone());
                ladder.estimates.push(candidate);
                continue;
            }

            if let Some(predicted_peak_bytes) = mlx_selected_phase_peak(
                contract,
                plan,
                mode_key,
                overlay,
                geometry,
                &floor_selection,
                conditioning_windows,
            )
            .filter(|_| contract.validate_selection(&floor_selection).is_ok())
            {
                ladder.estimates.push(SynthesizedEstimate {
                    selection: floor_selection,
                    evidence: estimate_evidence(
                        contract,
                        gen_core::MemoryBackend::Mlx,
                        plan.tier,
                        mode_key,
                        overlay,
                        geometry,
                        floor_selection,
                        predicted_peak_bytes,
                        calibration_fingerprint,
                    ),
                    basis: CandidateBasis::EstimateFloor,
                    unmodeled_activation_bytes: None,
                    activation_term: None,
                    decode_quality: parameter_candidate.decode_quality.clone(),
                });
                continue;
            }

            // 2. Anchor-derived (epic 22505 feature-end fix round, E2/E7): the measured image
            //    anchor for this (model, tier, mlx lane) prices the request analytically —
            //    per-phase allocator envelopes upper-bounding every optimized composition — and
            //    OUTRANKS the generic weights+headroom floor below whenever the anchor is current
            //    and every identity/regime conjunct holds (`mlx_image_anchor_derived_peak`). The
            //    selector charges it the lane's same-cell recapture spread and nothing else
            //    (`ladder_margin_policy`: the image law fits and widens nothing, so that spread
            //    is the one uncertainty left over its peak — sc-22663). At this pin the packaged
            //    flux2 rows are outside the law's domain and this arm yields to the floor; see
            //    `MemoryAnchor::derive_mlx_image_phase_peaks`.
            let anchored =
                mlx_image_anchor_derived_peak(contract, plan, mode_key, overlay, geometry)
                    .and_then(|(predicted_peak_bytes, anchor_id)| {
                        let selection = MemorySelection {
                            strategy,
                            parameters: parameter_candidate.parameters,
                            tier: plan.tier,
                        };
                        if contract.validate_selection(&selection).is_err() {
                            return None;
                        }
                        tracing::info!(
                        route = contract.provider_id,
                        backend = "mlx",
                        ?strategy,
                        anchor = anchor_id.as_str(),
                        raw_peak_bytes = predicted_peak_bytes,
                        "synthesized anchor-derived estimate candidate from the measured image \
                         anchor"
                    );
                        Some(SynthesizedEstimate {
                            selection,
                            evidence: estimate_evidence(
                                contract,
                                gen_core::MemoryBackend::Mlx,
                                plan.tier,
                                mode_key,
                                overlay,
                                geometry,
                                selection,
                                predicted_peak_bytes,
                                calibration_fingerprint,
                            ),
                            basis: CandidateBasis::EstimateAnchorDerived {
                                lane: crate::memory_strategy::AnchorDerivationLane::Image,
                            },
                            // The anchor derivation decomposes its peak by PHASE, not into counted
                            // weights plus an activation remainder — same reason as the fitted arm.
                            unmodeled_activation_bytes: None,
                            activation_term: None,
                            decode_quality: parameter_candidate.decode_quality.clone(),
                        })
                    });
            if let Some(candidate) = anchored {
                ladder
                    .decode_quality_decisions
                    .push(candidate.decode_quality.clone());
                ladder.estimates.push(candidate);
                continue;
            }

            // 3. Weights + headroom floor — no measured basis, so the binding-phase constraint does
            //    not gate it (scope sentence on the constraint's doc).
            let selection = MemorySelection {
                strategy,
                parameters: parameter_candidate.parameters,
                tier: plan.tier,
            };
            if contract.validate_selection(&selection).is_err() {
                continue;
            }
            //    Its ACTIVATION term is the sc-22663 law's per-phase residue for this rung's own
            //    regime whenever the anchor can price it (sc-22665, E4) — so the tile, the
            //    attention chunk and the transformer window reach the MLX estimate instead of the
            //    geometry-blind generic headroom, which stays the fallback for every request the
            //    anchor or the law refuses (at this pin: all of them — see
            //    `mlx_image_anchor_activation_residue`).
            let facts = crate::video_admission::architecture_facts_from_contract(contract);
            let derived_residue = mlx_image_anchor_activation_residue(
                contract,
                plan,
                mode_key,
                overlay,
                geometry,
                &[],
                gen_core::MemoryStrategyParameters::default(),
                facts,
            );
            let (floor_activation_bytes, activation_term) = match &derived_residue {
                Some((residue, _)) => (*residue, ImageFloorActivationTerm::LawResidue),
                None => (
                    plan.generic_headroom_bytes(geometry),
                    ImageFloorActivationTerm::GenericHeadroom,
                ),
            };
            // The weights term states rung 4's resident window as the law does (sc-22667): the
            // image lane's floor and the candle ladder price ONE windowed residency. It is told
            // which activation term it is composed with, because without a block count that is
            // what decides whether the window slice is carried by the headroom or must stay in
            // the weights.
            // Generic headroom includes fixed OS/app reserve; only actual activations receive
            // the allocator allowance. A law-derived residue is already activation-only.
            let unmodeled_activation_bytes = match activation_term {
                ImageFloorActivationTerm::GenericHeadroom => floor_activation_bytes
                    .saturating_sub(plan.fixed_reserve_bytes.min(plan.activation_headroom_bytes)),
                ImageFloorActivationTerm::LawResidue => floor_activation_bytes,
            };
            let predicted_peak_bytes =
                mlx_fallback_weights_bytes(contract, &engaged, parameter_candidate.parameters)
                    .saturating_add(floor_activation_bytes);
            tracing::info!(
                route = contract.provider_id,
                backend = "mlx",
                ?strategy,
                raw_peak_bytes = predicted_peak_bytes,
                headroom_bytes = floor_activation_bytes,
                activation_bytes = unmodeled_activation_bytes,
                activation_anchor = derived_residue
                    .as_ref()
                    .map(|(_, anchor_id)| anchor_id.as_str()),
                "synthesized weights+headroom floor estimate candidate"
            );
            let candidate = SynthesizedEstimate {
                selection,
                evidence: estimate_evidence(
                    contract,
                    gen_core::MemoryBackend::Mlx,
                    plan.tier,
                    mode_key,
                    overlay,
                    geometry,
                    selection,
                    predicted_peak_bytes,
                    calibration_fingerprint,
                ),
                basis: CandidateBasis::EstimateFloor,
                unmodeled_activation_bytes: Some(unmodeled_activation_bytes),
                activation_term: Some(activation_term),
                decode_quality: parameter_candidate.decode_quality,
            };
            ladder
                .decode_quality_decisions
                .push(candidate.decode_quality.clone());
            ladder.estimates.push(candidate);
        }
    }
    ladder
}

/// Select the largest strictly lower, same-aspect geometry backed by an exact record of the
/// installed artifact that fits the live host boundary. This is the only source for a named
/// refusal alternative: no formula, interpolation, tier heuristic, or aspect-ratio rewrite is
/// admitted.
///
/// The binding's provider closure is not a conjunct (sc-22738): the alternative is advice the very
/// next request will price from the same record, and that request grades the record at its
/// measured peak whatever closure it was captured under, so the advice is honoured either way.
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
                && binding.geometry.batch == request_batch(inputs)
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
                transformer_variant: None,
                decoder: None,
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
                    parameters: binding.selection_parameters,
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
) -> Option<CalibrationGeometry> {
    verified_lower_alternative(bundle, calibration, plan, inputs, mode_key, budget)
        .map(|alternative| alternative.geometry)
}

/// Pure request selector used by production and unit/hardware seams. Additional provider evidence is
/// accepted explicitly; absent exact verified cells, only the resident baseline can be admitted.
#[allow(clippy::too_many_arguments)]
/// The pre-sc-18317 shape, kept for the fixture suite: an evaluation with nothing to decide about the
/// request's execution policy. Production always goes through
/// [`evaluate_request_with_budget_and_warm_policy`] so a grant cannot be dropped on the way in.
#[cfg(test)]
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
    evaluate_request_with_budget_and_warm_policy(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
        budget,
        total_peak_bytes,
        external_committed_bytes,
        additional_evidence,
        None,
    )
}

/// [`evaluate_request_with_budget`] carrying this request's warm-policy proposal (sc-18317).
#[allow(clippy::too_many_arguments)]
fn evaluate_request_with_budget_and_warm_policy(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
    reference_pricing: Option<ReferencePricing<'_>>,
) -> WorkerResult<MlxRequestEvaluation> {
    evaluate_request_with_budget_using_bundle(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        warm_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        additional_evidence,
        None,
        reference_pricing,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_request_with_budget_using_bundle(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
    evidence_bundle: Option<&EvidenceBundle>,
    reference_pricing: Option<ReferencePricing<'_>>,
) -> WorkerResult<MlxRequestEvaluation> {
    select_request_with_budget_using_bundle(
        MlxAdmissionPhase::Generation,
        generator
            .descriptor()
            .capabilities
            .component_precision_floors,
        generator.memory_strategy_contract(),
        crate::execution_planner::prefix_kv_cache_surcharge_bytes(
            plan.engine_id,
            &generator.descriptor().capabilities.execution,
            plan.tier.quant,
            inputs.width,
            inputs.height,
            inputs.reference_count,
            inputs.negative_branch,
        )
        .filter(|_| inputs.prefix_kv_cache_eligible),
        &|forced_tile_edge| {
            mlx_prefix_kv_cache_shed_transient_bytes(
                plan.engine_id,
                plan.tier.quant,
                inputs,
                forced_tile_edge,
            )
        },
        |base| generator.predicted_memory_peak_from_base(base),
        plan,
        inputs,
        cache_state,
        load_policy,
        warm_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        additional_evidence,
        evidence_bundle,
        reference_pricing,
    )?
    .into_evaluation()
}

/// The LINKED provider's own cache-shed request transient for a prefix-KV-cache provider
/// (`derived::request_transient_budget_bytes` — the budget the generator holds a prefill render
/// to), at the loaded tier and the rung's forced decode tile; `None` for every other provider.
///
/// sc-25909 (owner-approved E6 exception): the shed ladder compares this against the activation a
/// reference request's rung was charged — the activation slice of the generic area-scaled
/// headroom, on a rung whose activation term is that headroom — and prices the shortfall into
/// every such rung's peak. Every admission that runs the shed ladder grades it: the prefill path,
/// the cached path (corrected shed rung + cache surcharge) and the pre-load tier probe. Same
/// 256-row conditioning assumption as the generator's own bound.
fn mlx_prefix_kv_cache_shed_transient_bytes(
    engine_id: &str,
    tier_quant: Option<gen_core::Quant>,
    inputs: &MlxRequestInputs,
    forced_tile_edge: Option<u32>,
) -> Option<u64> {
    if !sceneworks_core::jobs_store::is_qwen_image_2_1_model(engine_id) {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        use runtime_macos::providers::qwen_image_2_1 as provider;
        let tier = match tier_quant {
            None => provider::quant::Tier::Bf16,
            Some(gen_core::Quant::Q8) => provider::quant::Tier::Q8,
            Some(gen_core::Quant::Q4) => provider::quant::Tier::Q4,
            Some(_) => return None,
        };
        let derived = provider::memory_strategy::derived::request_transient_budget_bytes;
        Some(derived(
            tier,
            inputs.width,
            inputs.height,
            provider::memory_strategy::derived::TABLE_CONDITIONING_TOKENS,
            inputs.reference_count,
            inputs.negative_branch,
            forced_tile_edge,
        ))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (tier_quant, inputs, forced_tile_edge);
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MlxAdmissionPhase {
    TierProbe,
    Generation,
}

/// A budget refusal is distinct from invalid artifacts/contracts: only the former can try a
/// lower installed tier. Keep the original request-scoped diagnostic through tier selection.
pub(crate) enum MlxRequestAdmission {
    Admitted(Box<MlxRequestEvaluation>),
    Rejected(WorkerError),
}

impl MlxRequestAdmission {
    fn into_evaluation(self) -> WorkerResult<MlxRequestEvaluation> {
        match self {
            Self::Admitted(evaluation) => Ok(*evaluation),
            Self::Rejected(error) => Err(error),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn select_request_with_budget_using_bundle(
    phase: MlxAdmissionPhase,
    declared_floors: &'static [gen_core::ComponentPrecisionFloor],
    declared_contract: Option<&MemoryProviderContract>,
    // The provider's prefix-KV-cache surcharge for this request
    // (`execution_planner::prefix_kv_cache_surcharge_bytes`, priced against the LOADED generator's
    // declaration), when this admission may select the cache. `None` for every provider without
    // one — and on the pre-load tier probe: a tier fits when it fits with the cache shed (the cache
    // is only ever added to the rung the shed demand selects, never a reason to drop a tier), so the
    // probe grades exactly the shed demand.
    prefix_kv_cache_surcharge: Option<u64>,
    // sc-25909: the provider's own cache-shed request transient at a rung's forced decode tile
    // (`mlx_prefix_kv_cache_shed_transient_bytes`), or `None`. Priced into every rung whose
    // activation term is the generic area-scaled headroom — into the shed ladder itself, so the
    // prefill and cached paths both grade it (owner-approved E6 exception, 2026-10-16).
    prefix_kv_cache_shed_transient: &dyn Fn(Option<u32>) -> Option<u64>,
    predict_peak: impl FnOnce(u64) -> gen_core::MemoryPeakBreakdown,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
    evidence_bundle: Option<&EvidenceBundle>,
    reference_pricing: Option<ReferencePricing<'_>>,
) -> WorkerResult<MlxRequestAdmission> {
    use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};

    let reference_workspace_bytes = if let Some(resources) = &inputs.reference_resources {
        if !inputs.has_reference
            || usize::try_from(inputs.reference_count).ok()
                != Some(resources.prepared_canvases.len())
        {
            return Err(WorkerError::InvalidPayload(
                "reference resource slots do not match the admitted ordered reference count"
                    .to_owned(),
            ));
        }
        let pricing = reference_pricing.ok_or_else(|| missing_reference_pricing(plan.engine_id))?;
        crate::memory_strategy::reference_workspace_bytes_using(
            pricing.registry,
            plan.engine_id,
            pricing.spec,
            Some(resources),
        )?
    } else {
        0
    };

    // Component precision floors are a provider property, not a manifest guess. Bind them only
    // from the exact registry descriptor or loaded generator, then use that tier for every evidence/cache identity
    // in this request so uniform-q4 measurements cannot authorize a mixed-precision provider.
    let provider_floors = active_component_floors(declared_floors, plan.tier.quant);
    let mut effective_plan;
    let plan = if plan.tier.component_precision_floors == provider_floors {
        plan
    } else {
        effective_plan = plan.clone();
        effective_plan.tier.component_precision_floors = provider_floors;
        &effective_plan
    };

    let provider_inputs = provider_request_inputs(plan.engine_id, inputs);
    let inputs = provider_inputs.as_ref();
    let geometry = request_geometry(inputs);
    let (mode, mode_key) = provider_request_mode(plan.engine_id, inputs);
    let mut fallback_contract;
    let provider_contract = if let Some(contract) = declared_contract {
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
    let refused_contract;
    let contract = if plan.load_shape_declaration_result == LoadShapeDeclarationResult::Refused
        || !crate::memory_route_registry::has_gate_policy(
            crate::memory_route_registry::MemoryRouteBackend::Mlx,
            plan.engine_id,
        ) {
        // A declaration refusal is terminal authority, not an invitation for the later request
        // selector to rediscover lower optimized rungs directly from the provider. Preserve only
        // the exact provider/load/calibration handshake needed for a safe resident request; the
        // provider's optimized table remains unreachable until a complete declaration applies.
        let mut resident = MemoryProviderContract::compatibility_default(
            provider_contract.provider_id.clone(),
            provider_contract.backend.clone(),
        );
        resident.load_shape = provider_contract.load_shape;
        resident.calibration = provider_contract.calibration.clone();
        resident.asset_facts = provider_contract.asset_facts;
        resident.resident_request_memory = provider_contract.resident_request_memory;
        refused_contract = resident;
        &refused_contract
    } else {
        provider_contract
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
    // No live closure is resolved here (sc-22738). Between sc-17774 and sc-22738 this gate read
    // the provider's compile-closure digest and graded every measured candidate against it; now a
    // measured candidate is graded at its measured peak whatever closure it was captured under.
    let mut admission = if plan.tier.component_precision_floors.is_empty() {
        match evidence_bundle {
            Some(bundle) => evidence_admission_route(bundle, plan, inputs, mode_key, budget)?,
            None => packaged_admission_route(plan, inputs, mode_key, budget)?,
        }
    } else {
        // Persisted calibration bindings currently identify only the coarse tier token (for
        // example `q4`). They cannot distinguish uniform q4 from a provider whose descriptor keeps
        // selected components at q8, so they must not be relabeled with the live descriptor floors.
        // Until the on-disk evidence schema carries the floors explicitly, fail closed to the
        // provider's conservative resident estimate.
        AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::IdentityMismatch),
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
    // pre-epic code degraded to legacy first (the retired closure pre-demotion caught every
    // drifted binding before eligibility ran). The filter now mirrors those legs too,
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
            let selection = MemorySelection {
                strategy: evidence.key.strategy,
                parameters: evidence.key.parameters,
                tier: evidence.key.tier,
            };
            let composition_agrees = evidence.key.engaged_composition
                == contract.engaged_composition_for_selection(&selection);
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
                fallback_reason: Some(LegacyAdmissionReason::IdentityMismatch),
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
            contract
                .calibration
                .as_ref()
                .is_none_or(|identity| alternative.load_shape != identity.load_shape)
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
                .is_none_or(|alternative| {
                    identity.abi == alternative.calibration_abi
                        && identity.fingerprint == alternative.calibration_fingerprint
                        && contract.engaged_composition_for_selection(&MemorySelection {
                            strategy: alternative.strategy,
                            parameters: alternative.parameters,
                            tier: plan.tier,
                        }) == alternative.engaged_composition
                })
        })
    {
        admission = AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::IdentityMismatch),
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
    if plan.engine_id.starts_with("mage_flow") {
        // The expected identity is the (route, tier) cell of THIS plan: a loaded q8 contract must
        // not admit a q4 request, and a route or tier the engine never ships names no cell at all.
        let expected_fingerprint = mage_calibration_fingerprint(plan.engine_id, plan.tier.quant)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{} {:?} names no Mage-Flow calibration cell; refusing request admission \
                     against an unpaired estimator",
                    plan.engine_id, plan.tier.quant
                ))
            })?;
        if !matches!(
            contract.calibration.as_ref(),
            Some(identity)
                if identity.abi == gen_core::MEMORY_CALIBRATION_ABI
                    && identity.fingerprint == expected_fingerprint
        ) {
            return Err(WorkerError::InvalidPayload(format!(
                "{} loaded provider calibration does not match ABI {} / fingerprint {}; refusing \
                 request admission against an unpaired estimator",
                plan.engine_id,
                gen_core::MEMORY_CALIBRATION_ABI,
                expected_fingerprint
            )));
        }
    }
    // The caller estimates the base-model pipeline. Let the provider's canonical contract seam add
    // any separately declared auxiliary networks before either fit selection or warm-cache credit.
    // Exact evidence already describes the whole request peak and therefore remains authoritative.
    let contract_base_peak_bytes = plan.contract_base_peak_bytes(total_peak_bytes, contract);
    let base_prediction = predict_peak(contract_base_peak_bytes);
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
    tracing::info!(
        route = plan.engine_id,
        total_bytes = budget.total_bytes,
        active_bytes = budget.committed_bytes,
        external_baseline_bytes = external_committed_bytes,
        provider_credit_bytes = attributable_resident_bytes,
        reserved_bytes = budget.reserved_headroom_bytes,
        "MLX request memory attribution"
    );
    // Every candidate is a whole-pipeline peak, including synthesized ladder estimates.
    // Remove this provider's already-resident allocation from the committed side once, so
    // cold/warm Resident and staged candidates use the same accounting domain. Keep unrelated
    // allocations charged, and apply safety allowances to the original modeled terms.
    budget.committed_bytes = budget
        .committed_bytes
        .saturating_sub(attributable_resident_bytes);
    let predicted_peak_bytes = modeled_peak_bytes;
    // `request_total_peak_bytes` is the generic weights+headroom total on every route except
    // `mage_flow`, whose resident peak is the provider's own `generation_peak_gb` scalar.
    let resident_peak_is_provider_scalar = plan.engine_id.starts_with("mage_flow");
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
    let synthesized_ladder = if admission.path == AdmissionPath::Legacy {
        synthesize_estimate_ladder_with_conditioning(
            contract,
            plan,
            mode_key,
            inputs.overlay.as_deref(),
            geometry,
            inputs.use_pid,
            calibration_fingerprint,
            &admission.estimate_bases,
            inputs.conditioning_windows,
        )
    } else {
        SynthesizedEstimateLadder::default()
    };
    let synthesized_estimates = &synthesized_ladder.estimates;
    let mut selections = Vec::new();
    let mut evidence = Vec::new();
    // Index-aligned basis axis (sc-18096): synthesized candidates carry their estimate basis, the
    // resident baseline is the rung-0 weights+headroom floor, and everything measured stays
    // `Measured`. Pushed at the same sites as `evidence` for the same fail-open reason as the
    // digests above.
    let mut candidate_bases: Vec<crate::memory_strategy::CandidateBasis> = Vec::new();
    // Index-aligned ACTIVATION axis (sc-22508), pushed at the same sites for the same reason: only
    // the site that CONSTRUCTED a peak knows whether it decomposes into counted weights and a flat
    // activation term, and what that term is. A blanket declaration here was wrong for two peaks —
    // a `mage_flow` route, whose peak is `providers::mage::memory::generation_peak_gb` and has no
    // weights/headroom split at all, and any peak a generator rewrote through
    // `predicted_memory_peak_from_base`. Both now carry `None` and take the policy's documented
    // undeclared-floor arm instead of being charged a headroom they do not contain.
    let mut candidate_activation_bytes: Vec<Option<u64>> = Vec::new();
    // Index-aligned, pushed at the same sites (sc-25909): whether the candidate's peak IS counted
    // weights + the generic area-scaled headroom — the resident baseline on every non-`mage_flow`
    // route and a floor estimate composed with `ImageFloorActivationTerm::GenericHeadroom`. Only
    // those rungs take the provider-transient correction below; a measured, fitted-curve,
    // anchor-derived, phase or law-residue rung prices its own activation.
    let mut candidate_charges_generic_headroom: Vec<bool> = Vec::new();
    // Index-aligned with `evidence`: the Evidence path's per-candidate pre-check budget (GiB, at the
    // candidate's CAPTURED foreign reserve), so a cache-surcharged pass re-runs the same pre-check
    // at the surcharged peak. `None` where no pre-check applies (the Legacy path).
    let mut candidate_precheck_available_gb: Vec<Option<f64>> = Vec::new();
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
            // which the selector's uniform zero-reserve Evidence budget cannot carry. It grades the
            // measured peak exactly, as the selector does (sc-22738: no widening for a moved
            // closure).
            if candidate_budget.effective_gb().is_some_and(|available| {
                exact
                    .predicted_peak_bytes
                    .saturating_add(reference_workspace_bytes) as f64
                    / BYTES_PER_GIB
                    <= available
            }) {
                selections.push(MemorySelection {
                    strategy: exact.key.strategy,
                    parameters: exact.key.parameters,
                    tier: exact.key.tier,
                });
                evidence.push(exact);
                candidate_bases.push(crate::memory_strategy::CandidateBasis::Measured);
                // A measured envelope is one observed number; it does not decompose.
                candidate_activation_bytes.push(None);
                candidate_charges_generic_headroom.push(false);
                candidate_precheck_available_gb.push(candidate_budget.effective_gb());
            }
        }
        if evidence.is_empty() {
            let minimum_required_host = admission
                .evidence
                .iter()
                // The smallest host that satisfies the reserve policy. The current-host enforced
                // sum is a useful diagnostic but not a portable minimum: the reserve changes when
                // the host capacity changes.
                .map(|candidate| candidate.minimum_host_bytes)
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
            return Ok(MlxRequestAdmission::Rejected(WorkerError::InvalidPayload(
                format!(
                    "{} request {}x{} count {} needs at least {:.2} GiB at its smallest verified \
                 MLX host boundary, but no exact candidate fits the live unified-memory budget\
                 {alternative}",
                    plan.model_id,
                    inputs.width,
                    inputs.height,
                    inputs.count.max(1),
                    minimum_required_host as f64 / BYTES_PER_GIB,
                ),
            )));
        }
    } else {
        use crate::memory_strategy::CandidateBasis;

        let capacity = 1 + additional_evidence.len() + synthesized_estimates.len();
        selections.reserve(capacity);
        evidence.reserve(capacity);
        candidate_bases.reserve(capacity);
        candidate_activation_bytes.reserve(capacity);
        candidate_charges_generic_headroom.reserve(capacity);
        candidate_precheck_available_gb.resize(capacity, None);
        selections.push(resident_selection);
        evidence.push(&resident);
        // The resident baseline IS the rung-0 weights+headroom floor estimate (sc-18096): its
        // peak source is unchanged, but it is now graded behind the estimate margin like every
        // other unmeasured candidate instead of at its raw guess.
        candidate_bases.push(CandidateBasis::EstimateFloor);
        // ...but its PEAK does not decompose here, so it declares no activation term (sc-22508).
        // This candidate's peak is `predicted_memory_peak_from_base(contract_base_peak_bytes(
        // request_total_peak_bytes(..)))`. Two of those three
        // steps can destroy the weights+headroom shape: `request_total_peak_bytes` returns the
        // `mage_flow` provider's own `generation_peak_gb` scalar on that route, and
        // `predicted_memory_peak_from_base` is a gen-core trait method the loaded provider owns
        // and may rewrite arbitrarily. Naming `generic_headroom_bytes` here would have declared a
        // term the peak need not contain — on the mage route, a term the peak DEMONSTRABLY does
        // not contain. `None` takes the policy's undeclared-floor arm, whose doc says the fix for
        // a lane that wants better is to declare a real split, not to widen a number.
        candidate_activation_bytes.push(None);
        candidate_charges_generic_headroom.push(!resident_peak_is_provider_scalar);
        selections.extend(additional_evidence.iter().map(|item| MemorySelection {
            strategy: item.key.strategy,
            parameters: item.key.parameters,
            tier: item.key.tier,
        }));
        evidence.extend(additional_evidence);
        candidate_bases.extend(additional_evidence.iter().map(|_| CandidateBasis::Measured));
        candidate_activation_bytes.extend(additional_evidence.iter().map(|_| None));
        candidate_charges_generic_headroom.extend(additional_evidence.iter().map(|_| false));
        for estimate in synthesized_estimates {
            selections.push(estimate.selection);
            evidence.push(&estimate.evidence);
            candidate_bases.push(estimate.basis);
            candidate_activation_bytes.push(estimate.unmodeled_activation_bytes);
            candidate_charges_generic_headroom
                .push(estimate.activation_term == Some(ImageFloorActivationTerm::GenericHeadroom));
        }
    }
    debug_assert_eq!(evidence.len(), candidate_bases.len());
    debug_assert_eq!(evidence.len(), candidate_activation_bytes.len());
    debug_assert_eq!(evidence.len(), candidate_charges_generic_headroom.len());
    // sc-25909 — OWNER-APPROVED E6 EXCEPTION (Michael, 2026-10-16): a reference request on a rung
    // whose activation term is the generic area-scaled headroom is priced by the provider's own
    // cache-shed request transient (`derived::request_transient_budget_bytes`, the bound the
    // generator holds the render to) wherever that exceeds the headroom's activation slice. At
    // <=1024^2 with many references on a packed tier (the q8/q4 DiT computes in f32) the generic
    // figure ignores reference count and dtype width and under-prices the render, so the rung's
    // priced activation becomes max(generic activation slice, derived transient) + the fixed
    // reserve. This deliberately grades the SHED ladder — prefill path, cached path and the
    // pre-load tier probe alike — so a request admitted before the epic may now shed to a further
    // rung or be refused with the ordinary "needs … GiB" reason. The correction itself never
    // changes the tier, the references or the size. Its tier consequence is the EXISTING one: a
    // request with no explicit tier pick whose requested tier no longer fits the probe goes through
    // the pre-existing downtier ladder (`downtier_candidate_tiers` / `choose_mlx_request_tier`,
    // bounded by `min_quality_floor`), exactly as any other over-budget request does; an explicit
    // tier pick is refused.
    // The generic headroom carries the held-back fixed OS/app reserve, which is not activation, so
    // only its activation slice is compared (as the floor arm's `unmodeled_activation_bytes` does).
    // Where the derived transient is at or under that slice (bf16, no references) the shortfall is
    // zero and the request is graded byte for byte as before.
    let generic_headroom_shortfall = |rung: &MemorySelection| -> u64 {
        if admission.path != AdmissionPath::Legacy || geometry.reference_count == 0 {
            return 0;
        }
        let forced_tile_edge = contract
            .engages_selection(rung, MemoryStrategy::BoundedDecode)
            .then_some(rung.parameters.decode_tile_edge)
            .flatten();
        let generic_activation_bytes = plan
            .generic_headroom_bytes(geometry)
            .saturating_sub(plan.fixed_reserve_bytes.min(plan.activation_headroom_bytes));
        prefix_kv_cache_shed_transient(forced_tile_edge).map_or(0, |transient| {
            transient.saturating_sub(generic_activation_bytes)
        })
    };
    let corrected_evidence = selections
        .iter()
        .zip(&evidence)
        .zip(&candidate_charges_generic_headroom)
        .map(|((selection, evidence), charges_generic_headroom)| {
            let shortfall = if *charges_generic_headroom {
                generic_headroom_shortfall(selection)
            } else {
                0
            };
            let request_extra = shortfall.saturating_add(reference_workspace_bytes);
            (request_extra > 0).then(|| {
                let mut corrected = (*evidence).clone();
                corrected.predicted_peak_bytes =
                    corrected.predicted_peak_bytes.saturating_add(request_extra);
                (corrected, shortfall)
            })
        })
        .collect::<Vec<_>>();
    let candidates = selections
        .iter()
        .zip(evidence)
        .zip(&candidate_bases)
        .zip(&candidate_activation_bytes)
        .zip(&corrected_evidence)
        .map(
            |((((selection, evidence), basis), activation), corrected)| Candidate {
                selection: *selection,
                evidence: corrected
                    .as_ref()
                    .map_or(evidence, |(corrected, _)| corrected),
                basis: *basis,
                // sc-22508: carried from the site that BUILT each peak (see
                // `candidate_activation_bytes`), never inferred from the basis label here. A
                // `CandidateBasis::EstimateFloor` label says how much evidence stands behind a
                // number; it does not say the number is `weights + generic_headroom_bytes`.
                // sc-25909: a corrected rung's activation carries its shortfall too.
                unmodeled_activation_bytes: match corrected {
                    Some((_, shortfall)) => {
                        activation.map(|bytes| bytes.saturating_add(*shortfall))
                    }
                    None => *activation,
                },
            },
        )
        .collect::<Vec<_>>();
    let request_scope = RequestScope {
        resolved_route: plan.engine_id,
        backend: "mlx",
        tier: plan.tier,
        mode: mode_key,
        overlay: inputs.overlay.as_deref(),
        geometry,
    };
    let selector_budget = Some(Budget {
        available_gb: budget.total_bytes.saturating_sub(budget.committed_bytes) as f64
            / BYTES_PER_GIB,
        reclaimable_gb: budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
        total_gb: budget.total_bytes as f64 / BYTES_PER_GIB,
        reserved_headroom_gb: if admission.path == AdmissionPath::Evidence {
            0.0
        } else {
            budget.reserved_headroom_bytes as f64 / BYTES_PER_GIB
        },
    });
    let resident_allowance_credit = if admission.path == AdmissionPath::Legacy {
        if phase == MlxAdmissionPhase::TierProbe && load_policy == OffloadPolicy::Resident {
            // A cold probe charges the FULL pipeline and gives no live allocation credit. Its
            // Resident uncertainty allowance must nevertheless exclude the declared constructor
            // weights, just as the loaded gate excludes weights already present. Otherwise moving
            // this decision before load inflates the allowance and falsely rejects fitting tiers.
            // This affects only Resident's margin, never the peak/budget or optimized candidates;
            // the constructor floor and final live-allocation check still apply independently.
            contract.total_resident_bytes()
        } else {
            attributable_resident_bytes
        }
    } else {
        0
    };
    // Epic 25899: the provider's prefix KV cache, added rung-for-rung. The whole ladder below —
    // baseline and warm-policy floor alike — runs first at the shed demand, which is exactly the
    // pre-cache admission and picks the rung; the cache is then taken only if that SAME rung
    // (`only_rung`) also fits with the surcharge on its peak. The cache never moves a request to
    // another rung. A provider with no cache prices `None` and runs the ladder once, unchanged.
    let run_ladder = |extra_bytes: u64,
                      only_rung: Option<MemorySelection>|
     -> (Selection, Option<crate::execution_planner::GrantOutcome>) {
        let surcharged_evidence = (extra_bytes > 0).then(|| {
            candidates
                .iter()
                .map(|candidate| {
                    let mut evidence = candidate.evidence.clone();
                    evidence.predicted_peak_bytes =
                        evidence.predicted_peak_bytes.saturating_add(extra_bytes);
                    evidence
                })
                .collect::<Vec<_>>()
        });
        let surcharged_candidates = surcharged_evidence.as_ref().map(|surcharged| {
            candidates
                .iter()
                .zip(surcharged)
                .zip(&candidate_precheck_available_gb)
                // The Evidence path's captured-reserve pre-check, re-run at the surcharged peak:
                // a measured cell that fitted shed but not cached is not a cached candidate.
                .filter(|((_, evidence), precheck)| {
                    precheck.is_none_or(|available| {
                        evidence.predicted_peak_bytes as f64 / BYTES_PER_GIB <= available
                    })
                })
                .map(|((candidate, evidence), _)| Candidate {
                    evidence,
                    ..*candidate
                })
                .collect::<Vec<_>>()
        });
        let pass_candidates: &[Candidate<'_>] =
            surcharged_candidates.as_deref().unwrap_or(&candidates);
        let same_rung;
        let pass_candidates: &[Candidate<'_>] = match only_rung {
            Some(rung) => {
                same_rung = pass_candidates
                    .iter()
                    .filter(|candidate| candidate.selection == rung)
                    .copied()
                    .collect::<Vec<_>>();
                &same_rung
            }
            None => pass_candidates,
        };
        let baseline = crate::memory_strategy::select_strategy_with_resident_credit(
            request_scope,
            contract,
            selector_budget,
            pass_candidates,
            resident_allowance_credit,
        );
        // sc-18317: a GRANTED warm-policy switch takes effect HERE or nowhere.
        //
        // `select_strategy` picks the first fitting candidate in resident -> staged -> bounded-decode
        // order, so a request whose weights fit resident never stages no matter what policy the caller
        // asked for. Honoring a grant therefore means removing the looser candidates from the set: offer
        // only those that engage `StagedResidency` and let the selector apply the SAME budget and margins
        // it always does. A floored selection that comes back `Selected` has passed exactly the fit check
        // the baseline would have had to pass, so it cannot exceed the admitted envelope — monotonicity is
        // preserved by construction rather than asserted after the fact.
        //
        // The outcome (not the seam's proposal) is what gets reported: `rematerialized` is emitted only
        // when the floor actually moved the selection.
        let outcome = if warm_policy.requires_staged_residency() {
            let staged = pass_candidates
                .iter()
                .filter(|candidate| {
                    contract
                        .engages_selection(&candidate.selection, MemoryStrategy::StagedResidency)
                })
                .cloned()
                .collect::<Vec<_>>();
            let baseline_stages = matches!(&baseline, Selection::Selected { selection, .. }
                if contract.engages_selection(selection, MemoryStrategy::StagedResidency));
            if baseline_stages {
                (
                    baseline,
                    Some(crate::execution_planner::GrantOutcome::AlreadyStaged),
                )
            } else if staged.is_empty() {
                (
                    baseline,
                    Some(crate::execution_planner::GrantOutcome::NoStagedCandidate),
                )
            } else {
                let floored = crate::memory_strategy::select_strategy(
                    request_scope,
                    contract,
                    selector_budget,
                    &staged,
                );
                match &floored {
                    // A fitting staged selection, and it necessarily differs from the baseline's: control
                    // only reaches here when the baseline does NOT engage staged residency, while every
                    // candidate offered to this call does. Asserted rather than re-tested, so the
                    // reasoning is checked in debug builds instead of hidden in an unreachable arm.
                    Selection::Selected {
                        selection: chosen, ..
                    } => {
                        debug_assert!(
                            !matches!(&baseline, Selection::Selected { selection: base, .. } if base == chosen),
                            "a floored staged selection cannot equal a non-staging baseline"
                        );
                        (
                            floored,
                            Some(crate::execution_planner::GrantOutcome::SelectionMovedToStaged),
                        )
                    }
                    // The staged candidates did not fit this request's budget, or their evidence is
                    // structurally unusable. The baseline stands; not a refusal, just the ordinary fit
                    // check every selection passes.
                    _ => (
                        baseline,
                        Some(crate::execution_planner::GrantOutcome::StagedCandidateDidNotFit),
                    ),
                }
            }
        } else {
            (baseline, None)
        };
        outcome
    };
    let ((selection, grant_outcome), prefix_kv_cache) =
        crate::execution_planner::plan_prefix_kv_cache(
            prefix_kv_cache_surcharge,
            run_ladder(0, None),
            |(selection, _)| matches!(selection, Selection::Selected { .. }),
            |(shed, grant_outcome), priced_bytes| {
                let Selection::Selected {
                    selection: rung, ..
                } = shed
                else {
                    return None;
                };
                // The rung's peak already carries its sc-25909 transient shortfall (corrected
                // above), so the cached path is that graded shed rung + the cache surcharge.
                let (cached, _) = run_ladder(priced_bytes, Some(*rung));
                // The shed pass's warm-policy outcome stands: it decided the rung.
                matches!(&cached, Selection::Selected { selection, .. } if selection == rung)
                    .then_some((cached, *grant_outcome))
            },
        );
    // The demand a cached selection was graded at on top of its (corrected) shed peak.
    let prefix_kv_cache_extra = prefix_kv_cache.map_or(0, |plan| plan.admitted_surcharge_bytes());
    warm_policy.settle_with_selection(
        grant_outcome.unwrap_or(crate::execution_planner::GrantOutcome::AlreadyStaged),
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
            return Ok(MlxRequestAdmission::Rejected(WorkerError::InvalidPayload(format!(
                "{} request {}x{} count {} needs {:.2} GiB for the complete pipeline but only {:.2} GiB is safely available \
                 ({:.2} GiB total, {:.2} GiB unrelated active allocations, {:.2} GiB reserved){}",
                plan.engine_id,
                inputs.width,
                inputs.height,
                inputs.count.max(1),
                needed_gb,
                available_gb,
                budget.total_bytes as f64 / BYTES_PER_GIB,
                budget.committed_bytes as f64 / BYTES_PER_GIB,
                budget.reserved_headroom_bytes as f64 / BYTES_PER_GIB,
                alternative,
            ))));
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
    let selected_basis = candidates
        .iter()
        .find(|candidate| candidate.selection == selection)
        .map(|candidate| candidate.basis)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{} selected a strategy with no candidate provenance",
                plan.engine_id
            ))
        })?;
    let optimization_authority = if !selection.strategy.is_optimized() {
        MemoryOptimizationAuthority::Resident
    } else if selected_basis.is_estimate() {
        MemoryOptimizationAuthority::Estimated
    } else {
        MemoryOptimizationAuthority::Calibrated
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
        needed_gb = evidence
            .predicted_peak_bytes
            .saturating_add(reference_workspace_bytes)
            .saturating_add(prefix_kv_cache_extra)
            .saturating_add(reserve) as f64
            / BYTES_PER_GIB;
        available_gb = selected_budget.effective_gb().unwrap_or(0.0);
        budget.reserved_headroom_bytes = reserve;
        process_limit_bytes = Some(budget.total_bytes.saturating_sub(reserve));
        selected_record_id = Some(candidate.record_id.clone());
        evidence
            .predicted_peak_bytes
            .saturating_add(reference_workspace_bytes)
    } else if selection == resident_selection
        || synthesized_estimates
            .iter()
            .any(|estimate| estimate.selection == selection)
    {
        // Keep the selected full peak paired with the normalized budget used by selection: the
        // peak of the candidate the ladder GRADED (the same lookup `selected_basis` uses), which
        // carries its sc-25909 transient shortfall where the rung took one — one source of truth.
        candidates
            .iter()
            .find(|candidate| candidate.selection == selection)
            .map_or(
                predicted_peak_bytes.saturating_add(reference_workspace_bytes),
                |candidate| candidate.evidence.predicted_peak_bytes,
            )
    } else {
        predicted_peak_bytes.saturating_add(reference_workspace_bytes)
    };
    // The demand the admission approved: a cached selection was graded at its (corrected) shed
    // peak + the cache surcharge (`prefix_kv_cache_extra`).
    let predicted_peak_bytes = predicted_peak_bytes.saturating_add(prefix_kv_cache_extra);
    tracing::info!(
        event = if phase == MlxAdmissionPhase::TierProbe {
            "memory_strategy_tier_candidate_selected"
        } else {
            "memory_strategy_request_selected"
        },
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
        cache_eviction = contract.engages_selection(
            &selection,
            MemoryStrategy::StagedResidency,
        ),
        parameters = ?selection.parameters,
        evidence_record_id = selected_record_id.as_deref().unwrap_or("none"),
        predicted_peak_bytes,
        effective_budget_bytes = budget.effective_bytes(),
        needed_gb,
        available_gb,
        optimization_authority = ?optimization_authority,
        prefix_kv_cache = prefix_kv_cache.map(|plan| plan.mode.label()),
        prefix_kv_cache_priced_bytes = prefix_kv_cache.map(|plan| plan.priced_bytes),
        "selected request-scoped MLX memory strategy"
    );
    let mut memory = memory_for_selection(contract, selection);
    if let Some(plan) = prefix_kv_cache {
        plan.write_onto(&mut memory);
    }
    let evaluation = MlxRequestEvaluation {
        memory,
        prefix_kv_cache,
        process_limit_bytes,
        decode_quality_decisions: synthesized_ladder.decode_quality_decisions,
        context: MemoryRunContext {
            selection,
            optimization_authority,
            calibration_abi,
            calibration_fingerprint: calibration_fingerprint.unwrap_or_default().to_owned(),
            load_shape: contract.load_shape,
            mode,
            has_reference: inputs.has_reference,
            use_pid: inputs.use_pid,
            has_phases: inputs.has_phases,
            geometry,
            reference_resources: inputs.reference_resources.clone(),
            overlay: inputs.overlay.clone(),
            budget,
            predicted_peak_bytes,
            cache_state,
            evidence_revision: selected_record_id
                .or(admission.evidence_revision)
                .unwrap_or_else(|| REQUEST_EVIDENCE_REVISION.to_owned()),
        },
    };
    tracing::info!(
        event = if phase == MlxAdmissionPhase::TierProbe {
            "mlx_decode_quality_tier_candidate_audit"
        } else {
            "mlx_decode_quality_request_audit"
        },
        route = plan.engine_id,
        decisions = ?evaluation.decode_quality_decisions,
        "recorded typed request-scoped decode-quality decisions"
    );
    Ok(MlxRequestAdmission::Admitted(Box::new(evaluation)))
}

#[cfg(target_os = "macos")]
pub(crate) fn live_request_budget(engine_id: &str) -> WorkerResult<MemoryBudget> {
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

/// Preserve the constructor's weights floor for eager/unadopted loads. Sequential specs have
/// already been authorized by the exact declaration or the existing load gate. This diagnostic
/// describes the limiting load working set, not the unrelated resident activation estimate.
#[cfg(target_os = "macos")]
pub(crate) fn preflight_load_rejection(engine_id: &str, spec: &LoadSpec) -> Option<WorkerError> {
    if spec.offload_policy == OffloadPolicy::Sequential {
        return None;
    }
    let ResidencyOutcome::Reject { available_gb, .. } = decide_residency_for_spec(engine_id, spec)
    else {
        return None;
    };
    let (total, text, _) = spec_component_bytes(engine_id, spec);
    let staged = engine_supports_sequential(engine_id);
    let weights = if staged {
        staged_weights_gb(total, text)
    } else {
        total as f64 / BYTES_PER_GIB
    };
    let reserve = crate::fit_gate::legacy_unified_reserve(available_gb).gb;
    Some(WorkerError::InvalidPayload(format!(
        "{engine_id} cannot load this tier: its {} model weights need {weights:.2} GiB, \
         but only {:.2} GiB is available after the OS reserve ({available_gb:.2} GiB total). \
         No supported deferred load was selected; use a smaller installed tier or a Mac with more memory.",
        if staged { "largest simultaneous component set of" } else { "resident" },
        (available_gb - reserve).max(0.0),
    )))
}

/// Evaluate a fully prepared candidate without constructing a generator or allocating tensors.
/// The registry contract is queried against the same spec handed to the cache on selection.
#[cfg(target_os = "macos")]
pub(crate) fn preflight_request(
    spec: &LoadSpec,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    budget: MemoryBudget,
) -> WorkerResult<MlxRequestAdmission> {
    // Fail closed before deriving a contract for this actual candidate spec. The selector also
    // resolves the same pure provider bound before grading, including every warm-cache request.
    crate::memory_strategy::reference_workspace_bytes(
        plan.engine_id,
        spec,
        inputs.reference_resources.as_ref(),
    )?;
    let contract = crate::inference_runtime::media()
        .memory_strategy_contract(plan.engine_id, spec)
        .map_err(|error| {
            crate::classify_engine_error("MLX pre-load memory contract failed", error)
        })?;
    preflight_request_with_contract(
        contract.as_ref(),
        spec.offload_policy,
        plan,
        inputs,
        budget,
        Some(ReferencePricing {
            spec,
            registry: crate::inference_runtime::media(),
        }),
    )
}

/// [`preflight_request`] once the registry contract is resolved — split out so the tier probe's
/// pricing is testable against a fixture contract without a `LoadSpec` or the linked registry.
#[cfg(target_os = "macos")]
fn preflight_request_with_contract(
    contract: Option<&MemoryProviderContract>,
    load_policy: OffloadPolicy,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    budget: MemoryBudget,
    reference_pricing: Option<ReferencePricing<'_>>,
) -> WorkerResult<MlxRequestAdmission> {
    select_request_with_budget_using_bundle(
        MlxAdmissionPhase::TierProbe,
        declared_component_floors(plan.engine_id),
        contract,
        None,
        // sc-25909: the probe grades exactly the shed demand, which now carries the provider
        // transient shortfall — a tier the loaded gate would refuse is not probed as fitting.
        &|forced_tile_edge| {
            mlx_prefix_kv_cache_shed_transient_bytes(
                plan.engine_id,
                plan.tier.quant,
                inputs,
                forced_tile_edge,
            )
        },
        |base| {
            contract.map_or(
                gen_core::MemoryPeakBreakdown::from_unattributed(base),
                |contract| contract.predicted_peak_from_base(base),
            )
        },
        plan,
        inputs,
        MemoryCacheState::Cold,
        load_policy,
        crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
        budget,
        request_total_peak_bytes(plan, request_geometry(inputs)),
        0,
        &[],
        None,
        reference_pricing,
    )
}

/// Evaluate one real MLX request after cache lookup and immediately before generation.
#[cfg(target_os = "macos")]
pub(crate) fn evaluate_request(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    external_committed_bytes: u64,
) -> WorkerResult<MlxRequestEvaluation> {
    if inputs.reference_resources.is_some() {
        return Err(missing_reference_pricing(plan.engine_id));
    }
    let geometry = request_geometry(inputs);
    let budget = live_request_budget(plan.engine_id)?;
    let total_peak_bytes = request_total_peak_bytes(plan, geometry);
    evaluate_request_with_budget_and_warm_policy(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        warm_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        &[],
        None,
    )
}

/// Loaded FLUX.3 reference request: price against the same resolved spec used for this cache lookup.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_request_with_reference_spec(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    external_committed_bytes: u64,
    reference_spec: &LoadSpec,
) -> WorkerResult<MlxRequestEvaluation> {
    let geometry = request_geometry(inputs);
    let budget = live_request_budget(plan.engine_id)?;
    let total_peak_bytes = request_total_peak_bytes(plan, geometry);
    evaluate_request_with_budget_and_warm_policy(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        warm_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        &[],
        Some(ReferencePricing {
            spec: reference_spec,
            registry: crate::inference_runtime::media(),
        }),
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

/// Whether `engine_id` physically stages heavyweight phases, either because the provider exposes
/// the selectable [`OffloadPolicy::Sequential`] control or because it does so unconditionally.
///
/// This is static descriptor truth for publication and capability reporting. Load-time selection
/// must continue to use [`engine_supports_sequential`]: an unconditional-only provider does not
/// consume `OffloadPolicy::Sequential`, so treating this broader predicate as authorization to set
/// that control would turn an accurate declaration into a no-op request.
#[cfg(test)]
pub(crate) fn engine_engages_staged_residency(engine_id: &str) -> bool {
    crate::inference_runtime::media()
        .generators()
        .find(|reg| (reg.descriptor)().id == engine_id)
        .is_some_and(|reg| {
            (reg.descriptor)()
                .capabilities
                .staged_residency_availability()
                != StagedResidencyAvailability::Absent
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
///
/// Declared in `sceneworks_core::memory_anchor` since sc-22738 so the memory adapter's LTX-2.3
/// admission floors its projection with the SAME allowance the video gate floors with
/// (`video_admission::floor_phase_peaks`); this is the worker's read of it, not a second
/// declaration.
const HEADROOM_GB: f64 = sceneworks_core::memory_anchor::MLX_GENERIC_HEADROOM_GB;
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
pub(crate) const OS_APP_RESERVE_GB: f64 = 4.0;

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

/// The numeric tier a `LoadSpec` loads, exactly as [`MlxRequestPlan::for_spec_and_manifest`]
/// derives it (sc-18814 factored it out so the video gate cannot drift into a second derivation).
/// The manifest-driven `fixed_artifact_tier` override is deliberately NOT applied here: a video
/// route carries no calibration binding to resolve one from.
pub(crate) fn spec_numeric_tier(engine_id: &str, spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: active_component_floors(
            declared_component_floors(engine_id),
            spec.quantize,
        ),
    }
}

/// Resolve the numeric tier that the loaded video checkpoint actually carries.
///
/// Provider-owned resolution runs first for video loaders that carry their tier outside
/// `LoadSpec.quantize` (Wan TI2V-5B and Bernini). LTX then resolves its immutable `split_model.json`.
/// An explicit assertion that disagrees with the checkpoint fails closed before selection.
pub(crate) fn resolved_video_numeric_tier(
    engine_id: &str,
    spec: &LoadSpec,
) -> WorkerResult<MemoryNumericTier> {
    // Provider-owned load-exact video tiers win whenever the selected runtime exposes one. Wan and
    // Bernini read immutable packed config/header surfaces (Wan also carries the Q4 text-encoder Q8
    // floor); deriving from `LoadSpec.quantize` would misprice their normal prepacked loads, whose
    // request-side quant hint is deliberately `None`.
    #[cfg(target_os = "macos")]
    if let Some(tier) =
        runtime_macos::resolved_video_memory_numeric_tier(engine_id, spec).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "{engine_id} admission cannot resolve the provider-owned numeric tier: {error}"
            ))
        })?
    {
        return Ok(tier);
    }
    if engine_id != "ltx_2_3" {
        return Ok(spec_numeric_tier(engine_id, spec));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path,
        WeightsSource::File(path) => {
            return Err(WorkerError::InvalidPayload(format!(
                "ltx_2_3 admission expected a model directory, got {}",
                path.display()
            )));
        }
    };
    let manifest_path = root.join("split_model.json");
    let resolved_quant = if manifest_path.exists() {
        let raw = std::fs::read_to_string(&manifest_path).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "ltx_2_3 admission cannot read {}: {error}",
                manifest_path.display()
            ))
        })?;
        let value: Value = serde_json::from_str(&raw).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "ltx_2_3 admission cannot parse {}: {error}",
                manifest_path.display()
            ))
        })?;
        if value
            .get("quantized")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            match value
                .get("quantization_bits")
                .and_then(Value::as_i64)
                .unwrap_or(4)
            {
                4 => Some(gen_core::Quant::Q4),
                8 => Some(gen_core::Quant::Q8),
                bits => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "ltx_2_3 admission refuses unsupported packed checkpoint tier q{bits} in {}",
                        manifest_path.display()
                    )));
                }
            }
        } else {
            None
        }
    } else {
        None
    };
    if spec.quantize != resolved_quant && spec.quantize.is_some() {
        return Err(WorkerError::InvalidPayload(format!(
            "ltx_2_3 admission quant assertion {:?} disagrees with checkpoint tier {:?} in {}",
            spec.quantize,
            resolved_quant,
            manifest_path.display()
        )));
    }
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant: resolved_quant,
        component_precision_floors: active_component_floors(
            declared_component_floors(engine_id),
            resolved_quant,
        ),
    })
}

/// The activation-headroom allowance this load already charges — the same
/// [`HeadroomAllowance`] the load-time residency gate uses (sc-18814 exposes it so the video gate
/// reuses that number instead of inventing one).
pub(crate) fn spec_headroom_bytes(engine_id: &str, spec: &LoadSpec) -> u64 {
    gib_to_bytes(spec_component_bytes(engine_id, spec).2.total_gb)
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
    crate::fit_gate::fit_decision(needed_gb, budget.map(|budget| budget.total_gb))
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
    crate::fit_gate::sequential_overflow_gb(
        sequential_needed_gb,
        budget.map(|budget| budget.total_gb),
    )
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

#[derive(Clone, Debug)]
struct ResolvedTextEncoderSource {
    path: PathBuf,
    direct_shard_bytes: u64,
}

fn text_encoder_discovery_roots(source: &WeightsSource) -> gen_core::Result<Vec<PathBuf>> {
    let entry = std::path::absolute(crate::weights_source::WeightsSourceExt::path(source))?;
    let lexical_root = match source {
        WeightsSource::Dir(_) => entry.clone(),
        WeightsSource::File(_) => entry.parent().map(Path::to_path_buf).ok_or_else(|| {
            gen_core::Error::Unsupported(
                "selected text-encoder file has no lexical parent directory".into(),
            )
        })?,
    };
    let canonical = std::fs::canonicalize(&entry)?;
    let canonical_root = match source {
        WeightsSource::Dir(_) => canonical,
        WeightsSource::File(_) => canonical.parent().map(Path::to_path_buf).ok_or_else(|| {
            gen_core::Error::Unsupported(
                "selected text-encoder file has no canonical parent directory".into(),
            )
        })?,
    };
    let mut roots = vec![lexical_root, canonical_root];
    roots.sort();
    roots.dedup();
    Ok(roots)
}

/// Resolve path-free planning facts exactly as gen-core's loader does. A prepared spec reuses its
/// retained acquisition receipt; compatibility mode uses bounded metadata discovery. Neither path
/// acquires or discards an executable artifact seal.
fn resolved_text_encoder_source(
    spec: &LoadSpec,
    source: &WeightsSource,
) -> gen_core::Result<ResolvedTextEncoderSource> {
    let facts = match spec.prepared_text_encoder_planning_facts()? {
        Some(facts) => facts,
        None => gen_core::text_encoder_planning_facts_for_discovery(
            source,
            &text_encoder_discovery_roots(source)?,
        )?,
    };
    let path = match (source, facts.source_layout()) {
        (WeightsSource::File(path), gen_core::TextEncoderSourceLayout::File)
        | (WeightsSource::Dir(path), gen_core::TextEncoderSourceLayout::DirectDirectory) => {
            path.clone()
        }
        (WeightsSource::Dir(path), gen_core::TextEncoderSourceLayout::CompleteSnapshot) => {
            path.join("text_encoder")
        }
        _ => {
            return Err(gen_core::Error::Unsupported(
                "selected text-encoder planning layout does not match its LoadSpec source".into(),
            ));
        }
    };
    Ok(ResolvedTextEncoderSource {
        path,
        direct_shard_bytes: facts.direct_shard_bytes(),
    })
}

fn adapter_path_bytes(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => sum_safetensors_bytes(path),
        Ok(metadata)
            if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") =>
        {
            metadata.len()
        }
        _ => 0,
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
        let bytes = adapter_path_bytes(&adapter.path);
        total.saturating_add(bytes)
    });
    adapter_resident_source_bytes(engine_id, spec, source_bytes)
}

fn adapter_resident_source_bytes(engine_id: &str, spec: &LoadSpec, source_bytes: u64) -> u64 {
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

fn prepared_file_sizes(spec: &LoadSpec) -> gen_core::Result<BTreeMap<PathBuf, u64>> {
    spec.validate_prepared_file_pins()?;
    if !spec.prepared_file_pins().is_prepared() {
        return Err(gen_core::Error::Unsupported(
            "prepared accounting requires a finalized prepared LoadSpec".into(),
        ));
    }
    Ok(spec
        .prepared_file_pins()
        .iter()
        .map(|(path, pin)| (path.clone(), pin.target_fingerprint().size))
        .collect())
}

fn prepared_source_bytes(source: &WeightsSource, file_sizes: &BTreeMap<PathBuf, u64>) -> u64 {
    match source {
        WeightsSource::Dir(dir) => sum_safetensors_bytes(dir),
        WeightsSource::File(file) => std::path::absolute(file)
            .ok()
            .and_then(|file| file_sizes.get(&file).copied())
            .unwrap_or(0),
    }
}

fn prepared_adapter_source_bytes_for_gate_where(
    engine_id: &str,
    spec: &LoadSpec,
    file_sizes: &BTreeMap<PathBuf, u64>,
    include: impl Fn(&Path) -> bool,
) -> u64 {
    let source_bytes = spec.adapters.iter().fold(0_u64, |total, adapter| {
        if include(&adapter.path) {
            let bytes = std::path::absolute(&adapter.path)
                .ok()
                .and_then(|path| file_sizes.get(&path).copied())
                .unwrap_or(0);
            total.saturating_add(bytes)
        } else {
            total
        }
    });
    adapter_resident_source_bytes(engine_id, spec, source_bytes)
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

/// Total unified memory (GiB) via the shared bounded physical-memory probe, cached process-wide (physical RAM
/// never changes at runtime). The blocking sibling of `gpu::total_unified_memory_gb` — the gate runs
/// on the generator-cache thread, which is already blocking on the weight load, so the bounded
/// subprocess read does not require a Tokio runtime. `None` off macOS or on probe failure retains
/// fail-open admission; failures are retried so a transient launch failure cannot disable the gate.
pub(crate) fn probe_total_unified_memory_gib() -> Option<f64> {
    probe_total_unified_memory_bytes().map(|bytes| bytes as f64 / BYTES_PER_GIB)
}

/// Total unified memory (bytes) — the raw probe [`probe_total_unified_memory_gib`] scales. Cached
/// process-wide after a successful bounded read, so the `sysctl` subprocess is shared however many
/// callers ask. Also read by [`crate::generator_cache::apply_gpu_memory_limit`] at worker startup,
/// which needs the byte figure rather than the GiB one.
pub(crate) fn probe_total_unified_memory_bytes() -> Option<u64> {
    crate::gpu::total_unified_memory_bytes()
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
    crate::fit_gate::exceeds_budget(needed_gb, budget.total_gb).then(|| {
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
        MemoryBackendRealization, MemoryEvidenceKey, MemoryGeometry, MemoryNumericTier,
        MemoryProviderContract, MemorySelection, MemoryStrategy,
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
    let headroom_bytes = (headroom_gb * BYTES_PER_GIB)
        .max(0.0)
        .ceil()
        .clamp(0.0, u64::MAX as f64) as u64;
    let evidence = sceneworks_core::memory_contracts::unverified_estimate(
        MemoryEvidenceKey {
            model_family: "generic_mlx_cold_load".into(),
            resolved_route: "generic_mlx_cold_load".into(),
            backend: gen_core::MemoryBackend::Mlx,
            tier,
            // The generic estimator models the historical bulk cold load; no generic route
            // defers transformer materialization.
            load_shape: gen_core::LoadShape::EagerMaterialization,
            mode: memory_mode_from_mode_key("image_generation"),
            reference_shape: MemoryReferenceShape::None,
            overlay: Some("resolved_load_spec".into()),
            geometry,
            frames_per_second: None,
            strategy: MemoryStrategy::Resident,
            engaged_composition: vec![MemoryStrategy::Resident],
            parameters: Default::default(),
        },
        gen_core::MEMORY_CALIBRATION_ABI,
        String::new(),
        "sc-15449-contract-v1".into(),
        // sc-22508: the peak is the WHOLE floor — resolved-spec weights plus the activation
        // headroom — because the allowance declared below must be a slice of the peak it grades,
        // not a second charge sitting outside it (`memory_strategy::Candidate
        // ::unmodeled_activation_bytes`). Headroom used to live only in the budget's
        // `reserved_headroom_gb`, which made the declaration an out-of-band charge.
        //
        // Moving it is exactly value-preserving, not a policy change: writing the MLX allowance as
        // A (`ladder_margin_policy::FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE`), the comparison was
        // `weights + A*headroom <= total - headroom` and is now
        // `(weights + headroom) + A*headroom <= total`, the same inequality for ANY A —
        // which is the point of writing it symbolically: the identity is independent of whatever
        // the corpus derives A to be, and the earlier prose that spelled it `0.17` went stale the
        // moment that derivation moved. `reserved_headroom_gb` is zeroed below so headroom is
        // counted once.
        total_bytes.saturating_add(headroom_bytes),
    );
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
        },
        &contract,
        budget.map(|budget| Budget {
            available_gb: budget.total_gb,
            reclaimable_gb: 0.0,
            total_gb: budget.total_gb,
            // Zero, because the headroom is now IN the candidate's peak (see the peak's comment).
            // Leaving it here as well would charge the term twice.
            reserved_headroom_gb: 0.0,
        }),
        &[Candidate {
            selection,
            evidence: &evidence,
            // The generic cold-load estimate is exactly a weights+headroom floor (sc-18096).
            basis: crate::memory_strategy::CandidateBasis::EstimateFloor,
            // sc-22508: the headroom half of that floor is the only uncertain term; the weights
            // half is the resolved load spec's counted bytes. This is a genuine slice of
            // `predicted_peak_bytes` above, which is what the field's contract requires.
            unmodeled_activation_bytes: Some(headroom_bytes),
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
    crate::memory_route_registry::apply_registered_load_shape(
        crate::memory_route_registry::MemoryRouteBackend::Mlx,
        engine_id,
        crate::memory_route_registry::MemoryRouteMode::TextToImage,
        spec,
        true,
    )
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
    if spec.prepared_file_pins().is_prepared() {
        spec.validate_prepared_file_pins().map_err(|error| {
            crate::classify_engine_error("MLX residency source validation failed", error)
        })?;
    }
    // MOSS's generic stream admits the authenticated SELECTED precision closure plus request
    // workspace inside the cache transaction. The generic source sum includes the unselected SFT
    // transformer during full-FP32 acceleration and must not choose a different load policy.
    if sceneworks_core::moss_image::is_moss(engine_id) {
        return Ok(spec);
    }
    // Respect an offload policy already chosen upstream (defensive: the MLX cache seam normally sees
    // the default `Resident`, but never downgrade a `Sequential` set by another gate).
    if spec.offload_policy == OffloadPolicy::Sequential {
        return Ok(spec);
    }
    let outcome = if spec.prepared_file_pins().is_prepared() {
        try_decide_residency_for_spec(engine_id, &spec)?
    } else {
        decide_residency_for_spec(engine_id, &spec)
    };
    match outcome {
        ResidencyOutcome::Resident => Ok(spec),
        ResidencyOutcome::Sequential => {
            let (total_bytes, te_bytes, _) = if spec.prepared_file_pins().is_prepared() {
                spec_component_bytes_checked(engine_id, &spec)?
            } else {
                spec_component_bytes(engine_id, &spec)
            };
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

fn spec_component_bytes_checked(
    engine_id: &str,
    spec: &LoadSpec,
) -> WorkerResult<(u64, u64, HeadroomAllowance)> {
    let footprint = crate::inference_runtime::media()
        .footprint(engine_id, spec)
        .map_err(|error| crate::classify_engine_error("MLX footprint query failed", error))?;
    spec_component_bytes_with_provider_footprint_checked(engine_id, spec, footprint).map_err(
        |error| crate::classify_engine_error("MLX prepared source accounting failed", error),
    )
}

/// Provider-injected core of [`spec_component_bytes`]. The live path obtains `footprint` from the
/// active registry; keeping the arithmetic independent of registry composition lets platform-neutral
/// tests exercise the exact Lens materialization and overlay accounting used on macOS.
fn spec_component_bytes_with_provider_footprint(
    engine_id: &str,
    spec: &LoadSpec,
    footprint: Option<PerComponentBytes>,
) -> (u64, u64, HeadroomAllowance) {
    let external_adapter_bytes = external_adapter_source_bytes_for_gate(engine_id, spec);
    // The infallible compatibility/planning seam predates prepared specs. Valid selected encoders
    // resolve through gen-core; an invalid ad-hoc spec retains the old conservative recursive
    // estimate and will still fail contract validation before a real provider load.
    let selected_text_encoder = spec.text_encoder.as_ref().map(|source| {
        resolved_text_encoder_source(spec, source).unwrap_or_else(|_| ResolvedTextEncoderSource {
            path: crate::weights_source::WeightsSourceExt::path(source).to_path_buf(),
            direct_shard_bytes: weights_source_bytes(source),
        })
    });
    spec_component_bytes_with_provider_footprint_and_sizes(
        engine_id,
        spec,
        footprint,
        selected_text_encoder.as_ref(),
        weights_source_bytes,
        external_adapter_bytes,
        adapter_path_bytes,
    )
}

fn spec_component_bytes_with_provider_footprint_and_sizes(
    engine_id: &str,
    spec: &LoadSpec,
    footprint: Option<PerComponentBytes>,
    selected_text_encoder: Option<&ResolvedTextEncoderSource>,
    source_bytes: impl Fn(&WeightsSource) -> u64,
    external_adapter_bytes: u64,
    adapter_source_bytes: impl Fn(&Path) -> u64,
) -> (u64, u64, HeadroomAllowance) {
    let footprint_te = footprint.map(|fp| fp.text_encoder);
    // A provider that accepts a primary single-file checkpoint owns the meaning of its named
    // companions. Its footprint is the exact consumed assembly (imported DiT + companion TE/VAE),
    // not a split of `spec.weights` alone. Treat it as authoritative so a `base_snapshot` supplying
    // tokenizer/config does not recursively re-price the snapshot transformer the primary File
    // replaces.
    let provider_owned_file_footprint = matches!(&spec.weights, WeightsSource::File(_))
        .then_some(footprint)
        .flatten();
    let mut headroom = HeadroomAllowance::GENERIC;
    let (mut total_bytes, mut te_bytes) = match &spec.weights {
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
        // A single-file source has no component tree of its own. A File-aware provider footprint
        // prices the complete consumed assembly; otherwise keep the primary file here and add only
        // the well-known companion subtrees below.
        WeightsSource::File(_) => provider_owned_file_footprint.map_or_else(
            || (source_bytes(&spec.weights), footprint_te.unwrap_or(0)),
            |components| {
                (
                    components
                        .text_encoder
                        .saturating_add(components.dit)
                        .saturating_add(components.vae),
                    components.text_encoder,
                )
            },
        ),
    };
    // Every directory source in this set was priced recursively. A selected encoder at the same
    // path or below one is already covered; a selected encoder above one is not, because its exact
    // loader inventory contains direct shards only and excludes nested descendants.
    let mut recursively_priced_paths =
        vec![crate::weights_source::WeightsSourceExt::path(&spec.weights)];
    if let Some(control) = &spec.control {
        let bytes = source_bytes(control);
        total_bytes = total_bytes.saturating_add(bytes);
        recursively_priced_paths.push(crate::weights_source::WeightsSourceExt::path(control));
    }
    // Read the actual adapter sources at the same pre-load seam as controls. Provider-specific
    // residency matters: packed Wan keeps additive residuals, while dense Wan folds them into the
    // base and adds zero independent bytes. Other providers conservatively retain the source bytes;
    // a typed component contract may replace them with a more exact resident measurement below.
    total_bytes = total_bytes.saturating_add(external_adapter_bytes);
    if external_adapter_bytes > 0 {
        recursively_priced_paths.extend(spec.adapters.iter().filter_map(|adapter| {
            let inside = match &spec.weights {
                WeightsSource::Dir(root) => adapter.path.starts_with(root),
                WeightsSource::File(_) => false,
            };
            (!inside && adapter_source_bytes(&adapter.path) > 0).then_some(adapter.path.as_path())
        }));
    }
    // Caller-provisioned components (epic 13657) are staged from a DIFFERENT snapshot than
    // `spec.weights`, so the dir scan above cannot see them (sc-15154). Mage-Flow's per-tier dir
    // holds the DiT alone — its text encoder and VAE are bit-identical across the six variants and
    // hosted once in a shared mirror — so an unstaged sum scored a q4 edit install at 2.17 GiB
    // against a real 6.52 GiB, which both under-quoted the over-budget message and let the
    // the legacy override admit tiers that do not fit. A component resolved to a path INSIDE the
    // weights dir is skipped, because the scan already counted it.
    for (component_id, source) in &spec.components {
        if provider_owned_file_footprint.is_some() {
            // The File-aware footprint already reflects every named source the provider consumes.
            continue;
        }
        if matches!(&spec.weights, WeightsSource::File(_))
            && component_id == gen_core::BASE_SNAPSHOT_COMPONENT
        {
            // Compatibility fallback for a File-capable provider that has not published its
            // footprint yet. The companion snapshot supplies TE/VAE/tokenizer/config only; its
            // transformer is replaced by `spec.weights`. Explicit ComfyUI components replace the
            // corresponding snapshot subtree and are counted normally on their own map entries.
            if let WeightsSource::Dir(root) = source {
                if !spec
                    .components
                    .contains_key(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT)
                {
                    let companion_te = sum_safetensors_bytes(&root.join("text_encoder"));
                    total_bytes = total_bytes.saturating_add(companion_te);
                    te_bytes = te_bytes.saturating_add(companion_te);
                }
                if !spec
                    .components
                    .contains_key(gen_core::COMFYUI_VAE_COMPONENT)
                {
                    total_bytes =
                        total_bytes.saturating_add(sum_safetensors_bytes(&root.join("vae")));
                }
            }
            continue;
        }
        let inside = match source {
            WeightsSource::Dir(path) | WeightsSource::File(path) => match &spec.weights {
                WeightsSource::Dir(root) => path.starts_with(root),
                WeightsSource::File(_) => false,
            },
        };
        if !inside {
            let component_bytes = source_bytes(source);
            total_bytes = total_bytes.saturating_add(component_bytes);
            recursively_priced_paths.push(crate::weights_source::WeightsSourceExt::path(source));
            if matches!(&spec.weights, WeightsSource::File(_))
                && component_id == gen_core::COMFYUI_TEXT_ENCODER_COMPONENT
            {
                te_bytes = te_bytes.saturating_add(component_bytes);
            }
            // A component staged from OUTSIDE a `Dir` weights root SUPERSEDES any identically named
            // subtree INSIDE that root: the loader reads the staged copy, so the in-root copy is dead
            // weight the `sum_safetensors_bytes(dir)` above already (wrongly) counted. Subtract it.
            //
            // MiniMax-H3 is the motivating case (sc-17137). `spec.weights` is the shared UPSTREAM
            // snapshot — the VAEs and tokenizer a render genuinely loads — but that same snapshot also
            // carries the dense `transformer/`, `transformer_ref/` and `text_encoder/`, ~180 GB a
            // q4/q8 render never touches because the tiered DiT (`components["transformer"]`) and the
            // packed encoder (`components["text_encoder"]`) are staged in from `SceneWorks/
            // minimax-h3-mlx`. Without this subtraction the flat sum quotes ~250 GB for a model whose
            // real q4 resident set is ~55 GB, so a 128 GB Mac refuses a model that fits with room to
            // spare. The `<name>_ref` sibling covers a DiT split into base/reference partitions (the
            // reference partition is likewise staged from the tier repo, so the upstream copy is dead).
            //
            // No-op for every model whose weights dir has no subdir named like a staged component —
            // i.e. whose `spec.weights` IS its DiT rather than a companion root — so it cannot move any
            // other model's admission. Exact cancellation: the subtree scan re-walks the same files the
            // whole-tree scan counted (`sum_safetensors_bytes` dedups by canonical dir per call).
            if let WeightsSource::Dir(root) = &spec.weights {
                let superseded = sum_safetensors_bytes(&root.join(component_id)).saturating_add(
                    sum_safetensors_bytes(&root.join(format!("{component_id}_ref"))),
                );
                total_bytes = total_bytes.saturating_sub(superseded);
                // The dir scan measured the DENSE upstream encoder into `te_bytes`; once superseded,
                // the resident (and sequential-droppable) text encoder is the staged packed one. Keep
                // the `Σweights − te` staged peak coherent rather than subtracting bytes no longer
                // resident. Guarded to the packed-TE component key MiniMax-H3 stages (sc-19120).
                if superseded > 0 && component_id == "text_encoder" {
                    te_bytes = component_bytes;
                }
            }
        }
    }
    // The optional selected encoder is a typed source, not a named component. Include its exact
    // resolved direct-shard bytes in the resident total. Equal/child paths are already covered by a
    // recursively priced source. A selected encoder that is a parent of the base/control/component
    // still adds all of its direct shards: gen-core's loader does not recurse into those descendants.
    // A selection-aware provider owns the staged/materialized TE footprint; exact direct-shard bytes
    // are the fallback only when that provider fact is absent or zero. Provider-owned File
    // footprints are complete assemblies and remain authoritative.
    if provider_owned_file_footprint.is_none() {
        if let Some(selected) = selected_text_encoder {
            let already_priced = recursively_priced_paths
                .iter()
                .any(|path| selected.path.starts_with(path));
            let additional_bytes = if already_priced {
                0
            } else {
                selected.direct_shard_bytes
            };
            total_bytes = total_bytes.saturating_add(additional_bytes);
            te_bytes = footprint_te
                .filter(|bytes| *bytes > 0)
                .unwrap_or(selected.direct_shard_bytes);
        }
    }
    (total_bytes, te_bytes, headroom)
}

fn spec_component_bytes_with_provider_footprint_checked(
    engine_id: &str,
    spec: &LoadSpec,
    footprint: Option<PerComponentBytes>,
) -> gen_core::Result<(u64, u64, HeadroomAllowance)> {
    let file_sizes = prepared_file_sizes(spec)?;
    let selected_text_encoder = spec
        .text_encoder
        .as_ref()
        .map(|source| resolved_text_encoder_source(spec, source))
        .transpose()?;
    // Fill a File slot the provider left EMPTY with the exact prepared target size — and only such a
    // slot. A non-zero value in a File slot is a provider fact about the assembly it will actually
    // materialize (the checkpoint-import lane prices the DECODED plan: an fp8/int8 single-file import
    // is resident at its decode width, not at its on-disk width), so overwriting it with the raw file
    // length under-prices the modeled side of the resident-attribution guard by exactly the decode
    // delta and refuses imports that fit (sc-20651).
    //
    // The overwrite was introduced to stop compatibility-mode provider code from smuggling its own
    // `stat` into prepared admission identity. It never bought that: `prepared_file_sizes` above runs
    // `validate_prepared_file_pins` → `ensure_unchanged` on every configured File, so a prepared spec
    // whose file no longer matches its token is a hard error before any byte is priced, and for a
    // genuinely byte-preserving provider the substituted value is the value already there. What the
    // overwrite did buy was silently discarding every plan-priced fact. The infallible sibling
    // `spec_component_bytes_with_provider_footprint` never had it; the two seams now agree.
    let footprint = footprint.map(|mut footprint| {
        let fill = |slot: &mut u64, source: &WeightsSource| {
            if *slot == 0 {
                *slot = prepared_source_bytes(source, &file_sizes);
            }
        };
        if matches!(spec.weights, WeightsSource::File(_)) {
            fill(&mut footprint.dit, &spec.weights);
        }
        if let Some(source @ WeightsSource::File(_)) = spec
            .components
            .get(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT)
        {
            fill(&mut footprint.text_encoder, source);
        }
        if let Some(source @ WeightsSource::File(_)) =
            spec.components.get(gen_core::COMFYUI_VAE_COMPONENT)
        {
            fill(&mut footprint.vae, source);
        }
        footprint
    });
    let external_adapter_bytes =
        prepared_adapter_source_bytes_for_gate_where(engine_id, spec, &file_sizes, |path| {
            match &spec.weights {
                WeightsSource::Dir(root) => !path.starts_with(root),
                WeightsSource::File(_) => true,
            }
        });
    Ok(spec_component_bytes_with_provider_footprint_and_sizes(
        engine_id,
        spec,
        footprint,
        selected_text_encoder.as_ref(),
        |source| prepared_source_bytes(source, &file_sizes),
        external_adapter_bytes,
        |path| {
            std::path::absolute(path)
                .ok()
                .and_then(|path| file_sizes.get(&path).copied())
                .unwrap_or(0)
        },
    ))
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

fn try_decide_residency_for_spec(
    engine_id: &str,
    spec: &LoadSpec,
) -> WorkerResult<ResidencyOutcome> {
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    let (total_bytes, te_bytes, headroom) = spec_component_bytes_checked(engine_id, spec)?;
    Ok(decide_residency_with_headroom(
        total_bytes,
        te_bytes,
        budget,
        engine_supports_sequential(engine_id),
        headroom.total_gb,
    ))
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
    crate::fit_gate::exceeds_budget(needed_gb, budget.total_gb).then(|| {
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
mod tests;

#[cfg(test)]
#[path = "mlx_tier_admission_tests.rs"]
mod tier_admission_tests;

/// sc-24806: installed-artifact check of the tier chooser's declared-Sequential release. Runs the
/// production policy step and both pre-load admissions without loading tensors.
#[cfg(all(test, target_os = "macos"))]
mod declared_sequential_release_artifact_tests;
