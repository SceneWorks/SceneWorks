//! Candle image-provider adoption of the shared request-scoped memory selector.
//!
//! Static capability declarations never authorize an optimized request at a MEASURED grade. Exact
//! authoritative records from the packaged evidence bundle are the only measured optimized
//! candidates. Since sc-18097 (epic 18093 R1b, the candle mirror of sc-18096) every other
//! implemented optimized rung of a CERTIFIED artifact additionally carries a synthesized
//! ESTIMATE-floor candidate —
//! manifest `vramGbByTier`/`sequentialPeakGb` rows plus the standard headroom, never a promised
//! unmeasured saving — graded by the shared selector behind the candle estimate margin
//! (`crate::ladder_margin_policy::CANDLE_ESTIMATE_MARGIN`; CUDA OOM is a recoverable `Err`, so the
//! margin is looser than MLX's). Any eligible measured candidate at the same rung supersedes the
//! estimate, so measured-current admission is byte-for-byte unchanged; an UNCERTIFIED artifact
//! gets no floors at all (the manifest rows describe the certified bytes, not an imported
//! checkpoint) and keeps its resident-estimate-only behavior.

use gen_core::{
    GenerationMemory, LoadSpec, MemoryBackend, MemoryCacheState, MemoryConformanceState,
    MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryEvidenceVerdict,
    MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryParityContract, MemoryParityResult,
    MemoryRunContext, MemorySelection, MemoryStrategy, Precision, Quant,
};
use sceneworks_core::memory_calibration::{
    Backend as CalibrationBackend, BundleLoad, CalibrationBinding, EvidenceQuery, EvidenceVerdict,
    Geometry as CalibrationGeometry, LoadShapeKey, QualityResult, RequiredNullable, StrategyRung,
};
use serde_json::{Map as JsonObject, Value};

use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};
use crate::vram_gate::VramBudget;
use crate::{WorkerError, WorkerResult};

const Z_IMAGE_REQUEST_EVIDENCE_REVISION: &str = "sc-15815-candle-z-image-request-scope-v1";
const QWEN_IMAGE_REQUEST_EVIDENCE_REVISION: &str = "sc-15817-candle-qwen-image-request-scope-v1";
const FLUX1_REQUEST_EVIDENCE_REVISION: &str = "sc-15823-candle-flux1-request-scope-v1";
const FLUX2_DEV_REQUEST_EVIDENCE_REVISION: &str = "sc-15833-candle-flux2-dev-request-scope-v1";
const FLUX2_KLEIN_REQUEST_EVIDENCE_REVISION: &str = "sc-15831-candle-flux2-klein-request-scope-v1";
const MAGE_FLOW_REQUEST_EVIDENCE_REVISION: &str = "sc-15813-candle-mage-flow-request-scope-v1";
const PULID_FLUX_REQUEST_EVIDENCE_REVISION: &str = "sc-15839-candle-pulid-flux-request-scope-v1";
const DECLARATION_REQUEST_EVIDENCE_REVISION: &str = "sc-18456-candle-declaration-request-scope-v1";
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

pub(crate) struct CandleMemoryEvaluation {
    pub memory: Option<GenerationMemory>,
    pub context: MemoryRunContext,
    pub predicted_peak_gb: f64,
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
    &[]
}

fn active_component_floors(
    engine_id: &str,
    selected: Option<Quant>,
) -> &'static [gen_core::ComponentPrecisionFloor] {
    let declared = declared_component_floors(engine_id);
    match selected {
        Some(selected)
            if !declared.is_empty() && declared.iter().all(|floor| floor.applies_to(selected)) =>
        {
            declared
        }
        _ => &[],
    }
}

fn numeric_tier(engine_id: &str, tier: &str) -> Option<MemoryNumericTier> {
    let quant = match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        _ => return None,
    };
    Some(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: active_component_floors(engine_id, quant),
    })
}

struct RequestModeBinding {
    mode: MemoryMode,
    /// Catalog/matrix axis used to find the entry-specific calibration binding.
    calibration_key: String,
    /// Typed gen-core key used by the exact request scope (`MemoryMode::as_key`).
    scope_key: String,
}

fn request_mode(engine_id: &str, mode: &str) -> RequestModeBinding {
    let (mode, calibration_key) = match mode {
        "image_generation" | "text_to_image" => {
            (MemoryMode::TextToImage, "text_to_image".to_owned())
        }
        "style_variations" => (
            MemoryMode::Other("style_variations".to_owned()),
            "style_variations".to_owned(),
        ),
        "character_image" => (
            MemoryMode::Other("character_image".to_owned()),
            "character_image".to_owned(),
        ),
        // FLUX.2 Klein's reference alias is the same typed Edit coordinate as edit_image. Keep the
        // catalog axis at edit_image so an exact matrix calibration can promote that cell, while
        // the gen-core selector and provider receive MemoryMode::Edit / "edit".
        "reference" | "image_to_image" if engine_id == "flux2_klein_9b" => {
            (MemoryMode::Edit, "edit_image".to_owned())
        }
        "image_to_image" => (MemoryMode::ImageToImage, "image_to_image".to_owned()),
        "edit" | "edit_image" => (MemoryMode::Edit, "edit_image".to_owned()),
        _ => (MemoryMode::Other(mode.to_owned()), mode.to_owned()),
    };
    let scope_key = mode.as_key().to_owned();
    RequestModeBinding {
        mode,
        calibration_key,
        scope_key,
    }
}

fn request_mode_with_provider_override(
    engine_id: &str,
    public_mode: &str,
    provider_mode: Option<&str>,
) -> RequestModeBinding {
    let mut binding = request_mode(engine_id, public_mode);
    if let Some(provider_mode) = provider_mode {
        let provider_binding = request_mode(engine_id, provider_mode);
        binding.mode = provider_binding.mode;
        binding.scope_key = provider_binding.scope_key;
    }
    binding
}

fn strategy(rung: StrategyRung) -> MemoryStrategy {
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
/// The Candle sibling of `mlx_fit_gate::parse_evidence_parameters`, with the same rule and the same
/// shared derivation: a rung the composition engages must name its parameters, and a rung it does
/// not engage must not. Keying this on the selected rung's ordinal made a provider that implements
/// rung 4 with rungs 2 and 3 `Missing` unable to bind evidence at all.
fn parse_parameters(
    engaged: &[StrategyRung],
    parameters: &JsonObject<String, Value>,
) -> Option<gen_core::MemoryStrategyParameters> {
    const KEYS: [&str; 5] = [
        "decodeTileEdge",
        "decodeOverlap",
        "attentionChunkSize",
        "transformerWindowSize",
        "transformerWindowComponent",
    ];
    if parameters.keys().any(|key| !KEYS.contains(&key.as_str())) {
        return None;
    }
    let integer = |key: &str, minimum: u32| {
        parameters
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value >= minimum)
    };
    let required = crate::memory_strategy::required_numeric_parameters(engaged);
    if required
        .iter()
        .any(|(key, minimum)| integer(key, *minimum).is_none())
    {
        return None;
    }
    if KEYS[..4]
        .iter()
        .filter(|key| !required.iter().any(|(required, _)| required == *key))
        .any(|key| parameters.contains_key(*key))
    {
        return None;
    }
    let engages_transformer = engaged.contains(&StrategyRung::BoundedTransformerResidency);
    let transformer_window_component = match parameters.get("transformerWindowComponent") {
        None => None,
        Some(Value::String(value)) if engages_transformer => Some(match value.as_str() {
            "dit" => gen_core::TransformerComponent::Dit,
            "text_encoder" => gen_core::TransformerComponent::TextEncoder,
            "both" => gen_core::TransformerComponent::Both,
            _ => return None,
        }),
        Some(_) => return None,
    };
    Some(gen_core::MemoryStrategyParameters {
        decode_tile_edge: integer("decodeTileEdge", 1),
        decode_overlap: integer("decodeOverlap", 0),
        attention_chunk_size: integer("attentionChunkSize", 1),
        transformer_window_size: integer("transformerWindowSize", 1),
        transformer_window_component,
    })
}

fn text<'a>(object: &'a JsonObject<String, Value>, key: &str) -> Option<&'a str> {
    object
        .get(key)?
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn binding(object: &JsonObject<String, Value>) -> Option<CalibrationBinding> {
    let abi = u32::try_from(object.get("abi")?.as_u64()?).ok()?;
    let load_shape = match object.get("loadShape").and_then(Value::as_str) {
        Some("eager_materialization") => LoadShapeKey::EagerMaterialization,
        Some("deferred_materialization") => LoadShapeKey::DeferredMaterialization,
        Some(_) => return None,
        None if abi >= 2 => return None,
        None => LoadShapeKey::EagerMaterialization,
    };
    Some(CalibrationBinding {
        abi,
        load_shape,
        fingerprint: text(object, "fingerprint")?.to_owned(),
        scene_works_revision: text(object, "sceneWorksRevision")?.to_owned(),
        matrix_source_revision: text(object, "matrixSourceRevision")?.to_owned(),
        inference_revision: text(object, "inferenceRevision")?.to_owned(),
        inference_closure_digest: text(object, "inferenceClosureDigest")?.to_owned(),
        artifact_repository: text(object, "artifactRepository")?.to_owned(),
        artifact_resolved_revision: text(object, "artifactResolvedRevision")?.to_owned(),
        artifact_variant: text(object, "artifactVariant")?.to_owned(),
        resolved_path_fingerprint: text(object, "resolvedPathFingerprint")?.to_owned(),
    })
}

fn rung(value: &str) -> Option<StrategyRung> {
    match value {
        "resident" => Some(StrategyRung::Resident),
        "staged_residency" => Some(StrategyRung::StagedResidency),
        "bounded_decode" => Some(StrategyRung::BoundedDecode),
        "bounded_attention" => Some(StrategyRung::BoundedAttention),
        "bounded_transformer_residency" => Some(StrategyRung::BoundedTransformerResidency),
        _ => None,
    }
}

fn evidence_provider(engine_id: &str) -> &str {
    engine_id.strip_suffix("_control").unwrap_or(engine_id)
}

fn binding_matches_request(
    item: &JsonObject<String, Value>,
    provider: &str,
    tier: &str,
    mode: &str,
    overlay: &str,
    geometry: MemoryGeometry,
) -> bool {
    if text(item, "provider") != Some(provider)
        || text(item, "tier") != Some(tier)
        || text(item, "mode") != Some(mode)
        || text(item, "overlay") != Some(overlay)
    {
        return false;
    }
    let Some(item_geometry) = item.get("geometry").and_then(Value::as_object) else {
        return false;
    };
    ["width", "height", "batch", "frames"]
        .into_iter()
        .zip([
            geometry.width,
            geometry.height,
            geometry.batch,
            geometry.frames,
        ])
        .all(|(key, expected)| {
            item_geometry.get(key).and_then(Value::as_u64) == Some(u64::from(expected))
        })
}

#[allow(clippy::too_many_arguments)]
fn verified_candidates(
    manifest: &JsonObject<String, Value>,
    model_id: &str,
    runtime_provider: &str,
    tier: &str,
    mode: &RequestModeBinding,
    overlay: &str,
    geometry: MemoryGeometry,
    closure_digests: &mut Vec<String>,
) -> WorkerResult<Vec<MemoryEvidence>> {
    let evidence_provider = evidence_provider(runtime_provider);
    let loaded = sceneworks_core::memory_calibration::load_packaged_bundle().map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "packaged memory-calibration evidence is invalid: {error}"
        ))
    })?;
    let BundleLoad::Ready(bundle) = loaded else {
        return Ok(Vec::new());
    };
    let Some(bindings) = manifest
        .get("candle")
        .and_then(|value| value.get("calibrations"))
        .and_then(Value::as_array)
    else {
        return Ok(Vec::new());
    };
    let calibration_geometry = CalibrationGeometry {
        width: geometry.width,
        height: geometry.height,
        batch: geometry.batch,
        frames: geometry.frames,
    };
    let mut candidates = Vec::new();
    for value in bindings {
        let Some(item) = value.as_object() else {
            continue;
        };
        if !binding_matches_request(
            item,
            evidence_provider,
            tier,
            &mode.calibration_key,
            overlay,
            geometry,
        ) {
            continue;
        }
        let Some(rung) = text(item, "rung").and_then(rung) else {
            continue;
        };
        let parameters = item
            .get("parameters")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let declared_engaged = match item.get("engagedRungs") {
            None => None,
            Some(Value::Array(values)) => {
                let Some(declared) = values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .and_then(crate::memory_strategy::rung_from_key)
                    })
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                Some(declared)
            }
            Some(_) => continue,
        };
        let Ok(engaged_rungs) =
            crate::memory_strategy::engaged_composition(rung, declared_engaged.as_deref())
        else {
            continue;
        };
        let Some(selection_parameters) = parse_parameters(&engaged_rungs, &parameters) else {
            continue;
        };
        let Some(calibration) = binding(item) else {
            continue;
        };
        // sc-17774: there is no per-model compatibility hatch any more. `flux2_dev` used to be the
        // one provider that could reach a later runtime pin, via a hand-audited
        // `compatibleInferenceRevision` naming exactly one target revision — spent the moment the
        // pin moved once more, and available to no other model. Currency is now decided for every
        // provider identically, by `calibration.inference_closure_digest` inside `evidence_for`.
        // `inferenceRevision` is capture provenance and is never compared.
        let query = EvidenceQuery {
            backend: CalibrationBackend::Candle,
            model_id: model_id.to_owned(),
            provider: evidence_provider.to_owned(),
            tier: tier.to_owned(),
            mode: mode.calibration_key.clone(),
            overlay: overlay.to_owned(),
            geometry: calibration_geometry,
            rung,
            parameters,
            calibration: calibration.clone(),
        };
        let EvidenceVerdict::Verified(record) = bundle.evidence_for(&query) else {
            continue;
        };
        let RequiredNullable::Value(predicted) = &record.predicted_peak_bytes else {
            continue;
        };
        if record.quality.result != Some(QualityResult::Passed) {
            continue;
        }
        let selected_strategy = strategy(rung);
        let load_shape = match record.load_shape {
            LoadShapeKey::EagerMaterialization => gen_core::LoadShape::EagerMaterialization,
            LoadShapeKey::DeferredMaterialization => gen_core::LoadShape::DeferredMaterialization,
        };
        let evidence_key = MemoryEvidenceKey {
            resolved_route: runtime_provider.to_owned(),
            backend: MemoryBackend::Candle,
            tier: numeric_tier(runtime_provider, tier).expect("validated numeric tier"),
            mode: mode.mode.clone(),
            load_shape,
            overlay: (overlay != "none").then(|| overlay.to_owned()),
            geometry,
            strategy: selected_strategy,
            engaged_composition: record
                .strategy
                .engaged_rungs
                .iter()
                .copied()
                .map(strategy)
                .collect(),
            parameters: selection_parameters,
        };
        // Index-aligned with `candidates`: `MemoryEvidenceKey` is a gen-core type without `Hash`,
        // and the two vectors are pushed together in this one loop.
        closure_digests.push(calibration.inference_closure_digest.clone());
        candidates.push(MemoryEvidence {
            key: evidence_key,
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions::VERIFIED,
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint,
            sceneworks_revision: calibration.scene_works_revision,
            inference_revision: calibration.inference_revision.clone(),
            harness_version: record.harness_version.clone(),
            predicted_peak_bytes: predicted.overall(),
            observed_peak_bytes: match &record.observed_memory {
                RequiredNullable::Value(observed) => Some(observed.overall_non_reclaimable_bytes()),
                RequiredNullable::Null => None,
            },
            parity: if record.quality.maximum_error_threshold == Some(0.0)
                && record.quality.mean_error_threshold == Some(0.0)
            {
                MemoryParityContract::Exact
            } else {
                MemoryParityContract::Tolerance {
                    metric: "maximum RGB8 channel error".to_owned(),
                    maximum_error: record.quality.maximum_error_threshold.unwrap_or_default(),
                }
            },
            parity_result: MemoryParityResult::Passed,
        });
    }
    Ok(candidates)
}

fn account_for_runtime_overlay_bytes(
    candidates: &mut [MemoryEvidence],
    runtime_overlay_bytes: u64,
) {
    for candidate in candidates {
        candidate.predicted_peak_bytes = candidate
            .predicted_peak_bytes
            .saturating_add(runtime_overlay_bytes);
    }
}

/// The smallest declared value for every numeric knob the engaged composition requires — the most
/// deeply bounding parameters the provider publishes, which keeps the true runtime transient as
/// far below the floor's unreduced peak as the provider allows (sc-18097, the candle mirror of the
/// MLX gate's helper from sc-18096). `None` when a required knob has no declared range: such a
/// selection cannot be validated, so no estimate candidate is synthesized for the rung.
fn estimate_floor_parameters(
    contract: &gen_core::MemoryProviderContract,
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

/// Synthesize an estimate-floor candidate for every implemented optimized rung (sc-18097, epic
/// 18093 R1b — the candle mirror of `mlx_fit_gate::synthesize_estimate_ladder`'s floor arm).
///
/// Peak source per rung, from the manifest rows the legacy candle gate already trusts
/// (`crate::vram_gate::predicted_peak_gb` conventions) — never a tuned coefficient and never a
/// promised unmeasured saving:
///
/// * `StagedResidency` — the measured `candle.sequentialPeakGb` row (the staged working set) plus
///   the standard headroom, exactly the number the legacy sequential-offload gate compares. Absent
///   the row, the resident estimate itself: staging is still selectable, but no unmeasured saving
///   is promised so it can only admit where resident does.
/// * Rungs 2–4 bound transients/residency the manifest has NOT measured for this cell, so they
///   take the STAGED floor unreduced — selectable without promising an unmeasured saving. (Where a
///   measured record for a rung exists it is a `verified_candidates` candidate and supersedes the
///   floor in the selector.) The staged row prices the staged WORKING SET, so it is a sound floor
///   only for a composition that actually engages `StagedResidency` (sc-18253). A provider may
///   implement a deep rung whose engaged composition excludes staging
///   (`gen_core::MemoryProviderContract::engaged_composition`) — such a request runs whole-model
///   resident, so its floor clamps to the RESIDENT estimate instead: the candle mirror of the MLX
///   floor's `engaged.contains(&StagedResidency)` max-vs-sum split
///   (`mlx_fit_gate::estimate_floor_weights_bytes`), which keeps the rung selectable without ever
///   under-predicting a resident working set behind the estimate margin.
///
/// The candle estimate margin is NOT applied here — the selector owns margin widening
/// (`crate::memory_strategy::select_strategy`), exactly as it owns the sc-18095 stale widening.
#[allow(clippy::too_many_arguments)]
fn synthesize_estimate_floors(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    tier: MemoryNumericTier,
    mode: &RequestModeBinding,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    resident_peak_bytes: u64,
    runtime_overlay_bytes: u64,
    request_evidence_revision: &str,
) -> Vec<(MemorySelection, MemoryEvidence)> {
    let calibration = contract.calibration.as_ref();
    let staged_floor_bytes = crate::vram_gate::predicted_sequential_peak_gb(manifest, tier_key)
        .map(|gb| {
            ((gb * BYTES_PER_GIB).ceil().clamp(0.0, u64::MAX as f64) as u64)
                .saturating_add(runtime_overlay_bytes)
        })
        .unwrap_or(resident_peak_bytes);
    let mut synthesized = Vec::new();
    for strategy in MemoryStrategy::ALL {
        if strategy == MemoryStrategy::Resident {
            // The resident live estimate is already submitted on every request.
            continue;
        }
        if !matches!(
            contract.capability(strategy).map(|cap| &cap.support),
            Some(gen_core::MemoryStrategySupport::Implemented)
        ) {
            continue;
        }
        let engaged = contract.engaged_composition(strategy);
        let Some(parameters) = estimate_floor_parameters(contract, &engaged) else {
            continue;
        };
        let selection = MemorySelection {
            strategy,
            parameters,
            tier,
        };
        if contract.validate_selection(&selection).is_err() {
            continue;
        }
        // sc-18253: the staged row is only a sound floor for a composition that engages staging;
        // a deep rung excluding `StagedResidency` runs whole-model resident and clamps to the
        // resident estimate (see the doc comment above).
        let predicted_peak_bytes = if engaged.contains(&MemoryStrategy::StagedResidency) {
            staged_floor_bytes
        } else {
            resident_peak_bytes
        };
        tracing::info!(
            route = engine_id,
            backend = "candle",
            ?strategy,
            raw_peak_bytes = predicted_peak_bytes,
            "synthesized manifest-row floor estimate candidate"
        );
        synthesized.push((
            selection,
            MemoryEvidence {
                key: MemoryEvidenceKey {
                    resolved_route: engine_id.to_owned(),
                    backend: MemoryBackend::Candle,
                    tier,
                    mode: mode.mode.clone(),
                    load_shape: contract.load_shape,
                    overlay: overlay.map(str::to_owned),
                    geometry,
                    strategy,
                    engaged_composition: engaged,
                    parameters,
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
                calibration_abi: calibration
                    .map_or(gen_core::MEMORY_CALIBRATION_ABI, |item| item.abi),
                calibration_fingerprint: calibration
                    .map_or_else(String::new, |item| item.fingerprint.clone()),
                sceneworks_revision: request_evidence_revision.to_owned(),
                inference_revision: crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION
                    .to_owned(),
                harness_version: String::new(),
                predicted_peak_bytes,
                observed_peak_bytes: None,
                parity: MemoryParityContract::Exact,
                parity_result: MemoryParityResult::NotRun,
            },
        ));
    }
    synthesized
}

fn memory_for_selection(
    contract: &gen_core::MemoryProviderContract,
    selection: MemorySelection,
) -> Option<GenerationMemory> {
    (selection.strategy != MemoryStrategy::Resident).then(|| GenerationMemory {
        stage_residency: contract.engages(selection.strategy, MemoryStrategy::StagedResidency),
        tile_vae_decode: contract.engages(selection.strategy, MemoryStrategy::BoundedDecode),
        decode_tile_edge: selection.parameters.decode_tile_edge,
        decode_overlap: selection.parameters.decode_overlap,
        chunk_attention: contract.engages(selection.strategy, MemoryStrategy::BoundedAttention),
        stream_transformer_blocks: contract.engages(
            selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ),
        transformer_window_size: selection.parameters.transformer_window_size,
        transformer_window_component: selection.parameters.transformer_window_component,
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_image(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    let (contract_override, provider_overlay_override, provider_mode_override) =
        if spec.load_shape_declaration_result == gen_core::LoadShapeDeclarationResult::Eligible {
            let mode =
                crate::memory_route_registry::MemoryRouteMode::from_request(request_mode_value);
            let Some(contract) = crate::memory_route_registry::declared_candle_selector_contract(
                engine_id,
                Some(tier_key),
                mode,
                manifest,
                spec,
            ) else {
                return Ok(None);
            };
            (Some(contract), None, None)
        } else {
            let Some(mode) =
                crate::memory_route_registry::MemoryRouteMode::from_request(request_mode_value)
            else {
                return Ok(None);
            };
            match crate::memory_route_registry::declared_candle_request_strategy_contract(
            engine_id,
            Some(tier_key),
            manifest,
            spec,
            crate::memory_route_registry::MemoryRouteRequestContext {
                mode,
                reference_count: geometry.reference_count,
                use_pid,
                has_phases: request_has_phases,
            },
        ) {
            crate::memory_route_registry::DeclaredCandleStrategyContract::NoRelevantDeclaration => {
                (None, None, None)
            }
            crate::memory_route_registry::DeclaredCandleStrategyContract::Applied {
                contract,
                provider_overlay,
                provider_mode,
            } => (Some(*contract), Some(provider_overlay), Some(provider_mode)),
            crate::memory_route_registry::DeclaredCandleStrategyContract::Refused => {
                return Ok(None)
            }
        }
        };
    let provider_overlay = provider_overlay_override
        .as_ref()
        .map(|overlay| overlay.as_deref())
        .unwrap_or(overlay);
    evaluate_shared_image_inner(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        provider_overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        provider_mode_override.as_deref(),
        contract_override,
        None,
    )
}

/// Shared-selector entry point for a bespoke Candle provider whose exact contract is built from
/// caller-provisioned paths rather than the registered provider catalog.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_bespoke_image(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    budget: Option<VramBudget>,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    contract: gen_core::MemoryProviderContract,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_inner(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        has_phases,
        budget,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        None,
        Some(contract),
        Some(PULID_FLUX_REQUEST_EVIDENCE_REVISION),
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_shared_image_inner(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    provider_overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    worker_multipass: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    provider_mode_override: Option<&str>,
    contract_override: Option<gen_core::MemoryProviderContract>,
    request_evidence_revision_override: Option<&'static str>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    let request_evidence_revision = request_evidence_revision_override.unwrap_or(match engine_id {
        "z_image" | "z_image_turbo" | "z_image_control" | "z_image_turbo_control" => {
            Z_IMAGE_REQUEST_EVIDENCE_REVISION
        }
        "qwen_image" | "qwen_image_edit" => QWEN_IMAGE_REQUEST_EVIDENCE_REVISION,
        "flux1_schnell" | "flux1_dev" => FLUX1_REQUEST_EVIDENCE_REVISION,
        "flux2_dev" => FLUX2_DEV_REQUEST_EVIDENCE_REVISION,
        "flux2_klein_9b" => FLUX2_KLEIN_REQUEST_EVIDENCE_REVISION,
        "mage_flow_base"
        | "mage_flow"
        | "mage_flow_turbo"
        | "mage_flow_edit_base"
        | "mage_flow_edit"
        | "mage_flow_edit_turbo" => MAGE_FLOW_REQUEST_EVIDENCE_REVISION,
        _ => DECLARATION_REQUEST_EVIDENCE_REVISION,
    });
    // Hires-fix is two independently shaped denoise passes. The generic generation harness still
    // reuses one context for both, so it cannot truthfully carry a request-scoped geometry yet.
    // Keep that surface on its established path until it mints one scope per pass.
    if worker_multipass {
        return Ok(None);
    }
    let mode =
        request_mode_with_provider_override(engine_id, request_mode_value, provider_mode_override);
    let Some(tier) = numeric_tier(engine_id, tier_key) else {
        return Ok(None);
    };
    let Some(budget) = budget else {
        return Ok(None);
    };
    let Some(resident_peak_gb) = predicted_peak_gb else {
        return Ok(None);
    };
    let contract = match contract_override {
        Some(contract) => contract,
        None => {
            let Some(contract) = crate::inference_runtime::media()
                .memory_strategy_contract(engine_id, spec)
                .map_err(|error| {
                    WorkerError::Engine(format!("{engine_id} memory contract failed: {error}"))
                })?
            else {
                return Ok(None);
            };
            contract
        }
    };
    let calibration = contract.calibration.as_ref();
    let resident_selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier,
    };
    let mut resident = MemoryEvidence {
        key: MemoryEvidenceKey {
            resolved_route: engine_id.to_owned(),
            backend: MemoryBackend::Candle,
            tier,
            mode: mode.mode.clone(),
            load_shape: contract.load_shape,
            overlay: provider_overlay.map(str::to_owned),
            geometry,
            strategy: MemoryStrategy::Resident,
            engaged_composition: contract.engaged_composition(MemoryStrategy::Resident),
            parameters: resident_selection.parameters,
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
        calibration_abi: calibration.map_or(gen_core::MEMORY_CALIBRATION_ABI, |item| item.abi),
        calibration_fingerprint: calibration
            .map_or_else(String::new, |item| item.fingerprint.clone()),
        sceneworks_revision: request_evidence_revision.to_owned(),
        inference_revision: crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION.to_owned(),
        harness_version: String::new(),
        predicted_peak_bytes: (resident_peak_gb * BYTES_PER_GIB).ceil() as u64,
        observed_peak_bytes: None,
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::NotRun,
    };
    let exact_overlay = overlay.unwrap_or("none");
    // Matrix/evidence overlays remain cells of the advertised base provider (`z_image` or
    // `z_image_turbo`), while the runtime registers the strict-control implementation under a
    // dedicated `_control` id. Query the packaged evidence by its catalog identity, then bind the
    // returned candidate to the exact runtime route expected by the provider contract.
    let mut closure_digests: Vec<String> = Vec::new();
    let mut verified = if artifact_is_certified {
        verified_candidates(
            manifest,
            model_id,
            engine_id,
            tier_key,
            &mode,
            exact_overlay,
            geometry,
            &mut closure_digests,
        )?
    } else {
        Vec::new()
    };
    // A closure digest answers whether the compiled provider code changed; the provider-owned
    // calibration identity answers whether the measured memory semantics changed. Both have to
    // match. FLUX.2-dev's caption-upsample lifecycle intentionally rotated v2 -> v3 while retaining
    // the old historical records, so accepting closure-stale evidence merely because the selector
    // can widen it would feed an obsolete prompt-conditioning peak into the live v3 contract.
    let mut current_verified = Vec::with_capacity(verified.len());
    let mut current_closure_digests = Vec::with_capacity(closure_digests.len());
    if let Some(current_calibration) = calibration {
        for (candidate, closure_digest) in verified.into_iter().zip(closure_digests) {
            if candidate.calibration_fingerprint == current_calibration.fingerprint {
                current_verified.push(candidate);
                current_closure_digests.push(closure_digest);
            }
        }
    }
    verified = current_verified;
    closure_digests = current_closure_digests;
    // Packaged bindings are looked up by the public catalog/matrix overlay above. Once admitted,
    // however, every candidate submitted to the provider selector must carry the exact provider
    // evidence identity declared by the route. Krea adapters are load identity, not a provider
    // `MemoryRunContext` overlay, so their public `lora` cell is normalized to `None` here.
    for candidate in &mut verified {
        candidate.key.overlay = provider_overlay.map(str::to_owned);
    }
    // Calibration records describe the certified overlay fixture. User-provided adapters can be
    // larger, so every optimized candidate must reserve the bytes for the actual request before
    // the common selector performs its fit check. The resident estimate already includes these
    // bytes through `predicted_peak_gb_with_adapter_bytes` and must not be adjusted twice.
    account_for_runtime_overlay_bytes(&mut verified, runtime_overlay_bytes);
    if let Some(exact) = verified.first() {
        resident
            .inference_revision
            .clone_from(&exact.inference_revision);
    }
    // sc-18097: estimate-floor candidates for every implemented optimized rung, so an unmeasured
    // cell can still engage the ladder behind the candle estimate margin instead of freezing to
    // resident-or-legacy behavior. Where an exact measured record exists the selector's
    // measured-supersedes-estimate rule keeps admission byte-for-byte unchanged.
    //
    // Gated on `artifact_is_certified`, the same conjunct that gates the packaged records above
    // (sc-18097 review, major finding). The floors are read from the SHIPPED manifest's
    // `sequentialPeakGb`/`vramGbByTier` rows, which describe the certified artifact. An imported
    // or community checkpoint on a supported route is different bytes: those rows do not describe
    // it, so extending its ladder from them would be a static capability declaration authorizing
    // an optimized request — exactly what this module's header forbids — and CUDA OOM being
    // recoverable does not make a wrong prediction safe. An uncertified artifact therefore keeps
    // its resident-estimate-only behavior, byte-for-byte as before this story. The sibling control
    // lane gates its floors the same way (`krea_control_fit.rs`, `runtime_verified`).
    let synthesized = if artifact_is_certified {
        synthesize_estimate_floors(
            engine_id,
            &contract,
            manifest,
            tier_key,
            tier,
            &mode,
            provider_overlay,
            geometry,
            resident.predicted_peak_bytes,
            runtime_overlay_bytes,
            request_evidence_revision,
        )
    } else {
        Vec::new()
    };
    let capacity = verified.len() + synthesized.len() + 1;
    let mut selections = Vec::with_capacity(capacity);
    let mut evidence = Vec::with_capacity(capacity);
    // The resident candidate is a live estimate, not a calibrated record, so it carries the live
    // digest and is never staled by this gate.
    let live_closure_digest = sceneworks_core::memory_calibration::packaged_closure_digest(
        "candle",
        evidence_provider(engine_id),
    )
    .unwrap_or_default();
    let mut candidate_digests = Vec::with_capacity(capacity);
    // Index-aligned basis axis (sc-18097): the synthesized floors carry their estimate basis;
    // the resident live estimate and every packaged record stay `Measured`.
    let mut candidate_bases = Vec::with_capacity(capacity);
    selections.push(resident_selection);
    evidence.push(&resident);
    candidate_digests.push(live_closure_digest.clone());
    candidate_bases.push(crate::memory_strategy::CandidateBasis::Measured);
    for (index, item) in verified.iter().enumerate() {
        candidate_digests.push(closure_digests.get(index).cloned().unwrap_or_default());
        selections.push(MemorySelection {
            strategy: item.key.strategy,
            parameters: item.key.parameters,
            tier: item.key.tier,
        });
        evidence.push(item);
        candidate_bases.push(crate::memory_strategy::CandidateBasis::Measured);
    }
    for (selection, item) in &synthesized {
        selections.push(*selection);
        evidence.push(item);
        // A floor is a declaration of the manifest rows under the LIVE closure, not a calibrated
        // record; there is nothing there for currency to invalidate.
        candidate_digests.push(live_closure_digest.clone());
        candidate_bases.push(crate::memory_strategy::CandidateBasis::EstimateFloor);
    }
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
    let selected = crate::memory_strategy::select_strategy(
        RequestScope {
            resolved_route: engine_id,
            backend: "candle",
            tier,
            mode: &mode.scope_key,
            overlay: provider_overlay,
            geometry,
            // sc-17774: one mechanism, same as every other lane. `unwrap_or_default` fails closed.
            expected_closure_digest: &live_closure_digest,
        },
        &contract,
        Some(Budget {
            available_gb: budget.free_gb,
            reclaimable_gb: 0.0,
            total_gb: budget.total_gb,
            reserved_headroom_gb: 2.0,
        }),
        &candidates,
    );
    let Selection::Selected { selection, .. } = selected else {
        return Ok(None);
    };
    let matches_selection = |item: &MemoryEvidence| {
        item.key.strategy == selection.strategy
            && item.key.parameters == selection.parameters
            && item.key.tier == selection.tier
    };
    let (selected_evidence, estimate_scoped) = if selection.strategy == MemoryStrategy::Resident {
        (&resident, false)
    } else if let Some(item) = verified.iter().find(|item| matches_selection(item)) {
        (item, false)
    } else if let Some((_, item)) = synthesized.iter().find(|(_, item)| matches_selection(item)) {
        // sc-18097: a synthesized floor was selected — legacy-scoped telemetry below, exactly like
        // the resident estimate (no measured harness version to claim).
        (item, true)
    } else {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} selected a memory strategy without exact packaged evidence"
        )));
    };
    let to_bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
    Ok(Some(CandleMemoryEvaluation {
        memory: memory_for_selection(&contract, selection),
        predicted_peak_gb: selected_evidence.predicted_peak_bytes as f64 / BYTES_PER_GIB,
        context: MemoryRunContext {
            selection,
            optimization_authority: gen_core::MemoryOptimizationAuthority::Calibrated,
            calibration_abi: selected_evidence.calibration_abi,
            calibration_fingerprint: selected_evidence.calibration_fingerprint.clone(),
            mode: mode.mode,
            load_shape: contract.load_shape,
            has_reference,
            use_pid,
            has_phases: request_has_phases,
            geometry,
            overlay: provider_overlay.map(str::to_owned),
            budget: gen_core::MemoryBudget {
                total_bytes: to_bytes(budget.total_gb),
                committed_bytes: to_bytes((budget.total_gb - budget.free_gb).max(0.0)),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: to_bytes(2.0),
            },
            predicted_peak_bytes: selected_evidence.predicted_peak_bytes,
            cache_state,
            evidence_revision: if selection.strategy == MemoryStrategy::Resident || estimate_scoped
            {
                request_evidence_revision.to_owned()
            } else {
                selected_evidence.harness_version.clone()
            },
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::WeightsSource;
    use serde_json::json;
    use std::path::PathBuf;

    /// sc-17728, the Candle sibling of the MLX fit gate's coverage. The required parameter set
    /// follows the ENGAGED composition, so a provider that implements rung 4 with rungs 2 and 3
    /// `Missing` reads cleanly, while an engaged rung must still name every parameter it owns and a
    /// withheld rung's parameters stay unnameable.
    #[test]
    fn candle_parameters_follow_the_engaged_composition_not_the_rung_ordinal() {
        let withheld = [
            StrategyRung::Resident,
            StrategyRung::StagedResidency,
            StrategyRung::BoundedTransformerResidency,
        ];
        let cumulative = crate::memory_strategy::default_engaged_composition(
            StrategyRung::BoundedTransformerResidency,
        );
        let object = |value: Value| value.as_object().expect("parameter object").clone();

        let parsed = parse_parameters(
            &withheld,
            &object(json!({ "transformerWindowSize": 1, "transformerWindowComponent": "both" })),
        )
        .expect("a rung-4 composition that withholds rungs 2 and 3 must read");
        assert_eq!(parsed.transformer_window_size, Some(1));
        assert_eq!(parsed.decode_tile_edge, None);
        assert_eq!(parsed.attention_chunk_size, None);

        // Naming a withheld rung's parameter, and omitting an engaged rung's, both fail closed.
        assert!(parse_parameters(
            &withheld,
            &object(
                json!({ "transformerWindowSize": 1, "decodeTileEdge": 512, "decodeOverlap": 64 })
            ),
        )
        .is_none());
        assert!(parse_parameters(
            &withheld,
            &object(json!({ "transformerWindowComponent": "dit" }))
        )
        .is_none());

        // The cumulative default is unchanged for a provider with the whole ladder implemented.
        assert!(
            parse_parameters(&cumulative, &object(json!({ "transformerWindowSize": 1 })),)
                .is_none()
        );
        assert!(parse_parameters(
            &cumulative,
            &object(json!({
                "decodeTileEdge": 512,
                "decodeOverlap": 64,
                "attentionChunkSize": 256,
                "transformerWindowSize": 1,
            })),
        )
        .is_some());
    }

    #[test]
    fn strict_control_uses_the_advertised_base_provider_evidence_cell() {
        assert_eq!(evidence_provider("z_image_control"), "z_image");
        assert_eq!(evidence_provider("z_image_turbo_control"), "z_image_turbo");
        assert_eq!(evidence_provider("z_image"), "z_image");
    }

    #[test]
    fn flux2_klein_reference_modes_preserve_catalog_and_typed_keys() {
        let cases = [
            ("edit_image", MemoryMode::Edit, "edit_image", "edit"),
            ("reference", MemoryMode::Edit, "edit_image", "edit"),
            ("image_to_image", MemoryMode::Edit, "edit_image", "edit"),
            (
                "character_image",
                MemoryMode::Other("character_image".to_owned()),
                "character_image",
                "character_image",
            ),
            (
                "style_variations",
                MemoryMode::Other("style_variations".to_owned()),
                "style_variations",
                "style_variations",
            ),
        ];
        for (request, expected_mode, calibration_key, scope_key) in cases {
            let binding = request_mode("flux2_klein_9b", request);
            assert_eq!(binding.mode, expected_mode, "request={request}");
            assert_eq!(
                binding.calibration_key, calibration_key,
                "request={request}"
            );
            assert_eq!(binding.scope_key, scope_key, "request={request}");
            assert_eq!(
                binding.mode.as_key(),
                binding.scope_key,
                "request={request}"
            );
        }

        let generic = request_mode("z_image", "image_to_image");
        assert_eq!(generic.mode, MemoryMode::ImageToImage);
        assert_eq!(generic.calibration_key, "image_to_image");
        assert_eq!(generic.scope_key, "image_to_image");
    }

    #[test]
    fn declaration_provider_mode_keeps_the_public_matrix_coordinate() {
        let raw_reference = request_mode_with_provider_override(
            "krea_2_raw",
            "text_to_image",
            Some("image_to_image"),
        );
        assert_eq!(raw_reference.mode, MemoryMode::ImageToImage);
        assert_eq!(raw_reference.scope_key, "image_to_image");
        assert_eq!(raw_reference.calibration_key, "text_to_image");

        let ordinary = request_mode_with_provider_override(
            "krea_2_raw",
            "text_to_image",
            Some("text_to_image"),
        );
        assert_eq!(ordinary.mode, MemoryMode::TextToImage);
        assert_eq!(ordinary.scope_key, "text_to_image");
        assert_eq!(ordinary.calibration_key, "text_to_image");
    }

    #[test]
    fn declaration_provider_overlay_binds_staged_selection_and_request_context() {
        let mut contract = gen_core::MemoryProviderContract::compatibility_default(
            "krea_2_edit",
            gen_core::MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        for capability in &mut contract.strategies {
            capability.support = if matches!(
                capability.strategy,
                MemoryStrategy::Resident | MemoryStrategy::StagedResidency
            ) {
                gen_core::MemoryStrategySupport::Implemented
            } else {
                gen_core::MemoryStrategySupport::Missing
            };
        }
        // Declaring StagedResidency Implemented is not free: gen-core's `conformance_errors()`
        // requires the lifecycle that rung actually needs — the three phases plus synchronized
        // release of completed ones — because staging IS phase-scoped residency. Without them the
        // contract is non-conformant, and `select_strategy`'s first gate rejects the WHOLE request
        // as `Unverified { Invalid }` before any candidate is considered, which reads as "nothing
        // fit" rather than "the fixture declared a rung it did not equip".
        contract.lifecycle = gen_core::MemoryLifecycleCapabilities {
            phases: vec![
                gen_core::MemoryPhase::Conditioning,
                gen_core::MemoryPhase::Denoise,
                gen_core::MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            ..contract.lifecycle
        };
        assert!(
            contract.conformance_errors().is_empty(),
            "the staged fixture must be a shape a real provider could declare: {:?}",
            contract.conformance_errors()
        );
        // `sequentialPeakGb` is read PER TIER by `vram_gate::predicted_sequential_peak_gb`
        // (`sequential.get(tier_key)`), so a bare scalar looks up as `None` and the staged floor
        // silently falls back to the RESIDENT peak — which can never fit where resident does not,
        // making this test's premise unsatisfiable by construction. Same map shape the sibling
        // `eligible_lens_selector_contract_...` fixture uses.
        const STAGED_ROW_GB: f64 = 2.5;
        const RESIDENT_PEAK_GB: f64 = 16.0;
        // `evaluate_shared_image_inner` hands the selector a fixed 2 GiB reserved headroom, and
        // `Budget::effective_gb` subtracts it, so the staged floor competes against
        // `free_gb - SELECTOR_RESERVE_GB`.
        const SELECTOR_RESERVE_GB: f64 = 2.0;
        let manifest = json!({ "candle": { "sequentialPeakGb": { "q4": STAGED_ROW_GB } } })
            .as_object()
            .expect("manifest object")
            .clone();
        // Derived through the production formula rather than guessed: the staged row is padded by
        // the allocator reserve and then widened by the candle estimate margin before the fit check.
        // The previous literals (4.0 row, 8.0 free) missed by 0.24 GiB even with the map shape, and
        // a margin re-derivation would silently move the window again.
        let staged_floor_gb = STAGED_ROW_GB + crate::vram_gate::HEADROOM_GB;
        let widened_staged_gb =
            staged_floor_gb * (1.0 + crate::ladder_margin_policy::CANDLE_ESTIMATE_MARGIN);
        let free_gb = widened_staged_gb + SELECTOR_RESERVE_GB + 0.3;
        assert!(
            free_gb - SELECTOR_RESERVE_GB < RESIDENT_PEAK_GB,
            "the budget must stay below the resident estimate, or 'staged fits where resident does \
             not' proves nothing"
        );
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("krea-q4")));
        let evaluation = evaluate_shared_image_inner(
            "krea_2_edit",
            "krea_2_raw",
            &spec,
            true,
            &manifest,
            "q4",
            "edit_image",
            Some("lora"),
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            true,
            false,
            false,
            false,
            Some(VramBudget {
                free_gb,
                total_gb: 32.0,
            }),
            Some(RESIDENT_PEAK_GB),
            0,
            MemoryCacheState::Cold,
            Some("edit_image"),
            Some(contract),
            None,
        )
        .expect("selector evaluation")
        .expect("staged estimate fits where resident does not");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        assert_eq!(evaluation.context.overlay, None);
        assert!(
            evaluation
                .memory
                .expect("staged selection configures generation memory")
                .stage_residency
        );
    }

    #[test]
    fn mage_routes_preserve_text_to_image_and_edit_scope_keys() {
        for engine_id in ["mage_flow_base", "mage_flow", "mage_flow_turbo"] {
            let binding = request_mode(engine_id, "image_generation");
            assert_eq!(binding.mode, MemoryMode::TextToImage, "engine={engine_id}");
            assert_eq!(binding.calibration_key, "text_to_image");
            assert_eq!(binding.scope_key, "text_to_image");
        }

        for engine_id in [
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
        ] {
            let binding = request_mode(engine_id, "edit_image");
            assert_eq!(binding.mode, MemoryMode::Edit, "engine={engine_id}");
            assert_eq!(binding.calibration_key, "edit_image");
            assert_eq!(binding.scope_key, "edit");
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mage_numeric_tiers_bind_only_the_active_q4_precision_floors() {
        for engine_id in [
            "mage_flow_base",
            "mage_flow",
            "mage_flow_turbo",
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
        ] {
            let declared = crate::inference_runtime::media_descriptor(engine_id)
                .unwrap_or_else(|| panic!("missing registered Mage descriptor {engine_id}"))
                .capabilities
                .component_precision_floors;
            assert!(!declared.is_empty(), "engine={engine_id}");
            assert!(
                declared.iter().all(|floor| floor.applies_to(Quant::Q4)),
                "engine={engine_id}"
            );
            assert_eq!(
                numeric_tier(engine_id, "q4")
                    .expect("q4 tier")
                    .component_precision_floors,
                declared,
                "engine={engine_id}"
            );
            for tier in ["q8", "bf16"] {
                assert!(
                    numeric_tier(engine_id, tier)
                        .unwrap_or_else(|| panic!("missing {tier} tier"))
                        .component_precision_floors
                        .is_empty(),
                    "engine={engine_id} tier={tier}"
                );
            }
        }

        for engine_id in [
            "z_image",
            "z_image_turbo",
            "z_image_control",
            "z_image_turbo_control",
            "qwen_image",
            "qwen_image_edit",
            "flux1_schnell",
            "flux1_dev",
            "flux2_dev",
            "flux2_klein_9b",
        ] {
            for tier in ["q4", "q8", "bf16"] {
                assert!(
                    numeric_tier(engine_id, tier)
                        .unwrap_or_else(|| panic!("missing {tier} tier"))
                        .component_precision_floors
                        .is_empty(),
                    "engine={engine_id} tier={tier}"
                );
            }
        }
    }

    #[test]
    fn mage_manifests_declare_unverified_capabilities_without_calibrations() {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(source);
        let root: Value = serde_json::from_str(&stripped).expect("model manifest parses");
        let models = root["models"].as_array().expect("models array");

        for model_id in [
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
            "mage_flow_base",
            "mage_flow",
            "mage_flow_turbo",
        ] {
            let candle = models
                .iter()
                .find(|model| model["id"] == model_id)
                .unwrap_or_else(|| panic!("missing Mage model {model_id}"))["candle"]
                .as_object()
                .expect("Candle manifest block");
            assert_eq!(candle["supportsSequentialOffload"], true);
            assert_eq!(candle["measured"], false);
            assert!(candle.get("calibrations").is_none());

            let capabilities = candle["memoryStrategyCapabilities"]
                .as_object()
                .expect("memory strategy capabilities");
            assert_eq!(capabilities.len(), 2);
            assert!(
                !capabilities.contains_key("bounded_decode"),
                "{model_id}: CoD full-field normalization makes bounded decode structurally inapplicable"
            );
            let bounded_decode_exemption =
                &candle["memoryStrategyStructuralExemptions"]["bounded_decode"];
            assert_eq!(
                bounded_decode_exemption["overlays"],
                json!(["none", "lora"])
            );
            assert_eq!(
                bounded_decode_exemption["evidence"]
                    .as_array()
                    .expect("bounded decode structural evidence")
                    .len(),
                2
            );
            assert_eq!(
                capabilities["bounded_attention"]["parameters"],
                json!({ "attentionChunkSize": 67_108_864 })
            );
            assert_eq!(
                capabilities["bounded_transformer_residency"]["parameters"],
                json!({ "transformerWindowSize": 1, "transformerWindowComponent": "Dit" })
            );
            assert!(capabilities
                .values()
                .all(|capability| capability["overlays"] == json!(["none"])));
        }
    }

    #[test]
    fn character_identity_and_control_bindings_require_the_canonical_exact_keys() {
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 1,
        };
        let binding = |overlay| {
            json!({
                "provider": "flux1_dev",
                "tier": "q4",
                "mode": "character_image",
                "overlay": overlay,
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            })
            .as_object()
            .expect("binding object")
            .clone()
        };

        for overlay in ["identity", "control"] {
            let exact = binding(overlay);
            assert!(binding_matches_request(
                &exact,
                "flux1_dev",
                "q4",
                "character_image",
                overlay,
                geometry,
            ));
            assert!(!binding_matches_request(
                &exact,
                "flux1_dev",
                "q4",
                "image_to_image",
                overlay,
                geometry,
            ));
        }

        let identity = binding("identity");
        assert!(!binding_matches_request(
            &identity,
            "flux1_dev",
            "q4",
            "character_image",
            "ip_adapter",
            geometry,
        ));
    }

    #[test]
    fn flux1_base_bindings_are_current_selectable_and_overlay_free() {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(source);
        let root: Value = serde_json::from_str(&stripped).expect("model manifest parses");
        let models = root["models"].as_array().expect("models array");
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };

        for (model_id, provider) in [("flux_schnell", "flux1_schnell"), ("flux_dev", "flux1_dev")] {
            let model = models
                .iter()
                .find(|model| model["id"] == model_id)
                .expect("FLUX model");
            let manifest = model.as_object().expect("model object");
            let bindings = manifest["candle"]["calibrations"]
                .as_array()
                .expect("calibration bindings");
            assert_eq!(bindings.len(), 5);
            assert!(bindings.iter().all(|binding| binding["overlay"] == "none"));

            let candidates = verified_candidates(
                manifest,
                model_id,
                provider,
                "q4",
                &request_mode(provider, "text_to_image"),
                "none",
                geometry,
                &mut Vec::new(),
            )
            .expect("packaged FLUX evidence");
            assert_eq!(candidates.len(), 5);
            assert_eq!(
                candidates
                    .iter()
                    .map(|candidate| candidate.key.strategy)
                    .collect::<Vec<_>>(),
                vec![
                    MemoryStrategy::Resident,
                    MemoryStrategy::StagedResidency,
                    MemoryStrategy::BoundedDecode,
                    MemoryStrategy::BoundedAttention,
                    MemoryStrategy::BoundedTransformerResidency,
                ]
            );
            assert!(matches!(candidates[0].parity, MemoryParityContract::Exact));
            assert!(matches!(candidates[1].parity, MemoryParityContract::Exact));
            assert!(candidates[2..].iter().all(|candidate| matches!(
                candidate.parity,
                MemoryParityContract::Tolerance { .. }
            )));

            for overlay in ["identity", "control"] {
                assert!(verified_candidates(
                    manifest,
                    model_id,
                    provider,
                    "q4",
                    &request_mode(provider, "character_image"),
                    overlay,
                    MemoryGeometry {
                        reference_count: 1,
                        ..geometry
                    },
                    &mut Vec::new(),
                )
                .expect("uncertified overlay query")
                .is_empty());
            }
        }
    }

    #[test]
    fn flux2_dev_v2_bindings_stay_historical_but_cannot_enter_the_v3_selector() {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(source);
        let root: Value = serde_json::from_str(&stripped).expect("model manifest parses");
        let model = root["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|model| model["id"] == "flux2_dev")
            .expect("FLUX.2-dev model");
        let manifest = model.as_object().expect("FLUX.2-dev model object");
        let bindings = manifest["candle"]["calibrations"]
            .as_array()
            .expect("FLUX.2-dev calibration bindings");
        assert_eq!(bindings.len(), 5);
        assert!(bindings.iter().all(|binding| {
            binding["provider"] == "flux2_dev"
                && binding["tier"] == "q4"
                && binding["mode"] == "text_to_image"
                && binding["overlay"] == "none"
                && binding["inferenceRevision"] == "5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d"
                // sc-17774: `compatibleInferenceRevision` is gone — it was flux2_dev's one-shot
                // hand-audited hatch. The binding now carries the closure digest it was captured
                // under, which is what currency compares.
                && binding["inferenceClosureDigest"].is_string()
        }));

        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let candidates = verified_candidates(
            manifest,
            "flux2_dev",
            "flux2_dev",
            "q4",
            &request_mode("flux2_dev", "text_to_image"),
            "none",
            geometry,
            &mut Vec::new(),
        )
        .expect("packaged FLUX.2-dev evidence");
        assert_eq!(candidates.len(), 5);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.predicted_peak_bytes)
                .collect::<Vec<_>>(),
            vec![
                47_700_000_000,
                44_300_000_000,
                34_700_000_000,
                25_200_000_000,
                14_300_000_000
            ]
        );
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.observed_peak_bytes)
                .collect::<Vec<_>>(),
            vec![
                Some(44_911_779_404),
                Some(34_070_566_540),
                Some(30_325_300_672),
                Some(23_724_359_104),
                Some(13_537_107_920),
            ]
        );
        assert!(candidates.iter().all(|candidate| {
            candidate
                .observed_peak_bytes
                .is_some_and(|active| candidate.predicted_peak_bytes > active)
        }));
        let mut unaudited_manifest = manifest.clone();
        for binding in unaudited_manifest["candle"]["calibrations"]
            .as_array_mut()
            .expect("mutable FLUX.2-dev calibration bindings")
        {
            // sc-17774: as above — the deleted hatch cannot make a binding unaudited any more, so
            // move the mutation onto the closure digest currency actually compares.
            binding["inferenceClosureDigest"] = json!("a".repeat(64));
        }
        assert!(verified_candidates(
            &unaudited_manifest,
            "flux2_dev",
            "flux2_dev",
            "q4",
            &request_mode("flux2_dev", "text_to_image"),
            "none",
            geometry,
            &mut Vec::new(),
        )
        .expect("unaudited compatibility query")
        .is_empty());
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.key.strategy)
                .collect::<Vec<_>>(),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
        assert!(verified_candidates(
            manifest,
            "flux2_dev",
            "flux2_dev",
            "q4",
            &request_mode("flux2_dev", "text_to_image"),
            "control",
            geometry,
            &mut Vec::new(),
        )
        .expect("uncertified FLUX.2-dev control query")
        .is_empty());

        // The five exact historical bindings remain queryable for auditability. Admission is a
        // separate step: their v2 calibration identity no longer describes the live v3
        // caption-upsample lifecycle, so none of these ladder rungs may reach the selector.
        let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("flux2-dev-q4-fixture")))
            .with_quant(Quant::Q4);
        spec.load_shape = gen_core::LoadShape::DeferredMaterialization;
        let live_predicted_peak = crate::vram_gate::predicted_peak_gb(manifest, "q4")
            .expect("measured FLUX.2-dev q4 high-water");
        let evaluate = |free_gb: f64| {
            evaluate_shared_image(
                "flux2_dev",
                "flux2_dev",
                &spec,
                true,
                manifest,
                "q4",
                "text_to_image",
                None,
                geometry,
                false,
                false,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                }),
                Some(live_predicted_peak),
                0,
                MemoryCacheState::Cold,
            )
            .expect("FLUX.2-dev safe-device-peak evaluation")
        };
        for budget_gb in [16.0, 24.0] {
            assert!(
                evaluate(budget_gb).is_none(),
                "the obsolete v2 ladder must not reopen live v3 FLUX.2-dev on a {budget_gb} GB budget"
            );
        }

        assert!(
            candidates.iter().all(|candidate| candidate
                .calibration_fingerprint
                .ends_with("device-format-blocks-v2")),
            "the retained packaged rows must remain truthfully labeled as v2 history"
        );
        let live = evaluate(96.0).expect("live v3 FLUX.2-dev resident route");
        assert_eq!(live.context.selection.strategy, MemoryStrategy::Resident);
        assert_eq!(
            live.context.calibration_fingerprint,
            "flux2-dev-cuda-caption-upsample-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v3"
        );
    }

    /// sc-18097 headline (epic 18093 R1b): an UNMEASURED provider cell — no packaged calibration
    /// records at all — under a small emulated VRAM budget (`SCENEWORKS_CUDA_VRAM_CAP_GB`
    /// scenario, driven through the same pure seam the cap feeds) engages the ladder by
    /// estimate-floor instead of freezing to resident-or-nothing, and refuses below the widened
    /// floors.
    ///
    /// Floor arithmetic: resident estimate 8.0 GiB (caller-predicted, manifest row); staged floor
    /// = `sequentialPeakGb` row 2.5 + 2.0 headroom = 4.5 GiB, widened by the 4% candle estimate
    /// margin to 4.68. A 7 GiB budget (5 GiB effective after the selector's 2 GiB reserve) admits
    /// exactly the staged floor.
    #[test]
    fn unmeasured_provider_under_a_small_budget_engages_the_estimate_floor_ladder() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        let evaluate_as = |free_gb: f64, artifact_is_certified: bool| {
            evaluate_shared_image(
                "z_image_turbo",
                "z_image_turbo",
                &spec,
                artifact_is_certified,
                &manifest,
                "q4",
                "text_to_image",
                None,
                MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                false,
                false,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                }),
                Some(8.0),
                0,
                MemoryCacheState::Cold,
            )
            .expect("unmeasured z-image evaluation")
        };
        let evaluate = |free_gb: f64| evaluate_as(free_gb, true);

        let evaluation = evaluate(7.0).expect(
            "an implemented rung's estimate floor must admit where the resident estimate cannot",
        );
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::StagedResidency,
            "the cheapest fitting estimate rung must win"
        );
        // The floor is the manifest staged row + headroom, raw (the selector owns the widening).
        assert!((evaluation.predicted_peak_gb - 4.5).abs() < 1e-6);
        let memory = evaluation
            .memory
            .expect("optimized selection carries memory");
        assert!(memory.stage_residency);
        assert!(!memory.tile_vae_decode);
        // Estimate admissions are legacy-scoped in telemetry, exactly like the resident estimate.
        assert_eq!(
            evaluation.context.evidence_revision,
            Z_IMAGE_REQUEST_EVIDENCE_REVISION
        );

        // Margin mutation arm: at 6.55 GiB free (4.55 effective) the RAW staged floor (4.5) fits
        // but the widened one (4.68) does not — the selector rejects, this lane falls back to the
        // established legacy gates (`None`), and a zeroed estimate margin would admit instead and
        // flip this arm red.
        assert!(
            evaluate(6.55).is_none(),
            "an estimate must be graded at its WIDENED peak, not its raw floor"
        );

        // Control: the resident estimate itself still admits on a roomy card without engaging any
        // rung — the floors extend the ladder downward, they do not perturb the fast path.
        let roomy = evaluate(64.0).expect("resident admits on a roomy card");
        assert_eq!(roomy.context.selection.strategy, MemoryStrategy::Resident);
        assert!(roomy.memory.is_none());

        // sc-18097 review (major): the floors are gated on artifact certification, the SAME
        // conjunct that gates the packaged records. The manifest rows describe the certified
        // bytes; an imported/community checkpoint on this route is different bytes, so at the
        // exact budget where the certified artifact engages a floor rung, the uncertified one
        // gets no floors, its resident estimate does not fit, and the lane hands back to the
        // established gates (`None`). Removing the certification conjunct flips this red.
        assert!(
            evaluate_as(7.0, false).is_none(),
            "an uncertified artifact must not engage a rung off the certified manifest's rows"
        );
        // …and it keeps its pre-sc-18097 resident behavior where the resident estimate DOES fit,
        // so the gate withholds the floors rather than refusing the artifact.
        let uncertified_roomy =
            evaluate_as(64.0, false).expect("an uncertified artifact still admits resident");
        assert_eq!(
            uncertified_roomy.context.selection.strategy,
            MemoryStrategy::Resident
        );
    }

    /// A synthetic provider contract with every rung implemented and rungs 2-4 pinned to either
    /// side of the staging composition split (sc-18253). `bind_deep_rungs_to_staging` mirrors the
    /// shipped z-image contract's `additional_prerequisites` edges; without them the gen-core
    /// default composition excludes `StagedResidency` from every deep rung. `staged_implemented`
    /// false models the finding's contract — a provider implementing deep rungs with no staging
    /// rung at all.
    fn composition_probe_contract(
        staged_implemented: bool,
        bind_deep_rungs_to_staging: bool,
    ) -> gen_core::MemoryProviderContract {
        let mut contract = gen_core::MemoryProviderContract::compatibility_default(
            "z_image_turbo",
            gen_core::MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| gen_core::MemoryStrategyCapability {
                strategy,
                support: if strategy == MemoryStrategy::StagedResidency && !staged_implemented {
                    gen_core::MemoryStrategySupport::Missing
                } else {
                    gen_core::MemoryStrategySupport::Implemented
                },
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => gen_core::MemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![128],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => gen_core::MemoryParameterRanges {
                        attention_chunk_sizes: vec![1024],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency => {
                        gen_core::MemoryParameterRanges {
                            transformer_window_sizes: vec![1],
                            ..Default::default()
                        }
                    }
                    _ => Default::default(),
                },
            })
            .collect();
        contract.lifecycle = gen_core::MemoryLifecycleCapabilities {
            phases: vec![
                gen_core::MemoryPhase::Conditioning,
                gen_core::MemoryPhase::Denoise,
                gen_core::MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        };
        // Rung 4's SHARED prerequisite is the deferred load shape, not staged residency — which is
        // exactly why a rung-4-without-staging contract is a legal shape.
        contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
        contract.formula = gen_core::MemoryFormulaKind::AssetBytesPlusHeadroom;
        contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
            "sc-18253-composition-probe-v1",
            gen_core::LoadShape::DeferredMaterialization,
        ));
        if bind_deep_rungs_to_staging {
            contract.additional_prerequisites = [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
            .into_iter()
            .map(|strategy| {
                (
                    strategy,
                    gen_core::MemoryStrategyPrerequisite::Rung {
                        rung: MemoryStrategy::StagedResidency,
                        scope: gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
                    },
                )
            })
            .collect();
        }
        contract
    }

    #[test]
    fn eligible_lens_selector_contract_can_select_sequential_without_mutating_the_load_spec() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-lens-q4")))
            .with_quant(Quant::Q4)
            .with_resolved_route("lens")
            .with_eligible_load_shape_declaration();
        let mut contract = composition_probe_contract(true, true);
        contract.provider_id = "lens".to_owned();
        let evaluation = evaluate_shared_image_inner(
            "lens",
            "lens",
            &spec,
            true,
            &manifest,
            "q4",
            "text_to_image",
            None,
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            false,
            false,
            Some(VramBudget {
                free_gb: 7.0,
                total_gb: 96.0,
            }),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
            None,
            Some(contract),
            None,
        )
        .expect("weights-free Lens selector")
        .expect("the staged estimate fits while resident does not");
        assert_ne!(
            evaluation.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert!(
            evaluation
                .memory
                .expect("optimized selection carries request memory")
                .stage_residency
        );
        assert_eq!(
            spec.load_shape_declaration_result,
            gen_core::LoadShapeDeclarationResult::Eligible
        );
        assert_eq!(spec.load_shape, gen_core::LoadShape::EagerMaterialization);
    }

    /// sc-18253: the staged manifest row prices the staged WORKING SET, so it is only a sound
    /// floor for a composition that actually engages `StagedResidency`. The gen-core engagement
    /// mechanism permits a provider to implement a deep rung whose engaged composition excludes
    /// staging — such a request runs whole-model resident, so its floor must clamp to the
    /// RESIDENT estimate (the candle mirror of the MLX floor's
    /// `engaged.contains(&StagedResidency)` max-vs-sum split) instead of under-predicting a whole
    /// resident working set behind only the estimate margin.
    ///
    /// Both mutation directions are pinned:
    ///  * deleting the composition check (every deep rung takes the staged row again) flips the
    ///    staging-free contract's deep-rung assertions red;
    ///  * inverting it (clamping every optimized floor to the resident row) flips the staged
    ///    rung's own assertion and the staging-bound contract's deep-rung assertions red.
    #[test]
    fn deep_rung_floors_follow_the_engaged_composition_not_the_rung_ordinal() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let resident_peak_bytes = (8.0 * BYTES_PER_GIB) as u64;
        let staged_floor_bytes =
            ((2.5 + crate::vram_gate::HEADROOM_GB) * BYTES_PER_GIB).ceil() as u64;
        assert_ne!(
            resident_peak_bytes, staged_floor_bytes,
            "the two floor sources must be distinguishable for the assertions below to bite"
        );
        let floors = |contract: &gen_core::MemoryProviderContract| {
            synthesize_estimate_floors(
                "z_image_turbo",
                contract,
                &manifest,
                "q4",
                numeric_tier("z_image_turbo", "q4").expect("q4 tier"),
                &request_mode("z_image_turbo", "text_to_image"),
                None,
                geometry,
                resident_peak_bytes,
                0,
                Z_IMAGE_REQUEST_EVIDENCE_REVISION,
            )
        };
        let floor_of = |synthesized: &[(MemorySelection, MemoryEvidence)],
                        strategy: MemoryStrategy| {
            synthesized
                .iter()
                .find(|(selection, _)| selection.strategy == strategy)
                .map(|(_, evidence)| evidence.predicted_peak_bytes)
        };
        const DEEP_RUNGS: [MemoryStrategy; 3] = [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ];

        // Staging-free deep rungs (the gen-core default composition — no staged prerequisite
        // edges): the staged row must not price them.
        let synthesized = floors(&composition_probe_contract(true, false));
        assert_eq!(
            floor_of(&synthesized, MemoryStrategy::StagedResidency),
            Some(staged_floor_bytes),
            "the staged rung itself still takes the staged working-set row"
        );
        for deep in DEEP_RUNGS {
            assert_eq!(
                floor_of(&synthesized, deep),
                Some(resident_peak_bytes),
                "a {deep:?} composition that excludes staging runs whole-model resident and must \
                 clamp to the resident estimate, not the staged working-set row"
            );
        }

        // Control: a contract that binds every deep rung to staging (the shipped z-image shape)
        // keeps the staged working-set floor on those rungs — clamping regardless of composition
        // would flip these red.
        let synthesized = floors(&composition_probe_contract(true, true));
        for rung in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                floor_of(&synthesized, rung),
                Some(staged_floor_bytes),
                "a {rung:?} composition that engages staging keeps the staged working-set floor"
            );
        }
    }

    /// The end-to-end arm of sc-18253: the exact contract the finding names — deep rungs
    /// implemented, no staging rung at all — must NOT admit a request on the staged row. Its
    /// floors clamp to the resident estimate, nothing fits below the resident peak, and the lane
    /// hands back to the established legacy gates (`None`). Before the composition check the
    /// staging-free `BoundedDecode` floor took the staged row and ADMITTED here at a
    /// whole-model-resident working set behind only the 4% estimate margin — deleting the check
    /// flips this arm red.
    #[test]
    fn a_staging_free_ladder_never_admits_on_the_staged_row() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        // A budget where the WIDENED staged-row floor fits but the resident estimate (8.0 GiB)
        // does not — recomputed from the policy margin, never a frozen literal.
        let staged_row_floor_gb = 2.5 + crate::vram_gate::HEADROOM_GB;
        let free_gb = staged_row_floor_gb
            * (1.0 + crate::ladder_margin_policy::CANDLE_ESTIMATE_MARGIN)
            + crate::vram_gate::HEADROOM_GB
            + 0.3;
        assert!(
            free_gb - crate::vram_gate::HEADROOM_GB < 8.0,
            "the budget must stay below the resident estimate to discriminate"
        );
        let evaluation = evaluate_shared_bespoke_image(
            "z_image_turbo",
            "z_image_turbo",
            &spec,
            true,
            &manifest,
            "q4",
            "text_to_image",
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            false,
            Some(VramBudget {
                free_gb,
                total_gb: 96.0,
            }),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
            composition_probe_contract(false, false),
        )
        .expect("staging-free bespoke evaluation");
        assert!(
            evaluation.is_none(),
            "a staging-free deep rung must be graded at the resident clamp, not admitted on the \
             staged working-set row"
        );
    }

    /// An uncertified (imported / community) FLUX identity artifact never reaches an optimized
    /// rung — not through packaged records, and since sc-18097 not through estimate floors either.
    ///
    /// The budget is DELIBERATELY tight and the manifest populated (sc-18097 review): on a roomy
    /// card with an empty manifest the resident estimate wins for everyone, so the assertion could
    /// not tell a withheld floor from a floor that simply never competed. Here the certified
    /// control admits a floor rung at exactly the budget where the uncertified artifact must not —
    /// removing the certification conjunct from the floor guard turns the second arm red.
    #[test]
    fn uncertified_flux_identity_artifacts_fall_back_to_resident() {
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("alternate-flux-q4")))
            .with_ip_adapter(WeightsSource::File(PathBuf::from(
                "alternate-ip-adapter.safetensors",
            )))
            .with_component(
                "flux_ip_image_encoder",
                WeightsSource::Dir(PathBuf::from("alternate-image-encoder")),
            )
            .with_offload_policy(gen_core::OffloadPolicy::Sequential);
        // Staged floor = `sequentialPeakGb` 2.5 + 2.0 headroom = 4.5 GiB, widened by the 4% candle
        // estimate margin to 4.68; the resident estimate is 8.0. A 7 GiB budget (5.0 effective)
        // therefore separates "floor available" from "resident only".
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let evaluate = |artifact_is_certified: bool, free_gb: f64| {
            evaluate_shared_image(
                "flux1_dev",
                "flux_dev",
                &spec,
                artifact_is_certified,
                &manifest,
                "q4",
                "character_image",
                Some("identity"),
                MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 1,
                },
                true,
                false,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 32.0,
                }),
                Some(8.0),
                0,
                MemoryCacheState::Cold,
            )
            .expect("FLUX identity evaluation")
        };

        // Roomy card: resident for both, with the typed scope preserved.
        let evaluation = evaluate(false, 32.0).expect("resident fallback");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert_eq!(
            evaluation.context.mode,
            MemoryMode::Other("character_image".to_owned())
        );
        assert_eq!(evaluation.context.overlay.as_deref(), Some("identity"));
        assert!(evaluation.memory.is_none());

        // The discriminating pair at 7 GiB free: certified engages the staged floor…
        let certified = evaluate(true, 7.0)
            .expect("a certified artifact's estimate floor must admit at this budget");
        assert_eq!(
            certified.context.selection.strategy,
            MemoryStrategy::StagedResidency,
            "the control arm must actually reach a rung, or the assertion below proves nothing"
        );
        // …and uncertified gets no floors at all, so the lane hands back to the established gates.
        assert!(
            evaluate(false, 7.0).is_none(),
            "an uncertified artifact must not engage a rung off the certified manifest's rows"
        );
    }

    #[test]
    fn edit_alias_enters_turbo_contract_as_an_edit_reference_scope() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 8.0 },
                "supportsSequentialOffload": true,
                "memoryStrategyCapabilities": {
                    "bounded_decode": {
                        "parameters": { "decodeTileEdge": 512, "decodeOverlap": 128 },
                        "overlays": ["none", "lora"]
                    }
                }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        let evaluation = evaluate_shared_image(
            "z_image_turbo",
            "z_image_edit",
            &spec,
            true,
            &manifest,
            "q4",
            "edit_image",
            None,
            MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            true,
            false,
            false,
            false,
            Some(VramBudget {
                free_gb: 32.0,
                total_gb: 32.0,
            }),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
        )
        .unwrap()
        .expect("resident alias selection");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert_eq!(evaluation.context.mode, MemoryMode::Edit);
        assert!(evaluation.context.has_reference);
        assert_eq!(
            evaluation.context.calibration_fingerprint,
            "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2"
        );
        assert!(evaluation.memory.is_none());
    }

    #[test]
    fn hires_fix_does_not_reuse_one_geometry_scope_for_both_passes() {
        let manifest = JsonObject::new();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        let evaluation = evaluate_shared_image(
            "z_image",
            "z_image",
            &spec,
            true,
            &manifest,
            "q4",
            "text_to_image",
            None,
            MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            true,
            false,
            Some(VramBudget {
                free_gb: 32.0,
                total_gb: 32.0,
            }),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
        )
        .unwrap();
        assert!(evaluation.is_none());

        // The route MUST be the probe contract's own provider (`z_image_turbo`), the same pairing
        // the sibling `evaluate_shared_bespoke_image` probe uses. `candidate_exclusion` fails closed
        // on `request.resolved_route != contract.provider_id` with a structural `Invalid`, so a
        // mismatched pair excludes every candidate and returns `None` — which would have made the
        // two `is_none()` assertions below pass for a reason that has nothing to do with the
        // Hires.fix / phases axes they exist to probe.
        let evaluate_axes = |worker_multipass: bool, request_has_phases: bool| {
            evaluate_shared_image_inner(
                "z_image_turbo",
                "z_image_turbo",
                &spec,
                true,
                &manifest,
                "q4",
                "text_to_image",
                None,
                None,
                MemoryGeometry {
                    width: 512,
                    height: 512,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                false,
                false,
                worker_multipass,
                request_has_phases,
                Some(VramBudget {
                    free_gb: 32.0,
                    total_gb: 32.0,
                }),
                Some(8.0),
                0,
                MemoryCacheState::Cold,
                None,
                Some(composition_probe_contract(true, false)),
                None,
            )
            .expect("request-axis evaluation")
        };
        // Control arm first, or the two `is_none()` assertions below prove nothing: this fixture must
        // actually be selectable when neither axis is set.
        let plain = evaluate_axes(false, false)
            .expect("the control arm must reach a selection, or the None assertions are vacuous");
        assert!(!plain.context.has_phases);
        assert!(
            evaluate_axes(true, false).is_none(),
            "Hires.fix alone must remain outside one-scope declaration authority"
        );
        assert!(
            evaluate_axes(true, true).is_none(),
            "Hires.fix plus GenerationRequest phases cannot collapse into one scope"
        );
        let phases = evaluate_axes(false, true).expect("actual request phases stay selectable");
        assert!(phases.context.has_phases);
    }

    #[test]
    fn optimized_candidates_reserve_the_actual_runtime_adapter_bytes() {
        let mut evidence = MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: "z_image".to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier("z_image", "q4").expect("q4 is a supported numeric tier"),
                load_shape: gen_core::LoadShape::EagerMaterialization,
                mode: gen_core::MemoryMode::TextToImage,
                overlay: Some("lora".to_owned()),
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    // Text-to-image fixture: no reference images (sc-17054).
                    reference_count: 0,
                },
                strategy: MemoryStrategy::BoundedTransformerResidency,
                engaged_composition: vec![MemoryStrategy::BoundedTransformerResidency],
                parameters: Default::default(),
            },
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions::VERIFIED,
            calibration_abi: 1,
            calibration_fingerprint: "fixture".to_owned(),
            sceneworks_revision: "fixture".to_owned(),
            inference_revision: "fixture".to_owned(),
            harness_version: "fixture".to_owned(),
            predicted_peak_bytes: 8 * 1024,
            observed_peak_bytes: Some(8 * 1024),
            parity: MemoryParityContract::Exact,
            parity_result: MemoryParityResult::Passed,
        };

        account_for_runtime_overlay_bytes(std::slice::from_mut(&mut evidence), 512);

        assert_eq!(evidence.predicted_peak_bytes, 8 * 1024 + 512);
    }
}
