//! Candle image-provider adoption of the shared request-scoped memory selector.
//!
//! Static capability declarations never authorize an optimized request. The only optimized
//! candidates admitted here are exact authoritative records from the packaged evidence bundle;
//! every other state falls back to the resident candidate and the established Candle gate.

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
const PULID_FLUX_REQUEST_EVIDENCE_REVISION: &str = "sc-15839-candle-pulid-flux-request-scope-v1";
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

pub(crate) struct CandleMemoryEvaluation {
    pub memory: Option<GenerationMemory>,
    pub context: MemoryRunContext,
    pub predicted_peak_gb: f64,
}

fn numeric_tier(tier: &str) -> Option<MemoryNumericTier> {
    let quant = match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        _ => return None,
    };
    Some(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
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
            tier: numeric_tier(tier).expect("validated numeric tier"),
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
                RequiredNullable::Value(observed) => {
                    Some(observed.overall_device_or_active_bytes())
                }
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
    budget: Option<VramBudget>,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
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
        geometry,
        has_reference,
        use_pid,
        has_phases,
        budget,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        None,
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
        geometry,
        has_reference,
        use_pid,
        has_phases,
        budget,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
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
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    budget: Option<VramBudget>,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    contract_override: Option<gen_core::MemoryProviderContract>,
    request_evidence_revision_override: Option<&'static str>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    if !matches!(
        engine_id,
        "z_image"
            | "z_image_turbo"
            | "z_image_control"
            | "z_image_turbo_control"
            | "qwen_image"
            | "qwen_image_edit"
            | "flux1_schnell"
            | "flux1_dev"
            | "flux2_dev"
            | "flux2_klein_9b"
    ) && contract_override.is_none()
    {
        return Ok(None);
    }
    let request_evidence_revision = request_evidence_revision_override.unwrap_or(match engine_id {
        "qwen_image" | "qwen_image_edit" => QWEN_IMAGE_REQUEST_EVIDENCE_REVISION,
        "flux1_schnell" | "flux1_dev" => FLUX1_REQUEST_EVIDENCE_REVISION,
        "flux2_dev" => FLUX2_DEV_REQUEST_EVIDENCE_REVISION,
        "flux2_klein_9b" => FLUX2_KLEIN_REQUEST_EVIDENCE_REVISION,
        _ => Z_IMAGE_REQUEST_EVIDENCE_REVISION,
    });
    // Hires-fix is two independently shaped denoise passes. The generic generation harness still
    // reuses one context for both, so it cannot truthfully carry a request-scoped geometry yet.
    // Keep that surface on its established path until it mints one scope per pass.
    if has_phases {
        return Ok(None);
    }
    let mode = request_mode(engine_id, request_mode_value);
    let Some(tier) = numeric_tier(tier_key) else {
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
            overlay: overlay.map(str::to_owned),
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
    let mut selections = Vec::with_capacity(verified.len() + 1);
    let mut evidence = Vec::with_capacity(verified.len() + 1);
    // The resident candidate is a live estimate, not a calibrated record, so it carries the live
    // digest and is never staled by this gate.
    let live_closure_digest = sceneworks_core::memory_calibration::packaged_closure_digest(
        "candle",
        evidence_provider(engine_id),
    )
    .unwrap_or_default();
    let mut candidate_digests = Vec::with_capacity(verified.len() + 1);
    selections.push(resident_selection);
    evidence.push(&resident);
    candidate_digests.push(live_closure_digest.clone());
    for (index, item) in verified.iter().enumerate() {
        candidate_digests.push(closure_digests.get(index).cloned().unwrap_or_default());
        selections.push(MemorySelection {
            strategy: item.key.strategy,
            parameters: item.key.parameters,
            tier: item.key.tier,
        });
        evidence.push(item);
    }
    let candidates = selections
        .iter()
        .zip(evidence)
        .zip(&candidate_digests)
        .map(|((selection, evidence), closure_digest)| Candidate {
            selection: *selection,
            evidence,
            closure_digest,
        })
        .collect::<Vec<_>>();
    let selected = crate::memory_strategy::select_strategy(
        RequestScope {
            resolved_route: engine_id,
            backend: "candle",
            tier,
            mode: &mode.scope_key,
            overlay,
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
    let selected_evidence = if selection.strategy == MemoryStrategy::Resident {
        &resident
    } else {
        verified
            .iter()
            .find(|item| {
                item.key.strategy == selection.strategy
                    && item.key.parameters == selection.parameters
                    && item.key.tier == selection.tier
            })
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{engine_id} selected a memory strategy without exact packaged evidence"
                ))
            })?
    };
    let to_bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
    Ok(Some(CandleMemoryEvaluation {
        memory: memory_for_selection(&contract, selection),
        predicted_peak_gb: selected_evidence.predicted_peak_bytes as f64 / BYTES_PER_GIB,
        context: MemoryRunContext {
            selection,
            calibration_abi: selected_evidence.calibration_abi,
            calibration_fingerprint: selected_evidence.calibration_fingerprint.clone(),
            mode: mode.mode,
            load_shape: contract.load_shape,
            has_reference,
            use_pid,
            has_phases,
            geometry,
            overlay: overlay.map(str::to_owned),
            budget: gen_core::MemoryBudget {
                total_bytes: to_bytes(budget.total_gb),
                committed_bytes: to_bytes((budget.total_gb - budget.free_gb).max(0.0)),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: to_bytes(2.0),
            },
            predicted_peak_bytes: selected_evidence.predicted_peak_bytes,
            cache_state,
            evidence_revision: if selection.strategy == MemoryStrategy::Resident {
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
    fn flux2_dev_base_bindings_are_current_selectable_and_overlay_free() {
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

        // At 40 GiB free, the staged allocator high-water plus the selector's 2 GiB reserve
        // would fit, but its 44.3 GB conservative device peak does not. Admission must advance
        // to bounded decode, whose 34.7 GB device peak does fit.
        let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("flux2-dev-q4-fixture")))
            .with_quant(Quant::Q4);
        spec.load_shape = gen_core::LoadShape::DeferredMaterialization;
        let evaluation = evaluate_shared_image(
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
            Some(VramBudget {
                free_gb: 40.0,
                total_gb: 96.0,
            }),
            Some(44.0),
            0,
            MemoryCacheState::Cold,
        )
        .expect("FLUX.2-dev safe-device-peak evaluation");

        // sc-17774: everything above is closure-independent — the packaged evidence is RETAINED
        // whatever the pin is, which is why `verified_candidates` still returns all five rungs.
        // ADMISSION is the step that carries the currency term.
        //
        // `flux2_dev`'s packaged bindings were captured at `5ffd7612`, and its compile closure HAS
        // moved since. That is not a defect in this branch and not a fixture problem: sc-17760
        // reached the identical verdict by an independent method (a linked-artifact digest on the
        // RTX box, ARTIFACTS DIFFER), and `main`'s own generated matrix already shows those five q4
        // cells at `Implemented/unverified`. The demotion stands until sc-15922 re-captures.
        //
        // So the honest assertion is that admission REFUSES, and the reason is the closure. An
        // earlier revision of this branch asserted a fit here on the belief that the binding and the
        // packaged closure table agreed — they do not, and the belief was never checked.
        //
        // This replaces a block that skipped the rest of the test whenever the pin had moved past
        // the audited window, which meant the one lane with a hand-audited hatch was also the one
        // whose selector test went dark after every bump. Refusal is asserted rather than skipped.
        assert!(
            evaluation.is_none(),
            "flux2_dev's closure moved since 5ffd7612, so its packaged evidence must not admit an \
             optimized strategy at the live pin"
        );
    }

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
        let evaluation = evaluate_shared_image(
            "flux1_dev",
            "flux_dev",
            &spec,
            false,
            &JsonObject::new(),
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
            Some(VramBudget {
                free_gb: 32.0,
                total_gb: 32.0,
            }),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
        )
        .expect("FLUX identity evaluation")
        .expect("resident fallback");

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
    }

    #[test]
    fn optimized_candidates_reserve_the_actual_runtime_adapter_bytes() {
        let mut evidence = MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: "z_image".to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier("q4").expect("q4 is a supported numeric tier"),
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
