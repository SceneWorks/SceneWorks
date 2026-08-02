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

fn request_mode(mode: &str) -> (MemoryMode, &'static str) {
    match mode {
        "image_generation" | "text_to_image" => (MemoryMode::TextToImage, "text_to_image"),
        "style_variations" => (MemoryMode::ImageToImage, "style_variations"),
        "character_image" | "image_to_image" => (MemoryMode::ImageToImage, "image_to_image"),
        "edit_image" => (MemoryMode::Edit, "edit"),
        _ => (MemoryMode::Other(mode.to_owned()), "other"),
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

fn parse_parameters(
    rung: StrategyRung,
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
    let required: &[(&str, u32)] = match rung {
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
    let transformer_window_component = match parameters.get("transformerWindowComponent") {
        None => None,
        Some(Value::String(value)) if rung == StrategyRung::BoundedTransformerResidency => {
            Some(match value.as_str() {
                "dit" => gen_core::TransformerComponent::Dit,
                "text_encoder" => gen_core::TransformerComponent::TextEncoder,
                "both" => gen_core::TransformerComponent::Both,
                _ => return None,
            })
        }
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

fn verified_candidates(
    manifest: &JsonObject<String, Value>,
    model_id: &str,
    runtime_provider: &str,
    tier: &str,
    mode: &str,
    overlay: &str,
    geometry: MemoryGeometry,
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
        if text(item, "provider") != Some(evidence_provider)
            || text(item, "tier") != Some(tier)
            || text(item, "mode") != Some(mode)
            || text(item, "overlay") != Some(overlay)
        {
            continue;
        }
        let Some(item_geometry) = item.get("geometry").and_then(Value::as_object) else {
            continue;
        };
        let exact_geometry = ["width", "height", "batch", "frames"]
            .into_iter()
            .zip([
                geometry.width,
                geometry.height,
                geometry.batch,
                geometry.frames,
            ])
            .all(|(key, expected)| {
                item_geometry.get(key).and_then(Value::as_u64) == Some(u64::from(expected))
            });
        if !exact_geometry {
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
        let Some(selection_parameters) = parse_parameters(rung, &parameters) else {
            continue;
        };
        let Some(calibration) = binding(item) else {
            continue;
        };
        let query = EvidenceQuery {
            backend: CalibrationBackend::Candle,
            model_id: model_id.to_owned(),
            provider: evidence_provider.to_owned(),
            tier: tier.to_owned(),
            mode: mode.to_owned(),
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
        candidates.push(MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: runtime_provider.to_owned(),
                backend: MemoryBackend::Candle,
                tier: numeric_tier(tier).expect("validated numeric tier"),
                mode: request_mode(mode).0,
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
            },
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions::VERIFIED,
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint,
            sceneworks_revision: calibration.scene_works_revision,
            inference_revision: calibration.inference_revision,
            harness_version: record.harness_version.clone(),
            predicted_peak_bytes: predicted.overall,
            observed_peak_bytes: match &record.observed_memory {
                RequiredNullable::Value(observed) => Some(observed.overall.device_bytes),
                RequiredNullable::Null => None,
            },
            parity: if record.quality.maximum_error_threshold == Some(0.0)
                && record.quality.mean_error_threshold == Some(0.0)
            {
                MemoryParityContract::Exact
            } else {
                MemoryParityContract::Tolerance {
                    metric: "maximum normalized RGB8 error".to_owned(),
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
    if !matches!(
        engine_id,
        "z_image"
            | "z_image_turbo"
            | "z_image_control"
            | "z_image_turbo_control"
            | "qwen_image"
            | "qwen_image_edit"
    ) {
        return Ok(None);
    }
    let request_evidence_revision = if matches!(engine_id, "qwen_image" | "qwen_image_edit") {
        QWEN_IMAGE_REQUEST_EVIDENCE_REVISION
    } else {
        Z_IMAGE_REQUEST_EVIDENCE_REVISION
    };
    // Hires-fix is two independently shaped denoise passes. The generic generation harness still
    // reuses one context for both, so it cannot truthfully carry a request-scoped geometry yet.
    // Keep that surface on its established path until it mints one scope per pass.
    if has_phases {
        return Ok(None);
    }
    let (mode, mode_key) = request_mode(request_mode_value);
    let Some(tier) = numeric_tier(tier_key) else {
        return Ok(None);
    };
    let Some(budget) = budget else {
        return Ok(None);
    };
    let Some(resident_peak_gb) = predicted_peak_gb else {
        return Ok(None);
    };
    let Some(contract) = crate::inference_runtime::media()
        .memory_strategy_contract(engine_id, spec)
        .map_err(|error| {
            WorkerError::Engine(format!("{engine_id} memory contract failed: {error}"))
        })?
    else {
        return Ok(None);
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
            mode: mode.clone(),
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
    let mut verified = if artifact_is_certified {
        verified_candidates(
            manifest,
            model_id,
            engine_id,
            tier_key,
            mode_key,
            exact_overlay,
            geometry,
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
    selections.push(resident_selection);
    evidence.push(&resident);
    for item in &verified {
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
        .map(|(selection, evidence)| Candidate {
            selection: *selection,
            evidence,
        })
        .collect::<Vec<_>>();
    let selected = crate::memory_strategy::select_strategy(
        RequestScope {
            resolved_route: engine_id,
            backend: "candle",
            tier,
            mode: mode_key,
            overlay,
            geometry,
            expected_inference_revision: crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION,
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
            mode,
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

    #[test]
    fn strict_control_uses_the_advertised_base_provider_evidence_cell() {
        assert_eq!(evidence_provider("z_image_control"), "z_image");
        assert_eq!(evidence_provider("z_image_turbo_control"), "z_image_turbo");
        assert_eq!(evidence_provider("z_image"), "z_image");
    }

    #[test]
    fn style_variations_preserves_its_exact_evidence_key() {
        let (mode, key) = request_mode("style_variations");
        assert_eq!(mode, MemoryMode::ImageToImage);
        assert_eq!(key, "style_variations");
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
