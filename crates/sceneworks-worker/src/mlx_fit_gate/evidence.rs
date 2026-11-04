//! mlx_fit_gate evidence owner.
use super::*;

pub(super) fn resident_evidence(
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

pub(super) fn mismatch_fallback_reason(reason: EvidenceMismatchReason) -> LegacyAdmissionReason {
    if reason == EvidenceMismatchReason::CalibrationFingerprint {
        LegacyAdmissionReason::FingerprintMismatch
    } else {
        LegacyAdmissionReason::IdentityMismatch
    }
}

pub(super) fn stronger_fallback_reason(
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

pub(super) fn evidence_strategy(rung: StrategyRung) -> MemoryStrategy {
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
pub(super) fn parse_evidence_parameters(
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
pub(super) fn packaged_admission_route(
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

pub(super) fn evidence_admission_route(
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
pub(super) fn collect_estimate_bases(
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
pub(super) fn mlx_image_anchor_store(
) -> Option<&'static sceneworks_core::memory_anchor::MemoryAnchorStore> {
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
pub(super) fn mlx_image_anchor_derived_peak(
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
pub(super) fn mlx_image_anchor_match(
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
pub(super) fn request_regime(
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
pub(super) fn mlx_image_anchor_activation_residue(
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

pub(super) const fn strategy_rung(strategy: MemoryStrategy) -> StrategyRung {
    match strategy {
        MemoryStrategy::Resident => StrategyRung::Resident,
        MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
        MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
        MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency => StrategyRung::BoundedTransformerResidency,
    }
}

/// The phase carrying the peak of a `(conditioning, denoise, decode)` triple. Ties resolve to the
/// LATER phase deterministically; the comparison below only ever contrasts two triples scaled by
/// the same per-phase rule, so tie handling cannot manufacture a flip on its own.
pub(super) fn binding_phase(conditioning: u64, denoise: u64, decode: u64) -> EstimatePhase {
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
pub(super) fn intra_transformer_evicted_bytes(contract: &MemoryProviderContract) -> u64 {
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
pub(super) fn floor_weights_bytes(
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
pub(super) fn estimate_floor_parameters(
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
pub(super) fn mlx_selected_phase_peak(
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
pub(super) fn synthesize_estimate_ladder(
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
pub(super) fn synthesize_estimate_ladder_with_conditioning(
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
