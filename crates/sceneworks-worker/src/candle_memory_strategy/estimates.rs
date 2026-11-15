//! candle_memory_strategy estimates owner.
use super::*;

pub(crate) fn manifest_measured_pixels(manifest: &JsonObject<String, Value>) -> u64 {
    manifest
        .get("candle")
        .and_then(|value| value.get("vramMeasuredPixels"))
        .and_then(Value::as_u64)
        .filter(|pixels| *pixels > 0)
        .unwrap_or(MANIFEST_ROW_DEFAULT_MEASURED_PIXELS)
}

/// The geometry the derivation law prices for one request (sc-25690).
///
/// Normally the request's own. An entry that DECLARES its per-megapixel activation term
/// (`candle.imageActivationGbPerMegapixel` — Iris-3B) has every image of a native prompt batch and
/// every megapixel beyond the ONE image at 1024² its floors carry charged by
/// `lane_controls::declared_activation_overlay_bytes`, which reaches every candidate here as
/// `runtime_overlay_bytes`. The law scales an anchor's activation residue by the request's batch
/// and area too, so pricing such an entry at the request's own geometry would charge the same
/// activations twice: once declared, once derived. For a declaring entry the law therefore prices
/// one image at no more than the floors' area, and the declared term carries the rest.
///
/// The admitted `MemoryGeometry` the provider's request scope receives is NOT this value — it
/// stays the request's own, because the scope bounds the images a request may render by `batch`.
pub(super) fn law_geometry(
    manifest: &JsonObject<String, Value>,
    geometry: MemoryGeometry,
) -> MemoryGeometry {
    if crate::image_jobs::declared_activation_gb_per_megapixel(manifest).is_none() {
        return geometry;
    }
    let floor_pixels = crate::image_jobs::DECLARED_ACTIVATION_FLOOR_PIXELS;
    let pixels = u64::from(geometry.width) * u64::from(geometry.height);
    let (width, height) = if pixels > floor_pixels {
        let edge = u32::try_from(floor_pixels.isqrt())
            .unwrap_or(u32::MAX)
            .max(1);
        (edge, edge)
    } else {
        (geometry.width, geometry.height)
    };
    MemoryGeometry {
        width,
        height,
        batch: 1,
        ..geometry
    }
}

pub(super) fn scale_ideogram_hires_envelope(
    engine_id: &str,
    mode: &MemoryMode,
    manifest: &JsonObject<String, Value>,
    geometry: MemoryGeometry,
    bytes: u64,
) -> u64 {
    if !is_ideogram(engine_id) || *mode != MemoryMode::ImageToImage {
        return bytes;
    }
    let measured_pixels = manifest_measured_pixels(manifest);
    let request_pixels = u64::from(geometry.width).saturating_mul(u64::from(geometry.height));
    let numerator = request_pixels.max(measured_pixels);
    bytes
        .saturating_mul(numerator)
        .saturating_add(measured_pixels - 1)
        / measured_pixels
}

/// The smallest declared value for every numeric knob the engaged composition requires — the most
/// deeply bounding parameters the provider publishes, which keeps the true runtime transient as
/// far below the floor's unreduced peak as the provider allows (sc-18097, the candle mirror of the
/// MLX gate's helper from sc-18096). `None` when a required knob has no declared range: such a
/// selection cannot be validated, so no estimate candidate is synthesized for the rung.
pub(super) fn estimate_floor_parameters(
    contract: &gen_core::MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> Option<gen_core::MemoryStrategyParameters> {
    sceneworks_core::memory_contracts::smallest_parameters(
        contract,
        engaged,
        sceneworks_core::memory_contracts::TransformerScopeParameters::ProviderOrder,
    )
}

/// Synthesize an estimate-floor candidate for every implemented optimized rung (sc-18097, epic
/// 18093 R1b — the candle mirror of `mlx_fit_gate::synthesize_estimate_ladder`'s floor arm).
///
/// Peak source per rung, from the manifest rows the legacy candle gate already trusts
/// (`crate::vram_gate::predicted_peak_gb` conventions) — never a tuned coefficient and never a
/// promised unmeasured saving:
///
/// * `StagedResidency` — the `candle.sequentialPeakGb` row plus [`crate::vram_gate::HEADROOM_GB`]
///   (`vram_gate::predicted_sequential_peak_gb`, the same padded row the legacy sequential-offload
///   gate compares — a measured working set for legacy adopters). Chroma and the other
///   receipt-priced families instead carry a structural floor: the provider receipt's largest
///   materialized phase plus headroom, maxed against the padded row, required to remain genuinely
///   below resident. Absent the row, other adopters use the resident estimate: staging stays
///   selectable but promises no unmeasured saving.
/// * Rungs 2–4 bound transients/residency the manifest has NOT measured for this cell. Since
///   sc-22664 they are priced per rung through the law (below); before it they took the STAGED
///   floor unreduced. (Where a measured record for a rung exists it is a `verified_candidates`
///   candidate and supersedes the estimate in the selector.) The staged row prices the staged
///   WORKING SET, so it is a sound floor only for a composition that actually engages
///   `StagedResidency` (sc-18253). A provider may
///   implement a deep rung whose engaged composition excludes staging
///   (`gen_core::MemoryProviderContract::engaged_composition`) — such a request runs whole-model
///   resident, so its floor clamps to the RESIDENT estimate instead: the candle mirror of the MLX
///   floor's `engaged.contains(&StagedResidency)` max-vs-sum split
///   (`mlx_fit_gate::estimate_floor_weights_bytes`), which keeps the rung selectable without ever
///   under-predicting a resident working set behind the estimate margin.
///
/// The candle estimate margin is NOT applied here — the selector owns margin widening
/// (`crate::memory_strategy::select_strategy`), exactly as it owns the sc-18095 stale widening.
///
/// sc-22509 (epic 22505) inserted one rung ABOVE the manifest rows: where a measured memory ANCHOR
/// exists for this `(model, tier, candle lane)` cell, the rung floor is derived analytically from it
/// (`sceneworks_core::memory_anchor`) instead of read from `candle.sequentialPeakGb` /
/// `candle.vramGbByTier`. The manifest rows are a single scalar measured at one geometry
/// (`vramMeasuredPixels`); the derived estimate is geometry-aware, so a request at a geometry the
/// campaign never measured is priced from the anchor plus architecture facts rather than being
/// held to a 1024x1024 row.
///
/// sc-22664 (epic 22657 E4) prices EVERY rung through the law, each from its own regime:
///
/// * With an anchor for the cell, each implemented rung builds a [`RequestRegime`] from the
///   parameters [`estimate_floor_parameters`] selects for its engaged composition — the decode
///   tile, the attention chunk (the candle providers publish it as the score-element budget
///   `CONSTRAINED_ATTN_SCORES_BUDGET`, the unit the law takes), the transformer window — and
///   [`MemoryAnchor::derive_phase_peaks`] returns its three phase peaks; the rung's estimate is
///   the max over phases plus the runtime overlay. Basis:
///   [`CandidateBasis::EstimateAnchorDerived`]. No deeper rung reuses the staged floor.
/// * Without an anchor, the CONTRACT-ONLY path: every staged composition takes the law over the
///   manifest staged row, by treating the row as one staged measurement of every phase at the
///   geometry the manifest MEASURED it at — `candle.vramMeasuredPixels`, never the request's
///   ([`floor_pseudo_anchor`], sc-22667) — and decomposing it against the contract's component
///   bytes. The law then scales the row's residue UP to the request geometry before a deeper
///   rung's ratios apply, so a request above the measured geometry prices above the row on
///   every rung, and a rung's tile/chunk fractions are taken of the request-sized residue rather
///   than of a residue the row never measured at that size. At the measured geometry the staged
///   rung is the row itself (every ratio is 1). The row is phase-blind, so it is every phase's
///   peak; a deeper rung's ratios move the phases they bound (decode under the tile, denoise
///   under the chunk and the window) and its reported phase peaks say so, but NO rung bounds
///   conditioning, so at the measured geometry the admission peak — the max over phases — stays
///   at the row. A floor cannot promise a saving only a measured anchor can show, and the ladder
///   invents none. With the default architecture facts (this pin — see
///   `crate::video_admission::architecture_facts_from_contract`) every rung ratio is inert; the
///   geometry scaling needs no fact. A staging-free composition keeps the resident clamp of
///   sc-18253. Basis: [`CandidateBasis::EstimateFloor`] — the row, not a measured anchor, is the
///   basis.
///
/// WHERE THE RESERVE IS PAID (sc-22664, the rule `crate::memory_strategy::ReserveCharge` states
/// once): a manifest-row floor — the staged row, scaled or not — carries
/// [`crate::vram_gate::HEADROOM_GB`] INSIDE its peak exactly as before sc-22664, and the selector
/// compares it against the unreserved pool; an anchor-derived candidate carries NO pad, because
/// the anchor measured every phase as a device delta above the pre-load residency, and pays the
/// operational reserve ([`crate::vram_gate::ladder_reserve_gb`]) on the budget side instead. The
/// receipt-priced families' floors are STRUCTURAL weights-plus-headroom floors sealed from the
/// provider receipt, in which the headroom is the modelled activation term, and are pad-carrying
/// the same way.
///
/// [`RequestRegime`]: sceneworks_core::memory_anchor::RequestRegime
/// [`MemoryAnchor::derive_phase_peaks`]: sceneworks_core::memory_anchor::MemoryAnchor::derive_phase_peaks
/// [`CandidateBasis::EstimateAnchorDerived`]: crate::memory_strategy::CandidateBasis::EstimateAnchorDerived
/// [`CandidateBasis::EstimateFloor`]: crate::memory_strategy::CandidateBasis::EstimateFloor
#[allow(clippy::too_many_arguments)]
pub(super) fn synthesize_estimate_floors(
    engine_id: &str,
    model_id: &str,
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
    anchors: CandleLadderAnchors<'_>,
) -> Vec<EstimateCandidate> {
    use sceneworks_core::memory_anchor::{ImageDeriveRequest, RequestRegime};

    let calibration = contract.calibration.as_ref();
    let receipt_priced = is_receipt_priced(engine_id);
    let staged_floor_bytes = if receipt_priced {
        let structural = receipt_base_phase_floor_bytes(contract)
            .saturating_add((crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB).ceil() as u64);
        let declared = crate::vram_gate::predicted_sequential_peak_gb(manifest, tier_key)
            .map(|gb| (gb * BYTES_PER_GIB).ceil().clamp(0.0, u64::MAX as f64) as u64)
            .unwrap_or(0);
        let declared =
            scale_ideogram_hires_envelope(engine_id, &mode.mode, manifest, geometry, declared);
        contract
            .predicted_peak_from_base(declared.max(structural))
            .predicted_peak_bytes()
    } else {
        // The staged row plus the structural pad, exactly as before sc-22664: the pad is INSIDE a
        // manifest-row floor, so the selector compares such a floor against the unreserved pool
        // (`crate::memory_strategy::ReserveCharge`). Absent the row, the resident estimate.
        crate::vram_gate::predicted_sequential_peak_gb(manifest, tier_key)
            .map(|gb| {
                ((gb * BYTES_PER_GIB).ceil().clamp(0.0, u64::MAX as f64) as u64)
                    .saturating_add(runtime_overlay_bytes)
            })
            .unwrap_or(resident_peak_bytes)
    };
    let headroom_bytes = (crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB).ceil() as u64;
    // The RAW staged row — what the manifest measured, without the pad or the request's overlay —
    // is what the law decomposes for the contract-only deeper rungs; both charges are folded back
    // over the derivation.
    let raw_staged_row_bytes = crate::vram_gate::measured_sequential_peak_gb(manifest, tier_key)
        .map(|gb| (gb * BYTES_PER_GIB).ceil().clamp(0.0, u64::MAX as f64) as u64);
    // Chroma claims a staged saving only when the receipt-derived largest phase plus exact
    // auxiliaries is genuinely below its receipt-derived co-resident floor. Equal or crossed rows
    // are not an optimized strategy and cannot become selectable through a tier label.
    if receipt_priced && staged_floor_bytes >= resident_peak_bytes {
        return Vec::new();
    }
    // NOTE (sc-22509, reopened by sc-22666): this lookup now MATCHES packaged anchors. Every
    // retained corpus is compiled in since epic 22657 E5, so `z_image_turbo:candle` (q4/q8/bf16,
    // sc-15859) and the qwen candle rows answer here through the shared-image ladder. The
    // per-model allow-list that used to keep them out is gone; identity and loader-closure
    // currency are the only guards left, and a cell with no anchor row falls through to
    // `floor_anchor` below -- the contract-only per-rung ladder (sc-22664), never a bare
    // manifest scalar repeated across every rung.
    // The geometry the LAW prices — the request's own, except where the manifest's declared
    // activation term already charges the batch and the area (see `law_geometry`).
    let law = law_geometry(manifest, geometry);
    let anchor = candle_image_anchor(
        anchors, engine_id, model_id, contract, tier_key, mode, overlay, law,
    );
    let components = crate::video_admission::anchor_component_bytes(contract.asset_facts);
    // The contract-only basis for a staged composition with no anchor: the staged row treated as
    // one staged measurement of every phase at the request geometry (see the doc comment). Only
    // for the shared ladder — a receipt-priced floor is a structural sum, not a measured working
    // set, and stays unscaled.
    let floor_anchor = if anchor.is_none() && !receipt_priced {
        raw_staged_row_bytes.map(|row| floor_pseudo_anchor(engine_id, manifest, row))
    } else {
        None
    };
    let mut synthesized = Vec::new();
    // Scratch-bounding rungs remain independently usable with resident weights. SDXL additionally
    // offers their explicit staged composition, so the selector can reach the measured fully
    // optimized endpoint without changing the provider's cost-order defaults or bespoke support.
    for (strategy, explicit_staging) in MemoryStrategy::ALL.into_iter().flat_map(|strategy| {
        std::iter::once((strategy, false)).chain(
            (engine_id == "sdxl"
                && model_id == "sdxl"
                && mode.mode == MemoryMode::TextToImage
                && geometry.reference_count == 0
                && geometry.batch == 1
                && matches!(overlay, None | Some("lora"))
                && matches!(
                    strategy,
                    MemoryStrategy::BoundedDecode | MemoryStrategy::BoundedAttention
                ))
            .then_some((strategy, true)),
        )
    }) {
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
        let Some(mut parameters) = estimate_floor_parameters(contract, &engaged) else {
            continue;
        };
        parameters.stage_residency = explicit_staging.then_some(true);
        let selection = MemorySelection {
            strategy,
            parameters,
            tier,
        };
        if contract.validate_selection(&selection).is_err() {
            continue;
        }
        let engaged = contract.engaged_composition_for_selection(&selection);
        let staged = engaged.contains(&MemoryStrategy::StagedResidency);
        // The rung's own regime: its engaged composition and the parameters it was selected with.
        // Every rung is priced from THIS, never from the staged rung's number (sc-22664).
        let regime = request_regime(&engaged, &parameters);
        let request = |regime: RequestRegime| ImageDeriveRequest {
            width: law.width,
            height: law.height,
            batch: law.batch.max(1),
            conditioning_tokens: None,
            regime,
        };
        // The retained candle phase peaks are device-usage DELTAS above the process's pre-load
        // residency; that residency is the reserve the selector budget charges once
        // (`vram_gate::ladder_reserve_gb`), so nothing is added here beyond the request's own
        // runtime overlay.
        let derived = anchor.zip(regime).and_then(|(anchor, regime)| {
            anchor
                .derive_phase_peaks(&request(regime), components, anchors.facts)
                .map(|phases| (phases, anchor.id.as_str()))
        });
        let (predicted_peak_bytes, basis, phase_peaks) = match derived {
            Some((phases, anchor_id)) => {
                let bytes = phases.peak_bytes().saturating_add(runtime_overlay_bytes);
                tracing::info!(
                    route = engine_id,
                    backend = "candle",
                    ?strategy,
                    anchor = anchor_id,
                    conditioning_peak_bytes = phases.conditioning,
                    denoise_peak_bytes = phases.denoise,
                    decode_peak_bytes = phases.decode,
                    raw_peak_bytes = bytes,
                    "synthesized anchor-derived estimate candidate"
                );
                (
                    bytes,
                    crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                        lane: crate::memory_strategy::AnchorDerivationLane::Image,
                    },
                    Some(phases),
                )
            }
            None => {
                // sc-18253: the staged row is only a sound floor for a composition that engages
                // staging; a deep rung excluding `StagedResidency` runs whole-model resident and
                // clamps to the resident estimate (see the doc comment above). A staged
                // composition takes the law's ratios over the row (`floor_pseudo_anchor`); with
                // no ratio to apply — the staged rung itself, or a regime the row cannot price —
                // the row stands unscaled.
                let from_floor = if staged {
                    floor_anchor
                        .as_ref()
                        .zip(regime)
                        .and_then(|(floor_anchor, regime)| {
                            floor_anchor.derive_phase_peaks(
                                &request(regime),
                                components,
                                anchors.facts,
                            )
                        })
                } else {
                    None
                };
                // A law-scaled row carries the same structural pad and overlay the unscaled
                // staged floor does: it is still a manifest-row floor.
                let (bytes, phase_peaks) = match from_floor {
                    Some(phases) => (
                        phases
                            .peak_bytes()
                            .saturating_add(headroom_bytes)
                            .saturating_add(runtime_overlay_bytes),
                        Some(phases),
                    ),
                    None if staged => (staged_floor_bytes, None),
                    None => (resident_peak_bytes, None),
                };
                tracing::info!(
                    route = engine_id,
                    backend = "candle",
                    ?strategy,
                    raw_peak_bytes = bytes,
                    law_scaled = phase_peaks.is_some(),
                    "synthesized manifest-row floor estimate candidate"
                );
                (
                    bytes,
                    crate::memory_strategy::CandidateBasis::EstimateFloor,
                    phase_peaks,
                )
            }
        };
        synthesized.push(EstimateCandidate {
            selection,
            phase_peaks,
            basis,
            evidence: sceneworks_core::memory_contracts::unverified_estimate(
                MemoryEvidenceKey {
                    model_family: engine_id.to_owned(),
                    resolved_route: engine_id.to_owned(),
                    backend: MemoryBackend::Candle,
                    tier,
                    mode: mode.mode.clone(),
                    reference_shape: if geometry.reference_count == 0 {
                        gen_core::MemoryReferenceShape::None
                    } else {
                        gen_core::MemoryReferenceShape::Image
                    },
                    load_shape: contract.load_shape,
                    overlay: overlay.map(str::to_owned),
                    geometry,
                    frames_per_second: None,
                    strategy,
                    engaged_composition: engaged,
                    parameters,
                },
                calibration.map_or(gen_core::MEMORY_CALIBRATION_ABI, |item| item.abi),
                calibration.map_or_else(String::new, |item| item.fingerprint.clone()),
                request_evidence_revision.to_owned(),
                predicted_peak_bytes,
            ),
        });
    }
    synthesized
}

/// Price a validated endpoint at its literal coordinate, or derive the same executable
/// composition at another geometry. The capture measures memory, not numerical parity: it
/// supplies an estimate-scoped candidate with the normal uncertainty and a separate reserve.
pub(super) fn sdxl_endpoint_candidate(
    contract: &gen_core::MemoryProviderContract,
    resident: &MemoryEvidence,
    anchor: &sceneworks_core::memory_anchor::MemoryAnchor,
    selection: MemorySelection,
    load_peak_bytes: u64,
) -> Option<EstimateCandidate> {
    use sceneworks_core::memory_anchor::{
        AnchorBackend, AnchorDerivedPhases, AnchorLoadShape, ImageDeriveRequest,
    };

    let geometry = resident.key.geometry;
    let load_shape = match anchor.load_shape {
        AnchorLoadShape::EagerMaterialization => gen_core::LoadShape::EagerMaterialization,
        AnchorLoadShape::DeferredMaterialization => gen_core::LoadShape::DeferredMaterialization,
    };
    if anchor.model_id != "sdxl"
        || anchor.provider != contract.provider_id
        || anchor.route != resident.key.resolved_route
        || anchor.backend != AnchorBackend::Candle
        || anchor.tier != "q4"
        || anchor.mode != resident.key.mode.as_key()
        || anchor.overlay.is_some()
        || resident.key.overlay.is_some()
        || anchor.reference_count != 0
        || geometry.reference_count != 0
        || geometry.batch != 1
        || geometry.frames != 1
        || load_shape != contract.load_shape
        || !contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == anchor.source.calibration_fingerprint)
        || contract.validate_selection(&selection).is_err()
    {
        return None;
    }
    let engaged = contract.engaged_composition_for_selection(&selection);
    let regime = request_regime(&engaged, &selection.parameters)?;
    // A bounded endpoint can only price its own exact bounded composition. The pair loader binds
    // the actual numeric knobs; this check also prevents borrowing it for an unbounded request.
    if anchor.measured_regime.staged != regime.staged
        || anchor.measured_regime.decode_tiled != regime.decode_tile.is_some()
        || anchor.measured_regime.attention_chunked != regime.attention_chunk_scores.is_some()
        || anchor.measured_regime.transformer_windowed != regime.transformer_window.is_some()
    {
        return None;
    }
    let exact_geometry =
        geometry.width == anchor.geometry.width && geometry.height == anchor.geometry.height;
    let phases = if exact_geometry {
        AnchorDerivedPhases {
            conditioning: anchor.phase_active_peak_bytes.conditioning,
            denoise: anchor.phase_active_peak_bytes.denoise,
            decode: anchor.phase_active_peak_bytes.decode,
        }
    } else {
        anchor.derive_phase_peaks(
            &ImageDeriveRequest {
                width: geometry.width,
                height: geometry.height,
                batch: 1,
                conditioning_tokens: None,
                regime,
            },
            crate::video_admission::anchor_component_bytes(contract.asset_facts),
            crate::video_admission::architecture_facts_from_contract(contract),
        )?
    };
    let peak = if exact_geometry {
        anchor.overall_allocator_envelope_bytes
    } else {
        phases.peak_bytes().max(load_peak_bytes)
    };
    let mut evidence = resident.clone();
    evidence.key.strategy = selection.strategy;
    evidence.key.parameters = selection.parameters;
    evidence.key.engaged_composition = engaged;
    evidence.predicted_peak_bytes = peak;
    evidence.observed_peak_bytes = exact_geometry.then_some(peak);
    Some(EstimateCandidate {
        selection,
        evidence,
        basis: if exact_geometry {
            crate::memory_strategy::CandidateBasis::EstimateMeasuredEndpoint
        } else {
            crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                lane: crate::memory_strategy::AnchorDerivationLane::Image,
            }
        },
        phase_peaks: Some(phases),
    })
}

impl CandleLadderAnchors<'static> {
    /// The production source: the packaged store -- catalog-wide and unscoped since sc-22666 --
    /// and the architecture facts the contract states.
    pub(crate) fn packaged(contract: &gen_core::MemoryProviderContract) -> Self {
        Self {
            store: sceneworks_core::memory_anchor::packaged_memory_anchors(),
            facts: crate::video_admission::architecture_facts_from_contract(contract),
        }
    }
}

/// The [`RequestRegime`] one ladder rung executes in: its engaged composition plus the parameters
/// [`estimate_floor_parameters`] selected for it. `None` when an engaged rung's parameter is
/// absent (the selection would already have failed `validate_selection`).
///
/// The attention chunk is translated as SCORE ELEMENTS: every candle provider publishes
/// `attention_chunk_sizes` as `CONSTRAINED_ATTN_SCORES_BUDGET` (or a per-model score budget of the
/// same unit), which is the quantity `RequestRegime::attention_chunk_scores` prices.
///
/// [`RequestRegime`]: sceneworks_core::memory_anchor::RequestRegime
pub(crate) fn request_regime(
    engaged: &[MemoryStrategy],
    parameters: &gen_core::MemoryStrategyParameters,
) -> Option<sceneworks_core::memory_anchor::RequestRegime> {
    // Candle has historically refused incomplete engaged parameters; retain that boundary.
    for (rung, complete) in [
        (
            MemoryStrategy::BoundedDecode,
            parameters.decode_tile_edge.is_some() && parameters.decode_overlap.is_some(),
        ),
        (
            MemoryStrategy::BoundedAttention,
            parameters.attention_chunk_size.is_some(),
        ),
        (
            MemoryStrategy::BoundedTransformerResidency,
            parameters.transformer_window_size.is_some(),
        ),
    ] {
        if engaged.contains(&rung) && !complete {
            return None;
        }
    }
    Some(sceneworks_core::memory_contracts::request_regime(
        engaged, parameters,
    ))
}

/// The manifest staged row as a pseudo-anchor (sc-22664, the contract-only path): one STAGED
/// measurement of every phase at the geometry the manifest MEASURED the row at, which is exactly
/// the claim `candle.sequentialPeakGb` makes — the largest single working set of the staged path
/// at `candle.vramMeasuredPixels` — stated phase-blind. Running it through
/// `MemoryAnchor::derive_phase_peaks` at that geometry returns the row itself for the staged rung
/// (every ratio is 1) and the law's ratios over the row for a deeper rung; at a larger request
/// geometry the law scales the row's residue up first, then applies the rung's ratios to the
/// request-sized residue. A row below the component set of some phase is outside the law's
/// domain and the derivation refuses it, in which case the caller keeps the row unscaled.
///
/// WHY THE MEASURED GEOMETRY, NOT THE REQUEST'S (sc-22667, epic 22657 feature-end round): the
/// row is geometry-blind, so stamping it as a measurement AT the request geometry made a 2048²
/// bounded rung take its tile/chunk fractions of a residue that was measured at 1024² — scaling
/// the row's phases DOWN, with no measurement behind the saving. Anchored at the measured
/// geometry the residue is scaled UP to 2048² before any fraction is taken, which is the erring-
/// large reading the epic requires. The measured pixel count is read as the largest square that
/// fits it (`isqrt`), so the anchor side of every geometry ratio is at or below the true measured
/// count and the scale-up is never understated.
///
/// It is a floor, not evidence: its identity fields are the request's, it cites no source, and the
/// candidate it prices carries `CandidateBasis::EstimateFloor`.
pub(crate) fn floor_pseudo_anchor(
    engine_id: &str,
    manifest: &JsonObject<String, Value>,
    staged_floor_bytes: u64,
) -> sceneworks_core::memory_anchor::MemoryAnchor {
    use sceneworks_core::memory_anchor::{
        AnchorBackend, AnchorGeometry, AnchorLoadShape, AnchorMeasuredRegime, AnchorPhaseBytes,
        AnchorSource, MemoryAnchor,
    };
    let measured_edge = u32::try_from(manifest_measured_pixels(manifest).isqrt())
        .unwrap_or(u32::MAX)
        .max(1);
    MemoryAnchor {
        id: format!("floor:{engine_id}:candle:sequentialPeakGb"),
        model_id: engine_id.to_owned(),
        model_family: String::new(),
        route: engine_id.to_owned(),
        provider: engine_id.to_owned(),
        backend: AnchorBackend::Candle,
        tier: String::new(),
        transformer_variant: None,
        decoder: None,
        mode: String::new(),
        overlay: None,
        reference_count: 0,
        load_shape: AnchorLoadShape::EagerMaterialization,
        measured_regime: AnchorMeasuredRegime {
            decode_tiled: false,
            transformer_windowed: false,
            staged: true,
            attention_chunked: false,
        },
        source: AnchorSource {
            path: String::new(),
            sha256: String::new(),
            record_id: String::new(),
            calibration_fingerprint: String::new(),
            loader_closure_digest: String::new(),
            currency_attestation: None,
        },
        geometry: AnchorGeometry {
            width: measured_edge,
            height: measured_edge,
            frames: 1,
            fps: None,
        },
        phase_active_peak_bytes: AnchorPhaseBytes {
            conditioning: staged_floor_bytes,
            denoise: staged_floor_bytes,
            decode: staged_floor_bytes,
        },
        phase_allocator_envelope_bytes: None,
        overall_allocator_envelope_bytes: staged_floor_bytes,
        underived_reason: None,
        component_bytes: None,
        // sc-22734: an ordinary staged candle cell, not one whose engine declares staged
        // residency structurally impossible, so it keeps the staged-only derivation law.
        staged_residency_structurally_not_applicable: false,
    }
}

/// The measured memory anchor for this candle image request, or `None` to keep the manifest-row
/// floor (sc-22509, epic 22505).
///
/// Identity is strict and fail-closed, the candle mirror of `video_admission::anchor_currency_
/// matches` and `vram_gate::krea_store_anchor`: the anchor was measured on ONE catalog model,
/// route, provider, tier, mode and materialization shape, overlay-free and reference-free (checked
/// on BOTH the request and the anchor row), and its loader closure must still be current
/// (sc-22511). Anything else keeps the established floor. The
/// measured GEOMETRY is deliberately NOT part of the guard — pricing a never-measured geometry from
/// one anchor is the entire point of the derivation — and neither is `model_family`, which has no
/// source in the record (see [`sceneworks_core::memory_anchor::MemoryAnchor::model_family`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn candle_image_anchor<'a>(
    anchors: CandleLadderAnchors<'a>,
    engine_id: &str,
    model_id: &str,
    contract: &gen_core::MemoryProviderContract,
    tier_key: &str,
    mode: &RequestModeBinding,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
) -> Option<&'a sceneworks_core::memory_anchor::MemoryAnchor> {
    // The anchors were measured overlay-free, reference-free and single-batch on a still image.
    if overlay.is_some() || geometry.reference_count != 0 || geometry.batch != 1 {
        return None;
    }
    // NO MODEL SCOPE (sc-22666, epic 22657 E5). A `CANDLE_ANCHOR_COEFFICIENT_MODELS` allow-list
    // (`["krea_2_turbo"]`) used to stand here, because the sc-22509 candle law was three per-pixel
    // slopes fitted on Krea Turbo and pricing another model's row through them would have been
    // borrowing empirics. The core law fits nothing since sc-22663 and this lane prices every rung
    // through it since sc-22664 -- it decomposes THIS anchor's own measured peaks against THIS
    // contract's component bytes and rescales the residues by architecture facts -- so a store
    // anchor for any `(model, tier)` cell prices its own cell and nobody else's. Nor is it a
    // packaging question any more: every retained corpus is compiled in, so a packaged row is
    // priced the day it lands, which is the point of packaging them.
    //
    // The identity conjuncts below (route, provider, mode, overlay, references, materialization
    // shape, loader-closure currency) are the whole guard now: per-row facts, not a model census.
    let anchor = anchors.store?.image_anchor_for(
        model_id,
        sceneworks_core::memory_anchor::AnchorBackend::Candle,
        tier_key,
    )?;
    // An anchor the extractor marked underived validates its own measured point but is not
    // published as a derivation basis (the memory matrix prints the cell `Anchored/underived`);
    // the worker agrees with the matrix and does not price from it. Lifting that per model is
    // sc-22666's packaging call, not this lane's.
    if anchor.underived_reason.is_some() {
        return None;
    }
    // `model_family` is deliberately NOT a conjunct (sc-22509 review): the source calibration
    // record carries no family field, so the store's copy is catalog-derived and unvalidated, and
    // keying on it would make an extractor's catalog read load-bearing for admission. `model_id` is
    // bound to the record and already selects the cell.
    //
    // The ANCHOR side of the overlay/reference guard, mirroring `vram_gate::krea_store_anchor` and
    // `video_admission::anchor_currency_matches`: the request being overlay-free is only half the
    // question — an overlay-measured or reference-bearing anchor row must not be borrowed for an
    // overlay-free request either.
    if anchor.route != engine_id
        || anchor.provider != contract.provider_id
        || anchor.mode != mode.mode.as_key()
        || anchor.overlay.is_some()
        || anchor.reference_count != 0
    {
        return None;
    }
    // Materialization shape decides whether the text encoder is still resident, so an anchor
    // measured under the other shape does not price this request.
    let anchor_load_shape = match anchor.load_shape {
        sceneworks_core::memory_anchor::AnchorLoadShape::EagerMaterialization => {
            gen_core::LoadShape::EagerMaterialization
        }
        sceneworks_core::memory_anchor::AnchorLoadShape::DeferredMaterialization => {
            gen_core::LoadShape::DeferredMaterialization
        }
    };
    if anchor_load_shape != contract.load_shape {
        return None;
    }
    // No currency conjunct (sc-22738): the model's loader-closure digest (sc-22511) is what the
    // probe tooling re-captures on, and the calibration ABI and fingerprint are provenance bound
    // to the source record by `validate_anchor`. None of them demotes the anchor here — it prices
    // this request whether or not the loader has moved since it was measured.
    Some(anchor)
}
