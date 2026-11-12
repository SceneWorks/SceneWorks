use super::*;

impl ControlLadderContext<'_> {
    pub(super) fn candidate_rows(
        &self,
        peak: f64,
        sequential_peak_gb: Option<f64>,
        decode_tile_save_gb: Option<f64>,
        chunk_attn_save_gb: Option<f64>,
    ) -> ControlCandidateRows {
        let Self {
            contract,
            numeric_tier,
            request_geometry,
            measured_geometry,
            overlay,
            evidence_is_current,
            runtime_verified,
            adapter_gb,
            measured_closure_digest: _,
        } = *self;
        // Catalog helpers retain their legacy admission numbers (raw observation + HEADROOM_GB). Evidence
        // stores the raw rendered-device peak; the shared Budget removes headroom exactly once.
        let mut measured = vec![(MemoryStrategy::Resident, (peak - HEADROOM_GB).max(0.0))];
        if let Some(staged) = sequential_peak_gb {
            let staged = (staged - HEADROOM_GB).max(0.0);
            measured.push((MemoryStrategy::StagedResidency, staged));
            if let Some(tile_save) = decode_tile_save_gb {
                measured.push((MemoryStrategy::BoundedDecode, staged - tile_save));
            }
            if let Some(chunk_save) = chunk_attn_save_gb {
                let decode_save = if contract.engages(
                    MemoryStrategy::BoundedAttention,
                    MemoryStrategy::BoundedDecode,
                ) {
                    // Q4's provider contract composes attention chunking over tiled decode. Without the
                    // decode row, the full engaged composition is not measured and must not become a
                    // falsely verified attention candidate. Dense/Q8 contracts exclude that default edge,
                    // so their independently measured chunking candidate remains valid without tiling.
                    decode_tile_save_gb
                } else {
                    Some(0.0)
                };
                if let Some(decode_save) = decode_save {
                    measured.push((
                        MemoryStrategy::BoundedAttention,
                        staged - decode_save - chunk_save,
                    ));
                }
            }
        }
        let selections = measured
            .iter()
            .map(|(strategy, _)| MemorySelection {
                strategy: *strategy,
                parameters: control_parameters(contract, *strategy),
                tier: numeric_tier,
            })
            .collect::<Vec<_>>();
        let evidence = selections
            .iter()
            .zip(&measured)
            .map(|(selection, (_, measured_peak_gb))| {
                self.measured_evidence(*selection, *measured_peak_gb)
            })
            .collect::<Vec<_>>();
        // ── sc-18097 (epic 18093 R1b): estimate-floor candidates for every implemented rung. ──
        //
        // The control lane's mirror of the estimate ladder: manifest-row floors, never a promised
        // unmeasured saving. The resident floor is the measured `peakGbByTier` row (with adapters
        // already folded into `peak`); the staged floor is the measured `sequentialPeakGbByTier` row
        // where present, else the resident floor; every deeper rung whose engaged composition
        // includes `StagedResidency` takes the staged floor UNREDUCED — selectable without promising
        // an unmeasured saving. The staged row prices the staged WORKING SET, so a deep rung whose
        // engaged composition EXCLUDES staging (sc-18253 — the `engagedRungs` mechanism permits it)
        // runs whole-model resident and clamps to the resident floor instead, mirroring the MLX
        // floor's `engaged.contains(&StagedResidency)` max-vs-sum split
        // (`mlx_fit_gate::estimate_floor_weights_bytes`). Where a rung's savings ARE measured, its
        // measured candidate exists in `measured` above and supersedes the floor in the selector, so
        // the priced 1024² ladder is byte-for-byte unchanged.
        //
        // Floors are synthesized only when every conjunct holds:
        //  * `evidence_is_current` — dimension-level staleness (`measured: false` / `supersededBy`)
        //    means the rows were ALREADY known non-current when recorded; they still exclude and may
        //    not seed floors either (the `BestEffort` never-reject contract stays for them).
        //  * `runtime_verified` — the rows price the shipped base and control artifacts. Known adapter
        //    bytes are added to every estimate; they do not turn it into a measured result.
        //  * the loaded contract's calibration identity matches the sc-16013 rows (the sc-18096
        //    drifted-provider gate).
        //  * the request has the same single-image/single-pose shape. The rows are whole-render
        //    peaks read VERBATIM: at or below the measured area every phase is at most its measured
        //    value, so the constant extrapolation is an upper bound and no binding-phase flip can
        //    exceed it — which is why these are `EstimateFloor` candidates outside
        //    `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`'s fitted-curve scope (no per-phase
        //    decomposition exists to re-check). Above the measured area the image derivation law
        //    scales the row before admission. A failure to fit that conservative estimate preserves
        //    the recoverable best-effort fallback instead of hard-rejecting a bounded render.
        //
        // The candle estimate margin is applied by the shared selector, not here.
        let request_pixels = u64::from(request_geometry.width) * u64::from(request_geometry.height);
        let measured_pixels =
            u64::from(measured_geometry.width) * u64::from(measured_geometry.height);
        let contract_identity_matches = contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == KREA_CONTROL_CALIBRATION);
        let resident_floor_gb = (peak - HEADROOM_GB).max(0.0);
        let staged_floor_gb = sequential_peak_gb
            .map(|gb| (gb - HEADROOM_GB).max(0.0))
            .unwrap_or(resident_floor_gb);
        let mut estimates: Vec<(MemorySelection, MemoryEvidence)> = Vec::new();
        if evidence_is_current
            && runtime_verified
            && contract_identity_matches
            && request_geometry.batch == 1
            && request_geometry.frames == 1
            && request_geometry.reference_count == KREA_CONTROL_REFERENCE_COUNT
        {
            for strategy in MemoryStrategy::ALL {
                if !matches!(
                    contract.capability(strategy).map(|cap| &cap.support),
                    Some(gen_core::MemoryStrategySupport::Implemented)
                ) {
                    continue;
                }
                let selection = MemorySelection {
                    strategy,
                    parameters: control_parameters(contract, strategy),
                    tier: numeric_tier,
                };
                if contract.validate_selection(&selection).is_err() {
                    continue;
                }
                let engaged = contract.engaged_composition(strategy);
                // sc-18253: the staged row is only a sound floor for a composition that actually
                // engages staging; a rung excluding `StagedResidency` runs whole-model resident and
                // clamps to the resident floor (see the header comment above).
                let staged = strategy != MemoryStrategy::Resident
                    && engaged.contains(&MemoryStrategy::StagedResidency);
                let floor_gb = if staged {
                    staged_floor_gb
                } else {
                    resident_floor_gb
                };
                // Above the measured area, price the request through the existing image derivation
                // law. Treat the scalar as a phase-blind measurement at 1024², never at the requested
                // size. Do not promise unmeasured tile/chunk savings. Adapter bytes are a fixed
                // resident addition, not an activation residue that shrinks with geometry.
                let floor_gb = if request_pixels > measured_pixels {
                    let mut anchor = crate::candle_memory_strategy::floor_pseudo_anchor(
                        KREA_CONTROL_ROUTE,
                        &JsonObject::new(),
                        bytes((floor_gb - adapter_gb).max(0.0)),
                    );
                    anchor.measured_regime.staged = staged;
                    let request = sceneworks_core::memory_anchor::ImageDeriveRequest::new(
                        request_geometry.width,
                        request_geometry.height,
                        if staged {
                            sceneworks_core::memory_anchor::RequestRegime::staged()
                        } else {
                            sceneworks_core::memory_anchor::RequestRegime::resident()
                        },
                    );
                    let Some(phases) = anchor.derive_phase_peaks(
                        &request,
                        crate::video_admission::anchor_component_bytes(contract.asset_facts),
                        crate::video_admission::architecture_facts_from_contract(contract),
                    ) else {
                        continue;
                    };
                    phases.peak_bytes() as f64 / crate::fit_gate::BYTES_PER_GIB + adapter_gb
                } else {
                    floor_gb
                };
                let predicted_peak_bytes = bytes(floor_gb);
                tracing::info!(
                    route = KREA_CONTROL_ROUTE,
                    backend = "candle",
                    ?strategy,
                    raw_peak_bytes = predicted_peak_bytes,
                    "synthesized manifest-row floor estimate candidate"
                );
                estimates.push((
                    selection,
                    sceneworks_core::memory_contracts::unverified_estimate(
                        MemoryEvidenceKey {
                            model_family: KREA_CONTROL_ROUTE.to_owned(),
                            resolved_route: KREA_CONTROL_ROUTE.to_owned(),
                            backend: gen_core::MemoryBackend::Candle,
                            tier: numeric_tier,
                            load_shape: contract.load_shape,
                            mode: crate::memory_strategy::memory_mode_from_mode_key(
                                KREA_CONTROL_MODE,
                            ),
                            reference_shape: gen_core::MemoryReferenceShape::Image,
                            overlay: Some(overlay.clone()),
                            geometry: request_geometry,
                            frames_per_second: None,
                            strategy,
                            engaged_composition: engaged,
                            parameters: selection.parameters,
                        },
                        contract
                            .calibration
                            .as_ref()
                            .expect("control calibration")
                            .abi,
                        KREA_CONTROL_CALIBRATION.to_owned(),
                        KREA_CONTROL_ESTIMATE_REVISION.to_owned(),
                        predicted_peak_bytes,
                    ),
                ));
            }
        }

        ControlCandidateRows {
            measured,
            selections,
            evidence,
            estimates,
        }
    }
}

impl ControlCandidateRows {
    pub(super) fn candidates(&self) -> Vec<Candidate<'_>> {
        let mut candidates = self
            .selections
            .iter()
            .zip(&self.evidence)
            .map(|(selection, evidence)| Candidate {
                selection: *selection,
                evidence,
                basis: crate::memory_strategy::CandidateBasis::Measured,
                unmodeled_activation_bytes: None,
            })
            .collect::<Vec<_>>();
        candidates.extend(
            self.estimates
                .iter()
                .map(|(selection, evidence)| Candidate {
                    selection: *selection,
                    evidence,
                    basis: crate::memory_strategy::CandidateBasis::EstimateFloor,
                    // sc-22508: this control-lane floor is a manifest-row declaration, not a weights+headroom
                    // split, so it declares no activation term and takes the candle whole-peak residual.
                    unmodeled_activation_bytes: None,
                }),
        );
        candidates
    }
}
