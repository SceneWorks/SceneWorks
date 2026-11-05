use super::*;

impl ControlLadderContext<'_> {
    pub(super) fn measured_evidence(
        &self,
        selection: MemorySelection,
        predicted_peak_gb: f64,
    ) -> MemoryEvidence {
        let Self {
            contract,
            numeric_tier,
            request_geometry,
            measured_geometry,
            overlay,
            evidence_is_current,
            runtime_verified,
            adapter_gb,
            measured_closure_digest,
        } = *self;
        let calibration_matches = contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == KREA_CONTROL_CALIBRATION);
        let historical_verified = evidence_is_current && adapter_gb == 0.0;
        let dimensions = MemoryEvidenceDimensions {
            static_implementation: if contract.conformance_errors().is_empty()
                && contract.validate_selection(&selection).is_ok()
            {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::Invalid
            },
            declared_calibration: if calibration_matches {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::FingerprintMismatch
            },
            historical_verification: if historical_verified {
                MemoryEvidenceVerdict::Satisfied
            } else if evidence_is_current {
                MemoryEvidenceVerdict::OutOfEnvelope
            } else {
                MemoryEvidenceVerdict::Stale
            },
            current_environment_verification: if historical_verified
                && runtime_verified
                && request_geometry == measured_geometry
            {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::OutOfEnvelope
            },
            canonical_route_loadability: if runtime_verified {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::Unverified
            },
            exact_strategy_parameters: if contract.validate_selection(&selection).is_ok() {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::OutOfEnvelope
            },
        };
        let verified = dimensions.all_satisfied();
        MemoryEvidence {
            key: MemoryEvidenceKey {
                model_family: KREA_CONTROL_ROUTE.to_owned(),
                resolved_route: KREA_CONTROL_ROUTE.to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier,
                load_shape: contract.load_shape,
                mode: crate::memory_strategy::memory_mode_from_mode_key(KREA_CONTROL_MODE),
                reference_shape: gen_core::MemoryReferenceShape::Image,
                overlay: Some(overlay.clone()),
                geometry: measured_geometry,
                frames_per_second: None,
                strategy: selection.strategy,
                engaged_composition: contract.engaged_composition(selection.strategy),
                parameters: selection.parameters,
            },
            conformance: if verified {
                MemoryConformanceState::Verified
            } else {
                MemoryConformanceState::ImplementedUnverified
            },
            dimensions,
            calibration_abi: contract
                .calibration
                .as_ref()
                .expect("control calibration")
                .abi,
            calibration_fingerprint: KREA_CONTROL_CALIBRATION.to_owned(),
            sceneworks_revision: "sc-16013".to_owned(),
            inference_revision: measured_closure_digest.to_owned(),
            harness_version: "krea-control-cuda-direct-v1".to_owned(),
            predicted_peak_bytes: bytes(predicted_peak_gb),
            // The manifest records the no-adapter rendered-device observation. Adapter bytes may extend
            // the prediction through the declared OverlayBytes variable, but never rewrite observation.
            observed_peak_bytes: Some(bytes((predicted_peak_gb - adapter_gb).max(0.0))),
            parity: MemoryParityContract::Exact,
            parity_result: if verified {
                MemoryParityResult::Passed
            } else {
                MemoryParityResult::NotRun
            },
        }
    }
}
