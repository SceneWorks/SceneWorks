use super::*;

/// sc-22738: a recovery clip outside the determinism envelope is RECORDED, not refused — and
/// the recording has to survive into the blocker list, because that list is the only place the
/// excursion is named. Every half is asserted here: the envelope still discriminates (a delta
/// inside it adds nothing, one that leaves it on ANY of the three metrics adds a finding), each
/// fault is reported independently, and the finding carries both references and the phase.
///
/// The over-envelope case is the real `ltx_2_3:q4:mlx` reading from the 2026-09-07 MLX walk, so
/// a future reader can match the finding text against the capture that motivated this.
#[test]
fn an_ltx_recovery_outside_the_envelope_is_recorded_as_a_blocker_not_refused() {
    let inside = LtxRecoveryEvidence::default();
    assert!(
        ltx_lifecycle_blockers("q4", MemoryPhase::Denoise, inside, inside).is_empty(),
        "a bit-identical recovery must add no finding"
    );

    // A delta that leaves the envelope on EXACTLY ONE metric, three times over, so no metric
    // can be dropped from `LtxClipDelta::passes` unnoticed.
    for solo in [
        LtxClipDelta {
            maximum: LTX_MAX_THRESHOLD * 1.5,
            mean: 0.0,
            rms: 0.0,
        },
        LtxClipDelta {
            maximum: 0.0,
            mean: LTX_MEAN_THRESHOLD * 1.5,
            rms: 0.0,
        },
        LtxClipDelta {
            maximum: 0.0,
            mean: 0.0,
            rms: LTX_RMS_THRESHOLD * 1.5,
        },
    ] {
        assert!(!solo.passes(), "{solo:?} must leave the envelope");
        let evidence = LtxRecoveryEvidence {
            vs_measured: solo,
            vs_clean_warm: LtxClipDelta::default(),
            clip: LtxClipSummary::default(),
        };
        assert_eq!(
            ltx_lifecycle_blockers("q4", MemoryPhase::Denoise, evidence, inside).len(),
            1,
            "a cancellation excursion on one metric must be recorded: {solo:?}"
        );
        assert_eq!(
            ltx_lifecycle_blockers("q4", MemoryPhase::Denoise, inside, evidence).len(),
            1,
            "an injected-error excursion on one metric must be recorded: {solo:?}"
        );
        assert_eq!(
            ltx_lifecycle_blockers("q4", MemoryPhase::Denoise, evidence, evidence).len(),
            2,
            "the two faults must be reported independently: {solo:?}"
        );
    }

    // The envelope is exactly `ltx_quality_passes` — the warm repeat's own rule — so a delta AT
    // each threshold passes and one ULP-scale step past any of them does not. This is what
    // makes "the recovery check is stricter than the warm-repeat check" false by construction.
    let at_threshold = LtxClipDelta {
        maximum: LTX_MAX_THRESHOLD,
        mean: LTX_MEAN_THRESHOLD,
        rms: LTX_RMS_THRESHOLD,
    };
    assert!(at_threshold.passes(), "the boundary itself must pass");
    assert_eq!(
        at_threshold.passes(),
        ltx_quality_passes(at_threshold.maximum, at_threshold.mean, at_threshold.rms),
        "the recovery envelope must be the warm-repeat helper, not a second literal"
    );

    // The measured `ltx_2_3:q4:mlx` event of 2026-09-07, both faults, to the digit: a recovery
    // clip a FULL SCALE away from a measured clip its own run had already repeated
    // bit-identically. Recorded, and the capture is NOT refused.
    let measured = LtxRecoveryEvidence {
        vs_measured: LtxClipDelta {
            maximum: 1.0,
            mean: 0.190_591,
            rms: 0.287_367,
        },
        vs_clean_warm: LtxClipDelta {
            maximum: 1.0,
            mean: 0.190_591,
            rms: 0.287_367,
        },
        clip: LtxClipSummary {
            degenerate_first_frame: true,
            mean_level: 0.0,
        },
    };
    let blockers = ltx_lifecycle_blockers("q4", MemoryPhase::Denoise, measured, measured);
    assert_eq!(
        blockers.len(),
        2,
        "both faults must be recorded: {blockers:?}"
    );
    let finding = &blockers[0];
    for expected in [
        "LTX-2.3 q4",
        "cancellation",
        "Denoise",
        "against the measured clip max=1.000000, mean=0.190591, rms=0.287367",
        "against the clean warm control max=1.000000",
        "mean level is 0.000000 with a degenerate first frame",
        "rather than discarding the capture",
    ] {
        assert!(
            finding.contains(expected),
            "the finding must carry {expected:?}: {finding}"
        );
    }
}

/// The seventeen recovery diagnostics ship on every lifecycle capture, clean or not, and each one
/// carries its OWN metric. A single transposed field here would publish a mislabelled number
/// that no reader could catch, so every name is bound to a distinct value.
#[test]
fn the_recovery_measurements_name_both_faults_both_references_and_all_three_metrics() {
    let cancel = LtxRecoveryEvidence {
        vs_measured: LtxClipDelta {
            maximum: 1.0 / 255.0,
            mean: 2.0 / 255.0,
            rms: 3.0 / 255.0,
        },
        vs_clean_warm: LtxClipDelta {
            maximum: 4.0 / 255.0,
            mean: 5.0 / 255.0,
            rms: 6.0 / 255.0,
        },
        clip: LtxClipSummary {
            degenerate_first_frame: true,
            mean_level: 13.0 / 255.0,
        },
    };
    let error = LtxRecoveryEvidence {
        vs_measured: LtxClipDelta {
            maximum: 7.0 / 255.0,
            mean: 8.0 / 255.0,
            rms: 9.0 / 255.0,
        },
        vs_clean_warm: LtxClipDelta {
            maximum: 10.0 / 255.0,
            mean: 11.0 / 255.0,
            rms: 12.0 / 255.0,
        },
        clip: LtxClipSummary {
            degenerate_first_frame: false,
            mean_level: 14.0 / 255.0,
        },
    };
    let measured = LtxClipSummary {
        degenerate_first_frame: false,
        mean_level: 15.0 / 255.0,
    };
    let measurements = ltx_recovery_measurements(cancel, error, measured);
    assert_eq!(measurements.len(), 17);
    let named: std::collections::BTreeMap<_, _> = measurements
        .iter()
        .map(|(name, unit, value)| {
            assert_eq!(*unit, "count", "{name} must be a count");
            (*name, *value)
        })
        .collect();
    assert_eq!(named.len(), 17, "every diagnostic name must be distinct");
    for (name, expected) in [
        ("cancelRecoveryVersusMeasuredMaximumErrorMilliPer255", 1_000),
        ("cancelRecoveryVersusMeasuredMeanErrorMilliPer255", 2_000),
        (
            "cancelRecoveryVersusMeasuredRootMeanSquareErrorMilliPer255",
            3_000,
        ),
        (
            "cancelRecoveryVersusCleanWarmMaximumErrorMilliPer255",
            4_000,
        ),
        ("cancelRecoveryVersusCleanWarmMeanErrorMilliPer255", 5_000),
        (
            "cancelRecoveryVersusCleanWarmRootMeanSquareErrorMilliPer255",
            6_000,
        ),
        ("errorRecoveryVersusMeasuredMaximumErrorMilliPer255", 7_000),
        ("errorRecoveryVersusMeasuredMeanErrorMilliPer255", 8_000),
        (
            "errorRecoveryVersusMeasuredRootMeanSquareErrorMilliPer255",
            9_000,
        ),
        (
            "errorRecoveryVersusCleanWarmMaximumErrorMilliPer255",
            10_000,
        ),
        ("errorRecoveryVersusCleanWarmMeanErrorMilliPer255", 11_000),
        (
            "errorRecoveryVersusCleanWarmRootMeanSquareErrorMilliPer255",
            12_000,
        ),
        ("cancelRecoveryMeanLevelMilliPer255", 13_000),
        ("errorRecoveryMeanLevelMilliPer255", 14_000),
        ("measuredClipMeanLevelMilliPer255", 15_000),
        // The boolean travels as 1/0 and is NOT shared between the two faults: the cancel clip
        // collapsed here and the error clip did not.
        ("cancelRecoveryFirstFrameDegenerate", 1),
        ("errorRecoveryFirstFrameDegenerate", 0),
    ] {
        assert_eq!(named.get(name), Some(&expected), "{name}");
    }
    // Thousandths, not levels: a delta a third of a level below the mean threshold must not
    // round to zero, which is the whole reason for the unit.
    assert_eq!(
        LtxClipDelta::milli_per_255(LTX_MEAN_THRESHOLD / 3.0),
        333,
        "a sub-level delta must survive the integer channel"
    );
}

fn ltx_fixture_contract(quant: Option<Quant>) -> mlx_gen::gen_core::MemoryProviderContract {
    let registry = mlx_gen_ltx::provider_registry().unwrap();
    let fixture = registry
        .memory_contract_fixture_registrations()
        .find(|fixture| fixture.provider_id == LTX_PROVIDER)
        .expect("the provider-owned LTX contract fixture");
    (fixture.contract)(&flux2_tests::weights_free_spec(quant)).unwrap()
}

#[test]
fn the_pinned_ltx_provider_fixture_exposes_the_campaign_rungs_and_exact_parameters() {
    for quant in [Some(Quant::Q8), Some(Quant::Q4), None] {
        let contract = ltx_fixture_contract(quant);
        assert_eq!(contract.provider_id, LTX_PROVIDER);
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::StagedResidency),
            [MemoryStrategy::Resident, MemoryStrategy::StagedResidency]
        );
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ]
        );
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(
            decode.parameters.decode_tile_edges,
            vec![768, 640, 512, 448, 384, 320, 256, 192]
        );
        assert_eq!(decode.parameters.decode_overlaps, vec![64]);
    }
}

/// A minimal, otherwise-valid LTX target. Every field the arm reads before it touches the
/// environment or the weights is present, so a test that mutates exactly one axis is testing
/// that axis.
fn ltx_request_json(width: u32, height: u32, frames: u32) -> Value {
    json!({
        "hardware": { "memoryBytes": 128_u64 * 1024 * 1024 * 1024 },
        "planned": {
            "target": {
                "provider": LTX_PROVIDER,
                "modelId": LTX_PROVIDER,
                "tier": "q8",
                "mode": "text_to_video",
                "overlay": "none",
                "geometry": { "width": width, "height": height, "batch": 1, "frames": frames }
            },
            "backend": "mlx",
            "loadShape": "eager_materialization",
            "strategy": {
                "rung": "staged_residency",
                "engagedRungs": ["resident", "staged_residency"],
                "parameters": {}
            },
            "calibrationFingerprint": ltx_calibration_fingerprint("q8")
                .expect("q8 is a shipped LTX tier"),
            "fixture": format!("ltx-2-3-mlx-q8-{width}x{height}-f{frames}-fps24-seed{LTX_SEED}")
        }
    })
}

fn ltx_canary_request_json() -> Value {
    json!({
        "action": "canary",
        "hardware": { "memoryBytes": 128_u64 * 1024 * 1024 * 1024 },
        "planned": {
            "_diagnosticOnly": true,
            "evidenceScope": "fixture",
            "target": {
                "provider": LTX_PROVIDER,
                "modelId": LTX_PROVIDER,
                "tier": "q4",
                "mode": "text_to_video",
                "overlay": "none",
                "geometry": {
                    "width": LTX_CANARY_WIDTH,
                    "height": LTX_CANARY_HEIGHT,
                    "batch": 1,
                    "frames": LTX_CANARY_FRAMES,
                },
            },
            "backend": "mlx",
            "loadShape": "eager_materialization",
            "strategy": {
                "rung": "bounded_decode",
                "engagedRungs": ["resident", "staged_residency", "bounded_decode"],
                "parameters": {
                    "decodeTileEdge": LTX_CANARY_TILE_EDGE,
                    "decodeOverlap": LTX_CANARY_OVERLAP,
                },
            },
            "calibrationFingerprint": ltx_calibration_fingerprint("q4")
                .expect("q4 is a shipped LTX tier"),
            "fixture": LTX_CANARY_FIXTURE,
            "_watchdog": {
                "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            },
            "_canary": {
                "identity": LtxCanaryProfile::Safety.identity(),
                "videoMode": "no_audio",
                "fps": LTX_CANARY_FPS,
                "seed": LTX_CANARY_SEED,
            },
            "_artifact": {
                "repository": protocol::LTX_REPOSITORY,
                "revision": LTX_CANARY_ARTIFACT_REVISION,
                "numericTierInventory": {
                    "files": LTX_CANARY_Q4_INVENTORY_FILES,
                    "bytes": LTX_Q4_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_Q4_INVENTORY_SHA256,
                },
                "textEncoderInventory": {
                    "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                    "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                },
            },
        },
    })
}

fn ltx_product_envelope_canary_request_json() -> Value {
    let mut request = ltx_canary_request_json();
    request["action"] = json!("product_envelope_canary");
    request["planned"]["target"]["geometry"] = json!({
        "width": LTX_PRODUCT_CANARY_WIDTH,
        "height": LTX_PRODUCT_CANARY_HEIGHT,
        "batch": 1,
        "frames": LTX_PRODUCT_CANARY_FRAMES,
    });
    request["planned"]["fixture"] = json!(LTX_PRODUCT_CANARY_FIXTURE);
    request["planned"]["_canary"] = json!({
        "identity": LtxCanaryProfile::ProductEnvelope.identity(),
        "videoMode": "default_av",
        "fps": LTX_PRODUCT_CANARY_FPS,
        "seed": LTX_CANARY_SEED,
    });
    request
}

/// Presents one attestation line the adapter must refuse before its ACK and returns the
/// refusal. The watchdog end stays open until the refusal returns: on macOS `setsockopt`
/// fails with EINVAL once the peer has closed, so a peer thread that wrote and hung up before
/// the adapter configured its socket timeouts turned the refusal under test into a
/// timing-dependent "configure LTX watchdog read timeout: Invalid argument" (sc-25690).
fn refuse_ltx_watchdog_attestation(request: &Value, payload: &Value, why: &str) -> String {
    let (mut watchdog, adapter) = UnixStream::pair().unwrap();
    watchdog
        .write_all(format!("{payload}\n").as_bytes())
        .unwrap();
    let error = consume_ltx_canary_watchdog_attestation_stream(request, adapter).expect_err(why);
    drop(watchdog);
    error
}

fn ltx_campaign_entry_request_json() -> Value {
    let mut request = ltx_request_json(
        LTX_CAMPAIGN_ENTRY_WIDTH,
        LTX_CAMPAIGN_ENTRY_HEIGHT,
        LTX_CAMPAIGN_ENTRY_FRAMES,
    );
    request["action"] = json!(LTX_CAMPAIGN_ENTRY_ACTION);
    request["hardware"] = json!({ "memoryBytes": 128_u64 * 1024 * 1024 * 1024 });
    request["planned"]["logicalCaseId"] = json!(LTX_CAMPAIGN_ENTRY_LOGICAL_CASE_ID);
    request["planned"]["evidenceScope"] = json!("authoritative");
    request["planned"]["target"]["tier"] = json!("q4");
    // The identity is per tier (sc-22737), so retargeting the row retargets its key too.
    request["planned"]["calibrationFingerprint"] =
        json!(ltx_calibration_fingerprint("q4").expect("q4 is a shipped LTX tier"));
    request["planned"]["fixture"] = json!(LTX_CAMPAIGN_ENTRY_FIXTURE);
    request["planned"]["negative"] = json!(false);
    request["planned"]["expectedResult"] = json!("passed");
    request["planned"]["modelLoadPolicy"] = json!("fresh_per_case");
    request["planned"]["modelLoadGroup"] = Value::Null;
    request["planned"]["_watchdog"] =
        json!({ "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES });
    request["planned"]["_campaignEntry"] = json!({
        "identity": LTX_CAMPAIGN_ENTRY_IDENTITY,
        "artifact": {
            "repository": protocol::LTX_REPOSITORY,
            "revision": LTX_CANARY_ARTIFACT_REVISION,
            "numericTierInventory": {
                "files": LTX_CANARY_Q4_INVENTORY_FILES,
                "bytes": LTX_Q4_INVENTORY_BYTES,
                "sha256": LTX_CANARY_Q4_INVENTORY_SHA256,
            },
            "textEncoderInventory": {
                "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
            },
        },
    });
    request["planned"]["_measurementSafety"] = json!({
        "disposition": LTX_SAFETY_REFUSED_OPEN,
        "tierInventoryBytes": LTX_Q4_INVENTORY_BYTES,
        "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
        "incidentCase": "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
        "commonLoad": "complete numeric tier plus shared Gemma stack before geometry-specific work",
        "predictedDecodeBytes": 19_476_906_240_u64,
        "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
        "incidentCalibratedProjectionBytes": 97_906_593_920_u64,
        "projectionAssumptions": [
            "pinned provider decode cost is the only geometry-varying term used",
            "immutable tier inventory delta is added byte-for-byte",
            "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
        ],
        "reason": "incident-calibrated projection is diagnostic only; no proved bound or hard containment admits this row",
    });
    request
}

fn ltx_bounded_carrier_request_json() -> Value {
    let mut request = ltx_canary_request_json();
    request["action"] = json!(LTX_BOUNDED_CARRIER_ACTION);
    request["planned"]["logicalCaseId"] = json!(LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID);
    request["planned"]["target"]["geometry"] = json!({
        "width": LTX_CAMPAIGN_ENTRY_WIDTH,
        "height": LTX_CAMPAIGN_ENTRY_HEIGHT,
        "batch": 1,
        "frames": LTX_CAMPAIGN_ENTRY_FRAMES,
    });
    request["planned"]["fixture"] = json!(LTX_BOUNDED_CARRIER_FIXTURE);
    request["planned"]["negative"] = json!(false);
    request["planned"]["expectedResult"] = json!("passed");
    request["planned"]["modelLoadPolicy"] = json!("fresh_per_case");
    request["planned"]["modelLoadGroup"] = Value::Null;
    request["planned"]
        .as_object_mut()
        .unwrap()
        .remove("_canary");
    let artifact = request["planned"]
        .as_object_mut()
        .unwrap()
        .remove("_artifact")
        .unwrap();
    request["planned"]["_boundedCarrier"] = json!({
        "identity": LTX_BOUNDED_CARRIER_IDENTITY,
        "fps": LTX_CAMPAIGN_ENTRY_FPS,
        "seed": LTX_SEED,
        "videoMode": "default_av",
        "artifact": artifact,
    });
    request
}

fn ltx_bounded_campaign_request_json() -> Value {
    let mut request = ltx_bounded_carrier_request_json();
    request["action"] = json!(LTX_BOUNDED_CAMPAIGN_ACTION);
    request["planned"]
        .as_object_mut()
        .unwrap()
        .remove("_diagnosticOnly");
    request["planned"]["logicalCaseId"] = json!(LTX_BOUNDED_CAMPAIGN_LOGICAL_CASE_ID);
    request["planned"]["evidenceScope"] = json!("authoritative");
    request["planned"]["fixture"] = json!(LTX_BOUNDED_CAMPAIGN_FIXTURE);
    let carrier = request["planned"]
        .as_object_mut()
        .unwrap()
        .remove("_boundedCarrier")
        .unwrap();
    request["planned"]["_boundedCampaignEntry"] = json!({
        "identity": LTX_BOUNDED_CAMPAIGN_IDENTITY,
        "fps": LTX_CAMPAIGN_ENTRY_FPS,
        "seed": LTX_SEED,
        "videoMode": "default_av",
        "spatialDecodeTiles": 24,
        "artifact": carrier["artifact"],
    });
    request["planned"]["_measurementSafety"] = json!({
        "disposition": LTX_SAFETY_REFUSED_OPEN,
        "tierInventoryBytes": LTX_Q4_INVENTORY_BYTES,
        "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
        "incidentCase": "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
        "commonLoad": "complete numeric tier plus shared Gemma stack before geometry-specific work",
        "predictedDecodeBytes": 6_264_848_640_u64,
        "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
        "incidentCalibratedProjectionBytes": 84_694_536_320_u64,
        "projectionAssumptions": [
            "pinned provider decode cost is the only geometry-varying term used",
            "immutable tier inventory delta is added byte-for-byte",
            "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
        ],
        "reason": "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20318 action is admitted",
    });
    request
}

fn ltx_bounded_campaign_request_json_for(tier: &str) -> Value {
    let mut request = ltx_bounded_campaign_request_json();
    let spec = ltx_bounded_campaign_spec(tier).expect("test tier must be allowlisted");
    request["planned"]["target"]["tier"] = json!(spec.tier);
    // The identity is keyed on (provider, tier): a q8 or bf16 row carries its OWN member of the
    // family, never the q4 key the base carrier request was built with.
    request["planned"]["calibrationFingerprint"] =
        json!(ltx_calibration_fingerprint(spec.tier).expect("allowlisted tier"));
    request["planned"]["logicalCaseId"] = json!(spec.logical_case_id);
    request["planned"]["fixture"] = json!(spec.fixture);
    request["planned"]["_boundedCampaignEntry"]["identity"] = json!(spec.identity);
    request["planned"]["_boundedCampaignEntry"]["artifact"]["numericTierInventory"] = json!({
        "files": spec.inventory_files,
        "bytes": spec.inventory_bytes,
        "sha256": spec.inventory_sha256,
    });
    request["planned"]["_measurementSafety"]["tierInventoryBytes"] = json!(spec.inventory_bytes);
    request["planned"]["_measurementSafety"]["incidentCalibratedProjectionBytes"] =
        json!(spec.projection_bytes);
    request["planned"]["_measurementSafety"]["reason"] = json!(if tier == "q4" {
        "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20318 action is admitted"
    } else {
        "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20430 action is admitted"
    });
    request
}

#[test]
fn sc_20318_admits_only_the_exact_two_render_bounded_campaign_row() {
    let request = ltx_bounded_campaign_request_json();
    prevalidate_ltx_bounded_campaign_entry(&request)
        .expect("the exact new bounded campaign row is admissible");
    let geometry = LtxGeometry {
        width: LTX_CAMPAIGN_ENTRY_WIDTH,
        height: LTX_CAMPAIGN_ENTRY_HEIGHT,
        frames: LTX_CAMPAIGN_ENTRY_FRAMES,
        latent_frames: 1 + (LTX_CAMPAIGN_ENTRY_FRAMES - 1) / LTX_TEMPORAL_SCALE,
    };
    let generation_request = ltx_request(geometry, LTX_CAMPAIGN_ENTRY_FPS, LTX_SEED);
    validate_ltx_bounded_carrier_generation_request(&generation_request)
        .expect("both SC-20318 renders use the exact default-A/V request");
    for mutate in [
        |value: &mut GenerationRequest| value.seed = Some(LTX_SEED + 1),
        |value: &mut GenerationRequest| value.fps = Some(LTX_CAMPAIGN_ENTRY_FPS - 1),
        |value: &mut GenerationRequest| value.video_mode = Some("no_audio".to_owned()),
    ] {
        let mut changed = generation_request.clone();
        mutate(&mut changed);
        assert!(validate_ltx_bounded_carrier_generation_request(&changed).is_err());
    }
    for (pointer, value) in [
        ("/action", json!("run")),
        (
            "/planned/logicalCaseId",
            json!(LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID),
        ),
        ("/planned/fixture", json!(LTX_BOUNDED_CARRIER_FIXTURE)),
        ("/planned/target/tier", json!("q8")),
        ("/planned/strategy/parameters/decodeTileEdge", json!(193)),
        ("/planned/strategy/parameters/decodeOverlap", json!(63)),
        ("/planned/_boundedCampaignEntry/identity", json!("mutated")),
        ("/planned/_boundedCampaignEntry/seed", json!(LTX_SEED + 1)),
        (
            "/planned/_boundedCampaignEntry/videoMode",
            json!("no_audio"),
        ),
        (
            "/planned/_boundedCampaignEntry/spatialDecodeTiles",
            json!(23),
        ),
        (
            "/planned/_measurementSafety/predictedDecodeBytes",
            json!(6_264_848_639_u64),
        ),
        ("/planned/_measurementSafety/incidentCase", json!("mutated")),
        ("/planned/_measurementSafety/reason", json!("mutated")),
    ] {
        let mut mutated = request.clone();
        *mutated.pointer_mut(pointer).expect("mutation pointer") = value;
        let error = prevalidate_ltx_bounded_campaign_entry(&mutated)
            .expect_err("every SC-20318 identity mutation must fail before weights");
        assert!(
            error.contains("SC-20318") || error.contains("SC-20430") || error.contains("SC-18946"),
            "{pointer}: {error}"
        );
    }
    let direct = run_ltx_bounded_campaign_entry(&request)
        .expect_err("private JSON alone cannot bypass the live watchdog");
    assert!(
        direct.contains("live external watchdog channel"),
        "{direct}"
    );

    // sc-22738: through the ordinary action the same row is an ordinary capture — admitted
    // by the production budget, not refused by a disposition — so it reaches the environment
    // read that precedes the load.
    let mut ordinary = request.clone();
    ordinary["action"] = json!("run");
    ordinary["planned"]
        .as_object_mut()
        .unwrap()
        .remove("_boundedCampaignEntry");
    assert_ordinary_ltx_run_reaches_the_environment(&ordinary);
}

#[test]
fn sc_20430_exactly_allowlists_q8_and_bf16_bounded_campaign_rows() {
    for tier in ["q8", "bf16"] {
        let request = ltx_bounded_campaign_request_json_for(tier);
        prevalidate_ltx_bounded_campaign_entry(&request)
            .unwrap_or_else(|error| panic!("exact {tier} bounded entry rejected: {error}"));
        for (pointer, mutation) in [
            ("/action", json!("run")),
            ("/planned/target/tier", json!("q4")),
            // The q4 cell's identity is FOREIGN to a q8 or bf16 row: calibration is keyed on
            // (provider, artifact-proven tier), so borrowing the calibrated family's other
            // member must be refused before the row can cost a load.
            (
                "/planned/calibrationFingerprint",
                json!(ltx_calibration_fingerprint("q4").unwrap()),
            ),
            (
                "/planned/logicalCaseId",
                json!(LTX_BOUNDED_CAMPAIGN_LOGICAL_CASE_ID),
            ),
            ("/planned/fixture", json!(LTX_BOUNDED_CAMPAIGN_FIXTURE)),
            ("/planned/_boundedCampaignEntry/identity", json!("mutated")),
            (
                "/planned/_boundedCampaignEntry/artifact/numericTierInventory/bytes",
                json!(LTX_Q4_INVENTORY_BYTES),
            ),
            ("/planned/_boundedCampaignEntry/seed", json!(LTX_SEED + 1)),
            (
                "/planned/strategy/parameters/decodeTileEdge",
                json!(LTX_CANARY_TILE_EDGE + 1),
            ),
        ] {
            let mut changed = request.clone();
            *changed.pointer_mut(pointer).expect("mutation pointer") = mutation;
            assert!(
                prevalidate_ltx_bounded_campaign_entry(&changed).is_err(),
                "{tier} mutation {pointer} reached model resolution"
            );
        }
        let direct = run_ltx_bounded_campaign_entry(&request)
            .expect_err("private JSON alone cannot bypass the watchdog");
        assert!(
            direct.contains("live external watchdog channel"),
            "{direct}"
        );
        // sc-22738: the ordinary action admits every tier's row through the production
        // budget rather than refusing it by disposition.
        let mut ordinary = request.clone();
        ordinary["action"] = json!("run");
        ordinary["planned"]
            .as_object_mut()
            .unwrap()
            .remove("_boundedCampaignEntry");
        assert_ordinary_ltx_run_reaches_the_environment(&ordinary);
    }
    assert!(ltx_bounded_campaign_spec("fp16").is_err());
}

#[test]
fn sc_20254_admits_only_the_exact_single_render_bounded_carrier() {
    let request = ltx_bounded_carrier_request_json();
    let (geometry, selection) = prevalidate_ltx_bounded_carrier_proof(&request)
        .expect("the exact diagnostic bounded carrier is admissible");
    let decode = LtxDecodePlan::resolve_for_selection(&selection, geometry).unwrap();
    assert!(decode.tiling_engaged());
    assert_eq!(decode.spatial_tile_px(), u64::from(LTX_CANARY_TILE_EDGE));
    assert_eq!(decode.spatial_overlap_px(), u64::from(LTX_CANARY_OVERLAP));
    assert_eq!(decode.spatial_tile_count(geometry).unwrap(), 24);

    for (pointer, value) in [
        ("/action", json!(LTX_CAMPAIGN_ENTRY_ACTION)),
        (
            "/planned/logicalCaseId",
            json!(LTX_CAMPAIGN_ENTRY_LOGICAL_CASE_ID),
        ),
        ("/planned/fixture", json!(LTX_CAMPAIGN_ENTRY_FIXTURE)),
        ("/planned/strategy/rung", json!("staged_residency")),
        (
            "/planned/strategy/parameters/decodeTileEdge",
            json!(LTX_CANARY_TILE_EDGE + 1),
        ),
        (
            "/planned/strategy/parameters/decodeOverlap",
            json!(LTX_CANARY_OVERLAP + 1),
        ),
        ("/planned/_boundedCarrier/identity", json!("mutated")),
        ("/planned/_boundedCarrier/seed", json!(LTX_SEED + 1)),
        (
            "/planned/_watchdog/maxFootprintBytes",
            json!(LTX_CANARY_MAX_FOOTPRINT_BYTES + 1),
        ),
    ] {
        let mut mutated = request.clone();
        *mutated.pointer_mut(pointer).expect("mutation pointer") = value;
        let error = prevalidate_ltx_bounded_carrier_proof(&mutated)
            .expect_err("every bounded-carrier identity mutation must fail closed");
        assert!(error.contains("SC-20254"), "{pointer}: {error}");
    }

    let generation_request = ltx_request(geometry, LTX_CAMPAIGN_ENTRY_FPS, LTX_SEED);
    validate_ltx_bounded_carrier_generation_request(&generation_request)
        .expect("the exact provider request is bound independently");
    for mutate in [
        |value: &mut GenerationRequest| value.seed = Some(LTX_SEED + 1),
        |value: &mut GenerationRequest| value.fps = Some(LTX_CAMPAIGN_ENTRY_FPS + 1),
        |value: &mut GenerationRequest| value.frames = Some(LTX_CAMPAIGN_ENTRY_FRAMES - 1),
        |value: &mut GenerationRequest| value.video_mode = Some("no_audio".to_owned()),
    ] {
        let mut mutated = generation_request.clone();
        mutate(&mut mutated);
        assert!(validate_ltx_bounded_carrier_generation_request(&mutated).is_err());
    }

    let direct = run_ltx_bounded_carrier_proof(&request)
        .expect_err("private JSON alone cannot bypass the live watchdog");
    assert!(
        direct.contains("live external watchdog channel"),
        "{direct}"
    );
}

#[test]
fn sc_20191_admits_only_the_exact_private_campaign_row_before_weight_work() {
    let request = ltx_campaign_entry_request_json();
    prevalidate_ltx_campaign_entry(&request).expect("the exact canonical row is admissible");

    for (pointer, value) in [
        ("/action", json!("run")),
        ("/planned/logicalCaseId", json!("implan-mutated")),
        ("/planned/target/tier", json!("q8")),
        ("/planned/target/geometry/frames", json!(97)),
        (
            "/planned/fixture",
            json!("ltx-2-3-mlx-q4-768x512-f121-fps30-seed1"),
        ),
        (
            "/planned/strategy/parameters",
            json!({ "decodeTileEdge": 192 }),
        ),
        ("/planned/evidenceScope", json!("fixture")),
        ("/planned/modelLoadPolicy", json!("batch_rungs")),
        (
            "/planned/_watchdog/maxFootprintBytes",
            json!(LTX_CANARY_MAX_FOOTPRINT_BYTES - 1),
        ),
        (
            "/planned/_campaignEntry/identity",
            json!("sc-20191-mutated"),
        ),
        (
            "/planned/_measurementSafety/predictedDecodeBytes",
            json!(19_476_906_239_u64),
        ),
    ] {
        let mut mutated = request.clone();
        *mutated.pointer_mut(pointer).expect("mutation pointer") = value;
        let error = prevalidate_ltx_campaign_entry(&mutated)
            .expect_err("every campaign-entry identity mutation must fail closed");
        assert!(
            error.contains("SC-20191") || error.contains("SC-18946"),
            "{pointer}: {error}"
        );
    }

    let direct = run_ltx_campaign_entry(&request)
        .expect_err("canonical JSON alone cannot bypass the live watchdog");
    assert!(
        direct.contains("live external watchdog channel"),
        "{direct}"
    );
}

#[test]
fn sc_20216_phase_channel_is_exact_monotonic_and_action_bound() {
    let request = ltx_campaign_entry_request_json();
    let host_memory = request["hardware"]["memoryBytes"].as_u64().unwrap();
    let nonce = "cd".repeat(32);
    let base = json!({
        "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
        "nonce": nonce,
        "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
        "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
        "hostMemoryBytes": host_memory,
        "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
        "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
    });
    let error = refuse_ltx_watchdog_attestation(
        &request,
        &base,
        "campaign entry must require authenticated phases",
    );
    assert!(
        error.contains("action-bound authenticated provider phase contract"),
        "{error}"
    );

    let (adapter, mut watchdog) = UnixStream::pair().unwrap();
    let mut attestation = LtxCanaryWatchdogAttestation {
        max_footprint_bytes: LTX_CANARY_MAX_FOOTPRINT_BYTES,
        max_runtime_seconds: LTX_CANARY_MAX_RUNTIME_SECONDS,
        host_memory_bytes: host_memory,
        min_initial_memory_free_bytes: 0,
        min_memory_free_bytes: 0,
        nonce: nonce.clone(),
        stream: Some(adapter),
    };
    let lease = attestation.start_lease().unwrap();
    let (marked_sender, marked_receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut lease = lease;
        let result = lease.mark("common_load");
        marked_sender.send((lease, result)).unwrap();
    });
    assert_eq!(
        read_watchdog_line(&mut watchdog).unwrap(),
        format!("PHASE {nonce} 1 common_load")
    );
    assert!(matches!(
        marked_receiver.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    watchdog
        .write_all(format!("PHASE_ACK {nonce} 1 common_load\n").as_bytes())
        .unwrap();
    let (mut lease, result) = marked_receiver.recv().unwrap();
    result.expect("phase entry remains blocked until the watchdog acknowledges it");

    let remaining_nonce = nonce.clone();
    let peer = std::thread::spawn(move || {
        for (index, name) in LTX_PROVIDER_PHASE_NAMES.iter().enumerate().skip(1) {
            assert_eq!(
                read_watchdog_line(&mut watchdog).unwrap(),
                format!("PHASE {remaining_nonce} {} {name}", index + 1)
            );
            watchdog
                .write_all(format!("PHASE_ACK {remaining_nonce} {} {name}\n", index + 1).as_bytes())
                .unwrap();
        }
        assert_eq!(
            read_watchdog_line(&mut watchdog).unwrap(),
            format!("DONE {remaining_nonce}")
        );
        watchdog
            .write_all(format!("BYE {remaining_nonce}\n").as_bytes())
            .unwrap();
    });
    for name in LTX_PROVIDER_PHASE_NAMES.iter().skip(1) {
        lease.mark(name).expect("exact next phase");
    }
    assert!(lease.mark("cleanup").unwrap_err().contains("exceeded"));
    lease.complete().unwrap();
    peer.join().unwrap();

    let (_timeout_sender, timeout_receiver) = std::sync::mpsc::sync_channel(1);
    assert!(wait_for_ltx_phase_acknowledgement(
        &timeout_receiver,
        1,
        "common_load",
        Duration::ZERO,
    )
    .unwrap_err()
    .contains("acknowledgement"));
    let (foreign_sender, foreign_receiver) = std::sync::mpsc::sync_channel(1);
    foreign_sender
        .send(Ok((2, "primary_conditioning".to_owned())))
        .unwrap();
    assert!(wait_for_ltx_phase_acknowledgement(
        &foreign_receiver,
        1,
        "common_load",
        Duration::ZERO,
    )
    .unwrap_err()
    .contains("foreign provider phase"));

    let (writer, _reader) = UnixStream::pair().unwrap();
    let (_sender, completion) = std::sync::mpsc::sync_channel(1);
    let (_phase_sender, phase_acknowledgements) = std::sync::mpsc::sync_channel(1);
    let mut reordered = LtxCanaryWatchdogLease {
        writer,
        completion,
        phase_acknowledgements,
        nonce,
        phase_sequence: 0,
        expected_phases: &LTX_PROVIDER_PHASE_NAMES,
    };
    assert!(reordered
        .mark("primary_decode")
        .unwrap_err()
        .contains("reordered"));
}

#[test]
fn sc_20254_phase_attestation_accepts_only_its_exact_five_phase_socket_contract() {
    let request = ltx_bounded_carrier_request_json();
    let host_memory = request["hardware"]["memoryBytes"].as_u64().unwrap();
    let nonce = "ef".repeat(32);
    let payload = json!({
        "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
        "nonce": nonce,
        "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
        "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
        "hostMemoryBytes": host_memory,
        "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
        "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
        "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
        "providerPhaseProfile": LTX_BOUNDED_CARRIER_PHASE_PROFILE,
        "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
    });
    let (mut watchdog, adapter) = UnixStream::pair().unwrap();
    let peer_nonce = nonce.clone();
    let peer_payload = payload.clone();
    let peer = std::thread::spawn(move || {
        watchdog
            .write_all(format!("{peer_payload}\n").as_bytes())
            .unwrap();
        assert_eq!(
            read_watchdog_line(&mut watchdog).unwrap(),
            format!("ACK {peer_nonce}")
        );
        watchdog
            .write_all(format!("GO {peer_nonce}\n").as_bytes())
            .unwrap();
        for (index, name) in LTX_BOUNDED_CARRIER_PHASE_NAMES.iter().enumerate() {
            assert_eq!(
                read_watchdog_line(&mut watchdog).unwrap(),
                format!("PHASE {peer_nonce} {} {name}", index + 1)
            );
            watchdog
                .write_all(format!("PHASE_ACK {peer_nonce} {} {name}\n", index + 1).as_bytes())
                .unwrap();
        }
        assert_eq!(
            read_watchdog_line(&mut watchdog).unwrap(),
            format!("DONE {peer_nonce}")
        );
        watchdog
            .write_all(format!("BYE {peer_nonce}\n").as_bytes())
            .unwrap();
    });
    let mut attestation =
        consume_ltx_canary_watchdog_attestation_stream(&request, adapter).unwrap();
    let mut lease = attestation
        .start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)
        .unwrap();
    for name in LTX_BOUNDED_CARRIER_PHASE_NAMES {
        lease
            .mark(name)
            .expect("watchdog must ACK each exact phase");
    }
    lease.complete().unwrap();
    peer.join().unwrap();

    for mutate in [
        |value: &mut Value| {
            value["providerPhaseProfile"] = json!(LTX_CAMPAIGN_ENTRY_PHASE_PROFILE);
        },
        |value: &mut Value| {
            value["providerPhases"] = json!(LTX_PROVIDER_PHASE_NAMES);
        },
        |value: &mut Value| {
            value["providerPhases"] = json!([
                "primary_conditioning",
                "common_load",
                "primary_denoise",
                "primary_decode",
                "cleanup",
            ]);
        },
    ] {
        let mut mutated = json!({
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "nonce": nonce,
            "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
            "hostMemoryBytes": host_memory,
            "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
            "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
            "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
            "providerPhaseProfile": LTX_BOUNDED_CARRIER_PHASE_PROFILE,
            "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
        });
        mutate(&mut mutated);
        let error = refuse_ltx_watchdog_attestation(
            &request,
            &mutated,
            "profile, list and order mutations must fail closed",
        );
        assert!(
            error.contains("action-bound authenticated provider phase contract"),
            "{error}"
        );
    }
}

#[test]
fn sc_20318_phase_attestation_is_action_bound_to_its_exact_five_phases() {
    let request = ltx_bounded_campaign_request_json();
    let host_memory = request["hardware"]["memoryBytes"].as_u64().unwrap();
    let nonce = "ab".repeat(32);
    let payload = json!({
        "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
        "nonce": nonce,
        "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
        "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
        "hostMemoryBytes": host_memory,
        "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
        "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
        "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
        "providerPhaseProfile": LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE,
        "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
    });
    let (mut watchdog, adapter) = UnixStream::pair().unwrap();
    let peer_nonce = nonce.clone();
    let peer_payload = payload.clone();
    let peer = std::thread::spawn(move || {
        watchdog
            .write_all(format!("{peer_payload}\n").as_bytes())
            .unwrap();
        assert_eq!(
            read_watchdog_line(&mut watchdog).unwrap(),
            format!("ACK {peer_nonce}")
        );
        watchdog
            .write_all(format!("GO {peer_nonce}\n").as_bytes())
            .unwrap();
        for (index, name) in LTX_BOUNDED_CARRIER_PHASE_NAMES.iter().enumerate() {
            assert_eq!(
                read_watchdog_line(&mut watchdog).unwrap(),
                format!("PHASE {peer_nonce} {} {name}", index + 1)
            );
            watchdog
                .write_all(format!("PHASE_ACK {peer_nonce} {} {name}\n", index + 1).as_bytes())
                .unwrap();
        }
        assert_eq!(
            read_watchdog_line(&mut watchdog).unwrap(),
            format!("DONE {peer_nonce}")
        );
        watchdog
            .write_all(format!("BYE {peer_nonce}\n").as_bytes())
            .unwrap();
    });
    let mut attestation =
        consume_ltx_canary_watchdog_attestation_stream(&request, adapter).unwrap();
    let mut lease = attestation
        .start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)
        .unwrap();
    for name in LTX_BOUNDED_CARRIER_PHASE_NAMES {
        lease.mark(name).unwrap();
    }
    lease.complete().unwrap();
    peer.join().unwrap();

    for mutate in [
        |value: &mut Value| {
            value["providerPhaseProfile"] = json!(LTX_BOUNDED_CARRIER_PHASE_PROFILE)
        },
        |value: &mut Value| value["providerPhaseProfile"] = json!("foreign-profile"),
        |value: &mut Value| value["providerPhases"] = json!(LTX_PROVIDER_PHASE_NAMES),
        |value: &mut Value| {
            value["providerPhases"] = json!([
                "common_load",
                "primary_conditioning",
                "primary_denoise",
                "primary_decode"
            ])
        },
        |value: &mut Value| {
            value["providerPhases"] = json!([
                "primary_conditioning",
                "common_load",
                "primary_denoise",
                "primary_decode",
                "cleanup"
            ])
        },
    ] {
        let mut changed = payload.clone();
        mutate(&mut changed);
        let error = refuse_ltx_watchdog_attestation(
            &request,
            &changed,
            "SC-20318 phase profile/list/order mutations must fail",
        );
        assert!(
            error.contains("action-bound authenticated provider phase contract"),
            "{error}"
        );
    }
}

#[test]
fn sc_20191_requires_the_exact_untiled_full_av_response_carrier() {
    let diagnostics = |name: &str, value: u64| json!({ "name": name, "value": value });
    let mut fragment = json!({
        "strategy": {
            "rung": "staged_residency",
            "engagedRungs": ["resident", "staged_residency"],
            "parameters": {},
        },
        "output": {
            "frames": 121,
            "fps": 30,
            "audio": { "present": true, "samples": 1, "sampleRate": 48_000, "channels": 2 },
            "firstFrameNondegenerate": true,
        },
        "diagnostics": { "measurements": [
            diagnostics("renderedFrames", 121),
            diagnostics("outputFps", 30),
            diagnostics("audioTrackDecoded", 1),
            diagnostics("decodeTilingEngaged", 0),
            diagnostics("decodeTileSpatialPx", 0),
            diagnostics("decodeTileOverlapPx", 0),
            diagnostics("latentTemporalDepth", 16),
            diagnostics("latentTokens", 6_144),
        ] },
    });
    validate_ltx_campaign_entry_fragment(&fragment).expect("exact carrier");
    fragment["diagnostics"]["measurements"][3]["value"] = json!(1);
    let tiled = validate_ltx_campaign_entry_fragment(&fragment)
        .expect_err("a tiled carrier cannot publish the untiled canonical row");
    assert!(tiled.contains("decodeTilingEngaged"), "{tiled}");
}

#[test]
fn the_ltx_safety_canary_is_an_exact_non_promotable_multi_tile_tuple() {
    let request = ltx_canary_request_json();
    let selection = validate_ltx_canary_plan(&request).expect("exact canary tuple");
    assert_eq!(selection.strategy, MemoryStrategy::BoundedDecode);
    assert_eq!(selection.tier.quant, Some(Quant::Q4));
    assert_eq!(
        selection.parameters.decode_tile_edge,
        Some(LTX_CANARY_TILE_EDGE)
    );
    assert_eq!(
        selection.parameters.decode_overlap,
        Some(LTX_CANARY_OVERLAP)
    );
    let generated = ltx_canary_generation_request();
    assert_eq!(generated.prompt, "a sunlit pine branch, static camera");
    assert_eq!(
        (generated.width, generated.height, generated.frames),
        (LTX_CANARY_WIDTH, LTX_CANARY_HEIGHT, Some(LTX_CANARY_FRAMES))
    );
    assert_eq!(generated.video_mode.as_deref(), Some("no_audio"));
    let memory = generated.memory.expect("bounded decode carrier");
    assert!(memory.tile_vae_decode);
    assert_eq!(memory.decode_tile_edge, Some(LTX_CANARY_TILE_EDGE));
    assert_eq!(memory.decode_overlap, Some(LTX_CANARY_OVERLAP));

    let mut admitted = ltx_canary_request_for_provider_admission(ltx_canary_generation_request())
        .expect("the exact diagnostic tuple may use the private admission bridge");
    assert_eq!(admitted.video_mode, None);
    restore_ltx_canary_no_audio_after_configuration(&mut admitted)
        .expect("no_audio is restored only after provider configuration");
    assert_eq!(admitted.video_mode.as_deref(), Some("no_audio"));

    let mut production_default = ltx_canary_generation_request();
    production_default.video_mode = Some("default".to_owned());
    assert!(ltx_canary_request_for_provider_admission(production_default).is_err());
    let mut provider_override = ltx_canary_generation_request();
    provider_override.video_mode = Some("default".to_owned());
    assert!(restore_ltx_canary_no_audio_after_configuration(&mut provider_override).is_err());
}

#[test]
fn the_ltx_product_envelope_canary_is_exact_full_av_and_physically_multi_tile() {
    let request = ltx_product_envelope_canary_request_json();
    let selection = validate_ltx_product_envelope_canary_plan(&request)
        .expect("exact product-envelope canary tuple");
    assert_eq!(selection.strategy, MemoryStrategy::BoundedDecode);
    let generated = ltx_product_envelope_canary_generation_request();
    assert_eq!(
        (
            generated.width,
            generated.height,
            generated.frames,
            generated.fps
        ),
        (
            LTX_PRODUCT_CANARY_WIDTH,
            LTX_PRODUCT_CANARY_HEIGHT,
            Some(LTX_PRODUCT_CANARY_FRAMES),
            Some(LTX_PRODUCT_CANARY_FPS),
        )
    );
    assert_eq!(
        generated.prompt,
        "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
    );
    assert_eq!(generated.video_mode, None, "default A/V must stay unset");
    let geometry = LtxGeometry {
        width: LTX_PRODUCT_CANARY_WIDTH,
        height: LTX_PRODUCT_CANARY_HEIGHT,
        frames: LTX_PRODUCT_CANARY_FRAMES,
        latent_frames: 1 + (LTX_PRODUCT_CANARY_FRAMES - 1) / LTX_TEMPORAL_SCALE,
    };
    let decode = LtxDecodePlan::resolve_for_selection(&selection, geometry)
        .expect("exact bounded decode plan");
    assert_eq!(decode.spatial_tile_count(geometry).unwrap(), 24);
    assert!(validate_diagnostic_audio(
        LtxCanaryProfile::ProductEnvelope,
        Some(DiagnosticAudioIdentity {
            samples: 1,
            sample_rate: 24_000,
            channels: 2,
        }),
    )
    .is_ok());
    for audio in [
        None,
        Some(DiagnosticAudioIdentity {
            samples: 0,
            sample_rate: 24_000,
            channels: 2,
        }),
        Some(DiagnosticAudioIdentity {
            samples: 1,
            sample_rate: 0,
            channels: 2,
        }),
        Some(DiagnosticAudioIdentity {
            samples: 1,
            sample_rate: 24_000,
            channels: 0,
        }),
    ] {
        assert!(validate_diagnostic_audio(LtxCanaryProfile::ProductEnvelope, audio).is_err());
    }
    assert!(validate_diagnostic_audio(
        LtxCanaryProfile::Safety,
        Some(DiagnosticAudioIdentity {
            samples: 1,
            sample_rate: 24_000,
            channels: 2,
        }),
    )
    .is_err());
    let direct = run_ltx_product_envelope_canary(&request)
        .expect_err("product-envelope canary must refuse without the external watchdog");
    assert!(direct.contains("live external watchdog channel"));
}

#[test]
fn the_ltx_safety_canary_accounts_for_only_the_exact_av_bfloat16_ones_cache() {
    let expected = ltx_canary_ones_cache_bytes().expect("checked ONES_CACHE arithmetic");
    assert_eq!(
        expected,
        (LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION + LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION)
            * BFLOAT16_BYTES_PER_ELEMENT
    );
    assert!(validate_ltx_canary_pre_provider(AllocatorState {
        active: 0,
        cache: 0,
    })
    .is_ok());
    let pre_provider = AllocatorState {
        active: 7,
        cache: 0,
    };
    assert!(validate_ltx_canary_cleanup(
        pre_provider,
        AllocatorState {
            active: pre_provider.active + expected,
            cache: 0,
        },
        expected,
    )
    .is_ok());

    let low = validate_ltx_canary_cleanup(
        pre_provider,
        AllocatorState {
            active: pre_provider.active + expected - 1,
            cache: 0,
        },
        expected,
    )
    .expect_err("one byte below the exact active identity must fail");
    assert!(low.contains("intentional persistent active"));
    let high = validate_ltx_canary_cleanup(
        pre_provider,
        AllocatorState {
            active: pre_provider.active + expected + 1,
            cache: 0,
        },
        expected,
    )
    .expect_err("one byte above the exact active identity must fail");
    assert!(high.contains("intentional persistent active"));

    let dirty_pre = validate_ltx_canary_pre_provider(AllocatorState {
        active: 0,
        cache: 1,
    })
    .expect_err("a non-empty baseline cache must fail");
    assert!(dirty_pre.contains("preProviderCacheBytes 1"));
    let dirty_post = validate_ltx_canary_cleanup(
        pre_provider,
        AllocatorState {
            active: pre_provider.active + expected,
            cache: 1,
        },
        expected,
    )
    .expect_err("a retained post-cleanup cache byte must fail");
    assert!(dirty_post.contains("postCleanupCacheBytes 1"));

    let overflow = validate_ltx_canary_cleanup(
        AllocatorState {
            active: u64::MAX,
            cache: 0,
        },
        AllocatorState::default(),
        expected,
    )
    .expect_err("pre-provider plus persistent active bytes must not wrap");
    assert_eq!(
        overflow,
        "LTX safety canary cleanup active-byte arithmetic overflowed"
    );
}

#[test]
fn the_ltx_safety_canary_requires_the_live_watchdog_ack_go_channel_before_limits() {
    let request = ltx_canary_request_json();
    let host_memory = request["hardware"]["memoryBytes"]
        .as_u64()
        .expect("canary host memory");
    let nonce = "ab".repeat(32);
    let payload = json!({
        "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
        "nonce": nonce,
        "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
        "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
        "hostMemoryBytes": host_memory,
        "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
        "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
    });
    let (mut watchdog, adapter) = UnixStream::pair().expect("watchdog socket pair");
    let expected_nonce = nonce.clone();
    let valid_payload = payload.clone();
    let watchdog_thread = std::thread::spawn(move || {
        watchdog
            .write_all(format!("{valid_payload}\n").as_bytes())
            .expect("send attestation");
        assert_eq!(
            read_watchdog_line(&mut watchdog).expect("adapter ACK"),
            format!("ACK {expected_nonce}")
        );
        watchdog
            .write_all(format!("GO {expected_nonce}\n").as_bytes())
            .expect("release canary");
        watchdog
            .write_all(format!("PING {expected_nonce}\n").as_bytes())
            .expect("maintain live watchdog lease");
        assert_eq!(
            read_watchdog_line(&mut watchdog).expect("adapter DONE"),
            format!("DONE {expected_nonce}")
        );
        watchdog
            .write_all(format!("BYE {expected_nonce}\n").as_bytes())
            .expect("complete canary lease");
    });
    let mut attested = consume_ltx_canary_watchdog_attestation_stream(&request, adapter)
        .expect("exact live watchdog handshake");
    let lease = attested.start_lease().expect("held watchdog lease");
    lease.complete().expect("DONE/BYE completion handshake");
    watchdog_thread.join().expect("watchdog thread");
    assert_eq!(attested.max_footprint_bytes, LTX_CANARY_MAX_FOOTPRINT_BYTES);
    assert_eq!(attested.max_runtime_seconds, LTX_CANARY_MAX_RUNTIME_SECONDS);
    assert!(std::env::var_os("SCENEWORKS_MEMORY_WATCHDOG_SOCKET").is_none());
    let direct = consume_ltx_canary_watchdog_attestation(&request)
        .expect_err("direct stdin canary has no live watchdog channel");
    assert!(direct.contains("live external watchdog channel"));
    let direct_action = run_ltx_canary(&request)
        .expect_err("public canary action must refuse without the live watchdog");
    assert!(direct_action.contains("live external watchdog channel"));

    let mut swap_gated = payload.clone();
    swap_gated["minSwapFreeBytes"] = json!(1024_u64 * 1024 * 1024);
    let error = refuse_ltx_watchdog_attestation(
        &request,
        &swap_gated,
        "an arbitrary swap reserve must not enter the reviewed canary bounds",
    );
    assert!(error.contains("exact reviewed bounds"), "{error}");

    let mut percent_gated = payload.clone();
    percent_gated["minInitialMemoryFreePercent"] = json!(70);
    let error = refuse_ltx_watchdog_attestation(
        &request,
        &percent_gated,
        "a redundant percentage floor must not enter the reviewed canary bounds",
    );
    assert!(error.contains("exact reviewed bounds"), "{error}");

    let mut weakened = payload;
    weakened["minInitialMemoryFreeBytes"] =
        json!(2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100) - 1);
    let error = refuse_ltx_watchdog_attestation(
        &request,
        &weakened,
        "weakened runtime floor must not attest",
    );
    assert!(error.contains("exact reviewed bounds"), "{error}");
}

#[test]
fn the_ltx_safety_canary_rejects_every_identity_or_safety_mutation_before_load() {
    type CanaryMutation = (&'static str, fn(&mut Value));
    let mutations: [CanaryMutation; 16] = [
        ("wrong canary identity", |request| {
            request["planned"]["_canary"]["identity"] = json!("sc-20169-product-envelope")
        }),
        ("promotable scope", |request| {
            request["planned"]["evidenceScope"] = json!("authoritative")
        }),
        ("campaign geometry", |request| {
            request["planned"]["target"]["geometry"]["width"] = json!(768)
        }),
        ("single pass", |request| {
            request["planned"]["strategy"]["rung"] = json!("staged_residency")
        }),
        ("wrong backend", |request| {
            request["planned"]["backend"] = json!("candle")
        }),
        ("missing staged rung", |request| {
            request["planned"]["strategy"]["engagedRungs"] = json!(["resident", "bounded_decode"])
        }),
        ("wrong tile edge", |request| {
            request["planned"]["strategy"]["parameters"]["decodeTileEdge"] = json!(256)
        }),
        ("wrong overlap", |request| {
            request["planned"]["strategy"]["parameters"]["decodeOverlap"] = json!(32)
        }),
        ("audio-enabled variant", |request| {
            request["planned"]["_canary"]["videoMode"] = json!("default")
        }),
        ("ceiling drift", |request| {
            request["planned"]["_watchdog"]["maxFootprintBytes"] =
                json!(LTX_CANARY_MAX_FOOTPRINT_BYTES + 1)
        }),
        ("wrong fixture", |request| {
            request["planned"]["fixture"] = json!("campaign-row")
        }),
        ("wrong artifact revision", |request| {
            request["planned"]["_artifact"]["revision"] = json!("0".repeat(40))
        }),
        ("wrong artifact inventory", |request| {
            request["planned"]["_artifact"]["numericTierInventory"]["sha256"] =
                json!("0".repeat(64))
        }),
        ("wrong text encoder bytes", |request| {
            request["planned"]["_artifact"]["textEncoderInventory"]["bytes"] =
                json!(LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES - 1)
        }),
        ("wrong text encoder digest", |request| {
            request["planned"]["_artifact"]["textEncoderInventory"]["sha256"] =
                json!("0".repeat(64))
        }),
        ("undersized host", |request| {
            request["hardware"]["memoryBytes"] = json!(LTX_CANARY_MAX_FOOTPRINT_BYTES * 2 - 1)
        }),
    ];
    for (name, mutate) in mutations {
        let mut request = ltx_canary_request_json();
        mutate(&mut request);
        assert!(
            validate_ltx_canary_plan(&request).is_err(),
            "{name} must fail before environment/provider/model access"
        );
    }
}

#[test]
fn the_ltx_product_envelope_canary_rejects_every_tuple_mutation_before_load() {
    type CanaryMutation = (&'static str, fn(&mut Value));
    let mutations: [CanaryMutation; 10] = [
        ("wrong action", |request| {
            request["action"] = json!("canary")
        }),
        ("no audio", |request| {
            request["planned"]["_canary"]["videoMode"] = json!("no_audio")
        }),
        ("wrong width", |request| {
            request["planned"]["target"]["geometry"]["width"] = json!(512)
        }),
        ("wrong height", |request| {
            request["planned"]["target"]["geometry"]["height"] = json!(768)
        }),
        ("wrong frames", |request| {
            request["planned"]["target"]["geometry"]["frames"] = json!(89)
        }),
        ("wrong fps", |request| {
            request["planned"]["_canary"]["fps"] = json!(30)
        }),
        ("wrong carrier", |request| {
            request["planned"]["strategy"]["parameters"]["decodeTileEdge"] = json!(384)
        }),
        ("campaign scope", |request| {
            request["planned"]["evidenceScope"] = json!("campaign")
        }),
        ("watchdog threshold", |request| {
            request["planned"]["_watchdog"]["maxFootprintBytes"] =
                json!(LTX_CANARY_MAX_FOOTPRINT_BYTES - 1)
        }),
        ("artifact drift", |request| {
            request["planned"]["_artifact"]["numericTierInventory"]["sha256"] =
                json!("0".repeat(64))
        }),
    ];
    for (name, mutate) in mutations {
        let mut request = ltx_product_envelope_canary_request_json();
        mutate(&mut request);
        assert!(
            validate_ltx_product_envelope_canary_plan(&request).is_err(),
            "{name} must fail before environment/provider/model access"
        );
    }
}

#[test]
fn the_ltx_canary_wired_limit_never_exceeds_the_device_ceiling() {
    let requested = LTX_CANARY_MAX_FOOTPRINT_BYTES as usize;
    assert_eq!(ltx_canary_wired_limit(requested, 120), 80);
    assert_eq!(ltx_canary_wired_limit(40, 120), 40);
    assert_eq!(ltx_canary_wired_limit(requested, 0), 0);
}

#[test]
fn the_ltx_canary_process_global_limits_are_serialized_and_restore_memory() {
    let previous;
    {
        let mut limits = LtxCanaryLimits::install().expect("install exact canary limits");
        previous = limits.previous_memory;
        assert_eq!(get_memory_limit(), LTX_CANARY_MAX_FOOTPRINT_BYTES as usize);
        limits.restore();
        assert_eq!(get_memory_limit(), previous);
        let restored_wired = set_wired_limit(limits.wired);
        set_wired_limit(restored_wired);
        assert_eq!(restored_wired, limits.previous_wired);
    }
    let _restoration_guard = LTX_MEMORY_LIMIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(get_memory_limit(), previous);
}

/// The port of the shipped LTX frame ladder must reproduce these 18 transcribed values, one per
/// point the declared limits can reach. These 18 values ARE the envelope, so an unnoticed drift
/// would silently widen or narrow what this arm accepts.
///
/// This is a TRANSCRIPTION check, not a call into `sceneworks_core::video_request::ltx_frame_count`
/// — this crate depends on core, but the shipped helper is not `const fn`, while this port
/// computes compile-time bucket constants (see `ltx_snapped_frame_count`).
/// The other half of the binding is `ltx_frame_count_matches_the_sc_18808_calibration_ladder` in
/// `crates/sceneworks-core/src/video_request.rs`, which pins the same 18 pairs against the shipped
/// function in a default member. Keep the two tables identical; changing one alone is the drift
/// both exist to catch (sc-18808 review).
#[test]
fn ltx_frame_ladder_port_matches_the_transcribed_shipped_ladder() {
    for (duration, fps, expected) in [
        (4, 24, 97),
        (4, 25, 97),
        (4, 30, 121),
        (6, 24, 145),
        (6, 25, 153),
        (6, 30, 177),
        (8, 24, 193),
        (8, 25, 201),
        (8, 30, 241),
        (10, 24, 241),
        (10, 25, 249),
        (10, 30, 297),
        (12, 24, 289),
        (12, 25, 297),
        (12, 30, 361),
        (15, 24, 361),
        (15, 25, 377),
        (15, 30, 449),
    ] {
        let frames = ltx_snapped_frame_count(duration * fps);
        assert_eq!(frames, expected, "{duration}s at {fps}fps");
        assert_eq!(
            frames % LTX_TEMPORAL_SCALE,
            1,
            "every reachable frame count is on the 1 + 8k lattice"
        );
    }
    // The ladder's own floor, independent of the declared durations.
    assert_eq!(ltx_snapped_frame_count(0), 9);
    assert_eq!(ltx_snapped_frame_count(1), 9);
}

/// The envelope is DERIVED from the declared arrays through the ladder above, not written down.
#[test]
fn ltx_frame_envelope_is_derived_from_the_declared_durations_and_fps() {
    assert_eq!(LTX_FRAME_ENVELOPE, (97, 449));
    let (minimum, maximum) = LTX_FRAME_ENVELOPE;
    assert_eq!(minimum, ltx_snapped_frame_count(4 * 24));
    assert_eq!(maximum, ltx_snapped_frame_count(15 * 30));
}

#[test]
fn the_ltx_arm_accepts_the_declared_video_envelope() {
    for (width, height) in LTX_RESOLUTIONS {
        for frames in [LTX_FRAME_ENVELOPE.0, 241, LTX_FRAME_ENVELOPE.1] {
            let geometry = validate_ltx_geometry(width, height, frames)
                .unwrap_or_else(|error| panic!("{width}x{height} f{frames}: {error}"));
            assert_eq!(geometry.frames, frames);
            assert_eq!(
                geometry.latent_frames,
                1 + (frames - 1) / LTX_TEMPORAL_SCALE
            );
            assert!(
                geometry.latent_frames > 1,
                "a video record is multi-latent-frame"
            );
        }
    }
}

/// The negative half of the envelope, mirroring the shape of the pinned image-arm negative test.
#[test]
fn the_ltx_arm_rejects_out_of_envelope_geometry_with_a_named_reason() {
    for (width, height, frames, expected) in [
        // An undeclared resolution, including one that is 64-aligned and would otherwise render.
        (1024, 1024, 97, "declared limits.resolutions"),
        (800, 512, 97, "declared limits.resolutions"),
        // Off the temporal lattice the LTX VAE's 8x causal compression requires.
        (768, 512, 96, "1 + 8k"),
        (768, 512, 100, "1 + 8k"),
        // On the lattice but outside what a declared duration/fps pair can produce.
        (768, 512, 1, "duration/fps envelope"),
        (768, 512, 9, "duration/fps envelope"),
        (768, 512, 457, "duration/fps envelope"),
    ] {
        let error = validate_ltx_geometry(width, height, frames)
            .expect_err("an out-of-envelope LTX geometry must be refused");
        assert!(
            error.contains(expected),
            "{width}x{height} f{frames}: {error}"
        );
    }
}

/// A still geometry is on the temporal lattice (`1 % 8 == 1`), so it can only be caught by the
/// envelope floor. The video arm must not be able to capture a single-frame record.
#[test]
fn the_ltx_arm_refuses_a_single_frame_capture() {
    let error =
        run_ltx(&ltx_request_json(768, 512, 1)).expect_err("a video arm must not capture a still");
    assert!(error.contains("duration/fps envelope"), "{error}");
}

/// The three `mlx:ltx_2_3` identities are ONE family minted off the engine's retained key: the
/// calibrated tier IS that key byte-for-byte, and each other shipped tier is the same stem and
/// version with its own tier token. Every member is compared with what the ENGINE's own
/// weights-free minting function (`production_calibration_fingerprint`) returns for that
/// artifact tier, so a re-spelled format on either side reds here rather than in a capture; and
/// the calibrated tier is the engine's `CALIBRATED_TIER`, not a local spelling.
#[test]
fn the_ltx_identity_family_is_the_engines_retained_key_with_a_tier_token() {
    let retained = mlx_gen_ltx::memory_strategy::CALIBRATION_FINGERPRINT;
    assert_eq!(
        LTX_CALIBRATED_TIER,
        mlx_gen_ltx::memory_strategy::CALIBRATED_TIER,
        "the calibrated tier is read off the engine, never spelled"
    );
    assert!(
        LTX_TIERS.contains(&LTX_CALIBRATED_TIER),
        "the engine's calibrated tier {LTX_CALIBRATED_TIER:?} must be a shipped tier"
    );
    assert_eq!(
        ltx_calibration_fingerprint(LTX_CALIBRATED_TIER).unwrap(),
        retained,
        "the calibrated cell must reproduce the engine's retained key byte-for-byte"
    );
    let (stem, version) = retained
        .rsplit_once("-mlx-memory-ladder-")
        .unwrap_or_else(|| panic!("{retained} is not an LTX memory-ladder key"));
    assert!(
        !stem.ends_with(&format!("-{LTX_CALIBRATED_TIER}")),
        "the retained key carries no tier token, so it is the family's bare member: {retained}"
    );
    let mut identities = std::collections::BTreeSet::new();
    for tier in LTX_TIERS {
        let minted = ltx_calibration_fingerprint(tier).unwrap();
        let quant = match tier {
            "q4" => Some(Quant::Q4),
            "q8" => Some(Quant::Q8),
            _ => None,
        };
        let engine =
            mlx_gen_ltx::memory_strategy::production_calibration_fingerprint(&MemoryNumericTier {
                precision: Precision::Bf16,
                quant,
                component_precision_floors: &[],
            })
            .unwrap_or_else(|| panic!("the engine mints no identity for shipped tier {tier}"));
        assert_eq!(
            minted, engine,
            "the {tier} cell must be byte-for-byte what the pinned engine mints for that tier"
        );
        if tier != LTX_CALIBRATED_TIER {
            assert_eq!(
                minted,
                format!("{stem}-{tier}-mlx-memory-ladder-{version}"),
                "the {tier} cell must be the engine's key carrying its own tier token"
            );
        }
        assert!(
            identities.insert(minted),
            "{tier} reuses another tier's identity"
        );
    }
    assert_eq!(identities.len(), LTX_TIERS.len());
}

/// The shipped plan names, for every `mlx:ltx_2_3` row, exactly the identity this arm mints
/// for that row's tier — and the arm mints it off the engine's `CALIBRATED_TIER`. This is the
/// binding that reds when the ENGINE re-tiers its retained key: the family above follows the
/// engine, the plan does not, and a capture would otherwise be the first place to notice.
#[test]
fn every_planned_ltx_mlx_row_names_the_identity_this_arm_mints_for_its_tier() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan is valid JSON");
    let mut planned_tiers = std::collections::BTreeSet::new();
    for (key, row) in plan["anchors"].as_object().expect("anchors is an object") {
        let parts: Vec<&str> = key.split(':').collect();
        if parts.len() != 3 || parts[0] != LTX_PROVIDER || parts[2] != "mlx" {
            continue;
        }
        let tier = parts[1];
        assert_eq!(row["provider"].as_str(), Some(LTX_PROVIDER), "{key}");
        assert_eq!(
            row["calibrationFingerprint"]
                .as_str()
                .expect("fingerprint is a string"),
            ltx_calibration_fingerprint(tier).unwrap(),
            "{key} names an identity the pinned provider does not mint for tier {tier}"
        );
        assert!(
            planned_tiers.insert(tier.to_owned()),
            "{key} is planned twice"
        );
    }
    assert_eq!(
        planned_tiers,
        LTX_TIERS.iter().map(|tier| (*tier).to_owned()).collect(),
        "every shipped tier has exactly one planned MLX row"
    );
}

/// A tier the family does not ship has no identity and is refused BY NAME, rather than
/// interpolated into a plausible key no production load can return.
#[test]
fn a_tier_the_ltx_family_does_not_ship_has_no_identity() {
    for tier in ["fp8", "q6", "", "Q4"] {
        let error = ltx_calibration_fingerprint(tier)
            .expect_err("an unshipped tier must not mint an identity");
        assert!(
            error.contains(&format!("{tier:?}")) && error.contains("no production identity"),
            "{error}"
        );
    }
}

/// The plan's tier chooses the identity: a q8 plan naming the q4 cell's key is refused before
/// any environment or weight work, naming both strings and the tier.
#[test]
fn a_plan_naming_another_tiers_ltx_identity_is_refused_before_the_load() {
    let mut request = ltx_request_json(768, 512, 97);
    let foreign = ltx_calibration_fingerprint("q4").unwrap();
    request["planned"]["calibrationFingerprint"] = json!(foreign);
    let error = run_ltx(&request).expect_err("a foreign tier identity is not capturable");
    assert!(
        error.contains("plan/adapter calibration mismatch")
            && error.contains(&foreign)
            && error.contains(&ltx_calibration_fingerprint("q8").unwrap())
            && error.contains("tier q8"),
        "{error}"
    );
    // Not an env error: the refusal fired before `SCENEWORKS_LTX_*` was read.
    assert!(!error.contains("required environment variable"), "{error}");
}

#[test]
fn the_ltx_arm_refuses_a_foreign_provider_before_environment_or_weight_work() {
    for provider in ["ltx_2_3_distilled", "ltx_2_3_eros", "wan_2_2", "flux2_dev"] {
        let mut request = ltx_request_json(768, 512, 97);
        request["planned"]["target"]["provider"] = json!(provider);
        let error = run_ltx(&request).expect_err("a foreign provider must not reach the LTX arm");
        assert_eq!(
            error,
            format!("MLX LTX-2.3 calibration does not implement provider {provider:?}")
        );
    }
}

#[test]
fn the_ltx_arm_fails_closed_on_a_non_t2v_target_before_weight_work() {
    for (pointer, value, expected) in [
        (
            "/planned/target/modelId",
            json!("ltx_2_3_eros"),
            "requires modelId",
        ),
        (
            "/planned/target/mode",
            json!("image_to_video"),
            "requires reference-free text_to_video mode",
        ),
        (
            "/planned/target/geometry/batch",
            json!(2),
            "requires geometry.batch == 1",
        ),
        (
            "/planned/target/overlay",
            json!("lora"),
            "refusing rather than recording false overlay coverage",
        ),
        // `resident` is unreachable on this provider on EVERY host, so the refusal is
        // host-independent — but WHICH rung the engine measures instead is not: this geometry
        // is single-pass on a 128 GiB Mac and tiled on a small CI runner. Assert the
        // machine-independent half of the message; the rung itself is pinned, at a fixed
        // budget, by `the_ltx_arm_follows_the_engine_across_the_decode_tiling_boundary`.
        (
            "/planned/loadShape",
            json!("deferred_materialization"),
            "calibrates only eager_materialization",
        ),
        (
            "/planned/calibrationFingerprint",
            json!("sc-18808-ltx-2-3-mlx-t2v-staged-capture-v0"),
            "plan/adapter calibration mismatch",
        ),
        (
            "/planned/fixture",
            json!("ltx-2-3-mlx-q8-768x512-f97-fps60-seed18808"),
            "not one of the declared limits.fps",
        ),
        (
            "/planned/fixture",
            json!("ltx-2-3-mlx-q8-768x512-f193-fps24-seed18808"),
            "must start with",
        ),
        (
            "/planned/fixture",
            json!("ltx-2-3-mlx-q8-768x512-f97-fps24-seed42"),
            "does not match the LTX-2.3 calibration seed",
        ),
        (
            "/planned/target/tier",
            json!("q2"),
            "unsupported MLX numeric tier",
        ),
    ] {
        let mut request = ltx_request_json(768, 512, 97);
        *request.pointer_mut(pointer).unwrap() = value.clone();
        let error = run_ltx(&request)
            .expect_err("a non-T2V LTX target must fail before environment or weights");
        assert!(error.contains(expected), "{pointer}={value}: {error}");
    }
}

/// Pure request malformations must still fail before environment or weight resolution.
#[test]
fn the_ltx_arm_refuses_a_malformed_plan_before_it_consults_the_host_budget() {
    // Force the smoke geometry onto the provider's auto-tiled path. Every mutation below must
    // still report its own deterministic input error; consulting the live selector first made
    // this test intermittently report a decode-budget outcome under default-parallel tests.
    let _constrained_budget = LtxInjectedBudget::install(8.0);
    for (pointer, value, expected) in [
        (
            "/planned/fixture",
            json!("ltx-2-3-mlx-q8-768x512-f97-fps24-seed42"),
            "does not match the LTX-2.3 calibration seed",
        ),
        (
            "/planned/calibrationFingerprint",
            json!("sc-18808-ltx-2-3-mlx-t2v-staged-capture-v0"),
            "plan/adapter calibration mismatch",
        ),
        (
            "/planned/loadShape",
            json!("deferred_materialization"),
            "calibrates only eager_materialization",
        ),
        (
            "/planned/target/tier",
            json!("q2"),
            "unsupported MLX numeric tier",
        ),
    ] {
        let mut request = ltx_request_json(768, 512, 97);
        *request.pointer_mut(pointer).unwrap() = value.clone();
        let error = run_ltx(&request).expect_err("a malformed LTX plan must be refused");
        assert!(
            error.contains(expected),
            "{pointer}={value} must be refused for the malformation: {error}"
        );
    }
}

#[test]
fn the_ltx_arm_refuses_a_parameterized_staged_residency_row() {
    let mut request = ltx_request_json(768, 512, 97);
    request["planned"]["strategy"]["parameters"] = json!({ "decodeTileEdge": 512 });
    let selection = planned_selection(&request).unwrap();
    let error = ltx_fixture_contract(Some(Quant::Q8))
        .validate_selection(&selection)
        .expect_err("staged residency has no decode parameter domain");
    assert!(error.to_string().contains("decode"), "{error}");
}

const LTX_TEST_HOST_BYTES: u64 = 128 * 1024 * 1024 * 1024;

/// The request the harness's `planAnchor` would send for one plan row: the row's own fields,
/// the lane's default composition where the row carries no `strategy` override (read from the
/// same `config/anchor-lane-default-strategy.json` the harness and the worker read), and the
/// probed host memory.
fn ltx_plan_row_request(key: &str, row: &Value) -> Value {
    let lane_defaults: Value = serde_json::from_str(include_str!(
        "../../../../config/anchor-lane-default-strategy.json"
    ))
    .expect("the lane default strategy declaration is valid JSON");
    let parts: Vec<&str> = key.split(':').collect();
    let strategy = row
        .get("strategy")
        .cloned()
        .unwrap_or_else(|| lane_defaults["lanes"]["mlx"].clone());
    json!({
        "hardware": { "memoryBytes": LTX_TEST_HOST_BYTES },
        "planned": {
            "target": {
                "provider": row["provider"],
                "modelId": parts[0],
                "tier": parts[1],
                "mode": row["mode"],
                "overlay": row["overlay"],
                "geometry": row["geometry"],
            },
            "backend": "mlx",
            "loadShape": row["loadShape"],
            "strategy": {
                "rung": strategy["rung"],
                "engagedRungs": strategy["engagedRungs"],
                "parameters": {},
            },
            "calibrationFingerprint": row["calibrationFingerprint"],
            "fixture": row["fixture"],
        }
    })
}

fn ltx_tier_quant(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        other => panic!("unplanned LTX tier {other:?}"),
    }
}

/// An ordinary `run` of this request passes every plan check and reaches the first
/// environment read that precedes the load — the proof that no arm-local refusal stands in
/// its way. Refuses to run with a real LTX root configured in the process, because past the
/// environment read the next thing is a weights load.
fn assert_ordinary_ltx_run_reaches_the_environment(request: &Value) {
    for name in ["SCENEWORKS_LTX_REPOSITORY", "SCENEWORKS_LTX_ROOT"] {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} is set in this test process; this assertion proves the plan reaches the \
                 environment read WITHOUT loading weights, so it refuses to run with a real root \
                 configured"
        );
    }
    let error = run_ltx(request).expect_err("no LTX root is configured in this process");
    assert!(
        error.contains("required environment variable SCENEWORKS_LTX_REPOSITORY"),
        "the ordinary run was refused before the environment read: {error}"
    );
    assert!(
        !error.contains("safety_refused_open") && !error.contains("SC-19642"),
        "an SC-19642 disposition refusal is back: {error}"
    );
}

/// sc-22738. Every planned `ltx_2_3:*:mlx` row — exactly as the harness sends it — passes
/// the arm's whole request validation and reaches the environment read that precedes the
/// load, and its planned composition is one the pinned contract admits at the production
/// budget. SC-19642 used to stop every one of these rows at a plan-supplied block the plan
/// schema cannot carry, before any of this was asked.
#[test]
fn every_planned_ltx_mlx_row_passes_the_arms_request_validation_up_to_the_load() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan is valid JSON");
    let mut rows = 0;
    for (key, row) in plan["anchors"].as_object().expect("anchors is an object") {
        let parts: Vec<&str> = key.split(':').collect();
        if parts.len() != 3 || parts[0] != LTX_PROVIDER || parts[2] != "mlx" {
            continue;
        }
        rows += 1;
        let request = ltx_plan_row_request(key, row);
        // Everything before the load, in the arm's own order: target, load shape, selection,
        // tier, fixture, identity, host budget — then the first environment read.
        assert_ordinary_ltx_run_reaches_the_environment(&request);
        // And the planned composition is admissible by the pinned contract at the production
        // budget, so the load is the next thing that happens on a host with the weights.
        let tier = parts[1];
        let geometry =
            ltx_target_geometry(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        let selection =
            planned_selection(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        let contract = ltx_fixture_contract(ltx_tier_quant(tier));
        contract
            .validate_selection(&selection)
            .unwrap_or_else(|error| {
                panic!("{key}: the pinned contract refuses the planned composition: {error}")
            });
        let calibration = contract.calibration.as_ref().expect("fixture calibration");
        let (context, projection) = ltx_ordinary_admission(
            &contract,
            calibration,
            selection,
            geometry,
            LTX_TEST_HOST_BYTES,
        )
        .unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(
            context.predicted_peak_bytes, projection.predicted_peak_bytes,
            "{key}"
        );
        assert_eq!(context.budget.total_bytes, LTX_TEST_HOST_BYTES, "{key}");
        // The budget carries the worker's fixed unified-memory reserve
        // (`live_request_budget`, non-Mage branch: `legacy_unified_reserve` = 2 GiB), read
        // from the shared declaration — `effective_bytes` subtracts it, so a zero here is a
        // budget 2,147,483,648 bytes more permissive than the worker's (sc-22738 review).
        assert_eq!(
            context.budget.reserved_headroom_bytes,
            2 * 1024 * 1024 * 1024,
            "{key}"
        );
        assert_eq!(
            context.budget.reserved_headroom_bytes,
            sceneworks_core::memory_anchor::gib_to_bytes(
                sceneworks_core::memory_anchor::LEGACY_UNIFIED_FALLBACK_RESERVE_GB
            ),
            "{key}"
        );
        assert!(!context.has_phases, "{key}");
    }
    assert_eq!(
        rows,
        LTX_TIERS.len(),
        "every shipped tier has one planned MLX row"
    );
}

/// sc-22738. The post-render exact-fit probe is posed in the currency the gate spends.
///
/// The regression this pins is the one that killed every `ltx_2_3:*:mlx` anchor after a full
/// load: with the worker's real presentation carrying a 2 GiB reserve, a budget whose
/// NAMEPLATE total equals the predicted peak has an EFFECTIVE budget a whole reserve below it,
/// so `standard_memory_strategy_safety_check`'s `budget.fits` clause refuses it — by design.
/// The first assertion below is that refusal, stated as a fact about the old shape rather than
/// left implicit; the second is that [`ltx_exact_fit_budget`] restores the boundary the
/// scenario is defined at, `effective_bytes() == predicted`, and is admitted there.
/// Reverting the helper to `total_bytes = predicted` reds the second on every shipped tier.
#[test]
fn the_exact_fit_probe_presents_the_boundary_host_not_a_nameplate_total() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan is valid JSON");
    let mut rows = 0;
    for (key, row) in plan["anchors"].as_object().expect("anchors is an object") {
        let parts: Vec<&str> = key.split(':').collect();
        if parts.len() != 3 || parts[0] != LTX_PROVIDER || parts[2] != "mlx" {
            continue;
        }
        rows += 1;
        let request = ltx_plan_row_request(key, row);
        let geometry =
            ltx_target_geometry(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        let selection =
            planned_selection(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        let contract = ltx_fixture_contract(ltx_tier_quant(parts[1]));
        let calibration = contract.calibration.as_ref().expect("fixture calibration");
        let (context, projection) = ltx_ordinary_admission(
            &contract,
            calibration,
            selection,
            geometry,
            LTX_TEST_HOST_BYTES,
        )
        .unwrap_or_else(|error| panic!("{key}: {error}"));
        // Stands in for the measured overall peak the live arm probes with.
        let predicted = projection.predicted_peak_bytes;

        let mut nameplate = context.clone();
        nameplate.predicted_peak_bytes = predicted;
        nameplate.budget.total_bytes = predicted;
        assert!(
            nameplate.budget.effective_bytes() < predicted,
            "{key}: a nameplate-total budget must sit below the need once production's \
                 reserve is presented"
        );
        assert!(
            matches!(
                mlx_gen::gen_core::default_memory_strategy_safety_check(&contract, &nameplate),
                MemorySafetyDecision::Reject { .. }
            ),
            "{key}: the pre-sc-22738 exact-fit shape must still be refused; if it is admitted \
                 the reserve has silently left the production presentation"
        );

        let mut exact = context.clone();
        exact.predicted_peak_bytes = predicted;
        exact.budget = ltx_exact_fit_budget(context.budget, predicted);
        assert_eq!(
            exact.budget.effective_bytes(),
            predicted,
            "{key}: the exact-fit budget must spend exactly the predicted peak"
        );
        assert_eq!(
            exact.budget.reserved_headroom_bytes, context.budget.reserved_headroom_bytes,
            "{key}: the probe must carry production's reserve through untouched"
        );
        assert_eq!(
            exact.budget.committed_bytes, context.budget.committed_bytes,
            "{key}: the probe must carry the live committed snapshot through untouched"
        );
        assert!(
            matches!(
                mlx_gen::gen_core::default_memory_strategy_safety_check(&contract, &exact),
                MemorySafetyDecision::Accept
            ),
            "{key}: the boundary host must be admitted"
        );
    }
    assert_eq!(
        rows,
        LTX_TIERS.len(),
        "every shipped tier has one planned MLX row"
    );
}

/// sc-22738. The ordinary `run` path has NO arm-local refusal between the plan checks and the
/// load: the admission match's `Ordinary` arm is empty, the production admission is what
/// stands before the load, and the post-load lifecycle arm executes rather than returning.
/// Reintroducing an `Err` in either arm, or dropping the admission call, reds this.
#[test]
fn the_ordinary_ltx_path_has_no_unconditional_refusal_between_the_plan_and_the_load() {
    let source = crate::mlx_source_for_tests();
    // Assembled so this test's own literal cannot satisfy the search.
    let refusal = format!("fn {}(", "refuse_unsafe_ltx_capture");
    assert!(
        !source.contains(&refusal),
        "the SC-19642 unconditional refusal is back"
    );
    let body = &source[source.find("fn run_ltx_with_admission(").unwrap()..];
    let load_at = body
        .find(".load(LTX_PROVIDER, &spec)")
        .expect("the arm loads the provider");
    let pre_load = &body[..load_at];
    let arm_at = pre_load
        .find("LtxRunAdmission::Ordinary =>")
        .expect("the admission match has an Ordinary arm");
    let arm = &pre_load[arm_at..];
    let arm = &arm[..arm
        .find("LtxRunAdmission::CampaignEntry =>")
        .expect("the CampaignEntry arm follows")];
    assert!(
        !arm.contains("Err(") && !arm.contains("return") && !arm.contains('?'),
        "the Ordinary admission arm refuses or exits: {arm}"
    );
    let admission_at = pre_load
        .find("ltx_ordinary_admission(")
        .expect("the production admission is called before the load");
    assert!(
        admission_at > arm_at,
        "the production admission must stand between the plan checks and the load"
    );
    let post_load = &body[load_at..];
    let lifecycle = &post_load[..post_load
        .find("// Runtime-complete keeps the schema's unexecuted mutation slot null.")
        .expect("the lifecycle match precedes the falsifiability mutation")];
    let ordinary = &lifecycle[lifecycle
        .rfind("LtxRunAdmission::Ordinary =>")
        .expect("the lifecycle match has an Ordinary arm")..];
    assert!(
        !ordinary.contains("Err(") && !ordinary.contains("return"),
        "the Ordinary lifecycle arm refuses or exits: {ordinary}"
    );
    // sc-22738: the ordinary capture is ONE measured render on the video lane. Its lifecycle
    // arm renders nothing — no clean warm control, no fault, no recovery — and files the
    // `not_run` receipt instead; only the SC-20318 campaign entries execute a lifecycle.
    assert!(
        !ordinary.contains("verify_ltx_lifecycle(")
            && !ordinary.contains("verify_ltx_bounded_warm_repeat("),
        "the Ordinary lifecycle arm must not render a warm pass: {ordinary}"
    );
    assert!(
        ordinary.contains("(None, None)"),
        "the Ordinary lifecycle arm files no lifecycle metrics: {ordinary}"
    );
    let campaign = &lifecycle[lifecycle
        .find("LtxRunAdmission::CampaignEntry =>")
        .expect("the lifecycle match has a CampaignEntry arm")..];
    assert!(
        campaign.contains("verify_ltx_lifecycle("),
        "the campaign entry keeps its runtime-complete lifecycle proof"
    );
}

/// sc-22738. A request whose projection exceeds the host budget is refused BY THE BUDGET —
/// gen-core's shared predicate, in its own words, carrying the engine's own decode figure —
/// and the same request fits a real host. Bypassing the admission, or replacing the engine's
/// profile with a number of this arm's own, reds this.
#[test]
fn an_ltx_request_beyond_the_host_budget_is_refused_by_the_budget_in_its_own_words() {
    let request = ltx_plan_row_request(
        "ltx_2_3:q8:mlx",
        &json!({
            "provider": LTX_PROVIDER,
            "mode": "text_to_video",
            "overlay": "none",
            "geometry": { "width": 768, "height": 512, "batch": 1, "frames": 121 },
            "loadShape": "eager_materialization",
            "calibrationFingerprint": ltx_calibration_fingerprint("q8").unwrap(),
            "fixture": format!("ltx-2-3-mlx-q8-768x512-f121-fps30-seed{LTX_SEED}"),
        }),
    );
    let geometry = ltx_target_geometry(&request).unwrap();
    let selection = planned_selection(&request).unwrap();
    let contract = ltx_fixture_contract(Some(Quant::Q8));
    let calibration = contract.calibration.as_ref().unwrap();
    // The weights-free fixture declares zero asset facts, so the projection IS the engine's
    // conservative decode working set for this geometry — the figure the worker's
    // `packaged_video_decode_profile` resolves for a non-bounded MLX candidate.
    let engine_decode =
        mlx_gen_ltx::conservative_video_decode_memory_profile(LTX_PROVIDER, 768, 512, 121)
            .expect("the pinned provider profiles the anchor geometry")
            .working_set_bytes();
    let small_host = 8 * 1024 * 1024 * 1024;
    let error = ltx_ordinary_admission(&contract, calibration, selection, geometry, small_host)
        .expect_err("a projection above the host budget must be refused");
    assert!(
        error.starts_with("LTX-2.3 admission refused by the production budget before load: "),
        "{error}"
    );
    assert!(
        error.contains(&format!(
            "incremental live demand {engine_decode} exceeds effective budget"
        )),
        "the refusal must be the shared predicate's own, carrying the engine's figure: {error}"
    );
    assert!(
        error.contains(&format!("hardware.memoryBytes {small_host}")),
        "{error}"
    );
    let (context, projection) = ltx_ordinary_admission(
        &contract,
        calibration,
        selection,
        geometry,
        LTX_TEST_HOST_BYTES,
    )
    .expect("the same request fits a 128 GiB host");
    assert_eq!(projection.decode_working_set_bytes, engine_decode);
    assert_eq!(projection.predicted_peak_bytes, engine_decode);
    assert_eq!(projection.weights_bytes, 0, "weights-free fixture");
    assert_eq!(context.predicted_peak_bytes, engine_decode);
    assert!(context.budget.fits(engine_decode));
}

/// The weights term IS the worker's floor for the two compositions an LTX anchor can plan —
/// the shared `sceneworks_core::memory_anchor::floor_weights_bytes` over the same lift of the
/// contract: resident sums the stacks, staged residency takes the larger of the conditioning
/// stack and the post-eviction heavy stack.
#[test]
fn the_ltx_admission_weights_floor_follows_the_workers_composition_rule() {
    let contract = ltx_fixture_contract(Some(Quant::Q8));
    let resident = contract.engaged_composition(MemoryStrategy::Resident);
    let staged = contract.engaged_composition(MemoryStrategy::StagedResidency);
    assert_eq!(ltx_floor_weights_bytes(&contract, &resident).unwrap(), 0);
    assert_eq!(ltx_floor_weights_bytes(&contract, &staged).unwrap(), 0);
    let mut declared = contract.clone();
    declared.asset_facts.conditioning_bytes = 26;
    declared.asset_facts.transformer_bytes = 20;
    declared.asset_facts.decoder_bytes = 4;
    declared.asset_facts.base_bytes = 50;
    assert_eq!(ltx_floor_weights_bytes(&declared, &resident).unwrap(), 50);
    // Staged: max(conditioning 26, heavy 24 with nothing evicted below the transformer) = 26.
    assert_eq!(ltx_floor_weights_bytes(&declared, &staged).unwrap(), 26);
    // And the lift agrees with the shared function called directly with the same facts.
    assert_eq!(
        ltx_floor_weights_bytes(&declared, &staged).unwrap(),
        sceneworks_core::memory_anchor::floor_weights_bytes(
            sceneworks_core::memory_anchor::FloorWeightsFacts {
                conditioning_bytes: 26,
                base_bytes: 50,
                transformer_bytes: 20,
                intra_transformer_evicted_bytes: 0,
                auxiliary_resident_bytes: 0,
            },
            sceneworks_core::memory_anchor::FloorWeightsComposition {
                staged: true,
                bounded_transformer: false,
                resident_window_bytes: 0,
            },
        )
    );
}

/// sc-22738 review. The pinned provider declares rung 4 UNIMPLEMENTED (`support: Missing`);
/// a composition engaging `bounded_transformer_residency` is refused EXPLICITLY, not priced
/// by an arm that carries no window for it and not `debug_assert`ed away in release.
#[test]
fn a_rung_4_ltx_composition_is_refused_explicitly_rather_than_priced() {
    let contract = ltx_fixture_contract(Some(Quant::Q8));
    assert!(
        matches!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .expect("every rung is declared")
                .support,
            mlx_gen::gen_core::MemoryStrategySupport::Missing
        ),
        "the pinned LTX-2.3 provider declares bounded_transformer_residency unimplemented"
    );
    let rung4 = [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedTransformerResidency,
    ];
    let error = ltx_floor_weights_bytes(&contract, &rung4)
        .expect_err("a rung-4 composition must be refused");
    assert!(
        error.contains("bounded_transformer_residency")
            && error.contains("declares rung 4 unimplemented"),
        "{error}"
    );
    let resident = contract.engaged_composition(MemoryStrategy::Resident);
    assert!(ltx_floor_weights_bytes(&contract, &resident).is_ok());
}

/// sc-22738 review. The projection is the worker's `profiled_floor_phase_peaks`, not the
/// composed profile alone: `max(weights + 18 GiB generic headroom, composed)`. Over every
/// shipped frame count (the declared durations × fps through the shipped ladder, at the
/// planned 768×512) the identity holds with the headroom read from the shared declaration —
/// and it is LOAD-BEARING below f121: at f97 the engine's decode term is under the generic
/// allowance, so dropping the floor under-predicts by exactly the gap.
#[test]
fn the_ltx_projection_never_falls_below_the_workers_generic_floor_at_any_shipped_frame_count() {
    let headroom = ltx_generic_headroom_bytes();
    assert_eq!(
        headroom, 19_327_352_832,
        "HeadroomAllowance::GENERIC, 18 GiB"
    );
    let mut contract = ltx_fixture_contract(Some(Quant::Q8));
    // A declared weights term so the identity is exercised with a non-zero floor too.
    contract.asset_facts.base_bytes = LTX_Q8_INVENTORY_BYTES;
    contract.asset_facts.transformer_bytes = LTX_Q8_INVENTORY_BYTES;
    let weights_free = ltx_fixture_contract(Some(Quant::Q8));
    let mut floor_bound = Vec::new();
    let mut profile_bound = Vec::new();
    for duration in LTX_DURATIONS_SECONDS {
        for fps in LTX_FPS {
            let frames = ltx_snapped_frame_count(duration * fps);
            let profile = mlx_gen_ltx::conservative_video_decode_memory_profile(
                LTX_PROVIDER,
                768,
                512,
                frames,
            )
            .unwrap_or_else(|| panic!("the pinned provider profiles 768x512 f{frames}"));
            let decode = profile.working_set_bytes();
            for (label, contract) in [("declared", &contract), ("weights-free", &weights_free)] {
                let engaged = contract.engaged_composition(MemoryStrategy::Resident);
                let weights = ltx_floor_weights_bytes(contract, &engaged).unwrap();
                let composed = profile
                    .checked_composed_peak(weights, contract.asset_facts.decoder_bytes)
                    .unwrap();
                let predicted = ltx_projected_peak_bytes(contract, weights, profile).unwrap();
                assert_eq!(
                    predicted,
                    composed.max(weights + headroom),
                    "{label} f{frames}: predicted = max(weights + 18 GiB, composed)"
                );
                assert!(predicted >= weights + headroom, "{label} f{frames}");
                assert!(predicted >= composed, "{label} f{frames}");
            }
            if decode < headroom {
                floor_bound.push(frames);
            } else {
                profile_bound.push(frames);
            }
        }
    }
    // The decode term grows with frames, so the generic floor binds exactly the short end of
    // the ladder: f97 (the SC-18808 calibration geometry) is floor-bound, f121 (the planned
    // anchor geometry) and everything longer is profile-bound.
    assert_eq!(floor_bound, vec![97, 97], "4s at 24/25 fps");
    assert!(profile_bound.contains(&121) && !floor_bound.contains(&121));
    let f97 = mlx_gen_ltx::conservative_video_decode_memory_profile(LTX_PROVIDER, 768, 512, 97)
        .unwrap()
        .working_set_bytes();
    let f121 = mlx_gen_ltx::conservative_video_decode_memory_profile(LTX_PROVIDER, 768, 512, 121)
        .unwrap()
        .working_set_bytes();
    assert_eq!(
        f121, 19_476_906_240,
        "the planned anchor's engine decode term"
    );
    assert_eq!(
        headroom - f97,
        3_059_089_152,
        "the under-prediction the composed-only projection carried at f97"
    );
}

/// sc-22738 review. What the three planned cells PROJECT, derived from checked-in
/// declarations only — each tier's immutable inventory (`LTX_*_INVENTORY_BYTES`, the SC-18946
/// tier roots) as the resident weights, the engine's decode profile at the planned 768×512 f121
/// — pinned against the incident footprint. Derivation: weights = inventory (the resident
/// composition sums conditioning + heavy = `base_bytes`, and the weights-free fixture declares
/// no decoder to substitute), composed = inventory + decode term, predicted =
/// max(inventory + 18 GiB, composed) = composed here because the f121 decode term exceeds the
/// generic allowance; budget = 128 GiB host − 2 GiB reserve with nothing committed. These are
/// NOT machine measurements. The production contract at capture adds its projected
/// `conditioning_bytes` (the Gemma co-requisite root, which no checked-in constant declares)
/// on top of every figure here, and the record's `admissionPredictedPeakBytes` carries that.
#[test]
fn the_planned_ltx_tiers_project_below_the_incident_footprint_from_declared_inventories_alone() {
    let decode = 19_476_906_240_u64;
    assert_eq!(
        mlx_gen_ltx::conservative_video_decode_memory_profile(LTX_PROVIDER, 768, 512, 121)
            .unwrap()
            .working_set_bytes(),
        decode
    );
    let budget = MemoryBudget {
        total_bytes: LTX_TEST_HOST_BYTES,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: ltx_unified_reserve_bytes(),
    };
    assert_eq!(budget.effective_bytes(), 135_291_469_824, "128 GiB − 2 GiB");
    for (tier, inventory, composed, headroom_vs_incident) in [
        (
            "q4",
            LTX_Q4_INVENTORY_BYTES,
            39_944_596_700_u64,
            57_025_487_780_u64,
        ),
        ("q8", LTX_Q8_INVENTORY_BYTES, 49_205_626_956, 47_764_457_524),
        (
            "bf16",
            LTX_BF16_INVENTORY_BYTES,
            66_569_718_232,
            30_400_366_248,
        ),
    ] {
        let mut contract = ltx_fixture_contract(ltx_tier_quant(tier));
        contract.asset_facts.base_bytes = inventory;
        contract.asset_facts.transformer_bytes = inventory;
        let engaged = contract.engaged_composition(MemoryStrategy::Resident);
        let weights = ltx_floor_weights_bytes(&contract, &engaged).unwrap();
        assert_eq!(weights, inventory, "{tier}: resident floor = the inventory");
        let profile =
            mlx_gen_ltx::conservative_video_decode_memory_profile(LTX_PROVIDER, 768, 512, 121)
                .unwrap();
        let predicted = ltx_projected_peak_bytes(&contract, weights, profile).unwrap();
        assert_eq!(predicted, composed, "{tier}: composed = inventory + decode");
        assert_eq!(predicted, inventory + decode, "{tier}");
        assert!(
            predicted > weights + ltx_generic_headroom_bytes(),
            "{tier}: the f121 decode term, not the generic floor, binds"
        );
        assert_eq!(
            LTX_Q4_F305_CRASH_FOOTPRINT_BYTES - predicted,
            headroom_vs_incident,
            "{tier}: headroom below the SC-18946 footprint"
        );
        assert!(budget.fits(predicted), "{tier} fits a 128 GiB host");
    }
}

/// The exact composition the record claims, pinned so a silent widening of `engagedRungs` would
/// have to be an explicit edit here.
#[test]
fn the_ltx_arm_attests_exactly_resident_plus_staged_residency() {
    let request = ltx_request_json(768, 512, 97);
    let selection = planned_selection(&request).unwrap();
    let contract = ltx_fixture_contract(Some(Quant::Q8));
    contract.validate_selection(&selection).unwrap();
    let strategy = ltx_attested_strategy(&request, &selection, &contract).unwrap();
    assert_eq!(strategy["rung"], "staged_residency");
    assert_eq!(
        strategy["engagedRungs"],
        json!(["resident", "staged_residency"])
    );
    assert_eq!(strategy["parameters"], json!({}));
}

/// sc-18810: the staged-residency witness must be a PHASE BOUNDARY reading, not a peak. The
/// numbers here are the real ones from this story's captures — the first row is the geometry
/// that broke sc-18808's `overall_peak < text_encoder + transformer` form.
#[test]
fn the_staging_proof_survives_a_peak_above_the_costaged_bound() {
    const TEXT_ENCODER: u64 = 32_733_043_614;
    const TRANSFORMER: u64 = 20_614_103_249;
    const COSTAGED: u64 = TEXT_ENCODER + TRANSFORMER; // 53,347,146,863
    let entering_denoise = |active: u64| AllocatorState { active, cache: 0 };

    // q8 704x1280 x 177: measured overall peak 54,153,098,156 — ABOVE the co-staged bound — while
    // the denoise boundary held 20.7 GB, i.e. the transformer and nothing else. The peak is the
    // decode phase's output buffers. This run is valid and must be accepted.
    assert!(
        ltx_staging_is_proven(
            entering_denoise(20_765_156_098),
            COSTAGED,
            TEXT_ENCODER,
            TRANSFORMER
        )
        .is_ok(),
        "a decode-dominated peak above the co-staged bound is not a staging regression"
    );

    // The regression the check exists for: the text encoder still resident when the AvDiT is up.
    let regressed = ltx_staging_is_proven(
        entering_denoise(COSTAGED + 1),
        COSTAGED,
        TEXT_ENCODER,
        TRANSFORMER,
    )
    .expect_err("a co-resident text encoder must be refused");
    assert!(
        regressed.contains("was not dropped before the AvDiT"),
        "{regressed}"
    );

    // The vacuity guard: a boundary sampled before the transformer materializes would make the
    // upper bound pass while proving nothing.
    let too_early = ltx_staging_is_proven(
        entering_denoise(TRANSFORMER - 1),
        COSTAGED,
        TEXT_ENCODER,
        TRANSFORMER,
    )
    .expect_err("a boundary sampled before the AvDiT exists must be refused");
    assert!(too_early.contains("proves nothing"), "{too_early}");
}

/// sc-18810: WHERE rung 2 engages, derived from the engine's own bound rather than from the
/// measured decode peak. `writable_frame_cap = i32::MAX / (8 * h * w)` is pure arithmetic over
/// `VaeTiling::LTX` and does not move with the host, so it is asserted exactly. The memory bound
/// (`3.3 GB + 340 B/voxel` vs `get_memory_limit() * 0.85`) DOES move with the host, which is
/// precisely why the boundary is attributed to the write bound and not to a machine.
#[test]
fn the_ltx_decode_tiling_boundary_is_the_machine_independent_write_bound() {
    let caps = LTX_RESOLUTIONS.map(|(width, height)| {
        (
            (width, height),
            VaeTiling::LTX.writable_frame_cap(height as i32, width as i32),
        )
    });
    assert_eq!(
        caps,
        [
            ((768, 512), 682),
            ((512, 768), 682),
            ((640, 640), 655),
            ((1280, 704), 297),
            ((704, 1280), 297),
        ]
    );
    let (_, envelope_maximum) = LTX_FRAME_ENVELOPE;
    for ((width, height), cap) in caps {
        // Only the two 0.90 MP buckets can reach their cap inside the declared envelope. The
        // others cannot tile at ANY declared geometry, on any host, for this reason.
        let reachable = cap < i64::from(envelope_maximum);
        assert_eq!(
            reachable,
            (width, height) == (1280, 704) || (width, height) == (704, 1280),
            "{width}x{height} cap {cap} against envelope maximum {envelope_maximum}"
        );
    }
}

/// The companion behavioural half: the arm's rung FOLLOWS the engine across that boundary, in
/// both directions, **at a FIXED budget so both directions are asserted on every host** — a CI
/// runner included. `resolve` budgets against `get_memory_limit()`, so a test that resolved
/// against the live host would assert one thing on a 128 GiB Mac and another on a 16 GiB runner;
/// guarding it behind "only if the memory bound does not bind" would make it vacuous exactly
/// where it runs unattended.
#[test]
fn the_ltx_arm_follows_the_engine_across_the_decode_tiling_boundary() {
    // A budget far above any single-pass cost in the declared envelope: 1280x704 x 297 costs
    // 3.3e9 + 340 * 267,632,640 B = 87.8 GiB single-pass, and the selector's own 0.85 factor
    // leaves 217.6 GiB here. Whatever tiles under THIS budget tiles for the write bound alone.
    const UNCONSTRAINED_GIB: f64 = 256.0;
    // 297 is the cap itself (a declared 10 s x 30 fps cell) and 305 is the next lattice step
    // above it. Above the cap `budgeted_plan` tiles unconditionally — memory cannot buy it back.
    let tiled = LtxDecodePlan::resolve_with_budget(
        validate_ltx_geometry(1280, 704, 305).unwrap(),
        UNCONSTRAINED_GIB,
    )
    .unwrap();
    assert!(
        tiled.tiling.is_some(),
        "305 frames is over the 297 write cap, so no budget keeps it single-pass"
    );
    assert_eq!(tiled.rung(), "bounded_decode");
    assert_eq!(
        tiled.engaged_rungs(),
        ["resident", "staged_residency", "bounded_decode"]
    );
    assert!(
        tiled.spatial_tile_px() > 0 || tiled.temporal_tile_frames() > 0,
        "a selected tiling must name at least one tiled axis"
    );

    // The other direction, pinned at the same fixed budget: AT the cap and with memory to spare,
    // the decode stays single-pass. Together with the row above this is the write bound, isolated
    // from the host.
    let single = LtxDecodePlan::resolve_with_budget(
        validate_ltx_geometry(1280, 704, 297).unwrap(),
        UNCONSTRAINED_GIB,
    )
    .unwrap();
    assert!(
        single.tiling.is_none(),
        "at the cap and inside the memory budget the decode must stay single-pass"
    );
    assert_eq!(single.rung(), "staged_residency");
    assert_eq!(single.lifecycle_fault_phase(), MemoryPhase::Denoise);
    let staged_selection = planned_selection(&ltx_request_json(1280, 704, 297)).unwrap();
    single
        .validate_selected_strategy(&staged_selection)
        .unwrap();
    // SC-19109 replaces historical host inference with an explicit provider-owned selection:
    // the same geometry can deliberately request bounded decode with its exact carrier tuple.
    let mut request = ltx_request_json(1280, 704, 297);
    request["planned"]["strategy"] = json!({
        "rung": "bounded_decode",
        "engagedRungs": ["resident", "staged_residency", "bounded_decode"],
        "parameters": { "decodeTileEdge": 384, "decodeOverlap": 64 }
    });
    let selection = planned_selection(&request).unwrap();
    let contract = ltx_fixture_contract(Some(Quant::Q8));
    contract.validate_selection(&selection).unwrap();
    let attested = ltx_attested_strategy(&request, &selection, &contract).unwrap();
    assert_eq!(attested, request["planned"]["strategy"]);
    let explicit = LtxDecodePlan::resolve_for_selection(
        &selection,
        validate_ltx_geometry(1280, 704, 297).unwrap(),
    )
    .unwrap();
    explicit.validate_selected_strategy(&selection).unwrap();
    assert_eq!(explicit.lifecycle_fault_phase(), MemoryPhase::Decode);
    assert_eq!(explicit.spatial_tile_px(), 384);
    assert_eq!(explicit.spatial_overlap_px(), 64);

    // The one-sided half of the claim, asserted rather than assumed: the write cap is a CEILING
    // on single-pass frames, not the place tiling starts. A smaller host tiles the very same
    // f297 geometry the row above kept single-pass — and 768x512 x 97, which is 585 frames below
    // its 682 cap and is exactly what a hosted CI runner does to this arm's own smoke geometry.
    // `slack` is how far each row sits below the write bound: 0 at the cap, 585 at the smoke
    // geometry. Both tile anyway, which is the point — the write bound is not what forces it.
    for (width, height, frames, budget_gib, slack) in [
        (1280u32, 704u32, 297u32, 32.0f64, 0i64),
        (768, 512, 97, 16.0, 585),
    ] {
        // The precondition is DERIVED, not asserted in prose: this budget must actually be below
        // the engine's own single-pass cost, or the row proves nothing.
        let single_pass_gib = ltx_decode_cost::single_pass_gib(width, height, frames);
        assert!(
            single_pass_gib > budget_gib * ltx_decode_cost::SAFETY_FACTOR,
            "{width}x{height} f{frames}: {budget_gib} GiB does not constrain a \
                 {single_pass_gib:.2} GiB single-pass decode, so this row would be vacuous"
        );
        let constrained = LtxDecodePlan::resolve_with_budget(
            validate_ltx_geometry(width, height, frames).unwrap(),
            budget_gib,
        )
        .unwrap();
        assert!(
            constrained.tiling.is_some(),
            "{width}x{height} f{frames} must tile for MEMORY under a {budget_gib} GiB budget"
        );
        assert_eq!(constrained.rung(), "bounded_decode");
        assert_eq!(constrained.lifecycle_fault_phase(), MemoryPhase::Decode);
        let staged_selection = planned_selection(&ltx_request_json(width, height, frames)).unwrap();
        let mismatch = constrained
            .validate_selected_strategy(&staged_selection)
            .expect_err("an auto-tiled render must not attest a staged single-pass row");
        assert!(
            mismatch.contains("auto-selected bounded decode"),
            "{mismatch}"
        );
        // The write bound PERMITTED a single pass here (`f <= cap`) and memory still tiled it.
        assert_eq!(
            constrained.writable_frame_cap - i64::from(frames),
            slack,
            "{width}x{height} f{frames} must tile with the write bound still permitting a \
                 single pass (cap {})",
            constrained.writable_frame_cap
        );
    }

    // The third outcome, which CI found and this test did not originally model: below the
    // full-output ACCUMULATOR floor (`3.3 GB + 40 B/voxel`, 13.05 GiB at the cap geometry) no
    // tiling helps — the accumulators hold the assembled video — so the engine refuses outright
    // before any render. A hosted macOS runner lands here at ~6 GiB of safe budget. Pinned at a
    // fixed budget so the refusal is asserted on every host, not only on a small one.
    const BELOW_THE_ACCUMULATOR_FLOOR_GIB: f64 = 8.0;
    let accumulators_gib = ltx_decode_cost::accumulator_floor_gib(1280, 704, 297);
    assert!(
        accumulators_gib > BELOW_THE_ACCUMULATOR_FLOOR_GIB * ltx_decode_cost::SAFETY_FACTOR,
        "this budget must sit below the {accumulators_gib:.2} GiB accumulator floor or the \
             refusal below would be proving something else"
    );
    let refused = LtxDecodePlan::resolve_with_budget(
        validate_ltx_geometry(1280, 704, 297).unwrap(),
        BELOW_THE_ACCUMULATOR_FLOOR_GIB,
    )
    .expect_err("under the accumulator floor the decode must be refused, not tiled");
    assert!(refused.contains("just for the output buffers"), "{refused}");
    assert!(
        refused.contains("refuses 1280x704 x 297 frames before any render"),
        "{refused}"
    );
}

/// The margin the fixed-budget injections above ride on, asserted instead of assumed.
///
/// `run_ltx` reads the process-global MLX limit without `LTX_MEMORY_LIMIT_LOCK`, so an
/// injected budget below the `run_ltx` smoke geometry's accumulator floor would make the engine
/// refuse that geometry and replace the message those refusal tests assert. The lowest budget
/// injected today (8 GiB, 6.8 GiB safe) clears the 4.49 GiB floor by 1.51x — a margin that used
/// to be load-bearing and unstated.
#[test]
#[should_panic(expected = "accumulator floor of the 768x512 x 97 smoke geometry")]
fn an_injected_budget_below_the_smoke_geometrys_accumulator_floor_is_refused() {
    // 5.2 GiB leaves 4.42 GiB safe, just under the 4.49 GiB floor. This is exactly the row a
    // future author would add to probe a smaller host, and it must fail HERE — loudly, at the
    // injection site — rather than intermittently in an unrelated `run_ltx` refusal test.
    let _ = LtxDecodePlan::resolve_with_budget(validate_ltx_geometry(768, 512, 97).unwrap(), 5.2);
}

/// The other side of that boundary: the assertion is a FLOOR, not a blanket refusal. 5.3 GiB
/// leaves 4.505 GiB safe against the same 4.494 GiB floor the 5.2 GiB row falls through, so the
/// two rows bracket it to within 0.1 GiB and neither can pass by being trivially true.
#[test]
fn an_injected_budget_just_above_the_smoke_geometrys_accumulator_floor_is_accepted() {
    // Capture the prior limit only after `install` owns the injection lock. Reading it before
    // the lock is a TOCTOU race: another test may temporarily expose its 32 GiB injection,
    // restore the real host limit, and then let this test acquire the lock. CI caught exactly
    // that 32 GiB -> 7.14 GB transition.
    let budget = LtxInjectedBudget::install(5.3);
    let previous = budget.previous;
    drop(budget);

    // Reacquire before observing the process-global limit. An intervening injection is allowed
    // to run, but must finish and restore `previous` before this assertion reads the value.
    let _restoration_guard = LTX_MEMORY_LIMIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(get_memory_limit(), previous);
}

/// The injected limit is process-global, so failing to restore it leaks into every later test.
#[test]
fn an_injected_budget_is_restored_on_the_normal_and_the_unwind_path() {
    let budget = LtxInjectedBudget::install(32.0);
    let previous = budget.previous;
    assert_eq!(
        get_memory_limit(),
        (32.0 * ltx_decode_cost::GIB) as usize,
        "the budget must actually be installed while the guard lives"
    );
    drop(budget);
    let restoration_guard = LTX_MEMORY_LIMIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(get_memory_limit(), previous, "restored on the normal path");
    drop(restoration_guard);
    // The path a trailing `set_memory_limit(previous)` statement misses entirely — and which
    // `unwrap_or_else(PoisonError::into_inner)` would then hide, because the poisoned lock is
    // the only signal the leak leaves behind.
    let unwind_budget = LtxInjectedBudget::install(32.0);
    let unwind_previous = unwind_budget.previous;
    let outcome = std::panic::catch_unwind(|| {
        let _budget = unwind_budget;
        panic!("the selector blew up mid-plan");
    });
    assert!(outcome.is_err(), "the closure must actually have panicked");
    let _restoration_guard = LTX_MEMORY_LIMIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        get_memory_limit(),
        unwind_previous,
        "the injected limit leaked past a panic"
    );
}

/// The live host, stated as its own claim so the fixed-budget test above cannot be mistaken for
/// a statement about this machine.
///
/// The cap geometry has THREE outcomes, not two, and which one a host gets is a total function
/// of its budget. Exactly one arm below fires on any machine, so this is host-adaptive rather
/// than host-conditional — nothing is skipped anywhere:
///
/// * the full-output accumulators alone (`3.3 GB + 40 B/voxel`) exceed the budget → **refused**
///   before any render. A hosted CI runner lands here: ~13 GB of buffers against a ~6 GB safe
///   budget. Tiling cannot help, because the accumulators hold the assembled video.
/// * a single pass (`3.3 GB + 340 B/voxel`) fits → **single-pass**. This Mac lands here.
/// * in between → **tiled**, or refused if not even the smallest tile fits.
#[test]
fn the_ltx_arm_resolves_the_cap_geometry_against_this_hosts_live_budget() {
    // Read the limit and resolve under the SAME lock `LtxInjectedBudget` swaps under.
    // MLX's memory limit is process-global: without this, the boundary test's injected budget
    // can land between the read and the resolve and this comparison reads two different hosts.
    let guard = LTX_MEMORY_LIMIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let budget_gib =
        get_memory_limit() as f64 / ltx_decode_cost::GIB * ltx_decode_cost::SAFETY_FACTOR;
    let live = LtxDecodePlan::resolve(validate_ltx_geometry(1280, 704, 297).unwrap());
    drop(guard);
    let accumulators_gib = ltx_decode_cost::accumulator_floor_gib(1280, 704, 297);
    let single_pass_gib = ltx_decode_cost::single_pass_gib(1280, 704, 297);
    let where_we_are = format!(
        "accumulators {accumulators_gib:.2} GiB, single-pass {single_pass_gib:.2} GiB, this \
             host's safe budget {budget_gib:.2} GiB"
    );
    if accumulators_gib >= budget_gib {
        let error = live.expect_err("the accumulators alone do not fit — this must be refused");
        assert!(
            error.contains("just for the output buffers"),
            "{where_we_are}: {error}"
        );
    } else if single_pass_gib <= budget_gib {
        let plan = live.unwrap_or_else(|error| panic!("{where_we_are}: {error}"));
        assert!(plan.tiling.is_none(), "{where_we_are}");
        assert_eq!(plan.rung(), "staged_residency");
    } else {
        assert!(
            live.map_or(true, |plan| plan.tiling.is_some()),
            "{where_we_are}: a decode over budget must not resolve single-pass"
        );
    }
}

/// AC3, and the reason a video arm is safe to add: EVERY image arm still refuses a multi-frame
/// geometry, before it does environment or weight work. Four of the six never validated the
/// axis at all before sc-18808 — they read only width/height and hardcoded `frames: 1` into the
/// admission context, so a `frames: 2` plan row would have rendered one frame and recorded a
/// geometry it was never asked for.
///
/// The list below is the IMAGE arms, and it stays six long as video arms are added: `ltx_2_3`
/// (sc-18808) and `minimax_h3` (sc-18663) are the only two allowed to accept a multi-frame
/// geometry, and each buys that by validating against its own engine's lattice rather than by
/// dropping the axis — see `the_minimax_arm_accepts_a_multi_frame_geometry_the_image_guard_refuses`
/// and `validate_ltx_geometry`. A NEW arm added here without such an envelope is the defect
/// this test exists to prevent.
#[test]
fn every_image_arm_still_refuses_a_multi_frame_geometry() {
    type Arm = fn(&Value) -> Result<Value, String>;
    // (provider, model id, declared overlay, refusal label, arm). The model id and overlay are
    // columns because sc-22728, sc-22727 and sc-22735 each added a family whose members share
    // one provider and differ in exactly those two axes; every other row still carries
    // `modelId == provider`, `overlay` "none" (or Krea's control overlay).
    let arms: [(&str, &str, &str, &str, Arm); 12] = [
        (
            KREA_BASE_PROVIDER,
            KREA_BASE_PROVIDER,
            "none",
            "MLX Krea base calibration",
            run_krea_base,
        ),
        // sc-22735: the second member of the Krea base family. It shares the arm, so it must
        // also still refuse a multi-frame geometry under its OWN refusal label — the video
        // widening in this story is the separate `krea_realtime_14b` arm, not this one.
        (
            KREA_RAW_PROVIDER,
            KREA_RAW_PROVIDER,
            "none",
            "MLX Krea 2 Raw calibration",
            run_krea_base,
        ),
        (
            SDXL_PROVIDER,
            SDXL_PROVIDER,
            "none",
            "MLX SDXL base calibration",
            run_sdxl,
        ),
        (
            Z_IMAGE_PROVIDER,
            Z_IMAGE_PROVIDER,
            "none",
            "MLX Z-Image base calibration",
            run_z_image_reference,
        ),
        // sc-22724: the base model shares the arm; its refusal carries its own label.
        (
            Z_IMAGE_BASE_PROVIDER,
            Z_IMAGE_BASE_PROVIDER,
            "none",
            "MLX Z-Image base-model calibration",
            run_z_image_reference,
        ),
        (
            KREA_PROVIDER,
            KREA_PROVIDER,
            "control:1",
            "MLX Krea pose-control calibration",
            run_krea_control,
        ),
        (
            QWEN_PROVIDER,
            QWEN_PROVIDER,
            "none",
            "MLX Qwen base calibration",
            run_qwen_provider,
        ),
        // sc-22728: both Qwen edit catalog ids, each with its own label and declared overlay.
        (
            QWEN_EDIT_PROVIDER,
            QWEN_EDIT_MODEL,
            "none",
            "MLX Qwen edit calibration",
            run_qwen_edit_provider,
        ),
        (
            QWEN_EDIT_PROVIDER,
            QWEN_EDIT_LIGHTNING_MODEL,
            "lora",
            "MLX Qwen edit Lightning calibration",
            run_qwen_edit_provider,
        ),
        (
            FLUX2_PROVIDER,
            FLUX2_PROVIDER,
            "none",
            "MLX FLUX.2-dev calibration",
            run_flux2,
        ),
        // sc-22727: the two klein catalog models share `flux2_klein_9b` and each refuses under
        // its OWN label, so neither can be mistaken for the other in a failure report.
        (
            FLUX2_KLEIN_PROVIDER,
            "flux2_klein_9b",
            "none",
            "MLX FLUX.2-klein-9B calibration",
            run_flux2,
        ),
        (
            FLUX2_KLEIN_PROVIDER,
            "flux2_klein_9b_kv",
            "none",
            "MLX FLUX.2-klein-9B KV calibration",
            run_flux2,
        ),
    ];
    for (provider, model_id, overlay, label, arm) in arms {
        for frames in [0_u64, 2, 97] {
            let request = json!({
                "planned": {
                    "target": {
                        "provider": provider,
                        "modelId": model_id,
                        "tier": "q4",
                        "mode": if provider == QWEN_EDIT_PROVIDER { "edit_image" } else { "text_to_image" },
                        "overlay": overlay,
                        "geometry": { "width": 768, "height": 768, "batch": 1, "frames": frames }
                    },
                    "backend": "mlx",
                    "loadShape": "deferred_materialization",
                    "strategy": {
                        "rung": "bounded_decode",
                        "engagedRungs": ["resident", "bounded_decode"],
                        "parameters": { "decodeTileEdge": 512, "decodeOverlap": 64 }
                    },
                    "calibrationFingerprint": "unused",
                    "fixture": "unused"
                }
            });
            let error = arm(&request).expect_err("an image arm must refuse a multi-frame geometry");
            assert_eq!(
                error,
                format!("{label} requires geometry.frames == 1, got {frames}"),
                "{provider}/{model_id} at frames={frames}"
            );
        }
    }
}

/// And the still geometry itself must still get PAST the guard on every image arm, so the
/// refusals above are the frames axis rather than a blanket rejection.
#[test]
fn the_still_geometry_guard_is_not_a_blanket_refusal() {
    for label in [
        "MLX Krea base calibration",
        "MLX Krea 2 Raw calibration",
        "MLX SDXL base calibration",
        "MLX Z-Image base calibration",
        "MLX Z-Image base-model calibration",
        "MLX Z-Image edit calibration",
        "MLX Krea pose-control calibration",
        "MLX Qwen base calibration",
        "MLX Qwen edit calibration",
        "MLX Qwen edit Lightning calibration",
        "MLX FLUX.2-klein-9B calibration",
        "MLX FLUX.2-klein-9B KV calibration",
        "MLX FLUX.2-dev calibration",
    ] {
        let request = json!({
            "planned": { "target": { "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 } } }
        });
        assert!(
            protocol::validate_still_geometry(&request, label).is_ok(),
            "{label}"
        );
    }
}

/// The video output unwrapper refuses the two shapes an image provider would return, so a
/// misrouted still could never be recorded as a clip.
#[test]
fn the_video_unwrapper_refuses_image_and_audio_shaped_output() {
    let frame = Image {
        width: 2,
        height: 1,
        pixels: vec![0, 0, 0, 255, 255, 255],
    };
    assert!(video_frames(GenerationOutput::Images(vec![frame.clone()]))
        .unwrap_err()
        .contains("returned images, not a video clip"));
    assert!(video_frames(GenerationOutput::Video {
        frames: Vec::new(),
        fps: 24,
        audio: None,
    })
    .unwrap_err()
    .contains("returned no frames"));
    let (frames, fps, audio) = video_frames(GenerationOutput::Video {
        frames: vec![frame],
        fps: 25,
        audio: None,
    })
    .unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(fps, 25);
    assert!(!audio);
}

/// The clip comparator is per-frame and mutation-sensitive: a divergence confined to a LATE
/// frame must be seen, which a first-frame spot check would miss.
#[test]
fn the_clip_comparator_sees_a_late_frame_divergence() {
    let frame = |value: u8| Image {
        width: 2,
        height: 2,
        pixels: vec![value; 12],
    };
    let left = vec![frame(10), frame(10), frame(10)];
    let identical = left.clone();
    let (maximum, mean, rms) = video_max_mean_rms_abs(&left, &identical).unwrap();
    assert_eq!((maximum, mean, rms), (0.0, 0.0, 0.0));
    assert!(ltx_quality_passes(maximum, mean, rms));

    let mut late = left.clone();
    late[2] = frame(200);
    let (maximum, mean, rms) = video_max_mean_rms_abs(&left, &late).unwrap();
    assert!(maximum > LTX_MAX_THRESHOLD, "max={maximum}");
    assert!(!ltx_quality_passes(maximum, mean, rms));

    assert!(video_max_mean_rms_abs(&left, &left[..2])
        .unwrap_err()
        .contains("frame-count mismatch"));
}

#[test]
fn the_ltx_runtime_context_binds_the_exact_video_route_and_selection() {
    let request = ltx_request_json(1280, 704, 305);
    let selection = planned_selection(&request).unwrap();
    let contract = ltx_fixture_contract(Some(Quant::Q8));
    let calibration = contract.calibration.as_ref().unwrap();
    let context = ltx_context(
        selection,
        calibration,
        &calibration.fingerprint,
        validate_ltx_geometry(1280, 704, 305).unwrap(),
        64 * 1024 * 1024 * 1024,
        1,
    );
    assert_eq!(context.mode.as_key(), "text_to_video");
    // The worker's video context (`video_admission.rs`) carries `false`; the pinned provider's
    // route gate rejects `true` outright.
    assert!(!context.has_phases);
    assert_eq!(context.geometry.frames, 305);
    assert_eq!(context.selection.strategy, MemoryStrategy::StagedResidency);
    assert_eq!(
        context.evidence_revision,
        format!("sc-18946@{}", protocol::INFERENCE_PIN)
    );
}

/// The staged-residency bound is a real inequality over real byte counts, not a comment.
#[test]
fn safetensors_bytes_sums_only_weight_shards_and_fails_closed_when_empty() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let root =
        fixtures.path("safetensors-bytes-sums-only-weight-shards-and-fails-closed-when-empty");
    std::fs::create_dir_all(&root).unwrap();
    assert!(safetensors_bytes(&root)
        .unwrap_err()
        .contains("no .safetensors weights"));
    std::fs::write(root.join("config.json"), b"{}").unwrap();
    assert!(
        safetensors_bytes(&root).is_err(),
        "a config is not a weight shard"
    );
    std::fs::write(root.join("model-00001-of-00002.safetensors"), vec![0_u8; 7]).unwrap();
    std::fs::write(root.join("model-00002-of-00002.safetensors"), vec![0_u8; 5]).unwrap();
    assert_eq!(safetensors_bytes(&root).unwrap(), 12);
}
