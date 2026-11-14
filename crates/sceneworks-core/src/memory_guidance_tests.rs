use super::*;
use serde_json::json;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn hf(variant: &str, footprint: Value) -> Value {
    json!({"provider": "huggingface", "repo": "o/m", "variant": variant, "footprint": footprint})
}

fn guidance(model: &Value) -> Value {
    guidance_with_endpoints(model, None)
}

fn tier<'a>(guidance: &'a Value, lane: &str, tier: &str) -> &'a Value {
    &guidance[lane]["tiers"][tier]
}

/// The served policy values the web no longer holds (SW-7032). A change here changes every
/// served threshold, which the catalog fixture and the web's served-guidance tests then see.
#[test]
fn policy_constants_are_the_calibrated_values() {
    assert_eq!(DISK_TO_RESIDENT_MULTIPLIER, 1.0);
    assert_eq!(TRANSIENT_HEADROOM_BYTES, 15_032_385_536);
    assert_eq!(TRANSIENT_HEADROOM_MEASURED_PIXELS, 1_048_576);
    assert_eq!(MEASURED_SIBLING_DELTA_TOLERANCE, 1.05);
    assert_eq!(MEMORY_HEADROOM_FRACTION, 0.9);
    assert_eq!(RESOLUTION_BASELINE_PIXELS, 2_359_296);
    assert_eq!(HIGHRES_TRANSIENT_GB_PER_MP, 13.0);
    assert_eq!(DEDICATED_VRAM_ALLOCATOR_SLACK_GB, 2.0);
    assert_eq!(CANDLE_ACTIVE_RECAPTURE_SPREAD, 0.02);
}

#[test]
fn mlx_fit_threshold_follows_the_peak_priority() {
    let model = json!({"downloads": [
        hf("bf16", json!({"diskSizeBytes": 30.0 * GIB, "peakMemoryBytes": 40.0 * GIB})),
        hf("q8", json!({"diskSizeBytes": 20.0 * GIB, "residentMemoryBytes": 18.0 * GIB})),
        hf("q4", json!({"diskSizeBytes": 10.0 * GIB})),
        {"provider": "huggingface", "repo": "o/m", "variant": "training", "estimatedSizeBytes": 6_u64 << 30},
    ]});
    let served = guidance(&model);
    let threshold = |name: &str| tier(&served, "mlx", name)["minHostGb"].as_f64().unwrap();
    // Measured peak, then measured resident + the fixed 14 GiB transient, then disk + transient,
    // then the download size + transient; each over the 0.9 unified-memory budget.
    assert_eq!(threshold("bf16"), 40.0 / MEMORY_HEADROOM_FRACTION);
    assert_eq!(threshold("q8"), (18.0 + 14.0) / MEMORY_HEADROOM_FRACTION);
    assert_eq!(threshold("q4"), (10.0 + 14.0) / MEMORY_HEADROOM_FRACTION);
    assert_eq!(
        threshold("training"),
        (6.0 + 14.0) / MEMORY_HEADROOM_FRACTION
    );
    // Suggestion order is fidelity order over the declared quant tiers only.
    let order: Vec<_> = served["mlx"]["suggestion"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["tier"].as_str().unwrap())
        .collect();
    assert_eq!(order, ["bf16", "q8", "q4"]);
}

#[test]
fn a_measured_sibling_extrapolates_only_when_declared_and_calibrated() {
    let q4 = hf(
        "q4",
        json!({"diskSizeBytes": 20.0 * GIB, "peakMemoryBytes": 30.0 * GIB, "measuredPixels": 1_048_576}),
    );
    let bf16 = hf("bf16", json!({"diskSizeBytes": 60.0 * GIB}));
    let declared =
        json!({"mlx": {"measuredSiblingActiveDiskFraction": 0.5}, "downloads": [q4, bf16]});
    let expected =
        (30.0 * GIB + (40.0 * GIB * 0.5 * MEASURED_SIBLING_DELTA_TOLERANCE).round()) / GIB;
    assert_eq!(
        tier(&guidance(&declared), "mlx", "bf16")["minHostGb"],
        json!(expected / MEMORY_HEADROOM_FRACTION)
    );
    // No product declaration ⇒ the dense estimate; a sibling peak at another geometry is not usable.
    let dense = (60.0 + 14.0) / MEMORY_HEADROOM_FRACTION;
    let undeclared = json!({"downloads": declared["downloads"].clone()});
    assert_eq!(
        tier(&guidance(&undeclared), "mlx", "bf16")["minHostGb"],
        json!(dense)
    );
    let mut other_geometry = declared.clone();
    other_geometry["downloads"][0]["footprint"]["measuredPixels"] = json!(2_097_152);
    assert_eq!(
        tier(&guidance(&other_geometry), "mlx", "bf16")["minHostGb"],
        json!(dense)
    );
}

#[test]
fn candle_reads_only_its_own_lane_and_adds_the_allocator_slack() {
    let model = json!({
        "candle": {"vramGbByTier": {"q4": 37.3, "int8-convrot": 34.7}, "minMemoryGbByTier": {"q8": 40}},
        "downloads": [
            hf("q4", json!({"diskSizeBytes": 10.0 * GIB, "peakMemoryBytes": 30.5 * GIB})),
            hf("q8", json!({"diskSizeBytes": 20.0 * GIB})),
            hf("bf16", json!({"diskSizeBytes": 40.0 * GIB})),
        ],
    });
    let served = guidance(&model);
    // The MLX footprint never prices the candle lane.
    assert_eq!(
        tier(&served, "candle", "q4")["minHostGb"],
        json!(37.3 + 2.0)
    );
    // A declared per-tier floor row is a veto on its own; no candle row and no floor ⇒ unknown.
    assert_eq!(tier(&served, "candle", "q8")["minHostGb"], json!(40.0));
    assert_eq!(tier(&served, "candle", "bf16")["minHostGb"], Value::Null);
    assert_eq!(
        tier(&served, "candle", "q4")["installedHostGb"],
        json!(40.0)
    );
    assert_eq!(
        tier(&served, "candle", "int8-convrot")["measuredHostGb"],
        json!(37.0)
    );
    // The MLX host conversion is multiplicative: 30.50 GiB needs a 34 GB host.
    assert_eq!(tier(&served, "mlx", "q4")["installedHostGb"], json!(34.0));
    assert_eq!(
        tier(&served, "mlx", "q4")["installedEstimated"],
        json!(false)
    );
    // An unmeasured MLX tier is filled from its estimate and flagged; candle never fills.
    assert_eq!(
        tier(&served, "mlx", "q8")["installedEstimated"],
        json!(true)
    );
    assert_eq!(
        tier(&served, "candle", "q8")["installedHostGb"],
        Value::Null
    );
}

#[test]
fn a_candle_only_tier_without_its_row_takes_the_q8_proxy_as_an_estimate() {
    let model = json!({
        "candle": {"vramGbByTier": {"q8": 30.0}, "minMemoryGb": 32, "measured": false},
        "downloads": [hf("q8", json!({})), hf("nvfp4", json!({}))],
    });
    let served = guidance(&model);
    let nvfp4 = tier(&served, "candle", "nvfp4");
    assert_eq!(nvfp4["installedHostGb"], json!(32.0));
    assert_eq!(nvfp4["installedEstimated"], json!(true));
    assert_eq!(nvfp4["measuredHostGb"], Value::Null);
    assert_eq!(served["candle"]["uncalibrated"], json!(true));
    assert_eq!(served["mlx"]["uncalibrated"], json!(false));
    assert_eq!(nvfp4["floorGb"], json!(32.0));
}

#[test]
fn entry_floor_skips_a_pending_tier() {
    let mut pending = hf("q4", json!({}));
    pending["pendingArtifact"] = json!(true);
    let model = json!({
        "mlx": {"minMemoryGb": 45, "minMemoryGbByTier": {"q4": 21, "q8": 30}},
        "downloads": [pending, hf("q8", json!({})), hf("bf16", json!({}))],
    });
    let served = guidance(&model);
    assert_eq!(served["mlx"]["entryFloorGb"], json!(30.0));
    assert_eq!(served["mlx"]["blanketGb"], json!(45.0));
    assert_eq!(tier(&served, "mlx", "bf16")["floorGb"], json!(45.0));
}

#[test]
fn convert_at_install_tiers_are_estimated_from_their_disk_size() {
    let model = json!({
        "downloads": [{"provider": "huggingface", "repo": "o/convert"}],
        "mlxTierStates": [{"tier": "q4", "diskSizeBytes": 6.0 * GIB}, {"tier": "q8", "diskSizeBytes": null}],
    });
    let served = guidance(&model);
    assert_eq!(
        tier(&served, "mlx", "q4")["installedHostGb"],
        json!(((6.0 + 14.0) / MEMORY_HEADROOM_FRACTION).ceil())
    );
    assert_eq!(tier(&served, "mlx", "q8")["installedHostGb"], Value::Null);
    assert_eq!(served["mlx"]["suggestion"], json!([]));
}

#[test]
fn resolution_thresholds_cover_only_declared_buckets_above_the_baseline() {
    let model = json!({
        "mlx": {"minMemoryGb": 48},
        "candle": {"minMemoryGb": 32, "vramGbByTier": {"q4": 20.0}, "vramMeasuredPixels": 1_048_576},
        "downloads": [hf("q4", json!({"peakMemoryBytes": 22.0 * GIB, "measuredPixels": 1_048_576}))],
        "limits": {"resolutions": ["1024x1024", "1536x1536", "2048x2048", "bogus"]},
    });
    let served = guidance(&model);
    let extra = (2048.0 * 2048.0 - 1536.0 * 1536.0) / 1_000_000.0;
    let lift = (1536.0 * 1536.0 - 1024.0 * 1024.0) / 1_000_000.0;
    assert_eq!(
        served["mlx"]["resolutionMinHostGb"],
        json!({"2048x2048": (48.0 + 13.0 * extra) / MEMORY_HEADROOM_FRACTION})
    );
    // Candle's floor is already a dedicated-VRAM host requirement.
    assert_eq!(
        served["candle"]["resolutionMinHostGb"],
        json!({"2048x2048": 32.0 + 13.0 * extra})
    );
    // A selected tier's own lane peak is lifted from its 1024² geometry to the 1536² baseline.
    assert_eq!(
        tier(&served, "candle", "q4")["resolutionMinHostGb"],
        json!({"2048x2048": 20.0 + 13.0 * lift + 13.0 * extra})
    );
    assert_eq!(
        tier(&served, "mlx", "q4")["resolutionMinHostGb"],
        json!({"2048x2048": (22.0 + 13.0 * lift + 13.0 * extra) / MEMORY_HEADROOM_FRACTION})
    );
    // A peak with no measurement geometry is not a resolution basis.
    let mut no_geometry = model.clone();
    no_geometry["downloads"][0]["footprint"]
        .as_object_mut()
        .unwrap()
        .remove("measuredPixels");
    assert!(tier(&guidance(&no_geometry), "mlx", "q4")
        .get("resolutionMinHostGb")
        .is_none());
}

#[test]
fn the_packaged_sdxl_endpoint_prices_its_own_coordinate_only() {
    let pair =
        crate::memory_anchor::sdxl_cuda_memory_endpoints::packaged_sdxl_cuda_memory_endpoints()
            .expect("packaged SDXL endpoint pair");
    let sdxl = json!({"id": pair.sequential.model_id, "downloads": [hf("q4", json!({})), hf("q8", json!({}))]});
    let served = guidance_with_endpoints(&sdxl, Some(pair));
    let estimate = &served["candle"]["endpointEstimate"];
    let optimized =
        (pair.sequential.overall_allocator_envelope_bytes as f64 * 1.02).ceil() / GIB + 2.0;
    assert_eq!(estimate["optimizedHostGb"], json!(optimized));
    assert_eq!(estimate["tier"], json!("q4"));
    assert_eq!(
        estimate["label"],
        json!(format!("Q4 optimized ~{optimized:.1} GiB"))
    );
    let description = estimate["description"].as_str().unwrap();
    assert!(
        description.starts_with(
            "Stock SDXL Q4, 1024 × 1024, one image, default sampling (30 steps, guidance 7, DDIM), no adapters."
        ),
        "{description}"
    );
    assert!(
        description.contains("including 2% allowance and 2 GiB reserve"),
        "{description}"
    );
    assert_eq!(tier(&served, "candle", "q4")["minHostGb"], json!(optimized));
    assert!(served["mlx"].get("endpointEstimate").is_none());
    // Another model, or a pending measured tier, gets no endpoint guidance.
    let other = json!({"id": "realvisxl", "downloads": sdxl["downloads"].clone()});
    assert!(guidance_with_endpoints(&other, Some(pair))["candle"]
        .get("endpointEstimate")
        .is_none());
    let mut pending = sdxl.clone();
    pending["downloads"][0]["pendingArtifact"] = json!(true);
    assert!(guidance_with_endpoints(&pending, Some(pair))["candle"]
        .get("endpointEstimate")
        .is_none());
}

/// The served object is a published API→web wire contract: every shipped row's guidance, on every
/// OS projection, validates against `packages/schemas/model-memory-guidance.schema.json`.
#[test]
fn served_guidance_matches_its_published_schema() {
    let schema: Value = serde_json::from_str(include_str!(
        "../../../packages/schemas/model-memory-guidance.schema.json"
    ))
    .unwrap();
    let validator = jsonschema::validator_for(&schema).expect("valid guidance schema");
    let manifest = crate::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap();
    let mut validated = 0;
    for os in ["macos", "windows", "linux"] {
        for model in manifest["models"].as_array().unwrap() {
            let mut entry = model.clone();
            crate::model_artifacts::artifact_selection::retain_downloads_for_os(&mut entry, os);
            let served = model_memory_guidance(&entry);
            if let Some(error) = validator.iter_errors(&served).next() {
                panic!("{os}/{}: {error}", entry["id"]);
            }
            validated += 1;
        }
    }
    assert!(validated > 0);
    let mut drifted = model_memory_guidance(&json!({"downloads": [hf("q4", json!({}))]}));
    drifted["mlx"]["unexpected"] = json!(1);
    assert!(
        !validator.is_valid(&drifted),
        "an unpublished field is refused"
    );
}

/// Catalog-driven invariants over every shipped builtin on every OS projection: each installed
/// host size satisfies its OWN lane's fit criterion at that size and fails it one GB lower (never an
/// under-statement), and a threshold is never quoted for a lane with no number.
#[test]
fn every_shipped_model_quotes_host_sizes_that_satisfy_its_own_lane() {
    let manifest = crate::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap();
    let models = manifest["models"].as_array().unwrap();
    let mut quoted = 0;
    for os in ["macos", "windows", "linux"] {
        for model in models {
            let mut entry = model.clone();
            crate::model_artifacts::artifact_selection::retain_downloads_for_os(&mut entry, os);
            let served = model_memory_guidance(&entry);
            let peaks = Entry::new(&entry);
            for lane in [Lane::Mlx, Lane::Candle] {
                for (name, gb) in peaks.measured_peaks_gb(lane) {
                    let Some(peak) = gb else { continue };
                    let host = served[lane.key()]["tiers"][&name]["measuredHostGb"]
                        .as_f64()
                        .unwrap();
                    let fits = |host: f64| match lane {
                        Lane::Mlx => peak <= host * MEMORY_HEADROOM_FRACTION,
                        Lane::Candle => peak + DEDICATED_VRAM_ALLOCATOR_SLACK_GB <= host,
                    };
                    assert!(
                        fits(host) && !fits(host - 1.0),
                        "{os}/{}/{name}",
                        entry["id"]
                    );
                    quoted += 1;
                }
            }
        }
    }
    assert!(quoted > 0, "the sweep must cover measured tiers");
}
