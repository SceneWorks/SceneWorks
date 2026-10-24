//! Physical Candle/CUDA capture arm for Supra2-IMG (sc-24217, epic 24212).
//!
//! The Candle twin of `mlx_supra2.rs`: the TOTAL pipeline the worker runs — Flan-T5-Base, the
//! converted DiT and the SD VAE, all resident on the device — at the published request (256x256,
//! seed 0, 50 Euler steps, default CFG 3.0), loaded through `runtime_cuda::catalog().media()` with
//! the spec the worker builds (the converted bundle as `LoadSpec::weights`, the two pinned
//! co-requisites as `text_encoder` / `vae` components, resident and eager), under the provider's
//! own safety check and request scope.
//!
//! The bundle is converted in a SEPARATE adapter process before the VRAM probe starts and before
//! this process loads anything (or an already-pinned bundle is reused), so the conversion is never
//! inside a measured window. Every phase figure is the `nvidia-smi` device-memory peak over that
//! phase's window above the probe's idle baseline — the one residency counter this lane has —
//! and each phase also carries its wall time. After the count-1 anchor render, the same loaded
//! model renders the product maximum ([`protocol::SUPRA2_IMG_PRODUCT_MAX_COUNT`]) once, and its
//! device peak and latency travel in the diagnostics.
//!
//! The contract implements `Resident` alone (gen-core's `compatibility_default`: staged residency
//! is `Missing`, not structurally inapplicable), which is why this lane's plan row overrides the
//! staged lane default with the resident rung. It publishes no calibration identity, so the
//! `stale_evidence` scenario is `not_applicable`; the record is `gated`, as every anchor capture
//! on this lane is.

use super::*;
use runtime_cuda::gen_core::supra2::{TEXT_ENCODER_COMPONENT, VAE_COMPONENT};
use runtime_cuda::gen_core::MemoryCalibrationIdentity;
use std::time::Instant;

const LANE: &str = "candle";

/// The env family the catalog runner exports for this arm (`PROVIDER_FAMILIES.supra2_img`).
pub(super) const ENVS: protocol::Supra2Envs = protocol::Supra2Envs {
    repository_env: "SCENEWORKS_SUPRA2_IMG_REPOSITORY",
    revision_env: "SCENEWORKS_SUPRA2_IMG_REVISION",
    root_env: "SCENEWORKS_SUPRA2_IMG_ROOT",
    text_encoder_env: "SCENEWORKS_SUPRA2_IMG_COMPONENT_TEXT_ENCODER",
    vae_env: "SCENEWORKS_SUPRA2_IMG_COMPONENT_VAE",
};

/// The spec the worker loads Supra2 under: the converted bundle plus both pinned components,
/// resident, eager, default (dense) precision, no quantization.
pub(super) fn load_spec(bundle: PathBuf, text_encoder: PathBuf, vae: PathBuf) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(bundle))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization)
        .with_component(TEXT_ENCODER_COMPONENT, WeightsSource::Dir(text_encoder))
        .with_component(VAE_COMPONENT, WeightsSource::Dir(vae))
}

/// The published request at `count` images (guidance left to the engine's default CFG 3.0).
pub(super) fn capture_request(count: u32) -> GenerationRequest {
    let size = runtime_cuda::gen_core::supra2::IMAGE_SIZE;
    GenerationRequest {
        prompt: protocol::SUPRA2_IMG_PROMPT.to_owned(),
        width: size,
        height: size,
        count,
        seed: Some(protocol::SUPRA2_IMG_SEED),
        steps: Some(protocol::SUPRA2_IMG_STEPS),
        ..Default::default()
    }
}

fn millis(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn context(
    selection: MemorySelection,
    calibration: Option<&MemoryCalibrationIdentity>,
    fingerprint: &str,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    let size = runtime_cuda::gen_core::supra2::IMAGE_SIZE;
    MemoryRunContext {
        selection,
        optimization_authority: if calibration.is_some() {
            MemoryOptimizationAuthority::Calibrated
        } else {
            MemoryOptimizationAuthority::Resident
        },
        calibration_abi: calibration
            .map(|identity| identity.abi)
            .unwrap_or(runtime_cuda::gen_core::MEMORY_CALIBRATION_ABI),
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: LoadShape::EagerMaterialization,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: size,
            height: size,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        reference_resources: None,
        overlay: None,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-24217@{}", protocol::INFERENCE_PIN),
    }
}

/// The two request boundaries the anchor render is split at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    ConditioningDone,
    ForwardDone,
}

/// The boundary one progress event crosses: conditioning ends at the first solver step, the
/// forward at `Progress::Decoding`.
fn boundary_of(progress: &Progress) -> Option<Boundary> {
    match progress {
        Progress::Step { current: 1, .. } => Some(Boundary::ConditioningDone),
        Progress::Decoding => Some(Boundary::ForwardDone),
        _ => None,
    }
}

/// Run one request under the provider's own request scope, the way the worker runs it. Returns the
/// number of images rendered.
fn render(
    generator: &dyn Generator,
    request: &GenerationRequest,
    scoped: &MemoryRunContext,
    on_boundary: &mut dyn FnMut(Boundary),
) -> Result<usize, String> {
    let mut scope = open_memory_request_scope(generator, scoped, "supra2_img")?;
    let mut request = request.clone();
    scope
        .configure_request(&mut request)
        .map_err(|error| format!("configure the supra2_img request scope: {error}"))?;
    let outcome = generator.generate(&request, &mut |progress| {
        if let Some(crossed) = boundary_of(&progress) {
            on_boundary(crossed);
        }
    });
    match outcome {
        Ok(GenerationOutput::Images(images)) => {
            scope
                .finish(MemoryRunOutcome::Complete)
                .map_err(|error| format!("finish the supra2_img request scope: {error}"))?;
            Ok(images.len())
        }
        Ok(_) => {
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: "non-image output".to_owned(),
            });
            Err("supra2_img returned a non-image output".to_owned())
        }
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            Err(format!("supra2_img measured generation failed: {message}"))
        }
    }
}

/// The `candle:supra2_img` arm.
pub(super) fn run(request: &Value) -> Result<Value, String> {
    // Provider, mode, tier, the native still, overlay, load shape, rung, fixture and identity —
    // all before an env var or a weight.
    let (width, height) = protocol::validate_supra2_plan(request, LANE)?;
    let selection = planned_selection(request)?;
    let planned_fingerprint = protocol::supra2_calibration_fingerprint(LANE);
    let inputs = protocol::supra2_inputs(ENVS)?;
    // Converted (in a child process) BEFORE the probe asserts a stable idle device.
    let bundle = protocol::ensure_supra2_bundle(&inputs)?;
    let spec = load_spec(
        bundle.dir.clone(),
        inputs.text_encoder.clone(),
        inputs.vae.clone(),
    );

    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let load_started = Instant::now();
    let generator = runtime_cuda::catalog()
        .map_err(|error| format!("build CUDA catalog: {error}"))?
        .media()
        .load(protocol::SUPRA2_IMG_PROVIDER, &spec)
        .map_err(|error| format!("load real supra2_img provider: {error}"))?;
    let load_millis = millis(load_started);
    vram.end_load(load_sample);

    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded supra2_img exposed no memory-strategy contract".to_owned())?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned supra2_img provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref();
    if let Some(identity) = calibration {
        if identity.fingerprint != planned_fingerprint {
            return Err(format!(
                "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
                identity.fingerprint
            ));
        }
    }
    if contract.load_shape != LoadShape::EagerMaterialization {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            protocol::LOAD_SHAPE_EAGER,
            load_shape_key(contract.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let admitted = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        context(selection, calibration, fingerprint, total_bytes, predicted)
    };
    // Admission mutation hygiene: the gate must ACCEPT a fitting request, so the rejections below
    // cannot pass through a blanket refusal.
    if !matches!(
        generator.memory_strategy_safety_check(&admitted(&planned_fingerprint, hardware_bytes, 1)),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "supra2_img admission rejected a fitting probe budget; the scenario rejections below \
             would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        generator.memory_strategy_safety_check(&admitted(&planned_fingerprint, 0, 1)),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("supra2_img admission accepted an unknown/zero memory budget".to_owned());
    }
    let stale_scenario = match calibration {
        Some(_) => {
            if !matches!(
                generator.memory_strategy_safety_check(&admitted(
                    "stale-supra2-fingerprint",
                    hardware_bytes,
                    1
                )),
                MemorySafetyDecision::Reject { .. }
            ) {
                return Err("supra2_img admission accepted stale calibration evidence".to_owned());
            }
            json!({ "name": "stale_evidence", "result": "passed", "reason": "the loaded Supra2 admission check rejected a mutated calibration fingerprint" })
        }
        None => json!({
            "name": "stale_evidence",
            "result": "not_applicable",
            "reason": format!(
                "the pinned supra2_img provider at inference {} publishes no calibration identity \
                 (gen-core compatibility_default contract), so its admission carries no \
                 calibration handshake for stale evidence to fail",
                protocol::INFERENCE_PIN
            ),
        }),
    };

    // The anchor render: count 1, three device-peak phases, each with its wall time.
    let scoped = admitted(&planned_fingerprint, hardware_bytes, 1);
    let run_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut phase_started = Instant::now();
    let mut conditioning_millis = 0;
    let mut denoise_millis = 0;
    let render_started = Instant::now();
    let rendered = render(
        generator.as_ref(),
        &capture_request(1),
        &scoped,
        &mut |boundary| {
            let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
            match boundary {
                Boundary::ConditioningDone => {
                    conditioning_peak_gb = peak;
                    conditioning_millis = millis(phase_started);
                }
                Boundary::ForwardDone => {
                    denoise_peak_gb = peak;
                    denoise_millis = millis(phase_started);
                }
            }
            phase_started = Instant::now();
            phase_sample = Some(vram.phase());
        },
    );
    let decode_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
    let decode_millis = millis(phase_started);
    let render_millis = millis(render_started);
    vram.end_gen(run_sample);
    if rendered? != 1 {
        return Err("the measured supra2_img render did not return exactly one image".to_owned());
    }

    let phase_bytes = |peak: Option<f64>, phase: &str| {
        peak.map(decimal_gb_to_bytes)
            .ok_or_else(|| format!("supra2_img did not expose a {phase} boundary"))
    };
    let conditioning_bytes = phase_bytes(conditioning_peak_gb, "conditioning")?;
    let denoise_bytes = phase_bytes(denoise_peak_gb, "forward")?;
    let decode_bytes = phase_bytes(decode_peak_gb, "decode")?;
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);

    // The product maximum: one request at the job cap on the same loaded model.
    let product_count = protocol::SUPRA2_IMG_PRODUCT_MAX_COUNT;
    let batch_sample = vram.phase();
    let batch_started = Instant::now();
    let batch_rendered = render(
        generator.as_ref(),
        &capture_request(product_count),
        &scoped,
        &mut |_| {},
    );
    let batch_millis = millis(batch_started);
    let batch_peak_bytes = decimal_gb_to_bytes(vram.end_observed(batch_sample));
    if batch_rendered? != product_count as usize {
        return Err(format!(
            "the supra2_img count-{product_count} render did not return {product_count} images"
        ));
    }

    let blocker = concat!(
        "sc-24217 anchor capture measures exact per-phase device memory and the strategy identity ",
        "for the Candle Supra2-IMG lane; it stays gated because this run does not repeat the full ",
        "promotion-quality, negative-mutation and lifecycle scenario suite, and the pinned provider ",
        "publishes no calibration identity"
    );
    let execution_path = "the supra2_img resident path";
    let mut fragment = json!({
        "status": "gated",
        "strategy": strategy,
        "loadShape": load_shape_key(LoadShape::EagerMaterialization),
        "artifact": inputs.artifact_json(),
        "sweep": protocol::reference_sweep(request, "passed")?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": blocker },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded Supra2 admission check rejected a zero/unknown budget" },
            stale_scenario,
            { "name": "warm_repeat", "result": "not_run", "reason": blocker },
            { "name": "cancel", "result": "not_run", "reason": blocker },
            { "name": "error", "result": "not_run", "reason": blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": null,
        "observedMemory": {
            "conditioning": cuda_phase_metrics(conditioning_bytes),
            "denoise": cuda_phase_metrics(denoise_bytes),
            "decode": cuda_phase_metrics(decode_bytes),
            "overall": cuda_phase_metrics(overall_bytes),
        },
        "quality": { "result": "not_run" },
        "negativeMutation": Value::Null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": inputs.loadability_fingerprint(),
        },
        "diagnostics": protocol::diagnostics(
            "memory-candle-adapter:supra2-img-anchor",
            "executed",
            [blocker.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
                ("bundleConvertedInChildProcess", "count", u64::from(bundle.converted)),
                ("bundleConvertWallMilliseconds", "milliseconds", bundle.convert_millis),
                ("loadWallMilliseconds", "milliseconds", load_millis),
                ("conditioningWallMilliseconds", "milliseconds", conditioning_millis),
                ("denoiseWallMilliseconds", "milliseconds", denoise_millis),
                ("decodeWallMilliseconds", "milliseconds", decode_millis),
                ("count1RenderWallMilliseconds", "milliseconds", render_millis),
                ("productMaxCount", "count", u64::from(product_count)),
                ("productMaxCountDevicePeakDelta", "bytes", batch_peak_bytes),
                ("productMaxCountRenderWallMilliseconds", "milliseconds", batch_millis),
                ("admittedGeometryWidth", "count", u64::from(width)),
                ("admittedGeometryHeight", "count", u64::from(height)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, execution_path)?;
    Ok(fragment)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec is the worker's: bundle weights, both pinned components by the engine's own ids,
    /// resident, eager, dense.
    #[test]
    fn the_load_spec_is_the_worker_spec() {
        let spec = load_spec("bundle".into(), "te".into(), "vae".into());
        assert_eq!(spec.weights, WeightsSource::Dir("bundle".into()));
        assert_eq!(
            spec.components.get(TEXT_ENCODER_COMPONENT),
            Some(&WeightsSource::Dir("te".into()))
        );
        assert_eq!(
            spec.components.get(VAE_COMPONENT),
            Some(&WeightsSource::Dir("vae".into()))
        );
        assert_eq!(spec.components.len(), 2);
        assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
        assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(spec.precision, Precision::Bf16);
        assert_eq!(spec.quantize, None);
    }

    /// The capture requests resolve through the engine's own request boundary.
    #[test]
    fn the_capture_requests_resolve_through_the_engines_own_boundary() {
        use runtime_cuda::gen_core::supra2::{Supra2Resolved, DEFAULT_GUIDANCE};
        for count in [1, protocol::SUPRA2_IMG_PRODUCT_MAX_COUNT] {
            let resolved = Supra2Resolved::resolve(&capture_request(count))
                .unwrap_or_else(|error| panic!("count {count}: {error}"));
            assert_eq!(resolved.count, count);
            assert_eq!(resolved.steps, protocol::SUPRA2_IMG_STEPS);
            assert_eq!(resolved.guidance, DEFAULT_GUIDANCE);
            assert_eq!(resolved.seed, protocol::SUPRA2_IMG_SEED);
        }
    }

    /// Conditioning closes at the first solver step and the forward at `Decoding`.
    #[test]
    fn the_anchor_render_splits_at_the_first_step_and_decoding() {
        let step = |current| Progress::Step { current, total: 50 };
        assert_eq!(boundary_of(&step(1)), Some(Boundary::ConditioningDone));
        assert_eq!(boundary_of(&step(2)), None);
        assert_eq!(
            boundary_of(&Progress::Decoding),
            Some(Boundary::ForwardDone)
        );
    }

    /// The planned row is the one this arm captures, the dispatch const is the engine id, and the
    /// row's resident override is the only rung the provider's contract implements.
    #[test]
    fn the_planned_supra2_row_is_the_one_this_arm_captures() {
        assert_eq!(
            SUPRA2_ID,
            protocol::SUPRA2_IMG_PROVIDER,
            "the dispatch const"
        );
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let entry = &plan["anchors"]["supra2_img:bf16:candle"];
        assert_eq!(entry["provider"], protocol::SUPRA2_IMG_PROVIDER);
        assert_eq!(entry["strategy"]["rung"], "resident");
        assert_eq!(entry["strategy"]["engagedRungs"], json!(["resident"]));
        let request = json!({
            "planned": {
                "backend": LANE,
                "loadShape": entry["loadShape"],
                "calibrationFingerprint": entry["calibrationFingerprint"],
                "fixture": entry["fixture"],
                "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
                "target": {
                    "provider": entry["provider"],
                    "mode": entry["mode"],
                    "tier": "bf16",
                    "overlay": entry["overlay"],
                    "geometry": entry["geometry"],
                },
            },
        });
        assert_eq!(
            protocol::validate_supra2_plan(&request, LANE),
            Ok((256, 256))
        );
    }
}
