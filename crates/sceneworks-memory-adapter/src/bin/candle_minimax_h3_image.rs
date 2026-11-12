//! Physical Candle/CUDA capture arm for MiniMax-H3-Image (SW-7100, feature SW-F363).
//!
//! The Candle twin of `mlx_minimax_h3_image.rs`: the TOTAL pipeline the worker runs — MiniMax-H3's
//! text encoder, the image DiT and the image VAE, staged by the engine — at the anchor request (a
//! 1024x1024 still, seed 7100, the fixed six steps, guidance 1.0), loaded through
//! `runtime_cuda::catalog().media()` with the spec the worker's
//! `image_jobs::minimax_h3_image::attach_components` builds (the upstream snapshot as
//! `LoadSpec::weights`, the tier's quant, H3's same-tier text encoder and, for a packed tier, its
//! DiT), under the provider's own safety check and request scope. Repositories and revisions are
//! the catalog's own rows, so a `pendingArtifact` tier is refused by name before any env var.
//!
//! Every phase figure is the `nvidia-smi` device-memory peak over that phase's window above the
//! probe's idle baseline, and each phase carries its wall time. The contract publishes its
//! calibration identity on every load ([`plan_fingerprint`], the engine constant), and the plan
//! row runs at this lane's default staged-residency rung, which the contract implements.

use super::*;
use runtime_cuda::gen_core::MemoryCalibrationIdentity;
use std::time::Instant;

const LANE: &str = "candle";

/// The env family the catalog runner exports for this arm (`PROVIDER_FAMILIES.minimax_h3_image`).
pub(super) const ENVS: protocol::MinimaxH3ImageEnvs = protocol::MinimaxH3ImageEnvs {
    upstream_repository_env: "SCENEWORKS_MINIMAX_H3_IMAGE_UPSTREAM_REPOSITORY",
    upstream_revision_env: "SCENEWORKS_MINIMAX_H3_IMAGE_UPSTREAM_REVISION",
    upstream_root_env: "SCENEWORKS_MINIMAX_H3_IMAGE_UPSTREAM_ROOT",
    tier_repository_env: "SCENEWORKS_MINIMAX_H3_IMAGE_REPOSITORY",
    tier_revision_env: "SCENEWORKS_MINIMAX_H3_IMAGE_REVISION",
    tier_root_env: "SCENEWORKS_MINIMAX_H3_IMAGE_ROOT",
    text_encoder_env: "SCENEWORKS_MINIMAX_H3_IMAGE_COMPONENT_TEXT_ENCODER",
};

/// The identity the Candle plan rows name: the engine's own constant, one for every tier.
pub(super) fn plan_fingerprint() -> &'static str {
    runtime_cuda::providers::minimax_h3::image_memory_strategy::IMAGE_MEMORY_CALIBRATION_FINGERPRINT
}

use protocol::elapsed_millis as millis;

fn context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    let edge = protocol::MINIMAX_H3_IMAGE_EDGE;
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: LoadShape::EagerMaterialization,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: edge,
            height: edge,
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
        evidence_revision: format!("sw-7100@{}", protocol::INFERENCE_PIN),
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
    let mut scope = open_memory_request_scope(generator, scoped, "minimax_h3_image")?;
    let mut request = request.clone();
    scope
        .configure_request(&mut request)
        .map_err(|error| format!("configure the minimax_h3_image request scope: {error}"))?;
    let outcome = generator.generate(&request, &mut |progress| {
        if let Some(crossed) = boundary_of(&progress) {
            on_boundary(crossed);
        }
    });
    match outcome {
        Ok(GenerationOutput::Images(images)) => {
            scope
                .finish(MemoryRunOutcome::Complete)
                .map_err(|error| format!("finish the minimax_h3_image request scope: {error}"))?;
            Ok(images.len())
        }
        Ok(_) => {
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: "non-image output".to_owned(),
            });
            Err("minimax_h3_image returned a non-image output".to_owned())
        }
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            Err(format!(
                "minimax_h3_image measured generation failed: {message}"
            ))
        }
    }
}

/// The `candle:minimax_h3_image` arm.
pub(super) fn run(request: &Value) -> Result<Value, String> {
    // Provider, mode, tier, the still, overlay, load shape, fixture and identity — all before an
    // env var or a weight.
    let planned_fingerprint = plan_fingerprint();
    let (width, height, tier) =
        protocol::validate_minimax_h3_image_plan(request, LANE, planned_fingerprint)?;
    let selection = planned_selection(request)?;
    let inputs = protocol::minimax_h3_image_inputs(ENVS, tier)?;
    let spec = protocol::minimax_h3_image_load_spec(&inputs);

    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let load_started = Instant::now();
    let generator = runtime_cuda::catalog()
        .map_err(|error| format!("build CUDA catalog: {error}"))?
        .media()
        .load(protocol::MINIMAX_H3_IMAGE_PROVIDER, &spec)
        .map_err(|error| format!("load real minimax_h3_image provider: {error}"))?;
    let load_millis = millis(load_started);
    vram.end_load(load_sample);

    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded minimax_h3_image exposed no memory-strategy contract".to_owned())?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned minimax_h3_image provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        "the loaded minimax_h3_image contract published no calibration identity".to_owned()
    })?;
    if calibration.fingerprint != planned_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
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
        generator.memory_strategy_safety_check(&admitted(planned_fingerprint, hardware_bytes, 1)),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "minimax_h3_image admission rejected a fitting probe budget; the scenario rejections \
             below would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        generator.memory_strategy_safety_check(&admitted(planned_fingerprint, 0, 1)),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("minimax_h3_image admission accepted an unknown/zero memory budget".to_owned());
    }
    if !matches!(
        generator.memory_strategy_safety_check(&admitted(
            "stale-minimax-h3-image-fingerprint",
            hardware_bytes,
            1
        )),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("minimax_h3_image admission accepted stale calibration evidence".to_owned());
    }

    // The anchor render: three device-peak phases, each with its wall time.
    let scoped = admitted(planned_fingerprint, hardware_bytes, 1);
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
        &protocol::minimax_h3_image_capture_request(),
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
        return Err(
            "the measured minimax_h3_image render did not return exactly one image".to_owned(),
        );
    }

    let phase_bytes = |peak: Option<f64>, phase: &str| {
        peak.map(decimal_gb_to_bytes)
            .ok_or_else(|| format!("minimax_h3_image did not expose a {phase} boundary"))
    };
    let conditioning_bytes = phase_bytes(conditioning_peak_gb, "conditioning")?;
    let denoise_bytes = phase_bytes(denoise_peak_gb, "forward")?;
    let decode_bytes = phase_bytes(decode_peak_gb, "decode")?;
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);

    let blocker = concat!(
        "SW-7100 anchor capture measures exact per-phase device memory and the strategy identity ",
        "for the Candle MiniMax-H3-Image lane; it stays gated because this run does not repeat the ",
        "full promotion-quality, negative-mutation and lifecycle scenario suite"
    );
    let execution_path = "the minimax_h3_image staged text-to-image path";
    let mut fragment = json!({
        "status": "gated",
        "strategy": strategy,
        "loadShape": load_shape_key(LoadShape::EagerMaterialization),
        "artifact": inputs.artifact_json(),
        "sweep": protocol::reference_sweep(request, "passed")?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": blocker },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded MiniMax-H3-Image admission check rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded MiniMax-H3-Image admission check rejected a mutated calibration fingerprint" },
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
            "memory-candle-adapter:minimax-h3-image-anchor",
            "executed",
            [blocker.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
                ("loadWallMilliseconds", "milliseconds", load_millis),
                ("conditioningWallMilliseconds", "milliseconds", conditioning_millis),
                ("denoiseWallMilliseconds", "milliseconds", denoise_millis),
                ("decodeWallMilliseconds", "milliseconds", decode_millis),
                ("renderWallMilliseconds", "milliseconds", render_millis),
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

    /// Conditioning closes at the first solver step and the forward at `Decoding`.
    #[test]
    fn the_anchor_render_splits_at_the_first_step_and_decoding() {
        let step = |current| Progress::Step { current, total: 6 };
        assert_eq!(boundary_of(&step(1)), Some(Boundary::ConditioningDone));
        assert_eq!(boundary_of(&step(2)), None);
        assert_eq!(
            boundary_of(&Progress::Decoding),
            Some(Boundary::ForwardDone)
        );
    }

    /// The planned rows are the ones this arm captures, the dispatch const is the engine id, and
    /// every row names the engine's own identity.
    ///
    /// *Mutation that reds this:* change a row's fixture, identity, tier or geometry in
    /// `config/memory-calibration-plan.json`.
    #[test]
    fn the_planned_minimax_h3_image_rows_are_the_ones_this_arm_captures() {
        assert_eq!(MINIMAX_H3_IMAGE_ID, protocol::MINIMAX_H3_IMAGE_PROVIDER);
        assert_eq!(
            protocol::MINIMAX_H3_IMAGE_PROVIDER,
            runtime_cuda::providers::minimax_h3::image_model::IMAGE_MODEL_ID
        );
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        for tier in protocol::MINIMAX_H3_IMAGE_TIERS {
            let entry = &plan["anchors"][format!("minimax_h3_image:{tier}:candle")];
            assert!(
                entry.get("strategy").is_none(),
                "the lane default is staged"
            );
            let request = json!({
                "planned": {
                    "backend": LANE,
                    "loadShape": entry["loadShape"],
                    "calibrationFingerprint": entry["calibrationFingerprint"],
                    "fixture": entry["fixture"],
                    "target": {
                        "provider": entry["provider"],
                        "mode": entry["mode"],
                        "tier": tier,
                        "overlay": entry["overlay"],
                        "geometry": entry["geometry"],
                    },
                },
            });
            assert_eq!(
                protocol::validate_minimax_h3_image_plan(&request, LANE, plan_fingerprint()),
                Ok((1024, 1024, tier))
            );
        }
    }
}
