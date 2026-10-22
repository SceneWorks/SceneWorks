//! Native registry FLUX3/CUDA measurement producer. No capture runs merely by
//! linking this module. Publisher device floors and receipt checks remain the
//! native loader's typed refusals, never a fallback to a different numeric tier.
use super::flux3_common as common;
use super::*;

pub(super) fn run(request: &Value) -> Result<Value, String> {
    let plan = common::plan(request, "candle")?;
    let spec = common::spec("candle", &plan)?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let actual_spec = common::actual_spec_observation(&plan, &spec)?;
    let generator = catalog
        .media()
        .load(plan.checkpoint.provider, &spec)
        .map_err(|error| format!("load native FLUX3 {} (no substitution): {error}", plan.tier))?;
    vram.end_load(load_sample);
    let context = common::admit(generator.as_ref(), &plan, request)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded FLUX3 has no contract".to_owned())?;
    let strategy = measured_strategy(
        request,
        &plan.selection,
        &contract.engaged_composition(plan.selection.strategy),
    )?;
    let run_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut conditioning = None;
    let mut denoise = None;
    let mut pup = None;
    let result = common::render(&plan, generator.as_ref(), &context, &mut |next| {
        let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
        match next {
            protocol::ReferencePhase::Conditioning => pup = peak,
            protocol::ReferencePhase::Denoise => conditioning = peak,
            protocol::ReferencePhase::Decode => denoise = peak,
        }
        phase_sample = Some(vram.phase());
    });
    let decode = phase_sample.take().map(|sample| vram.end_observed(sample));
    vram.end_gen(run_sample);
    let output = result?;
    let bytes = |peak: Option<f64>, phase| {
        peak.map(decimal_gb_to_bytes)
            .ok_or_else(|| format!("FLUX3 omitted the {phase} memory boundary"))
    };
    let conditioning = bytes(conditioning, "conditioning")?;
    let denoise = bytes(denoise, "denoise")?;
    let decode = bytes(decode, "decode")?;
    let pup = pup.map(decimal_gb_to_bytes);
    let overall = conditioning
        .max(pup.unwrap_or(0))
        .max(denoise)
        .max(decode)
        .max(decimal_gb_to_bytes(vram.report().peak_gb));
    common::fragment(request, "candle", &plan, strategy, json!({
        "promptUpsampling": pup.map(cuda_phase_metrics),
        "conditioning": cuda_phase_metrics(conditioning),
        "denoise": cuda_phase_metrics(denoise),
        "decode": cuda_phase_metrics(decode),
        "overall": cuda_phase_metrics(overall),
    }), protocol::diagnostics("memory-candle-adapter:flux3-klein-native", "executed",
        ["CUDA device-memory delta above the stable idle baseline; no allocator/RSS values or promotion-quality claims".to_owned()], [
            ("conditioningDevicePeakDelta", "bytes", conditioning),
            ("denoiseDevicePeakDelta", "bytes", denoise),
            ("decodeDevicePeakDelta", "bytes", decode),
            ("overallDevicePeakDelta", "bytes", overall),
        ]), output).and_then(|fragment| common::attach_loaded_spec(fragment, actual_spec, generator.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flux3_candle_fingerprints_bind_the_native_engine() {
        use runtime_cuda::providers::flux3::memory_strategy as engine;
        for tier in ["bf16", "q4", "q8", "fp8r", "nvfp4"] {
            assert_eq!(
                common::fingerprint("candle", tier).unwrap(),
                engine::tier_fingerprint(
                    common::PROVIDER,
                    engine::MEMORY_CALIBRATION_FINGERPRINT,
                    common::quant("candle", tier).unwrap()
                )
                .unwrap()
            );
        }
    }
    #[test]
    fn flux3_base_fingerprints_come_from_its_distinct_native_contract() {
        use runtime_cuda::providers::flux3::memory_strategy as engine;
        let spec = common::load_spec(
            "candle",
            "bf16",
            PathBuf::from("unused"),
            PathBuf::from("unused"),
        )
        .unwrap();
        let contract =
            engine::weights_free_memory_strategy_contract(common::BASE_PROVIDER, &spec).unwrap();
        assert_eq!(contract.provider_id, common::BASE_PROVIDER);
        let base = contract.calibration.unwrap().fingerprint;
        for tier in ["bf16", "q4", "q8", "fp8r", "nvfp4"] {
            assert_eq!(
                common::fingerprint_for("candle", tier, common::BASE_PROVIDER).unwrap(),
                engine::tier_fingerprint(
                    common::BASE_PROVIDER,
                    &base,
                    common::quant("candle", tier).unwrap()
                )
                .unwrap()
            );
        }
    }
}
