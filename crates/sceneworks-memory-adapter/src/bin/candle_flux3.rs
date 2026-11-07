//! Native registry FLUX3/CUDA measurement producer. No capture runs merely by
//! linking this module. Publisher device floors and receipt checks remain the
//! native loader's typed refusals, never a fallback to a different numeric tier.
use super::flux3_common as common;
use super::*;

#[cfg(target_os = "windows")]
#[path = "flux3_owned_process_memory.rs"]
mod owned_process_memory;

pub(super) fn run(request: &Value) -> Result<Value, String> {
    #[cfg(target_os = "windows")]
    {
        run_owned_process(request)
    }
    #[cfg(not(target_os = "windows"))]
    {
        run_device_delta(request)
    }
}

#[cfg(not(target_os = "windows"))]
fn run_device_delta(request: &Value) -> Result<Value, String> {
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

#[cfg(target_os = "windows")]
fn run_owned_process(request: &Value) -> Result<Value, String> {
    use owned_process_memory::OwnedProcessMemory;
    use std::cell::RefCell;
    let plan = common::plan(request, "candle")?;
    let spec = common::spec("candle", &plan)?;
    // Acquire the provider's own safe device seam in THIS process before counter READY.
    // No weights or generation occur; retain its context baseline in absolute accounting.
    let _device = runtime_cuda::media::default_device()
        .map_err(|error| format!("initialize owned FLUX3 CUDA context: {error}"))?;
    let sampler = RefCell::new(OwnedProcessMemory::start()?);
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let actual_spec = common::actual_spec_observation(&plan, &spec)?;
    let generator = catalog
        .media()
        .load(plan.checkpoint.provider, &spec)
        .map_err(|error| format!("load native FLUX3 {} (no substitution): {error}", plan.tier))?;
    let context = common::admit(generator.as_ref(), &plan, request)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or("loaded FLUX3 has no contract")?;
    let strategy = measured_strategy(
        request,
        &plan.selection,
        &contract.engaged_composition(plan.selection.strategy),
    )?;
    let pup = request
        .pointer("/resourceCase/generation/pup")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    sampler.borrow_mut().boundary(Some(if pup {
        "promptUpsampling"
    } else {
        "conditioning"
    }))?;
    let output = common::render_monitored(
        &plan,
        generator.as_ref(),
        &context,
        &mut |next| {
            let phase = match next {
                protocol::ReferencePhase::Conditioning => "conditioning",
                protocol::ReferencePhase::Denoise => "denoise",
                protocol::ReferencePhase::Decode => "decode",
            };
            sampler.borrow_mut().boundary(Some(phase)).map(|_| ())
        },
        &mut || sampler.borrow_mut().check_health(),
    )?;
    let observed = sampler.into_inner().finish()?;
    let peak = |phase: &str, metric: &str| {
        observed["phases"][phase][metric]
            .as_u64()
            .ok_or_else(|| format!("owned FLUX3 accounting omitted {phase}.{metric}"))
    };
    let diagnostics = protocol::diagnostics("memory-candle-adapter:flux3-klein-native", "executed",
        ["Sampled Windows own-process dedicated/shared/committed accounting; physical residency and CUDA allocator activity are unavailable; no exclusive throughput or idle-device delta claim".to_owned()], [
            ("conditioningOwnedDedicatedAccountingPeak", "bytes", peak("conditioning", "dedicatedPeakBytes")?),
            ("denoiseOwnedDedicatedAccountingPeak", "bytes", peak("denoise", "dedicatedPeakBytes")?),
            ("decodeOwnedDedicatedAccountingPeak", "bytes", peak("decode", "dedicatedPeakBytes")?),
        ]);
    common::fragment(
        request,
        "candle",
        &plan,
        strategy,
        observed,
        diagnostics,
        output,
    )
    .and_then(|fragment| common::attach_loaded_spec(fragment, actual_spec, generator.as_ref()))
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
