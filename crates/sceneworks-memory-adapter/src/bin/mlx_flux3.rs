//! MLX native-registry FLUX3 producer. Windows source/format checks do not
//! establish MLX execution; actual phase figures require an explicit macOS run.
use super::flux3_common as common;
use super::*;

fn phase_now() -> PhaseMemory {
    PhaseMemory {
        active: (get_peak_memory() as u64).max(get_active_memory() as u64),
        cache: get_cache_memory() as u64,
    }
}

pub(super) fn run(request: &Value) -> Result<Value, String> {
    let plan = common::plan(request, "mlx")?;
    let spec = common::spec("mlx", &plan)?;
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(plan.checkpoint.provider, &spec)
        .map_err(|error| format!("load native FLUX3 {} (no substitution): {error}", plan.tier))?;
    let context = common::admit(generator.as_ref(), &plan, request)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded FLUX3 has no contract".to_owned())?;
    let strategy = attested_strategy(
        request,
        &plan.selection,
        &contract.engaged_composition(plan.selection.strategy),
    )?;
    clear_cache();
    reset_peak_memory();
    let mut conditioning = None;
    let mut denoise = None;
    // The native provider fences its image stream at conditioning/decoding and
    // returns materialized host images. Do not mix process RSS or CUDA counters
    // into MLX's allocator-active/cache observations.
    common::render(&plan, generator.as_ref(), &context, &mut |next| {
        match next {
            protocol::ReferencePhase::Denoise => conditioning = Some(phase_now()),
            protocol::ReferencePhase::Decode => denoise = Some(phase_now()),
            _ => {}
        }
        reset_peak_memory();
    })?;
    let decode = phase_now();
    let conditioning =
        conditioning.ok_or_else(|| "FLUX3 omitted conditioning boundary".to_owned())?;
    let denoise = denoise.ok_or_else(|| "FLUX3 omitted denoise boundary".to_owned())?;
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "FLUX3 reported a zero allocator-active phase peak; no capture attested".to_owned(),
        );
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    common::fragment(request, "mlx", &plan, strategy, json!({
        "conditioning": conditioning.json(), "denoise": denoise.json(),
        "decode": decode.json(), "overall": overall.json(),
    }), protocol::diagnostics("memory-mlx-adapter:flux3-klein-native", "executed",
        ["MLX allocator-active/cache phase observations; no RSS/device-memory substitution or promotion-quality claims".to_owned()], [
            ("conditioningActivePeak", "bytes", conditioning.active),
            ("denoiseActivePeak", "bytes", denoise.active),
            ("decodeActivePeak", "bytes", decode.active),
            ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
        ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flux3_mlx_fingerprints_bind_the_native_engine() {
        use runtime_macos::providers::flux3::memory_strategy as engine;
        // Capture uses loaded-package identities: its native affine conversion
        // receipts fix group64, while dense storage has no quantization group.
        for (tier, group) in [("bf16", None), ("q4", Some(64)), ("q8", Some(64))] {
            assert_eq!(
                common::fingerprint("mlx", tier).unwrap(),
                engine::tier_fingerprint(
                    engine::MEMORY_CALIBRATION_FINGERPRINT,
                    common::quant("mlx", tier).unwrap(),
                    group,
                )
            );
        }
    }
    #[test]
    fn flux3_base_fingerprints_come_from_its_distinct_native_contract() {
        use runtime_macos::providers::flux3::memory_strategy as engine;
        let spec = common::load_spec(
            "mlx",
            "bf16",
            PathBuf::from("unused"),
            PathBuf::from("unused"),
        )
        .unwrap();
        let contract =
            engine::weights_free_memory_strategy_contract(common::BASE_PROVIDER, &spec).unwrap();
        assert_eq!(contract.provider_id, common::BASE_PROVIDER);
        let base = contract.calibration.unwrap().fingerprint;
        for tier in ["bf16", "q4", "q8"] {
            assert_eq!(
                common::fingerprint_for("mlx", tier, common::BASE_PROVIDER).unwrap(),
                engine::tier_fingerprint(
                    &base,
                    common::quant("mlx", tier).unwrap(),
                    if tier == "bf16" { None } else { Some(64) }
                )
            );
        }
    }
}
