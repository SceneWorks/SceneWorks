#[cfg(target_os = "macos")]
use super::is_dense_te_tier;
use super::mlx_model;
use super::nvfp4_host_eligible;
#[cfg(target_os = "macos")]
use super::quantization::dense_te_requested_tier_bits;
use super::quantization::resolve_quant_gated;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::quantization::tier_key_from_resolved_dir;
use super::sampling::resolve_guidance;
use super::GenerationMetrics;
use super::ImageRequest;
use super::Quant;
use super::NVFP4_TIER;
use crate::Path;
use crate::Value;
/// Consume the streamed generation events (step / decoding / image) from the blocking
/// thread: write each finished image as an asset fact, stream progress, and poll cancel
/// ~every 2s (draining the channel after a cancel so the blocking sender never blocks).
/// Shared by the base txt2img path ([`generate_stream`]) and the Z-Image strict-pose
/// control path ([`generate_zimage_control_stream`]). `total` is the number of images
/// the job produces (the request count, or the pose count).
/// Best-effort human label for a LoRA entry in a request (epic 10402). Accepts a
/// bare string or an object with an id/name field.
pub(super) fn lora_label(value: &Value) -> Option<String> {
    match value {
        Value::String(name) => Some(name.clone()),
        Value::Object(map) => map
            .get("id")
            .or_else(|| map.get("name"))
            .or_else(|| map.get("loraId"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

/// Assemble the EFFECTIVE-settings metrics block for an image job (epic 10402,
/// sc-10406). Reports the value the run actually used — not the sparse `advanced`
/// payload where defaults are omitted: sampler/scheduler default to "default"
/// (model-native), guidanceMethod to "cfg", `use_pid` to false, steps come from
/// the observed denoise total, guidance from the resolver, quant from
/// [`effective_quant_label`]. A default-settings run is fully populated, never
/// blank — which is what makes the comparison charts meaningful (sc-10409).
pub(super) fn image_settings_metrics(
    request: &ImageRequest,
    effective_steps: Option<u32>,
    effective_guidance: Option<f32>,
    quant_label: Option<String>,
    quant_bits: Option<i64>,
    image_count: u32,
) -> GenerationMetrics {
    // MOSS edit loads its frozen BF16 transformer/text encoder contract with no quantization.
    // The generic resolver's omitted-setting Q8 default is not an executed tier for this lane.
    let (quant_label, quant_bits) =
        if sceneworks_core::moss_image::is_moss(&request.model) && request.mode == "edit_image" {
            (Some("bf16".to_owned()), None)
        } else {
            (quant_label, quant_bits)
        };
    let adv = &request.advanced;
    let string_or = |key: &str, default: &str| -> Option<String> {
        Some(
            adv.get(key)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_owned(),
        )
    };
    let number_field = |key: &str| -> Option<serde_json::Number> {
        adv.get(key)
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str()?.trim().parse().ok())
            })
            .and_then(serde_json::Number::from_f64)
    };
    let loras: Vec<String> = request.loras.iter().filter_map(lora_label).collect();
    // Two more facts beside the tier (sc-21484, epic 11037). `quant_label` answers "which variant
    // was selected"; it does NOT answer "what is this checkpoint stored in" or "what did this host
    // execute", and a stats row that reports only the tier lets a reader conclude that a
    // `nvfp4`-tier run executed NVFP4 natively on a box that in fact took the dense BF16 fallback.
    //
    // Both come from the verified source classification and the load receipt, never from
    // `quant_label` and never from the host's capability — see `checkpoint_weight_facts_host`.
    let (source_codec, execution_representation) =
        crate::checkpoint_weight_facts_host::manifest_entry_metrics_pair(
            &request.model_manifest_entry,
        );
    GenerationMetrics {
        model: (!request.model.is_empty()).then(|| request.model.clone()),
        quant_label,
        quant_bits: quant_bits.map(|bits| bits as u32),
        source_codec,
        execution_representation,
        sampler: string_or("sampler", "default"),
        scheduler: string_or("scheduler", "default"),
        scheduler_shift: number_field("schedulerShift"),
        steps: effective_steps,
        image_count: Some(image_count),
        guidance_scale: effective_guidance
            .map(|scale| scale as f64)
            .and_then(serde_json::Number::from_f64),
        true_cfg_scale: number_field("trueCfgScale"),
        guidance_method: string_or("guidanceMethod", "cfg"),
        use_pid: Some(adv.get("usePid").and_then(Value::as_bool).unwrap_or(false)),
        pid_target: adv
            .get("pidTarget")
            .and_then(Value::as_str)
            .map(str::to_owned),
        width: Some(request.width),
        height: Some(request.height),
        seed: request.seed.or_else(|| request.seeds.first().copied()),
        loras: (!loras.is_empty()).then_some(loras),
        ..Default::default()
    }
}

/// The effective quant label + bit count for a request (epic 10402): the Krea
/// INT8-ConvRot tier, then dense-TE turnkey tiers (which `resolve_quant` reports
/// as bf16 to keep the dense TE full-precision), else `resolve_quant`.
///
/// **The NVFP4 arm matches the VARIANT, not the bit count (sc-11042, epic 11037 SC#5)** — the image
/// lane's instance of the same footgun `video_quant_label` carries, and it fails in the opposite
/// direction. This maps [`resolve_quant`]'s *bits* onto a label, and NVFP4's bits are deliberately
/// `None` (~4.5 EFFECTIVE bits/weight — `Some(4)` would alias it onto q4), so a bits-only match would
/// drop NVFP4 into the `_ => bf16` arm and stamp an NVFP4 render as **`"bf16"`**: a full-precision
/// label on a 4-bit render, the inverse mislabel but the same SC#5 violation. Matching the variant is
/// what makes both directions impossible. The tier's own arm is placed alongside int8-convrot's — both
/// are tiers with no honest integer width, and both report `None` bits for that reason.
///
/// **The label describes what RAN — so it is disk-aware, not just host-aware (sc-11042).** `tier_dir`
/// is the RESOLVED tier subdir, and NVFP4 is labelled only when that dir IS the `nvfp4/` one (the
/// [`nvfp4_selected`] third half). Host-awareness alone was NOT enough: `standard_tier_subdir`
/// independently falls back to q8 when the `nvfp4/` dir is absent — the shipping case on every model
/// today, since sc-11043 has not converted a tier yet — so an explicit pick on a Blackwell host
/// recorded `"nvfp4"` on a render whose weights were **q8**. Same SC#5 creative-choice aliasing this
/// tier exists to eliminate, merely displaced out of selection and into telemetry. Reading the label
/// off the resolver's own output is what makes the two agree by construction.
#[cfg(native_lane)]
pub(super) fn effective_quant_label(
    request: &ImageRequest,
    tier_dir: Option<&Path>,
) -> (Option<String>, Option<i64>) {
    effective_quant_label_gated(request, nvfp4_host_eligible(), tier_dir)
}

/// [`effective_quant_label`] with the NVFP4 **host** gate passed in rather than probed (sc-11042).
/// Split out for testability, exactly like [`resolve_quant_gated`], which it delegates to. Production
/// has one caller.
#[cfg(native_lane)]
pub(super) fn effective_quant_label_gated(
    request: &ImageRequest,
    nvfp4_host: bool,
    tier_dir: Option<&Path>,
) -> (Option<String>, Option<i64>) {
    if request
        .advanced
        .get("convRot")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return (Some("int8-convrot".to_owned()), None);
    }
    #[cfg(target_os = "macos")]
    if is_dense_te_tier(request) {
        return match dense_te_requested_tier_bits(request) {
            Some(8) => (Some("q8".to_owned()), Some(8)),
            Some(4) => (Some("q4".to_owned()), Some(4)),
            _ => (Some("bf16".to_owned()), None),
        };
    }
    // FLUX.1's generic Candle route accepts only an exact packed q4/q8 directory. Its omitted
    // selection defaults to q4 (rather than the shared resolver's q8 default), so derive the
    // LoadSpec/asset telemetry from that resolved directory instead of re-deriving a different
    // request default here.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if matches!(request.model.as_str(), "flux_schnell" | "flux_dev") {
        if let Some(tier) = tier_dir.and_then(tier_key_from_resolved_dir) {
            return match tier {
                "q4" => (Some("q4".to_owned()), Some(4)),
                "q8" => (Some("q8".to_owned()), Some(8)),
                _ => unreachable!("FLUX.1 generic Candle route resolves only q4/q8"),
            };
        }
    }
    let (selected_quant, selected_bits) = resolve_quant_gated(request, nvfp4_host, tier_dir);
    let resolved = match (selected_quant, selected_bits) {
        // The distinct NVFP4 tier, matched on the VARIANT (see the note above): its bit count is
        // `None`, so a bits-only match would silently label it "bf16".
        (Some(Quant::Nvfp4), _) => (Some(NVFP4_TIER.to_owned()), None),
        (_, Some(8)) => (Some("q8".to_owned()), Some(8)),
        (_, Some(4)) => (Some("q4".to_owned()), Some(4)),
        _ => (Some("bf16".to_owned()), None),
    };
    let Some(selected) = selected_quant else {
        return resolved;
    };
    let Some(model) = mlx_model(&request.model) else {
        return resolved;
    };
    let active = model
        .descriptor
        .capabilities
        .component_precision_floors
        .iter()
        .copied()
        .filter(|floor| floor.applies_to(selected))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return resolved;
    }
    let selected_label = match selected {
        Quant::Q4 => "q4",
        Quant::Q8 => "q8",
        Quant::Nvfp4 => NVFP4_TIER,
        Quant::Fp8Rowwise => "fp8r",
    };
    let floors = active
        .iter()
        .map(|floor| {
            let resident = match floor.resident_tier {
                Quant::Q4 => "q4",
                Quant::Q8 => "q8",
                Quant::Nvfp4 => NVFP4_TIER,
                Quant::Fp8Rowwise => "fp8r",
            };
            format!("{}:{resident}", floor.component.as_str())
        })
        .collect::<Vec<_>>()
        .join(",");
    (
        Some(format!("{selected_label}+[{floors}]")),
        None, // A mixed component profile has no honest single bit width.
    )
}

/// Resolve quant + guidance with the generation's own rules and assemble the
/// effective-settings metrics for an image job (epic 10402, sc-10406). A build
/// with neither the MLX nor candle backend reports quant/guidance as none.
#[cfg(native_lane)]
pub(super) fn build_image_metrics(
    request: &ImageRequest,
    effective_steps: Option<u32>,
    image_count: u32,
    tier_dir: Option<&Path>,
) -> GenerationMetrics {
    let (quant_label, quant_bits) = effective_quant_label(request, tier_dir);
    let guidance = mlx_model(&request.model).and_then(|model| resolve_guidance(request, &model));
    image_settings_metrics(
        request,
        effective_steps,
        guidance,
        quant_label,
        quant_bits,
        image_count,
    )
}
#[cfg(not(native_lane))]
pub(super) fn build_image_metrics(
    request: &ImageRequest,
    effective_steps: Option<u32>,
    image_count: u32,
    _tier_dir: Option<&Path>,
) -> GenerationMetrics {
    image_settings_metrics(request, effective_steps, None, None, None, image_count)
}

#[cfg(native_lane)]
#[cfg(test)]
#[path = "tests/metrics_settings_tests.rs"]
mod metrics_settings_tests;
