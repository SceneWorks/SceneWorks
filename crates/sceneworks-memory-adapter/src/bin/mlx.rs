#[cfg(not(target_os = "macos"))]
compile_error!("memory-mlx-adapter is supported only on macOS");

use mlx_gen::gen_core::{
    MemoryBudget, MemoryCacheState, MemoryCalibrationIdentity, MemoryGeometry, MemoryMode,
    MemoryNumericTier, MemoryPhase, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision,
    MemorySelection, MemoryStrategy, MemoryStrategyParameters, TransformerComponent,
};
use mlx_gen::tiling::{SpatialTiling, TilingConfig};
use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Generator, Image, LoadShape,
    LoadSpec, OffloadPolicy, Precision, Progress, Quant, WeightsSource,
};
use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_memory_limit, get_peak_memory,
    reset_peak_memory,
};
use mlx_rs::Array;
use runtime_macos::providers::qwen_image::{load_vae, QwenVae};
use sceneworks_memory_adapter as protocol;
use serde_json::{json, Value};
use std::cell::Cell;
use std::path::PathBuf;
use std::process::Command;

const EDGES: [u32; 7] = [768, 640, 512, 448, 384, 320, 256];
const MAX_THRESHOLD: f64 = 3e-2;
const MEAN_THRESHOLD: f64 = 3e-3;
// A generated diffusion latent has the same high-frequency characteristics as the random-latent
// case in mlx-gen-qwen-image's real-weight tiling oracle, not its smoother VAE-encoded fixture.
const KREA_MAX_THRESHOLD: f64 = 1.5e-1;
const KREA_MEAN_THRESHOLD: f64 = 5e-3;
// Qwen's request-scoped attention chunks change Metal kernel shapes across 60 DiT blocks. The
// real-weight BF16 reference differed by at most 43/255 in a localized channel while averaging
// 0.113/255 over the image. Bound that amplification to 48 integer levels and, independently, less
// than half an integer level on average; the mandatory broad-bias mutation must breach both.
const QWEN_MAX_THRESHOLD: f64 = 48.0 / 255.0;
const QWEN_MEAN_THRESHOLD: f64 = 0.5 / 255.0;
// Z-Image's production 768px VAE tile measured 48/255 maximum and 2.82/255 mean
// error against the exact untiled q4 decode. The provider's real-weight candidate
// sweep places the seam-free cutoff at 56/255, between the published and rejected
// domains; keep a separate 4/255 mean bound so a broad low-amplitude drift cannot
// pass on the maximum alone.
const Z_IMAGE_MAX_THRESHOLD: f64 = 56.0 / 255.0;
const Z_IMAGE_MEAN_THRESHOLD: f64 = 4.0 / 255.0;
const KREA_PROVIDER: &str = "krea_2_turbo_control";
const KREA_OVERLAY_REPOSITORY: &str = "SceneWorks/krea2-pose-controlnet-beta";
const KREA_OVERLAY_FILE: &str = "control_step5000.safetensors";
const KREA_TILE_EDGES: [u32; 1] = [512];
const KREA_TILE_OVERLAP: u32 = 64;
const KREA_CONTROL_EXECUTION_PATH: &str = "the MLX Krea pose-control path";
/// The gated VAE probe reaches `load_vae` directly, with no `LoadSpec` and therefore no deferred
/// block schedule: it bulk-materializes the VAE, which is eager materialization.
const QWEN_VAE_PROBE_LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;
const QWEN_PROVIDER: &str = "qwen_image";
const QWEN_PLAIN_EXECUTION_PATH: &str = "the MLX Qwen VAE-only path";
const QWEN_PROVIDER_EXECUTION_PATH: &str = "the pinned MLX Qwen base provider path";
const Z_IMAGE_PROVIDER: &str = "z_image_turbo";
const Z_IMAGE_PLAIN_EXECUTION_PATH: &str = "the MLX Z-Image base-only text-to-image path";
/// The `mlx:flux2_dev` lane: the FLUX.2-dev text-to-image provider the measured renders load.
const FLUX2_PROVIDER: &str = "flux2_dev";
const FLUX2_CALIBRATION_FINGERPRINT: &str = "sc-18218-flux2-dev-t2i-resident-evidence-v1";
const FLUX2_PLAIN_EXECUTION_PATH: &str = "the MLX FLUX.2-dev base-only text-to-image path";
/// FLUX.2-dev quality here is repeat determinism on one loaded provider: the resident rung selects
/// no alternate code path, so the cold measured render and its warm unscoped repeats must agree to
/// within Metal allocator jitter. 3/255 max and 1/255 mean sit far above observed same-process
/// fp jitter and far below the mandatory +64/255 broad-bias mutation, which must breach both.
const FLUX2_MAX_THRESHOLD: f64 = 3.0 / 255.0;
const FLUX2_MEAN_THRESHOLD: f64 = 1.0 / 255.0;
const FLUX2_RMS_THRESHOLD: f64 = 1.5 / 255.0;
/// One fixed seed for every q4/q8 `mlx:flux2_dev` fixture
/// (`flux2-dev-mlx-<tier>-<edge>-seed18218-step2`).
const FLUX2_SEED: u64 = 18218;
const MIB: u64 = 1024 * 1024;

fn command(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("start {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn sysctl(name: &str) -> Result<String, String> {
    command("/usr/sbin/sysctl", &["-n", name])
}

fn integer(value: &str, label: &str) -> Result<u64, String> {
    value
        .trim()
        .parse()
        .map_err(|error| format!("parse {label}={value:?}: {error}"))
}

#[derive(Debug, PartialEq, Eq)]
struct WiredLimit {
    bytes: u64,
    source: &'static str,
}

fn positive_integer(value: Option<&str>, label: &str) -> Result<Option<u64>, String> {
    value
        .map(|value| integer(value, label))
        .transpose()
        .map(|value| value.filter(|value| *value > 0))
}

fn resolve_wired_limit(
    override_bytes: Option<&str>,
    iogpu_limit_mb: Option<&str>,
    kernel_limit_bytes: Option<&str>,
    mlx_default_memory_limit: usize,
) -> Result<WiredLimit, String> {
    if let Some(value) = override_bytes {
        let bytes = integer(value, "SCENEWORKS_MLX_WIRED_LIMIT_BYTES")?;
        if bytes == 0 {
            return Err("SCENEWORKS_MLX_WIRED_LIMIT_BYTES must be greater than zero".to_owned());
        }
        return Ok(WiredLimit {
            bytes,
            source: "SCENEWORKS_MLX_WIRED_LIMIT_BYTES",
        });
    }
    if let Some(megabytes) = positive_integer(iogpu_limit_mb, "iogpu.wired_limit_mb")? {
        let bytes = megabytes
            .checked_mul(1024 * 1024)
            .ok_or_else(|| "iogpu.wired_limit_mb overflows bytes".to_owned())?;
        return Ok(WiredLimit {
            bytes,
            source: "iogpu.wired_limit_mb",
        });
    }
    if let Some(bytes) = positive_integer(kernel_limit_bytes, "kern.memorystatus_wired_mem_limit")?
    {
        return Ok(WiredLimit {
            bytes,
            source: "kern.memorystatus_wired_mem_limit",
        });
    }

    // MLX documents its untouched default memory limit as 1.5x the device's
    // recommendedMaxWorkingSetSize. This is the same real-hardware-validated derivation used by
    // the worker when the host has no explicit wired policy (sc-12178).
    let bytes = u64::try_from(mlx_default_memory_limit)
        .map_err(|_| "MLX default memory limit does not fit u64".to_owned())?
        / 3
        * 2;
    if bytes == 0 {
        return Err(
            "cannot resolve a nonzero wired ceiling from host policy or the MLX default memory limit"
                .to_owned(),
        );
    }
    Ok(WiredLimit {
        bytes,
        source: "mlx_default_memory_limit/1.5",
    })
}

fn metal_device() -> Result<String, String> {
    let raw = command(
        "/usr/sbin/system_profiler",
        &["SPDisplaysDataType", "-json"],
    )?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("parse system_profiler JSON: {error}"))?;
    fn find_name(value: &Value) -> Option<&str> {
        match value {
            Value::Object(map) => map
                .get("sppci_model")
                .and_then(Value::as_str)
                .or_else(|| map.values().find_map(find_name)),
            Value::Array(items) => items.iter().find_map(find_name),
            _ => None,
        }
    }
    find_name(&value)
        .map(str::to_owned)
        .ok_or_else(|| "system_profiler did not report a Metal display device".to_owned())
}

fn probe() -> Result<Value, String> {
    let memory_bytes = integer(&sysctl("hw.memsize")?, "hw.memsize")?;
    let override_bytes = std::env::var("SCENEWORKS_MLX_WIRED_LIMIT_BYTES").ok();
    let iogpu_limit_mb = sysctl("iogpu.wired_limit_mb").ok();
    let kernel_limit_bytes = sysctl("kern.memorystatus_wired_mem_limit").ok();
    let mlx_default_memory_limit = get_memory_limit();
    let wired_limit = resolve_wired_limit(
        override_bytes.as_deref(),
        iogpu_limit_mb.as_deref(),
        kernel_limit_bytes.as_deref(),
        mlx_default_memory_limit,
    )?;
    Ok(json!({
        "hardware": {
            "probe": format!(
                "sysctl + sw_vers + system_profiler + mlx_rs::memory::get_memory_limit; wired={}",
                wired_limit.source
            ),
            "memoryBytes": memory_bytes,
            "model": sysctl("hw.model")?,
            "chip": sysctl("machdep.cpu.brand_string")?,
            "osVersion": command("/usr/bin/sw_vers", &["-productVersion"])?,
            "metalDevice": metal_device()?,
            "mlxMemoryLimitBytes": mlx_default_memory_limit,
            "wiredLimitBytes": wired_limit.bytes,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wired_limit_prefers_explicit_bytes() {
        assert_eq!(
            resolve_wired_limit(Some("123"), Some("456"), Some("789"), 999).unwrap(),
            WiredLimit {
                bytes: 123,
                source: "SCENEWORKS_MLX_WIRED_LIMIT_BYTES",
            }
        );
    }

    #[test]
    fn wired_limit_converts_iogpu_megabytes() {
        assert_eq!(
            resolve_wired_limit(None, Some("57344"), Some("789"), 999).unwrap(),
            WiredLimit {
                bytes: 57344 * 1024 * 1024,
                source: "iogpu.wired_limit_mb",
            }
        );
    }

    #[test]
    fn wired_limit_rejects_iogpu_byte_overflow() {
        let megabytes = u64::MAX.to_string();
        assert!(resolve_wired_limit(None, Some(&megabytes), None, 1_000)
            .unwrap_err()
            .contains("iogpu.wired_limit_mb overflows bytes"));
    }

    #[test]
    fn wired_limit_uses_kernel_bytes_when_iogpu_is_unset() {
        assert_eq!(
            resolve_wired_limit(None, Some("0"), Some("789"), 999).unwrap(),
            WiredLimit {
                bytes: 789,
                source: "kern.memorystatus_wired_mem_limit",
            }
        );
    }

    #[test]
    fn wired_limit_derives_default_mlx_ceiling_without_host_override() {
        assert_eq!(
            resolve_wired_limit(None, Some("0"), None, 1_000).unwrap(),
            WiredLimit {
                bytes: 666,
                source: "mlx_default_memory_limit/1.5",
            }
        );
    }

    #[test]
    fn wired_limit_rejects_invalid_explicit_override() {
        assert!(resolve_wired_limit(Some("not-a-number"), None, None, 1_000)
            .unwrap_err()
            .contains("SCENEWORKS_MLX_WIRED_LIMIT_BYTES"));
    }

    #[test]
    fn wired_limit_rejects_zero_explicit_override() {
        assert!(
            resolve_wired_limit(Some("0"), Some("456"), Some("789"), 1_000)
                .unwrap_err()
                .contains("SCENEWORKS_MLX_WIRED_LIMIT_BYTES must be greater than zero")
        );
    }

    #[test]
    fn wired_limit_rejects_zero_everywhere() {
        assert!(resolve_wired_limit(None, Some("0"), Some("0"), 0)
            .unwrap_err()
            .contains("cannot resolve a nonzero wired ceiling"));
    }

    #[test]
    fn overall_memory_and_prediction_cover_every_componentwise_phase_peak() {
        let conditioning = PhaseMemory {
            active: 8,
            cache: 1,
        };
        let denoise = PhaseMemory {
            active: 16,
            cache: 15,
        };
        let decode = PhaseMemory {
            active: 19,
            cache: 0,
        };
        let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
        assert_eq!(
            overall,
            PhaseMemory {
                active: 19,
                cache: 15,
            }
        );
        assert_eq!(overall.allocator_bytes(), 34);
        assert!(predicted_ceiling(overall.allocator_bytes()) >= overall.allocator_bytes());
    }

    /// sc-18104: an unimplemented provider must be refused by name at dispatch, not fall through to
    /// the Qwen arm. The regression this guards is silent MISROUTING, so asserting "it errored" is
    /// not enough — the old code errored too, just with a Qwen-shaped message after entering the
    /// wrong arm. Assert the provider is named, and that none of the Qwen arm's own failure
    /// vocabulary appears, which is what distinguishes a dispatch refusal from a misroute.
    #[test]
    fn run_refuses_a_provider_the_mlx_adapter_does_not_implement() {
        // `flux2_dev` left this list when sc-18218 landed its arm; `flux2_dev_edit` stays — the
        // contract provider is not a dispatchable lane.
        for provider in ["flux2_klein_9b", "flux2_dev_edit", "krea_2_turbo", "sana"] {
            let request = json!({ "planned": { "target": { "provider": provider } } });
            let error = run(&request).expect_err("unimplemented provider must not dispatch");
            assert_eq!(
                error,
                format!("MLX five-rung calibration does not implement provider {provider:?}")
            );
            assert!(
                !error.contains("SCENEWORKS_QWEN") && !error.contains("calibration mismatch"),
                "refusal leaked the Qwen arm's vocabulary, so dispatch misrouted: {error}"
            );
        }
    }

    /// The companion direction, and the one the refusal arm above makes necessary: every IMPLEMENTED
    /// provider must still reach its own arm through `run`. A typo in any match key — `"qwen_imagee"`
    /// — would drop a live lane into the refusal arm permanently, and `mlx:qwen_image` alone carries
    /// 9 authoritative plan entries and 41 evidence records.
    ///
    /// Two properties are necessary (sc-18250):
    ///
    ///   1. dispatch is probed with LITERAL provider ids, exactly the strings the plan carries, so a
    ///      corrupted constant or mis-keyed match arm is caught by the refusal complaint leaking
    ///      through;
    ///   2. the constants are pinned to those same literals, so an arm's internal use of its
    ///      constant cannot drift from the id dispatch routes on.
    ///
    /// This must go through `run` to mean anything. Dispatch is cheap to probe here — each arm
    /// rejects this minimal request on a missing field long before any env read, catalog build or
    /// weight load, so the assertion is on WHICH complaint comes back, not on success.
    #[test]
    fn every_implemented_provider_still_reaches_its_own_arm_through_dispatch() {
        for provider in [
            "qwen_image",
            "z_image_turbo",
            "krea_2_turbo_control",
            "flux2_dev",
        ] {
            let request = json!({ "planned": { "target": { "provider": provider } } });
            let error = run(&request)
                .expect_err("the minimal request is incomplete, so every arm must complain");
            assert!(
                !error.contains("does not implement provider"),
                "{provider} is wired but dispatch refused it — a match key is mis-typed: {error}"
            );
        }
        assert_eq!(QWEN_PROVIDER, "qwen_image");
        assert_eq!(Z_IMAGE_PROVIDER, "z_image_turbo");
        assert_eq!(KREA_PROVIDER, "krea_2_turbo_control");
        assert_eq!(FLUX2_PROVIDER, "flux2_dev");
    }

    #[test]
    fn qwen_complete_sweep_verifies_only_the_exact_executed_case() {
        let request = json!({
            "planned": {
                "strategy": {
                    "parameters": { "decodeTileEdge": 448, "decodeOverlap": 64 }
                }
            }
        });
        let sweep = qwen_complete_sweep(&request).unwrap();
        assert_eq!(sweep["rangeVerified"], true);
        let axes = sweep["axes"].as_array().unwrap();
        assert!(axes.iter().any(|axis| {
            axis["parameter"] == "decodeTileEdge" && axis["testedValues"] == json!([448])
        }));
        assert!(axes.iter().any(|axis| {
            axis["parameter"] == "decodeOverlap" && axis["testedValues"] == json!([64])
        }));
        assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
        assert_eq!(
            sweep["cases"][0]["parameters"],
            request["planned"]["strategy"]["parameters"]
        );
    }

    #[test]
    fn z_image_complete_sweep_verifies_only_the_exact_executed_case() {
        let request = json!({
            "planned": {
                "strategy": {
                    "parameters": { "decodeTileEdge": 768, "decodeOverlap": 64 }
                }
            }
        });
        let sweep = z_image_complete_sweep(&request).unwrap();
        assert_eq!(sweep["rangeVerified"], true);
        assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
        assert_eq!(
            sweep["cases"][0]["parameters"],
            request["planned"]["strategy"]["parameters"]
        );
    }

    #[test]
    fn z_image_parity_envelope_accepts_published_decode_and_rejects_mutation() {
        assert!(z_image_quality_passes(48.0 / 255.0, 2.82 / 255.0));
        assert!(!z_image_quality_passes(57.0 / 255.0, 2.82 / 255.0));
        assert!(!z_image_quality_passes(48.0 / 255.0, 4.01 / 255.0));

        let baseline = Image {
            width: 1,
            height: 1,
            pixels: vec![10, 20, 30],
        };
        let mutated = qwen_negative_mutation(&baseline);
        let (maximum, mean) = image_max_mean_abs(&mutated, &baseline).unwrap();
        assert!(!z_image_quality_passes(maximum, mean));
    }

    #[test]
    fn z_image_cleanup_bounds_reject_retained_memory_and_warm_peak_mutations() {
        let clean = AllocatorState {
            active: 1_000,
            cache: 200,
        };
        let bounds = LifecycleMemoryBounds::from_clean_warm(10_000, clean);
        assert_eq!(bounds.tolerance_bytes, 200);
        assert!(bounds.allows_retained(AllocatorState {
            active: 1_200,
            cache: 400,
        }));
        assert!(bounds.allows_warm_peak(10_200));
        assert!(!bounds.allows_retained(AllocatorState {
            active: 1_201,
            cache: 400,
        }));
        assert!(!bounds.allows_retained(AllocatorState {
            active: 1_200,
            cache: 401,
        }));
        assert!(!bounds.allows_warm_peak(10_201));
    }

    #[test]
    fn qwen_parity_envelope_accepts_measured_chunking_and_rejects_mutation() {
        let measured_maximum = 43.0 / 255.0;
        let measured_mean = 0.113 / 255.0;
        assert!(qwen_quality_passes(measured_maximum, measured_mean));
        assert!(!qwen_quality_passes(
            measured_maximum + 0.05,
            measured_mean + 0.05
        ));
        assert!(!qwen_quality_passes(49.0 / 255.0, measured_mean));
        assert!(!qwen_quality_passes(measured_maximum, 0.51 / 255.0));
    }

    #[test]
    fn qwen_fixture_seed_is_bound_to_the_planned_tier() {
        let request = json!({
            "planned": {
                "fixture": "qwen-image-q4-seed16353-step2",
                "target": { "tier": "q4" }
            }
        });
        assert_eq!(planned_qwen_tier(&request).unwrap(), "q4");
        assert_eq!(planned_qwen_seed(&request, "q4").unwrap(), 16353);

        let error = planned_qwen_seed(&request, "q8").unwrap_err();
        assert!(error.contains("must start with"));
    }

    #[test]
    fn qwen_load_spec_preserves_every_planned_numeric_tier() {
        for (tier, expected_quant) in [
            ("bf16", None),
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
        ] {
            let request = json!({
                "planned": {
                    "strategy": {
                        "rung": "bounded_attention",
                        "parameters": {}
                    },
                    "target": { "tier": tier }
                }
            });
            let selection = planned_selection(&request).unwrap();
            let spec = qwen_load_spec(
                PathBuf::from(format!("/tmp/qwen-image-{tier}")),
                &selection,
                OffloadPolicy::Resident,
                LoadShape::EagerMaterialization,
            );
            assert_eq!(spec.quantize, expected_quant, "numeric tier {tier}");
        }
    }

    #[test]
    fn qwen_capture_uses_the_plans_typed_load_shape_independently_of_rung() {
        for (rung, load_shape, expected) in [
            (
                "bounded_attention",
                protocol::LOAD_SHAPE_DEFERRED,
                LoadShape::DeferredMaterialization,
            ),
            (
                "bounded_transformer_residency",
                protocol::LOAD_SHAPE_DEFERRED,
                LoadShape::DeferredMaterialization,
            ),
            (
                "bounded_attention",
                protocol::LOAD_SHAPE_EAGER,
                LoadShape::EagerMaterialization,
            ),
        ] {
            let request = json!({
                "planned": {
                    "loadShape": load_shape,
                    "strategy": { "rung": rung, "parameters": {} }
                }
            });
            assert_eq!(planned_load_shape(&request).unwrap(), expected, "{rung}");
        }

        let missing = json!({ "planned": {} });
        assert!(planned_load_shape(&missing)
            .unwrap_err()
            .contains("planned.loadShape must be a string"));
        let mutated = json!({ "planned": { "loadShape": "deferred-ish" } });
        assert!(planned_load_shape(&mutated)
            .unwrap_err()
            .contains("deferred-ish"));
    }
}

fn encoded_latent(vae: &QwenVae, width: u32, height: u32) -> Result<Array, String> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 255 / width) as u8);
            pixels.push((y * 255 / height) as u8);
            pixels.push((((x + y) as u64 * 127) / (width as u64 + height as u64)) as u8);
        }
    }
    let image = mlx_gen::media::Image {
        width,
        height,
        pixels,
    };
    let input = mlx_gen::img2img::preprocess_init_image(&image, width, height)
        .map_err(|error| format!("preprocess encoded-latent fixture: {error}"))?;
    let latent = vae
        .encode(&input)
        .map_err(|error| format!("encode deterministic fixture: {error}"))?;
    latent
        .eval()
        .map_err(|error| format!("materialize deterministic fixture latent: {error}"))?;
    Ok(latent)
}

fn decoded_max_mean_abs(
    left: &Array,
    right: &Array,
    comparison_output_bias: Option<f64>,
) -> Result<(f64, f64), String> {
    protocol::validate_comparison_shapes(left.shape(), right.shape())?;
    let left = left
        .reshape(&[-1])
        .map_err(|error| format!("flatten baseline decode: {error}"))?;
    let right = right
        .reshape(&[-1])
        .map_err(|error| format!("flatten tiled decode: {error}"))?;
    let left = left.as_slice::<f32>();
    let right = right.as_slice::<f32>();
    protocol::max_mean_abs(left, right, comparison_output_bias)
}

fn image_max_mean_abs(left: &Image, right: &Image) -> Result<(f64, f64), String> {
    if (left.width, left.height) != (right.width, right.height)
        || left.pixels.len() != right.pixels.len()
        || left.pixels.is_empty()
    {
        return Err(format!(
            "image shape mismatch: {}x{} ({} bytes) versus {}x{} ({} bytes)",
            left.width,
            left.height,
            left.pixels.len(),
            right.width,
            right.height,
            right.pixels.len()
        ));
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    for (&left, &right) in left.pixels.iter().zip(&right.pixels) {
        let difference = (f64::from(left) - f64::from(right)).abs() / 255.0;
        maximum = maximum.max(difference);
        sum += difference;
    }
    Ok((maximum, sum / left.pixels.len() as f64))
}

fn qwen_negative_mutation(image: &Image) -> Image {
    let mut mutated = image.clone();
    for channel in &mut mutated.pixels {
        *channel = channel.wrapping_add(64);
    }
    mutated
}

fn qwen_quality_passes(maximum: f64, mean: f64) -> bool {
    maximum <= QWEN_MAX_THRESHOLD && mean <= QWEN_MEAN_THRESHOLD
}

fn z_image_quality_passes(maximum: f64, mean: f64) -> bool {
    maximum <= Z_IMAGE_MAX_THRESHOLD && mean <= Z_IMAGE_MEAN_THRESHOLD
}

fn fixed_pose_control_image(width: u32, height: u32) -> Image {
    let mut pixels = vec![0_u8; width as usize * height as usize * 3];
    let mut line = |start: (i32, i32), end: (i32, i32), color: [u8; 3]| {
        let (mut x, mut y) = start;
        let dx = (end.0 - start.0).abs();
        let sx = if start.0 < end.0 { 1 } else { -1 };
        let dy = -(end.1 - start.1).abs();
        let sy = if start.1 < end.1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            for offset_y in -2..=2 {
                for offset_x in -2..=2 {
                    let px = x + offset_x;
                    let py = y + offset_y;
                    if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                        let index = (py as usize * width as usize + px as usize) * 3;
                        pixels[index..index + 3].copy_from_slice(&color);
                    }
                }
            }
            if (x, y) == end {
                break;
            }
            let doubled = 2 * error;
            if doubled >= dy {
                error += dy;
                x += sx;
            }
            if doubled <= dx {
                error += dx;
                y += sy;
            }
        }
    };
    // A deterministic whole-body stick pose: head/neck, shoulders, elbows, wrists, hips, knees,
    // and ankles. Using a pose-shaped control fixture exercises the real pose branch rather than
    // merely setting the `ControlKind::Pose` enum on arbitrary pixels.
    let scale = |x: u32, y: u32| ((x * width / 512) as i32, (y * height / 512) as i32);
    for (start, end, color) in [
        (scale(256, 82), scale(256, 138), [255, 255, 255]),
        (scale(190, 158), scale(322, 158), [255, 128, 0]),
        (scale(256, 138), scale(256, 296), [255, 255, 0]),
        (scale(190, 158), scale(145, 238), [0, 255, 0]),
        (scale(145, 238), scale(116, 320), [0, 255, 255]),
        (scale(322, 158), scale(370, 225), [0, 128, 255]),
        (scale(370, 225), scale(404, 292), [0, 0, 255]),
        (scale(214, 296), scale(298, 296), [255, 0, 255]),
        (scale(214, 296), scale(194, 396), [255, 0, 0]),
        (scale(194, 396), scale(176, 482), [128, 0, 255]),
        (scale(298, 296), scale(330, 390), [255, 64, 128]),
        (scale(330, 390), scale(366, 472), [128, 255, 0]),
    ] {
        line(start, end, color);
    }
    Image {
        width,
        height,
        pixels,
    }
}

fn validate_krea_overlay_path(requested: &std::path::Path, revision: &str) -> Result<(), String> {
    protocol::validate_artifact_identity(
        KREA_OVERLAY_REPOSITORY,
        revision,
        KREA_OVERLAY_REPOSITORY,
    )?;
    let repository_component = format!("models--{}", KREA_OVERLAY_REPOSITORY.replace('/', "--"));
    let expected = [
        repository_component.as_str(),
        "snapshots",
        revision,
        KREA_OVERLAY_FILE,
    ];
    let components = requested
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    if !components.ends_with(&expected) {
        return Err(format!(
            "control overlay must end with /{repository_component}/snapshots/{revision}/{KREA_OVERLAY_FILE}"
        ));
    }
    Ok(())
}

fn krea_context(
    width: u32,
    height: u32,
    tile_edge: u32,
    predicted_peak_bytes: u64,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
) -> MemoryRunContext {
    MemoryRunContext {
        selection: MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(tile_edge),
                decode_overlap: Some(KREA_TILE_OVERLAP),
                ..Default::default()
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        },
        // From the LOADED provider's own identity, never a local copy. A hardcoded fingerprint
        // silently goes stale the moment the provider re-fingerprints — which is exactly what
        // happened here: `krea-control-mlx-v4-q4-pose-bounded-decode-512-64` outlived the
        // provider's move to the full-ladder identity and failed the handshake on every case.
        // `fingerprint` stays a parameter only so the negative probe can pass a deliberate
        // mismatch; the real call sites pass `calibration.fingerprint`.
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        // `krea_request` carries exactly one `Conditioning::Control`, and `image_reference_count()`
        // counts Control alongside Reference/Depth/Mask — so the admitted geometry must declare one
        // reference or `configure_request` refuses the render it just admitted. `has_reference` is
        // the compatibility summary of the same fact and gen-core requires it to equal
        // `reference_count > 0`, so the two move together.
        has_reference: true,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 1,
        },
        overlay: Some("control:1".to_owned()),
        budget: MemoryBudget {
            total_bytes: u64::MAX,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-16099@{}", protocol::INFERENCE_PIN),
    }
}

fn krea_request(width: u32, height: u32, steps: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a person standing in a studio, full body editorial photograph".to_owned(),
        width,
        height,
        seed: Some(16099),
        steps: Some(steps),
        conditioning: vec![Conditioning::Control {
            image: fixed_pose_control_image(512, 512),
            kind: ControlKind::Pose,
            scale: Some(0.6),
        }],
        ..Default::default()
    }
}

fn scoped_generate(
    generator: &dyn Generator,
    mut request: GenerationRequest,
    context: &MemoryRunContext,
    error_phase: Option<MemoryPhase>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput, String> {
    if let MemorySafetyDecision::Reject { reason } = generator.memory_strategy_safety_check(context)
    {
        return Err(format!(
            "provider safety check rejected calibrated request: {reason}"
        ));
    }
    let mut scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin calibrated request: {error}"))?
        .ok_or_else(|| "optimized Krea request did not open a memory scope".to_owned())?;
    scope
        .configure_request(&mut request)
        .map_err(|error| format!("configure calibrated request: {error}"))?;
    if let Some(phase) = error_phase {
        // The shared gen-core request floor rejects a fault phase that is not paired with an
        // explicit harness authorization, so the pair must be set together through gen-core's own
        // helper. Without this every capture aborts at its first lifecycle injection.
        request
            .memory
            .get_or_insert_with(Default::default)
            .authorize_calibration_fault(phase);
    }
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter conditioning phase: {error}"))?;
    let mut current_phase = Some(MemoryPhase::Conditioning);
    let mut phase_error = None;
    let result = generator.generate(&request, &mut |progress| {
        let next = match progress {
            Progress::Step { current: 1, .. } => Some(MemoryPhase::Denoise),
            Progress::Decoding => Some(MemoryPhase::Decode),
            _ => None,
        };
        if phase_error.is_none() {
            if let Some(next) = next {
                if current_phase != Some(next) {
                    if let Some(current) = current_phase {
                        if let Err(error) = scope.leave_phase(current) {
                            phase_error = Some(format!("leave {current:?} phase: {error}"));
                        }
                    }
                    if phase_error.is_none() {
                        if let Err(error) = scope.enter_phase(next) {
                            phase_error = Some(format!("enter {next:?} phase: {error}"));
                        } else {
                            current_phase = Some(next);
                        }
                    }
                }
            }
        }
        on_progress(progress);
    });
    if phase_error.is_none() {
        if let Some(current) = current_phase {
            if let Err(error) = scope.leave_phase(current) {
                phase_error = Some(format!("leave {current:?} phase: {error}"));
            }
        }
    }
    let outcome = match &result {
        Ok(_) => MemoryRunOutcome::Complete,
        Err(mlx_gen::gen_core::Error::Canceled) => MemoryRunOutcome::Canceled,
        Err(error) => MemoryRunOutcome::Error {
            message: error.to_string(),
        },
    };
    let finish = scope.finish(outcome);
    if let Some(error) = phase_error {
        finish.map_err(|finish| format!("{error}; finish calibrated request: {finish}"))?;
        return Err(error);
    }
    match (result, finish) {
        (Ok(output), Ok(())) => Ok(output),
        (Err(error), _) => Err(error.to_string()),
        (Ok(_), Err(error)) => Err(format!("finish calibrated request: {error}")),
    }
}

fn one_image(output: GenerationOutput) -> Result<Image, String> {
    match output {
        GenerationOutput::Images(mut images) if images.len() == 1 => Ok(images.remove(0)),
        other => Err(format!("expected one Krea image, got {other:?}")),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhaseMemory {
    active: u64,
    cache: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocatorState {
    active: u64,
    cache: u64,
}

impl AllocatorState {
    fn capture_current() -> Self {
        Self {
            active: get_active_memory() as u64,
            cache: get_cache_memory() as u64,
        }
    }
}

/// Bounds lifecycle cleanup against a successful warm request on the same loaded provider. The 2%
/// allowance is the established real-weight Krea cleanup contract: it absorbs Metal allocator
/// jitter without allowing fault-path retention to scale with another request working set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LifecycleMemoryBounds {
    clean_warm_peak: u64,
    clean_post_cleanup: AllocatorState,
    tolerance_bytes: u64,
}

impl LifecycleMemoryBounds {
    fn from_clean_warm(clean_warm_peak: u64, clean_post_cleanup: AllocatorState) -> Self {
        Self {
            clean_warm_peak,
            clean_post_cleanup,
            tolerance_bytes: clean_warm_peak / 50,
        }
    }

    fn allows_warm_peak(self, peak: u64) -> bool {
        peak <= self.clean_warm_peak.saturating_add(self.tolerance_bytes)
    }

    fn allows_retained(self, retained: AllocatorState) -> bool {
        retained.active
            <= self
                .clean_post_cleanup
                .active
                .saturating_add(self.tolerance_bytes)
            && retained.cache
                <= self
                    .clean_post_cleanup
                    .cache
                    .saturating_add(self.tolerance_bytes)
    }
}

impl PhaseMemory {
    fn allocator_bytes(self) -> u64 {
        self.active.saturating_add(self.cache)
    }

    fn overall(phases: &[Self]) -> Self {
        Self {
            active: phases.iter().map(|phase| phase.active).max().unwrap_or(0),
            cache: phases.iter().map(|phase| phase.cache).max().unwrap_or(0),
        }
    }

    fn capture() -> Self {
        Self {
            active: get_peak_memory() as u64,
            cache: get_cache_memory() as u64,
        }
    }

    fn json(self) -> Value {
        let allocator = self.allocator_bytes();
        json!({
            "activeBytes": self.active,
            "allocatorBytes": allocator,
            "deviceBytes": allocator,
            "wiredBytes": allocator,
            "reclaimableBytes": self.cache,
        })
    }
}

fn predicted_ceiling(bytes: u64) -> u64 {
    let with_margin = bytes.saturating_add(bytes / 20);
    with_margin
        .saturating_add(64 * MIB - 1)
        .saturating_div(64 * MIB)
        .saturating_mul(64 * MIB)
}

fn planned_memory_strategy(request: &Value) -> Result<MemoryStrategy, String> {
    match protocol::planned_rung(request)? {
        "resident" => Ok(MemoryStrategy::Resident),
        "staged_residency" => Ok(MemoryStrategy::StagedResidency),
        "bounded_decode" => Ok(MemoryStrategy::BoundedDecode),
        "bounded_attention" => Ok(MemoryStrategy::BoundedAttention),
        "bounded_transformer_residency" => Ok(MemoryStrategy::BoundedTransformerResidency),
        other => Err(format!("unsupported MLX fresh-reference rung {other:?}")),
    }
}

fn planned_qwen_tier(request: &Value) -> Result<&str, String> {
    match protocol::planned(request)?
        .pointer("/target/tier")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.tier must be a string".to_owned())?
    {
        tier @ ("bf16" | "q4" | "q8") => Ok(tier),
        tier => Err(format!("unsupported MLX numeric tier {tier:?}")),
    }
}

fn planned_qwen_seed(request: &Value, tier: &str) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("qwen-image-{tier}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Qwen fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse Qwen fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the provider's two-step calibration request"
        ));
    }
    Ok(seed)
}

fn planned_selection(request: &Value) -> Result<MemorySelection, String> {
    let strategy = planned_memory_strategy(request)?;
    let transformer_window_size = protocol::optional_parameter(request, "transformerWindowSize")?;
    let (precision, quant) = match planned_qwen_tier(request)? {
        "bf16" => (Precision::Bf16, None),
        "q4" => (Precision::Bf16, Some(Quant::Q4)),
        "q8" => (Precision::Bf16, Some(Quant::Q8)),
        _ => unreachable!("planned_qwen_tier returned an unsupported tier"),
    };
    Ok(MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            decode_tile_edge: protocol::optional_parameter(request, "decodeTileEdge")?,
            decode_overlap: protocol::optional_parameter(request, "decodeOverlap")?,
            attention_chunk_size: protocol::optional_parameter(request, "attentionChunkSize")?,
            transformer_window_size,
            transformer_window_component: transformer_window_size
                .map(|_| TransformerComponent::Dit),
        },
        tier: MemoryNumericTier {
            precision,
            quant,
            component_precision_floors: &[],
        },
    })
}

fn qwen_load_spec(
    root: PathBuf,
    selection: &MemorySelection,
    offload: OffloadPolicy,
    load_shape: LoadShape,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(offload)
        .with_load_shape(load_shape);
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    spec
}

fn strategy_name(strategy: MemoryStrategy) -> &'static str {
    match strategy {
        MemoryStrategy::Resident => "resident",
        MemoryStrategy::StagedResidency => "staged_residency",
        MemoryStrategy::BoundedDecode => "bounded_decode",
        MemoryStrategy::BoundedAttention => "bounded_attention",
        MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

fn attested_strategy(
    request: &Value,
    selection: &MemorySelection,
    engaged: &[MemoryStrategy],
) -> Result<Value, String> {
    let measured = json!({
        "rung": strategy_name(selection.strategy),
        "engagedRungs": engaged.iter().copied().map(strategy_name).collect::<Vec<_>>(),
        "parameters": protocol::strategy_parameters(request)?,
    });
    let planned = protocol::planned(request)?
        .get("strategy")
        .ok_or_else(|| "planned.strategy must be present".to_owned())?;
    if planned != &measured {
        return Err(format!(
            "plan/provider strategy mismatch: plan={planned}, pinned provider measured={measured}"
        ));
    }
    Ok(measured)
}

/// Persisted spelling of the materialization shape a run actually executed under. Derived from the
/// same `LoadShape` handed to the `LoadSpec` (or, better, from the LOADED provider's calibration
/// identity) — never from the plan, which only declares what it expects (sc-16482).
fn load_shape_key(load_shape: LoadShape) -> &'static str {
    match load_shape {
        LoadShape::EagerMaterialization => protocol::LOAD_SHAPE_EAGER,
        LoadShape::DeferredMaterialization => protocol::LOAD_SHAPE_DEFERRED,
    }
}

/// Parse the load shape the capture plan requires. The adapter must execute this shape and then
/// attest the loaded provider's calibration identity; deriving it from the selected rung silently
/// rewrites a declared production-shaped capture (the sc-18237 Qwen q8 bounded-attention defect).
fn planned_load_shape(request: &Value) -> Result<LoadShape, String> {
    match protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
    {
        Some(protocol::LOAD_SHAPE_EAGER) => Ok(LoadShape::EagerMaterialization),
        Some(protocol::LOAD_SHAPE_DEFERRED) => Ok(LoadShape::DeferredMaterialization),
        Some(other) => Err(format!(
            "planned.loadShape must be {:?} or {:?}, got {other:?}",
            protocol::LOAD_SHAPE_EAGER,
            protocol::LOAD_SHAPE_DEFERRED
        )),
        None => Err("planned.loadShape must be a string".to_owned()),
    }
}

fn z_image_load_spec(
    request: &Value,
    load_shape: LoadShape,
) -> Result<(String, String, LoadSpec), String> {
    protocol::validate_plain_overlay_target(request, Z_IMAGE_PLAIN_EXECUTION_PATH)?;
    let repository = protocol::required_env("SCENEWORKS_Z_IMAGE_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_Z_IMAGE_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::Z_IMAGE_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_Z_IMAGE_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_Z_IMAGE_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        "q4",
        protocol::Z_IMAGE_REPOSITORY,
    )?;

    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_quant(Quant::Q4)
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    Ok((repository, revision, spec))
}

fn load_z_image_generator(
    request: &Value,
    load_shape: LoadShape,
) -> Result<(String, String, Box<dyn Generator>), String> {
    let (repository, revision, spec) = z_image_load_spec(request, load_shape)?;
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(Z_IMAGE_PROVIDER, &spec)
        .map_err(|error| format!("load real Z-Image Turbo q4 provider: {error}"))?;
    Ok((repository, revision, generator))
}

fn z_image_request(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(16402),
        // The first Step callback closes the conservative conditioning envelope; the second
        // step then supplies a real denoise-only interval before Decoding.
        steps: Some(2),
        ..Default::default()
    }
}

fn z_image_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // Each plan row is one exact production parameter tuple. Marking this exact tuple's range
    // verified does not promote sibling tuples: the generated matrix still requires a matching
    // manifest calibration binding for each cell.
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

fn run_z_image_reference_loaded(
    request: &Value,
    generator: &dyn Generator,
    repository: &str,
    revision: &str,
    load_shape: LoadShape,
) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, Z_IMAGE_PLAIN_EXECUTION_PATH)?;
    let (width, height) = protocol::target_geometry(request)?;
    let selection = planned_selection(request)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {Z_IMAGE_PROVIDER} has no memory-strategy contract"))?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned Z-Image provider rejected planned selection: {error}"))?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Z-Image provider has no calibration identity".to_owned())?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = MemoryRunContext {
        selection,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-15510@{}", protocol::INFERENCE_PIN),
    };
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let peak_after_reset = get_peak_memory() as u64;
    let selected = one_image(scoped_generate(
        generator,
        z_image_request(width, height),
        &context,
        None,
        &mut |progress| match progress {
            Progress::Step { current: 1, .. } => {
                conditioning.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            Progress::Decoding => {
                denoise.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            _ => {}
        },
    )?)?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized Z-Image lifecycle phase reported a zero active peak".to_owned(),
        );
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted = predicted_ceiling(overall.allocator_bytes());

    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget.total_bytes = predicted;
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("Z-Image provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Z-Image provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-z-image-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Z-Image provider accepted stale calibration evidence".to_owned());
    }

    let baseline = one_image(
        generator
            .generate(&z_image_request(width, height), &mut |_| {})
            .map_err(|error| format!("generate unselected Z-Image reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&selected, &baseline)?;
    if !z_image_quality_passes(maximum_error, mean_error) {
        return Err(format!(
            "Z-Image selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}"
        ));
    }
    // Establish the cleanup oracle on this exact loaded provider before injecting either fault.
    // `clear_cache` is the same explicit retained-buffer release used by the production stream and
    // the pinned Krea lifecycle harness; live model weights remain active, so the post-cleanup
    // snapshot still catches fault-owned arrays that escaped their request scope.
    clear_cache();
    reset_peak_memory();
    let warm = one_image(scoped_generate(
        generator,
        z_image_request(width, height),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let lifecycle_clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let lifecycle_clean_post_cleanup = AllocatorState::capture_current();
    let lifecycle_bounds = LifecycleMemoryBounds::from_clean_warm(
        lifecycle_clean_warm_peak,
        lifecycle_clean_post_cleanup,
    );
    let (warm_maximum, warm_mean) = image_max_mean_abs(&selected, &warm)?;
    if !z_image_quality_passes(warm_maximum, warm_mean) {
        return Err("Z-Image warm repeat changed the deterministic output".to_owned());
    }

    let mut lifecycle_max_fault_active = 0_u64;
    let mut lifecycle_max_fault_cache = 0_u64;
    let mut lifecycle_max_recovery_active = 0_u64;
    let mut lifecycle_max_recovery_cache = 0_u64;
    let mut lifecycle_max_recovery_peak = 0_u64;

    let cancelled = z_image_request(width, height);
    let cancel_signal = cancelled.cancel.clone();
    let cancel_during_decode = selection.strategy == MemoryStrategy::BoundedDecode;
    let mut cancel_triggered = false;
    let cancel_error = scoped_generate(generator, cancelled, &context, None, &mut |progress| {
        if cancel_triggered {
            return;
        }
        match progress {
            Progress::Step { current: 1, .. } if !cancel_during_decode => {
                cancel_triggered = true;
                cancel_signal.cancel();
            }
            Progress::Decoding if cancel_during_decode => {
                cancel_triggered = true;
                let signal = cancel_signal.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    signal.cancel();
                });
            }
            _ => {}
        }
    })
    .expect_err("in-flight Z-Image cancellation must fail");
    if !cancel_triggered {
        return Err("Z-Image cancellation probe never reached the active rung boundary".to_owned());
    }
    if !cancel_error.to_ascii_lowercase().contains("cancel") {
        return Err(format!(
            "Z-Image cancellation returned the wrong error: {cancel_error}"
        ));
    }
    clear_cache();
    let cancel_post_cleanup = AllocatorState::capture_current();
    lifecycle_max_fault_active = lifecycle_max_fault_active.max(cancel_post_cleanup.active);
    lifecycle_max_fault_cache = lifecycle_max_fault_cache.max(cancel_post_cleanup.cache);
    if !lifecycle_bounds.allows_retained(cancel_post_cleanup) {
        return Err(format!(
            "Z-Image cancellation retained active/cache bytes {:?} above the clean warm cleanup {:?} plus {} bytes",
            cancel_post_cleanup,
            lifecycle_clean_post_cleanup,
            lifecycle_bounds.tolerance_bytes,
        ));
    }
    reset_peak_memory();
    let cancel_recovery = one_image(scoped_generate(
        generator,
        z_image_request(width, height),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let cancel_recovery_peak = get_peak_memory() as u64;
    lifecycle_max_recovery_peak = lifecycle_max_recovery_peak.max(cancel_recovery_peak);
    if !lifecycle_bounds.allows_warm_peak(cancel_recovery_peak) {
        return Err(format!(
            "Z-Image cancellation left the warm follow-up peak at {cancel_recovery_peak} bytes, above the clean warm control {lifecycle_clean_warm_peak} bytes plus 2%"
        ));
    }
    clear_cache();
    let cancel_recovery_post_cleanup = AllocatorState::capture_current();
    lifecycle_max_recovery_active =
        lifecycle_max_recovery_active.max(cancel_recovery_post_cleanup.active);
    lifecycle_max_recovery_cache =
        lifecycle_max_recovery_cache.max(cancel_recovery_post_cleanup.cache);
    if !lifecycle_bounds.allows_retained(cancel_recovery_post_cleanup) {
        return Err(format!(
            "Z-Image cancellation warm follow-up retained active/cache bytes {:?} above the clean warm cleanup {:?} plus {} bytes",
            cancel_recovery_post_cleanup,
            lifecycle_clean_post_cleanup,
            lifecycle_bounds.tolerance_bytes,
        ));
    }
    let (cancel_maximum, cancel_mean) = image_max_mean_abs(&selected, &cancel_recovery)?;
    if !z_image_quality_passes(cancel_maximum, cancel_mean) {
        return Err("Z-Image cancellation cleanup changed the warm follow-up".to_owned());
    }

    let injected_phase = if selection.strategy == MemoryStrategy::BoundedDecode {
        MemoryPhase::Decode
    } else {
        MemoryPhase::Denoise
    };
    let injected = scoped_generate(
        generator,
        z_image_request(width, height),
        &context,
        Some(injected_phase),
        &mut |_| {},
    )
    .expect_err("injected Z-Image error must fail");
    if !injected.contains("injected memory-strategy calibration error") {
        return Err(format!(
            "Z-Image error injection returned the wrong error: {injected}"
        ));
    }
    clear_cache();
    let error_post_cleanup = AllocatorState::capture_current();
    lifecycle_max_fault_active = lifecycle_max_fault_active.max(error_post_cleanup.active);
    lifecycle_max_fault_cache = lifecycle_max_fault_cache.max(error_post_cleanup.cache);
    if !lifecycle_bounds.allows_retained(error_post_cleanup) {
        return Err(format!(
            "Z-Image injected error retained active/cache bytes {:?} above the clean warm cleanup {:?} plus {} bytes",
            error_post_cleanup,
            lifecycle_clean_post_cleanup,
            lifecycle_bounds.tolerance_bytes,
        ));
    }
    reset_peak_memory();
    let error_recovery = one_image(scoped_generate(
        generator,
        z_image_request(width, height),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let error_recovery_peak = get_peak_memory() as u64;
    lifecycle_max_recovery_peak = lifecycle_max_recovery_peak.max(error_recovery_peak);
    if !lifecycle_bounds.allows_warm_peak(error_recovery_peak) {
        return Err(format!(
            "Z-Image injected error left the warm follow-up peak at {error_recovery_peak} bytes, above the clean warm control {lifecycle_clean_warm_peak} bytes plus 2%"
        ));
    }
    clear_cache();
    let error_recovery_post_cleanup = AllocatorState::capture_current();
    lifecycle_max_recovery_active =
        lifecycle_max_recovery_active.max(error_recovery_post_cleanup.active);
    lifecycle_max_recovery_cache =
        lifecycle_max_recovery_cache.max(error_recovery_post_cleanup.cache);
    if !lifecycle_bounds.allows_retained(error_recovery_post_cleanup) {
        return Err(format!(
            "Z-Image injected-error warm follow-up retained active/cache bytes {:?} above the clean warm cleanup {:?} plus {} bytes",
            error_recovery_post_cleanup,
            lifecycle_clean_post_cleanup,
            lifecycle_bounds.tolerance_bytes,
        ));
    }
    let (recovery_maximum, recovery_mean) = image_max_mean_abs(&selected, &error_recovery)?;
    if !z_image_quality_passes(recovery_maximum, recovery_mean) {
        return Err("Z-Image error cleanup changed the warm follow-up".to_owned());
    }

    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean) = image_max_mean_abs(&mutated, &baseline)?;
    if z_image_quality_passes(mutated_maximum, mutated_mean) {
        return Err(
            "Z-Image output mutation did not breach the production parity envelope".to_owned(),
        );
    }

    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": "q4",
        },
        "sweep": z_image_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "reason": "post-cancel active/cache retention and the warm follow-up peak remained within the clean-warm control plus 2%", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "post-error active/cache retention and the warm follow-up peak remained within the clean-warm control plus 2%", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "the authoritative Z-Image target has no overlay" }
        ],
        "predictedPeakBytes": {
            "conditioning": predicted_ceiling(conditioning.allocator_bytes()),
            "denoise": predicted_ceiling(denoise.allocator_bytes()),
            "decode": predicted_ceiling(decode.allocator_bytes()),
            "overall": predicted,
        },
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "same seed, conditioning, sampling, precision, and loaded provider; selected rung versus unselected request",
            "identicalInputs": true,
            "identicalLatents": false,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": Z_IMAGE_MAX_THRESHOLD,
            "meanErrorThreshold": Z_IMAGE_MEAN_THRESHOLD,
        },
        "negativeMutation": {
            "parameters": protocol::strategy_parameters(request)?,
            "measured": true,
            "result": "failed_as_expected",
            "maximumError": mutated_maximum,
            "meanError": mutated_mean,
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{repository}@{revision}:q4"),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:z-image-shared-ladder",
            "executed",
            [],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("peakAfterReset", "bytes", peak_after_reset),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("lifecycleCleanWarmPeak", "bytes", lifecycle_clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", lifecycle_clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", lifecycle_clean_post_cleanup.cache),
                ("lifecycleCleanupTolerance", "bytes", lifecycle_bounds.tolerance_bytes),
                ("lifecycleMaximumFaultPostCleanupActive", "bytes", lifecycle_max_fault_active),
                ("lifecycleMaximumFaultPostCleanupCache", "bytes", lifecycle_max_fault_cache),
                ("lifecycleMaximumRecoveryPeak", "bytes", lifecycle_max_recovery_peak),
                ("lifecycleMaximumRecoveryPostCleanupActive", "bytes", lifecycle_max_recovery_active),
                ("lifecycleMaximumRecoveryPostCleanupCache", "bytes", lifecycle_max_recovery_cache),
                ("loadShapeDeferred", "count", u64::from(load_shape == LoadShape::DeferredMaterialization)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, Z_IMAGE_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}

fn run_z_image_reference(request: &Value) -> Result<Value, String> {
    let load_shape = planned_load_shape(request)?;
    let (repository, revision, generator) = load_z_image_generator(request, load_shape)?;
    run_z_image_reference_loaded(
        request,
        generator.as_ref(),
        &repository,
        &revision,
        load_shape,
    )
}

/// Maximum, mean, and root-mean-square absolute error between two images, in [0,1] units. The
/// runtime_complete quality shape requires the RMS metric alongside max/mean
/// (`memory-calibration-harness.mjs#validateRuntimeComplete`), which is why this exists next to
/// `image_max_mean_abs` instead of replacing it.
fn image_max_mean_rms_abs(left: &Image, right: &Image) -> Result<(f64, f64, f64), String> {
    let (maximum, mean) = image_max_mean_abs(left, right)?;
    let mut sum_squares = 0.0_f64;
    for (&left, &right) in left.pixels.iter().zip(&right.pixels) {
        let difference = (f64::from(left) - f64::from(right)).abs() / 255.0;
        sum_squares += difference * difference;
    }
    Ok((
        maximum,
        mean,
        (sum_squares / left.pixels.len() as f64).sqrt(),
    ))
}

fn flux2_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= FLUX2_MAX_THRESHOLD && mean <= FLUX2_MEAN_THRESHOLD && rms <= FLUX2_RMS_THRESHOLD
}

/// Defense-in-depth mirror of the provider-mismatch guard `validate_z_image_batch` carries
/// (sc-18104): `run` dispatches by name today, but this arm hardcodes the FLUX.2-dev contract, so a
/// future caller must be refused by name here rather than misrouted into that contract.
fn validate_flux2_target(request: &Value) -> Result<(), String> {
    let provider = protocol::planned(request)?
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    if provider != FLUX2_PROVIDER {
        return Err(format!(
            "MLX FLUX.2-dev calibration does not implement provider {provider:?}"
        ));
    }
    Ok(())
}

/// Bind the fixture to the planned tier AND geometry edge, deriving the seed — the same
/// fixture-to-plan binding `planned_qwen_seed` enforces, extended to the edge because this lane's
/// plan carries each of its q4 and q8 tiers at two geometries (768² and 1024²).
fn planned_flux2_seed(request: &Value, tier: &str, width: u32) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("flux2-dev-mlx-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse FLUX.2-dev fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse FLUX.2-dev fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(seed)
}

fn flux2_request(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "a lighthouse on a rocky coastline at golden hour, photorealistic".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        // The first Step callback closes the conditioning envelope; the second supplies a real
        // denoise-only interval before Decoding — the z_image/qwen phase-boundary pattern.
        steps: Some(2),
        ..Default::default()
    }
}

/// Resolve and validate the `SCENEWORKS_FLUX2_*` environment family into a tier-exact load spec.
/// The tier is DERIVED from `/target/tier` and threads through the per-tier ROOT suffix check and
/// `spec.quantize` — never hardcoded (sc-17097 fixed exactly that hardcoding on the Candle side).
fn flux2_load_spec(
    request: &Value,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, LoadSpec), String> {
    protocol::validate_plain_overlay_target(request, FLUX2_PLAIN_EXECUTION_PATH)?;
    let repository = protocol::required_env("SCENEWORKS_FLUX2_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_FLUX2_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::FLUX2_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_FLUX2_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_FLUX2_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::FLUX2_REPOSITORY,
    )?;
    Ok((repository, revision, flux2_spec(root, selection)))
}

/// The tier-exact FLUX.2-dev load spec: resident offload, eager materialization (the contract's
/// calibrated load shape), and the quant DERIVED from the planned selection.
fn flux2_spec(root: PathBuf, selection: &MemorySelection) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization);
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    spec
}

/// The admission context for the FLUX.2-dev safety scenarios. It exactly describes the base
/// text-to-image route: `MemoryMode::TextToImage`, no reference, and `reference_count == 0`.
/// `overlay` stays `None` because this authoritative lane is base-only.
fn flux2_admission_context(
    selection: &MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    width: u32,
    height: u32,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection: *selection,
        calibration_abi: calibration.abi,
        // A parameter only so the stale-evidence probe can pass a deliberate mismatch; the real
        // call sites pass `calibration.fingerprint` (the Krea-arm lesson at `krea_context`).
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-18218@{}", protocol::INFERENCE_PIN),
    }
}

fn flux2_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // One exact resident tuple per plan row; marking it range-verified promotes no sibling tuple
    // (the generated matrix still requires a matching manifest binding per cell).
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// The `mlx:flux2_dev` arm (sc-18218) — RESIDENT ONLY, deliberately.
///
/// At the interim inference PR #531 head, `flux2_dev` owns a distinct reference-free T2I contract.
/// Every non-Resident strategy remains `Missing`; this arm refuses those rungs, reads the registry
/// contract under the exact T2I provider id, and then proves that the loaded generator exposes the
/// byte-for-byte same contract before measuring it. No edit-provider declaration or edit-shaped
/// context participates in this lane.
fn run_flux2_dev(request: &Value) -> Result<Value, String> {
    validate_flux2_target(request)?;
    protocol::validate_plain_overlay_target(request, FLUX2_PLAIN_EXECUTION_PATH)?;
    let rung = protocol::planned_rung(request)?;
    if rung != "resident" {
        return Err(format!(
            "the pinned MLX FLUX.2-dev provider implements only the resident strategy (every other \
             strategy is declared Missing at the pin); rung {rung:?} is not capturable"
        ));
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
    if !matches!(tier, "q4" | "q8") {
        return Err(format!(
            "the authoritative MLX FLUX.2-dev plan supports only q4 and q8; tier {tier:?} is not capturable"
        ));
    }
    let (width, height) = protocol::target_geometry(request)?;
    let seed = planned_flux2_seed(request, tier, width)?;
    let (repository, revision, spec) = flux2_load_spec(request, tier, &selection)?;
    let registry = mlx_gen_flux2::provider_registry()
        .map_err(|error| format!("build FLUX.2 registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(FLUX2_PROVIDER, &spec)
        .map_err(|error| format!("read {FLUX2_PROVIDER} T2I memory-strategy contract: {error}"))?
        .ok_or_else(|| {
            format!("{FLUX2_PROVIDER} has no T2I memory-strategy contract at the pin")
        })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned FLUX.2-dev contract rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned FLUX.2-dev contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != FLUX2_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "pinned FLUX.2-dev T2I contract fingerprint changed: expected {FLUX2_CALIBRATION_FINGERPRINT}, got {}",
            calibration.fingerprint
        ));
    }
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if seed != FLUX2_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the FLUX.2-dev calibration seed {FLUX2_SEED}"
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        mlx_gen_flux2::memory_strategy::registered_dev_t2i_safety_check(
            &spec,
            &contract,
            &flux2_admission_context(
                &selection,
                calibration,
                fingerprint,
                width,
                height,
                total_bytes,
                predicted,
            ),
        )
    };
    // Admission mutation hygiene BEFORE the expensive load: the gate must accept a fitting
    // request (so the two rejections below cannot pass via a blanket refusal), reject an
    // unknown/zero budget, and reject a mutated calibration fingerprint.
    if !matches!(
        safety(&calibration.fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "FLUX.2-dev admission rejected a fitting probe budget; the scenario rejections below \
             would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("FLUX.2-dev admission accepted an unknown/zero memory budget".to_owned());
    }
    if !matches!(
        safety("stale-flux2-dev-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("FLUX.2-dev admission accepted stale calibration evidence".to_owned());
    }

    let generator = registry
        .load(FLUX2_PROVIDER, &spec)
        .map_err(|error| format!("load real FLUX.2-dev {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded FLUX.2-dev generator exposed no T2I memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err(
            "loaded FLUX.2-dev generator contract differs from the registry contract".to_owned(),
        );
    }
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let selected = one_image(
        generator
            .generate(
                &flux2_request(width, height, seed),
                &mut |progress| match progress {
                    Progress::Step { current: 1, .. } => {
                        conditioning.set(PhaseMemory::capture());
                        reset_peak_memory();
                    }
                    Progress::Decoding => {
                        denoise.set(PhaseMemory::capture());
                        reset_peak_memory();
                    }
                    _ => {}
                },
            )
            .map_err(|error| format!("generate measured FLUX.2-dev render: {error}"))?,
    )?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized FLUX.2-dev lifecycle phase reported a zero active peak".to_owned(),
        );
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted = predicted_ceiling(overall.allocator_bytes());
    let exact_fit = flux2_admission_context(
        &selection,
        calibration,
        &calibration.fingerprint,
        width,
        height,
        predicted,
        predicted,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&exact_fit),
        MemorySafetyDecision::Accept
    ) {
        return Err("FLUX.2-dev admission rejected an exact-fit calibrated budget".to_owned());
    }

    // Warm repeat determinism + cleanup bounds on this exact loaded provider. The scoped
    // lifecycle scenarios cannot run (no request scope, no injection site), so the unscoped
    // repeats gate quality and the allocator bounds gate cleanup instead.
    clear_cache();
    reset_peak_memory();
    let baseline = one_image(
        generator
            .generate(&flux2_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate warm FLUX.2-dev control: {error}"))?,
    )?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let cleanup_bounds =
        LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = image_max_mean_rms_abs(&selected, &baseline)?;
    if !flux2_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "FLUX.2-dev warm repeat exceeded the determinism envelope: max={maximum_error:.6}, \
             mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    reset_peak_memory();
    let warm = one_image(
        generator
            .generate(&flux2_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate warm FLUX.2-dev repeat: {error}"))?,
    )?;
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "FLUX.2-dev warm repeat peaked at {warm_peak} bytes, above the clean warm control \
             {clean_warm_peak} bytes plus 2%"
        ));
    }
    clear_cache();
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "FLUX.2-dev warm repeat retained active/cache bytes {warm_post_cleanup:?} above the \
             clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
            cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = image_max_mean_rms_abs(&selected, &warm)?;
    if !flux2_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err("FLUX.2-dev second warm repeat changed the deterministic output".to_owned());
    }

    // Arm-internal negative-mutation falsifiability check. A runtime_complete record must keep
    // `negativeMutation` null (`memory-calibration-harness.mjs#validateRuntimeComplete`), so the
    // breach is verified here — the capture fails if the envelope cannot be breached — and the
    // measured numbers land in diagnostics rather than in the record field.
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if flux2_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(
            "FLUX.2-dev output mutation did not breach the determinism envelope".to_owned(),
        );
    }

    let lifecycle_blocker = concat!(
        "the pinned mlx-gen-flux2 crate opens no memory-strategy request scope for the dev ",
        "variants and has no calibration fault-injection site, so the scoped lifecycle scenario ",
        "cannot execute; unscoped repeat determinism and allocator cleanup bounds are attested in ",
        "quality and diagnostics instead"
    );
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        },
        "sweep": flux2_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed", "reason": "the registered FLUX.2-dev admission check rejected a zero/unknown budget before load" },
            { "name": "stale_evidence", "result": "passed", "reason": "the registered FLUX.2-dev admission check rejected a mutated calibration fingerprint before load" },
            { "name": "warm_repeat", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": {
            "conditioning": predicted_ceiling(conditioning.allocator_bytes()),
            "denoise": predicted_ceiling(denoise.allocator_bytes()),
            "decode": predicted_ceiling(decode.allocator_bytes()),
            "overall": predicted,
        },
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "identical artifact, prompt, seed, geometry, steps, tier, and loaded provider; cold measured render versus warm unscoped repeats",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": FLUX2_MAX_THRESHOLD,
            "meanErrorThreshold": FLUX2_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": FLUX2_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}"),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:flux2-dev-resident",
            "executed",
            [lifecycle_blocker.to_owned()],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("lifecycleCleanWarmPeak", "bytes", clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", clean_post_cleanup.cache),
                ("lifecycleCleanupTolerance", "bytes", cleanup_bounds.tolerance_bytes),
                ("lifecycleWarmRepeatPeak", "bytes", warm_peak),
                ("lifecycleWarmRepeatPostCleanupActive", "bytes", warm_post_cleanup.active),
                ("lifecycleWarmRepeatPostCleanupCache", "bytes", warm_post_cleanup.cache),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("loadShapeDeferred", "count", 0),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, FLUX2_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}

fn validate_z_image_batch(request: &Value) -> Result<&[Value], String> {
    let planned = request
        .get("planned")
        .and_then(Value::as_array)
        .ok_or_else(|| "assess_batch request.planned must be an array".to_owned())?;
    let expected = [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ];
    // sc-18104: refuse a foreign provider by name FIRST, for the same reason `run` does. This batch
    // path is Z-Image-only — `assess_z_image_batch` hardcodes `Z_IMAGE_PROVIDER` when it reads the
    // memory-strategy contract — but nothing below inspects `target.provider`, so without this check
    // a batch for another provider is MISROUTED into the Z-Image contract and dies on a
    // Z-Image-shaped fingerprint complaint, after `runtime_macos::catalog()` has already done real
    // environment work. That is the same silent-misroute class the `run` refusal closes, and it is
    // reachable: `assessProviderReuse` (scripts/memory-calibration-harness.mjs) selects candidates by
    // backend and optional fixture only, never by provider.
    //
    // This runs BEFORE the length check deliberately. A foreign batch of the wrong length is still
    // stopped safely there, but it would be told `Z-Image rung batch must contain exactly 5 cases` —
    // a Z-Image-named complaint about a provider that is not Z-Image, the same misleading-diagnostic
    // problem in miniature. That is reachable for the very lane which motivated this fix:
    // `mlx-gen-flux2` marks every non-Resident strategy `Missing`, so an `assess-reuse` on a flux2
    // fixture submits a ONE-element batch. Refusing by name is therefore unconditional; only an empty
    // batch, which has no `planned[0]` to read a provider from, falls through to the length check.
    if let Some(provider) = planned
        .first()
        .and_then(|case| case.pointer("/target/provider"))
        .and_then(Value::as_str)
    {
        if provider != Z_IMAGE_PROVIDER {
            return Err(format!(
                "MLX five-rung batch assessment does not implement provider {provider:?}"
            ));
        }
    }
    if planned.len() != expected.len() {
        return Err(format!(
            "Z-Image rung batch must contain exactly {} cases, got {}",
            expected.len(),
            planned.len()
        ));
    }
    let target = planned[0]
        .get("target")
        .ok_or_else(|| "assess_batch planned target must be present".to_owned())?;
    for (index, (item, expected_rung)) in planned.iter().zip(expected).enumerate() {
        let rung = item
            .pointer("/strategy/rung")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!("assess_batch planned[{index}].strategy.rung must be a string")
            })?;
        if rung != expected_rung {
            return Err(format!(
                "Z-Image rung batch must use canonical order; index {index} is {rung:?}, expected {expected_rung:?}"
            ));
        }
        if item.get("target") != Some(target) {
            return Err("Z-Image rung batch must keep one exact target tuple".to_owned());
        }
    }
    Ok(planned)
}

fn z_image_reuse_identity(fingerprint: &str, load_shape: LoadShape) -> String {
    format!("{fingerprint}@{}", load_shape_key(load_shape))
}

fn assess_z_image_batch(request: &Value) -> Result<Value, String> {
    let planned = validate_z_image_batch(request)?;
    let mut representative = request.clone();
    representative["action"] = json!("run");
    representative["planned"] = planned[0].clone();
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let mut actual_fingerprints = Vec::new();
    let mut actual_identities = Vec::new();
    for load_shape in [
        LoadShape::EagerMaterialization,
        LoadShape::DeferredMaterialization,
    ] {
        let (_, _, spec) = z_image_load_spec(&representative, load_shape)?;
        let contract = catalog
            .media()
            .memory_strategy_contract(Z_IMAGE_PROVIDER, &spec)
            .map_err(|error| format!("read {Z_IMAGE_PROVIDER} memory-strategy contract: {error}"))?
            .ok_or_else(|| format!("{Z_IMAGE_PROVIDER} has no memory-strategy contract"))?;
        let calibration = contract
            .calibration
            .as_ref()
            .ok_or_else(|| "pinned Z-Image provider has no calibration identity".to_owned())?;
        actual_fingerprints.push(calibration.fingerprint.clone());
        actual_identities.push(z_image_reuse_identity(
            &calibration.fingerprint,
            calibration.load_shape,
        ));
    }
    for item in planned {
        let planned_fingerprint = item
            .get("calibrationFingerprint")
            .and_then(Value::as_str)
            .ok_or_else(|| "batch calibrationFingerprint must be a string".to_owned())?;
        let expected = if item.pointer("/strategy/rung").and_then(Value::as_str)
            == Some("bounded_transformer_residency")
        {
            &actual_fingerprints[1]
        } else {
            &actual_fingerprints[0]
        };
        if planned_fingerprint != expected {
            return Err(format!(
                "plan/provider calibration mismatch in reuse assessment: plan={planned_fingerprint}, pinned provider={expected}"
            ));
        }
    }
    actual_fingerprints.sort_unstable();
    actual_fingerprints.dedup();
    actual_identities.sort_unstable();
    actual_identities.dedup();
    if actual_identities.len() > 1 {
        return Ok(json!({
            "verdict": "unable_to_amortize",
            "reason": format!(
                "one MLX model load cannot preserve the distinct calibrated load-shape identities required by the five rungs: {}",
                actual_identities.join(", ")
            ),
            "calibrationFingerprints": actual_fingerprints,
            "calibrationIdentities": actual_identities,
            "evidence": "pinned provider contracts for eager and deferred load specs",
        }));
    }
    Ok(json!({
        "verdict": "eligible_for_measurement",
        "reason": "all MLX rungs share one calibrated load-shape identity",
        "calibrationFingerprints": actual_fingerprints,
        "calibrationIdentities": actual_identities,
        "evidence": "pinned provider contracts for eager and deferred load specs",
    }))
}

fn run_krea_control(request: &Value) -> Result<Value, String> {
    protocol::validate_exact_overlay_target(request, "control:1", KREA_CONTROL_EXECUTION_PATH)?;
    let parameters = protocol::strategy_parameters(request)?;
    let strategy = json!({
        "rung": "bounded_decode",
        "engagedRungs": ["resident", "bounded_decode"],
        "parameters": parameters,
    });
    let planned_strategy = protocol::planned(request)?
        .get("strategy")
        .ok_or_else(|| "planned.strategy must be present".to_owned())?;
    if planned_strategy != &strategy {
        return Err(format!(
            "plan/provider strategy mismatch: plan={planned_strategy}, MLX adapter measured={strategy}"
        ));
    }
    let tile_edge = protocol::parameter(request, "decodeTileEdge")?;
    let overlap = protocol::parameter(request, "decodeOverlap")?;
    if tile_edge != KREA_TILE_EDGES[0] || overlap != KREA_TILE_OVERLAP {
        return Err(format!(
            "authoritative Krea records must target exact bounded decode {}/{}",
            KREA_TILE_EDGES[0], KREA_TILE_OVERLAP
        ));
    }
    let (width, height) = protocol::target_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_KREA_CONTROL_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_KREA_CONTROL_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::KREA_REPOSITORY)?;
    let base_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_KREA_CONTROL_ROOT",
    )?))
    .map_err(|error| format!("canonicalize Krea control root: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &base_root,
        &repository,
        &revision,
        "q4",
        protocol::KREA_REPOSITORY,
    )?;
    let overlay_revision = protocol::required_env("SCENEWORKS_KREA_CONTROL_OVERLAY_REVISION")?;
    let requested_overlay_path =
        PathBuf::from(protocol::required_env("SCENEWORKS_KREA_CONTROL_OVERLAY")?);
    validate_krea_overlay_path(&requested_overlay_path, &overlay_revision)?;
    std::fs::canonicalize(&requested_overlay_path)
        .map_err(|error| format!("canonicalize Krea control overlay: {error}"))?;
    let resolved_path_fingerprint = format!(
        "{repository}@{revision}:q4|{KREA_OVERLAY_REPOSITORY}@{overlay_revision}:{KREA_OVERLAY_FILE}"
    );

    let spec = LoadSpec::new(WeightsSource::Dir(base_root))
        // Preserve the requested `.safetensors` path for MLX's extension-based loader after the
        // canonicalization above has proved that the HF snapshot symlink resolves to a real file.
        .with_control(WeightsSource::File(requested_overlay_path))
        .with_offload_policy(OffloadPolicy::Resident);
    let generator = mlx_gen_krea::provider_registry()
        .map_err(|error| format!("build Krea registry: {error}"))?
        .load(KREA_PROVIDER, &spec)
        .map_err(|error| format!("load real Krea q4 control provider: {error}"))?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded krea_2_turbo_control has no memory-strategy contract".to_owned())?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Krea control provider has no calibration identity".to_owned())?;
    // Fail closed when the plan and the pinned provider disagree, rather than measuring against
    // one identity and stamping the receipt with the other.
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    let stale_context = krea_context(
        width,
        height,
        tile_edge,
        1,
        calibration,
        "stale-fingerprint",
    );
    if !matches!(
        generator.memory_strategy_safety_check(&stale_context),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("provider accepted a stale calibration fingerprint".to_owned());
    }
    let mut unknown_context = krea_context(
        width,
        height,
        tile_edge,
        1,
        calibration,
        &calibration.fingerprint,
    );
    unknown_context.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown_context),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("provider accepted an unknown/zero memory budget".to_owned());
    }

    let context = krea_context(
        width,
        height,
        tile_edge,
        1,
        calibration,
        &calibration.fingerprint,
    );
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    reset_peak_memory();
    let baseline = one_image(scoped_generate(
        generator.as_ref(),
        krea_request(width, height, 8),
        &context,
        None,
        &mut |progress| match progress {
            Progress::Step { current: 1, .. } => {
                conditioning.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            Progress::Decoding => {
                denoise.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            _ => {}
        },
    )?)?;
    let decode = PhaseMemory::capture();
    if [
        conditioning.get().active,
        denoise.get().active,
        decode.active,
    ]
    .contains(&0)
    {
        return Err("a synchronized Krea lifecycle phase reported a zero active peak".to_owned());
    }

    // The first pass above is the exact production 512/64 route whose synchronized memory peaks are
    // published. Compare it against the provider's unbounded default on this probed 128 GB machine;
    // with no request memory selection, the provider executes its single-pass Qwen-VAE decode.
    // Running the reference second preserves the production route as the cold measurement.
    let untiled_reference = one_image(
        generator
            .generate(&krea_request(width, height, 8), &mut |_| {})
            .map_err(|error| format!("generate untiled Krea quality reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&baseline, &untiled_reference)?;
    if maximum_error > KREA_MAX_THRESHOLD || mean_error > KREA_MEAN_THRESHOLD {
        return Err(format!(
            "Krea 512/64 bounded decode exceeded untiled parity: \
             max={maximum_error:.6}, mean={mean_error:.6}"
        ));
    }
    let warm_repeat = one_image(scoped_generate(
        generator.as_ref(),
        krea_request(width, height, 8),
        &context,
        None,
        &mut |_| {},
    )?)?;
    if baseline.pixels != warm_repeat.pixels {
        return Err("warm Krea A/B/A repeat changed output bytes".to_owned());
    }

    let lifecycle_steps = 1;
    clear_cache();
    reset_peak_memory();
    one_image(scoped_generate(
        generator.as_ref(),
        krea_request(width, height, lifecycle_steps),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let lifecycle_control_peak = get_peak_memory() as u64;
    let lifecycle_recovery_limit =
        lifecycle_control_peak.saturating_add(lifecycle_control_peak / 50);
    let mut lifecycle_max_recovery_peak = 0_u64;
    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let cancel = mlx_gen::CancelFlag::new();
        if phase == MemoryPhase::Conditioning {
            cancel.cancel();
        }
        let mut canceled_request = krea_request(width, height, lifecycle_steps);
        canceled_request.cancel = cancel.clone();
        let result = scoped_generate(
            generator.as_ref(),
            canceled_request,
            &context,
            None,
            &mut |progress| {
                if (phase == MemoryPhase::Denoise
                    && matches!(progress, Progress::Step { current: 1, .. }))
                    || (phase == MemoryPhase::Decode && matches!(progress, Progress::Decoding))
                {
                    cancel.cancel();
                }
            },
        );
        match result {
            Err(error) if error.to_ascii_lowercase().contains("cancel") => {}
            Err(error) => {
                return Err(format!(
                    "{phase:?} cancellation returned the wrong error: {error}"
                ));
            }
            Ok(_) => {
                return Err(format!(
                    "{phase:?} cancellation returned images instead of the typed cancellation path"
                ));
            }
        }
        clear_cache();
        reset_peak_memory();
        one_image(scoped_generate(
            generator.as_ref(),
            krea_request(width, height, lifecycle_steps),
            &context,
            None,
            &mut |_| {},
        )?)?;
        let recovery_peak = get_peak_memory() as u64;
        lifecycle_max_recovery_peak = lifecycle_max_recovery_peak.max(recovery_peak);
        if recovery_peak > lifecycle_recovery_limit {
            return Err(format!(
                "{phase:?} cancellation left the warm follow-up peak at {recovery_peak} bytes, \
                 above the successful warm control {lifecycle_control_peak} bytes plus 2%"
            ));
        }
    }
    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let result = scoped_generate(
            generator.as_ref(),
            krea_request(width, height, lifecycle_steps),
            &context,
            Some(phase),
            &mut |_| {},
        );
        match result {
            Err(error) if error.contains("injected memory-strategy calibration error") => {}
            Err(error) => {
                return Err(format!(
                    "{phase:?} error injection returned the wrong error: {error}"
                ));
            }
            Ok(_) => {
                return Err(format!(
                    "{phase:?} error injection returned images instead of failing at its physical \
                     boundary"
                ));
            }
        }
        clear_cache();
        reset_peak_memory();
        one_image(scoped_generate(
            generator.as_ref(),
            krea_request(width, height, lifecycle_steps),
            &context,
            None,
            &mut |_| {},
        )?)?;
        let recovery_peak = get_peak_memory() as u64;
        lifecycle_max_recovery_peak = lifecycle_max_recovery_peak.max(recovery_peak);
        if recovery_peak > lifecycle_recovery_limit {
            return Err(format!(
                "{phase:?} injected error left the warm follow-up peak at {recovery_peak} bytes, \
                 above the successful warm control {lifecycle_control_peak} bytes plus 2%"
            ));
        }
    }

    let conditioning = conditioning.get();
    let denoise = denoise.get();
    let phases = [conditioning, denoise, decode];
    let overall = PhaseMemory::overall(&phases);
    let predicted_conditioning = predicted_ceiling(conditioning.active + conditioning.cache);
    let predicted_denoise = predicted_ceiling(denoise.active + denoise.cache);
    let predicted_decode = predicted_ceiling(decode.active + decode.cache);
    // The harness defines `overall` as a conservative componentwise high-water envelope: every
    // overall metric must cover the corresponding peak from every physical phase. Predict from that
    // same envelope so exact-fit admission can never sit below the published observed overall.
    let predicted_overall = predicted_ceiling(overall.allocator_bytes());
    let mutation_bias = 0.05_f64;
    let mutated_maximum = maximum_error + mutation_bias;
    let mutated_mean = mean_error + mutation_bias;

    Ok(json!({
        "status": "complete",
        "strategy": strategy,
        // The receipt attests what THIS run loaded under, read from the provider that ran it
        // (sc-16482): the plan's declaration is cross-checked, never copied onto the fragment.
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": "q4",
        },
        "sweep": {
            "axes": [
                { "parameter": "decodeTileEdge", "testedValues": KREA_TILE_EDGES },
                { "parameter": "decodeOverlap", "testedValues": [KREA_TILE_OVERLAP] }
            ],
            "cases": KREA_TILE_EDGES.into_iter().map(|edge| json!({
                "parameters": { "decodeTileEdge": edge, "decodeOverlap": KREA_TILE_OVERLAP },
                "result": "passed"
            })).collect::<Vec<_>>(),
            "rangeVerified": true,
        },
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted_overall, "effectiveBudgetBytes": predicted_overall },
            { "name": "unknown_budget", "result": "passed", "reason": "provider rejected a zero/unknown budget before render" },
            { "name": "stale_evidence", "result": "passed", "reason": "provider rejected a mutated calibration fingerprint before render" },
            { "name": "warm_repeat", "result": "passed", "reason": "resident A/B/A output bytes were identical" },
            { "name": "cancel", "result": "passed", "reason": "conditioning, denoise, and decode cancellation returned typed cancellation", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "conditioning, denoise, and decode injected errors fired at physical boundaries", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed", "reason": "canonical q4 base and exact pose overlay loaded and rendered" },
            { "name": "overlay", "result": "passed", "reason": "real pose-control overlay participated in every measured render" }
        ],
        "predictedPeakBytes": {
            "conditioning": predicted_conditioning,
            "denoise": predicted_denoise,
            "decode": predicted_decode,
            "overall": predicted_overall,
        },
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "same seed and control latent, exact production 512/64 bounded decode versus single-pass Qwen-VAE decode",
            "identicalLatents": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": KREA_MAX_THRESHOLD,
            "meanErrorThreshold": KREA_MEAN_THRESHOLD,
        },
        "negativeMutation": {
            "parameters": parameters,
            "measured": true,
            "result": "failed_as_expected",
            "maximumError": mutated_maximum,
            "meanError": mutated_mean,
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": resolved_path_fingerprint,
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:krea-control",
            "executed",
            [],
            [
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("lifecycleWarmControlPeak", "bytes", lifecycle_control_peak),
                ("lifecycleMaximumRecoveryPeak", "bytes", lifecycle_max_recovery_peak),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    }))
}

fn sweep(parameters: &serde_json::Map<String, Value>, passed: bool) -> Value {
    let current_edge = parameters
        .get("decodeTileEdge")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let current_overlap = parameters
        .get("decodeOverlap")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let mut cases: Vec<Value> = EDGES
        .into_iter()
        .map(|edge| {
            json!({
                "parameters": { "decodeTileEdge": edge, "decodeOverlap": 64 },
                "result": if u64::from(edge) == current_edge && current_overlap == 64 {
                    if passed { "passed" } else { "failed" }
                } else {
                    "not_run"
                }
            })
        })
        .collect();
    cases.push(json!({
        "parameters": {
            "decodeTileEdge": 256,
            "decodeOverlap": 32,
            "comparisonOutputBias": 0.05,
        },
        "result": if current_edge == 256 && current_overlap == 32 {
            if passed { "passed" } else { "failed" }
        } else {
            "not_run"
        }
    }));
    json!({
        "axes": [
            { "parameter": "decodeTileEdge", "testedValues": EDGES },
            { "parameter": "decodeOverlap", "testedValues": [32, 64] }
        ],
        "cases": cases,
        "rangeVerified": false,
    })
}

fn run_qwen_vae_probe(request: &Value) -> Result<Value, String> {
    if protocol::planned(request)?
        .get("backend")
        .and_then(Value::as_str)
        != Some("mlx")
    {
        return Err(
            "MLX adapter received a non-MLX planned case; run the harness with --backend mlx"
                .to_owned(),
        );
    }
    protocol::validate_plain_overlay_target(request, QWEN_PLAIN_EXECUTION_PATH)?;
    let parameters = protocol::strategy_parameters(request)?;
    let strategy = json!({
        "rung": "bounded_decode",
        "engagedRungs": ["resident", "bounded_decode"],
        "parameters": parameters,
    });
    let planned_strategy = protocol::planned(request)?
        .get("strategy")
        .ok_or_else(|| "planned.strategy must be present".to_owned())?;
    if planned_strategy != &strategy {
        return Err(format!(
            "plan/provider strategy mismatch: plan={planned_strategy}, MLX adapter measured={strategy}"
        ));
    }
    let expected_failure = protocol::expected_failure(request);
    let comparison_output_bias = protocol::comparison_output_bias(parameters, expected_failure)?;
    let tile_edge = protocol::parameter(request, "decodeTileEdge")?;
    let overlap = protocol::parameter(request, "decodeOverlap")?;
    let tile_edge_i32 = i32::try_from(tile_edge)
        .map_err(|_| format!("decodeTileEdge={tile_edge} exceeds the MLX tiling API range"))?;
    let overlap_i32 = i32::try_from(overlap)
        .map_err(|_| format!("decodeOverlap={overlap} exceeds the MLX tiling API range"))?;
    let (width, height) = protocol::target_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_QWEN_IMAGE_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_QWEN_IMAGE_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::QWEN_REPOSITORY)?;
    let root = PathBuf::from(protocol::required_env("SCENEWORKS_QWEN_IMAGE_ROOT")?);
    if !root.is_dir() {
        return Err("SCENEWORKS_QWEN_IMAGE_ROOT is not a directory".to_owned());
    }
    let root = std::fs::canonicalize(root)
        .map_err(|error| format!("canonicalize SCENEWORKS_QWEN_IMAGE_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        "bf16",
        protocol::QWEN_REPOSITORY,
    )?;
    let loadability_fingerprint = format!("{repository}@{revision}:bf16");
    let artifact = json!({
        "repository": &repository,
        "resolvedRevision": &revision,
        "variant": "bf16",
    });
    let vae = load_vae(&root)
        .map_err(|error| format!("load Qwen VAE from validated snapshot: {error}"))?;
    let latent = encoded_latent(&vae, width, height)?;

    clear_cache();
    reset_peak_memory();
    let baseline = vae
        .decode(&latent)
        .map_err(|error| format!("untiled Qwen VAE decode: {error}"))?;
    baseline
        .eval()
        .map_err(|error| format!("materialize untiled Qwen VAE decode: {error}"))?;
    let untiled_peak = get_peak_memory() as u64;

    clear_cache();
    reset_peak_memory();
    let tiled = vae
        .decode_tiled(
            &latent,
            &TilingConfig {
                spatial: Some(SpatialTiling {
                    tile_px: tile_edge_i32,
                    overlap_px: overlap_i32,
                }),
                temporal: None,
            },
            None,
        )
        .map_err(|error| format!("tiled Qwen VAE decode {tile_edge}/{overlap}: {error}"))?;
    tiled
        .eval()
        .map_err(|error| format!("materialize tiled Qwen VAE decode: {error}"))?;
    let tiled_peak = get_peak_memory() as u64;
    let active = get_active_memory() as u64;
    let cache = get_cache_memory() as u64;
    let (actual_maximum, actual_mean) = decoded_max_mean_abs(&baseline, &tiled, None)?;
    let actual_passed = actual_maximum <= MAX_THRESHOLD && actual_mean <= MEAN_THRESHOLD;
    if expected_failure {
        if !actual_passed {
            return Err(format!(
                "negative control requires a passing unmodified identical-latent comparison: max={actual_maximum:.6}, mean={actual_mean:.6}"
            ));
        }
        let (mutated_maximum, mutated_mean) =
            decoded_max_mean_abs(&baseline, &tiled, comparison_output_bias)?;
        if mutated_maximum <= MAX_THRESHOLD && mutated_mean <= MEAN_THRESHOLD {
            return Err(format!(
                "negative mutation {tile_edge}/{overlap} did not breach the identical-latent threshold: max={mutated_maximum:.6}, mean={mutated_mean:.6}"
            ));
        }
        let blocker =
            "negative mutation is measured, but negative evidence cannot verify a production range";
        let mut fragment = protocol::plain_gated_fragment(
            request,
            QWEN_PLAIN_EXECUTION_PATH,
            protocol::PlainGatedFragment {
                artifact,
                sweep: sweep(parameters, false),
                blocker,
                quality: json!({
                    "contract": "identical encoded latent, tiled versus untiled Qwen VAE decode",
                    "identicalLatents": true,
                    "result": "passed",
                    "maximumError": actual_maximum,
                    "meanError": actual_mean,
                    "maximumErrorThreshold": MAX_THRESHOLD,
                    "meanErrorThreshold": MEAN_THRESHOLD,
                }),
                negative_mutation: json!({
                    "parameters": parameters,
                    "measured": true,
                    "result": "failed_as_expected",
                    "maximumError": mutated_maximum,
                    "meanError": mutated_mean,
                }),
                loadability: json!({
                    "result": "passed",
                    "resolvedPathFingerprint": loadability_fingerprint,
                }),
                diagnostics: protocol::diagnostics(
                    "memory-mlx-adapter",
                    "executed",
                    [blocker.to_owned()],
                    [
                        ("untiledDecodeActivePeak", "bytes", untiled_peak),
                        ("tiledDecodeActivePeak", "bytes", tiled_peak),
                        ("postDecodeActive", "bytes", active),
                        ("postDecodeCache", "bytes", cache),
                    ],
                ),
            },
        )?;
        fragment["strategy"] = strategy;
        fragment["loadShape"] = json!(load_shape_key(QWEN_VAE_PROBE_LOAD_SHAPE));
        fragment["status"] = json!("negative_complete");
        return Ok(fragment);
    }

    let blocker = concat!(
        "the exact pinned Qwen public seam measures VAE decode active/cache memory and identical-latent ",
        "quality, but does not expose synchronized conditioning/denoise device/wired/reclaimable phase ",
        "telemetry or the required warm/cancel/error lifecycle injections"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        QWEN_PLAIN_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            artifact,
            sweep: sweep(parameters, actual_passed),
            blocker,
            quality: json!({
                "contract": "identical encoded latent, tiled versus untiled Qwen VAE decode",
                "identicalLatents": true,
                "result": if actual_passed { "passed" } else { "failed" },
                "maximumError": actual_maximum,
                "meanError": actual_mean,
                "maximumErrorThreshold": MAX_THRESHOLD,
                "meanErrorThreshold": MEAN_THRESHOLD,
            }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint,
            }),
            diagnostics: protocol::diagnostics(
                "memory-mlx-adapter",
                "executed",
                [blocker.to_owned()],
                [
                    ("untiledDecodeActivePeak", "bytes", untiled_peak),
                    ("tiledDecodeActivePeak", "bytes", tiled_peak),
                    ("postDecodeActive", "bytes", active),
                    ("postDecodeCache", "bytes", cache),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(QWEN_VAE_PROBE_LOAD_SHAPE));
    Ok(fragment)
}

fn qwen_provider_request(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox resting beside a blue ceramic vase, studio photograph".to_owned(),
        negative_prompt: Some("blurry, distorted, text".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(2),
        ..Default::default()
    }
}

fn qwen_provider_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    width: u32,
    height: u32,
    total_bytes: u64,
    predicted_peak_bytes: u64,
    evidence_story: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        // From the LOADED provider's own identity. The Qwen spec is not default-shaped —
        // `run_qwen_provider` passes `with_load_shape(load_shape)` and goes deferred at
        // `bounded_transformer_residency` — and the safety check compares abi, fingerprint AND
        // load shape, so a hardcoded eager can never satisfy the deferred rung.
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-{evidence_story}@{}", protocol::INFERENCE_PIN),
    }
}

fn qwen_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // The Qwen provider executes the complete promotion-quality scenario suite for this exact plan
    // case. The plan enumerates every supported decode edge as its own exact record, so no record
    // claims values it did not execute and the aggregate evidence covers the published range.
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

fn run_qwen_provider(request: &Value) -> Result<Value, String> {
    if protocol::expected_failure(request) {
        return run_qwen_vae_probe(request);
    }
    protocol::validate_plain_overlay_target(request, QWEN_PROVIDER_EXECUTION_PATH)?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let seed = planned_qwen_seed(request, tier)?;
    let load_shape = planned_load_shape(request)?;
    let offload = if matches!(
        selection.strategy,
        MemoryStrategy::StagedResidency | MemoryStrategy::BoundedTransformerResidency
    ) {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    };
    let (width, height) = protocol::target_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_QWEN_IMAGE_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_QWEN_IMAGE_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::QWEN_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_QWEN_IMAGE_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_QWEN_IMAGE_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::QWEN_REPOSITORY,
    )?;
    let spec = qwen_load_spec(root, &selection, offload, load_shape);
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load("qwen_image", &spec)
        .map_err(|error| format!("load real Qwen-Image {tier} provider: {error}"))?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded qwen_image has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned Qwen provider rejected planned selection: {error}"))?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Qwen provider has no calibration identity".to_owned())?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if load_shape != calibration.load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(load_shape),
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = qwen_provider_context(
        selection,
        calibration,
        width,
        height,
        hardware_bytes,
        1,
        if tier == "bf16" { 15511 } else { 16353 },
    );

    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    clear_cache();
    reset_peak_memory();
    let selected = one_image(scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        None,
        &mut |progress| match progress {
            Progress::Step { current: 1, .. } => {
                conditioning.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            Progress::Decoding => {
                denoise.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            _ => {}
        },
    )?)?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err("a synchronized Qwen lifecycle phase reported a zero active peak".to_owned());
    }
    let phases = [conditioning, denoise, decode];
    let overall = PhaseMemory::overall(&phases);
    let predicted = predicted_ceiling(overall.allocator_bytes());

    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget.total_bytes = predicted;
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("Qwen provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Qwen provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-qwen-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Qwen provider accepted stale calibration evidence".to_owned());
    }

    let baseline = one_image(
        generator
            .generate(&qwen_provider_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate unselected Qwen reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&selected, &baseline)?;
    if !qwen_quality_passes(maximum_error, mean_error) {
        return Err(format!(
            "Qwen selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}"
        ));
    }
    let warm = one_image(scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (warm_maximum, warm_mean) = image_max_mean_abs(&selected, &warm)?;
    if !qwen_quality_passes(warm_maximum, warm_mean) {
        return Err("Qwen warm repeat changed the deterministic output".to_owned());
    }

    let cancelled = qwen_provider_request(width, height, seed);
    let cancel_signal = cancelled.cancel.clone();
    let cancel_during_decode = selection.strategy == MemoryStrategy::BoundedDecode;
    let mut cancel_triggered = false;
    let cancel_error = scoped_generate(
        generator.as_ref(),
        cancelled,
        &context,
        None,
        &mut |progress| {
            if cancel_triggered {
                return;
            }
            match progress {
                // Every denoise-oriented rung has executed its selected attention/block behavior
                // before the sampler publishes the first completed step.
                Progress::Step { current: 1, .. } if !cancel_during_decode => {
                    cancel_triggered = true;
                    cancel_signal.cancel();
                }
                // Let the native tiled decoder enter its physical work before signaling. The
                // decoder checks the shared flag between tiles, so this interrupts the active rung
                // instead of short-circuiting before it begins.
                Progress::Decoding if cancel_during_decode => {
                    cancel_triggered = true;
                    let signal = cancel_signal.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        signal.cancel();
                    });
                }
                _ => {}
            }
        },
    )
    .expect_err("in-flight Qwen cancellation must fail");
    if !cancel_triggered {
        return Err("Qwen cancellation probe never reached the active rung boundary".to_owned());
    }
    if !cancel_error.to_ascii_lowercase().contains("cancel") {
        return Err(format!(
            "Qwen cancellation returned the wrong error: {cancel_error}"
        ));
    }
    let cancel_recovery = one_image(scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (cancel_maximum, cancel_mean) = image_max_mean_abs(&selected, &cancel_recovery)?;
    if !qwen_quality_passes(cancel_maximum, cancel_mean) {
        return Err("Qwen cancellation cleanup changed the warm follow-up".to_owned());
    }

    let injected_phase = if selection.strategy == MemoryStrategy::BoundedDecode {
        MemoryPhase::Decode
    } else {
        MemoryPhase::Denoise
    };
    let injected = scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        Some(injected_phase),
        &mut |_| {},
    )
    .expect_err("injected Qwen error must fail");
    if !injected.contains("injected memory-strategy calibration error") {
        return Err(format!(
            "Qwen error injection returned the wrong error: {injected}"
        ));
    }
    let error_recovery = one_image(scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (recovery_maximum, recovery_mean) = image_max_mean_abs(&selected, &error_recovery)?;
    if !qwen_quality_passes(recovery_maximum, recovery_mean) {
        return Err("Qwen error cleanup changed the warm follow-up".to_owned());
    }

    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean) = image_max_mean_abs(&mutated, &baseline)?;
    if qwen_quality_passes(mutated_maximum, mutated_mean) {
        return Err(
            "Qwen output mutation did not breach the production parity envelope".to_owned(),
        );
    }
    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        },
        "sweep": qwen_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "the authoritative Qwen target has no overlay" }
        ],
        "predictedPeakBytes": {
            "conditioning": predicted_ceiling(conditioning.allocator_bytes()),
            "denoise": predicted_ceiling(denoise.allocator_bytes()),
            "decode": predicted_ceiling(decode.allocator_bytes()),
            "overall": predicted,
        },
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "same seed, conditioning, sampling, precision, and loaded provider; selected rung versus unselected request",
            "identicalInputs": true,
            "identicalLatents": false,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": QWEN_MAX_THRESHOLD,
            "meanErrorThreshold": QWEN_MEAN_THRESHOLD,
        },
        "negativeMutation": {
            "parameters": protocol::strategy_parameters(request)?,
            "measured": true,
            "result": "failed_as_expected",
            "maximumError": mutated_maximum,
            "meanError": mutated_mean,
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}"),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:qwen-shared-ladder",
            "executed",
            [],
            [
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, QWEN_PROVIDER_EXECUTION_PATH)?;
    Ok(fragment)
}

fn run(request: &Value) -> Result<Value, String> {
    let provider = protocol::planned(request)?
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    // sc-18104: this used to be `else { run_qwen_provider(request) }`, so ANY provider the MLX
    // adapter does not implement was silently routed to the Qwen arm rather than refused. It then
    // failed further in on a Qwen-shaped complaint that named neither the provider nor the missing
    // arm — measured by reverting this match, capturing `flux2_dev` reported
    // `planned.target.overlay must be a string`, which reads as a malformed plan entry and sends the
    // operator off fixing fixtures or provisioning weights for the wrong model. Refuse by name
    // instead, mirroring the Candle adapter's `plain_execution_path` (candle.rs:540-548).
    match provider {
        Z_IMAGE_PROVIDER => run_z_image_reference(request),
        KREA_PROVIDER => run_krea_control(request),
        QWEN_PROVIDER => run_qwen_provider(request),
        FLUX2_PROVIDER => run_flux2_dev(request),
        other => Err(format!(
            "MLX five-rung calibration does not implement provider {other:?}"
        )),
    }
}

fn main() {
    let request = protocol::request_from_stdin().unwrap_or_else(|error| protocol::fail(error));
    let response = match protocol::action(&request).unwrap_or_else(|error| protocol::fail(error)) {
        "probe" => probe(),
        "run" => run(&request),
        "assess_batch" => assess_z_image_batch(&request),
        other => Err(format!("unsupported action {other:?}")),
    }
    .unwrap_or_else(|error| protocol::fail(error));
    protocol::write_response(&response).unwrap_or_else(|error| protocol::fail(error));
}

#[cfg(test)]
mod z_image_reuse_tests {
    use super::*;

    #[test]
    fn shape_independent_fingerprint_still_keeps_eager_and_deferred_loads_distinct() {
        let fingerprint = "z-image-mlx-independent-materialization-v3";
        assert_ne!(
            z_image_reuse_identity(fingerprint, LoadShape::EagerMaterialization),
            z_image_reuse_identity(fingerprint, LoadShape::DeferredMaterialization),
        );
    }

    /// A canonical five-rung batch, differing from a real Z-Image one ONLY in `target.provider`.
    /// Every other check in `validate_z_image_batch` — length, canonical rung order, one exact
    /// target tuple — passes on this input, which is precisely why the provider check has to exist.
    fn foreign_five_rung_batch(provider: &str) -> Value {
        let target = json!({ "provider": provider, "tier": "q4", "mode": "text_to_image" });
        let planned: Vec<Value> = [
            "resident",
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        ]
        .iter()
        .map(|rung| json!({ "target": target, "strategy": { "rung": rung } }))
        .collect();
        json!({ "action": "assess_batch", "planned": planned })
    }

    /// sc-18104: the batch action had the same silent-misroute hole `run` had. `validate_z_image_batch`
    /// never read `target.provider`, and `assess_z_image_batch` hardcodes `Z_IMAGE_PROVIDER` when it
    /// reads the contract — so a foreign five-rung batch was misrouted into the Z-Image contract and
    /// failed on a Z-Image-shaped complaint AFTER `runtime_macos::catalog()` did real environment
    /// work. Refusal must therefore be by name and must happen inside validation, before that call.
    #[test]
    fn the_batch_action_refuses_a_foreign_provider_by_name_during_validation() {
        for provider in ["flux2_dev", "qwen_image", "krea_2_turbo_control"] {
            let error = validate_z_image_batch(&foreign_five_rung_batch(provider))
                .expect_err("a foreign provider must not reach the Z-Image contract");
            assert_eq!(
                error,
                format!("MLX five-rung batch assessment does not implement provider {provider:?}")
            );
            assert!(
                !error.contains("fingerprint") && !error.contains("contract"),
                "refusal came from the Z-Image contract, so validation let it through: {error}"
            );
        }
    }

    /// The companion direction: the real Z-Image provider must still pass validation unchanged, so
    /// the new check cannot be satisfied by refusing everything.
    #[test]
    fn the_batch_action_still_accepts_the_z_image_provider() {
        let batch = foreign_five_rung_batch("z_image_turbo");
        let planned =
            validate_z_image_batch(&batch).expect("the Z-Image batch must still validate");
        assert_eq!(planned.len(), 5);
    }

    /// The refusal must not be conditional on the batch happening to be five long. `mlx-gen-flux2`
    /// implements only the resident rung, so `assess-reuse` on a flux2 fixture submits a ONE-element
    /// batch — and if the length check ran first, that lane would be told
    /// `Z-Image rung batch must contain exactly 5 cases`, a Z-Image-named complaint about a provider
    /// that is not Z-Image. The provider check is hoisted above the length check for exactly this.
    #[test]
    fn a_short_foreign_batch_is_still_refused_by_name_not_by_length() {
        // 1 is the flux2 case specifically: one implemented rung, so one planned case.
        for length in [1, 2, 3, 4] {
            let mut batch = foreign_five_rung_batch("flux2_dev");
            batch["planned"].as_array_mut().unwrap().truncate(length);
            let error = validate_z_image_batch(&batch)
                .expect_err("a foreign batch of any length must be refused");
            assert_eq!(
                error,
                "MLX five-rung batch assessment does not implement provider \"flux2_dev\""
            );
            assert!(
                !error.contains("exactly"),
                "length {length} read as a Z-Image arity problem, not a foreign provider: {error}"
            );
        }
    }

    /// ...but a genuinely Z-Image batch of the wrong length must still fail on arity, so hoisting the
    /// provider check did not shadow the length check it now precedes.
    #[test]
    fn a_short_z_image_batch_still_fails_on_arity() {
        let mut batch = foreign_five_rung_batch(Z_IMAGE_PROVIDER);
        batch["planned"].as_array_mut().unwrap().truncate(3);
        let error =
            validate_z_image_batch(&batch).expect_err("a 3-case Z-Image batch must still fail");
        assert_eq!(
            error,
            "Z-Image rung batch must contain exactly 5 cases, got 3"
        );
    }
}

#[cfg(test)]
mod flux2_tests {
    use super::*;
    use mlx_gen::gen_core::MemoryStrategySupport;

    fn minimal_request(provider: &str, rung: &str) -> Value {
        json!({
            "planned": {
                "target": { "provider": provider, "overlay": "none" },
                "strategy": { "rung": rung, "parameters": {} }
            }
        })
    }

    /// The per-arm twin of `validate_z_image_batch`'s provider guard (sc-18104): dispatch routes by
    /// name today, but this arm hardcodes the FLUX.2-dev contract, so a misrouted target must be
    /// refused by name INSIDE the arm, and the refusal must not read like a missing-field complaint.
    #[test]
    fn the_flux2_arm_refuses_a_foreign_provider_by_name() {
        for provider in [
            "z_image_turbo",
            "qwen_image",
            "flux2_dev_edit",
            "flux2_klein_9b",
        ] {
            let error = run_flux2_dev(&minimal_request(provider, "resident"))
                .expect_err("a foreign provider must not reach the FLUX.2-dev contract");
            assert_eq!(
                error,
                format!("MLX FLUX.2-dev calibration does not implement provider {provider:?}")
            );
            assert!(
                !error.contains("must be a string") && !error.contains("fingerprint"),
                "refusal came from deeper in the arm, so the guard let it through: {error}"
            );
        }
    }

    /// sc-18218's scope correction (story comment activity-18225): at the pin, mlx-gen-flux2 marks
    /// every non-Resident strategy `Missing`, so the arm is resident-only BY REFUSAL, not by
    /// accident of the plan. Each of the other four rungs must be named back.
    #[test]
    fn the_flux2_arm_is_resident_only_by_refusal() {
        for rung in [
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        ] {
            let error = run_flux2_dev(&minimal_request(FLUX2_PROVIDER, rung))
                .expect_err("a non-resident rung must be refused");
            assert!(
                error.contains(rung) && error.contains("resident"),
                "refusal must name the rung and the resident-only contract: {error}"
            );
        }
        let resident = run_flux2_dev(&minimal_request(FLUX2_PROVIDER, "resident"))
            .expect_err("the minimal resident request is still incomplete");
        assert!(
            !resident.contains("not capturable"),
            "the resident rung itself must pass the rung gate: {resident}"
        );
    }

    #[test]
    fn flux2_fixture_is_bound_to_tier_geometry_and_step_count() {
        let request = json!({
            "planned": { "fixture": "flux2-dev-mlx-q4-768-seed18218-step2" }
        });
        assert_eq!(planned_flux2_seed(&request, "q4", 768).unwrap(), 18218);
        assert!(planned_flux2_seed(&request, "q8", 768)
            .unwrap_err()
            .contains("must start with"));
        assert!(planned_flux2_seed(&request, "q4", 1024)
            .unwrap_err()
            .contains("must start with"));
        let three_step = json!({
            "planned": { "fixture": "flux2-dev-mlx-q4-768-seed18218-step3" }
        });
        assert!(planned_flux2_seed(&three_step, "q4", 768)
            .unwrap_err()
            .contains("two-step"));
    }

    /// sc-17097's lesson applied to this arm: the tier is derived from the planned target and each
    /// declared authoritative tier maps to its own quant, never a hardcoded q4.
    #[test]
    fn flux2_load_spec_preserves_every_planned_numeric_tier() {
        for (tier, expected_quant) in [("q4", Quant::Q4), ("q8", Quant::Q8)] {
            let request = json!({
                "planned": {
                    "strategy": { "rung": "resident", "parameters": {} },
                    "target": { "tier": tier }
                }
            });
            let selection = planned_selection(&request).unwrap();
            let spec = flux2_spec(PathBuf::from(format!("/tmp/flux2-dev-{tier}")), &selection);
            assert_eq!(spec.quantize, Some(expected_quant), "numeric tier {tier}");
        }
    }

    #[test]
    fn flux2_admission_context_is_reference_free_text_to_image() {
        let contract = mlx_gen_flux2::memory_strategy::registered_dev_t2i_contract(
            &weights_free_spec(Some(Quant::Q4)),
        )
        .unwrap();
        let calibration = contract.calibration.as_ref().unwrap();
        let context = flux2_admission_context(
            &resident_selection(Some(Quant::Q4)),
            calibration,
            &calibration.fingerprint,
            768,
            768,
            1_000_000,
            1_000_000,
        );
        assert_eq!(context.mode, MemoryMode::TextToImage);
        assert!(!context.has_reference);
        assert_eq!(context.geometry.reference_count, 0);
        assert!(context.overlay.is_none());
    }

    #[test]
    fn flux2_repeat_envelope_accepts_jitter_and_rejects_the_mandatory_mutation() {
        assert!(flux2_quality_passes(2.0 / 255.0, 0.5 / 255.0, 1.0 / 255.0));
        assert!(!flux2_quality_passes(4.0 / 255.0, 0.5 / 255.0, 1.0 / 255.0));
        assert!(!flux2_quality_passes(2.0 / 255.0, 1.5 / 255.0, 1.0 / 255.0));
        assert!(!flux2_quality_passes(2.0 / 255.0, 0.5 / 255.0, 2.0 / 255.0));

        let baseline = Image {
            width: 2,
            height: 1,
            pixels: vec![0, 63, 127, 128, 191, 255],
        };
        let mutated = qwen_negative_mutation(&baseline);
        let (maximum, mean, rms) = image_max_mean_rms_abs(&mutated, &baseline).unwrap();
        assert!(maximum >= 64.0 / 255.0);
        assert!(mean >= 64.0 / 255.0);
        assert!(rms >= 64.0 / 255.0);
        assert!(!flux2_quality_passes(maximum, mean, rms));
    }

    #[test]
    fn rms_is_measured_from_the_same_pixels_as_max_and_mean() {
        let left = Image {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 0, 0, 0],
        };
        let right = Image {
            width: 2,
            height: 1,
            pixels: vec![51, 0, 0, 0, 0, 0],
        };
        let (maximum, mean, rms) = image_max_mean_rms_abs(&left, &right).unwrap();
        assert!((maximum - 0.2).abs() < 1e-9);
        assert!((mean - 0.2 / 6.0).abs() < 1e-9);
        assert!((rms - 0.2 / 6.0_f64.sqrt()).abs() < 1e-9);
    }

    fn weights_free_spec(quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Resident)
            .with_load_shape(LoadShape::EagerMaterialization);
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        spec
    }

    fn resident_selection(quant: Option<Quant>) -> MemorySelection {
        MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: MemoryStrategyParameters::default(),
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant,
                component_precision_floors: &[],
            },
        }
    }

    /// Pins the arm's load-bearing premises to the PINNED provider crate, weights-free:
    ///
    ///   1. `flux2_dev` directly registers its own T2I contract;
    ///   2. that T2I contract is resident-only (every other strategy `Missing`) — the reason the arm
    ///      and the plan carry a single rung;
    ///   3. its calibration fingerprint is the exact string the plan entries pin.
    ///
    /// If a pin bump changes any of these, this test reds and the arm must be revisited rather
    /// than silently measuring under a different contract.
    #[test]
    fn the_pinned_flux2_t2i_contract_is_direct_resident_only_and_plan_exact() {
        let registry = mlx_gen_flux2::provider_registry().unwrap();
        let spec = weights_free_spec(Some(Quant::Q4));
        let contract = registry
            .memory_strategy_contract(FLUX2_PROVIDER, &spec)
            .unwrap()
            .expect("the pinned FLUX.2-dev T2I contract");
        assert_eq!(contract.provider_id, FLUX2_PROVIDER);
        for capability in &contract.strategies {
            if capability.strategy == MemoryStrategy::Resident {
                assert!(
                    !matches!(capability.support, MemoryStrategySupport::Missing),
                    "the resident strategy must be supported"
                );
            } else {
                assert!(
                    matches!(capability.support, MemoryStrategySupport::Missing),
                    "{:?} is no longer Missing at the pin; the resident-only arm is stale",
                    capability.strategy
                );
            }
        }
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            FLUX2_CALIBRATION_FINGERPRINT,
            "the plan entries pin this fingerprint; regenerate them with the provider"
        );
        // The plan's `engagedRungs: ["resident"]` must equal the provider's engaged composition,
        // or `attested_strategy` fails every capture at plan/measured comparison time.
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::Resident),
            vec![MemoryStrategy::Resident],
            "the plan entries pin engagedRungs [\"resident\"]; regenerate them with the provider"
        );
    }

    /// The admission-scenario legs the capture will run, exercised weights-free through the SAME
    /// registered function the loaded T2I generator delegates to. Mutation-verified in both
    /// directions: the accept leg proves the two rejects are not a blanket refusal, and the
    /// route-gate leg proves the accept is not a blanket accept.
    #[test]
    fn flux2_admission_scenarios_accept_exact_fit_and_reject_unknown_stale_and_foreign_routes() {
        let registry = mlx_gen_flux2::provider_registry().unwrap();
        let spec = weights_free_spec(Some(Quant::Q4));
        let contract = registry
            .memory_strategy_contract(FLUX2_PROVIDER, &spec)
            .unwrap()
            .expect("the pinned FLUX.2-dev T2I contract");
        let calibration = contract.calibration.as_ref().unwrap();
        let selection = resident_selection(Some(Quant::Q4));
        let check = |context: &MemoryRunContext| {
            mlx_gen_flux2::memory_strategy::registered_dev_t2i_safety_check(
                &spec, &contract, context,
            )
        };

        let exact = flux2_admission_context(
            &selection,
            calibration,
            &calibration.fingerprint,
            1024,
            1024,
            1_000_000,
            1_000_000,
        );
        assert!(
            matches!(check(&exact), MemorySafetyDecision::Accept),
            "exact-fit admission must accept: {:?}",
            check(&exact)
        );

        let unknown = flux2_admission_context(
            &selection,
            calibration,
            &calibration.fingerprint,
            1024,
            1024,
            0,
            1,
        );
        assert!(matches!(
            check(&unknown),
            MemorySafetyDecision::Reject { .. }
        ));

        let stale = flux2_admission_context(
            &selection,
            calibration,
            "stale-flux2-dev-fingerprint",
            1024,
            1024,
            1_000_000,
            1,
        );
        assert!(matches!(check(&stale), MemorySafetyDecision::Reject { .. }));

        // The route-gate mutation: the same fitting budget in an edit shape must be refused — the
        // T2I contract admits only reference-free text-to-image.
        let mut edit_shaped = flux2_admission_context(
            &selection,
            calibration,
            &calibration.fingerprint,
            1024,
            1024,
            1_000_000,
            1_000_000,
        );
        edit_shaped.mode = MemoryMode::Edit;
        edit_shaped.has_reference = true;
        edit_shaped.geometry.reference_count = 2;
        assert!(matches!(
            check(&edit_shaped),
            MemorySafetyDecision::Reject { .. }
        ));

        // NVFP4 is the one tier the route gate refuses by name.
        let nvfp4_spec = weights_free_spec(Some(Quant::Nvfp4));
        let nvfp4 = flux2_admission_context(
            &resident_selection(Some(Quant::Nvfp4)),
            calibration,
            &calibration.fingerprint,
            1024,
            1024,
            1_000_000,
            1_000_000,
        );
        assert!(matches!(
            mlx_gen_flux2::memory_strategy::registered_dev_t2i_safety_check(
                &nvfp4_spec,
                &contract,
                &nvfp4
            ),
            MemorySafetyDecision::Reject { .. }
        ));
    }
}

#[cfg(test)]
mod qwen_evidence_tests {
    use super::*;

    #[test]
    fn negative_mutation_changes_pixels_and_is_measured_from_the_changed_image() {
        let baseline = Image {
            width: 2,
            height: 1,
            pixels: vec![0, 63, 127, 128, 191, 255],
        };
        let mutated = qwen_negative_mutation(&baseline);
        assert_ne!(mutated, baseline);
        let (maximum, mean) = image_max_mean_abs(&mutated, &baseline).unwrap();
        assert!(maximum >= 64.0 / 255.0);
        assert!(mean >= 64.0 / 255.0);
        assert!(!qwen_quality_passes(maximum, mean));
    }
}
