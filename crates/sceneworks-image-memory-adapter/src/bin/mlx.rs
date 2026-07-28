#[cfg(not(target_os = "macos"))]
compile_error!("image-memory-mlx-adapter is supported only on macOS");

use mlx_gen::tiling::{SpatialTiling, TilingConfig};
use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_memory_limit, get_peak_memory,
    reset_peak_memory,
};
use mlx_rs::Array;
use runtime_macos::providers::qwen_image::{load_vae, QwenVae};
use sceneworks_image_memory_adapter as protocol;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Command;

const EDGES: [u32; 7] = [768, 640, 512, 448, 384, 320, 256];
const MAX_THRESHOLD: f64 = 3e-2;
const MEAN_THRESHOLD: f64 = 3e-3;

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
    let mlx_memory_limit = get_memory_limit() as u64;
    let wired_limit = std::env::var("SCENEWORKS_MLX_WIRED_LIMIT_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .or_else(|| {
            sysctl("kern.memorystatus_wired_mem_limit")
                .ok()
                .and_then(|value| value.parse().ok())
        })
        // MLX's untouched default memory limit is 1.5x Metal's
        // recommendedMaxWorkingSetSize. This is the same current-host policy
        // derivation used by the production worker before it mutates the MLX
        // limit. Divide before multiplying so rounding stays below the ceiling.
        .or_else(|| Some(mlx_memory_limit / 3 * 2))
        .filter(|value: &u64| *value > 0)
        .ok_or_else(|| {
            "cannot resolve a nonzero wired ceiling from the host sysctl or MLX default memory limit; set SCENEWORKS_MLX_WIRED_LIMIT_BYTES from the current host policy"
                .to_owned()
        })?;
    Ok(json!({
        "hardware": {
            "probe": "sysctl + sw_vers + system_profiler + mlx_rs::memory::get_memory_limit",
            "memoryBytes": memory_bytes,
            "model": sysctl("hw.model")?,
            "chip": sysctl("machdep.cpu.brand_string")?,
            "osVersion": command("/usr/bin/sw_vers", &["-productVersion"])?,
            "metalDevice": metal_device()?,
            "mlxMemoryLimitBytes": mlx_memory_limit,
            "wiredLimitBytes": wired_limit,
        }
    }))
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

fn max_mean_abs(left: &Array, right: &Array) -> Result<(f64, f64), String> {
    let left = left
        .reshape(&[-1])
        .map_err(|error| format!("flatten baseline decode: {error}"))?;
    let right = right
        .reshape(&[-1])
        .map_err(|error| format!("flatten tiled decode: {error}"))?;
    let left = left.as_slice::<f32>();
    let right = right.as_slice::<f32>();
    if left.len() != right.len() || left.is_empty() {
        return Err(format!(
            "decode output length mismatch: baseline={} tiled={}",
            left.len(),
            right.len()
        ));
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        let difference = f64::from((left - right).abs());
        maximum = maximum.max(difference);
        sum += difference;
    }
    Ok((maximum, sum / left.len() as f64))
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
        "parameters": { "decodeTileEdge": 256, "decodeOverlap": 32 },
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

fn run(request: &Value) -> Result<Value, String> {
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
    let parameters = protocol::strategy_parameters(request)?;
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
    let (maximum, mean) = max_mean_abs(&baseline, &tiled)?;
    let passed = maximum <= MAX_THRESHOLD && mean <= MEAN_THRESHOLD;
    let expected_failure = protocol::expected_failure(request);
    if expected_failure {
        if passed {
            return Err(format!(
                "negative mutation {tile_edge}/{overlap} did not breach the identical-latent threshold: max={maximum:.6}, mean={mean:.6}"
            ));
        }
        let blocker =
            "negative mutation is measured, but negative evidence cannot verify a production range";
        let mut fragment = protocol::gated_fragment(
            artifact,
            sweep(parameters, false),
            blocker,
            json!({
                "contract": "identical encoded latent, tiled versus untiled Qwen VAE decode",
                "identicalLatents": true,
                "result": "failed",
                "maximumError": maximum,
                "meanError": mean,
                "maximumErrorThreshold": MAX_THRESHOLD,
                "meanErrorThreshold": MEAN_THRESHOLD,
            }),
            json!({
                "parameters": parameters,
                "measured": true,
                "result": "failed_as_expected",
                "maximumError": maximum,
                "meanError": mean,
            }),
            json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint,
            }),
            protocol::diagnostics(
                "image-memory-mlx-adapter",
                "executed",
                [blocker.to_owned()],
                [
                    ("untiledDecodeActivePeak", "bytes", untiled_peak),
                    ("tiledDecodeActivePeak", "bytes", tiled_peak),
                    ("postDecodeActive", "bytes", active),
                    ("postDecodeCache", "bytes", cache),
                ],
            ),
        );
        fragment["status"] = json!("negative_complete");
        return Ok(fragment);
    }

    let blocker = concat!(
        "the exact pinned Qwen public seam measures VAE decode active/cache memory and identical-latent ",
        "quality, but does not expose synchronized conditioning/denoise device/wired/reclaimable phase ",
        "telemetry or the required warm/cancel/error lifecycle injections"
    );
    Ok(protocol::gated_fragment(
        artifact,
        sweep(parameters, passed),
        blocker,
        json!({
            "contract": "identical encoded latent, tiled versus untiled Qwen VAE decode",
            "identicalLatents": true,
            "result": if passed { "passed" } else { "failed" },
            "maximumError": maximum,
            "meanError": mean,
            "maximumErrorThreshold": MAX_THRESHOLD,
            "meanErrorThreshold": MEAN_THRESHOLD,
        }),
        Value::Null,
        json!({
            "result": "passed",
            "resolvedPathFingerprint": loadability_fingerprint,
        }),
        protocol::diagnostics(
            "image-memory-mlx-adapter",
            "executed",
            [blocker.to_owned()],
            [
                ("untiledDecodeActivePeak", "bytes", untiled_peak),
                ("tiledDecodeActivePeak", "bytes", tiled_peak),
                ("postDecodeActive", "bytes", active),
                ("postDecodeCache", "bytes", cache),
            ],
        ),
    ))
}

fn main() {
    let request = protocol::request_from_stdin().unwrap_or_else(|error| protocol::fail(error));
    let response = match protocol::action(&request).unwrap_or_else(|error| protocol::fail(error)) {
        "probe" => probe(),
        "run" => run(&request),
        other => Err(format!("unsupported action {other:?}")),
    }
    .unwrap_or_else(|error| protocol::fail(error));
    protocol::write_response(&response).unwrap_or_else(|error| protocol::fail(error));
}
