//! Candle/CUDA FLUX 3 Image Klein terminal footprint cells (sc-26243).
//!
//! This is the off-Mac sibling of `footprint_measure.rs`: the same real load + single generation per
//! cell, but CUDA has no worker-side allocator sampler, so memory is measured as the WHOLE-DEVICE
//! delta from `nvidia-smi --query-gpu=memory.used`:
//!
//! * baseline before the load;
//! * peak = max sample during load, each progress tick, and after generation;
//! * resident after generation = device used − baseline.
//!
//! The receipt records `measurementScope: "whole-device-delta-candle"` and the physical device id.
//! Treat these as device-residency numbers, not an allocator breakdown.
//!
//! ## Running ONE cell per process
//!
//! All tests are `#[ignore]`d and MUST be invoked one at a time: the whole-device baseline and peak
//! are process/device-global, so a second cell in the same test binary would inherit the first cell's
//! residency. The module has the same one-shot guard as the MLX harness and panics if a second cell
//! starts in the same process.
//!
//! ```text
//! FLUX3_FOOTPRINT_OUT_DIR=/absolute/outside/the/repo \
//!   cargo test -p sceneworks-worker --release --features backend-candle \
//!     flux3_candle_footprint::footprint_flux3_klein_candle_q8 -- --ignored --exact --nocapture
//! ```
//!
//! Required environment:
//! * `FLUX3_FOOTPRINT_OUT_DIR` — required, absolute, outside the checkout. The cell retains
//!   `<model>_<tier>_candle.png` there; the printed `[[FOOTPRINT]]` JSON is the receipt.
//! * `SCENEWORKS_DATA_DIR` — optional; unset uses the production `Settings` platform default.
//! * `CUDA_VISIBLE_DEVICES` — optional; the first entry selects the `nvidia-smi --id`, default `0`.
//! * `FP_PROMPT` — optional prompt override; resolution/steps/seed are fixed by the cell contract.
//!
//! Dense `bf16` and publisher `fp8r`/`nvfp4` packages are read in place from the pinned snapshot
//! (`variants/<tier>` for the publisher formats). Native `q4`/`q8` are NOT downloaded: they must
//! already exist under `<data_dir>/models/derived/<model>/<tier>/<conversion>/` from an OFFLINE CPU
//! conversion. A missing or hash-mismatched derived package fails with its exact expected path and
//! the conversion command and never falls back to bf16. Nothing here downloads or converts.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

use crate::smoke_support::{
    env_or, flux3_fit_receipt, flux3_footprint_png_path, image_std, save_png, sha256_file_hex,
    DEGENERATE_STD_FLOOR_DEFAULT,
};

/// One-cell-per-process guard, matching the MLX harness's `FOOTPRINT_RAN` invariant.
static FOOTPRINT_RAN: AtomicBool = AtomicBool::new(false);

const FLUX3_DISTILLED: &str = "flux3_klein";
const FLUX3_BASE: &str = "flux3_klein_base";
const FLUX3_WIDTH: u32 = 1024;
const FLUX3_HEIGHT: u32 = 1024;
const FLUX3_SEED: u64 = 26243;
const FLUX3_DISTILLED_STEPS: u32 = 16;
const FLUX3_BASE_STEPS: u32 = 50;
const FLUX3_BASE_GUIDANCE: f32 = 3.5;
const FLUX3_CANDLE_SCOPE: &str = "whole-device-delta-candle";
const MIB: u64 = 1024 * 1024;

fn flux3_manifest_entry(model: &str) -> serde_json::Value {
    crate::tests::builtin_model_entry(model)
}

fn flux3_dense_row<'a>(model: &str, entry: &'a serde_json::Value) -> &'a serde_json::Value {
    entry["downloads"]
        .as_array()
        .unwrap_or_else(|| panic!("{model}: manifest entry has no downloads array"))
        .iter()
        .find(|row| row.get("variant").and_then(serde_json::Value::as_str) == Some("bf16"))
        .unwrap_or_else(|| panic!("{model}: manifest entry has no dense bf16 download row"))
}

fn flux3_snapshot_root(
    model: &str,
    entry: &serde_json::Value,
    settings: &crate::Settings,
) -> PathBuf {
    let row = flux3_dense_row(model, entry);
    let repo = row["repo"]
        .as_str()
        .unwrap_or_else(|| panic!("{model}: dense row has no repo"));
    let revision = row["revision"]
        .as_str()
        .unwrap_or_else(|| panic!("{model}: dense row has no pinned revision"));
    let root = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        repo,
        revision,
    )
    .unwrap_or_else(|| {
        panic!(
            "{model}: the pinned dense FLUX 3 snapshot is not installed (repo {repo}@{revision} under \
             {}); install it through Model Manager or point SCENEWORKS_DATA_DIR at the installed data \
             directory",
            settings.data_dir.display()
        )
    });
    for member in ["config.json", "model.safetensors"] {
        let path = root.join(member);
        assert!(
            path.is_file(),
            "{model}: pinned dense snapshot is incomplete; expected {}",
            path.display()
        );
    }
    root
}

fn candle_platform() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

fn flux3_derivation_row<'a>(
    model: &str,
    tier: &str,
    platform: &str,
    entry: &'a serde_json::Value,
) -> &'a serde_json::Value {
    entry["downloads"]
        .as_array()
        .unwrap_or_else(|| panic!("{model}: manifest entry has no downloads array"))
        .iter()
        .find(|row| {
            row.get("variant").and_then(serde_json::Value::as_str) == Some(tier)
                && row.get("localDerivation").is_some()
                && row
                    .get("platforms")
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(|platforms| platforms.iter().any(|p| p.as_str() == Some(platform)))
        })
        .unwrap_or_else(|| {
            panic!("{model}: the catalog declares no {tier} local derivation for {platform}")
        })
}

fn require_flux3_derived_package(
    model: &str,
    tier: &str,
    platform: &str,
    entry: &serde_json::Value,
    settings: &crate::Settings,
    snapshot_root: &Path,
) -> PathBuf {
    use sceneworks_core::model_artifacts::artifact_selection::{
        derived_snapshot_state, local_derivation, local_derivation_snapshot_dir,
        DerivedSnapshotState,
    };

    let row = flux3_derivation_row(model, tier, platform, entry);
    let derivation = local_derivation(row)
        .unwrap_or_else(|| panic!("{model}: the {tier} localDerivation row is malformed"));
    let expected = local_derivation_snapshot_dir(&settings.data_dir, model, tier, &derivation)
        .unwrap_or_else(|| {
            panic!(
                "{model}: the {tier} derivation path has a non-plain model/tier/conversion segment"
            )
        });
    match derived_snapshot_state(&expected, &derivation) {
        DerivedSnapshotState::Verified => expected,
        state => panic!(
            "{model}: the {tier} package is not installed/verified at exactly {}\n\
             state: {state:?}\n\
             Derived tiers are produced OFFLINE on the CPU first (never downloaded, never replaced by \
             bf16). Convert it once, then re-run this cell:\n\
             \n  FLUX3_CAPTURE_MODEL_ID={model} \\\n  FLUX3_CAPTURE_SOURCE_DIR={} \\\n  FLUX3_CAPTURE_OUT_DIR={} \\\n  FLUX3_CAPTURE_CONVERSIONS={} \\\n  cargo test -p sceneworks-worker --lib local_derivation::tests::flux3_capture_conversion_artifacts -- --ignored --nocapture\n",
            expected.display(),
            snapshot_root.display(),
            settings.data_dir.join("models").join("derived").display(),
            derivation.conversion,
        ),
    }
}

fn require_published_variant(model: &str, tier: &str, snapshot_root: &Path) -> PathBuf {
    let package = snapshot_root.join("variants").join(tier);
    for member in ["config.json", "model.safetensors"] {
        let path = package.join(member);
        assert!(
            path.is_file(),
            "{model}: the {tier} publisher package is incomplete; expected exactly {}",
            path.display()
        );
    }
    package
}

fn flux3_image_request(
    model: &str,
    tier: &str,
    entry: &serde_json::Value,
) -> sceneworks_core::image_request::ImageRequest {
    let steps = if model == FLUX3_BASE {
        FLUX3_BASE_STEPS
    } else {
        FLUX3_DISTILLED_STEPS
    };
    let guidance = (model == FLUX3_BASE).then_some(FLUX3_BASE_GUIDANCE);
    let payload = serde_json::json!({
        "projectId": "sc-26243-footprint",
        "model": model,
        "prompt": env_or(
            "FP_PROMPT",
            "a photorealistic portrait of a red fox in a sunlit autumn forest, sharp focus",
        ),
        "width": FLUX3_WIDTH,
        "height": FLUX3_HEIGHT,
        "count": 1,
        "seed": FLUX3_SEED,
        "advanced": {"quantTier": tier, "steps": steps, "guidanceScale": guidance},
        "modelManifestEntry": entry,
    });
    sceneworks_core::image_request::ImageRequest::from_payload(
        payload
            .as_object()
            .expect("footprint payload is a JSON object"),
    )
}

fn flux3_generation_request(model: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: env_or(
            "FP_PROMPT",
            "a photorealistic portrait of a red fox in a sunlit autumn forest, sharp focus",
        ),
        width: FLUX3_WIDTH,
        height: FLUX3_HEIGHT,
        count: 1,
        seed: Some(FLUX3_SEED),
        steps: Some(if model == FLUX3_BASE {
            FLUX3_BASE_STEPS
        } else {
            FLUX3_DISTILLED_STEPS
        }),
        guidance: (model == FLUX3_BASE).then_some(FLUX3_BASE_GUIDANCE),
        ..Default::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceComputeApp {
    pid: u32,
    process_name: String,
    used_memory_bytes: Option<u64>,
}

/// `nvidia-smi` whole-device memory reader, mirroring the memory adapter's device-delta sampler.
/// The binary is resolved from PATH, exactly as the worker's production `gpu.rs` probe does.
struct NvidiaSmi {
    physical_id: String,
}

impl NvidiaSmi {
    fn resolve() -> Self {
        let physical_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|value| value.split(',').next().map(str::trim).map(str::to_owned))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        Self { physical_id }
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        let output = Command::new("nvidia-smi").args(args).output().unwrap_or_else(|error| {
            panic!(
                "nvidia-smi is required to measure whole-device CUDA memory but was not runnable from PATH: {error}"
            )
        });
        assert!(
            output.status.success(),
            "nvidia-smi {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
        output
    }

    fn id_arg(&self) -> String {
        format!("--id={}", self.physical_id)
    }

    fn used_bytes(&self) -> u64 {
        let id = self.id_arg();
        let output = self.run(&[
            &id,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]);
        let text = String::from_utf8_lossy(&output.stdout);
        let first = text
            .lines()
            .next()
            .unwrap_or_else(|| panic!("nvidia-smi returned no memory.used row"));
        first
            .trim()
            .parse::<u64>()
            .unwrap_or_else(|error| panic!("parse nvidia-smi memory.used {first:?}: {error}"))
            * MIB
    }

    fn compute_cap(&self) -> f32 {
        let id = self.id_arg();
        let output = self.run(&[
            &id,
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ]);
        let text = String::from_utf8_lossy(&output.stdout);
        crate::gpu::parse_selected_compute_cap(&text).unwrap_or_else(|| {
            panic!(
                "nvidia-smi reported no parseable compute_cap for CUDA device {}: {text:?}",
                self.physical_id
            )
        })
    }

    /// Seed the production compute-cap cache before the plan resolves its device floor, exactly as
    /// `discover_gpu` does in the worker (`crate::gpu::cache_compute_cap`).
    fn seed_compute_cap(&self) -> f32 {
        let cap = self.compute_cap();
        crate::gpu::cache_compute_cap(Some(cap));
        cap
    }

    fn compute_apps(&self) -> Vec<DeviceComputeApp> {
        let id = self.id_arg();
        let output = self.run(&[
            &id,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]);
        parse_compute_apps(&String::from_utf8_lossy(&output.stdout))
    }
}

fn parse_compute_apps(output: &str) -> Vec<DeviceComputeApp> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.eq_ignore_ascii_case("No running processes found"))
        .map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            let pid = fields
                .first()
                .and_then(|pid| pid.parse::<u32>().ok())
                .unwrap_or_else(|| panic!("parse nvidia-smi compute app pid from {line:?}"));
            let process_name = fields.get(1).copied().unwrap_or_default().to_owned();
            let used_memory_bytes = fields
                .get(2)
                .and_then(|mib| mib.parse::<u64>().ok())
                .map(|mib| mib * MIB);
            DeviceComputeApp {
                pid,
                process_name,
                used_memory_bytes,
            }
        })
        .collect()
}

fn foreign_compute_apps(apps: &[DeviceComputeApp]) -> Vec<DeviceComputeApp> {
    let own_pid = std::process::id();
    apps.iter()
        .filter(|app| app.pid != own_pid)
        .cloned()
        .collect()
}

fn foreign_pids(apps: &[DeviceComputeApp]) -> std::collections::BTreeSet<u32> {
    apps.iter().map(|app| app.pid).collect()
}

fn foreign_tenants_json(apps: &[DeviceComputeApp]) -> serde_json::Value {
    serde_json::Value::Array(
        apps.iter()
            .map(|app| {
                serde_json::json!({
                    "pid": app.pid,
                    "processName": app.process_name,
                    "usedMemoryBytes": app.used_memory_bytes,
                })
            })
            .collect(),
    )
}

fn assert_device_floor(tier: &str, physical_id: &str, compute_cap: f32) {
    let floor = match tier {
        "fp8r" => crate::gpu::FP8R_MIN_COMPUTE_CAP,
        "nvfp4" => crate::gpu::NVFP4_MIN_COMPUTE_CAP,
        _ => return,
    };
    assert!(
        compute_cap >= floor,
        "CUDA device {physical_id} probed compute capability {compute_cap:.1}, below the {tier} floor {floor:.1}; refusing to measure"
    );
}

fn sample_peak(sampler: &NvidiaSmi, peak: &mut u64) {
    *peak = (*peak).max(sampler.used_bytes());
}

fn measure_candle_cell(
    model: &str,
    tier: &str,
    entry: &serde_json::Value,
    spec: &LoadSpec,
    req: &GenerationRequest,
    weights_dir: &Path,
    sampler: &NvidiaSmi,
) {
    assert!(
        !FOOTPRINT_RAN.swap(true, Ordering::SeqCst),
        "Candle footprint harness measured a second cell in one process — the whole-device baseline \
         and peak are device-global, so run exactly one `footprint_flux3_*_candle_<tier>` per \
         `cargo test … -- --ignored` invocation. The just-attempted {model} {tier} measurement would \
         be corrupt."
    );

    let png_path = flux3_footprint_png_path(model, tier, "candle");
    let baseline_apps = sampler.compute_apps();
    let foreign_baseline = foreign_compute_apps(&baseline_apps);
    let baseline = sampler.used_bytes();
    // A Windows WDDM desktop always has UI/other clients with a CUDA context, so presence alone is
    // not evidence of contention. Use the same 2 GiB baseline admission bound as the installed Candle
    // app lifecycle, and require the tenant pid set to be stable for the whole measurement below.
    assert!(
        baseline <= 2048 * MIB,
        "CUDA device {} baseline device memory {} MiB exceeds the 2048 MiB campaign admission bound \
         (foreign compute apps: {foreign_baseline:?}); refusing to measure a whole-device delta on a \
         busy device",
        sampler.physical_id,
        baseline / MIB,
    );
    let mut peak = baseline;

    println!(
        "[footprint] loading {model} ({tier}) from {} on CUDA device {} (baseline {:.2} GiB) ...",
        weights_dir.display(),
        sampler.physical_id,
        baseline as f64 / 1024.0 / 1024.0 / 1024.0,
    );
    let generator = crate::inference_runtime::load(model, spec)
        .unwrap_or_else(|error| panic!("load {model} ({tier}): {error:?}"));
    sample_peak(sampler, &mut peak);

    let mut last = String::new();
    let output = generator
        .generate(req, &mut |progress| {
            sample_peak(sampler, &mut peak);
            let line = format!("{progress:?}");
            if line != last {
                println!("[progress] {line}");
                last = line;
            }
        })
        .unwrap_or_else(|error| panic!("{model} {tier} generate: {error:?}"));
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };
    assert_eq!(
        (image.width, image.height),
        (req.width, req.height),
        "{model} {tier}: engine returned {}x{}, request was {}x{}",
        image.width,
        image.height,
        req.width,
        req.height
    );
    let std = image_std(&image);
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "{model} {tier} render looks degenerate (std {std:.2}) — measured footprint would be bogus"
    );
    let after = sampler.used_bytes();
    peak = peak.max(after);
    let after_apps = sampler.compute_apps();
    let foreign_after = foreign_compute_apps(&after_apps);
    let baseline_foreign_pids = foreign_pids(&foreign_baseline);
    let after_foreign_pids = foreign_pids(&foreign_after);
    assert_eq!(
        after_foreign_pids, baseline_foreign_pids,
        "CUDA device {} foreign compute processes changed during {model} {tier}: baseline pids {:?}, \
         after pids {:?}; the whole-device delta is contaminated",
        sampler.physical_id, baseline_foreign_pids, after_foreign_pids
    );

    save_png(&image, &png_path);
    let png_sha256 = sha256_file_hex(&png_path);
    let resident = after.saturating_sub(baseline);
    let measured_pixels = u64::from(req.width) * u64::from(req.height);
    let steps = req
        .steps
        .unwrap_or_else(|| panic!("{model} {tier}: footprint request must name steps"));
    let seed = req
        .seed
        .unwrap_or_else(|| panic!("{model} {tier}: footprint request must pin a seed"));

    let receipt = serde_json::json!({
        "model": model,
        "tier": tier,
        "backend": "candle",
        "engineId": model,
        "residentMemoryBytes": resident,
        "peakMemoryBytes": peak,
        "measuredPixels": measured_pixels,
        "width": req.width,
        "height": req.height,
        "steps": steps,
        "seed": seed,
        "pngSha256": png_sha256,
        "measurementScope": FLUX3_CANDLE_SCOPE,
        "physicalDeviceId": sampler.physical_id.clone(),
        "baselineUsedBytes": baseline,
        "afterUsedBytes": after,
        "foreignTenants": foreign_tenants_json(&foreign_after),
        "fit": flux3_fit_receipt(entry, "candle", tier, peak),
    });
    println!("[[FOOTPRINT]] {receipt}");
    println!(
        "[footprint] retained {} (sha256 {})",
        png_path.display(),
        png_sha256
    );
    println!(
        "[footprint] {model} {tier}: device resident {:.2} GiB (used {:.2}, baseline {:.2}) | peak {:.2} GiB | render std {:.1}",
        resident as f64 / 1024.0 / 1024.0 / 1024.0,
        after as f64 / 1024.0 / 1024.0 / 1024.0,
        baseline as f64 / 1024.0 / 1024.0 / 1024.0,
        peak as f64 / 1024.0 / 1024.0 / 1024.0,
        std,
    );
}

/// One Candle FLUX 3 cell: resolve through the production `candle_flux3_plan`, build the same
/// package + quant + component `LoadSpec`, then run the single measured generation.
fn flux3_candle_cell(model: &str, tier: &str) {
    let settings = crate::Settings::from_env();
    let entry = flux3_manifest_entry(model);
    let snapshot_root = flux3_snapshot_root(model, &entry, &settings);
    let request = flux3_image_request(model, tier, &entry);

    // Production seeds this cache in `discover_gpu`; a test process never does, so a publisher tier
    // would resolve against `None` (CPU) and be refused before load. Probe the SAME physical device
    // and seed the SAME cache the planner reads, without bypassing the production floor.
    let sampler = NvidiaSmi::resolve();
    let probed_cap = sampler.seed_compute_cap();
    assert_device_floor(tier, &sampler.physical_id, probed_cap);

    let expected = match tier {
        "bf16" => Some(snapshot_root.clone()),
        "q4" | "q8" => Some(require_flux3_derived_package(
            model,
            tier,
            candle_platform(),
            &entry,
            &settings,
            &snapshot_root,
        )),
        "fp8r" | "nvfp4" => Some(require_published_variant(model, tier, &snapshot_root)),
        other => panic!("{model}: {other} is not a FLUX 3 Candle campaign tier"),
    };

    let descriptor = crate::inference_runtime::media_descriptor(model)
        .unwrap_or_else(|| panic!("{model}: no media descriptor is registered on this build"));
    let plan = crate::image_jobs::flux3_precision::candle_flux3_plan(
        &request,
        &settings,
        &snapshot_root,
        descriptor.capabilities.supported_quants,
    )
    .unwrap_or_else(|error| panic!("{model} {tier}: FLUX 3 Candle plan refused: {error:?}"))
    .unwrap_or_else(|| panic!("{model}: not a FLUX 3 model in the production planner"));
    assert_eq!(
        plan.tier.key(),
        tier,
        "{model}: planner resolved a different package"
    );
    if let Some(expected) = expected {
        if tier == "bf16" {
            assert_eq!(
                plan.weights_dir, expected,
                "{model}: bf16 must load the pinned snapshot root"
            );
        } else {
            let canonical =
                std::fs::canonicalize(&expected).expect("resolved package canonicalizes");
            assert_eq!(
                plan.weights_dir, canonical,
                "{model} {tier}: planner and catalog resolve different packages"
            );
        }
    }

    let mut spec = LoadSpec::new(WeightsSource::Dir(plan.weights_dir.clone()));
    if let Some(quant) = plan.quant() {
        spec = spec.with_quant(quant);
    }
    spec = plan.attach_components(spec);
    if tier == "fp8r" {
        assert_eq!(plan.quant(), Some(Quant::Fp8Rowwise), "{model}: fp8r quant");
    }
    if tier == "nvfp4" {
        assert_eq!(plan.quant(), Some(Quant::Nvfp4), "{model}: nvfp4 quant");
    }

    let generation = flux3_generation_request(model);
    measure_candle_cell(
        model,
        tier,
        &entry,
        &spec,
        &generation,
        &plan.weights_dir,
        &sampler,
    );
}

#[test]
#[ignore = "terminal campaign cell; needs the pinned FLUX 3 dense snapshot + a CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_candle_bf16() {
    flux3_candle_cell(FLUX3_DISTILLED, "bf16");
}

#[test]
#[ignore = "terminal campaign cell; needs the offline-derived flux3_klein Candle q4 package + a CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_candle_q4() {
    flux3_candle_cell(FLUX3_DISTILLED, "q4");
}

#[test]
#[ignore = "terminal campaign cell; needs the offline-derived flux3_klein Candle q8 package + a CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_candle_q8() {
    flux3_candle_cell(FLUX3_DISTILLED, "q8");
}

#[test]
#[ignore = "terminal campaign cell; needs the published flux3_klein fp8r variant + an sm_89+ CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_candle_fp8r() {
    flux3_candle_cell(FLUX3_DISTILLED, "fp8r");
}

#[test]
#[ignore = "terminal campaign cell; needs the published flux3_klein nvfp4 variant + an sm_120+ CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_candle_nvfp4() {
    flux3_candle_cell(FLUX3_DISTILLED, "nvfp4");
}

#[test]
#[ignore = "terminal campaign cell; needs the pinned FLUX 3 base dense snapshot + a CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_base_candle_bf16() {
    flux3_candle_cell(FLUX3_BASE, "bf16");
}

#[test]
#[ignore = "terminal campaign cell; needs the offline-derived flux3_klein_base Candle q4 package + a CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_base_candle_q4() {
    flux3_candle_cell(FLUX3_BASE, "q4");
}

#[test]
#[ignore = "terminal campaign cell; needs the offline-derived flux3_klein_base Candle q8 package + a CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_base_candle_q8() {
    flux3_candle_cell(FLUX3_BASE, "q8");
}

#[test]
#[ignore = "terminal campaign cell; needs the published flux3_klein_base fp8r variant + an sm_89+ CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_base_candle_fp8r() {
    flux3_candle_cell(FLUX3_BASE, "fp8r");
}

#[test]
#[ignore = "terminal campaign cell; needs the published flux3_klein_base nvfp4 variant + an sm_120+ CUDA device + FLUX3_FOOTPRINT_OUT_DIR"]
fn footprint_flux3_klein_base_candle_nvfp4() {
    flux3_candle_cell(FLUX3_BASE, "nvfp4");
}
