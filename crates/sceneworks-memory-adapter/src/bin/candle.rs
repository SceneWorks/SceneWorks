#[cfg(target_os = "macos")]
compile_error!("memory-candle-adapter is supported only on CUDA hosts");

use candle_gen::testkit::{StableIdleConfig, VramProbe};
use runtime_cuda::gen_core::{
    adapter_stack_identity, AdapterKind, AdapterSpec, Conditioning, GenerationMemory,
    GenerationOutput, GenerationRequest, Image, LoadShape, LoadSpec, MemoryBudget,
    MemoryCacheState, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryOptimizationAuthority,
    MemoryPhase, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemorySelection,
    MemoryStrategy, MemoryStrategyParameters, OffloadPolicy, Precision, Progress, Quant,
    TransformerComponent, WeightsSource,
};
use runtime_cuda::providers::pulid::PulidFluxRequest;
// sc-22728: the bespoke Candle edit provider the WORKER drives by name — it is not a registered
// generator, so the capture calls the same constructor the worker does rather than the catalog.
use runtime_cuda::providers::qwen_image::{QwenEdit, QwenEditPaths, QwenEditRequest};
use sceneworks_memory_adapter as protocol;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::process::Command;

// sc-22736: the Wan 2.2 family and SCAIL-2. Four engine providers on one arm, kept in their own
// module for the same reason LTX-2.5's would be — a video family's carrier, rate menu and per-tier
// identity table are a self-contained apparatus.
#[path = "candle_wan_scail2.rs"]
mod candle_wan_scail2;

/// Engine registry ids, not catalog ids: `wan_2_2` is the SceneWorks name for the TI2V-5B route
/// (worker `engines.rs` `video_engine_ids`), and SCAIL-2's two names coincide.
const WAN_TI2V_5B_ID: &str = "wan2_2_ti2v_5b";
const WAN_T2V_A14B_ID: &str = "wan2_2_t2v_14b";
const WAN_I2V_A14B_ID: &str = "wan2_2_i2v_14b";
const SCAIL2_ID: &str = "scail2_14b";

const KREA_ID: &str = "krea_2_turbo";
const KREA_PLAIN_EXECUTION_PATH: &str = "the Candle Krea base-only text-to-image path";
/// The label the Krea arm refuses a non-still geometry under (sc-18808); see
/// [`still_calibration_label`].
const KREA_STILL_CALIBRATION: &str = "Candle Krea base calibration";
const QWEN_ID: &str = "qwen_image";
const QWEN_PLAIN_EXECUTION_PATH: &str = "the Candle Qwen-Image base-only text-to-image path";
/// The label the Qwen arm refuses a non-still geometry under (sc-18808); see
/// [`still_calibration_label`].
const QWEN_STILL_CALIBRATION: &str = "Candle Qwen base calibration";
/// The Z-Image-Turbo provider (sc-15859). Registry id of `candle-gen-z-image`'s Turbo generator —
/// the catalog's `z_image_turbo` route, NOT the base `z_image` provider, which has its own contract
/// and its own plan cells.
const Z_IMAGE_TURBO_ID: &str = "z_image_turbo";
const Z_IMAGE_TURBO_PLAIN_EXECUTION_PATH: &str =
    "the Candle Z-Image-Turbo base-only text-to-image path";
/// The label the Z-Image-Turbo arm refuses a non-still geometry under; see
/// [`still_calibration_label`].
const Z_IMAGE_TURBO_STILL_CALIBRATION: &str = "Candle Z-Image-Turbo base calibration";
/// `z_image_edit` is a catalog alias for the Turbo provider driven in `edit_image` mode (worker
/// `engines.rs`; on Candle the registered Turbo generator's `Conditioning::Reference` route,
/// sc-11783). Its anchors plan `provider: z_image_turbo, mode: edit_image`; the SAME loaded Turbo
/// generator is conditioned on one reference image (sc-22724).
const Z_IMAGE_TURBO_EDIT_EXECUTION_PATH: &str =
    "the Candle Z-Image-Turbo reference-conditioned edit path (the z_image_edit route)";
/// The label the edit route refuses a non-still geometry under. The edit route is its own route
/// with its own plan cells, so it names itself in the refusal rather than borrowing the
/// text-to-image label — the same split the MLX arm carries (`Z_IMAGE_EDIT_ARM.still_calibration`).
const Z_IMAGE_TURBO_EDIT_STILL_CALIBRATION: &str = "Candle Z-Image-Turbo edit calibration";
/// The worker's production edit strength default (`resolve_zimage_edit_init`, `advanced.strength`).
const Z_IMAGE_EDIT_STRENGTH: f32 = 0.6;
/// Edit captures run four steps: the img2img start step is `floor(steps * strength)` (the shared
/// `init_time_step` law), so `4 * 0.6` starts at step 2 and leaves two executed denoise steps —
/// the same two-step conditioning/denoise phase shape the text-to-image captures use.
const Z_IMAGE_EDIT_STEPS: u32 = 4;
/// The undistilled Z-Image BASE provider (sc-22724): registry id of `candle-gen-z-image`'s
/// `base` generator, its own artifact family (`SceneWorks/z-image-mlx`, `SCENEWORKS_Z_IMAGE_BASE_*`)
/// and real CFG in the denoise loop.
const Z_IMAGE_ID: &str = "z_image";
const Z_IMAGE_PLAIN_EXECUTION_PATH: &str = "the Candle Z-Image base-model text-to-image path";
/// The label the Z-Image base arm refuses a non-still geometry under; see
/// [`still_calibration_label`].
const Z_IMAGE_STILL_CALIBRATION: &str = "Candle Z-Image base-model calibration";
/// The FLUX.2 family on Candle (sc-22727). `candle-gen-flux2` registers exactly two txt2img
/// providers — the 32B `flux2_dev` flagship and the distilled `flux2_klein_9b` — and the worker
/// routes THREE catalog models onto them (`crates/sceneworks-worker/src/engines.rs`):
/// `flux2_dev`, `flux2_klein_9b`, and the separately distilled `flux2_klein_9b_kv`, which shares
/// the klein engine id and differs only in its artifact. There is no inline arm: every FLUX.2
/// anchor is a five-rung reference capture.
const FLUX2_DEV_ID: &str = "flux2_dev";
const FLUX2_KLEIN_ID: &str = "flux2_klein_9b";
const FLUX2_DEV_PLAIN_EXECUTION_PATH: &str = "the Candle FLUX.2-dev base-only text-to-image path";
const FLUX2_KLEIN_PLAIN_EXECUTION_PATH: &str =
    "the Candle FLUX.2-klein-9B base-only text-to-image path";
const FLUX2_KLEIN_KV_PLAIN_EXECUTION_PATH: &str =
    "the Candle FLUX.2-klein-9B KV-cache base-only text-to-image path";
const FLUX2_DEV_STILL_CALIBRATION: &str = "Candle FLUX.2-dev base calibration";
const FLUX2_KLEIN_STILL_CALIBRATION: &str = "Candle FLUX.2-klein-9B base calibration";
const FLUX2_KLEIN_KV_STILL_CALIBRATION: &str = "Candle FLUX.2-klein-9B KV calibration";

/// One member of the Candle FLUX.2 family, resolved from the plan's `(target.provider,
/// target.modelId)`. Two members share `provider` and are told apart ONLY by `model_id`, which is
/// also what the worker binds as `LoadSpec::resolved_route`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Flux2Arm {
    /// The registry id handed to `catalog.media().load` — the production loader (E4).
    provider: &'static str,
    /// The catalog model id: the anchor key's `modelId` and the load spec's `resolved_route`.
    model_id: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// The record's diagnostics source, `memory-candle-adapter:<slug>-five-rung-reference`.
    slug: &'static str,
    /// Whether the planned tier's quant reaches the loader as `LoadSpec::quantize` — the same
    /// per-member fact the MLX arm carries, and the worker's own Candle decision:
    /// `candle_quant_for_resolved_tier` (`image_jobs/base.rs`) returns `(None, resolved_bits)` for
    /// a dense-TE turnkey (`is_dense_te_tier`, gated on the manifest's
    /// `mlx.denseTextEncoderTier: true`, which BOTH klein entries declare), and folds the tier
    /// otherwise. `candle-gen-flux2` quantizes the DiT on-the-fly whenever `spec.quantize` is set,
    /// so a klein q4/q8 spec carrying the quant would re-quantize an already-packed transformer and
    /// measure a load the app never performs (E4). Dev takes the fold; both klein members do not.
    tier_quant_reaches_the_loader: bool,
}

const FLUX2_DEV_ARM: Flux2Arm = Flux2Arm {
    provider: FLUX2_DEV_ID,
    model_id: "flux2_dev",
    execution_path: FLUX2_DEV_PLAIN_EXECUTION_PATH,
    still_calibration: FLUX2_DEV_STILL_CALIBRATION,
    repository_env: "SCENEWORKS_FLUX2_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX2_REVISION",
    root_env: "SCENEWORKS_FLUX2_ROOT",
    expected_repository: protocol::FLUX2_REPOSITORY,
    slug: "flux2-dev",
    tier_quant_reaches_the_loader: true,
};

const FLUX2_KLEIN_ARM: Flux2Arm = Flux2Arm {
    provider: FLUX2_KLEIN_ID,
    model_id: "flux2_klein_9b",
    execution_path: FLUX2_KLEIN_PLAIN_EXECUTION_PATH,
    still_calibration: FLUX2_KLEIN_STILL_CALIBRATION,
    repository_env: "SCENEWORKS_FLUX2_KLEIN_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX2_KLEIN_REVISION",
    root_env: "SCENEWORKS_FLUX2_KLEIN_ROOT",
    expected_repository: protocol::FLUX2_KLEIN_REPOSITORY,
    slug: "flux2-klein-9b",
    tier_quant_reaches_the_loader: false,
};

const FLUX2_KLEIN_KV_ARM: Flux2Arm = Flux2Arm {
    provider: FLUX2_KLEIN_ID,
    model_id: "flux2_klein_9b_kv",
    execution_path: FLUX2_KLEIN_KV_PLAIN_EXECUTION_PATH,
    still_calibration: FLUX2_KLEIN_KV_STILL_CALIBRATION,
    repository_env: "SCENEWORKS_FLUX2_KLEIN_KV_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX2_KLEIN_KV_REVISION",
    root_env: "SCENEWORKS_FLUX2_KLEIN_KV_ROOT",
    expected_repository: protocol::FLUX2_KLEIN_KV_REPOSITORY,
    slug: "flux2-klein-9b-kv",
    tier_quant_reaches_the_loader: false,
};

const FLUX2_ARMS: [Flux2Arm; 3] = [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM];

/// Which FLUX.2 member the plan asks for, or `None` when the plan is not a FLUX.2 one at all.
/// A FLUX.2 provider with a model id no member serves is an ERROR, not a `None`: the KV plan must
/// never be satisfied by the base klein artifact, which shares the provider id.
fn flux2_arm(request: &Value) -> Result<Option<Flux2Arm>, String> {
    let provider = planned_provider(request)?;
    if !FLUX2_ARMS.iter().any(|arm| arm.provider == provider) {
        return Ok(None);
    }
    let model_id = protocol::planned(request)?
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    FLUX2_ARMS
        .into_iter()
        .find(|arm| arm.provider == provider && arm.model_id == model_id)
        .map(Some)
        .ok_or_else(|| {
            format!(
                "the Candle FLUX.2 arm does not implement provider {provider:?} for model \
                 {model_id:?}"
            )
        })
}

/// The two FLUX.1 base text-to-image providers (sc-22726). Registry ids of `candle-gen-flux`'s
/// registered generators (`candle_gen_flux::FLUX1_DEV_ID` / `FLUX1_SCHNELL_ID`) — the same ids the
/// worker hands `inference_runtime::load` for the `flux_dev` / `flux_schnell` catalog models
/// (`engines.rs` MODEL_TABLE `engine_id`).
const FLUX1_DEV_ID: &str = "flux1_dev";
const FLUX1_DEV_PLAIN_EXECUTION_PATH: &str = "the Candle FLUX.1-dev base-only text-to-image path";
const FLUX1_DEV_STILL_CALIBRATION: &str = "Candle FLUX.1-dev calibration";
const FLUX1_SCHNELL_ID: &str = "flux1_schnell";
const FLUX1_SCHNELL_PLAIN_EXECUTION_PATH: &str =
    "the Candle FLUX.1-schnell base-only text-to-image path";
const FLUX1_SCHNELL_STILL_CALIBRATION: &str = "Candle FLUX.1-schnell calibration";
/// The three SD3.5 base text-to-image providers (sc-22730). Registry ids of `candle-gen-sd3`'s
/// registered generators (`candle_gen_sd3::MODEL_ID` / `MODEL_ID_TURBO` / `MODEL_ID_MEDIUM`) — the
/// same ids the worker hands `inference_runtime::load`, and the same ids the MLX lane uses, so no
/// aliasing is needed on either side. Each member binds its OWN tiered rehost through its OWN env
/// family: serving one route from another's artifact would re-label that route's peaks.
const SD3_5_LARGE_ID: &str = "sd3_5_large";
const SD3_5_LARGE_PLAIN_EXECUTION_PATH: &str =
    "the Candle SD3.5 Large base-only text-to-image path";
const SD3_5_LARGE_STILL_CALIBRATION: &str = "Candle SD3.5 Large base calibration";
const SD3_5_LARGE_TURBO_ID: &str = "sd3_5_large_turbo";
const SD3_5_LARGE_TURBO_PLAIN_EXECUTION_PATH: &str =
    "the Candle SD3.5 Large Turbo base-only text-to-image path";
const SD3_5_LARGE_TURBO_STILL_CALIBRATION: &str = "Candle SD3.5 Large Turbo base calibration";
const SD3_5_MEDIUM_ID: &str = "sd3_5_medium";
const SD3_5_MEDIUM_PLAIN_EXECUTION_PATH: &str =
    "the Candle SD3.5 Medium base-only text-to-image path";
const SD3_5_MEDIUM_STILL_CALIBRATION: &str = "Candle SD3.5 Medium base calibration";
/// The fixture slug each SD3.5 member's five-rung reference capture carries, after the shared
/// `fresh-five-rung-` prefix. Kept beside the ids so a member can never borrow another's slug.
const SD3_5_FIXTURE_SLUGS: [(&str, &str); 3] = [
    (SD3_5_LARGE_ID, "sd3-5-large"),
    (SD3_5_LARGE_TURBO_ID, "sd3-5-large-turbo"),
    (SD3_5_MEDIUM_ID, "sd3-5-medium"),
];
/// PuLID-FLUX (sc-22726). On this lane it is NOT a registered generator: `candle-gen-pulid`
/// registers nothing (its `lib.rs` says so in as many words, and the checked-in capability dump
/// lists it under `bespokeMemoryRouteWaivers`). The worker loads it as
/// `runtime_cuda::providers::pulid::PulidFlux::load_with_memory_context(&PulidFluxPaths, ctx)`
/// (`image_jobs/pulid_candle.rs`), so this arm does exactly that — going through the registry
/// would be a different code path, not the production one (E4).
const PULID_FLUX_ID: &str = "pulid_flux";
const PULID_FLUX_EXECUTION_PATH: &str = "the Candle PuLID-FLUX identity-conditioned character path";
const PULID_FLUX_STILL_CALIBRATION: &str = "Candle PuLID-FLUX calibration";
/// The manifest's `ui.referenceStrengthDefault` for `pulid_flux_dev` — the `id_weight` the worker
/// sends when the user leaves the reference strength alone (`pulid_candle_id_weight`).
const PULID_FLUX_ID_WEIGHT: f32 = 1.0;
/// Torch/MLX-parity guidance for the `pulid_flux_dev` "photoreal" preset (`pulid_candle_guidance`).
const PULID_FLUX_GUIDANCE: f32 = 4.0;
/// The PuLID capture seed, shared with the MLX arm so both lanes' PuLID fixtures name one number.
/// The two FLUX.1 BASE providers ride the shared five-rung reference path and render at ITS seed
/// ([`FIVE_RUNG_SEED`]) — their fixtures say so.
const FLUX1_SEED: u64 = 22726;
/// The seed every five-rung reference render uses (`five_rung_generation_request`), and the
/// number the `fresh-five-rung-*-seed16402-step2` fixtures carry.
const FIVE_RUNG_SEED: u64 = 16402;
/// The two SANA routes (sc-22731). Registry ids of `candle-gen-sana`'s registered generators
/// (`candle_gen_sana::MODEL_ID` / `SPRINT_MODEL_ID`), which are also the catalog model ids.
///
/// **bf16 is the only tier this lane has.** `candle-gen-sana`'s `validate_load_spec` refuses any
/// `LoadSpec::quantize` by name ("Candle supports only the dense physical tier"), the worker pins
/// the candle SANA tier label to `bf16` outright, and there is no packed SANA artifact off-Mac at
/// all — the only `platforms: ["windows", "linux"]` download is the upstream dense diffusers
/// snapshot. So `sana_1600m:q4:candle` is not an unmeasured cell; it is not a cell.
const SANA_ID: &str = "sana_1600m";
const SANA_PLAIN_EXECUTION_PATH: &str = "the Candle SANA 1.6B dense text-to-image path";
const SANA_STILL_CALIBRATION: &str = "Candle SANA 1.6B calibration";
const SANA_SPRINT_ID: &str = "sana_sprint_1600m";
const SANA_SPRINT_PLAIN_EXECUTION_PATH: &str =
    "the Candle SANA-Sprint 1.6B dense text-to-image path";
const SANA_SPRINT_STILL_CALIBRATION: &str = "Candle SANA-Sprint 1.6B calibration";
/// The three Chroma1 routes (sc-22731). `candle-gen-chroma` registers one generator per route and
/// keeps them as three separate receipt/evidence domains (SC-20788), so each binds its own artifact
/// family and publishes its own per-tier identity.
const CHROMA1_HD_ID: &str = "chroma1_hd";
const CHROMA1_HD_PLAIN_EXECUTION_PATH: &str = "the Candle Chroma1-HD base-only text-to-image path";
const CHROMA1_HD_STILL_CALIBRATION: &str = "Candle Chroma1-HD calibration";
const CHROMA1_BASE_ID: &str = "chroma1_base";
const CHROMA1_BASE_PLAIN_EXECUTION_PATH: &str =
    "the Candle Chroma1-Base base-only text-to-image path";
const CHROMA1_BASE_STILL_CALIBRATION: &str = "Candle Chroma1-Base calibration";
const CHROMA1_FLASH_ID: &str = "chroma1_flash";
const CHROMA1_FLASH_PLAIN_EXECUTION_PATH: &str =
    "the Candle Chroma1-Flash base-only text-to-image path";
const CHROMA1_FLASH_STILL_CALIBRATION: &str = "Candle Chroma1-Flash calibration";
const LTX25_ID: &str = "ltx_2_5_distilled";
const LTX25_EXECUTION_PATH: &str =
    "the Candle LTX-2.5 text-to-video base recipe (including the official dev refinement LoRA)";
const LTX25_DIFFUSION_VAE_COMPONENT: &str = "diffusion_video_vae";
const LTX25_DISTILL_LORA_RELATIVE_PATH: &str =
    "distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors";
const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
// The shipped q4/1024 golden approved mean_absolute_rgb_delta_255 <= 0.01681. Preserve that
// source-of-truth policy exactly. The historical contract did not
// constrain a single outlier channel, so the maximum metric remains diagnostic while the mean is
// the promotion gate. The required broad mutation must still breach at least one envelope bound.
const KREA_CANDLE_MAX_THRESHOLD: f64 = 1.0;
const KREA_CANDLE_MEAN_THRESHOLD: f64 = 0.01681;

fn certifying_wddm_idle_config() -> StableIdleConfig {
    // GPU 1's otherwise-idle WDDM graphics residency measured 1.6 GB in run 33188922159. The
    // pinned testkit's stable-idle proof is deliberately stricter than a raised one-shot ceiling:
    // it repeats the samples, bounds drift, and rejects any pure-compute process before allowing
    // the device-level delta capture.
    StableIdleConfig::new(2.0, 5, 64, 200)
}

fn certifying_vram_probe() -> VramProbe {
    VramProbe::start_rendered().assert_stable_idle(certifying_wddm_idle_config())
}

#[derive(Clone)]
struct NvidiaSmi {
    executable: PathBuf,
    physical_id: String,
}

impl NvidiaSmi {
    fn resolve() -> Result<Self, String> {
        let executable = if cfg!(windows) {
            let system_root =
                std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_owned());
            PathBuf::from(system_root)
                .join("System32")
                .join("nvidia-smi.exe")
        } else {
            PathBuf::from("/usr/bin/nvidia-smi")
        };
        if !executable.is_file() {
            return Err(format!(
                "trusted nvidia-smi path does not exist: {}",
                executable.display()
            ));
        }
        let physical_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|value| value.split(',').next().map(str::trim).map(str::to_owned))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        Ok(Self {
            executable,
            physical_id,
        })
    }

    fn query(&self, fields: &str) -> Result<String, String> {
        let output = Command::new(&self.executable)
            .arg(format!("--id={}", self.physical_id))
            .arg(format!("--query-gpu={fields}"))
            .arg("--format=csv,noheader,nounits")
            .output()
            .map_err(|error| format!("start {}: {error}", self.executable.display()))?;
        if !output.status.success() {
            return Err(format!(
                "{} query failed: {}",
                self.executable.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }

    fn used_bytes(&self) -> Result<u64, String> {
        self.query("memory.used")?
            .parse::<u64>()
            .map(|used_mib| used_mib * MIB)
            .map_err(|error| format!("parse nvidia-smi memory.used: {error}"))
    }
}

fn decimal_gb_to_bytes(value: f64) -> u64 {
    (value * 1.0e9).round() as u64
}

fn cuda_phase_metrics(device_bytes: u64) -> Value {
    // Candle's exact CUDA backend allocates directly through cudarc/CUDA and has no caching
    // allocator counter. On the required idle single-process GPU the `nvidia-smi memory.used` delta
    // is therefore the one truthful residency counter, and it is non-reclaimable: discrete CUDA
    // device allocations are physically non-pageable. So `activeBytes` carries the reading,
    // `reclaimableBytes` is a measured zero, and `allocatorBytes` is their sum by the schema-v5
    // identity. sc-18864 dropped `deviceBytes` and `wiredBytes`, which were further copies of this
    // same number under names claiming to be distinct quantities.
    json!({
        "activeBytes": device_bytes,
        "allocatorBytes": device_bytes,
        "reclaimableBytes": 0,
    })
}

fn nvcc_runtime() -> Result<String, String> {
    let executable = if cfg!(windows) {
        PathBuf::from(protocol::required_env("CUDA_PATH")?)
            .join("bin")
            .join("nvcc.exe")
    } else {
        PathBuf::from("/usr/local/cuda/bin/nvcc")
    };
    let output = Command::new(&executable)
        .arg("--version")
        .output()
        .map_err(|error| format!("start {}: {error}", executable.display()))?;
    if !output.status.success() {
        return Err(format!("{} --version failed", executable.display()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split("release ")
        .nth(1)
        .and_then(|tail| tail.split(',').next())
        .map(str::trim)
        .map(str::to_owned)
        .ok_or_else(|| format!("cannot parse CUDA runtime from {}", executable.display()))
}

fn probe() -> Result<Value, String> {
    let smi = NvidiaSmi::resolve()?;
    let fields = smi.query("index,name,compute_cap,driver_version,memory.total")?;
    let columns: Vec<_> = fields.split(',').map(str::trim).collect();
    if columns.len() != 5 {
        return Err(format!(
            "nvidia-smi returned {} fields instead of 5: {fields:?}",
            columns.len()
        ));
    }
    let total_mib: u64 = columns[4]
        .parse()
        .map_err(|error| format!("parse nvidia-smi memory.total: {error}"))?;
    Ok(json!({
        "hardware": {
            "probe": format!("{} selected through CUDA_VISIBLE_DEVICES", smi.executable.display()),
            "memoryBytes": total_mib * MIB,
            "deviceId": columns[0],
            "name": columns[1],
            "computeCapability": columns[2],
            "driverVersion": columns[3],
            "runtimeVersion": nvcc_runtime()?,
        }
    }))
}

fn sweep(request: &Value, parameters: &Map<String, Value>, result: &str) -> Result<Value, String> {
    let fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    let candidates: &[(u64, u64, u64, u64)] = match fingerprint {
        "krea-turbo-cuda-phase-curves-v1" => &[(512, 128, 134_217_728, 1)],
        "krea-turbo-cuda-phase-curves-v2" => {
            &[(384, 64, 67_108_864, 1), (640, 128, 134_217_728, 2)]
        }
        other => return Err(format!("unknown Krea calibration fingerprint {other:?}")),
    };
    let current = |name: &str| parameters.get(name).and_then(Value::as_u64);
    let rows = candidates
        .iter()
        .map(|(edge, overlap, attention, window)| {
            let selected = current("decodeTileEdge") == Some(*edge)
                && current("decodeOverlap") == Some(*overlap)
                && current("attentionChunkSize") == Some(*attention)
                && current("transformerWindowSize") == Some(*window);
            json!({
                "parameters": {
                    "decodeTileEdge": edge,
                    "decodeOverlap": overlap,
                    "attentionChunkSize": attention,
                    "transformerWindowSize": window,
                },
                "result": if selected { result } else { "not_run" },
            })
        })
        .collect::<Vec<_>>();
    let values = |index: usize| {
        let mut values = candidates
            .iter()
            .map(|candidate| match index {
                0 => candidate.0,
                1 => candidate.1,
                2 => candidate.2,
                3 => candidate.3,
                _ => unreachable!("Krea sweep has exactly four axes"),
            })
            .collect::<Vec<_>>();
        values.sort_unstable();
        values.dedup();
        values
    };
    Ok(json!({
        "axes": [
            { "parameter": "decodeTileEdge", "testedValues": values(0) },
            { "parameter": "decodeOverlap", "testedValues": values(1) },
            { "parameter": "attentionChunkSize", "testedValues": values(2) },
            { "parameter": "transformerWindowSize", "testedValues": values(3) }
        ],
        "cases": rows,
        "rangeVerified": false,
    }))
}

fn complete_sweep(request: &Value, parameters: &Map<String, Value>) -> Result<Value, String> {
    let mut result = sweep(request, parameters, "passed")?;
    // Every v1 record executes the only published tuple. Marking that singleton range verified
    // certifies exactly the selected parameters without claiming the unexecuted v2 candidates.
    result["rangeVerified"] = json!(true);
    Ok(result)
}

fn artifact(repository: &str, revision: &str, tier: &str) -> Value {
    json!({
        "repository": repository,
        "resolvedRevision": revision,
        "variant": tier,
    })
}

fn loadability_fingerprint(repository: &str, revision: &str, tier: &str) -> String {
    format!("{repository}@{revision}:{tier}")
}

#[allow(clippy::too_many_arguments)]
fn execute_lifecycle_request(
    generator: &dyn runtime_cuda::gen_core::Generator,
    context: &MemoryRunContext,
    edge: u32,
    overlap: u32,
    attention: u32,
    window: u32,
    fault_phase: Option<MemoryPhase>,
    cancel_phase: Option<MemoryPhase>,
) -> Result<Option<runtime_cuda::gen_core::Image>, String> {
    let mut scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin lifecycle Krea scope: {error}"))?
        .ok_or_else(|| "lifecycle Krea selection did not create a provider scope".to_owned())?;
    scope
        .configure_decode(edge, overlap, context.geometry)
        .map_err(|error| format!("configure lifecycle decode tuple: {error}"))?;
    scope
        .configure_attention(attention)
        .map_err(|error| format!("configure lifecycle attention tuple: {error}"))?;
    scope
        .materialize_transformer_window(0, window)
        .map_err(|error| format!("configure lifecycle transformer tuple: {error}"))?;

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width: context.geometry.width,
        height: context.geometry.height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply lifecycle request strategy: {error}"))?;
    let memory = generation
        .memory
        .as_mut()
        .ok_or_else(|| "optimized lifecycle request did not receive GenerationMemory".to_owned())?;
    if let Some(phase) = fault_phase {
        memory.authorize_calibration_fault(phase);
    }
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter lifecycle conditioning phase: {error}"))?;

    let cancel = generation.cancel.clone();
    let mut phase = MemoryPhase::Conditioning;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::TextEncoder)
            if cancel_phase == Some(MemoryPhase::Conditioning) =>
        {
            cancel.cancel();
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
            if cancel_phase == Some(MemoryPhase::Denoise) {
                cancel.cancel();
            }
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
        }
        Progress::Decoding if cancel_phase == Some(MemoryPhase::Decode) => {
            cancel.cancel();
        }
        _ => {}
    });

    match (fault_phase, cancel_phase, result) {
        (None, None, Ok(runtime_cuda::gen_core::GenerationOutput::Images(mut images)))
            if images.len() == 1 =>
        {
            scope
                .leave_phase(phase)
                .map_err(|error| format!("leave successful lifecycle phase: {error}"))?;
            scope
                .finish(MemoryRunOutcome::Complete)
                .map_err(|error| format!("finish successful lifecycle request: {error}"))?;
            Ok(Some(images.remove(0)))
        }
        (Some(expected), None, Err(error))
            if error.to_string().contains("injected memory-strategy calibration error")
                && error.to_string().contains(&format!("{expected:?}")) =>
        {
            scope
                .finish(MemoryRunOutcome::Error {
                    message: error.to_string(),
                })
                .map_err(|finish| format!("finish injected-error lifecycle request: {finish}"))?;
            Ok(None)
        }
        (None, Some(_), Err(runtime_cuda::gen_core::Error::Canceled)) => {
            scope
                .finish(MemoryRunOutcome::Canceled)
                .map_err(|error| format!("finish canceled lifecycle request: {error}"))?;
            Ok(None)
        }
        (expected_fault, expected_cancel, actual) => Err(format!(
            "lifecycle outcome mismatch: fault={expected_fault:?}, cancel={expected_cancel:?}, actual={}",
            match actual {
                Ok(_) => "success".to_owned(),
                Err(error) => format!("error: {error}"),
            }
        )),
    }
}

fn execute_parity_request(
    generator: &dyn runtime_cuda::gen_core::Generator,
    baseline_context: &MemoryRunContext,
    strategy: MemoryStrategy,
    parameters: MemoryStrategyParameters,
) -> Result<runtime_cuda::gen_core::Image, String> {
    let mut context = baseline_context.clone();
    context.selection.strategy = strategy;
    context.selection.parameters = parameters;
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin parity Krea scope for {strategy:?}: {error}"))?
        .ok_or_else(|| format!("parity Krea strategy {strategy:?} did not create a scope"))?;
    if strategy.is_optimized() {
        let edge = parameters
            .decode_tile_edge
            .ok_or_else(|| format!("parity {strategy:?} is missing decode_tile_edge"))?;
        let overlap = parameters
            .decode_overlap
            .ok_or_else(|| format!("parity {strategy:?} is missing decode_overlap"))?;
        let attention = parameters
            .attention_chunk_size
            .ok_or_else(|| format!("parity {strategy:?} is missing attention_chunk_size"))?;
        let window = parameters
            .transformer_window_size
            .ok_or_else(|| format!("parity {strategy:?} is missing transformer_window_size"))?;
        scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure parity decode tuple: {error}"))?;
        scope
            .configure_attention(attention)
            .map_err(|error| format!("configure parity attention tuple: {error}"))?;
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| format!("configure parity transformer tuple: {error}"))?;
    }

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width: context.geometry.width,
        height: context.geometry.height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply parity request strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter parity conditioning phase: {error}"))?;
    let mut phase = MemoryPhase::Conditioning;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
        }
        _ => {}
    });
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!(
                "parity Krea {strategy:?} generation failed: {message}"
            ));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave parity terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish parity Krea request: {error}"))?;
    match output {
        runtime_cuda::gen_core::GenerationOutput::Images(mut images) if images.len() == 1 => {
            Ok(images.remove(0))
        }
        runtime_cuda::gen_core::GenerationOutput::Images(images) => Err(format!(
            "parity Krea {strategy:?} run returned {} images, expected 1",
            images.len()
        )),
        _ => Err(format!(
            "parity Krea {strategy:?} run returned non-image output"
        )),
    }
}

fn pixel_error(
    reference: &runtime_cuda::gen_core::Image,
    candidate: &runtime_cuda::gen_core::Image,
) -> Result<(f64, f64), String> {
    if (reference.width, reference.height, reference.pixels.len())
        != (candidate.width, candidate.height, candidate.pixels.len())
    {
        return Err(format!(
            "parity image shape mismatch: reference={}x{}x{}, candidate={}x{}x{}",
            reference.width,
            reference.height,
            reference.pixels.len(),
            candidate.width,
            candidate.height,
            candidate.pixels.len()
        ));
    }
    if reference.pixels.is_empty() {
        return Err("parity image is empty".to_owned());
    }
    let mut maximum = 0.0_f64;
    let mut total = 0.0_f64;
    for (&left, &right) in reference.pixels.iter().zip(&candidate.pixels) {
        let error = f64::from(left.abs_diff(right)) / 255.0;
        maximum = maximum.max(error);
        total += error;
    }
    Ok((maximum, total / reference.pixels.len() as f64))
}

fn negative_mutation(image: &runtime_cuda::gen_core::Image) -> runtime_cuda::gen_core::Image {
    let mut mutated = image.clone();
    for channel in &mut mutated.pixels {
        *channel = channel.wrapping_add(64);
    }
    mutated
}

fn ensure_krea_quality(maximum: f64, mean: f64, label: &str) -> Result<(), String> {
    if maximum > KREA_CANDLE_MAX_THRESHOLD || mean > KREA_CANDLE_MEAN_THRESHOLD {
        return Err(format!(
            "{label} exceeded the approved Candle Krea parity envelope: max={maximum:.6}, mean={mean:.6}"
        ));
    }
    Ok(())
}

fn preflight_fragment(
    request: &Value,
    strategy: &Value,
    load_shape: LoadShape,
    blocker: String,
    measurement_name: &'static str,
    repository: &str,
    revision: &str,
) -> Result<Value, String> {
    let mut fragment = protocol::plain_gated_fragment(
        request,
        KREA_PLAIN_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            artifact: artifact(repository, revision, planned_tier(request)?),
            sweep: sweep(request, protocol::strategy_parameters(request)?, "failed")?,
            blocker: &blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({ "result": "not_run", "resolvedPathFingerprint": null }),
            diagnostics: protocol::diagnostics(
                "memory-candle-adapter",
                "gated_before_execution",
                [blocker.clone()],
                [(measurement_name, "count", 1)],
            ),
        },
    )?;
    fragment["strategy"] = strategy.clone();
    fragment["loadShape"] = json!(load_shape_key(load_shape));
    Ok(fragment)
}

/// Persisted spelling of `gen_core::LoadShape` for the schema-v4 receipt field.
///
/// Callers pass the shape the run actually executed under — in practice
/// `contract.calibration.load_shape` from the LOADED provider, never the plan's declared value and
/// never a literal. A receipt may only testify to its own run (sc-16482).
fn load_shape_key(load_shape: LoadShape) -> &'static str {
    match load_shape {
        LoadShape::EagerMaterialization => protocol::LOAD_SHAPE_EAGER,
        LoadShape::DeferredMaterialization => protocol::LOAD_SHAPE_DEFERRED,
    }
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

fn planned_memory_strategy(request: &Value) -> Result<MemoryStrategy, String> {
    match protocol::planned_rung(request)? {
        "resident" => Ok(MemoryStrategy::Resident),
        "staged_residency" => Ok(MemoryStrategy::StagedResidency),
        "bounded_decode" => Ok(MemoryStrategy::BoundedDecode),
        "bounded_attention" => Ok(MemoryStrategy::BoundedAttention),
        "bounded_transformer_residency" => Ok(MemoryStrategy::BoundedTransformerResidency),
        other => Err(format!("unsupported Candle fresh-reference rung {other:?}")),
    }
}

fn planned_provider(request: &Value) -> Result<&str, String> {
    protocol::planned(request)?
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())
}

fn planned_mode(request: &Value) -> Result<&str, String> {
    protocol::planned(request)?
        .pointer("/target/mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())
}

/// Whether this case is the `z_image_edit` route: the Turbo provider in `edit_image` mode. Every
/// other provider this adapter implements is measured text-to-image regardless of the mode
/// spelling, exactly as before; only the Turbo arm has a second mode it can actually execute.
fn is_z_image_edit(request: &Value) -> Result<bool, String> {
    Ok(planned_provider(request)? == Z_IMAGE_TURBO_ID && planned_mode(request)? == "edit_image")
}

fn plain_execution_path(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        "qwen_image" => Ok(QWEN_PLAIN_EXECUTION_PATH),
        "krea_2_turbo" => Ok(KREA_PLAIN_EXECUTION_PATH),
        "z_image_turbo" => Ok(if is_z_image_edit(request)? {
            Z_IMAGE_TURBO_EDIT_EXECUTION_PATH
        } else {
            Z_IMAGE_TURBO_PLAIN_EXECUTION_PATH
        }),
        // sc-22724: the undistilled base is its own registry id with its own artifact family.
        "z_image" => Ok(Z_IMAGE_PLAIN_EXECUTION_PATH),
        // sc-22727: three catalog models over two registry ids; the member decides the path.
        FLUX2_DEV_ID | FLUX2_KLEIN_ID => Ok(flux2_arm(request)?
            .expect("a FLUX.2 provider always resolves a member or errors")
            .execution_path),
        // sc-22726: the two FLUX.1 base providers ride the same five-rung reference path.
        "flux1_dev" => Ok(FLUX1_DEV_PLAIN_EXECUTION_PATH),
        "flux1_schnell" => Ok(FLUX1_SCHNELL_PLAIN_EXECUTION_PATH),
        // sc-22726: PuLID-FLUX is a bespoke route with its own arm; it is named here so the shared
        // refusal cannot claim this adapter does not implement it.
        "pulid_flux" => Ok(PULID_FLUX_EXECUTION_PATH),
        // sc-22733: the six Mage-Flow routes, each its own registered engine provider. Spelled out
        // one bare literal per arm, like the FLUX.1 rows above: `adapterCapturableProviders`
        // INTERSECTS every dispatch gate carrying the refusal phrase and refuses any arm pattern
        // that is not a literal or a single `&str` const, so a `provider if …` guard would make the
        // derived report throw instead of listing these six as capturable.
        "mage_flow" => mage_execution_path("mage_flow"),
        "mage_flow_base" => mage_execution_path("mage_flow_base"),
        "mage_flow_turbo" => mage_execution_path("mage_flow_turbo"),
        "mage_flow_edit" => mage_execution_path("mage_flow_edit"),
        "mage_flow_edit_base" => mage_execution_path("mage_flow_edit_base"),
        "mage_flow_edit_turbo" => mage_execution_path("mage_flow_edit_turbo"),
        // sc-22730: the three SD3.5 base providers ride the same five-rung reference path.
        SD3_5_LARGE_ID => Ok(SD3_5_LARGE_PLAIN_EXECUTION_PATH),
        SD3_5_LARGE_TURBO_ID => Ok(SD3_5_LARGE_TURBO_PLAIN_EXECUTION_PATH),
        SD3_5_MEDIUM_ID => Ok(SD3_5_MEDIUM_PLAIN_EXECUTION_PATH),
        // sc-22731: the SANA and Chroma1 families ride the same five-rung reference path.
        "sana_1600m" => Ok(SANA_PLAIN_EXECUTION_PATH),
        "sana_sprint_1600m" => Ok(SANA_SPRINT_PLAIN_EXECUTION_PATH),
        "chroma1_hd" => Ok(CHROMA1_HD_PLAIN_EXECUTION_PATH),
        "chroma1_base" => Ok(CHROMA1_BASE_PLAIN_EXECUTION_PATH),
        "chroma1_flash" => Ok(CHROMA1_FLASH_PLAIN_EXECUTION_PATH),
        // sc-22737: the STILL Bernini entry. `bernini` and `bernini_image` are two catalog entries
        // on ONE engine provider id, and only the still one reaches this far — the video one is
        // dispatched above `validate_still_geometry` by `sc22737_video_arm`. Resolved by name
        // rather than assumed, so a video plan row that somehow reached the still gate is refused
        // naming the member instead of silently borrowing the still route's execution path.
        BERNINI_CANDLE_ID => {
            bernini_candle_still_member(request).map(|_| BERNINI_CANDLE_IMAGE_EXECUTION_PATH)
        }
        provider => Err(format!(
            "Candle five-rung calibration does not implement provider {provider:?}"
        )),
    }
}

/// Refuse any Bernini catalog entry but the STILL one at the still gate (sc-22737).
///
/// `bernini` (video) and `bernini_image` (still) share the engine provider id `bernini`, so the
/// provider alone cannot say which member a plan row means. The video member is dispatched above
/// the still gate and never arrives here; if it does, the plan is asking for a video record on the
/// image path, and the two are not interchangeable — a still record carries `frames: 1` and the
/// `text_to_image` mode key, which is a different admission key from the one the video route asks
/// under. Refused BY NAME rather than defaulted.
fn bernini_candle_still_member(request: &Value) -> Result<&'static str, String> {
    let model_id = protocol::planned(request)?
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != BERNINI_CANDLE_IMAGE_MODEL_ID {
        return Err(format!(
            "the Candle Bernini still path serves catalog entry \
             {BERNINI_CANDLE_IMAGE_MODEL_ID:?}, got modelId {model_id:?}; \
             {BERNINI_CANDLE_VIDEO_MODEL_ID:?} is the video entry and is measured by the sc-22737 \
             video arm instead"
        ));
    }
    Ok(BERNINI_CANDLE_IMAGE_MODEL_ID)
}

/// The calibration label this Candle target refuses a non-still geometry under (sc-18808).
///
/// BOTH Candle arms are image arms, and both carried the same latent defect the MLX image arms did:
/// they read only `width`/`height` through [`protocol::target_geometry`] and then wrote `frames: 1`
/// straight into `MemoryGeometry`. A plan row declaring `frames: 2` would therefore have rendered ONE
/// frame and emitted a well-formed record whose geometry envelope claimed a single frame it was never
/// asked for — the exact defect class this apparatus exists to make impossible.
///
/// No Candle plan row declares a non-unit frames axis today (all 154 rows are `frames: 1`), so this
/// is not a live exposure. It is added anyway because epic 18803 IS the video lane and
/// `ltx_2_3_distilled` is a Candle engine id: the shape becomes reachable, and a refusal is the only
/// thing that keeps the record honest when it does. Mirrors [`plain_execution_path`] so a provider
/// this adapter does not implement is rejected by the same sentence in both.
fn still_calibration_label(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        QWEN_ID => Ok(QWEN_STILL_CALIBRATION),
        KREA_ID => Ok(KREA_STILL_CALIBRATION),
        Z_IMAGE_TURBO_ID => Ok(if is_z_image_edit(request)? {
            Z_IMAGE_TURBO_EDIT_STILL_CALIBRATION
        } else {
            Z_IMAGE_TURBO_STILL_CALIBRATION
        }),
        Z_IMAGE_ID => Ok(Z_IMAGE_STILL_CALIBRATION),
        FLUX2_DEV_ID | FLUX2_KLEIN_ID => Ok(flux2_arm(request)?
            .expect("a FLUX.2 provider always resolves a member or errors")
            .still_calibration),
        FLUX1_DEV_ID => Ok(FLUX1_DEV_STILL_CALIBRATION),
        FLUX1_SCHNELL_ID => Ok(FLUX1_SCHNELL_STILL_CALIBRATION),
        PULID_FLUX_ID => Ok(PULID_FLUX_STILL_CALIBRATION),
        // One bare literal per arm — see the note in `plain_execution_path`.
        "mage_flow" => mage_still_calibration("mage_flow"),
        "mage_flow_base" => mage_still_calibration("mage_flow_base"),
        "mage_flow_turbo" => mage_still_calibration("mage_flow_turbo"),
        "mage_flow_edit" => mage_still_calibration("mage_flow_edit"),
        "mage_flow_edit_base" => mage_still_calibration("mage_flow_edit_base"),
        "mage_flow_edit_turbo" => mage_still_calibration("mage_flow_edit_turbo"),
        SD3_5_LARGE_ID => Ok(SD3_5_LARGE_STILL_CALIBRATION),
        SD3_5_LARGE_TURBO_ID => Ok(SD3_5_LARGE_TURBO_STILL_CALIBRATION),
        SD3_5_MEDIUM_ID => Ok(SD3_5_MEDIUM_STILL_CALIBRATION),
        SANA_ID => Ok(SANA_STILL_CALIBRATION),
        SANA_SPRINT_ID => Ok(SANA_SPRINT_STILL_CALIBRATION),
        CHROMA1_HD_ID => Ok(CHROMA1_HD_STILL_CALIBRATION),
        CHROMA1_BASE_ID => Ok(CHROMA1_BASE_STILL_CALIBRATION),
        CHROMA1_FLASH_ID => Ok(CHROMA1_FLASH_STILL_CALIBRATION),
        // sc-22737: the still Bernini entry only — see `bernini_candle_still_member`.
        BERNINI_CANDLE_ID => {
            bernini_candle_still_member(request).map(|_| BERNINI_CANDLE_STILL_CALIBRATION)
        }
        provider => Err(format!(
            "Candle five-rung calibration does not implement provider {provider:?}"
        )),
    }
}

/// The numeric tier this case plans to measure, read from the plan rather than assumed.
///
/// sc-17097: this used to be hardcoded `q4`, which silently capped the Candle lane at one tier — the
/// `krea_2_turbo` turbo fit ships `q4`, `q8` and `bf16` phase curves, so two thirds of it could not be
/// re-measured at all. The MLX adapter has always derived its tier from `/target/tier`; this mirrors
/// that, and [`planned_tier_variant`] keeps the on-disk artifact bound to the same token so a q8 plan
/// can never be satisfied by q4 weights.
fn planned_tier(request: &Value) -> Result<&str, String> {
    match protocol::planned(request)?
        .pointer("/target/tier")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.tier must be a string".to_owned())?
    {
        tier @ ("bf16" | "q4" | "q8") => Ok(tier),
        tier => Err(format!("unsupported Candle numeric tier {tier:?}")),
    }
}

/// The fixture must name the tier and geometry it measured, so a bf16 record can never be emitted
/// against a q4 capture that merely reused the fixture string.
///
/// Scoped to `krea_2_turbo` and the FLUX.1 family. Krea's plan spans several (tier, geometry) legs
/// through one adapter path — six of them, which is exactly how a mislabelled capture would arise.
/// The FLUX.1 members (sc-22726) get the stricter member/tier/edge/seed/step binding the MLX arm
/// applies ([`validate_flux_one_fixture`]), so a fixture can never name a seed the capture did not
/// render at. The Qwen legs declare a single tier and geometry each and their fixture names
/// (`qwen-image-candle-q4-seed15817-step2`) predate this convention: applying the geometry token
/// requirement to them would reject five plan rows that measure correctly today. Widen this when
/// those fixtures are renamed, not before.
/// Whether this Candle route's weights root descends into a `<tier>` sub-directory.
///
/// True for every packed SceneWorks turnkey. False for the two SANA routes alone (sc-22731): the
/// worker resolves the UPSTREAM dense diffusers snapshot root for them, which has no tier
/// component, and `candle-gen-sana` requires exactly that root.
fn five_rung_root_is_tiered(provider_id: &str) -> bool {
    !matches!(provider_id, SANA_ID | SANA_SPRINT_ID)
}

/// Refuse a planned tier this LANE cannot open, by name, before any environment is read.
///
/// Only SANA constrains this today: `candle-gen-sana`'s `validate_load_spec` refuses any
/// `LoadSpec::quantize`, the worker pins the candle SANA tier to `bf16`, and no packed SANA
/// artifact ships off-Mac. A `sana_1600m:q4:candle` plan row would otherwise reach the engine and
/// come back as a quantization complaint, which reads as a spec bug rather than as a cell that
/// does not exist.
///
/// HAND-WRITTEN, and deliberately so: it is a REFUSAL that must fire before any environment read,
/// on a host with no manifest loaded, so it cannot be derived from the manifest at the point of
/// use. It is therefore a second spelling of a fact the manifest also states, and the binding that
/// keeps the two honest is on the JS side —
/// `scripts/generate-memory-matrix.test.mjs`'s "every published lane tier is one that lane's route
/// rules admit" and "no lane advertises a tier whose only downloads that lane's host would never
/// fetch". Between them, the manifest's `sana_*` off-Mac download (the untiered
/// `Efficient-Large-Model/Sana_1600M_1024px_diffusers` snapshot, `platforms: ["windows","linux"]`)
/// and `memory_route_registry.rs`'s `BF16_ONLY` candle SANA rows must keep agreeing with the
/// bf16-only rule below; publishing a packed candle SANA tier reds there, and the plan-row set
/// derived from this function in `sana_chroma_candle_tests` reds here.
fn validate_five_rung_lane_tier(provider_id: &str, tier: &str) -> Result<(), String> {
    if matches!(provider_id, SANA_ID | SANA_SPRINT_ID) && tier != "bf16" {
        return Err(format!(
            "the Candle {provider_id} route loads the upstream dense snapshot only; there is no \
             {tier} artifact on this lane and candle-gen-sana refuses a LoadSpec quant"
        ));
    }
    Ok(())
}

/// The production calibration identity the loaded Candle generator publishes for one
/// `(provider, tier)` cell of the two families sc-22731 armed.
///
/// `candle-gen-sana` mints `sana-candle-dense-<route>-full-ladder-v1` per route (bf16 is its only
/// tier). `candle-gen-chroma` mints `<route>-<tier>-cuda-<ladder revision>` per (route, tier) from
/// the LOAD RECEIPT's tier — inference PR 951; before it, `build_contract` hard-coded
/// `calibration: None` and no Chroma anchor could be recorded on this lane at all.
///
/// `None` for every other provider: their arms predate this pre-load binding and are still checked
/// against the loaded contract in `run_five_rung_reference_loaded`.
fn five_rung_calibration_fingerprint(provider_id: &str, tier: &str) -> Option<String> {
    match provider_id {
        SANA_ID => Some("sana-candle-dense-base-full-ladder-v1".to_owned()),
        SANA_SPRINT_ID => Some("sana-candle-dense-sprint-full-ladder-v1".to_owned()),
        CHROMA1_HD_ID | CHROMA1_BASE_ID | CHROMA1_FLASH_ID => Some(format!(
            "{}-{tier}-cuda-{CHROMA1_CANDLE_LADDER_REVISION}",
            provider_id.replace('_', "-")
        )),
        // sc-22737. `candle-gen-bernini` mints `bernini-{route}-{tier}-candle-dual-expert-ladder-v1`
        // per (provider, ARTIFACT-PROVEN tier) — the packing its `production_assets` has already
        // reconciled against the transformer's own `quantize_config.json` and tensor geometry, so a
        // dense root asked for as q4 never reaches the identity at all. Every SceneWorks load
        // resolves the `image` route (`bernini_renderer` is registered by the engine but named by
        // no load path in this repository), so the route token is constant and the tier is the only
        // axis. The capture still refuses a loaded contract whose identity differs from this, so a
        // drift is caught at the pin bump rather than inside a record.
        BERNINI_CANDLE_ID => Some(format!("bernini-image-{tier}-candle-dual-expert-ladder-v1")),
        _ => None,
    }
}

/// `candle-gen-chroma`'s `CANDLE_LADDER_REVISION`: the shape every Chroma1 Candle identity names.
/// Bumped upstream when the request-scoped Resident/Staged surface changes, which is what makes
/// every Chroma anchor recaptured rather than silently re-bound.
const CHROMA1_CANDLE_LADDER_REVISION: &str = "request-scoped-staged-residency-v1";

/// The five-rung fixture slug of each sc-22731 route: `fresh-five-rung-<slug>-<tier>-<edge>-seed…`.
fn five_rung_family_slug(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        SANA_ID => Some("sana-1600m"),
        SANA_SPRINT_ID => Some("sana-sprint"),
        CHROMA1_HD_ID => Some("chroma1-hd"),
        CHROMA1_BASE_ID => Some("chroma1-base"),
        CHROMA1_FLASH_ID => Some("chroma1-flash"),
        // sc-22737. `bernini-image`, not `bernini`: the slug names the CATALOG ENTRY this fixture
        // measures, and the bare `bernini` fixtures belong to the video member's own arm.
        BERNINI_CANDLE_ID => Some("bernini-image"),
        _ => None,
    }
}

fn validate_fixture_binds_tier_and_geometry(request: &Value) -> Result<(), String> {
    let provider = planned_provider(request)?;
    // sc-22733: the Mage members get the same member/tier/edge/seed/step binding.
    if let Some(arm) = mage_arm(request)? {
        return validate_mage_fixture(request, arm, planned_tier(request)?);
    }
    if matches!(provider, FLUX1_DEV_ID | FLUX1_SCHNELL_ID | PULID_FLUX_ID) {
        return validate_flux_one_fixture(request, provider, planned_tier(request)?);
    }
    // sc-22731: the SANA and Chroma1 fixtures get the same member/tier/edge/seed/step binding, so a
    // bf16 record can never be emitted against a q4 capture that merely reused the fixture string.
    if let Some(slug) = five_rung_family_slug(provider) {
        return validate_five_rung_family_fixture(request, slug, planned_tier(request)?);
    }
    // sc-22730: the SD3.5 members get the same member/tier/edge/seed/step binding, so a Medium q8
    // record can never be attributed to a Large bf16 capture that merely reused the string.
    if SD3_5_FIXTURE_SLUGS.iter().any(|(id, _)| *id == provider) {
        return validate_sd35_fixture(request, provider, planned_tier(request)?);
    }
    if provider != KREA_ID {
        return Ok(());
    }
    let planned = protocol::planned(request)?;
    let fixture = planned
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let tier = planned_tier(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    if width != height {
        return Err(format!(
            "Candle Krea calibration fixtures are square; planned geometry is {width}x{height}"
        ));
    }
    for token in [format!("-{tier}-"), format!("-{width}-")] {
        if !fixture.contains(&token) {
            return Err(format!(
                "planned.fixture {fixture:?} must contain {token:?} so the capture cannot be \
                 attributed to another tier or geometry"
            ));
        }
    }
    Ok(())
}

/// The SD3.5 fixture binds the member, the tier, the geometry edge, the seed and the step count —
/// the MLX arm's `planned_sd3_seed`, on this lane's spellings.
///
/// All three members ride the shared five-rung reference path, so they render at [`FIVE_RUNG_SEED`]
/// and their fixtures must say so: `fresh-five-rung-sd3-5-<route>-<tier>-<edge>-seed16402-step2`.
/// A fixture naming the MLX arm's own seed is refused — the record's fixture is the one claim about
/// the render that nothing downstream can re-derive.
fn validate_sd35_fixture(request: &Value, provider: &str, tier: &str) -> Result<(), String> {
    let slug = SD3_5_FIXTURE_SLUGS
        .iter()
        .find_map(|(id, slug)| (*id == provider).then_some(*slug))
        .ok_or_else(|| {
            format!("the Candle SD3.5 fixture binding does not implement provider {provider:?}")
        })?;
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{FIVE_RUNG_FIXTURE_PREFIX}{slug}-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (planned_seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let planned_seed = planned_seed
        .parse::<u64>()
        .map_err(|error| format!("parse SD3.5 fixture seed {planned_seed:?}: {error}"))?;
    if planned_seed != FIVE_RUNG_SEED {
        return Err(format!(
            "planned.fixture seed {planned_seed} does not match the seed {provider} renders at \
             on this lane ({FIVE_RUNG_SEED})"
        ));
    }
    if steps != "2" {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// The FLUX.1 fixture binds the member, the tier, the geometry edge, the seed and the step count
/// — the MLX arm's `validate_flux_one_fixture`, on this lane's spellings. The base providers ride
/// the five-rung reference path (`fresh-five-rung-flux1-<route>-…-seed16402-step2`, at
/// [`FIVE_RUNG_SEED`]); PuLID is bespoke (`pulid-flux-candle-…-seed22726-step2`, at
/// [`FLUX1_SEED`]). A fixture naming the other seed is refused: the record's fixture is the one
/// claim about the render that nothing downstream can re-derive.
fn validate_flux_one_fixture(request: &Value, provider: &str, tier: &str) -> Result<(), String> {
    let (prefix, seed) = match provider {
        FLUX1_DEV_ID => ("fresh-five-rung-flux1-dev", FIVE_RUNG_SEED),
        FLUX1_SCHNELL_ID => ("fresh-five-rung-flux1-schnell", FIVE_RUNG_SEED),
        PULID_FLUX_ID => ("pulid-flux-candle", FLUX1_SEED),
        other => {
            return Err(format!(
                "the Candle FLUX.1 fixture binding does not implement provider {other:?}"
            ))
        }
    };
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{prefix}-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (planned_seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let planned_seed = planned_seed
        .parse::<u64>()
        .map_err(|error| format!("parse FLUX.1 fixture seed {planned_seed:?}: {error}"))?;
    if planned_seed != seed {
        return Err(format!(
            "planned.fixture seed {planned_seed} does not match the seed {provider} renders at \
             ({seed})"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse FLUX.1 fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// The sc-22731 five-rung fixture binding: `fresh-five-rung-<slug>-<tier>-<edge>-seed16402-step2`.
///
/// Same claim `validate_flux_one_fixture` makes on the FLUX.1 base routes, on this family's
/// spellings — the record's fixture is the one claim about the render that nothing downstream can
/// re-derive, so it must name the route, the tier, the geometry edge, the seed and the step count.
fn validate_five_rung_family_fixture(
    request: &Value,
    slug: &str,
    tier: &str,
) -> Result<(), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{FIVE_RUNG_FIXTURE_PREFIX}{slug}-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse {slug} fixture seed {seed:?}: {error}"))?;
    if seed != FIVE_RUNG_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the five-rung reference seed \
             {FIVE_RUNG_SEED}"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse {slug} fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// The tier a FLUX.1 base snapshot actually declares (`transformer/config.json`), read through
/// `candle-gen-flux`'s own resolver, compared against the PLANNED tier. The root suffix proved
/// the plan and the export agree on the directory NAME; this proves the weights inside agree
/// too — the worker's doctrine (`image_jobs/pulid_candle.rs`: "directory basenames are never
/// tier evidence; the packed transformer config is authoritative"), which the PuLID arm already
/// applied and the base arm did not.
fn validate_flux_one_snapshot_tier(
    spec: &LoadSpec,
    provider_id: &str,
    tier: &str,
) -> Result<(), String> {
    let resolved =
        runtime_cuda::providers::flux::memory_strategy::resolved_numeric_tier(spec, provider_id)
            .map_err(|error| format!("resolve {provider_id} numeric tier: {error}"))?;
    let expected = numeric_tier(tier)?;
    if (resolved.precision, resolved.quant) != (expected.precision, expected.quant) {
        return Err(format!(
            "planned tier {tier} does not match the tier the {provider_id} snapshot declares \
             (precision={:?}, quant={:?})",
            resolved.precision, resolved.quant
        ));
    }
    Ok(())
}

fn numeric_tier(tier: &str) -> Result<MemoryNumericTier, String> {
    // Matches the worker's `tier_to_quant`: bf16 is the dense base, q4/q8 are the packed tiers.
    let quant = match tier {
        "bf16" => None,
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        other => return Err(format!("unsupported Candle numeric tier {other:?}")),
    };
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

fn planned_selection(request: &Value) -> Result<MemorySelection, String> {
    let strategy = planned_memory_strategy(request)?;
    let transformer_window_size = protocol::optional_parameter(request, "transformerWindowSize")?;
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
        tier: numeric_tier(planned_tier(request)?)?,
    })
}

fn reference_phase(phase: MemoryPhase) -> protocol::ReferencePhase {
    match phase {
        MemoryPhase::Conditioning => protocol::ReferencePhase::Conditioning,
        MemoryPhase::Denoise => protocol::ReferencePhase::Denoise,
        MemoryPhase::Decode => protocol::ReferencePhase::Decode,
    }
}

fn memory_phase(phase: protocol::ReferencePhase) -> MemoryPhase {
    match phase {
        protocol::ReferencePhase::Conditioning => MemoryPhase::Conditioning,
        protocol::ReferencePhase::Denoise => MemoryPhase::Denoise,
        protocol::ReferencePhase::Decode => MemoryPhase::Decode,
    }
}

fn measured_strategy(
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

/// Everything one five-rung capture needs after the artifact identity is validated and the real
/// generator is resident: `(provider id, plain execution path, repository, resolved revision,
/// generator, VRAM probe already holding the load sample)`.
type LoadedFiveRungGenerator = (
    &'static str,
    &'static str,
    String,
    String,
    Box<dyn runtime_cuda::gen_core::Generator>,
    VramProbe,
);

fn mage_registry_bypass(provider: &str) -> String {
    format!(
        "{provider} binds two artifact triples and stages two components; it is served by \
         load_mage_generator and must not reach the single-root five-rung loader"
    )
}

/// The Mage half of [`load_five_rung_generator`]. Split out because a Mage load binds TWO artifact
/// triples and stages two components, which the single-root env tuple below cannot express.
fn load_mage_generator(request: &Value, arm: MageArm) -> Result<LoadedFiveRungGenerator, String> {
    validate_mage_mode(request, arm)?;
    let tier = planned_tier(request)?;
    validate_fixture_binds_tier_and_geometry(request)?;
    validate_mage_plan_identity(request, arm, tier)?;
    let load_shape = mage_planned_load_shape(request)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    protocol::validate_artifact_identity(&repository, &revision, arm.repository)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
    // q8 or bf16 plan and quietly re-label another tier's peaks.
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        arm.repository,
    )?;
    let components_repository =
        protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REPOSITORY")?;
    let components_revision = protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REVISION")?;
    protocol::validate_artifact_identity(
        &components_repository,
        &components_revision,
        protocol::MAGE_COMPONENTS_REPOSITORY,
    )?;
    let components_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_MAGE_FLOW_COMPONENTS_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_MAGE_FLOW_COMPONENTS_ROOT: {error}"))?;
    protocol::validate_huggingface_revision_root(
        &components_root,
        &components_repository,
        &components_revision,
        protocol::MAGE_COMPONENTS_REPOSITORY,
    )?;
    let spec = mage_load_spec(tier, load_shape, root, &components_root)?;
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(arm.provider, &spec)
        .map_err(|error| format!("load real {} {tier} generator: {error}", arm.provider))?;
    vram.end_load(load_sample);
    Ok((
        arm.provider,
        arm.execution_path,
        repository,
        revision,
        generator,
        vram,
    ))
}

fn load_five_rung_generator(request: &Value) -> Result<LoadedFiveRungGenerator, String> {
    // sc-22733: Mage binds two artifact triples and stages two components; it has its own loader.
    if let Some(arm) = mage_arm(request)? {
        return load_mage_generator(request, arm);
    }
    let (provider_id, execution_path, repository_env, revision_env, root_env, expected_repository) =
        match planned_provider(request)? {
            "qwen_image" => (
                QWEN_ID,
                QWEN_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_QWEN_IMAGE_REPOSITORY",
                "SCENEWORKS_QWEN_IMAGE_REVISION",
                "SCENEWORKS_QWEN_IMAGE_ROOT",
                protocol::QWEN_REPOSITORY,
            ),
            "krea_2_turbo" => (
                KREA_ID,
                KREA_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_KREA_REPOSITORY",
                "SCENEWORKS_KREA_REVISION",
                "SCENEWORKS_KREA_ROOT",
                protocol::KREA_REPOSITORY,
            ),
            // sc-15859. The artifact family is `SceneWorks/z-image-turbo-mlx` (`Z_IMAGE_REPOSITORY`),
            // the same per-tier `q4/ q8/ bf16/` re-host the MLX arm measures, so the env family is
            // `SCENEWORKS_Z_IMAGE_*` on both adapters (docs/calibration-runbook.md, "Adapter
            // environment").
            "z_image_turbo" => (
                Z_IMAGE_TURBO_ID,
                // The edit route (`z_image_edit`) loads the same Turbo provider from the same
                // artifact; only the generation request and the admitted mode differ.
                plain_execution_path(request)?,
                "SCENEWORKS_Z_IMAGE_REPOSITORY",
                "SCENEWORKS_Z_IMAGE_REVISION",
                "SCENEWORKS_Z_IMAGE_ROOT",
                protocol::Z_IMAGE_REPOSITORY,
            ),
            // sc-22724. The base model's own rehost, bound through its own env family so a base
            // plan can never be satisfied by Turbo weights.
            "z_image" => (
                Z_IMAGE_ID,
                Z_IMAGE_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_Z_IMAGE_BASE_REPOSITORY",
                "SCENEWORKS_Z_IMAGE_BASE_REVISION",
                "SCENEWORKS_Z_IMAGE_BASE_ROOT",
                protocol::Z_IMAGE_BASE_REPOSITORY,
            ),
            // sc-22727. Each FLUX.2 catalog model binds its OWN artifact family, so a KV plan can
            // never be satisfied by the base klein rehost even though both load through
            // `flux2_klein_9b`.
            FLUX2_DEV_ID | FLUX2_KLEIN_ID => {
                let arm = flux2_arm(request)?
                    .expect("a FLUX.2 provider always resolves a member or errors");
                (
                    arm.provider,
                    arm.execution_path,
                    arm.repository_env,
                    arm.revision_env,
                    arm.root_env,
                    arm.expected_repository,
                )
            }
            // sc-22726. PuLID is a BESPOKE route: `candle-gen-pulid` registers no `Generator`, so
            // there is nothing for `catalog.media().load` to return and reaching here at all means
            // the dispatch in `run` was bypassed. Named rather than left to the catch-all so the
            // refusal says which arm owns it — and so the derived capturability report
            // (`stale-lane-report.mjs#adapterCapturableProviders`, which intersects every dispatch
            // gate carrying the refusal phrase) does not read this lane as having no arm.
            // Expression-bodied ON PURPOSE: a block-bodied arm carries no trailing comma after
            // rustfmt, and the report's arm parser splits on depth-0 commas — a braced arm here
            // silently swallowed the NEXT arm's pattern and dropped `flux1_dev` from the derived
            // capturable set.
            "pulid_flux" => return Err(
                "pulid_flux is a bespoke Candle route and must not reach the provider registry; \
                 it is served by run_pulid_flux_capture"
                    .to_owned(),
            ),
            // Each FLUX.1 base provider binds its OWN tiered rehost, so a schnell plan can never be
            // satisfied by dev weights.
            "flux1_dev" => (
                FLUX1_DEV_ID,
                FLUX1_DEV_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_FLUX1_DEV_REPOSITORY",
                "SCENEWORKS_FLUX1_DEV_REVISION",
                "SCENEWORKS_FLUX1_DEV_ROOT",
                protocol::FLUX1_DEV_REPOSITORY,
            ),
            "flux1_schnell" => (
                FLUX1_SCHNELL_ID,
                FLUX1_SCHNELL_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_FLUX1_SCHNELL_REPOSITORY",
                "SCENEWORKS_FLUX1_SCHNELL_REVISION",
                "SCENEWORKS_FLUX1_SCHNELL_ROOT",
                protocol::FLUX1_SCHNELL_REPOSITORY,
            ),
            // sc-22733. Each Mage variant binds TWO artifact triples (its own tiered rehost plus
            // the shared text-encoder/VAE components snapshot) and stages two components, which
            // the single-root tuple above cannot express — so the early return at the top of this
            // function serves them and reaching here means that return was bypassed. Named rather
            // than left to the catch-all for the same two reasons `pulid_flux` is: the refusal must
            // say which loader owns the route, and `adapterCapturableProviders` INTERSECTS every
            // dispatch gate carrying the refusal phrase, so an absent arm here would erase all six
            // from the derived capturable set. Expression-bodied ON PURPOSE (see `pulid_flux`).
            "mage_flow" => return Err(mage_registry_bypass("mage_flow")),
            "mage_flow_base" => return Err(mage_registry_bypass("mage_flow_base")),
            "mage_flow_turbo" => return Err(mage_registry_bypass("mage_flow_turbo")),
            "mage_flow_edit" => return Err(mage_registry_bypass("mage_flow_edit")),
            "mage_flow_edit_base" => return Err(mage_registry_bypass("mage_flow_edit_base")),
            "mage_flow_edit_turbo" => return Err(mage_registry_bypass("mage_flow_edit_turbo")),
            // sc-22730. Each SD3.5 member binds its OWN tiered rehost, so a Medium plan can never
            // be satisfied by Large weights. The rehost is the SAME artifact family both lanes
            // load (`SceneWorks/sd3.5-<route>-mlx`, per-tier `q4/ q8/ bf16/` subdirs), so the env
            // family is `SCENEWORKS_SD3_5_<ROUTE>_*` on both adapters.
            SD3_5_LARGE_ID => (
                SD3_5_LARGE_ID,
                SD3_5_LARGE_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SD3_5_LARGE_REPOSITORY",
                "SCENEWORKS_SD3_5_LARGE_REVISION",
                "SCENEWORKS_SD3_5_LARGE_ROOT",
                protocol::SD3_5_LARGE_REPOSITORY,
            ),
            SD3_5_LARGE_TURBO_ID => (
                SD3_5_LARGE_TURBO_ID,
                SD3_5_LARGE_TURBO_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SD3_5_LARGE_TURBO_REPOSITORY",
                "SCENEWORKS_SD3_5_LARGE_TURBO_REVISION",
                "SCENEWORKS_SD3_5_LARGE_TURBO_ROOT",
                protocol::SD3_5_LARGE_TURBO_REPOSITORY,
            ),
            SD3_5_MEDIUM_ID => (
                SD3_5_MEDIUM_ID,
                SD3_5_MEDIUM_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SD3_5_MEDIUM_REPOSITORY",
                "SCENEWORKS_SD3_5_MEDIUM_REVISION",
                "SCENEWORKS_SD3_5_MEDIUM_ROOT",
                protocol::SD3_5_MEDIUM_REPOSITORY,
            ),
            // sc-22731. The SANA routes bind the UPSTREAM DENSE diffusers snapshot, not the
            // SceneWorks MLX turnkey: `resolve_weights_dir` returns
            // `huggingface_pinned_snapshot_dir(SANA_CANDLE_DIFFUSERS_REPO, …)` for this lane, whose
            // root has no tier component at all. Their own env families keep a base plan from
            // being satisfied by Sprint weights.
            "sana_1600m" => (
                SANA_ID,
                SANA_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SANA_DENSE_REPOSITORY",
                "SCENEWORKS_SANA_DENSE_REVISION",
                "SCENEWORKS_SANA_DENSE_ROOT",
                protocol::SANA_DENSE_REPOSITORY,
            ),
            "sana_sprint_1600m" => (
                SANA_SPRINT_ID,
                SANA_SPRINT_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_SANA_SPRINT_DENSE_REPOSITORY",
                "SCENEWORKS_SANA_SPRINT_DENSE_REVISION",
                "SCENEWORKS_SANA_SPRINT_DENSE_ROOT",
                protocol::SANA_SPRINT_DENSE_REPOSITORY,
            ),
            // sc-22731. Each Chroma1 route binds its OWN rehost — `candle-gen-chroma`'s `ROUTES`
            // table pins one repository and revision per provider and refuses a root bound to
            // another, so an HD plan can never be satisfied by Flash weights.
            "chroma1_hd" => (
                CHROMA1_HD_ID,
                CHROMA1_HD_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_CHROMA1_HD_REPOSITORY",
                "SCENEWORKS_CHROMA1_HD_REVISION",
                "SCENEWORKS_CHROMA1_HD_ROOT",
                protocol::CHROMA1_HD_REPOSITORY,
            ),
            "chroma1_base" => (
                CHROMA1_BASE_ID,
                CHROMA1_BASE_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_CHROMA1_BASE_REPOSITORY",
                "SCENEWORKS_CHROMA1_BASE_REVISION",
                "SCENEWORKS_CHROMA1_BASE_ROOT",
                protocol::CHROMA1_BASE_REPOSITORY,
            ),
            "chroma1_flash" => (
                CHROMA1_FLASH_ID,
                CHROMA1_FLASH_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_CHROMA1_FLASH_REPOSITORY",
                "SCENEWORKS_CHROMA1_FLASH_REVISION",
                "SCENEWORKS_CHROMA1_FLASH_ROOT",
                protocol::CHROMA1_FLASH_REPOSITORY,
            ),
            // sc-22737. The STILL Bernini entry. Its artifact is the CANDLE rehost
            // `SceneWorks/bernini` — a different repository from the macOS `SceneWorks/bernini-mlx`
            // one — which the manifest ships untiered at the download level while carrying `q4/`,
            // `q8/` and `bf16/` subtrees inside the snapshot, so the load root is still the tier
            // directory (`five_rung_root_is_tiered` is true here). The video member never reaches
            // this loader: it is dispatched above the still gate by `sc22737_video_arm`.
            BERNINI_CANDLE_ID => {
                bernini_candle_still_member(request)?;
                (
                    BERNINI_CANDLE_ID,
                    BERNINI_CANDLE_IMAGE_EXECUTION_PATH,
                    "SCENEWORKS_BERNINI_CANDLE_REPOSITORY",
                    "SCENEWORKS_BERNINI_CANDLE_REVISION",
                    "SCENEWORKS_BERNINI_CANDLE_ROOT",
                    protocol::BERNINI_CANDLE_REPOSITORY,
                )
            }
            provider => {
                return Err(format!(
                    "Candle five-rung calibration does not implement provider {provider:?}"
                ))
            }
        };
    let tier = planned_tier(request)?;
    validate_fixture_binds_tier_and_geometry(request)?;
    validate_five_rung_lane_tier(provider_id, tier)?;
    // The plan row must name the identity the LOADED contract publishes for this cell — checked
    // against the weights-free table BEFORE the load, so a stale row fails in milliseconds instead
    // of after a multi-gigabyte load (`run_five_rung_reference_loaded` still re-checks it against
    // the real contract afterwards, which is what keeps the two copies from drifting).
    if let Some(expected) = five_rung_calibration_fingerprint(provider_id, tier) {
        let planned = protocol::planned(request)?
            .get("calibrationFingerprint")
            .and_then(Value::as_str)
            .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
        if planned != expected {
            return Err(format!(
                "plan/provider calibration mismatch: plan={planned}, the {provider_id} {tier} \
                 production identity is {expected}"
            ));
        }
    }
    let repository = protocol::required_env(repository_env)?;
    let revision = protocol::required_env(revision_env)?;
    protocol::validate_artifact_identity(&repository, &revision, expected_repository)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(root_env)?))
        .map_err(|error| format!("canonicalize {root_env}: {error}"))?;
    if five_rung_root_is_tiered(provider_id) {
        // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
        // q8 or bf16 plan and quietly re-label another tier's peaks.
        protocol::validate_huggingface_snapshot_root(
            &root,
            &repository,
            &revision,
            tier,
            expected_repository,
        )?;
    } else {
        // The SANA dense snapshot has no tier sub-directory: the worker hands the engine the
        // snapshot root itself, and `candle-gen-sana`'s `validate_immutable_root` requires exactly
        // that. Inventing a `bf16/` component here would bind a path no production load opens.
        protocol::validate_huggingface_revision_root(
            &root,
            &repository,
            &revision,
            expected_repository,
        )?;
    }
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    // sc-22731: the worker binds the exact resolved route on every Candle image load
    // (`image_jobs/base.rs`: `.with_resolved_route(request.model.clone())`), and
    // `candle-gen-chroma`'s `validate_load_shape` refuses a spec without it by name. Set for the
    // two families whose engines were ported against that shape; the older arms keep the spec they
    // have always been measured with.
    let spec = if matches!(
        provider_id,
        SANA_ID | SANA_SPRINT_ID | CHROMA1_HD_ID | CHROMA1_BASE_ID | CHROMA1_FLASH_ID
    ) {
        spec.with_resolved_route(provider_id.to_owned())
    } else {
        spec
    };
    let spec = match (provider_id, numeric_tier(tier)?.quant) {
        // Krea's loader takes the packed tier's quant explicitly; bf16 is the dense base and must
        // carry no quant at all (`Quant::None` — the same shape the worker's `tier_to_quant` uses).
        (KREA_ID, Some(quant)) => spec.with_quant(quant),
        (KREA_ID, None) => spec,
        // FLUX.2 is per MEMBER (sc-22727 review): the dev route folds the planned tier the way the
        // worker's `candle_quant_for_resolved_tier` does, while both klein turnkeys are dense-TE
        // tiers the worker loads with `(None, resolved_bits)` — `candle-gen-flux2` quantizes the DiT
        // on-the-fly whenever `spec.quantize` is set, so folding it on a packed klein tier would
        // re-quantize the transformer and measure a load the app never performs. bf16 carries
        // `Quant::None` on every member, the worker's `tier_to_quant`.
        (FLUX2_DEV_ID | FLUX2_KLEIN_ID, Some(quant)) => {
            let arm =
                flux2_arm(request)?.expect("a FLUX.2 provider always resolves a member or errors");
            if arm.tier_quant_reaches_the_loader {
                spec.with_quant(quant)
            } else {
                spec
            }
        }
        (FLUX2_DEV_ID | FLUX2_KLEIN_ID, None) => spec,
        // Qwen, Z-Image-Turbo and the Z-Image base packed Diffusers snapshots declare their
        // device-format quantization in transformer/config.json (`snapshot_quant_tier` in
        // candle-gen-z-image's memory_strategy.rs). Passing LoadSpec.quant would request a second,
        // unsupported runtime quantization pass — every one of those loaders rejects it by name —
        // instead of loading the packed artifact as authored.
        //
        // sc-22730: the SD3.5 turnkeys are packed the same way, and `candle-gen-sd3`'s
        // `validate_load_shape` refuses `LoadSpec::quantize` on them OUTRIGHT — a request knob can
        // never outrank the artifact. `Sd35LoadReceipt::capture` then reads the transformer's
        // packing off the safetensors headers and cross-checks it against the path tier, so the
        // tier in the published identity is the tier on disk. Falling through here is therefore
        // what the worker does, not an omission.
        //
        // sc-22731 puts SANA and Chroma1 in the same class, from the worker itself:
        // `candle_quant_for_resolved_tier` returns `(None, _)` for both families at EVERY tier, so
        // `LoadSpec::quantize` is `None` on every shipped Candle render of them. Both engines
        // refuse a quant by name (`"Candle supports only the dense physical tier"`,
        // `"turnkey q4/q8/bf16 all require precision=Bf16 and LoadSpec.quantize=None"`), and
        // Chroma's tier comes from the artifact path plus the transformer's own packed marker.
        _ => spec,
    };
    // The catalog model id reaches the engine as `resolved_route` — the same lever the worker
    // sets (`image_jobs/base.rs`, `spec.with_resolved_route(request.model)`), and the only thing
    // that distinguishes two catalog models sharing one registry id (sc-22727).
    let spec = match flux2_arm(request)? {
        Some(arm) => spec.with_resolved_route(arm.model_id),
        None => spec,
    };
    // sc-22726: the FLUX.1 base snapshots declare their packed tier the same way; the directory
    // name proved nothing about the weights, so read the tier off the transformer config before
    // paying for the load.
    if matches!(provider_id, FLUX1_DEV_ID | FLUX1_SCHNELL_ID) {
        validate_flux_one_snapshot_tier(&spec, provider_id, tier)?;
    }
    // sc-22730: `candle-gen-sd3`'s `validate_load_shape` REFUSES a spec whose `resolved_route` is
    // not exactly this provider id — it is the first thing every SD3.5 production load checks
    // (`Sd35LoadReceipt::capture`), and the worker sets it on every render
    // (`image_jobs/base.rs` `.with_resolved_route(request.model.clone())`). No arm on this lane set
    // it before, because no provider on this lane required it; without it all nine SD3.5 candle
    // cells would be refused at load rather than measured. Scoped to the family that demands it so
    // no other provider's spec shape moves.
    let spec = if matches!(
        provider_id,
        SD3_5_LARGE_ID | SD3_5_LARGE_TURBO_ID | SD3_5_MEDIUM_ID
    ) {
        spec.with_resolved_route(provider_id)
    } else {
        spec
    };
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(provider_id, &spec)
        .map_err(|error| format!("load real {provider_id} {tier} generator: {error}"))?;
    vram.end_load(load_sample);
    Ok((
        provider_id,
        execution_path,
        repository,
        revision,
        generator,
        vram,
    ))
}

fn run_five_rung_reference_loaded(
    request: &Value,
    provider_id: &str,
    execution_path: &str,
    generator: &dyn runtime_cuda::gen_core::Generator,
    vram: &mut VramProbe,
    repository: &str,
    revision: &str,
) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, execution_path)?;
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {provider_id} has no memory-strategy contract"))?;
    let selection = planned_selection(request)?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {provider_id} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| format!("pinned {provider_id} provider has no calibration identity"))?;
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
    let planned_load_shape = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?;
    let actual_load_shape = match calibration.load_shape {
        LoadShape::EagerMaterialization => "eager_materialization",
        LoadShape::DeferredMaterialization => "deferred_materialization",
    };
    if planned_load_shape != actual_load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_load_shape}, pinned provider={actual_load_shape}"
        ));
    }
    let (width, height) = protocol::target_geometry(request)?;
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    // sc-22724: the mode the plan declared, as the worker would admit it. The `z_image_edit`
    // route carries one reference and is admitted under `MemoryMode::Edit` (the contract the
    // `z_image_edit` manifest entry declares for the Turbo provider).
    // sc-22733: the three Mage instruction editors are admitted under `MemoryMode::Edit` too — the
    // engine's own route gate requires it (`memory_strategy.rs` `route_is_supported`).
    let mage = mage_arm(request)?;
    let edit = is_z_image_edit(request)? || mage.is_some_and(|arm| arm.edit);
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: if edit {
            MemoryMode::Edit
        } else {
            MemoryMode::TextToImage
        },
        has_reference: edit,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: u32::from(edit),
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
        evidence_revision: format!(
            "{}@{}",
            five_rung_evidence_story(provider_id),
            protocol::INFERENCE_PIN
        ),
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin {provider_id} fresh-reference scope: {error}"))?
        .ok_or_else(|| {
            format!("{provider_id} fresh-reference selection did not create a provider scope")
        })?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {provider_id} fresh-reference decode: {error}"))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{provider_id} decode edge and overlap must be selected together"
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope.configure_attention(attention).map_err(|error| {
            format!("configure {provider_id} fresh-reference attention: {error}")
        })?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| {
                format!("configure {provider_id} fresh-reference transformer: {error}")
            })?;
    }
    let mut generation = match mage {
        Some(arm) => mage_generation_request(arm, width, height),
        None => five_rung_generation_request(width, height, edit),
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {provider_id} fresh-reference strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {provider_id} fresh-reference conditioning: {error}"))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let mut phase_error = None;
    let result = generator.generate(&generation, &mut |progress| {
        if phase_error.is_some() {
            return;
        }
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = peak,
            MemoryPhase::Denoise => denoise_peak_gb = peak,
            MemoryPhase::Decode => decode_peak_gb = peak,
        }
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {provider_id} {phase:?}: {error}"));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {provider_id} {next:?}: {error}"));
            return;
        }
        phase = next;
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    match result {
        Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)) if images.len() == 1 => {}
        Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)) => {
            return Err(format!(
                "{provider_id} fresh reference returned {} images",
                images.len()
            ));
        }
        Ok(_) => {
            return Err(format!(
                "{provider_id} fresh reference returned non-image output"
            ))
        }
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!(
                "{provider_id} fresh-reference generation failed: {message}"
            ));
        }
    }
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {provider_id} fresh-reference terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {provider_id} fresh-reference scope: {error}"))?;
    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        format!("{provider_id} fresh reference did not expose conditioning boundary")
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            format!("{provider_id} fresh reference did not expose denoise boundary")
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb
            .ok_or_else(|| format!("{provider_id} fresh reference did not complete decode"))?,
    );
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);
    let blocker = if provider_id == QWEN_ID {
        concat!(
            "SC-15817 five-rung conformance measures exact per-rung memory, strategy identity, ",
            "and loadability; it intentionally remains gated because this run does not repeat ",
            "each sibling story's promotion-quality, negative-mutation, and lifecycle suite"
        )
    } else if provider_id == Z_IMAGE_TURBO_ID {
        concat!(
            "SC-15859 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle Z-Image-Turbo lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if provider_id == Z_IMAGE_ID {
        concat!(
            "sc-22724 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle Z-Image base lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if provider_id == FLUX2_DEV_ID || provider_id == FLUX2_KLEIN_ID {
        concat!(
            "sc-22727 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle FLUX.2 lanes; it intentionally remains gated because this run does not ",
            "repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if provider_id == FLUX1_DEV_ID {
        concat!(
            "sc-22726 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle FLUX.1-dev lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if provider_id == FLUX1_SCHNELL_ID {
        concat!(
            "sc-22726 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle FLUX.1-schnell lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if MAGE_ARMS.iter().any(|arm| arm.provider == provider_id) {
        concat!(
            "sc-22733 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle Mage-Flow lane; it intentionally remains gated because this run does not ",
            "repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else if matches!(
        provider_id,
        SD3_5_LARGE_ID | SD3_5_LARGE_TURBO_ID | SD3_5_MEDIUM_ID
    ) {
        concat!(
            "sc-22730 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle SD3.5 lane; it intentionally remains gated because this run does not ",
            "repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else {
        concat!(
            "five-rung oracle capture measures exact per-rung memory and strategy identity for ",
            "SC-16059; it intentionally remains gated because this run does not repeat the full ",
            "promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    };
    let mut fragment = protocol::plain_gated_fragment(
        request,
        execution_path,
        protocol::PlainGatedFragment {
            artifact: artifact(repository, revision, planned_tier(request)?),
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint(
                    repository,
                    revision,
                    planned_tier(request)?,
                ),
            }),
            diagnostics: protocol::diagnostics(
                // sc-22724: `provider_id` is `z_image_turbo` for BOTH the text-to-image and the
                // edit capture, so the route has to be in the source or the two records are
                // indistinguishable by their own diagnostics. Mirrors `ZImageArm::slug` on MLX.
                // `provider_id` is `z_image_turbo` for BOTH the Turbo text-to-image and the edit
                // capture (sc-22724), and `flux2_klein_9b` for BOTH klein catalog models
                // (sc-22727), so the route has to be in the source or the records are
                // indistinguishable by their own diagnostics. Mirrors the MLX arms' slugs.
                &match flux2_arm(request)? {
                    Some(arm) => format!("memory-candle-adapter:{}-five-rung-reference", arm.slug),
                    None => format!(
                        "memory-candle-adapter:{provider_id}{}-five-rung-reference",
                        if edit { "-edit" } else { "" }
                    ),
                },
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    // The edit route conditions every request on one reference image.
                    ("referenceImages", "count", u64::from(edit)),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    Ok(fragment)
}

// ---------------------------------------------------------------------------------------------
// Mage-Flow (sc-22733) — six registered engine providers, on the shared five-rung reference path
// ---------------------------------------------------------------------------------------------

/// One fixed seed for every `candle:mage_flow*` fixture.
const MAGE_SEED: u64 = 22733;

/// One member of the Mage-Flow family. `candle-gen-mage` registers SIX generators (`lib.rs`
/// `REGISTRATION`, `BASE_`, `TURBO_`, `EDIT_`, `EDIT_BASE_`, `EDIT_TURBO_`), one per catalog id, and
/// the CUDA catalog registers all six (`candle-gen-catalog` `candle_gen_mage::register_providers`).
/// Each publishes its OWN per-tier calibration identity (`memory_strategy.rs`
/// `production_calibration_fingerprint`: `mage-flow-cuda-<provider>-<tier>-shared-ladder-v3`, from
/// `resolved_quant(spec)`), so a member or a tier measured under another's key would be caught by
/// the fingerprint comparison — the table below still binds member, artifact and fixture, and
/// [`mage_calibration_fingerprint`] binds the identity, so both are caught BEFORE the load is paid
/// for.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MageArm {
    provider: &'static str,
    execution_path: &'static str,
    still_calibration: &'static str,
    fixture_prefix: &'static str,
    repository: &'static str,
    /// Three literals, not a composed prefix — see the note on the MLX arm's identical fields:
    /// `measure-memory-catalog.mjs`'s test binds this table to `PROVIDER_FAMILIES` by searching
    /// this source for each exact name.
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    edit: bool,
    steps: u32,
    guidance: f32,
    evidence_story: &'static str,
}

const MAGE_ARMS: [MageArm; 6] = [
    MageArm {
        provider: "mage_flow",
        execution_path: "the Candle Mage-Flow RL text-to-image path",
        still_calibration: "Candle Mage-Flow calibration",
        fixture_prefix: "mage-flow-candle",
        repository: protocol::MAGE_FLOW_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_ROOT",
        edit: false,
        steps: 2,
        guidance: 5.0,
        evidence_story: "sc-22733",
    },
    MageArm {
        provider: "mage_flow_base",
        execution_path: "the Candle Mage-Flow-Base text-to-image path",
        still_calibration: "Candle Mage-Flow-Base calibration",
        fixture_prefix: "mage-flow-base-candle",
        repository: protocol::MAGE_FLOW_BASE_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_BASE_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_BASE_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_BASE_ROOT",
        edit: false,
        steps: 2,
        guidance: 5.0,
        evidence_story: "sc-22733",
    },
    MageArm {
        provider: "mage_flow_turbo",
        execution_path: "the Candle Mage-Flow-Turbo distilled text-to-image path",
        still_calibration: "Candle Mage-Flow-Turbo calibration",
        fixture_prefix: "mage-flow-turbo-candle",
        repository: protocol::MAGE_FLOW_TURBO_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_TURBO_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_TURBO_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_TURBO_ROOT",
        edit: false,
        steps: 4,
        guidance: 1.0,
        evidence_story: "sc-22733",
    },
    MageArm {
        provider: "mage_flow_edit",
        execution_path: "the Candle Mage-Flow-Edit instruction-editing path",
        still_calibration: "Candle Mage-Flow-Edit calibration",
        fixture_prefix: "mage-flow-edit-candle",
        repository: protocol::MAGE_FLOW_EDIT_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_EDIT_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_EDIT_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_EDIT_ROOT",
        edit: true,
        steps: 2,
        guidance: 5.0,
        evidence_story: "sc-22733",
    },
    MageArm {
        provider: "mage_flow_edit_base",
        execution_path: "the Candle Mage-Flow-Edit-Base instruction-editing path",
        still_calibration: "Candle Mage-Flow-Edit-Base calibration",
        fixture_prefix: "mage-flow-edit-base-candle",
        repository: protocol::MAGE_FLOW_EDIT_BASE_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_EDIT_BASE_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_EDIT_BASE_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_EDIT_BASE_ROOT",
        edit: true,
        steps: 2,
        guidance: 5.0,
        evidence_story: "sc-22733",
    },
    MageArm {
        provider: "mage_flow_edit_turbo",
        execution_path: "the Candle Mage-Flow-Edit-Turbo distilled instruction-editing path",
        still_calibration: "Candle Mage-Flow-Edit-Turbo calibration",
        fixture_prefix: "mage-flow-edit-turbo-candle",
        repository: protocol::MAGE_FLOW_EDIT_TURBO_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_EDIT_TURBO_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_EDIT_TURBO_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_EDIT_TURBO_ROOT",
        edit: true,
        steps: 4,
        guidance: 1.0,
        evidence_story: "sc-22733",
    },
];

/// The table lookup behind the six literal dispatch arms in [`plain_execution_path`] and
/// [`still_calibration_label`]. Infallible by construction — every caller passes a
/// [`MAGE_ARMS`] provider — but returns a `Result` so the dispatch arms stay expression-bodied
/// literals, which is what the derived capturability report can parse.
fn mage_execution_path(provider: &str) -> Result<&'static str, String> {
    MAGE_ARMS
        .iter()
        .find(|arm| arm.provider == provider)
        .map(|arm| arm.execution_path)
        .ok_or_else(|| format!("no Candle Mage-Flow arm for provider {provider:?}"))
}

fn mage_still_calibration(provider: &str) -> Result<&'static str, String> {
    MAGE_ARMS
        .iter()
        .find(|arm| arm.provider == provider)
        .map(|arm| arm.still_calibration)
        .ok_or_else(|| format!("no Candle Mage-Flow arm for provider {provider:?}"))
}

/// The Mage member this case names, or `None` when the plan is not a Mage case at all. Keyed on
/// `(provider, modelId)` like the MLX arm: the catalog id and the engine provider id are equal on
/// all six Mage rows (`engines.rs` `MODEL_TABLE`), and this asserts that rather than assuming it.
fn mage_arm(request: &Value) -> Result<Option<MageArm>, String> {
    let provider = planned_provider(request)?;
    let Some(arm) = MAGE_ARMS.into_iter().find(|arm| arm.provider == provider) else {
        return Ok(None);
    };
    let model_id = protocol::planned(request)?
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != arm.provider {
        return Err(format!(
            "the Candle Mage-Flow arm does not implement provider {provider:?} for model \
             {model_id:?}"
        ));
    }
    Ok(Some(arm))
}

/// The mode the ARM will actually render — `edit_image` on the three instruction editors.
fn mage_mode(arm: MageArm) -> &'static str {
    if arm.edit {
        "edit_image"
    } else {
        "text_to_image"
    }
}

/// The plan's declared mode must be the one this member renders. The record's `mode` comes from the
/// PLAN while the reference and the admitted `MemoryMode` come from the ARM, so an unchecked
/// mismatch would emit a reference-conditioned render's peaks under a reference-free label.
fn validate_mage_mode(request: &Value, arm: MageArm) -> Result<(), String> {
    let declared = planned_mode(request)?;
    let expected = mage_mode(arm);
    if declared != expected {
        return Err(format!(
            "{} renders {expected:?}, but the plan declares mode {declared:?}",
            arm.provider
        ));
    }
    Ok(())
}

/// The Mage fixture binds the member, the tier, the edge, the seed and the recipe's step count.
fn validate_mage_fixture(request: &Value, arm: MageArm, tier: &str) -> Result<(), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{}-{tier}-{width}-seed", arm.fixture_prefix);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Mage-Flow fixture seed {seed:?}: {error}"))?;
    if seed != MAGE_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the seed {} renders at ({MAGE_SEED})",
            arm.provider
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse Mage-Flow fixture step count {steps:?}: {error}"))?;
    if steps != arm.steps {
        return Err(format!(
            "planned.fixture {fixture:?} must use this arm's {}-step calibration request",
            arm.steps
        ));
    }
    Ok(())
}

/// The `LoadSpec` one Mage capture opens, built exactly as the worker's generic Candle stream does.
///
/// * A Mage variant rehost ships the DiT ALONE. The text encoder and the VAE come from the shared
///   components snapshot, staged EXPLICITLY through `LoadSpec::with_component` under the two ids
///   `candle-gen-mage` advertises — this is the "Mage's split text-encoder/VAE component paths"
///   the worker's own selector comment names (`image_jobs/base.rs`).
/// * The quant is the PLANNED tier's. `candle-gen-mage` `memory_strategy::resolved_quant` reads
///   `spec.quantize` directly, so unlike the packed Diffusers snapshots on this lane the tier is
///   carried on the spec rather than read out of `transformer/config.json`.
/// * The offload policy is `Sequential`: the candle anchor rung is `staged_residency`
///   (`memory-calibration-harness.mjs` `ANCHOR_STRATEGY.candle`), which is exactly the rung the
///   worker selects `use_sequential` for (`image_jobs/base.rs` binds `OffloadPolicy::Sequential`
///   after the declaration pass).
/// * The load shape is the PLAN's, and the plan's is the worker's, per tier. Every Mage manifest
///   entry's `candle.memoryStrategyContract.implementations` declares
///   `bounded_transformer_residency`, and the six Candle Mage rows of
///   `memory_route_registry::RULES` (`ALL_TIERS` × `ALL_MODES`, `PLAIN`,
///   `requires_sequential_selection: false`) exist for every planned cell, so the worker's
///   `apply_declared_candle_image_load_shape` → `evaluate_declared_candle_load_shape` owns the
///   shape rather than leaving it at its default. That evaluator matches the declaration on the
///   BTR row's OWN `tiers`, and the generated BTR row lists `["bf16"]` only — `candle-gen-mage`
///   publishes BTR `Implemented` where `memory_strategy::streamable` AND
///   `transformer_has_device_format` hold, and the stage-1 dump found device-format blocks on the
///   bf16 snapshot alone. So bf16 is `Applied + DeferredMaterialization` (the engine is asked about
///   a Deferred candidate and `streamable` holds for this plain directory load), and q4/q8 are
///   `Refused + EagerMaterialization`; both regardless of the selected rung. The worker's own
///   `mage_candle_production_load_shape_is_deferred_on_bf16_and_eager_on_the_packed_tiers` drives
///   that evaluator over the real manifest entries and pins the 18 `mage_flow*:*:candle` plan rows
///   to it; `the_load_spec_is_the_shape_the_worker_loads` below pins THIS spec to those same rows,
///   and a capture re-asserts the loaded contract's `MemoryCalibrationIdentity::load_shape` against
///   the plan. The candle Mage identity is keyed on the resolved tier alone (`-v3`), so the shape is
///   not an identity axis and neither planned shape can name the other tier's cell.
fn mage_load_spec(
    tier: &str,
    load_shape: LoadShape,
    root: PathBuf,
    components_root: &std::path::Path,
) -> Result<LoadSpec, String> {
    let tier_components = components_root.join(tier);
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(load_shape)
        .with_component(
            protocol::MAGE_COMPONENT_TEXT_ENCODER,
            WeightsSource::Dir(tier_components.join(protocol::MAGE_COMPONENT_TEXT_ENCODER)),
        )
        .with_component(
            protocol::MAGE_COMPONENT_VAE,
            WeightsSource::Dir(tier_components.join(protocol::MAGE_COMPONENT_VAE)),
        );
    if let Some(quant) = numeric_tier(tier)?.quant {
        spec = spec.with_quant(quant);
    }
    Ok(spec)
}

/// The production calibration identity the loaded Candle Mage generator publishes for one
/// `(member, tier)` cell — the table `candle-gen-mage::memory_strategy::production_calibration_fingerprint`
/// mints (inference PR 953): `mage-flow-cuda-<provider>-<tier>-shared-ladder-v3`, eighteen distinct
/// strings, the tier read off `resolved_quant(spec)` (the quant this arm carries on the spec).
/// Written here as well so the plan/arm binding is weights-free and holds at inference
/// `c6d6a4db`, whose engine still publishes the tier-free `-v2` string; the capture refuses a
/// loaded contract whose identity differs from the plan, so the two copies cannot drift unnoticed
/// once the epic's pin bump lands.
fn mage_calibration_fingerprint(arm: MageArm, tier: &str) -> String {
    format!(
        "mage-flow-cuda-{}-{tier}-shared-ladder-v3",
        arm.provider.replace('_', "-")
    )
}

/// The load shape the plan declares for this cell, which the capture must execute under and then
/// re-assert against the loaded contract's identity. Deriving it from the selected rung would
/// silently rewrite the per-tier shape the worker binds (see [`mage_load_spec`]).
fn mage_planned_load_shape(request: &Value) -> Result<LoadShape, String> {
    match protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?
    {
        protocol::LOAD_SHAPE_EAGER => Ok(LoadShape::EagerMaterialization),
        protocol::LOAD_SHAPE_DEFERRED => Ok(LoadShape::DeferredMaterialization),
        other => Err(format!("unsupported planned.loadShape {other:?}")),
    }
}

/// The plan row must name the production identity this cell's loaded generator publishes —
/// checked against the weights-free table BEFORE the load, so a row still carrying the retired
/// tier-free `-v2` string (or an MLX identity) fails in milliseconds rather than after a
/// multi-gigabyte load.
fn validate_mage_plan_identity(request: &Value, arm: MageArm, tier: &str) -> Result<(), String> {
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    let expected_fingerprint = mage_calibration_fingerprint(arm, tier);
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} {tier} \
             production identity is {expected_fingerprint}",
            arm.provider
        ));
    }
    Ok(())
}

/// The generation request one Mage capture renders — the worker's request shape for this member.
/// The edit members refuse a reference-free request (`candle-gen-mage` `edit_provider.rs`), and the
/// engine's route gate additionally requires `MemoryMode::Edit` with 1..=8 references
/// (`memory_strategy.rs` `route_is_supported`), so the single fitted reference is what makes an edit
/// capture measure the edit path rather than text-to-image under an edit label.
fn mage_generation_request(arm: MageArm, width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: if arm.edit {
            "replace the background with a plain grey studio backdrop".to_owned()
        } else {
            "a weathered brass astrolabe on a linen cloth, soft window light".to_owned()
        },
        // The distilled members run CFG genuinely off at guidance 1.0, at which the engine builds no
        // unconditional branch at all.
        negative_prompt: (arm.guidance > 1.0).then(|| "blurry, distorted, text".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(MAGE_SEED),
        steps: Some(arm.steps),
        guidance: Some(arm.guidance),
        conditioning: if arm.edit {
            vec![Conditioning::Reference {
                image: Image {
                    width,
                    height,
                    // `strength` stays None: Mage's `edit_references` consumes the reference image
                    // itself and never reads a strength lever, so declaring one would state a
                    // parameter the render does not use.
                    pixels: protocol::synthetic_reference_rgb(width, height),
                },
                strength: None,
            }]
        } else {
            Vec::new()
        },
        ..Default::default()
    }
}

/// The one fresh planned request every five-rung reference renders. Two steps are intentional:
/// a resident image provider has no loading boundary between text encode and denoise, so the
/// first Step callback closes a conservative conditioning envelope and the second gives denoise
/// its own measured interval before Decoding. The edit route (sc-22724) is the worker's edit
/// request — one `Conditioning::Reference` fitted to the request geometry plus the strength
/// lever (`resolve_zimage_edit_init`) — with the step count raised so the engine-derived start
/// step (`floor(steps * strength)`) still leaves two executed denoise steps.
fn five_rung_generation_request(width: u32, height: u32, edit: bool) -> GenerationRequest {
    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(FIVE_RUNG_SEED),
        steps: Some(2),
        ..Default::default()
    };
    if edit {
        generation.steps = Some(Z_IMAGE_EDIT_STEPS);
        // `request.strength` stays None: the worker sets ONLY the per-reference strength
        // (`build_lane_conditioning`, image_jobs/base.rs:7136) and leaves the request-level lever —
        // gen-core's documented fallback for a single `Reference` with no strength of its own —
        // unset. This arm reproduces the worker's request shape, so it does the same (sc-22724).
        generation.conditioning = vec![Conditioning::Reference {
            image: Image {
                width,
                height,
                pixels: protocol::synthetic_reference_rgb(width, height),
            },
            strength: Some(Z_IMAGE_EDIT_STRENGTH),
        }];
    }
    generation
}

// ---------------------------------------------------------------------------------------------
// The bespoke Candle PuLID-FLUX arm (sc-22726).
// ---------------------------------------------------------------------------------------------

/// Everything one PuLID capture binds before a weight file is opened: the FLUX.1-dev backbone at
/// the PLANNED tier, the staged identity stack, and the paths struct the bespoke provider takes.
struct PulidFluxBinding {
    repository: String,
    revision: String,
    tier: &'static str,
    bundle: protocol::PulidIdentityBundle,
    paths: runtime_cuda::providers::pulid::PulidFluxPaths,
}

/// Hand-written because `PulidFluxPaths` (an inference type) derives no `Debug`; the fingerprint
/// already names every path the binding resolved, so it is the whole useful content.
impl std::fmt::Debug for PulidFluxBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PulidFluxBinding")
            .field("fingerprint", &self.loadability_fingerprint())
            .finish()
    }
}

impl PulidFluxBinding {
    /// Names the identity stack by CONTENT (the bundle's composite SHA-256), never by the host
    /// path it was staged at — the same token the MLX arm's fingerprint carries.
    fn loadability_fingerprint(&self) -> String {
        format!(
            "{}@{}:{}+identity:{}",
            self.repository, self.revision, self.tier, self.bundle.composite_sha256
        )
    }

    /// The record's `artifact`: the backbone snapshot plus every bundle file's digest.
    fn artifact_json(&self) -> Value {
        let mut artifact = artifact(&self.repository, &self.revision, self.tier);
        artifact["identityBundle"] = self.bundle.artifact_json();
        artifact
    }
}

/// The env-free half of [`pulid_flux_binding`], so the tier and identity bindings are provable
/// without weights or a GPU.
fn pulid_flux_binding_at(
    request: &Value,
    repository: String,
    revision: String,
    root: PathBuf,
    bundle: protocol::PulidIdentityBundle,
) -> Result<PulidFluxBinding, String> {
    if planned_provider(request)? != PULID_FLUX_ID {
        return Err(format!(
            "the Candle PuLID-FLUX arm does not implement provider {:?}",
            planned_provider(request)?
        ));
    }
    // PuLID is text-to-image-WITH-A-FACE only: the worker's `pulid_candle_available` requires
    // `character_image` and a reference, and the provider's own route gate refuses anything else.
    let mode = planned_mode(request)?;
    if mode != "character_image" {
        return Err(format!(
            "the Candle PuLID-FLUX arm does not implement mode {mode:?}; the route is \
             character_image only"
        ));
    }
    protocol::validate_exact_overlay_target(request, "identity", PULID_FLUX_EXECUTION_PATH)?;
    let tier = match planned_tier(request)? {
        "bf16" => "bf16",
        "q4" => "q4",
        "q8" => "q8",
        _ => unreachable!("planned_tier returned an unsupported tier"),
    };
    validate_flux_one_fixture(request, PULID_FLUX_ID, tier)?;
    // The backbone IS the FLUX.1-dev artifact on this route (`PULID_CANDLE_FLUX_REPO`), so it binds
    // the FLUX1_DEV family — and the root must still end in the PLANNED tier's directory, so a
    // stale `…/q4` export cannot satisfy a q8 or bf16 plan.
    protocol::validate_artifact_identity(&repository, &revision, protocol::FLUX1_DEV_REPOSITORY)?;
    let root = std::fs::canonicalize(&root)
        .map_err(|error| format!("canonicalize SCENEWORKS_FLUX1_DEV_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::FLUX1_DEV_REPOSITORY,
    )?;
    let paths = runtime_cuda::providers::pulid::PulidFluxPaths {
        flux_base: root,
        pulid_weights: bundle.adapter.clone(),
        eva_weights: bundle.eva.clone(),
        face_dir: bundle.face_dir.clone(),
        // No LoRA adapters: the worker gates the PuLID memory ladder on
        // `request.loras.is_empty()` (`pulid_memory_ladder_eligible`), so a ladder-admitted PuLID
        // render carries none, and an anchor must measure the admitted shape.
        adapters: Vec::new(),
    };
    Ok(PulidFluxBinding {
        repository,
        revision,
        tier,
        bundle,
        paths,
    })
}

fn pulid_flux_binding(request: &Value) -> Result<PulidFluxBinding, String> {
    let repository = protocol::required_env("SCENEWORKS_FLUX1_DEV_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_FLUX1_DEV_REVISION")?;
    let root = PathBuf::from(protocol::required_env("SCENEWORKS_FLUX1_DEV_ROOT")?);
    let bundle = protocol::pulid_identity_bundle()?;
    pulid_flux_binding_at(request, repository, revision, root, bundle)
}

/// The one request every PuLID capture renders, in the worker's shape: the manifest's photoreal
/// preset guidance and id_weight, the native sampler/scheduler defaults, and two steps so the first
/// Step callback closes a conservative conditioning envelope and the second gives denoise its own
/// measured interval before Decoding.
fn pulid_flux_generation_request(width: u32, height: u32) -> PulidFluxRequest {
    PulidFluxRequest {
        prompt: "a portrait of a person in a sunlit studio, editorial photograph".to_owned(),
        width,
        height,
        steps: 2,
        guidance: PULID_FLUX_GUIDANCE,
        id_weight: PULID_FLUX_ID_WEIGHT,
        seed: FLUX1_SEED,
        use_pid: false,
        ..Default::default()
    }
}

/// The admission context the worker admits this route under (`evaluate_shared_bespoke_image`,
/// `pulid_candle.rs`): the PROVIDER mode `character_image` with exactly one reference and
/// `overlay: identity`, no PiD, no request phases. `candle-gen-pulid`'s `safety_check` refuses
/// anything else by name, so this shape is not a choice.
fn pulid_flux_context(
    selection: MemorySelection,
    calibration: &runtime_cuda::gen_core::MemoryCalibrationIdentity,
    fingerprint: &str,
    width: u32,
    height: u32,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("character_image".to_owned()),
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
        overlay: Some("identity".to_owned()),
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22726@{}", protocol::INFERENCE_PIN),
    }
}

/// The `candle:pulid_flux` arm. Unlike every other Candle arm this one never touches the provider
/// registry: `candle-gen-pulid` registers no `Generator` at all, and the worker loads it as
/// `PulidFlux::load_with_memory_context(&PulidFluxPaths, ctx)` — so that is the load path measured
/// here (E4). The memory contract is likewise path-shaped rather than spec-shaped
/// (`memory_strategy::provider_contract(&paths)`).
fn run_pulid_flux_capture(request: &Value) -> Result<Value, String> {
    use runtime_cuda::providers::pulid::{memory_strategy as pulid_memory, PulidFlux};

    // Before any environment or weight work, under this route's own label.
    protocol::validate_still_geometry(request, PULID_FLUX_STILL_CALIBRATION)?;
    let binding = pulid_flux_binding(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    let selection = planned_selection(request)?;

    // The tier the SNAPSHOT actually declares (`transformer/config.json`), read through the
    // provider's own resolver. The root suffix already proved the plan and the export agree on the
    // directory NAME; this proves the weights inside it agree too, which is the half a renamed
    // directory could otherwise fake.
    let resolved = pulid_memory::resolved_numeric_tier(&binding.paths)
        .map_err(|error| format!("resolve PuLID-FLUX numeric tier: {error}"))?;
    let expected = numeric_tier(binding.tier)?;
    if (resolved.precision, resolved.quant) != (expected.precision, expected.quant) {
        return Err(format!(
            "planned tier {} does not match the tier the PuLID backbone snapshot declares \
             (precision={:?}, quant={:?})",
            binding.tier, resolved.precision, resolved.quant
        ));
    }

    let contract = pulid_memory::provider_contract(&binding.paths)
        .map_err(|error| format!("read the pinned PuLID-FLUX memory contract: {error}"))?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned PuLID-FLUX provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "the pinned PuLID-FLUX contract has no calibration identity".to_owned())?;
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
    let planned_load_shape = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?;
    if planned_load_shape != load_shape_key(calibration.load_shape) {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_load_shape}, pinned provider={}",
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        pulid_memory::safety_check(
            &binding.paths,
            &contract,
            &pulid_flux_context(
                selection,
                calibration,
                fingerprint,
                width,
                height,
                total_bytes,
                predicted,
            ),
        )
    };
    // Admission mutation hygiene BEFORE the expensive load: the gate must ACCEPT a fitting request,
    // so the two rejections below cannot pass through a blanket refusal.
    if !matches!(
        safety(&calibration.fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "PuLID-FLUX admission rejected a fitting probe budget; the scenario rejections below \
             would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("PuLID-FLUX admission accepted an unknown/zero memory budget".to_owned());
    }
    if !matches!(
        safety("stale-pulid-flux-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("PuLID-FLUX admission accepted stale calibration evidence".to_owned());
    }

    let context = pulid_flux_context(
        selection,
        calibration,
        &calibration.fingerprint,
        width,
        height,
        hardware_bytes,
        1,
    );
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    // The production load path: the bespoke provider, admitted at load with the exact context it
    // will then be asked to honour. `generate_with_memory_context` refuses a context that differs.
    let model = PulidFlux::load_with_memory_context(&binding.paths, context.clone())
        .map_err(|error| format!("load real pulid_flux {} provider: {error}", binding.tier))?;
    vram.end_load(load_sample);

    let reference = runtime_cuda::gen_core::Image {
        width,
        height,
        pixels: protocol::synthetic_reference_rgb(width, height),
    };
    let generation = pulid_flux_generation_request(width, height);
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    // No `MemoryRequestScope` exists on this route — the provider admits at load rather than
    // opening a per-request scope — so the phase boundaries are driven by the progress stream
    // alone, exactly as the MLX image arms drive theirs.
    let result =
        model.generate_with_memory_context(&context, &generation, &reference, &mut |progress| {
            let boundary = match progress {
                Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                    protocol::ReferenceBoundary::RendererLoad
                }
                Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
                Progress::Decoding => protocol::ReferenceBoundary::Decoding,
                _ => return,
            };
            let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary)
            else {
                return;
            };
            let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
            match phase {
                MemoryPhase::Conditioning => conditioning_peak_gb = peak,
                MemoryPhase::Denoise => denoise_peak_gb = peak,
                MemoryPhase::Decode => decode_peak_gb = peak,
            }
            phase = memory_phase(next);
            phase_sample = Some(vram.phase());
        });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    result.map_err(|error| format!("pulid_flux measured generation failed: {error}"))?;

    let conditioning_bytes = decimal_gb_to_bytes(
        conditioning_peak_gb
            .ok_or_else(|| "pulid_flux did not expose a conditioning boundary".to_owned())?,
    );
    let denoise_bytes = decimal_gb_to_bytes(
        denoise_peak_gb.ok_or_else(|| "pulid_flux did not expose a denoise boundary".to_owned())?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb.ok_or_else(|| "pulid_flux did not complete decode".to_owned())?,
    );
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);

    let blocker = concat!(
        "sc-22726 anchor capture measures exact per-phase memory and strategy identity for the ",
        "Candle PuLID-FLUX lane; it intentionally remains gated because this run does not repeat ",
        "the full promotion-quality, negative-mutation, and lifecycle scenario suite, and the ",
        "bespoke route opens no memory-strategy request scope to inject a calibration fault into"
    );
    Ok(json!({
        "status": "gated",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": binding.artifact_json(),
        "sweep": protocol::reference_sweep(request, "passed")?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": blocker },
            { "name": "unknown_budget", "result": "passed", "reason": "the pinned PuLID-FLUX admission check rejected a zero/unknown budget before load" },
            { "name": "stale_evidence", "result": "passed", "reason": "the pinned PuLID-FLUX admission check rejected a mutated calibration fingerprint before load" },
            { "name": "warm_repeat", "result": "not_run", "reason": blocker },
            { "name": "cancel", "result": "not_run", "reason": blocker },
            { "name": "error", "result": "not_run", "reason": blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "passed", "reason": "the PuLID identity stack (adapter, EVA tower, and the three face models) was resident for the measured render and is declared as its own resident component by the pinned contract" }
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
            "resolvedPathFingerprint": binding.loadability_fingerprint(),
        },
        "diagnostics": protocol::diagnostics(
            "memory-candle-adapter:pulid-flux-bespoke-reference",
            "executed",
            [blocker.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
                ("referenceImages", "count", 1),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    }))
}

/// The story whose evidence a five-rung reference record cites in `evidence_revision`: the
/// story that gave the provider its arm on this lane. Krea/Qwen/Z-Image keep the SC-16402 tag
/// their packaged records already carry; the FLUX.1 members (sc-22726) and the SD3.5 members
/// (sc-22730) cite their own.
fn five_rung_evidence_story(provider_id: &str) -> &'static str {
    if let Some(arm) = MAGE_ARMS.iter().find(|arm| arm.provider == provider_id) {
        return arm.evidence_story;
    }
    match provider_id {
        FLUX1_DEV_ID | FLUX1_SCHNELL_ID => "sc-22726",
        SD3_5_LARGE_ID | SD3_5_LARGE_TURBO_ID | SD3_5_MEDIUM_ID => "sc-22730",
        BERNINI_CANDLE_ID => "sc-22737",
        _ => "sc-16402",
    }
}

fn run_five_rung_reference(request: &Value) -> Result<Value, String> {
    let execution_path = plain_execution_path(request)?;
    protocol::validate_plain_overlay_target(request, execution_path)?;
    // Before `load_five_rung_generator`, for the same reason the overlay check is duplicated here:
    // a geometry this arm cannot honour must be refused before it costs a real weight load.
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    let (provider_id, execution_path, repository, revision, generator, mut vram) =
        load_five_rung_generator(request)?;
    let mut fragment = run_five_rung_reference_loaded(
        request,
        provider_id,
        execution_path,
        generator.as_ref(),
        &mut vram,
        &repository,
        &revision,
    )?;
    // sc-22733: a Mage load opens TWO artifact triples, and the shared loadability fingerprint
    // names only the variant's. A load that took the DiT from the planned variant but the text
    // encoder from another tier's components would otherwise leave no trace in the record.
    if mage_arm(request)?.is_some() {
        let tier = planned_tier(request)?;
        let components_repository =
            protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REPOSITORY")?;
        let components_revision =
            protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REVISION")?;
        let variant = fragment
            .pointer("/loadability/resolvedPathFingerprint")
            .and_then(Value::as_str)
            .ok_or_else(|| "the five-rung fragment carries no loadability fingerprint".to_owned())?
            .to_owned();
        fragment["loadability"]["resolvedPathFingerprint"] = json!(format!(
            "{variant}+{components_repository}@{components_revision}:{tier}"
        ));
    }
    Ok(fragment)
}

fn update_warmed_retention_baseline(
    settled_after_resident: &mut Option<u64>,
    after: u64,
) -> Result<(), String> {
    if let Some(baseline) = *settled_after_resident {
        if after > baseline.saturating_add(64 * MIB) {
            return Err(format!(
                "reused Krea rung retained {} bytes above the warmed resident baseline; refusing contaminated batching",
                after - baseline
            ));
        }
    } else {
        *settled_after_resident = Some(after);
    }
    Ok(())
}

fn run_five_rung_batch(request: &Value) -> Result<Value, String> {
    let planned = request
        .get("planned")
        .and_then(Value::as_array)
        .ok_or_else(|| "run_batch request.planned must be an array".to_owned())?;
    let expected_rungs = [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ];
    let actual_rungs = planned
        .iter()
        .map(|item| {
            item.pointer("/strategy/rung")
                .and_then(Value::as_str)
                .ok_or_else(|| "batched planned strategy.rung must be a string".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if actual_rungs != expected_rungs {
        return Err(format!(
            "run_batch requires one canonical five-rung target, got {actual_rungs:?}"
        ));
    }
    let first_target = planned[0]
        .get("target")
        .ok_or_else(|| "batched planned target is missing".to_owned())?;
    if planned
        .iter()
        .any(|item| item.get("target") != Some(first_target))
    {
        return Err("run_batch cannot mix calibration targets in one model load".to_owned());
    }
    for item in planned {
        let mut per_rung_request = request.clone();
        per_rung_request["action"] = json!("run");
        per_rung_request["planned"] = item.clone();
        let execution_path = plain_execution_path(&per_rung_request)?;
        protocol::validate_plain_overlay_target(&per_rung_request, execution_path)?;
        protocol::validate_still_geometry(
            &per_rung_request,
            still_calibration_label(&per_rung_request)?,
        )?;
    }

    let mut first_request = request.clone();
    first_request["action"] = json!("run");
    first_request["planned"] = planned[0].clone();
    let (provider_id, execution_path, repository, revision, generator, mut vram) =
        load_five_rung_generator(&first_request)?;
    let smi = NvidiaSmi::resolve()?;
    // Krea uses DeferredMaterialization, so loading the generator does not establish its
    // steady-state device residency. The canonical batch starts with `resident`; use the
    // memory retained after that first rung as the contamination baseline, then require every
    // later rung to release its transient allocations back to that warmed state.
    let mut settled_after_resident = None;
    let mut fragments = Vec::with_capacity(planned.len());
    for item in planned {
        let mut per_rung_request = request.clone();
        per_rung_request["action"] = json!("run");
        per_rung_request["planned"] = item.clone();
        fragments.push(run_five_rung_reference_loaded(
            &per_rung_request,
            provider_id,
            execution_path,
            generator.as_ref(),
            &mut vram,
            &repository,
            &revision,
        )?);
        let after = smi.used_bytes()?;
        update_warmed_retention_baseline(&mut settled_after_resident, after)?;
    }
    // sc-22667 (epic 22657 E6): the LOADED provider's component bytes, architecture facts and
    // published rung parameters ride along with the fragments, so the derived side of the
    // falsification prices the five rungs from exactly the contract this measurement ran under
    // rather than a hand-constructed twin. Additive: the harness reads `fragments` only.
    let provider_contract = generator
        .memory_strategy_contract()
        .map(loaded_contract_facts)
        .ok_or_else(|| format!("loaded {provider_id} has no memory-strategy contract"))?;
    Ok(json!({
        "modelLoads": 1,
        "fragments": fragments,
        "providerContract": provider_contract,
    }))
}

/// The facts of a loaded provider contract the worker's candle ladder prices from
/// (`anchor_component_bytes(contract.asset_facts)`, `architecture_facts_from_contract`,
/// `estimate_floor_parameters`), serialized verbatim in the contract's own field names.
fn loaded_contract_facts(contract: &runtime_cuda::gen_core::MemoryProviderContract) -> Value {
    let assets = contract.asset_facts;
    let facts = contract.architecture_facts;
    let strategies: Vec<Value> = contract
        .strategies
        .iter()
        .map(|capability| {
            let ranges = &capability.parameters;
            json!({
                "strategy": strategy_name(capability.strategy),
                "support": format!("{:?}", capability.support),
                "engagedRungs": contract
                    .engaged_composition(capability.strategy)
                    .into_iter()
                    .map(strategy_name)
                    .collect::<Vec<_>>(),
                "parameters": {
                    "decodeTileEdges": ranges.decode_tile_edges,
                    "decodeOverlaps": ranges.decode_overlaps,
                    "attentionChunkSizes": ranges.attention_chunk_sizes,
                    "transformerWindowSizes": ranges.transformer_window_sizes,
                    "transformerWindowComponents": ranges
                        .transformer_window_components
                        .iter()
                        .map(|component| format!("{component:?}"))
                        .collect::<Vec<_>>(),
                },
            })
        })
        .collect();
    json!({
        "providerId": contract.provider_id,
        "loadShape": load_shape_key(contract.load_shape),
        "calibrationFingerprint": contract
            .calibration
            .as_ref()
            .map(|calibration| calibration.fingerprint.clone()),
        "assetFacts": {
            "baseBytes": assets.base_bytes,
            "conditioningBytes": assets.conditioning_bytes,
            "transformerBytes": assets.transformer_bytes,
            "decoderBytes": assets.decoder_bytes,
            "overlayBytes": assets.overlay_bytes,
        },
        "architectureFacts": {
            "attentionHeads": facts.attention_heads,
            "headDim": facts.head_dim,
            "transformerBlocks": facts.transformer_blocks,
            "patchSize": facts.patch_size,
            "latentChannels": facts.latent_channels,
            "vaeSpatialScale": facts.vae_spatial_scale,
            "vaeTemporalScale": facts.vae_temporal_scale,
            "activationDtypeWidth": facts.activation_dtype_width,
        },
        "strategies": strategies,
    })
}

/// The declared materialization shape, with no per-provider variant rule (sc-22736).
///
/// [`ltx25_planned_load_shape`] asks LTX-2.5's transformer variant first, because that family's two
/// variants genuinely differ. Every other Candle video route declares no bounded-transformer
/// residency, so `memory_route_registry::evaluate_declared_candle_load_shape` hands its spec
/// straight back and the shape is simply what the plan says.
fn planned_video_load_shape(request: &Value) -> Result<LoadShape, String> {
    match protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?
    {
        protocol::LOAD_SHAPE_EAGER => Ok(LoadShape::EagerMaterialization),
        protocol::LOAD_SHAPE_DEFERRED => Ok(LoadShape::DeferredMaterialization),
        other => Err(format!("unsupported Candle video loadShape {other:?}")),
    }
}

fn ltx25_planned_load_shape(
    request: &Value,
    transformer_variant: protocol::Ltx25TransformerVariant,
) -> Result<LoadShape, String> {
    let declared = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?;
    transformer_variant.validate_load_shape(declared)?;
    match declared {
        protocol::LOAD_SHAPE_EAGER => Ok(LoadShape::EagerMaterialization),
        protocol::LOAD_SHAPE_DEFERRED => Ok(LoadShape::DeferredMaterialization),
        other => Err(format!("unsupported LTX-2.5 Candle loadShape {other:?}")),
    }
}

fn ltx25_official_dev_adapter(revision: &str) -> Result<AdapterSpec, String> {
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX25_DISTILL_LORA_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX25_DISTILL_LORA_ROOT: {error}"))?;
    protocol::validate_huggingface_revision_root(
        &root,
        protocol::LTX_2_5_REPOSITORY,
        revision,
        protocol::LTX_2_5_REPOSITORY,
    )?;
    let path = root.join(LTX25_DISTILL_LORA_RELATIVE_PATH);
    if !path.is_file() {
        return Err(format!(
            "LTX-2.5 dev capture requires the pinned official stage-two refinement LoRA at {}",
            path.display()
        ));
    }
    // The base scale MUST be the stage-one scale (0.0), not 1.0: production's
    // `resolve_ltx_distill_adapter` builds `AdapterSpec::new(path, stage1, ..)` with the manifest's
    // required `[0, 1]` contract, and the MLX capture arm does the same. gen_core's
    // `adapter_stack_identity` digests the scale bits into the admission overlay, so capturing at
    // 1.0 would mint evidence under an overlay identity no production request can ever present.
    Ok(AdapterSpec::new(path, 0.0, AdapterKind::Lora).with_pass_scales(vec![0.0, 1.0]))
}

struct Ltx25LoadPlan {
    repository: String,
    revision: String,
    inventory_sha256: String,
    load_shape: LoadShape,
    adapters: Vec<AdapterSpec>,
    spec: LoadSpec,
}

fn ltx25_load_spec(
    request: &Value,
    target: &protocol::Ltx25CandleTarget,
) -> Result<Ltx25LoadPlan, String> {
    let load_shape = ltx25_planned_load_shape(request, target.transformer_variant)?;
    let repository = protocol::required_env("SCENEWORKS_LTX25_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_LTX25_REVISION")?;
    protocol::validate_ltx25_artifact_identity(&repository, &revision)?;
    let inventory_sha256 = protocol::required_env("SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256")?;
    protocol::validate_lowercase_sha256(
        &inventory_sha256,
        "SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256",
    )?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX25_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX25_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_subpath(
        &root,
        &repository,
        &revision,
        &[target.transformer_variant.as_str(), target.tier.as_str()],
        protocol::LTX_2_5_REPOSITORY,
    )?;
    let adapters = if target
        .transformer_variant
        .requires_official_refinement_lora()
    {
        vec![ltx25_official_dev_adapter(&revision)?]
    } else {
        Vec::new()
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(load_shape);
    if let Some(quant) = numeric_tier(&target.tier)?.quant {
        spec = spec.with_quant(quant);
    }
    if target.decoder == protocol::Ltx25Decoder::DiffVae {
        spec = spec.with_component(
            LTX25_DIFFUSION_VAE_COMPONENT,
            WeightsSource::File(root.join("vae_diffusion_decoder.safetensors")),
        );
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters.clone());
    }
    Ok(Ltx25LoadPlan {
        repository,
        revision,
        inventory_sha256,
        load_shape,
        adapters,
        spec,
    })
}

/// Execute a real selected LTX-2.5 provider path while leaving promotion decisions outside this
/// apparatus. The full published ladder — `q4`, `q8`, `bf16` — is executable here, symmetric with
/// the MLX arm. Candle's distinct NVFP4 evaluation selectors still need an inference-owned
/// producer and are never aliased to an ordinary bundle tier here.
fn run_ltx25_capture(request: &Value) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, LTX25_EXECUTION_PATH)?;
    let target = protocol::ltx25_candle_target(request)?;
    let Ltx25LoadPlan {
        repository,
        revision,
        inventory_sha256,
        load_shape,
        adapters,
        spec,
    } = ltx25_load_spec(request, &target)?;
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = VramProbe::start_rendered().assert_idle(1.0);
    let load_sample = vram.phase();
    let generator = catalog.media().load(LTX25_ID, &spec).map_err(|error| {
        format!(
            "load real {LTX25_ID} {}/{}/{} generator: {error}",
            target.transformer_variant.as_str(),
            target.decoder.as_str(),
            target.tier
        )
    })?;
    vram.end_load(load_sample);
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {LTX25_ID} has no memory-strategy contract"))?;
    let selection = planned_selection(request)?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {LTX25_ID} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| format!("loaded {LTX25_ID} has no calibration identity"))?;
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
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: target.width,
            height: target.height,
            batch: 1,
            frames: target.frames,
            reference_count: 0,
        },
        overlay: adapter_stack_identity(&adapters),
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-18783-adapter@{}", protocol::INFERENCE_PIN),
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin {LTX25_ID} capture scope: {error}"))?
        .ok_or_else(|| format!("{LTX25_ID} selection did not create a provider scope"))?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {LTX25_ID} decode: {error}"))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{LTX25_ID} decode edge and overlap must be selected together"
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope
            .configure_attention(attention)
            .map_err(|error| format!("configure {LTX25_ID} attention: {error}"))?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| format!("configure {LTX25_ID} transformer window: {error}"))?;
    }
    let mut generation = GenerationRequest {
        prompt: "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
            .to_owned(),
        width: target.width,
        height: target.height,
        count: 1,
        seed: Some(target.seed),
        steps: Some(target.transformer_variant.steps()),
        // The dev provider owns its fixed multimodal guider parameters; neither packed variant
        // advertises the generic request-level guidance axis.
        guidance: None,
        frames: Some(target.frames),
        fps: Some(target.fps),
        // The production default A/V T2V path leaves this unset. `Some("no_audio")` is the only
        // LTX T2V override; the evidence mode belongs in `MemoryRunContext`, not this provider knob.
        video_mode: None,
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {LTX25_ID} capture strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {LTX25_ID} conditioning: {error}"))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut peaks = [None, None, None];
    let mut phase_error = None;
    let result = generator.generate(&generation, &mut |progress| {
        if phase_error.is_some() {
            return;
        }
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let index = match phase {
            MemoryPhase::Conditioning => 0,
            MemoryPhase::Denoise => 1,
            MemoryPhase::Decode => 2,
        };
        peaks[index] = phase_sample.take().map(|sample| vram.end_observed(sample));
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {LTX25_ID} {phase:?}: {error}"));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {LTX25_ID} {next:?}: {error}"));
            return;
        }
        phase = next;
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let index = match phase {
            MemoryPhase::Conditioning => 0,
            MemoryPhase::Denoise => 1,
            MemoryPhase::Decode => 2,
        };
        peaks[index] = Some(vram.end_observed(sample));
    }
    vram.end_gen(generation_sample);
    let cumulative_run_peak_bytes = decimal_gb_to_bytes(vram.report().peak_gb);
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!("{LTX25_ID} generation failed: {message}"));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {LTX25_ID} terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {LTX25_ID} capture scope: {error}"))?;
    let (frames, fps, audio) = match output {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        GenerationOutput::Images(_) => {
            return Err(format!("{LTX25_ID} returned images, not a video clip"))
        }
        GenerationOutput::Audio(_) => {
            return Err(format!("{LTX25_ID} returned audio without video frames"))
        }
    };
    if fps != target.fps || audio.is_none() {
        return Err(format!(
            "{LTX25_ID} returned {fps} fps with audio={}, expected {} fps with audio",
            audio.is_some(),
            target.fps
        ));
    }
    let frame_shapes = frames
        .iter()
        .map(|frame| (frame.width, frame.height, frame.pixels.len()))
        .collect::<Vec<_>>();
    protocol::validate_ltx25_rgb_frames(
        usize::try_from(target.frames)
            .map_err(|_| "LTX-2.5 frame count does not fit usize".to_owned())?,
        target.width,
        target.height,
        &frame_shapes,
    )?;
    let conditioning_bytes = decimal_gb_to_bytes(
        peaks[0].ok_or_else(|| format!("{LTX25_ID} did not expose the conditioning boundary"))?,
    );
    let denoise_bytes = decimal_gb_to_bytes(
        peaks[1].ok_or_else(|| format!("{LTX25_ID} did not expose the denoise boundary"))?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        peaks[2].ok_or_else(|| format!("{LTX25_ID} did not complete decode sampling"))?,
    );
    let overall_bytes = protocol::validated_cumulative_peak(
        cumulative_run_peak_bytes,
        [conditioning_bytes, denoise_bytes, decode_bytes],
    )?;
    let blocker = concat!(
        "SC-18783 LTX-2.5 Candle capture measured the selected real provider path; promotion remains ",
        "gated on terminal CUDA repetition/quality evidence; advanced INT8-ConvRot/NVFP4 receipt production remains inference-owned"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        LTX25_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            // Artifact variant retains the manifest/download identity. Transformer and decoder are
            // independent target axes and are stamped into the final record from `planned.target`.
            artifact: {
                let mut artifact = artifact(&repository, &revision, &target.tier);
                artifact["inventorySha256"] = json!(inventory_sha256);
                artifact
            },
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": format!(
                    "{}:transformer={}:decoder={}:f{}:{}x{}:fps{}:seed{}",
                    loadability_fingerprint(&repository, &revision, &target.tier),
                    target.transformer_variant.as_str(),
                    target.decoder.as_str(),
                    target.frames,
                    target.width,
                    target.height,
                    target.fps,
                    target.seed,
                ),
            }),
            diagnostics: protocol::diagnostics(
                "memory-candle-adapter:ltx-2.5",
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("renderedFrames", "count", u64::from(target.frames)),
                    ("renderedFps", "fps", u64::from(target.fps)),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    Ok(fragment)
}

// ---------------------------------------------------------------------------------------------
// sc-22737 — the Bernini, LTX-2.3 and MiniMax-H3 Candle arms
//
// Three families, one shared capture. Each is a VIDEO route, so each dispatches ABOVE the shared
// still-geometry gate the way `run_ltx25_capture` does, and each validates against its own engine's
// geometry envelope instead. What differs between them is exactly two things — how the load spec is
// STAGED, and what the target's geometry envelope admits — so those are the two function-pointer
// fields of [`Sc22737VideoArm`], and everything else is shared.
//
// Deliberately its own table and its own capture rather than a generalisation of
// `run_ltx25_capture`: sc-22736 is adding the Wan/SCAIL-2 arms against the same precedent at the
// same time, and refactoring the shared dispatch would turn two additive changes into one conflict.
// ---------------------------------------------------------------------------------------------

/// The Bernini engine provider id — ONE id for BOTH SceneWorks catalog entries. `engines.rs` maps
/// the still entry `bernini_image` onto it and `video_jobs/bernini.rs` calls
/// `inference_runtime::load("bernini")` for the video entry, so the engine's `renderer` identity
/// route (`bernini_renderer`) is never reached from this repository on either lane.
const BERNINI_CANDLE_ID: &str = "bernini";
const BERNINI_CANDLE_VIDEO_MODEL_ID: &str = "bernini";
const BERNINI_CANDLE_IMAGE_MODEL_ID: &str = "bernini_image";
const BERNINI_CANDLE_VIDEO_EXECUTION_PATH: &str =
    "the Candle Bernini dual-expert text-to-video path";
const BERNINI_CANDLE_IMAGE_EXECUTION_PATH: &str = "the Candle Bernini still text-to-image path";
const BERNINI_CANDLE_STILL_CALIBRATION: &str = "Candle Bernini still calibration";
/// The single cadence the shipped `bernini` manifest entry publishes (`limits.fps: [16]`).
const BERNINI_CANDLE_FPS: u32 = 16;
/// 3 s at [`BERNINI_CANDLE_FPS`], coerced onto the Wan `1 mod 4` lattice the A14B renderer requires.
/// The manifest's shortest published duration, because this is the cell's ONE capture.
const BERNINI_CANDLE_FRAMES: u32 = 49;
/// One seed for every sc-22737 Candle fixture. The fixture binds the family, member, tier and full
/// geometry, so the seed does not also have to carry the route.
const SC22737_CANDLE_SEED: u64 = 22737;

/// The LTX-2.3 Candle engine id. DIFFERENT from the MLX one (`ltx_2_3`), the same way LTX-2.5's two
/// lanes differ: `candle-gen-ltx` registers the 2.3 route as `ltx_2_3_distilled`, which is what
/// `video_jobs/candle.rs#candle_video_engine_id` resolves and therefore what a candle plan row's
/// `provider` must name.
const LTX23_CANDLE_ID: &str = "ltx_2_3_distilled";
const LTX23_CANDLE_MODEL_ID: &str = "ltx_2_3";
const LTX23_CANDLE_EXECUTION_PATH: &str = "the Candle LTX-2.3 base text-to-video path";
/// `limits.requiresDimensionsMultipleOf` of the shipped `ltx_2_3` entry, mirroring the engine's
/// `SIZE_MULTIPLE = 2 * SPATIAL_SCALE`.
const LTX23_CANDLE_DIMENSION_MULTIPLE: u32 = 64;
/// `limits.resolutions` of the shipped `ltx_2_3` entry, verbatim.
const LTX23_CANDLE_RESOLUTIONS: [(u32, u32); 5] =
    [(768, 512), (512, 768), (640, 640), (1280, 704), (704, 1280)];
/// `limits.fps` of the shipped `ltx_2_3` entry, verbatim.
const LTX23_CANDLE_FPS: [u32; 3] = [24, 25, 30];
/// The LTX video VAE is 8x causal in time, so the engine's `validate_request` hard-rejects any
/// `num_frames` that is not `1 + 8k`.
const LTX23_CANDLE_TEMPORAL_SCALE: u32 = 8;
/// The frame count this lane's anchor renders: 4 s at 24 fps snapped onto the `1 + 8k` lattice by
/// the shipped ladder (`sceneworks_core::video_request::ltx_frame_count`), i.e. the SHORTEST cell of
/// the declared `durations x fps` cross product. The four constants above and this one are copies of
/// the manifest's `limits` block, bound to it by `the sc-22737 Candle video arms' manifest constants
/// match their shipped limits` in `scripts/platform-review-contracts.test.mjs` — this crate carries
/// two dependencies on purpose and cannot reach a JSONC reader at test time, so the binding lives on
/// the node side exactly as the MLX LTX arm's does.
const LTX23_CANDLE_FRAMES: u32 = 97;

/// The MiniMax-H3 Candle engine id — one id for both catalog entries, the same way MLX has one.
const MINIMAX_CANDLE_ID: &str = "minimax_h3";
const MINIMAX_CANDLE_BASE_MODEL_ID: &str = "minimax_h3";
const MINIMAX_CANDLE_REFERENCE_MODEL_ID: &str = "minimax_h3_ref";
const MINIMAX_CANDLE_BASE_EXECUTION_PATH: &str =
    "the Candle MiniMax-H3 joint audio+video text-to-video path";
const MINIMAX_CANDLE_REFERENCE_EXECUTION_PATH: &str =
    "the Candle MiniMax-H3 joint audio+video reference-to-video path";
/// The released checkpoint generates at 24 fps and nothing else.
const MINIMAX_CANDLE_FPS: u32 = 24;
/// The shortest cell of the `17n + 5` lattice the released checkpoint admits.
const MINIMAX_CANDLE_FRAMES: u32 = 124;
/// The manifest's `limits.hardMinSteps` for both MiniMax entries.
const MINIMAX_CANDLE_STEPS: u32 = 2;

/// Everything one sc-22737 Candle capture STAGED, in the shape the record needs it.
struct Sc22737LoadPlan {
    /// The `artifact` block of the emitted fragment — a family may name more than one triple.
    artifact: Value,
    /// The `loadability.resolvedPathFingerprint` stem. Two records that opened different trees must
    /// be distinguishable from this string alone.
    resolved_path_fingerprint: String,
    spec: LoadSpec,
}

/// One `(catalog entry, engine provider)` cell this block measures.
#[derive(Clone, Copy)]
struct Sc22737VideoArm {
    /// The id handed to `catalog.media().load`.
    engine_id: &'static str,
    /// The SceneWorks catalog entry the record is filed under.
    model_id: &'static str,
    execution_path: &'static str,
    /// `<fixture_prefix>-<tier>-<w>x<h>-f<frames>-fps<fps>-seed<seed>`.
    fixture_prefix: &'static str,
    /// The `MemoryMode` key the runtime asks admission under. An EVIDENCE KEY, not a label:
    /// gen-core matches it against each adopted decode-geometry record's own mode, so a probe run
    /// under one spelling cannot answer a request asked under another.
    mode: &'static str,
    /// How many references the measured request carries. Non-zero selects a different conditioning
    /// — and, for MiniMax-H3, a different DiT partition.
    reference_count: u32,
    /// The cadences the family's manifest entry publishes. One value for the families whose engine
    /// admits exactly one.
    legal_fps: &'static [u32],
    /// The clip this arm renders. Fixed per family so the measured render and the record describe
    /// the same work.
    frames: u32,
    /// Whether the family denoises a soundtrack jointly with the video. A joint A/V record that did
    /// not observe an audio track is not a record of the render it claims.
    requires_audio: bool,
    prompt: &'static str,
    /// The label the emitted `diagnostics.source` carries.
    diagnostics_lane: &'static str,
    /// The family's geometry envelope, read off the pinned engine crate.
    validate_geometry: fn(u32, u32, u32) -> Result<(), String>,
    /// How the family stages its load. The one thing that genuinely differs between the three.
    load_plan: fn(&Value, &Sc22737VideoTarget) -> Result<Sc22737LoadPlan, String>,
}

/// The exact cell a plan row asks for, after the target and the fixture have been reconciled.
struct Sc22737VideoTarget {
    tier: String,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    seed: u64,
}

const BERNINI_CANDLE_VIDEO_ARM: Sc22737VideoArm = Sc22737VideoArm {
    engine_id: BERNINI_CANDLE_ID,
    model_id: BERNINI_CANDLE_VIDEO_MODEL_ID,
    execution_path: BERNINI_CANDLE_VIDEO_EXECUTION_PATH,
    fixture_prefix: "bernini-video-candle",
    mode: "text_to_video",
    reference_count: 0,
    legal_fps: &[BERNINI_CANDLE_FPS],
    frames: BERNINI_CANDLE_FRAMES,
    requires_audio: false,
    prompt: "a slow crane over a terracotta rooftop at golden hour, swallows turning, cinematic",
    diagnostics_lane: "memory-candle-adapter:bernini-dual-expert",
    validate_geometry: validate_bernini_candle_geometry,
    load_plan: bernini_candle_load_plan,
};

const LTX23_CANDLE_ARM: Sc22737VideoArm = Sc22737VideoArm {
    engine_id: LTX23_CANDLE_ID,
    model_id: LTX23_CANDLE_MODEL_ID,
    execution_path: LTX23_CANDLE_EXECUTION_PATH,
    fixture_prefix: "ltx-2-3-candle",
    mode: "text_to_video",
    reference_count: 0,
    legal_fps: &LTX23_CANDLE_FPS,
    frames: LTX23_CANDLE_FRAMES,
    requires_audio: false,
    prompt: "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic",
    diagnostics_lane: "memory-candle-adapter:ltx-2.3",
    validate_geometry: validate_ltx23_candle_geometry,
    load_plan: ltx23_candle_load_plan,
};

const MINIMAX_CANDLE_BASE_ARM: Sc22737VideoArm = Sc22737VideoArm {
    engine_id: MINIMAX_CANDLE_ID,
    model_id: MINIMAX_CANDLE_BASE_MODEL_ID,
    execution_path: MINIMAX_CANDLE_BASE_EXECUTION_PATH,
    fixture_prefix: "minimax-h3-candle",
    mode: "text_to_video",
    reference_count: 0,
    legal_fps: &[MINIMAX_CANDLE_FPS],
    frames: MINIMAX_CANDLE_FRAMES,
    requires_audio: true,
    prompt: "a slow dolly along a rain-slick harbour wall at dusk, gulls calling, cinematic",
    diagnostics_lane: "memory-candle-adapter:minimax-h3-joint-av",
    validate_geometry: validate_minimax_candle_geometry,
    load_plan: minimax_candle_load_plan,
};

/// The reference partition. ONE image reference — the smallest set the shared verdict
/// (`sceneworks_core::video_request::classify_reference_set`) calls `Conditionable`: an empty set
/// would resolve t2va on the BASE checkpoint, which is not the one this entry loads, and an
/// audio-only set never reaches the reference conditioner and is refused by the worker, the API and
/// the MCP tool alike.
const MINIMAX_CANDLE_REFERENCE_ARM: Sc22737VideoArm = Sc22737VideoArm {
    engine_id: MINIMAX_CANDLE_ID,
    model_id: MINIMAX_CANDLE_REFERENCE_MODEL_ID,
    execution_path: MINIMAX_CANDLE_REFERENCE_EXECUTION_PATH,
    fixture_prefix: "minimax-h3-ref-candle",
    mode: "reference_to_video",
    reference_count: 1,
    legal_fps: &[MINIMAX_CANDLE_FPS],
    frames: MINIMAX_CANDLE_FRAMES,
    requires_audio: true,
    prompt: "a slow dolly along a rain-slick harbour wall at dusk, gulls calling, cinematic",
    diagnostics_lane: "memory-candle-adapter:minimax-h3-joint-av",
    validate_geometry: validate_minimax_candle_geometry,
    load_plan: minimax_candle_load_plan,
};

/// Every cell this block serves, in one place so the tables and the tests cannot disagree.
const SC22737_VIDEO_ARMS: [Sc22737VideoArm; 4] = [
    BERNINI_CANDLE_VIDEO_ARM,
    LTX23_CANDLE_ARM,
    MINIMAX_CANDLE_BASE_ARM,
    MINIMAX_CANDLE_REFERENCE_ARM,
];

/// Which cell the plan asks for, resolved from `(target.provider, target.modelId)`.
///
/// `None` — not an error — when the plan names a provider this block does not serve, so the caller
/// falls through to the arms below it. A provider this block DOES serve with an unknown model id is
/// an error: one engine id per family serves several catalog entries whose records are filed
/// separately, and nothing else distinguishes them.
fn sc22737_video_arm(request: &Value) -> Result<Option<Sc22737VideoArm>, String> {
    let provider = planned_provider(request)?;
    if !SC22737_VIDEO_ARMS
        .iter()
        .any(|arm| arm.engine_id == provider)
    {
        return Ok(None);
    }
    let model_id = protocol::planned(request)?
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    // The still Bernini entry rides the shared five-rung reference arm below, not this one: it is an
    // image route with an image geometry, and `run_five_rung_reference` already implements exactly
    // that shape. Reported as "no video arm" so the caller falls through rather than erroring.
    if provider == BERNINI_CANDLE_ID && model_id == BERNINI_CANDLE_IMAGE_MODEL_ID {
        return Ok(None);
    }
    SC22737_VIDEO_ARMS
        .iter()
        .copied()
        .find(|arm| arm.engine_id == provider && arm.model_id == model_id)
        .map(Some)
        .ok_or_else(|| {
            format!(
                "the Candle {provider} arm serves catalog entries {:?}, got modelId {model_id:?}",
                SC22737_VIDEO_ARMS
                    .iter()
                    .filter(|arm| arm.engine_id == provider)
                    .map(|arm| arm.model_id)
                    .collect::<Vec<_>>()
            )
        })
}

/// Bernini's own geometry envelope: the engine's `validate_geometry`, which is an exact membership
/// test against `ADVERTISED_GEOMETRIES` and is PROVIDER-WIDE — `candle-gen-bernini` applies it to
/// the still route and the video route alike. Read off the crate rather than transcribed.
fn validate_bernini_candle_geometry(width: u32, height: u32, frames: u32) -> Result<(), String> {
    let advertised = runtime_cuda::providers::bernini::memory_strategy::ADVERTISED_GEOMETRIES;
    if !advertised.contains(&(width, height)) {
        return Err(format!(
            "Candle Bernini memory evidence requires one of the advertised geometries \
             {advertised:?}, got {width}x{height}"
        ));
    }
    // The renderer is Wan2.2-A14B, whose frame count is `1 mod 4` (`video_jobs/wan.rs`'s
    // `wan_frame_count`, which the Bernini video path calls for exactly that reason).
    if frames % 4 != 1 || frames < 5 {
        return Err(format!(
            "Candle Bernini requires geometry.frames on the Wan 1 mod 4 lattice (>= 5), got {frames}"
        ));
    }
    Ok(())
}

/// LTX-2.3's own geometry envelope, the same three constraints the MLX arm applies: the declared
/// `limits.resolutions`, `limits.requiresDimensionsMultipleOf`, and the `1 + 8k` temporal lattice
/// the engine's `validate_request` hard-rejects outside of. A still geometry is on the lattice but
/// below the floor, so it is refused here too — this arm may not capture a single-frame record for
/// a video model.
fn validate_ltx23_candle_geometry(width: u32, height: u32, frames: u32) -> Result<(), String> {
    if !LTX23_CANDLE_RESOLUTIONS.contains(&(width, height)) {
        return Err(format!(
            "Candle LTX-2.3 calibration requires one of the declared limits.resolutions \
             {LTX23_CANDLE_RESOLUTIONS:?}, got {width}x{height}"
        ));
    }
    if !width.is_multiple_of(LTX23_CANDLE_DIMENSION_MULTIPLE)
        || !height.is_multiple_of(LTX23_CANDLE_DIMENSION_MULTIPLE)
    {
        return Err(format!(
            "Candle LTX-2.3 calibration requires geometry divisible by \
             {LTX23_CANDLE_DIMENSION_MULTIPLE}, got {width}x{height}"
        ));
    }
    if frames % LTX23_CANDLE_TEMPORAL_SCALE != 1 || frames < 9 {
        return Err(format!(
            "Candle LTX-2.3 calibration requires geometry.frames == 1 + \
             {LTX23_CANDLE_TEMPORAL_SCALE}k (>= 9; the LTX video VAE is \
             {LTX23_CANDLE_TEMPORAL_SCALE}x causal in time), got {frames}"
        ));
    }
    Ok(())
}

/// MiniMax-H3's own geometry envelope, read off the pinned Candle crate: the `17n + 5` temporal
/// lattice, the spatial stride, and the canvas budget as a PRODUCT rather than per edge — the
/// published resolution list contains 1536x672 and 1344x768, whose long edges differ by 192 px and
/// whose areas are identical, so a per-edge cap would refuse the first and admit the second while
/// both sit exactly at the budget.
fn validate_minimax_candle_geometry(width: u32, height: u32, frames: u32) -> Result<(), String> {
    let lattice = runtime_cuda::providers::minimax_h3::denoise::geometry::LEGAL_FRAME_COUNTS;
    let frames_usize =
        usize::try_from(frames).map_err(|_| "MiniMax-H3 frame count must fit usize".to_owned())?;
    if !lattice.contains(&frames_usize) {
        return Err(format!(
            "Candle MiniMax-H3 requires geometry.frames on the 17n+5 lattice {lattice:?}, got \
             {frames}"
        ));
    }
    let stride = runtime_cuda::providers::minimax_h3::pipeline::SPATIAL_STRIDE;
    if !width.is_multiple_of(stride) || !height.is_multiple_of(stride) {
        return Err(format!(
            "Candle MiniMax-H3 requires geometry divisible by the {stride}px stride, got \
             {width}x{height}"
        ));
    }
    let budget = runtime_cuda::providers::minimax_h3::pipeline::CANVAS_MAX_PIXELS;
    let pixels = width.saturating_mul(height);
    if pixels > budget {
        return Err(format!(
            "Candle MiniMax-H3 requires width*height within the {budget}px canvas budget, got \
             {width}x{height} ({pixels}px)"
        ));
    }
    Ok(())
}

/// Bernini's Candle staging: ONE tier directory inside the off-Mac bundle.
///
/// `SceneWorks/bernini` is the only download either Bernini entry ships for
/// `platforms: ["windows", "linux"]`, and it carries the three tier sub-directories INSIDE it —
/// which is why `memory_route_registry.rs` declares `BF16_Q4_Q8` for `candle:bernini` while the
/// download row names no `variant` at all. `quantize` stays unset: the tiers ship pre-packed, so the
/// worker's own resolver returns no load-time quant for a resolved tier, and `production_assets`
/// reconciles the packing against the transformer's own `quantize_config.json` before the identity
/// is minted.
fn bernini_candle_load_plan(
    _request: &Value,
    target: &Sc22737VideoTarget,
) -> Result<Sc22737LoadPlan, String> {
    let repository = protocol::required_env("SCENEWORKS_BERNINI_CANDLE_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_BERNINI_CANDLE_REVISION")?;
    protocol::validate_artifact_identity(
        &repository,
        &revision,
        protocol::BERNINI_CANDLE_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_BERNINI_CANDLE_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_BERNINI_CANDLE_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        &target.tier,
        protocol::BERNINI_CANDLE_REPOSITORY,
    )?;
    Ok(Sc22737LoadPlan {
        artifact: artifact(&repository, &revision, &target.tier),
        resolved_path_fingerprint: loadability_fingerprint(&repository, &revision, &target.tier),
        spec: LoadSpec::new(WeightsSource::Dir(root))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_resolved_route(BERNINI_CANDLE_VIDEO_MODEL_ID.to_owned()),
    })
}

/// LTX-2.3's Candle staging: the packed tier sub-directory plus the separate Gemma-3-12B encoder.
///
/// TWO roots under ONE repository, exactly as the MLX arm resolves them. The encoder is a hard
/// load-time requirement — `video_jobs/candle.rs` resolves `ltx_gemma_dir` for every non-2.5 LTX
/// load and the provider surfaces a required-`text_encoder` error without it — so it is threaded
/// through `LoadSpec::text_encoder` and snapshot-validated with the same identity check as the tier
/// root: a mismatched TE would silently change the measured conditioning peak.
///
/// `quantize` stays unset because the worker's own resolver leaves it unset:
/// `candle_ltx_tier_subdir` returns `(dir, None)` for LTX-2.3 ("LTX-2.3 keeps its legacy `None`
/// marker"), and the tier is carried by the directory the load opens.
fn ltx23_candle_load_plan(
    _request: &Value,
    target: &Sc22737VideoTarget,
) -> Result<Sc22737LoadPlan, String> {
    let repository = protocol::required_env("SCENEWORKS_LTX_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_LTX_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::LTX_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        &target.tier,
        protocol::LTX_REPOSITORY,
    )?;
    let text_encoder = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX_TEXT_ENCODER_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX_TEXT_ENCODER_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &text_encoder,
        &repository,
        &revision,
        "gemma",
        protocol::LTX_REPOSITORY,
    )?;
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    spec.text_encoder = Some(WeightsSource::Dir(text_encoder));
    Ok(Sc22737LoadPlan {
        artifact: artifact(&repository, &revision, &target.tier),
        resolved_path_fingerprint: format!(
            "{}:gemma",
            loadability_fingerprint(&repository, &revision, &target.tier)
        ),
        spec,
    })
}

/// MiniMax-H3's Candle staging: the UPSTREAM snapshot root, with the packed components redirected
/// onto the tiered rehost.
///
/// The direction is the opposite of what the repository names suggest, and it is the worker's:
/// `resolve_candle_minimax_h3_load` makes `MiniMaxAI/MiniMax-H3` the load ROOT on every tier — it is
/// the only tree carrying `vae/`, `audio_vae/`, `tokenizer/` and the `FL2VA/` documents — and stages
/// `transformer/` and `text_encoder/` out of `SceneWorks/minimax-h3-mlx/<tier>` when the tier is
/// packed. `bf16` on the BASE entry stages neither, because the dense partitions live in the
/// upstream root already; `minimax_h3_ref` stages the tier tree at every tier, because the reference
/// partition is only published in the rehost.
///
/// The `transformer` component is ALWAYS the base partition, even for the reference entry: the
/// provider derives `transformer_ref/` as its sibling, so the worker validates both halves and
/// stages only the base. This arm does the same, and its `resolved_path_fingerprint` names the
/// partition the CONDITIONING will select so two records over one triple stay distinguishable.
fn minimax_candle_load_plan(
    request: &Value,
    target: &Sc22737VideoTarget,
) -> Result<Sc22737LoadPlan, String> {
    let arm = sc22737_video_arm(request)?
        .ok_or_else(|| "the MiniMax-H3 Candle plan resolved no member".to_owned())?;
    let upstream_repository = protocol::required_env("SCENEWORKS_MINIMAX_H3_UPSTREAM_REPOSITORY")?;
    let upstream_revision = protocol::required_env("SCENEWORKS_MINIMAX_H3_UPSTREAM_REVISION")?;
    protocol::validate_artifact_identity(
        &upstream_repository,
        &upstream_revision,
        protocol::MINIMAX_UPSTREAM_REPOSITORY,
    )?;
    let upstream_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_MINIMAX_H3_UPSTREAM_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_MINIMAX_H3_UPSTREAM_ROOT: {error}"))?;
    protocol::validate_huggingface_revision_root(
        &upstream_root,
        &upstream_repository,
        &upstream_revision,
        protocol::MINIMAX_UPSTREAM_REPOSITORY,
    )?;

    let mut spec = LoadSpec::new(WeightsSource::Dir(upstream_root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    let quant = numeric_tier(&target.tier)?.quant;
    let is_reference = arm.model_id == MINIMAX_CANDLE_REFERENCE_MODEL_ID;
    let staged = if quant.is_some() || is_reference {
        let repository = protocol::required_env("SCENEWORKS_MINIMAX_H3_REPOSITORY")?;
        let revision = protocol::required_env("SCENEWORKS_MINIMAX_H3_REVISION")?;
        protocol::validate_artifact_identity(&repository, &revision, protocol::MINIMAX_REPOSITORY)?;
        let tier_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
            "SCENEWORKS_MINIMAX_H3_ROOT",
        )?))
        .map_err(|error| format!("canonicalize SCENEWORKS_MINIMAX_H3_ROOT: {error}"))?;
        protocol::validate_huggingface_snapshot_root(
            &tier_root,
            &repository,
            &revision,
            &target.tier,
            protocol::MINIMAX_REPOSITORY,
        )?;
        spec = spec.with_component(
            "transformer",
            WeightsSource::Dir(tier_root.join("transformer")),
        );
        // The packed text encoder ships beside the packed DiT; the dense one lives upstream and the
        // loader resolves it from the spec's own weights root, so it is not redirected.
        if quant.is_some() {
            spec = spec.with_component(
                "text_encoder",
                WeightsSource::Dir(tier_root.join("text_encoder")),
            );
        }
        Some((repository, revision))
    } else {
        None
    };
    if let Some(quant) = quant {
        // Never a load-time pack: the loader RECONCILES `spec.quantize` against the staged tier's
        // own marker and refuses a disagreement, so passing it is an assertion about the directory
        // on disk rather than an instruction.
        spec = spec.with_quant(quant);
    }
    let partition = if is_reference {
        "transformer_ref"
    } else {
        "transformer"
    };
    let tier_artifact = staged
        .as_ref()
        .map(|(repository, revision)| artifact(repository, revision, &target.tier));
    Ok(Sc22737LoadPlan {
        artifact: json!({
            "repository": upstream_repository,
            "resolvedRevision": upstream_revision,
            "variant": target.tier,
            "stagedTierArtifact": tier_artifact,
        }),
        resolved_path_fingerprint: format!(
            "{upstream_repository}@{upstream_revision}:{}+partition:{partition}+staged:{}",
            target.tier,
            staged
                .as_ref()
                .map(|(repository, revision)| format!("{repository}@{revision}/{}", target.tier))
                .unwrap_or_else(|| "upstream-dense".to_owned()),
        ),
        spec,
    })
}

/// Reconcile the plan's target with its fixture into the exact cell this capture renders.
///
/// The fixture carries the cadence and the seed, which `planned.target.geometry` has no axis for
/// (`GeometryEnvelope` has no temporal-cadence field), and it repeats the tier and every geometry
/// axis so a bf16 record can never be emitted against a q4 capture that merely reused the string.
fn sc22737_video_target(
    request: &Value,
    arm: Sc22737VideoArm,
) -> Result<Sc22737VideoTarget, String> {
    let planned = protocol::planned(request)?;
    let target = planned
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != arm.mode {
        return Err(format!(
            "{} is measured in {:?} mode, got {mode:?}",
            arm.model_id, arm.mode
        ));
    }
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(u64::from(arm.reference_count)) {
                return Err(format!(
                    "{} requires {field} == {}; a record measured with a different reference set \
                     describes a different conditioning and, for MiniMax-H3, a different DiT \
                     partition",
                    arm.model_id, arm.reference_count
                ));
            }
        }
    }
    let tier = planned_tier(request)?.to_owned();
    let geometry = target
        .get("geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let axis = |name: &str| {
        geometry
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("planned.target.geometry.{name} must fit u32"))
    };
    let batch = axis("batch")?;
    if batch != 1 {
        return Err(format!(
            "{} renders one clip per request; got geometry.batch {batch}",
            arm.model_id
        ));
    }
    let width = axis("width")?;
    let height = axis("height")?;
    let frames = axis("frames")?;
    if frames != arm.frames {
        return Err(format!(
            "{} is measured at {} frames, got {frames}",
            arm.model_id, arm.frames
        ));
    }
    (arm.validate_geometry)(width, height, frames)?;
    let fixture = planned
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!(
        "{}-{tier}-{width}x{height}-f{frames}-fps",
        arm.fixture_prefix
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse {} fixture fps {fps:?}: {error}", arm.model_id))?;
    if !arm.legal_fps.contains(&fps) {
        return Err(format!(
            "planned.fixture declares fps {fps}, but {} publishes {:?} only",
            arm.model_id, arm.legal_fps
        ));
    }
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse {} fixture seed {seed:?}: {error}", arm.model_id))?;
    if seed != SC22737_CANDLE_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the sc-22737 Candle calibration seed \
             {SC22737_CANDLE_SEED}"
        ));
    }
    Ok(Sc22737VideoTarget {
        tier,
        width,
        height,
        frames,
        fps,
        seed,
    })
}

/// The measured request. The CONDITIONING is what selects a reference route — and, for MiniMax-H3,
/// which DiT partition the engine resolves — so the reference member carries exactly one synthetic
/// image reference at the target geometry and every other member carries none.
fn sc22737_generation_request(
    arm: Sc22737VideoArm,
    target: &Sc22737VideoTarget,
) -> GenerationRequest {
    let mut request = GenerationRequest {
        prompt: arm.prompt.to_owned(),
        width: target.width,
        height: target.height,
        count: 1,
        seed: Some(target.seed),
        frames: Some(target.frames),
        fps: Some(target.fps),
        // Left unset for the two families whose engine owns the recipe; MiniMax-H3 renders at the
        // manifest's `limits.hardMinSteps`, which is the cheapest admissible cell of the one
        // capture this anchor is allowed.
        steps: (arm.engine_id == MINIMAX_CANDLE_ID).then_some(MINIMAX_CANDLE_STEPS),
        ..Default::default()
    };
    if arm.reference_count > 0 {
        request.conditioning = vec![Conditioning::Reference {
            image: Image {
                width: target.width,
                height: target.height,
                pixels: protocol::synthetic_reference_rgb(target.width, target.height),
            },
            // The engine owns the reference conditioning strength; the request-level lever stays
            // unset, exactly as the worker's own conditioning resolver leaves it.
            strength: None,
        }];
    }
    request
}

/// Execute one sc-22737 Candle video cell: stage the family's load, prove the plan and the loaded
/// contract name the same calibration identity, then measure three phase peaks off the boundaries
/// the shipped `generate` already emits.
///
/// The shape is `run_ltx25_capture`'s, which is the established Candle video precedent: a real
/// selected provider path, a memory-strategy request scope around the render, and the
/// frames/fps/audio result validation that keeps a record from claiming a clip the engine did not
/// produce.
fn run_sc22737_video_capture(request: &Value, arm: Sc22737VideoArm) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let target = sc22737_video_target(request, arm)?;
    let Sc22737LoadPlan {
        artifact,
        resolved_path_fingerprint,
        spec,
    } = (arm.load_plan)(request, &target)?;
    let load_shape = spec.load_shape;
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(arm.engine_id, &spec)
        .map_err(|error| {
            format!(
                "load real {} {} {} generator: {error}",
                arm.engine_id, arm.model_id, target.tier
            )
        })?;
    vram.end_load(load_sample);
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {} has no memory-strategy contract", arm.engine_id))?;
    let selection = planned_selection(request)?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} provider rejected planned selection: {error}",
            arm.engine_id
        )
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "loaded {} has no calibration identity; the cell cannot be recorded",
            arm.engine_id
        )
    })?;
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
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other(arm.mode.to_owned()),
        has_reference: arm.reference_count > 0,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: target.width,
            height: target.height,
            batch: 1,
            frames: target.frames,
            reference_count: arm.reference_count,
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
        evidence_revision: format!("sc-22737-adapter@{}", protocol::INFERENCE_PIN),
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin {} capture scope: {error}", arm.engine_id))?
        .ok_or_else(|| {
            format!(
                "{} selection did not create a provider scope",
                arm.engine_id
            )
        })?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {} decode: {error}", arm.engine_id))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{} decode edge and overlap must be selected together",
                arm.engine_id
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope
            .configure_attention(attention)
            .map_err(|error| format!("configure {} attention: {error}", arm.engine_id))?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| format!("configure {} transformer window: {error}", arm.engine_id))?;
    }
    let mut generation = sc22737_generation_request(arm, &target);
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {} capture strategy: {error}", arm.engine_id))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {} conditioning: {error}", arm.engine_id))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut peaks = [None, None, None];
    let mut phase_error = None;
    let result = generator.generate(&generation, &mut |progress| {
        if phase_error.is_some() {
            return;
        }
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let index = match phase {
            MemoryPhase::Conditioning => 0,
            MemoryPhase::Denoise => 1,
            MemoryPhase::Decode => 2,
        };
        peaks[index] = phase_sample.take().map(|sample| vram.end_observed(sample));
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {} {phase:?}: {error}", arm.engine_id));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {} {next:?}: {error}", arm.engine_id));
            return;
        }
        phase = next;
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let index = match phase {
            MemoryPhase::Conditioning => 0,
            MemoryPhase::Denoise => 1,
            MemoryPhase::Decode => 2,
        };
        peaks[index] = Some(vram.end_observed(sample));
    }
    vram.end_gen(generation_sample);
    let cumulative_run_peak_bytes = decimal_gb_to_bytes(vram.report().peak_gb);
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!("{} generation failed: {message}", arm.engine_id));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {} terminal phase: {error}", arm.engine_id))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {} capture scope: {error}", arm.engine_id))?;
    let (frames, fps, audio) = match output {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        GenerationOutput::Images(_) => {
            return Err(format!(
                "{} returned images, not a video clip",
                arm.model_id
            ))
        }
        GenerationOutput::Audio(_) => {
            return Err(format!(
                "{} returned audio without video frames",
                arm.model_id
            ))
        }
    };
    if fps != target.fps {
        return Err(format!(
            "{} returned {fps} fps for a {} fps request",
            arm.model_id, target.fps
        ));
    }
    // The soundtrack is half of what a joint A/V family denoises; a record that did not observe one
    // is not a record of the render it claims.
    let audio = audio
        .filter(|track| !track.samples.is_empty() && track.sample_rate > 0 && track.channels > 0);
    if arm.requires_audio && audio.is_none() {
        return Err(format!(
            "{} render returned no non-empty audio track, but this family denoises video and audio \
             jointly",
            arm.model_id
        ));
    }
    let rendered = u32::try_from(frames.len())
        .map_err(|_| format!("{} frame count does not fit u32", arm.model_id))?;
    if rendered != target.frames {
        return Err(format!(
            "{} rendered {rendered} frames for a {}-frame request",
            arm.model_id, target.frames
        ));
    }
    let frame_shapes = frames
        .iter()
        .map(|frame| (frame.width, frame.height, frame.pixels.len()))
        .collect::<Vec<_>>();
    protocol::validate_ltx25_rgb_frames(
        usize::try_from(target.frames)
            .map_err(|_| format!("{} frame count does not fit usize", arm.model_id))?,
        target.width,
        target.height,
        &frame_shapes,
    )?;
    let conditioning_bytes =
        decimal_gb_to_bytes(peaks[0].ok_or_else(|| {
            format!("{} did not expose the conditioning boundary", arm.engine_id)
        })?);
    let denoise_bytes = decimal_gb_to_bytes(
        peaks[1].ok_or_else(|| format!("{} did not expose the denoise boundary", arm.engine_id))?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        peaks[2].ok_or_else(|| format!("{} did not complete decode sampling", arm.engine_id))?,
    );
    let overall_bytes = protocol::validated_cumulative_peak(
        cumulative_run_peak_bytes,
        [conditioning_bytes, denoise_bytes, decode_bytes],
    )?;
    let blocker = concat!(
        "sc-22737 capture measured the selected real provider path on this cell; promotion remains ",
        "gated on terminal CUDA repetition/quality evidence, which this arm does not produce"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        arm.execution_path,
        protocol::PlainGatedFragment {
            artifact,
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": format!(
                    "{resolved_path_fingerprint}:{}:f{}:{}x{}:fps{}:seed{}",
                    arm.model_id,
                    target.frames,
                    target.width,
                    target.height,
                    target.fps,
                    target.seed,
                ),
            }),
            diagnostics: protocol::diagnostics(
                arm.diagnostics_lane,
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("renderedFrames", "count", u64::from(rendered)),
                    ("renderedFps", "fps", u64::from(fps)),
                    ("renderedAudio", "count", u64::from(audio.is_some())),
                    ("referenceCount", "count", u64::from(arm.reference_count)),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    Ok(fragment)
}

/// The fixture prefix that marks a plan row as a five-rung reference capture.
const FIVE_RUNG_FIXTURE_PREFIX: &str = "fresh-five-rung-";

/// Which of [`run`]'s two branches a plan row takes: the five-rung reference path, or the inline
/// Krea arm.
///
/// Named rather than inlined so the decision is testable on its own (sc-18808 re-review). It is what
/// determines which arm [`run`]'s geometry guard is standing in front of, and every case in the
/// original regression table happened to answer `true` — so the inline arm, and with it `run`'s own
/// guard, went unexercised while the redundant copy at the head of [`run_five_rung_reference`]
/// produced the byte-identical message. Five shipped Candle plan rows answer `false`
/// (`krea-q4-1024-seed42` and its q8/bf16/768/v2 siblings).
fn routes_to_five_rung_reference(request: &Value) -> Result<bool, String> {
    let is_five_rung_fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .is_some_and(|fixture| fixture.starts_with(FIVE_RUNG_FIXTURE_PREFIX));
    // Qwen, Z-Image-Turbo and the Z-Image base have no inline arm at all, so every fixture on
    // them is a five-rung reference capture regardless of its spelling.
    let provider = planned_provider(request)?;
    // sc-22726: the FLUX.1 BASE providers have no inline arm either. PuLID is deliberately absent —
    // it is a bespoke route with its own arm, dispatched before this is ever consulted.
    // sc-22733: the six Mage routes have no inline arm either — they ride the shared five-rung
    // lifecycle behind their own loader (`load_mage_generator`) and their own request shape.
    // sc-22730: the three SD3.5 base providers have no inline arm either.
    Ok(is_five_rung_fixture
        || provider == QWEN_ID
        || provider == Z_IMAGE_TURBO_ID
        || provider == Z_IMAGE_ID
        // sc-22727: neither FLUX.2 provider has an inline arm on this adapter either.
        || provider == FLUX2_DEV_ID
        || provider == FLUX2_KLEIN_ID
        || provider == FLUX1_DEV_ID
        || provider == FLUX1_SCHNELL_ID
        || MAGE_ARMS.iter().any(|arm| arm.provider == provider)
        || provider == SD3_5_LARGE_ID
        || provider == SD3_5_LARGE_TURBO_ID
        || provider == SD3_5_MEDIUM_ID
        // sc-22737: the STILL Bernini entry has no inline arm either. The VIDEO entry never
        // reaches this function — `sc22737_video_arm` dispatches it above the still gate — so
        // naming the shared engine provider id here claims only the still member.
        || provider == BERNINI_CANDLE_ID)
}

// ---------------------------------------------------------------------------------------------
// Qwen-Image-Edit-2511 (sc-22728) — the bespoke Candle edit provider
// ---------------------------------------------------------------------------------------------

/// One member of the Qwen edit family this arm measures, resolved from the plan's
/// `(target.provider, target.modelId)`. Both members load the SAME engine provider from the SAME
/// artifact family; the Lightning member additionally stacks the built-in distill LoRA and runs the
/// CFG-off few-step recipe that LoRA was distilled for. Mirrors `QwenEditArm` on the MLX adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QwenEditArm {
    model_id: &'static str,
    execution_path: &'static str,
    still_calibration: &'static str,
    fixture_prefix: &'static str,
    overlay: &'static str,
    lightning: bool,
    steps: usize,
    slug: &'static str,
}

const QWEN_EDIT_ARM: QwenEditArm = QwenEditArm {
    model_id: "qwen_image_edit_2511",
    execution_path: "the Candle Qwen-Image-Edit-2511 reference-conditioned edit path",
    still_calibration: "Candle Qwen edit calibration",
    fixture_prefix: "qwen-edit-candle",
    overlay: "none",
    lightning: false,
    steps: 2,
    slug: "qwen-edit",
};

const QWEN_EDIT_LIGHTNING_ARM: QwenEditArm = QwenEditArm {
    model_id: "qwen_image_edit_2511_lightning",
    execution_path:
        "the Candle Qwen-Image-Edit-2511 Lightning distill reference-conditioned edit path",
    still_calibration: "Candle Qwen edit Lightning calibration",
    fixture_prefix: "qwen-edit-lightning-candle",
    overlay: "lora",
    lightning: true,
    // The official lightx2v 4-step recipe (`pipeline::lightning_sigmas`) and the worker's default.
    steps: 4,
    slug: "qwen-edit-lightning",
};

/// The engine provider id both catalog ids load (`candle-gen-qwen-image` `edit.rs`; the worker's
/// `QWEN_EDIT_PROVIDER_ID`). It is NOT a registered generator: the edit provider is bespoke and the
/// worker drives it by name, which is why this arm cannot ride `load_five_rung_generator`.
const QWEN_EDIT_ID: &str = "qwen_image_edit";
/// The edit prompt every Qwen edit capture renders. Fixed with the seed and the reference so two
/// captures of one anchor are the same request.
const QWEN_EDIT_PROMPT: &str = "replace the background with a plain grey studio backdrop";
/// The production true-CFG guidance the worker resolves for the multi-step edit path
/// (`resolve_qwen_edit_guidance`, manifest `variationStrength.default`). Ignored on the Lightning
/// path, which the engine forces CFG-off.
const QWEN_EDIT_GUIDANCE: f32 = 4.0;

fn qwen_edit_arm(request: &Value) -> Result<QwenEditArm, String> {
    let planned = protocol::planned(request)?;
    let provider = planned_provider(request)?;
    let model_id = planned
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    match (provider, model_id) {
        (QWEN_EDIT_ID, id) if id == QWEN_EDIT_ARM.model_id => Ok(QWEN_EDIT_ARM),
        (QWEN_EDIT_ID, id) if id == QWEN_EDIT_LIGHTNING_ARM.model_id => {
            Ok(QWEN_EDIT_LIGHTNING_ARM)
        }
        (provider, model_id) => Err(format!(
            "the Candle Qwen edit arm does not implement provider {provider:?} for model {model_id:?}"
        )),
    }
}

/// The seed and step count this member's fixture binds, checked against the arm's own prefix, the
/// planned tier and the recipe's step count — the MLX arm's `planned_qwen_edit_seed` rule, on this
/// lane's fixture spelling.
fn planned_qwen_edit_seed(request: &Value, arm: QwenEditArm, tier: &str) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("{}-{tier}-seed", arm.fixture_prefix);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Qwen edit fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<usize>()
        .map_err(|error| format!("parse Qwen edit fixture step count {steps:?}: {error}"))?;
    if steps != arm.steps {
        return Err(format!(
            "planned.fixture {fixture:?} must use this arm's {}-step calibration request",
            arm.steps
        ));
    }
    Ok(seed)
}

/// The built-in Lightning distill adapter, exactly as the worker stacks it and exactly as the engine
/// pins it: the one file in the pinned snapshot, one LoRA at scale 1.0, no pass scales and no MoE
/// expert. `validate_memory_artifact_recipe` refuses anything else by name, so this is asserted here
/// rather than left to the environment.
fn qwen_edit_lightning_adapter(source: &QwenEditLightningSource) -> Result<AdapterSpec, String> {
    protocol::validate_artifact_identity(
        &source.repository,
        &source.revision,
        protocol::QWEN_EDIT_LIGHTNING_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(&source.root).map_err(|error| {
        format!("canonicalize SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_ROOT: {error}")
    })?;
    let path = root.join(protocol::QWEN_EDIT_LIGHTNING_FILE);
    if !path.is_file() {
        return Err(format!(
            "the Lightning distill LoRA is not at {}",
            path.display()
        ));
    }
    Ok(AdapterSpec::new(path, 1.0, AdapterKind::Lora))
}

/// Where the built-in Lightning distill snapshot lives, as the three
/// `SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_*` values name it — lifted out of the environment so the
/// attachment that makes the Lightning member Lightning is unit-testable.
#[derive(Clone, Debug)]
struct QwenEditLightningSource {
    repository: String,
    revision: String,
    root: PathBuf,
}

fn qwen_edit_lightning_source() -> Result<QwenEditLightningSource, String> {
    Ok(QwenEditLightningSource {
        repository: protocol::required_env("SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_REPOSITORY")?,
        revision: protocol::required_env("SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_REVISION")?,
        root: PathBuf::from(protocol::required_env(
            "SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_ROOT",
        )?),
    })
}

/// The three `SCENEWORKS_QWEN_IMAGE_EDIT_*` values naming the base snapshot one capture opens, plus
/// the distill snapshot the Lightning member stacks — lifted out of the environment so both the tier
/// binding and the adapter attachment are unit-testable.
#[derive(Clone, Debug)]
struct QwenEditArtifactSource {
    repository: String,
    revision: String,
    root: PathBuf,
    lightning: Option<QwenEditLightningSource>,
}

fn qwen_edit_artifact_source(arm: QwenEditArm) -> Result<QwenEditArtifactSource, String> {
    Ok(QwenEditArtifactSource {
        repository: protocol::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_REPOSITORY")?,
        revision: protocol::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_REVISION")?,
        root: PathBuf::from(protocol::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_ROOT")?),
        lightning: arm.lightning.then(qwen_edit_lightning_source).transpose()?,
    })
}

/// The artifact one Candle Qwen edit capture loads: the canonical snapshot root the loader is handed
/// as `QwenEditPaths.root`, and the `LoadSpec` that opens it.
#[derive(Debug)]
struct QwenEditArtifact {
    root: PathBuf,
    spec: LoadSpec,
}

/// The env-free half of the Candle Qwen edit load, so both of its load-time bindings are
/// unit-testable:
///
/// * the root must end in the PLANNED tier's directory. The engine independently pins the same
///   suffix (`exact_base_tier`), so this refusal only makes the diagnosis local instead of a load
///   failure three hundred lines later; and
/// * the built-in distill lands in `spec.adapters` exactly when the arm is the Lightning member —
///   the ONE thing that makes that member Lightning at load time. Every downstream overlay claim in
///   [`run_qwen_edit`] is read back off `spec.adapters` rather than off `arm.lightning`, so a record
///   can never assert an overlay the load did not carry.
///
/// The spec deliberately leaves `offload_policy` at the gen-core default `Resident`, which is what
/// the worker's own `provider_load_spec` for this lane produces (`qwen_edit_candle.rs` never sets
/// it; request-scoped residency travels in `GenerationMemory.stage_residency` instead).
fn qwen_edit_load_spec(
    arm: QwenEditArm,
    tier: &str,
    source: &QwenEditArtifactSource,
    load_shape: LoadShape,
) -> Result<QwenEditArtifact, String> {
    protocol::validate_artifact_identity(
        &source.repository,
        &source.revision,
        protocol::QWEN_EDIT_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(&source.root)
        .map_err(|error| format!("canonicalize SCENEWORKS_QWEN_IMAGE_EDIT_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &source.repository,
        &source.revision,
        tier,
        protocol::QWEN_EDIT_REPOSITORY,
    )?;
    let adapters = if arm.lightning {
        let lightning = source.lightning.as_ref().ok_or_else(|| {
            format!(
                "{} is the Lightning member but no distill snapshot was supplied",
                arm.model_id
            )
        })?;
        vec![qwen_edit_lightning_adapter(lightning)?]
    } else {
        Vec::new()
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_load_shape(load_shape)
        .with_adapters(adapters)
        .with_resolved_route(arm.model_id.to_owned());
    // The edit loader REQUIRES the tier's quant to be stated and to equal the packed snapshot's
    // (`exact_base_tier` + the `spec.quantize != loaded_quant` refusal) — unlike the txt2img Qwen
    // loader, which infers it from `transformer/config.json` and rejects a stated one.
    if let Some(quant) = numeric_tier(tier)?.quant {
        spec = spec.with_quant(quant);
    }
    Ok(QwenEditArtifact { root, spec })
}

/// The materialization shape the plan declares, as a typed `LoadShape`. The edit contract ECHOES
/// `spec.load_shape` back (`candle-gen-qwen-image` `memory_strategy.rs`), so the plan's declaration
/// is what the capture must execute under, and the echo is then re-asserted against it — deriving it
/// from the selected rung would silently rewrite a declared production shape (sc-16482).
fn qwen_edit_planned_load_shape(request: &Value) -> Result<LoadShape, String> {
    match protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?
    {
        protocol::LOAD_SHAPE_EAGER => Ok(LoadShape::EagerMaterialization),
        protocol::LOAD_SHAPE_DEFERRED => Ok(LoadShape::DeferredMaterialization),
        other => Err(format!("unsupported planned.loadShape {other:?}")),
    }
}

/// The one reference every Qwen edit capture conditions on, at the request geometry — the shape the
/// worker hands the engine after `fit_edit_references`. The engine hard-requires at least one
/// (`edit.rs` `encode_conditioning`), which is what makes this capture measure the edit path (the VL
/// vision tower plus the dual-latent VAE encode) rather than text-to-image under an edit label.
fn qwen_edit_reference(width: u32, height: u32) -> Image {
    Image {
        width,
        height,
        pixels: protocol::synthetic_reference_rgb(width, height),
    }
}

/// One Qwen-Image-Edit-2511 anchor capture on the Candle lane, on either catalog id, at any shipped
/// tier.
///
/// E4: the model comes from `QwenEdit::load_with_memory_context` — the exact call the worker makes
/// (`image_jobs/qwen_edit_candle.rs`), with the same `QwenEditPaths`, the same admitted `LoadSpec`
/// (tier quant, resolved route and the Lightning adapter stack) and the same `MemoryRunContext` —
/// and the render is `generate_with_memory_context`, the worker's own generate. The bespoke edit
/// provider is deliberately NOT a registered generator (`edit.rs`: "driven **directly** by the
/// worker … the registered `qwen_image` descriptor stays txt2img-only"), so it cannot ride
/// `load_five_rung_generator`; going through the catalog here would measure the txt2img provider.
fn run_qwen_edit(request: &Value) -> Result<Value, String> {
    let arm = qwen_edit_arm(request)?;
    protocol::validate_exact_overlay_target(request, arm.overlay, arm.execution_path)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let tier = planned_tier(request)?;
    let seed = planned_qwen_edit_seed(request, arm, tier)?;
    let (width, height) = protocol::target_geometry(request)?;
    let source = qwen_edit_artifact_source(arm)?;
    let repository = source.repository.clone();
    let revision = source.revision.clone();
    let selection = planned_selection(request)?;
    let planned_load_shape_value = qwen_edit_planned_load_shape(request)?;
    // The root must end in the PLANNED tier's directory, and the distill is attached here, on
    // exactly the Lightning member.
    let QwenEditArtifact { root, mut spec } =
        qwen_edit_load_spec(arm, tier, &source, planned_load_shape_value)?;
    // Every overlay claim below is read off the stack the LOAD carries, never off `arm.lightning`:
    // if the attachment in `qwen_edit_load_spec` were ever lost, the record must say so rather than
    // assert a distill that never participated.
    let adapters = spec.adapters.clone();
    let loaded_adapters = adapters.len();
    spec.prepare_file_sources()
        .map_err(|error| format!("prepare Qwen edit file sources: {error}"))?;

    // The contract is read weights-free from the registered memory surface, so the planned
    // fingerprint and load shape are checked BEFORE a 28-57 GB load.
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let contract = catalog
        .media()
        .memory_strategy_contract(QWEN_EDIT_ID, &spec)
        .map_err(|error| format!("read {QWEN_EDIT_ID} memory-strategy contract: {error}"))?
        .ok_or_else(|| format!("{QWEN_EDIT_ID} has no memory-strategy contract"))?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {QWEN_EDIT_ID} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| format!("pinned {QWEN_EDIT_ID} provider has no calibration identity"))?;
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
    if planned_load_shape_value != calibration.load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(planned_load_shape_value),
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let stage_residency = matches!(
        selection.strategy,
        MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedDecode
            | MemoryStrategy::BoundedAttention
            | MemoryStrategy::BoundedTransformerResidency
    );
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Edit,
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
        // `validate_edit_route` requires this to be exactly `Some("lora")` when the load carries
        // adapters and `None` when it does not — so it is derived from the spec's own stack.
        overlay: (loaded_adapters > 0).then(|| "lora".to_owned()),
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22728@{}", protocol::INFERENCE_PIN),
    };

    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let model = QwenEdit::load_with_memory_context(
        &QwenEditPaths {
            root,
            text_encoder: None,
            adapters,
            // Compatibility-only load field; residency is request-scoped, exactly as the worker
            // passes it.
            offload_policy: OffloadPolicy::Resident,
        },
        &spec,
        &context,
    )
    .map_err(|error| format!("load real {} {tier} provider: {error}", arm.model_id))?;
    vram.end_load(load_sample);

    let generation = QwenEditRequest {
        prompt: QWEN_EDIT_PROMPT.to_owned(),
        negative: String::new(),
        width,
        height,
        steps: arm.steps,
        guidance: QWEN_EDIT_GUIDANCE,
        seed,
        lightning: arm.lightning,
        stage_residency,
        memory: Some(GenerationMemory {
            stage_residency,
            tile_vae_decode: selection.parameters.decode_tile_edge.is_some(),
            chunk_attention: selection.parameters.attention_chunk_size.is_some(),
            stream_transformer_blocks: selection.parameters.transformer_window_size.is_some(),
            decode_tile_edge: selection.parameters.decode_tile_edge,
            decode_overlap: selection.parameters.decode_overlap,
            attention_chunk_size: selection.parameters.attention_chunk_size,
            transformer_window_size: selection.parameters.transformer_window_size,
            ..Default::default()
        }),
        ..Default::default()
    };
    let references = [qwen_edit_reference(width, height)];

    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let result =
        model.generate_with_memory_context(&context, &generation, &references, &mut |progress| {
            let boundary = match progress {
                Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                    protocol::ReferenceBoundary::RendererLoad
                }
                Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
                Progress::Decoding => protocol::ReferenceBoundary::Decoding,
                _ => return,
            };
            let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary)
            else {
                return;
            };
            let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
            match phase {
                MemoryPhase::Conditioning => conditioning_peak_gb = peak,
                MemoryPhase::Denoise => denoise_peak_gb = peak,
                MemoryPhase::Decode => decode_peak_gb = peak,
            }
            phase = memory_phase(next);
            phase_sample = Some(vram.phase());
        });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    let image =
        result.map_err(|error| format!("{} edit generation failed: {error}", arm.model_id))?;
    if image.width != width || image.height != height {
        return Err(format!(
            "{} returned {}x{}, not the requested {width}x{height}",
            arm.model_id, image.width, image.height
        ));
    }
    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        format!(
            "{} capture did not expose a conditioning boundary",
            arm.model_id
        )
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            format!("{} capture did not expose a denoise boundary", arm.model_id)
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb
            .ok_or_else(|| format!("{} capture did not complete decode", arm.model_id))?,
    );
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);

    let blocker = concat!(
        "sc-22728 anchor capture measures exact per-phase memory and strategy identity for the ",
        "Candle Qwen-Image-Edit-2511 lane; it intentionally remains gated because this run does ",
        "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
    );
    let sweep = protocol::reference_sweep(request, "passed")?;
    let parts = || protocol::PlainGatedFragment {
        artifact: artifact(&repository, &revision, tier),
        sweep: sweep.clone(),
        blocker,
        quality: json!({ "result": "not_run" }),
        negative_mutation: Value::Null,
        loadability: json!({
            "result": "passed",
            "resolvedPathFingerprint": loadability_fingerprint(&repository, &revision, tier),
        }),
        diagnostics: protocol::diagnostics(
            &format!("memory-candle-adapter:{}-anchor", arm.slug),
            "executed",
            [blocker.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
                ("referenceImages", "count", 1),
                ("builtInAdapters", "count", loaded_adapters as u64),
            ],
        ),
    };
    let mut fragment = if loaded_adapters > 0 {
        protocol::overlay_gated_fragment(
            request,
            arm.overlay,
            arm.execution_path,
            "the built-in lightx2v Lightning distill LoRA was folded into the MMDiT at load and participated in the measured render",
            parts(),
        )?
    } else {
        protocol::plain_gated_fragment(request, arm.execution_path, parts())?
    };
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    Ok(fragment)
}

fn run(request: &Value) -> Result<Value, String> {
    if protocol::planned(request)?
        .get("backend")
        .and_then(Value::as_str)
        != Some("candle")
    {
        return Err(
            "Candle adapter received a non-Candle planned case; run the harness with --backend candle"
                .to_owned(),
        );
    }
    let provider = planned_provider(request)?;
    if provider == LTX25_ID {
        return run_ltx25_capture(request);
    }
    // sc-22736: the Wan 2.2 family and SCAIL-2 dispatch ABOVE the shared still gate, like LTX-2.5 —
    // they are VIDEO arms, and `validate_still_geometry` would refuse the multi-frame geometry they
    // exist to measure.
    if candle_wan_scail2::implements(provider) {
        return candle_wan_scail2::run(request);
    }
    // sc-22737: Bernini's VIDEO entry, LTX-2.3 and both MiniMax-H3 entries dispatch above the
    // shared still gate for the same reason the two blocks above do — `validate_still_geometry`
    // would refuse the multi-frame geometry these arms exist to measure. `sc22737_video_arm`
    // answers `None` for a provider this block does not serve AND for Bernini's STILL entry
    // (`bernini_image`), which is an image route and rides the shared five-rung reference path
    // below; it errors by name only for an unknown `(provider, modelId)` pair on a provider it
    // does serve.
    if let Some(arm) = sc22737_video_arm(request)? {
        return run_sc22737_video_capture(request, arm);
    }
    // sc-22726: PuLID-FLUX dispatches ABOVE the shared plain-overlay gate, like LTX-2.5. Its
    // declared overlay is `identity`, so routing it through `validate_plain_overlay_target` would
    // refuse the one target it exists to measure.
    if provider == PULID_FLUX_ID {
        return run_pulid_flux_capture(request);
    }
    // sc-22728: likewise before the plain-overlay gate below, because the Lightning member declares
    // a material `lora` overlay — its built-in distill — and validates it exactly instead.
    if provider == QWEN_EDIT_ID {
        return run_qwen_edit(request);
    }
    let execution_path = plain_execution_path(request)?;
    protocol::validate_plain_overlay_target(request, execution_path)?;
    // Both dispatch targets below are image arms; refuse a non-still geometry here, before either of
    // them resolves an environment variable or touches a weight snapshot (sc-18808).
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    if routes_to_five_rung_reference(request)? {
        return run_five_rung_reference(request);
    }
    if provider != KREA_ID {
        return Err(format!(
            "unsupported Candle calibration provider {provider:?}"
        ));
    }
    let parameters = protocol::strategy_parameters(request)?;
    let tier = planned_tier(request)?;
    validate_fixture_binds_tier_and_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_KREA_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_KREA_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::KREA_REPOSITORY)?;
    let root = std::env::var("SCENEWORKS_KREA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_default();
    let root = if root.is_dir() {
        let canonical = std::fs::canonicalize(root)
            .map_err(|error| format!("canonicalize SCENEWORKS_KREA_ROOT: {error}"))?;
        protocol::validate_huggingface_snapshot_root(
            &canonical,
            &repository,
            &revision,
            tier,
            protocol::KREA_REPOSITORY,
        )?;
        canonical
    } else {
        root
    };
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Sequential);
    let spec = match numeric_tier(tier)?.quant {
        Some(quant) => spec.with_quant(quant),
        None => spec,
    };
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let contract = catalog
        .media()
        .memory_strategy_contract(KREA_ID, &spec)
        .map_err(|error| format!("read {KREA_ID} memory-strategy contract: {error}"))?
        .ok_or_else(|| {
            format!(
                "{KREA_ID} has no memory-strategy contract at {}",
                protocol::INFERENCE_PIN
            )
        })?;
    let edge = protocol::parameter(request, "decodeTileEdge")?;
    let overlap = protocol::parameter(request, "decodeOverlap")?;
    let attention = protocol::parameter(request, "attentionChunkSize")?;
    let window = protocol::parameter(request, "transformerWindowSize")?;
    let selected = MemoryStrategyParameters {
        decode_tile_edge: Some(edge),
        decode_overlap: Some(overlap),
        attention_chunk_size: Some(attention),
        transformer_window_size: Some(window),
        transformer_window_component: Some(TransformerComponent::Dit),
    };
    let selection = MemorySelection {
        strategy: MemoryStrategy::BoundedTransformerResidency,
        parameters: selected,
        tier: numeric_tier(tier)?,
    };
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    let actual_calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "{KREA_ID} has no calibration identity at {}",
            protocol::INFERENCE_PIN
        )
    })?;
    let actual_fingerprint = actual_calibration.fingerprint.as_str();
    if planned_fingerprint != actual_fingerprint {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!(
                "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={actual_fingerprint} at {}",
                protocol::INFERENCE_PIN
            ),
            "contractFingerprintMismatch",
            &repository,
            &revision,
        );
    }

    if let Err(reason) = contract.validate_selection(&selection) {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!("pinned provider rejected planned parameters before load: {reason}"),
            "contractParameterRejection",
            &repository,
            &revision,
        );
    }
    if !root.is_dir() {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!(
                "supported provider tuple requires real weights; set SCENEWORKS_KREA_ROOT to                  the validated {tier} snapshot"
            ),
            "missingWeights",
            &repository,
            &revision,
        );
    }

    let hardware = request
        .get("hardware")
        .and_then(Value::as_object)
        .ok_or_else(|| "run request.hardware must be an object".to_owned())?;
    let total_bytes = hardware
        .get("memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let (width, height) = protocol::target_geometry(request)?;
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: actual_calibration.abi,
        calibration_fingerprint: actual_calibration.fingerprint.clone(),
        load_shape: actual_calibration.load_shape,
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
            reserved_headroom_bytes: 2 * GIB,
        },
        predicted_peak_bytes: total_bytes.saturating_sub(2 * GIB),
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-21714-certifying@{}", protocol::INFERENCE_PIN),
    };

    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(KREA_ID, &spec)
        .map_err(|error| format!("load real {KREA_ID} {tier} generator: {error}"))?;
    vram.end_load(load_sample);
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin real Krea memory-strategy scope: {error}"))?
        .ok_or_else(|| "optimized Krea selection did not create a provider scope".to_owned())?;
    scope
        .configure_decode(edge, overlap, context.geometry)
        .map_err(|error| format!("configure Krea decode tuple: {error}"))?;
    scope
        .configure_attention(attention)
        .map_err(|error| format!("configure Krea attention tuple: {error}"))?;
    scope
        .materialize_transformer_window(0, window)
        .map_err(|error| format!("configure Krea transformer tuple: {error}"))?;

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply Krea request-scoped strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter Krea conditioning phase: {error}"))?;

    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            conditioning_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
            phase_sample = Some(vram.phase());
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            denoise_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
            phase_sample = Some(vram.phase());
        }
        _ => {}
    });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    let report = vram.report();
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!("real Krea {tier} generation failed: {message}"));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave terminal Krea phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish real Krea memory-strategy scope: {error}"))?;
    let selected_image = match output {
        runtime_cuda::gen_core::GenerationOutput::Images(mut images) if images.len() == 1 => {
            images.remove(0)
        }
        runtime_cuda::gen_core::GenerationOutput::Images(images) => {
            return Err(format!(
                "real Krea run returned {} images, expected 1",
                images.len()
            ));
        }
        _ => return Err("real Krea run returned non-image output".to_owned()),
    };

    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        "Krea run did not expose a conditioning-to-denoise phase boundary".to_owned()
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            "Krea run did not expose a denoise-to-decode phase boundary".to_owned()
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb.ok_or_else(|| "Krea run did not complete decode sampling".to_owned())?,
    );
    let overall_bytes = decimal_gb_to_bytes(report.peak_gb)
        .max(conditioning_bytes)
        .max(denoise_bytes)
        .max(decode_bytes);
    let baseline = decimal_gb_to_bytes(report.baseline_gb);

    let mut exact = context.clone();
    exact.predicted_peak_bytes = overall_bytes;
    exact.budget = MemoryBudget {
        total_bytes: overall_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("Candle Krea provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = exact.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Candle Krea provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = exact.clone();
    stale.calibration_fingerprint = "stale-krea-turbo-candle-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Candle Krea provider accepted stale calibration evidence".to_owned());
    }

    let lifecycle_phases = [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let smi = NvidiaSmi::resolve()?;
    let cleanup_tolerance_bytes = 64 * MIB;
    let mut maximum_cleanup_growth_bytes = 0_u64;
    let mut maximum_recovery_maximum_error = 0.0_f64;
    let mut maximum_recovery_mean_error = 0.0_f64;
    for lifecycle_phase in lifecycle_phases {
        let before_fault_bytes = smi.used_bytes()?;
        let canceled_output = execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            Some(lifecycle_phase),
        )?;
        if canceled_output.is_some() {
            return Err(format!(
                "{lifecycle_phase:?} cancellation unexpectedly produced an image"
            ));
        }
        let after_fault_bytes = smi.used_bytes()?;
        let cleanup_growth_bytes = after_fault_bytes.saturating_sub(before_fault_bytes);
        maximum_cleanup_growth_bytes = maximum_cleanup_growth_bytes.max(cleanup_growth_bytes);
        if cleanup_growth_bytes > cleanup_tolerance_bytes {
            return Err(format!(
                "{lifecycle_phase:?} cancellation retained {cleanup_growth_bytes} device bytes above its pre-request baseline"
            ));
        }
        let recovered = execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            None,
        )?
        .ok_or_else(|| {
            format!("{lifecycle_phase:?} cancellation warm follow-up produced no image")
        })?;
        let (maximum, mean) = pixel_error(&selected_image, &recovered)?;
        ensure_krea_quality(
            maximum,
            mean,
            &format!("{lifecycle_phase:?} cancellation recovery"),
        )?;
        maximum_recovery_maximum_error = maximum_recovery_maximum_error.max(maximum);
        maximum_recovery_mean_error = maximum_recovery_mean_error.max(mean);
    }
    for lifecycle_phase in lifecycle_phases {
        let before_fault_bytes = smi.used_bytes()?;
        let fault_output = execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            Some(lifecycle_phase),
            None,
        )?;
        if fault_output.is_some() {
            return Err(format!(
                "{lifecycle_phase:?} injected error unexpectedly produced an image"
            ));
        }
        let after_fault_bytes = smi.used_bytes()?;
        let cleanup_growth_bytes = after_fault_bytes.saturating_sub(before_fault_bytes);
        maximum_cleanup_growth_bytes = maximum_cleanup_growth_bytes.max(cleanup_growth_bytes);
        if cleanup_growth_bytes > cleanup_tolerance_bytes {
            return Err(format!(
                "{lifecycle_phase:?} injected error retained {cleanup_growth_bytes} device bytes above its pre-request baseline"
            ));
        }
        let recovered = execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            None,
        )?
        .ok_or_else(|| format!("{lifecycle_phase:?} error warm follow-up produced no image"))?;
        let (maximum, mean) = pixel_error(&selected_image, &recovered)?;
        ensure_krea_quality(
            maximum,
            mean,
            &format!("{lifecycle_phase:?} error recovery"),
        )?;
        maximum_recovery_maximum_error = maximum_recovery_maximum_error.max(maximum);
        maximum_recovery_mean_error = maximum_recovery_mean_error.max(mean);
    }
    let resident_parameters = MemoryStrategyParameters::default();
    let resident_a = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::Resident,
        resident_parameters,
    )?;
    let bounded_b = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::BoundedTransformerResidency,
        selected,
    )?;
    let resident_a_repeat = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::Resident,
        resident_parameters,
    )?;
    let (resident_repeat_max_error, resident_repeat_mean_error) =
        pixel_error(&resident_a, &resident_a_repeat)?;
    if resident_repeat_max_error != 0.0 || resident_repeat_mean_error != 0.0 {
        return Err(format!(
            "resident A-B-A repeat was not deterministic: max={resident_repeat_max_error:.6}, mean={resident_repeat_mean_error:.6}"
        ));
    }
    let (bounded_max_error, bounded_mean_error) = pixel_error(&resident_a, &bounded_b)?;
    ensure_krea_quality(
        bounded_max_error,
        bounded_mean_error,
        "bounded-versus-resident A-B-A parity",
    )?;
    let mutated = negative_mutation(&bounded_b);
    let (mutated_max_error, mutated_mean_error) = pixel_error(&resident_a, &mutated)?;
    if mutated_max_error <= KREA_CANDLE_MAX_THRESHOLD
        && mutated_mean_error <= KREA_CANDLE_MEAN_THRESHOLD
    {
        return Err("Candle Krea output mutation did not breach the parity envelope".to_owned());
    }

    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(actual_calibration.load_shape),
        "artifact": artifact(&repository, &revision, tier),
        "sweep": complete_sweep(request, parameters)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": overall_bytes, "effectiveBudgetBytes": overall_bytes },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "reason": "conditioning, denoise, and decode cancellation returned typed cancellation; retained memory stayed within 64 MiB and every warm recovery passed the approved parity envelope", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "conditioning, denoise, and decode injected errors fired at physical boundaries; retained memory stayed within 64 MiB and every warm recovery passed the approved parity envelope", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": {
            "conditioning": conditioning_bytes,
            "denoise": denoise_bytes,
            "decode": decode_bytes,
            "overall": overall_bytes,
        },
        "observedMemory": {
            "conditioning": cuda_phase_metrics(conditioning_bytes),
            "denoise": cuda_phase_metrics(denoise_bytes),
            "decode": cuda_phase_metrics(decode_bytes),
            "overall": cuda_phase_metrics(overall_bytes),
        },
        "quality": {
            "contract": "identical artifact, prompt, seed, geometry, steps and tier; production bounded-transformer tuple versus resident A-B-A control",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": bounded_max_error,
            "meanError": bounded_mean_error,
            "maximumErrorThreshold": KREA_CANDLE_MAX_THRESHOLD,
            "meanErrorThreshold": KREA_CANDLE_MEAN_THRESHOLD,
        },
        "negativeMutation": {
            "parameters": parameters,
            "measured": true,
            "result": "failed_as_expected",
            "maximumError": mutated_max_error,
            "meanError": mutated_mean_error,
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": loadability_fingerprint(&repository, &revision, tier),
        },
        "diagnostics": protocol::diagnostics(
            "memory-candle-adapter:krea-turbo-certifying",
            "executed",
            [],
            [
                    ("preLoadDeviceUsed", "bytes", baseline),
                    (
                        "loadDevicePeakDelta",
                        "bytes",
                        decimal_gb_to_bytes(report.load_peak_gb),
                    ),
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("allocatorCounterAliasesDeviceDelta", "boolean", 1),
                    ("wiredCounterAliasesDiscreteDeviceDelta", "boolean", 1),
                    ("cudaCachingAllocatorPresent", "boolean", 0),
                    ("phaseCancelInjections", "count", 3),
                    ("phaseErrorInjections", "count", 3),
                    ("postFaultWarmFollowUps", "count", 6),
                    (
                        "maximumPostFaultDeviceGrowth",
                        "bytes",
                        maximum_cleanup_growth_bytes,
                    ),
                    (
                        "postFaultDeviceGrowthTolerance",
                        "bytes",
                        cleanup_tolerance_bytes,
                    ),
                    (
                        "abaResidentRepeatMaximumErrorPer255",
                        "count",
                        (resident_repeat_max_error * 255.0).round() as u64,
                    ),
                    (
                        "abaResidentRepeatMeanErrorMicroUnits",
                        "count",
                        (resident_repeat_mean_error * 1_000_000.0).round() as u64,
                    ),
                    (
                        "abaBoundedMaximumErrorPer255",
                        "count",
                        (bounded_max_error * 255.0).round() as u64,
                    ),
                    (
                        "abaBoundedMeanErrorMicroUnits",
                        "count",
                        (bounded_mean_error * 1_000_000.0).round() as u64,
                    ),
                    (
                        "maximumRecoveryMaximumErrorPer255",
                        "count",
                        (maximum_recovery_maximum_error * 255.0).round() as u64,
                    ),
                    (
                        "maximumRecoveryMeanErrorMicroUnits",
                        "count",
                        (maximum_recovery_mean_error * 1_000_000.0).round() as u64,
                    ),
                    (
                        "negativeMutationMaximumErrorPer255",
                        "count",
                        (mutated_max_error * 255.0).round() as u64,
                    ),
                    (
                        "negativeMutationMeanErrorMicroUnits",
                        "count",
                        (mutated_mean_error * 1_000_000.0).round() as u64,
                    ),
                ],
            ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, KREA_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}

fn main() {
    let request = protocol::request_from_stdin().unwrap_or_else(|error| protocol::fail(error));
    let response = match protocol::action(&request).unwrap_or_else(|error| protocol::fail(error)) {
        "probe" => probe(),
        "run" => run(&request),
        "run_batch" => run_five_rung_batch(&request),
        other => Err(format!("unsupported action {other:?}")),
    }
    .unwrap_or_else(|error| protocol::fail(error));
    protocol::write_response(&response).unwrap_or_else(|error| protocol::fail(error));
}

#[cfg(test)]
mod mage_tests {
    use super::*;

    fn mage_planned(provider: &str, model_id: &str, tier: &str, steps: u32) -> Value {
        let slug = provider.replace('_', "-");
        json!({
            "planned": {
                "target": {
                    "provider": provider,
                    "modelId": model_id,
                    "tier": tier,
                    "mode": if provider.contains("edit") { "edit_image" } else { "text_to_image" },
                    "overlay": "none",
                    "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
                },
                "backend": "candle",
                "loadShape": if tier == "bf16" { "deferred_materialization" } else { "eager_materialization" },
                "strategy": {
                    "rung": "staged_residency",
                    "engagedRungs": ["resident", "staged_residency"],
                    "parameters": {}
                },
                "calibrationFingerprint":
                    format!("mage-flow-cuda-{slug}-{tier}-shared-ladder-v3"),
                "fixture": format!("{slug}-candle-{tier}-768-seed{MAGE_SEED}-step{steps}"),
            }
        })
    }

    /// Every Mage member resolves an arm, a crossed `(provider, modelId)` pair is refused by name,
    /// and a non-Mage provider yields `None` rather than being swept into this family.
    #[test]
    fn the_arm_is_resolved_from_the_provider_and_model_pair() {
        for arm in MAGE_ARMS {
            let request = mage_planned(arm.provider, arm.provider, "q4", arm.steps);
            assert_eq!(mage_arm(&request).unwrap(), Some(arm));
        }
        let crossed = mage_planned("mage_flow_edit", "mage_flow_edit_turbo", "q4", 2);
        assert!(mage_arm(&crossed)
            .unwrap_err()
            .contains("does not implement"));
        let foreign = mage_planned(QWEN_ID, QWEN_ID, "q4", 2);
        assert_eq!(mage_arm(&foreign).unwrap(), None);
    }

    /// The three gates that carry the shared refusal phrase all name every Mage member, and the
    /// five-rung loader refuses one by name rather than trying to open it with a single root.
    /// `adapterCapturableProviders` INTERSECTS those gates, so a member missing from any one of them
    /// disappears from the derived capturable set.
    #[test]
    fn every_mage_member_is_named_by_every_dispatch_gate() {
        for arm in MAGE_ARMS {
            let request = mage_planned(arm.provider, arm.provider, "q4", arm.steps);
            assert_eq!(plain_execution_path(&request).unwrap(), arm.execution_path);
            assert_eq!(
                still_calibration_label(&request).unwrap(),
                arm.still_calibration
            );
            assert!(routes_to_five_rung_reference(&request).unwrap());
            assert_eq!(five_rung_evidence_story(arm.provider), "sc-22733");
            assert!(mage_registry_bypass(arm.provider).contains("load_mage_generator"));
        }
    }

    /// A plan row declaring the OTHER mode for a member is refused.
    #[test]
    fn a_plan_declaring_the_wrong_mode_for_a_member_is_refused() {
        for arm in MAGE_ARMS {
            let mut request = mage_planned(arm.provider, arm.provider, "q4", arm.steps);
            let wrong = if arm.edit {
                "text_to_image"
            } else {
                "edit_image"
            };
            request["planned"]["target"]["mode"] = json!(wrong);
            assert_eq!(
                validate_mage_mode(&request, arm).unwrap_err(),
                format!(
                    "{} renders {:?}, but the plan declares mode {wrong:?}",
                    arm.provider,
                    mage_mode(arm)
                )
            );
        }
    }

    /// No two rows share an artifact, an env family, a fixture prefix or an execution path.
    #[test]
    fn every_mage_arm_row_is_unique_on_every_identifying_axis() {
        assert_eq!(MAGE_ARMS.len(), 6);
        for axis in [
            MAGE_ARMS.map(|arm| arm.provider),
            MAGE_ARMS.map(|arm| arm.repository),
            MAGE_ARMS.map(|arm| arm.repository_env),
            MAGE_ARMS.map(|arm| arm.fixture_prefix),
            MAGE_ARMS.map(|arm| arm.execution_path),
            MAGE_ARMS.map(|arm| arm.still_calibration),
        ] {
            let mut sorted = axis.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                6,
                "an identifying axis is shared by two Mage arms"
            );
        }
        for arm in MAGE_ARMS {
            assert_eq!(arm.edit, arm.provider.contains("edit"));
            let distilled = arm.provider.ends_with("turbo");
            assert_eq!(arm.steps, if distilled { 4 } else { 2 });
            assert_eq!(arm.guidance, if distilled { 1.0 } else { 5.0 });
        }
    }

    /// The fixture binds member, tier, edge, seed and the recipe's step count.
    #[test]
    fn the_fixture_binds_the_member_the_tier_and_the_step_count() {
        let arm = MAGE_ARMS[5];
        let good = mage_planned(arm.provider, arm.provider, "bf16", arm.steps);
        validate_mage_fixture(&good, arm, "bf16").unwrap();
        assert!(validate_mage_fixture(&good, arm, "q4")
            .unwrap_err()
            .contains("must start with"));
        let wrong_steps = mage_planned(arm.provider, arm.provider, "bf16", 2);
        assert!(validate_mage_fixture(&wrong_steps, arm, "bf16")
            .unwrap_err()
            .contains("4-step calibration request"));
        let mut wrong_seed = good.clone();
        wrong_seed["planned"]["fixture"] =
            json!(format!("{}-bf16-768-seed16402-step4", arm.fixture_prefix));
        assert!(validate_mage_fixture(&wrong_seed, arm, "bf16")
            .unwrap_err()
            .contains("does not match the seed"));
    }

    /// The composed `LoadSpec` is the shape the WORKER loads at the candle anchor rung: `Sequential`
    /// (the `staged_residency` composition), the PLAN's per-tier load shape — deferred on bf16
    /// (the Applied BTR declaration), eager on q4/q8 (the refused one; see [`mage_load_spec`]) —
    /// the planned tier's quant on the spec (`resolved_quant` reads it directly), and both shared
    /// components staged from the components snapshot's own tier directory.
    ///
    /// The shape is asserted against the committed plan's 18 `mage_flow*:*:candle` rows, read
    /// through the same `mage_planned_load_shape` a capture uses, rather than a literal: the
    /// worker's `memory_route_registry` test pins those rows to the registry's own evaluation over
    /// the real manifest entries, so this closes the chain arm == plan == registry. The per-tier
    /// split is ALSO asserted literally, so a plan regenerated all-eager or all-deferred cannot
    /// carry this test along with it.
    #[test]
    fn the_load_spec_is_the_shape_the_worker_loads() {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let anchors = plan["anchors"].as_object().expect("anchors object");
        for tier in ["bf16", "q4", "q8"] {
            let root = PathBuf::from("/hub/models--SceneWorks--Mage-Flow/snapshots/abc").join(tier);
            let components =
                PathBuf::from("/hub/models--SceneWorks--Mage-Flow-Components-mlx/snapshots/def");
            let expected_shape = if tier == "bf16" {
                LoadShape::DeferredMaterialization
            } else {
                LoadShape::EagerMaterialization
            };
            for arm in MAGE_ARMS {
                let key = format!("{}:{tier}:candle", arm.provider);
                let request =
                    json!({ "planned": { "loadShape": anchors[&key]["loadShape"].clone() } });
                let load_shape = mage_planned_load_shape(&request)
                    .unwrap_or_else(|error| panic!("{key}: {error}"));
                assert_eq!(
                    load_shape, expected_shape,
                    "{key}: the plan must bind the shape the worker loads on this tier"
                );
                let spec = mage_load_spec(tier, load_shape, root.clone(), &components).unwrap();
                assert_eq!(spec.weights, WeightsSource::Dir(root.clone()));
                assert_eq!(spec.offload_policy, OffloadPolicy::Sequential);
                assert_eq!(
                    spec.load_shape, load_shape,
                    "{key}: the arm must load the shape the plan (and the worker) binds"
                );
            }
            let spec = mage_load_spec(tier, expected_shape, root.clone(), &components).unwrap();
            assert_eq!(spec.quantize, numeric_tier(tier).unwrap().quant);
            for component in [
                protocol::MAGE_COMPONENT_TEXT_ENCODER,
                protocol::MAGE_COMPONENT_VAE,
            ] {
                let Some(WeightsSource::Dir(dir)) = spec.components.get(component) else {
                    panic!("the {component} component was not staged");
                };
                assert_eq!(dir, &components.join(tier).join(component));
            }
        }
    }

    /// Every candle Mage plan row names the per-(member, tier) production identity the loaded
    /// generator publishes (inference PR 953's `-v3` table), the 18 identities are distinct, no row
    /// carries the retired tier-free `-v2` string or an MLX identity, and the pre-load check
    /// refuses a row that does — before any env or weights are touched.
    #[test]
    fn every_planned_mage_candle_row_names_the_production_identity_and_is_checked_before_the_load()
    {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let anchors = plan["anchors"].as_object().expect("anchors object");
        let mut identities = std::collections::BTreeSet::new();
        let mut seen = std::collections::BTreeSet::new();
        for arm in MAGE_ARMS {
            for tier in ["bf16", "q4", "q8"] {
                let key = format!("{}:{tier}:candle", arm.provider);
                let row = &anchors[&key];
                let expected = mage_calibration_fingerprint(arm, tier);
                assert_eq!(
                    row["calibrationFingerprint"].as_str(),
                    Some(expected.as_str()),
                    "{key}"
                );
                assert!(expected.ends_with("-shared-ladder-v3"), "{expected}");
                assert!(expected.contains(&format!("-{tier}-")), "{expected}");
                assert!(!expected.contains("-mlx-"), "{expected}");
                assert!(
                    identities.insert(expected),
                    "{key}: identity shared with another cell"
                );
                seen.insert(key.clone());
                // The checked-in row passes the pre-load check; a retired or foreign string fails
                // it by name.
                let request = json!({ "planned": {
                    "target": { "provider": arm.provider, "modelId": arm.provider, "tier": tier },
                    "calibrationFingerprint": row["calibrationFingerprint"].clone(),
                }});
                validate_mage_plan_identity(&request, arm, tier).unwrap();
                assert!(mage_planned_load_shape(
                    &json!({ "planned": { "loadShape": "streamed" } })
                )
                .unwrap_err()
                .contains("unsupported planned.loadShape"));
                for stale in [
                    format!(
                        "mage-flow-cuda-shared-ladder-provider-abi-v2-{}",
                        arm.provider.replace('_', "-")
                    ),
                    format!(
                        "mage-flow-{}-{tier}-mlx-shared-ladder-v1",
                        arm.provider.replace('_', "-")
                    ),
                ] {
                    let mut wrong = request.clone();
                    wrong["planned"]["calibrationFingerprint"] = json!(stale);
                    let error = validate_mage_plan_identity(&wrong, arm, tier).unwrap_err();
                    assert!(error.contains("calibration mismatch"), "{key}: {error}");
                    assert!(error.contains(&stale), "{key}: {error}");
                }
            }
        }
        let expected: std::collections::BTreeSet<String> = MAGE_ARMS
            .iter()
            .flat_map(|arm| {
                ["bf16", "q4", "q8"]
                    .iter()
                    .map(move |tier| format!("{}:{tier}:candle", arm.provider))
            })
            .collect();
        assert_eq!(seen, expected);
        assert_eq!(identities.len(), 18);
    }

    /// The edit members condition on exactly one reference, the text-to-image members on none, and
    /// only the undistilled members carry a negative prompt (the distilled ones run CFG off).
    #[test]
    fn only_the_edit_members_carry_a_reference() {
        for arm in MAGE_ARMS {
            let request = mage_generation_request(arm, 768, 768);
            assert_eq!(request.conditioning.len(), usize::from(arm.edit));
            if arm.edit {
                let Conditioning::Reference { image, strength } = &request.conditioning[0] else {
                    panic!("{} must condition on a Reference", arm.provider);
                };
                assert_eq!((image.width, image.height), (768, 768));
                assert_eq!(
                    *strength, None,
                    "Mage's edit_references never reads a strength lever"
                );
            }
            assert_eq!(request.steps, Some(arm.steps));
            assert_eq!(request.guidance, Some(arm.guidance));
            assert_eq!(request.negative_prompt.is_some(), arm.guidance > 1.0);
            assert_eq!(request.seed, Some(MAGE_SEED));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_krea_wddm_idle_proof_keeps_the_measured_strict_bounds() {
        let config = certifying_wddm_idle_config();
        assert_eq!(config.max_baseline_gb, 2.0);
        assert_eq!(config.sample_count, 5);
        assert_eq!(config.max_drift_mib, 64);
        assert_eq!(config.sample_interval_ms, 200);
    }

    #[test]
    fn candle_krea_quality_uses_normalized_channel_error_and_the_approved_mean_bound() {
        let golden = runtime_cuda::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 128, 255],
        };
        let within = runtime_cuda::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 128, 250],
        };
        let (maximum, mean) = pixel_error(&golden, &within).unwrap();
        assert_eq!(maximum, 5.0 / 255.0);
        assert_eq!(mean, 5.0 / (3.0 * 255.0));
        ensure_krea_quality(maximum, mean, "golden policy").unwrap();

        let above_mean = runtime_cuda::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![6, 134, 249],
        };
        let (maximum, mean) = pixel_error(&golden, &above_mean).unwrap();
        assert!(maximum < KREA_CANDLE_MAX_THRESHOLD);
        assert!(mean > KREA_CANDLE_MEAN_THRESHOLD);
        assert!(ensure_krea_quality(maximum, mean, "regression").is_err());
        assert!(ensure_krea_quality(1.0, 0.01681, "exact boundary").is_ok());
        assert!(ensure_krea_quality(1.0, 0.01681 + 1e-12, "mean above boundary").is_err());
        assert!(ensure_krea_quality(1.0 + 1e-12, 0.01681, "maximum above boundary").is_err());
    }

    #[test]
    fn candle_krea_negative_mutation_breaches_the_certifying_mean_envelope() {
        let image = runtime_cuda::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 128, 255],
        };
        let mutated = negative_mutation(&image);
        assert_eq!(mutated.pixels, vec![64, 192, 63]);
        let (maximum, mean) = pixel_error(&image, &mutated).unwrap();
        assert!(maximum > KREA_CANDLE_MEAN_THRESHOLD);
        assert!(mean > KREA_CANDLE_MEAN_THRESHOLD);
    }

    #[test]
    fn candle_krea_complete_sweep_certifies_only_the_v1_production_tuple() {
        let request = json!({
            "planned": {
                "calibrationFingerprint": "krea-turbo-cuda-phase-curves-v1"
            }
        });
        let parameters = Map::from_iter([
            ("decodeTileEdge".to_owned(), json!(512)),
            ("decodeOverlap".to_owned(), json!(128)),
            ("attentionChunkSize".to_owned(), json!(134_217_728)),
            ("transformerWindowSize".to_owned(), json!(1)),
        ]);
        let sweep = complete_sweep(&request, &parameters).unwrap();
        assert_eq!(sweep["rangeVerified"], json!(true));
        assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
        assert_eq!(sweep["cases"][0]["result"], json!("passed"));
        assert_eq!(sweep["cases"][0]["parameters"], json!(parameters));
    }

    fn qwen_request() -> Value {
        json!({
            "planned": {
                "target": { "provider": "qwen_image", "overlay": "none" },
                "strategy": { "rung": "resident", "parameters": {} },
                "loadShape": "deferred_materialization"
            }
        })
    }

    #[test]
    fn qwen_plan_routes_to_the_qwen_base_execution_path() {
        let request = qwen_request();
        assert_eq!(planned_provider(&request).unwrap(), "qwen_image");
        assert_eq!(
            plain_execution_path(&request).unwrap(),
            QWEN_PLAIN_EXECUTION_PATH
        );
        assert_eq!(
            planned_memory_strategy(&request).unwrap(),
            MemoryStrategy::Resident
        );
    }

    /// The edit provider must never be measured through the five-rung txt2img path. sc-22728 gave it
    /// its own arm (`run_qwen_edit`, dispatched ahead of this one), and the five-rung route still
    /// refuses it by name — which is what keeps a mis-ordered dispatch a refusal rather than a
    /// txt2img record wearing an edit label.
    #[test]
    fn the_edit_provider_is_never_served_by_the_five_rung_txt2img_path() {
        let mut request = qwen_request();
        request["planned"]["target"]["provider"] = json!("qwen_image_edit");
        let error = plain_execution_path(&request).unwrap_err();
        assert!(error.contains("qwen_image_edit"));
        assert!(error.contains("does not implement"));
        assert!(still_calibration_label(&request).is_err());
        assert!(load_five_rung_generator(&request).is_err());
    }

    fn qwen_edit_planned(model_id: &str, tier: &str, overlay: &str, fixture: &str) -> Value {
        json!({
            "planned": {
                "target": {
                    "provider": "qwen_image_edit",
                    "modelId": model_id,
                    "tier": tier,
                    "mode": "edit_image",
                    "overlay": overlay,
                    "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
                },
                "backend": "candle",
                "loadShape": "deferred_materialization",
                "strategy": { "rung": "staged_residency", "engagedRungs": ["resident", "staged_residency"], "parameters": {} },
                "calibrationFingerprint": "unused",
                "fixture": fixture
            }
        })
    }

    /// sc-22728: the two shipped edit catalog ids are ONE engine provider, so only the model id
    /// separates them; the arm is resolved from `(provider, modelId)` and any other pair is refused
    /// by name rather than measured as its neighbour.
    #[test]
    fn the_candle_qwen_edit_arm_is_resolved_from_the_plans_provider_and_model_id() {
        let base = qwen_edit_arm(&qwen_edit_planned(
            "qwen_image_edit_2511",
            "q4",
            "none",
            "qwen-edit-candle-q4-seed15817-step2",
        ))
        .unwrap();
        assert_eq!(base, QWEN_EDIT_ARM);
        assert!(!base.lightning);
        assert_eq!(base.overlay, "none");
        let lightning = qwen_edit_arm(&qwen_edit_planned(
            "qwen_image_edit_2511_lightning",
            "q4",
            "lora",
            "qwen-edit-lightning-candle-q4-seed15817-step4",
        ))
        .unwrap();
        assert_eq!(lightning, QWEN_EDIT_LIGHTNING_ARM);
        assert!(lightning.lightning);
        assert_eq!(lightning.overlay, "lora");
        assert_eq!(lightning.steps, 4, "the official lightx2v 4-step recipe");
        assert_ne!(base.slug, lightning.slug, "one diagnostics source each");
        for (provider, model_id) in [
            ("qwen_image_edit", "qwen_image_edit_2509"),
            ("qwen_image", "qwen_image_edit_2511"),
        ] {
            let mut request = qwen_edit_planned(model_id, "q4", "none", "unused");
            request["planned"]["target"]["provider"] = json!(provider);
            let error = qwen_edit_arm(&request).unwrap_err();
            assert!(
                error.contains(&format!("provider {provider:?} for model {model_id:?}")),
                "{provider}/{model_id}: {error}"
            );
        }
    }

    /// sc-22728: the fixture binds the member, the tier and the recipe's step count, so a 4-step
    /// distilled capture can never be recorded under the 2-step production fixture or another tier's.
    #[test]
    fn the_candle_qwen_edit_fixture_binds_the_member_the_tier_and_the_step_count() {
        for (arm, model_id, prefix, steps) in [
            (
                QWEN_EDIT_ARM,
                "qwen_image_edit_2511",
                "qwen-edit-candle",
                2_usize,
            ),
            (
                QWEN_EDIT_LIGHTNING_ARM,
                "qwen_image_edit_2511_lightning",
                "qwen-edit-lightning-candle",
                4,
            ),
        ] {
            for tier in ["q4", "q8", "bf16"] {
                let fixture = format!("{prefix}-{tier}-seed15817-step{steps}");
                let request = qwen_edit_planned(model_id, tier, arm.overlay, &fixture);
                assert_eq!(planned_qwen_edit_seed(&request, arm, tier).unwrap(), 15817);
                let other = if tier == "q4" { "q8" } else { "q4" };
                let error = planned_qwen_edit_seed(&request, arm, other).unwrap_err();
                assert!(error.contains(&format!("{prefix}-{other}-seed")), "{error}");
            }
            let wrong_steps = if steps == 2 { 4 } else { 2 };
            let fixture = format!("{prefix}-q4-seed15817-step{wrong_steps}");
            let error = planned_qwen_edit_seed(
                &qwen_edit_planned(model_id, "q4", arm.overlay, &fixture),
                arm,
                "q4",
            )
            .unwrap_err();
            assert!(error.contains(&format!("{steps}-step")), "{error}");
        }
    }

    const QWEN_EDIT_TEST_REVISION: &str = "bb2bc9893b3c49ae96c813350775f791a2e8bc80";

    fn qwen_edit_scratch_dir(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("sc-22728-{label}-{}-{nonce}", std::process::id()))
    }

    /// A `models--<repo>/snapshots/<revision>/<tier>` root laid out exactly as the HF cache does, so
    /// the tier suffix the validator pins is real rather than mocked.
    fn qwen_edit_snapshot_root(tier: &str) -> PathBuf {
        let root = qwen_edit_scratch_dir("qwen-edit-candle-root")
            .join(format!(
                "models--{}",
                protocol::QWEN_EDIT_REPOSITORY.replace('/', "--")
            ))
            .join("snapshots")
            .join(QWEN_EDIT_TEST_REVISION)
            .join(tier);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// The distill snapshot, with the one pinned file actually on disk — the adapter helper refuses
    /// any other file name, so a fixture that skipped this would assert nothing.
    fn qwen_edit_lightning_fixture() -> QwenEditLightningSource {
        let root = qwen_edit_scratch_dir("qwen-edit-candle-lightning");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(protocol::QWEN_EDIT_LIGHTNING_FILE), b"distill").unwrap();
        QwenEditLightningSource {
            repository: protocol::QWEN_EDIT_LIGHTNING_REPOSITORY.to_owned(),
            revision: QWEN_EDIT_TEST_REVISION.to_owned(),
            root,
        }
    }

    fn qwen_edit_source(arm: QwenEditArm, tier: &str) -> QwenEditArtifactSource {
        QwenEditArtifactSource {
            repository: protocol::QWEN_EDIT_REPOSITORY.to_owned(),
            revision: QWEN_EDIT_TEST_REVISION.to_owned(),
            root: qwen_edit_snapshot_root(tier),
            lightning: arm.lightning.then(qwen_edit_lightning_fixture),
        }
    }

    const QWEN_EDIT_MEMBERS: [(QwenEditArm, &str); 2] = [
        (QWEN_EDIT_ARM, "qwen_image_edit_2511"),
        (QWEN_EDIT_LIGHTNING_ARM, "qwen_image_edit_2511_lightning"),
    ];

    /// sc-22728: the ONE thing that makes the Lightning member Lightning at load time. Every
    /// downstream claim — the record's `builtInAdapters` count, the gated fragment's overlay
    /// selection, and the run context's `overlay` (which `validate_edit_route` pins against the
    /// load's own adapter stack) — is read off `spec.adapters`, so if this attachment were lost the
    /// arm would publish an authoritative record asserting a distill that never participated. The
    /// stack itself is therefore asserted here, on BOTH members and at every shipped tier.
    #[test]
    fn the_candle_qwen_edit_load_spec_carries_the_distill_on_exactly_the_lightning_member() {
        for (arm, model_id) in QWEN_EDIT_MEMBERS {
            for (tier, quant) in [
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
                ("bf16", None),
            ] {
                let source = qwen_edit_source(arm, tier);
                let artifact =
                    qwen_edit_load_spec(arm, tier, &source, LoadShape::DeferredMaterialization)
                        .unwrap_or_else(|error| panic!("{model_id}/{tier}: {error}"));
                assert_eq!(
                    artifact.spec.quantize, quant,
                    "{model_id}/{tier} load quant"
                );
                assert_eq!(
                    artifact.spec.resolved_route.as_deref(),
                    Some(arm.model_id),
                    "{model_id}/{tier} resolved route"
                );
                // The worker's `provider_load_spec` for this lane never sets `offload_policy`, so
                // the capture must load under the gen-core default it leaves in place.
                assert_eq!(
                    artifact.spec.offload_policy,
                    OffloadPolicy::Resident,
                    "{model_id}/{tier} offload policy"
                );
                assert_eq!(
                    artifact.spec.adapters.len(),
                    usize::from(arm.lightning),
                    "{model_id}/{tier}: the distill stack must be exactly the member's"
                );
                if arm.lightning {
                    let expected = source
                        .lightning
                        .as_ref()
                        .unwrap()
                        .root
                        .canonicalize()
                        .unwrap()
                        .join(protocol::QWEN_EDIT_LIGHTNING_FILE);
                    let adapter = &artifact.spec.adapters[0];
                    assert_eq!(adapter.path, expected, "{model_id}/{tier} distill path");
                    assert!(
                        adapter.path.ends_with(protocol::QWEN_EDIT_LIGHTNING_FILE),
                        "{model_id}/{tier}: {}",
                        adapter.path.display()
                    );
                    assert_eq!(adapter.scale, 1.0, "{model_id}/{tier} distill scale");
                    assert!(
                        matches!(adapter.kind, AdapterKind::Lora),
                        "{model_id}/{tier}: {:?}",
                        adapter.kind
                    );
                }
            }
        }
    }

    /// sc-22728: the root the capture opens must end in the PLANNED tier's directory, per member —
    /// a stale `…/q4` export cannot satisfy a q8 plan and quietly re-label another tier's peaks.
    #[test]
    fn a_candle_qwen_edit_root_of_another_tier_is_refused_naming_the_planned_tier() {
        for (arm, model_id) in QWEN_EDIT_MEMBERS {
            // The plan is q8; the root on disk is the q4 export.
            let source = qwen_edit_source(arm, "q4");
            let error = qwen_edit_load_spec(arm, "q8", &source, LoadShape::DeferredMaterialization)
                .expect_err("a q8 plan must not be satisfied by a q4 root");
            assert!(
                error.ends_with(&format!("/snapshots/{QWEN_EDIT_TEST_REVISION}/q8")),
                "{model_id}: {error}"
            );
            // The wrong artifact family is refused before the root is even looked at.
            let mut foreign = qwen_edit_source(arm, "q8");
            foreign.repository = protocol::QWEN_EDIT_LIGHTNING_REPOSITORY.to_owned();
            let error =
                qwen_edit_load_spec(arm, "q8", &foreign, LoadShape::DeferredMaterialization)
                    .expect_err("the distill repository is not the base artifact");
            assert!(
                error.contains(protocol::QWEN_EDIT_REPOSITORY),
                "{model_id}: {error}"
            );
        }
    }

    /// sc-22728: the reference is one interleaved RGB frame at the request geometry — the shape the
    /// engine hard-requires and the worker always supplies.
    #[test]
    fn the_candle_qwen_edit_reference_is_one_frame_at_the_request_geometry() {
        let reference = qwen_edit_reference(1024, 768);
        assert_eq!((reference.width, reference.height), (1024, 768));
        assert_eq!(reference.pixels.len(), 1024 * 768 * 3);
    }

    #[test]
    fn deferred_materialization_establishes_retention_baseline_after_resident_rung() {
        let mut baseline = None;
        update_warmed_retention_baseline(&mut baseline, 12 * GIB).unwrap();
        assert_eq!(baseline, Some(12 * GIB));
        update_warmed_retention_baseline(&mut baseline, 12 * GIB + 64 * MIB).unwrap();
    }

    #[test]
    fn warmed_retention_baseline_rejects_later_growth() {
        let mut baseline = Some(12 * GIB);
        let error =
            update_warmed_retention_baseline(&mut baseline, 12 * GIB + 64 * MIB + 1).unwrap_err();
        assert!(error.contains("above the warmed resident baseline"));
    }

    /// A single planned case at `frames`, minimal enough that the geometry guard is the FIRST thing
    /// that can reject it — no weight root, no environment.
    ///
    /// `fixture` is LOAD-BEARING here, not decoration. [`run`] picks its dispatch branch from it: a
    /// `fresh-five-rung-` prefix routes into [`run_five_rung_reference`], and ANY OTHER fixture on
    /// `krea_2_turbo` falls through to the inline Krea arm instead. A table that only ever passes
    /// one prefix therefore exercises only one of the two branches, which is precisely how this
    /// test shipped blind to [`run`]'s own guard once (sc-18808 re-review).
    fn still_planned_case_with_fixture(
        provider: &str,
        rung: &str,
        frames: u64,
        fixture: &str,
    ) -> Value {
        json!({
            "backend": "candle",
            "target": {
                "provider": provider,
                "modelId": provider,
                "tier": "q4",
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": frames }
            },
            "loadShape": "deferred_materialization",
            "strategy": { "rung": rung, "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": fixture
        })
    }

    /// The five-rung shape — the only one [`run_five_rung_batch`] accepts.
    fn still_planned_case(provider: &str, rung: &str, frames: u64) -> Value {
        still_planned_case_with_fixture(provider, rung, frames, "fresh-five-rung-unused")
    }

    /// The same shape with the CATALOG model id spelled independently of the provider — the axis
    /// the two klein models differ on (sc-22727).
    fn still_planned_case_for(provider: &str, model_id: &str, rung: &str, frames: u64) -> Value {
        let mut planned = still_planned_case(provider, rung, frames);
        planned["target"]["modelId"] = json!(model_id);
        planned
    }

    /// The same shape in a declared plan mode. `edit_image` on the Turbo provider is the
    /// `z_image_edit` route (sc-22724), which is its own execution path with its own refusal label.
    fn still_planned_case_in_mode(provider: &str, rung: &str, frames: u64, mode: &str) -> Value {
        let mut planned = still_planned_case(provider, rung, frames);
        planned["target"]["mode"] = json!(mode);
        planned
    }

    /// The canonical five-rung batch shape `run_five_rung_batch` requires, at `frames` and `mode`.
    fn still_batch_request_in_mode(provider: &str, frames: u64, mode: &str) -> Value {
        let planned: Vec<Value> = [
            "resident",
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        ]
        .into_iter()
        .map(|rung| still_planned_case_in_mode(provider, rung, frames, mode))
        .collect();
        json!({ "action": "run_batch", "planned": planned })
    }

    /// sc-18808 — the Candle twin of the MLX adapter's
    /// `every_image_arm_still_refuses_a_multi_frame_geometry`.
    ///
    /// BOTH Candle arms hardcoded `frames: 1` into `MemoryGeometry` while reading only
    /// `width`/`height` from the plan, so a plan row declaring any other frame count would have
    /// rendered ONE frame and emitted a record claiming a geometry it was never asked for.
    ///
    /// Two entry points are reachable from `main`, and both must refuse with the exact pinned
    /// wording before any environment or weight work:
    ///
    /// * `run` — the dispatcher. Its guard stands in front of BOTH of its branches, and the branch
    ///   is chosen by `planned.fixture`, which is why the table below carries the fixture instead of
    ///   hardcoding one. A `fresh-five-rung-` prefix (or the `qwen_image` provider) routes into
    ///   `run_five_rung_reference`; ANY OTHER fixture on `krea_2_turbo` falls through to the INLINE
    ///   Krea arm — the arm that resolves `SCENEWORKS_KREA_REPOSITORY` and then writes its own
    ///   `MemoryGeometry { frames: 1 }`. Five SHIPPED Candle plan rows carry the second shape
    ///   (`krea-q4-1024-seed42` and its q8/bf16/768/v2 siblings), so it is the live one; the third
    ///   row below is one of them verbatim. Until it was added, every case in this table began
    ///   `fresh-five-rung-`, so all of them short-circuited into `run_five_rung_reference` and
    ///   `run`'s own guard was shadowed by the redundant copy at the head of that function —
    ///   deleting `run`'s guard left this test green.
    /// * `run_five_rung_batch` — reached straight from `main`, so `run`'s guard never sees it. Its
    ///   per-item pre-load loop is the guard under test; the fixture is irrelevant to it, so the
    ///   canonical five-rung shape is the only one it is exercised with.
    ///
    /// The other two copies of the refusal are redundant defense-in-depth and are NOT what this
    /// test pins: the one at the head of `run_five_rung_reference` (whose only caller is `run`,
    /// which already refused) and the one in `run_five_rung_reference_loaded` (reachable only after
    /// a real generator load, so no unit test can enter it). Removing either of those alone leaves
    /// this suite green — which is exactly why a future reader must not "clean up" `run`'s or
    /// `run_five_rung_batch`'s on the strength of them still being there.
    #[test]
    fn every_candle_arm_still_refuses_a_multi_frame_geometry() {
        for (provider, model_id, mode, label, fixture) in [
            (
                QWEN_ID,
                QWEN_ID,
                "text_to_image",
                QWEN_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            (
                KREA_ID,
                KREA_ID,
                "text_to_image",
                KREA_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            (
                Z_IMAGE_TURBO_ID,
                Z_IMAGE_TURBO_ID,
                "text_to_image",
                Z_IMAGE_TURBO_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            // The `z_image_edit` route is its own execution path and refuses under its own label
            // (sc-22724): the two Turbo rows must not report the same sentence.
            (
                Z_IMAGE_TURBO_ID,
                Z_IMAGE_TURBO_ID,
                "edit_image",
                Z_IMAGE_TURBO_EDIT_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            (
                Z_IMAGE_ID,
                Z_IMAGE_ID,
                "text_to_image",
                Z_IMAGE_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            // sc-22727: three FLUX.2 catalog models over two registry ids. The two klein rows share
            // `flux2_klein_9b` and must NOT report the same sentence.
            (
                FLUX2_DEV_ID,
                "flux2_dev",
                "text_to_image",
                FLUX2_DEV_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            (
                FLUX2_KLEIN_ID,
                "flux2_klein_9b",
                "text_to_image",
                FLUX2_KLEIN_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            (
                FLUX2_KLEIN_ID,
                "flux2_klein_9b_kv",
                "text_to_image",
                FLUX2_KLEIN_KV_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            // The inline Krea arm — a real shipped plan fixture, which the rows above cannot
            // reach.
            (
                KREA_ID,
                KREA_ID,
                "text_to_image",
                KREA_STILL_CALIBRATION,
                "krea-q4-1024-seed42",
            ),
        ] {
            for frames in [0_u64, 2, 97] {
                let expected = format!("{label} requires geometry.frames == 1, got {frames}");
                let mut planned =
                    still_planned_case_with_fixture(provider, "resident", frames, fixture);
                planned["target"]["modelId"] = json!(model_id);
                planned["target"]["mode"] = json!(mode);
                let request = json!({ "action": "run", "planned": planned });
                assert_eq!(
                    run(&request).expect_err("the Candle dispatcher must refuse a video geometry"),
                    expected,
                    "run: {provider}/{model_id}/{mode} at frames={frames} via fixture {fixture:?}"
                );
            }
        }
        for (provider, mode, label) in [
            (QWEN_ID, "text_to_image", QWEN_STILL_CALIBRATION),
            (KREA_ID, "text_to_image", KREA_STILL_CALIBRATION),
            (
                Z_IMAGE_TURBO_ID,
                "text_to_image",
                Z_IMAGE_TURBO_STILL_CALIBRATION,
            ),
            (
                Z_IMAGE_TURBO_ID,
                "edit_image",
                Z_IMAGE_TURBO_EDIT_STILL_CALIBRATION,
            ),
            (Z_IMAGE_ID, "text_to_image", Z_IMAGE_STILL_CALIBRATION),
        ] {
            for frames in [0_u64, 2, 97] {
                let expected = format!("{label} requires geometry.frames == 1, got {frames}");
                assert_eq!(
                    run_five_rung_batch(&still_batch_request_in_mode(provider, frames, mode))
                        .expect_err("the Candle batch arm must refuse a video geometry"),
                    expected,
                    "run_batch: {provider}/{mode} at frames={frames}"
                );
            }
        }
    }

    /// The third row above is not decoration: the fixtures the SHIPPED Candle plan gives the inline
    /// Krea arm really do take that branch, so `run`'s own guard — not the redundant copy inside
    /// [`run_five_rung_reference`] — is the one standing in front of them.
    ///
    /// Asserted against the dispatch predicate itself rather than by observing an error, because
    /// both branches resolve the same `SCENEWORKS_KREA_REPOSITORY` and would report the same
    /// sentence: the error is not a routing witness, the predicate is. Widening the prefix (or
    /// renaming these fixtures into it) would re-shadow `run`'s guard, and this reds when it does.
    #[test]
    fn the_shipped_krea_fixtures_take_the_inline_arm_not_the_five_rung_branch() {
        for fixture in [
            "krea-q4-1024-seed42",
            "krea-q8-1024-seed42",
            "krea-bf16-1024-seed42",
            "krea-q4-768-seed42",
            "krea-q4-1024-seed42-v2-candidate",
        ] {
            let request = json!({
                "planned": still_planned_case_with_fixture(KREA_ID, "resident", 1, fixture)
            });
            assert!(
                !routes_to_five_rung_reference(&request).unwrap(),
                "{fixture} must reach the inline Krea arm"
            );
        }
        for (provider, fixture) in [
            (KREA_ID, "fresh-five-rung-krea-q4-1024-seed16402-step2"),
            (QWEN_ID, "qwen-image-candle-q4-seed15817-step2"),
            (
                Z_IMAGE_TURBO_ID,
                "fresh-five-rung-z-image-turbo-q4-1024-seed16402-step2",
            ),
            // No inline arm exists for Z-Image-Turbo, so an off-prefix fixture still routes here.
            (Z_IMAGE_TURBO_ID, "z-image-turbo-any-other-fixture"),
            // sc-22724: nor for the base, whose shipped fixtures carry the sc-16170 spelling.
            (
                Z_IMAGE_ID,
                "sc-16170-z-image-q4-1024-text_to_image-none-seed16170",
            ),
        ] {
            let request = json!({
                "planned": still_planned_case_with_fixture(provider, "resident", 1, fixture)
            });
            assert!(
                routes_to_five_rung_reference(&request).unwrap(),
                "{fixture} must reach the five-rung reference path"
            );
        }
    }

    /// sc-22727: the FLUX.2 family is three catalog models over two registry ids, and each member
    /// carries its OWN execution path, refusal label, artifact family and diagnostics slug. Two
    /// members share `flux2_klein_9b`, so `modelId` — never `provider` — is the discriminator, and
    /// a pair no member serves is refused by name rather than measured as its nearest neighbour.
    #[test]
    fn the_candle_flux2_family_is_resolved_from_the_plans_provider_and_model_id() {
        for (provider, model_id, expected) in [
            (FLUX2_DEV_ID, "flux2_dev", FLUX2_DEV_ARM),
            (FLUX2_KLEIN_ID, "flux2_klein_9b", FLUX2_KLEIN_ARM),
            (FLUX2_KLEIN_ID, "flux2_klein_9b_kv", FLUX2_KLEIN_KV_ARM),
        ] {
            let request =
                json!({ "planned": still_planned_case_for(provider, model_id, "resident", 1) });
            assert_eq!(flux2_arm(&request).unwrap(), Some(expected));
            assert_eq!(
                plain_execution_path(&request).unwrap(),
                expected.execution_path
            );
            assert_eq!(
                still_calibration_label(&request).unwrap(),
                expected.still_calibration
            );
            // No inline arm exists for FLUX.2, so every fixture routes to the five-rung path.
            assert!(routes_to_five_rung_reference(&request).unwrap());
        }
        for (provider, model_id) in [
            (FLUX2_DEV_ID, "flux2_klein_9b"),
            (FLUX2_KLEIN_ID, "flux2_dev"),
            // A real catalog model on the klein provider that this adapter does NOT serve: its
            // snapshot is an assembled convert dir, not a tiered rehost.
            (FLUX2_KLEIN_ID, "flux2_klein_9b_true_v2"),
        ] {
            let request =
                json!({ "planned": still_planned_case_for(provider, model_id, "resident", 1) });
            let error = flux2_arm(&request).expect_err("an unserved pair must be refused by name");
            assert!(
                error.contains(&format!("provider {provider:?} for model {model_id:?}")),
                "{provider}/{model_id}: {error}"
            );
            // And the refusal reaches the callers rather than being swallowed into a default path.
            assert_eq!(plain_execution_path(&request).unwrap_err(), error);
            assert_eq!(still_calibration_label(&request).unwrap_err(), error);
        }
        // A non-FLUX.2 plan resolves no member at all, and says so without erroring.
        assert_eq!(
            flux2_arm(&json!({ "planned": still_planned_case(Z_IMAGE_ID, "resident", 1) }))
                .unwrap(),
            None
        );
        // Every member is distinguishable from every other on every identity axis: a collision
        // would let one artifact satisfy another's plan, or make two records indistinguishable.
        for field in [
            FLUX2_ARMS.map(|arm| arm.model_id),
            FLUX2_ARMS.map(|arm| arm.execution_path),
            FLUX2_ARMS.map(|arm| arm.still_calibration),
            FLUX2_ARMS.map(|arm| arm.repository_env),
            FLUX2_ARMS.map(|arm| arm.revision_env),
            FLUX2_ARMS.map(|arm| arm.root_env),
            FLUX2_ARMS.map(|arm| arm.expected_repository),
            FLUX2_ARMS.map(|arm| arm.slug),
        ] {
            let mut unique = field.to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), FLUX2_ARMS.len(), "collision in {field:?}");
        }
        assert_eq!(FLUX2_KLEIN_ARM.provider, FLUX2_KLEIN_KV_ARM.provider);
        assert_eq!(
            FLUX2_KLEIN_ARM.expected_repository,
            protocol::FLUX2_KLEIN_REPOSITORY
        );
        assert_eq!(
            FLUX2_KLEIN_KV_ARM.expected_repository,
            protocol::FLUX2_KLEIN_KV_REPOSITORY
        );
        // Which member hands the planned tier to the loader, stated as data (sc-22727 review):
        // only dev folds it; both klein turnkeys are dense-TE tiers the worker loads with
        // `Quant::None`, and candle-gen-flux2 would otherwise re-quantize their packed DiT.
        assert_eq!(
            FLUX2_ARMS.map(|arm| arm.tier_quant_reaches_the_loader),
            [true, false, false],
            "only the dev route folds the planned tier into LoadSpec::quantize"
        );
        // ...and bound to the manifest the worker reads that decision from: `is_dense_te_tier` is
        // exactly `mlx.denseTextEncoderTier == true`, so the flag must be its negation per member.
        let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
            include_str!("../../../../config/manifests/builtin.models.jsonc"),
        ))
        .expect("the shipped models manifest parses");
        for arm in FLUX2_ARMS {
            let entry = manifest["models"]
                .as_array()
                .expect("models")
                .iter()
                .find(|entry| entry["id"] == arm.model_id)
                .unwrap_or_else(|| panic!("{} is not a shipped model", arm.model_id));
            let dense_te = entry["mlx"]["denseTextEncoderTier"] == json!(true);
            assert_eq!(
                arm.tier_quant_reaches_the_loader, !dense_te,
                "{}: the worker loads a dense-TE tier with Quant::None (is_dense_te_tier)",
                arm.model_id
            );
        }
    }

    /// sc-22724: the `z_image_edit` route is the Turbo provider in `edit_image` mode — the same
    /// loader, a distinct execution path, and one reference on the request — and only the Turbo
    /// arm has that second mode.
    #[test]
    fn the_z_image_edit_route_is_the_turbo_arm_in_edit_mode() {
        let mut edit = still_planned_case(Z_IMAGE_TURBO_ID, "resident", 1);
        edit["target"]["mode"] = json!("edit_image");
        edit["target"]["modelId"] = json!("z_image_edit");
        let request = json!({ "planned": edit });
        assert!(is_z_image_edit(&request).unwrap());
        assert_eq!(
            plain_execution_path(&request).unwrap(),
            Z_IMAGE_TURBO_EDIT_EXECUTION_PATH
        );
        assert!(routes_to_five_rung_reference(&request).unwrap());
        let plain = json!({ "planned": still_planned_case(Z_IMAGE_TURBO_ID, "resident", 1) });
        assert!(!is_z_image_edit(&plain).unwrap());
        assert_eq!(
            plain_execution_path(&plain).unwrap(),
            Z_IMAGE_TURBO_PLAIN_EXECUTION_PATH
        );
        // The base has no edit route; its mode does not change its path.
        let mut base = still_planned_case(Z_IMAGE_ID, "resident", 1);
        base["target"]["mode"] = json!("edit_image");
        let base = json!({ "planned": base });
        assert!(!is_z_image_edit(&base).unwrap());
        assert_eq!(
            plain_execution_path(&base).unwrap(),
            Z_IMAGE_PLAIN_EXECUTION_PATH
        );
    }

    /// sc-22724: the edit request is the worker's — one reference at the target geometry plus the
    /// production strength — and the text-to-image request carries none.
    #[test]
    fn the_edit_generation_request_carries_one_reference_at_the_target_geometry() {
        let edit = five_rung_generation_request(1024, 768, true);
        assert_eq!(edit.conditioning.len(), 1);
        match &edit.conditioning[0] {
            Conditioning::Reference { image, strength } => {
                assert_eq!((image.width, image.height), (1024, 768));
                assert_eq!(image.pixels.len(), 1024 * 768 * 3);
                assert_eq!(*strength, Some(Z_IMAGE_EDIT_STRENGTH));
            }
            other => panic!("expected one Reference, got {other:?}"),
        }
        // The worker sets ONLY the per-reference strength (`build_lane_conditioning`,
        // image_jobs/base.rs:7136); the request-level lever stays unset, and so does this arm's.
        assert_eq!(edit.strength, None);
        // floor(4 * 0.6) = 2, so two executed denoise steps remain behind the conditioning
        // boundary — the engine's `init_time_step` law, not this arm's.
        assert_eq!(edit.steps, Some(Z_IMAGE_EDIT_STEPS));
        let plain = five_rung_generation_request(1024, 1024, false);
        assert!(plain.conditioning.is_empty());
        assert_eq!(plain.strength, None);
        assert_eq!(plain.steps, Some(2));
    }

    /// And the guard is the frames axis rather than a blanket rejection: the same still geometry
    /// passes it on both Candle labels, so the refusals above cannot be an unconditional error.
    #[test]
    fn the_candle_still_geometry_guard_is_not_a_blanket_refusal() {
        for (provider, model_id) in [
            (QWEN_ID, QWEN_ID),
            (KREA_ID, KREA_ID),
            (Z_IMAGE_TURBO_ID, Z_IMAGE_TURBO_ID),
            (Z_IMAGE_ID, Z_IMAGE_ID),
            (FLUX2_DEV_ID, "flux2_dev"),
            (FLUX2_KLEIN_ID, "flux2_klein_9b"),
            (FLUX2_KLEIN_ID, "flux2_klein_9b_kv"),
            (FLUX1_DEV_ID, FLUX1_DEV_ID),
            (FLUX1_SCHNELL_ID, FLUX1_SCHNELL_ID),
            (PULID_FLUX_ID, PULID_FLUX_ID),
        ] {
            let request =
                json!({ "planned": still_planned_case_for(provider, model_id, "resident", 1) });
            let label = still_calibration_label(&request).unwrap();
            protocol::validate_still_geometry(&request, label)
                .unwrap_or_else(|error| panic!("{provider}: {error}"));
        }
    }

    // -----------------------------------------------------------------------------------------
    // sc-22730 — the SD3.5 family on the Candle lane.
    // -----------------------------------------------------------------------------------------

    /// All three SD3.5 members reach the shared five-rung reference path under their OWN execution
    /// paths and refusal labels. None has an inline arm, so an off-prefix fixture must still route
    /// here — proving the provider-id branch does it, not the fixture-prefix branch.
    #[test]
    fn the_sd3_5_members_route_to_the_five_rung_reference() {
        for (provider, path, label) in [
            (
                SD3_5_LARGE_ID,
                SD3_5_LARGE_PLAIN_EXECUTION_PATH,
                SD3_5_LARGE_STILL_CALIBRATION,
            ),
            (
                SD3_5_LARGE_TURBO_ID,
                SD3_5_LARGE_TURBO_PLAIN_EXECUTION_PATH,
                SD3_5_LARGE_TURBO_STILL_CALIBRATION,
            ),
            (
                SD3_5_MEDIUM_ID,
                SD3_5_MEDIUM_PLAIN_EXECUTION_PATH,
                SD3_5_MEDIUM_STILL_CALIBRATION,
            ),
        ] {
            let request = json!({
                "planned": still_planned_case_with_fixture(provider, "staged_residency", 1, "sc-22730-off-prefix")
            });
            assert!(
                routes_to_five_rung_reference(&request).unwrap(),
                "{provider}"
            );
            assert_eq!(plain_execution_path(&request).unwrap(), path, "{provider}");
            assert_eq!(
                still_calibration_label(&request).unwrap(),
                label,
                "{provider}"
            );
            assert_eq!(five_rung_evidence_story(provider), "sc-22730", "{provider}");
        }
    }

    /// Every member refuses a non-still geometry under its OWN label, before any env or weight
    /// work — the sc-18808 guard, on this family.
    #[test]
    fn every_candle_sd3_5_member_refuses_a_multi_frame_geometry() {
        for (provider, label) in [
            (SD3_5_LARGE_ID, SD3_5_LARGE_STILL_CALIBRATION),
            (SD3_5_LARGE_TURBO_ID, SD3_5_LARGE_TURBO_STILL_CALIBRATION),
            (SD3_5_MEDIUM_ID, SD3_5_MEDIUM_STILL_CALIBRATION),
        ] {
            for frames in [0_u64, 2, 97] {
                let request =
                    json!({ "planned": still_planned_case(provider, "staged_residency", frames) });
                let error = run(&request).expect_err("a non-still geometry must be refused");
                assert_eq!(
                    error,
                    format!("{label} requires geometry.frames == 1, got {frames}"),
                    "{provider}"
                );
            }
        }
    }

    /// The fixture binds member, tier, geometry edge, seed and step count, so a Medium q8 record
    /// can never be attributed to a Large bf16 capture that merely reused the string.
    #[test]
    fn the_candle_sd3_5_fixture_binds_member_tier_edge_seed_and_steps() {
        let case = |provider: &str, tier: &str, fixture: &str| {
            let mut planned =
                still_planned_case_with_fixture(provider, "staged_residency", 1, fixture);
            planned["target"]["tier"] = json!(tier);
            json!({ "planned": planned })
        };
        for (provider, slug) in SD3_5_FIXTURE_SLUGS {
            for tier in ["q4", "q8", "bf16"] {
                let good = format!("fresh-five-rung-{slug}-{tier}-1024-seed{FIVE_RUNG_SEED}-step2");
                validate_fixture_binds_tier_and_geometry(&case(provider, tier, &good))
                    .unwrap_or_else(|error| panic!("{provider} {tier}: {error}"));

                // Another tier's token, another edge, the MLX arm's seed, and a step count this
                // arm never renders are each refused by name.
                for (bad, expected) in [
                    (
                        format!("fresh-five-rung-{slug}-{tier}-768-seed{FIVE_RUNG_SEED}-step2"),
                        "must start with",
                    ),
                    (
                        format!("fresh-five-rung-{slug}-{tier}-1024-seed22730-step2"),
                        "does not match the seed",
                    ),
                    (
                        format!("fresh-five-rung-{slug}-{tier}-1024-seed{FIVE_RUNG_SEED}-step3"),
                        "two-step",
                    ),
                    (
                        format!("{slug}-{tier}-1024-seed{FIVE_RUNG_SEED}-step2"),
                        "must start with",
                    ),
                ] {
                    let error =
                        validate_fixture_binds_tier_and_geometry(&case(provider, tier, &bad))
                            .expect_err(&format!("{provider} {tier} {bad} must be refused"));
                    assert!(
                        error.contains(expected),
                        "{provider} {tier} {bad}: {error} lacks {expected:?}"
                    );
                }

                // A SIBLING member's fixture is refused: the slug is part of the prefix.
                for (other, other_slug) in SD3_5_FIXTURE_SLUGS {
                    if other == provider {
                        continue;
                    }
                    let crossed = format!(
                        "fresh-five-rung-{other_slug}-{tier}-1024-seed{FIVE_RUNG_SEED}-step2"
                    );
                    // `sd3-5-large` is a PREFIX of `sd3-5-large-turbo`, so the Large binding would
                    // accept the Turbo fixture on a naive `starts_with`; the tier token that
                    // follows the slug is what actually separates them.
                    assert!(
                        validate_fixture_binds_tier_and_geometry(&case(provider, tier, &crossed))
                            .is_err(),
                        "{provider} {tier} must refuse {other}'s fixture {crossed}"
                    );
                }
            }
        }
        // An unknown provider is refused by name rather than silently skipped.
        let error = validate_sd35_fixture(
            &case("sd3_5_enormous", "q4", "fresh-five-rung-x"),
            "sd3_5_enormous",
            "q4",
        )
        .expect_err("an unknown SD3.5 member must be refused");
        assert!(error.contains("sd3_5_enormous"), "{error}");
    }

    /// Every SD3.5 candle cell the committed plan declares is served by an arm: an execution path,
    /// a refusal label, a tier, a fixture that satisfies the binding, and the five-rung route.
    /// The exact cell set is asserted, not a count.
    #[test]
    fn every_planned_sd3_5_candle_cell_is_served_by_an_arm() {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let mut seen = std::collections::BTreeSet::new();
        for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
            if !key.ends_with(":candle") {
                continue;
            }
            let provider = entry["provider"].as_str().unwrap();
            if !SD3_5_FIXTURE_SLUGS.iter().any(|(id, _)| *id == provider) {
                continue;
            }
            seen.insert(key.clone());
            let (_, rest) = key.split_once(':').unwrap();
            let tier = rest.split_once(':').unwrap().0;
            let request = json!({ "planned": {
                "backend": "candle",
                "target": {
                    "provider": provider,
                    "tier": tier,
                    "mode": entry["mode"].clone(),
                    "overlay": entry["overlay"].clone(),
                    "geometry": entry["geometry"].clone(),
                },
                "loadShape": entry["loadShape"].clone(),
                "strategy": { "rung": "staged_residency", "parameters": {} },
                "calibrationFingerprint": entry["calibrationFingerprint"].clone(),
                "fixture": entry["fixture"].clone(),
            }});
            plain_execution_path(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
            still_calibration_label(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
            assert_eq!(planned_tier(&request).unwrap(), tier, "{key}");
            validate_fixture_binds_tier_and_geometry(&request)
                .unwrap_or_else(|error| panic!("{key}: {error}"));
            assert!(routes_to_five_rung_reference(&request).unwrap(), "{key}");
            // The candle anchor is the SHALLOW STAGED composition the derivation law prices
            // (`extract-memory-anchors.mjs` `isDerivable`), which the manifest declares under
            // `requiredOffloadPolicy: sequential` — the shape `load_five_rung_generator` builds.
            assert_eq!(
                entry["loadShape"].as_str().unwrap(),
                "deferred_materialization",
                "{key}"
            );
        }
        let expected: std::collections::BTreeSet<String> =
            ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]
                .iter()
                .flat_map(|model| {
                    ["bf16", "q4", "q8"]
                        .iter()
                        .map(move |tier| format!("{model}:{tier}:candle"))
                })
                .collect();
        assert_eq!(seen, expected);
    }

    // -----------------------------------------------------------------------------------------
    // sc-22726 — the FLUX.1 family on the Candle lane.
    // -----------------------------------------------------------------------------------------

    fn flux_one_temp_dir(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sc-22726-candle-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn flux_one_snapshot_root(repository: &str, revision: &str, tier: &str) -> PathBuf {
        let root = flux_one_temp_dir("flux1")
            .join(format!("models--{}", repository.replace('/', "--")))
            .join("snapshots")
            .join(revision)
            .join(tier);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn staged_pulid_bundle() -> protocol::PulidIdentityBundle {
        let root = flux_one_temp_dir("pulid-bundle");
        std::fs::create_dir_all(&root).unwrap();
        for file in protocol::PULID_IDENTITY_BUNDLE_FILES {
            std::fs::write(root.join(file), b"weights").unwrap();
        }
        protocol::pulid_identity_bundle_at(root).unwrap()
    }

    /// The two FLUX.1 BASE providers reach the shared five-rung reference path under their own
    /// execution paths and refusal labels; PuLID deliberately does NOT, because it is a bespoke
    /// route that never touches the provider registry.
    #[test]
    fn the_flux_one_base_providers_route_to_the_five_rung_reference_and_pulid_does_not() {
        for (provider, path, label) in [
            (
                FLUX1_DEV_ID,
                FLUX1_DEV_PLAIN_EXECUTION_PATH,
                FLUX1_DEV_STILL_CALIBRATION,
            ),
            (
                FLUX1_SCHNELL_ID,
                FLUX1_SCHNELL_PLAIN_EXECUTION_PATH,
                FLUX1_SCHNELL_STILL_CALIBRATION,
            ),
        ] {
            // An off-prefix fixture must still route here: these providers have no inline arm.
            let request = json!({
                "planned": still_planned_case_with_fixture(provider, "staged_residency", 1, "sc-22726-off-prefix")
            });
            assert!(
                routes_to_five_rung_reference(&request).unwrap(),
                "{provider}"
            );
            assert_eq!(plain_execution_path(&request).unwrap(), path);
            assert_eq!(still_calibration_label(&request).unwrap(), label);
        }
        let pulid = json!({
            "planned": still_planned_case_with_fixture(PULID_FLUX_ID, "staged_residency", 1, "sc-22726-pulid")
        });
        assert!(
            !routes_to_five_rung_reference(&pulid).unwrap(),
            "PuLID must not be served by the registry five-rung path"
        );
        assert_eq!(
            plain_execution_path(&pulid).unwrap(),
            PULID_FLUX_EXECUTION_PATH
        );
        assert_eq!(
            still_calibration_label(&pulid).unwrap(),
            PULID_FLUX_STILL_CALIBRATION
        );
    }

    /// Every FLUX.1 member still refuses a video geometry under its OWN label, before it resolves
    /// an environment variable or opens a snapshot (sc-18808). PuLID goes through its own
    /// dispatch, so it is checked through `run` too.
    #[test]
    fn every_candle_flux_one_member_refuses_a_multi_frame_geometry() {
        for (provider, mode, label) in [
            (FLUX1_DEV_ID, "text_to_image", FLUX1_DEV_STILL_CALIBRATION),
            (
                FLUX1_SCHNELL_ID,
                "text_to_image",
                FLUX1_SCHNELL_STILL_CALIBRATION,
            ),
            (
                PULID_FLUX_ID,
                "character_image",
                PULID_FLUX_STILL_CALIBRATION,
            ),
        ] {
            for frames in [0_u64, 2, 97] {
                let mut planned = still_planned_case_in_mode(provider, "resident", frames, mode);
                if provider == PULID_FLUX_ID {
                    planned["target"]["overlay"] = json!("identity");
                }
                let error = run(&json!({ "action": "run", "planned": planned }))
                    .expect_err("a video geometry must be refused");
                assert_eq!(
                    error,
                    format!("{label} requires geometry.frames == 1, got {frames}"),
                    "{provider}/{frames}"
                );
            }
        }
    }

    /// The PuLID arm binds the FLUX.1-dev backbone at the PLANNED tier and the staged identity
    /// stack into the exact `PulidFluxPaths` the worker builds — a q8 plan against a q4 export is
    /// refused naming the tier, and all three tiers round-trip.
    #[test]
    fn the_candle_pulid_binding_carries_the_planned_tier_and_the_identity_stack() {
        const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
        let pulid_case = |tier: &str| {
            let mut planned =
                still_planned_case_in_mode(PULID_FLUX_ID, "staged_residency", 1, "character_image");
            planned["target"]["tier"] = json!(tier);
            planned["target"]["overlay"] = json!("identity");
            planned["target"]["modelId"] = json!("pulid_flux_dev");
            planned["fixture"] = json!(format!(
                "pulid-flux-candle-{tier}-1024-seed{FLUX1_SEED}-step2"
            ));
            json!({ "planned": planned })
        };
        let q4_root = flux_one_snapshot_root(protocol::FLUX1_DEV_REPOSITORY, REVISION, "q4");
        let error = pulid_flux_binding_at(
            &pulid_case("q8"),
            protocol::FLUX1_DEV_REPOSITORY.to_owned(),
            REVISION.to_owned(),
            q4_root.clone(),
            staged_pulid_bundle(),
        )
        .expect_err("a q8 plan must not be satisfied by a q4 root");
        assert!(
            error.ends_with(&format!("/snapshots/{REVISION}/q8")),
            "{error}"
        );

        for tier in ["q4", "q8", "bf16"] {
            let bundle = staged_pulid_bundle();
            let root = flux_one_snapshot_root(protocol::FLUX1_DEV_REPOSITORY, REVISION, tier);
            let binding = pulid_flux_binding_at(
                &pulid_case(tier),
                protocol::FLUX1_DEV_REPOSITORY.to_owned(),
                REVISION.to_owned(),
                root,
                bundle.clone(),
            )
            .unwrap_or_else(|error| panic!("{tier}: {error}"));
            assert_eq!(binding.tier, tier);
            assert!(binding.paths.flux_base.ends_with(tier));
            assert_eq!(binding.paths.pulid_weights, bundle.adapter);
            assert_eq!(binding.paths.eva_weights, bundle.eva);
            // The engine reads scrfd / arcface / bisenet out of `face_dir` BY NAME, so the bundle
            // root IS the face dir.
            assert_eq!(binding.paths.face_dir, bundle.root);
            // A ladder-admitted PuLID render carries no LoRA (`pulid_memory_ladder_eligible`).
            assert!(binding.paths.adapters.is_empty());
            assert!(binding.loadability_fingerprint().starts_with(&format!(
                "{}@{REVISION}:{tier}",
                protocol::FLUX1_DEV_REPOSITORY
            )));
            // By CONTENT, never by the host path the bundle was staged at.
            assert!(binding
                .loadability_fingerprint()
                .ends_with(&format!("+identity:{}", bundle.composite_sha256)));
            assert!(!binding
                .loadability_fingerprint()
                .contains(&bundle.root.display().to_string()));
            let artifact = binding.artifact_json();
            assert_eq!(artifact["variant"].as_str(), Some(tier));
            assert_eq!(
                artifact["identityBundle"]["compositeSha256"].as_str(),
                Some(bundle.composite_sha256.as_str())
            );
            for (file, sha256) in &bundle.file_sha256 {
                assert_eq!(
                    artifact["identityBundle"]["files"][*file].as_str(),
                    Some(sha256.as_str())
                );
            }
        }
        // A fixture naming the five-rung seed is refused: PuLID renders at FLUX1_SEED.
        let mut wrong_seed = pulid_case("q4");
        wrong_seed["planned"]["fixture"] = json!(format!(
            "pulid-flux-candle-q4-1024-seed{FIVE_RUNG_SEED}-step2"
        ));
        let error = pulid_flux_binding_at(
            &wrong_seed,
            protocol::FLUX1_DEV_REPOSITORY.to_owned(),
            REVISION.to_owned(),
            q4_root.clone(),
            staged_pulid_bundle(),
        )
        .expect_err("the fixture seed must be the seed the capture renders at");
        assert!(error.contains("does not match the seed"), "{error}");

        // The wrong artifact family is refused before the root is looked at.
        let error = pulid_flux_binding_at(
            &pulid_case("q4"),
            protocol::FLUX1_SCHNELL_REPOSITORY.to_owned(),
            REVISION.to_owned(),
            q4_root.clone(),
            staged_pulid_bundle(),
        )
        .expect_err("the schnell artifact must be refused");
        assert!(error.contains(protocol::FLUX1_DEV_REPOSITORY), "{error}");

        // PuLID is character_image only, and its overlay is exactly `identity`.
        let mut wrong_mode = pulid_case("q4");
        wrong_mode["planned"]["target"]["mode"] = json!("text_to_image");
        let error = pulid_flux_binding_at(
            &wrong_mode,
            protocol::FLUX1_DEV_REPOSITORY.to_owned(),
            REVISION.to_owned(),
            q4_root.clone(),
            staged_pulid_bundle(),
        )
        .expect_err("PuLID has no text-to-image route");
        assert!(error.contains("character_image only"), "{error}");
        let mut wrong_overlay = pulid_case("q4");
        wrong_overlay["planned"]["target"]["overlay"] = json!("none");
        let error = pulid_flux_binding_at(
            &wrong_overlay,
            protocol::FLUX1_DEV_REPOSITORY.to_owned(),
            REVISION.to_owned(),
            q4_root,
            staged_pulid_bundle(),
        )
        .expect_err("the identity route must require its own overlay");
        assert!(error.contains("executes exactly \"identity\""), "{error}");
    }

    /// An incomplete staged bundle is refused before the load, naming the missing files.
    #[test]
    fn an_incomplete_candle_pulid_bundle_is_refused_naming_the_missing_files() {
        let root = flux_one_temp_dir("pulid-partial");
        std::fs::create_dir_all(&root).unwrap();
        for file in protocol::PULID_IDENTITY_BUNDLE_FILES.iter().skip(1) {
            std::fs::write(root.join(file), b"weights").unwrap();
        }
        let error = protocol::pulid_identity_bundle_at(root)
            .expect_err("an incomplete bundle must be refused");
        assert!(error.contains(protocol::PULID_ADAPTER_FILE), "{error}");
        assert!(
            error.contains(protocol::PULID_IDENTITY_BUNDLE_ENV),
            "{error}"
        );
    }

    /// `pulid_flux_binding` reads the identity bundle from `SCENEWORKS_PULID_WEIGHTS` and the
    /// backbone from the FLUX1_DEV family — never the schnell one. Serialized on
    /// [`PULID_ENV_LOCK`]: the process environment is global, and every other PuLID test in this
    /// binary fails before it reads an env var.
    #[test]
    fn the_env_bound_pulid_binding_reads_the_bundle_env_and_the_dev_family() {
        const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
        let _guard = PULID_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let restore = PulidEnv::capture();
        let bundle = staged_pulid_bundle();
        let root = flux_one_snapshot_root(protocol::FLUX1_DEV_REPOSITORY, REVISION, "q4");
        let mut planned =
            still_planned_case_in_mode(PULID_FLUX_ID, "staged_residency", 1, "character_image");
        planned["target"]["overlay"] = json!("identity");
        planned["fixture"] = json!(format!("pulid-flux-candle-q4-1024-seed{FLUX1_SEED}-step2"));
        let request = json!({ "planned": planned });
        std::env::set_var(
            "SCENEWORKS_FLUX1_DEV_REPOSITORY",
            protocol::FLUX1_DEV_REPOSITORY,
        );
        std::env::set_var("SCENEWORKS_FLUX1_DEV_REVISION", REVISION);
        std::env::set_var("SCENEWORKS_FLUX1_DEV_ROOT", &root);
        std::env::remove_var(protocol::PULID_IDENTITY_BUNDLE_ENV);
        let error = pulid_flux_binding(&request).expect_err("the bundle env is required");
        assert!(
            error.contains(protocol::PULID_IDENTITY_BUNDLE_ENV),
            "{error}"
        );
        std::env::set_var(protocol::PULID_IDENTITY_BUNDLE_ENV, &bundle.root);
        let binding = pulid_flux_binding(&request).unwrap();
        assert_eq!(binding.bundle.composite_sha256, bundle.composite_sha256);
        assert_eq!(
            binding.bundle.root,
            std::fs::canonicalize(&bundle.root).unwrap(),
            "the env value is canonicalized, not used as spelled"
        );
        // The schnell family is not consulted on the dev backbone.
        std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_ROOT", "/nonexistent/sc-22726");
        std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_REPOSITORY", "not/a-repo");
        std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_REVISION", "junk");
        pulid_flux_binding(&request).unwrap();
        drop(restore);
    }

    /// Serializes the tests that mutate the FLUX.1 env families and the PuLID bundle env.
    static PULID_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Every env var the PuLID binding reads, restored on drop — including on a panic.
    struct PulidEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);

    impl PulidEnv {
        const NAMES: [&'static str; 7] = [
            "SCENEWORKS_FLUX1_DEV_REPOSITORY",
            "SCENEWORKS_FLUX1_DEV_REVISION",
            "SCENEWORKS_FLUX1_DEV_ROOT",
            "SCENEWORKS_FLUX1_SCHNELL_REPOSITORY",
            "SCENEWORKS_FLUX1_SCHNELL_REVISION",
            "SCENEWORKS_FLUX1_SCHNELL_ROOT",
            protocol::PULID_IDENTITY_BUNDLE_ENV,
        ];

        fn capture() -> Self {
            Self(
                Self::NAMES
                    .iter()
                    .map(|name| (*name, std::env::var_os(name)))
                    .collect(),
            )
        }
    }

    impl Drop for PulidEnv {
        fn drop(&mut self) {
            for (name, value) in &self.0 {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    /// The PuLID request and admission context are the worker's: the photoreal preset, one
    /// reference at the target geometry, `overlay: identity`, and the provider mode
    /// `character_image` the pinned route gate checks.
    #[test]
    fn the_candle_pulid_request_and_context_match_the_worker_route() {
        let generation = pulid_flux_generation_request(1024, 768);
        assert_eq!((generation.width, generation.height), (1024, 768));
        assert_eq!(generation.steps, 2);
        assert_eq!(generation.guidance, PULID_FLUX_GUIDANCE);
        assert_eq!(generation.id_weight, PULID_FLUX_ID_WEIGHT);
        assert_eq!(generation.seed, FLUX1_SEED);
        assert!(!generation.use_pid);
        assert!(generation.sampler.is_none() && generation.scheduler.is_none());

        let calibration = runtime_cuda::gen_core::MemoryCalibrationIdentity::new(
            "pulid-flux-cuda-identity-stack-staged-decode-attention-block-window-v1",
            LoadShape::DeferredMaterialization,
        );
        let mut planned =
            still_planned_case_in_mode(PULID_FLUX_ID, "staged_residency", 1, "character_image");
        planned["target"]["overlay"] = json!("identity");
        let selection = planned_selection(&json!({ "planned": planned })).unwrap();
        let context = pulid_flux_context(
            selection,
            &calibration,
            &calibration.fingerprint,
            1024,
            1024,
            1,
            1,
        );
        assert_eq!(
            context.mode,
            MemoryMode::Other("character_image".to_owned())
        );
        assert!(context.has_reference);
        assert_eq!(context.geometry.reference_count, 1);
        assert_eq!(context.overlay.as_deref(), Some("identity"));
        assert!(!context.use_pid);
        assert!(!context.has_phases);
    }

    /// Every FLUX.1 cell the committed plan declares for this lane must name a provider this
    /// adapter implements and a mode its arm serves — the plan/arm agreement E3 asks for, checked
    /// against the checked-in plan rather than a hand-written sample.
    #[test]
    fn every_planned_flux_one_candle_cell_is_served_by_an_arm() {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let mut seen = std::collections::BTreeSet::new();
        for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
            if !key.ends_with(":candle") {
                continue;
            }
            let provider = entry["provider"].as_str().unwrap();
            if !matches!(provider, FLUX1_DEV_ID | FLUX1_SCHNELL_ID | PULID_FLUX_ID) {
                continue;
            }
            seen.insert(key.clone());
            let (_, rest) = key.split_once(':').unwrap();
            let tier = rest.split_once(':').unwrap().0;
            let request = json!({ "planned": {
                "backend": "candle",
                "target": {
                    "provider": provider,
                    "tier": tier,
                    "mode": entry["mode"].clone(),
                    "overlay": entry["overlay"].clone(),
                    "geometry": entry["geometry"].clone(),
                },
                "loadShape": entry["loadShape"].clone(),
                "strategy": { "rung": "staged_residency", "parameters": {} },
                "calibrationFingerprint": entry["calibrationFingerprint"].clone(),
                "fixture": entry["fixture"].clone(),
            }});
            plain_execution_path(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
            still_calibration_label(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
            assert_eq!(planned_tier(&request).unwrap(), tier, "{key}");
            // The fixture must satisfy the same binding a capture applies: member, tier, edge,
            // the seed that member renders at, and the step count.
            validate_fixture_binds_tier_and_geometry(&request)
                .unwrap_or_else(|error| panic!("{key}: {error}"));
            // The base providers ride the registry five-rung path; PuLID is bespoke.
            assert_eq!(
                routes_to_five_rung_reference(&request).unwrap(),
                provider != PULID_FLUX_ID,
                "{key}"
            );
        }
        // The exact cell set, not a count: three members x three tiers, each named once.
        let expected: std::collections::BTreeSet<String> =
            ["flux_dev", "flux_schnell", "pulid_flux_dev"]
                .iter()
                .flat_map(|model| {
                    ["bf16", "q4", "q8"]
                        .iter()
                        .map(move |tier| format!("{model}:{tier}:candle"))
                })
                .collect();
        assert_eq!(seen, expected);
    }

    /// The FLUX.1 fixture binding on this lane: the base members name the five-rung seed they
    /// render at, PuLID names its own, and the member, tier, edge and step count are all bound.
    #[test]
    fn the_candle_flux_one_fixture_binds_member_tier_edge_seed_and_steps() {
        let case = |provider: &str, tier: &str, fixture: &str| {
            let mut planned = still_planned_case_with_fixture(provider, "resident", 1, fixture);
            planned["target"]["tier"] = json!(tier);
            json!({ "planned": planned })
        };
        for (provider, prefix, seed) in [
            (FLUX1_DEV_ID, "fresh-five-rung-flux1-dev", FIVE_RUNG_SEED),
            (
                FLUX1_SCHNELL_ID,
                "fresh-five-rung-flux1-schnell",
                FIVE_RUNG_SEED,
            ),
            (PULID_FLUX_ID, "pulid-flux-candle", FLUX1_SEED),
        ] {
            for tier in ["q4", "q8", "bf16"] {
                let good = format!("{prefix}-{tier}-1024-seed{seed}-step2");
                validate_flux_one_fixture(&case(provider, tier, &good), provider, tier)
                    .unwrap_or_else(|error| panic!("{provider}/{tier}: {error}"));
            }
            let other_seed = if seed == FLUX1_SEED {
                FIVE_RUNG_SEED
            } else {
                FLUX1_SEED
            };
            for (fixture, expected) in [
                (
                    format!("{prefix}-q8-1024-seed{seed}-step2"),
                    "must start with",
                ),
                (
                    format!("{prefix}-q4-768-seed{seed}-step2"),
                    "must start with",
                ),
                (
                    format!("{prefix}-q4-1024-seed{other_seed}-step2"),
                    "does not match the seed",
                ),
                (format!("{prefix}-q4-1024-seed{seed}-step3"), "two-step"),
                ("fresh-five-rung-unused".to_owned(), "must start with"),
            ] {
                let error =
                    validate_flux_one_fixture(&case(provider, "q4", &fixture), provider, "q4")
                        .expect_err("the fixture must be bound to its cell");
                assert!(error.contains(expected), "{provider}: {fixture}: {error}");
            }
        }
        // The member prefixes are not interchangeable: a schnell fixture on a dev plan is refused.
        let crossed = format!("fresh-five-rung-flux1-schnell-q4-1024-seed{FIVE_RUNG_SEED}-step2");
        let error =
            validate_flux_one_fixture(&case(FLUX1_DEV_ID, "q4", &crossed), FLUX1_DEV_ID, "q4")
                .unwrap_err();
        assert!(error.contains("must start with"), "{error}");
        // ...and the shared entry point routes the FLUX.1 members here while leaving Qwen alone.
        validate_fixture_binds_tier_and_geometry(&case(
            FLUX1_DEV_ID,
            "q4",
            "fresh-five-rung-unused",
        ))
        .expect_err("the shared validator must apply the FLUX.1 binding");
        validate_fixture_binds_tier_and_geometry(&case(QWEN_ID, "q4", "fresh-five-rung-unused"))
            .expect("Qwen's fixtures predate the convention and stay unbound");
    }

    /// The base arm reads the tier off the snapshot's `transformer/config.json` through
    /// `candle-gen-flux`'s own resolver, so a directory merely NAMED `q8` cannot satisfy a q8
    /// plan with q4 (or dense) weights inside it.
    #[test]
    fn the_candle_flux_one_base_arm_reads_the_tier_off_the_transformer_config() {
        const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
        for (tier, config) in [
            (
                "q4",
                Some(json!({ "quantization": { "bits": 4, "group_size": 64 } })),
            ),
            (
                "q8",
                Some(json!({ "quantization": { "bits": 8, "group_size": 64 } })),
            ),
            ("bf16", None),
        ] {
            let root = flux_one_snapshot_root(protocol::FLUX1_DEV_REPOSITORY, REVISION, tier);
            if let Some(config) = &config {
                std::fs::create_dir_all(root.join("transformer")).unwrap();
                std::fs::write(
                    root.join("transformer/config.json"),
                    serde_json::to_vec(config).unwrap(),
                )
                .unwrap();
            }
            let spec = LoadSpec::new(WeightsSource::Dir(root));
            validate_flux_one_snapshot_tier(&spec, FLUX1_DEV_ID, tier)
                .unwrap_or_else(|error| panic!("{tier}: {error}"));
            for other in ["q4", "q8", "bf16"] {
                if other == tier {
                    continue;
                }
                let error = validate_flux_one_snapshot_tier(&spec, FLUX1_DEV_ID, other)
                    .expect_err("the declared tier must match the planned one");
                assert!(
                    error.contains(&format!("planned tier {other} does not match")),
                    "{tier}/{other}: {error}"
                );
            }
        }
    }
}

/// sc-22731 — the SANA and Chroma1 Candle arms. Weights-free and env-free: everything asserted
/// here is decided before a snapshot is opened, so it holds on any host.
#[cfg(test)]
mod sana_chroma_candle_tests {
    use super::*;

    const SANA_CHROMA_PROVIDERS: [&str; 5] = [
        SANA_ID,
        SANA_SPRINT_ID,
        CHROMA1_HD_ID,
        CHROMA1_BASE_ID,
        CHROMA1_FLASH_ID,
    ];

    /// The three shipped tiers of the packed turnkeys, one spelling for every expectation below.
    const SANA_CHROMA_CANDLE_TIERS: [&str; 3] = ["q4", "q8", "bf16"];

    /// The Candle plan keys these two families must serve, DERIVED: the five routes crossed with
    /// the shipped tiers, MINUS every (route, tier) `validate_five_rung_lane_tier` refuses — which
    /// is SANA's q4/q8 alone, because there is no packed SANA artifact off-Mac. A frozen `11` said
    /// nothing about WHICH eleven; this fails naming the cell that appeared or vanished, and it
    /// re-derives itself the moment the lane's tier rule changes.
    fn expected_sana_chroma_candle_keys() -> std::collections::BTreeSet<String> {
        SANA_CHROMA_PROVIDERS
            .iter()
            .flat_map(|provider| {
                SANA_CHROMA_CANDLE_TIERS
                    .iter()
                    .map(move |tier| (*provider, *tier))
            })
            .filter(|(provider, tier)| validate_five_rung_lane_tier(provider, tier).is_ok())
            .map(|(provider, tier)| format!("{provider}:{tier}:candle"))
            .collect()
    }

    fn planned(provider: &str, tier: &str, fixture: &str) -> Value {
        json!({
            "planned": {
                "target": {
                    "provider": provider,
                    "tier": tier,
                    "mode": "text_to_image",
                    "overlay": "none",
                    "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
                },
                "backend": "candle",
                "loadShape": "deferred_materialization",
                "strategy": { "rung": "staged_residency", "engagedRungs": ["resident", "staged_residency"], "parameters": {} },
                "calibrationFingerprint": "unused",
                "fixture": fixture
            }
        })
    }

    /// Every SANA/Chroma1 CANDLE plan row names the identity its loaded generator publishes, and
    /// the identities are distinct. `candle-gen-sana` mints one per route (bf16 is its only tier);
    /// `candle-gen-chroma` mints one per (route, tier) from the load receipt (inference PR 951 —
    /// before it, `build_contract` hard-coded `calibration: None` and no Chroma anchor could be
    /// recorded on this lane at all).
    #[test]
    fn every_planned_sana_chroma_candle_cell_names_the_identity_its_loaded_generator_publishes() {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let mut identities = std::collections::BTreeMap::new();
        let mut visited = std::collections::BTreeSet::new();
        for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
            let provider = entry["provider"].as_str().unwrap();
            if !key.ends_with(":candle") || !SANA_CHROMA_PROVIDERS.contains(&provider) {
                continue;
            }
            visited.insert(key.clone());
            let tier = key.split(':').nth(1).unwrap();
            let planned = entry["calibrationFingerprint"]
                .as_str()
                .unwrap_or_else(|| panic!("{key}: calibrationFingerprint must be a string"));
            assert_eq!(
                Some(planned.to_owned()),
                five_rung_calibration_fingerprint(provider, tier),
                "{key}: the plan row must name the loaded generator's production identity"
            );
            assert!(
                identities.insert(planned.to_owned(), key.clone()).is_none(),
                "{key}: {planned} is already claimed by {}",
                identities[planned]
            );
        }
        // A SET, not a count (sc-22731 review): see `expected_sana_chroma_candle_keys`.
        assert_eq!(
            visited,
            expected_sana_chroma_candle_keys(),
            "the Candle plan rows must be exactly the routes x tiers this lane can open"
        );
        // ...and every one carries a DISTINCT identity, so the map covers the same set.
        assert_eq!(
            identities
                .values()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            expected_sana_chroma_candle_keys()
        );
        for (key, identity) in [
            (
                "sana_1600m:bf16:candle",
                "sana-candle-dense-base-full-ladder-v1",
            ),
            (
                "sana_sprint_1600m:bf16:candle",
                "sana-candle-dense-sprint-full-ladder-v1",
            ),
            (
                "chroma1_hd:q4:candle",
                "chroma1-hd-q4-cuda-request-scoped-staged-residency-v1",
            ),
            (
                "chroma1_base:q8:candle",
                "chroma1-base-q8-cuda-request-scoped-staged-residency-v1",
            ),
            (
                "chroma1_flash:bf16:candle",
                "chroma1-flash-bf16-cuda-request-scoped-staged-residency-v1",
            ),
        ] {
            assert_eq!(identities[identity], key);
        }
    }

    /// The Candle SANA lane has ONE tier. A packed plan row is refused by name, before any
    /// environment is read, rather than reaching `candle-gen-sana` and coming back as a
    /// quantization complaint that reads like a spec bug.
    #[test]
    fn the_candle_sana_lane_refuses_a_packed_tier_by_name() {
        for provider in [SANA_ID, SANA_SPRINT_ID] {
            for tier in ["q4", "q8"] {
                let error = validate_five_rung_lane_tier(provider, tier)
                    .expect_err("there is no packed SANA artifact off-Mac");
                assert!(error.contains(provider) && error.contains(tier), "{error}");
            }
            validate_five_rung_lane_tier(provider, "bf16").expect("bf16 is the dense lane tier");
        }
        // ...and the constraint is SANA's alone: every Chroma1 tier is a real Candle cell.
        for provider in [CHROMA1_HD_ID, CHROMA1_BASE_ID, CHROMA1_FLASH_ID] {
            for tier in ["q4", "q8", "bf16"] {
                validate_five_rung_lane_tier(provider, tier).unwrap();
            }
        }
    }

    /// The SANA routes load the upstream dense SNAPSHOT ROOT — no tier component — because that is
    /// what the worker resolves for them (`huggingface_pinned_snapshot_dir`) and what
    /// `candle-gen-sana`'s `validate_immutable_root` requires. Every other route is tiered.
    #[test]
    fn only_the_sana_routes_load_an_untiered_snapshot_root() {
        assert!(!five_rung_root_is_tiered(SANA_ID));
        assert!(!five_rung_root_is_tiered(SANA_SPRINT_ID));
        for provider in [
            CHROMA1_HD_ID,
            CHROMA1_BASE_ID,
            CHROMA1_FLASH_ID,
            FLUX1_DEV_ID,
            QWEN_ID,
            KREA_ID,
        ] {
            assert!(five_rung_root_is_tiered(provider), "{provider}");
        }
    }

    /// Every one of the five routes is named by BOTH shared dispatch tables, so neither the plain
    /// execution path nor the still-geometry refusal can claim this adapter does not implement it.
    #[test]
    fn every_route_is_named_by_both_shared_dispatch_tables() {
        for provider in SANA_CHROMA_PROVIDERS {
            let request = planned(provider, "bf16", "unused");
            let path = plain_execution_path(&request)
                .unwrap_or_else(|error| panic!("{provider}: {error}"));
            assert!(path.starts_with("the Candle "), "{provider}: {path}");
            let label = still_calibration_label(&request)
                .unwrap_or_else(|error| panic!("{provider}: {error}"));
            assert!(label.starts_with("Candle "), "{provider}: {label}");
            assert!(five_rung_family_slug(provider).is_some(), "{provider}");
        }
    }

    /// The fixture binds the route, the tier, the geometry edge, the seed and the step count.
    #[test]
    fn the_candle_fixture_must_name_the_route_tier_edge_and_seed() {
        for (fixture, needle) in [
            // another route
            (
                "fresh-five-rung-chroma1-base-q8-1024-seed16402-step2",
                "must start with",
            ),
            // another tier
            (
                "fresh-five-rung-chroma1-hd-q4-1024-seed16402-step2",
                "must start with",
            ),
            // another edge
            (
                "fresh-five-rung-chroma1-hd-q8-768-seed16402-step2",
                "must start with",
            ),
            // another seed
            (
                "fresh-five-rung-chroma1-hd-q8-1024-seed22726-step2",
                "does not match",
            ),
            // another step count
            (
                "fresh-five-rung-chroma1-hd-q8-1024-seed16402-step4",
                "two-step",
            ),
        ] {
            let request = planned(CHROMA1_HD_ID, "q8", fixture);
            let error = validate_fixture_binds_tier_and_geometry(&request)
                .expect_err("a mislabelled fixture must be refused");
            assert!(error.contains(needle), "{fixture}: {error}");
        }
        validate_fixture_binds_tier_and_geometry(&planned(
            CHROMA1_HD_ID,
            "q8",
            "fresh-five-rung-chroma1-hd-q8-1024-seed16402-step2",
        ))
        .expect("the planned fixture must be accepted");
    }

    /// Every plan row's fixture is one this arm accepts AND one that routes to the five-rung
    /// reference path, so a plan row can never land in the Krea arm's catch-all.
    #[test]
    fn every_planned_candle_fixture_is_accepted_and_routes_to_the_five_rung_path() {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let mut checked = std::collections::BTreeSet::new();
        for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
            let provider = entry["provider"].as_str().unwrap();
            if !key.ends_with(":candle") || !SANA_CHROMA_PROVIDERS.contains(&provider) {
                continue;
            }
            let tier = key.split(':').nth(1).unwrap();
            let request = planned(provider, tier, entry["fixture"].as_str().unwrap());
            assert!(
                routes_to_five_rung_reference(&request).unwrap(),
                "{key}: the fixture must route to the five-rung reference arm"
            );
            validate_fixture_binds_tier_and_geometry(&request)
                .unwrap_or_else(|error| panic!("{key}: {error}"));
            validate_five_rung_lane_tier(provider, tier)
                .unwrap_or_else(|error| panic!("{key}: {error}"));
            checked.insert(key.clone());
        }
        // Set equality, not a count: see `expected_sana_chroma_candle_keys`.
        assert_eq!(checked, expected_sana_chroma_candle_keys());
    }
}
