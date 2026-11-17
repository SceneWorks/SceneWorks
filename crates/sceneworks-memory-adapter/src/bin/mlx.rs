#[cfg(test)]
mod adapter_test_support;

#[cfg(not(target_os = "macos"))]
compile_error!("memory-mlx-adapter is supported only on macOS");

#[path = "mlx_ltx_family.rs"]
mod mlx_ltx_family;
use mlx_ltx_family::*;

#[path = "mlx_minimax_family.rs"]
mod mlx_minimax_family;
use mlx_minimax_family::*;

#[path = "mlx_runtime_family.rs"]
mod mlx_runtime_family;
use mlx_runtime_family::*;

#[path = "mlx_mage_family.rs"]
mod mlx_mage_family;
use mlx_mage_family::*;

#[path = "mlx_qwen_family.rs"]
mod mlx_qwen_family;
use mlx_qwen_family::*;

#[path = "mlx_krea_control_family.rs"]
mod mlx_krea_control_family;
use mlx_krea_control_family::*;

#[path = "mlx_flux_one_family.rs"]
mod mlx_flux_one_family;
use mlx_flux_one_family::*;

#[path = "mlx_turnkey_family.rs"]
mod mlx_turnkey_family;
use mlx_turnkey_family::*;

#[path = "mlx_sana_chroma_family.rs"]
mod mlx_sana_chroma_family;
use mlx_sana_chroma_family::*;

#[path = "mlx_flux2_family.rs"]
mod mlx_flux2_family;
use mlx_flux2_family::*;

#[path = "mlx_krea_base_family.rs"]
mod mlx_krea_base_family;
use mlx_krea_base_family::*;

#[path = "mlx_sdxl_family.rs"]
mod mlx_sdxl_family;
use mlx_sdxl_family::*;

#[path = "mlx_instantid_family.rs"]
mod mlx_instantid_family;
use mlx_instantid_family::*;

#[path = "mlx_qwen_edit_family.rs"]
mod mlx_qwen_edit_family;
use mlx_qwen_edit_family::*;

#[path = "mlx_bernini_family.rs"]
mod mlx_bernini_family;
use mlx_bernini_family::*;

#[path = "mlx_sensenova_family.rs"]
mod mlx_sensenova_family;
use mlx_sensenova_family::*;

#[path = "mlx_sd3_family.rs"]
mod mlx_sd3_family;
use mlx_sd3_family::*;

#[path = "mlx_krea_realtime_family.rs"]
mod mlx_krea_realtime_family;
use mlx_krea_realtime_family::*;

use mlx_gen::gen_core::{
    Capabilities, ComponentPrecisionFloor, GenerationMemory, LoadPhase, MemoryBudget,
    MemoryCacheState, MemoryCalibrationIdentity, MemoryGeometry, MemoryMode, MemoryNumericTier,
    MemoryOptimizationAuthority, MemoryPhase, MemoryProviderContract, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyParameters, TransformerComponent,
};
use mlx_gen::tiling::{SpatialTiling, TilingConfig, VaeTiling};
#[cfg(test)]
use mlx_gen::AdapterKind;
use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Generator, IdentityWeights,
    Image, LoadShape, LoadSpec, OffloadPolicy, Precision, Progress, Quant, WeightsSource,
};
use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_memory_limit, get_peak_memory,
    reset_peak_memory, set_memory_limit, set_wired_limit,
};
use mlx_rs::Array;
use runtime_macos::providers::qwen_image::{load_vae, QwenVae};
use sceneworks_memory_adapter as protocol;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

#[path = "mlx_ltx25.rs"]
mod mlx_ltx25;

// sc-22736: the Wan 2.2 family and SCAIL-2. Four engine providers on one arm, kept in their own
// module for the same reason LTX-2.5 is — a video family's carrier, rate menu and per-tier identity
// table are a self-contained apparatus, not another still-image variation.
#[path = "mlx_wan_scail2.rs"]
mod mlx_wan_scail2;

#[path = "flux3_common.rs"]
mod flux3_common;
#[path = "mlx_flux3.rs"]
mod mlx_flux3;
const FLUX3_PROVIDER: &str = "flux3_klein";
const FLUX3_BASE_PROVIDER: &str = "flux3_klein_base";

#[path = "mlx_iris.rs"]
mod mlx_iris;

/// The three Iris-3B task routes (sc-25690), served by `mlx_iris`. Registry ids of `mlx-gen-iris`
/// (`gen_core_iris::GENERATION_MODEL_ID` / `DEPTH_MODEL_ID` / `restoration::MODEL_ID`), which are
/// also the catalog model ids. Spelled here, as consts, because
/// `scripts/stale-lane-report.mjs::adapterCapturableProviders` reads the dispatch arms off this file.
const IRIS_PROVIDER: &str = "iris_3b";
const IRIS_DEPTH_PROVIDER: &str = "iris_3b_depth";
const IRIS_RESTORE_PROVIDER: &str = "iris_3b_restore";

#[path = "mlx_supra2.rs"]
mod mlx_supra2;

/// Supra2-IMG (sc-24217, epic 24212), served by `mlx_supra2`: `mlx-gen-supra2`'s registry id
/// (`gen_core_checkpoint::supra2::PROVIDER_ID`), which is also the catalog model id. Spelled here, as a const,
/// because `scripts/stale-lane-report.mjs::adapterCapturableProviders` reads the dispatch arms off
/// this file; `mlx_supra2`'s tests bind it to the engine constant through `protocol`.
const SUPRA2_PROVIDER: &str = "supra2_img";

#[path = "mlx_minimax_h3_image.rs"]
mod mlx_minimax_h3_image;

/// MiniMax-H3-Image (SW-7100, feature SW-F363), served by `mlx_minimax_h3_image`: the
/// `mlx-gen-minimax-h3::h3_image` registry id, which is also the catalog model id. Spelled here, as a
/// const, because `scripts/stale-lane-report.mjs::adapterCapturableProviders` reads the dispatch arms
/// off this file; `mlx_minimax_h3_image`'s tests bind it to the engine constant.
const MINIMAX_H3_IMAGE_PROVIDER: &str = "minimax_h3_image";

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
/// Plain, reference-free Krea 2 Turbo text-to-image. This is a distinct calibration lane from the
/// pose-control provider below even though both providers live in `mlx-gen-krea`.
const KREA_BASE_PROVIDER: &str = "krea_2_turbo";
/// The string `krea_2_turbo` (and, before sc-22735, every other base route) published at all three
/// tiers. Retired in the engine and retained by no cell: no MLX turbo record was ever measured
/// against it. Kept here only so the tests below can assert it never comes back.
#[cfg(test)]
const RETIRED_KREA_BASE_CALIBRATION_FINGERPRINT: &str =
    "krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3";
const KREA_BASE_SEED: u64 = 18377;
/// The undistilled full-CFG Krea 2 base (`mlx_gen_krea::KREA_2_RAW_ID`), the second member of the
/// Krea base arm's family table (sc-22735). Same engine crate and same shipped text-to-image route
/// as Turbo, a DIFFERENT artifact family (`SceneWorks/krea-2-raw-mlx`), a different planned load
/// shape, and — unlike Turbo — no `LoadSpec` quant at any tier.
const KREA_RAW_PROVIDER: &str = "krea_2_raw";
const KREA_RAW_PLAIN_EXECUTION_PATH: &str = "the MLX Krea 2 Raw base-only text-to-image path";
const KREA_RAW_SEED: u64 = 22735;
/// Recommended, reference-free SDXL base text-to-image. The pinned provider exposes native VAE
/// bounded decode; bounded attention remains Missing.
const SDXL_PROVIDER: &str = "sdxl";
const SDXL_CALIBRATION_FINGERPRINT: &str = "sdxl-mlx-unet-shared-ladder-v5";
const SDXL_SEED: u64 = 18379;
// RMS is bounded by the same per-pixel envelope as maximum absolute error. This avoids inventing a
// tighter SDXL-specific tolerance before the first physical campaign while still making the real
// harness's runtime-complete quality contract explicit and mutation-sensitive.
const SDXL_RMS_THRESHOLD: f64 = MAX_THRESHOLD;
const SDXL_PLAIN_EXECUTION_PATH: &str = "the MLX SDXL base-only text-to-image path";
/// sc-22729. The InstantID identity route — a BESPOKE provider, not a registry id. `mlx-gen-catalog`
/// lists `instantid` in `BESPOKE_UTILITY_CRATES` and registers no `ModelDescriptor` for it, so this
/// arm loads through the crate's own production entry point (`InstantId::load_with_memory_context`,
/// the one `image_jobs/instantid.rs` calls) rather than through `registry.load`.
const INSTANTID_PROVIDER: &str = "instantid";
const INSTANTID_MODEL_ID: &str = "instantid_realvisxl";
/// `mlx-gen-instantid` publishes its request-contract revision AS the calibration fingerprint
/// (`memory_strategy.rs:16,88` at the pin) — unlike the candle twin, which mints a separate string.
const INSTANTID_CALIBRATION_FINGERPRINT: &str = "instantid-request-contract-v1";
const INSTANTID_EXECUTION_PATH: &str = "the MLX InstantID identity-conditioned character path";
const INSTANTID_STILL_CALIBRATION: &str = "MLX InstantID identity calibration";
/// The worker's production identity-route defaults (`image_jobs/instantid.rs`
/// `INSTANTID_IP_SCALE` / `INSTANTID_CONTROLNET_SCALE`).
const INSTANTID_IP_SCALE: f32 = 0.8;
const INSTANTID_CONTROLNET_SCALE: f32 = 0.8;
const INSTANTID_SEED: u64 = 18381;
const KREA_PROVIDER: &str = "krea_2_turbo_control";
const KREA_OVERLAY_REPOSITORY: &str = "SceneWorks/krea2-pose-controlnet-beta";
const KREA_OVERLAY_FILE: &str = "control_step5000.safetensors";
const KREA_TILE_EDGES: [u32; 1] = [512];
const KREA_TILE_OVERLAP: u32 = 64;
const KREA_CONTROL_EXECUTION_PATH: &str = "the MLX Krea pose-control path";
const KREA_PLAIN_EXECUTION_PATH: &str = "the MLX Krea base-only text-to-image path";
/// The gated VAE probe reaches `load_vae` directly, with no `LoadSpec` and therefore no deferred
/// block schedule: it bulk-materializes the VAE, which is eager materialization.
const QWEN_VAE_PROBE_LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;
const QWEN_PROVIDER: &str = "qwen_image";
const QWEN_PLAIN_EXECUTION_PATH: &str = "the MLX Qwen VAE-only path";
const QWEN_PROVIDER_EXECUTION_PATH: &str = "the pinned MLX Qwen base provider path";
/// The Qwen-Image-**Edit** engine provider (sc-22728): `mlx-gen-qwen-image`'s `model_edit::MODEL_ID`,
/// the registry id the worker loads for BOTH shipped edit catalog ids
/// (`image_jobs/qwen.rs` `qwen_edit_engine_id`). Reference-conditioned: the engine refuses a request
/// that carries no `Conditioning::Reference` at all (`model_edit.rs` `validate`).
const QWEN_EDIT_PROVIDER: &str = "qwen_image_edit";
/// The two catalog ids that provider serves. They are ONE checkpoint on ONE artifact family; the
/// Lightning id differs only in the built-in distill LoRA it stacks and the `lightning` sampler
/// recipe that LoRA was distilled for — not in the engine model.
const QWEN_EDIT_MODEL: &str = "qwen_image_edit_2511";
const QWEN_EDIT_LIGHTNING_MODEL: &str = "qwen_image_edit_2511_lightning";
const QWEN_EDIT_EXECUTION_PATH: &str = "the pinned MLX Qwen-Image-Edit-2511 provider path";
const QWEN_EDIT_LIGHTNING_EXECUTION_PATH: &str =
    "the pinned MLX Qwen-Image-Edit-2511 Lightning distill provider path";
/// The engine recipe the distill LoRA was trained for (`model_edit.rs`: `req.sampler == "lightning"`
/// selects the CFG-off few-step schedule, and the matching LoRA must be supplied via `spec.adapters`).
const QWEN_EDIT_LIGHTNING_SAMPLER: &str = "lightning";
/// Two steps for the production edit path — the first `Step` callback closes the conditioning
/// envelope and the second gives denoise its own measured interval before `Decoding`, the shape
/// every other image arm here uses. The Lightning path runs the distill's own FOUR steps: its
/// schedule (`lightning_sigmas`) is the official 4-step lightx2v recipe and the worker's default,
/// so a 2-step capture would measure a request the product never issues.
const QWEN_EDIT_STEPS: u32 = 2;
const QWEN_EDIT_LIGHTNING_STEPS: u32 = 4;
/// The edit prompt every Qwen edit capture renders. Fixed with the seed and the reference so two
/// captures of one anchor are the same request.
const QWEN_EDIT_PROMPT: &str = "replace the background with a plain grey studio backdrop";
const Z_IMAGE_PROVIDER: &str = "z_image_turbo";
const Z_IMAGE_PLAIN_EXECUTION_PATH: &str = "the MLX Z-Image base-only text-to-image path";
/// The undistilled Z-Image BASE provider (sc-22724): the same engine crate as Turbo
/// (`mlx-gen-z-image`, registry id `model_base::MODEL_ID`), a distinct artifact family
/// (`SceneWorks/z-image-mlx`, `SCENEWORKS_Z_IMAGE_BASE_*`), and real CFG in the denoise loop.
const Z_IMAGE_BASE_PROVIDER: &str = "z_image";
const Z_IMAGE_BASE_PLAIN_EXECUTION_PATH: &str = "the MLX Z-Image base-model text-to-image path";
/// `z_image_edit` is a catalog alias for the Turbo provider driven in `edit_image` mode (worker
/// `engines.rs`), so its anchors plan `provider: z_image_turbo, mode: edit_image` and this arm
/// conditions the SAME loaded Turbo generator on one reference image.
const Z_IMAGE_EDIT_EXECUTION_PATH: &str =
    "the MLX Z-Image-Turbo reference-conditioned edit path (the z_image_edit route)";
/// The worker's production edit strength default (`resolve_zimage_edit_init`, `advanced.strength`).
const Z_IMAGE_EDIT_STRENGTH: f32 = 0.6;
/// Edit captures run four steps: the img2img start step is `floor(steps * strength)` (shared
/// `img2img::init_time_step`), so `4 * 0.6` starts at step 2 and leaves two executed denoise steps
/// — the same two-step conditioning/denoise phase shape the text-to-image captures use.
const Z_IMAGE_EDIT_STEPS: u32 = 4;
/// The `mlx:flux2_dev` lane: the FLUX.2-dev text-to-image provider the measured renders load.
const FLUX2_PROVIDER: &str = "flux2_dev";
const FLUX2_CALIBRATION_FINGERPRINT: &str = "sc-18218-flux2-dev-t2i-resident-evidence-v1";
const FLUX2_PLAIN_EXECUTION_PATH: &str = "the MLX FLUX.2-dev base-only text-to-image path";
/// The FLUX.2-klein-9B text-to-image provider (sc-22727). BOTH klein catalog models —
/// `flux2_klein_9b` and the separately distilled `flux2_klein_9b_kv` — load through this ONE engine
/// registry id (`crates/sceneworks-worker/src/engines.rs`); they differ only in the artifact, which
/// the engine discriminates by snapshot path and `LoadSpec::resolved_route`.
const FLUX2_KLEIN_PROVIDER: &str = "flux2_klein_9b";
const FLUX2_KLEIN_PLAIN_EXECUTION_PATH: &str =
    "the MLX FLUX.2-klein-9B base-only text-to-image path";
const FLUX2_KLEIN_KV_PLAIN_EXECUTION_PATH: &str =
    "the MLX FLUX.2-klein-9B KV-cache base-only text-to-image path";
/// The WEIGHTS-FREE registry declaration identity for the klein T2I provider:
/// `KLEIN_STATIC_BEHAVIOR_FINGERPRINT` suffixed with the provider id
/// (`provider_id.replace('_', "-")`), published by
/// `mlx-gen-flux2::memory_strategy::weights_free_klein_contract`. This is the string
/// `config/manifests/builtin.models.jsonc` declares for every klein rung on both klein models —
/// a DECLARATION, not the identity a load of real weights publishes. sc-22727: the two are no
/// longer the same string; see [`flux2_calibration_fingerprint`].
const FLUX2_KLEIN_REGISTRY_FINGERPRINT: &str =
    "flux2-klein-static-registry-behavior-v2-flux2-klein-9b";
/// One fixed seed for every `mlx:flux2_klein_9b*` fixture. Distinct from [`FLUX2_SEED`] so a klein
/// receipt can never be traced to the dev lane's calibration seed.
const FLUX2_KLEIN_SEED: u64 = 22727;
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
/// LTX-2.3 quality is the same KIND of claim as FLUX.2-dev's — repeat determinism on one loaded
/// provider, with no alternate code path selected between the two renders — so it adopts the same
/// numeric envelope deliberately rather than inventing a looser one for video. The values are
/// restated under LTX names because the record embeds them as `maximumErrorThreshold` and friends:
/// an `mlx:ltx_2_3` receipt must not be traceable to a constant that asserts a FLUX.2 provenance.
/// The margin is if anything wider here: LTX's distilled schedule is fully seeded and the measured
/// warm repeat is bit-identical (0/255 on all three metrics), while the mandatory broad-bias
/// mutation must breach all three.
const LTX_MAX_THRESHOLD: f64 = FLUX2_MAX_THRESHOLD;
const LTX_MEAN_THRESHOLD: f64 = FLUX2_MEAN_THRESHOLD;
const LTX_RMS_THRESHOLD: f64 = FLUX2_RMS_THRESHOLD;
/// The `mlx:ltx_2_3` lane (sc-18808) — the FIRST video arm in this adapter. Every arm above it is an
/// image arm and keeps its `geometry.frames == 1` refusal (`protocol::validate_still_geometry`);
/// this one is the single arm allowed to accept a multi-frame geometry, and it pays for that by
/// validating against LTX's OWN declared envelope instead of accepting any frame count at all.
const LTX_PROVIDER: &str = "ltx_2_3";
const LTX25_PROVIDER: &str = "ltx_2_5";
// sc-22736. Engine registry ids, not catalog ids: `wan_2_2` is the SceneWorks name for the TI2V-5B
// route (worker `engines.rs` `video_engine_ids`), and SCAIL-2's two names coincide.
const WAN_TI2V_5B_PROVIDER: &str = "wan2_2_ti2v_5b";
const WAN_T2V_A14B_PROVIDER: &str = "wan2_2_t2v_14b";
const WAN_I2V_A14B_PROVIDER: &str = "wan2_2_i2v_14b";
const SCAIL2_PROVIDER: &str = "scail2_14b";
const LTX_PLAIN_EXECUTION_PATH: &str = "the MLX LTX-2.3 base-only text-to-video path";
/// How [`diagnostic_video_frames`] names this lane when it refuses a non-video output. Extracted
/// verbatim from that function's own messages when the second video arm made the label a parameter
/// (sc-18663), so LTX's refusals are unchanged.
const LTX_VIDEO_LABEL: &str = "MLX LTX-2.3";
/// Every numeric tier the `mlx:ltx_2_3` family ships, and therefore every tier that has a
/// production calibration identity. Kept beside [`ltx_calibration_fingerprint`] so the identity
/// family and the plan parser cannot disagree about the inventory.
const LTX_TIERS: [&str; 3] = ["bf16", "q4", "q8"];
/// The tier the engine's retained [`mlx_gen_ltx::memory_strategy::CALIBRATION_FINGERPRINT`] names,
/// READ off the pinned crate rather than spelled here.
///
/// LTX's retained key carries no tier token (`sc-20772-ltx-2-3-mlx-memory-ladder-v2`), so nothing
/// in the string itself says which tier it belongs to; the engine's `CALIBRATED_TIER` is the only
/// source of that fact. Binding to it means a re-tiered engine moves this family's bare member
/// with it, and `every_planned_ltx_mlx_row_names_the_identity_this_arm_mints_for_its_tier` then
/// reds until the plan follows — the plan is bound to the same source, so the two cannot drift
/// apart silently. `the_ltx_identity_family_is_the_engines_retained_key_with_a_tier_token` binds
/// every member to the engine's own weights-free minting function, byte for byte.
const LTX_CALIBRATED_TIER: &str = mlx_gen_ltx::memory_strategy::CALIBRATED_TIER;

/// One fixed seed for every `mlx:ltx_2_3` fixture
/// (`ltx-2-3-mlx-<tier>-<width>x<height>-f<frames>-fps<fps>-seed18946`). Historical SC-18810
/// evidence remains bound to seed 18808 and is never relabelled.
const LTX_SEED: u64 = 18946;
// SC-19642 incident evidence. These are the exact immutable tier inventories prepared for
// SC-18946, not estimates. The q4 f305 provider reached the kernel-maintained physical footprint
// below before the host watchdog panicked. A soft MLX allocator limit cannot make these safe: MLX
// documents that one allocation may exceed it, and an in-process monitor can stall with Metal.
const LTX_Q4_INVENTORY_BYTES: u64 = 20_467_690_460;
const LTX_Q8_INVENTORY_BYTES: u64 = 29_728_720_716;
const LTX_BF16_INVENTORY_BYTES: u64 = 47_092_811_992;
const LTX_Q4_F305_CRASH_FOOTPRINT_BYTES: u64 = 96_970_084_480;
/// The exact q8 text-encoder + transformer co-staged safetensor arithmetic captured by SC-18808 at
/// 768x512x97. This is not a historical `phys_footprint` measurement or a complete-load bound: the
/// canary deliberately reuses the smaller value as a conservative external stop, with separate
/// host-pressure gates. It must never be reused to admit a campaign row.
const LTX_CANARY_MAX_FOOTPRINT_BYTES: u64 = 53_347_146_863;
const LTX_CANARY_MAX_RUNTIME_SECONDS: f64 = 1_800.0;
const LTX_CANARY_WATCHDOG_PROTOCOL: &str = "sceneworks-memory-watchdog-v1";
const LTX_PROVIDER_PHASE_PROTOCOL: &str = "sceneworks-provider-phase-v1";
const LTX_CAMPAIGN_ENTRY_PHASE_PROFILE: &str = "campaign-entry";
const LTX_BOUNDED_CARRIER_PHASE_PROFILE: &str = "bounded-carrier";
const LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE: &str = "bounded-campaign-entry";
const LTX_PROVIDER_PHASE_NAMES: [&str; 10] = [
    "common_load",
    "primary_conditioning",
    "primary_denoise",
    "primary_decode",
    "lifecycle_warm_repeat",
    "lifecycle_cancel",
    "lifecycle_cancel_recovery",
    "lifecycle_error",
    "lifecycle_error_recovery",
    "cleanup",
];
const LTX_BOUNDED_CARRIER_PHASE_NAMES: [&str; 5] = [
    "common_load",
    "primary_conditioning",
    "primary_denoise",
    "primary_decode",
    "cleanup",
];
const LTX_CANARY_WIDTH: u32 = 256;
const LTX_CANARY_HEIGHT: u32 = 256;
const LTX_CANARY_FRAMES: u32 = 9;
const LTX_CANARY_FPS: u32 = 24;
const LTX_CANARY_SEED: u64 = 1234;
const LTX_CANARY_TILE_EDGE: u32 = 192;
const LTX_CANARY_OVERLAP: u32 = 64;
const LTX_CANARY_FIXTURE: &str = "ltx-2-3-mlx-q4-256x256-f9-fps24-seed1234-safety-canary";
const LTX_PRODUCT_CANARY_WIDTH: u32 = 768;
const LTX_PRODUCT_CANARY_HEIGHT: u32 = 512;
const LTX_PRODUCT_CANARY_FRAMES: u32 = 97;
const LTX_PRODUCT_CANARY_FPS: u32 = 24;
const LTX_PRODUCT_CANARY_FIXTURE: &str =
    "ltx-2-3-mlx-q4-768x512-f97-fps24-seed1234-product-envelope-canary";
const LTX_CAMPAIGN_ENTRY_ACTION: &str = "campaign_entry";
const LTX_CAMPAIGN_ENTRY_IDENTITY: &str = "sc-20191-q4-768x512-f121-fps30-staged-v1";
const LTX_CAMPAIGN_ENTRY_LOGICAL_CASE_ID: &str = "implan-9b107d4d1ca0d61d4faa";
const LTX_CAMPAIGN_ENTRY_FIXTURE: &str = "ltx-2-3-mlx-q4-768x512-f121-fps30-seed18946";
const LTX_CAMPAIGN_ENTRY_WIDTH: u32 = 768;
const LTX_CAMPAIGN_ENTRY_HEIGHT: u32 = 512;
const LTX_CAMPAIGN_ENTRY_FRAMES: u32 = 121;
const LTX_CAMPAIGN_ENTRY_FPS: u32 = 30;
const LTX_BOUNDED_CARRIER_ACTION: &str = "bounded_carrier_proof";
const LTX_BOUNDED_CARRIER_IDENTITY: &str = "sc-20254-q4-768x512-f121-fps30-bounded-192-64-v1";
const LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID: &str =
    "diagnostic-sc20254-q4-768x512-f121-fps30-bounded-192-64";
const LTX_BOUNDED_CARRIER_FIXTURE: &str =
    "ltx-2-3-mlx-q4-768x512-f121-fps30-seed18946-bounded-decode-192-64-proof";
const LTX_BOUNDED_CAMPAIGN_ACTION: &str = "bounded_campaign_entry";
const LTX_BOUNDED_CAMPAIGN_IDENTITY: &str =
    "sc-20318-q4-768x512-f121-fps30-bounded-192-64-authoritative-v1";
const LTX_BOUNDED_CAMPAIGN_LOGICAL_CASE_ID: &str = "implan-964db61ed3789af6386b";
const LTX_BOUNDED_CAMPAIGN_FIXTURE: &str =
    "ltx-2-3-mlx-q4-768x512-f121-fps30-bounded-decode-192-64-seed18946";
const LTX_BOUNDED_CAMPAIGN_Q8_IDENTITY: &str =
    "sc-20430-q8-768x512-f121-fps30-bounded-192-64-authoritative-v1";
const LTX_BOUNDED_CAMPAIGN_Q8_LOGICAL_CASE_ID: &str = "implan-d47640caa0c469f2ee13";
const LTX_BOUNDED_CAMPAIGN_Q8_FIXTURE: &str =
    "ltx-2-3-mlx-q8-768x512-f121-fps30-bounded-decode-192-64-seed18946";
const LTX_BOUNDED_CAMPAIGN_BF16_IDENTITY: &str =
    "sc-20430-bf16-768x512-f121-fps30-bounded-192-64-authoritative-v1";
const LTX_BOUNDED_CAMPAIGN_BF16_LOGICAL_CASE_ID: &str = "implan-b3926164bf6bfbee98e1";
const LTX_BOUNDED_CAMPAIGN_BF16_FIXTURE: &str =
    "ltx-2-3-mlx-bf16-768x512-f121-fps30-bounded-decode-192-64-seed18946";
const LTX_CANARY_ARTIFACT_REVISION: &str = "01df27d308466533aa09d251e3aebdcc627d07eb";
const LTX_CANARY_Q4_INVENTORY_FILES: u64 = 11;
const LTX_CANARY_Q4_INVENTORY_SHA256: &str =
    "4e811932e87bb258f642ada790525e36ef2a55959c520e755f1807caf6fa225a";
const LTX_CANARY_Q8_INVENTORY_FILES: u64 = 11;
const LTX_CANARY_Q8_INVENTORY_SHA256: &str =
    "bb0bb7577157a158ca39494837d64cb36ded0380ca7ee0c930fea7311f22a247";
const LTX_CANARY_BF16_INVENTORY_FILES: u64 = 10;
const LTX_CANARY_BF16_INVENTORY_SHA256: &str =
    "006caeaa9a8638b337cdf5a8622ce8535380b18ebaf90b36c3e2d5d15354f2a8";
const LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES: u64 = 17;
const LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES: u64 = 26_427_894_918;
const LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256: &str =
    "abde2d155aa8991747cc2999d40688d29a50261c080c0d51fac20357653928d7";
// The pinned mlx-gen-ltx `transformer::ONES_CACHE` retains one bf16 unit-weight Array for each
// AvDiT stream's weightless RMSNorm dimension on the current thread. The exact LTX-2.3 config is
// video 32 heads x 128 = 4096 and audio 32 x 64 = 2048. These six Ki elements intentionally
// outlive the dropped generator so production denoise can reuse them; the diagnostic canary
// accounts for this one named allocation by checked arithmetic, never by a tolerance.
const LTX_CANARY_ONES_CACHE_IDENTITY: &str = "mlx-gen-ltx-transformer-ones-cache-av-bfloat16-v1";
const LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION: u64 = 4_096;
const LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION: u64 = 2_048;
const BFLOAT16_BYTES_PER_ELEMENT: u64 = 2;
const LTX_INCIDENT_FORBIDDEN: &str = "incident_forbidden";
const LTX_ARITHMETIC_UNMEASURABLE: &str = "arithmetic_unmeasurable";
const LTX_SAFETY_REFUSED_OPEN: &str = "safety_refused_open";
const LTX_INCIDENT_PREDICTED_DECODE_BYTES: u64 =
    3_300_000_000 + 40 * (1280 * 704 * 305) + 300 * 384 * 384 * 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LtxCanaryProfile {
    Safety,
    ProductEnvelope,
}

impl LtxCanaryProfile {
    fn action(self) -> &'static str {
        match self {
            Self::Safety => "canary",
            Self::ProductEnvelope => "product_envelope_canary",
        }
    }

    fn width(self) -> u32 {
        match self {
            Self::Safety => LTX_CANARY_WIDTH,
            Self::ProductEnvelope => LTX_PRODUCT_CANARY_WIDTH,
        }
    }

    fn height(self) -> u32 {
        match self {
            Self::Safety => LTX_CANARY_HEIGHT,
            Self::ProductEnvelope => LTX_PRODUCT_CANARY_HEIGHT,
        }
    }

    fn frames(self) -> u32 {
        match self {
            Self::Safety => LTX_CANARY_FRAMES,
            Self::ProductEnvelope => LTX_PRODUCT_CANARY_FRAMES,
        }
    }

    fn fps(self) -> u32 {
        match self {
            Self::Safety => LTX_CANARY_FPS,
            Self::ProductEnvelope => LTX_PRODUCT_CANARY_FPS,
        }
    }

    fn fixture(self) -> &'static str {
        match self {
            Self::Safety => LTX_CANARY_FIXTURE,
            Self::ProductEnvelope => LTX_PRODUCT_CANARY_FIXTURE,
        }
    }

    fn prompt(self) -> &'static str {
        match self {
            Self::Safety => "a sunlit pine branch, static camera",
            Self::ProductEnvelope => {
                "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
            }
        }
    }

    fn video_mode(self) -> Option<&'static str> {
        match self {
            Self::Safety => Some("no_audio"),
            Self::ProductEnvelope => None,
        }
    }

    fn video_mode_identity(self) -> &'static str {
        match self {
            Self::Safety => "no_audio",
            Self::ProductEnvelope => "default_av",
        }
    }

    fn completion_status(self) -> &'static str {
        match self {
            Self::Safety => "diagnostic_canary_complete",
            Self::ProductEnvelope => "diagnostic_product_envelope_canary_complete",
        }
    }

    fn identity(self) -> &'static str {
        match self {
            Self::Safety => "sc-19741-safety",
            Self::ProductEnvelope => "sc-20169-product-envelope",
        }
    }
}
/// LTX's video VAE is x32 spatial and **x8 causal temporal**, so `out_f = 1 + (t_lat - 1) * 8` and
/// the engine's `validate_request` hard-rejects any `num_frames` that is not `1 + 8k`. Latent
/// temporal depth — not raw frame count — is the physically motivated regressor for a frames-aware
/// phase curve (sc-18810), so the arm reports it alongside the raw count.
const LTX_TEMPORAL_SCALE: u32 = 8;
/// The four constants below are copies of the `limits` block of the `ltx_2_3` entry in
/// **`config/manifests/builtin.models.jsonc`**, which is their source of truth.
///
/// This crate carries two dependencies on purpose and cannot reach `sceneworks-core`'s JSONC reader
/// to consult the manifest at test time, so the binding lives on the node side instead: `npm run
/// check` runs `the MLX LTX arm's manifest constants match the shipped ltx_2_3 limits` in
/// `scripts/platform-review-contracts.test.mjs`, which parses the manifest, parses these four
/// declarations out of this file, and re-derives [`LTX_FRAME_ENVELOPE`] from the manifest's own
/// durations x fps. Edit the manifest limits without editing these and that gate reds (sc-18808
/// review). If you rename a constant, rename it there too — the gate asserts each match rather than
/// skipping what it cannot find.
///
/// `limits.requiresDimensionsMultipleOf`, which mirrors the engine's
/// `SIZE_MULTIPLE = 2 * SPATIAL_SCALE` (stage 1 renders at half resolution, so a dimension must
/// divide by twice the 32x VAE compression).
const LTX_DIMENSION_MULTIPLE: u32 = 64;
/// `limits.resolutions`, verbatim.
const LTX_RESOLUTIONS: [(u32, u32); 5] =
    [(768, 512), (512, 768), (640, 640), (1280, 704), (704, 1280)];
/// `limits.durations` and `limits.fps`, verbatim. Together they span the frame envelope below.
const LTX_DURATIONS_SECONDS: [u32; 6] = [4, 6, 8, 10, 12, 15];
const LTX_FPS: [u32; 3] = [24, 25, 30];
const MIB: u64 = 1024 * 1024;

const LTX_FRAME_ENVELOPE: (u32, u32) = ltx_frame_envelope();

/// The `mlx:minimax_h3` lane (sc-18663) — the SECOND video arm, and the first joint audio+video one.
///
/// It accepts a multi-frame geometry for the same reason `mlx:ltx_2_3` does, and pays for it the
/// same way: by validating against MiniMax-H3's OWN envelope. Unlike the LTX arm, that envelope is
/// not transcribed from the manifest — it is READ from the engine crate this arm links against
/// ([`minimax_legal_frame_counts`], [`mlx_gen_minimax_h3::SPATIAL_STRIDE`],
/// [`mlx_gen_minimax_h3::CANVAS_MAX_PIXELS`], [`mlx_gen_minimax_h3::MINIMAX_H3_FPS`]), so it cannot
/// drift from the pinned provider at all. `minimax_envelope_is_the_pinned_engines_own` pins the
/// values a plan may rely on, so a pin bump that moves one reds here rather than silently widening
/// what this arm will capture.
const MINIMAX_PROVIDER: &str = "minimax_h3";
const MINIMAX_PLAIN_EXECUTION_PATH: &str =
    "the MLX MiniMax-H3 base-only t2va text-to-audio-video path";
/// How this arm names itself in a geometry, target or output-shape refusal.
const MINIMAX_LABEL: &str = "MLX MiniMax-H3 calibration";
/// How [`diagnostic_video_frames`] names this lane when it refuses a non-video output.
const MINIMAX_VIDEO_LABEL: &str = "MLX MiniMax-H3";
/// Every numeric tier the `mlx:minimax_h3` family ships, and therefore every tier that has a
/// production calibration identity. Kept beside [`minimax_calibration_fingerprint`] so the identity
/// family and the plan parser cannot disagree about the inventory.
const MINIMAX_TIERS: [&str; 3] = ["bf16", "q4", "q8"];
/// The tier the engine's retained
/// [`mlx_gen_minimax_h3::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT`] names, read off the
/// pinned crate: that key carries no tier token, so the engine is the only source of which tier
/// keeps it.
const MINIMAX_CALIBRATED_TIER: &str = mlx_gen_minimax_h3::memory_strategy::CALIBRATED_TIER;

/// One fixed seed for every `mlx:minimax_h3` fixture
/// (`minimax-h3-mlx-<tier>-<width>x<height>-f<frames>-fps<fps>-seed17137`).
const MINIMAX_SEED: u64 = 17137;
/// Model evaluations per calibration render. Two is the scheduler's own
/// `mlx_gen_minimax_h3::MIN_INFERENCE_STEPS` (and the manifest's `limits.hardMinSteps`), and it is
/// what makes the phase boundaries real: the DiT is mapped and the AdaLN schedule projected before
/// step 1, and the second step supplies a denoise-only interval before `Decoding`.
const MINIMAX_STEPS: u32 = 2;
/// MiniMax-H3 determinism thresholds, in [0,1] units.
///
/// These are DELIBERATELY spelled as their own literals rather than aliased onto the FLUX.2 or LTX
/// constants. The record embeds them as `maximumErrorThreshold` and friends, and an
/// `mlx:minimax_h3` receipt must not be traceable to a constant that asserts another provider's
/// provenance (the rule stated at [`LTX_MAX_THRESHOLD`]). The magnitudes match the other lanes
/// because the CLAIM is the same kind — repeat determinism on one loaded provider with no alternate
/// code path selected between the two renders — not because the constants are shared.
///
/// The envelope is a tolerance, not a prediction: this checkpoint is guidance-distilled and fully
/// seeded, so a warm repeat is expected to be bit-identical on all three metrics, while the
/// mandatory broad-bias mutation must breach all three.
const MINIMAX_MAX_THRESHOLD: f64 = 3.0 / 255.0;
const MINIMAX_MEAN_THRESHOLD: f64 = 1.0 / 255.0;
const MINIMAX_RMS_THRESHOLD: f64 = 1.5 / 255.0;

/// The `mlx:krea_realtime_14b` arm (sc-22735) — the third MLX video lane, and the first
/// AUTOREGRESSIVE one.
///
/// Krea Realtime 14B is Wan 2.1 T2V 14B weight-for-weight; what makes it its own lane is the
/// inference regime, not the weights: a short per-block few-step denoise driven by a rolling causal
/// KV cache. The KV cache is what makes this arm's peaks unlike every other video lane's — it holds
/// bf16 ACTIVATIONS, so a q4 DiT does not shrink it, and it scales with the read window rather than
/// with the tier.
const KREA_REALTIME_PROVIDER: &str = "krea_realtime_14b";
const KREA_REALTIME_PLAIN_EXECUTION_PATH: &str =
    "the MLX Krea Realtime 14B reference-free text-to-video path";
/// How this arm names itself in a geometry, target or output-shape refusal.
const KREA_REALTIME_LABEL: &str = "MLX Krea Realtime 14B calibration";
/// How [`diagnostic_video_frames`] names this lane when it refuses a non-video output.
const KREA_REALTIME_VIDEO_LABEL: &str = "MLX Krea Realtime 14B";
/// One fixed seed for every `mlx:krea_realtime_14b` fixture
/// (`krea-realtime-14b-mlx-<tier>-<width>x<height>-f<frames>-fps<fps>-seed22735`).
const KREA_REALTIME_SEED: u64 = 22735;
/// The ONLY cadence this checkpoint generates at. Unlike MiniMax-H3 the engine crate exports no fps
/// symbol, so this is TRANSCRIBED from the shipped manifest's `limits.fps: [24]` for
/// `krea_realtime_14b` (`config/builtin.models.jsonc`) — the same single-valued cadence declaration
/// `MINIMAX_H3_FPS` makes in code. A fixture that declares any other cadence is refused, so a record
/// cannot silently carry a playback rate the model never produced.
const KREA_REALTIME_FPS: u32 = 24;
/// Model evaluations per calibration render, overriding the config's `denoising_step_list` length.
/// Two is the smallest count that still produces a real denoise interval between the first `Step`
/// and `Decoding`, which is what makes the middle phase boundary a measurement rather than an
/// artifact — the same reason [`MINIMAX_STEPS`] is 2.
const KREA_REALTIME_STEPS: u32 = 2;
/// z16 Wan VAE temporal compression. TRANSCRIBED from
/// `mlx-gen-krea-realtime/src/t2v.rs:74-75` (`TEMPORAL_STRIDE = VAE_TILING.temporal_scale`) and
/// `t2v.rs:124-131` (`latent_frame_count(n) = (n − 1) / TEMPORAL_STRIDE + 1`): both `TEMPORAL_STRIDE`
/// and `latent_frame_count` are `pub(crate)` in the pinned crate, so there is no symbol to read.
/// `the_realtime_latent_derivation_matches_the_engines_own_arithmetic` re-derives the same values
/// from the documented formula so a transcription drift is caught locally.
const KREA_REALTIME_TEMPORAL_STRIDE: u32 = 4;
/// Model-local ceiling on the full latent clip the AR loop allocates. TRANSCRIBED from
/// `mlx-gen-krea-realtime/src/t2v.rs:78` (`pub(crate) const MAX_LATENT_FRAMES: usize = 257`),
/// enforced by `bounded_latent_frame_count` at `t2v.rs:134-143`. `pub(crate)`, so unreadable from
/// here; 257 latent frames is 1,028 source/output frames.
const KREA_REALTIME_MAX_LATENT_FRAMES: u32 = 257;
/// Latent frames the autoregressive loop denoises as ONE chunk. TRANSCRIBED from
/// `mlx-gen-krea-realtime/src/config.rs:165` + `:268`
/// (`KreaArConfig::num_frames_per_block: 3`), which is a field on a config this arm does not build
/// and therefore cannot read without constructing the engine's preset. Reported as
/// `autoregressiveBlocks`, never asserted away: the planned 45 frames is 12 latent frames, exactly
/// four whole blocks.
const KREA_REALTIME_LATENT_FRAMES_PER_BLOCK: u32 = 3;
/// Krea Realtime 14B determinism thresholds, in [0,1] units. Spelled as their own literals for the
/// reason stated at [`MINIMAX_MAX_THRESHOLD`]: the record embeds them, and a
/// `mlx:krea_realtime_14b` receipt must not be traceable to a constant asserting another provider's
/// provenance. The claim is the same kind — repeat determinism on one loaded provider — so the
/// magnitudes match, but a fully-seeded few-step AR render is expected to be bit-identical while the
/// mandatory broad-bias mutation must breach all three.
const KREA_REALTIME_MAX_THRESHOLD: f64 = 3.0 / 255.0;
const KREA_REALTIME_MEAN_THRESHOLD: f64 = 1.0 / 255.0;
const KREA_REALTIME_RMS_THRESHOLD: f64 = 1.5 / 255.0;

#[derive(Debug, PartialEq, Eq)]
struct WiredLimit {
    bytes: u64,
    source: &'static str,
}

#[cfg(test)]
#[path = "mlx_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScopedGenerationFailureKind {
    Canceled,
    Error,
    Finish,
}

type AfterConfigurationHook = fn(&mut GenerationRequest) -> Result<(), String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhaseMemory {
    active: u64,
    cache: u64,
}

/// The physical quantity an MLX calibration lane promotes into admission evidence.
///
/// Image evidence deliberately preserves its historical componentwise co-existence bound. That
/// bound adds peak-active-over-window to end-of-phase reclaimable cache, so it is conservative but
/// not a simultaneous resident measurement. Moving those records to `ResidentActive` would loosen
/// shipped image admission without a replacement campaign (sc-19115). Video evidence uses the
/// resident active peak because staged video leaves an enormous elastic cache at phase boundaries;
/// charging that cache would make successful LTX captures inadmissible on their capture host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PredictedPeakBasis {
    HistoricalImageAllocatorBound,
    ResidentActive,
}

const IMAGE_PREDICTED_PEAK_BASIS: PredictedPeakBasis =
    PredictedPeakBasis::HistoricalImageAllocatorBound;
const VIDEO_PREDICTED_PEAK_BASIS: PredictedPeakBasis = PredictedPeakBasis::ResidentActive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PredictedPhasePeaks {
    conditioning: u64,
    denoise: u64,
    decode: u64,
    overall: u64,
}

impl PredictedPhasePeaks {
    fn json(self) -> Value {
        json!({
            "conditioning": self.conditioning,
            "denoise": self.denoise,
            "decode": self.decode,
            "overall": self.overall,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AllocatorState {
    active: u64,
    cache: u64,
}

/// The live `mlx_rs::memory` counters behind [`protocol::ResidencyCounters`], so a phase window's
/// opening order is the one the adapter crate proves on every host (sc-22667 D3).
struct MlxResidencyCounters;

impl protocol::ResidencyCounters for MlxResidencyCounters {
    fn clear_cache(&mut self) {
        clear_cache();
    }
    fn reset_peak(&mut self) {
        reset_peak_memory();
    }
    fn active(&self) -> u64 {
        get_active_memory() as u64
    }
    fn cache(&self) -> u64 {
        get_cache_memory() as u64
    }
    fn peak(&self) -> u64 {
        get_peak_memory() as u64
    }
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

    /// TWO DIFFERENT INSTANTS, and every consumer of `allocatorBytes` has to know it (sc-18810
    /// review). `active` is `get_peak_memory()` — the MAXIMUM over the phase window since the last
    /// `reset_peak_memory`. `cache` is `get_cache_memory()` — the INSTANTANEOUS reading at the
    /// moment of capture, i.e. the end of the phase. MLX exposes no "cache at the active peak", so
    /// their sum is an UPPER BOUND on what co-existed, not a simultaneous maximum: during an LTX
    /// decode the cache is ~0 on entry and grows monotonically to its end-of-phase value, so the
    /// bound is loosest exactly where it is largest.
    fn capture() -> Self {
        Self {
            active: get_peak_memory() as u64,
            cache: get_cache_memory() as u64,
        }
    }

    /// ONE NAMED MLX QUANTITY PER FIELD, and nothing else (sc-18864).
    ///
    /// MLX exposes exactly two per-phase counters, so the record carries exactly two counters plus
    /// their documented sum. Schema v4 also carried `deviceBytes` and `wiredBytes`, and this
    /// function set BOTH of them to `active + cache` — the same number as `allocatorBytes`, under
    /// two names that claim to be different quantities. Neither claim was measurable: MLX has no
    /// device-residency counter and no wired-residency counter, and Metal's
    /// `recommendedMaxWorkingSetSize` does not bound `active + cache` either (a completing LTX
    /// render co-existed 7.46 GiB above it, sc-18810). The consequence was a physically impossible
    /// record: `wiredBytes` 99.7-159.6 GB against a probed `wiredLimitBytes` of 87.0 GB on every
    /// committed MLX capture. Schema v5 removes both fields rather than inventing readings for
    /// them.
    ///
    /// - `activeBytes` is [`get_peak_memory`] — the MAXIMUM live-array byte count since the last
    ///   `reset_peak_memory`, i.e. the phase window. This is the quantity MLX's own
    ///   [`get_memory_limit`] enforces, and the only one a hardware or wired ceiling may be
    ///   checked against.
    /// - `reclaimableBytes` is [`get_cache_memory`] — the allocator's free-buffer cache, read
    ///   INSTANTANEOUSLY at the phase boundary. MLX releases it under pressure.
    /// - `allocatorBytes` is their sum, and is an UPPER BOUND ON CO-EXISTENCE, not a simultaneous
    ///   maximum: it adds a peak-over-window to an instantaneous-at-boundary reading, and MLX
    ///   exposes no "cache at the active peak". During an LTX decode the cache is ~0 on entry and
    ///   grows monotonically to its end-of-phase value, so the bound is loosest exactly where it
    ///   is largest.
    fn json(self) -> Value {
        json!({
            "activeBytes": self.active,
            "allocatorBytes": self.allocator_bytes(),
            "reclaimableBytes": self.cache,
        })
    }
}

/// One member of the Z-Image family this arm measures, resolved from the plan's
/// `(target.provider, target.mode)` — never assumed. Three members today: Turbo text-to-image,
/// Turbo edit (the `z_image_edit` catalog alias) and the undistilled base (sc-22724).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ZImageArm {
    /// The registry id handed to `catalog.media().load` — the production loader (E4).
    provider: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>-shared-ladder`.
    slug: &'static str,
    /// Condition every request on one reference image (`Conditioning::Reference`) and measure
    /// under `MemoryMode::Edit`.
    edit: bool,
}

const Z_IMAGE_TURBO_ARM: ZImageArm = ZImageArm {
    provider: Z_IMAGE_PROVIDER,
    execution_path: Z_IMAGE_PLAIN_EXECUTION_PATH,
    still_calibration: "MLX Z-Image base calibration",
    repository_env: "SCENEWORKS_Z_IMAGE_REPOSITORY",
    revision_env: "SCENEWORKS_Z_IMAGE_REVISION",
    root_env: "SCENEWORKS_Z_IMAGE_ROOT",
    expected_repository: protocol::Z_IMAGE_REPOSITORY,
    slug: "z-image",
    edit: false,
};

const Z_IMAGE_EDIT_ARM: ZImageArm = ZImageArm {
    provider: Z_IMAGE_PROVIDER,
    execution_path: Z_IMAGE_EDIT_EXECUTION_PATH,
    still_calibration: "MLX Z-Image edit calibration",
    repository_env: "SCENEWORKS_Z_IMAGE_REPOSITORY",
    revision_env: "SCENEWORKS_Z_IMAGE_REVISION",
    root_env: "SCENEWORKS_Z_IMAGE_ROOT",
    expected_repository: protocol::Z_IMAGE_REPOSITORY,
    slug: "z-image-edit",
    edit: true,
};

const Z_IMAGE_BASE_ARM: ZImageArm = ZImageArm {
    provider: Z_IMAGE_BASE_PROVIDER,
    execution_path: Z_IMAGE_BASE_PLAIN_EXECUTION_PATH,
    still_calibration: "MLX Z-Image base-model calibration",
    repository_env: "SCENEWORKS_Z_IMAGE_BASE_REPOSITORY",
    revision_env: "SCENEWORKS_Z_IMAGE_BASE_REVISION",
    root_env: "SCENEWORKS_Z_IMAGE_BASE_ROOT",
    expected_repository: protocol::Z_IMAGE_BASE_REPOSITORY,
    slug: "z-image-base",
    edit: false,
};

/// The artifact one Z-Image capture loads: the env-bound repository and revision, the PLANNED
/// tier, and the `LoadSpec` that opens exactly that tier's snapshot directory.
#[derive(Debug)]
struct ZImageArtifact {
    arm: ZImageArm,
    repository: String,
    revision: String,
    tier: &'static str,
    spec: LoadSpec,
}

impl ZImageArtifact {
    fn loadability_fingerprint(&self) -> String {
        format!("{}@{}:{}", self.repository, self.revision, self.tier)
    }
}

// ---------------------------------------------------------------------------------------------
// The FLUX.1 family arm (sc-22726): `flux_dev`, `flux_schnell` and `pulid_flux_dev`.
// ---------------------------------------------------------------------------------------------

/// One member of the FLUX.1 family this arm measures, resolved from the plan's
/// `(target.provider, target.mode)` — never assumed. Three members: the two base text-to-image
/// providers of `mlx-gen-flux`, and the PuLID-FLUX character route of `mlx-gen-pulid`, which loads
/// the SAME FLUX.1-dev backbone artifact with an identity stack threaded through
/// `LoadSpec::identity` (worker `image_jobs/pulid.rs:448`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FluxOneArm {
    /// The registry id handed to `catalog.media().load` — the production loader (E4). This is the
    /// id the worker passes to `inference_runtime::load` (`engines.rs` MODEL_TABLE `engine_id`,
    /// `image_jobs/pulid.rs` `PULID_ENGINE_ID`).
    provider: &'static str,
    /// The plan `target.mode` this member serves.
    mode: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>-shared-ladder`.
    slug: &'static str,
    /// The PuLID character route: `overlay: identity`, one reference, and an identity stack on the
    /// `LoadSpec`. The base routes are plain reference-free text-to-image.
    identity: bool,
}

const FLUX1_DEV_PROVIDER: &str = "flux1_dev";
const FLUX1_SCHNELL_PROVIDER: &str = "flux1_schnell";
/// The PuLID-FLUX registry id. The CATALOG model id is `pulid_flux_dev`; the ENGINE id is the bare
/// `pulid_flux` (`mlx-gen-pulid` `pulid_flux::MODEL_ID`), which is why the anchor-loader closure
/// declares it as an `engineId` alias.
const PULID_FLUX_PROVIDER: &str = "pulid_flux";
/// The base-route seed. One constant for the family: the fixture binds tier and edge, so the seed
/// does not also have to carry the route.
const FLUX1_SEED: u64 = 22726;
/// The identity route's `character_image` reference strength — the manifest's
/// `ui.referenceStrengthDefault` for `pulid_flux_dev`, which is what the worker sends.
const PULID_REFERENCE_STRENGTH: f32 = 1.0;

const FLUX1_DEV_ARM: FluxOneArm = FluxOneArm {
    provider: FLUX1_DEV_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX FLUX.1-dev base-only text-to-image path",
    still_calibration: "MLX FLUX.1-dev calibration",
    repository_env: "SCENEWORKS_FLUX1_DEV_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX1_DEV_REVISION",
    root_env: "SCENEWORKS_FLUX1_DEV_ROOT",
    expected_repository: protocol::FLUX1_DEV_REPOSITORY,
    slug: "flux1-dev",
    identity: false,
};

const FLUX1_SCHNELL_ARM: FluxOneArm = FluxOneArm {
    provider: FLUX1_SCHNELL_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX FLUX.1-schnell base-only text-to-image path",
    still_calibration: "MLX FLUX.1-schnell calibration",
    repository_env: "SCENEWORKS_FLUX1_SCHNELL_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX1_SCHNELL_REVISION",
    root_env: "SCENEWORKS_FLUX1_SCHNELL_ROOT",
    expected_repository: protocol::FLUX1_SCHNELL_REPOSITORY,
    slug: "flux1-schnell",
    identity: false,
};

const PULID_FLUX_ARM: FluxOneArm = FluxOneArm {
    provider: PULID_FLUX_PROVIDER,
    mode: "character_image",
    execution_path: "the MLX PuLID-FLUX identity-conditioned character path",
    still_calibration: "MLX PuLID-FLUX calibration",
    // The backbone IS `SceneWorks/flux1-dev-mlx` on this route too, so it binds the same env
    // family. What makes it a different measurement is the identity stack, which rides the
    // `LoadSpec` from its own bundle env and is priced as a separate resident component.
    repository_env: "SCENEWORKS_FLUX1_DEV_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX1_DEV_REVISION",
    root_env: "SCENEWORKS_FLUX1_DEV_ROOT",
    expected_repository: protocol::FLUX1_DEV_REPOSITORY,
    slug: "pulid-flux",
    identity: true,
};

/// The artifact one FLUX.1 capture loads: the env-bound repository and revision, the PLANNED tier,
/// the identity bundle on the character route, and the `LoadSpec` that opens exactly that tier.
#[derive(Debug)]
struct FluxOneArtifact {
    arm: FluxOneArm,
    repository: String,
    revision: String,
    tier: &'static str,
    /// `Some` on the identity route only; the CONTENT digest of its five files is part of the
    /// loadability fingerprint (never the host path it was staged at), because two different
    /// identity stacks over the same backbone are two different measurements and a record must
    /// name which one it measured on any host.
    identity: Option<protocol::PulidIdentityBundle>,
    spec: LoadSpec,
}

impl FluxOneArtifact {
    fn loadability_fingerprint(&self) -> String {
        let base = format!("{}@{}:{}", self.repository, self.revision, self.tier);
        match &self.identity {
            Some(bundle) => format!("{base}+identity:{}", bundle.composite_sha256),
            None => base,
        }
    }

    /// The backbone snapshot plus, on the identity route, the ordered identity-file inventory digest.
    fn artifact_json(&self) -> Value {
        let mut artifact = json!({
            "repository": self.repository,
            "resolvedRevision": self.revision,
            "variant": self.tier,
        });
        if let Some(bundle) = &self.identity {
            artifact["inventorySha256"] = json!(bundle.composite_sha256);
        }
        artifact
    }
}

// ---------------------------------------------------------------------------------------------
// The turnkey still-image arm (sc-22732): `kolors`, `ideogram_4`, `ideogram_4_turbo`, `lens` and
// `lens_turbo`. Five catalog models over three engine crates, one arm.
// ---------------------------------------------------------------------------------------------

/// One artifact family a turnkey capture may bind: the env triple the harness exports and the
/// repository id the record must name.
///
/// A member normally has exactly one. Ideogram has two, because its bf16 tier ships from a
/// DIFFERENT repository at a DIFFERENT revision than its packed `q4`/`q8` tiers
/// (`image_jobs/artifact_resolution.rs` `IDEOGRAM_BF16_REPO`, and the manifest's own third `downloads[]` entry).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TurnkeyFamily {
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// Whether the load root is `<snapshot>/<tier>` (`true`) or the snapshot ROOT itself (`false`).
    /// Only Qwen-Image 2.1's bf16 tier is untiered: it IS the released upstream snapshot, which has
    /// no tier sub-directory (sc-24112).
    tiered: bool,
}

/// One member of the turnkey still-image family this arm measures, resolved from the plan's
/// `(target.provider, target.mode)` — never assumed.
///
/// All five are plain reference-free text-to-image routes whose text encoder, transformer and
/// decoder are packed INSIDE the per-tier snapshot (Kolors' ChatGLM3-6B, Lens' gpt-oss-20b MoE,
/// Ideogram's Qwen3-VL-8B), so a capture opens exactly one weights root and binds no second
/// repository — unlike `minimax_h3`, whose rehost is not self-sufficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TurnkeyArm {
    /// The registry id handed to `catalog.media().load` — the production loader (E4). For all five
    /// members the engine id EQUALS the catalog model id
    /// (`crates/sceneworks-worker/src/engines.rs` MODEL_TABLE), which is why no anchor-loader
    /// closure row needs an `engineId` alias.
    provider: &'static str,
    /// The plan `target.mode` this member serves. Every member is text-to-image: Kolors and
    /// Ideogram also advertise edit routes, and Kolors a character route, but those are separate
    /// providers or separate measurements and no anchor plans them.
    mode: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    /// The artifact family for the packed tiers, and for bf16 too unless [`Self::bf16_family`]
    /// overrides it.
    family: TurnkeyFamily,
    /// `Some` only where the bf16 tier is a different repository — Ideogram. Binding bf16 through
    /// the packed family would name the wrong repo AND the wrong revision in the record's
    /// loadability fingerprint, which is the one claim about the snapshot nothing downstream can
    /// re-derive.
    bf16_family: Option<TurnkeyFamily>,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>-shared-ladder`, and the fixture
    /// prefix.
    slug: &'static str,
}

const KOLORS_PROVIDER: &str = "kolors";
const IDEOGRAM_PROVIDER: &str = "ideogram_4";
const IDEOGRAM_TURBO_PROVIDER: &str = "ideogram_4_turbo";
const LENS_PROVIDER: &str = "lens";
const LENS_TURBO_PROVIDER: &str = "lens_turbo";
/// Qwen-Image 2.1 (sc-24112). The engine id equals the catalog model id
/// (`crates/sceneworks-worker/src/engines.rs` MODEL_TABLE).
const QWEN_IMAGE_2_1_PROVIDER: &str = "qwen_image_2_1";
/// Qwen-Image 2.1 Turbo (sc-25907, epic 25899): the distilled sibling's OWN engine id, registered by
/// the same `mlx-gen-qwen-image-2-1` crate (`TURBO_MODEL_ID`) and equal to its catalog model id.
const QWEN_IMAGE_2_1_TURBO_PROVIDER: &str = "qwen_image_2_1_turbo";
/// The seed every turnkey capture renders at. The fixture binds member, tier and edge, so the seed
/// does not also have to carry the route.
const TURNKEY_SEED: u64 = 22732;
/// Two steps, the shared still-image phase-boundary pattern: the first Step callback closes a
/// conservative conditioning envelope and the second gives denoise its own measured interval before
/// Decoding.
const TURNKEY_STEPS: u32 = 2;

const KOLORS_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_KOLORS_REPOSITORY",
    revision_env: "SCENEWORKS_KOLORS_REVISION",
    root_env: "SCENEWORKS_KOLORS_ROOT",
    expected_repository: protocol::KOLORS_REPOSITORY,
    tiered: true,
};

/// The packed `q4`/`q8` Ideogram turnkey, shared BY BOTH Ideogram members at the same revision.
const IDEOGRAM_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_IDEOGRAM_REPOSITORY",
    revision_env: "SCENEWORKS_IDEOGRAM_REVISION",
    root_env: "SCENEWORKS_IDEOGRAM_ROOT",
    expected_repository: protocol::IDEOGRAM_REPOSITORY,
    tiered: true,
};

const IDEOGRAM_BF16_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_IDEOGRAM_BF16_REPOSITORY",
    revision_env: "SCENEWORKS_IDEOGRAM_BF16_REVISION",
    root_env: "SCENEWORKS_IDEOGRAM_BF16_ROOT",
    expected_repository: protocol::IDEOGRAM_BF16_REPOSITORY,
    tiered: true,
};

const LENS_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_LENS_REPOSITORY",
    revision_env: "SCENEWORKS_LENS_REVISION",
    root_env: "SCENEWORKS_LENS_ROOT",
    expected_repository: protocol::LENS_REPOSITORY,
    tiered: true,
};

const LENS_TURBO_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_LENS_TURBO_REPOSITORY",
    revision_env: "SCENEWORKS_LENS_TURBO_REVISION",
    root_env: "SCENEWORKS_LENS_TURBO_ROOT",
    expected_repository: protocol::LENS_TURBO_REPOSITORY,
    tiered: true,
};

/// The packed Qwen-Image 2.1 `q8/` / `q4/` re-host (sc-24112).
const QWEN_IMAGE_2_1_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_QWEN_IMAGE_2_1_REPOSITORY",
    revision_env: "SCENEWORKS_QWEN_IMAGE_2_1_REVISION",
    root_env: "SCENEWORKS_QWEN_IMAGE_2_1_ROOT",
    expected_repository: protocol::QWEN_IMAGE_2_1_REPOSITORY,
    tiered: true,
};

/// The Qwen-Image 2.1 bf16 tier: the upstream `Qwen/Qwen-Image-2.1` snapshot at its ROOT, which is
/// the directory the worker's `qwen_image_2_1_declared_tier_dir` hands the engine for bf16.
const QWEN_IMAGE_2_1_BF16_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_QWEN_IMAGE_2_1_BF16_REPOSITORY",
    revision_env: "SCENEWORKS_QWEN_IMAGE_2_1_BF16_REVISION",
    root_env: "SCENEWORKS_QWEN_IMAGE_2_1_BF16_ROOT",
    expected_repository: protocol::QWEN_IMAGE_2_1_BF16_REPOSITORY,
    tiered: false,
};

/// The packed Qwen-Image 2.1 Turbo `q8/` / `q4/` re-host (sc-25907): its OWN repository and env
/// family, split from base's the way Lens-Turbo is split from Lens, so a Turbo plan can never be
/// satisfied by base weights.
const QWEN_IMAGE_2_1_TURBO_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_REPOSITORY",
    revision_env: "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_REVISION",
    root_env: "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_ROOT",
    expected_repository: protocol::QWEN_IMAGE_2_1_TURBO_REPOSITORY,
    tiered: true,
};

/// The Qwen-Image 2.1 Turbo bf16 tier: the upstream `Qwen/Qwen-Image-2.1-Turbo` snapshot at its
/// ROOT, the directory the worker hands the engine for Turbo bf16 (base's untiered shape).
const QWEN_IMAGE_2_1_TURBO_BF16_FAMILY: TurnkeyFamily = TurnkeyFamily {
    repository_env: "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_REPOSITORY",
    revision_env: "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_REVISION",
    root_env: "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_ROOT",
    expected_repository: protocol::QWEN_IMAGE_2_1_TURBO_BF16_REPOSITORY,
    tiered: false,
};

const KOLORS_ARM: TurnkeyArm = TurnkeyArm {
    provider: KOLORS_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX Kolors base-only text-to-image path",
    still_calibration: "MLX Kolors calibration",
    family: KOLORS_FAMILY,
    bf16_family: None,
    slug: "kolors",
};

const IDEOGRAM_ARM: TurnkeyArm = TurnkeyArm {
    provider: IDEOGRAM_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX Ideogram 4 base-only text-to-image path",
    still_calibration: "MLX Ideogram 4 calibration",
    family: IDEOGRAM_FAMILY,
    bf16_family: Some(IDEOGRAM_BF16_FAMILY),
    slug: "ideogram-4",
};

const IDEOGRAM_TURBO_ARM: TurnkeyArm = TurnkeyArm {
    provider: IDEOGRAM_TURBO_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX Ideogram 4 Turbo base-only text-to-image path",
    still_calibration: "MLX Ideogram 4 Turbo calibration",
    // Same two repositories as the base member, at the same revisions: the turbo tier is the same
    // snapshot plus the `turbo_lora.safetensors` the engine installs at load. What makes it a
    // different measurement is the provider, not the artifact.
    family: IDEOGRAM_FAMILY,
    bf16_family: Some(IDEOGRAM_BF16_FAMILY),
    slug: "ideogram-4-turbo",
};

const LENS_ARM: TurnkeyArm = TurnkeyArm {
    provider: LENS_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX Lens base-only text-to-image path",
    still_calibration: "MLX Lens calibration",
    family: LENS_FAMILY,
    bf16_family: None,
    slug: "lens",
};

const LENS_TURBO_ARM: TurnkeyArm = TurnkeyArm {
    provider: LENS_TURBO_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX Lens-Turbo base-only text-to-image path",
    still_calibration: "MLX Lens-Turbo calibration",
    // Its OWN rehost at its OWN revision, split from base Lens the way `flux1_schnell` is split
    // from `flux1_dev`, so a turbo plan can never be satisfied by base weights.
    family: LENS_TURBO_FAMILY,
    bf16_family: None,
    slug: "lens-turbo",
};

/// Qwen-Image 2.1 (sc-24112): Ideogram's split-repo shape with the halves swapped — the packed
/// tiers are the SceneWorks re-host and bf16 is the upstream snapshot. Upstream ships one pipeline
/// whose text-to-image face is the same call with no condition images, which is what this measures;
/// the reference/edit faces are separate admission contexts no anchor plans.
const QWEN_IMAGE_2_1_ARM: TurnkeyArm = TurnkeyArm {
    provider: QWEN_IMAGE_2_1_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX Qwen-Image 2.1 base-only text-to-image path",
    still_calibration: "MLX Qwen-Image 2.1 calibration",
    family: QWEN_IMAGE_2_1_FAMILY,
    bf16_family: Some(QWEN_IMAGE_2_1_BF16_FAMILY),
    slug: "qwen-image-2-1",
};

/// Qwen-Image 2.1 Turbo (sc-25907): base's split-repo shape over the distilled checkpoint's OWN two
/// repositories. Same reference-free text-to-image measurement as base; the engine refuses the
/// other checkpoint's snapshot by name, and the families keep the env binding from ever offering it.
const QWEN_IMAGE_2_1_TURBO_ARM: TurnkeyArm = TurnkeyArm {
    provider: QWEN_IMAGE_2_1_TURBO_PROVIDER,
    mode: "text_to_image",
    execution_path: "the MLX Qwen-Image 2.1 Turbo base-only text-to-image path",
    still_calibration: "MLX Qwen-Image 2.1 Turbo calibration",
    family: QWEN_IMAGE_2_1_TURBO_FAMILY,
    bf16_family: Some(QWEN_IMAGE_2_1_TURBO_BF16_FAMILY),
    slug: "qwen-image-2-1-turbo",
};

impl TurnkeyArm {
    fn family_for(self, tier: &str) -> TurnkeyFamily {
        match (tier, self.bf16_family) {
            ("bf16", Some(family)) => family,
            _ => self.family,
        }
    }
}

/// The artifact one turnkey capture loads: the env-bound repository and revision for the PLANNED
/// tier's family, and the `LoadSpec` that opens exactly that tier.
#[derive(Debug)]
struct TurnkeyArtifact {
    arm: TurnkeyArm,
    repository: String,
    revision: String,
    tier: &'static str,
    spec: LoadSpec,
}

impl TurnkeyArtifact {
    fn loadability_fingerprint(&self) -> String {
        format!("{}@{}:{}", self.repository, self.revision, self.tier)
    }

    fn artifact_json(&self) -> Value {
        json!({
            "repository": self.repository,
            "resolvedRevision": self.revision,
            "variant": self.tier,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The SANA and Chroma1 family arm (sc-22731): `sana_1600m`, `sana_sprint_1600m`, `chroma1_hd`,
// `chroma1_base`, `chroma1_flash`.
// ---------------------------------------------------------------------------------------------

/// Which engine crate a member belongs to. The two families share one arm because the measurement
/// is the same shape — a plain, reference-free, still text-to-image resident capture off a per-tier
/// SceneWorks turnkey — but they publish their production identities from different tables and
/// their loads are shaped differently, so the family stays in the type rather than being inferred
/// from the provider id at each use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SanaChromaFamily {
    /// `mlx-gen-sana`. Its identity key carries the OFFLOAD POLICY as well as (route, tier) —
    /// rung 4 is declared only on `Sequential`, so the two policies publish genuinely different
    /// ladders (inference PR 951).
    Sana,
    /// `mlx-gen-chroma`. Its identity key is (route, tier) alone; the policy and the load shape are
    /// deliberately not in it.
    Chroma,
}

/// One member of the two families this arm measures, resolved from the plan's `target.provider`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SanaChromaArm {
    family: SanaChromaFamily,
    /// The registry id handed to `catalog.media().load` — the production loader (E4), and the same
    /// id the worker passes to `inference_runtime::load` (these five catalog ids ARE their engine
    /// ids, `engines.rs` MODEL_TABLE).
    provider: &'static str,
    /// The route segment the engine's identity table interpolates: `base`/`sprint` for SANA,
    /// `hd`/`base`/`flash` for Chroma1.
    route: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// The record's diagnostics source and fixture prefix, `<slug>-mlx-<tier>-<edge>-seed…`.
    slug: &'static str,
    /// The materialization shape every plan row of this member names, and the one the loaded
    /// generator's calibration identity must carry (sc-22731 review). The two families genuinely
    /// differ: `mlx-gen-sana` publishes its resident ladder EAGER — the windowed rung 4 is declared
    /// only under `OffloadPolicy::Sequential`, which no anchor plans — while `mlx-gen-chroma`'s
    /// shared ladder is deferred at every route and tier. Carried as a FIELD so the plan rows, the
    /// LoadSpec and the capture cannot each spell it independently.
    load_shape: LoadShape,
}

const SANA_PROVIDER: &str = "sana_1600m";
const SANA_SPRINT_PROVIDER: &str = "sana_sprint_1600m";
const CHROMA1_HD_PROVIDER: &str = "chroma1_hd";
const CHROMA1_BASE_PROVIDER: &str = "chroma1_base";
const CHROMA1_FLASH_PROVIDER: &str = "chroma1_flash";
/// One seed for both families: the fixture binds route, tier and edge, so the seed does not also
/// have to carry the route.
const SANA_CHROMA_SEED: u64 = 22731;

const SANA_ARM: SanaChromaArm = SanaChromaArm {
    family: SanaChromaFamily::Sana,
    provider: SANA_PROVIDER,
    route: "base",
    execution_path: "the MLX SANA 1.6B base-only text-to-image path",
    still_calibration: "MLX SANA 1.6B calibration",
    repository_env: "SCENEWORKS_SANA_REPOSITORY",
    revision_env: "SCENEWORKS_SANA_REVISION",
    root_env: "SCENEWORKS_SANA_ROOT",
    expected_repository: protocol::SANA_REPOSITORY,
    slug: "sana-1600m",
    load_shape: LoadShape::EagerMaterialization,
};

const SANA_SPRINT_ARM: SanaChromaArm = SanaChromaArm {
    family: SanaChromaFamily::Sana,
    provider: SANA_SPRINT_PROVIDER,
    route: "sprint",
    execution_path: "the MLX SANA-Sprint 1.6B base-only text-to-image path",
    still_calibration: "MLX SANA-Sprint 1.6B calibration",
    repository_env: "SCENEWORKS_SANA_SPRINT_REPOSITORY",
    revision_env: "SCENEWORKS_SANA_SPRINT_REVISION",
    root_env: "SCENEWORKS_SANA_SPRINT_ROOT",
    expected_repository: protocol::SANA_SPRINT_REPOSITORY,
    slug: "sana-sprint",
    load_shape: LoadShape::EagerMaterialization,
};

const CHROMA1_HD_ARM: SanaChromaArm = SanaChromaArm {
    family: SanaChromaFamily::Chroma,
    provider: CHROMA1_HD_PROVIDER,
    route: "hd",
    execution_path: "the MLX Chroma1-HD base-only text-to-image path",
    still_calibration: "MLX Chroma1-HD calibration",
    repository_env: "SCENEWORKS_CHROMA1_HD_REPOSITORY",
    revision_env: "SCENEWORKS_CHROMA1_HD_REVISION",
    root_env: "SCENEWORKS_CHROMA1_HD_ROOT",
    expected_repository: protocol::CHROMA1_HD_REPOSITORY,
    slug: "chroma1-hd",
    load_shape: LoadShape::DeferredMaterialization,
};

const CHROMA1_BASE_ARM: SanaChromaArm = SanaChromaArm {
    family: SanaChromaFamily::Chroma,
    provider: CHROMA1_BASE_PROVIDER,
    route: "base",
    execution_path: "the MLX Chroma1-Base base-only text-to-image path",
    still_calibration: "MLX Chroma1-Base calibration",
    repository_env: "SCENEWORKS_CHROMA1_BASE_REPOSITORY",
    revision_env: "SCENEWORKS_CHROMA1_BASE_REVISION",
    root_env: "SCENEWORKS_CHROMA1_BASE_ROOT",
    expected_repository: protocol::CHROMA1_BASE_REPOSITORY,
    slug: "chroma1-base",
    load_shape: LoadShape::DeferredMaterialization,
};

const CHROMA1_FLASH_ARM: SanaChromaArm = SanaChromaArm {
    family: SanaChromaFamily::Chroma,
    provider: CHROMA1_FLASH_PROVIDER,
    route: "flash",
    execution_path: "the MLX Chroma1-Flash base-only text-to-image path",
    still_calibration: "MLX Chroma1-Flash calibration",
    repository_env: "SCENEWORKS_CHROMA1_FLASH_REPOSITORY",
    revision_env: "SCENEWORKS_CHROMA1_FLASH_REVISION",
    root_env: "SCENEWORKS_CHROMA1_FLASH_ROOT",
    expected_repository: protocol::CHROMA1_FLASH_REPOSITORY,
    slug: "chroma1-flash",
    load_shape: LoadShape::DeferredMaterialization,
};

/// Every member this arm serves, in one place so the tables and the tests cannot disagree about
/// which five cells exist.
const SANA_CHROMA_ARMS: [SanaChromaArm; 5] = [
    SANA_ARM,
    SANA_SPRINT_ARM,
    CHROMA1_HD_ARM,
    CHROMA1_BASE_ARM,
    CHROMA1_FLASH_ARM,
];

/// The artifact one SANA/Chroma1 capture loads: the env-bound repository and revision, the PLANNED
/// tier, and the `LoadSpec` that opens exactly that tier.
#[derive(Debug)]
struct SanaChromaArtifact {
    arm: SanaChromaArm,
    repository: String,
    revision: String,
    tier: &'static str,
    spec: LoadSpec,
}

impl SanaChromaArtifact {
    fn loadability_fingerprint(&self) -> String {
        format!("{}@{}:{}", self.repository, self.revision, self.tier)
    }

    fn artifact_json(&self) -> Value {
        json!({
            "repository": self.repository,
            "resolvedRevision": self.revision,
            "variant": self.tier,
        })
    }
}

/// One clip-versus-clip comparison, in [0,1] units, plus the two REFERENCES it was taken against.
///
/// A recovery render is compared twice on purpose. Against the measured (cold, first) clip it
/// answers the record's question — "is the clip after a fault the clip this record characterizes?"
/// — and against the clean warm control (render two, the render immediately BEFORE the fault) it
/// answers the diagnostic one: a recovery that matches the clean warm control but not the measured
/// clip is a cold-versus-warm difference that the fault did not cause, while a recovery that
/// matches NEITHER, on a pair whose own clean warm comparison passed, isolates the fault.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LtxClipDelta {
    maximum: f64,
    mean: f64,
    rms: f64,
}

impl LtxClipDelta {
    fn new((maximum, mean, rms): (f64, f64, f64)) -> Self {
        Self { maximum, mean, rms }
    }

    /// The SAME rule the warm repeat is judged by — [`ltx_quality_passes`], one helper, no second
    /// literal — so a recovery comparison can never be stricter than the determinism check whose
    /// envelope it borrows.
    fn passes(self) -> bool {
        ltx_quality_passes(self.maximum, self.mean, self.rms)
    }

    /// Thousandths of one 0-255 level. The record's integer measurement channel would round a
    /// sub-level delta to 0 or 1 and lose exactly the magnitude a reader needs.
    fn milli_per_255(value: f64) -> u64 {
        (value * 255.0 * 1000.0).round().max(0.0) as u64
    }
}

/// One recovery render's evidence: the clip delta against the measured clip and against the clean
/// warm control, and the phase the fault was injected at.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LtxRecoveryEvidence {
    vs_measured: LtxClipDelta,
    vs_clean_warm: LtxClipDelta,
    /// What the recovery clip IS, not just how far it is from the reference. A delta alone cannot
    /// tell a clip that collapsed — the shape a post-fault render takes when the state it reads was
    /// released rather than rebuilt — from one that merely rendered something else.
    clip: LtxClipSummary,
}

/// A clip in its own terms: whether its first frame is a single constant value, and its mean level.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LtxClipSummary {
    degenerate_first_frame: bool,
    mean_level: f64,
}

impl LtxClipSummary {
    fn of(frames: &[Image]) -> Self {
        let degenerate_first_frame = frames.first().is_none_or(|frame| {
            frame.pixels.is_empty() || frame.pixels.iter().all(|pixel| *pixel == frame.pixels[0])
        });
        let mut sum = 0.0_f64;
        let mut samples = 0_usize;
        for frame in frames {
            for &pixel in &frame.pixels {
                sum += f64::from(pixel) / 255.0;
                samples += 1;
            }
        }
        Self {
            degenerate_first_frame,
            mean_level: if samples == 0 {
                0.0
            } else {
                sum / samples as f64
            },
        }
    }
}

/// One member of the FLUX.2 family this arm measures, resolved from the plan's
/// `(target.provider, target.modelId)` — never assumed (sc-22727). Three members today: the 32B
/// `flux2_dev` flagship and the two klein-9B catalog models, which share ONE engine provider id
/// and are told apart by their artifact family and `LoadSpec::resolved_route`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Flux2Arm {
    /// The registry id handed to the production catalog loader (E4).
    provider: &'static str,
    /// The CATALOG model id. It is the anchor key's `modelId`, and it is also what the worker binds
    /// as `LoadSpec::resolved_route` (`image_jobs/load_spec.rs` — `spec.with_resolved_route(request.model)`),
    /// which is the discriminator `KleinArtifactInventory::validate_resolved_route` refuses a
    /// cross-variant artifact with. Two members share `provider` and differ only here.
    model_id: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// The fixture's family segment: `flux2-<fixture_slug>-mlx-<tier>-<edge>-seed<seed>-step2`.
    fixture_slug: &'static str,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>-resident`.
    slug: &'static str,
    /// The WEIGHTS-FREE registry declaration identity for this member's provider — the string
    /// `config/manifests/builtin.models.jsonc` declares on every rung row. Distinct from the
    /// identity a load of real weights publishes on the klein routes (sc-22727); see
    /// [`flux2_calibration_fingerprint`], which is what the plan row and the capture bind.
    registry_fingerprint: &'static str,
    /// The load shape and offload policy captured for this member's plain T2I rung.
    ///
    /// Dev: `Resident` + `EagerMaterialization`, the resident rung this arm captures. The provider
    /// also supports request-selected staging, but this capture deliberately sends unscoped
    /// resident requests and compares them with warm resident repeats.
    ///
    /// Klein: `Resident` + `EagerMaterialization` — the shape of the RUNG THIS ANCHOR PRICES.
    ///
    /// Both klein manifest entries also declare an MLX `bounded_transformer_residency` row with
    /// `requiredOffloadPolicy: "sequential"` for every tier of the plain T2I route, and
    /// `evaluate_declared_mlx_load_shape_for_request` + `apply_declared_mlx_load_policy_for_request`
    /// bind `Applied` + Deferred + Sequential when that rung is the one being engaged. That is the
    /// BTR rung's shape, not the resident rung's: a deferred klein transformer CANNOT render
    /// without a window — `mlx-gen-flux2/src/transformer.rs:949-955` refuses with "flux2: a
    /// deferred transformer requires an explicit block window" the moment `block_stream` is set and
    /// no window is selected. Measured, not reasoned: the sc-22727 proof capture of
    /// `flux2_klein_9b:q4:mlx` at inference `d5270c68` cleared the whole contract-and-load path
    /// under `Sequential` + `Deferred` and then died on exactly that sentence, 123 s in, at the
    /// resident-rung render. The resident rung therefore loads eager, which is what this arm
    /// mirrors.
    ///
    /// sc-22727 / inference PR 948 is what makes that measurable at all: `klein_contract_from_parts`
    /// now publishes a production identity for EVERY load shape, keyed on the admitted inventory
    /// and `resolved_quant()`, while rung 4 stays gated on `klein_streamable` (exactly
    /// `Sequential && DeferredMaterialization && quantize.is_none()`). Before it, a resident klein
    /// spec yielded `calibration: None` and this arm had no identity to bind.
    load_shape: LoadShape,
    offload_policy: OffloadPolicy,
    seed: u64,
    /// The story that established this member's evidence, as it appears in
    /// `MemoryRunContext::evidence_revision`. Per member, so a klein receipt never claims the
    /// dev lane's provenance.
    evidence_tag: &'static str,
    /// Whether this arm captures only the resident rung. True on Dev: its unscoped request and
    /// warm-repeat evidence do not measure the provider's supported staged lifecycle. False on
    /// klein: its manifest declares five rungs, and under the eager shape this arm loads
    /// (sc-22727) the contract implements resident
    /// plus bounded decode/attention while staged residency and rung 4 stay `Missing` — they need
    /// the streamable shape. Which of those a plan may select is left to
    /// `contract.validate_selection` rather than being second-guessed here.
    captures_resident_only: bool,
    /// Whether the planned tier's quant reaches the loader as `LoadSpec::quantize`.
    ///
    /// The dev route takes it: its loader folds the requested width. The klein TURNKEY rehosts do
    /// NOT — their tier is declared by the snapshot directory itself
    /// (`turnkey_identity` reads `…/snapshots/<rev>/<tier>` and
    /// `verify_turnkey_with_contracts` validates the transformer headers against it), and the same
    /// function refuses any spec that also carries a quant: "flux2 Klein turnkey tiers require BF16
    /// execution with LoadSpec.quantize=None"
    /// (`mlx-gen-flux2/src/artifact_inventory.rs`). Measured, not assumed: the sc-22727 proof
    /// capture of `flux2_klein_9b:q4:mlx` failed on exactly that sentence while reading the
    /// contract, before any weights were loaded.
    ///
    /// The tier is still DERIVED from the plan and still binds the artifact — through the root
    /// suffix check, which for these rehosts is the tier declaration the engine itself reads.
    tier_quant_reaches_the_loader: bool,
}

const FLUX2_DEV_ARM: Flux2Arm = Flux2Arm {
    provider: FLUX2_PROVIDER,
    model_id: "flux2_dev",
    execution_path: FLUX2_PLAIN_EXECUTION_PATH,
    still_calibration: "MLX FLUX.2-dev calibration",
    repository_env: "SCENEWORKS_FLUX2_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX2_REVISION",
    root_env: "SCENEWORKS_FLUX2_ROOT",
    expected_repository: protocol::FLUX2_REPOSITORY,
    fixture_slug: "dev",
    slug: "flux2-dev",
    registry_fingerprint: FLUX2_CALIBRATION_FINGERPRINT,
    load_shape: LoadShape::EagerMaterialization,
    offload_policy: OffloadPolicy::Resident,
    seed: FLUX2_SEED,
    evidence_tag: "sc-18218",
    captures_resident_only: true,
    tier_quant_reaches_the_loader: true,
};

const FLUX2_KLEIN_ARM: Flux2Arm = Flux2Arm {
    provider: FLUX2_KLEIN_PROVIDER,
    model_id: "flux2_klein_9b",
    execution_path: FLUX2_KLEIN_PLAIN_EXECUTION_PATH,
    still_calibration: "MLX FLUX.2-klein-9B calibration",
    repository_env: "SCENEWORKS_FLUX2_KLEIN_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX2_KLEIN_REVISION",
    root_env: "SCENEWORKS_FLUX2_KLEIN_ROOT",
    expected_repository: protocol::FLUX2_KLEIN_REPOSITORY,
    fixture_slug: "klein-9b",
    slug: "flux2-klein-9b",
    registry_fingerprint: FLUX2_KLEIN_REGISTRY_FINGERPRINT,
    load_shape: LoadShape::EagerMaterialization,
    offload_policy: OffloadPolicy::Resident,
    seed: FLUX2_KLEIN_SEED,
    evidence_tag: "sc-22727",
    captures_resident_only: false,
    tier_quant_reaches_the_loader: false,
};

const FLUX2_KLEIN_KV_ARM: Flux2Arm = Flux2Arm {
    provider: FLUX2_KLEIN_PROVIDER,
    model_id: "flux2_klein_9b_kv",
    execution_path: FLUX2_KLEIN_KV_PLAIN_EXECUTION_PATH,
    still_calibration: "MLX FLUX.2-klein-9B KV calibration",
    repository_env: "SCENEWORKS_FLUX2_KLEIN_KV_REPOSITORY",
    revision_env: "SCENEWORKS_FLUX2_KLEIN_KV_REVISION",
    root_env: "SCENEWORKS_FLUX2_KLEIN_KV_ROOT",
    expected_repository: protocol::FLUX2_KLEIN_KV_REPOSITORY,
    fixture_slug: "klein-9b-kv",
    slug: "flux2-klein-9b-kv",
    registry_fingerprint: FLUX2_KLEIN_REGISTRY_FINGERPRINT,
    load_shape: LoadShape::EagerMaterialization,
    offload_policy: OffloadPolicy::Resident,
    seed: FLUX2_KLEIN_SEED,
    evidence_tag: "sc-22727",
    captures_resident_only: false,
    tier_quant_reaches_the_loader: false,
};

/// The admission context for the FLUX.2-dev safety scenarios. It exactly describes the base
/// text-to-image route: `MemoryMode::TextToImage`, no reference, and `reference_count == 0`.
/// `overlay` stays `None` because this authoritative lane is base-only.
/// The three levers an admission SCENARIO varies, bundled so the context builder keeps one
/// parameter per concern. `fingerprint` is a parameter at all only so the stale-evidence probe can
/// pass a deliberate mismatch; every real call site passes `calibration.fingerprint` (the Krea-arm
/// lesson at `krea_context`).
#[derive(Clone, Copy, Debug)]
struct Flux2AdmissionProbe<'a> {
    fingerprint: &'a str,
    total_bytes: u64,
    predicted_peak_bytes: u64,
}

impl<'a> Flux2AdmissionProbe<'a> {
    /// The exact-fit probe: the calibrated fingerprint, and the measured peak as both the budget
    /// and the prediction.
    fn exact_fit(calibration: &'a MemoryCalibrationIdentity, predicted: u64) -> Self {
        Self {
            fingerprint: &calibration.fingerprint,
            total_bytes: predicted,
            predicted_peak_bytes: predicted,
        }
    }
}

/// One member of the Krea 2 base family this arm measures, resolved from the plan's
/// `target.provider` — never assumed. Two members today: the distilled Turbo text-to-image lane and
/// the undistilled Raw base (sc-22735). Both live in `mlx-gen-krea` and share the shipped
/// reference-free text-to-image route; everything that differs between them is a field here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KreaBaseArm {
    /// The registry id handed to `registry.load` — the production loader (E4).
    provider: &'static str,
    execution_path: &'static str,
    /// The still-geometry and target refusal label.
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>`.
    slug: &'static str,
    seed: u64,
    /// The fixture prefix stem: `<stem>-<tier>-<width>-seed<seed>-step2`.
    fixture_stem: &'static str,
    /// The materialization shape the PLAN must declare for this member, and the one this arm
    /// executes. A property of the member rather than a constant: Turbo's shipped route defers, Raw's
    /// materializes eagerly, and a plan row naming the other shape is refused by name.
    load_shape: LoadShape,
    /// Whether [`krea_base_load_spec`] puts the planned tier's `Quant` on the `LoadSpec`.
    ///
    /// **FALSE for Raw, and that is the point of this field.** The shipped worker passes
    /// `LoadSpec::quantize = None` for the Krea image engines at EVERY tier —
    /// `crates/sceneworks-worker/src/image_jobs/quantization.rs`,
    /// `mlx_load_quant_for_resolved_artifact` returns `None` for
    /// `"krea_2_raw" | "krea_2_turbo" | "krea_2_edit" | …` — and the tier travels instead on
    /// `MlxRequestPlan::with_resolved_artifact_tier`, i.e. on the directory the spec opens. Turbo
    /// keeps `true` because its cells are already-priced anchors measured under that spec and are not
    /// this story's to re-measure; Raw is captured in the worker's own shape.
    load_spec_carries_quant: bool,
}

const KREA_TURBO_ARM: KreaBaseArm = KreaBaseArm {
    provider: KREA_BASE_PROVIDER,
    execution_path: KREA_PLAIN_EXECUTION_PATH,
    still_calibration: "MLX Krea base calibration",
    repository_env: "SCENEWORKS_KREA_REPOSITORY",
    revision_env: "SCENEWORKS_KREA_REVISION",
    root_env: "SCENEWORKS_KREA_ROOT",
    expected_repository: protocol::KREA_REPOSITORY,
    slug: "krea-base-shared-ladder",
    seed: KREA_BASE_SEED,
    fixture_stem: "krea-base-mlx",
    load_shape: LoadShape::DeferredMaterialization,
    load_spec_carries_quant: true,
};

const KREA_RAW_ARM: KreaBaseArm = KreaBaseArm {
    provider: KREA_RAW_PROVIDER,
    execution_path: KREA_RAW_PLAIN_EXECUTION_PATH,
    still_calibration: "MLX Krea 2 Raw calibration",
    repository_env: "SCENEWORKS_KREA_RAW_REPOSITORY",
    revision_env: "SCENEWORKS_KREA_RAW_REVISION",
    root_env: "SCENEWORKS_KREA_RAW_ROOT",
    expected_repository: protocol::KREA_RAW_REPOSITORY,
    slug: "krea-raw-shared-ladder",
    seed: KREA_RAW_SEED,
    fixture_stem: "krea-raw-mlx",
    load_shape: LoadShape::EagerMaterialization,
    load_spec_carries_quant: false,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KreaBaseLifecycleMetrics {
    clean_warm_peak: u64,
    clean_post_cleanup: AllocatorState,
    max_fault_post_cleanup: AllocatorState,
    max_recovery_peak: u64,
    max_recovery_post_cleanup: AllocatorState,
}

/// One member of the MLX SDXL family (sc-22729).
///
/// The worker routes five catalog models onto the single `sdxl` engine id
/// (`crates/sceneworks-worker/src/engines.rs` `MODEL_TABLE`), each with its own independently
/// pinned tiered rehost. `mlx-gen-sdxl` registers only `MODEL_ID = "sdxl"` and mints ONE
/// route-independent fingerprint, so the arm is keyed on `(provider, modelId)`: the provider names
/// the engine, the model id names the artifact family the capture must actually open. Binding the
/// engine id alone would let a `realvisxl` plan be satisfied by base-SDXL weights and re-label base
/// SDXL's peaks as the finetune's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SdxlArm {
    /// The catalog model id — `planned.target.modelId`, and the spec's `resolved_route` (the
    /// worker binds exactly this at `image_jobs/quantization.rs`).
    model_id: &'static str,
    execution_path: &'static str,
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// The `planned.fixture` stem, so a capture of one family member can never be attributed to
    /// another (`<prefix>-<tier>-<width>-seed<n>-step2`).
    fixture_prefix: &'static str,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>-shared-ladder`.
    slug: &'static str,
}

const SDXL_BASE_ARM: SdxlArm = SdxlArm {
    model_id: "sdxl",
    execution_path: SDXL_PLAIN_EXECUTION_PATH,
    still_calibration: "MLX SDXL base calibration",
    repository_env: "SCENEWORKS_SDXL_REPOSITORY",
    revision_env: "SCENEWORKS_SDXL_REVISION",
    root_env: "SCENEWORKS_SDXL_ROOT",
    expected_repository: protocol::SDXL_REPOSITORY,
    fixture_prefix: "sdxl-base-mlx",
    slug: "sdxl",
};

const REALVISXL_ARM: SdxlArm = SdxlArm {
    model_id: "realvisxl",
    execution_path: "the MLX RealVisXL base-only text-to-image path",
    still_calibration: "MLX RealVisXL base calibration",
    repository_env: "SCENEWORKS_REALVISXL_REPOSITORY",
    revision_env: "SCENEWORKS_REALVISXL_REVISION",
    root_env: "SCENEWORKS_REALVISXL_ROOT",
    expected_repository: protocol::REALVISXL_REPOSITORY,
    fixture_prefix: "realvisxl-mlx",
    slug: "realvisxl",
};

const REALVISXL_LIGHTNING_ARM: SdxlArm = SdxlArm {
    model_id: "realvisxl_lightning",
    execution_path: "the MLX RealVisXL Lightning base-only text-to-image path",
    still_calibration: "MLX RealVisXL Lightning base calibration",
    repository_env: "SCENEWORKS_REALVISXL_LIGHTNING_REPOSITORY",
    revision_env: "SCENEWORKS_REALVISXL_LIGHTNING_REVISION",
    root_env: "SCENEWORKS_REALVISXL_LIGHTNING_ROOT",
    expected_repository: protocol::REALVISXL_LIGHTNING_REPOSITORY,
    fixture_prefix: "realvisxl-lightning-mlx",
    slug: "realvisxl-lightning",
};

const ILLUSTRIOUS_XL_V1_ARM: SdxlArm = SdxlArm {
    model_id: "illustrious_xl_v1",
    execution_path: "the MLX Illustrious-XL v1 base-only text-to-image path",
    still_calibration: "MLX Illustrious-XL v1 base calibration",
    repository_env: "SCENEWORKS_ILLUSTRIOUS_XL_V1_REPOSITORY",
    revision_env: "SCENEWORKS_ILLUSTRIOUS_XL_V1_REVISION",
    root_env: "SCENEWORKS_ILLUSTRIOUS_XL_V1_ROOT",
    expected_repository: protocol::ILLUSTRIOUS_XL_V1_REPOSITORY,
    fixture_prefix: "illustrious-xl-v1-mlx",
    slug: "illustrious-xl-v1",
};

const ILLUSTRIOUS_XL_V2_ARM: SdxlArm = SdxlArm {
    model_id: "illustrious_xl_v2",
    execution_path: "the MLX Illustrious-XL v2 base-only text-to-image path",
    still_calibration: "MLX Illustrious-XL v2 base calibration",
    repository_env: "SCENEWORKS_ILLUSTRIOUS_XL_V2_REPOSITORY",
    revision_env: "SCENEWORKS_ILLUSTRIOUS_XL_V2_REVISION",
    root_env: "SCENEWORKS_ILLUSTRIOUS_XL_V2_ROOT",
    expected_repository: protocol::ILLUSTRIOUS_XL_V2_REPOSITORY,
    fixture_prefix: "illustrious-xl-v2-mlx",
    slug: "illustrious-xl-v2",
};

const SDXL_FAMILY: [SdxlArm; 5] = [
    SDXL_BASE_ARM,
    REALVISXL_ARM,
    REALVISXL_LIGHTNING_ARM,
    ILLUSTRIOUS_XL_V1_ARM,
    ILLUSTRIOUS_XL_V2_ARM,
];

// ----------------------------------------------------------------------------------------------
// InstantID (sc-22729)
// ----------------------------------------------------------------------------------------------

/// Everything one InstantID capture loads, resolved from the environment. The backbone is the
/// PLAIN `realvisxl` tiered rehost (`image_jobs/instantid.rs` `INSTANTID_SDXL_REPO`); the identity
/// stack is fetched on first use by the worker rather than declared as a manifest download, so both
/// halves are bound through the same env seams the worker itself reads.
struct InstantIdBinding {
    repository: String,
    revision: String,
    paths: mlx_gen_instantid::InstantIdPaths,
    bundle: protocol::InstantIdIdentityBundle,
    artifact_fingerprint: String,
}

// ---------------------------------------------------------------------------------------------
// Qwen-Image-Edit-2511 (sc-22728) — one engine provider, two shipped catalog ids
// ---------------------------------------------------------------------------------------------

/// One member of the Qwen edit family this arm measures, resolved from the plan's
/// `(target.provider, target.modelId)` — never assumed. Both members load the SAME engine provider
/// from the SAME artifact family; the Lightning member additionally stacks the built-in distill LoRA
/// and runs the recipe that LoRA was distilled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QwenEditArm {
    /// The catalog id, which is what the anchor key is keyed on.
    model_id: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    /// The `<prefix>-<tier>-seed<n>-step<n>` fixture spelling this member's plan rows must use, so a
    /// Lightning capture can never be recorded under the production id's fixture and vice versa.
    fixture_prefix: &'static str,
    /// The overlay the plan must declare for this member.
    overlay: &'static str,
    /// Stack the built-in distill LoRA and select the `lightning` engine recipe.
    lightning: bool,
    steps: u32,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>-shared-ladder`.
    slug: &'static str,
}

const QWEN_EDIT_ARM: QwenEditArm = QwenEditArm {
    model_id: QWEN_EDIT_MODEL,
    execution_path: QWEN_EDIT_EXECUTION_PATH,
    still_calibration: "MLX Qwen edit calibration",
    fixture_prefix: "qwen-edit-mlx",
    overlay: "none",
    lightning: false,
    steps: QWEN_EDIT_STEPS,
    slug: "qwen-edit",
};

const QWEN_EDIT_LIGHTNING_ARM: QwenEditArm = QwenEditArm {
    model_id: QWEN_EDIT_LIGHTNING_MODEL,
    execution_path: QWEN_EDIT_LIGHTNING_EXECUTION_PATH,
    still_calibration: "MLX Qwen edit Lightning calibration",
    fixture_prefix: "qwen-edit-lightning-mlx",
    overlay: "lora",
    lightning: true,
    steps: QWEN_EDIT_LIGHTNING_STEPS,
    slug: "qwen-edit-lightning",
};

use protocol::QwenEditArtifactSource;

use protocol::QwenEditArtifact;

// ---------------------------------------------------------------------------------------------
// Mage-Flow (sc-22733) — six registered engine providers on ONE arm
// ---------------------------------------------------------------------------------------------

/// The square edge every checked-in `mage_flow*:*:mlx` plan row declares, restated here so the
/// tests can drive the arm at the geometry the campaign will.
///
/// It is NOT a production input: [`run_mage_provider`] takes width and height from the plan row
/// (`protocol::target_geometry`) and never from a constant, so an anchor re-planned at another
/// legal edge is measured at that edge rather than silently relabelled.
///
/// 768 is what the plan declares because `mlx-gen-mage`'s `config::MIN_SIZE` is 512 and both sides
/// must be multiples of `SIZE_MULTIPLE` (16), so 768 is a legal native resolution — and it is the
/// edge the family's retired SC-15509 identity was measured at (`model.rs`
/// `production_calibration_fingerprint` doc: "At 768²/one step, …"), kept so the per-tier cells
/// this arm measures are comparable with that record.
/// `every_planned_mage_mlx_row_names_the_production_identity_and_is_checked_before_the_load` binds
/// this constant to all 18 plan rows, so the claim cannot go stale in prose.
#[cfg(test)]
const MAGE_EDGE: u32 = 768;
/// The seed every checked-in `mage_flow*:*:mlx` plan row spells into its fixture
/// (`<prefix>-<tier>-768-seed22733-step<n>`), bound to those rows by the same test.
///
/// Also not a production input: [`planned_mage_seed`] PARSES the seed out of the plan's own
/// fixture name — the fixture is the single source — and only checks that the fixture's prefix
/// names this member, this tier and the rendered edge, and that its step count is the arm's recipe.
#[cfg(test)]
const MAGE_SEED: u64 = 22733;
/// Mage's quality claim is the same KIND as FLUX.2-dev's: the resident anchor selects no alternate
/// code path, so the selected render and the unselected reference are the same computation on one
/// loaded provider and must agree to within Metal allocator jitter. The envelope is restated under
/// Mage names because the record embeds it as `maximumErrorThreshold`/`meanErrorThreshold` — a
/// `mage_flow` receipt must not be traceable to a constant asserting a FLUX.2 provenance.
const MAGE_MAX_THRESHOLD: f64 = 3.0 / 255.0;
const MAGE_MEAN_THRESHOLD: f64 = 1.0 / 255.0;

/// One member of the Mage-Flow family this arm measures.
///
/// Unlike the Qwen edit family (one engine provider, two catalog ids), Mage is SIX registered
/// engine providers — `mlx-gen-mage` `model::MODEL_IDS`, registered one `Generator` each in
/// `lib.rs::register_providers` — each publishing its own per-tier production identity
/// ([`mage_calibration_fingerprint`], bound by the engine only to a tier PROVEN off the component
/// directories it opened), and whose catalog ids are identical to their provider ids
/// (`crates/sceneworks-worker/src/engines.rs` `MODEL_TABLE`: `sceneworks_id == engine_id` on all
/// six rows). The arm still resolves from `(provider, modelId)` and refuses any other pair by name:
/// the two are separate namespaces, and a plan that crossed them would measure one checkpoint under
/// another's anchor key.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MageArm {
    /// The engine provider id AND the catalog model id — asserted equal, never assumed.
    provider: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    /// The `<prefix>-<tier>-<edge>-seed<n>-step<n>` fixture spelling this member's plan rows must
    /// use, so a Turbo capture can never be recorded under the Base id's fixture.
    fixture_prefix: &'static str,
    /// The variant's OWN tiered rehost. Each of the six ships only the DiT; the shared text encoder
    /// and VAE come from [`protocol::MAGE_COMPONENTS_REPOSITORY`].
    repository: &'static str,
    /// The `SCENEWORKS_<env>_{REPOSITORY,REVISION,ROOT}` family naming that rehost. Spelled out as
    /// three literals rather than composed from a prefix: `measure-memory-catalog.mjs`
    /// `PROVIDER_FAMILIES` derives these exact names and exports them into this process, and its
    /// test binds the two sides by searching THIS source for each literal — a `format!`-composed
    /// name would leave a renamed family green here and fail mid-campaign, after the weights are
    /// staged.
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    /// The instruction-editing checkpoints consume exactly one reference image
    /// (`model.rs` `MageVariant::is_edit`, `edit_references`).
    edit: bool,
    /// The engine's own published default step count for this variant (`MageVariant::default_steps`)
    /// for the distilled members, whose 4-step Decoupled-DMD schedule IS the recipe the product
    /// issues; two steps for the undistilled members, the shape every other image arm here uses.
    /// The conditioning/denoise edge is cut on the engine's `Loading(Renderer)` boundary, not on
    /// the first `Progress::Step` (which fires only AFTER a full denoise step — sc-22738), so both
    /// steps land in the denoise window.
    steps: u32,
    /// `MageVariant::default_cfg`: 1.0 on the distilled members, at which the reference builds no
    /// unconditional branch at all, and 5.0 on the rest. Passed explicitly so a capture cannot
    /// silently measure a CFG branch the product does not run (or omit one it does).
    ///
    /// It is the SCALE, never the decision: whether a `guidance` reaches the engine at all is read
    /// off the loaded descriptor's `supports_guidance`, exactly as production reads it
    /// (`image_jobs/sampling.rs` `resolve_guidance`). On a distilled member the engine advertises
    /// `supports_guidance: false` and refuses ANY scale — including this 1.0 — so the value here is
    /// simply never sent. See [`mage_request`].
    guidance: f32,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>-shared-ladder`.
    slug: &'static str,
}

const MAGE_ARMS: [MageArm; 6] = [
    MageArm {
        provider: "mage_flow",
        execution_path: "the pinned MLX Mage-Flow RL text-to-image path",
        still_calibration: "MLX Mage-Flow calibration",
        fixture_prefix: "mage-flow-mlx",
        repository: protocol::MAGE_FLOW_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_ROOT",
        edit: false,
        steps: 2,
        guidance: 5.0,
        slug: "mage-flow",
    },
    MageArm {
        provider: "mage_flow_base",
        execution_path: "the pinned MLX Mage-Flow-Base text-to-image path",
        still_calibration: "MLX Mage-Flow-Base calibration",
        fixture_prefix: "mage-flow-base-mlx",
        repository: protocol::MAGE_FLOW_BASE_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_BASE_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_BASE_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_BASE_ROOT",
        edit: false,
        steps: 2,
        guidance: 5.0,
        slug: "mage-flow-base",
    },
    MageArm {
        provider: "mage_flow_turbo",
        execution_path: "the pinned MLX Mage-Flow-Turbo distilled text-to-image path",
        still_calibration: "MLX Mage-Flow-Turbo calibration",
        fixture_prefix: "mage-flow-turbo-mlx",
        repository: protocol::MAGE_FLOW_TURBO_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_TURBO_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_TURBO_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_TURBO_ROOT",
        edit: false,
        steps: 4,
        guidance: 1.0,
        slug: "mage-flow-turbo",
    },
    MageArm {
        provider: "mage_flow_edit",
        execution_path: "the pinned MLX Mage-Flow-Edit instruction-editing path",
        still_calibration: "MLX Mage-Flow-Edit calibration",
        fixture_prefix: "mage-flow-edit-mlx",
        repository: protocol::MAGE_FLOW_EDIT_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_EDIT_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_EDIT_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_EDIT_ROOT",
        edit: true,
        steps: 2,
        guidance: 5.0,
        slug: "mage-flow-edit",
    },
    MageArm {
        provider: "mage_flow_edit_base",
        execution_path: "the pinned MLX Mage-Flow-Edit-Base instruction-editing path",
        still_calibration: "MLX Mage-Flow-Edit-Base calibration",
        fixture_prefix: "mage-flow-edit-base-mlx",
        repository: protocol::MAGE_FLOW_EDIT_BASE_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_EDIT_BASE_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_EDIT_BASE_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_EDIT_BASE_ROOT",
        edit: true,
        steps: 2,
        guidance: 5.0,
        slug: "mage-flow-edit-base",
    },
    MageArm {
        provider: "mage_flow_edit_turbo",
        execution_path: "the pinned MLX Mage-Flow-Edit-Turbo distilled instruction-editing path",
        still_calibration: "MLX Mage-Flow-Edit-Turbo calibration",
        fixture_prefix: "mage-flow-edit-turbo-mlx",
        repository: protocol::MAGE_FLOW_EDIT_TURBO_REPOSITORY,
        repository_env: "SCENEWORKS_MAGE_FLOW_EDIT_TURBO_REPOSITORY",
        revision_env: "SCENEWORKS_MAGE_FLOW_EDIT_TURBO_REVISION",
        root_env: "SCENEWORKS_MAGE_FLOW_EDIT_TURBO_ROOT",
        edit: true,
        steps: 4,
        guidance: 1.0,
        slug: "mage-flow-edit-turbo",
    },
];

/// The prompt every Mage text-to-image capture renders, and the edit instruction every Mage edit
/// capture applies. Fixed with the seed, the geometry and (on the edit members) the reference, so
/// two captures of one anchor are the same request.
const MAGE_PROMPT: &str = "a weathered brass astrolabe on a linen cloth, soft window light";
const MAGE_EDIT_PROMPT: &str = "replace the background with a plain grey studio backdrop";

use protocol::MageArtifactSource;

/// The artifact one Mage capture loads: the canonical variant tier root, and the `LoadSpec` that
/// opens it.
#[derive(Debug)]
struct MageArtifact {
    root: PathBuf,
    spec: LoadSpec,
}

#[cfg(test)]
#[path = "mlx_mage_tests.rs"]
mod mage_tests;

/// The exact target geometry an `mlx:ltx_2_3` calibration case renders. `fps` is NOT part of it:
/// `GeometryEnvelope` has no temporal-cadence axis, so the arm binds fps through the fixture name
/// instead (see [`planned_ltx_capture`]). That gap is real and is reported rather than papered over —
/// the audio stream's latent length is a function of fps, so two records with identical
/// `{width, height, batch, frames}` can legitimately differ in peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LtxGeometry {
    width: u32,
    height: u32,
    frames: u32,
    /// `1 + (frames - 1) / 8` — the LTX video VAE's causal temporal depth.
    latent_frames: u32,
}

#[derive(Clone, Copy)]
struct LtxBoundedCampaignSpec {
    tier: &'static str,
    identity: &'static str,
    logical_case_id: &'static str,
    fixture: &'static str,
    inventory_files: u64,
    inventory_bytes: u64,
    inventory_sha256: &'static str,
    projection_bytes: u64,
    story: &'static str,
}

/// What the engine's OWN decode-tiling selector decides for this exact geometry.
///
/// sc-18808 recorded `bounded_decode` as "implemented but not engaged", true at the single
/// 768x512x97 geometry it captured — and the arm then hardcoded `staged_residency` for every
/// geometry. That is only safe while the sweep stays small. `budgeted_plan` engages tiling on TWO
/// independent bounds, and the claim that survives on EVERY host is one-sided — the machine-
/// independent bound is a CEILING on single-pass frames, not a prediction of where tiling starts:
///
/// * the **write bound** — `VaeTiling::LTX.writable_frame_cap(h, w) = i32::MAX / (8 * h * w)`
///   (`gen-core/src/tiling.rs:167`, `:65`, `full_res_channels: 8`). At the 0.90 MP buckets
///   (1280x704 / 704x1280) that is **297 output frames**; at the 0.39-0.41 MP buckets it is 682 /
///   655, above the declared 449-frame envelope maximum. It is a correctness bound, not a memory
///   one, so it does not move with the host.
/// * the **memory bound** — single-pass `3.3 GB + 340 B/voxel` against `get_memory_limit() * 0.85`
///   (`mlx-gen-ltx/src/pipeline.rs:218-256`, `:264-269`). This one DOES move with the host, and it
///   binds EARLIER on a smaller machine: a CI runner with a fraction of 128 GiB tiles at
///   768x512 x 97, hundreds of frames below any write cap.
///
/// So: **no host can exceed 297 single-pass output frames at 0.90 MP; smaller hosts tile earlier via
/// the memory bound.** Saying instead that tiling "engages at 298 frames on every host" is falsified
/// by this repository's own CI.
///
/// So asking the selector is the only honest way to know which rung a capture engaged, and a record
/// that claims `staged_residency` through a tiled decode is a false attestation. This calls the
/// production entry point rather than re-deriving either bound, so the arm cannot drift from the
/// engine it is measuring.
#[derive(Clone, Copy, Debug)]
struct LtxDecodePlan {
    tiling: Option<TilingConfig>,
    writable_frame_cap: i64,
}

/// The engine's own committed decode cost model (`mlx-gen-ltx/src/pipeline.rs:218-256`), restated
/// for TEST preconditions only — never to make a production decision. Production always asks
/// `auto_tiling_budgeted_ltx`, which is the point of `LtxDecodePlan`. Three call sites derived
/// these numbers inline before; one definition means a mutation of the model moves all of them.
#[cfg(test)]
/// The engine's own committed decode cost model (`mlx-gen-ltx/src/pipeline.rs:218-256`), restated
/// for TEST preconditions only — never to make a production decision. Production always asks
/// `auto_tiling_budgeted_ltx`, which is the point of `LtxDecodePlan`. Three call sites derived
/// these numbers inline before; one definition means a mutation of the model moves all of them.
#[cfg(test)]
mod ltx_decode_cost {
    pub(super) const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    /// `auto_tiling_budgeted_ltx`'s own headroom factor, applied on top of the MLX limit it reads.
    pub(super) const SAFETY_FACTOR: f64 = 0.85;
    const FIXED_BYTES: f64 = 3.3e9;
    const ACCUMULATOR_BYTES_PER_VOXEL: f64 = 40.0;
    const SINGLE_PASS_BYTES_PER_VOXEL: f64 = 340.0;

    fn voxels(width: u32, height: u32, frames: u32) -> f64 {
        f64::from(width) * f64::from(height) * f64::from(frames)
    }

    /// The full-output accumulators, which hold the assembled video and therefore cost the same at
    /// EVERY tiling. Below this the engine refuses outright; no tile size buys it back.
    pub(super) fn accumulator_floor_gib(width: u32, height: u32, frames: u32) -> f64 {
        (FIXED_BYTES + ACCUMULATOR_BYTES_PER_VOXEL * voxels(width, height, frames)) / GIB
    }

    /// One untiled decode pass.
    pub(super) fn single_pass_gib(width: u32, height: u32, frames: u32) -> f64 {
        (FIXED_BYTES + SINGLE_PASS_BYTES_PER_VOXEL * voxels(width, height, frames)) / GIB
    }
}

/// The geometry every `run_ltx` test drives (`ltx_request_json(768, 512, 97)`). It is named here
/// because an injected budget's blast radius reaches it — see `LTX_MEMORY_LIMIT_LOCK`.
#[cfg(test)]
const LTX_UNLOCKED_SMOKE_GEOMETRY: (u32, u32, u32) = (768, 512, 97);

/// Serialises every adapter-owned MLX-global memory-limit swap. Production uses it for the one
/// diagnostic canary; tests use it for injected selector budgets and restoration assertions.
///
/// **What it guarantees:** two *lock takers* never overlap. No injected-budget resolve observes
/// another's budget, and each restore pairs with its own swap.
///
/// 🔴 **What it does NOT guarantee:** that no future valid `run_ltx` test can observe an injected
/// budget. The MLX memory limit is process-GLOBAL, and production decode-plan resolution cannot
/// take this test-only lock. Cheap malformed-plan tests are deliberately ordered before that
/// selector and one holds an injected tiling budget while asserting the ordering, but any future
/// test that reaches physical resolution must either accept the host-dependent result or take this
/// lock around its own observation. Production `run_ltx` does not mutate the limit and therefore
/// does not take this lock; the diagnostic canary holds it for its whole process-global swap.
///
/// Those tests are safe only because every injected budget leaves `LTX_UNLOCKED_SMOKE_GEOMETRY`'s
/// full-output accumulators affordable — 4.49 GiB against a lowest safe budget of 6.8 GiB, a 1.51x
/// margin that nothing used to assert. `LtxInjectedBudget::install` asserts it on **every**
/// injection, so a new row below the floor fails loudly at its own injection site instead of
/// turning an unrelated test into an intermittent failure carrying the wrong message. That covers
/// future `run_ltx` call sites too, which taking the lock at today's four would not.
static LTX_MEMORY_LIMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// An injected MLX memory limit, installed for as long as this value lives.
///
/// RAII rather than a restore statement because `auto_tiling_budgeted_ltx` can panic: on that path
/// a trailing `set_memory_limit(previous)` never runs, and the injected limit leaks to every later
/// test in the process. `unwrap_or_else(PoisonError::into_inner)` then deliberately ignores the
/// poisoning, so the leak would have had no signal at all.
#[cfg(test)]
struct LtxInjectedBudget {
    previous: usize,
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl LtxInjectedBudget {
    fn install(budget_gib: f64) -> Self {
        let (width, height, frames) = LTX_UNLOCKED_SMOKE_GEOMETRY;
        let floor = ltx_decode_cost::accumulator_floor_gib(width, height, frames);
        let safe = budget_gib * ltx_decode_cost::SAFETY_FACTOR;
        assert!(
            safe > floor,
            "an injected budget of {budget_gib} GiB leaves {safe:.2} GiB safe, at or below the \
             {floor:.2} GiB accumulator floor of the {width}x{height} x {frames} smoke geometry. \
             `run_ltx` reads the process-global MLX limit WITHOUT LTX_MEMORY_LIMIT_LOCK, so a \
             concurrently running `run_ltx` refusal test would be refused by the engine and fail \
             carrying the wrong message. Raise the budget, or take this lock around every \
             `run_ltx` call site."
        );
        let guard = LTX_MEMORY_LIMIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous =
            mlx_rs::memory::set_memory_limit((budget_gib * ltx_decode_cost::GIB) as usize);
        Self {
            previous,
            _guard: guard,
        }
    }
}

#[cfg(test)]
impl Drop for LtxInjectedBudget {
    fn drop(&mut self) {
        mlx_rs::memory::set_memory_limit(self.previous);
    }
}

/// Process-global MLX limits for the one diagnostic LTX canary.
///
/// `set_memory_limit` is soft backpressure, not a sandbox; the external phys-footprint watchdog is
/// still mandatory. Setting the wired limit as well prevents MLX from pinning more unified memory
/// than the same evidence-derived ceiling. The device clamp mirrors the worker's production policy:
/// MLX's untouched memory limit is 1.5x the device wired ceiling, and asking `set_wired_limit` for
/// more than that ceiling terminates the process.
struct LtxCanaryLimits {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous_memory: usize,
    previous_wired: usize,
    wired: usize,
    restored: bool,
}

impl LtxCanaryLimits {
    fn install() -> Result<Self, String> {
        let guard = LTX_MEMORY_LIMIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let requested = usize::try_from(LTX_CANARY_MAX_FOOTPRINT_BYTES)
            .map_err(|_| "LTX canary footprint ceiling does not fit usize".to_owned())?;
        let untouched_memory_limit = get_memory_limit();
        let wired = ltx_canary_wired_limit(requested, untouched_memory_limit);
        if wired == 0 {
            return Err("LTX canary could not derive a non-zero device wired ceiling".to_owned());
        }
        let previous_memory = set_memory_limit(requested);
        let previous_wired = set_wired_limit(wired);
        Ok(Self {
            _guard: guard,
            previous_memory,
            previous_wired,
            wired,
            restored: false,
        })
    }

    fn restore(&mut self) {
        if self.restored {
            return;
        }
        set_wired_limit(self.previous_wired);
        set_memory_limit(self.previous_memory);
        self.restored = true;
    }
}

impl Drop for LtxCanaryLimits {
    fn drop(&mut self) {
        self.restore();
    }
}

impl LtxDecodePlan {
    #[cfg(test)]
    fn resolve(geometry: LtxGeometry) -> Result<Self, String> {
        Self::resolve_against_live_budget(geometry)
    }

    fn resolve_against_live_budget(geometry: LtxGeometry) -> Result<Self, String> {
        let (height, width, frames) = Self::i32_geometry(geometry)?;
        // Argument order mirrors the production call site (`pipeline.rs:168`), which passes
        // (out_h, out_w, out_f) — NOT the (out_w, out_h, out_f) spelling one of that crate's own
        // sweep tests uses. At a non-square bucket the two disagree, so this is load-bearing.
        let tiling = mlx_gen_ltx::pipeline::auto_tiling_budgeted_ltx(height, width, frames)
            .map_err(|error| Self::budget_refusal(geometry, &error.to_string()))?;
        Ok(Self {
            tiling,
            writable_frame_cap: VaeTiling::LTX.writable_frame_cap(height, width),
        })
    }

    /// Resolve the decode path that the provider will physically execute. An explicit bounded
    /// selection always routes through `decode_tiled`; the ordinary staged carrier leaves the
    /// provider to its live-budget auto selector, which may still tile on a constrained host.
    fn resolve_for_selection(
        selection: &MemorySelection,
        geometry: LtxGeometry,
    ) -> Result<Self, String> {
        if selection.strategy == MemoryStrategy::BoundedDecode {
            let tile_px = selection.parameters.decode_tile_edge.ok_or_else(|| {
                "bounded LTX decode is missing its selected spatial tile edge".to_owned()
            })?;
            let overlap_px = selection.parameters.decode_overlap.ok_or_else(|| {
                "bounded LTX decode is missing its selected spatial overlap".to_owned()
            })?;
            let (height, width, _) = Self::i32_geometry(geometry)?;
            return Ok(Self {
                tiling: Some(TilingConfig {
                    spatial: Some(SpatialTiling {
                        tile_px: i32::try_from(tile_px)
                            .map_err(|_| "LTX decode tile edge must fit i32".to_owned())?,
                        overlap_px: i32::try_from(overlap_px)
                            .map_err(|_| "LTX decode overlap must fit i32".to_owned())?,
                    }),
                    temporal: None,
                }),
                writable_frame_cap: VaeTiling::LTX.writable_frame_cap(height, width),
            });
        }
        Self::resolve_against_live_budget(geometry)
    }

    /// A campaign row may only attest the strategy it physically executes. In particular, a
    /// staged/single-pass row on a smaller host must fail closed when the provider auto-tiler
    /// engages; relabeling that render after the fact would violate the frozen plan selector.
    fn validate_selected_strategy(self, selection: &MemorySelection) -> Result<(), String> {
        let requested_tiling = selection.strategy == MemoryStrategy::BoundedDecode;
        if self.tiling.is_some() != requested_tiling {
            return Err(format!(
                "planned LTX strategy {:?} does not match the physical decode path {:?}; this host auto-selected bounded decode, so the staged single-pass row is not capturable here",
                selection.strategy,
                self.rung(),
            ));
        }
        Ok(())
    }

    fn lifecycle_fault_phase(self) -> MemoryPhase {
        if self.tiling.is_some() {
            MemoryPhase::Decode
        } else {
            MemoryPhase::Denoise
        }
    }

    fn tiling_engaged(self) -> bool {
        self.tiling.is_some()
    }

    /// The same production selector, resolved against an INJECTED memory budget instead of this
    /// machine's.
    ///
    /// `budget_gib` is the MLX memory limit the selector sees; `auto_tiling_budgeted_ltx` applies
    /// its own 0.85 safety factor on top, exactly as it does in production. The pinned crate exposes
    /// no budget-taking entry point (`plan_ltx_tiling` is private), so the budget is injected by
    /// swapping `mlx_rs::memory::set_memory_limit` around the production call and restoring the
    /// previous value — the ONE production selector still decides both bounds, rather than this arm
    /// re-deriving either of them from copied constants.
    ///
    /// Test-only, and the reason it exists: `resolve` is host-dependent by construction, so an
    /// assertion about WHERE tiling engages is otherwise either machine-specific or silently skipped
    /// on the machine (CI) where it most needs to run.
    #[cfg(test)]
    fn resolve_with_budget(geometry: LtxGeometry, budget_gib: f64) -> Result<Self, String> {
        let (height, width, frames) = Self::i32_geometry(geometry)?;
        let planned = {
            // Installed for this call only, and restored on the unwind path as well as the normal
            // one — `auto_tiling_budgeted_ltx` panicking must not leak the budget to later tests.
            let _budget = LtxInjectedBudget::install(budget_gib);
            mlx_gen_ltx::pipeline::auto_tiling_budgeted_ltx(height, width, frames)
        };
        Ok(Self {
            tiling: planned.map_err(|error| Self::budget_refusal(geometry, &error.to_string()))?,
            writable_frame_cap: VaeTiling::LTX.writable_frame_cap(height, width),
        })
    }

    fn i32_geometry(geometry: LtxGeometry) -> Result<(i32, i32, i32), String> {
        Ok((
            i32::try_from(geometry.height).map_err(|_| "LTX height must fit i32".to_owned())?,
            i32::try_from(geometry.width).map_err(|_| "LTX width must fit i32".to_owned())?,
            i32::try_from(geometry.frames).map_err(|_| "LTX frames must fit i32".to_owned())?,
        ))
    }

    fn budget_refusal(geometry: LtxGeometry, error: &str) -> String {
        format!(
            "the pinned MLX LTX-2.3 decode budget refuses {}x{} x {} frames before any render: \
             {error}",
            geometry.width, geometry.height, geometry.frames
        )
    }

    fn rung(self) -> &'static str {
        if self.tiling.is_some() {
            "bounded_decode"
        } else {
            "staged_residency"
        }
    }

    #[cfg(test)]
    fn engaged_rungs(self) -> Vec<&'static str> {
        let mut rungs = vec!["resident", "staged_residency"];
        if self.tiling.is_some() {
            rungs.push("bounded_decode");
        }
        rungs
    }

    /// The selected spatial tile edge in OUTPUT pixels, or 0 when that axis is not tiled. Reported
    /// so a rung-2 record says which plan produced its decode peak, not merely that one was chosen.
    fn spatial_tile_px(self) -> u64 {
        self.tiling
            .and_then(|config| config.spatial)
            .map_or(0, |spatial| u64::from(spatial.tile_px.max(0) as u32))
    }

    /// The selected temporal tile length in OUTPUT frames, or 0 when that axis is not tiled.
    #[cfg(test)]
    fn temporal_tile_frames(self) -> u64 {
        self.tiling
            .and_then(|config| config.temporal)
            .map_or(0, |temporal| u64::from(temporal.tile_frames.max(0) as u32))
    }

    fn spatial_overlap_px(self) -> u64 {
        self.tiling
            .and_then(|config| config.spatial)
            .map_or(0, |spatial| u64::from(spatial.overlap_px.max(0) as u32))
    }

    /// Count physical spatial decode tiles from the same shared plan geometry the provider executes.
    /// A configured tiling object alone is insufficient evidence: a carrier smaller than its tile
    /// edge can still produce a one-tile plan.
    fn spatial_tile_count(self, geometry: LtxGeometry) -> Result<u64, String> {
        let Some(config) = self.tiling else {
            return Ok(1);
        };
        let spatial_scale = u32::try_from(VaeTiling::LTX.spatial_scale)
            .map_err(|_| "LTX spatial scale must fit u32".to_owned())?;
        let latent_height = i32::try_from(geometry.height / spatial_scale)
            .map_err(|_| "LTX latent height must fit i32".to_owned())?;
        let latent_width = i32::try_from(geometry.width / spatial_scale)
            .map_err(|_| "LTX latent width must fit i32".to_owned())?;
        let latent_frames = i32::try_from(geometry.latent_frames)
            .map_err(|_| "LTX latent frames must fit i32".to_owned())?;
        let plan = config.plan(VaeTiling::LTX, latent_frames, latent_height, latent_width);
        [plan.h.len(), plan.w.len()]
            .into_iter()
            .try_fold(1_u64, |count, axis| {
                count
                    .checked_mul(
                        u64::try_from(axis)
                            .map_err(|_| "LTX decode tile-axis count must fit u64".to_owned())?,
                    )
                    .ok_or_else(|| "LTX decode tile-count arithmetic overflowed".to_owned())
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DiagnosticAudioIdentity {
    samples: u64,
    sample_rate: u32,
    channels: u16,
}

/// What the ordinary LTX-2.3 capture projects and presents to the production admission before the
/// load, so a refusal names the same figures the worker's gate would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LtxOrdinaryProjection {
    /// The worker's floor WEIGHTS term for the planned composition
    /// (`mlx_fit_gate::floor_weights_bytes`, sc-18096), from the contract's own asset facts.
    weights_bytes: u64,
    /// The engine's conservative single-pass VAE decode working set for this geometry
    /// (`mlx_gen_ltx::conservative_video_decode_memory_profile`), the same profile the worker's
    /// `packaged_video_decode_profile` resolves for a non-bounded MLX candidate.
    decode_working_set_bytes: u64,
    /// The peak presented as `predicted_peak_bytes`, exactly the worker's
    /// `profiled_floor_phase_peaks`: the larger of the generic floor (weights +
    /// [`LTX_GENERIC_HEADROOM_BYTES`], `floor_phase_peaks`) and the profile composed onto the
    /// weights (`VideoDecodeMemoryProfile::checked_composed_peak`). The generic allowance remains
    /// a lower bound: a decode-only profile cannot prove that conditioning or denoise need less.
    predicted_peak_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LtxLifecycleMetrics {
    clean_warm_peak: u64,
    clean_post_cleanup: AllocatorState,
    maximum_error: f64,
    mean_error: f64,
    rms_error: f64,
    max_fault_post_cleanup: AllocatorState,
    max_recovery_peak: u64,
    max_recovery_post_cleanup: AllocatorState,
    /// Recorded, not refused — see [`ltx_lifecycle_blockers`].
    cancel_recovery: LtxRecoveryEvidence,
    error_recovery: LtxRecoveryEvidence,
    /// The measured clip in its own terms, so a recovery summary has something to be read against.
    measured_clip: LtxClipSummary,
}

#[derive(Clone, Copy, Debug)]
struct LtxLifecycleInput {
    geometry: LtxGeometry,
    fps: u32,
    seed: u64,
    fault_phase: MemoryPhase,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LtxBoundedWarmMetrics {
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
    maximum_error: f64,
    mean_error: f64,
    rms_error: f64,
    audio: DiagnosticAudioIdentity,
}

#[derive(Debug)]
struct LtxCanaryWatchdogAttestation {
    max_footprint_bytes: u64,
    max_runtime_seconds: f64,
    host_memory_bytes: u64,
    min_initial_memory_free_bytes: u64,
    min_memory_free_bytes: u64,
    nonce: String,
    stream: Option<UnixStream>,
}

struct LtxCanaryWatchdogLease {
    writer: UnixStream,
    completion: std::sync::mpsc::Receiver<Result<(), String>>,
    phase_acknowledgements: std::sync::mpsc::Receiver<Result<(usize, String), String>>,
    nonce: String,
    phase_sequence: usize,
    expected_phases: &'static [&'static str],
}

trait LtxCampaignPhaseSink {
    fn mark(&mut self, name: &'static str) -> Result<(), String>;
}

struct NoLtxCampaignPhases;

impl LtxCampaignPhaseSink for NoLtxCampaignPhases {
    fn mark(&mut self, _name: &'static str) -> Result<(), String> {
        Ok(())
    }
}

impl LtxCanaryWatchdogAttestation {
    fn start_lease(&mut self) -> Result<LtxCanaryWatchdogLease, String> {
        self.start_lease_for(&LTX_PROVIDER_PHASE_NAMES)
    }

    fn start_lease_for(
        &mut self,
        expected_phases: &'static [&'static str],
    ) -> Result<LtxCanaryWatchdogLease, String> {
        let stream = self
            .stream
            .take()
            .ok_or_else(|| "LTX safety canary watchdog lease was already consumed".to_owned())?;
        let mut reader = stream
            .try_clone()
            .map_err(|error| format!("clone LTX watchdog lease socket: {error}"))?;
        reader
            .set_read_timeout(Some(Duration::from_secs(2)))
            .map_err(|error| format!("configure LTX watchdog lease reader: {error}"))?;
        let expected = self.nonce.clone();
        let (sender, completion) = std::sync::mpsc::sync_channel(1);
        let (phase_sender, phase_acknowledgements) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let result = loop {
                match read_watchdog_line(&mut reader) {
                    Ok(line) if line == format!("PING {expected}") => continue,
                    Ok(line) if line.starts_with(&format!("PHASE_ACK {expected} ")) => {
                        let fields = line.split_whitespace().collect::<Vec<_>>();
                        let acknowledgement = if fields.len() == 4 {
                            fields[2]
                                .parse::<usize>()
                                .map(|sequence| (sequence, fields[3].to_owned()))
                                .map_err(|_| {
                                    "SC-20216 watchdog returned a malformed phase acknowledgement"
                                        .to_owned()
                                })
                        } else {
                            Err(
                                "SC-20216 watchdog returned a malformed phase acknowledgement"
                                    .to_owned(),
                            )
                        };
                        if phase_sender.send(acknowledgement).is_err() {
                            break Err(
                                "SC-20216 phase acknowledgement receiver disappeared".to_owned()
                            );
                        }
                    }
                    Ok(line) if line == format!("BYE {expected}") => break Ok(()),
                    Ok(_) => {
                        break Err(
                            "LTX safety canary watchdog returned an invalid lease message"
                                .to_owned(),
                        )
                    }
                    Err(error) => break Err(error),
                }
            };
            if result.is_err() {
                // Loss of the monitor after GO must terminate the adapter while weights may be
                // resident. Returning to the unguarded render would repeat the incident class.
                std::process::abort();
            }
            let _ = sender.send(result);
        });
        Ok(LtxCanaryWatchdogLease {
            writer: stream,
            completion,
            phase_acknowledgements,
            nonce: self.nonce.clone(),
            phase_sequence: 0,
            expected_phases,
        })
    }
}

impl LtxCanaryWatchdogLease {
    fn complete(mut self) -> Result<(), String> {
        if self.phase_sequence != 0 && self.phase_sequence != self.expected_phases.len() {
            return Err(format!(
                "SC-20216 provider phase sequence completed at {} of {}",
                self.phase_sequence,
                self.expected_phases.len()
            ));
        }
        self.writer
            .write_all(format!("DONE {}\n", self.nonce).as_bytes())
            .map_err(|error| format!("complete LTX canary watchdog lease: {error}"))?;
        self.completion
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| {
                format!("wait for LTX canary watchdog completion release: {error}")
            })??;
        Ok(())
    }
}

impl LtxCampaignPhaseSink for LtxCanaryWatchdogLease {
    fn mark(&mut self, name: &'static str) -> Result<(), String> {
        let expected = self
            .expected_phases
            .get(self.phase_sequence)
            .ok_or_else(|| {
                "SC-20216 provider phase sequence exceeded its exact bound".to_owned()
            })?;
        if name != *expected {
            return Err(format!(
                "SC-20216 provider phase reordered: expected {expected}, got {name}"
            ));
        }
        let sequence = self.phase_sequence + 1;
        self.writer
            .write_all(format!("PHASE {} {sequence} {name}\n", self.nonce).as_bytes())
            .map_err(|error| format!("report SC-20216 provider phase {name}: {error}"))?;
        wait_for_ltx_phase_acknowledgement(
            &self.phase_acknowledgements,
            sequence,
            name,
            Duration::from_secs(2),
        )?;
        self.phase_sequence = sequence;
        Ok(())
    }
}

/// The `mlx:ltx_2_3` SC-18946 arm. SC-19109 moved strategy ownership into the provider: this path
/// reads the exact registry contract, proves the loaded generator exposes the same contract, drives
/// the selected request scope, and executes every runtime-complete admission/lifecycle scenario.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LtxRunAdmission {
    Ordinary,
    CampaignEntry,
    BoundedCampaignEntry,
}

// ==== mlx:minimax_h3 (sc-18663) =================================================================

/// The exact target geometry an `mlx:minimax_h3` calibration case renders, plus the two derived
/// latent counts the joint denoise actually carries.
///
/// `fps` is NOT a field, for the same reason it is not one on [`LtxGeometry`]: `GeometryEnvelope`
/// has no temporal-cadence axis. Here the gap is closed rather than merely reported — this model
/// generates at 24 fps and nothing else ([`mlx_gen_minimax_h3::MINIMAX_H3_FPS`]), so the fixture's
/// declared cadence is checked against that single legal value and cannot silently vary a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MinimaxGeometry {
    width: u32,
    height: u32,
    frames: u32,
    /// `17n + 5` frames ⇒ `5n + 2` video latent frames.
    video_latent_frames: u32,
    /// `round(frames / 24 · 40)` audio latent tokens, denoised jointly with the video in ONE packed
    /// sequence — so they are part of what every phase peak below is a peak OF.
    audio_latent_frames: u32,
}

/// Which MiniMax-H3 catalog entry — that is, which DiT PARTITION — a plan row measures.
///
/// `minimax_h3` and `minimax_h3_ref` are NOT two providers (`video_jobs/minimax_h3.rs`, fact 1):
/// `mlx-gen-minimax-h3` registers one `MODEL_ID`, and the two entries are two partition directories
/// of it. The engine picks between them from the CONDITIONING — upstream's
/// `MiniMaxH3Task::resolve(has_keyframes, has_references)` maps `(false, false)` to t2va on
/// `transformer/` and `(false, true)` to ref2va on `transformer_ref/` — so the request's reference
/// set is the only thing that selects the checkpoint a capture actually denoises on.
///
/// That is why this is a table rather than a boolean: the two checkpoints ship the SAME
/// `config.json` and the SAME 638 tensor names, so a capture routed at the wrong one RUNS, produces
/// plausible video, and files a record for a partition it never touched. The member fixes the mode
/// key, the reference count and the fixture slug together, and [`minimax_request`] builds the
/// conditioning from the same row — so the three cannot disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MinimaxMember {
    model_id: &'static str,
    /// The DiT partition the engine resolves for this member's conditioning. Recorded in the
    /// loadability fingerprint so two records over one artifact triple stay distinguishable.
    partition: &'static str,
    /// The `MemoryMode` key the runtime asks admission under for this member.
    mode: &'static str,
    /// The fixture name's leading token: `<fixture_prefix>-<tier>-<w>x<h>-f<frames>-fps<fps>-seed…`.
    ///
    /// Spelled per member rather than composed from a slug so the BASE entry keeps the exact
    /// `minimax-h3-mlx-…` names its plan rows already carry (sc-18663). A composed slug would have
    /// renamed three committed fixtures for no measurement reason.
    fixture_prefix: &'static str,
    /// How many references the measured request carries. `0` selects t2va, `1` selects ref2va.
    reference_count: u32,
}

/// The base t2va entry, on `transformer/`.
const MINIMAX_BASE_MEMBER: MinimaxMember = MinimaxMember {
    model_id: "minimax_h3",
    partition: "transformer",
    mode: "text_to_video",
    fixture_prefix: "minimax-h3-mlx",
    reference_count: 0,
};

/// The reference-to-video entry, on `transformer_ref/` (sc-19508).
///
/// ONE image reference, which is the smallest set `classify_reference_set` calls
/// `Conditionable`: an empty set would resolve t2va on the base checkpoint (the wrong partition for
/// this record), and an audio-only set is refused by the worker, the API and the MCP tool alike
/// because an audio reference never reaches the reference conditioner.
const MINIMAX_REFERENCE_MEMBER: MinimaxMember = MinimaxMember {
    model_id: "minimax_h3_ref",
    partition: "transformer_ref",
    mode: "reference_to_video",
    fixture_prefix: "minimax-h3-ref-mlx",
    reference_count: 1,
};

const MINIMAX_MEMBERS: [MinimaxMember; 2] = [MINIMAX_BASE_MEMBER, MINIMAX_REFERENCE_MEMBER];

/// The text encoder came from the tier tree the DiT did.
const MINIMAX_TIERED_TEXT_ENCODER: &str = "tier";
/// The text encoder came from the upstream snapshot, dense.
const MINIMAX_UPSTREAM_TEXT_ENCODER: &str = "upstream-dense";

/// Everything a capture resolved, kept together so the record's provenance is built from the values
/// the load actually used rather than from the plan.
struct MinimaxArtifact {
    repository: String,
    revision: String,
    upstream_repository: String,
    upstream_revision: String,
    upstream_root: PathBuf,
    dit_root: PathBuf,
    text_encoder_root: PathBuf,
    text_encoder_source: &'static str,
    spec: LoadSpec,
}

impl MinimaxArtifact {
    /// The two artifact triples and the text-encoder provenance in one string. A `mlx:minimax_h3`
    /// record CANNOT be identified by the rehost triple alone: two captures at the same
    /// repository, revision and tier differ materially if one took its conditioning stage from the
    /// packed tier and the other from the dense upstream snapshot.
    ///
    /// The PARTITION is part of it too (sc-22737): `minimax_h3` and `minimax_h3_ref` are two
    /// checkpoints inside one tier tree that ship identical `config.json` files and identical
    /// tensor names, so two records over the same repository, revision and tier are otherwise
    /// indistinguishable — and the engine picks between them from the conditioning rather than from
    /// anything the artifact triple can show.
    fn resolved_path_fingerprint(&self, member: MinimaxMember, tier: &str) -> String {
        format!(
            "{}@{}:{tier}+partition:{}+text_encoder:{}+shared:{}@{}",
            self.repository,
            self.revision,
            member.partition,
            self.text_encoder_source,
            self.upstream_repository,
            self.upstream_revision
        )
    }
}

// ==== mlx:bernini (sc-22737) ====================================================================
//
// The THIRD video arm, and the first that also serves a STILL catalog entry from the same engine
// provider. Three facts shape the whole block:
//
//  1. **ONE engine id, TWO catalog entries.** `mlx-gen-bernini` registers the full pipeline under
//     `crate::bernini::MODEL_ID == "bernini"`, and BOTH shipped SceneWorks entries resolve to it:
//     `video_jobs/bernini.rs#bernini_engine_id` maps the video entry `bernini`, and
//     `engines.rs`'s MODEL_TABLE maps the still entry `bernini_image` (`engine_id: "bernini"`).
//     The renderer-only sibling `bernini_renderer` (`crate::pipeline::MODEL_ID`) is registered by
//     the engine and published as its own identity family, but NO SceneWorks load path names it —
//     `video_jobs/bernini.rs` calls `inference_runtime::load("bernini")` — so this arm binds the
//     `image` identity route on both members and never the `renderer` one.
//
//  2. **The geometry envelope is the ENGINE's, and it is shared by both members.**
//     `memory_strategy::validate_geometry` admits exactly `ADVERTISED_GEOMETRIES` for either
//     provider id, so a still anchor is constrained to the same four canvases the video route
//     publishes rather than to `bernini_image`'s wider manifest resolution list. Read off the
//     engine constant, never transcribed.
//
//  3. **The tier is proven by the ARTIFACT, not asked for.** The rehost ships pre-packed `q4/`,
//     `q8/` and `bf16/` sub-directories, so the worker loads a resolved tier with `quant = None`
//     (`video_jobs/bernini.rs#resolve_bernini_tier_dir_and_quant`: "a pre-packed tier loads with
//     `quant = None`; config sidecars authoritative"). This arm does the same, and the engine's
//     `production_calibration_fingerprint` then reads the tier back out of the renderer/planner
//     manifests and the safetensors headers — which is what makes the published identity a claim
//     about the bytes on disk rather than about the plan row.
/// The engine provider id. Spelled as a LITERAL because the dispatch arm below is parsed out of
/// this source by `stale-lane-report.mjs#adapterCapturableProviders`, which resolves an arm only to
/// a bare literal or to a `const NAME: &str = "…";` — a const bound to an engine path is a value it
/// cannot read, and it refuses to guess rather than reporting this lane as having no Bernini arm.
///
/// The literal is still not a transcription that can drift: the const assertion below compares it,
/// byte for byte at COMPILE time, against the id `mlx-gen-bernini` keys its calibration route on
/// (`calibration_route(FULL_ID)`). A rename in the engine fails the build here instead of silently
/// mis-dispatching, which is the property reading `FULL_ID` directly was buying.
const BERNINI_PROVIDER: &str = "bernini";
const _: () = {
    let ours = BERNINI_PROVIDER.as_bytes();
    let engine = runtime_macos::providers::bernini::memory_strategy::FULL_ID.as_bytes();
    assert!(
        ours.len() == engine.len(),
        "BERNINI_PROVIDER no longer spells mlx-gen-bernini's memory_strategy::FULL_ID"
    );
    let mut index = 0;
    while index < ours.len() {
        assert!(
            ours[index] == engine[index],
            "BERNINI_PROVIDER no longer spells mlx-gen-bernini's memory_strategy::FULL_ID"
        );
        index += 1;
    }
};
/// The video catalog entry (`VideoModelCaps::new("bernini", …)`).
const BERNINI_VIDEO_MODEL_ID: &str = "bernini";
/// The still catalog entry, which loads [`BERNINI_PROVIDER`] with `frames == 1`.
const BERNINI_IMAGE_MODEL_ID: &str = "bernini_image";
/// One fixed seed for every `mlx:bernini` fixture, on both members. The fixture binds the member,
/// tier and full geometry, so the seed does not also have to carry the route.
const BERNINI_SEED: u64 = 22737;
/// The single cadence the manifest publishes for the video entry (`limits.fps: [16]`). Bernini has
/// no second legal rate, so — like MiniMax-H3 and unlike LTX — the fixture's declared cadence is
/// checked against one value and cannot silently vary a record.
const BERNINI_FPS: u32 = 16;
/// The video anchor's frame count: 3 s at [`BERNINI_FPS`] coerced onto the Wan `1 mod 4` lattice
/// the renderer requires (`video_jobs/wan.rs#wan_frame_count`, which the Bernini video path calls
/// because the renderer IS Wan2.2-A14B). The manifest's shortest published duration, because this
/// is the cell's ONE capture and a dual-expert A14B render is the most expensive in the catalog.
const BERNINI_VIDEO_FRAMES: u32 = 49;
const BERNINI_VIDEO_LABEL: &str = "Bernini";
/// Warm-repeat determinism envelope, shared with the other MLX video arms: an identical seed,
/// prompt, geometry and loaded provider must reproduce the clip, and the mandatory output mutation
/// must breach the same bound.
const BERNINI_MAX_THRESHOLD: f64 = 3e-2;
const BERNINI_MEAN_THRESHOLD: f64 = 3e-3;
const BERNINI_RMS_THRESHOLD: f64 = 3e-3;

/// Which Bernini catalog entry the plan asks for. Both members load the SAME engine provider from
/// the SAME artifact family; they differ only in the modality of the render and therefore in the
/// mode key the record is filed under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BerniniArm {
    model_id: &'static str,
    /// The `MemoryMode` key the runtime asks admission under for this member — an EVIDENCE KEY, not
    /// a label, for the reason [`minimax_context`] documents at length.
    mode: &'static str,
    execution_path: &'static str,
    /// `<slug>` in the fixture name `bernini-<slug>-mlx-<tier>-<w>x<h>-f<frames>-fps<fps>-seed…`.
    slug: &'static str,
    /// `frames` this member renders. The still entry is `1`; the video entry is the Wan lattice.
    frames: u32,
}

const BERNINI_VIDEO_ARM: BerniniArm = BerniniArm {
    model_id: BERNINI_VIDEO_MODEL_ID,
    mode: "text_to_video",
    execution_path: "the MLX Bernini dual-expert text-to-video path",
    slug: "video",
    frames: BERNINI_VIDEO_FRAMES,
};

const BERNINI_IMAGE_ARM: BerniniArm = BerniniArm {
    model_id: BERNINI_IMAGE_MODEL_ID,
    mode: "text_to_image",
    execution_path: "the MLX Bernini still text-to-image path",
    slug: "image",
    frames: 1,
};

/// Every member this arm serves, in one place so the tables and the tests cannot disagree about
/// which two catalog entries exist.
const BERNINI_ARMS: [BerniniArm; 2] = [BERNINI_VIDEO_ARM, BERNINI_IMAGE_ARM];

/// Every numeric tier the `mlx:bernini` family ships, and therefore every tier that has a
/// production calibration identity. Kept beside [`bernini_calibration_fingerprint`] so the identity
/// family and the plan parser cannot disagree about the inventory.
const BERNINI_TIERS: [&str; 3] = ["bf16", "q4", "q8"];

/// The tier the engine's retained
/// [`MEMORY_CALIBRATION_FINGERPRINT`](runtime_macos::providers::bernini::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT)
/// names, read off the pinned crate rather than spelled: a re-tiered retained key moves this
/// family's bare member with it, and `the_bernini_identity_family_reproduces_the_engines_retained_key`
/// proves the retained string IS the `(route, CALIBRATED_TIER)` member the format below mints.
const BERNINI_CALIBRATED_TIER: &str =
    runtime_macos::providers::bernini::memory_strategy::CALIBRATED_TIER;

/// The exact target geometry a `mlx:bernini` calibration case renders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BerniniGeometry {
    width: u32,
    height: u32,
    frames: u32,
}

/// Everything a Bernini capture resolved, kept together so the record's provenance is built from
/// the values the load actually used rather than from the plan.
struct BerniniArtifact {
    repository: String,
    revision: String,
    tier_root: PathBuf,
    spec: LoadSpec,
}

impl BerniniArtifact {
    fn resolved_path_fingerprint(&self, arm: BerniniArm, tier: &str) -> String {
        format!(
            "{}@{}:{tier}+route:{}",
            self.repository, self.revision, arm.slug
        )
    }
}

/// What the record says about the admission scenarios it did not run, and why. Stated once and
/// carried into every `not_run` reason, the lifecycle blocker and the diagnostics.
const BERNINI_ADMISSION_BLOCKER: &str = concat!(
    "reference-free Bernini has NO admission surface in production: ",
    "`video_admission.rs#bernini_memory_attempt` answers false for a reference-free text_to_video ",
    "request and the MLX still lane passes no memory context at all, so the worker builds no ",
    "MemoryRunContext and never calls the provider's safety check — and `mlx-gen-bernini`'s ",
    "route gate admits only the reference-carrying video routes. This arm therefore asks the ",
    "admission gate nothing, exactly as the worker asks it nothing: the exact-fit, unknown-budget ",
    "and stale-evidence scenarios are unexecuted rather than answered on a surface production ",
    "never presents. This anchor prices the RESIDENT load of the plain reference-free ladder and ",
    "claims nothing about any optimized rung or admission decision"
);

// ---------------------------------------------------------------------------------------------
// The SenseNova-U1 family arm (sc-22734): six catalog models over two engine ids.
// ---------------------------------------------------------------------------------------------

/// One member of the MLX SenseNova-U1 family.
///
/// The worker routes SIX catalog models onto TWO engine ids (`crates/sceneworks-worker/src/
/// engines.rs` `MODEL_TABLE`): the quality trio on `sensenova_u1_8b` and the 8-step distilled trio
/// on `sensenova_u1_8b_fast`. Each of the six has its OWN independently pinned tiered rehost, so
/// the arm is keyed on `(provider, modelId)`: the provider names the engine to load through, the
/// model id names the artifact family the capture must actually open and the route the engine
/// mints its per-route identity for. Binding the engine id alone would let an
/// `sensenova_u1_8b_infographic_v3` plan be satisfied by base-SenseNova weights and re-label the
/// base model's peaks as the finetune's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SenseNovaArm {
    /// The catalog model id — `planned.target.modelId`, and the spec's `resolved_route`. The
    /// worker binds exactly this (`image_jobs/load_spec.rs` `.with_resolved_route(request.model)`), and
    /// `mlx-gen-sensenova`'s `validate_resolved_artifact_binding` refuses a route whose repository
    /// identity the weights path does not carry.
    model_id: &'static str,
    /// The registry id handed to `catalog.media().load` — the production loader (E4), the same id
    /// the worker passes to `inference_runtime::load`.
    provider: &'static str,
    /// The route slug the engine's per-route calibration identity is spelled with, and the
    /// record's diagnostics source (`memory-mlx-adapter:sensenova-u1-<slug>-shared-ladder`).
    slug: &'static str,
    execution_path: &'static str,
    /// The still-geometry refusal label (sc-18808).
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
}

/// The quality engine id (`mlx_gen_sensenova::MODEL_ID`).
const SENSENOVA_PROVIDER: &str = "sensenova_u1_8b";
/// The 8-step distilled engine id (`mlx_gen_sensenova::MODEL_ID_FAST`). A SEPARATE registry id, not
/// a mode of the quality one: the distill LoRA is pre-merged into each `_fast` rehost at convert
/// time and the tier subdir carries the `distill_merged.json` marker `load_fast` reads, so the
/// worker attaches nothing and selects the engine instead (`engines.rs`, `image_jobs/sensenova.rs`
/// "No user adapters by design").
const SENSENOVA_FAST_PROVIDER: &str = "sensenova_u1_8b_fast";
/// The family seed. The fixture binds tier and edge, so the seed does not also carry the route.
const SENSENOVA_SEED: u64 = 22734;

const SENSENOVA_BASE_ARM: SenseNovaArm = SenseNovaArm {
    model_id: "sensenova_u1_8b",
    provider: SENSENOVA_PROVIDER,
    slug: "quality",
    execution_path: "the MLX SenseNova-U1 8B base text-to-image path",
    still_calibration: "MLX SenseNova-U1 8B calibration",
    repository_env: "SCENEWORKS_SENSENOVA_U1_8B_REPOSITORY",
    revision_env: "SCENEWORKS_SENSENOVA_U1_8B_REVISION",
    root_env: "SCENEWORKS_SENSENOVA_U1_8B_ROOT",
    expected_repository: protocol::SENSENOVA_U1_8B_REPOSITORY,
};

const SENSENOVA_INFOGRAPHIC_V2_ARM: SenseNovaArm = SenseNovaArm {
    model_id: "sensenova_u1_8b_infographic_v2",
    provider: SENSENOVA_PROVIDER,
    slug: "infographic-v2",
    execution_path: "the MLX SenseNova-U1 Infographic V2 text-to-image path",
    still_calibration: "MLX SenseNova-U1 Infographic V2 calibration",
    repository_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_REPOSITORY",
    revision_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_REVISION",
    root_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_ROOT",
    expected_repository: protocol::SENSENOVA_U1_8B_INFOGRAPHIC_V2_REPOSITORY,
};

const SENSENOVA_INFOGRAPHIC_V3_ARM: SenseNovaArm = SenseNovaArm {
    model_id: "sensenova_u1_8b_infographic_v3",
    provider: SENSENOVA_PROVIDER,
    slug: "infographic-v3",
    execution_path: "the MLX SenseNova-U1 Infographic V3 text-to-image path",
    still_calibration: "MLX SenseNova-U1 Infographic V3 calibration",
    repository_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_REPOSITORY",
    revision_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_REVISION",
    root_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_ROOT",
    expected_repository: protocol::SENSENOVA_U1_8B_INFOGRAPHIC_V3_REPOSITORY,
};

const SENSENOVA_FAST_ARM: SenseNovaArm = SenseNovaArm {
    model_id: "sensenova_u1_8b_fast",
    provider: SENSENOVA_FAST_PROVIDER,
    slug: "fast",
    execution_path: "the MLX SenseNova-U1 8B distilled text-to-image path",
    still_calibration: "MLX SenseNova-U1 8B distilled calibration",
    repository_env: "SCENEWORKS_SENSENOVA_U1_8B_FAST_REPOSITORY",
    revision_env: "SCENEWORKS_SENSENOVA_U1_8B_FAST_REVISION",
    root_env: "SCENEWORKS_SENSENOVA_U1_8B_FAST_ROOT",
    expected_repository: protocol::SENSENOVA_U1_8B_FAST_REPOSITORY,
};

const SENSENOVA_INFOGRAPHIC_V2_FAST_ARM: SenseNovaArm = SenseNovaArm {
    model_id: "sensenova_u1_8b_infographic_v2_fast",
    provider: SENSENOVA_FAST_PROVIDER,
    slug: "infographic-v2-fast",
    execution_path: "the MLX SenseNova-U1 Infographic V2 distilled text-to-image path",
    still_calibration: "MLX SenseNova-U1 Infographic V2 distilled calibration",
    repository_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST_REPOSITORY",
    revision_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST_REVISION",
    root_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST_ROOT",
    expected_repository: protocol::SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST_REPOSITORY,
};

const SENSENOVA_INFOGRAPHIC_V3_FAST_ARM: SenseNovaArm = SenseNovaArm {
    model_id: "sensenova_u1_8b_infographic_v3_fast",
    provider: SENSENOVA_FAST_PROVIDER,
    slug: "infographic-v3-fast",
    execution_path: "the MLX SenseNova-U1 Infographic V3 distilled text-to-image path",
    still_calibration: "MLX SenseNova-U1 Infographic V3 distilled calibration",
    repository_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST_REPOSITORY",
    revision_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST_REVISION",
    root_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST_ROOT",
    expected_repository: protocol::SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST_REPOSITORY,
};

const SENSENOVA_FAMILY: [SenseNovaArm; 6] = [
    SENSENOVA_BASE_ARM,
    SENSENOVA_INFOGRAPHIC_V2_ARM,
    SENSENOVA_INFOGRAPHIC_V3_ARM,
    SENSENOVA_FAST_ARM,
    SENSENOVA_INFOGRAPHIC_V2_FAST_ARM,
    SENSENOVA_INFOGRAPHIC_V3_FAST_ARM,
];

/// The progress-stream fold behind the SenseNova attribution — pure, so a synthetic stream per
/// route drives it in tests (sc-22738).
///
/// WHICH LIFECYCLE BOUNDARIES THE PINNED SENSENOVA ENGINE EMITS. `mlx-gen-sensenova/src/t2i.rs`
/// at inference e34d7b46a (#966) gives `StepReporter` the two phase boundaries beside its per-step
/// ticks: `open_denoise_phase` emits `Progress::Loading(LoadPhase::Renderer)` after every prefill
/// — the conditioning work — and before the first denoise step, in both denoise loops
/// (`T2iModel::denoise`, `T2iModel::it2i_denoise`); `open_decode_phase` emits `Progress::Decoding`
/// after the last step, before `unpatchify` + the host copy turn the RGB patches into an image.
/// `SenseNova::generate` (`src/model.rs`) hands each per-image reporter
/// `request_phase_bounds(i, count)`, so a request opens each boundary exactly ONCE whatever its
/// count — the first image's loop opens denoise, the last image's loop opens decode — and the
/// interleave route opens denoise on its first image and closes the run with a single `Decoding`
/// after the folded bar is filled (`finish_interleave_progress`). The contract declares all three
/// phases (`memory_strategy.rs` `build_memory_strategy_contract`, `[Conditioning, Denoise,
/// Decode]`).
///
/// So this fold cuts conditioning on `Loading(Renderer)`, denoise on `Decoding`, and the arm
/// closes decode at the returned image — the same three-phase cut the Mage-Flow and Bernini arms
/// use since inference #963. The first `Progress::Step` is deliberately NOT cut on: it fires only
/// after a full denoise step, with the whole backbone resident, and is not the conditioning peak.
/// The pin before this one (e16c6a55e) emitted only `Step`; this arm then filed an overall-only
/// ceiling with every phase declared unattributed. That relaxation is gone: a stream that is not
/// the one above is refused by name in [`sensenova_phase_peaks`], never filed under a stale model.
#[derive(Debug, Default)]
struct SenseNovaPhases {
    /// The conditioning peak, read at `Loading(Renderer)` — the peak since the pre-generate
    /// reset. The same closure resets the peak, so the denoise phase opens on that boundary.
    conditioning: Option<PhaseMemory>,
    /// The denoise peak, read at the `Decoding` that follows it; the reset there opens decode.
    denoise: Option<PhaseMemory>,
    /// How many times each boundary arrived. The engine opens each exactly once per request, so
    /// any other count means the stream is not the one this cut was written against.
    renderer_boundaries: u32,
    decoding_boundaries: u32,
    /// `Loading(_)` boundaries other than `Renderer`. The engine emits none; one arriving means
    /// the engine changed and the conditioning edge may no longer be where this cut takes it.
    other_loading_boundaries: u32,
}

impl SenseNovaPhases {
    /// Fold one event. `close_phase` reads the counters and resets the peak; it runs at most
    /// twice — at the first `Loading(Renderer)` and at the first `Decoding` after it.
    fn observe(&mut self, progress: &Progress, close_phase: impl FnOnce() -> PhaseMemory) {
        match progress {
            Progress::Loading(LoadPhase::Renderer) => {
                self.renderer_boundaries += 1;
                if self.conditioning.is_none() && self.denoise.is_none() {
                    self.conditioning = Some(close_phase());
                }
            }
            Progress::Decoding => {
                self.decoding_boundaries += 1;
                if self.conditioning.is_some() && self.denoise.is_none() {
                    self.denoise = Some(close_phase());
                }
            }
            Progress::Loading(_) => self.other_loading_boundaries += 1,
            Progress::Step { .. } => {}
            Progress::PromptEncodingComplete
            | Progress::PromptUpsamplingComplete
            | Progress::PostProcessing => {}
            Progress::ConditioningComplete | Progress::Upsampling { .. } => {}
        }
    }
}

/// The three declared phases the pinned engine bounds, each with a non-zero active peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SenseNovaPhasePeaks {
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
}

// -------------------------------------------------------------------------------------------
// SD3.5 family (sc-22730, epic 22723 S7)
// -------------------------------------------------------------------------------------------

/// One member of the SD3.5 family this arm measures.
///
/// The three routes are DISTINCT engine providers with the same ids on both lanes and no aliasing
/// (`crates/sceneworks-worker/src/engines.rs` sets `engine_id == sceneworks_id` for all three), and
/// each has its OWN tiered rehost. Serving one route from another's artifact would re-label that
/// route's peaks, so the env family, the expected repository literal and the published calibration
/// identity are all per member and the arm refuses an unknown `(provider, mode)` by name.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sd3Arm {
    provider: &'static str,
    execution_path: &'static str,
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
    /// The record's diagnostics source, `memory-mlx-adapter:<slug>-shared-ladder`, and the route
    /// half of the published calibration identity — see [`sd3_calibration_fingerprint`].
    slug: &'static str,
    /// True-CFG routes forward guidance and a negative prompt; the ADD-distilled Turbo advertises
    /// `supports_guidance = false`, so `resolve_guidance` returns `None` and the worker never
    /// forwards either. The request shape this arm measures is the worker's.
    guidance: Option<f32>,
}

const SD3_LARGE_PROVIDER: &str = "sd3_5_large";
const SD3_LARGE_TURBO_PROVIDER: &str = "sd3_5_large_turbo";
const SD3_MEDIUM_PROVIDER: &str = "sd3_5_medium";
// The same reasoning as `SDXL_RMS_THRESHOLD`, which this arm's parity comparison is a twin of: RMS
// is bounded by the same per-pixel envelope as maximum absolute error rather than by a tighter
// SD3.5-specific tolerance invented ahead of a physical campaign. Both arms compare a selected rung
// against the unselected request at the shared `MAX_THRESHOLD`/`MEAN_THRESHOLD` pair.
const SD3_RMS_THRESHOLD: f64 = MAX_THRESHOLD;

const SD3_LARGE_ARM: Sd3Arm = Sd3Arm {
    provider: SD3_LARGE_PROVIDER,
    execution_path: "the MLX SD3.5 Large base-only text-to-image path",
    still_calibration: "MLX SD3.5 Large base calibration",
    repository_env: "SCENEWORKS_SD3_5_LARGE_REPOSITORY",
    revision_env: "SCENEWORKS_SD3_5_LARGE_REVISION",
    root_env: "SCENEWORKS_SD3_5_LARGE_ROOT",
    expected_repository: protocol::SD3_5_LARGE_REPOSITORY,
    slug: "sd3-5-large",
    guidance: Some(3.5),
};

const SD3_LARGE_TURBO_ARM: Sd3Arm = Sd3Arm {
    provider: SD3_LARGE_TURBO_PROVIDER,
    execution_path: "the MLX SD3.5 Large Turbo base-only text-to-image path",
    still_calibration: "MLX SD3.5 Large Turbo base calibration",
    repository_env: "SCENEWORKS_SD3_5_LARGE_TURBO_REPOSITORY",
    revision_env: "SCENEWORKS_SD3_5_LARGE_TURBO_REVISION",
    root_env: "SCENEWORKS_SD3_5_LARGE_TURBO_ROOT",
    expected_repository: protocol::SD3_5_LARGE_TURBO_REPOSITORY,
    slug: "sd3-5-large-turbo",
    guidance: None,
};

const SD3_MEDIUM_ARM: Sd3Arm = Sd3Arm {
    provider: SD3_MEDIUM_PROVIDER,
    execution_path: "the MLX SD3.5 Medium base-only text-to-image path",
    still_calibration: "MLX SD3.5 Medium base calibration",
    repository_env: "SCENEWORKS_SD3_5_MEDIUM_REPOSITORY",
    revision_env: "SCENEWORKS_SD3_5_MEDIUM_REVISION",
    root_env: "SCENEWORKS_SD3_5_MEDIUM_ROOT",
    expected_repository: protocol::SD3_5_MEDIUM_REPOSITORY,
    slug: "sd3-5-medium",
    guidance: Some(5.0),
};

const SD3_ARMS: [Sd3Arm; 3] = [SD3_LARGE_ARM, SD3_LARGE_TURBO_ARM, SD3_MEDIUM_ARM];

/// The artifact one SD3.5 capture loads, and the `LoadSpec` that opens exactly that tier.
#[derive(Debug)]
struct Sd3Artifact {
    arm: Sd3Arm,
    repository: String,
    revision: String,
    tier: &'static str,
    spec: LoadSpec,
}

const SD3_SEED: u64 = 22730;

/// The geometry one `mlx:krea_realtime_14b` capture renders, with the AR loop's own derived
/// quantities carried beside the declared axes rather than asserted away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KreaRealtimeGeometry {
    width: u32,
    height: u32,
    frames: u32,
    /// `(frames − 1) / 4 + 1` — the z16 Wan VAE's 4x temporal compression.
    latent_frames: u32,
    /// Autoregressive chunks the AR loop denoises to cover `latent_frames`, at
    /// [`KREA_REALTIME_LATENT_FRAMES_PER_BLOCK`] latent frames per chunk. REPORTED, not gated: a clip
    /// whose latent count is not a whole multiple of the block size renders — the final chunk is
    /// short — so refusing one here would be a gate the engine does not have.
    autoregressive_blocks: u32,
}

/// The artifact one Krea Realtime capture loads.
#[derive(Debug)]
struct KreaRealtimeArtifact {
    repository: String,
    revision: String,
    root: PathBuf,
    spec: LoadSpec,
}

impl KreaRealtimeArtifact {
    fn resolved_path_fingerprint(&self, tier: &str) -> String {
        format!("{}@{}:{tier}", self.repository, self.revision)
    }
}

/// The admission SURFACE the plan's target declares.
///
/// Read back off the target rather than inferred from [`validate_krea_realtime_target`]'s refusals,
/// so [`krea_realtime_probes_admission`] is a decision computed from the request the arm was handed
/// and not a constant a future reference-carrying anchor could inherit silently.
struct KreaRealtimeSurface {
    mode: String,
    reference_count: u32,
}

/// What the record says about the admission scenarios it did not run, and why. Stated once and
/// carried into every `not_run` reason, the lifecycle blocker and the diagnostics.
const KREA_REALTIME_ADMISSION_BLOCKER: &str = concat!(
    "reference-free Krea Realtime has NO admission surface in production: ",
    "`mlx-gen-krea-realtime`'s registered safety_check admits only the image_to_video and ",
    "video_to_video routes carrying exactly one reference under a sealed ",
    "provider-resident-video-request-v3 receipt, and refuses every other request mode by name; ",
    "the worker's admission gate asks the provider's own safety check before attaching a run ",
    "context and drops the one it would refuse (sc-22738), and the context ",
    "video_admission.rs would build carries text_to_video, reference_count 0 and a curve/anchor ",
    "evidence id rather than that receipt. This arm therefore asks the admission gate nothing, ",
    "exactly as the worker asks it nothing: the exact-fit, unknown-budget and stale-evidence ",
    "scenarios are unexecuted rather than answered on a surface production never presents. This ",
    "anchor prices the RESIDENT load of the plain reference-free t2v ladder and claims nothing ",
    "about any optimized rung or admission decision"
);

fn run(request: &Value) -> Result<Value, String> {
    let closures = packaged_anchor_loader_closures()?;
    run_with(
        sceneworks_core::memory_anchor::packaged_memory_anchors(),
        &closures,
        request,
    )
}

/// [`run`] over a caller-supplied store and declarations, so the entry-point wiring can be driven
/// with a bound whose currency is STATED rather than whatever the packaged store carries at the
/// current pin (a packaged bound is stale on the pin after the one it was measured at, so a
/// wiring test on the packaged store alone would go green by refusing nothing).
fn run_with(
    store: Option<&sceneworks_core::memory_anchor::MemoryAnchorStore>,
    closures: &sceneworks_core::memory_anchor::AnchorLoaderClosures,
    request: &Value,
) -> Result<Value, String> {
    // sc-22738: the ONE seam every arm passes through. A cell a CURRENT footprint hard stop has
    // already bounded is refused here, before the provider is dispatched — generic by construction,
    // so no arm has to remember to ask. A STALE bound refuses nothing: the capture proceeds and
    // supersedes it (see `exceeded_bound_capture_refusal_in`).
    if let Some(store) = store {
        if let Some(refusal) = exceeded_bound_capture_refusal_in(store, closures, request)? {
            return Err(refusal);
        }
    }
    let provider = protocol::planned_provider(request)?;
    // sc-18104: this used to be `else { run_qwen_provider(request) }`, so ANY provider the MLX
    // adapter does not implement was silently routed to the Qwen arm rather than refused. It then
    // failed further in on a Qwen-shaped complaint that named neither the provider nor the missing
    // arm — measured by reverting this match, capturing `flux2_dev` reported
    // `planned.target.overlay must be a string`, which reads as a malformed plan entry and sends the
    // operator off fixing fixtures or provisioning weights for the wrong model. Refuse by name
    // instead, mirroring the Candle adapter's `candle_runtime_family::plain_execution_path`.
    match provider {
        FLUX3_PROVIDER | FLUX3_BASE_PROVIDER => mlx_flux3::run(request),
        Z_IMAGE_PROVIDER => run_z_image_reference(request),
        // sc-22724: the undistilled base rides the same arm under its own registry id, artifact
        // family and execution path; the arm resolves which member from `(provider, mode)`.
        Z_IMAGE_BASE_PROVIDER => run_z_image_reference(request),
        KREA_BASE_PROVIDER => run_krea_base(request),
        // sc-22735: the undistilled Krea 2 Raw base rides the same arm under its own registry id,
        // artifact family, planned load shape and — unlike Turbo — no `LoadSpec` quant at any tier.
        // The arm resolves which member from `target.provider`.
        KREA_RAW_PROVIDER => run_krea_base(request),
        // sc-22729: the whole SDXL family rides one engine id; the arm resolves the member from
        // `planned.target.modelId` and refuses an id no member serves.
        SDXL_PROVIDER => run_sdxl(request),
        // sc-22729: the bespoke identity route — its own crate, its own loader, no registry entry.
        INSTANTID_PROVIDER => run_instantid(request),
        KREA_PROVIDER => run_krea_control(request),
        QWEN_PROVIDER => run_qwen_provider(request),
        // sc-22728: the Qwen edit lane. One engine provider serves both shipped catalog ids, so the
        // arm resolves which from `(provider, modelId)` and refuses any other pair by name.
        QWEN_EDIT_PROVIDER => run_qwen_edit_provider(request),
        FLUX2_PROVIDER => run_flux2(request),
        // sc-22727: both klein catalog models (`flux2_klein_9b`, `flux2_klein_9b_kv`) load through
        // this ONE engine provider id; `flux2_arm` resolves which from `(provider, modelId)` and
        // refuses an unknown pair by name.
        FLUX2_KLEIN_PROVIDER => run_flux2(request),
        // sc-22734: the SenseNova-U1 family. Six catalog models over two registry ids; the arm
        // resolves which member from `(provider, modelId)` and refuses any other pair by name.
        // One arm per line: `stale-lane-report.mjs#adapterCapturableProviders` parses these arms
        // and accepts only a bare literal or a single `&str` const per arm.
        SENSENOVA_PROVIDER => run_sensenova(request),
        SENSENOVA_FAST_PROVIDER => run_sensenova(request),
        // sc-22726: the FLUX.1 family. Three registry ids on one arm — the two base text-to-image
        // providers of `mlx-gen-flux`, and `mlx-gen-pulid`'s identity route over the same
        // FLUX.1-dev backbone. The arm resolves which member from `(provider, mode)`.
        // One arm per line: `stale-lane-report.mjs#adapterCapturableProviders` derives the
        // capturable provider set by parsing these arms, and only accepts a bare literal or a
        // single `&str` const per arm — an or-pattern would make it throw.
        FLUX1_DEV_PROVIDER => run_flux_one(request),
        FLUX1_SCHNELL_PROVIDER => run_flux_one(request),
        PULID_FLUX_PROVIDER => run_flux_one(request),
        // sc-22733: the Mage-Flow family. SIX registered engine providers on one arm — the three
        // text-to-image checkpoints and the three instruction editors of `mlx-gen-mage`
        // (`model::MODEL_IDS`) — each with its own tiered rehost, all sharing one text-encoder/VAE
        // components snapshot. The arm resolves which member from `(provider, modelId)`.
        // One arm per line, bare literals: see the FLUX.1 note above — the derived capturable set
        // in `stale-lane-report.mjs#adapterCapturableProviders` parses these arms.
        "mage_flow" => run_mage_provider(request),
        "mage_flow_base" => run_mage_provider(request),
        "mage_flow_turbo" => run_mage_provider(request),
        "mage_flow_edit" => run_mage_provider(request),
        "mage_flow_edit_base" => run_mage_provider(request),
        "mage_flow_edit_turbo" => run_mage_provider(request),
        // sc-22731: the SANA and Chroma1 families. Five registry ids on one arm — the two
        // `mlx-gen-sana` routes and the three `mlx-gen-chroma` ones. One arm per line, for the
        // same reason the FLUX.1 block above is: `stale-lane-report.mjs#adapterCapturableProviders`
        // parses these arms and accepts only a bare literal or a single `&str` const per arm.
        SANA_PROVIDER => run_sana_chroma(request),
        SANA_SPRINT_PROVIDER => run_sana_chroma(request),
        CHROMA1_HD_PROVIDER => run_sana_chroma(request),
        CHROMA1_BASE_PROVIDER => run_sana_chroma(request),
        CHROMA1_FLASH_PROVIDER => run_sana_chroma(request),
        // sc-22732: the turnkey still family. Five registry ids on one arm, over three engine
        // crates — `mlx-gen-kolors`, `mlx-gen-ideogram` and `mlx-gen-lens`. The arm resolves which
        // member from `(provider, mode)`, and the member carries its own artifact family (Ideogram's
        // bf16 tier ships from a second repository).
        KOLORS_PROVIDER => run_turnkey_still(request),
        IDEOGRAM_PROVIDER => run_turnkey_still(request),
        IDEOGRAM_TURBO_PROVIDER => run_turnkey_still(request),
        LENS_PROVIDER => run_turnkey_still(request),
        LENS_TURBO_PROVIDER => run_turnkey_still(request),
        // sc-24112: Qwen-Image 2.1 rides the same arm — a plain reference-free text-to-image load
        // off one tier root, with a split-repo bf16 leg the member carries like Ideogram's.
        QWEN_IMAGE_2_1_PROVIDER => run_turnkey_still(request),
        // sc-25907: and the distilled Turbo id, its own member over its own two repositories.
        QWEN_IMAGE_2_1_TURBO_PROVIDER => run_turnkey_still(request),
        // sc-18808: the first VIDEO arm. Every arm above it refuses `geometry.frames != 1`; this one
        // validates against LTX's own resolution/temporal envelope instead.
        LTX_PROVIDER => run_ltx(request),
        // SC-18783: LTX-2.5 is a separate provider contract. Its variant and decoder axes are
        // validated inside the arm rather than being folded into the legacy 2.3 route.
        LTX25_PROVIDER => mlx_ltx25::run(request),
        // sc-18663: the second video arm, and the first joint audio+video one. Same rule as LTX —
        // it accepts a multi-frame geometry only by validating against MiniMax-H3's own lattice,
        // stride and canvas budget, read off the pinned engine crate.
        MINIMAX_PROVIDER => run_minimax_h3(request),
        // sc-22735: the third video arm, and the first AUTOREGRESSIVE one. Same rule as LTX and
        // MiniMax — it accepts a multi-frame geometry only by validating against Krea Realtime's own
        // spatial grid, per-edge range and temporal lattice, read off the pinned engine crate's
        // advertised descriptor. It deliberately applies NO area cap: the engine has none.
        KREA_REALTIME_PROVIDER => run_krea_realtime(request),
        // sc-22730: the SD3.5 family. Three distinct engine providers sharing one arm, each with
        // its own artifact family and published identity; `sd3_arm` resolves which from
        // `(provider, mode)` and refuses every other pair by name.
        //
        // THREE SEPARATE ARMS ON PURPOSE, like the FLUX.1 members above. The derived capturability
        // report parses this block arm by arm
        // (`stale-lane-report.mjs#adapterCapturableProviders`) and recognizes a string literal, a
        // single `&str` const, or the refusal arm's binding — never an or-pattern, which it refuses
        // to guess about rather than silently reading this lane as having no arm for the family.
        SD3_LARGE_PROVIDER => run_sd3(request),
        SD3_LARGE_TURBO_PROVIDER => run_sd3(request),
        SD3_MEDIUM_PROVIDER => run_sd3(request),
        // sc-22736: the Wan 2.2 family and SCAIL-2 — the third video block, and the first with a
        // CARRIER (I2V's Reference, SCAIL-2's character + driving clip). One arm per line, for the
        // same reason the FLUX.1 and SANA blocks above are: `stale-lane-report.mjs`'s
        // `adapterCapturableProviders` parses these arms and accepts only a bare literal or a
        // single `&str` const per arm.
        WAN_TI2V_5B_PROVIDER => mlx_wan_scail2::run(request),
        WAN_T2V_A14B_PROVIDER => mlx_wan_scail2::run(request),
        WAN_I2V_A14B_PROVIDER => mlx_wan_scail2::run(request),
        SCAIL2_PROVIDER => mlx_wan_scail2::run(request),
        // sc-22737: the third video arm, and the only one that also serves a STILL catalog entry —
        // `bernini` and `bernini_image` are two SceneWorks entries on ONE engine provider id, and
        // `bernini_target` resolves which from `(provider, modelId)` and refuses an unknown pair by
        // name. Validates against Bernini's own advertised canvases rather than refusing multi-frame
        // geometry, exactly as the two video arms above it do.
        BERNINI_PROVIDER => run_bernini(request),
        // sc-25690: the three Iris-3B task routes. Generation is a registry generator; depth and
        // restoration are NOT generators (a provider-specific estimator and a `Transform`), so the
        // arm drives each the way its own task is driven — one estimate / one tiled pass at the
        // plan geometry — after the same batch/frames validation every image arm keeps.
        IRIS_PROVIDER => mlx_iris::run(request),
        IRIS_DEPTH_PROVIDER => mlx_iris::run(request),
        IRIS_RESTORE_PROVIDER => mlx_iris::run(request),
        // sc-24217: Supra2-IMG's total pipeline (text encoder + DiT + VAE, resident) at the
        // published request, over a bundle converted in a separate adapter process beforehand.
        SUPRA2_PROVIDER => mlx_supra2::run(request),
        // SW-7100: MiniMax-H3-Image's staged text-to-image pipeline at the anchor still, loaded with
        // the worker's staged components (the upstream root, the tier's DiT, H3's text encoder).
        MINIMAX_H3_IMAGE_PROVIDER => mlx_minimax_h3_image::run(request),
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
        "canary" => run_ltx_canary(&request),
        "product_envelope_canary" => run_ltx_product_envelope_canary(&request),
        LTX_CAMPAIGN_ENTRY_ACTION => run_ltx_campaign_entry(&request),
        LTX_BOUNDED_CARRIER_ACTION => run_ltx_bounded_carrier_proof(&request),
        LTX_BOUNDED_CAMPAIGN_ACTION => run_ltx_bounded_campaign_entry(&request),
        "assess_batch" => assess_z_image_batch(&request),
        // Hidden zero-budget CPU self-test of THIS sealed binary's FLUX3 permit boundary; refused
        // unless its own environment guard is set and no production invocation exists.
        flux3_common::boundary_self_test::SELF_TEST_ACTION => {
            flux3_common::boundary_self_test::native_boundary_self_test(&request, "mlx")
        }
        // sc-24217: the child-process half of `protocol::ensure_supra2_bundle` — never a capture.
        protocol::SUPRA2_IMG_CONVERT_ACTION => protocol::supra2_convert(&request),
        other => Err(format!("unsupported action {other:?}")),
    }
    .unwrap_or_else(|error| protocol::fail(error));
    // The success path has no failure line to sit behind, so the deferred render notes go out here.
    protocol::flush_deferred_notes();
    protocol::write_response(&response).unwrap_or_else(|error| protocol::fail(error));
}

#[cfg(test)]
#[path = "mlx_z_image_reuse_tests.rs"]
mod z_image_reuse_tests;

/// sc-22726 — the FLUX.1 family arm: `flux_dev`, `flux_schnell` and `pulid_flux_dev`.
///
/// Everything here is env-free and weights-free, which is the point: the arm's plan binding (member
/// resolution, tier, root suffix, identity stack, request shape, admission context) is decided
/// before a single weight file is opened, so it can be proven on a CPU-only host. The measured
/// render itself belongs to the terminal capture campaign.
#[cfg(test)]
#[path = "mlx_flux_one_tests.rs"]
mod flux_one_tests;

/// sc-22730 — the SD3.5 family on the MLX lane.
///
/// Every claim here is settled from the plan, the arm table and a staged directory tree, BEFORE a
/// single weight file is opened, so the whole family is provable on a CPU-only host. The measured
/// render itself belongs to the terminal capture campaign.
#[cfg(test)]
#[path = "mlx_sd3_tests.rs"]
mod sd3_tests;

/// sc-22731 — the SANA and Chroma1 family arm.
///
/// Everything here is env-free and weights-free, which is the point: the arm's plan binding
/// (member resolution, tier, root suffix, request shape, LoadSpec shape) is decided before a single
/// weight file is opened, so it can be proven on a CPU-only host. The measured render itself
/// belongs to the terminal capture campaign.
#[cfg(test)]
#[path = "mlx_sana_chroma_tests.rs"]
mod sana_chroma_tests;

#[cfg(test)]
#[path = "mlx_flux2_tests.rs"]
mod flux2_tests;

#[cfg(test)]
#[path = "mlx_krea_base_tests.rs"]
mod krea_base_tests;

#[cfg(test)]
#[path = "mlx_krea_realtime_tests.rs"]
mod krea_realtime_tests;

/// The plan/adapter agreement sc-22735 exists to make undriftable: every planned MLX row of the two
/// widened models must name a calibration identity the production contract can actually emit, and
/// none of those identities may collide with Turbo's already-priced key.
#[cfg(test)]
#[path = "mlx_mlx_production_fingerprint_tests.rs"]
mod mlx_production_fingerprint_tests;

#[cfg(test)]
#[path = "mlx_sdxl_tests.rs"]
mod sdxl_tests;

#[cfg(test)]
#[path = "mlx_qwen_evidence_tests.rs"]
mod qwen_evidence_tests;

/// LTX video-arm regression suite. Historical SC-18810 artifacts stay immutable while this suite
/// follows the permanent provider contract used by SC-18946.
#[cfg(test)]
#[path = "mlx_ltx_tests.rs"]
mod ltx_tests;

#[cfg(test)]
#[path = "mlx_minimax_tests.rs"]
mod minimax_tests;

/// sc-22734 — the SenseNova-U1 family arm: six catalog models over the `sensenova_u1_8b` and
/// `sensenova_u1_8b_fast` registry ids.
///
/// Everything here is env-free and weights-free, which is the point: the arm's plan binding
/// (member resolution, tier, root suffix, load composition, request shape) is decided before a
/// single weight file is opened, so it can be proven on a CPU-only host. The measured render
/// belongs to the terminal capture campaign.
#[cfg(test)]
#[path = "mlx_bernini_tests.rs"]
mod bernini_tests;

#[cfg(test)]
#[path = "mlx_sensenova_tests.rs"]
mod sensenova_tests;

/// sc-22732 — the turnkey still arm: `kolors`, `ideogram_4`, `ideogram_4_turbo`, `lens` and
/// `lens_turbo`.
///
/// Everything here is env-free and weights-free, which is the point: the arm's plan binding (member
/// resolution, tier, artifact family, root suffix, request shape, fixture) is decided before a
/// single weight file is opened, so it can be proven on a CPU-only host. The measured render itself
/// belongs to the terminal capture campaign.
#[cfg(test)]
#[path = "mlx_turnkey_still_tests.rs"]
mod turnkey_still_tests;

/// sc-22738 — the rendered-vs-requested frame equality, swept across EVERY MLX video arm.
///
/// The campaign's `scail2_14b:bf16:mlx` anchor refused a completed 2h25m render because a bare
/// `measured.len() == geometry.frames` equality does not hold for a non-causal VAE decode. That
/// equality was copied into every video arm in this file, so the fix is only real if each remaining
/// copy is shown to be right for a stated reason rather than left unexamined. Three reasons appear
/// here, and this module pins the fact each one rests on so a pin bump reds in milliseconds instead
/// of after a multi-hour render:
///
///  - **Causal decode on a matching lattice** (LTX-2.3, LTX-2.5): the equality IS the shared rule.
///  - **The engine trims** (Krea Realtime): the equality is what production produces even though
///    the raw decode over-delivers, so the rule must NOT be applied.
///  - **No published VAE geometry** (MiniMax-H3): there is no rule to apply, and the engine asserts
///    the equality itself.
///
/// The non-causal, non-trimming families — Bernini here, SCAIL-2 and the Wan A14B experts in
/// `mlx_wan_scail2` — are the ones that had to change, and they carry their own tests.
#[cfg(test)]
#[path = "mlx_video_frame_equality_sweep.rs"]
mod video_frame_equality_sweep;

#[cfg(test)]
#[path = "mlx_exceeded_bound_tests.rs"]
mod exceeded_bound_tests;

/// sc-22738 (2026-09-08): a video capture is ONE measured render; the image lane keeps its two
/// warm passes. The lane is the PLAN's — derived from the row's frame count, exactly as
/// `measure-memory-catalog.mjs` derives the budget lane — never a per-model choice, and every
/// changed arm reads it through `protocol::capture_policy` before it renders.
#[cfg(test)]
#[path = "mlx_single_render_video_lane_tests.rs"]
mod single_render_video_lane_tests;

/// Source guards inspect the real moved owners; visibility does not change their body contract.
#[cfg(test)]
fn mlx_source_for_tests() -> &'static str {
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SOURCE
        .get_or_init(|| {
            concat!(
                include_str!("mlx.rs"),
                include_str!("mlx_ltx_family.rs"),
                include_str!("mlx_minimax_family.rs"),
                include_str!("mlx_runtime_family.rs"),
                include_str!("mlx_mage_family.rs"),
                include_str!("mlx_qwen_family.rs"),
                include_str!("mlx_krea_control_family.rs"),
                include_str!("mlx_flux_one_family.rs"),
                include_str!("mlx_turnkey_family.rs"),
                include_str!("mlx_sana_chroma_family.rs"),
                include_str!("mlx_flux2_family.rs"),
                include_str!("mlx_krea_base_family.rs"),
                include_str!("mlx_sdxl_family.rs"),
                include_str!("mlx_instantid_family.rs"),
                include_str!("mlx_qwen_edit_family.rs"),
                include_str!("mlx_bernini_family.rs"),
                include_str!("mlx_sensenova_family.rs"),
                include_str!("mlx_sd3_family.rs"),
                include_str!("mlx_krea_realtime_family.rs"),
                include_str!("mlx_tests.rs"),
                include_str!("mlx_mage_tests.rs"),
                include_str!("mlx_z_image_reuse_tests.rs"),
                include_str!("mlx_flux_one_tests.rs"),
                include_str!("mlx_sd3_tests.rs"),
                include_str!("mlx_sana_chroma_tests.rs"),
                include_str!("mlx_flux2_tests.rs"),
                include_str!("mlx_krea_base_tests.rs"),
                include_str!("mlx_krea_realtime_tests.rs"),
                include_str!("mlx_mlx_production_fingerprint_tests.rs"),
                include_str!("mlx_sdxl_tests.rs"),
                include_str!("mlx_qwen_evidence_tests.rs"),
                include_str!("mlx_ltx_tests.rs"),
                include_str!("mlx_minimax_tests.rs"),
                include_str!("mlx_bernini_tests.rs"),
                include_str!("mlx_sensenova_tests.rs"),
                include_str!("mlx_turnkey_still_tests.rs"),
                include_str!("mlx_video_frame_equality_sweep.rs"),
                include_str!("mlx_exceeded_bound_tests.rs"),
                include_str!("mlx_single_render_video_lane_tests.rs"),
            )
            .replace("pub(super) fn ", "fn ")
        })
        .as_str()
}
