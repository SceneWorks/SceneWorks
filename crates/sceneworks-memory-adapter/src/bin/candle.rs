#[cfg(test)]
mod adapter_test_support;

#[cfg(target_os = "macos")]
compile_error!("memory-candle-adapter is supported only on CUDA hosts");

#[path = "candle_sdxl_family.rs"]
mod candle_sdxl_family;
use candle_sdxl_family::*;

#[path = "candle_flux2_family.rs"]
mod candle_flux2_family;
use candle_flux2_family::*;

#[path = "candle_runtime_family.rs"]
mod candle_runtime_family;
use candle_runtime_family::*;

#[path = "candle_krea_control_family.rs"]
mod candle_krea_control_family;
use candle_krea_control_family::*;

#[path = "candle_mage_family.rs"]
mod candle_mage_family;
use candle_mage_family::*;

#[path = "candle_instantid_family.rs"]
mod candle_instantid_family;
use candle_instantid_family::*;

#[path = "candle_bernini_family.rs"]
mod candle_bernini_family;
use candle_bernini_family::*;

#[path = "candle_five_rung_family.rs"]
mod candle_five_rung_family;
use candle_five_rung_family::*;

#[path = "candle_sd3_family.rs"]
mod candle_sd3_family;
use candle_sd3_family::*;

#[path = "candle_flux_one_family.rs"]
mod candle_flux_one_family;
use candle_flux_one_family::*;

#[path = "candle_turnkey_family.rs"]
mod candle_turnkey_family;
use candle_turnkey_family::*;

#[path = "candle_sensenova_family.rs"]
mod candle_sensenova_family;
use candle_sensenova_family::*;

#[path = "candle_ltx_family.rs"]
mod candle_ltx_family;
use candle_ltx_family::*;

#[path = "candle_minimax_family.rs"]
mod candle_minimax_family;
use candle_minimax_family::*;

#[path = "candle_qwen_edit_family.rs"]
mod candle_qwen_edit_family;
use candle_qwen_edit_family::*;

#[path = "candle_dispatch_family.rs"]
mod candle_dispatch_family;
use candle_dispatch_family::*;

use candle_gen::testkit::{StableIdleConfig, VramProbe};
use runtime_cuda::gen_core::{
    adapter_stack_identity, AdapterKind, AdapterSpec, Capabilities, ComponentPrecisionFloor,
    Conditioning, GenerationMemory, GenerationOutput, GenerationRequest, Generator, Image,
    LoadShape, LoadSpec, MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryMode,
    MemoryNumericTier, MemoryOptimizationAuthority, MemoryPhase, MemoryRequestScope,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyParameters, ModelDescriptor, OffloadPolicy, Precision, Progress, Quant,
    TransformerComponent, WeightsSource,
};
use runtime_cuda::providers::pulid::PulidFluxRequest;
// sc-22728: the bespoke Candle edit provider the WORKER drives by name — it is not a registered
// generator, so the capture calls the same constructor the worker does rather than the catalog.
use runtime_cuda::providers::qwen_image::{QwenEdit, QwenEditPaths, QwenEditRequest};
use sceneworks_memory_adapter as protocol;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// sc-22736: the Wan 2.2 family and SCAIL-2. Four engine providers on one arm, kept in their own
// module for the same reason LTX-2.5's would be — a video family's carrier, rate menu and per-tier
// identity table are a self-contained apparatus.
#[path = "candle_wan_scail2.rs"]
mod candle_wan_scail2;

#[path = "candle_flux3.rs"]
mod candle_flux3;
#[path = "flux3_common.rs"]
mod flux3_common;
const FLUX3_ID: &str = "flux3_klein";
const FLUX3_BASE_ID: &str = "flux3_klein_base";

#[path = "candle_iris.rs"]
mod candle_iris;

/// The three Iris-3B task routes (sc-25690), served by `candle_iris`. Registry ids of
/// `candle-gen-iris`, which are also the catalog model ids. Spelled here, as consts, for the same
/// reason the Wan/SCAIL-2 ids below are: `adapterCapturableProviders` reads the pre-gate off this
/// file.
const IRIS_ID: &str = "iris_3b";
const IRIS_DEPTH_ID: &str = "iris_3b_depth";
const IRIS_RESTORE_ID: &str = "iris_3b_restore";

#[path = "candle_supra2.rs"]
mod candle_supra2;

/// Supra2-IMG (sc-24217, epic 24212), served by `candle_supra2`: `candle-gen-supra2`'s registry id
/// (`gen_core_checkpoint::supra2::PROVIDER_ID`), which is also the catalog model id. Spelled here, as a const,
/// for the reason the Iris ids above are: `adapterCapturableProviders` reads the pre-gate off this
/// file.
const SUPRA2_ID: &str = "supra2_img";

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
/// The UNDISTILLED Krea 2 base (sc-22735): the same engine crate as Turbo (`candle-gen-krea`,
/// `KREA_2_RAW_ID`) on a DIFFERENT execution seam — Turbo rides the phase-curve contract, Raw the
/// request-scoped staged-residency one (`build_krea_request_scoped_memory_strategy_contract`) —
/// and off its own tiered rehost. Its own env family, so a Raw plan can never be satisfied by
/// Turbo weights.
const KREA_RAW_ID: &str = "krea_2_raw";
const KREA_RAW_PLAIN_EXECUTION_PATH: &str = "the Candle Krea 2 Raw base-only text-to-image path";
/// The label the Raw arm refuses a non-still geometry under; see [`still_calibration_label`].
const KREA_RAW_STILL_CALIBRATION: &str = "Candle Krea 2 Raw calibration";
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

// ----------------------------------------------------------------------------------------------
// The SDXL family (sc-22729)
// ----------------------------------------------------------------------------------------------

/// The single engine id `candle-gen-sdxl` registers (`lib.rs` `MODEL_ID`). Five catalog models ride
/// it; the member is named by `LoadSpec::resolved_route`, which the worker binds to `request.model`
/// (`image_jobs/base.rs:8785`) and which `candle-gen-sdxl`'s `route()` REQUIRES — a spec with no
/// resolved route is refused ("optimized memory admission requires an exact resolved catalog route").
const SDXL_ID: &str = "sdxl";

/// The three caller-staged SDXL components. `candle-gen-sdxl` `validate_shared_component_revisions`
/// requires all three at EXACT pinned upstream revisions, so a capture stages the same corequisite
/// snapshots the worker's `attach_required_components` stages. Each is bound by its own env var.
const SDXL_COMPONENTS: [(&str, &str); 3] = [
    (
        "tokenizer_clip_l",
        "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_L",
    ),
    (
        "tokenizer_clip_bigg",
        "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_BIGG",
    ),
    ("vae_fp16_fix", "SCENEWORKS_SDXL_COMPONENT_VAE_FP16_FIX"),
];

/// One member of the Candle SDXL family.
///
/// The revision this member's route pins is NOT a field here. It used to be — transcribed from
/// `candle-gen-sdxl`'s own `SDXL_ROUTES` table — and sc-22738 is what a transcription costs: the
/// inference-side sc-22729 repair moved `illustrious_xl_v1` to `778c3f02…` and `illustrious_xl_v2`
/// to `672e9851…`, the copies here stayed at `c5a92a90…` / `7c5c8b2b…`, and on CUDA run
/// 34272596969 all six Illustrious cells failed with this adapter's own drift refusal — against a
/// pin at which the engine and the manifest AGREED. The runner's plan-time check
/// (`sdxlCandleRouteDrift` in `scripts/measure-memory-catalog.mjs`) derives both sides and had
/// correctly classified every one of them `runnable`.
///
/// So the revision is read from the linked engine crate at the pin, by
/// [`sdxl_candle_route_revision`]. The check itself stays: the engine's `path_has_snapshot` matches
/// a staged root against that literal, so a member whose route revision has drifted from the
/// shipped manifest cannot seal a contract and cannot load at all — and saying which two revisions
/// disagree is more useful than the engine's own message. It simply can no longer be wrong about
/// what the engine pins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SdxlCandleArm {
    model_id: &'static str,
    execution_path: &'static str,
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
}

const SDXL_CANDLE_BASE_ARM: SdxlCandleArm = SdxlCandleArm {
    model_id: "sdxl",
    execution_path: "the Candle SDXL base-only text-to-image path",
    still_calibration: "Candle SDXL base calibration",
    repository_env: "SCENEWORKS_SDXL_REPOSITORY",
    revision_env: "SCENEWORKS_SDXL_REVISION",
    root_env: "SCENEWORKS_SDXL_ROOT",
    expected_repository: protocol::SDXL_REPOSITORY,
};

const SDXL_CANDLE_REALVISXL_ARM: SdxlCandleArm = SdxlCandleArm {
    model_id: "realvisxl",
    execution_path: "the Candle RealVisXL base-only text-to-image path",
    still_calibration: "Candle RealVisXL base calibration",
    repository_env: "SCENEWORKS_REALVISXL_REPOSITORY",
    revision_env: "SCENEWORKS_REALVISXL_REVISION",
    root_env: "SCENEWORKS_REALVISXL_ROOT",
    expected_repository: protocol::REALVISXL_REPOSITORY,
};

const SDXL_CANDLE_LIGHTNING_ARM: SdxlCandleArm = SdxlCandleArm {
    model_id: "realvisxl_lightning",
    execution_path: "the Candle RealVisXL Lightning base-only text-to-image path",
    still_calibration: "Candle RealVisXL Lightning base calibration",
    repository_env: "SCENEWORKS_REALVISXL_LIGHTNING_REPOSITORY",
    revision_env: "SCENEWORKS_REALVISXL_LIGHTNING_REVISION",
    root_env: "SCENEWORKS_REALVISXL_LIGHTNING_ROOT",
    expected_repository: protocol::REALVISXL_LIGHTNING_REPOSITORY,
};

const SDXL_CANDLE_ILLUSTRIOUS_V1_ARM: SdxlCandleArm = SdxlCandleArm {
    model_id: "illustrious_xl_v1",
    execution_path: "the Candle Illustrious-XL v1 base-only text-to-image path",
    still_calibration: "Candle Illustrious-XL v1 base calibration",
    repository_env: "SCENEWORKS_ILLUSTRIOUS_XL_V1_REPOSITORY",
    revision_env: "SCENEWORKS_ILLUSTRIOUS_XL_V1_REVISION",
    root_env: "SCENEWORKS_ILLUSTRIOUS_XL_V1_ROOT",
    expected_repository: protocol::ILLUSTRIOUS_XL_V1_REPOSITORY,
};

const SDXL_CANDLE_ILLUSTRIOUS_V2_ARM: SdxlCandleArm = SdxlCandleArm {
    model_id: "illustrious_xl_v2",
    execution_path: "the Candle Illustrious-XL v2 base-only text-to-image path",
    still_calibration: "Candle Illustrious-XL v2 base calibration",
    repository_env: "SCENEWORKS_ILLUSTRIOUS_XL_V2_REPOSITORY",
    revision_env: "SCENEWORKS_ILLUSTRIOUS_XL_V2_REVISION",
    root_env: "SCENEWORKS_ILLUSTRIOUS_XL_V2_ROOT",
    expected_repository: protocol::ILLUSTRIOUS_XL_V2_REPOSITORY,
};

const SDXL_CANDLE_FAMILY: [SdxlCandleArm; 5] = [
    SDXL_CANDLE_BASE_ARM,
    SDXL_CANDLE_REALVISXL_ARM,
    SDXL_CANDLE_LIGHTNING_ARM,
    SDXL_CANDLE_ILLUSTRIOUS_V1_ARM,
    SDXL_CANDLE_ILLUSTRIOUS_V2_ARM,
];

// ----------------------------------------------------------------------------------------------
// InstantID (sc-22729)
// ----------------------------------------------------------------------------------------------

/// The bespoke identity provider. `candle-gen-catalog` registers no descriptor for it, so this
/// adapter loads through the crate's own `InstantId::load_with_memory_context` — the exact call
/// `image_jobs/instantid.rs` makes — instead of `catalog.media().load`.
const INSTANTID_ID: &str = "instantid";
const INSTANTID_MODEL_ID: &str = "instantid_realvisxl";
/// `candle-gen-instantid` mints its OWN calibration string, deliberately distinct from the
/// request-contract revision it shares with the MLX twin (`memory_strategy.rs:22-26` at the pin).
const INSTANTID_CALIBRATION_FINGERPRINT: &str = "instantid-candle-staged-conditioning-v1";
const INSTANTID_EXECUTION_PATH: &str = "the Candle InstantID identity-conditioned character path";
const INSTANTID_STILL_CALIBRATION: &str = "Candle InstantID identity calibration";
/// The candle InstantID stack is DENSE-ONLY: the worker hard-pins its tier subdir to `bf16/`
/// (`image_jobs/instantid.rs` `instantid_memory_backend_keys` / `instantid_tier_subdir` on the
/// non-macOS branch) and the provider declares `resolved_numeric_tier()` as Bf16 with no quant. A
/// q4/q8 candle plan would measure bf16 weights and file the peaks under a packed tier.
const INSTANTID_CANDLE_TIER: &str = "bf16";
const INSTANTID_IP_SCALE: f32 = 0.8;
const INSTANTID_CONTROLNET_SCALE: f32 = 0.8;
const INSTANTID_SEED: u64 = 18381;
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
// sc-22732: the turnkey still family. Five catalog models over three engine crates, every one a
// plain reference-free text-to-image route on the shared five-rung reference path. The engine id
// EQUALS the catalog model id for all five (`crates/sceneworks-worker/src/engines.rs` MODEL_TABLE
// and `sceneworks-core`'s candle routing catalog), which is why no anchor-loader closure row needs
// an `engineId` alias. Kolors' IP-Adapter and strict-pose ControlNet routes are separate candle
// providers (`candle_kolors_ipadapter`, `candle_kolors_control`) and are deliberately NOT served
// here: they are different measurements with different overlays, and no anchor plans them.
const KOLORS_ID: &str = "kolors";
const KOLORS_PLAIN_EXECUTION_PATH: &str = "the Candle Kolors base-only text-to-image path";
const KOLORS_STILL_CALIBRATION: &str = "Candle Kolors calibration";
const IDEOGRAM_ID: &str = "ideogram_4";
const IDEOGRAM_PLAIN_EXECUTION_PATH: &str = "the Candle Ideogram 4 base-only text-to-image path";
const IDEOGRAM_STILL_CALIBRATION: &str = "Candle Ideogram 4 calibration";
const IDEOGRAM_TURBO_ID: &str = "ideogram_4_turbo";
const IDEOGRAM_TURBO_PLAIN_EXECUTION_PATH: &str =
    "the Candle Ideogram 4 Turbo base-only text-to-image path";
const IDEOGRAM_TURBO_STILL_CALIBRATION: &str = "Candle Ideogram 4 Turbo calibration";
const LENS_ID: &str = "lens";
const LENS_PLAIN_EXECUTION_PATH: &str = "the Candle Lens base-only text-to-image path";
const LENS_STILL_CALIBRATION: &str = "Candle Lens calibration";
const LENS_TURBO_ID: &str = "lens_turbo";
const LENS_TURBO_PLAIN_EXECUTION_PATH: &str = "the Candle Lens-Turbo base-only text-to-image path";
const LENS_TURBO_STILL_CALIBRATION: &str = "Candle Lens-Turbo calibration";
/// Qwen-Image 2.1 (sc-24112) on the turnkey still family's five-rung path. Registry id of
/// `candle-gen-qwen-image-2-1` (`MODEL_ID`), which is also the catalog model id. Its bf16 tier is the
/// upstream `Qwen/Qwen-Image-2.1` snapshot ROOT and its packed tiers the `q8/`/`q4/` subdirs of the
/// SceneWorks re-host — see [`qwen_image_2_1_five_rung_family`].
const QWEN_IMAGE_2_1_ID: &str = "qwen_image_2_1";
const QWEN_IMAGE_2_1_PLAIN_EXECUTION_PATH: &str =
    "the Candle Qwen-Image 2.1 base-only text-to-image path";
const QWEN_IMAGE_2_1_STILL_CALIBRATION: &str = "Candle Qwen-Image 2.1 calibration";
/// Qwen-Image 2.1 Turbo (sc-25907, epic 25899): the distilled sibling's OWN registry id, served by
/// the same `candle-gen-qwen-image-2-1` crate (`TURBO_MODEL_ID`) and equal to its catalog model id.
/// Same five-rung shape as base over its OWN two repositories — see
/// [`qwen_image_2_1_turbo_five_rung_family`].
const QWEN_IMAGE_2_1_TURBO_ID: &str = "qwen_image_2_1_turbo";
const QWEN_IMAGE_2_1_TURBO_PLAIN_EXECUTION_PATH: &str =
    "the Candle Qwen-Image 2.1 Turbo base-only text-to-image path";
const QWEN_IMAGE_2_1_TURBO_STILL_CALIBRATION: &str = "Candle Qwen-Image 2.1 Turbo calibration";
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

    // Endpoint admission has a wall-clock ceiling; a stalled telemetry child must not bypass it.
    fn query_bounded(&self, fields: &str, timeout: Duration) -> Result<String, String> {
        let started = Instant::now();
        let mut child = Command::new(&self.executable)
            .arg(format!("--id={}", self.physical_id))
            .arg(format!("--query-gpu={fields}"))
            .arg("--format=csv,noheader,nounits")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("start {}: {error}", self.executable.display()))?;
        loop {
            let status = match child.try_wait() {
                Ok(status) => status,
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("poll telemetry child: {error}"));
                }
            };
            if status.is_some() {
                let output = child
                    .wait_with_output()
                    .map_err(|error| error.to_string())?;
                return if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
                } else {
                    Err(format!(
                        "nvidia-smi query failed: {}",
                        String::from_utf8_lossy(&output.stderr).trim()
                    ))
                };
            }
            if started.elapsed() >= timeout {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("selected GPU telemetry query exceeded {timeout:?}"));
            }
            std::thread::sleep(
                Duration::from_millis(10).min(timeout.saturating_sub(started.elapsed())),
            );
        }
    }
}

/// Everything one Candle InstantID capture loads. The backbone is the PLAIN `realvisxl` rehost at
/// its `bf16/` tier — the candle stack is dense-only — plus the identity bundle and IdentityNet the
/// worker stages through the same env seams, plus the three SDXL components
/// `SdxlComponents::from_spec` requires.
struct InstantIdCandleBinding {
    repository: String,
    revision: String,
    /// The candle `InstantIdPaths` carries the whole priced composition — including `face_dir`,
    /// which is the identity bundle's own root — so the resolved bundle needs no separate field.
    paths: runtime_cuda::providers::instantid::InstantIdPaths,
    artifact_fingerprint: String,
}

/// The half of a Candle InstantID binding that is resolved purely from the ENVIRONMENT: the tier
/// guard, the backbone artifact identity, the staged identity bundle, the IdentityNet directory and
/// the ordered artifact digest over all five.
///
/// Split out of [`instantid_candle_binding`] (sc-22729 review) because the remainder —
/// `SdxlComponents::from_spec` — reads real component weights, so nothing past it is drivable on a
/// synthetic tree. Every env seam this arm honours therefore has a test that actually executes it
/// (`instantid_candle_identity_names_the_missing_identitynet_weight`), instead of a tier guard whose
/// refusal is the only thing anything ever reaches.
struct InstantIdCandleIdentity {
    repository: String,
    revision: String,
    root: PathBuf,
    identitynet: PathBuf,
    bundle: protocol::InstantIdIdentityBundle,
    artifact_fingerprint: String,
}

/// `candle-gen-chroma`'s `CANDLE_LADDER_REVISION`: the shape every Chroma1 Candle identity names.
/// Bumped upstream when the request-scoped Resident/Staged surface changes, which is what makes
/// every Chroma anchor recaptured rather than silently re-bound.
const CHROMA1_CANDLE_LADDER_REVISION: &str = "request-scoped-staged-residency-v1";

/// One member of the turnkey still family on this lane (sc-22732): the per-member decision the
/// shared five-rung path cannot make for itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TurnkeyCandleMember {
    provider_id: &'static str,
    /// Whether the PLANNED packed tier reaches the loader as `LoadSpec::quantize`. `true` mirrors
    /// the worker's `candle_quant_for_resolved_tier` for a member it does not carve out; `false`
    /// is the two Ideogram routes, whose exact directory route refuses `quantize: Some(_)` and
    /// proves the tier off the packed safetensors headers instead.
    tier_quant_reaches_the_loader: bool,
}

const TURNKEY_CANDLE_MEMBERS: [TurnkeyCandleMember; 7] = [
    TurnkeyCandleMember {
        provider_id: KOLORS_ID,
        tier_quant_reaches_the_loader: true,
    },
    TurnkeyCandleMember {
        provider_id: IDEOGRAM_ID,
        tier_quant_reaches_the_loader: false,
    },
    TurnkeyCandleMember {
        provider_id: IDEOGRAM_TURBO_ID,
        tier_quant_reaches_the_loader: false,
    },
    TurnkeyCandleMember {
        provider_id: LENS_ID,
        tier_quant_reaches_the_loader: true,
    },
    TurnkeyCandleMember {
        provider_id: LENS_TURBO_ID,
        tier_quant_reaches_the_loader: true,
    },
    // sc-24112: `candle-gen-qwen-image-2-1`'s `quant::resolve_requested_tier` treats
    // `LoadSpec::quantize` as a TIER SELECTOR and refuses a packed tier whose quant does not match
    // it (including `None`), and the worker forwards the resolved tier's quant on this lane.
    TurnkeyCandleMember {
        provider_id: QWEN_IMAGE_2_1_ID,
        tier_quant_reaches_the_loader: true,
    },
    // sc-25907: the Turbo id is the same crate's `resolve_requested_tier`, so the same rule.
    TurnkeyCandleMember {
        provider_id: QWEN_IMAGE_2_1_TURBO_ID,
        tier_quant_reaches_the_loader: true,
    },
];

/// `candle-gen-qwen-image-2-1`'s `memory_strategy::MEMORY_CALIBRATION_FINGERPRINT` at the pin, as a
/// literal for the same reason the table above is one (the binary is `compile_error!` on macOS). The
/// capture still refuses a loaded contract whose identity differs, and
/// `the_qwen_image_2_1_candle_identity_is_the_engine_constant` binds it on a CUDA host.
const QWEN_IMAGE_2_1_CANDLE_FINGERPRINT: &str = "qwen-image-2-1-candle-derived-2026-09-22-v1";

/// `candle-gen-qwen-image-2-1`'s `memory_strategy::TURBO_MEMORY_CALIBRATION_FINGERPRINT` at the pin,
/// a literal for the same reason; `the_qwen_image_2_1_candle_identity_is_the_engine_constant` binds
/// it to the engine constant on a CUDA host.
const QWEN_IMAGE_2_1_TURBO_CANDLE_FINGERPRINT: &str =
    "qwen-image-2-1-turbo-candle-derived-2026-10-13-v1";

/// The request scope one capture renders under, opened exactly the way the worker opens one.
///
/// `scope` is `None` when the provider opened none: `Generator::begin_memory_strategy_request`
/// returns `Ok(None)` for a contract that implements no optimized rung (`candle-gen-minimax-h3`
/// declares `Resident` alone and overrides neither seam), and the worker's `generate_with_scope`
/// renders such a request without a scope. The capture then measures its phases off the progress
/// boundaries alone: the lifecycle calls are no-ops, while a selected parameter that has no scope
/// to receive it is refused rather than dropped (sc-22738: "minimax_h3 selection did not create a
/// provider scope" was this arm demanding a scope the production path never asks for).
struct CaptureScope<'a> {
    label: String,
    scope: Option<Box<dyn MemoryRequestScope + 'a>>,
}

impl CaptureScope<'_> {
    fn unsupported(&self, what: &str) -> runtime_cuda::gen_core::Error {
        runtime_cuda::gen_core::Error::Unsupported(format!(
            "{}: the planned selection carries {what}, but the provider opened no request scope to \
             receive it",
            self.label
        ))
    }

    fn configure_request(
        &mut self,
        request: &mut GenerationRequest,
    ) -> runtime_cuda::gen_core::Result<()> {
        match self.scope.as_mut() {
            Some(scope) => scope.configure_request(request),
            None => Ok(()),
        }
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> runtime_cuda::gen_core::Result<()> {
        match self.scope.as_mut() {
            Some(scope) => scope.enter_phase(phase),
            None => Ok(()),
        }
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> runtime_cuda::gen_core::Result<()> {
        match self.scope.as_mut() {
            Some(scope) => scope.leave_phase(phase),
            None => Ok(()),
        }
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> runtime_cuda::gen_core::Result<()> {
        match self.scope.as_mut() {
            Some(scope) => scope.configure_decode(tile_edge, overlap, geometry),
            None => Err(self.unsupported("a decode tile")),
        }
    }

    fn configure_attention(&mut self, chunk_size: u32) -> runtime_cuda::gen_core::Result<()> {
        match self.scope.as_mut() {
            Some(scope) => scope.configure_attention(chunk_size),
            None => Err(self.unsupported("an attention chunk")),
        }
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> runtime_cuda::gen_core::Result<()> {
        match self.scope.as_mut() {
            Some(scope) => scope.materialize_transformer_window(first_block, block_count),
            None => Err(self.unsupported("a transformer window")),
        }
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> runtime_cuda::gen_core::Result<()> {
        match self.scope.as_mut() {
            Some(scope) => scope.finish(outcome),
            None => Ok(()),
        }
    }
}

type LoadedFiveRungGenerator = (
    &'static str,
    &'static str,
    String,
    String,
    Box<dyn runtime_cuda::gen_core::Generator>,
    VramProbe,
);

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

    /// The backbone snapshot plus the content digest of the ordered identity-file inventory.
    fn artifact_json(&self) -> Value {
        let mut artifact = artifact(&self.repository, &self.revision, self.tier);
        artifact["inventorySha256"] = json!(self.bundle.composite_sha256);
        artifact
    }
}

// ---------------------------------------------------------------------------------------------
// The SenseNova-U1 family arm (sc-22734): six catalog models over two registry ids.
// ---------------------------------------------------------------------------------------------

/// One member of the Candle SenseNova-U1 family.
///
/// The worker routes SIX catalog models onto TWO registry ids on this lane too — the quality trio
/// on `sensenova_u1_8b`, the 8-step distilled trio on `sensenova_u1_8b_fast`
/// (`crates/sceneworks-worker/src/engines.rs` `MODEL_TABLE`; the Candle lane has no bespoke
/// SenseNova stream, everything rides the generic image path in `image_jobs/base.rs`). Each of the
/// six has its OWN independently pinned tiered rehost, so the arm is keyed on
/// `(provider, modelId)`: `candle-gen-sensenova`'s `validate_resolved_artifact_binding` refuses a
/// `LoadSpec` whose `resolved_route` names a repository the weights path does not carry, and its
/// `validate_load_spec` refuses a route that belongs to the OTHER provider by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SenseNovaCandleArm {
    /// The catalog model id — `planned.target.modelId`, and the spec's `resolved_route` (the
    /// worker binds `request.model` on every Candle image load).
    model_id: &'static str,
    /// The registry id handed to `catalog.media().load` — the production loader (E4).
    provider: &'static str,
    /// The route slug the engine's per-route identity is spelled with, and the record's
    /// diagnostics source.
    slug: &'static str,
    execution_path: &'static str,
    still_calibration: &'static str,
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    expected_repository: &'static str,
}

/// The quality registry id (`candle_gen_sensenova::MODEL_ID`).
const SENSENOVA_ID: &str = "sensenova_u1_8b";
/// The distilled registry id (`candle_gen_sensenova::MODEL_ID_FAST`). The distill LoRA is
/// pre-merged into each `_fast` rehost at convert time, so the worker attaches nothing here either
/// (`model.supports_adapters()` is false for these ids).
const SENSENOVA_FAST_ID: &str = "sensenova_u1_8b_fast";
/// The family seed, shared with the MLX arm so one plan row's fixture reads the same on both lanes
/// apart from the lane token.
const SENSENOVA_SEED: u64 = 22734;

const SENSENOVA_CANDLE_FAMILY: [SenseNovaCandleArm; 6] = [
    SenseNovaCandleArm {
        model_id: "sensenova_u1_8b",
        provider: SENSENOVA_ID,
        slug: "quality",
        execution_path: "the Candle SenseNova-U1 8B base text-to-image path",
        still_calibration: "Candle SenseNova-U1 8B calibration",
        repository_env: "SCENEWORKS_SENSENOVA_U1_8B_REPOSITORY",
        revision_env: "SCENEWORKS_SENSENOVA_U1_8B_REVISION",
        root_env: "SCENEWORKS_SENSENOVA_U1_8B_ROOT",
        expected_repository: protocol::SENSENOVA_U1_8B_REPOSITORY,
    },
    SenseNovaCandleArm {
        model_id: "sensenova_u1_8b_infographic_v2",
        provider: SENSENOVA_ID,
        slug: "infographic-v2",
        execution_path: "the Candle SenseNova-U1 Infographic V2 text-to-image path",
        still_calibration: "Candle SenseNova-U1 Infographic V2 calibration",
        repository_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_REPOSITORY",
        revision_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_REVISION",
        root_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_ROOT",
        expected_repository: protocol::SENSENOVA_U1_8B_INFOGRAPHIC_V2_REPOSITORY,
    },
    SenseNovaCandleArm {
        model_id: "sensenova_u1_8b_infographic_v3",
        provider: SENSENOVA_ID,
        slug: "infographic-v3",
        execution_path: "the Candle SenseNova-U1 Infographic V3 text-to-image path",
        still_calibration: "Candle SenseNova-U1 Infographic V3 calibration",
        repository_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_REPOSITORY",
        revision_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_REVISION",
        root_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_ROOT",
        expected_repository: protocol::SENSENOVA_U1_8B_INFOGRAPHIC_V3_REPOSITORY,
    },
    SenseNovaCandleArm {
        model_id: "sensenova_u1_8b_fast",
        provider: SENSENOVA_FAST_ID,
        slug: "fast",
        execution_path: "the Candle SenseNova-U1 8B distilled text-to-image path",
        still_calibration: "Candle SenseNova-U1 8B distilled calibration",
        repository_env: "SCENEWORKS_SENSENOVA_U1_8B_FAST_REPOSITORY",
        revision_env: "SCENEWORKS_SENSENOVA_U1_8B_FAST_REVISION",
        root_env: "SCENEWORKS_SENSENOVA_U1_8B_FAST_ROOT",
        expected_repository: protocol::SENSENOVA_U1_8B_FAST_REPOSITORY,
    },
    SenseNovaCandleArm {
        model_id: "sensenova_u1_8b_infographic_v2_fast",
        provider: SENSENOVA_FAST_ID,
        slug: "infographic-v2-fast",
        execution_path: "the Candle SenseNova-U1 Infographic V2 distilled text-to-image path",
        still_calibration: "Candle SenseNova-U1 Infographic V2 distilled calibration",
        repository_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST_REPOSITORY",
        revision_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST_REVISION",
        root_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST_ROOT",
        expected_repository: protocol::SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST_REPOSITORY,
    },
    SenseNovaCandleArm {
        model_id: "sensenova_u1_8b_infographic_v3_fast",
        provider: SENSENOVA_FAST_ID,
        slug: "infographic-v3-fast",
        execution_path: "the Candle SenseNova-U1 Infographic V3 distilled text-to-image path",
        still_calibration: "Candle SenseNova-U1 Infographic V3 distilled calibration",
        repository_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST_REPOSITORY",
        revision_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST_REVISION",
        root_env: "SCENEWORKS_SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST_ROOT",
        expected_repository: protocol::SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST_REPOSITORY,
    },
];

struct Ltx25LoadPlan {
    repository: String,
    revision: String,
    inventory_sha256: String,
    load_shape: LoadShape,
    adapters: Vec<AdapterSpec>,
    spec: LoadSpec,
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
    "the Candle Bernini dual-expert reference-to-video path";
const BERNINI_CANDLE_IMAGE_EXECUTION_PATH: &str = "the Candle Bernini still text-to-image path";
const BERNINI_CANDLE_STILL_CALIBRATION: &str = "Candle Bernini still calibration";
/// The single cadence the shipped `bernini` manifest entry publishes (`limits.fps: [16]`).
const BERNINI_CANDLE_FPS: u32 = 16;
/// The shortest frame count admitted by Bernini's reference-video memory contract.
const BERNINI_CANDLE_FRAMES: u32 = 45;
/// One seed for every sc-22737 Candle fixture. The fixture binds the family, member, tier and full
/// geometry, so the seed does not also have to carry the route.
const SC22737_CANDLE_SEED: u64 = 22737;

/// The LTX-2.3 Candle engine id. DIFFERENT from the MLX one (`ltx_2_3`), the same way LTX-2.5's two
/// lanes differ: `candle-gen-ltx` registers the 2.3 route as `ltx_2_3_distilled`, which is what
/// `video_jobs/candle.rs#candle_video_engine_id` resolves and therefore what a candle plan row's
/// `provider` must name.
const LTX23_CANDLE_ID: &str = "ltx_2_3_distilled";
const LTX23_CANDLE_MODEL_ID: &str = "ltx_2_3";
const LTX23_CANDLE_EXECUTION_PATH: &str = "the Candle LTX-2.3 image-to-video path";
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
    /// The load shape `load_plan` stages, which is the ONLY shape the engine executes for this
    /// family (sc-22738): `candle-gen-ltx` and `candle-gen-bernini` refuse a non-eager spec on
    /// their production contract path, and `candle-gen-minimax-h3` pins the loaded contract to
    /// eager whatever the spec asks. The plan rows this arm serves must declare the same shape —
    /// `sc22737_plan_rows_declare_the_load_shape_their_arm_stages` holds them to it weights-free,
    /// because `run_sc22737_video_capture`'s plan/provider re-assert only fires on a CUDA host.
    #[cfg(test)]
    load_shape: LoadShape,
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

/// `candle-gen-bernini`'s production contract: "Bernini Candle memory contract requires
/// EagerMaterialization" (sc-22738).
const BERNINI_CANDLE_LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;
/// `candle-gen-ltx`'s production contract path (`memory_strategy_contract` -> `tier_paths`):
/// "calibrated q4 memory admission requires eager bf16 component loading" (sc-22738).
const LTX23_CANDLE_LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;
/// `candle-gen-minimax-h3` pins `LOAD_SHAPE = EagerMaterialization` on the loader "whatever a
/// caller asks for", so the loaded contract's calibration identity is eager (sc-22738).
const MINIMAX_CANDLE_LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;

const BERNINI_CANDLE_VIDEO_ARM: Sc22737VideoArm = Sc22737VideoArm {
    engine_id: BERNINI_CANDLE_ID,
    model_id: BERNINI_CANDLE_VIDEO_MODEL_ID,
    execution_path: BERNINI_CANDLE_VIDEO_EXECUTION_PATH,
    fixture_prefix: "bernini-r2v-candle",
    mode: "reference_to_video",
    reference_count: 1,
    legal_fps: &[BERNINI_CANDLE_FPS],
    frames: BERNINI_CANDLE_FRAMES,
    requires_audio: false,
    prompt: "a slow crane over a terracotta rooftop at golden hour, swallows turning, cinematic",
    diagnostics_lane: "memory-candle-adapter:bernini-dual-expert",
    validate_geometry: validate_bernini_candle_geometry,
    load_plan: bernini_candle_load_plan,
    #[cfg(test)]
    load_shape: BERNINI_CANDLE_LOAD_SHAPE,
};

const LTX23_CANDLE_ARM: Sc22737VideoArm = Sc22737VideoArm {
    engine_id: LTX23_CANDLE_ID,
    model_id: LTX23_CANDLE_MODEL_ID,
    execution_path: LTX23_CANDLE_EXECUTION_PATH,
    fixture_prefix: "ltx-2-3-i2v-candle",
    mode: "image_to_video",
    reference_count: 1,
    legal_fps: &LTX23_CANDLE_FPS,
    frames: LTX23_CANDLE_FRAMES,
    requires_audio: false,
    prompt: "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic",
    diagnostics_lane: "memory-candle-adapter:ltx-2.3",
    validate_geometry: validate_ltx23_candle_geometry,
    load_plan: ltx23_candle_load_plan,
    #[cfg(test)]
    load_shape: LTX23_CANDLE_LOAD_SHAPE,
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
    #[cfg(test)]
    load_shape: MINIMAX_CANDLE_LOAD_SHAPE,
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
    #[cfg(test)]
    load_shape: MINIMAX_CANDLE_LOAD_SHAPE,
};

/// Every cell this block serves, in one place so the tables and the tests cannot disagree.
const SC22737_VIDEO_ARMS: [Sc22737VideoArm; 4] = [
    BERNINI_CANDLE_VIDEO_ARM,
    LTX23_CANDLE_ARM,
    MINIMAX_CANDLE_BASE_ARM,
    MINIMAX_CANDLE_REFERENCE_ARM,
];

/// The fixture prefix that marks a plan row as a five-rung reference capture.
const FIVE_RUNG_FIXTURE_PREFIX: &str = "fresh-five-rung-";

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
/// (`resolve_qwen_edit_guidance`, manifest `variationStrength.default`). Lightning requires 1.0
/// explicitly: the engine rejects other guidance values rather than overriding them.
const QWEN_EDIT_GUIDANCE: f32 = 4.0;

use protocol::QwenEditArtifactSource;

use protocol::QwenEditArtifact;

// A persistent idle CUDA context is part of the sampled baseline, not evidence of activity.
// This manual profile verifies repeated physical-device activity and stable residency before load.

// Issue 3197: bounded manual endpoint capture; the catalog keeps its one-anchor contract.

fn main() {
    let request = protocol::request_from_stdin().unwrap_or_else(|error| protocol::fail(error));
    let response = match protocol::action(&request).unwrap_or_else(|error| protocol::fail(error)) {
        "probe" => probe(),
        "sdxl_endpoint" => run_sdxl_endpoint(&request),
        "sdxl_endpoint_probe" => probe_sdxl_endpoint(),
        "sdxl_inspect_output" => inspect_sdxl_output(&request),
        "run" => run(&request),
        "run_batch" => run_five_rung_batch(&request),
        // Hidden zero-budget CPU self-test of THIS sealed binary's FLUX3 permit boundary; refused
        // unless its own environment guard is set and no production invocation exists.
        flux3_common::boundary_self_test::SELF_TEST_ACTION => {
            flux3_common::boundary_self_test::native_boundary_self_test(&request, "candle")
        }
        // sc-24217: the child-process half of `protocol::ensure_supra2_bundle` — never a capture.
        protocol::SUPRA2_IMG_CONVERT_ACTION => protocol::supra2_convert(&request),
        other => Err(format!("unsupported action {other:?}")),
    }
    .unwrap_or_else(|error| protocol::fail(error));
    protocol::write_response(&response).unwrap_or_else(|error| protocol::fail(error));
}

#[cfg(test)]
#[path = "candle_sdxl_family_tests.rs"]
mod sdxl_family_tests;

#[cfg(test)]
#[path = "candle_mage_tests.rs"]
mod mage_tests;

#[cfg(test)]
#[path = "candle_tests.rs"]
mod tests;

/// sc-22731 — the SANA and Chroma1 Candle arms. Weights-free and env-free: everything asserted
/// here is decided before a snapshot is opened, so it holds on any host.
#[cfg(test)]
#[path = "candle_sana_chroma_candle_tests.rs"]
mod sana_chroma_candle_tests;

/// sc-22734 — the Candle SenseNova-U1 family arm: six catalog models over the `sensenova_u1_8b`
/// and `sensenova_u1_8b_fast` registry ids.
///
/// Env-free and weights-free by construction, so the plan binding is proven at typecheck/test time
/// rather than on a capture host. (This lane's tests execute on CI only —
/// `scripts/check-candle-build.mjs` typechecks them locally but cannot link libcuda — so every
/// assertion here is arithmetic or string work over checked-in data.)
#[cfg(test)]
#[path = "candle_sensenova_candle_tests.rs"]
mod sensenova_candle_tests;

/// sc-22738 (CUDA run 34356681566): the four adapter-arm classes that diverged from the worker's
/// production path — the safety handshake before `begin`, the loaded provider's component floors
/// on the selected tier, a resident render with no provider scope, and the Bernini spec the engine
/// prices its contract from. Each is asserted against a synthetic provider surface or the produced
/// `LoadSpec`, weights-free.
#[cfg(test)]
#[path = "candle_sc22738_arm_tests.rs"]
mod sc22738_arm_tests;
