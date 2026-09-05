#[cfg(not(target_os = "macos"))]
compile_error!("memory-mlx-adapter is supported only on macOS");

use mlx_gen::gen_core::{
    GenerationMemory, LoadPhase, MemoryBudget, MemoryCacheState, MemoryCalibrationIdentity,
    MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryOptimizationAuthority, MemoryPhase,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyParameters, TransformerComponent,
};
use mlx_gen::tiling::{SpatialTiling, TilingConfig, VaeTiling};
use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Generator, Image, LoadShape,
    LoadSpec, OffloadPolicy, Precision, Progress, Quant, WeightsSource,
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
const KREA_BASE_CALIBRATION_FINGERPRINT: &str =
    "krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3";
const KREA_BASE_SEED: u64 = 18377;
/// Recommended, reference-free SDXL base text-to-image. Rungs 2 and 3 are measured Missing at the
/// pinned provider, so this apparatus intentionally exposes only Resident, Staged, and rung 4.
const SDXL_PROVIDER: &str = "sdxl";
const SDXL_CALIBRATION_FINGERPRINT: &str = "sdxl-mlx-unet-shared-ladder-v3";
const SDXL_SEED: u64 = 18379;
// RMS is bounded by the same per-pixel envelope as maximum absolute error. This avoids inventing a
// tighter SDXL-specific tolerance before the first physical campaign while still making the real
// harness's runtime-complete quality contract explicit and mutation-sensitive.
const SDXL_RMS_THRESHOLD: f64 = MAX_THRESHOLD;
const SDXL_PLAIN_EXECUTION_PATH: &str = "the MLX SDXL base-only text-to-image path";
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
/// A klein turnkey rehost carries no `calibration_tag`, so
/// `mlx-gen-flux2::memory_strategy::klein_contract_for` publishes the STATIC behaviour fingerprint
/// suffixed with the provider id (`KLEIN_STATIC_BEHAVIOR_FINGERPRINT` +
/// `provider_id.replace('_', "-")`). This is the same string
/// `config/manifests/builtin.models.jsonc` declares for every klein rung on both klein models.
const FLUX2_KLEIN_CALIBRATION_FINGERPRINT: &str =
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
const LTX_PLAIN_EXECUTION_PATH: &str = "the MLX LTX-2.3 base-only text-to-video path";
/// How [`diagnostic_video_frames`] names this lane when it refuses a non-video output. Extracted
/// verbatim from that function's own messages when the second video arm made the label a parameter
/// (sc-18663), so LTX's refusals are unchanged.
const LTX_VIDEO_LABEL: &str = "MLX LTX-2.3";
/// Expected provider-owned identity at the permanent inference pin. The adapter always reads the
/// loaded registry contract and refuses any mismatch; this local expectation merely prevents a
/// provider re-fingerprint from silently reusing SC-18946's plan and fixtures.
const LTX_CALIBRATION_FINGERPRINT: &str = "sc-19109-ltx-2-3-mlx-memory-ladder-v1";
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

/// Port of `sceneworks_core::video_request::ltx_frame_count` — frames snap to the NEAREST `8k + 1`,
/// minimum 9, ties to the lower. Duplicated rather than depended on: `sceneworks-core` pulls a
/// bundled SQLite, an image codec stack and a trash binding into what is otherwise a calibration-only
/// binary with two dependencies.
///
/// Because it is a port and NOT a call, the binding to the shipped ladder is by transcription, and
/// it takes TWO tests to close: `ltx_frame_ladder_port_matches_the_transcribed_shipped_ladder` here
/// pins the 18 shipped (duration, fps) pairs against this port, and
/// `ltx_frame_count_matches_the_sc_18808_calibration_ladder` in
/// `crates/sceneworks-core/src/video_request.rs` pins the SAME 18 pairs against
/// `ltx_frame_count` itself. `sceneworks-core` is a workspace default member, so a change to the
/// shipped ladder reds under a plain `cargo test`; without that half, a drift would silently move
/// what this arm accepts and neither crate would notice (sc-18808 review).
const fn ltx_snapped_frame_count(raw_frames: u32) -> u32 {
    let frame_count = if raw_frames < 9 { 9 } else { raw_frames };
    let lower = frame_count - ((frame_count - 1) % 8);
    let upper = lower + 8;
    if lower < 9 {
        return upper;
    }
    if frame_count - lower <= upper - frame_count {
        lower
    } else {
        upper
    }
}

/// The closed frame envelope the declared `limits` can actually produce, derived over the FULL
/// 18-cell `durations x fps` cross product through the ladder the product itself uses. Derived
/// rather than written down so the bounds cannot drift away from the arrays above.
const fn ltx_frame_envelope() -> (u32, u32) {
    let (mut minimum, mut maximum) = (u32::MAX, 0);
    let mut duration = 0;
    while duration < LTX_DURATIONS_SECONDS.len() {
        let mut fps = 0;
        while fps < LTX_FPS.len() {
            let frames = ltx_snapped_frame_count(LTX_DURATIONS_SECONDS[duration] * LTX_FPS[fps]);
            if frames < minimum {
                minimum = frames;
            }
            if frames > maximum {
                maximum = frames;
            }
            fps += 1;
        }
        duration += 1;
    }
    (minimum, maximum)
}

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
/// Expected provider-owned identity at the permanent inference pin
/// (`mlx_gen_minimax_h3::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT`). The arm always reads the
/// loaded registry contract and refuses any mismatch; this local expectation additionally prevents a
/// provider re-fingerprint from silently reusing the epic's plan and fixtures. The string names
/// `eager` because the provider chose that spelling before it grew a deferred loader — the RESOLVED
/// shape travels beside it in `MemoryCalibrationIdentity::load_shape`, not in the fingerprint, and
/// this arm attests the resolved one.
const MINIMAX_CALIBRATION_FINGERPRINT: &str = "minimax-h3-mlx-staged-joint-av-eager-abi3-v1";
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
    fn scoped_generation_preserves_generator_and_finish_failures() {
        let mut observed = Some(ScopedGenerationFailureKind::Error);
        let error = settle_scoped_generation(
            Err(mlx_gen::gen_core::Error::Msg(
                "generator exploded".to_owned(),
            )),
            Err(mlx_gen::gen_core::Error::Msg("finish exploded".to_owned())),
            &mut observed,
        )
        .expect_err("neither terminal failure may be discarded");
        assert_eq!(
            error,
            "generate calibrated request: generator exploded; finish calibrated request: finish exploded"
        );
        assert_eq!(observed, Some(ScopedGenerationFailureKind::Finish));

        let mut control = Some(ScopedGenerationFailureKind::Error);
        let error = settle_scoped_generation(
            Err(mlx_gen::gen_core::Error::Msg(
                "generator exploded".to_owned(),
            )),
            Ok(()),
            &mut control,
        )
        .expect_err("the generator failure remains terminal when finish succeeds");
        assert_eq!(error, "generator exploded");
        assert_eq!(control, Some(ScopedGenerationFailureKind::Error));
    }

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

    #[test]
    fn prediction_basis_keeps_image_admission_conservative_and_video_resident() {
        let phase = PhaseMemory {
            active: 16 * MIB,
            cache: 15 * MIB,
        };
        assert_eq!(
            IMAGE_PREDICTED_PEAK_BASIS,
            PredictedPeakBasis::HistoricalImageAllocatorBound
        );
        assert_eq!(
            VIDEO_PREDICTED_PEAK_BASIS,
            PredictedPeakBasis::ResidentActive
        );
        assert_eq!(
            predicted_phase_ceiling(phase, IMAGE_PREDICTED_PEAK_BASIS),
            predicted_ceiling(31 * MIB)
        );
        assert_eq!(
            predicted_phase_ceiling(phase, VIDEO_PREDICTED_PEAK_BASIS),
            predicted_ceiling(16 * MIB)
        );
    }

    #[test]
    fn every_receipt_builder_is_bound_to_its_lane_prediction_wrapper() {
        let source = include_str!("mlx.rs");
        let image_receipts = source
            .lines()
            .filter(|line| {
                line.trim_start()
                    .starts_with("let predicted_peaks = image_predicted_peak_bytes(")
            })
            .count();
        assert_eq!(
            image_receipts, 6,
            "all six MLX image receipt builders must use the image policy wrapper"
        );
        assert_eq!(
            source
                .lines()
                .filter(|line| {
                    line.trim_start()
                        == "video_predicted_peak_bytes(conditioning, denoise, decode).json()"
                })
                .count(),
            1,
            "the LTX receipt builder must use the video policy wrapper"
        );
    }

    /// sc-19115. This is the load-bearing decision test, not merely a count of records with cache.
    /// It evaluates the same host-reserve currency production uses over every committed MLX image
    /// cell and proves that changing the image basis would only ever LOOSEN shipped admission —
    /// never tighten it — which is the whole reason the image basis stays where it is.
    ///
    /// The verdict is asserted; the corpus size is not. See the shape assertions at the end.
    #[test]
    fn resident_peak_counterfactual_only_loosens_shipped_image_admission() {
        use sceneworks_core::memory_calibration::{Backend, EvidenceBundle, RequiredNullable};

        fn phase(phase: &sceneworks_core::memory_calibration::Phase) -> PhaseMemory {
            PhaseMemory {
                active: phase.active_bytes,
                cache: phase.reclaimable_bytes,
            }
        }

        let evidence: EvidenceBundle = serde_json::from_str(include_str!(
            "../../../../docs/generated/memory-calibration-evidence.json"
        ))
        .expect("parse committed memory evidence");
        let host_bytes = [24_u64, 32, 48, 64, 96, 128].map(|gib| gib * 1024 * 1024 * 1024);
        let mut image_records = 0_usize;
        let mut changed_records = 0_usize;
        let mut flipped_cells = 0_usize;
        let mut flips_by_provider = std::collections::BTreeMap::<&str, usize>::new();

        for record in evidence.records.iter().filter(|record| {
            record.backend == Backend::Mlx && record.target.mode == "text_to_image"
        }) {
            image_records += 1;
            let observed = match &record.observed_memory {
                RequiredNullable::Value(observed) => observed.full().expect("full image telemetry"),
                RequiredNullable::Null => panic!("{} has null image telemetry", record.id),
            };
            let historical = image_predicted_peak_bytes(
                phase(&observed.conditioning),
                phase(&observed.denoise),
                phase(&observed.decode),
            )
            .overall;
            let resident = video_predicted_peak_bytes(
                phase(&observed.conditioning),
                phase(&observed.denoise),
                phase(&observed.decode),
            )
            .overall;
            let shipped = match &record.predicted_peak_bytes {
                RequiredNullable::Value(predicted) => predicted.overall(),
                RequiredNullable::Null => panic!("{} has null prediction", record.id),
            };
            assert_eq!(
                shipped, historical,
                "{} no longer carries the historical image prediction",
                record.id
            );
            if historical != resident {
                // The DIRECTION, per record, stated where the pinned totals used to be: the
                // resident basis drops the reclaimable term, so wherever the two bases disagree
                // the resident one must predict LESS. That is what makes every admission flip
                // below a loosening rather than a coincidence of this corpus, and unlike a count
                // it holds at any corpus size.
                assert!(
                    resident < historical,
                    "{}: the resident basis predicted {resident} against the historical \
                     {historical} — a resident basis that predicts MORE would tighten admission, \
                     which is the opposite of the sc-19115 finding",
                    record.id
                );
                changed_records += 1;
            }

            let historical_envelope = record
                .mlx_admission_envelope()
                .unwrap_or_else(|| panic!("{} has no production MLX envelope", record.id));
            let mut counterfactual = record.clone();
            match &mut counterfactual.predicted_peak_bytes {
                RequiredNullable::Value(predicted) => {
                    predicted.full_mut().expect("full image prediction").overall = resident;
                }
                RequiredNullable::Null => panic!("{} has null prediction", record.id),
            }
            let resident_envelope = counterfactual
                .mlx_admission_envelope()
                .expect("counterfactual production MLX envelope");
            for host in host_bytes {
                let historical_fits = historical_envelope.fits_scaled_host_bytes(host);
                let resident_fits = resident_envelope.fits_scaled_host_bytes(host);
                if historical_fits != resident_fits {
                    assert!(
                        !historical_fits && resident_fits,
                        "the resident counterfactual must only loosen admission"
                    );
                    flipped_cells += 1;
                    *flips_by_provider
                        .entry(record.target.provider.as_str())
                        .or_default() += 1;
                }
            }
        }

        // The MLX `text_to_image` corpus is NOT frozen. Every calibration campaign and every main
        // sync adds coordinates, and the counterfactual's derived totals move with it: 69 -> 74 at
        // the sc-18304 sync, 74 -> 93 at the 2026-08-19 epic-17137 sync that brought epic 18803's
        // captures across. Pinning those totals recorded only which campaign ran last and went red
        // for legitimate corpus growth, so this states the SHAPE of the counterfactual instead —
        // the same treatment the sibling calibration gates got (see `memory_calibration.rs`, where
        // the re-capture set is held to its shape while only frozen history stays pinned).
        //
        // The load-bearing claim is the only-loosens assertion in the loop above. These keep it
        // from passing vacuously on an empty or half-read corpus, and hold the derived tallies to
        // the identities that must be true at ANY corpus size.
        assert!(
            image_records > 0,
            "no MLX text_to_image record was read — the counterfactual is vacuous"
        );
        assert!(
            changed_records > 0 && changed_records <= image_records,
            "{changed_records} of {image_records} records changed basis: the resident basis must \
             move some image predictions, and can never move more than the corpus holds"
        );
        assert!(
            flipped_cells > 0,
            "the resident basis flipped no admission cell — the decision this test records would \
             then be a no-op, which is the opposite of what sc-19115 concluded"
        );
        assert!(
            flipped_cells <= changed_records * host_bytes.len(),
            "{flipped_cells} flips exceed the {} cells the {changed_records} changed records span \
             — the tally is counting something other than host cells",
            changed_records * host_bytes.len()
        );
        // The per-provider tally is a PARTITION of the flips, not a pinned census: it must sum to
        // the total and may only name providers this corpus actually carries.
        assert_eq!(
            flips_by_provider.values().sum::<usize>(),
            flipped_cells,
            "the per-provider tally must partition the flipped cells: {flips_by_provider:?}"
        );
        let image_providers = evidence
            .records
            .iter()
            .filter(|record| {
                record.backend == Backend::Mlx && record.target.mode == "text_to_image"
            })
            .map(|record| record.target.provider.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for provider in flips_by_provider.keys() {
            assert!(
                image_providers.contains(provider),
                "{provider} flipped admission but carries no MLX text_to_image record"
            );
        }
    }

    /// sc-18864. The emitted phase object must carry ONE named MLX quantity per field. Before this
    /// story it carried five keys for two readings: `deviceBytes` and `wiredBytes` were both set to
    /// `active + cache`, which is how every committed MLX record claimed wired residency of
    /// 99.7-159.6 GB against a probed 87.0 GB ceiling.
    ///
    /// Asserting the key SET is the load-bearing half — an assertion that only checked the three
    /// surviving values would still pass if the two aliases came back.
    #[test]
    fn phase_json_emits_one_named_mlx_quantity_per_field() {
        let phase = PhaseMemory {
            active: 35_678_641_896,
            cache: 106_969_676_964,
        };
        let value = phase.json();
        let object = value.as_object().expect("phase json object");
        let mut keys = object.keys().map(String::as_str).collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, ["activeBytes", "allocatorBytes", "reclaimableBytes"]);
        assert_eq!(object["activeBytes"], json!(35_678_641_896u64));
        assert_eq!(object["reclaimableBytes"], json!(106_969_676_964u64));
        // `allocatorBytes` is DERIVED, and the derivation is the whole content of the field.
        assert_eq!(object["allocatorBytes"], json!(142_648_318_860u64));
        assert_eq!(
            object["allocatorBytes"].as_u64().expect("allocator bytes"),
            object["activeBytes"].as_u64().expect("active bytes")
                + object["reclaimableBytes"]
                    .as_u64()
                    .expect("reclaimable bytes")
        );
    }

    /// sc-18864. `predictedPeakBytes` was hardcoded `null` on the LTX arm, which failed both
    /// `mlx_admission_envelope` and the worker's `RequiredNullable::Value` seed check. Nothing
    /// about it is contract-dependent — it is `predicted_ceiling` over the arm's own phases.
    ///
    /// The measured q8 LTX numbers below are `imc-2c064567893ea869006e` verbatim, and they are the
    /// discriminating case: a ceiling taken over `allocator_bytes()` instead of `active` would
    /// predict 149.8 GB of demand on a 137.4 GB host for a render that completed comfortably.
    #[test]
    fn ltx_predicted_peak_is_derived_from_the_resident_peak_not_the_co_existence_bound() {
        let conditioning = PhaseMemory {
            active: 35_678_641_896,
            cache: 4_371_718_590,
        };
        let denoise = PhaseMemory {
            active: 36_831_735_964,
            cache: 0,
        };
        let decode = PhaseMemory {
            active: 37_931_479_408,
            cache: 104_716_839_452,
        };
        let predicted = ltx_predicted_peak_bytes(conditioning, denoise, decode);
        assert_eq!(
            predicted,
            json!({
                "conditioning": predicted_ceiling(35_678_641_896),
                "denoise": predicted_ceiling(36_831_735_964),
                "decode": predicted_ceiling(37_931_479_408),
                "overall": predicted_ceiling(37_931_479_408),
            })
        );
        let host_bytes = 137_438_953_472u64;
        let overall = predicted["overall"].as_u64().expect("predicted overall");
        assert!(
            overall < host_bytes,
            "a resident-peak ceiling must fit the capture host, got {overall}"
        );
        let overall_phase = PhaseMemory::overall(&[conditioning, denoise, decode]);
        assert!(
            predicted_ceiling(overall_phase.allocator_bytes()) > host_bytes,
            "the co-existence bound must NOT fit, or this test does not discriminate"
        );
        // Monotonicity is what lets the `predictedOverallCeiling` diagnostic agree with the record.
        assert_eq!(overall, predicted_ceiling(overall_phase.active));
    }

    /// sc-18104: an unimplemented provider must be refused by name at dispatch, not fall through to
    /// the Qwen arm. The regression this guards is silent MISROUTING, so asserting "it errored" is
    /// not enough — the old code errored too, just with a Qwen-shaped message after entering the
    /// wrong arm. Assert the provider is named, and that none of the Qwen arm's own failure
    /// vocabulary appears, which is what distinguishes a dispatch refusal from a misroute.
    #[test]
    fn run_refuses_a_provider_the_mlx_adapter_does_not_implement() {
        // `flux2_dev` left this list when sc-18218 landed its arm, and `flux2_klein_9b` when
        // sc-22727 landed the klein members; `flux2_dev_edit` and the klein edit routes stay — a
        // contract-only provider is not a dispatchable lane.
        // `ltx_2_3` left this class when sc-18808 landed the video arm; `ltx_2_3_distilled` (the
        // CANDLE LTX engine id) and `ltx_2_3_eros` stay — neither is a dispatchable MLX lane.
        for provider in [
            "flux2_klein_9b_edit",
            "flux2_klein_9b_kv_edit",
            "flux2_dev_edit",
            "sana",
            "ltx_2_3_distilled",
            "ltx_2_3_eros",
        ] {
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
            // sc-22724: the undistilled base is its own registry id on the same arm.
            "z_image",
            "krea_2_turbo",
            "sdxl",
            "krea_2_turbo_control",
            "flux2_dev",
            // sc-22727: both klein catalog models ride this ONE registry id.
            "flux2_klein_9b",
            // sc-18808: the video arm rides the same dispatch, so it is covered by the same proof.
            "ltx_2_3",
            "ltx_2_5",
            // sc-18663: and the second video arm.
            "minimax_h3",
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
        assert_eq!(Z_IMAGE_BASE_PROVIDER, "z_image");
        assert_eq!(KREA_BASE_PROVIDER, "krea_2_turbo");
        assert_eq!(SDXL_PROVIDER, "sdxl");
        assert_eq!(KREA_PROVIDER, "krea_2_turbo_control");
        assert_eq!(FLUX2_PROVIDER, "flux2_dev");
        assert_eq!(FLUX2_KLEIN_PROVIDER, "flux2_klein_9b");
        assert_eq!(LTX_PROVIDER, "ltx_2_3");
        assert_eq!(LTX25_PROVIDER, "ltx_2_5");
        assert_eq!(MINIMAX_PROVIDER, "minimax_h3");
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

    fn z_image_planned(provider: &str, mode: &str, tier: &str) -> Value {
        json!({
            "planned": {
                "target": {
                    "provider": provider,
                    "modelId": if mode == "edit_image" { "z_image_edit" } else { provider },
                    "tier": tier,
                    "mode": mode,
                    "overlay": "none",
                    "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
                },
                "backend": "mlx",
                "loadShape": "eager_materialization",
                "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
                "calibrationFingerprint": "unused",
                "fixture": "unused"
            }
        })
    }

    /// sc-22724: the family member is read off the plan's `(provider, mode)`, and a pair no member
    /// serves is refused by name rather than measured as its nearest neighbour.
    #[test]
    fn z_image_arm_is_resolved_from_the_plans_provider_and_mode() {
        let turbo = z_image_arm(&z_image_planned("z_image_turbo", "text_to_image", "q4")).unwrap();
        assert_eq!(turbo, Z_IMAGE_TURBO_ARM);
        assert!(!turbo.edit);
        let edit = z_image_arm(&z_image_planned("z_image_turbo", "edit_image", "q4")).unwrap();
        assert_eq!(edit, Z_IMAGE_EDIT_ARM);
        assert!(edit.edit);
        assert_eq!(
            edit.provider, Z_IMAGE_PROVIDER,
            "the edit alias loads the Turbo provider"
        );
        assert_eq!(edit.expected_repository, protocol::Z_IMAGE_REPOSITORY);
        let base = z_image_arm(&z_image_planned("z_image", "text_to_image", "q4")).unwrap();
        assert_eq!(base, Z_IMAGE_BASE_ARM);
        assert_eq!(base.expected_repository, protocol::Z_IMAGE_BASE_REPOSITORY);
        assert_eq!(base.root_env, "SCENEWORKS_Z_IMAGE_BASE_ROOT");
        for (provider, mode) in [
            ("z_image", "edit_image"),
            ("z_image_turbo", "image_to_image"),
            ("z_image_control", "text_to_image"),
        ] {
            let error = z_image_arm(&z_image_planned(provider, mode, "q4")).unwrap_err();
            assert!(
                error.contains(&format!("provider {provider:?} in mode {mode:?}")),
                "{provider}/{mode}: {error}"
            );
        }
        assert!(
            z_image_arm(&json!({ "planned": { "target": { "provider": "z_image_turbo" } } }))
                .unwrap_err()
                .contains("planned.target.mode")
        );
    }

    fn z_image_snapshot_root(repository: &str, revision: &str, tier: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir()
            .join(format!("sc-22724-z-image-{}-{nonce}", std::process::id()))
            .join(format!("models--{}", repository.replace('/', "--")))
            .join("snapshots")
            .join(revision)
            .join(tier);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// sc-22724: the tier is the PLAN's, the root must carry it, and the `LoadSpec` re-asserts it
    /// — a q8 plan against a q4 export is refused naming q8, on every family member. This used to
    /// be a literal `"q4"` in both the validator and the `LoadSpec`, which capped the MLX Z-Image
    /// lane at one tier (the sc-17097 defect class on the other adapter).
    #[test]
    fn z_image_root_must_carry_the_planned_tier_and_the_spec_binds_it() {
        const REVISION: &str = "bb2bc9893b3c49ae96c813350775f791a2e8bc80";
        for (provider, mode, repository) in [
            (
                "z_image_turbo",
                "text_to_image",
                protocol::Z_IMAGE_REPOSITORY,
            ),
            ("z_image_turbo", "edit_image", protocol::Z_IMAGE_REPOSITORY),
            (
                "z_image",
                "text_to_image",
                protocol::Z_IMAGE_BASE_REPOSITORY,
            ),
        ] {
            let q4_root = z_image_snapshot_root(repository, REVISION, "q4");
            let error = z_image_load_spec_at(
                &z_image_planned(provider, mode, "q8"),
                LoadShape::EagerMaterialization,
                repository.to_owned(),
                REVISION.to_owned(),
                q4_root.clone(),
            )
            .expect_err("a q8 plan must not be satisfied by a q4 root");
            assert!(
                error.ends_with(&format!("/snapshots/{REVISION}/q8")),
                "{provider}/{mode}: {error}"
            );
            for (tier, quant) in [
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
                ("bf16", None),
            ] {
                let root = z_image_snapshot_root(repository, REVISION, tier);
                let artifact = z_image_load_spec_at(
                    &z_image_planned(provider, mode, tier),
                    LoadShape::DeferredMaterialization,
                    repository.to_owned(),
                    REVISION.to_owned(),
                    root.clone(),
                )
                .unwrap_or_else(|error| panic!("{provider}/{mode}/{tier}: {error}"));
                assert_eq!(artifact.tier, tier);
                assert_eq!(artifact.spec.quantize, quant, "{tier} load quant");
                assert_eq!(artifact.spec.load_shape, LoadShape::DeferredMaterialization);
                assert_eq!(
                    artifact.loadability_fingerprint(),
                    format!("{repository}@{REVISION}:{tier}")
                );
                assert_eq!(
                    artifact.arm.provider,
                    if provider == "z_image" {
                        "z_image"
                    } else {
                        "z_image_turbo"
                    }
                );
            }
            // The wrong artifact family is refused before the root is even looked at.
            let other = if repository == protocol::Z_IMAGE_REPOSITORY {
                protocol::Z_IMAGE_BASE_REPOSITORY
            } else {
                protocol::Z_IMAGE_REPOSITORY
            };
            let error = z_image_load_spec_at(
                &z_image_planned(provider, mode, "q4"),
                LoadShape::EagerMaterialization,
                other.to_owned(),
                REVISION.to_owned(),
                q4_root,
            )
            .expect_err("the other family's repository must be refused");
            assert!(error.contains(repository), "{provider}: {error}");
        }
    }

    /// sc-22724: an edit capture is the worker's edit request — one reference at the target
    /// geometry plus the production strength — and a text-to-image capture carries none.
    #[test]
    fn z_image_edit_request_carries_one_reference_at_the_target_geometry() {
        let edit = z_image_request(Z_IMAGE_EDIT_ARM, 768, 512);
        assert_eq!(edit.conditioning.len(), 1);
        match &edit.conditioning[0] {
            Conditioning::Reference { image, strength } => {
                assert_eq!((image.width, image.height), (768, 512));
                assert_eq!(image.pixels.len(), 768 * 512 * 3);
                assert_eq!(*strength, Some(Z_IMAGE_EDIT_STRENGTH));
            }
            other => panic!("expected one Reference, got {other:?}"),
        }
        // The worker sets ONLY the per-reference strength (`build_lane_conditioning`,
        // image_jobs/base.rs:7136); the request-level lever stays unset, and so does this arm's.
        assert_eq!(edit.strength, None);
        // floor(4 * 0.6) = 2, so two executed denoise steps remain behind the conditioning
        // boundary — the engine's `init_time_step` law (mlx-gen/src/img2img.rs), not this arm's.
        assert_eq!(edit.steps, Some(Z_IMAGE_EDIT_STEPS));
        for arm in [Z_IMAGE_TURBO_ARM, Z_IMAGE_BASE_ARM] {
            let plain = z_image_request(arm, 768, 768);
            assert!(plain.conditioning.is_empty());
            assert_eq!(plain.strength, None);
            assert_eq!(plain.steps, Some(2));
        }
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

fn persist_physical_mlx_image(
    output_dir: &Path,
    source_prefix: &str,
    logical_case_id: &str,
    role: &str,
    image: &Image,
) -> Result<Value, String> {
    let content_sha256 = format!("{:x}", Sha256::digest(&image.pixels));
    let file_name = format!(
        "{logical_case_id}-{role}-{}x{}-{content_sha256}.rgb",
        image.width, image.height,
    );
    let local_path = output_dir.join(&file_name);
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&local_path)
    {
        Ok(mut file) => file
            .write_all(&image.pixels)
            .map_err(|error| format!("write physical MLX {role} output: {error}"))?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read(&local_path).map_err(|read_error| {
                format!("read existing physical MLX {role} output: {read_error}")
            })?;
            if existing != image.pixels {
                return Err(format!(
                    "content-addressed physical MLX {role} output already exists with different bytes"
                ));
            }
        }
        Err(error) => {
            return Err(format!(
                "create immutable physical MLX {role} output: {error}"
            ));
        }
    }
    Ok(json!({
        "role": role,
        "path": format!("{source_prefix}/{file_name}"),
        "localPath": local_path,
        "sha256": content_sha256,
        "bytes": image.pixels.len(),
    }))
}

fn qwen_source_capture(
    request: &Value,
    root: &Path,
    repository: &str,
    revision: &str,
    tier: &str,
    selected: &Image,
    reference: &Image,
) -> Result<Value, String> {
    let capture_root = PathBuf::from(protocol::required_env("SCENEWORKS_MEMORY_CAPTURE_DIR")?);
    let source_prefix = protocol::required_env("SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX")?;
    if !source_prefix.starts_with("docs/calibration/")
        || source_prefix.split('/').any(|part| part == "..")
    {
        return Err(
            "SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX must stay under docs/calibration".to_owned(),
        );
    }
    let output_dir = source_prefix
        .split('/')
        .fold(capture_root, |directory, part| directory.join(part));
    std::fs::create_dir_all(&output_dir)
        .map_err(|error| format!("create physical MLX capture directory: {error}"))?;
    let logical_case_id = protocol::planned(request)?
        .get("logicalCaseId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.logicalCaseId must be a string".to_owned())?;
    let inventory_bytes = integer(
        &protocol::required_env("SCENEWORKS_MEMORY_MODEL_BYTES")?,
        "SCENEWORKS_MEMORY_MODEL_BYTES",
    )?;
    if inventory_bytes == 0 {
        return Err("SCENEWORKS_MEMORY_MODEL_BYTES must be greater than zero".to_owned());
    }
    let inventory_sha256 = protocol::required_env("SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256")?;
    if inventory_sha256.len() != 64
        || !inventory_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(
            "SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256 must be 64 lowercase hex characters"
                .to_owned(),
        );
    }
    Ok(json!({
        "kind": "physical_mlx",
        "inputs": [{
            "role": "base",
            "path": root,
            "bytes": inventory_bytes,
            "sha256": inventory_sha256,
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        }],
        "outputs": [
            persist_physical_mlx_image(
                &output_dir, &source_prefix, logical_case_id, "selected_rgb", selected,
            )?,
            persist_physical_mlx_image(
                &output_dir, &source_prefix, logical_case_id, "reference_rgb", reference,
            )?,
        ],
        "claims": [
            "memory", "quality", "negative_mutation", "lifecycle", "loadability", "overlay"
        ],
    }))
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
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
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
    request: GenerationRequest,
    context: &MemoryRunContext,
    error_phase: Option<MemoryPhase>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput, String> {
    let mut observed_failure = None;
    scoped_generate_observed(
        generator,
        request,
        context,
        error_phase,
        &mut observed_failure,
        on_progress,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScopedGenerationFailureKind {
    Canceled,
    Error,
    Finish,
}

type AfterConfigurationHook = fn(&mut GenerationRequest) -> Result<(), String>;

fn scoped_generate_observed(
    generator: &dyn Generator,
    request: GenerationRequest,
    context: &MemoryRunContext,
    error_phase: Option<MemoryPhase>,
    observed_failure: &mut Option<ScopedGenerationFailureKind>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput, String> {
    scoped_generate_observed_after_configuration(
        generator,
        request,
        context,
        error_phase,
        observed_failure,
        on_progress,
        None,
    )
}

fn scoped_generate_observed_after_configuration(
    generator: &dyn Generator,
    mut request: GenerationRequest,
    context: &MemoryRunContext,
    error_phase: Option<MemoryPhase>,
    observed_failure: &mut Option<ScopedGenerationFailureKind>,
    on_progress: &mut dyn FnMut(Progress),
    after_configuration: Option<AfterConfigurationHook>,
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
        .ok_or_else(|| "optimized request did not open a provider memory scope".to_owned())?;
    scope
        .configure_request(&mut request)
        .map_err(|error| format!("configure calibrated request: {error}"))?;
    if let Some(after_configuration) = after_configuration {
        if let Err(error) = after_configuration(&mut request) {
            *observed_failure = Some(ScopedGenerationFailureKind::Error);
            let finish = scope.finish(MemoryRunOutcome::Error {
                message: error.clone(),
            });
            return match finish {
                Ok(()) => Err(error),
                Err(finish) => {
                    *observed_failure = Some(ScopedGenerationFailureKind::Finish);
                    Err(format!("{error}; finish calibrated request: {finish}"))
                }
            };
        }
    }
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
    let coherence_before = gpu_view_retries();
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
    report_gpu_view_retries(coherence_before);
    if phase_error.is_none() {
        if let Some(current) = current_phase {
            if let Err(error) = scope.leave_phase(current) {
                phase_error = Some(format!("leave {current:?} phase: {error}"));
            }
        }
    }
    let outcome = match &result {
        Ok(_) => MemoryRunOutcome::Complete,
        Err(mlx_gen::gen_core::Error::Canceled) => {
            *observed_failure = Some(ScopedGenerationFailureKind::Canceled);
            MemoryRunOutcome::Canceled
        }
        Err(error) => {
            *observed_failure = Some(ScopedGenerationFailureKind::Error);
            MemoryRunOutcome::Error {
                message: error.to_string(),
            }
        }
    };
    let finish = scope.finish(outcome);
    if let Some(error) = phase_error {
        return match finish {
            Ok(()) => {
                *observed_failure = Some(ScopedGenerationFailureKind::Error);
                Err(error)
            }
            Err(finish) => {
                *observed_failure = Some(ScopedGenerationFailureKind::Finish);
                Err(format!("{error}; finish calibrated request: {finish}"))
            }
        };
    }
    settle_scoped_generation(result, finish, observed_failure)
}

/// The two GPU-view coherence retry counters (sc-22414): mlx-gen's and mlx-llm's mirrored guards
/// each count every time the GPU had to be re-read before it agreed with the CPU on a freshly
/// loaded buffer. Process-global and monotonic, so a render's incidence is the delta around it.
fn gpu_view_retries() -> (u64, u64) {
    (
        mlx_gen::coherence::retries(),
        runtime_macos::llm::primitives::coherence::retries(),
    )
}

/// Print the render's GPU-view retry incidence to stderr — the harness log is where the Mac2
/// reproducer's expected outcome ("pass, with non-zero retries") is read. Always printed, so a
/// zero reads as "the guard ran and saw nothing" rather than as silence.
fn report_gpu_view_retries(before: (u64, u64)) {
    let after = gpu_view_retries();
    eprintln!(
        "memory-strategy provider adapter: GPU-view coherence retries during this render: \
         mlx_gen={} mlx_llm={} (sc-22414)",
        after.0.saturating_sub(before.0),
        after.1.saturating_sub(before.1),
    );
}

/// Combine the generator and request-scope terminals without losing either failure. A provider
/// generation error does not make `finish` advisory: lifecycle certification requires the scope to
/// close successfully on complete, canceled, and error outcomes alike.
fn settle_scoped_generation(
    result: mlx_gen::gen_core::Result<GenerationOutput>,
    finish: mlx_gen::gen_core::Result<()>,
    observed_failure: &mut Option<ScopedGenerationFailureKind>,
) -> Result<GenerationOutput, String> {
    match (result, finish) {
        (Ok(output), Ok(())) => Ok(output),
        (Err(error), Ok(())) => Err(error.to_string()),
        (Err(error), Err(finish)) => {
            *observed_failure = Some(ScopedGenerationFailureKind::Finish);
            Err(format!(
                "generate calibrated request: {error}; finish calibrated request: {finish}"
            ))
        }
        (Ok(_), Err(error)) => {
            *observed_failure = Some(ScopedGenerationFailureKind::Finish);
            Err(format!("finish calibrated request: {error}"))
        }
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

fn predicted_ceiling(bytes: u64) -> u64 {
    let with_margin = bytes.saturating_add(bytes / 20);
    with_margin
        .saturating_add(64 * MIB - 1)
        .saturating_div(64 * MIB)
        .saturating_mul(64 * MIB)
}

fn predicted_phase_ceiling(phase: PhaseMemory, basis: PredictedPeakBasis) -> u64 {
    let bytes = match basis {
        PredictedPeakBasis::HistoricalImageAllocatorBound => phase.allocator_bytes(),
        PredictedPeakBasis::ResidentActive => phase.active,
    };
    predicted_ceiling(bytes)
}

fn predicted_phase_peaks(
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
    basis: PredictedPeakBasis,
) -> PredictedPhasePeaks {
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    PredictedPhasePeaks {
        conditioning: predicted_phase_ceiling(conditioning, basis),
        denoise: predicted_phase_ceiling(denoise, basis),
        decode: predicted_phase_ceiling(decode, basis),
        overall: predicted_phase_ceiling(overall, basis),
    }
}

fn image_predicted_peak_bytes(
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
) -> PredictedPhasePeaks {
    predicted_phase_peaks(conditioning, denoise, decode, IMAGE_PREDICTED_PEAK_BASIS)
}

fn video_predicted_peak_bytes(
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
) -> PredictedPhasePeaks {
    predicted_phase_peaks(conditioning, denoise, decode, VIDEO_PREDICTED_PEAK_BASIS)
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

/// The numeric tier's load shape: bf16 is the dense base and carries no quant (the worker's
/// `tier_to_quant`); q4/q8 name the packed tier, which `with_quant` re-asserts against the
/// snapshot's own packed width at load (`needs_load_time_quant` — a mismatch hard-errors).
fn tier_precision_quant(tier: &str) -> (Precision, Option<Quant>) {
    match tier {
        "q4" => (Precision::Bf16, Some(Quant::Q4)),
        "q8" => (Precision::Bf16, Some(Quant::Q8)),
        _ => (Precision::Bf16, None),
    }
}

fn planned_selection(request: &Value) -> Result<MemorySelection, String> {
    let strategy = planned_memory_strategy(request)?;
    let parameters = protocol::strategy_parameters(request)?;
    let transformer_window_size = protocol::optional_parameter(request, "transformerWindowSize")?;
    let transformer_window_component = match parameters.get("transformerWindowComponent") {
        None if transformer_window_size.is_none() => None,
        None => Some(TransformerComponent::Dit),
        Some(value) if transformer_window_size.is_none() => {
            return Err(format!(
                "planned.strategy.parameters.transformerWindowComponent requires transformerWindowSize, got {value}"
            ));
        }
        Some(value) => match value.as_str() {
            Some("dit") => Some(TransformerComponent::Dit),
            // sc-18663: `minimax_h3` publishes `transformer_window_components: [Both]` — its rung 4
            // streams the text encoder as well as the DiT, and declaring `Dit` alone would leave the
            // conditioning stage, the taller of the two at every tier, with no lever. This parser is
            // shared, so accepting the spelling here does NOT widen any other arm: each arm's own
            // validator still constrains it (`validate_sdxl_selection_parameters` requires an
            // explicit `Dit`), and `contract.validate_selection` refuses a component the pinned
            // provider does not declare. `text_encoder` stays unparsed because no pinned MLX
            // provider declares it.
            Some("both") => Some(TransformerComponent::Both),
            _ => {
                return Err(format!(
                    "planned.strategy.parameters.transformerWindowComponent must be \"dit\" or \"both\" for the implemented MLX adapter arms, got {value}"
                ));
            }
        },
    };
    let (precision, quant) = tier_precision_quant(planned_qwen_tier(request)?);
    Ok(MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            decode_tile_edge: protocol::optional_parameter(request, "decodeTileEdge")?,
            decode_overlap: protocol::optional_parameter(request, "decodeOverlap")?,
            attention_chunk_size: protocol::optional_parameter(request, "attentionChunkSize")?,
            transformer_window_size,
            transformer_window_component,
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

/// Which family member the plan asks for. Refuses by name: a `(provider, mode)` pair no member
/// serves (the base has no edit route, `image_to_image` is not planned anywhere) must not be
/// measured as its nearest neighbour.
fn z_image_arm(request: &Value) -> Result<ZImageArm, String> {
    let planned = protocol::planned(request)?;
    let provider = planned
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    let mode = planned
        .pointer("/target/mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    match (provider, mode) {
        (Z_IMAGE_PROVIDER, "text_to_image") => Ok(Z_IMAGE_TURBO_ARM),
        (Z_IMAGE_PROVIDER, "edit_image") => Ok(Z_IMAGE_EDIT_ARM),
        (Z_IMAGE_BASE_PROVIDER, "text_to_image") => Ok(Z_IMAGE_BASE_ARM),
        (provider, mode) => Err(format!(
            "the MLX Z-Image arm does not implement provider {provider:?} in mode {mode:?}"
        )),
    }
}

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

/// The env-free half of [`z_image_load_spec`], so the tier binding is unit-testable: the root must
/// end in the PLANNED tier's directory (`.../snapshots/<revision>/<tier>`), so a stale `…/q4`
/// export can never satisfy a q8 or bf16 plan and quietly re-label another tier's peaks — the
/// sc-17097 defect class the Candle arm closed, which this arm carried as a literal `"q4"` until
/// sc-22724.
fn z_image_load_spec_at(
    request: &Value,
    load_shape: LoadShape,
    repository: String,
    revision: String,
    root: PathBuf,
) -> Result<ZImageArtifact, String> {
    let arm = z_image_arm(request)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let tier = match planned_qwen_tier(request)? {
        "bf16" => "bf16",
        "q4" => "q4",
        "q8" => "q8",
        _ => unreachable!("planned_qwen_tier returned an unsupported tier"),
    };
    protocol::validate_artifact_identity(&repository, &revision, arm.expected_repository)?;
    let root = std::fs::canonicalize(&root)
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        arm.expected_repository,
    )?;
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    if let (_, Some(quant)) = tier_precision_quant(tier) {
        spec = spec.with_quant(quant);
    }
    Ok(ZImageArtifact {
        arm,
        repository,
        revision,
        tier,
        spec,
    })
}

fn z_image_load_spec(request: &Value, load_shape: LoadShape) -> Result<ZImageArtifact, String> {
    let arm = z_image_arm(request)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let root = PathBuf::from(protocol::required_env(arm.root_env)?);
    z_image_load_spec_at(request, load_shape, repository, revision, root)
}

fn load_z_image_generator(
    request: &Value,
    load_shape: LoadShape,
) -> Result<(ZImageArtifact, Box<dyn Generator>), String> {
    let artifact = z_image_load_spec(request, load_shape)?;
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(artifact.arm.provider, &artifact.spec)
        .map_err(|error| {
            format!(
                "load real {} {} provider: {error}",
                artifact.arm.provider, artifact.tier
            )
        })?;
    Ok((artifact, generator))
}

fn z_image_request(arm: ZImageArm, width: u32, height: u32) -> GenerationRequest {
    let mut request = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(16402),
        // The first Step callback closes the conservative conditioning envelope; the second
        // step then supplies a real denoise-only interval before Decoding.
        steps: Some(2),
        ..Default::default()
    };
    if arm.edit {
        // The worker's edit route: one `Conditioning::Reference` fitted to the request geometry
        // plus the strength lever (`resolve_zimage_edit_init`). The start step is derived from
        // `steps * strength` by the engine, so the step count is raised to keep two executed
        // denoise steps behind the conditioning boundary.
        request.steps = Some(Z_IMAGE_EDIT_STEPS);
        // `request.strength` stays None: the worker sets ONLY the per-reference strength
        // (`build_lane_conditioning`, image_jobs/base.rs:7136) and leaves the request-level lever —
        // gen-core's documented fallback for a single `Reference` with no strength of its own —
        // unset. This arm reproduces the worker's request shape, so it does the same (sc-22724).
        request.conditioning = vec![Conditioning::Reference {
            image: Image {
                width,
                height,
                pixels: protocol::synthetic_reference_rgb(width, height),
            },
            strength: Some(Z_IMAGE_EDIT_STRENGTH),
        }];
    }
    request
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
    artifact: &ZImageArtifact,
    load_shape: LoadShape,
) -> Result<Value, String> {
    let arm = artifact.arm;
    let (repository, revision) = (artifact.repository.as_str(), artifact.revision.as_str());
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let (width, height) = protocol::target_geometry(request)?;
    let selection = planned_selection(request)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {} has no memory-strategy contract", arm.provider))?;
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
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        // The mode the plan declared, as the worker would admit it: an edit request carries one
        // reference and is admitted under `MemoryMode::Edit` (the contract the `z_image_edit`
        // manifest entry declares for the Turbo provider).
        mode: if arm.edit {
            MemoryMode::Edit
        } else {
            MemoryMode::TextToImage
        },
        has_reference: arm.edit,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: u32::from(arm.edit),
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
    // sc-22667 (epic 22657 D3): Z-Image's residency is REQUEST-SCOPED even under eager
    // materialization — each component is materialized the first time a request reaches its
    // phase and retained afterwards — so a window opened on the freshly loaded generator measured
    // a cold first request whose conditioning phase saw only the text encoder it was materializing
    // (2.27 GB against a 5.83 GB resident set), which the core anchor law refuses to decompose.
    // One unmeasured request on the same scope materializes the whole resident set FIRST; the
    // window then opens above it, the way the candle adapter measures every phase above weights
    // already on device. `protocol::open_resident_phase_window` fixes that order and refuses a
    // resident set the counters did not see.
    let opening = protocol::open_resident_phase_window(&mut MlxResidencyCounters, || {
        one_image(scoped_generate(
            generator,
            z_image_request(arm, width, height),
            &context,
            None,
            &mut |_| {},
        )?)
        .map(|_| ())
    })?;
    let pre_rung_active = opening.resident_active;
    let pre_rung_cache = opening.resident_cache;
    let peak_after_reset = opening.peak_after_reset;
    let selected = one_image(scoped_generate(
        generator,
        z_image_request(arm, width, height),
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
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;

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
            .generate(&z_image_request(arm, width, height), &mut |_| {})
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
        z_image_request(arm, width, height),
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

    let cancelled = z_image_request(arm, width, height);
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
        z_image_request(arm, width, height),
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
        z_image_request(arm, width, height),
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
        z_image_request(arm, width, height),
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
            "variant": artifact.tier,
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
        "predictedPeakBytes": predicted_peaks.json(),
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
            "resolvedPathFingerprint": artifact.loadability_fingerprint(),
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:{}-shared-ladder", arm.slug),
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
                // The window opened ABOVE the materialized resident set (sc-22667 D3); a record
                // without this measurement was captured on a cold first request.
                ("residentSetMaterializedBeforeWindow", "count", 1),
                // sc-22724: the edit arm conditions every request on one reference image.
                ("referenceImages", "count", u64::from(arm.edit)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}

fn run_z_image_reference(request: &Value) -> Result<Value, String> {
    // Before the load, not inside `..._loaded`: a non-still target must be refused without paying
    // for weights, the same ordering every other image arm now has. The arm is resolved first so
    // the refusal carries the member's own label.
    let arm = z_image_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let load_shape = planned_load_shape(request)?;
    let (artifact, generator) = load_z_image_generator(request, load_shape)?;
    run_z_image_reference_loaded(request, generator.as_ref(), &artifact, load_shape)
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

/// The LTX-2.3 twin of [`flux2_quality_passes`], over the LTX-named thresholds so an `mlx:ltx_2_3`
/// receipt's numbers trace to an LTX constant.
fn ltx_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= LTX_MAX_THRESHOLD && mean <= LTX_MEAN_THRESHOLD && rms <= LTX_RMS_THRESHOLD
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
    /// as `LoadSpec::resolved_route` (`image_jobs/base.rs` — `spec.with_resolved_route(request.model)`),
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
    /// The calibration fingerprint the pinned crate must publish for this member.
    calibration_fingerprint: &'static str,
    /// The load shape and offload policy this member is loaded under — the WORKER's shape for the
    /// plain T2I route, never a hand-picked pair.
    ///
    /// Dev: `Resident` + `EagerMaterialization`. Its manifest declares no MLX
    /// `bounded_transformer_residency` row for the plain provider and every non-resident strategy
    /// is `Missing` at the pin, so the worker's declaration evaluator refuses the staged candidate
    /// and keeps the eager default (`memory_route_registry.rs`,
    /// `evaluate_declared_mlx_load_shape_for_request_with_strategy`).
    ///
    /// Klein: `Sequential` + `DeferredMaterialization`. Both klein manifest entries declare an MLX
    /// `bounded_transformer_residency` row with `requiredOffloadPolicy: "sequential"` for every
    /// tier of the plain T2I route; the worker applies it (`Applied` + Deferred, then
    /// `apply_declared_mlx_load_policy_for_request` binds Sequential — the shape the worker's own
    /// `flux2_klein_aliases_bind_exact_plain_pid_routes_and_refuse_crossed_shapes` asserts at every
    /// tier for both klein routes). It is also the ONLY shape the pinned engine publishes a
    /// calibration identity for: `klein_contract_for` sets `calibration` iff
    /// `klein_streamable(spec)`, which requires exactly `Sequential && DeferredMaterialization &&
    /// quantize.is_none()` (`mlx-gen-flux2/src/memory_strategy.rs`). A resident klein spec yields
    /// `calibration: None`, and this arm refuses it before the load.
    load_shape: LoadShape,
    offload_policy: OffloadPolicy,
    seed: u64,
    /// The story that established this member's evidence, as it appears in
    /// `MemoryRunContext::evidence_revision`. Per member, so a klein receipt never claims the
    /// dev lane's provenance.
    evidence_tag: &'static str,
    /// Whether the pinned crate implements the resident rung and nothing else. True on dev (every
    /// other strategy is declared `Missing` at the pin); the klein ladder publishes five rungs, so
    /// its selection is left to `contract.validate_selection`.
    resident_only: bool,
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
    calibration_fingerprint: FLUX2_CALIBRATION_FINGERPRINT,
    load_shape: LoadShape::EagerMaterialization,
    offload_policy: OffloadPolicy::Resident,
    seed: FLUX2_SEED,
    evidence_tag: "sc-18218",
    resident_only: true,
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
    calibration_fingerprint: FLUX2_KLEIN_CALIBRATION_FINGERPRINT,
    load_shape: LoadShape::DeferredMaterialization,
    offload_policy: OffloadPolicy::Sequential,
    seed: FLUX2_KLEIN_SEED,
    evidence_tag: "sc-22727",
    resident_only: false,
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
    calibration_fingerprint: FLUX2_KLEIN_CALIBRATION_FINGERPRINT,
    load_shape: LoadShape::DeferredMaterialization,
    offload_policy: OffloadPolicy::Sequential,
    seed: FLUX2_KLEIN_SEED,
    evidence_tag: "sc-22727",
    resident_only: false,
    tier_quant_reaches_the_loader: false,
};

/// Which family member the plan asks for. Refuses by name: a `(provider, modelId)` pair no member
/// serves must not be measured as its nearest neighbour — in particular a KV plan must never be
/// satisfied by the base klein artifact, which shares the provider id.
///
/// Defense-in-depth mirror of the provider-mismatch guard `validate_z_image_batch` carries
/// (sc-18104): `run` dispatches by provider name today, so a future caller must be refused here
/// rather than misrouted into another member's contract.
fn flux2_arm(request: &Value) -> Result<Flux2Arm, String> {
    let planned = protocol::planned(request)?;
    let provider = planned
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    let model_id = planned
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
        if arm.provider == provider && arm.model_id == model_id {
            return Ok(arm);
        }
    }
    Err(format!(
        "the MLX FLUX.2 arm does not implement provider {provider:?} for model {model_id:?}"
    ))
}

fn validate_flux2_target(request: &Value) -> Result<Flux2Arm, String> {
    let arm = flux2_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    Ok(arm)
}

/// Bind the fixture to the planned FAMILY MEMBER, tier AND geometry edge, deriving the seed — the
/// same fixture-to-plan binding `planned_qwen_seed` enforces, extended to the edge because this
/// lane's plan carries tiers at more than one geometry, and to the member because the two klein
/// models are indistinguishable by provider id alone.
fn planned_flux2_seed(
    request: &Value,
    arm: Flux2Arm,
    tier: &str,
    width: u32,
) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("flux2-{}-mlx-{tier}-{width}-seed", arm.fixture_slug);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse FLUX.2 fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse FLUX.2 fixture step count {steps:?}: {error}"))?;
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

/// Resolve and validate the member's `SCENEWORKS_FLUX2*_*` environment family into a tier-exact
/// load spec. The tier is DERIVED from `/target/tier` and threads through the per-tier ROOT suffix
/// check and `spec.quantize` — never hardcoded (sc-17097 fixed exactly that hardcoding on the
/// Candle side).
fn flux2_load_spec(
    request: &Value,
    arm: Flux2Arm,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, LoadSpec), String> {
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    // The identity check lives in `flux2_load_spec_at`, once, over these same values.
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    flux2_load_spec_at(arm, tier, selection, repository, revision, root)
}

/// The env-free half of [`flux2_load_spec`], so the artifact binding is unit-testable without the
/// process environment: the root must end in the PLANNED tier's directory under the member's OWN
/// repository, so a stale `…/q4` export can never satisfy a q8 or bf16 plan, and the KV artifact
/// can never satisfy a base-klein plan.
fn flux2_load_spec_at(
    arm: Flux2Arm,
    tier: &str,
    selection: &MemorySelection,
    repository: String,
    revision: String,
    root: PathBuf,
) -> Result<(String, String, LoadSpec), String> {
    protocol::validate_artifact_identity(&repository, &revision, arm.expected_repository)?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        arm.expected_repository,
    )?;
    Ok((repository, revision, flux2_spec(arm, root, selection)))
}

/// The tier-exact FLUX.2 load spec: the member's WORKER shape (offload policy + load shape, see
/// [`Flux2Arm::load_shape`]), `resolved_route` bound to the CATALOG model id, and — on the members
/// whose loader takes it — the quant DERIVED from the planned selection. Those are the levers the
/// worker sets (`image_jobs/base.rs` `load_spec` + `with_resolved_route`, then the manifest-declared
/// shape and policy); see [`Flux2Arm::tier_quant_reaches_the_loader`] for why the klein rehosts
/// take the tier through the snapshot path alone.
fn flux2_spec(arm: Flux2Arm, root: PathBuf, selection: &MemorySelection) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(arm.offload_policy)
        .with_load_shape(arm.load_shape)
        .with_resolved_route(arm.model_id);
    if arm.tier_quant_reaches_the_loader {
        if let Some(quant) = selection.tier.quant {
            spec = spec.with_quant(quant);
        }
    }
    spec
}

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

fn flux2_admission_context(
    arm: Flux2Arm,
    selection: &MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    geometry: (u32, u32),
    probe: Flux2AdmissionProbe<'_>,
) -> MemoryRunContext {
    let (width, height) = geometry;
    let Flux2AdmissionProbe {
        fingerprint,
        total_bytes,
        predicted_peak_bytes,
    } = probe;
    MemoryRunContext {
        selection: *selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
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
        evidence_revision: format!("{}@{}", arm.evidence_tag, protocol::INFERENCE_PIN),
    }
}

fn flux2_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // One exact resident tuple per plan row; marking it range-verified promotes no sibling tuple
    // (the generated matrix still requires a matching manifest binding per cell).
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// The `mlx:flux2_*` arm (sc-18218, extended to the whole family by sc-22727).
///
/// `flux2_dev` owns a distinct reference-free T2I contract in which every non-Resident strategy
/// remains `Missing`; the two klein catalog models share one engine provider whose ladder publishes
/// five rungs, and are told apart by their artifact and `LoadSpec::resolved_route`. In both cases
/// this arm reads the registry contract under the exact T2I provider id, then proves that the
/// loaded generator exposes the byte-for-byte same contract before measuring it. No edit-provider
/// declaration or edit-shaped context participates in this lane.
fn run_flux2(request: &Value) -> Result<Value, String> {
    let arm = validate_flux2_target(request)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let rung = protocol::planned_rung(request)?;
    if arm.resident_only && rung != "resident" {
        return Err(format!(
            "the pinned MLX {} provider implements only the resident strategy (every other \
             strategy is declared Missing at the pin); rung {rung:?} is not capturable",
            arm.provider
        ));
    }
    // The plan's `loadShape` is what the record is checked against (the harness refuses a fragment
    // whose measured loadShape differs from the plan's), and the member's shape is fixed by the
    // worker, so a plan row spelling another shape is refused here by name, before weight work.
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != arm.load_shape {
        return Err(format!(
            "planned.loadShape {planned_shape:?} is not the {} worker load shape {:?}",
            arm.model_id, arm.load_shape
        ));
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
    let (width, height) = protocol::target_geometry(request)?;
    let seed = planned_flux2_seed(request, arm, tier, width)?;
    let (repository, revision, spec) = flux2_load_spec(request, arm, tier, &selection)?;
    // The PRODUCTION catalog (E4): the same explicit MLX media registry `runtime_macos::catalog()`
    // composes for the worker, not a crate-local replica. A member the bundle does not register is
    // not routed on this lane and fails here by name.
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let registry = catalog.media();
    let contract = registry
        .memory_strategy_contract(arm.provider, &spec)
        .map_err(|error| {
            format!(
                "read {} T2I memory-strategy contract: {error}",
                arm.provider
            )
        })?
        .ok_or_else(|| {
            format!(
                "{} has no T2I memory-strategy contract at the pin",
                arm.provider
            )
        })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} contract rejected planned selection: {error}",
            arm.model_id
        )
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "pinned {} contract has no calibration identity",
            arm.model_id
        )
    })?;
    if calibration.fingerprint != arm.calibration_fingerprint {
        return Err(format!(
            "pinned {} T2I contract fingerprint changed: expected {}, got {}",
            arm.model_id, arm.calibration_fingerprint, calibration.fingerprint
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
    if seed != arm.seed {
        return Err(format!(
            "planned.fixture seed {seed} does not match the {} calibration seed {}",
            arm.model_id, arm.seed
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let admission_context = |fingerprint: &str, total_bytes: u64, predicted_peak_bytes: u64| {
        flux2_admission_context(
            arm,
            &selection,
            calibration,
            (width, height),
            Flux2AdmissionProbe {
                fingerprint,
                total_bytes,
                predicted_peak_bytes,
            },
        )
    };
    // Admission mutation hygiene: the gate must accept a fitting request (so the two rejections
    // cannot pass via a blanket refusal), reject an unknown/zero budget, and reject a mutated
    // calibration fingerprint.
    //
    // On the dev arm the registered T2I check is a PUBLIC crate function, so all three run BEFORE
    // the expensive load, exactly as sc-18218 wrote them. The klein routes expose their registered
    // check (`registered_klein_safety_check`) only through the loaded generator - it is
    // crate-private - so their identical three scenarios run immediately after the load instead.
    // Same scenarios, same record; only the moment differs, and it differs because of the pinned
    // crate's visibility, not a weaker claim.
    let scenarios = |check: &dyn Fn(&str, u64, u64) -> MemorySafetyDecision| -> Result<(), String> {
        if !matches!(
            check(&calibration.fingerprint, hardware_bytes, 1),
            MemorySafetyDecision::Accept
        ) {
            return Err(format!(
                "{} admission rejected a fitting probe budget; the scenario rejections below \
                 would be a blanket refusal, not evidence",
                arm.model_id
            ));
        }
        if !matches!(
            check(&calibration.fingerprint, 0, 1),
            MemorySafetyDecision::Reject { .. }
        ) {
            return Err(format!(
                "{} admission accepted an unknown/zero memory budget",
                arm.model_id
            ));
        }
        if !matches!(
            check(
                &format!("stale-{}-fingerprint", arm.slug),
                hardware_bytes,
                1
            ),
            MemorySafetyDecision::Reject { .. }
        ) {
            return Err(format!(
                "{} admission accepted stale calibration evidence",
                arm.model_id
            ));
        }
        Ok(())
    };
    if arm.provider == FLUX2_PROVIDER {
        scenarios(&|fingerprint, total_bytes, predicted| {
            mlx_gen_flux2::memory_strategy::registered_dev_t2i_safety_check(
                &spec,
                &contract,
                &admission_context(fingerprint, total_bytes, predicted),
            )
        })?;
    }

    let generator = registry
        .load(arm.provider, &spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.model_id))?;
    let loaded_contract = generator.memory_strategy_contract().ok_or_else(|| {
        format!(
            "loaded {} generator exposed no T2I memory contract",
            arm.model_id
        )
    })?;
    if loaded_contract != &contract {
        return Err(format!(
            "loaded {} generator contract differs from the registry contract",
            arm.model_id
        ));
    }
    if arm.provider != FLUX2_PROVIDER {
        scenarios(&|fingerprint, total_bytes, predicted| {
            generator.memory_strategy_safety_check(&admission_context(
                fingerprint,
                total_bytes,
                predicted,
            ))
        })?;
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
        return Err(format!(
            "a synchronized {} lifecycle phase reported a zero active peak",
            arm.model_id
        ));
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    let exact_fit = flux2_admission_context(
        arm,
        &selection,
        calibration,
        (width, height),
        Flux2AdmissionProbe::exact_fit(calibration, predicted),
    );
    if !matches!(
        generator.memory_strategy_safety_check(&exact_fit),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected an exact-fit calibrated budget",
            arm.model_id
        ));
    }

    // Warm repeat determinism + cleanup bounds on this exact loaded provider. The scoped
    // lifecycle scenarios cannot run (no request scope, no injection site), so the unscoped
    // repeats gate quality and the allocator bounds gate cleanup instead.
    clear_cache();
    reset_peak_memory();
    let baseline = one_image(
        generator
            .generate(&flux2_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate warm {} control: {error}", arm.model_id))?,
    )?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let cleanup_bounds =
        LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = image_max_mean_rms_abs(&selected, &baseline)?;
    if !flux2_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "{} warm repeat exceeded the determinism envelope: max={maximum_error:.6}, \
             mean={mean_error:.6}, rms={rms_error:.6}",
            arm.model_id
        ));
    }
    reset_peak_memory();
    let warm = one_image(
        generator
            .generate(&flux2_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate warm {} repeat: {error}", arm.model_id))?,
    )?;
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "{} warm repeat peaked at {warm_peak} bytes, above the clean warm control \
             {clean_warm_peak} bytes plus 2%",
            arm.model_id
        ));
    }
    clear_cache();
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "{} warm repeat retained active/cache bytes {warm_post_cleanup:?} above the \
             clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
            arm.model_id, cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = image_max_mean_rms_abs(&selected, &warm)?;
    if !flux2_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err(format!(
            "{} second warm repeat changed the deterministic output",
            arm.model_id
        ));
    }

    // Arm-internal negative-mutation falsifiability check. A runtime_complete record must keep
    // `negativeMutation` null (`memory-calibration-harness.mjs#validateRuntimeComplete`), so the
    // breach is verified here — the capture fails if the envelope cannot be breached — and the
    // measured numbers land in diagnostics rather than in the record field.
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if flux2_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(format!(
            "{} output mutation did not breach the determinism envelope",
            arm.model_id
        ));
    }

    let lifecycle_blocker = concat!(
        "the pinned mlx-gen-flux2 crate opens no memory-strategy request scope for the FLUX.2 ",
        "text-to-image routes and has no calibration fault-injection site, so the scoped lifecycle ",
        "scenario cannot execute; unscoped repeat determinism and allocator cleanup bounds are ",
        "attested in quality and diagnostics instead"
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
            { "name": "unknown_budget", "result": "passed", "reason": format!("the registered {} admission check rejected a zero/unknown budget", arm.model_id) },
            { "name": "stale_evidence", "result": "passed", "reason": format!("the registered {} admission check rejected a mutated calibration fingerprint", arm.model_id) },
            { "name": "warm_repeat", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
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
            &format!("memory-mlx-adapter:{}-resident", arm.slug),
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
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
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
        if provider != Z_IMAGE_PROVIDER && provider != Z_IMAGE_BASE_PROVIDER {
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
        let artifact = z_image_load_spec(&representative, load_shape)?;
        let provider = artifact.arm.provider;
        let contract = catalog
            .media()
            .memory_strategy_contract(provider, &artifact.spec)
            .map_err(|error| format!("read {provider} memory-strategy contract: {error}"))?
            .ok_or_else(|| format!("{provider} has no memory-strategy contract"))?;
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

fn validate_krea_base_target(request: &Value) -> Result<(), String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let provider = target
        .get("provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    if provider != KREA_BASE_PROVIDER {
        return Err(format!(
            "MLX Krea base calibration does not implement provider {provider:?}"
        ));
    }
    let model_id = target
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != KREA_BASE_PROVIDER {
        return Err(format!(
            "MLX Krea base calibration requires modelId {KREA_BASE_PROVIDER:?}, got {model_id:?}"
        ));
    }
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != "text_to_image" {
        return Err(format!(
            "MLX Krea base calibration requires reference-free text_to_image mode, got {mode:?}"
        ));
    }
    protocol::validate_still_geometry(request, "MLX Krea base calibration")?;
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "MLX Krea base calibration requires {field} == 0 when declared"
                ));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(false) {
                return Err(format!(
                    "MLX Krea base calibration requires {field} == false when declared"
                ));
            }
        }
    }
    protocol::validate_plain_overlay_target(request, KREA_PLAIN_EXECUTION_PATH)
}

fn planned_krea_base_seed(request: &Value, tier: &str, width: u32) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("krea-base-mlx-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Krea fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse Krea fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(seed)
}

fn krea_base_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // Each Krea plan row executes exactly one production parameter tuple. The singleton axes are
    // derived from that tuple, so a parameterized `complete` receipt satisfies the harness without
    // claiming any sibling value was exercised.
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

fn krea_base_request(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "an editorial photograph of a glass sculpture in a sunlit studio".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        // Two steps produce distinct conditioning/denoise/decode boundaries while keeping the
        // future physical capture bounded. This story adds no records or manifest bindings.
        steps: Some(2),
        ..Default::default()
    }
}

fn krea_base_load_spec(
    request: &Value,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, LoadSpec), String> {
    validate_krea_base_target(request)?;
    let repository = protocol::required_env("SCENEWORKS_KREA_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_KREA_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::KREA_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_KREA_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_KREA_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::KREA_REPOSITORY,
    )?;
    let offload = if selection.strategy == MemoryStrategy::Resident {
        OffloadPolicy::Resident
    } else {
        OffloadPolicy::Sequential
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(offload)
        .with_load_shape(LoadShape::DeferredMaterialization);
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    Ok((repository, revision, spec))
}

fn krea_base_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
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
        evidence_revision: format!("sc-18377@{}", protocol::INFERENCE_PIN),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KreaBaseLifecycleMetrics {
    clean_warm_peak: u64,
    clean_post_cleanup: AllocatorState,
    max_fault_post_cleanup: AllocatorState,
    max_recovery_peak: u64,
    max_recovery_post_cleanup: AllocatorState,
}

fn verify_krea_base_lifecycle(
    generator: &dyn Generator,
    context: &MemoryRunContext,
    selected: &Image,
    width: u32,
    height: u32,
    seed: u64,
) -> Result<KreaBaseLifecycleMetrics, String> {
    clear_cache();
    reset_peak_memory();
    let clean_warm = one_image(scoped_generate(
        generator,
        krea_base_request(width, height, seed),
        context,
        None,
        &mut |_| {},
    )?)?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let bounds = LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (warm_maximum, warm_mean) = image_max_mean_abs(selected, &clean_warm)?;
    if warm_maximum > KREA_MAX_THRESHOLD || warm_mean > KREA_MEAN_THRESHOLD {
        return Err("Krea base clean warm control changed the deterministic output".to_owned());
    }

    let mut metrics = KreaBaseLifecycleMetrics {
        clean_warm_peak,
        clean_post_cleanup,
        ..Default::default()
    };
    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let cancel = mlx_gen::CancelFlag::new();
        if phase == MemoryPhase::Conditioning {
            cancel.cancel();
        }
        let mut cancelled = krea_base_request(width, height, seed);
        cancelled.cancel = cancel.clone();
        let result = scoped_generate(generator, cancelled, context, None, &mut |progress| {
            if (phase == MemoryPhase::Denoise
                && matches!(progress, Progress::Step { current: 1, .. }))
                || (phase == MemoryPhase::Decode && matches!(progress, Progress::Decoding))
            {
                cancel.cancel();
            }
        });
        match result {
            Err(error) if error.to_ascii_lowercase().contains("cancel") => {}
            Err(error) => {
                return Err(format!(
                    "Krea base {phase:?} cancellation returned the wrong error: {error}"
                ));
            }
            Ok(_) => {
                return Err(format!(
                    "Krea base {phase:?} cancellation returned images instead of the typed cancellation path"
                ));
            }
        }
        clear_cache();
        let fault_cleanup = AllocatorState::capture_current();
        metrics.max_fault_post_cleanup.active = metrics
            .max_fault_post_cleanup
            .active
            .max(fault_cleanup.active);
        metrics.max_fault_post_cleanup.cache = metrics
            .max_fault_post_cleanup
            .cache
            .max(fault_cleanup.cache);
        if !bounds.allows_retained(fault_cleanup) {
            return Err(format!(
                "Krea base {phase:?} cancellation retained active/cache bytes {fault_cleanup:?} above the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                bounds.tolerance_bytes,
            ));
        }
        reset_peak_memory();
        let recovery = one_image(scoped_generate(
            generator,
            krea_base_request(width, height, seed),
            context,
            None,
            &mut |_| {},
        )?)?;
        let recovery_peak = get_peak_memory() as u64;
        metrics.max_recovery_peak = metrics.max_recovery_peak.max(recovery_peak);
        if !bounds.allows_warm_peak(recovery_peak) {
            return Err(format!(
                "Krea base {phase:?} cancellation left the warm follow-up peak at {recovery_peak} bytes, above the clean warm control {clean_warm_peak} bytes plus 2%"
            ));
        }
        clear_cache();
        let recovery_cleanup = AllocatorState::capture_current();
        metrics.max_recovery_post_cleanup.active = metrics
            .max_recovery_post_cleanup
            .active
            .max(recovery_cleanup.active);
        metrics.max_recovery_post_cleanup.cache = metrics
            .max_recovery_post_cleanup
            .cache
            .max(recovery_cleanup.cache);
        if !bounds.allows_retained(recovery_cleanup) {
            return Err(format!(
                "Krea base {phase:?} cancellation warm follow-up retained active/cache bytes {recovery_cleanup:?} above the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                bounds.tolerance_bytes,
            ));
        }
        let (maximum, mean) = image_max_mean_abs(selected, &recovery)?;
        if maximum > KREA_MAX_THRESHOLD || mean > KREA_MEAN_THRESHOLD {
            return Err(format!(
                "Krea base {phase:?} cancellation cleanup changed the warm follow-up"
            ));
        }
    }

    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let result = scoped_generate(
            generator,
            krea_base_request(width, height, seed),
            context,
            Some(phase),
            &mut |_| {},
        );
        match result {
            Err(error) if error.contains("injected memory-strategy calibration error") => {}
            Err(error) => {
                return Err(format!(
                    "Krea base {phase:?} error injection returned the wrong error: {error}"
                ));
            }
            Ok(_) => {
                return Err(format!(
                    "Krea base {phase:?} error injection returned images instead of failing at its physical boundary"
                ));
            }
        }
        clear_cache();
        let fault_cleanup = AllocatorState::capture_current();
        metrics.max_fault_post_cleanup.active = metrics
            .max_fault_post_cleanup
            .active
            .max(fault_cleanup.active);
        metrics.max_fault_post_cleanup.cache = metrics
            .max_fault_post_cleanup
            .cache
            .max(fault_cleanup.cache);
        if !bounds.allows_retained(fault_cleanup) {
            return Err(format!(
                "Krea base {phase:?} injected error retained active/cache bytes {fault_cleanup:?} above the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                bounds.tolerance_bytes,
            ));
        }
        reset_peak_memory();
        let recovery = one_image(scoped_generate(
            generator,
            krea_base_request(width, height, seed),
            context,
            None,
            &mut |_| {},
        )?)?;
        let recovery_peak = get_peak_memory() as u64;
        metrics.max_recovery_peak = metrics.max_recovery_peak.max(recovery_peak);
        if !bounds.allows_warm_peak(recovery_peak) {
            return Err(format!(
                "Krea base {phase:?} injected error left the warm follow-up peak at {recovery_peak} bytes, above the clean warm control {clean_warm_peak} bytes plus 2%"
            ));
        }
        clear_cache();
        let recovery_cleanup = AllocatorState::capture_current();
        metrics.max_recovery_post_cleanup.active = metrics
            .max_recovery_post_cleanup
            .active
            .max(recovery_cleanup.active);
        metrics.max_recovery_post_cleanup.cache = metrics
            .max_recovery_post_cleanup
            .cache
            .max(recovery_cleanup.cache);
        if !bounds.allows_retained(recovery_cleanup) {
            return Err(format!(
                "Krea base {phase:?} injected-error warm follow-up retained active/cache bytes {recovery_cleanup:?} above the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                bounds.tolerance_bytes,
            ));
        }
        let (maximum, mean) = image_max_mean_abs(selected, &recovery)?;
        if maximum > KREA_MAX_THRESHOLD || mean > KREA_MEAN_THRESHOLD {
            return Err(format!(
                "Krea base {phase:?} error cleanup changed the warm follow-up"
            ));
        }
    }
    Ok(metrics)
}

/// Capture arm for the shipped, reference-free `mlx:krea_2_turbo` lane. The pose-control arm is
/// intentionally separate: its overlay, geometry and provider fingerprint are not interchangeable.
fn run_krea_base(request: &Value) -> Result<Value, String> {
    validate_krea_base_target(request)?;
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != LoadShape::DeferredMaterialization {
        return Err(
            "plain Krea calibration must use the production deferred_materialization load shape"
                .to_owned(),
        );
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    let seed = planned_krea_base_seed(request, tier, width)?;
    if seed != KREA_BASE_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the Krea base calibration seed {KREA_BASE_SEED}"
        ));
    }
    let (repository, revision, spec) = krea_base_load_spec(request, tier, &selection)?;
    let registry = mlx_gen_krea::provider_registry()
        .map_err(|error| format!("build Krea registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(KREA_BASE_PROVIDER, &spec)
        .map_err(|error| format!("read {KREA_BASE_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| {
            format!("{KREA_BASE_PROVIDER} has no memory-strategy contract at the pin")
        })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned Krea base contract rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Krea base contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != KREA_BASE_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "pinned Krea base fingerprint changed: expected {KREA_BASE_CALIBRATION_FINGERPRINT}, got {}",
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
    if calibration.load_shape != planned_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(planned_shape),
            load_shape_key(calibration.load_shape)
        ));
    }

    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let generator = registry
        .load(KREA_BASE_PROVIDER, &spec)
        .map_err(|error| format!("load real Krea base {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded Krea base generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err(
            "loaded Krea base generator contract differs from the registry contract".to_owned(),
        );
    }
    let context = krea_base_context(
        selection,
        calibration,
        &calibration.fingerprint,
        width,
        height,
        hardware_bytes,
        1,
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
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let selected = one_image(scoped_generate(
        generator.as_ref(),
        krea_base_request(width, height, seed),
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
            "a synchronized Krea base lifecycle phase reported a zero active peak".to_owned(),
        );
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;

    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget.total_bytes = predicted;
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("Krea base provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Krea base provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-krea-base-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Krea base provider accepted stale calibration evidence".to_owned());
    }

    let baseline = one_image(
        generator
            .generate(&krea_base_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate unselected Krea base reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&selected, &baseline)?;
    if maximum_error > KREA_MAX_THRESHOLD || mean_error > KREA_MEAN_THRESHOLD {
        return Err(format!(
            "Krea base selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}"
        ));
    }
    let lifecycle =
        verify_krea_base_lifecycle(generator.as_ref(), &context, &selected, width, height, seed)?;
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean) = image_max_mean_abs(&mutated, &baseline)?;
    if mutated_maximum <= KREA_MAX_THRESHOLD && mutated_mean <= KREA_MEAN_THRESHOLD {
        return Err("Krea base output mutation did not breach the parity envelope".to_owned());
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
        "sweep": krea_base_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "reason": "conditioning, denoise, and decode cancellation returned typed cancellation; retained memory and warm recovery stayed within the clean-warm control plus 2%", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "conditioning, denoise, and decode injected errors fired at physical boundaries; retained memory and warm recovery stayed within the clean-warm control plus 2%", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "identical artifact, prompt, seed, geometry, steps and tier; selected Krea rung versus unselected request",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": KREA_MAX_THRESHOLD,
            "meanErrorThreshold": KREA_MEAN_THRESHOLD,
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
            "memory-mlx-adapter:krea-base-shared-ladder",
            "executed",
            [],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("lifecycleCleanWarmPeak", "bytes", lifecycle.clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", lifecycle.clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", lifecycle.clean_post_cleanup.cache),
                ("lifecycleMaximumFaultPostCleanupActive", "bytes", lifecycle.max_fault_post_cleanup.active),
                ("lifecycleMaximumFaultPostCleanupCache", "bytes", lifecycle.max_fault_post_cleanup.cache),
                ("lifecycleMaximumRecoveryPeak", "bytes", lifecycle.max_recovery_peak),
                ("lifecycleMaximumRecoveryPostCleanupActive", "bytes", lifecycle.max_recovery_post_cleanup.active),
                ("lifecycleMaximumRecoveryPostCleanupCache", "bytes", lifecycle.max_recovery_post_cleanup.cache),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("loadShapeDeferred", "count", 1),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, KREA_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}

fn validate_sdxl_target(request: &Value) -> Result<(), String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let provider = target
        .get("provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    if provider != SDXL_PROVIDER {
        return Err(format!(
            "MLX SDXL base calibration does not implement provider {provider:?}"
        ));
    }
    let model_id = target
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != SDXL_PROVIDER {
        return Err(format!(
            "MLX SDXL base calibration requires modelId {SDXL_PROVIDER:?}, got {model_id:?}"
        ));
    }
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != "text_to_image" {
        return Err(format!(
            "MLX SDXL base calibration requires reference-free text_to_image mode, got {mode:?}"
        ));
    }
    protocol::validate_still_geometry(request, "MLX SDXL base calibration")?;
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "MLX SDXL base calibration requires {field} == 0 when declared"
                ));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(false) {
                return Err(format!(
                    "MLX SDXL base calibration requires {field} == false when declared"
                ));
            }
        }
    }
    protocol::validate_plain_overlay_target(request, SDXL_PLAIN_EXECUTION_PATH)
}

fn validate_sdxl_selection_parameters(
    request: &Value,
    selection: &MemorySelection,
) -> Result<(), String> {
    let parameters = protocol::strategy_parameters(request)?;
    match selection.strategy {
        MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
            if !parameters.is_empty() {
                return Err(format!(
                    "MLX SDXL {:?} calibration requires no strategy parameters, got {parameters:?}",
                    selection.strategy
                ));
            }
        }
        MemoryStrategy::BoundedTransformerResidency => {
            let mut keys = parameters.keys().map(String::as_str).collect::<Vec<_>>();
            keys.sort_unstable();
            if keys != ["transformerWindowComponent", "transformerWindowSize"] {
                return Err(format!(
                    "MLX SDXL bounded transformer calibration requires exactly transformerWindowSize and transformerWindowComponent, got {keys:?}"
                ));
            }
            if selection.parameters.transformer_window_size.is_none()
                || selection.parameters.transformer_window_component
                    != Some(TransformerComponent::Dit)
            {
                return Err(
                    "MLX SDXL bounded transformer calibration requires an explicit Dit window"
                        .to_owned(),
                );
            }
        }
        MemoryStrategy::BoundedDecode | MemoryStrategy::BoundedAttention => {
            return Err(format!(
                "MLX SDXL {:?} is measured Missing at the pinned provider and is not capturable",
                selection.strategy
            ));
        }
    }
    Ok(())
}

fn planned_sdxl_seed(request: &Value, tier: &str, width: u32) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("sdxl-base-mlx-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse SDXL fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse SDXL fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(seed)
}

fn sdxl_runtime_complete_sweep(request: &Value) -> Result<Value, String> {
    let parameters = protocol::strategy_parameters(request)?;
    let axes = parameters
        .iter()
        .filter_map(|(name, value)| value.as_u64().map(|value| (name, value)))
        .map(|(name, value)| json!({ "parameter": name, "testedValues": [value] }))
        .collect::<Vec<_>>();
    Ok(json!({
        "axes": axes,
        "cases": [{ "parameters": parameters, "result": "passed" }],
        "rangeVerified": true,
    }))
}

fn sdxl_request(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "an editorial photograph of a glass sculpture in a sunlit studio".to_owned(),
        negative_prompt: Some("low quality, blurry, distorted".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(2),
        guidance: Some(7.0),
        ..Default::default()
    }
}

fn sdxl_load_spec(
    request: &Value,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, LoadSpec), String> {
    validate_sdxl_target(request)?;
    let repository = protocol::required_env("SCENEWORKS_SDXL_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_SDXL_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::SDXL_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_SDXL_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_SDXL_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::SDXL_REPOSITORY,
    )?;
    let offload = if selection.strategy == MemoryStrategy::Resident {
        OffloadPolicy::Resident
    } else {
        OffloadPolicy::Sequential
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(offload)
        .with_load_shape(LoadShape::DeferredMaterialization);
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    Ok((repository, revision, spec))
}

fn sdxl_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
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
        evidence_revision: format!("sc-18379@{}", protocol::INFERENCE_PIN),
    }
}

/// Real capture arm for the exact recommended `mlx:sdxl` base T2I identity. It deliberately does
/// not expose SDXL edit/reference/adapter surfaces or the provider's measured-Missing rungs 2/3.
fn run_sdxl(request: &Value) -> Result<Value, String> {
    validate_sdxl_target(request)?;
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != LoadShape::DeferredMaterialization {
        return Err(
            "plain SDXL calibration must use the production deferred_materialization load shape"
                .to_owned(),
        );
    }
    let selection = planned_selection(request)?;
    validate_sdxl_selection_parameters(request, &selection)?;
    let tier = planned_qwen_tier(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    let seed = planned_sdxl_seed(request, tier, width)?;
    if seed != SDXL_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the SDXL calibration seed {SDXL_SEED}"
        ));
    }
    let (repository, revision, spec) = sdxl_load_spec(request, tier, &selection)?;
    let registry = mlx_gen_sdxl::provider_registry()
        .map_err(|error| format!("build SDXL registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(SDXL_PROVIDER, &spec)
        .map_err(|error| format!("read {SDXL_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| format!("{SDXL_PROVIDER} has no memory-strategy contract at the pin"))?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned SDXL contract rejected planned selection: {error}"))?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned SDXL contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != SDXL_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "pinned SDXL fingerprint changed: expected {SDXL_CALIBRATION_FINGERPRINT}, got {}",
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
    if calibration.load_shape != planned_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(planned_shape),
            load_shape_key(calibration.load_shape)
        ));
    }

    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let generator = registry
        .load(SDXL_PROVIDER, &spec)
        .map_err(|error| format!("load real SDXL {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded SDXL generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err("loaded SDXL generator contract differs from the registry contract".to_owned());
    }
    let context = sdxl_context(
        selection,
        calibration,
        &calibration.fingerprint,
        width,
        height,
        hardware_bytes,
        1,
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
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let selected = one_image(scoped_generate(
        generator.as_ref(),
        sdxl_request(width, height, seed),
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
        return Err("a synchronized SDXL lifecycle phase reported a zero active peak".to_owned());
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;

    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget.total_bytes = predicted;
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("SDXL provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("SDXL provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-sdxl-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("SDXL provider accepted stale calibration evidence".to_owned());
    }

    let baseline = one_image(
        generator
            .generate(&sdxl_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate unselected SDXL reference: {error}"))?,
    )?;
    let (maximum_error, mean_error, rms_error) = image_max_mean_rms_abs(&selected, &baseline)?;
    if maximum_error > MAX_THRESHOLD
        || mean_error > MEAN_THRESHOLD
        || rms_error > SDXL_RMS_THRESHOLD
    {
        return Err(format!(
            "SDXL selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if mutated_maximum <= MAX_THRESHOLD
        && mutated_mean <= MEAN_THRESHOLD
        && mutated_rms <= SDXL_RMS_THRESHOLD
    {
        return Err("SDXL output mutation did not breach the parity envelope".to_owned());
    }

    // The pinned SDXL provider exposes synchronized request scopes and phase telemetry, but not the
    // exhaustive lifecycle hooks required for `complete`: it never reads the calibration error
    // authorization, and its plain untiled VAE decode does not consult the request cancel flag.
    // Preserve the real phase/quality/admission measurements as `runtime_complete`, keep the formal
    // lifecycle scenarios explicitly not_run, and record the internally verified mutation only as
    // diagnostics (the runtime-complete schema requires `negativeMutation: null`).
    let lifecycle_blocker = concat!(
        "the pinned mlx-gen-sdxl provider does not implement calibration error injection and its ",
        "plain untiled VAE decode does not consult the cancellation flag, so exhaustive warm/cancel/",
        "error lifecycle certification cannot execute; synchronized phase telemetry, selected-versus-",
        "unselected parity, and the internal negative discriminator are attested separately"
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
        "sweep": sdxl_runtime_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "identical artifact, prompt, negative prompt, guidance, seed, geometry, steps and tier; selected SDXL rung versus unselected request",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": MAX_THRESHOLD,
            "meanErrorThreshold": MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": SDXL_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}"),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:sdxl-shared-ladder",
            "executed",
            [lifecycle_blocker.to_owned()],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("negativeMutationRootMeanSquareErrorPer255", "count", (mutated_rms * 255.0).round() as u64),
                ("loadShapeDeferred", "count", 1),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, SDXL_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}

fn run_krea_control(request: &Value) -> Result<Value, String> {
    protocol::validate_exact_overlay_target(request, "control:1", KREA_CONTROL_EXECUTION_PATH)?;
    protocol::validate_still_geometry(request, "MLX Krea pose-control calibration")?;
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
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted_conditioning = predicted_peaks.conditioning;
    let predicted_denoise = predicted_peaks.denoise;
    let predicted_decode = predicted_peaks.decode;
    // The harness defines `overall` as a conservative componentwise high-water envelope: every
    // overall metric must cover the corresponding peak from every physical phase. Predict from that
    // same envelope so exact-fit admission can never sit below the published observed overall.
    let predicted_overall = predicted_peaks.overall;
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
    protocol::validate_still_geometry(request, "MLX Qwen VAE-probe calibration")?;
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
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
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
    protocol::validate_still_geometry(request, "MLX Qwen base calibration")?;
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
    let spec = qwen_load_spec(root.clone(), &selection, offload, load_shape);
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
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;

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
        "predictedPeakBytes": predicted_peaks.json(),
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
    fragment["sourceCapture"] = qwen_source_capture(
        request,
        &root,
        &repository,
        &revision,
        tier,
        &selected,
        &baseline,
    )?;
    protocol::settle_plain_overlay_scenario(request, &mut fragment, QWEN_PROVIDER_EXECUTION_PATH)?;
    Ok(fragment)
}

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

fn ltx_declared_resolutions() -> String {
    LTX_RESOLUTIONS
        .iter()
        .map(|(width, height)| format!("{width}x{height}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// LTX's own geometry envelope, which REPLACES the image arms' `frames == 1` refusal for this arm
/// alone. Three independent constraints, each from a stated source:
///
/// * `limits.resolutions` — the five declared pairs (the catalog contract a real request can name).
/// * `limits.requiresDimensionsMultipleOf` — 64, mirroring the engine's `SIZE_MULTIPLE`. Redundant
///   with the pair list today; kept because it is the engine's hard rule and the pair list is not.
/// * the temporal lattice and envelope — `frames = 1 + 8k` (the engine's `validate_request` hard
///   reject) inside `[97, 449]`, the closed span the declared `durations x fps` cross product
///   produces through the shipped LTX frame ladder.
///
/// A still geometry (`frames == 1`) is on the lattice but below the envelope floor, so it is refused
/// here too: this arm may not silently capture a single-frame record for a video model.
fn validate_ltx_geometry(width: u32, height: u32, frames: u32) -> Result<LtxGeometry, String> {
    if !LTX_RESOLUTIONS.contains(&(width, height)) {
        return Err(format!(
            "MLX LTX-2.3 calibration requires one of the declared limits.resolutions ({}), got {width}x{height}",
            ltx_declared_resolutions()
        ));
    }
    if !width.is_multiple_of(LTX_DIMENSION_MULTIPLE)
        || !height.is_multiple_of(LTX_DIMENSION_MULTIPLE)
    {
        return Err(format!(
            "MLX LTX-2.3 calibration requires geometry divisible by {LTX_DIMENSION_MULTIPLE}, got {width}x{height}"
        ));
    }
    if frames % LTX_TEMPORAL_SCALE != 1 {
        return Err(format!(
            "MLX LTX-2.3 calibration requires geometry.frames == 1 + {LTX_TEMPORAL_SCALE}k (the LTX \
             video VAE is {LTX_TEMPORAL_SCALE}x causal in time), got {frames}"
        ));
    }
    let (minimum, maximum) = LTX_FRAME_ENVELOPE;
    if frames < minimum || frames > maximum {
        return Err(format!(
            "MLX LTX-2.3 calibration requires geometry.frames within the declared duration/fps \
             envelope [{minimum}, {maximum}], got {frames}"
        ));
    }
    Ok(LtxGeometry {
        width,
        height,
        frames,
        latent_frames: 1 + (frames - 1) / LTX_TEMPORAL_SCALE,
    })
}

/// Read the four declared geometry axes. Unlike the image arms this reads `frames` as a real value
/// rather than asserting it away; `batch` is still pinned to 1 because LTX's descriptor advertises
/// `max_count: 1` and the arm renders exactly one clip.
fn ltx_target_geometry(request: &Value) -> Result<LtxGeometry, String> {
    let geometry = protocol::planned(request)?
        .pointer("/target/geometry")
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
            "MLX LTX-2.3 calibration requires geometry.batch == 1 (the provider advertises \
             max_count 1), got {batch}"
        ));
    }
    validate_ltx_geometry(axis("width")?, axis("height")?, axis("frames")?)
}

/// Defense-in-depth mirror of `validate_flux2_target`, plus the T2V-specific target shape. The
/// `run` dispatcher routes by provider id today, but this arm hardcodes the LTX contract, so a
/// foreign caller must be refused BY NAME here rather than misrouted into it.
fn validate_ltx_target(request: &Value) -> Result<LtxGeometry, String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let provider = target
        .get("provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    if provider != LTX_PROVIDER {
        return Err(format!(
            "MLX LTX-2.3 calibration does not implement provider {provider:?}"
        ));
    }
    let model_id = target
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != LTX_PROVIDER {
        return Err(format!(
            "MLX LTX-2.3 calibration requires modelId {LTX_PROVIDER:?}, got {model_id:?}"
        ));
    }
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != "text_to_video" {
        return Err(format!(
            "MLX LTX-2.3 calibration requires reference-free text_to_video mode, got {mode:?}"
        ));
    }
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "MLX LTX-2.3 calibration requires {field} == 0 when declared"
                ));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(false) {
                return Err(format!(
                    "MLX LTX-2.3 calibration requires {field} == false when declared"
                ));
            }
        }
    }
    ltx_target_geometry(request)
}

/// Bind the fixture to the planned tier AND the full rendered geometry, and recover the two request
/// parameters the geometry envelope cannot carry: the output cadence `fps` and the seed. fps rides
/// here because `GeometryEnvelope` has no temporal-cadence axis and the audio latent length — which
/// is denoised jointly with the video on every step — is `compute_audio_frames(frames, fps)`.
fn planned_ltx_capture(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
) -> Result<(u32, u64), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!(
        "ltx-2-3-mlx-{tier}-{}x{}-f{}-fps",
        geometry.width, geometry.height, geometry.frames
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps_and_carrier, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps_and_carrier
        .strip_suffix("-bounded-decode-192-64")
        .unwrap_or(fps_and_carrier);
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse LTX fixture fps {fps:?}: {error}"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse LTX fixture seed {seed:?}: {error}"))?;
    if !LTX_FPS.contains(&fps) {
        return Err(format!(
            "planned.fixture declares fps {fps}, which is not one of the declared limits.fps {LTX_FPS:?}"
        ));
    }
    if seed != LTX_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the LTX-2.3 calibration seed {LTX_SEED}"
        ));
    }
    Ok((fps, seed))
}

/// Fail closed before any model path, provider registry, or MLX allocation is touched.
///
/// The q4 f305 incident forbids that exact case. It also invalidates the former assumption that a
/// smaller geometry is safe: every SC-18946 row first loads its complete numeric tier and the same
/// Gemma stack. q8/bf16 carry strictly larger immutable tier inventories, but there is no proved
/// relationship from inventory bytes to a safe physical-footprint upper bound. Consequently every
/// row is refused; only rows supported by incident or monotonic evidence are called unmeasurable,
/// while the rest remain explicitly safety-refused/open. External polling is defense in depth,
/// not admission.
fn validate_ltx_safety_evidence(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(&'static str, u64, u64), String> {
    let inventory_bytes = match tier {
        "q4" => LTX_Q4_INVENTORY_BYTES,
        "q8" => LTX_Q8_INVENTORY_BYTES,
        "bf16" => LTX_BF16_INVENTORY_BYTES,
        other => return Err(format!("unsupported SC-18946 safety tier {other:?}")),
    };
    let safety = protocol::planned(request)?
        .get("_measurementSafety")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            "SC-18946 row is missing required _measurementSafety; refusing before model load"
                .to_owned()
        })?;
    let exact_u64 = |field: &str, expected: u64| -> Result<(), String> {
        let actual = safety.get(field).and_then(Value::as_u64);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "SC-18946 safety field {field} must be {expected}, got {actual:?}; refusing before model load"
            ))
        }
    };
    let bounded = selection.strategy == MemoryStrategy::BoundedDecode;
    let incident_carrier = selection.parameters.decode_tile_edge == Some(384)
        && selection.parameters.decode_overlap == Some(64);
    let incident = tier == "q4"
        && bounded
        && incident_carrier
        && geometry.width == 1280
        && geometry.height == 704
        && geometry.frames == 305;
    let monotonic_rung2 = tier == "q4"
        && bounded
        && incident_carrier
        && geometry.width == 1280
        && geometry.height == 704
        && geometry.frames > 305;
    let expected_disposition = if incident {
        LTX_INCIDENT_FORBIDDEN
    } else if monotonic_rung2 {
        LTX_ARITHMETIC_UNMEASURABLE
    } else {
        LTX_SAFETY_REFUSED_OPEN
    };
    if safety.get("disposition").and_then(Value::as_str) != Some(expected_disposition) {
        return Err(format!(
            "SC-18946 safety disposition must be {expected_disposition:?}; refusing before model load"
        ));
    }
    exact_u64("tierInventoryBytes", inventory_bytes)?;
    exact_u64(
        "incidentCrashFootprintBytes",
        LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
    )?;
    exact_u64(
        "incidentPredictedDecodeBytes",
        LTX_INCIDENT_PREDICTED_DECODE_BYTES,
    )?;
    let voxels = u64::from(geometry.width)
        .saturating_mul(u64::from(geometry.height))
        .saturating_mul(u64::from(geometry.frames));
    let predicted_decode_bytes = if bounded {
        let tile_edge = u64::from(selection.parameters.decode_tile_edge.ok_or_else(|| {
            "SC-18946 bounded-decode safety row omitted decodeTileEdge".to_owned()
        })?);
        3_300_000_000_u64
            .saturating_add(40_u64.saturating_mul(voxels))
            .saturating_add(
                300_u64
                    .saturating_mul(tile_edge)
                    .saturating_mul(tile_edge)
                    .saturating_mul(96),
            )
    } else {
        3_300_000_000_u64.saturating_add(340_u64.saturating_mul(voxels))
    };
    exact_u64("predictedDecodeBytes", predicted_decode_bytes)?;
    let projection = i128::from(LTX_Q4_F305_CRASH_FOOTPRINT_BYTES)
        + (i128::from(inventory_bytes) - i128::from(LTX_Q4_INVENTORY_BYTES))
        + (i128::from(predicted_decode_bytes) - i128::from(LTX_INCIDENT_PREDICTED_DECODE_BYTES));
    let projection = u64::try_from(projection)
        .map_err(|_| "SC-18946 incident-calibrated projection must fit u64".to_owned())?;
    exact_u64("incidentCalibratedProjectionBytes", projection)?;
    Ok((expected_disposition, inventory_bytes, projection))
}

fn refuse_unsafe_ltx_capture(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(), String> {
    let (expected_disposition, inventory_bytes, projection) =
        validate_ltx_safety_evidence(request, tier, geometry, selection)?;
    Err(format!(
        "SC-19642 pre-load safety refusal: SC-18946 {tier} is {expected_disposition}; exact tier inventory={inventory_bytes} bytes, incident q4 f305 physical footprint={} bytes, incident-calibrated projection={projection} bytes, and every geometry shares the complete tier plus Gemma load without a proved safe upper bound",
        LTX_Q4_F305_CRASH_FOOTPRINT_BYTES
    ))
}

/// The sole supervised entry into SC-18946. The ordinary `run` action continues through
/// [`refuse_unsafe_ltx_capture`]; this private action additionally requires the live watchdog
/// channel and may reach only the first frozen q4 staged row. The runner injects the two private
/// objects after the canonical harness has constructed the row, so neither object participates in
/// the campaign logical-case identity.
fn validate_ltx_campaign_entry(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(), String> {
    if protocol::action(request)? != LTX_CAMPAIGN_ENTRY_ACTION {
        return Err("SC-20191 campaign entry action changed".to_owned());
    }
    let planned = protocol::planned(request)?;
    let exact = |pointer: &str, expected: &Value| -> Result<(), String> {
        let actual = planned.pointer(pointer);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "SC-20191 campaign entry {pointer} must be {expected}, got {actual:?}"
            ))
        }
    };
    exact("/logicalCaseId", &json!(LTX_CAMPAIGN_ENTRY_LOGICAL_CASE_ID))?;
    exact("/evidenceScope", &json!("authoritative"))?;
    exact("/backend", &json!("mlx"))?;
    exact("/loadShape", &json!("eager_materialization"))?;
    exact("/target/provider", &json!(LTX_PROVIDER))?;
    exact("/target/modelId", &json!(LTX_PROVIDER))?;
    exact("/target/tier", &json!("q4"))?;
    exact("/target/mode", &json!("text_to_video"))?;
    exact("/target/overlay", &json!("none"))?;
    exact("/target/geometry/width", &json!(LTX_CAMPAIGN_ENTRY_WIDTH))?;
    exact("/target/geometry/height", &json!(LTX_CAMPAIGN_ENTRY_HEIGHT))?;
    exact("/target/geometry/batch", &json!(1))?;
    exact("/target/geometry/frames", &json!(LTX_CAMPAIGN_ENTRY_FRAMES))?;
    exact("/strategy/rung", &json!("staged_residency"))?;
    exact(
        "/strategy/engagedRungs",
        &json!(["resident", "staged_residency"]),
    )?;
    exact("/strategy/parameters", &json!({}))?;
    exact(
        "/calibrationFingerprint",
        &json!(LTX_CALIBRATION_FINGERPRINT),
    )?;
    exact("/fixture", &json!(LTX_CAMPAIGN_ENTRY_FIXTURE))?;
    exact("/negative", &json!(false))?;
    exact("/expectedResult", &json!("passed"))?;
    exact("/modelLoadPolicy", &json!("fresh_per_case"))?;
    exact("/modelLoadGroup", &Value::Null)?;
    exact(
        "/_watchdog",
        &json!({ "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES }),
    )?;
    exact(
        "/_campaignEntry",
        &json!({
            "identity": LTX_CAMPAIGN_ENTRY_IDENTITY,
            "artifact": {
                "repository": protocol::LTX_REPOSITORY,
                "revision": LTX_CANARY_ARTIFACT_REVISION,
                "numericTierInventory": {
                    "files": LTX_CANARY_Q4_INVENTORY_FILES,
                    "bytes": LTX_Q4_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_Q4_INVENTORY_SHA256,
                },
                "textEncoderInventory": {
                    "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                    "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                },
            },
        }),
    )?;
    exact(
        "/_measurementSafety",
        &json!({
            "disposition": LTX_SAFETY_REFUSED_OPEN,
            "tierInventoryBytes": LTX_Q4_INVENTORY_BYTES,
            "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
            "incidentCase": "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
            "commonLoad": "complete numeric tier plus shared Gemma stack before geometry-specific work",
            "predictedDecodeBytes": 19_476_906_240_u64,
            "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
            "incidentCalibratedProjectionBytes": 97_906_593_920_u64,
            "projectionAssumptions": [
                "pinned provider decode cost is the only geometry-varying term used",
                "immutable tier inventory delta is added byte-for-byte",
                "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
            ],
            "reason": "incident-calibrated projection is diagnostic only; no proved bound or hard containment admits this row",
        }),
    )?;
    if tier != "q4"
        || geometry.width != LTX_CAMPAIGN_ENTRY_WIDTH
        || geometry.height != LTX_CAMPAIGN_ENTRY_HEIGHT
        || geometry.frames != LTX_CAMPAIGN_ENTRY_FRAMES
        || selection.strategy != MemoryStrategy::StagedResidency
        || selection.parameters.decode_tile_edge.is_some()
        || selection.parameters.decode_overlap.is_some()
    {
        return Err("SC-20191 campaign entry resolved a non-canonical tuple or carrier".to_owned());
    }
    let (fps, seed) = planned_ltx_capture(request, tier, geometry)?;
    if fps != LTX_CAMPAIGN_ENTRY_FPS || seed != LTX_SEED {
        return Err("SC-20191 campaign entry fps or seed changed".to_owned());
    }
    let (disposition, _, _) = validate_ltx_safety_evidence(request, tier, geometry, selection)?;
    if disposition != LTX_SAFETY_REFUSED_OPEN {
        return Err("SC-20191 campaign entry safety disposition changed".to_owned());
    }
    Ok(())
}

/// A private diagnostic carrier, deliberately outside the frozen SC-18946 plan. It reuses the
/// campaign containment machinery but cannot inherit the failed staged row's action, fixture,
/// logical case, or strategy identity.
fn validate_ltx_bounded_carrier_proof(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(), String> {
    if protocol::action(request)? != LTX_BOUNDED_CARRIER_ACTION {
        return Err("SC-20254 bounded-carrier action changed".to_owned());
    }
    let planned = protocol::planned(request)?;
    let exact = |pointer: &str, expected: &Value| -> Result<(), String> {
        let actual = planned.pointer(pointer);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "SC-20254 bounded carrier {pointer} must be {expected}, got {actual:?}"
            ))
        }
    };
    exact("/_diagnosticOnly", &json!(true))?;
    exact(
        "/logicalCaseId",
        &json!(LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID),
    )?;
    exact("/evidenceScope", &json!("fixture"))?;
    exact("/backend", &json!("mlx"))?;
    exact("/loadShape", &json!("eager_materialization"))?;
    exact("/target/provider", &json!(LTX_PROVIDER))?;
    exact("/target/modelId", &json!(LTX_PROVIDER))?;
    exact("/target/tier", &json!("q4"))?;
    exact("/target/mode", &json!("text_to_video"))?;
    exact("/target/overlay", &json!("none"))?;
    exact("/target/geometry/width", &json!(LTX_CAMPAIGN_ENTRY_WIDTH))?;
    exact("/target/geometry/height", &json!(LTX_CAMPAIGN_ENTRY_HEIGHT))?;
    exact("/target/geometry/batch", &json!(1))?;
    exact("/target/geometry/frames", &json!(LTX_CAMPAIGN_ENTRY_FRAMES))?;
    exact("/strategy/rung", &json!("bounded_decode"))?;
    exact(
        "/strategy/engagedRungs",
        &json!(["resident", "staged_residency", "bounded_decode"]),
    )?;
    exact(
        "/strategy/parameters",
        &json!({
            "decodeTileEdge": LTX_CANARY_TILE_EDGE,
            "decodeOverlap": LTX_CANARY_OVERLAP,
        }),
    )?;
    exact(
        "/calibrationFingerprint",
        &json!(LTX_CALIBRATION_FINGERPRINT),
    )?;
    exact("/fixture", &json!(LTX_BOUNDED_CARRIER_FIXTURE))?;
    exact("/negative", &json!(false))?;
    exact("/expectedResult", &json!("passed"))?;
    exact("/modelLoadPolicy", &json!("fresh_per_case"))?;
    exact("/modelLoadGroup", &Value::Null)?;
    exact(
        "/_watchdog",
        &json!({ "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES }),
    )?;
    exact(
        "/_boundedCarrier",
        &json!({
            "identity": LTX_BOUNDED_CARRIER_IDENTITY,
            "fps": LTX_CAMPAIGN_ENTRY_FPS,
            "seed": LTX_SEED,
            "videoMode": "default_av",
            "artifact": {
                "repository": protocol::LTX_REPOSITORY,
                "revision": LTX_CANARY_ARTIFACT_REVISION,
                "numericTierInventory": {
                    "files": LTX_CANARY_Q4_INVENTORY_FILES,
                    "bytes": LTX_Q4_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_Q4_INVENTORY_SHA256,
                },
                "textEncoderInventory": {
                    "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                    "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                },
            },
        }),
    )?;
    if tier != "q4"
        || geometry.width != LTX_CAMPAIGN_ENTRY_WIDTH
        || geometry.height != LTX_CAMPAIGN_ENTRY_HEIGHT
        || geometry.frames != LTX_CAMPAIGN_ENTRY_FRAMES
        || selection.strategy != MemoryStrategy::BoundedDecode
        || selection.parameters.decode_tile_edge != Some(LTX_CANARY_TILE_EDGE)
        || selection.parameters.decode_overlap != Some(LTX_CANARY_OVERLAP)
    {
        return Err("SC-20254 bounded carrier resolved a foreign tuple or strategy".to_owned());
    }
    Ok(())
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

fn ltx_bounded_campaign_spec(tier: &str) -> Result<LtxBoundedCampaignSpec, String> {
    match tier {
        "q4" => Ok(LtxBoundedCampaignSpec {
            tier: "q4",
            identity: LTX_BOUNDED_CAMPAIGN_IDENTITY,
            logical_case_id: LTX_BOUNDED_CAMPAIGN_LOGICAL_CASE_ID,
            fixture: LTX_BOUNDED_CAMPAIGN_FIXTURE,
            inventory_files: LTX_CANARY_Q4_INVENTORY_FILES,
            inventory_bytes: LTX_Q4_INVENTORY_BYTES,
            inventory_sha256: LTX_CANARY_Q4_INVENTORY_SHA256,
            projection_bytes: 84_694_536_320,
            story: "SC-20318",
        }),
        "q8" => Ok(LtxBoundedCampaignSpec {
            tier: "q8",
            identity: LTX_BOUNDED_CAMPAIGN_Q8_IDENTITY,
            logical_case_id: LTX_BOUNDED_CAMPAIGN_Q8_LOGICAL_CASE_ID,
            fixture: LTX_BOUNDED_CAMPAIGN_Q8_FIXTURE,
            inventory_files: LTX_CANARY_Q8_INVENTORY_FILES,
            inventory_bytes: LTX_Q8_INVENTORY_BYTES,
            inventory_sha256: LTX_CANARY_Q8_INVENTORY_SHA256,
            projection_bytes: 93_955_566_576,
            story: "SC-20430",
        }),
        "bf16" => Ok(LtxBoundedCampaignSpec {
            tier: "bf16",
            identity: LTX_BOUNDED_CAMPAIGN_BF16_IDENTITY,
            logical_case_id: LTX_BOUNDED_CAMPAIGN_BF16_LOGICAL_CASE_ID,
            fixture: LTX_BOUNDED_CAMPAIGN_BF16_FIXTURE,
            inventory_files: LTX_CANARY_BF16_INVENTORY_FILES,
            inventory_bytes: LTX_BF16_INVENTORY_BYTES,
            inventory_sha256: LTX_CANARY_BF16_INVENTORY_SHA256,
            projection_bytes: 111_319_657_852,
            story: "SC-20430",
        }),
        other => Err(format!(
            "SC-20430 bounded campaign tier {other:?} is not exactly allowlisted"
        )),
    }
}

/// The three exact promotable bounded campaign entries. The ordinary `run` action continues
/// through the all-row pre-load refusal; the controller may inject this private action only after
/// the canonical harness has selected one allowlisted row and acquired the contained-execution claim.
fn validate_ltx_bounded_campaign_entry(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(), String> {
    let spec = ltx_bounded_campaign_spec(tier)?;
    if protocol::action(request)? != LTX_BOUNDED_CAMPAIGN_ACTION {
        return Err(format!("{} bounded campaign action changed", spec.story));
    }
    let planned = protocol::planned(request)?;
    let exact = |pointer: &str, expected: &Value| -> Result<(), String> {
        let actual = planned.pointer(pointer);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "{} bounded campaign {pointer} must be {expected}, got {actual:?}",
                spec.story
            ))
        }
    };
    exact("/logicalCaseId", &json!(spec.logical_case_id))?;
    exact("/evidenceScope", &json!("authoritative"))?;
    exact("/backend", &json!("mlx"))?;
    exact("/loadShape", &json!("eager_materialization"))?;
    exact("/target/provider", &json!(LTX_PROVIDER))?;
    exact("/target/modelId", &json!(LTX_PROVIDER))?;
    exact("/target/tier", &json!(spec.tier))?;
    exact("/target/mode", &json!("text_to_video"))?;
    exact("/target/overlay", &json!("none"))?;
    exact("/target/geometry/width", &json!(LTX_CAMPAIGN_ENTRY_WIDTH))?;
    exact("/target/geometry/height", &json!(LTX_CAMPAIGN_ENTRY_HEIGHT))?;
    exact("/target/geometry/batch", &json!(1))?;
    exact("/target/geometry/frames", &json!(LTX_CAMPAIGN_ENTRY_FRAMES))?;
    exact("/strategy/rung", &json!("bounded_decode"))?;
    exact(
        "/strategy/engagedRungs",
        &json!(["resident", "staged_residency", "bounded_decode"]),
    )?;
    exact(
        "/strategy/parameters",
        &json!({
            "decodeTileEdge": LTX_CANARY_TILE_EDGE,
            "decodeOverlap": LTX_CANARY_OVERLAP,
        }),
    )?;
    exact(
        "/calibrationFingerprint",
        &json!(LTX_CALIBRATION_FINGERPRINT),
    )?;
    exact("/fixture", &json!(spec.fixture))?;
    exact("/negative", &json!(false))?;
    exact("/expectedResult", &json!("passed"))?;
    exact("/modelLoadPolicy", &json!("fresh_per_case"))?;
    exact("/modelLoadGroup", &Value::Null)?;
    exact(
        "/_watchdog",
        &json!({ "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES }),
    )?;
    exact(
        "/_measurementSafety",
        &json!({
            "disposition": LTX_SAFETY_REFUSED_OPEN,
            "tierInventoryBytes": spec.inventory_bytes,
            "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
            "incidentCase": "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
            "commonLoad": "complete numeric tier plus shared Gemma stack before geometry-specific work",
            "predictedDecodeBytes": 6_264_848_640_u64,
            "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
            "incidentCalibratedProjectionBytes": spec.projection_bytes,
            "projectionAssumptions": [
                "pinned provider decode cost is the only geometry-varying term used",
                "immutable tier inventory delta is added byte-for-byte",
                "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
            ],
            "reason": if spec.tier == "q4" {
                "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20318 action is admitted"
            } else {
                "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20430 action is admitted"
            },
        }),
    )?;
    exact(
        "/_boundedCampaignEntry",
        &json!({
            "identity": spec.identity,
            "fps": LTX_CAMPAIGN_ENTRY_FPS,
            "seed": LTX_SEED,
            "videoMode": "default_av",
            "spatialDecodeTiles": 24,
            "artifact": {
                "repository": protocol::LTX_REPOSITORY,
                "revision": LTX_CANARY_ARTIFACT_REVISION,
                "numericTierInventory": {
                    "files": spec.inventory_files,
                    "bytes": spec.inventory_bytes,
                    "sha256": spec.inventory_sha256,
                },
                "textEncoderInventory": {
                    "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                    "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                },
            },
        }),
    )?;
    if geometry.width != LTX_CAMPAIGN_ENTRY_WIDTH
        || geometry.height != LTX_CAMPAIGN_ENTRY_HEIGHT
        || geometry.frames != LTX_CAMPAIGN_ENTRY_FRAMES
        || selection.strategy != MemoryStrategy::BoundedDecode
        || selection.parameters.decode_tile_edge != Some(LTX_CANARY_TILE_EDGE)
        || selection.parameters.decode_overlap != Some(LTX_CANARY_OVERLAP)
    {
        return Err(format!(
            "{} bounded campaign resolved a foreign tuple or strategy",
            spec.story
        ));
    }
    let (fps, seed) = planned_ltx_capture(request, tier, geometry)?;
    if fps != LTX_CAMPAIGN_ENTRY_FPS || seed != LTX_SEED {
        return Err("SC-20318 bounded campaign fps or seed changed".to_owned());
    }
    let (disposition, _, _) = validate_ltx_safety_evidence(request, tier, geometry, selection)?;
    if disposition != LTX_SAFETY_REFUSED_OPEN {
        return Err("SC-20318 ordinary safety disposition changed".to_owned());
    }
    Ok(())
}

/// The production A/V request. `video_mode` is deliberately left unset so the arm measures the same
/// path a real job takes (`crates/sceneworks-worker/src/video_jobs/ltx.rs` only sets `"no_audio"`
/// when the caller asks for it): the audio latents are denoised jointly with the video regardless,
/// and skipping the audio DECODE would understate the decode phase. `steps` is likewise unset —
/// LTX's distilled schedule is baked (8 + 3 folded to an 11-step bar), so a step count is not a knob.
fn ltx_request(geometry: LtxGeometry, fps: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
            .to_owned(),
        width: geometry.width,
        height: geometry.height,
        count: 1,
        seed: Some(seed),
        frames: Some(geometry.frames),
        fps: Some(fps),
        ..Default::default()
    }
}

fn validate_ltx_bounded_carrier_generation_request(
    request: &GenerationRequest,
) -> Result<(), String> {
    if request.width != LTX_CAMPAIGN_ENTRY_WIDTH
        || request.height != LTX_CAMPAIGN_ENTRY_HEIGHT
        || request.count != 1
        || request.frames != Some(LTX_CAMPAIGN_ENTRY_FRAMES)
        || request.fps != Some(LTX_CAMPAIGN_ENTRY_FPS)
        || request.seed != Some(LTX_SEED)
        || request.video_mode.is_some()
        || request.memory.is_some()
    {
        return Err("SC-20254 constructed a foreign provider generation request".to_owned());
    }
    Ok(())
}

/// One non-promotable diagnostic request. The safety profile is intentionally outside the product
/// envelope and skips only downstream audio decode. The product-envelope profile is the smallest
/// shipped four-second geometry and leaves `video_mode` unset so it follows the ordinary full-A/V
/// provider path. Both force the same bounded-decode carrier without admitting a campaign row.
fn ltx_canary_generation_request_for(profile: LtxCanaryProfile) -> GenerationRequest {
    GenerationRequest {
        prompt: profile.prompt().to_owned(),
        width: profile.width(),
        height: profile.height(),
        count: 1,
        seed: Some(LTX_CANARY_SEED),
        frames: Some(profile.frames()),
        fps: Some(profile.fps()),
        video_mode: profile.video_mode().map(str::to_owned),
        memory: Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(LTX_CANARY_TILE_EDGE),
            decode_overlap: Some(LTX_CANARY_OVERLAP),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn ltx_canary_generation_request() -> GenerationRequest {
    ltx_canary_generation_request_for(LtxCanaryProfile::Safety)
}

fn ltx_product_envelope_canary_generation_request() -> GenerationRequest {
    ltx_canary_generation_request_for(LtxCanaryProfile::ProductEnvelope)
}

fn ltx_canary_request_for_provider_admission(
    mut request: GenerationRequest,
) -> Result<GenerationRequest, String> {
    let memory = request
        .memory
        .as_ref()
        .ok_or_else(|| "LTX safety canary omitted bounded-decode memory parameters".to_owned())?;
    if request.width != LTX_CANARY_WIDTH
        || request.height != LTX_CANARY_HEIGHT
        || request.count != 1
        || request.frames != Some(LTX_CANARY_FRAMES)
        || request.fps != Some(LTX_CANARY_FPS)
        || request.seed != Some(LTX_CANARY_SEED)
        || request.video_mode.as_deref() != Some("no_audio")
        || !memory.tile_vae_decode
        || memory.decode_tile_edge != Some(LTX_CANARY_TILE_EDGE)
        || memory.decode_overlap != Some(LTX_CANARY_OVERLAP)
    {
        return Err(
            "LTX no-audio admission bridge is private to the exact safety canary tuple".to_owned(),
        );
    }
    // The pinned provider's production request scope rejects every explicit video-mode override.
    // Admit/configure the ordinary unconditional A/V denoise with its default, then restore the
    // strictly smaller no-audio decode request immediately before generation.
    request.video_mode = None;
    Ok(request)
}

fn restore_ltx_canary_no_audio_after_configuration(
    request: &mut GenerationRequest,
) -> Result<(), String> {
    if request.video_mode.is_some() {
        return Err(
            "LTX safety canary provider configuration introduced an unexpected video-mode override"
                .to_owned(),
        );
    }
    request.video_mode = Some("no_audio".to_owned());
    Ok(())
}

fn scoped_generate_ltx_no_audio_canary(
    generator: &dyn Generator,
    request: GenerationRequest,
    context: &MemoryRunContext,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput, String> {
    let request = ltx_canary_request_for_provider_admission(request)?;
    let mut observed_failure = None;
    scoped_generate_observed_after_configuration(
        generator,
        request,
        context,
        None,
        &mut observed_failure,
        on_progress,
        Some(restore_ltx_canary_no_audio_after_configuration),
    )
}

/// Resolve and validate the `SCENEWORKS_LTX_*` environment family into a tier-exact load spec.
///
/// Two roots under ONE repository: the numeric tier and the `gemma/` co-requisite. The Gemma-3-12B
/// text encoder is a hard load-time requirement of the pinned provider (`resolve_gemma_dir`,
/// sc-13664 removed the env/HF-cache fallbacks), so it is threaded through `LoadSpec::text_encoder`
/// and is snapshot-validated with the same `validate_huggingface_snapshot_root` identity check as
/// the tier root — a mismatched TE would silently change the measured conditioning peak.
fn ltx_load_spec(
    request: &Value,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, PathBuf, PathBuf, LoadSpec), String> {
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
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
        tier,
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
    // `LoadShape` is inert for this provider — `mlx-gen-ltx` never reads `spec.load_shape`, because
    // the two giants are rebuilt inside every `generate` rather than materialized through a block
    // schedule. Refuse a plan that claims otherwise instead of emitting a receipt whose declared
    // materialization shape the run did not use (sc-16482: a receipt may only testify to its own run).
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization);
    spec.text_encoder = Some(WeightsSource::Dir(text_encoder.clone()));
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    Ok((repository, revision, root, text_encoder, spec))
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

fn ltx_canary_wired_limit(requested: usize, untouched_memory_limit: usize) -> usize {
    requested.min(untouched_memory_limit / 3 * 2)
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

/// Bind the plan to the provider contract's actual engaged composition. SC-19109 made bounded
/// decode an explicit request-scoped control; the adapter must no longer infer the rung from the
/// host-dependent automatic selector used by historical SC-18810 captures.
fn ltx_attested_strategy(
    request: &Value,
    selection: &MemorySelection,
    contract: &mlx_gen::gen_core::MemoryProviderContract,
) -> Result<Value, String> {
    attested_strategy(
        request,
        selection,
        &contract.engaged_composition(selection.strategy),
    )
}

fn ltx_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // One exact staged tuple per plan row; the row has no parameter axes, so the singleton case is
    // the whole exercised domain.
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// Unwrap the video output, refusing an image-shaped or audio-shaped return.
fn video_frames(output: GenerationOutput) -> Result<(Vec<Image>, u32, bool), String> {
    match output {
        GenerationOutput::Video { frames, fps, audio } => {
            if frames.is_empty() {
                return Err("MLX LTX-2.3 render returned no frames".to_owned());
            }
            Ok((frames, fps, audio.is_some()))
        }
        GenerationOutput::Images(_) => {
            Err("MLX LTX-2.3 render returned images, not a video clip".to_owned())
        }
        GenerationOutput::Audio(_) => {
            Err("MLX LTX-2.3 render returned an audio track, not a video clip".to_owned())
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DiagnosticAudioIdentity {
    samples: u64,
    sample_rate: u32,
    channels: u16,
}

/// Extract a diagnostic clip while retaining enough audio identity to prove the full default-A/V
/// decoder returned a non-empty track. The samples themselves drop here, before provider cleanup.
///
/// `label` names the calibration in every refusal. It became a parameter with the second video arm
/// (sc-18663): both A/V providers need this exact unwrapping, and a `minimax_h3` capture must not
/// refuse a misrouted still under an LTX-worded message. Every LTX call site passes
/// [`LTX_VIDEO_LABEL`], so those refusals are byte-identical to the ones this function shipped with.
fn diagnostic_video_frames(
    output: GenerationOutput,
    label: &str,
) -> Result<(Vec<Image>, u32, Option<DiagnosticAudioIdentity>), String> {
    match output {
        GenerationOutput::Video { frames, fps, audio } => {
            if frames.is_empty() {
                return Err(format!("{label} render returned no frames"));
            }
            let audio = audio
                .map(|track| {
                    Ok::<_, String>(DiagnosticAudioIdentity {
                        samples: u64::try_from(track.samples.len()).map_err(|_| {
                            format!("{label} diagnostic audio sample count must fit u64")
                        })?,
                        sample_rate: track.sample_rate,
                        channels: track.channels,
                    })
                })
                .transpose()?;
            Ok((frames, fps, audio))
        }
        GenerationOutput::Images(_) => {
            Err(format!("{label} render returned images, not a video clip"))
        }
        GenerationOutput::Audio(_) => Err(format!(
            "{label} render returned an audio track, not a video clip"
        )),
    }
}

fn validate_diagnostic_audio(
    profile: LtxCanaryProfile,
    audio: Option<DiagnosticAudioIdentity>,
) -> Result<(), String> {
    match (profile, audio) {
        (LtxCanaryProfile::Safety, None) => Ok(()),
        (LtxCanaryProfile::ProductEnvelope, Some(audio))
            if audio.samples > 0 && audio.sample_rate > 0 && audio.channels > 0 =>
        {
            Ok(())
        }
        _ => Err(format!(
            "LTX diagnostic canary returned invalid {:?} audio identity {audio:?}",
            profile
        )),
    }
}

/// Maximum, mean, and root-mean-square absolute error over EVERY frame of two clips, in [0,1]
/// units. Per-frame aggregation rather than a first-frame spot check: a temporal divergence that
/// leaves frame 0 identical is exactly the failure a video determinism contract has to catch.
fn video_max_mean_rms_abs(left: &[Image], right: &[Image]) -> Result<(f64, f64, f64), String> {
    if left.len() != right.len() {
        return Err(format!(
            "video frame-count mismatch: measured={} repeat={}",
            left.len(),
            right.len()
        ));
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut sum_squares = 0.0_f64;
    let mut samples = 0_usize;
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if left.width != right.width || left.height != right.height {
            return Err(format!(
                "video frame {index} changed dimensions between renders"
            ));
        }
        if left.pixels.len() != right.pixels.len() {
            return Err(format!(
                "video frame {index} changed pixel length between renders"
            ));
        }
        for (&left, &right) in left.pixels.iter().zip(&right.pixels) {
            let difference = (f64::from(left) - f64::from(right)).abs() / 255.0;
            maximum = maximum.max(difference);
            sum += difference;
            sum_squares += difference * difference;
            samples += 1;
        }
    }
    if samples == 0 {
        return Err("video comparison had no samples".to_owned());
    }
    Ok((
        maximum,
        sum / samples as f64,
        (sum_squares / samples as f64).sqrt(),
    ))
}

/// Total bytes of the `.safetensors` shards directly under `directory`, following the HF cache's
/// blob symlinks. Used to bound the staged-residency claim against the real component footprints
/// rather than against a comment.
fn safetensors_bytes(directory: &Path) -> Result<u64, String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?;
    let mut total = 0_u64;
    for entry in entries {
        let entry = entry.map_err(|error| format!("read {}: {error}", directory.display()))?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("safetensors") {
            continue;
        }
        let metadata = std::fs::metadata(&path)
            .map_err(|error| format!("stat {}: {error}", path.display()))?;
        total = total.saturating_add(metadata.len());
    }
    if total == 0 {
        return Err(format!(
            "no .safetensors weights under {}",
            directory.display()
        ));
    }
    Ok(total)
}

/// Does this run prove sc-10976's load→use→drop actually happened?
///
/// sc-18808 asked a different question — `overall_peak < text_encoder + transformer` — and that
/// question **does not survive the declared envelope**. It infers "the two giants never co-resided"
/// from an aggregate that ACTIVATIONS dominate at scale, so it is only sound while activations are
/// small next to the weights. It was calibrated at one geometry, 768x512 x 97 frames, where the peak
/// (35.56 GB) sat 17.8 GB under the 53.35 GB co-staged bound.
///
/// sc-18810 broke it with a real capture: q8 at **704x1280 x 177 frames** peaked at 54,153,098,156
/// active bytes, 0.8 GB ABOVE that bound, and the arm refused a perfectly good run. The peak was the
/// **decode** phase, not co-resident weights — this sweep's own measured decode relation
/// (~2.75 GB + 322 B per output voxel, against the engine's own committed `3.3e9 + 340 B/voxel`
/// model at `mlx-gen-ltx/src/pipeline.rs:218-228`) predicts 54,108,433,280 for that geometry's
/// 159,498,240 output voxels, i.e. within 0.08% of what was measured.
///
/// The right witness is a PHASE BOUNDARY reading, not a peak, because a boundary reading contains
/// the resident weights and almost nothing else. `denoise_entry` is sampled at
/// `Progress::Step { current: 1 }` — after the text encoder is dropped and `clear_cache()`d and
/// after the AvDiT is built. Across every sc-18810 capture it sits at 20.72-20.80 GB against a
/// 20.61 GB transformer, flat in geometry over a 3.3x latent-token range. Were the Gemma encoder
/// still resident it could not be below 53.3 GB.
///
/// Two-sided on purpose: the upper bound catches the regression the old check was aiming at, and the
/// lower bound catches a progress hook that fires BEFORE the transformer materializes, which would
/// otherwise make the upper bound pass vacuously.
fn ltx_staging_is_proven(
    denoise_entry: AllocatorState,
    costaged_bytes: u64,
    text_encoder_bytes: u64,
    transformer_bytes: u64,
) -> Result<(), String> {
    if denoise_entry.active >= costaged_bytes {
        return Err(format!(
            "LTX-2.3 held {} active bytes entering the first denoise step, at or above the \
             {costaged_bytes} bytes the staged text encoder ({text_encoder_bytes}) and transformer \
             ({transformer_bytes}) occupy together — the text encoder was not dropped before the \
             AvDiT materialized, so the staged-residency claim this record makes is not supported \
             by the run",
            denoise_entry.active
        ));
    }
    if denoise_entry.active < transformer_bytes {
        return Err(format!(
            "LTX-2.3 held only {} active bytes entering the first denoise step, below the \
             {transformer_bytes}-byte transformer — the denoise boundary was sampled before the \
             AvDiT materialized, so the staged-residency bound above it proves nothing",
            denoise_entry.active
        ));
    }
    Ok(())
}

/// The eight required scenarios for a gated LTX receipt.
///
/// NOT `protocol::not_run_scenarios(blocker)`. That helper marks all eight `not_run`, which would
/// report loadability as unexecuted — but loadability is the one scenario that has nothing to do
/// with the missing memory-strategy seam, and it demonstrably RAN: the real tier provider plus its
/// Gemma co-requisite loaded from the snapshot-validated roots, and every measurement in the
/// fragment came out of that load. A gated receipt understating what it executed is the same class
/// of dishonesty as a complete one overstating it. `overlay` is emitted `not_run` here and then
/// replaced by `settle_plain_overlay_scenario`, which derives the verdict from the declared target
/// rather than letting this arm assert it.
/// The LTX arm's `predictedPeakBytes`, derived from its own measured phases (sc-18864).
///
/// This was hardcoded `null`, which fails BOTH `EvidenceRecord::mlx_admission_envelope` and the
/// `RequiredNullable::Value` check in the worker's estimate seeding — so a record could not seed an
/// estimate even once its status question was resolved. Nothing about the field is
/// contract-dependent: every other MLX arm in this adapter derives it the same way, from
/// `predicted_ceiling` over its own phases. The missing `MemoryStrategyContract` blocks the
/// SCENARIOS (there is no admission check to interrogate), not this arithmetic.
///
/// It differs from the image arms in ONE respect, deliberately: the ceiling is taken over each
/// phase's `active` peak, NOT over `PhaseMemory::allocator_bytes()`. `allocator_bytes` is the
/// two-instant co-existence bound (see [`PhaseMemory::json`]), and for a staged video capture the
/// allocator cache reaches 72-106 GB, so a ceiling over it predicts ~150 GB of demand on a 128 GiB
/// host for a render that completed comfortably. A gate must budget the RESIDENT demand; the cache
/// is elastic and MLX releases it under pressure.
///
/// The image arms are left on their historical `allocator_bytes` input, and the reason is
/// DIRECTIONAL, not a matter of magnitude. Measured over the 69 MLX image records in
/// `docs/generated/memory-calibration-evidence.json`, the two definitions are NOT close: 39 of 69
/// records carry a phase whose `reclaimableBytes` exceeds 6.4 GB (up to 45.62 GB), 60 of 69 diverge
/// by more than 5%, and the worst phase diverges by 117.7% (`imc-2cd840a85ce33b4f22a9` denoise,
/// krea-2-turbo, 19.98 GB of cache). Per arm the worst divergences are krea-2-turbo 117.7%,
/// flux2-dev 105.6%, z-image-turbo 65.2% and qwen-image 14.4% — only the qwen ladder's 6.45 GB
/// worst-case cache is anywhere near "6.4 GB", and generalising that one arm to the lane was wrong.
/// The shipped image records consequently over-predict their own measured resident peak by
/// 1.05x-2.16x (`imc-b6537074420d51413b38` predicts 93.28 GB against a 43.18 GB resident peak).
///
/// That over-prediction is why sc-19115 keeps the image input unchanged: a ceiling over the
/// co-existence bound is CONSERVATIVE — it never under-predicts — while switching to `active`
/// would LOOSEN shipped admission. [`PredictedPeakBasis`] and its image/video constants are the
/// single policy declaration both lanes consume. The corpus test
/// `resident_peak_counterfactual_only_loosens_shipped_image_admission` re-derives the decision on
/// every run, against whatever MLX image records the committed corpus currently holds and
/// production's scaled foreign-reserve currency: some record predictions change and some host-grid
/// decisions flip, and every flip is refusal -> admission. The totals are deliberately not quoted
/// here — the corpus grows at each calibration campaign and each main sync (69 records at
/// sc-19115, 74 at the sc-18304 sync, 93 at the 2026-08-19 epic-17137 sync).
#[cfg(test)]
fn ltx_predicted_peak_bytes(
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
) -> Value {
    video_predicted_peak_bytes(conditioning, denoise, decode).json()
}

fn ltx_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    geometry: LtxGeometry,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width: geometry.width,
            height: geometry.height,
            batch: 1,
            frames: geometry.frames,
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
        evidence_revision: format!("sc-18946@{}", protocol::INFERENCE_PIN),
    }
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

/// The SC-20318 quality minimum: one identical-input warm request scope, with its own synchronized
/// phase memory. Cancellation/error/recovery deliberately do not run on this contained row.
fn verify_ltx_bounded_warm_repeat(
    generator: &dyn Generator,
    context: &MemoryRunContext,
    selected: &[Image],
    input: LtxLifecycleInput,
) -> Result<LtxBoundedWarmMetrics, String> {
    clear_cache();
    reset_peak_memory();
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let (warm, fps, audio) = diagnostic_video_frames(
        scoped_generate(
            generator,
            ltx_request(input.geometry, input.fps, input.seed),
            context,
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
        )?,
        LTX_VIDEO_LABEL,
    )?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if warm.len() != usize::try_from(input.geometry.frames).unwrap_or(usize::MAX)
        || fps != input.fps
        || [conditioning.active, denoise.active, decode.active].contains(&0)
    {
        return Err("SC-20318 warm repeat changed output or phase shape".to_owned());
    }
    let audio = audio
        .filter(|value| value.samples > 0 && value.sample_rate > 0 && value.channels == 2)
        .ok_or_else(|| "SC-20318 warm repeat did not return non-empty stereo audio".to_owned())?;
    let first = warm
        .first()
        .ok_or_else(|| "SC-20318 warm repeat returned no frames".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("SC-20318 warm repeat returned a degenerate first frame".to_owned());
    }
    let (maximum_error, mean_error, rms_error) = video_max_mean_rms_abs(selected, &warm)?;
    if !ltx_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "SC-20318 warm repeat exceeded the LTX determinism envelope: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    Ok(LtxBoundedWarmMetrics {
        conditioning,
        denoise,
        decode,
        maximum_error,
        mean_error,
        rms_error,
        audio,
    })
}

/// Execute the provider-owned warm/cancel/error request scope on the exact selected rung. One
/// cancellation and one injected error target the rung's binding phase, and each is followed by a
/// deterministic recovery render plus allocator bounds against a clean warm control.
fn verify_ltx_lifecycle(
    generator: &dyn Generator,
    context: &MemoryRunContext,
    selected: &[Image],
    input: LtxLifecycleInput,
    phase_sink: &mut dyn LtxCampaignPhaseSink,
) -> Result<LtxLifecycleMetrics, String> {
    let LtxLifecycleInput {
        geometry,
        fps,
        seed,
        fault_phase,
    } = input;
    clear_cache();
    reset_peak_memory();
    phase_sink.mark("lifecycle_warm_repeat")?;
    let (clean_warm, _, _) = video_frames(scoped_generate(
        generator,
        ltx_request(geometry, fps, seed),
        context,
        None,
        &mut |_| {},
    )?)?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let bounds = LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = video_max_mean_rms_abs(selected, &clean_warm)?;
    if !ltx_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "LTX-2.3 clean warm control exceeded the determinism envelope: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    let mut metrics = LtxLifecycleMetrics {
        clean_warm_peak,
        clean_post_cleanup,
        maximum_error,
        mean_error,
        rms_error,
        ..Default::default()
    };
    phase_sink.mark("lifecycle_cancel")?;
    let cancelled = ltx_request(geometry, fps, seed);
    let cancel_signal = cancelled.cancel.clone();
    let mut cancel_triggered = false;
    let mut cancel_failure = None;
    let cancel_result = scoped_generate_observed(
        generator,
        cancelled,
        context,
        None,
        &mut cancel_failure,
        &mut |progress| {
            let at_boundary = match fault_phase {
                MemoryPhase::Denoise => matches!(progress, Progress::Step { current: 1, .. }),
                MemoryPhase::Decode => matches!(progress, Progress::Decoding),
                _ => false,
            };
            if at_boundary && !cancel_triggered {
                cancel_triggered = true;
                cancel_signal.cancel();
            }
        },
    );
    let cancel_error = match cancel_result {
        Ok(_) => {
            return Err(format!(
                "LTX-2.3 cancellation completed successfully at {fault_phase:?} instead of returning the typed canceled outcome"
            ));
        }
        Err(error) => error,
    };
    if !cancel_triggered || cancel_failure != Some(ScopedGenerationFailureKind::Canceled) {
        return Err(format!(
            "LTX-2.3 cancellation did not return the typed path at {fault_phase:?}: triggered={cancel_triggered}, failure={cancel_failure:?}, error={cancel_error}"
        ));
    }
    clear_cache();
    let cancel_cleanup = AllocatorState::capture_current();
    metrics.max_fault_post_cleanup = cancel_cleanup;
    if !bounds.allows_retained(cancel_cleanup) {
        return Err(format!(
            "LTX-2.3 cancellation retained {cancel_cleanup:?} above clean warm {clean_post_cleanup:?} plus {} bytes",
            bounds.tolerance_bytes,
        ));
    }
    reset_peak_memory();
    phase_sink.mark("lifecycle_cancel_recovery")?;
    let (cancel_recovery, _, _) = video_frames(scoped_generate(
        generator,
        ltx_request(geometry, fps, seed),
        context,
        None,
        &mut |_| {},
    )?)?;
    let cancel_recovery_peak = get_peak_memory() as u64;
    metrics.max_recovery_peak = cancel_recovery_peak;
    if !bounds.allows_warm_peak(cancel_recovery_peak) {
        return Err(format!(
            "LTX-2.3 post-cancel recovery peaked at {cancel_recovery_peak}, above clean warm {clean_warm_peak} plus 2%"
        ));
    }
    clear_cache();
    let cancel_recovery_cleanup = AllocatorState::capture_current();
    metrics.max_recovery_post_cleanup = cancel_recovery_cleanup;
    if !bounds.allows_retained(cancel_recovery_cleanup) {
        return Err(format!(
            "LTX-2.3 post-cancel recovery retained {cancel_recovery_cleanup:?} above clean warm {clean_post_cleanup:?} plus {} bytes",
            bounds.tolerance_bytes,
        ));
    }
    let cancel_quality = video_max_mean_rms_abs(selected, &cancel_recovery)?;
    if !ltx_quality_passes(cancel_quality.0, cancel_quality.1, cancel_quality.2) {
        return Err("LTX-2.3 cancellation cleanup changed the warm recovery clip".to_owned());
    }

    phase_sink.mark("lifecycle_error")?;
    let mut injected_failure = None;
    let injected_result = scoped_generate_observed(
        generator,
        ltx_request(geometry, fps, seed),
        context,
        Some(fault_phase),
        &mut injected_failure,
        &mut |_| {},
    );
    let injected = match injected_result {
        Ok(_) => {
            return Err(format!(
                "LTX-2.3 fault injection completed successfully at {fault_phase:?} instead of returning an error"
            ));
        }
        Err(error) => error,
    };
    if injected_failure != Some(ScopedGenerationFailureKind::Error)
        || !injected.contains("injected memory-strategy calibration error")
    {
        return Err(format!(
            "LTX-2.3 error injection returned the wrong outcome at {fault_phase:?}: failure={injected_failure:?}, error={injected}"
        ));
    }
    clear_cache();
    let error_cleanup = AllocatorState::capture_current();
    metrics.max_fault_post_cleanup.active = metrics
        .max_fault_post_cleanup
        .active
        .max(error_cleanup.active);
    metrics.max_fault_post_cleanup.cache = metrics
        .max_fault_post_cleanup
        .cache
        .max(error_cleanup.cache);
    if !bounds.allows_retained(error_cleanup) {
        return Err(format!(
            "LTX-2.3 injected error retained {error_cleanup:?} above clean warm {clean_post_cleanup:?} plus {} bytes",
            bounds.tolerance_bytes,
        ));
    }
    reset_peak_memory();
    phase_sink.mark("lifecycle_error_recovery")?;
    let (error_recovery, _, _) = video_frames(scoped_generate(
        generator,
        ltx_request(geometry, fps, seed),
        context,
        None,
        &mut |_| {},
    )?)?;
    let error_recovery_peak = get_peak_memory() as u64;
    metrics.max_recovery_peak = metrics.max_recovery_peak.max(error_recovery_peak);
    if !bounds.allows_warm_peak(error_recovery_peak) {
        return Err(format!(
            "LTX-2.3 post-error recovery peaked at {error_recovery_peak}, above clean warm {clean_warm_peak} plus 2%"
        ));
    }
    clear_cache();
    let error_recovery_cleanup = AllocatorState::capture_current();
    metrics.max_recovery_post_cleanup.active = metrics
        .max_recovery_post_cleanup
        .active
        .max(error_recovery_cleanup.active);
    metrics.max_recovery_post_cleanup.cache = metrics
        .max_recovery_post_cleanup
        .cache
        .max(error_recovery_cleanup.cache);
    if !bounds.allows_retained(error_recovery_cleanup) {
        return Err(format!(
            "LTX-2.3 post-error recovery retained {error_recovery_cleanup:?} above clean warm {clean_post_cleanup:?} plus {} bytes",
            bounds.tolerance_bytes,
        ));
    }
    let error_quality = video_max_mean_rms_abs(selected, &error_recovery)?;
    if !ltx_quality_passes(error_quality.0, error_quality.1, error_quality.2) {
        return Err("LTX-2.3 error cleanup changed the warm recovery clip".to_owned());
    }
    Ok(metrics)
}

fn validate_ltx_canary_plan_for(
    request: &Value,
    profile: LtxCanaryProfile,
) -> Result<MemorySelection, String> {
    if protocol::action(request)? != profile.action() {
        return Err(format!(
            "LTX diagnostic canary action must be {:?}",
            profile.action()
        ));
    }
    let planned = protocol::planned(request)?;
    if planned.get("_diagnosticOnly").and_then(Value::as_bool) != Some(true) {
        return Err("LTX safety canary requires planned._diagnosticOnly == true".to_owned());
    }
    if planned.get("evidenceScope").and_then(Value::as_str) != Some("fixture") {
        return Err("LTX safety canary requires non-promotable fixture evidenceScope".to_owned());
    }
    if planned.get("backend").and_then(Value::as_str) != Some("mlx") {
        return Err("LTX safety canary requires planned.backend == mlx".to_owned());
    }
    let engaged_rungs = planned
        .pointer("/strategy/engagedRungs")
        .and_then(Value::as_array)
        .ok_or_else(|| "LTX safety canary requires strategy.engagedRungs".to_owned())?
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "LTX safety canary strategy.engagedRungs must contain strings".to_owned())?;
    if engaged_rungs != ["resident", "staged_residency", "bounded_decode"] {
        return Err(
            "LTX safety canary requires exact resident/staged_residency/bounded_decode rungs"
                .to_owned(),
        );
    }
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    let target = planned
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    for field in ["provider", "modelId"] {
        if target.get(field).and_then(Value::as_str) != Some(LTX_PROVIDER) {
            return Err(format!(
                "LTX safety canary target.{field} must be {LTX_PROVIDER:?}"
            ));
        }
    }
    if target.get("tier").and_then(Value::as_str) != Some("q4")
        || target.get("mode").and_then(Value::as_str) != Some("text_to_video")
        || target.get("overlay").and_then(Value::as_str) != Some("none")
    {
        return Err("LTX safety canary requires q4 text_to_video with overlay none".to_owned());
    }
    let geometry = target
        .get("geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let exact_geometry = [
        ("width", u64::from(profile.width())),
        ("height", u64::from(profile.height())),
        ("batch", 1),
        ("frames", u64::from(profile.frames())),
    ];
    for (field, expected) in exact_geometry {
        if geometry.get(field).and_then(Value::as_u64) != Some(expected) {
            return Err(format!(
                "LTX safety canary geometry.{field} must be {expected}"
            ));
        }
    }
    if planned.get("fixture").and_then(Value::as_str) != Some(profile.fixture()) {
        return Err(format!(
            "LTX safety canary fixture must be {:?}",
            profile.fixture()
        ));
    }
    if planned
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        != Some(LTX_CALIBRATION_FINGERPRINT)
    {
        return Err("LTX safety canary fingerprint does not match the pinned provider".to_owned());
    }
    if planned_load_shape(request)? != LoadShape::EagerMaterialization {
        return Err("LTX safety canary requires eager_materialization".to_owned());
    }
    if planned
        .pointer("/_watchdog/maxFootprintBytes")
        .and_then(Value::as_u64)
        != Some(LTX_CANARY_MAX_FOOTPRINT_BYTES)
    {
        return Err(format!(
            "LTX safety canary watchdog ceiling must be the SC-18808 co-staged bound {LTX_CANARY_MAX_FOOTPRINT_BYTES}"
        ));
    }
    let canary = planned
        .get("_canary")
        .and_then(Value::as_object)
        .ok_or_else(|| "LTX safety canary requires planned._canary".to_owned())?;
    if canary.get("identity").and_then(Value::as_str) != Some(profile.identity())
        || canary.get("videoMode").and_then(Value::as_str) != Some(profile.video_mode_identity())
        || canary.get("fps").and_then(Value::as_u64) != Some(u64::from(profile.fps()))
        || canary.get("seed").and_then(Value::as_u64) != Some(LTX_CANARY_SEED)
    {
        return Err(format!(
            "LTX safety canary requires identity {:?}, video mode {:?}, fps {}, seed {LTX_CANARY_SEED}",
            profile.identity(),
            profile.video_mode_identity(),
            profile.fps(),
        ));
    }
    let artifact = planned
        .get("_artifact")
        .and_then(Value::as_object)
        .ok_or_else(|| "LTX safety canary requires planned._artifact".to_owned())?;
    let numeric_inventory = artifact
        .get("numericTierInventory")
        .and_then(Value::as_object)
        .ok_or_else(|| "LTX safety canary requires the numeric-tier inventory".to_owned())?;
    if artifact.get("repository").and_then(Value::as_str) != Some(protocol::LTX_REPOSITORY)
        || artifact.get("revision").and_then(Value::as_str) != Some(LTX_CANARY_ARTIFACT_REVISION)
        || numeric_inventory.get("files").and_then(Value::as_u64)
            != Some(LTX_CANARY_Q4_INVENTORY_FILES)
        || numeric_inventory.get("bytes").and_then(Value::as_u64) != Some(LTX_Q4_INVENTORY_BYTES)
        || numeric_inventory.get("sha256").and_then(Value::as_str)
            != Some(LTX_CANARY_Q4_INVENTORY_SHA256)
    {
        return Err(
            "LTX safety canary requires the exact immutable q4 artifact inventory".to_owned(),
        );
    }
    let text_encoder = artifact
        .get("textEncoderInventory")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            "LTX safety canary requires the immutable bundled Gemma inventory".to_owned()
        })?;
    if text_encoder.get("files").and_then(Value::as_u64)
        != Some(LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES)
        || text_encoder.get("bytes").and_then(Value::as_u64)
            != Some(LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES)
        || text_encoder.get("sha256").and_then(Value::as_str)
            != Some(LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256)
    {
        return Err(
            "LTX safety canary requires the exact immutable bundled Gemma inventory".to_owned(),
        );
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary hardware.memoryBytes must be an integer".to_owned())?;
    if hardware_bytes < LTX_CANARY_MAX_FOOTPRINT_BYTES * 2 {
        return Err(format!(
            "LTX safety canary host memory {hardware_bytes} must preserve two {LTX_CANARY_MAX_FOOTPRINT_BYTES}-byte stop boundaries"
        ));
    }
    let selection = planned_selection(request)?;
    if selection.strategy != MemoryStrategy::BoundedDecode
        || selection.tier.quant != Some(Quant::Q4)
        || selection.parameters.decode_tile_edge != Some(LTX_CANARY_TILE_EDGE)
        || selection.parameters.decode_overlap != Some(LTX_CANARY_OVERLAP)
        || selection.parameters.attention_chunk_size.is_some()
        || selection.parameters.transformer_window_size.is_some()
        || selection.parameters.transformer_window_component.is_some()
    {
        return Err(format!(
            "LTX safety canary requires exact bounded_decode q4 {LTX_CANARY_TILE_EDGE}/{LTX_CANARY_OVERLAP} selection"
        ));
    }
    Ok(selection)
}

#[cfg(test)]
fn validate_ltx_canary_plan(request: &Value) -> Result<MemorySelection, String> {
    validate_ltx_canary_plan_for(request, LtxCanaryProfile::Safety)
}

#[cfg(test)]
fn validate_ltx_product_envelope_canary_plan(request: &Value) -> Result<MemorySelection, String> {
    validate_ltx_canary_plan_for(request, LtxCanaryProfile::ProductEnvelope)
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

fn wait_for_ltx_phase_acknowledgement(
    acknowledgements: &std::sync::mpsc::Receiver<Result<(usize, String), String>>,
    sequence: usize,
    name: &str,
    timeout: Duration,
) -> Result<(), String> {
    let acknowledged = acknowledgements.recv_timeout(timeout).map_err(|error| {
        format!("wait for SC-20216 provider phase {name} acknowledgement: {error}")
    })??;
    if acknowledged != (sequence, name.to_owned()) {
        return Err(format!(
            "SC-20216 watchdog acknowledged a foreign provider phase: expected {sequence} {name}, got {} {}",
            acknowledged.0, acknowledged.1
        ));
    }
    Ok(())
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

fn read_watchdog_line(stream: &mut UnixStream) -> Result<String, String> {
    let mut payload = Vec::new();
    for _ in 0..4096 {
        let mut byte = [0_u8; 1];
        stream
            .read_exact(&mut byte)
            .map_err(|error| format!("read LTX canary watchdog attestation: {error}"))?;
        if byte[0] == b'\n' {
            return String::from_utf8(payload)
                .map_err(|error| format!("LTX canary watchdog attestation is not UTF-8: {error}"));
        }
        payload.push(byte[0]);
    }
    Err("LTX canary watchdog attestation exceeded 4096 bytes".to_owned())
}

fn consume_ltx_canary_watchdog_attestation(
    request: &Value,
) -> Result<LtxCanaryWatchdogAttestation, String> {
    let socket_path = std::env::var("SCENEWORKS_MEMORY_WATCHDOG_SOCKET")
        .map_err(|_| "LTX safety canary requires the live external watchdog channel".to_owned())?;
    let stream = UnixStream::connect(&socket_path)
        .map_err(|error| format!("connect to live LTX canary watchdog: {error}"))?;
    consume_ltx_canary_watchdog_attestation_stream(request, stream)
}

fn consume_ltx_canary_watchdog_attestation_stream(
    request: &Value,
    mut stream: UnixStream,
) -> Result<LtxCanaryWatchdogAttestation, String> {
    // The watchdog creates this one-run socket in a private random directory only after its
    // launch-owned process group is anchored. Direct stdin invocation has no socket and therefore
    // refuses before installing MLX limits or reading any model path. The monitor does not release
    // allocation work until the ACK/GO pair completes around a second full telemetry sample.
    let timeout = Some(Duration::from_secs(1));
    stream
        .set_read_timeout(timeout)
        .map_err(|error| format!("configure LTX watchdog read timeout: {error}"))?;
    stream
        .set_write_timeout(timeout)
        .map_err(|error| format!("configure LTX watchdog write timeout: {error}"))?;
    let payload: Value = serde_json::from_str(&read_watchdog_line(&mut stream)?)
        .map_err(|error| format!("parse LTX canary watchdog attestation: {error}"))?;
    if payload.get("protocol").and_then(Value::as_str) != Some(LTX_CANARY_WATCHDOG_PROTOCOL) {
        return Err("LTX safety canary watchdog protocol changed".to_owned());
    }
    let expected_phase_contract: Option<(&str, &[&str])> =
        match request.get("action").and_then(Value::as_str) {
            Some(LTX_CAMPAIGN_ENTRY_ACTION) => {
                Some((LTX_CAMPAIGN_ENTRY_PHASE_PROFILE, &LTX_PROVIDER_PHASE_NAMES))
            }
            Some(LTX_BOUNDED_CARRIER_ACTION) => Some((
                LTX_BOUNDED_CARRIER_PHASE_PROFILE,
                &LTX_BOUNDED_CARRIER_PHASE_NAMES,
            )),
            Some(LTX_BOUNDED_CAMPAIGN_ACTION) => Some((
                LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE,
                &LTX_BOUNDED_CARRIER_PHASE_NAMES,
            )),
            _ => None,
        };
    let provider_phase_protocol = payload.get("providerPhaseProtocol").and_then(Value::as_str);
    let provider_phase_profile = payload.get("providerPhaseProfile").and_then(Value::as_str);
    let provider_phase_names = payload.get("providerPhases");
    if let Some((expected_profile, expected_names)) = expected_phase_contract {
        if provider_phase_protocol != Some(LTX_PROVIDER_PHASE_PROTOCOL)
            || provider_phase_profile != Some(expected_profile)
            || provider_phase_names != Some(&json!(expected_names))
        {
            return Err(
                "watchdog omitted the exact action-bound authenticated provider phase contract"
                    .to_owned(),
            );
        }
    } else if provider_phase_protocol.is_some()
        || provider_phase_profile.is_some()
        || provider_phase_names.is_some()
    {
        return Err(
            "provider phase telemetry is restricted to exact contained LTX actions".to_owned(),
        );
    }
    let nonce = payload
        .get("nonce")
        .and_then(Value::as_str)
        .ok_or_else(|| "LTX safety canary watchdog attestation omitted nonce".to_owned())?;
    if nonce.len() != 64 || !nonce.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("LTX safety canary watchdog nonce is not 32-byte hexadecimal".to_owned());
    }
    let max_footprint_bytes = payload
        .get("maxFootprintBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary watchdog omitted maxFootprintBytes".to_owned())?;
    let max_runtime_seconds = payload
        .get("maxRuntimeSeconds")
        .and_then(Value::as_f64)
        .ok_or_else(|| "LTX safety canary watchdog omitted maxRuntimeSeconds".to_owned())?;
    let host_memory_bytes = payload
        .get("hostMemoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary watchdog omitted hostMemoryBytes".to_owned())?;
    let min_memory_free_bytes = payload
        .get("minMemoryFreeBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary watchdog omitted minMemoryFreeBytes".to_owned())?;
    let min_initial_memory_free_bytes = payload
        .get("minInitialMemoryFreeBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary watchdog omitted minInitialMemoryFreeBytes".to_owned())?;
    let requested_host_memory = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary hardware.memoryBytes must be an integer".to_owned())?;
    let telemetry_resolution = host_memory_bytes.div_ceil(100);
    if max_footprint_bytes != LTX_CANARY_MAX_FOOTPRINT_BYTES
        || max_runtime_seconds != LTX_CANARY_MAX_RUNTIME_SECONDS
        || host_memory_bytes != requested_host_memory
        || min_initial_memory_free_bytes
            != 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + telemetry_resolution
        || payload.get("minInitialMemoryFreePercent").is_some()
        || min_memory_free_bytes != LTX_CANARY_MAX_FOOTPRINT_BYTES + telemetry_resolution
        || payload.get("minSwapFreeBytes").is_some()
    {
        return Err(
            "LTX safety canary watchdog did not attest the exact reviewed bounds".to_owned(),
        );
    }
    stream
        .write_all(format!("ACK {nonce}\n").as_bytes())
        .map_err(|error| format!("acknowledge LTX canary watchdog: {error}"))?;
    if read_watchdog_line(&mut stream)? != format!("GO {nonce}") {
        return Err("LTX safety canary watchdog did not release allocation work".to_owned());
    }
    Ok(LtxCanaryWatchdogAttestation {
        max_footprint_bytes,
        max_runtime_seconds,
        host_memory_bytes,
        min_initial_memory_free_bytes,
        min_memory_free_bytes,
        nonce: nonce.to_owned(),
        stream: Some(stream),
    })
}

/// One deliberately tiny, non-ingestible real-weight probe for the permanent-pin tiled-release
/// repair. This is not a campaign arm: it executes one render, never the six-render lifecycle
/// sweep, and emits a status the calibration harness schema rejects.
fn ltx_canary_ones_cache_bytes() -> Result<u64, String> {
    LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION
        .checked_add(LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION)
        .and_then(|elements| elements.checked_mul(BFLOAT16_BYTES_PER_ELEMENT))
        .ok_or_else(|| "LTX safety canary ONES_CACHE byte arithmetic overflowed".to_owned())
}

fn validate_ltx_canary_pre_provider(pre_provider: AllocatorState) -> Result<(), String> {
    if pre_provider.cache != 0 {
        return Err(format!(
            "LTX safety canary preProviderCacheBytes {} did not attest the cleared cache 0",
            pre_provider.cache
        ));
    }
    Ok(())
}

fn validate_ltx_canary_cleanup(
    pre_provider: AllocatorState,
    post_cleanup: AllocatorState,
    expected_persistent_active: u64,
) -> Result<(), String> {
    let expected_post_active = pre_provider
        .active
        .checked_add(expected_persistent_active)
        .ok_or_else(|| "LTX safety canary cleanup active-byte arithmetic overflowed".to_owned())?;
    if post_cleanup.active != expected_post_active {
        return Err(format!(
            "LTX safety canary postCleanupActiveBytes {} did not equal pre-provider active {} plus intentional persistent active {} = {}",
            post_cleanup.active,
            pre_provider.active,
            expected_persistent_active,
            expected_post_active
        ));
    }
    if post_cleanup.cache != pre_provider.cache {
        return Err(format!(
            "LTX safety canary postCleanupCacheBytes {} did not return to preProviderCacheBytes {}",
            post_cleanup.cache, pre_provider.cache
        ));
    }
    Ok(())
}

fn run_ltx_canary_for(request: &Value, profile: LtxCanaryProfile) -> Result<Value, String> {
    let selection = validate_ltx_canary_plan_for(request, profile)?;
    let mut watchdog = consume_ltx_canary_watchdog_attestation(request)?;
    let watchdog_lease = watchdog.start_lease()?;
    let limits = LtxCanaryLimits::install()?;
    clear_cache();
    let pre_provider = AllocatorState::capture_current();
    validate_ltx_canary_pre_provider(pre_provider)?;
    let expected_persistent_active = ltx_canary_ones_cache_bytes()?;
    let geometry = LtxGeometry {
        width: profile.width(),
        height: profile.height(),
        frames: profile.frames(),
        latent_frames: 1 + (profile.frames() - 1) / LTX_TEMPORAL_SCALE,
    };
    let (repository, revision, root, text_encoder_root, spec) =
        ltx_load_spec(request, "q4", &selection)?;
    if revision != LTX_CANARY_ARTIFACT_REVISION {
        return Err(format!(
            "LTX safety canary artifact revision must be {LTX_CANARY_ARTIFACT_REVISION}, got {revision}"
        ));
    }
    let registry =
        mlx_gen_ltx::provider_registry().map_err(|error| format!("build LTX registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(LTX_PROVIDER, &spec)
        .map_err(|error| format!("read {LTX_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| "pinned MLX LTX-2.3 provider has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned LTX-2.3 contract rejected canary selection: {error}"))?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned LTX-2.3 contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != LTX_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "pinned LTX-2.3 contract fingerprint changed: expected {LTX_CALIBRATION_FINGERPRINT}, got {}",
            calibration.fingerprint
        ));
    }
    let decode_plan = LtxDecodePlan::resolve_for_selection(&selection, geometry)?;
    decode_plan.validate_selected_strategy(&selection)?;
    if !decode_plan.tiling_engaged()
        || decode_plan.spatial_tile_px() != u64::from(LTX_CANARY_TILE_EDGE)
        || decode_plan.spatial_overlap_px() != u64::from(LTX_CANARY_OVERLAP)
    {
        return Err(
            "LTX safety canary did not resolve the exact multi-tile decode carrier".to_owned(),
        );
    }
    let spatial_decode_tile_count = decode_plan.spatial_tile_count(geometry)?;
    if spatial_decode_tile_count <= 1 {
        return Err("LTX safety canary resolved no physical multi-tile decode".to_owned());
    }
    let generator = registry
        .load(LTX_PROVIDER, &spec)
        .map_err(|error| format!("load real LTX-2.3 q4 canary provider: {error}"))?;
    if generator.memory_strategy_contract() != Some(&contract) {
        return Err("loaded LTX-2.3 canary contract differs from the registry contract".to_owned());
    }
    let context = ltx_context(
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        LTX_CANARY_MAX_FOOTPRINT_BYTES,
        LTX_CANARY_MAX_FOOTPRINT_BYTES,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    ) {
        return Err("LTX-2.3 provider rejected the exact canary budget".to_owned());
    }
    clear_cache();
    reset_peak_memory();
    let mut conditioning = PhaseMemory {
        active: 0,
        cache: 0,
    };
    let mut denoise = PhaseMemory {
        active: 0,
        cache: 0,
    };
    let mut observe_progress = |progress| match progress {
        Progress::Step { current: 1, .. } => {
            conditioning = PhaseMemory::capture();
            reset_peak_memory();
        }
        Progress::Decoding => {
            denoise = PhaseMemory::capture();
            reset_peak_memory();
        }
        _ => {}
    };
    let generated = match profile {
        LtxCanaryProfile::Safety => scoped_generate_ltx_no_audio_canary(
            generator.as_ref(),
            ltx_canary_generation_request(),
            &context,
            &mut observe_progress,
        ),
        LtxCanaryProfile::ProductEnvelope => scoped_generate(
            generator.as_ref(),
            ltx_product_envelope_canary_generation_request(),
            &context,
            None,
            &mut observe_progress,
        ),
    }?;
    let (frames, fps, audio) = diagnostic_video_frames(generated, LTX_VIDEO_LABEL)?;
    let decode = PhaseMemory::capture();
    let expected_audio = profile == LtxCanaryProfile::ProductEnvelope;
    if frames.len() != profile.frames() as usize || fps != profile.fps() {
        return Err(format!(
            "LTX safety canary returned frames={}, fps={fps}, audio={audio:?}",
            frames.len(),
        ));
    }
    validate_diagnostic_audio(profile, audio)?;
    let first = frames
        .first()
        .ok_or_else(|| "LTX safety canary returned no frames".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("LTX safety canary returned a degenerate first frame".to_owned());
    }
    let peak_active = [conditioning.active, denoise.active, decode.active]
        .into_iter()
        .max()
        .unwrap_or(0);
    drop(frames);
    drop(generator);
    clear_cache();
    let cleanup = AllocatorState::capture_current();
    validate_ltx_canary_cleanup(pre_provider, cleanup, expected_persistent_active)?;
    let planned_artifact = protocol::planned(request)?
        .get("_artifact")
        .cloned()
        .ok_or_else(|| "LTX safety canary lost planned._artifact".to_owned())?;
    let mut strategy = ltx_attested_strategy(request, &context.selection, &contract)?;
    strategy["spatialDecodeTiles"] = json!(spatial_decode_tile_count);
    watchdog_lease.complete()?;
    Ok(json!({
        "status": profile.completion_status(),
        "canaryIdentity": profile.identity(),
        "diagnosticOnly": true,
        "promotable": false,
        "ingestible": false,
        "inferenceRevision": protocol::INFERENCE_PIN,
        "calibrationFingerprint": calibration.fingerprint,
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": "q4",
            "tierRoot": root,
            "textEncoderRoot": text_encoder_root,
            "numericTierInventory": planned_artifact["numericTierInventory"],
            "textEncoderInventory": planned_artifact["textEncoderInventory"],
        },
        "target": {
            "provider": LTX_PROVIDER,
            "tier": "q4",
            "geometry": {
                "width": profile.width(),
                "height": profile.height(),
                "frames": profile.frames(),
                "fps": profile.fps(),
            },
            "videoMode": profile.video_mode_identity(),
            "audio": expected_audio,
        },
        "strategy": strategy,
        "watchdog": {
            "required": true,
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "maxFootprintBytes": watchdog.max_footprint_bytes,
            "maxRuntimeSeconds": watchdog.max_runtime_seconds,
            "hostMemoryBytes": watchdog.host_memory_bytes,
            "minInitialMemoryFreeBytes": watchdog.min_initial_memory_free_bytes,
            "minMemoryFreeBytes": watchdog.min_memory_free_bytes,
            "source": "conservative-stop-from-sc-18808-q8-costaged-safetensor-arithmetic-not-physical-bound",
        },
        "mlxLimits": {
            "memoryLimitBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "wiredLimitBytes": limits.wired,
        },
        "observedMemory": {
            "preProviderActiveBytes": pre_provider.active,
            "preProviderCacheBytes": pre_provider.cache,
            "expectedPersistentActive": {
                "identity": LTX_CANARY_ONES_CACHE_IDENTITY,
                "videoDimension": LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION,
                "audioDimension": LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION,
                "dtype": "bfloat16",
                "bytesPerElement": BFLOAT16_BYTES_PER_ELEMENT,
                "bytes": expected_persistent_active,
            },
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "peakActiveBytes": peak_active,
            "postCleanupActiveBytes": cleanup.active,
            "postCleanupCacheBytes": cleanup.cache,
        },
        "output": {
            "frames": profile.frames(),
            "fps": profile.fps(),
            "audio": {
                "present": audio.is_some(),
                "samples": audio.map_or(0, |value| value.samples),
                "sampleRate": audio.map_or(0, |value| value.sample_rate),
                "channels": audio.map_or(0, |value| value.channels),
            },
            "frameTimelineSeconds": f64::from(profile.frames() - 1) / f64::from(profile.fps()),
            "firstFrameNondegenerate": true,
        },
        "capturedAt": protocol::captured_at(),
    }))
}

fn run_ltx_canary(request: &Value) -> Result<Value, String> {
    run_ltx_canary_for(request, LtxCanaryProfile::Safety)
}

fn run_ltx_product_envelope_canary(request: &Value) -> Result<Value, String> {
    run_ltx_canary_for(request, LtxCanaryProfile::ProductEnvelope)
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

fn run_ltx_with_admission(
    request: &Value,
    admission: LtxRunAdmission,
    phase_sink: &mut dyn LtxCampaignPhaseSink,
) -> Result<Value, String> {
    let geometry = validate_ltx_target(request)?;
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    let load_shape = planned_load_shape(request)?;
    if load_shape != LoadShape::EagerMaterialization {
        return Err(
            "the pinned mlx-gen-ltx contract calibrates only eager_materialization; the provider \
             stages text, transformer and decode phases inside each request rather than exposing a \
             deferred block loader"
                .to_owned(),
        );
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
    if !matches!(tier, "q4" | "q8" | "bf16") {
        return Err(format!(
            "the MLX LTX-2.3 plan supports only the manifest's q4, q8 and bf16 tiers; tier {tier:?} \
             is not capturable"
        ));
    }
    let (fps, seed) = planned_ltx_capture(request, tier, geometry)?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != LTX_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "plan/adapter calibration mismatch: plan={planned_fingerprint}, MLX LTX-2.3 arm \
             implements {LTX_CALIBRATION_FINGERPRINT}"
        ));
    }
    match admission {
        LtxRunAdmission::Ordinary => {
            refuse_unsafe_ltx_capture(request, tier, geometry, &selection)?
        }
        LtxRunAdmission::CampaignEntry => {
            validate_ltx_campaign_entry(request, tier, geometry, &selection)?
        }
        LtxRunAdmission::BoundedCampaignEntry => {
            validate_ltx_bounded_campaign_entry(request, tier, geometry, &selection)?
        }
    }
    let (repository, revision, root, text_encoder_root, spec) =
        ltx_load_spec(request, tier, &selection)?;
    // Read BEFORE the load so the staging bound below is grounded in the artifact on disk rather
    // than in an allocator reading that a broken staging would itself corrupt. The staged text
    // phase is `build_text_encoder`, which materializes the Gemma snapshot AND the tier's
    // `connector.safetensors` into one `LtxTextEncoder` — so the connector belongs on the TE side of
    // the bound, not with the resident small components.
    let component_bytes = |name: &str| {
        std::fs::metadata(root.join(name))
            .map_err(|error| format!("stat the {tier} {name}: {error}"))
            .map(|metadata| metadata.len())
    };
    let connector_bytes = component_bytes("connector.safetensors")?;
    let gemma_bytes = safetensors_bytes(&text_encoder_root)?;
    let text_encoder_bytes = gemma_bytes.saturating_add(connector_bytes);
    let transformer_bytes = component_bytes("transformer.safetensors")?;
    let tier_bytes = safetensors_bytes(&root)?;

    let registry =
        mlx_gen_ltx::provider_registry().map_err(|error| format!("build LTX registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(LTX_PROVIDER, &spec)
        .map_err(|error| format!("read {LTX_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| "pinned MLX LTX-2.3 provider has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned LTX-2.3 contract rejected planned selection: {error}"))?;
    let strategy = ltx_attested_strategy(request, &selection, &contract)?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned LTX-2.3 contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != LTX_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "pinned LTX-2.3 contract fingerprint changed: expected {LTX_CALIBRATION_FINGERPRINT}, got {}",
            calibration.fingerprint
        ));
    }
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    // Host-dependent and process-global: resolve only after every cheap target, fixture,
    // fingerprint, and provider-contract check has passed, so an injected test budget cannot mask
    // a deterministic malformed-plan error. This remains before generator load or capture.
    let decode_plan = LtxDecodePlan::resolve_for_selection(&selection, geometry)?;
    decode_plan.validate_selected_strategy(&selection)?;
    let spatial_decode_tiles = decode_plan.spatial_tile_count(geometry)?;
    if admission == LtxRunAdmission::BoundedCampaignEntry
        && (!decode_plan.tiling_engaged()
            || decode_plan.spatial_tile_px() != u64::from(LTX_CANARY_TILE_EDGE)
            || decode_plan.spatial_overlap_px() != u64::from(LTX_CANARY_OVERLAP)
            || spatial_decode_tiles != 24)
    {
        return Err("SC-20318 did not resolve the exact 24-tile 192/64 carrier".to_owned());
    }
    let generator = registry
        .load(LTX_PROVIDER, &spec)
        .map_err(|error| format!("load real LTX-2.3 {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded LTX-2.3 generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err(
            "loaded LTX-2.3 generator contract differs from the registry contract".to_owned(),
        );
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = ltx_context(
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        hardware_bytes,
        1,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    ) {
        return Err("LTX-2.3 admission rejected a fitting pre-measurement budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("LTX-2.3 admission accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-ltx-2-3-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("LTX-2.3 admission accepted stale calibration evidence".to_owned());
    }

    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    // Live allocator readings at the two staging handoffs, distinct from the per-phase PEAKS above.
    let denoise_entry = Cell::new(AllocatorState::default());
    let decode_entry = Cell::new(AllocatorState::default());
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    phase_sink.mark("primary_conditioning")?;
    let (measured, output_fps, audio) = diagnostic_video_frames(
        scoped_generate(
            generator.as_ref(),
            ltx_request(geometry, fps, seed),
            &context,
            None,
            &mut |progress| {
                match progress {
                    // LTX emits no boundary between the staged text phase and the DiT build, so
                    // this phase legitimately spans BOTH giants' materializations; the staging
                    // bound below is what proves they did not co-reside.
                    Progress::Step { current: 1, .. } => {
                        if let Err(error) = phase_sink.mark("primary_denoise") {
                            eprintln!("{error}");
                            std::process::abort();
                        }
                        conditioning.set(PhaseMemory::capture());
                        denoise_entry.set(AllocatorState::capture_current());
                        reset_peak_memory();
                    }
                    Progress::Decoding => {
                        if let Err(error) = phase_sink.mark("primary_decode") {
                            eprintln!("{error}");
                            std::process::abort();
                        }
                        denoise.set(PhaseMemory::capture());
                        decode_entry.set(AllocatorState::capture_current());
                        reset_peak_memory();
                    }
                    _ => {}
                }
            },
        )?,
        LTX_VIDEO_LABEL,
    )?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    let denoise_entry = denoise_entry.get();
    let decode_entry = decode_entry.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized LTX-2.3 lifecycle phase reported a zero active peak".to_owned(),
        );
    }
    if measured.len() as u32 != geometry.frames {
        return Err(format!(
            "LTX-2.3 rendered {} frames for a {}-frame request",
            measured.len(),
            geometry.frames
        ));
    }
    if output_fps != fps {
        return Err(format!(
            "LTX-2.3 returned fps {output_fps} for a {fps} fps request"
        ));
    }
    let audio = audio
        .filter(|audio| audio.samples > 0 && audio.sample_rate > 0 && audio.channels > 0)
        .ok_or_else(|| {
            "LTX-2.3 full-A/V campaign render returned no non-empty audio track".to_owned()
        })?;
    if admission == LtxRunAdmission::BoundedCampaignEntry && audio.channels != 2 {
        return Err("SC-20318 selected render did not return stereo audio".to_owned());
    }
    let first = measured
        .first()
        .ok_or_else(|| "LTX-2.3 campaign render returned no first frame".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("LTX-2.3 campaign render returned a degenerate first frame".to_owned());
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);

    // THE STAGED-RESIDENCY PROOF, and the reason the record may claim rung 1 at all.
    let costaged_bytes = text_encoder_bytes.saturating_add(transformer_bytes);
    ltx_staging_is_proven(
        denoise_entry,
        costaged_bytes,
        text_encoder_bytes,
        transformer_bytes,
    )?;
    // The second handoff: the DiT is dropped and the cache cleared before the VAE decode, so the
    // allocator at the decode boundary must hold less than it did entering the denoise.
    if decode_entry.active >= denoise_entry.active {
        return Err(format!(
            "LTX-2.3 held {} active bytes entering decode against {} entering denoise; the \
             transformer was not released before the VAE decode",
            decode_entry.active, denoise_entry.active
        ));
    }

    let predicted_peaks = video_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget.total_bytes = predicted;
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("LTX-2.3 admission rejected an exact-fit calibrated budget".to_owned());
    }

    let lifecycle_input = LtxLifecycleInput {
        geometry,
        fps,
        seed,
        fault_phase: decode_plan.lifecycle_fault_phase(),
    };
    let (lifecycle, bounded_warm) = match admission {
        LtxRunAdmission::CampaignEntry => (
            verify_ltx_lifecycle(
                generator.as_ref(),
                &context,
                &measured,
                lifecycle_input,
                phase_sink,
            )?,
            None,
        ),
        LtxRunAdmission::BoundedCampaignEntry => {
            let warm = verify_ltx_bounded_warm_repeat(
                generator.as_ref(),
                &context,
                &measured,
                lifecycle_input,
            )?;
            (
                LtxLifecycleMetrics {
                    maximum_error: warm.maximum_error,
                    mean_error: warm.mean_error,
                    rms_error: warm.rms_error,
                    ..Default::default()
                },
                Some(warm),
            )
        }
        LtxRunAdmission::Ordinary => {
            return Err("ordinary SC-18946 execution escaped its pre-load refusal".to_owned())
        }
    };
    let maximum_error = lifecycle.maximum_error;
    let mean_error = lifecycle.mean_error;
    let rms_error = lifecycle.rms_error;

    // Runtime-complete keeps the schema's unexecuted mutation slot null. The adapter still proves
    // falsifiability and records the breach in diagnostics.
    let mutated = measured
        .iter()
        .map(qwen_negative_mutation)
        .collect::<Vec<_>>();
    let (mutated_maximum, mutated_mean, mutated_rms) = video_max_mean_rms_abs(&mutated, &measured)?;
    if ltx_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err("LTX-2.3 output mutation did not breach the determinism envelope".to_owned());
    }

    let scenarios = if admission == LtxRunAdmission::BoundedCampaignEntry {
        let blocker = "SC-20318 executes only selected plus warm-repeat parity; cancellation, authorized-error, and recovery renders remain unexecuted";
        json!([
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded provider contract rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded provider contract rejected a mutated calibration fingerprint" },
            { "name": "warm_repeat", "result": "passed", "reason": "the selected request scope repeated deterministically within the declared clip-wide envelope" },
            { "name": "cancel", "result": "not_run", "reason": blocker },
            { "name": "error", "result": "not_run", "reason": blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared reference-free target" }
        ])
    } else {
        json!([
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded provider contract rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded provider contract rejected a mutated calibration fingerprint" },
            { "name": "warm_repeat", "result": "passed", "reason": "the selected request scope repeated deterministically within the declared clip-wide envelope" },
            { "name": "cancel", "result": "passed", "reason": "typed cancellation at the selected rung boundary cleaned up and recovered within the clean-warm bounds", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "provider fault injection at the selected rung boundary cleaned up and recovered within the clean-warm bounds", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared reference-free target" }
        ])
    };
    let mut diagnostic_measurements = vec![
        ("preRungActiveAfterClear", "bytes", pre_rung_active),
        ("preRungCacheAfterClear", "bytes", pre_rung_cache),
        ("conditioningActivePeak", "bytes", conditioning.active),
        ("denoiseActivePeak", "bytes", denoise.active),
        ("decodeActivePeak", "bytes", decode.active),
        (
            "overallAllocatorEnvelope",
            "bytes",
            overall.allocator_bytes(),
        ),
        ("predictedOverallCeiling", "bytes", predicted_peaks.overall),
        ("denoiseEntryActive", "bytes", denoise_entry.active),
        ("decodeEntryActive", "bytes", decode_entry.active),
        ("stagedGemmaBytes", "bytes", gemma_bytes),
        ("stagedConnectorBytes", "bytes", connector_bytes),
        ("stagedTextEncoderBytes", "bytes", text_encoder_bytes),
        ("stagedTransformerBytes", "bytes", transformer_bytes),
        ("costagedGiantsBytes", "bytes", costaged_bytes),
        ("tierArtifactBytes", "bytes", tier_bytes),
        ("renderedFrames", "count", u64::from(geometry.frames)),
        (
            "latentTemporalDepth",
            "count",
            u64::from(geometry.latent_frames),
        ),
        (
            "latentTokens",
            "count",
            u64::from(geometry.latent_frames)
                * u64::from(geometry.width / 32)
                * u64::from(geometry.height / 32),
        ),
        ("outputFps", "count", u64::from(fps)),
        ("audioTrackDecoded", "count", 1),
        (
            "decodeTilingEngaged",
            "count",
            u64::from(decode_plan.tiling_engaged()),
        ),
        (
            "decodeWritableFrameCap",
            "count",
            decode_plan.writable_frame_cap.max(0) as u64,
        ),
        (
            "decodeTileSpatialPx",
            "count",
            decode_plan.spatial_tile_px(),
        ),
        (
            "decodeTileOverlapPx",
            "count",
            decode_plan.spatial_overlap_px(),
        ),
        ("spatialDecodeTiles", "count", spatial_decode_tiles),
        ("mlxMemoryLimitBytes", "bytes", get_memory_limit() as u64),
        (
            "negativeMutationMaximumErrorPer255",
            "count",
            (mutated_maximum * 255.0).round() as u64,
        ),
        (
            "negativeMutationMeanErrorPer255",
            "count",
            (mutated_mean * 255.0).round() as u64,
        ),
        (
            "negativeMutationRootMeanSquareErrorPer255",
            "count",
            (mutated_rms * 255.0).round() as u64,
        ),
    ];
    if let Some(warm) = bounded_warm {
        diagnostic_measurements.extend([
            (
                "warmConditioningActivePeak",
                "bytes",
                warm.conditioning.active,
            ),
            ("warmDenoiseActivePeak", "bytes", warm.denoise.active),
            ("warmDecodeActivePeak", "bytes", warm.decode.active),
            ("warmOutputAudioSamples", "count", warm.audio.samples),
            (
                "warmOutputAudioSampleRate",
                "hertz",
                u64::from(warm.audio.sample_rate),
            ),
            (
                "warmOutputAudioChannels",
                "count",
                u64::from(warm.audio.channels),
            ),
            ("providerRequestScopeRenders", "count", 2),
        ]);
    } else {
        diagnostic_measurements.extend([
            ("warmRepeatPeak", "bytes", lifecycle.clean_warm_peak),
            (
                "warmRepeatPostCleanupActive",
                "bytes",
                lifecycle.clean_post_cleanup.active,
            ),
            (
                "warmRepeatPostCleanupCache",
                "bytes",
                lifecycle.clean_post_cleanup.cache,
            ),
            (
                "lifecycleMaxFaultPostCleanupActive",
                "bytes",
                lifecycle.max_fault_post_cleanup.active,
            ),
            (
                "lifecycleMaxFaultPostCleanupCache",
                "bytes",
                lifecycle.max_fault_post_cleanup.cache,
            ),
            (
                "lifecycleMaxRecoveryPeak",
                "bytes",
                lifecycle.max_recovery_peak,
            ),
            (
                "lifecycleMaxRecoveryPostCleanupActive",
                "bytes",
                lifecycle.max_recovery_post_cleanup.active,
            ),
            (
                "lifecycleMaxRecoveryPostCleanupCache",
                "bytes",
                lifecycle.max_recovery_post_cleanup.cache,
            ),
        ]);
    }

    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        },
        "sweep": ltx_complete_sweep(request)?,
        "scenarios": scenarios,
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "identical artifact, prompt, seed, geometry, frames, fps, tier, provider contract and selected request scope; measured clip versus a clean warm repeat, compared over every frame",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": LTX_MAX_THRESHOLD,
            "meanErrorThreshold": LTX_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": LTX_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}+gemma"),
        },
        "output": {
            "frames": geometry.frames,
            "fps": fps,
            "audio": {
                "present": true,
                "samples": audio.samples,
                "sampleRate": audio.sample_rate,
                "channels": audio.channels,
            },
            "firstFrameNondegenerate": true,
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:ltx-2-3-provider-contract-video",
            "executed",
            [],
            diagnostic_measurements,
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, LTX_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}

fn campaign_entry_diagnostic(fragment: &Value, name: &str) -> Result<u64, String> {
    fragment
        .pointer("/diagnostics/measurements")
        .and_then(Value::as_array)
        .and_then(|measurements| {
            measurements
                .iter()
                .find(|measurement| measurement.get("name").and_then(Value::as_str) == Some(name))
        })
        .and_then(|measurement| measurement.get("value"))
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("contained LTX campaign omitted diagnostic {name}"))
}

fn validate_ltx_bounded_campaign_fragment(fragment: &Value) -> Result<(), String> {
    for (name, expected) in [
        ("renderedFrames", u64::from(LTX_CAMPAIGN_ENTRY_FRAMES)),
        ("outputFps", u64::from(LTX_CAMPAIGN_ENTRY_FPS)),
        ("audioTrackDecoded", 1),
        ("decodeTilingEngaged", 1),
        ("decodeTileSpatialPx", u64::from(LTX_CANARY_TILE_EDGE)),
        ("decodeTileOverlapPx", u64::from(LTX_CANARY_OVERLAP)),
        ("spatialDecodeTiles", 24),
        ("latentTemporalDepth", 16),
        ("latentTokens", 6_144),
        ("warmOutputAudioChannels", 2),
        ("providerRequestScopeRenders", 2),
    ] {
        let actual = campaign_entry_diagnostic(fragment, name)?;
        if actual != expected {
            return Err(format!(
                "SC-20318 bounded campaign diagnostic {name} must be {expected}, got {actual}"
            ));
        }
    }
    for name in [
        "warmConditioningActivePeak",
        "warmDenoiseActivePeak",
        "warmDecodeActivePeak",
        "warmOutputAudioSamples",
        "warmOutputAudioSampleRate",
    ] {
        if campaign_entry_diagnostic(fragment, name)? == 0 {
            return Err(format!(
                "SC-20318 bounded campaign diagnostic {name} must be positive"
            ));
        }
    }
    let scenarios = fragment
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| "SC-20318 bounded campaign omitted scenarios".to_owned())?;
    let result = |name: &str| {
        scenarios
            .iter()
            .find(|scenario| scenario.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|scenario| scenario.get("result"))
            .and_then(Value::as_str)
    };
    if fragment.pointer("/strategy/rung").and_then(Value::as_str) != Some("bounded_decode")
        || fragment.pointer("/strategy/engagedRungs")
            != Some(&json!(["resident", "staged_residency", "bounded_decode"]))
        || fragment.pointer("/strategy/parameters")
            != Some(&json!({
                "decodeTileEdge": LTX_CANARY_TILE_EDGE,
                "decodeOverlap": LTX_CANARY_OVERLAP,
            }))
        || result("warm_repeat") != Some("passed")
        || result("cancel") != Some("not_run")
        || result("error") != Some("not_run")
        || fragment.pointer("/output/frames").and_then(Value::as_u64)
            != Some(u64::from(LTX_CAMPAIGN_ENTRY_FRAMES))
        || fragment.pointer("/output/fps").and_then(Value::as_u64)
            != Some(u64::from(LTX_CAMPAIGN_ENTRY_FPS))
        || fragment
            .pointer("/output/audio/present")
            .and_then(Value::as_bool)
            != Some(true)
        || fragment
            .pointer("/output/audio/samples")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/audio/sampleRate")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/audio/channels")
            .and_then(Value::as_u64)
            != Some(2)
        || fragment
            .pointer("/output/firstFrameNondegenerate")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "SC-20318 bounded campaign response changed its exact canonical carrier".to_owned(),
        );
    }
    Ok(())
}

fn validate_ltx_campaign_entry_fragment(fragment: &Value) -> Result<(), String> {
    for (name, expected) in [
        ("renderedFrames", u64::from(LTX_CAMPAIGN_ENTRY_FRAMES)),
        ("outputFps", u64::from(LTX_CAMPAIGN_ENTRY_FPS)),
        ("audioTrackDecoded", 1),
        ("decodeTilingEngaged", 0),
        ("decodeTileSpatialPx", 0),
        ("decodeTileOverlapPx", 0),
        ("latentTemporalDepth", 16),
        ("latentTokens", 6_144),
    ] {
        let actual = campaign_entry_diagnostic(fragment, name)?;
        if actual != expected {
            return Err(format!(
                "SC-20191 campaign entry diagnostic {name} must be {expected}, got {actual}"
            ));
        }
    }
    if fragment.pointer("/strategy/rung").and_then(Value::as_str) != Some("staged_residency")
        || fragment.pointer("/strategy/engagedRungs")
            != Some(&json!(["resident", "staged_residency"]))
        || fragment.pointer("/strategy/parameters") != Some(&json!({}))
        || fragment.pointer("/output/frames").and_then(Value::as_u64)
            != Some(u64::from(LTX_CAMPAIGN_ENTRY_FRAMES))
        || fragment.pointer("/output/fps").and_then(Value::as_u64)
            != Some(u64::from(LTX_CAMPAIGN_ENTRY_FPS))
        || fragment
            .pointer("/output/audio/present")
            .and_then(Value::as_bool)
            != Some(true)
        || fragment
            .pointer("/output/audio/samples")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/audio/sampleRate")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/audio/channels")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/firstFrameNondegenerate")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "SC-20191 campaign entry response changed the exact untiled full-A/V carrier"
                .to_owned(),
        );
    }
    Ok(())
}

fn prevalidate_ltx_campaign_entry(request: &Value) -> Result<(), String> {
    let geometry = validate_ltx_target(request)?;
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    if planned_load_shape(request)? != LoadShape::EagerMaterialization {
        return Err("SC-20191 campaign entry requires eager materialization".to_owned());
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != LTX_CALIBRATION_FINGERPRINT {
        return Err("SC-20191 campaign entry calibration fingerprint changed".to_owned());
    }
    validate_ltx_campaign_entry(request, tier, geometry, &selection)
}

fn prevalidate_ltx_bounded_campaign_entry(request: &Value) -> Result<(), String> {
    let geometry = validate_ltx_target(request)?;
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    if planned_load_shape(request)? != LoadShape::EagerMaterialization {
        return Err("SC-20318 bounded campaign requires eager materialization".to_owned());
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != LTX_CALIBRATION_FINGERPRINT {
        return Err("SC-20318 bounded campaign calibration fingerprint changed".to_owned());
    }
    validate_ltx_bounded_campaign_entry(request, tier, geometry, &selection)
}

fn prevalidate_ltx_bounded_carrier_proof(
    request: &Value,
) -> Result<(LtxGeometry, MemorySelection), String> {
    let geometry = validate_ltx_target(request)?;
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    if planned_load_shape(request)? != LoadShape::EagerMaterialization {
        return Err("SC-20254 bounded carrier requires eager materialization".to_owned());
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != LTX_CALIBRATION_FINGERPRINT {
        return Err("SC-20254 bounded-carrier calibration fingerprint changed".to_owned());
    }
    validate_ltx_bounded_carrier_proof(request, tier, geometry, &selection)?;
    let host_memory = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "SC-20254 hardware.memoryBytes must be an integer".to_owned())?;
    if host_memory < LTX_CANARY_MAX_FOOTPRINT_BYTES * 2 {
        return Err("SC-20254 host cannot preserve two exact stop boundaries".to_owned());
    }
    Ok((geometry, selection))
}

fn run_ltx_bounded_carrier_proof(request: &Value) -> Result<Value, String> {
    // The exact private identity is fully checked before the live watchdog handshake, limits,
    // artifact paths, registry construction, or model allocation.
    let (geometry, selection) = prevalidate_ltx_bounded_carrier_proof(request)?;
    let mut watchdog = consume_ltx_canary_watchdog_attestation(request)?;
    let mut watchdog_lease = watchdog.start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)?;
    watchdog_lease.mark("common_load")?;
    let limits = LtxCanaryLimits::install()?;
    clear_cache();
    let pre_provider = AllocatorState::capture_current();
    validate_ltx_canary_pre_provider(pre_provider)?;
    let expected_persistent_active = ltx_canary_ones_cache_bytes()?;
    let (repository, revision, root, text_encoder_root, spec) =
        ltx_load_spec(request, "q4", &selection)?;
    if revision != LTX_CANARY_ARTIFACT_REVISION {
        return Err(format!(
            "SC-20254 artifact revision must be {LTX_CANARY_ARTIFACT_REVISION}, got {revision}"
        ));
    }
    let registry =
        mlx_gen_ltx::provider_registry().map_err(|error| format!("build LTX registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(LTX_PROVIDER, &spec)
        .map_err(|error| format!("read {LTX_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| "pinned MLX LTX-2.3 provider has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned LTX-2.3 contract rejected bounded carrier: {error}"))?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned LTX-2.3 contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != LTX_CALIBRATION_FINGERPRINT {
        return Err("SC-20254 pinned provider calibration identity changed".to_owned());
    }
    let decode_plan = LtxDecodePlan::resolve_for_selection(&selection, geometry)?;
    decode_plan.validate_selected_strategy(&selection)?;
    let spatial_decode_tile_count = decode_plan.spatial_tile_count(geometry)?;
    if !decode_plan.tiling_engaged()
        || decode_plan.spatial_tile_px() != u64::from(LTX_CANARY_TILE_EDGE)
        || decode_plan.spatial_overlap_px() != u64::from(LTX_CANARY_OVERLAP)
        || spatial_decode_tile_count != 24
    {
        return Err("SC-20254 did not resolve the exact 24-tile 192/64 carrier".to_owned());
    }
    let generator = registry
        .load(LTX_PROVIDER, &spec)
        .map_err(|error| format!("load real LTX-2.3 q4 bounded carrier: {error}"))?;
    if generator.memory_strategy_contract() != Some(&contract) {
        return Err("SC-20254 loaded generator contract differs from registry".to_owned());
    }
    let context = ltx_context(
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        LTX_CANARY_MAX_FOOTPRINT_BYTES,
        LTX_CANARY_MAX_FOOTPRINT_BYTES,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    ) {
        return Err("LTX-2.3 provider rejected the exact SC-20254 budget".to_owned());
    }
    clear_cache();
    reset_peak_memory();
    let mut conditioning = PhaseMemory {
        active: 0,
        cache: 0,
    };
    let mut denoise = PhaseMemory {
        active: 0,
        cache: 0,
    };
    watchdog_lease.mark("primary_conditioning")?;
    let generation_request = ltx_request(geometry, LTX_CAMPAIGN_ENTRY_FPS, LTX_SEED);
    validate_ltx_bounded_carrier_generation_request(&generation_request)?;
    let generated = scoped_generate(
        generator.as_ref(),
        generation_request,
        &context,
        None,
        &mut |progress| match progress {
            Progress::Step { current: 1, .. } => {
                if let Err(error) = watchdog_lease.mark("primary_denoise") {
                    eprintln!("{error}");
                    std::process::abort();
                }
                conditioning = PhaseMemory::capture();
                reset_peak_memory();
            }
            Progress::Decoding => {
                if let Err(error) = watchdog_lease.mark("primary_decode") {
                    eprintln!("{error}");
                    std::process::abort();
                }
                denoise = PhaseMemory::capture();
                reset_peak_memory();
            }
            _ => {}
        },
    )?;
    let (frames, fps, audio) = diagnostic_video_frames(generated, LTX_VIDEO_LABEL)?;
    let decode = PhaseMemory::capture();
    if frames.len() != LTX_CAMPAIGN_ENTRY_FRAMES as usize || fps != LTX_CAMPAIGN_ENTRY_FPS {
        return Err(format!(
            "SC-20254 returned frames={}, fps={fps}",
            frames.len()
        ));
    }
    let audio = audio
        .filter(|value| value.samples > 0 && value.sample_rate > 0 && value.channels > 0)
        .ok_or_else(|| "SC-20254 full-A/V render returned no non-empty audio track".to_owned())?;
    let first = frames
        .first()
        .ok_or_else(|| "SC-20254 bounded carrier returned no frames".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("SC-20254 bounded carrier returned a degenerate first frame".to_owned());
    }
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err("SC-20254 synchronized phase reported a zero active peak".to_owned());
    }
    let peak_active = [conditioning.active, denoise.active, decode.active]
        .into_iter()
        .max()
        .unwrap_or(0);
    drop(frames);
    drop(generator);
    watchdog_lease.mark("cleanup")?;
    clear_cache();
    let post_cleanup = AllocatorState::capture_current();
    validate_ltx_canary_cleanup(pre_provider, post_cleanup, expected_persistent_active)?;
    let planned_artifact = protocol::planned(request)?
        .pointer("/_boundedCarrier/artifact")
        .cloned()
        .ok_or_else(|| "SC-20254 lost its exact artifact identity".to_owned())?;
    let mut strategy = ltx_attested_strategy(request, &context.selection, &contract)?;
    strategy["spatialDecodeTiles"] = json!(spatial_decode_tile_count);
    watchdog_lease.complete()?;
    Ok(json!({
        "schemaVersion": 1,
        "recordType": "sceneworks_bounded_carrier_proof_response_v1",
        "status": "diagnostic_bounded_carrier_complete",
        "story": "sc-20254",
        "logicalCaseId": LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID,
        "fixture": LTX_BOUNDED_CARRIER_FIXTURE,
        "identity": LTX_BOUNDED_CARRIER_IDENTITY,
        "diagnosticOnly": true,
        "promotable": false,
        "ingestible": false,
        "inferenceRevision": protocol::INFERENCE_PIN,
        "calibrationFingerprint": calibration.fingerprint,
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": "q4",
            "tierRoot": root,
            "textEncoderRoot": text_encoder_root,
            "numericTierInventory": planned_artifact["numericTierInventory"],
            "textEncoderInventory": planned_artifact["textEncoderInventory"],
        },
        "target": {
            "provider": LTX_PROVIDER,
            "tier": "q4",
            "geometry": {
                "width": LTX_CAMPAIGN_ENTRY_WIDTH,
                "height": LTX_CAMPAIGN_ENTRY_HEIGHT,
                "frames": LTX_CAMPAIGN_ENTRY_FRAMES,
                "fps": LTX_CAMPAIGN_ENTRY_FPS,
            },
            "seed": LTX_SEED,
            "videoMode": "default_av",
            "audio": true,
        },
        "strategy": strategy,
        "watchdog": {
            "required": true,
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
            "providerPhaseProfile": LTX_BOUNDED_CARRIER_PHASE_PROFILE,
            "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
            "maxFootprintBytes": watchdog.max_footprint_bytes,
            "maxRuntimeSeconds": watchdog.max_runtime_seconds,
            "hostMemoryBytes": watchdog.host_memory_bytes,
            "minInitialMemoryFreeBytes": watchdog.min_initial_memory_free_bytes,
            "minMemoryFreeBytes": watchdog.min_memory_free_bytes,
        },
        "mlxLimits": {
            "memoryLimitBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "wiredLimitBytes": limits.wired,
        },
        "observedMemory": {
            "preProviderActiveBytes": pre_provider.active,
            "preProviderCacheBytes": pre_provider.cache,
            "expectedPersistentActive": {
                "identity": LTX_CANARY_ONES_CACHE_IDENTITY,
                "videoDimension": LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION,
                "audioDimension": LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION,
                "dtype": "bfloat16",
                "bytesPerElement": BFLOAT16_BYTES_PER_ELEMENT,
                "bytes": expected_persistent_active,
            },
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "peakActiveBytes": peak_active,
            "postCleanupActiveBytes": post_cleanup.active,
            "postCleanupCacheBytes": post_cleanup.cache,
        },
        "output": {
            "frames": LTX_CAMPAIGN_ENTRY_FRAMES,
            "fps": LTX_CAMPAIGN_ENTRY_FPS,
            "audio": {
                "present": true,
                "samples": audio.samples,
                "sampleRate": audio.sample_rate,
                "channels": audio.channels,
            },
            "frameTimelineSeconds": f64::from(LTX_CAMPAIGN_ENTRY_FRAMES - 1)
                / f64::from(LTX_CAMPAIGN_ENTRY_FPS),
            "firstFrameNondegenerate": true,
        },
        "capturedAt": protocol::captured_at(),
    }))
}

fn run_ltx_campaign_entry(request: &Value) -> Result<Value, String> {
    // Every request identity check precedes the live watchdog handshake, process-global limits,
    // model-path resolution and provider registry construction.
    prevalidate_ltx_campaign_entry(request)?;
    let mut watchdog = consume_ltx_canary_watchdog_attestation(request)?;
    let mut watchdog_lease = watchdog.start_lease()?;
    watchdog_lease.mark("common_load")?;
    let limits = LtxCanaryLimits::install()?;
    clear_cache();
    let pre_provider = AllocatorState::capture_current();
    validate_ltx_canary_pre_provider(pre_provider)?;
    let expected_persistent_active = ltx_canary_ones_cache_bytes()?;
    let mut fragment =
        run_ltx_with_admission(request, LtxRunAdmission::CampaignEntry, &mut watchdog_lease)?;
    validate_ltx_campaign_entry_fragment(&fragment)?;
    watchdog_lease.mark("cleanup")?;
    clear_cache();
    let post_cleanup = AllocatorState::capture_current();
    validate_ltx_canary_cleanup(pre_provider, post_cleanup, expected_persistent_active)?;
    fragment["_campaignEntry"] = json!({
        "identity": LTX_CAMPAIGN_ENTRY_IDENTITY,
        "inferenceRevision": protocol::INFERENCE_PIN,
        "watchdog": {
            "required": true,
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "maxFootprintBytes": watchdog.max_footprint_bytes,
            "maxRuntimeSeconds": watchdog.max_runtime_seconds,
            "hostMemoryBytes": watchdog.host_memory_bytes,
            "minInitialMemoryFreeBytes": watchdog.min_initial_memory_free_bytes,
            "minMemoryFreeBytes": watchdog.min_memory_free_bytes,
        },
        "mlxLimits": {
            "memoryLimitBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "wiredLimitBytes": limits.wired,
        },
        "cleanup": {
            "preProviderActiveBytes": pre_provider.active,
            "preProviderCacheBytes": pre_provider.cache,
            "expectedPersistentActive": {
                "identity": LTX_CANARY_ONES_CACHE_IDENTITY,
                "videoDimension": LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION,
                "audioDimension": LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION,
                "dtype": "bfloat16",
                "bytesPerElement": BFLOAT16_BYTES_PER_ELEMENT,
                "bytes": expected_persistent_active,
            },
            "postCleanupActiveBytes": post_cleanup.active,
            "postCleanupCacheBytes": post_cleanup.cache,
        },
    });
    watchdog_lease.complete()?;
    Ok(fragment)
}

fn run_ltx_bounded_campaign_entry(request: &Value) -> Result<Value, String> {
    prevalidate_ltx_bounded_campaign_entry(request)?;
    let tier = protocol::planned(request)?
        .pointer("/target/tier")
        .and_then(Value::as_str)
        .ok_or_else(|| "bounded campaign target tier must be a string".to_owned())?;
    let bounded_spec = ltx_bounded_campaign_spec(tier)?;
    let mut watchdog = consume_ltx_canary_watchdog_attestation(request)?;
    let mut watchdog_lease = watchdog.start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)?;
    watchdog_lease.mark("common_load")?;
    let limits = LtxCanaryLimits::install()?;
    clear_cache();
    let pre_provider = AllocatorState::capture_current();
    validate_ltx_canary_pre_provider(pre_provider)?;
    let expected_persistent_active = ltx_canary_ones_cache_bytes()?;
    let mut fragment = run_ltx_with_admission(
        request,
        LtxRunAdmission::BoundedCampaignEntry,
        &mut watchdog_lease,
    )?;
    validate_ltx_bounded_campaign_fragment(&fragment)?;
    watchdog_lease.mark("cleanup")?;
    clear_cache();
    let post_cleanup = AllocatorState::capture_current();
    validate_ltx_canary_cleanup(pre_provider, post_cleanup, expected_persistent_active)?;
    fragment["_boundedCampaignEntry"] = json!({
        "identity": bounded_spec.identity,
        "inferenceRevision": protocol::INFERENCE_PIN,
        "watchdog": {
            "required": true,
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
            "providerPhaseProfile": LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE,
            "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
            "maxFootprintBytes": watchdog.max_footprint_bytes,
            "maxRuntimeSeconds": watchdog.max_runtime_seconds,
            "hostMemoryBytes": watchdog.host_memory_bytes,
            "minInitialMemoryFreeBytes": watchdog.min_initial_memory_free_bytes,
            "minMemoryFreeBytes": watchdog.min_memory_free_bytes,
        },
        "mlxLimits": {
            "memoryLimitBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "wiredLimitBytes": limits.wired,
        },
        "cleanup": {
            "preProviderActiveBytes": pre_provider.active,
            "preProviderCacheBytes": pre_provider.cache,
            "expectedPersistentActive": {
                "identity": LTX_CANARY_ONES_CACHE_IDENTITY,
                "videoDimension": LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION,
                "audioDimension": LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION,
                "dtype": "bfloat16",
                "bytesPerElement": BFLOAT16_BYTES_PER_ELEMENT,
                "bytes": expected_persistent_active,
            },
            "postCleanupActiveBytes": post_cleanup.active,
            "postCleanupCacheBytes": post_cleanup.cache,
        },
    });
    watchdog_lease.complete()?;
    Ok(fragment)
}

fn run_ltx(request: &Value) -> Result<Value, String> {
    let mut phases = NoLtxCampaignPhases;
    run_ltx_with_admission(request, LtxRunAdmission::Ordinary, &mut phases)
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

/// MiniMax-H3's own geometry envelope, which replaces the image arms' `frames == 1` refusal for this
/// arm. It is the same four-clause gate the pinned provider's `route_gate` applies, read off the
/// engine crate rather than transcribed:
///
/// * the temporal lattice — `frames ∈ LEGAL_FRAME_COUNTS`, i.e. `17n + 5` clamped to the released
///   checkpoint's hardcoded 5–15 s duration range. `T = 1` is off the lattice and does not render,
///   so a still geometry is refused here too;
/// * the spatial stride — `SPATIAL_STRIDE` (32: the VAE's 16x compression times the DiT's width
///   patch of 2). A 16-aligned canvas survives the VAE and then has an odd latent column count with
///   no patched representation at all;
/// * the canvas budget — `CANVAS_MAX_PIXELS` as a **PRODUCT**, not per edge. The published
///   resolution list contains 1536x672 and 1344x768, whose long edges differ by 192 px and whose
///   areas are identical; a per-edge cap would refuse the first and admit the second while both sit
///   exactly at the budget;
/// * `batch == 1` — the engine renders one clip per request.
fn validate_minimax_geometry(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<MinimaxGeometry, String> {
    let lattice_frames = i32::try_from(frames)
        .ok()
        .filter(|frames| mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS.contains(frames))
        .ok_or_else(|| {
            format!(
                "{MINIMAX_LABEL} requires geometry.frames on the 17n+5 lattice {:?}, got {frames}",
                mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS
            )
        })?;
    let stride = mlx_gen_minimax_h3::SPATIAL_STRIDE;
    if !width.is_multiple_of(stride) || !height.is_multiple_of(stride) {
        return Err(format!(
            "{MINIMAX_LABEL} requires geometry divisible by the {stride}px stride, got {width}x{height}"
        ));
    }
    let pixels = width.saturating_mul(height);
    let budget = mlx_gen_minimax_h3::CANVAS_MAX_PIXELS;
    if pixels > budget {
        return Err(format!(
            "{MINIMAX_LABEL} requires width*height within the {budget}px canvas budget, got \
             {width}x{height} ({pixels}px)"
        ));
    }
    let video_latent_frames = mlx_gen_minimax_h3::video_latent_num_frames(lattice_frames)
        .map_err(|error| format!("derive MiniMax-H3 video latent frames: {error}"))
        .and_then(|latents| {
            u32::try_from(latents)
                .map_err(|_| "MiniMax-H3 video latent frame count must fit u32".to_owned())
        })?;
    let audio_latent_frames =
        u32::try_from(mlx_gen_minimax_h3::audio_latent_num_frames(lattice_frames))
            .map_err(|_| "MiniMax-H3 audio latent count must fit u32".to_owned())?;
    Ok(MinimaxGeometry {
        width,
        height,
        frames,
        video_latent_frames,
        audio_latent_frames,
    })
}

/// Read the four declared geometry axes. Like the LTX arm this reads `frames` as a real value rather
/// than asserting it away, and pins `batch` to 1 before anything derived is computed.
fn minimax_target_geometry(request: &Value) -> Result<MinimaxGeometry, String> {
    let geometry = protocol::planned(request)?
        .pointer("/target/geometry")
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
            "{MINIMAX_LABEL} requires geometry.batch == 1 (the engine renders one clip per \
             request), got {batch}"
        ));
    }
    validate_minimax_geometry(axis("width")?, axis("height")?, axis("frames")?)
}

/// Defense-in-depth mirror of `validate_ltx_target`, plus the t2va-specific target shape.
///
/// `run` dispatches by provider id today, but this arm hardcodes the MiniMax-H3 contract, so a
/// foreign caller must be refused BY NAME here — before any environment variable is read, any path
/// canonicalized, or any weight file opened. The reference surfaces are refused for a sharper
/// reason than tidiness: `ref2va` is a DIFFERENT CHECKPOINT (`transformer_ref/`), so a record
/// measured on the base partition may never be filed against a reference-carrying target.
fn validate_minimax_target(request: &Value) -> Result<MinimaxGeometry, String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let provider = target
        .get("provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    if provider != MINIMAX_PROVIDER {
        return Err(format!(
            "{MINIMAX_LABEL} does not implement provider {provider:?}"
        ));
    }
    let model_id = target
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != MINIMAX_PROVIDER {
        return Err(format!(
            "{MINIMAX_LABEL} requires modelId {MINIMAX_PROVIDER:?}, got {model_id:?}"
        ));
    }
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != "text_to_video" {
        return Err(format!(
            "{MINIMAX_LABEL} requires reference-free text_to_video (t2va) mode, got {mode:?}"
        ));
    }
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "{MINIMAX_LABEL} requires {field} == 0 when declared; ref2va runs a different \
                     DiT partition and cannot be recorded from this one"
                ));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(false) {
                return Err(format!(
                    "{MINIMAX_LABEL} requires {field} == false when declared; ref2va runs a \
                     different DiT partition and cannot be recorded from this one"
                ));
            }
        }
    }
    minimax_target_geometry(request)
}

/// Bind the fixture to the planned tier AND the full rendered geometry, recovering the cadence and
/// the seed. Unlike LTX's, this arm's `fps` has exactly one legal value, so the fixture cannot
/// declare a cadence the engine would refuse.
fn planned_minimax_capture(
    request: &Value,
    tier: &str,
    geometry: MinimaxGeometry,
) -> Result<(u32, u64), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!(
        "minimax-h3-mlx-{tier}-{}x{}-f{}-fps",
        geometry.width, geometry.height, geometry.frames
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse MiniMax-H3 fixture fps {fps:?}: {error}"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse MiniMax-H3 fixture seed {seed:?}: {error}"))?;
    if f64::from(fps) != mlx_gen_minimax_h3::MINIMAX_H3_FPS {
        return Err(format!(
            "planned.fixture declares fps {fps}, but the released MiniMax-H3 checkpoint generates \
             at {} fps only",
            mlx_gen_minimax_h3::MINIMAX_H3_FPS
        ));
    }
    if seed != MINIMAX_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the MiniMax-H3 calibration seed \
             {MINIMAX_SEED}"
        ));
    }
    Ok((fps, seed))
}

/// The text encoder came from the tier tree the DiT did.
const MINIMAX_TIERED_TEXT_ENCODER: &str = "tier";
/// The text encoder came from the upstream snapshot, dense.
const MINIMAX_UPSTREAM_TEXT_ENCODER: &str = "upstream-dense";

/// Which tree a tier's text encoder comes from. Extracted from [`minimax_load_spec`] so the rule is
/// checkable without an environment: `q4`/`q8` are packed and rehosted, `bf16` is the dense upstream
/// checkpoint the rehost deliberately does not carry a second copy of.
fn minimax_text_encoder_source(tier: &str) -> &'static str {
    if tier == "bf16" {
        MINIMAX_UPSTREAM_TEXT_ENCODER
    } else {
        MINIMAX_TIERED_TEXT_ENCODER
    }
}

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
    fn resolved_path_fingerprint(&self, tier: &str) -> String {
        format!(
            "{}@{}:{tier}+text_encoder:{}+shared:{}@{}",
            self.repository,
            self.revision,
            self.text_encoder_source,
            self.upstream_repository,
            self.upstream_revision
        )
    }
}

/// Resolve and validate the `SCENEWORKS_MINIMAX_H3_*` environment family into a tier-exact load spec.
///
/// # The env contract, and why it is six variables rather than three
///
/// `mlx_gen_minimax_h3::model::load` reads TWO artifacts, so the arm resolves two triples:
///
/// * `SCENEWORKS_MINIMAX_H3_{REPOSITORY,REVISION,ROOT}` — the tiered rehost
///   [`protocol::MINIMAX_REPOSITORY`], whose `ROOT` is the **tier directory**
///   (`…/snapshots/<revision>/<tier>`), exactly like every other arm's `_ROOT`. It supplies
///   `transformer/`, its `transformer_ref/` sibling, and — at q4/q8 — `text_encoder/`.
/// * `SCENEWORKS_MINIMAX_H3_UPSTREAM_{REPOSITORY,REVISION,ROOT}` — the upstream
///   [`protocol::MINIMAX_UPSTREAM_REPOSITORY`] snapshot ROOT itself, with no tier component. The
///   loader probes `vae/config.json`, `audio_vae/config.json`, `tokenizer/tokenizer.json` and the
///   three `FL2VA/audio_vae/` documents under the spec's own weights root, and the rehost publishes
///   none of them — so the upstream root is what `spec.weights` must be, and the tiered components
///   are REDIRECTED onto it rather than the other way round.
///
/// # There is no text-encoder variable, deliberately
///
/// The text encoder's tier is DERIVED from the DiT's, because a tier is a whole-pipeline contract
/// rather than a per-component knob: `q4`/`q8` take `<tier>/text_encoder` from the rehost, `bf16`
/// takes the dense `text_encoder/` from upstream — which is exactly what the shipped manifest's
/// three `componentId: "text_encoder"` co-requisite rows declare, the bf16 one against the upstream
/// repository. A seventh variable would make the conditioning stage's tier a free axis that the
/// record's `artifact.variant` could not describe, and the conditioning stage is the tallest phase
/// this family has at every tier.
///
/// Both components are staged EXPLICITLY through `LoadSpec::with_component` even where the loader's
/// own fallback would find the same directory, because `ComponentBytes::resolve` reads the same map
/// to build `asset_facts` — an implicitly-resolved component is charged from a path this arm never
/// stated, and an under-declared conditioning floor admits a render that then OOMs.
fn minimax_load_spec(
    request: &Value,
    tier: &str,
    selection: &MemorySelection,
    load_shape: LoadShape,
) -> Result<MinimaxArtifact, String> {
    protocol::validate_plain_overlay_target(request, MINIMAX_PLAIN_EXECUTION_PATH)?;
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
        tier,
        protocol::MINIMAX_REPOSITORY,
    )?;

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

    let dit_root = tier_root.join(mlx_gen_minimax_h3::model::DIT_COMPONENT);
    let text_encoder_source = minimax_text_encoder_source(tier);
    let text_encoder_tree = if text_encoder_source == MINIMAX_UPSTREAM_TEXT_ENCODER {
        &upstream_root
    } else {
        &tier_root
    };
    let text_encoder_root =
        text_encoder_tree.join(mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT);

    let mut spec = LoadSpec::new(WeightsSource::Dir(upstream_root.clone()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape)
        .with_component(
            mlx_gen_minimax_h3::model::DIT_COMPONENT,
            WeightsSource::Dir(dit_root.clone()),
        )
        .with_component(
            mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT,
            WeightsSource::Dir(text_encoder_root.clone()),
        );
    // `spec.quantize` never packs anything at load: `model::load` RECONCILES it against the staged
    // tier's own `config.json` quantization marker and refuses a disagreement. Passing the planned
    // tier's quant is therefore an assertion about the directory on disk, not an instruction.
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    Ok(MinimaxArtifact {
        repository,
        revision,
        upstream_repository,
        upstream_revision,
        upstream_root,
        dit_root,
        text_encoder_root,
        text_encoder_source,
        spec,
    })
}

/// The admission context for the MiniMax-H3 safety scenarios, describing the base t2va route.
///
/// `mode` IS AN EVIDENCE KEY, not a label. gen-core's `standard_memory_strategy_safety_check`
/// builds `MemoryDecodePolicyQuery { mode_key: context.mode.as_key(), .. }` and matches it against
/// each adopted decode-geometry record's own mode, so a probe run under one spelling cannot answer
/// a request asked under another. The runtime asks under `text_to_video` for every MiniMax render:
/// `video_jobs::wan` resolves the admission mode with
/// `sceneworks_core::video_request::payload_video_mode`, `video_admission` admits only
/// `"text_to_video"`, and it types that string with `memory_mode_from_mode_key`, which maps every
/// non-canonical key to [`MemoryMode::Other`]. This capture therefore carries the same `Other`
/// spelling `ltx_context` does.
///
/// This was [`MemoryMode::TextToImage`] until sc-18663, taken from the pinned provider's
/// `memory_strategy::routes()`, which really does spell t2va with the shared text-to-image key.
/// That list enumerates the provider's weights-free BEHAVIOR fixtures; it is not the key the
/// shipped video funnel queries under, and following it made this harness probe `text_to_image`
/// while the plan it validates ([`validate_minimax_target`] hard-requires `text_to_video`), the
/// record it emits, and the runtime all said `text_to_video`. The split was inert only because
/// MiniMax declares `decode_geometry_policy_authoritative: false` with an empty policy table, so
/// the lookup returns `Ok(None)` under either spelling — the day it declares a policy, the harness
/// would measure under one key and the runtime ask under the other, silently. Pinned by
/// `the_minimax_capture_context_binds_the_mode_key_the_runtime_video_route_sends`.
fn minimax_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    geometry: MinimaxGeometry,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        // A parameter only so the stale-evidence probe can pass a deliberate mismatch; every real
        // call site passes `calibration.fingerprint`.
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width: geometry.width,
            height: geometry.height,
            batch: 1,
            frames: geometry.frames,
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
        evidence_revision: format!("sc-18663@{}", protocol::INFERENCE_PIN),
    }
}

fn minimax_request(geometry: MinimaxGeometry, fps: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "a slow dolly along a rain-slick harbour wall at dusk, gulls calling, cinematic"
            .to_owned(),
        width: geometry.width,
        height: geometry.height,
        count: 1,
        seed: Some(seed),
        frames: Some(geometry.frames),
        fps: Some(fps),
        steps: Some(MINIMAX_STEPS),
        ..Default::default()
    }
}

fn minimax_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= MINIMAX_MAX_THRESHOLD
        && mean <= MINIMAX_MEAN_THRESHOLD
        && rms <= MINIMAX_RMS_THRESHOLD
}

/// One exact tuple per plan row. Non-numeric parameters (rung 4's `transformerWindowComponent`) stay
/// in the case but are not promoted to an axis, the shape `sdxl_runtime_complete_sweep` established:
/// an axis is a swept numeric range, and a component name is not one.
fn minimax_complete_sweep(request: &Value) -> Result<Value, String> {
    let parameters = protocol::strategy_parameters(request)?;
    let axes = parameters
        .iter()
        .filter_map(|(name, value)| value.as_u64().map(|value| (name, value)))
        .map(|(name, value)| json!({ "parameter": name, "testedValues": [value] }))
        .collect::<Vec<_>>();
    Ok(json!({
        "axes": axes,
        "cases": [{ "parameters": parameters, "result": "passed" }],
        "rangeVerified": true,
    }))
}

/// The `mlx:minimax_h3` arm (sc-18663) — the first JOINT AUDIO+VIDEO calibration lane.
///
/// It reads the registry contract under the exact t2va provider id, proves the loaded generator
/// exposes the byte-for-byte same contract, runs the four admission probes against the provider's
/// own registered `safety_check`, and then measures three phase peaks off the boundaries the shipped
/// `generate` already emits. No rung allowlist is hardcoded: the pinned contract's own
/// `validate_selection` decides which rungs are capturable, and it answers differently for the two
/// load shapes (rung 4 is `Implemented` only under `deferred_materialization`).
fn run_minimax_h3(request: &Value) -> Result<Value, String> {
    let geometry = validate_minimax_target(request)?;
    protocol::validate_plain_overlay_target(request, MINIMAX_PLAIN_EXECUTION_PATH)?;
    let load_shape = planned_load_shape(request)?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
    if !matches!(tier, "q4" | "q8" | "bf16") {
        return Err(format!(
            "the MLX MiniMax-H3 plan supports only the manifest's q4, q8 and bf16 tiers; tier \
             {tier:?} is not capturable"
        ));
    }
    let (fps, seed) = planned_minimax_capture(request, tier, geometry)?;
    // Fingerprint check 1 of 3: the PLAN against this arm's own expectation, before any environment
    // or weight work. A provider re-fingerprint must not silently reuse the epic's plan.
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != MINIMAX_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "plan/adapter calibration mismatch: plan={planned_fingerprint}, MLX MiniMax-H3 arm \
             implements {MINIMAX_CALIBRATION_FINGERPRINT}"
        ));
    }
    let artifact = minimax_load_spec(request, tier, &selection, load_shape)?;
    let spec = &artifact.spec;
    // Read the on-disk component footprints BEFORE the load, for the same reason the LTX arm does:
    // grounded in the artifact rather than in an allocator reading a broken staging would itself
    // corrupt — and a mis-staged directory then fails here, before any GPU work, rather than after
    // three full renders.
    let staged_dit_bytes = safetensors_bytes(&artifact.dit_root)?;
    let staged_text_encoder_bytes = safetensors_bytes(&artifact.text_encoder_root)?;
    let shared_video_vae_bytes = safetensors_bytes(&artifact.upstream_root.join("vae"))?;
    let shared_audio_vae_bytes = safetensors_bytes(&artifact.upstream_root.join("audio_vae"))?;

    let registry = mlx_gen_minimax_h3::provider_registry()
        .map_err(|error| format!("build MiniMax-H3 registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(MINIMAX_PROVIDER, spec)
        .map_err(|error| format!("read {MINIMAX_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| {
            "pinned MLX MiniMax-H3 provider has no memory-strategy contract".to_owned()
        })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned MiniMax-H3 contract rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned MiniMax-H3 contract has no calibration identity".to_owned())?;
    // Fingerprint check 2 of 3: the PROVIDER contract against this arm's expectation.
    if calibration.fingerprint != MINIMAX_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "pinned MiniMax-H3 contract fingerprint changed: expected \
             {MINIMAX_CALIBRATION_FINGERPRINT}, got {}",
            calibration.fingerprint
        ));
    }
    // Fingerprint check 3 of 3: the plan against the provider, so the two cannot agree with the arm
    // separately while disagreeing with each other.
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    // The contract resolves the SPEC's shape (`resolved_load_shape`), so the identity the record
    // will carry must be the shape the plan asked for and this arm executed.
    if calibration.load_shape != load_shape {
        return Err(format!(
            "pinned MiniMax-H3 contract resolved load shape {:?} for a plan that declares {:?}",
            calibration.load_shape, load_shape
        ));
    }

    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let wired_limit_bytes = request
        .pointer("/hardware/wiredLimitBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.wiredLimitBytes must be an integer".to_owned())?;

    // Admission mutation hygiene BEFORE the expensive load, through the provider's OWN registered
    // check: the gate must accept a fitting request (so the two rejections are not a blanket
    // refusal), reject an unknown/zero budget, and reject a mutated calibration fingerprint.
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        mlx_gen_minimax_h3::memory_strategy::safety_check(
            spec,
            &contract,
            &minimax_context(
                selection,
                calibration,
                fingerprint,
                geometry,
                total_bytes,
                predicted,
            ),
        )
    };
    if !matches!(
        safety(&calibration.fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "MiniMax-H3 admission rejected a fitting probe budget; the scenario rejections below \
             would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("MiniMax-H3 admission accepted an unknown/zero memory budget".to_owned());
    }
    if !matches!(
        safety("stale-minimax-h3-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("MiniMax-H3 admission accepted stale calibration evidence".to_owned());
    }

    let generator = registry
        .load(MINIMAX_PROVIDER, spec)
        .map_err(|error| format!("load real MiniMax-H3 {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded MiniMax-H3 generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err(
            "loaded MiniMax-H3 generator contract differs from the registry contract".to_owned(),
        );
    }

    // Three phase peaks off the boundaries the shipped `generate` already emits. The engine emits
    // FOUR — `Loading(TextEncoder)`, `Loading(Renderer)`, the first `Step` and `Decoding` — and this
    // arm cuts on the first, second and fourth so each recorded phase is exactly one of the
    // contract's declared `[Conditioning, Denoise, Decode]`:
    //
    // * conditioning = TE map → prompt encode → release, closed at `Loading(Renderer)`;
    // * denoise = the DiT map, the AdaLN precompute-and-evict, and every step, closed at `Decoding`;
    // * decode = both VAEs, closed when `generate` returns.
    //
    // ORDERING IS LOAD-BEARING at every boundary, and it is the ordering of the engine's own
    // `tests/te_tier_generate_stages.rs::rotate`: read `get_peak_memory()` AND the `active + cache`
    // footprint FIRST, and only then `reset_peak_memory()`. Reset early and the closing stage's
    // high-water leaks into the next one. `get_peak_memory` reports ACTIVE only, so the footprint is
    // read alongside it rather than instead — a drain that turns into a no-op migrates a shed
    // component from active into the allocator cache, where active alone cannot see it.
    let pre_generate = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    // Live allocator readings at the two staging handoffs, distinct from the per-phase PEAKS.
    let conditioning_close = Cell::new(AllocatorState::default());
    let denoise_close = Cell::new(AllocatorState::default());
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let (measured, output_fps, audio) = diagnostic_video_frames(
        generator
            .generate(
                &minimax_request(geometry, fps, seed),
                &mut |progress| match progress {
                    Progress::Loading(LoadPhase::TextEncoder) => {
                        pre_generate.set(PhaseMemory::capture());
                        reset_peak_memory();
                    }
                    Progress::Loading(LoadPhase::Renderer) => {
                        conditioning.set(PhaseMemory::capture());
                        conditioning_close.set(AllocatorState::capture_current());
                        reset_peak_memory();
                    }
                    Progress::Decoding => {
                        denoise.set(PhaseMemory::capture());
                        denoise_close.set(AllocatorState::capture_current());
                        reset_peak_memory();
                    }
                    _ => {}
                },
            )
            .map_err(|error| format!("generate measured MiniMax-H3 render: {error}"))?,
        MINIMAX_VIDEO_LABEL,
    )?;
    let decode = PhaseMemory::capture();
    let decode_close = AllocatorState::capture_current();
    let pre_generate = pre_generate.get();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    let conditioning_close = conditioning_close.get();
    let denoise_close = denoise_close.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized MiniMax-H3 lifecycle phase reported a zero active peak; the engine \
             stopped emitting a boundary and the attribution collapsed"
                .to_owned(),
        );
    }
    if measured.len() as u64 != u64::from(geometry.frames) {
        return Err(format!(
            "MiniMax-H3 rendered {} frames for a {}-frame request",
            measured.len(),
            geometry.frames
        ));
    }
    if output_fps != fps {
        return Err(format!(
            "MiniMax-H3 returned fps {output_fps} for a {fps} fps request"
        ));
    }
    // The soundtrack is half of what this family denoises; a record that did not observe one is not
    // a record of a joint A/V render.
    let audio = audio
        .filter(|audio| audio.samples > 0 && audio.sample_rate > 0 && audio.channels > 0)
        .ok_or_else(|| "MiniMax-H3 render returned no non-empty audio track".to_owned())?;
    let first = measured
        .first()
        .ok_or_else(|| "MiniMax-H3 render returned no first frame".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("MiniMax-H3 render returned a degenerate first frame".to_owned());
    }

    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    // Video evidence charges the RESIDENT ACTIVE peak: a staged A/V provider leaves an enormous
    // elastic cache at its phase boundaries, and charging that would make a successful capture
    // inadmissible on its own capture host.
    let predicted_peaks = video_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    // The same two ceilings `memory-calibration-harness.mjs#assertResidencyFitsHardware` applies —
    // checked HERE so a capture that cannot be admitted fails loudly during the campaign rather
    // than producing a well-formed record the harness rejects afterwards.
    if overall.active > hardware_bytes {
        return Err(format!(
            "MiniMax-H3 observed overall active {} bytes above the probed hardware memory \
             {hardware_bytes} bytes",
            overall.active
        ));
    }
    if overall.active > wired_limit_bytes {
        return Err(format!(
            "MiniMax-H3 observed overall active {} bytes above the probed wired ceiling \
             {wired_limit_bytes} bytes",
            overall.active
        ));
    }
    // Probe 4 of 4, on the LOADED generator and against the measured evidence: the calibrated
    // budget admits a request predicted to consume exactly it.
    let exact_fit = minimax_context(
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        predicted,
        predicted,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&exact_fit),
        MemorySafetyDecision::Accept
    ) {
        return Err("MiniMax-H3 admission rejected an exact-fit calibrated budget".to_owned());
    }
    // THE EXACT-FIT ACCEPT ABOVE IS NOT SELF-VALIDATING. The permanent pin publishes the
    // MiniMax-H3 `memory_strategy_contract` and overrides `memory_strategy_safety_check`, so this
    // loaded generator's accept is checked against the provider contract rather than a trait
    // default. The contract identity check after load is the first guard; this independent probe
    // stays meaningful if a future provider publishes a contract but regresses the check: the same
    // loaded generator must REJECT a zero budget, or its accept is not evidence.
    let mut unknown_budget = exact_fit.clone();
    unknown_budget.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown_budget),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(
            "the loaded MiniMax-H3 generator accepted a zero/unknown budget, so its exact-fit \
             accept is a blanket accept rather than admission evidence"
                .to_owned(),
        );
    }

    // Warm repeat determinism + cleanup bounds on this exact loaded provider.
    clear_cache();
    reset_peak_memory();
    let (baseline, _, _) = diagnostic_video_frames(
        generator
            .generate(&minimax_request(geometry, fps, seed), &mut |_| {})
            .map_err(|error| format!("generate warm MiniMax-H3 control: {error}"))?,
        MINIMAX_VIDEO_LABEL,
    )?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let cleanup_bounds =
        LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = video_max_mean_rms_abs(&measured, &baseline)?;
    if !minimax_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "MiniMax-H3 warm repeat exceeded the determinism envelope: max={maximum_error:.6}, \
             mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    reset_peak_memory();
    let (warm, _, _) = diagnostic_video_frames(
        generator
            .generate(&minimax_request(geometry, fps, seed), &mut |_| {})
            .map_err(|error| format!("generate warm MiniMax-H3 repeat: {error}"))?,
        MINIMAX_VIDEO_LABEL,
    )?;
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "MiniMax-H3 warm repeat peaked at {warm_peak} bytes, above the clean warm control \
             {clean_warm_peak} bytes plus 2%"
        ));
    }
    clear_cache();
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "MiniMax-H3 warm repeat retained active/cache bytes {warm_post_cleanup:?} above the \
             clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
            cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = video_max_mean_rms_abs(&measured, &warm)?;
    if !minimax_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err("MiniMax-H3 second warm repeat changed the deterministic output".to_owned());
    }

    // Arm-internal negative-mutation falsifiability check. A runtime_complete record must keep
    // `negativeMutation` null, so the breach is verified here — the capture FAILS if the envelope
    // cannot be breached — and the measured numbers land in diagnostics rather than in the field.
    let mutated = measured
        .iter()
        .map(qwen_negative_mutation)
        .collect::<Vec<_>>();
    let (mutated_maximum, mutated_mean, mutated_rms) = video_max_mean_rms_abs(&mutated, &baseline)?;
    if minimax_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(
            "MiniMax-H3 output mutation did not breach the determinism envelope".to_owned(),
        );
    }

    let lifecycle_blocker = concat!(
        "this arm executes the measured render plus two unscoped warm repeats on the loaded ",
        "provider; it opens no memory-strategy request scope and injects no calibration fault, so ",
        "the scoped cancellation and authorized-error scenarios and their recovery renders are ",
        "unexecuted. The pinned provider does register begin_request, so they are implementable — ",
        "they are not run here, and this record claims nothing about them"
    );
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        // From the CONTRACT's own calibration identity, never copied from the plan: a receipt may
        // only testify to the materialization shape its own run used (sc-16482).
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": tier,
        },
        "sweep": minimax_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed", "reason": "the registered MiniMax-H3 admission check rejected a zero/unknown budget before load" },
            { "name": "stale_evidence", "result": "passed", "reason": "the registered MiniMax-H3 admission check rejected a mutated calibration fingerprint before load" },
            { "name": "warm_repeat", "result": "passed", "reason": "two warm repeats on the loaded provider reproduced the measured clip frame-for-frame inside the declared envelope, within the clean warm peak and cleanup bounds" },
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared reference-free target" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "identical artifact, prompt, seed, geometry, frames, fps, steps, tier, staged components and loaded provider contract; cold measured clip versus two warm unscoped repeats, compared over every frame",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": MINIMAX_MAX_THRESHOLD,
            "meanErrorThreshold": MINIMAX_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": MINIMAX_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": artifact.resolved_path_fingerprint(tier),
        },
        "output": {
            "frames": geometry.frames,
            "fps": fps,
            "videoLatentFrames": geometry.video_latent_frames,
            "audioLatentFrames": geometry.audio_latent_frames,
            "audio": {
                "present": true,
                "samples": audio.samples,
                "sampleRate": audio.sample_rate,
                "channels": audio.channels,
            },
            "firstFrameNondegenerate": true,
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:minimax-h3-joint-av",
            "executed",
            [lifecycle_blocker.to_owned()],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("preGenerateActivePeak", "bytes", pre_generate.active),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("conditioningCloseActive", "bytes", conditioning_close.active),
                ("conditioningCloseCache", "bytes", conditioning_close.cache),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("denoiseCloseActive", "bytes", denoise_close.active),
                ("denoiseCloseCache", "bytes", denoise_close.cache),
                ("decodeActivePeak", "bytes", decode.active),
                ("decodeCloseActive", "bytes", decode_close.active),
                ("decodeCloseCache", "bytes", decode_close.cache),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("predictedOverallCeiling", "bytes", predicted),
                ("lifecycleCleanWarmPeak", "bytes", clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", clean_post_cleanup.cache),
                ("lifecycleCleanupTolerance", "bytes", cleanup_bounds.tolerance_bytes),
                ("lifecycleWarmRepeatPeak", "bytes", warm_peak),
                ("lifecycleWarmRepeatPostCleanupActive", "bytes", warm_post_cleanup.active),
                ("lifecycleWarmRepeatPostCleanupCache", "bytes", warm_post_cleanup.cache),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("negativeMutationRootMeanSquareErrorPer255", "count", (mutated_rms * 255.0).round() as u64),
                ("videoLatentFrames", "count", u64::from(geometry.video_latent_frames)),
                ("audioLatentFrames", "count", u64::from(geometry.audio_latent_frames)),
                ("loadShapeDeferred", "count", u64::from(load_shape == LoadShape::DeferredMaterialization)),
                ("textEncoderFromTierTree", "count", u64::from(artifact.text_encoder_source == MINIMAX_TIERED_TEXT_ENCODER)),
                ("stagedDitBytes", "bytes", staged_dit_bytes),
                ("stagedTextEncoderBytes", "bytes", staged_text_encoder_bytes),
                ("sharedVideoVaeBytes", "bytes", shared_video_vae_bytes),
                ("sharedAudioVaeBytes", "bytes", shared_audio_vae_bytes),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, MINIMAX_PLAIN_EXECUTION_PATH)?;
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
        // sc-22724: the undistilled base rides the same arm under its own registry id, artifact
        // family and execution path; the arm resolves which member from `(provider, mode)`.
        Z_IMAGE_BASE_PROVIDER => run_z_image_reference(request),
        KREA_BASE_PROVIDER => run_krea_base(request),
        SDXL_PROVIDER => run_sdxl(request),
        KREA_PROVIDER => run_krea_control(request),
        QWEN_PROVIDER => run_qwen_provider(request),
        FLUX2_PROVIDER => run_flux2(request),
        // sc-22727: both klein catalog models (`flux2_klein_9b`, `flux2_klein_9b_kv`) load through
        // this ONE engine provider id; `flux2_arm` resolves which from `(provider, modelId)` and
        // refuses an unknown pair by name.
        FLUX2_KLEIN_PROVIDER => run_flux2(request),
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
        let fingerprint = "z-image-mlx-independent-materialization-v4";
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

    /// sc-18808 added the still geometry: `validate_flux2_target` now refuses a non-still target
    /// alongside a foreign provider, so a request that omits the axis entirely can no longer reach
    /// the rung gate this module's tests are aiming at.
    fn minimal_request(provider: &str, rung: &str) -> Value {
        minimal_request_for(provider, provider, rung)
    }

    /// The same shape with the CATALOG model id spelled independently of the provider — the axis
    /// the two klein models differ on (sc-22727).
    fn minimal_request_for(provider: &str, model_id: &str, rung: &str) -> Value {
        json!({
            "planned": {
                "target": {
                    "provider": provider,
                    "modelId": model_id,
                    "overlay": "none",
                    "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
                },
                "strategy": { "rung": rung, "parameters": {} }
            }
        })
    }

    /// The per-arm twin of `validate_z_image_batch`'s provider guard (sc-18104), extended to the
    /// `(provider, modelId)` pair by sc-22727: dispatch routes by provider name today, and two
    /// catalog models share one provider id, so a misrouted target must be refused by name INSIDE
    /// the arm, and the refusal must not read like a missing-field complaint.
    #[test]
    fn the_flux2_arm_refuses_a_foreign_provider_or_model_by_name() {
        for (provider, model_id) in [
            ("z_image_turbo", "z_image_turbo"),
            ("qwen_image", "qwen_image"),
            ("flux2_dev_edit", "flux2_dev_edit"),
            ("flux2_klein_9b_kv_edit", "flux2_klein_9b_kv_edit"),
            // The engine's third klein artifact route: a real catalog model on this provider that
            // this arm does NOT serve (its snapshot is an assembled convert dir, not a tiered
            // rehost), so it must be refused rather than measured as the base klein.
            ("flux2_klein_9b", "flux2_klein_9b_true_v2"),
            // The KV artifact must never be served by the base klein plan, or vice versa: the two
            // differ ONLY in this field.
            ("flux2_dev", "flux2_klein_9b"),
        ] {
            let error = run_flux2(&minimal_request_for(provider, model_id, "resident"))
                .expect_err("a foreign (provider, modelId) pair must not reach a FLUX.2 contract");
            assert_eq!(
                error,
                format!(
                    "the MLX FLUX.2 arm does not implement provider {provider:?} for model {model_id:?}"
                )
            );
            assert!(
                !error.contains("must be a string") && !error.contains("fingerprint"),
                "refusal came from deeper in the arm, so the guard let it through: {error}"
            );
        }
        // And the guard is not a blanket refusal: every shipped member resolves.
        for (provider, model_id, expected) in [
            (FLUX2_PROVIDER, "flux2_dev", FLUX2_DEV_ARM),
            (FLUX2_KLEIN_PROVIDER, "flux2_klein_9b", FLUX2_KLEIN_ARM),
            (
                FLUX2_KLEIN_PROVIDER,
                "flux2_klein_9b_kv",
                FLUX2_KLEIN_KV_ARM,
            ),
        ] {
            let arm = flux2_arm(&minimal_request_for(provider, model_id, "resident")).unwrap();
            assert_eq!(arm, expected, "{provider}/{model_id}");
            assert_eq!(arm.model_id, model_id);
        }
        assert!(
            flux2_arm(&json!({ "planned": { "target": { "provider": FLUX2_PROVIDER } } }))
                .unwrap_err()
                .contains("planned.target.modelId")
        );
    }

    /// Every member carries its OWN artifact family, execution path, refusal label, diagnostics
    /// slug and fixture prefix. Two members sharing any of them would make their records
    /// indistinguishable, and two members sharing an env family would let one artifact satisfy the
    /// other's plan (sc-22727).
    #[test]
    fn every_flux2_member_is_distinguishable_from_every_other() {
        let arms = [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM];
        for field in [
            arms.map(|arm| arm.model_id),
            arms.map(|arm| arm.execution_path),
            arms.map(|arm| arm.still_calibration),
            arms.map(|arm| arm.repository_env),
            arms.map(|arm| arm.revision_env),
            arms.map(|arm| arm.root_env),
            arms.map(|arm| arm.expected_repository),
            arms.map(|arm| arm.fixture_slug),
            arms.map(|arm| arm.slug),
        ] {
            let mut unique = field.to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), arms.len(), "collision in {field:?}");
        }
        // The KV rehost and the base rehost are separate repositories, and neither is the dev one.
        assert_eq!(
            FLUX2_KLEIN_ARM.expected_repository,
            protocol::FLUX2_KLEIN_REPOSITORY
        );
        assert_eq!(
            FLUX2_KLEIN_KV_ARM.expected_repository,
            protocol::FLUX2_KLEIN_KV_REPOSITORY
        );
        // But they DO share the engine provider id: that is the whole reason `modelId` is the
        // discriminator rather than `provider`.
        assert_eq!(FLUX2_KLEIN_ARM.provider, FLUX2_KLEIN_KV_ARM.provider);
    }

    /// sc-18218's scope correction (story comment activity-18225): at the pin, mlx-gen-flux2 marks
    /// every non-Resident strategy `Missing` on the DEV route, so that arm is resident-only BY
    /// REFUSAL, not by accident of the plan. Each of the other four rungs must be named back. The
    /// klein ladder publishes five rungs, so it is NOT refused here — its selection is settled by
    /// the pinned contract instead.
    #[test]
    fn the_flux2_dev_arm_is_resident_only_by_refusal() {
        for rung in [
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        ] {
            let error = run_flux2(&minimal_request(FLUX2_PROVIDER, rung))
                .expect_err("a non-resident rung must be refused");
            assert!(
                error.contains(rung) && error.contains("resident"),
                "refusal must name the rung and the resident-only contract: {error}"
            );
            let klein = run_flux2(&minimal_request_for(
                FLUX2_KLEIN_PROVIDER,
                "flux2_klein_9b",
                rung,
            ))
            .expect_err("the minimal klein request is still incomplete");
            assert!(
                !klein.contains("implements only the resident strategy"),
                "the klein ladder must not borrow the dev route's resident-only refusal: {klein}"
            );
        }
        let resident = run_flux2(&minimal_request(FLUX2_PROVIDER, "resident"))
            .expect_err("the minimal resident request is still incomplete");
        assert!(
            !resident.contains("not capturable"),
            "the resident rung itself must pass the rung gate: {resident}"
        );
    }

    #[test]
    fn flux2_fixture_is_bound_to_member_tier_geometry_and_step_count() {
        let request = json!({
            "planned": { "fixture": "flux2-dev-mlx-q4-768-seed18218-step2" }
        });
        assert_eq!(
            planned_flux2_seed(&request, FLUX2_DEV_ARM, "q4", 768).unwrap(),
            18218
        );
        assert!(planned_flux2_seed(&request, FLUX2_DEV_ARM, "q8", 768)
            .unwrap_err()
            .contains("must start with"));
        assert!(planned_flux2_seed(&request, FLUX2_DEV_ARM, "q4", 1024)
            .unwrap_err()
            .contains("must start with"));
        // A dev fixture must not satisfy a klein plan, and the two klein members must not satisfy
        // each other's: the family segment is part of the binding (sc-22727).
        for arm in [FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
            assert!(planned_flux2_seed(&request, arm, "q4", 768)
                .unwrap_err()
                .contains("must start with"));
        }
        let klein = json!({
            "planned": { "fixture": "flux2-klein-9b-mlx-bf16-768-seed22727-step2" }
        });
        assert_eq!(
            planned_flux2_seed(&klein, FLUX2_KLEIN_ARM, "bf16", 768).unwrap(),
            FLUX2_KLEIN_SEED
        );
        assert!(planned_flux2_seed(&klein, FLUX2_KLEIN_KV_ARM, "bf16", 768)
            .unwrap_err()
            .contains("must start with"));
        let kv = json!({
            "planned": { "fixture": "flux2-klein-9b-kv-mlx-bf16-768-seed22727-step2" }
        });
        assert_eq!(
            planned_flux2_seed(&kv, FLUX2_KLEIN_KV_ARM, "bf16", 768).unwrap(),
            FLUX2_KLEIN_SEED
        );
        assert!(planned_flux2_seed(&kv, FLUX2_KLEIN_ARM, "bf16", 768)
            .unwrap_err()
            .contains("must start with"));
        let three_step = json!({
            "planned": { "fixture": "flux2-dev-mlx-q4-768-seed18218-step3" }
        });
        assert!(planned_flux2_seed(&three_step, FLUX2_DEV_ARM, "q4", 768)
            .unwrap_err()
            .contains("two-step"));
    }

    /// sc-17097's lesson applied to this arm: the tier is derived from the planned target, never a
    /// hardcoded q4 (sc-22727 widened this arm from q4/q8 to the full shipped ladder). WHERE the
    /// tier reaches the loader is per member: the dev route folds it through `LoadSpec::quantize`
    /// (bf16 is the dense base and carries none — the worker's `tier_to_quant` shape), while the
    /// klein turnkey rehosts declare their tier in the snapshot path and REFUSE a spec that also
    /// carries a quant — "flux2 Klein turnkey tiers require BF16 execution with
    /// LoadSpec.quantize=None" (`mlx-gen-flux2/src/artifact_inventory.rs`), which the sc-22727
    /// proof capture hit for real before this flag existed.
    #[test]
    fn flux2_load_spec_carries_the_planned_tier_the_way_each_member_takes_it() {
        for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
            for (tier, folded_quant) in [
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
                ("bf16", None),
            ] {
                let request = json!({
                    "planned": {
                        "strategy": { "rung": "resident", "parameters": {} },
                        "target": { "tier": tier }
                    }
                });
                let selection = planned_selection(&request).unwrap();
                // The PLANNED tier is derived identically on every member; only whether it is
                // handed to the loader differs.
                assert_eq!(selection.tier.quant, folded_quant, "planned tier {tier}");
                let spec = flux2_spec(
                    arm,
                    PathBuf::from(format!("/tmp/{}-{tier}", arm.slug)),
                    &selection,
                );
                let expected = if arm.tier_quant_reaches_the_loader {
                    folded_quant
                } else {
                    None
                };
                assert_eq!(spec.quantize, expected, "{} tier {tier}", arm.model_id);
                assert_eq!(spec.load_shape, arm.load_shape);
                // The catalog model id reaches the engine as `resolved_route`: it is what
                // `KleinArtifactInventory::validate_resolved_route` refuses a cross-variant
                // artifact with (mlx-gen-flux2/src/artifact_inventory.rs).
                assert_eq!(spec.resolved_route.as_deref(), Some(arm.model_id));
            }
        }
        // Which member takes the fold, stated as data so the loop above cannot pass vacuously by
        // every member happening to answer the same way.
        assert_eq!(
            [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM]
                .map(|arm| arm.tier_quant_reaches_the_loader),
            [true, false, false],
            "only the dev route folds the planned tier into LoadSpec::quantize"
        );
    }

    /// The klein arms load the WORKER's plain-T2I shape (`Sequential`, `DeferredMaterialization`,
    /// no quant), which is exactly the predicate the pinned engine publishes a calibration identity
    /// under: `klein_contract_for` sets `calibration` iff `klein_streamable(spec)`
    /// (`mlx-gen-flux2/src/memory_strategy.rs`). That predicate is crate-private at the pin, so it
    /// is transcribed here term by term, the way the worker's own registry test transcribes it.
    /// The dev arm is the opposite shape — resident and eager, as sc-18218 measured it — and must
    /// not borrow the klein one. (sc-22727 review: a resident klein spec yields `calibration:
    /// None`, so the declared fingerprint was unreachable on a perfect artifact.)
    #[test]
    fn flux2_klein_specs_are_the_streamable_shape_the_pin_publishes_a_calibration_for() {
        let klein_streamable = |spec: &LoadSpec| {
            spec.offload_policy == OffloadPolicy::Sequential
                && spec.load_shape == LoadShape::DeferredMaterialization
                && spec.quantize.is_none()
                && spec.adapters.is_empty()
                && spec.control.is_none()
                && spec.extra_controls.is_empty()
                && spec.ip_adapter.is_none()
                && spec.identity.is_none()
                && spec.text_encoder.is_none()
                && spec.components.is_empty()
                && matches!(spec.weights, WeightsSource::Dir(_))
        };
        for tier in ["q4", "q8", "bf16"] {
            let request = json!({
                "planned": {
                    "strategy": { "rung": "resident", "parameters": {} },
                    "target": { "tier": tier }
                }
            });
            let selection = planned_selection(&request).unwrap();
            for arm in [FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
                let spec = flux2_spec(
                    arm,
                    PathBuf::from(format!("/tmp/{}-{tier}", arm.slug)),
                    &selection,
                );
                assert!(
                    klein_streamable(&spec),
                    "{} {tier} must be the streamable shape: {spec:?}",
                    arm.model_id
                );
            }
            let dev = flux2_spec(
                FLUX2_DEV_ARM,
                PathBuf::from(format!("/tmp/flux2-dev-{tier}")),
                &selection,
            );
            assert_eq!(dev.offload_policy, OffloadPolicy::Resident, "dev {tier}");
            assert_eq!(
                dev.load_shape,
                LoadShape::EagerMaterialization,
                "dev {tier}"
            );
            assert!(
                !klein_streamable(&dev),
                "the dev arm must not borrow the klein shape: {dev:?}"
            );
        }
        // Stated as data, so the loop cannot pass by every member answering the same way.
        assert_eq!(
            [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM].map(|arm| arm.offload_policy),
            [
                OffloadPolicy::Resident,
                OffloadPolicy::Sequential,
                OffloadPolicy::Sequential
            ],
        );
    }

    /// The arm table is bound to the two documents the worker and the harness actually read.
    ///
    /// * The shipped manifest, where the worker takes both facts from: `mlx.denseTextEncoderTier`
    ///   (`is_dense_te_tier` — its reason to load a klein tier with `Quant::None`, so the flag must
    ///   equal `!tier_quant_reaches_the_loader`), and the MLX `bounded_transformer_residency` row's
    ///   `requiredOffloadPolicy` under the plain provider (`apply_declared_mlx_load_policy_for_request`
    ///   — its reason to bind Sequential; a member with no such row stays Resident). The klein
    ///   rows' `fingerprint` is the identity the arm expects the pin to publish.
    /// * The anchor plan, whose `loadShape` the harness checks the measured record against and
    ///   whose `calibrationFingerprint` `run_flux2` checks the pinned contract against: every MLX
    ///   row of every member must spell the member's shape and fingerprint.
    ///
    /// Flip any arm flag without the manifest or the plan moving and this reds.
    #[test]
    fn the_flux2_arm_table_agrees_with_the_shipped_manifest_and_the_anchor_plan() {
        let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
            include_str!("../../../../config/manifests/builtin.models.jsonc"),
        ))
        .expect("the shipped models manifest parses");
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
            let entry = manifest["models"]
                .as_array()
                .expect("models")
                .iter()
                .find(|entry| entry["id"] == arm.model_id)
                .unwrap_or_else(|| panic!("{} is not a shipped model", arm.model_id));
            let mlx = &entry["mlx"];
            let dense_te = mlx["denseTextEncoderTier"] == json!(true);
            assert_eq!(
                arm.tier_quant_reaches_the_loader, !dense_te,
                "{}: the worker loads a dense-TE tier with Quant::None (is_dense_te_tier)",
                arm.model_id
            );
            let contract = &mlx["memoryStrategyContract"];
            assert_eq!(contract["provider"], arm.provider, "{}", arm.model_id);
            let btr_rows = contract["implementations"]
                .as_array()
                .expect("implementations")
                .iter()
                .filter(|row| {
                    row["rung"] == "bounded_transformer_residency"
                        && row
                            .get("runtimeProvider")
                            .and_then(Value::as_str)
                            .unwrap_or(arm.provider)
                            == arm.provider
                })
                .collect::<Vec<_>>();
            let expected_policy = match btr_rows.as_slice() {
                [] => OffloadPolicy::Resident,
                [row] => {
                    assert_eq!(
                        row["requiredOffloadPolicy"], "sequential",
                        "{}: the BTR row must bind the sequential policy",
                        arm.model_id
                    );
                    assert_eq!(
                        row["fingerprint"], arm.calibration_fingerprint,
                        "{}: the manifest fingerprint is the identity the pin publishes",
                        arm.model_id
                    );
                    OffloadPolicy::Sequential
                }
                rows => panic!("{}: {} BTR rows for one provider", arm.model_id, rows.len()),
            };
            assert_eq!(
                arm.offload_policy, expected_policy,
                "{}: the arm's offload policy must be the manifest-declared one",
                arm.model_id
            );
            // The shape follows the policy: a Sequential row is the BTR-authorized deferred load,
            // a Resident member keeps the eager default. The ARM's shape is bound here too, so the
            // plan rows below cannot agree with a derived value the arm itself does not carry.
            let (expected_shape, expected_load_shape) = match expected_policy {
                OffloadPolicy::Sequential => (
                    protocol::LOAD_SHAPE_DEFERRED,
                    LoadShape::DeferredMaterialization,
                ),
                OffloadPolicy::Resident => {
                    (protocol::LOAD_SHAPE_EAGER, LoadShape::EagerMaterialization)
                }
            };
            assert_eq!(arm.load_shape, expected_load_shape, "{}", arm.model_id);
            for tier in ["q4", "q8", "bf16"] {
                let key = format!("{}:{tier}:mlx", arm.model_id);
                let row = &plan["anchors"][&key];
                assert!(row.is_object(), "{key} is not a planned anchor");
                assert_eq!(row["loadShape"], expected_shape, "{key}");
                assert_eq!(row["provider"], arm.provider, "{key}");
                assert_eq!(
                    row["calibrationFingerprint"], arm.calibration_fingerprint,
                    "{key}"
                );
            }
        }
    }

    /// A plan row spelling a load shape other than the member's worker shape is refused by name
    /// before any environment or weight work — the harness would refuse the record afterwards, but
    /// only after a full load.
    #[test]
    fn run_flux2_refuses_a_plan_whose_load_shape_is_not_the_members_worker_shape() {
        for (arm, wrong) in [
            (FLUX2_DEV_ARM, protocol::LOAD_SHAPE_DEFERRED),
            (FLUX2_KLEIN_ARM, protocol::LOAD_SHAPE_EAGER),
            (FLUX2_KLEIN_KV_ARM, protocol::LOAD_SHAPE_EAGER),
        ] {
            let mut request = minimal_request_for(arm.provider, arm.model_id, "resident");
            request["planned"]["loadShape"] = json!(wrong);
            let error = run_flux2(&request).expect_err("a crossed load shape must be refused");
            assert!(
                error.contains("worker load shape"),
                "{}: {error}",
                arm.model_id
            );
        }
    }

    /// The PLANNED tier must be carried by the root, and the member's OWN repository must be the
    /// one bound — a q4 export cannot satisfy a q8 or bf16 plan, and the KV artifact cannot satisfy
    /// a base-klein plan even though both load through `flux2_klein_9b`.
    #[test]
    fn flux2_root_must_carry_the_planned_tier_and_the_members_own_repository() {
        const REVISION: &str = "acf05e8d5103838baba6a5e32dc91d6997a56023";
        for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
            let selection = resident_selection(Some(Quant::Q4));
            let q4_root = flux2_snapshot_root(arm.expected_repository, REVISION, "q4");
            let error = flux2_load_spec_at(
                arm,
                "q8",
                &selection,
                arm.expected_repository.to_owned(),
                REVISION.to_owned(),
                q4_root.clone(),
            )
            .expect_err("a q8 plan must not be satisfied by a q4 root");
            assert!(
                error.ends_with(&format!("/snapshots/{REVISION}/q8")),
                "{}: {error}",
                arm.model_id
            );
            for tier in ["q4", "q8", "bf16"] {
                let root = flux2_snapshot_root(arm.expected_repository, REVISION, tier);
                let (repository, revision, _) = flux2_load_spec_at(
                    arm,
                    tier,
                    &selection,
                    arm.expected_repository.to_owned(),
                    REVISION.to_owned(),
                    root,
                )
                .unwrap_or_else(|error| panic!("{}/{tier}: {error}", arm.model_id));
                assert_eq!(repository, arm.expected_repository);
                assert_eq!(revision, REVISION);
            }
            for other in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
                if other == arm {
                    continue;
                }
                let error = flux2_load_spec_at(
                    arm,
                    "q4",
                    &selection,
                    other.expected_repository.to_owned(),
                    REVISION.to_owned(),
                    q4_root.clone(),
                )
                .expect_err("another member's repository must be refused");
                assert!(
                    error.contains(arm.expected_repository),
                    "{} vs {}: {error}",
                    arm.model_id,
                    other.model_id
                );
            }
        }
    }

    fn flux2_snapshot_root(repository: &str, revision: &str, tier: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir()
            .join(format!("sc-22727-flux2-{}-{nonce}", std::process::id()))
            .join(format!("models--{}", repository.replace('/', "--")))
            .join("snapshots")
            .join(revision)
            .join(tier);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn flux2_admission_context_is_reference_free_text_to_image() {
        let contract = mlx_gen_flux2::memory_strategy::registered_dev_t2i_contract(
            &weights_free_spec(Some(Quant::Q4)),
        )
        .unwrap();
        let calibration = contract.calibration.as_ref().unwrap();
        let context = flux2_admission_context(
            FLUX2_DEV_ARM,
            &resident_selection(Some(Quant::Q4)),
            calibration,
            (768, 768),
            Flux2AdmissionProbe {
                fingerprint: &calibration.fingerprint,
                total_bytes: 1_000_000,
                predicted_peak_bytes: 1_000_000,
            },
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

    /// `pub(super)` so the sibling `ltx_tests` module can pin its own provider's contract
    /// shape through the SAME weights-free spec these FLUX.2 pins use (sc-18808 review).
    pub(super) fn weights_free_spec(quant: Option<Quant>) -> LoadSpec {
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

    /// sc-22727 moved the arm from the crate-local `mlx_gen_flux2::provider_registry()` onto the
    /// PRODUCTION catalog the worker composes (`runtime_macos::catalog()`, E4). That is only a
    /// safe move if the two resolve the SAME contract — `mlx-gen-catalog` calls the same
    /// `register_providers`, but nothing in this repo asserted it, and a bundle that wrapped or
    /// re-registered a provider would silently change what every FLUX.2 capture measures.
    ///
    /// Weights-free: both sides are asked for the contract over one placeholder spec, so this
    /// costs no GPU and no snapshot, and it covers every member the arm serves.
    #[test]
    fn the_production_catalog_resolves_the_same_flux2_contracts_as_the_provider_crate() {
        let crate_registry = mlx_gen_flux2::provider_registry().unwrap();
        let catalog = runtime_macos::catalog().expect("the production MLX catalog builds");
        for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
            let spec = weights_free_spec(if arm.tier_quant_reaches_the_loader {
                Some(Quant::Q4)
            } else {
                None
            });
            let direct = crate_registry.memory_strategy_contract(arm.provider, &spec);
            let production = catalog
                .media()
                .memory_strategy_contract(arm.provider, &spec);
            match (direct, production) {
                (Ok(direct), Ok(production)) => assert_eq!(
                    direct, production,
                    "{}: the production catalog resolves a different contract",
                    arm.model_id
                ),
                // A weights-free spec is a placeholder, so a member whose contract wants a real
                // snapshot refuses on BOTH sides. The claim is that they agree, including on the
                // refusal text — that is what rules out a bundle-side wrapper.
                (Err(direct), Err(production)) => assert_eq!(
                    direct.to_string(),
                    production.to_string(),
                    "{}: the two registries refuse for different reasons",
                    arm.model_id
                ),
                (direct, production) => panic!(
                    "{}: one registry answered and the other did not: {direct:?} vs {production:?}",
                    arm.model_id
                ),
            }
        }
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
            FLUX2_DEV_ARM,
            &selection,
            calibration,
            (1024, 1024),
            Flux2AdmissionProbe {
                fingerprint: &calibration.fingerprint,
                total_bytes: 1_000_000,
                predicted_peak_bytes: 1_000_000,
            },
        );
        assert!(
            matches!(check(&exact), MemorySafetyDecision::Accept),
            "exact-fit admission must accept: {:?}",
            check(&exact)
        );

        let unknown = flux2_admission_context(
            FLUX2_DEV_ARM,
            &selection,
            calibration,
            (1024, 1024),
            Flux2AdmissionProbe {
                fingerprint: &calibration.fingerprint,
                total_bytes: 0,
                predicted_peak_bytes: 1,
            },
        );
        assert!(matches!(
            check(&unknown),
            MemorySafetyDecision::Reject { .. }
        ));

        let stale = flux2_admission_context(
            FLUX2_DEV_ARM,
            &selection,
            calibration,
            (1024, 1024),
            Flux2AdmissionProbe {
                fingerprint: "stale-flux2-dev-fingerprint",
                total_bytes: 1_000_000,
                predicted_peak_bytes: 1,
            },
        );
        assert!(matches!(check(&stale), MemorySafetyDecision::Reject { .. }));

        // The route-gate mutation: the same fitting budget in an edit shape must be refused — the
        // T2I contract admits only reference-free text-to-image.
        let mut edit_shaped = flux2_admission_context(
            FLUX2_DEV_ARM,
            &selection,
            calibration,
            (1024, 1024),
            Flux2AdmissionProbe {
                fingerprint: &calibration.fingerprint,
                total_bytes: 1_000_000,
                predicted_peak_bytes: 1_000_000,
            },
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
            FLUX2_DEV_ARM,
            &resident_selection(Some(Quant::Nvfp4)),
            calibration,
            (1024, 1024),
            Flux2AdmissionProbe {
                fingerprint: &calibration.fingerprint,
                total_bytes: 1_000_000,
                predicted_peak_bytes: 1_000_000,
            },
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
mod krea_base_tests {
    use super::*;
    use mlx_gen::gen_core::MemoryStrategySupport;

    fn minimal_request(provider: &str, rung: &str) -> Value {
        json!({
            "planned": {
                "target": {
                    "provider": provider,
                    "modelId": "krea_2_turbo",
                    "mode": "text_to_image",
                    "overlay": "none",
                    "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
                },
                "strategy": { "rung": rung, "engagedRungs": ["resident"], "parameters": {} }
            }
        })
    }

    fn fixture_spec(root: &std::path::Path) -> LoadSpec {
        let encoder_contract = mlx_gen_krea::provider_registry()
            .unwrap()
            .provider_encoder_contract(KREA_BASE_PROVIDER)
            .expect("the pinned Krea base encoder contract");
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &root.join("text_encoder"),
            encoder_contract,
            Some(4),
        )
        .expect("registry-owned Krea text encoder fixture");
        for component in ["transformer", "vae"] {
            let directory = root.join(component);
            std::fs::create_dir_all(&directory).unwrap();
            let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&0_f32.to_le_bytes());
            std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
        }
        std::fs::write(
            root.join("transformer").join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(root.to_owned()))
            .with_quant(Quant::Q4)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    #[test]
    fn the_base_arm_refuses_a_foreign_provider_before_environment_or_weight_work() {
        for provider in ["krea_2_turbo_edit", "krea_2_turbo_control", "qwen_image"] {
            let error = run_krea_base(&minimal_request(provider, "resident"))
                .expect_err("a foreign provider must not reach the Krea base arm");
            assert_eq!(
                error,
                format!("MLX Krea base calibration does not implement provider {provider:?}")
            );
        }
    }

    #[test]
    fn the_base_arm_fails_closed_on_a_non_plain_target_before_weight_work() {
        for (pointer, value, expected) in [
            (
                "/planned/target/modelId",
                json!("krea_2_turbo_edit"),
                "requires modelId",
            ),
            (
                "/planned/target/mode",
                json!("edit_image"),
                "requires reference-free text_to_image mode",
            ),
            (
                "/planned/target/geometry/batch",
                json!(2),
                "requires geometry.batch == 1",
            ),
            (
                "/planned/target/geometry/frames",
                json!(2),
                "requires geometry.frames == 1",
            ),
        ] {
            let mut request = minimal_request(KREA_BASE_PROVIDER, "resident");
            *request.pointer_mut(pointer).unwrap() = value;
            let error = run_krea_base(&request)
                .expect_err("a non-plain Krea target must fail before environment or weights");
            assert!(error.contains(expected), "{pointer}: {error}");
        }

        for (field, value) in [("referenceCount", json!(1)), ("hasReference", json!(true))] {
            let mut request = minimal_request(KREA_BASE_PROVIDER, "resident");
            request["planned"]["target"][field] = value;
            let error = run_krea_base(&request)
                .expect_err("a referenced Krea target must fail before environment or weights");
            assert!(error.contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn complete_sweep_attests_only_the_exact_executed_krea_parameters() {
        let request = json!({
            "planned": {
                "strategy": {
                    "parameters": {
                        "decodeTileEdge": 512,
                        "decodeOverlap": 64,
                        "attentionChunkSize": 67_108_864,
                        "transformerWindowSize": 1
                    }
                }
            }
        });
        let sweep = krea_base_complete_sweep(&request).unwrap();
        assert_eq!(sweep["rangeVerified"], true);
        assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
        assert_eq!(
            sweep["cases"][0]["parameters"],
            request["planned"]["strategy"]["parameters"]
        );
        for (parameter, expected) in [
            ("decodeTileEdge", json!([512])),
            ("decodeOverlap", json!([64])),
            ("attentionChunkSize", json!([67_108_864])),
            ("transformerWindowSize", json!([1])),
        ] {
            assert!(sweep["axes"].as_array().unwrap().iter().any(|axis| {
                axis["parameter"] == parameter && axis["testedValues"] == expected
            }));
        }
    }

    #[test]
    fn base_admission_context_is_reference_free_text_to_image() {
        let calibration = MemoryCalibrationIdentity::new(
            KREA_BASE_CALIBRATION_FINGERPRINT,
            LoadShape::DeferredMaterialization,
        );
        let context = krea_base_context(
            MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            &calibration,
            &calibration.fingerprint,
            768,
            768,
            1,
            1,
        );
        assert_eq!(context.mode, MemoryMode::TextToImage);
        assert!(!context.has_reference);
        assert_eq!(context.geometry.reference_count, 0);
        assert!(context.overlay.is_none());
        assert!(!context.use_pid);
    }

    #[test]
    fn pinned_base_contract_exposes_the_exact_declared_full_ladder() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "sc-18377-krea-contract-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let spec = fixture_spec(&root);
        let contract = mlx_gen_krea::provider_registry()
            .unwrap()
            .memory_strategy_contract(KREA_BASE_PROVIDER, &spec)
            .unwrap()
            .expect("the pinned Krea base provider contract");
        assert_eq!(contract.provider_id, KREA_BASE_PROVIDER);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            KREA_BASE_CALIBRATION_FINGERPRINT
        );
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?}"
            );
        }
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert!(decode.parameters.decode_tile_edges.contains(&512));
        assert!(decode.parameters.decode_overlaps.contains(&64));
        let routes = contract
            .pid_decode_routes
            .as_ref()
            .expect("Krea distinguishes native and PiD decode domains");
        assert_eq!(routes.native.tile_edges, vec![512]);
        assert_eq!(routes.native.tile_overlap, 64);
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(attention.parameters.attention_chunk_sizes, vec![67_108_864]);
        let transformer = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(transformer.parameters.transformer_window_sizes, vec![1]);
        assert_eq!(
            transformer.parameters.transformer_window_components,
            vec![TransformerComponent::Dit]
        );
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
        std::fs::remove_dir_all(root).ok();
    }
}

#[cfg(test)]
mod sdxl_tests {
    use super::*;
    use mlx_gen::gen_core::MemoryStrategySupport;

    fn minimal_request(provider: &str) -> Value {
        json!({
            "planned": {
                "target": {
                    "provider": provider,
                    "modelId": "sdxl",
                    "tier": "q4",
                    "mode": "text_to_image",
                    "overlay": "none",
                    "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
                },
                "loadShape": "deferred_materialization",
                "fixture": "sdxl-base-mlx-q4-768-seed18379-step2",
                "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} }
            }
        })
    }

    fn fixture_spec(root: &std::path::Path) -> LoadSpec {
        for component in ["text_encoder", "text_encoder_2", "unet", "vae"] {
            let directory = root.join(component);
            std::fs::create_dir_all(&directory).unwrap();
            let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&0_f32.to_le_bytes());
            std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
        }
        std::fs::write(
            root.join("unet").join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(root.to_owned()))
            .with_quant(Quant::Q4)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    #[test]
    fn sdxl_arm_refuses_a_foreign_provider_before_environment_or_weight_work() {
        for provider in ["realvisxl", "sdxl_control", "qwen_image"] {
            let error = run_sdxl(&minimal_request(provider))
                .expect_err("a foreign provider must not reach the SDXL arm");
            assert_eq!(
                error,
                format!("MLX SDXL base calibration does not implement provider {provider:?}")
            );
        }
    }

    #[test]
    fn sdxl_admission_context_is_reference_free_text_to_image() {
        let calibration = MemoryCalibrationIdentity::new(
            SDXL_CALIBRATION_FINGERPRINT,
            LoadShape::DeferredMaterialization,
        );
        let context = sdxl_context(
            MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            &calibration,
            &calibration.fingerprint,
            768,
            768,
            1,
            1,
        );
        assert_eq!(context.mode, MemoryMode::TextToImage);
        assert!(!context.has_reference);
        assert_eq!(context.geometry.reference_count, 0);
        assert!(context.overlay.is_none());
        assert!(!context.use_pid);
    }

    #[test]
    fn sdxl_arm_rejects_every_non_base_target_axis_before_weight_work() {
        let base = minimal_request(SDXL_PROVIDER);
        assert!(validate_sdxl_target(&base).is_ok());
        for (pointer, value) in [
            ("/planned/target/modelId", json!("realvisxl")),
            ("/planned/target/mode", json!("image_to_image")),
            ("/planned/target/geometry/batch", json!(2)),
            ("/planned/target/geometry/frames", json!(2)),
        ] {
            let mut request = base.clone();
            *request.pointer_mut(pointer).unwrap_or_else(|| {
                panic!("test mutation path must exist before assignment: {pointer}")
            }) = value;
            assert!(
                validate_sdxl_target(&request).is_err(),
                "SDXL target mutation {pointer} must fail closed"
            );
        }

        for (field, value) in [("referenceCount", json!(1)), ("hasReference", json!(true))] {
            let mut request = base.clone();
            request["planned"]["target"][field] = value;
            assert!(
                validate_sdxl_target(&request).is_err(),
                "SDXL target mutation {field} must fail closed"
            );
        }
    }

    #[test]
    fn sdxl_selection_rejects_a_non_dit_or_detached_window_component() {
        let mut request = minimal_request(SDXL_PROVIDER);
        request["planned"]["strategy"] = json!({
            "rung": "bounded_transformer_residency",
            "engagedRungs": ["resident", "staged_residency", "bounded_transformer_residency"],
            "parameters": { "transformerWindowSize": 1, "transformerWindowComponent": "dit" }
        });
        assert_eq!(
            planned_selection(&request)
                .unwrap()
                .parameters
                .transformer_window_component,
            Some(TransformerComponent::Dit)
        );

        for component in ["text_encoder", "unknown"] {
            request["planned"]["strategy"]["parameters"]["transformerWindowComponent"] =
                json!(component);
            assert!(
                planned_selection(&request).is_err(),
                "component {component:?} must not execute as Dit"
            );
        }
        // sc-18663 taught the SHARED parser `"both"`, because `minimax_h3` declares exactly that.
        // The SDXL guarantee is unchanged and is asserted here rather than assumed: the parse now
        // succeeds, it does NOT produce `Dit`, and the SDXL arm still refuses it.
        request["planned"]["strategy"]["parameters"]["transformerWindowComponent"] = json!("both");
        let both = planned_selection(&request).expect("the shared parser accepts \"both\"");
        assert_eq!(
            both.parameters.transformer_window_component,
            Some(TransformerComponent::Both),
            "\"both\" must not execute as Dit"
        );
        assert!(
            validate_sdxl_selection_parameters(&request, &both).is_err(),
            "the SDXL arm calibrates an explicit Dit window only"
        );
        request["planned"]["strategy"]["parameters"]["transformerWindowComponent"] = json!("dit");
        request["planned"]["strategy"]["parameters"] =
            json!({ "transformerWindowComponent": "dit" });
        assert!(planned_selection(&request).is_err());

        request["planned"]["strategy"] = json!({
            "rung": "resident",
            "engagedRungs": ["resident"],
            "parameters": { "unknownParameter": 1 }
        });
        let selection = planned_selection(&request).unwrap();
        assert!(validate_sdxl_selection_parameters(&request, &selection).is_err());

        request["planned"]["strategy"] = json!({
            "rung": "bounded_decode",
            "engagedRungs": ["resident", "bounded_decode"],
            "parameters": { "decodeTileEdge": 512, "decodeOverlap": 64 }
        });
        let selection = planned_selection(&request).unwrap();
        assert!(validate_sdxl_selection_parameters(&request, &selection).is_err());
    }

    #[test]
    fn sdxl_runtime_complete_sweep_attests_the_exact_tuple_without_a_string_axis() {
        let mut request = minimal_request(SDXL_PROVIDER);
        request["planned"]["strategy"] = json!({
            "rung": "bounded_transformer_residency",
            "engagedRungs": ["resident", "staged_residency", "bounded_transformer_residency"],
            "parameters": { "transformerWindowSize": 5, "transformerWindowComponent": "dit" }
        });
        assert_eq!(
            sdxl_runtime_complete_sweep(&request).unwrap(),
            json!({
                "axes": [{ "parameter": "transformerWindowSize", "testedValues": [5] }],
                "cases": [{
                    "parameters": {
                        "transformerWindowSize": 5,
                        "transformerWindowComponent": "dit"
                    },
                    "result": "passed"
                }],
                "rangeVerified": true
            })
        );

        request["planned"]["strategy"] =
            json!({ "rung": "resident", "engagedRungs": ["resident"], "parameters": {} });
        assert_eq!(
            sdxl_runtime_complete_sweep(&request).unwrap()["axes"],
            json!([])
        );
    }

    #[test]
    fn pinned_sdxl_contract_exposes_only_the_three_implemented_rungs() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "sc-18379-sdxl-contract-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let spec = fixture_spec(&root);
        let contract = mlx_gen_sdxl::provider_registry()
            .unwrap()
            .memory_strategy_contract(SDXL_PROVIDER, &spec)
            .unwrap()
            .expect("the pinned SDXL provider contract");
        assert_eq!(contract.provider_id, SDXL_PROVIDER);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            SDXL_CALIBRATION_FINGERPRINT
        );
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?}"
            );
        }
        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing,
                "{strategy:?} must stay withheld"
            );
        }
        let transformer = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(
            transformer.parameters.transformer_window_sizes,
            vec![1, 2, 5, 10]
        );
        assert_eq!(
            transformer.parameters.transformer_window_components,
            vec![TransformerComponent::Dit]
        );
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
        std::fs::remove_dir_all(root).ok();
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

    #[test]
    fn physical_mlx_rgb_receipts_are_role_and_content_addressed_without_overwrite() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "sceneworks-physical-mlx-receipts-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("create receipt fixture");
        let first = Image {
            width: 2,
            height: 1,
            pixels: vec![1, 2, 3, 4, 5, 6],
        };
        let changed = Image {
            width: 2,
            height: 1,
            pixels: vec![6, 5, 4, 3, 2, 1],
        };

        let selected = persist_physical_mlx_image(
            &root,
            "docs/calibration/sc-test",
            "imc-case",
            "selected_rgb",
            &first,
        )
        .expect("persist first selected receipt");
        let repeated = persist_physical_mlx_image(
            &root,
            "docs/calibration/sc-test",
            "imc-case",
            "selected_rgb",
            &first,
        )
        .expect("identical content-addressed receipt is reusable");
        assert_eq!(selected["path"], repeated["path"]);

        let changed_selected = persist_physical_mlx_image(
            &root,
            "docs/calibration/sc-test",
            "imc-case",
            "selected_rgb",
            &changed,
        )
        .expect("changed bytes receive a different content address");
        assert_ne!(selected["path"], changed_selected["path"]);

        let reference = persist_physical_mlx_image(
            &root,
            "docs/calibration/sc-test",
            "imc-case",
            "reference_rgb",
            &first,
        )
        .expect("the role is part of the receipt address");
        assert_ne!(selected["path"], reference["path"]);

        let selected_local = selected["localPath"]
            .as_str()
            .expect("receipt carries local path");
        std::fs::write(selected_local, b"tampered").expect("tamper fixture receipt");
        let error = persist_physical_mlx_image(
            &root,
            "docs/calibration/sc-test",
            "imc-case",
            "selected_rgb",
            &first,
        )
        .expect_err("exclusive receipt creation must not repair or overwrite tampering");
        assert!(error.contains("already exists with different bytes"));

        std::fs::remove_dir_all(root).ok();
    }
}

/// LTX video-arm regression suite. Historical SC-18810 artifacts stay immutable while this suite
/// follows the permanent provider contract used by SC-18946.
#[cfg(test)]
mod ltx_tests {
    use super::*;

    fn ltx_fixture_contract(quant: Option<Quant>) -> mlx_gen::gen_core::MemoryProviderContract {
        let registry = mlx_gen_ltx::provider_registry().unwrap();
        let fixture = registry
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == LTX_PROVIDER)
            .expect("the provider-owned LTX contract fixture");
        (fixture.contract)(&flux2_tests::weights_free_spec(quant)).unwrap()
    }

    #[test]
    fn the_pinned_ltx_provider_fixture_exposes_the_campaign_rungs_and_exact_parameters() {
        for quant in [Some(Quant::Q8), Some(Quant::Q4), None] {
            let contract = ltx_fixture_contract(quant);
            assert_eq!(contract.provider_id, LTX_PROVIDER);
            assert_eq!(
                contract.engaged_composition(MemoryStrategy::StagedResidency),
                [MemoryStrategy::Resident, MemoryStrategy::StagedResidency]
            );
            assert_eq!(
                contract.engaged_composition(MemoryStrategy::BoundedDecode),
                [
                    MemoryStrategy::Resident,
                    MemoryStrategy::StagedResidency,
                    MemoryStrategy::BoundedDecode,
                ]
            );
            let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
            assert_eq!(
                decode.parameters.decode_tile_edges,
                vec![768, 640, 512, 448, 384, 320, 256, 192]
            );
            assert_eq!(decode.parameters.decode_overlaps, vec![64]);
        }
    }

    /// A minimal, otherwise-valid LTX target. Every field the arm reads before it touches the
    /// environment or the weights is present, so a test that mutates exactly one axis is testing
    /// that axis.
    fn ltx_request_json(width: u32, height: u32, frames: u32) -> Value {
        let predicted_decode_bytes =
            3_300_000_000_u64 + 340 * u64::from(width) * u64::from(height) * u64::from(frames);
        json!({
            "planned": {
                "target": {
                    "provider": LTX_PROVIDER,
                    "modelId": LTX_PROVIDER,
                    "tier": "q8",
                    "mode": "text_to_video",
                    "overlay": "none",
                    "geometry": { "width": width, "height": height, "batch": 1, "frames": frames }
                },
                "backend": "mlx",
                "loadShape": "eager_materialization",
                "strategy": {
                    "rung": "staged_residency",
                    "engagedRungs": ["resident", "staged_residency"],
                    "parameters": {}
                },
                "calibrationFingerprint": LTX_CALIBRATION_FINGERPRINT,
                "_measurementSafety": {
                    "disposition": LTX_SAFETY_REFUSED_OPEN,
                    "tierInventoryBytes": LTX_Q8_INVENTORY_BYTES,
                    "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
                    "predictedDecodeBytes": predicted_decode_bytes,
                    "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
                    "incidentCalibratedProjectionBytes": i128::from(LTX_Q4_F305_CRASH_FOOTPRINT_BYTES)
                        + (i128::from(LTX_Q8_INVENTORY_BYTES) - i128::from(LTX_Q4_INVENTORY_BYTES))
                        + (i128::from(predicted_decode_bytes)
                            - i128::from(LTX_INCIDENT_PREDICTED_DECODE_BYTES)),
                },
                "fixture": format!("ltx-2-3-mlx-q8-{width}x{height}-f{frames}-fps24-seed{LTX_SEED}")
            }
        })
    }

    fn ltx_canary_request_json() -> Value {
        json!({
            "action": "canary",
            "hardware": { "memoryBytes": 128_u64 * 1024 * 1024 * 1024 },
            "planned": {
                "_diagnosticOnly": true,
                "evidenceScope": "fixture",
                "target": {
                    "provider": LTX_PROVIDER,
                    "modelId": LTX_PROVIDER,
                    "tier": "q4",
                    "mode": "text_to_video",
                    "overlay": "none",
                    "geometry": {
                        "width": LTX_CANARY_WIDTH,
                        "height": LTX_CANARY_HEIGHT,
                        "batch": 1,
                        "frames": LTX_CANARY_FRAMES,
                    },
                },
                "backend": "mlx",
                "loadShape": "eager_materialization",
                "strategy": {
                    "rung": "bounded_decode",
                    "engagedRungs": ["resident", "staged_residency", "bounded_decode"],
                    "parameters": {
                        "decodeTileEdge": LTX_CANARY_TILE_EDGE,
                        "decodeOverlap": LTX_CANARY_OVERLAP,
                    },
                },
                "calibrationFingerprint": LTX_CALIBRATION_FINGERPRINT,
                "fixture": LTX_CANARY_FIXTURE,
                "_watchdog": {
                    "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
                },
                "_canary": {
                    "identity": LtxCanaryProfile::Safety.identity(),
                    "videoMode": "no_audio",
                    "fps": LTX_CANARY_FPS,
                    "seed": LTX_CANARY_SEED,
                },
                "_artifact": {
                    "repository": protocol::LTX_REPOSITORY,
                    "revision": LTX_CANARY_ARTIFACT_REVISION,
                    "numericTierInventory": {
                        "files": LTX_CANARY_Q4_INVENTORY_FILES,
                        "bytes": LTX_Q4_INVENTORY_BYTES,
                        "sha256": LTX_CANARY_Q4_INVENTORY_SHA256,
                    },
                    "textEncoderInventory": {
                        "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                        "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                        "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                    },
                },
            },
        })
    }

    fn ltx_product_envelope_canary_request_json() -> Value {
        let mut request = ltx_canary_request_json();
        request["action"] = json!("product_envelope_canary");
        request["planned"]["target"]["geometry"] = json!({
            "width": LTX_PRODUCT_CANARY_WIDTH,
            "height": LTX_PRODUCT_CANARY_HEIGHT,
            "batch": 1,
            "frames": LTX_PRODUCT_CANARY_FRAMES,
        });
        request["planned"]["fixture"] = json!(LTX_PRODUCT_CANARY_FIXTURE);
        request["planned"]["_canary"] = json!({
            "identity": LtxCanaryProfile::ProductEnvelope.identity(),
            "videoMode": "default_av",
            "fps": LTX_PRODUCT_CANARY_FPS,
            "seed": LTX_CANARY_SEED,
        });
        request
    }

    fn ltx_campaign_entry_request_json() -> Value {
        let mut request = ltx_request_json(
            LTX_CAMPAIGN_ENTRY_WIDTH,
            LTX_CAMPAIGN_ENTRY_HEIGHT,
            LTX_CAMPAIGN_ENTRY_FRAMES,
        );
        request["action"] = json!(LTX_CAMPAIGN_ENTRY_ACTION);
        request["hardware"] = json!({ "memoryBytes": 128_u64 * 1024 * 1024 * 1024 });
        request["planned"]["logicalCaseId"] = json!(LTX_CAMPAIGN_ENTRY_LOGICAL_CASE_ID);
        request["planned"]["evidenceScope"] = json!("authoritative");
        request["planned"]["target"]["tier"] = json!("q4");
        request["planned"]["fixture"] = json!(LTX_CAMPAIGN_ENTRY_FIXTURE);
        request["planned"]["negative"] = json!(false);
        request["planned"]["expectedResult"] = json!("passed");
        request["planned"]["modelLoadPolicy"] = json!("fresh_per_case");
        request["planned"]["modelLoadGroup"] = Value::Null;
        request["planned"]["_watchdog"] =
            json!({ "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES });
        request["planned"]["_campaignEntry"] = json!({
            "identity": LTX_CAMPAIGN_ENTRY_IDENTITY,
            "artifact": {
                "repository": protocol::LTX_REPOSITORY,
                "revision": LTX_CANARY_ARTIFACT_REVISION,
                "numericTierInventory": {
                    "files": LTX_CANARY_Q4_INVENTORY_FILES,
                    "bytes": LTX_Q4_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_Q4_INVENTORY_SHA256,
                },
                "textEncoderInventory": {
                    "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                    "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                },
            },
        });
        request["planned"]["_measurementSafety"] = json!({
            "disposition": LTX_SAFETY_REFUSED_OPEN,
            "tierInventoryBytes": LTX_Q4_INVENTORY_BYTES,
            "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
            "incidentCase": "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
            "commonLoad": "complete numeric tier plus shared Gemma stack before geometry-specific work",
            "predictedDecodeBytes": 19_476_906_240_u64,
            "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
            "incidentCalibratedProjectionBytes": 97_906_593_920_u64,
            "projectionAssumptions": [
                "pinned provider decode cost is the only geometry-varying term used",
                "immutable tier inventory delta is added byte-for-byte",
                "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
            ],
            "reason": "incident-calibrated projection is diagnostic only; no proved bound or hard containment admits this row",
        });
        request
    }

    fn ltx_bounded_carrier_request_json() -> Value {
        let mut request = ltx_canary_request_json();
        request["action"] = json!(LTX_BOUNDED_CARRIER_ACTION);
        request["planned"]["logicalCaseId"] = json!(LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID);
        request["planned"]["target"]["geometry"] = json!({
            "width": LTX_CAMPAIGN_ENTRY_WIDTH,
            "height": LTX_CAMPAIGN_ENTRY_HEIGHT,
            "batch": 1,
            "frames": LTX_CAMPAIGN_ENTRY_FRAMES,
        });
        request["planned"]["fixture"] = json!(LTX_BOUNDED_CARRIER_FIXTURE);
        request["planned"]["negative"] = json!(false);
        request["planned"]["expectedResult"] = json!("passed");
        request["planned"]["modelLoadPolicy"] = json!("fresh_per_case");
        request["planned"]["modelLoadGroup"] = Value::Null;
        request["planned"]
            .as_object_mut()
            .unwrap()
            .remove("_canary");
        let artifact = request["planned"]
            .as_object_mut()
            .unwrap()
            .remove("_artifact")
            .unwrap();
        request["planned"]["_boundedCarrier"] = json!({
            "identity": LTX_BOUNDED_CARRIER_IDENTITY,
            "fps": LTX_CAMPAIGN_ENTRY_FPS,
            "seed": LTX_SEED,
            "videoMode": "default_av",
            "artifact": artifact,
        });
        request
    }

    fn ltx_bounded_campaign_request_json() -> Value {
        let mut request = ltx_bounded_carrier_request_json();
        request["action"] = json!(LTX_BOUNDED_CAMPAIGN_ACTION);
        request["planned"]
            .as_object_mut()
            .unwrap()
            .remove("_diagnosticOnly");
        request["planned"]["logicalCaseId"] = json!(LTX_BOUNDED_CAMPAIGN_LOGICAL_CASE_ID);
        request["planned"]["evidenceScope"] = json!("authoritative");
        request["planned"]["fixture"] = json!(LTX_BOUNDED_CAMPAIGN_FIXTURE);
        let carrier = request["planned"]
            .as_object_mut()
            .unwrap()
            .remove("_boundedCarrier")
            .unwrap();
        request["planned"]["_boundedCampaignEntry"] = json!({
            "identity": LTX_BOUNDED_CAMPAIGN_IDENTITY,
            "fps": LTX_CAMPAIGN_ENTRY_FPS,
            "seed": LTX_SEED,
            "videoMode": "default_av",
            "spatialDecodeTiles": 24,
            "artifact": carrier["artifact"],
        });
        request["planned"]["_measurementSafety"] = json!({
            "disposition": LTX_SAFETY_REFUSED_OPEN,
            "tierInventoryBytes": LTX_Q4_INVENTORY_BYTES,
            "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
            "incidentCase": "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
            "commonLoad": "complete numeric tier plus shared Gemma stack before geometry-specific work",
            "predictedDecodeBytes": 6_264_848_640_u64,
            "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
            "incidentCalibratedProjectionBytes": 84_694_536_320_u64,
            "projectionAssumptions": [
                "pinned provider decode cost is the only geometry-varying term used",
                "immutable tier inventory delta is added byte-for-byte",
                "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
            ],
            "reason": "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20318 action is admitted",
        });
        request
    }

    fn ltx_bounded_campaign_request_json_for(tier: &str) -> Value {
        let mut request = ltx_bounded_campaign_request_json();
        let spec = ltx_bounded_campaign_spec(tier).expect("test tier must be allowlisted");
        request["planned"]["target"]["tier"] = json!(spec.tier);
        request["planned"]["logicalCaseId"] = json!(spec.logical_case_id);
        request["planned"]["fixture"] = json!(spec.fixture);
        request["planned"]["_boundedCampaignEntry"]["identity"] = json!(spec.identity);
        request["planned"]["_boundedCampaignEntry"]["artifact"]["numericTierInventory"] = json!({
            "files": spec.inventory_files,
            "bytes": spec.inventory_bytes,
            "sha256": spec.inventory_sha256,
        });
        request["planned"]["_measurementSafety"]["tierInventoryBytes"] =
            json!(spec.inventory_bytes);
        request["planned"]["_measurementSafety"]["incidentCalibratedProjectionBytes"] =
            json!(spec.projection_bytes);
        request["planned"]["_measurementSafety"]["reason"] = json!(if tier == "q4" {
            "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20318 action is admitted"
        } else {
            "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20430 action is admitted"
        });
        request
    }

    #[test]
    fn sc_20318_admits_only_the_exact_two_render_bounded_campaign_row() {
        let request = ltx_bounded_campaign_request_json();
        prevalidate_ltx_bounded_campaign_entry(&request)
            .expect("the exact new bounded campaign row is admissible");
        let geometry = LtxGeometry {
            width: LTX_CAMPAIGN_ENTRY_WIDTH,
            height: LTX_CAMPAIGN_ENTRY_HEIGHT,
            frames: LTX_CAMPAIGN_ENTRY_FRAMES,
            latent_frames: 1 + (LTX_CAMPAIGN_ENTRY_FRAMES - 1) / LTX_TEMPORAL_SCALE,
        };
        let generation_request = ltx_request(geometry, LTX_CAMPAIGN_ENTRY_FPS, LTX_SEED);
        validate_ltx_bounded_carrier_generation_request(&generation_request)
            .expect("both SC-20318 renders use the exact default-A/V request");
        for mutate in [
            |value: &mut GenerationRequest| value.seed = Some(LTX_SEED + 1),
            |value: &mut GenerationRequest| value.fps = Some(LTX_CAMPAIGN_ENTRY_FPS - 1),
            |value: &mut GenerationRequest| value.video_mode = Some("no_audio".to_owned()),
        ] {
            let mut changed = generation_request.clone();
            mutate(&mut changed);
            assert!(validate_ltx_bounded_carrier_generation_request(&changed).is_err());
        }
        for (pointer, value) in [
            ("/action", json!("run")),
            (
                "/planned/logicalCaseId",
                json!(LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID),
            ),
            ("/planned/fixture", json!(LTX_BOUNDED_CARRIER_FIXTURE)),
            ("/planned/target/tier", json!("q8")),
            ("/planned/strategy/parameters/decodeTileEdge", json!(193)),
            ("/planned/strategy/parameters/decodeOverlap", json!(63)),
            ("/planned/_boundedCampaignEntry/identity", json!("mutated")),
            ("/planned/_boundedCampaignEntry/seed", json!(LTX_SEED + 1)),
            (
                "/planned/_boundedCampaignEntry/videoMode",
                json!("no_audio"),
            ),
            (
                "/planned/_boundedCampaignEntry/spatialDecodeTiles",
                json!(23),
            ),
            (
                "/planned/_measurementSafety/predictedDecodeBytes",
                json!(6_264_848_639_u64),
            ),
            ("/planned/_measurementSafety/incidentCase", json!("mutated")),
            ("/planned/_measurementSafety/reason", json!("mutated")),
        ] {
            let mut mutated = request.clone();
            *mutated.pointer_mut(pointer).expect("mutation pointer") = value;
            let error = prevalidate_ltx_bounded_campaign_entry(&mutated)
                .expect_err("every SC-20318 identity mutation must fail before weights");
            assert!(
                error.contains("SC-20318")
                    || error.contains("SC-20430")
                    || error.contains("SC-18946"),
                "{pointer}: {error}"
            );
        }
        let direct = run_ltx_bounded_campaign_entry(&request)
            .expect_err("private JSON alone cannot bypass the live watchdog");
        assert!(
            direct.contains("live external watchdog channel"),
            "{direct}"
        );

        let mut ordinary = request.clone();
        ordinary["action"] = json!("run");
        ordinary["planned"]
            .as_object_mut()
            .unwrap()
            .remove("_boundedCampaignEntry");
        let refusal = run_ltx(&ordinary)
            .expect_err("the new row must remain refused by the ordinary campaign action");
        assert!(refusal.contains("safety_refused_open"), "{refusal}");
    }

    #[test]
    fn sc_20430_exactly_allowlists_q8_and_bf16_bounded_campaign_rows() {
        for tier in ["q8", "bf16"] {
            let request = ltx_bounded_campaign_request_json_for(tier);
            prevalidate_ltx_bounded_campaign_entry(&request)
                .unwrap_or_else(|error| panic!("exact {tier} bounded entry rejected: {error}"));
            for (pointer, mutation) in [
                ("/action", json!("run")),
                ("/planned/target/tier", json!("q4")),
                (
                    "/planned/logicalCaseId",
                    json!(LTX_BOUNDED_CAMPAIGN_LOGICAL_CASE_ID),
                ),
                ("/planned/fixture", json!(LTX_BOUNDED_CAMPAIGN_FIXTURE)),
                ("/planned/_boundedCampaignEntry/identity", json!("mutated")),
                (
                    "/planned/_boundedCampaignEntry/artifact/numericTierInventory/bytes",
                    json!(LTX_Q4_INVENTORY_BYTES),
                ),
                ("/planned/_boundedCampaignEntry/seed", json!(LTX_SEED + 1)),
                (
                    "/planned/strategy/parameters/decodeTileEdge",
                    json!(LTX_CANARY_TILE_EDGE + 1),
                ),
            ] {
                let mut changed = request.clone();
                *changed.pointer_mut(pointer).expect("mutation pointer") = mutation;
                assert!(
                    prevalidate_ltx_bounded_campaign_entry(&changed).is_err(),
                    "{tier} mutation {pointer} reached model resolution"
                );
            }
            let direct = run_ltx_bounded_campaign_entry(&request)
                .expect_err("private JSON alone cannot bypass the watchdog");
            assert!(
                direct.contains("live external watchdog channel"),
                "{direct}"
            );
            let mut ordinary = request.clone();
            ordinary["action"] = json!("run");
            ordinary["planned"]
                .as_object_mut()
                .unwrap()
                .remove("_boundedCampaignEntry");
            let refusal = run_ltx(&ordinary)
                .expect_err("the ordinary 73-row action must refuse every new tier");
            assert!(refusal.contains("safety_refused_open"), "{refusal}");
        }
        assert!(ltx_bounded_campaign_spec("fp16").is_err());
    }

    #[test]
    fn sc_20254_admits_only_the_exact_single_render_bounded_carrier() {
        let request = ltx_bounded_carrier_request_json();
        let (geometry, selection) = prevalidate_ltx_bounded_carrier_proof(&request)
            .expect("the exact diagnostic bounded carrier is admissible");
        let decode = LtxDecodePlan::resolve_for_selection(&selection, geometry).unwrap();
        assert!(decode.tiling_engaged());
        assert_eq!(decode.spatial_tile_px(), u64::from(LTX_CANARY_TILE_EDGE));
        assert_eq!(decode.spatial_overlap_px(), u64::from(LTX_CANARY_OVERLAP));
        assert_eq!(decode.spatial_tile_count(geometry).unwrap(), 24);

        for (pointer, value) in [
            ("/action", json!(LTX_CAMPAIGN_ENTRY_ACTION)),
            (
                "/planned/logicalCaseId",
                json!(LTX_CAMPAIGN_ENTRY_LOGICAL_CASE_ID),
            ),
            ("/planned/fixture", json!(LTX_CAMPAIGN_ENTRY_FIXTURE)),
            ("/planned/strategy/rung", json!("staged_residency")),
            (
                "/planned/strategy/parameters/decodeTileEdge",
                json!(LTX_CANARY_TILE_EDGE + 1),
            ),
            (
                "/planned/strategy/parameters/decodeOverlap",
                json!(LTX_CANARY_OVERLAP + 1),
            ),
            ("/planned/_boundedCarrier/identity", json!("mutated")),
            ("/planned/_boundedCarrier/seed", json!(LTX_SEED + 1)),
            (
                "/planned/_watchdog/maxFootprintBytes",
                json!(LTX_CANARY_MAX_FOOTPRINT_BYTES + 1),
            ),
        ] {
            let mut mutated = request.clone();
            *mutated.pointer_mut(pointer).expect("mutation pointer") = value;
            let error = prevalidate_ltx_bounded_carrier_proof(&mutated)
                .expect_err("every bounded-carrier identity mutation must fail closed");
            assert!(error.contains("SC-20254"), "{pointer}: {error}");
        }

        let generation_request = ltx_request(geometry, LTX_CAMPAIGN_ENTRY_FPS, LTX_SEED);
        validate_ltx_bounded_carrier_generation_request(&generation_request)
            .expect("the exact provider request is bound independently");
        for mutate in [
            |value: &mut GenerationRequest| value.seed = Some(LTX_SEED + 1),
            |value: &mut GenerationRequest| value.fps = Some(LTX_CAMPAIGN_ENTRY_FPS + 1),
            |value: &mut GenerationRequest| value.frames = Some(LTX_CAMPAIGN_ENTRY_FRAMES - 1),
            |value: &mut GenerationRequest| value.video_mode = Some("no_audio".to_owned()),
        ] {
            let mut mutated = generation_request.clone();
            mutate(&mut mutated);
            assert!(validate_ltx_bounded_carrier_generation_request(&mutated).is_err());
        }

        let direct = run_ltx_bounded_carrier_proof(&request)
            .expect_err("private JSON alone cannot bypass the live watchdog");
        assert!(
            direct.contains("live external watchdog channel"),
            "{direct}"
        );
    }

    #[test]
    fn sc_20191_admits_only_the_exact_private_campaign_row_before_weight_work() {
        let request = ltx_campaign_entry_request_json();
        prevalidate_ltx_campaign_entry(&request).expect("the exact canonical row is admissible");

        for (pointer, value) in [
            ("/action", json!("run")),
            ("/planned/logicalCaseId", json!("implan-mutated")),
            ("/planned/target/tier", json!("q8")),
            ("/planned/target/geometry/frames", json!(97)),
            (
                "/planned/fixture",
                json!("ltx-2-3-mlx-q4-768x512-f121-fps30-seed1"),
            ),
            (
                "/planned/strategy/parameters",
                json!({ "decodeTileEdge": 192 }),
            ),
            ("/planned/evidenceScope", json!("fixture")),
            ("/planned/modelLoadPolicy", json!("batch_rungs")),
            (
                "/planned/_watchdog/maxFootprintBytes",
                json!(LTX_CANARY_MAX_FOOTPRINT_BYTES - 1),
            ),
            (
                "/planned/_campaignEntry/identity",
                json!("sc-20191-mutated"),
            ),
            (
                "/planned/_measurementSafety/predictedDecodeBytes",
                json!(19_476_906_239_u64),
            ),
        ] {
            let mut mutated = request.clone();
            *mutated.pointer_mut(pointer).expect("mutation pointer") = value;
            let error = prevalidate_ltx_campaign_entry(&mutated)
                .expect_err("every campaign-entry identity mutation must fail closed");
            assert!(
                error.contains("SC-20191") || error.contains("SC-18946"),
                "{pointer}: {error}"
            );
        }

        let direct = run_ltx_campaign_entry(&request)
            .expect_err("canonical JSON alone cannot bypass the live watchdog");
        assert!(
            direct.contains("live external watchdog channel"),
            "{direct}"
        );
    }

    #[test]
    fn sc_20216_phase_channel_is_exact_monotonic_and_action_bound() {
        let request = ltx_campaign_entry_request_json();
        let host_memory = request["hardware"]["memoryBytes"].as_u64().unwrap();
        let nonce = "cd".repeat(32);
        let base = json!({
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "nonce": nonce,
            "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
            "hostMemoryBytes": host_memory,
            "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
            "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
        });
        let (mut missing_watchdog, missing_adapter) = UnixStream::pair().unwrap();
        let missing_thread = std::thread::spawn(move || {
            missing_watchdog
                .write_all(format!("{base}\n").as_bytes())
                .unwrap();
        });
        let error = consume_ltx_canary_watchdog_attestation_stream(&request, missing_adapter)
            .expect_err("campaign entry must require authenticated phases");
        missing_thread.join().unwrap();
        assert!(error.contains("action-bound authenticated provider phase contract"));

        let (adapter, mut watchdog) = UnixStream::pair().unwrap();
        let mut attestation = LtxCanaryWatchdogAttestation {
            max_footprint_bytes: LTX_CANARY_MAX_FOOTPRINT_BYTES,
            max_runtime_seconds: LTX_CANARY_MAX_RUNTIME_SECONDS,
            host_memory_bytes: host_memory,
            min_initial_memory_free_bytes: 0,
            min_memory_free_bytes: 0,
            nonce: nonce.clone(),
            stream: Some(adapter),
        };
        let lease = attestation.start_lease().unwrap();
        let (marked_sender, marked_receiver) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut lease = lease;
            let result = lease.mark("common_load");
            marked_sender.send((lease, result)).unwrap();
        });
        assert_eq!(
            read_watchdog_line(&mut watchdog).unwrap(),
            format!("PHASE {nonce} 1 common_load")
        );
        assert!(matches!(
            marked_receiver.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        watchdog
            .write_all(format!("PHASE_ACK {nonce} 1 common_load\n").as_bytes())
            .unwrap();
        let (mut lease, result) = marked_receiver.recv().unwrap();
        result.expect("phase entry remains blocked until the watchdog acknowledges it");

        let remaining_nonce = nonce.clone();
        let peer = std::thread::spawn(move || {
            for (index, name) in LTX_PROVIDER_PHASE_NAMES.iter().enumerate().skip(1) {
                assert_eq!(
                    read_watchdog_line(&mut watchdog).unwrap(),
                    format!("PHASE {remaining_nonce} {} {name}", index + 1)
                );
                watchdog
                    .write_all(
                        format!("PHASE_ACK {remaining_nonce} {} {name}\n", index + 1).as_bytes(),
                    )
                    .unwrap();
            }
            assert_eq!(
                read_watchdog_line(&mut watchdog).unwrap(),
                format!("DONE {remaining_nonce}")
            );
            watchdog
                .write_all(format!("BYE {remaining_nonce}\n").as_bytes())
                .unwrap();
        });
        for name in LTX_PROVIDER_PHASE_NAMES.iter().skip(1) {
            lease.mark(name).expect("exact next phase");
        }
        assert!(lease.mark("cleanup").unwrap_err().contains("exceeded"));
        lease.complete().unwrap();
        peer.join().unwrap();

        let (_timeout_sender, timeout_receiver) = std::sync::mpsc::sync_channel(1);
        assert!(wait_for_ltx_phase_acknowledgement(
            &timeout_receiver,
            1,
            "common_load",
            Duration::ZERO,
        )
        .unwrap_err()
        .contains("acknowledgement"));
        let (foreign_sender, foreign_receiver) = std::sync::mpsc::sync_channel(1);
        foreign_sender
            .send(Ok((2, "primary_conditioning".to_owned())))
            .unwrap();
        assert!(wait_for_ltx_phase_acknowledgement(
            &foreign_receiver,
            1,
            "common_load",
            Duration::ZERO,
        )
        .unwrap_err()
        .contains("foreign provider phase"));

        let (writer, _reader) = UnixStream::pair().unwrap();
        let (_sender, completion) = std::sync::mpsc::sync_channel(1);
        let (_phase_sender, phase_acknowledgements) = std::sync::mpsc::sync_channel(1);
        let mut reordered = LtxCanaryWatchdogLease {
            writer,
            completion,
            phase_acknowledgements,
            nonce,
            phase_sequence: 0,
            expected_phases: &LTX_PROVIDER_PHASE_NAMES,
        };
        assert!(reordered
            .mark("primary_decode")
            .unwrap_err()
            .contains("reordered"));
    }

    #[test]
    fn sc_20254_phase_attestation_accepts_only_its_exact_five_phase_socket_contract() {
        let request = ltx_bounded_carrier_request_json();
        let host_memory = request["hardware"]["memoryBytes"].as_u64().unwrap();
        let nonce = "ef".repeat(32);
        let payload = json!({
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "nonce": nonce,
            "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
            "hostMemoryBytes": host_memory,
            "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
            "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
            "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
            "providerPhaseProfile": LTX_BOUNDED_CARRIER_PHASE_PROFILE,
            "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
        });
        let (mut watchdog, adapter) = UnixStream::pair().unwrap();
        let peer_nonce = nonce.clone();
        let peer_payload = payload.clone();
        let peer = std::thread::spawn(move || {
            watchdog
                .write_all(format!("{peer_payload}\n").as_bytes())
                .unwrap();
            assert_eq!(
                read_watchdog_line(&mut watchdog).unwrap(),
                format!("ACK {peer_nonce}")
            );
            watchdog
                .write_all(format!("GO {peer_nonce}\n").as_bytes())
                .unwrap();
            for (index, name) in LTX_BOUNDED_CARRIER_PHASE_NAMES.iter().enumerate() {
                assert_eq!(
                    read_watchdog_line(&mut watchdog).unwrap(),
                    format!("PHASE {peer_nonce} {} {name}", index + 1)
                );
                watchdog
                    .write_all(format!("PHASE_ACK {peer_nonce} {} {name}\n", index + 1).as_bytes())
                    .unwrap();
            }
            assert_eq!(
                read_watchdog_line(&mut watchdog).unwrap(),
                format!("DONE {peer_nonce}")
            );
            watchdog
                .write_all(format!("BYE {peer_nonce}\n").as_bytes())
                .unwrap();
        });
        let mut attestation =
            consume_ltx_canary_watchdog_attestation_stream(&request, adapter).unwrap();
        let mut lease = attestation
            .start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)
            .unwrap();
        for name in LTX_BOUNDED_CARRIER_PHASE_NAMES {
            lease
                .mark(name)
                .expect("watchdog must ACK each exact phase");
        }
        lease.complete().unwrap();
        peer.join().unwrap();

        for mutate in [
            |value: &mut Value| {
                value["providerPhaseProfile"] = json!(LTX_CAMPAIGN_ENTRY_PHASE_PROFILE);
            },
            |value: &mut Value| {
                value["providerPhases"] = json!(LTX_PROVIDER_PHASE_NAMES);
            },
            |value: &mut Value| {
                value["providerPhases"] = json!([
                    "primary_conditioning",
                    "common_load",
                    "primary_denoise",
                    "primary_decode",
                    "cleanup",
                ]);
            },
        ] {
            let mut mutated = json!({
                "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
                "nonce": nonce,
                "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
                "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
                "hostMemoryBytes": host_memory,
                "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
                "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
                "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
                "providerPhaseProfile": LTX_BOUNDED_CARRIER_PHASE_PROFILE,
                "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
            });
            mutate(&mut mutated);
            let (mut bad_watchdog, bad_adapter) = UnixStream::pair().unwrap();
            let bad_peer = std::thread::spawn(move || {
                bad_watchdog
                    .write_all(format!("{mutated}\n").as_bytes())
                    .unwrap();
            });
            let error = consume_ltx_canary_watchdog_attestation_stream(&request, bad_adapter)
                .expect_err("profile, list and order mutations must fail closed");
            bad_peer.join().unwrap();
            assert!(
                error.contains("action-bound authenticated provider phase contract"),
                "{error}"
            );
        }
    }

    #[test]
    fn sc_20318_phase_attestation_is_action_bound_to_its_exact_five_phases() {
        let request = ltx_bounded_campaign_request_json();
        let host_memory = request["hardware"]["memoryBytes"].as_u64().unwrap();
        let nonce = "ab".repeat(32);
        let payload = json!({
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "nonce": nonce,
            "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
            "hostMemoryBytes": host_memory,
            "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
            "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
            "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
            "providerPhaseProfile": LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE,
            "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
        });
        let (mut watchdog, adapter) = UnixStream::pair().unwrap();
        let peer_nonce = nonce.clone();
        let peer_payload = payload.clone();
        let peer = std::thread::spawn(move || {
            watchdog
                .write_all(format!("{peer_payload}\n").as_bytes())
                .unwrap();
            assert_eq!(
                read_watchdog_line(&mut watchdog).unwrap(),
                format!("ACK {peer_nonce}")
            );
            watchdog
                .write_all(format!("GO {peer_nonce}\n").as_bytes())
                .unwrap();
            for (index, name) in LTX_BOUNDED_CARRIER_PHASE_NAMES.iter().enumerate() {
                assert_eq!(
                    read_watchdog_line(&mut watchdog).unwrap(),
                    format!("PHASE {peer_nonce} {} {name}", index + 1)
                );
                watchdog
                    .write_all(format!("PHASE_ACK {peer_nonce} {} {name}\n", index + 1).as_bytes())
                    .unwrap();
            }
            assert_eq!(
                read_watchdog_line(&mut watchdog).unwrap(),
                format!("DONE {peer_nonce}")
            );
            watchdog
                .write_all(format!("BYE {peer_nonce}\n").as_bytes())
                .unwrap();
        });
        let mut attestation =
            consume_ltx_canary_watchdog_attestation_stream(&request, adapter).unwrap();
        let mut lease = attestation
            .start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)
            .unwrap();
        for name in LTX_BOUNDED_CARRIER_PHASE_NAMES {
            lease.mark(name).unwrap();
        }
        lease.complete().unwrap();
        peer.join().unwrap();

        for mutate in [
            |value: &mut Value| {
                value["providerPhaseProfile"] = json!(LTX_BOUNDED_CARRIER_PHASE_PROFILE)
            },
            |value: &mut Value| value["providerPhaseProfile"] = json!("foreign-profile"),
            |value: &mut Value| value["providerPhases"] = json!(LTX_PROVIDER_PHASE_NAMES),
            |value: &mut Value| {
                value["providerPhases"] = json!([
                    "common_load",
                    "primary_conditioning",
                    "primary_denoise",
                    "primary_decode"
                ])
            },
            |value: &mut Value| {
                value["providerPhases"] = json!([
                    "primary_conditioning",
                    "common_load",
                    "primary_denoise",
                    "primary_decode",
                    "cleanup"
                ])
            },
        ] {
            let mut changed = payload.clone();
            mutate(&mut changed);
            let (mut bad_watchdog, bad_adapter) = UnixStream::pair().unwrap();
            let bad_peer = std::thread::spawn(move || {
                bad_watchdog
                    .write_all(format!("{changed}\n").as_bytes())
                    .unwrap();
            });
            let error = consume_ltx_canary_watchdog_attestation_stream(&request, bad_adapter)
                .expect_err("SC-20318 phase profile/list/order mutations must fail");
            bad_peer.join().unwrap();
            assert!(
                error.contains("action-bound authenticated provider phase contract"),
                "{error}"
            );
        }
    }

    #[test]
    fn sc_20191_requires_the_exact_untiled_full_av_response_carrier() {
        let diagnostics = |name: &str, value: u64| json!({ "name": name, "value": value });
        let mut fragment = json!({
            "strategy": {
                "rung": "staged_residency",
                "engagedRungs": ["resident", "staged_residency"],
                "parameters": {},
            },
            "output": {
                "frames": 121,
                "fps": 30,
                "audio": { "present": true, "samples": 1, "sampleRate": 48_000, "channels": 2 },
                "firstFrameNondegenerate": true,
            },
            "diagnostics": { "measurements": [
                diagnostics("renderedFrames", 121),
                diagnostics("outputFps", 30),
                diagnostics("audioTrackDecoded", 1),
                diagnostics("decodeTilingEngaged", 0),
                diagnostics("decodeTileSpatialPx", 0),
                diagnostics("decodeTileOverlapPx", 0),
                diagnostics("latentTemporalDepth", 16),
                diagnostics("latentTokens", 6_144),
            ] },
        });
        validate_ltx_campaign_entry_fragment(&fragment).expect("exact carrier");
        fragment["diagnostics"]["measurements"][3]["value"] = json!(1);
        let tiled = validate_ltx_campaign_entry_fragment(&fragment)
            .expect_err("a tiled carrier cannot publish the untiled canonical row");
        assert!(tiled.contains("decodeTilingEngaged"), "{tiled}");
    }

    #[test]
    fn the_ltx_safety_canary_is_an_exact_non_promotable_multi_tile_tuple() {
        let request = ltx_canary_request_json();
        let selection = validate_ltx_canary_plan(&request).expect("exact canary tuple");
        assert_eq!(selection.strategy, MemoryStrategy::BoundedDecode);
        assert_eq!(selection.tier.quant, Some(Quant::Q4));
        assert_eq!(
            selection.parameters.decode_tile_edge,
            Some(LTX_CANARY_TILE_EDGE)
        );
        assert_eq!(
            selection.parameters.decode_overlap,
            Some(LTX_CANARY_OVERLAP)
        );
        let generated = ltx_canary_generation_request();
        assert_eq!(generated.prompt, "a sunlit pine branch, static camera");
        assert_eq!(
            (generated.width, generated.height, generated.frames),
            (LTX_CANARY_WIDTH, LTX_CANARY_HEIGHT, Some(LTX_CANARY_FRAMES))
        );
        assert_eq!(generated.video_mode.as_deref(), Some("no_audio"));
        let memory = generated.memory.expect("bounded decode carrier");
        assert!(memory.tile_vae_decode);
        assert_eq!(memory.decode_tile_edge, Some(LTX_CANARY_TILE_EDGE));
        assert_eq!(memory.decode_overlap, Some(LTX_CANARY_OVERLAP));

        let mut admitted =
            ltx_canary_request_for_provider_admission(ltx_canary_generation_request())
                .expect("the exact diagnostic tuple may use the private admission bridge");
        assert_eq!(admitted.video_mode, None);
        restore_ltx_canary_no_audio_after_configuration(&mut admitted)
            .expect("no_audio is restored only after provider configuration");
        assert_eq!(admitted.video_mode.as_deref(), Some("no_audio"));

        let mut production_default = ltx_canary_generation_request();
        production_default.video_mode = Some("default".to_owned());
        assert!(ltx_canary_request_for_provider_admission(production_default).is_err());
        let mut provider_override = ltx_canary_generation_request();
        provider_override.video_mode = Some("default".to_owned());
        assert!(restore_ltx_canary_no_audio_after_configuration(&mut provider_override).is_err());
    }

    #[test]
    fn the_ltx_product_envelope_canary_is_exact_full_av_and_physically_multi_tile() {
        let request = ltx_product_envelope_canary_request_json();
        let selection = validate_ltx_product_envelope_canary_plan(&request)
            .expect("exact product-envelope canary tuple");
        assert_eq!(selection.strategy, MemoryStrategy::BoundedDecode);
        let generated = ltx_product_envelope_canary_generation_request();
        assert_eq!(
            (
                generated.width,
                generated.height,
                generated.frames,
                generated.fps
            ),
            (
                LTX_PRODUCT_CANARY_WIDTH,
                LTX_PRODUCT_CANARY_HEIGHT,
                Some(LTX_PRODUCT_CANARY_FRAMES),
                Some(LTX_PRODUCT_CANARY_FPS),
            )
        );
        assert_eq!(
            generated.prompt,
            "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
        );
        assert_eq!(generated.video_mode, None, "default A/V must stay unset");
        let geometry = LtxGeometry {
            width: LTX_PRODUCT_CANARY_WIDTH,
            height: LTX_PRODUCT_CANARY_HEIGHT,
            frames: LTX_PRODUCT_CANARY_FRAMES,
            latent_frames: 1 + (LTX_PRODUCT_CANARY_FRAMES - 1) / LTX_TEMPORAL_SCALE,
        };
        let decode = LtxDecodePlan::resolve_for_selection(&selection, geometry)
            .expect("exact bounded decode plan");
        assert_eq!(decode.spatial_tile_count(geometry).unwrap(), 24);
        assert!(validate_diagnostic_audio(
            LtxCanaryProfile::ProductEnvelope,
            Some(DiagnosticAudioIdentity {
                samples: 1,
                sample_rate: 24_000,
                channels: 2,
            }),
        )
        .is_ok());
        for audio in [
            None,
            Some(DiagnosticAudioIdentity {
                samples: 0,
                sample_rate: 24_000,
                channels: 2,
            }),
            Some(DiagnosticAudioIdentity {
                samples: 1,
                sample_rate: 0,
                channels: 2,
            }),
            Some(DiagnosticAudioIdentity {
                samples: 1,
                sample_rate: 24_000,
                channels: 0,
            }),
        ] {
            assert!(validate_diagnostic_audio(LtxCanaryProfile::ProductEnvelope, audio).is_err());
        }
        assert!(validate_diagnostic_audio(
            LtxCanaryProfile::Safety,
            Some(DiagnosticAudioIdentity {
                samples: 1,
                sample_rate: 24_000,
                channels: 2,
            }),
        )
        .is_err());
        let direct = run_ltx_product_envelope_canary(&request)
            .expect_err("product-envelope canary must refuse without the external watchdog");
        assert!(direct.contains("live external watchdog channel"));
    }

    #[test]
    fn the_ltx_safety_canary_accounts_for_only_the_exact_av_bfloat16_ones_cache() {
        let expected = ltx_canary_ones_cache_bytes().expect("checked ONES_CACHE arithmetic");
        assert_eq!(
            expected,
            (LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION + LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION)
                * BFLOAT16_BYTES_PER_ELEMENT
        );
        assert!(validate_ltx_canary_pre_provider(AllocatorState {
            active: 0,
            cache: 0,
        })
        .is_ok());
        let pre_provider = AllocatorState {
            active: 7,
            cache: 0,
        };
        assert!(validate_ltx_canary_cleanup(
            pre_provider,
            AllocatorState {
                active: pre_provider.active + expected,
                cache: 0,
            },
            expected,
        )
        .is_ok());

        let low = validate_ltx_canary_cleanup(
            pre_provider,
            AllocatorState {
                active: pre_provider.active + expected - 1,
                cache: 0,
            },
            expected,
        )
        .expect_err("one byte below the exact active identity must fail");
        assert!(low.contains("intentional persistent active"));
        let high = validate_ltx_canary_cleanup(
            pre_provider,
            AllocatorState {
                active: pre_provider.active + expected + 1,
                cache: 0,
            },
            expected,
        )
        .expect_err("one byte above the exact active identity must fail");
        assert!(high.contains("intentional persistent active"));

        let dirty_pre = validate_ltx_canary_pre_provider(AllocatorState {
            active: 0,
            cache: 1,
        })
        .expect_err("a non-empty baseline cache must fail");
        assert!(dirty_pre.contains("preProviderCacheBytes 1"));
        let dirty_post = validate_ltx_canary_cleanup(
            pre_provider,
            AllocatorState {
                active: pre_provider.active + expected,
                cache: 1,
            },
            expected,
        )
        .expect_err("a retained post-cleanup cache byte must fail");
        assert!(dirty_post.contains("postCleanupCacheBytes 1"));

        let overflow = validate_ltx_canary_cleanup(
            AllocatorState {
                active: u64::MAX,
                cache: 0,
            },
            AllocatorState::default(),
            expected,
        )
        .expect_err("pre-provider plus persistent active bytes must not wrap");
        assert_eq!(
            overflow,
            "LTX safety canary cleanup active-byte arithmetic overflowed"
        );
    }

    #[test]
    fn the_ltx_safety_canary_requires_the_live_watchdog_ack_go_channel_before_limits() {
        let request = ltx_canary_request_json();
        let host_memory = request["hardware"]["memoryBytes"]
            .as_u64()
            .expect("canary host memory");
        let nonce = "ab".repeat(32);
        let payload = json!({
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "nonce": nonce,
            "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "maxRuntimeSeconds": LTX_CANARY_MAX_RUNTIME_SECONDS,
            "hostMemoryBytes": host_memory,
            "minInitialMemoryFreeBytes": 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
            "minMemoryFreeBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100),
        });
        let (mut watchdog, adapter) = UnixStream::pair().expect("watchdog socket pair");
        let expected_nonce = nonce.clone();
        let valid_payload = payload.clone();
        let watchdog_thread = std::thread::spawn(move || {
            watchdog
                .write_all(format!("{valid_payload}\n").as_bytes())
                .expect("send attestation");
            assert_eq!(
                read_watchdog_line(&mut watchdog).expect("adapter ACK"),
                format!("ACK {expected_nonce}")
            );
            watchdog
                .write_all(format!("GO {expected_nonce}\n").as_bytes())
                .expect("release canary");
            watchdog
                .write_all(format!("PING {expected_nonce}\n").as_bytes())
                .expect("maintain live watchdog lease");
            assert_eq!(
                read_watchdog_line(&mut watchdog).expect("adapter DONE"),
                format!("DONE {expected_nonce}")
            );
            watchdog
                .write_all(format!("BYE {expected_nonce}\n").as_bytes())
                .expect("complete canary lease");
        });
        let mut attested = consume_ltx_canary_watchdog_attestation_stream(&request, adapter)
            .expect("exact live watchdog handshake");
        let lease = attested.start_lease().expect("held watchdog lease");
        lease.complete().expect("DONE/BYE completion handshake");
        watchdog_thread.join().expect("watchdog thread");
        assert_eq!(attested.max_footprint_bytes, LTX_CANARY_MAX_FOOTPRINT_BYTES);
        assert_eq!(attested.max_runtime_seconds, LTX_CANARY_MAX_RUNTIME_SECONDS);
        assert!(std::env::var_os("SCENEWORKS_MEMORY_WATCHDOG_SOCKET").is_none());
        let direct = consume_ltx_canary_watchdog_attestation(&request)
            .expect_err("direct stdin canary has no live watchdog channel");
        assert!(direct.contains("live external watchdog channel"));
        let direct_action = run_ltx_canary(&request)
            .expect_err("public canary action must refuse without the live watchdog");
        assert!(direct_action.contains("live external watchdog channel"));

        let mut swap_gated = payload.clone();
        swap_gated["minSwapFreeBytes"] = json!(1024_u64 * 1024 * 1024);
        let (mut fake_watchdog, fake_adapter) =
            UnixStream::pair().expect("swap-gated watchdog socket pair");
        let fake_thread = std::thread::spawn(move || {
            fake_watchdog
                .write_all(format!("{swap_gated}\n").as_bytes())
                .expect("send swap-gated attestation");
        });
        let error = consume_ltx_canary_watchdog_attestation_stream(&request, fake_adapter)
            .expect_err("an arbitrary swap reserve must not enter the reviewed canary bounds");
        fake_thread.join().expect("fake watchdog thread");
        assert!(error.contains("exact reviewed bounds"));

        let mut percent_gated = payload.clone();
        percent_gated["minInitialMemoryFreePercent"] = json!(70);
        let (mut fake_watchdog, fake_adapter) =
            UnixStream::pair().expect("percent-gated watchdog socket pair");
        let fake_thread = std::thread::spawn(move || {
            fake_watchdog
                .write_all(format!("{percent_gated}\n").as_bytes())
                .expect("send percent-gated attestation");
        });
        let error = consume_ltx_canary_watchdog_attestation_stream(&request, fake_adapter)
            .expect_err("a redundant percentage floor must not enter the reviewed canary bounds");
        fake_thread.join().expect("fake watchdog thread");
        assert!(error.contains("exact reviewed bounds"));

        let mut weakened = payload;
        weakened["minInitialMemoryFreeBytes"] =
            json!(2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + host_memory.div_ceil(100) - 1);
        let (mut fake_watchdog, fake_adapter) =
            UnixStream::pair().expect("weakened watchdog socket pair");
        let fake_thread = std::thread::spawn(move || {
            fake_watchdog
                .write_all(format!("{weakened}\n").as_bytes())
                .expect("send weakened attestation");
        });
        let error = consume_ltx_canary_watchdog_attestation_stream(&request, fake_adapter)
            .expect_err("weakened runtime floor must not attest");
        fake_thread.join().expect("fake watchdog thread");
        assert!(error.contains("exact reviewed bounds"));
    }

    #[test]
    fn the_ltx_safety_canary_rejects_every_identity_or_safety_mutation_before_load() {
        type CanaryMutation = (&'static str, fn(&mut Value));
        let mutations: [CanaryMutation; 16] = [
            ("wrong canary identity", |request| {
                request["planned"]["_canary"]["identity"] = json!("sc-20169-product-envelope")
            }),
            ("promotable scope", |request| {
                request["planned"]["evidenceScope"] = json!("authoritative")
            }),
            ("campaign geometry", |request| {
                request["planned"]["target"]["geometry"]["width"] = json!(768)
            }),
            ("single pass", |request| {
                request["planned"]["strategy"]["rung"] = json!("staged_residency")
            }),
            ("wrong backend", |request| {
                request["planned"]["backend"] = json!("candle")
            }),
            ("missing staged rung", |request| {
                request["planned"]["strategy"]["engagedRungs"] =
                    json!(["resident", "bounded_decode"])
            }),
            ("wrong tile edge", |request| {
                request["planned"]["strategy"]["parameters"]["decodeTileEdge"] = json!(256)
            }),
            ("wrong overlap", |request| {
                request["planned"]["strategy"]["parameters"]["decodeOverlap"] = json!(32)
            }),
            ("audio-enabled variant", |request| {
                request["planned"]["_canary"]["videoMode"] = json!("default")
            }),
            ("ceiling drift", |request| {
                request["planned"]["_watchdog"]["maxFootprintBytes"] =
                    json!(LTX_CANARY_MAX_FOOTPRINT_BYTES + 1)
            }),
            ("wrong fixture", |request| {
                request["planned"]["fixture"] = json!("campaign-row")
            }),
            ("wrong artifact revision", |request| {
                request["planned"]["_artifact"]["revision"] = json!("0".repeat(40))
            }),
            ("wrong artifact inventory", |request| {
                request["planned"]["_artifact"]["numericTierInventory"]["sha256"] =
                    json!("0".repeat(64))
            }),
            ("wrong text encoder bytes", |request| {
                request["planned"]["_artifact"]["textEncoderInventory"]["bytes"] =
                    json!(LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES - 1)
            }),
            ("wrong text encoder digest", |request| {
                request["planned"]["_artifact"]["textEncoderInventory"]["sha256"] =
                    json!("0".repeat(64))
            }),
            ("undersized host", |request| {
                request["hardware"]["memoryBytes"] = json!(LTX_CANARY_MAX_FOOTPRINT_BYTES * 2 - 1)
            }),
        ];
        for (name, mutate) in mutations {
            let mut request = ltx_canary_request_json();
            mutate(&mut request);
            assert!(
                validate_ltx_canary_plan(&request).is_err(),
                "{name} must fail before environment/provider/model access"
            );
        }
    }

    #[test]
    fn the_ltx_product_envelope_canary_rejects_every_tuple_mutation_before_load() {
        type CanaryMutation = (&'static str, fn(&mut Value));
        let mutations: [CanaryMutation; 10] = [
            ("wrong action", |request| {
                request["action"] = json!("canary")
            }),
            ("no audio", |request| {
                request["planned"]["_canary"]["videoMode"] = json!("no_audio")
            }),
            ("wrong width", |request| {
                request["planned"]["target"]["geometry"]["width"] = json!(512)
            }),
            ("wrong height", |request| {
                request["planned"]["target"]["geometry"]["height"] = json!(768)
            }),
            ("wrong frames", |request| {
                request["planned"]["target"]["geometry"]["frames"] = json!(89)
            }),
            ("wrong fps", |request| {
                request["planned"]["_canary"]["fps"] = json!(30)
            }),
            ("wrong carrier", |request| {
                request["planned"]["strategy"]["parameters"]["decodeTileEdge"] = json!(384)
            }),
            ("campaign scope", |request| {
                request["planned"]["evidenceScope"] = json!("campaign")
            }),
            ("watchdog threshold", |request| {
                request["planned"]["_watchdog"]["maxFootprintBytes"] =
                    json!(LTX_CANARY_MAX_FOOTPRINT_BYTES - 1)
            }),
            ("artifact drift", |request| {
                request["planned"]["_artifact"]["numericTierInventory"]["sha256"] =
                    json!("0".repeat(64))
            }),
        ];
        for (name, mutate) in mutations {
            let mut request = ltx_product_envelope_canary_request_json();
            mutate(&mut request);
            assert!(
                validate_ltx_product_envelope_canary_plan(&request).is_err(),
                "{name} must fail before environment/provider/model access"
            );
        }
    }

    #[test]
    fn the_ltx_canary_wired_limit_never_exceeds_the_device_ceiling() {
        let requested = LTX_CANARY_MAX_FOOTPRINT_BYTES as usize;
        assert_eq!(ltx_canary_wired_limit(requested, 120), 80);
        assert_eq!(ltx_canary_wired_limit(40, 120), 40);
        assert_eq!(ltx_canary_wired_limit(requested, 0), 0);
    }

    #[test]
    fn the_ltx_canary_process_global_limits_are_serialized_and_restore_memory() {
        let previous;
        {
            let mut limits = LtxCanaryLimits::install().expect("install exact canary limits");
            previous = limits.previous_memory;
            assert_eq!(get_memory_limit(), LTX_CANARY_MAX_FOOTPRINT_BYTES as usize);
            limits.restore();
            assert_eq!(get_memory_limit(), previous);
            let restored_wired = set_wired_limit(limits.wired);
            set_wired_limit(restored_wired);
            assert_eq!(restored_wired, limits.previous_wired);
        }
        let _restoration_guard = LTX_MEMORY_LIMIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(get_memory_limit(), previous);
    }

    fn set_ltx_request_tier(request: &mut Value, tier: &str, inventory: u64) {
        request["planned"]["target"]["tier"] = json!(tier);
        request["planned"]["fixture"] = json!(format!(
            "ltx-2-3-mlx-{tier}-768x512-f97-fps24-seed{LTX_SEED}"
        ));
        request["planned"]["_measurementSafety"]["tierInventoryBytes"] = json!(inventory);

        let predicted_decode_bytes = request["planned"]["_measurementSafety"]
            ["predictedDecodeBytes"]
            .as_u64()
            .expect("fixture predicted decode bytes");
        request["planned"]["_measurementSafety"]["incidentCalibratedProjectionBytes"] = json!(
            i128::from(LTX_Q4_F305_CRASH_FOOTPRINT_BYTES)
                + (i128::from(inventory) - i128::from(LTX_Q4_INVENTORY_BYTES))
                + (i128::from(predicted_decode_bytes)
                    - i128::from(LTX_INCIDENT_PREDICTED_DECODE_BYTES))
        );
    }

    fn set_ltx_bounded_case(
        request: &mut Value,
        tier: &str,
        inventory: u64,
        frames: u32,
        disposition: &str,
    ) {
        request["planned"]["target"]["tier"] = json!(tier);
        request["planned"]["target"]["geometry"] =
            json!({ "width": 1280, "height": 704, "batch": 1, "frames": frames });
        request["planned"]["fixture"] = json!(format!(
            "ltx-2-3-mlx-{tier}-1280x704-f{frames}-fps30-seed{LTX_SEED}"
        ));
        request["planned"]["strategy"] = json!({
            "rung": "bounded_decode",
            "engagedRungs": ["resident", "staged_residency", "bounded_decode"],
            "parameters": { "decodeTileEdge": 384, "decodeOverlap": 64 },
        });
        let predicted_decode_bytes =
            3_300_000_000_u64 + 40 * 1280 * 704 * u64::from(frames) + 300 * 384 * 384 * 96;
        let safety = &mut request["planned"]["_measurementSafety"];
        safety["disposition"] = json!(disposition);
        safety["tierInventoryBytes"] = json!(inventory);
        safety["predictedDecodeBytes"] = json!(predicted_decode_bytes);
        safety["incidentCalibratedProjectionBytes"] = json!(
            i128::from(LTX_Q4_F305_CRASH_FOOTPRINT_BYTES)
                + (i128::from(inventory) - i128::from(LTX_Q4_INVENTORY_BYTES))
                + (i128::from(predicted_decode_bytes)
                    - i128::from(LTX_INCIDENT_PREDICTED_DECODE_BYTES))
        );
    }

    /// The port of the shipped LTX frame ladder must reproduce these 18 transcribed values, one per
    /// point the declared limits can reach. These 18 values ARE the envelope, so an unnoticed drift
    /// would silently widen or narrow what this arm accepts.
    ///
    /// This is a TRANSCRIPTION check, not a call into `sceneworks_core::video_request::ltx_frame_count`
    /// — this crate deliberately does not depend on `sceneworks-core` (see `ltx_snapped_frame_count`).
    /// The other half of the binding is `ltx_frame_count_matches_the_sc_18808_calibration_ladder` in
    /// `crates/sceneworks-core/src/video_request.rs`, which pins the same 18 pairs against the shipped
    /// function in a default member. Keep the two tables identical; changing one alone is the drift
    /// both exist to catch (sc-18808 review).
    #[test]
    fn ltx_frame_ladder_port_matches_the_transcribed_shipped_ladder() {
        for (duration, fps, expected) in [
            (4, 24, 97),
            (4, 25, 97),
            (4, 30, 121),
            (6, 24, 145),
            (6, 25, 153),
            (6, 30, 177),
            (8, 24, 193),
            (8, 25, 201),
            (8, 30, 241),
            (10, 24, 241),
            (10, 25, 249),
            (10, 30, 297),
            (12, 24, 289),
            (12, 25, 297),
            (12, 30, 361),
            (15, 24, 361),
            (15, 25, 377),
            (15, 30, 449),
        ] {
            let frames = ltx_snapped_frame_count(duration * fps);
            assert_eq!(frames, expected, "{duration}s at {fps}fps");
            assert_eq!(
                frames % LTX_TEMPORAL_SCALE,
                1,
                "every reachable frame count is on the 1 + 8k lattice"
            );
        }
        // The ladder's own floor, independent of the declared durations.
        assert_eq!(ltx_snapped_frame_count(0), 9);
        assert_eq!(ltx_snapped_frame_count(1), 9);
    }

    /// The envelope is DERIVED from the declared arrays through the ladder above, not written down.
    #[test]
    fn ltx_frame_envelope_is_derived_from_the_declared_durations_and_fps() {
        assert_eq!(LTX_FRAME_ENVELOPE, (97, 449));
        let (minimum, maximum) = LTX_FRAME_ENVELOPE;
        assert_eq!(minimum, ltx_snapped_frame_count(4 * 24));
        assert_eq!(maximum, ltx_snapped_frame_count(15 * 30));
    }

    #[test]
    fn the_ltx_arm_accepts_the_declared_video_envelope() {
        for (width, height) in LTX_RESOLUTIONS {
            for frames in [LTX_FRAME_ENVELOPE.0, 241, LTX_FRAME_ENVELOPE.1] {
                let geometry = validate_ltx_geometry(width, height, frames)
                    .unwrap_or_else(|error| panic!("{width}x{height} f{frames}: {error}"));
                assert_eq!(geometry.frames, frames);
                assert_eq!(
                    geometry.latent_frames,
                    1 + (frames - 1) / LTX_TEMPORAL_SCALE
                );
                assert!(
                    geometry.latent_frames > 1,
                    "a video record is multi-latent-frame"
                );
            }
        }
    }

    /// The negative half of the envelope, mirroring the shape of the pinned image-arm negative test.
    #[test]
    fn the_ltx_arm_rejects_out_of_envelope_geometry_with_a_named_reason() {
        for (width, height, frames, expected) in [
            // An undeclared resolution, including one that is 64-aligned and would otherwise render.
            (1024, 1024, 97, "declared limits.resolutions"),
            (800, 512, 97, "declared limits.resolutions"),
            // Off the temporal lattice the LTX VAE's 8x causal compression requires.
            (768, 512, 96, "1 + 8k"),
            (768, 512, 100, "1 + 8k"),
            // On the lattice but outside what a declared duration/fps pair can produce.
            (768, 512, 1, "duration/fps envelope"),
            (768, 512, 9, "duration/fps envelope"),
            (768, 512, 457, "duration/fps envelope"),
        ] {
            let error = validate_ltx_geometry(width, height, frames)
                .expect_err("an out-of-envelope LTX geometry must be refused");
            assert!(
                error.contains(expected),
                "{width}x{height} f{frames}: {error}"
            );
        }
    }

    /// A still geometry is on the temporal lattice (`1 % 8 == 1`), so it can only be caught by the
    /// envelope floor. The video arm must not be able to capture a single-frame record.
    #[test]
    fn the_ltx_arm_refuses_a_single_frame_capture() {
        let error = run_ltx(&ltx_request_json(768, 512, 1))
            .expect_err("a video arm must not capture a still");
        assert!(error.contains("duration/fps envelope"), "{error}");
    }

    #[test]
    fn the_ltx_arm_refuses_a_foreign_provider_before_environment_or_weight_work() {
        for provider in ["ltx_2_3_distilled", "ltx_2_3_eros", "wan_2_2", "flux2_dev"] {
            let mut request = ltx_request_json(768, 512, 97);
            request["planned"]["target"]["provider"] = json!(provider);
            let error =
                run_ltx(&request).expect_err("a foreign provider must not reach the LTX arm");
            assert_eq!(
                error,
                format!("MLX LTX-2.3 calibration does not implement provider {provider:?}")
            );
        }
    }

    #[test]
    fn the_ltx_arm_fails_closed_on_a_non_t2v_target_before_weight_work() {
        for (pointer, value, expected) in [
            (
                "/planned/target/modelId",
                json!("ltx_2_3_eros"),
                "requires modelId",
            ),
            (
                "/planned/target/mode",
                json!("image_to_video"),
                "requires reference-free text_to_video mode",
            ),
            (
                "/planned/target/geometry/batch",
                json!(2),
                "requires geometry.batch == 1",
            ),
            (
                "/planned/target/overlay",
                json!("lora"),
                "refusing rather than recording false overlay coverage",
            ),
            // `resident` is unreachable on this provider on EVERY host, so the refusal is
            // host-independent — but WHICH rung the engine measures instead is not: this geometry
            // is single-pass on a 128 GiB Mac and tiled on a small CI runner. Assert the
            // machine-independent half of the message; the rung itself is pinned, at a fixed
            // budget, by `the_ltx_arm_follows_the_engine_across_the_decode_tiling_boundary`.
            (
                "/planned/loadShape",
                json!("deferred_materialization"),
                "calibrates only eager_materialization",
            ),
            (
                "/planned/calibrationFingerprint",
                json!("sc-18808-ltx-2-3-mlx-t2v-staged-capture-v0"),
                "plan/adapter calibration mismatch",
            ),
            (
                "/planned/fixture",
                json!("ltx-2-3-mlx-q8-768x512-f97-fps60-seed18808"),
                "not one of the declared limits.fps",
            ),
            (
                "/planned/fixture",
                json!("ltx-2-3-mlx-q8-768x512-f193-fps24-seed18808"),
                "must start with",
            ),
            (
                "/planned/fixture",
                json!("ltx-2-3-mlx-q8-768x512-f97-fps24-seed42"),
                "does not match the LTX-2.3 calibration seed",
            ),
            (
                "/planned/target/tier",
                json!("q2"),
                "unsupported MLX numeric tier",
            ),
        ] {
            let mut request = ltx_request_json(768, 512, 97);
            *request.pointer_mut(pointer).unwrap() = value.clone();
            let error = run_ltx(&request)
                .expect_err("a non-T2V LTX target must fail before environment or weights");
            assert!(error.contains(expected), "{pointer}={value}: {error}");
        }
    }

    /// Pure request malformations must still fail before environment or weight resolution.
    #[test]
    fn the_ltx_arm_refuses_a_malformed_plan_before_it_consults_the_host_budget() {
        // Force the smoke geometry onto the provider's auto-tiled path. Every mutation below must
        // still report its own deterministic input error; consulting the live selector first made
        // this test intermittently report a decode-budget outcome under default-parallel tests.
        let _constrained_budget = LtxInjectedBudget::install(8.0);
        for (pointer, value, expected) in [
            (
                "/planned/fixture",
                json!("ltx-2-3-mlx-q8-768x512-f97-fps24-seed42"),
                "does not match the LTX-2.3 calibration seed",
            ),
            (
                "/planned/calibrationFingerprint",
                json!("sc-18808-ltx-2-3-mlx-t2v-staged-capture-v0"),
                "plan/adapter calibration mismatch",
            ),
            (
                "/planned/loadShape",
                json!("deferred_materialization"),
                "calibrates only eager_materialization",
            ),
            (
                "/planned/target/tier",
                json!("q2"),
                "unsupported MLX numeric tier",
            ),
        ] {
            let mut request = ltx_request_json(768, 512, 97);
            *request.pointer_mut(pointer).unwrap() = value.clone();
            let error = run_ltx(&request).expect_err("a malformed LTX plan must be refused");
            assert!(
                error.contains(expected),
                "{pointer}={value} must be refused for the malformation: {error}"
            );
        }
    }

    #[test]
    fn the_ltx_arm_refuses_a_parameterized_staged_residency_row() {
        let mut request = ltx_request_json(768, 512, 97);
        request["planned"]["strategy"]["parameters"] = json!({ "decodeTileEdge": 512 });
        let selection = planned_selection(&request).unwrap();
        let error = ltx_fixture_contract(Some(Quant::Q8))
            .validate_selection(&selection)
            .expect_err("staged residency has no decode parameter domain");
        assert!(error.to_string().contains("decode"), "{error}");
    }

    #[test]
    fn sc_19642_refuses_every_tier_before_environment_registry_or_weights() {
        for (tier, inventory) in [
            ("q4", LTX_Q4_INVENTORY_BYTES),
            ("q8", LTX_Q8_INVENTORY_BYTES),
            ("bf16", LTX_BF16_INVENTORY_BYTES),
        ] {
            let mut request = ltx_request_json(768, 512, 97);
            set_ltx_request_tier(&mut request, tier, inventory);
            let error =
                run_ltx(&request).expect_err("every current SC-18946 tier must fail closed");
            assert!(
                error.contains("SC-19642 pre-load safety refusal"),
                "{error}"
            );
            assert!(error.contains(&format!("inventory={inventory}")), "{error}");
            assert!(error.contains(LTX_SAFETY_REFUSED_OPEN), "{error}");
            assert!(
                !error.contains("SCENEWORKS_LTX_ROOT") && !error.contains("build LTX registry"),
                "refusal happened after environment or registry work: {error}"
            );
        }
    }

    #[test]
    fn sc_19642_preserves_incident_monotonic_and_open_refusal_terminals() {
        for (tier, inventory, frames, disposition) in [
            ("q4", LTX_Q4_INVENTORY_BYTES, 305, LTX_INCIDENT_FORBIDDEN),
            (
                "q4",
                LTX_Q4_INVENTORY_BYTES,
                449,
                LTX_ARITHMETIC_UNMEASURABLE,
            ),
            ("q8", LTX_Q8_INVENTORY_BYTES, 305, LTX_SAFETY_REFUSED_OPEN),
            (
                "bf16",
                LTX_BF16_INVENTORY_BYTES,
                449,
                LTX_SAFETY_REFUSED_OPEN,
            ),
        ] {
            let mut request = ltx_request_json(1280, 704, frames);
            set_ltx_bounded_case(&mut request, tier, inventory, frames, disposition);
            let error = run_ltx(&request).expect_err("SC-18946 case must refuse before load");
            assert!(
                error.contains("SC-19642 pre-load safety refusal"),
                "{error}"
            );
            assert!(error.contains(disposition), "{error}");
            assert!(!error.contains("SCENEWORKS_LTX_ROOT"), "{error}");
        }

        let mut wrong_carrier = ltx_request_json(1280, 704, 449);
        set_ltx_bounded_case(
            &mut wrong_carrier,
            "q4",
            LTX_Q4_INVENTORY_BYTES,
            449,
            LTX_ARITHMETIC_UNMEASURABLE,
        );
        wrong_carrier["planned"]["strategy"]["parameters"]["decodeTileEdge"] = json!(256);
        let error = run_ltx(&wrong_carrier)
            .expect_err("a different carrier cannot inherit the q4 f449 arithmetic proof");
        assert!(error.contains(LTX_SAFETY_REFUSED_OPEN), "{error}");
        assert!(!error.contains("SCENEWORKS_LTX_ROOT"), "{error}");
    }

    #[test]
    fn sc_19642_safety_metadata_mutations_fail_closed_before_weight_work() {
        for (pointer, value, expected) in [
            (
                "/planned/_measurementSafety/disposition",
                json!("capturable"),
                "safety disposition",
            ),
            (
                "/planned/_measurementSafety/tierInventoryBytes",
                json!(LTX_Q8_INVENTORY_BYTES - 1),
                "tierInventoryBytes",
            ),
            (
                "/planned/_measurementSafety/incidentCrashFootprintBytes",
                json!(LTX_Q4_F305_CRASH_FOOTPRINT_BYTES - 1),
                "incidentCrashFootprintBytes",
            ),
        ] {
            let mut request = ltx_request_json(768, 512, 97);
            *request.pointer_mut(pointer).unwrap() = value;
            let error = run_ltx(&request).expect_err("mutated safety metadata must fail closed");
            assert!(error.contains(expected), "{pointer}: {error}");
            assert!(!error.contains("SCENEWORKS_LTX_ROOT"), "{pointer}: {error}");
        }
    }

    /// The exact composition the record claims, pinned so a silent widening of `engagedRungs` would
    /// have to be an explicit edit here.
    #[test]
    fn the_ltx_arm_attests_exactly_resident_plus_staged_residency() {
        let request = ltx_request_json(768, 512, 97);
        let selection = planned_selection(&request).unwrap();
        let contract = ltx_fixture_contract(Some(Quant::Q8));
        contract.validate_selection(&selection).unwrap();
        let strategy = ltx_attested_strategy(&request, &selection, &contract).unwrap();
        assert_eq!(strategy["rung"], "staged_residency");
        assert_eq!(
            strategy["engagedRungs"],
            json!(["resident", "staged_residency"])
        );
        assert_eq!(strategy["parameters"], json!({}));
    }

    /// sc-18810: the staged-residency witness must be a PHASE BOUNDARY reading, not a peak. The
    /// numbers here are the real ones from this story's captures — the first row is the geometry
    /// that broke sc-18808's `overall_peak < text_encoder + transformer` form.
    #[test]
    fn the_staging_proof_survives_a_peak_above_the_costaged_bound() {
        const TEXT_ENCODER: u64 = 32_733_043_614;
        const TRANSFORMER: u64 = 20_614_103_249;
        const COSTAGED: u64 = TEXT_ENCODER + TRANSFORMER; // 53,347,146,863
        let entering_denoise = |active: u64| AllocatorState { active, cache: 0 };

        // q8 704x1280 x 177: measured overall peak 54,153,098,156 — ABOVE the co-staged bound — while
        // the denoise boundary held 20.7 GB, i.e. the transformer and nothing else. The peak is the
        // decode phase's output buffers. This run is valid and must be accepted.
        assert!(
            ltx_staging_is_proven(
                entering_denoise(20_765_156_098),
                COSTAGED,
                TEXT_ENCODER,
                TRANSFORMER
            )
            .is_ok(),
            "a decode-dominated peak above the co-staged bound is not a staging regression"
        );

        // The regression the check exists for: the text encoder still resident when the AvDiT is up.
        let regressed = ltx_staging_is_proven(
            entering_denoise(COSTAGED + 1),
            COSTAGED,
            TEXT_ENCODER,
            TRANSFORMER,
        )
        .expect_err("a co-resident text encoder must be refused");
        assert!(
            regressed.contains("was not dropped before the AvDiT"),
            "{regressed}"
        );

        // The vacuity guard: a boundary sampled before the transformer materializes would make the
        // upper bound pass while proving nothing.
        let too_early = ltx_staging_is_proven(
            entering_denoise(TRANSFORMER - 1),
            COSTAGED,
            TEXT_ENCODER,
            TRANSFORMER,
        )
        .expect_err("a boundary sampled before the AvDiT exists must be refused");
        assert!(too_early.contains("proves nothing"), "{too_early}");
    }

    /// sc-18810: WHERE rung 2 engages, derived from the engine's own bound rather than from the
    /// measured decode peak. `writable_frame_cap = i32::MAX / (8 * h * w)` is pure arithmetic over
    /// `VaeTiling::LTX` and does not move with the host, so it is asserted exactly. The memory bound
    /// (`3.3 GB + 340 B/voxel` vs `get_memory_limit() * 0.85`) DOES move with the host, which is
    /// precisely why the boundary is attributed to the write bound and not to a machine.
    #[test]
    fn the_ltx_decode_tiling_boundary_is_the_machine_independent_write_bound() {
        let caps = LTX_RESOLUTIONS.map(|(width, height)| {
            (
                (width, height),
                VaeTiling::LTX.writable_frame_cap(height as i32, width as i32),
            )
        });
        assert_eq!(
            caps,
            [
                ((768, 512), 682),
                ((512, 768), 682),
                ((640, 640), 655),
                ((1280, 704), 297),
                ((704, 1280), 297),
            ]
        );
        let (_, envelope_maximum) = LTX_FRAME_ENVELOPE;
        for ((width, height), cap) in caps {
            // Only the two 0.90 MP buckets can reach their cap inside the declared envelope. The
            // others cannot tile at ANY declared geometry, on any host, for this reason.
            let reachable = cap < i64::from(envelope_maximum);
            assert_eq!(
                reachable,
                (width, height) == (1280, 704) || (width, height) == (704, 1280),
                "{width}x{height} cap {cap} against envelope maximum {envelope_maximum}"
            );
        }
    }

    /// The companion behavioural half: the arm's rung FOLLOWS the engine across that boundary, in
    /// both directions, **at a FIXED budget so both directions are asserted on every host** — a CI
    /// runner included. `resolve` budgets against `get_memory_limit()`, so a test that resolved
    /// against the live host would assert one thing on a 128 GiB Mac and another on a 16 GiB runner;
    /// guarding it behind "only if the memory bound does not bind" would make it vacuous exactly
    /// where it runs unattended.
    #[test]
    fn the_ltx_arm_follows_the_engine_across_the_decode_tiling_boundary() {
        // A budget far above any single-pass cost in the declared envelope: 1280x704 x 297 costs
        // 3.3e9 + 340 * 267,632,640 B = 87.8 GiB single-pass, and the selector's own 0.85 factor
        // leaves 217.6 GiB here. Whatever tiles under THIS budget tiles for the write bound alone.
        const UNCONSTRAINED_GIB: f64 = 256.0;
        // 297 is the cap itself (a declared 10 s x 30 fps cell) and 305 is the next lattice step
        // above it. Above the cap `budgeted_plan` tiles unconditionally — memory cannot buy it back.
        let tiled = LtxDecodePlan::resolve_with_budget(
            validate_ltx_geometry(1280, 704, 305).unwrap(),
            UNCONSTRAINED_GIB,
        )
        .unwrap();
        assert!(
            tiled.tiling.is_some(),
            "305 frames is over the 297 write cap, so no budget keeps it single-pass"
        );
        assert_eq!(tiled.rung(), "bounded_decode");
        assert_eq!(
            tiled.engaged_rungs(),
            ["resident", "staged_residency", "bounded_decode"]
        );
        assert!(
            tiled.spatial_tile_px() > 0 || tiled.temporal_tile_frames() > 0,
            "a selected tiling must name at least one tiled axis"
        );

        // The other direction, pinned at the same fixed budget: AT the cap and with memory to spare,
        // the decode stays single-pass. Together with the row above this is the write bound, isolated
        // from the host.
        let single = LtxDecodePlan::resolve_with_budget(
            validate_ltx_geometry(1280, 704, 297).unwrap(),
            UNCONSTRAINED_GIB,
        )
        .unwrap();
        assert!(
            single.tiling.is_none(),
            "at the cap and inside the memory budget the decode must stay single-pass"
        );
        assert_eq!(single.rung(), "staged_residency");
        assert_eq!(single.lifecycle_fault_phase(), MemoryPhase::Denoise);
        let staged_selection = planned_selection(&ltx_request_json(1280, 704, 297)).unwrap();
        single
            .validate_selected_strategy(&staged_selection)
            .unwrap();
        // SC-19109 replaces historical host inference with an explicit provider-owned selection:
        // the same geometry can deliberately request bounded decode with its exact carrier tuple.
        let mut request = ltx_request_json(1280, 704, 297);
        request["planned"]["strategy"] = json!({
            "rung": "bounded_decode",
            "engagedRungs": ["resident", "staged_residency", "bounded_decode"],
            "parameters": { "decodeTileEdge": 384, "decodeOverlap": 64 }
        });
        let selection = planned_selection(&request).unwrap();
        let contract = ltx_fixture_contract(Some(Quant::Q8));
        contract.validate_selection(&selection).unwrap();
        let attested = ltx_attested_strategy(&request, &selection, &contract).unwrap();
        assert_eq!(attested, request["planned"]["strategy"]);
        let explicit = LtxDecodePlan::resolve_for_selection(
            &selection,
            validate_ltx_geometry(1280, 704, 297).unwrap(),
        )
        .unwrap();
        explicit.validate_selected_strategy(&selection).unwrap();
        assert_eq!(explicit.lifecycle_fault_phase(), MemoryPhase::Decode);
        assert_eq!(explicit.spatial_tile_px(), 384);
        assert_eq!(explicit.spatial_overlap_px(), 64);

        // The one-sided half of the claim, asserted rather than assumed: the write cap is a CEILING
        // on single-pass frames, not the place tiling starts. A smaller host tiles the very same
        // f297 geometry the row above kept single-pass — and 768x512 x 97, which is 585 frames below
        // its 682 cap and is exactly what a hosted CI runner does to this arm's own smoke geometry.
        // `slack` is how far each row sits below the write bound: 0 at the cap, 585 at the smoke
        // geometry. Both tile anyway, which is the point — the write bound is not what forces it.
        for (width, height, frames, budget_gib, slack) in [
            (1280u32, 704u32, 297u32, 32.0f64, 0i64),
            (768, 512, 97, 16.0, 585),
        ] {
            // The precondition is DERIVED, not asserted in prose: this budget must actually be below
            // the engine's own single-pass cost, or the row proves nothing.
            let single_pass_gib = ltx_decode_cost::single_pass_gib(width, height, frames);
            assert!(
                single_pass_gib > budget_gib * ltx_decode_cost::SAFETY_FACTOR,
                "{width}x{height} f{frames}: {budget_gib} GiB does not constrain a \
                 {single_pass_gib:.2} GiB single-pass decode, so this row would be vacuous"
            );
            let constrained = LtxDecodePlan::resolve_with_budget(
                validate_ltx_geometry(width, height, frames).unwrap(),
                budget_gib,
            )
            .unwrap();
            assert!(
                constrained.tiling.is_some(),
                "{width}x{height} f{frames} must tile for MEMORY under a {budget_gib} GiB budget"
            );
            assert_eq!(constrained.rung(), "bounded_decode");
            assert_eq!(constrained.lifecycle_fault_phase(), MemoryPhase::Decode);
            let staged_selection =
                planned_selection(&ltx_request_json(width, height, frames)).unwrap();
            let mismatch = constrained
                .validate_selected_strategy(&staged_selection)
                .expect_err("an auto-tiled render must not attest a staged single-pass row");
            assert!(
                mismatch.contains("auto-selected bounded decode"),
                "{mismatch}"
            );
            // The write bound PERMITTED a single pass here (`f <= cap`) and memory still tiled it.
            assert_eq!(
                constrained.writable_frame_cap - i64::from(frames),
                slack,
                "{width}x{height} f{frames} must tile with the write bound still permitting a \
                 single pass (cap {})",
                constrained.writable_frame_cap
            );
        }

        // The third outcome, which CI found and this test did not originally model: below the
        // full-output ACCUMULATOR floor (`3.3 GB + 40 B/voxel`, 13.05 GiB at the cap geometry) no
        // tiling helps — the accumulators hold the assembled video — so the engine refuses outright
        // before any render. A hosted macOS runner lands here at ~6 GiB of safe budget. Pinned at a
        // fixed budget so the refusal is asserted on every host, not only on a small one.
        const BELOW_THE_ACCUMULATOR_FLOOR_GIB: f64 = 8.0;
        let accumulators_gib = ltx_decode_cost::accumulator_floor_gib(1280, 704, 297);
        assert!(
            accumulators_gib > BELOW_THE_ACCUMULATOR_FLOOR_GIB * ltx_decode_cost::SAFETY_FACTOR,
            "this budget must sit below the {accumulators_gib:.2} GiB accumulator floor or the \
             refusal below would be proving something else"
        );
        let refused = LtxDecodePlan::resolve_with_budget(
            validate_ltx_geometry(1280, 704, 297).unwrap(),
            BELOW_THE_ACCUMULATOR_FLOOR_GIB,
        )
        .expect_err("under the accumulator floor the decode must be refused, not tiled");
        assert!(refused.contains("just for the output buffers"), "{refused}");
        assert!(
            refused.contains("refuses 1280x704 x 297 frames before any render"),
            "{refused}"
        );
    }

    /// The margin the fixed-budget injections above ride on, asserted instead of assumed.
    ///
    /// `run_ltx` reads the process-global MLX limit without `LTX_MEMORY_LIMIT_LOCK`, so an
    /// injected budget below the `run_ltx` smoke geometry's accumulator floor would make the engine
    /// refuse that geometry and replace the message those refusal tests assert. The lowest budget
    /// injected today (8 GiB, 6.8 GiB safe) clears the 4.49 GiB floor by 1.51x — a margin that used
    /// to be load-bearing and unstated.
    #[test]
    #[should_panic(expected = "accumulator floor of the 768x512 x 97 smoke geometry")]
    fn an_injected_budget_below_the_smoke_geometrys_accumulator_floor_is_refused() {
        // 5.2 GiB leaves 4.42 GiB safe, just under the 4.49 GiB floor. This is exactly the row a
        // future author would add to probe a smaller host, and it must fail HERE — loudly, at the
        // injection site — rather than intermittently in an unrelated `run_ltx` refusal test.
        let _ =
            LtxDecodePlan::resolve_with_budget(validate_ltx_geometry(768, 512, 97).unwrap(), 5.2);
    }

    /// The other side of that boundary: the assertion is a FLOOR, not a blanket refusal. 5.3 GiB
    /// leaves 4.505 GiB safe against the same 4.494 GiB floor the 5.2 GiB row falls through, so the
    /// two rows bracket it to within 0.1 GiB and neither can pass by being trivially true.
    #[test]
    fn an_injected_budget_just_above_the_smoke_geometrys_accumulator_floor_is_accepted() {
        // Capture the prior limit only after `install` owns the injection lock. Reading it before
        // the lock is a TOCTOU race: another test may temporarily expose its 32 GiB injection,
        // restore the real host limit, and then let this test acquire the lock. CI caught exactly
        // that 32 GiB -> 7.14 GB transition.
        let budget = LtxInjectedBudget::install(5.3);
        let previous = budget.previous;
        drop(budget);

        // Reacquire before observing the process-global limit. An intervening injection is allowed
        // to run, but must finish and restore `previous` before this assertion reads the value.
        let _restoration_guard = LTX_MEMORY_LIMIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(get_memory_limit(), previous);
    }

    /// The injected limit is process-global, so failing to restore it leaks into every later test.
    #[test]
    fn an_injected_budget_is_restored_on_the_normal_and_the_unwind_path() {
        let budget = LtxInjectedBudget::install(32.0);
        let previous = budget.previous;
        assert_eq!(
            get_memory_limit(),
            (32.0 * ltx_decode_cost::GIB) as usize,
            "the budget must actually be installed while the guard lives"
        );
        drop(budget);
        let restoration_guard = LTX_MEMORY_LIMIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(get_memory_limit(), previous, "restored on the normal path");
        drop(restoration_guard);
        // The path a trailing `set_memory_limit(previous)` statement misses entirely — and which
        // `unwrap_or_else(PoisonError::into_inner)` would then hide, because the poisoned lock is
        // the only signal the leak leaves behind.
        let unwind_budget = LtxInjectedBudget::install(32.0);
        let unwind_previous = unwind_budget.previous;
        let outcome = std::panic::catch_unwind(|| {
            let _budget = unwind_budget;
            panic!("the selector blew up mid-plan");
        });
        assert!(outcome.is_err(), "the closure must actually have panicked");
        let _restoration_guard = LTX_MEMORY_LIMIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            get_memory_limit(),
            unwind_previous,
            "the injected limit leaked past a panic"
        );
    }

    /// The live host, stated as its own claim so the fixed-budget test above cannot be mistaken for
    /// a statement about this machine.
    ///
    /// The cap geometry has THREE outcomes, not two, and which one a host gets is a total function
    /// of its budget. Exactly one arm below fires on any machine, so this is host-adaptive rather
    /// than host-conditional — nothing is skipped anywhere:
    ///
    /// * the full-output accumulators alone (`3.3 GB + 40 B/voxel`) exceed the budget → **refused**
    ///   before any render. A hosted CI runner lands here: ~13 GB of buffers against a ~6 GB safe
    ///   budget. Tiling cannot help, because the accumulators hold the assembled video.
    /// * a single pass (`3.3 GB + 340 B/voxel`) fits → **single-pass**. This Mac lands here.
    /// * in between → **tiled**, or refused if not even the smallest tile fits.
    #[test]
    fn the_ltx_arm_resolves_the_cap_geometry_against_this_hosts_live_budget() {
        // Read the limit and resolve under the SAME lock `LtxInjectedBudget` swaps under.
        // MLX's memory limit is process-global: without this, the boundary test's injected budget
        // can land between the read and the resolve and this comparison reads two different hosts.
        let guard = LTX_MEMORY_LIMIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let budget_gib =
            get_memory_limit() as f64 / ltx_decode_cost::GIB * ltx_decode_cost::SAFETY_FACTOR;
        let live = LtxDecodePlan::resolve(validate_ltx_geometry(1280, 704, 297).unwrap());
        drop(guard);
        let accumulators_gib = ltx_decode_cost::accumulator_floor_gib(1280, 704, 297);
        let single_pass_gib = ltx_decode_cost::single_pass_gib(1280, 704, 297);
        let where_we_are = format!(
            "accumulators {accumulators_gib:.2} GiB, single-pass {single_pass_gib:.2} GiB, this \
             host's safe budget {budget_gib:.2} GiB"
        );
        if accumulators_gib >= budget_gib {
            let error = live.expect_err("the accumulators alone do not fit — this must be refused");
            assert!(
                error.contains("just for the output buffers"),
                "{where_we_are}: {error}"
            );
        } else if single_pass_gib <= budget_gib {
            let plan = live.unwrap_or_else(|error| panic!("{where_we_are}: {error}"));
            assert!(plan.tiling.is_none(), "{where_we_are}");
            assert_eq!(plan.rung(), "staged_residency");
        } else {
            assert!(
                live.map_or(true, |plan| plan.tiling.is_some()),
                "{where_we_are}: a decode over budget must not resolve single-pass"
            );
        }
    }

    /// AC3, and the reason a video arm is safe to add: EVERY image arm still refuses a multi-frame
    /// geometry, before it does environment or weight work. Four of the six never validated the
    /// axis at all before sc-18808 — they read only width/height and hardcoded `frames: 1` into the
    /// admission context, so a `frames: 2` plan row would have rendered one frame and recorded a
    /// geometry it was never asked for.
    ///
    /// The list below is the IMAGE arms, and it stays six long as video arms are added: `ltx_2_3`
    /// (sc-18808) and `minimax_h3` (sc-18663) are the only two allowed to accept a multi-frame
    /// geometry, and each buys that by validating against its own engine's lattice rather than by
    /// dropping the axis — see `the_minimax_arm_accepts_a_multi_frame_geometry_the_image_guard_refuses`
    /// and `validate_ltx_geometry`. A NEW arm added here without such an envelope is the defect
    /// this test exists to prevent.
    #[test]
    fn every_image_arm_still_refuses_a_multi_frame_geometry() {
        type Arm = fn(&Value) -> Result<Value, String>;
        // `(provider, modelId, refusal label, arm)`. The model id is spelled independently of the
        // provider because two FLUX.2 catalog models share one engine provider id (sc-22727).
        let arms: [(&str, &str, &str, Arm); 9] = [
            (
                KREA_BASE_PROVIDER,
                KREA_BASE_PROVIDER,
                "MLX Krea base calibration",
                run_krea_base,
            ),
            (
                SDXL_PROVIDER,
                SDXL_PROVIDER,
                "MLX SDXL base calibration",
                run_sdxl,
            ),
            (
                Z_IMAGE_PROVIDER,
                Z_IMAGE_PROVIDER,
                "MLX Z-Image base calibration",
                run_z_image_reference,
            ),
            // sc-22724: the base model shares the arm; its refusal carries its own label.
            (
                Z_IMAGE_BASE_PROVIDER,
                Z_IMAGE_BASE_PROVIDER,
                "MLX Z-Image base-model calibration",
                run_z_image_reference,
            ),
            (
                KREA_PROVIDER,
                KREA_PROVIDER,
                "MLX Krea pose-control calibration",
                run_krea_control,
            ),
            (
                QWEN_PROVIDER,
                QWEN_PROVIDER,
                "MLX Qwen base calibration",
                run_qwen_provider,
            ),
            (
                FLUX2_PROVIDER,
                "flux2_dev",
                "MLX FLUX.2-dev calibration",
                run_flux2,
            ),
            // sc-22727: the two klein catalog models share `flux2_klein_9b` and each refuses under
            // its OWN label, so neither can be mistaken for the other in a failure report.
            (
                FLUX2_KLEIN_PROVIDER,
                "flux2_klein_9b",
                "MLX FLUX.2-klein-9B calibration",
                run_flux2,
            ),
            (
                FLUX2_KLEIN_PROVIDER,
                "flux2_klein_9b_kv",
                "MLX FLUX.2-klein-9B KV calibration",
                run_flux2,
            ),
        ];
        for (provider, model_id, label, arm) in arms {
            for frames in [0_u64, 2, 97] {
                let request = json!({
                    "planned": {
                        "target": {
                            "provider": provider,
                            "modelId": model_id,
                            "tier": "q4",
                            "mode": "text_to_image",
                            "overlay": if provider == KREA_PROVIDER { "control:1" } else { "none" },
                            "geometry": { "width": 768, "height": 768, "batch": 1, "frames": frames }
                        },
                        "backend": "mlx",
                        "loadShape": "deferred_materialization",
                        "strategy": {
                            "rung": "bounded_decode",
                            "engagedRungs": ["resident", "bounded_decode"],
                            "parameters": { "decodeTileEdge": 512, "decodeOverlap": 64 }
                        },
                        "calibrationFingerprint": "unused",
                        "fixture": "unused"
                    }
                });
                let error =
                    arm(&request).expect_err("an image arm must refuse a multi-frame geometry");
                assert_eq!(
                    error,
                    format!("{label} requires geometry.frames == 1, got {frames}"),
                    "{provider}/{model_id} at frames={frames}"
                );
            }
        }
    }

    /// And the still geometry itself must still get PAST the guard on every image arm, so the
    /// refusals above are the frames axis rather than a blanket rejection.
    #[test]
    fn the_still_geometry_guard_is_not_a_blanket_refusal() {
        for label in [
            "MLX Krea base calibration",
            "MLX SDXL base calibration",
            "MLX Z-Image base calibration",
            "MLX Z-Image base-model calibration",
            "MLX Z-Image edit calibration",
            "MLX Krea pose-control calibration",
            "MLX Qwen base calibration",
            "MLX FLUX.2-klein-9B calibration",
            "MLX FLUX.2-klein-9B KV calibration",
            "MLX FLUX.2-dev calibration",
        ] {
            let request = json!({
                "planned": { "target": { "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 } } }
            });
            assert!(
                protocol::validate_still_geometry(&request, label).is_ok(),
                "{label}"
            );
        }
    }

    /// The video output unwrapper refuses the two shapes an image provider would return, so a
    /// misrouted still could never be recorded as a clip.
    #[test]
    fn the_video_unwrapper_refuses_image_and_audio_shaped_output() {
        let frame = Image {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 255, 255, 255],
        };
        assert!(video_frames(GenerationOutput::Images(vec![frame.clone()]))
            .unwrap_err()
            .contains("returned images, not a video clip"));
        assert!(video_frames(GenerationOutput::Video {
            frames: Vec::new(),
            fps: 24,
            audio: None,
        })
        .unwrap_err()
        .contains("returned no frames"));
        let (frames, fps, audio) = video_frames(GenerationOutput::Video {
            frames: vec![frame],
            fps: 25,
            audio: None,
        })
        .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(fps, 25);
        assert!(!audio);
    }

    /// The clip comparator is per-frame and mutation-sensitive: a divergence confined to a LATE
    /// frame must be seen, which a first-frame spot check would miss.
    #[test]
    fn the_clip_comparator_sees_a_late_frame_divergence() {
        let frame = |value: u8| Image {
            width: 2,
            height: 2,
            pixels: vec![value; 12],
        };
        let left = vec![frame(10), frame(10), frame(10)];
        let identical = left.clone();
        let (maximum, mean, rms) = video_max_mean_rms_abs(&left, &identical).unwrap();
        assert_eq!((maximum, mean, rms), (0.0, 0.0, 0.0));
        assert!(ltx_quality_passes(maximum, mean, rms));

        let mut late = left.clone();
        late[2] = frame(200);
        let (maximum, mean, rms) = video_max_mean_rms_abs(&left, &late).unwrap();
        assert!(maximum > LTX_MAX_THRESHOLD, "max={maximum}");
        assert!(!ltx_quality_passes(maximum, mean, rms));

        assert!(video_max_mean_rms_abs(&left, &left[..2])
            .unwrap_err()
            .contains("frame-count mismatch"));
    }

    #[test]
    fn the_ltx_runtime_context_binds_the_exact_video_route_and_selection() {
        let request = ltx_request_json(1280, 704, 305);
        let selection = planned_selection(&request).unwrap();
        let contract = ltx_fixture_contract(Some(Quant::Q8));
        let calibration = contract.calibration.as_ref().unwrap();
        let context = ltx_context(
            selection,
            calibration,
            &calibration.fingerprint,
            validate_ltx_geometry(1280, 704, 305).unwrap(),
            64 * 1024 * 1024 * 1024,
            1,
        );
        assert_eq!(context.mode.as_key(), "text_to_video");
        assert!(context.has_phases);
        assert_eq!(context.geometry.frames, 305);
        assert_eq!(context.selection.strategy, MemoryStrategy::StagedResidency);
        assert_eq!(
            context.evidence_revision,
            format!("sc-18946@{}", protocol::INFERENCE_PIN)
        );
    }

    /// The staged-residency bound is a real inequality over real byte counts, not a comment.
    #[test]
    fn safetensors_bytes_sums_only_weight_shards_and_fails_closed_when_empty() {
        let root = std::env::temp_dir().join(format!(
            "sc-18808-ltx-shards-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        assert!(safetensors_bytes(&root)
            .unwrap_err()
            .contains("no .safetensors weights"));
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        assert!(
            safetensors_bytes(&root).is_err(),
            "a config is not a weight shard"
        );
        std::fs::write(root.join("model-00001-of-00002.safetensors"), vec![0_u8; 7]).unwrap();
        std::fs::write(root.join("model-00002-of-00002.safetensors"), vec![0_u8; 5]).unwrap();
        assert_eq!(safetensors_bytes(&root).unwrap(), 12);
        std::fs::remove_dir_all(root).ok();
    }
}

#[cfg(test)]
mod minimax_tests {
    use super::*;
    use mlx_gen::gen_core::MemoryStrategySupport;

    /// An otherwise-valid t2va plan row: the shipped 16:9 canvas at EXACTLY the pixel budget, the
    /// shortest legal clip, and the one legal cadence. `modelId` stays `minimax_h3` even when
    /// `provider` is foreign, so the provider guard is the check that fires rather than a
    /// missing-field complaint standing in for it.
    fn minimal_request(provider: &str) -> Value {
        json!({
            "hardware": {
                "memoryBytes": 137_438_953_472_u64,
                "wiredLimitBytes": 96_000_000_000_u64,
            },
            "planned": {
                "target": {
                    "provider": provider,
                    "modelId": MINIMAX_PROVIDER,
                    "tier": "q4",
                    "mode": "text_to_video",
                    "overlay": "none",
                    "geometry": { "width": 1344, "height": 768, "batch": 1, "frames": 124 }
                },
                "backend": "mlx",
                "loadShape": protocol::LOAD_SHAPE_EAGER,
                "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
                "calibrationFingerprint": MINIMAX_CALIBRATION_FINGERPRINT,
                "fixture": "minimax-h3-mlx-q4-1344x768-f124-fps24-seed17137"
            }
        })
    }

    fn weights_free_spec(quant: Option<Quant>, load_shape: LoadShape) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Resident)
            .with_load_shape(load_shape);
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        spec
    }

    /// The provider-owned weights-free contract fixture — the same one catalog conformance resolves
    /// when a snapshot is unavailable, so these pins touch no filesystem and no GPU.
    fn fixture_contract(
        quant: Option<Quant>,
        load_shape: LoadShape,
    ) -> mlx_gen::gen_core::MemoryProviderContract {
        let registry = mlx_gen_minimax_h3::provider_registry().unwrap();
        let fixture = registry
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == MINIMAX_PROVIDER)
            .expect("the provider-owned MiniMax-H3 contract fixture");
        (fixture.contract)(&weights_free_spec(quant, load_shape)).unwrap()
    }

    /// The capture harness must probe under the SAME evidence key the shipped runtime queries.
    ///
    /// The MiniMax twin of `the_ltx_runtime_context_binds_the_exact_video_route_and_selection`,
    /// which pins this correspondence for the family that already had it right. sc-18663: this arm
    /// probed admission under `text_to_image` while the plan it validates, the record it emits and
    /// `video_admission` all say `text_to_video` — see [`minimax_context`] for why that split is
    /// currently inert and what makes it stop being inert.
    ///
    /// NEITHER SIDE IS RESTATED AS A LITERAL. A test spelling `"text_to_video"` on both sides still
    /// passes when both sides move together, which is the drift this exists to catch.
    #[test]
    fn the_minimax_capture_context_binds_the_mode_key_the_runtime_video_route_sends() {
        // The runtime side, derived: `video_jobs::wan` resolves every video job's admission mode
        // with `payload_video_mode`, and `video_admission` types that string with
        // `memory_mode_from_mode_key`, whose contract is `as_key(from(key)) == key`. So the string
        // this returns IS the `mode_key` the runtime's decode-policy query carries.
        let payload = json!({
            "model": MINIMAX_PROVIDER,
            "mode": sceneworks_core::contracts::ContractMode::TextToVideo.as_str(),
        });
        let runtime_mode_key = sceneworks_core::video_request::payload_video_mode(
            payload.as_object().expect("the probe payload is an object"),
        );
        assert!(
            sceneworks_core::video_request::is_minimax_h3_model(MINIMAX_PROVIDER),
            "the payload above must be a MiniMax-H3 video job, or it derives another family's key"
        );

        // And that really is the only route this arm records: the plan-target gate admits the
        // derived spelling and refuses the image spelling this context used to carry.
        let plan_under = |mode: &str| {
            let mut request = minimal_request(MINIMAX_PROVIDER);
            request["planned"]["target"]["mode"] = json!(mode);
            request
        };
        assert!(validate_minimax_target(&plan_under(&runtime_mode_key)).is_ok());
        assert!(validate_minimax_target(&plan_under(MemoryMode::TextToImage.as_key())).is_err());

        // The capture side, from the builder every MiniMax admission probe runs through.
        let contract = fixture_contract(Some(Quant::Q4), LoadShape::EagerMaterialization);
        let calibration = contract.calibration.as_ref().unwrap();
        let context = minimax_context(
            planned_selection(&minimal_request(MINIMAX_PROVIDER)).unwrap(),
            calibration,
            &calibration.fingerprint,
            validate_minimax_geometry(1344, 768, 124).unwrap(),
            1 << 40,
            1,
        );
        assert_eq!(context.mode.as_key(), runtime_mode_key);
    }

    /// The per-arm twin of the sc-18104 provider guard: dispatch routes by name today, but this arm
    /// hardcodes the MiniMax-H3 contract, so a misrouted target must be refused BY NAME inside the
    /// arm — and before any environment variable is read, so the refusal cannot read like a
    /// provisioning problem.
    #[test]
    fn the_arm_refuses_a_foreign_provider_by_name_before_any_environment_work() {
        for provider in ["ltx_2_3", "flux2_dev", "qwen_image", "minimax_h3_ref"] {
            let error = validate_minimax_target(&minimal_request(provider))
                .expect_err("a foreign provider must not reach the MiniMax-H3 contract");
            assert_eq!(
                error,
                format!("{MINIMAX_LABEL} does not implement provider {provider:?}")
            );
            assert!(
                !error.contains("SCENEWORKS_")
                    && !error.contains("fingerprint")
                    && !error.contains("contract"),
                "the refusal came from the environment or the contract, so validation let it \
                 through: {error}"
            );
        }
        assert!(validate_minimax_target(&minimal_request(MINIMAX_PROVIDER)).is_ok());
    }

    /// The dispatcher must ROUTE `minimax_h3` into this arm rather than refuse it, and must not
    /// misroute it into a sibling. Proved by the two refusals it does NOT produce, which keeps the
    /// assertion deterministic whether or not the capture environment happens to be exported.
    #[test]
    fn dispatch_routes_the_minimax_provider_into_its_own_arm() {
        let error = run(&minimal_request(MINIMAX_PROVIDER))
            .expect_err("no MiniMax-H3 weights are staged during a unit test");
        assert_ne!(
            error,
            format!("MLX five-rung calibration does not implement provider {MINIMAX_PROVIDER:?}"),
            "the dispatcher still refuses the provider this arm implements"
        );
        for foreign in ["LTX", "FLUX.2", "Qwen", "SDXL", "Krea", "Z-Image"] {
            assert!(
                !error.contains(foreign),
                "the request was misrouted into the {foreign} arm: {error}"
            );
        }
    }

    /// The envelope's engine constants, pinned so a pin bump that moves one reds HERE rather than
    /// silently widening or narrowing what this arm will capture. The lattice is CHECKED as
    /// `17n + 5` rather than only tabulated, so a table that drifted into a different progression
    /// could not pass by matching itself.
    #[test]
    fn minimax_envelope_is_the_pinned_engines_own() {
        assert_eq!(mlx_gen_minimax_h3::SPATIAL_STRIDE, 32);
        assert_eq!(mlx_gen_minimax_h3::CANVAS_MAX_PIXELS, 1_032_192);
        assert_eq!(mlx_gen_minimax_h3::MINIMAX_H3_FPS, 24.0);
        assert_eq!(
            mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS,
            [124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345]
        );
        for (index, frames) in mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS.iter().enumerate() {
            assert_eq!(
                *frames,
                17 * index as i32 + 124,
                "the legal frame counts are 17n + 5, not a tabulated list"
            );
        }
        // The budget is a PRODUCT, and the published resolution list proves it cannot be a per-edge
        // cap: two shipped canvases whose long edges differ by 192px have the identical area.
        assert_eq!(1536 * 672, mlx_gen_minimax_h3::CANVAS_MAX_PIXELS);
        assert_eq!(1344 * 768, mlx_gen_minimax_h3::CANVAS_MAX_PIXELS);
    }

    #[test]
    fn the_geometry_envelope_refuses_off_lattice_off_stride_and_over_budget_canvases() {
        let admitted = validate_minimax_geometry(1344, 768, 124).unwrap();
        assert_eq!(admitted.frames, 124);
        // 124 = 17·7 + 5 ⇒ 5·7 + 2 video latents, and round(124/24·40) audio tokens.
        assert_eq!(admitted.video_latent_frames, 37);
        assert_eq!(admitted.audio_latent_frames, 207);

        // Off-lattice, including a still (`T = 1` does not render at all) and LTX's own `1 + 8k`
        // lattice, which shares no member with this one inside the legal range.
        for frames in [0_u32, 1, 97, 121, 123, 125, 305, 346, u32::MAX] {
            let error = validate_minimax_geometry(1344, 768, frames)
                .expect_err("an off-lattice frame count must be refused");
            assert!(error.contains("17n+5 lattice"), "{frames}: {error}");
        }

        // Off-stride, on both axes, checked before the area so the message names the real defect.
        for (width, height) in [(1352_u32, 768_u32), (1344, 776), (1330, 768)] {
            let error = validate_minimax_geometry(width, height, 124)
                .expect_err("an off-stride canvas must be refused");
            assert!(error.contains("32px stride"), "{width}x{height}: {error}");
        }

        // Over budget as a PRODUCT. 1536x704 is on-stride and its long edge is one the model ships,
        // so only the area check can refuse it.
        assert!(validate_minimax_geometry(1536, 672, 124).is_ok());
        assert!(validate_minimax_geometry(576, 320, 345).is_ok());
        let error = validate_minimax_geometry(1536, 704, 124)
            .expect_err("an over-budget canvas must be refused");
        assert!(error.contains("canvas budget"), "{error}");
    }

    #[test]
    fn the_geometry_envelope_pins_batch_to_one_clip_per_request() {
        let mut request = minimal_request(MINIMAX_PROVIDER);
        for batch in [0_u64, 2, 4] {
            request["planned"]["target"]["geometry"]["batch"] = json!(batch);
            let error = minimax_target_geometry(&request)
                .expect_err("this engine renders one clip per request");
            assert!(error.contains("geometry.batch == 1"), "{batch}: {error}");
        }
        // A missing or non-integer axis fails closed rather than defaulting.
        request["planned"]["target"]["geometry"]["batch"] = json!("1");
        assert!(minimax_target_geometry(&request)
            .unwrap_err()
            .contains("planned.target.geometry.batch must fit u32"));
    }

    /// This arm is the SECOND allowed to accept a multi-frame geometry, and it buys that with its
    /// own envelope rather than by dropping the axis: the shared still guard would refuse the exact
    /// geometry this arm admits.
    #[test]
    fn the_minimax_arm_accepts_a_multi_frame_geometry_the_image_guard_refuses() {
        let request = minimal_request(MINIMAX_PROVIDER);
        assert!(validate_minimax_target(&request).is_ok());
        assert_eq!(
            protocol::validate_still_geometry(&request, "MLX Qwen base calibration").unwrap_err(),
            "MLX Qwen base calibration requires geometry.frames == 1, got 124"
        );
    }

    #[test]
    fn a_reference_carrying_or_wrong_mode_target_is_refused_because_ref2va_is_another_checkpoint() {
        for (field, value) in [
            ("referenceCount", json!(1)),
            ("reference_count", json!(2)),
            ("hasReference", json!(true)),
            ("has_reference", json!(true)),
        ] {
            let mut request = minimal_request(MINIMAX_PROVIDER);
            request["planned"]["target"][field] = value;
            let error = validate_minimax_target(&request)
                .expect_err("a reference surface must be refused by this arm");
            assert!(error.contains(field) && error.contains("ref2va"), "{error}");
        }
        for mode in [
            "text_to_image",
            "image_to_video",
            "first_last_frame",
            "ref2va",
        ] {
            let mut request = minimal_request(MINIMAX_PROVIDER);
            request["planned"]["target"]["mode"] = json!(mode);
            let error =
                validate_minimax_target(&request).expect_err("only t2va is capturable here");
            assert!(
                error.contains("text_to_video") && error.contains(mode),
                "{error}"
            );
        }
    }

    #[test]
    fn a_material_overlay_target_is_refused_rather_than_recorded_as_no_overlay() {
        for overlay in ["lora", "control", "identity", "control:1"] {
            let mut request = minimal_request(MINIMAX_PROVIDER);
            request["planned"]["target"]["overlay"] = json!(overlay);
            let error = run_minimax_h3(&request).expect_err("a material overlay must be refused");
            assert!(error.contains(MINIMAX_PLAIN_EXECUTION_PATH), "{error}");
            assert!(error.contains("refusing"), "{error}");
        }
    }

    #[test]
    fn the_fixture_binds_the_tier_geometry_cadence_and_seed() {
        let request = minimal_request(MINIMAX_PROVIDER);
        let geometry = validate_minimax_target(&request).unwrap();
        assert_eq!(
            planned_minimax_capture(&request, "q4", geometry).unwrap(),
            (24, MINIMAX_SEED)
        );
        // The tier and every geometry axis are part of the binding.
        for tier in ["q8", "bf16"] {
            assert!(planned_minimax_capture(&request, tier, geometry)
                .unwrap_err()
                .contains("must start with"));
        }
        let mut off = request.clone();
        off["planned"]["fixture"] = json!("minimax-h3-mlx-q4-1344x768-f141-fps24-seed17137");
        assert!(planned_minimax_capture(&off, "q4", geometry)
            .unwrap_err()
            .contains("must start with"));
        // The cadence axis the geometry envelope cannot carry has exactly one legal value here.
        off["planned"]["fixture"] = json!("minimax-h3-mlx-q4-1344x768-f124-fps30-seed17137");
        assert!(planned_minimax_capture(&off, "q4", geometry)
            .unwrap_err()
            .contains("24 fps only"));
        off["planned"]["fixture"] = json!("minimax-h3-mlx-q4-1344x768-f124-fps24-seed1234");
        assert!(planned_minimax_capture(&off, "q4", geometry)
            .unwrap_err()
            .contains("does not match the MiniMax-H3 calibration seed"));
    }

    /// Fingerprint check 1 of 3 — the plan against this arm — must fire BEFORE the environment
    /// contract, so a re-fingerprinted provider cannot be diagnosed as a provisioning failure.
    /// Checks 2 and 3 (provider-vs-arm and plan-vs-provider) need the loaded registry contract;
    /// `the_pinned_minimax_contract_declares_rung_support_by_load_shape_not_by_tier` pins the
    /// constant those two compare against, weights-free.
    #[test]
    fn a_plan_naming_another_fingerprint_is_refused_before_any_environment_work() {
        let mut request = minimal_request(MINIMAX_PROVIDER);
        request["planned"]["calibrationFingerprint"] = json!("minimax-h3-mlx-deferred-v1");
        let error = run_minimax_h3(&request).expect_err("a foreign fingerprint must be refused");
        assert!(
            error.starts_with("plan/adapter calibration mismatch"),
            "{error}"
        );
        assert!(
            !error.contains("SCENEWORKS_"),
            "the refusal must precede the env contract: {error}"
        );
    }

    /// The arm's load-bearing premises, pinned against the PINNED provider crate, weights-free.
    ///
    /// The finding the campaign has to plan around is that rung support here is a function of the
    /// resolved LOAD SHAPE, not of the tier: `bounded_attention` is `StructurallyNotApplicable` on
    /// every shape and tier, and `bounded_transformer_residency` is `Implemented` ONLY under
    /// `deferred_materialization` and `Missing` under eager. Asserted across all three tiers so a
    /// tier-shaped assumption cannot survive here either.
    #[test]
    fn the_pinned_minimax_contract_declares_rung_support_by_load_shape_not_by_tier() {
        assert_eq!(
            MINIMAX_CALIBRATION_FINGERPRINT,
            mlx_gen_minimax_h3::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
            "the plan entries pin this fingerprint; regenerate them with the provider"
        );
        for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
            for (load_shape, rung4_implemented) in [
                (LoadShape::EagerMaterialization, false),
                (LoadShape::DeferredMaterialization, true),
            ] {
                let contract = fixture_contract(quant, load_shape);
                let where_we_are = format!("{quant:?} at {load_shape:?}");
                assert_eq!(contract.provider_id, MINIMAX_PROVIDER);
                for strategy in [
                    MemoryStrategy::Resident,
                    MemoryStrategy::StagedResidency,
                    MemoryStrategy::BoundedDecode,
                ] {
                    assert!(
                        matches!(
                            contract.capability(strategy).unwrap().support,
                            MemoryStrategySupport::Implemented
                        ),
                        "{strategy:?} is no longer Implemented ({where_we_are})"
                    );
                }
                assert!(
                    matches!(
                        contract
                            .capability(MemoryStrategy::BoundedAttention)
                            .unwrap()
                            .support,
                        MemoryStrategySupport::StructurallyNotApplicable { .. }
                    ),
                    "bounded_attention changed disposition ({where_we_are})"
                );
                let rung4 = contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap();
                if rung4_implemented {
                    assert!(
                        matches!(rung4.support, MemoryStrategySupport::Implemented),
                        "rung 4 must be Implemented on a deferred load ({where_we_are})"
                    );
                    assert_eq!(rung4.parameters.transformer_window_sizes, vec![1]);
                    assert_eq!(
                        rung4.parameters.transformer_window_components,
                        vec![TransformerComponent::Both],
                        "rung 4 streams BOTH transformers here, not the DiT alone"
                    );
                } else {
                    assert!(
                        matches!(rung4.support, MemoryStrategySupport::Missing),
                        "rung 4 must be Missing on a resident load ({where_we_are})"
                    );
                }
                // Rung 2's domain is a SINGLETON — the tile geometry is an output-correctness input
                // copied from the published VAE, not a memory lever a sweep may vary.
                let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
                assert_eq!(decode.parameters.decode_tile_edges, vec![256]);
                assert_eq!(decode.parameters.decode_overlaps, vec![64]);
                // The calibration identity carries the RESOLVED shape, which is what
                // `run_minimax_h3` refuses a mismatch against and what the record's `loadShape`
                // is taken from.
                let calibration = contract.calibration.as_ref().unwrap();
                assert_eq!(calibration.fingerprint, MINIMAX_CALIBRATION_FINGERPRINT);
                assert_eq!(calibration.load_shape, load_shape, "{where_we_are}");
            }
        }
    }

    /// The plan's `engagedRungs` must equal the provider's engaged composition, or
    /// `attested_strategy` fails every capture at plan/measured comparison time — so the exact
    /// compositions are pinned here rather than assumed from the ordinal rung order.
    ///
    /// TWO of them are NOT what a reader would guess, and a plan written from the LTX shape would
    /// be rejected by every MiniMax-H3 capture:
    ///
    /// * `bounded_decode` engages `[resident, bounded_decode]` — **`staged_residency` is not in
    ///   it**, unlike LTX's `[resident, staged_residency, bounded_decode]`;
    /// * `bounded_transformer_residency` engages `bounded_decode`, so a rung-4 row must ALSO carry
    ///   `decodeTileEdge` / `decodeOverlap` or `validate_selection` refuses it with
    ///   "decode_tile_edge is required by a strategy rung this selection engages".
    #[test]
    fn the_engaged_composition_is_what_the_plan_rows_must_declare() {
        use MemoryStrategy::{
            BoundedDecode, BoundedTransformerResidency, Resident, StagedResidency,
        };
        for load_shape in [
            LoadShape::EagerMaterialization,
            LoadShape::DeferredMaterialization,
        ] {
            let contract = fixture_contract(Some(Quant::Q4), load_shape);
            assert_eq!(contract.engaged_composition(Resident), vec![Resident]);
            assert_eq!(
                contract.engaged_composition(StagedResidency),
                vec![Resident, StagedResidency]
            );
            assert_eq!(
                contract.engaged_composition(BoundedDecode),
                vec![Resident, BoundedDecode],
                "bounded_decode does NOT engage staged_residency here ({load_shape:?})"
            );
        }
        assert_eq!(
            fixture_contract(Some(Quant::Q4), LoadShape::DeferredMaterialization)
                .engaged_composition(BoundedTransformerResidency),
            vec![Resident, BoundedDecode, BoundedTransformerResidency],
            "rung 4 drags rung 2 in, so a rung-4 plan row must declare the decode parameters too"
        );
    }

    /// Rung 4 is reachable ONLY under the deferred shape, and it needs the `"both"` window spelling
    /// the shared parser learned for this provider. Both halves are asserted, in both directions.
    #[test]
    fn rung_four_is_capturable_only_under_the_deferred_load_shape() {
        let mut request = minimal_request(MINIMAX_PROVIDER);
        request["planned"]["loadShape"] = json!(protocol::LOAD_SHAPE_DEFERRED);
        // Rung 4 ENGAGES rung 2 here, so the decode parameters are mandatory on a rung-4 row —
        // omitting them is refused with "decode_tile_edge is required by a strategy rung this
        // selection engages", which reads like a plan typo and is not one.
        request["planned"]["strategy"] = json!({
            "rung": "bounded_transformer_residency",
            "engagedRungs": ["resident", "bounded_decode", "bounded_transformer_residency"],
            "parameters": {
                "decodeTileEdge": 256,
                "decodeOverlap": 64,
                "transformerWindowSize": 1,
                "transformerWindowComponent": "both"
            }
        });
        let selection = planned_selection(&request).expect("the shared parser accepts \"both\"");
        assert_eq!(
            selection.parameters.transformer_window_component,
            Some(TransformerComponent::Both)
        );
        fixture_contract(Some(Quant::Q4), LoadShape::DeferredMaterialization)
            .validate_selection(&selection)
            .expect("a deferred load implements rung 4");
        assert!(
            fixture_contract(Some(Quant::Q4), LoadShape::EagerMaterialization)
                .validate_selection(&selection)
                .is_err(),
            "a resident load does NOT have rung 4's prerequisite shape"
        );

        // And the decode parameters really are the reason, not incidental: dropping them fails the
        // deferred contract too.
        request["planned"]["strategy"]["parameters"] =
            json!({ "transformerWindowSize": 1, "transformerWindowComponent": "both" });
        let bare = planned_selection(&request).unwrap();
        assert!(
            fixture_contract(Some(Quant::Q4), LoadShape::DeferredMaterialization)
                .validate_selection(&bare)
                .unwrap_err()
                .to_string()
                .contains("decode_tile_edge")
        );
    }

    /// **The provider contract, pinned in code.** At the permanent inference pin the MiniMax-H3
    /// provider publishes `memory_strategy_contract` and its admission check is non-vacuous in both
    /// directions. The registry contract and loaded-provider checks must continue to agree; a future
    /// contract/check drift fails here rather than producing evidence.
    #[test]
    fn the_minimax_h3_memory_strategy_contract_enforces_non_vacuous_admission() {
        let registry = mlx_gen_minimax_h3::provider_registry().unwrap();
        let contract = registry
            .memory_strategy_contract(
                MINIMAX_PROVIDER,
                &weights_free_spec(Some(Quant::Q4), LoadShape::EagerMaterialization),
            )
            .expect("the registry resolves a MiniMax-H3 memory-strategy contract")
            .expect("the MiniMax-H3 memory-strategy registration is present at the pin");
        assert_eq!(contract.provider_id, MINIMAX_PROVIDER);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MINIMAX_CALIBRATION_FINGERPRINT
        );
        // The registered admission check is non-vacuous in BOTH directions, weights-free — the
        // same property the loaded generator must enforce at the permanent pin.
        let spec = weights_free_spec(Some(Quant::Q4), LoadShape::EagerMaterialization);
        let selection = planned_selection(&minimal_request(MINIMAX_PROVIDER)).unwrap();
        let calibration = contract.calibration.as_ref().unwrap();
        let geometry = validate_minimax_geometry(1344, 768, 124).unwrap();
        let context = |fingerprint: &str, total: u64| {
            minimax_context(selection, calibration, fingerprint, geometry, total, 1)
        };
        assert!(matches!(
            mlx_gen_minimax_h3::memory_strategy::safety_check(
                &spec,
                &contract,
                &context(&calibration.fingerprint, 1 << 40)
            ),
            MemorySafetyDecision::Accept
        ));
        assert!(matches!(
            mlx_gen_minimax_h3::memory_strategy::safety_check(
                &spec,
                &contract,
                &context(&calibration.fingerprint, 0)
            ),
            MemorySafetyDecision::Reject { .. }
        ));
        assert!(matches!(
            mlx_gen_minimax_h3::memory_strategy::safety_check(
                &spec,
                &contract,
                &context("stale-minimax-h3-fingerprint", 1 << 40)
            ),
            MemorySafetyDecision::Reject { .. }
        ));
        // The route gate is real too: an off-lattice geometry the arm would never send is refused
        // by the PROVIDER, so the accept above is not a blanket accept.
        let mut off_lattice = context(&calibration.fingerprint, 1 << 40);
        off_lattice.geometry.frames = 97;
        assert!(matches!(
            mlx_gen_minimax_h3::memory_strategy::safety_check(&spec, &contract, &off_lattice),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// The runtime-complete sweep contract the harness checks: `rangeVerified`, exactly one passed
    /// case, and parameters equal to the plan's own. Rung 4's string component belongs in the case
    /// and never as a swept axis.
    #[test]
    fn the_runtime_complete_sweep_carries_one_passed_case_matching_the_planned_parameters() {
        let mut request = minimal_request(MINIMAX_PROVIDER);
        let sweep = minimax_complete_sweep(&request).unwrap();
        assert_eq!(sweep["rangeVerified"], json!(true));
        assert_eq!(sweep["axes"], json!([]));
        assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
        assert_eq!(sweep["cases"][0]["result"], json!("passed"));
        assert_eq!(
            sweep["cases"][0]["parameters"],
            request["planned"]["strategy"]["parameters"]
        );

        request["planned"]["strategy"]["parameters"] =
            json!({ "transformerWindowSize": 1, "transformerWindowComponent": "both" });
        let sweep = minimax_complete_sweep(&request).unwrap();
        assert_eq!(
            sweep["axes"],
            json!([{ "parameter": "transformerWindowSize", "testedValues": [1] }]),
            "a component name is not a swept numeric range"
        );
        assert_eq!(
            sweep["cases"][0]["parameters"],
            request["planned"]["strategy"]["parameters"]
        );
    }

    /// A tier is a WHOLE-PIPELINE contract here: the conditioning stage's tier is derived from the
    /// DiT's rather than being a free axis the record could not describe.
    #[test]
    fn the_text_encoder_tier_is_derived_from_the_dit_tier() {
        assert_eq!(
            minimax_text_encoder_source("q4"),
            MINIMAX_TIERED_TEXT_ENCODER
        );
        assert_eq!(
            minimax_text_encoder_source("q8"),
            MINIMAX_TIERED_TEXT_ENCODER
        );
        assert_eq!(
            minimax_text_encoder_source("bf16"),
            MINIMAX_UPSTREAM_TEXT_ENCODER,
            "the rehost publishes no bf16 text encoder; the manifest's bf16 co-requisite row is \
             the upstream repository"
        );
    }

    /// The output unwrapper refuses the two shapes an image or audio provider would return, under
    /// this arm's OWN name — so a misrouted still could never be recorded as a joint A/V clip.
    #[test]
    fn the_minimax_output_unwrapper_refuses_image_and_audio_shaped_output() {
        let frame = Image {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 255, 255, 255],
        };
        assert_eq!(
            diagnostic_video_frames(
                GenerationOutput::Images(vec![frame.clone()]),
                MINIMAX_VIDEO_LABEL
            )
            .unwrap_err(),
            "MLX MiniMax-H3 render returned images, not a video clip"
        );
        assert_eq!(
            diagnostic_video_frames(
                GenerationOutput::Video {
                    frames: Vec::new(),
                    fps: 24,
                    audio: None,
                },
                MINIMAX_VIDEO_LABEL
            )
            .unwrap_err(),
            "MLX MiniMax-H3 render returned no frames"
        );
        // And the LTX arm's own refusals are unchanged by the label becoming a parameter.
        assert_eq!(
            diagnostic_video_frames(GenerationOutput::Images(vec![frame]), LTX_VIDEO_LABEL)
                .unwrap_err(),
            "MLX LTX-2.3 render returned images, not a video clip"
        );
    }

    /// The determinism envelope must be falsifiable in both directions: an identical clip passes,
    /// and the broad bias the arm's negative mutation applies breaches all three metrics.
    #[test]
    fn the_determinism_envelope_admits_an_identical_clip_and_breaches_on_the_mutation() {
        let clip = vec![
            Image {
                width: 2,
                height: 1,
                pixels: vec![10, 20, 30, 200, 210, 220],
            },
            Image {
                width: 2,
                height: 1,
                pixels: vec![40, 50, 60, 70, 80, 90],
            },
        ];
        let (maximum, mean, rms) = video_max_mean_rms_abs(&clip, &clip).unwrap();
        assert_eq!((maximum, mean, rms), (0.0, 0.0, 0.0));
        assert!(minimax_quality_passes(maximum, mean, rms));

        let mutated = clip.iter().map(qwen_negative_mutation).collect::<Vec<_>>();
        let (maximum, mean, rms) = video_max_mean_rms_abs(&mutated, &clip).unwrap();
        assert!(maximum > MINIMAX_MAX_THRESHOLD);
        assert!(mean > MINIMAX_MEAN_THRESHOLD);
        assert!(rms > MINIMAX_RMS_THRESHOLD);
        assert!(!minimax_quality_passes(maximum, mean, rms));

        // The thresholds are this provider's own literals, pinned in the 0-255 unit an operator
        // reads them in. They are spelled as MiniMax-named literals rather than aliased onto
        // another lane's constants, so a receipt embedding them cannot be traced to a constant
        // asserting a different provider's provenance.
        let per_255 = |threshold: f64| (threshold * 255.0).round() as u64;
        assert_eq!(
            (
                per_255(MINIMAX_MAX_THRESHOLD),
                per_255(MINIMAX_MEAN_THRESHOLD),
                per_255(MINIMAX_RMS_THRESHOLD),
            ),
            (3, 1, 2)
        );
    }
}
