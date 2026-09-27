//! YuE2 memory admission, stage residency and residency leases (sc-23001, epic 22988).
//!
//! YuE2 (`candle-audio-yue2`, the noncommercial experimental song model) is priced here before
//! anything loads: the job hands over the exact [`GenerationRequest`] it will send, this module
//! prices every stage of that render against the live memory budget, CHOOSES the per-request memory
//! controls the engine honours (inference `accd49ea`, `provider::memory_options`) so the render
//! fits, and refuses a render that cannot fit — before the loader is called — with the binding
//! stage, the shortfall and what would fit instead. Like YuE1's gate it is deliberately NOT routed
//! through the shared image memory ladder (`memory_strategy` / `candle_memory_strategy`), which is
//! image-lane only by construction.
//!
//! # One MoT, not independent stages
//!
//! YuE1 loaded one LM per stage and released it before the next, so its floor was the largest single
//! stage. YuE2 is ONE Mixture-of-Transformers that serves every stage — the ABC plan and the
//! semantic tokens run its AR path, the acoustic flow matching runs the AR path over the song
//! prefix **and** the NAR twins, and the VAE decodes while the MoT is still loaded. So the model's
//! weights are resident in EVERY stage, and each stage adds only its own transient working set,
//! which the engine releases before the next stage starts. The floor is therefore
//! `max over stages (resident weights + that stage's working set)` — never a sum over stages and
//! never a V1-style max over independently loaded models:
//!
//! | stage | working set on top of the resident weights |
//! |---|---|
//! | load | the tier's weights file, mapped while it is copied into the resident layout |
//! | transcription (cover from a recording) | SheetSage2 + MERT-v2 (F32) and one 300 s window; the transcriber unloads before the MoT loads, so it is its own stage and adds no MoT weights |
//! | plan (ABC) | one KV cache of `prefix + abc max_tokens` positions + the AR prefill workspace |
//! | semantic | one KV cache per CFG branch (two under guidance) of `prefix + semantic max_tokens` positions + the prefill workspace |
//! | acoustic prefill | per chunk: a KV cache of `ar + nar` positions + the AR prefill workspace over the chunk's AR sequence (AR path resident) |
//! | acoustic solve | per chunk: the same cache + the NAR score tiles ([`Yue2Controls::attention_elements`]) + the NAR activations + the latents; with AR offload the AR-only weights sit in host memory for the solve |
//! | decode | the FP32 VAE decoder + one halo/crop tile ([`Yue2Controls::decode_core_frames`]) + the song's waveform |
//!
//! Every KV cache is `layers × 2 (K, V) × kv_heads × head_dim × positions` in the compute dtype
//! (`StaticKvCache`, batch 1 per branch), capped at the released 24 576-position context the engine
//! enforces (`protocol::check_generation_budget`).
//!
//! # Weights
//!
//! [`weight_residency`] mirrors `candle_audio_yue2::precision::weight_residency` over the pinned
//! YuE2-3B tensor table (`manifests/yue2_3b.json` at the pinned revision), enumerated here from the
//! architecture rather than linked, because the worker reaches the engine only through the runtime
//! bundle: the V2 precision map (every matmul weight follows the tier as GGML Q8_0 / Q4_K / Q4_0;
//! embeddings, the latent position table, norms and biases stay BF16), the backend's dense dtype
//! (F32 on the CPU, BF16 on an accelerator unless the load asks for F32), Candle's CUDA GGML row
//! padding, and the experimental FP8 AR mode's layout (FP8 AR projections on the device, their BF16
//! originals retained in host memory until they are restored before the acoustic stage). The totals
//! reproduce sc-22995's recorded residencies exactly (CPU q8 5.12 GB / q4 3.52 GB; CUDA bf16 7.26 GB,
//! FP8 5.85 GB + 2.82 GB host, q8 4.26 GB, q4 2.66 GB) — content-derived numbers, not measurements of
//! a machine. The VAE is FP32 at every tier and is priced from the decoder component's own file.
//!
//! # Budgets
//!
//! * **Apple silicon (candle-Metal)** — one unified pool: every byte (device and "host") counts
//!   against the GPU's recommended working set, less the residency other live YuE2 leases still
//!   hold ([`Yue2Lease`]). AR offload buys nothing there (it copies into the same memory), so it is
//!   never chosen.
//! * **CUDA** — the device pool (live free VRAM plus the dedicated-VRAM reserve, with the same
//!   evict-then-reclaim every candle lane runs) and the host pool (retained FP8 originals, offloaded
//!   AR weights, the mapped weights file), each checked separately.
//! * **No reading** — admitted with the engine's production controls: the gate never blocks without
//!   evidence ([`crate::fit_gate::FitDecision::Unknown`]).
//!
//! # Evidence
//!
//! No YuE2 render has been measured on Metal or CUDA yet; the one final calibration campaign is the
//! terminal story's (sc-23002, `config/yue2-memory-profile-plan.json`). Until then every Metal figure
//! is an estimate pending that campaign, CUDA carries measured weights under shape-derived
//! activations, and the decode tile constant is the engine's own CPU measurement (sc-22993). Every
//! refusal says which ([`Yue2Evidence`]).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use gen_core::{GenerationMemory, GenerationRequest, Progress, SongDecoder, SongPlanning};
use serde_json::Value;

use crate::fit_gate::BYTES_PER_GIB;
use crate::WorkerError;

// ---- Architecture: the pinned `m-a-p/YuE2-3B@1a96eca` config / tensor table ----------------------

/// `num_hidden_layers`.
const LAYERS: u64 = 28;
/// `hidden_size`.
const HIDDEN: u64 = 2048;
/// `num_attention_heads` (query heads).
const HEADS: u64 = 16;
/// `num_key_value_heads` (grouped-query).
const KV_HEADS: u64 = 8;
/// `head_dim`.
const HEAD_DIM: u64 = 128;
/// `intermediate_size` (SwiGLU inner width).
const INTERMEDIATE: u64 = 6144;
/// `vocab_size` (text, ABC, protocol specials, 32 768 codec ids, latent markers).
const VOCAB: u64 = 184_704;
/// The released context (`protocol::CONTEXT` = `max_position_embeddings` = the latent position
/// table's rows) — the job contract's own `CONTEXT_TOKENS`, not a copy.
const CONTEXT: u64 = sceneworks_core::yue2_score::jobs::CONTEXT_TOKENS as u64;
/// `latent_dim`: acoustic latent channels.
const LATENT_DIM: u64 = 64;
/// `TimestepEmbedder.frequency_embedding_size` (`nar::TIME_FREQUENCIES`).
const TIME_FREQUENCIES: u64 = 256;

// ---- Engine schedule constants (`candle-audio-yue2` / `candle-llm` at the pin) ---------------------

/// `model::PREFILL_CHUNK`: positions per AR prefill forward.
const PREFILL_CHUNK: u64 = 512;
/// `candle-llm` `EAGER_ATTN_QUERY_CHUNK_SIZE` = `nar::UPSTREAM_QUERY_TILE`: query rows per score
/// tile of an AR prefill forward, and the historical NAR tile.
const EAGER_QUERY_TILE: u64 = 256;
/// `nar::SCORE_TILES_LIVE`: score-sized temporaries one attention call holds at once (the raw `QKᵀ`
/// product, the masked/scaled copy and the softmax).
const SCORE_TILES_LIVE: u64 = 3;
/// `Sampling::abc_default().max_tokens` — the job contract's `ABC_TOKEN_DEFAULTS` maximum.
const ABC_MAX_TOKENS_DEFAULT: u64 = sceneworks_core::yue2_score::jobs::ABC_TOKEN_DEFAULTS.1 as u64;
/// `Sampling::semantic_default().max_tokens` — the job contract's `SEMANTIC_TOKEN_DEFAULTS` maximum.
const SEMANTIC_MAX_TOKENS_DEFAULT: u64 =
    sceneworks_core::yue2_score::jobs::SEMANTIC_TOKEN_DEFAULTS.1 as u64;
/// A lower bound on the semantic prefix, priced as zero tokens (the real one always carries at
/// least `EOD`, an instruction line and the score framing, so zero is conservative). It bounds the
/// longest song and the longest acoustic chunk — a shorter prefix leaves more of the context to the
/// song — so it, never the upper-bound prefix, sizes them.
const MIN_SEMANTIC_PREFIX: u64 = 0;
/// Allowance for the fixed prompt framing around the style and lyrics text: `EOD`, the mode's
/// instruction line (≤ 160 bytes), the field labels and the ABC / music markers. Qwen BPE never emits
/// more tokens than a text has bytes, so text bytes plus this bound every prefix.
const PROMPT_FRAMING_TOKENS: u64 = 256;

/// `decode::TILE_BYTES_PER_FRAME`: decode-tile peak growth per latent frame (28 MiB, measured on the
/// Candle CPU backend by sc-22993; the engine treats it as a conservative bound elsewhere).
const DECODE_TILE_BYTES_PER_FRAME: u64 = 28 << 20;
/// `decode::TILE_RESERVE_BYTES`: the fixed part of a tile decode's peak (1 GiB).
const DECODE_TILE_RESERVE_BYTES: u64 = 1 << 30;
/// `vae::DEFAULT_HALO_FRAMES`.
const DECODE_HALO_FRAMES: u64 = 16;
/// `decode::DEFAULT_DECODE_BUDGET_GIB`: the production tiling's budget. Its core
/// (`DecodeOptions::production()`, 224 frames) is the largest this gate ever selects.
const DECODE_DEFAULT_BUDGET_BYTES: u64 = 8 << 30;
/// `vae::DEFAULT_CORE_FRAMES`: the largest core the engine accepts.
const DECODE_MAX_CORE_FRAMES: u64 = 1024;
/// Output samples per latent frame (48 kHz, 40 ms).
const SAMPLES_PER_LATENT_FRAME: u64 = 1920;
/// Output channels (stereo).
const AUDIO_CHANNELS: u64 = 2;
/// The decoded waveform is held about three times over while tiles are concatenated, clamped and
/// interleaved (`decode::estimated_tile_bytes` docs).
const WAVEFORM_COPIES: u64 = 3;

/// Candle's CUDA `MATRIX_ROW_PADDING`: extra elements every GGML tensor carries on a CUDA device.
const CUDA_GGML_ROW_PADDING: u64 = 512;
/// `candle_quant_kernels::FP8_COMPUTE_CAP_FLOOR`: the FP8 AR mode needs CUDA sm_89+.
const FP8_COMPUTE_CAP_FLOOR: f32 = 8.9;

// ---- Transcription (SheetSage2 + MERT-v2-FullSong, `candle-audio-sheetsage2`) -----------------------

/// MERT-v2's encoder rate: frames per second of source audio (ConvNeXt subsampler to 25 Hz).
const MERT_FRAMES_PER_SEC: u64 = 25;
/// MERT-v2 hidden width.
const MERT_HIDDEN: u64 = 1024;
/// MERT-v2 attention heads.
const MERT_HEADS: u64 = 16;
/// `mert::ATTENTION_QUERY_CHUNK`.
const MERT_QUERY_CHUNK: u64 = 512;
/// MERT-v2's feed-forward inner width (`ffn{1,2}.w_1`: [4096, 1024]).
const MERT_FFN: u64 = 4096;
/// SheetSage2 transcribes one fixed 300 s context window at a time.
const TRANSCRIPTION_WINDOW_SECS: u64 = 300;

// ---- Tiers, backends, precision ---------------------------------------------------------------------

/// A YuE2 weight tier (`candle_audio_yue2::precision::Tier`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Yue2Tier {
    Bf16,
    Q8,
    Q4,
}

impl Yue2Tier {
    /// Every tier, densest first.
    pub(crate) const ALL: [Yue2Tier; 3] = [Yue2Tier::Bf16, Yue2Tier::Q8, Yue2Tier::Q4];

    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    /// The tier a catalog variant names. Anything else is `None`, which every caller refuses — an
    /// unknown tier is never priced as zero.
    pub(crate) fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tier| tier.key() == key)
    }
}

/// The Candle backend a render runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Yue2Backend {
    /// Candle CPU: dense weights in F32 (no BF16 matmul), one host pool.
    Cpu,
    /// Candle CUDA: a dedicated device pool plus host RAM.
    Cuda,
    /// Candle Metal: Apple unified memory, one pool.
    Metal,
}

impl Yue2Backend {
    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }

    /// The backend this build's audio lane runs YuE2 on (`candle_audio::default_device`: the
    /// `audio-metal` bundle on macOS, the CUDA bundle under `backend-candle`).
    pub(crate) fn of_this_build() -> Self {
        if cfg!(target_os = "macos") {
            Self::Metal
        } else if cfg!(feature = "backend-candle") {
            Self::Cuda
        } else {
            Self::Cpu
        }
    }

    /// Whether device and host share one pool (so moving bytes between them frees nothing).
    fn unified(self) -> bool {
        !matches!(self, Self::Cuda)
    }
}

/// The dense compute precision (`LoadSpec::precision`): the default is BF16 on an accelerator and
/// F32 on the CPU; `Fp32` forces F32 everywhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Yue2Precision {
    #[default]
    Default,
    /// An explicit F32 load. The audio job loads with the default precision; priced so the
    /// accelerator F32 residency (14.52 GB) is never read as the BF16 one.
    #[cfg_attr(not(test), allow(dead_code))]
    Fp32,
}

/// The AR stages' mode (`fp8::ArPrecision`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Yue2ArMode {
    #[default]
    Native,
    /// The experimental FP8 E4M3 AR mode: CUDA sm_89+, the `bf16` tier, BF16 compute only. Not
    /// reachable through the registered provider at the pin (owner decision
    /// `fp8_not_on_the_load_spec`: `LoadSpec` has no FP8 value); priced so the retained BF16
    /// originals are never silently omitted if it becomes reachable.
    #[cfg_attr(not(test), allow(dead_code))]
    Fp8,
}

/// Bytes per dense element (weights stored BF16 and every activation / KV element).
fn dense_bytes(backend: Yue2Backend, precision: Yue2Precision) -> u64 {
    match (backend, precision) {
        (Yue2Backend::Cpu, _) | (_, Yue2Precision::Fp32) => 4,
        (Yue2Backend::Cuda | Yue2Backend::Metal, Yue2Precision::Default) => 2,
    }
}

// ---- Weight residency -------------------------------------------------------------------------------

/// A tensor's class under the V2 precision map (`precision::TensorClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorClass {
    TokenEmbedding,
    LatentPositions,
    /// An RMSNorm weight; `ar_only` for the AR path's norms (moved by AR offload).
    Norm {
        ar_only: bool,
    },
    Bias,
    ArProjection,
    NarProjection,
    LmHead,
    NarHead,
}

impl TensorClass {
    /// A matmul weight: the classes a quantized tier stores quantized.
    fn follows_tier(self) -> bool {
        matches!(
            self,
            Self::ArProjection | Self::NarProjection | Self::LmHead | Self::NarHead
        )
    }

    /// Moved to host memory by AR offload (`Yue2Lm::offload_ar`: `embed_tokens`, `lm_head` and every
    /// layer's AR path).
    fn ar_only(self) -> bool {
        matches!(
            self,
            Self::TokenEmbedding | Self::LmHead | Self::ArProjection | Self::Norm { ar_only: true }
        )
    }
}

/// `count` tensors of logical shape `[rows, cols]` (a vector is `[1, len]`).
#[derive(Clone, Copy, Debug)]
struct TensorGroup {
    class: TensorClass,
    rows: u64,
    cols: u64,
    count: u64,
}

const fn group(class: TensorClass, rows: u64, cols: u64, count: u64) -> TensorGroup {
    TensorGroup {
        class,
        rows,
        cols,
        count,
    }
}

/// The released checkpoint's 628 tensors, grouped by class and shape.
fn tensor_table() -> [TensorGroup; 22] {
    use TensorClass::*;
    const Q: u64 = HEADS * HEAD_DIM;
    const KV: u64 = KV_HEADS * HEAD_DIM;
    [
        group(TokenEmbedding, VOCAB, HIDDEN, 1),
        group(LatentPositions, CONTEXT, HIDDEN, 1),
        group(LmHead, VOCAB, HIDDEN, 1),
        group(Norm { ar_only: false }, 1, HIDDEN, 1), // model.norm
        // NAR heads: vae2llm [2048, 64], llm2vae [64, 2048], time_embedder.mlp.{0,2}.
        group(NarHead, HIDDEN, LATENT_DIM, 1),
        group(NarHead, LATENT_DIM, HIDDEN, 1),
        group(NarHead, HIDDEN, TIME_FREQUENCIES, 1),
        group(NarHead, HIDDEN, HIDDEN, 1),
        // Their biases: vae2llm, time_embedder.mlp.{0,2} [2048]; llm2vae [64].
        group(Bias, 1, HIDDEN, 3),
        group(Bias, 1, LATENT_DIM, 1),
        // Per layer, AR path: q, o [2048, 2048]; k, v [1024, 2048]; gate, up [6144, 2048]; down
        // [2048, 6144]; input / post-attention norms [2048]; q / k norms [128].
        group(ArProjection, Q, HIDDEN, 2 * LAYERS),
        group(ArProjection, KV, HIDDEN, 2 * LAYERS),
        group(ArProjection, INTERMEDIATE, HIDDEN, 2 * LAYERS),
        group(ArProjection, HIDDEN, INTERMEDIATE, LAYERS),
        group(Norm { ar_only: true }, 1, HIDDEN, 2 * LAYERS),
        group(Norm { ar_only: true }, 1, HEAD_DIM, 2 * LAYERS),
        // The NAR twins of the same (`nar_self_attn`, `nar_mlp`, `nar_input_layernorm`,
        // `nar_pre_mlp_layernorm`, `nar_self_attn.{q,k}_norm`).
        group(NarProjection, Q, HIDDEN, 2 * LAYERS),
        group(NarProjection, KV, HIDDEN, 2 * LAYERS),
        group(NarProjection, INTERMEDIATE, HIDDEN, 2 * LAYERS),
        group(NarProjection, HIDDEN, INTERMEDIATE, LAYERS),
        group(Norm { ar_only: false }, 1, HIDDEN, 2 * LAYERS),
        group(Norm { ar_only: false }, 1, HEAD_DIM, 2 * LAYERS),
    ]
}

/// How a tier stores one tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Storage {
    /// The released BF16 values.
    Dense,
    /// A GGML block tensor: `block` elements per `type_size` bytes.
    Ggml { block: u64, type_size: u64 },
}

/// GGML Q8_0 (32 elements → 34 bytes).
const Q8_0: Storage = Storage::Ggml {
    block: 32,
    type_size: 34,
};
/// GGML Q4_K (256 elements → 144 bytes).
const Q4_K: Storage = Storage::Ggml {
    block: 256,
    type_size: 144,
};
/// GGML Q4_0 (32 elements → 18 bytes): Q4 for an input width that is 32- but not 256-aligned.
const Q4_0: Storage = Storage::Ggml {
    block: 32,
    type_size: 18,
};

/// `precision::storage`: a matmul weight follows the tier; everything else stays BF16. An input
/// width the tier's blocks do not divide is an error, never a silent dense fallback.
fn storage(tier: Yue2Tier, class: TensorClass, cols: u64) -> Result<Storage, String> {
    if tier == Yue2Tier::Bf16 || !class.follows_tier() {
        return Ok(Storage::Dense);
    }
    let storage = match tier {
        Yue2Tier::Q8 => Q8_0,
        Yue2Tier::Q4 if cols.is_multiple_of(256) => Q4_K,
        Yue2Tier::Q4 => Q4_0,
        Yue2Tier::Bf16 => unreachable!("handled above"),
    };
    let Storage::Ggml { block, .. } = storage else {
        unreachable!("a quantized tier stores GGML blocks")
    };
    if cols == 0 || !cols.is_multiple_of(block) {
        return Err(format!(
            "a {tier:?} matmul weight with input width {cols} does not divide into {block}-element \
             GGML blocks"
        ));
    }
    Ok(storage)
}

fn stored_bytes(storage: Storage, elems: u64) -> u64 {
    match storage {
        Storage::Dense => elems * 2,
        Storage::Ggml { block, type_size } => elems / block * type_size,
    }
}

/// `precision::loaded_bytes`: dense weights in the compute dtype; GGML blocks as stored, plus
/// Candle's row padding on CUDA.
fn loaded_bytes(storage: Storage, elems: u64, backend: Yue2Backend, dense: u64) -> u64 {
    match storage {
        Storage::Dense => elems * dense,
        Storage::Ggml { block, type_size } => {
            let payload = elems / block * type_size;
            match backend {
                Yue2Backend::Cuda => payload + CUDA_GGML_ROW_PADDING * type_size / block,
                Yue2Backend::Cpu | Yue2Backend::Metal => payload,
            }
        }
    }
}

/// Resident weight bytes of the loaded MoT (both paths, embedding, `lm_head`, the NAR heads).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Yue2WeightResidency {
    /// Bytes on the model device while the AR stages run (host memory for a CPU model). Under FP8
    /// this is the FP8 layout; the acoustic stage and the decode run the restored BF16 layout
    /// ([`Self::restored_device_bytes`]).
    pub device_bytes: u64,
    /// Bytes held in host memory **in addition**: the FP8 mode's retained BF16 AR originals.
    pub host_bytes: u64,
    /// The native (non-FP8) resident layout — what the acoustic stage and the decode hold.
    pub restored_device_bytes: u64,
    /// Of [`Self::restored_device_bytes`], the AR-only weights AR offload moves to host memory.
    pub ar_only_bytes: u64,
    /// The tier's weights file (the safetensors payload the load maps).
    pub stored_bytes: u64,
}

/// Price the resident weights of `tier` on `backend` (`precision::weight_residency` over the
/// pinned tensor table). Unsupported combinations are refused explicitly.
pub(crate) fn weight_residency(
    tier: Yue2Tier,
    backend: Yue2Backend,
    precision: Yue2Precision,
    ar: Yue2ArMode,
    compute_cap: Option<f32>,
) -> Result<Yue2WeightResidency, String> {
    if ar == Yue2ArMode::Fp8 {
        fp8_supported(tier, backend, precision, compute_cap)?;
    }
    let dense = dense_bytes(backend, precision);
    let mut residency = Yue2WeightResidency::default();
    for group in tensor_table() {
        let elems = group.rows * group.cols;
        let storage = storage(tier, group.class, group.cols)?;
        let loaded = loaded_bytes(storage, elems, backend, dense) * group.count;
        residency.stored_bytes += stored_bytes(storage, elems) * group.count;
        residency.restored_device_bytes += loaded;
        if group.class.ar_only() {
            residency.ar_only_bytes += loaded;
        }
        if ar == Yue2ArMode::Fp8 && group.class == TensorClass::ArProjection {
            // FP8 E4M3 values plus one F32 scale each on the device; the BF16 originals in host
            // memory (`fp8::prepare_fp8_ar`).
            residency.device_bytes += (elems + 4) * group.count;
            residency.host_bytes += elems * 2 * group.count;
        } else {
            residency.device_bytes += loaded;
        }
    }
    Ok(residency)
}

/// `fp8::prepare_fp8_ar`'s preconditions: CUDA compute capability ≥ 8.9, the released BF16 weights,
/// BF16 compute. Any other combination is refused, never folded onto another precision.
fn fp8_supported(
    tier: Yue2Tier,
    backend: Yue2Backend,
    precision: Yue2Precision,
    compute_cap: Option<f32>,
) -> Result<(), String> {
    if backend != Yue2Backend::Cuda {
        return Err(format!(
            "the experimental FP8 AR mode runs only on CUDA (sm_89+); this render is on {}",
            backend.key()
        ));
    }
    if tier != Yue2Tier::Bf16 {
        return Err(format!(
            "the experimental FP8 AR mode needs the released bf16 weights, not the {} tier",
            tier.key()
        ));
    }
    if precision == Yue2Precision::Fp32 {
        return Err("the experimental FP8 AR mode needs BF16 compute, not an F32 load".into());
    }
    match compute_cap {
        Some(cap) if cap + 1e-3 >= FP8_COMPUTE_CAP_FLOOR => Ok(()),
        Some(cap) => Err(format!(
            "the experimental FP8 AR mode needs CUDA compute capability {FP8_COMPUTE_CAP_FLOOR}+; \
             this GPU is {cap}"
        )),
        None => Err(
            "the experimental FP8 AR mode needs CUDA compute capability 8.9+, and this GPU's could \
             not be read"
                .into(),
        ),
    }
}

// ---- The render ------------------------------------------------------------------------------------

/// How the score is obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Yue2Planning {
    /// The ABC stage samples a score of at most `max_tokens`.
    Sample { max_tokens: u64 },
    /// A supplied score (external ABC, a cover's reviewed score) or a restored plan: at most
    /// `abc_tokens` ABC tokens, nothing sampled.
    Supplied { abc_tokens: u64 },
    /// `cot = off`: no score.
    Off,
}

/// What the render does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Yue2Work {
    /// The full pipeline.
    Generate {
        planning: Yue2Planning,
        semantic_max_tokens: u64,
        /// Classifier-free guidance on the semantic stage (two KV caches).
        cfg: bool,
    },
    /// Plan and publish the plan only.
    PlanOnly { planning: Yue2Planning },
    /// Decode a completed run's cached latents of at most `frames` latent frames.
    DecodeCached { frames: u64 },
}

/// How a request fixes the NAR attention chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttentionPin {
    /// The historical 256-row query tiles (`chunk_attention: false`, or `true` without a size) —
    /// priced at the default chunk; `chunk_attention` is echoed back exactly as the job sent it.
    Historical { chunk_attention: bool },
    /// At most this many score elements per call.
    Elements(u64),
}

/// How a request fixes the VAE decode tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodePin {
    /// The production tiling (`tile_vae_decode: false`, or `true` without an edge);
    /// `tile_vae_decode` is echoed back exactly as sent.
    Production { tile_vae_decode: bool },
    /// This tile core, in latent frames.
    Core(u64),
}

/// Memory controls the request or its load already fixes. A fixed control is a user's knob: it is
/// priced as given and sent back unchanged. The gate chooses only the free ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Yue2Pins {
    pub offload_ar: Option<bool>,
    pub attention: Option<AttentionPin>,
    pub decode: Option<DecodePin>,
}

impl Yue2Pins {
    /// Pins from a job's own memory controls, field by field: a field the job set (`Some`) is
    /// pinned, a field it left unset stays free for the gate to choose. `offload` is the load's
    /// explicit offload policy (`Some(true)` Sequential, `Some(false)` Resident), read only when the
    /// job sets no `stage_residency`. Values the engine refuses are refused here too.
    pub(crate) fn from_controls(
        stage_residency: Option<bool>,
        offload: Option<bool>,
        chunk_attention: Option<bool>,
        attention_chunk_size: Option<u32>,
        tile_vae_decode: Option<bool>,
        decode_tile_edge: Option<u32>,
    ) -> Result<Self, String> {
        let attention = match (chunk_attention, attention_chunk_size) {
            (Some(true), Some(elements)) => {
                let elements = u64::from(elements);
                if elements < ATTENTION_ELEMENTS_MIN {
                    return Err(format!(
                        "memory.attention_chunk_size {elements} cannot hold one query row at the \
                         model's full context ({HEADS} heads × {CONTEXT} keys = \
                         {ATTENTION_ELEMENTS_MIN} elements)"
                    ));
                }
                Some(AttentionPin::Elements(elements))
            }
            (Some(chunk_attention), _) => Some(AttentionPin::Historical { chunk_attention }),
            (None, Some(_)) => {
                return Err("memory.attention_chunk_size is read only with chunk_attention".into())
            }
            (None, None) => None,
        };
        let decode = match (tile_vae_decode, decode_tile_edge) {
            (Some(true), Some(core)) => {
                let core = u64::from(core);
                if !(1..=DECODE_MAX_CORE_FRAMES).contains(&core) {
                    return Err(format!(
                        "memory.decode_tile_edge {core} is not a decode tile core of \
                         1..={DECODE_MAX_CORE_FRAMES} latent frames"
                    ));
                }
                Some(DecodePin::Core(core))
            }
            (Some(tile_vae_decode), _) => Some(DecodePin::Production { tile_vae_decode }),
            (None, Some(_)) => {
                return Err("memory.decode_tile_edge is read only with tile_vae_decode".into())
            }
            (None, None) => None,
        };
        Ok(Self {
            offload_ar: stage_residency.or(offload),
            attention,
            decode,
        })
    }

    /// Pins from a request that already carries a complete `memory` block (every field is then a
    /// decision the request made), or from the load alone when it carries none.
    pub(crate) fn of_request(
        memory: Option<&GenerationMemory>,
        sequential_offload: bool,
    ) -> Result<Self, String> {
        match memory {
            None => Ok(Self {
                offload_ar: sequential_offload.then_some(true),
                ..Self::default()
            }),
            Some(memory) => Self::from_controls(
                Some(memory.stage_residency),
                None,
                Some(memory.chunk_attention),
                memory.attention_chunk_size,
                Some(memory.tile_vae_decode),
                memory.decode_tile_edge,
            ),
        }
    }

    fn attention_elements(&self) -> Option<u64> {
        self.attention.map(|pin| match pin {
            AttentionPin::Historical { .. } => ATTENTION_ELEMENTS_DEFAULT,
            AttentionPin::Elements(elements) => elements,
        })
    }

    fn decode_core_frames(&self) -> Option<u64> {
        self.decode.map(|pin| match pin {
            DecodePin::Production { .. } => default_decode_core(),
            DecodePin::Core(core) => core,
        })
    }

    fn any(&self) -> bool {
        self.offload_ar.is_some() || self.attention.is_some() || self.decode.is_some()
    }

    /// The request's `memory` block: every pinned control exactly as the request set it, every free
    /// one as the gate chose it (explicitly, so what runs is what was priced).
    pub(crate) fn memory_block(&self, chosen: &Yue2Controls) -> GenerationMemory {
        let chosen_block = chosen.generation_memory();
        let (chunk_attention, attention_chunk_size) = match self.attention {
            Some(AttentionPin::Historical { chunk_attention }) => (chunk_attention, None),
            Some(AttentionPin::Elements(elements)) => (
                true,
                Some(u32::try_from(elements).expect("a u32 the request sent")),
            ),
            None => (true, chosen_block.attention_chunk_size),
        };
        let (tile_vae_decode, decode_tile_edge) = match self.decode {
            Some(DecodePin::Production { tile_vae_decode }) => (tile_vae_decode, None),
            Some(DecodePin::Core(core)) => (
                true,
                Some(u32::try_from(core).expect("a u32 the request sent")),
            ),
            None => (true, chosen_block.decode_tile_edge),
        };
        GenerationMemory {
            stage_residency: self.offload_ar.unwrap_or(chosen.offload_ar),
            chunk_attention,
            attention_chunk_size,
            tile_vae_decode,
            decode_tile_edge,
            ..GenerationMemory::default()
        }
    }
}

/// What the load fixes: the tier it loads, its dense precision and its offload policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Yue2LoadFacts {
    pub tier: Yue2Tier,
    pub precision: Yue2Precision,
    /// `LoadSpec::offload_policy == Sequential`: the AR weights offload during the acoustic stage
    /// whenever the request does not choose otherwise.
    pub sequential_offload: bool,
}

impl Yue2LoadFacts {
    /// The facts of `spec` loading `tier`.
    pub(crate) fn of(tier: Yue2Tier, spec: &gen_core::LoadSpec) -> Self {
        Self {
            tier,
            precision: match spec.precision {
                gen_core::Precision::Fp32 => Yue2Precision::Fp32,
                _ => Yue2Precision::Default,
            },
            sequential_offload: spec.offload_policy == gen_core::OffloadPolicy::Sequential,
        }
    }
}

/// Everything a YuE2 render's memory depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Yue2Shape {
    pub tier: Yue2Tier,
    pub precision: Yue2Precision,
    pub ar: Yue2ArMode,
    /// Upper bound on the text tokens of the prompt (style + lyrics + framing).
    pub text_tokens: u64,
    pub work: Yue2Work,
    /// The selected decoder's FP32 weights file (bytes), from the catalog.
    pub decoder_bytes: u64,
    /// Seconds of source recording to transcribe first (a cover from a recording), when any.
    pub transcription_secs: Option<u64>,
    /// SheetSage2 + MERT-v2 weights files (bytes, F32), when transcription runs.
    pub transcriber_bytes: u64,
    /// Controls the request or its load fixes.
    pub pins: Yue2Pins,
}

impl Yue2Shape {
    fn planner_prefix(&self) -> u64 {
        self.text_tokens
    }

    /// The semantic (and acoustic) prefix: the planner prefix, the score and its framing.
    fn semantic_prefix(&self, planning: Yue2Planning) -> u64 {
        let abc = match planning {
            Yue2Planning::Sample { max_tokens } => max_tokens,
            Yue2Planning::Supplied { abc_tokens } => abc_tokens,
            Yue2Planning::Off => 0,
        };
        (self.text_tokens + abc + 3).min(CONTEXT)
    }
}

/// The memory controls this gate chooses for a render (`GenerationMemory`'s YuE2 fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Yue2Controls {
    /// `stage_residency`: the AR-only weights in host memory while each acoustic chunk is solved.
    pub offload_ar: bool,
    /// `attention_chunk_size`: at most this many NAR attention-score elements per call.
    pub attention_elements: u64,
    /// `decode_tile_edge`: the VAE tile core, in latent frames.
    pub decode_core_frames: u64,
}

/// The largest NAR score chunk this gate selects: the historical 256-row tile at the full context.
const ATTENTION_ELEMENTS_DEFAULT: u64 = HEADS * EAGER_QUERY_TILE * CONTEXT;
/// The smallest the engine accepts: one query row at the full context (`heads × positions`). One
/// constant, shared with the job contract's submission check.
const ATTENTION_ELEMENTS_MIN: u64 =
    sceneworks_core::yue2_score::jobs::MIN_ATTENTION_CHUNK_ELEMENTS as u64;

/// `DecodeOptions::production()`'s core: the largest whose tile fits the 8 GiB production budget.
fn default_decode_core() -> u64 {
    ((DECODE_DEFAULT_BUDGET_BYTES - DECODE_TILE_RESERVE_BYTES) / DECODE_TILE_BYTES_PER_FRAME)
        .saturating_sub(2 * DECODE_HALO_FRAMES)
        .min(DECODE_MAX_CORE_FRAMES)
}

impl Yue2Controls {
    /// The engine's own production behaviour: no offload, the historical 256-row tiles, the
    /// production decode tiling.
    pub(crate) fn production() -> Self {
        Self {
            offload_ar: false,
            attention_elements: ATTENTION_ELEMENTS_DEFAULT,
            decode_core_frames: default_decode_core(),
        }
    }

    /// The request's `memory` block. Every field YuE2 reads is set explicitly, so what runs is
    /// exactly what was priced (the engine refuses any field it does not honour).
    pub(crate) fn generation_memory(&self) -> GenerationMemory {
        // The engine refuses a chunk that cannot hold one query row at the full context.
        debug_assert!(self.attention_elements >= ATTENTION_ELEMENTS_MIN);
        GenerationMemory {
            stage_residency: self.offload_ar,
            chunk_attention: true,
            attention_chunk_size: Some(
                u32::try_from(self.attention_elements).expect("bounded by the default chunk"),
            ),
            tile_vae_decode: true,
            decode_tile_edge: Some(
                u32::try_from(self.decode_core_frames).expect("bounded by the 1024-frame core"),
            ),
            ..GenerationMemory::default()
        }
    }
}

/// A stage of a render.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Yue2Stage {
    Transcription,
    Load,
    Plan,
    Semantic,
    AcousticPrefill,
    AcousticSolve,
    Decode,
}

impl Yue2Stage {
    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::Transcription => "transcription",
            Self::Load => "load",
            Self::Plan => "plan",
            Self::Semantic => "semantic",
            Self::AcousticPrefill => "acoustic_prefill",
            Self::AcousticSolve => "acoustic_solve",
            Self::Decode => "decode",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Transcription => "the recording transcription (SheetSage2 + MERT-v2)",
            Self::Load => "the model load",
            Self::Plan => "the ABC plan stage",
            Self::Semantic => "the semantic-token stage",
            Self::AcousticPrefill => "the acoustic stage's per-chunk AR prefill",
            Self::AcousticSolve => "the acoustic flow-matching solve",
            Self::Decode => "the FP32 VAE decode",
        }
    }
}

/// One named term of a stage's residency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Yue2Term {
    pub what: &'static str,
    pub device_bytes: u64,
    pub host_bytes: u64,
}

/// One stage's residency: the device pool and the host pool (one pool on a unified backend).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Yue2StageResidency {
    pub stage: Yue2Stage,
    pub terms: Vec<Yue2Term>,
}

impl Yue2StageResidency {
    pub(crate) fn device_bytes(&self) -> u64 {
        self.terms.iter().map(|t| t.device_bytes).sum()
    }

    pub(crate) fn host_bytes(&self) -> u64 {
        self.terms.iter().map(|t| t.host_bytes).sum()
    }

    /// Everything, for a unified pool.
    pub(crate) fn total_bytes(&self) -> u64 {
        self.device_bytes() + self.host_bytes()
    }
}

/// What the numbers rest on, per backend (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Yue2Evidence {
    /// CPU: weights (sc-22995) and the decode tile (sc-22993) measured; activations shape-derived.
    CpuMeasuredWeightsAndTile,
    /// CUDA: weights measured on CUDA (sc-22995); activations shape-derived and the decode tile
    /// CPU-measured — pending the terminal calibration campaign (sc-23002).
    CudaMeasuredWeightsEstimatePending,
    /// Metal: no YuE2 Metal measurement exists — an estimate pending the terminal calibration
    /// campaign (sc-23002).
    MetalEstimatePending,
}

impl Yue2Evidence {
    pub(crate) fn of(backend: Yue2Backend) -> Self {
        match backend {
            Yue2Backend::Cpu => Self::CpuMeasuredWeightsAndTile,
            Yue2Backend::Cuda => Self::CudaMeasuredWeightsEstimatePending,
            Yue2Backend::Metal => Self::MetalEstimatePending,
        }
    }

    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::CpuMeasuredWeightsAndTile => "measured_weights_and_decode_tile",
            Self::CudaMeasuredWeightsEstimatePending => {
                "measured_weights_estimate_pending_terminal_calibration"
            }
            Self::MetalEstimatePending => "estimate_pending_terminal_calibration",
        }
    }

    fn sentence(self) -> &'static str {
        match self {
            Self::CpuMeasuredWeightsAndTile => {
                "CPU weights and decode tile are measured (sc-22995, sc-22993); the AR/NAR working \
                 sets are derived from the model's shapes"
            }
            Self::CudaMeasuredWeightsEstimatePending => {
                "CUDA weights are measured (sc-22995); the working sets are shape-derived and the \
                 decode tile CPU-measured — an estimate pending the terminal calibration (sc-23002)"
            }
            Self::MetalEstimatePending => {
                "no YuE2 Metal measurement exists yet — this is a shape-derived estimate pending \
                 the terminal calibration (sc-23002)"
            }
        }
    }
}

/// A priced render: each stage's residency under `controls`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Yue2Estimate {
    pub tier: Yue2Tier,
    pub backend: Yue2Backend,
    pub controls: Yue2Controls,
    pub weights: Yue2WeightResidency,
    pub stages: Vec<Yue2StageResidency>,
}

impl Yue2Estimate {
    pub(crate) fn stage(&self, stage: Yue2Stage) -> Option<&Yue2StageResidency> {
        self.stages.iter().find(|s| s.stage == stage)
    }

    /// The stage that binds a unified pool and its bytes — the floor.
    pub(crate) fn unified_floor(&self) -> (Yue2Stage, u64) {
        self.stages
            .iter()
            .map(|s| (s.stage, s.total_bytes()))
            .max_by_key(|&(_, bytes)| bytes)
            .expect("every render has a load stage")
    }

    /// The largest device-pool residency of any stage.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn device_peak(&self) -> (Yue2Stage, u64) {
        self.stages
            .iter()
            .map(|s| (s.stage, s.device_bytes()))
            .max_by_key(|&(_, bytes)| bytes)
            .expect("every render has a load stage")
    }

    /// The largest host-pool residency of any stage.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn host_peak(&self) -> (Yue2Stage, u64) {
        self.stages
            .iter()
            .map(|s| (s.stage, s.host_bytes()))
            .max_by_key(|&(_, bytes)| bytes)
            .expect("every render has a load stage")
    }
}

fn kv_cache_bytes(positions: u64, dense: u64) -> u64 {
    LAYERS * 2 * KV_HEADS * HEAD_DIM * positions.min(CONTEXT) * dense
}

/// One AR prefill forward's transient working set over `keys` visible keys: the score tiles of one
/// 256-row query tile, the SwiGLU inner activations (gate, up, product) and the residual-width
/// activations of a 512-position chunk, and the host logits rows the sampler shapes.
fn ar_prefill_workspace_bytes(keys: u64, dense: u64) -> u64 {
    let rows = PREFILL_CHUNK.min(EAGER_QUERY_TILE);
    SCORE_TILES_LIVE * HEADS * rows * keys.min(CONTEXT) * dense
        + 3 * PREFILL_CHUNK * INTERMEDIATE * dense
        + 4 * PREFILL_CHUNK * HIDDEN * dense
        + 4 * VOCAB * 4
}

/// The NAR score tiles of one attention call under a score-element budget of `elements`, over a
/// chunk of `rows` query rows and `keys` keys (a chunk smaller than the budget holds only itself).
fn nar_score_bytes(elements: u64, rows: u64, keys: u64, dense: u64) -> u64 {
    SCORE_TILES_LIVE * elements.min(HEADS * rows * keys) * dense
}

/// The NAR evaluation's untiled activations over `rows` query rows: the SwiGLU inner activations and
/// the residual-width ones (input, normed, attention output, velocity head input).
fn nar_activation_bytes(rows: u64, dense: u64) -> u64 {
    3 * rows * INTERMEDIATE * dense + 4 * rows * HIDDEN * dense
}

/// The song's latents: the F32 noise over the whole song on the host, plus the chunk's ODE state,
/// midpoint state and two velocities in the compute dtype.
fn latent_bytes(song_frames: u64, rows: u64, dense: u64) -> u64 {
    song_frames * LATENT_DIM * 4 + 4 * rows * LATENT_DIM * dense
}

/// The per-frame part of `DecodeOptions::estimated_peak_bytes`: one tile of `core + 2 × halo`
/// latent frames, or the whole song when it is shorter than a tile (its fixed part,
/// `TILE_RESERVE_BYTES`, is the decode stage's "decoder + reserve" term).
fn decode_tile_activation_bytes(core: u64, song_frames: u64) -> u64 {
    let tile = (core + 2 * DECODE_HALO_FRAMES).min(song_frames.max(1));
    DECODE_TILE_BYTES_PER_FRAME * tile
}

fn waveform_bytes(frames: u64) -> u64 {
    WAVEFORM_COPIES * frames * SAMPLES_PER_LATENT_FRAME * AUDIO_CHANNELS * 4
        + frames * LATENT_DIM * 4
}

/// SheetSage2 + MERT-v2 over one window: the weights (plus, on CUDA, their mapped files in host
/// RAM while loading — see the load stage); in the encoder, one query chunk's score tiles over the
/// whole window and its SwiGLU/residual activations.
fn transcription_terms(secs: u64, transcriber_bytes: u64, backend: Yue2Backend) -> Vec<Yue2Term> {
    let frames = secs.min(TRANSCRIPTION_WINDOW_SECS) * MERT_FRAMES_PER_SEC;
    let encoder = SCORE_TILES_LIVE * MERT_HEADS * MERT_QUERY_CHUNK.min(frames) * frames * 4
        + (4 * MERT_HIDDEN + 4 * MERT_FFN) * frames * 4;
    vec![
        Yue2Term {
            what: "SheetSage2 + MERT-v2 weights (F32)",
            device_bytes: transcriber_bytes,
            host_bytes: 0,
        },
        Yue2Term {
            what: "their mapped weights files while loading",
            device_bytes: 0,
            host_bytes: mapped_file_bytes(backend, transcriber_bytes),
        },
        Yue2Term {
            what: "one 300 s encoder window",
            device_bytes: encoder,
            host_bytes: 0,
        },
    ]
}

/// The host-pool bytes a weights file mapped during a load costs on `backend`. Mapped safetensors
/// pages are clean and file-backed: in unified memory (Metal, CPU) they are not Metal buffers, do not
/// count against the GPU working set (or the process footprint) and are reclaimable, so they cost
/// nothing there. On CUDA they sit in host RAM beside the device copy and are charged to the host
/// pool, which is compared with `MemAvailable`.
fn mapped_file_bytes(backend: Yue2Backend, file_bytes: u64) -> u64 {
    match backend {
        Yue2Backend::Cuda => file_bytes,
        Yue2Backend::Cpu | Yue2Backend::Metal => 0,
    }
}

/// Price `shape` on `backend` under `controls`.
pub(crate) fn estimate(
    shape: &Yue2Shape,
    backend: Yue2Backend,
    controls: Yue2Controls,
    compute_cap: Option<f32>,
) -> Result<Yue2Estimate, String> {
    let weights = weight_residency(shape.tier, backend, shape.precision, shape.ar, compute_cap)?;
    let dense = dense_bytes(backend, shape.precision);
    let term = |what, device_bytes, host_bytes| Yue2Term {
        what,
        device_bytes,
        host_bytes,
    };
    // A unified backend has one pool: "host" bytes are still in it, and offload moves nothing.
    let offload = controls.offload_ar && !backend.unified();
    let mut stages = Vec::new();

    if let Some(secs) = shape.transcription_secs {
        stages.push(Yue2StageResidency {
            stage: Yue2Stage::Transcription,
            terms: transcription_terms(secs, shape.transcriber_bytes, backend),
        });
    }
    stages.push(Yue2StageResidency {
        stage: Yue2Stage::Load,
        terms: vec![
            term("resident weights", weights.restored_device_bytes, 0),
            term(
                "the mapped weights file",
                0,
                mapped_file_bytes(backend, weights.stored_bytes),
            ),
        ],
    });
    // Resident weights while the AR stages run (FP8: FP8 on the device, originals on the host).
    let ar_weights = || {
        vec![
            term("resident weights", weights.device_bytes, 0),
            term("retained FP8 BF16 originals", 0, weights.host_bytes),
        ]
    };
    let mut ar_stage = |stage, planning_prefix: u64, max_tokens: u64, branches: u64| {
        let positions = (planning_prefix + max_tokens).min(CONTEXT);
        let mut terms = ar_weights();
        terms.push(term(
            if branches == 2 {
                "two KV caches (CFG branches)"
            } else {
                "KV cache"
            },
            branches * kv_cache_bytes(positions, dense),
            0,
        ));
        terms.push(term(
            "AR prefill workspace",
            ar_prefill_workspace_bytes(positions, dense),
            0,
        ));
        stages.push(Yue2StageResidency { stage, terms });
    };
    let (planning, song_frames) = match shape.work {
        Yue2Work::Generate {
            planning,
            semantic_max_tokens,
            cfg,
        } => {
            if let Yue2Planning::Sample { max_tokens } = planning {
                ar_stage(Yue2Stage::Plan, shape.planner_prefix(), max_tokens, 1);
            }
            let prefix = shape.semantic_prefix(planning);
            ar_stage(
                Yue2Stage::Semantic,
                prefix,
                semantic_max_tokens,
                if cfg { 2 } else { 1 },
            );
            // The song can be as long as the SHORTEST prefix allows (no score, no text: the
            // protocol checks `prefix + max_tokens <= CONTEXT` against the real prefix).
            (
                Some(planning),
                semantic_max_tokens.min(CONTEXT - MIN_SEMANTIC_PREFIX),
            )
        }
        Yue2Work::PlanOnly { planning } => {
            if let Yue2Planning::Sample { max_tokens } = planning {
                ar_stage(Yue2Stage::Plan, shape.planner_prefix(), max_tokens, 1);
            }
            (None, 0)
        }
        Yue2Work::DecodeCached { frames } => (None, frames.min(CONTEXT)),
    };

    if let Some(planning) = planning {
        // `protocol::chunk_ranges`: chunks of `(CONTEXT − prefix − 3) / 2` frames; each chunk's AR
        // sequence is the prefix, its codec ids and MUSIC_END, its NAR sequence its latents plus two
        // slots, and its cache holds both. A SHORTER real prefix makes a LONGER chunk, so the NAR
        // rows are sized with the smallest prefix the protocol allows, while the AR prefill and the
        // cache take the upper-bound prefix (capped at the context, which bounds every real cache).
        let prefix = shape.semantic_prefix(planning);
        let chunk = ((CONTEXT - MIN_SEMANTIC_PREFIX - 3) / 2)
            .min(song_frames)
            .max(1);
        let ar_len = (prefix + chunk + 1).min(CONTEXT);
        let nar_rows = chunk + 2;
        let cache_positions = (prefix + chunk + 1 + nar_rows).min(CONTEXT);
        let restored = || term("resident weights", weights.restored_device_bytes, 0);
        stages.push(Yue2StageResidency {
            stage: Yue2Stage::AcousticPrefill,
            terms: vec![
                restored(),
                term("chunk KV cache", kv_cache_bytes(cache_positions, dense), 0),
                term(
                    "AR prefill workspace",
                    ar_prefill_workspace_bytes(ar_len, dense),
                    0,
                ),
            ],
        });
        let mut solve = vec![restored()];
        if offload {
            solve.push(term(
                "AR-only weights offloaded to host",
                0,
                weights.ar_only_bytes,
            ));
            solve[0].device_bytes -= weights.ar_only_bytes;
        }
        solve.extend([
            term("chunk KV cache", kv_cache_bytes(cache_positions, dense), 0),
            term(
                "NAR attention score tiles",
                nar_score_bytes(
                    controls.attention_elements,
                    nar_rows,
                    cache_positions,
                    dense,
                ),
                0,
            ),
            term("NAR activations", nar_activation_bytes(nar_rows, dense), 0),
            term("latents", latent_bytes(song_frames, nar_rows, dense), 0),
        ]);
        stages.push(Yue2StageResidency {
            stage: Yue2Stage::AcousticSolve,
            terms: solve,
        });
    }
    if song_frames > 0 {
        stages.push(Yue2StageResidency {
            stage: Yue2Stage::Decode,
            terms: vec![
                term("resident weights", weights.restored_device_bytes, 0),
                // The engine's `TILE_RESERVE_BYTES` (1 GiB) already covers the process, the FP32
                // decoder-only weights (folded at load) and their mapped pages, so the decoder is not
                // charged again on top of it — unless a decoder file ever outgrows the reserve.
                term(
                    "FP32 VAE decoder + decode reserve",
                    DECODE_TILE_RESERVE_BYTES.max(shape.decoder_bytes),
                    0,
                ),
                term(
                    "VAE decode tile activations",
                    decode_tile_activation_bytes(controls.decode_core_frames, song_frames),
                    0,
                ),
                term("song waveform", waveform_bytes(song_frames), 0),
            ],
        });
    }
    Ok(Yue2Estimate {
        tier: shape.tier,
        backend,
        controls,
        weights,
        stages,
    })
}

// ---- Budgets and the decision -----------------------------------------------------------------------

/// The memory a render may use.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Yue2Budget {
    /// One pool (Apple unified memory, or a CPU host): every byte counts against `capacity_bytes`,
    /// less `resident_bytes` other allocations of this process already hold there (on macOS the
    /// MLX allocator's active + cached bytes — the image generator the cache keeps warm for 300 s,
    /// the refine model). `reclaimable_bytes` of those are freed by evicting the cached generator
    /// and clearing the MLX cache.
    #[cfg_attr(not(any(test, target_os = "macos")), allow(dead_code))]
    Unified {
        backend: Yue2Backend,
        capacity_bytes: u64,
        resident_bytes: u64,
        reclaimable_bytes: u64,
    },
    /// A CUDA card: live free VRAM, its total, what evicting the cached generator would return, and
    /// the host RAM available right now (`None`: unread — the host pool is then not checked).
    #[cfg_attr(
        not(any(test, all(not(target_os = "macos"), feature = "backend-candle"))),
        allow(dead_code)
    )]
    Dedicated {
        free_bytes: u64,
        total_bytes: u64,
        reclaimable_bytes: u64,
        host_available_bytes: Option<u64>,
        gpu_id: String,
        compute_cap: Option<f32>,
    },
}

impl Yue2Budget {
    fn backend(&self) -> Yue2Backend {
        match self {
            Self::Unified { backend, .. } => *backend,
            Self::Dedicated { .. } => Yue2Backend::Cuda,
        }
    }

    fn compute_cap(&self) -> Option<f32> {
        match self {
            Self::Dedicated { compute_cap, .. } => *compute_cap,
            Self::Unified { .. } => None,
        }
    }
}

/// The dedicated-VRAM allocator/context reserve charged on top of the device estimate.
fn dedicated_reserve_bytes() -> u64 {
    (crate::fit_gate::DEDICATED_VRAM_ALLOCATOR_SLACK_GB * BYTES_PER_GIB) as u64
}

/// Whether an estimate fits a budget, and if not, which stage and pool binds.
#[derive(Clone, Debug, PartialEq)]
enum Fit {
    Fits,
    FitsAfterEvict,
    Short {
        stage: Yue2Stage,
        pool: Pool,
        needed: u64,
        available: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pool {
    Unified,
    Device,
    Host,
}

/// `other_live` is the residency other live YuE2 leases hold (unified pools only: a dedicated card's
/// free reading already reflects what they hold).
fn fit(estimate: &Yue2Estimate, budget: &Yue2Budget, other_live: u64) -> Fit {
    match budget {
        Yue2Budget::Unified {
            capacity_bytes,
            resident_bytes,
            reclaimable_bytes,
            ..
        } => {
            let base = capacity_bytes.saturating_sub(other_live);
            let available = base.saturating_sub(*resident_bytes);
            let reclaimed = base.saturating_sub(resident_bytes.saturating_sub(*reclaimable_bytes));
            let short =
                |available: u64| estimate.stages.iter().find(|s| s.total_bytes() > available);
            if short(available).is_none() {
                Fit::Fits
            } else if short(reclaimed).is_none() {
                Fit::FitsAfterEvict
            } else {
                let s = short(reclaimed).expect("checked above");
                Fit::Short {
                    stage: s.stage,
                    pool: Pool::Unified,
                    needed: s.total_bytes(),
                    available: reclaimed,
                }
            }
        }
        Yue2Budget::Dedicated {
            free_bytes,
            total_bytes,
            reclaimable_bytes,
            host_available_bytes,
            ..
        } => {
            let reserve = dedicated_reserve_bytes();
            if let Some(host) = host_available_bytes {
                if let Some(s) = estimate.stages.iter().find(|s| s.host_bytes() > *host) {
                    return Fit::Short {
                        stage: s.stage,
                        pool: Pool::Host,
                        needed: s.host_bytes(),
                        available: *host,
                    };
                }
            }
            let reclaimed = (free_bytes + reclaimable_bytes).min(*total_bytes);
            let short = |available: u64| {
                estimate
                    .stages
                    .iter()
                    .find(|s| s.device_bytes() + reserve > available)
            };
            if short(*free_bytes).is_none() {
                Fit::Fits
            } else if short(reclaimed).is_none() {
                Fit::FitsAfterEvict
            } else {
                let s = short(reclaimed).expect("checked above");
                Fit::Short {
                    stage: s.stage,
                    pool: Pool::Device,
                    needed: s.device_bytes() + reserve,
                    available: reclaimed,
                }
            }
        }
    }
}

/// NAR score chunks, largest first: the historical 256-row tile at the full context, halving to one
/// row.
fn attention_candidates() -> impl Iterator<Item = u64> {
    // Halve the rows from the historical 256 to one: the last candidate is exactly
    // `ATTENTION_ELEMENTS_MIN`.
    (0..=EAGER_QUERY_TILE.trailing_zeros()).map(|shift| ATTENTION_ELEMENTS_DEFAULT >> shift)
}

/// Choose the controls under which `shape` fits `budget`: the production controls when they fit;
/// otherwise the largest NAR score chunk (then, on CUDA only, AR offload) that lets the acoustic
/// solve fit, and the largest decode tile that lets the decode fit. Each control shrinks only its
/// own stage, so a shortfall in any other stage is final. `Err` carries the estimate at the
/// smallest controls tried and its shortfall.
fn choose(
    shape: &Yue2Shape,
    budget: &Yue2Budget,
    other_live: u64,
) -> Result<(Yue2Estimate, Fit), (Yue2Estimate, Fit)> {
    let backend = budget.backend();
    let price = |controls| {
        estimate(shape, backend, controls, budget.compute_cap())
            .expect("the combination was validated before choosing")
    };
    let pins = shape.pins;
    let production_core = pins
        .decode_core_frames()
        .unwrap_or(Yue2Controls::production().decode_core_frames);
    // AR offload moves bytes into the SAME pool on a unified backend, so it is never chosen there.
    let offloads: Vec<bool> = match pins.offload_ar {
        Some(pinned) => vec![pinned],
        None if backend.unified() => vec![false],
        None => vec![false, true],
    };
    let attentions: Vec<u64> = match pins.attention_elements() {
        Some(pinned) => vec![pinned],
        None => attention_candidates().collect(),
    };
    let mut tightest = None;
    for &offload_ar in &offloads {
        for &attention_elements in &attentions {
            let controls = Yue2Controls {
                offload_ar,
                attention_elements,
                decode_core_frames: production_core,
            };
            let estimate = price(controls);
            let verdict = fit(&estimate, budget, other_live);
            match verdict {
                Fit::Fits | Fit::FitsAfterEvict => return Ok((estimate, verdict)),
                Fit::Short {
                    stage: Yue2Stage::AcousticSolve,
                    ..
                } => {
                    // Report the shortfall of the first offload choice (resident AR weights where
                    // offload is free to choose): a later pass that also fails, e.g. on the host
                    // pool the offload fills, would otherwise hide the device shortfall.
                    if tightest.is_none() || offload_ar == offloads[0] {
                        tightest = Some((estimate, verdict));
                    }
                }
                Fit::Short {
                    stage: Yue2Stage::Decode,
                    ..
                } => {
                    // Every earlier stage fits; only the tile can help. The largest core that fits
                    // (none to search when the request fixes the tile).
                    let mut smallest = None;
                    let free_cores = if pins.decode.is_some() {
                        1..1
                    } else {
                        1..production_core
                    };
                    for core in free_cores.rev() {
                        let estimate = price(Yue2Controls {
                            decode_core_frames: core,
                            ..controls
                        });
                        match fit(&estimate, budget, other_live) {
                            verdict @ (Fit::Fits | Fit::FitsAfterEvict) => {
                                return Ok((estimate, verdict))
                            }
                            verdict => smallest = Some((estimate, verdict)),
                        }
                    }
                    return Err(smallest.unwrap_or_else(|| {
                        let estimate = price(controls);
                        let verdict = fit(&estimate, budget, other_live);
                        (estimate, verdict)
                    }));
                }
                Fit::Short { .. } => return Err((estimate, verdict)),
            }
        }
    }
    Err(tightest.expect("the acoustic solve was short at every candidate"))
}

/// The admission decision.
#[derive(Debug)]
pub(crate) enum Yue2Admission {
    /// Fits now under these controls.
    Admit(Yue2Estimate),
    /// Fits once the cached generator is evicted and its pool reclaimed.
    AdmitAfterEvict(Yue2Estimate),
    Refuse(WorkerError),
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB
}

/// The pure decision for `shape` against `budget` (`None`: no reading — admitted under the
/// production controls; the gate never blocks without evidence). `other_live` is the residency other
/// live YuE2 leases hold.
pub(crate) fn decide(
    model: &str,
    shape: &Yue2Shape,
    budget: Option<&Yue2Budget>,
    other_live: u64,
) -> Yue2Admission {
    let Some(budget) = budget else {
        return match estimate(
            shape,
            Yue2Backend::of_this_build(),
            Yue2Controls::production(),
            None,
        ) {
            Ok(estimate) => Yue2Admission::Admit(estimate),
            // An unsupported combination is refused even without a budget reading.
            Err(why) => Yue2Admission::Refuse(unsupported(model, &why)),
        };
    };
    // Unsupported tier/backend/precision combinations refuse before any pricing loop.
    if let Err(why) = weight_residency(
        shape.tier,
        budget.backend(),
        shape.precision,
        shape.ar,
        budget.compute_cap(),
    ) {
        return Yue2Admission::Refuse(unsupported(model, &why));
    }
    match choose(shape, budget, other_live) {
        Ok((estimate, Fit::FitsAfterEvict)) => Yue2Admission::AdmitAfterEvict(estimate),
        Ok((estimate, _)) => Yue2Admission::Admit(estimate),
        Err((estimate, verdict)) => Yue2Admission::Refuse(refusal(
            model, shape, budget, other_live, &estimate, &verdict,
        )),
    }
}

fn unsupported(model: &str, why: &str) -> WorkerError {
    WorkerError::InvalidPayload(format!("{model}: {why}."))
}

/// The refusal: which stage, how much over, what would fit.
fn refusal(
    model: &str,
    shape: &Yue2Shape,
    budget: &Yue2Budget,
    other_live: u64,
    estimate: &Yue2Estimate,
    verdict: &Fit,
) -> WorkerError {
    let Fit::Short {
        stage,
        pool,
        needed,
        available,
    } = verdict
    else {
        unreachable!("a refusal carries a shortfall")
    };
    let breakdown = estimate
        .stage(*stage)
        .map(|s| {
            s.terms
                .iter()
                .filter_map(|t| {
                    let bytes = match pool {
                        Pool::Unified => t.device_bytes + t.host_bytes,
                        Pool::Device => t.device_bytes,
                        Pool::Host => t.host_bytes,
                    };
                    (bytes > 0).then(|| format!("~{:.1} GB {}", gb(bytes), t.what))
                })
                .collect::<Vec<_>>()
                .join(" + ")
        })
        .unwrap_or_default();
    let pool_text = match (pool, budget) {
        (
            Pool::Unified,
            Yue2Budget::Unified {
                capacity_bytes,
                resident_bytes,
                reclaimable_bytes,
                ..
            },
        ) => {
            let mut held = Vec::new();
            if other_live > 0 {
                held.push(format!(
                    "~{:.1} GB of it is still held by another YuE2 render that has not \
                     released its memory yet",
                    gb(other_live)
                ));
            }
            let pinned = resident_bytes.saturating_sub(*reclaimable_bytes);
            if pinned > 0 {
                held.push(format!(
                    "~{:.1} GB is held by other models in this process that evicting the cached \
                     generator does not free",
                    gb(pinned)
                ));
            }
            let held = if held.is_empty() {
                String::new()
            } else {
                format!(" ({})", held.join("; "))
            };
            format!(
                "this machine's GPU working set is ~{:.1} GB{held}",
                gb(*capacity_bytes)
            )
        }
        (
            Pool::Device,
            Yue2Budget::Dedicated {
                total_bytes,
                gpu_id,
                ..
            },
        ) if *needed <= *total_bytes => format!(
            "GPU {gpu_id} has ~{:.1} GB in total — enough — but only ~{:.1} GB is available right \
             now: another process or model is holding VRAM. Free it and retry",
            gb(*total_bytes),
            gb(*available)
        ),
        (Pool::Device, Yue2Budget::Dedicated { gpu_id, .. }) => format!(
            "GPU {gpu_id} has only ~{:.1} GB of VRAM (incl. the {:.1} GB allocator reserve)",
            gb(*available),
            gb(dedicated_reserve_bytes())
        ),
        (Pool::Host, _) => format!("only ~{:.1} GB of host RAM is available", gb(*available)),
        _ => format!("only ~{:.1} GB is available", gb(*available)),
    };
    let c = estimate.controls;
    let controls = match stage {
        Yue2Stage::Decode => format!(
            " The VAE decode tile was tried down to a {}-frame core.",
            c.decode_core_frames
        ),
        Yue2Stage::AcousticSolve => format!(
            " The NAR attention chunk was tried down to {} score elements{}.",
            c.attention_elements,
            match (
                c.offload_ar,
                shape.pins.offload_ar,
                budget.backend().unified()
            ) {
                (true, _, _) => ", with the AR weights offloaded to host memory",
                (false, Some(false), _) => " (AR offload is disabled by the request)",
                (false, _, true) => " (AR offload frees nothing in unified memory)",
                (false, _, false) => "",
            }
        ),
        _ => String::new(),
    };
    let mut alternatives = alternatives(shape, budget, other_live, *stage);
    if shape.pins.any() {
        let free = Yue2Shape {
            pins: Yue2Pins::default(),
            ..*shape
        };
        if let Ok((fitting, _)) = choose(&free, budget, other_live) {
            let c = fitting.controls;
            alternatives = format!(
                "The requested memory controls do not fit, but these do: stage_residency {}, \
                 attention_chunk_size {} (with chunk_attention), decode_tile_edge {} (with \
                 tile_vae_decode) — or leave them unset and admission chooses. {alternatives}",
                c.offload_ar, c.attention_elements, c.decode_core_frames
            );
        }
    }
    let evidence = Yue2Evidence::of(budget.backend()).sentence();
    WorkerError::InvalidPayload(format!(
        "{model}: this render needs ~{:.1} GB for {} at the {} tier on {} ({breakdown}), but \
         {pool_text} — ~{:.1} GB short.{controls} {alternatives} ({evidence}.)",
        gb(*needed),
        stage.label(),
        shape.tier.key(),
        budget.backend().key(),
        gb(needed.saturating_sub(*available)),
    ))
}

/// What would fit instead: a lighter tier, a shorter song, a shorter plan.
fn alternatives(
    shape: &Yue2Shape,
    budget: &Yue2Budget,
    other_live: u64,
    stage: Yue2Stage,
) -> String {
    let fits = |candidate: &Yue2Shape| {
        weight_residency(
            candidate.tier,
            budget.backend(),
            candidate.precision,
            candidate.ar,
            budget.compute_cap(),
        )
        .is_ok()
            && choose(candidate, budget, other_live).is_ok()
    };
    let mut out = Vec::new();
    let lighter: Vec<&str> = Yue2Tier::ALL
        .into_iter()
        .filter(|&tier| tier > shape.tier)
        .filter(|&tier| {
            fits(&Yue2Shape {
                tier,
                ar: Yue2ArMode::Native,
                ..*shape
            })
        })
        .map(Yue2Tier::key)
        .collect();
    if !lighter.is_empty() {
        out.push(format!(
            "The {} tier fits this request.",
            lighter.join(" or ")
        ));
    }
    if let Yue2Work::Generate {
        planning,
        semantic_max_tokens,
        cfg,
    } = shape.work
    {
        if matches!(
            stage,
            Yue2Stage::Semantic
                | Yue2Stage::AcousticPrefill
                | Yue2Stage::AcousticSolve
                | Yue2Stage::Decode
        ) {
            // The longest song (semantic tokens = latent frames, 25 per second) that fits.
            let with = |tokens| Yue2Shape {
                work: Yue2Work::Generate {
                    planning,
                    semantic_max_tokens: tokens,
                    cfg,
                },
                ..*shape
            };
            if let Some(tokens) = largest_fitting(semantic_max_tokens, |t| fits(&with(t))) {
                out.push(format!(
                    "At this tier a semantic budget of up to {tokens} tokens (~{} s of audio) fits.",
                    tokens / 25
                ));
            }
        }
        if stage == Yue2Stage::Plan {
            if let Yue2Planning::Sample { max_tokens } = planning {
                let with = |tokens| Yue2Shape {
                    work: Yue2Work::Generate {
                        planning: Yue2Planning::Sample { max_tokens: tokens },
                        semantic_max_tokens,
                        cfg,
                    },
                    ..*shape
                };
                if let Some(tokens) = largest_fitting(max_tokens, |t| fits(&with(t))) {
                    out.push(format!(
                        "At this tier an ABC plan budget of up to {tokens} tokens fits."
                    ));
                }
            }
        }
    }
    if out.is_empty() {
        "No smaller tier or shorter request fits this machine.".to_owned()
    } else {
        out.join(" ")
    }
}

/// The largest `t` in `1..current` for which `fits(t)` (monotone), or `None`.
fn largest_fitting(current: u64, fits: impl Fn(u64) -> bool) -> Option<u64> {
    if current <= 1 || !fits(1) {
        return None;
    }
    let (mut lo, mut hi) = (1, current - 1);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}

// ---- The request → shape --------------------------------------------------------------------------

fn download_bytes(row: &Value) -> Option<u64> {
    row.get("estimatedSizeBytes")
        .and_then(Value::as_u64)
        .or_else(|| row.pointer("/footprint/diskSizeBytes")?.as_u64())
        .filter(|&bytes| bytes > 0)
}

/// The selected decoder's file bytes (`componentId` `vae` / `vae_legacy`).
fn decoder_bytes(manifest_entry: &Value, decoder: SongDecoder) -> Result<u64, String> {
    let component = match decoder {
        SongDecoder::Standard => "vae",
        SongDecoder::Legacy => "vae_legacy",
    };
    manifest_entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| row.get("componentId").and_then(Value::as_str) == Some(component))
        })
        .and_then(download_bytes)
        .ok_or_else(|| format!("the catalog entry declares no sized `{component}` decoder"))
}

/// SheetSage2 + MERT-v2's weights (the `requiredFor: ["cover"]` conditional components).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn transcriber_bytes(manifest_entry: &Value) -> Result<u64, String> {
    let rows = manifest_entry
        .get("conditionalComponents")
        .and_then(Value::as_array)
        .ok_or("the catalog entry declares no transcription components")?;
    let mut total = 0;
    for id in ["yue2_sheetsage2", "yue2_mert_v2_fullsong"] {
        total += rows
            .iter()
            .find(|row| row.get("componentId").and_then(Value::as_str) == Some(id))
            .and_then(download_bytes)
            .ok_or_else(|| format!("the catalog entry declares no sized `{id}` component"))?;
    }
    Ok(total)
}

/// An `f32` request value as the `f64` its shortest decimal names (the engine's `decimal_f64`).
fn decimal_f64(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(f64::from(v))
}

/// The ABC tokens of a restored plan: `abc_tokens.npy` holds them as int32, so its size bounds them.
fn saved_plan_abc_tokens(dir: &std::path::Path) -> Result<u64, String> {
    let file = dir.join("abc_tokens.npy");
    std::fs::metadata(&file)
        .map(|meta| (meta.len() / 4).min(CONTEXT))
        .map_err(|error| format!("cannot read the saved plan's {}: {error}", file.display()))
}

/// Latent frames of a completed run's cached latents (`latent.npy`, `[frames, 64]` F32), or the
/// protocol's bound when the file cannot be sized.
fn cached_latent_frames(dir: &std::path::Path) -> u64 {
    std::fs::metadata(dir.join("latent.npy"))
        .map(|meta| (meta.len() / (LATENT_DIM * 4)).clamp(1, CONTEXT))
        .unwrap_or(CONTEXT)
}

/// The shape of the render `request` describes, read the way the engine reads it
/// (`provider::map_request`): `None` fields take the engine's defaults. `Err` names what cannot be
/// priced (the caller fails closed).
pub(crate) fn shape_of(
    manifest_entry: &Value,
    request: &GenerationRequest,
    load: Yue2LoadFacts,
    pins: Option<Yue2Pins>,
    ar: Yue2ArMode,
) -> Result<Yue2Shape, String> {
    // The caller's field-by-field pins when it has them (a job's own memory controls); otherwise
    // what the request's block and the load fix.
    let pins = match pins {
        Some(pins) => pins,
        None => Yue2Pins::of_request(request.memory.as_ref(), load.sequential_offload)?,
    };
    let audio = request.audio.clone().unwrap_or_default();
    let song = audio.song.clone().unwrap_or_default();
    let lyrics = audio.lyrics.as_deref().unwrap_or_default();
    let translated = song
        .cover
        .as_ref()
        .and_then(|cover| cover.translated_from.as_deref())
        .unwrap_or_default();
    let text_tokens =
        PROMPT_FRAMING_TOKENS + (request.prompt.len() + lyrics.len() + translated.len()) as u64;
    let decoder = decoder_bytes(
        manifest_entry,
        song.decoder.unwrap_or(SongDecoder::Standard),
    )?;
    let mode = song.planning.unwrap_or(SongPlanning::Full);
    let planning = if let Some(cover) = &song.cover {
        Yue2Planning::Supplied {
            abc_tokens: (cover.score.len() as u64).min(CONTEXT),
        }
    } else if let Some(plan) = &song.plan {
        Yue2Planning::Supplied {
            abc_tokens: saved_plan_abc_tokens(&plan.dir)?,
        }
    } else if let Some(score) = &song.score {
        Yue2Planning::Supplied {
            abc_tokens: (score.len() as u64).min(CONTEXT),
        }
    } else if mode == SongPlanning::Off {
        Yue2Planning::Off
    } else {
        Yue2Planning::Sample {
            max_tokens: song
                .score_sampling
                .and_then(|s| s.max_tokens)
                .map_or(ABC_MAX_TOKENS_DEFAULT, u64::from),
        }
    };
    // `CotMode::default_guidance`: 1.01 without a plan (`off`), else 1 — and 1 is no guidance. A
    // restored plan carries its own request (and guidance) inside the plan directory, so without an
    // explicit guidance it is priced WITH the second CFG cache rather than guessed without it.
    let off = mode == SongPlanning::Off && song.cover.is_none();
    let guidance = request
        .guidance
        .map(decimal_f64)
        .unwrap_or(if off || song.plan.is_some() {
            1.01
        } else {
            1.0
        });
    let work = if let Some(dir) = &song.cached_latents {
        Yue2Work::DecodeCached {
            frames: cached_latent_frames(dir),
        }
    } else if song.plan_only {
        Yue2Work::PlanOnly { planning }
    } else {
        Yue2Work::Generate {
            planning,
            semantic_max_tokens: song
                .semantic_sampling
                .and_then(|s| s.max_tokens)
                .map_or(SEMANTIC_MAX_TOKENS_DEFAULT, u64::from),
            cfg: guidance != 1.0,
        }
    };
    Ok(Yue2Shape {
        tier: load.tier,
        precision: load.precision,
        ar,
        text_tokens,
        work,
        decoder_bytes: decoder,
        // A GenerationRequest cover starts from a reviewed score; transcribing a recording is a
        // separate step (`transcription_shape`).
        transcription_secs: None,
        transcriber_bytes: 0,
        pins,
    })
}

/// The transcription step of a cover from a recording (SheetSage2 + MERT-v2), priced on its own:
/// the transcriber unloads before the MoT loads, so it is never summed with a render. No job calls
/// it yet: the native transcriber is gated out of every runtime bundle until the owner records a
/// basis for its port code (the catalog's `conditionalComponents[].blocked`), and the job that
/// unblocks it prices its step here.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn transcription_shape(
    manifest_entry: &Value,
    tier: Yue2Tier,
    source_secs: u64,
) -> Result<Yue2Shape, String> {
    Ok(Yue2Shape {
        tier,
        precision: Yue2Precision::Default,
        ar: Yue2ArMode::Native,
        text_tokens: 0,
        work: Yue2Work::PlanOnly {
            planning: Yue2Planning::Off,
        },
        decoder_bytes: decoder_bytes(manifest_entry, SongDecoder::Standard)?,
        transcription_secs: Some(source_secs),
        transcriber_bytes: transcriber_bytes(manifest_entry)?,
        pins: Yue2Pins::default(),
    })
}

// ---- Residency leases -----------------------------------------------------------------------------

/// One live lease's `(device, host)` bytes and the thread that admitted it.
type LiveEntry = (u64, u64, std::thread::ThreadId);

/// Residency the live YuE2 renders of this process hold, keyed by lease id.
fn live_leases() -> &'static Mutex<BTreeMap<u64, LiveEntry>> {
    static LIVE: OnceLock<Mutex<BTreeMap<u64, LiveEntry>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// `(device, host)` bytes every live YuE2 lease holds. Under `cfg(test)` only the leases this
/// thread admitted count, so parallel tests that run YuE2 jobs cannot shrink each other's budgets.
pub(crate) fn live_residency_bytes() -> (u64, u64) {
    #[cfg(test)]
    let current = std::thread::current().id();
    live_leases()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .values()
        .filter(|_entry| {
            #[cfg(test)]
            return _entry.2 == current;
            #[cfg(not(test))]
            true
        })
        .fold((0, 0), |(d, h), &(dd, hh, _)| (d + dd, h + hh))
}

/// The admitted residency of one render, held from admission until the render's generator is
/// dropped — on completion, error, cancellation or unwind alike (drop it AFTER the generator). While
/// held, a later YuE2 admission on a unified pool counts it against the capacity, so a render whose
/// blocking task outlived its job (a cancelled task still winding down, or one the cancel join had to
/// abandon) cannot be double-booked. [`Self::observe`] follows the engine's progress through the
/// stages, so the lease holds each stage's residency as it runs: the KV caches of a finished AR
/// stage are released, and AR offload moves the AR-only weights to the host pool during the
/// acoustic solve.
#[derive(Debug)]
pub(crate) struct Yue2Lease {
    id: u64,
    estimate: Yue2Estimate,
    /// The stages the engine reports progress for, in order.
    order: Vec<Yue2Stage>,
    /// Index into `order` (the load stage before any progress).
    at: Option<usize>,
    last_step: Option<(u32, u32)>,
    /// The thread that admitted the render (its residency is charged to it under `cfg(test)`).
    owner: std::thread::ThreadId,
    /// The step totals the AR stages report (their `max_tokens`): a resumed run reuses finished
    /// stages without reporting them, so a restarted count is matched to its stage by its total.
    plan_total: Option<u64>,
    semantic_total: Option<u64>,
}

impl Yue2Lease {
    fn open(estimate: Yue2Estimate, work: Yue2Work) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let order = estimate
            .stages
            .iter()
            .map(|s| s.stage)
            .filter(|s| {
                matches!(
                    s,
                    Yue2Stage::Plan
                        | Yue2Stage::Semantic
                        | Yue2Stage::AcousticSolve
                        | Yue2Stage::Decode
                )
            })
            .collect();
        let (plan_total, semantic_total) = match work {
            Yue2Work::Generate {
                planning,
                semantic_max_tokens,
                ..
            } => (
                match planning {
                    Yue2Planning::Sample { max_tokens } => Some(max_tokens),
                    _ => None,
                },
                Some(semantic_max_tokens),
            ),
            Yue2Work::PlanOnly {
                planning: Yue2Planning::Sample { max_tokens },
            } => (Some(max_tokens), None),
            _ => (None, None),
        };
        let lease = Self {
            id,
            estimate,
            order,
            at: None,
            last_step: None,
            owner: std::thread::current().id(),
            plan_total,
            semantic_total,
        };
        lease.publish();
        lease
    }

    /// This lease's id in the live-residency table.
    #[cfg(test)]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The stage the render is in.
    pub(crate) fn stage(&self) -> Yue2Stage {
        self.at.map_or(Yue2Stage::Load, |i| self.order[i])
    }

    /// `(device, host)` bytes the current stage holds. The acoustic stage holds the larger of its
    /// prefill and solve phases on each pool, since they alternate chunk by chunk.
    pub(crate) fn held_bytes(&self) -> (u64, u64) {
        let of = |stage| {
            self.estimate
                .stage(stage)
                .map_or((0, 0), |s| (s.device_bytes(), s.host_bytes()))
        };
        match self.stage() {
            Yue2Stage::AcousticSolve | Yue2Stage::AcousticPrefill => {
                let (pd, ph) = of(Yue2Stage::AcousticPrefill);
                let (sd, sh) = of(Yue2Stage::AcousticSolve);
                (pd.max(sd), ph.max(sh))
            }
            stage => of(stage),
        }
    }

    fn publish(&self) {
        let (device, host) = self.held_bytes();
        live_leases()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(self.id, (device, host, self.owner));
    }

    fn enter(&mut self, stage: Yue2Stage) {
        if let Some(i) = self.order.iter().position(|&s| s == stage) {
            if self.at.is_none_or(|at| i > at) {
                self.at = Some(i);
                self.publish();
            }
        }
    }

    /// Follow one engine progress event (`provider`'s contract: each AR token and each midpoint step
    /// is a `Step` counted from 1 against its own stage's total, stage after stage; `Decoding` marks
    /// the decoder's start).
    pub(crate) fn observe(&mut self, progress: &Progress) {
        match *progress {
            Progress::Decoding => self.enter(Yue2Stage::Decode),
            Progress::Step { current, total } => {
                let restarted = self
                    .last_step
                    .is_none_or(|(c, t)| current <= c || total != t);
                self.last_step = Some((current, total));
                if restarted {
                    let next = self.at.map_or(0, |i| i + 1);
                    let total = u64::from(total);
                    // The first remaining stage this total identifies (an AR stage reports its own
                    // `max_tokens`; the acoustic stage's midpoint-step count matches neither), else
                    // simply the next stage.
                    let identified = self.order[next.min(self.order.len())..]
                        .iter()
                        .copied()
                        .find(|&stage| match stage {
                            Yue2Stage::Plan => self.plan_total == Some(total),
                            Yue2Stage::Semantic => self.semantic_total == Some(total),
                            Yue2Stage::AcousticSolve => {
                                self.plan_total != Some(total) && self.semantic_total != Some(total)
                            }
                            _ => false,
                        });
                    if let Some(stage) = identified.or_else(|| self.order.get(next).copied()) {
                        self.enter(stage);
                    }
                }
            }
            Progress::Loading(_) => {}
        }
    }

    /// The priced render this lease holds.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn estimate(&self) -> &Yue2Estimate {
        &self.estimate
    }
}

impl Drop for Yue2Lease {
    fn drop(&mut self) {
        live_leases()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&self.id);
    }
}

// ---- Live budget and the job hook -------------------------------------------------------------------

#[cfg(test)]
thread_local! {
    /// A budget a test installs in place of the hardware probe (`Some(None)`: "no reading").
    static BUDGET_OVERRIDE: std::cell::RefCell<Option<Option<Yue2Budget>>> =
        const { std::cell::RefCell::new(None) };
    /// Hardware budget probes [`check`] made on this thread.
    static BUDGET_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Whether this thread's admissions read the real hardware budget (the profile capture).
    static HARDWARE_PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Evictions [`check`] requested on this thread.
    static EVICTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Restores the hardware probe when dropped ([`override_budget`]).
#[cfg(test)]
pub(crate) struct BudgetOverride;

#[cfg(test)]
impl Drop for BudgetOverride {
    fn drop(&mut self) {
        BUDGET_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Install `budget` as this thread's live budget until the guard drops (tests only).
#[cfg(test)]
pub(crate) fn override_budget(budget: Option<Yue2Budget>) -> BudgetOverride {
    BUDGET_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(budget));
    BudgetOverride
}

#[cfg(test)]
pub(crate) fn budget_probes() -> usize {
    BUDGET_PROBES.with(std::cell::Cell::get)
}

/// Let this thread's admissions read the real hardware budget (the profile capture entrypoint).
#[cfg(test)]
#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn probe_hardware_in_this_test() {
    HARDWARE_PROBE.with(|probe| probe.set(true));
}

/// Evictions [`check`] requested on this thread (tests only).
#[cfg(test)]
pub(crate) fn evictions() -> usize {
    EVICTIONS.with(std::cell::Cell::get)
}

/// Whether lease `id` still holds residency (tests only).
#[cfg(test)]
pub(crate) fn lease_is_live(id: u64) -> bool {
    live_leases()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .contains_key(&id)
}

/// Read this host's budget for the YuE2 render.
///
/// Under `cfg(test)` the hardware is never read unless a test asks: an installed override wins, a
/// thread that opted in ([`probe_hardware_in_this_test`], the profile capture) reads the real
/// budget, and every other test reads "no budget" — so a job test never depends on the machine.
async fn live_budget(gpu_id: &str) -> Option<Yue2Budget> {
    #[cfg(test)]
    {
        BUDGET_PROBES.with(|probes| probes.set(probes.get() + 1));
        if let Some(budget) = BUDGET_OVERRIDE.with(|slot| slot.borrow().clone()) {
            return budget;
        }
        if !HARDWARE_PROBE.with(std::cell::Cell::get) {
            return None;
        }
    }
    probe_budget(gpu_id).await
}

#[cfg(target_os = "macos")]
async fn probe_budget(_gpu_id: &str) -> Option<Yue2Budget> {
    let working_set = working_set_ceiling_bytes().or_else(|| {
        // No working-set reading: the unified total less the legacy reserve.
        let total = crate::mlx_fit_gate::probe_total_unified_memory_bytes()?;
        let reserve = crate::fit_gate::legacy_unified_reserve(gb(total)).gb * BYTES_PER_GIB;
        Some(total.saturating_sub(reserve as u64))
    })?;
    // The small-Mac emulation cap caps the reading, as for every other unified-memory gate.
    let capacity_bytes = crate::mlx_fit_gate::mlx_memory_cap_gb().map_or(working_set, |cap| {
        working_set.min((cap * BYTES_PER_GIB) as u64)
    });
    // The rest of this process shares the working set: the MLX allocator's live and cached bytes —
    // the image generator the cache keeps warm for 300 s after an image job, the refine model.
    // Evicting the cached generator frees its own load's bytes; clearing the MLX cache frees the
    // cached ones. Anything else MLX holds stays charged.
    let active = mlx_rs::memory::get_active_memory() as u64;
    let cached = mlx_rs::memory::get_cache_memory() as u64;
    let generator = crate::generator_cache::cached_generator_resident_bytes()
        .await
        .unwrap_or(None)
        .unwrap_or(0);
    Some(unified_budget(capacity_bytes, active, cached, generator))
}

/// A Metal budget from the working set and what the MLX allocator holds (`active`, `cached`), of
/// which `cached_generator` bytes belong to the generator the cache keeps resident.
#[cfg_attr(not(any(test, target_os = "macos")), allow(dead_code))]
pub(crate) fn unified_budget(
    capacity_bytes: u64,
    active: u64,
    cached: u64,
    cached_generator: u64,
) -> Yue2Budget {
    let resident_bytes = active + cached;
    Yue2Budget::Unified {
        backend: Yue2Backend::Metal,
        capacity_bytes,
        resident_bytes,
        reclaimable_bytes: (cached + cached_generator.min(active)).min(resident_bytes),
    }
}

/// The GPU's recommended working set. Read in test builds too (the profile capture admits against
/// the real ceiling); reading it runs no workload.
#[cfg(target_os = "macos")]
fn working_set_ceiling_bytes() -> Option<u64> {
    Some(crate::generator_cache::device_wired_ceiling_bytes() as u64).filter(|&bytes| bytes > 0)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn probe_budget(gpu_id: &str) -> Option<Yue2Budget> {
    let budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    )?;
    let bytes = |gib: f64| (gib.max(0.0) * BYTES_PER_GIB) as u64;
    Some(Yue2Budget::Dedicated {
        free_bytes: bytes(budget.free_gb),
        total_bytes: bytes(budget.total_gb),
        reclaimable_bytes: bytes(crate::vram_gate::reclaimable_pool_gb(gpu_id)),
        host_available_bytes: host_available_bytes().await,
        gpu_id: gpu_id.to_owned(),
        compute_cap: crate::gpu::cached_compute_cap(),
    })
}

#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
async fn probe_budget(_gpu_id: &str) -> Option<Yue2Budget> {
    None
}

/// Host RAM available right now: `MemAvailable` on Linux, `FreePhysicalMemory` on Windows. `None`
/// when it cannot be read (the host pool is then not checked).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn host_available_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = tokio::fs::read_to_string("/proc/meminfo").await.ok()?;
        parse_meminfo_available(&meminfo)
    }
    #[cfg(windows)]
    {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio::process::Command::new("powershell.exe")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
                ])
                .kill_on_drop(true)
                .output(),
        )
        .await
        .ok()?
        .ok()?;
        output
            .status
            .success()
            .then(|| {
                String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<u64>()
                    .ok()
            })
            .flatten()
            .map(|kib| kib * 1024)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// `MemAvailable` (kB) of a `/proc/meminfo` body, in bytes.
#[cfg_attr(
    not(all(target_os = "linux", feature = "backend-candle")),
    allow(dead_code)
)]
fn parse_meminfo_available(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        let kib: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kib * 1024)
    })
}

/// A render the gate admitted: the controls to send and the lease that holds its residency.
#[derive(Debug)]
pub(crate) struct Yue2Admitted {
    /// The request's `memory` block to send: the controls the request fixed, exactly as it set
    /// them, and the free ones as the gate chose them.
    pub memory: GenerationMemory,
    /// Move into the task that owns the generator and drop it after the generator.
    pub lease: Yue2Lease,
}

/// Evict the cached generator so a render that fits only without it can load: on Metal it frees
/// the MLX generator's share of the unified working set (and clears the MLX cache), on CUDA its pool.
async fn evict_for_admission(gpu_id: &str) -> Result<(), WorkerError> {
    #[cfg(test)]
    EVICTIONS.with(|evictions| evictions.set(evictions.get() + 1));
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        let evicted = crate::generator_cache::evict_cached_generator().await?;
        tracing::info!(
            gpu_id,
            evicted,
            "YuE2 admission: evicted the resident generator to reclaim its memory (sc-23001)"
        );
    }
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let _ = gpu_id;
    Ok(())
}

/// The pre-load gate a YuE2 job runs: price `request` as `load` will run it, choose the memory
/// controls the request leaves free against the live budget, and refuse before anything loads when
/// it cannot fit. A render that fits only once the cached generator is evicted evicts it first (on
/// Metal and on CUDA). Call it with the exact request the job will send, the exact load it will
/// make, and — when the job knows which memory controls its user set — those as `pins`.
pub(crate) async fn check(
    model: &str,
    manifest_entry: &Value,
    request: &GenerationRequest,
    load: Yue2LoadFacts,
    pins: Option<Yue2Pins>,
    gpu_id: &str,
) -> Result<Yue2Admitted, WorkerError> {
    // A request that cannot be priced fails closed BEFORE the hardware is probed.
    let shape =
        shape_of(manifest_entry, request, load, pins, Yue2ArMode::Native).map_err(|why| {
            WorkerError::InvalidPayload(format!(
                "{model}: YuE2 memory admission cannot price this render ({why}); the installed \
             catalog entry or saved plan is incomplete."
            ))
        })?;
    let budget = live_budget(gpu_id).await;
    let (other_device, other_host) = live_residency_bytes();
    let estimate = match decide(model, &shape, budget.as_ref(), other_device + other_host) {
        Yue2Admission::Admit(estimate) => estimate,
        Yue2Admission::AdmitAfterEvict(estimate) => {
            evict_for_admission(gpu_id).await?;
            estimate
        }
        Yue2Admission::Refuse(error) => return Err(error),
    };
    let (stage, peak) = estimate.unified_floor();
    tracing::info!(
        model,
        tier = load.tier.key(),
        backend = estimate.backend.key(),
        evidence = Yue2Evidence::of(estimate.backend).key(),
        binding_stage = stage.key(),
        peak_gb = gb(peak),
        offload_ar = estimate.controls.offload_ar,
        attention_elements = estimate.controls.attention_elements,
        decode_core_frames = estimate.controls.decode_core_frames,
        "YuE2 admission: render admitted (sc-23001)"
    );
    Ok(Yue2Admitted {
        memory: shape.pins.memory_block(&estimate.controls),
        lease: Yue2Lease::open(estimate, shape.work),
    })
}

#[cfg(test)]
pub(crate) mod tests;
