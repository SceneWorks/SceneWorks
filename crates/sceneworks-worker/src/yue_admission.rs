//! YuE lyrics2song memory admission (sc-19386, epic 19373).
//!
//! YuE is the first tiered, staged, autoregressive model on the candle audio lane, which until now
//! had no worker-side memory gate at all (its other models carry only the web's advisory
//! `candle.minMemoryGb` blanket). This module prices one render before anything loads and refuses
//! a request that cannot fit with the reason stated. It is deliberately NOT routed through the
//! shared image memory ladder (`memory_strategy` / `candle_memory_strategy`), which is image-lane
//! only by construction.
//!
//! ## The estimate: max over stages, plus what the request asks for
//!
//! `candle-audio-yue` loads one stage at a time and releases it before the next loads (its engine
//! drops stage 1 before stage 2 loads, and both LMs before the codec and vocoders), so the floor is
//! the LARGEST single stage residency, never the sum:
//!
//! * **stage 1** — the 7B Llama at the selected tier, its KV cache and its per-layer attention
//!   workspace ([`stage1_attention_workspace_bytes`]). The cache holds the whole segment history
//!   (prompt blocks + every generated codebook-0 token and each segment's closing `<EOA>`),
//!   batch-of-2 under CFG, sized once to the render bound and capped at the checkpoint's 16 384
//!   positions. So it scales with `n_segments × (max_new_tokens_per_segment + 1)` until that cap —
//!   `n_segments` being what the engine actually renders, `min(requested, lyric sections)`
//!   ([`lyric_section_count`]).
//! * **stage 2** — the 1B Llama at the SAME tier (the manifest's per-tier `stage2` coRequisite)
//!   plus a preallocated static KV cache for a batch of up to four 300-frame chunks.
//! * **codec** — xcodec, both Vocos decoders and the HuBERT branch (the ICL encoder), stored and run
//!   in f32 at every tier (the approved R2 carve-out). The whole `xcodec` component is an upper
//!   bound for each of the codec, vocoder and ICL-encoder stages, which load separately.
//!
//! Weight bytes come from the manifest's per-tier `estimatedSizeBytes` (the pre-quantized on-disk
//! tier IS the resident set: GGML q4/q8 blocks load as stored, bf16 loads as bf16 on the GPU
//! backends, and the xcodec safetensors are already f32). The architecture constants below are the
//! upstream `config.json` of `m-a-p/YuE-s1-7B-anneal-*` and `m-a-p/YuE-s2-1B-general`, and the
//! schedule constants are `candle-audio-yue`'s. The KV element is bf16: candle-llm's compute dtype
//! on CUDA and Metal, the only lanes this gate has a budget for.
//!
//! ## How this relates to the engine's own admission
//!
//! candle-llm admits each LM *load* (`LlamaProvider::load`) against LIVE available memory — the
//! weights only; YuE's KV caches and attention workspace are allocated outside `generate`, so the
//! engine never prices them. This gate prices the whole render and runs first:
//!
//! * **Apple silicon** — capacity: the GPU's recommended working set
//!   (`recommendedMaxWorkingSetSize`), not `hw.memsize`. A render that cannot fit it is refused
//!   here, once; one that fits but not the memory free at this moment is admitted here and refused
//!   by the engine's live load check — one refusal either way.
//! * **CUDA** — live free VRAM plus the dedicated-VRAM reserve, with the same evict-then-reclaim
//!   every candle image lane runs (`image_jobs::base::gate_with_evict_reclaim`): when the raw
//!   reading refuses but crediting the cached generator's pool (`vram_gate::reclaimable_pool_gb`,
//!   clamped to the card total like `vram_gate::with_reclaimable`) admits, the cached generator is
//!   evicted and the render admitted. A refusal says which case it is — the card is too small
//!   (floor + reserve > total), or VRAM is held by another process or model right now. Because this
//!   gate prices a superset of the engine's figure against the same free reading, the engine's
//!   weights-only load check never refuses what this gate admitted for capacity.
//!
//! No live budget (no NVIDIA reading, a CPU host) admits: the gate never blocks without evidence,
//! the same contract as [`crate::fit_gate::FitDecision::Unknown`], and the engine's load admission
//! still stands behind it.

use serde_json::Value;

use crate::fit_gate::BYTES_PER_GIB;
use crate::WorkerError;

/// The manifest `family` every YuE entry declares.
pub(crate) const YUE_FAMILY: &str = "yue";

/// Stage-1 `config.json`: `num_hidden_layers`.
const STAGE1_LAYERS: u64 = 32;
/// Stage-1 `config.json`: `num_attention_heads` (query heads).
const STAGE1_HEADS: u64 = 32;
/// Stage-1 `config.json`: `num_key_value_heads` (GQA).
const STAGE1_KV_HEADS: u64 = 4;
/// Stage-1 head dim: `hidden_size / num_attention_heads` = 4096 / 32.
const STAGE1_HEAD_DIM: u64 = 128;
/// Stage-1 `config.json`: `max_position_embeddings` — the smart context never lets the cache
/// outgrow it.
const STAGE1_CONTEXT: u64 = 16_384;
/// candle-llm's eager-attention query tile (`primitives/attention.rs:60`
/// `EAGER_ATTN_QUERY_CHUNK_SIZE`).
const ATTN_QUERY_CHUNK: u64 = 256;
/// candle-audio-yue's stage-1 prefill chunk (`stage1/lm.rs:59` `PREFILL_CHUNK`) — the query width
/// of the CFG additive mask.
const STAGE1_PREFILL_CHUNK: u64 = 512;

/// Stage-2 `config.json`: `num_hidden_layers`.
const STAGE2_LAYERS: u64 = 32;
/// Stage-2 `config.json`: `num_key_value_heads` (no GQA: 16 of 16).
const STAGE2_KV_HEADS: u64 = 16;
/// Stage-2 head dim: 2048 / 16.
const STAGE2_HEAD_DIM: u64 = 128;
/// `candle_audio_yue::stage2::CHUNK_FRAMES` — 6 s at 50 frames/s.
const STAGE2_CHUNK_FRAMES: u64 = 300;
/// `candle_audio_yue::stage2::DEFAULT_BATCH_SIZE` — full chunks decoded together.
const STAGE2_BATCH: u64 = 4;
/// Codebooks per frame (`tokens::NUM_CODEBOOKS`).
const NUM_CODEBOOKS: u64 = 8;

/// bf16 KV element (candle-llm's GPU compute dtype).
const KV_ELEMENT_BYTES: u64 = 2;

/// `candle_audio_yue::config::DEFAULT_SEGMENTS`.
const DEFAULT_SEGMENTS: u32 = 2;
/// `DecodeConfig::default().max_new_tokens`.
const DEFAULT_MAX_NEW_TOKENS: u32 = 3_000;
/// Fixed prompt framing (instruction line, `[Genre]` header, SOA/EOA) — an allowance on top of the
/// text bytes.
const PROMPT_HEAD_TOKENS: u64 = 64;
/// Per-segment framing tokens (`[start_of_segment]`, `[end_of_segment]`, SOA/stage markers).
const PROMPT_SEGMENT_TOKENS: u64 = 32;
/// The default ICL window end (upstream `prompt_end_time`, 30 s, whatever the start).
const ICL_DEFAULT_END_SECS: f64 = 30.0;
/// Codebook-0 frames per second per reference track (`tokens::FRAMES_PER_SECOND`); a `dual`
/// reference interleaves two tracks (vocal + instrumental) per frame, a `single` mix one.
const ICL_FRAMES_PER_SEC: f64 = 50.0;

/// The LM tier both stage 1 and stage 2 load at (epic R2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum YueTier {
    Bf16,
    Q8,
    Q4,
}

impl YueTier {
    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    /// The tier a resolved `AudioTier::name` names (`bf16` / `q8` / `q4`).
    pub(crate) fn from_key(key: &str) -> Option<Self> {
        match key {
            "bf16" => Some(Self::Bf16),
            "q8" => Some(Self::Q8),
            "q4" => Some(Self::Q4),
            _ => None,
        }
    }
}

/// Lyric sections the engine renders at most one segment each: the count of
/// `candle_audio_yue::tokenizer::split_lyrics` matches — the reference `split_lyrics`
/// `re.findall(r"\[(\w+)\](.*?)(?=\[|\Z)", lyrics, re.DOTALL)`, ported there (and here) as
/// `\[([\p{L}\p{N}_]+)\]([^\[]*)` because Python's Unicode `\w` is exactly `[\p{L}\p{N}_]` and a lazy
/// `.*?` up to the next `[` is `[^\[]*`. The engine renders `min(segments, sections)`
/// (`engine.rs`: `prompt.segments.len().min(req.segments)`).
pub(crate) fn lyric_section_count(lyrics: &str) -> usize {
    static SECTION: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    SECTION
        .get_or_init(|| {
            regex::Regex::new(r"\[([\p{L}\p{N}_]+)\]([^\[]*)").expect("section pattern compiles")
        })
        .find_iter(lyrics)
        .count()
}

/// What the audio job resolved for a YuE render — the sc-19384 `AudioRequest` fields and its own
/// tier resolution (`resolve_audio_tier`), passed in rather than re-parsed so the gate prices
/// exactly the render the job will run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct YueRequestFacts<'a> {
    /// The tier `resolve_audio_tier` picked (`None` ⇒ the entry declares no tiers: fail closed).
    pub tier: Option<YueTier>,
    pub segments: Option<u32>,
    pub max_new_tokens: Option<u32>,
    /// The top-level guidance the job sends (`AudioRequest::effective_guidance`): `None` ⇒ the
    /// 1.5 / 1.2 CFG schedule, `<= 1` ⇒ CFG off, `> 1` ⇒ on (the engine's `map_request`).
    pub guidance: Option<f32>,
    pub icl_mode: Option<&'a str>,
    pub icl_start_secs: Option<f32>,
    pub icl_end_secs: Option<f32>,
    /// Genre tags.
    pub prompt: &'a str,
    pub lyrics: &'a str,
}

/// The ICL reference block the stage-1 prompt head carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct IclPricing {
    /// 1 for a `single` mix, 2 for a `dual` vocal + instrumental pair (interleaved per frame).
    pub tracks: u64,
    pub start_secs: f64,
    pub end_secs: f64,
}

impl IclPricing {
    fn window_secs(&self) -> f64 {
        (self.end_secs - self.start_secs).max(0.0)
    }

    fn tokens(&self) -> u64 {
        (self.window_secs() * ICL_FRAMES_PER_SEC).ceil() as u64 * self.tracks
    }
}

/// The request facts the estimate depends on.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct YueRenderShape {
    pub tier: YueTier,
    /// Lyric segments the engine will render: `min(requested, lyric sections)`, at least 1.
    pub segments: u32,
    /// Stage-1 token budget per segment.
    pub max_new_tokens: u32,
    /// Classifier-free guidance on ⇒ stage 1 decodes batch-of-2 under an additive mask.
    pub cfg: bool,
    /// Upper bound on the stage-1 prompt tokens (every sentencepiece token covers at least one byte,
    /// so a byte count bounds the token count), including the ICL block.
    pub prompt_tokens: u64,
    /// The ICL reference block, when the job carries an `iclMode`.
    pub icl: Option<IclPricing>,
}

impl YueRenderShape {
    /// Price the render the job resolved. `None` when the tier is unknown (fail closed upstream).
    /// The ICL window is `(iclStartSecs or 0) .. (iclEndSecs or 30)` — an absent end is upstream's
    /// default `prompt_end_time` of 30 s whatever the start (the API refuses start >= end).
    pub(crate) fn new(facts: &YueRequestFacts<'_>) -> Option<Self> {
        let tier = facts.tier?;
        let requested = facts.segments.unwrap_or(DEFAULT_SEGMENTS);
        let sections = u32::try_from(lyric_section_count(facts.lyrics)).unwrap_or(u32::MAX);
        let segments = requested.min(sections).max(1);
        let max_new_tokens = facts.max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        let cfg = facts.guidance.is_none_or(|g| g > 1.0);
        let icl = facts
            .icl_mode
            .map(|mode| mode.trim().to_lowercase())
            .filter(|mode| !mode.is_empty())
            .map(|mode| IclPricing {
                // `single` is one mix track; `dual` (and anything the job refuses anyway) two.
                tracks: if mode == "single" { 1 } else { 2 },
                start_secs: f64::from(facts.icl_start_secs.unwrap_or(0.0)).max(0.0),
                end_secs: facts.icl_end_secs.map_or(ICL_DEFAULT_END_SECS, f64::from),
            });
        // The head carries the genre tags, the whole lyric sheet and the ICL block; every segment
        // block repeats its own section's lyrics — so the lyrics count twice.
        let prompt_tokens = PROMPT_HEAD_TOKENS
            + facts.prompt.len() as u64
            + 2 * facts.lyrics.len() as u64
            + PROMPT_SEGMENT_TOKENS * u64::from(segments)
            + icl.map_or(0, |icl| icl.tokens());
        Some(Self {
            tier,
            segments,
            max_new_tokens,
            cfg,
            prompt_tokens,
            icl,
        })
    }
}

/// Which stage binds the floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum YueStage {
    Stage1,
    Stage2,
    Codec,
}

impl YueStage {
    fn label(self) -> &'static str {
        match self {
            Self::Stage1 => "the 7B stage-1 LM",
            Self::Stage2 => "the 1B stage-2 LM",
            Self::Codec => "the xcodec/Vocos decode",
        }
    }
}

/// The priced render: each stage's residency and the floor (their max).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct YueEstimate {
    pub tier: YueTier,
    pub stage1_weights_bytes: u64,
    pub stage1_kv_bytes: u64,
    pub stage1_attention_bytes: u64,
    pub stage2_weights_bytes: u64,
    pub stage2_kv_bytes: u64,
    pub codec_bytes: u64,
}

impl YueEstimate {
    pub(crate) fn stage_bytes(&self, stage: YueStage) -> u64 {
        match stage {
            YueStage::Stage1 => {
                self.stage1_weights_bytes + self.stage1_kv_bytes + self.stage1_attention_bytes
            }
            YueStage::Stage2 => self.stage2_weights_bytes + self.stage2_kv_bytes,
            YueStage::Codec => self.codec_bytes,
        }
    }

    /// The binding stage and its residency — the floor. Stages load one at a time, so this is the
    /// max, not the sum.
    pub(crate) fn floor(&self) -> (YueStage, u64) {
        [YueStage::Stage1, YueStage::Stage2, YueStage::Codec]
            .into_iter()
            .map(|stage| (stage, self.stage_bytes(stage)))
            .max_by_key(|&(_, bytes)| bytes)
            .expect("three stages")
    }
}

/// Stage-1 KV positions — the engine's render bound, which is what it sizes its static KV cache to
/// (inference @ d5b18019b, `candle-audio-yue/src/stage1.rs:53-62` `render_positions`, called by
/// `engine.rs:300`): Σ(segment prompt blocks) + segments × (`max_new_tokens` + 1), the `+ 1` being
/// each segment's closing `<EOA>` (sampled or forced), capped at the checkpoint context
/// (`stage1/lm.rs:402`). `prompt_tokens` is an upper bound on Σ blocks (it also counts the head,
/// which segment 0's block carries).
pub(crate) fn stage1_kv_positions(shape: &YueRenderShape) -> u64 {
    let generated = u64::from(shape.segments) * (u64::from(shape.max_new_tokens) + 1);
    shape
        .prompt_tokens
        .saturating_add(generated)
        .min(STAGE1_CONTEXT)
}

fn stage1_batch(shape: &YueRenderShape) -> u64 {
    if shape.cfg {
        2
    } else {
        1
    }
}

fn stage1_kv_bytes(shape: &YueRenderShape) -> u64 {
    2 * STAGE1_LAYERS
        * STAGE1_KV_HEADS
        * STAGE1_HEAD_DIM
        * KV_ELEMENT_BYTES
        * stage1_batch(shape)
        * stage1_kv_positions(shape)
}

/// The per-layer attention workspace stage 1 holds on top of its KV cache (one layer at a time —
/// each layer's temporaries drop before the next runs), at the full `P` = [`stage1_kv_positions`]
/// keys (inference @ d5b18019b):
///
/// * **scores** — the eager attention tiles `STAGE1_HEADS` × 256 query rows × `P` keys per batch
///   row (`candle-llm/src/primitives/attention.rs:60` `EAGER_ATTN_QUERY_CHUNK_SIZE`, tiled by
///   `sdpa_gqa` at `:576`), and three tile-sized tensors are live at once: the scaled scores, the
///   masked scores and the softmax weights (`attention.rs:590`, `:621`, `:624`).
/// * Both paths attend the static cache's K/V **un-expanded** through `sdpa_gqa`: no `repeat_kv`
///   expansion and no transposed-key copy (`attention.rs:579` — a static cache's `Kᵀ` view is read
///   in place). CFG off decodes batch-1 under `AttnMask::Causal` (`sdpa_gqa_causal`,
///   `attention.rs:476`). CFG (batch-of-2 under an additive mask) runs `forward_cfg` →
///   `CausalLm::decode_logits_masked_gqa` (`candle-audio-yue/src/stage1/lm.rs:345-366`,
///   `candle-llm/src/models/llama.rs:1317`), whose layers take the `sdpa_gqa` arm
///   (`llama.rs:2327-2331`), never the `repeat_kv` fallback (`:2333-2335`).
/// * **CFG masks** — a prefill chunk's `[batch, 1, 512, P]` additive mask, built in f32 and cast to
///   bf16 (`lm.rs:370-392`, 512 = `PREFILL_CHUNK`, `lm.rs:59`), plus the segment's resident
///   `[batch, 1, 1, P]` bf16 step mask, built once per segment before its prefill
///   (`lm.rs:246-259`, `:492`) and only narrowed per decode step (`lm.rs:294-299`).
pub(crate) fn stage1_attention_workspace_bytes(shape: &YueRenderShape) -> u64 {
    let batch = stage1_batch(shape);
    let positions = stage1_kv_positions(shape);
    let scores =
        3 * batch * STAGE1_HEADS * ATTN_QUERY_CHUNK.min(positions) * positions * KV_ELEMENT_BYTES;
    if !shape.cfg {
        return scores;
    }
    let prefill_mask =
        batch * STAGE1_PREFILL_CHUNK.min(positions) * positions * (4 + KV_ELEMENT_BYTES);
    let step_mask = batch * positions * KV_ELEMENT_BYTES;
    scores + prefill_mask + step_mask
}

/// Stage-2 static KV cache for the largest chunk group. Stage 1 interleaves vocal and instrumental
/// codebook-0 tokens, so each track carries half the generated tokens as frames.
fn stage2_kv_bytes(shape: &YueRenderShape) -> u64 {
    let frames = (u64::from(shape.segments) * u64::from(shape.max_new_tokens) / 2).max(1);
    let rows = frames.div_ceil(STAGE2_CHUNK_FRAMES).clamp(1, STAGE2_BATCH);
    let chunk = frames.min(STAGE2_CHUNK_FRAMES);
    // `stage2::chunk_capacity`: the prefix, then eight tokens per frame, less the last residual.
    let capacity = chunk + 3 + NUM_CODEBOOKS * chunk - 1;
    2 * STAGE2_LAYERS * STAGE2_KV_HEADS * STAGE2_HEAD_DIM * KV_ELEMENT_BYTES * rows * capacity
}

fn download_bytes(download: &Value) -> Option<u64> {
    download
        .get("estimatedSizeBytes")
        .and_then(Value::as_u64)
        .or_else(|| download.pointer("/footprint/diskSizeBytes")?.as_u64())
        .filter(|&bytes| bytes > 0)
}

/// Price a render from the manifest entry's per-tier downloads. `Err` names the missing catalog
/// fact (a YuE entry that does not price is a catalog defect, so the caller fails closed).
pub(crate) fn estimate(
    manifest_entry: &Value,
    shape: &YueRenderShape,
) -> Result<YueEstimate, String> {
    let downloads = manifest_entry
        .get("downloads")
        .and_then(Value::as_array)
        .ok_or("the manifest entry declares no downloads")?;
    let tier = shape.tier.key();
    let find = |component: Option<&str>, variant: Option<&str>| {
        downloads.iter().find(|d| {
            let co = d.get("coRequisite").and_then(Value::as_bool) == Some(true);
            let id = d.get("componentId").and_then(Value::as_str);
            co == component.is_some()
                && (component.is_none() || id == component)
                && (variant.is_none() || d.get("variant").and_then(Value::as_str) == variant)
        })
    };
    let stage1 = find(None, Some(tier))
        .and_then(download_bytes)
        .ok_or_else(|| format!("no sized {tier} stage-1 download"))?;
    let stage2 = find(Some("stage2"), Some(tier))
        .and_then(download_bytes)
        .ok_or_else(|| format!("no sized {tier} stage-2 coRequisite"))?;
    let codec = find(Some("xcodec"), None)
        .and_then(download_bytes)
        .ok_or("no sized xcodec coRequisite")?;
    Ok(YueEstimate {
        tier: shape.tier,
        stage1_weights_bytes: stage1,
        stage1_kv_bytes: stage1_kv_bytes(shape),
        stage1_attention_bytes: stage1_attention_workspace_bytes(shape),
        stage2_weights_bytes: stage2,
        stage2_kv_bytes: stage2_kv_bytes(shape),
        codec_bytes: codec,
    })
}

/// The pool a budget reading describes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum YueBudget {
    /// Apple silicon: the GPU's recommended working set (bytes the process may keep resident).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    UnifiedWorkingSet { gb: f64 },
    /// CUDA: live free VRAM on the selected card, its total, and the in-process pool evicting the
    /// cached generator would return (`vram_gate::reclaimable_pool_gb`). The dedicated-VRAM
    /// allocator/context reserve is charged on top of the estimate.
    #[cfg_attr(
        any(target_os = "macos", not(feature = "backend-candle")),
        allow(dead_code)
    )]
    DedicatedVram {
        free_gb: f64,
        total_gb: f64,
        reclaimable_gb: f64,
        gpu_id: String,
    },
}

/// The admission decision.
#[derive(Debug)]
pub(crate) enum YueAdmission {
    Admit,
    /// Fits only once the resident cached generator is evicted and its pool reclaimed — the same
    /// evict-then-reclaim every candle image lane runs (`image_jobs::base::gate_with_evict_reclaim`).
    AdmitAfterEvict,
    Refuse(WorkerError),
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB
}

/// The pure admission decision. No budget admits (no evidence ⇒ no block; the engine's own load
/// admission still stands).
pub(crate) fn decide(
    model: &str,
    estimate: &YueEstimate,
    shape: &YueRenderShape,
    budget: Option<&YueBudget>,
) -> YueAdmission {
    let Some(budget) = budget else {
        return YueAdmission::Admit;
    };
    let (stage, bytes) = estimate.floor();
    let floor_gb = gib(bytes);
    let fits = |available: f64, needed: f64| available + f64::EPSILON >= needed;
    let (needed_gb, shortfall) = match budget {
        YueBudget::UnifiedWorkingSet { gb } => {
            if fits(*gb, floor_gb) {
                return YueAdmission::Admit;
            }
            (
                floor_gb,
                format!("but this Mac's GPU working set is only ~{gb:.1} GB"),
            )
        }
        YueBudget::DedicatedVram {
            free_gb,
            total_gb,
            reclaimable_gb,
            gpu_id,
        } => {
            let needed = floor_gb + crate::fit_gate::dedicated_vram_reserve().gb;
            if fits(*free_gb, needed) {
                return YueAdmission::Admit;
            }
            // `vram_gate::with_reclaimable`: credit the pool the evict returns, clamped to total.
            let reclaimed = (free_gb + reclaimable_gb.max(0.0)).min(*total_gb);
            if *reclaimable_gb > 0.0 && fits(reclaimed, needed) {
                return YueAdmission::AdmitAfterEvict;
            }
            let shortfall = if fits(*total_gb, needed) {
                format!(
                    "and GPU {gpu_id} has ~{total_gb:.1} GB in total — enough — but only \
                     ~{free_gb:.1} GB is free right now: another process or model is holding \
                     VRAM. Free it and retry"
                )
            } else {
                format!("but GPU {gpu_id} has only ~{total_gb:.1} GB of VRAM in total")
            };
            (needed, shortfall)
        }
    };
    let tier = estimate.tier.key();
    let (weights, kv, workspace) = match stage {
        YueStage::Stage1 => (
            estimate.stage1_weights_bytes,
            estimate.stage1_kv_bytes,
            estimate.stage1_attention_bytes,
        ),
        YueStage::Stage2 => (estimate.stage2_weights_bytes, estimate.stage2_kv_bytes, 0),
        YueStage::Codec => (estimate.codec_bytes, 0, 0),
    };
    let lever = match stage {
        YueStage::Stage1 if estimate.tier != YueTier::Q4 => {
            "Select a smaller tier (q4 is the lightest), or render fewer segments / fewer tokens \
             per segment to shrink the stage-1 KV cache."
        }
        YueStage::Stage1 => {
            "Render fewer segments or fewer tokens per segment to shrink the stage-1 KV cache, or \
             turn guidance off (it doubles the cache)."
        }
        YueStage::Stage2 if estimate.tier != YueTier::Q4 => {
            "Select a smaller tier (q4 is the lightest)."
        }
        _ => "Run on a machine with more memory.",
    };
    let icl = shape.icl.map_or(String::new(), |icl| {
        format!(
            ", plus a {tracks}-track ICL reference window of {secs:.1} s ({start:.1}–{end:.1} s)",
            tracks = icl.tracks,
            secs = icl.window_secs(),
            start = icl.start_secs,
            end = icl.end_secs,
        )
    });
    YueAdmission::Refuse(WorkerError::InvalidPayload(format!(
        "{model} needs ~{needed_gb:.1} GB {shortfall}. YuE loads one stage at a time and the \
         largest is {stage_label} at the {tier} tier: ~{weights:.1} GB of weights + ~{kv:.1} GB of \
         KV cache + ~{workspace:.1} GB of attention workspace ({segments} segment(s) × {max_new} \
         tokens, guidance {cfg}{icl}). {lever}",
        stage_label = stage.label(),
        weights = gib(weights),
        kv = gib(kv),
        workspace = gib(workspace),
        segments = shape.segments,
        max_new = shape.max_new_tokens,
        cfg = if shape.cfg { "on" } else { "off" },
    )))
}

/// Read this host's budget for the candle audio lane.
#[cfg(target_os = "macos")]
pub(crate) async fn live_budget(_gpu_id: &str) -> Option<YueBudget> {
    let working_set_gb = working_set_ceiling_bytes().map(gib).or_else(|| {
        // No working-set reading: fall back to the unified total less the legacy reserve.
        let total_gb = gib(crate::mlx_fit_gate::probe_total_unified_memory_bytes()?);
        Some(total_gb - crate::fit_gate::legacy_unified_reserve(total_gb).gb)
    })?;
    // The small-Mac emulation cap (`SCENEWORKS_MLX_MEMORY_CAP_GB`) caps the reading, as it does for
    // every other unified-memory gate.
    let gb = crate::mlx_fit_gate::mlx_memory_cap_gb()
        .map_or(working_set_gb, |cap| cap.min(working_set_gb));
    Some(YueBudget::UnifiedWorkingSet { gb })
}

#[cfg(all(target_os = "macos", not(test)))]
fn working_set_ceiling_bytes() -> Option<u64> {
    Some(crate::generator_cache::device_wired_ceiling_bytes() as u64).filter(|&bytes| bytes > 0)
}

#[cfg(all(target_os = "macos", test))]
fn working_set_ceiling_bytes() -> Option<u64> {
    None
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn live_budget(gpu_id: &str) -> Option<YueBudget> {
    let budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    )?;
    Some(YueBudget::DedicatedVram {
        free_gb: budget.free_gb,
        total_gb: budget.total_gb,
        reclaimable_gb: crate::vram_gate::reclaimable_pool_gb(gpu_id),
        gpu_id: gpu_id.to_owned(),
    })
}

#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
pub(crate) async fn live_budget(_gpu_id: &str) -> Option<YueBudget> {
    None
}

/// Whether a manifest entry is a YuE model (the only family this gate prices).
pub(crate) fn is_yue(manifest_entry: &Value) -> bool {
    manifest_entry.get("family").and_then(Value::as_str) == Some(YUE_FAMILY)
}

/// The pre-load gate the audio job runs: `Ok(())` for a non-YuE model or an admitted render. On
/// CUDA a render that fits only once the cached generator's pool is reclaimed evicts it first.
pub(crate) async fn check(
    model: &str,
    manifest_entry: &Value,
    facts: &YueRequestFacts<'_>,
    gpu_id: &str,
) -> Result<(), WorkerError> {
    if !is_yue(manifest_entry) {
        return Ok(());
    }
    let cannot_price = |why: &str| {
        WorkerError::InvalidPayload(format!(
            "{model}: YuE memory admission cannot price this render ({why}); the installed \
             catalog entry is incomplete — update SceneWorks before retrying."
        ))
    };
    let shape = YueRenderShape::new(facts).ok_or_else(|| cannot_price("no tier resolved"))?;
    let estimate = estimate(manifest_entry, &shape).map_err(|why| cannot_price(&why))?;
    let budget = live_budget(gpu_id).await;
    match decide(model, &estimate, &shape, budget.as_ref()) {
        YueAdmission::Admit => Ok(()),
        YueAdmission::AdmitAfterEvict => {
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            {
                let evicted = crate::generator_cache::evict_cached_generator().await?;
                tracing::info!(
                    gpu_id,
                    evicted,
                    "YuE admission: evicted the resident generator to reclaim its cudarc pool \
                     (sc-19386)"
                );
            }
            Ok(())
        }
        YueAdmission::Refuse(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// The SHIPPED `builtin.models.jsonc` entry, so a re-sized or dropped tier download fails here.
    fn builtin(id: &str) -> Value {
        let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("builtin.models.jsonc present");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin.models.jsonc parses");
        manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(id))
            .cloned()
            .unwrap_or_else(|| panic!("builtin entry {id} present"))
    }

    fn shape(tier: YueTier, segments: u32, max_new: u32, cfg: bool) -> YueRenderShape {
        YueRenderShape {
            tier,
            segments,
            max_new_tokens: max_new,
            cfg,
            prompt_tokens: 256,
            icl: None,
        }
    }

    fn mac(gb: f64) -> YueBudget {
        YueBudget::UnifiedWorkingSet { gb }
    }

    fn cuda(free_gb: f64, total_gb: f64, reclaimable_gb: f64) -> YueBudget {
        YueBudget::DedicatedVram {
            free_gb,
            total_gb,
            reclaimable_gb,
            gpu_id: "0".to_owned(),
        }
    }

    fn refusal(outcome: YueAdmission) -> String {
        match outcome {
            YueAdmission::Refuse(WorkerError::InvalidPayload(message)) => message,
            other => panic!("expected a user-facing refusal, got {other:?}"),
        }
    }

    /// Four lyric sections, so the requested segment count is what renders (up to 4).
    const FOUR_SECTIONS: &str = "[verse]\na\n[chorus]\nb\n[verse]\nc\n[outro]\nd";

    fn facts<'a>(lyrics: &'a str) -> YueRequestFacts<'a> {
        YueRequestFacts {
            tier: Some(YueTier::Q4),
            segments: None,
            max_new_tokens: None,
            guidance: None,
            icl_mode: None,
            icl_start_secs: None,
            icl_end_secs: None,
            prompt: "pop",
            lyrics,
        }
    }

    #[test]
    fn every_shipped_yue_entry_prices_every_tier_from_its_manifest() {
        for id in [
            "yue_en_cot",
            "yue_en_icl",
            "yue_zh_cot",
            "yue_zh_icl",
            "yue_jp_kr_cot",
            "yue_jp_kr_icl",
        ] {
            let entry = builtin(id);
            assert_eq!(entry["family"], json!(YUE_FAMILY), "{id}");
            let mut previous = 0;
            for tier in [YueTier::Q4, YueTier::Q8, YueTier::Bf16] {
                let e = estimate(&entry, &shape(tier, 2, 3000, true)).expect(id);
                // Each tier is priced by ITS OWN download (R2): heavier tier ⇒ heavier stages.
                assert!(e.stage1_weights_bytes > previous, "{id} {tier:?}");
                previous = e.stage1_weights_bytes;
                assert!(
                    e.stage2_weights_bytes < e.stage1_weights_bytes,
                    "{id} {tier:?}"
                );
                // The trimmed xcodec coRequisite (the three files the engine opens), shared by all six.
                assert_eq!(e.codec_bytes, 888_703_120, "{id}: xcodec f32 component");
            }
        }
    }

    #[test]
    fn floor_is_the_max_stage_not_the_sum_and_uses_the_tier_sizes() {
        let entry = builtin("yue_en_cot");
        let e = estimate(&entry, &shape(YueTier::Q4, 2, 3000, true)).unwrap();
        // The shipped q4 sizes (manifest `estimatedSizeBytes`).
        assert_eq!(e.stage1_weights_bytes, 4_497_628_709);
        assert_eq!(e.stage2_weights_bytes, 1_604_865_100);
        assert_eq!(e.codec_bytes, 888_703_120);
        let (stage, bytes) = e.floor();
        let stages = [
            e.stage_bytes(YueStage::Stage1),
            e.stage_bytes(YueStage::Stage2),
            e.stage_bytes(YueStage::Codec),
        ];
        assert_eq!(bytes, *stages.iter().max().unwrap());
        assert!(bytes < stages.iter().sum::<u64>());
        assert_eq!(stage, YueStage::Stage1);

        let bf16 = estimate(&entry, &shape(YueTier::Bf16, 2, 3000, true)).unwrap();
        assert_eq!(bf16.stage1_weights_bytes, 12_456_344_476);
        assert_eq!(bf16.stage2_weights_bytes, 3_932_179_917);
    }

    #[test]
    fn stage1_kv_scales_with_segments_times_tokens_under_the_16k_cap() {
        // 2 (K,V) × 32 layers × 4 KV heads × 128 × 2 bytes = 64 KiB per position per batch row.
        let per_position = 2 * 32 * 4 * 128 * 2;
        let one = shape(YueTier::Q4, 1, 1000, false);
        let two = shape(YueTier::Q4, 2, 1000, false);
        // Each segment's budget carries its closing `<EOA>` (the engine's `render_positions`).
        assert_eq!(stage1_kv_bytes(&one), per_position * (256 + 1001));
        assert_eq!(stage1_kv_bytes(&two), per_position * (256 + 2002));
        // CFG doubles it (batch-of-2).
        let cfg = shape(YueTier::Q4, 2, 1000, true);
        assert_eq!(stage1_kv_bytes(&cfg), 2 * stage1_kv_bytes(&two));
        // Past the context the smart context caps the cache at 16 384 positions (~2 GiB at CFG).
        let long = shape(YueTier::Q4, 50, 3000, true);
        assert_eq!(stage1_kv_positions(&long), 16_384);
        assert_eq!(stage1_kv_bytes(&long), 2 * per_position * 16_384);

        // Stage-1 residency = weights + KV + the attention workspace, which is nonzero.
        let entry = builtin("yue_en_cot");
        let e = estimate(&entry, &long).unwrap();
        assert!(e.stage1_attention_bytes > 0);
        assert_eq!(
            e.stage_bytes(YueStage::Stage1),
            e.stage1_weights_bytes + e.stage1_kv_bytes + e.stage1_attention_bytes
        );
    }

    #[test]
    fn stage1_attention_workspace_follows_the_engine_attention_path() {
        let p = 16_384u64;
        // CFG: batch 2 under an additive mask ⇒ decode_logits_masked_gqa → sdpa_gqa, K/V read
        // un-expanded (no repeat_kv, no Kᵀ copy). Three 256-row score tiles, the f32→bf16 prefill
        // mask and the resident bf16 step mask.
        let cfg = shape(YueTier::Q4, 50, 3000, true);
        let scores = 3 * 2 * 32 * 256 * p * 2;
        let prefill_mask = 2 * 512 * p * (4 + 2);
        let step_mask = 2 * p * 2;
        assert_eq!(
            stage1_attention_workspace_bytes(&cfg),
            scores + prefill_mask + step_mask
        );
        // CFG off: batch 1, causal ⇒ sdpa_gqa_causal (Kᵀ is a view): the score tiles only.
        let plain = shape(YueTier::Q4, 50, 3000, false);
        assert_eq!(
            stage1_attention_workspace_bytes(&plain),
            3 * 32 * 256 * p * 2
        );
        // A short history tiles fewer than 256 rows.
        let short = YueRenderShape {
            prompt_tokens: 100,
            ..shape(YueTier::Q4, 1, 50, false)
        };
        assert_eq!(
            stage1_attention_workspace_bytes(&short),
            3 * 32 * 151 * 151 * 2
        );
    }

    #[test]
    fn stage2_kv_is_the_preallocated_chunk_group() {
        // A full group: four 300-frame chunks of 2702 positions each, 16 KV heads.
        let full = shape(YueTier::Q4, 2, 3000, true);
        assert_eq!(stage2_kv_bytes(&full), 2 * 32 * 16 * 128 * 2 * 4 * 2702);
        // A sub-chunk track decodes one ragged chunk alone.
        let tiny = shape(YueTier::Q4, 1, 200, true);
        assert_eq!(
            stage2_kv_bytes(&tiny),
            2 * 32 * 16 * 128 * 2 * (100 + 3 + 800 - 1)
        );
    }

    #[test]
    fn over_budget_is_refused_with_the_reason_and_a_fitting_render_is_admitted() {
        let entry = builtin("yue_en_cot");
        let s = shape(YueTier::Q4, 2, 3000, true);
        let e = estimate(&entry, &s).unwrap();
        let floor_gb = gib(e.floor().1);

        // Fits exactly at the floor on a Mac working set.
        assert!(matches!(
            decide("yue_en_cot", &e, &s, Some(&mac(floor_gb))),
            YueAdmission::Admit
        ));
        // Just below: refused, naming the binding stage, tier, and figures.
        let message = refusal(decide("yue_en_cot", &e, &s, Some(&mac(floor_gb - 0.01))));
        assert!(message.contains("7B stage-1 LM"), "{message}");
        assert!(message.contains("q4 tier"), "{message}");
        assert!(message.contains("2 segment(s) × 3000 tokens"), "{message}");
        assert!(message.contains("GPU working set"), "{message}");

        // CUDA charges the dedicated-VRAM reserve on top of the floor.
        let reserve = crate::fit_gate::dedicated_vram_reserve().gb;
        assert!(matches!(
            decide(
                "yue_en_cot",
                &e,
                &s,
                Some(&cuda(floor_gb + reserve, 96.0, 0.0))
            ),
            YueAdmission::Admit
        ));
        let message = refusal(decide(
            "yue_en_cot",
            &e,
            &s,
            Some(&cuda(floor_gb + reserve - 0.01, 96.0, 0.0)),
        ));
        assert!(message.contains("GPU 0"), "{message}");

        // No budget reading admits (no evidence ⇒ no block).
        assert!(matches!(
            decide("yue_en_cot", &e, &s, None),
            YueAdmission::Admit
        ));
    }

    #[test]
    fn cuda_reclaims_the_cached_generator_before_refusing_and_says_why_it_refuses() {
        let entry = builtin("yue_en_cot");
        let s = shape(YueTier::Q4, 2, 3000, true);
        let e = estimate(&entry, &s).unwrap();
        let floor_gb = gib(e.floor().1);
        let reserve = crate::fit_gate::dedicated_vram_reserve().gb;
        let needed = floor_gb + reserve;

        // Short by 1 GB of free VRAM, but evicting the cached generator returns 5 GB: admit after
        // evicting.
        assert!(matches!(
            decide("yue_en_cot", &e, &s, Some(&cuda(needed - 1.0, 24.0, 5.0))),
            YueAdmission::AdmitAfterEvict
        ));
        // The reclaim credit is clamped to the card total (`with_reclaimable`): a card smaller than
        // the floor never admits on credit.
        let clamped = refusal(decide(
            "yue_en_cot",
            &e,
            &s,
            Some(&cuda(needed - 1.0, needed - 0.5, 5.0)),
        ));
        assert!(clamped.contains("of VRAM in total"), "{clamped}");
        // The card is big enough but something else holds the VRAM: say so, not "too small".
        let busy = refusal(decide(
            "yue_en_cot",
            &e,
            &s,
            Some(&cuda(needed - 1.0, 24.0, 0.0)),
        ));
        assert!(
            busy.contains("another process or model is holding VRAM"),
            "{busy}"
        );
        assert!(!busy.contains("of VRAM in total"), "{busy}");
        // The card itself is too small: capacity wording.
        let small = refusal(decide(
            "yue_en_cot",
            &e,
            &s,
            Some(&cuda(floor_gb - 1.0, needed - 1.0, 0.0)),
        ));
        assert!(small.contains("of VRAM in total"), "{small}");
        assert!(!small.contains("another process"), "{small}");
    }

    #[test]
    fn a_16gb_mac_admits_q4_and_refuses_bf16() {
        let entry = builtin("yue_en_cot");
        let budget = mac(10.5); // ~2/3 of 16 GB: the M-series recommended working set.
        let q4 = shape(YueTier::Q4, 2, 3000, true);
        let bf16 = shape(YueTier::Bf16, 2, 3000, true);
        let e4 = estimate(&entry, &q4).unwrap();
        let e16 = estimate(&entry, &bf16).unwrap();
        assert!(matches!(
            decide("yue_en_cot", &e4, &q4, Some(&budget)),
            YueAdmission::Admit
        ));
        let message = refusal(decide("yue_en_cot", &e16, &bf16, Some(&budget)));
        assert!(message.contains("smaller tier"), "{message}");
    }

    #[test]
    fn the_floor_follows_whichever_stage_is_largest() {
        // Shipped entry, short render: the floor is the largest stage, whichever it is.
        let entry = builtin("yue_en_cot");
        let s = shape(YueTier::Q4, 1, 100, false);
        let e = estimate(&entry, &s).unwrap();
        let (stage, bytes) = e.floor();
        assert_eq!(bytes, e.stage_bytes(stage));
        for other in [YueStage::Stage1, YueStage::Stage2, YueStage::Codec] {
            assert!(e.stage_bytes(other) <= bytes);
        }
        // Synthetic entry where stage 2 is the largest resident: the floor must follow it.
        let synthetic = json!({
            "downloads": [
                {"variant": "q4", "estimatedSizeBytes": GIB},
                {"coRequisite": true, "componentId": "stage2", "variant": "q4", "estimatedSizeBytes": 3 * GIB},
                {"coRequisite": true, "componentId": "xcodec", "estimatedSizeBytes": GIB},
            ]
        });
        let e = estimate(&synthetic, &s).unwrap();
        assert_eq!(e.floor().0, YueStage::Stage2);
        let message = refusal(decide("x", &e, &s, Some(&mac(1.0))));
        assert!(message.contains("1B stage-2 LM"), "{message}");
    }

    #[test]
    fn the_shape_prices_what_the_job_resolved() {
        let defaults = YueRenderShape::new(&facts(FOUR_SECTIONS)).unwrap();
        assert_eq!(defaults.tier, YueTier::Q4);
        assert_eq!((defaults.segments, defaults.max_new_tokens), (2, 3000));
        assert!(defaults.cfg, "unset guidance ⇒ the 1.5/1.2 schedule");
        assert_eq!(defaults.icl, None, "no iclMode ⇒ no reference block");
        // Head allowance + genre bytes + the lyrics twice + per-segment framing.
        assert_eq!(
            defaults.prompt_tokens,
            64 + 3 + 2 * FOUR_SECTIONS.len() as u64 + 32 * 2
        );

        let custom = YueRenderShape::new(&YueRequestFacts {
            tier: Some(YueTier::Q8),
            segments: Some(3),
            max_new_tokens: Some(800),
            guidance: Some(1.0),
            ..facts(FOUR_SECTIONS)
        })
        .unwrap();
        assert_eq!(custom.tier, YueTier::Q8);
        assert_eq!((custom.segments, custom.max_new_tokens), (3, 800));
        assert!(
            !custom.cfg,
            "guidance <= 1 (incl. the 0.0 guidanceEnabled=false sends) ⇒ off"
        );
        let on = YueRenderShape::new(&YueRequestFacts {
            guidance: Some(3.0),
            ..facts(FOUR_SECTIONS)
        })
        .unwrap();
        assert!(on.cfg, "guidance > 1 keeps CFG on");
        // No tier resolved ⇒ nothing to price (the gate fails closed).
        assert!(YueRenderShape::new(&YueRequestFacts {
            tier: None,
            ..facts(FOUR_SECTIONS)
        })
        .is_none());
    }

    #[test]
    fn segments_are_capped_at_the_lyric_sections_the_engine_renders() {
        let two_sections = "[verse]\nhello\n[chorus]\nworld";
        let eight = YueRenderShape::new(&YueRequestFacts {
            segments: Some(8),
            ..facts(two_sections)
        })
        .unwrap();
        let two = YueRenderShape::new(&YueRequestFacts {
            segments: Some(2),
            ..facts(two_sections)
        })
        .unwrap();
        assert_eq!(eight.segments, 2);
        assert_eq!(stage1_kv_positions(&eight), stage1_kv_positions(&two));
        // Never below one (the engine refuses section-less lyrics itself).
        let none = YueRenderShape::new(&YueRequestFacts {
            segments: Some(4),
            ..facts("no sections here")
        })
        .unwrap();
        assert_eq!(none.segments, 1);
    }

    #[test]
    fn lyric_sections_follow_the_reference_split() {
        // `\w+` labels only: `[verse 1]` is not a section (a space is not `\w`) and ends nothing.
        assert_eq!(lyric_section_count("[verse]\na\n[chorus]\nb"), 2);
        assert_eq!(lyric_section_count("[verse 1]\na\n[chorus]\nb"), 1);
        // Unicode letters and numbers are `\w` (Python's `re` semantics): CJK, digits, underscore.
        assert_eq!(lyric_section_count("[副歌]\n啊\n[verse_2]\nb\n[２]\nc"), 3);
        // Empty label, unclosed bracket, and text before the first section are not sections.
        assert_eq!(lyric_section_count("intro text [] [open\n[outro]\nz"), 1);
        assert_eq!(lyric_section_count(""), 0);
    }

    #[test]
    fn icl_is_priced_by_the_actual_window_and_track_count() {
        let base = YueRenderShape::new(&facts(FOUR_SECTIONS))
            .unwrap()
            .prompt_tokens;
        let icl = |mode: &'static str, start: Option<f32>, end: Option<f32>| {
            YueRenderShape::new(&YueRequestFacts {
                icl_mode: Some(mode),
                icl_start_secs: start,
                icl_end_secs: end,
                ..facts(FOUR_SECTIONS)
            })
            .unwrap()
        };
        let window = |shape: &YueRenderShape| {
            let p = shape.icl.expect("ICL priced");
            (p.tracks, p.start_secs, p.end_secs, p.window_secs())
        };

        // Both ends absent: the upstream 0–30 s default; dual = 2 × 50 tokens/s.
        let default = icl("Dual", None, None);
        assert_eq!(window(&default), (2, 0.0, 30.0, 30.0));
        assert_eq!(default.prompt_tokens, base + 3000);
        // Explicit window, single mix: 20 s × 50 = 1000 tokens.
        let explicit = icl("single", Some(5.0), Some(25.0));
        assert_eq!(window(&explicit), (1, 5.0, 25.0, 20.0));
        assert_eq!(explicit.prompt_tokens, base + 1000);
        // Absent end = the upstream default end of 30 s whatever the start: 30 − 10 = 20 s.
        let open_end = icl("dual", Some(10.0), None);
        assert_eq!(window(&open_end), (2, 10.0, 30.0, 20.0));
        assert_eq!(open_end.prompt_tokens, base + 2000);
        // Absent start = 0: a long explicit end is priced in full (0–90 s dual = 9000 tokens).
        let long_window = icl("dual", None, Some(90.0));
        assert_eq!(window(&long_window), (2, 0.0, 90.0, 90.0));
        assert_eq!(long_window.prompt_tokens, base + 9000);

        // A longer window moves the estimate, and the refusal names the window it priced.
        let entry = builtin("yue_en_icl");
        let short = estimate(&entry, &explicit).unwrap();
        let long = estimate(&entry, &long_window).unwrap();
        assert!(long.stage1_kv_bytes > short.stage1_kv_bytes);
        let message = refusal(decide("yue_en_icl", &long, &long_window, Some(&mac(1.0))));
        assert!(
            message.contains("2-track ICL reference window of 90.0 s (0.0–90.0 s)"),
            "{message}"
        );
    }

    #[test]
    fn an_incomplete_catalog_entry_fails_closed_with_a_reason() {
        let entry = json!({"downloads": [{"variant": "q4", "estimatedSizeBytes": GIB}]});
        let why = estimate(&entry, &shape(YueTier::Q4, 2, 3000, true)).unwrap_err();
        assert!(why.contains("stage-2"), "{why}");
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let mut yue = entry.clone();
        yue["family"] = json!("yue");
        let err = rt
            .block_on(check("yue_en_cot", &yue, &facts(FOUR_SECTIONS), "0"))
            .unwrap_err();
        assert!(err.to_string().contains("cannot price"), "{err}");
        let untiered = YueRequestFacts {
            tier: None,
            ..facts(FOUR_SECTIONS)
        };
        let err = rt
            .block_on(check("yue_en_cot", &builtin("yue_en_cot"), &untiered, "0"))
            .unwrap_err();
        assert!(err.to_string().contains("no tier resolved"), "{err}");
        // A non-YuE model is never touched by this gate.
        assert!(!is_yue(&json!({"family": "ace"})));
        assert!(rt
            .block_on(check("ace_step", &json!({"family": "ace"}), &untiered, "0"))
            .is_ok());
    }
}
