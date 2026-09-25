//! YuE lyrics2song memory admission (sc-19386, epic 19373).
//!
//! YuE is the first tiered, staged, autoregressive model on the candle audio lane, which until now
//! had no worker-side memory gate at all (its other models carry only the web's advisory
//! `candle.minMemoryGb` blanket). This module prices one render before anything loads and refuses
//! a request that cannot fit this machine with the reason stated. It is deliberately NOT routed
//! through the shared image memory ladder (`memory_strategy` / `candle_memory_strategy`), which is
//! image-lane only by construction.
//!
//! ## The estimate: max over stages, plus the KV cache the request asks for
//!
//! `candle-audio-yue` loads one stage at a time and releases it before the next loads (its engine
//! drops stage 1 before stage 2 loads, and both LMs before the codec and vocoders), so the floor is
//! the LARGEST single stage residency, never the sum:
//!
//! * **stage 1** — the 7B Llama at the selected tier plus its KV cache. The cache holds the whole
//!   segment history (prompt blocks + every generated codebook-0 token), batch-of-2 under CFG, and
//!   the engine's smart context caps it at the checkpoint's 16 384 positions. So it scales with
//!   `n_segments × max_new_tokens_per_segment` until that cap.
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
//! weights only; YuE's KV caches are allocated outside `generate`, so the engine never prices them.
//! This gate is the whole-render CAPACITY check and runs first:
//!
//! * a render that cannot fit this machine at all is refused HERE, once, naming the stage, the tier
//!   and the levers — the engine is never reached, so it never adds a second, differently-worded
//!   refusal;
//! * a render that fits the machine but not its memory at this moment (another process holding
//!   memory) is admitted here and refused by the engine's live load check — one refusal again.
//!
//! On CUDA both read the same live free VRAM, and this gate prices a superset of what the engine
//! prices (weights + KV + the dedicated-VRAM reserve), so whenever the engine would refuse a stage
//! load for the weights alone, this gate has already refused. On Apple silicon the capacity is the
//! GPU's recommended working set (`recommendedMaxWorkingSetSize`), not `hw.memsize`.
//!
//! No live budget (no NVIDIA reading, a CPU host) admits: the gate never blocks without evidence,
//! the same contract as [`crate::fit_gate::FitDecision::Unknown`], and the engine's load admission
//! still stands behind it.

use serde_json::Value;

use crate::fit_gate::BYTES_PER_GIB;
use crate::{JsonObject, WorkerError};

/// The manifest `family` every YuE entry declares.
pub(crate) const YUE_FAMILY: &str = "yue";

/// Stage-1 `config.json`: `num_hidden_layers`.
const STAGE1_LAYERS: u64 = 32;
/// Stage-1 `config.json`: `num_key_value_heads` (GQA).
const STAGE1_KV_HEADS: u64 = 4;
/// Stage-1 head dim: `hidden_size / num_attention_heads` = 4096 / 32.
const STAGE1_HEAD_DIM: u64 = 128;
/// Stage-1 `config.json`: `max_position_embeddings` — the smart context never lets the cache
/// outgrow it.
const STAGE1_CONTEXT: u64 = 16_384;

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

    fn from_key(key: &str) -> Option<Self> {
        match key {
            "bf16" => Some(Self::Bf16),
            "q8" => Some(Self::Q8),
            "q4" => Some(Self::Q4),
            _ => None,
        }
    }
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
    /// Lyric segments to render (the engine caps the effective count at the lyric sections, so the
    /// requested count is an upper bound).
    pub segments: u32,
    /// Stage-1 token budget per segment.
    pub max_new_tokens: u32,
    /// Classifier-free guidance on ⇒ stage 1 decodes batch-of-2.
    pub cfg: bool,
    /// Upper bound on the stage-1 prompt tokens (every sentencepiece token covers at least one byte,
    /// so a byte count bounds the token count), including the ICL block.
    pub prompt_tokens: u64,
    /// The ICL reference block, when the job carries an `iclMode`.
    pub icl: Option<IclPricing>,
}

impl YueRenderShape {
    /// Read the render shape from an audio job payload — the sc-19384 job surface:
    /// * `quantTier` (`bf16` | `q8` | `q4`); unset ⇒ the manifest's default tier when `installed`,
    ///   else the first installed tier (the order the job's own tier resolution uses);
    /// * `segments` / `maxNewTokensPerSegment`, defaulting to the reference pipeline's 2 / 3000;
    /// * `guidanceEnabled: false` ⇒ CFG off; otherwise `guidance` unset ⇒ the 1.5 / 1.2 schedule,
    ///   `<= 1` ⇒ off, `> 1` ⇒ on (the engine's `map_request`);
    /// * `iclMode` (`single` | `dual`) with the window `(iclStartSecs or 0) .. (iclEndSecs or 30)` —
    ///   an absent end is the upstream default `prompt_end_time` of 30 s whatever the start (the API
    ///   refuses start >= end).
    pub(crate) fn from_payload(
        payload: &JsonObject,
        manifest_entry: &Value,
        installed: &dyn Fn(&str) -> bool,
    ) -> Self {
        let tier = requested_tier(payload, manifest_entry, installed);
        let segments = payload
            .get("segments")
            .and_then(Value::as_u64)
            .map(|n| n.min(u64::from(u32::MAX)) as u32)
            .unwrap_or(DEFAULT_SEGMENTS);
        let max_new_tokens = payload
            .get("maxNewTokensPerSegment")
            .and_then(Value::as_u64)
            .map(|n| n.min(u64::from(u32::MAX)) as u32)
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        let cfg = payload.get("guidanceEnabled").and_then(Value::as_bool) != Some(false)
            && payload
                .get("guidance")
                .and_then(Value::as_f64)
                .is_none_or(|g| g > 1.0);
        let text_bytes = |key: &str| {
            payload
                .get(key)
                .and_then(Value::as_str)
                .map_or(0, |s| s.len() as u64)
        };
        let icl = icl_pricing(payload);
        // The head carries the genre tags, the whole lyric sheet and the ICL block; every segment
        // block repeats its own section's lyrics — so the lyrics count twice.
        let prompt_tokens = PROMPT_HEAD_TOKENS
            + text_bytes("prompt")
            + 2 * text_bytes("lyrics")
            + PROMPT_SEGMENT_TOKENS * u64::from(segments)
            + icl.map_or(0, |icl| icl.tokens());
        Self {
            tier,
            segments,
            max_new_tokens,
            cfg,
            prompt_tokens,
            icl,
        }
    }
}

fn icl_pricing(payload: &JsonObject) -> Option<IclPricing> {
    let mode = payload
        .get("iclMode")
        .and_then(Value::as_str)
        .map(|mode| mode.trim().to_lowercase())
        .filter(|mode| !mode.is_empty())?;
    // `single` is one mix track; `dual` (and anything the job will refuse anyway) prices two.
    let tracks = if mode == "single" { 1 } else { 2 };
    let secs = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
    };
    Some(IclPricing {
        tracks,
        start_secs: secs("iclStartSecs").unwrap_or(0.0).max(0.0),
        end_secs: secs("iclEndSecs").unwrap_or(ICL_DEFAULT_END_SECS),
    })
}

/// The stage-1 tier rows (`variant`, tier subdir, default) in manifest order — the same rows the
/// job's own tier resolution reads.
fn tier_rows(manifest_entry: &Value) -> Vec<(YueTier, String, bool)> {
    manifest_entry
        .get("downloads")
        .and_then(Value::as_array)
        .map(|downloads| {
            downloads
                .iter()
                .filter(|d| d.get("coRequisite").and_then(Value::as_bool) != Some(true))
                .filter_map(|d| {
                    let variant = d.get("variant").and_then(Value::as_str)?.trim();
                    let tier = YueTier::from_key(&variant.to_lowercase())?;
                    let subdir = d
                        .get("subdir")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .unwrap_or(variant)
                        .to_owned();
                    Some((
                        tier,
                        subdir,
                        d.get("default").and_then(Value::as_bool) == Some(true),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn requested_tier(
    payload: &JsonObject,
    manifest_entry: &Value,
    installed: &dyn Fn(&str) -> bool,
) -> YueTier {
    if let Some(tier) = payload
        .get("quantTier")
        .and_then(Value::as_str)
        .and_then(|tier| YueTier::from_key(&tier.trim().to_lowercase()))
    {
        return tier;
    }
    // Unset (or a tier the job itself will refuse by name): default first, then manifest order;
    // the first installed wins, and with nothing installed (the job refuses that too) the default.
    let rows = tier_rows(manifest_entry);
    let ordered = || {
        rows.iter()
            .filter(|(_, _, default)| *default)
            .chain(rows.iter().filter(|(_, _, default)| !*default))
    };
    ordered()
        .find(|(_, subdir, _)| installed(subdir))
        .or_else(|| ordered().next())
        .map_or(YueTier::Q4, |(tier, _, _)| *tier)
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
    pub stage2_weights_bytes: u64,
    pub stage2_kv_bytes: u64,
    pub codec_bytes: u64,
}

impl YueEstimate {
    pub(crate) fn stage_bytes(&self, stage: YueStage) -> u64 {
        match stage {
            YueStage::Stage1 => self.stage1_weights_bytes + self.stage1_kv_bytes,
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

/// Stage-1 KV positions: the whole history up to the checkpoint context.
pub(crate) fn stage1_kv_positions(shape: &YueRenderShape) -> u64 {
    let generated = u64::from(shape.segments) * u64::from(shape.max_new_tokens);
    shape
        .prompt_tokens
        .saturating_add(generated)
        .min(STAGE1_CONTEXT)
}

fn stage1_kv_bytes(shape: &YueRenderShape) -> u64 {
    let batch = if shape.cfg { 2 } else { 1 };
    2 * STAGE1_LAYERS
        * STAGE1_KV_HEADS
        * STAGE1_HEAD_DIM
        * KV_ELEMENT_BYTES
        * batch
        * stage1_kv_positions(shape)
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
    /// CUDA: live free VRAM on the selected card; the dedicated-VRAM allocator/context reserve is
    /// charged on top of the estimate.
    #[cfg_attr(
        any(target_os = "macos", not(feature = "backend-candle")),
        allow(dead_code)
    )]
    DedicatedVram { free_gb: f64, gpu_id: String },
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB
}

/// The admission decision: `Some(refusal)` when the floor does not fit, `None` to admit. No budget
/// admits (no evidence ⇒ no block; the engine's own load admission still stands).
pub(crate) fn admission_error(
    model: &str,
    estimate: &YueEstimate,
    shape: &YueRenderShape,
    budget: Option<&YueBudget>,
) -> Option<WorkerError> {
    let budget = budget?;
    let (stage, bytes) = estimate.floor();
    let (needed_gb, available_gb, pool) = match budget {
        YueBudget::UnifiedWorkingSet { gb } => {
            (gib(bytes), *gb, "of GPU working set on this Mac".to_owned())
        }
        YueBudget::DedicatedVram { free_gb, gpu_id } => (
            gib(bytes) + crate::fit_gate::dedicated_vram_reserve().gb,
            *free_gb,
            format!("of free VRAM on GPU {gpu_id}"),
        ),
    };
    if available_gb + f64::EPSILON >= needed_gb {
        return None;
    }
    let tier = estimate.tier.key();
    let (weights, kv) = match stage {
        YueStage::Stage1 => (estimate.stage1_weights_bytes, estimate.stage1_kv_bytes),
        YueStage::Stage2 => (estimate.stage2_weights_bytes, estimate.stage2_kv_bytes),
        YueStage::Codec => (estimate.codec_bytes, 0),
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
    Some(WorkerError::InvalidPayload(format!(
        "{model} needs ~{needed:.1} GB {pool} but only ~{available:.1} GB is available. YuE loads \
         one stage at a time and the largest is {stage_label} at the {tier} tier: ~{weights:.1} GB \
         of weights + ~{kv:.1} GB of KV cache ({segments} segment(s) × {max_new} tokens, guidance \
         {cfg}{icl}). {lever}",
        needed = needed_gb,
        available = available_gb,
        stage_label = stage.label(),
        weights = gib(weights),
        kv = gib(kv),
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

/// The pre-load gate the audio job runs: `Ok(())` for a non-YuE model or an admitted render.
/// `installed` answers whether a tier subdir is present under the model's snapshot.
pub(crate) async fn check(
    model: &str,
    payload: &JsonObject,
    manifest_entry: &Value,
    gpu_id: &str,
    // `Sync` so the audio job's future stays `Send` across the budget probe's await.
    installed: &(dyn Fn(&str) -> bool + Sync),
) -> Result<(), WorkerError> {
    if !is_yue(manifest_entry) {
        return Ok(());
    }
    let shape = YueRenderShape::from_payload(payload, manifest_entry, installed);
    let estimate = estimate(manifest_entry, &shape).map_err(|why| {
        WorkerError::InvalidPayload(format!(
            "{model}: YuE memory admission cannot price this render ({why}); the installed \
             catalog entry is incomplete — update SceneWorks before retrying."
        ))
    })?;
    let budget = live_budget(gpu_id).await;
    match admission_error(model, &estimate, &shape, budget.as_ref()) {
        Some(error) => Err(error),
        None => Ok(()),
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

    fn payload(value: Value) -> JsonObject {
        value.as_object().expect("object").clone()
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
                assert!(e.codec_bytes > GIB, "{id}: xcodec f32 component");
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
        assert_eq!(e.codec_bytes, 1_273_277_645);
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
        assert_eq!(stage1_kv_bytes(&one), per_position * (256 + 1000));
        assert_eq!(stage1_kv_bytes(&two), per_position * (256 + 2000));
        // CFG doubles it (batch-of-2).
        let cfg = shape(YueTier::Q4, 2, 1000, true);
        assert_eq!(stage1_kv_bytes(&cfg), 2 * stage1_kv_bytes(&two));
        // Past the context the smart context caps the cache at 16 384 positions (~2 GiB at CFG).
        let long = shape(YueTier::Q4, 50, 3000, true);
        assert_eq!(stage1_kv_positions(&long), 16_384);
        assert_eq!(stage1_kv_bytes(&long), 2 * per_position * 16_384);
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
        assert!(admission_error("yue_en_cot", &e, &s, Some(&mac(floor_gb))).is_none());
        // Just below: refused, naming the binding stage, tier, and figures.
        let refusal = admission_error("yue_en_cot", &e, &s, Some(&mac(floor_gb - 0.01)))
            .expect("over budget must refuse");
        let WorkerError::InvalidPayload(message) = refusal else {
            panic!("refusal must be a user-facing InvalidPayload");
        };
        assert!(message.contains("7B stage-1 LM"), "{message}");
        assert!(message.contains("q4 tier"), "{message}");
        assert!(message.contains("2 segment(s) × 3000 tokens"), "{message}");

        // CUDA charges the dedicated-VRAM reserve on top of the floor.
        let reserve = crate::fit_gate::dedicated_vram_reserve().gb;
        let cuda = |free_gb| YueBudget::DedicatedVram {
            free_gb,
            gpu_id: "0".to_owned(),
        };
        assert!(admission_error("yue_en_cot", &e, &s, Some(&cuda(floor_gb + reserve))).is_none());
        let cuda_refusal =
            admission_error("yue_en_cot", &e, &s, Some(&cuda(floor_gb + reserve - 0.01)));
        assert!(
            matches!(cuda_refusal, Some(WorkerError::InvalidPayload(ref m)) if m.contains("GPU 0"))
        );

        // No budget reading admits (no evidence ⇒ no block).
        assert!(admission_error("yue_en_cot", &e, &s, None).is_none());
    }

    #[test]
    fn a_16gb_mac_admits_q4_and_refuses_bf16() {
        let entry = builtin("yue_en_cot");
        let budget = mac(10.5); // ~2/3 of 16 GB: the M-series recommended working set.
        let q4 = shape(YueTier::Q4, 2, 3000, true);
        let bf16 = shape(YueTier::Bf16, 2, 3000, true);
        let e4 = estimate(&entry, &q4).unwrap();
        let e16 = estimate(&entry, &bf16).unwrap();
        assert!(admission_error("yue_en_cot", &e4, &q4, Some(&budget)).is_none());
        let refusal = admission_error("yue_en_cot", &e16, &bf16, Some(&budget)).unwrap();
        assert!(refusal.to_string().contains("smaller tier"), "{refusal}");
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
        let refusal = admission_error("x", &e, &s, Some(&mac(1.0))).unwrap();
        assert!(refusal.to_string().contains("1B stage-2 LM"), "{refusal}");
    }

    fn all_installed(_: &str) -> bool {
        true
    }

    fn from(value: Value, entry: &Value) -> YueRenderShape {
        YueRenderShape::from_payload(&payload(value), entry, &all_installed)
    }

    #[test]
    fn payload_shape_follows_the_sc19384_job_surface() {
        let entry = builtin("yue_en_cot");
        let defaults = from(json!({}), &entry);
        assert_eq!(
            defaults.tier,
            YueTier::Q4,
            "manifest default tier, installed"
        );
        assert_eq!((defaults.segments, defaults.max_new_tokens), (2, 3000));
        assert!(defaults.cfg, "unset guidance ⇒ the 1.5/1.2 schedule");
        assert_eq!(defaults.icl, None, "no iclMode ⇒ no reference block");

        let custom = from(
            json!({
                "segments": 5,
                "maxNewTokensPerSegment": 800,
                "guidance": 1.0,
                "quantTier": "Q8",
                "lyrics": "[verse]\nabc",
                "prompt": "pop"
            }),
            &entry,
        );
        assert_eq!(custom.tier, YueTier::Q8, "quantTier, case-insensitive");
        assert_eq!((custom.segments, custom.max_new_tokens), (5, 800));
        assert!(!custom.cfg, "guidance <= 1 turns CFG off");
        assert_eq!(custom.prompt_tokens, 64 + 3 + 2 * 11 + 32 * 5);

        let bf16 = from(json!({"quantTier": "bf16", "guidance": 3.0}), &entry);
        assert_eq!(bf16.tier, YueTier::Bf16);
        assert!(bf16.cfg, "guidance > 1 keeps CFG on");
        // `guidanceEnabled: false` switches CFG off whatever the scale says.
        let off = from(json!({"guidanceEnabled": false, "guidance": 3.0}), &entry);
        assert!(!off.cfg);
        let on = from(json!({"guidanceEnabled": true}), &entry);
        assert!(on.cfg);
        // The legacy image-lane knob is not the audio tier key.
        let legacy = from(json!({"advanced": {"mlxQuantize": 8}}), &entry);
        assert_eq!(legacy.tier, YueTier::Q4);
    }

    #[test]
    fn an_unset_tier_prices_the_tier_the_job_will_load() {
        let entry = builtin("yue_en_cot");
        let only = |dir: &'static str| move |subdir: &str| subdir == dir;
        let q8_only = only("q8");
        let shape = YueRenderShape::from_payload(&payload(json!({})), &entry, &q8_only);
        assert_eq!(
            shape.tier,
            YueTier::Q8,
            "default (q4) not installed ⇒ first installed"
        );
        let bf16_only = only("bf16");
        let shape = YueRenderShape::from_payload(&payload(json!({})), &entry, &bf16_only);
        assert_eq!(shape.tier, YueTier::Bf16);
        let none = |_: &str| false;
        let shape = YueRenderShape::from_payload(&payload(json!({})), &entry, &none);
        assert_eq!(shape.tier, YueTier::Q4, "nothing installed ⇒ the default");
        // The default wins over manifest order when it is installed.
        let reordered = json!({"downloads": [
            {"variant": "bf16"},
            {"variant": "q8", "default": true},
            {"variant": "q4"},
        ]});
        let shape = YueRenderShape::from_payload(&payload(json!({})), &reordered, &all_installed);
        assert_eq!(shape.tier, YueTier::Q8);
        // An explicit tier is priced as asked, installed or not (the job refuses a missing one).
        let shape =
            YueRenderShape::from_payload(&payload(json!({"quantTier": "bf16"})), &entry, &q8_only);
        assert_eq!(shape.tier, YueTier::Bf16);
    }

    #[test]
    fn icl_is_priced_by_the_actual_window_and_track_count() {
        let entry = builtin("yue_en_icl");
        let base = from(json!({}), &entry).prompt_tokens;
        let icl = |value: Value| from(value, &entry);
        let window = |shape: &YueRenderShape| {
            let p = shape.icl.expect("ICL priced");
            (p.tracks, p.start_secs, p.end_secs, p.window_secs())
        };

        // Both ends absent: the upstream 0–30 s default; dual = 2 × 50 tokens/s.
        let default = icl(json!({"iclMode": "Dual"}));
        assert_eq!(window(&default), (2, 0.0, 30.0, 30.0));
        assert_eq!(default.prompt_tokens, base + 3000);

        // Explicit window, single mix: 20 s × 50 = 1000 tokens.
        let explicit = icl(json!({"iclMode": "single", "iclStartSecs": 5.0, "iclEndSecs": 25.0}));
        assert_eq!(window(&explicit), (1, 5.0, 25.0, 20.0));
        assert_eq!(explicit.prompt_tokens, base + 1000);

        // Absent end = the upstream default end of 30 s whatever the start: 30 − 10 = 20 s.
        let open_end = icl(json!({"iclMode": "dual", "iclStartSecs": 10.0}));
        assert_eq!(window(&open_end), (2, 10.0, 30.0, 20.0));
        assert_eq!(open_end.prompt_tokens, base + 2000);

        // Absent start = 0: a long explicit end is priced in full (0–90 s dual = 9000 tokens).
        let long_window = icl(json!({"iclMode": "dual", "iclEndSecs": 90.0}));
        assert_eq!(window(&long_window), (2, 0.0, 90.0, 90.0));
        assert_eq!(long_window.prompt_tokens, base + 9000);

        // A longer window moves the estimate (the stage-1 KV grows with it), and the refusal
        // names the window it priced.
        let short = estimate(&entry, &explicit).unwrap();
        let long = estimate(&entry, &long_window).unwrap();
        assert!(long.stage1_kv_bytes > short.stage1_kv_bytes);
        let refusal = admission_error("yue_en_icl", &long, &long_window, Some(&mac(1.0))).unwrap();
        assert!(
            refusal
                .to_string()
                .contains("2-track ICL reference window of 90.0 s (0.0–90.0 s)"),
            "{refusal}"
        );

        // No iclMode ⇒ no reference block, whatever the window fields say.
        assert_eq!(icl(json!({"iclEndSecs": 90.0})).icl, None);
    }

    #[test]
    fn the_audio_job_runs_the_gate_before_touching_the_project_or_weights() {
        let source = include_str!("audio_jobs.rs");
        let body = source
            .split_once("pub(crate) async fn run_audio_generate_job(")
            .expect("audio job entry point")
            .1;
        let gate = body
            .find("crate::yue_admission::check(")
            .expect("the audio job must run the YuE admission gate");
        for later in [
            "get_project(",
            "build_audio_edit(",
            "resolve_voice_clone_plan(",
            "run_audio_synthesis(",
        ] {
            let at = body
                .find(later)
                .unwrap_or_else(|| panic!("{later} in the job"));
            assert!(gate < at, "the gate must run before {later}");
        }
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
            .block_on(check(
                "yue_en_cot",
                &payload(json!({})),
                &yue,
                "0",
                &all_installed,
            ))
            .unwrap_err();
        assert!(err.to_string().contains("cannot price"), "{err}");
        // A non-YuE model is never touched by this gate.
        assert!(!is_yue(&json!({"family": "ace"})));
        assert!(rt
            .block_on(check(
                "ace_step",
                &payload(json!({})),
                &json!({"family": "ace"}),
                "0",
                &all_installed,
            ))
            .is_ok());
    }
}
