//! The YuE2 job contract (sc-22999): what `POST /api/v1/projects/:id/yue2/jobs` accepts, what the
//! queued `audio_generate` payload's `yue2` block carries, and the one validation both sides run.
//!
//! The API validates a request with [`validate_request`] and refuses an invalid combination with a
//! typed error before anything is queued; the worker re-reads the stored block and runs
//! [`validate_for_execution`] before it touches weights, so a retried or duplicated payload is held
//! to the same rules as a fresh one. The engine's own gate (`candle_audio_yue2::provider`) runs last,
//! at load/validate time, and is the authority on the resolved set (its context-budget refusals are
//! not mirrored here).
//!
//! # Job kinds
//!
//! | kind | runs | needs |
//! |---|---|---|
//! | `create` | plan → semantic → acoustic → decode | `lyrics` |
//! | `plan` | plan only; publishes the exact plan (`plan_only`) | `lyrics` |
//! | `fromPlan` | synthesis from a saved plan (`audio.song.plan`) | `planJobId` |
//! | `cover` | a zero-shot cover of a reviewed score (`audio.song.cover`) | `lyrics`, `cover.score` or `cover.versionId` |
//! | `renderVersion` | renders a sc-22997 score version's request with its ABC | `versionId` |
//! | `decode` | decodes a completed run's cached latents (`audio.song.cached_latents`) | `sourceJobId` |
//! | `transcribe` | recording → reviewable score transcription (SheetSage2 + MERT-v2-FullSong) | `sourceAudioAssetId` |
//!
//! Every other field is accepted only by the kinds it means something for ([`FIELD_KINDS`]); a
//! field sent to a kind that would ignore it is a typed `yue2_invalid_combination`, never dropped.
//!
//! # A cover from a recording is two jobs
//!
//! Transcription (SheetSage2 + MERT-v2-FullSong — the cover closure, conditional cover
//! dependencies acquired only for covers, under YuE2's licence acknowledgment and noncommercial
//! policy) runs as its own `transcribe` job: the worker persists the replay-verified review
//! artifact under the project ([`TRANSCRIPTIONS_DIR`]) and unloads the transcriber, and the API
//! imports its full and melody-only scores as score versions linked to the source recording. The
//! cover is then a SEPARATE `cover` job over a reviewed (optionally edited) score version
//! (`cover.versionId`). A cover that named a recording directly would skip that review, so
//! `cover.sourceAudioAssetId` is refused with [`TRANSCRIPTION_REVIEW_REQUIRED`].

use serde::{Deserialize, Serialize};

use super::{MAX_ABC_BYTES, MAX_LYRICS_CHARS, MAX_STYLE_CHARS};

/// The catalog / provider id this contract serves.
pub const MODEL_ID: &str = "yue2";
/// The payload key the block is stored under on an `audio_generate` job.
pub const PAYLOAD_KEY: &str = "yue2";
/// Largest queued batch (one job per take, run serially on the admitted GPU).
pub const MAX_BATCH: u32 = 8;
/// Largest ODE step count accepted (the shared audio route's bound).
pub const MAX_STEPS: u32 = 10_000;
/// The model's context (`candle_audio_yue2::protocol::CONTEXT`): no phase may budget more.
pub const CONTEXT_TOKENS: u32 = 24_576;
/// Smallest NAR attention chunk the engine accepts, in score elements: one query row at the full
/// context, `heads × positions` (`candle_audio_yue2::engine::Yue2Engine::attention_bounds`, 16 × the
/// context). A smaller chunk is refused by the engine, so it is refused at submission (sc-23001).
pub const MIN_ATTENTION_CHUNK_ELEMENTS: u32 = 16 * CONTEXT_TOKENS;
/// Largest decode tile core, in latent frames (`candle_audio_yue2::decode::DecodeOptions::tiled`).
pub const MAX_DECODE_TILE_FRAMES: u32 = 1024;
/// The engine's truncation warning codes.
pub const TRUNCATION_CODES: [&str; 2] = ["abc_truncated", "semantic_truncated"];
/// Where a job's run directory lives inside its project.
pub const RUNS_DIR: &str = "yue2/runs";
/// Prefix of a server-assigned run id.
pub const RUN_ID_PREFIX: &str = "yue2run_";
/// Where a transcription's review artifact lives inside its project (`<dir>/<runId>`).
pub const TRANSCRIPTIONS_DIR: &str = "yue2/transcriptions";
/// Prefix of a transcription record id (`yue2t_` + the transcribe job's run-id suffix).
pub const TRANSCRIPTION_ID_PREFIX: &str = "yue2t_";
/// SheetSage2's fixed context window, seconds (`Tokenizer::audio_length_seconds` of the pinned
/// checkpoint): the window overlap must be shorter than it.
pub const TRANSCRIPTION_WINDOW_SECONDS: f64 = 300.0;
/// `candle_audio_sheetsage2::pipeline::DEFAULT_OVERLAP_SECONDS` (upstream's default preset).
pub const TRANSCRIPTION_DEFAULT_OVERLAP_SECONDS: f64 = 200.0;
/// `candle_audio_sheetsage2::pipeline::DEFAULT_LOOKAHEAD_SECONDS` (upstream's default preset).
pub const TRANSCRIPTION_DEFAULT_LOOKAHEAD_SECONDS: f64 = 100.0;

/// The (min_tokens, max_tokens) defaults of the planning (ABC) and semantic phases
/// (`candle_audio_yue2::protocol::Sampling::{abc_default, semantic_default}`), so a request that
/// overrides only one bound is checked against the resolved pair.
pub const ABC_TOKEN_DEFAULTS: (u32, u32) = (32, 4096);
pub const SEMANTIC_TOKEN_DEFAULTS: (u32, u32) = (200, 9000);

/// What a YuE2 job does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Yue2JobKind {
    Create,
    Plan,
    FromPlan,
    Cover,
    RenderVersion,
    Decode,
    Transcribe,
}

impl Yue2JobKind {
    pub const ALL: [Self; 7] = [
        Self::Create,
        Self::Plan,
        Self::FromPlan,
        Self::Cover,
        Self::RenderVersion,
        Self::Decode,
        Self::Transcribe,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Plan => "plan",
            Self::FromPlan => "fromPlan",
            Self::Cover => "cover",
            Self::RenderVersion => "renderVersion",
            Self::Decode => "decode",
            Self::Transcribe => "transcribe",
        }
    }

    /// Whether the job renders audio (every kind but `plan` and `transcribe`).
    pub fn renders_audio(self) -> bool {
        !matches!(self, Self::Plan | Self::Transcribe)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Planning {
    Full,
    Melody,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decoder {
    Standard,
    Legacy,
}

impl Decoder {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Legacy => "legacy",
        }
    }
    /// The catalog `componentId` of the decoder snapshot.
    pub fn component_id(self) -> &'static str {
        match self {
            Self::Standard => "vae",
            Self::Legacy => "vae_legacy",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Bf16,
    Q8,
    Q4,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }
}

/// `LoadSpec::precision`: `default` computes in BF16 on an accelerator and F32 on the CPU; `fp32`
/// forces F32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputePrecision {
    Default,
    Fp32,
}

/// `LoadSpec::offload_policy`: `sequential` moves the AR-only weights to host memory while the
/// acoustic stage runs (upstream `offload_ar`) unless a request's `memory.stageResidency` decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OffloadPolicy {
    Resident,
    Sequential,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CoverMode {
    Melody,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CoverKeep {
    Both,
    Vocal,
    Instrumental,
}

/// One autoregressive phase's sampling overrides (`gen_core::TokenSampling`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TokenSampling {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub penalty_window: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// The per-request memory controls YuE2 honours (`gen_core::GenerationMemory`). Passed through
/// unsized: admission (sc-23001) decides them, this job only carries what it is given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryControls {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_residency: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_attention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_chunk_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile_vae_decode: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tile_edge: Option<u32>,
}

impl MemoryControls {
    fn acoustic_set(&self) -> bool {
        self.stage_residency.is_some()
            || self.chunk_attention.is_some()
            || self.attention_chunk_size.is_some()
    }
    fn decode_set(&self) -> bool {
        self.tile_vae_decode.is_some() || self.decode_tile_edge.is_some()
    }
}

/// A cover's reviewed score and options (`gen_core::SongCover`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CoverSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<CoverMode>,
    /// The reviewed score, inline…
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<String>,
    /// …or a sc-22997 score version whose ABC is the reviewed score (resolved to `score` at
    /// submission, its SHA-256 recorded in `sources.coverVersion`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep: Option<CoverKeep>,
    /// Source lyrics, when `lyrics` is their section-aligned translation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translated_from: Option<String>,
    /// A recording to cover directly. Always refused with [`TRANSCRIPTION_REVIEW_REQUIRED`]: a
    /// cover from a recording is a `transcribe` job, a reviewed score version, then a cover of that
    /// version (see the module docs). Kept as a field so the refusal says so, instead of a generic
    /// unknown-field error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_audio_asset_id: Option<String>,
}

/// How a `transcribe` job transcribes (`candle_audio_sheetsage2::review::TranscriptionSettings`).
/// The task prompts are always upstream's full default set, which the review and both cover modes
/// need. An unset field takes the engine's default-preset value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TranscriptionSettings {
    /// Crop the recording to this many seconds before transcribing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seconds: Option<f64>,
    /// Overlap between consecutive 300 s windows, seconds (default 200).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlap_seconds: Option<f64>,
    /// Right-hand look-ahead of each non-final window, seconds (default 100).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookahead_seconds: Option<f64>,
}

impl TranscriptionSettings {
    /// The overlap the engine will use.
    pub fn resolved_overlap_seconds(&self) -> f64 {
        self.overlap_seconds
            .unwrap_or(TRANSCRIPTION_DEFAULT_OVERLAP_SECONDS)
    }

    /// The look-ahead the engine will use.
    pub fn resolved_lookahead_seconds(&self) -> f64 {
        self.lookahead_seconds
            .unwrap_or(TRANSCRIPTION_DEFAULT_LOOKAHEAD_SECONDS)
    }
}

/// A completed run another job reads (a plan to restore, latents to decode). Server-resolved from
/// the source job's recorded result; the worker verifies `identity` before it reads the directory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RunSource {
    pub job_id: String,
    /// Project-relative run directory.
    pub run_dir: String,
    /// The run's recorded identity (`result.json` `identity`).
    pub identity: String,
    /// The recorded plan identity (`plan_identity`), which `SavedPlan::identity` is checked against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_identity: Option<String>,
}

/// A score version a job renders or covers, with the digests it must still have at execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VersionSource {
    pub id: String,
    pub score_sha256: String,
    pub request_sha256: String,
}

/// Server-resolved inputs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sources {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<RunSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_run: Option<RunSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<VersionSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover_version: Option<VersionSource>,
    /// The recording a `transcribe` job transcribes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recording: Option<RecordingSource>,
    /// The transcription a covered score version was imported (or edited) from, so the cover
    /// names its source recording.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcription: Option<TranscriptionSource>,
}

/// A project audio asset a `transcribe` job reads, with the SHA-256 of its media file at
/// submission; the worker refuses a file that no longer has it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RecordingSource {
    pub asset_id: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The transcription behind a covered score version.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TranscriptionSource {
    pub id: String,
    pub job_id: String,
    pub source_audio_asset_id: String,
    /// SHA-256 of the review artifact's `transcription.json`.
    pub manifest_sha256: String,
    /// Which transcribed score the version started from.
    pub mode: CoverMode,
}

/// One take of a queued batch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BatchMember {
    pub id: String,
    pub index: u32,
    pub count: u32,
}

/// The request body of `POST /api/v1/projects/:id/yue2/jobs`, and — with the envelope fields
/// cleared and the server fields set — the stored `yue2` block of the queued job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Yue2JobSpec {
    pub kind: Yue2JobKind,
    /// The style prompt (`GenerationRequest::prompt`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lyrics: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f64>,
    /// Midpoint ODE steps of the acoustic stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planning: Option<Planning>,
    /// An external ABC score to plan from instead of sampling one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_sampling: Option<TokenSampling>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_sampling: Option<TokenSampling>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder: Option<Decoder>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<ComputePrecision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offload_policy: Option<OffloadPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryControls>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover: Option<CoverSpec>,
    /// The recording a `transcribe` job transcribes (a project audio asset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_audio_asset_id: Option<String>,
    /// How a `transcribe` job transcribes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcription: Option<TranscriptionSettings>,

    // ---- request envelope (never stored in the block) ----
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    /// Takes to queue (`create` / `plan` / `cover`); each is its own job with seed `seed + i`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    /// The caller declares a commercial use for the output. YuE2 is refused on it (E2).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub commercial_use: bool,
    /// The caller asserts the user accepted the licence now (recorded server-side).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub license_acknowledged: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_gpu: Option<String>,

    // ---- server-assigned (refused in a request) ----
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<BatchMember>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Sources>,
}

/// Why a spec is refused. `code` is the typed API error code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yue2JobError {
    pub code: &'static str,
    pub field: String,
    pub message: String,
}

impl std::fmt::Display for Yue2JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for Yue2JobError {}

/// A field sent to a kind that does not read it.
pub const INVALID_COMBINATION: &str = "yue2_invalid_combination";
/// A field outside its protocol limits.
pub const INVALID_VALUE: &str = "yue2_invalid_value";
/// A field the kind requires is absent.
pub const MISSING_FIELD: &str = "yue2_missing_field";
/// A cover named a recording directly: a cover from a recording goes through a reviewed
/// transcription (see the module docs).
pub const TRANSCRIPTION_REVIEW_REQUIRED: &str = "yue2_transcription_review_required";

fn error(code: &'static str, field: &str, message: impl Into<String>) -> Yue2JobError {
    Yue2JobError {
        code,
        field: field.to_owned(),
        message: message.into(),
    }
}

use Yue2JobKind::Transcribe as K_TRANSCRIBE;
use Yue2JobKind::{Cover as K_COVER, Create as K_CREATE, Decode as K_DECODE};
use Yue2JobKind::{FromPlan as K_FROM_PLAN, Plan as K_PLAN, RenderVersion as K_RENDER};

/// Which kinds read each optional field. A field set on any other kind is refused.
pub const FIELD_KINDS: &[(&str, &[Yue2JobKind])] = &[
    ("style", &[K_CREATE, K_PLAN, K_FROM_PLAN, K_COVER]),
    ("lyrics", &[K_CREATE, K_PLAN, K_FROM_PLAN, K_COVER]),
    ("seed", &[K_CREATE, K_PLAN, K_COVER]),
    ("cfgScale", &[K_CREATE, K_PLAN, K_COVER]),
    ("steps", &[K_CREATE, K_FROM_PLAN, K_COVER, K_RENDER]),
    ("planning", &[K_CREATE, K_PLAN]),
    ("score", &[K_CREATE, K_PLAN]),
    ("scoreSampling", &[K_CREATE, K_PLAN]),
    (
        "semanticSampling",
        &[K_CREATE, K_FROM_PLAN, K_COVER, K_RENDER],
    ),
    (
        "decoder",
        &[K_CREATE, K_FROM_PLAN, K_COVER, K_RENDER, K_DECODE],
    ),
    (
        "tier",
        &[K_CREATE, K_PLAN, K_FROM_PLAN, K_COVER, K_RENDER, K_DECODE],
    ),
    (
        "precision",
        &[K_CREATE, K_PLAN, K_FROM_PLAN, K_COVER, K_RENDER, K_DECODE],
    ),
    ("offloadPolicy", &[K_CREATE, K_FROM_PLAN, K_COVER, K_RENDER]),
    (
        "memory.acoustic",
        &[K_CREATE, K_FROM_PLAN, K_COVER, K_RENDER],
    ),
    (
        "memory.decode",
        &[K_CREATE, K_FROM_PLAN, K_COVER, K_RENDER, K_DECODE],
    ),
    ("planJobId", &[K_FROM_PLAN]),
    ("sourceJobId", &[K_DECODE]),
    ("versionId", &[K_RENDER]),
    ("cover", &[K_COVER]),
    ("sourceAudioAssetId", &[K_TRANSCRIBE]),
    ("transcription", &[K_TRANSCRIBE]),
    ("count", &[K_CREATE, K_PLAN, K_COVER]),
];

fn why_not(field: &str, kind: Yue2JobKind) -> String {
    let reason = match (field, kind) {
        ("seed" | "cfgScale" | "planning" | "score" | "scoreSampling", K_FROM_PLAN) => {
            "a saved plan fixes it (an edited plan is a new request)"
        }
        (_, K_RENDER) => {
            "the score version fixes its style, lyrics, planning, seed and guidance; edit the \
             version to change them"
        }
        (_, K_DECODE) => {
            "a cached decode re-renders the source run's latents and generates nothing"
        }
        (_, K_PLAN) => "a plan-only job stops after planning the score",
        (_, K_COVER) => "a cover plans from its own reviewed score in its own mode",
        (_, K_TRANSCRIBE) => {
            "transcription takes only the source recording and its transcription settings"
        }
        ("count", _) => {
            "a restored plan, a score version and a cached decode render the same take every time"
        }
        _ => "this kind does not read it",
    };
    format!("not accepted by a `{}` job: {reason}", kind.as_str())
}

fn set_fields(spec: &Yue2JobSpec) -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut push = |name: &'static str, set: bool| {
        if set {
            out.push(name);
        }
    };
    push("style", spec.style.is_some());
    push("lyrics", spec.lyrics.is_some());
    push("seed", spec.seed.is_some());
    push("cfgScale", spec.cfg_scale.is_some());
    push("steps", spec.steps.is_some());
    push("planning", spec.planning.is_some());
    push("score", spec.score.is_some());
    push("scoreSampling", spec.score_sampling.is_some());
    push("semanticSampling", spec.semantic_sampling.is_some());
    push("decoder", spec.decoder.is_some());
    push("tier", spec.tier.is_some());
    push("precision", spec.precision.is_some());
    push("offloadPolicy", spec.offload_policy.is_some());
    push(
        "memory.acoustic",
        spec.memory.is_some_and(|m| m.acoustic_set()),
    );
    push("memory.decode", spec.memory.is_some_and(|m| m.decode_set()));
    push("planJobId", spec.plan_job_id.is_some());
    push("sourceJobId", spec.source_job_id.is_some());
    push("versionId", spec.version_id.is_some());
    push("cover", spec.cover.is_some());
    push("sourceAudioAssetId", spec.source_audio_asset_id.is_some());
    push("transcription", spec.transcription.is_some());
    push("count", spec.count.is_some_and(|c| c != 1));
    out
}

fn check_text(field: &str, value: &str, max: usize) -> Result<(), Yue2JobError> {
    let count = value.chars().count();
    if count > max {
        return Err(error(
            INVALID_VALUE,
            field,
            format!("is {count} characters; the limit is {max}"),
        ));
    }
    Ok(())
}

fn check_id(field: &str, value: &str) -> Result<(), Yue2JobError> {
    let safe = !value.trim().is_empty()
        && value.len() <= 200
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'));
    if !safe {
        return Err(error(INVALID_VALUE, field, "is not a valid id"));
    }
    Ok(())
}

fn check_sampling(
    field: &str,
    sampling: &TokenSampling,
    defaults: (u32, u32),
) -> Result<(), Yue2JobError> {
    let finite = |name: &str, value: Option<f64>| match value {
        Some(v) if !v.is_finite() => Err(error(
            INVALID_VALUE,
            &format!("{field}.{name}"),
            "must be finite",
        )),
        _ => Ok(()),
    };
    finite("temperature", sampling.temperature)?;
    finite("topP", sampling.top_p)?;
    finite("repetitionPenalty", sampling.repetition_penalty)?;
    if let Some(t) = sampling.temperature {
        if !(0.0..=5.0).contains(&t) {
            return Err(error(
                INVALID_VALUE,
                &format!("{field}.temperature"),
                "must be in [0, 5]",
            ));
        }
    }
    if let Some(p) = sampling.top_p {
        if !(p > 0.0 && p <= 1.0) {
            return Err(error(
                INVALID_VALUE,
                &format!("{field}.topP"),
                "must be in (0, 1]",
            ));
        }
    }
    if sampling.top_k == Some(0) {
        return Err(error(
            INVALID_VALUE,
            &format!("{field}.topK"),
            "must be >= 1",
        ));
    }
    if sampling.repetition_penalty.is_some_and(|r| r <= 0.0) {
        return Err(error(
            INVALID_VALUE,
            &format!("{field}.repetitionPenalty"),
            "must be > 0",
        ));
    }
    if sampling
        .penalty_window
        .is_some_and(|w| !(1..=100).contains(&w))
    {
        return Err(error(
            INVALID_VALUE,
            &format!("{field}.penaltyWindow"),
            "must be in [1, 100]",
        ));
    }
    let min = sampling.min_tokens.unwrap_or(defaults.0);
    let max = sampling.max_tokens.unwrap_or(defaults.1);
    if max < 1 || min > max {
        return Err(error(
            INVALID_VALUE,
            &format!("{field}.minTokens/maxTokens"),
            format!(
                "require 0 <= minTokens <= maxTokens and maxTokens >= 1 (resolved {min} / {max})"
            ),
        ));
    }
    if max > CONTEXT_TOKENS {
        return Err(error(
            INVALID_VALUE,
            &format!("{field}.maxTokens"),
            format!("{max} exceeds the model's {CONTEXT_TOKENS}-token context"),
        ));
    }
    Ok(())
}

/// Validate a request body (see the [module docs](self)). Server-assigned fields are refused.
pub fn validate_request(spec: &Yue2JobSpec) -> Result<(), Yue2JobError> {
    for (field, set) in [
        ("runId", spec.run_id.is_some()),
        ("batch", spec.batch.is_some()),
        ("sources", spec.sources.is_some()),
    ] {
        if set {
            return Err(error(
                INVALID_COMBINATION,
                field,
                "is assigned by the server and cannot be sent",
            ));
        }
    }
    if let Some(cover) = &spec.cover {
        if cover.score.is_some() && cover.version_id.is_some() {
            return Err(error(
                INVALID_COMBINATION,
                "cover",
                "name exactly one of cover.score and cover.versionId",
            ));
        }
    }
    if let Some(count) = spec.count {
        if !(1..=MAX_BATCH).contains(&count) {
            return Err(error(
                INVALID_VALUE,
                "count",
                format!("must be in [1, {MAX_BATCH}]"),
            ));
        }
    }
    validate_common(spec)
}

/// Validate a stored block before execution: the request rules, plus the server-resolved inputs
/// each kind needs. The envelope fields must have been cleared.
pub fn validate_for_execution(spec: &Yue2JobSpec) -> Result<(), Yue2JobError> {
    validate_common(spec)?;
    let run_id = spec
        .run_id
        .as_deref()
        .ok_or_else(|| error(MISSING_FIELD, "runId", "the job carries no run id"))?;
    if !run_id.starts_with(RUN_ID_PREFIX) {
        return Err(error(INVALID_VALUE, "runId", "is not a YuE2 run id"));
    }
    check_id("runId", run_id)?;
    let sources = spec.sources.clone().unwrap_or_default();
    let need = |set: bool, field: &str| {
        if set {
            Ok(())
        } else {
            Err(error(
                MISSING_FIELD,
                field,
                "the job was not resolved by the server",
            ))
        }
    };
    match spec.kind {
        K_FROM_PLAN => need(sources.plan.is_some(), "sources.plan")?,
        K_DECODE => need(sources.source_run.is_some(), "sources.sourceRun")?,
        K_RENDER => need(sources.version.is_some(), "sources.version")?,
        K_COVER => need(
            spec.cover.as_ref().is_some_and(|c| c.score.is_some()),
            "cover.score",
        )?,
        K_TRANSCRIBE => need(sources.recording.is_some(), "sources.recording")?,
        _ => {}
    }
    if let Some(recording) = &sources.recording {
        check_id("sources.recording.assetId", &recording.asset_id)?;
        if spec.source_audio_asset_id.as_deref() != Some(recording.asset_id.as_str()) {
            return Err(error(
                INVALID_COMBINATION,
                "sources.recording",
                "is not the job's sourceAudioAssetId",
            ));
        }
        check_sha256("sources.recording.sha256", &recording.sha256)?;
    }
    if let Some(transcription) = &sources.transcription {
        if !transcription.id.starts_with(TRANSCRIPTION_ID_PREFIX) {
            return Err(error(
                INVALID_VALUE,
                "sources.transcription.id",
                "is not a transcription id",
            ));
        }
        check_id("sources.transcription.id", &transcription.id)?;
        check_id("sources.transcription.jobId", &transcription.job_id)?;
        check_id(
            "sources.transcription.sourceAudioAssetId",
            &transcription.source_audio_asset_id,
        )?;
        check_sha256(
            "sources.transcription.manifestSha256",
            &transcription.manifest_sha256,
        )?;
    }
    for run in [&sources.plan, &sources.source_run].into_iter().flatten() {
        check_id("sources.jobId", &run.job_id)?;
        if !is_run_dir(&run.run_dir) {
            return Err(error(
                INVALID_VALUE,
                "sources.runDir",
                "is not a project YuE2 run directory",
            ));
        }
        if !is_lower_hex(&run.identity) {
            return Err(error(
                INVALID_VALUE,
                "sources.identity",
                "is not a hex identity",
            ));
        }
    }
    Ok(())
}

fn check_sha256(field: &str, value: &str) -> Result<(), Yue2JobError> {
    if value.len() != 64 || !is_lower_hex(value) {
        return Err(error(INVALID_VALUE, field, "is not a SHA-256"));
    }
    Ok(())
}

fn is_lower_hex(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

/// Whether `rel` is exactly `yue2/runs/<run id>` (no traversal, no nesting).
pub fn is_run_dir(rel: &str) -> bool {
    rel.strip_prefix(RUNS_DIR)
        .and_then(|rest| rest.strip_prefix('/'))
        .is_some_and(|id| {
            id.starts_with(RUN_ID_PREFIX)
                && id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        })
}

/// The project-relative run directory of `run_id`.
pub fn run_dir(run_id: &str) -> String {
    format!("{RUNS_DIR}/{run_id}")
}

/// The project-relative review-artifact directory of a `transcribe` job's `run_id`.
pub fn transcription_dir(run_id: &str) -> String {
    format!("{TRANSCRIPTIONS_DIR}/{run_id}")
}

/// The transcription record id of a `transcribe` job's `run_id` — deterministic, so a retried job
/// and the API's idempotent side effects name the same record.
pub fn transcription_id(run_id: &str) -> Option<String> {
    run_id
        .strip_prefix(RUN_ID_PREFIX)
        .filter(|rest| !rest.is_empty())
        .map(|rest| format!("{TRANSCRIPTION_ID_PREFIX}{rest}"))
}

/// Whether `rel` is exactly `yue2/transcriptions/<run id>` (no traversal, no nesting).
pub fn is_transcription_dir(rel: &str) -> bool {
    rel.strip_prefix(TRANSCRIPTIONS_DIR)
        .and_then(|rest| rest.strip_prefix('/'))
        .is_some_and(|id| {
            id.starts_with(RUN_ID_PREFIX)
                && id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        })
}

fn validate_common(spec: &Yue2JobSpec) -> Result<(), Yue2JobError> {
    let kind = spec.kind;
    for field in set_fields(spec) {
        let allowed = FIELD_KINDS
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, kinds)| kinds.contains(&kind))
            .unwrap_or(false);
        if !allowed {
            let shown = field.split('.').next().unwrap_or(field);
            return Err(error(INVALID_COMBINATION, shown, why_not(field, kind)));
        }
    }
    // Per-kind requirements.
    let lyrics_needed = matches!(kind, K_CREATE | K_PLAN | K_COVER);
    if lyrics_needed && spec.lyrics.as_deref().is_none_or(|l| l.trim().is_empty()) {
        return Err(error(MISSING_FIELD, "lyrics", "is required"));
    }
    match kind {
        K_FROM_PLAN => {
            spec.plan_job_id
                .as_deref()
                .ok_or_else(|| error(MISSING_FIELD, "planJobId", "is required"))
                .and_then(|id| check_id("planJobId", id))?;
        }
        K_DECODE => {
            spec.source_job_id
                .as_deref()
                .ok_or_else(|| error(MISSING_FIELD, "sourceJobId", "is required"))
                .and_then(|id| check_id("sourceJobId", id))?;
        }
        K_RENDER => {
            spec.version_id
                .as_deref()
                .ok_or_else(|| error(MISSING_FIELD, "versionId", "is required"))
                .and_then(|id| check_id("versionId", id))?;
        }
        K_TRANSCRIBE => {
            spec.source_audio_asset_id
                .as_deref()
                .ok_or_else(|| error(MISSING_FIELD, "sourceAudioAssetId", "is required"))
                .and_then(|id| check_id("sourceAudioAssetId", id))?;
            if let Some(settings) = &spec.transcription {
                check_transcription(settings)?;
            }
        }
        K_COVER => {
            let cover = spec
                .cover
                .as_ref()
                .ok_or_else(|| error(MISSING_FIELD, "cover", "is required"))?;
            if cover.source_audio_asset_id.is_some() {
                return Err(error(
                    TRANSCRIPTION_REVIEW_REQUIRED,
                    "cover.sourceAudioAssetId",
                    "a cover from a recording is two steps: transcribe the recording (kind \
                     \"transcribe\"), review or edit the score version it imports, then cover \
                     that version (cover.versionId)",
                ));
            }
            if cover.mode.is_none() {
                return Err(error(
                    MISSING_FIELD,
                    "cover.mode",
                    "is required (melody or full)",
                ));
            }
            let sources = [cover.score.is_some(), cover.version_id.is_some()]
                .iter()
                .filter(|s| **s)
                .count();
            // At execution the server has inlined the version's score, so both are present then;
            // what must hold is that a request names exactly one source.
            let resolved_version = cover.version_id.is_some() && cover.score.is_some();
            if sources == 0 {
                return Err(error(
                    MISSING_FIELD,
                    "cover.score",
                    "a cover needs a reviewed score (cover.score or cover.versionId)",
                ));
            }
            if sources > 1 && !resolved_version {
                return Err(error(
                    INVALID_COMBINATION,
                    "cover",
                    "name exactly one of cover.score and cover.versionId",
                ));
            }
            if cover.keep.is_some() && cover.mode != Some(CoverMode::Melody) {
                return Err(error(
                    INVALID_COMBINATION,
                    "cover.keep",
                    "only a melody cover chooses which melodies it keeps",
                ));
            }
            if let Some(score) = &cover.score {
                check_score("cover.score", score)?;
            }
            if let Some(id) = &cover.version_id {
                check_id("cover.versionId", id)?;
            }
            if let Some(source) = &cover.translated_from {
                check_text("cover.translatedFrom", source, MAX_LYRICS_CHARS)?;
                if source.trim().is_empty() {
                    return Err(error(INVALID_VALUE, "cover.translatedFrom", "is empty"));
                }
            }
        }
        _ => {}
    }
    if kind == K_PLAN && spec.planning == Some(Planning::Off) {
        return Err(error(
            INVALID_COMBINATION,
            "planning",
            "a plan-only job needs a planning mode with a score (full or melody)",
        ));
    }
    if let Some(score) = &spec.score {
        if spec.planning == Some(Planning::Off) {
            return Err(error(
                INVALID_COMBINATION,
                "score",
                "an external score needs planning full or melody",
            ));
        }
        check_score("score", score)?;
        if spec.score_sampling.is_some() {
            return Err(error(
                INVALID_COMBINATION,
                "scoreSampling",
                "with an external score nothing is sampled in the planning phase",
            ));
        }
    }
    if spec.planning == Some(Planning::Off) && spec.score_sampling.is_some() {
        return Err(error(
            INVALID_COMBINATION,
            "scoreSampling",
            "planning off samples no score",
        ));
    }
    if let Some(style) = &spec.style {
        check_text("style", style, MAX_STYLE_CHARS)?;
    }
    if let Some(lyrics) = &spec.lyrics {
        check_text("lyrics", lyrics, MAX_LYRICS_CHARS)?;
    }
    if spec.seed.is_some_and(|s| s >= 1 << 63) {
        return Err(error(
            INVALID_VALUE,
            "seed",
            "must be an integer in [0, 2^63)",
        ));
    }
    if let Some(cfg) = spec.cfg_scale {
        if !cfg.is_finite() || !(0.0..=20.0).contains(&cfg) {
            return Err(error(
                INVALID_VALUE,
                "cfgScale",
                "must be finite and in [0, 20]",
            ));
        }
    }
    if let Some(steps) = spec.steps {
        if !(1..=MAX_STEPS).contains(&steps) {
            return Err(error(
                INVALID_VALUE,
                "steps",
                format!("must be in [1, {MAX_STEPS}]"),
            ));
        }
    }
    if let Some(s) = &spec.score_sampling {
        check_sampling("scoreSampling", s, ABC_TOKEN_DEFAULTS)?;
    }
    if let Some(s) = &spec.semantic_sampling {
        check_sampling("semanticSampling", s, SEMANTIC_TOKEN_DEFAULTS)?;
    }
    if let Some(memory) = &spec.memory {
        if memory.attention_chunk_size.is_some() && memory.chunk_attention != Some(true) {
            return Err(error(
                INVALID_COMBINATION,
                "memory.attentionChunkSize",
                "is read only with memory.chunkAttention",
            ));
        }
        if memory
            .attention_chunk_size
            .is_some_and(|elements| elements < MIN_ATTENTION_CHUNK_ELEMENTS)
        {
            return Err(error(
                INVALID_VALUE,
                "memory.attentionChunkSize",
                format!(
                    "must be >= {MIN_ATTENTION_CHUNK_ELEMENTS} score elements (one query row at \
                     the full context: 16 heads × {CONTEXT_TOKENS} keys)"
                ),
            ));
        }
        if memory.decode_tile_edge.is_some() && memory.tile_vae_decode != Some(true) {
            return Err(error(
                INVALID_COMBINATION,
                "memory.decodeTileEdge",
                "is read only with memory.tileVaeDecode",
            ));
        }
        if memory
            .decode_tile_edge
            .is_some_and(|edge| !(1..=MAX_DECODE_TILE_FRAMES).contains(&edge))
        {
            return Err(error(
                INVALID_VALUE,
                "memory.decodeTileEdge",
                format!("must be in [1, {MAX_DECODE_TILE_FRAMES}] latent frames"),
            ));
        }
    }
    for (field, id) in [
        ("planJobId", &spec.plan_job_id),
        ("sourceJobId", &spec.source_job_id),
        ("versionId", &spec.version_id),
        ("sourceAudioAssetId", &spec.source_audio_asset_id),
    ] {
        if let Some(id) = id {
            check_id(field, id)?;
        }
    }
    Ok(())
}

fn check_score(field: &str, abc: &str) -> Result<(), Yue2JobError> {
    if abc.len() > MAX_ABC_BYTES {
        return Err(error(
            INVALID_VALUE,
            field,
            format!("is {} bytes; the limit is {MAX_ABC_BYTES}", abc.len()),
        ));
    }
    super::parse_score(abc)
        .map(|_| ())
        .map_err(|e| error("yue2_unsupported_notation", field, e.to_string()))
}

/// A duplicate is a NEW take, never a resume: give a `yue2` payload block a fresh run id and drop
/// its batch membership, so two jobs never share a run directory. (A retry keeps its run id — it
/// resumes the same run's verified checkpoints.) A payload without a block is left untouched.
pub fn refresh_block_for_duplicate(payload: &mut serde_json::Map<String, serde_json::Value>) {
    let Some(block) = payload
        .get_mut(PAYLOAD_KEY)
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    block.insert(
        "runId".to_owned(),
        serde_json::Value::String(format!("{RUN_ID_PREFIX}{}", fresh_hex_id())),
    );
    block.remove("batch");
}

fn fresh_hex_id() -> String {
    let mut bytes = [0u8; 16];
    // A failed OS RNG must not hand two duplicates the same run: fall back to a time-and-address
    // mix, which is unique per call in one process.
    if getrandom::fill(&mut bytes).is_err() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let marker = &bytes as *const _ as usize as u128;
        bytes = (nanos ^ marker.rotate_left(64)).to_le_bytes();
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The engine's window-plan rule (`sliding_window_plan`) over the resolved values —
/// `0 <= lookahead <= overlap < 300 s` — and a finite, positive crop.
fn check_transcription(settings: &TranscriptionSettings) -> Result<(), Yue2JobError> {
    for (field, value) in [
        ("transcription.maxSeconds", settings.max_seconds),
        ("transcription.overlapSeconds", settings.overlap_seconds),
        ("transcription.lookaheadSeconds", settings.lookahead_seconds),
    ] {
        if value.is_some_and(|v| !v.is_finite()) {
            return Err(error(INVALID_VALUE, field, "must be finite"));
        }
    }
    if settings.max_seconds.is_some_and(|v| v <= 0.0) {
        return Err(error(
            INVALID_VALUE,
            "transcription.maxSeconds",
            "must be positive",
        ));
    }
    let overlap = settings.resolved_overlap_seconds();
    let lookahead = settings.resolved_lookahead_seconds();
    if !(0.0..TRANSCRIPTION_WINDOW_SECONDS).contains(&overlap) {
        return Err(error(
            INVALID_VALUE,
            "transcription.overlapSeconds",
            format!(
                "must be in [0, {TRANSCRIPTION_WINDOW_SECONDS}) seconds (resolved {overlap}); \
                 SheetSage2 transcribes {TRANSCRIPTION_WINDOW_SECONDS} s windows"
            ),
        ));
    }
    if !(0.0..=overlap).contains(&lookahead) {
        return Err(error(
            INVALID_VALUE,
            "transcription.lookaheadSeconds",
            format!("must be in [0, overlapSeconds] (resolved {lookahead} / {overlap})"),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "jobs_tests.rs"]
mod tests;
