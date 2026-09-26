//! YuE2 score inspection, bounded edits and versioned listening comparisons (sc-22997,
//! epic 22988).
//!
//! YuE2 (experimental, noncommercial weights) plans a song as a two-voice ABC score before it
//! generates audio, and its released workflow edits that score — reharmonize, revise melody,
//! lyrics, style, tempo or form — and then regenerates the whole recording. Upstream keeps the
//! score tooling in its agent *skill* (`skills/yue2-music/scripts/abc_tools.py` at YuE commit
//! `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`), not in the engine, so this module is an app
//! concern with no engine dependency:
//!
//! - [`abc`] — native port of the bounded ABC dialect parser/inspector.
//! - [`compare`] — change contracts and the musical-invariant check (parsed events, not text).
//! - [`ops`] — the bounded edit-operation set; every result is re-parsed and invariant-checked.
//! - [`store`] — immutable per-project version, render and comparison records.
//!
//! An edit never overwrites anything: it creates a new version linked to its source. Rendering a
//! version (sc-22999) regenerates the complete recording — see [`REGENERATION_NOTICE`].

pub mod abc;
pub mod compare;
pub mod ops;
pub mod store;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use crate::project_store::ProjectStoreError;

pub use abc::{inspect, parse_score, AbcError, Score, ScoreInspection};
pub use compare::{ChangeContract, InvariantReport};
pub use ops::{apply_operation, ScoreEditOperation};

/// Stated on every version, render and comparison, and in the agent tool descriptions: YuE2 edits
/// are score/prompt edits, not waveform edits.
pub const REGENERATION_NOTICE: &str = "Rendering a YuE2 score version regenerates the whole \
recording from its score, style and lyrics. It does not edit or preserve the earlier waveform: \
singing, timbre and arrangement can differ everywhere, including outside the edited bars, and a \
matching score does not guarantee identical audio. Compare complete recordings.";

/// Largest accepted ABC score (bytes). A several-minute native score is a few tens of KiB.
pub const MAX_ABC_BYTES: usize = 256 * 1024;
/// Largest accepted style prompt (characters) — the API's shared prompt cap.
pub const MAX_STYLE_CHARS: usize = crate::MAX_PROMPT_CHARS;
/// Largest accepted lyrics text (characters).
pub const MAX_LYRICS_CHARS: usize = 16_000;
/// Largest accepted edit brief / listening note (characters).
pub const MAX_BRIEF_CHARS: usize = 4_000;
/// Upstream `SongRequest` default seed.
pub const DEFAULT_SEED: u64 = 831_001;

/// YuE2 chain-of-thought mode for a score-conditioned request. External ABC requires `full`
/// (melody + harmony) or `melody` (chord-free); `off` has no score and cannot be edited.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Cot {
    Full,
    Melody,
}

impl Cot {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Melody => "melody",
        }
    }
}

fn default_seed() -> u64 {
    DEFAULT_SEED
}

/// The generation request a score version renders with (upstream `SongRequest` minus `abc`,
/// which is the version's score, and `id`, which is the version id).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SongRequest {
    pub style: String,
    pub lyrics: String,
    pub cot: Cot,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f64>,
}

/// Errors from score inspection, editing and version storage.
#[derive(Debug)]
pub enum Yue2ScoreError {
    /// Outside the supported native dialect (not necessarily invalid under the full ABC standard).
    Notation(AbcError),
    /// The edit changed something its contract fixes.
    Invariant(Box<InvariantReport>),
    BadRequest(String),
    NotFound(String),
    Conflict(String),
    Store(ProjectStoreError),
}

impl std::fmt::Display for Yue2ScoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Notation(error) => write!(
                formatter,
                "Unsupported YuE2 ABC notation: {error} (outside the supported native dialect)"
            ),
            Self::Invariant(report) => write!(
                formatter,
                "The edit violates its declared musical invariants: {}",
                report.violations.join("; ")
            ),
            Self::BadRequest(detail) | Self::NotFound(detail) | Self::Conflict(detail) => {
                formatter.write_str(detail)
            }
            Self::Store(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for Yue2ScoreError {}

impl From<ProjectStoreError> for Yue2ScoreError {
    fn from(error: ProjectStoreError) -> Self {
        Self::Store(error)
    }
}

impl From<std::io::Error> for Yue2ScoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Store(ProjectStoreError::Io(error))
    }
}

impl From<serde_json::Error> for Yue2ScoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Store(ProjectStoreError::Json(error))
    }
}

pub(crate) fn check_abc_size(abc: &str) -> Result<(), Yue2ScoreError> {
    if abc.len() > MAX_ABC_BYTES {
        return Err(Yue2ScoreError::BadRequest(format!(
            "ABC score is {} bytes; the limit is {MAX_ABC_BYTES}",
            abc.len()
        )));
    }
    Ok(())
}

pub(crate) fn check_text(field: &str, value: &str, max: usize) -> Result<(), Yue2ScoreError> {
    let count = value.chars().count();
    if count > max {
        return Err(Yue2ScoreError::BadRequest(format!(
            "{field} is {count} characters; the limit is {max}"
        )));
    }
    Ok(())
}

/// Validate a request against the score it will render with.
pub fn validate_request(request: &SongRequest, score: &Score) -> Result<(), Yue2ScoreError> {
    check_text("style", &request.style, MAX_STYLE_CHARS)?;
    check_text("lyrics", &request.lyrics, MAX_LYRICS_CHARS)?;
    if request.seed >= 1 << 63 {
        return Err(Yue2ScoreError::BadRequest(
            "seed must be an integer in [0, 2^63)".to_owned(),
        ));
    }
    if let Some(cfg) = request.cfg_scale {
        if !cfg.is_finite() || !(0.0..=20.0).contains(&cfg) {
            return Err(Yue2ScoreError::BadRequest(
                "cfgScale must be finite and in [0, 20]".to_owned(),
            ));
        }
    }
    if request.cot == Cot::Melody && score.voices.iter().any(|voice| !voice.chords.is_empty()) {
        return Err(Yue2ScoreError::BadRequest(
            "cot=melody conditions on a chord-free score, and YuE2 does not remove chord symbols \
             automatically; apply strip_chords (or use cot=full to keep the harmony)"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Parse `abc` with the size bound applied first.
pub fn parse_bounded(abc: &str) -> Result<Score, Yue2ScoreError> {
    check_abc_size(abc)?;
    parse_score(abc).map_err(Yue2ScoreError::Notation)
}
