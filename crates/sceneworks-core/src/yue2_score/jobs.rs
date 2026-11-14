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

pub use sceneworks_options::yue2_validation::*;

/// The native score-content parser checks the bounded ABC dialect after portable option policy.
/// It is content syntax, rather than an option limit or compatibility rule.
fn validate_score_content(spec: &Yue2JobSpec) -> Result<(), Yue2JobError> {
    for (field, score) in [
        ("score", spec.score.as_deref()),
        (
            "cover.score",
            spec.cover.as_ref().and_then(|cover| cover.score.as_deref()),
        ),
    ] {
        if let Some(score) = score {
            super::parse_score(score).map_err(|error| Yue2JobError {
                code: "yue2_unsupported_notation",
                field: field.to_owned(),
                message: error.to_string(),
            })?;
        }
    }
    Ok(())
}

/// Validate a fresh request using the portable option owner and native score syntax.
pub fn validate_request(spec: &Yue2JobSpec) -> Result<(), Yue2JobError> {
    sceneworks_options::yue2_validation::validate_request(spec)?;
    validate_score_content(spec)
}

/// Fresh submissions require the explicit compute policy, while legacy jobs retain omission.
pub fn validate_new_submission(spec: &Yue2JobSpec) -> Result<(), Yue2JobError> {
    sceneworks_options::yue2_validation::validate_new_submission(spec)?;
    validate_score_content(spec)
}

/// Verify a stored resolved block before execution using the portable owner and score parser.
pub fn validate_for_execution(spec: &Yue2JobSpec) -> Result<(), Yue2JobError> {
    sceneworks_options::yue2_validation::validate_for_execution(spec)?;
    validate_score_content(spec)
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
    if let Ok(value) = crate::hashing::random_hex(16) {
        return value;
    }
    let bytes = [0u8; 16];
    // A failed OS RNG must not hand two duplicates the same run: fall back to a time-and-address
    // mix, which is unique per call in one process.
    {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let marker = &bytes as *const _ as usize as u128;
        crate::hashing::hex(&(nanos ^ marker.rotate_left(64)).to_le_bytes())
    }
}

#[cfg(test)]
#[path = "jobs_tests.rs"]
mod tests;
