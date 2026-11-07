//! Saved-cut assembly, reconciliation, editing and export.

use super::*;

/// How close two `ln(w/h)` distances must be to count as a tie when picking a timeline aspect
/// ratio. 4:3 is EXACTLY equidistant from 1:1 and 16:9 on that scale (4/3 is their geometric
/// mean), so the tie is real arithmetic, not float noise, and must be broken deliberately.
pub(super) const ASPECT_TIE_EPSILON: f64 = 1e-9;

/// The exact export job an explicit Film workspace request started or adopted. Keeping this
/// separate from the HTTP controller lets the route return as soon as the durable running record
/// exists while a background task settles that same job by id.
#[derive(Debug, Clone)]
pub struct ExplicitExportTask {
    pub job_id: String,
    pub timeline_revision: Option<u64>,
}

/// The timeline aspect ratio a take of this geometry is CREATED at: the admitted ratio closest to
/// the take's own, derived from the real geometry rather than from its orientation alone.
///
/// The timeline route admits only what `sceneworks_core::project_store::TIMELINE_ASPECT_RATIOS`
/// lists (`16:9` / `9:16` / `1:1`), and the fixture's 576x320 takes are 9:5 — so SOMETHING has to
/// be coerced, and the export pads the take into the frame (a 576x320 take in a 640-tall 16:9
/// export lands as 1138x640 with bars). Distance is measured on `ln(w/h)` — the scale-free
/// comparison, on which 2:1 is as far from 16:9 as 16:9 is from 1.0 — and a tie goes to the
/// candidate whose ORIENTATION matches the take's, because 4:3 sits exactly halfway between 1:1
/// and 16:9 on that scale and a landscape take belongs in a landscape frame.
/// [`reduced_aspect_ratio`] records what the takes actually are, so the record never states the
/// coerced ratio as a fact about the footage.
pub(super) fn aspect_ratio_for(width: u32, height: u32) -> &'static str {
    let take = (f64::from(width.max(1)) / f64::from(height.max(1))).ln();
    let orientation = width.cmp(&height);
    let mut best: Option<(&'static str, f64, bool)> = None;
    for (name, candidate_width, candidate_height) in
        sceneworks_core::project_store::TIMELINE_ASPECT_RATIOS
    {
        let distance =
            (take - (f64::from(*candidate_width) / f64::from(*candidate_height)).ln()).abs();
        let matches_orientation = candidate_width.cmp(candidate_height) == orientation;
        let better = match best {
            None => true,
            Some((_, best_distance, best_matches)) => {
                if (distance - best_distance).abs() <= ASPECT_TIE_EPSILON {
                    matches_orientation && !best_matches
                } else {
                    distance < best_distance
                }
            }
        };
        if better {
            best = Some((name, distance, matches_orientation));
        }
    }
    best.map_or("16:9", |(name, _, _)| name)
}

/// The takes' own ratio in lowest terms, e.g. `9:5` for 576x320.
pub(super) fn reduced_aspect_ratio(width: u32, height: u32) -> String {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a.max(1)
        } else {
            gcd(b, a % b)
        }
    }
    let divisor = gcd(width, height);
    format!("{}:{}", width / divisor, height / divisor)
}

pub(super) fn export_resolution_for(height: u32) -> u32 {
    crate::TIMELINE_EXPORT_RESOLUTIONS
        .iter()
        .copied()
        .find(|candidate| *candidate >= height)
        .unwrap_or_else(|| {
            crate::TIMELINE_EXPORT_RESOLUTIONS
                .iter()
                .copied()
                .max()
                .unwrap_or(1280)
        })
}

/// One shot's selected take, resolved to everything the timeline needs.
pub(super) struct SelectedTake {
    pub(super) shot_id: String,
    /// The attempt number the take belongs to — stamped into the picture item's harness block
    /// (`filmHarness.attempt`) so a later merge can tell "the selection changed" apart from "a
    /// person pointed this item somewhere else" (sc-22715).
    pub(super) attempt: u32,
    pub(super) take: TakeRecord,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) job_id: Option<String>,
}

// -------------------------------------------------------------------------------------------
// Editable picture and continuous sound (sc-22712)
// -------------------------------------------------------------------------------------------
//
// The sequence the harness assembles is a SceneWorks timeline and nothing else: a picture track of
// selected takes in cut order, a dialogue track whose clips sit against their shots, and two beds
// placed once across the whole thing. Everything the harness needs in order to re-lay that sequence
// later — which shot an item belongs to, what a dialogue line's offset means, where a bed starts —
// travels inside the timeline document under `filmHarness`, NOT in the run record.
//
// That is deliberate. The timeline is the editable artifact and the thing the export reads; a
// layout that could only be recomputed from a run record would be a layout the editor could break
// silently. The store's validators add and validate known keys and never strip unknown ones, so the
// block survives every round trip through `PUT /timelines/:id`.

/// Track ids the harness owns. The picture id is the store's own default track, so a harness
/// timeline opens in the editor with its picture where the editor expects it.
pub const PICTURE_TRACK_ID: &str = "track_main";

/// The index of the sequence's picture track: `track_main` by id, or — if the store's default
/// track ids ever change — the first `kind: "video"` track. Same rule the assembly uses to decide
/// where the takes go (sc-22710), so a re-layout can never disagree with the assembly about which
/// track holds the shots.
pub(super) fn picture_track_index(timeline: &Value) -> Option<usize> {
    let tracks = timeline.get("tracks")?.as_array()?;
    tracks
        .iter()
        .position(|track| track.get("id").and_then(Value::as_str) == Some(PICTURE_TRACK_ID))
        .or_else(|| {
            tracks
                .iter()
                .position(|track| track.get("kind").and_then(Value::as_str) == Some("video"))
        })
}
pub const DIALOGUE_TRACK_ID: &str = "track_dialogue";
pub const AMBIENCE_TRACK_ID: &str = "track_ambience";
pub const MUSIC_TRACK_ID: &str = "track_music";
pub const SFX_TRACK_PREFIX: &str = "track_sfx";

/// Where the harness's own annotation lives on a timeline item.
pub(super) const HARNESS_KEY: &str = "filmHarness";

pub(super) const ROLE_PICTURE: &str = "picture";
pub(super) const ROLE_DIALOGUE: &str = "dialogue";
pub(super) const ROLE_AMBIENCE: &str = "ambience";
pub(super) const ROLE_MUSIC: &str = "music";
pub(super) const ROLE_SFX: &str = "sfx";

/// Tag every harness-imported sound clip carries beside its role tag.
pub(super) const SOUND_TAG: &str = "film-harness-sound";

/// The store refuses `timelineEnd <= timelineStart`, so every placed item needs a floor. One frame
/// at 25 fps is a length no edit can round away.
pub(super) const MIN_ITEM_SECONDS: f64 = 0.04;

/// A sound file the run imported, as the API reported it back.
#[derive(Debug, Clone)]
pub(super) struct SoundAsset {
    pub(super) asset_id: String,
    /// Measured off the STORED wav by the import route, not claimed by the plan.
    pub(super) duration_seconds: Option<f64>,
}

pub(super) fn audio_content_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("mp3") => "audio/mpeg",
        Some("m4a" | "aac") => "audio/mp4",
        Some("flac") => "audio/flac",
        Some("ogg" | "opus") => "audio/ogg",
        _ => "audio/wav",
    }
}

/// Which generated-audio policy a shot actually runs under: its own if it declared one, the run's
/// otherwise. Both default to `mute`, which is the only default that cannot double a dialogue clip.
pub(super) fn resolved_generated_audio(
    plan: &ProductionPlan,
    shot: &film_plan::Shot,
) -> GeneratedAudio {
    shot.generated_audio.unwrap_or(plan.sound.generated_audio)
}

pub(super) fn harness_block(role: &str, run_id: &str, shot_id: Option<&str>, offset: f64) -> Value {
    json!({
        "runId": run_id,
        "role": role,
        "shotId": shot_id,
        "offsetSeconds": offset,
    })
}

/// A picture item's harness block: the shot, plus the attempt whose take the item was ALIGNED
/// with when the harness last wrote it (sc-22715). A merge compares this against the shot's
/// current `selectedAttempt` to decide whether the selection moved.
pub(super) fn picture_block(
    run_id: &str,
    shot_id: &str,
    planned_index: usize,
    attempt: u32,
    job_id: Option<&str>,
    take_asset_id: &str,
) -> Value {
    let mut block = harness_block(ROLE_PICTURE, run_id, Some(shot_id), 0.0);
    block["plannedIndex"] = json!(planned_index);
    block["attempt"] = json!(attempt);
    block["jobId"] = json!(job_id);
    block["selectedTakeAssetId"] = json!(take_asset_id);
    block
}

pub(super) fn harness_str<'a>(item: &'a Value, key: &str) -> Option<&'a str> {
    item.get(HARNESS_KEY)?.get(key)?.as_str()
}

pub(super) fn harness_f64(item: &Value, key: &str) -> f64 {
    item.get(HARNESS_KEY)
        .and_then(|block| block.get(key))
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0)
}

pub(super) fn number(item: &Value, key: &str, fallback: f64) -> f64 {
    item.get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

/// How long an item occupies the timeline, from its own source range and speed. This — not the
/// current `timelineStart`/`timelineEnd` — is what survives a trim or a reorder, which is why every
/// layout is recomputed from it rather than nudged.
pub(super) fn item_span(item: &Value) -> f64 {
    let source_in = number(item, "sourceIn", 0.0).max(0.0);
    let source_out = number(item, "sourceOut", source_in + MIN_ITEM_SECONDS);
    let speed = number(item, "speed", 1.0).clamp(0.1, 8.0);
    ((source_out - source_in) / speed).max(MIN_ITEM_SECONDS)
}

/// Positions are written UNROUNDED, and one accumulated cursor supplies both an item's end and the
/// next item's start.
///
/// Rounding here would be the bug, not the tidiness: `plan_segments` inserts a black gap wherever
/// an item does not abut the previous one, so a half-millisecond of rounding drift between two
/// shots that are supposed to cut straight together becomes a black frame in the export. Writing
/// the same `f64` to both ends makes the two exactly equal, by construction, at any precision.
pub(super) fn ms(value: f64) -> f64 {
    value
}

pub(super) fn audio_track(
    id: &str,
    name: &str,
    role: &str,
    bus: &SoundBus,
    items: Vec<Value>,
) -> Value {
    json!({
        "id": id,
        "name": name,
        "kind": "audio",
        "role": role,
        "locked": false,
        "muted": bus.muted,
        "gain": bus.gain.clamp(0.0, 4.0),
        "items": items,
    })
}

/// Merge a freshly derived harness audio track (`fresh`, with only harness-placed items) onto the
/// saved one of the same id, if the sequence already has it (sc-22715).
///
/// The saved track wins on everything a person may have changed — its fader (`gain`, `muted`),
/// its name, and every item the harness did NOT place (no `filmHarness.role`); the harness's own
/// items are the fresh ones, because they are re-derived from the plan on every pass exactly as
/// `relayout_timeline` re-places them. A fresh item that has a saved counterpart — the same
/// `filmHarness.role` and `shotId` — keeps that item's `volume`, `fadeInSeconds` and
/// `fadeOutSeconds`, which are the editor's per-item controls and were being reset to the plan's
/// values on every resume and replacement (sc-22715). With no saved track the fresh one is used
/// as is.
///
/// Dropping the saved harness items is only safe because the fresh set is derived from the shots
/// the merged PICTURE holds, not from the selection: a saved harness item whose shot is still in
/// the cut has a fresh counterpart here, and one whose shot has left the cut is meant to go. When
/// the fresh set came from `selected_takes()` instead, a shot that kept its picture item but lost
/// its selection (a rejected take) had neither — and its line vanished from the sequence.
pub(super) fn merge_harness_audio_track(existing_tracks: &[Value], fresh: Value) -> Value {
    let id = fresh.get("id").and_then(Value::as_str).unwrap_or_default();
    let Some(saved) = existing_tracks
        .iter()
        .find(|track| track.get("id").and_then(Value::as_str) == Some(id))
    else {
        return fresh;
    };
    let saved_items: Vec<Value> = saved
        .get("items")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut merged = saved.clone();
    let mut items: Vec<Value> = fresh
        .get("items")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|mut item| {
            let counterpart = saved_items.iter().find(|candidate| {
                harness_str(candidate, "role").is_some()
                    && harness_str(candidate, "role") == harness_str(&item, "role")
                    && harness_str(candidate, "shotId") == harness_str(&item, "shotId")
            });
            if let Some(counterpart) = counterpart {
                for key in ["volume", "fadeInSeconds", "fadeOutSeconds"] {
                    if let Some(kept) = counterpart.get(key) {
                        item[key] = kept.clone();
                    }
                }
            }
            item
        })
        .collect();
    items.extend(
        saved_items
            .iter()
            .filter(|item| harness_str(item, "role").is_none())
            .cloned(),
    );
    merged["items"] = Value::Array(items);
    merged
}

/// A bed track carrying exactly one item, placed once for the whole sequence.
pub(super) fn bed_track(
    track_id: &str,
    name: &str,
    role: &str,
    bed: &SoundBed,
    asset: &SoundAsset,
    run_id: &str,
) -> Value {
    let mut block = harness_block(role, run_id, None, 0.0);
    block["startSeconds"] = json!(bed.start_seconds);
    if let Some(duration) = asset
        .duration_seconds
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        // A sequence bed is placed once, but no sound role implies looping. Keep the measured source
        // bound beside the placement so relayout can extend a long bed with later picture while a
        // short effect remains its real length.
        block["sourceDurationSeconds"] = json!(duration);
    }
    let item = json!({
        "id": format!("item_{role}_{}_{}", bed.role, &run_id[4..12]),
        "trackId": track_id,
        "assetId": asset.asset_id,
        "type": "audio",
        "displayName": format!("{name} — {}", bed.role).chars().take(160).collect::<String>(),
        "sourceIn": bed.source_in_seconds,
        // Given its real span by `relayout_timeline`, which is the only place that knows how long
        // the assembled picture turned out to be.
        "sourceOut": bed.source_in_seconds + MIN_ITEM_SECONDS,
        "timelineStart": 0.0,
        "timelineEnd": MIN_ITEM_SECONDS,
        "speed": 1.0,
        "fit": "fit",
        "volume": 1.0,
        "fadeInSeconds": bed.fade_in_seconds,
        "fadeOutSeconds": bed.fade_out_seconds,
        HARNESS_KEY: block,
    });
    json!({
        "id": track_id,
        "name": name,
        "kind": "audio",
        "role": role,
        "locked": false,
        "muted": bed.muted,
        "gain": bed.gain.clamp(0.0, 4.0),
        "items": [item],
    })
}

/// Re-lay the whole sequence and return its new duration.
///
/// This is the single place that decides where anything sits, and every editing command goes
/// through it rather than adjusting positions itself — which is what makes "trim", "reorder" and
/// "replace the take" three ways of changing ONE input to the same function instead of three
/// chances to get the ripple wrong.
///
/// 1. Picture items are laid end to end from zero in `order` (or in their current order), each
///    taking exactly the span its own source range and speed imply. There are no gaps: the cuts are
///    the cuts.
/// 2. Each dialogue clip is re-placed at its shot's new start plus the offset it has always had, so
///    a line stays against its beat no matter what happened to the shots before it.
/// 3. Each bed is re-spanned from its declared start to the new end of the picture. A bed is placed
///    ONCE, so it plays straight through the cuts rather than restarting at each one.
///
/// Sound is clamped to the picture: a clip that would overrun the last frame is shortened and one
/// that would start past it is dropped. That is what keeps the saved timeline's duration and the
/// exported file's duration the same number.
pub(super) fn relayout_timeline(
    timeline: &mut Value,
    order: Option<&[String]>,
) -> Result<f64, HarnessError> {
    let picture_index = picture_track_index(timeline).ok_or_else(|| {
        HarnessError::Transport(
            "timeline has no track_main and no video track to re-lay".to_owned(),
        )
    })?;
    let tracks = timeline
        .get_mut("tracks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| HarnessError::Transport("timeline has no tracks array".to_owned()))?;

    let mut shot_starts: BTreeMap<String, f64> = BTreeMap::new();
    let mut duration = 0.0_f64;
    if let Some(track) = tracks.get_mut(picture_index) {
        let items = track
            .get_mut("items")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| HarnessError::Transport("picture track has no items".to_owned()))?;
        if let Some(order) = order {
            let present: BTreeSet<String> = items
                .iter()
                .filter_map(|item| harness_str(item, "shotId").map(str::to_owned))
                .collect();
            let requested: BTreeSet<String> = order.iter().cloned().collect();
            if present != requested {
                return Err(HarnessError::Transport(format!(
                    "the requested order names {:?} but the sequence holds {:?}; a reorder must \
                     list every shot on the picture track exactly once",
                    requested, present
                )));
            }
            items.sort_by_key(|item| {
                harness_str(item, "shotId")
                    .and_then(|shot| order.iter().position(|id| id == shot))
                    .unwrap_or(usize::MAX)
            });
        } else {
            items.sort_by(|left, right| {
                number(left, "timelineStart", 0.0).total_cmp(&number(right, "timelineStart", 0.0))
            });
        }
        let mut cursor = 0.0_f64;
        for item in items.iter_mut() {
            let shot_id = harness_str(item, "shotId").map(str::to_owned);
            let span = item_span(item);
            let start = ms(cursor);
            cursor += span;
            let end = ms(cursor);
            item["timelineStart"] = json!(start);
            item["timelineEnd"] = json!(end);
            if let Some(shot_id) = shot_id {
                shot_starts.insert(shot_id, start);
            }
        }
        duration = ms(cursor);
    }

    for track in tracks.iter_mut() {
        if track.get("kind").and_then(Value::as_str) != Some("audio") {
            continue;
        }
        let Some(items) = track.get_mut("items").and_then(Value::as_array_mut) else {
            continue;
        };
        items.retain_mut(|item| {
            let role = harness_str(item, "role").unwrap_or_default().to_owned();
            match role.as_str() {
                ROLE_DIALOGUE | ROLE_AMBIENCE | ROLE_MUSIC | ROLE_SFX => {}
                // A clip the harness did not place — the editor's own — is NOT the harness's to
                // delete. Dropping anything that merely lands near the new end (which the shared
                // rule below does, for items the harness owns and can re-place from the plan)
                // would destroy a user's own work with no diagnostic and no undo, so an unowned
                // clip is only clamped into the sequence, and dropped only when the re-layout
                // leaves it no room at all.
                _ => {
                    let start = number(item, "timelineStart", 0.0).max(0.0);
                    let end = number(item, "timelineEnd", start + item_span(item)).min(duration);
                    if start >= duration || end <= start {
                        let item_id = item
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        tracing::warn!(
                            item_id,
                            start,
                            duration,
                            "film-harness: an editor-placed audio clip starts past the end of the \
                             re-laid sequence and has been dropped"
                        );
                        return false;
                    }
                    item["timelineStart"] = json!(ms(start));
                    item["timelineEnd"] = json!(ms(end));
                    return true;
                }
            }
            let (start, end) = match role.as_str() {
                ROLE_DIALOGUE => {
                    let Some(shot_start) =
                        harness_str(item, "shotId").and_then(|shot| shot_starts.get(shot).copied())
                    else {
                        // The shot this line belongs to is no longer in the sequence.
                        return false;
                    };
                    let start = shot_start + harness_f64(item, "offsetSeconds");
                    (start, start + item_span(item))
                }
                // Sequence-level beds, the only other roles the match above admits. A bed may span
                // the sequence only while measured source remains; looping is never implicit.
                _ => {
                    let start = harness_f64(item, "startSeconds");
                    let source_in = number(item, "sourceIn", 0.0).max(0.0);
                    let source_duration = item
                        .get(HARNESS_KEY)
                        .and_then(|block| block.get("sourceDurationSeconds"))
                        .and_then(Value::as_f64)
                        .filter(|source_duration| source_duration.is_finite());
                    let end = source_duration
                        .map(|source_duration| start + (source_duration - source_in).max(0.0))
                        .unwrap_or(duration);
                    (start, end)
                }
            };
            if start >= duration - MIN_ITEM_SECONDS {
                return false;
            }
            let end = end.min(duration);
            if end <= start {
                return false;
            }
            item["timelineStart"] = json!(ms(start));
            item["timelineEnd"] = json!(ms(end));
            if matches!(role.as_str(), ROLE_AMBIENCE | ROLE_MUSIC | ROLE_SFX) {
                // A bed's source range follows its playable span, bounded by measured source.
                let source_in = number(item, "sourceIn", 0.0).max(0.0);
                let source_out = item
                    .get(HARNESS_KEY)
                    .and_then(|block| block.get("sourceDurationSeconds"))
                    .and_then(Value::as_f64)
                    .filter(|source_duration| source_duration.is_finite())
                    .map(|source_duration| {
                        let source_remaining = (source_duration - source_in).max(0.0);
                        let picture_remaining = (duration - start).max(0.0);
                        if source_remaining <= picture_remaining {
                            // Preserve the measured endpoint exactly. Deriving it by subtracting
                            // the placement after adding it turns 0.35 into
                            // 0.34999999999999964 and leaves persisted source truth needlessly
                            // dependent on the timeline start.
                            source_duration
                        } else {
                            source_in + (end - start)
                        }
                    })
                    .unwrap_or(source_in + (end - start));
                item["sourceOut"] = json!(ms(source_out));
            }
            true
        });
    }

    timeline["duration"] = json!(duration);
    Ok(duration)
}

pub(super) fn item_record(item: &Value, track_gain: f64) -> TimelineItemRecord {
    let source_in = number(item, "sourceIn", 0.0);
    TimelineItemRecord {
        shot_id: harness_str(item, "shotId").map(str::to_owned),
        item_id: item
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        asset_id: item
            .get("assetId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        timeline_start: number(item, "timelineStart", 0.0),
        timeline_end: number(item, "timelineEnd", 0.0),
        source_in,
        source_out: number(item, "sourceOut", source_in),
        gain: track_gain * number(item, "volume", 1.0),
        fade_in_seconds: number(item, "fadeInSeconds", 0.0),
        fade_out_seconds: number(item, "fadeOutSeconds", 0.0),
        generated_audio: match item.get("generatedAudio").and_then(Value::as_str) {
            Some("include") => Some(GeneratedAudio::Include),
            Some("mute") => Some(GeneratedAudio::Mute),
            _ => None,
        },
    }
}

/// Describe the assembled sequence for the run record, read back off the timeline document rather
/// than rebuilt from the plan — so the record says what was SAVED, including anything an edit
/// changed.
#[allow(clippy::too_many_arguments)]
pub(super) fn timeline_record(
    timeline_id: &str,
    name: &str,
    aspect_ratio: &str,
    source_aspect_ratio: Option<String>,
    source_width: Option<u32>,
    source_height: Option<u32>,
    fps: u32,
    duration: f64,
    timeline: &Value,
    picture_track_id: &str,
    generated_audio_default: GeneratedAudio,
    edits: Vec<TimelineEditRecord>,
) -> TimelineRecord {
    let mut tracks = Vec::new();
    let mut picture_items = Vec::new();
    for track in timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let track_id = track
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let kind = track
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("video")
            .to_owned();
        let role = track
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("sound")
            .to_owned();
        let gain = number(track, "gain", 1.0);
        let muted = track.get("muted").and_then(Value::as_bool).unwrap_or(false);
        let items: Vec<TimelineItemRecord> = track
            .get("items")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|item| item_record(item, gain))
            .collect();
        if items.is_empty() && track_id != picture_track_id {
            continue;
        }
        if track_id == picture_track_id {
            picture_items = items.clone();
        }
        tracks.push(TimelineTrackRecord {
            track_id,
            kind,
            role,
            gain,
            muted,
            items,
        });
    }
    TimelineRecord {
        revision: sceneworks_core::film_timeline::revision(timeline),
        timeline_id: timeline_id.to_owned(),
        name: name.to_owned(),
        aspect_ratio: aspect_ratio.to_owned(),
        source_aspect_ratio,
        source_width,
        source_height,
        fps,
        duration_seconds: duration,
        items: picture_items,
        tracks,
        generated_audio_default,
        edits,
    }
}

/// The saved cut owns editing state; the run continues to own attempts and human decisions.
pub(crate) fn reconcile_saved_cut(record: &mut RunRecord, saved: &Value) {
    if let Some(prior) = record.timeline.take() {
        let track_id = saved["tracks"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|t| t["kind"] == "video")
            .and_then(|t| t["id"].as_str())
            .unwrap_or("track_main");
        record.timeline = Some(timeline_record(
            &prior.timeline_id,
            &prior.name,
            &prior.aspect_ratio,
            prior.source_aspect_ratio,
            prior.source_width,
            prior.source_height,
            prior.fps,
            number(saved, "duration", 0.0),
            saved,
            track_id,
            prior.generated_audio_default,
            prior.edits,
        ));
    }
    if let Some(export) = record.export.as_mut() {
        export.stale =
            export.timeline_revision != Some(sceneworks_core::film_timeline::revision(saved));
    }
}

/// Dispatch the `timeline_export` job and wait for it under the run's budgets.
#[allow(clippy::too_many_arguments)]
pub(super) async fn export_timeline(
    client: &Client<'_>,
    project_id: &str,
    timeline_id: &str,
    resolution: u32,
    fps: u32,
    bounds: PollBounds,
    limits: &film_plan::PlanLimits,
) -> Result<(ExportRecord, PollStop), HarnessError> {
    let export_job = client
        .expect_ok(
            "POST",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
            Some(json!({
                "resolution": resolution,
                "fps": fps,
                "requestedGpu": "auto",
            })),
        )
        .await?;
    let export_job_id = export_job
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| HarnessError::Transport(format!("export response has no id: {export_job}")))?
        .to_owned();
    let settle_grace = bounds.settle_grace;
    let (view, poll_stop) = client.wait_for_job(&export_job_id, bounds).await?;
    let asset_id = view
        .result
        .get("assetIds")
        .and_then(Value::as_array)
        .and_then(|ids| ids.first())
        .and_then(Value::as_str)
        .map(str::to_owned);
    let render_path = view
        .result
        .get("renderPath")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let status = match poll_stop {
        PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
        PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
        PollStop::Operator => "canceled_by_operator".to_owned(),
    };
    let ok = status == "completed" && asset_id.is_some();
    Ok((
        ExportRecord {
            timeline_revision: export_job["payload"]["timelineRevision"].as_u64(),
            job_id: export_job_id,
            status,
            // This export just ran against the timeline as it stands, so it matches the selected
            // takes by construction. Only a LATER take change makes an export stale (sc-22711).
            stale: false,
            asset_id,
            render_path,
            dropped_audio_layers: dropped_audio_layers(&view.result),
            error: (!ok).then(|| match poll_stop {
                PollStop::Terminal => view.failure_text(),
                PollStop::AssetsUnsettled => format!(
                    "the export job reached {} but its assets never settled within {:.0}s",
                    view.status,
                    settle_grace.as_secs_f64()
                ),
                PollStop::Operator => "canceled by operator during the export".to_owned(),
                PollStop::ShotBudget => format!(
                    "export exceeded the per-job budget of {}s",
                    limits.max_shot_seconds
                ),
                PollStop::RunBudget => format!(
                    "run exceeded its budget of {}s during the export",
                    limits.max_run_seconds
                ),
            }),
        },
        poll_stop,
    ))
}

/// Start (or adopt) the explicit export for a Film workspace run and durably record its job id.
///
/// This operation never renders shots and never runs implicitly at the end of shot generation.
/// It snapshots the currently saved timeline through the ordinary timeline-export route, marks
/// an older export superseded, and returns once `run.json` names the exact in-flight job.
pub async fn start_explicit_export(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
) -> Result<(RunRecord, ExplicitExportTask), HarnessError> {
    let mut record = read_run_record(&options.out_dir)?;
    let project_id = record
        .project_id
        .clone()
        .ok_or_else(|| HarnessError::Refused("run record has no project id".to_owned()))?;
    let timeline_id = record
        .timeline
        .as_ref()
        .map(|timeline| timeline.timeline_id.clone())
        .ok_or_else(|| {
            HarnessError::Refused("run has no assembled timeline to export".to_owned())
        })?;
    let client = Client {
        transport,
        control: &options.control,
    };
    let saved = client
        .expect_ok(
            "GET",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            None,
        )
        .await?;
    reconcile_saved_cut(&mut record, &saved);

    if let Some(export) = record
        .export
        .as_ref()
        .filter(|export| export.status == "running")
    {
        let task = ExplicitExportTask {
            job_id: export.job_id.clone(),
            timeline_revision: export.timeline_revision,
        };
        return Ok((record, task));
    }

    if record.export_pending.is_none() {
        if let Some(previous) = record.export.take() {
            if !record.superseded_export_job_ids.contains(&previous.job_id) {
                record
                    .superseded_export_job_ids
                    .push(previous.job_id.clone());
            }
            record.export_pending = Some(ExportPending {
                requested_at: utc_now(),
                supersedes: Some(previous.job_id),
            });
        } else {
            record.export_pending = Some(ExportPending {
                requested_at: utc_now(),
                supersedes: None,
            });
        }
        persist_record(
            &record,
            &options.out_dir,
            &options.out_dir.join("plan.json"),
            &options.out_dir.join("references.json"),
        )?;
    }

    let requested_at = record
        .export_pending
        .as_ref()
        .map(|pending| pending.requested_at.as_str())
        .unwrap_or_default();
    let existing = client
        .find_export_job(
            &project_id,
            &timeline_id,
            &record.superseded_export_job_ids,
            requested_at,
        )
        .await?;
    let response = match existing {
        Some(job_id) => {
            client
                .expect_ok("GET", &format!("/api/v1/jobs/{job_id}"), None)
                .await?
        }
        None => {
            if options.control.is_canceled() {
                return record_action(&options.out_dir, "export", None, async {
                    Err(HarnessError::Canceled(
                        "export canceled before dispatch; the saved cut and pending export identity are retained".to_owned(),
                    ))
                }).await;
            }
            let height = saved
                .get("height")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .or_else(|| {
                    record
                        .timeline
                        .as_ref()
                        .and_then(|timeline| timeline.source_height)
                })
                .unwrap_or(720);
            let fps = saved
                .get("fps")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .or_else(|| record.timeline.as_ref().map(|timeline| timeline.fps))
                .unwrap_or(24);
            client
                .expect_ok(
                    "POST",
                    &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
                    Some(json!({
                        "resolution": export_resolution_for(height),
                        "fps": fps,
                        "requestedGpu": "auto",
                    })),
                )
                .await?
        }
    };
    let job_id = response
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| HarnessError::Transport(format!("export response has no id: {response}")))?
        .to_owned();
    let timeline_revision = response
        .get("payload")
        .and_then(|payload| payload.get("timelineRevision"))
        .and_then(Value::as_u64);
    record.export = Some(ExportRecord {
        timeline_revision,
        job_id: job_id.clone(),
        status: "running".to_owned(),
        stale: false,
        asset_id: None,
        render_path: None,
        error: None,
        dropped_audio_layers: Vec::new(),
    });
    record.export_pending = None;
    persist_record(
        &record,
        &options.out_dir,
        &options.out_dir.join("plan.json"),
        &options.out_dir.join("references.json"),
    )?;
    Ok((
        record,
        ExplicitExportTask {
            job_id,
            timeline_revision,
        },
    ))
}

/// Settle an explicit Film workspace export and copy its output, failure, and dropped-audio report
/// into the durable run record. If a later request superseded this task, its result is left alone.
pub async fn finish_explicit_export(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
    task: &ExplicitExportTask,
) -> Result<RunRecord, HarnessError> {
    let client = Client {
        transport,
        control: &options.control,
    };
    let started = Instant::now();
    let budget = Duration::from_secs(options_limit_seconds(&options.out_dir)?);
    let settle_grace = ASSET_SETTLE_GRACE.min(budget);
    let (view, poll_stop) = client
        .wait_for_job(
            &task.job_id,
            PollBounds {
                shot_deadline: started + budget,
                run_deadline: None,
                poll_interval: options.poll_interval,
                cancel_grace: CANCEL_GRACE.min(budget),
                settle_grace,
            },
        )
        .await?;
    let mut record = read_run_record(&options.out_dir)?;
    if record.export.as_ref().map(|export| export.job_id.as_str()) != Some(task.job_id.as_str()) {
        return Ok(record);
    }
    let asset_id = view
        .result
        .get("assetIds")
        .and_then(Value::as_array)
        .and_then(|ids| ids.first())
        .and_then(Value::as_str)
        .map(str::to_owned);
    let render_path = view
        .result
        .get("renderPath")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let status = match poll_stop {
        PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
        PollStop::Operator => "canceled_by_operator".to_owned(),
        PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
    };
    let export_ok = status == "completed" && asset_id.is_some();
    record.export = Some(ExportRecord {
        timeline_revision: task.timeline_revision,
        job_id: task.job_id.clone(),
        status,
        stale: false,
        asset_id,
        render_path,
        dropped_audio_layers: dropped_audio_layers(&view.result),
        error: (!export_ok).then(|| match poll_stop {
            PollStop::Terminal => view.failure_text(),
            PollStop::AssetsUnsettled => format!(
                "the export job reached {} but its assets never settled within {:.0}s",
                view.status,
                settle_grace.as_secs_f64()
            ),
            PollStop::Operator => "canceled by operator during the export".to_owned(),
            PollStop::ShotBudget | PollStop::RunBudget => format!(
                "export exceeded the per-job budget of {}s",
                budget.as_secs()
            ),
        }),
    });
    persist_record(
        &record,
        &options.out_dir,
        &options.out_dir.join("plan.json"),
        &options.out_dir.join("references.json"),
    )?;
    Ok(record)
}

pub(super) fn options_limit_seconds(out_dir: &Path) -> Result<u64, HarnessError> {
    Ok(read_run_record(out_dir)?.limits.max_shot_seconds)
}

/// One change to an assembled sequence.
///
/// Each of these changes exactly one input to [`relayout_timeline`] and then lets it recompute
/// every position, which is why a trim ripples, a reorder keeps every line against its beat, and a
/// replaced take re-times the sequence around its new length — without any of the three knowing
/// about the other two.
#[derive(Debug, Clone)]
pub enum TimelineEdit {
    /// Change a shot's source range. `None` leaves that end where it is.
    Trim {
        shot_id: String,
        source_in: Option<f64>,
        source_out: Option<f64>,
    },
    /// Put the picture track in this order. Must name every shot on it, exactly once.
    Reorder { shot_ids: Vec<String> },
    /// Point a shot at a different take. The asset must already exist in the project — choosing it
    /// is sc-22711's job, placing it is this one's.
    SwapTake { shot_id: String, asset_id: String },
}

impl TimelineEdit {
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Trim { .. } => "trim",
            Self::Reorder { .. } => "reorder",
            Self::SwapTake { .. } => "swap_take",
        }
    }
}

/// Everything an edit needs beyond the transport.
#[derive(Debug, Clone)]
pub struct EditOptions {
    /// The `run.json` written by [`run`]. It names the project and the timeline, and it is
    /// rewritten in place with the edited sequence.
    pub run_record_path: PathBuf,
    /// Re-export the MP4 after the edit.
    pub export: bool,
    pub poll_interval: Duration,
}

/// Apply one edit to a run's assembled sequence: change the timeline, save it, rewrite the run
/// record, and optionally re-export.
///
/// Shot -> asset links are never rebuilt from the plan here. The picture item already carries its
/// shot id and its version history, so a trim or a reorder moves the item that is already bound to
/// the take a human chose, and a replacement appends to that history instead of overwriting it.
pub async fn edit_timeline(
    transport: &dyn ApiTransport,
    options: &EditOptions,
    edit: TimelineEdit,
) -> Result<RunRecord, HarnessError> {
    let out_dir = options
        .run_record_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let lease =
        ControllerLease::acquire(&out_dir, format!("edit_{}", uuid::Uuid::new_v4().simple()))?;
    edit_timeline_with_lease(transport, options, edit, lease).await
}

pub(super) async fn edit_timeline_with_lease(
    transport: &dyn ApiTransport,
    options: &EditOptions,
    edit: TimelineEdit,
    _lease: ControllerLease,
) -> Result<RunRecord, HarnessError> {
    // An edit re-exports at most one job and is interruptible through the same control a run uses;
    // nothing here dispatches a render, so the default (never canceled) is the whole contract.
    let control = RunControl::new();
    let client = Client {
        transport,
        control: &control,
    };
    let text = std::fs::read_to_string(&options.run_record_path)?;
    let mut record: RunRecord = serde_json::from_str(&text).map_err(|error| {
        HarnessError::Io(format!(
            "{} is not a film-harness run record: {error}",
            options.run_record_path.display()
        ))
    })?;
    let out_dir = options
        .run_record_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let (_, plan_bytes) = read_source(
        Path::new(&record.plan.path),
        &out_dir.join("plan.json"),
        "plan",
    )?;
    let plan_sha256 = sha256_hex(&plan_bytes);
    if plan_sha256 != record.plan.sha256 {
        return Err(HarnessError::Refused(format!(
            "the plan changed since run {} started (recorded {}, found {}); refusing to change a take against different dependencies",
            record.run_id, record.plan.sha256, plan_sha256
        )));
    }
    let pinned_plan = film_plan::parse_plan(std::str::from_utf8(&plan_bytes).unwrap_or_default())
        .map_err(|error| HarnessError::Refused(format!("plan: {error}")))?;
    let project_id = record
        .project_id
        .clone()
        .ok_or_else(|| HarnessError::Io("run record has no project id".to_owned()))?;
    let existing = record.timeline.clone().ok_or_else(|| {
        HarnessError::Io("run record has no assembled timeline to edit".to_owned())
    })?;
    let timeline_id = existing.timeline_id.clone();

    let mut timeline = client
        .expect_ok(
            "GET",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            None,
        )
        .await?;

    let (order, detail, swapped_attempt) = match &edit {
        TimelineEdit::Reorder { shot_ids } => (Some(shot_ids.clone()), shot_ids.join(" -> "), None),
        TimelineEdit::Trim {
            shot_id,
            source_in,
            source_out,
        } => {
            // Measure the take FIRST, the way `SwapTake` does. An out point past the end of the
            // media is silent otherwise: `relayout_timeline` writes a `timelineEnd` longer than the
            // file, `render_item_segment` reports the DECLARED duration while `-t` yields a short
            // segment, and the picture then comes out shorter than the sequence it was saved from —
            // the same drift a crossfade used to cause, plus a wrong duration in the sidecar.
            let take_id = picture_item_mut(&mut timeline, shot_id)?
                .get("assetId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let take_seconds = if take_id.is_empty() {
                None
            } else {
                client
                    .expect_ok(
                        "GET",
                        &format!("/api/v1/projects/{project_id}/assets/{take_id}"),
                        None,
                    )
                    .await?
                    .get("file")
                    .and_then(|file| file.get("duration"))
                    .and_then(Value::as_f64)
                    .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            };
            let item = picture_item_mut(&mut timeline, shot_id)?;
            let current_in = number(item, "sourceIn", 0.0);
            let current_out = number(item, "sourceOut", current_in + MIN_ITEM_SECONDS);
            let new_in = source_in.unwrap_or(current_in).max(0.0);
            let requested_out = source_out.unwrap_or(current_out);
            // Clamp rather than refuse when there is still a usable range left: "keep everything
            // from 1s on" is a reasonable thing to ask of a take whose length the caller does not
            // know. Only a range that lands entirely past the end of the media is an error, and it
            // says how long the take actually is.
            let new_out = match take_seconds {
                Some(seconds) if requested_out > seconds => {
                    if new_in + MIN_ITEM_SECONDS >= seconds {
                        return Err(HarnessError::Io(format!(
                            "trim of {shot_id} asks for {new_in:.3}..{requested_out:.3} but its \
                             take is only {seconds:.3}s long; the in point must be at least \
                             {MIN_ITEM_SECONDS}s before the end of the take"
                        )));
                    }
                    seconds
                }
                _ => requested_out,
            };
            if !new_in.is_finite() || !new_out.is_finite() || new_out <= new_in + MIN_ITEM_SECONDS {
                return Err(HarnessError::Io(format!(
                    "trim of {shot_id} would leave a source range of {new_in}..{new_out}; the out \
                     point must be at least {MIN_ITEM_SECONDS}s after the in point"
                )));
            }
            item["sourceIn"] = json!(ms(new_in));
            item["sourceOut"] = json!(ms(new_out));
            (
                None,
                format!("{shot_id} source range {new_in:.3}..{new_out:.3}"),
                None,
            )
        }
        TimelineEdit::SwapTake { shot_id, asset_id } => {
            // Measure the replacement before touching the timeline, so a bad asset id fails before
            // the sequence is half-edited.
            let asset = client
                .expect_ok(
                    "GET",
                    &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
                    None,
                )
                .await?;
            let duration = asset
                .get("file")
                .and_then(|file| file.get("duration"))
                .and_then(Value::as_f64)
                .filter(|seconds| *seconds > 0.0);
            // The attempt this asset belongs to, when it is one of the shot's OWN takes. The
            // selection follows it below, and so does the item's aligned-attempt stamp — a later
            // merge then reads "selection N, item aligned with N" and leaves the swap alone. A
            // swap onto a FOREIGN asset has no generation attempt to select and leaves the stamp
            // as it is. The saved timeline owns that cut choice while selectedAttempt continues
            // to name generation provenance; a later generated replacement has a new attempt and
            // can therefore be merged without erasing the edit (sc-22715).
            let own_attempt = record.shot(shot_id).and_then(|shot| {
                shot.attempts
                    .iter()
                    .find(|attempt| {
                        attempt
                            .take
                            .as_ref()
                            .is_some_and(|take| &take.asset_id == asset_id)
                    })
                    .map(|attempt| attempt.attempt)
            });
            let item = picture_item_mut(&mut timeline, shot_id)?;
            let previous = item
                .get("assetId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let span = duration.unwrap_or_else(|| item_span(item));
            let source_in = number(item, "sourceIn", 0.0);
            let source_out = number(item, "sourceOut", source_in + span);
            if duration.is_some_and(|length| source_out > length + 0.000001) {
                return Err(HarnessError::Refused(format!(
                    "timeline_replacement_trim_conflict: {shot_id} is trimmed to {source_in}..{source_out}, but the new take is {span}s. Keep the current take, or explicitly trim this item to a fitting range (0..{span} for a reset) and retry the swap."
                )));
            }
            item["assetId"] = json!(asset_id);
            item["currentVersionAssetId"] = json!(asset_id);
            if let Some(attempt) = own_attempt {
                item[HARNESS_KEY]["attempt"] = json!(attempt);
            }
            if let Some(history) = item.get_mut("versionHistory").and_then(Value::as_array_mut) {
                history.push(json!({
                    "assetId": asset_id,
                    "source": "replacement",
                    "createdAt": utc_now(),
                    "note": format!("film-harness take replacement for {shot_id}"),
                }));
            }
            if let Some(versions) = item
                .get_mut("versionAssetIds")
                .and_then(Value::as_array_mut)
            {
                if !versions.iter().any(|value| value == &json!(asset_id)) {
                    versions.push(json!(asset_id));
                }
            }
            (
                None,
                format!("{shot_id} take {previous} -> {asset_id} ({span:.3}s)"),
                own_attempt,
            )
        }
    };

    let duration = if matches!(edit, TimelineEdit::SwapTake { .. }) {
        number(&timeline, "duration", 0.0)
    } else {
        relayout_timeline(&mut timeline, order.as_deref())?
    };
    let path = format!("/api/v1/projects/{project_id}/timelines/{timeline_id}");
    let expected_revision = sceneworks_core::film_timeline::revision(&timeline);
    let response = client
        .json(
            "PUT",
            &path,
            Some(json!({
                "timeline": timeline, "expectedRevision": expected_revision,
            })),
        )
        .await?;
    if response.status == 409 && response.body["code"] == "timeline_revision_conflict" {
        let latest = client.expect_ok("GET", &path, None).await?;
        return Err(HarnessError::Refused(format!(
            "timeline_revision_conflict: {} changed from revision {} to {} during {}. No edit was saved. Review the current cut and retry the command to apply it to that version.",
            timeline_id, expected_revision, sceneworks_core::film_timeline::revision(&latest), edit.kind()
        )));
    }
    if !(200..300).contains(&response.status) {
        return Err(HarnessError::Api {
            method: "PUT",
            path,
            status: response.status,
            detail: api_detail(&response.body),
        });
    }
    let saved = response.body;

    let mut edits = existing.edits.clone();
    edits.push(TimelineEditRecord {
        kind: edit.kind().to_owned(),
        applied_at: utc_now(),
        detail: detail.clone(),
        duration_seconds: duration,
    });
    // An edit is a HUMAN decision about this run, so it belongs in the run's decision log next to
    // the generation-side ones (sc-22711 schema 2). `timeline.edits` records what the sequence now
    // IS; the decision log records that somebody changed it, and when — the two are not the same
    // question, and only the decision log is read in run order beside the takes.
    record.decisions.push(ProductionDecision {
        at: utc_now(),
        action: edit.kind().to_owned(),
        shot_id: match &edit {
            TimelineEdit::Trim { shot_id, .. } | TimelineEdit::SwapTake { shot_id, .. } => {
                Some(shot_id.clone())
            }
            TimelineEdit::Reorder { .. } => None,
        },
        detail,
    });
    // The saved cut owns what is on screen. A swap therefore moves the generation selection only
    // when the chosen asset is one of this shot's attempts; a foreign asset has no generation
    // attempt, so selectedAttempt continues to record generation provenance. Either way an earlier
    // accept/reject no longer describes the take in the cut, so its standing
    // `humanDecision` is cleared while the append-only decision log keeps the provenance. A person
    // may explicitly reinstate an earlier rejected take; the old rejection remains in the decision
    // log, while the attempt becomes a live candidate again and still requires a fresh acceptance.
    if let TimelineEdit::SwapTake { shot_id, .. } = &edit {
        let previous = record.shot(shot_id).and_then(|shot| shot.selected_attempt);
        if let Some(shot) = record.shot_mut(shot_id) {
            shot.human_decision = None;
            if let Some(attempt) = swapped_attempt {
                shot.selected_attempt = Some(attempt);
                if let Some(chosen) = shot
                    .attempts
                    .iter_mut()
                    .find(|candidate| candidate.attempt == attempt)
                {
                    chosen.rejection = None;
                }
            }
        }
        if swapped_attempt.is_none() || previous != swapped_attempt {
            flag_dependents(
                &mut record,
                &pinned_plan,
                shot_id,
                "its take selection was changed by hand",
            );
        }
    }
    let picture_track_id = picture_track_index(&saved)
        .and_then(|index| saved["tracks"][index]["id"].as_str())
        .unwrap_or(PICTURE_TRACK_ID)
        .to_owned();
    // The take geometry the sequence was sized from does not change when the picture is re-cut, so
    // it is carried over from the record rather than re-derived (sc-22710).
    record.timeline = Some(timeline_record(
        &timeline_id,
        &existing.name,
        &existing.aspect_ratio,
        existing.source_aspect_ratio.clone(),
        existing.source_width,
        existing.source_height,
        existing.fps,
        saved
            .get("duration")
            .and_then(Value::as_f64)
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .unwrap_or(duration),
        &saved,
        &picture_track_id,
        existing.generated_audio_default,
        edits,
    ));

    if options.export {
        let tallest = saved
            .get("height")
            .and_then(Value::as_u64)
            .unwrap_or(u64::from(DEFAULT_EXPORT_HEIGHT)) as u32;
        let shot_budget = Duration::from_secs(record.limits.max_shot_seconds);
        // Human-requested work, like a replacement (sc-22715): bounded by the export's own
        // per-job budget, charged to `humanRequestedElapsedSeconds`, and never classified
        // `run_budget` — an edit's re-export must not spend the run's automatic wall-clock.
        let export_started = Instant::now();
        let (export_record, _) = export_timeline(
            &client,
            &project_id,
            &timeline_id,
            export_resolution_for(tallest),
            existing.fps,
            PollBounds {
                shot_deadline: export_started + shot_budget,
                run_deadline: None,
                poll_interval: options.poll_interval,
                cancel_grace: CANCEL_GRACE.min(shot_budget),
                settle_grace: ASSET_SETTLE_GRACE.min(shot_budget),
            },
            &record.limits,
        )
        .await?;
        record.human_requested_elapsed_seconds += export_started.elapsed().as_secs_f64();
        let export_ok = export_record.status == "completed" && export_record.asset_id.is_some();
        record.export = Some(export_record);
        // The run's `outcome` / `stop` describe the RUN, and an edit is not a run (sc-22715): a
        // successful re-export leaves them exactly as they were — a `canceled` or an
        // `attempts_exhausted` stop is still true after a trim — except an `export_failed` stop,
        // which is about precisely the export this one just replaced. A failed re-export is
        // recorded as one, resumable, unless the run already carries a terminal stop of its own.
        if export_ok {
            if record
                .stop
                .as_ref()
                .is_some_and(|stop| stop.reason == "export_failed")
            {
                record.outcome = RunOutcome::Completed;
                record.stop = None;
            }
        } else if record.stop.as_ref().is_none_or(|stop| stop.resumable) {
            record.outcome = RunOutcome::Failed;
            record.stop = Some(RunStop {
                reason: "export_failed".to_owned(),
                detail: "the edit's re-export did not complete; `film-harness resume` retries the \
                         export"
                    .to_owned(),
                resumable: true,
            });
        }
    } else if let Some(export) = &mut record.export {
        // The sequence just changed under the MP4 that was exported from it, so that MP4 no longer
        // matches the run. 22711's rule holds here exactly as it does for a re-rendered take: the
        // export is FLAGGED, never silently re-run — re-exporting stays an explicit `--export`.
        export.stale = true;
    }

    // The same write every controller makes (sc-22715): atomic, and mirrored to
    // `<project>/film-harness/<run_id>/run.json` — an edit rewrote the record with a plain
    // `fs::write` and left the project's copy describing a sequence that no longer existed.
    persist_record(
        &record,
        &out_dir,
        Path::new(&record.plan.path),
        Path::new(&record.reference_pack.path),
    )?;
    Ok(record)
}

/// Fallback export height when a timeline document does not carry one.
pub(super) const DEFAULT_EXPORT_HEIGHT: u32 = 720;

pub(super) fn picture_item_mut<'a>(
    timeline: &'a mut Value,
    shot_id: &str,
) -> Result<&'a mut Value, HarnessError> {
    let index = picture_track_index(timeline);
    index
        .and_then(|index| {
            timeline
                .get_mut("tracks")
                .and_then(Value::as_array_mut)
                .and_then(|tracks| tracks.get_mut(index))
        })
        .and_then(|track| track.get_mut("items").and_then(Value::as_array_mut))
        .and_then(|items| {
            items
                .iter_mut()
                .find(|item| harness_str(item, "shotId") == Some(shot_id))
        })
        .ok_or_else(|| {
            HarnessError::Io(format!(
                "shot {shot_id:?} is not on the sequence's picture track"
            ))
        })
}
impl Session<'_> {
    // -----------------------------------------------------------------------------------------
    // Timeline and export
    // -----------------------------------------------------------------------------------------

    /// The selected take of every selected shot, in plan order — what the timeline is built from.
    pub(super) fn selected_takes(&self) -> Vec<SelectedTake> {
        self.record
            .shots
            .iter()
            .filter(|shot| self.is_selected(&shot.shot_id))
            .filter_map(|shot| {
                let attempt = shot.selected()?;
                Some(SelectedTake {
                    shot_id: shot.shot_id.clone(),
                    attempt: attempt.attempt,
                    take: attempt.take.clone()?,
                    width: shot.intended.width,
                    height: shot.intended.height,
                    job_id: attempt.job_id.clone(),
                })
            })
            .collect()
    }

    /// The tallest selected take, which fixes the export ladder rung the render uses.
    pub(super) fn tallest_selected(&self) -> u32 {
        self.selected_takes()
            .iter()
            .map(|selected| selected.height)
            .max()
            .unwrap_or(0)
    }

    /// The id of this project's timeline named `name`, if it already has one.
    pub(super) async fn find_timeline_by_name(
        &self,
        project_id: &str,
        name: &str,
    ) -> Result<Option<String>, HarnessError> {
        Ok(self
            .client
            .expect_ok(
                "GET",
                &format!("/api/v1/projects/{project_id}/timelines"),
                None,
            )
            .await?
            .as_array()
            .into_iter()
            .flatten()
            .find(|timeline| timeline.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|timeline| timeline.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned))
    }

    /// Create (or MERGE INTO) the run's timeline from the selected takes. Pure API writes, no job:
    /// re-running it after a take changed rewrites that one item and leaves the rest — order,
    /// source ranges, version histories, editor-placed items, bus faders — exactly as the saved
    /// sequence holds them (sc-22715).
    pub(super) async fn assemble_timeline(&mut self) -> Result<bool, HarnessError> {
        let project_id = self.project_id()?;
        let takes = self.selected_takes();
        if takes.is_empty() {
            return Ok(false);
        }
        let (first_width, first_height) = (takes[0].width, takes[0].height);
        let aspect_ratio = aspect_ratio_for(first_width, first_height);
        let existing_id = self
            .record
            .timeline
            .as_ref()
            .map(|timeline| timeline.timeline_id.clone());
        let timeline_name = self
            .record
            .timeline
            .as_ref()
            .map(|timeline| timeline.name.clone())
            .unwrap_or_else(|| format!("{} ({})", self.plan.title, self.record.run_id));
        let mut timeline = match &existing_id {
            Some(id) => {
                self.client
                    .expect_ok(
                        "GET",
                        &format!("/api/v1/projects/{project_id}/timelines/{id}"),
                        None,
                    )
                    .await?
            }
            // Adopt by identity, exactly as the project is. `POST /timelines` ALWAYS creates a new
            // row, and the record's `timeline` is only written after the PUT lands — so a
            // controller killed in that window leaves a timeline nothing names, and a resume that
            // just POSTed again would leave the project holding two identical sequences with the
            // run record pointing at one of them. The name carries the run id, which is what makes
            // the lookup exact (sc-22711).
            None => match self
                .find_timeline_by_name(&project_id, &timeline_name)
                .await?
            {
                Some(id) => {
                    self.client
                        .expect_ok(
                            "GET",
                            &format!("/api/v1/projects/{project_id}/timelines/{id}"),
                            None,
                        )
                        .await?
                }
                None => {
                    self.client
                        .expect_ok(
                            "POST",
                            &format!("/api/v1/projects/{project_id}/timelines"),
                            Some(json!({
                                "name": timeline_name,
                                "aspectRatio": aspect_ratio,
                                "fps": self.fps
                            })),
                        )
                        .await?
                }
            },
        };
        let timeline_id = timeline
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessError::Transport(format!("timeline response has no id: {timeline}"))
            })?
            .to_owned();
        // The main video track, by id or — if the project store's default track ids ever change —
        // by kind. Writing the items nowhere and saving an EMPTY timeline while the record still
        // listed every shot is the one outcome that must not happen.
        let tracks = timeline
            .get("tracks")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let track_index = tracks
            .iter()
            .position(|track| track.get("id").and_then(Value::as_str) == Some("track_main"))
            .or_else(|| {
                tracks
                    .iter()
                    .position(|track| track.get("kind").and_then(Value::as_str) == Some("video"))
            })
            .ok_or_else(|| {
                HarnessError::Transport(format!(
                    "timeline {timeline_id} has no track_main and no video track to hold the \
                     rendered takes: {timeline}"
                ))
            })?;
        let track_id = tracks[track_index]
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("track_main")
            .to_owned();
        // MERGE, never rebuild (sc-22715). The first assembly of a run creates every picture item;
        // every later pass — a resume, a replacement — starts from the SAVED sequence and changes
        // only what the selection changed. Rebuilding from `selected_takes()` (the sc-22710 shape)
        // silently undid every `trim` / `reorder` / `swap-take` a person had applied through
        // `edit_timeline`, while `timeline.edits` and the decision log went on recording them: a
        // record that said "trimmed, reordered" over a sequence that was neither.
        //
        // What is kept: every existing picture item — its order, its source range, its
        // `versionHistory`, its `generatedAudio`, and an item the harness did not place at all.
        // What changes: an item whose shot now selects a DIFFERENT take is pointed at it (source
        // range reset to the new take's length, one `replacement` entry appended to its history),
        // and a shot with no item yet gets one. Positions are then handed to `relayout_timeline`,
        // the single place that knows how the sequence is timed, in the sequence's EXISTING order
        // with any new shots after it.
        let existing_tracks: Vec<Value> = tracks.to_vec();
        let mut items: Vec<Value> = existing_tracks[track_index]
            .get("items")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        items.sort_by(|left, right| {
            number(left, "timelineStart", 0.0).total_cmp(&number(right, "timelineStart", 0.0))
        });
        for SelectedTake {
            shot_id,
            attempt,
            take,
            job_id,
            ..
        } in &takes
        {
            let shot = self
                .plan
                .shots
                .iter()
                .find(|shot| &shot.id == shot_id)
                .expect("selected takes come from the plan");
            let planned_index = self
                .plan
                .shots
                .iter()
                .position(|planned| planned.id == shot.id)
                .expect("selected takes come from the plan");
            let length = take
                .encoded_duration_seconds
                .filter(|seconds| *seconds > 0.0)
                .unwrap_or(shot.target_duration_seconds);
            let existing = items.iter_mut().find(|item| {
                harness_str(item, "shotId") == Some(shot_id.as_str())
                    && harness_str(item, "runId") == Some(self.record.run_id.as_str())
            });
            // "Did the SELECTION change since this item was last aligned with it?" — not "does
            // the item show the selected asset?". The two differ exactly when a person used
            // `swap-take` to point the item at a foreign asset (an imported clip): the selection
            // is unchanged, the item deliberately is not, and rewriting it would revert their
            // edit. The item's harness block records the attempt it was aligned with; an item
            // from before that stamp existed falls back to the asset comparison.
            let selection_changed = |item: &Value| match item
                .get(HARNESS_KEY)
                .and_then(|block| block.get("attempt"))
                .and_then(Value::as_u64)
            {
                Some(aligned) => aligned != u64::from(*attempt),
                None => item.get("assetId").and_then(Value::as_str) != Some(&take.asset_id),
            };
            match existing {
                Some(item) if !selection_changed(item) => {
                    // The sequence already reflects this selection: nothing about the item moves.
                }
                Some(item) => {
                    // A replacement landed for this shot. Only THIS item changes, and its history
                    // grows rather than restarting — the take that was there stays addressable.
                    item["assetId"] = json!(take.asset_id);
                    item["currentVersionAssetId"] = json!(take.asset_id);
                    item["sourceIn"] = json!(0.0);
                    item["sourceOut"] = json!(length);
                    item[HARNESS_KEY]["attempt"] = json!(attempt);
                    item[HARNESS_KEY]["plannedIndex"] = json!(planned_index);
                    item[HARNESS_KEY]["jobId"] = json!(job_id);
                    item[HARNESS_KEY]["selectedTakeAssetId"] = json!(take.asset_id);
                    let entry = json!({
                        "assetId": take.asset_id,
                        "source": "replacement",
                        "jobId": job_id,
                        "createdAt": utc_now(),
                        "note": format!("film-harness {} replaced shot {}", self.record.run_id, shot.id),
                    });
                    match item.get_mut("versionHistory").and_then(Value::as_array_mut) {
                        Some(history) => history.push(entry),
                        None => item["versionHistory"] = json!([entry]),
                    }
                    if let Some(versions) = item
                        .get_mut("versionAssetIds")
                        .and_then(Value::as_array_mut)
                    {
                        if !versions.iter().any(|value| value == &json!(take.asset_id)) {
                            versions.push(json!(take.asset_id));
                        }
                    }
                }
                None => {
                    let item_id = format!(
                        "item_{}_{}",
                        shot.id.to_ascii_lowercase(),
                        &self.record.run_id[4..12]
                    );
                    items.push(json!({
                        "id": item_id,
                        "trackId": track_id,
                        "assetId": take.asset_id,
                        "type": "video",
                        "displayName": format!("{} — {}", shot.id, shot.beat).chars().take(160).collect::<String>(),
                        "sourceIn": 0.0,
                        "sourceOut": length,
                        "timelineStart": 0.0,
                        "timelineEnd": length,
                        "speed": 1.0,
                        "fit": "fit",
                        "volume": 1.0,
                        // The RESOLVED policy, written into the timeline itself so the export
                        // obeys the saved document rather than re-deriving anything from the plan
                        // (sc-22712).
                        "generatedAudio": resolved_generated_audio(&self.plan, shot).as_timeline_str(),
                        "versionHistory": [{
                            "assetId": take.asset_id,
                            "source": "original",
                            "jobId": job_id,
                            "note": format!("film-harness {} shot {}", self.record.run_id, shot.id),
                        }],
                        HARNESS_KEY: picture_block(
                            &self.record.run_id,
                            &shot.id,
                            planned_index,
                            *attempt,
                            job_id.as_deref(),
                            &take.asset_id,
                        ),
                    }));
                }
            }
        }
        // The cut order the re-layout keeps: the saved sequence's own order, new shots after it.
        let order: Vec<String> = items
            .iter()
            .filter_map(|item| harness_str(item, "shotId").map(str::to_owned))
            .collect();

        // Sound: one dialogue clip per shot that has one, each keeping its offset from the start of
        // its OWN shot, and the two beds placed ONCE across the whole sequence rather than
        // restarting at every cut. `relayout_timeline` gives all of them their positions. These are
        // the HARNESS-placed items (each carries a `filmHarness.role`), re-derived from the plan on
        // every pass exactly as a re-layout after an edit re-derives them; an item on one of these
        // tracks that the harness did NOT place — the editor's own — is carried over untouched,
        // as is the track's own fader (`gain` / `muted`), which a person may have moved.
        //
        // Derived from `order` — the shots the merged PICTURE holds — and not from
        // `selected_takes()` (sc-22715). The two are not the same set: the picture is merged onto
        // the saved sequence, so a shot whose take was rejected (`reject-take`, or a replacement
        // that produced nothing) keeps its item and stays in the cut, while its selection is gone.
        // Deriving the lines from the selection dropped exactly those shots' dialogue on the next
        // re-assembly — the 2026-09-14 evaluation's own film lost SH050's line to a `replace-take`
        // on SH040, and the line is not the harness's to delete while the shot it belongs to is
        // still on screen. A line whose shot HAS left the cut is still dropped, by
        // `relayout_timeline`, which is where "is this shot in the sequence" is actually known.
        let mut dialogue_items = Vec::new();
        for shot_id in &order {
            let Some(shot) = self.plan.shots.iter().find(|shot| &shot.id == shot_id) else {
                // An item the harness placed for a shot the plan no longer names. Its picture item
                // is carried over untouched above; there is no plan clip to derive a line from.
                continue;
            };
            let Some(clip) = &shot.dialogue_clip else {
                continue;
            };
            let Some(asset) = self.sound_assets.get(&clip.role) else {
                continue;
            };
            let length = clip
                .duration_seconds
                .filter(|seconds| *seconds > 0.0)
                .or_else(|| {
                    asset
                        .duration_seconds
                        .map(|total| (total - clip.source_in_seconds).max(MIN_ITEM_SECONDS))
                })
                .unwrap_or(MIN_ITEM_SECONDS)
                .max(MIN_ITEM_SECONDS);
            dialogue_items.push(json!({
                "id": format!(
                    "item_line_{}_{}",
                    shot.id.to_ascii_lowercase(),
                    &self.record.run_id[4..12]
                ),
                "trackId": DIALOGUE_TRACK_ID,
                "assetId": asset.asset_id,
                "type": "audio",
                "displayName": format!("{} — dialogue ({})", shot.id, clip.role).chars().take(160).collect::<String>(),
                "sourceIn": clip.source_in_seconds,
                "sourceOut": clip.source_in_seconds + length,
                "timelineStart": 0.0,
                "timelineEnd": length,
                "speed": 1.0,
                "fit": "fit",
                "volume": clip.gain.clamp(0.0, 2.0),
                "fadeInSeconds": clip.fade_in_seconds,
                "fadeOutSeconds": clip.fade_out_seconds,
                HARNESS_KEY: harness_block(
                    ROLE_DIALOGUE,
                    &self.record.run_id,
                    Some(&shot.id),
                    clip.offset_seconds,
                ),
            }));
        }

        let mut picture = existing_tracks[track_index].clone();
        picture["items"] = Value::Array(items);
        let mut tracks = vec![
            picture,
            merge_harness_audio_track(
                &existing_tracks,
                audio_track(
                    DIALOGUE_TRACK_ID,
                    "Dialogue",
                    ROLE_DIALOGUE,
                    &self.plan.sound.dialogue,
                    dialogue_items,
                ),
            ),
        ];
        for (bed_track_id, name, role, bed) in [
            (
                AMBIENCE_TRACK_ID,
                "Ambience",
                ROLE_AMBIENCE,
                self.plan.sound.ambience.as_ref(),
            ),
            (
                MUSIC_TRACK_ID,
                "Music",
                ROLE_MUSIC,
                self.plan.sound.music.as_ref(),
            ),
        ] {
            let Some(bed) = bed else { continue };
            let Some(asset) = self.sound_assets.get(&bed.role) else {
                continue;
            };
            tracks.push(merge_harness_audio_track(
                &existing_tracks,
                bed_track(bed_track_id, name, role, bed, asset, &self.record.run_id),
            ));
        }
        for (index, bed) in self.plan.sound.sfx.iter().enumerate() {
            let Some(asset) = self.sound_assets.get(&bed.role) else {
                continue;
            };
            let track_id = format!("{SFX_TRACK_PREFIX}_{index}");
            tracks.push(merge_harness_audio_track(
                &existing_tracks,
                bed_track(
                    &track_id,
                    "Sound effect",
                    ROLE_SFX,
                    bed,
                    asset,
                    &self.record.run_id,
                ),
            ));
        }
        // Keep every track the API created that the harness does not own (the overlay lane and the
        // editor's default audio lane) so a harness timeline opens in the editor unchanged.
        for track in &existing_tracks {
            let id = track.get("id").and_then(Value::as_str).unwrap_or_default();
            if !tracks
                .iter()
                .any(|kept| kept.get("id").and_then(Value::as_str) == Some(id))
            {
                tracks.push(track.clone());
            }
        }
        timeline["tracks"] = Value::Array(tracks);

        let planned_duration = relayout_timeline(&mut timeline, Some(&order))?;
        let saved = self
            .client
            .expect_ok(
                "POST",
                &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/film-deliveries"),
                Some(json!({"runId": self.record.run_id,
                "shotOrder": self.plan.shots.iter().map(|shot| &shot.id).collect::<Vec<_>>(),
                "timeline": timeline})),
            )
            .await?;
        for take in &takes {
            let present = saved["tracks"]
                .as_array()
                .into_iter()
                .flatten()
                .flat_map(|track| track["items"].as_array().into_iter().flatten())
                .any(|item| {
                    harness_str(item, "runId") == Some(self.record.run_id.as_str())
                        && harness_str(item, "shotId") == Some(take.shot_id.as_str())
                        && harness_str(item, "role") == Some(ROLE_PICTURE)
                });
            if !present
                && saved["filmAssembly"]["runs"][&self.record.run_id]["deletedShots"][&take.shot_id]
                    .is_null()
            {
                return Err(HarnessError::Transport(format!(
                    "saved timeline {timeline_id} lost shot {} without a deletion tombstone",
                    take.shot_id
                )));
            }
        }
        // The store recomputes `duration` across every track on save, so read it back rather than
        // republishing the harness's own arithmetic.
        let duration = saved
            .get("duration")
            .and_then(Value::as_f64)
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .unwrap_or(planned_duration);
        // Edits a human applied to this sequence are HISTORY, not something a re-assembly erases: a
        // resume or a replacement re-lays the picture, but the record must still say the trim or
        // reorder happened (sc-22712 under sc-22711's record).
        let prior_edits = self
            .record
            .timeline
            .as_ref()
            .map(|timeline| timeline.edits.clone())
            .unwrap_or_default();
        self.record.timeline = Some(timeline_record(
            &timeline_id,
            &timeline_name,
            aspect_ratio,
            Some(reduced_aspect_ratio(first_width, first_height)),
            Some(first_width),
            Some(first_height),
            self.fps,
            duration,
            &saved,
            &track_id,
            self.plan.sound.generated_audio,
            prior_edits,
        ));
        if let Some(export) = self.record.export.as_mut() {
            export.stale =
                export.timeline_revision != Some(sceneworks_core::film_timeline::revision(&saved));
        }
        self.persist()?;
        Ok(true)
    }

    /// Render the timeline through the `timeline_export` job, adopting an export this run already
    /// dispatched rather than starting a second render.
    pub(super) async fn run_export(&mut self) -> Result<bool, HarnessError> {
        let project_id = self.project_id()?;
        let Some(timeline) = self.record.timeline.clone() else {
            return Ok(false);
        };
        if let Some(export) = &self.record.export {
            if export.status == "completed" && export.asset_id.is_some() && !export.stale {
                return Ok(true);
            }
        }
        if self.canceled()
            && self.record.export_pending.is_none()
            && !self
                .record
                .export
                .as_ref()
                .is_some_and(|export| export.status == "running")
        {
            self.halt(
                RunOutcome::Canceled,
                "canceled",
                "canceled before the timeline export was dispatched".to_owned(),
                true,
            );
            return Ok(false);
        }
        // An export that is stale or finished badly is SUPERSEDED: the next one is a different job,
        // and saying so is what lets a resume tell the new export apart from the old one on the same
        // timeline. An export merely recorded as `running` is not superseded — it is the very job
        // this pass is about to adopt again.
        let superseded = self
            .record
            .export
            .as_ref()
            .filter(|export| {
                export.stale
                    || (export.status != "running"
                        && !(export.status == "completed" && export.asset_id.is_some()))
            })
            .map(|export| export.job_id.clone());
        // An export recorded as `running` is the very job this pass is about to poll again, so it
        // is adopted BY ID — never looked up in the listing, which is what makes the `requested_at`
        // floor below safe to apply to everything the listing can return.
        let in_flight = self
            .record
            .export
            .as_ref()
            .filter(|export| Some(&export.job_id) != superseded.as_ref())
            .map(|export| export.job_id.clone());
        if let Some(job_id) = superseded.clone() {
            // The exclusion set is CUMULATIVE: a second re-export must exclude the first export as
            // well as the second, or it adopts the first one's finished job and records its
            // pre-replacement asset as the current MP4.
            if !self.record.superseded_export_job_ids.contains(&job_id) {
                self.record.superseded_export_job_ids.push(job_id);
            }
        }
        if self.record.export_pending.is_none() {
            self.record.export_pending = Some(ExportPending {
                requested_at: utc_now(),
                supersedes: superseded,
            });
            self.record.export = None;
            // Persisted BEFORE the export job exists, for the same reason a shot attempt is.
            self.persist()?;
        }
        let requested_at = self
            .record
            .export_pending
            .as_ref()
            .map(|pending| pending.requested_at.clone())
            .unwrap_or_else(utc_now);
        let superseded_ids = self.record.superseded_export_job_ids.clone();
        let existing = match in_flight {
            Some(job_id) => Some(job_id),
            None => {
                self.client
                    .find_export_job(
                        &project_id,
                        &timeline.timeline_id,
                        &superseded_ids,
                        &requested_at,
                    )
                    .await?
            }
        };
        let export_job_id = match existing {
            Some(job_id) => job_id,
            None => {
                if self.canceled() {
                    self.halt(
                        RunOutcome::Canceled,
                        "canceled",
                        "canceled before the timeline export was dispatched".to_owned(),
                        true,
                    );
                    self.persist()?;
                    return Ok(false);
                }
                let export_job = self
                    .client
                    .expect_ok(
                        "POST",
                        &format!(
                            "/api/v1/projects/{project_id}/timelines/{}/exports",
                            timeline.timeline_id
                        ),
                        Some(json!({
                            "resolution": export_resolution_for(self.tallest_selected()),
                            "fps": self.fps,
                            "requestedGpu": "auto",
                        })),
                    )
                    .await?;
                export_job
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        HarnessError::Transport(format!("export response has no id: {export_job}"))
                    })?
                    .to_owned()
            }
        };
        self.record.export = Some(ExportRecord {
            timeline_revision: self
                .client
                .expect_ok("GET", &format!("/api/v1/jobs/{export_job_id}"), None)
                .await?["payload"]["timelineRevision"]
                .as_u64(),
            job_id: export_job_id.clone(),
            status: "running".to_owned(),
            stale: false,
            asset_id: None,
            render_path: None,
            error: None,
            dropped_audio_layers: Vec::new(),
        });
        self.record.export_pending = None;
        self.persist()?;

        let export_started = Instant::now();
        let settle_grace = ASSET_SETTLE_GRACE.min(self.shot_budget());
        let (view, poll_stop) = self
            .client
            .wait_for_job(
                &export_job_id,
                self.bounds(export_started + self.shot_budget()),
            )
            .await?;
        let asset_id = view
            .result
            .get("assetIds")
            .and_then(Value::as_array)
            .and_then(|ids| ids.first())
            .and_then(Value::as_str)
            .map(str::to_owned);
        let render_path = view
            .result
            .get("renderPath")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let status = match poll_stop {
            PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
            PollStop::Operator => "canceled_by_operator".to_owned(),
            PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
        };
        let export_ok = status == "completed" && asset_id.is_some();
        self.record.export = Some(ExportRecord {
            timeline_revision: self
                .record
                .export
                .as_ref()
                .and_then(|export| export.timeline_revision),
            job_id: export_job_id,
            status,
            stale: false,
            asset_id,
            render_path,
            dropped_audio_layers: dropped_audio_layers(&view.result),
            error: (!export_ok).then(|| match poll_stop {
                PollStop::Terminal => view.failure_text(),
                PollStop::AssetsUnsettled => format!(
                    "the export job reached {} but its assets never settled within {:.0}s",
                    view.status,
                    settle_grace.as_secs_f64()
                ),
                PollStop::Operator => "canceled by operator during the export".to_owned(),
                PollStop::ShotBudget => format!(
                    "export exceeded the per-job budget of {}s",
                    self.plan.limits.max_shot_seconds
                ),
                PollStop::RunBudget => format!(
                    "run exceeded its budget of {}s during the export",
                    self.plan.limits.max_run_seconds
                ),
            }),
        });
        self.persist()?;
        match poll_stop {
            PollStop::Operator => self.halt(
                RunOutcome::Canceled,
                "canceled",
                "canceled while the timeline export was in flight".to_owned(),
                true,
            ),
            PollStop::RunBudget => self.halt(
                RunOutcome::StoppedRunBudget,
                "run_budget",
                format!(
                    "the run's {}s wall-clock budget ran out during the export; raise \
                     limits.maxRunSeconds and start a new run",
                    self.plan.limits.max_run_seconds
                ),
                false,
            ),
            _ => {}
        }
        Ok(export_ok)
    }

    /// Assemble and export, unless a stop already landed. A canceled run stops here with its takes
    /// intact: the assets are already the project's, and a resume finishes the assembly.
    pub(super) async fn assemble_and_export(&mut self) -> Result<bool, HarnessError> {
        if matches!(
            self.stop.as_ref().map(|(outcome, _)| *outcome),
            Some(RunOutcome::Canceled | RunOutcome::StoppedRunBudget)
        ) {
            return Ok(false);
        }
        if !self.assemble_timeline().await? {
            return Ok(false);
        }
        if !self.export {
            return Ok(true);
        }
        self.run_export().await
    }
}
