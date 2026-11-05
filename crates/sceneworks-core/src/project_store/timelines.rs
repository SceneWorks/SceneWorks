//! ProjectStore timelines aggregate.

use super::*;

/// Every aspect ratio a timeline may declare, with the frame it renders at. One declaration
/// (sc-22710): a caller that must CHOOSE a ratio for footage of some other shape — the film
/// harness picks one before it creates a timeline — reads this list rather than hand-copying it
/// and drifting into a 400 here.
pub const TIMELINE_ASPECT_RATIOS: &[(&str, u32, u32)] = &[
    ("16:9", 1280, 720),
    ("9:16", 720, 1280),
    ("1:1", 1024, 1024),
];

fn timeline_dimensions(aspect_ratio: &str) -> ProjectStoreResult<(u32, u32)> {
    TIMELINE_ASPECT_RATIOS
        .iter()
        .find(|(name, _, _)| *name == aspect_ratio)
        .map(|(_, width, height)| (*width, *height))
        .ok_or_else(|| {
            ProjectStoreError::BadRequest(
                "Aspect ratio must be one of 16:9, 9:16, or 1:1".to_owned(),
            )
        })
}

/// Roles a timeline track may declare. `kind` says how a track is rendered (picture, overlay,
/// mixed audio); `role` says what it CARRIES, which is what lets dialogue, ambience and music be
/// three independently gained and muted buses rather than one undifferentiated "Audio" lane
/// (sc-22712). The exporter mixes every non-muted `kind: "audio"` track regardless of role — role
/// is descriptive, never a routing switch — so an unrecognised bed still reaches the MP4.
pub const TIMELINE_TRACK_ROLES: &[&str] = &[
    "picture", "overlay", "dialogue", "ambience", "music", "sfx", "sound",
];

/// What a picture item's own (generated) audio does in the mix. Defaults to `mute`, which is the
/// only default that cannot silently double a placed dialogue clip (sc-22712): a generated take
/// whose model spoke the line and a recorded dialogue clip for the same beat would otherwise both
/// land in the mix, and nothing in the timeline would say which one the editor meant.
pub const TIMELINE_GENERATED_AUDIO: &[&str] = &["include", "mute"];

/// The role a track's `kind` implies when a document does not declare one. Keeps every timeline
/// written before sc-22712 loading unchanged: its three default tracks become picture / overlay /
/// sound, and `sound` is mixed exactly like a named bed.
fn default_track_role(kind: &str) -> &'static str {
    match kind {
        "overlay" => "overlay",
        "audio" => "sound",
        _ => "picture",
    }
}

fn default_timeline_tracks() -> Value {
    json!([
        {"id": "track_main", "name": "Main", "kind": "video", "role": "picture", "locked": false, "muted": false, "gain": 1.0, "items": []},
        {"id": "track_overlay", "name": "Overlay", "kind": "overlay", "role": "overlay", "locked": false, "muted": false, "gain": 1.0, "items": []},
        {"id": "track_audio", "name": "Audio", "kind": "audio", "role": "sound", "locked": false, "muted": false, "gain": 1.0, "items": []}
    ])
}

fn compute_timeline_duration(timeline: &Value) -> f64 {
    timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .filter_map(|item| item.get("timelineEnd").and_then(Value::as_f64))
        .fold(0.0, f64::max)
}

fn validate_and_default_timeline(timeline: &mut Value) -> ProjectStoreResult<()> {
    let object = timeline
        .as_object_mut()
        .ok_or_else(|| ProjectStoreError::BadRequest("Timeline must be an object".to_owned()))?;
    object
        .entry("schemaVersion".to_owned())
        .or_insert_with(|| json!(1));
    object
        .entry("aspectRatio".to_owned())
        .or_insert_with(|| Value::String("16:9".to_owned()));
    object
        .entry("width".to_owned())
        .or_insert_with(|| json!(1280));
    object
        .entry("height".to_owned())
        .or_insert_with(|| json!(720));
    object.entry("fps".to_owned()).or_insert_with(|| json!(30));
    object
        .entry("duration".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("tracks".to_owned())
        .or_insert_with(|| json!([]));
    object
        .entry("transitions".to_owned())
        .or_insert_with(|| json!([]));

    validate_u64_range(timeline, "schemaVersion", 0, u32::MAX as u64)?;
    validate_enum(timeline, "aspectRatio", &["16:9", "9:16", "1:1"])?;
    validate_u64_range(timeline, "width", 256, 3840)?;
    validate_u64_range(timeline, "height", 256, 3840)?;
    validate_u64_range(timeline, "fps", 1, 60)?;
    validate_f64_range(timeline, "duration", 0.0, f64::INFINITY)?;

    let tracks = timeline
        .get_mut("tracks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("tracks must be an array".to_owned()))?;
    for track in tracks {
        validate_timeline_track(track)?;
    }

    let transitions = timeline
        .get_mut("transitions")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("transitions must be an array".to_owned()))?;
    for transition in transitions {
        validate_timeline_transition(transition)?;
    }
    Ok(())
}

fn validate_timeline_track(track: &mut Value) -> ProjectStoreResult<()> {
    required_str(track, "id")?;
    required_str(track, "name")?;
    validate_enum(track, "kind", &["video", "overlay", "audio"])?;
    let kind = required_str(track, "kind")?.to_owned();
    let object = track.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline track must be an object".to_owned())
    })?;
    object
        .entry("locked".to_owned())
        .or_insert_with(|| Value::Bool(false));
    object
        .entry("muted".to_owned())
        .or_insert_with(|| Value::Bool(false));
    // sc-22712: `gain` is the track's bus fader, and `role` names the bus. Both are defaulted from
    // what the document already says, so a timeline saved before this existed round-trips to the
    // same audible result (unity gain, role implied by kind) rather than being rejected or muted.
    object
        .entry("gain".to_owned())
        .or_insert_with(|| json!(1.0));
    if !object.contains_key("role") || object.get("role") == Some(&Value::Null) {
        let role = default_track_role(&kind);
        object.insert("role".to_owned(), Value::String(role.to_owned()));
    }
    object
        .entry("items".to_owned())
        .or_insert_with(|| json!([]));
    validate_bool(track, "locked")?;
    validate_bool(track, "muted")?;
    validate_enum(track, "role", TIMELINE_TRACK_ROLES)?;
    let gain = validate_f64_range(track, "gain", 0.0, 4.0)?;
    track
        .as_object_mut()
        .expect("checked above")
        .insert("gain".to_owned(), json!(gain));
    let items = track
        .get_mut("items")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("track items must be an array".to_owned()))?;
    for item in items {
        validate_timeline_item(item)?;
    }
    Ok(())
}

fn validate_timeline_item(item: &mut Value) -> ProjectStoreResult<()> {
    required_str(item, "trackId")?;
    validate_required_string(item, "assetId", Some(1), None)?;
    validate_required_string(item, "displayName", Some(1), Some(160))?;
    let object = item.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item must be an object".to_owned())
    })?;
    if !object.contains_key("id") || object.get("id") == Some(&Value::Null) {
        object.insert(
            "id".to_owned(),
            Value::String(format!("item_{}", random_hex(16)?)),
        );
    }
    object
        .entry("type".to_owned())
        .or_insert_with(|| Value::String("video".to_owned()));
    object
        .entry("sourceIn".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("sourceOut".to_owned())
        .or_insert_with(|| json!(4.0));
    object
        .entry("timelineStart".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("timelineEnd".to_owned())
        .or_insert_with(|| json!(4.0));
    object
        .entry("speed".to_owned())
        .or_insert_with(|| json!(1.0));
    object
        .entry("fit".to_owned())
        .or_insert_with(|| Value::String("fit".to_owned()));
    object
        .entry("volume".to_owned())
        .or_insert_with(|| json!(1.0));
    // sc-22712. `fadeInSeconds`/`fadeOutSeconds` are AUDIO fades, measured from the item's own
    // ends, and are what lets a continuous bed come up under the first cut and away under the last
    // without being chopped into per-shot pieces. They are deliberately separate from
    // `transitionIn`/`transitionOut`, which are picture transitions with their own vocabulary.
    object
        .entry("fadeInSeconds".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("fadeOutSeconds".to_owned())
        .or_insert_with(|| json!(0.0));
    // A picture item's own audio track is MUTED unless the document says otherwise. See
    // `TIMELINE_GENERATED_AUDIO`: this default is the doubling guard, so it must stay `mute`.
    object
        .entry("generatedAudio".to_owned())
        .or_insert_with(|| Value::String("mute".to_owned()));
    object
        .entry("versionAssetIds".to_owned())
        .or_insert_with(|| json!([]));
    object
        .entry("versionHistory".to_owned())
        .or_insert_with(|| json!([]));

    validate_required_string(item, "id", Some(1), None)?;
    validate_enum(item, "type", &["video", "image", "audio"])?;
    let source_in = validate_f64_range(item, "sourceIn", 0.0, f64::INFINITY)?;
    let source_out = validate_f64_range(item, "sourceOut", 0.0, f64::INFINITY)?;
    let timeline_start = validate_f64_range(item, "timelineStart", 0.0, f64::INFINITY)?;
    let timeline_end = validate_f64_range(item, "timelineEnd", 0.0, f64::INFINITY)?;
    let speed = validate_f64_range(item, "speed", 0.1, 8.0)?;
    validate_enum(item, "fit", &["fit", "fill", "stretch"])?;
    let volume = validate_f64_range(item, "volume", 0.0, 2.0)?;
    let fade_in = validate_f64_range(item, "fadeInSeconds", 0.0, 60.0)?;
    let fade_out = validate_f64_range(item, "fadeOutSeconds", 0.0, 60.0)?;
    validate_enum(item, "generatedAudio", TIMELINE_GENERATED_AUDIO)?;
    let object = item.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item must be an object".to_owned())
    })?;
    object.insert("sourceIn".to_owned(), json!(source_in));
    object.insert("sourceOut".to_owned(), json!(source_out));
    object.insert("timelineStart".to_owned(), json!(timeline_start));
    object.insert("timelineEnd".to_owned(), json!(timeline_end));
    object.insert("speed".to_owned(), json!(speed));
    object.insert("volume".to_owned(), json!(volume));
    object.insert("fadeInSeconds".to_owned(), json!(fade_in));
    object.insert("fadeOutSeconds".to_owned(), json!(fade_out));
    object
        .entry("transitionIn".to_owned())
        .or_insert(Value::Null);
    object
        .entry("transitionOut".to_owned())
        .or_insert(Value::Null);
    if source_out <= source_in {
        return Err(ProjectStoreError::BadRequest(
            "sourceOut must be greater than sourceIn.".to_owned(),
        ));
    }
    if timeline_end <= timeline_start {
        return Err(ProjectStoreError::BadRequest(
            "timelineEnd must be greater than timelineStart.".to_owned(),
        ));
    }

    item.get("versionAssetIds")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProjectStoreError::BadRequest("versionAssetIds must be an array".to_owned())
        })?;
    let versions = item
        .get_mut("versionHistory")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            ProjectStoreError::BadRequest("versionHistory must be an array".to_owned())
        })?;
    for version in versions {
        validate_timeline_item_version(version)?;
    }
    if let Some(transition) = item
        .get_mut("transitionIn")
        .filter(|value| !value.is_null())
    {
        validate_timeline_transition(transition)?;
    }
    if let Some(transition) = item
        .get_mut("transitionOut")
        .filter(|value| !value.is_null())
    {
        validate_timeline_transition(transition)?;
    }
    Ok(())
}

fn validate_timeline_item_version(version: &mut Value) -> ProjectStoreResult<()> {
    validate_required_string(version, "assetId", Some(1), None)?;
    let object = version.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item version must be an object".to_owned())
    })?;
    object
        .entry("source".to_owned())
        .or_insert_with(|| Value::String("manual".to_owned()));
    validate_enum(
        version,
        "source",
        &[
            "original",
            "replacement",
            "extension",
            "bridge",
            "restore",
            "manual",
        ],
    )
}

fn validate_timeline_transition(transition: &mut Value) -> ProjectStoreResult<()> {
    let object = transition.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline transition must be an object".to_owned())
    })?;
    if !object.contains_key("id") || object.get("id") == Some(&Value::Null) {
        object.insert(
            "id".to_owned(),
            Value::String(format!("transition_{}", random_hex(16)?)),
        );
    }
    object
        .entry("type".to_owned())
        .or_insert_with(|| Value::String("cut".to_owned()));
    object
        .entry("duration".to_owned())
        .or_insert_with(|| json!(0.0));
    validate_required_string(transition, "id", Some(1), None)?;
    validate_enum(
        transition,
        "type",
        &["cut", "crossfade", "fade_from_black", "fade_to_black"],
    )?;
    validate_f64_range(transition, "duration", 0.0, 10.0)?;
    Ok(())
}

fn normalize_timeline_items(timeline: &mut Value) -> ProjectStoreResult<()> {
    let Some(tracks) = timeline.get_mut("tracks").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for track in tracks {
        let Some(items) = track.get_mut("items").and_then(Value::as_array_mut) else {
            continue;
        };
        for item in items {
            let Some(object) = item.as_object_mut() else {
                continue;
            };
            let Some(asset_id) = object
                .get("assetId")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            object
                .entry("currentVersionAssetId".to_owned())
                .or_insert_with(|| Value::String(asset_id.clone()));
            let version_ids = object
                .entry("versionAssetIds".to_owned())
                .or_insert_with(|| json!([]));
            if let Some(version_ids) = version_ids.as_array_mut() {
                let has_asset = version_ids
                    .iter()
                    .any(|value| value.as_str() == Some(&asset_id));
                if !has_asset {
                    version_ids.push(Value::String(asset_id.clone()));
                }
            }
            let needs_history = object
                .get("versionHistory")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty);
            if needs_history {
                object.insert(
                    "versionHistory".to_owned(),
                    json!([{
                        "assetId": asset_id,
                        "createdAt": Value::Null,
                        "source": "original",
                        "jobId": Value::Null,
                        "note": Value::Null
                    }]),
                );
            }
        }
    }
    Ok(())
}

fn validate_required_string(
    payload: &Value,
    key: &str,
    min_length: Option<usize>,
    max_length: Option<usize>,
) -> ProjectStoreResult<()> {
    let value = required_str(payload, key)?;
    let length = value.chars().count();
    if let Some(min) = min_length.filter(|min| length < *min) {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be at least {min} characters"
        )));
    }
    if let Some(max) = max_length.filter(|max| length > *max) {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be at most {max} characters"
        )));
    }
    Ok(())
}

fn validate_enum(payload: &Value, key: &str, allowed: &[&str]) -> ProjectStoreResult<()> {
    let value = required_str(payload, key)?;
    if !allowed.contains(&value) {
        return Err(ProjectStoreError::BadRequest(format!(
            "Unsupported value for {key}: {value}"
        )));
    }
    Ok(())
}

fn validate_bool(payload: &Value, key: &str) -> ProjectStoreResult<()> {
    if payload.get(key).and_then(Value::as_bool).is_none() {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be a boolean"
        )));
    }
    Ok(())
}

fn validate_u64_range(payload: &Value, key: &str, min: u64, max: u64) -> ProjectStoreResult<u64> {
    let value = payload
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("{key} must be an integer")))?;
    if value < min || value > max {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be between {min} and {max}"
        )));
    }
    Ok(value)
}

fn validate_f64_range(payload: &Value, key: &str, min: f64, max: f64) -> ProjectStoreResult<f64> {
    let value = payload
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("{key} must be a number")))?;
    if !value.is_finite() || value < min || value > max {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be between {min} and {max}"
        )));
    }
    Ok(value)
}

fn timeline_file_path(project_path: &Path, timeline_id: &str, name: &str) -> PathBuf {
    let suffix = timeline_id
        .chars()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    project_path.join("timelines").join(format!(
        "{}-{suffix}.sceneworks.timeline.json",
        slugify(name, "timeline", Some(48))
    ))
}

pub(super) fn index_timeline(
    project_path: &Path,
    timeline: &Value,
    rel_path: &str,
) -> ProjectStoreResult<()> {
    let mut connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute(
        "
        insert or replace into timelines (
          id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
        ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        ",
        params![
            required_str(timeline, "id")?,
            optional_str(timeline, "name").unwrap_or("Timeline"),
            rel_path,
            optional_str(timeline, "aspectRatio").unwrap_or("16:9"),
            optional_u64(timeline, "width").unwrap_or(1280),
            optional_u64(timeline, "height").unwrap_or(720),
            optional_u64(timeline, "fps").unwrap_or(30),
            optional_f64(timeline, "duration").unwrap_or(0.0),
            optional_str(timeline, "createdAt").unwrap_or(""),
            optional_str(timeline, "updatedAt")
                .or_else(|| optional_str(timeline, "createdAt"))
                .unwrap_or(""),
        ],
    )?;
    transaction.commit()?;
    Ok(())
}

/// Remove every other readable timeline document with this id. Used by
/// [`ProjectStore::save_timeline`] to make the replacement file authoritative
/// without trusting the current database path alone.
fn remove_stale_timeline_files(
    project_path: &Path,
    timeline_id: &str,
    authoritative_path: &Path,
) -> ProjectStoreResult<()> {
    for candidate in read_dir_paths(&project_path.join("timelines"))? {
        if candidate == authoritative_path
            || !candidate
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        {
            continue;
        }
        let Ok(candidate_timeline) = read_json(&candidate) else {
            continue;
        };
        if candidate_timeline.get("id").and_then(Value::as_str) == Some(timeline_id) {
            fs::remove_file(candidate)?;
        }
    }
    Ok(())
}

pub(super) fn find_timeline_file(
    project_path: &Path,
    timeline_id: &str,
) -> ProjectStoreResult<TimelineFile> {
    ensure_project_db_ready(project_path)?;
    let mut db = connect_project_db_migrated(project_path)?;
    let connection = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let indexed_path = connection
        .query_row(
            "select file_path from timelines where id = ?1",
            params![timeline_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if let Some(indexed_path) = indexed_path.as_deref() {
        let path = if is_safe_relative_path(indexed_path) {
            Some(project_path.join(indexed_path))
        } else {
            None
        };
        if let Some(path) = path.filter(|path| path.exists()) {
            let project_root = project_path.canonicalize()?;
            let canonical = path.canonicalize()?;
            if !canonical.starts_with(&project_root) {
                return Err(ProjectStoreError::BadRequest(
                    "Timeline file path must stay inside the project".to_owned(),
                ));
            }
            return Ok(TimelineFile {
                path: canonical,
                relative_path: indexed_path.to_owned(),
            });
        }
    }

    let timeline_dir = project_path.join("timelines");
    for candidate in read_dir_paths(&timeline_dir)? {
        if !candidate
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        {
            continue;
        }
        let Ok(timeline) = read_json(&candidate) else {
            continue;
        };
        if timeline.get("id").and_then(Value::as_str) == Some(timeline_id) {
            let rel_path = relative_string(project_path, &candidate)?;
            connection.execute(
                "update timelines set file_path = ?1 where id = ?2",
                params![rel_path, timeline_id],
            )?;
            connection.commit()?;
            return Ok(TimelineFile {
                path: candidate,
                relative_path: rel_path,
            });
        }
    }

    if let Some(indexed_path) = indexed_path {
        return Err(ProjectStoreError::NotFound(format!(
            "Timeline file not found at indexed path {indexed_path}; reindex required"
        )));
    }
    Err(ProjectStoreError::NotFound("Timeline not found".to_owned()))
}

impl ProjectStore {
    pub fn list_timelines(&self, project_id: &str) -> ProjectStoreResult<Vec<TimelineSummary>> {
        let project_path = self.find_project_path(project_id)?;
        ensure_project_db_ready(&project_path)?;
        let connection = connect_project_db(&project_path)?;
        let mut statement = connection.prepare(
            "
            select id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
              from timelines
             order by updated_at desc
            ",
        )?;
        let timelines = statement
            .query_map([], |row| {
                Ok(TimelineSummary {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    file_path: row.get(2)?,
                    aspect_ratio: row.get(3)?,
                    width: row.get(4)?,
                    height: row.get(5)?,
                    fps: row.get(6)?,
                    duration: row.get(7)?,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProjectStoreError::from)?;
        Ok(timelines)
    }

    pub fn create_timeline(
        &self,
        project_id: &str,
        name: &str,
        aspect_ratio: &str,
        fps: u32,
    ) -> ProjectStoreResult<Value> {
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name is required".to_owned(),
            ));
        }
        if name.chars().count() > 120 {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name must be at most 120 characters".to_owned(),
            ));
        }
        if !(1..=60).contains(&fps) {
            return Err(ProjectStoreError::BadRequest(
                "FPS must be between 1 and 60".to_owned(),
            ));
        }
        let (width, height) = timeline_dimensions(aspect_ratio)?;
        let timeline = json!({
            "schemaVersion": 1,
            "id": format!("timeline_{}", random_hex(16)?),
            "projectId": project_id,
            "name": name,
            "aspectRatio": aspect_ratio,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": 0.0,
            "tracks": default_timeline_tracks(),
            "transitions": [],
            "createdAt": Value::Null,
            "updatedAt": Value::Null,
        });
        self.save_timeline(project_id, timeline)
    }

    pub fn get_timeline(&self, project_id: &str, timeline_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let timeline_file = find_timeline_file(&project_path, timeline_id)?;
        read_json(&timeline_file.path)
    }

    pub fn save_timeline(
        &self,
        project_id: &str,
        mut timeline: Value,
    ) -> ProjectStoreResult<Value> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let timeline_id = required_str(&timeline, "id")?.to_owned();
        // The client supplies `id`, and `timeline_file_path` embeds its last 8
        // chars into the write path while `relative_string`'s lexical strip lets
        // an unnormalized `..`-bearing id through — charset-check it (matching the
        // asset/character/track/dataset guards) BEFORE any path is derived so the
        // write can never escape the project dir (sc-8871 / F-069).
        if !is_safe_id(&timeline_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid timeline id".to_owned(),
            ));
        }
        let existing = match find_timeline_file(&project_path, &timeline_id) {
            Ok(file) => Some(read_json(&file.path)?),
            Err(ProjectStoreError::NotFound(_)) => None,
            Err(error) => return Err(error),
        };
        let current_revision = existing
            .as_ref()
            .map(crate::film_timeline::revision)
            .unwrap_or(0);
        if timeline
            .get("revision")
            .is_some_and(|value| value.as_u64().is_none())
        {
            return Err(ProjectStoreError::BadRequest(
                "revision must be a non-negative integer".into(),
            ));
        }
        let expected_revision = crate::film_timeline::revision(&timeline);
        if expected_revision != current_revision {
            return Err(ProjectStoreError::TimelineConflict {
                code: "timeline_revision_conflict",
                context: json!({"timelineId": timeline_id, "expectedRevision": expected_revision, "currentRevision": current_revision}),
            });
        }
        crate::film_timeline::validate_metadata(&timeline)?;
        if let Some(previous) = &existing {
            crate::film_timeline::validate_metadata(previous)?;
            crate::film_timeline::reconcile_cut(previous, &mut timeline, current_revision + 1);
        }
        timeline["revision"] = json!(current_revision + 1);
        let timeline_project_id = required_str(&timeline, "projectId")?;
        if timeline_project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Project ID mismatch".to_owned(),
            ));
        }
        let name = required_str(&timeline, "name")?.to_owned();
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name is required".to_owned(),
            ));
        }
        let now = utc_now();
        validate_and_default_timeline(&mut timeline)?;
        let duration = compute_timeline_duration(&timeline);
        let object = timeline.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Timeline must be an object".to_owned())
        })?;
        if !object.contains_key("createdAt") || object.get("createdAt") == Some(&Value::Null) {
            object.insert("createdAt".to_owned(), Value::String(now.clone()));
        }
        object.insert("updatedAt".to_owned(), Value::String(now));
        object.insert("duration".to_owned(), json!(duration));
        if !object
            .get("tracks")
            .and_then(Value::as_array)
            .is_some_and(|tracks| !tracks.is_empty())
        {
            object.insert("tracks".to_owned(), default_timeline_tracks());
        }
        object
            .entry("transitions".to_owned())
            .or_insert_with(|| json!([]));
        normalize_timeline_items(&mut timeline)?;

        let path = timeline_file_path(&project_path, &timeline_id, &name);
        let rel_path = relative_string(&project_path, &path)?;
        write_json(&path, &timeline)?;
        // A rename changes the slug portion of the timeline file name while keeping
        // its id. Remove every prior document for this id after the replacement has
        // reached disk; otherwise a later reindex sees duplicate ids and can
        // nondeterministically revive the pre-rename content.
        remove_stale_timeline_files(&project_path, &timeline_id, &path)?;
        index_timeline(&project_path, &timeline, &rel_path)?;
        Ok(timeline)
    }

    pub fn save_existing_timeline(
        &self,
        project_id: &str,
        timeline_id: &str,
        timeline: Value,
    ) -> ProjectStoreResult<Value> {
        if required_str(&timeline, "id")? != timeline_id {
            return Err(ProjectStoreError::BadRequest(
                "Timeline ID mismatch".to_owned(),
            ));
        }
        let (_path, _guard) = self.lock_project(project_id)?;
        self.get_timeline(project_id, timeline_id)?;
        self.save_timeline(project_id, timeline)
    }

    /// The same project lock covers delivery read, merge, validation and CAS write.
    pub fn deliver_film_timeline(
        &self,
        project_id: &str,
        timeline_id: &str,
        delivery: Value,
    ) -> ProjectStoreResult<Value> {
        let (_path, _guard) = self.lock_project(project_id)?;
        let mut timeline = self.get_timeline(project_id, timeline_id)?;
        if let Some(expected) = delivery.get("expectedRevision").and_then(Value::as_u64) {
            let current = crate::film_timeline::revision(&timeline);
            if expected != current {
                return Err(ProjectStoreError::TimelineConflict {
                    code: "timeline_revision_conflict",
                    context: json!({"timelineId": timeline_id, "expectedRevision": expected, "currentRevision": current}),
                });
            }
        }
        let before = timeline.clone();
        crate::film_timeline::deliver(&mut timeline, delivery, |asset_id| {
            let asset = self.get_asset(project_id, asset_id)?;
            asset
                .get("file")
                .and_then(|file| file.get("duration"))
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| {
                    ProjectStoreError::BadRequest(format!(
                        "Asset {asset_id} has no measured duration"
                    ))
                })
        })?;
        if timeline == before {
            return Ok(before);
        }
        self.save_existing_timeline(project_id, timeline_id, timeline)
    }

    /// Freeze the document the export worker will actually read. A later edit cannot change it.
    pub fn snapshot_timeline_export(
        &self,
        project_id: &str,
        timeline_id: &str,
    ) -> ProjectStoreResult<TimelineFileDocument> {
        let (project_path, _guard) = self.lock_project(project_id)?;
        let mut result = self.timeline_file_and_document(project_id, timeline_id)?;
        let dir = project_path.join("timeline-exports");
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}_{}.json", timeline_id, random_hex(16)?));
        write_json(&path, &result.document)?;
        result.file.relative_path = relative_string(&project_path, &path)?;
        result.file.path = path;
        Ok(result)
    }

    pub fn timeline_file(
        &self,
        project_id: &str,
        timeline_id: &str,
    ) -> ProjectStoreResult<TimelineFile> {
        let project_path = self.find_project_path(project_id)?;
        find_timeline_file(&project_path, timeline_id)
    }

    pub fn timeline_file_and_document(
        &self,
        project_id: &str,
        timeline_id: &str,
    ) -> ProjectStoreResult<TimelineFileDocument> {
        let project_path = self.find_project_path(project_id)?;
        let file = find_timeline_file(&project_path, timeline_id)?;
        let document = read_json(&file.path)?;
        Ok(TimelineFileDocument { file, document })
    }
}
