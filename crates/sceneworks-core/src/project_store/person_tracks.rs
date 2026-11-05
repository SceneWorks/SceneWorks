//! ProjectStore person tracks aggregate.

use super::*;

fn normalize_person_track(project_path: &Path, track_path: &Path) -> ProjectStoreResult<Value> {
    let mut track = read_json(track_path)?;
    let track_rel = relative_string(project_path, track_path)?;
    if let Some(object) = track.as_object_mut() {
        object.insert("path".to_owned(), Value::String(track_rel));
    }
    Ok(track)
}

/// Validate and canonicalize the incoming correction set for a person track.
/// Each entry targets an in-range frame and carries a box adjustment, a
/// rejection, or both; the server stamps author/createdAt/source. Entries that
/// neither adjust a box nor reject the frame are dropped (a reset frame).
/// Duplicate frame indices keep the last occurrence so the set is one entry per
/// frame.
fn normalize_person_track_corrections(
    corrections: Vec<Value>,
    frame_count: usize,
) -> ProjectStoreResult<Vec<Value>> {
    let created_at = utc_now();
    let mut normalized: Vec<(u64, Value)> = Vec::with_capacity(corrections.len());
    for correction in corrections {
        let object = correction.as_object().ok_or_else(|| {
            ProjectStoreError::BadRequest("Each correction must be an object".to_owned())
        })?;
        let frame_index = object
            .get("frameIndex")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest(
                    "Correction frameIndex must be a non-negative integer".to_owned(),
                )
            })?;
        if frame_count == 0 || frame_index as usize >= frame_count {
            return Err(ProjectStoreError::BadRequest(format!(
                "Correction frameIndex {frame_index} is outside the track's {frame_count} frames"
            )));
        }
        let rejected = object
            .get("rejected")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let box_value = match object.get("box") {
            Some(Value::Null) | None => None,
            Some(value) => Some(validate_normalized_box(value)?),
        };
        if box_value.is_none() && !rejected {
            continue;
        }
        let author = object
            .get("author")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("ui")
            .to_owned();
        let source = object
            .get("source")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("manual")
            .to_owned();
        let mut entry = serde_json::Map::new();
        entry.insert("frameIndex".to_owned(), json!(frame_index));
        if let Some(box_value) = box_value {
            entry.insert("box".to_owned(), box_value);
        }
        entry.insert("rejected".to_owned(), Value::Bool(rejected));
        entry.insert("author".to_owned(), Value::String(author));
        entry.insert("source".to_owned(), Value::String(source));
        entry.insert("createdAt".to_owned(), Value::String(created_at.clone()));
        if let Some(slot) = normalized
            .iter_mut()
            .find(|(index, _)| *index == frame_index)
        {
            slot.1 = Value::Object(entry);
        } else {
            normalized.push((frame_index, Value::Object(entry)));
        }
    }
    Ok(normalized.into_iter().map(|(_, value)| value).collect())
}

/// Validate a normalized 0..1 bounding box and return a canonical copy holding
/// only x/y/width/height as finite numbers in range with positive extent.
fn validate_normalized_box(value: &Value) -> ProjectStoreResult<Value> {
    let object = value.as_object().ok_or_else(|| {
        ProjectStoreError::BadRequest("Correction box must be an object".to_owned())
    })?;
    let mut box_out = serde_json::Map::new();
    for key in ["x", "y", "width", "height"] {
        let component = object.get(key).and_then(Value::as_f64).ok_or_else(|| {
            ProjectStoreError::BadRequest(format!("Correction box.{key} must be a number"))
        })?;
        if !component.is_finite() || !(0.0..=1.0).contains(&component) {
            return Err(ProjectStoreError::BadRequest(format!(
                "Correction box.{key} must be between 0 and 1"
            )));
        }
        box_out.insert(key.to_owned(), json!(component));
    }
    let width = box_out.get("width").and_then(Value::as_f64).unwrap_or(0.0);
    let height = box_out.get("height").and_then(Value::as_f64).unwrap_or(0.0);
    if width <= 0.0 || height <= 0.0 {
        return Err(ProjectStoreError::BadRequest(
            "Correction box must have positive width and height".to_owned(),
        ));
    }
    Ok(Value::Object(box_out))
}

impl ProjectStore {
    pub fn list_person_tracks(&self, project_id: &str) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        let folder = project_path.join("person-tracks");
        if !folder.exists() {
            return Ok(Vec::new());
        }

        let mut tracks = Vec::new();
        for entry in read_dir_paths(&folder)? {
            if !entry
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(".sceneworks.person-track.json"))
            {
                continue;
            }
            if let Ok(track) = normalize_person_track(&project_path, &entry) {
                tracks.push(track);
            }
        }
        tracks.sort_by(|left, right| {
            right
                .get("createdAt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .cmp(
                    left.get("createdAt")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )
        });
        Ok(tracks)
    }

    pub fn get_person_track(&self, project_id: &str, track_id: &str) -> ProjectStoreResult<Value> {
        if !is_safe_id(track_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let tracks_dir = project_path.join("person-tracks");
        let track_path = project_path
            .join("person-tracks")
            .join(format!("{track_id}.sceneworks.person-track.json"));
        if !track_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Person track not found".to_owned(),
            ));
        }
        let tracks_root = fs::canonicalize(&tracks_dir)?;
        let resolved_path = fs::canonicalize(&track_path)?;
        if !resolved_path.starts_with(&tracks_root) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        normalize_person_track(&project_path, &track_path)
    }

    /// Replace the correction set on a person track (sc-1485). Each incoming
    /// correction targets a frame by index and either adjusts its box or rejects
    /// the frame; the server stamps author/createdAt/source so the sidecar always
    /// records who corrected it and when. The corrections array is the UI's full
    /// view, so it is written wholesale (a frame omitted from the set is treated
    /// as having no correction). Returns the normalized, updated track.
    pub fn set_person_track_corrections(
        &self,
        project_id: &str,
        track_id: &str,
        corrections: Vec<Value>,
    ) -> ProjectStoreResult<Value> {
        if !is_safe_id(track_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let tracks_dir = project_path.join("person-tracks");
        let track_path = tracks_dir.join(format!("{track_id}.sceneworks.person-track.json"));
        if !track_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Person track not found".to_owned(),
            ));
        }
        let tracks_root = fs::canonicalize(&tracks_dir)?;
        let resolved_path = fs::canonicalize(&track_path)?;
        if !resolved_path.starts_with(&tracks_root) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }

        let mut track = read_json(&track_path)?;
        let frame_count = track
            .get("frames")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let mut normalized = normalize_person_track_corrections(corrections, frame_count)?;
        let has_corrections = !normalized.is_empty();
        normalized.sort_by_key(|correction| {
            correction
                .get("frameIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        });

        let object = track.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Person track sidecar must be an object".to_owned())
        })?;
        object.insert("corrections".to_owned(), Value::Array(normalized));
        let status = object
            .entry("status")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Person track status must be an object".to_owned())
            })?;
        status.insert(
            "correctionState".to_owned(),
            Value::String(
                if has_corrections {
                    "box_corrections_applied"
                } else {
                    "ready_for_box_corrections"
                }
                .to_owned(),
            ),
        );

        write_json(&track_path, &track)?;
        normalize_person_track(&project_path, &track_path)
    }
}
