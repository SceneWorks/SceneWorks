//! Narrow, idempotent film deliveries over the authoritative saved cut.
use crate::project_store::{ProjectStoreError, ProjectStoreResult};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn revision(timeline: &Value) -> u64 {
    timeline
        .get("revision")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}
pub fn validate_metadata(timeline: &Value) -> ProjectStoreResult<()> {
    let Some(assembly) = timeline.get("filmAssembly").filter(|v| !v.is_null()) else {
        return Ok(());
    };
    let invalid = || {
        ProjectStoreError::BadRequest("filmAssembly must contain object runs with array shotOrder and object delivery metadata".into())
    };
    if !assembly.is_object() {
        return Err(invalid());
    }
    if let Some(runs) = assembly.get("runs").filter(|v| !v.is_null()) {
        let runs = runs.as_object().ok_or_else(invalid)?;
        for data in runs.values() {
            if !data.is_object() {
                return Err(invalid());
            }
            for key in [
                "deletedShots",
                "trimConflicts",
                "audioDelivered",
                "appliedDeliveries",
            ] {
                if data
                    .get(key)
                    .is_some_and(|v| !v.is_null() && !v.is_object())
                {
                    return Err(invalid());
                }
            }
            if let Some(order) = data.get("shotOrder") {
                let order = order.as_array().ok_or_else(invalid)?;
                if order.iter().any(|v| v.as_str().is_none()) {
                    return Err(invalid());
                }
            }
        }
    }
    Ok(())
}

fn field<'a>(item: &'a Value, key: &str) -> &'a str {
    item.get("filmHarness")
        .and_then(|v| v.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}
fn number(item: &Value, key: &str) -> f64 {
    item[key].as_f64().unwrap_or(0.0)
}
fn pictures(timeline: &Value) -> Vec<&Value> {
    timeline["tracks"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|track| track["items"].as_array().into_iter().flatten())
        .filter(|item| field(item, "role") == "picture" && !field(item, "runId").is_empty())
        .collect()
}

fn picture_duration(timeline: &Value) -> f64 {
    timeline["tracks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|track| track["kind"] == "video")
        .flat_map(|track| track["items"].as_array().into_iter().flatten())
        .map(|item| number(item, "timelineEnd"))
        .fold(0.0, f64::max)
}

/// Extend a run-owned sequence bed only when its placement and source trim still describe the
/// complete pre-delivery cut. Gain, mute, fades and item volume are independent editor controls and
/// deliberately do not participate in this test. An explicit placement or duration edit prevents
/// automatic extension, preserving the editor's chosen bounds.
fn extend_unedited_sequence_beds(
    timeline: &mut Value,
    incoming: &Value,
    run: &str,
    old_duration: f64,
    new_duration: f64,
) {
    if new_duration <= old_duration {
        return;
    }
    let close = |left: f64, right: f64| (left - right).abs() <= 0.000_001;
    for track in timeline["tracks"]
        .as_array_mut()
        .into_iter()
        .flatten()
        .filter(|track| track["kind"] == "audio")
    {
        for item in track["items"].as_array_mut().into_iter().flatten() {
            let role = field(item, "role");
            if field(item, "runId") != run
                || !matches!(role, "ambience" | "music" | "sfx")
                || !field(item, "shotId").is_empty()
            {
                continue;
            }
            let declared_start = item["filmHarness"]["startSeconds"]
                .as_f64()
                .unwrap_or(0.0)
                .max(0.0);
            let source_in = number(item, "sourceIn");
            let expected_span = (old_duration - declared_start).max(0.0);
            let item_id = field(item, "id");
            let desired = incoming["tracks"]
                .as_array()
                .into_iter()
                .flatten()
                .flat_map(|track| track["items"].as_array().into_iter().flatten())
                .find(|candidate| {
                    field(candidate, "runId") == run
                        && field(candidate, "id") == item_id
                        && field(candidate, "role") == role
                        && field(candidate, "shotId").is_empty()
                });
            let untouched = close(number(item, "timelineStart"), declared_start)
                && close(number(item, "timelineEnd"), old_duration)
                && close(number(item, "sourceOut") - source_in, expected_span);
            let desired_is_full_cut = desired.is_some_and(|desired| {
                close(number(desired, "timelineStart"), declared_start)
                    && close(number(desired, "timelineEnd"), new_duration)
                    && close(
                        number(desired, "sourceOut") - number(desired, "sourceIn"),
                        (new_duration - declared_start).max(0.0),
                    )
            });
            if untouched && desired_is_full_cut {
                item["timelineEnd"] = json!(new_duration);
                item["sourceOut"] = json!(source_in + (new_duration - declared_start));
            }
        }
    }
}

/// Saving a cut is also the durable deletion/restore and ordering operation. CAS has already
/// established that this user actually edited the current document, not a pre-delivery snapshot.
pub fn reconcile_cut(previous: &Value, next: &mut Value, next_revision: u64) {
    if next.get("filmAssembly").is_none() && previous.get("filmAssembly").is_some() {
        next["filmAssembly"] = previous["filmAssembly"].clone();
    }
    if !next.get("filmAssembly").is_some_and(Value::is_object) {
        return;
    }
    let old = pictures(previous);
    let present: Vec<(String, String, String, f64)> = pictures(next)
        .iter()
        .map(|v| {
            (
                field(v, "runId").into(),
                field(v, "shotId").into(),
                v["id"].as_str().unwrap_or("").into(),
                number(v, "timelineStart"),
            )
        })
        .collect();
    for item in old {
        let run = field(item, "runId");
        let shot = field(item, "shotId");
        if !present.iter().any(|(r, s, _, _)| r == run && s == shot) {
            next["filmAssembly"]["runs"][run]["deletedShots"][shot] =
                json!({"itemId":item["id"], "deletedRevision":next_revision});
        }
    }
    if let Some(runs) = previous["filmAssembly"]["runs"].as_object() {
        for (run, data) in runs {
            if let Some(deleted) = data["deletedShots"].as_object() {
                for (shot, stamp) in deleted {
                    if !present.iter().any(|(r, s, _, _)| r == run && s == shot) {
                        next["filmAssembly"]["runs"][run]["deletedShots"][shot] = stamp.clone();
                    }
                }
            }
        }
    }
    for (run, shot, _, _) in &present {
        if let Some(deleted) = next["filmAssembly"]["runs"][run]["deletedShots"].as_object_mut() {
            deleted.remove(shot);
        }
    }
    if let Some(runs) = next
        .get_mut("filmAssembly")
        .and_then(|assembly| assembly.get_mut("runs"))
        .and_then(Value::as_object_mut)
    {
        for (run, data) in runs {
            let mut ordered: Vec<_> = present.iter().filter(|(r, _, _, _)| r == run).collect();
            ordered.sort_by(|a, b| a.3.total_cmp(&b.3));
            let mut cursor = ordered.iter();
            if let Some(order) = data["shotOrder"].as_array_mut() {
                for slot in order {
                    if ordered
                        .iter()
                        .any(|(_, shot, _, _)| slot.as_str() == Some(shot))
                    {
                        if let Some((_, shot, _, _)) = cursor.next() {
                            *slot = json!(shot);
                        }
                    }
                }
            }
        }
    }
}

/// The caller holds the project lock and resolves duration from project-owned asset metadata.
/// Conflicting trims are retained in the timeline so interrupted clients can reopen and resolve
/// them. No partially applied replacement, implicit clamping or automatic export is possible.
pub fn deliver(
    timeline: &mut Value,
    payload: Value,
    mut duration: impl FnMut(&str) -> ProjectStoreResult<f64>,
) -> ProjectStoreResult<()> {
    validate_metadata(timeline)?;
    let run = payload["runId"]
        .as_str()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| ProjectStoreError::BadRequest("runId is required".into()))?
        .to_owned();
    let mut incoming = payload["timeline"].clone();
    let resolve = payload["resolveShotId"].as_str();
    let choice = payload["resolution"].as_str();
    if let Some(shot) = resolve {
        if payload["expectedRevision"].as_u64().is_none() {
            return Err(ProjectStoreError::BadRequest(
                "Resolving a trim requires expectedRevision".into(),
            ));
        }
        let pending = timeline["filmAssembly"]["runs"][&run]["trimConflicts"][shot].clone();
        if pending.is_null() {
            return Err(ProjectStoreError::BadRequest(
                "This trim conflict is no longer pending".into(),
            ));
        }
        if ![Some("clamp"), Some("reset"), Some("keepCurrent")].contains(&choice) {
            return Err(ProjectStoreError::BadRequest(
                "Choose clamp, reset or keepCurrent".into(),
            ));
        }
        incoming = json!({"tracks":[{"items":[pending["take"].clone()]}]});
    }
    let incoming_items: Vec<Value> = incoming["tracks"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|t| t["items"].as_array().into_iter().flatten())
        .filter(|item| field(item, "runId") == run && field(item, "role") == "picture")
        .cloned()
        .collect();
    let order = payload["shotOrder"].as_array().cloned().unwrap_or_default();
    if timeline["filmAssembly"]["runs"][&run]["shotOrder"].is_null() {
        timeline["filmAssembly"]["runs"][&run]["shotOrder"] = json!(order);
    }
    timeline["filmAssembly"]["schemaVersion"] = json!(1);
    for mut take in incoming_items {
        let shot = field(&take, "shotId").to_owned();
        if shot.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Delivery shotId is required".into(),
            ));
        }
        if !timeline["filmAssembly"]["runs"][&run]["deletedShots"][&shot].is_null() {
            continue;
        }
        let attempt = take["filmHarness"]["attempt"].as_u64().unwrap_or(0);
        let delivery_id = format!("{run}:{shot}:a{attempt}");
        if resolve.is_none()
            && timeline["filmAssembly"]["runs"][&run]["appliedDeliveries"][&delivery_id] == true
        {
            continue;
        }
        let existing = timeline["tracks"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .find_map(|(t, track)| {
                track["items"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .position(|item| {
                        field(item, "runId") == run
                            && field(item, "shotId") == shot
                            && field(item, "role") == "picture"
                    })
                    .map(|i| (t, i))
            });
        if let Some((t, i)) = existing {
            let current = &timeline["tracks"][t]["items"][i];
            if resolve.is_none()
                && (current["filmHarness"]["deliveryId"] == delivery_id
                    || current["filmHarness"]["attempt"]
                        .as_u64()
                        .is_some_and(|aligned| aligned >= attempt))
            {
                continue;
            }
        }
        if resolve.is_none()
            && timeline["filmAssembly"]["runs"][&run]["trimConflicts"][&shot]["take"]["filmHarness"]
                ["deliveryId"]
                == delivery_id
        {
            continue;
        }
        let asset_id = take["assetId"]
            .as_str()
            .ok_or_else(|| ProjectStoreError::BadRequest("Delivery assetId is required".into()))?;
        let length = duration(asset_id)?;
        take["filmHarness"]["deliveryId"] = json!(delivery_id);
        take["filmHarness"]["alignedDurationSeconds"] = json!(length);
        if let Some((t, i)) = existing {
            let current = &timeline["tracks"][t]["items"][i];
            let source_in = number(current, "sourceIn");
            let source_out = number(current, "sourceOut");
            let incompatible =
                source_in < 0.0 || source_out <= source_in || source_out > length + 0.000001;
            if incompatible && resolve != Some(shot.as_str()) {
                let mut choices = vec!["reset", "keepCurrent"];
                if length - source_in >= 0.1 {
                    choices.insert(0, "clamp");
                }
                timeline["filmAssembly"]["runs"][&run]["trimConflicts"][&shot] = json!({
                    "code":"timeline_replacement_trim_conflict", "itemId":current["id"], "shotId":shot,
                    "sourceIn":source_in, "sourceOut":source_out, "newDuration":length, "choices":choices, "take":take});
                continue;
            }
            let current = &mut timeline["tracks"][t]["items"][i];
            if choice != Some("keepCurrent") {
                if incompatible || choice == Some("reset") {
                    let start = if choice == Some("reset") {
                        0.0
                    } else {
                        source_in
                    };
                    if length - start < 0.1 {
                        return Err(ProjectStoreError::BadRequest(
                            "This trim cannot be clamped; choose reset or keepCurrent".into(),
                        ));
                    }
                    current["sourceIn"] = json!(start);
                    current["sourceOut"] = json!(length);
                    current["timelineEnd"] = json!(
                        number(current, "timelineStart")
                            + (length - start) / current["speed"].as_f64().unwrap_or(1.0).max(0.1)
                    );
                }
                current["assetId"] = take["assetId"].clone();
                current["currentVersionAssetId"] = take["assetId"].clone();
                for key in ["versionAssetIds", "versionHistory"] {
                    if !current[key].is_array() {
                        current[key] = json!([]);
                    }
                }
                if !current["versionAssetIds"]
                    .as_array()
                    .unwrap()
                    .contains(&take["assetId"])
                {
                    current["versionAssetIds"]
                        .as_array_mut()
                        .unwrap()
                        .push(take["assetId"].clone());
                    current["versionHistory"].as_array_mut().unwrap().push(json!({"assetId":take["assetId"],"source":"replacement","jobId":take["filmHarness"]["jobId"]}));
                }
            }
            current["filmHarness"] = take["filmHarness"].clone();
            if let Some(conflicts) =
                timeline["filmAssembly"]["runs"][&run]["trimConflicts"].as_object_mut()
            {
                conflicts.remove(&shot);
            }
        } else {
            let old_duration = picture_duration(timeline);
            let order = timeline["filmAssembly"]["runs"][&run]["shotOrder"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let rank = |id: &str| {
                order
                    .iter()
                    .position(|s| s.as_str() == Some(id))
                    .unwrap_or(usize::MAX)
            };
            let tracks = timeline["tracks"]
                .as_array_mut()
                .ok_or_else(|| ProjectStoreError::BadRequest("Timeline has no tracks".into()))?;
            let t = tracks
                .iter()
                .position(|t| t["kind"] == "video")
                .ok_or_else(|| {
                    ProjectStoreError::BadRequest("Timeline has no picture track".into())
                })?;
            let track_id = tracks[t]["id"].clone();
            let items = tracks[t]["items"].as_array_mut().ok_or_else(|| {
                ProjectStoreError::BadRequest("Picture track has no items".into())
            })?;
            let start = items
                .iter()
                .filter(|item| {
                    field(item, "runId") == run && rank(field(item, "shotId")) > rank(&shot)
                })
                .map(|v| number(v, "timelineStart"))
                .min_by(f64::total_cmp)
                .unwrap_or_else(|| {
                    items
                        .iter()
                        .map(|v| number(v, "timelineEnd"))
                        .fold(0.0, f64::max)
                });
            let mut shifted = Vec::new();
            for item in items
                .iter_mut()
                .filter(|item| number(item, "timelineStart") >= start)
            {
                item["timelineStart"] = json!(number(item, "timelineStart") + length);
                item["timelineEnd"] = json!(number(item, "timelineEnd") + length);
                if field(item, "runId") == run {
                    shifted.push(field(item, "shotId").to_owned());
                }
            }
            let digest = Sha256::digest(format!("{run}\0{shot}").as_bytes());
            take["id"] = json!(format!("item_film_{:x}", digest)[..34].to_owned());
            take["trackId"] = track_id;
            take["sourceIn"] = json!(0.0);
            take["sourceOut"] = json!(length);
            take["timelineStart"] = json!(start);
            take["timelineEnd"] = json!(start + length);
            items.push(take);
            items.sort_by(|a, b| number(a, "timelineStart").total_cmp(&number(b, "timelineStart")));
            for track in tracks.iter_mut().filter(|t| t["kind"] == "audio") {
                for item in track["items"].as_array_mut().into_iter().flatten() {
                    if field(item, "runId") == run
                        && shifted.iter().any(|s| s == field(item, "shotId"))
                    {
                        item["timelineStart"] = json!(number(item, "timelineStart") + length);
                        item["timelineEnd"] = json!(number(item, "timelineEnd") + length);
                    }
                }
            }
            let new_duration = picture_duration(timeline);
            extend_unedited_sequence_beds(timeline, &incoming, &run, old_duration, new_duration);
        }
        timeline["filmAssembly"]["runs"][&run]["appliedDeliveries"][&delivery_id] = json!(true);
    }
    // Establish harness-owned lanes in their declared order even when a lane has no clip in the
    // first incremental delivery. Later dialogue then lands in its deterministic default position;
    // an existing lane is never moved, preserving an editor's track order.
    for track in incoming["tracks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|track| track["kind"] == "audio")
    {
        let Some(id) = track["id"].as_str().filter(|id| !id.is_empty()) else {
            continue;
        };
        let tracks = timeline["tracks"].as_array_mut().unwrap();
        if !tracks.iter().any(|existing| existing["id"] == id) {
            let mut empty = track.clone();
            empty["items"] = json!([]);
            tracks.push(empty);
        }
    }
    // Add each owned audio item once; its subsequent trim, gain, placement or deletion belongs
    // to the editor. Repeated assembly must never regenerate the user's audio track.
    for track in incoming["tracks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t["kind"] == "audio")
    {
        for item in track["items"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|i| field(i, "runId") == run)
        {
            let id = item["id"].as_str().unwrap_or("");
            if timeline["filmAssembly"]["runs"][&run]["audioDelivered"][id] == true {
                continue;
            }
            let shot = field(item, "shotId");
            let picture = pictures(timeline)
                .into_iter()
                .find(|i| field(i, "runId") == run && field(i, "shotId") == shot)
                .cloned();
            if !shot.is_empty() && picture.is_none() {
                continue;
            }
            let mut audio = item.clone();
            if let Some(picture) = picture {
                let length = number(item, "timelineEnd") - number(item, "timelineStart");
                let start = number(&picture, "timelineStart")
                    + item["filmHarness"]["offsetSeconds"].as_f64().unwrap_or(0.0);
                audio["timelineStart"] = json!(start);
                audio["timelineEnd"] = json!(start + length);
            }
            let tracks = timeline["tracks"].as_array_mut().unwrap();
            let t = match tracks.iter().position(|t| t["id"] == track["id"]) {
                Some(i) => i,
                None => {
                    let mut empty = track.clone();
                    empty["items"] = json!([]);
                    tracks.push(empty);
                    tracks.len() - 1
                }
            };
            let items = tracks[t]["items"].as_array_mut().unwrap();
            if !items.iter().any(|i| i["id"] == item["id"]) {
                items.push(audio);
            }
            timeline["filmAssembly"]["runs"][&run]["audioDelivered"][id] = json!(true);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cut() -> Value {
        json!({"id":"timeline_test", "revision":1,"tracks":[{"id":"main","kind":"video","items":[]},{"id":"sound","kind":"audio","gain":0.6,"items":[]}]})
    }
    fn delivery(shot: &str, attempt: u64) -> Value {
        json!({"runId":"run_test","shotOrder":["A","B","C"],"timeline":{"tracks":[{"kind":"video","items":[{"id":"ignored","assetId":format!("asset_{shot}_{attempt}"),"filmHarness":{"role":"picture","runId":"run_test","shotId":shot,"attempt":attempt},"sourceIn":0,"sourceOut":5,"timelineStart":0,"timelineEnd":5,"speed":1,"fit":"crop","volume":0.8}]}]}})
    }
    fn send(cut: &mut Value, shot: &str, attempt: u64, length: f64) {
        deliver(cut, delivery(shot, attempt), |_| Ok(length)).unwrap();
    }
    #[test]
    fn reconciling_an_ordinary_timeline_does_not_add_film_metadata() {
        let previous = cut();
        let mut next = previous.clone();
        next["name"] = json!("Edited cut");
        reconcile_cut(&previous, &mut next, 2);
        assert!(next.get("filmAssembly").is_none());

        let mut sparse = previous.clone();
        sparse["filmAssembly"] = json!({"schemaVersion": 1});
        reconcile_cut(&previous, &mut sparse, 2);
        assert_eq!(sparse["filmAssembly"], json!({"schemaVersion": 1}));
    }

    #[test]
    fn out_of_order_delivery_replay_and_user_deletion_are_idempotent() {
        let mut saved = cut();
        send(&mut saved, "C", 1, 5.0);
        send(&mut saved, "A", 1, 4.0);
        send(&mut saved, "B", 1, 3.0);
        let items = saved["tracks"][0]["items"].as_array().unwrap();
        assert_eq!(
            items.iter().map(|i| field(i, "shotId")).collect::<Vec<_>>(),
            vec!["A", "B", "C"]
        );
        assert_eq!(items[2]["timelineStart"], 7.0);
        let before = saved.clone();
        send(&mut saved, "A", 1, 4.0);
        assert_eq!(saved, before);
        saved["tracks"][0]["items"]
            .as_array_mut()
            .unwrap()
            .remove(1);
        reconcile_cut(&before, &mut saved, 2);
        let deleted = saved.clone();
        send(&mut saved, "B", 2, 4.0);
        assert_eq!(saved, deleted);
        // An explicit restore (including Undo) clears the durable tombstone.
        let mut restored = before.clone();
        reconcile_cut(&deleted, &mut restored, 3);
        assert!(restored["filmAssembly"]["runs"]["run_test"]["deletedShots"]["B"].is_null());
    }
    #[test]
    fn replacement_preserves_trim_position_audio_and_surfaces_short_take_choices() {
        let mut saved = cut();
        send(&mut saved, "A", 1, 5.0);
        send(&mut saved, "B", 1, 5.0);
        saved["tracks"][0]["items"][0]["sourceIn"] = json!(1.5);
        saved["tracks"][0]["items"][0]["sourceOut"] = json!(4.0);
        saved["tracks"][0]["items"][0]["timelineStart"] = json!(2.0);
        saved["tracks"][1]["items"] =
            json!([{"id":"music","volume":0.3,"timelineStart":1,"timelineEnd":9}]);
        let unrelated = saved["tracks"][0]["items"][1].clone();
        let audio = saved["tracks"][1].clone();
        let old = saved["tracks"][0]["items"][0].clone();
        send(&mut saved, "A", 2, 6.0);
        for key in [
            "id",
            "sourceIn",
            "sourceOut",
            "timelineStart",
            "timelineEnd",
            "speed",
            "fit",
            "volume",
        ] {
            assert_eq!(saved["tracks"][0]["items"][0][key], old[key], "{key}");
        }
        assert_eq!(saved["tracks"][0]["items"][1], unrelated);
        assert_eq!(saved["tracks"][1], audio);
        let fitted = saved.clone();
        send(&mut saved, "A", 3, 2.0);
        assert_eq!(saved["tracks"], fitted["tracks"]);
        let pending = &saved["filmAssembly"]["runs"]["run_test"]["trimConflicts"]["A"];
        assert_eq!(pending["code"], "timeline_replacement_trim_conflict");
        for (choice, expected_in, expected_asset) in [
            ("clamp", 1.5, "asset_A_3"),
            ("reset", 0.0, "asset_A_3"),
            ("keepCurrent", 1.5, "asset_A_2"),
        ] {
            let mut resolved = saved.clone();
            deliver(&mut resolved,json!({"runId":"run_test","resolveShotId":"A","resolution":choice,"expectedRevision":1}), |_| Ok(2.0)).unwrap();
            assert_eq!(resolved["tracks"][0]["items"][0]["sourceIn"], expected_in);
            assert_eq!(resolved["tracks"][0]["items"][0]["assetId"], expected_asset);
            let before_replay = resolved.clone();
            send(&mut resolved, "A", 3, 2.0);
            assert_eq!(resolved, before_replay);
        }
    }
    #[test]
    fn pending_slots_follow_user_order_without_rebuilding_existing_items() {
        let mut saved = cut();
        send(&mut saved, "A", 1, 4.0);
        send(&mut saved, "C", 1, 5.0);
        let previous = saved.clone();
        saved["tracks"][0]["items"][0]["timelineStart"] = json!(5.0);
        saved["tracks"][0]["items"][0]["timelineEnd"] = json!(9.0);
        saved["tracks"][0]["items"][1]["timelineStart"] = json!(0.0);
        saved["tracks"][0]["items"][1]["timelineEnd"] = json!(5.0);
        reconcile_cut(&previous, &mut saved, 2);
        assert_eq!(
            saved["filmAssembly"]["runs"]["run_test"]["shotOrder"],
            json!(["C", "B", "A"])
        );
        send(&mut saved, "B", 1, 2.0);
        assert_eq!(saved["tracks"][0]["items"][0]["filmHarness"]["shotId"], "C");
        assert_eq!(saved["tracks"][0]["items"][2]["timelineStart"], 7.0);
    }

    #[test]
    fn later_delivery_extends_only_untrimmed_beds_and_keeps_lane_and_mix_edits() {
        let mut saved = cut();
        let delivery = |shot: &str, shot_length: f64, bed_end: f64, dialogue: bool| {
            let dialogue_items = if dialogue {
                json!([{"id":"line_b","trackId":"dialogue","assetId":"line","timelineStart":4.2,
                    "timelineEnd":5.2,"sourceIn":0,"sourceOut":1,
                    "filmHarness":{"role":"dialogue","runId":"run_test","shotId":"B","offsetSeconds":0.2}}])
            } else {
                json!([])
            };
            json!({"runId":"run_test","shotOrder":["A","B","C"],"timeline":{"tracks":[
                {"id":"main","kind":"video","items":[{"id":"take","assetId":format!("asset_{shot}_1"),
                    "filmHarness":{"role":"picture","runId":"run_test","shotId":shot,"attempt":1},
                    "sourceIn":0,"sourceOut":shot_length,"timelineStart":0,"timelineEnd":shot_length,"speed":1}]},
                {"id":"dialogue","kind":"audio","role":"dialogue","gain":1,"muted":false,"items":dialogue_items},
                {"id":"ambience","kind":"audio","role":"ambience","gain":0.35,"muted":false,"items":[
                    {"id":"room","assetId":"room","timelineStart":0,"timelineEnd":bed_end,"sourceIn":0,
                     "sourceOut":bed_end,"volume":1,"fadeInSeconds":1,"fadeOutSeconds":1.5,
                     "filmHarness":{"role":"ambience","runId":"run_test","startSeconds":0}}]},
                {"id":"music","kind":"audio","role":"music","gain":0.2,"muted":false,"items":[]}
            ]}})
        };

        deliver(&mut saved, delivery("A", 4.0, 4.0, false), |_| Ok(4.0)).unwrap();
        let ids = saved["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|track| track["id"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["main", "sound", "dialogue", "ambience", "music"]);
        let ambience = saved["tracks"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|track| track["id"] == "ambience")
            .unwrap();
        ambience["gain"] = json!(0.6);
        ambience["muted"] = json!(true);
        ambience["items"][0]["volume"] = json!(0.4);
        ambience["items"][0]["fadeInSeconds"] = json!(0.25);

        deliver(&mut saved, delivery("B", 3.0, 7.0, true), |_| Ok(3.0)).unwrap();
        let ambience = saved["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|track| track["id"] == "ambience")
            .unwrap();
        assert_eq!(ambience["gain"], 0.6);
        assert_eq!(ambience["muted"], true);
        assert_eq!(ambience["items"][0]["volume"], 0.4);
        assert_eq!(ambience["items"][0]["fadeInSeconds"], 0.25);
        assert_eq!(ambience["items"][0]["timelineEnd"], 7.0);
        assert_eq!(ambience["items"][0]["sourceOut"], 7.0);
        let ids = saved["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|track| track["id"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids[2..], ["dialogue", "ambience", "music"]);

        let ambience = saved["tracks"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|track| track["id"] == "ambience")
            .unwrap();
        ambience["items"][0]["timelineEnd"] = json!(6.0);
        ambience["items"][0]["sourceOut"] = json!(6.0);
        deliver(&mut saved, delivery("C", 2.0, 9.0, false), |_| Ok(2.0)).unwrap();
        let ambience = saved["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|track| track["id"] == "ambience")
            .unwrap();
        assert_eq!(
            ambience["items"][0]["timelineEnd"], 6.0,
            "an explicit bed trim is preserved when a later shot arrives"
        );
        assert_eq!(ambience["items"][0]["sourceOut"], 6.0);
    }
}
