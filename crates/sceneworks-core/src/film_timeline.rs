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

/// Saving a cut is also the durable deletion/restore and ordering operation. CAS has already
/// established that this user actually edited the current document, not a pre-delivery snapshot.
pub fn reconcile_cut(previous: &Value, next: &mut Value, next_revision: u64) {
    if next.get("filmAssembly").is_none() && previous.get("filmAssembly").is_some() {
        next["filmAssembly"] = previous["filmAssembly"].clone();
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
    if let Some(runs) = next["filmAssembly"]["runs"].as_object_mut() {
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
        }
        timeline["filmAssembly"]["runs"][&run]["appliedDeliveries"][&delivery_id] = json!(true);
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
}
