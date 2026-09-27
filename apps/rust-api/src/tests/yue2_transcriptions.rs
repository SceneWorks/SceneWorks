//! sc-23002: recording → transcription → cover through the real HTTP routes over the LIVE builtin
//! YuE2 entry — the cover closure's acquisition door, the `transcribe` job's submission (licence
//! gate, closure presence, the recording pinned by its media digest), the terminal side effect that
//! records the review artifact and imports its scores as versions linked to the recording, the
//! review reads and exports, and a cover of the reviewed version naming its source recording.
use super::support::*;
use super::yue2_catalog::{app_with_yue1_and_yue2, builtin_yue2, seed_snapshot};
use super::yue2_jobs::{claim, finish, job, project, register, submit, submit_ok, WORKER};
use sha2::Digest;

const MELODY: &str =
    include_str!("../../../../crates/sceneworks-core/src/yue2_score/fixtures/melody.abc");
const SCORE: &str =
    include_str!("../../../../crates/sceneworks-core/src/yue2_score/fixtures/score.abc");

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(bytes))
}

fn cover_rows() -> Vec<Value> {
    builtin_yue2()["conditionalComponents"]
        .as_array()
        .unwrap()
        .clone()
}

/// A 16-bit mono WAV of `samples` zero samples.
fn wav(samples: usize) -> Vec<u8> {
    let data = samples * 2;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&((36 + data) as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&24_000u32.to_le_bytes());
    bytes.extend_from_slice(&48_000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data as u32).to_le_bytes());
    bytes.resize(44 + data, 0);
    bytes
}

async fn upload_recording(app: &axum::Router, project_id: &str, bytes: &[u8]) -> String {
    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "take.wav",
        "audio/wav",
        bytes,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{asset}");
    assert_eq!(asset["type"], "audio", "{asset}");
    asset["id"].as_str().unwrap().to_owned()
}

async fn project_path(app: &axum::Router, project_id: &str) -> std::path::PathBuf {
    let (status, project) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{project}");
    std::path::PathBuf::from(project["path"].as_str().expect("project path"))
}

/// The review artifact `candle_audio_sheetsage2::review::Transcription::save` writes, reduced to
/// the fields the import reads: every file first, then `transcription.json` naming each digest.
fn write_artifact(dir: &std::path::Path, recording_sha: &str, melody_ready: bool) -> String {
    std::fs::create_dir_all(dir).unwrap();
    let mut files: Vec<(&str, Vec<u8>)> = vec![
        ("score.abc", SCORE.as_bytes().to_vec()),
        ("chord.lab", b"0.000\t2.000\tC:maj\n".to_vec()),
        ("melody_full.lab", b"0.000\t0.500\t64\n".to_vec()),
        ("transcription.mid", b"MThd-bytes".to_vec()),
    ];
    if melody_ready {
        files.push(("score_melody.abc", MELODY.as_bytes().to_vec()));
    }
    let mut artifacts = serde_json::Map::new();
    for (rel, bytes) in &files {
        std::fs::write(dir.join(rel), bytes).unwrap();
        artifacts.insert((*rel).to_owned(), json!(sha(bytes)));
    }
    let readiness = |ready: bool| {
        if ready {
            json!({"ready": true})
        } else {
            json!({"ready": false, "reason": "the transcription has no melody notes"})
        }
    };
    let manifest = json!({
        "schema": "sceneworks-sheetsage2-transcription-v1",
        "source": {"sha256": "ab", "samples": 24000, "sample_rate": 24000, "duration_seconds": 1.0,
                   "name": "take.wav", "original_sha256": recording_sha, "conversion": "none"},
        "settings": {"overlap_seconds": 200.0, "lookahead_seconds": 100.0, "max_seconds": null},
        "closure": {"device": "cpu"},
        "abc_error": {"full": null, "melody": null},
        "decode_warnings": [],
        "octave_evidence": {"notes_checked": 10, "fraction_f0_half_dominant": 0.8},
        "review": {
            "voices": {"vocal": {"notes": 20}, "instrumental": {"notes": 0}},
            "distinct_chords": ["C:maj"], "keys": ["C:major"], "sections": ["verse"], "bars": 8,
            "warnings": [{"code": "octave_f0_half_evidence", "message": "8 of 10 checked vocal notes"}],
            "cover": {"melody": readiness(melody_ready), "full": readiness(melody_ready)},
        },
        "artifacts": artifacts,
    });
    let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    std::fs::write(dir.join("transcription.json"), &bytes).unwrap();
    sha(&bytes)
}

/// The acquisition door: licence-gated exactly like the install (403 without the acknowledgment,
/// recorded server-side with it), it queues each cover component at its pinned revision and file
/// list — and never anything else — and the catalog reports the purpose installed only once both
/// are on disk. Mutations that red this: dropping the acknowledgment check (201 without it), or
/// dropping `apply_conditional_component_state` (no `conditionalPurposes`).
#[tokio::test]
async fn the_cover_closure_installs_only_through_its_licence_gated_door() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let route = "/api/v1/models/yue2/conditional-components/cover/download";
    let (status, response) = request(app.clone(), "POST", route, json!({})).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{response}");
    assert_eq!(response["code"], "license_acknowledgment_required");
    let (status, response) = request(
        app.clone(),
        "POST",
        "/api/v1/models/yue2/conditional-components/generation/download",
        json!({"licenseAcknowledged": true}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{response}");
    assert_eq!(response["code"], "conditional_components_not_declared");

    let (status, response) = request(
        app.clone(),
        "POST",
        route,
        json!({"licenseAcknowledged": true}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{response}");
    let jobs = response["jobs"].as_array().unwrap();
    assert_eq!(jobs.len(), 2, "{response}");
    for (queued, row) in jobs.iter().zip(cover_rows()) {
        let payload = &queued["payload"];
        assert_eq!(queued["type"], "model_download");
        assert_eq!(payload["repo"], row["repo"]);
        assert_eq!(payload["revision"], row["revision"]);
        assert_eq!(payload["files"], row["files"]);
        assert_eq!(payload["licenseAcknowledged"], true);
        // A cover component is never the model's primary download.
        assert!(payload.get("variant").is_none(), "{payload}");
        assert!(payload.get("family").is_none(), "{payload}");
    }
    let (_, ack) = request(
        app.clone(),
        "GET",
        "/api/v1/models/yue2/license-acknowledgment",
        Value::Null,
    )
    .await;
    assert_eq!(ack["acknowledged"], true, "{ack}");

    // A fresh app per read: the catalog snapshot is cached per app, and the seeding below happens
    // behind its back.
    let catalog_state = || async {
        let app = app_with_yue1_and_yue2(&temp_dir);
        let (_, models) = request(app, "GET", "/api/v1/models", Value::Null).await;
        models
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == "yue2")
            .unwrap()
            .clone()
    };
    let yue2 = catalog_state().await;
    assert_eq!(
        yue2["conditionalPurposes"]["cover"]["installState"],
        "missing"
    );
    seed_snapshot(&temp_dir, &cover_rows()[0]);
    let yue2 = catalog_state().await;
    assert_eq!(
        yue2["conditionalPurposes"]["cover"]["installState"], "missing",
        "one of two components is not the purpose"
    );
    assert_eq!(
        yue2["conditionalComponents"][0]["installState"],
        "installed"
    );
    seed_snapshot(&temp_dir, &cover_rows()[1]);
    let yue2 = catalog_state().await;
    assert_eq!(
        yue2["conditionalPurposes"]["cover"]["installState"],
        "installed"
    );
    assert_eq!(yue2["conditionalPurposes"]["cover"]["blocked"], false);
    // Installed components are reported, never re-fetched.
    let app = app_with_yue1_and_yue2(&temp_dir);
    let (status, response) = request(
        app.clone(),
        "POST",
        route,
        json!({"licenseAcknowledged": true}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{response}");
    assert!(
        response["jobs"].as_array().unwrap().is_empty(),
        "{response}"
    );
    assert!(response["components"]
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["status"] == "installed"));
    // YuE2's own install never queues a cover component.
    let (_, all) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(all.as_array().unwrap().len(), 2);
}

/// AT2 end to end through the API: a transcribe job queues only with the licence acknowledgment and
/// the closure installed, pins the recording by its media file's SHA-256 and maps its settings; its
/// completion records the transcription and imports BOTH scores as versions linked to the
/// recording and carrying the usage policy; the review reads and exports verify their digests; and a
/// cover of the (edited) melody version names the recording. Mutations that red this: dropping the
/// `Transcribe` arm of `resolve_sources` (no recording), or the transcription import side effect
/// (no versions), or `version_source_with_transcription`'s link (the cover names no recording).
#[tokio::test]
async fn a_recording_transcribes_into_linked_versions_that_a_cover_reviews() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    for row in cover_rows() {
        seed_snapshot(&temp_dir, &row);
    }
    let project_id = project(&app).await;
    register(&app, WORKER).await;
    let recording = wav(24_000);
    let asset_id = upload_recording(&app, &project_id, &recording).await;
    let body = json!({
        "kind": "transcribe", "sourceAudioAssetId": asset_id,
        "transcription": {"maxSeconds": 30.5, "overlapSeconds": 150, "lookaheadSeconds": 50},
    });
    let (status, response) = submit(&app, &project_id, body.clone()).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{response}");
    assert_eq!(response["code"], "license_acknowledgment_required");
    let mut acknowledged = body.clone();
    acknowledged["licenseAcknowledged"] = json!(true);
    // A foreign asset is refused.
    let mut foreign = acknowledged.clone();
    foreign["sourceAudioAssetId"] = json!("asset_nope");
    let (status, response) = submit(&app, &project_id, foreign).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{response}");
    assert_eq!(response["code"], "yue2_source_unavailable");

    let jobs = submit_ok(&app, &project_id, acknowledged).await;
    assert_eq!(jobs.len(), 1);
    let transcribe_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let block = &jobs[0]["payload"]["yue2"];
    assert_eq!(block["kind"], "transcribe");
    assert_eq!(
        block["transcription"],
        json!({"maxSeconds": 30.5, "overlapSeconds": 150.0, "lookaheadSeconds": 50.0})
    );
    assert_eq!(block["sources"]["recording"]["assetId"], asset_id);
    assert_eq!(block["sources"]["recording"]["sha256"], sha(&recording));
    assert_eq!(jobs[0]["payload"]["usagePolicy"]["nonCommercial"], true);
    let run_id = block["runId"].as_str().unwrap().to_owned();
    let claimed = claim(&app, WORKER).await;
    assert_eq!(claimed["id"], transcribe_id);

    // The worker persisted the artifact and reports where; the API re-reads it.
    let dir_rel = format!("yue2/transcriptions/{run_id}");
    let manifest_sha = write_artifact(
        &project_path(&app, &project_id).await.join(&dir_rel),
        &sha(&recording),
        true,
    );
    finish(
        &app,
        &transcribe_id,
        "completed",
        json!({"yue2": {
            "status": "completed", "kind": "transcribe", "runId": run_id,
            "transcription": {
                "dir": dir_rel, "manifestSha256": manifest_sha, "device": "cpu",
                "replay": {"artifactsMatched": 5, "windows": 1},
                "unload": {"released": true, "parameterBytes": 123, "liveModelsAfter": 0},
            },
        }}),
        None,
    )
    .await;
    let finished = job(&app, &transcribe_id).await;
    let result = &finished["result"]["yue2"];
    assert!(result.get("sideEffectErrors").is_none(), "{result}");
    let transcription_id = result["transcriptionId"].as_str().unwrap().to_owned();
    let melody_id = result["scoreVersionIds"]["melody"]
        .as_str()
        .unwrap()
        .to_owned();
    let full_id = result["scoreVersionIds"]["full"]
        .as_str()
        .unwrap()
        .to_owned();

    let (status, detail) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/yue2/transcriptions/{transcription_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{detail}");
    let record = &detail["transcription"];
    assert_eq!(record["sourceAudioAssetId"], asset_id);
    assert_eq!(record["device"], "cpu");
    assert_eq!(record["jobId"], transcribe_id);
    assert_eq!(record["unload"]["released"], true);
    assert_eq!(record["warnings"][0]["code"], "octave_f0_half_evidence");
    assert_eq!(record["usagePolicy"]["modelId"], "yue2");
    assert_eq!(detail["scores"]["melody"], MELODY);
    assert_eq!(detail["scores"]["full"], SCORE);
    let (status, headers, bytes) = request_raw(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/yue2/transcriptions/{transcription_id}/files/transcription.mid"
        ),
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "audio/midi");
    assert_eq!(bytes, b"MThd-bytes".as_slice());

    for (id, cot) in [(&melody_id, "melody"), (&full_id, "full")] {
        let (_, version) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/yue2/score-versions/{id}"),
            Value::Null,
        )
        .await;
        let version = &version["version"];
        assert_eq!(version["origin"], "transcription");
        assert_eq!(version["request"]["cot"], cot);
        assert_eq!(version["transcription"]["sourceAudioAssetId"], asset_id);
        assert_eq!(
            version["transcription"]["transcriptionId"],
            transcription_id
        );
        assert_eq!(version["usagePolicy"]["nonCommercial"], true);
    }

    // Review: edit the melody version, then cover the edited version.
    let (status, edited) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/yue2/score-versions/{melody_id}/edits"),
        json!({"operation": {"op": "set_tempo", "bpm": 92}, "brief": "a touch faster",
               "provenance": {"actor": "user", "channel": "ui"}}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{edited}");
    let edited = &edited["version"];
    assert_eq!(edited["transcription"]["sourceAudioAssetId"], asset_id);
    assert_eq!(edited["usagePolicy"]["nonCommercial"], true);
    let edited_id = edited["id"].as_str().unwrap();
    let covers = submit_ok(
        &app,
        &project_id,
        json!({"kind": "cover", "style": "bossa nova", "lyrics": "[verse]\nla la",
               "cover": {"mode": "melody", "versionId": edited_id}, "licenseAcknowledged": true}),
    )
    .await;
    let sources = &covers[0]["payload"]["yue2"]["sources"];
    assert_eq!(sources["coverVersion"]["id"], edited_id);
    assert_eq!(sources["transcription"]["sourceAudioAssetId"], asset_id);
    assert_eq!(sources["transcription"]["id"], transcription_id);
    assert_eq!(sources["transcription"]["manifestSha256"], manifest_sha);
    assert_eq!(sources["transcription"]["mode"], "melody");
}

/// A deliberate silent recording: the transcription completes, NOTHING is imported for a refused
/// mode, and the refusal reason is on the job result and the record. Mutation that reds this:
/// importing regardless of the review's readiness.
#[tokio::test]
async fn an_empty_melody_is_a_visible_refusal_not_a_version() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    for row in cover_rows() {
        seed_snapshot(&temp_dir, &row);
    }
    let project_id = project(&app).await;
    register(&app, WORKER).await;
    let recording = wav(12_000);
    let asset_id = upload_recording(&app, &project_id, &recording).await;
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "transcribe", "sourceAudioAssetId": asset_id, "licenseAcknowledged": true}),
    )
    .await;
    let job_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let run_id = jobs[0]["payload"]["yue2"]["runId"]
        .as_str()
        .unwrap()
        .to_owned();
    claim(&app, WORKER).await;
    let dir_rel = format!("yue2/transcriptions/{run_id}");
    let manifest_sha = write_artifact(
        &project_path(&app, &project_id).await.join(&dir_rel),
        &sha(&recording),
        false,
    );
    finish(
        &app,
        &job_id,
        "completed",
        json!({"yue2": {"status": "completed", "kind": "transcribe", "runId": run_id,
                        "transcription": {"dir": dir_rel, "manifestSha256": manifest_sha}}}),
        None,
    )
    .await;
    let result = job(&app, &job_id).await["result"]["yue2"].clone();
    assert_eq!(
        result["scoreVersionIds"],
        json!({"melody": null, "full": null})
    );
    assert_eq!(result["readiness"]["melody"]["ready"], false);
    assert_eq!(
        result["readiness"]["melody"]["reason"],
        "the transcription has no melody notes"
    );
    let (_, versions) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/yue2/score-versions"),
        Value::Null,
    )
    .await;
    assert!(
        versions["items"].as_array().unwrap().is_empty(),
        "{versions}"
    );
}
