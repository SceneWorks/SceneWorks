//! sc-22999: the YuE2 job route, execution-time eligibility, the server-side licence
//! acknowledgment, source resolution, serial batches, the terminal side effects (score versions
//! and render records) and the usage policy on exports — through the real HTTP routes over the
//! LIVE builtin YuE2 entry and the live builtin YuE1 entries that ship beside it.
use super::support::*;
use super::yue2_catalog::{app_with_yue1_and_yue2, app_with_yue1_and_yue2_state, builtin_yue2};
use crate::AppState;
use sceneworks_core::contracts::JobType;
use sceneworks_core::jobs_store::CreateJob;

const SCORE: &str =
    include_str!("../../../../crates/sceneworks-core/src/yue2_score/fixtures/score.abc");
const WORKER: &str = "yue2-test-worker";

async fn project(app: &axum::Router) -> String {
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "YuE2 songs" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    created["id"].as_str().unwrap().to_owned()
}

async fn submit(app: &axum::Router, project_id: &str, body: Value) -> (StatusCode, Value) {
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/yue2/jobs"),
        body,
    )
    .await
}

async fn submit_ok(app: &axum::Router, project_id: &str, body: Value) -> Vec<Value> {
    let (status, response) = submit(app, project_id, body).await;
    assert_eq!(status, StatusCode::CREATED, "{response}");
    response["jobs"].as_array().unwrap().clone()
}

async fn register(app: &axum::Router, worker: &str) {
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": worker, "gpuId": "test-gpu", "gpuName": "Test GPU",
            "capabilities": ["gpu", "audio_generate"], "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

async fn claim(app: &axum::Router, worker: &str) -> Value {
    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": worker }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{claimed}");
    claimed["job"].clone()
}

async fn finish(
    app: &axum::Router,
    job_id: &str,
    status: &str,
    result: Value,
    error: Option<&str>,
) {
    let (code, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": status, "stage": status, "progress": 1, "message": "done",
            "workerId": WORKER, "result": result, "error": error
        }),
    )
    .await;
    assert_eq!(code, StatusCode::OK, "{body}");
}

async fn job(app: &axum::Router, job_id: &str) -> Value {
    let (status, job) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    job
}

fn create_body() -> Value {
    json!({"kind": "create", "style": "dream pop", "lyrics": "[verse]\nla la", "licenseAcknowledged": true})
}

fn audio_fact(asset: &str) -> Value {
    json!({
        "type": "audio", "assetId": asset, "mediaPath": format!("assets/audios/gs/{asset}.wav"),
        "mimeType": "audio/wav", "duration": 1.0, "sampleRate": 48000, "channels": 2,
        "family": "yue2", "displayName": "song", "createdAt": "2026-09-26T00:00:00Z",
        "mode": "yue2_create", "model": "yue2", "adapter": "yue2", "prompt": "dream pop",
        "extra": {"usagePolicy": {"modelId": "yue2", "commercialUse": {"verdict": "refused", "modelId": "yue2", "reason": "CC BY-NC", "alternatives": ["yue_en_cot"]}}}
    })
}

/// A completed song's `result`, in the exact shape the worker's `provenance_block` /
/// `publish_audio` emit (sceneworks-worker `yue2_jobs.rs`): every key, `model` / `decoder` as the
/// score store's `{id, revision}` identities and the tier in `effectiveSettings`.
fn song_result(run_id: &str, asset: &str, extra: Value) -> Value {
    let mut yue2 = json!({
        "status": "completed", "kind": "create", "runId": run_id,
        "run": {"dir": format!("yue2/runs/{run_id}"), "kind": "song", "identity": "cafe01", "planIdentity": "ab12"},
        "truncated": {"abc": false, "semantic": true},
        "warnings": [{"code": "semantic_truncated", "message": "budget"}],
        "effectiveSettings": {"kind": "create", "tier": "bf16", "decoder": "standard", "stagedDecoders": ["standard"]},
        "request": null,
        "model": {"id": "m-a-p/YuE2-3B", "revision": "1a96eca688d6ae5d7f0feb88573fec89920fcd19"},
        "decoder": {"id": "m-a-p/YuE2-Vae", "revision": "95535e72a97bc0f09b8ada125d26b4009428c0e8"},
        "weights": {"mot": "stub"},
        "engineLicense": {"weights": "cc-by-nc-4.0"},
        "sources": null,
        "batch": null,
        "usagePolicy": {"modelId": "yue2"},
    });
    for (key, value) in extra.as_object().unwrap() {
        yue2[key] = value.clone();
    }
    json!({
        "generationSetId": "gs", "expectedCount": 1, "adapter": "yue2", "model": "yue2",
        "generationSet": {"id": "gs", "mode": "yue2_create", "model": "yue2", "prompt": "p", "count": 1, "createdAt": "2026-09-26T00:00:00Z"},
        "assetWrites": [audio_fact(asset)],
        "yue2": yue2,
    })
}

#[tokio::test]
async fn a_create_batch_queues_serial_takes_with_their_own_runs_and_seeds() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    let mut body = create_body();
    body["count"] = json!(3);
    body["seed"] = json!(100);
    body["steps"] = json!(16);
    body["decoder"] = json!("legacy");
    body["memory"] = json!({"tileVaeDecode": true, "decodeTileEdge": 128});
    let jobs = submit_ok(&app, &project_id, body).await;
    assert_eq!(jobs.len(), 3);
    let mut run_ids = std::collections::BTreeSet::new();
    let mut batch_ids = std::collections::BTreeSet::new();
    for (index, job) in jobs.iter().enumerate() {
        assert_eq!(job["type"], "audio_generate");
        let payload = &job["payload"];
        assert_eq!(payload["model"], "yue2");
        assert_eq!(payload["modelManifestEntry"]["id"], "yue2");
        assert_eq!(payload["commercialUse"], false);
        assert_eq!(payload["usagePolicy"]["nonCommercial"], true);
        assert_eq!(
            payload["usagePolicy"]["commercialUse"]["verdict"],
            "refused"
        );
        assert!(payload["usagePolicy"]["licenseAcknowledgment"]["termsSha256"].is_string());
        let block = &payload["yue2"];
        // Every take: its own seed and run, the same settings, the envelope cleared.
        assert_eq!(block["seed"], 100 + index as u64);
        assert_eq!(block["steps"], 16);
        assert_eq!(block["decoder"], "legacy");
        assert_eq!(block["memory"]["decodeTileEdge"], 128);
        assert_eq!(block["batch"]["index"], index);
        assert_eq!(block["batch"]["count"], 3);
        assert!(block.get("count").is_none() && block.get("licenseAcknowledged").is_none());
        run_ids.insert(block["runId"].as_str().unwrap().to_owned());
        batch_ids.insert(block["batch"]["id"].as_str().unwrap().to_owned());
    }
    assert_eq!(run_ids.len(), 3, "each take resumes only its own run");
    assert_eq!(batch_ids.len(), 1);

    // Serial on the admitted GPU: a second worker on the same GPU claims nothing while the first
    // take runs, and the next take once it finishes. Mutation that reds this: a YuE2 job claimed
    // as a non-GPU job type (the second claim succeeds concurrently).
    register(&app, WORKER).await;
    register(&app, "yue2-other-worker").await;
    let first = claim(&app, WORKER).await;
    assert_eq!(first["id"], jobs[0]["id"]);
    assert!(
        claim(&app, "yue2-other-worker").await.is_null(),
        "one active GPU job per GPU"
    );
    finish(
        &app,
        first["id"].as_str().unwrap(),
        "failed",
        json!({}),
        Some("x"),
    )
    .await;
    let second = claim(&app, "yue2-other-worker").await;
    assert_eq!(second["id"], jobs[1]["id"]);
}

#[tokio::test]
async fn invalid_combinations_and_blocked_transcription_are_typed_refusals() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    for (body, status, code) in [
        (
            json!({"kind": "decode", "sourceJobId": "j", "seed": 3}),
            StatusCode::BAD_REQUEST,
            "yue2_invalid_combination",
        ),
        (
            json!({"kind": "create", "lyrics": "l", "cfgScale": 99}),
            StatusCode::BAD_REQUEST,
            "yue2_invalid_value",
        ),
        (
            json!({"kind": "fromPlan"}),
            StatusCode::BAD_REQUEST,
            "yue2_missing_field",
        ),
        (
            json!({"kind": "create", "lyrics": "l", "score": "X:1\nK:C\nabc"}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "yue2_unsupported_notation",
        ),
        (
            json!({"kind": "create", "lyrics": "l", "runId": "yue2run_x"}),
            StatusCode::BAD_REQUEST,
            "yue2_invalid_combination",
        ),
        (
            json!({"kind": "transcribe", "sourceAudioAssetId": "asset_1", "licenseAcknowledged": true}),
            StatusCode::FORBIDDEN,
            "component_blocked",
        ),
        (
            json!({"kind": "cover", "lyrics": "l", "cover": {"mode": "melody", "sourceAudioAssetId": "asset_1"}}),
            StatusCode::FORBIDDEN,
            "component_blocked",
        ),
    ] {
        let (got, response) = submit(&app, &project_id, body.clone()).await;
        assert_eq!(got, status, "{body}: {response}");
        assert_eq!(response["code"], code, "{body}: {response}");
    }
    // The transcription refusal carries the catalog's recorded reason AND unblock condition.
    let (_, response) = submit(
        &app,
        &project_id,
        json!({"kind": "transcribe", "sourceAudioAssetId": "asset_1"}),
    )
    .await;
    let blocked = response["context"]["blocked"].as_array().unwrap();
    assert!(!blocked.is_empty());
    for entry in blocked {
        assert!(
            entry["reason"]
                .as_str()
                .unwrap()
                .contains("owner licensing decision"),
            "{entry}"
        );
        assert!(!entry["unblock"].as_str().unwrap().is_empty(), "{entry}");
    }
    // The generic audio route never queues YuE2 past its eligibility.
    let (status, response) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({"projectId": project_id, "prompt": "x", "model": "yue2", "lyrics": "l"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["code"], "yue2_song_route_required");
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(
        jobs.as_array().unwrap().is_empty(),
        "nothing was queued: {jobs}"
    );
}

/// The licence must be acknowledged to queue; the acknowledgment is recorded server-side; and the
/// execution-time check the worker calls refuses a queued job once it is withdrawn — the state
/// change happens BETWEEN submission and execution. A declared commercial use is refused with the
/// pointer to YuE1.
#[tokio::test]
async fn eligibility_is_rechecked_at_execution_and_commercial_use_points_to_yue1() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    let mut body = create_body();
    body.as_object_mut().unwrap().remove("licenseAcknowledged");
    let (status, response) = submit(&app, &project_id, body.clone()).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{response}");
    assert_eq!(response["code"], "license_acknowledgment_required");

    let jobs = submit_ok(&app, &project_id, create_body()).await;
    let job_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let (_, ack) = request(
        app.clone(),
        "GET",
        "/api/v1/models/yue2/license-acknowledgment",
        Value::Null,
    )
    .await;
    assert_eq!(ack["acknowledged"], true, "{ack}");
    assert_eq!(ack["acknowledgment"]["channel"], "yue2_job");
    // Recorded, so a later submission needs no assertion.
    submit_ok(&app, &project_id, body).await;

    let eligibility = format!("/api/v1/jobs/{job_id}/yue2-eligibility");
    let (status, answer) = request(app.clone(), "GET", &eligibility, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["eligible"], true);

    // Withdraw it after the job was queued: execution now refuses.
    // Mutation that reds this: the eligibility route reading the payload's submission-time
    // policy instead of the acknowledgment store.
    let (status, _) = request(
        app.clone(),
        "DELETE",
        "/api/v1/models/yue2/license-acknowledgment",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, answer) = request(app.clone(), "GET", &eligibility, Value::Null).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{answer}");
    assert_eq!(answer["code"], "license_acknowledgment_required");
    let (status, _) = request(
        app.clone(),
        "PUT",
        "/api/v1/models/yue2/license-acknowledgment",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = request(app.clone(), "GET", &eligibility, Value::Null).await;
    assert_eq!(status, StatusCode::OK);

    // Commercial use: refused on submission with the verdict's pointer to the YuE1 entries.
    let mut commercial = create_body();
    commercial["commercialUse"] = json!(true);
    let (status, response) = submit(&app, &project_id, commercial).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{response}");
    assert_eq!(response["code"], "commercial_use_refused");
    assert_eq!(response["context"]["verdict"], "refused");
    let alternatives = response["context"]["alternatives"].as_array().unwrap();
    assert!(
        !alternatives.is_empty()
            && alternatives
                .iter()
                .all(|id| id.as_str().unwrap().starts_with("yue_")),
        "{response}"
    );
    assert!(
        response["detail"].as_str().unwrap().contains("yue_"),
        "{response}"
    );
}

/// Restored-plan and cached-decode jobs resolve a COMPLETED run of THIS project, with the identity
/// its job recorded; anything else is refused before it queues.
#[tokio::test]
async fn plan_and_decode_sources_resolve_completed_runs_of_the_project_only() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    let jobs = submit_ok(&app, &project_id, create_body()).await;
    let song_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let run_id = jobs[0]["payload"]["yue2"]["runId"]
        .as_str()
        .unwrap()
        .to_owned();

    // Not completed yet: refused.
    let (status, response) = submit(
        &app,
        &project_id,
        json!({"kind": "decode", "sourceJobId": song_id}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{response}");
    assert_eq!(response["code"], "yue2_source_unavailable");

    register(&app, WORKER).await;
    claim(&app, WORKER).await;
    finish(
        &app,
        &song_id,
        "completed",
        song_result(&run_id, "asset_song1", json!({})),
        None,
    )
    .await;

    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "decode", "sourceJobId": song_id, "decoder": "legacy"}),
    )
    .await;
    let source = &jobs[0]["payload"]["yue2"]["sources"]["sourceRun"];
    assert_eq!(source["jobId"], song_id);
    assert_eq!(source["runDir"], format!("yue2/runs/{run_id}"));
    assert_eq!(source["identity"], "cafe01");
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "fromPlan", "planJobId": song_id, "steps": 8}),
    )
    .await;
    assert_eq!(
        jobs[0]["payload"]["yue2"]["sources"]["plan"]["planIdentity"],
        "ab12"
    );

    // Another project's run is never a source.
    let other = project(&app).await;
    let (status, response) = submit(
        &app,
        &other,
        json!({"kind": "decode", "sourceJobId": song_id}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{response}");
}

/// A completed plan becomes a score version linked to its job, once; a score-version render —
/// completed or failed — is recorded with `record_yue2_render`, linking version, job and asset; a
/// render whose score digest is not its version's is NOT recorded (409 surfaced on the result).
#[tokio::test]
async fn finished_runs_become_score_versions_and_render_records() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    register(&app, WORKER).await;

    // Plan → score version.
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "plan", "style": "folk", "lyrics": "[verse]\nhey", "seed": 7, "licenseAcknowledged": true}),
    )
    .await;
    let plan_id = jobs[0]["id"].as_str().unwrap().to_owned();
    claim(&app, WORKER).await;
    let request_json = json!({"style": "folk", "lyrics": "[verse]\nhey", "cot": "full", "seed": 7});
    finish(
        &app,
        &plan_id,
        "completed",
        json!({"yue2": {
            "status": "completed", "kind": "plan",
            "run": {"dir": "yue2/runs/yue2run_p", "kind": "plan", "identity": "aa", "planIdentity": "bb"},
            "score": {"abc": SCORE, "sha256": "x"}, "request": request_json,
        }}),
        None,
    )
    .await;
    let version_id = job(&app, &plan_id).await["result"]["yue2"]["scoreVersionId"]
        .as_str()
        .expect("the plan was versioned")
        .to_owned();
    let (_, versions) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/yue2/score-versions"),
        Value::Null,
    )
    .await;
    let items = versions["items"].as_array().unwrap();
    assert_eq!(items.len(), 1, "{versions}");
    assert_eq!(items[0]["origin"], "plan");
    assert_eq!(items[0]["provenance"]["source"]["id"], plan_id);

    // Render that version: completed → a render record linked to job and asset.
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "renderVersion", "versionId": version_id, "steps": 4}),
    )
    .await;
    let render_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let queued = jobs[0]["payload"]["yue2"]["sources"]["version"].clone();
    assert_eq!(queued["id"], version_id);
    claim(&app, WORKER).await;
    let rendered = json!({
        "kind": "renderVersion", "versionId": version_id,
        "renderedScoreSha256": queued["scoreSha256"], "renderedRequestSha256": queued["requestSha256"],
    });
    finish(
        &app,
        &render_id,
        "completed",
        song_result("yue2run_r", "asset_render1", rendered),
        None,
    )
    .await;
    let finished = job(&app, &render_id).await;
    assert!(
        finished["result"]["yue2"]["renderRecordId"].is_string(),
        "{finished}"
    );
    let (_, detail) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/yue2/score-versions/{version_id}"),
        Value::Null,
    )
    .await;
    let render = &detail["renders"][0];
    assert_eq!(render["status"], "completed", "{detail}");
    assert_eq!(render["jobId"], render_id);
    assert_eq!(render["audioAssetId"], "asset_render1");
    assert_eq!(render["truncated"], json!({"abc": false, "semantic": true}));
    assert_eq!(render["decoder"]["id"], "m-a-p/YuE2-Vae");
    assert_eq!(render["effectiveSettings"]["tier"], "bf16");
    assert_eq!(render["provenance"]["source"]["id"], render_id);

    // A failed render is recorded as failed, with its error.
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "renderVersion", "versionId": version_id}),
    )
    .await;
    let failed_id = jobs[0]["id"].as_str().unwrap().to_owned();
    claim(&app, WORKER).await;
    finish(
        &app,
        &failed_id,
        "failed",
        json!({"yue2": {"status": "failed"}}),
        Some("engine exploded"),
    )
    .await;
    let (_, detail) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/yue2/score-versions/{version_id}"),
        Value::Null,
    )
    .await;
    let failed = detail["renders"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["jobId"] == failed_id)
        .expect("the failed render is recorded")
        .clone();
    assert_eq!(failed["status"], "failed");
    // Never observed, never asserted. Mutation that reds this: defaulting the failed render's
    // truncation to {abc: false, semantic: false}.
    assert!(failed["truncated"].is_null(), "{failed}");
    assert!(failed["error"]
        .as_str()
        .unwrap()
        .contains("engine exploded"));

    // A render that reports another score's digest is refused by the contract (409), surfaced on
    // the result and never recorded. Mutation that reds this: recording with the queued digests.
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "renderVersion", "versionId": version_id}),
    )
    .await;
    let wrong_id = jobs[0]["id"].as_str().unwrap().to_owned();
    claim(&app, WORKER).await;
    let wrong = json!({"renderedScoreSha256": "0".repeat(64), "renderedRequestSha256": queued["requestSha256"]});
    finish(
        &app,
        &wrong_id,
        "completed",
        song_result("yue2run_w", "asset_render2", wrong),
        None,
    )
    .await;
    let finished = job(&app, &wrong_id).await;
    let errors = finished["result"]["yue2"]["sideEffectErrors"]
        .as_array()
        .expect("surfaced");
    assert!(
        errors[0].as_str().unwrap().contains("render record"),
        "{finished}"
    );
    let (_, detail) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/yue2/score-versions/{version_id}"),
        Value::Null,
    )
    .await;
    assert!(detail["renders"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["jobId"] != wrong_id));
}

/// The usage policy a YuE2 asset carries travels with a timeline export: a commercial export that
/// places it is refused, and an ordinary export stamps the policy onto the exported file.
#[tokio::test]
async fn timeline_exports_carry_and_enforce_the_usage_policy() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    register(&app, WORKER).await;
    let jobs = submit_ok(&app, &project_id, create_body()).await;
    let song_id = jobs[0]["id"].as_str().unwrap().to_owned();
    claim(&app, WORKER).await;
    finish(
        &app,
        &song_id,
        "completed",
        song_result("yue2run_s", "asset_song9", json!({})),
        None,
    )
    .await;

    let (status, timeline) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines"),
        json!({"name": "Cut", "aspectRatio": "16:9", "fps": 30}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{timeline}");
    let timeline_id = timeline["id"].as_str().unwrap().to_owned();
    let mut document = timeline.clone();
    let track_id = document["tracks"][2]["id"].clone();
    document["tracks"][2]["items"] = json!([{
        "id": "item-1", "trackId": track_id, "assetId": "asset_song9", "type": "audio",
        "displayName": "Song", "sourceIn": 0, "sourceOut": 1, "timelineStart": 0,
        "timelineEnd": 1, "speed": 1, "fit": "fit", "volume": 1
    }]);
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({"timeline": document}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");

    let exports = format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports");
    let (status, response) = request(
        app.clone(),
        "POST",
        &exports,
        json!({"commercialUse": true}),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{response}");
    assert_eq!(response["code"], "commercial_use_refused");
    assert!(response["detail"].as_str().unwrap().contains("asset_song9"));

    let (status, export) = request(app.clone(), "POST", &exports, json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{export}");
    let policies = export["payload"]["usagePolicies"]
        .as_array()
        .expect("policies travel");
    assert_eq!(policies[0]["assetId"], "asset_song9");
    // The exported file's sidecar inherits them.
    let export_id = export["id"].as_str().unwrap().to_owned();
    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({"workerId": "export-worker", "gpuId": "cpu", "gpuName": "cpu",
               "capabilities": ["timeline_export"], "loadedModels": []}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{claimed}");
    let claimed = claim(&app, "export-worker").await;
    assert_eq!(claimed["id"], export_id);
    let (status, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{export_id}/progress"),
        json!({"status": "completed", "stage": "completed", "progress": 1, "message": "done",
        "workerId": "export-worker",
        "result": {"generationSetId": "gsx", "assetWrites": [{
            "type": "video", "assetId": "asset_export1", "mediaPath": "assets/videos/x.mp4",
            "mimeType": "video/mp4", "createdAt": "2026-09-26T00:00:00Z", "displayName": "Cut"
        }]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, asset) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/asset_export1"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{asset}");
    assert_eq!(
        asset["extra"]["usagePolicies"][0]["assetId"], "asset_song9",
        "{asset}"
    );
}

#[test]
fn the_builtin_entry_is_the_one_under_test() {
    // The live entry, not a copy: it declares the licence gate and the commercial refusal.
    let entry = builtin_yue2();
    assert_eq!(entry["requiresLicenseAcknowledgment"], true);
    assert_eq!(entry["commercialUse"]["eligible"], false);
}

/// A retried YuE2 job keeps its run id — so the worker resumes that run's verified checkpoints —
/// and a retry that rewrites the `yue2` block into one the worker would refuse is refused here.
/// Mutation that reds this: dropping `validate_replayed_yue2_block` from the replay path.
#[tokio::test]
async fn a_retry_resumes_the_same_run_and_a_tampered_block_is_refused() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    register(&app, WORKER).await;
    let jobs = submit_ok(&app, &project_id, create_body()).await;
    let job_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let run_id = jobs[0]["payload"]["yue2"]["runId"].clone();
    claim(&app, WORKER).await;
    finish(
        &app,
        &job_id,
        "failed",
        json!({"yue2": {"status": "failed"}}),
        Some("interrupted"),
    )
    .await;

    let (status, retried) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/retry"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{retried}");
    assert_eq!(
        retried["payload"]["yue2"]["runId"], run_id,
        "the retry resumes its run"
    );

    let mut tampered = jobs[0]["payload"]["yue2"].clone();
    tampered["seed"] = json!(5);
    tampered["kind"] = json!("decode");
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/retry"),
        json!({"payloadChanges": {"yue2": tampered}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["code"], "yue2_invalid_combination");
}

/// Item 1 (sc-22999 review): a replay cannot point a YuE2 job at another model, inject a `yue2`
/// block into a generic audio job, or drop one; a duplicate is a new take with its own run.
#[tokio::test]
async fn replays_cannot_swap_the_model_inject_a_block_or_share_a_run() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let (app, state) = app_with_yue1_and_yue2_state(&temp_dir);
    let project_id = project(&app).await;
    let jobs = submit_ok(&app, &project_id, create_body()).await;
    let job_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let run_id = jobs[0]["payload"]["yue2"]["runId"].clone();

    // Mutation that reds this: dropping the model check in `canonicalize_replayed_audio_payload`.
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/duplicate"),
        json!({"payloadChanges": {"model": "yue_en_cot", "commercialUse": true}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["code"], "yue2_invalid_combination");
    assert_eq!(response["context"]["field"], "model");

    // A plain duplicate: a fresh run (never the original's working directory), no batch.
    // Mutation that reds this: dropping `refresh_block_for_duplicate` from `duplicate_job`.
    let (status, duplicate) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/duplicate"),
        json!({"payloadChanges": {}}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{duplicate}");
    assert_ne!(duplicate["payload"]["yue2"]["runId"], run_id);
    assert!(duplicate["payload"]["yue2"]["runId"]
        .as_str()
        .unwrap()
        .starts_with("yue2run_"));

    // Removing the block is refused too.
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/duplicate"),
        json!({"payloadChanges": {"yue2": null}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");

    // A generic audio job cannot gain a block through its replay.
    let generic = state
        .jobs_store
        .create_job(CreateJob {
            job_type: JobType::AudioGenerate,
            project_id: Some(project_id.clone()),
            project_name: None,
            payload: json!({"projectId": project_id, "model": "kokoro_82m", "prompt": "hi"})
                .as_object()
                .unwrap()
                .clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap();
    // Mutation that reds this: accepting a block the persisted payload did not carry.
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/duplicate", generic.id),
        json!({"payloadChanges": {"model": "yue2", "yue2": jobs[0]["payload"]["yue2"]}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["context"]["field"], "yue2");
    // Nor become a block-less symbolic-song job.
    // Mutation that reds this: dropping the block-less `refuse_symbolic_song_on_audio_route`.
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/duplicate", generic.id),
        json!({"payloadChanges": {"model": "yue2"}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["code"], "yue2_song_route_required");
}

fn stored_yue2_job(state: &AppState, project_id: &str, payload_extra: Value) -> String {
    let mut payload = json!({
        "projectId": project_id, "model": "yue2", "prompt": "",
        "yue2": {"kind": "create", "lyrics": "[verse]\nla", "runId": "yue2run_stored"},
    });
    for (key, value) in payload_extra.as_object().unwrap() {
        payload[key] = value.clone();
    }
    state
        .jobs_store
        .create_job(CreateJob {
            job_type: JobType::AudioGenerate,
            project_id: Some(project_id.to_owned()),
            project_name: None,
            payload: payload.as_object().unwrap().clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap()
        .id
}

/// Items 1a and 9: the execution-time check always evaluates YuE2 — whatever `model` a stored
/// payload names — and refuses a job that declares a commercial use.
#[tokio::test]
async fn execution_eligibility_is_always_yue2s_and_refuses_a_declared_commercial_use() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let (app, state) = app_with_yue1_and_yue2_state(&temp_dir);
    let project_id = project(&app).await;
    // No acknowledgment recorded: YuE2 is not eligible. A payload claiming a YuE1 model must not
    // make it so. Mutation that reds this: evaluating `payload.model` again.
    let job_id = stored_yue2_job(&state, &project_id, json!({"model": "yue_en_cot"}));
    let (status, answer) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}/yue2-eligibility"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{answer}");
    assert_eq!(answer["code"], "license_acknowledgment_required");
    assert_eq!(answer["context"]["modelId"], "yue2");

    let (status, _) = request(
        app.clone(),
        "PUT",
        "/api/v1/models/yue2/license-acknowledgment",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // Mutation that reds this: reading `commercialUse` as false at execution.
    let job_id = stored_yue2_job(&state, &project_id, json!({"commercialUse": true}));
    let (status, answer) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}/yue2-eligibility"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{answer}");
    assert_eq!(answer["code"], "commercial_use_refused");
    let job_id = stored_yue2_job(&state, &project_id, json!({}));
    let (status, answer) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}/yue2-eligibility"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{answer}");
}

/// Item 5: a YuE2 output's policy survives derivation — an asset made from it inherits it, an
/// export of a timeline placing it carries it, and a timeline that places THAT export (or the
/// derived asset) is still refused a commercial export.
#[tokio::test]
async fn usage_policies_survive_derivation_and_re_export() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let (app, state) = app_with_yue1_and_yue2_state(&temp_dir);
    let project_id = project(&app).await;
    register(&app, WORKER).await;
    let jobs = submit_ok(&app, &project_id, create_body()).await;
    let song_id = jobs[0]["id"].as_str().unwrap().to_owned();
    claim(&app, WORKER).await;
    finish(
        &app,
        &song_id,
        "completed",
        song_result("yue2run_s", "asset_song9", json!({})),
        None,
    )
    .await;

    // A generic audio edit whose source is the YuE2 song.
    let edit = state
        .jobs_store
        .create_job(CreateJob {
            job_type: JobType::AudioGenerate,
            project_id: Some(project_id.clone()),
            project_name: None,
            payload: json!({"projectId": project_id, "model": "ace_step", "prompt": "extend"})
                .as_object()
                .unwrap()
                .clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap();
    let claimed = claim(&app, WORKER).await;
    assert_eq!(claimed["id"], edit.id);
    let mut derived = audio_fact("asset_edit1");
    derived.as_object_mut().unwrap().remove("extra");
    derived["sourceAssetId"] = json!("asset_song9");
    finish(
        &app,
        &edit.id,
        "completed",
        json!({"generationSetId": "gs2", "assetWrites": [derived]}),
        None,
    )
    .await;
    let (_, asset) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/asset_edit1"),
        Value::Null,
    )
    .await;
    // Mutation that reds this: dropping `inherit_usage_policies` from `persist_reported_assets`.
    assert_eq!(
        asset["extra"]["usagePolicies"][0]["assetId"], "asset_song9",
        "{asset}"
    );

    // Export a timeline placing the derived asset; then place THAT export in a new timeline.
    let export_asset = export_timeline(&app, &project_id, "asset_edit1", "asset_export1").await;
    assert_eq!(
        export_asset["extra"]["usagePolicies"][0]["assetId"],
        "asset_song9"
    );
    let timeline_id = timeline_placing(&app, &project_id, "asset_export1").await;
    // Mutation that reds this: reading only `extra.usagePolicy` in `timeline_usage_policies`.
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
        json!({"commercialUse": true}),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{response}");
    assert_eq!(response["code"], "commercial_use_refused");
}

async fn timeline_placing(app: &axum::Router, project_id: &str, asset_id: &str) -> String {
    let (status, timeline) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines"),
        json!({"name": "Cut", "aspectRatio": "16:9", "fps": 30}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{timeline}");
    let timeline_id = timeline["id"].as_str().unwrap().to_owned();
    let mut document = timeline.clone();
    let track_id = document["tracks"][2]["id"].clone();
    document["tracks"][2]["items"] = json!([{
        "id": "item-1", "trackId": track_id, "assetId": asset_id, "type": "audio",
        "displayName": "Song", "sourceIn": 0, "sourceOut": 1, "timelineStart": 0,
        "timelineEnd": 1, "speed": 1, "fit": "fit", "volume": 1
    }]);
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({"timeline": document}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");
    timeline_id
}

async fn export_timeline(
    app: &axum::Router,
    project_id: &str,
    placed: &str,
    export_asset: &str,
) -> Value {
    let timeline_id = timeline_placing(app, project_id, placed).await;
    let (status, export) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{export}");
    let export_id = export["id"].as_str().unwrap().to_owned();
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({"workerId": "export-worker", "gpuId": "cpu", "gpuName": "cpu",
               "capabilities": ["timeline_export"], "loadedModels": []}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let claimed = claim(app, "export-worker").await;
    assert_eq!(claimed["id"], export_id);
    let (status, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{export_id}/progress"),
        json!({"status": "completed", "stage": "completed", "progress": 1, "message": "done",
               "workerId": "export-worker",
               "result": {"generationSetId": "gsx", "assetWrites": [{
                   "type": "video", "assetId": export_asset, "mediaPath": format!("assets/videos/{export_asset}.mp4"),
                   "mimeType": "video/mp4", "createdAt": "2026-09-26T00:00:00Z", "displayName": "Cut"
               }]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, asset) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{export_asset}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{asset}");
    asset
}

/// Items 10 and 12: a malformed worker identity is a loud side-effect error (never replaced by the
/// catalog's), and a plan whose score was truncated is never versioned.
#[tokio::test]
async fn malformed_identities_are_loud_and_truncated_plans_are_not_versioned() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    register(&app, WORKER).await;

    // Mutation that reds this: skipping versioning only on a non-truncated plan check removed.
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "plan", "style": "folk", "lyrics": "[verse]\nhey", "seed": 7, "licenseAcknowledged": true}),
    )
    .await;
    let plan_id = jobs[0]["id"].as_str().unwrap().to_owned();
    claim(&app, WORKER).await;
    finish(
        &app,
        &plan_id,
        "completed",
        json!({"yue2": {
            "status": "completed", "kind": "plan", "truncated": {"abc": true},
            "run": {"dir": "yue2/runs/yue2run_p", "kind": "plan", "identity": "aa", "planIdentity": "bb"},
            "score": {"abc": SCORE, "sha256": "x"},
            "request": {"style": "folk", "lyrics": "[verse]\nhey", "cot": "full", "seed": 7},
        }}),
        None,
    )
    .await;
    let block = job(&app, &plan_id).await["result"]["yue2"].clone();
    assert!(block.get("scoreVersionId").is_none(), "{block}");
    assert_eq!(block["scoreVersionSkipped"], "abc_truncated");

    // A complete plan to render.
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "plan", "style": "folk", "lyrics": "[verse]\nhey", "seed": 8}),
    )
    .await;
    let plan_id = jobs[0]["id"].as_str().unwrap().to_owned();
    claim(&app, WORKER).await;
    finish(
        &app,
        &plan_id,
        "completed",
        json!({"yue2": {
            "status": "completed", "kind": "plan", "truncated": {"abc": false},
            "run": {"dir": "yue2/runs/yue2run_p2", "kind": "plan", "identity": "aa", "planIdentity": "bb"},
            "score": {"abc": SCORE, "sha256": "x"},
            "request": {"style": "folk", "lyrics": "[verse]\nhey", "cot": "full", "seed": 8},
        }}),
        None,
    )
    .await;
    let version_id = job(&app, &plan_id).await["result"]["yue2"]["scoreVersionId"]
        .as_str()
        .expect("versioned")
        .to_owned();
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "renderVersion", "versionId": version_id}),
    )
    .await;
    let render_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let queued = jobs[0]["payload"]["yue2"]["sources"]["version"].clone();
    claim(&app, WORKER).await;
    // The pre-fix worker shape: `tier` inside the model identity. Mutation that reds this:
    // `identity_from` falling back to the catalog identity on a parse failure.
    let malformed = json!({
        "renderedScoreSha256": queued["scoreSha256"], "renderedRequestSha256": queued["requestSha256"],
        "model": {"id": "m-a-p/YuE2-3B", "revision": "1a96eca688d6ae5d7f0feb88573fec89920fcd19", "tier": "bf16"},
    });
    finish(
        &app,
        &render_id,
        "completed",
        song_result("yue2run_r", "asset_r1", malformed),
        None,
    )
    .await;
    let block = job(&app, &render_id).await["result"]["yue2"].clone();
    let errors = block["sideEffectErrors"].as_array().expect("loud");
    assert!(
        errors[0].as_str().unwrap().contains("`model` identity"),
        "{block}"
    );
    assert!(block.get("renderRecordId").is_none());
}
