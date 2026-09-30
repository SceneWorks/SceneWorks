//! sc-22999: the YuE2 job route, execution-time eligibility, the server-side licence
//! acknowledgment, source resolution, serial batches, the terminal side effects (score versions
//! and render records) and the usage policy on exports — through the real HTTP routes over the
//! LIVE builtin YuE2 entry and the live builtin YuE1 entries that ship beside it.
use super::support::*;
use super::yue2_catalog::{
    app_with_yue1_and_yue2, app_with_yue1_and_yue2_state, builtin_yue2, write_yue1_and,
};
use crate::AppState;
use sceneworks_core::contracts::JobType;
use sceneworks_core::jobs_store::CreateJob;

const SCORE: &str =
    include_str!("../../../../crates/sceneworks-core/src/yue2_score/fixtures/score.abc");
pub(super) const WORKER: &str = "yue2-test-worker";
/// The Song Lab's field table and the bodies its request builder produces
/// (`apps/web/src/yue2Lab.test.js` writes and checks this file).
const WEB_JOB_REQUESTS: &str = include_str!(
    "../../../../crates/sceneworks-core/src/yue2_score/fixtures/web-job-requests.json"
);

pub(super) async fn project(app: &axum::Router) -> String {
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

pub(super) async fn submit(
    app: &axum::Router,
    project_id: &str,
    mut body: Value,
) -> (StatusCode, Value) {
    // These route fixtures predate the explicit compute control. Keep each fresh request honest;
    // dedicated migration tests below exercise missing and conflicting policies without this seam.
    if body["kind"] != "transcribe" && body.get("computePolicy").is_none() {
        let policy = if body["precision"] == "fp32" {
            "fp32"
        } else {
            "auto"
        };
        body.as_object_mut().unwrap().remove("precision");
        body["computePolicy"] = json!(policy);
    }
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/yue2/jobs"),
        body,
    )
    .await
}

pub(super) async fn submit_ok(app: &axum::Router, project_id: &str, body: Value) -> Vec<Value> {
    let (status, response) = submit(app, project_id, body).await;
    assert_eq!(status, StatusCode::CREATED, "{response}");
    response["jobs"].as_array().unwrap().clone()
}

#[tokio::test]
async fn fresh_submission_requires_explicit_compute_policy() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    let route = format!("/api/v1/projects/{project_id}/yue2/jobs");
    for body in [
        json!({"kind":"create","lyrics":"la la"}),
        json!({"kind":"create","lyrics":"la la","precision":"default"}),
    ] {
        let (status, response) = request(app.clone(), "POST", &route, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert_eq!(response["code"], "yue2_missing_field", "{response}");
    }
    let (status, response) = request(
        app,
        "POST",
        &route,
        json!({"kind":"create","lyrics":"la la","computePolicy":"bf16","precision":"fp32"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["code"], "yue2_invalid_combination", "{response}");
}

pub(super) async fn register(app: &axum::Router, worker: &str) {
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

pub(super) async fn claim(app: &axum::Router, worker: &str) -> Value {
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

pub(super) async fn finish(
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

pub(super) async fn job(app: &axum::Router, job_id: &str) -> Value {
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
        // The catalog's declared licence identifier travels in the policy (sc-22988 review item
        // 15). Mutation that reds this: removing `license` from the builtin yue2 entry.
        assert_eq!(payload["usagePolicy"]["license"]["license"], "CC-BY-NC-4.0");
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
async fn invalid_combinations_and_unready_transcription_are_typed_refusals() {
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
        // sc-23002: the cover closure is not installed in this fresh data dir.
        (
            json!({"kind": "transcribe", "sourceAudioAssetId": "asset_1", "licenseAcknowledged": true}),
            StatusCode::CONFLICT,
            "yue2_cover_components_missing",
        ),
        // A cover from a recording goes through a reviewed transcription (AT2).
        (
            json!({"kind": "cover", "lyrics": "l", "cover": {"mode": "melody", "sourceAudioAssetId": "asset_1"}}),
            StatusCode::BAD_REQUEST,
            "yue2_transcription_review_required",
        ),
    ] {
        let (got, response) = submit(&app, &project_id, body.clone()).await;
        assert_eq!(got, status, "{body}: {response}");
        assert_eq!(response["code"], code, "{body}: {response}");
    }
    // The missing-closure refusal names every missing component and the one route that installs
    // them.
    let (_, response) = submit(
        &app,
        &project_id,
        json!({"kind": "transcribe", "sourceAudioAssetId": "asset_1"}),
    )
    .await;
    let missing = response["context"]["missing"].as_array().unwrap();
    let repos: Vec<&str> = missing
        .iter()
        .map(|m| m["repo"].as_str().unwrap())
        .collect();
    assert_eq!(
        repos,
        ["m-a-p/SheetSage2", "m-a-p/MERT-v2-FullSong"],
        "{response}"
    );
    assert_eq!(
        response["context"]["install"],
        "/api/v1/models/yue2/conditional-components/cover/download"
    );
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

/// A restored plan renders the plan's own style and lyrics: a `fromPlan` job whose style or lyrics
/// differ from the run's recorded `request.json` is refused at submission (409
/// `yue2_plan_request_mismatch`) and queues nothing — the engine would refuse it only after a
/// multi-GB load. The plan's exact text, or none, is accepted.
#[tokio::test]
async fn a_restored_plan_with_other_words_is_refused_before_it_queues() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "plan", "style": "folk", "lyrics": "[verse]\nhey", "licenseAcknowledged": true}),
    )
    .await;
    let plan_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let run_id = jobs[0]["payload"]["yue2"]["runId"]
        .as_str()
        .unwrap()
        .to_owned();
    register(&app, WORKER).await;
    claim(&app, WORKER).await;
    let (_, created) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}"),
        Value::Null,
    )
    .await;
    let run_dir = std::path::PathBuf::from(created["path"].as_str().unwrap())
        .join(format!("yue2/runs/{run_id}"));
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(
        run_dir.join("request.json"),
        json!({"style": "folk", "lyrics": "[verse]\nhey", "cot": "full", "seed": 7}).to_string(),
    )
    .unwrap();
    finish(
        &app,
        &plan_id,
        "completed",
        json!({"yue2": {
            "status": "completed", "kind": "plan",
            "run": {"dir": format!("yue2/runs/{run_id}"), "kind": "plan", "identity": "aa", "planIdentity": "bb"},
        }}),
        None,
    )
    .await;
    let job_count = |jobs: Value| jobs.as_array().expect("jobs is an array").len();
    let queued_before = job_count(
        request(app.clone(), "GET", "/api/v1/jobs", Value::Null)
            .await
            .1,
    );

    for (mut body, field) in [
        (json!({"lyrics": "[verse]\nnew words"}), "lyrics"),
        (json!({"style": "metal", "lyrics": "[verse]\nhey"}), "style"),
    ] {
        body["kind"] = json!("fromPlan");
        body["planJobId"] = json!(plan_id);
        let (status, response) = submit(&app, &project_id, body).await;
        assert_eq!(status, StatusCode::CONFLICT, "{response}");
        assert_eq!(response["code"], "yue2_plan_request_mismatch", "{response}");
        assert_eq!(response["context"]["field"], field, "{response}");
    }
    let (status, response) = submit(
        &app,
        &project_id,
        json!({"kind": "fromPlan", "planJobId": plan_id, "tier": "bf16", "arMode": "experimentalFp8"}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{response}");
    assert_eq!(response["code"], "yue2_plan_request_mismatch", "{response}");
    assert_eq!(response["context"]["field"], "arMode", "{response}");
    assert_eq!(
        job_count(
            request(app.clone(), "GET", "/api/v1/jobs", Value::Null)
                .await
                .1
        ),
        queued_before,
        "a refused restore queues nothing"
    );

    let exact = submit_ok(
        &app,
        &project_id,
        json!({"kind": "fromPlan", "planJobId": plan_id, "style": "folk", "lyrics": "[verse]\nhey"}),
    )
    .await;
    assert_eq!(exact[0]["payload"]["yue2"]["lyrics"], "[verse]\nhey");
    submit_ok(
        &app,
        &project_id,
        json!({"kind": "fromPlan", "planJobId": plan_id}),
    )
    .await;
}

#[tokio::test]
async fn a_restored_plan_cannot_switch_ar_mode_before_it_queues() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().unwrap();
    let app = app_with_yue1_and_yue2(&temp_dir);
    let project_id = project(&app).await;
    let jobs = submit_ok(
        &app,
        &project_id,
        json!({"kind": "plan", "lyrics": "[verse]\nhey", "tier": "bf16", "arMode": "experimentalFp8", "licenseAcknowledged": true}),
    )
    .await;
    let plan_id = jobs[0]["id"].as_str().unwrap().to_owned();
    let run_id = jobs[0]["payload"]["yue2"]["runId"].as_str().unwrap();
    register(&app, WORKER).await;
    claim(&app, WORKER).await;
    let (_, created) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}"),
        Value::Null,
    )
    .await;
    let run_dir = std::path::PathBuf::from(created["path"].as_str().unwrap())
        .join(format!("yue2/runs/{run_id}"));
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(
        run_dir.join("request.json"),
        json!({"style": "", "lyrics": "[verse]\nhey", "cot": "full", "seed": 7}).to_string(),
    )
    .unwrap();
    finish(
        &app,
        &plan_id,
        "completed",
        json!({"yue2": {
            "status": "completed", "kind": "plan",
            "run": {"dir": format!("yue2/runs/{run_id}"), "kind": "plan", "identity": "aa", "planIdentity": "bb"},
        }}),
        None,
    )
    .await;
    let count = |jobs: Value| jobs.as_array().expect("jobs is an array").len();
    let queued_before = count(
        request(app.clone(), "GET", "/api/v1/jobs", Value::Null)
            .await
            .1,
    );
    for ar_mode in [None, Some("native")] {
        let mut body = json!({"kind": "fromPlan", "planJobId": plan_id, "tier": "bf16"});
        if let Some(ar_mode) = ar_mode {
            body["arMode"] = json!(ar_mode);
        }
        let (status, response) = submit(&app, &project_id, body).await;
        assert_eq!(status, StatusCode::CONFLICT, "{response}");
        assert_eq!(response["code"], "yue2_plan_request_mismatch", "{response}");
        assert_eq!(response["context"]["field"], "arMode", "{response}");
    }
    assert_eq!(
        count(
            request(app.clone(), "GET", "/api/v1/jobs", Value::Null)
                .await
                .1
        ),
        queued_before
    );
    let restored = submit_ok(
        &app,
        &project_id,
        json!({"kind": "fromPlan", "planJobId": plan_id, "tier": "bf16", "arMode": "experimentalFp8"}),
    )
    .await;
    assert_eq!(restored[0]["payload"]["yue2"]["arMode"], "experimentalFp8");
    let selected = submit_ok(
        &app,
        &project_id,
        json!({"kind": "create", "lyrics": "[verse]\nla", "tier": "bf16", "arMode": "experimentalFp8", "requestedGpu": "0", "licenseAcknowledged": true}),
    )
    .await;
    assert_eq!(
        selected[0]["requestedGpu"], "0",
        "numeric CUDA worker id persists in the queue envelope"
    );
    assert_eq!(selected[0]["payload"]["yue2"]["arMode"], "experimentalFp8");
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

    // A generic (V1) retry cannot store `"yue2": null` — a key the worker once read as a block.
    // Mutation that reds this: dropping the null-block refusal in
    // `canonicalize_replayed_audio_payload` (the block-less path accepts it).
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/duplicate", generic.id),
        json!({"payloadChanges": {"yue2": null}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["context"]["field"], "yue2");
    let stored = job(&app, &generic.id).await;
    assert!(
        stored["payload"].get("yue2").is_none(),
        "the V1 job stays V1: {stored}"
    );
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

/// Queue a generic (non-YuE2) audio job, claim it and complete it with `asset_writes`.
async fn finish_generic_audio(
    app: &axum::Router,
    state: &AppState,
    project_id: &str,
    asset_writes: Value,
) {
    let job = state
        .jobs_store
        .create_job(CreateJob {
            job_type: JobType::AudioGenerate,
            project_id: Some(project_id.to_owned()),
            project_name: None,
            payload: json!({"projectId": project_id, "model": "yue_en_icl", "prompt": "icl"})
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
    let claimed = claim(app, WORKER).await;
    assert_eq!(claimed["id"], job.id);
    finish(
        app,
        &job.id,
        "completed",
        json!({"generationSetId": "gs_icl", "assetWrites": asset_writes}),
        None,
    )
    .await;
}

async fn asset(app: &axum::Router, project_id: &str, asset_id: &str) -> Value {
    let (status, asset) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{asset}");
    asset
}

/// A YuE1 ICL render that uses a YuE2 output as its reference writes its mix AND its stems in one
/// batch; each stem names only the (not yet persisted) mix as its parent. The stems still inherit
/// the YuE2 policy through the mix, so a commercial export of a stem is refused. An input that is
/// in neither the batch nor the library is recorded as unresolved lineage — never as "no policy" —
/// and a commercial export of it fails closed.
#[tokio::test]
async fn stems_of_an_icl_render_from_a_yue2_song_inherit_its_policy() {
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
        song_result("yue2run_i", "asset_song_icl", json!({})),
        None,
    )
    .await;

    // The worker's shape (`record_song_settings` / `stem_asset_fact`): the mix's parents are the
    // ICL references; a stem replaces them with the mix and replaces `extra`.
    let mut mix = audio_fact("asset_icl_mix");
    mix.as_object_mut().unwrap().remove("extra");
    mix["parents"] = json!(["asset_song_icl"]);
    let mut stem = audio_fact("asset_icl_vocal");
    stem["parents"] = json!(["asset_icl_mix"]);
    stem["extra"] = json!({"audioStem": "vocal", "mixAssetId": "asset_icl_mix"});
    // Stem first: the resolution does not depend on the batch order.
    finish_generic_audio(&app, &state, &project_id, json!([stem, mix])).await;

    for id in ["asset_icl_mix", "asset_icl_vocal"] {
        let stored = asset(&app, &project_id, id).await;
        // Mutation that reds this: resolving parents only from the library (the pre-fix
        // `usage_policies_of` on each fact), which finds no persisted mix for the stem.
        assert_eq!(
            stored["extra"]["usagePolicies"][0]["assetId"], "asset_song_icl",
            "{id}: {stored}"
        );
    }
    let timeline_id = timeline_placing(&app, &project_id, "asset_icl_vocal").await;
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
        json!({"commercialUse": true}),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{response}");
    assert_eq!(response["code"], "commercial_use_refused");

    // A parent in neither the batch nor the library: recorded, and a commercial export refuses it.
    let mut orphan = audio_fact("asset_orphan");
    orphan.as_object_mut().unwrap().remove("extra");
    orphan["parents"] = json!(["asset_deleted_meanwhile"]);
    finish_generic_audio(&app, &state, &project_id, json!([orphan])).await;
    let stored = asset(&app, &project_id, "asset_orphan").await;
    // Mutation that reds this: `NotFound => continue` in `usage_policies_of` (the pre-fix
    // "missing = no policy").
    assert_eq!(
        stored["extra"]["usagePolicies"],
        json!([{"assetId": "asset_deleted_meanwhile", "policy": null, "unresolved": "not_found"}]),
        "{stored}"
    );
    let timeline_id = timeline_placing(&app, &project_id, "asset_orphan").await;
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
        json!({"commercialUse": true}),
    )
    .await;
    // Mutation that reds this: dropping the `unresolved` arm of `refuse_commercial_export`.
    assert_eq!(status, StatusCode::FORBIDDEN, "{response}");
    // Told apart from a noncommercial refusal: no noncommercial model is involved here.
    // Mutation that reds this: refusing unknown lineage with `COMMERCIAL_USE_REFUSED_CODE`.
    assert_eq!(
        response["code"], "commercial_use_lineage_unknown",
        "{response}"
    );
    let detail = response["detail"].as_str().unwrap();
    assert!(
        detail.contains("lineage is unknown")
            && detail.contains("asset_deleted_meanwhile")
            && !detail.contains("noncommercial assets"),
        "{detail}"
    );
}

/// Both refusals in one export: the noncommercial one wins the code, and the message still names
/// the asset whose lineage is unknown.
#[test]
fn a_noncommercial_asset_wins_the_code_over_unknown_lineage() {
    let noncommercial = json!({"assetId": "asset_nc", "policy": {"modelId": "yue2", "commercialUse": {"verdict": "refused"}}});
    let unknown = json!({"assetId": "asset_gone", "policy": null, "unresolved": "not_found"});
    let error =
        crate::yue2_jobs::refuse_commercial_export(&[noncommercial.clone(), unknown.clone()])
            .expect_err("refused");
    // Mutation that reds this: checking unknown lineage before the noncommercial verdicts.
    assert_eq!(error.code, Some("commercial_use_refused"));
    assert!(error.detail.contains("asset_nc (yue2)") && error.detail.contains("asset_gone"));
    let error = crate::yue2_jobs::refuse_commercial_export(&[unknown]).expect_err("refused");
    assert_eq!(error.code, Some("commercial_use_lineage_unknown"));
    assert!(crate::yue2_jobs::refuse_commercial_export(&[
        json!({"assetId": "ok", "policy": {"commercialUse": {"verdict": "eligible"}}})
    ])
    .is_ok());
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

/// E5: an agent renders a score version (and a cover of it) through MCP, over the same job route
/// the Song Lab uses. Without the USER's licence acceptance the render is refused with the
/// acknowledgment reason and nothing queues — the tool never acknowledges for them, and it has no
/// field to try; once the user accepts, the render queues, carries the regeneration notice and is
/// polled with `yue2_get_render`.
#[tokio::test]
async fn mcp_agent_renders_a_score_version_only_after_the_user_accepts_the_licence() {
    use rmcp::model::CallToolRequestParams;
    use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
    use rmcp::transport::StreamableHttpClientTransport;
    use rmcp::ServiceExt;
    use sceneworks_core::yue2_score::REGENERATION_NOTICE;

    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_yue1_and(&temp_dir, builtin_yue2());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral listener");
    let addr = listener.local_addr().expect("listener addr");
    let mut settings = test_settings(&temp_dir);
    settings.trust_loopback = true;
    settings.mcp_api_url = format!("http://{addr}");
    let (app, _state) = crate::create_app_with_state(settings).expect("app creates");
    let http = app.clone();
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await;
    });
    let project_id = project(&http).await;
    let (status, version) = request(
        http.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/yue2/score-versions"),
        json!({
            "abc": SCORE,
            "request": {"style": "warm piano pop", "lyrics": "[Verse]\nNeon fades", "cot": "full", "seed": 5},
            "origin": "import",
            "provenance": {"actor": "user", "channel": "ui"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{version}");
    let version_id = version["id"].as_str().unwrap().to_owned();

    let client = rmcp::model::ClientInfo::default()
        .serve(StreamableHttpClientTransport::from_config(
            StreamableHttpClientTransportConfig::with_uri(format!("http://{addr}/mcp")),
        ))
        .await
        .expect("MCP client initializes");
    let tools = client.list_tools(None).await.expect("tools/list");
    for expected in [
        "yue2_render_score_version",
        "yue2_cover_score_version",
        "yue2_get_render",
    ] {
        let tool = tools
            .tools
            .iter()
            .find(|tool| tool.name == expected)
            .unwrap_or_else(|| panic!("missing {expected}"));
        let description = tool.description.as_deref().unwrap_or("");
        assert!(
            description.contains("NONCOMMERCIAL") || expected == "yue2_get_render",
            "{expected}: {description}"
        );
    }
    let call = |name: &'static str, arguments: Value| {
        CallToolRequestParams::new(name).with_arguments(arguments.as_object().unwrap().clone())
    };
    let job_count = || async {
        request(http.clone(), "GET", "/api/v1/jobs", Value::Null)
            .await
            .1
            .as_array()
            .unwrap()
            .len()
    };

    // No acceptance yet: refused with the reason, nothing queued.
    let refused = client
        .call_tool(call(
            "yue2_render_score_version",
            json!({"projectId": project_id, "versionId": version_id, "computePolicy": "auto"}),
        ))
        .await
        .expect("a refusal is a tool result");
    assert_eq!(refused.is_error, Some(true), "{refused:?}");
    let text = format!("{refused:?}");
    // Mutation that reds this: `submit_render` sending `licenseAcknowledged: true`.
    assert!(
        text.contains("license_acknowledgment_required") && text.contains("USER must"),
        "{text}"
    );
    assert_eq!(job_count().await, 0, "a refused render queues nothing");
    // The agent cannot acknowledge through the tool: there is no such field.
    let smuggled = client
        .call_tool(call(
            "yue2_render_score_version",
            json!({"projectId": project_id, "versionId": version_id, "computePolicy": "auto", "licenseAcknowledged": true}),
        ))
        .await;
    assert!(
        smuggled.is_err() || smuggled.as_ref().unwrap().is_error == Some(true),
        "{smuggled:?}"
    );
    assert_eq!(job_count().await, 0);

    // The user accepts in SceneWorks; now the agent's render queues.
    let (status, _) = request(
        http.clone(),
        "PUT",
        "/api/v1/models/yue2/license-acknowledgment",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let rendered = client
        .call_tool(call(
            "yue2_render_score_version",
            json!({"projectId": project_id, "versionId": version_id, "steps": 8, "computePolicy": "auto"}),
        ))
        .await
        .expect("render call");
    assert_ne!(rendered.is_error, Some(true), "{rendered:?}");
    let rendered = mcp_tool_content_json(&rendered);
    assert_eq!(rendered["renderNotice"], REGENERATION_NOTICE);
    assert_eq!(rendered["jobs"][0]["kind"], "renderVersion");
    assert_eq!(rendered["usagePolicy"]["nonCommercial"], true);
    let job_id = rendered["jobs"][0]["jobId"].as_str().unwrap().to_owned();
    let stored = job(&http, &job_id).await;
    assert_eq!(stored["payload"]["yue2"]["versionId"], version_id);
    assert_eq!(stored["payload"]["yue2"]["steps"], 8);
    assert_eq!(stored["payload"]["yue2"]["computePolicy"], "auto");
    assert_eq!(stored["payload"]["commercialUse"], false);

    let polled = client
        .call_tool(call("yue2_get_render", json!({"jobId": job_id})))
        .await
        .expect("poll call");
    assert_ne!(polled.is_error, Some(true), "{polled:?}");
    let polled = mcp_tool_content_json(&polled);
    assert_eq!(polled["status"], "queued");
    assert_eq!(polled["yue2"]["kind"], "renderVersion");
    assert_eq!(polled["yue2"]["versionId"], version_id);
    assert_eq!(polled["renderNotice"], REGENERATION_NOTICE);

    let covered = client
        .call_tool(call(
            "yue2_cover_score_version",
            json!({"projectId": project_id, "versionId": version_id, "mode": "melody",
                   "keep": "vocal", "lyrics": "[Verse]\nNuevas palabras",
                   "translatedFrom": "[Verse]\nNew words", "computePolicy": "auto"}),
        ))
        .await
        .expect("cover call");
    assert_ne!(covered.is_error, Some(true), "{covered:?}");
    let covered = mcp_tool_content_json(&covered);
    assert_eq!(covered["jobs"][0]["kind"], "cover");
    // The translation's source lyrics reach the stored cover block (round-2 item 3).
    // Mutation that reds this: dropping the `translatedFrom` insert from `cover_body`.
    let cover_job = job(&http, covered["jobs"][0]["jobId"].as_str().unwrap()).await;
    assert_eq!(
        cover_job["payload"]["yue2"]["cover"]["translatedFrom"], "[Verse]\nNew words",
        "{cover_job}"
    );
    assert_eq!(cover_job["payload"]["yue2"]["computePolicy"], "auto");
    assert_eq!(covered["renderNotice"], REGENERATION_NOTICE);

    let missing = client
        .call_tool(call("yue2_get_render", json!({"jobId": "job_missing"})))
        .await
        .expect("a missing job is a tool result");
    assert_eq!(missing.is_error, Some(true));

    let _ = client.cancel().await;
}

/// sc-22988 review item 7: the web lab's field table is core `FIELD_KINDS`, and every body its
/// request builder produces deserializes through the route's `Yue2JobSpec` (deny_unknown_fields)
/// and passes `validate_request`. The web suite pins the same file to what the builder sends, so a
/// drift on either side reds one of the two suites.
#[test]
fn the_web_lab_request_bodies_deserialize_and_validate_against_the_core_contract() {
    use sceneworks_core::yue2_score::jobs::{self as contract, Yue2JobSpec};
    let fixture: Value = serde_json::from_str(WEB_JOB_REQUESTS).expect("the fixture is JSON");
    let core: serde_json::Map<String, Value> = contract::FIELD_KINDS
        .iter()
        .map(|(field, kinds)| {
            let kinds: Vec<&str> = kinds.iter().map(|kind| kind.as_str()).collect();
            ((*field).to_owned(), json!(kinds))
        })
        .collect();
    // Mutation that reds this: adding or removing one kind in core `FIELD_KINDS`.
    assert_eq!(fixture["fieldKinds"], Value::Object(core));
    let bodies = fixture["bodies"].as_array().expect("bodies");
    assert!(bodies.len() >= 6, "{fixture}");
    for body in bodies {
        // Mutation that reds this: a field the builder sends that `Yue2JobSpec` does not declare.
        let spec: Yue2JobSpec =
            serde_json::from_value(body.clone()).unwrap_or_else(|error| panic!("{body}: {error}"));
        contract::validate_new_submission(&spec).unwrap_or_else(|error| panic!("{body}: {error}"));
    }
}
