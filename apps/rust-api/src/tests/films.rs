use super::support::*;

#[tokio::test]
async fn film_routes_create_edit_reopen_and_pin_a_reference_free_draft() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let settings = test_settings(&temporary);
    let app = create_app(settings.clone()).expect("app creates");
    let (status, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Film route project"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let project_id = project["id"].as_str().unwrap();

    let (status, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Manual film"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{draft}");
    assert_eq!(draft["planning"]["provider"], "prompt_refiner");
    assert_eq!(draft["referencePack"]["references"], json!([]));
    let draft_id = draft["id"].as_str().unwrap().to_owned();

    draft["title"] = json!("Workshop delivery");
    draft["productionPlan"]["shots"][0]["prompt"] =
        json!("A courier enters a workshop carrying a red parcel.");
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");
    assert_eq!(saved["revision"], 2);

    let restarted = create_app(settings).expect("restarted app creates");
    let (status, reopened) = request(
        restarted.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{reopened}");
    assert_eq!(reopened["title"], "Workshop delivery");

    let (status, run) = request(
        restarted,
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/runs"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{run}");
    assert_eq!(run["locator"]["draftRevision"], 2);
    assert!(run["record"].is_null());
    let relative = run["locator"]["recordDirectory"].as_str().unwrap();
    assert!(!relative.starts_with('/'));
    let root = std::path::Path::new(project["path"].as_str().unwrap());
    assert!(root.join(relative).join("plan.json").exists());
    assert!(root.join(relative).join("references.json").exists());
}

#[tokio::test]
async fn invalid_manual_shot_is_reported_before_any_run_or_video_job_exists() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Invalid film"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (_, draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Incomplete"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap();
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/runs"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error["detail"].as_str().unwrap().contains("prompt"));
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs, json!([]), "validation must dispatch no job");
}

#[tokio::test]
async fn film_script_parse_and_unavailable_qwen_preserve_the_draft_and_manual_path() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Planning project"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (_, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Screenplay"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap().to_owned();
    let source = "INT. WORKSHOP - NIGHT\nA courier arrives.\n\nMARA\nPut it on the bench.";
    let (status, parsed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/brief/parse"),
        json!({"script": source}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{parsed}");
    assert!(parsed["beats"].as_array().unwrap().len() >= 2, "{parsed}");
    assert_eq!(parsed["dialogue"][0]["speaker"], "MARA");

    draft["originalScript"] = json!(source);
    draft["structuredBrief"] = parsed;
    draft["planning"] = json!({
        "provider": "native",
        "modelId": "film_planner_qwen3_6_27b",
        "thinkingMode": "enabled",
        "refinePrompts": false
    });
    draft["productionPlan"]["shots"][0]["prompt"] = json!("Manual prompt stays intact.");
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");

    let (status, operation) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/planning"),
        json!({"maxRepairRounds": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{operation}");
    assert_eq!(operation["status"], "failed", "{operation}");
    assert_eq!(operation["stage"], "unavailable");
    assert!(operation["findings"][0]["message"]
        .as_str()
        .unwrap()
        .contains("not installed"));
    assert_eq!(operation["plannerModel"], "Qwen/Qwen3.6-27B");
    assert_eq!(operation["videoModelId"], "minimax_h3");

    let (_, reopened) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(reopened["originalScript"], source);
    assert_eq!(
        reopened["productionPlan"]["shots"][0]["prompt"],
        "Manual prompt stays intact."
    );
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        jobs,
        json!([]),
        "unavailable planning must dispatch nothing"
    );
}

#[tokio::test]
async fn native_planner_checkpoint_is_routed_separately_from_the_video_model() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "prompt": "Return a film plan JSON object.",
            "task": "film_plan",
            "workflow": "video",
            "model": "Qwen/Qwen3.6-27B",
            "modelId": "minimax_h3",
            "thinkingMode": "enabled"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["model"], "Qwen/Qwen3.6-27B");
    assert_eq!(job["payload"]["modelId"], "minimax_h3");
    assert_eq!(job["payload"]["thinkingMode"], "enabled");
}

#[tokio::test]
async fn saved_external_connection_keeps_credentials_out_of_projects_and_fails_closed_when_missing()
{
    let temporary = tempfile::tempdir().expect("temp dir");
    let settings = test_settings(&temporary);
    let app = create_app(settings.clone()).expect("app creates");

    let secret = "planner-secret-must-stay-in-secret-store";
    let (status, credentials) = request(
        app.clone(),
        "PUT",
        "/api/v1/credentials",
        json!({
            "host": "planner.fixture.test",
            "label": "Planner fixture",
            "scheme": "bearer",
            "token": secret
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{credentials}");
    assert!(!credentials.to_string().contains(secret));

    let (status, mismatch) = request(
        app.clone(),
        "PUT",
        "/api/v1/film-planner-connections/mismatched-secret",
        json!({
            "label": "Mismatched",
            "baseUrl": "https://different.fixture.test/v1",
            "credentialHost": "planner.fixture.test",
            "supportsModelListing": false,
            "supportsImageInput": false,
            "timeoutSeconds": 30,
            "maxOutputTokens": 4096
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{mismatch}");
    assert!(mismatch["detail"]
        .as_str()
        .unwrap()
        .contains("base URL host"));

    let (status, saved_connection) = request(
        app.clone(),
        "PUT",
        "/api/v1/film-planner-connections/fixture",
        json!({
            "label": "Fixture",
            "baseUrl": "https://planner.fixture.test/v1",
            "credentialHost": "planner.fixture.test",
            "supportsModelListing": false,
            "supportsImageInput": false,
            "timeoutSeconds": 30,
            "maxOutputTokens": 4096
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved_connection}");
    assert!(!saved_connection.to_string().contains(secret));

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "External planning boundary"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (_, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "External planning"}),
    )
    .await;
    assert_eq!(draft["planning"]["provider"], "prompt_refiner");
    let draft_id = draft["id"].as_str().unwrap().to_owned();
    draft["originalScript"] = json!("A courier crosses a quiet room.");
    draft["planning"] = json!({
        "provider": "openai_compatible",
        "connectionId": "fixture",
        "modelId": "fixture-model",
        "thinkingMode": "disabled",
        "refinePrompts": false,
        "sendReferencePixels": false
    });
    let (status, saved_draft) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved_draft}");
    assert_eq!(saved_draft["planning"]["connectionId"], "fixture");
    assert_eq!(saved_draft["productionPlan"]["model"]["id"], "minimax_h3");
    assert!(!saved_draft.to_string().contains(secret));

    fn tree_contains(root: &std::path::Path, needle: &[u8]) -> bool {
        std::fs::read_dir(root)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .any(|entry| {
                let path = entry.path();
                if path.is_dir() {
                    tree_contains(&path, needle)
                } else {
                    std::fs::read(path).ok().is_some_and(|bytes| {
                        bytes.windows(needle.len()).any(|window| window == needle)
                    })
                }
            })
    }
    assert!(!tree_contains(
        std::path::Path::new(project["path"].as_str().unwrap()),
        secret.as_bytes()
    ));
    let connection_bytes = std::fs::read(settings.config_dir.join("film-planner-connections.json"))
        .expect("connection settings saved");
    assert!(!String::from_utf8_lossy(&connection_bytes).contains(secret));
    assert!(
        std::fs::read(settings.credentials_dir.join("credentials.json"))
            .unwrap()
            .windows(secret.len())
            .any(|window| window == secret.as_bytes())
    );

    std::fs::remove_file(settings.credentials_dir.join("credentials.json")).unwrap();
    let (status, operation) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/planning"),
        json!({"maxRepairRounds": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{operation}");
    assert_eq!(operation["status"], "failed");
    assert_eq!(operation["stage"], "authentication");
    assert!(operation["findings"][0]["message"]
        .as_str()
        .unwrap()
        .contains("missing"));
}

#[tokio::test]
async fn generated_plan_is_only_installed_by_explicit_revision_checked_apply() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Candidate apply project"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (_, draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Candidate review"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap();
    let mut candidate = draft["productionPlan"].clone();
    candidate["shots"][0]["prompt"] = json!("Generated candidate prompt.");
    let planning_root = std::path::Path::new(project["path"].as_str().unwrap())
        .join("films")
        .join("planning")
        .join(draft_id);
    std::fs::create_dir_all(&planning_root).unwrap();
    std::fs::write(
        planning_root.join("latest.json"),
        serde_json::to_vec_pretty(&json!({
            "schemaVersion": 1,
            "id": "filmplan_candidate",
            "projectId": project_id,
            "draftId": draft_id,
            "draftRevision": 1,
            "status": "ready",
            "stage": "review",
            "provider": "prompt_refiner",
            "plannerModel": "TheDrummer/Anubis-Mini-8B-v1",
            "videoModelId": "minimax_h3",
            "thinkingMode": "disabled",
            "jobIds": [],
            "findings": [],
            "executions": [],
            "candidatePlan": candidate,
            "createdAt": "2026-09-16T00:00:00Z",
            "updatedAt": "2026-09-16T00:00:00Z"
        }))
        .unwrap(),
    )
    .unwrap();

    let (_, before) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        Value::Null,
    )
    .await;
    assert_ne!(
        before["productionPlan"]["shots"][0]["prompt"],
        "Generated candidate prompt."
    );

    let (status, applied) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/planning/apply"),
        json!({"operationId": "filmplan_candidate"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(
        applied["productionPlan"]["shots"][0]["prompt"],
        "Generated candidate prompt."
    );

    let (status, stale) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/planning/apply"),
        json!({"operationId": "filmplan_candidate"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{stale}");
    assert!(stale["detail"].as_str().unwrap().contains("changed after"));
}
