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
async fn film_reference_routes_stage_assets_roundtrip_bindings_and_pin_run_inputs() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let settings = test_settings(&temporary);
    let app = create_app(settings.clone()).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Film references"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "courier.png",
        "image/png",
        PNG_32X32,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{asset}");
    let asset_id = asset["id"].as_str().unwrap();

    let (_, draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Mixed references"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap();
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/references"),
        json!({
            "draftRevision": 1,
            "assetId": asset_id,
            "role": "courier",
            "kind": "costume",
            "approved": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"].as_str().unwrap().contains("Reference kind"));

    let (status, mut referenced) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/references"),
        json!({
            "draftRevision": 1,
            "assetId": asset_id,
            "role": "courier",
            "kind": "character",
            "description": "Blue jacket",
            "approved": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{referenced}");
    assert_eq!(referenced["revision"], 2);
    assert_eq!(
        referenced["referencePack"]["references"][0]["sourceAssetId"],
        asset_id
    );

    let (status, exported) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/reference-pack"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(exported, referenced["referencePack"]);
    let mut imported = exported.clone();
    imported["description"] = json!("Imported without dropping fields");
    let (status, imported_draft) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/reference-pack"),
        json!({"draftRevision": 2, "referencePack": imported}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{imported_draft}");
    assert_eq!(
        imported_draft["referencePack"]["references"][0]["sourceAssetId"],
        asset_id
    );

    referenced = imported_draft;
    referenced["productionPlan"]["shots"][0]["prompt"] = json!("Courier close-up");
    referenced["productionPlan"]["shots"][0]["conditioning"] = json!({
        "mode": "reference_to_video",
        "referenceRoles": ["courier"]
    });
    let mut plain_shot = referenced["productionPlan"]["shots"][0].clone();
    plain_shot["id"] = json!("SH020");
    plain_shot["prompt"] = json!("An empty workshop establishing shot");
    plain_shot["conditioning"] = json!({"mode": "text_to_video", "referenceRoles": []});
    plain_shot["continuityRoles"] = json!([]);
    referenced["productionPlan"]["shots"]
        .as_array_mut()
        .unwrap()
        .push(plain_shot);
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        referenced,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");

    let (status, run) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/runs"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{run}");
    let root = std::path::Path::new(project["path"].as_str().unwrap());
    let run_dir = root.join(run["locator"]["recordDirectory"].as_str().unwrap());
    let pinned_pack: Value = serde_json::from_slice(
        &std::fs::read(run_dir.join("references.json")).expect("pinned pack reads"),
    )
    .expect("pinned pack parses");
    let pinned_file = pinned_pack["references"][0]["file"].as_str().unwrap();
    assert!(run_dir.join(pinned_file).is_file());

    let mut edited = saved;
    edited["referencePack"]["references"][0]["description"] = json!("Green jacket now");
    let (status, edited) = request(
        app,
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        edited,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{edited}");
    assert_eq!(
        edited["revision"].as_u64(),
        run["locator"]["draftRevision"]
            .as_u64()
            .map(|value| value + 1)
    );
    let still_pinned: Value = serde_json::from_slice(
        &std::fs::read(run_dir.join("references.json")).expect("pinned pack remains"),
    )
    .expect("pinned pack parses");
    assert_eq!(
        still_pinned, pinned_pack,
        "editing the draft cannot rewrite a run pin"
    );
}

#[tokio::test]
async fn unapproved_reference_binding_is_a_named_finding() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Approval"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "look.png",
        "image/png",
        PNG_32X32,
    )
    .await;
    let (_, draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Approval"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap();
    let (_, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/references"),
        json!({"draftRevision": 1, "assetId": asset["id"], "role": "hero", "kind": "character", "approved": false}),
    )
    .await;
    draft["productionPlan"]["shots"][0]["prompt"] = json!("Hero enters");
    draft["productionPlan"]["shots"][0]["conditioning"] =
        json!({"mode": "reference_to_video", "referenceRoles": ["hero"]});
    let (_, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    let (status, error) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/runs"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{saved}");
    let detail = error["detail"].as_str().unwrap();
    assert!(
        detail.contains("SH010") && detail.contains("hero") && detail.contains("not approved"),
        "{detail}"
    );
}
