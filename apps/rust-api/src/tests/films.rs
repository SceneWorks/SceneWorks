use super::support::*;

#[tokio::test]
async fn film_render_options_are_previewable_and_preserve_explicit_or_legacy_controls() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Render choices"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (status, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Render choices"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{draft}");
    let draft_id = draft["id"].as_str().unwrap().to_owned();
    assert_eq!(
        draft["renderRegime"], "quality",
        "the empty test catalog cannot honestly select Turbo"
    );

    let route = format!("/api/v1/projects/{project_id}/films/{draft_id}/render-options");
    let (status, options) = request(app.clone(), "GET", &route, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{options}");
    assert_eq!(options["selectedRegime"], "quality");
    assert_eq!(options["recommendedTurbo"]["available"], false);
    assert_eq!(
        options["recommendedTurbo"]["unavailableReason"],
        "model_unavailable"
    );
    assert_eq!(options["quality"]["adapterIds"], json!([]));

    let (status, stale) = request(
        app.clone(),
        "POST",
        &route,
        json!({
            "draftRevision": 0,
            "productionPlan": draft["productionPlan"],
            "referencePack": draft["referencePack"],
            "renderRegime": "custom"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{stale}");

    let (status, preview) = request(
        app.clone(),
        "POST",
        &route,
        json!({
            "draftRevision": 1,
            "productionPlan": draft["productionPlan"],
            "referencePack": draft["referencePack"],
            "renderRegime": "custom"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{preview}");
    assert_eq!(preview["selectedRegime"], "custom");

    let (status, legacy_preview) = request(
        app.clone(),
        "POST",
        &route,
        json!({
            "draftRevision": 1,
            "productionPlan": draft["productionPlan"],
            "referencePack": draft["referencePack"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{legacy_preview}");
    assert_eq!(legacy_preview["selectedRegime"], "custom");

    draft["renderRegime"] = json!("custom");
    draft["productionPlan"]["model"]["loras"] = json!(["minimax_h3_turbo_8step"]);
    draft["productionPlan"]["model"]["advanced"] = json!({"steps": 7});
    let (status, mut custom) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{custom}");
    assert_eq!(
        custom["productionPlan"]["model"]["loras"],
        json!(["minimax_h3_turbo_8step"])
    );
    assert_eq!(custom["productionPlan"]["model"]["advanced"]["steps"], 7);

    custom["renderRegime"] = json!("quality");
    let (status, mut quality) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        custom,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{quality}");
    assert!(quality["productionPlan"]["model"]["loras"].is_null());
    assert!(quality["productionPlan"]["model"]["advanced"].is_null());

    quality.as_object_mut().unwrap().remove("renderRegime");
    quality["productionPlan"]["model"]["loras"] = json!(["minimax_h3_turbo_8step"]);
    quality["productionPlan"]["model"]["advanced"] = json!({"steps": 9});
    let (status, legacy) = request(
        app,
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        quality,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{legacy}");
    assert!(legacy.get("renderRegime").is_none());
    assert_eq!(
        legacy["productionPlan"]["model"]["loras"],
        json!(["minimax_h3_turbo_8step"])
    );
    assert_eq!(legacy["productionPlan"]["model"]["advanced"]["steps"], 9);
}

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
        restarted.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/runs"),
        json!({"selectedShotIds": ["SH010"]}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{run}");
    assert_eq!(run["locator"]["draftRevision"], 2);
    assert_eq!(run["locator"]["selectedShotIds"], json!(["SH010"]));
    assert!(run["record"].is_null());
    let relative = run["locator"]["recordDirectory"].as_str().unwrap();
    assert!(!relative.starts_with('/'));
    let root = std::path::Path::new(project["path"].as_str().unwrap());
    assert!(root.join(relative).join("plan.json").exists());
    assert!(root.join(relative).join("references.json").exists());

    let (status, runs) = request(
        restarted.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/film-runs"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{runs}");
    assert_eq!(runs.as_array().unwrap().len(), 1);
    assert_eq!(runs[0]["locator"]["id"], run["locator"]["id"]);
    assert_eq!(runs[0]["controllerActive"], false);
    let run_id = run["locator"]["id"].as_str().unwrap();
    let (status, progress) = request(
        restarted.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/film-runs/{run_id}/progress"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{progress}");
    assert!(progress["record"].is_null());
    for action in ["resume", "cancel"] {
        let (status, error) = request(
            restarted.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/film-runs/{run_id}/{action}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{action}: {error}");
        assert!(error["detail"].as_str().unwrap().contains("not started"));
    }
}

#[tokio::test]
async fn local_planning_persists_a_positive_timeout_and_rejects_zero_before_dispatch() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Bounded local planner"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (_, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Bounded local planner"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap().to_owned();
    draft["originalScript"] = json!("A courier enters a quiet workshop.");
    draft["renderRegime"] = json!("custom");
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");

    let route = format!("/api/v1/projects/{project_id}/films/{draft_id}/planning");
    let (status, error) = request(
        app.clone(),
        "POST",
        &route,
        json!({"maxRepairRounds": 0, "llmTimeoutSeconds": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error["detail"]
        .as_str()
        .unwrap()
        .contains("at least 1 second"));
    let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs, json!([]), "invalid timeout must dispatch nothing");

    let (status, operation) = request(
        app,
        "POST",
        &route,
        json!({"maxRepairRounds": 0, "llmTimeoutSeconds": 37}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{operation}");
    assert_eq!(operation["llmTimeoutSeconds"], 37);
    assert_eq!(operation["maxRepairRounds"], 0);
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
    referenced["originalScript"] = json!("A courier crosses the workshop carrying a parcel.");
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        referenced,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");

    let (status, planning) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/planning"),
        json!({"maxRepairRounds": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{planning}");
    let root = std::path::Path::new(project["path"].as_str().unwrap());
    let operation_id = planning["id"].as_str().unwrap();
    let operation_dir = root
        .join("films/planning")
        .join(draft_id)
        .join("operations")
        .join(operation_id);
    let planning_pack: Value = serde_json::from_slice(
        &std::fs::read(operation_dir.join("references.json"))
            .expect("planning reference pack reads"),
    )
    .expect("planning reference pack parses");
    let planning_file = planning_pack["references"][0]["file"].as_str().unwrap();
    let staged_image = operation_dir.join(planning_file);
    assert_eq!(std::fs::read(&staged_image).unwrap(), PNG_32X32);
    assert!(
        staged_image
            .canonicalize()
            .unwrap()
            .starts_with(operation_dir.canonicalize().unwrap()),
        "planning reference bytes must remain inside the operation"
    );
    let draft_image = root
        .join("films/draft-assets")
        .join(draft_id)
        .join(planning_file);
    std::fs::write(&draft_image, b"later draft mutation").unwrap();
    assert_eq!(
        std::fs::read(&staged_image).unwrap(),
        PNG_32X32,
        "planning keeps an immutable byte snapshot"
    );
    std::fs::write(&draft_image, PNG_32X32).unwrap();

    let (status, run) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/runs"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{run}");
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
async fn film_sound_route_stages_prerecorded_audio_and_pins_it_with_the_run() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Film sound"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let wav = crate::film_harness::fixture_sound_wav(1.0, 440, 4_000);
    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "line.wav",
        "audio/wav",
        &wav,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{asset}");
    let (_, draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Recorded line"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap();
    let (status, mut staged) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/sound"),
        json!({
            "draftRevision": 1,
            "assetId": asset["id"],
            "role": "courier_line",
            "kind": "dialogue",
            "description": "Courier's recorded line"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{staged}");
    assert_eq!(staged["referencePack"]["sound"][0]["role"], "courier_line");
    let staged_file = staged["referencePack"]["sound"][0]["file"]
        .as_str()
        .unwrap()
        .to_owned();
    staged["productionPlan"]["shots"][0]["prompt"] = json!("A courier speaks.");
    staged["productionPlan"]["shots"][0]["dialogue"] = json!("The parcel is here.");
    staged["productionPlan"]["shots"][0]["dialogueClip"] = json!({
        "role": "courier_line", "offsetSeconds": 0.25, "sourceInSeconds": 0,
        "durationSeconds": 0.7, "gain": 0.8, "fadeInSeconds": 0.05, "fadeOutSeconds": 0.05
    });
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        staged,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");
    let (status, run) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}/runs"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{run}");
    let run_dir = std::path::Path::new(project["path"].as_str().unwrap())
        .join(run["locator"]["recordDirectory"].as_str().unwrap());
    assert_eq!(std::fs::read(run_dir.join(&staged_file)).unwrap(), wav);
    let pinned: Value =
        serde_json::from_slice(&std::fs::read(run_dir.join("references.json")).unwrap()).unwrap();
    assert_eq!(pinned["sound"][0]["role"], "courier_line");
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
        json!({"maxRepairRounds": 2, "llmTimeoutSeconds": 37}),
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
    assert_eq!(operation["llmTimeoutSeconds"], 37);

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
        json!({"maxRepairRounds": 1, "llmTimeoutSeconds": 0}),
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
    let (_, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title": "Candidate review"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap().to_owned();
    draft["reviewPlan"]["shots"]["SH010"]["questions"][0]["ask"] =
        json!("Is the custom authored action visible?");
    let (status, draft) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{draft}");
    let authored_review = draft["reviewPlan"]["shots"]["SH010"].clone();
    let mut candidate = draft["productionPlan"].clone();
    candidate["shots"][0]["prompt"] = json!("Generated candidate prompt.");
    let mut added_shot = candidate["shots"][0].clone();
    added_shot["id"] = json!("SH020");
    candidate["shots"].as_array_mut().unwrap().push(added_shot);
    let planning_root = std::path::Path::new(project["path"].as_str().unwrap())
        .join("films")
        .join("planning")
        .join(&draft_id);
    std::fs::create_dir_all(&planning_root).unwrap();
    let mut compiled = json!({
        "schemaVersion": 3,
        "planId": draft_id,
        "planVersion": 1,
        "planSha256": "synthetic-candidate-fixture",
        "referencePackId": format!("{draft_id}-references"),
        "referencePackVersion": 1,
        "compiledAt": "2026-09-16T00:00:00Z",
        "model": {"id": "minimax_h3", "tier": "q4", "fps": 24, "lane": "mlx"},
        "requests": [{
            "shotId": "SH010", "beat": "Opening shot", "mode": "text_to_video",
            "model": "minimax_h3", "partitionReason": "base conditioning",
            "prompt": "Generated candidate prompt.", "promptSource": "refined",
            "authoredPrompt": "", "durationSeconds": 5.1667, "fps": 24,
            "width": 576, "height": 320, "referenceRoles": [], "continuityRoles": []
        }]
    });
    let mut second_request = compiled["requests"][0].clone();
    second_request["shotId"] = json!("SH020");
    compiled["requests"]
        .as_array_mut()
        .unwrap()
        .push(second_request);
    std::fs::write(
        planning_root.join("latest.json"),
        serde_json::to_vec_pretty(&json!({
            "schemaVersion": 1,
            "id": "filmplan_candidate",
            "projectId": project_id,
            "draftId": draft_id,
            "draftRevision": draft["revision"],
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
            "compiled": compiled,
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
    assert_eq!(applied["reviewPlan"]["shots"]["SH010"], authored_review);
    assert!(!applied["reviewPlan"]["shots"]["SH020"]["questions"]
        .as_array()
        .unwrap()
        .is_empty());
    let (_, reopened) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(reopened["reviewPlan"], applied["reviewPlan"]);
    for shot in reopened["productionPlan"]["shots"].as_array().unwrap() {
        assert!(reopened["reviewPlan"]["shots"]
            .get(shot["id"].as_str().unwrap())
            .is_some());
    }
    assert_eq!(
        applied["productionPlan"]["shots"][0]["prompt"],
        "Generated candidate prompt."
    );
    assert_eq!(
        applied["compiledPlan"]["planVersion"],
        applied["productionPlan"]["version"]
    );
    assert_eq!(
        applied["compiledPlan"]["planSha256"]
            .as_str()
            .unwrap()
            .len(),
        64
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

#[tokio::test]
async fn film_render_refuses_an_authorized_revision_changed_by_another_session() {
    let temporary = tempfile::tempdir().unwrap();
    let settings = test_settings(&temporary);
    let app = create_app(settings).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Revision"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let (_, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title":"Revision"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap().to_owned();
    let authorized_revision = draft["revision"].clone();
    draft["productionPlan"]["shots"][0]["prompt"] = json!("Another session changed the prompt");
    let (status, _) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    for suffix in ["preflight", "runs"] {
        let (status, refusal) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/films/{draft_id}/{suffix}"),
            json!({"expectedDraftRevision": authorized_revision, "selectedShotIds":["SH010"]}),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{refusal}");
        assert!(refusal.to_string().contains("revision conflict"));
    }
    let (_, runs) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/film-runs"),
        Value::Null,
    )
    .await;
    assert_eq!(runs, json!([]));
    let directory = std::path::Path::new(project["path"].as_str().unwrap()).join("films/runs");
    assert_eq!(std::fs::read_dir(directory).unwrap().count(), 0);
    let (status, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(jobs, json!([]));
}

#[tokio::test]
async fn film_render_store_pin_rechecks_revision_after_route_validation() {
    let temporary = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temporary)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"Pin race"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap().to_owned();
    let (_, mut draft) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/films"),
        json!({"title":"Pin race"}),
    )
    .await;
    let draft_id = draft["id"].as_str().unwrap().to_owned();
    draft["productionPlan"]["shots"][0]["prompt"] =
        json!("A courier walks across a quiet workshop.");
    let (_, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        draft,
    )
    .await;
    let (validated, resume) = crate::films::film_pin_barrier(&draft_id);
    let app_request = app.clone();
    let route = format!("/api/v1/projects/{project_id}/films/{draft_id}/runs");
    let revision = saved["revision"].clone();
    let rendering = tokio::spawn(async move {
        request(
            app_request,
            "POST",
            &route,
            json!({"expectedDraftRevision":revision, "selectedShotIds":["SH010"]}),
        )
        .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(10), validated)
        .await
        .unwrap()
        .unwrap();
    let mut concurrent = saved;
    concurrent["productionPlan"]["shots"][0]["prompt"] = json!("Changed after route validation");
    let (status, _) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
        concurrent,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    resume.send(()).unwrap();
    let (status, conflict) = rendering.await.unwrap();
    assert_eq!(status, StatusCode::CONFLICT, "{conflict}");
    assert!(conflict.to_string().contains("revision conflict"));
    let (_, runs) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/film-runs"),
        Value::Null,
    )
    .await;
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(runs, json!([]));
    assert_eq!(jobs, json!([]));
}

#[tokio::test]
async fn planning_repair_ceiling_is_rejected_before_creating_an_operation() {
    let temporary = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temporary)).unwrap();
    let (status, failure) = request(
        app.clone(),
        "POST",
        "/api/v1/projects/no-project/films/no-draft/planning",
        json!({"maxRepairRounds": 999}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{failure}");
    assert!(failure["detail"]
        .as_str()
        .unwrap()
        .contains("between 0 and 5"));
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs, json!([]));
}
