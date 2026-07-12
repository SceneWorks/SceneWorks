//! rust-api projects tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[tokio::test]
async fn project_and_asset_routes_persist_contract_state() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "My Project" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(created["id"]
        .as_str()
        .is_some_and(|value| value.starts_with("project_")));
    assert!(created["path"]
        .as_str()
        .unwrap()
        .ends_with("my-project.sceneworks"));

    let project_id = created["id"].as_str().expect("project id").to_owned();
    let (status, projects) = request(app.clone(), "GET", "/api/v1/projects", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(projects[0]["id"], project_id);

    let (status, uploaded) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Hero Image.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(uploaded["projectId"], project_id);
    assert_eq!(uploaded["type"], "image");
    assert_eq!(uploaded["status"]["trashed"], false);
    assert!(uploaded["url"]
        .as_str()
        .unwrap()
        .contains("/files/assets/uploads/"));

    let (status, heic_upload) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Plate.HEIC",
        "application/octet-stream",
        b"heic-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(heic_upload["type"], "image");
    assert_eq!(heic_upload["file"]["mimeType"], "image/heic");

    let asset_id = uploaded["id"].as_str().expect("asset id").to_owned();
    let (status, assets) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets?includeRejected=true&includeTrashed=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(assets.as_array().unwrap().len(), 2);

    let (status, detail) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail["id"], asset_id);

    let (status, updated) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}/status"),
        json!({ "favorite": true, "rating": 4, "rejected": true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["status"]["favorite"], true);
    assert_eq!(updated["status"]["rating"], 4);
    assert_eq!(updated["status"]["rejected"], true);

    let (status, tagged) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
        json!({ "tags": [" Portrait ", "portrait", "Reference"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(tagged["tags"], json!(["portrait", "reference"]));

    let (status, deleted) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted, json!({ "id": asset_id, "status": "trashed" }));

    let (status, reindex) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/reindex"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reindex["assets"], 2);

    // permanent=true exercises the deterministic hard-delete path; the default
    // (move-to-OS-trash) is environment-dependent and validated manually.
    let (status, purged) = request(
        app,
        "DELETE",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}/purge?permanent=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(purged, json!({ "id": asset_id, "status": "purged" }));
}

#[tokio::test]
async fn timeline_routes_persist_and_create_worker_jobs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, created_project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Timeline Project" }),
    )
    .await;
    let project_id = created_project["id"]
        .as_str()
        .expect("project id")
        .to_owned();

    let (status, mut timeline) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines"),
        json!({ "name": "Main timeline", "aspectRatio": "16:9", "fps": 30 }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(timeline["projectId"], project_id);
    assert_eq!(timeline["tracks"].as_array().unwrap().len(), 3);

    let timeline_id = timeline["id"].as_str().expect("timeline id").to_owned();
    timeline["tracks"][0]["items"] = json!([
        {
            "id": "item-1",
            "trackId": "track_main",
            "assetId": "asset-1",
            "type": "video",
            "displayName": "Clip",
            "sourceIn": 2,
            "sourceOut": 6,
            "timelineStart": 10,
            "timelineEnd": 14,
            "speed": 1,
            "fit": "fit",
            "volume": 1
        }
    ]);
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": timeline }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["duration"].as_f64(), Some(14.0));
    assert_eq!(
        saved["tracks"][0]["items"][0]["currentVersionAssetId"],
        "asset-1"
    );
    assert_eq!(
        saved["tracks"][0]["items"][0]["versionHistory"][0]["source"],
        "original"
    );

    let (status, timelines) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/timelines"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(timelines[0]["id"], timeline_id);
    assert_eq!(
        timelines[0]["filePath"],
        format!(
            "timelines/main-timeline-{}.sceneworks.timeline.json",
            &timeline_id[timeline_id.len() - 8..]
        )
    );

    let (status, export_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
        json!({ "resolution": 720, "fps": 30, "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(export_job["type"], "timeline_export");
    assert_eq!(export_job["payload"]["timelineId"], timeline_id);

    let (status, frame_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/items/item-1/frames"),
        json!({ "playheadSeconds": 12.5, "intendedUse": "first_frame" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(frame_job["type"], "frame_extract");
    assert_eq!(frame_job["payload"]["sourceAssetId"], "asset-1");
    assert_eq!(frame_job["payload"]["sourceTimestamp"], 4.5);

    let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["queued"], 2);
}

#[tokio::test]
async fn timeline_routes_reject_invalid_payloads() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, created_project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Invalid Timeline Project" }),
    )
    .await;
    let project_id = created_project["id"]
        .as_str()
        .expect("project id")
        .to_owned();

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines"),
        json!({ "name": "Main timeline", "aspectRatio": "4:3", "fps": 30 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (_, mut timeline) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines"),
        json!({ "name": "Main timeline" }),
    )
    .await;
    let timeline_id = timeline["id"].as_str().expect("timeline id").to_owned();
    timeline["tracks"][0]["items"] = json!([
        {
            "id": "item-1",
            "trackId": "track_main",
            "assetId": "asset-1",
            "type": "video",
            "displayName": "Clip",
            "sourceIn": 4,
            "sourceOut": 2,
            "timelineStart": 0,
            "timelineEnd": 4
        }
    ]);
    let (status, _) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": timeline.clone() }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    timeline["tracks"][0]["items"][0]["sourceOut"] = json!(6);
    timeline["tracks"][0]["kind"] = json!("audio_v2");
    let (status, _) = request(
        app,
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": timeline }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn character_studio_routes_manage_references_loras_and_test_jobs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let config_dir = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z-Image",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "character_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["z-image"] },
                  "ui": {}
                }
              ]
            }
            "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Characters" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "reference.png",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (status, character) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters"),
        json!({ "name": "Mira", "type": "person" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(character["name"], "Mira");
    let character_id = character["id"].as_str().expect("character id").to_owned();

    let (status, with_reference) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/references"),
        json!({ "assetId": asset_id, "approved": false }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        with_reference["references"][0]["asset"]["displayName"],
        "reference.png"
    );

    let (status, updated) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/references/{asset_id}"),
        json!({ "approved": true, "role": "hero" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["approvedReferences"][0]["assetId"], asset_id);

    let sidecar_path = project_path.join(
        asset["sidecarPath"]
            .as_str()
            .expect("asset sidecar path")
            .replace('/', std::path::MAIN_SEPARATOR_STR),
    );
    let asset_sidecar: Value =
        serde_json::from_str(&std::fs::read_to_string(sidecar_path).expect("asset sidecar reads"))
            .expect("asset sidecar parses");
    assert_eq!(
        asset_sidecar["metadata"]["characterReferences"][0]["characterId"],
        character_id
    );
    assert_eq!(
        asset_sidecar["metadata"]["characterReferences"][0]["approved"],
        true
    );

    let (status, with_look) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/looks"),
            json!({ "name": "Rain coat", "approvedReferenceIds": [asset_id], "recipeSettings": { "style": "noir" } }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(with_look["looks"][0]["recipeSettings"]["style"], "noir");
    let look_id = with_look["looks"][0]["id"]
        .as_str()
        .expect("look id")
        .to_owned();

    let lora_dir = data_dir.join("loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    let lora_source = lora_dir.join("mira.safetensors");
    write_test_safetensors(&lora_source);
    let (status, with_lora) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
        json!({
            "name": "Mira LoRA",
            "sourcePath": lora_source.display().to_string(),
            "compatibility": { "families": ["z-image"] },
            "triggerWords": ["mira"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(with_lora["loras"][0]["copiedIntoProject"], true);
    let project_lora_path = project_path.join(
        with_lora["loras"][0]["projectPath"]
            .as_str()
            .expect("project lora path")
            .replace('/', std::path::MAIN_SEPARATOR_STR),
    );
    assert_eq!(
        std::fs::read(project_lora_path).expect("lora copied"),
        test_safetensors_bytes()
    );

    let (status, test_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/test-jobs"),
        json!({ "prompt": "portrait", "lookId": look_id, "count": 2 }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(test_job["type"], "image_generate");
    assert_eq!(test_job["payload"]["mode"], "character_image");
    assert_eq!(test_job["payload"]["characterId"], character_id);
    // Regression (sc-2074): the worker's image_request_from_job requires payload.projectId;
    // the test-job handler must inject it (it isn't carried by the column alone).
    assert_eq!(test_job["payload"]["projectId"], project_id);
    assert_eq!(
        test_job["payload"]["advanced"]["approvedReferenceIds"][0],
        asset_id
    );

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/archive"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, visible) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/characters"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(visible.as_array().unwrap().len(), 0);
    let (status, archived) = request(
        app,
        "GET",
        &format!("/api/v1/projects/{project_id}/characters?includeArchived=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(archived.as_array().unwrap().len(), 1);
}
