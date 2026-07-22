//! rust-api training tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[tokio::test]
async fn training_targets_route_returns_builtin_registry() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (status, registry) = request(app, "GET", "/api/v1/training/targets", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(registry["schemaVersion"], 1);
    assert_eq!(registry["targets"][0]["id"], "z_image_turbo_lora");
    assert_eq!(registry["targets"][0]["defaults"]["rank"], 16);
    assert_eq!(
        registry["targets"][0]["defaults"]["advanced"]["qualityPreset"],
        "balanced"
    );
}

#[tokio::test]
async fn training_presets_route_returns_builtin_registry() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (status, registry) = request(app, "GET", "/api/v1/training/presets", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(registry["schemaVersion"], 1);
    assert_eq!(
        registry["presets"][0]["id"],
        "z_image_turbo_lora.character.adamw8bit.balanced"
    );
    assert_eq!(registry["presets"][0]["config"]["steps"], 3000);
    assert_eq!(
        registry["presets"][0]["config"]["advanced"]["sampleSteps"],
        8
    );
    let prodigy = registry["presets"]
        .as_array()
        .expect("preset array")
        .iter()
        .find(|preset| preset["id"] == "z_image_turbo_lora.character.prodigyopt.balanced")
        .expect("prodigy preset");
    assert_eq!(prodigy["config"]["optimizer"], "prodigyopt");
    assert_eq!(prodigy["config"]["learningRate"], 1.0);
}

#[tokio::test]
async fn training_dataset_routes_persist_and_validate_project_assets() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Mira LoRA set",
            "items": [{
                "assetId": asset_id,
                "caption": {
                    "text": "miraStyle portrait",
                    "triggerWords": ["miraStyle"]
                }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(dataset["projectId"], project_id);
    assert_eq!(dataset["version"], 1);
    assert_eq!(dataset["items"][0]["assetId"], asset_id);
    assert_eq!(dataset["items"][0]["path"], "images/item_0001.png");
    assert_eq!(dataset["items"][0]["caption"]["source"], "manual");
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    assert!(project_path
        .join("training")
        .join("datasets")
        .join(&dataset_id)
        .join("images")
        .join("item_0001.png")
        .exists());

    let (status, listed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed[0]["id"], dataset_id);
    assert_eq!(listed[0]["itemCount"], 1);
    // The summary carries a cover thumbnail path (sc-2025) for the dataset selector.
    assert_eq!(
        listed[0]["coverPath"],
        format!("training/datasets/{dataset_id}/images/item_0001.png")
    );

    let reloaded_app = create_app(settings).expect("app reloads");
    let (status, detail) = request(
        reloaded_app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail["items"][0]["caption"]["text"], "miraStyle portrait");

    let (status, updated) = request(
        reloaded_app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        json!({
            "items": [{
                "assetId": asset_id,
                "caption": { "text": "miraStyle close portrait" }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["version"], 2);
    assert_eq!(
        updated["items"][0]["caption"]["text"],
        "miraStyle close portrait"
    );
    let dataset_image_path = project_path
        .join("training")
        .join("datasets")
        .join(&dataset_id)
        .join("images")
        .join("item_0001.png");
    assert_eq!(
        std::fs::read(&dataset_image_path).expect("dataset image remains"),
        b"png-bytes"
    );

    let (status, error) = request(
        reloaded_app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        json!({
            "items": [
                { "assetId": asset_id },
                { "assetId": "asset_missing" }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["detail"], "Asset not found");
    assert_eq!(
        std::fs::read(&dataset_image_path).expect("old dataset image survives failed update"),
        b"png-bytes"
    );
    let (status, detail_after_failed_update) = request(
        reloaded_app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail_after_failed_update["version"], 2);

    let (status, sidecars) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-sidecars"),
        json!({
            "items": [{
                "itemId": "item_0001",
                "caption": {
                    "text": "studio portrait",
                    "triggerWords": ["miraStyle"]
                }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(sidecars["dataset"]["version"], 3);
    assert_eq!(
        sidecars["sidecars"][0]["captionPath"],
        format!("training/datasets/{dataset_id}/images/item_0001.txt")
    );
    assert_eq!(
        std::fs::read_to_string(
            project_path
                .join("training")
                .join("datasets")
                .join(&dataset_id)
                .join("images")
                .join("item_0001.txt")
        )
        .expect("caption sidecar writes"),
        "miraStyle, studio portrait\n"
    );

    let (status, caption_job) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-jobs"),
        json!({
            "recaption": true,
            "requestedGpu": "auto",
            "options": {
                "captionType": "Straightforward",
                "captionLength": "40",
                "extraOptions": ["Include information about lighting."],
                "nameInput": "Mira",
                "temperature": 0.5,
                "topP": 0.8,
                "maxNewTokens": 128
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(caption_job["type"], "training_caption");
    assert_eq!(caption_job["payload"]["captioner"], "joy_caption");
    assert_eq!(
        caption_job["payload"]["modelNameOrPath"],
        "fancyfeast/llama-joycaption-beta-one-hf-llava"
    );
    assert_eq!(caption_job["payload"]["items"][0]["itemId"], "item_0001");
    assert_eq!(
        caption_job["payload"]["items"][0]["triggerWords"],
        json!(["miraStyle"])
    );
    let caption_image_path = caption_job["payload"]["items"][0]["imagePath"]
        .as_str()
        .expect("caption image path");
    assert!(caption_image_path.contains(&dataset_id));
    assert!(caption_image_path.ends_with("item_0001.png"));

    // sc-2025: itemIds targets a single image and recaptions it even though it
    // already has a caption (recaption:false would otherwise skip it).
    let (status, single_item_job) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-jobs"),
        json!({ "recaption": false, "itemIds": ["item_0001"] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let single_items = single_item_job["payload"]["items"]
        .as_array()
        .expect("caption items");
    assert_eq!(single_items.len(), 1);
    assert_eq!(single_items[0]["itemId"], "item_0001");

    // sc-6535: the Dataset Doctor analysis job enqueues with the right type + embedder + a per-item
    // work list (the worker claims it once mlx-gen-clip is linked; here we assert the enqueue
    // contract). An empty body uses the defaults (clip_vit_l14, every item).
    let (status, analysis_job) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-jobs"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(analysis_job["type"], "dataset_analysis");
    assert_eq!(analysis_job["payload"]["embedder"], "clip_vit_l14");
    assert_eq!(analysis_job["payload"]["items"][0]["itemId"], "item_0001");
    assert_eq!(
        analysis_job["payload"]["items"][0]["captionText"],
        "studio portrait"
    );
    assert_eq!(
        analysis_job["payload"]["items"][0]["captionHash"],
        sceneworks_core::dataset_quality::caption_hash("studio portrait")
    );
    let analysis_image_path = analysis_job["payload"]["items"][0]["imagePath"]
        .as_str()
        .expect("analysis image path");
    assert!(analysis_image_path.ends_with("item_0001.png"));

    // sc-6538: the Dataset Doctor face pass enqueues with its own type + a per-item work list (item id
    // + image path + content hash, no caption — the face stack ignores captions). The worker claims it
    // once the SCRFD+ArcFace stack is advertised (slice 4b); here we assert the enqueue contract.
    let (status, face_job) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/face-analysis-jobs"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(face_job["type"], "dataset_face_analysis");
    assert_eq!(face_job["payload"]["items"][0]["itemId"], "item_0001");
    assert!(face_job["payload"]["items"][0]["imagePath"]
        .as_str()
        .expect("face image path")
        .ends_with("item_0001.png"));
    // The face pass carries no caption fields (unlike the CLIP analysis pass).
    assert!(face_job["payload"]["items"][0]["captionText"].is_null());

    // An unknown embedder is rejected up front.
    let (status, _) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-jobs"),
        json!({ "embedder": "not_a_real_embedder" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // sc-6535: the analysis worker persists its computed embeddings to the content-hash-keyed sidecar.
    let (status, stored) = request(
        reloaded_app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-embeddings"
        ),
        json!({
            "space": "clip-vit-l14",
            "items": [
                { "contentHash": "hash_a", "embedding": [1.0, 0.0, 0.0] },
                { "contentHash": "hash_b", "embedding": [0.0, 1.0, 0.0] }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stored["stored"], 2);

    let (status, renamed) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-rename"),
        json!({
            "items": [{
                "itemId": "item_0001",
                "newItemId": "item_0007",
                "fileStem": "mira_0007",
                "displayName": "mira_0007.png"
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(renamed["version"], 4);
    assert_eq!(renamed["items"][0]["id"], "item_0007");
    assert_eq!(renamed["items"][0]["path"], "images/mira_0007.png");
    assert_eq!(renamed["items"][0]["displayName"], "mira_0007.png");
    let renamed_image_path = project_path
        .join("training")
        .join("datasets")
        .join(&dataset_id)
        .join("images")
        .join("mira_0007.png");
    assert_eq!(
        std::fs::read(&renamed_image_path).expect("renamed dataset image remains"),
        b"png-bytes"
    );
    assert!(!dataset_image_path.exists());
    assert_eq!(
        std::fs::read_to_string(renamed_image_path.with_extension("txt"))
            .expect("caption sidecar follows rename"),
        "miraStyle, studio portrait\n"
    );

    let (status, error) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Bad Path",
            "items": [{ "path": "../outside.png" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "Invalid dataset item path");

    let (status, error) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Duplicate Items",
            "items": [
                { "id": "same_item", "assetId": asset_id },
                { "id": "same_item", "assetId": asset_id }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "Training dataset item IDs must be unique");

    let (_, other_project) = request(
        reloaded_app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Other Training Project" }),
    )
    .await;
    let other_project_id = other_project["id"].as_str().expect("project id").to_owned();
    let (status, other_asset) = request_multipart_upload(
        reloaded_app.clone(),
        &format!("/api/v1/projects/{other_project_id}/assets"),
        "Other.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let other_asset_id = other_asset["id"].as_str().expect("asset id").to_owned();
    let (status, error) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Cross Project",
            "items": [{ "assetId": other_asset_id }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["detail"], "Asset not found");

    let (status, deleted) = request(
        reloaded_app.clone(),
        "DELETE",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted, json!({ "id": dataset_id, "status": "deleted" }));
    let (status, listed_after_delete) = request(
        reloaded_app,
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed_after_delete, json!([]));
}

// sc-2022: a dataset can be associated with a character at create time or via a
// later PATCH, and the association surfaces on the detail body and list summary
// so the Character Studio can scope its dataset list client-side. General
// datasets leave `characterId` null.
#[tokio::test]
async fn training_datasets_associate_with_a_character() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Character Datasets" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    // Created with an explicit character association (the "create from a
    // character's images" path).
    let (status, scoped) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Mira identity set",
            "characterId": "character_mira",
            "items": [{ "assetId": asset_id }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(scoped["characterId"], "character_mira");
    let scoped_id = scoped["id"].as_str().expect("dataset id").to_owned();

    // A general dataset leaves the association null.
    let (status, general) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Style set", "items": [{ "assetId": asset_id }] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(general["characterId"], Value::Null);
    let general_id = general["id"].as_str().expect("dataset id").to_owned();

    // The association round-trips through a reload and the list summary.
    let reloaded_app = create_app(settings).expect("app reloads");
    let (status, listed) = request(
        reloaded_app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let by_id = |id: &str| -> Value {
        listed
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == id)
            .cloned()
            .expect("dataset present in list")
    };
    assert_eq!(by_id(&scoped_id)["characterId"], "character_mira");
    assert_eq!(by_id(&general_id)["characterId"], Value::Null);

    // PATCHing a general dataset associates it (the import-from-character path,
    // which always saves a full item set alongside the new association).
    let (status, associated) = request(
        reloaded_app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/training/datasets/{general_id}"),
        json!({ "characterId": "character_kelsie", "items": [{ "assetId": asset_id }] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(associated["characterId"], "character_kelsie");

    let (status, relisted) = request(
        reloaded_app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let associated_summary = relisted
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == general_id)
        .cloned()
        .expect("associated dataset present");
    assert_eq!(associated_summary["characterId"], "character_kelsie");
}

#[tokio::test]
async fn training_dataset_uploads_are_dataset_owned_not_assets() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Dataset Upload Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, upload) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/training/uploads"),
        "DatasetOnly.PNG",
        "image/png",
        b"dataset-only-png",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(upload["datasetOnly"], true);
    let staged_path = upload["file"]["path"]
        .as_str()
        .expect("staged path")
        .to_owned();
    assert!(staged_path.starts_with("training/uploads/"));

    let (status, listed_assets) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed_assets.as_array().expect("asset list").len(), 0);

    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Dataset-owned import",
            "items": [{
                "path": staged_path,
                "displayName": "DatasetOnly.PNG",
                "caption": { "text": "dataset only portrait" }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(dataset["items"][0]["assetId"].is_null());
    assert_eq!(dataset["items"][0]["path"], "images/item_0001.png");
    let dataset_id = dataset["id"].as_str().expect("dataset id");
    assert_eq!(
        std::fs::read(
            project_path
                .join("training")
                .join("datasets")
                .join(dataset_id)
                .join("images")
                .join("item_0001.png")
        )
        .expect("dataset-owned image copied"),
        b"dataset-only-png"
    );

    let (status, listed_assets_after_dataset) = request(
        app,
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        listed_assets_after_dataset
            .as_array()
            .expect("asset list after dataset")
            .len(),
        0
    );
}

#[tokio::test]
async fn training_dataset_readiness_reports_and_persists_tier0_cache() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Dataset Readiness Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (status, upload) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/training/uploads"),
        "Tiny.PNG",
        "image/png",
        PNG_32X32,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(upload["file"]["width"], 32);
    assert_eq!(upload["file"]["height"], 32);
    let upload_hash = upload["file"]["contentHash"]
        .as_str()
        .expect("content hash")
        .to_owned();
    assert_eq!(upload_hash.len(), 64);
    let staged_path = upload["file"]["path"]
        .as_str()
        .expect("staged path")
        .to_owned();

    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Tiny readiness set",
            "items": [{
                "path": staged_path,
                "displayName": "Tiny.PNG",
                "caption": { "text": "tiny test image" }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(dataset["items"][0]["width"], 32);
    assert_eq!(dataset["items"][0]["height"], 32);
    assert_eq!(dataset["items"][0]["contentHash"], upload_hash);
    assert!(dataset["items"][0]["tier0Scalars"].is_null());
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let item_id = dataset["items"][0]["id"]
        .as_str()
        .expect("item id")
        .to_owned();

    let (status, report) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=style&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(report["gate"], "needs_attention");
    assert_eq!(report["itemCount"], 1);
    assert_eq!(report["datasetFlags"], json!([]));
    let flags = report["items"][0]["flags"].as_array().expect("flags array");
    assert!(flags.iter().any(|flag| flag["check"] == "resolution"));
    assert!(!flags.iter().any(|flag| flag["check"] == "decode"));
    // sc-6535: before the analysis job persists an embedding sidecar there is no Tier-1 sub-score.
    assert!(report["subScores"]["diversity"].is_null());
    // sc-6537: nor an aesthetic sub-score (style dataset, but no embeddings yet).
    assert!(report["subScores"]["aesthetic"].is_null());
    // sc-6540: the resolved kind is echoed so the client can branch its recommendations.
    assert_eq!(report["kind"], "style");

    let (status, detail) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail["items"][0]["id"], item_id);
    let cache = &detail["items"][0]["tier0Scalars"];
    assert_eq!(cache["contentHash"], upload_hash);
    assert_eq!(cache["bucketEdge"], 64);
    assert!(cache["scalars"]["blurVariance"].is_number());
    assert!(!cache["scalars"]["phash"]
        .as_array()
        .expect("phash array")
        .is_empty());

    // sc-6535 read-back: once the analysis worker POSTs an embedding sidecar (keyed by content hash),
    // the readiness report folds the Tier-1 diversity sub-score in. This is the seam that lights up
    // the Variety meter + embedding findings in the UI. A full 768-d embedding also lights up the
    // sc-6537 aesthetic sub-score (the LAION head is 768-in); a sparse unit vector suffices for the
    // "is a number" assertions.
    let embedding: Vec<f64> = {
        let mut values = vec![0.0; 768];
        values[0] = 0.6;
        values[1] = 0.8;
        values
    };
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-embeddings"
        ),
        json!({
            "space": "clip-vit-l14",
            "items": [{ "contentHash": upload_hash.clone(), "embedding": embedding }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, report) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=style&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        report["subScores"]["diversity"].is_number(),
        "the embedding sidecar lights up the Tier-1 diversity sub-score"
    );
    // sc-6537: the same style sidecar lights up the aesthetic sub-score (LAION head over the persisted
    // 768-d CLIP embedding); a person/object dataset would leave it null.
    assert!(
        report["subScores"]["aesthetic"].is_number(),
        "a style dataset + embedding sidecar lights up the aesthetic sub-score"
    );
    assert!(
        report["subScores"]["alignment"].is_null(),
        "image-only sidecars do not pretend caption alignment has run"
    );

    let (status, error) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-embeddings"
        ),
        json!({
            "space": "clip-vit-l14",
            "items": [{
                "contentHash": upload_hash.clone(),
                "embedding": embedding.clone(),
                "captionHash": sceneworks_core::dataset_quality::caption_hash("tiny test image"),
                "textEmbedding": [1.0, 0.0]
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        error["detail"],
        "textEmbedding length must match embedding length."
    );

    let stale_caption_hash = sceneworks_core::dataset_quality::caption_hash("old caption");
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-embeddings"
        ),
        json!({
            "space": "clip-vit-l14",
            "items": [{
                "contentHash": upload_hash.clone(),
                "embedding": embedding.clone(),
                "captionHash": stale_caption_hash,
                "textEmbedding": embedding.clone()
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, report) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=style&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        report["subScores"]["alignment"].is_null(),
        "a text embedding keyed to an old caption must not be reused after re-captioning"
    );

    let current_caption_hash = sceneworks_core::dataset_quality::caption_hash("tiny test image");
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-embeddings"
        ),
        json!({
            "space": "clip-vit-l14",
            "items": [{
                "contentHash": upload_hash.clone(),
                "embedding": embedding.clone(),
                "captionHash": current_caption_hash,
                "textEmbedding": embedding
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, report) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=style&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        report["subScores"]["alignment"].is_number(),
        "current caption text + text embedding lights up caption alignment"
    );

    let (status, error) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/items/{item_id}/quality-ack"
        ),
        json!({
            "checks": ["caption_alignment"],
            "expectedContentHash": upload_hash.clone(),
            "expectedCaptionHash": sceneworks_core::dataset_quality::caption_hash("old caption")
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        error["detail"],
        "Caption acknowledgement is stale; refresh dataset readiness."
    );

    let (status, ack) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/items/{item_id}/quality-ack"
        ),
        json!({
            "checks": ["caption_alignment"],
            "expectedContentHash": upload_hash,
            "expectedCaptionHash": current_caption_hash
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ack["captionHash"], current_caption_hash);
    assert_eq!(ack["checks"], json!(["caption_alignment"]));
}

/// sc-6538 face stack (slice 2): the worker face pass POSTs a face sidecar; a PERSON readiness report
/// then folds in subject-prominence / identity findings, while the same sidecar is ignored for a
/// non-person dataset. Single-item set so the per-image `SmallSubject` check (which needs no
/// clustering) carries the proof end-to-end through the new `face-embeddings` route + the fold.
#[tokio::test]
async fn training_dataset_face_sidecar_folds_into_person_readiness_only() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Face Stack Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (status, upload) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/training/uploads"),
        "Face.PNG",
        "image/png",
        PNG_32X32,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let upload_hash = upload["file"]["contentHash"]
        .as_str()
        .expect("content hash")
        .to_owned();
    let staged_path = upload["file"]["path"]
        .as_str()
        .expect("staged path")
        .to_owned();

    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Face set",
            "items": [{
                "path": staged_path,
                "displayName": "Face.PNG",
                "caption": { "text": "a photo of the subject" }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

    // The worker face pass POSTs the largest-face record. A detected face (non-empty embedding) whose
    // bbox covers only 0.5% of the frame is below the 2% floor → a SmallSubject finding.
    let (status, stored) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/face-embeddings"),
        json!({
            "space": "arcface-r100",
            "items": [{
                "contentHash": upload_hash.clone(),
                "embedding": [1.0, 0.0],
                "faceFraction": 0.005
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stored["stored"], 1);

    // PERSON readiness folds the face sidecar in: the SmallSubject flag surfaces on the item. With a
    // single face (below the clustering minimum) there is no identity *score* yet — only the per-image
    // prominence check fires.
    let (status, person) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=person&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(person["kind"], "person");
    let flags = person["items"][0]["flags"].as_array().expect("flags array");
    assert!(
        flags.iter().any(|flag| flag["check"] == "small_subject"),
        "the person readiness folds the face sidecar's SmallSubject finding in"
    );
    assert!(
        person["subScores"]["identity"].is_null(),
        "one detected face is below the clustering minimum — no identity score"
    );

    // The SAME sidecar is ignored for a non-person dataset (the fold is Person-gated): no face flags,
    // no identity sub-score.
    let (status, style) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=style&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(style["kind"], "style");
    let style_flags = style["items"][0]["flags"].as_array().expect("flags array");
    assert!(
        !style_flags
            .iter()
            .any(|flag| flag["check"] == "small_subject"),
        "the face fold is Person-gated — a style dataset ignores the face sidecar"
    );
    assert!(style["subScores"]["identity"].is_null());
}

/// sc-6538 face stack (slice 2): with enough detected faces the fold runs identity *clustering* and
/// lights up the `identity` sub-score. Three items over the same content hash all resolve to one face
/// record → a single coherent identity cluster → score 1.0 (a number, not null). This exercises the
/// embedding through clustering, which the single-item SmallSubject test deliberately skips.
#[tokio::test]
async fn training_dataset_face_sidecar_lights_up_identity_score_when_clustered() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Identity Score Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (status, upload) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/training/uploads"),
        "Subject.PNG",
        "image/png",
        PNG_32X32,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let upload_hash = upload["file"]["contentHash"]
        .as_str()
        .expect("content hash")
        .to_owned();
    let staged_path = upload["file"]["path"]
        .as_str()
        .expect("staged path")
        .to_owned();

    // Three items over the same staged image → three items sharing one content hash. (The clustering
    // minimum is 3 detected faces.)
    let item = json!({
        "path": staged_path,
        "displayName": "Subject.PNG",
        "caption": { "text": "a photo of the subject" }
    });
    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Identity set",
            "items": [item.clone(), item.clone(), item]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let hashes: Vec<&str> = dataset["items"]
        .as_array()
        .expect("items")
        .iter()
        .map(|i| i["contentHash"].as_str().expect("content hash"))
        .collect();
    assert_eq!(
        hashes,
        vec![upload_hash.as_str(); 3],
        "items share one hash"
    );

    // One face record (face well above the prominence floor). All three items resolve to it.
    let (status, stored) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/face-embeddings"),
        json!({
            "space": "arcface-r100",
            "items": [{
                "contentHash": upload_hash.clone(),
                "embedding": [1.0, 0.0],
                "faceFraction": 0.25
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stored["stored"], 1);

    let (status, person) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=person&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(person["subScores"]["identity"], json!(1.0));
    // A coherent single identity → no mismatch flags on any item.
    for item in person["items"].as_array().expect("items") {
        let flags = item["flags"].as_array().expect("flags array");
        assert!(!flags
            .iter()
            .any(|flag| flag["check"] == "identity_mismatch"));
        assert!(!flags.iter().any(|flag| flag["check"] == "small_subject"));
    }
}

#[tokio::test]
async fn training_dataset_caption_alignment_requires_current_text_embedding_coverage() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Alignment Coverage Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (status, upload) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/training/uploads"),
        "Tiny.PNG",
        "image/png",
        PNG_32X32,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let upload_hash = upload["file"]["contentHash"]
        .as_str()
        .expect("content hash")
        .to_owned();
    let staged_path = upload["file"]["path"]
        .as_str()
        .expect("staged path")
        .to_owned();

    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Caption coverage set",
            "items": [
                {
                    "path": staged_path,
                    "displayName": "Tiny A.PNG",
                    "caption": { "text": "first caption" }
                },
                {
                    "path": staged_path,
                    "displayName": "Tiny B.PNG",
                    "caption": { "text": "second caption" }
                }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    assert_eq!(dataset["items"].as_array().expect("items").len(), 2);

    let embedding: Vec<f64> = {
        let mut values = vec![0.0; 768];
        values[0] = 1.0;
        values
    };
    let first_caption_hash = sceneworks_core::dataset_quality::caption_hash("first caption");
    let second_caption_hash = sceneworks_core::dataset_quality::caption_hash("second caption");

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-embeddings"
        ),
        json!({
            "space": "clip-vit-l14",
            "items": [{
                "contentHash": upload_hash.clone(),
                "embedding": embedding.clone(),
                "captionHash": first_caption_hash,
                "textEmbedding": embedding.clone()
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, report) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=style&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        report["subScores"]["alignment"].is_null(),
        "one fresh caption embedding must not look like whole-dataset caption coverage"
    );

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-embeddings"
        ),
        json!({
            "space": "clip-vit-l14",
            "items": [{
                "contentHash": upload_hash.clone(),
                "embedding": embedding.clone(),
                "captionHash": second_caption_hash,
                "textEmbedding": embedding
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, report) = request(
        app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness?targetResolution=64&recommendedFor=style&minItems=1"
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        report["subScores"]["alignment"].is_number(),
        "same-space sidecar merges text embeddings until every current caption is covered"
    );
}

#[tokio::test]
async fn asset_library_scope_excludes_character_studio_outputs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Library Scope Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    // Write a normal Image Studio output and a Character Studio test output
    // directly as sidecars (no explicit origin → derived on the first reindex,
    // which the initial /assets call triggers because the table is empty).
    let image_dir = project_path.join("assets/images");
    for (id, mode) in [
        ("img_studio_1", "text_to_image"),
        ("char_test_1", "character_image"),
    ] {
        std::fs::write(image_dir.join(format!("{id}.png")), b"png-bytes").expect("media");
        std::fs::write(
            image_dir.join(format!("{id}.sceneworks.json")),
            serde_json::to_string_pretty(&json!({
                "id": id,
                "type": "image",
                "displayName": id,
                "createdAt": "2026-05-23T00:00:00Z",
                "file": {"path": format!("assets/images/{id}.png")},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false},
                "recipe": {"mode": mode},
            }))
            .expect("json"),
        )
        .expect("sidecar");
    }

    // Default (all) scope returns both, each tagged with a derived origin.
    let (status, all) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let all = all.as_array().expect("all list");
    assert_eq!(all.len(), 2);
    assert!(all
        .iter()
        .any(|asset| asset["id"] == "char_test_1" && asset["origin"] == "character_studio"));

    // Library scope drops the Character Studio output.
    let (status, library) = request(
        app,
        "GET",
        &format!("/api/v1/projects/{project_id}/assets?scope=library"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let library = library.as_array().expect("library list");
    assert_eq!(library.len(), 1);
    assert_eq!(library[0]["id"], "img_studio_1");
    assert_eq!(library[0]["origin"], "image_studio");
}

#[tokio::test]
async fn create_training_job_resolves_plan_and_queues_lora_train() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target = registry["targets"][0].clone();
    let target_id = target["id"].as_str().expect("target id").to_owned();
    let config = target["defaults"].clone();

    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_train");
    assert_eq!(job["status"], "queued");
    assert_eq!(job["requestedGpu"], "auto");
    assert_eq!(job["projectId"], project_id);
    assert_eq!(job["payload"]["dryRun"], true);

    let job_id = job["id"].as_str().expect("job id").to_owned();
    let plan = &job["payload"]["plan"];
    // The resolved plan is self-referential and fully normalized in Rust.
    assert_eq!(plan["jobId"], job_id);
    assert_eq!(plan["provenance"]["sourceJobId"], job_id);
    assert_eq!(plan["target"]["targetId"], target_id);
    assert_eq!(plan["dataset"]["datasetId"], dataset_id);
    assert_eq!(plan["dataset"]["datasetVersion"], 1);
    assert_eq!(plan["dataset"]["items"].as_array().unwrap().len(), 1);
    let lora_id = plan["output"]["loraId"].as_str().expect("lora id");
    assert!(lora_id.starts_with("lora_"));
    assert_eq!(plan["output"]["fileName"], "aurora_style.safetensors");

    // Item image paths resolve under the dataset root on disk.
    let expected_image = project_path
        .join("training")
        .join("datasets")
        .join(&dataset_id)
        .join("images")
        .join("item_0001.png");
    assert_eq!(
        plan["dataset"]["items"][0]["imagePath"]
            .as_str()
            .expect("image path"),
        expected_image.display().to_string()
    );
    // The default target scope is `project`, so the adapter is written into
    // the project's LoRA store (not the shared data dir).
    assert_eq!(
        plan["output"]["outputDir"].as_str().expect("output dir"),
        project_path
            .join("loras")
            .join(lora_id)
            .display()
            .to_string()
    );
    // The submit-time manifest entry carries provenance for the LoRA that
    // registration will recompute and upsert on completion. The manifest path
    // itself is intentionally NOT persisted in the payload — it is recomputed
    // from trusted inputs at completion so a tampered payload cannot redirect
    // the write.
    assert_eq!(job["payload"]["manifestEntry"]["scope"], "project");
    assert_eq!(job["payload"]["manifestEntry"]["family"], "z-image");
    assert_eq!(
        job["payload"]["manifestEntry"]["source"]["path"],
        format!("loras/{lora_id}")
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["provenance"]["datasetId"],
        dataset_id
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["provenance"]["trainingJobId"],
        job_id
    );
    assert!(job["payload"]["manifestPath"].is_null());

    // The job is queued and visible to the queue/worker surface.
    let (status, queued) = request(
        app.clone(),
        "GET",
        "/api/v1/jobs?status=queued",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queued[0]["id"], job_id);
    assert_eq!(queued[0]["type"], "lora_train");

    let (_, preset_registry) =
        request(app.clone(), "GET", "/api/v1/training/presets", Value::Null).await;
    let prodigy_preset = preset_registry["presets"]
        .as_array()
        .expect("preset array")
        .iter()
        .find(|preset| preset["id"] == "z_image_turbo_lora.character.prodigyopt.balanced")
        .expect("prodigy preset")
        .clone();
    let (status, preset_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "presetId": prodigy_preset["id"],
            "presetVersion": prodigy_preset["version"],
            "config": prodigy_preset["config"],
            "outputName": "Aurora Prodigy",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        preset_job["payload"]["plan"]["provenance"]["presetId"],
        "z_image_turbo_lora.character.prodigyopt.balanced"
    );
    assert_eq!(
        preset_job["payload"]["plan"]["provenance"]["presetName"],
        "Prodigy character (experimental)"
    );
    assert_eq!(
        preset_job["payload"]["plan"]["provenance"]["presetConfigSnapshot"]["learningRate"],
        1.0
    );
    assert_eq!(
        preset_job["payload"]["manifestEntry"]["provenance"]["presetId"],
        "z_image_turbo_lora.character.prodigyopt.balanced"
    );

    let (status, error) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "presetId": "z_image_turbo_lora.character.prodigyopt.balanced",
            "presetVersion": 99,
            "config": prodigy_preset["config"],
            "outputName": "Aurora Prodigy",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
            error["detail"],
            "Training preset 'z_image_turbo_lora.character.prodigyopt.balanced' is version 1, but the request pinned version 99."
        );
}

#[tokio::test]
async fn create_training_job_rejects_unknown_target_and_missing_dataset() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let config = registry["targets"][0]["defaults"].clone();

    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": "not_a_target",
            "datasetId": "ds_missing",
            "config": config,
            "outputName": "Aurora"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .unwrap()
        .contains("Unknown training target"));

    let target_id = registry["targets"][0]["id"].as_str().unwrap().to_owned();
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": "ds_missing",
            "config": registry["targets"][0]["defaults"].clone(),
            "outputName": "Aurora"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["detail"], "Training dataset not found");
}

#[tokio::test]
async fn create_training_job_queues_real_run_when_not_dry_run() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    // A real run requires the base model installed (story 1419 guardrail).
    seed_installed_base_model(&settings.data_dir);
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target = registry["targets"][0].clone();
    let target_id = target["id"].as_str().expect("target id").to_owned();
    let config = target["defaults"].clone();

    // Real execution exists (story 1417): a non-dry-run job resolves the same
    // plan and queues for the worker's Z-Image LoRA kernel.
    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_train");
    assert_eq!(job["status"], "queued");
    assert_eq!(job["payload"]["dryRun"], false);
    // The plan is resolved and embedded just like the dry-run path.
    assert_eq!(job["payload"]["plan"]["planVersion"], 1);
    assert_eq!(job["payload"]["plan"]["target"]["kernel"], "z_image_lora");

    let job_id = job["id"].as_str().expect("job id").to_owned();
    let (status, queued) = request(
        app.clone(),
        "GET",
        "/api/v1/jobs?status=queued",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queued[0]["id"], job_id);
    assert_eq!(queued[0]["type"], "lora_train");
}

#[tokio::test]
async fn completed_training_job_registers_lora_with_provenance() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (job_id, output_dir, adapter_path) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;

    // The worker writes the final adapter into the resolved output dir before
    // it reports completion, alongside step checkpoints it does not clean up.
    // Registration must pick the declared final adapter, not a checkpoint.
    std::fs::create_dir_all(&output_dir).expect("output dir creates");
    write_test_safetensors(&adapter_path);
    let final_name = adapter_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("final adapter name")
        .to_owned();
    let stem = adapter_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("adapter stem");
    write_test_safetensors(&output_dir.join(format!("{stem}-step000250.safetensors")));

    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Trained LoRA saved.",
            "result": { "outputPath": adapter_path.display().to_string() }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["status"], "completed");
    // The registration outcome is folded into the job result so it is
    // observable (rather than silently dropped on failure).
    assert_eq!(completed["result"]["loraRegistered"], true);
    assert!(completed["result"]["loraId"]
        .as_str()
        .is_some_and(|id| id.starts_with("lora_")));
    assert!(completed["result"]["loraManifestPath"]
        .as_str()
        .is_some_and(|path| path.ends_with("manifest.jsonc")));

    // The trained adapter is now a normal, installed, project-scoped LoRA.
    let (status, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let entry = loras
        .as_array()
        .expect("loras array")
        .iter()
        .find(|item| item["name"] == json!("Aurora Style"))
        .expect("trained LoRA appears in catalog")
        .clone();
    assert_eq!(entry["scope"], "project");
    assert_eq!(entry["family"], "z-image");
    assert_eq!(entry["baseModel"], "z_image_turbo");
    assert_eq!(entry["triggerWords"], json!(["auroraStyle"]));
    assert_eq!(entry["installState"], "installed");
    // The final adapter is registered, not the step checkpoint that shares
    // the output directory.
    assert_eq!(entry["files"], json!([final_name]));
    // installedPath resolves to the trained adapter's directory (the same
    // convention as imported LoRAs), and the adapter file lives under it.
    let lora_id = entry["id"].as_str().expect("lora id");
    let installed_path = entry["installedPath"].as_str().expect("installed path");
    assert!(
        installed_path.contains(lora_id),
        "installed path {installed_path} should point at the trained adapter dir"
    );
    assert!(adapter_path.exists());
    assert_eq!(entry["source"]["provider"], "training");
    assert_eq!(entry["provenance"]["trainingJobId"], job_id);
    assert!(entry["provenance"]["configSnapshot"].is_object());

    // Provenance survives an app restart (manifest is on disk).
    let reloaded = create_app(settings).expect("app reloads");
    let (status, loras) = request(
        reloaded,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(loras
        .as_array()
        .expect("loras array")
        .iter()
        .any(|item| item["name"] == json!("Aurora Style")
            && item["provenance"]["trainingJobId"] == json!(job_id)));
}

#[tokio::test]
async fn failed_or_unwritten_training_job_registers_no_lora() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // A failed job never registers, even though a manifest entry was staged.
    let (failed_job_id, _output_dir, _adapter_path) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{failed_job_id}/progress"),
        json!({
            "status": "failed",
            "stage": "failed",
            "progress": 1,
            "message": "Training failed.",
            "error": "CUDA out of memory"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // A job that reports completed but produced no weights must not leave a
    // broken registry entry either, and the failure is surfaced in the result.
    let (completed_no_weights_id, _, _) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{completed_no_weights_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Reported complete without weights."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["result"]["loraRegistered"], false);
    assert!(completed["result"]["loraRegistrationError"].is_string());

    let (status, loras) = request(
        app,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(loras
        .as_array()
        .expect("loras array")
        .iter()
        .all(|item| item["name"] != json!("Aurora Style")));
}

/// sc-11213 (F-028): a `completed` report that lost the race with cancel/sweep/
/// reclaim (the job is already terminal) must receive the 409 AND must NOT
/// register a ghost LoRA — the completion side-effects have to run strictly after
/// ownership + terminal status are confirmed, not before.
#[tokio::test]
async fn terminal_training_job_completed_report_registers_no_lora_and_409s() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (job_id, output_dir, adapter_path) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    // Stage the final adapter so registration would succeed if the gate were absent.
    stage_trained_adapter(&output_dir, &adapter_path);

    // The user cancels the job before the (late/racing) worker report arrives:
    // the queued job goes straight to the terminal `canceled` status.
    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/cancel"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["status"], "canceled");

    // The losing-race `completed` report must be rejected with 409 (terminal job
    // is immutable) and must NOT have registered the adapter as a LoRA.
    let (status, _rejected) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Trained LoRA saved.",
            "result": { "outputPath": adapter_path.display().to_string() }
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "a completed report for an already-canceled job must 409"
    );

    // No ghost catalog entry: the canceled training run left nothing registered.
    let (status, loras) = request(
        app,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        loras
            .as_array()
            .expect("loras array")
            .iter()
            .all(|item| item["name"] != json!("Aurora Style")),
        "a canceled training job must not register a ghost LoRA"
    );
}

/// sc-11213 (F-028): a `completed` report from a caller that does not own the job
/// (here, any report carrying a `workerId` for a job whose owner is unset) must
/// receive the 409 AND must NOT register a LoRA — the ownership check has to gate
/// the completion side-effects, not fire too late.
#[tokio::test]
async fn non_owner_completed_report_registers_no_lora_and_409s() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (job_id, output_dir, adapter_path) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    // Stage the final adapter so registration would succeed if the gate were absent.
    stage_trained_adapter(&output_dir, &adapter_path);

    // The job was never claimed, so its owner is unset. A report carrying a
    // `workerId` is therefore from a non-owner and must be rejected as such —
    // exactly the "any authenticated caller can trigger the writes" hole.
    let (status, _rejected) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Trained LoRA saved.",
            "workerId": "impostor-worker",
            "result": { "outputPath": adapter_path.display().to_string() }
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "a completed report from a non-owner must 409"
    );

    // No catalog entry: a non-owned report must not register a LoRA.
    let (status, loras) = request(
        app,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        loras
            .as_array()
            .expect("loras array")
            .iter()
            .all(|item| item["name"] != json!("Aurora Style")),
        "a non-owner report must not register a LoRA"
    );
}

/// sc-11213 (F-028): the guard must preserve the happy path — a legitimate
/// winning `completed` report (owner reporting a non-terminal job) still
/// registers the trained LoRA into the catalog exactly as before.
#[tokio::test]
async fn winning_completed_report_still_registers_lora() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (job_id, output_dir, adapter_path) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    stage_trained_adapter(&output_dir, &adapter_path);

    // A trusted (owner-equivalent, worker_id unset both sides) report against the
    // still-non-terminal job wins the race and registers normally.
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Trained LoRA saved.",
            "result": { "outputPath": adapter_path.display().to_string() }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["result"]["loraRegistered"], true);

    let (status, loras) = request(
        app,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        loras
            .as_array()
            .expect("loras array")
            .iter()
            .any(|item| item["name"] == json!("Aurora Style")),
        "a legitimate winning completed report must still register the LoRA"
    );
}

#[tokio::test]
async fn crafted_training_job_cannot_register_outside_canonical_manifest() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    // A `lora_train` job can be crafted directly via the generic job endpoint
    // with an attacker-chosen payload. Stage a real adapter under the canonical
    // project output dir so registration would succeed if (and only if) it uses
    // the recomputed path.
    let crafted_lora_id = "lora_crafted01";
    let adapter_dir = project_path.join("loras").join(crafted_lora_id);
    std::fs::create_dir_all(&adapter_dir).expect("adapter dir creates");
    write_test_safetensors(&adapter_dir.join("crafted.safetensors"));

    // The payload points the manifest write and the source path at locations
    // outside the canonical project manifest. Both must be ignored.
    let evil_manifest = temp_dir.path().join("evil-manifest.jsonc");
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "lora_train",
            "projectId": project_id,
            "projectName": "Training Project",
            "requestedGpu": "auto",
            "payload": {
                "dryRun": false,
                "manifestPath": evil_manifest.display().to_string(),
                "manifestEntry": {
                    "id": crafted_lora_id,
                    "name": "Crafted LoRA",
                    "scope": "project",
                    "family": "z-image",
                    "source": { "provider": "evil", "path": "../../../../escape/loras" },
                    "files": ["crafted.safetensors"]
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = job["id"].as_str().expect("job id").to_owned();

    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Crafted completion."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // The write went to the canonical project manifest, not the payload path.
    assert!(
        !evil_manifest.exists(),
        "payload manifestPath must be ignored"
    );
    assert_eq!(completed["result"]["loraRegistered"], true);
    assert_eq!(
        completed["result"]["loraManifestPath"]
            .as_str()
            .expect("manifest path"),
        project_path
            .join("loras")
            .join("manifest.jsonc")
            .display()
            .to_string()
    );

    // The registered entry's source path was recomputed, not taken from the
    // attacker payload.
    let (status, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let entry = loras
        .as_array()
        .expect("loras array")
        .iter()
        .find(|item| item["id"] == json!(crafted_lora_id))
        .expect("crafted LoRA registered under canonical manifest")
        .clone();
    assert_eq!(entry["scope"], "project");
    assert_eq!(entry["source"]["provider"], "training");
    assert_eq!(entry["source"]["path"], format!("loras/{crafted_lora_id}"));
    // `files` was validated against the recomputed output dir (the declared
    // name exists there), so the registered entry points only inside the
    // canonical LoRA directory.
    assert_eq!(entry["files"], json!(["crafted.safetensors"]));

    // A traversal id is rejected outright: nothing registers and the failure
    // is visible.
    let (_, evil_job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "lora_train",
            "projectId": project_id,
            "projectName": "Training Project",
            "requestedGpu": "auto",
            "payload": {
                "dryRun": false,
                "manifestEntry": {
                    "id": "../../pwned",
                    "name": "Traversal",
                    "scope": "project",
                    "source": { "provider": "evil", "path": "loras/x" }
                }
            }
        }),
    )
    .await;
    let evil_job_id = evil_job["id"].as_str().expect("job id").to_owned();
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{evil_job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Traversal completion."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["result"]["loraRegistered"], false);
    assert!(completed["result"]["loraRegistrationError"].is_string());

    // A `..`-traversing files entry is rejected even when a valid adapter
    // exists under the canonical output dir: registration only accepts plain
    // in-tree file names, so generation can never be pointed outside it.
    let traversal_lora_id = "lora_filestrav01";
    let traversal_dir = project_path.join("loras").join(traversal_lora_id);
    std::fs::create_dir_all(&traversal_dir).expect("adapter dir creates");
    write_test_safetensors(&traversal_dir.join("real.safetensors"));
    let (_, files_job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "lora_train",
            "projectId": project_id,
            "projectName": "Training Project",
            "requestedGpu": "auto",
            "payload": {
                "dryRun": false,
                "manifestEntry": {
                    "id": traversal_lora_id,
                    "name": "Files Traversal",
                    "scope": "project",
                    "source": { "provider": "evil", "path": "loras/x" },
                    "files": ["../../../../escape/evil.safetensors"]
                }
            }
        }),
    )
    .await;
    let files_job_id = files_job["id"].as_str().expect("job id").to_owned();
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{files_job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Files traversal completion."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["result"]["loraRegistered"], false);
    assert!(completed["result"]["loraRegistrationError"].is_string());
}

#[tokio::test]
async fn real_training_job_rejected_when_base_model_missing() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target_id = registry["targets"][0]["id"]
        .as_str()
        .expect("target id")
        .to_owned();
    let config = registry["targets"][0]["defaults"].clone();

    // No base model is installed: a real run is rejected with an actionable
    // message, but a dry run (plan preview) still succeeds.
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .unwrap()
        .contains("is not installed"));

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
async fn training_job_rejects_cpu_target() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target_id = registry["targets"][0]["id"]
        .as_str()
        .expect("target id")
        .to_owned();
    let mut config = registry["targets"][0]["defaults"].clone();
    // Targeting a CPU worker for a GPU-only job is rejected with an
    // actionable message (a dry run is GPU-routed too, so this also holds).
    config["advanced"]["requestedGpu"] = json!("cpu");

    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .unwrap()
        .contains("cannot target CPU workers"));
}

#[test]
fn insufficient_disk_space_threshold_is_strict_less_than() {
    assert!(insufficient_disk_space(100, 200));
    assert!(!insufficient_disk_space(200, 200));
    assert!(!insufficient_disk_space(300, 200));
}

#[test]
fn resolve_base_model_path_descends_into_hf_snapshot() {
    // Trainers read their weight tree (z-image/sdxl: tokenizer/ text_encoder/ unet|transformer/
    // vae/; ltx: transformer.safetensors / vae_*.safetensors / connector.safetensors) from inside
    // the HF snapshot dir, not the repo cache root. Resolving to the repo root made every
    // HF-cache base model fail at trainer load — z-image "tokenizer: No such file or directory",
    // sdxl "read vocab.json: No such file or directory", ltx "Path must point to a local file".
    let _env = isolate_hf_cache(); // seed under the tempdir, never a developer's real HF cache (sc-13834)
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "z_image_turbo")
        .expect("z_image_turbo target");
    let repo = target.base_model_repo.clone().expect("repo set");

    // Materialize an HF hub cache: refs/main -> a snapshot holding the tokenizer.
    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
    let revision = "abc123";
    let snapshot = repo_root.join("snapshots").join(revision);
    std::fs::create_dir_all(snapshot.join("tokenizer")).expect("create snapshot tree");
    std::fs::write(snapshot.join("tokenizer").join("tokenizer.json"), "{}")
        .expect("write tokenizer.json");
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("write refs/main");

    let resolved = resolve_base_model_path(&target, &data_dir);

    assert_eq!(
        resolved,
        snapshot.display().to_string(),
        "resolver must descend into the snapshot dir, not stop at the repo root"
    );
    assert!(
        std::path::Path::new(&resolved)
            .join("tokenizer")
            .join("tokenizer.json")
            .is_file(),
        "the component tree must be reachable from the resolved path"
    );
}

#[test]
fn tiered_turnkey_base_trains_on_bf16_tier() {
    // epic 9992 Krea 2 Raw (Path 1): SceneWorks/krea-2-raw-mlx ships bf16/ q8/ q4/ tier subdirs with NO
    // component tree at the snapshot root. Training reads the DENSE bf16 tier; a repo with only the q8
    // GENERATION tier installed is NOT training-ready (no dense weights to LoRA-train on).
    let _env = isolate_hf_cache(); // seed under the tempdir, never a developer's real HF cache (sc-13834)
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "krea_2_raw")
        .expect("krea_2_raw target");
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("SceneWorks/krea-2-raw-mlx"),
        "Path 1: training shares the generation turnkey re-host"
    );
    let repo = target.base_model_repo.clone().expect("repo set");

    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
    let revision = "abc123";
    let snapshot = repo_root.join("snapshots").join(revision);
    // Only the q8 GENERATION tier installed so far (no bf16 dense weights).
    std::fs::create_dir_all(snapshot.join("q8").join("transformer")).expect("q8 tree");
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("write refs/main");

    // The resolver points training at the bf16 tier (whether or not it is present yet).
    let resolved = resolve_base_model_path(&target, &data_dir);
    assert_eq!(
        resolved,
        snapshot.join("bf16").display().to_string(),
        "training must read the dense bf16 tier of a tiered turnkey"
    );
    // q8-only → NOT training-ready (the run-gate blocks it).
    assert!(
        !training_base_model_installed(&data_dir, &target),
        "a q8-only tiered turnkey is not training-ready"
    );

    // Install the dense bf16 component tree → training-ready.
    std::fs::create_dir_all(snapshot.join("bf16").join("transformer")).expect("bf16 transformer");
    std::fs::create_dir_all(snapshot.join("bf16").join("text_encoder")).expect("bf16 te");
    std::fs::create_dir_all(snapshot.join("bf16").join("vae")).expect("bf16 vae");
    assert!(
        training_base_model_installed(&data_dir, &target),
        "bf16 tier present → training-ready"
    );
}

#[test]
fn sdxl_family_tiered_turnkey_trains_on_its_bf16_unet_tier() {
    // sc-10613: the SDXL family packs its backbone under `unet/`, never `transformer/`. The readiness
    // gate tested `transformer/` only, so a fully-installed SDXL tiered turnkey (Illustrious) read as
    // un-installed forever and Training Studio blocked submit — while the resolver happily pointed at
    // the bf16 tier. Krea/LTX never caught this: they are DiTs and really do have `transformer/`.
    let _env = isolate_hf_cache(); // seed under the tempdir, never a developer's real HF cache (sc-13834)
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    // sc-10617: the real `illustrious_xl_v1_lora` registry target now bakes in the tiered-turnkey
    // repo, so assert the registry entry itself — not a synthetic override — resolves to bf16 and
    // gates on that tier. This pins that the shipped training target is training-ready off its turnkey.
    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "illustrious_xl_v1")
        .expect("illustrious_xl_v1 target");
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("SceneWorks/illustrious-xl-v1-mlx"),
        "the Illustrious v1 training target points at its tiered-turnkey re-host"
    );
    assert_eq!(
        target.kernel, "sdxl_lora",
        "Illustrious reuses the SDXL kernel"
    );
    let repo = target.base_model_repo.clone().expect("repo set");

    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
    let revision = "abc123";
    let snapshot = repo_root.join("snapshots").join(revision);
    // Only the q4 GENERATION tier installed so far (no dense weights).
    std::fs::create_dir_all(snapshot.join("q4").join("unet")).expect("q4 tree");
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("write refs/main");

    assert_eq!(
        resolve_base_model_path(&target, &data_dir),
        snapshot.join("bf16").display().to_string(),
        "an SDXL tiered turnkey must resolve training to its dense bf16 tier"
    );
    assert!(
        !training_base_model_installed(&data_dir, &target),
        "a q4-only SDXL turnkey carries no dense weights → not training-ready"
    );

    std::fs::create_dir_all(snapshot.join("bf16").join("unet")).expect("bf16 unet");
    std::fs::create_dir_all(snapshot.join("bf16").join("text_encoder")).expect("bf16 te");
    std::fs::create_dir_all(snapshot.join("bf16").join("vae")).expect("bf16 vae");
    assert!(
        training_base_model_installed(&data_dir, &target),
        "a bf16 tier with a unet/ backbone is training-ready"
    );
}

#[test]
fn flat_diffusers_snapshot_is_not_treated_as_a_tiered_turnkey() {
    // Backward-compat guard for sc-10613: `unet/` at the snapshot root marks a FLAT diffusers tree.
    // It must resolve unchanged, never descend into a `bf16/` subdir — even if a stray tier dir sits
    // alongside it. Driven off a synthetic flat repo rather than a shipped target's `base_model_repo`
    // so it exercises the resolution branch itself, independent of which targets ship a flat repo
    // (the stock `sdxl` target moved to its tiered turnkey in issue #1694 — see
    // `stock_sdxl_target_points_at_installed_turnkey`).
    let _env = isolate_hf_cache(); // seed under the tempdir, never a developer's real HF cache (sc-13834)
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    // A real SDXL-family target struct with its repo overridden to a flat upstream diffusers repo,
    // so only the resolution branch under test — not the shipped repo string — drives the assertion.
    let mut target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "sdxl")
        .expect("sdxl target");
    target.base_model_repo = Some("stabilityai/stable-diffusion-xl-base-1.0".to_owned());
    let repo = target.base_model_repo.clone().expect("repo set");

    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
    let revision = "def456";
    let snapshot = repo_root.join("snapshots").join(revision);
    for component in ["unet", "text_encoder", "vae", "bf16"] {
        std::fs::create_dir_all(snapshot.join(component)).expect("component dir");
    }
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("write refs/main");

    assert_eq!(
        resolve_base_model_path(&target, &data_dir),
        snapshot.display().to_string(),
        "a flat diffusers snapshot resolves to the snapshot root, not a bf16 tier"
    );
}

#[test]
fn stock_sdxl_target_points_at_installed_turnkey() {
    // Regression guard for issue #1694: the stock `sdxl` training target must name the same
    // `SceneWorks/sdxl-base-mlx` turnkey the catalog + engine install — pointing at the flat
    // upstream `stabilityai/stable-diffusion-xl-base-1.0` (which nothing downloads) made the
    // pre-flight gate report the installed base as missing and block every real SDXL run.
    // A dense `bf16/` tier resolves for training and passes the install gate; a q4-only install
    // (the generation default) carries no dense weights and must NOT be reported training-ready.
    let _env = isolate_hf_cache(); // seed under the tempdir, never a developer's real HF cache (sc-13834)
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "sdxl")
        .expect("sdxl target");
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("SceneWorks/sdxl-base-mlx"),
        "the stock SDXL target trains off the installed SceneWorks turnkey"
    );
    let repo = target.base_model_repo.clone().expect("repo set");

    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
    let revision = "def456";
    let snapshot = repo_root.join("snapshots").join(revision);
    // Only the q4 GENERATION tier installed so far (no dense weights).
    std::fs::create_dir_all(snapshot.join("q4").join("unet")).expect("q4 tree");
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("write refs/main");

    assert_eq!(
        resolve_base_model_path(&target, &data_dir),
        snapshot.join("bf16").display().to_string(),
        "the SDXL tiered turnkey resolves training to its dense bf16 tier"
    );
    assert!(
        !training_base_model_installed(&data_dir, &target),
        "a q4-only SDXL turnkey carries no dense weights → not training-ready"
    );

    std::fs::create_dir_all(snapshot.join("bf16").join("unet")).expect("bf16 unet");
    std::fs::create_dir_all(snapshot.join("bf16").join("text_encoder")).expect("bf16 te");
    std::fs::create_dir_all(snapshot.join("bf16").join("vae")).expect("bf16 vae");
    assert!(
        training_base_model_installed(&data_dir, &target),
        "a bf16 tier with a unet/ backbone is training-ready"
    );
}

#[test]
fn resolve_base_model_path_prefers_converted_mlx_dir_for_conversion_models() {
    // `requiresConversion` models (Wan) keep usable weights in <data>/models/mlx/<id>, while the
    // HF cache holds only the native *source* checkpoint the converter consumes. Resolving Wan
    // training to the HF source made the trainer fail ("wan umt5 tokenizer: No such file"); it
    // must read the converted dir, mirroring inference's local_mlx_dir.
    let _env = isolate_hf_cache(); // seed under the tempdir, never a developer's real HF cache (sc-13834)
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "wan_2_2")
        .expect("wan_2_2 (TI2V-5B) target");
    let repo = target.base_model_repo.clone().expect("repo set");

    // Materialize BOTH: the HF source snapshot (native checkpoint) and the converted MLX dir.
    // The converted dir must win.
    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
    let snapshot = repo_root.join("snapshots").join("rev0");
    std::fs::create_dir_all(&snapshot).expect("create snapshot");
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), "rev0").expect("write refs/main");

    let converted = data_dir.join("models").join("mlx").join("wan_2_2");
    std::fs::create_dir_all(&converted).expect("create converted dir");
    std::fs::write(converted.join("config.json"), "{}").expect("write config.json");

    let resolved = resolve_base_model_path(&target, &data_dir);

    assert_eq!(
        resolved,
        converted.display().to_string(),
        "conversion models must resolve to the converted MLX dir, not the HF source snapshot"
    );

    // Without the converted dir (config.json gates it), it falls back to the HF snapshot.
    std::fs::remove_file(converted.join("config.json")).expect("remove config.json");
    assert_eq!(
        resolve_base_model_path(&target, &data_dir),
        snapshot.display().to_string(),
        "with no converted dir, fall back to the HF snapshot"
    );
}
