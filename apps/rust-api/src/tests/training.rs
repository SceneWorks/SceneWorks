//! rust-api training tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[test]
fn platform_effective_training_catalog_preserves_mlx_defaults_and_seeds_candle_limits() {
    let mlx = crate::training::effective_training_targets_for_candle(false);
    let candle = crate::training::effective_training_targets_for_candle(true);

    for kernel in ["kolors_lora", "sd3_lora", "mage_flow_lora"] {
        let mlx_target = mlx
            .targets
            .iter()
            .find(|target| target.kernel == kernel)
            .expect("MLX target");
        let candle_target = candle
            .targets
            .iter()
            .find(|target| target.id == mlx_target.id)
            .expect("Candle projection of the same target");
        assert_eq!(
            mlx_target.defaults.advanced["gradientCheckpointing"],
            json!(true),
            "{kernel} keeps its pre-existing MLX checkpointing default"
        );
        assert!(
            mlx_target.defaults.advanced["sampleEvery"]
                .as_u64()
                .is_some_and(|value| value > 0),
            "{kernel} keeps its pre-existing MLX preview cadence"
        );
        assert_eq!(
            candle_target.defaults.advanced["gradientCheckpointing"],
            json!(false),
            "{kernel} Candle metadata must not advertise unsupported checkpointing"
        );
        assert_eq!(candle_target.defaults.advanced["sampleEvery"], json!(0));
        if kernel == "mage_flow_lora" {
            assert!(mlx_target
                .defaults
                .advanced
                .get("fullFinetuneConfig")
                .is_none());
            assert_eq!(
                candle_target.defaults.advanced["fullFinetuneConfig"],
                json!({
                    "mixedPrecision": "f32",
                    "gradientCheckpointing": false
                })
            );
        }
    }

    let candle_wan5 = candle
        .targets
        .iter()
        .find(|target| target.base_model == "wan_2_2")
        .expect("Wan TI2V-5B target");
    assert_eq!(candle_wan5.defaults.advanced["sampleEvery"], json!(0));
    let candle_wan14 = candle
        .targets
        .iter()
        .filter(|target| target.kernel == "wan_moe_lora")
        .collect::<Vec<_>>();
    assert_eq!(candle_wan14.len(), 2);
    assert!(candle_wan14.iter().all(|target| {
        target.defaults.advanced["gradientCheckpointing"] == json!(true)
            && target.defaults.advanced["sampleEvery"] == json!(0)
    }));

    let mlx_presets = crate::training::effective_training_presets_for_candle(false);
    let candle_presets = crate::training::effective_training_presets_for_candle(true);
    let mlx_kolors = mlx_presets
        .presets
        .iter()
        .find(|preset| preset.target_id == "kolors_lora")
        .expect("Kolors preset");
    let candle_kolors = candle_presets
        .presets
        .iter()
        .find(|preset| preset.id == mlx_kolors.id)
        .expect("Candle Kolors preset");
    assert_eq!(
        mlx_kolors.config.advanced["gradientCheckpointing"],
        json!(true)
    );
    assert_eq!(
        candle_kolors.config.advanced["gradientCheckpointing"],
        json!(false)
    );
    assert_eq!(candle_kolors.config.advanced["sampleEvery"], json!(0));
}

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
async fn parquet_import_job_queues_for_empty_dataset_and_finalizes_staged_items() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Parquet Import Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let project_path = std::path::PathBuf::from(project["path"].as_str().expect("project path"));
    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "LAION subset", "items": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let dataset_id = dataset["id"].as_str().expect("dataset id");
    let source = temp_dir.path().join("laion-metadata");
    std::fs::create_dir_all(&source).expect("source directory creates");
    std::fs::write(source.join("part-00000.snappy.parquet"), b"PAR1")
        .expect("placeholder shard writes");

    let (status, rejected) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/parquet-import-jobs"
        ),
        json!({
            "sourcePath": source.clone(),
            "maxItems": 100_001
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(rejected["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("100000")));

    let (status, job) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/parquet-import-jobs"
        ),
        json!({
            "sourcePath": source,
            "maxItems": 100_000,
            "minEdge": 384,
            "concurrency": 8,
            "captionIncludes": ["portrait", "person"],
            "captionExcludes": ["logo"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "dataset_parquet_import");
    assert_eq!(job["payload"]["urlColumn"], "URL");
    assert_eq!(job["payload"]["captionColumn"], "TEXT");
    assert_eq!(job["payload"]["maxItems"], 100_000);
    assert_eq!(job["payload"]["minEdge"], 384);
    assert_eq!(job["payload"]["concurrency"], 8);
    assert_eq!(
        job["payload"]["captionIncludes"],
        json!(["portrait", "person"])
    );
    assert_eq!(job["payload"]["captionExcludes"], json!(["logo"]));

    let job_id = job["id"].as_str().expect("job id");
    let staging = project_path
        .join("training")
        .join("datasets")
        .join(dataset_id)
        .join("parquet-imports")
        .join(job_id);
    std::fs::create_dir_all(&staging).expect("staging creates");
    std::fs::write(staging.join("pq_abc.png"), b"png-bytes").expect("staged image writes");
    let relative = format!("training/datasets/{dataset_id}/parquet-imports/{job_id}/pq_abc.png");
    let (status, finalized) = request(
        app,
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/parquet-import-finalize"
        ),
        json!({
            "jobId": job_id,
            "datasetVersion": dataset["version"],
            "items": [{
                "id": "pq_abc",
                "path": relative,
                "displayName": "pq_abc.png",
                "caption": {
                    "text": "a cinematic portrait",
                    "source": "imported",
                    "triggerWords": []
                },
                "width": 1024,
                "height": 768
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(finalized["version"], 2);
    assert_eq!(finalized["items"][0]["id"], "pq_abc");
    assert_eq!(
        finalized["items"][0]["caption"]["text"],
        "a cinematic portrait"
    );
    assert_eq!(finalized["items"][0]["caption"]["source"], "imported");
    assert!(project_path
        .join("training")
        .join("datasets")
        .join(dataset_id)
        .join("images")
        .join("pq_abc.png")
        .exists());
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
    let _env = isolate_hf_cache(); // hermetic: resolve the seeded base under the tempdir, never a dev's real HF cache (sc-13834/sc-13860)
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
    let _env = isolate_hf_cache(); // hermetic: resolve the seeded base under the tempdir, never a dev's real HF cache (sc-13834/sc-13860)
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
    claim_training_job(&app, &job_id).await;

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
            "workerId": TEST_TRAINING_WORKER_ID,
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

fn seed_installed_wan_i2v_a14b_training_base(data_dir: &std::path::Path) {
    #[cfg(target_os = "macos")]
    {
        let snapshot = materialize_snapshot(
            data_dir,
            "SceneWorks/wan2.2-i2v-a14b-mlx",
            "test-wan-i2v-a14b-training",
        );
        seed_wan_mlx_tier(&snapshot.join("bf16"), true);
    }

    #[cfg(not(target_os = "macos"))]
    seed_installed_training_base(data_dir, "wan_2_2_i2v_14b");
}

#[tokio::test]
async fn wan_a14b_registration_requires_and_registers_both_expert_files() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    seed_installed_wan_i2v_a14b_training_base(&settings.data_dir);
    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Wan Experts Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Wan expert set",
            "items": [{
                "assetId": asset["id"],
                "caption": { "text": "wanSubject portrait" }
            }]
        }),
    )
    .await;
    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target = registry["targets"]
        .as_array()
        .expect("targets")
        .iter()
        .find(|target| target["baseModel"] == "wan_2_2_i2v_14b")
        .expect("Wan I2V A14B target")
        .clone();

    let (status, missing_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target["id"],
            "datasetId": dataset["id"],
            "config": target["defaults"],
            "outputName": "Wan Missing",
            "dryRun": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{missing_job}");
    let missing_job_id = missing_job["id"].as_str().expect("missing job id");
    claim_training_job(&app, missing_job_id).await;
    let missing_output_dir = std::path::PathBuf::from(
        missing_job["payload"]["plan"]["output"]["outputDir"]
            .as_str()
            .expect("missing output dir"),
    );
    std::fs::create_dir_all(&missing_output_dir).expect("missing output dir creates");
    write_test_safetensors(&missing_output_dir.join("wan_missing.high_noise.safetensors"));
    let (status, missing_completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{missing_job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Only one Wan expert was written.",
            "workerId": TEST_TRAINING_WORKER_ID,
            "result": {
                "outputPath": missing_output_dir.join("wan_missing.high_noise.safetensors").display().to_string()
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(missing_completed["result"]["loraRegistered"], false);
    assert!(missing_completed["result"]["loraRegistrationError"]
        .as_str()
        .is_some_and(|detail| detail.contains("No declared trained adapter")));

    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target["id"],
            "datasetId": dataset["id"],
            "config": target["defaults"],
            "outputName": "Wan Pair",
            "dryRun": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["manifestEntry"]["files"],
        json!([
            "wan_pair.high_noise.safetensors",
            "wan_pair.low_noise.safetensors"
        ])
    );

    let job_id = job["id"].as_str().expect("job id");
    claim_training_job(&app, job_id).await;
    let output_dir = std::path::PathBuf::from(
        job["payload"]["plan"]["output"]["outputDir"]
            .as_str()
            .expect("output dir"),
    );
    std::fs::create_dir_all(&output_dir).expect("output dir creates");
    for name in [
        "wan_pair.high_noise.safetensors",
        "wan_pair.low_noise.safetensors",
    ] {
        write_test_safetensors(&output_dir.join(name));
    }
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Trained both Wan experts.",
            "workerId": TEST_TRAINING_WORKER_ID,
            "result": {
                "outputPath": output_dir.join("wan_pair.high_noise.safetensors").display().to_string()
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["result"]["loraRegistered"], true);

    let (_, loras) = request(
        app,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    let registered = loras
        .as_array()
        .expect("loras")
        .iter()
        .find(|entry| entry["name"] == "Wan Pair")
        .expect("registered Wan pair");
    assert_eq!(
        registered["files"],
        json!([
            "wan_pair.high_noise.safetensors",
            "wan_pair.low_noise.safetensors"
        ])
    );
}

#[test]
fn trusted_adapter_files_rejects_a_missing_wan_expert() {
    let dir = tempfile::tempdir().expect("temp dir");
    write_test_safetensors(&dir.path().join("wan_pair.high_noise.safetensors"));
    let declared = json!([
        "wan_pair.high_noise.safetensors",
        "wan_pair.low_noise.safetensors"
    ]);
    assert!(crate::training::trusted_adapter_files(Some(&declared), dir.path()).is_none());
    write_test_safetensors(&dir.path().join("wan_pair.low_noise.safetensors"));
    assert_eq!(
        crate::training::trusted_adapter_files(Some(&declared), dir.path()),
        Some(vec![
            "wan_pair.high_noise.safetensors".to_owned(),
            "wan_pair.low_noise.safetensors".to_owned()
        ])
    );
}

#[tokio::test]
async fn failed_or_unwritten_training_job_registers_no_lora() {
    let _env = isolate_hf_cache(); // hermetic: resolve the seeded base under the tempdir, never a dev's real HF cache (sc-13834/sc-13860)
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
    claim_training_job(&app, &failed_job_id).await;
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{failed_job_id}/progress"),
        json!({
            "status": "failed",
            "stage": "failed",
            "progress": 1,
            "message": "Training failed.",
            "workerId": TEST_TRAINING_WORKER_ID,
            "error": "CUDA out of memory"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // A job that reports completed but produced no weights must not leave a
    // broken registry entry either, and the failure is surfaced in the result.
    let (completed_no_weights_id, _, _) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    claim_training_job(&app, &completed_no_weights_id).await;
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{completed_no_weights_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Reported complete without weights.",
            "workerId": TEST_TRAINING_WORKER_ID
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
    let _env = isolate_hf_cache(); // hermetic: resolve the seeded base under the tempdir, never a dev's real HF cache (sc-13834/sc-13860)
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

/// The pre-side-effect snapshot fixed the already-terminal case but still left
/// a TOCTOU window: another terminal report could commit while completion was
/// registering its LoRA. Hold completion immediately before authoritative
/// acceptance, let `failed` win, then prove the losing 409 wrote no catalog.
#[tokio::test]
async fn racing_terminal_training_report_loses_before_lora_side_effects() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let (app, state) = create_app_with_state(settings.clone()).expect("app and state create");
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
    claim_training_job(&app, &job_id).await;

    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));
    *state.progress_before_accept_once.lock() = Some(barrier.clone());
    let completing_app = app.clone();
    let completing_job_id = job_id.clone();
    let completing_adapter_path = adapter_path.display().to_string();
    let completing = tokio::spawn(async move {
        request(
            completing_app,
            "POST",
            &format!("/api/v1/jobs/{completing_job_id}/progress"),
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Trained LoRA saved.",
                "workerId": TEST_TRAINING_WORKER_ID,
                "result": { "outputPath": completing_adapter_path }
            }),
        )
        .await
    });

    // Completion has parsed the request but has not yet attempted the atomic
    // ownership/terminal transition.
    barrier.wait().await;
    let (failed_status, failed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "failed",
            "stage": "failed",
            "progress": 1,
            "message": "Competing terminal report won.",
            "workerId": TEST_TRAINING_WORKER_ID,
            "error": "training failed"
        }),
    )
    .await;
    assert_eq!(failed_status, StatusCode::OK);
    assert_eq!(failed["status"], "failed");
    barrier.wait().await;

    let (completed_status, _) = completing.await.expect("completion request joins");
    assert_eq!(
        completed_status,
        StatusCode::CONFLICT,
        "the completion report that loses authoritative acceptance must 409"
    );

    let (status, job) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "failed");

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
        "the losing completed report must not register a ghost LoRA"
    );
}

/// A terminal row is committed before catalog/project side effects so a losing
/// report cannot create ghosts. A production worker does not repeat terminal
/// progress after the API has already committed `completed`, so recovery must
/// be API-driven across restart. The test explicitly replays one failing
/// same-terminal request only to pin event idempotency; it never completes side
/// effects and recovery still occurs without a successful worker retry.
#[tokio::test]
async fn startup_drain_recovers_accepted_terminal_side_effect_failure_without_successful_worker_retry(
) {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let (app, state) = create_app_with_state(settings.clone()).expect("app and state create");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Retry Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let (job_id, output_dir, adapter_path) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    stage_trained_adapter(&output_dir, &adapter_path);
    claim_training_job(&app, &job_id).await;

    let completion = json!({
        "status": "completed",
        "stage": "completed",
        "progress": 1,
        "message": "Trained LoRA saved.",
        "workerId": TEST_TRAINING_WORKER_ID,
        "result": { "outputPath": adapter_path.display().to_string() }
    });
    let mut failure_events = state.events.subscribe();
    *state.progress_side_effects_fail_once.lock() = true;
    let (first_status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        completion.clone(),
    )
    .await;
    assert_eq!(
        first_status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "the injected failure must happen after terminal acceptance"
    );
    let failure_event_names = drain_event_names(&mut failure_events).await;
    assert_eq!(
        failure_event_names
            .iter()
            .filter(|name| name.as_str() == "job.updated")
            .count(),
        1,
        "the committed terminal snapshot must be published exactly once on side-effect failure: \
         {failure_event_names:?}"
    );
    assert_eq!(
        failure_event_names
            .iter()
            .filter(|name| name.as_str() == "queue.updated")
            .count(),
        1,
        "the committed terminal transition must refresh queue counts despite the 500: \
         {failure_event_names:?}"
    );

    // A same-terminal worker retry does not create another queue transition.
    // Keep the handoff pending for startup recovery and prove the retry emits
    // neither a duplicate terminal snapshot nor a duplicate queue refresh.
    let mut retry_events = state.events.subscribe();
    *state.progress_side_effects_fail_once.lock() = true;
    let (retry_status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        completion,
    )
    .await;
    assert_eq!(retry_status, StatusCode::INTERNAL_SERVER_ERROR);
    let retry_event_names = drain_event_names(&mut retry_events).await;
    assert!(
        retry_event_names.is_empty(),
        "a failed same-terminal retry changes neither the job snapshot nor queue composition: \
         {retry_event_names:?}"
    );

    let (status, accepted) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(accepted["status"], "completed");
    assert_eq!(
        accepted["result"]["loraRegistered"],
        Value::Null,
        "failed post-acceptance work must remain pending, not look finalized"
    );
    let (status, failed_queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        failed_queue["activeJobs"]
            .as_array()
            .expect("active jobs array")
            .iter()
            .filter(|job| job["id"] == json!(job_id))
            .count(),
        0,
        "the durably terminal job must leave the active queue even while side effects are pending"
    );
    let (status, partially_registered) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        partially_registered
            .as_array()
            .expect("loras array")
            .iter()
            .filter(|item| item["name"] == json!("Aurora Style"))
            .count(),
        1,
        "the injected interruption occurs after the first idempotent catalog upsert"
    );

    // Drop every clone of the first process's router/state and reconstruct the
    // application from the same durable data. The production lifecycle task
    // invokes this drain immediately at startup and once per second thereafter;
    // call one cycle directly so the regression is deterministic and does not
    // substitute a manual terminal progress retry.
    drop(app);
    drop(state);
    let (restarted_app, restarted_state) =
        create_app_with_state(settings).expect("restarted app and state create");
    let mut recovery_events = restarted_state.events.subscribe();
    let recovery_task =
        crate::jobs::spawn_terminal_progress_side_effect_recovery(restarted_state.clone());
    let recovered = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let (status, job) = request(
                restarted_app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            if job["result"]["loraRegistered"] == json!(true) {
                break job;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("production startup recovery must finalize the durable pending row");
    assert_eq!(recovered["status"], "completed");
    assert_eq!(recovered["result"]["loraRegistered"], true);
    // The durable row flips BEFORE the recovery task publishes its augmented snapshot, so
    // aborting on the polled completion can race the `job.updated` publish away entirely
    // (observed under full-suite load 2026-08-17: event count 0). Await the event itself,
    // bounded, before aborting; the quiet-window drain below still collects any duplicates.
    let mut recovery_event_names = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        while let Some(message) = recovery_events.next().await {
            let is_job_updated = message.event == "job.updated";
            recovery_event_names.push(message.event);
            if is_job_updated {
                break;
            }
        }
    })
    .await;
    recovery_task.abort();
    recovery_event_names.extend(drain_event_names(&mut recovery_events).await);
    assert_eq!(
        recovery_event_names
            .iter()
            .filter(|name| name.as_str() == "job.updated")
            .count(),
        1,
        "recovery must publish the one augmented terminal snapshot: {recovery_event_names:?}"
    );
    assert_eq!(
        recovery_event_names
            .iter()
            .filter(|name| name.as_str() == "queue.updated")
            .count(),
        0,
        "recovery must not duplicate the queue transition already published at terminal acceptance: \
         {recovery_event_names:?}"
    );
    let (status, recovered_queue) =
        request(restarted_app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        recovered_queue["activeJobs"]
            .as_array()
            .expect("active jobs array")
            .iter()
            .filter(|job| job["id"] == json!(job_id))
            .count(),
        0,
        "queue counts must remain converged after recovery"
    );
    let (status, reconnect_events) =
        request_sse_prefix(restarted_app.clone(), "/api/v1/jobs/events", 3).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reconnect_events[0].0, "ready");
    assert_eq!(reconnect_events[1].0, "jobs.snapshot");
    assert!(reconnect_events[1].1["jobs"].is_array());
    assert_eq!(reconnect_events[2].0, "queue.updated");
    assert_eq!(
        reconnect_events[2].1["activeJobs"]
            .as_array()
            .expect("reconnect active jobs array")
            .iter()
            .filter(|job| job["id"] == json!(job_id))
            .count(),
        0,
        "a reconnect after crash recovery must receive an authoritative queue snapshot even \
         though the original terminal event and the same-terminal retry emitted no usable refresh"
    );

    assert_eq!(
        crate::jobs::recover_pending_terminal_progress_side_effects_once(&restarted_state)
            .await
            .expect("second recovery scans"),
        0,
        "a cleared handoff must not be processed twice"
    );
    let (status, loras) = request(
        restarted_app,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        loras
            .as_array()
            .expect("loras array")
            .iter()
            .filter(|item| item["name"] == json!("Aurora Style"))
            .count(),
        1,
        "retry-safe registration must upsert rather than duplicate"
    );
}

/// A fixed oldest-first batch must not let persistent failures monopolize the
/// recovery queue. The fixture creates one more row than fits in a batch and
/// poisons the full batch, leaving exactly one healthy row that the first pass
/// never reaches. The durable retry schedule must rotate the poison rows out so
/// that healthy row completes on the next pass, without retrying the poison
/// rows every second — and must then bring them back once that backoff expires,
/// rather than deferring them out of the queue for good.
///
/// (All 129 rows go terminal within the same second, and `updated_at` has
/// second granularity, so the row left out is the largest-`id` one rather than
/// the newest. Which row it is does not matter — only that it is healthy and
/// that both enumerations agree on it, which the shared ordering guarantees.)
///
/// Every *scan* below is driven at an explicit instant rather than at the wall
/// clock, so the schedule decides the outcome and machine speed cannot. The
/// deferrals themselves still stamp real time; the assertions are written to
/// hold for any instant those stamps can land on (sc-17640). The exact length
/// of the backoff is pinned separately, next to the schedule that computes it,
/// by `terminal_side_effect_retry_backoff_doubles_and_survives_restart` in
/// `sceneworks-core`.
#[tokio::test]
async fn terminal_side_effect_recovery_backoff_prevents_poison_batch_starvation() {
    use sceneworks_core::contracts::{JobStatus, JobType, ProgressStage};
    use sceneworks_core::jobs_store::{CreateJob, ProgressUpdate, RegisterWorker};
    use sceneworks_core::time::now_unix_seconds;

    const BATCH: usize = crate::jobs::PROGRESS_SIDE_EFFECT_RECOVERY_BATCH;

    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let (_, state) = create_app_with_state(settings.clone()).expect("app and state create");
    state
        .jobs_store
        .register_worker(RegisterWorker {
            worker_id: "recovery-fairness-worker".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("Test GPU".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");

    for _ in 0..=BATCH {
        state
            .jobs_store
            .create_job(CreateJob {
                job_type: JobType::ImageGenerate,
                project_id: None,
                project_name: None,
                payload: serde_json::Map::new(),
                requested_gpu: "auto".to_owned(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            })
            .expect("recovery fixture job creates");
        let claimed = state
            .jobs_store
            .claim_next_job("recovery-fairness-worker")
            .expect("fixture claim succeeds")
            .expect("fixture job is claimable");
        state
            .jobs_store
            .update_job_progress_with_outcome(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Failed,
                    stage: ProgressStage::Failed,
                    progress: 1.0,
                    message: "terminal fixture".to_owned(),
                    error: Some("fixture failure".to_owned()),
                    result: Some(serde_json::Map::new()),
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                    worker_id: Some("recovery-fairness-worker".to_owned()),
                },
            )
            .expect("terminal fixture is accepted");
    }

    let poison_ids = state
        .jobs_store
        .pending_terminal_progress_side_effect_job_ids(BATCH)
        .expect("initial recovery batch enumerates");
    assert_eq!(
        poison_ids.len(),
        BATCH,
        "fixture must poison one complete production batch"
    );
    state
        .progress_side_effects_fail_job_ids
        .lock()
        .extend(poison_ids.iter().cloned());

    // Freeze the instant every scan below evaluates dueness against, captured
    // BEFORE the first pass runs. Each poison row is deferred to
    // `(its own deferral instant) + backoff`, and every deferral happens at or
    // after `scan_at`, so relative to `scan_at` the whole batch is durably in
    // the future no matter how slow this machine is. Scanning at the real wall
    // clock instead made the final assertion a race against the 5-second
    // first-failure backoff: a pass slow enough to outlast it (ordinary load, a
    // busy CI runner) let correctly-deferred rows come back due, and the test
    // reddened unrelated PRs through the shared `rust:test` gate (sc-17640).
    let scan_at = now_unix_seconds();
    assert_eq!(
        crate::jobs::recover_pending_terminal_progress_side_effects_as_of(&state, scan_at)
            .await
            .expect("poison batch recovery returns"),
        0
    );
    let first_attempts = state.progress_side_effects_attempts.lock().clone();
    assert!(
        poison_ids
            .iter()
            .all(|job_id| first_attempts.get(job_id) == Some(&1)),
        "every poison row must be attempted exactly once in the first pass"
    );
    // Every deferral above committed at or after `scan_at`, so no row's durable
    // deadline can have landed on or before it — the assertions below are
    // decided by the retry schedule, never by elapsed real time.
    let deferred_by = now_unix_seconds();

    drop(state);
    let (_, restarted_state) =
        create_app_with_state(settings).expect("restarted app and state create");
    restarted_state
        .progress_side_effects_fail_job_ids
        .lock()
        .extend(poison_ids.iter().cloned());
    assert_eq!(
        crate::jobs::recover_pending_terminal_progress_side_effects_as_of(
            &restarted_state,
            scan_at
        )
        .await
        .expect("healthy recovery returns"),
        1,
        "durably deferred poison rows must not starve the newer healthy handoff after restart"
    );
    assert_eq!(
        crate::jobs::recover_pending_terminal_progress_side_effects_as_of(
            &restarted_state,
            scan_at
        )
        .await
        .expect("bounded retry scan returns"),
        0
    );
    assert_eq!(
        *restarted_state.progress_side_effects_attempts.lock(),
        std::collections::HashMap::new(),
        "poison rows must stay out of the due set across restart until durable backoff expires"
    );

    // ...and the deferral is a bounded backoff, not a silent drop. Without this
    // the assertion above passes just as happily if a row is deferred forever,
    // which would strand its side effects instead of retrying them.
    //
    // `deferred_by` was read after every deferral committed, so no deadline can
    // exceed it by more than one delay, and one delay is capped at
    // `PROGRESS_SIDE_EFFECT_RETRY_MAX_SECONDS` (5 minutes, in `sceneworks-core`).
    // A day past `deferred_by` is therefore beyond every deadline the schedule
    // can emit, by three orders of magnitude — so this scan too is decided by
    // the schedule, not by the clock. Raising that cap past a day would fail
    // this test loudly, on the `expired_attempts` assertion below.
    //
    // That assertion is the one with teeth: the `== 0` here is also what an
    // empty batch returns, so the two must stay together.
    assert_eq!(
        crate::jobs::recover_pending_terminal_progress_side_effects_as_of(
            &restarted_state,
            deferred_by.saturating_add(24 * 60 * 60),
        )
        .await
        .expect("expired backoff scan returns"),
        0,
        "the poison rows still fail, so none of them recovers"
    );
    let expired_attempts = restarted_state
        .progress_side_effects_attempts
        .lock()
        .clone();
    assert!(
        poison_ids
            .iter()
            .all(|job_id| expired_attempts.get(job_id) == Some(&1)),
        "once the durable backoff expires every poison row must be retried again"
    );
}

/// sc-11213 (F-028): a `completed` report from a caller that does not own the job
/// (here, any report carrying a `workerId` for a job whose owner is unset) must
/// receive the 409 AND must NOT register a LoRA — the ownership check has to gate
/// the completion side-effects, not fire too late.
#[tokio::test]
async fn non_owner_completed_report_registers_no_lora_and_409s() {
    let _env = isolate_hf_cache(); // hermetic: resolve the seeded base under the tempdir, never a dev's real HF cache (sc-13834/sc-13860)
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

    // Omitting workerId is not a privileged/internal path: the same unclaimed
    // job must reject the report before registration side effects.
    let (status, _rejected) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Ownerless completion.",
            "result": { "outputPath": adapter_path.display().to_string() }
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "an ownerless completed report must 409"
    );
    let (status, unchanged) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(unchanged["status"], "queued");

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
    let _env = isolate_hf_cache(); // hermetic: resolve the seeded base under the tempdir, never a dev's real HF cache (sc-13834/sc-13860)
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
    claim_training_job(&app, &job_id).await;

    // The claimed worker's completion wins the race and registers normally.
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Trained LoRA saved.",
            "workerId": TEST_TRAINING_WORKER_ID,
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
    claim_training_job(&app, &job_id).await;

    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Crafted completion.",
            "workerId": TEST_TRAINING_WORKER_ID
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
    claim_training_job(&app, &evil_job_id).await;
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{evil_job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Traversal completion.",
            "workerId": TEST_TRAINING_WORKER_ID
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
    claim_training_job(&app, &files_job_id).await;
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{files_job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Files traversal completion.",
            "workerId": TEST_TRAINING_WORKER_ID
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
fn ltx_training_resolves_and_requires_the_turnkey_q4_tier() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");
    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.id == "ltx_video_lora")
        .expect("LTX training target");
    let repo = target.base_model_repo.as_deref().expect("turnkey repo");
    let repo_root = huggingface_repo_cache_path(&data_dir, repo).expect("repo cache path");
    let revision = "ltx-q4-training";
    let snapshot = repo_root.join("snapshots").join(revision);
    let q4 = snapshot.join("q4");
    std::fs::create_dir_all(&q4).expect("q4 tier");
    for file in [
        "quantize_config.json",
        "transformer.safetensors",
        "connector.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
    ] {
        std::fs::write(q4.join(file), "x").expect("q4 training component");
    }
    let gemma = snapshot.join("gemma");
    std::fs::create_dir_all(&gemma).expect("Gemma co-requisite");
    std::fs::write(gemma.join("config.json"), r#"{"model_type":"gemma3_text"}"#)
        .expect("Gemma config");
    std::fs::write(gemma.join("tokenizer.json"), r#"{"model":{"type":"BPE"}}"#)
        .expect("Gemma tokenizer");
    std::fs::write(
        gemma.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#,
    )
    .expect("Gemma shard index");
    write_test_safetensors_with_keys(
        &gemma.join("model-00001-of-00001.safetensors"),
        &["model.embed_tokens.weight".to_owned()],
    );
    // A bf16 tier existing beside q4 must not divert the packed QLoRA trainer.
    std::fs::create_dir_all(snapshot.join("bf16")).expect("bf16 tier");
    std::fs::create_dir_all(repo_root.join("refs")).expect("refs");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("refs/main");

    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::Ready
    );
    assert_eq!(
        resolve_base_model_path(&target, &data_dir),
        q4.display().to_string()
    );

    // API and worker share one readiness predicate. A shard index must not be able to escape the
    // Gemma snapshot even when the referenced outside file is itself a valid safetensors file.
    let outside = snapshot.join("outside.safetensors");
    write_test_safetensors_with_keys(&outside, &["outside.weight".to_owned()]);
    std::fs::write(
        gemma.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.embed_tokens.weight":"../outside.safetensors"}}"#,
    )
    .expect("traversal index");
    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::TrainingTierMissing,
        "an index traversal must never pass API readiness"
    );
    std::fs::write(
        gemma.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#,
    )
    .expect("restore safe index");

    std::fs::write(
        gemma.join("model-00001-of-00001.safetensors"),
        b"filename-only placeholder",
    )
    .expect("corrupt Gemma shard");
    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::TrainingTierMissing,
        "a structurally invalid indexed shard must never pass API readiness"
    );
    write_test_safetensors_with_keys(
        &gemma.join("model-00001-of-00001.safetensors"),
        &["model.embed_tokens.weight".to_owned()],
    );

    std::fs::remove_file(q4.join("vae_encoder.safetensors")).expect("tear q4 training tier");
    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::TrainingTierMissing
    );
    std::fs::write(q4.join("vae_encoder.safetensors"), "x").expect("repair q4 training tier");
    std::fs::remove_file(gemma.join("model-00001-of-00001.safetensors"))
        .expect("tear Gemma co-requisite");
    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::TrainingTierMissing,
        "a torn Gemma co-requisite must block the real run before the worker claims it"
    );
    let message = training_base_unavailable_message(
        TrainingBaseStatus::TrainingTierMissing,
        &target.base_model,
    )
    .expect("missing tier blocks");
    assert!(message.contains("packed q4"), "{message}");
}

#[test]
fn huggingface_snapshot_dirs_ranks_materialized_over_torn_refs_main() {
    // sc-13915: a torn/empty `refs/main` (or a partial download) must not front an empty snapshot over a
    // fully-materialized sibling — the `.into_iter().next()` callers (training gate/resolver, model
    // catalog) would otherwise report an installed base as missing. Ports the worker's sc-13834 file-count
    // hardening to the rust-api resolver. No HF-cache env here: this drives the helper on an explicit root.
    let temp = tempfile::tempdir().expect("tempdir");
    let repo_root = temp.path().join("models--Foo--bar");
    let snapshots = repo_root.join("snapshots");

    let full = snapshots.join("full_rev");
    let empty = snapshots.join("empty_rev");
    std::fs::create_dir_all(&empty).expect("empty snapshot dir");
    std::fs::create_dir_all(full.join("transformer")).expect("full snapshot tree");
    std::fs::write(full.join("config.json"), "{}").expect("config");
    std::fs::write(full.join("transformer").join("model.safetensors"), "x").expect("weight");
    std::fs::create_dir_all(repo_root.join("refs")).expect("refs dir");

    // `refs/main` points at the EMPTY (torn) snapshot → the materialized one must still be first.
    std::fs::write(repo_root.join("refs").join("main"), "empty_rev").expect("refs/main");
    let ordered = crate::huggingface_snapshot_dirs(&repo_root);
    assert_eq!(
        ordered.first(),
        Some(&full),
        "a materialized snapshot must outrank an empty one a torn refs/main points to"
    );

    // A `refs/main` that names a materialized snapshot is authoritative (stays first); the empty sibling
    // is deprioritized, NOT dropped, so iterate-all scans still see it.
    std::fs::write(repo_root.join("refs").join("main"), "full_rev").expect("refs/main");
    let ordered = crate::huggingface_snapshot_dirs(&repo_root);
    assert_eq!(
        ordered.first(),
        Some(&full),
        "a materialized refs/main is authoritative"
    );
    assert!(
        ordered.contains(&empty),
        "empty snapshots are deprioritized, not removed from the list"
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
fn mage_flow_foundation_targets_resolve_the_installed_flat_mirrors() {
    // sc-14054: Mage's q4/q8/bf16 catalog variants all materialize the same flat dense snapshot on
    // every platform. The training targets must therefore resolve their live SceneWorks mirrors
    // directly and report them ready without inventing a physical `bf16/` subdirectory.
    // Base only since sc-15277: `mage_flow_edit_base` is still a catalog GENERATION model, but it
    // is no longer a training target (no `mlx-gen-mage` edit trainer exists — sc-15320), so there is
    // no training pre-flight to resolve for it.
    {
        let (base_model, expected_repo) = ("mage_flow_base", "SceneWorks/Mage-Flow-Base");
        let _env = isolate_hf_cache();
        let temp = tempfile::tempdir().expect("tempdir");
        let data_dir = temp.path().join("data");
        let target = crate::builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.base_model == base_model)
            .unwrap_or_else(|| panic!("{base_model} target present"));
        assert_eq!(target.base_model_repo.as_deref(), Some(expected_repo));

        let repo_root =
            huggingface_repo_cache_path(&data_dir, expected_repo).expect("repo cache path");
        let revision = "rev14054";
        let snapshot = repo_root.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        std::fs::create_dir_all(repo_root.join("refs")).expect("refs dir");
        std::fs::write(repo_root.join("refs").join("main"), revision).expect("refs/main");
        // A readable config/payload certifies a materialized flat snapshot for the resolver path.
        std::fs::write(snapshot.join("config.json"), "{}").expect("flat snapshot config");

        assert_eq!(
            resolve_base_model_path(&target, &data_dir),
            snapshot.display().to_string(),
            "{base_model}: training resolves the installed flat mirror root"
        );
        assert_eq!(
            training_base_model_status(&data_dir, &target),
            TrainingBaseStatus::Ready,
            "{base_model}: the installed flat mirror is training-ready"
        );
    }
}

#[test]
fn mage_flow_foundation_targets_are_missing_when_not_installed() {
    // sc-14056: the counterpart to the "installed ⇒ Ready" case above. An absent install is
    // `Missing` — the generic "install it" message, not the tier-specific one — whatever the mirror
    // layout is.
    //
    // This case is NOT evidence that a tiered layout is handled: an empty data dir yields `Missing`
    // under either layout, so it does not discriminate, and an earlier revision of this file wrongly
    // cited it as proof that "a future tiered re-host fails loudly". The layout-sensitivity proof is
    // `mage_flow_training_base_status_discriminates_flat_from_tiered_layouts` below. That is now the
    // live case rather than a hypothetical: sc-14980 re-hosts Mage to PHYSICAL per-tier artifacts
    // with shared TE/VAE co-requisites, so `TrainingTierMissing` is genuinely REACHABLE for Mage —
    // a q4/q8-only install must produce it rather than a false `Ready`.
    // Base only since sc-15277 (see `mage_flow_foundation_targets_resolve_the_installed_flat_mirrors`).
    {
        let base_model = "mage_flow_base";
        let _env = isolate_hf_cache();
        let temp = tempfile::tempdir().expect("tempdir");
        let data_dir = temp.path().join("data");
        let target = crate::builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.base_model == base_model)
            .unwrap_or_else(|| panic!("{base_model} target present"));

        // Nothing on disk (no repo cache, no managed install) ⇒ Missing, not TrainingTierMissing.
        assert_eq!(
            training_base_model_status(&data_dir, &target),
            TrainingBaseStatus::Missing,
            "{base_model}: an uninstalled flat mirror is Missing"
        );
        let message = training_base_unavailable_message(
            training_base_model_status(&data_dir, &target),
            &target.base_model,
        )
        .unwrap_or_else(|| panic!("{base_model}: a real run must be blocked when uninstalled"));
        assert!(
            message.contains("not installed"),
            "{base_model}: Missing message should say 'not installed': {message}"
        );
    }
}

/// sc-14056 / sc-14980 — the "a future tiered re-host fails loudly" guard, made DISCRIMINATING.
///
/// The first cut of this coverage asserted `Missing` against an EMPTY data dir. That outcome is
/// identical whether the resolver understands tiered layouts or not, so the test could not have
/// failed if the tiered guard were deleted — it proved nothing about the claim it was cited for.
/// sc-14980 is concurrently re-hosting Mage to a tiered layout, so this is a live risk, not a
/// hypothetical: a tiered snapshot whose dense `bf16/` tier was never downloaded must be reported as
/// `TrainingTierMissing`, not silently green-lit and then discovered mid-run.
///
/// Fabricate the three reachable on-disk shapes and assert they land on THREE distinct outcomes.
/// Delete the `snapshot_is_tiered_turnkey` branch in `training_base_model_status` and the third case
/// flips to `Ready`; break the flat path and the first flips.
#[test]
fn mage_flow_training_base_status_discriminates_flat_from_tiered_layouts() {
    // Base only since sc-15277 (see `mage_flow_foundation_targets_resolve_the_installed_flat_mirrors`).
    {
        let (base_model, repo) = ("mage_flow_base", "SceneWorks/Mage-Flow-Base");
        let target = crate::builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.base_model == base_model)
            .unwrap_or_else(|| panic!("{base_model} target present"));

        // Materialize one HF snapshot for `repo` whose layout the closure builds, and report the
        // status + the path training would load from.
        let status_for = |layout: &dyn Fn(&std::path::Path)| {
            let _env = isolate_hf_cache();
            let temp = tempfile::tempdir().expect("tempdir");
            let data_dir = temp.path().join("data");
            let repo_root = huggingface_repo_cache_path(&data_dir, repo).expect("repo cache path");
            let revision = "rev14980";
            let snapshot = repo_root.join("snapshots").join(revision);
            std::fs::create_dir_all(&snapshot).expect("snapshot dir");
            std::fs::create_dir_all(repo_root.join("refs")).expect("refs dir");
            std::fs::write(repo_root.join("refs").join("main"), revision).expect("refs/main");
            layout(&snapshot);
            (
                training_base_model_status(&data_dir, &target),
                resolve_base_model_path(&target, &data_dir),
                snapshot,
            )
        };

        // 1. TODAY'S mirror: a flat dense diffusers snapshot. Ready, resolved at the snapshot root.
        let (status, path, snapshot) = status_for(&|snapshot: &std::path::Path| {
            std::fs::write(snapshot.join("config.json"), "{}").expect("flat config");
            for component in ["transformer", "text_encoder", "vae"] {
                std::fs::create_dir_all(snapshot.join(component)).expect("flat component");
            }
        });
        assert_eq!(
            status,
            TrainingBaseStatus::Ready,
            "{base_model}: a flat dense snapshot is training-ready"
        );
        assert_eq!(
            path,
            snapshot.display().to_string(),
            "{base_model}: a flat snapshot resolves at its root, never a bf16 tier"
        );

        // 2. A tiered re-host (sc-14980) WITH the dense tier installed. Still Ready — but now the
        //    trainer must be pointed INTO `bf16/`, which is the other half of the guard.
        let (status, path, snapshot) = status_for(&|snapshot: &std::path::Path| {
            for tier in ["q4", "q8", "bf16"] {
                std::fs::create_dir_all(snapshot.join(tier)).expect("tier dir");
            }
            for component in ["transformer", "text_encoder", "vae"] {
                std::fs::create_dir_all(snapshot.join("bf16").join(component))
                    .expect("bf16 component");
            }
        });
        assert_eq!(
            status,
            TrainingBaseStatus::Ready,
            "{base_model}: a tiered re-host with a complete bf16/ tier is training-ready"
        );
        assert_eq!(
            path,
            snapshot.join("bf16").display().to_string(),
            "{base_model}: a tiered re-host must train from the dense bf16/ tier"
        );

        // 3. The failure this guard exists for: a tiered re-host where only the quantized generation
        //    tiers were downloaded. There ARE weights on disk and the repo-level completion marker
        //    would say "installed" — but none of them are dense, so training must refuse LOUDLY with
        //    the tier-specific status rather than the generic Missing (or, worse, Ready).
        let (status, _, _) = status_for(&|snapshot: &std::path::Path| {
            for tier in ["q4", "q8"] {
                for component in ["transformer", "text_encoder", "vae"] {
                    std::fs::create_dir_all(snapshot.join(tier).join(component))
                        .expect("quant component");
                }
            }
        });
        assert_eq!(
            status,
            TrainingBaseStatus::TrainingTierMissing,
            "{base_model}: a tiered re-host without the dense bf16/ tier must fail loudly as \
             TrainingTierMissing — NOT Ready, and not the generic Missing"
        );
        let message = training_base_unavailable_message(status, &target.base_model)
            .unwrap_or_else(|| panic!("{base_model}: TrainingTierMissing must block a real run"));
        // It must be the TIER message, not the generic "install the model" one: the repo IS
        // installed, just not with dense weights, and the actionable step is different.
        assert!(
            message.contains("training tier") || message.contains("bf16"),
            "{base_model}: TrainingTierMissing must produce the tier-specific message naming the \
             dense tier, not the generic install prompt: {message}"
        );
        assert_ne!(
            message,
            training_base_unavailable_message(TrainingBaseStatus::Missing, &target.base_model)
                .expect("Missing blocks too"),
            "{base_model}: the tier message must differ from the plain not-installed message"
        );
    }
}

#[test]
fn training_is_full_finetune_reads_the_network_type() {
    // sc-14056: the submit-gate predicate that routes a run to the full base fine-tune memory envelope.
    let mut config = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.base_model == "mage_flow_base")
        .expect("mage_flow_base target present")
        .defaults;
    // The Mage default (advanced.networkType == "lora") is an adapter run, not a full fine-tune.
    assert!(!training_is_full_finetune(&config));
    // A LoKr adapter is also not a full fine-tune.
    config
        .advanced
        .insert("networkType".to_owned(), serde_json::json!("lokr"));
    assert!(!training_is_full_finetune(&config));
    // Only "full" selects the full base fine-tune path (case/whitespace-insensitive).
    config
        .advanced
        .insert("networkType".to_owned(), serde_json::json!("  Full "));
    assert!(training_is_full_finetune(&config));
}

/// sc-14056 review — the full-tune rejection named the TARGET ("Mage-Flow Base LoRA"), producing
/// "A full base fine-tune of Mage-Flow Base LoRA at 1024px…". A target is named for the artifact it
/// normally produces, which is exactly the wrong noun in a message about fine-tuning the base itself.
#[test]
fn training_base_model_label_strips_the_output_kind_suffix() {
    let registry = crate::builtin_training_targets();
    let label_of = |base_model: &str| {
        let target = registry
            .targets
            .iter()
            .find(|target| target.base_model == base_model)
            .unwrap_or_else(|| panic!("{base_model} target present"));
        (target.name.clone(), training_base_model_label(target))
    };

    let (name, label) = label_of("mage_flow_base");
    assert_eq!(name, "Mage-Flow Base LoRA", "the target name is unchanged");
    assert_eq!(
        label, "Mage-Flow Base",
        "the rejection must name the base model, not the LoRA target"
    );

    // The rule is the trailing output-kind word ONLY — never more. Across the whole built-in
    // registry, every label must be its target name minus exactly that suffix.
    assert_eq!(label_of("sdxl").1, "Stable Diffusion XL");
    assert_eq!(label_of("z_image_turbo").1, "Z-Image-Turbo");
    for target in &registry.targets {
        let label = training_base_model_label(target);
        assert!(
            target.name.starts_with(&label) && !label.is_empty(),
            "label {label:?} must be a non-empty prefix of target name {:?}",
            target.name
        );
        assert!(
            target.name.len() - label.len() <= " ControlNet".len(),
            "label {label:?} dropped more than one trailing word from {:?}",
            target.name
        );
    }

    // Degenerate inputs stay safe rather than producing an empty label.
    let mut bare = registry
        .targets
        .iter()
        .find(|target| target.base_model == "mage_flow_base")
        .expect("target")
        .clone();
    bare.name = "LoRA".to_owned();
    assert_eq!(
        training_base_model_label(&bare),
        "LoRA",
        "a name that is ONLY the suffix must not strip to empty"
    );
    bare.name = "Anima".to_owned();
    assert_eq!(training_base_model_label(&bare), "Anima");
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
fn epic_8506_rehost_targets_point_at_their_installed_turnkeys() {
    // sc-13860: the epic-8506 MLX re-host moved the catalog to `SceneWorks/*-mlx` turnkeys, but these
    // four targets still named the flat UPSTREAM repos the app no longer downloads, so the pre-flight
    // gate reported the installed base as missing and 400'd every real run (the SDXL #1694 class). Pin
    // that each now names its turnkey AND resolves training to the dense `bf16/` tier the trainer needs,
    // gating on that tier (a q4-only generation install must NOT read as training-ready). One case per
    // backbone shape: DiT turnkeys pack `transformer/` (z-image, SD3.5), the SDXL-arch Kolors packs
    // `unet/` — the same split `bf16_component_tree_present` probes.
    for (base_model, expected_repo, backbone) in [
        (
            "z_image_turbo",
            "SceneWorks/z-image-turbo-mlx",
            "transformer",
        ),
        ("sd3_5_large", "SceneWorks/sd3.5-large-mlx", "transformer"),
        ("sd3_5_medium", "SceneWorks/sd3.5-medium-mlx", "transformer"),
        ("kolors", "SceneWorks/kolors-mlx", "unet"),
    ] {
        let _env = isolate_hf_cache(); // seed under the tempdir, never a developer's real HF cache (sc-13834)
        let temp = tempfile::tempdir().expect("tempdir");
        let data_dir = temp.path().join("data");

        let target = crate::builtin_training_targets()
            .targets
            .into_iter()
            .find(|t| t.base_model == base_model)
            .unwrap_or_else(|| panic!("{base_model} target present"));
        assert_eq!(
            target.base_model_repo.as_deref(),
            Some(expected_repo),
            "{base_model} trains off the installed SceneWorks turnkey, not the dead upstream repo"
        );
        let repo = target.base_model_repo.clone().expect("repo set");

        let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
        let revision = "rev13860";
        let snapshot = repo_root.join("snapshots").join(revision);
        // Only the q4 GENERATION tier installed so far (no dense weights).
        std::fs::create_dir_all(snapshot.join("q4").join(backbone)).expect("q4 tree");
        std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
        std::fs::write(repo_root.join("refs").join("main"), revision).expect("write refs/main");

        assert_eq!(
            resolve_base_model_path(&target, &data_dir),
            snapshot.join("bf16").display().to_string(),
            "{base_model}: a tiered turnkey resolves training to its dense bf16 tier"
        );
        assert!(
            !training_base_model_installed(&data_dir, &target),
            "{base_model}: a q4-only turnkey carries no dense weights → not training-ready"
        );

        std::fs::create_dir_all(snapshot.join("bf16").join(backbone)).expect("bf16 backbone");
        std::fs::create_dir_all(snapshot.join("bf16").join("text_encoder")).expect("bf16 te");
        std::fs::create_dir_all(snapshot.join("bf16").join("vae")).expect("bf16 vae");
        assert!(
            training_base_model_installed(&data_dir, &target),
            "{base_model}: a bf16 tier with the backbone + TE + VAE is training-ready"
        );
    }
}

#[test]
fn training_base_status_distinguishes_missing_from_tier_missing() {
    // sc-13860 AC #4: a tiered turnkey installed for GENERATION (q4 only, no dense bf16) must gate as
    // `TrainingTierMissing` — an actionable "install the training tier (bf16)" message — NOT the bare
    // `Missing` "not installed", which sent #1694 reporters in circles trying to reinstall a base they
    // already had. A repo with nothing on disk stays `Missing`; a full bf16 tree is `Ready`.
    let _env = isolate_hf_cache(); // seed under the tempdir, never a developer's real HF cache (sc-13834)
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "z_image_turbo")
        .expect("z_image_turbo target");
    let repo = target.base_model_repo.clone().expect("repo set");
    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");

    // Nothing on disk → Missing, with the original "not installed / install it from the catalog" text.
    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::Missing
    );
    let missing =
        training_base_unavailable_message(TrainingBaseStatus::Missing, &target.base_model)
            .expect("Missing blocks a real run");
    assert!(missing.contains("is not installed"), "{missing}");
    assert!(
        missing.contains("Install it from the model catalog"),
        "{missing}"
    );
    assert!(
        !missing.contains("training tier"),
        "the Missing message must not claim a tier is installed: {missing}"
    );

    // The q4 GENERATION tier installed, no dense bf16 → TrainingTierMissing, with the tier-specific text.
    let revision = "rev13860";
    let snapshot = repo_root.join("snapshots").join(revision);
    std::fs::create_dir_all(snapshot.join("q4").join("transformer")).expect("q4 tree");
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("write refs/main");
    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::TrainingTierMissing
    );
    let tier = training_base_unavailable_message(
        TrainingBaseStatus::TrainingTierMissing,
        &target.base_model,
    )
    .expect("TrainingTierMissing blocks a real run");
    assert!(tier.contains("training tier"), "{tier}");
    assert!(tier.contains("bf16"), "{tier}");
    assert!(
        tier.contains("installed for generation"),
        "the tier message tells the user their generation install is fine: {tier}"
    );

    // Dense bf16 component tree present → Ready, no error message.
    std::fs::create_dir_all(snapshot.join("bf16").join("transformer")).expect("bf16 transformer");
    std::fs::create_dir_all(snapshot.join("bf16").join("text_encoder")).expect("bf16 te");
    std::fs::create_dir_all(snapshot.join("bf16").join("vae")).expect("bf16 vae");
    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::Ready
    );
    assert_eq!(
        training_base_unavailable_message(TrainingBaseStatus::Ready, &target.base_model),
        None,
        "a ready base does not block the run"
    );
}

// ---- macOS Wan training-base resolution (sc-13878) ----
//
// On macOS the Wan targets' diffusers `base_model_repo` is never installed; the native MLX trainer
// loads the `SceneWorks/wan2.2-*-mlx` turnkey instead, so the gate + resolver must key off THAT repo.
// These pin the macOS behavior; the platform-blind manifest side is guarded in sceneworks-core's
// `macos_wan_targets_have_a_macos_installable_mlx_turnkey_base`. macOS-only: off-Mac the overrides are
// inert (`None`) and the shared diffusers-turnkey path — covered above — runs unchanged.

/// Seed a flat MLX Wan tier (the layout the native trainer reads) at `dir`.
#[cfg(target_os = "macos")]
fn seed_wan_mlx_tier(dir: &std::path::Path, dual_expert: bool) {
    std::fs::create_dir_all(dir).expect("tier dir");
    for file in [
        "config.json",
        "t5_encoder.safetensors",
        "vae.safetensors",
        "tokenizer.json",
    ] {
        std::fs::write(dir.join(file), "{}").expect("seed shared component");
    }
    if dual_expert {
        std::fs::write(dir.join("low_noise_model.safetensors"), "x").expect("low expert");
        std::fs::write(dir.join("high_noise_model.safetensors"), "x").expect("high expert");
    } else {
        std::fs::write(dir.join("model.safetensors"), "x").expect("dense expert");
    }
}

/// Materialize an HF-cache snapshot for `repo` under `data_dir` and return the snapshot dir.
#[cfg(target_os = "macos")]
fn materialize_snapshot(
    data_dir: &std::path::Path,
    repo: &str,
    revision: &str,
) -> std::path::PathBuf {
    let repo_root = huggingface_repo_cache_path(data_dir, repo).expect("repo cache path");
    let snapshot = repo_root.join("snapshots").join(revision);
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    std::fs::create_dir_all(repo_root.join("refs")).expect("refs dir");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("refs/main");
    snapshot
}

#[cfg(target_os = "macos")]
#[test]
fn macos_wan_5b_dense_flat_root_is_ready_and_resolves_to_root() {
    // The installed 5B turnkey ships its dense (bf16) weights as flat MLX files at the snapshot ROOT
    // (legacy layout, `quantization: None`), with q4 as a generation subdir. The gate must read Ready
    // and the resolver must hand the trainer the ROOT (its flat files), NOT the diffusers repo.
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "wan_2_2")
        .expect("wan_lora target");
    // Independently pin the turnkey repo (not read from the map) so a map typo is caught: the gate
    // resolves via `macos_wan_mlx_repo`, and if that drifts from this seed location it finds nothing.
    let snapshot = materialize_snapshot(&data_dir, "SceneWorks/wan2.2-ti2v-5b-mlx", "revwan5b");
    seed_wan_mlx_tier(&snapshot, false); // dense flat root
    seed_wan_mlx_tier(&snapshot.join("q4"), false); // a generation tier that must NOT be selected

    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::Ready,
        "the dense flat-root turnkey is training-ready on macOS"
    );
    assert_eq!(
        resolve_base_model_path(&target, &data_dir),
        snapshot.display().to_string(),
        "the trainer loads the dense flat root, not the q4 tier or the diffusers repo"
    );

    // Mutation check: tear one dense component from the root → the dense tier is gone, and with only the
    // q4 generation tier left the gate is TrainingTierMissing (proves `wan_tier_complete` is load-bearing).
    std::fs::remove_file(snapshot.join("model.safetensors")).expect("remove dense DiT");
    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::TrainingTierMissing,
        "a torn dense tier with only q4 present is not training-ready"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn macos_wan_5b_q4_only_is_training_tier_missing() {
    // A macOS Wan install for GENERATION is q4 by default (no dense bf16). Training needs full precision,
    // so a q4-only turnkey gates as TrainingTierMissing with the actionable "install the bf16 tier" text —
    // NOT the bare Missing that would send the user to reinstall a base they already have.
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "wan_2_2")
        .expect("wan_lora target");
    let snapshot = materialize_snapshot(&data_dir, "SceneWorks/wan2.2-ti2v-5b-mlx", "revwan5bq4");
    seed_wan_mlx_tier(&snapshot.join("q4"), false); // only the generation tier

    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::TrainingTierMissing
    );
    let message = training_base_unavailable_message(
        TrainingBaseStatus::TrainingTierMissing,
        &target.base_model,
    )
    .expect("TrainingTierMissing blocks a real run");
    assert!(message.contains("bf16"), "{message}");
    assert!(message.contains("installed for generation"), "{message}");
}

#[cfg(target_os = "macos")]
#[test]
fn macos_wan_5b_uninstalled_turnkey_is_missing() {
    // No turnkey on disk → Missing (the base was never installed), NOT a resolve against the diffusers
    // repo that can't exist on macOS.
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "wan_2_2")
        .expect("wan_lora target");

    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::Missing
    );
}

#[cfg(target_os = "macos")]
#[test]
fn macos_wan_a14b_bf16_subdir_dual_expert_is_ready() {
    // The A14B MoE turnkeys ship their dense tier as a `bf16/` SUBDIR of dual-expert files
    // (low_noise + high_noise). The resolver descends into `bf16/`; the gate reads Ready. Covers BOTH
    // A14B siblings (T2V + I2V) — the story is titled "5B" but the fix must cover all three Wan targets.
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    for (base_model, repo, revision) in [
        (
            "wan_2_2_t2v_14b",
            "SceneWorks/wan2.2-t2v-a14b-mlx",
            "revt2v14b",
        ),
        (
            "wan_2_2_i2v_14b",
            "SceneWorks/wan2.2-i2v-a14b-mlx",
            "revi2v14b",
        ),
    ] {
        let target = crate::builtin_training_targets()
            .targets
            .into_iter()
            .find(|t| t.base_model == base_model)
            .unwrap_or_else(|| panic!("{base_model} training target present"));
        let snapshot = materialize_snapshot(&data_dir, repo, revision);
        seed_wan_mlx_tier(&snapshot.join("q4"), true); // generation tier (not selected)
        seed_wan_mlx_tier(&snapshot.join("bf16"), true); // dense training tier

        assert_eq!(
            training_base_model_status(&data_dir, &target),
            TrainingBaseStatus::Ready,
            "{base_model}: dense bf16 dual-expert tier is training-ready"
        );
        assert_eq!(
            resolve_base_model_path(&target, &data_dir),
            snapshot.join("bf16").display().to_string(),
            "{base_model}: the dual-expert trainer loads the dense bf16 subdir"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real weights: needs the SceneWorks/wan2.2-ti2v-5b-mlx turnkey (dense/bf16 tier) in the HF cache"]
fn macos_wan_5b_real_turnkey_is_training_ready() {
    // Walking-skeleton proof against REAL installed weights: run the production gate + resolver against
    // the developer's actual HF hub cache and confirm the installed Wan 5B turnkey is training-ready and
    // the resolved dir holds the flat MLX files the native trainer loads. Run with:
    //   cargo test -p sceneworks-rust-api --lib macos_wan_5b_real_turnkey_is_training_ready -- --ignored
    // (In production the app defaults HF_HOME to ~/.cache/huggingface at startup, server.rs; the test
    // runner strips it, so pin the real hub explicitly.)
    let home = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"));
    let _env = pin_hf_hub_cache(&home.join(".cache").join("huggingface").join("hub"));
    let data_dir = home.join("SceneWorks").join("data");
    let target = crate::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "wan_2_2")
        .expect("wan_lora target");

    assert_eq!(
        training_base_model_status(&data_dir, &target),
        TrainingBaseStatus::Ready,
        "the installed Wan 5B turnkey must be training-ready (install its dense bf16 tier if this fails)"
    );

    let resolved = resolve_base_model_path(&target, &data_dir);
    let dir = std::path::Path::new(&resolved);
    for file in [
        "model.safetensors",
        "config.json",
        "t5_encoder.safetensors",
        "vae.safetensors",
        "tokenizer.json",
    ] {
        assert!(
            dir.join(file).exists(),
            "resolved training dir {resolved} must hold the flat MLX file `{file}` the trainer reads"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn macos_wan_repos_match_manifest_macos_downloads() {
    // Pin the rust-api `macos_wan_mlx_repo` map to the manifest's macOS download repo for each Wan base,
    // so the two cannot drift (the gate/resolver only work if they resolve the repo macOS actually
    // installs). Complements the platform-blind sceneworks-core guard.
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;
    use std::collections::BTreeSet;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc embedded");
    let catalog: Value = serde_json::from_str(&strip_jsonc_comments(raw)).expect("manifest parses");
    let models = catalog["models"].as_array().expect("models array");

    // The BASE download repos macOS installs, excluding co-requisites — the A14B entries carry a
    // `lightx2v/Wan2.2-Lightning` co-requisite (sc-10030) that installs ALONGSIDE the base, not AS it,
    // so it is not the training base the map resolves.
    let macos_base_download_repos = |base_model: &str| -> BTreeSet<String> {
        models
            .iter()
            .find(|m| m["id"].as_str() == Some(base_model))
            .and_then(|m| m["downloads"].as_array())
            .expect("base model entry with downloads")
            .iter()
            .filter(|d| !crate::models::is_co_requisite_download(d))
            .filter(|d| match d.get("platforms").and_then(Value::as_array) {
                None => true,
                Some(platforms) => platforms.iter().any(|p| p.as_str() == Some("macos")),
            })
            .filter_map(|d| d["repo"].as_str().map(str::to_owned))
            .collect()
    };

    for base_model in ["wan_2_2", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"] {
        let mapped = crate::training::macos_wan_mlx_repo(base_model)
            .unwrap_or_else(|| panic!("{base_model} is mapped to a macOS turnkey"));
        assert_eq!(
            macos_base_download_repos(base_model),
            BTreeSet::from([mapped.to_owned()]),
            "{base_model}: the rust-api macOS turnkey map ({mapped}) must equal the manifest's macOS \
             base download repo(s) — they drifted"
        );
    }
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

/// sc-15036 (epic 14034 F6) — the whole point of the story, end to end through the real API.
///
/// A completed FULL base fine-tune must register its checkpoint as a **model**, not a LoRA, and
/// that model must be selectable at generation. Before this, the same run registered nothing: the
/// plan declared `outputKind: "lora"`, the LoRA registrar claimed it, and `trusted_adapter_files`
/// (which requires `is_file()` on a declared adapter name) found no adapter — so an ~8 GB trained
/// checkpoint sat on disk with no home.
///
/// Discriminating on four independent axes, each of which fails under a different regression:
///   1. the output dir is in the MODEL tree, not the LoRA tree (submit-side fork);
///   2. the result carries `baseCheckpointRegistered` and NOT `loraRegistered` (the LoRA registrar
///      must defer, and the base registrar must claim);
///   3. `/api/v1/models` — the exact payload the Image Studio picker filters — offers the entry
///      with the family surface that makes it selectable, and `/api/v1/loras` does NOT;
///   4. a second completion of the same job does not duplicate the catalog entry.
#[tokio::test]
async fn completed_full_finetune_registers_a_selectable_model_not_a_lora() {
    let _env = isolate_hf_cache();
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

    let (job_id, output_dir) =
        submit_full_finetune_job(app.clone(), &project_id, &settings.data_dir).await;

    // (1) A full fine-tune produces a MODEL, so its weights land in the model tree — never
    // `loras/`, whose entries the Studio offers as adapters to stack onto a base.
    assert!(
        output_dir.starts_with(settings.data_dir.join("models").join("finetunes")),
        "a full fine-tune must not write into the LoRA tree: {}",
        output_dir.display()
    );

    claim_training_job_as_mlx_worker(&app, &job_id).await;
    stage_trained_base_checkpoint(&output_dir);

    let progress_path = format!("/api/v1/jobs/{job_id}/progress");
    let complete = || {
        request(
            app.clone(),
            "POST",
            &progress_path,
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Fine-tuned base saved.",
                "workerId": TEST_TRAINING_WORKER_ID,
                "result": {}
            }),
        )
    };
    let (status, completed) = complete().await;
    assert_eq!(status, StatusCode::OK);

    // (2) Exactly one registrar claims the job, and it is the base-checkpoint one.
    assert_eq!(
        completed["result"]["baseCheckpointRegistered"], true,
        "registration outcome: {}",
        completed["result"]
    );
    assert!(
        completed["result"].get("loraRegistered").is_none(),
        "the LoRA registrar must DEFER on a base checkpoint, not claim or fail it: {}",
        completed["result"]
    );
    let model_id = completed["result"]["baseCheckpointModelId"]
        .as_str()
        .expect("model id")
        .to_owned();
    assert!(
        model_id.starts_with("finetune_"),
        "an 8 GB base checkpoint must not be labelled lora_…: {model_id}"
    );
    assert!(completed["result"]["baseCheckpointManifestPath"]
        .as_str()
        .is_some_and(|path| path.ends_with("user.models.jsonc")));

    // (3) The trained base is a normal, installed, selectable image model. This is the exact
    // payload `generationModelsForType` / `imageModelServesMode` filter in the Image Studio, so
    // asserting it here IS the selectability contract.
    let (status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let entry = models
        .as_array()
        .expect("models array")
        .iter()
        .find(|model| model["id"] == json!(model_id.as_str()))
        .unwrap_or_else(|| panic!("the fine-tuned base must appear in the model catalog"))
        .clone();
    assert_eq!(entry["type"], "image");
    assert_eq!(entry["family"], "mage-flow");
    assert_eq!(entry["catalogScope"], "user");
    assert_eq!(
        entry["importSourceShape"], "transformer_directory",
        "the completion trust boundary must stamp the exact Mage directory route"
    );
    // The common picker gates remain healthy. Operation capabilities come from the exact active
    // provider topology: Mage is MLX-only, so macOS advertises t2i while a Candle-only host
    // truthfully withdraws it. The lane-parameterized projection test covers both directions.
    assert_eq!(entry["installState"], "installed");
    assert_ne!(
        entry["usable"],
        json!(false),
        "an explicitly unusable entry is dropped from every picker"
    );
    let serves_t2i = entry["capabilities"]
        .as_array()
        .expect("capabilities")
        .contains(&json!("text_to_image"));
    assert_eq!(serves_t2i, cfg!(target_os = "macos"));
    assert_eq!(entry["macSupport"]["supported"], json!(true));
    // The family surface it inherits from its base, without which the Studio falls back to a bare
    // 4-option resolution list.
    assert_eq!(entry["adapter"], "mlx_mage");
    assert_eq!(entry["defaults"]["resolution"], "1024x1024");
    assert_eq!(entry["limits"]["requiresDimensionsMultipleOf"], json!(16));
    // ...and `paths.model` is the recomputed output dir the worker's render lane resolves.
    assert_eq!(
        entry["paths"]["model"],
        json!(output_dir.display().to_string())
    );
    assert_eq!(entry["provenance"]["kind"], "training");
    assert_eq!(entry["provenance"]["trainingJobId"], json!(job_id.as_str()));
    // sc-15328 — and it must NOT advertise adapters. `apply_model_manifest_defaults` synthesizes
    // `loraCompatibility.families = [family]` from the family token, which for a fine-tune is an
    // advertisement nothing can honour: the router's `mage-flow` arm, `mage_finetuned_available`
    // and `mlx_gen_mage::load_finetuned` all refuse adapters. Advertised, the API accepted the job
    // and no worker could claim it — it queued forever with no error.
    //
    // 🔴 This originally asserted only that the manifest KEY was absent, and passed while the hang
    // was still live: `families_from_value_chain` falls back to the top-level `family` (asserted
    // present above), so the "no declared LoRA families" branch never fired and the strip was a
    // no-op. Assert the WITHDRAWAL SHAPE the projection emits, and — below — that a submission
    // carrying a LoRA is actually refused. Absence of a key proves nothing; the 400 does.
    //
    // Generated Mage full fine-tunes render plain text-to-image on both native lanes after
    // sc-18480. Neither provider accepts adapters, so every deployment projection must publish the
    // same explicit withdrawal while leaving the full model selectable.
    assert_eq!(
        entry["loraCompatibility"]["families"],
        json!([]),
        "the withdrawal must be an EXPLICIT empty family list — removing the key alone falls back \
         to `family` and silently re-advertises: {:?}",
        entry.get("loraCompatibility")
    );
    assert_eq!(
        entry["loraCompatibility"]["supported"],
        json!(false),
        "the web picker fails OPEN on an empty family list, so it needs this explicit signal"
    );

    // ...and the withdrawal is LOAD-BEARING, not decorative: a generation carrying a LoRA is
    // refused at submit, terminally, instead of being accepted and queued forever.
    //
    // This is the assertion whose absence let the original no-op ship. The two failure modes
    // are distinguishable by their message, and only one of them is this gate: "has no declared
    // LoRA families" means the advertisement was withdrawn and the request died here; "LoRA not
    // found" would mean the families check PASSED (the `family` fallback fired) and we only got
    // as far as hydrating the adapter — which is exactly what the pre-fix code did.
    let (status, refused) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "model": model_id,
            "prompt": "a fine-tune with an adapter it cannot load",
            "loras": [{ "id": "any_adapter" }]
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a fine-tune + LoRA submission must be refused, not queued: {refused}"
    );
    let detail = refused["detail"].as_str().unwrap_or_default().to_owned();
    assert!(
        detail.contains("no declared LoRA families"),
        "the refusal must come from the withdrawn advertisement; \"LoRA not found\" would mean \
         the `family` fallback re-advertised it and the hang is still live. Got: {detail}"
    );

    // It is NOT a LoRA — registering it as one would offer it as an adapter the engine cannot load.
    let (status, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !loras
            .as_array()
            .expect("loras array")
            .iter()
            .any(|lora| lora["id"] == json!(model_id.as_str())),
        "a full fine-tune must never appear in the LoRA library"
    );

    // (4) A repeated terminal report refreshes rather than duplicates.
    let created_at = entry["createdAt"].clone();
    let (status, _) = complete().await;
    assert_eq!(status, StatusCode::OK);
    let (_, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    let matches: Vec<Value> = models
        .as_array()
        .expect("models array")
        .iter()
        .filter(|model| model["id"] == json!(model_id.as_str()))
        .cloned()
        .collect();
    assert_eq!(matches.len(), 1, "a re-run must not duplicate the entry");
    assert_eq!(
        matches[0]["createdAt"], created_at,
        "the original createdAt is preserved across a re-registration"
    );

    // (5) …and it is REMOVABLE. An ~8 GB base checkpoint that the catalog offers but the Models
    // page cannot delete would be a one-way disk leak, so the lifecycle is closed here rather than
    // assumed: `<data>/models/finetunes/<id>` sits under the `<data>/models` root
    // `remove_owned_artifacts` allows, and `paths.model` is what `model_artifact_paths` resolves.
    let (status, deleted) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/models/{model_id}?permanent=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "delete failed: {deleted}");
    assert_eq!(deleted["removedManifestEntry"], true);
    assert!(
        !output_dir.exists(),
        "the checkpoint directory must be removed from disk: {}",
        output_dir.display()
    );
    let (_, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert!(
        !models
            .as_array()
            .expect("models array")
            .iter()
            .any(|model| model["id"] == json!(model_id.as_str())),
        "the deleted fine-tune must leave the catalog without a restart"
    );
}

/// sc-15036 — a TORN checkpoint (weights written, architecture config missing, or vice versa) must
/// be refused rather than registered as a model that then fails at load. Surfaced on the job result
/// so the outcome is observable, never silently dropped.
///
/// Discriminating: the same job, same everything, differing only in which of the two component
/// files exists.
#[tokio::test]
async fn a_torn_full_finetune_checkpoint_is_refused_with_a_reason() {
    let _env = isolate_hf_cache();
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

    let (job_id, output_dir) =
        submit_full_finetune_job(app.clone(), &project_id, &settings.data_dir).await;
    claim_training_job_as_mlx_worker(&app, &job_id).await;

    // Weights, but no architecture config — `MageTransformer::load` cannot read it.
    std::fs::create_dir_all(&output_dir).expect("output dir creates");
    write_test_safetensors(
        &output_dir.join(sceneworks_core::base_weights::MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE),
    );

    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Fine-tuned base saved.",
            "workerId": TEST_TRAINING_WORKER_ID,
            "result": {}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["result"]["baseCheckpointRegistered"], false);
    let reason = completed["result"]["baseCheckpointRegistrationError"]
        .as_str()
        .expect("a refusal reason");
    assert!(
        reason.contains("config.json"),
        "the refusal must name what was missing: {reason}"
    );

    // Nothing was written to the catalog.
    let (_, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert!(
        !models
            .as_array()
            .expect("models array")
            .iter()
            .any(|model| model["catalogScope"] == json!("user")),
        "a torn checkpoint must not register a user model"
    );
}
