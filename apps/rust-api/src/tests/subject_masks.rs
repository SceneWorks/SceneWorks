//! rust-api subject-mask tests (epic 2123, sc-2126): job enqueue, worker mask write, coverage
//! report, served mask files, and the replacement-mask upload with its field-level validation.
use super::support::*;
use base64::Engine as _;

fn png_rgb(w: u32, h: u32, shade: u8) -> Vec<u8> {
    let mut out = Vec::new();
    image::RgbImage::from_pixel(w, h, image::Rgb([shade, 64, 128]))
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("encode image");
    out
}

fn png_mask(w: u32, h: u32, value: u8) -> Vec<u8> {
    let mut out = Vec::new();
    image::GrayImage::from_pixel(w, h, image::Luma([value]))
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("encode mask");
    out
}

/// A project with one dataset of three distinct 16x8 images. Returns (app, project id, dataset id,
/// item ids, content hashes).
async fn masked_dataset_fixture(
    temp_dir: &tempfile::TempDir,
) -> (axum::Router, String, String, Vec<String>, Vec<String>) {
    let app = create_app(test_settings(temp_dir)).expect("app creates");
    let (project_id, dataset_id, item_ids, hashes) = masked_dataset_in(&app).await;
    (app, project_id, dataset_id, item_ids, hashes)
}

/// [`masked_dataset_fixture`] on an existing app: (project id, dataset id, item ids, hashes).
async fn masked_dataset_in(app: &axum::Router) -> (String, String, Vec<String>, Vec<String>) {
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Subject mask project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let mut items = Vec::new();
    for index in 0..3u8 {
        let (status, upload) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/training/uploads"),
            &format!("image_{index}.png"),
            "image/png",
            &png_rgb(16, 8, index * 60),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{upload}");
        items.push(json!({
            "path": upload["file"]["path"],
            "displayName": format!("image_{index}.png"),
        }));
    }
    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Masked set", "items": items }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{dataset}");
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let item_ids = dataset["items"]
        .as_array()
        .expect("items")
        .iter()
        .map(|item| item["id"].as_str().expect("item id").to_owned())
        .collect();
    let hashes = dataset["items"]
        .as_array()
        .expect("items")
        .iter()
        .map(|item| item["contentHash"].as_str().expect("hash").to_owned())
        .collect();
    (project_id, dataset_id, item_ids, hashes)
}

#[tokio::test]
async fn subject_mask_job_enqueues_every_image_and_refuses_an_unknown_item() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, hashes) = masked_dataset_fixture(&temp_dir).await;
    let base = format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}");

    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("{base}/subject-mask-jobs"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "dataset_subject_mask");
    assert_eq!(job["payload"]["datasetName"], "Masked set");
    let job_items = job["payload"]["items"].as_array().expect("job items");
    assert_eq!(job_items.len(), 3, "one work item per image");
    for ((job_item, item_id), hash) in job_items.iter().zip(&item_ids).zip(&hashes) {
        assert_eq!(job_item["itemId"], item_id.as_str());
        assert_eq!(job_item["contentHash"], hash.as_str());
        assert!(job_item["imagePath"]
            .as_str()
            .unwrap()
            .contains(&dataset_id));
    }

    let (status, subset) = request(
        app.clone(),
        "POST",
        &format!("{base}/subject-mask-jobs"),
        json!({ "itemIds": [item_ids[1]] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(subset["payload"]["items"].as_array().unwrap().len(), 1);

    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("{base}/subject-mask-jobs"),
        json!({ "itemIds": ["not_an_item"] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "subject_mask_unknown_item");
    assert_eq!(error["context"]["field"], "itemIds");
}

#[tokio::test]
async fn worker_masks_persist_report_full_coverage_and_flag_the_empty_one() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, hashes) = masked_dataset_fixture(&temp_dir).await;
    let base = format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}");

    let (status, before) = request(
        app.clone(),
        "GET",
        &format!("{base}/subject-masks"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        (before["masked"].as_u64(), before["total"].as_u64()),
        (Some(0), Some(3))
    );

    let b64 = |bytes: Vec<u8>| base64::engine::general_purpose::STANDARD.encode(bytes);
    let (status, stored) = request(
        app.clone(),
        "POST",
        &format!("{base}/subject-masks"),
        json!({
            "space": "sam3-person",
            "items": [
                { "contentHash": hashes[0], "maskPng": b64(png_mask(16, 8, 255)) },
                { "contentHash": hashes[1], "maskPng": b64(png_mask(16, 8, 0)) },
                { "contentHash": hashes[2], "maskPng": b64(png_mask(16, 8, 255)) },
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stored}");
    assert_eq!(stored["stored"], 3);

    let (status, report) = request(
        app.clone(),
        "GET",
        &format!("{base}/subject-masks"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(report["masked"], 3);
    assert_eq!(report["total"], 3);
    assert_eq!(report["empty"], 1);
    assert_eq!(report["items"][1]["itemId"], item_ids[1].as_str());
    assert_eq!(report["items"][1]["empty"], true);
    assert_eq!(report["items"][0]["empty"], false);
    assert_eq!(report["items"][0]["source"], "auto");

    // The mask path is servable through the project files route as a single-channel PNG.
    let mask_path = report["items"][0]["maskPath"].as_str().expect("mask path");
    assert_eq!(
        mask_path,
        format!("training/datasets/{dataset_id}/masks/{}.png", hashes[0])
    );
    let (status, _, bytes) = request_raw(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/files/{mask_path}"),
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let decoded = image::load_from_memory(&bytes).expect("served mask decodes");
    assert_eq!(decoded.color(), image::ColorType::L8);
    assert_eq!((decoded.width(), decoded.height()), (16, 8));

    // A space the API does not know is refused rather than mislabelled.
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("{base}/subject-masks"),
        json!({ "space": "other", "items": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn uploaded_replacement_mask_persists_and_invalid_uploads_are_field_errors() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _hashes) = masked_dataset_fixture(&temp_dir).await;
    let base = format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}");
    let endpoint = format!("{base}/items/{}/subject-mask", item_ids[2]);

    // Half-size, same aspect → accepted and resized to the image.
    let (status, report) = request_multipart_upload(
        app.clone(),
        &endpoint,
        "mask.png",
        "image/png",
        &png_mask(8, 4, 255),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["masked"], 1);
    assert_eq!(report["uploaded"], 1);
    assert_eq!(report["items"][2]["source"], "upload");
    assert_eq!(report["items"][2]["empty"], false);
    // Persisted: a fresh GET sees it.
    let (_, reread) = request(
        app.clone(),
        "GET",
        &format!("{base}/subject-masks"),
        Value::Null,
    )
    .await;
    assert_eq!(reread["items"][2]["source"], "upload");

    let (status, error) = request_multipart_upload(
        app.clone(),
        &endpoint,
        "mask.txt",
        "text/plain",
        b"not an image",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "subject_mask_unsupported_type");
    assert_eq!(error["context"]["field"], "file");

    let (status, error) = request_multipart_upload(
        app.clone(),
        &endpoint,
        "mask.png",
        "image/png",
        &png_mask(8, 8, 255),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "subject_mask_dimensions_mismatch");
    assert_eq!(error["context"]["field"], "file");

    let (status, _) = request_multipart_upload(
        app.clone(),
        &format!("{base}/items/not_an_item/subject-mask"),
        "mask.png",
        "image/png",
        &png_mask(16, 8, 255),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Review fix (sc-2126): a mask whose image left the dataset while the GPU job ran is skipped — the
/// worker POST succeeds, the live mask is stored, and the vanished hash is reported, instead of a
/// 404 that throws away every generated mask.
#[tokio::test]
async fn worker_masks_for_a_vanished_image_are_skipped_not_a_batch_failure() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, _item_ids, hashes) = masked_dataset_fixture(&temp_dir).await;
    let base = format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}");
    let removed = "ab".repeat(32);
    let b64 = |bytes: Vec<u8>| base64::engine::general_purpose::STANDARD.encode(bytes);
    let (status, stored) = request(
        app.clone(),
        "POST",
        &format!("{base}/subject-masks"),
        json!({
            "space": "sam3-person",
            "items": [
                { "contentHash": hashes[0], "maskPng": b64(png_mask(16, 8, 255)) },
                { "contentHash": removed, "maskPng": b64(png_mask(16, 8, 255)) },
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stored}");
    assert_eq!(stored["stored"], 1);
    assert_eq!(stored["skipped"], json!([removed]));
    assert_eq!(stored["report"]["masked"], 1);
    assert_eq!(stored["report"]["items"][0]["hasMask"], true);
}

/// The default Z-Image character preset and its config, as the Training Studio submits it.
async fn zimage_character_preset(app: &axum::Router) -> Value {
    let (_, presets) = request(app.clone(), "GET", "/api/v1/training/presets", Value::Null).await;
    presets["presets"]
        .as_array()
        .expect("preset list")
        .iter()
        .find(|preset| preset["id"] == "z_image_turbo_lora.character.adamw8bit.balanced")
        .expect("Z-Image character preset")
        .clone()
}

fn preset_submission(preset: &Value, dataset_id: &str, dry_run: bool) -> Value {
    json!({
        "targetId": "z_image_turbo_lora",
        "presetId": preset["id"],
        "presetVersion": preset["version"],
        "datasetId": dataset_id,
        "config": preset["config"],
        "outputName": "Character",
        "dryRun": dry_run
    })
}

/// sc-2124: the Z-Image character preset turns subject-masked loss on, so a freshly imported dataset
/// (no masks yet) must still be accepted: the job carries a `subjectMaskPrepass` naming exactly the
/// images still without a mask, for the worker to generate with SAM3 before it trains. A fully
/// masked dataset carries none. Mutation: drop the prepass (or refuse missing masks) ⇒ red.
#[tokio::test]
async fn character_preset_queues_mask_generation_for_unmasked_images() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, hashes) = masked_dataset_fixture(&temp_dir).await;
    let preset = zimage_character_preset(&app).await;
    assert_eq!(preset["config"]["advanced"]["subjectMaskLoss"], true);
    let jobs = format!("/api/v1/projects/{project_id}/training/jobs");
    let submit = || {
        request(
            app.clone(),
            "POST",
            &jobs,
            preset_submission(&preset, &dataset_id, true),
        )
    };

    let (status, job) = submit().await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let prepass = &job["payload"]["subjectMaskPrepass"];
    assert_eq!(prepass["projectId"], project_id.as_str());
    assert_eq!(prepass["datasetId"], dataset_id.as_str());
    let listed = |prepass: &Value| {
        prepass["items"]
            .as_array()
            .expect("prepass items")
            .iter()
            .map(|item| {
                assert!(item["imagePath"].as_str().unwrap().contains(&dataset_id));
                (
                    item["itemId"].as_str().unwrap().to_owned(),
                    item["contentHash"].as_str().unwrap().to_owned(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        listed(prepass),
        item_ids
            .iter()
            .cloned()
            .zip(hashes.iter().cloned())
            .collect::<Vec<_>>()
    );

    // One image masked: only the other two are generated.
    let b64 = |bytes: Vec<u8>| base64::engine::general_purpose::STANDARD.encode(bytes);
    let masks =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/subject-masks");
    let (status, stored) = request(
        app.clone(),
        "POST",
        &masks,
        json!({ "space": "sam3-person", "items": [
            { "contentHash": hashes[0], "maskPng": b64(png_mask(16, 8, 255)) },
        ] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stored}");
    let (status, job) = submit().await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        listed(&job["payload"]["subjectMaskPrepass"])
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        item_ids[1..].to_vec()
    );

    // Every image masked: nothing to generate.
    let (status, stored) = request(
        app.clone(),
        "POST",
        &masks,
        json!({ "space": "sam3-person", "items": [
            { "contentHash": hashes[1], "maskPng": b64(png_mask(16, 8, 255)) },
            { "contentHash": hashes[2], "maskPng": b64(png_mask(16, 8, 255)) },
        ] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stored}");
    let (status, job) = submit().await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert!(job["payload"].get("subjectMaskPrepass").is_none(), "{job}");
}

/// sc-2124: an image SAM3 already examined and found no person in cannot be fixed by generating
/// again, so it is refused at submit as a field error on `subjectMaskLoss` naming the image and the
/// fix. Mutation: skip the empty-mask check ⇒ the job queues ⇒ red.
#[tokio::test]
async fn character_preset_refuses_an_empty_subject_mask_with_a_field_error() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, _, hashes) = masked_dataset_fixture(&temp_dir).await;
    let preset = zimage_character_preset(&app).await;
    let b64 = |bytes: Vec<u8>| base64::engine::general_purpose::STANDARD.encode(bytes);
    let (status, stored) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/subject-masks"),
        json!({ "space": "sam3-person", "items": [
            { "contentHash": hashes[1], "maskPng": b64(png_mask(16, 8, 0)) },
        ] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stored}");
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        preset_submission(&preset, &dataset_id, true),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "training_field_error");
    assert_eq!(error["context"]["field"], "subjectMaskLoss");
    let detail = error["detail"].as_str().unwrap();
    assert!(detail.contains("1 of 3: image_1.png"), "{detail}");
    assert!(detail.contains("Upload a subject mask"), "{detail}");
}

/// sc-2124: a REAL run that must generate masks needs the SAM3 Person Segmenter installed (nothing
/// is downloaded mid-job): without it the submit is a field error naming the model; with it the run
/// queues with the prepass and the SAM3 catalog entry attached for the worker. Mutation: drop the
/// install check ⇒ the first submit queues ⇒ red.
#[tokio::test]
async fn real_character_run_needs_the_sam3_segmenter_to_generate_masks() {
    use sceneworks_core::training::training_adapter::ZIMAGE_TURBO_TRAINING_ADAPTER_V2 as V2;
    let _env = isolate_hf_cache();
    // The shipped catalog, so the SAM3 entry (and its install state) is the production one.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let seed = |settings: &Settings, sam3: bool| {
        sceneworks_core::builtin_manifests::seed_builtin_manifests(
            &settings.config_dir,
            sceneworks_core::builtin_manifests::SeedMode::Overwrite,
        )
        .expect("builtin manifests seed");
        seed_installed_base_model(&settings.data_dir);
        let hub = sceneworks_core::hf_home::huggingface_hub_cache_dir(&settings.data_dir);
        let snapshot = |repo: &str, revision: &str| {
            let dir = hub
                .join(format!("models--{}", repo.replace('/', "--")))
                .join("snapshots")
                .join(revision);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        };
        std::fs::write(snapshot(V2.repo, V2.revision).join(V2.file), b"weights").unwrap();
        if sam3 {
            let dir = snapshot(
                "SceneWorks/sam3-mlx",
                "3ed10b164a755c00b1a4d671dde95719c127e1a7",
            );
            std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
            std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        }
    };

    let without = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&without);
    seed(&settings, false);
    let app = create_app(settings).expect("app creates");
    let (project_id, dataset_id, _, _) = masked_dataset_in(&app).await;
    let preset = zimage_character_preset(&app).await;
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        preset_submission(&preset, &dataset_id, false),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "training_field_error");
    assert_eq!(error["context"]["field"], "subjectMaskLoss");
    assert!(
        error["detail"]
            .as_str()
            .unwrap()
            .contains("SAM3 Person Segmenter"),
        "{error}"
    );

    let with = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&with);
    seed(&settings, true);
    let app = create_app(settings).expect("app creates");
    let (project_id, dataset_id, item_ids, _) = masked_dataset_in(&app).await;
    let preset = zimage_character_preset(&app).await;
    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        preset_submission(&preset, &dataset_id, false),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["subjectMaskPrepass"]["items"]
            .as_array()
            .unwrap()
            .len(),
        item_ids.len()
    );
    assert!(
        job["payload"]["modelManifestEntries"]
            .as_array()
            .is_some_and(|entries| entries
                .iter()
                .any(|entry| entry["id"] == "sam3_person_segment")),
        "{job}"
    );
}
