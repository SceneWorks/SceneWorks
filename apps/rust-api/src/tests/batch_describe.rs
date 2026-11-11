//! rust-api dataset batch describe tests (SW-7082, SW-7083): job enqueue, item subset in dataset
//! order, default/custom prompt carriage, JoyCaption as the second describer, and the 400s for an
//! unsupported model, an unknown item and an oversized prompt.
use super::support::*;
use sceneworks_core::dataset_batch_describe::{
    JOYCAPTION_REPO, MAX_DESCRIBE_PROMPT_CHARS, MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS,
    VISION_CAPTIONER_REPO,
};

fn png_rgb(w: u32, h: u32, shade: u8) -> Vec<u8> {
    let mut out = Vec::new();
    image::RgbImage::from_pixel(w, h, image::Rgb([shade, 64, 128]))
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("encode image");
    out
}

/// A project with one dataset of three images. Returns (app, project id, dataset id, item ids,
/// item relative paths).
async fn dataset_fixture(
    temp_dir: &tempfile::TempDir,
) -> (axum::Router, String, String, Vec<String>, Vec<String>) {
    let app = create_app(test_settings(temp_dir)).expect("app creates");
    let (_, project) = create_project(app.clone(), &"Batch describe project").await;
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
        json!({ "name": "Describe set", "items": items }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{dataset}");
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let field = |key: &str| -> Vec<String> {
        dataset["items"]
            .as_array()
            .expect("items")
            .iter()
            .map(|item| item[key].as_str().expect("item field").to_owned())
            .collect()
    };
    (app, project_id, dataset_id, field("id"), field("path"))
}

#[tokio::test]
async fn batch_describe_job_enqueues_every_image_with_the_vision_captioner_by_default() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, paths) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");

    let (status, job) = request(app.clone(), "POST", &route, json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "dataset_batch_describe");
    assert_eq!(job["title"], "Describe Dataset Images — Describe set");
    let payload = &job["payload"];
    assert_eq!(payload["kind"], "dataset_batch_describe");
    assert_eq!(payload["model"], VISION_CAPTIONER_REPO);
    assert_eq!(payload["datasetId"], dataset_id.as_str());
    assert_eq!(payload["datasetName"], "Describe set");
    assert!(
        payload.get("prompt").is_none(),
        "no prompt means the worker's default instruction"
    );
    let job_items = payload["items"].as_array().expect("job items");
    assert_eq!(job_items.len(), 3);
    for ((job_item, item_id), path) in job_items.iter().zip(&item_ids).zip(&paths) {
        assert_eq!(job_item["itemId"], item_id.as_str());
        assert_eq!(job_item["relativePath"], path.as_str());
        assert!(job_item["imagePath"]
            .as_str()
            .unwrap()
            .ends_with(path.as_str()));
    }
}

#[tokio::test]
async fn batch_describe_subset_keeps_dataset_order_and_carries_the_custom_prompt() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");

    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({
            "model": VISION_CAPTIONER_REPO,
            "prompt": "  Describe only the clothing.  ",
            // Asked for out of order: the job still describes them in dataset order.
            "itemIds": [item_ids[2], item_ids[0]],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["prompt"], "Describe only the clothing.");
    let ids: Vec<_> = job["payload"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["itemId"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(ids, vec![item_ids[0].clone(), item_ids[2].clone()]);

    // A blank prompt is the default instruction, not an empty one.
    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "prompt": "   ", "itemIds": [item_ids[1]] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert!(job["payload"].get("prompt").is_none());
    assert_eq!(job["payload"]["items"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn batch_describe_refuses_other_models_unknown_items_and_oversized_prompts() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, _, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");
    for (body, field, expect) in [
        (
            json!({ "model": "someone/other-vlm" }),
            "model",
            "Unsupported describe model",
        ),
        (
            json!({ "model": "fancyfeast/llama-joycaption-alpha-two-hf-llava" }),
            "model",
            "Unsupported describe model",
        ),
        (
            json!({ "itemIds": ["not_an_item"] }),
            "itemIds",
            "Unknown dataset item id(s): not_an_item.",
        ),
        (
            json!({ "itemIds": [] }),
            "itemIds",
            "at least one dataset item",
        ),
        (
            json!({ "prompt": "x".repeat(MAX_DESCRIBE_PROMPT_CHARS + 1) }),
            "prompt",
            "at most",
        ),
        (
            json!({
                "model": JOYCAPTION_REPO,
                "prompt": "x".repeat(MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS + 1),
            }),
            "prompt",
            "at most 3000 characters",
        ),
    ] {
        let (status, error) = request(app.clone(), "POST", &route, body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {error}");
        assert_eq!(error["context"]["field"], field, "{error}");
        assert!(
            error["detail"]
                .as_str()
                .unwrap_or_default()
                .contains(expect)
                || error["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains(expect)
                || error.to_string().contains(expect),
            "{error}"
        );
    }
}

#[tokio::test]
async fn batch_describe_accepts_joycaption_with_its_own_prompt_limit() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");

    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "model": format!("  {JOYCAPTION_REPO} "), "itemIds": [item_ids[1]] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "dataset_batch_describe");
    assert_eq!(job["payload"]["model"], JOYCAPTION_REPO);
    assert!(job["payload"].get("prompt").is_none());
    assert_eq!(job["payload"]["items"].as_array().unwrap().len(), 1);

    // An instruction exactly at JoyCaption's limit is accepted.
    let at_limit = "x".repeat(MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS);
    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "model": JOYCAPTION_REPO, "prompt": at_limit }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["prompt"].as_str().unwrap().chars().count(),
        MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS
    );
}

/// SW-7082: the describe worker appends each finished image to `/result-items` (its progress
/// result carries only counts), and the dialog pages them back in dataset order. Only the owning
/// worker may append, and a canceled job still serves the images it finished.
#[tokio::test]
async fn batch_describe_items_are_appended_by_the_owner_and_paged_back_in_order() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");
    let (status, job) = request(app.clone(), "POST", &route, json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let job_id = job["id"].as_str().expect("job id").to_owned();
    const WORKER: &str = "test-describe-worker";
    claim_job_as_worker(&app, &job_id, WORKER, &["gpu", "prompt_refine"]).await;

    let items_route = format!("/api/v1/jobs/{job_id}/result-items");
    let item = |index: usize| {
        json!({
            "position": index,
            "item": { "itemId": item_ids[index], "description": format!("image {index}") },
        })
    };
    // Appended out of order; read back by position.
    for index in [1, 0] {
        let (status, body) = request(
            app.clone(),
            "POST",
            &items_route,
            json!({ "workerId": WORKER, "items": [item(index)] }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let (status, body) = request(
        app.clone(),
        "POST",
        &items_route,
        json!({ "workerId": "someone-else", "items": [item(2)] }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let (status, body) = request(
        app.clone(),
        "POST",
        &items_route,
        json!({ "workerId": WORKER, "items": [{ "position": 2, "item": { "description": "x" } }] }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "an item needs an id: {body}"
    );

    let (status, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "workerId": WORKER, "status": "canceled", "stage": "canceled", "progress": 1.0,
            "message": "canceled", "result": { "processedCount": 2, "itemCount": 3 },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, page) = request(
        app.clone(),
        "GET",
        &format!("{items_route}?offset=0&limit=1"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{page}");
    assert_eq!(page["total"], 2);
    assert_eq!(page["nextOffset"], 1);
    assert_eq!(page["items"][0]["itemId"], item_ids[0].as_str());
    let (_, page) = request(
        app.clone(),
        "GET",
        &format!("{items_route}?offset=1"),
        Value::Null,
    )
    .await;
    assert_eq!(page["items"][0]["itemId"], item_ids[1].as_str());
    assert_eq!(page["items"][0]["description"], "image 1");
    assert_eq!(page["nextOffset"], Value::Null);

    let (status, _) = request(
        app.clone(),
        "GET",
        "/api/v1/jobs/missing-job/result-items",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn batch_describe_jobs_list_the_datasets_runs_newest_first() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");

    let (status, empty) = request(app.clone(), "GET", &route, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{empty}");
    assert_eq!(empty, json!([]));

    let (status, first) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "itemIds": [item_ids[0]] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{first}");
    let (status, second) = request(app.clone(), "POST", &route, json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{second}");
    // A canceled run still lists: its stored descriptions are the user's work.
    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/cancel", first["id"].as_str().unwrap()),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{canceled}");

    let (status, jobs) = request(app.clone(), "GET", &route, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{jobs}");
    let ids: Vec<_> = jobs
        .as_array()
        .expect("job list")
        .iter()
        .map(|job| job["id"].clone())
        .collect();
    assert_eq!(ids, vec![second["id"].clone(), first["id"].clone()]);

    // Another dataset's history is separate.
    let other = format!(
        "/api/v1/projects/{project_id}/training/datasets/other-dataset/batch-describe-jobs"
    );
    let (status, none) = request(app.clone(), "GET", &other, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{none}");
    assert_eq!(none, json!([]));
}
