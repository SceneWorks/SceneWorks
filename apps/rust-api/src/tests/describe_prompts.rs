//! rust-api describing-prompt tests (SW-7085): the default describing instruction route, and the
//! prompt-batch metadata the dataset prompt-batch dialog saves.
use super::support::*;
use sceneworks_core::dataset_batch_describe::MAX_DESCRIBE_PROMPT_CHARS;

const WORKER_ASSET: &str =
    include_str!("../../../../crates/sceneworks-worker/src/image_batch_describe_v1.txt");

/// The asset's `[SYSTEM]` section, read independently of the API's section parser.
fn asset_system_section() -> &'static str {
    WORKER_ASSET
        .split("\n[SYSTEM]\n")
        .nth(1)
        .and_then(|rest| rest.split("\n[OUTPUT]\n").next())
        .expect("asset has a SYSTEM section before OUTPUT")
        .trim()
}

#[tokio::test]
async fn default_describe_prompt_route_serves_the_worker_assets_system_section() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, body) = request(
        app,
        "GET",
        "/api/v1/training/describe-prompt/default",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["prompt"], asset_system_section());
    assert_eq!(body["maxChars"], MAX_DESCRIBE_PROMPT_CHARS);
}

#[tokio::test]
async fn prompt_batch_keeps_the_described_images_pixel_sizes() {
    // The dataset prompt-batch dialog saves each image's exact pixel size beside its prompt; the
    // prompt-batch store keeps fields it does not validate.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/prompt-batches",
        json!({
            "name": "Film prints",
            "prompts": ["[3:4]\nDescription: a woman, {{woman_1}}"],
            "variables": [{ "key": "woman_1", "values": [] }],
            "sourceImages": [{ "itemId": "item_a", "width": 2400, "height": 3200, "ratio": "3:4" }],
            "source": { "kind": "training_dataset", "datasetId": "ds_1", "datasetVersion": 2 },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("id").to_owned();
    let (status, loaded) = request(
        app,
        "GET",
        &format!("/api/v1/prompt-batches/{id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{loaded}");
    assert_eq!(loaded["sourceImages"][0]["width"], 2400);
    assert_eq!(loaded["sourceImages"][0]["height"], 3200);
    assert_eq!(loaded["source"]["datasetId"], "ds_1");
}
