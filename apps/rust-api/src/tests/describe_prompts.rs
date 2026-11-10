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

const PRESETS: &str = "/api/v1/training/describe-prompt/presets";

#[tokio::test]
async fn describe_prompt_presets_persist_and_restore_the_exact_text() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    // Leading/trailing whitespace and inner newlines are kept as typed.
    let text = "  Describe only the clothing.\n\nNever the faces.  \n";
    let (status, created) = request(
        app.clone(),
        "POST",
        PRESETS,
        json!({ "name": "Clothing only", "prompt": text, "builtin": true }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["id"], "clothing_only");
    assert_eq!(created["prompt"], text);
    assert_eq!(created["scope"], "global");
    assert!(
        created.get("builtin").is_none(),
        "a client cannot mark a stored preset built-in"
    );

    // A fresh app over the same config dir reads it back from disk.
    let app = create_app(test_settings(&temp_dir)).expect("app re-creates");
    let (status, list) = request(app.clone(), "GET", PRESETS, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{list}");
    let list = list.as_array().expect("preset list");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0]["id"], "default");
    assert_eq!(list[0]["builtin"], true);
    assert_eq!(list[0]["prompt"], asset_system_section());
    assert_eq!(list[1]["id"], "clothing_only");
    assert_eq!(list[1]["prompt"], text);

    let (status, updated) = request(
        app.clone(),
        "PATCH",
        &format!("{PRESETS}/clothing_only"),
        json!({ "prompt": "Describe only the shoes." }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{updated}");
    assert_eq!(updated["prompt"], "Describe only the shoes.");
    assert_eq!(updated["name"], "Clothing only");

    let (status, _) = request(
        app.clone(),
        "DELETE",
        &format!("{PRESETS}/clothing_only"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, list) = request(app, "GET", PRESETS, Value::Null).await;
    assert_eq!(list.as_array().expect("preset list").len(), 1);
}

#[tokio::test]
async fn a_deleted_preset_name_can_be_saved_again() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let first = json!({ "name": "Clothing only", "prompt": "First text." });
    let (status, _) = request(app.clone(), "POST", PRESETS, first.clone()).await;
    assert_eq!(status, StatusCode::CREATED);
    // A live preset still owns its name.
    let (status, error) = request(app.clone(), "POST", PRESETS, first).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");

    let (status, _) = request(
        app.clone(),
        "DELETE",
        &format!("{PRESETS}/clothing_only"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, created) = request(
        app.clone(),
        "POST",
        PRESETS,
        json!({ "name": "Clothing only", "prompt": "Second text." }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["id"], "clothing_only");
    assert_eq!(created["prompt"], "Second text.");
    assert!(created.get("archived").is_none(), "{created}");

    let (_, list) = request(app, "GET", PRESETS, Value::Null).await;
    let list = list.as_array().expect("preset list");
    assert_eq!(list.len(), 2, "{list:?}");
    assert_eq!(list[1]["prompt"], "Second text.");
}

#[tokio::test]
async fn the_builtin_default_preset_cannot_be_overwritten_or_deleted() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let attempts = [
        (
            "POST",
            PRESETS.to_owned(),
            json!({ "name": "Default", "prompt": "x" }),
        ),
        (
            "POST",
            PRESETS.to_owned(),
            json!({ "id": "default", "name": "Mine", "prompt": "x" }),
        ),
        (
            "PATCH",
            format!("{PRESETS}/default"),
            json!({ "prompt": "x" }),
        ),
        ("DELETE", format!("{PRESETS}/default"), Value::Null),
    ];
    for (method, uri, body) in attempts {
        let (status, error) = request(app.clone(), method, &uri, body).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{method} {uri}: {error}");
    }
    let (_, list) = request(app, "GET", PRESETS, Value::Null).await;
    let list = list.as_array().expect("preset list");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["prompt"], asset_system_section());
}

#[tokio::test]
async fn describe_prompt_presets_refuse_a_blank_or_oversized_prompt() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    for body in [
        json!({ "name": "Blank", "prompt": "   " }),
        json!({ "name": "Missing" }),
        json!({ "name": "Huge", "prompt": "x".repeat(MAX_DESCRIBE_PROMPT_CHARS + 1) }),
        json!({ "prompt": "No name" }),
    ] {
        let (status, error) = request(app.clone(), "POST", PRESETS, body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {error}");
    }
}
