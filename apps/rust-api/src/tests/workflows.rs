//! `POST /api/v1/workflows/inspect` (sc-15950, epic 15945).
//!
//! Three of these are load-bearing: [`inspect_creates_no_asset_no_job_and_no_project_mutation`]
//! (the endpoint's whole reason for existing is that sc-15951 can prefill from a file the user may
//! never import), [`inspect_reports_a_plain_png_as_no_workflow_not_as_a_failure`] (a foreign PNG
//! with no chunk is the COMMON case), and
//! [`inspect_reports_a_catalog_known_but_absent_model_as_installable`] (the actionable middle case,
//! proven through the real model catalog rather than a hand-built lookup).

use super::support::*;

use sceneworks_core::workflow_share::WorkflowShare;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const INSPECT_ROUTE: &str = "/api/v1/workflows/inspect";

fn rgb_fixture() -> image::RgbImage {
    image::RgbImage::from_fn(8, 8, |x, y| image::Rgb([(x * 8) as u8, (y * 8) as u8, 128]))
}

/// An envelope in the shape a real generated image carries. Built through the parser, so it is
/// reduced by exactly the rules a stranger's file is.
fn image_envelope(model: &str, prompt: &str) -> WorkflowShare {
    sceneworks_core::workflow_share::parse_workflow_share_json(&format!(
        r#"{{
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": {{ "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" }},
            "mode": "text_to_image",
            "model": "{model}",
            "prompt": "{prompt}",
            "stylePreset": "cinematic",
            "loras": [{{ "name": "Aurora Portrait v3", "weight": 0.8 }}]
        }}"#
    ))
    .expect("the fixture envelope parses")
}

/// PNG bytes carrying `share` in a `sceneworks:workflow` iTXt chunk (or none at all), written
/// through the ONE writer (sc-15947) rather than a hand-rolled chunk.
fn sceneworks_png(temp_dir: &tempfile::TempDir, share: Option<&WorkflowShare>) -> Vec<u8> {
    let path = temp_dir
        .path()
        .join(format!("fixture-{}.png", uuid::Uuid::new_v4().simple()));
    sceneworks_core::workflow_png::write_workflow_chunk(&rgb_fixture(), &path, share)
        .expect("the fixture PNG writes");
    std::fs::read(&path).expect("the fixture PNG reads")
}

/// POST a multipart body to the inspect route with the given `file` bytes and optional fields.
async fn inspect(
    app: axum::Router,
    filename: &str,
    bytes: &[u8],
    fields: &[(&str, &str)],
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_INSPECT_BOUNDARY";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let (status, _, response) = request_raw(
        app,
        "POST",
        INSPECT_ROUTE,
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&response).expect("json body parses");
    (status, value)
}

/// Every `upload-*` left under `cache/uploads`. Must always be empty: inspect stages the image
/// there and owns removing it on every path.
fn staged_uploads(data_dir: &std::path::Path) -> Vec<String> {
    std::fs::read_dir(data_dir.join("cache").join("uploads"))
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .filter(|name| name.starts_with("upload-"))
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// The happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_returns_the_workflow_and_its_resolution_report() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");
    let share = image_envelope("krea_2_turbo", "a lighthouse in heavy fog");

    let (status, body) = inspect(
        app,
        "shared.png",
        &sceneworks_png(&temp_dir, Some(&share)),
        &[],
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["status"], json!("workflow"));
    assert_eq!(
        body["workflow"]["prompt"],
        json!("a lighthouse in heavy fog")
    );
    assert_eq!(body["workflow"]["model"], json!("krea_2_turbo"));
    // The report is present and classifies every class the envelope carried.
    assert!(body["resolution"].is_object(), "body: {body}");
    assert_eq!(body["resolution"]["model"]["slug"], json!("krea_2_turbo"));
    assert_eq!(
        body["resolution"]["loras"][0]["name"],
        json!("Aurora Portrait v3")
    );
    assert!(body["resolution"]["replayable"].is_boolean());
    // `stylePreset: "cinematic"` is the inert wire default EVERY generated image carries, so it is
    // not a style requirement — otherwise every shared image would report an unresolved style.
    assert_eq!(body["resolution"]["styles"], json!([]));
    assert!(body.get("detail").is_none(), "success carries no detail");
    assert!(
        staged_uploads(&data_dir).is_empty(),
        "the staged temp is removed"
    );
}

#[tokio::test]
async fn inspect_creates_no_asset_no_job_and_no_project_mutation() {
    // The endpoint's reason for existing. A test that only reads the response body would pass with
    // an asset quietly created behind it, so this reads the STORES: the project's asset list, the
    // job queue, and a recursive listing of the project directory.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");

    let (status, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Inspect Purity" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().expect("project path"));

    let before_assets = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await
    .1;
    let before_jobs = request(app.clone(), "GET", "/api/v1/jobs", Value::Null)
        .await
        .1;
    let before_tree = directory_tree(&project_path);
    assert_eq!(before_assets, json!([]), "the project starts empty");

    let share = image_envelope("krea_2_turbo", "a lighthouse");
    // `projectId` is passed on purpose: it widens the LoRA/preset lookups to the project scope,
    // which is the only branch that touches the project store at all.
    let (status, body) = inspect(
        app.clone(),
        "shared.png",
        &sceneworks_png(&temp_dir, Some(&share)),
        &[("projectId", &project_id)],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["status"], json!("workflow"));
    // No asset id anywhere in the response — there is no asset.
    assert!(body.get("id").is_none());
    assert!(body.get("asset").is_none());

    let after_assets = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await
    .1;
    let after_jobs = request(app.clone(), "GET", "/api/v1/jobs", Value::Null)
        .await
        .1;
    assert_eq!(after_assets, json!([]), "inspect created no asset");
    assert_eq!(after_jobs, before_jobs, "inspect created no job");
    assert_eq!(
        directory_tree(&project_path),
        before_tree,
        "inspect wrote nothing into the project"
    );
    assert!(
        staged_uploads(&data_dir).is_empty(),
        "no staged temp survives"
    );
}

/// A sorted recursive listing of `root`, relative, so a project directory can be compared
/// before/after. Content is not hashed — a mutation this endpoint could plausibly make (a sidecar,
/// an index marker, a media file) shows up as a new path.
fn directory_tree(root: &std::path::Path) -> Vec<String> {
    fn walk(base: &std::path::Path, dir: &std::path::Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if let Ok(relative) = path.strip_prefix(base) {
                out.push(relative.to_string_lossy().replace('\\', "/"));
            }
            if path.is_dir() {
                walk(base, &path, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// The "no workflow" branch — distinct, and NOT an error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_reports_a_plain_png_as_no_workflow_not_as_a_failure() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");

    let (status, body) = inspect(app, "foreign.png", &sceneworks_png(&temp_dir, None), &[]).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "a PNG from anywhere else in the world is the COMMON case, not a failure: {body}"
    );
    assert_eq!(body["status"], json!("no_workflow"));
    // Both contract keys are PRESENT and null, so sc-15951 never has to tell absent from null.
    assert_eq!(body["workflow"], Value::Null);
    assert_eq!(body["resolution"], Value::Null);
    assert!(body.get("workflow").is_some() && body.get("resolution").is_some());
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("no SceneWorks workflow")));
    assert!(staged_uploads(&data_dir).is_empty());
}

#[tokio::test]
async fn inspect_uses_a_real_32x32_png_from_the_repo_as_the_foreign_case() {
    // A PNG this project did not write at all (the desktop icon), so the no-workflow answer is not
    // an artifact of the fixture writer.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, body) = inspect(app, "32x32.png", PNG_32X32, &[]).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["status"], json!("no_workflow"));
}

// ---------------------------------------------------------------------------
// Typed errors — never a 500
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_rejects_a_non_image_with_a_typed_400() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");

    let (status, body) = inspect(app, "notes.txt", b"this is not a PNG at all", &[]).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert_eq!(body["code"], json!("workflow_inspect_not_png"));
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("not a PNG")));
    assert!(staged_uploads(&data_dir).is_empty());
}

#[tokio::test]
async fn inspect_rejects_a_png_whose_workflow_this_build_will_not_read_with_a_typed_422() {
    // A VIDEO envelope handed to the image reader (sc-15956's half of the marker key). The chunk is
    // there and well-formed, so this is neither "no workflow" nor "not a PNG" — it is a file whose
    // recipe we refuse to guess at, and the reader's own sentence says which.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");
    let mut share = image_envelope("krea_2_turbo", "a lighthouse");
    share.kind = "video".to_owned();

    let (status, body) = inspect(
        app,
        "video.png",
        &sceneworks_png(&temp_dir, Some(&share)),
        &[],
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "body: {body}");
    assert_eq!(body["code"], json!("workflow_inspect_unreadable"));
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("video")));
    assert!(staged_uploads(&data_dir).is_empty());
}

#[tokio::test]
async fn inspect_rejects_an_oversized_body_with_413_and_leaves_no_temp() {
    // The real cap is `MAX_UPLOAD_BYTES` (2 GiB), which no test can send, so the branch is reached
    // through the same lowered-cap override the LoRA import uses.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");
    let bytes = sceneworks_png(
        &temp_dir,
        Some(&image_envelope("krea_2_turbo", "a lighthouse")),
    );

    TEST_MAX_WORKFLOW_INSPECT_BYTES.with(|cap| cap.set(8));
    let (status, body) = inspect(app, "huge.png", &bytes, &[]).await;
    TEST_MAX_WORKFLOW_INSPECT_BYTES.with(|cap| cap.set(0));

    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "body: {body}");
    assert_eq!(body["detail"], json!("Uploaded image is too large"));
    assert!(staged_uploads(&data_dir).is_empty());
}

#[tokio::test]
async fn inspect_requires_a_file_field() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let boundary = "SCENEWORKS_INSPECT_EMPTY";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"projectId\"\r\n\r\n");
    body.extend_from_slice(b"project-1");
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let (status, _, response) = request_raw(
        app,
        "POST",
        INSPECT_ROUTE,
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let value: Value = serde_json::from_slice(&response).expect("json body parses");
    assert_eq!(value["detail"], json!("Upload file field is required"));
}

#[tokio::test]
async fn inspect_rejects_a_duplicate_file_field_and_cleans_the_first_temp() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");
    let bytes = sceneworks_png(&temp_dir, None);

    let boundary = "SCENEWORKS_INSPECT_DUPE";
    let mut body = Vec::new();
    for _ in 0..2 {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"a.png\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
        body.extend_from_slice(&bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    let (status, _, response) = request_raw(
        app,
        "POST",
        INSPECT_ROUTE,
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let value: Value = serde_json::from_slice(&response).expect("json body parses");
    assert_eq!(value["detail"], json!("Only one file field is allowed"));
    assert!(
        staged_uploads(&data_dir).is_empty(),
        "the first staged temp must not be orphaned"
    );
}

// ---------------------------------------------------------------------------
// The report, through the REAL catalogs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_reports_a_catalog_known_but_absent_model_as_installable() {
    // The actionable middle case, end to end: the seeded manifest knows `fixture_model`, nothing is
    // on disk for it, and the report must point at the Model Manager's own download flow rather
    // than call it missing.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let config_dir = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    single_model_manifest(&config_dir, "fixture_model", "acme/fixture-model");
    let app = create_app(settings).expect("app creates");
    let _env = isolate_hf_cache();
    let share = image_envelope("fixture_model", "a lighthouse");

    let (status, body) = inspect(
        app,
        "shared.png",
        &sceneworks_png(&temp_dir, Some(&share)),
        &[],
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let model = &body["resolution"]["model"];
    assert_eq!(model["state"], json!("installable"), "model: {model}");
    assert_eq!(model["catalogId"], json!("fixture_model"));
    assert_eq!(
        model["install"],
        json!({ "method": "POST", "path": "/api/v1/models/fixture_model/download" }),
        "the middle case must name the existing Model Manager flow"
    );
    assert_eq!(body["resolution"]["replayable"], json!(false));
    // The user-trained LoRA the envelope names resolves to nothing here, and is LISTED.
    assert_eq!(body["resolution"]["loras"][0]["state"], json!("missing"));
    assert!(staged_uploads(&data_dir).is_empty());
}

#[tokio::test]
async fn inspect_reports_a_model_this_install_never_heard_of_as_missing() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let config_dir = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    single_model_manifest(&config_dir, "fixture_model", "acme/fixture-model");
    let app = create_app(settings).expect("app creates");
    let _env = isolate_hf_cache();
    let share = image_envelope("a_model_from_their_install", "a lighthouse");

    let (status, body) = inspect(
        app,
        "shared.png",
        &sceneworks_png(&temp_dir, Some(&share)),
        &[],
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let model = &body["resolution"]["model"];
    assert_eq!(model["state"], json!("missing"));
    assert_eq!(model["slug"], json!("a_model_from_their_install"));
    assert!(model.get("catalogId").is_none(), "nothing matched");
    assert!(
        model.get("install").is_none(),
        "there is nothing to offer to install"
    );
    // Never substituted: the one model this install DOES know is not reached for.
    assert!(!model["detail"]
        .as_str()
        .expect("detail")
        .contains("fixture_model"));
}

#[tokio::test]
async fn inspect_reports_an_input_image_the_recipe_needs_by_shape() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let share = sceneworks_core::workflow_share::parse_workflow_share_json(
        r#"{
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" },
            "mode": "edit_image",
            "model": "krea_2_turbo",
            "prompt": "make it night",
            "inputs": [{ "kind": "source", "count": 1 }, { "kind": "reference", "count": 2 }]
        }"#,
    )
    .expect("the envelope parses");

    let (status, body) = inspect(
        app,
        "edit.png",
        &sceneworks_png(&temp_dir, Some(&share)),
        &[],
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let inputs = body["resolution"]["inputs"]
        .as_array()
        .expect("inputs is an array");
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs[0]["state"], json!("userSupplied"));
    assert_eq!(inputs[0]["detail"], json!("Needs a source image."));
    assert_eq!(inputs[1]["detail"], json!("Needs 2 reference images."));
    assert_eq!(body["resolution"]["replayable"], json!(false));
}

#[tokio::test]
async fn inspect_surfaces_a_dropped_collection_so_replay_can_be_withheld() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let share = sceneworks_core::workflow_share::parse_workflow_share_json(
        r#"{
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "krea_2_turbo",
            "prompt": "a lighthouse",
            "omitted": ["loras"]
        }"#,
    )
    .expect("the envelope parses");

    let (status, body) = inspect(
        app,
        "dropped.png",
        &sceneworks_png(&temp_dir, Some(&share)),
        &[],
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["resolution"]["loras"], json!([]));
    assert_eq!(body["resolution"]["omitted"][0]["field"], json!("loras"));
    assert!(body["resolution"]["omitted"][0]["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("NOT a LoRA-free recipe")));
    assert_eq!(
        body["resolution"]["replayable"],
        json!(false),
        "an omitted collection withholds one-click replay"
    );
}

#[tokio::test]
async fn inspect_requires_a_token_when_one_is_configured() {
    // Same posture as the upload route: no auth change, so the shared `access_control` middleware
    // guards it like every other `/api/v1` route.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");
    let (status, _) = inspect(app, "shared.png", &sceneworks_png(&temp_dir, None), &[]).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}
