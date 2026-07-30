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
    assert!(body["resolution"]["runnable"].is_boolean());
    assert!(body["resolution"]["inputImagesRequired"].is_u64());
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
    // The real cap is `MAX_WORKFLOW_INSPECT_BYTES` (512 MiB), which no test wants to send, so the
    // branch is reached through the same lowered-cap override the LoRA import uses.
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
    assert_eq!(body["code"], json!("workflow_inspect_too_large"));
    assert!(staged_uploads(&data_dir).is_empty());
}

#[tokio::test]
async fn the_typed_413_wins_at_the_real_cap_boundary_not_just_far_under_it() {
    // The test above lowers the field cap to 8 bytes — far under the ROUTER limit, so it proves
    // nothing about the boundary that actually matters. With `DefaultBodyLimit` equal to the field
    // cap, a body AT the cap plus its multipart framing (boundaries, part headers, the trailing
    // CRLF — ~200 bytes here) exceeds the router limit, axum rejects the body first, and
    // `field.chunk()` surfaces it as a plain 400 with no typed code at all.
    //
    // `max_inspect_multipart_body_bytes` derives the router limit from the live per-field cap, so
    // lowering the cap BEFORE `create_app` scales the router limit with it and the same
    // cap-plus-framing relationship is exercised at a sendable size.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let png = sceneworks_png(
        &temp_dir,
        Some(&image_envelope("krea_2_turbo", "a lighthouse")),
    );
    // The cap is set to the PNG's exact length, so the file field lands precisely ON it and the
    // framing is the only thing that could push the request over a router limit.
    let cap = png.len();

    TEST_MAX_WORKFLOW_INSPECT_BYTES.with(|limit| limit.set(cap));
    let app = create_app(settings).expect("app creates");
    let (at_cap, at_cap_body) = inspect(app.clone(), "exact.png", &png, &[]).await;
    // One byte over: the field cap is the thing that must reject it, with the typed 413.
    let mut over = png.clone();
    over.push(0);
    let (over_cap, over_cap_body) = inspect(app, "over.png", &over, &[]).await;
    TEST_MAX_WORKFLOW_INSPECT_BYTES.with(|limit| limit.set(0));

    assert_eq!(
        at_cap,
        StatusCode::OK,
        "a body exactly at the cap must reach the handler, framing and all: {at_cap_body}"
    );
    assert_eq!(at_cap_body["status"], json!("workflow"));
    assert_eq!(
        over_cap,
        StatusCode::PAYLOAD_TOO_LARGE,
        "one byte over the cap is the typed 413, not axum's plain 400: {over_cap_body}"
    );
    assert_eq!(over_cap_body["code"], json!("workflow_inspect_too_large"));
    assert!(staged_uploads(&data_dir).is_empty());
}

#[tokio::test]
async fn a_malformed_multipart_request_still_carries_a_machine_readable_code() {
    // The extractor rejects these before the handler body runs. Left to axum the response is a
    // PLAIN-TEXT body — so sc-15951, which branches on `code`, would be handed something it cannot
    // parse as JSON at all. Every shape that fails at the extractor is covered here.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    for (label, content_type, body) in [
        (
            "an empty body",
            "multipart/form-data; boundary=B",
            Vec::new(),
        ),
        (
            "a JSON body",
            "application/json",
            br#"{"file":"nope"}"#.to_vec(),
        ),
        (
            "a boundary-less content type",
            "multipart/form-data",
            b"--B\r\n\r\n--B--\r\n".to_vec(),
        ),
    ] {
        let (status, _, response) = request_raw(
            app.clone(),
            "POST",
            INSPECT_ROUTE,
            body,
            &[("content-type", content_type)],
        )
        .await;
        assert!(
            status.is_client_error(),
            "{label} must be a typed 4xx, got {status}"
        );
        let value: Value = serde_json::from_slice(&response)
            .unwrap_or_else(|error| panic!("{label} must render a JSON envelope: {error}"));
        assert_eq!(
            value["code"],
            json!("workflow_inspect_bad_multipart"),
            "{label}: {value}"
        );
        assert!(
            value["detail"].as_str().is_some_and(|d| !d.is_empty()),
            "{label} needs a sentence: {value}"
        );
    }
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
    assert_eq!(value["code"], json!("workflow_inspect_bad_multipart"));
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
    assert_eq!(value["code"], json!("workflow_inspect_bad_multipart"));
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
    assert_eq!(body["resolution"]["runnable"], json!(false));
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
    // The two answers, separately: this install cannot RUN it (the model is unknown here) and it
    // would additionally need three images from the user.
    assert_eq!(body["resolution"]["runnable"], json!(false));
    assert_eq!(body["resolution"]["inputImagesRequired"], json!(3));
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
        body["resolution"]["runnable"],
        json!(false),
        "an omitted collection withholds one-click replay"
    );
}

#[tokio::test]
async fn inspect_never_matches_a_scrubbed_model_to_a_catalog_row_that_has_no_id() {
    // `workflow_share` scrubs a path-shaped `model` to `""` (it is unshareable), and a manifest may
    // omit `id` entirely — `load_manifest_entries` validates nothing and `apply_model_catalog_entry`
    // defaults it. Reduced to `id: ""`, the row matched an envelope that named NO model, and the
    // report said a model the file never mentioned was on this machine.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let config_dir = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    write_empty_sibling_manifests(&config_dir);
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [{
            "name": "Some Other Model", "type": "image", "family": "test",
            "downloads": [{ "provider": "huggingface", "repo": "acme/some-other-model" }]
        }] }"#,
    )
    .expect("id-less models manifest writes");
    let app = create_app(settings).expect("app creates");
    let _env = isolate_hf_cache();

    let share = sceneworks_core::workflow_share::parse_workflow_share_json(
        r#"{
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "C:\\models\\theirs\\weights.safetensors",
            "prompt": "a lighthouse"
        }"#,
    )
    .expect("the envelope parses");
    assert_eq!(share.model, "", "the parser scrubbed the path-shaped model");

    let (status, body) = inspect(
        app,
        "scrubbed.png",
        &sceneworks_png(&temp_dir, Some(&share)),
        &[],
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let model = &body["resolution"]["model"];
    assert_eq!(model["state"], json!("missing"), "model: {model}");
    assert!(
        model.get("catalogId").is_none(),
        "a blank slug matched a row: {model}"
    );
    assert!(model.get("install").is_none(), "model: {model}");
    assert!(
        !model["detail"]
            .as_str()
            .expect("detail")
            .contains("Some Other Model"),
        "the sentence names a model the envelope never did: {model}"
    );
    assert_eq!(body["resolution"]["runnable"], json!(false));
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

// ---------------------------------------------------------------------------
// `GET /api/v1/projects/:project_id/assets/:asset_id/workflow` (sc-15952)
// ---------------------------------------------------------------------------

/// Create a project, import `bytes` as an asset, and answer `(project_id, asset_id)`.
async fn project_with_imported_image(
    app: axum::Router,
    name: &str,
    filename: &str,
    bytes: &[u8],
) -> (String, String) {
    let (status, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": name }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let (status, asset) = request_multipart_upload(
        app,
        &format!("/api/v1/projects/{project_id}/assets"),
        filename,
        "image/png",
        bytes,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "asset: {asset}");
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    (project_id, asset_id)
}

fn asset_workflow_route(project_id: &str, asset_id: &str) -> String {
    format!("/api/v1/projects/{project_id}/assets/{asset_id}/workflow")
}

#[tokio::test]
async fn the_asset_route_reports_the_envelope_import_recorded_and_a_fresh_report() {
    // The gap sc-15950 left: `build_resolution_report` had exactly one production caller and
    // nothing read `extra.importedWorkflow` back into a report, so an imported asset carried a
    // recipe nothing could say anything about.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");
    let _env = isolate_hf_cache();
    let share = image_envelope("krea_2_turbo", "a lighthouse in heavy fog");
    let (project_id, asset_id) = project_with_imported_image(
        app.clone(),
        "Imported Workflow",
        "shared.png",
        &sceneworks_png(&temp_dir, Some(&share)),
    )
    .await;

    let (status, body) = request(
        app,
        "GET",
        &asset_workflow_route(&project_id, &asset_id),
        Value::Null,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["status"], json!("workflow"));
    assert_eq!(
        body["workflow"]["prompt"],
        json!("a lighthouse in heavy fog")
    );
    assert_eq!(body["workflow"]["model"], json!("krea_2_turbo"));
    // The report is the point of the route — the envelope was already on the asset.
    assert_eq!(body["resolution"]["model"]["slug"], json!("krea_2_turbo"));
    assert_eq!(
        body["resolution"]["loras"][0]["name"],
        json!("Aurora Portrait v3")
    );
    assert!(body["resolution"]["runnable"].is_boolean());
    assert!(body["resolution"]["inputImagesRequired"].is_u64());
}

#[tokio::test]
async fn the_asset_route_reports_a_catalog_known_but_absent_model_as_installable() {
    // The actionable middle case, through the REAL model catalog: the row is in
    // `builtin.models.jsonc` and nothing is on disk, so the report must offer the Model Manager's
    // own download rather than calling it missing. That action is what sc-15952's install button
    // routes into, and what makes re-resolving after it meaningful.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let config_dir = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    single_model_manifest(&config_dir, "fixture_model", "acme/fixture-model");
    let app = create_app(settings).expect("app creates");
    let _env = isolate_hf_cache();
    let share = image_envelope("fixture_model", "a lighthouse");
    let (project_id, asset_id) = project_with_imported_image(
        app.clone(),
        "Installable Model",
        "shared.png",
        &sceneworks_png(&temp_dir, Some(&share)),
    )
    .await;

    let (status, body) = request(
        app,
        "GET",
        &asset_workflow_route(&project_id, &asset_id),
        Value::Null,
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
    assert_eq!(body["resolution"]["runnable"], json!(false));
}

#[tokio::test]
async fn the_asset_route_reports_an_asset_with_no_chunk_as_no_workflow() {
    // Every image in the world that did not come from SceneWorks. A 200 with its own status, the
    // same shape the inspect route uses, so the web has one branch and no error band.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");
    let (project_id, asset_id) = project_with_imported_image(
        app.clone(),
        "Plain Image",
        "foreign.png",
        &sceneworks_png(&temp_dir, None),
    )
    .await;

    let (status, body) = request(
        app,
        "GET",
        &asset_workflow_route(&project_id, &asset_id),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["status"], json!("no_workflow"));
    assert_eq!(body["workflow"], Value::Null);
    assert_eq!(body["resolution"], Value::Null);
    assert!(body["detail"].is_string(), "body: {body}");
}

#[tokio::test]
async fn the_asset_route_mutates_nothing() {
    // A GET that writes is a GET that turns a panel refresh into a project edit — and the
    // re-resolution after an install calls this again on every catalog change. Read the STORES,
    // not just the body.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");
    let _env = isolate_hf_cache();
    let share = image_envelope("krea_2_turbo", "a lighthouse");
    let (project_id, asset_id) = project_with_imported_image(
        app.clone(),
        "Read Only",
        "shared.png",
        &sceneworks_png(&temp_dir, Some(&share)),
    )
    .await;
    let (_, project) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}"),
        Value::Null,
    )
    .await;
    let project_path = std::path::PathBuf::from(project["path"].as_str().expect("project path"));
    let before_tree = directory_tree(&project_path);
    let before_jobs = request(app.clone(), "GET", "/api/v1/jobs", Value::Null)
        .await
        .1;

    for _ in 0..3 {
        let (status, _) = request(
            app.clone(),
            "GET",
            &asset_workflow_route(&project_id, &asset_id),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    assert_eq!(
        directory_tree(&project_path),
        before_tree,
        "reading the workflow wrote into the project"
    );
    assert_eq!(
        request(app, "GET", "/api/v1/jobs", Value::Null).await.1,
        before_jobs,
        "reading the workflow queued a job"
    );
}

#[tokio::test]
async fn the_asset_route_404s_for_an_asset_that_does_not_exist() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");
    let (status, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Missing Asset" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let project_id = project["id"].as_str().expect("project id");
    let (status, _) = request(
        app,
        "GET",
        &asset_workflow_route(project_id, "does-not-exist"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
