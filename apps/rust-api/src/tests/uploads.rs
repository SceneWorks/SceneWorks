//! rust-api uploads tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[test]
fn stale_lora_upload_sweep_removes_only_upload_dirs_before_cutoff() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let upload_root = temp_dir.path().join("data/cache/lora-uploads");
    let expired = upload_root.join("upload-expired");
    let fresh = upload_root.join("upload-fresh");
    let unrelated = upload_root.join("keep-me");
    std::fs::create_dir_all(&expired).expect("expired dir creates");
    std::fs::create_dir_all(&fresh).expect("fresh dir creates");
    std::fs::create_dir_all(&unrelated).expect("unrelated dir creates");

    let removed = sweep_stale_lora_uploads_before(
        &temp_dir.path().join("data"),
        SystemTime::now() + Duration::from_secs(1),
    )
    .expect("stale uploads sweep");

    assert_eq!(removed, 2);
    assert!(!expired.exists());
    assert!(!fresh.exists());
    assert!(unrelated.exists());
}

#[test]
fn stale_asset_upload_sweep_removes_only_upload_tmp_before_cutoff() {
    // sc-4204 (F-API-6): cache/uploads now has a startup-sweep backstop.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let upload_root = temp_dir.path().join("data/cache/uploads");
    std::fs::create_dir_all(&upload_root).expect("upload root creates");
    let expired = upload_root.join("upload-expired.tmp");
    let fresh = upload_root.join("upload-fresh.tmp");
    let unrelated = upload_root.join("keep-me.txt");
    std::fs::write(&expired, b"x").expect("expired writes");
    std::fs::write(&fresh, b"y").expect("fresh writes");
    std::fs::write(&unrelated, b"z").expect("unrelated writes");

    let removed = sweep_stale_asset_uploads_before(
        &temp_dir.path().join("data"),
        SystemTime::now() + Duration::from_secs(1),
    )
    .expect("asset uploads sweep");

    assert_eq!(removed, 2);
    assert!(!expired.exists());
    assert!(!fresh.exists());
    assert!(unrelated.exists(), "non upload-* files are left alone");
}

#[tokio::test]
async fn import_asset_removes_staged_temp_file_when_a_later_field_errors() {
    // sc-4204 (F-API-6): the `file` field is staged to cache/uploads before a later
    // field is parsed; an invalid `provenance` JSON must not leave an orphan tmp.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");

    let boundary = "SCENEWORKS_IMPORT_BOUNDARY";
    let mut body = Vec::new();
    // `file` first so it is staged, then an invalid `provenance` that errors.
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body.extend_from_slice(b"\x89PNG\r\n\x1a\n payload bytes");
    body.extend_from_slice(format!("\r\n--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"provenance\"\r\n\r\n");
    body.extend_from_slice(b"{not valid json");
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let (status, _, response) = request_raw(
        app,
        "POST",
        "/api/v1/projects/project-1/assets",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let value: Value = serde_json::from_slice(&response).expect("json body parses");
    assert!(value["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("Invalid provenance JSON")));

    // No orphaned temp file should remain under cache/uploads.
    let upload_root = data_dir.join("cache").join("uploads");
    let leaked: Vec<_> = std::fs::read_dir(&upload_root)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("upload-"))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        leaked.is_empty(),
        "staged upload temp file leaked on error: {leaked:?}"
    );
}

#[tokio::test]
async fn import_asset_rejects_a_provenance_that_is_not_an_object() {
    // sc-15949 review: any non-blank JSON text was accepted here, and the store then made the
    // sidecar's `extra` slot that value verbatim. Once sc-15949 gave `extra` a second writer, that
    // became a silent-loss path — `provenance=42` (or `null`, or `[1,2]`) discarded a workflow the
    // import had already read out of the PNG. `extra` is an object in the contract, so anything
    // else is refused at the boundary, and the staged temp is still cleaned up.
    for provenance in ["42", "\"editor\"", "[1,2]", "null"] {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let data_dir = settings.data_dir.clone();
        let app = create_app(settings).expect("app creates");

        let boundary = "SCENEWORKS_SCALAR_PROVENANCE_BOUNDARY";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
        body.extend_from_slice(b"\x89PNG\r\n\x1a\n payload bytes");
        body.extend_from_slice(format!("\r\n--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"provenance\"\r\n\r\n");
        body.extend_from_slice(provenance.as_bytes());
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let (status, _, response) = request_raw(
            app,
            "POST",
            "/api/v1/projects/project-1/assets",
            body,
            &[(
                "content-type",
                &format!("multipart/form-data; boundary={boundary}"),
            )],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "provenance={provenance} must be refused"
        );
        let value: Value = serde_json::from_slice(&response).expect("json body parses");
        // The API's own sentence, matched exactly. `ProjectStore::import_asset` refuses the same
        // shape with a different one ("Upload provenance must be a JSON object"), and a loose
        // `contains` would match either — so this test would pass with the boundary check removed
        // and only the store's belt holding.
        assert!(
            value["detail"].as_str().is_some_and(
                |detail| detail == "Invalid provenance JSON: provenance must be a JSON object"
            ),
            "provenance={provenance} produced {:?}",
            value["detail"]
        );

        let upload_root = data_dir.join("cache").join("uploads");
        let leaked: Vec<_> = std::fs::read_dir(&upload_root)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter(|entry| entry.file_name().to_string_lossy().starts_with("upload-"))
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            leaked.is_empty(),
            "staged upload temp file leaked on provenance={provenance}: {leaked:?}"
        );
    }
}

#[tokio::test]
async fn import_asset_rejects_duplicate_file_field_and_cleans_first_temp() {
    // sc-8883 (F-081): a second `file` field must be rejected (400), and the first
    // already-staged temp must be cleaned up rather than orphaned until the 24h sweep.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");

    let boundary = "SCENEWORKS_DUP_FILE_BOUNDARY";
    let mut body = Vec::new();
    for name in ["file", "file"] {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"x.png\"\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
        body.extend_from_slice(b"\x89PNG\r\n\x1a\n payload bytes");
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let (status, _, response) = request_raw(
        app,
        "POST",
        "/api/v1/projects/project-1/assets",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let value: Value = serde_json::from_slice(&response).expect("json body parses");
    assert!(value["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("Only one file field is allowed")));

    // The first staged temp must not leak under cache/uploads.
    let upload_root = data_dir.join("cache").join("uploads");
    let leaked: Vec<_> = std::fs::read_dir(&upload_root)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("upload-"))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        leaked.is_empty(),
        "first staged temp file leaked on duplicate file field: {leaked:?}"
    );
}

#[test]
fn shared_sweep_removes_stale_files_and_dirs_leaves_fresh_and_unrelated() {
    // sc-8885 (F-083): the unified sweeper handles both loose files and per-upload
    // directories, and only touches `upload-*` entries at/older than the cutoff.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let root = temp_dir.path().join("data/cache/pose-uploads");
    std::fs::create_dir_all(&root).expect("root creates");
    let stale_file = root.join("upload-old.tmp");
    let stale_dir = root.join("upload-old-dir");
    let unrelated = root.join("keep-me.txt");
    std::fs::write(&stale_file, b"x").expect("stale file writes");
    std::fs::create_dir_all(&stale_dir).expect("stale dir creates");
    std::fs::write(stale_dir.join("inner"), b"y").expect("inner writes");
    std::fs::write(&unrelated, b"z").expect("unrelated writes");

    // Cutoff in the future -> everything upload-* qualifies as stale.
    let removed = sweep_stale_uploads(
        &temp_dir.path().join("data"),
        "pose-uploads",
        SystemTime::now() + Duration::from_secs(1),
    )
    .expect("shared sweep");
    assert_eq!(removed, 2, "both the file and the directory are removed");
    assert!(!stale_file.exists());
    assert!(!stale_dir.exists());
    assert!(unrelated.exists(), "non upload-* entries are left alone");

    // A missing root is not an error.
    assert_eq!(
        sweep_stale_uploads(
            &temp_dir.path().join("data"),
            "does-not-exist",
            SystemTime::now(),
        )
        .expect("missing root is ok"),
        0
    );
}

#[cfg(unix)]
#[test]
fn shared_sweep_is_best_effort_when_one_entry_cannot_be_removed() {
    // sc-8885 (F-083): per-entry reclamation is best-effort — a single unremovable
    // stale entry (here a directory whose contents can't be deleted) is logged and
    // skipped so every other stale entry in the sweep is still reclaimed. The original
    // per-area sweepers used `let _ =` and continued; the unified helper must too.
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let root = temp_dir.path().join("data/cache/pose-uploads");
    std::fs::create_dir_all(&root).expect("root creates");

    // Two removable stale entries flanking one that cannot be removed.
    let ok_file_a = root.join("upload-a.tmp");
    let ok_file_b = root.join("upload-b.tmp");
    std::fs::write(&ok_file_a, b"a").expect("a writes");
    std::fs::write(&ok_file_b, b"b").expect("b writes");

    // An unremovable stale directory: it holds an inner file, then we strip write
    // permission on the directory itself so `remove_dir_all` fails (deleting the inner
    // entry requires write on its parent) without touching the siblings' removability.
    let locked_dir = root.join("upload-locked-dir");
    std::fs::create_dir_all(&locked_dir).expect("locked dir creates");
    std::fs::write(locked_dir.join("inner"), b"x").expect("inner writes");
    std::fs::set_permissions(&locked_dir, std::fs::Permissions::from_mode(0o500))
        .expect("chmod locked dir");

    // Cutoff in the future -> every upload-* entry qualifies as stale.
    let removed = sweep_stale_uploads(
        &temp_dir.path().join("data"),
        "pose-uploads",
        SystemTime::now() + Duration::from_secs(1),
    )
    .expect("sweep does not abort on a single unremovable entry");

    // Restore permissions so the tempdir can be torn down cleanly.
    let _ = std::fs::set_permissions(&locked_dir, std::fs::Permissions::from_mode(0o700));

    assert_eq!(
        removed, 2,
        "both removable stale entries are reclaimed despite the locked directory"
    );
    assert!(
        !ok_file_a.exists(),
        "sibling before the locked entry removed"
    );
    assert!(
        !ok_file_b.exists(),
        "sibling after the locked entry removed"
    );
    assert!(
        locked_dir.exists(),
        "the unremovable entry is left in place, not aborting the sweep"
    );
}

// ---------------------------------------------------------------------------
// A transparent INPUT survives the import (sc-24111)
// ---------------------------------------------------------------------------
//
// The other direction from the thumbnail tests in `tests::media`: a user-provided RGBA reference
// or edit source, uploaded here, has to reach the worker with its alpha intact. References travel
// by ASSET ID (`ImageRequest::reference_asset_id` / `reference_asset_ids`), not as payload bytes,
// so "the bytes handed to the worker" is exactly the stored file this test reads back — the worker
// opens that path. What the engine then does with the channel is the worker/engine PR.
//
// `normalize_image_upload` stores a natively-supported PNG byte-for-byte with no decode at all, so
// this should hold — but every upload fixture in this crate is RGB or a stub, so nothing asserted
// it, and a future normalization step (a re-encode, an EXIF rewrite, a "canonicalize to RGB") would
// land silently.

#[tokio::test]
async fn an_uploaded_rgba_png_reaches_the_asset_store_with_its_alpha_intact() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Transparent reference" }),
    )
    .await;
    let project_id = created["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(created["path"].as_str().unwrap());

    let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("alpha")
        .join("alpha-64.png");
    let fixture_bytes = std::fs::read(&fixture_path).expect("RGBA fixture reads");

    let boundary = "SCENEWORKS_ALPHA_BOUNDARY";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"cutout.png\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body.extend_from_slice(&fixture_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let (status, _, response) = request_raw(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/assets"),
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset: Value = serde_json::from_slice(&response).expect("json body parses");
    assert_eq!(asset["type"], "image");
    assert_eq!(asset["file"]["mimeType"], "image/png");

    let stored_rel = asset["file"]["path"]
        .as_str()
        .expect("the sidecar records a stored path");
    let stored = project_path.join(stored_rel);
    assert_eq!(
        std::fs::read(&stored).expect("stored asset reads"),
        fixture_bytes,
        "the import re-encoded the uploaded PNG instead of storing it byte-for-byte"
    );

    let decoded = image::open(&stored).expect("stored asset decodes");
    assert_eq!(
        decoded.color(),
        image::ColorType::Rgba8,
        "the stored reference lost its alpha channel before the worker could read it"
    );
    let decoded = decoded.to_rgba8();
    let mut histogram = std::collections::BTreeMap::new();
    for pixel in decoded.pixels() {
        *histogram.entry(pixel.0[3]).or_insert(0) += 1;
    }
    assert!(
        histogram.get(&0).copied().unwrap_or(0) > 0,
        "no transparency survived"
    );
    assert!(
        histogram.get(&255).copied().unwrap_or(0) > 0,
        "no opaque pixels survived"
    );
    assert!(
        histogram.len() >= 8,
        "the soft edge collapsed to {} alpha values",
        histogram.len()
    );
}

#[tokio::test]
async fn an_uploaded_rgb_png_is_still_stored_unchanged() {
    // The control for the test above: widening nothing here must also change nothing here.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Opaque reference" }),
    )
    .await;
    let project_id = created["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(created["path"].as_str().unwrap());

    let fixture_bytes = std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("alpha")
            .join("opaque-rgb-64.png"),
    )
    .expect("RGB fixture reads");

    let boundary = "SCENEWORKS_OPAQUE_BOUNDARY";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"flat.png\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body.extend_from_slice(&fixture_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let (status, _, response) = request_raw(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/assets"),
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset: Value = serde_json::from_slice(&response).expect("json body parses");
    let stored = project_path.join(asset["file"]["path"].as_str().expect("stored path"));
    assert_eq!(
        std::fs::read(&stored).expect("stored asset reads"),
        fixture_bytes
    );
    assert_eq!(
        image::open(&stored).expect("decodes").color(),
        image::ColorType::Rgb8,
        "an opaque upload grew an alpha channel"
    );
}
