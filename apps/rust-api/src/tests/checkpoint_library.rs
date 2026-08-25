//! The linked-library lifecycle at the wire (epic 20398, sc-20635).
//!
//! AC1's six verbs — added, renamed, rescanned, relocated, removed, relinked — are only true "from
//! a client" if they cross HTTP. These tests drive the whole chain a client actually walks:
//! approve a folder, see what it holds, compile one candidate, feed the resulting `linkedRootId`
//! to `POST /api/v1/models/import`, survive the library moving, and forget it again.
//!
//! What the compile at the end of this file produces is the SAME `importPlan.checkpointId` the
//! worker stamps onto the manifest entry, which is what
//! `image_jobs::tests::an_approved_root_becomes_a_selectable_plan_backed_model_through_the_import_stamp`
//! then hands to `prepare_image_route`. `prepare_image_route` is private to the worker's
//! backend-gated image lane, so the two halves cannot live in one test; they join on that
//! identity, and both assert it explicitly.

use super::support::*;
use serde_json::Value;
use std::path::Path;

/// Adding and relinking a library name an absolute host path, so both are loopback-only.
const LOCAL_PEER: &str = "127.0.0.1:54321";

/// A minimal single-file Krea 2 native DiT — the `txtfusion.` marker the family detector keys on,
/// every tensor dense bf16. Enough for a real full-content compile, with no weights and no GPU.
fn write_krea_native_file(path: &Path) {
    let mut header = serde_json::Map::new();
    let mut offset = 0_u64;
    for name in [
        "model.diffusion_model.txtfusion.projector.weight",
        "model.diffusion_model.blocks.0.attn.wq.weight",
        "model.diffusion_model.first.weight",
    ] {
        header.insert(
            name.to_owned(),
            json!({"dtype": "BF16", "shape": [1], "data_offsets": [offset, offset + 2]}),
        );
        offset += 2;
    }
    let encoded = serde_json::to_vec(&Value::Object(header)).expect("header serializes");
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, 0x5a);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("library dir creates");
    std::fs::write(path, bytes).expect("checkpoint writes");
}

fn seed_manifests(config_dir: &Path) {
    let manifests = config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest dir creates");
    for name in [
        "builtin.models.jsonc",
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
    ] {
        let field = if name.ends_with("models.jsonc") {
            "models"
        } else {
            "loras"
        };
        std::fs::write(
            manifests.join(name),
            format!("{{ \"schemaVersion\": 1, \"{field}\": [] }}"),
        )
        .expect("manifest writes");
    }
}

/// The whole AC1 lifecycle over HTTP, in the order a client walks it.
///
/// This is the seam sc-20634's review found missing: with no approve route the `linkedRootId` that
/// `POST /api/v1/models/import` validates could never be produced by a client, and with no scan
/// route `Ready` / `Needs Relink` / `Needs Rescan` and the header-only candidate list were
/// internal enums nothing could render.
///
/// Failing mutations: drop any one route from the router in `lib.rs` and the corresponding step
/// returns 404/405; make `scan_library_root` report a candidate `selectable` before it compiles
/// and the pre-compile assertion goes red.
#[tokio::test]
async fn the_library_root_lifecycle_is_reachable_over_http() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    seed_manifests(&temp_dir.path().join("config"));
    let settings = test_settings(&temp_dir);
    std::fs::create_dir_all(&settings.data_dir).expect("data dir creates");

    let library = temp_dir.path().join("comfy-library");
    write_krea_native_file(&library.join("checkpoints/kreamania.safetensors"));
    let library = std::fs::canonicalize(&library).expect("library canonicalizes");
    let app = create_app(settings).expect("app creates");

    // ---- added ----------------------------------------------------------------------------
    let (status, root) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({ "path": library.to_str().expect("utf-8"), "label": "ComfyUI" }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{root:?}");
    let root_id = root["rootId"].as_str().expect("root id").to_owned();
    assert_eq!(root["label"], json!("ComfyUI"));
    assert_eq!(root["displayLabel"], json!("ComfyUI"));

    // The path-bearing writes are LOCAL-only, exactly like the model-library relocation seam: a
    // LAN peer gets a typed refusal, not a filesystem oracle.
    let (status, denied) = request(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({ "path": library.to_str().expect("utf-8") }),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{denied:?}");
    assert_eq!(denied["context"]["reason"], json!("not_a_local_client"));

    let (status, listed) = request(
        app.clone(),
        "GET",
        "/api/v1/models/library-roots",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{listed:?}");
    assert_eq!(listed["roots"].as_array().expect("roots").len(), 1);

    // ---- renamed --------------------------------------------------------------------------
    let (status, renamed) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/models/library-roots/{root_id}"),
        json!({ "label": "Shared checkpoints" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{renamed:?}");
    assert_eq!(renamed["label"], json!("Shared checkpoints"));
    assert_eq!(renamed["rootId"], json!(root_id), "identity never moves");

    // ---- rescanned (library scope): visible, and NOT selectable until it compiles (E7) ------
    let (status, scan) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/models/library-roots/{root_id}/scan"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{scan:?}");
    assert_eq!(scan["available"], json!(true));
    let candidate = &scan["candidates"][0];
    assert_eq!(
        candidate["candidate"]["relativePath"],
        json!("checkpoints/kreamania.safetensors")
    );
    assert_eq!(
        candidate["selectable"],
        json!(false),
        "header evidence never promotes a candidate on its own: {scan:?}"
    );
    assert_eq!(candidate["status"], Value::Null);

    // ---- rescanned (checkpoint scope): the full-content compile ----------------------------
    let (status, compiled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/models/library-roots/{root_id}/rescan"),
        json!({ "relativePath": "checkpoints/kreamania.safetensors" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{compiled:?}");
    assert_eq!(compiled["state"], json!("ready"));
    let checkpoint_id = compiled["checkpointId"].as_str().expect("id").to_owned();
    assert_eq!(
        checkpoint_id,
        format!("linked/{root_id}/checkpoints/kreamania.safetensors"),
        "the identity is rootId + relativePath, and it is what the worker stamps as \
         importPlan.checkpointId"
    );

    let (_, scan) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/models/library-roots/{root_id}/scan"),
        Value::Null,
    )
    .await;
    assert_eq!(
        scan["candidates"][0]["selectable"],
        json!(true),
        "a compiled candidate is selectable: {scan:?}"
    );

    // ---- the import seam accepts the id this surface produced -------------------------------
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/models/import",
        json!({
            "linkedRootId": root_id,
            "linkedRelativePath": "checkpoints/kreamania.safetensors",
            "type": "image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job:?}");
    assert_eq!(job["payload"]["linkedRootId"], json!(root_id));
    assert_eq!(job["payload"]["targetDir"], Value::Null);

    // ---- relocated / relinked ---------------------------------------------------------------
    let moved = temp_dir.path().join("moved-library");
    std::fs::create_dir_all(moved.join("checkpoints")).expect("moved dir creates");
    std::fs::rename(
        library.join("checkpoints/kreamania.safetensors"),
        moved.join("checkpoints/kreamania.safetensors"),
    )
    .expect("library moves");
    std::fs::remove_dir_all(&library).expect("old library goes away");
    let moved = std::fs::canonicalize(&moved).expect("moved canonicalizes");

    let (status, scan) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/models/library-roots/{root_id}/scan"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{scan:?}");
    assert_eq!(scan["available"], json!(false));
    assert_eq!(scan["unmatched"][0]["state"], json!("needs_relink"));

    let (status, relinked) = request_with_peer(
        app.clone(),
        "PATCH",
        &format!("/api/v1/models/library-roots/{root_id}"),
        json!({ "path": moved.to_str().expect("utf-8") }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{relinked:?}");
    assert_eq!(relinked["rootId"], json!(root_id));
    let (_, scan) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/models/library-roots/{root_id}/scan"),
        Value::Null,
    )
    .await;
    assert_eq!(
        scan["candidates"][0]["selectable"],
        json!(true),
        "a relink restores selection WITHOUT recompiling: {scan:?}"
    );

    // ---- removed ----------------------------------------------------------------------------
    // Forgetting a library destroys the plans, records and derivatives that path produced, so it
    // carries the same loopback gate as adding and relinking (sc-20651): a LAN peer cannot tear
    // down a library it could never have added.
    let (status, denied) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/models/library-roots/{root_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{denied:?}");
    assert_eq!(denied["context"]["reason"], json!("not_a_local_client"));
    let (_, still_there) = request(
        app.clone(),
        "GET",
        "/api/v1/models/library-roots",
        Value::Null,
    )
    .await;
    assert_eq!(
        still_there["roots"].as_array().expect("roots").len(),
        1,
        "the refused delete changed nothing"
    );

    let (status, removal) = request_with_peer(
        app.clone(),
        "DELETE",
        &format!("/api/v1/models/library-roots/{root_id}"),
        Value::Null,
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{removal:?}");
    assert_eq!(removal["removedCheckpoints"], json!([checkpoint_id]));
    // E6: only SceneWorks' own documents. The library file is untouched.
    assert!(moved.join("checkpoints/kreamania.safetensors").is_file());
    let (_, listed) = request(
        app.clone(),
        "GET",
        "/api/v1/models/library-roots",
        Value::Null,
    )
    .await;
    assert_eq!(listed["roots"].as_array().expect("roots").len(), 0);

    // And the id is gone: a stale client cannot import against it any more.
    let (status, gone) = request(
        app,
        "GET",
        &format!("/api/v1/models/library-roots/{root_id}/scan"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{gone:?}");
    assert_eq!(gone["context"]["reason"], json!("unknown-root"));
}

/// The import route validates the TRIMMED pair, so it must queue the trimmed pair.
///
/// It previously validated `"  root-abc  "`.trim() and then serialised the raw payload onto the
/// job, so a padded id passed the API and failed in the worker — and the queued `manifestEntry`
/// (built from the trimmed values) disagreed with the job payload about which checkpoint was being
/// imported.
///
/// Failing mutation: delete the two `payload.linked_* = Some(...)` write-backs in
/// `queue_model_import_job`.
#[tokio::test]
async fn a_padded_linked_import_queues_the_validated_values() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    seed_manifests(&temp_dir.path().join("config"));
    let settings = test_settings(&temp_dir);
    std::fs::create_dir_all(&settings.data_dir).expect("data dir creates");
    let library = temp_dir.path().join("library");
    write_krea_native_file(&library.join("checkpoints/kreamania.safetensors"));
    let library = std::fs::canonicalize(&library).expect("canonicalizes");
    let app = create_app(settings).expect("app creates");

    let (_, root) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({ "path": library.to_str().expect("utf-8") }),
        LOCAL_PEER,
    )
    .await;
    let root_id = root["rootId"].as_str().expect("root id").to_owned();

    let (status, job) = request(
        app,
        "POST",
        "/api/v1/models/import",
        json!({
            "linkedRootId": format!("  {root_id}  "),
            "linkedRelativePath": "  checkpoints/kreamania.safetensors  ",
            "type": "image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job:?}");
    assert_eq!(
        job["payload"]["linkedRootId"],
        json!(root_id),
        "the worker gets the id the API validated: {job:?}"
    );
    assert_eq!(
        job["payload"]["linkedRelativePath"],
        json!("checkpoints/kreamania.safetensors")
    );
    // The queued entry and the job payload must name the SAME checkpoint.
    assert_eq!(
        job["payload"]["manifestEntry"]["source"]["rootId"],
        job["payload"]["linkedRootId"]
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["source"]["relativePath"],
        job["payload"]["linkedRelativePath"]
    );
}

/// Deleting a plan-backed model must reach the plan store.
///
/// `delete_model` removes the manifest entry and the app-owned files; a linked model has no
/// app-owned files, so without this the record, plan, bindings and every cached derivative stayed
/// behind with nothing able to reach or reclaim them. E6 is unchanged: the library file and the
/// approved root both survive.
///
/// Failing mutation: delete the `forget_linked_checkpoint` call in `delete_model`.
#[tokio::test]
async fn deleting_a_linked_model_forgets_its_plan_but_never_the_library() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    seed_manifests(&temp_dir.path().join("config"));
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let config_dir = settings.config_dir.clone();
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    let library = temp_dir.path().join("library");
    write_krea_native_file(&library.join("checkpoints/kreamania.safetensors"));
    let library = std::fs::canonicalize(&library).expect("canonicalizes");
    let app = create_app(settings).expect("app creates");

    let (_, root) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({ "path": library.to_str().expect("utf-8") }),
        LOCAL_PEER,
    )
    .await;
    let root_id = root["rootId"].as_str().expect("root id").to_owned();
    let (_, compiled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/models/library-roots/{root_id}/rescan"),
        json!({ "relativePath": "checkpoints/kreamania.safetensors" }),
    )
    .await;
    let checkpoint_id = compiled["checkpointId"].as_str().expect("id").to_owned();

    // The entry the worker's import stamp produces: the plan identity, and no install path.
    std::fs::write(
        config_dir.join("manifests/user.models.jsonc"),
        serde_json::to_vec_pretty(&json!({
            "schemaVersion": 1,
            "models": [{
                "id": "linked_kreamania",
                "name": "kreamania",
                "type": "image",
                "family": "krea_2",
                "catalogScope": "user",
                "source": {
                    "provider": "linked-library",
                    "rootId": root_id,
                    "relativePath": "checkpoints/kreamania.safetensors",
                },
                "importPlan": { "checkpointId": checkpoint_id },
            }],
        }))
        .expect("manifest serializes"),
    )
    .expect("manifest writes");

    let store = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&data_dir);
    assert!(
        store.record(&checkpoint_id).is_ok(),
        "SANITY: the plan store holds the record before the delete"
    );

    let (status, body) = request(
        app,
        "DELETE",
        "/api/v1/models/linked_kreamania?permanent=true",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body:?}");
    assert_eq!(body["removedManifestEntry"], json!(true));
    assert!(
        matches!(
            store.record(&checkpoint_id),
            Err(
                sceneworks_core::checkpoint_plan_store::CheckpointPlanError::UnknownCheckpoint { .. }
            )
        ),
        "the plan record must be gone: {:?}",
        store.record(&checkpoint_id)
    );
    // E6: the library file and the approved root both survive a model delete.
    assert!(library.join("checkpoints/kreamania.safetensors").is_file());
    assert!(store
        .approved_roots()
        .expect("roots load")
        .get(&root_id)
        .is_some());
}

/// Every plan-store refusal reaches the client as a typed code, not as prose or a 500.
///
/// Failing mutation: map every `CheckpointPlanError` to one status in `plan_error_to_api_error`
/// and the differing-status assertions go red; drop `context.reason` and every reason assertion
/// goes red.
#[tokio::test]
async fn library_root_refusals_are_typed_and_status_mapped() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    seed_manifests(&temp_dir.path().join("config"));
    let settings = test_settings(&temp_dir);
    std::fs::create_dir_all(&settings.data_dir).expect("data dir creates");
    let app = create_app(settings).expect("app creates");

    // Not a directory that exists → the request named something unapprovable.
    let (status, body) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({ "path": temp_dir.path().join("nope").to_str().expect("utf-8") }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body:?}");
    assert_eq!(body["context"]["reason"], json!("root-not-approvable"));

    // A relative path never becomes a root.
    let (status, body) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({ "path": "relative/library" }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body:?}");
    assert_eq!(body["context"]["reason"], json!("root-not-approvable"));

    // An unknown root is 404, not 400 or 500.
    let (status, body) = request(
        app.clone(),
        "PATCH",
        "/api/v1/models/library-roots/root-0000000000000000",
        json!({ "label": "whatever" }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body:?}");
    assert_eq!(body["context"]["reason"], json!("unknown-root"));

    // A blank label is the store's own label validator, surfaced verbatim.
    let library = temp_dir.path().join("library");
    std::fs::create_dir_all(&library).expect("library creates");
    let library = std::fs::canonicalize(&library).expect("canonicalizes");
    let (_, root) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({ "path": library.to_str().expect("utf-8") }),
        LOCAL_PEER,
    )
    .await;
    let root_id = root["rootId"].as_str().expect("root id").to_owned();
    let (status, body) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/models/library-roots/{root_id}"),
        json!({ "label": "   " }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body:?}");
    assert_eq!(body["context"]["reason"], json!("invalid-root-label"));

    // An empty PATCH is a refusal rather than a silent no-op.
    let (status, body) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/models/library-roots/{root_id}"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body:?}");

    // A traversal path is refused by the same validator a compile applies (AC2).
    let (status, body) = request(
        app,
        "POST",
        &format!("/api/v1/models/library-roots/{root_id}/rescan"),
        json!({ "relativePath": "../../etc/passwd" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body:?}");
    assert_eq!(body["context"]["reason"], json!("invalid-relative-path"));
}
