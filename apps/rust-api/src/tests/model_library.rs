//! HTTP-level contract for the unavailable-external-library recovery flow (sc-19709).
//!
//! The desktop prompt is driven entirely by what crosses this boundary: a typed `code` plus a
//! typed `context` object, a write-free re-probe endpoint, and typed relocation rejections. These
//! tests pin that contract at the wire, because a client that has to read `detail` prose — or
//! re-derive availability itself — is exactly the failure mode the seam exists to prevent.
use super::support::*;
use sceneworks_core::model_artifacts::external_library::{
    ExternalArtifactRequirement, ExternalLibraryBindingStore, EXTERNAL_LIBRARY_UNAVAILABLE_CODE,
};
use std::path::{Path, PathBuf};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
/// Relocation is a local-only operation, so every accepted call presents a loopback peer.
const LOCAL_PEER: &str = "127.0.0.1:54321";

/// The hub root the API resolves when no HF-cache env var is set — the cross-platform way to give
/// a test an isolated "external library" it can rename away to simulate an unplugged drive.
fn isolated_hub(data_dir: &Path) -> PathBuf {
    data_dir.join("cache").join("huggingface").join("hub")
}

fn seed_snapshot(hub: &Path, repository: &str) {
    let safe = sceneworks_core::hf_home::safe_repo_dir_name(repository).expect("safe repo name");
    let snapshot = hub
        .join(format!("models--{safe}"))
        .join("snapshots")
        .join(REVISION);
    std::fs::create_dir_all(&snapshot).expect("snapshot dir creates");
    std::fs::write(snapshot.join("model.safetensors"), b"weights").expect("weights write");
}

/// A durable download receipt: the install evidence that lets a disconnected library read
/// `installed_external_unavailable` instead of `missing`.
fn write_receipt(data_dir: &Path, repository: &str) {
    let managed = data_dir
        .join("models")
        .join(sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(repository));
    std::fs::create_dir_all(&managed).expect("managed dir creates");
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        serde_json::to_vec(&json!({
            "receipts": [{
                "repo": repository,
                "modelId": "relocatable",
                "variant": "default",
                "resolvedFiles": ["model.safetensors"],
                "snapshotRevision": REVISION,
            }]
        }))
        .expect("receipt serializes"),
    )
    .expect("receipt writes");
}

fn requirement(repository: &str) -> ExternalArtifactRequirement {
    ExternalArtifactRequirement {
        repository: repository.to_owned(),
        revision: Some(REVISION.to_owned()),
        variant: "default".to_owned(),
        files: vec![PathBuf::from("model.safetensors")],
        is_primary: true,
    }
}

/// Bind the library, then rename it away so the configured path no longer resolves — the CI-safe
/// stand-in for unplugging the drive.
fn install_then_disconnect(data_dir: &Path, repository: &str) -> PathBuf {
    let hub = isolated_hub(data_dir);
    seed_snapshot(&hub, repository);
    write_receipt(data_dir, repository);
    ExternalLibraryBindingStore::new(data_dir)
        .expect("binding store")
        .bind_or_probe_validated(&hub, &[requirement(repository)])
        .expect("library binds while connected");
    let detached = data_dir.join("detached-library");
    std::fs::rename(&hub, &detached).expect("library detaches");
    detached
}

/// The walking skeleton, end to end at the wire: a submission whose model lives on a disconnected
/// library is rejected with the typed 503 AND a typed context object naming the model and the
/// expected library — everything the prompt needs, none of it parsed out of prose. Reconnecting
/// the library makes the identical submission succeed, with no download and no reinstall.
#[tokio::test]
async fn a_disconnected_library_rejects_with_typed_context_and_a_reconnect_resumes_the_same_job() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let _env = isolate_hf_cache();
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    single_model_manifest(
        &settings.config_dir.join("manifests"),
        "relocatable",
        "owner/model",
    );
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    let detached = install_then_disconnect(&data_dir, "owner/model");
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Library Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let submission = json!({
        "projectId": project_id,
        "prompt": "a lighthouse",
        "model": "relocatable",
        "count": 1,
        "width": 512,
        "height": 512,
    });

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        submission.clone(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "a disconnected library must be a typed refusal, not a job: {body}"
    );
    assert_eq!(body["code"], EXTERNAL_LIBRARY_UNAVAILABLE_CODE);
    let context = &body["context"];
    assert_eq!(context["availability"], "installed_external_unavailable");
    assert_eq!(context["modelId"], "relocatable");
    assert_eq!(context["modelName"], "relocatable");
    assert!(
        context["configuredLibraryPath"]
            .as_str()
            .is_some_and(|path| !path.is_empty()),
        "the prompt names the expected library location: {body}"
    );
    assert!(
        context["expectedLibraryPath"].as_str().is_some(),
        "a bound library must report where it is expected: {body}"
    );

    // The re-probe endpoint agrees, and is the single boolean a retry gates on.
    let (status, probe) = request(app.clone(), "GET", "/api/v1/model-library", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(probe["available"], false);
    assert_eq!(probe["probeStatus"], "unavailable");

    // Reconnect the same physical library: the binding is restored by identity alone.
    std::fs::rename(&detached, isolated_hub(&data_dir)).expect("library reconnects");
    let (status, probe) = request(app.clone(), "GET", "/api/v1/model-library", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        probe["available"], true,
        "reconnect must re-probe available"
    );

    let (status, body) = request(app.clone(), "POST", "/api/v1/image/jobs", submission).await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "the resumed submission must enqueue after reconnect: {body}"
    );
    assert_eq!(body["payload"]["model"], "relocatable");
}

/// Relocation at the wire: an unrelated folder is refused with a typed `reason` the prompt can
/// branch on, and the durable binding is left untouched so nothing is lost by trying.
#[tokio::test]
async fn relocating_to_an_unrelated_folder_is_refused_with_a_typed_reason() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let _env = isolate_hf_cache();
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    install_then_disconnect(&data_dir, "owner/model");
    let app = create_app(settings).expect("app creates");

    let unrelated = temp_dir.path().join("holiday-photos");
    std::fs::create_dir_all(&unrelated).expect("unrelated dir creates");
    let (status, body) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/model-library/relocate",
        json!({ "path": unrelated.to_string_lossy() }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "model_library_relocation_rejected");
    assert_eq!(body["context"]["reason"], "not_a_model_library");
    assert!(
        body["detail"]
            .as_str()
            .is_some_and(|detail| !detail.is_empty()),
        "a refusal still carries human guidance: {body}"
    );
}

/// The fail-open this seam must not have: an installation whose evidence is RECEIPTS ONLY (no
/// validated-closure ledger record — an install that predates the ledger, or was never validated on
/// a bound library) must still be checked against the candidate. A decoy Hugging Face cache that
/// merely has SOME `models--*` directory is refused, naming the model it does not contain, and the
/// durable binding is untouched.
#[tokio::test]
async fn a_decoy_library_cannot_capture_a_receipt_backed_install() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let _env = isolate_hf_cache();
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    // Receipt evidence WITHOUT any binding or validated-closure record.
    let hub = isolated_hub(&data_dir);
    seed_snapshot(&hub, "owner/model");
    write_receipt(&data_dir, "owner/model");
    let app = create_app(settings).expect("app creates");

    let decoy = temp_dir.path().join("decoy").join("hub");
    std::fs::create_dir_all(&decoy).expect("decoy dir creates");
    seed_snapshot(&decoy, "someone/unrelated");
    let (status, body) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/model-library/relocate",
        json!({ "path": decoy.to_string_lossy() }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["context"]["reason"], "missing_installed_models");
    assert_eq!(body["context"]["repositories"][0], "owner/model");
    assert!(
        ExternalLibraryBindingStore::new(&data_dir)
            .expect("binding store")
            .load()
            .expect("binding reads")
            .is_none(),
        "a refused relocation must not have written a binding"
    );
}

/// Relocation names a host path and rewrites durable local state, so it is refused for any caller
/// that is not on this machine — including one holding a valid token.
#[tokio::test]
async fn relocation_is_refused_for_a_remote_caller() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let _env = isolate_hf_cache();
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    install_then_disconnect(&data_dir, "owner/model");
    let app = create_app(settings).expect("app creates");

    for peer in [Some("10.0.0.7:5555"), None] {
        let path = temp_dir
            .path()
            .join("anything")
            .to_string_lossy()
            .into_owned();
        let body = json!({ "path": path });
        let (status, body) = match peer {
            Some(peer) => {
                request_with_peer(
                    app.clone(),
                    "POST",
                    "/api/v1/model-library/relocate",
                    body,
                    peer,
                )
                .await
            }
            // No connect info at all: "we cannot tell who is calling" is not permission.
            None => request(app.clone(), "POST", "/api/v1/model-library/relocate", body).await,
        };
        assert_eq!(status, StatusCode::FORBIDDEN, "peer {peer:?}: {body}");
        assert_eq!(body["code"], "model_library_relocation_not_permitted");
        assert_eq!(body["context"]["reason"], "not_a_local_client");
    }
}

/// The dry run answers the same refusals with nothing written, which is what lets the client order
/// its two durable writes (the shell's `HF_HOME`, the server's binding) safely.
#[tokio::test]
async fn a_dry_run_validates_without_adopting() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let _env = isolate_hf_cache();
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    let detached = install_then_disconnect(&data_dir, "owner/model");
    let store = ExternalLibraryBindingStore::new(&data_dir).expect("binding store");
    let before = store.load().expect("binding reads");
    let relocated_home = temp_dir.path().join("relocated");
    std::fs::create_dir_all(&relocated_home).expect("relocated home creates");
    std::fs::rename(&detached, relocated_home.join("hub")).expect("library moves");
    let app = create_app(settings).expect("app creates");

    let (status, body) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/model-library/relocate",
        json!({ "path": relocated_home.to_string_lossy(), "dryRun": true }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["adopted"], false);
    assert_eq!(store.load().expect("binding reads"), before);

    // A dry run refuses exactly what the real call would, still without writing.
    let unrelated = temp_dir.path().join("holiday-photos");
    std::fs::create_dir_all(&unrelated).expect("unrelated dir creates");
    let (status, body) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/model-library/relocate",
        json!({ "path": unrelated.to_string_lossy(), "dryRun": true }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(store.load().expect("binding reads"), before);
}

/// The accepted relocation: the library moved, so the operator names its new home. The seam adopts
/// it and reports the `HF_HOME` the shell must persist — without redownloading.
#[tokio::test]
async fn relocating_to_the_moved_library_adopts_it_and_names_the_home_to_persist() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let _env = isolate_hf_cache();
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    let detached = install_then_disconnect(&data_dir, "owner/model");
    let relocated_home = temp_dir.path().join("relocated");
    std::fs::create_dir_all(&relocated_home).expect("relocated home creates");
    std::fs::rename(&detached, relocated_home.join("hub")).expect("library moves");
    let app = create_app(settings).expect("app creates");

    let (status, body) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/model-library/relocate",
        json!({ "path": relocated_home.to_string_lossy() }),
        LOCAL_PEER,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["adopted"], true);
    assert!(
        body["hfHome"]
            .as_str()
            .is_some_and(|home| Path::new(home) == relocated_home),
        "the shell is told exactly which HF_HOME to persist: {body}"
    );
    assert!(body["libraryRoot"]
        .as_str()
        .is_some_and(|root| Path::new(root) == relocated_home.join("hub")));
}
