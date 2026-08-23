//! The model source library's live status and relocation seam (sc-19709).
//!
//! When a model is installed on an external library that is currently unavailable, the desktop
//! prompt has exactly two recovery paths, and both need a server seam:
//!
//! * **Connect drive and retry** — `GET /api/v1/model-library` re-probes the configured library's
//!   durable physical identity once. Write-free and cheap: a retry gates on the single `available`
//!   boolean rather than rebuilding the model catalog or re-deriving availability in the client.
//! * **Choose a different library location** — `POST /api/v1/model-library/relocate` adopts a new
//!   root, but only after it physically carries every closure this install recorded. Rejections are
//!   typed (`reason` discriminants), never raw filesystem errors, so the prompt can render
//!   actionable guidance instead of an `ENOENT`.
//!
//! Relocation re-binds the durable identity ledger; it never rewrites download receipts, never
//! clears the validated-closure ledger, and never downloads. The persisted `HF_HOME` that survives
//! a relaunch is desktop state, so the response returns the exact `hfHome` the shell must store —
//! the API resolves the pair, the shell owns persistence, and neither guesses.
//!
//! Relocation is a LOCAL operation (loopback peers only). It names an absolute host path, rewrites
//! durable local state, and its typed refusals would otherwise answer "does this path exist?" for
//! any token-holding LAN client — who could also re-point a shared server's library.

use crate::error::ApiError;
use crate::AppState;
use axum::extract::{ConnectInfo, State};
use axum::http::StatusCode;
use axum::Json;
use sceneworks_core::model_artifacts::external_library::{
    probe_model_source_library, resolve_fresh_cache_home, resolve_relocation_target,
    ExternalLibraryBindingStore, ExternalLibraryError, LibraryRelocationError,
    LibraryRelocationRejection, ModelSourceLibraryStatus, RelocationTarget,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::SocketAddr;
use std::path::PathBuf;

pub(crate) const RELOCATION_REJECTED_CODE: &str = "model_library_relocation_rejected";
pub(crate) const RELOCATION_NOT_PERMITTED_CODE: &str = "model_library_relocation_not_permitted";

fn configured_library(state: &AppState) -> PathBuf {
    sceneworks_core::hf_home::model_source_library(&state.settings.data_dir)
        .root()
        .to_path_buf()
}

/// `GET /api/v1/model-library` — one write-free identity probe of the configured library.
pub(crate) async fn get_model_library(
    State(state): State<AppState>,
) -> Result<Json<ModelSourceLibraryStatus>, ApiError> {
    let data_dir = state.settings.data_dir.clone();
    let configured = configured_library(&state);
    let status =
        tokio::task::spawn_blocking(move || probe_model_source_library(&data_dir, &configured))
            .await
            .map_err(|error| ApiError::internal(format!("model library probe failed: {error}")))?;
    Ok(Json(status))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RelocateModelLibraryRequest {
    /// The folder the operator picked. Either a Hugging Face cache home (the folder containing
    /// `hub`) or the `hub` directory itself; the seam resolves which.
    pub path: String,
    /// Validate the candidate and resolve the target WITHOUT writing anything.
    ///
    /// This exists so the client can order its two durable writes safely: the shell must persist
    /// the new `HF_HOME` and the server must re-bind the identity ledger, and whichever runs second
    /// can fail. A dry run moves every ordinary refusal (wrong folder, missing models) ahead of
    /// BOTH writes, leaving only an exceptional race in the window where the two could disagree.
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RelocateModelLibraryResponse {
    #[serde(flatten)]
    pub target: RelocationTarget,
    /// Whether the durable binding was actually re-pointed at `libraryRoot`. False for a dry run.
    ///
    /// Deliberately NOT a probe of the adopted root: that would report `available: true` while
    /// `GET /api/v1/model-library` — which probes the CONFIGURED library, the one this process is
    /// still using — reports false, because the new root only takes effect on the next launch.
    pub adopted: bool,
}

/// `POST /api/v1/model-library/relocate` — adopt a new library root without redownloading.
///
/// Two shapes, decided by the installation's durable install evidence: with evidence, the picked
/// folder must be a library carrying every recorded model (the recovery prompt's "my drive moved"
/// case); without any, it is a fresh cache home — any existing directory, whose `hub` child is
/// created on adopt — so the Settings "change model library" control works before the first
/// download exactly like the first-run storage step does.
///
/// Local-only (see [`require_local_peer`]): it takes an absolute host path, and its typed refusals
/// would otherwise be a filesystem-existence oracle for any token-holding LAN client — which could
/// also re-point a shared server's library out from under everyone else.
pub(crate) async fn relocate_model_library(
    State(state): State<AppState>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    Json(request): Json<RelocateModelLibraryRequest>,
) -> Result<Json<RelocateModelLibraryResponse>, ApiError> {
    require_local_peer(connect_info.map(|ConnectInfo(addr)| addr))?;
    let picked = request.path.trim().to_owned();
    if picked.is_empty() {
        return Err(ApiError::bad_request("A library folder is required."));
    }
    let dry_run = request.dry_run;
    let data_dir = state.settings.data_dir.clone();
    tokio::task::spawn_blocking(move || {
        let store = ExternalLibraryBindingStore::new(&data_dir).map_err(|error| {
            relocation_failed("model library binding ledger is unavailable", &error)
        })?;
        let has_evidence = store.has_install_evidence().map_err(|error| {
            relocation_failed("model library install evidence could not be read", &error)
        })?;
        // The fresh-home shape additionally requires that no library was ever bound: evidence
        // detection is receipt/ledger-based, so an install whose receipts became unreadable must
        // not slide into binding an arbitrary folder — an existing binding is itself proof this
        // installation already had a library, and the strict shape judges the candidate.
        let bound = store
            .load()
            .map_err(|error| relocation_failed("model library binding could not be read", &error))?
            .is_some();
        let fresh = !has_evidence && !bound;
        let target = if fresh {
            resolve_fresh_cache_home(std::path::Path::new(&picked))
        } else {
            resolve_relocation_target(std::path::Path::new(&picked))
        }
        .map_err(rejection_to_api_error)?;
        if dry_run {
            // A fresh home's `hub` may not exist yet (it is created on adopt), so the dry run
            // validates the home directory itself: with no install evidence that is the same
            // canonical-directory + volume-identity check the adopt makes, moving every ordinary
            // refusal ahead of both durable writes. Writability of the folder is the one thing a
            // write-free dry run cannot prove.
            store
                .validate_relocation(if fresh {
                    &target.hf_home
                } else {
                    &target.library_root
                })
                .map_err(relocation_error_to_api_error)?;
            return Ok(Json(RelocateModelLibraryResponse {
                target,
                adopted: false,
            }));
        }
        if fresh {
            std::fs::create_dir_all(&target.library_root).map_err(|error| {
                relocation_failed(
                    "model library folder could not be created",
                    &ExternalLibraryError(error.to_string()),
                )
            })?;
        }
        store
            .relocate_binding(&target.library_root)
            .map_err(relocation_error_to_api_error)?;
        Ok(Json(RelocateModelLibraryResponse {
            target,
            adopted: true,
        }))
    })
    .await
    .map_err(|error| {
        relocation_failed(
            "model library relocation could not be completed",
            &ExternalLibraryError(error.to_string()),
        )
    })?
}

/// Relocation names a host path and rewrites durable local state, so it is a local operation only.
/// A missing peer fails closed: "we could not tell who is calling" is not permission.
fn require_local_peer(peer: Option<SocketAddr>) -> Result<(), ApiError> {
    if peer.is_some_and(|addr| addr.ip().is_loopback()) {
        return Ok(());
    }
    Err(ApiError::typed(
        StatusCode::FORBIDDEN,
        "The model library can only be relocated from SceneWorks running on this machine. \
         Reconnect the library on the host, or change its location there.",
        RELOCATION_NOT_PERMITTED_CODE,
        json!({ "reason": "not_a_local_client" }),
    ))
}

fn relocation_error_to_api_error(error: LibraryRelocationError) -> ApiError {
    match error {
        LibraryRelocationError::Rejected(rejection) => rejection_to_api_error(rejection),
        LibraryRelocationError::Failed(failure) => {
            relocation_failed("model library relocation could not be completed", &failure)
        }
    }
}

/// An internal failure the operator cannot act on. The underlying error is LOGGED, never returned:
/// a raw `io::Error` ("Permission denied (os error 13)") rendered into a dialog is exactly the raw
/// filesystem text this whole seam exists to keep away from users.
fn relocation_failed(summary: &'static str, error: &ExternalLibraryError) -> ApiError {
    tracing::error!(
        event = "model_library_relocation_failed",
        detail = %error,
        "{summary}"
    );
    ApiError::internal(
        "SceneWorks could not update the model library location. The previous library is \
         unchanged; check that the folder is readable and try again.",
    )
}

/// Typed rejection → typed 409. The `detail` sentence is a human summary; the `reason` inside
/// `context` is what the client branches on, so guidance never depends on message text.
fn rejection_to_api_error(rejection: LibraryRelocationRejection) -> ApiError {
    let detail = match &rejection {
        LibraryRelocationRejection::NotADirectory => {
            "That location could not be opened as a folder. Reconnect the drive, then choose the \
             model library folder again."
                .to_owned()
        }
        LibraryRelocationRejection::NotAModelLibrary => {
            "That folder does not contain a SceneWorks model library. Choose the Hugging Face \
             cache folder that holds your downloaded models."
                .to_owned()
        }
        LibraryRelocationRejection::MissingInstalledModels { repositories } => format!(
            "That library is missing models this installation already has: {}. Choose the library \
             that contains them, or reconnect the original drive.",
            repositories.join(", ")
        ),
        LibraryRelocationRejection::HubDirectoryExpected => {
            "SceneWorks reads models from a `hub` folder inside the Hugging Face cache. Choose the \
             folder that contains `hub` (or the `hub` folder itself)."
                .to_owned()
        }
        LibraryRelocationRejection::IdentityUnavailable { .. } => {
            "That volume's identity could not be read, so SceneWorks cannot bind it. Choose a \
             library on a mounted local or external disk."
                .to_owned()
        }
    };
    let context = serde_json::to_value(&rejection).unwrap_or(serde_json::Value::Null);
    ApiError::typed(
        StatusCode::CONFLICT,
        detail,
        RELOCATION_REJECTED_CODE,
        context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::model_artifacts::external_library::ExternalArtifactRequirement;
    use std::path::Path;

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn requirement(repository: &str) -> ExternalArtifactRequirement {
        ExternalArtifactRequirement {
            repository: repository.to_owned(),
            revision: Some(REVISION.to_owned()),
            variant: "default".to_owned(),
            files: vec![PathBuf::from("model.safetensors")],
            is_primary: true,
        }
    }

    fn seed_snapshot(hub: &Path, repository: &str) {
        let safe = sceneworks_core::hf_home::safe_repo_dir_name(repository).unwrap();
        let snapshot = hub
            .join(format!("models--{safe}"))
            .join("snapshots")
            .join(REVISION);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();
    }

    /// The one place a raw `io::Error` could still have reached a user: an internal failure whose
    /// underlying cause is a filesystem error. It is logged, never returned — the dialog renders
    /// `detail` verbatim, so "Permission denied (os error 13)" crossing this boundary would defeat
    /// the entire typed seam.
    #[test]
    fn an_internal_failure_never_returns_raw_filesystem_text() {
        let raw = ExternalLibraryError(
            std::io::Error::from(std::io::ErrorKind::PermissionDenied).to_string(),
        );
        assert!(
            raw.0.contains("denied"),
            "the fixture must actually carry raw io text: {raw}"
        );
        let error = relocation_failed("model library relocation could not be completed", &raw);
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        let detail = error.detail.to_lowercase();
        for leak in ["os error", "permission denied", "no such file", "errno"] {
            assert!(
                !detail.contains(leak),
                "raw filesystem text leaked: {detail}"
            );
        }
        assert!(
            detail.contains("previous library is unchanged"),
            "a failure must still say what state the operator is in: {detail}"
        );
    }

    /// An unrelated folder is rejected with the typed `not_a_model_library` reason and a 409 —
    /// never a raw filesystem error, and never a silent adoption that would orphan the install.
    #[test]
    fn an_unrelated_folder_is_rejected_with_typed_guidance() {
        let temp = tempfile::tempdir().unwrap();
        let unrelated = temp.path().join("holiday-photos");
        std::fs::create_dir_all(&unrelated).unwrap();
        let error = rejection_to_api_error(resolve_relocation_target(&unrelated).unwrap_err());
        assert_eq!(error.status, StatusCode::CONFLICT);
        assert_eq!(error.code, Some(RELOCATION_REJECTED_CODE));
        assert_eq!(
            error
                .context
                .as_ref()
                .and_then(|value| value["reason"].as_str()),
            Some("not_a_model_library")
        );
    }

    /// The happy path: the operator picks the Hugging Face cache home of a relocated library and
    /// the seam resolves the hub root plus the `HF_HOME` the shell must persist.
    #[test]
    fn picking_a_cache_home_resolves_the_hub_root_and_the_home_to_persist() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("relocated");
        let hub = home.join("hub");
        std::fs::create_dir_all(&hub).unwrap();
        seed_snapshot(&hub, "owner/model");
        let target = resolve_relocation_target(&home).unwrap();
        assert_eq!(target.library_root, hub);
        assert_eq!(target.hf_home, home);

        // Picking the hub directory itself resolves to the same pair.
        let target = resolve_relocation_target(&hub).unwrap();
        assert_eq!(target.library_root, hub);
        assert_eq!(target.hf_home, home);
    }

    /// Relocation must never orphan installed weights: a library that lacks a recorded closure is
    /// rejected, naming the repositories it is missing.
    #[test]
    fn relocation_rejects_a_library_missing_recorded_models_and_preserves_the_binding() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        let original = temp.path().join("original").join("hub");
        std::fs::create_dir_all(&original).unwrap();
        seed_snapshot(&original, "owner/model");
        let store = ExternalLibraryBindingStore::new(&data).unwrap();
        let bound = store
            .bind_or_probe_validated(&original, &[requirement("owner/model")])
            .unwrap()
            .0;

        let empty = temp.path().join("elsewhere").join("hub");
        std::fs::create_dir_all(&empty).unwrap();
        seed_snapshot(&empty, "someone/else");
        let error = store.relocate_binding(&empty).unwrap_err();
        assert_eq!(
            error,
            LibraryRelocationError::Rejected(LibraryRelocationRejection::MissingInstalledModels {
                repositories: vec!["owner/model".to_owned()],
            })
        );
        assert_eq!(store.load().unwrap(), Some(bound));
    }

    /// The accepted path re-points the durable binding at the new root and leaves the
    /// validated-closure ledger (the install evidence) byte-identical — no redownload, no lost
    /// installed state.
    #[test]
    fn relocation_rebinds_without_touching_install_evidence() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        let original = temp.path().join("original").join("hub");
        std::fs::create_dir_all(&original).unwrap();
        seed_snapshot(&original, "owner/model");
        let store = ExternalLibraryBindingStore::new(&data).unwrap();
        store
            .bind_or_probe_validated(&original, &[requirement("owner/model")])
            .unwrap();
        let closures_path = data
            .join("models")
            .join(".sceneworks-external-library-closures.json");
        let evidence_before = std::fs::read(&closures_path).unwrap();

        let moved = temp.path().join("moved").join("hub");
        std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
        std::fs::rename(&original, &moved).unwrap();
        let binding = store.relocate_binding(&moved).unwrap();
        assert_eq!(
            binding.canonical_path,
            std::fs::canonicalize(&moved).unwrap()
        );
        assert_eq!(std::fs::read(&closures_path).unwrap(), evidence_before);

        let status = probe_model_source_library(&data, &moved);
        assert!(status.available, "relocated library must probe available");
    }
}
