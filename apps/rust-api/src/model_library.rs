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
//! a relaunch is desktop state, so the response returns the exact `huggingFaceHome` the shell must
//! store — the API resolves the pair, the shell owns persistence, and neither guesses.

use crate::error::ApiError;
use crate::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use sceneworks_core::model_artifacts::external_library::{
    probe_model_source_library, resolve_relocation_target, ExternalLibraryBindingStore,
    LibraryRelocationError, LibraryRelocationRejection, ModelSourceLibraryStatus, RelocationTarget,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub(crate) const RELOCATION_REJECTED_CODE: &str = "model_library_relocation_rejected";

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
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RelocateModelLibraryResponse {
    #[serde(flatten)]
    pub target: RelocationTarget,
    pub status: ModelSourceLibraryStatus,
}

/// `POST /api/v1/model-library/relocate` — adopt a new library root without redownloading.
pub(crate) async fn relocate_model_library(
    State(state): State<AppState>,
    Json(request): Json<RelocateModelLibraryRequest>,
) -> Result<Json<RelocateModelLibraryResponse>, ApiError> {
    let picked = request.path.trim().to_owned();
    if picked.is_empty() {
        return Err(ApiError::bad_request("A library folder is required."));
    }
    let data_dir = state.settings.data_dir.clone();
    tokio::task::spawn_blocking(move || {
        let target = resolve_relocation_target(std::path::Path::new(&picked))
            .map_err(rejection_to_api_error)?;
        let store = ExternalLibraryBindingStore::new(&data_dir).map_err(|error| {
            ApiError::internal(format!("model library ledger unavailable: {error}"))
        })?;
        store
            .relocate_binding(&target.library_root)
            .map_err(|error| match error {
                LibraryRelocationError::Rejected(rejection) => rejection_to_api_error(rejection),
                LibraryRelocationError::Failed(failure) => {
                    ApiError::internal(format!("model library relocation failed: {failure}"))
                }
            })?;
        let status = probe_model_source_library(&data_dir, &target.library_root);
        Ok(Json(RelocateModelLibraryResponse { target, status }))
    })
    .await
    .map_err(|error| ApiError::internal(format!("model library relocation failed: {error}")))?
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
        assert_eq!(target.huggingface_home, home);

        // Picking the hub directory itself resolves to the same pair.
        let target = resolve_relocation_target(&hub).unwrap();
        assert_eq!(target.library_root, hub);
        assert_eq!(target.huggingface_home, home);
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
