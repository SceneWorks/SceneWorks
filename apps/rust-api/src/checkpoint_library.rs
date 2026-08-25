//! The linked-library lifecycle seam (epic 20398, sc-20635).
//!
//! AC1 says a library root can be **added, renamed, rescanned, relocated, removed and relinked**.
//! Every one of those is a `CheckpointPlanStore` primitive; this module is what makes them
//! reachable from a client, which is what makes the rest of the epic mean anything: without an
//! approve route no client can produce the `linkedRootId` that `POST /api/v1/models/import`
//! validates, and without a scan route the `Ready` / `Needs Relink` / `Needs Rescan` verdicts and
//! the header-only candidate list are internal enums nothing can render.
//!
//! | AC1 verb | route |
//! | --- | --- |
//! | added | `POST /api/v1/models/library-roots` |
//! | renamed | `PATCH /api/v1/models/library-roots/:root_id` with `label` |
//! | relocated / relinked | `PATCH /api/v1/models/library-roots/:root_id` with `path` |
//! | rescanned | `GET .../:root_id/scan` (the library) and `POST .../:root_id/rescan` (one checkpoint) |
//! | removed | `DELETE /api/v1/models/library-roots/:root_id` |
//!
//! Two properties this seam keeps:
//!
//! * **E6 — the app never owns the library.** The only absolute path that crosses the boundary is
//!   the one the user picks when approving or relinking. Everything afterwards is `rootId +
//!   relativePath`, and removal drops SceneWorks' own plan documents and derivatives only.
//! * **Local-only for the path-bearing writes.** Approving and relinking name an absolute host
//!   path and re-point durable local state, exactly like `model_library::relocate_model_library`,
//!   so they take the same loopback-peer requirement. A missing peer fails closed.
//!
//! Every refusal is the plan store's own typed `[checkpoint-plan:<code>]` diagnostic, mapped to a
//! status by its `code()` and carried in `context.reason` so a client branches on the code rather
//! than on prose.

use axum::extract::{ConnectInfo, Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::SocketAddr;
use std::path::PathBuf;

use crate::error::ApiError;
use crate::AppState;
use sceneworks_core::checkpoint_derivative::{
    CheckpointDerivativeError, CheckpointDerivativeStore,
};
use sceneworks_core::checkpoint_plan_store::{
    ApprovedRootV1, CheckpointPlanError, CheckpointPlanStore, LinkedCheckpointStatusV1,
    LinkedRootScanV1,
};

/// The typed `code` every plan-store refusal is returned under. The actionable discriminant is
/// `context.reason` — the store's own stable kebab-case code.
pub(crate) const CHECKPOINT_LIBRARY_REJECTED_CODE: &str = "checkpoint_library_rejected";
pub(crate) const CHECKPOINT_LIBRARY_NOT_PERMITTED_CODE: &str = "checkpoint_library_not_permitted";

/// One approved root as the wire sees it: the persisted record plus the label a client should
/// actually render, so no client re-implements the "user label, else directory name" fallback.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LibraryRootView {
    #[serde(flatten)]
    root: ApprovedRootV1,
    display_label: String,
}

impl From<ApprovedRootV1> for LibraryRootView {
    fn from(root: ApprovedRootV1) -> Self {
        Self {
            display_label: root.display_label(),
            root,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LibraryRootsResponse {
    roots: Vec<LibraryRootView>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ApproveLibraryRootRequest {
    /// The absolute directory the user picked. Canonicalized by the store, so a symlinked pick is
    /// bound to its real directory once and every later confinement compares against that.
    path: String,
    #[serde(default)]
    label: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UpdateLibraryRootRequest {
    /// Rename: presentation only, and the root id and every compiled plan are untouched.
    #[serde(default)]
    label: Option<String>,
    /// Relocate / relink: the library's new absolute directory, keeping the root id so every
    /// persisted `Linked { rootId, relativePath }` stays valid.
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RescanCheckpointRequest {
    /// The checkpoint to recompile, relative to the root. `/`-separated and confined; the store
    /// applies the same rules a compile would.
    relative_path: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoveLibraryRootResponse {
    root: ApprovedRootV1,
    /// The checkpoint identities this dropped. Store documents only — no library file is touched.
    removed_checkpoints: Vec<String>,
    /// The cached derivatives produced from this root that were reclaimed. A derivative that is
    /// leased, pinned or being produced refuses the whole removal (the shared store's own rule),
    /// so this is never a partial list of successes beside silent failures.
    removed_derivatives: Vec<String>,
    reclaimed_derivative_bytes: u64,
}

/// `GET /api/v1/models/library-roots` — every approved linked-library root.
pub(crate) async fn list_library_roots(
    State(state): State<AppState>,
) -> Result<Json<LibraryRootsResponse>, ApiError> {
    let store = open_store(&state);
    let roots = blocking(move || store.approved_roots()).await?;
    Ok(Json(LibraryRootsResponse {
        roots: roots.roots.into_iter().map(LibraryRootView::from).collect(),
    }))
}

/// `POST /api/v1/models/library-roots` — approve a directory as a linked library (AC1 "added").
///
/// Idempotent on the directory: re-approving one already bound to a root returns that root rather
/// than minting a second identity for the same files, and adopts the label if the caller sent a
/// different one.
pub(crate) async fn approve_library_root(
    State(state): State<AppState>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    Json(request): Json<ApproveLibraryRootRequest>,
) -> Result<(StatusCode, Json<LibraryRootView>), ApiError> {
    require_local_peer(connect_info.map(|ConnectInfo(addr)| addr))?;
    let path = required_path(&request.path)?;
    let label = request.label.as_deref().map(str::trim).map(str::to_owned);
    let store = open_store(&state);
    let root = blocking(move || match label.as_deref() {
        Some(label) => store.approve_root_with_label(&path, label),
        None => store.approve_root(&path),
    })
    .await?;
    Ok((StatusCode::CREATED, Json(root.into())))
}

/// `PATCH /api/v1/models/library-roots/:root_id` — rename and/or relocate one root.
///
/// Both edits are independent by construction (`rename_root` moves only the label, `relink_root`
/// only the path), so a request carrying both applies both and a request carrying neither is a
/// refusal rather than a silent no-op.
pub(crate) async fn update_library_root(
    State(state): State<AppState>,
    Path(root_id): Path<String>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    Json(request): Json<UpdateLibraryRootRequest>,
) -> Result<Json<LibraryRootView>, ApiError> {
    let label = request.label.as_deref().map(str::trim).map(str::to_owned);
    let path = match request.path.as_deref().map(str::trim) {
        Some(path) if !path.is_empty() => Some(required_path(path)?),
        Some(_) => return Err(ApiError::bad_request("A library location is required.")),
        None => None,
    };
    if label.is_none() && path.is_none() {
        return Err(ApiError::bad_request(
            "Provide a label to rename the library, a path to relink it, or both.",
        ));
    }
    if path.is_some() {
        // Only the relink half names a host path, so only it is local-only.
        require_local_peer(connect_info.map(|ConnectInfo(addr)| addr))?;
    }
    let store = open_store(&state);
    let root = blocking(move || {
        let mut root = None;
        if let Some(label) = label.as_deref() {
            root = Some(store.rename_root(&root_id, label)?);
        }
        if let Some(path) = path.as_deref() {
            root = Some(store.relink_root(&root_id, path)?);
        }
        Ok(root.expect("one of label/path is present"))
    })
    .await?;
    Ok(Json(root.into()))
}

/// `DELETE /api/v1/models/library-roots/:root_id` — forget a library (AC1 "removed").
///
/// E6: this removes SceneWorks' own documents — the approved-root binding, the catalog records,
/// the plans and bindings — and the derivatives produced from them. The library directory is never
/// opened for writing, enumerated for deletion, or referenced at all.
pub(crate) async fn remove_library_root(
    State(state): State<AppState>,
    Path(root_id): Path<String>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
) -> Result<Json<RemoveLibraryRootResponse>, ApiError> {
    // The same local-only gate `approve_library_root` and the relink half of `update_library_root`
    // carry. Approving and relinking are gated because they name a host path; forgetting a library
    // destroys the plans, records and derivatives that path produced, which is no less a host-state
    // change, and leaving it open let a remote caller tear down a library it could not have added.
    require_local_peer(connect_info.map(|ConnectInfo(addr)| addr))?;
    let data_dir = state.settings.data_dir.clone();
    let (removal, derivatives) = tokio::task::spawn_blocking(move || {
        CheckpointDerivativeStore::open(&data_dir)?.remove_root(&root_id)
    })
    .await
    .map_err(task_failed)?
    .map_err(derivative_error_to_api_error)?;
    Ok(Json(RemoveLibraryRootResponse {
        root: removal.root,
        removed_checkpoints: removal.removed_checkpoints,
        reclaimed_derivative_bytes: derivatives
            .iter()
            .map(|outcome| outcome.reclaimed_bytes)
            .sum(),
        removed_derivatives: derivatives
            .into_iter()
            .map(|outcome| outcome.cache_key)
            .collect(),
    }))
}

/// `GET /api/v1/models/library-roots/:root_id/scan` — what the library holds and what is
/// selectable (AC1 "rescanned", library scope).
///
/// Read-only with respect to the library: bounded header-only discovery joined with the compile
/// state of everything already persisted under the root. A header-only candidate is NEVER
/// `selectable` until a full-content compile has produced a `Ready` plan for it (E7).
pub(crate) async fn scan_library_root(
    State(state): State<AppState>,
    Path(root_id): Path<String>,
) -> Result<Json<LinkedRootScanV1>, ApiError> {
    let store = open_store(&state);
    let scan = blocking(move || store.scan_root(&root_id)).await?;
    Ok(Json(scan))
}

/// `POST /api/v1/models/library-roots/:root_id/rescan` — recompile ONE checkpoint from the bytes
/// on disk right now (AC1 "rescanned", checkpoint scope).
///
/// Identity is preserved, so nothing that references the checkpoint has to be rewritten; the plan,
/// its layer fingerprints and its source stamps are replaced. The response is the post-rescan
/// verdict, so a client that rescanned a `Needs Rescan` entry sees immediately whether it is
/// selectable now.
pub(crate) async fn rescan_library_checkpoint(
    State(state): State<AppState>,
    Path(root_id): Path<String>,
    Json(request): Json<RescanCheckpointRequest>,
) -> Result<Json<LinkedCheckpointStatusV1>, ApiError> {
    let relative_path = request.relative_path.trim().to_owned();
    if relative_path.is_empty() {
        return Err(ApiError::bad_request("A checkpoint path is required."));
    }
    let store = open_store(&state);
    let status = blocking(move || {
        let compiled = store.compile_linked(&root_id, &relative_path)?;
        store.checkpoint_status(&compiled.checkpoint_id)
    })
    .await?;
    Ok(Json(status))
}

fn open_store(state: &AppState) -> CheckpointPlanStore {
    CheckpointPlanStore::open(&state.settings.data_dir)
}

/// Run one plan-store operation off the async runtime. Compiles hash multi-gigabyte files and even
/// a scan walks a whole library, so none of this may run on a runtime worker.
async fn blocking<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T, CheckpointPlanError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(task_failed)?
        .map_err(plan_error_to_api_error)
}

fn task_failed(error: tokio::task::JoinError) -> ApiError {
    tracing::error!(
        event = "checkpoint_library_task_failed",
        detail = %error,
        "a checkpoint library operation could not be completed"
    );
    ApiError::internal("SceneWorks could not complete that library operation.")
}

fn required_path(path: &str) -> Result<PathBuf, ApiError> {
    let path = path.trim();
    if path.is_empty() {
        return Err(ApiError::bad_request("A library folder is required."));
    }
    Ok(PathBuf::from(path))
}

/// Approving and relinking take an absolute host path and rewrite durable local state, so they are
/// local operations only — the same rule the model-library relocation seam applies, and for the
/// same two reasons: the typed refusals are a filesystem-existence oracle, and a token-holding LAN
/// client could otherwise re-point a shared server's library. A missing peer fails closed.
fn require_local_peer(peer: Option<SocketAddr>) -> Result<(), ApiError> {
    if peer.is_some_and(|addr| addr.ip().is_loopback()) {
        return Ok(());
    }
    Err(ApiError::typed(
        StatusCode::FORBIDDEN,
        "A model library can only be added, relinked, or forgotten from SceneWorks running on this \
         machine.",
        CHECKPOINT_LIBRARY_NOT_PERMITTED_CODE,
        json!({ "reason": "not_a_local_client" }),
    ))
}

fn derivative_error_to_api_error(error: CheckpointDerivativeError) -> ApiError {
    match error {
        CheckpointDerivativeError::Plan(error) => plan_error_to_api_error(error),
        other => {
            tracing::error!(
                event = "checkpoint_library_derivative_failed",
                detail = %other,
                "a checkpoint derivative operation could not be completed"
            );
            ApiError::internal("SceneWorks could not reclaim this library's cached derivatives.")
        }
    }
}

/// Typed plan-store refusal → typed HTTP refusal.
///
/// The status separates "the request named something that is not there or not allowed" (400/404)
/// from "the request is fine but the library's current state forbids it" (409) from "the store
/// itself is broken" (500), and `context.reason` carries the store's stable `code()` so a client
/// branches on the code rather than on the sentence.
fn plan_error_to_api_error(error: CheckpointPlanError) -> ApiError {
    let code = error.code();
    let status = match &error {
        CheckpointPlanError::UnknownRoot { .. } | CheckpointPlanError::UnknownCheckpoint { .. } => {
            StatusCode::NOT_FOUND
        }
        CheckpointPlanError::RootNotApprovable { .. }
        | CheckpointPlanError::InvalidRelativePath { .. }
        | CheckpointPlanError::InvalidRootLabel { .. }
        | CheckpointPlanError::InvalidPlanId { .. }
        | CheckpointPlanError::UnsupportedLocator { .. }
        // sc-20636's managed refusals. This surface is linked-only, so none of them is reachable
        // through these routes today; they are mapped rather than lumped into a 500 so that
        // adding a managed route later does not silently report a malformed install id as a
        // server fault.
        | CheckpointPlanError::InvalidInstallId { .. }
        | CheckpointPlanError::LocatorOwnershipMismatch { .. }
        | CheckpointPlanError::Contract(_) => StatusCode::BAD_REQUEST,
        CheckpointPlanError::RootUnavailable { .. }
        | CheckpointPlanError::RootAlreadyApproved { .. }
        | CheckpointPlanError::RootIdCollision { .. }
        | CheckpointPlanError::UnrunnableSource { .. }
        | CheckpointPlanError::MissingPlan { .. }
        | CheckpointPlanError::PlanTampered { .. }
        | CheckpointPlanError::SourceMissing { .. }
        | CheckpointPlanError::SourceDrifted { .. }
        | CheckpointPlanError::InstallUnavailable { .. }
        | CheckpointPlanError::InstallIdTaken { .. }
        | CheckpointPlanError::PathEscapesRoot { .. } => StatusCode::CONFLICT,
        CheckpointPlanError::Corrupt { .. } | CheckpointPlanError::Io { .. } => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    // The message already carries `[checkpoint-plan:<code>] …`, which is the actionable detail for
    // an unrunnable source (the inspector's diagnostics) and for drift (the two digests).
    ApiError::typed(
        status,
        error.to_string(),
        CHECKPOINT_LIBRARY_REJECTED_CODE,
        json!({ "reason": code }),
    )
}
