//! Read + control surface for the app-owned resolved-model hot cache (sc-19711).
//!
//! Four routes back the Settings storage card and the Model Manager's per-model local-copy
//! controls:
//!
//! * `GET  /api/v1/model-cache` — the effective policy this API process is running under, the
//!   store's usage accounting, and one row per entry.
//! * `POST /api/v1/model-cache/removal-preview` — what removing one entry would actually do.
//! * `POST /api/v1/model-cache/remove` — the manual removal itself.
//! * `POST /api/v1/model-cache/pin` — "keep locally" / "allow automatic removal".
//!
//! Three properties are load-bearing and must survive any edit here:
//!
//! 1. **The status read is cheap and write-free.** It is a UI-polled surface, so it does exactly
//!    one journal listing through [`ResolvedCacheStore::inspect`] — no session slot, no usage
//!    stamping, and no re-hashing of bundle bytes (that would put gigabytes of I/O behind a poll).
//!    Byte totals come from the ledger's recorded `verified_bytes`, never from a filesystem walk
//!    per row. The one filesystem measurement in this module is inside the removal preview, which
//!    is a single deliberate user action.
//! 2. **Pin state is reported as the resolver typed it.** `manual_removal_preview` answers with
//!    [`ManualRemovalPins::Known`] or [`ManualRemovalPins::Unknown`]; the DTO keeps that
//!    distinction (`pins.known` is absent for `Unknown`) so a UI cannot render "state could not be
//!    read" as "not pinned".
//! 3. **Policy is reported, not persisted, here.** The three sidecar processes capture the policy
//!    from their environment at spawn, so the desktop shell owns the durable copy
//!    (`set_resolved_cache_policy`). This API reports what it is *actually* running under, which is
//!    what lets the UI say "restart to apply" honestly instead of claiming a change already landed.

use crate::error::ApiError;
use crate::AppState;
use axum::extract::State;
use axum::Json;
use sceneworks_core::model_artifacts::resolved_cache::{
    ManualRemovalPins, ResolvedCacheEntryState, ResolvedCacheEntrySummary, ResolvedCacheError,
    ResolvedCacheStore, SourceVolumeRelation,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CachePolicyDto {
    pub(crate) enabled: bool,
    pub(crate) max_bytes: u64,
    pub(crate) inactivity_seconds: u64,
}

/// One resolved-cache entry as the UI sees it. `state` is the store's own typed entry state and is
/// never re-derived from strings; `pinned` is the *effective* pin (artifact pin OR any model
/// owner's pin), which is exactly what blocks automatic and manual removal.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CacheEntryDto {
    pub(crate) cache_key: String,
    pub(crate) state: ResolvedCacheEntryState,
    pub(crate) repository: Option<String>,
    pub(crate) revision: Option<String>,
    pub(crate) variant: Option<String>,
    pub(crate) tier: Option<String>,
    pub(crate) bytes: u64,
    pub(crate) pinned: bool,
    pub(crate) artifact_pinned: bool,
    pub(crate) model_pin_owners: Vec<String>,
    pub(crate) created_at: Option<u64>,
    pub(crate) last_used_at: Option<u64>,
    /// Catalog model ids whose own (non-co-requisite) source repositories include this entry's
    /// primary repository. Resolved here rather than in the UI so the identity mapping has exactly
    /// one implementation, and computed from the already-cached catalog snapshot — no filesystem.
    pub(crate) model_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CacheStatusDto {
    /// The policy THIS process is running under, captured from its environment at spawn.
    pub(crate) policy: CachePolicyDto,
    /// False when no cache store has ever been created under the data directory.
    pub(crate) initialized: bool,
    /// Present when the store exists but could not be listed. The UI must say "couldn't read"
    /// rather than render an empty, reassuring cache.
    pub(crate) error: Option<String>,
    pub(crate) used_bytes: u64,
    pub(crate) pinned_bytes: u64,
    /// Bytes held by complete, unpinned entries — the ceiling on what automatic cleanup or a
    /// manual removal could reclaim. Automatic eviction additionally requires a re-verified
    /// source, so this is an upper bound, not a promise.
    pub(crate) reclaimable_bytes: u64,
    pub(crate) entry_count: usize,
    /// `same` means the local copies live on the same volume as the source library, so they buy
    /// no disconnect or performance isolation.
    pub(crate) source_volume_relation: SourceVolumeRelation,
    pub(crate) source_library_path: PathBuf,
    pub(crate) entries: Vec<CacheEntryDto>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CacheKeyRequest {
    pub(crate) cache_key: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CachePinRequest {
    pub(crate) cache_key: String,
    pub(crate) pinned: bool,
}

/// The typed pin answer, preserved. `Unknown` carries no `known` object at all, so a client that
/// forgets to branch renders nothing rather than rendering "no pins".
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub(crate) enum CachePinsDto {
    Known {
        artifact_pinned: bool,
        owners: Vec<String>,
    },
    Unknown,
}

impl From<ManualRemovalPins> for CachePinsDto {
    fn from(pins: ManualRemovalPins) -> Self {
        match pins {
            ManualRemovalPins::Known {
                artifact_pinned,
                owners,
            } => Self::Known {
                artifact_pinned,
                owners,
            },
            ManualRemovalPins::Unknown => Self::Unknown,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemovalPreviewDto {
    pub(crate) cache_key: String,
    pub(crate) state: ResolvedCacheEntryState,
    /// Measured on disk under the metadata lock — the bytes this removal would actually free.
    pub(crate) reclaimable_bytes: u64,
    pub(crate) pins: CachePinsDto,
    /// Set when the authoritative external source is unreachable or incomplete: removing the local
    /// copy leaves the model unusable until that source returns. Removal is still allowed.
    pub(crate) source_unavailable_warning: Option<String>,
    /// Set when removal would currently be refused, with the refusal reason.
    pub(crate) blocked: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemovalOutcomeDto {
    pub(crate) cache_key: String,
    pub(crate) reclaimed_bytes: u64,
    pub(crate) source_unavailable_warning: Option<String>,
}

fn cache_error(error: ResolvedCacheError) -> ApiError {
    ApiError::bad_request(error.to_string())
}

/// Map every entry's primary repository to the catalog model ids that own it. One pass over the
/// cached catalog snapshot; no filesystem and no per-entry catalog scan.
async fn repository_owners(state: &AppState) -> BTreeMap<String, Vec<String>> {
    // The CACHED snapshot, not `model_catalog`: the join needs identity only, and the live
    // external-availability refresh is a probe this read has no reason to pay for.
    let Ok(models) = crate::models::model_catalog_snapshot(state).await else {
        // The join is a convenience: without it entries still render with their repository. Never
        // fail the status read because the catalog is momentarily unavailable.
        return BTreeMap::new();
    };
    let mut owners: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for model in models.iter() {
        let Some(id) = model.get("id").and_then(Value::as_str) else {
            continue;
        };
        for repository in crate::models::owned_source_repositories(model) {
            owners.entry(repository).or_default().push(id.to_owned());
        }
    }
    for ids in owners.values_mut() {
        ids.sort();
        ids.dedup();
    }
    owners
}

fn entry_dto(
    summary: ResolvedCacheEntrySummary,
    owners: &BTreeMap<String, Vec<String>>,
) -> CacheEntryDto {
    let ResolvedCacheEntrySummary {
        cache_key,
        state,
        metadata,
    } = summary;
    let Some(metadata) = metadata else {
        // Evicting / corrupt residue has no readable identity. It is still listed — a status
        // surface that hid it would under-report what the cache is holding.
        return CacheEntryDto {
            cache_key,
            state,
            repository: None,
            revision: None,
            variant: None,
            tier: None,
            bytes: 0,
            pinned: false,
            artifact_pinned: false,
            model_pin_owners: Vec::new(),
            created_at: None,
            last_used_at: None,
            model_ids: Vec::new(),
        };
    };
    let identity = &metadata.artifact.identity;
    CacheEntryDto {
        cache_key,
        state,
        model_ids: owners.get(&identity.repository).cloned().unwrap_or_default(),
        repository: Some(identity.repository.clone()),
        revision: Some(identity.revision.clone()),
        variant: Some(identity.variant.clone()),
        tier: metadata.artifact.provenance.fixed_artifact_tier.clone(),
        bytes: metadata.verified_bytes,
        pinned: metadata.effective_pin,
        artifact_pinned: metadata.artifact_pinned,
        model_pin_owners: metadata.model_pin_owners.iter().cloned().collect(),
        created_at: Some(metadata.created_at),
        last_used_at: metadata.last_used_at,
    }
}

pub(crate) async fn get_model_cache(
    State(state): State<AppState>,
) -> Result<Json<CacheStatusDto>, ApiError> {
    let owners = repository_owners(&state).await;
    let data_dir = state.settings.data_dir.clone();
    let source_library = sceneworks_core::hf_home::model_source_library(&data_dir)
        .root()
        .to_path_buf();
    let probe_root = source_library.clone();
    // ONE listing on a blocking thread. Everything below is arithmetic over that listing.
    let listed = tokio::task::spawn_blocking(move || {
        let Some(store) = ResolvedCacheStore::open_for_inspection(&data_dir)? else {
            return Ok::<_, ResolvedCacheError>(None);
        };
        let entries = store.inspect()?;
        let relation = store
            .compare_source_volume(&probe_root)
            .map(|observation| observation.relation)
            .unwrap_or(SourceVolumeRelation::Unknown);
        Ok(Some((entries, relation)))
    })
    .await
    .map_err(|error| ApiError::internal(format!("resolved-cache status task failed: {error}")))?;

    let policy = CachePolicyDto {
        enabled: state.settings.resolved_cache.enabled,
        max_bytes: state.settings.resolved_cache.max_bytes,
        inactivity_seconds: state.settings.resolved_cache.inactivity_seconds,
    };
    let (entries, relation, initialized, error) = match listed {
        Ok(Some((entries, relation))) => (entries, relation, true, None),
        Ok(None) => (Vec::new(), SourceVolumeRelation::Unknown, false, None),
        Err(error) => (
            Vec::new(),
            SourceVolumeRelation::Unknown,
            true,
            Some(error.to_string()),
        ),
    };
    let entries = entries
        .into_iter()
        .map(|summary| entry_dto(summary, &owners))
        .collect::<Vec<_>>();
    let used_bytes = entries.iter().map(|entry| entry.bytes).sum();
    let pinned_bytes = entries
        .iter()
        .filter(|entry| entry.pinned)
        .map(|entry| entry.bytes)
        .sum();
    let reclaimable_bytes = entries
        .iter()
        .filter(|entry| entry.state == ResolvedCacheEntryState::Complete && !entry.pinned)
        .map(|entry| entry.bytes)
        .sum();
    Ok(Json(CacheStatusDto {
        policy,
        initialized,
        error,
        used_bytes,
        pinned_bytes,
        reclaimable_bytes,
        entry_count: entries.len(),
        source_volume_relation: relation,
        source_library_path: source_library,
        entries,
    }))
}

/// Open the store for a mutating operation. Never creates the cache root as a side effect of a
/// pin or a removal: with no store there is nothing to act on.
async fn open_mutable(state: &AppState) -> Result<ResolvedCacheStore, ApiError> {
    let data_dir = state.settings.data_dir.clone();
    if !data_dir.join("models").join("resolved").exists() {
        return Err(ApiError::bad_request(
            "No local model cache exists on this machine.",
        ));
    }
    tokio::task::spawn_blocking(move || ResolvedCacheStore::open(&data_dir))
        .await
        .map_err(|error| ApiError::internal(format!("resolved-cache open task failed: {error}")))?
        .map_err(cache_error)
}

pub(crate) async fn preview_model_cache_removal(
    State(state): State<AppState>,
    Json(request): Json<CacheKeyRequest>,
) -> Result<Json<RemovalPreviewDto>, ApiError> {
    let store = open_mutable(&state).await?;
    let preview = tokio::task::spawn_blocking(move || store.manual_removal_preview(&request.cache_key))
        .await
        .map_err(|error| {
            ApiError::internal(format!("resolved-cache preview task failed: {error}"))
        })?
        .map_err(cache_error)?;
    Ok(Json(RemovalPreviewDto {
        cache_key: preview.cache_key,
        state: preview.state,
        reclaimable_bytes: preview.reclaimable_bytes,
        pins: preview.pins.into(),
        source_unavailable_warning: preview.source_unavailable_warning,
        blocked: preview.blocked,
    }))
}

pub(crate) async fn remove_model_cache_entry(
    State(state): State<AppState>,
    Json(request): Json<CacheKeyRequest>,
) -> Result<Json<RemovalOutcomeDto>, ApiError> {
    let store = open_mutable(&state).await?;
    let now = sceneworks_core::time::now_unix_seconds().max(0) as u64;
    let outcome = tokio::task::spawn_blocking(move || store.remove_entry(&request.cache_key, now))
        .await
        .map_err(|error| ApiError::internal(format!("resolved-cache removal task failed: {error}")))?
        .map_err(cache_error)?;
    Ok(Json(RemovalOutcomeDto {
        cache_key: outcome.cache_key,
        reclaimed_bytes: outcome.reclaimed_bytes,
        source_unavailable_warning: outcome.source_unavailable_warning,
    }))
}

/// "Keep locally" / "Allow automatic removal". This sets the ARTIFACT pin, which is the user's own
/// intent for this bundle; per-model owner pins are held by running loads and are not the user's to
/// clear from here.
pub(crate) async fn set_model_cache_pin(
    State(state): State<AppState>,
    Json(request): Json<CachePinRequest>,
) -> Result<Json<CacheEntryDto>, ApiError> {
    let store = open_mutable(&state).await?;
    let metadata =
        tokio::task::spawn_blocking(move || store.set_artifact_pin(&request.cache_key, request.pinned))
            .await
            .map_err(|error| ApiError::internal(format!("resolved-cache pin task failed: {error}")))?
            .map_err(cache_error)?;
    let owners = repository_owners(&state).await;
    Ok(Json(entry_dto(
        ResolvedCacheEntrySummary {
            cache_key: metadata.cache_key.clone(),
            state: metadata.state.clone(),
            metadata: Some(metadata),
        },
        &owners,
    )))
}
