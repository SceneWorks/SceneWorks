import { apiFetch } from "../api.js";

// assets route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const assetsApi = {
  listIncludingHidden: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets?includeRejected=true&includeTrashed=true`, token, options),
  // Catalog/grid shape: summary items (`listView: "summary"`) without the replay bulk. Open,
  // replay and regenerate paths load the full asset through `item` (see assetCatalog.js).
  listSummaryIncludingHidden: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets?includeRejected=true&includeTrashed=true&view=summary`, token, options),
  // Incremental summary catalog: `since` "" returns everything plus a cursor; a cursor returns
  // only `{ assets, removedIds }` changed after it (`reset: true` means "replace, don't merge").
  // `scope` is "library" (no Character Studio / workflow-intermediate output) or "all". A cursor
  // is only meaningful for the scope that issued it: changing scope needs a fresh `since` "".
  summaryChangesIncludingHidden: (projectId, since, token, options = {}, { scope = "all" } = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets?includeRejected=true&includeTrashed=true&view=summary&scope=${encodeURIComponent(scope)}&since=${encodeURIComponent(since ?? "")}`, token, options),
  status: (projectId, assetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/status`, token, options),
  tags: (projectId, assetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/tags`, token, options),
  moveToLibrary: (projectId, assetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/move-to-library`, token, options),
  moveToCharacter: (projectId, assetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/move-to-character`, token, options),
  item: (projectId, assetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}`, token, options),
  purge: (projectId, assetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/purge`, token, options),
  purgePermanently: (projectId, assetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/purge?permanent=true`, token, options),
  list: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets`, token, options),
};
