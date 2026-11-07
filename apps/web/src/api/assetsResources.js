import { apiFetch } from "../api.js";

// assets route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const assetsApi = {
  listIncludingHidden: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/assets?includeRejected=true&includeTrashed=true`, token, options),
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
