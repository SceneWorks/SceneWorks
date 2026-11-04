import { apiFetch } from "../api.js";

// modelCache route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const modelCacheApi = {
  modelCache: (token, ...options) =>
    apiFetch(`/api/v1/model-cache`, token, ...options),
  modelCacheRemovalPreview: (token, options = {}) =>
    apiFetch(`/api/v1/model-cache/removal-preview`, token, options),
  modelCacheRemove: (token, options = {}) =>
    apiFetch(`/api/v1/model-cache/remove`, token, options),
  modelCachePin: (token, options = {}) =>
    apiFetch(`/api/v1/model-cache/pin`, token, options),
};
