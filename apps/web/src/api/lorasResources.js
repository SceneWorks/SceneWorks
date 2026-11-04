import { apiFetch } from "../api.js";

// loras route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const lorasApi = {
  list: (query = "", token, options) => apiFetch(`/api/v1/loras${query}`, token, options),
  itemWithQuery: (loraId, querySuffix, token, options = {}) =>
    apiFetch(`/api/v1/loras/${encodeURIComponent(loraId)}${querySuffix}`, token, options),
  item: (loraId, querySuffix, token, options = {}) =>
    apiFetch(`/api/v1/loras/${encodeURIComponent(loraId)}?${querySuffix}`, token, options),
  embeddedTags: (loraId, querySuffix, token, options = {}) =>
    apiFetch(`/api/v1/loras/${encodeURIComponent(loraId)}/embedded-tags${querySuffix}`, token, options),
  import: (token, options = {}) =>
    apiFetch(`/api/v1/loras/import`, token, options),
  download: (loraId, token, options = {}) =>
    apiFetch(`/api/v1/loras/${encodeURIComponent(loraId)}/download`, token, options),
};
