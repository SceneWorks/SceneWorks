import { apiFetch } from "../api.js";

// models route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const modelsApi = {
  importRequest: (endpoint, token, options) => apiFetch(endpoint, token, options),
  list: (token, options = {}) =>
    apiFetch(`/api/v1/models`, token, options),
  libraryRoots: (token, options = {}) =>
    apiFetch(`/api/v1/models/library-roots`, token, options),
  libraryRoot: (libraryRootId, token, options = {}) =>
    apiFetch(`/api/v1/models/library-roots/${encodeURIComponent(libraryRootId)}`, token, options),
  scanLibraryRoot: (libraryRootId, token, options = {}) =>
    apiFetch(`/api/v1/models/library-roots/${encodeURIComponent(libraryRootId)}/scan`, token, options),
  rescanLibraryRoot: (libraryRootId, token, options = {}) =>
    apiFetch(`/api/v1/models/library-roots/${encodeURIComponent(libraryRootId)}/rescan`, token, options),
  mossImageEditDownload: (modelId, token, options = {}) =>
    modelsApi.conditionalDownload(modelId, "moss_image_edit", token, options),
  item: (modelId, token, options = {}) =>
    apiFetch(`/api/v1/models/${encodeURIComponent(modelId)}`, token, options),
  removePermanently: (modelId, token, options = {}) =>
    apiFetch(`/api/v1/models/${encodeURIComponent(modelId)}?permanent=true`, token, options),
  variant: (modelId, variantId, token, options = {}) =>
    apiFetch(`/api/v1/models/${encodeURIComponent(modelId)}/variants/${encodeURIComponent(variantId)}`, token, options),
  download: (modelId, token, options = {}) =>
    apiFetch(`/api/v1/models/${encodeURIComponent(modelId)}/download`, token, options),
  convert: (modelId, token, options = {}) =>
    apiFetch(`/api/v1/models/${encodeURIComponent(modelId)}/convert`, token, options),
  conditionalDownload: (modelId, conditionalComponentId, token, options = {}) =>
    apiFetch(`/api/v1/models/${encodeURIComponent(modelId)}/conditional-components/${encodeURIComponent(conditionalComponentId)}/download`, token, options),
};
