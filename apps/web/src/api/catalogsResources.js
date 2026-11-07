import { apiFetch } from "../api.js";

// catalogs route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const catalogsApi = {
  catalogsItemCurationQuery: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/curation/query`, token, options),
  catalogsItemCurationFacets: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/curation/facets`, token, options),
  catalogsItemSavedViews: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/saved-views`, token, options),
  catalogsItemRecordsItemReview: (catalogId, recordId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/records/${encodeURIComponent(recordId)}/review`, token, options),
  catalogsItemMaterialize: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/materialize`, token, options),
  catalogs: (token, options = {}) =>
    apiFetch(`/api/v1/catalogs`, token, options),
  catalogsItemStatus: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/status`, token, options),
  catalogsAttach: (token, options = {}) =>
    apiFetch(`/api/v1/catalogs/attach`, token, options),
  catalogsItemItem: (catalogId, action, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/${encodeURIComponent(action)}`, token, options),
  catalogsItemAnalyzerConfig: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/analyzer-config`, token, options),
  catalogsItemAnalyze: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/analyze`, token, options),
  catalogsItem: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}`, token, options),
  catalogsItemOnDisk: (catalogId, token, options = {}) =>
    apiFetch(`/api/v1/catalogs/${encodeURIComponent(catalogId)}/on-disk`, token, options),
};
