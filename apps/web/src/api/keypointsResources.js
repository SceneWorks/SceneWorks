import { apiFetch } from "../api.js";

// keypoints route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const keypointsApi = {
  presets: (token, options = {}) =>
    apiFetch(`/api/v1/keypoints/presets`, token, options),
  collections: (token, options = {}) =>
    apiFetch(`/api/v1/keypoints/collections`, token, options),
  sources: (token, options = {}) =>
    apiFetch(`/api/v1/keypoints/sources`, token, options),
  create: (token, options = {}) =>
    apiFetch(`/api/v1/keypoints`, token, options),
  defaultCollection: (collectionId, token, options = {}) =>
    apiFetch(`/api/v1/keypoints/collections/${encodeURIComponent(collectionId)}/default`, token, options),
  collection: (collectionId, token, options = {}) =>
    apiFetch(`/api/v1/keypoints/collections/${encodeURIComponent(collectionId)}`, token, options),
};
