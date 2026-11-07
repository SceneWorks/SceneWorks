import { apiFetch } from "../api.js";

// personTracks route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const personTracksApi = {
  list: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/person-tracks`, token, options),
  detect: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/person-tracks/detections`, token, options),
  jobs: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/person-tracks/jobs`, token, options),
  corrections: (projectId, personTrackId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/person-tracks/${encodeURIComponent(personTrackId)}/corrections`, token, options),
};
