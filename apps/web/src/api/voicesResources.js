import { apiFetch } from "../api.js";

// voices route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const voicesApi = {
  list: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/voices`, token, options),
  item: (projectId, voiceId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/voices/${encodeURIComponent(voiceId)}`, token, options),
};
