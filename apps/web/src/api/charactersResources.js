import { apiFetch } from "../api.js";

// characters route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const charactersApi = {
  list: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters`, token, options),
  item: (projectId, characterId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}`, token, options),
  archive: (projectId, characterId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}/archive`, token, options),
  listIncludingArchived: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters?include_archived=true`, token, options),
  references: (projectId, characterId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}/references`, token, options),
  reference: (projectId, characterId, referenceId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}/references/${encodeURIComponent(referenceId)}`, token, options),
  looks: (projectId, characterId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}/looks`, token, options),
  look: (projectId, characterId, lookId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}/looks/${encodeURIComponent(lookId)}`, token, options),
  loras: (projectId, characterId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}/loras`, token, options),
  lora: (projectId, characterId, loraId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}/loras/${encodeURIComponent(loraId)}`, token, options),
  testJobs: (projectId, characterId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/characters/${encodeURIComponent(characterId)}/test-jobs`, token, options),
};
