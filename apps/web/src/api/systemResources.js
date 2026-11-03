import { apiFetch } from "../api.js";

// system route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const systemApi = {
  workers: (token, options) => apiFetch("/api/v1/workers", token, options),
  uiPreferences: (token, options = {}) =>
    apiFetch(`/api/v1/ui-preferences`, token, options),
  health: (token, options = {}) =>
    apiFetch(`/api/v1/health`, token, options),
  capabilitiesMac: (token, options = {}) =>
    apiFetch(`/api/v1/capabilities/mac`, token, options),
  hostCapabilities: (token, ...options) =>
    apiFetch(`/api/v1/host-capabilities`, token, ...options),
  workerRestart: (token, options = {}) =>
    apiFetch(`/api/v1/worker/restart`, token, options),
};
