import { apiFetch } from "../api.js";

// credentials route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const credentialsApi = {
  credentials: (token, ...options) =>
    apiFetch(`/api/v1/credentials`, token, ...options),
  credentialsItem: (credentialId, token, options = {}) =>
    apiFetch(`/api/v1/credentials/${encodeURIComponent(credentialId)}`, token, options),
};
