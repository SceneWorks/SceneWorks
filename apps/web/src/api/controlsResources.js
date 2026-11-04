import { apiFetch } from "../api.js";

// controls route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const controlsApi = {
  controlOverlays: (querySuffix, token, options = {}) =>
    apiFetch(`/api/v1/control-overlays?${querySuffix}`, token, options),
};
