import { apiFetch } from "../api.js";

// imageRestoration route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const imageRestorationApi = {
  imageRestorationPlan: (token, options = {}) =>
    apiFetch(`/api/v1/image-restoration/plan`, token, options),
};
