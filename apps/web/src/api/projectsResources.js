import { apiFetch } from "../api.js";

// projects route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const projectsApi = {
  projects: (token, options = {}) =>
    apiFetch(`/api/v1/projects`, token, options),
};
