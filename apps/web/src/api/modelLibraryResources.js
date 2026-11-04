import { apiFetch } from "../api.js";

// modelLibrary route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const modelLibraryApi = {
  modelLibrary: (token, options = {}) =>
    apiFetch(`/api/v1/model-library`, token, options),
  modelLibraryRelocate: (token, options = {}) =>
    apiFetch(`/api/v1/model-library/relocate`, token, options),
};
