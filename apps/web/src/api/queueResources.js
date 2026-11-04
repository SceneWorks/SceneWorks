import { apiFetch } from "../api.js";

// queue route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const queueApi = {
  queue: (token, options = {}) =>
    apiFetch(`/api/v1/queue`, token, options),
};
