import { apiFetch } from "../api.js";

// The two catalogs share global/project scope query semantics. Paths are chosen
// here, while callers continue to own their domain names and state.
function catalog(resource) {
  const base = `/api/v1/${resource}`;
  return (id, token, options = {}, { projectId, scope, duplicate = false } = {}) => {
    const params = new URLSearchParams();
    if (scope) params.set("scope", scope);
    if (projectId) params.set("projectId", projectId);
    const path = id == null ? base : `${base}/${encodeURIComponent(id)}${duplicate ? "/duplicate" : ""}`;
    return apiFetch(`${path}${params.size ? `?${params}` : ""}`, token, options);
  };
}
export const recipePresetsRequest = catalog("recipe-presets");
export const promptBatchesRequest = catalog("prompt-batches");
