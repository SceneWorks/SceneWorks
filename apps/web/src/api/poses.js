import { apiFetch } from "../api.js";

export const posesApi = {
  sources: (token, options) => apiFetch("/api/v1/poses/sources", token, options),
  collection: (token, options) => apiFetch("/api/v1/poses", token, options),
};
