import { apiFetch } from "../api.js";

export const metricsApi = {
  list: (query = "", token, options) => apiFetch(`/api/v1/metrics${query}`, token, options),
};
