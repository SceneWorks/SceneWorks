import { apiFetch } from "../api.js";

export const logsApi = {
  list: (query = "", token, options) => apiFetch(`/api/v1/logs${query}`, token, options),
};
