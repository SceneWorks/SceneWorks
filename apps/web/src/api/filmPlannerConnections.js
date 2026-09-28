import { apiFetch } from "../api.js";

const ROOT = "/api/v1/film-planner-connections";

export function listFilmPlannerConnections(token) {
  return apiFetch(ROOT, token);
}

export function saveFilmPlannerConnection(id, connection, token) {
  return apiFetch(`${ROOT}/${encodeURIComponent(id)}`, token, {
    method: "PUT",
    body: JSON.stringify(connection),
  });
}

export function testFilmPlannerConnection(id, token) {
  return apiFetch(`${ROOT}/${encodeURIComponent(id)}/test`, token, { method: "POST" });
}
