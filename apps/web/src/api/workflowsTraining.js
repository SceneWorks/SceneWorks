import { apiFetch } from "../api.js";

// ---- Workflow inspect (sc-15950 / sc-15951) -------------------------------------
// Read a file's embedded workflow WITHOUT importing it: no asset, no job, no project
// mutation. `projectId` is optional and only widens the LoRA/preset lookups the
// resolution report is built against.
//
// Resolves to `{ status, workflow, resolution, detail }` — branch on `status`, since
// `workflow`/`resolution` are always present and are `null` for `no_workflow`, which is
// the common case for any image that did not come from SceneWorks.
//
// Failures arrive as the usual `ApiError`. A malformed multipart is rejected by axum
// with a PLAIN-TEXT body and no `code`; `apiFetch` already degrades a non-JSON body to
// `payload = null`, so such a rejection surfaces as a status with `code === undefined`
// rather than throwing inside the parse.
export async function inspectWorkflowFile(file, { projectId, token, signal } = {}) {
  const body = new FormData();
  body.append("file", file);
  if (projectId) {
    body.append("projectId", projectId);
  }
  return apiFetch("/api/v1/workflows/inspect", token, { method: "POST", body, signal });
}

// The recipe an ALREADY-IMPORTED asset is carrying, and what this install can do about it right
// now (sc-15952). Same `{ status, workflow, resolution }` body as `inspectWorkflowFile`, so a
// reader branches on `status` either way.
//
// The envelope itself needs no request — it rides on the asset at `extra.importedWorkflow`
// (sc-15949). The RESOLUTION is what this fetches: it is computed against catalogs that live
// behind the API, and it stops being true the moment a download finishes, which is why this is a
// plain re-GET rather than something cached beside the envelope.
export async function fetchAssetWorkflow(projectId, assetId, { token, signal } = {}) {
  return apiFetch(
    `/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/workflow`,
    token,
    { signal },
  );
}

// ---- Training data-walk preflight (sc-25685) ------------------------------------
// Which items of an N-item dataset a training target's sampler will actually visit at a given
// batch size. Iris' upstream ranged sampler walks whole 640-item chunks, so a dataset of
// 641–1279 items only trains on the first 640; the Studio shows `warning` inline before launch.
// Resolves to `{ applies, items, covered, unusedItems, chunks, warning }`.
// Iris curricula (sc-25689): the official stage presets, the pre-launch plan check, a run's metrics.
export const fetchIrisStagePresets = (token, signal) =>
  apiFetch("/api/v1/training/iris/stage-presets", token, { signal });
export const checkIrisPlan = (targetId, config, token, signal) =>
  apiFetch("/api/v1/training/iris/plan-check", token, { method: "POST", body: JSON.stringify({ targetId, config }), signal });
export const fetchIrisMetrics = (jobId, token, signal) =>
  apiFetch(`/api/v1/training/jobs/${encodeURIComponent(jobId)}/iris-metrics`, token, { signal });

export async function fetchTrainingDataWalk({ targetId, items, batchSize }, { token, signal } = {}) {
  const params = new URLSearchParams({
    targetId: String(targetId ?? ""),
    items: String(items ?? 0),
    batchSize: String(batchSize ?? 1),
  });
  return apiFetch(`/api/v1/training/data-walk?${params.toString()}`, token, { signal });
}

// ---- Iris depth / restoration training datasets (sc-25687 / sc-25688) -------------------------
function trainingDatasetPath(projectId, datasetId) {
  return `/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}`;
}

// Attach (`file` given) or remove (`file` null) one item's depth map / mask / degraded input.
// `fields` (depth only): optional `scale`, `minDepth`, `maxDepth`. Resolves to the updated dataset.
export async function irisItemSidecar({ projectId, datasetId, itemId, kind, file = null, fields = {} }, { token } = {}) {
  const path = `${trainingDatasetPath(projectId, datasetId)}/items/${encodeURIComponent(itemId)}/iris-sidecar/${encodeURIComponent(kind)}`;
  if (!file) return apiFetch(path, token, { method: "DELETE" });
  const body = new FormData();
  body.append("file", file);
  for (const [key, value] of Object.entries(fields)) {
    if (String(value ?? "").trim()) body.append(key, String(value).trim());
  }
  return apiFetch(path, token, { method: "POST", body });
}

// The pre-launch dataset report of a depth / restoration run (`config` = the submit snapshot's).
export async function checkIrisDataset({ projectId, datasetId, targetId, config }, { token } = {}) {
  return apiFetch(`${trainingDatasetPath(projectId, datasetId)}/iris-check`, token, {
    method: "POST",
    body: JSON.stringify({ targetId, config }),
  });
}
