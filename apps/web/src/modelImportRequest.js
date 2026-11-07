import { modelImportPath } from "./api/modelImport.js";
import { bytesToGib } from "./modelCache.js";

const maxModelUploadBytes = 256 * 1024 * 1024 * 1024;

function uploadLimitLabel(bytes) {
  const gib = bytesToGib(bytes);
  return Number.isInteger(gib) ? `${gib}GB` : `${gib.toFixed(1)}GB`;
}

// Builds a model import (or repair) request. Loaded on first use by useModelsAndLoras so the
// import-only request shape stays out of the first-paint bundle (bundle-budgets.json `initial`).
export function buildModelImportRequest(payload) {
  const { file, repairModelId, ...metadata } = payload;
  if (file?.size > maxModelUploadBytes) {
    throw new Error(`Uploaded model file exceeds the ${uploadLimitLabel(maxModelUploadBytes)} limit`);
  }
  let body;
  if (file) {
    body = new FormData();
    // Multipart fields are appended verbatim — the backend reads them by literal field name
    // (models.rs `model_import_request_from_multipart`) with none of the `#[serde(alias)]`
    // tolerance the JSON branch below gets. Callers must key metadata by the backend's own
    // multipart field names (e.g. `type`, not `modelType`) or the value is silently dropped
    // and the import defaults to `image` (sc-14020).
    Object.entries(metadata).forEach(([key, value]) => {
      if (value == null || value === "") {
        return;
      }
      // The multipart parser reads `source` as a JSON DOCUMENT (models.rs
      // `model_import_request_from_multipart` runs `serde_json::from_str` on it), so an object
      // appended verbatim would arrive as the literal "[object Object]" and be refused as an
      // invalid import source. Every other field is a scalar and is unchanged (epic 20398).
      body.append(key, typeof value === "object" && !(value instanceof Blob) ? JSON.stringify(value) : value);
    });
    body.append("file", file);
  } else {
    body = JSON.stringify(metadata);
  }
  const endpoint = modelImportPath(repairModelId);
  return { endpoint, body };
}
