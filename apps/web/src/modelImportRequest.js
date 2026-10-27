import { licenseAckRefusalMessage, licenseAcknowledgmentBlocked, requiresLicenseAcknowledgment } from "./licenseAcknowledgment.js";

const maxModelUploadBytes = 256 * 1024 * 1024 * 1024;

function uploadLimitLabel(bytes) {
  const gib = bytes / (1024 * 1024 * 1024);
  return Number.isInteger(gib) ? `${gib}GB` : `${gib.toFixed(1)}GB`;
}

// Builds a model import (or repair) request. Loaded on first use by useModelsAndLoras so the
// import-only request shape stays out of the first-paint bundle (bundle-budgets.json `initial`).
export function buildModelImportRequest(payload, { models }) {
  const { file, repairModelId, ...metadata } = payload;
  // Fine-tune component acquisition is the selected base's ordinary download workflow. Reuse
  // consent taken on its existing model card, for both JSON/link and multipart/upload requests.
  const base = models.find((model) => model.id === metadata.baseModel);
  if (requiresLicenseAcknowledgment(base)) {
    if (licenseAcknowledgmentBlocked(base)) {
      throw new Error(licenseAckRefusalMessage(base));
    }
    metadata.licenseAcknowledged = true;
  }
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
  const endpoint = repairModelId ? `/api/v1/models/${encodeURIComponent(repairModelId)}/repair` : "/api/v1/models/import";
  return { endpoint, body };
}
