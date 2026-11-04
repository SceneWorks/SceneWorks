export function modelImportPath(repairModelId) {
  return repairModelId ? `/api/v1/models/${encodeURIComponent(repairModelId)}/repair` : "/api/v1/models/import";
}
