import { apiFetch } from "../api.js";

// training route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const trainingApi = {
  targets: (token, options = {}) =>
    apiFetch(`/api/v1/training/targets`, token, options),
  presets: (token, options = {}) =>
    apiFetch(`/api/v1/training/presets`, token, options),
  datasets: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets`, token, options),
  dataset: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}`, token, options),
  readiness: (projectId, datasetId, querySuffix, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/readiness${querySuffix}`, token, options),
  qualityAck: (projectId, datasetId, itemId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/items/${encodeURIComponent(itemId)}/quality-ack`, token, options),
  uploads: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/uploads`, token, options),
  ltxPreparedBundle: (projectId, datasetId, itemId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/items/${encodeURIComponent(itemId)}/ltx-prepared-bundle`, token, options),
  batchRename: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/batch-rename`, token, options),
  captionSidecars: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/caption-sidecars`, token, options),
  captionJobs: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/caption-jobs`, token, options),
  parquetImportJobs: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/parquet-import-jobs`, token, options),
  upscaleJobs: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/upscale-jobs`, token, options),
  analysisJobs: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/analysis-jobs`, token, options),
  batchDescribeJobs: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/batch-describe-jobs`, token, options),
  defaultDescribePrompt: (token, options = {}) =>
    apiFetch(`/api/v1/training/describe-prompt/default`, token, options),
  describePromptPresets: (token, options = {}) =>
    apiFetch(`/api/v1/training/describe-prompt/presets`, token, options),
  describePromptPreset: (presetId, token, options = {}) =>
    apiFetch(`/api/v1/training/describe-prompt/presets/${encodeURIComponent(presetId)}`, token, options),
  faceAnalysisJobs: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/face-analysis-jobs`, token, options),
  subjectMaskJobs: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/subject-mask-jobs`, token, options),
  subjectMasks: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/subject-masks`, token, options),
  subjectMask: (projectId, datasetId, itemId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/items/${encodeURIComponent(itemId)}/subject-mask`, token, options),
  smartCrop: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/smart-crop`, token, options),
  stripExif: (projectId, datasetId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/datasets/${encodeURIComponent(datasetId)}/strip-exif`, token, options),
  jobs: (projectId, token, options = {}) =>
    apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/training/jobs`, token, options),
};
