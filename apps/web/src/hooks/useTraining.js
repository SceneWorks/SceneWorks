import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { useProjectScopedList } from "./useProjectScopedList.js";
import { trainingApi } from "../api/trainingResources.js";
import { useCallback, useState } from "react";
import { isCurrentProjectRequest } from "../appStateHelpers.js";
import { rethrowUnlessPrompted } from "../modelLibrary.js";
import { upsertJobNewest } from "../sorters.js";

// Owns the project's training-dataset state plus dataset CRUD, caption sidecar/job,
// and training-job creation. Extracted from App.jsx (sc-1651). The training target/
// preset catalogs stay in App (bulk-loaded by refreshData) — only the project-scoped
// dataset workflow lives here. Shared concerns (token, activeProject, error, jobs) are
// passed in; caption/training jobs push onto the shared jobs list via setJobs.
//
// sc-4194: actions are wrapped in useCallback so their identity is stable across App's
// SSE-driven re-renders, letting appContextValue memoize.
export function useTraining({ token, activeProject, activeProjectRef, setError, setJobs }) {
  const [trainingDatasets, setTrainingDatasets] = useState([]);
  const [trainingDatasetsProjectId, setTrainingDatasetsProjectId] = useState(null);
  const [loadingTrainingDatasets, setLoadingTrainingDatasets] = useState(false);
  const [trainingDatasetsError, setTrainingDatasetsError] = useState("");
  const isCurrentTrainingRequest = useCallback(
    (projectId) =>
      isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId),
    [activeProjectRef],
  );

  const refreshTrainingDatasets = useProjectScopedList({
    token, activeProject, activeProjectRef,
    load: trainingApi.datasets,
    admitFromRef: true,
    setItems: setTrainingDatasets,
    onMissing: () => {
      setTrainingDatasets([]);
      setTrainingDatasetsProjectId(null);
      setTrainingDatasetsError("");
      setLoadingTrainingDatasets(false);
    },
    onStart: () => setLoadingTrainingDatasets(true),
    onSuccess: (_items, projectId) => {
      setTrainingDatasetsProjectId(projectId);
      setTrainingDatasetsError("");
    },
    onError: (error, projectId) => {
      setTrainingDatasets([]);
      setTrainingDatasetsProjectId(projectId);
      setTrainingDatasetsError(error.message);
    },
    onSettled: ({ ownsRequest, aborted }) => {
      if (ownsRequest && !aborted) setLoadingTrainingDatasets(false);
    },
  });

  const loadTrainingDataset = useCallback(
    async (datasetId, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        return null;
      }
      return trainingApi.dataset(projectId, datasetId, token);
    },
    [token, activeProject],
  );

  // Dataset Doctor readiness report (sc-6533/6534). A read over the SAVED dataset
  // — pixel metrics (blur/dup/exposure) need the files on disk — so callers fetch
  // it keyed on the dataset version, not the live unsaved selection. `query` is the
  // pre-built query string (target resolution / kind / min) from readinessQueryParams;
  // `signal` lets a stale fetch be aborted when the target or dataset changes.
  const loadTrainingDatasetReadiness = useCallback(
    async (datasetId, query = "", projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId || !datasetId) {
        return null;
      }
      const suffix = query ? `?${query}` : "";
      return trainingApi.readiness(projectId, datasetId, suffix, token, { signal });
    },
    [token, activeProject],
  );

  // Persist (or clear) a per-image quality override (sc-6534). `checks` is the full set of dismissed
  // findings for the item (the endpoint replaces the stored set); an empty array clears it. A
  // metadata write — it does not bump the dataset version — so callers refetch readiness, not the
  // dataset, afterward.
  const setTrainingDatasetItemQualityAck = useCallback(
    async (datasetId, itemId, checks, options = {}) => {
      // sc-8942 (F-140): single options object. The optional target project is a named
      // `projectId` field (defaulting to the active project) rather than the old
      // type-switching 4th param (string=projectId | object=options), which was unique in
      // the codebase and easy to call wrong — a mistyped arg silently fell through to the
      // active project. No caller ever used the string form, so this is behavior-preserving.
      const { projectId = activeProject?.id, expectedContentHash, expectedCaptionHash } = options ?? {};
      if (!projectId || !datasetId || !itemId) {
        throw new Error("Select a training dataset item first.");
      }
      const body = {
        checks,
        ...(expectedContentHash ? { expectedContentHash } : {}),
        ...(expectedCaptionHash ? { expectedCaptionHash } : {}),
      };
      return trainingApi.qualityAck(projectId, datasetId, itemId, token, { method: "POST", body: JSON.stringify(body) });
    },
    [token, activeProject],
  );

  const createTrainingDataset = useCallback(
    async (payload, projectId = activeProject?.id) => {
      if (!projectId) {
        throw new Error("Create or open a project first.");
      }
      const created = await trainingApi.datasets(projectId, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (isCurrentTrainingRequest(projectId)) {
        await refreshTrainingDatasets(projectId);
      }
      return created;
    },
    [token, activeProject, isCurrentTrainingRequest, refreshTrainingDatasets],
  );

  const uploadTrainingDatasetItem = useCallback(
    async (file, projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId) {
        throw new Error("Create or open a project first.");
      }
      const form = new FormData();
      form.append("file", file);
      return trainingApi.uploads(projectId, token, {
        method: "POST",
        body: form,
        signal,
      });
    },
    [token, activeProject],
  );

  const uploadLtxPreparedBundle = useCallback(
    async (datasetId, itemId, file, projectId = activeProject?.id) => {
      if (!projectId || !datasetId || !itemId) {
        throw new Error("Save the dataset before uploading an LTX prepared bundle.");
      }
      const form = new FormData();
      form.append("file", file);
      const updated = await trainingApi.ltxPreparedBundle(projectId, datasetId, itemId, token, { method: "POST", body: form });
      if (isCurrentTrainingRequest(projectId)) {
        await refreshTrainingDatasets(projectId);
      }
      return updated;
    },
    [token, activeProject, isCurrentTrainingRequest, refreshTrainingDatasets],
  );

  const updateTrainingDataset = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const updated = await trainingApi.dataset(projectId, datasetId, token, {
        method: "PATCH",
        body: JSON.stringify(payload),
      });
      if (isCurrentTrainingRequest(projectId)) {
        await refreshTrainingDatasets(projectId);
      }
      return updated;
    },
    [token, activeProject, isCurrentTrainingRequest, refreshTrainingDatasets],
  );

  const batchRenameTrainingDataset = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const updated = await trainingApi.batchRename(projectId, datasetId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
      if (isCurrentTrainingRequest(projectId)) {
        await refreshTrainingDatasets(projectId);
      }
      return updated;
    },
    [token, activeProject, isCurrentTrainingRequest, refreshTrainingDatasets],
  );

  // Permanently delete a dataset: the DELETE endpoint removes the on-disk dataset root
  // (images, caption sidecars, embedding/face sidecars, manifest) AND the SQLite index
  // row in one cascade. Like create/update, refresh the list afterward so the removed
  // dataset drops out of the picker.
  const deleteTrainingDataset = useCallback(
    async (datasetId, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const result = await trainingApi.dataset(projectId, datasetId, token, { method: "DELETE" });
      if (isCurrentTrainingRequest(projectId)) {
        await refreshTrainingDatasets(projectId);
      }
      return result;
    },
    [token, activeProject, isCurrentTrainingRequest, refreshTrainingDatasets],
  );

  const writeTrainingDatasetCaptionSidecars = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const result = await trainingApi.captionSidecars(projectId, datasetId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
      if (isCurrentTrainingRequest(projectId)) {
        await refreshTrainingDatasets(projectId);
      }
      return result;
    },
    [token, activeProject, isCurrentTrainingRequest, refreshTrainingDatasets],
  );

  const createTrainingDatasetCaptionJob = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const job = await trainingApi.captionJobs(projectId, datasetId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  const createTrainingDatasetParquetImportJob = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select an empty training dataset first.");
      }
      const job = await trainingApi.parquetImportJobs(projectId, datasetId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  // sc-6539: enqueue a Real-ESRGAN upscale over the low-resolution-flagged dataset items.
  const createTrainingDatasetUpscaleJob = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const job = await trainingApi.upscaleJobs(projectId, datasetId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  // sc-6535: enqueue the CLIP embedding pass over the dataset. It produces the embedding sidecar the
  // readiness fold reads for the Variety / aesthetic / off-style-outlier / caption-alignment checks
  // (every kind). Async (GPU); the readiness updates when it finishes.
  const createTrainingDatasetAnalysisJob = useCallback(
    async (datasetId, payload = {}, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const job = await trainingApi.analysisJobs(projectId, datasetId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  // sc-6538: enqueue the SCRFD+ArcFace face pass over the dataset. It produces the face sidecar the
  // readiness fold reads for the Person identity / no-face / small-subject checks. Async (GPU); the
  // readiness updates when it finishes, like the CLIP analysis pass.
  const createTrainingDatasetFaceAnalysisJob = useCallback(
    async (datasetId, payload = {}, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const job = await trainingApi.faceAnalysisJobs(projectId, datasetId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  // sc-2126: enqueue SAM3 subject-mask generation over the dataset (one mask per image, consumed by
  // the masked-loss trainer). Async (GPU); the mask report refreshes when the job completes.
  const createTrainingDatasetSubjectMaskJob = useCallback(
    async (datasetId, payload = {}, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const job = await trainingApi.subjectMaskJobs(projectId, datasetId, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  // sc-2126: the dataset's subject-mask coverage + per-image mask status (path, empty, source).
  const loadTrainingDatasetSubjectMasks = useCallback(
    async (datasetId, projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId || !datasetId) {
        return null;
      }
      return trainingApi.subjectMasks(projectId, datasetId, token, { signal });
    },
    [token, activeProject],
  );

  // sc-2126: replace one image's subject mask with an uploaded file. Returns the updated report.
  const uploadTrainingDatasetSubjectMask = useCallback(
    async (datasetId, itemId, file, projectId = activeProject?.id) => {
      if (!projectId || !datasetId || !itemId) {
        throw new Error("Save the dataset before uploading a subject mask.");
      }
      const form = new FormData();
      form.append("file", file);
      return trainingApi.subjectMask(projectId, datasetId, itemId, token, { method: "POST", body: form });
    },
    [token, activeProject],
  );

  // sc-6539 synchronous one-tap fixes: POST returns { dataset, applied } for immediate refresh.
  const smartCropTrainingDataset = useCallback(
    async (datasetId, itemIds, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const result = await trainingApi.smartCrop(projectId, datasetId, token, { method: "POST", body: JSON.stringify({ itemIds }) });
      setError("");
      return result;
    },
    [token, activeProject, setError],
  );

  const stripExifTrainingDataset = useCallback(
    async (datasetId, itemIds, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const result = await trainingApi.stripExif(projectId, datasetId, token, { method: "POST", body: JSON.stringify(itemIds ? { itemIds } : {}) });
      setError("");
      return result;
    },
    [token, activeProject, setError],
  );

  const createTrainingJob = useCallback(
    async function submitTrainingJob(request, projectId = activeProject?.id) {
      if (!projectId) {
        throw new Error("Select a workspace before creating a training job.");
      }
      let job;
      try {
        job = await trainingApi.jobs(projectId, token, {
          method: "POST",
          body: JSON.stringify(request),
        });
      } catch (error) {
        // A base model on a disconnected library is recoverable, not a failure to print: hand the
        // prompt this exact submission to resume (sc-19709). It owns single-fire, so a reconnect
        // cannot queue two training runs.
        rethrowUnlessPrompted(error, () => submitTrainingJob(request, projectId));
      }
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  const context = useAppDomainValue({
    trainingDatasets,
    trainingDatasetsProjectId,
    loadingTrainingDatasets,
    trainingDatasetsError,
    refreshTrainingDatasets,
    loadTrainingDataset,
    loadTrainingDatasetReadiness,
    setTrainingDatasetItemQualityAck,
    createTrainingDataset,
    uploadTrainingDatasetItem,
    uploadLtxPreparedBundle,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    deleteTrainingDataset,
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingDatasetParquetImportJob,
    createTrainingDatasetUpscaleJob,
    createTrainingDatasetAnalysisJob,
    createTrainingDatasetFaceAnalysisJob,
    createTrainingDatasetSubjectMaskJob,
    loadTrainingDatasetSubjectMasks,
    uploadTrainingDatasetSubjectMask,
    smartCropTrainingDataset,
    stripExifTrainingDataset,
    createTrainingJob
  });
  return { ...context, setTrainingDatasets, setTrainingDatasetsProjectId, setTrainingDatasetsError, context };
}
