import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { isCurrentProjectRequest } from "../appStateHelpers.js";
import { refreshFailure, refreshSuccess } from "../refreshResult.js";
import { appConfirm } from "../appConfirm.jsx";
import {
  borrowedLicenseAcknowledgmentBlocked,
  licenseAckRefusalMessage,
  licenseAcknowledgmentBlocked,
  licenseAcknowledgmentSource,
  requiresLicenseAcknowledgment,
} from "../licenseAcknowledgment.js";
import { upsertJobNewest } from "../sorters.js";

const maxLoraUploadBytes = 2 * 1024 * 1024 * 1024;
const maxModelUploadBytes = 256 * 1024 * 1024 * 1024;

function uploadLimitLabel(bytes) {
  const gib = bytes / (1024 * 1024 * 1024);
  return Number.isInteger(gib) ? `${gib}GB` : `${gib.toFixed(1)}GB`;
}

// A delete first tries to move the artifacts to the OS trash (Recycle Bin / Trash).
// When that fails the API removes nothing and returns `trashUnavailable`; the user is
// then asked whether to fall back to a permanent delete. Resolves true to proceed.
// Routed through the desktop-safe appConfirm (sc-12068) — window.confirm silently
// no-ops in the Tauri WebView; the delete actions below await this Promise<boolean>.
export const TRASH_UNAVAILABLE_CONFIRM = "Cannot move to trash. Continue to permanently delete.";
function confirmPermanentDelete() {
  return appConfirm({
    title: "Move to trash failed",
    message: TRASH_UNAVAILABLE_CONFIRM,
    confirmLabel: "Delete permanently",
    cancelLabel: "Cancel",
    tone: "danger",
  });
}

// Owns the model + LoRA catalogs and their import/download/convert/delete actions.
// Extracted from App.jsx (sc-1651). Models and LoRAs are coupled (a LoRA delete/
// import re-pulls both via the lora overlay), so they share one hook. App keeps the
// cross-cutting orchestrators — refreshData (bulk loader; seeds models+loras through
// the returned setters) and refreshDataWithLoraOverlay (refreshData + refreshLoras,
// also called by the SSE handler) — and passes them in. Both props MUST be
// identity-stable (sc-8811): they are useCallback deps of deleteModel/deleteLora,
// which sit in appContextValue's dependency array, so an unstable prop rebuilds the
// context value every App render and defeats the sc-4194 memoization. App passes
// ref-delegating useCallbacks for both.
export function useModelsAndLoras({
  token,
  activeProject,
  activeProjectRef,
  setError,
  setLoraError = setError,
  setJobs,
  setActiveView,
  refreshData,
  refreshDataWithLoraOverlay,
}) {
  const [models, setModels] = useState([]);
  const [loras, setLoras] = useState([]);

  // sc-4194: actions wrapped in useCallback so their identity is stable across App's
  // SSE-driven re-renders, letting appContextValue memoize.
  const refreshLoras = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      if (
        projectId &&
        !isCurrentProjectRequest(activeProject?.id ?? null, projectId)
      ) {
        return refreshFailure("stale");
      }
      try {
        const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
        const items = await apiFetch(`/api/v1/loras${query}`, token, { signal });
        // sc-8858: an SSE-triggered refresh for a specific project can resolve after
        // the user switches away; committing then would clobber the new project's
        // LoRA overlay with the old one's. Drop the stale response — mirrors
        // refreshTimelines' guard (useTimelines.js). Only project-scoped refreshes
        // are guarded; a global refresh (projectId undefined) still commits.
        if (
          projectId &&
          !isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId)
        ) {
          return refreshFailure("stale");
        }
        setLoras(items);
        setLoraError("");
        return refreshSuccess(items);
      } catch (err) {
        if (
          projectId &&
          !isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId)
        ) {
          return refreshFailure("stale", err);
        }
        if (isAbortError(err)) return refreshFailure("aborted", err);
        setLoraError(err.message);
        return refreshFailure("error", err);
      }
    },
    [token, activeProject, activeProjectRef, setLoraError],
  );

  const deleteModel = useCallback(
    async (model) => {
      let result = await apiFetch(`/api/v1/models/${encodeURIComponent(model.id)}`, token, {
        method: "DELETE",
      });
      if (result?.trashUnavailable) {
        if (!(await confirmPermanentDelete())) {
          return { cancelled: true };
        }
        result = await apiFetch(
          `/api/v1/models/${encodeURIComponent(model.id)}?permanent=true`,
          token,
          { method: "DELETE" },
        );
      }
      if (result.removedManifestEntry) {
        setModels((items) => items.filter((item) => item.id !== model.id));
      }
      setError("");
      await refreshData();
      return result;
    },
    [token, setError, refreshData],
  );

  // Delete ONE installed quant tier of a model and reclaim its disk (sc-12024). Counterpart to
  // createModelDownloadJob's per-tier install: hits DELETE …/variants/:variant, which removes only
  // that tier's files/blobs and NEVER the registry entry. So — unlike deleteModel — the model card
  // stays put; the catalog refetch flips the tier from "installed" to "not installed". Unlike
  // deleteModel this deletes PERMANENTLY (the backend never uses the OS trash for a tier): a tier is
  // many loose HF-cache blobs, so trashing them one-by-one drove a macOS per-file permission-prompt
  // loop (sc-12088). One DELETE; the confirm dialog makes the permanence explicit.
  const deleteModelVariant = useCallback(
    async (model, variant) => {
      const result = await apiFetch(
        `/api/v1/models/${encodeURIComponent(model.id)}/variants/${encodeURIComponent(variant)}`,
        token,
        { method: "DELETE" },
      );
      setError("");
      await refreshData();
      return result;
    },
    [token, setError, refreshData],
  );

  const deleteLora = useCallback(
    async (lora) => {
    const params = new URLSearchParams();
    if (lora.scope) {
      params.set("scope", lora.scope);
    }
    if (lora.scope === "project" && activeProject?.id) {
      params.set("projectId", activeProject.id);
    }
    const query = params.toString() ? `?${params.toString()}` : "";
    let result = await apiFetch(`/api/v1/loras/${encodeURIComponent(lora.id)}${query}`, token, {
      method: "DELETE",
    });
    if (result?.trashUnavailable) {
      if (!(await confirmPermanentDelete())) {
        return { cancelled: true };
      }
      params.set("permanent", "true");
      result = await apiFetch(
        `/api/v1/loras/${encodeURIComponent(lora.id)}?${params.toString()}`,
        token,
        { method: "DELETE" },
      );
    }
    if (result.removedManifestEntry) {
      setLoras((items) => items.filter((item) => item.id !== lora.id || item.scope !== lora.scope));
    }
    setLoraError("");
    await refreshDataWithLoraOverlay(activeProject?.id);
    return result;
    },
    [token, activeProject, setLoraError, refreshDataWithLoraOverlay],
  );

  // Edit a catalog LoRA's trigger keywords / notes after import (epic 10328). Only
  // the fields present are sent; the backend leaves the rest untouched.
  const updateLora = useCallback(
    async (lora, updates) => {
      const params = new URLSearchParams();
      if (lora.scope) {
        params.set("scope", lora.scope);
      }
      if (lora.scope === "project" && activeProject?.id) {
        params.set("projectId", activeProject.id);
      }
      const query = params.toString() ? `?${params.toString()}` : "";
      const updated = await apiFetch(`/api/v1/loras/${encodeURIComponent(lora.id)}${query}`, token, {
        method: "PATCH",
        body: JSON.stringify(updates),
      });
      setLoraError("");
      await refreshDataWithLoraOverlay(activeProject?.id);
      return updated;
    },
    [token, activeProject, setLoraError, refreshDataWithLoraOverlay],
  );

  // Best-effort trigger-keyword suggestions read from the installed LoRA's embedded
  // ss_tag_frequency metadata; returns [] when unavailable (epic 10328).
  const fetchLoraEmbeddedTags = useCallback(
    async (lora) => {
      const params = new URLSearchParams();
      if (lora.scope) {
        params.set("scope", lora.scope);
      }
      if (lora.scope === "project" && activeProject?.id) {
        params.set("projectId", activeProject.id);
      }
      const query = params.toString() ? `?${params.toString()}` : "";
      const result = await apiFetch(
        `/api/v1/loras/${encodeURIComponent(lora.id)}/embedded-tags${query}`,
        token,
      );
      return Array.isArray(result?.tags) ? result.tags : [];
    },
    [token, activeProject],
  );

  const createModelImportJob = useCallback(
    async (payload, options = {}) => {
    const { file, ...metadata } = payload;
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
    const job = await apiFetch("/api/v1/models/import", token, {
      method: "POST",
      body,
    });
    setJobs((items) => upsertJobNewest(items, job));
    if (options.navigateToQueue ?? false) {
      setActiveView("Queue");
    }
    setError("");
    return job;
    },
    [token, setJobs, setActiveView, setError],
  );

  const createLoraImportJob = useCallback(
    async (payload, options = {}) => {
    if (payload.scope === "project" && !activeProject) {
      throw new Error("Create or open a project first.");
    }
    const { file, secondaryFile, ...metadata } = payload;
    if (file?.size > maxLoraUploadBytes) {
      throw new Error("Uploaded LoRA file exceeds the 2GB limit");
    }
    if (secondaryFile?.size > maxLoraUploadBytes) {
      throw new Error("Uploaded low-noise expert file exceeds the 2GB limit");
    }
    let body;
    if (file) {
      body = new FormData();
      Object.entries({
        ...metadata,
        projectId: metadata.scope === "project" ? activeProject.id : null,
        projectName: metadata.scope === "project" ? activeProject.name : null,
      }).forEach(([key, value]) => {
        if (value != null && value !== "") {
          body.append(key, value);
        }
      });
      body.append("file", file);
      // Wan A14B MoE pair (sc-1991): the low-noise expert half rides along as a
      // second file part the API stages under the high/low_noise convention.
      if (secondaryFile) {
        body.append("secondaryFile", secondaryFile);
      }
    } else {
      body = JSON.stringify({
        ...metadata,
        projectId: metadata.scope === "project" ? activeProject.id : null,
        projectName: metadata.scope === "project" ? activeProject.name : null,
      });
    }
    const job = await apiFetch("/api/v1/loras/import", token, {
      method: "POST",
      body,
    });
    setJobs((items) => upsertJobNewest(items, job));
    if (options.navigateToQueue ?? false) {
      setActiveView("Queue");
    }
    setLoraError("");
    return job;
    },
    [token, activeProject, setJobs, setActiveView, setLoraError],
  );

  const createModelDownloadJob = useCallback(
    async (model, options = {}) => {
      // Licence-acknowledgment CHOKE POINT (sc-17227). Every download-starting surface funnels
      // through here — the Models screen, the Simple UI's model manager, the first-run Setup
      // Wizard, the studio availability gates, the workflow drop, the Update button — so this is
      // the one place the gate binds all of them. Enforcing it per-surface is how MiniMax-H3
      // shipped downloadable from three screens that render no licence UI at all.
      //
      // Refusing here rather than letting the request through is not belt-and-braces: for a
      // `requiresLicenseAcknowledgment` model the repo is PUBLIC, so nothing upstream refuses an
      // unacknowledged fetch and the weights would simply land. The API refuses the same request
      // independently (`apps/rust-api/src/models.rs`) for clients this code never runs in.
      if (licenseAcknowledgmentBlocked(model)) {
        setError(licenseAckRefusalMessage(model));
        return null;
      }
      try {
        // sc-8509: install a specific quant tier when the caller passes one (the Models-page tier
        // picker for a quant-matrix model). Absent `variant` installs the model's default tier —
        // the back-compat single-download behavior every other caller relies on.
        const body = { requestedGpu: "auto" };
        if (options.variant) {
          body.variant = options.variant;
        }
        // Carry the acknowledgment to the API, which refuses the download without it. Sent only
        // when the model actually requires one, so no other download's body changes shape.
        if (requiresLicenseAcknowledgment(model)) {
          body.licenseAcknowledged = true;
        }
        const job = await apiFetch(`/api/v1/models/${model.id}/download`, token, {
          method: "POST",
          body: JSON.stringify(body),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, setJobs, setError],
  );

  const createModelConvertJob = useCallback(
    async (model) => {
      try {
        const job = await apiFetch(`/api/v1/models/${model.id}/convert`, token, {
          method: "POST",
          body: JSON.stringify({ requestedGpu: "auto" }),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, setJobs, setError],
  );

  // Built-in LoRA explicit download (sc-5944): queues a `lora_download` job that fetches
  // the catalog LoRA's HF files into the cache, flipping its installState to "installed".
  // Mirrors createModelDownloadJob.
  const createLoraDownloadJob = useCallback(
    async (lora) => {
      // The LoRA half of the licence CHOKE POINT (sc-17227). `POST /api/v1/loras/:id/download`
      // gates on the repo the catalog row resolves to, so a LoRA whose `source.repo` is a
      // restricted repo is refused exactly as the model download is — and without this the refusal
      // is a bare 403 no shipped surface can clear: nothing sends the assertion, and no LoRA row
      // renders a licence checkbox. The server stamps the gating model onto the row; the gate here
      // reads that stamp and the ack the Models screen already persists, so accepting the licence
      // once on the model's card is what makes its LoRAs downloadable.
      const licenseSource = licenseAcknowledgmentSource(lora);
      if (borrowedLicenseAcknowledgmentBlocked(lora)) {
        setLoraError(licenseAckRefusalMessage(licenseSource));
        return null;
      }
      try {
        const body = { requestedGpu: "auto" };
        // Sent only for a row the server flagged, so no other LoRA download changes shape.
        if (licenseSource) {
          body.licenseAcknowledged = true;
        }
        const job = await apiFetch(`/api/v1/loras/${encodeURIComponent(lora.id)}/download`, token, {
          method: "POST",
          body: JSON.stringify(body),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setLoraError("");
        return job;
      } catch (err) {
        // The server probes installState live from the HF cache; our badge is a snapshot
        // from the last catalog fetch. When the cache is filled out-of-band (the on-demand
        // pull at first generation, another client) the row is already "installed" server-
        // side while we still render "Not Installed", so the click lands on a Download
        // button that cannot succeed. That disagreement is benign and self-correcting:
        // resync the catalog so the badge flips, rather than showing an error that
        // contradicts what the user is looking at.
        if (err?.code === "lora_already_installed") {
          await refreshLoras();
          return null;
        }
        setLoraError(err.message);
        return null;
      }
    },
    [token, setJobs, setLoraError, refreshLoras],
  );

  return {
    models,
    setModels,
    loras,
    setLoras,
    refreshLoras,
    deleteModel,
    deleteModelVariant,
    deleteLora,
    updateLora,
    fetchLoraEmbeddedTags,
    createModelImportJob,
    createLoraImportJob,
    createModelDownloadJob,
    createLoraDownloadJob,
    createModelConvertJob,
  };
}
