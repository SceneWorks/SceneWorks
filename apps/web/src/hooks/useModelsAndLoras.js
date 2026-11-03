import { useProjectScopedList } from "./useProjectScopedList.js";
import { modelsApi } from "../api/modelsResources.js";
import { lorasApi } from "../api/lorasResources.js";
import { useCallback, useState } from "react";
import { appConfirm } from "../appConfirm.jsx";
import {
  borrowedLicenseAcknowledgmentBlocked,
  licenseAckRefusalMessage,
  licenseAcknowledgmentBlocked,
  licenseAcknowledgmentSource,
} from "../licenseAcknowledgment.js";
import { upsertJobNewest } from "../sorters.js";

const maxLoraUploadBytes = 2 * 1024 * 1024 * 1024;

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
  const refreshLoras = useProjectScopedList({
    token, activeProject, activeProjectRef,
    load: (projectId, token, options) => lorasApi.list(projectId ? `?projectId=${encodeURIComponent(projectId)}` : "", token, options),
    setItems: setLoras,
    setError: setLoraError,
    optionalProject: true,
  });

  const deleteModel = useCallback(
    async (model) => {
      let result = await modelsApi.item(model.id, token, {
        method: "DELETE",
      });
      if (result?.trashUnavailable) {
        if (!(await confirmPermanentDelete())) {
          return { cancelled: true };
        }
        result = await modelsApi.removePermanently(model.id, token, { method: "DELETE" });
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
      const result = await modelsApi.variant(model.id, variant, token, { method: "DELETE" });
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
    let result = await lorasApi.itemWithQuery(lora.id, query, token, {
      method: "DELETE",
    });
    if (result?.trashUnavailable) {
      if (!(await confirmPermanentDelete())) {
        return { cancelled: true };
      }
      params.set("permanent", "true");
      result = await lorasApi.item(lora.id, params.toString(), token, { method: "DELETE" });
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
      const updated = await lorasApi.itemWithQuery(lora.id, query, token, {
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
      const result = await lorasApi.embeddedTags(lora.id, query, token);
      return Array.isArray(result?.tags) ? result.tags : [];
    },
    [token, activeProject],
  );

  const createModelImportJob = useCallback(
    async (payload, options = {}) => {
    // Request building is loaded on first use so it stays out of the first-paint bundle.
    const { buildModelImportRequest } = await import("../modelImportRequest.js");
    const { endpoint, body } = buildModelImportRequest(payload, { models });
    const job = await modelsApi.importRequest(endpoint, token, {
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
    [token, models, setJobs, setActiveView, setError],
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
    const job = await lorasApi.import(token, {
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
      try {
        if (await licenseAcknowledgmentBlocked(model, token)) {
          setError(licenseAckRefusalMessage(model));
          return null;
        }
        // sc-8509: install a specific quant tier when the caller passes one (the Models-page tier
        // picker for a quant-matrix model). Absent `variant` installs the model's default tier —
        // the back-compat single-download behavior every other caller relies on.
        const body = { requestedGpu: "auto" };
        if (options.variant) {
          body.variant = options.variant;
        }
        if (options.componentId) {
          body.componentId = options.componentId;
        }
        // Co-requisite choice groups (sc-22998 / sc-23000): e.g. YuE2's `{ decoder: "legacy" }`
        // installs the legacy decoder add-on instead of the manifest default. Sent only when a
        // caller names one, so every other download's body is unchanged.
        if (options.choices && typeof options.choices === "object" && Object.keys(options.choices).length) {
          body.choices = options.choices;
        }
        // The server reads its current terms-bound record. No new acceptance assertion
        // may be inferred from browser state, even after the preflight succeeded.
        const job = await modelsApi.download(model.id, token, {
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
        const job = await modelsApi.convert(model.id, token, {
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
      try {
        if (await borrowedLicenseAcknowledgmentBlocked(lora, token)) {
          setLoraError(licenseAckRefusalMessage(licenseSource));
          return null;
        }
        const body = { requestedGpu: "auto" };
        const job = await lorasApi.download(lora.id, token, {
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

  const downloadConditionalComponents = useCallback(
    (model, purpose) => modelsApi.conditionalDownload(model, purpose, token, {
      method: "POST", body: JSON.stringify({ licenseAcknowledged: true }),
    }),
    [token],
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
    downloadConditionalComponents,
    createLoraDownloadJob,
    createModelConvertJob,
  };
}
