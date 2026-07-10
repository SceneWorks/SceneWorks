import { useEffect } from "react";
import { apiFetch, eventUrl } from "../api.js";
import { sortWorkers, upsertJobNewest } from "../sorters.js";
import { terminalStatuses } from "../constants.js";
import {
  failedJobNotice,
  generatedResultAssetCount,
  noticeKindForJob,
  parseSseJson,
} from "../appHelpers.js";

// Owns the live job/worker/queue SSE stream (the EventSource lifecycle: ticket mint,
// connect, event handlers, exponential-backoff reconnect, teardown). Extracted verbatim
// from App.jsx (sc-9750, F-052 follow-up) so App no longer carries the ~150-line stream
// effect inline. Behavior-preserving — same effect body, same [access.authRequired,
// ready, token] dependency array, same cleanup.
//
// The handlers reach back into App state through the setters/refs/callbacks passed in.
// App feeds them identity-stable (useCallback actions, useState setters, and refs), so
// the effect's deps are exactly the three inputs that must re-subscribe the stream —
// auth mode, readiness, and the token — mirroring the pre-extraction effect. `access`
// is read only for `access.authRequired`; the whole object is passed so the dep array
// (`access.authRequired`) matches the original render-scope reference.
export function useJobEvents({
  access,
  ready,
  token,
  setJobs,
  setWorkers,
  setQueueSummary,
  setLatestGenerationSetId,
  setError,
  pushNotice,
  dismissNoticeKind,
  generatedAssetRefreshesRef,
  refreshAssetsRef,
  refreshDataRef,
  refreshDataWithLoraOverlayRef,
  refreshPersonTracksRef,
  activeProjectRef,
  enqueueTimelineGenerationApply,
  hasVisibleLocalFailure,
}) {
  useEffect(() => {
    // Gated on `ready` (not just `authenticated`): SSE job updates carry assets whose
    // thumbnails render immediately, so the media-ticket mint must have settled first
    // (sc-8810; sc-9063 lets a failed mint through — media degrades, data flows).
    if (!ready) {
      return undefined;
    }

    let events = null;
    let reconnectTimer = null;
    let reconnectAttempt = 0;
    let closed = false;

    function handleJobUpdated(event) {
      const job = parseSseJson(event, "job");
      if (!job) {
        return;
      }
      const hasGeneratedAssets = Boolean(job.result?.generationSetId || job.result?.assetIds?.length || job.result?.assets?.length);
      const resultAssetCount = generatedResultAssetCount(job);
      const generationSetId = job.result?.generationSetId ?? "";
      const refreshKey = job.id ?? generationSetId;
      const previousRefresh = generatedAssetRefreshesRef.current.get(refreshKey) ?? { assetCount: 0, generationSetId: "" };
      const shouldRefreshGeneratedAssets =
        Boolean(job.projectId) &&
        hasGeneratedAssets &&
        (resultAssetCount > previousRefresh.assetCount ||
          (resultAssetCount === 0 && generationSetId && generationSetId !== previousRefresh.generationSetId));
      setJobs((items) => upsertJobNewest(items, job));
      if (hasGeneratedAssets) {
        if (job.result?.generationSetId) {
          setLatestGenerationSetId(job.result.generationSetId);
        }
        generatedAssetRefreshesRef.current.set(refreshKey, {
          assetCount: Math.max(resultAssetCount, previousRefresh.assetCount),
          generationSetId: generationSetId || previousRefresh.generationSetId,
        });
        if (shouldRefreshGeneratedAssets) {
          refreshAssetsRef.current?.(job.projectId);
        }
      }
      if (job.status === "completed" && hasGeneratedAssets) {
        enqueueTimelineGenerationApply(job);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_track") {
        refreshPersonTracksRef.current?.(job.projectId);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_detect") {
        refreshAssetsRef.current?.(job.projectId);
      }
      // A completed download flips the catalog's `installState`; a completed conversion writes the
      // MLX artifact that flips `mlxConversionState` to "converted". Both are derived server-side
      // from the filesystem, so the catalog must be refetched or the Models row keeps offering
      // "Convert to MLX" for an already-converted model until the app is restarted.
      if (job.status === "completed" && (job.type === "model_download" || job.type === "model_convert")) {
        refreshDataRef.current?.();
      }
      // A completed built-in LoRA download (sc-5944) flips the catalog entry to
      // installed; refresh models+loras so the Models row and any Studio gate update.
      if (job.status === "completed" && job.type === "lora_download") {
        refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "completed" && job.type === "lora_import") {
        dismissNoticeKind("lora-import");
        refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "completed" && job.type === "lora_train" && job.payload?.dryRun === false) {
        if (job.result?.loraRegistered === false) {
          pushNotice("lora-train", `lora training: ${job.result?.loraRegistrationError ?? "Completed training but could not register the LoRA."}`);
        } else {
          dismissNoticeKind("lora-train");
          refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
        }
      }
      if (job.status === "failed" && !hasVisibleLocalFailure(job)) {
        pushNotice(noticeKindForJob(job), failedJobNotice(job));
      }
      // Evict the per-job refresh bookkeeping once the job is terminal (sc-8944): the
      // dedupe counters are only consulted while a job is still emitting incremental
      // asset updates. Without this the Map grows one entry per asset-producing job for
      // the session's lifetime (unbounded, slow leak in multi-day sessions).
      if (terminalStatuses.has(job.status)) {
        generatedAssetRefreshesRef.current.delete(refreshKey);
      }
    }

    function handleWorkerUpdated(event) {
      const worker = parseSseJson(event, "worker");
      if (!worker) {
        return;
      }
      setWorkers((items) => [worker, ...items.filter((item) => item.id !== worker.id)].sort(sortWorkers));
    }

    function handleQueueUpdated(event) {
      const summary = parseSseJson(event, "queue");
      if (!summary) {
        return;
      }
      setQueueSummary(summary);
      if (Array.isArray(summary.workers)) {
        setWorkers(summary.workers.sort(sortWorkers));
      }
    }

    async function connect() {
      let ticket = "";
      try {
        if (access.authRequired) {
          const response = await apiFetch("/api/v1/jobs/events/ticket", token, { method: "POST" });
          ticket = response.ticket;
        }
      } catch (err) {
        setError(err.message);
        if (!closed) {
          const delay = Math.min(30000, 1000 * 2 ** reconnectAttempt);
          reconnectAttempt += 1;
          reconnectTimer = window.setTimeout(connect, delay);
        }
        return;
      }

      if (closed) {
        return;
      }

      const source = new EventSource(eventUrl("/api/v1/jobs/events", ticket));
      events = source;
      source.addEventListener("job.updated", handleJobUpdated);
      source.addEventListener("worker.updated", handleWorkerUpdated);
      source.addEventListener("queue.updated", handleQueueUpdated);
      source.onopen = () => {
        reconnectAttempt = 0;
      };
      source.onerror = () => {
        source.close();
        if (closed) {
          return;
        }
        const delay = Math.min(30000, 1000 * 2 ** reconnectAttempt);
        reconnectAttempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      closed = true;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      events?.close();
    };
    // Deps deliberately limited to the three inputs that must re-subscribe the stream
    // (auth mode / readiness / token) — mirrors the pre-extraction App effect. The
    // setters/refs/callbacks are fed identity-stable and read live inside the handlers,
    // so listing them would only re-open the EventSource on unrelated re-renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [access.authRequired, ready, token]);
}
