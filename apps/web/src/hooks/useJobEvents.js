import { useEffect } from "react";
import { apiFetch, eventUrl, isReauthenticating } from "../api.js";
import {
  acceptsJobUpdate,
  jobRevision,
  MAX_TERMINAL_JOBS,
  sortWorkers,
  upsertJobNewest,
} from "../sorters.js";
import { terminalStatuses } from "../constants.js";
import {
  failedJobNotice,
  generatedResultAssetCount,
  noticeKindForJob,
  parseSseJson,
  reconcileAuthoritativeJobs,
  trainingRegistrationFailure,
} from "../appHelpers.js";

// Owns the live job/worker/queue SSE stream (the EventSource lifecycle: ticket mint,
// connect, event handlers, exponential-backoff reconnect, teardown). Extracted verbatim
// from App.jsx (sc-9750, F-052 follow-up) so App no longer carries the ~150-line stream
// effect inline. Behavior-preserving — same effect body, same [access.authRequired,
// ready, token] dependency array, same cleanup.
//
// The handlers reach back into App state through the setters/refs/callbacks passed in.
// Every one of these is fed identity-stable: useState setters (stable by React),
// useRef handles, and the two callbacks — `hasVisibleLocalFailure` (an empty-dep
// useCallback reading only refs), `enqueueTimelineGenerationApply` (a stable
// ref-delegating useCallback in useTimelines), and `refreshHealth` — which sc-11231
// (F-037) hardened after
// finding them plain per-render declarations that the SSE effect captured stale at
// subscribe time. Because they are stable, the effect's deps are exactly the three
// inputs that must re-subscribe the stream — auth mode, readiness, and the token —
// mirroring the pre-extraction effect. `access` is read only for `access.authRequired`;
// the whole object is passed so the dep array (`access.authRequired`) matches the
// original render-scope reference.
export function useJobEvents({
  access,
  ready,
  token,
  jobsRef,
  setJobs,
  setWorkers,
  setQueueSummary,
  setLatestGenerationSetId,
  setError,
  pushNotice,
  dismissNoticeKind,
  generatedAssetRefreshesRef,
  refreshAssetsRef,
  refreshModelsRef,
  refreshModelAndLorasRef,
  refreshPersonTracksRef,
  activeProjectRef,
  enqueueTimelineGenerationApply,
  hasVisibleLocalFailure,
  refreshHealth,
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
    let snapshotRevision = 0;
    let reconnectKnownJobIds = new Set();
    let maxRetainedClearTombstones = MAX_TERMINAL_JOBS;
    const appliedTerminalRevisionByJob = new Map();
    // Clear tombstones protect against a job publication delayed past its clear
    // event. Bound them to the reconnect's complete active set plus the same
    // 200-row tail as visible terminal history instead of accumulating for the
    // connection lifetime. Map insertion order is oldest -> newest, with the
    // clear event revision used to preserve post-snapshot-barrier clears during
    // authoritative replacement.
    const clearedJobRevisionById = new Map();

    function pruneClearedJobs() {
      while (clearedJobRevisionById.size > maxRetainedClearTombstones) {
        clearedJobRevisionById.delete(clearedJobRevisionById.keys().next().value);
      }
    }

    function growClearedJobLimit(visibleJobs) {
      const activeJobCount = visibleJobs.filter(
        (job) => !terminalStatuses.has(job.status),
      ).length;
      maxRetainedClearTombstones = Math.max(
        maxRetainedClearTombstones,
        activeJobCount + MAX_TERMINAL_JOBS,
      );
    }

    function rememberClearedJob(jobId, revision) {
      if (!jobId) {
        return;
      }
      clearedJobRevisionById.delete(jobId);
      clearedJobRevisionById.set(jobId, revision);
      pruneClearedJobs();
    }

    function pruneAppliedTerminalRevisions(visibleJobs) {
      const visibleIds = new Set(visibleJobs.map((job) => job.id));
      for (const jobId of appliedTerminalRevisionByJob.keys()) {
        if (!visibleIds.has(jobId)) {
          appliedTerminalRevisionByJob.delete(jobId);
        }
      }
    }

    function applyJobUpdate(job) {
      if (!job?.id || clearedJobRevisionById.has(job.id)) {
        return;
      }
      const currentJobs = jobsRef?.current ?? [];
      // State freshness is the first authority. A globally newer SSE publication
      // can still carry an older durable job row; rejected state must never cause
      // terminal effects.
      if (!acceptsJobUpdate(currentJobs, job)) {
        return;
      }
      const terminalRevision = jobRevision(job);
      const previousTerminalRevision = appliedTerminalRevisionByJob.get(job.id);
      const duplicateTerminalEffects =
        terminalStatuses.has(job.status) &&
        terminalRevision !== null &&
        previousTerminalRevision !== undefined &&
        previousTerminalRevision >= terminalRevision;
      if (terminalStatuses.has(job.status) && terminalRevision !== null) {
        appliedTerminalRevisionByJob.set(
          job.id,
          Math.max(previousTerminalRevision ?? terminalRevision, terminalRevision),
        );
      }
      setJobs((items) => {
        const next = upsertJobNewest(items, job);
        growClearedJobLimit(next);
        pruneAppliedTerminalRevisions(next);
        if (jobsRef) {
          jobsRef.current = next;
        }
        return next;
      });
      // A post-barrier publication can carry the exact terminal row already
      // captured by the snapshot. Its global SSE id is newer, but replaying the
      // same durable job revision would duplicate non-idempotent effects such as
      // timeline mutation and user notices.
      if (duplicateTerminalEffects) {
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
      // MLX artifact that flips `mlxConversionState` to "converted"; a completed import upserts the
      // model into `user.models.jsonc` (sc-10688). All three are derived server-side from the
      // filesystem/manifests, so the catalog must be refetched or the Models page keeps rendering
      // the pre-job row — still offering "Convert to MLX" for an already-converted model, or
      // omitting the imported one entirely — until the app is restarted.
      //
      // `model_import` now fires (epic 14015): S0d (sc-14019) opened the backend kill-switch and
      // gates `create_model_import_job` on a base-weight compatibility verdict, and S0e (sc-14020)
      // exposed the point-at-file import affordance. A completed import upserts a
      // `catalogScope:"user"` model into `user.models.jsonc`, so the refetch below is required for
      // it to appear on the Models page without an app restart.
      if (
        job.status === "completed" &&
        (job.type === "model_download" || job.type === "model_convert" || job.type === "model_import")
      ) {
        refreshModelsRef.current?.();
      }
      // A completed built-in LoRA download (sc-5944) flips the catalog entry to
      // installed; refresh models+loras so the Models row and any Studio gate update.
      if (job.status === "completed" && job.type === "lora_download") {
        refreshModelAndLorasRef.current?.(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "completed" && job.type === "lora_import") {
        dismissNoticeKind("lora-import");
        refreshModelAndLorasRef.current?.(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "completed" && job.type === "lora_train" && job.payload?.dryRun === false) {
        // sc-15036: a `lora_train` job now produces EITHER an adapter or a full base checkpoint,
        // and each registrar reports under its own result key. See `trainingRegistrationFailure`.
        const registrationError = trainingRegistrationFailure(job.result);
        if (registrationError) {
          pushNotice("lora-train", `lora training: ${registrationError}`);
        } else {
          dismissNoticeKind("lora-train");
          refreshModelAndLorasRef.current?.(job.projectId ?? activeProjectRef.current?.id);
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

    function handleJobUpdated(event) {
      const revision = Number(event.lastEventId ?? 0);
      if (Number.isFinite(revision) && revision > 0 && revision <= snapshotRevision) {
        return;
      }
      const job = parseSseJson(event, "job");
      if (job) {
        applyJobUpdate(job);
      }
    }

    function handleWorkerUpdated(event) {
      const worker = parseSseJson(event, "worker");
      if (!worker) {
        return;
      }
      setWorkers((items) => [worker, ...items.filter((item) => item.id !== worker.id)].sort(sortWorkers));
    }

    function handleJobsSnapshot(event) {
      const payload = parseSseJson(event, "jobs snapshot");
      const snapshotJobs = Array.isArray(payload) ? payload : payload?.jobs;
      if (!Array.isArray(snapshotJobs)) {
        return;
      }
      const revision = Number(Array.isArray(payload) ? event.lastEventId : payload.revision);
      if (Number.isFinite(revision) && revision > 0) {
        snapshotRevision = Math.max(snapshotRevision, revision);
      }
      const snapshotClearedJobIds = Array.isArray(payload?.clearedJobIds)
        ? payload.clearedJobIds
        : [];
      const barrierRevision =
        Number.isFinite(revision) && revision > 0 ? revision : null;
      const postBarrierClears = [...clearedJobRevisionById.entries()].filter(
        ([, clearRevision]) =>
          barrierRevision === null ||
          (clearRevision !== null && clearRevision > barrierRevision),
      );
      // The snapshot is authoritative at its revision. Replace connection-old
      // tombstones with the bounded subset relevant to the IDs carried by this
      // reconnect, while retaining any live clear delivered after the barrier.
      // Snapshot IDs preserve request order (newest visible jobs first), so insert
      // the retained prefix in reverse to keep Map eviction oldest-first.
      const relevantSnapshotClears = snapshotClearedJobIds
        .filter((jobId) => reconnectKnownJobIds.has(jobId))
        .slice(0, maxRetainedClearTombstones);
      clearedJobRevisionById.clear();
      for (const jobId of [...relevantSnapshotClears].reverse()) {
        rememberClearedJob(jobId, barrierRevision);
      }
      for (const [jobId, clearRevision] of postBarrierClears) {
        rememberClearedJob(jobId, clearRevision);
      }
      const authoritativeClearedJobIds = new Set([
        ...snapshotClearedJobIds,
        ...postBarrierClears.map(([jobId]) => jobId),
      ]);
      for (const jobId of authoritativeClearedJobIds) {
        appliedTerminalRevisionByJob.delete(jobId);
      }
      const jobs = snapshotJobs.filter((job) => !authoritativeClearedJobIds.has(job.id));
      const currentJobs = jobsRef?.current ?? [];
      const currentById = new Map(currentJobs.map((job) => [job.id, job]));
      const missedTerminalTransitions = jobs.filter((job) => {
        const current = currentById.get(job.id);
        return (
          current &&
          !terminalStatuses.has(current.status) &&
          terminalStatuses.has(job.status) &&
          acceptsJobUpdate([current], job)
        );
      });
      const missedTerminalIds = new Set(missedTerminalTransitions.map((job) => job.id));
      // Authoritative terminal rows that were already terminal locally are
      // history, not transitions. Seed their durable revisions so a delayed
      // post-barrier publication cannot replay effects.
      for (const job of jobs) {
        const revision = jobRevision(job);
        if (
          terminalStatuses.has(job.status) &&
          revision !== null &&
          !missedTerminalIds.has(job.id)
        ) {
          appliedTerminalRevisionByJob.set(
            job.id,
            Math.max(appliedTerminalRevisionByJob.get(job.id) ?? revision, revision),
          );
        }
      }
      setJobs((items) => {
        const next = reconcileAuthoritativeJobs(items, jobs, [
          ...authoritativeClearedJobIds,
        ]);
        growClearedJobLimit(next);
        pruneAppliedTerminalRevisions(next);
        if (jobsRef) {
          jobsRef.current = next;
        }
        return next;
      });
      // Re-run the bounded effects a disconnected client missed when an active
      // row became terminal: catalog refresh, per-project asset refresh, and
      // timeline generation application use the same idempotent handlers as a
      // live job.updated event. Historical terminal rows are not replayed.
      for (const job of missedTerminalTransitions) {
        applyJobUpdate(job);
      }
    }

    function handleJobsCleared(event) {
      const payload = parseSseJson(event, "cleared jobs");
      if (!Array.isArray(payload?.ids)) {
        return;
      }
      const revision = Number(event.lastEventId);
      const clearRevision =
        Number.isFinite(revision) && revision > 0 ? revision : null;
      if (clearRevision !== null && clearRevision <= snapshotRevision) {
        return;
      }
      const cleared = new Set(payload.ids);
      const visibleJobIds = new Set((jobsRef?.current ?? []).map((job) => job.id));
      for (const jobId of cleared) {
        if (
          visibleJobIds.has(jobId) ||
          reconnectKnownJobIds.has(jobId) ||
          clearedJobRevisionById.has(jobId)
        ) {
          rememberClearedJob(jobId, clearRevision);
        }
        appliedTerminalRevisionByJob.delete(jobId);
      }
      setJobs((items) => {
        const next = items.filter((job) => !cleared.has(job.id));
        pruneAppliedTerminalRevisions(next);
        if (jobsRef) {
          jobsRef.current = next;
        }
        return next;
      });
    }

    function handleQueueUpdated(event) {
      const summary = parseSseJson(event, "queue");
      if (!summary) {
        return;
      }
      setQueueSummary(summary);
      if (Array.isArray(summary.workers)) {
        setWorkers([...summary.workers].sort(sortWorkers));
      }
    }

    async function connect() {
      let ticket = "";
      // Carry the complete reconnect set in the authenticated POST body. The
      // EventSource URL stays a bounded opaque ticket even for hundreds of jobs.
      const activeJobIds = (jobsRef?.current ?? [])
        .filter(
          (job) =>
            !terminalStatuses.has(job.status) &&
            !clearedJobRevisionById.has(job.id),
        )
        .map((job) => job.id)
        .filter(Boolean);
      const knownTerminalJobIds = (jobsRef?.current ?? [])
        .filter(
          (job) =>
            terminalStatuses.has(job.status) &&
            !clearedJobRevisionById.has(job.id),
        )
        .map((job) => job.id)
        .filter(Boolean);
      reconnectKnownJobIds = new Set([...activeJobIds, ...knownTerminalJobIds]);
      maxRetainedClearTombstones = activeJobIds.length + MAX_TERMINAL_JOBS;
      pruneClearedJobs();
      try {
        if (access.authRequired || activeJobIds.length > 0 || knownTerminalJobIds.length > 0) {
          const response = await apiFetch("/api/v1/jobs/events/ticket", token, {
            method: "POST",
            body: JSON.stringify({ activeJobIds, knownTerminalJobIds }),
          });
          ticket = response.ticket;
        }
      } catch (err) {
        if (isReauthenticating(err)) {
          // sc-15105: the token went stale mid-session and the access gate has claimed the
          // session. Don't push the raw "SceneWorks access token required" into the error
          // band and don't arm a reconnect that can only 401 again — `ready` drops with the
          // gate, tearing this effect down, and it re-subscribes if the re-verify keeps the
          // session. A 401 the gate did NOT claim stays an ordinary failure below.
          return;
        }
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

      // Event revisions are process-local. An API restart can reset the counter,
      // so each newly opened source establishes its own snapshot baseline.
      snapshotRevision = 0;
      const source = new EventSource(eventUrl("/api/v1/jobs/events", ticket));
      events = source;
      source.addEventListener("jobs.snapshot", handleJobsSnapshot);
      source.addEventListener("jobs.cleared", handleJobsCleared);
      source.addEventListener("job.updated", handleJobUpdated);
      source.addEventListener("worker.updated", handleWorkerUpdated);
      source.addEventListener("queue.updated", handleQueueUpdated);
      source.onopen = () => {
        reconnectAttempt = 0;
        refreshHealth();
      };
      source.onerror = () => {
        source.close();
        if (closed) {
          return;
        }
        refreshHealth();
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
    // setters/refs/callbacks are fed identity-stable (see the header note; sc-11231
    // closed the last two gaps) and read live inside the handlers, so listing them would
    // only churn the EventSource, never satisfy the stream's actual re-subscribe contract.
    // The disable stays because exhaustive-deps can't prove that stability statically — it
    // would still demand ~15 always-stable handles in the array.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [access.authRequired, ready, token]);
}
