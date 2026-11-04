import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { useProjectScopedList } from "../hooks/useProjectScopedList.js";
import { systemApi } from "../api/systemResources.js";
import { trainingApi } from "../api/trainingResources.js";
import { assetsApi } from "../api/assetsResources.js";
import { projectsApi } from "../api/projectsResources.js";
import { jobsApi } from "../api/jobsResources.js";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { isAbortError } from "../api.js";
import { beginAssetRequest, settleAssetRequest } from "../startupTiming.js";
import { sortWorkers } from "../sorters.js";
import { isDesktop as isDesktopShell } from "../runtime.js";
import { SIMPLE_MODE } from "../simple/uiMode.js";
import { mergeFreshJobs } from "../appHelpers.js";
import { applySuccessfulProjectRefresh, reconcileSelectedAssetId } from "../appStateHelpers.js";
import { hydrationDomainsForRoute, hydrationLedgerKey, PROJECT_HYDRATION_DOMAINS, isProjectHydrationDomain } from "../appHydration.js";
import { refreshFailure, refreshSuccess } from "../refreshResult.js";

// F-189: extracted from App along its existing concern boundary.
export function useShellHydration({
  ready, token, activeProject, setAssets, setLoadedAssetsProjectId, setAssetLoadState, setCharacters, setSavedVoices, setPersonTracks, setTimelines, setTimelinesProjectId, setPresets, setPromptBatches, setLoras, setTrainingDatasets, setTrainingDatasetsProjectId, setTrainingDatasetsError, setSelectedTimelineId, setActiveTimeline, setMacCapabilitiesAuthoritative, setMacCapabilitiesLoading, setMacCapabilities, setMacCapabilitiesError, setDomainError, setProjects, setActiveProject, setProjectsLoaded, setJobs, setQueueSummary, setWorkers, refreshModels, domainErrors, setTrainingTargets, setTrainingTargetsError, setTrainingPresets, setTrainingPresetsError, activeProjectRef, loadedAssetsProjectId, setSelectedAssetId, refreshLoras, setupCompleted, authenticated, uiMode, simpleActiveScreen, activeView, refreshCharacters, refreshSavedVoices, refreshPresets, refreshPromptBatches, refreshTrainingDatasets, refreshPersonTracks, refreshTimelines, projectsLoaded, activeAssetLoadState, hasLibraryConsumerAsset
}) {

  // A retry or token transition supersedes any older capabilities request. Fetch aborts normally
  // provide this guarantee too, but the request id keeps a late/non-cooperative response from
  // restoring stale platform facts after a newer request has started.
  const macCapabilitiesRequestRef = useRef(0);

  const hydrationLedgerRef = useRef(new Map());

  const committedHydrationRouteRef = useRef(null);

  const previousHydrationProjectIdRef = useRef(null);

  // Project changes reset only project-owned catalogs. Global models/training
  // metadata and shared shell state survive. useLayoutEffect prevents the next
  // project from painting the previous project's unscoped hook state.
  useLayoutEffect(() => {
    const nextProjectId = activeProject?.id ?? null;
    const previousProjectId = previousHydrationProjectIdRef.current;
    if (previousProjectId === nextProjectId) {
      return;
    }
    previousHydrationProjectIdRef.current = nextProjectId;
    if (previousProjectId !== null) {
      // Discarded catalogs must reload on a return visit. Leave pending entries
      // available for the route effect to abort their original requests.
      for (const domain of PROJECT_HYDRATION_DOMAINS) {
        const key = hydrationLedgerKey(domain, previousProjectId);
        if (hydrationLedgerRef.current.get(key)?.status !== "loading") {
          hydrationLedgerRef.current.delete(key);
        }
      }
      setAssets([]);
      setLoadedAssetsProjectId(null);
      setAssetLoadState({
        projectId: nextProjectId,
        status: nextProjectId ? "loading" : "idle",
        error: "",
      });
      setCharacters([]);
      setSavedVoices([]);
      setPersonTracks([]);
      setTimelines([]);
      setTimelinesProjectId(null);
      setPresets([]);
      setPromptBatches([]);
      setLoras([]);
      setTrainingDatasets([]);
      setTrainingDatasetsProjectId(null);
      setTrainingDatasetsError("");
      setSelectedTimelineId(null);
      setActiveTimeline(null);
    }
  }, [activeProject?.id, setActiveTimeline, setAssetLoadState, setAssets, setCharacters, setLoadedAssetsProjectId, setLoras, setPersonTracks, setPresets, setPromptBatches, setSavedVoices, setSelectedTimelineId, setTimelines, setTimelinesProjectId, setTrainingDatasets, setTrainingDatasetsError, setTrainingDatasetsProjectId]);

  const refreshMacCapabilities = useCallback(async ({ signal } = {}) => {
    const requestId = macCapabilitiesRequestRef.current + 1;
    macCapabilitiesRequestRef.current = requestId;
    // Preserve the last payload for inert Mac-gating helpers, but it is not authorization for
    // decoder reconciliation/submission while this request is unsettled.
    setMacCapabilitiesAuthoritative(false);
    setMacCapabilitiesLoading(true);
    try {
      const value = await systemApi.capabilitiesMac(token, { signal });
      if (signal?.aborted || macCapabilitiesRequestRef.current !== requestId) {
        return false;
      }
      if (
        !value ||
        typeof value !== "object" ||
        Array.isArray(value) ||
        typeof value.macGatingActive !== "boolean"
      ) {
        throw new Error("the API returned an invalid capability response");
      }
      setMacCapabilities(value);
      setMacCapabilitiesAuthoritative(true);
      setMacCapabilitiesError("");
      return true;
    } catch (err) {
      if (
        isAbortError(err) ||
        signal?.aborted ||
        macCapabilitiesRequestRef.current !== requestId
      ) {
        return false;
      }
      setMacCapabilitiesError(
        `Could not load engine capabilities: ${err?.message || "unknown error"}`,
      );
      return false;
    } finally {
      if (macCapabilitiesRequestRef.current === requestId) {
        setMacCapabilitiesLoading(false);
      }
    }
  }, [setMacCapabilities, setMacCapabilitiesAuthoritative, setMacCapabilitiesError, setMacCapabilitiesLoading, token]);

  const refreshShellData = useCallback(async ({ signal } = {}) => {
    const fetchInitial = async (domain, label, load, fallback, optional = false) => {
      try {
        const value = await load(token, { signal });
        setDomainError(domain, "");
        return { label, value, error: "" };
      } catch (err) {
        if (isAbortError(err)) {
          return { label, value: fallback, error: "", aborted: true };
        }
        const error = optional ? "" : `${label}: ${err.message}`;
        setDomainError(domain, error);
        return { label, value: fallback, error, failed: true };
      }
    };
    const projectsPromise = fetchInitial(
      "projects",
      "Projects",
      projectsApi.projects,
      [],
    ).then((result) => {
      if (result.aborted) return result;
      applySuccessfulProjectRefresh(result, setProjects, setActiveProject);
      setProjectsLoaded(true);
      return result;
    });
    // This optional surface must not hold up shell hydration. Its own state keeps alternate
    // decoder requests fail-closed and gives affected users retry/reset recovery on failure.
    void refreshMacCapabilities({ signal });
    const shellPromises = [
      fetchInitial("jobs", "Jobs", jobsApi.list, []).then((result) => {
        if (result.aborted) return result;
        setJobs((current) => mergeFreshJobs(current, result.value));
        setQueueSummary(null);
        return result;
      }),
      fetchInitial("workers", "Workers", systemApi.workers, []).then((result) => {
        if (result.aborted) return result;
        setWorkers(result.value.sort(sortWorkers));
        return result;
      }),
    ];
    await Promise.all([projectsPromise, ...shellPromises]);
  }, [refreshMacCapabilities, setActiveProject, setDomainError, setJobs, setProjects, setProjectsLoaded, setQueueSummary, setWorkers, token]);

  const refreshTrainingTargets = useCallback(async ({ signal } = {}) => {
    try {
      const catalog = await trainingApi.targets(token, { signal });
      setTrainingTargets(catalog);
      setTrainingTargetsError("");
      domainErrors.trainingTargets("");
      return refreshSuccess(catalog);
    } catch (err) {
      if (isAbortError(err)) return refreshFailure("aborted", err);
      setTrainingTargetsError(err.message);
      domainErrors.trainingTargets(err.message);
      return refreshFailure("error", err);
    }
  }, [domainErrors, setTrainingTargets, setTrainingTargetsError, token]);

  const refreshTrainingPresets = useCallback(async ({ signal } = {}) => {
    try {
      const catalog = await trainingApi.presets(token, { signal });
      setTrainingPresets(catalog);
      setTrainingPresetsError("");
      domainErrors.trainingPresets("");
      return refreshSuccess(catalog);
    } catch (err) {
      if (isAbortError(err)) return refreshFailure("aborted", err);
      setTrainingPresetsError(err.message);
      domainErrors.trainingPresets(err.message);
      return refreshFailure("error", err);
    }
  }, [domainErrors, setTrainingPresets, setTrainingPresetsError, token]);

  const refreshAssets = useProjectScopedList({
    token, activeProject, activeProjectRef,
    load: assetsApi.listIncludingHidden,
    setItems: setAssets,
    setError: domainErrors.assets,
    onStart: (projectId) => {
      setAssetLoadState((current) => loadedAssetsProjectId === projectId
        ? { ...current, projectId, error: "" }
        : { projectId, status: "loading", error: "" });
      return beginAssetRequest();
    },
    onSuccess: (items, projectId) => {
      setLoadedAssetsProjectId(projectId);
      setSelectedAssetId((current) => reconcileSelectedAssetId(items, current));
      setAssetLoadState({ projectId, status: "ready", error: "" });
    },
    onError: (error, projectId) => setAssetLoadState({ projectId, status: "error", error: error.message }),
    onSettled: ({ context }) => settleAssetRequest(context),
  });

  useEffect(() => {
    if (!ready) {
      return;
    }
    hydrationLedgerRef.current.clear();
    const controller = new AbortController();
    refreshShellData({ signal: controller.signal });
    return () => controller.abort();
  }, [ready, refreshShellData, token]);

  // Publish only committed route ownership for cleanup/reuse decisions.
  useLayoutEffect(() => {
    const setupRoute =
      isDesktopShell && setupCompleted === false && authenticated;
    const simpleRoute = !setupRoute && uiMode === SIMPLE_MODE;
    const route = setupRoute
      ? "Setup"
      : simpleRoute
        ? simpleActiveScreen
        : activeView;
    const committedRoute = {
      route,
      simple: simpleRoute,
      projectId: activeProject?.id ?? null,
    };
    committedHydrationRouteRef.current = committedRoute;
    return () => {
      if (committedHydrationRouteRef.current === committedRoute) {
        committedHydrationRouteRef.current = null;
      }
    };
  });

  const hydrationRefreshers = useMemo(() => ({
    assets: refreshAssets, characters: refreshCharacters, savedVoices: refreshSavedVoices,
    loras: refreshLoras, presets: refreshPresets, promptBatches: refreshPromptBatches,
    trainingDatasets: refreshTrainingDatasets, personTracks: refreshPersonTracks, timelines: refreshTimelines,
    models: (_id, options) => refreshModels(options),
    trainingTargets: (_id, options) => refreshTrainingTargets(options),
    trainingPresets: (_id, options) => refreshTrainingPresets(options),
  }), [refreshAssets, refreshCharacters, refreshLoras, refreshModels, refreshPersonTracks, refreshPresets, refreshPromptBatches, refreshSavedVoices, refreshTimelines, refreshTrainingDatasets, refreshTrainingPresets, refreshTrainingTargets]);
  const hydrationRefresher = useCallback((domain) => hydrationRefreshers[domain], [hydrationRefreshers]);

  const ensureHydrationDomain = useCallback((domain, projectId, options = {}) => {
    if (isProjectHydrationDomain(domain) && !projectId) {
      return Promise.resolve(refreshFailure("missing-project"));
    }
    const ledger = hydrationLedgerRef.current;
    const key = hydrationLedgerKey(domain, projectId);
    const current = ledger.get(key);
    if (current?.status === "ready") {
      return Promise.resolve(refreshSuccess());
    }
    if (current?.status === "loading") {
      return current.promise;
    }
    const refresh = hydrationRefresher(domain);
    if (!refresh) {
      return Promise.resolve(refreshFailure("missing-refresher"));
    }
    const { abort: abortRequest = null, ...refreshOptions } = options;
    const entry = { status: "loading", promise: null, abort: abortRequest };
    entry.promise = Promise.resolve()
      .then(() => refresh(projectId, refreshOptions))
      .then((result) => {
        const normalized = result?.ok === true
          ? result
          : result?.ok === false
            ? result
            : refreshFailure("error", new Error(`${domain} refresh did not report success`));
        if (ledger.get(key) === entry) {
          ledger.set(
            key,
            normalized.ok
              ? { status: "ready" }
              : { status: "error", reason: normalized.reason, error: normalized.error },
          );
        }
        return normalized;
      })
      .catch((error) => {
        const result = refreshFailure("error", error);
        if (ledger.get(key) === entry) {
          ledger.set(key, { status: "error", reason: result.reason, error });
        }
        return result;
      });
    ledger.set(key, entry);
    return entry.promise;
  }, [hydrationRefresher]);

  // Hydrate only the active screen's domains. Each ledger entry is global or
  // project-keyed, so keep-alive navigation reuses settled catalogs while a
  // project switch loads fresh overlays without discarding global state.
  useEffect(() => {
    if (!ready || !projectsLoaded) {
      return;
    }
    const setupRoute =
      isDesktopShell && setupCompleted === false && authenticated;
    const simpleRoute = !setupRoute && uiMode === SIMPLE_MODE;
    const route = setupRoute
      ? "Setup"
      : simpleRoute
        ? simpleActiveScreen
        : activeView;
    const domains = hydrationDomainsForRoute(route, { simple: simpleRoute });
    const projectId = activeProject?.id ?? null;
    const controller = new AbortController();
    const started = [];
    const hydrationLedger = hydrationLedgerRef.current;

    domains.forEach((domain) => {
      if (isProjectHydrationDomain(domain) && !projectId) {
        return;
      }
      const key = hydrationLedgerKey(domain, projectId);
      const current = hydrationLedger.get(key);
      if (current?.status === "loading") {
        started.push({ domain, key });
        return;
      }
      if (current?.status === "ready") {
        return;
      }
      started.push({ domain, key });
      ensureHydrationDomain(domain, projectId, {
        signal: controller.signal,
        abort: () => controller.abort(),
      });
    });

    return () => {
      const next = committedHydrationRouteRef.current;
      const nextDomains = next
        ? new Set([
            ...hydrationDomainsForRoute(next.route, { simple: next.simple }),
            ...hydrationDomainsForRoute(next.route, {
              simple: next.simple,
              deferred: true,
            }),
          ])
        : new Set();
      if (
        next?.projectId === projectId &&
        started.every(({ domain }) => nextDomains.has(domain))
      ) {
        return;
      }
      controller.abort();
      started.forEach(({ key }) => {
        const entry = hydrationLedger.get(key);
        if (entry?.status === "loading") {
          entry.abort?.();
          hydrationLedger.delete(key);
        }
      });
    };
  }, [
    activeProject?.id,
    activeView,
    authenticated,
    ensureHydrationDomain,
    projectsLoaded,
    ready,
    setupCompleted,
    simpleActiveScreen,
    token,
    uiMode,
  ]);

  // The default Library route paints from its project asset request first. Only
  // after that catalog settles do we hydrate the secondary detail-panel domains:
  // characters power "Move to Character", while models power image VQA and audio
  // model metadata. This keeps startup assets-first without leaving direct Library
  // entry with permanently incomplete controls.
  useEffect(() => {
    if (
      !ready ||
      !projectsLoaded ||
      uiMode === SIMPLE_MODE ||
      activeView !== "Library" ||
      !activeProject?.id ||
      activeAssetLoadState.status !== "ready" ||
      !hasLibraryConsumerAsset
    ) {
      return;
    }
    const controller = new AbortController();
    const domains = hydrationDomainsForRoute("Library", { deferred: true });
    const startedKeys = [];
    const hydrationLedger = hydrationLedgerRef.current;
    domains.forEach((domain) => {
      const key = hydrationLedgerKey(domain, activeProject.id);
      const current = hydrationLedger.get(key);
      if (current?.status !== "ready") {
        startedKeys.push(key);
      }
      ensureHydrationDomain(domain, activeProject.id, {
        signal: controller.signal,
        abort: () => controller.abort(),
      });
    });
    return () => {
      const next = committedHydrationRouteRef.current;
      const nextDomains = next
        ? new Set([
            ...hydrationDomainsForRoute(next.route, { simple: next.simple }),
            ...hydrationDomainsForRoute(next.route, {
              simple: next.simple,
              deferred: true,
            }),
          ])
        : new Set();
      if (
        next?.projectId === activeProject.id &&
        domains.every((domain) => nextDomains.has(domain))
      ) {
        return;
      }
      controller.abort();
      startedKeys.forEach((key) => {
        const entry = hydrationLedger.get(key);
        if (entry?.status === "loading") {
          entry.abort?.();
          hydrationLedger.delete(key);
        }
      });
    };
  }, [
    activeAssetLoadState.status,
    activeProject?.id,
    activeView,
    ensureHydrationDomain,
    hasLibraryConsumerAsset,
    projectsLoaded,
    ready,
    uiMode,
  ]);

  const hydrateLaunchDomains = useCallback(async (view, projectId = activeProjectRef.current?.id) => {
    const domains = hydrationDomainsForRoute(view);
    const results = await Promise.all(
      domains.map((domain) => ensureHydrationDomain(domain, projectId)),
    );
    const failed = results.filter((result) => result?.ok !== true);
    return failed.length
      ? { ok: false, failures: failed }
      : refreshSuccess();
  }, [activeProjectRef, ensureHydrationDomain]);
  const context = useAppDomainValue({
    refreshMacCapabilities
  });
  return { ...context, refreshAssets, hydrateLaunchDomains, context };
}
