import { useProjectScopedList } from "../hooks/useProjectScopedList.js";
import { systemApi } from "../api/systemResources.js";
import { modelsApi } from "../api/modelsResources.js";
import { trainingApi } from "../api/trainingResources.js";
import { assetsApi } from "../api/assetsResources.js";
import { projectsApi } from "../api/projectsResources.js";
import { jobsApi } from "../api/jobsResources.js";
import { useCallback, useEffect, useLayoutEffect, useRef } from "react";
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
  refreshModelsRef, refreshModelAndLorasRef, hydrateLaunchDomainsRef, refreshAssetsRef, refreshCharactersRef, refreshPersonTracksRef, ready, token, activeProject, setAssets, setLoadedAssetsProjectId, setAssetLoadState, setCharacters, setSavedVoices, setPersonTracks, setTimelines, setTimelinesProjectId, setPresets, setPromptBatches, setLoras, setTrainingDatasets, setTrainingDatasetsProjectId, setTrainingDatasetsError, setSelectedTimelineId, setActiveTimeline, setMacCapabilitiesAuthoritative, setMacCapabilitiesLoading, setMacCapabilities, setMacCapabilitiesError, setDomainError, setProjects, setActiveProject, setProjectsLoaded, setJobs, setQueueSummary, setWorkers, setModelCatalogStatus, setModels, domainErrors, setTrainingTargets, setTrainingTargetsError, setTrainingPresets, setTrainingPresetsError, activeProjectRef, loadedAssetsProjectId, setSelectedAssetId, refreshLoras, setupCompleted, authenticated, uiMode, simpleActiveScreen, activeView, refreshCharacters, refreshSavedVoices, refreshPresets, refreshPromptBatches, refreshTrainingDatasets, refreshPersonTracks, refreshTimelines, projectsLoaded, activeAssetLoadState, hasLibraryConsumerAsset
}) {
  const refreshShellDataRef = useRef(null);

  // A retry or token transition supersedes any older capabilities request. Fetch aborts normally
  // provide this guarantee too, but the request id keeps a late/non-cooperative response from
  // restoring stale platform facts after a newer request has started.
  const macCapabilitiesRequestRef = useRef(0);

  const modelCatalogRequestRef = useRef(0);

  const refreshTrainingTargetsRef = useRef(null);

  const refreshTrainingPresetsRef = useRef(null);

  const hydrationLedgerRef = useRef(new Map());

  const committedHydrationRouteRef = useRef(null);

  const previousHydrationProjectIdRef = useRef(null);

  const refreshSavedVoicesRef = useRef(null);

  const refreshLorasRef = useRef(null);

  const refreshPresetsRef = useRef(null);

  const refreshPromptBatchesRef = useRef(null);

  const refreshTrainingDatasetsRef = useRef(null);

  const refreshTimelinesRef = useRef(null);

  useEffect(() => {
    if (!ready) {
      return;
    }
    hydrationLedgerRef.current.clear();
    const controller = new AbortController();
    refreshShellDataRef.current?.({ signal: controller.signal });
    return () => controller.abort();
  }, [ready, token]);

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

  async function refreshShellData({ signal } = {}) {
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
  }

  async function refreshModels({ signal } = {}) {
    const requestId = ++modelCatalogRequestRef.current;
    // Do not blank an already-authoritative catalog during an SSE/manual refresh. This state is
    // specifically the cold-start boundary where no availability answer exists yet.
    setModelCatalogStatus((current) => (current === "ready" ? current : "loading"));
    try {
      const items = await modelsApi.list(token, { signal });
      if (requestId !== modelCatalogRequestRef.current) {
        return refreshFailure("stale");
      }
      setModels(items);
      setModelCatalogStatus("ready");
      domainErrors.models("");
      return refreshSuccess(items);
    } catch (err) {
      if (requestId !== modelCatalogRequestRef.current) {
        return refreshFailure("stale", err);
      }
      if (isAbortError(err)) {
        setModelCatalogStatus((current) => (current === "loading" ? "idle" : current));
        return refreshFailure("aborted", err);
      }
      setModelCatalogStatus((current) => (current === "loading" ? "error" : current));
      domainErrors.models(err.message);
      return refreshFailure("error", err);
    }
  }

  async function refreshTrainingTargets({ signal } = {}) {
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
  }

  async function refreshTrainingPresets({ signal } = {}) {
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
  }

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

  async function refreshModelAndLoras(projectId = activeProjectRef.current?.id) {
    await Promise.all([
      refreshModels(),
      refreshLoras(projectId || undefined),
    ]);
  }

  // sc-8940: Publish the latest refresh closures into their refs from a post-commit
  // effect rather than the render body — React documents render-body ref mutation as
  // unsafe (a discarded concurrent/StrictMode render could leave a ref pointing at an
  // uncommitted closure). No dep array means this runs on every commit, so the refs
  // always hold the newest *committed* body (fresh token / activeProject). We use
  // useLayoutEffect (not useEffect) so the assignment flushes before every passive
  // effect on the same commit: the consumers of these refs (the [ready,token] load
  // effect, the project-switch effect, the SSE handler) are all plain useEffects/event
  // handlers, and React runs all layout effects before any passive effect — preserving
  // the original ordering where consumers saw the fresh closure.
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
    refreshShellDataRef.current = refreshShellData;
    refreshModelsRef.current = refreshModels;
    refreshModelAndLorasRef.current = refreshModelAndLoras;
    refreshTrainingTargetsRef.current = refreshTrainingTargets;
    refreshTrainingPresetsRef.current = refreshTrainingPresets;
    hydrateLaunchDomainsRef.current = hydrateLaunchDomains;
    refreshAssetsRef.current = refreshAssets;
    refreshCharactersRef.current = refreshCharacters;
    refreshSavedVoicesRef.current = refreshSavedVoices;
    refreshLorasRef.current = refreshLoras;
    refreshPresetsRef.current = refreshPresets;
    refreshPromptBatchesRef.current = refreshPromptBatches;
    refreshTrainingDatasetsRef.current = refreshTrainingDatasets;
    refreshPersonTracksRef.current = refreshPersonTracks;
    refreshTimelinesRef.current = refreshTimelines;
    return () => {
      if (committedHydrationRouteRef.current === committedRoute) {
        committedHydrationRouteRef.current = null;
      }
    };
  });

  const hydrationRefresher = useCallback((domain) => {
    const refreshers = {
      assets: (id, options) => refreshAssetsRef.current?.(id, options),
      characters: (id, options) => refreshCharactersRef.current?.(id, options),
      savedVoices: (id, options) => refreshSavedVoicesRef.current?.(id, options),
      loras: (id, options) => refreshLorasRef.current?.(id, options),
      presets: (id, options) => refreshPresetsRef.current?.(id, options),
      promptBatches: (id, options) => refreshPromptBatchesRef.current?.(id, options),
      trainingDatasets: (id, options) =>
        refreshTrainingDatasetsRef.current?.(id, options),
      personTracks: (id, options) => refreshPersonTracksRef.current?.(id, options),
      timelines: (id, options) => refreshTimelinesRef.current?.(id, options),
      models: (_id, options) => refreshModelsRef.current?.(options),
      trainingTargets: (_id, options) =>
        refreshTrainingTargetsRef.current?.(options),
      trainingPresets: (_id, options) =>
        refreshTrainingPresetsRef.current?.(options),
    };
    return refreshers[domain];
  }, [refreshAssetsRef, refreshCharactersRef, refreshModelsRef, refreshPersonTracksRef]);

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

  async function hydrateLaunchDomains(view, projectId = activeProjectRef.current?.id) {
    const domains = hydrationDomainsForRoute(view);
    const results = await Promise.all(
      domains.map((domain) => ensureHydrationDomain(domain, projectId)),
    );
    const failed = results.filter((result) => result?.ok !== true);
    return failed.length
      ? { ok: false, failures: failed }
      : refreshSuccess();
  }
  return { refreshMacCapabilities };
}
