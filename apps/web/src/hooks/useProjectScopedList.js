import { useCallback, useLayoutEffect, useRef } from "react";
import { isAbortError } from "../api.js";
import { isCurrentProjectRequest } from "../appStateHelpers.js";
import { refreshFailure, refreshSuccess } from "../refreshResult.js";

// One owner for project-list request admission, stale/abort suppression and results.
// Domain callbacks retain selection, loading, normalization and error presentation.
// The committed options ref keeps refresh stable without publishing abandoned renders.
export function useProjectScopedList(options) {
  const { token, activeProject, activeProjectRef, optionalProject = false, resident = true, scopeKey } = options;
  const optionsRef = useRef(options);
  const requestIdRef = useRef(0);
  const controllersRef = useRef(new Set());
  useLayoutEffect(() => { optionsRef.current = options; });
  useLayoutEffect(() => () => {
    requestIdRef.current += 1;
    for (const controller of controllersRef.current) controller.abort();
    controllersRef.current.clear();
  }, [resident, activeProject?.id, token, scopeKey]);
  return useCallback(async (projectId = activeProject?.id, { signal } = {}) => {
    const current = () => !projectId && optionalProject ||
      isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId ?? null);
    const config = optionsRef.current;
    if (config.resident === false) return refreshFailure("inactive");
    // Training mutations explicitly reload a named project through the live ref.
    // Render-owned lists admit from this render before passive refs catch up.
    const admissionProjectId = config.admitFromRef ? activeProjectRef?.current?.id : activeProject?.id;
    if (projectId && !isCurrentProjectRequest(admissionProjectId ?? null, projectId)) {
      return refreshFailure("stale");
    }
    if (config.admitFromRef && !current()) return refreshFailure("stale");
    const requestId = ++requestIdRef.current;
    const ownsRequest = () => optionsRef.current.resident !== false && current() && requestIdRef.current === requestId;
    if (!projectId && !optionalProject) {
      config.onMissing?.();
      return refreshFailure("missing-project");
    }
    // Progress can request another refresh before the previous fetch settles.
    // This is a latest-result list: abort superseded work instead of retaining
    // one in-flight response per completed image during a long batch.
    for (const previous of controllersRef.current) previous.abort();
    controllersRef.current.clear();
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) controller.abort();
    controllersRef.current.add(controller);
    const context = config.onStart?.(projectId ?? null);
    try {
      const items = await config.load(projectId, token, { signal: controller.signal });
      if (!ownsRequest()) return refreshFailure("stale");
      if (controller.signal.aborted) return refreshFailure("aborted");
      config.setItems?.(config.normalize ? config.normalize(items) : items);
      config.onSuccess?.(items, projectId ?? null);
      config.setError?.("");
      return refreshSuccess(items);
    } catch (error) {
      if (!ownsRequest()) return refreshFailure("stale", error);
      if (isAbortError(error) || controller.signal.aborted) return refreshFailure("aborted", error);
      config.onError?.(error, projectId ?? null);
      config.setError?.(error.message);
      return refreshFailure("error", error);
    } finally {
      controllersRef.current.delete(controller);
      signal?.removeEventListener("abort", abort);
      config.onSettled?.({ projectId, context, ownsRequest: ownsRequest(), aborted: controller.signal.aborted });
    }
  }, [activeProject, activeProjectRef, optionalProject, token]);
}
