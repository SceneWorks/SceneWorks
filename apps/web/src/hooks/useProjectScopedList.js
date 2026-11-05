import { useCallback, useLayoutEffect, useRef } from "react";
import { isAbortError } from "../api.js";
import { isCurrentProjectRequest } from "../appStateHelpers.js";
import { refreshFailure, refreshSuccess } from "../refreshResult.js";

// One owner for project-list request admission, stale/abort suppression and results.
// Domain callbacks retain selection, loading, normalization and error presentation.
// The committed options ref keeps refresh stable without publishing abandoned renders.
export function useProjectScopedList(options) {
  const { token, activeProject, activeProjectRef, optionalProject = false } = options;
  const optionsRef = useRef(options);
  const requestIdRef = useRef(0);
  useLayoutEffect(() => { optionsRef.current = options; });
  return useCallback(async (projectId = activeProject?.id, { signal } = {}) => {
    const current = () => !projectId && optionalProject ||
      isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId ?? null);
    const config = optionsRef.current;
    // Training mutations explicitly reload a named project through the live ref.
    // Render-owned lists admit from this render before passive refs catch up.
    const admissionProjectId = config.admitFromRef ? activeProjectRef?.current?.id : activeProject?.id;
    if (projectId && !isCurrentProjectRequest(admissionProjectId ?? null, projectId)) {
      return refreshFailure("stale");
    }
    if (config.admitFromRef && !current()) return refreshFailure("stale");
    const requestId = ++requestIdRef.current;
    const ownsRequest = () => current() && requestIdRef.current === requestId;
    if (!projectId && !optionalProject) {
      config.onMissing?.();
      return refreshFailure("missing-project");
    }
    const context = config.onStart?.(projectId ?? null);
    try {
      const items = await config.load(projectId, token, { signal });
      if (!ownsRequest()) return refreshFailure("stale");
      if (signal?.aborted) return refreshFailure("aborted");
      config.setItems?.(config.normalize ? config.normalize(items) : items);
      config.onSuccess?.(items, projectId ?? null);
      config.setError?.("");
      return refreshSuccess(items);
    } catch (error) {
      if (!ownsRequest()) return refreshFailure("stale", error);
      if (isAbortError(error) || signal?.aborted) return refreshFailure("aborted", error);
      config.onError?.(error, projectId ?? null);
      config.setError?.(error.message);
      return refreshFailure("error", error);
    } finally {
      config.onSettled?.({ projectId, context, ownsRequest: ownsRequest(), aborted: signal?.aborted === true });
    }
  }, [activeProject, activeProjectRef, optionalProject, token]);
}
