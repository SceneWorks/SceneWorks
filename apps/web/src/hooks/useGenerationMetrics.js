import { useCallback, useEffect, useRef, useState } from "react";

import { apiFetch, isAbortError } from "../api.js";

// Fetches the aggregate generation-metrics feed (epic 10402, GET /api/v1/metrics)
// for the Generation Stats screen. Screen-local rather than in the app context:
// metrics are low-churn and only this screen consumes them, so keeping them out of
// the perf-sensitive split context (sc-8855) avoids re-render churn on the live
// path. Each row is a GenerationMetricsRow: { jobId, type, status, projectId,
// createdAt, metrics: { model, quantLabel, sampler, ... loadMs, peakMemoryPct } }.
//
// `params` may carry { type, model, quant, limit } — server-side filters. Refetches
// whenever `token`, `enabled`, or the params change; exposes `refresh()` for a
// manual reload (e.g. after a job completes).
export function useGenerationMetrics({ token, enabled = true, params = {} } = {}) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestControllerRef = useRef(null);
  // Stable dependency for the filter object without depending on its identity.
  const paramsKey = JSON.stringify(params ?? {});

  const load = useCallback(
    async (signal) => {
      setLoading(true);
      setError("");
      try {
        const query = new URLSearchParams();
        for (const [key, value] of Object.entries(JSON.parse(paramsKey))) {
          if (value !== undefined && value !== null && value !== "") {
            query.set(key, String(value));
          }
        }
        const suffix = query.toString() ? `?${query.toString()}` : "";
        const items = await apiFetch(`/api/v1/metrics${suffix}`, token, { signal });
        if (!signal?.aborted) {
          setRows(Array.isArray(items) ? items : []);
        }
      } catch (err) {
        if (!signal?.aborted && !isAbortError(err)) {
          setError(err?.message ?? "Failed to load generation metrics");
        }
      } finally {
        if (!signal?.aborted) {
          setLoading(false);
        }
      }
    },
    [token, paramsKey],
  );

  const startLoad = useCallback(() => {
    requestControllerRef.current?.abort();
    const controller = new AbortController();
    requestControllerRef.current = controller;
    const request = load(controller.signal);
    request.finally(() => {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
      }
    });
    return request;
  }, [load]);

  useEffect(() => {
    if (!enabled) {
      requestControllerRef.current?.abort();
      requestControllerRef.current = null;
      setLoading(false);
      return undefined;
    }
    startLoad();
    return () => {
      requestControllerRef.current?.abort();
    };
  }, [enabled, startLoad]);

  // Manual and filter-driven loads share one controller. Starting either aborts
  // the previous request so a superseded filter can never commit its rows.
  const refresh = useCallback(() => startLoad(), [startLoad]);

  return { rows, loading, error, refresh };
}
