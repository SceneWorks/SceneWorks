import { useEffect, useState } from "react";

import { checkIrisPlan, fetchIrisMetrics, fetchIrisStagePresets, isAbortError } from "../api.js";

// Iris curricula hooks (sc-25689). All non-blocking: a failed request shows nothing; the API still
// validates the run itself at submit.

export function useIrisStagePresets(enabled, token) {
  const [presets, setPresets] = useState([]);
  useEffect(() => {
    if (!enabled) return undefined;
    const controller = new AbortController();
    fetchIrisStagePresets(token, controller.signal)
      .then((body) => !controller.signal.aborted && setPresets(body?.presets ?? []))
      .catch(() => {});
    return () => controller.abort();
  }, [enabled, token]);
  return enabled ? presets : [];
}

// The pre-launch plan check, re-asked (debounced) whenever the submitted config changes.
export function useIrisPlanCheck({ enabled, targetId, config, token }) {
  const [result, setResult] = useState(null);
  const key = enabled && targetId && config ? JSON.stringify(config) : "";
  useEffect(() => {
    if (!key) {
      setResult(null);
      return undefined;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      checkIrisPlan(targetId, JSON.parse(key), token, controller.signal)
        .then((check) => !controller.signal.aborted && setResult(check ?? null))
        .catch((err) => !isAbortError(err) && !controller.signal.aborted && setResult(null));
    }, 400);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key, targetId, token]);
  return key ? result : null;
}

// A run's metrics; re-read every `pollMs` while `live`.
export function useIrisMetrics(jobId, token, { live = false, pollMs = 15000 } = {}) {
  const [metrics, setMetrics] = useState(null);
  useEffect(() => {
    if (!jobId) {
      setMetrics(null);
      return undefined;
    }
    const controller = new AbortController();
    const load = () =>
      fetchIrisMetrics(jobId, token, controller.signal)
        .then((body) => !controller.signal.aborted && setMetrics(body ?? null))
        .catch(() => {});
    load();
    const timer = live ? setInterval(load, pollMs) : null;
    return () => {
      if (timer) clearInterval(timer);
      controller.abort();
    };
  }, [jobId, token, live, pollMs]);
  return jobId ? metrics : null;
}
