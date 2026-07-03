import { createContext, useContext, useMemo } from "react";

// Shared app data/actions provider (sc-1651 Phase B). App composes the per-domain
// hooks (Phase A) and exposes the primitives here so screens read what they need via
// useAppContext() instead of receiving dozens of drilled props. Screens build any
// screen-specific wrappers (e.g. send-to-studio with a mode) from these primitives.
//
// sc-8855 (F-053): the single monolithic context re-rendered EVERY consumer on every
// SSE tick, because jobs/workers change identity continuously and that gave the one
// combined value a new identity each tick. Split into two providers:
//   - AppLiveContext  — high-churn fields derived from jobs/workers (change per SSE tick)
//   - AppStaticContext — low-churn actions/catalogs/project/presets/characters/training
// A consumer that reads ONLY static fields subscribes via useAppStatic() and no longer
// re-renders on job/worker ticks. Consumers that read live fields (or both) use
// useAppContext(), the backward-compatible combined hook that merges both contexts so
// every existing destructure keeps working with no field loss.
//
// AppContext (the legacy combined context) is retained as a THIRD provider that carries a
// pre-merged flat value. App does NOT use it (App renders the two split providers), but
// tests and any external code that render a single <AppContext.Provider value={merged}>
// keep working: each hook below falls back to the legacy combined value when its split
// context is absent. This is what makes the split backward compatible.

// The keys carried by the high-churn (live) context. Derived from the SSE-updated
// `jobs`/`workers` state, so they change identity on virtually every tick. Kept here so
// App can build the two value objects from one source of truth.
export const LIVE_CONTEXT_KEYS = Object.freeze([
  "jobs",
  "filteredJobs",
  "imageLocalJobs",
  "videoLocalJobs",
  "documentLocalJobs",
  "visibleWorkers",
  "workersById",
  "personReadiness",
  "gpuOptions",
]);

export const AppStaticContext = createContext(null);
export const AppLiveContext = createContext(null);

// Legacy combined context. Feeds a single flat value (static + live merged). App renders
// the two split providers instead; this exists so a lone <AppContext.Provider value={…}>
// (existing tests, external callers) still satisfies every hook via the fallbacks below.
export const AppContext = createContext(null);

export function useAppStatic() {
  const split = useContext(AppStaticContext);
  const combined = useContext(AppContext);
  const value = split ?? combined;
  if (value === null || value === undefined) {
    throw new Error("useAppStatic must be used within an <AppStaticContext.Provider> or <AppContext.Provider>");
  }
  return value;
}

export function useAppLive() {
  const split = useContext(AppLiveContext);
  const combined = useContext(AppContext);
  const value = split ?? combined;
  if (value === null || value === undefined) {
    throw new Error("useAppLive must be used within an <AppLiveContext.Provider> or <AppContext.Provider>");
  }
  return value;
}

// Backward-compatible combined hook: merges the two split contexts into the flat shape
// the monolithic context used to expose. Consumers that read live fields (or a mix of
// live + static) keep using this and lose no field. Consumers that read only static
// fields should switch to useAppStatic() so they stop re-rendering on job/worker ticks.
//
// Reading the live context here still subscribes the caller to it, so a useAppContext()
// consumer re-renders on ticks exactly as before — intentional (no behavior change for
// those callers). The win is that cold-only callers opt out via useAppStatic().
//
// When only the legacy combined <AppContext.Provider> is present (tests), both split
// contexts are null and we return the combined value directly.
export function useAppContext() {
  const staticValue = useContext(AppStaticContext);
  const liveValue = useContext(AppLiveContext);
  const combined = useContext(AppContext);
  const merged = useMemo(() => {
    if (staticValue === null && liveValue === null) {
      return combined;
    }
    return { ...(combined ?? {}), ...(staticValue ?? {}), ...(liveValue ?? {}) };
  }, [staticValue, liveValue, combined]);
  if (merged === null || merged === undefined) {
    throw new Error("useAppContext must be used within the App context providers");
  }
  return merged;
}
