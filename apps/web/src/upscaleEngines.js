// Single source of truth for the upscale-engine table + its stale-selection
// fallback (sc-8853). The table lived twice — here-adjacent in imageJobs.js
// (keyed `key`) and copy-pasted into ImageStudio (keyed `id`) — and the
// "gated-out engine falls back to real-esrgan" effect was copy-pasted into
// both ImageStudio and ImageEditor. Unifying here means adding/gating an engine
// is a one-place edit.
//
// The canonical shape keys on `key` (imageJobs.js already exported it that way
// and ImageEditor reads `entry.key`); the former ImageStudio `.id` consumers
// were migrated to `.key`.
//
// This module owns the React hook; imageJobs.js re-exports the pure table +
// helpers so its React/Konva-free contract (and existing importers) are
// unchanged.
import { useEffect } from "react";
import { UPSCALE_ENGINES, upscaleFactorsForEngine } from "./imageJobs.js";
import { macUpscaleEngineBlocked } from "./macGating.js";

export { UPSCALE_ENGINES, upscaleFactorsForEngine, upscaleEngineHasSoftness } from "./imageJobs.js";

// The guaranteed-available cross-platform upscaler; the fallback target when a
// saved/selected engine is gated out on this platform.
export const DEFAULT_UPSCALE_ENGINE = "real-esrgan";

// The engines offered in the picker on this platform (AuraSR is dropped
// everywhere, SeedVR2 is platform-intrinsic) — filters the shared table so both
// studios build the dropdown identically.
export function availableUpscaleEngines(macCapabilities) {
  return UPSCALE_ENGINES.filter((engine) => !macUpscaleEngineBlocked(macCapabilities, engine.key));
}

// If the currently-selected upscale engine is gated out on this platform (e.g. a
// stale saved AuraSR selection restored from settings), snap it back to the
// default engine and clamp the factor to one the default supports — so the user
// never submits a job the native workers refuse. Shared by ImageStudio and
// ImageEditor, which previously each copy-pasted this effect.
export function useUpscaleEngineFallback({
  macCapabilities,
  upscaleEngine,
  setUpscaleEngine,
  upscaleFactor,
  setUpscaleFactor,
}) {
  useEffect(() => {
    if (!macUpscaleEngineBlocked(macCapabilities, upscaleEngine)) return;
    setUpscaleEngine(DEFAULT_UPSCALE_ENGINE);
    const factors = upscaleFactorsForEngine(DEFAULT_UPSCALE_ENGINE);
    if (!factors.includes(upscaleFactor)) setUpscaleFactor(factors[0]);
  }, [macCapabilities, upscaleEngine, upscaleFactor, setUpscaleEngine, setUpscaleFactor]);
}
