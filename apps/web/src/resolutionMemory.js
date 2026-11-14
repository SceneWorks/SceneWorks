// Resolution memory gating (sc-13959, epic 13879). A bucket above the historical 1536² ceiling is
// hidden when the host cannot fit its predicted peak. The API owns the prediction: it serves, per
// lane, the minimum host memory for every declared bucket above the ceiling
// (`memoryGuidance.<lane>.resolutionMinHostGb`), and per installed tier whose own lane evidence
// carries its measurement geometry (`memoryGuidance.<lane>.tiers.<tier>.resolutionMinHostGb`).
//
// Never withheld on missing data: an unknown host reading, an entry without served guidance, a
// malformed or at/below-ceiling bucket, or a model with no floor on the active lane is always offered.

import { installedTiers, memoryLane } from "./quantTier.js";

// Whether `resolution` should be OFFERED for `model` on a host with `unifiedMemoryGb`.
// `options.tier` is the tier the studio will generate with; a stale (no longer installed) tier
// reads the lane-level thresholds, never a removed tier's evidence (sc-16020).
export function resolutionFitsMemory(model, resolution, unifiedMemoryGb, options = {}) {
  const { backend, tier = null } = options;
  if (unifiedMemoryGb == null || !Number.isFinite(unifiedMemoryGb)) {
    return true;
  }
  const lane = memoryLane(model, backend);
  if (lane === null) {
    return true;
  }
  const selected =
    typeof tier === "string" && tier && installedTiers(model, { backend, tier }).includes(tier)
      ? lane.tiers?.[tier]?.resolutionMinHostGb
      : null;
  const minHostGb = (selected ?? lane.resolutionMinHostGb)?.[resolution];
  return typeof minHostGb !== "number" || unifiedMemoryGb >= minHostGb;
}

// Filter a list of "WxH" buckets to those that fit `unifiedMemoryGb`, preserving order.
export function fitsResolutionOptions(model, resolutions, unifiedMemoryGb, options = {}) {
  if (!Array.isArray(resolutions)) {
    return [];
  }
  return resolutions.filter((resolution) =>
    resolutionFitsMemory(model, resolution, unifiedMemoryGb, options),
  );
}
