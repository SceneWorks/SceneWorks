// Licence-required UI attribution, on the surfaces where the model is actually USED (sc-17161).
//
// Some upstream licences oblige the product to display a specific string on its user interface —
// MiniMax H3's §IV.2 is the case that forced this: "You shall prominently display 'MiniMax H3' on
// the user interface of commercial product or service that uses MiniMax H3 or MiniMax H3 Works."
// sc-17227 landed it on the Models card, which discharges the obligation on exactly one screen a
// user may never revisit after installing. The GENERATION surfaces are where the model is used,
// and `docs/minimax-h3-use-restriction-safeguards.md` S4 names landing it there as a shipping
// precondition rather than an extra.
//
// The manifest field `ui.attribution` is the single source. Nothing here hard-codes a string: a
// second copy is a copy that can drift out of agreement with the licence it exists to satisfy,
// and the exact wording (a SPACE, not the hyphenated repo name) is the part §IV.2 requires.
//
// Two ways to reach the string, because the two surface families have different data in hand:
//   * the studios hold the resolved catalog entry, and pass `model`;
//   * the queue and the asset detail hold only a model ID string (`job.payload.model`,
//     `asset.recipe.model`) and never join it back to the catalog, so they pass `modelId` and
//     this component does the lookup against the catalog on the app context.
//
// Rendering nothing is always correct: a model that declares no attribution owes none, and this
// is per-model, never a global banner.

import React, { useContext, useMemo } from "react";
import { AppStaticContext, AppContext } from "../context/AppContext.js";

// The catalog, or an empty list when rendered outside the providers. Deliberately does NOT use
// `useAppContextOptional()`: that merges all three contexts on every job/worker tick, and this
// component renders once per queue row and once per asset card. `models` lives on the static
// context, so subscribing to it alone keeps the attribution off the live re-render path.
function useCatalogModels() {
  const staticValue = useContext(AppStaticContext);
  const combined = useContext(AppContext);
  const models = staticValue?.models ?? combined?.models;
  return Array.isArray(models) ? models : [];
}

/**
 * The licence-required attribution line for a model, or `null`.
 *
 * Pass `model` (a resolved catalog entry) where you have one; pass `modelId` where you only have
 * the id the recipe or job payload recorded.
 */
export function ModelAttribution({ model = null, modelId = "", className = "model-attribution" }) {
  const models = useCatalogModels();
  const resolved = useMemo(() => {
    if (model) return model;
    if (!modelId) return null;
    return models.find((item) => item?.id === modelId) ?? null;
  }, [model, modelId, models]);
  const attribution = resolved?.ui?.attribution;
  if (typeof attribution !== "string" || !attribution.trim()) return null;
  return <p className={className}>{attribution.trim()}</p>;
}
