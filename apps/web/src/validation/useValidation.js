// The single call a form makes to learn whether it may submit and what it owes the
// user (epic 10644, sc-10645). Both answers come from one `summarize()`, so a screen
// cannot enable its CTA and hide the reason it should have been disabled.
//
// `rules` is a plain `(draft, ctx) => Issue[]` colocated with the draft module it
// validates, not with the screen that renders it (epic 10644, R3).

import { useMemo } from "react";

import { summarize } from "./issues.js";

// The memo earns its keep only when `rules`, `draft` and `ctx` are themselves stable;
// a `ctx` object literal built inline changes identity every render, so the summary
// recomputes and the result is a fresh object. That costs nothing but a few predicate
// calls over a draft, and it never costs correctness — memoize `ctx` if a downstream
// memo depends on the summary's identity.
export function useValidation(rules, draft, ctx) {
  return useMemo(() => summarize(rules(draft, ctx)), [rules, draft, ctx]);
}
