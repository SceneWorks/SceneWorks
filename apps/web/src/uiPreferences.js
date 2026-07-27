// The single client seam for writing the durable UI-preferences blob (sc-15136).
//
// `/api/v1/ui-preferences` is METHOD-ASYMMETRIC on the server (sc-8869, F-067 —
// `auth::requires_token`): the GET is public so the pre-auth theme read can load before
// the access gate and avoid a flash, but the PUT writes `ui-preferences.json` to disk and
// is GATED like every other write (epic 4484: every write is authenticated). Sending the
// PUT without the access token therefore 401s on any remote-auth deployment.
//
// That is exactly what shipped before sc-15136: six independent call sites (theme, accent,
// the Simple-UI default, the default generation quality in both shells, and the per-model
// tier sticky) each hand-rolled the same `apiFetch(path, "", { method: "PUT" })`, three of
// them under a stale "it's a public route" comment — so on a LAN browser none of those
// preferences ever reached disk. Five of the six swallowed the 401 with `.catch(() => {})`,
// and every one of them also writes a localStorage instant-paint cache, so the UI looked
// correct until localStorage was cleared or another device read the file. (The exception is
// the advanced Settings screen, which awaits the PUT and DOES surface the failure while
// rolling its chip back — the one site where the bug was visible rather than silent.)
//
// Routing every write through this one function is the fix AND the guard: the token seam
// exists once, so a seventh preference can't silently reintroduce the same 401. The GET is
// deliberately NOT routed here — it stays credential-less in `App.jsx` because it runs
// before the gate has a token to send, and the server exempts it.

import { apiFetch } from "./api.js";
import { readAccessToken } from "./accessToken.js";

// PUT a partial preferences patch. The endpoint MERGES, so a caller sends only the field it
// owns and cannot clobber the others.
//
// The token comes from `readAccessToken()` rather than a prop/context so the non-component
// callers (`lastTierStore.js`) reach it without prop-drilling. It is read at CALL time, not at
// import time: the token is promoted mid-session when the user answers the access gate, so a
// cached one would leave every write in that session unauthenticated. (`credentials.js`'s
// `serverToken()` is the same value under a different name — one source, in `accessToken.js`.)
//
// Reading storage is the right call rather than a bug to route around: it is strictly
// fresher than the gate's state on promotion, which is the case that decides whether a
// session's writes are authenticated at all.
//
// The CROSS-TAB divergence that used to qualify that is closed (sc-15165): the gate now
// subscribes to `storage` events via `subscribeAccessToken`, so a "forget" in a second tab
// locks this one instead of leaving it live while writes here go out credential-less to a
// 401 that (correctly) never raises the gate. One case survives, deliberately: a same-tab
// `storeAccessToken` that the browser REJECTS (private-mode / quota — swallowed by design,
// see the degradation note in `accessToken.js`) leaves the gate accepted with storage
// empty, and writes here go out credential-less again. No `storage` event can fix that one;
// it needs the gate to stop treating an unverifiable write as a successful one.
//
// On the desktop shell it is "" and the request still succeeds: `SCENEWORKS_TRUST_LOOPBACK`
// bypasses the token check for loopback peers before any comparison, so local use stays
// password-free — which is also why the fix is invisible on the desktop.
//
// Returns the `apiFetch` promise so a caller that reports failure (SettingsScreen) can await
// it; the fire-and-forget callers attach their own `.catch`.
export function putUiPreferences(patch) {
  return apiFetch("/api/v1/ui-preferences", readAccessToken(), {
    method: "PUT",
    body: JSON.stringify(patch),
  });
}

let pendingNavigationPatch = null;
let navigationDrainPromise = null;
let navigationDrainScheduled = false;

async function drainNavigationPreferences() {
  navigationDrainScheduled = false;
  while (pendingNavigationPatch) {
    const patch = pendingNavigationPatch;
    pendingNavigationPatch = null;
    // Read the access token when this request starts, not when the patch was
    // enqueued. A token can be promoted or rotated while an earlier PUT is in flight.
    await putUiPreferences(patch).catch(() => {});
  }
  navigationDrainPromise = null;
}

// Serialize durable navigation writes and coalesce intent that arrives before
// the next request starts. With at most one PUT in flight, an older delayed
// response cannot persist after a newer selection or destination.
export function persistNavigationPreferences(patch) {
  pendingNavigationPatch = { ...(pendingNavigationPatch ?? {}), ...patch };
  if (!navigationDrainPromise) {
    navigationDrainPromise = new Promise((resolve) => {
      if (!navigationDrainScheduled) {
        navigationDrainScheduled = true;
        queueMicrotask(() => resolve(drainNavigationPreferences()));
      }
    });
  }
  return navigationDrainPromise;
}

export async function resetNavigationPreferenceQueueForTests() {
  await navigationDrainPromise;
  pendingNavigationPatch = null;
  navigationDrainPromise = null;
  navigationDrainScheduled = false;
}
