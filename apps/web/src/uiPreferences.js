import { apiFetch } from "./api.js";

let pendingPatch = null;
let pendingToken = "";
let drainPromise = null;
let scheduled = false;

async function drain() {
  scheduled = false;
  while (pendingPatch) {
    const patch = pendingPatch;
    const token = pendingToken;
    pendingPatch = null;
    await apiFetch("/api/v1/ui-preferences", token, {
      method: "PUT",
      body: JSON.stringify(patch),
    }).catch(() => {});
  }
  drainPromise = null;
}

/// Serializes durable navigation writes and coalesces intent that arrives before
/// the next request starts. There can be at most one PUT in flight, so an older
/// delayed response can never persist after a newer selection or destination.
export function persistNavigationPreferences(patch, token = "") {
  pendingPatch = { ...(pendingPatch ?? {}), ...patch };
  pendingToken = token;
  if (!drainPromise) {
    drainPromise = new Promise((resolve) => {
      if (!scheduled) {
        scheduled = true;
        queueMicrotask(() => resolve(drain()));
      }
    });
  }
  return drainPromise;
}

export async function resetNavigationPreferenceQueueForTests() {
  await drainPromise;
  pendingPatch = null;
  pendingToken = "";
  drainPromise = null;
  scheduled = false;
}
