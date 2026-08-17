import { apiFetch } from "./api.js";

// The unavailable-external-library recovery flow (sc-19709).
//
// A model can be fully installed and still unloadable: its weights live on an external library
// (an unplugged drive, an unmounted network volume) that is not attached right now. The API
// answers that with a TYPED refusal — `code` plus a `context` object naming the model and the
// expected library — precisely so the app can offer a recovery instead of printing a filesystem
// error. Nothing here parses a message, and nothing re-derives availability client-side: the
// state comes from the seam, or it does not exist.
//
// The gate below owns one correctness property that is not a UI nicety: a blocked action resumes
// EXACTLY ONCE. Repeated clicks, a poll that notices the drive at the same moment the user hits
// retry, and a reconnect arriving while a retry is already in flight must all converge on a single
// submission — a duplicated generation job costs the user real GPU minutes.

export const MODEL_LIBRARY_UNAVAILABLE_CODE = "external_model_library_unavailable";
export const MODEL_LIBRARY_RELOCATION_REJECTED_CODE = "model_library_relocation_rejected";
const UNAVAILABLE_AVAILABILITY = "installed_external_unavailable";

/// The typed context of an unavailable-library rejection, or `null` for every other failure.
// Both the code and the typed availability must agree: a `code` alone could survive a contract
// change that dropped the payload, and a prompt with no library to name is worse than a notice.
export function modelLibraryContext(error) {
  if (!error || error.code !== MODEL_LIBRARY_UNAVAILABLE_CODE) return null;
  const context = error.context;
  if (!context || context.availability !== UNAVAILABLE_AVAILABILITY) return null;
  return context;
}

// A catalog row is blocked when the seam says so. The listing re-probes live, so this is the same
// judgement the submission preflight would make — read, never recomputed.
export function modelLibraryUnavailable(model) {
  return model?.modelAvailability === UNAVAILABLE_AVAILABILITY;
}

// The context shape the catalog row carries, so selecting an unavailable model opens the same
// prompt as a rejected submission without inventing a second payload shape.
export function modelLibraryContextForModel(model) {
  if (!modelLibraryUnavailable(model)) return null;
  const resolution = model?.modelResolution ?? {};
  const binding = resolution.expectedLibrary ?? null;
  return {
    schemaVersion: resolution.schemaVersion ?? 1,
    availability: UNAVAILABLE_AVAILABILITY,
    modelId: model.id,
    modelName: model.name ?? null,
    configuredLibraryPath: resolution.configuredLibraryPath ?? null,
    expectedLibraryPath: binding?.canonicalPath ?? null,
    expectedVolumeId: binding?.physicalIdentity?.volumeId ?? null,
  };
}

// One write-free identity probe of the configured library. This is what a retry gates on.
export function fetchModelLibraryStatus(token, options) {
  return apiFetch("/api/v1/model-library", token, options);
}

// Check a candidate library root without writing anything, anywhere. Every ordinary refusal — the
// wrong folder, a library missing installed models — lands here, BEFORE either durable write.
export function validateModelLibrary(token, path) {
  return apiFetch("/api/v1/model-library/relocate", token, {
    method: "POST",
    body: JSON.stringify({ path, dryRun: true }),
  });
}

// Adopt a different library root: the API re-binds the durable identity. It never downloads and
// never rewrites install receipts.
export function relocateModelLibrary(token, path) {
  return apiFetch("/api/v1/model-library/relocate", token, {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

// The one place that reacts to an unavailable model library, registered by the app shell. Module
// level — the same pattern as `setUnauthorizedHandler` — so EVERY submission path participates
// without threading a callback through six job creators and two studios, and so a path added later
// is covered by construction rather than by remembering.
let blockedHandler = null;

export function setModelLibraryHandler(handler) {
  const next = typeof handler === "function" ? handler : null;
  blockedHandler = next;
  return () => {
    if (blockedHandler === next) blockedHandler = null;
  };
}

// Hand a rejection to the prompt. Returns true when the prompt owns it, so the caller suppresses
// its own error surface — a raw filesystem/loader message must never reach the user for this case.
export function handleModelLibraryRejection(error, action) {
  const context = modelLibraryContext(error);
  if (!context) return false;
  return blockedHandler?.(context, action) === true;
}

// The failure a call site rethrows once the prompt has taken ownership. Its message is empty on
// purpose: the ~85 `setError(err.message)` / `setConfigError(err.message)` handlers in this app then
// CLEAR their banner instead of printing a second, redundant surface beside the dialog.
export class ModelLibraryPrompted extends Error {
  constructor() {
    super("");
    this.name = "ModelLibraryPrompted";
  }
}

// The throw-shaped counterpart of `handleModelLibraryRejection`, for the submission paths that
// propagate their failure to a caller rather than swallowing it (training). Always throws.
export function rethrowUnlessPrompted(error, action) {
  if (handleModelLibraryRejection(error, action)) {
    throw new ModelLibraryPrompted();
  }
  throw error;
}

// Raise the prompt for a model the catalog already reports as unavailable (selection time), with
// no queued action: the recovery is the point, and there is nothing to resume.
export function raiseModelLibraryPrompt(context) {
  if (!context) return false;
  return blockedHandler?.(context, null) === true;
}

const IDLE = Object.freeze({
  status: "idle",
  context: null,
  hint: "",
  error: "",
  relocated: null,
});

/// A blocked-action gate: at most one pending action, resumed at most once.
//
// Injected so the state machine is testable without a transport:
//   `probe`    — the seam's library status; a retry gates on its `available`.
//   `validate` — check a candidate root, writing nothing anywhere.
//   `adopt`    — re-bind the server's durable identity to that root.
//   `persist`  — store the client's own copy of the location; may return an undo function.
//
// Relocation has TWO durable writes in different places (the shell's `HF_HOME` and the server's
// binding ledger) and they must not be allowed to disagree: `validate` first so every ordinary
// refusal happens before either, then `persist`, then `adopt`, and if `adopt` fails the persist is
// undone. What the user is told always matches what actually happened.
export function createModelLibraryGate({ probe, validate, adopt, persist } = {}) {
  let state = IDLE;
  // The single pending action. It is REMOVED from this slot before it runs, so no second caller —
  // a double click, a poll, a reconnect event — can ever find it again to run it twice.
  let pending = null;
  // Bumped by every attempt and by cancel. An attempt whose epoch is stale when its probe finally
  // answers has been superseded (the user cancelled, or a newer attempt owns the gate) and must
  // neither run the action nor write state — otherwise "Cancel" could still submit the job.
  let epoch = 0;
  const listeners = new Set();

  function emit(next) {
    state = Object.freeze({ ...state, ...next });
    for (const listener of listeners) listener();
  }

  function takePending() {
    const action = pending;
    pending = null;
    return action;
  }

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    // Capture a rejection. Returns true when the gate owns it, so the caller suppresses its own
    // error surface. A second rejection while blocked is DROPPED rather than queued: one dialog,
    // one pending action, and no hidden backlog to fire later.
    block(context, action) {
      if (!context) return false;
      if (state.status !== "idle") return true;
      pending = typeof action === "function" ? action : null;
      emit({ status: "blocked", context, hint: "", error: "", relocated: null });
      return true;
    },

    // Re-probe and resume. A no-op unless the gate is settled in `blocked`, which is what makes a
    // reconnect that lands mid-retry harmless. `auto` marks a poll-driven attempt, whose failure is
    // silence rather than an error the user did not ask for.
    async retry({ auto = false } = {}) {
      if (state.status !== "blocked") return null;
      const action = takePending();
      const attempt = ++epoch;
      emit({ status: "retrying", hint: "", error: "" });
      let available = false;
      try {
        available = (await probe?.())?.available === true;
      } catch (error) {
        if (attempt !== epoch) return null;
        pending = action;
        emit({
          status: "blocked",
          error: auto ? "" : error?.message || "The model library could not be checked.",
        });
        return null;
      }
      if (attempt !== epoch) return null;
      if (!available) {
        pending = action;
        emit({
          status: "blocked",
          // Clearing `error` matters: a stale failure from an earlier attempt must not sit in the
          // live region next to this attempt's hint, reading as two simultaneous answers.
          error: "",
          hint: auto ? "" : "That library is still not connected.",
        });
        return null;
      }
      emit({ ...IDLE });
      return action ? action() : null;
    },

    // Adopt a different library root. The pending action is dropped, not resumed: the relocated
    // path becomes effective for the sidecars on the next launch, so resuming now would only be
    // refused again. Dropping it is the same guarantee cancel gives — nothing left queued.
    async relocate(path) {
      if (state.status !== "blocked") return null;
      const action = takePending();
      const attempt = ++epoch;
      emit({ status: "relocating", hint: "", error: "" });

      // 1. Validate. Nothing is written, so a refusal here leaves the blocked action intact and
      //    the user simply picks again.
      let target = null;
      try {
        target = (await validate?.(path)) ?? { hfHome: path, libraryRoot: path };
      } catch (error) {
        if (attempt !== epoch) return null;
        pending = action;
        emit({
          status: "blocked",
          hint: "",
          error: error?.message || "That library could not be used.",
        });
        return null;
      }

      // 2. Persist the client's copy, then 3. re-bind the server's. If the re-bind fails, undo the
      //    persist so the two cannot disagree — and say what actually happened either way.
      let undoPersist = null;
      try {
        undoPersist = (await persist?.(target)) ?? null;
        await adopt?.(path);
      } catch (error) {
        let restored = true;
        try {
          await undoPersist?.();
        } catch {
          restored = false;
        }
        if (attempt !== epoch) return null;
        pending = action;
        emit({
          status: "blocked",
          hint: "",
          error: restored
            ? `SceneWorks could not finish moving the model library, so the previous location is still in use. ${error?.message ?? ""}`.trim()
            : "SceneWorks could not finish moving the model library, and could not restore the previous location. Set the model library folder in Settings.",
        });
        return null;
      }
      if (attempt !== epoch) return null;
      emit({ ...IDLE, relocated: target });
      return target;
    },

    // Abandon the blocked action outright. Nothing is queued, nothing partial is left behind, and
    // an attempt already in flight is superseded so it cannot resume behind the user's back.
    cancel() {
      takePending();
      epoch += 1;
      emit({ ...IDLE });
    },
  };
}
