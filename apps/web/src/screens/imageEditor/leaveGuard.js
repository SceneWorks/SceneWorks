

// Browser unload and page navigation both release the editor document.
// Dirty edits and an unfinished AI edit require a loss-of-work warning.
export function leaveGuardMessage({ dirty, aiOpPending }) {
  if (dirty) return "You have unsaved edits in the Image Editor that will be lost if you close or reload.";
  if (aiOpPending) return "An image edit is still running; its result will be lost if you close or reload.";
  return null;
}

// The browser warning and desktop-safe navigation confirm protect the same
// document; a clean and idle editor can leave without a prompt.
export function leaveGuardArming({ dirty, aiOpPending }) {
  const message = leaveGuardMessage({ dirty, aiOpPending });
  return {
    message,
    beforeUnload: Boolean(message),
    inApp: Boolean(message),
  };
}

// The confirm prompt shown when the user EXPLICITLY closes/discards the working image
// (the Close button, sc-11968). Unlike the passive leave guard, a deliberate close of a
// CLEAN document loses nothing, so it needs no confirm (null → proceed silently). Unsaved
// edits win the wording; an in-flight AI op is the fallback (its result would be abandoned).
export function closeConfirmMessage({ dirty, aiOpPending }) {
  if (dirty) return "Discard your unsaved edits and close this image?";
  if (aiOpPending) return "An image edit is still running. Close and abandon it?";
  return null;
}

// Which save-state indicator the top bar shows (sc-11968): the unsaved-edits pill while
// `dirty`, the "Saved ✓" hint once a Save has landed and nothing has changed since, else
// nothing. Pure so the badge logic is unit-testable without a mounted canvas.
export function saveStatusIndicator({ dirty, savedAssetId }) {
  if (dirty) return "unsaved";
  if (savedAssetId) return "saved";
  return null;
}
