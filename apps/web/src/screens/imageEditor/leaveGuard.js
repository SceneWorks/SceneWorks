

// The browser CLOSE/REFRESH (beforeunload) warning shown while there is unsaved work, or
// null when a real unload would lose nothing (sc-2434 / sc-8850). Under keep-alive the
// editor stays mounted across in-app navigation, so a plain nav is non-destructive and gets
// NO prompt (see leaveGuardArming); only a genuine browser unload drops the in-memory state,
// which is what this wording addresses. Unsaved edits win the message. It also fires while an
// AI op is in flight — starting one does NOT set `dirty` (its result only lands on success) —
// so an unload mid-op is still flagged (its result would be lost).
export function leaveGuardMessage({ dirty, aiOpPending }) {
  if (dirty) return "You have unsaved edits in the Image Editor that will be lost if you close or reload.";
  if (aiOpPending) return "An image edit is still running; its result will be lost if you close or reload.";
  return null;
}

// Decide which leave-guard to arm (sc-11959 / sc-11968). Under selective keep-alive an in-app
// navigation away does NOT unmount the editor — unsaved edits, undo history, and an in-flight
// AI op all survive on the still-resident editor and the op's result lands back on return — so
// a plain nav is NON-DESTRUCTIVE and the in-app nav guard is permanently disarmed (`inApp:
// false`); prompting there would falsely warn users of loss that won't happen, undermining the
// keep-alive promise. Only a real browser close/refresh drops the in-memory state, so the
// beforeunload guard still arms whenever there are unsaved edits or an in-flight op — EVEN when
// the editor is backgrounded under keep-alive (an app close/refresh would otherwise silently
// discard a backgrounded editor's unsaved edits, a safeguard that predates keep-alive).
export function leaveGuardArming({ dirty, aiOpPending }) {
  const message = leaveGuardMessage({ dirty, aiOpPending });
  return {
    message,
    beforeUnload: Boolean(message),
    inApp: false,
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
