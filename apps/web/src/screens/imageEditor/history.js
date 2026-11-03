

// ── Undo/redo history (sc-6106) ────────────────────────────────────────────
// A bounded, backend-free history over opaque working-image snapshots. The
// reducer is pure — it only shuffles snapshots between the past (undo) and future
// (redo) stacks; the caller owns capturing a snapshot (the working bitmap blob +
// box/provenance overlay state) and restoring one (decode + install). Snapshots
// hold a Blob, never a live object URL, so an evicted snapshot is plain garbage —
// there is nothing to revoke, which keeps the "no leak of evicted snapshots"
// guarantee trivial. The stack depth is bounded so retained bitmaps stay capped.
export const HISTORY_LIMIT = 30;

export function emptyHistory() {
  return { past: [], future: [] };
}

// Push the current snapshot onto the undo stack and drop the redo stack. Call at
// the START of an operation, before the working state mutates, with a snapshot of
// the pre-operation state. Bounded to the `limit` most-recent entries.
export function historyCheckpoint(history, snapshot, limit = HISTORY_LIMIT) {
  return { past: [...history.past, snapshot].slice(-limit), future: [] };
}

// Step back one operation. `present` is the current on-screen snapshot, captured
// fresh by the caller so a later redo restores exactly what is on screen now.
// Returns the next history plus the snapshot to restore (`restore` is null when
// there is nothing to undo, in which case `history` is returned unchanged).
export function historyUndo(history, present, limit = HISTORY_LIMIT) {
  if (!history.past.length) return { history, restore: null };
  const restore = history.past[history.past.length - 1];
  return {
    history: { past: history.past.slice(0, -1), future: [present, ...history.future].slice(0, limit) },
    restore,
  };
}

// Step forward one operation, symmetric to historyUndo.
export function historyRedo(history, present, limit = HISTORY_LIMIT) {
  if (!history.future.length) return { history, restore: null };
  const [restore, ...rest] = history.future;
  return {
    history: { past: [...history.past, present].slice(-limit), future: rest },
    restore,
  };
}

export const canUndo = (history) => history.past.length > 0;

export const canRedo = (history) => history.future.length > 0;

// Serialize a single undo/redo step so a rapid second invocation can't race a
// restore that is still in flight (sc-8852). Restoring is async — each changed
// layer is decoded via `blobToImage` before the working stack is re-installed —
// so without a guard a key-repeat Cmd-Z would run `step()` again while the first
// restore is mid-flight, capturing the STALE working state as "present" and
// pushing a duplicate onto the redo (`future`) branch. Both undo AND redo route
// through here and share one `guardRef`: while a restore is in flight, any
// further undo/redo is ignored (the finding's simplest fix — holding Cmd-Z steps
// back predictably rather than dropping-then-jumping). The guard is set BEFORE
// the reducer step runs and cleared in a `finally`, so a mid-restore error can't
// wedge undo/redo forever. Returns true when a step actually ran.
//   - guardRef:     a mutable ref-like `{ current: boolean }` shared by undo/redo.
//   - step:         () => ({ history, restore }) — the pure reducer step, called
//                   only once we hold the guard, capturing the live "present".
//   - commitHistory:(nextHistory) => void — install the new history + sync flags.
//   - restore:      async (snapshot) => void — the async snapshot re-install.
export async function runGuardedRestore({ guardRef, step, commitHistory, restore }) {
  if (guardRef.current) return false;
  guardRef.current = true;
  try {
    const { history: next, restore: target } = step();
    if (!target) return false;
    commitHistory(next);
    await restore(target);
    return true;
  } finally {
    guardRef.current = false;
  }
}
