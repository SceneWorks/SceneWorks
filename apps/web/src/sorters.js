import { terminalStatuses } from "./jobTypes.js";

export function sortNewest(a, b) {
  return b.createdAt.localeCompare(a.createdAt);
}

export function sortOldest(a, b) {
  return a.createdAt.localeCompare(b.createdAt);
}

export function sortWorkers(a, b) {
  return `${a.gpuId}-${a.id}`.localeCompare(`${b.gpuId}-${b.id}`);
}

// sc-8860 (F-058): cap on retained *terminal* jobs (completed/failed/canceled/
// interrupted). Active/non-terminal jobs are NEVER dropped — only the count of
// finished jobs the client hangs onto is bounded so a long session can't grow the
// `jobs` array (and its per-SSE-tick copy) without limit. 200 keeps the Queue's
// recent-history view and every studio's "remembered" run (buildLocalJobStack only
// ever remembers the most-recent runs) fully intact while capping the tail; a busy
// day rarely leaves more than a few dozen finished jobs on screen at once, so 200 is
// generous headroom, not a functional ceiling.
export const MAX_TERMINAL_JOBS = 200;

// Insert `job` into a newest-first (`sortNewest`, i.e. createdAt-descending) list in
// O(n): drop any existing entry with the same id, then splice the job in at its
// createdAt-ordered position. Equivalent output to `[job, ...rest].sort(sortNewest)`
// but without re-sorting the whole array on every `job.updated` tick (F-058). The
// insertion point is derived straight from `sortNewest` so ties and edge cases (e.g.
// an in-flight entry whose createdAt sorts to the front) match the old behavior
// exactly: `job` lands before the first `item` for which sortNewest(job, item) <= 0.
export function insertNewest(items, job) {
  const next = [];
  let inserted = false;
  for (const item of items) {
    if (item.id === job.id) {
      continue; // replaced by `job`
    }
    if (!inserted && sortNewest(job, item) <= 0) {
      next.push(job);
      inserted = true;
    }
    next.push(item);
  }
  if (!inserted) {
    next.push(job);
  }
  return next;
}

// Prune the retained-terminal-jobs tail. Keeps EVERY active (non-terminal) job plus
// the newest `MAX_TERMINAL_JOBS` terminal ones. Expects a newest-first list (as
// produced by insertNewest / sortNewest) so the terminal jobs kept are the most
// recent by createdAt. Returns the same array reference when nothing is pruned so an
// unchanged tick doesn't allocate.
export function capTerminalJobs(items, max = MAX_TERMINAL_JOBS) {
  let terminalCount = 0;
  for (const item of items) {
    if (terminalStatuses.has(item.status)) {
      terminalCount += 1;
    }
  }
  if (terminalCount <= max) {
    return items;
  }
  const kept = [];
  let keptTerminal = 0;
  for (const item of items) {
    if (terminalStatuses.has(item.status)) {
      if (keptTerminal >= max) {
        continue;
      }
      keptTerminal += 1;
    }
    kept.push(item);
  }
  return kept;
}

// The canonical "a job just changed" state update (F-058): ordered insertion by
// createdAt (newest-first) followed by the terminal-jobs cap. Replaces the
// `[job, ...items.filter((i) => i.id !== job.id)].sort(sortNewest)` copy-sort that ran
// on every SSE `job.updated` and each enqueue across App.jsx and the data hooks.
export function upsertJobNewest(items, job) {
  return capTerminalJobs(insertNewest(items, job));
}
